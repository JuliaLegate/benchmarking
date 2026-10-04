using LinearAlgebra, Statistics, Krylov

length(ARGS) == 4 || error("Usage: julia krylov.jl {CuArray|Dagger|cuNumeric} {cg|bicgstab} {stock|local} N")
const BACKEND, SOLVER, MODE = ARGS[1:3]
const N = parse(Int, ARGS[4])
const GPUS = parse(Int, get(ENV, "BENCH_GPUS", "1"))
BACKEND in ("cuNumeric", "CuArray", "Dagger") || error("Unknown backend: $BACKEND")
SOLVER in ("cg", "bicgstab") || error("Unknown solver: $SOLVER")
MODE in ("stock", "local") || error("Unknown mode: $MODE")
(BACKEND == "cuNumeric" || MODE == "stock") || error("Local loops require cuNumeric")
(BACKEND != "CuArray" || GPUS == 1) || error("CuArray baseline uses one GPU")
N > 1 || error("N must exceed 1")
GPUS > 0 || error("BENCH_GPUS must be positive")
const T = get(ENV, "BENCH_ELTYPE", "Float32") == "Float64" ? Float64 : Float32
const TOL = T === Float32 ? T(1e-5) : T(1e-8)
const TUNING = get(ENV, "COMPOSABILITY_TUNE", "0") == "1"
const MAXITERS = TUNING ? 5 : 200
const SAMPLES = TUNING ? 2 : parse(Int, get(ENV, "BENCH_SAMPLES", "5"))
SAMPLES >= 1 || error("BENCH_SAMPLES must be positive")
BLAS.set_num_threads(1)

# The timed operator is dense for every backend. cuNumeric attaches row-major
# host storage, so construct that layout directly from the three diagonals.
# NDArray(Matrix(reference)) would transpose and then copy the entire matrix.
function dense_operator_buffer(reference::Tridiagonal; row_major=false)
    stored = row_major ? Tridiagonal(reference.du, reference.d, reference.dl) : reference
    return Matrix(stored)
end

# Full-height column strips, with vectors split at the same columns.
function dagger_krylov_layout(a, procs)
    block = cld(N, GPUS * BLOCKS_PER_GPU)
    nb = cld(N, block)
    owner(j) = procs[cld(j * length(procs), nb)]
    a isa AbstractVector && return (block,), [owner(j) for j in 1:nb]
    return (N, block), [owner(j) for _ in 1:1, j in 1:nb]
end

make_operator(a) = make_array(a)
if BACKEND == "cuNumeric"
    @eval using cuNumeric
    cuNumeric.allowscalar(false)
    @eval make_array(a) = NDArray(a)
    # Use the same attachment as NDArray's matrix constructor, with an already
    # packed buffer. The returned NDArray retains the buffer as its parent.
    @eval make_operator(a::Matrix) = cuNumeric.nda_attach_external(a; shape=reverse(size(a)))
    @eval sync(w) = cuNumeric.issue_execution_fence(; block=true)
    @eval host_array(x) = Array(x)
    @eval permitted_solve!(w, A, b) = @allowpromotion @allowautofetch solve!(w, A, b)
    MODE == "local" && include("local.jl")
elseif BACKEND == "CuArray"
    @eval using CUDA
    CUDA.allowscalar(false)
    @eval make_array(a) = CuArray(a)
    @eval sync(w) = CUDA.synchronize()
    @eval host_array(x) = Array(x)
    @eval permitted_solve!(w, A, b) = solve!(w, A, b)
else
    @eval using Dagger, CUDA
    const BLOCKS_PER_GPU = parse(Int, get(ENV, "DAGGER_BLOCKS_PER_GPU", "1"))
    BLOCKS_PER_GPU > 0 || error("DAGGER_BLOCKS_PER_GPU must be positive")
    CUDA.allowscalar(false)
    # Current Dagger tile GEMV calls CPU BLAS.gemv! on CuArrays. This
    # benchmark-only tile method keeps Krylov's solver unmodified.
    @eval function Dagger.matvecmul!(y::CuArray, trans::Char, A::CuArray, x::CuArray, α, β)
        opA = trans == 'N' ? A : trans == 'T' ? transpose(A) : adjoint(A)
        return mul!(y, opA, x, α, β)
    end
    @eval function make_array(a)
        procs = sort(collect(filter(p -> p isa Dagger.CuArrayDeviceProc,
                                    Dagger.compatible_processors())); by=p -> p.device)
        length(procs) == GPUS || error("Dagger sees $(length(procs)) of $GPUS requested GPUs")
        blocks, grid = dagger_krylov_layout(a, procs)
        result = Dagger.distribute(a, Dagger.Blocks(blocks...), grid)
        wait(result)
        chunks = [fetch(chunk; raw=true) for chunk in result.chunks]
        all(chunk -> chunk isa Dagger.Chunk{<:CuArray}, chunks) ||
            error("Dagger array contains non-CUDA chunks")
        Set(chunk.processor.device + 1 for chunk in chunks) == Set(1:GPUS) ||
            error("Dagger tiles did not land on all requested GPUs")
        return result
    end
    @eval function sync(w)
        for field in fieldnames(typeof(w))
            value = getfield(w, field)
            value isa Dagger.DArray && wait(value)
        end
        Dagger.gpu_synchronize(:CUDA)
    end
    @eval host_array(x) = collect(x)
    @eval permitted_solve!(w, A, b) = solve!(w, A, b)
end

function solve!(w, A, b)
    if MODE == "stock"
        if SOLVER == "cg"
            Krylov.cg!(w, A, b; atol=zero(T), rtol=TOL, itmax=MAXITERS)
        else
            Krylov.bicgstab!(w, A, b; atol=zero(T), rtol=TOL, itmax=MAXITERS)
        end
        return w.x, w.stats.niter, w.stats.solved
    end
    return SOLVER == "cg" ? local_cg!(w, A, b) : local_bicgstab!(w, A, b)
end

function checked_solve!(w, A, b, reference, bh; solution=nothing)
    x, iterations, _ = solution === nothing ? permitted_solve!(w, A, b) : solution
    sync(w)
    # Validate the answer independently of the solver's convergence flag and
    # the final vector's placement; input placement is checked in make_array.
    xh = Float64.(host_array(x))
    xh isa AbstractVector && length(xh) == length(bh) ||
        error("Expected a solution vector of length $(length(bh)); got $(typeof(xh)) with size $(size(xh))")
    residual = norm(reference * xh - Float64.(bh)) / norm(Float64.(bh))
    # A short tuning solve need not converge, but must reduce the residual.
    valid = TUNING ? isfinite(residual) && residual < 1 : residual <= TOL
    valid || error("Invalid relative residual $residual (tuning=$TUNING)")
    return iterations, residual
end

function benchmark()
    println("Worker PID=$(getpid()): $BACKEND $SOLVER $MODE, G=$GPUS N=$N tuning=$TUNING maxiters=$MAXITERS")
    flush(stdout)
    lower = fill(T(SOLVER == "cg" ? -0.5 : -0.3), N - 1)
    upper = fill(T(SOLVER == "cg" ? -0.5 : -0.8), N - 1)
    diagonal = T.(range(2.0, 4.0; length=N))
    reference = Tridiagonal(lower, diagonal, upper)
    Ah = dense_operator_buffer(reference; row_major=BACKEND == "cuNumeric")
    bh = T[sin(i) + 1 for i in 1:N]
    A, b = make_operator(Ah), make_array(bh)
    w = MODE == "stock" ?
        (SOLVER == "cg" ? Krylov.CgWorkspace(A, b) : Krylov.BicgstabWorkspace(A, b)) :
        local_workspace(b)
    Ah = nothing
    GC.gc()

    checked_solve!(w, A, b, reference, bh)
    println("Warmup complete; starting $SAMPLES timed solve(s)")
    flush(stdout)
    samples = Float64[]
    solution = nothing
    for _ in 1:SAMPLES
        GC.gc(); sync(w)
        start = time_ns()
        solution = permitted_solve!(w, A, b)
        sync(w)
        push!(samples, (time_ns() - start) / 1e6)
    end
    # Validate the actual timed solution without running an additional solve.
    iterations, residual = checked_solve!(w, A, b, reference, bh; solution)
    label = BACKEND == "CuArray" ? "CUDA" :
            BACKEND == "cuNumeric" && MODE != "stock" ? "cuNumeric $(MODE)" : BACKEND
    stderr = length(samples) > 1 ? std(samples) / sqrt(length(samples)) : NaN
    println("RESULT,$label,$SOLVER,$MODE,$T,$GPUS,$N,$iterations,$(mean(samples)),$stderr,$(median(samples)),$(minimum(samples)),$(maximum(samples)),$residual,$(join(samples, ';'))")
    flush(stdout)
end

if BACKEND == "Dagger"
    available_gpus = length(collect(CUDA.devices()))
    available_gpus >= GPUS || error("Requested $GPUS GPUs, found $available_gpus")
    Dagger.with_options(; scope=Dagger.scope(cuda_gpus=collect(1:GPUS))) do
        benchmark()
    end
else
    benchmark()
end
