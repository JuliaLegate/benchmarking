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
BLAS.set_num_threads(1)
if BACKEND == "cuNumeric"
    @eval using cuNumeric
    cuNumeric.allowscalar(false)
    @eval make_array(a) = NDArray(a)
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
    CUDA.allowscalar(false)
    # Current Dagger tile GEMV calls CPU BLAS.gemv! on CuArrays. This
    # benchmark-only tile method keeps Krylov's solver unmodified.
    @eval function Dagger.matvecmul!(y::CuArray, trans::Char, A::CuArray, x::CuArray, α, β)
        opA = trans == 'N' ? A : trans == 'T' ? transpose(A) : adjoint(A)
        return mul!(y, opA, x, α, β)
    end
    @eval function make_array(a)
        block = cld(N, GPUS) # Square tiles align solver input/output vectors.
        procs = sort(collect(filter(p -> p isa Dagger.CuArrayDeviceProc,
                                    Dagger.compatible_processors())); by=p -> p.device)
        length(procs) == GPUS || error("Dagger sees $(length(procs)) of $GPUS requested GPUs")
        grid = Array{Dagger.Processor}(undef,
                                      ntuple(i -> cld(size(a, i), block), ndims(a)))
        for I in CartesianIndices(grid)
            grid[I] = procs[mod1(I[1], GPUS)]
        end
        result = Dagger.distribute(a, Dagger.Blocks(ntuple(_ -> block, ndims(a))...), grid)
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
            Krylov.cg!(w, A, b; atol=zero(T), rtol=TOL, itmax=200)
        else
            Krylov.bicgstab!(w, A, b; atol=zero(T), rtol=TOL, itmax=200)
        end
        return w.x, w.stats.niter, w.stats.solved
    end
    return SOLVER == "cg" ? local_cg!(w, A, b) : local_bicgstab!(w, A, b)
end

function checked_solve!(w, A, b, reference, bh)
    x, iterations, solved = permitted_solve!(w, A, b)
    sync(w)
    solved || error("$SOLVER did not converge")
    residual = norm(reference * Float64.(host_array(x)) - Float64.(bh)) / norm(Float64.(bh))
    residual <= TOL || error("Relative residual $residual exceeds $TOL")
    return iterations, residual
end

function benchmark()
    lower = fill(T(SOLVER == "cg" ? -0.5 : -0.3), N - 1)
    upper = fill(T(SOLVER == "cg" ? -0.5 : -0.8), N - 1)
    diagonal = T.(range(2.0, 4.0; length=N))
    reference = Tridiagonal(lower, diagonal, upper)
    Ah = Matrix(reference) # All backends time the same dense operator.
    bh = T[sin(i) + 1 for i in 1:N]
    A, b = make_array(Ah), make_array(bh)
    w = MODE == "stock" ?
        (SOLVER == "cg" ? Krylov.CgWorkspace(A, b) : Krylov.BicgstabWorkspace(A, b)) :
        local_workspace(b)
    Ah = nothing
    GC.gc()

    for _ in 1:2
        checked_solve!(w, A, b, reference, bh)
    end
    samples = Float64[]
    for _ in 1:5
        GC.gc(); sync(w)
        start = time_ns()
        permitted_solve!(w, A, b)
        sync(w)
        push!(samples, (time_ns() - start) / 1e6)
    end
    iterations, residual = checked_solve!(w, A, b, reference, bh)
    label = BACKEND == "CuArray" ? "CUDA" :
            BACKEND == "cuNumeric" && MODE != "stock" ? "cuNumeric $(MODE)" : BACKEND
    println("RESULT,$label,$SOLVER,$MODE,$T,$GPUS,$N,$iterations,$(median(samples)),$(minimum(samples)),$(maximum(samples)),$residual,$(join(samples, ';'))")
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
