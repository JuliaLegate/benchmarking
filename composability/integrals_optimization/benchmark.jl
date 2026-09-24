# Fit plume amplitude and width with a derivative-free Optimization.jl solver.
# The parameter vector is small and host-resident; Integrals.jl evaluates the
# image-wide forward model on the selected array backend.
using Optimization
using OptimizationOptimJL
using Random: MersenneTwister, randn
using Statistics: mean, median, std
using LinearAlgebra: norm

include(joinpath(@__DIR__, "model.jl"))

length(ARGS) == 2 || error("Usage: julia benchmark.jl {cpu|CuArray|Dagger|cuNumeric} N")
backend = ARGS[1]
n = parse(Int, ARGS[2])
n >= 4 || error("N must be at least 4")

const PRECISION = get(ENV, "INTOPT_ELTYPE", "Float32")
PRECISION in ("Float32", "Float64") || error("INTOPT_ELTYPE must be Float32 or Float64")
const T = PRECISION == "Float64" ? Float64 : Float32
const BANDS = parse(Int, get(ENV, "INTOPT_BANDS", "4"))
const ORDER = parse(Int, get(ENV, "INTOPT_ORDER", "12"))
const ITERS = parse(Int, get(ENV, "INTOPT_ITERS", "80"))
const SAMPLES = parse(Int, get(ENV, "INTOPT_SAMPLES", "5"))
const GPUS = parse(Int, get(ENV, "INTOPT_GPUS", "1"))
const NOISE = parse(T, get(ENV, "INTOPT_NOISE", "0.001"))
ITERS > 0 && SAMPLES >= 2 && GPUS > 0 && NOISE >= 0 ||
    error("Invalid iteration, sample, or noise setting")
(backend != "CuArray" || GPUS == 1) || error("CuArray baseline uses one GPU")

if backend == "cpu"
    make_state(a) = a
    synchronized_time_ns() = time_ns()
    correct_storage(a) = a isa Matrix{T}
    run_with_scalar_fetch(f) = f()
elseif backend == "CuArray"
    using CUDA
    CUDA.allowscalar(false)
    make_state(a) = CUDA.CuArray(a)
    synchronized_time_ns() = (CUDA.synchronize(); time_ns())
    correct_storage(a) = a isa CUDA.CuArray{T,2}
    run_with_scalar_fetch(f) = f()
elseif backend == "cuNumeric"
    using cuNumeric
    cuNumeric.allowscalar(false)
    make_state(a) = NDArray(a)
    synchronized_time_ns() = cuNumeric.get_time_nanoseconds()
    correct_storage(a) = a isa NDArray{T,2}
    run_with_scalar_fetch(f) = cuNumeric.allowautofetch(f)
elseif backend == "Dagger"
    using Dagger, CUDA
    CUDA.allowscalar(false)
    length(collect(CUDA.devices())) >= GPUS || error("Requested $GPUS GPUs are unavailable")
    dagger_scope = Dagger.scope(; cuda_gpus=collect(1:GPUS))
    function make_state(a)
        procs = sort(collect(filter(p -> p isa Dagger.CuArrayDeviceProc,
                                    Dagger.compatible_processors())); by=p -> p.device)
        length(procs) == GPUS || error("Dagger sees $(length(procs)) of $GPUS requested GPUs")
        block = cld(size(a, 1), GPUS)
        grid = Array{Dagger.Processor}(undef, cld(size(a, 1), block), 1)
        for i in axes(grid, 1)
            grid[i, 1] = procs[i]
        end
        result = Dagger.distribute(a, Dagger.Blocks(block, size(a, 2)), grid)
        wait(result)
        correct_storage(result) || error("Dagger array left the requested CUDA devices")
        return result
    end
    synchronized_time_ns() = (Dagger.gpu_synchronize(:CUDA); time_ns())
    function correct_storage(a)
        a isa Dagger.DArray || return false
        chunks = [fetch(chunk; raw=true) for chunk in a.chunks]
        return all(chunk -> chunk isa Dagger.Chunk{<:CUDA.CuArray}, chunks) &&
               Set(chunk.processor.device + 1 for chunk in chunks) == Set(1:GPUS)
    end
    function run_with_scalar_fetch(f)
        Dagger.with_options(; scope=dagger_scope) do
            f()
        end
    end
else
    error("Unknown backend $backend")
end

const TRUTH = [0.7, 0.2] # Peak plume amplitude and spatial width.
const INITIAL = [0.5, 0.3]

function run_case(n)
    model = AbsorptionModel(T, BANDS, ORDER)
    observation_model = AbsorptionModel(T, BANDS, 2 * ORDER)
    radius2_host = [T(((i - 1) / (n - 1) - 0.4)^2 +
                      ((j - 1) / (n - 1) - 0.6)^2) for i in 1:n, j in 1:n]
    radius2 = make_state(radius2_host)
    correct_storage(radius2) || error("Radius grid left the $backend backend")
    truth_image = plume(log.(TRUTH), radius2)
    correct_storage(truth_image) || error("Plume image left the $backend backend")
    rng = MersenneTwister(20260923)
    observations = [
        band_integral(truth_image, center, observation_model) .+
        make_state(NOISE .* randn(rng, T, n, n))
        for center in model.centers
    ]
    all(correct_storage, observations) || error("Integral result left the $backend backend")

    # Nelder-Mead manages two host log parameters. Each loss evaluation runs
    # array arithmetic and Integrals.jl quadrature on the selected backend.
    objective(u, _=nothing) = loss(plume(u, radius2), observations, model)
    optf = OptimizationFunction(objective)
    initial = log.(INITIAL)
    do_solve() = Optimization.solve(OptimizationProblem(optf, copy(initial)),
                                    NelderMead(); maxiters=ITERS, progress=false)

    initial_loss = objective(initial)
    for _ in 1:2
        warmup = do_solve()
        warmup.u isa Vector{Float64} || error("Unexpected optimizer parameter storage")
    end
    synchronized_time_ns()

    elapsed_ms = Float64[]
    local solution
    for _ in 1:SAMPLES
        started = synchronized_time_ns()
        solution = do_solve()
        push!(elapsed_ms, (synchronized_time_ns() - started) / 1e6)
    end

    final_loss = objective(solution.u)
    final_loss < initial_loss || error("Optimization failed to reduce the loss")
    estimated = exp.(solution.u)
    parameter_error = norm(estimated .- TRUTH) / norm(TRUTH)
    parameter_error < 0.1 || error("Plume parameters were not recovered")
    println("parameters=$estimated objective_evals=$(solution.stats.fevals)")
    stderr = std(elapsed_ms) / sqrt(length(elapsed_ms))
    println("RESULT,$backend,$T,$GPUS,$n,$BANDS,$ORDER,$ITERS,$(solution.stats.fevals),$(mean(elapsed_ms)),$stderr,$(median(elapsed_ms)),$(minimum(elapsed_ms)),$(maximum(elapsed_ms)),$initial_loss,$final_loss,$parameter_error,$(join(elapsed_ms, ';'))")
end

println("backend=$backend eltype=$T gpus=$GPUS N=$n bands=$BANDS order=$ORDER maxiters=$ITERS Julia=$VERSION")
println("Integrals=$(pkgversion(Integrals)) Optimization=$(pkgversion(Optimization)) OptimizationOptimJL=$(pkgversion(OptimizationOptimJL))")
run_with_scalar_fetch(() -> run_case(n))
