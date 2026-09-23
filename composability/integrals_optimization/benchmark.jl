# Optimize an array-valued gas concentration map through Integrals.jl.
# Run one backend per Julia process.
using Optimization
using OptimizationOptimisers
using Random: MersenneTwister, randn
using Statistics: median
using LinearAlgebra: norm

include(joinpath(@__DIR__, "model.jl"))

length(ARGS) == 2 || error("Usage: julia benchmark.jl {cpu|CuArray|cuNumeric} N")
backend = ARGS[1]
n = parse(Int, ARGS[2])
n >= 4 || error("N must be at least 4")

const PRECISION = get(ENV, "INTOPT_ELTYPE", "Float32")
PRECISION in ("Float32", "Float64") || error("INTOPT_ELTYPE must be Float32 or Float64")
const T = PRECISION == "Float64" ? Float64 : Float32
const BANDS = parse(Int, get(ENV, "INTOPT_BANDS", "4"))
const ORDER = parse(Int, get(ENV, "INTOPT_ORDER", "12"))
const ITERS = parse(Int, get(ENV, "INTOPT_ITERS", "40"))
const SAMPLES = parse(Int, get(ENV, "INTOPT_SAMPLES", "3"))
const RATE = parse(T, get(ENV, "INTOPT_RATE", "0.05"))
const NOISE = parse(T, get(ENV, "INTOPT_NOISE", "0.001"))
ITERS > 0 && SAMPLES > 0 && RATE > 0 && NOISE >= 0 ||
    error("Invalid iteration, sample, learning-rate, or noise setting")

if backend == "cpu"
    make_state(a) = a
    synchronized_time_ns() = time_ns()
    host_state(a) = a
    correct_storage(a) = a isa Matrix{T}
    run_with_scalar_fetch(f) = f()
elseif backend == "CuArray"
    using CUDA
    CUDA.allowscalar(false)
    make_state(a) = CUDA.CuArray(a)
    synchronized_time_ns() = (CUDA.synchronize(); time_ns())
    host_state(a) = Array(a)
    correct_storage(a) = a isa CUDA.CuArray{T,2}
    run_with_scalar_fetch(f) = f()
elseif backend == "cuNumeric"
    using cuNumeric
    cuNumeric.allowscalar(false)
    make_state(a) = NDArray(a)
    synchronized_time_ns() = cuNumeric.get_time_nanoseconds()
    host_state(a) = Array(a)
    correct_storage(a) = a isa NDArray{T,2}
    run_with_scalar_fetch(f) = cuNumeric.allowautofetch(f)

    # Prototype compatibility shim: OptimizationOptimisers checks this each
    # iteration. Move this method into cuNumeric once validated on a GPU.
    Base.all(::typeof(isfinite), a::NDArray) = Bool(all(isfinite.(a)))
else
    error("Unknown backend $backend")
end

# A smooth synthetic plume, with concentration between approximately 0.2 and 0.9.
function ground_truth(n)
    return [T(0.2 + 0.7 * exp(-(((i - 1) / (n - 1) - 0.4)^2 +
                                 ((j - 1) / (n - 1) - 0.6)^2) / 0.04))
            for i in 1:n, j in 1:n]
end

function run_case(n)
    model = AbsorptionModel(T, BANDS, ORDER)
    observation_model = AbsorptionModel(T, BANDS, 2 * ORDER)
    truth_host = ground_truth(n)
    truth_device = make_state(truth_host)
    rng = MersenneTwister(20260923)
    observations = [
        band_integral(truth_device, center, observation_model) .+
        make_state(NOISE .* randn(rng, T, n, n))
        for center in model.centers
    ]
    all(correct_storage, observations) || error("Integral result left the $backend backend")
    initial = make_state(fill(T(0.45), n, n))

    # The one host scalar in loss is required by Optimization.jl. The state,
    # gradients and predictions remain backend arrays.
    objective(u, _=nothing) = loss(u, observations, model)
    function derivative!(G, u, _=nothing)
        correct_storage(G) || error("Optimizer gradient left the $backend backend")
        return gradient!(G, u, observations, model)
    end
    optf = OptimizationFunction(objective; grad=derivative!)
    problem = OptimizationProblem(optf, initial)
    do_solve() = Optimization.solve(problem, OptimizationOptimisers.Adam(RATE);
                                     maxiters=ITERS, save_best=false, progress=false)

    initial_loss = objective(initial)
    warmup = do_solve()
    correct_storage(warmup.u) || error("Optimizer moved the array state off $backend")
    synchronized_time_ns()

    elapsed_ms = Float64[]
    local solution
    for _ in 1:SAMPLES
        started = synchronized_time_ns()
        solution = do_solve()
        push!(elapsed_ms, (synchronized_time_ns() - started) / 1e6)
        correct_storage(solution.u) || error("Optimizer moved the array state off $backend")
    end

    final_loss = objective(solution.u)
    final_loss < initial_loss || error("Optimization failed to reduce the loss")
    relative_error = norm(host_state(solution.u) .- truth_host) / norm(truth_host)
    println("RESULT,$backend,$T,$n,$BANDS,$ORDER,$ITERS,$(median(elapsed_ms)),$(minimum(elapsed_ms)),$(maximum(elapsed_ms)),$initial_loss,$final_loss,$relative_error,$(join(elapsed_ms, ';'))")
end

println("backend=$backend eltype=$T N=$n bands=$BANDS order=$ORDER iters=$ITERS Julia=$VERSION")
println("Integrals=$(pkgversion(Integrals)) Optimization=$(pkgversion(Optimization)) OptimizationOptimisers=$(pkgversion(OptimizationOptimisers))")
run_with_scalar_fetch(() -> run_case(n))
