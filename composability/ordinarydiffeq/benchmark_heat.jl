# The same OrdinaryDiffEq solver and heat-equation RHS run on each array backend.
# run_samples.jl launches this worker with ODE_SAMPLES=1 for sample isolation.
import OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54
using SciMLBase: ODEProblem, solve, successful_retcode, FullSpecialize
using Statistics: mean, median, std
using LinearAlgebra: transpose

length(ARGS) >= 1 || error("Usage: julia benchmark_heat.jl {cpu|CuArray|cuNumeric|Dagger} [N ...]")
backend = ARGS[1]
const T = get(ENV, "ODE_ELTYPE", "Float32") == "Float64" ? Float64 : Float32
const KAPPA = T(0.2)
const T_END = T(1)
const NSTEPS = parse(Int, get(ENV, "ODE_STEPS", "20"))
const SAMPLES = parse(Int, get(ENV, "ODE_SAMPLES", "5"))
const GPUS = parse(Int, get(ENV, "ODE_GPUS", "1"))
NSTEPS > 0 || error("ODE_STEPS must be positive")
SAMPLES >= 1 || error("ODE_SAMPLES must be positive")
GPUS > 0 || error("ODE_GPUS must be positive")
(backend != "CuArray" || GPUS == 1) || error("CuArray baseline uses one GPU")

include(joinpath(@__DIR__, "heat_rhs.jl"))

if backend == "cpu"
    make_state(a) = a
    synchronized_time_ns(_=nothing) = time_ns()
    host_state(a) = a
elseif backend == "CuArray"
    using CUDA
    CUDA.allowscalar(false)
    make_state(a) = CUDA.CuArray(a)
    synchronized_time_ns(_=nothing) = (CUDA.synchronize(); time_ns())
    host_state(a) = Array(a)
elseif backend == "cuNumeric"
    using cuNumeric
    cuNumeric.allowscalar(false)
    make_state(a) = NDArray(a)
    synchronized_time_ns(_=nothing) = cuNumeric.get_time_nanoseconds()
    host_state(a) = Array(a)
elseif backend == "Dagger"
    using Dagger, CUDA
    CUDA.allowscalar(false)
    Dagger.allowscalar!(false)
    heat_slice(u::Dagger.DArray, rows, cols) = u[rows, cols]
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
        state = Dagger.distribute(a, Dagger.Blocks(block, size(a, 2)), grid)
        wait(state)
        correct_initial_storage(state) || error("Dagger initial state is not distributed across requested CUDA devices")
        return state
    end
    function synchronized_time_ns(a=nothing)
        a === nothing || foreach(wait, a.chunks)
        Dagger.gpu_synchronize(:CUDA)
        return time_ns()
    end
    host_state(a) = collect(a)
    function correct_initial_storage(a)
        a isa Dagger.DArray || return false
        chunks = [fetch(chunk; raw=true) for chunk in a.chunks]
        return all(chunk -> chunk isa Dagger.Chunk{<:CUDA.CuArray}, chunks) &&
               Set(chunk.processor.device + 1 for chunk in chunks) == Set(1:GPUS)
    end
else
    error("Unknown backend $backend")
end

# A discrete sine mode is an exact eigenvector of this spatial operator.
# It supplies an independent answer without running a host ODE solve.
function initial_state(n)
    k = max(1, round(Int, (n - 1) / 4))
    s = T[sin(pi * k * (i - 1) / (n - 1)) for i in 1:n]
    s[1] = s[end] = zero(T)
    return s * transpose(s), k
end

const ALG = CarpenterKennedy2N54(; williamson_condition=false)

function cleanup_before_solve()
    synchronized_time_ns()
    GC.gc(true)
    if backend == "cuNumeric" && isdefined(cuNumeric, :drain_pending_frees!)
        # NDArray finalizers can defer handle destruction to the launch thread.
        cuNumeric.drain_pending_frees!()
    end
    synchronized_time_ns()
    return nothing
end

function run_case(n)
    n >= 4 || error("N must be at least 4")
    host_u0, k = initial_state(n)
    u0 = make_state(host_u0)
    prob = ODEProblem{true,FullSpecialize}(heat!, u0, (zero(T), T_END), KAPPA)
    dt = T_END / NSTEPS
    # The benchmark checks the final state below; skip SciML's per-step
    # instability scan, which otherwise scalar-iterates custom array types.
    do_solve() = solve(prob, ALG; dt, adaptive=false,
                       save_everystep=false, save_start=false, dense=false,
                       unstable_check=(dt, u, p, t) -> false)

    sol = nothing
    println("warmup=1 backend=$backend N=$n")
    flush(stdout)
    cleanup_before_solve()
    sol = do_solve()
    synchronized_time_ns(sol.u[end])
    @assert successful_retcode(sol)

    elapsed_ms = Float64[]
    for sample_index in 1:SAMPLES
        println("sample=$sample_index backend=$backend N=$n")
        flush(stdout)
        # Drop the previous solution before GC; collecting with it still live
        # would retain its state while the next solve allocates a new workspace.
        sol = nothing
        cleanup_before_solve()
        started = synchronized_time_ns()
        sol = do_solve()
        push!(elapsed_ms, (synchronized_time_ns(sol.u[end]) - started) / 1e6)
        @assert successful_retcode(sol)
    end

    theta = pi * k / (n - 1)
    eigenvalue = 4 * Float64(KAPPA) / Float64(DX)^2 * (cos(theta) - 1)
    reference = exp(eigenvalue * Float64(T_END)) .* host_u0
    # Validate the numerical answer independently of its final GPU placement.
    actual = host_state(sol.u[end])
    error_rel = maximum(abs.(actual .- reference)) / maximum(abs.(reference))
    tolerance = T == Float32 ? 1e-4 : 1e-7
    @assert error_rel < tolerance "relative error $error_rel exceeds $tolerance"
    # A single worker sample has no standard-error estimate. The coordinator
    # computes it from all independent samples before publishing the CSV row.
    stderr = length(elapsed_ms) > 1 ? std(elapsed_ms) / sqrt(length(elapsed_ms)) : NaN
    println("RESULT,$backend,$T,$GPUS,$n,$NSTEPS,$(mean(elapsed_ms)),$stderr,$(median(elapsed_ms)),$(minimum(elapsed_ms)),$(maximum(elapsed_ms)),$error_rel,$(join(elapsed_ms, ';'))")
    flush(stdout)
end

println("backend=$backend eltype=$T gpus=$GPUS dx=$DX steps=$NSTEPS Julia=$VERSION threads=$(Threads.nthreads()) pid=$(getpid())")
println("workspace=", get(ENV, "CUBLAS_WORKSPACE_CONFIG", "<default>"),
        " OrdinaryDiffEqLowStorageRK=", pkgversion(OrdinaryDiffEqLowStorageRK))
flush(stdout)
sizes = length(ARGS) > 1 ? parse.(Int, ARGS[2:end]) : [128, 1024, 4096]
if backend == "Dagger"
    Dagger.with_options(; scope=dagger_scope) do
        foreach(run_case, sizes)
    end
else
    foreach(run_case, sizes)
end
