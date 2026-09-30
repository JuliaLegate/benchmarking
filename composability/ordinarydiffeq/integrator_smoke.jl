# Check that the same NDArray heat problem runs with several OrdinaryDiffEq solvers.
using cuNumeric
using SciMLBase: ODEProblem, solve, successful_retcode, FullSpecialize
using OrdinaryDiffEqLowOrderRK: RK4
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54
using OrdinaryDiffEqTsit5: Tsit5
using OrdinaryDiffEqVerner: Vern7
using LinearAlgebra: transpose
using Statistics: mean, std

include(joinpath(@__DIR__, "heat_rhs.jl"))

cuNumeric.allowscalar(false)
const N = isempty(ARGS) ? 128 : parse(Int, only(ARGS))
const ADAPTIVE = get(ENV, "ODE_ADAPTIVE", "0") == "1"
const TIMED_SAMPLES = parse(Int, get(ENV, "ODE_TIMED_SAMPLES", "0"))
const KAPPA = 0.2f0
N >= 4 || error("N must be at least 4")
(TIMED_SAMPLES == 0 || TIMED_SAMPLES >= 2) || error("ODE_TIMED_SAMPLES must be at least 2")

k = max(1, round(Int, (N - 1) / 4))
s = Float32[sin(pi * k * (i - 1) / (N - 1)) for i in 1:N]
s[1] = s[end] = 0f0
host_u0 = s * transpose(s)
theta = pi * k / (N - 1)
eigenvalue = 4 * Float64(KAPPA) / DX^2 * (cos(theta) - 1)
reference = exp(eigenvalue) .* host_u0

function checked_error(solution, name)
    successful_retcode(solution) || error("$name returned $(solution.retcode)")
    state = solution.u[end]
    state isa NDArray{Float32,2} || error("$name returned $(typeof(state))")
    relative_error = maximum(abs.(Array(state) .- reference)) / maximum(abs.(reference))
    relative_error < 1e-4 || error("$name relative error $relative_error exceeds 1e-4")
    return relative_error
end

const ALGORITHMS = ADAPTIVE ?
    (("RK4", RK4()), ("Tsit5", Tsit5()), ("Vern7", Vern7())) :
    (("CarpenterKennedy2N54", CarpenterKennedy2N54(williamson_condition=false)),
     ("RK4", RK4()), ("Tsit5", Tsit5()), ("Vern7", Vern7()))

for (name, algorithm) in ALGORITHMS
    problem = ODEProblem{true,FullSpecialize}(
        heat!, NDArray(copy(host_u0)), (0f0, 1f0), KAPPA
    )
    run_solve() = solve(
        problem, algorithm;
        dt=0.05f0, adaptive=ADAPTIVE,
        save_everystep=false, save_start=false, dense=false,
        unstable_check=(dt, u, p, t) -> false,
    )
    # Adaptive controllers need host decisions; keep fetching scoped to solve.
    permitted_solve() = if ADAPTIVE
        cuNumeric.allowautofetch() do
            run_solve()
        end
    else
        run_solve()
    end
    solution = permitted_solve()
    checked_error(solution, name)
    samples_ms = Float64[]
    if TIMED_SAMPLES > 0
        for _ in 1:2
            solution = permitted_solve()
            cuNumeric.get_time_nanoseconds()
            successful_retcode(solution) || error("$name warmup failed")
            solution.u[end] isa NDArray{Float32,2} || error("$name warmup left NDArray storage")
        end
        for _ in 1:TIMED_SAMPLES
            started = cuNumeric.get_time_nanoseconds()
            solution = permitted_solve()
            push!(samples_ms, (cuNumeric.get_time_nanoseconds() - started) / 1e6)
            successful_retcode(solution) || error("$name timed solve failed")
            solution.u[end] isa NDArray{Float32,2} || error("$name timed solve left NDArray storage")
        end
    end
    relative_error = checked_error(solution, name)
    println("PASS,$name,N=$N,adaptive=$ADAPTIVE,steps=$(solution.stats.naccept),relative_error=$relative_error")
    if TIMED_SAMPLES > 0
        println("TIMING,$name,N=$N,adaptive=$ADAPTIVE,mean_ms=$(mean(samples_ms))," *
                "stderr_ms=$(std(samples_ms) / sqrt(TIMED_SAMPLES))," *
                "samples_ms=$(join(samples_ms, ';'))")
    end
end
