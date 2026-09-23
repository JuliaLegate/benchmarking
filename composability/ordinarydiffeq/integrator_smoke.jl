# Check that the same NDArray heat problem runs with several OrdinaryDiffEq solvers.
using cuNumeric
using SciMLBase: ODEProblem, solve, successful_retcode, FullSpecialize
using OrdinaryDiffEqLowOrderRK: RK4
using OrdinaryDiffEqTsit5: Tsit5
using OrdinaryDiffEqVerner: Vern7
using LinearAlgebra: transpose

include(joinpath(@__DIR__, "heat_rhs.jl"))

cuNumeric.allowscalar(false)
const N = isempty(ARGS) ? 128 : parse(Int, only(ARGS))
const ADAPTIVE = get(ENV, "ODE_ADAPTIVE", "0") == "1"
const KAPPA = 0.2f0
N >= 4 || error("N must be at least 4")

k = max(1, round(Int, (N - 1) / 4))
s = Float32[sin(pi * k * (i - 1) / (N - 1)) for i in 1:N]
s[1] = s[end] = 0f0
host_u0 = s * transpose(s)
theta = pi * k / (N - 1)
eigenvalue = 4 * Float64(KAPPA) / DX^2 * (cos(theta) - 1)
reference = exp(eigenvalue) .* host_u0

for (name, algorithm) in (("RK4", RK4()), ("Tsit5", Tsit5()), ("Vern7", Vern7()))
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
    solution = if ADAPTIVE
        cuNumeric.allowautofetch() do
            run_solve()
        end
    else
        run_solve()
    end
    successful_retcode(solution) || error("$name returned $(solution.retcode)")
    state = solution.u[end]
    state isa NDArray{Float32,2} || error("$name returned $(typeof(state))")
    relative_error = maximum(abs.(Array(state) .- reference)) / maximum(abs.(reference))
    relative_error < 1e-4 || error("$name relative error $relative_error exceeds 1e-4")
    println("PASS,$name,N=$N,adaptive=$ADAPTIVE,steps=$(solution.stats.naccept),relative_error=$relative_error")
end
