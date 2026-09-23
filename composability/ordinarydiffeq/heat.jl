using cuNumeric
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54
using SciMLBase: ODEProblem, solve, FullSpecialize

include(joinpath(@__DIR__, "heat_rhs.jl"))

n = 128
s = Float32.(sin.(range(0, pi; length=n)))
s[1] = s[end] = 0f0
u0 = NDArray(s * transpose(s))
problem = ODEProblem{true,FullSpecialize}(heat!, u0, (0f0, 1f0), 0.2f0)
solution = solve(problem, CarpenterKennedy2N54(williamson_condition=false);
                 dt=0.05f0, save_everystep=false,
                 unstable_check=(dt, u, p, t) -> false)

@show typeof(solution.u[end])
