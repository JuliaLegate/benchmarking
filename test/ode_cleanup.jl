module ODECleanupTests
using Test, LinearAlgebra, Statistics

# Exercise the actual solve loop with finalizable CPU objects, without GPU
# packages. A simulated deferred-free queue models the cuNumeric boundary.
source = Meta.parseall(read(joinpath(@__DIR__, "..", "composability",
    "ordinarydiffeq", "benchmark_heat.jl"), String))
for name in (:initial_state, :cleanup_before_solve, :run_case)
    definition = only(expr for expr in source.args if
        expr isa Expr && expr.head == :function &&
        expr.args[1] isa Expr && expr.args[1].head == :call &&
        expr.args[1].args[1] == name)
    Core.eval(@__MODULE__, definition)
end

const T = Float32
const KAPPA, T_END, DX = T(0.2), T(1), 1
const NSTEPS, SAMPLES, GPUS = 20, 5, 1
const ALG, FullSpecialize = nothing, Nothing
backend = "cpu"
const LIVE = Ref(0)
const CALLS = Ref(0)

module cuNumeric
const PENDING = Ref(0)
const DRAINS = Ref(0)
const CLOCK = Ref(0)
function drain_pending_frees!()
    PENDING[] = 0
    DRAINS[] += 1
    CLOCK[] += 100_000_000 # Deliberately expensive cleanup must stay untimed.
end
end

struct ODEProblem{IIP,S}
    u0::Matrix{Float32}
end
ODEProblem{IIP,S}(f, u0, tspan, p) where {IIP,S} = ODEProblem{IIP,S}(u0)
heat!(args...) = nothing
make_state(a) = copy(a)
host_state(a) = a
synchronized_time_ns(args...) = cuNumeric.CLOCK[]
successful_retcode(sol) = true

mutable struct Solution
    u::Vector{Matrix{Float32}}
end

function solve(prob, alg; kwargs...)
    # Detect a retained previous solution or cleanup postponed to this solve.
    @test LIVE[] == 0
    @test cuNumeric.PENDING[] == 0
    @test cuNumeric.DRAINS[] == (backend == "cuNumeric" ? CALLS[] + 1 : 0)
    CALLS[] += 1
    n = size(prob.u0, 1)
    k = max(1, round(Int, (n - 1) / 4))
    eigenvalue = 4 * Float64(KAPPA) / DX^2 * (cos(pi * k / (n - 1)) - 1)
    sol = Solution([T.(exp(eigenvalue * T_END) .* prob.u0)])
    LIVE[] += 1
    deferred = backend == "cuNumeric"
    finalizer(sol) do _
        LIVE[] -= 1
        deferred && (cuNumeric.PENDING[] += 1)
    end
    cuNumeric.CLOCK[] += 2_000_000
    return sol
end

@testset "ODE sample cleanup and timing" begin
    for label in ("cpu", "CuArray", "Dagger", "cuNumeric")
        global backend = label
        LIVE[] = CALLS[] = cuNumeric.PENDING[] = cuNumeric.DRAINS[] = cuNumeric.CLOCK[] = 0
        mktemp() do path, io
            redirect_stdout(io) do
                run_case(8)
            end
            flush(io)
            rows = filter(line -> startswith(line, "RESULT,"), readlines(path))
            fields = split(only(rows), ',')
            @test fields[2] == label
            @test parse.(Float64, split(fields[end], ';')) == fill(2.0, SAMPLES)
            @test parse(Float64, fields[end - 1]) < 1e-4
        end
        @test CALLS[] == 2 + SAMPLES
        GC.gc(true)
        @test LIVE[] == 0
    end
end
end
