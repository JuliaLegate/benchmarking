# CPU-only: evaluate worker configuration without loading GPU/solver packages.
using Test

@testset "Tuning cannot inherit full benchmark budgets" begin
    for (worker, prefix, limit, normal) in (
        ("krylov/krylov.jl", "BENCH", :MAXITERS, 200),
        ("ordinarydiffeq/benchmark_heat.jl", "ODE", :NSTEPS, 20),
        ("integrals_optimization/benchmark.jl", "INTOPT", :ITERS, 80),
    ), tuning in (false, true)
        source = Meta.parseall(read(joinpath(@__DIR__, "..", "composability", worker), String))
        # Read the actual worker constants, including their environment handling.
        settings = filter(source.args) do expr
            expr isa Expr && expr.head == :const &&
                expr.args[1].args[1] in (:TUNING, :SAMPLES, limit, :T_END)
        end
        config = Module(gensym(:WorkerConfig))
        Core.eval(config, :(const T = Float32))
        withenv("COMPOSABILITY_TUNE" => string(Int(tuning)),
                "$(prefix)_SAMPLES" => (tuning ? "999" : nothing),
                "ODE_STEPS" => (tuning ? "999" : nothing),
                "INTOPT_ITERS" => (tuning ? "999" : nothing)) do
            foreach(expr -> Core.eval(config, expr), settings)
        end
        Base.invokelatest() do
            @test config.TUNING == tuning
            @test config.SAMPLES == (tuning ? 2 : 5)
            @test getfield(config, limit) == (tuning ? 5 : normal)
            if prefix == "ODE"
                @test config.T_END / config.NSTEPS ≈ 0.05
            end
        end
    end
end
