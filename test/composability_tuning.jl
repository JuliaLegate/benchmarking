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

include("../composability/tune.jl")
using .ComposabilityTuning: tune, CASES, run_worker, mean_time

@testset "Composability tuning coordinator" begin
    mktempdir() do output
        preview = IOBuffer()
        unused = joinpath(output, "dry-run")
        @test tune("ordinarydiffeq"; gpus=8, output=unused, dry=true, io=preview) == 0
        @test !ispath(unused)
        @test occursin("N=92682", String(take!(preview)))
        config = joinpath(@__DIR__, "..", "composability", "sizes_80GB.toml")
        @test tune("ordinarydiffeq"; gpus=8, config, output=unused, dry=true, io=preview) == 0
        @test occursin("N=46341", String(take!(preview)))

        for (name, spec) in CASES
            calls = Int[]
            n = spec.workload == "krylov" ? 260000 : spec.workload == "ordinarydiffeq" ? 65536 : 20480
            function executor(cmd, log)
                env = Dict(Pair(split(entry, '='; limit=2)...) for entry in cmd.env)
                b = parse(Int, env["DAGGER_BLOCKS_PER_GPU"])
                push!(calls, b)
                @test env["COMPOSABILITY_TUNE"] == "1"
                @test env["$(spec.prefix)_GPUS"] == "4"
                @test env["$(spec.prefix)_ELTYPE"] == "Float32"
                @test cmd.exec[end-length(spec.args)-1:end] == ["Dagger"; spec.args; string(n)]
                ms = b == 1 ? 22 : b == 2 ? 55 : b == 4 ? 155 : b == 8 ? 176 : 177
                row = if spec.workload == "krylov"
                    "Dagger,$(spec.args[1]),stock,Float32,4,$n,5,$ms,0,$ms,$ms,$ms,0.01,$ms;$ms"
                elseif spec.workload == "ordinarydiffeq"
                    "Dagger,Float32,4,$n,5,$ms,0,$ms,$ms,$ms,0.000001,$ms;$ms"
                else
                    "Dagger,Float32,4,$n,4,12,5,10,$ms,0,$ms,$ms,$ms,1,0.1,0.2,$ms;$ms"
                end
                write(log, "RESULT,$row\n")
            end
            @test tune(name; gpus=4, output, executor, io=IOBuffer()) == 0
            @test calls == [1, 2, 4, 8, 16] # Continue at 8x; stop only above it.
            stem = replace(name, '_' => '-')
            @test length(readlines(joinpath(output, "$stem.csv"))) == 6
            best = split(last(readlines(joinpath(output, "$stem-best.csv"))), ',')
            @test best[6:7] == ["1", "22.0"]
        end

        # Process failures and malformed results cannot win or stop later candidates.
        for bad in ("failure", "malformed", "duplicate", "nonfinite")
            calls = Int[]
            function executor(cmd, log)
                env = Dict(Pair(split(entry, '='; limit=2)...) for entry in cmd.env)
                b = parse(Int, env["DAGGER_BLOCKS_PER_GPU"])
                push!(calls, b)
                b == 2 && bad == "failure" && error("Simulated worker failure")
                ms = b == 1 ? "10" : b == 2 && bad == "nonfinite" ? "NaN" : b <= 4 ? "50" : "90"
                row = "RESULT,Dagger,Float32,1,32768,5,$ms,0,$ms,$ms,$ms,0.01,$ms;$ms\n"
                write(log, b != 2 ? row : bad == "malformed" ? "bad\n" : bad == "duplicate" ? row * row : row)
            end
            @test tune("ordinarydiffeq"; output=joinpath(output, bad), gpus=1, executor, io=IOBuffer()) == 1
            @test calls == [1, 2, 4, 8]
        end

        log = joinpath(output, "real-process.log")
        run_worker(`$(Base.julia_cmd()) --startup-file=no -e 'println("RESULT,3.5")'`, log)
        @test mean_time(log, 1) == 3.5
        @test_throws ErrorException run_worker(`$(Base.julia_cmd()) --startup-file=no -e 'exit(7)'`, log)
    end
end
