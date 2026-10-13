# CPU-only test of Monte Carlo config keys and the array-worker protocol.
module MonteCarloLaunchTests
using Test, TOML, LinearAlgebra
include("../src/core.jl")
include_benchmarks(:cunumeric)
include("../src/models.jl")
include("../src/parse_benchmarks.jl")
include("../src/memory.jl")
include("../src/planning.jl")

const ROOT = normpath(joinpath(@__DIR__, ".."))
const CONFIG = joinpath(ROOT, "configs", "multi_gpu", "montecarlo_forms.toml")

@testset "Existing Monte Carlo config produces cuNumeric worker commands" begin
    gs, specs = parse_config(CONFIG; fusion_override=[true, false])
    runs = filter(r -> r.model === :cunumeric,
        plan_runs(specs, gs, TOML.parsefile(CONFIG), parse_plot_groups(CONFIG), 1_000_000))
    @test length(runs) == 16
    @test Set(r.spec.name for r in runs) == Set(("montecarlo", "montecarlo_naive"))
    @test BENCHMARKS["montecarlo"] === MonteCarloMapReduce
    @test BENCHMARKS["montecarlo_naive"] === MonteCarloBroadcast
    for r in runs
        s = r.spec
        @test r.model === :cunumeric
        b = build_benchmark(BENCHMARKS[s.name], parse_bench_type(s.T), r.N, r.M)
        request = WorkerRequest(s.gpus, s.cpus, s.name, s.T, r.N, r.M,
            s.n_iter, s.n_warmup, s.n_trial, gs.check_correctness,
            gs.n_correctness_iter, Float64(total_flops(b)))
        args = collect(common_worker_args(request))
        command = collect(model_worker_command(execution_model(r.model), request, ROOT))
        @test command[end-length(args)+1:end] == args
        @test joinpath(ROOT, "src", "cunumeric", "single.jl") in command
        @test args[2] == name(b)
        @test dims(b) == (parse(Int, args[4]), parse(Int, args[5]))
    end
end

# Stub the array backend; everything else is the production worker path.
const FUSED = Ref(true)
array_backend_entry() = (
    id=:cunumeric, mod=Base, label="CPU protocol test", save_as="cunumeric",
    clock=()->time_ns()/1e3, synchronize=()->nothing, fused=()->FUSED[],
)

mktempdir() do dir
    withenv("CUNUMERIC_BENCH_RESULTS_DIR"=>dir) do
        original_args = copy(ARGS)
        try
            empty!(ARGS)
            append!(ARGS, ["1", "montecarlo", "Float32", "32", "1", "1", "0", "1", "true", "1", "32.0"])
            include("../src/array_worker.jl")
        finally
            empty!(ARGS)
            append!(ARGS, original_args)
        end
    end
end

@testset "Both Monte Carlo keys run through the array worker on CPU" begin
    for key in ("montecarlo", "montecarlo_naive"), T in ("Float32", "Float64"), fused in (true, false)
        mktempdir() do dir
            FUSED[] = fused
            withenv("CUNUMERIC_BENCH_RESULTS_DIR"=>dir) do
                args = ["1", key, T, "64", "1", "2", "1", "2", "true", "1", "64.0"]
                run_array_worker(args)
            end
            backend = fused ? "cunumeric" : "cunumeric_nofusion"
            path = joinpath(dir, "$(key)_$(backend).csv")
            @test isfile(path)
            rows = split.(readlines(path), ',')
            @test length(rows) == 2
            @test all(row[1:4] == [backend, "1", "64", "1"] for row in rows)
            @test all(row[end] == "pass" for row in rows)
            @test all(parse(Float64, row[6]) > 0 for row in rows)
        end
    end
end
end
