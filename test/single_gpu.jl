# CPU-only parser, planner, and launcher checks. Also runnable independently:
# julia --startup-file=no --project=. test/single_gpu.jl
module SingleGPUSweeps
using Test, TOML, LinearAlgebra
include("../run.jl") # cuPyNumeric environment-name resolution, without running main
include("../src/core.jl")
include_benchmarks()
include("../src/models.jl")
include("../src/parse_benchmarks.jl")
include("../src/memory.jl")
include("../src/planning.jl")
include("../src/model_worker.jl")
include("../src/runner.jl")

const CONFIG_TEXT = """
[Global]
single_gpu = true
models = ["cunumeric", "cudajl"]
n_warmup = 1
n_iter = 3
n_trial = 2

[[montecarlo]]
T = ["Float32", "Float64"]
cpus = [2, 4]
fusion = ["on", "off"]
N = [64, 128]
"""

function with_config(f, source=CONFIG_TEXT)
    mktempdir() do dir
        path = joinpath(dir, "sweep.toml")
        write(path, source)
        f(path)
    end
end

@testset "Single-GPU size sweep" begin
    with_config() do path
        gs, specs = parse_config(path)
        raw = TOML.parsefile(path)
        groups = parse_plot_groups(path)
        @test length(specs) == 8
        @test all(s.gpus == 1 for s in specs)
        @test Set((s.args[1], s.args[2], s.cpus) for s in specs) ==
            Set(((64, 1, 2), (128, 1, 4)))
        @test Set((s.T, s.fusion) for s in specs) ==
            Set((T, f) for T in ("Float32", "Float64") for f in (true, false))
        runs = plan_runs(specs, gs, raw, groups, 1_000_000)
        @test length(runs) == 12
        @test count(r -> r.model == :cunumeric, runs) == 8
        @test count(r -> r.model == :cudajl, runs) == 4
        @test Set((r.spec.T, r.N, r.M) for r in runs if r.model == :cudajl) ==
            Set((T, N, 1) for T in ("Float32", "Float64") for N in (64, 128))

        _, filtered = parse_config(path; only="montecarlo", fusion_override=[false],
            models_override=[:cunumeric])
        @test length(filtered) == 4
        @test all(!s.fusion && s.models == [:cunumeric] && s.gpus == 1 for s in filtered)
        @test main(["--config=$path", "--dry-run"];
            budget_provider=(f, p) -> (@test p == 1; (1_000_000, f)),
            executor=(args...) -> error("dry-run launched workers")) == 0

        # Positional/programmatic callers cannot override the fixed GPU count.
        positional = positional_spec(
            ["2", "2", "montecarlo", "Float32", "64", "1", "3", "1", "2"], gs)
        @test_throws "requires exactly one GPU" plan_runs(
            [positional], gs, raw, groups, 1_000_000)

        # Multiple dimensions must still be rejected in GPU-count mode.
        raw["Global"]["single_gpu"] = false
        @test_throws "incompatible pinned dimensions" plan_runs(
            specs, gs, raw, groups, 1_000_000)
    end
end

@testset "Single-GPU configuration validation" begin
    for value in ("1", "[1]", "2", "[1, 2]")
        with_config(replace(CONFIG_TEXT, "cpus = [2, 4]" => "cpus = [2, 4]\ngpus = $value")) do path
            @test_throws "omit gpus" parse_config(path)
        end
    end
    for value in ("1", "\"true\"", "[true]")
        with_config(replace(CONFIG_TEXT, "single_gpu = true" => "single_gpu = $value")) do path
            @test_throws "Global.single_gpu must be true or false" parse_config(path)
        end
    end
    with_config(replace(CONFIG_TEXT, "N = [64, 128]" => "N = [64, 128, 256]")) do path
        @test_throws "must share one length" parse_config(path)
    end
    for flag in ("single_gpu = false", "")
        text = replace(CONFIG_TEXT, "single_gpu = true" => flag,
            "cpus = [2, 4]" => "cpus = [2, 4]\ngpus = [1, 2]")
        with_config(text) do path
            gs, specs = parse_config(path)
            @test Set(s.gpus for s in specs) == Set((1, 2))
            @test length(plan_runs(specs, gs, TOML.parsefile(path),
                parse_plot_groups(path), 1_000_000)) == 10
        end
    end
    with_config(replace(CONFIG_TEXT, "single_gpu = true" => "")) do path
        @test_throws KeyError parse_config(path)
    end
    auto = replace(CONFIG_TEXT, "single_gpu = true" => "single_gpu = true\nauto_size = true",
        "N = [64, 128]" => "N = \"auto\"", "cpus = [2, 4]" => "cpus = 2")
    with_config(auto) do path
        gs, specs = parse_config(path)
        runs = plan_runs(specs, gs, TOML.parsefile(path), parse_plot_groups(path), 1_000_000)
        @test all(s.autosize && s.gpus == 1 for s in specs)
        @test length(runs) == 6
        @test all(r.N > 0 && r.M == 1 && peak_bytes(r.memory) <= r.budget for r in runs)
    end
end

@testset "Single-GPU example configs and execution" begin
    for (file, series_per_size, n_sizes, reference) in (
        ("grayscott_forms.toml", 4, 6, "grayscott_plain"),
        ("montecarlo_forms.toml", 5, 5, "montecarlo"),
    )
        path = joinpath(@__DIR__, "..", "configs", "single_gpu", file)
        gs, specs = parse_config(path)
        raw = TOML.parsefile(path)
        runs = plan_runs(specs, gs, raw, parse_plot_groups(path), 10^12)
        @test all(r.spec.gpus == 1 for r in runs)
        @test length(runs) == n_sizes * series_per_size
        @test count(r -> r.model == :cudajl, runs) == n_sizes
        @test all(r.spec.name == reference for r in runs if r.model == :cudajl)
        @test count(r -> r.model == :cupynumeric, runs) == n_sizes
        native_name = startswith(file, "grayscott") ? "grayscott" : "montecarlo"
        @test all(r.spec.name == native_name for r in runs if r.model == :cupynumeric)
        @test !uses_fusion(execution_model(:cupynumeric))
        @test !supports_benchmark(execution_model(:cupynumeric), "grayscott_plain")
        @test !supports_benchmark(execution_model(:cupynumeric), "grayscott_function_accelerated")
        @test !supports_benchmark(execution_model(:cupynumeric), "montecarlo_naive")
        @test length(unique((r.N, r.M) for r in runs)) == n_sizes
        if startswith(file, "grayscott")
            @test all(r.N == r.M for r in runs)
            @test Set(r.spec.name for r in runs) ==
                Set(("grayscott_plain", "grayscott_function_accelerated", "grayscott"))
            @test all(r.model == :cupynumeric for r in runs if r.spec.name == "grayscott")
            @test all(!r.spec.fusion for r in runs if r.spec.name == "grayscott_plain")
            @test all(r.spec.fusion for r in runs if r.spec.name == "grayscott_function_accelerated")
        else
            @test all(r.spec.fusion for r in runs if r.spec.name == "montecarlo")
            @test Set(r.spec.fusion for r in runs if r.spec.name == "montecarlo_naive") ==
                Set((true, false))
        end

        mktempdir() do dir
            calls = Cmd[]
            # Commands are recorded, so no real Conda environment is required.
            status = withenv("CUPYNUMERIC_ENV" => "single-gpu-test") do
                execute_plan(runs, gs, cli_options(["--config=$path"]), 10^12, raw;
                    launch=cmd -> push!(calls, cmd), prepare=(f, v) -> nothing,
                    results_root=dir, preflight=runs -> nothing)
            end
            @test status == 0
            @test length(calls) == length(runs)
            @test !any(occursin("plot_results.jl", join(c.exec)) for c in calls)
            manifest = TOML.parsefile(joinpath(only(readdir(dir; join=true)), "manifest.toml"))
            @test manifest["status"] == "complete"
            @test manifest["config"]["Global"]["single_gpu"]
            @test all(r["gpus"] == 1 && r["status"] == "complete" for r in manifest["runs"])
        end
    end
end
end # module
