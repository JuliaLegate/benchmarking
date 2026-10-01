# Keep orchestration off the GPU; the shell launchers isolate each timed case.
module ComposabilityCLI

using Dates, TOML

const WORKLOADS = ["krylov", "ordinarydiffeq", "integrals_optimization"]
const ROOT = @__DIR__
const DEFAULT_CONFIG = joinpath(ROOT, "composability", "sizes_80GB.toml")
# --models names -> launcher backend names. CUDA.jl (CuArray) is single-GPU only.
const MODEL_BACKENDS = Dict("cuda" => "CuArray", "dagger" => "Dagger", "cunumeric" => "cuNumeric")
const MODELS = ["cuda", "dagger", "cunumeric"]
const LAUNCHERS = Dict(
    "krylov" => (script="run.sh", prefix="BENCH"),
    "ordinarydiffeq" => (script="run_benchmark.sh", prefix="ODE"),
    "integrals_optimization" => (script="run_benchmark.sh", prefix="INTOPT"),
)

function usage(io=stdout)
    println(io, """
    Usage: julia --project=. run_composability.jl [options]

      --only=all                  Run all three workloads (default), or select a
                                  comma-separated list: krylov,ordinarydiffeq,
                                  integrals_optimization
      --mode=single               single (default), multi, or both
      --solvers=cg                Krylov solvers: cg, bicgstab, or cg,bicgstab
      --local                     Also run cuNumeric local Krylov implementations
      --models=cuda,dagger,cunumeric
                                  Models to run (default: all); CUDA.jl runs only
                                  in single mode, e.g. --models=cuda,cunumeric
      --gpus=1,2,4,8              GPU counts for multi mode (default: 1,2,4,8);
                                  single mode always uses one GPU
      --config=PATH               Size config (default: composability/sizes_80GB.toml)
      --output=PATH               Result root (default: results/composability-<run-id>)
      --dry-run                   Print launch commands without running or writing files
      -h, --help                  Show this help

    Edit single and weak_base in the size config to set N. Defaults are Float32
    presets for 80 GB H100s; select composability/sizes_141GB.toml for 141 GB H200s.
    Multi mode uses N(G) = round(weak_base * sqrt(G)). Results go to
    PATH/single/<workload> and PATH/multi/<workload>.
    Existing result CSVs are never overwritten.

    Run ./instantiate_projects.sh first. Workers use environments/composability;
    JULIA or CUNUMERIC_BENCH_JULIA can override the current Julia executable.
    Existing BENCH_*, ODE_*, and INTOPT_* settings still control the workers.

    Examples:
      julia --project=. run_composability.jl --only=krylov
      julia --project=. run_composability.jl --only=krylov --solvers=cg,bicgstab --mode=both
      julia --project=. run_composability.jl --only=krylov --solvers=cg,bicgstab --local
      julia --project=. run_composability.jl --only=krylov,ordinarydiffeq --mode=multi
      julia --project=. run_composability.jl --mode=multi --models=cunumeric
      julia --project=. run_composability.jl --mode=both --gpus=1,2,4 --output=results/paper
      julia --project=. run_composability.jl --config=composability/sizes_141GB.toml
      julia --project=. run_composability.jl --config=my-sizes.toml --dry-run
    """)
end

function cli_options(args)
    only = "all"
    mode = "single"
    solver_arg = "cg"
    solvers_explicit = false
    include_local = false
    model_arg = join(MODELS, ',')
    gpu_arg = nothing
    output = nothing
    config = DEFAULT_CONFIG
    dry = false
    for arg in args
        if arg == "--dry-run"
            dry = true
        elseif arg == "--local"
            include_local = true
        elseif startswith(arg, "--only=")
            only = split(arg, '='; limit=2)[2]
        elseif startswith(arg, "--mode=")
            mode = split(arg, '='; limit=2)[2]
        elseif startswith(arg, "--solvers=")
            solver_arg = split(arg, '='; limit=2)[2]
            solvers_explicit = true
        elseif startswith(arg, "--models=")
            model_arg = split(arg, '='; limit=2)[2]
        elseif startswith(arg, "--gpus=")
            gpu_arg = split(arg, '='; limit=2)[2]
        elseif startswith(arg, "--config=")
            config = split(arg, '='; limit=2)[2]
            isempty(strip(config)) && throw(ArgumentError("--config requires a path"))
        elseif startswith(arg, "--output=")
            output = split(arg, '='; limit=2)[2]
            isempty(strip(output)) && throw(ArgumentError("--output requires a path"))
        else
            throw(ArgumentError("Unknown option $arg; use --help"))
        end
    end
    mode in ("single", "multi", "both") ||
        throw(ArgumentError("--mode must be single, multi, or both"))
    workloads = only == "all" ? copy(WORKLOADS) : String.(strip.(split(only, ',')))
    all(w -> w in WORKLOADS, workloads) ||
        throw(ArgumentError("--only must be all or a list of $(join(WORKLOADS, ","))"))
    allunique(workloads) || throw(ArgumentError("--only contains duplicate workloads"))
    solvers = String.(strip.(split(solver_arg, ',')))
    all(s -> s in ("cg", "bicgstab"), solvers) && allunique(solvers) ||
        throw(ArgumentError("--solvers must be cg, bicgstab, or a comma-separated list without duplicates"))
    solvers_explicit && !("krylov" in workloads) &&
        throw(ArgumentError("--solvers requires krylov in --only"))
    include_local && !("krylov" in workloads) &&
        throw(ArgumentError("--local requires krylov in --only"))
    models = String.(strip.(split(lowercase(model_arg), ',')))
    all(m -> m in MODELS, models) && allunique(models) ||
        throw(ArgumentError("--models must be a list of $(join(MODELS, ","))"))
    mode != "single" && models == ["cuda"] &&
        throw(ArgumentError("--models=cuda only runs in single mode"))
    gpu_values = gpu_arg === nothing ? (mode == "single" ? "1" : "1,2,4,8") : gpu_arg
    gpus = String.(strip.(split(gpu_values, ',')))
    all(g -> g in ("1", "2", "4", "8"), gpus) ||
        throw(ArgumentError("--gpus must be a list of 1,2,4,8"))
    allunique(gpus) || throw(ArgumentError("--gpus contains duplicate counts"))
    mode == "single" && gpus != ["1"] &&
        throw(ArgumentError("Use --mode=multi or --mode=both for multiple GPUs"))
    if output === nothing
        run_id = Dates.format(now(), "yyyymmdd-HHMMSS-sss") * "-$(getpid())"
        output = joinpath(ROOT, "results", "composability-$run_id")
    end
    return (; workloads, mode, solvers, include_local, models, gpus, output=abspath(output), config=abspath(config), dry)
end

function workload_sizes(config, workload)
    haskey(config, workload) || throw(ArgumentError("Missing [$workload] in size config"))
    values = config[workload]
    values isa AbstractDict && Set(keys(values)) == Set(["single", "weak_base"]) ||
        throw(ArgumentError("[$workload] must contain single and weak_base"))
    minimum_n = workload == "krylov" ? 2 : 4
    valid_n(n) = n isa Int && n >= minimum_n
    single, base = values["single"], values["weak_base"]
    single isa AbstractVector && !isempty(single) && all(valid_n, single) &&
        issorted(single) && allunique(single) ||
        throw(ArgumentError("$workload.single must be increasing, unique integers >= $minimum_n"))
    valid_n(base) || throw(ArgumentError("$workload.weak_base must be an integer >= $minimum_n"))
    return (; single, base)
end

function launch_plan(opts; env=ENV)
    config_text = read(opts.config, String)
    config = TOML.parse(config_text)
    modes = opts.mode == "both" ? ["single", "multi"] : [opts.mode]
    julia = get(env, "JULIA", get(env, "CUNUMERIC_BENCH_JULIA",
        joinpath(Sys.BINDIR, Base.julia_exename())))
    plan = NamedTuple[]
    for mode in modes, workload in opts.workloads
        spec = LAUNCHERS[workload]
        sizes = workload_sizes(config, workload)
        base_n = sizes.base
        output = joinpath(opts.output, mode, workload)
        args = if mode == "single"
            ["single", string.(sizes.single)...]
        else
            ["weak", string(base_n), opts.gpus...]
        end
        cmd = Cmd(["bash", joinpath(ROOT, "composability", workload, spec.script), args...])
        overrides = [
            "$(spec.prefix)_OUTPUT" => output,
            # CLI previews never invoke a launcher. A real run must not inherit
            # a workload's old dry-run setting from the calling shell.
            "$(spec.prefix)_DRY_RUN" => "0",
            "JULIA" => julia,
            # Workers disable startup.jl. Preserve depots added by the parent
            # startup (the container adds /depot there) so packages stay visible.
            "JULIA_DEPOT_PATH" => join(DEPOT_PATH, Sys.iswindows() ? ';' : ':'),
            "$(spec.prefix)_BACKENDS" => join(
                [MODEL_BACKENDS[m] for m in opts.models if mode == "single" || m != "cuda"], ' '),
        ]
        if workload == "krylov"
            push!(overrides, "BENCH_SOLVERS" => join(opts.solvers, ','),
                "BENCH_LOCAL" => (opts.include_local ? "1" : "0"))
        end
        cmd = addenv(cmd, overrides...)
        push!(plan, (; mode, workload, output, cmd, overrides, config_text, base_n))
    end
    return plan
end

# Unlike success(cmd), this preserves the launcher's progress and errors.
run_launcher(cmd) = success(pipeline(cmd; stdout, stderr))

function report_failure(launch, io)
    println(io, "FAILED: $(launch.workload) $(launch.mode); logs: $(launch.output)")
    # Startup checks redirect their output before any per-case logs exist.
    planned = joinpath(launch.output, "planned-cases.csv")
    if !isfile(planned) || length(readlines(planned)) <= 1
        filename = launch.workload == "krylov" ? "environment.txt" : "metadata.txt"
        path = joinpath(launch.output, filename)
        if isfile(path)
            println(io, "Startup diagnostics ($path, last 20 lines):")
            lines = readlines(path)
            for line in Iterators.drop(lines, max(0, length(lines) - 20))
                println(io, line)
            end
        end
    end
end

function main(args=ARGS; executor=run_launcher, io=stdout)
    if any(arg -> arg in ("-h", "--help"), args)
        usage(io)
        return 0
    end
    opts = cli_options(args)
    plan = launch_plan(opts)
    if !opts.dry
        for launch in plan
            csv = joinpath(launch.output, "results.csv")
            ispath(csv) && throw(ArgumentError("Existing results would be overwritten: $csv"))
        end
    end
    println(io, "Workloads: ", join(opts.workloads, ", "))
    "krylov" in opts.workloads && println(io, "Krylov solvers: ", join(opts.solvers, ", "))
    failed = false
    saved_modes = Set{String}()
    for launch in plan
        counts = launch.mode == "single" ? "1" : join(opts.gpus, ',')
        println(io, "$(launch.workload) $(launch.mode): GPUs=$counts; results=$(launch.output)")
        if opts.dry
            assignments = ["$key=$value" for (key, value) in launch.overrides]
            println(io, "  ", Cmd(["env", assignments..., launch.cmd.exec...]))
        else
            if !(launch.mode in saved_modes)
                mode_dir = dirname(launch.output)
                mkpath(mode_dir)
                write(joinpath(mode_dir, "sizes.toml"), launch.config_text)
                if launch.mode == "multi"
                    open(joinpath(mode_dir, "weak_scaling_plan.csv"), "w") do f
                        println(f, "workload,base_n,n_1,n_2,n_4,n_8")
                        for p in filter(p -> p.mode == "multi", plan)
                            dimensions = [round(Int, p.base_n * sqrt(g)) for g in (1, 2, 4, 8)]
                            println(f, join([p.workload, p.base_n, dimensions...], ','))
                        end
                    end
                end
                push!(saved_modes, launch.mode)
            end
            # Continue with the other workloads/modes while retaining failure logs.
            if !executor(launch.cmd)
                failed = true
                report_failure(launch, io)
            end
        end
    end
    return failed ? 1 : 0
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    code = try
        ComposabilityCLI.main()
    catch err
        err isa InterruptException && rethrow()
        showerror(stderr, err)
        println(stderr)
        err isa ArgumentError ? 2 : 1
    end
    exit(code)
end
