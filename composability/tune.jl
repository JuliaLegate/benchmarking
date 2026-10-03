# CPU-only coordinator used by ../tune_dagger.sh. GPU packages load in workers.
module ComposabilityTuning

using Dates, TOML
include("../run_composability.jl")

const SPLITS = [1, 2, 4, 8, 16, 32, 64]
const GIVE_UP = 8.0
const HEADER = "timestamp,name,eltype,n,gpus,blocks_per_gpu,mean_ms"
const CASES = Dict(
    "krylov_cg" => (workload="krylov", prefix="BENCH", worker="krylov/krylov.jl", args=["cg", "stock"], mean_column=8),
    "krylov_bicgstab" => (workload="krylov", prefix="BENCH", worker="krylov/krylov.jl", args=["bicgstab", "stock"], mean_column=8),
    "ordinarydiffeq" => (workload="ordinarydiffeq", prefix="ODE", worker="ordinarydiffeq/benchmark_heat.jl", args=String[], mean_column=6),
    "integrals_optimization" => (workload="integrals_optimization", prefix="INTOPT", worker="integrals_optimization/benchmark.jl", args=String[], mean_column=9),
)

function run_worker(cmd, log)
    open(log, "w") do io
        success(pipeline(cmd; stdout=io, stderr=io)) || error("Worker failed; see $log")
    end
end

function mean_time(log, column)
    rows = filter(startswith("RESULT,"), readlines(log))
    length(rows) == 1 || error("Expected one RESULT row in $log")
    ms = parse(Float64, split(only(rows), ',')[column + 1])
    isfinite(ms) && ms > 0 || error("Invalid timing in $log")
    return ms
end

function record(csv, row)
    fresh = !isfile(csv)
    open(csv, "a") do io
        fresh && println(io, HEADER)
        println(io, row)
    end
end

function tune(name; gpus=parse(Int, get(ENV, "CUNUMERIC_BENCH_GPUS", "1")),
              config=get(ENV, "DAGGER_TUNE_CONFIG", joinpath(@__DIR__, "sizes_141GB.toml")),
              output=get(ENV, "DAGGER_TUNE_OUTPUT", joinpath(@__DIR__, "..", "tunes", "composability")),
              dry=false, executor=run_worker, io=stdout)
    spec = CASES[name]
    gpus in (1, 2, 4, 8) || error("GPU count must be 1, 2, 4, or 8")
    base = ComposabilityCLI.workload_sizes(TOML.parsefile(config), spec.workload).base
    n = round(Int, base * sqrt(gpus))
    stem = replace(name, '_' => '-')
    csv, best_csv = joinpath(output, "$stem.csv"), joinpath(output, "$stem-best.csv")
    logdir = ""
    if !dry
        for path in (csv, best_csv)
            isfile(path) && readline(path) != HEADER && error("Unexpected CSV header: $path")
        end
        mkpath(joinpath(output, "logs"))
        logdir = mktempdir(joinpath(output, "logs"); prefix="$stem-g$gpus-", cleanup=false)
        cp(config, joinpath(logdir, "sizes.toml"))
    end
    project = dirname(Base.active_project())
    best, best_row, best_split = Inf, "", 0
    failed = false
    for split in SPLITS
        overrides = ["COMPOSABILITY_TUNE" => "1", "DAGGER_BLOCKS_PER_GPU" => string(split),
                     "$(spec.prefix)_GPUS" => string(gpus), "$(spec.prefix)_ELTYPE" => "Float32",
                     "JULIA_DEPOT_PATH" => join(DEPOT_PATH, Sys.iswindows() ? ';' : ':'),
                     "OPENBLAS_NUM_THREADS" => "1", "OMP_NUM_THREADS" => "1"]
        worker = joinpath(@__DIR__, spec.worker)
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project --threads=$(Threads.nthreads()) $worker Dagger $(spec.args) $n`
        println(io, "==> $name G=$gpus N=$n blocks_per_gpu=$split")
        if dry
            println(io, Cmd(["env"; ["$k=$v" for (k, v) in overrides]; cmd.exec]))
            continue
        end
        log = joinpath(logdir, "blocks-$split.log")
        try
            executor(addenv(cmd, overrides...), log)
            ms = mean_time(log, spec.mean_column)
            row = join((now(), name, "Float32", n, gpus, split, ms), ',')
            record(csv, row)
            if ms < best
                best, best_row, best_split = ms, row, split
            end
            println(io, "    $ms ms/run")
            ms > GIVE_UP * best && break
        catch err
            err isa InterruptException && rethrow()
            println(io, "Failed $name G=$gpus blocks=$split: ", sprint(showerror, err), "; log=$log")
            failed = true
        end
    end
    if !isempty(best_row)
        record(best_csv, best_row)
        println(io, "best: $name G=$gpus N=$n blocks_per_gpu=$best_split ($best ms/run) -> $csv")
    end
    return failed ? 1 : 0
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) in (1, 2) && (length(ARGS) == 1 || ARGS[2] == "--dry-run") ||
        error("Usage: tune.jl <krylov_cg|krylov_bicgstab|ordinarydiffeq|integrals_optimization> [--dry-run]")
    exit(ComposabilityTuning.tune(ARGS[1]; dry=length(ARGS) == 2))
end
