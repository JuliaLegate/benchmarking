# Run a Krylov or ODE case as independent samples, each in a fresh Julia process,
# and print one aggregated RESULT row once every sample validates.
#   julia run_samples.jl krylov CASE_LOG BACKEND SOLVER MODE N
#   julia run_samples.jl ordinarydiffeq CASE_LOG BACKEND N
# CPU-only: each worker owns its GPU runtime and exits before the next starts.
module RunSamples

using Statistics: mean, median, std

function collect_samples(read_sample, log, cmd, count)
    return map(1:count) do i
        sample_log = "$(splitext(log)[1])-sample-$i.log"
        println("Starting sample $i/$count; log=$sample_log")
        flush(stdout)
        try
            open(sample_log, "w") do io
                process = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
                try
                    success(process) || error("Sample $i failed: exit status $(process.exitcode), signal $(process.termsignal)")
                finally
                    # An interrupted coordinator must not leave its worker running.
                    if process_running(process)
                        kill(process, Base.SIGKILL)
                        wait(process)
                    end
                end
            end
            result = read_sample(sample_log)
            println("Completed sample $i/$count: $(result.elapsed) ms, relative_error=$(result.relative_error)")
            flush(stdout)
            return result
        catch
            if isfile(sample_log)
                println(stderr, "Last 20 lines of $sample_log:")
                lines = readlines(sample_log)
                foreach(line -> println(stderr, line), Iterators.drop(lines, max(0, length(lines) - 20)))
            end
            rethrow()
        end
    end
end

# The worker's single RESULT row, checked against the expected case identity.
function result_fields(log, expected, nfields)
    rows = filter(line -> startswith(line, "RESULT,"), readlines(log))
    length(rows) == 1 || error("Expected one RESULT row in $log")
    fields = split(only(rows), ',')
    length(fields) == nfields && fields[2:(length(expected) + 1)] == expected ||
        error("Unexpected sample identity or result format in $log")
    values = parse.(Float64, split(fields[end], ';'))
    length(values) == 1 && isfinite(only(values)) && only(values) > 0 ||
        error("Expected one positive, finite solve time in $log")
    return fields, only(values), parse(Float64, fields[end - 1])
end

function validated(relative_error, tolerance, log)
    isfinite(relative_error) && 0 <= relative_error <= tolerance ||
        error("Sample failed validation in $log")
    return relative_error
end

function sample_count(prefix)
    count = parse(Int, get(ENV, "$(prefix)_SAMPLES", "5"))
    count >= 2 || error("$(prefix)_SAMPLES must be at least 2 for a standard error")
    return count
end

function print_result(identity, results)
    elapsed_ms = [result.elapsed for result in results]
    relative_error = maximum(result.relative_error for result in results)
    stderr_ms = std(elapsed_ms) / sqrt(length(results))
    println("RESULT,$(join(identity, ',')),$(mean(elapsed_ms)),$stderr_ms,$(median(elapsed_ms)),$(minimum(elapsed_ms)),$(maximum(elapsed_ms)),$relative_error,$(join(elapsed_ms, ';'))")
    flush(stdout)
    return nothing
end

worker_cmd(worker, args, threads=nothing) = `$(Base.julia_cmd()) $(threads === nothing ? String[] : ["-t$threads"]) --startup-file=no --project=$(dirname(Base.active_project())) $worker $args`

function krylov(log, backend, solver, mode, n; worker=joinpath(@__DIR__, "krylov", "krylov.jl"))
    label = backend == "CuArray" ? "CUDA" :
            backend == "cuNumeric" && mode != "stock" ? "cuNumeric $mode" : backend
    eltype = get(ENV, "BENCH_ELTYPE", "Float32")
    expected = [label, solver, mode, eltype, get(ENV, "BENCH_GPUS", "1"), string(n)]
    tolerance = eltype == "Float32" ? Float64(Float32(1e-5)) : 1e-8
    cmd = addenv(worker_cmd(worker, [backend, solver, mode, string(n)], get(ENV, "BENCH_THREADS", "4")),
        "BENCH_SAMPLES" => "1")
    results = collect_samples(log, cmd, sample_count("BENCH")) do sample_log
        fields, elapsed, relative_error = result_fields(sample_log, expected, 15)
        iterations = parse(Int, fields[8])
        iterations >= 0 || error("Invalid iteration count in $sample_log")
        (; elapsed, relative_error=validated(relative_error, tolerance, sample_log), iterations)
    end
    # Per-sample iteration counts remain in the worker logs.
    return print_result([expected; string(maximum(r.iterations for r in results))], results)
end

function ordinarydiffeq(log, backend, n; worker=joinpath(@__DIR__, "ordinarydiffeq", "benchmark_heat.jl"))
    eltype = get(ENV, "ODE_ELTYPE", "Float32")
    expected = [backend, eltype, get(ENV, "ODE_GPUS", "1"), string(n), get(ENV, "ODE_STEPS", "20")]
    tolerance = eltype == "Float32" ? 1e-4 : 1e-7
    cmd = addenv(worker_cmd(worker, [backend, string(n)]), "ODE_SAMPLES" => "1")
    results = collect_samples(log, cmd, sample_count("ODE")) do sample_log
        _, elapsed, relative_error = result_fields(sample_log, expected, 13)
        (; elapsed, relative_error=validated(relative_error, tolerance, sample_log))
    end
    return print_result(expected, results)
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    workload, log, args... = ARGS
    if workload == "krylov" && length(args) == 4
        RunSamples.krylov(log, args[1:3]..., parse(Int, args[4]))
    elseif workload == "ordinarydiffeq" && length(args) == 2
        RunSamples.ordinarydiffeq(log, args[1], parse(Int, args[2]))
    else
        error("Usage: julia run_samples.jl krylov CASE_LOG BACKEND SOLVER MODE N | ordinarydiffeq CASE_LOG BACKEND N")
    end
end
