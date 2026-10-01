# CPU-only coordinator: each child owns its GPU runtime for one warmup and
# exactly one timed solve. Wait for it to exit before launching the next child.
module ODESamples

using Statistics: mean, median, std

function sample_result(log, expected)
    rows = filter(line -> startswith(line, "RESULT,"), readlines(log))
    length(rows) == 1 || error("Expected one RESULT row in $log")
    fields = split(only(rows), ',')
    length(fields) == 13 && fields[2:6] == expected ||
        error("Unexpected sample identity or result format in $log")
    values = parse.(Float64, split(fields[13], ';'))
    length(values) == 1 && isfinite(only(values)) && only(values) > 0 ||
        error("Expected one positive, finite solve time in $log")
    relative_error = parse(Float64, fields[12])
    tolerance = expected[2] == "Float32" ? 1e-4 : 1e-7
    isfinite(relative_error) && 0 <= relative_error < tolerance ||
        error("Sample failed validation in $log")
    return only(values), relative_error
end

function run_samples(log, backend, n; worker=joinpath(@__DIR__, "benchmark_heat.jl"))
    count = parse(Int, get(ENV, "ODE_SAMPLES", "5"))
    count >= 2 || error("ODE_SAMPLES must be at least 2 for a standard error")
    expected = [backend, get(ENV, "ODE_ELTYPE", "Float32"),
                get(ENV, "ODE_GPUS", "1"), string(n), get(ENV, "ODE_STEPS", "20")]
    project = dirname(Base.active_project())
    cmd = addenv(`$(Base.julia_cmd()) --startup-file=no --project=$project $worker $backend $n`,
                 "ODE_SAMPLES" => "1")
    elapsed_ms = Float64[]
    errors = Float64[]
    for sample_index in 1:count
        sample_log = "$(splitext(log)[1])-sample-$sample_index.log"
        println("Starting sample $sample_index/$count in a fresh Julia process; log=$sample_log")
        flush(stdout)
        try
            open(sample_log, "w") do io
                process = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
                try
                    success(process) || error("Sample $sample_index failed: exit status $(process.exitcode), signal $(process.termsignal)")
                finally
                    # An interrupted coordinator must not leave its worker running.
                    if process_running(process)
                        kill(process, Base.SIGKILL)
                        wait(process)
                    end
                end
            end
            elapsed, relative_error = sample_result(sample_log, expected)
            push!(elapsed_ms, elapsed)
            push!(errors, relative_error)
            println("Completed sample $sample_index/$count: $(elapsed) ms, relative_error=$relative_error")
            flush(stdout)
        catch
            if isfile(sample_log)
                println(stderr, "Last 20 lines of $sample_log:")
                lines = readlines(sample_log)
                foreach(line -> println(stderr, line), Iterators.drop(lines, max(0, length(lines) - 20)))
            end
            rethrow()
        end
    end
    # Publish a case only after all requested independent samples validate.
    stderr_ms = std(elapsed_ms) / sqrt(count)
    println("RESULT,$(join(expected, ',')),$(mean(elapsed_ms)),$stderr_ms,$(median(elapsed_ms)),$(minimum(elapsed_ms)),$(maximum(elapsed_ms)),$(maximum(errors)),$(join(elapsed_ms, ';'))")
    flush(stdout)
    return nothing
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 3 || error("Usage: julia run_samples.jl CASE_LOG BACKEND N")
    ODESamples.run_samples(ARGS[1], ARGS[2], parse(Int, ARGS[3]))
end
