# CPU-only coordination shared by ODE and Krylov. Each worker owns its GPU
# runtime, and must exit before the next sample starts.
module ProcessSamples

using Statistics: mean, median, std

function collect_samples(read_sample, log, cmd, count)
    return map(1:count) do sample_index
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
            result = read_sample(sample_log)
            println("Completed sample $sample_index/$count: $(result.elapsed) ms, relative_error=$(result.relative_error)")
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

function print_result(identity, results)
    elapsed_ms = [result.elapsed for result in results]
    relative_error = maximum(result.relative_error for result in results)
    stderr_ms = std(elapsed_ms) / sqrt(length(results))
    # Publish a case only after all requested independent samples validate.
    println("RESULT,$(join(identity, ',')),$(mean(elapsed_ms)),$stderr_ms,$(median(elapsed_ms)),$(minimum(elapsed_ms)),$(maximum(elapsed_ms)),$relative_error,$(join(elapsed_ms, ';'))")
    flush(stdout)
    return nothing
end

end # module
