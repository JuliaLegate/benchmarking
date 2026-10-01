# CPU-only coordinator; GPU packages are loaded only by the sample workers.
module KrylovSamples

include("../process_samples.jl")
using .ProcessSamples: collect_samples, print_result

function sample_result(log, expected)
    rows = filter(line -> startswith(line, "RESULT,"), readlines(log))
    length(rows) == 1 || error("Expected one RESULT row in $log")
    fields = split(only(rows), ',')
    length(fields) == 15 && fields[2:7] == expected ||
        error("Unexpected sample identity or result format in $log")
    values = parse.(Float64, split(fields[15], ';'))
    length(values) == 1 && isfinite(only(values)) && only(values) > 0 ||
        error("Expected one positive, finite solve time in $log")
    relative_error = parse(Float64, fields[14])
    tolerance = expected[4] == "Float32" ? Float64(Float32(1e-5)) : 1e-8
    isfinite(relative_error) && 0 <= relative_error <= tolerance ||
        error("Sample failed validation in $log")
    iterations = parse(Int, fields[8])
    iterations >= 0 || error("Invalid iteration count in $log")
    return (; elapsed=only(values), relative_error, iterations)
end

function run_samples(log, backend, solver, mode, n; worker=joinpath(@__DIR__, "krylov.jl"))
    count = parse(Int, get(ENV, "BENCH_SAMPLES", "5"))
    count >= 2 || error("BENCH_SAMPLES must be at least 2 for a standard error")
    label = backend == "CuArray" ? "CUDA" :
            backend == "cuNumeric" && mode != "stock" ? "cuNumeric $mode" : backend
    expected = [label, solver, mode, get(ENV, "BENCH_ELTYPE", "Float32"),
                get(ENV, "BENCH_GPUS", "1"), string(n)]
    project = dirname(Base.active_project())
    threads = get(ENV, "BENCH_THREADS", "4")
    cmd = addenv(`$(Base.julia_cmd()) -t$threads --startup-file=no --project=$project $worker $backend $solver $mode $n`,
                 "BENCH_SAMPLES" => "1")
    results = collect_samples(log, cmd, count) do sample_log
        sample_result(sample_log, expected)
    end
    # Preserve the existing CSV schema; per-sample counts remain in worker logs.
    iterations = maximum(result.iterations for result in results)
    return print_result([expected; string(iterations)], results)
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 5 || error("Usage: julia run_samples.jl CASE_LOG BACKEND SOLVER MODE N")
    KrylovSamples.run_samples(ARGS[1], ARGS[2], ARGS[3], ARGS[4], parse(Int, ARGS[5]))
end
