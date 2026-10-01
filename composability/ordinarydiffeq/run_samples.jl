# CPU-only coordinator: each child owns its GPU runtime for one warmup and
# exactly one timed solve. Wait for it to exit before launching the next child.
module ODESamples

include("../process_samples.jl")
using .ProcessSamples: collect_samples, print_result

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
    results = collect_samples(log, cmd, count) do sample_log
        elapsed, relative_error = sample_result(sample_log, expected)
        return (; elapsed, relative_error)
    end
    return print_result(expected, results)
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 3 || error("Usage: julia run_samples.jl CASE_LOG BACKEND N")
    ODESamples.run_samples(ARGS[1], ARGS[2], parse(Int, ARGS[3]))
end
