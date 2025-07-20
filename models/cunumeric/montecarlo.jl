using cuNumeric
using Legate
using LinearAlgebra
using Printf
using Statistics

get_time_us() = Legate.value(Legate.time_microseconds())

function initialize_cunumeric(N)
    A = (10.0f0 * cuNumeric.as_type(cuNumeric.rand(NDArray, N), Float32)) - 5.0f0
    GC.gc() # remove the intermediate FP64 arrays
    return A
end

function total_flops(N)
    return N 
end

function integrand(x)
    return exp(-square(x))
end

function do_work(x, N)
    return (10.0f0/N) * sum(integrand(x))
end

function mc_integration_cunumeric(N, n_samples, n_warmup)
    A = initialize_cunumeric(N)

    for idx in range(1,n_warmup)
        do_work(A, N)
        GC.gc()
    end

    times = []
    for idx in range(1, n_samples)
        GC.gc()
        t0 = get_time_us() # avoid timing GC as its not part of the algo
        res = do_work(A, N)
        t1 = get_time_us()
        push!(times, t1 - t0)
    end
    mean_time_ms = mean(times) / 1e3
    gflops = total_flops(N) / (mean_time_ms * 1e6) # GFLOP is 1e9

    return mean_time_ms, gflops
end

gpus = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
n_samples = parse(Int, ARGS[3])
n_warmup = 2

println("[cuNumeric.jl]  Monte-Carlo Integration benchmark on $(N) elements for $(n_samples) iterations, $(n_warmup) warmups")

mean_time_ms, gflops = mc_integration_cunumeric(N, n_samples, n_warmup)

println("[cuNumeric.jl]  Mean Run Time: $(mean_time_ms) ms")
println("[cuNumeric.jl]  FLOPS: $(gflops) GFLOPS")

open("./benchmarks/montecarlo/montecarlo.csv", "a") do io
    @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "cuNumeric.jl", gpus, N, 1, mean_time_ms, gflops)
end
