using cuNumeric
using Legate
using LinearAlgebra
using Printf

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

function mc_integration_cunumeric(N, n_samples, n_warmup)
    A = initialize_cunumeric(N)

    iter_start = nothing
    for idx in range(1, n_samples + n_warmup)
        if idx > n_warmup
            iter_start = get_time_us()
        end
        res = (10.0f0/N) * sum(integrand(A))

        if idx > n_warmup
            total_time_μs += get_time_us() - iter_start
        end

        GC.gc() # We are not timing the GC.gc() time here
    end
    mean_time_ms = total_time_μs / (n_samples * 1e3)
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

open("./montecarlo/mc.csv", "a") do io
    @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "cuNumeric.jl", gpus, N, 1, mean_time_ms, gflops)
end
