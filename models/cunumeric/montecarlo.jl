using cuNumeric
using Legate
using LinearAlgebra
using Printf

get_time_us() = Legate.value(Legate.time_microseconds())

function initialize_cunumeric(N)
    A = cuNumeric.as_type(cuNumeric.rand(NDArray, N), Float32)
    GC.gc() # remove the intermediate FP64 arrays
    return A
end

function total_flops(N, M)
    return N 
end

function integrand(x)
    return exp(square(x))
end

function mc_integration_cunumeric(N, M, n_samples, n_warmup)
    A = initialize_cunumeric(N)

    start_time = nothing
    for idx in range(1, n_samples + n_warmup)
        if idx == n_warmup + 1
            start_time = get_time_us()
        end

        res = mean(integrand(A))
    end
    total_time_μs = get_time_us() - start_time
    mean_time_ms = total_time_μs / (n_samples * 1e3)
    gflops = total_flops(N, M) / (mean_time_ms * 1e6) # GFLOP is 1e9

    return mean_time_ms, gflops
end

gpus = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
n_samples = parse(Int, ARGS[3])
n_warmup = 2

println("[cuNumeric.jl]  Monte-Carlo Integration benchmark on $(N) elements for $(n_samples) iterations, $(n_warmup) warmups")

mean_time_ms, gflops = gemm_cunumeric(N, M, n_samples, n_warmup)

println("[cuNumeric.jl]  Mean Run Time: $(mean_time_ms) ms")
println("[cuNumeric.jl]  FLOPS: $(gflops) GFLOPS")

open("./montecarlo/mc.csv", "a") do io
    @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "cuNumeric.jl", gpus, N, M, mean_time_ms, gflops)
end
