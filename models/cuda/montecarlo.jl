using CUDA
using Statistics

function total_flops(N)
    return N # missing prefactor
end

function integrand(x)
    return exp(x^2)
end

function monte_carlo_integration(N, n_samples, n_warmup)

    x = (5.0f0 .* CUDA.rand(Float32, N)) .- 10.0f0

    start_time = nothing
    for s in 1:n_warmup
        y = mean(integrand.(x))
    end

    t = CUDA.@elapsed begin
        for s in 1:n_samples
            y = mean(integrand.(x))
        end
    end

    total_time_μs = t * 1e6
    mean_time_ms = total_time_μs / (n_samples * 1e3)
    gflops = total_flops(N) / (mean_time_ms * 1e6) # GFLOP is 1e9
  
    return mean_time_ms, gflops

end

N = parse(Int, ARGS[1])
n_samples = parse(Int, ARGS[1])
n_warmup = parse(Int, ARGS[2])

println("[cuNumeric]  Monte-Carlo integration benchmark on $(N) elements for $(n_samples) iterations, $(n_warmup) warmups")

mean_time_ms, gflops = gemm_cunumeric(N, n_samples, n_warmup)

println("[cuNumeric]  Mean Run Time: $(mean_time_ms) ms")
println("[cuNumeric]  FLOPS: $(gflops) GFLOPS")

open("./gemm/gemm.csv", "a") do io
    @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "cunumeric", gpus, N, M, mean_time_ms, gflops)
end
