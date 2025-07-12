using CUDA
using LinearAlgebra
using BenchmarkTools
using Printf

include("matmul-multi.jl")

function initialize_CUDA(N, M)
    A = CUDA.rand(Float32, N, M)
    B = CUDA.rand(Float32, M, N)
    C = CUDA.zeros(Float32, N, N)
    return A, B, C
end

function total_flops(N, M)
    return N * N * ((2*M) - 1)
end

function total_space(N, M)
    return 2 * (N*M) * sizeof(Float32) + (N*N) * sizeof(Float32)
end

function gemm_cuda(N, M, n_samples, n_warmup)
    A,B,C = initialize_CUDA(N, M)

    start_time = nothing
    for s in 1:n_warmup
        mul!(C, A, B)
    end

    t = CUDA.@elapsed begin
        for s in 1:n_samples
            mul!(C, A, B)
        end
    end

    total_time_μs = t * 1e6
    mean_time_ms = total_time_μs / (n_samples * 1e3)
    gflops = total_flops(N, M) / (mean_time_ms * 1e6) # GFLOP is 1e9
  
    return mean_time_ms, gflops
end

gpus = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
M = parse(Int, ARGS[3])
n_samples = parse(Int, ARGS[4])
n_warmup = parse(Int, ARGS[5])

println("[CUDA] MATMUL benchmark on $(N)x$(M) matricies for $(n_samples) iterations, $(n_warmup) warmups")

if gpus == 1
    # performs a single GPU test
    mean_time_ms, gflops = gemm_cuda(N, M, n_samples, n_warmup)
else
    mean_time_ms, gflops = multi_gpu_matmul(gpus, N, M, n_samples, n_warmup)
end 

println("[CUDA] Mean Run Time: $(mean_time_ms) ms")
println("[CUDA] FLOPS: $(gflops) GFLOPS")

open("./gemm/gemm.csv", "a") do io
    @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "cuda", gpus, N, M, mean_time_ms, gflops)
end
