using CUDA
using LinearAlgebra
using BenchmarkTools


function initialize_CUDA(N)
    A = CUDA.rand(Float32, N, N)
    B = CUDA.rand(Float32, N, N)
    C = CUDA.zeros(Float32, N, N)
    return A, B, C
end

function total_flops(N)
    return N * N * ((2*N) - 1)
end

function total_space(N)
    return 3 * (N^2) * sizeof(Float32)
end

function gemm_cuda(N, n_samples, n_warmup)
    A,B,C = initialize_CUDA(N)
  
    start_time = nothing

    for idx in range(1, n_warmup)
        mul!(C, A, B)
    end

    t = CUDA.@elapsed begin
        for idx in range(n_warmup + 1, n_samples + n_warmup)

        mul!(C, A, B)
            
        end
    end

    total_time_μs = t * 1e6
    mean_time_ms = total_time_μs / (n_samples * 1e3)
    gflops = total_flops(N) / (mean_time_ms * 1e6) # GFLOP is 1e9
  
    return mean_time_ms, gflops
end
