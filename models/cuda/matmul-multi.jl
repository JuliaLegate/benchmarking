#######################################################################
# This file contains a multi-GPU GEMM. 
# C = A * B
# Each GPU will get the entirity of B and subset of the rows of a
# This will allow for localised computation of C without communication
# You must be able to fit 
#      [(N / GPUS) * M] + (M * N) + [(N^2 / GPUS)] elements on each GPU
#######################################################################

using CUDA
using LinearAlgebra
using BenchmarkTools

function total_flops(N, M)
    return N * N * ((2*M) - 1)
end

function total_space(N, M)
    return 2 * (N*M) * sizeof(Float32) + (N*N) * sizeof(Float32)
end

function synchronize_all(devices)
    for dev in devices
        CUDA.device!(dev)
        synchronize()
    end
end

function multi_gpu_matmul(gpus, N, M, n_samples, n_warmup)
    all_devices = collect(CUDA.devices())
    gpus = min(gpus, length(all_devices))
    devices = all_devices[1:gpus]

    n_devices = length(devices)

    rows_per_gpu = [div(N, n_devices) + (i <= rem(N, n_devices) ? 1 : 0) for i in 1:n_devices]

    A = Vector{CuArray{Float32, 2}}(undef, n_devices)
    C = Vector{CuArray{Float32, 2}}(undef, n_devices)
    B = Vector{CuArray{Float32, 2}}(undef, n_devices)

    B_host = CUDA.rand(Float32, M, N)

    for i in 1:n_devices
        CUDA.device!(devices[i])
        A[i] = CUDA.rand(Float32, rows_per_gpu[i], M)
        C[i] = CUDA.zeros(Float32, rows_per_gpu[i], N)
        B[i] = CuArray(B_host)  # replicate B
    end

    # Warmup
    for s in 1:n_warmup
        for i in 1:n_devices
            CUDA.device!(devices[i])
            mul!(C[i], A[i], B[i])
            synchronize()
        end
    end

    t = CUDA.@elapsed begin
        for s in 1:n_samples
            for i in 1:n_devices
                CUDA.device!(devices[i])
                mul!(C[i], A_parts[i], B[i])
            end
            synchronize_all(CUDA.devices())
        end
    end
    
    total_time_μs = t * 1e6
    mean_time_ms = total_time_μs / (n_samples * 1e3)
    gflops = total_flops(N, M) / (mean_time_ms * 1e6) # GFLOP is 1e9
    return mean_time_ms, gflops
end