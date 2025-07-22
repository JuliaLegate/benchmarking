using CUDA
import CUDA: i32
using Printf


function total_flops(N, T)
    return N*T # missing pre-factor
end

function unfused_cuda(u, v, f::Float32, k::Float32)
    @views F_u = (
        (
            -u[2:(end - 1)] .*
            (v[2:(end - 1)] .* v[2:(end - 1)])
        ) + f*(1.0f0 .- u[2:(end - 1)])
    )

    @views F_v = (
        (
            u[2:(end - 1)] .*
            (v[2:(end - 1)] .* v[2:(end - 1)])
        ) - (f+k)*v[2:(end - 1)]
    )
    return F_u, F_v
end

function fused_kernel(u, v, F_u, F_v, N::UInt32, f::Float32, k::Float32)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if i <= (N-2)
        @inbounds begin
            u_ij = u[i + 1]
            v_ij = v[i + 1]
            v_sq = v_ij * v_ij
            F_u[i] = (-u_ij * v_sq) + f*(1.0f0 - u_ij)
            F_v[i] = (u_ij * v_sq) - (f + k)*v_ij
        end
    end

    return nothing
end


function run_fused(N, threads, n_samples, n_warmup)
    blocks = cld(N, threads)

    u = CUDA.rand(Float32, N)
    v = CUDA.rand(Float32, N)

    F_u = CUDA.zeros(Float32, (N-2))
    F_v = CUDA.zeros(Float32, (N-2))

    f = 0.03f0
    k = 0.06f0


    for i in range(1, n_warmup)
        @cuda threads=threads blocks=blocks fused_kernel(u, v, F_u, F_v, UInt32(N), f, k)
    end

    t = CUDA.@elapsed begin
        for i in range(1, n_samples)
            @cuda threads=threads blocks=blocks fused_kernel(u, v, F_u, F_v, UInt32(N), f, k)
        end
    end

    total_time_μs = t * 1e6
    mean_time_ms = total_time_μs / (n_samples * 1e3)
    gflops = total_flops(N, n_samples) / (mean_time_ms * 1e6) # GFLOP is 1e9

    return mean_time_ms, gflops
end


function run_unfused(N, n_samples, n_warmup)
    u = CUDA.rand(Float32, N)
    v = CUDA.rand(Float32, N)

    f = 0.03f0
    k = 0.06f0

    for i in range(1, n_warmup)
        F_u, F_v = unfused_cuda(u, v, f, k)
    end

    t = CUDA.@elapsed begin
        for i in range(1, n_samples)
            F_u, F_v = unfused_cuda(u, v, f, k)
        end
    end

    total_time_μs = t * 1e6
    mean_time_ms = total_time_μs / (n_samples * 1e3)
    gflops = total_flops(N, n_samples) / (mean_time_ms * 1e6) # GFLOP is 1e9

    return mean_time_ms, gflops
end

use_fused = false

gpus = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
n_samples = parse(Int, ARGS[3])
use_fused = ARGS[4] == "1"
warmup=1


threads = 256

if use_fused == true
    println("[CUDA] Fused kernel benchmark on $(N) elements for $(n_samples) iterations")
    mean_time_ms, gflops = run_fused(N, threads, n_samples, warmup)
    println("[CUDA] Mean Run Time: $(mean_time_ms) ms")
    println("[CUDA] FLOPS: $(gflops) GFLOPS")

    open("./benchmarks/fusion/fusion.csv", "a") do io
        @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "CUDA.jl (fused)", gpus, N, N, mean_time_ms, gflops)
    end
else
    println("[CUDA] Unfused kernel benchmark on $(N) elements for $(n_samples) iterations")
    mean_time_ms, gflops = run_unfused(N, n_samples, warmup)

    println("[CUDA] Mean Run Time: $(mean_time_ms) ms")
    println("[CUDA] FLOPS: $(gflops) GFLOPS")

    open("./benchmarks/fusion/fusion.csv", "a") do io
        @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "CUDA.jl (unfused)", gpus, N, N, mean_time_ms, gflops)
    end
end
