using cuNumeric
using Legate
using CUDA
import CUDA: i32
using Printf

get_time_us() = Legate.value(Legate.time_microseconds())

function total_flops(N)
    return N*N # missing pre-factor
end

#* THIS WILL ALLOCATE FOR F_u and F_v
#* THE FUSED KERNEL WILL NOT HAVE THIS OVERHEAD
#* Also time creation of F_u and F_v??
function cuNumeric_unfused(u, v, f, k)

    F_u = (
        (
            -u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) + f*(1.0f0 .- u[2:(end - 1), 2:(end - 1)])
    )

    F_v = (
        (
            u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) - (f+k)*v[2:(end - 1), 2:(end - 1)]
    )

    return F_u, F_v
end


@inbounds function fused_kernel(u, v, F_u, F_v, N, f::Float32, k::Float32)
    
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1i32) * blockDim().y + threadIdx().y

    if i <= N - 1 && j <= N - 1 # index from 2 --> end - 1
        u_ij = u[i + 1, j + 1] 
        v_ij = v[i + 1, j + 1]
        v_sq = v_ij * v_ij
        F_u[i,j] = -u_ij + v_sq + f*(1.0f0 - u_ij)
        F_v[i,j] = u_ij + v_sq + f*(1.0f0 - u_ij) - (f + k)*v_ij
    end

    return nothing
end

function run_fused(N, threads, n_samples, n_warmup)
    
    blocks2d = (cld(N, threads), cld(N, threads))
    threads2d = (threads, threads)

    u = cuNumeric.random(Float32, (N, N))
    v = cuNumeric.random(Float32, (N, N))
    F_u = cuNumeric.zeros(Float32, (N-2, N-2))
    F_v = cuNumeric.zeros(Float32, (N-2, N-2))

    f = 0.03f0
    k = 0.06f0

    task = cuNumeric.@cuda_task fused_kernel(u, v, F_u, F_v, UInt32(N), f, k)

    for i in range(1, n_warmup)
        cuNumeric.@launch task=task threads=threads2d blocks=blocks2d inputs=(u, v) outputs=(F_u, F_v) scalars=(UInt32(N), f, k)
    end

    #* not sure what to do with scalars argument
    start_time = get_time_us()
    for i in range(1, n_samples)
        cuNumeric.@launch task=task threads=threads blocks=blocks inputs=(u, v) outputs=(F_u, F_v) scalars=(UInt32(N), f, k)
    end
    end_time = get_time_us()

    total_time_μs = end_time - start_time
    mean_time_ms = total_time_μs / (n_samples * 1e3)
    gflops = total_flops(N) / (mean_time_ms * 1e6)

    return mean_time_ms, gflops
end

function run_unfused(N, n_samples, n_warmup)
    gc_interval = 6
    u = cuNumeric.random(Float32, (N, N))
    v = cuNumeric.random(Float32, (N, N))

    f = 0.03f0
    k = 0.06f0

    for i in range(1, n_warmup)
        F_u, F_v = cuNumeric_unfused(u, v, f, k)
        if n % gc_interval
            GC.gc()
        end
    end

    #* not sure what to do with scalars argument
    start_time = get_time_us()
    for i in range(1, n_samples)
        F_u, F_v = cuNumeric_unfused(u, v, f, k)
        if n % gc_interval
            GC.gc()
        end
    end
    end_time = get_time_us()

    total_time_μs = end_time - start_time
    mean_time_ms = total_time_μs / (n_samples * 1e3)
    gflops = total_flops(N) / (mean_time_ms * 1e6)

    return mean_time_ms, gflops
end

gpus = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
n_samples = parse(Int, ARGS[3])
use_fused = parse(Bool, ARGS[4])
warmup=1


threads = 512

if use_fused
    println("[cuNumeric] Fused kernel benchmark on $(N)x$(N) matricies for $(n_samples) iterations")
    mean_time_ms, gflops = run_fused(N, threads, n_samples, warmup)
    println("[cuNumeric] Mean Run Time: $(mean_time_ms) ms")
    println("[cuNumeric] FLOPS: $(gflops) GFLOPS")

    open("./benchmarks/fusion/fusion.csv", "a") do io
        @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "Legate.jl + CUDA.jl (fused)", gpus, N, M, mean_time_ms, gflops)
    end
else
    println("[cuNumeric] Unfused kernel benchmark on $(N)x$(N) matricies for $(n_samples) iterations")
    mean_time_ms, gflops = run_unfused(N, n_samples, warmup)

    println("[cuNumeric] Mean Run Time: $(mean_time_ms) ms")
    println("[cuNumeric] FLOPS: $(gflops) GFLOPS")

    open("./benchmarks/fusion/fusion.csv", "a") do io
        @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "cuNumeric.jl (unfused)", gpus, N, M, mean_time_ms, gflops)
    end
end
