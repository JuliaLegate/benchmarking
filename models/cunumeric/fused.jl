using cuNumeric
using Legate
using CUDA
import CUDA: i32
using Printf

get_time_us() = Legate.value(Legate.time_microseconds())


#* THIS WILL ALLOCATE FOR F_u and F_v
#* THE FUSED KERNEL WILL NOT HAVE THIS OVERHEAD
#* I THINK THATS STILL FAIR? YOU KINDA OF HAVE TO DO IT THIS WAY
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

    if i <= N - 1 # index from 2 --> end - 1
        u_i = u[i + 1, i + 1] 
        v_i = v[i + 1, i + 1]
        v_i_sq = v_i * v_i
        F_u[i] = -u_i + v_i_sq + f*(1.0f0 - u_i)
        F_v[i] = u_i + v_i_sq + f*(1.0f0 - u_i) - (f + k)*v_i
    end

    return nothing
end

N = 1024 #* probably make this bigger
threads = 256
blocks = cld(N, threads)

u = cuNumeric.random(Float32, (N, N))
v = cuNumeric.random(Float32, (N, N))
F_u = cuNumeric.zeros(Float32, (N-2, N-2))
F_v = cuNumeric.zeros(Float32, (N-2, N-2))

f = 0.03f0
k = 0.06f0

task = cuNumeric.@cuda_task kernel_add(u, v, F_u, F_v, UInt32(N), f, k)

#* not sure what to do with scalars argument
cuNumeric.@launch task=task threads=threads blocks=blocks inputs=(u, v) outputs=(F_u, F_v) scalars=UInt32(N)

Fu_cpu = F_u[:]
Fv_cpu = F_v[:]
println("Result of F_u after kenel launch: ", Fu_cpu[1:100])
println("Result of F_v after kenel launch: ", Fv_cpu[1:100])