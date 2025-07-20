using CUDA
using LinearAlgebra
using BenchmarkTools
using Printf


function cuda_broadcasting(u, v, f::Float32, k::Float32)

    @views F_u = (
        (
            -u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) + f*(1.0f0 .- u[2:(end - 1), 2:(end - 1)])
    )

    @views F_v = (
        (
            u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) - (f+k)*v[2:(end - 1), 2:(end - 1)]
    )

    return F_u, F_v
end


gpus = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
n_samples = parse(Int, ARGS[3])
use_fused = parse(Bool, ARGS[4])
warmup=1