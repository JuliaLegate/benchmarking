using TensorOperations

abstract type AbstractTensorContraction{T} <: AbstractBenchmark{T} end
abstract type AbstractTensorProjection3{T} <: AbstractTensorContraction{T} end
abstract type AbstractTensorContract4{T} <: AbstractTensorContraction{T} end

dims(b::AbstractTensorContraction) = (b.N, 1)
function data(b::AbstractTensorContraction{T}) where {T}
    return "$(name(b)) with T=$(T), N=$(b.N)"
end

# Three rank-3 outputs, each containing an N-term dot product.
total_flops(b::AbstractTensorProjection3) = 3 * b.N^3 * (2 * b.N - 1)

# N^4 output elements, each containing an N^2-term dot product.
total_flops(b::AbstractTensorContract4) = b.N^4 * (2 * b.N^2 - 1)

# opt=true pairwise: T1[n,j,k] and T2[n,m,k] are both N³, plus A, D, B.
total_space(b::AbstractTensorProjection3{T}) where {T} = (4 * b.N^3 + b.N^2) * sizeof(T)

# opt=true is a (N²×N²)×(N²×N²) GEMM: X, Y, C plus one workspace.
total_space(b::AbstractTensorContract4{T}) where {T} = 4 * b.N^4 * sizeof(T)

function estimate_scaling(b::AbstractTensorProjection3, P::Integer)
    P == 1 && return dims(b)
    return (align2(floor(Int, b.N * Float64(P)^(1 / 4))), 1)
end

function estimate_scaling(b::AbstractTensorContract4, P::Integer)
    P == 1 && return dims(b)
    return (align2(floor(Int, b.N * Float64(P)^(1 / 6))), 1)
end

function fit_one_gpu(
    ::Type{B}, ::Type{T};
    budget::Int, N_hint=nothing, M_hint=nothing,
) where {B<:AbstractTensorProjection3,T}
    hi = max(4, Int(floor((Float64(budget) / (4 * sizeof(T)))^(1 / 3))))
    n = largest_feasible(4, hi, k -> total_space(B{T}(; N=k)) <= budget)
    n === nothing && error("tensor_projection3 does not fit in $(budget) bytes")
    return (n, 1)
end

function fit_one_gpu(
    ::Type{B}, ::Type{T};
    budget::Int, N_hint=nothing, M_hint=nothing,
) where {B<:AbstractTensorContract4,T}
    hi = max(4, Int(floor((Float64(budget) / (4 * sizeof(T)))^(1 / 4))))
    n = largest_feasible(4, hi, k -> total_space(B{T}(; N=k)) <= budget)
    n === nothing && error("tensor_contract4 does not fit in $(budget) bytes")
    return (n, 1)
end

function build_benchmark(
    ::Type{B}, ::Type{T}, N, M; kwargs...
) where {B<:AbstractTensorProjection3,T}
    return B{T}(; kwargs..., N=N)
end

function build_benchmark(
    ::Type{B}, ::Type{T}, N, M; kwargs...
) where {B<:AbstractTensorContract4,T}
    return B{T}(; kwargs..., N=N)
end

function initialize(b::AbstractTensorProjection3{T}; mod=benchmark_array_module(typeof(b))) where {T}
    A = rand_array(mod, T, b.N, b.N, b.N)
    B = rand_array(mod, T, b.N, b.N)
    D = zeros_array(mod, T, b.N, b.N, b.N)
    GC.gc()
    return D, A, B
end

function initialize(b::AbstractTensorContract4{T}; mod=benchmark_array_module(typeof(b))) where {T}
    X = rand_array(mod, T, b.N, b.N, b.N, b.N)
    Y = rand_array(mod, T, b.N, b.N, b.N, b.N)
    C = zeros_array(mod, T, b.N, b.N, b.N, b.N)
    GC.gc()
    return C, X, Y
end

function run!(::AbstractTensorProjection3, D, A, B)
    @tensor opt=true D[n, m, l] =
        A[i, j, k] * B[n, i] * B[m, j] * B[l, k]
    return D
end

function run!(::AbstractTensorContract4, C, X, Y)
    @tensor opt=true C[a, b, c, d] = X[a, i, c, j] * Y[i, b, j, d]
    return C
end

correctness_problem(b::AbstractTensorProjection3) = typeof(b)(; N=min(b.N, 4))
correctness_problem(b::AbstractTensorContract4) = typeof(b)(; N=min(b.N, 4))
function correctness_atol_rtol(::AbstractTensorContraction, ::Type{T}) where {T}
    tol = T === Float32 ? 2.0f-4 : 1e-11
    return tol, tol
end
