abstract type AbstractGEMM{T} <: AbstractBenchmark{T} end
# Interface: `name`, `dims`, `total_flops`, `total_space`, `estimate_scaling`,
# `fit_one_gpu`, `initialize`, `run!`. Peak-byte and P-scaling formulas live
# here so the orchestrator can dispatch without a name switch.

dims(g::AbstractGEMM) = (g.N, g.M)
data(g::AbstractGEMM{T}) where {T} = "GEMM with T=$(T), N=$(g.N), M=$(g.M)"

total_flops(s::AbstractGEMM) = s.N * s.N * ((2*s.M) - 1)

# Live arrays for `mul!(C, A, B)`: A (N×M), B (M×N), C (N×N).
total_space(s::AbstractGEMM{T}) where {T} = (2 * s.N * s.M + s.N * s.N) * sizeof(T)

function estimate_scaling(s::AbstractGEMM, P::Integer)
    P == 1 && return (s.N, s.M)
    return (scale_axis(s.N, P, 1//3), scale_axis(s.M, P, 1//3))
end

function fit_one_gpu(
    ::Type{B}, ::Type{T};
    budget::Int, N_hint=nothing, M_hint=nothing,
) where {B<:AbstractGEMM,T}
    hi = max(8, Int(floor(sqrt(Float64(budget) / sizeof(T)))))
    n = largest_feasible(8, hi, k -> total_space(B{T}(; N=k, M=k)) <= budget)
    n === nothing && error("gemm does not fit in $(budget) bytes")
    n = align8(n)
    return (n, n)
end

function initialize(s::AbstractGEMM{T}; mod=benchmark_array_module(typeof(s))) where {T}
    A = rand_array(mod, T, s.N, s.M)
    B = rand_array(mod, T, s.M, s.N)
    C = zeros_array(mod, T, s.N, s.N)
    GC.gc()
    return C, A, B
end

run!(::AbstractGEMM, C, A, B) = mul!(C, A, B)

correctness_problem(b::AbstractGEMM) = typeof(b)(; N=min(b.N, 8), M=min(b.M, 8))
function correctness_seed(b::AbstractGEMM{T}) where {T}
    A = reshape(T.(1:(b.N * b.M)), b.N, b.M) ./ T(b.N*b.M)
    B = reshape(T.(1:(b.M * b.N)), b.M, b.N) ./ T(b.M*b.N)
    return zeros(T, b.N, b.N), A, B
end
correctness_uses_cpu(::AbstractGEMM) = true
