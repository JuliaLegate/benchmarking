Base.@kwdef struct GEMM{T} <: AbstractGEMM{T}
    N::Int
    M::Int
end

name(::GEMM) = "gemm"
register_benchmark("gemm", GEMM)

function allowed_types(::Type{GEMM})
    return Union{cuNumeric.SUPPORTED_FLOAT_TYPES,cuNumeric.SUPPORTED_INT_TYPES}
end

benchmark_array_module(::Type{<:GEMM}) = cuNumeric
