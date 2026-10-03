Base.@kwdef struct CUDAGEMM{T} <: AbstractGEMM{T}
    N::Int
    M::Int
end

name(::CUDAGEMM) = "gemm"
register_benchmark("gemm", CUDAGEMM)

allowed_types(::Type{<:CUDAGEMM}) = Union{Float32,Float64}

benchmark_array_module(::Type{<:CUDAGEMM}) = CUDA
