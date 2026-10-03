Base.@kwdef struct CUDACGPlain{T} <: AbstractConjugateGradient{T}
    N::Int
    M::Int = 1
    check_every::Int = 10
    max_iter::Int = 1000
end

name(::CUDACGPlain) = "cg_plain"
register_benchmark("cg_plain", CUDACGPlain)

Base.@kwdef struct CUDACG{T} <: AbstractConjugateGradient{T}
    N::Int
    M::Int = 1
    check_every::Int = 10
    max_iter::Int = 1000
end

name(::CUDACG) = "cg"
register_benchmark("cg", CUDACG)

benchmark_array_module(::Type{<:Union{CUDACGPlain,CUDACG}}) = CUDA
