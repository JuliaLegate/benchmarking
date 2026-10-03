# NAS benchmark types are available to planning without loading a GPU runtime.

Base.@kwdef struct NASEmbarrassinglyParallel{T} <: AbstractNASEP{T}
    N::Int
    M::Int
    class::String = "S"
end

name(::NASEmbarrassinglyParallel) = "nas_ep"
register_benchmark("nas_ep", NASEmbarrassinglyParallel)

Base.@kwdef struct NASFourierTransform{T} <: AbstractNASFT{T}
    N::Int
    M::Int
    class::String = "S"
end

name(::NASFourierTransform) = "nas_ft"
register_benchmark("nas_ft", NASFourierTransform)

Base.@kwdef struct NASMultiGrid{T} <: AbstractNASMG{T}
    N::Int
    M::Int
    class::String = "S"
    implementation::String = "default"
end

name(::NASMultiGrid) = "nas_mg"
register_benchmark("nas_mg", NASMultiGrid)

benchmark_array_module(::Type{<:Union{NASEmbarrassinglyParallel,NASFourierTransform,NASMultiGrid}}) = cuNumeric
