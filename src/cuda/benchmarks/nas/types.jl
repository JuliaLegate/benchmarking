# NAS benchmark types are available to planning without loading a GPU runtime.

Base.@kwdef struct CUDANASEP{T} <: AbstractNASEP{T}
    N::Int
    M::Int
    class::String = "S"
end

name(::CUDANASEP) = "nas_ep"
register_benchmark("nas_ep", CUDANASEP)

Base.@kwdef struct CUDANASFT{T} <: AbstractNASFT{T}
    N::Int
    M::Int
    class::String = "S"
end

name(::CUDANASFT) = "nas_ft"
register_benchmark("nas_ft", CUDANASFT)

Base.@kwdef struct CUDANASMG{T} <: AbstractNASMG{T}
    N::Int
    M::Int
    class::String = "S"
    implementation::String = "default"
end

name(::CUDANASMG) = "nas_mg"
register_benchmark("nas_mg", CUDANASMG)

benchmark_array_module(::Type{<:Union{CUDANASEP,CUDANASFT,CUDANASMG}}) = CUDA
