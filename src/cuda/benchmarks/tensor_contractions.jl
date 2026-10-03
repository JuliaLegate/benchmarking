Base.@kwdef struct CUDATensorProjection3{T} <: AbstractTensorProjection3{T}
    N::Int
end

name(::CUDATensorProjection3) = "tensor_projection3"
register_benchmark("tensor_projection3", CUDATensorProjection3)

Base.@kwdef struct CUDATensorContract4{T} <: AbstractTensorContract4{T}
    N::Int
end

name(::CUDATensorContract4) = "tensor_contract4"
register_benchmark("tensor_contract4", CUDATensorContract4)

allowed_types(::Type{<:CUDATensorProjection3}) = Union{Float32,Float64}
allowed_types(::Type{<:CUDATensorContract4}) = Union{Float32,Float64}

benchmark_array_module(::Type{<:Union{CUDATensorProjection3,CUDATensorContract4}}) = CUDA

function benchmark_backend_label(
    ::Union{CUDATensorProjection3,CUDATensorContract4}, backend::String, default::String
)
    return backend == "cudajl" ? "TensorOperations.jl / cuTENSOR" : default
end

function benchmark_backend_save_as(
    ::Union{CUDATensorProjection3,CUDATensorContract4}, backend::String, default::String
)
    return backend == "cudajl" ? "tensoroperations_cuda" : default
end
