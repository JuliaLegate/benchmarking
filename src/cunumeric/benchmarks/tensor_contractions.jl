Base.@kwdef struct TensorProjection3{T} <: AbstractTensorProjection3{T}
    N::Int
end

name(::TensorProjection3) = "tensor_projection3"
register_benchmark("tensor_projection3", TensorProjection3)

Base.@kwdef struct TensorContract4{T} <: AbstractTensorContract4{T}
    N::Int
end

name(::TensorContract4) = "tensor_contract4"
register_benchmark("tensor_contract4", TensorContract4)

allowed_types(::Type{<:TensorProjection3}) = cuNumeric.SUPPORTED_FLOAT_TYPES
allowed_types(::Type{<:TensorContract4}) = cuNumeric.SUPPORTED_FLOAT_TYPES

benchmark_array_module(::Type{<:Union{TensorProjection3,TensorContract4}}) = cuNumeric
