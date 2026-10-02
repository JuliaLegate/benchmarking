Base.@kwdef struct CUDAGrayScottPlain{T} <: AbstractGrayScott{T}
    N::Int
    M::Int
end

name(::CUDAGrayScottPlain) = "grayscott_plain"
register_benchmark("grayscott_plain", CUDAGrayScottPlain)

Base.@kwdef struct CUDAGrayScott{T} <: AbstractGrayScott{T}
    N::Int
    M::Int
end

name(::CUDAGrayScott) = "grayscott"
register_benchmark("grayscott", CUDAGrayScott)

correctness_uses_cpu(::Union{CUDAGrayScottPlain,CUDAGrayScott}) = true

let body = deepcopy(GRAYSCOTT_STEP_BODY)
    @eval _gs_step!(::Union{CUDAGrayScott,CUDAGrayScottPlain}, u, v, u_new, v_new, args::GSParams) = $body
end
allowed_types(::Type{<:CUDAGrayScottPlain}) = Union{Float32,Float64}
allowed_types(::Type{<:CUDAGrayScott}) = Union{Float32,Float64}

benchmark_array_module(::Type{<:Union{CUDAGrayScottPlain,CUDAGrayScott}}) = CUDA
