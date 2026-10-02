Base.@kwdef struct GrayScottBaseline{T} <: AbstractGrayScott{T}
    N::Int
    M::Int
end

name(::GrayScottBaseline) = "grayscott_plain"
register_benchmark("grayscott_plain", GrayScottBaseline)

Base.@kwdef struct GrayScottAccelerated{T} <: AbstractGrayScott{T}
    N::Int
    M::Int
end

name(::GrayScottAccelerated) = "grayscott"
register_benchmark("grayscott", GrayScottAccelerated)

function cuda_runnable(b::GrayScottAccelerated{T}) where {T}
    return GrayScottBaseline{T}(; N=b.N, M=b.M)
end
correctness_uses_cpu(::Union{GrayScottBaseline,GrayScottAccelerated}) = true
# cuNumeric replaces this plain fallback in its worker.
let body = deepcopy(GRAYSCOTT_STEP_BODY)
    @eval _gs_step!(b::GrayScottBaseline, u, v, u_new, v_new, args::GSParams) = $body
    if !CUNUMERIC_BENCH_RUNTIME
        @eval _gs_step!(b::GrayScottAccelerated, u, v, u_new, v_new, args::GSParams) = $body
    end
end

allowed_types(::Type{<:GrayScottBaseline}) = cuNumeric.SUPPORTED_FLOAT_TYPES
allowed_types(::Type{<:GrayScottAccelerated}) = cuNumeric.SUPPORTED_FLOAT_TYPES

benchmark_array_module(::Type{<:Union{GrayScottBaseline,GrayScottAccelerated}}) = cuNumeric
