Base.@kwdef struct DMDBaseline{T} <: AbstractDMD{T}
    N::Int
    M::Int
end

name(::DMDBaseline) = "dmd_baseline"
register_benchmark("dmd_baseline", DMDBaseline)

Base.@kwdef struct DMDAccelerated{T} <: AbstractDMD{T}
    N::Int
    M::Int
end

name(::DMDAccelerated) = "dmd_accelerated"
register_benchmark("dmd_accelerated", DMDAccelerated)

let body = deepcopy(DMD_PROJECT_BODY)
    @eval _dmd_project(::DMDBaseline, X, X2, U, Vt, S) = $body
    if CUNUMERIC_BENCH_ACCELERATE
        @eval @accelerate function _dmd_project(
            ::DMDAccelerated, X, X2, U, Vt, S
        )
            $body
        end
    end
end

function cuda_runnable(b::DMDAccelerated{T}) where {T}
    return DMDBaseline{T}(; N=b.N, M=b.M)
end
allowed_types(::Type{<:DMDBaseline}) = cuNumeric.SUPPORTED_FLOAT_TYPES
allowed_types(::Type{<:DMDAccelerated}) = cuNumeric.SUPPORTED_FLOAT_TYPES

if CUNUMERIC_BENCH_RUNTIME
    _dmd_T(A::NDArray) = cuNumeric.transpose(A)
    _dmd_row(v::NDArray) = cuNumeric.reshape(v, (1, length(v)))
end

benchmark_array_module(::Type{<:Union{DMDBaseline,DMDAccelerated}}) = cuNumeric
