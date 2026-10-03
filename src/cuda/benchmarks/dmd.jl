Base.@kwdef struct CUDADMD{T} <: AbstractDMD{T}
    N::Int
    M::Int
end

name(::CUDADMD) = "dmd_baseline"
register_benchmark("dmd_baseline", CUDADMD)


let body = deepcopy(DMD_PROJECT_BODY)
    @eval _dmd_project(::CUDADMD, X, X2, U, Vt, S) = $body
end
allowed_types(::Type{<:CUDADMD}) = Union{Float32,Float64}

benchmark_array_module(::Type{<:CUDADMD}) = CUDA
