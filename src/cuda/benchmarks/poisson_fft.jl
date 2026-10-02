Base.@kwdef struct CUDAPoissonFFT{T} <: AbstractPoissonFFT{T}
    N::Int
    M::Int
end

name(::CUDAPoissonFFT) = "poisson_fft"
register_benchmark("poisson_fft", CUDAPoissonFFT)

allowed_types(::Type{<:CUDAPoissonFFT}) = Union{Float32,Float64}

benchmark_array_module(::Type{<:CUDAPoissonFFT}) = CUDA
