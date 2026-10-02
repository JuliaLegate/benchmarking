Base.@kwdef struct PoissonFFT{T} <: AbstractPoissonFFT{T}
    N::Int
    M::Int
end

name(::PoissonFFT) = "poisson_fft"
register_benchmark("poisson_fft", PoissonFFT)

if CUNUMERIC_BENCH_RUNTIME
    _batched_fft!(A::NDArray) = (cuNumeric.batched_fft!(A); A)
    _batched_ifft!(A::NDArray) = (cuNumeric.batched_ifft!(A); A)
end
allowed_types(::Type{PoissonFFT}) = cuNumeric.SUPPORTED_FLOAT_TYPES

benchmark_array_module(::Type{<:PoissonFFT}) = cuNumeric
