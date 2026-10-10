Base.@kwdef struct CUDAMonteCarlo{T} <: AbstractMonteCarloIntegration{T}
    n_samples::Int
end

name(::CUDAMonteCarlo) = "montecarlo"
register_benchmark("montecarlo", CUDAMonteCarlo)

function run!(mci::CUDAMonteCarlo{T}, x) where {T}
    total = mapreduce(_montecarlo_scalar_integrand, +, x; init=zero(T))
    return _montecarlo_weight(mci) * total
end

function benchmark_backend_label(
    ::CUDAMonteCarlo, backend::String, default::String
)
    return backend == "cudajl" ? "CUDA.jl (mapreduce)" : default
end

benchmark_array_module(::Type{<:CUDAMonteCarlo}) = CUDA
