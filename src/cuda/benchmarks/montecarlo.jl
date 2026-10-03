Base.@kwdef struct CUDAMonteCarlo{T} <: AbstractMonteCarloIntegration{T}
    n_samples::Int
end

name(::CUDAMonteCarlo) = "montecarlo"
register_benchmark("montecarlo", CUDAMonteCarlo)

allowed_types(::Type{<:CUDAMonteCarlo}) = Union{Float32,Float64}

if isdefined(@__MODULE__, :CUDA)
    run!(mci::CUDAMonteCarlo, x::CUDA.CuArray) = _montecarlo_mapreduce(mci, x)
end

function benchmark_backend_label(
    ::CUDAMonteCarlo, backend::String, default::String
)
    return backend == "cudajl" ? "CUDA.jl (mapreduce)" : default
end

benchmark_array_module(::Type{<:CUDAMonteCarlo}) = CUDA
