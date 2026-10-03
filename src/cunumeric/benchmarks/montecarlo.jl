Base.@kwdef struct MonteCarloIntegration{T} <: AbstractMonteCarloIntegration{T}
    n_samples::Int
end

name(::MonteCarloIntegration) = "montecarlo"
register_benchmark("montecarlo", MonteCarloIntegration)

Base.@kwdef struct MonteCarloNaive{T} <: AbstractMonteCarloIntegration{T}
    n_samples::Int
end

name(::MonteCarloNaive) = "montecarlo_naive"
register_benchmark("montecarlo_naive", MonteCarloNaive)

allowed_types(::Type{<:MonteCarloIntegration}) = cuNumeric.SUPPORTED_FLOAT_TYPES
allowed_types(::Type{<:MonteCarloNaive}) = cuNumeric.SUPPORTED_FLOAT_TYPES

if CUNUMERIC_BENCH_RUNTIME
    run!(mci::MonteCarloIntegration, x::NDArray) = _montecarlo_mapreduce(mci, x)

    function benchmark_backend_label(
        ::MonteCarloIntegration, backend::String, default::String
    )
        return backend == "cunumeric" ? "cuNumeric (mapreduce)" : default
    end
end

benchmark_array_module(::Type{<:Union{MonteCarloIntegration,MonteCarloNaive}}) = cuNumeric
