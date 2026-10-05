# Keep the existing config/result keys while naming types by their algorithm.
Base.@kwdef struct MonteCarloMapReduce{T} <: AbstractMonteCarloIntegration{T}
    n_samples::Int
end

name(::MonteCarloMapReduce) = "montecarlo"
register_benchmark("montecarlo", MonteCarloMapReduce)

function run!(mci::MonteCarloMapReduce{T}, x) where {T}
    total = mapreduce(_montecarlo_scalar_integrand, +, x; init=zero(T))
    return _montecarlo_weight(mci) * total
end

Base.@kwdef struct MonteCarloBroadcast{T} <: AbstractMonteCarloIntegration{T}
    n_samples::Int
end

name(::MonteCarloBroadcast) = "montecarlo_naive"
register_benchmark("montecarlo_naive", MonteCarloBroadcast)

function run!(mci::MonteCarloBroadcast, x)
    # Dot the negation too so the integrand materializes as one broadcast.
    integrand = exp.(.-(x .^ 2))
    return _montecarlo_weight(mci) * sum(integrand)
end

function benchmark_backend_label(
    ::MonteCarloMapReduce, backend::String, default::String
)
    return backend == "cunumeric" ? "cuNumeric (mapreduce)" : default
end

benchmark_array_module(::Type{<:Union{MonteCarloMapReduce,MonteCarloBroadcast}}) = cuNumeric
