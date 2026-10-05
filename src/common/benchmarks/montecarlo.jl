# Shared problem definition and setup only. Each concrete backend variant must
# define run! explicitly; the family does not choose a benchmark algorithm.
abstract type AbstractMonteCarloIntegration{T} <: AbstractBenchmark{T} end

dims(mci::AbstractMonteCarloIntegration) = (mci.n_samples, 1)
function data(mci::AbstractMonteCarloIntegration{T}) where {T}
    return "Monte Carlo Integration with T=$(T), n_samples=$(mci.n_samples)"
end

total_flops(s::AbstractMonteCarloIntegration) = s.n_samples
# Reserve one sample array plus one array-sized reduction/broadcast workspace.
# Model-specific memory accounting refines this conservative shared bound.
total_space(s::AbstractMonteCarloIntegration{T}) where {T} = 2 * s.n_samples * sizeof(T)

function estimate_scaling(s::AbstractMonteCarloIntegration, P::Integer)
    P == 1 && return dims(s)
    return (s.n_samples * P, 1)
end

function fit_one_gpu(
    ::Type{B}, ::Type{T};
    budget::Int, N_hint=nothing, M_hint=nothing,
) where {B<:AbstractMonteCarloIntegration,T}
    hi = max(8, Int(fld(budget, sizeof(T))))
    n = largest_feasible(8, hi, k -> total_space(B{T}(; n_samples=k)) <= budget)
    n === nothing && error("montecarlo does not fit in $(budget) bytes")
    return (align8(n), 1)
end

function initialize(mci::AbstractMonteCarloIntegration{T}; mod=benchmark_array_module(typeof(mci))) where {T}
    # Uniform samples over the integration domain [0, 10].
    x = T(10) .* rand_array(mod, T, mci.n_samples)
    GC.gc()
    return (x,)
end

_montecarlo_weight(mci::AbstractMonteCarloIntegration{T}) where {T} = T(10) / mci.n_samples
@inline _montecarlo_scalar_integrand(x) = exp(-(x*x))

# n_samples comes in as N; M is unused.
function build_benchmark(
    ::Type{B}, ::Type{T}, N, M; kwargs...
) where {B<:AbstractMonteCarloIntegration,T}
    return B{T}(; kwargs..., n_samples=N)
end

function correctness_problem(b::B) where {T,B<:AbstractMonteCarloIntegration{T}}
    return B(; n_samples=min(b.n_samples, 1024))
end
function correctness_seed(b::AbstractMonteCarloIntegration{T}) where {T}
    return (T.(range(T(0), T(10); length=b.n_samples)),)
end
correctness_uses_cpu(::AbstractMonteCarloIntegration) = true

# Independent CPU oracle used only by the correctness checker, never for timing.
function run_reference!(mci::AbstractMonteCarloIntegration{T}, x::Array) where {T}
    total = zero(T)
    for sample in x
        total += exp(-sample^2)
    end
    return _montecarlo_weight(mci) * total
end
