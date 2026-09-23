# A transparent inverse problem: infer a gas concentration image from
# wavelength-integrated transmission measurements.
import Integrals
using Integrals: IntegralProblem, GaussLegendre
using FastGaussQuadrature: gausslegendre

struct AbsorptionModel{T,A}
    centers::Vector{T}
    quadrature::A
end

function AbsorptionModel(::Type{T}, bands::Int, order::Int) where {T<:AbstractFloat}
    bands >= 2 || error("At least two spectral bands are required")
    order > 0 || error("Quadrature order must be positive")
    nodes, weights = gausslegendre(order)
    centers = collect(range(T(0.2), T(0.8); length=bands))
    return AbsorptionModel(centers, GaussLegendre(T.(nodes), T.(weights)))
end

# One smooth absorption line and a Gaussian response for each detector band.
absorption(λ::T) where {T} = T(0.15) + T(1.1) * exp(-((λ - T(0.5)) / T(0.16))^2)
response(λ::T, center::T) where {T} = exp(-((λ - center) / T(0.12))^2)

function band_integral(concentration, center, model; derivative=false)
    T = eltype(concentration)
    function integrand(λ, _)
        k = absorption(λ)
        factor = response(λ, center) * (derivative ? -k : one(T))
        return factor .* exp.(-k .* concentration)
    end
    problem = IntegralProblem(integrand, (zero(T), one(T)))
    return Integrals.solve(problem, model.quadrature).u
end

function loss(concentration, observations, model)
    squared_error = nothing
    for (center, observed) in zip(model.centers, observations)
        residual = band_integral(concentration, center, model) .- observed
        term = residual .* residual
        squared_error = isnothing(squared_error) ? term : squared_error .+ term
    end
    # Optimization.jl needs one host scalar per objective evaluation.
    return Float64(sum(squared_error)) / (length(concentration) * length(observations))
end

function gradient!(G, concentration, observations, model)
    G .= zero(eltype(G))
    scale = eltype(G)(2 / (length(concentration) * length(observations)))
    for (center, observed) in zip(model.centers, observations)
        predicted = band_integral(concentration, center, model)
        derivative = band_integral(concentration, center, model; derivative=true)
        G .+= scale .* (predicted .- observed) .* derivative
    end
    return nothing
end
