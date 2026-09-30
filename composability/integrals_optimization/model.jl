# Infer a two-parameter gas plume from wavelength-integrated measurements.
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

# Log parameters keep plume amplitude and width positive without solver bounds.
function plume(log_parameters, radius2)
    T = eltype(radius2)
    amplitude = exp(T(log_parameters[1]))
    width = exp(T(log_parameters[2]))
    return T(0.2) .+ amplitude .* exp.(-radius2 ./ (width * width))
end

absorption(λ::T) where {T} = T(0.15) + T(1.1) * exp(-((λ - T(0.5)) / T(0.16))^2)
response(λ::T, center::T) where {T} = exp(-((λ - center) / T(0.12))^2)

function band_integral(concentration, center, model)
    T = eltype(concentration)
    function integrand(λ, _)
        k = absorption(λ)
        return response(λ, center) .* exp.(-k .* concentration)
    end
    problem = IntegralProblem(integrand, (zero(T), one(T)))
    return Integrals.solve(problem, model.quadrature).u
end

squared_error_sum(a) = sum(a; init=zero(eltype(a)))

function loss(concentration, observations, model)
    squared_error = nothing
    for (center, observed) in zip(model.centers, observations)
        residual = band_integral(concentration, center, model) .- observed
        term = residual .* residual
        squared_error = isnothing(squared_error) ? term : squared_error .+ term
    end
    # The optimizer needs one host loss value per objective evaluation.
    # Give distributed GPU reductions a concrete identity for every tile.
    return Float64(squared_error_sum(squared_error)) /
           (length(concentration) * length(observations))
end
