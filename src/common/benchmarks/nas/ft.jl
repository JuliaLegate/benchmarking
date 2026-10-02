abstract type AbstractNASFT{T} <: AbstractBenchmark{T} end
include(joinpath(@__DIR__, "..", "..", "..", "nas", "ft.jl"))

dims(b::AbstractNASFT) = (b.N, b.M)
allowed_types(::Type{<:AbstractNASFT}) = Float64
# Self-verifies against analytic NPB checksums, not CUDA.jl; lets the CUDA.jl
# backend run its own check instead of skipping.
correctness_uses_cpu(::AbstractNASFT) = true

function data(b::AbstractNASFT)
    p = nas_ft_parameters(b.class)
    return "NAS FT class $(uppercase(b.class)): $(p.nx)×$(p.ny)×$(p.nz), $(p.niter) NAS time steps per run"
end

function validate_nas_ft(b::AbstractNASFT{T}) where {T}
    T === Float64 || error("NAS FT is defined in Float64; got $T")
    p = nas_ft_parameters(b.class)
    (b.N, b.M) == (p.nx, p.ny) || error(
        "NAS FT class $(uppercase(b.class)) requires N=$(p.nx), M=$(p.ny); " *
        "got N=$(b.N), M=$(b.M)",
    )
    return p
end

function total_flops(b::AbstractNASFT)
    p = validate_nas_ft(b)
    ntotal = Float64(p.nx)*p.ny*p.nz
    l = log(ntotal)
    return ntotal * (14.8157 + 7.19641*l + (5.23518 + 7.21113*l)*p.niter)
end

function total_space(b::AbstractNASFT)
    p = validate_nas_ft(b)
    n = big(p.nx)*p.ny*p.nz
    return 3n*sizeof(ComplexF64) + 2n*sizeof(Float64)
end

estimate_scaling(b::AbstractNASFT, ::Integer) = dims(b)
function class_dims(::Type{<:AbstractNASFT}, kwargs)
    p = nas_ft_parameters(get(kwargs, "class", "S"))
    return (p.nx, p.ny)
end
