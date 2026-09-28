include(joinpath(@__DIR__, "..", "..", "nas", "mg.jl"))

Base.@kwdef struct NASMultiGrid{T} <: AbstractBenchmark{T}
    N::Int
    M::Int
    class::String = "S"
    implementation::String = "default"
end

name(::NASMultiGrid) = "nas_mg"
dims(b::NASMultiGrid) = (b.N, b.M)
allowed_types(::Type{<:NASMultiGrid}) = Float64
# MG verifies against the official NPB norm; run the CUDA.jl check as well.
correctness_uses_cpu(::NASMultiGrid) = true
correctness_reference_label(mod, ::NASMultiGrid) = "NPB-GPU"

function data(b::NASMultiGrid)
    p = nas_mg_parameters(b.class)
    return "NAS MG class $(uppercase(b.class)): $(join(nas_mg_dims(p), "×")), $(p.niter) V-cycles per run"
end

function validate_nas_mg(b::NASMultiGrid{T}) where {T}
    T === Float64 || error("NAS MG is defined in Float64; got $T")
    p = nas_mg_parameters(b.class)
    nx, ny, _ = nas_mg_dims(p)
    (b.N, b.M) == (nx, ny) || error(
        "NAS MG class $(uppercase(b.class)) requires N=$nx, M=$ny; " *
        "got N=$(b.N), M=$(b.M)",
    )
    return p
end

function total_flops(b::NASMultiGrid)
    p = validate_nas_mg(b)
    return 58.0*p.niter*Float64(prod(nas_mg_dims(p)))
end

function total_space(b::NASMultiGrid)
    p = validate_nas_mg(b)
    hierarchy = sum(prod(big.(shape)) for shape in nas_mg_level_shapes(p))
    return (2hierarchy + prod(big.(nas_mg_dims(p) .+ 2)))*sizeof(Float64)
end

estimate_scaling(b::NASMultiGrid, ::Integer) = dims(b)
function class_dims(::Type{<:NASMultiGrid}, kwargs)
    nx, ny, _ = nas_mg_dims(nas_mg_parameters(get(kwargs, "class", "S")))
    return (nx, ny)
end
register_benchmark("nas_mg", NASMultiGrid)
