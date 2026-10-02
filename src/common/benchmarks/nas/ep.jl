abstract type AbstractNASEP{T} <: AbstractBenchmark{T} end
include(joinpath(@__DIR__, "..", "..", "..", "nas", "ep.jl"))

dims(b::AbstractNASEP) = (b.N, b.M)
allowed_types(::Type{<:AbstractNASEP}) = Float64
throughput_label(::AbstractNASEP) = "G random numbers/s"
# EP verifies against the official NPB sums; CUDA.jl must run its check too.
correctness_uses_cpu(::AbstractNASEP) = true
correctness_reference_label(mod, ::AbstractNASEP) = "NPB-GPU"

function data(b::AbstractNASEP)
    p = nas_ep_parameters(b.class)
    return "NAS EP class $(uppercase(b.class)): 2^$(p.m + 1) random numbers"
end

function validate_nas_ep(b::AbstractNASEP{T}) where {T}
    T === Float64 || error("NAS EP is defined in Float64; got $T")
    b.M == 1 || error("NAS EP requires M=1")
    p = nas_ep_parameters(b.class)
    expected = nas_ep_random_numbers(p)
    b.N == expected || error(
        "NAS EP class $(uppercase(b.class)) requires N=$expected, got $(b.N)"
    )
    return p
end

# NPB reports millions of random numbers generated per second for EP. The
# harness stores that nominal operation rate in its common throughput column.
total_flops(b::AbstractNASEP) = Float64(nas_ep_random_numbers(validate_nas_ep(b)))

function total_space(b::AbstractNASEP)
    p = validate_nas_ep(b)
    streams = big(nas_ep_batches(p))
    return streams * (13 + p.m - NAS_EP_MK) * sizeof(Float64)
end

estimate_scaling(b::AbstractNASEP, ::Integer) = dims(b)
function class_dims(::Type{<:AbstractNASEP}, kwargs)
    return (nas_ep_random_numbers(nas_ep_parameters(get(kwargs, "class", "S"))), 1)
end
