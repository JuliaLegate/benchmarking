# Host and device arrays use (nx, ny, nz). Each 3-D FFT is one pass.
# Checksums use a masked reduction whose scale accounts for `bfft!`.

struct CuNumericNASFTState{A,M,X,Y,Z,H,R,C}
    u0::A
    u1::A
    twiddle::A
    mask::M
    product::A
    ix2::X
    iy2::Y
    iz2::Z
    host_initial::H
    rng_scratch::R
    checksums::C
end

function cleanup_result!(::NASFourierTransform, result, s::CuNumericNASFTState)
    foreach(cuNumeric.destroy!, result)
    empty!(s.checksums)
    return nothing
end

function cunumeric_nas_ft_frequency_squares(n)
    return Float64[((i + n÷2) % n - n÷2)^2 for i in 0:(n - 1)]
end

function cunumeric_nas_ft_checksum_mask(p)
    mask = zeros(ComplexF64, p.nx, p.ny, p.nz)
    scale = 1/prod(size(mask))
    @inbounds for j in 1:NAS_FT_CHECKSUM_SAMPLES
        mask[mod(j, p.nx) + 1, mod(3j, p.ny) + 1, mod(5j, p.nz) + 1] += scale
    end
    return mask
end

function initialize(b::NASFourierTransform{Float64}; mod=cuNumeric)
    p = validate_nas_ft(b)
    shape = (p.nx, p.ny, p.nz)
    u0 = mod.zeros(ComplexF64, shape)
    u1 = mod.zeros(ComplexF64, shape)
    # Matching element types avoid promoted temporaries.
    twiddle = mod.zeros(ComplexF64, shape)
    mask = mod.NDArray(cunumeric_nas_ft_checksum_mask(p))
    product = mod.zeros(ComplexF64, shape)
    ix2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.nx), p.nx, 1, 1))
    iy2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.ny), 1, p.ny, 1))
    iz2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.nz), 1, 1, p.nz))
    host = Array{ComplexF64}(undef, p.nx, p.ny, p.nz)
    scratch = Vector{UInt64}(undef, min(2length(host), 1 << 20))
    return (CuNumericNASFTState(
        u0, u1, twiddle, mask, product, ix2, iy2, iz2, host, scratch, Any[],
    ),)
end

function run!(b::NASFourierTransform, s::CuNumericNASFTState)
    p = nas_ft_parameters(b.class)
    nas_ft_initial_conditions_uint64!(s.host_initial, s.rng_scratch)
    copyto!(s.u0, s.host_initial)
    ap = -4.0*NAS_FT_ALPHA*pi^2
    cuNumeric.@allowpromotion s.twiddle .= exp.(ap .* (s.ix2 .+ s.iy2 .+ s.iz2))
    fft!(s.u0)
    for _ in 1:p.niter
        map!(*, s.u0, s.u0, s.twiddle)
        bfft!(s.u1, s.u0)
        map!(*, s.product, s.u1, s.mask)
        push!(s.checksums, sum(s.product))
    end
    return s.checksums
end

function cleanup!(::NASFourierTransform, s::CuNumericNASFTState)
    foreach(cuNumeric.destroy!, s.checksums)
    empty!(s.checksums)
    foreach(cuNumeric.destroy!,
        (s.u0, s.u1, s.twiddle, s.mask, s.product, s.ix2, s.iy2, s.iz2))
    return nothing
end

function check_benchmark_correctness(
    b::NASFourierTransform, gs::GlobalSettings; mod=cuNumeric
)
    state = only(initialize(b; mod))
    try
        got = ComplexF64[cuNumeric.@allowscalar(x[]) for x in run!(b, state)]
        return nas_ft_status(b.class, got)
    finally
        cleanup!(b, state)
    end
end
