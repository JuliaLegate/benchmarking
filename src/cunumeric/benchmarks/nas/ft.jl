# Host and device arrays use (nx, ny, nz). Each 3-D FFT is one pass.
# The initial field and twiddle are built in `initialize` (untimed); `run!`
# restarts from a device copy of the field.
# Checksums use a masked reduction whose scale accounts for `bfft!`.

struct CuNumericNASFTState{A,M,C}
    initial::A
    u0::A
    u1::A
    twiddle::A
    mask::M
    product::A
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
    host = Array{ComplexF64}(undef, shape)
    scratch = Vector{UInt64}(undef, min(2length(host), 1 << 20))
    initial = mod.zeros(ComplexF64, shape)
    copyto!(initial, nas_ft_initial_conditions_uint64!(host, scratch))
    ix2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.nx), p.nx, 1, 1))
    iy2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.ny), 1, p.ny, 1))
    iz2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.nz), 1, 1, p.nz))
    # Matching element types avoid promoted temporaries.
    twiddle = mod.zeros(ComplexF64, shape)
    ap = -4.0*NAS_FT_ALPHA*pi^2
    cuNumeric.@allowpromotion twiddle .= exp.(ap .* (ix2 .+ iy2 .+ iz2))
    foreach(cuNumeric.destroy!, (ix2, iy2, iz2))
    return (CuNumericNASFTState(
        initial, mod.zeros(ComplexF64, shape), mod.zeros(ComplexF64, shape), twiddle,
        mod.NDArray(cunumeric_nas_ft_checksum_mask(p)), mod.zeros(ComplexF64, shape), Any[],
    ),)
end

function run!(b::NASFourierTransform, s::CuNumericNASFTState)
    p = nas_ft_parameters(b.class)
    copyto!(s.u0, s.initial)
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
    foreach(cuNumeric.destroy!, (s.initial, s.u0, s.u1, s.twiddle, s.mask, s.product))
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
