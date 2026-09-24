# Initial conditions are generated on the host. Full 3-D FFTs do not partition.
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
    mask = zeros(ComplexF64, p.nz, p.ny, p.nx)
    scale = 1/prod(size(mask))
    @inbounds for j in 1:NAS_FT_CHECKSUM_SAMPLES
        mask[mod(5j, p.nz) + 1, mod(3j, p.ny) + 1, mod(j, p.nx) + 1] += scale
    end
    return mask
end

function cunumeric_nas_ft_upload!(dest, host)
    # The host's Julia column-major (nx, ny, nz) bytes are the row-major bytes
    # of the reversed device shape (nz, ny, nx). Attach directly to avoid a
    # full-volume host transpose on every timed run. The copy must finish
    # before the host buffer is reused or the attachment is destroyed.
    attached = cuNumeric.nda_attach_external(host; shape=size(dest))
    try
        GC.@preserve host attached begin
            copyto!(dest, attached)
            cuNumeric.issue_execution_fence(; block=true)
        end
    finally
        cuNumeric.destroy!(attached)
    end
    return dest
end

function initialize(b::NASFourierTransform{Float64}; mod=cuNumeric)
    p = validate_nas_ft(b)
    shape = (p.nz, p.ny, p.nx)
    u0 = mod.zeros(ComplexF64, shape)
    u1 = mod.zeros(ComplexF64, shape)
    # Matching element types avoid promoted temporaries.
    twiddle = mod.zeros(ComplexF64, shape)
    mask = mod.NDArray(cunumeric_nas_ft_checksum_mask(p))
    product = mod.zeros(ComplexF64, shape)
    ix2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.nx), 1, 1, p.nx))
    iy2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.ny), 1, p.ny, 1))
    iz2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.nz), p.nz, 1, 1))
    host = Array{ComplexF64}(undef, p.nx, p.ny, p.nz)
    scratch = Vector{UInt64}(undef, min(2length(host), 1 << 20))
    return (CuNumericNASFTState(
        u0, u1, twiddle, mask, product, ix2, iy2, iz2, host, scratch, Any[],
    ),)
end

function run!(b::NASFourierTransform, s::CuNumericNASFTState)
    p = nas_ft_parameters(b.class)
    nas_ft_initial_conditions_uint64!(s.host_initial, s.rng_scratch)
    cunumeric_nas_ft_upload!(s.u0, s.host_initial)
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
        return nas_ft_verified(b.class, got) ? "pass" : "fail"
    finally
        cleanup!(b, state)
    end
end
