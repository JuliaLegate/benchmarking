# Initial conditions are generated on the host. Full 3-D FFTs do not partition.
# Checksums use a masked reduction whose scale accounts for `bfft!`.

struct CuNumericNASFTState{A,M,X,Y,Z,H,C}
    u0::A
    u1::A
    twiddle::A
    mask::M
    product::A
    ix2::X
    iy2::Y
    iz2::Z
    host_initial::H
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

function initialize(b::NASFourierTransform{Float64}; mod=cuNumeric)
    p = validate_nas_ft(b)
    shape = (p.nx, p.ny, p.nz)
    u0 = mod.zeros(ComplexF64, shape)
    u1 = mod.zeros(ComplexF64, shape)
    # Matching element types avoid promoted temporaries.
    twiddle = mod.zeros(ComplexF64, shape)
    mask = mod.NDArray(ComplexF64.(nas_ft_checksum_mask(p)) ./ prod(shape))
    product = mod.zeros(ComplexF64, shape)
    ix2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.nx), p.nx, 1, 1))
    iy2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.ny), 1, p.ny, 1))
    iz2 = mod.NDArray(reshape(cunumeric_nas_ft_frequency_squares(p.nz), 1, 1, p.nz))
    host = Array{ComplexF64}(undef, shape)
    return (CuNumericNASFTState(u0, u1, twiddle, mask, product, ix2, iy2, iz2, host, Any[]),)
end

function run!(b::NASFourierTransform, s::CuNumericNASFTState)
    p = nas_ft_parameters(b.class)
    nas_ft_initial_conditions!(s.host_initial)
    copyto!(s.u0, s.host_initial)
    ap = -4.0*NAS_FT_ALPHA*pi^2
    cuNumeric.@allowpromotion s.twiddle .= exp.(ap .* (s.ix2 .+ s.iy2 .+ s.iz2))
    fft!(s.u0)
    for _ in 1:p.niter
        map!(*, s.u0, s.u0, s.twiddle)
        copyto!(s.u1, s.u0)
        bfft!(s.u1)
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
