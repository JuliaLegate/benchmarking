# Multi-GPU NAS FT on JACC.Multi. The field lives as z-slabs U, (nx*ny, nz),
# for the 2-D (x, y) FFTs and as y-slabs W, (nz*nx, ny) with z fastest, for the
# 1-D z FFT; both FFTs then cover leading dimensions, so cuFFT batches each in
# one call. The spectrum is evolved in W. JACC has no FFT, so each GPU runs
# cuFFT on its part. The transpose between layouts packs with JACC.Multi
# kernels and exchanges blocks with custom GPU-to-GPU copies (JACC.Multi has no
# all-to-all). Inverse FFTs are unnormalized; the checksum applies 1/N.
# Checksums use the 1-D Multi.parallel_reduce and are fetched every iteration.
# Kernels are @inline: JACC.Multi does not inline `f`, costing 3-5x.

@inline function ftm_randlc(x, a)
    r23, t23 = 2.0^-23, 2.0^23
    r46, t46 = 2.0^-46, 2.0^46
    a1 = trunc(r23*a)
    a2 = a - t23*a1
    x1 = trunc(r23*x)
    x2 = x - t23*x1
    t1 = a1*x2 + a2*x1
    z = t1 - t23*trunc(r23*t1)
    t3 = t23*z + a2*x2
    next = t3 - t46*trunc(r46*t3)
    return next, r46*next
end

# Local sizes: Pz planes per GPU in U, Py columns per GPU in W, B elements per
# (source, destination) block of the transpose.
struct FTLayout
    nx::Int
    ny::Int
    nz::Int
    Pz::Int
    Py::Int
    B::Int
end

function FTLayout(p, nd)
    p.nz % nd == 0 && p.ny % nd == 0 ||
        error("NAS FT needs nz=$(p.nz) and ny=$(p.ny) divisible by $nd GPUs")
    Pz, Py = p.nz ÷ nd, p.ny ÷ nd
    return FTLayout(p.nx, p.ny, p.nz, Pz, Py, p.nx*Py*Pz)
end

@inline function ftm_flat(i, f, m, args...)
    return f((i - 1) % m + 1, (i - 1) ÷ m + 1, args...)
end

ftm_launch(ops, m, cols, f, args...) = jm_for(ops, m*cols, ftm_flat, f, m, args...)

@inline function ftm_initial(k, u, starts, plane)
    x = @inbounds starts[k]
    offset = (k - 1)*plane
    @inbounds for i in 1:plane
        x, re = ftm_randlc(x, NAS_FT_MULTIPLIER)
        x, im = ftm_randlc(x, NAS_FT_MULTIPLIER)
        u[offset + i] = ComplexF64(re, im)
    end
    return nothing
end

@inline function ftm_twiddle(q, jl, tw, L, ap)
    z, x = (q - 1) % L.nz, (q - 1) ÷ L.nz
    y = (tw.dev_id - 1)*L.Py + jl - 1
    ii = (x + L.nx ÷ 2) % L.nx - L.nx ÷ 2
    jj = (y + L.ny ÷ 2) % L.ny - L.ny ÷ 2
    kk = (z + L.nz ÷ 2) % L.nz - L.nz ÷ 2
    @inbounds tw[q + L.nx*L.nz*(jl - 1)] = exp(ap*(ii*ii + jj*jj + kk*kk))
    return nothing
end

@inline function ftm_evolve(q, jl, w0, w1, tw, L)
    index = q + L.nx*L.nz*(jl - 1)
    @inbounds w0[index] *= tw[index]
    @inbounds w1[index] = w0[index]
    return nothing
end

# z->y blocks are x-fastest (matching U); y->z blocks are z-fastest (matching
# W). Each pack is then contiguous and only its unpack reads with a stride.
@inline ftm_block(L, x, yl, zl) = x + L.nx*(yl - 1) + L.nx*L.Py*(zl - 1)
@inline ftm_block_z(L, x, yl, zl) = zl + L.Pz*(x - 1) + L.Pz*L.nx*(yl - 1)

@inline function ftm_pack_z2y(q, zl, send, u, L)
    x, y = (q - 1) % L.nx + 1, (q - 1) ÷ L.nx + 1
    dst, yl = (y - 1) ÷ L.Py + 1, (y - 1) % L.Py + 1
    @inbounds send[(dst - 1)*L.B + ftm_block(L, x, yl, zl)] = u[q + L.nx*L.ny*(zl - 1)]
    return nothing
end

@inline function ftm_unpack_y(q, yl, w, recv, L)
    z, x = (q - 1) % L.nz + 1, (q - 1) ÷ L.nz + 1
    src, zl = (z - 1) ÷ L.Pz + 1, (z - 1) % L.Pz + 1
    @inbounds w[q + L.nx*L.nz*(yl - 1)] = recv[(src - 1)*L.B + ftm_block(L, x, yl, zl)]
    return nothing
end

@inline function ftm_pack_y2z(q, yl, send, w, L)
    z, x = (q - 1) % L.nz + 1, (q - 1) ÷ L.nz + 1
    dst, zl = (z - 1) ÷ L.Pz + 1, (z - 1) % L.Pz + 1
    @inbounds send[(dst - 1)*L.B + ftm_block_z(L, x, yl, zl)] = w[q + L.nx*L.nz*(yl - 1)]
    return nothing
end

@inline function ftm_unpack_z(q, zl, u, recv, L)
    x, y = (q - 1) % L.nx + 1, (q - 1) ÷ L.nx + 1
    src, yl = (y - 1) ÷ L.Py + 1, (y - 1) % L.Py + 1
    @inbounds u[q + L.nx*L.ny*(zl - 1)] = recv[(src - 1)*L.B + ftm_block_z(L, x, yl, zl)]
    return nothing
end

# Block `dst` of every GPU's send buffer becomes block `src` of GPU dst's recv.
function ftm_exchange!(ops, recv, send, L)
    rp, sp = jm_parts(ops, recv), jm_parts(ops, send)
    for src in eachindex(sp), dst in eachindex(rp)
        jm_copy!(ops, rp[dst], dst, (src - 1)*L.B + 1, sp[src], src, (dst - 1)*L.B + 1, L.B)
    end
    return recv
end

function ftm_z_to_y!(ops, w, u, s)
    L = s.L
    ftm_launch(ops, L.nx*L.ny, L.nz, ftm_pack_z2y, s.send, u, L)
    ftm_exchange!(ops, s.recv, s.send, L)
    ftm_launch(ops, L.nx*L.nz, L.ny, ftm_unpack_y, w, s.recv, L)
    return w
end

function ftm_y_to_z!(ops, u, w, s)
    L = s.L
    ftm_launch(ops, L.nx*L.nz, L.ny, ftm_pack_y2z, s.send, w, L)
    ftm_exchange!(ops, s.recv, s.send, L)
    ftm_launch(ops, L.nx*L.ny, L.nz, ftm_unpack_z, u, s.recv, L)
    return u
end

@inline function ftm_sample(j, L)
    x, y, z = j % L.nx, (3j) % L.ny, (5j) % L.nz
    return x, y, z
end

@inline function ftm_checksum(j, _, u, L, imagpart)
    x, y, z = ftm_sample(j, L)
    owner = z ÷ L.Pz + 1
    owner == u.dev_id || return 0.0
    v = @inbounds u[x + y*L.nx + (z - (owner - 1)*L.Pz)*L.nx*L.ny + 1]
    v /= L.nx*L.ny*L.nz
    return imagpart ? imag(v) : real(v)
end

struct JACCMultiFT{O}
    ops::O
    class::String
    L::FTLayout
    u::Any
    w0::Any
    w1::Any
    tw::Any
    send::Any
    recv::Any
    starts::Any
    host_starts::Vector{Float64}
    checksums::Vector{ComplexF64}
end

function jacc_multi_ft(ops, class)
    p, nd = nas_ft_parameters(class), jm_ndev(ops)
    L = FTLayout(p, nd)
    arr(T, m, cols) = jm_array(ops, zeros(T, m, cols); ghost_dims=0)
    return JACCMultiFT(
        ops, class, L,
        arr(ComplexF64, p.nx*p.ny, p.nz),
        arr(ComplexF64, p.nx*p.nz, p.ny), arr(ComplexF64, p.nx*p.nz, p.ny),
        arr(Float64, p.nx*p.nz, p.ny),
        arr(ComplexF64, p.nx*p.ny*L.Pz, nd), arr(ComplexF64, p.nx*p.ny*L.Pz, nd),
        jm_array(ops, zeros(Float64, p.nz); ghost_dims=0),
        Vector{Float64}(undef, p.nz), ComplexF64[],
    )
end

function ftm_run!(s::JACCMultiFT)
    ops, L = s.ops, s.L
    p = nas_ft_parameters(s.class)
    nas_ft_plane_starts!(s.host_starts, L.nx, L.ny)
    for (d, part) in enumerate(jm_parts(ops, s.starts))
        jm_upload!(ops, part, d, s.host_starts[((d - 1)*L.Pz + 1):(d*L.Pz)], L.Pz)
    end
    jm_for(ops, L.nz, ftm_initial, s.u, s.starts, L.nx*L.ny)
    ftm_launch(ops, L.nx*L.nz, L.ny, ftm_twiddle, s.tw, L, -4.0*NAS_FT_ALPHA*pi^2)
    jm_fft!(ops, s.u, (L.nx, L.ny, L.Pz), (1, 2), false)
    ftm_z_to_y!(ops, s.w0, s.u, s)
    jm_fft!(ops, s.w0, (L.nz, L.nx, L.Py), 1, false)
    empty!(s.checksums)
    nd = jm_ndev(ops)
    for _ in 1:p.niter
        ftm_launch(ops, L.nx*L.nz, L.ny, ftm_evolve, s.w0, s.w1, s.tw, L)
        jm_fft!(ops, s.w1, (L.nz, L.nx, L.Py), 1, true)
        ftm_y_to_z!(ops, s.u, s.w1, s)
        jm_fft!(ops, s.u, (L.nx, L.ny, L.Pz), (1, 2), true)
        n = NAS_FT_CHECKSUM_SAMPLES
        re = jm_reduce(ops, n*nd, ftm_flat, ftm_checksum, n, s.u, L, false)
        im = jm_reduce(ops, n*nd, ftm_flat, ftm_checksum, n, s.u, L, true)
        push!(s.checksums, ComplexF64(re, im))
    end
    return s.checksums
end
