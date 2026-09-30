# Multi-GPU Gray-Scott on JACC.Multi, following nas/mg_multi.jl. The periodic
# N×M grid is stored as N×(ndev*P) columns: a halo column, data columns 1..M, a
# halo column, then inert padding. Device d owns global columns (d-1)P+1 : dP;
# one ghost column per side (sync_ghost_elems!) covers neighbours across
# devices, and the two halos (refreshed with jm_copy!) close the periodic wrap
# in j. `ops` is the JACC.Multi backend (a CPU mock in tests).

struct GSLayout
    N::Int
    M::Int
    P::Int
end

function gs_layout(N::Integer, M::Integer, ndev::Integer)
    P = cld(M + 2, ndev)
    P >= 2 || error("JACC grayscott needs at least 2 columns per GPU; got M=$M on $ndev GPUs")
    return GSLayout(N, M, P)
end

mutable struct GSMultiState{A}
    L::GSLayout
    u::A
    v::A
    u_new::A
    v_new::A
end

# Local column of global column g on device d (devices after the first hold a
# left ghost column).
gsm_local(L::GSLayout, d, g) = g - (d - 1)*L.P + (d > 1 ? 1 : 0)
gsm_owner(L::GSLayout, g) = cld(g, L.P)

function gsm_alloc(ops, L::GSLayout, host)
    x = zeros(eltype(host), L.N, jm_ndev(ops)*L.P)
    x[:, 2:(L.M + 1)] .= host
    x[:, 1] .= host[:, L.M]
    x[:, L.M + 2] .= host[:, 1]
    return jm_array(ops, x; ghost_dims=1)
end

function jacc_multi_grayscott_state(ops, u_host, v_host)
    L = gs_layout(size(u_host)..., jm_ndev(ops))
    return GSMultiState(
        L, gsm_alloc(ops, L, u_host), gsm_alloc(ops, L, v_host),
        gsm_alloc(ops, L, zero(u_host)), gsm_alloc(ops, L, zero(v_host)),
    )
end

# One item per (row, owned column); periodic in i within the column, and the
# ghost/halo columns supply the j neighbours. @inline: Multi.parallel_for calls
# `f(i, x...)` without inlining.
@inline function gsm_kernel(i, u, v, u_new, v_new, L, dt, dx2, cu, cv, f, k)
    N = L.N
    r, jl = (i - 1) % N + 1, (i - 1) ÷ N + 1
    g = (u.dev_id - 1)*L.P + jl
    (2 <= g <= L.M + 1) || return nothing
    c = r + N*(jl + (u.dev_id > 1 ? 1 : 0) - 1)
    im = r == 1 ? c + N - 1 : c - 1
    ip = r == N ? c - N + 1 : c + 1
    @inbounds begin
        up, vp = u[c], v[c]
        lu = (u[ip] - 2up + u[im]) / dx2 + (u[c + N] - 2up + u[c - N]) / dx2
        lv = (v[ip] - 2vp + v[im]) / dx2 + (v[c + N] - 2vp + v[c - N]) / dx2
        uvv = up * vp * vp
        u_new[c] = up + dt * (cu * lu - uvv + f * (one(up) - up))
        v_new[c] = vp + dt * (cv * lv + uvv - (f + k) * vp)
    end
    return nothing
end

# Periodic wrap: halo 1 <- data column M, halo M+2 <- data column 1, then
# refresh the ghosts, which may include a halo next to a device boundary.
function gsm_halo!(ops, a, L::GSLayout)
    ps = jm_parts(ops, a)
    for (dst, src) in ((1, L.M + 1), (L.M + 2, 2))
        dd, ds = gsm_owner(L, dst), gsm_owner(L, src)
        doff = (gsm_local(L, dd, dst) - 1)*L.N + 1
        soff = (gsm_local(L, ds, src) - 1)*L.N + 1
        jm_copy!(ops, ps[dd], dd, doff, ps[ds], ds, soff, L.N)
    end
    return jm_sync!(ops, a)
end

function gsm_step!(ops, s::GSMultiState, p)
    L = s.L
    jm_for(
        ops, L.N*jm_ndev(ops)*L.P, gsm_kernel,
        s.u, s.v, s.u_new, s.v_new, L, p.dt, p.dx2, p.cu, p.cv, p.f, p.k,
    )
    gsm_halo!(ops, s.u_new, L)
    gsm_halo!(ops, s.v_new, L)
    s.u, s.u_new = s.u_new, s.u
    s.v, s.v_new = s.v_new, s.v
    return s
end

gsm_to_host(ops, a, L::GSLayout) = jm_to_host(ops, a)[:, 2:(L.M + 1)]
