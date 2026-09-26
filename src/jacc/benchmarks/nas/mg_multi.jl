# Multi-GPU NAS MG on JACC.Multi. Each level is an (n^2, columns) array of z-planes.
# Large levels are z-slabs with one ghost plane per side (sync_ghost_elems!);
# P halves per coarser level so restriction/interpolation stay local. Small
# levels are replicated on every GPU and computed redundantly. Custom code: the
# periodic z wrap between slab owners and the per-cycle gather at the slab/
# replicated boundary. `ops` is the JACC.Multi backend (a CPU mock in tests).

# `rep`: every device holds all n planes. Otherwise device d owns planes
# (d-1)P+1 : dP (padding past n is inert).
struct MGLayout
    n::Int
    P::Int
    rep::Bool
end

mg_cols(L::MGLayout, ndev) = L.rep ? ndev*L.n : ndev*L.P

@inline mg_k(a, L::MGLayout, jl) = L.rep ? jl : (a.dev_id - 1)*L.P + jl

@inline function mg_col(a, L::MGLayout, k)
    L.rep && return k
    shift = a.dev_id > 1 && a.ghost_dims > 0 ? 1 : 0
    return k - (a.dev_id - 1)*L.P + shift
end

@inline mg_at(a, L::MGLayout, i, j, k) = i + L.n*(j - 1) + L.n*L.n*(mg_col(a, L, k) - 1)

# Smallest slab level whose finest-level padding stays within 10%.
function mg_multi_plan(sizes, ndev)
    nlev = length(sizes)
    for lt in 1:nlev
        P = max(2, cld(sizes[lt], ndev))
        ndev*P*2^(nlev - lt) <= 1.10*sizes[end] || continue
        return [l < lt ? MGLayout(sizes[l], 0, true) :
                MGLayout(sizes[l], P*2^(l - lt), false) for l in 1:nlev]
    end
    return [l < nlev ? MGLayout(sizes[l], 0, true) :
            MGLayout(sizes[l], cld(sizes[l], ndev), false) for l in 1:nlev]
end

function mg_multi_alloc(ops, L::MGLayout, ndev, host=nothing)
    cols = mg_cols(L, ndev)
    x = zeros(Float64, L.n*L.n, cols)
    if host !== nothing
        x[:, 1:L.n] .= reshape(host, L.n*L.n, L.n)
    end
    return jm_array(ops, x; ghost_dims=L.rep ? 0 : 1)
end

# Kernels take (q, jl): plane item q and local column jl. Launch them through
# the 1-D Multi.parallel_for so consecutive threads walk one plane; the 2-D
# launch's 16x16 blocks split warps across planes (about 5x slower here).
function mgm_flat(i, f, m, args...)
    return f((i - 1) % m + 1, (i - 1) ÷ m + 1, args...)
end

mg_launch(ops, m, cols, f, args...) = jm_for(ops, m*cols, mgm_flat, f, m, args...)
mg_for(ops, L::MGLayout, m, f, args...) = mg_launch(ops, m, mg_cols(L, jm_ndev(ops)), f, args...)

function mgm_zero(q, jl, out, n)
    col = q + n*n*(jl - 1)
    col <= length(out) && (@inbounds out[col] = 0.0)
    return nothing
end

function mgm_fill!(ops, out, L)
    # Zero owned and ghost planes so no ghost exchange is needed.
    cols = L.rep ? L.n : L.P + 2
    return mg_launch(ops, L.n*L.n, jm_ndev(ops)*cols, mgm_zero, out, L.n)
end

function mgm_comm_x(q, jl, out, L)
    k = mg_k(out, L, jl)
    n = L.n
    if 2 <= k <= n - 1
        j = q + 1
        @inbounds begin
            out[mg_at(out, L, 1, j, k)] = out[mg_at(out, L, n - 1, j, k)]
            out[mg_at(out, L, n, j, k)] = out[mg_at(out, L, 2, j, k)]
        end
    end
    return nothing
end

function mgm_comm_y(q, jl, out, L)
    k = mg_k(out, L, jl)
    n = L.n
    if 2 <= k <= n - 1
        @inbounds begin
            out[mg_at(out, L, q, 1, k)] = out[mg_at(out, L, q, n - 1, k)]
            out[mg_at(out, L, q, n, k)] = out[mg_at(out, L, q, 2, k)]
        end
    end
    return nothing
end

function mgm_comm_z_local(q, _, out, L)
    n = L.n
    i, j = (q - 1) % n + 1, (q - 1) ÷ n + 1
    @inbounds begin
        out[mg_at(out, L, i, j, 1)] = out[mg_at(out, L, i, j, n - 1)]
        out[mg_at(out, L, i, j, n)] = out[mg_at(out, L, i, j, 2)]
    end
    return nothing
end

mg_owner(L::MGLayout, k) = cld(k, L.P)

function mg_plane_offset(L::MGLayout, d, k)
    shift = d > 1 && L.P > 0 ? 1 : 0
    return (k - (d - 1)*L.P + shift - 1)*L.n*L.n + 1
end

# Periodic z wrap across slab owners, then refresh inter-device ghost planes.
function mgm_comm_z_slab!(ops, out, L)
    n, nd = L.n, jm_ndev(ops)
    ps = jm_parts(ops, out)
    for (dst, src) in ((1, n - 1), (n, 2))
        dd, ds = mg_owner(L, dst), mg_owner(L, src)
        nd == 1 && (dd = ds = 1)
        doff = nd == 1 ? (dst - 1)*n*n + 1 : mg_plane_offset(L, dd, dst)
        soff = nd == 1 ? (src - 1)*n*n + 1 : mg_plane_offset(L, ds, src)
        jm_copy!(ops, ps[dd], dd, doff, ps[ds], ds, soff, n*n)
    end
    return jm_sync!(ops, out)
end

function mgm_comm3!(ops, out, L)
    mg_for(ops, L, L.n - 2, mgm_comm_x, out, L)
    mg_for(ops, L, L.n, mgm_comm_y, out, L)
    if L.rep
        mg_launch(ops, L.n*L.n, jm_ndev(ops), mgm_comm_z_local, out, L)
    else
        mgm_comm_z_slab!(ops, out, L)
    end
    return out
end

@inline function mgm_interior(q, n)
    return (q - 1) % (n - 2) + 2, (q - 1) ÷ (n - 2) + 2
end

function mgm_resid(q, jl, r, u, v, L)
    k = mg_k(r, L, jl)
    (2 <= k <= L.n - 1) || return nothing
    i, j = mgm_interior(q, L.n)
    U(di, dj, dk) = @inbounds u[mg_at(u, L, i + di, j + dj, k + dk)]
    @inbounds r[mg_at(r, L, i, j, k)] =
        v[mg_at(v, L, i, j, k)] - NAS_MG_A[1]*U(0, 0, 0) -
        NAS_MG_A[3] * (
            U(0, -1, -1) + U(0, 1, -1) + U(0, -1, 1) + U(0, 1, 1) +
            U(-1, 0, -1) + U(1, 0, -1) + U(-1, 0, 1) + U(1, 0, 1) +
            U(-1, -1, 0) + U(1, -1, 0) + U(-1, 1, 0) + U(1, 1, 0)
        ) -
        NAS_MG_A[4] * (
            U(-1, -1, -1) + U(1, -1, -1) + U(-1, 1, -1) + U(1, 1, -1) +
            U(-1, -1, 1) + U(1, -1, 1) + U(-1, 1, 1) + U(1, 1, 1)
        )
    return nothing
end

function mgm_resid!(ops, r, u, v, L)
    mg_for(ops, L, (L.n - 2)^2, mgm_resid, r, u, v, L)
    return mgm_comm3!(ops, r, L)
end

function mgm_psinv(q, jl, u, r, L, c)
    k = mg_k(u, L, jl)
    (2 <= k <= L.n - 1) || return nothing
    i, j = mgm_interior(q, L.n)
    R(di, dj, dk) = @inbounds r[mg_at(r, L, i + di, j + dj, k + dk)]
    @inbounds u[mg_at(u, L, i, j, k)] +=
        c[1]*R(0, 0, 0) +
        c[2]*(R(-1, 0, 0) + R(1, 0, 0) + R(0, -1, 0) + R(0, 1, 0) + R(0, 0, -1) + R(0, 0, 1)) +
        c[3] * (
            R(0, -1, -1) + R(0, 1, -1) + R(0, -1, 1) + R(0, 1, 1) +
            R(-1, 0, -1) + R(1, 0, -1) + R(-1, 0, 1) + R(1, 0, 1) +
            R(-1, -1, 0) + R(1, -1, 0) + R(-1, 1, 0) + R(1, 1, 0)
        )
    return nothing
end

function mgm_psinv!(ops, u, r, L, c)
    mg_for(ops, L, (L.n - 2)^2, mgm_psinv, u, r, L, c)
    return mgm_comm3!(ops, u, L)
end

function mgm_restrict(q, jl, coarse, fine, Lc, Lf)
    k = mg_k(coarse, Lc, jl)
    (2 <= k <= Lc.n - 1) || return nothing
    i, j = mgm_interior(q, Lc.n)
    fi, fj, fk = 2i - 1, 2j - 1, 2k - 1
    F(di, dj, dk) = @inbounds fine[mg_at(fine, Lf, fi + di, fj + dj, fk + dk)]
    @inbounds coarse[mg_at(coarse, Lc, i, j, k)] =
        0.5*F(0, 0, 0) +
        0.25*(F(-1, 0, 0) + F(1, 0, 0) + F(0, -1, 0) + F(0, 1, 0) + F(0, 0, -1) + F(0, 0, 1)) +
        0.125 * (
            F(0, -1, -1) + F(0, 1, -1) + F(0, -1, 1) + F(0, 1, 1) +
            F(-1, 0, -1) + F(1, 0, -1) + F(-1, 0, 1) + F(1, 0, 1) +
            F(-1, -1, 0) + F(1, -1, 0) + F(-1, 1, 0) + F(1, 1, 0)
        ) +
        0.0625 * (
            F(-1, -1, -1) + F(1, -1, -1) + F(-1, 1, -1) + F(1, 1, -1) +
            F(-1, -1, 1) + F(1, -1, 1) + F(-1, 1, 1) + F(1, 1, 1)
        )
    return nothing
end

function mgm_restrict!(ops, coarse, fine, Lc, Lf)
    mg_for(ops, Lc, (Lc.n - 2)^2, mgm_restrict, coarse, fine, Lc, Lf)
    return mgm_comm3!(ops, coarse, Lc)
end

@inline mgm_lerp(a, b, weight) = muladd(weight, b - a, a)

function mgm_interp(q, jl, fine, coarse, Lf, Lc)
    k = mg_k(fine, Lf, jl)
    k <= Lf.n || return nothing
    n = Lf.n
    i, j = (q - 1) % n + 1, (q - 1) ÷ n + 1
    qi, qj, qk = i - 1, j - 1, k - 1
    i0, j0, k0 = qi ÷ 2 + 1, qj ÷ 2 + 1, qk ÷ 2 + 1
    i1, j1, k1 = i0 + (qi % 2), j0 + (qj % 2), k0 + (qk % 2)
    wi, wj, wk = 0.5*(qi % 2), 0.5*(qj % 2), 0.5*(qk % 2)
    C(a, b, c) = @inbounds coarse[mg_at(coarse, Lc, a, b, c)]
    z00 = mgm_lerp(C(i0, j0, k0), C(i1, j0, k0), wi)
    z10 = mgm_lerp(C(i0, j1, k0), C(i1, j1, k0), wi)
    z01 = mgm_lerp(C(i0, j0, k1), C(i1, j0, k1), wi)
    z11 = mgm_lerp(C(i0, j1, k1), C(i1, j1, k1), wi)
    @inbounds fine[mg_at(fine, Lf, i, j, k)] +=
        mgm_lerp(mgm_lerp(z00, z10, wj), mgm_lerp(z01, z11, wj), wk)
    return nothing
end

function mgm_interp!(ops, fine, coarse, Lf, Lc)
    mg_for(ops, Lf, Lf.n*Lf.n, mgm_interp, fine, coarse, Lf, Lc)
    Lf.rep || jm_sync!(ops, fine)
    return fine
end

function mgm_norm_term(q, jl, r, L)
    k = mg_k(r, L, jl)
    (2 <= k <= L.n - 1) || return 0.0
    i, j = mgm_interior(q, L.n)
    return @inbounds abs2(r[mg_at(r, L, i, j, k)])
end

# Replicated copy of a slab level, for restriction into the first replicated level.
function mgm_gather!(ops, dest, src, L)
    host = jm_to_host(ops, src)
    n3 = L.n^3
    for (d, part) in enumerate(jm_parts(ops, dest))
        jm_upload!(ops, part, d, host, n3)
    end
    return dest
end

# Slab (ghosted) and replicated levels have different JACC.Multi array types.
struct JACCMultiMG{O}
    ops::O
    layouts::Vector{MGLayout}
    u::Vector{Any}
    r::Vector{Any}
    rhs::Any
    gathered::Any
    c::NTuple{4,Float64}
end

function jacc_multi_mg(ops, class)
    p = nas_mg_parameters(class)
    layouts = mg_multi_plan(nas_mg_level_sizes(p), jm_ndev(ops))
    u = Any[mg_multi_alloc(ops, L, jm_ndev(ops)) for L in layouts]
    r = Any[mg_multi_alloc(ops, L, jm_ndev(ops)) for L in layouts]
    rhs = mg_multi_alloc(ops, layouts[end], jm_ndev(ops), nas_mg_rhs(p))
    lt = findfirst(L -> !L.rep, layouts)
    gathered = lt > 1 ? mg_multi_alloc(ops, MGLayout(layouts[lt].n, 0, true), jm_ndev(ops)) : nothing
    return JACCMultiMG(ops, layouts, u, r, rhs, gathered, nas_mg_smoother(class))
end

function mgm_cycle!(s::JACCMultiMG)
    ops, Ls = s.ops, s.layouts
    finest = length(Ls)
    for level in finest:-1:2
        Lf, Lc = Ls[level], Ls[level - 1]
        if Lc.rep && !Lf.rep
            G = MGLayout(Lf.n, 0, true)
            mgm_gather!(ops, s.gathered, s.r[level], Lf)
            mgm_restrict!(ops, s.r[level - 1], s.gathered, Lc, G)
        else
            mgm_restrict!(ops, s.r[level - 1], s.r[level], Lc, Lf)
        end
    end
    mgm_fill!(ops, s.u[1], Ls[1])
    mgm_psinv!(ops, s.u[1], s.r[1], Ls[1], s.c)
    for level in 2:(finest - 1)
        L = Ls[level]
        mgm_fill!(ops, s.u[level], L)
        mgm_interp!(ops, s.u[level], s.u[level - 1], L, Ls[level - 1])
        mgm_resid!(ops, s.r[level], s.u[level], s.r[level], L)
        mgm_psinv!(ops, s.u[level], s.r[level], L, s.c)
    end
    L = Ls[end]
    mgm_interp!(ops, s.u[end], s.u[end - 1], L, Ls[end - 1])
    mgm_resid!(ops, s.r[end], s.u[end], s.rhs, L)
    mgm_psinv!(ops, s.u[end], s.r[end], L, s.c)
    return nothing
end

# Returns the final L2 sum of squares (host value; JACC.Multi reductions sync).
function mgm_run!(s::JACCMultiMG, class)
    p = nas_mg_parameters(class)
    ops, L = s.ops, s.layouts[end]
    foreach(((u, Lu),) -> mgm_fill!(ops, u, Lu), zip(s.u, s.layouts))
    mgm_resid!(ops, s.r[end], s.u[end], s.rhs, L)
    mg_reduce(ops, L, (L.n - 2)^2, mgm_norm_term, s.r[end], L)
    for _ in 1:p.niter
        mgm_cycle!(s)
        mgm_resid!(ops, s.r[end], s.u[end], s.rhs, L)
    end
    return mg_reduce(ops, L, (L.n - 2)^2, mgm_norm_term, s.r[end], L)
end

# The 2-D Multi.parallel_reduce is also wrong in JACC 1.4: it drops partial
# sums when a device has more than 16 row blocks (`ii <= N` in
# _multi_reduce_kernel_cuda_MN).
mg_reduce(ops, L::MGLayout, m, f, args...) =
    jm_reduce(ops, m*mg_cols(L, jm_ndev(ops)), mgm_flat, f, m, args...)
