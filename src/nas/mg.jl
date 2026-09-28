const NAS_MG_NPB_GPU_COMMIT = "3f12d84920ee315ab00ef283717c1e74b68f4d00"
const NAS_MG_SEED = 314159265.0
const NAS_MG_MULTIPLIER = 1220703125.0
const NAS_MG_EXTREMA = 10

const NAS_MG_CLASSES = Dict(
    "S" => (; n=32, niter=4, norm=0.5307707005734e-4),
    "W" => (; n=128, niter=4, norm=0.6467329375339e-5),
    "A" => (; n=256, niter=4, norm=0.2433365309069e-5),
    "B" => (; n=256, niter=20, norm=0.1800564401355e-5),
    "C" => (; n=512, niter=20, norm=0.5706732285740e-6),
    "D" => (; n=1024, niter=50, norm=0.1583275060440e-9),
    "E" => (; n=2048, niter=50, norm=0.8157592357404e-10),
    # Weak scaling at a class's grid points per GPU, named <class>.<gpus>; each
    # doubling extends the slowest axis first. No NAS reference.
    "S.2" => (; dims=(32, 32, 64), niter=4, norm=nothing),
    "B.2" => (; dims=(256, 256, 512), niter=20, norm=nothing),
    "B.4" => (; dims=(256, 512, 512), niter=20, norm=nothing),
)

const NAS_MG_A = (-8.0/3.0, 0.0, 1.0/6.0, 1.0/12.0)

function nas_mg_parameters(class::AbstractString)
    key = uppercase(class)
    return get(NAS_MG_CLASSES, key) do
        known = join(sort!(collect(keys(NAS_MG_CLASSES))), ", ")
        return error("Unknown NAS MG class '$class'; expected one of $known")
    end
end

# Interior grid size (nx, ny, nz); NAS classes are cubes.
nas_mg_dims(p) = hasproperty(p, :dims) ? p.dims : (p.n, p.n, p.n)

function nas_mg_smoother(class::AbstractString)
    # Weak-scaling classes use their base class's smoother.
    return if uppercase(first(split(class, "."))) in ("S", "W", "A")
        (-3.0/8.0, 1.0/32.0, -1.0/64.0, 0.0)
    else
        (-3.0/17.0, 1.0/33.0, -1.0/61.0, 0.0)
    end
end

function nas_mg_randlc(x::Float64, a::Float64=NAS_MG_MULTIPLIER)
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

function nas_mg_insert_extreme!(values, indices, value, index, largest)
    if (largest && value <= values[1]) || (!largest && value >= values[1])
        return nothing
    end
    values[1], indices[1] = value, index
    for i in 1:(length(values) - 1)
        ordered = largest ? values[i] <= values[i + 1] : values[i] >= values[i + 1]
        ordered && break
        values[i], values[i + 1] = values[i + 1], values[i]
        indices[i], indices[i + 1] = indices[i + 1], indices[i]
    end
    return nothing
end

"""Construct the exact sparse NPB MG right-hand side, including periodic ghosts."""
function nas_mg_rhs(p)
    nx, ny, nz = nas_mg_dims(p)
    shape = (nx + 2, ny + 2, nz + 2)
    low_values, high_values = ones(NAS_MG_EXTREMA), zeros(NAS_MG_EXTREMA)
    low_indices = fill(CartesianIndex(1, 1, 1), NAS_MG_EXTREMA)
    high_indices = copy(low_indices)
    seed = NAS_MG_SEED
    @inbounds for k in 2:(nz + 1), j in 2:(ny + 1), i in 2:(nx + 1)
        seed, value = nas_mg_randlc(seed)
        index = CartesianIndex(i, j, k)
        nas_mg_insert_extreme!(low_values, low_indices, value, index, false)
        nas_mg_insert_extreme!(high_values, high_indices, value, index, true)
    end
    rhs = zeros(Float64, shape)
    rhs[low_indices] .= -1.0
    rhs[high_indices] .= 1.0
    nas_mg_comm3!(rhs)
    return rhs
end

nas_mg_level_sizes(p) = [2^level + 2 for level in 1:round(Int, log2(p.n))]

# Ghosted (nx, ny, nz) per level, coarsest first. Every axis halves per level
# until the smallest side is 2.
function nas_mg_level_shapes(p)
    d = nas_mg_dims(p)
    lt = round(Int, log2(minimum(d)))
    return [d .÷ 2^(lt - level) .+ 2 for level in 1:lt]
end

function nas_mg_comm3!(u)
    n1, n2, n3 = size(u)
    @views begin
        u[1:1, 2:(n2 - 1), 2:(n3 - 1)] .= u[(n1 - 1):(n1 - 1), 2:(n2 - 1), 2:(n3 - 1)]
        u[n1:n1, 2:(n2 - 1), 2:(n3 - 1)] .= u[2:2, 2:(n2 - 1), 2:(n3 - 1)]
        u[:, 1:1, 2:(n3 - 1)] .= u[:, (n2 - 1):(n2 - 1), 2:(n3 - 1)]
        u[:, n2:n2, 2:(n3 - 1)] .= u[:, 2:2, 2:(n3 - 1)]
        u[:, :, 1:1] .= u[:, :, (n3 - 1):(n3 - 1)]
        u[:, :, n3:n3] .= u[:, :, 2:2]
    end
    return u
end

function nas_mg_resid!(r, u, v, a=NAS_MG_A)
    x, y, z = axes(u)
    xi, yi, zi = 2:(last(x) - 1), 2:(last(y) - 1), 2:(last(z) - 1)
    xm, xp = 1:(last(x) - 2), 3:last(x)
    ym, yp = 1:(last(y) - 2), 3:last(y)
    zm, zp = 1:(last(z) - 2), 3:last(z)
    @views r[xi, yi, zi] .=
        v[xi, yi, zi] .- a[1] .* u[xi, yi, zi] .-
        a[3] .* (
            u[xi, ym, zm] .+ u[xi, yp, zm] .+ u[xi, ym, zp] .+ u[xi, yp, zp] .+
            u[xm, yi, zm] .+ u[xp, yi, zm] .+ u[xm, yi, zp] .+ u[xp, yi, zp] .+
            u[xm, ym, zi] .+ u[xp, ym, zi] .+ u[xm, yp, zi] .+ u[xp, yp, zi]
        ) .-
        a[4] .* (
            u[xm, ym, zm] .+ u[xp, ym, zm] .+ u[xm, yp, zm] .+ u[xp, yp, zm] .+
            u[xm, ym, zp] .+ u[xp, ym, zp] .+ u[xm, yp, zp] .+ u[xp, yp, zp]
        )
    return nas_mg_comm3!(r)
end

function nas_mg_psinv!(u, r, c)
    x, y, z = axes(r)
    xi, yi, zi = 2:(last(x) - 1), 2:(last(y) - 1), 2:(last(z) - 1)
    xm, xp = 1:(last(x) - 2), 3:last(x)
    ym, yp = 1:(last(y) - 2), 3:last(y)
    zm, zp = 1:(last(z) - 2), 3:last(z)
    @views u[xi, yi, zi] .+=
        c[1] .* r[xi, yi, zi] .+
        c[2] .* (
            r[xm, yi, zi] .+ r[xp, yi, zi] .+ r[xi, ym, zi] .+
            r[xi, yp, zi] .+ r[xi, yi, zm] .+ r[xi, yi, zp]
        ) .+
        c[3] .* (
            r[xi, ym, zm] .+ r[xi, yp, zm] .+ r[xi, ym, zp] .+ r[xi, yp, zp] .+
            r[xm, yi, zm] .+ r[xp, yi, zm] .+ r[xm, yi, zp] .+ r[xp, yi, zp] .+
            r[xm, ym, zi] .+ r[xp, ym, zi] .+ r[xm, yp, zi] .+ r[xp, yp, zi]
        )
    return nas_mg_comm3!(u)
end

function nas_mg_restrict!(coarse, fine)
    c = ntuple(d -> 2:(size(coarse, d) - 1), 3)
    f = ntuple(d -> 3:2:(size(fine, d) - 1), 3)
    m, q = map(r -> r .- 1, f), map(r -> r .+ 1, f)
    F(a, b, e) = @view fine[a, b, e]
    @views coarse[c...] .=
        0.5 .* F(f[1], f[2], f[3]) .+
        0.25 .* (
            F(m[1], f[2], f[3]) .+ F(q[1], f[2], f[3]) .+ F(f[1], m[2], f[3]) .+
            F(f[1], q[2], f[3]) .+ F(f[1], f[2], m[3]) .+ F(f[1], f[2], q[3])
        ) .+
        0.125 .* (
            F(f[1], m[2], m[3]) .+ F(f[1], q[2], m[3]) .+ F(f[1], m[2], q[3]) .+ F(f[1], q[2], q[3]) .+
            F(m[1], f[2], m[3]) .+ F(q[1], f[2], m[3]) .+ F(m[1], f[2], q[3]) .+ F(q[1], f[2], q[3]) .+
            F(m[1], m[2], f[3]) .+ F(q[1], m[2], f[3]) .+ F(m[1], q[2], f[3]) .+ F(q[1], q[2], f[3])
        ) .+
        0.0625 .* (
            F(m[1], m[2], m[3]) .+ F(q[1], m[2], m[3]) .+ F(m[1], q[2], m[3]) .+ F(q[1], q[2], m[3]) .+
            F(m[1], m[2], q[3]) .+ F(q[1], m[2], q[3]) .+ F(m[1], q[2], q[3]) .+ F(q[1], q[2], q[3])
        )
    return nas_mg_comm3!(coarse)
end

function nas_mg_interp!(fine, coarse)
    o = ntuple(d -> 1:2:(size(fine, d) - 1), 3)
    e = ntuple(d -> 2:2:size(fine, d), 3)
    lo = ntuple(d -> 1:(size(coarse, d) - 1), 3)
    hi = ntuple(d -> 2:size(coarse, d), 3)
    C(a, b, c) = @view coarse[a, b, c]
    @views begin
        fine[o[1], o[2], o[3]] .+= C(lo[1], lo[2], lo[3])
        fine[e[1], o[2], o[3]] .+= 0.5 .* (C(lo[1], lo[2], lo[3]) .+ C(hi[1], lo[2], lo[3]))
        fine[o[1], e[2], o[3]] .+= 0.5 .* (C(lo[1], lo[2], lo[3]) .+ C(lo[1], hi[2], lo[3]))
        fine[o[1], o[2], e[3]] .+= 0.5 .* (C(lo[1], lo[2], lo[3]) .+ C(lo[1], lo[2], hi[3]))
        fine[e[1], e[2], o[3]] .+=
            0.25 .* (
                C(lo[1], lo[2], lo[3]) .+ C(hi[1], lo[2], lo[3]) .+
                C(lo[1], hi[2], lo[3]) .+ C(hi[1], hi[2], lo[3])
            )
        fine[e[1], o[2], e[3]] .+=
            0.25 .* (
                C(lo[1], lo[2], lo[3]) .+ C(hi[1], lo[2], lo[3]) .+
                C(lo[1], lo[2], hi[3]) .+ C(hi[1], lo[2], hi[3])
            )
        fine[o[1], e[2], e[3]] .+=
            0.25 .* (
                C(lo[1], lo[2], lo[3]) .+ C(lo[1], hi[2], lo[3]) .+
                C(lo[1], lo[2], hi[3]) .+ C(lo[1], hi[2], hi[3])
            )
        fine[e[1], e[2], e[3]] .+=
            0.125 .* (
                C(lo[1], lo[2], lo[3]) .+ C(hi[1], lo[2], lo[3]) .+
                C(lo[1], hi[2], lo[3]) .+ C(hi[1], hi[2], lo[3]) .+
                C(lo[1], lo[2], hi[3]) .+ C(hi[1], lo[2], hi[3]) .+
                C(lo[1], hi[2], hi[3]) .+ C(hi[1], hi[2], hi[3])
            )
    end
    return fine
end

function nas_mg_cycle!(u, r, rhs, c)
    finest = length(u)
    for level in finest:-1:2
        nas_mg_restrict!(r[level - 1], r[level])
    end
    fill!(u[1], 0.0)
    nas_mg_psinv!(u[1], r[1], c)
    for level in 2:(finest - 1)
        fill!(u[level], 0.0)
        nas_mg_interp!(u[level], u[level - 1])
        nas_mg_resid!(r[level], u[level], r[level])
        nas_mg_psinv!(u[level], r[level], c)
    end
    nas_mg_interp!(u[finest], u[finest - 1])
    nas_mg_resid!(r[finest], u[finest], rhs)
    nas_mg_psinv!(u[finest], r[finest], c)
    return nothing
end

function nas_mg_run!(u, r, rhs, p, c)
    foreach(x -> fill!(x, 0.0), u)
    nas_mg_resid!(r[end], u[end], rhs)
    for _ in 1:p.niter
        nas_mg_cycle!(u, r, rhs, c)
        nas_mg_resid!(r[end], u[end], rhs)
    end
    return r[end]
end

function nas_mg_norm(residual, p)
    interior = @view residual[2:(end - 1), 2:(end - 1), 2:(end - 1)]
    return sqrt(sum(abs2, interior) / Float64(prod(nas_mg_dims(p))))
end

# "skipped" for weak-scaling sizes that have no NAS reference.
function nas_mg_status(class::AbstractString, norm)
    nas_mg_parameters(class).norm === nothing && return "skipped"
    return nas_mg_verified(class, norm) ? "pass" : "fail"
end

function nas_mg_verified(class::AbstractString, norm; tolerance=1.0e-8)
    reference = nas_mg_parameters(class).norm
    return abs((norm - reference)/reference) <= tolerance
end
