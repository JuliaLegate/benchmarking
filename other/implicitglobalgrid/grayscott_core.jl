# Array-generic local work, adapted from julia-con/models/diffeq/grayscott.jl.
# Physical cells include the outer border used by src/benchmarks/grayscott.jl.
# Two extra communication halos on each side let a border read the opposite
# second physical cell, preserving that reference's previous-step boundary copy.

function process_grid(n, ranks)
    n >= 4 && ranks >= 1 || error("N must be >= 4 and GPUS must be positive")
    for px in isqrt(ranks):-1:1
        ranks % px == 0 || continue
        py = ranks ÷ px
        if n % px == 0 && n % py == 0 && min(n ÷ px, n ÷ py) >= 2
            return (px, py)
        end
    end
    error("N=$n cannot be divided into $ranks equal tiles with at least two cells per axis")
end

function local_layout(n, dims, coords)
    owned = (n ÷ dims[1], n ÷ dims[2])
    offset = (coords[1] * owned[1], coords[2] * owned[2])
    physical = (3:(owned[1] + 2), 3:(owned[2] + 2))
    interior = ntuple(2) do d
        lo = first(physical[d]) + (coords[d] == 0)
        hi = last(physical[d]) - (coords[d] == dims[d] - 1)
        lo:hi
    end
    lower = ntuple(d -> (first(interior[d]) - 1):(last(interior[d]) - 1), 2)
    upper = ntuple(d -> (first(interior[d]) + 1):(last(interior[d]) + 1), 2)
    return (; n, owned, offset, physical, interior, lower, upper,
        low=(coords[1] == 0, coords[2] == 0),
        high=(coords[1] == dims[1] - 1, coords[2] == dims[2] - 1))
end

gs_parameters() = (; dx=1.0f0, dt=0.2f0, c_u=1.0f0, c_v=0.3f0, f=0.03f0, k=0.06f0)

function seed_patch(n; deterministic=false)
    s = min(150, n)
    if deterministic
        u = Float32[0.5f0 + 0.5f0 * sin(Float32(i)) * cos(Float32(j)) for i in 1:s, j in 1:s]
        v = Float32[0.25f0 + 0.25f0 * cos(Float32(i)) * sin(Float32(j)) for i in 1:s, j in 1:s]
    else
        # Every rank constructs the same small patch, independent of tiling.
        rng = Random.Xoshiro(1234)
        u, v = rand(rng, Float32, s, s), rand(rng, Float32, s, s)
    end
    return u, v
end

function initial_fields(array, layout; deterministic=false)
    nx, ny = layout.owned .+ 4
    u = fill!(array{Float32}(undef, nx, ny), 1.0f0)
    v = fill!(similar(u), 0.0f0)
    pu, pv = seed_patch(layout.n; deterministic)
    ranges = ntuple(d -> 1:min(layout.owned[d], size(pu, d) - layout.offset[d]), 2)
    if all(!isempty, ranges)
        local_indices = ntuple(d -> (first(ranges[d]) + 2):(last(ranges[d]) + 2), 2)
        seed_indices = ntuple(d -> ranges[d] .+ layout.offset[d], 2)
        u[local_indices...] .= array(pu[seed_indices...])
        v[local_indices...] .= array(pv[seed_indices...])
    end
    return u, v, zero(u), zero(v)
end

function workspace(u, layout)
    shape = length.(layout.interior)
    return ntuple(_ -> similar(u, shape), 4)
end

@views function local_step!(u, v, u_new, v_new, work, layout, p=gs_parameters())
    i, j = layout.interior
    im, jm = layout.lower
    ip, jp = layout.upper
    F_u, F_v, lap_u, lap_v = work
    F_u .= (-u[i, j] .* (v[i, j] .* v[i, j])) .+ p.f .* (1.0f0 .- u[i, j])
    F_v .= (u[i, j] .* (v[i, j] .* v[i, j])) .- (p.f + p.k) .* v[i, j]
    lap_u .= (u[ip, j] .- 2 .* u[i, j] .+ u[im, j]) ./ p.dx^2 .+
             (u[i, jp] .- 2 .* u[i, j] .+ u[i, jm]) ./ p.dx^2
    lap_v .= (v[ip, j] .- 2 .* v[i, j] .+ v[im, j]) ./ p.dx^2 .+
             (v[i, jp] .- 2 .* v[i, j] .+ v[i, jm]) ./ p.dx^2
    u_new[i, j] .= ((p.c_u .* lap_u) .+ F_u) .* p.dt .+ u[i, j]
    v_new[i, j] .= ((p.c_v .* lap_v) .+ F_v) .* p.dt .+ v[i, j]

    # Match the reference's assignment order too: row copies overwrite corners.
    x, y = layout.physical
    for (a, b) in ((u, u_new), (v, v_new))
        layout.low[2] && (b[x, first(y)] .= a[x, first(y) - 2])
        layout.high[2] && (b[x, last(y)] .= a[x, last(y) + 2])
        layout.low[1] && (b[first(x), y] .= a[first(x) - 2, y])
        layout.high[1] && (b[last(x), y] .= a[last(x) + 2, y])
    end
    return nothing
end
