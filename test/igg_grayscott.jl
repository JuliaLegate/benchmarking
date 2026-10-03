# Standalone CPU validation; no CUDA, MPI, or benchmark environment required.
module IGGGrayScottTests
using Test, Random
include("../other/implicitglobalgrid/grayscott_core.jl")

# Independent periodic array reference (the JACC/Dagger forward-Euler equations).
function reference_step(u, v, p)
    lap(a) = (circshift(a, (1, 0)) + circshift(a, (-1, 0)) +
              circshift(a, (0, 1)) + circshift(a, (0, -1)) - 4a) / p.dx2
    uvv = u .* v .* v
    return u .+ p.dt .* (p.cu .* lap(u) .- uvv .+ p.f .* (1 .- u)),
           v .+ p.dt .* (p.cv .* lap(v) .+ uvv .- (p.f + p.k) .* v)
end

function assemble(parts, dims, N)
    out = zeros(Float32, N, N)
    nx, ny = N ÷ dims[1], N ÷ dims[2]
    for cy in 0:dims[2]-1, cx in 0:dims[1]-1
        out[cx*nx+1:(cx+1)*nx, cy*ny+1:(cy+1)*ny] .=
            @view parts[cx+1, cy+1][2:end-1, 2:end-1]
    end
    return out
end

# Fill ghosts from a snapshot of the global state, including periodic wraps.
function exchange!(parts, global_state, dims, N)
    nx, ny = N ÷ dims[1], N ÷ dims[2]
    for cy in 0:dims[2]-1, cx in 0:dims[1]-1
        local_state = parts[cx+1, cy+1]
        for j in 1:ny+2, i in 1:nx+2
            if i == 1 || i == nx+2 || j == 1 || j == ny+2
                local_state[i, j] = global_state[mod1(cx*nx+i-1, N), mod1(cy*ny+j-1, N)]
            end
        end
    end
end

@testset "IGG periodic fused Gray-Scott" begin
    p = (dt=0.2f0, dx2=1.0f0, cu=1.0f0, cv=0.3f0, f=0.03f0, k=0.06f0)
    rng = MersenneTwister(42)
    # Tiny domains, patches crossing ranks, and ranks entirely outside the patch.
    for N in (12, 160, 320), dims in ((1, 1), (2, 1), (2, 2), (4, 2))
        seed = min(150, N)
        su, sv = rand(rng, Float32, seed, seed), rand(rng, Float32, seed, seed)
        u, v = ones(Float32, N, N), zeros(Float32, N, N)
        u[1:seed, 1:seed], v[1:seed, 1:seed] = su, sv
        nx, ny = N ÷ dims[1] + 2, N ÷ dims[2] + 2
        us = [ones(Float32, nx, ny) for _ in 1:dims[1], _ in 1:dims[2]]
        vs = [zeros(Float32, nx, ny) for _ in 1:dims[1], _ in 1:dims[2]]
        uns = [fill(NaN32, nx, ny) for _ in 1:dims[1], _ in 1:dims[2]]
        vns = deepcopy(uns)
        for cy in 0:dims[2]-1, cx in 0:dims[1]-1
            igg_seed!(us[cx+1, cy+1], vs[cx+1, cy+1], (cx, cy, 0), su, sv)
        end
        @test assemble(us, dims, N) == u
        @test assemble(vs, dims, N) == v
        exchange!(us, u, dims, N)
        exchange!(vs, v, dims, N)
        for _ in 1:5
            old_u, old_v = deepcopy(us), deepcopy(vs)
            for rank in eachindex(us), j in 2:ny-1, i in 2:nx-1
                igg_update_cell!(i, j, us[rank], vs[rank], uns[rank], vns[rank], p)
            end
            @test us == old_u && vs == old_v
            actual_u, actual_v = assemble(uns, dims, N), assemble(vns, dims, N)
            u, v = reference_step(u, v, p)
            @test isapprox(actual_u, u; rtol=1f-5, atol=1f-6)
            @test isapprox(actual_v, v; rtol=1f-5, atol=1f-6)
            exchange!(uns, actual_u, dims, N)
            exchange!(vns, actual_v, dims, N)
            us, uns = uns, us
            vs, vns = vns, vs
        end
    end
end
end # module
