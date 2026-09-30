using Test, Random
include(joinpath(@__DIR__, "..", "other", "implicitglobalgrid", "grayscott_core.jl"))
include(joinpath(@__DIR__, "..", "other", "implicitglobalgrid", "reference.jl"))

@testset "ImplicitGlobalGrid Gray-Scott matches the existing step" begin
    @test process_grid(28000, 4) == (2, 2)
    @test process_grid(79200, 8) == (2, 4)
    @test_throws ErrorException process_grid(31, 2)
    @test_throws ErrorException process_grid(4, 8)
    @test_throws ErrorException process_grid(3, 1)
    @test_throws ErrorException process_grid(32, 0)
    for n in (4, 12, 32, 160), ranks in (1, 2, 4, 8), deterministic in (false, true)
        n == 4 && ranks == 8 && continue
        dims = process_grid(n, ranks)
        layouts = [local_layout(n, dims, (x, y))
                   for x in 0:(dims[1] - 1), y in 0:(dims[2] - 1)]
        states = [initial_fields(Array, l; deterministic) for l in layouts]
        works = [workspace(first(st), l) for (st, l) in zip(states, layouts)]
        u, v = ones(Float32, n, n), zeros(Float32, n, n)
        pu, pv = seed_patch(n; deterministic)
        s = size(pu, 1)
        u[1:s, 1:s], v[1:s, 1:s] = pu, pv
        un, vn = zero(u), zero(v)
        for step in 0:5
            # Assemble only owned cells: communication halos never count as work.
            actual_u, actual_v = similar(u), similar(v)
            for (st, l) in zip(states, layouts)
                indices = ntuple(d -> (l.offset[d] + 1):(l.offset[d] + l.owned[d]), 2)
                actual_u[indices...] = st[1][l.physical...]
                actual_v[indices...] = st[2][l.physical...]
            end
            @test actual_u ≈ u rtol=1f-6 atol=1f-6
            @test actual_v ≈ v rtol=1f-6 atol=1f-6
            step == 5 && continue
            # Simulate periodic two-cell MPI halos, using the distributed result
            # (not the reference) so numerical errors propagate across steps.
            for (st, l) in zip(states, layouts)
                indices = ntuple(d -> mod1.((l.offset[d] - 1):(l.offset[d] + l.owned[d] + 2), n), 2)
                st[1] .= actual_u[indices...]
                st[2] .= actual_v[indices...]
            end
            for k in eachindex(states)
                a, b, an, bn = states[k]
                local_step!(a, b, an, bn, works[k], layouts[k])
                states[k] = (an, bn, a, b)
            end
            GrayScottReference.step!(u, v, un, vn, gs_parameters())
            u, un = un, u
            v, vn = vn, v
        end
    end
end
