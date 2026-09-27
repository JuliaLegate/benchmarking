# JACC.Multi NAS adapters on a CPU mock of JACC.Multi with simulated devices.
include(joinpath(@__DIR__, "jacc_multi_mock.jl"))
include(joinpath(@__DIR__, "..", "src", "jacc", "benchmarks", "nas", "mg_multi.jl"))
include(joinpath(@__DIR__, "..", "src", "jacc", "benchmarks", "nas", "ft_multi.jl"))
include(joinpath(@__DIR__, "..", "src", "jacc", "benchmarks", "grayscott_multi.jl"))

@testset "JACC.Multi Gray-Scott on simulated devices" begin
    p = grayscott_gs_params(Float64)
    # Uneven splits, and M=6 on 4 devices puts both halos at device boundaries.
    for (N, M) in ((16, 13), (9, 32), (5, 6)), nd in (1, 2, 3, 4, 8)
        cld(M + 2, nd) >= 2 || continue
        u0, v0 = grayscott_host_init(Float64, N, M; deterministic=true)
        s = jacc_multi_grayscott_state(MockOps(nd), u0, v0)
        foreach(_ -> gsm_step!(MockOps(nd), s, p), 1:10)
        cu, cv = grayscott_cpu_steps(Float64, u0, v0, 10, p)
        @test gsm_to_host(MockOps(nd), s.u, s.L) ≈ cu
        @test gsm_to_host(MockOps(nd), s.v, s.L) ≈ cv
    end
    @test_throws ErrorException gs_layout(4, 2, 4)
end

@testset "JACC.Multi MG on simulated devices" begin
    p = nas_mg_parameters("S")
    for nd in (1, 2, 4, 8)
        s = jacc_multi_mg(MockOps(nd), "S")
        @test nas_mg_verified("S", sqrt(mgm_run!(s, "S")/Float64(p.n)^3))
    end
    # Several slab levels: restriction/interpolation between slabs.
    layouts = mg_multi_plan(nas_mg_level_sizes(nas_mg_parameters("B")), 8)
    slabs = filter(L -> !L.rep, layouts)
    @test length(slabs) >= 2
    @test all(slabs[i + 1].P == 2slabs[i].P for i in 1:(length(slabs) - 1))
    @test all(8L.P >= L.n for L in slabs)
end

@testset "JACC.Multi FT on simulated devices" begin
    for nd in (1, 2, 4, 8)
        @test nas_ft_verified("S", ftm_run!(jacc_multi_ft(MockOps(nd), "S")))
    end
    @test_throws ErrorException FTLayout(nas_ft_parameters("W"), 64)
end
