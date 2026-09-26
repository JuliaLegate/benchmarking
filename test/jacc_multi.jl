# JACC.Multi NAS adapters on a CPU mock of JACC.Multi with simulated devices.
include(joinpath(@__DIR__, "jacc_multi_mock.jl"))
include(joinpath(@__DIR__, "..", "src", "jacc", "benchmarks", "nas", "mg_multi.jl"))

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
