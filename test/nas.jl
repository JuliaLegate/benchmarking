@testset "NAS EP contract" begin
    @test NAS_EP_NPB_GPU_COMMIT == "3f12d84920ee315ab00ef283717c1e74b68f4d00"
    @test Set(keys(NAS_EP_CLASSES)) == Set(["S", "W", "A", "B", "C", "D", "E", "B.2", "C.8", "C.4", "C.16"])
    # Weak-scaling classes keep class B's samples per GPU and have no NAS reference.
    @test [nas_ep_parameters(c).m for c in ("B", "B.2", "C", "C.8")] == [30, 31, 32, 33]
    @test nas_ep_status("B.2", 0.0, 0.0) == "skipped"
    @test nas_ep_status("S", nas_ep_parameters("S").sx, nas_ep_parameters("S").sy) == "pass"
    b = NASEmbarrassinglyParallel{Float64}(; N=33_554_432, M=1, class="S")
    p = validate_nas_ep(b)
    @test p == nas_ep_parameters("s")
    @test p.m == 24
    @test nas_ep_batches(p) == 65_536
    @test nas_ep_pairs_per_batch() == 256
    @test total_flops(b) == 33_554_432.0
    @test total_space(b) == 15_204_352
    @test nas_ep_verified("S", p.sx, p.sy)
    @test !nas_ep_verified("S", 1.01p.sx, p.sy)
    first_batch = nas_ep_batch(0)
    @test first_batch.q0 == 97
    @test first_batch.q1 == 86
    @test first_batch.sx ≈ -9.836968663257105
    class_s = nas_ep_combine(
        nas_ep_batch(i) for i in 0:(nas_ep_batches(p) - 1)
    )
    @test class_s.q == [6_140_517, 5_865_300, 1_100_361, 68_546, 1_648, 17, 0, 0, 0, 0]
    @test nas_ep_verified("S", class_s.sx, class_s.sy)
    @test_throws ErrorException validate_nas_ep(
        NASEmbarrassinglyParallel{Float32}(; N=33_554_432, M=1, class="S")
    )
    @test all(
        supports_benchmark(execution_model(model), "nas_ep") for
        model in (:cunumeric, :cupynumeric, :cudajl, :jacc, :dagger)
    )
    @test supports_run(execution_model(:jacc), "nas_ep", 2)
    @test supports_run(execution_model(:dagger), "nas_ep", 2)

    config = joinpath(@__DIR__, "..", "configs", "single_gpu", "nas_ep.toml")
    settings, specs = parse_config(config)
    runs = plan_runs(
        specs, settings, TOML.parsefile(config), parse_plot_groups(config), 10^12
    )
    @test Set(r.model for r in runs) ==
        Set([:cunumeric, :cupynumeric, :cudajl, :jacc, :dagger])
    @test all(runs) do r
        p = nas_ep_parameters(get(r.spec.kwargs, :class, "S"))
        (r.N, r.M) == (nas_ep_random_numbers(p), 1) && r.spec.n_iter == 10
    end
end

@testset "NAS FT contract" begin
    @test NAS_FT_NPB_GPU_COMMIT == "3f12d84920ee315ab00ef283717c1e74b68f4d00"
    @test Set(keys(NAS_FT_CLASSES)) == Set(["S", "W", "A", "B", "C", "D", "E", "A.2", "B.4", "B.8", "B.2", "C.2", "C.4"])
    # Weak-scaling classes keep class A's grid points and iterations per GPU.
    for (g, c) in ((1, "A"), (2, "A.2"), (4, "B.4"), (8, "B.8"))
        q = nas_ft_parameters(c)
        @test (q.nx*q.ny*q.nz ÷ g, q.niter) == (256*256*128, 6)
    end
    @test NAS_FT_CHECKSUMS["B.4"] == NAS_FT_CHECKSUMS["B"][1:6]
    @test nas_ft_status("A.2", ComplexF64[]) == "skipped"
    b = NASFourierTransform{Float64}(; N=64, M=64, class="S")
    p = validate_nas_ft(b)
    @test p == nas_ft_parameters("s")
    @test p.nz == 64 && p.niter == 6
    @test total_flops(b) ≈ 1.7716695575300533e8
    @test total_space(b) == 16_777_216
    @test length(NAS_FT_CHECKSUMS["S"]) == p.niter
    @test length(nas_ft_checksum_indices(p)) == NAS_FT_CHECKSUM_SAMPLES
    @test sum(nas_ft_checksum_mask(p)) == NAS_FT_CHECKSUM_SAMPLES
    initial = Array{ComplexF64}(undef, 2, 2, 1)
    nas_ft_initial_conditions!(initial)
    @test initial[1] ≈ 0.7945219111887383 + 0.8690652738745399im
    for cls in ("S", "W")
        p = nas_ft_parameters(cls)
        reference = nas_ft_initial_conditions(p)
        fast = similar(reference)
        scratch = Vector{UInt64}(undef, 1 << 16)
        nas_ft_initial_conditions_uint64!(fast, scratch)
        @test fast == reference
    end
    @test_throws ErrorException validate_nas_ft(
        NASFourierTransform{Float32}(; N=64, M=64, class="S")
    )
    @test all(
        supports_benchmark(execution_model(model), "nas_ft") for
        model in (:cunumeric, :cupynumeric, :cudajl, :jacc, :dagger)
    )
    @test supports_run(execution_model(:jacc), "nas_ft", 2)

    config = joinpath(@__DIR__, "..", "configs", "single_gpu", "nas_ft.toml")
    settings, specs = parse_config(config)
    runs = plan_runs(
        specs, settings, TOML.parsefile(config), parse_plot_groups(config), 10^12
    )
    @test Set(r.model for r in runs) ==
        Set([:cunumeric, :cupynumeric, :cudajl, :jacc, :dagger])
    @test all(runs) do r
        p = nas_ft_parameters(get(r.spec.kwargs, :class, "S"))
        (r.N, r.M) == (p.nx, p.ny) && r.spec.n_iter == 10
    end
end

@testset "NAS MG contract" begin
    @test NAS_MG_NPB_GPU_COMMIT == "3f12d84920ee315ab00ef283717c1e74b68f4d00"
    @test Set(keys(NAS_MG_CLASSES)) == Set(["S", "W", "A", "B", "C", "D", "E", "S.2", "B.2", "B.4", "C.2", "C.4"])
    # Weak-scaling classes keep their base class's points per GPU and coarsen
    # every axis; cube classes keep their original levels.
    for (g, c) in ((1, "B"), (2, "B.2"), (4, "B.4"), (8, "C"))
        @test prod(nas_mg_dims(nas_mg_parameters(c))) ÷ g == 256^3
    end
    @test nas_mg_level_shapes(nas_mg_parameters("B.2"))[1] == (4, 4, 6)
    @test first.(nas_mg_level_shapes(nas_mg_parameters("B"))) == [2^l + 2 for l in 1:8]
    @test nas_mg_status("B.2", 0.0) == "skipped"
    @test nas_mg_smoother("S.2") == nas_mg_smoother("S")
    b = NASMultiGrid{Float64}(; N=32, M=32, class="S")
    p = validate_nas_mg(b)
    @test p == nas_mg_parameters("s")
    @test p.niter == 4
    @test nas_mg_level_shapes(p) == [(n, n, n) for n in (4, 6, 10, 18, 34)]
    @test total_flops(b) == 7_602_176.0
    @test total_space(b) == 1_057_088

    rhs = nas_mg_rhs(p)
    interior = @view rhs[2:(end - 1), 2:(end - 1), 2:(end - 1)]
    @test count(!iszero, interior) == 2NAS_MG_EXTREMA
    @test sum(interior) == 0.0
    shapes = nas_mg_level_shapes(p)
    u = [zeros(Float64, shape) for shape in shapes]
    r = [zeros(Float64, shape) for shape in shapes]
    residual = nas_mg_run!(u, r, rhs, p, nas_mg_smoother("S"))
    norm = nas_mg_norm(residual, p)
    @test norm ≈ p.norm rtol=1.0e-12
    @test nas_mg_verified("S", norm)
    @test_throws ErrorException validate_nas_mg(
        NASMultiGrid{Float32}(; N=32, M=32, class="S")
    )
    @test all(
        supports_benchmark(execution_model(model), "nas_mg") for
        model in (:cunumeric, :cupynumeric, :cudajl, :jacc, :dagger)
    )
    @test supports_run(execution_model(:jacc), "nas_mg", 2)
    @test supports_run(execution_model(:dagger), "nas_mg", 2)

    config = joinpath(@__DIR__, "..", "configs", "single_gpu", "nas_mg.toml")
    settings, specs = parse_config(config)
    runs = plan_runs(
        specs, settings, TOML.parsefile(config), parse_plot_groups(config), 10^12
    )
    @test Set(r.model for r in runs) ==
        Set([:cunumeric, :cupynumeric, :cudajl, :jacc, :dagger])
    @test all(runs) do r
        p = nas_mg_parameters(get(r.spec.kwargs, :class, "S"))
        (r.N, r.M) == nas_mg_dims(p)[1:2] && r.spec.n_iter == 10
    end
end

@testset "NAS weak-scaling configs plan" begin
    for (name, classes) in (
        ("nas_ft_weak", ["A", "A.2", "B.4", "B.8"]),
        ("nas_ep_weak", ["B", "B.2", "C", "C.8"]),
        ("nas_mg_weak", ["B", "B.2", "B.4", "C"]),
    )
        config = joinpath(@__DIR__, "..", "configs", "multi_gpu", "$name.toml")
        settings, specs = parse_config(config)
        runs = plan_runs(
            specs, settings, TOML.parsefile(config), parse_plot_groups(config), 10^12
        )
        @test Set(r.model for r in runs) ==
            Set([:cunumeric, :cupynumeric, :cudajl, :jacc, :dagger])
        @test Set((r.spec.gpus, r.spec.kwargs[:class]) for r in runs) ==
            Set(zip([1, 2, 4, 8], classes))
    end
end

@testset "NAS classes supply N and M" begin
    mktempdir() do dir
        for (name, class, expected, auto) in (
            ("nas_ep", "B", [2^31, 1], true),
            ("nas_ft", "B", [512, 256], false),
            ("nas_mg", "S", [32, 32], true),
        )
            path = joinpath(dir, "$name.toml")
            write(path, """
            [Global]
            n_warmup = 1
            n_iter = 1
            auto_size = $auto
            models = ["cunumeric"]

            [[$name]]
            T = "Float64"
            gpus = 1
            cpus = 1
            kwargs = { class = "$class" }
            """)
            spec = only(last(parse_config(path)))
            @test spec.args == expected
            @test !spec.autosize
            validate_spec(spec)
        end
    end
end

@testset "NAS class sweeps" begin
    mktempdir() do dir
        path = joinpath(dir, "weak.toml")
        write(path, """
        [Global]
        n_warmup = 1
        n_iter = 1
        models = ["cunumeric"]

        [[nas_mg]]
        T = "Float64"
        gpus = [1, 2, 4]
        cpus = 1
        kwargs = { class = ["S", "W", "A"] }
        """)
        specs = last(parse_config(path))
        @test [s.gpus for s in specs] == [1, 2, 4]
        @test [s.kwargs[:class] for s in specs] == ["S", "W", "A"]
        @test [s.args for s in specs] == [[32, 32], [128, 128], [256, 256]]
        # One sweep writes one results directory and therefore one figure.
        @test length(unique(results_subdir(s) for s in specs)) == 1

        write(path, replace(read(path, String), "[1, 2, 4]" => "[1, 2]"))
        @test_throws ErrorException parse_config(path)
        write(path, replace(read(path, String),
            "class = [\"S\", \"W\", \"A\"]" => "class = \"S\", implementation = [\"a\"]"))
        @test_throws ErrorException parse_config(path)
    end
end
