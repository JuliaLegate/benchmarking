# CPU-only; run with the instantiated composability environment.
ENV["GKSwstype"] = "100"
using Test
include("../composability/krylov/plot_results.jl")
using .KrylovPlot: read_results, plot_results

@testset "Krylov plots" begin
    mktempdir() do tmp
        csv = joinpath(tmp, "results.csv")
        header = "experiment,base_n,backend,solver,mode,eltype,gpus,n,iterations,mean_ms,stderr_ms,median_ms,min_ms,max_ms,relative_residual,samples_ms\n"
        # Stale summary columns must not replace statistics computed from samples.
        row(mode, backend, gpus, n; precision="Float32", base="16", samples="1;3", solver="cg") =
            "$mode,$base,$backend,$solver,stock,$precision,$gpus,$n,2,999,999,999,999,999,0.001,$samples\n"
        for mode in ("single", "weak")
            rows = [row(mode, backend, g, n) for backend in ("Dagger", "cuNumeric", "cuNumeric local")
                for (g, n) in (mode == "single" ? [(1, 16), (1, 32)] : [(1, 16), (2, 23)])]
            write(csv, header * join(rows))
            results = read_results(mode, [csv])
            @test length(results) == 6
            @test all(r -> r.mean == 2.0 && r.stderr ≈ 1.0, results)
            for format in ("png", "svg")
                image = joinpath(tmp, "$mode.$format")
                plot_results(mode, [csv], image)
                @test filesize(image) > 1000
            end
            @test_throws ErrorException read_results(mode, [csv, csv])
            # Identical backend/size keys across solvers are independent curves.
            bicg_rows = [row(mode, backend, g, n; solver="bicgstab", samples="2;6")
                for backend in ("Dagger", "cuNumeric", "cuNumeric local")
                for (g, n) in (mode == "single" ? [(1, 16), (1, 32)] : [(1, 16), (2, 23)])]
            write(csv, header * join(rows) * join(bicg_rows))
            @test all(r -> r.solver == "cg" && r.mean == 2.0, read_results(mode, [csv]))
            results = read_results(mode, [csv]; solver="bicgstab")
            @test length(results) == 6
            @test all(r -> r.solver == "bicgstab" && r.mean == 4.0 && r.stderr ≈ 2.0, results)
            @test_throws ErrorException read_results(mode, [csv, csv]; solver="bicgstab")
            image = joinpath(tmp, "$mode-bicgstab.svg")
            KrylovPlot.main([mode, csv, "--solver=bicgstab", "--output", image])
            @test filesize(image) > 1000
            @test startswith(KrylovPlot.Plots.current()[1][:title], "BiCGStab")
            write(csv, header * join(bicg_rows))
            @test_throws ErrorException read_results(mode, [csv])
            @test length(read_results(mode, [csv]; solver="bicgstab")) == 6
        end
        for samples in ("1", "0;1", "NaN;1", "Inf;1")
            write(csv, header * row("weak", "Dagger", 1, 16; samples))
            @test_throws ErrorException read_results("weak", [csv])
        end
        for second in (row("weak", "Dagger", 2, 23; precision="Float64"),
            row("weak", "Dagger", 2, 23; base="32"), row("weak", "CUDA", 2, 23))
            write(csv, header * row("weak", "Dagger", 1, 16) * second)
            @test_throws ErrorException read_results("weak", [csv])
        end
        write(csv, header)
        @test_throws ErrorException read_results("weak", [csv])
        @test_throws ErrorException read_results("weak", [csv]; solver="bicgstab")
        @test_throws ErrorException read_results("weak", [csv]; solver="gmres")
    end
end
