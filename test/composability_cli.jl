# CPU-only: julia --startup-file=no test/composability_cli.jl
using Test
include("../run_composability.jl")
using .ComposabilityCLI: cli_options, launch_plan, main, run_launcher, WORKLOADS, DEFAULT_CONFIG

@testset "Launcher output and exit status" begin
    # Real child processes catch the output suppression hidden by executor mocks.
    for code in (0, 7)
        mktemp() do _, out
            mktemp() do _, err
                passed = redirect_stdout(out) do
                    redirect_stderr(err) do
                        script = "println(\"launcher progress\"); println(stderr, \"launcher error\"); exit($code)"
                        run_launcher(`$(Base.julia_cmd()) --startup-file=no -e $script`)
                    end
                end
                @test passed == (code == 0)
                seekstart(out)
                seekstart(err)
                @test occursin("launcher progress", read(out, String))
                @test occursin("launcher error", read(err, String))
            end
        end
    end
end

@testset "Composability CLI" begin
    mktempdir() do depot
        # Simulate a startup.jl adding a depot not present in the environment.
        pushfirst!(DEPOT_PATH, depot)
        try
            launch = first(launch_plan(cli_options(["--only=krylov"])))
            child = Cmd(`$(Base.julia_cmd()) --startup-file=no -e 'print(first(DEPOT_PATH))'`;
                env=launch.cmd.env)
            @test read(child, String) == depot
        finally
            popfirst!(DEPOT_PATH)
        end
    end
    @test cli_options(String[]).workloads == WORKLOADS
    @test cli_options(String[]).mode == "single"
    @test cli_options(["--mode=multi"]).gpus == ["1", "2", "4", "8"]
    for args in (["--only="], ["--only=missing"], ["--only=krylov,krylov"],
        ["--mode=weak"], ["--gpus=2"], ["--mode=multi", "--gpus=3"],
        ["--mode=multi", "--gpus=1,1"], ["--gpus="], ["--output="], ["--config="], ["--unknown"])
        @test_throws ArgumentError cli_options(args)
    end

    mktempdir() do tmp
        output = joinpath(tmp, "results with spaces")
        args = ["--only=krylov,ordinarydiffeq", "--mode=both", "--gpus=1,4", "--output=$output"]
        plan = launch_plan(cli_options(args); env=Dict("CUNUMERIC_BENCH_JULIA" => "/my julia/bin/julia"))
        @test length(plan) == 4
        @test [p.workload for p in plan] == ["krylov", "ordinarydiffeq", "krylov", "ordinarydiffeq"]
        @test basename(plan[1].cmd.exec[2]) == "run.sh"
        @test plan[1].cmd.exec[3:end] == ["single", "1024", "2048", "4096", "8192", "16384", "32768", "65536"]
        @test plan[2].cmd.exec[3:end] == ["single", "128", "512", "1024", "2048", "4096", "8192", "16384"]
        @test plan[3].cmd.exec[3:end] == ["weak", "65536", "1", "4"]
        @test plan[4].cmd.exec[3:end] == ["weak", "16384", "1", "4"]
        for launch in plan
            prefix = launch.workload == "krylov" ? "BENCH" : "ODE"
            @test isfile(launch.cmd.exec[2])
            @test "$(prefix)_OUTPUT=$(joinpath(output, launch.mode, launch.workload))" in launch.cmd.env
            @test "JULIA=/my julia/bin/julia" in launch.cmd.env
            @test "$(prefix)_DRY_RUN=0" in launch.cmd.env
            @test "JULIA_DEPOT_PATH=$(join(DEPOT_PATH, Sys.iswindows() ? ';' : ':'))" in launch.cmd.env
        end
        all_plan = launch_plan(cli_options(["--mode=both"]))
        @test length(all_plan) == 6
        @test all_plan[3].cmd.exec[3:end] == ["single", "32", "128", "512", "1024", "2048", "4096", "6144", "8192"]
        @test all_plan[6].cmd.exec[3:end] == ["weak", "8192", "1", "2", "4", "8"]
        @test any(startswith("INTOPT_OUTPUT="), all_plan[3].cmd.env)
        custom = joinpath(tmp, "custom sizes.toml")
        custom_args = ["--only=krylov", "--mode=both", "--config=$custom"]
        write(custom, "[krylov]\nsingle = [16, 32]\nweak_base = 32\n")
        custom_plan = launch_plan(cli_options(custom_args))
        @test custom_plan[1].cmd.exec[3:end] == ["single", "16", "32"]
        @test custom_plan[2].cmd.exec[3:end] == ["weak", "32", "1", "2", "4", "8"]
        @test cli_options(["--config=$custom"]).config == custom
        for body in ("single = []\nweak_base = 32", "single = [32, 16]\nweak_base = 32",
            "single = [16, 16]\nweak_base = 32", "single = [1]\nweak_base = 32",
            "single = [16]\nweak_base = -1", "single = [16]\nweak_base = true",
            "single = [16.0]\nweak_base = 32", "single = [16]\nbase = 32")
            write(custom, "[krylov]\n$body\n")
            @test_throws ArgumentError launch_plan(cli_options(custom_args))
        end
        write(custom, "[ordinarydiffeq]\nsingle = [16]\nweak_base = 16\n")
        @test_throws ArgumentError launch_plan(cli_options(custom_args))
        @test_throws SystemError launch_plan(cli_options(["--config=$(joinpath(tmp, "missing.toml"))"]))
        chosen = launch_plan(cli_options(args); env=Dict("JULIA" => "preferred", "CUNUMERIC_BENCH_JULIA" => "other"))
        @test "JULIA=preferred" in chosen[1].cmd.env

        calls = Cmd[]
        function executor(cmd)
            push!(calls, cmd)
            if length(calls) == 1
                directory = split(only(filter(startswith("BENCH_OUTPUT="), cmd.env)), '='; limit=2)[2]
                mkpath(directory)
                write(joinpath(directory, "planned-cases.csv"), "backend,mode\n")
                write(joinpath(directory, "environment.txt"), "startup failure details\n")
                return false
            end
            return true
        end
        preview = IOBuffer()
        @test main(vcat(args, ["--dry-run"]); executor, io=preview) == 0
        @test occursin("BENCH_OUTPUT=", String(take!(preview)))
        @test isempty(calls)
        @test !ispath(output)
        @test main(["--help"]; executor, io=IOBuffer()) == 0
        @test isempty(calls)
        failure_output = IOBuffer()
        @test main(args; executor, io=failure_output) == 1
        message = String(take!(failure_output))
        @test occursin("FAILED: krylov single", message)
        @test occursin("startup failure details", message)
        @test length(calls) == 4 # A failure does not suppress other workloads or modes.
        @test readlines(joinpath(output, "multi", "weak_scaling_plan.csv")) == [
            "workload,base_n,n_1,n_2,n_4,n_8",
            "krylov,65536,65536,92682,131072,185364",
            "ordinarydiffeq,16384,16384,23170,32768,46341",
        ]
        for mode in ("single", "multi")
            @test read(joinpath(output, mode, "sizes.toml"), String) == read(DEFAULT_CONFIG, String)
        end
        empty!(calls)
        @test main(args; executor=cmd -> true, io=IOBuffer()) == 0

        # Check both output modes before launching either one.
        existing = joinpath(output, "multi", "ordinarydiffeq", "results.csv")
        mkpath(dirname(existing))
        write(existing, "retained results\n")
        @test_throws ArgumentError main(args; executor, io=IOBuffer())
        @test isempty(calls)
        @test read(existing, String) == "retained results\n"
    end
end
