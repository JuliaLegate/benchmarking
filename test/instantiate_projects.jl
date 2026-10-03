# Integration test: runs the real setup script in fresh project directories.
# CUNUMERIC_SOURCE=/path/to/cuNumeric.jl julia --startup-file=no test/instantiate_projects.jl
module SetupIntegration

using Test, TOML

const ROOT = dirname(@__DIR__)
const PROJECTS = [".", "environments/cuda", "environments/jacc",
    "environments/dagger", "environments/implicitglobalgrid",
    "environments/cunumeric", "environments/composability"]

function stage_projects(destination)
    cp(joinpath(ROOT, "instantiate_projects.sh"), joinpath(destination, "instantiate_projects.sh"))
    igg_scripts = joinpath(destination, "other", "implicitglobalgrid")
    mkpath(igg_scripts)
    cp(joinpath(ROOT, "other", "implicitglobalgrid", "setup_igg.sh"), joinpath(igg_scripts, "setup_igg.sh"))
    for project in PROJECTS
        target = joinpath(destination, project)
        mkpath(target)
        cp(joinpath(ROOT, project, "Project.toml"), joinpath(target, "Project.toml"))
    end
end

function verify_projects(workspace, source)
    for project in PROJECTS
        @testset "$project" begin
            directory = joinpath(workspace, project)
            manifest_path = joinpath(directory, "Manifest.toml")
            @test isfile(manifest_path)
            isfile(manifest_path) || continue
            manifest = TOML.parsefile(manifest_path)
            @test VersionNumber(manifest["julia_version"]) == VERSION
            dependencies = manifest["deps"]
            declared = TOML.parsefile(joinpath(directory, "Project.toml"))["deps"]
            for (name, uuid) in declared
                @test haskey(dependencies, name)
                haskey(dependencies, name) || continue
                @test only(dependencies[name])["uuid"] == uuid
            end
            if haskey(declared, "Dagger")
                dagger = only(dependencies["Dagger"])
                @test dagger["version"] == "0.22.5"
                # Read the original source declaration: setup must neither erase
                # it from the staged project nor resolve to a registry release.
                expected = TOML.parsefile(joinpath(ROOT, project, "Project.toml"))["sources"]["Dagger"]
                actual_sources = get(TOML.parsefile(joinpath(directory, "Project.toml")), "sources", Dict())
                @test get(actual_sources, "Dagger", nothing) == expected
                @test get(dagger, "repo-url", nothing) == expected["url"]
                @test get(dagger, "repo-rev", nothing) == expected["rev"]
                @test !haskey(dagger, "path")
            end
            if haskey(declared, "cuNumeric")
                for (name, expected) in ("cuNumeric" => source,
                    "CNPreferences" => joinpath(source, "lib", "CNPreferences"))
                    path = only(dependencies[name])["path"]
                    @test realpath(joinpath(directory, path)) == realpath(expected)
                end
            end
        end
    end
end

function save_diagnostics(workspace, output)
    for project in PROJECTS
        directory = joinpath(workspace, project)
        isdir(directory) || continue
        target = joinpath(output, project == "." ? "orchestrator" : project)
        mkpath(target)
        for file in readdir(directory)
            endswith(file, ".toml") || continue
            cp(joinpath(directory, file), joinpath(target, file); force=true)
        end
    end
end

function main()
    Sys.islinux() || error("The setup integration test requires Linux (cuNumeric GPU artifacts).")
    source = abspath(get(ENV, "CUNUMERIC_SOURCE", joinpath(ROOT, "..")))
    isfile(joinpath(source, "Project.toml")) &&
        isfile(joinpath(source, "lib", "CNPreferences", "Project.toml")) ||
        error("Set CUNUMERIC_SOURCE to a cuNumeric.jl checkout including lib/CNPreferences.")
    output = abspath(get(ENV, "SETUP_TEST_OUTPUT", joinpath(ROOT, "results", "instantiate-tests")))
    julia = joinpath(Sys.BINDIR, Base.julia_exename())

    @testset "Real environment setup (Julia $VERSION)" begin
        mktempdir() do workspace
            stage_projects(workspace)
            cmd = addenv(`bash $(joinpath(workspace, "instantiate_projects.sh"))`,
                "CUNUMERIC_SOURCE" => source,
                "CUNUMERIC_BENCH_JULIA" => julia,
                "IGG_MPI_PREFIX" => joinpath(workspace, "igg-mpi"),
                "CONDA_OVERRIDE_CUDA" => get(ENV, "IGG_CUDA_VERSION", get(ENV, "CUDA_VERSION_MAJOR_MINOR", "13.0")),
                # Exercise resolution, downloads, builds, and JACC backend setup
                # on a CPU runner, without starting the GPU runtime or eager precompilation.
                "LEGATE_SKIP_RUNTIME" => "true",
                "LEGATE_AUTO_CONFIG" => "0",
                "JULIA_PKG_PRECOMPILE_AUTO" => "0",
            )
            try
                println("Running instantiate_projects.sh with Julia $VERSION in $workspace")
                # success(cmd) discards child output; keep resolver/build errors
                # visible in the CI log and its uploaded instantiate.log.
                passed = success(pipeline(cmd; stdout, stderr))
                @test passed
                if passed
                    verify_projects(workspace, source)
                    # Exercise actual headless plotting using the shared packages,
                    # without loading GPU backends or running a benchmark.
                    plot_test = `$julia --startup-file=no --project=$(joinpath(workspace, "environments/composability")) $(joinpath(ROOT, "test/krylov_plot.jl"))`
                    @test success(pipeline(plot_test; stdout, stderr))
                end
            finally
                save_diagnostics(workspace, output)
                println("Setup diagnostics: $output")
            end
        end
    end
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    SetupIntegration.main()
end
