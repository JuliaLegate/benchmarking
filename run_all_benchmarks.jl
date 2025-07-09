using BenchmarkTools
using DataFrames
using CSV
using Printf
using Pkg

function find_benchmarks(root::String)
    benchmarks = []
    for (dirpath, _, files) in walkdir(root)
        for file in files
            if occursin("cuda_vecadd.jl", file)
                push!(benchmarks, joinpath(dirpath, file))
            end
        end
    end
    return benchmarks
end

function activate_project(path::String)
    if isfile(joinpath(path, "Project.toml"))
        Pkg.activate(path; shared=false)
    else
        @warn "No Project.toml found in $path; skipping activation"
    end
end



function run_benchmark(script_path::String)
    mod = Module()
    path = dirname(script_path)
    activate_project(path)
    Base.include(mod, script_path)
    if isdefined(mod, :bench)
        return mod.bench()
    elseif isdefined(mod, :main)
        return mod.main()
    else
        error("No `bench` or `main` function in $script_path")
    end
end

function summarize(trial::BenchmarkTools.Trial)
    return (
        minimum = minimum(trial),
        median = median(trial),
        mean = mean(trial),
        allocations = trial.allocs,
        memory = trial.memory
    )
end

# === Main ===

results = DataFrame(
    Script = String[],
    Min_ns = Float64[],
    Median_ns = Float64[],
    Mean_ns = Float64[],
    Allocations = Int[],
    Memory_bytes = Int[]
)

root = "."
benchmarks = find_benchmarks(root)

for bench_path in benchmarks
   trial = run_benchmark(bench_path)
   stats = summarize(trial)
   push!(results, (
            bench_path,
            stats.minimum,
            stats.median,
            stats.mean,
            stats.allocations,
            stats.memory
   ))
end

CSV.write("benchmark_results.csv", results)
println("✅ Benchmark results written to benchmark_results.csv")

