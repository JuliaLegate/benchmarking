# Sweep blocks_per_gpu for any Dagger benchmark on every visible GPU; set the
# winner as `blocks_per_gpu` in that benchmark's config kwargs. Every run is
# appended to tunes/<name>.csv (e.g. tunes/nas-ft.csv), or tunes/<name>-<tag>.csv
# when DAGGER_TUNE_TAG is set (tune_dagger.sh uses "strong" for strong scaling).
#   LD_LIBRARY_PATH="" julia --project=environments/dagger src/dagger/tune.jl \
#       <name> <T> <N> <M> [kwargs TOML]
# e.g. grayscott Float32 24000 24000
#      nas_ft Float64 512 256 'class = "B"'
# run_benchmark.sh sets this when tune_dagger.sh launches us.
get!(ENV, "CUNUMERIC_BENCH_ACTIVE_MODEL", "dagger")
include(joinpath(@__DIR__, "single.jl"))

using Dates
using Printf
using TOML

length(ARGS) in (4, 5) || error("usage: tune.jl <name> <T> <N> <M> [kwargs TOML]")
const NAME, T_NAME = ARGS[1], ARGS[2]
const T = Dict("Float32" => Float32, "Float64" => Float64)[T_NAME]
const N, M = parse(Int, ARGS[3]), parse(Int, ARGS[4])
const KWARGS_TOML = length(ARGS) == 5 ? ARGS[5] : ""
const KWARGS = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in TOML.parse(KWARGS_TOML))
# The count run_benchmark.sh was asked for; each benchmark's build rejects a mismatch.
const GPUS = parse(Int, get(ENV, "CUNUMERIC_BENCH_GPUS", string(length(CUDA.devices()))))
const SPLITS = [1, 2, 4, 8, 16, 32, 64]
# Chunks smaller than a 2048×2048 tile are never faster, whatever the dimensionality.
const MIN_CHUNK = 2048^2
# Stop once a split is this much slower than the best; smaller blocks only get worse.
const GIVE_UP = 8.0
const TAG = get(ENV, "DAGGER_TUNE_TAG", "")
const CSV = normpath(joinpath(
    @__DIR__, "..", "..", "tunes", join(filter(!isempty, [replace(NAME, "_" => "-"), TAG]), "-") * ".csv"
))

# (chunk count, elements per chunk). 1-D ranges and 3-D z-slabs split one
# dimension into parts = gpus * split; 2-D tiles split both, giving parts^2.
split_1d(n, parts, plane=1) = (b = cld(n, parts); (cld(n, b), plane * b))
split_2d(n, m, parts) = (b = cld(n, parts); (cld(n, b)^2, b * min(b, m)))
tune_chunk(b, parts) = split_1d(b.N, parts)
tune_chunk(b::DaggerMonteCarlo, parts) = split_1d(b.n_samples, parts)
tune_chunk(b::DaggerGrayScott, parts) = split_2d(b.N, b.M, parts)
tune_chunk(b::DaggerGEMM, parts) = split_2d(b.N, b.N, parts)
tune_chunk(b::DaggerNASFT, parts) =
    (p = nas_ft_parameters(b.class); split_1d(p.nz, parts, p.nx * p.ny))
tune_chunk(b::DaggerNASMG, parts) =
    ((nx, ny, nz) = nas_mg_dims(nas_mg_parameters(b.class)); split_1d(nz, parts, nx * ny))

function tune_config(split)
    kwargs = merge(KWARGS, Dict{Symbol,Any}(:blocks_per_gpu => split))
    # gpus name T T_name N M n_iter n_warmup n_trial check n_correctness_iter flops
    return ModelWorkerConfig(GPUS, NAME, T, T_NAME, N, M, 10, 2, 1, true, 1, 0.0, kwargs)
end

const HEADER = "timestamp,name,T,N,M,gpus,kwargs,blocks_per_gpu,chunks,chunk_elements,ms_per_iter,correctness"
isfile(CSV) && readline(CSV) != HEADER &&
    error("$CSV has a different header; move it aside before tuning")

function record(split, chunks, chunk, ms, correctness)
    kwargs = isempty(KWARGS_TOML) ? "" : "\"" * replace(KWARGS_TOML, "\"" => "\"\"") * "\""
    mkpath(dirname(CSV))
    new = !isfile(CSV)
    open(CSV, "a") do io
        new && println(io, HEADER)
        @printf(
            io, "%s,%s,%s,%d,%d,%d,%s,%d,%d,%d,%.6f,%s\n",
            now(), NAME, T_NAME, N, M, GPUS, kwargs, split, chunks, chunk, ms, correctness,
        )
    end
end

results = Pair{Int,Float64}[]
for split in SPLITS
    config = tune_config(split)
    benchmark = model_build_benchmark(config)
    chunks, chunk = tune_chunk(benchmark, GPUS * split)
    split > 1 && chunk < MIN_CHUNK && break
    correctness = model_check_correctness(benchmark, config)
    ms, _ = model_trial(benchmark, config)
    @printf(
        "blocks_per_gpu=%3d  chunks=%5d  chunk=%10d  %10.3f ms/iter  correctness=%s\n",
        split, chunks, chunk, ms, correctness,
    )
    record(split, chunks, chunk, ms, correctness)
    correctness == "fail" && error("blocks_per_gpu=$split failed correctness")
    push!(results, split => ms)
    benchmark = nothing
    GC.gc()
    ms > GIVE_UP * minimum(last, results) && break
end

best = argmin(last, results)
println(
    "best: $NAME blocks_per_gpu = $(first(best))  " *
    "($(round(last(best); digits=3)) ms/iter, N=$N, M=$M, gpus=$GPUS) -> $CSV",
)
