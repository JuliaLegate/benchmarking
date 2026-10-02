function wait_for_darray(array)
    foreach(wait, array.chunks)
    return array
end

using TOML

# Tuned chunks per GPU, keyed by (benchmark, gpus, N, M, class): the fastest passing
# blocks_per_gpu in tunes/*.csv (written by tune_dagger.sh; a split's latest row wins).
# Points never tuned use 1.
function load_dagger_tunes(dir=normpath(joinpath(@__DIR__, "..", "..", "tunes")))
    row = r"^[^,\n]+,([^,]+),[^,]+,(\d+),(\d+),(\d+),(\"(?:[^\"]|\"\")*\"|[^,]*),(\d+),\d+,\d+,([\d.]+),pass$"m
    times = Dict{Tuple{String,Int,Int,Int,String},Dict{Int,Float64}}()
    isdir(dir) || return Dict{keytype(times),Int}()
    for file in filter(endswith(".csv"), readdir(dir; join=true)), m in eachmatch(row, read(file, String))
        name, N, M, gpus, kwargs, split, ms = m.captures
        kwargs = startswith(kwargs, '"') ? replace(kwargs[2:end-1], "\"\"" => "\"") : kwargs
        class = string(get(TOML.parse(kwargs), "class", ""))
        key = (name, parse(Int, gpus), parse(Int, N), parse(Int, M), class)
        get!(times, key, Dict{Int,Float64}())[parse(Int, split)] = parse(Float64, ms)
    end
    return Dict(key => first(argmin(last, collect(splits))) for (key, splits) in times)
end
const DAGGER_BLOCKS_PER_GPU = load_dagger_tunes()

# Chunks per GPU along each partitioned dimension: kwarg `blocks_per_gpu`
# (tune.jl sweeps it), else the tuned table, else one contiguous chunk per GPU.
dagger_tune_key(config) =
    (config.name, config.gpus, config.N, config.M, string(get(config.kwargs, :class, "")))
function dagger_blocks_per_gpu(config)
    tuned = get(DAGGER_BLOCKS_PER_GPU, dagger_tune_key(config), 1)
    return Int(get(config.kwargs, :blocks_per_gpu, tuned))
end
function dagger_blocks_per_gpu_source(config)
    haskey(config.kwargs, :blocks_per_gpu) && return "kwarg"
    return haskey(DAGGER_BLOCKS_PER_GPU, dagger_tune_key(config)) ? "tuned" : "default, not in tunes/"
end

# Owner of chunk i of n: consecutive chunks share a GPU, in processor order.
dagger_owner(processors, i, n) = processors[cld(i * length(processors), n)]

# Datadeps scheduler (kwarg `scheduler`). Greedy keeps tasks on their data; RoundRobin doesn't.
const DAGGER_DATADEPS_SCHEDULERS = Dict(
    "greedy" => Dagger.GreedyScheduler,
    "roundrobin" => Dagger.RoundRobinScheduler,
)
function dagger_datadeps_scheduler(config)
    name = string(get(config.kwargs, :scheduler, "greedy"))
    haskey(DAGGER_DATADEPS_SCHEDULERS, name) || error(
        "Unknown Dagger Datadeps scheduler '$name'; known: " *
        join(sort!(collect(keys(DAGGER_DATADEPS_SCHEDULERS))), ", "),
    )
    return DAGGER_DATADEPS_SCHEDULERS[name]()
end
with_dagger_scheduler(f, scheduler) = Dagger.with(f, Dagger.DATADEPS_SCHEDULER => scheduler)

# DArray finalizers only queue releases on MemPool; GC until that queue drains.
function dagger_release_memory()
    queue = Dagger.MemPool.SEND_QUEUE
    for _ in 1:4
        GC.gc(true)
        deadline = time() + 10
        while (isready(queue.queue) || queue.processing) && time() < deadline
            sleep(0.01)
        end
    end
    return
end
