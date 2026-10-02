function wait_for_darray(array)
    foreach(wait, array.chunks)
    return array
end

# Tuned chunks per GPU, keyed by (benchmark, gpus). Copy winners from the
# tunes/*.csv files written by tune_dagger.sh; anything not listed uses 1.
const DAGGER_BLOCKS_PER_GPU = Dict{Tuple{String,Int},Int}(
    # ("grayscott", 2) => 2,
)

# Chunks per GPU along each partitioned dimension: kwarg `blocks_per_gpu`
# (tune.jl sweeps it), else the tuned table, else one contiguous chunk per GPU.
function dagger_blocks_per_gpu(config)
    tuned = get(DAGGER_BLOCKS_PER_GPU, (config.name, config.gpus), 1)
    return Int(get(config.kwargs, :blocks_per_gpu, tuned))
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
