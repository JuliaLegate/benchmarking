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

# Datadeps scheduler for a benchmark's regions (kwarg `scheduler`). Dagger's
# default, RoundRobin, ignores where data lives: it matched owner-computes only
# when a region's task order happened to line up with the processor order
# (GEMM row-major on any GPU count, elementwise broadcasts on 2 GPUs), and
# otherwise moved every operand to another GPU and back (CG's in-place
# broadcasts: 3.8 ms on 2 GPUs, 170 ms on 4). Greedy's cost model keeps each
# task on its data.
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

# A dropped DArray is not freed by the collection itself: its chunks' and
# thunks' finalizers only queue releases on MemPool's work queue, which a task
# drains afterwards. Collecting and then allocating the next trial right away
# left the previous trial's arrays alive alongside it -- at auto-sized
# Gray-Scott on 4 GPUs, the near-full device pool then slowed every later
# trial by 40% (71 -> 101 ms/step). Collect until the queue is idle; each
# round's releases can unroot more objects for the next.
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
