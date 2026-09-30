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
