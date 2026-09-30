function wait_for_darray(array)
    foreach(wait, array.chunks)
    return array
end

# Chunks per GPU along each partitioned dimension (kwarg `blocks_per_gpu`,
# swept by tune.jl). 1 gives one contiguous chunk per GPU.
dagger_blocks_per_gpu(config) = Int(get(config.kwargs, :blocks_per_gpu, 1))

# Owner of chunk i of n: consecutive chunks share a GPU, in processor order.
dagger_owner(processors, i, n) = processors[cld(i * length(processors), n)]
