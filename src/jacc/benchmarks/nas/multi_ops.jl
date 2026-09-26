# JACC.Multi backend for the multi-GPU NAS adapters. Kernels only use parts'
# linear indexing and `dev_id`/`ghost_dims`, so tests can swap in a CPU mock.
struct JACCMultiOps end

jm_ndev(::JACCMultiOps) = JACC.Multi.ndev()
jm_array(::JACCMultiOps, x; ghost_dims) = JACC.Multi.array(x; ghost_dims)
jm_for(::JACCMultiOps, n::Integer, f, args...) = JACC.Multi.parallel_for(n, f, args...)
jm_reduce(::JACCMultiOps, n::Integer, f, args...) = JACC.Multi.parallel_reduce(n, f, args...)
jm_parts(::JACCMultiOps, a) = a.a2
jm_sync!(::JACCMultiOps, a) = JACC.Multi.sync_ghost_elems!(a)
jm_to_host(::JACCMultiOps, a) = JACC.to_host(a)

# JACC.Multi.copy! only copies part i to part i on the same GPU; moving data
# between GPUs (besides neighbour ghosts) needs CUDA.jl, which uses peer access
# when available and stages through host memory otherwise.
function jm_copy!(::JACCMultiOps, dst, dd, doff, src, ds, soff, n)
    CUDA.device!(ds - 1)
    copyto!(dst, doff, src, soff, n)
    CUDA.synchronize()
    CUDA.device!(0)
    return dst
end

function jm_upload!(::JACCMultiOps, part, d, host, n)
    CUDA.device!(d - 1)
    copyto!(part, 1, host, 1, n)
    CUDA.synchronize()
    CUDA.device!(0)
    return part
end

# Runs `f(part, d)` with device `d` current, e.g. for per-device cuFFT calls.
function jm_each_part(f, ::JACCMultiOps, a)
    for (d, part) in enumerate(a.a2)
        CUDA.device!(d - 1)
        f(part, d)
    end
    for d in eachindex(a.a2)
        CUDA.device!(d - 1)
        CUDA.synchronize()
    end
    CUDA.device!(0)
    return a
end

# In-place cuFFT over `dims` of each part viewed as `shape` (JACC has no FFT).
# The inverse is unnormalized (bfft!).
function jm_fft!(ops::JACCMultiOps, a, shape, dims, inverse)
    return jm_each_part(ops, a) do part, _
        v = reshape(part, shape)
        inverse ? bfft!(v, dims) : fft!(v, dims)
    end
end
