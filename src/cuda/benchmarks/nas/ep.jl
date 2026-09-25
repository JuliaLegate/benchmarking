# NPB permits changing MK without changing the generated sequence. This uses
# the harness-wide MK=8 so every programming model performs the same batching.
# CUDA.jl remains the intentionally single-GPU baseline.
# LIMITATION: Timing ends at per-stream histogram/sum partials, not a global
# reduction, consistently across models. See nas/README.md for the contract.

function cuda_nas_ep_impl(b::NASEmbarrassinglyParallel)
    impl = b.implementation == "default" ? get(ENV, "CUDA_NAS_EP_IMPL", "kernel") :
           b.implementation
    impl in ("kernel", "broadcast") || error("CUDA_NAS_EP_IMPL must be kernel or broadcast")
    return impl
end

function benchmark_backend_label(b::NASEmbarrassinglyParallel, backend::String, default::String)
    return backend == "cudajl" ? "CUDA.jl ($(cuda_nas_ep_impl(b)))" : default
end

function benchmark_backend_save_as(b::NASEmbarrassinglyParallel, backend::String, default::String)
    return backend == "cudajl" && cuda_nas_ep_impl(b) == "broadcast" ?
        "CUDA.jl_broadcast" : default
end

struct CUDANASEPState{A,I}
    partials::A
    indices::I
end

function initialize(b::NASEmbarrassinglyParallel{Float64}; mod=CUDA)
    p = validate_nas_ep(b)
    batches = nas_ep_batches(p)
    indices = cuda_nas_ep_impl(b) == "broadcast" ?
        CUDA.CuArray(collect(Int64, 0:batches-1)) : nothing
    return (CUDANASEPState(CUDA.CuArray{NASEPPartial}(undef, batches), indices),)
end

function cuda_nas_ep_kernel!(partials, jump)
    i = (CUDA.blockIdx().x - 1)*CUDA.blockDim().x + CUDA.threadIdx().x
    i <= length(partials) && (@inbounds partials[i] = nas_ep_batch(i - 1, jump))
    return nothing
end

function run!(b::NASEmbarrassinglyParallel, s::CUDANASEPState)
    if s.indices !== nothing
        jump = nas_ep_batch_jump()
        s.partials .= nas_ep_batch.(s.indices, Ref(jump))
        return s.partials
    end
    threads = 256
    CUDA.@cuda threads=threads blocks=cld(length(s.partials), threads) cuda_nas_ep_kernel!(
        s.partials, nas_ep_batch_jump()
    )
    return s.partials
end

function check_benchmark_correctness(
    b::NASEmbarrassinglyParallel, gs::GlobalSettings; mod=CUDA
)
    state = only(initialize(b; mod))
    result = nas_ep_combine(Array(run!(b, state)))
    return nas_ep_verified(b.class, result.sx, result.sy) ? "pass" : "fail"
end
