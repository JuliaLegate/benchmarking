# NPB permits changing MK without changing the generated sequence. This uses
# the harness-wide MK=8 so every programming model performs the same batching.
# CUDA.jl remains the intentionally single-GPU baseline.
# LIMITATION: Timing ends at per-stream histogram/sum partials, not a global
# reduction, consistently across models. See nas/README.md for the contract.

struct CUDANASEPState{A,I}
    partials::A
    indices::I
end

function initialize(b::CUDANASEP{Float64}; mod=CUDA)
    batches = nas_ep_batches(validate_nas_ep(b))
    return (CUDANASEPState(
        CUDA.CuArray{NASEPPartial}(undef, batches), CUDA.CuArray(collect(Int64, 0:batches-1))
    ),)
end

function run!(b::CUDANASEP, s::CUDANASEPState)
    s.partials .= nas_ep_batch.(s.indices, Ref(nas_ep_batch_jump()))
    return s.partials
end

function check_benchmark_correctness(
    b::CUDANASEP, gs::GlobalSettings; mod=CUDA
)
    state = only(initialize(b; mod))
    result = nas_ep_combine(Array(run!(b, state)))
    return nas_ep_status(b.class, result.sx, result.sy)
end
