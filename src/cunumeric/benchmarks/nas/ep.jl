# Each field of a batch partial has NDArray storage. The StructArray broadcast
# writes those fields on the GPU; timing ends at per-batch partials.

benchmark_backend_label(::NASEmbarrassinglyParallel, backend::String, default::String) =
    backend == "cunumeric" ? "cuNumeric (StructArray broadcast)" : default

benchmark_backend_save_as(::NASEmbarrassinglyParallel, backend::String, default::String) =
    backend == "cunumeric" ? "cunumeric_structarray" : default

struct CuNumericNASEPState{I,P}
    indices::I
    partials::P
end

function cunumeric_nas_ep_state(mod, n)
    indices = mod.NDArray(reshape(collect(Int64, 0:(n - 1)), n, 1))
    fields = ntuple(_ -> mod.zeros(Float64, n, 1), fieldcount(NASEPPartial))
    partials = StructArray{NASEPPartial}(NamedTuple{fieldnames(NASEPPartial)}(fields))
    return CuNumericNASEPState(indices, partials)
end

reset!(::NASEmbarrassinglyParallel, ::CuNumericNASEPState) = true

function run!(::NASEmbarrassinglyParallel, s::CuNumericNASEPState)
    s.partials .= nas_ep_batch.(s.indices, Ref(nas_ep_batch_jump()))
    return s.partials
end

function cleanup!(::NASEmbarrassinglyParallel, s::CuNumericNASEPState)
    foreach(cuNumeric.destroy!, Tuple(StructArrays.components(s.partials)))
    cuNumeric.destroy!(s.indices)
    return nothing
end

function initialize(b::NASEmbarrassinglyParallel{Float64}; mod=cuNumeric)
    p = validate_nas_ep(b)
    return (cunumeric_nas_ep_state(mod, nas_ep_batches(p)),)
end

function check_benchmark_correctness(
    b::NASEmbarrassinglyParallel, gs::GlobalSettings; mod=cuNumeric
)
    state = only(initialize(b; mod))
    try
        result = run!(b, state)
        sx = only(Array(sum(result.sx)))
        sy = only(Array(sum(result.sy)))
        names = fieldnames(NASEPPartial)[1:NAS_EP_NQ]
        q = [vec(Array(getproperty(result, name))) for name in names]
        samples = (1, 2, cld(length(q[1]), 2), length(q[1]))
        histogram_ok = all(samples) do i
            p = nas_ep_batch(i - 1)
            all(q[bin][i] == getfield(p, bin) for bin in 1:NAS_EP_NQ)
        end
        return histogram_ok && nas_ep_verified(b.class, sx, sy) ? "pass" : "fail"
    finally
        cleanup!(b, state)
    end
end
