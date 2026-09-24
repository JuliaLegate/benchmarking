# One Legate store holds the batch partials as fixed-size struct elements.

benchmark_backend_label(::NASEmbarrassinglyParallel, backend::String, default::String) =
    backend == "cunumeric" ? "cuNumeric (struct broadcast)" : default

benchmark_backend_save_as(::NASEmbarrassinglyParallel, backend::String, default::String) =
    backend == "cunumeric" ? "cunumeric_struct" : default

struct CuNumericNASEPState{I,P}
    indices::I
    partials::P
end

function cunumeric_nas_ep_state(mod, n)
    indices = mod.NDArray(reshape(collect(Int64, 0:(n - 1)), n, 1))
    partials = similar(indices, NASEPPartial, size(indices))
    return CuNumericNASEPState(indices, partials)
end

reset!(::NASEmbarrassinglyParallel, ::CuNumericNASEPState) = true

function run!(::NASEmbarrassinglyParallel, s::CuNumericNASEPState)
    s.partials .= nas_ep_batch.(s.indices, Ref(nas_ep_batch_jump()))
    return s.partials
end

function cleanup!(::NASEmbarrassinglyParallel, s::CuNumericNASEPState)
    cuNumeric.destroy!(s.partials)
    cuNumeric.destroy!(s.indices)
    return nothing
end

@inline cunumeric_nas_ep_field(p::NASEPPartial, ::Val{I}) where {I} = getfield(p, I)

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
        sx_field = cunumeric_nas_ep_field.(result, Ref(Val(11)))
        sy_field = cunumeric_nas_ep_field.(result, Ref(Val(12)))
        sx = fetch(sum(sx_field))
        sy = fetch(sum(sy_field))
        cuNumeric.destroy!(sx_field)
        cuNumeric.destroy!(sy_field)
        q = map(1:NAS_EP_NQ) do bin
            field = cunumeric_nas_ep_field.(result, Ref(Val(bin)))
            try
                return vec(Array(field))
            finally
                cuNumeric.destroy!(field)
            end
        end
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
