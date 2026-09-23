# cuNumeric maps the shared scalar EP stream over batch indices. mapreduce
# needs a scalar result, so each histogram bin and the sx/sy pair is mapped
# separately. Every map, allocation, and reduction is included in timing.

struct CuNumericNASEPState{A,F}
    indices::A
    maps::F
end

function cunumeric_nas_ep_state(mod, n)
    indices = mod.NDArray(reshape(collect(Int64, 0:(n - 1)), n, 1))
    jump = nas_ep_batch_jump()
    bins = ntuple(NAS_EP_NQ) do bin
        i -> getfield(nas_ep_batch(i, jump), bin)
    end
    sums = let jump=jump
        i -> begin
            p = nas_ep_batch(i, jump)
            ComplexF64(p.sx, p.sy)
        end
    end
    return CuNumericNASEPState(indices, (bins..., sums))
end

reset!(::NASEmbarrassinglyParallel, ::CuNumericNASEPState) = true

function run!(::NASEmbarrassinglyParallel, s::CuNumericNASEPState)
    outputs = Any[]
    try
        for bin in 1:NAS_EP_NQ
            push!(outputs, mapreduce(s.maps[bin], +, s.indices; dims=2, init=0.0))
        end
        sums = cuNumeric.@allowpromotion mapreduce(
            s.maps[end], +, s.indices; dims=2, init=ComplexF64(0, 0)
        )
        push!(outputs, sums)
        return outputs
    catch
        foreach(cuNumeric.destroy!, outputs)
        rethrow()
    end
end

function cleanup_result!(
    ::NASEmbarrassinglyParallel, result, ::CuNumericNASEPState
)
    foreach(cuNumeric.destroy!, result)
    return nothing
end

function cleanup!(::NASEmbarrassinglyParallel, s::CuNumericNASEPState)
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
    result = nothing
    try
        reset!(b, state)
        result = run!(b, state)
        z = only(Array(sum(result[end])))
        sx, sy = real(z), imag(z)
        q = [Array(result[bin]) for bin in 1:NAS_EP_NQ]
        samples = (1, 2, cld(length(q[1]), 2), length(q[1]))
        histogram_ok = all(samples) do i
            p = nas_ep_batch(i - 1)
            all(q[bin][i] == getfield(p, bin) for bin in 1:NAS_EP_NQ)
        end
        return histogram_ok && nas_ep_verified(b.class, sx, sy) ? "pass" : "fail"
    finally
        !isnothing(result) && cleanup_result!(b, result, state)
        cleanup!(b, state)
    end
end
