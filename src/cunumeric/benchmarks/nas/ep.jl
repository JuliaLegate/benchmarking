# cuNumeric's mapped reduction evaluates each 256-pair NPB stream locally.
# A singleton second axis makes mapreduce return one partial per stream. Its
# scalar outputs hold the ten histogram bins in two UInt64 values (9 bits/bin,
# since each batch has at most 256 accepted pairs) and sx/sy in ComplexF64.
# All three mappings and their output allocations are timed. The original
# array recurrence remains selectable for comparison, without custom tasks.

function cunumeric_nas_ep_impl()
    impl = get(ENV, "CUNUMERIC_NAS_EP_IMPL", "mapped")
    impl in ("mapped", "recurrence") || error(
        "CUNUMERIC_NAS_EP_IMPL must be mapped or recurrence; got '$impl'"
    )
    return impl
end

function benchmark_backend_label(
    ::NASEmbarrassinglyParallel, backend::String, default::String
)
    return backend == "cunumeric" ? "cuNumeric ($(cunumeric_nas_ep_impl()))" : default
end

function benchmark_backend_save_as(
    ::NASEmbarrassinglyParallel, backend::String, default::String
)
    return backend == "cunumeric" && cunumeric_nas_ep_impl() == "recurrence" ?
        "cunumeric_recurrence" : default
end

@inline function cunumeric_nas_ep_pack_low(p::NASEPPartial)
    return UInt64(p.q0) | UInt64(p.q1) << 9 | UInt64(p.q2) << 18 |
        UInt64(p.q3) << 27 | UInt64(p.q4) << 36 | UInt64(p.q5) << 45 |
        UInt64(p.q6) << 54
end

@inline function cunumeric_nas_ep_pack_high(p::NASEPPartial)
    return UInt64(p.q7) | UInt64(p.q8) << 9 | UInt64(p.q9) << 18
end

struct CuNumericNASEPMappedState{A,F}
    indices::A
    maps::F
end

function cunumeric_nas_ep_mapped_state(mod, n)
    indices = mod.NDArray(reshape(collect(Int64, 0:(n - 1)), n, 1))
    jump = nas_ep_batch_jump()
    low = let jump=jump
        i -> cunumeric_nas_ep_pack_low(nas_ep_batch(i, jump))
    end
    high = let jump=jump
        i -> cunumeric_nas_ep_pack_high(nas_ep_batch(i, jump))
    end
    sums = let jump=jump
        i -> begin
            p = nas_ep_batch(i, jump)
            ComplexF64(p.sx, p.sy)
        end
    end
    return CuNumericNASEPMappedState(indices, (low, high, sums))
end

reset!(::NASEmbarrassinglyParallel, ::CuNumericNASEPMappedState) = true

function run!(::NASEmbarrassinglyParallel, s::CuNumericNASEPMappedState)
    low, high, sums = s.maps
    a = mapreduce(low, +, s.indices; dims=2, init=UInt64(0))
    try
        b = mapreduce(high, +, s.indices; dims=2, init=UInt64(0))
        try
            c = cuNumeric.@allowpromotion mapreduce(
                sums, +, s.indices; dims=2, init=ComplexF64(0, 0)
            )
            return (a, b, c)
        catch
            cuNumeric.destroy!(b)
            rethrow()
        end
    catch
        cuNumeric.destroy!(a)
        rethrow()
    end
end

function cleanup_result!(
    ::NASEmbarrassinglyParallel, result, ::CuNumericNASEPMappedState
)
    foreach(cuNumeric.destroy!, result)
    return nothing
end

function cleanup!(::NASEmbarrassinglyParallel, s::CuNumericNASEPMappedState)
    cuNumeric.destroy!(s.indices)
    return nothing
end

mutable struct CuNumericNASEPState{V,M}
    values::V
    masks::M
end

function initialize(b::NASEmbarrassinglyParallel{Float64}; mod=cuNumeric)
    p = validate_nas_ep(b)
    n = nas_ep_batches(p)
    cunumeric_nas_ep_impl() == "mapped" && return (cunumeric_nas_ep_mapped_state(mod, n),)
    values = ntuple(_ -> mod.zeros(Float64, n), 13)
    masks = ntuple(p.m - NAS_EP_MK) do bit
        return mod.NDArray(Float64[((i >> (bit - 1)) & 1) for i in 0:(n - 1)])
    end
    return (CuNumericNASEPState(values, masks),)
end

function reset!(::NASEmbarrassinglyParallel, s::CuNumericNASEPState)
    fill!(first(s.values), NAS_EP_SEED)
    foreach(x -> fill!(x, 0.0), Base.tail(s.values))
    return true
end

function cleanup!(::NASEmbarrassinglyParallel, s::CuNumericNASEPState)
    foreach(cuNumeric.destroy!, s.values)
    foreach(cuNumeric.destroy!, s.masks)
    return nothing
end

function cunumeric_nas_ep_skip(seed, mask, multiplier)
    return cuNumeric.@accelerate let
        r23 = 2.0^-23
        t23 = 2.0^23
        r46 = 2.0^-46
        t46 = 2.0^46
        a1 = trunc(r23*multiplier)
        a2 = multiplier - t23*a1
        x1 = trunc.(r23 .* seed)
        x2 = seed .- t23 .* x1
        t1 = a1 .* x2 .+ a2 .* x1
        z = t1 .- t23 .* trunc.(r23 .* t1)
        t3 = t23 .* z .+ a2 .* x2
        candidate = t3 .- t46 .* trunc.(r46 .* t3)
        return seed .+ mask .* (candidate .- seed)
    end
end

function cunumeric_nas_ep_pair(values)
    seed, q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, sx, sy = values
    return cuNumeric.@accelerate let
        r23 = 2.0^-23
        t23 = 2.0^23
        r46 = 2.0^-46
        t46 = 2.0^46
        a1 = trunc(r23*NAS_EP_MULTIPLIER)
        a2 = NAS_EP_MULTIPLIER - t23*a1

        x11 = trunc.(r23 .* seed)
        x12 = seed .- t23 .* x11
        t11 = a1 .* x12 .+ a2 .* x11
        z1 = t11 .- t23 .* trunc.(r23 .* t11)
        t31 = t23 .* z1 .+ a2 .* x12
        seed1 = t31 .- t46 .* trunc.(r46 .* t31)
        u1 = r46 .* seed1

        x21 = trunc.(r23 .* seed1)
        x22 = seed1 .- t23 .* x21
        t12 = a1 .* x22 .+ a2 .* x21
        z2 = t12 .- t23 .* trunc.(r23 .* t12)
        t32 = t23 .* z2 .+ a2 .* x22
        seed2 = t32 .- t46 .* trunc.(r46 .* t32)
        u2 = r46 .* seed2

        gbase1 = 2.0 .* u1 .- 1.0
        gbase2 = 2.0 .* u2 .- 1.0
        radius = gbase1 .* gbase1 .+ gbase2 .* gbase2
        accepted = min.(floor.(1.0 ./ radius), 1.0)
        safe_radius = min.(radius, 1.0)
        scale = sqrt.(-2.0 .* log.(safe_radius) ./ safe_radius)
        g1 = gbase1 .* scale
        g2 = gbase2 .* scale
        magnitude = max.(abs.(g1), abs.(g2))
        bin = floor.(magnitude)

        nq0 = q0 .+ accepted .* max.(0.0, 1.0 .- abs.(bin .- 0.0))
        nq1 = q1 .+ accepted .* max.(0.0, 1.0 .- abs.(bin .- 1.0))
        nq2 = q2 .+ accepted .* max.(0.0, 1.0 .- abs.(bin .- 2.0))
        nq3 = q3 .+ accepted .* max.(0.0, 1.0 .- abs.(bin .- 3.0))
        nq4 = q4 .+ accepted .* max.(0.0, 1.0 .- abs.(bin .- 4.0))
        nq5 = q5 .+ accepted .* max.(0.0, 1.0 .- abs.(bin .- 5.0))
        nq6 = q6 .+ accepted .* max.(0.0, 1.0 .- abs.(bin .- 6.0))
        nq7 = q7 .+ accepted .* max.(0.0, 1.0 .- abs.(bin .- 7.0))
        nq8 = q8 .+ accepted .* max.(0.0, 1.0 .- abs.(bin .- 8.0))
        nq9 = q9 .+ accepted .* max.(0.0, 1.0 .- abs.(bin .- 9.0))
        nsx = sx .+ accepted .* g1
        nsy = sy .+ accepted .* g2
        return seed2, nq0, nq1, nq2, nq3, nq4, nq5, nq6, nq7, nq8, nq9, nsx, nsy
    end
end

function run!(b::NASEmbarrassinglyParallel, s::CuNumericNASEPState)
    jump = nas_ep_batch_jump()
    power = jump
    seed = first(s.values)
    for mask in s.masks
        next_seed = cunumeric_nas_ep_skip(seed, mask, power)
        cuNumeric.destroy!(seed)
        seed = next_seed
        power, _ = nas_ep_randlc(power, power)
    end
    values = (seed, Base.tail(s.values)...)
    for _ in 1:nas_ep_pairs_per_batch()
        next_values = cunumeric_nas_ep_pair(values)
        foreach(cuNumeric.destroy!, values)
        values = next_values
    end
    s.values = values
    return s.values
end

function check_benchmark_correctness(
    b::NASEmbarrassinglyParallel, gs::GlobalSettings; mod=cuNumeric
)
    state = only(initialize(b; mod))
    result = nothing
    try
        reset!(b, state)
        result = run!(b, state)
        if state isa CuNumericNASEPMappedState
            z = only(Array(sum(result[3])))
            sx, sy = real(z), imag(z)
            low, high = Array(result[1]), Array(result[2])
            samples = (1, 2, cld(length(low), 2), length(low))
            histogram_ok = all(samples) do i
                p = nas_ep_batch(i - 1)
                low[i] == cunumeric_nas_ep_pack_low(p) &&
                    high[i] == cunumeric_nas_ep_pack_high(p)
            end
        else
            sx, sy = only(Array(sum(result[12]))), only(Array(sum(result[13])))
            histogram_ok = true
        end
        return histogram_ok && nas_ep_verified(b.class, sx, sy) ? "pass" : "fail"
    finally
        !isnothing(result) && cleanup_result!(b, result, state)
        cleanup!(b, state)
    end
end
