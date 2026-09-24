# Dagger partitions independent EP streams across its requested CUDA scope.
# A DArray broadcast applies the common scalar EP function to each stream;
# there is no hand-written CUDA kernel. The correctness pass gathers partials
# after timing and checks the official verification sums.
# LIMITATION: Timing includes task completion but excludes global aggregation,
# consistently with the other EP adapters.

include(joinpath(@__DIR__, "..", "..", "..", "nas", "ep.jl"))

struct DaggerNASEP{S,P}
    class::String
    N::Int
    batches::Int
    gpus::Int
    scope::S
    processors::P
end

struct DaggerNASEPState{P,I}
    partials::P
    indices::I
end

function model_build_nas_ep(config::ModelWorkerConfig)
    config.T === Float64 || error("NAS EP requires Float64")
    config.M == 1 || error("NAS EP requires M=1")
    class = uppercase(string(get(config.kwargs, :class, "S")))
    p = nas_ep_parameters(class)
    expected = nas_ep_random_numbers(p)
    config.N == expected || error("NAS EP class $class requires N=$expected")
    available = length(collect(CUDA.devices()))
    available == config.gpus || error(
        "Dagger sees $available GPU(s), but this run requested $(config.gpus)"
    )
    scope = Dagger.scope(; cuda_gpus=collect(1:config.gpus))
    processors = sort!(collect(Dagger.compatible_processors(scope)); by=string)
    length(processors) == config.gpus || error("Dagger CUDA processor count mismatch")
    return DaggerNASEP(
        class, config.N, nas_ep_batches(p), config.gpus, scope, processors
    )
end

function model_initialize(b::DaggerNASEP)
    block = cld(b.batches, b.gpus)
    return Dagger.with_options(; scope=b.scope) do
        partials = Dagger.DArray(
            fill(NASEPPartial(), b.batches), Dagger.Blocks(block), b.processors
        )
        indices = Dagger.DArray(
            collect(Int64, 0:(b.batches - 1)), Dagger.Blocks(block), b.processors
        )
        foreach(wait_for_darray, (partials, indices))
        partial_chunks = map(task -> fetch(task; raw=true), partials.chunks)
        length(partial_chunks) == b.gpus || error("Dagger EP chunk count mismatch")
        processors = Dagger.processor.(partial_chunks)
        length(unique(processors)) == b.gpus || error(
            "Dagger did not place one EP chunk on each requested GPU"
        )
        return DaggerNASEPState(partials, indices)
    end
end

function model_run!(b::DaggerNASEP, s::DaggerNASEPState)
    Dagger.with_options(; scope=b.scope) do
        s.partials .= nas_ep_batch.(s.indices, Ref(nas_ep_batch_jump()))
    end
    return s.partials
end

model_synchronize(::DaggerNASEP) = Dagger.gpu_synchronize(:CUDA)
model_throughput_label(::DaggerNASEP) = "G random numbers/s"

function model_check_correctness(b::DaggerNASEP, config)
    state = model_initialize(b)
    model_run!(b, state)
    model_synchronize(b)
    partials = nas_ep_combine(collect(state.partials))
    return nas_ep_verified(b.class, partials.sx, partials.sy) ? "pass" : "fail"
end

function model_correctness_context(b::DaggerNASEP, config)
    return (; reference="NPB-GPU", dims=(b.N, 1))
end
