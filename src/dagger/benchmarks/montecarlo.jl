struct DaggerMonteCarlo{T,S,P}
    n_samples::Int
    gpus::Int
    blocks_per_gpu::Int
    scope::S
    processors::P
end

struct DaggerMonteCarloState{A,C,S}
    samples::A
    chunks::C
    scopes::S
end

function model_build_montecarlo(config::ModelWorkerConfig)
    available = length(collect(CUDA.devices()))
    available == config.gpus || error(
        "Dagger sees $available GPU(s), but this run was planned for $(config.gpus). " *
        "Set CUDA_VISIBLE_DEVICES to exactly the selected devices.",
    )
    scope = Dagger.scope(; cuda_gpus=collect(1:config.gpus))
    processors = sort!(collect(Dagger.compatible_processors(scope)); by=string)
    length(processors) == config.gpus || error(
        "Dagger CUDA scope contains $(length(processors)) processor(s), " *
        "expected $(config.gpus)",
    )
    return DaggerMonteCarlo{config.T,typeof(scope),typeof(processors)}(
        config.N, config.gpus, dagger_blocks_per_gpu(config), scope, processors
    )
end

dagger_montecarlo_block(b::DaggerMonteCarlo, n) = cld(n, b.gpus * b.blocks_per_gpu)

function dagger_montecarlo_assignment(b::DaggerMonteCarlo, n)
    nchunks = cld(n, dagger_montecarlo_block(b, n))
    return [dagger_owner(b.processors, i, nchunks) for i in 1:nchunks]
end

function dagger_montecarlo_state(samples, expected_gpus)
    chunks = map(samples.chunks) do task
        return fetch(task; raw=true)
    end
    all(chunk -> Dagger.chunktype(chunk) <: CUDA.CuArray, chunks) || error(
        "Dagger Monte Carlo chunks must be resident CUDA arrays"
    )
    processors = Dagger.processor.(chunks)
    length(unique(processors)) == expected_gpus || error(
        "Dagger did not place Monte Carlo chunks on every requested GPU"
    )
    scopes = Dagger.ExactScope.(processors)
    return DaggerMonteCarloState(samples, chunks, scopes)
end

@inline function dagger_montecarlo_integrand(x)
    return exp(-(x*x))
end

function dagger_montecarlo_chunk_sum(samples)
    return mapreduce(
        dagger_montecarlo_integrand, +, samples; init=zero(eltype(samples))
    )
end

dagger_montecarlo_scale!(x, s) = (x .*= s; nothing)

function model_initialize(benchmark::DaggerMonteCarlo{T}) where {T}
    n = benchmark.n_samples
    return Dagger.with_options(; scope=benchmark.scope) do
        samples = rand(
            Dagger.Blocks(dagger_montecarlo_block(benchmark, n)), T, n;
            assignment=dagger_montecarlo_assignment(benchmark, n),
        )
        wait_for_darray(samples)
        state = dagger_montecarlo_state(samples, benchmark.gpus)
        # Scale to [0, 10) in place on each chunk's GPU to avoid extra copies.
        foreach(wait, map(zip(state.chunks, state.scopes)) do (chunk, scope)
            Dagger.@spawn scope=scope dagger_montecarlo_scale!(chunk, T(10))
        end)
        return state
    end
end

function model_run!(benchmark::DaggerMonteCarlo{T}, state::DaggerMonteCarloState) where {T}
    partials = map(zip(state.chunks, state.scopes)) do (chunk, scope)
        Dagger.@spawn scope=scope dagger_montecarlo_chunk_sum(chunk)
    end
    total = mapreduce(fetch, +, partials; init=zero(T))
    return (T(10) / benchmark.n_samples) * total
end

function dagger_montecarlo_correctness_state(benchmark::DaggerMonteCarlo, host_samples)
    n = length(host_samples)
    return Dagger.with_options(; scope=benchmark.scope) do
        samples = Dagger.DArray(
            host_samples, Dagger.Blocks(dagger_montecarlo_block(benchmark, n)),
            dagger_montecarlo_assignment(benchmark, n),
        )
        wait_for_darray(samples)
        return dagger_montecarlo_state(samples, benchmark.gpus)
    end
end

model_synchronize(::DaggerMonteCarlo) = Dagger.gpu_synchronize(:CUDA)
function model_correctness_context(benchmark::DaggerMonteCarlo, config)
    return (; reference="CPU", dims=(min(benchmark.n_samples, 1024), 1))
end

function model_check_correctness(benchmark::DaggerMonteCarlo{T}, config) where {T}
    n = min(benchmark.n_samples, 1024)
    host_samples = montecarlo_correctness_samples(T, n)
    state = dagger_montecarlo_correctness_state(benchmark, host_samples)
    check_benchmark = DaggerMonteCarlo{
        T,typeof(benchmark.scope),typeof(benchmark.processors)
    }(
        n, benchmark.gpus, benchmark.blocks_per_gpu, benchmark.scope, benchmark.processors
    )
    actual = model_run!(check_benchmark, state)
    model_synchronize(check_benchmark)
    expected = montecarlo_correctness_reference(host_samples)
    return montecarlo_correctness_status(actual, expected, T)
end
