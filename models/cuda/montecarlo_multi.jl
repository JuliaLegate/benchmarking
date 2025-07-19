using Pkg
Pkg.activate(@__DIR__)
using Distributed, CUDA, BenchmarkTools
addprocs(length(devices()))
@everywhere using CUDA, Random, Statistics


@everywhere function mc_integration(N)
    x = CUDA.rand(Float32, N)
    return mean(exp.(x.*x))
end

function total_flops(N)
    return N # missing prefactor
end

function integrand(x)
    return exp(x^2)
end

function do_integration(N)
    x = (5.0f0 .* CUDA.rand(Float32, N)) .- 10.0f0
    return integrand.(x)
end

function do_distributed_embaressingly_parallel_work(N_total, work_fn)
    @everywhere N_dev = length(devices())
    futures = asyncmap((zip(workers(), devices()))) do (p, d)
        remotecall_wait(p) do
            # @info "Worker $p uses $d"
            device!(d)
            N = Int(div(N_total, N_dev))
            work_fn(N)
        end
    end
    return fetch.(futures)
end

function monte_carlo_integration(N, n_samples, n_warmup)

    @assert mod(N, length(devices())) == 0

    work_fn = (N) -> do_integration(N)

    for s in 1:n_warmup
        y = do_distributed_embaressingly_parallel_work(N, work_fn)
    end

    t = CUDA.@elapsed begin
        for s in 1:n_samples
            y = do_distributed_embaressingly_parallel_work(N, work_fn)
        end
    end

    total_time_μs = t * 1e6
    mean_time_ms = total_time_μs / (n_samples * 1e3)
    gflops = total_flops(N) / (mean_time_ms * 1e6) # GFLOP is 1e9
  
    return mean_time_ms, gflops

end

gpus = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
n_samples = parse(Int, ARGS[3])
n_warmup = parse(Int, ARGS[4])

println("[CUDA.jl]  Monte-Carlo integration benchmark on $(N) elements for $(n_samples) iterations, $(n_warmup) warmups on $(gpus) GPUs")

mean_time_ms, gflops = monte_carlo_integration(N, n_samples, n_warmup)

println("[CUDA.jl]  Mean Run Time: $(mean_time_ms) ms")
println("[CUDA.jl]  FLOPS: $(gflops) GFLOPS")

open("./montecarlo/mc.csv", "a") do io
    @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "CUDA.jl + Distributed.jl", gpus, N, M, mean_time_ms, gflops)
end