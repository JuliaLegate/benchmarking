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

function do_distributed_embaressingly_parallel_work(N_total, work_fn)
    @everywhere N_dev = length(devices())
    futures = asyncmap((zip(workers(), devices()))) do (p, d)
        remotecall_wait(p) do
            @info "Worker $p uses $d"
            device!(d)
            N = Int(div(N_total, N_dev))
            work_fn(N)
        end
    end
    return fetch.(futures)
end

# do_distributed_embaressingly_parallel_work(1e8, mc_integration)