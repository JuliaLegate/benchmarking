using CUDA # Import before ImplicitGlobalGrid to activate CUDA support.
using ImplicitGlobalGrid
using MPI
using Random
using Printf
using Statistics

include("grayscott_core.jl")

function grayscott_kernel!(u, v, un, vn, p)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + 1
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + 1
    if i < size(u, 1) && j < size(u, 2)
        igg_update_cell!(i, j, u, v, un, vn, p)
    end
    return nothing
end

# One trial: N_WARMUP untimed steps followed by N_ITER measured steps.
function grayscott(nx, ny, N, coords, n_iter, n_warmup, comm)
    p = (dt=0.2f0, dx2=1.0f0, cu=1.0f0, cv=0.3f0, f=0.03f0, k=0.06f0)
    u, v = CUDA.ones(Float32, nx, ny), CUDA.zeros(Float32, nx, ny)
    un, vn = similar(u), similar(v)

    # Same global initial-condition recipe as JACC/Dagger. Only the small seed
    # patch is held on the host, and every rank receives the same random values.
    seed = min(150, N)
    seed_u, seed_v = zeros(Float32, seed, seed), zeros(Float32, seed, seed)
    if MPI.Comm_rank(comm) == 0
        rand!(seed_u)
        rand!(seed_v)
    end
    MPI.Bcast!(seed_u, 0, comm)
    MPI.Bcast!(seed_v, 0, comm)
    igg_seed!(u, v, coords, CuArray(seed_u), CuArray(seed_v))
    CUDA.synchronize()
    update_halo!(u, v)

    threads = (32, 8)
    blocks = (cld(nx - 2, threads[1]), cld(ny - 2, threads[2]))
    start = 0.0
    for it = 1:(n_warmup + n_iter)
        # Exclude allocation and warmup from the measured timesteps.
        if it == n_warmup + 1
            CUDA.synchronize()
            MPI.Barrier(comm)
            start = MPI.Wtime()
        end

        @cuda threads=threads blocks=blocks grayscott_kernel!(u, v, un, vn, p)
        CUDA.synchronize()
        update_halo!(un, vn)
        u, un = un, u
        v, vn = vn, v
    end
    CUDA.synchronize()
    MPI.Barrier(comm)
    return MPI.Allreduce(MPI.Wtime() - start, max, comm)
end

length(ARGS) in (4, 5) || error("Usage: grayscott.jl GPUS N N_ITER N_WARMUP [N_TRIALS=5]")
gpus, N, n_iter, n_warmup = parse.(Int, ARGS[1:4])
n_trials = length(ARGS) == 5 ? parse(Int, ARGS[5]) : 5
gpus > 0 && N >= 4 && n_iter > 0 && n_warmup >= 0 && n_trials > 0 ||
    error("Invalid benchmark dimensions or iteration counts")

MPI.Init()
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
nprocs == gpus || error("Expected $gpus MPI ranks, got $nprocs")
# N counts global simulation cells; each local array also needs two halo cells.
dims = MPI.Dims_create(nprocs, [0, 0, 1])
all(d -> N % d == 0 && N ÷ d >= 2, dims[1:2]) ||
    error("N=$N must divide evenly across the $(dims[1])x$(dims[2]) process grid, with at least two cells per rank in each dimension")
nx, ny = N ÷ dims[1] + 2, N ÷ dims[2] + 2
me, dims, nprocs, coords, comm = init_global_grid(nx, ny, 1;
    dimx=dims[1], dimy=dims[2], dimz=1, periodx=1, periody=1, init_MPI=false)
me == 0 && @printf("Periodic simulation domain: %dx%d; local arrays: %dx%d including halos\n",
    N, N, nx, ny)
times_ms = zeros(n_trials)
for trial in 1:n_trials
    times_ms[trial] = grayscott(nx, ny, N, coords, n_iter, n_warmup, comm) / n_iter * 1e3
    me == 0 && @printf("Trial %d/%d: %.6f ms/step\n", trial, n_trials, times_ms[trial])
end

if me == 0
    mean_ms = mean(times_ms)
    std_ms = std(times_ms)
    sem_ms = std_ms / sqrt(n_trials)
    gupdates = Float64(N)^2 / (mean_ms * 1e6)
    @printf("Gray-Scott: %d GPUs, global %dx%d simulation cells, %d trials, %d iterations/trial, %d warmup steps/trial\n",
        gpus, N, N, n_trials, n_iter, n_warmup)
    @printf("Mean time: %.6f ms/step; stddev: %.6f ms; SEM: %.6f ms\n", mean_ms, std_ms, sem_ms)
    @printf("Throughput: %.6f G cell updates/s\n", gupdates)
    # Harness CSV row per trial: model,gpus,N,M,trial,ms/step,G cell updates/s,correctness.
    if haskey(ENV, "IGG_CSV")
        mkpath(dirname(ENV["IGG_CSV"]))
        open(ENV["IGG_CSV"], "a") do io
            for (trial, ms) in enumerate(times_ms)
                @printf(io, "igg,%d,%d,%d,%d,%.6f,%.6f,skipped\n", gpus, N, N, trial, ms, Float64(N)^2 / (ms * 1e6))
            end
        end
    end
end

finalize_global_grid()
