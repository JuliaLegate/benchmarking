using CUDA # Import before ImplicitGlobalGrid to activate CUDA support.
using ImplicitGlobalGrid
using MPI
using Random
using Printf
using Statistics

@views  inn(A) = A[2:end-1, 2:end-1]
@views lap1(A) = A[3:end, 2:end-1] .- (2.0f0 .* A[2:end-1, 2:end-1]) .+ A[1:end-2, 2:end-1]
@views lap2(A) = A[2:end-1, 3:end] .- (2.0f0 .* A[2:end-1, 2:end-1]) .+ A[2:end-1, 1:end-2]

# One trial: N_WARMUP untimed steps followed by N_ITER measured steps.
@views function grayscott(nx, ny, n_iter, n_warmup, comm)
    # Physics
    c_u = 1.0f0
    c_v = 0.3f0
    f = 0.03f0
    k = 0.06f0

    # Numerics
    dx = 1
    dy = dx
    dt = dx / 5

    u     = CUDA.zeros(Float32, nx, ny)
    v     = CUDA.zeros(Float32, nx, ny)
    F_u   = CUDA.zeros(Float32, nx-2, ny-2)
    F_v   = CUDA.zeros(Float32, nx-2, ny-2)
    lap_u = CUDA.zeros(Float32, nx-2, ny-2)
    lap_v = CUDA.zeros(Float32, nx-2, ny-2)

    Random.rand!(u[1:nx÷10, 1:ny÷10])
    Random.rand!(v[1:nx÷10, 1:ny÷10])

    start = 0.0
    for it = 1:(n_warmup + n_iter)
        # Exclude allocation and warmup from the measured timesteps.
        if it == n_warmup + 1
            CUDA.synchronize()
            MPI.Barrier(comm)
            start = MPI.Wtime()
        end

        F_u .= (-inn(u) .* (inn(v) .^ 2)) .+ f .* (1.0f0 .- inn(u))
        F_v .= (inn(u) .* (inn(v) .^ 2)) .- (f + k) .* inn(v)

        lap_u .= (lap1(u) ./ (dx * dx)) .+ (lap2(u) ./ (dy * dy))
        lap_v .= (lap1(v) ./ (dx * dx)) .+ (lap2(v) ./ (dy * dy))

        u[2:end-1, 2:end-1] .+= dt .* ((c_u .* lap_u) .+ F_u)
        v[2:end-1, 2:end-1] .+= dt .* ((c_v .* lap_v) .+ F_v)

        CUDA.synchronize()
        update_halo!(u, v)
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
    dimx=dims[1], dimy=dims[2], dimz=1, init_MPI=false)
me == 0 && @printf("Simulation domain: %dx%d (IGG's global grid above includes the outer halos); local arrays: %dx%d including halos\n",
    N, N, nx, ny)
times_ms = zeros(n_trials)
for trial in 1:n_trials
    times_ms[trial] = grayscott(nx, ny, n_iter, n_warmup, comm) / n_iter * 1e3
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
end

finalize_global_grid()
