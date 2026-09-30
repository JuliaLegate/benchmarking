using CUDA # Import before ImplicitGlobalGrid to activate CUDA support.
using ImplicitGlobalGrid
using MPI
using Random
using Printf

@views  inn(A) = A[2:end-1, 2:end-1]
@views lap1(A) = A[3:end, 2:end-1] .- (2.0f0 .* A[2:end-1, 2:end-1]) .+ A[1:end-2, 2:end-1]
@views lap2(A) = A[2:end-1, 3:end] .- (2.0f0 .* A[2:end-1, 2:end-1]) .+ A[2:end-1, 1:end-2]

@views function grayscott(nx, ny, nt, warmup, comm)
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
    for it = 1:(warmup + nt)
        # Exclude allocation and warmup from the measured timesteps.
        if it == warmup + 1
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

length(ARGS) == 4 || error("Usage: grayscott.jl GPUS N STEPS WARMUP")
gpus, N, nt, warmup = parse.(Int, ARGS)
gpus > 0 && N >= 4 && nt > 0 && warmup >= 0 || error("Invalid benchmark dimensions or iteration counts")

me, dims, nprocs, coords, comm = init_global_grid(N, N, 1; dimz=1)
nprocs == gpus || error("Expected $gpus MPI ranks, got $nprocs")
elapsed = grayscott(N, N, nt, warmup, comm)

if me == 0
    mean_ms = elapsed / nt * 1e3
    gupdates = gpus * Float64(N - 2)^2 * nt / elapsed / 1e9
    @printf("Gray-Scott: %d GPUs, local %dx%d, %d iterations\n", gpus, N, N, nt)
    @printf("Mean time: %.6f ms/step; %.6f G cell updates/s\n", mean_ms, gupdates)
end

finalize_global_grid()
