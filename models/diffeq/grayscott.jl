using CUDA                # Import CUDA before ImplicitGlobalGrid to activate its CUDA device support
using ImplicitGlobalGrid
using Random
using Printf


#export IGG_CUDAAWARE_MPI=1

@views  inn(A) = A[2:end-1,2:end-1]
@views lap1(A) = A[3:end, 2:end-1] .- (2.0f0 .* A[2:end-1, 2:end-1]) .+ A[1:end-2, 2:end-1]
@views lap2(A) = A[2:end-1, 3:end] .- (2.0f0 .* A[2:end-1, 2:end-1]) .+ A[2:end-1, 1:end-2]

@views function grayscott(nx, ny, nt, iswarmup)
    # Physics
    c_u = 1.0f0
    c_v = 0.3f0
    f = 0.03f0
    k = 0.06f0
    
    # Numerics
    dx         = 1
    dy         = dx
    dt         = dx / 5

    if iswarmup # setup global grid
        me, dims   = ImplicitGlobalGrid.init_global_grid(nx, ny, 1);  
    end

    # Array initializations
    u     = CUDA.zeros(Float32, nx, ny)
    v     = CUDA.zeros(Float32, nx, ny)
    F_u   = CUDA.zeros(Float32, nx-2, ny-2)
    F_v   = CUDA.zeros(Float32, nx-2, ny-2)
    lap_u = CUDA.zeros(Float32, nx-2, ny-2)
    lap_v = CUDA.zeros(Float32, nx-2, ny-2)
   
    # Initial conditions
    @views Random.rand!(u[1 : nx÷10, 1 : ny÷10])
    @views Random.rand!(v[1 : nx÷10, 1 : ny÷10])

    # Time loop
    for it = 1:nt
        F_u   .= (-inn(u) .* (inn(v) .^ 2)) .+ f .* (1.0f0 .- inn(u));                     
        F_v   .= (inn(u) .* (inn(v) .^ 2)) .- (f + k) .* inn(v);

        # Laplacian
        lap_u .= (lap1(u) ./ (dx * dx)) .+ (lap2(u) ./ (dy * dy));
        lap_v .= (lap1(v) ./ (dx * dx)) .+ (lap2(v) ./ (dy * dy));

        # Forward Euler step
        u[2:end-1,2:end-1] .+= dt .* ((c_u .* lap_u ).+ F_u);
        v[2:end-1,2:end-1] .+= dt .* ((c_v .* lap_v ).+ F_v);

        update_halo!(u, v); 
    end

    if !iswarmup # only mpi final non warmup
        finalize_global_grid();
    end
end


function total_flops(N, T)
    return N * N * T # O(N^2 * T)
end

gpus = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
M = parse(Int, ARGS[3])
n_samples = parse(Int, ARGS[4])
warmup=5

println("[DIFFEQ] GrayScott benchmark on $(N)x$(M) matricies for $(n_samples) iterations")
grayscott(N, M, warmup, true)

CUDA.synchronize()
t = CUDA.@elapsed begin
    grayscott(N, M, n_samples, false)
    CUDA.synchronize()  # Wait for GPU to finish
end

# t is time in seconds
avg_t = t / n_samples
mean_time_ms = avg_t * 1e3 # time in ms
gflops = total_flops(N, M) / (avg_t * 1e9) # GFLOP is 1e9

println("[DIFFEQ] Mean Run Time: $(mean_time_ms) ms")
println("[DIFFEQ] FLOPS: $(gflops) GFLOPS")

open("./grayscott/grayscott.csv", "a") do io
    @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "diffeq", gpus, N, M, mean_time_ms, gflops)
end