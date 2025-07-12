# gray-scott-dagger.jl

using Dagger
using Printf
using Statistics # For mean for checking

# Julia functions for the kernels (will operate on regular Arrays)
# Using `mod1` for periodic boundary conditions, matching the CUDA example's intent.

function laplacian_kernel_cpu!(
    u_prev::AbstractMatrix{Float64},
    Lu::AbstractMatrix{Float64},
    grid_size::Int,
)
    @inbounds for j in 1:grid_size
        for i in 1:grid_size
            # Get neighbor indices with periodic boundary conditions
            ip1 = mod1(i + 1, grid_size)
            im1 = mod1(i - 1, grid_size)
            jp1 = mod1(j + 1, grid_size)
            jm1 = mod1(j - 1, grid_size)

            Lu[i, j] = u_prev[im1, j] + u_prev[ip1, j] +
                       u_prev[i, jm1] + u_prev[i, jp1] -
                       4 * u_prev[i, j]
        end
    end
    return Lu
end

function gray_scott_kernel_cpu!(
    u_prev::AbstractMatrix{Float64},
    v_prev::AbstractMatrix{Float64},
    Lu::AbstractMatrix{Float64},
    Lv::AbstractMatrix{Float64},
    u_new::AbstractMatrix{Float64},
    v_new::AbstractMatrix{Float64},
    Du::Float64,
    Dv::Float64,
    f::Float64,
    k::Float64,
    dt::Float64,
)
    @inbounds for j in 1:size(u_prev, 2)
        for i in 1:size(u_prev, 1)
            u = u_prev[i, j]
            v = v_prev[i, j]
            Lu_val = Lu[i, j]
            Lv_val = Lv[i, j]

            uv_sq = u * v^2

            u_new[i, j] = u + dt * (Du * Lu_val - uv_sq + f * (1.0 - u))
            v_new[i, j] = v + dt * (Dv * Lv_val + uv_sq - (f + k) * v)
        end
    end
    return u_new, v_new
end

function gray_scott_dagger(;
    grid_size::Int = 256,
    Du::Float64 = 0.16,
    Dv::Float64 = 0.08,
    f::Float64 = 0.04,
    k::Float64 = 0.06,
    dt::Float64 = 1.0,
    steps::Int = 1000,
    block_size::Int = 64, # Dagger.DArray block size
)
    @info "Starting Gray-Scott simulation with Dagger.jl" grid_size steps Du Dv f k dt

    # Initialize CPU arrays first, then convert to Dagger.DArray
    # DArray constructor takes an Array, then dimensions, and infers chunking
    # Or, we can specify chunking explicitly: DArray(Array, dims, chunk_dims)
    # Let's create DArrays directly from `zeros` and `fill` for simplicity,
    # and then apply perturbation.
    # Dagger.zeros and Dagger.fill create DArrays with default chunking.
    # We will manually set a chunking scheme for `DArray` from regular arrays.
    
    # Initialize u and v arrays
    u_init_cpu = fill(1.0, grid_size, grid_size)
    v_init_cpu = zeros(Float64, grid_size, grid_size)

    # Add perturbation for v in a central square
    center_start_x = grid_size ÷ 2 - grid_size ÷ 16
    center_end_x = grid_size ÷ 2 + grid_size ÷ 16
    center_start_y = grid_size ÷ 2 - grid_size ÷ 16
    center_end_y = grid_size ÷ 2 + grid_size ÷ 16

    for j in center_start_y:center_end_y
        for i in center_start_x:center_end_x
            v_init_cpu[i, j] = 1.0
        end
    end

    # Convert to Dagger DArrays.
    # We specify `block_size` for chunking. Dagger will create chunks of this size.
    u_prev = Dagger.DArray(u_init_cpu, (grid_size, grid_size), (block_size, block_size))
    v_prev = Dagger.DArray(v_init_cpu, (grid_size, grid_size), (block_size, block_size))

    # Pre-allocate DArrays for outputs (Dagger.DArray operations are lazy)
    # The actual memory for these will be allocated as regular Arrays on workers.
    # We will be creating new DArrays from CPU arrays in the loop, so these are just for initial type.

    Lu_cpu_buffer = zeros(Float64, grid_size, grid_size)
    Lv_cpu_buffer = zeros(Float64, grid_size, grid_size)
    u_new_cpu_buffer = zeros(Float64, grid_size, grid_size)
    v_new_cpu_buffer = zeros(Float64, grid_size, grid_size)

    time_start = time()

    for step in 1:steps
        # Fetch DArrays to regular Julia Arrays on workers (asynchronously)
        u_prev_cpu_future = Dagger.@spawn Dagger.eager_fetch(u_prev)
        v_prev_cpu_future = Dagger.@spawn Dagger.eager_fetch(v_prev)

        # Get the actual CPU arrays (this implicitly waits for the fetches)
        u_prev_cpu = fetch(u_prev_cpu_future)
        v_prev_cpu = fetch(v_prev_cpu_future)

        # Spawn Laplacian computations on regular CPU arrays
        # Use new buffers for each computation to avoid race conditions with futures if not careful
        # Although Dagger.spawn ensures tasks run on distinct objects if they are futures.
        # But since we are creating a new DArray for output, this is safe.
        Lu_cpu_future = Dagger.@spawn laplacian_kernel_cpu!(u_prev_cpu, copy(Lu_cpu_buffer), grid_size)
        Lv_cpu_future = Dagger.@spawn laplacian_kernel_cpu!(v_prev_cpu, copy(Lv_cpu_buffer), grid_size)

        # Spawn Gray-Scott update computations
        u_new_future = Dagger.@spawn gray_scott_kernel_cpu!(
            u_prev_cpu, v_prev_cpu, fetch(Lu_cpu_future), fetch(Lv_cpu_future),
            copy(u_new_cpu_buffer), copy(v_new_cpu_buffer), # Use fresh buffers
            Du, Dv, f, k, dt
        )

        # v_new is returned with u_new, so we only need to fetch u_new_future
        u_new_cpu, v_new_cpu = fetch(u_new_future)

        # Create new DArrays from the updated CPU arrays
        # This incurs overhead for re-distribution if workers are involved.
        u_prev = Dagger.DArray(u_new_cpu, (grid_size, grid_size), (block_size, block_size))
        v_prev = Dagger.DArray(v_new_cpu, (grid_size, grid_size), (block_size, block_size))

        if step % 100 == 0 || step == steps
            mean_u_cpu = mean(Dagger.eager_fetch(u_prev))
            mean_v_cpu = mean(Dagger.eager_fetch(v_prev))
            @printf "Step %d: Mean u = %.6f, Mean v = %.6f\n" step mean_u_cpu mean_v_cpu
        end
    end

    time_end = time()
    elapsed_time = time_end - time_start
    @info "Simulation complete."
    @info "Total Dagger.jl time: $(elapsed_time) seconds"

    return Dagger.eager_fetch(u_prev), Dagger.eager_fetch(v_prev) # Return final state as regular Arrays
end

# Main execution block
if abspath(PROGRAM_FILE) == @__FILE__
    # To run Dagger with multiple processes, start Julia with `julia -p N`
    # or uncomment `addprocs()` here.
    # Using `addproprocs()` dynamically might be less stable than `julia -p N`.
    # using Distributed
    # if nprocs() == 1
    #     addprocs(Sys.CPU_THREADS - 1) # Add one worker per logical core
    # end
    # @everywhere using Dagger # Ensure Dagger is available on all workers

    # Parameters from the original gray-scott.jl example
    GRID_SIZE = 256
    STEPS = 1000
    BLOCK_SIZE = 64 # Size of chunks for DArray

    # Run the Dagger simulation
    u_final, v_final = gray_scott_dagger(
        grid_size=GRID_SIZE,
        steps=STEPS,
        block_size=BLOCK_SIZE
    )

    # Optionally, save or visualize results here
    # using Plots; heatmap(u_final)
end
