using CUDA
using Printf
using TimerOutputs
using Statistics

const to = TimerOutput()

function laplacian_kernel(u_prev, Lu, grid_size)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    NX, NY = grid_size, grid_size # assuming square grid

    # Periodic boundary conditions
    xm1 = (x - 2 + NX) % NX + 1
    xp1 = (x % NX) + 1
    ym1 = (y - 2 + NY) % NY + 1
    yp1 = (y % NY) + 1

    @inbounds Lu[x,y] = u_prev[xm1,y] + u_prev[xp1,y] +
                        u_prev[x,ym1] + u_prev[x,yp1] -
                        4*u_prev[x,y]
    return nothing
end

function gray_scott_kernel(u_prev, v_prev, Lu, Lv, u_new, v_new, Du, Dv, f, k, dt)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    @inbounds begin
        u = u_prev[x,y]
        v = v_prev[x,y]
        Lu_val = Lu[x,y]
        Lv_val = Lv[x,y]

        uv_sq = u*v^2

        u_new[x,y] = u + dt * (Du * Lu_val - uv_sq + f * (1.0 - u))
        v_new[x,y] = v + dt * (Dv * Lv_val + uv_sq - (f + k) * v)
    end
    return nothing
end

function gray_scott_cuda(;
    grid_size::Int=256,
    Du::Float64=0.16,
    Dv::Float64=0.08,
    f::Float64=0.04,
    k::Float64=0.06,
    dt::Float64=1.0,
    steps::Int=1000,
    threads_per_block::Int=16 # 2D threads (16x16 = 256 threads)
)
    @info "Starting Gray-Scott simulation with CUDA.jl" grid_size steps Du Dv f k dt threads_per_block

    # Initialise GPU arrays
    u = CUDA.fill(1.0, grid_size, grid_size)
    v = CUDA.zeros(Float64, grid_size, grid_size)

    # Add perturbation for v in a central square
    center_start_x = grid_size ÷ 2 - grid_size ÷ 16
    center_end_x = grid_size ÷ 2 + grid_size ÷ 16
    center_start_y = grid_size ÷ 2 - grid_size ÷ 16
    center_end_y = grid_size ÷ 2 + grid_size ÷ 16

    v[center_start_x:center_end_x, center_start_y:center_end_y] .= 1.0

    # Allocate output arrays for device-side computations
    Lu = CUDA.zeros(Float64, grid_size, grid_size)
    Lv = CUDA.zeros(Float64, grid_size, grid_size)
    u_new = CUDA.zeros(Float64, grid_size, grid_size)
    v_new = CUDA.zeros(Float64, grid_size, grid_size)

    # Configure kernel launch grid
    grid = (grid_size, grid_size)
    block = (threads_per_block, threads_per_block)

    time_start = time()

    @timeit to "Gray-Scott simulation" for step in 1:steps
        # Compute Laplacians
        CUDA.@sync begin
            @timeit to "Laplacian u" @cuda threads=block blocks=grid laplacian_kernel(u, Lu, grid_size)
            @timeit to "Laplacian v" @cuda threads=block blocks=grid laplacian_kernel(v, Lv, grid_size)
        end

        # Update u and v
        CUDA.@sync begin
            @timeit to "Gray-Scott update" @cuda threads=block blocks=grid gray_scott_kernel(u, v, Lu, Lv, u_new, v_new, Du, Dv, f, k, dt)
        end

        # Swap buffers
        u, u_new = u_new, u
        v, v_new = v_new, v

        if step % 100 == 0 || step == steps
            # Synchronize and fetch to CPU for display
            CUDA.@sync begin
                mean_u_cpu = mean(Array(u))
                mean_v_cpu = mean(Array(v))
            end
            @printf "Step %d: Mean u = %.6f, Mean v = %.6f\n" step mean_u_cpu mean_v_cpu
        end
    end

    time_end = time()
    elapsed_time = time_end - time_start
    @info "Simulation complete."
    @info "Total CUDA.jl time: $(elapsed_time) seconds"

    show(to)

    return Array(u), Array(v) # Return final state as regular Arrays
end

# Main execution block
if abspath(PROGRAM_FILE) == @__FILE__
    if !CUDA.has_cuda()
        error("No CUDA-enabled GPU found. Please ensure CUDA is installed and configured correctly.")
    end

    # Parameters
    GRID_SIZE = 256
    STEPS = 1000
    THREADS_PER_BLOCK = 16 # 16x16 = 256 threads per block

    # Run the CUDA simulation
    u_final_cuda, v_final_cuda = gray_scott_cuda(
        grid_size=GRID_SIZE,
        steps=STEPS,
        threads_per_block=THREADS_PER_BLOCK
    )
end

