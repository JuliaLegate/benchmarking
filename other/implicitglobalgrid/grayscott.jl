using CUDA # Load before ImplicitGlobalGrid to activate its CUDA extension.
using ImplicitGlobalGrid
using MPI
using Printf
using Random

include("grayscott_core.jl")
include("reference.jl")

function advance(state, work, layout, synchronize)
    u, v, u_new, v_new = state
    local_step!(u, v, u_new, v_new, work, layout)
    # IGG uses its own transfer streams; finish the producer kernels first.
    synchronize()
    update_halo!(u_new, v_new)
    return (u_new, v_new, u, v)
end

function check_result(state, layout, steps, comm)
    n = layout.n
    u, v = ones(Float32, n, n), zeros(Float32, n, n)
    pu, pv = seed_patch(n; deterministic=true)
    s = size(pu, 1)
    u[1:s, 1:s], v[1:s, 1:s] = pu, pv
    un, vn = zero(u), zero(v)
    for _ in 1:steps
        GrayScottReference.step!(u, v, un, vn, gs_parameters())
        u, un = un, u
        v, vn = vn, v
    end
    indices = ntuple(d -> (layout.offset[d] + 1):(layout.offset[d] + layout.owned[d]), 2)
    error = maximum(zip(state[1:2], (u, v))) do (actual, expected)
        maximum(abs, Array(view(actual, layout.physical...)) .- view(expected, indices...))
    end
    max_error = MPI.Allreduce(error, max, comm)
    isfinite(max_error) && max_error <= 1.0f-5 ||
        Base.error("Gray-Scott correctness failed: maximum absolute error = $max_error")
    MPI.Comm_rank(comm) == 0 && @printf("Correctness: pass (max absolute error %.8g)\n", max_error)
end

function main(args=ARGS; cpu=false)
    check = !isempty(args) && last(args) == "--check"
    values = check ? args[1:end-1] : args
    length(values) == 4 || error("Usage: grayscott.jl GPUS N STEPS WARMUP [--check]")
    gpus, n, steps, warmup = parse.(Int, values)
    steps >= 1 && warmup >= 0 || error("STEPS must be positive and WARMUP nonnegative")
    dims = process_grid(n, gpus)
    check && n > 512 && error("Use N <= 512 for --check (each rank runs a full CPU reference)")
    cpu || CUDA.functional(true) || error("A working CUDA GPU is required")
    cpu || CUDA.allowscalar(false)
    MPI.Init()
    try
        MPI.Comm_size(MPI.COMM_WORLD) == gpus || error("Launch exactly $gpus MPI ranks")
        me, _, _, coords, comm = init_global_grid(
            n ÷ dims[1] + 4, n ÷ dims[2] + 4, 1;
            dimx=dims[1], dimy=dims[2], dimz=1, periodx=1, periody=1,
            overlaps=(4, 4, 2), halowidths=(2, 2, 1),
            init_MPI=false, device_type=cpu ? "none" : "CUDA", quiet=true,
        )
        try
            layout = local_layout(n, dims, coords)
            array = cpu ? Array : CUDA.CuArray
            synchronize = cpu ? (() -> nothing) : CUDA.synchronize
            state = initial_fields(array, layout; deterministic=check)
            work = workspace(first(state), layout)
            synchronize()
            update_halo!(state[1], state[2])
            for _ in 1:warmup
                state = advance(state, work, layout, synchronize)
            end
            synchronize()
            MPI.Barrier(comm)
            start = MPI.Wtime()
            for _ in 1:steps
                state = advance(state, work, layout, synchronize)
            end
            synchronize()
            elapsed = MPI.Allreduce(MPI.Wtime() - start, max, comm)
            check && check_result(state, layout, warmup + steps, comm)
            if me == 0
                mean_ms = elapsed / steps * 1e3
                # The harness calls N*N "flops"; report the honest stencil metric.
                points_per_second = Float64(n - 2)^2 * steps / elapsed
                println("Gray-Scott Float32: global $(n)x$n, $gpus rank(s), topology $(dims[1])x$(dims[2])")
                @printf("Mean time: %.6f ms/step; %.6f G interior cell updates/s\n", mean_ms, points_per_second / 1e9)
                println("backend,eltype,gpus,N,steps,warmup,mean_ms,gupdates_per_second")
                @printf("implicitglobalgrid,Float32,%d,%d,%d,%d,%.6f,%.6f\n",
                    gpus, n, steps, warmup, mean_ms, points_per_second / 1e9)
            end
        finally
            finalize_global_grid(; finalize_MPI=false)
        end
    finally
        MPI.Finalize()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
