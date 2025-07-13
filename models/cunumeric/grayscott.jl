using cuNumeric
using Legate
using Printf

struct Params
    dx::Float64
    dt::Float64
    c_u::Float64
    c_v::Float64
    f::Float64
    k::Float64

    function Params(dx=1, c_u=1.0, c_v=0.3, f=0.03, k=0.06)
        new(dx, dx/5, c_u, c_v, f, k)
    end
end


get_time_us() = Legate.value(Legate.time_microseconds())

function step(u, v, u_new, v_new, args::Params)
    # calculate F_u and F_v functions
    # currently we don't have NDArray^x working yet. 
    F_u = (
        (
            -u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) + args.f*(1 .- u[2:(end - 1), 2:(end - 1)])
    )
    F_v = (
        (
            u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) - (args.f+args.k)*v[2:(end - 1), 2:(end - 1)]
    )
    # 2-D Laplacian of f using array slicing, excluding boundaries
    # For an N x N array f, f_lap is the Nend x Nend array in the "middle"
    u_lap = (
        (
            u[3:end, 2:(end - 1)] - 2*u[2:(end - 1), 2:(end - 1)] +
            u[1:(end - 2), 2:(end - 1)]
        ) ./ args.dx^2 +
        (
            u[2:(end - 1), 3:end] - 2*u[2:(end - 1), 2:(end - 1)] +
            u[2:(end - 1), 1:(end - 2)]
        ) ./ args.dx^2
    )
    v_lap = (
        (
            v[3:end, 2:(end - 1)] - 2*v[2:(end - 1), 2:(end - 1)] +
            v[1:(end - 2), 2:(end - 1)]
        ) ./ args.dx^2 +
        (
            v[2:(end - 1), 3:end] - 2*v[2:(end - 1), 2:(end - 1)] +
            v[2:(end - 1), 1:(end - 2)]
        ) ./ args.dx^2
    )

    # Forward-Euler time step for all points except the boundaries
    u_new[2:(end - 1), 2:(end - 1)] =
        ((args.c_u * u_lap) + F_u) * args.dt + u[2:(end - 1), 2:(end - 1)]
    v_new[2:(end - 1), 2:(end - 1)] =
        ((args.c_v * v_lap) + F_v) * args.dt + v[2:(end - 1), 2:(end - 1)]

    # Apply periodic boundary conditions
    u_new[:, 1] = u[:, end - 1]
    u_new[:, end] = u[:, 2]
    u_new[1, :] = u[end - 1, :]
    u_new[end, :] = u[2, :]
    v_new[:, 1] = v[:, end - 1]
    v_new[:, end] = v[:, 2]
    v_new[1, :] = v[end - 1, :]
    v_new[end, :] = v[2, :]
end

function grayscott(N, M, n_steps)
    dims = (N, M)
    FT = Float32
    args = Params()
    garbage_interval = 1

    u = cuNumeric.ones(dims)
    v = cuNumeric.zeros(dims)
    u_new = cuNumeric.zeros(dims)
    v_new = cuNumeric.zeros(dims)

    u[1:150, 1:150] = cuNumeric.random(FT, (150, 150))
    v[1:150, 1:150] = cuNumeric.random(FT, (150, 150))

    for n in 1:n_steps
        step(u, v, u_new, v_new, args)
        # update u and v 
        # this doesn't copy, this switching references 
        u, u_new = u_new, u
        v, v_new = v_new, v

        if n % garbage_interval == 0
            GC.gc()
        end
    end
end

function total_flops(N, T)
    return N * N * T # O(N^2 * T)
end

gpus = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
M = parse(Int, ARGS[3])
n_samples = parse(Int, ARGS[4])

println("[cuNumeric] GrayScott benchmark on $(N)x$(M) matricies for $(n_samples) iterations")

start_time = get_time_us()

grayscott(N, N, n_samples)

total_time_μs = get_time_us() - start_time
mean_time_ms = total_time_μs / (n_samples * 1e3)
gflops = total_flops(N, M) / (mean_time_ms * 1e6) # GFLOP is 1e9

println("[cuNumeric] Mean Run Time: $(mean_time_ms) ms")
println("[cuNumeric] FLOPS: $(gflops) GFLOPS")

open("./grayscott/grayscott.csv", "a") do io
    @printf(io, "%s,%d,%d,%d,%.6f,%.6f\n", "cunumeric", gpus, N, M, mean_time_ms, gflops)
end
