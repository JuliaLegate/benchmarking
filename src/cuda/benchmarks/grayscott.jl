# CUDA.jl copy of the shared Gray-Scott step. Keep its equations and periodic
# boundaries in sync with ../../benchmarks/grayscott.jl.
mutable struct CUDAGrayScottState{A,P}
    u::A
    v::A
    u_new::A
    v_new::A
    F_u::A
    F_v::A
    u_lap::A
    v_lap::A
    params::P
end

# Both public Gray-Scott names use this implementation in the CUDA worker.
function initialize(
    b::Union{GrayScottBaseline{T},GrayScottAccelerated{T}};
    mod=CUDA, deterministic::Bool=false,
) where {T}
    # Reuse the shared initial conditions, including the host correctness seed.
    st = initialize_grayscott_state(b; mod, deterministic)
    interior = (b.N - 2, b.M - 2)
    return (CUDAGrayScottState(
        st.u, st.v, st.u_new, st.v_new,
        zeros_array(mod, T, interior), zeros_array(mod, T, interior),
        zeros_array(mod, T, interior), zeros_array(mod, T, interior), st.params,
    ),)
end

function to_backend_state(mod, st::CUDAGrayScottState)
    return CUDAGrayScottState(
        to_backend_state(mod, st.u), to_backend_state(mod, st.v),
        to_backend_state(mod, st.u_new), to_backend_state(mod, st.v_new),
        to_backend_state(mod, st.F_u), to_backend_state(mod, st.F_v),
        to_backend_state(mod, st.u_lap), to_backend_state(mod, st.v_lap), st.params,
    )
end

@views function _cuda_gs_step!(u, v, u_new, v_new, F_u, F_v, u_lap, v_lap, args::GSParams)
    # Dot every operator so each RHS can fuse.
    F_u .= (
        (
            .-u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) .+ args.f .* (1.0f0 .- u[2:(end - 1), 2:(end - 1)])
    )
    F_v .= (
        (
            u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
        ) .- (args.f + args.k) .* v[2:(end - 1), 2:(end - 1)]
    )
    # 2-D Laplacian via slicing, excluding boundaries
    u_lap .= (
        (
            u[3:end, 2:(end - 1)] .- 2 .* u[2:(end - 1), 2:(end - 1)] .+
            u[1:(end - 2), 2:(end - 1)]
        ) ./ args.dx^2 .+
        (
            u[2:(end - 1), 3:end] .- 2 .* u[2:(end - 1), 2:(end - 1)] .+
            u[2:(end - 1), 1:(end - 2)]
        ) ./ args.dx^2
    )
    v_lap .= (
        (
            v[3:end, 2:(end - 1)] .- 2 .* v[2:(end - 1), 2:(end - 1)] .+
            v[1:(end - 2), 2:(end - 1)]
        ) ./ args.dx^2 .+
        (
            v[2:(end - 1), 3:end] .- 2 .* v[2:(end - 1), 2:(end - 1)] .+
            v[2:(end - 1), 1:(end - 2)]
        ) ./ args.dx^2
    )

    # Forward-Euler step for all interior points
    u_new[2:(end - 1), 2:(end - 1)] .=
        ((args.c_u .* u_lap) .+ F_u) .* args.dt .+ u[2:(end - 1), 2:(end - 1)]
    v_new[2:(end - 1), 2:(end - 1)] .=
        ((args.c_v .* v_lap) .+ F_v) .* args.dt .+ v[2:(end - 1), 2:(end - 1)]

    # Periodic boundary conditions
    u_new[:, 1] .= u[:, end - 1]
    u_new[:, end] .= u[:, 2]
    u_new[1, :] .= u[end - 1, :]
    u_new[end, :] .= u[2, :]
    v_new[:, 1] .= v[:, end - 1]
    v_new[:, end] .= v[:, 2]
    v_new[1, :] .= v[end - 1, :]
    v_new[end, :] .= v[2, :]
end

function run!(b::AbstractGrayScott, st::CUDAGrayScottState)
    if st.u isa Array
        # Preserve the independent shared CPU reference for correctness checks.
        _gs_step!(b, st.u, st.v, st.u_new, st.v_new, st.params)
    else
        _cuda_gs_step!(
            st.u, st.v, st.u_new, st.v_new,
            st.F_u, st.F_v, st.u_lap, st.v_lap, st.params,
        )
    end
    st.u, st.u_new = st.u_new, st.u
    st.v, st.v_new = st.v_new, st.v
    return nothing
end
