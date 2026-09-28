# Default (`multi`): JACC.Multi z-slabs on every GPU count; see mg_multi.jl.
# `JACC_NAS_MG_IMPL=single`: the original single-GPU version below, where every
# operator and the norm are JACC kernels/reductions with no CUDA.jl kernels.
# Transfers use direct per-cell kernels, unlike the array adapters' staged
# transfers. The common harness times initial zeroing and L2 sum-of-squares,
# but omits NPB's Linf norm; see nas/README.md.

include(joinpath(@__DIR__, "..", "..", "..", "nas", "mg.jl"))
include(joinpath(@__DIR__, "mg_multi.jl"))

struct JACCNASMG
    class::String
    N::Int
    M::Int
    impl::String
end

struct JACCNASMGState
    u
    r
    rhs
    c
    norm_reducer
end

function model_build_nas_mg(config::ModelWorkerConfig)
    config.T === Float64 || error("NAS MG requires Float64")
    impl = get(ENV, "JACC_NAS_MG_IMPL", "multi")
    impl in ("multi", "single") || error("JACC_NAS_MG_IMPL must be multi or single")
    if impl == "single"
        config.gpus == 1 || error("JACC_NAS_MG_IMPL=single supports one GPU")
    else
        JACC.Multi.ndev() == config.gpus || error(
            "JACC sees $(JACC.Multi.ndev()) GPU(s), but this run requested $(config.gpus)"
        )
    end
    class = uppercase(string(get(config.kwargs, :class, "S")))
    nx, ny, _ = nas_mg_dims(nas_mg_parameters(class))
    (config.N, config.M) == (nx, ny) || error(
        "NAS MG class $class requires N=$nx, M=$ny"
    )
    return JACCNASMG(class, config.N, config.M, impl)
end

function model_initialize(b::JACCNASMG)
    b.impl == "multi" && return jacc_multi_mg(JACCMultiOps(), b.class)
    p = nas_mg_parameters(b.class)
    shapes = nas_mg_level_shapes(p)
    u = [JACC.zeros(Float64, shape...) for shape in shapes]
    r = [JACC.zeros(Float64, shape...) for shape in shapes]
    return JACCNASMGState(
        u, r, JACC.array(nas_mg_rhs(p)), nas_mg_smoother(b.class),
        JACC.reducer(; range=prod(nas_mg_dims(p)), type=Float64, sync=false),
    )
end

# CUDA fills in and retains a LaunchSpec's grid dimensions. MG changes both
# grid size and kernel, so each launch must start with an unconfigured spec.
# None of these kernels uses dynamic shared memory.
function jacc_mg_launch(n, kernel, args...)
    return JACC.parallel_for(JACC.launch_spec(; sync=false, shmem_size=0), n, kernel, args...)
end

@inline function jacc_mg_decode(index, (n1, n2, _), offset)
    q = index - 1
    i = q % n1 + offset
    j = (q ÷ n1) % n2 + offset
    k = q ÷ (n1*n2) + offset
    return i, j, k
end

function jacc_mg_zero(index, out)
    @inbounds out[index] = 0.0
end

function jacc_mg_comm_x(index, out, (n1, n2, _))
    q = index - 1
    j = q % (n2 - 2) + 2
    k = q ÷ (n2 - 2) + 2
    @inbounds begin
        out[1, j, k] = out[n1 - 1, j, k]
        out[n1, j, k] = out[2, j, k]
    end
end

function jacc_mg_comm_y(index, out, (n1, n2, _))
    q = index - 1
    i = q % n1 + 1
    k = q ÷ n1 + 2
    @inbounds begin
        out[i, 1, k] = out[i, n2 - 1, k]
        out[i, n2, k] = out[i, 2, k]
    end
end

function jacc_mg_comm_z(index, out, (n1, _, n3))
    q = index - 1
    i, j = q % n1 + 1, q ÷ n1 + 1
    @inbounds begin
        out[i, j, 1] = out[i, j, n3 - 1]
        out[i, j, n3] = out[i, j, 2]
    end
end

function jacc_mg_comm3!(s::JACCNASMGState, out)
    n1, n2, n3 = d = size(out)
    jacc_mg_launch((n2 - 2)*(n3 - 2), jacc_mg_comm_x, out, d)
    jacc_mg_launch(n1*(n3 - 2), jacc_mg_comm_y, out, d)
    jacc_mg_launch(n1*n2, jacc_mg_comm_z, out, d)
    return out
end

function jacc_mg_resid(index, r, u, v, interior)
    i, j, k = jacc_mg_decode(index, interior, 2)
    @inbounds r[i, j, k] =
        v[i, j, k] - NAS_MG_A[1]*u[i, j, k] -
        NAS_MG_A[3] * (
            u[i, j - 1, k - 1] + u[i, j + 1, k - 1] + u[i, j - 1, k + 1] + u[i, j + 1, k + 1] +
            u[i - 1, j, k - 1] + u[i + 1, j, k - 1] + u[i - 1, j, k + 1] + u[i + 1, j, k + 1] +
            u[i - 1, j - 1, k] + u[i + 1, j - 1, k] + u[i - 1, j + 1, k] + u[i + 1, j + 1, k]
        ) -
        NAS_MG_A[4] * (
            u[i - 1, j - 1, k - 1] + u[i + 1, j - 1, k - 1] + u[i - 1, j + 1, k - 1] +
            u[i + 1, j + 1, k - 1] +
            u[i - 1, j - 1, k + 1] + u[i + 1, j - 1, k + 1] + u[i - 1, j + 1, k + 1] +
            u[i + 1, j + 1, k + 1]
        )
end

function jacc_mg_resid!(s, r, u, v)
    interior = size(r) .- 2
    jacc_mg_launch(prod(interior), jacc_mg_resid, r, u, v, interior)
    return jacc_mg_comm3!(s, r)
end

function jacc_mg_psinv(index, u, r, interior, c)
    i, j, k = jacc_mg_decode(index, interior, 2)
    @inbounds u[i, j, k] +=
        c[1]*r[i, j, k] +
        c[2] * (
            r[i - 1, j, k] + r[i + 1, j, k] + r[i, j - 1, k] +
            r[i, j + 1, k] + r[i, j, k - 1] + r[i, j, k + 1]
        ) +
        c[3] * (
            r[i, j - 1, k - 1] + r[i, j + 1, k - 1] + r[i, j - 1, k + 1] + r[i, j + 1, k + 1] +
            r[i - 1, j, k - 1] + r[i + 1, j, k - 1] + r[i - 1, j, k + 1] + r[i + 1, j, k + 1] +
            r[i - 1, j - 1, k] + r[i + 1, j - 1, k] + r[i - 1, j + 1, k] + r[i + 1, j + 1, k]
        )
end

function jacc_mg_psinv!(s, u, r)
    interior = size(u) .- 2
    jacc_mg_launch(prod(interior), jacc_mg_psinv, u, r, interior, s.c)
    return jacc_mg_comm3!(s, u)
end

function jacc_mg_restrict(index, coarse, fine, interior)
    i, j, k = jacc_mg_decode(index, interior, 2)
    fi, fj, fk = 2i - 1, 2j - 1, 2k - 1
    @inbounds coarse[i, j, k] =
        0.5*fine[fi, fj, fk] +
        0.25 * (
            fine[fi - 1, fj, fk] + fine[fi + 1, fj, fk] + fine[fi, fj - 1, fk] +
            fine[fi, fj + 1, fk] + fine[fi, fj, fk - 1] + fine[fi, fj, fk + 1]
        ) +
        0.125 * (
            fine[fi, fj - 1, fk - 1] + fine[fi, fj + 1, fk - 1] + fine[fi, fj - 1, fk + 1] +
            fine[fi, fj + 1, fk + 1] +
            fine[fi - 1, fj, fk - 1] + fine[fi + 1, fj, fk - 1] + fine[fi - 1, fj, fk + 1] +
            fine[fi + 1, fj, fk + 1] +
            fine[fi - 1, fj - 1, fk] + fine[fi + 1, fj - 1, fk] + fine[fi - 1, fj + 1, fk] +
            fine[fi + 1, fj + 1, fk]
        ) +
        0.0625 * (
            fine[fi - 1, fj - 1, fk - 1] + fine[fi + 1, fj - 1, fk - 1] +
            fine[fi - 1, fj + 1, fk - 1] + fine[fi + 1, fj + 1, fk - 1] +
            fine[fi - 1, fj - 1, fk + 1] + fine[fi + 1, fj - 1, fk + 1] +
            fine[fi - 1, fj + 1, fk + 1] + fine[fi + 1, fj + 1, fk + 1]
        )
end

function jacc_mg_restrict!(s, coarse, fine)
    interior = size(coarse) .- 2
    jacc_mg_launch(prod(interior), jacc_mg_restrict, coarse, fine, interior)
    return jacc_mg_comm3!(s, coarse)
end

@inline jacc_mg_lerp(a, b, weight) = muladd(weight, b - a, a)

function jacc_mg_interp(index, fine, coarse, shape)
    i, j, k = jacc_mg_decode(index, shape, 1)
    qi, qj, qk = i - 1, j - 1, k - 1
    i0, j0, k0 = qi ÷ 2 + 1, qj ÷ 2 + 1, qk ÷ 2 + 1
    i1, j1, k1 = i0 + (qi % 2), j0 + (qj % 2), k0 + (qk % 2)
    wi, wj, wk = 0.5*(qi % 2), 0.5*(qj % 2), 0.5*(qk % 2)
    @inbounds begin
        z00 = jacc_mg_lerp(coarse[i0, j0, k0], coarse[i1, j0, k0], wi)
        z10 = jacc_mg_lerp(coarse[i0, j1, k0], coarse[i1, j1, k0], wi)
        z01 = jacc_mg_lerp(coarse[i0, j0, k1], coarse[i1, j0, k1], wi)
        z11 = jacc_mg_lerp(coarse[i0, j1, k1], coarse[i1, j1, k1], wi)
        z0 = jacc_mg_lerp(z00, z10, wj)
        z1 = jacc_mg_lerp(z01, z11, wj)
        fine[i, j, k] += jacc_mg_lerp(z0, z1, wk)
    end
end

function jacc_mg_interp!(s, fine, coarse)
    jacc_mg_launch(length(fine), jacc_mg_interp, fine, coarse, size(fine))
    return fine
end

function jacc_mg_fill!(s, out)
    jacc_mg_launch(length(out), jacc_mg_zero, out)
    return out
end

function jacc_mg_cycle!(s)
    finest = length(s.u)
    for level in finest:-1:2
        jacc_mg_restrict!(s, s.r[level - 1], s.r[level])
    end
    jacc_mg_fill!(s, s.u[1])
    jacc_mg_psinv!(s, s.u[1], s.r[1])
    for level in 2:(finest - 1)
        jacc_mg_fill!(s, s.u[level])
        jacc_mg_interp!(s, s.u[level], s.u[level - 1])
        jacc_mg_resid!(s, s.r[level], s.u[level], s.r[level])
        jacc_mg_psinv!(s, s.u[level], s.r[level])
    end
    jacc_mg_interp!(s, s.u[end], s.u[end - 1])
    jacc_mg_resid!(s, s.r[end], s.u[end], s.rhs)
    jacc_mg_psinv!(s, s.u[end], s.r[end])
    return nothing
end

function jacc_mg_norm_term(index, residual, interior)
    i, j, k = jacc_mg_decode(index, interior, 2)
    return @inbounds abs2(residual[i, j, k])
end

function model_run!(b::JACCNASMG, s::JACCNASMGState)
    p = nas_mg_parameters(b.class)
    foreach(out -> jacc_mg_fill!(s, out), s.u)
    jacc_mg_resid!(s, s.r[end], s.u[end], s.rhs)
    s.norm_reducer(jacc_mg_norm_term, s.r[end], nas_mg_dims(p))
    for _ in 1:p.niter
        jacc_mg_cycle!(s)
        jacc_mg_resid!(s, s.r[end], s.u[end], s.rhs)
    end
    s.norm_reducer(jacc_mg_norm_term, s.r[end], nas_mg_dims(p))
    return s.norm_reducer.workspace.ret
end

model_run!(b::JACCNASMG, s::JACCMultiMG) = mgm_run!(s, b.class)

# JACC.Multi launches and copies synchronize every device before returning.
model_synchronize(b::JACCNASMG) = b.impl == "multi" ? nothing : JACC.synchronize()
model_save_id(b::JACCNASMG, ::Symbol) = b.impl == "multi" ? :jacc : :jacc_single
model_worker_label(b::JACCNASMG, ::String) = "JACC.jl ($(b.impl))"

function model_check_correctness(b::JACCNASMG, config)
    p = nas_mg_parameters(b.class)
    result = model_run!(b, model_initialize(b))
    model_synchronize(b)
    sumsq = result isa Real ? result : only(JACC.to_host(result))
    return nas_mg_status(b.class, sqrt(sumsq/Float64(prod(nas_mg_dims(p)))))
end

function model_correctness_context(b::JACCNASMG, config)
    return (; reference="NPB-GPU", dims=nas_mg_dims(nas_mg_parameters(b.class)))
end
