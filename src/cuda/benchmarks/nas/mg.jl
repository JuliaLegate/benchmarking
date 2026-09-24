# LIMITATION: This idiomatic CUDA.jl baseline expresses the NPB MG operators as
# fused CuArray broadcasts instead of copying NPB-GPU's hand-written kernels.
# CUDA.jl remains the harness's intentionally single-GPU baseline.
# Restriction/interpolation use strided views and multiple broadcasts, not
# JACC's one-kernel-per-operator approach. The common harness times initial
# zeroing and L2 sum-of-squares, but omits NPB's Linf norm (see nas/README.md).

function cuda_nas_mg_impl(b::NASMultiGrid)
    impl = b.implementation == "default" ? "direct" : b.implementation
    impl in ("direct", "separable") || error("CUDA NAS MG implementation must be direct or separable")
    return impl
end

benchmark_backend_label(b::NASMultiGrid, backend::String, default::String) =
    backend == "cudajl" ? "CUDA.jl ($(cuda_nas_mg_impl(b)))" : default

benchmark_backend_save_as(b::NASMultiGrid, backend::String, default::String) =
    backend == "cudajl" && cuda_nas_mg_impl(b) == "separable" ?
        "CUDA.jl_separable" : default

mutable struct CUDANASMGState{U,R,V,W}
    u::U
    r::R
    rhs::V
    interp_weights::W
end

function initialize(b::NASMultiGrid{Float64}; mod=CUDA)
    p = validate_nas_mg(b)
    sizes = nas_mg_level_sizes(p)
    u = [CUDA.zeros(Float64, n, n, n) for n in sizes]
    r = [CUDA.zeros(Float64, n, n, n) for n in sizes]
    rhs = CUDA.CuArray(nas_mg_rhs(p))
    weights = cuda_nas_mg_impl(b) == "separable" ?
        reshape(CUDA.CuArray([0.0, 0.5]), 2, 1, 1, 1) : nothing
    return (CUDANASMGState(u, r, rhs, weights),)
end

# Julia arrays put the first axis in contiguous storage; cuNumeric's equivalent
# transfer moves the active axis to the last position before pairing cells.
function cuda_mg_front(array, axis)
    axis == 1 && return array, (1, 2, 3)
    permutation = axis == 2 ? (2, 1, 3) : (3, 1, 2)
    return permutedims(array, permutation), invperm(permutation)
end

function cuda_mg_restrict_axis(array, axis)
    front, inverse = cuda_mg_front(array, axis)
    n, d2, d3 = size(front)
    physical = n - 2
    left = copy(@view front[2:(n - 1), :, :])
    right = copy(@view front[3:n, :, :])
    paired_shape = (2, physical ÷ 2, d2, d3)
    reduced = reshape(
        sum(reshape(left, paired_shape); dims=1) .+
        sum(reshape(right, paired_shape); dims=1),
        physical ÷ 2, d2, d3,
    )
    return axis == 1 ? reduced : permutedims(reduced, inverse)
end

function cuda_mg_restrict!(coarse, fine)
    reduced = cuda_mg_restrict_axis(fine, 1)
    reduced = cuda_mg_restrict_axis(reduced, 2)
    reduced = cuda_mg_restrict_axis(reduced, 3)
    n = size(coarse, 1)
    @views coarse[2:(n - 1), 2:(n - 1), 2:(n - 1)] .= reduced ./ 16.0
    return nas_mg_comm3!(coarse)
end

function cuda_mg_interp_axis(array, axis, weights)
    front, inverse = cuda_mg_front(array, axis)
    n, d2, d3 = size(front)
    lo = copy(@view front[1:(n - 1), :, :])
    hi = copy(@view front[2:n, :, :])
    lo4 = reshape(lo, 1, n - 1, d2, d3)
    hi4 = reshape(hi, 1, n - 1, d2, d3)
    mixed = lo4 .+ weights .* (hi4 .- lo4)
    interpolated = reshape(mixed, 2(n - 1), d2, d3)
    result = axis == 1 ? interpolated : permutedims(interpolated, inverse)
    return copy(result)
end

function cuda_mg_interp!(fine, coarse, weights)
    interpolated = cuda_mg_interp_axis(coarse, 1, weights)
    interpolated = cuda_mg_interp_axis(interpolated, 2, weights)
    interpolated = cuda_mg_interp_axis(interpolated, 3, weights)
    fine .+= interpolated
    return fine
end

function cuda_mg_separable_cycle!(s, c)
    finest = length(s.u)
    for level in finest:-1:2
        cuda_mg_restrict!(s.r[level - 1], s.r[level])
    end
    fill!(s.u[1], 0.0)
    nas_mg_psinv!(s.u[1], s.r[1], c)
    for level in 2:(finest - 1)
        fill!(s.u[level], 0.0)
        cuda_mg_interp!(s.u[level], s.u[level - 1], s.interp_weights)
        nas_mg_resid!(s.r[level], s.u[level], s.r[level])
        nas_mg_psinv!(s.u[level], s.r[level], c)
    end
    cuda_mg_interp!(s.u[end], s.u[end - 1], s.interp_weights)
    nas_mg_resid!(s.r[end], s.u[end], s.rhs)
    nas_mg_psinv!(s.u[end], s.r[end], c)
    return nothing
end

function cuda_nas_mg_norm2(residual)
    n = size(residual, 1)
    interior = @view residual[2:(n - 1), 2:(n - 1), 2:(n - 1)]
    return sum(abs2, interior; dims=(1, 2, 3))
end

function run!(b::NASMultiGrid, s::CUDANASMGState)
    p = nas_mg_parameters(b.class)
    c = nas_mg_smoother(b.class)
    foreach(x -> fill!(x, 0.0), s.u)
    nas_mg_resid!(s.r[end], s.u[end], s.rhs)
    cuda_nas_mg_norm2(s.r[end])
    cycle! = cuda_nas_mg_impl(b) == "separable" ?
        () -> cuda_mg_separable_cycle!(s, c) : () -> nas_mg_cycle!(s.u, s.r, s.rhs, c)
    for _ in 1:p.niter
        cycle!()
        nas_mg_resid!(s.r[end], s.u[end], s.rhs)
    end
    return cuda_nas_mg_norm2(s.r[end])
end

function check_benchmark_correctness(b::NASMultiGrid, gs::GlobalSettings; mod=CUDA)
    state = only(initialize(b; mod))
    squared = run!(b, state)
    norm = sqrt(only(Array(squared)) / Float64(b.N)^3)
    return nas_mg_verified(b.class, norm) ? "pass" : "fail"
end
