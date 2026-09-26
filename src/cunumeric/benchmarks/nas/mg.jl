# LIMITATION: NPB MG's exact sparse right-hand side is generated on the host,
# as in NPB-GPU, then uploaded once before timing. The V-cycle itself uses
# cuNumeric views and broadcasts, allowing Legate to partition every level.
# Restriction/interpolation are separable axis passes with temporary arrays,
# not JACC's direct per-cell kernels. Every operator is @accelerate, so slices
# and temporaries are freed at their last use instead of waiting for Julia GC. 
# The common harness times initial zeroing and L2 sum-of-squares, 
# but omits NPB's Linf norm; see nas/README.md.

mutable struct CuNumericNASMGState{U,R,V,W}
    u::U
    r::R
    rhs::V
    interp_weights::W
end

const CuNumericMGIndex = Union{Colon,AbstractUnitRange{<:Integer}}

function cunumeric_mg_slice(index::Colon, n)
    return (0, n)
end
cunumeric_mg_slice(index::AbstractUnitRange, n) = (Int(first(index)) - 1, Int(last(index)))

# cuNumeric's core currently defines contiguous slicing through rank 2. MG is
# rank 3, so provide the identical view operation locally until core generalizes
# those methods. The returned NDArray shares its parent's Legate store.
function Base.getindex(
    array::cuNumeric.NDArray{T,3}, i::CuNumericMGIndex,
    j::CuNumericMGIndex, k::CuNumericMGIndex,
) where {T}
    @boundscheck checkbounds(array, i, j, k)
    slices = cuNumeric.slice_array(
        cunumeric_mg_slice(i, size(array, 1)),
        cunumeric_mg_slice(j, size(array, 2)),
        cunumeric_mg_slice(k, size(array, 3)),
    )
    return cuNumeric.nda_get_slice(array, slices)
end

function Base.setindex!(
    lhs::cuNumeric.NDArray{T,3}, rhs::cuNumeric.NDArray,
    i::CuNumericMGIndex, j::CuNumericMGIndex, k::CuNumericMGIndex,
) where {T}
    view = lhs[i, j, k]
    copyto!(view, rhs)
    cuNumeric.destroy!(view)
    return rhs
end

# Disjoint slice-to-slice assignments take cuNumeric's native copy path;
# @accelerate releases each temporary view.
cuNumeric.@accelerate function nas_mg_comm3!(u::cuNumeric.NDArray{T,3}) where {T}
    n1, n2, n3 = size(u)
    yi, zi = 2:(n2 - 1), 2:(n3 - 1)
    u[1:1, yi, zi] .= u[(n1 - 1):(n1 - 1), yi, zi]
    u[n1:n1, yi, zi] .= u[2:2, yi, zi]
    u[:, 1:1, zi] .= u[:, (n2 - 1):(n2 - 1), zi]
    u[:, n2:n2, zi] .= u[:, 2:2, zi]
    u[:, :, 1:1] .= u[:, :, (n3 - 1):(n3 - 1)]
    u[:, :, n3:n3] .= u[:, :, 2:2]
    return u
end

function initialize(b::NASMultiGrid{Float64}; mod=cuNumeric)
    p = validate_nas_mg(b)
    sizes = nas_mg_level_sizes(p)
    u = [mod.zeros(Float64, n, n, n) for n in sizes]
    r = [mod.zeros(Float64, n, n, n) for n in sizes]
    rhs = mod.NDArray(nas_mg_rhs(p))
    weights = mod.reshape(mod.NDArray([0.0, 0.5]), 1, 1, 1, 2)
    return (CuNumericNASMGState(u, r, rhs, weights),)
end

cuNumeric.@accelerate function cunumeric_mg_resid!(
    r::cuNumeric.NDArray{T,3}, u::cuNumeric.NDArray{T,3}, v::cuNumeric.NDArray{T,3},
) where {T}
    r[2:(end - 1), 2:(end - 1), 2:(end - 1)] =
        v[2:(end - 1), 2:(end - 1), 2:(end - 1)] .- NAS_MG_A[1] .* u[2:(end - 1), 2:(end - 1), 2:(end - 1)] .-
        NAS_MG_A[3] .* (
            u[2:(end - 1), 1:(end - 2), 1:(end - 2)] .+ u[2:(end - 1), 3:end, 1:(end - 2)] .+
            u[2:(end - 1), 1:(end - 2), 3:end] .+ u[2:(end - 1), 3:end, 3:end] .+
            u[1:(end - 2), 2:(end - 1), 1:(end - 2)] .+ u[3:end, 2:(end - 1), 1:(end - 2)] .+
            u[1:(end - 2), 2:(end - 1), 3:end] .+ u[3:end, 2:(end - 1), 3:end] .+
            u[1:(end - 2), 1:(end - 2), 2:(end - 1)] .+ u[3:end, 1:(end - 2), 2:(end - 1)] .+
            u[1:(end - 2), 3:end, 2:(end - 1)] .+ u[3:end, 3:end, 2:(end - 1)]
        ) .-
        NAS_MG_A[4] .* (
            u[1:(end - 2), 1:(end - 2), 1:(end - 2)] .+ u[3:end, 1:(end - 2), 1:(end - 2)] .+
            u[1:(end - 2), 3:end, 1:(end - 2)] .+ u[3:end, 3:end, 1:(end - 2)] .+
            u[1:(end - 2), 1:(end - 2), 3:end] .+ u[3:end, 1:(end - 2), 3:end] .+
            u[1:(end - 2), 3:end, 3:end] .+ u[3:end, 3:end, 3:end]
        )
    nas_mg_comm3!(r)
    return r
end

cuNumeric.@accelerate function cunumeric_mg_psinv!(
    u::cuNumeric.NDArray{T,3}, r::cuNumeric.NDArray{T,3}, c,
) where {T}
    u[2:(end - 1), 2:(end - 1), 2:(end - 1)] =
        u[2:(end - 1), 2:(end - 1), 2:(end - 1)] .+ c[1] .* r[2:(end - 1), 2:(end - 1), 2:(end - 1)] .+
        c[2] .* (
            r[1:(end - 2), 2:(end - 1), 2:(end - 1)] .+ r[3:end, 2:(end - 1), 2:(end - 1)] .+
            r[2:(end - 1), 1:(end - 2), 2:(end - 1)] .+ r[2:(end - 1), 3:end, 2:(end - 1)] .+
            r[2:(end - 1), 2:(end - 1), 1:(end - 2)] .+ r[2:(end - 1), 2:(end - 1), 3:end]
        ) .+
        c[3] .* (
            r[2:(end - 1), 1:(end - 2), 1:(end - 2)] .+ r[2:(end - 1), 3:end, 1:(end - 2)] .+
            r[2:(end - 1), 1:(end - 2), 3:end] .+ r[2:(end - 1), 3:end, 3:end] .+
            r[1:(end - 2), 2:(end - 1), 1:(end - 2)] .+ r[3:end, 2:(end - 1), 1:(end - 2)] .+
            r[1:(end - 2), 2:(end - 1), 3:end] .+ r[3:end, 2:(end - 1), 3:end] .+
            r[1:(end - 2), 1:(end - 2), 2:(end - 1)] .+ r[3:end, 1:(end - 2), 2:(end - 1)] .+
            r[1:(end - 2), 3:end, 2:(end - 1)] .+ r[3:end, 3:end, 2:(end - 1)]
        )
    nas_mg_comm3!(u)
    return u
end

# Restriction and interpolation are separable axis passes. cuNumeric reshapes in
# C order, so each pass moves its axis last; one function per axis keeps every
# scope straight-line for @accelerate, which frees each named temporary.
cuNumeric.@accelerate function cunumeric_mg_pair_sum(back::cuNumeric.NDArray{T,3}) where {T}
    left = cuNumeric.reshape(
        back[:, :, 2:(end - 1)], size(back, 1), size(back, 2), (size(back, 3) - 2) ÷ 2, 2
    )
    right = cuNumeric.reshape(
        back[:, :, 3:end], size(back, 1), size(back, 2), (size(back, 3) - 2) ÷ 2, 2
    )
    total = sum(left; dims=4) .+ sum(right; dims=4)
    return cuNumeric.reshape(total, size(back, 1), size(back, 2), (size(back, 3) - 2) ÷ 2)
end

cuNumeric.@accelerate function cunumeric_mg_restrict_x(fine::cuNumeric.NDArray{T,3}) where {T}
    back = permutedims(fine, (2, 3, 1))
    reduced = cunumeric_mg_pair_sum(back)
    return permutedims(reduced, (3, 1, 2))
end

cuNumeric.@accelerate function cunumeric_mg_restrict_y(fine::cuNumeric.NDArray{T,3}) where {T}
    back = permutedims(fine, (1, 3, 2))
    reduced = cunumeric_mg_pair_sum(back)
    return permutedims(reduced, (1, 3, 2))
end

cuNumeric.@accelerate function cunumeric_mg_restrict!(
    coarse::cuNumeric.NDArray{T,3}, fine::cuNumeric.NDArray{T,3},
) where {T}
    rx = cunumeric_mg_restrict_x(fine)
    rxy = cunumeric_mg_restrict_y(rx)
    rxyz = cunumeric_mg_pair_sum(rxy)
    coarse[2:(end - 1), 2:(end - 1), 2:(end - 1)] = rxyz ./ 16.0
    nas_mg_comm3!(coarse)
    return coarse
end

cuNumeric.@accelerate function cunumeric_mg_interp_last(
    back::cuNumeric.NDArray{T,3}, weights,
) where {T}
    lo = cuNumeric.reshape(
        back[:, :, 1:(end - 1)], size(back, 1), size(back, 2), size(back, 3) - 1, 1
    )
    hi = cuNumeric.reshape(
        back[:, :, 2:end], size(back, 1), size(back, 2), size(back, 3) - 1, 1
    )
    mixed = lo .+ weights .* (hi .- lo)
    return cuNumeric.reshape(mixed, size(back, 1), size(back, 2), 2(size(back, 3) - 1))
end

# Legate cannot slice the composed reshape/transpose in the next pass, so the
# x and y passes copy their result.
cuNumeric.@accelerate function cunumeric_mg_interp_x(
    coarse::cuNumeric.NDArray{T,3}, weights,
) where {T}
    back = permutedims(coarse, (2, 3, 1))
    mixed = cunumeric_mg_interp_last(back, weights)
    front = permutedims(mixed, (3, 1, 2))
    return copy(front)
end

cuNumeric.@accelerate function cunumeric_mg_interp_y(
    coarse::cuNumeric.NDArray{T,3}, weights,
) where {T}
    back = permutedims(coarse, (1, 3, 2))
    mixed = cunumeric_mg_interp_last(back, weights)
    front = permutedims(mixed, (1, 3, 2))
    return copy(front)
end

cuNumeric.@accelerate function cunumeric_mg_interp!(
    fine::cuNumeric.NDArray{T,3}, coarse::cuNumeric.NDArray{T,3}, weights,
) where {T}
    ix = cunumeric_mg_interp_x(coarse, weights)
    ixy = cunumeric_mg_interp_y(ix, weights)
    ixyz = cunumeric_mg_interp_last(ixy, weights)
    fine .= fine .+ ixyz
    return fine
end

function cunumeric_mg_cycle!(s, c)
    finest = length(s.u)
    for level in finest:-1:2
        cunumeric_mg_restrict!(s.r[level - 1], s.r[level])
    end
    fill!(s.u[1], 0.0)
    cunumeric_mg_psinv!(s.u[1], s.r[1], c)
    for level in 2:(finest - 1)
        fill!(s.u[level], 0.0)
        cunumeric_mg_interp!(s.u[level], s.u[level - 1], s.interp_weights)
        cunumeric_mg_resid!(s.r[level], s.u[level], s.r[level])
        cunumeric_mg_psinv!(s.u[level], s.r[level], c)
    end
    cunumeric_mg_interp!(s.u[end], s.u[end - 1], s.interp_weights)
    cunumeric_mg_resid!(s.r[end], s.u[end], s.rhs)
    cunumeric_mg_psinv!(s.u[end], s.r[end], c)
    return nothing
end

function cunumeric_mg_norm2(residual)
    n = size(residual, 1)
    interior = residual[2:(n - 1), 2:(n - 1), 2:(n - 1)]
    squared = sum(abs2, interior)
    cuNumeric.destroy!(interior)
    return squared
end

function run!(b::NASMultiGrid, s::CuNumericNASMGState)
    p = nas_mg_parameters(b.class)
    c = nas_mg_smoother(b.class)
    foreach(x -> fill!(x, 0.0), s.u)
    cunumeric_mg_resid!(s.r[end], s.u[end], s.rhs)
    cunumeric_mg_norm2(s.r[end])
    for _ in 1:p.niter
        cunumeric_mg_cycle!(s, c)
        cunumeric_mg_resid!(s.r[end], s.u[end], s.rhs)
    end
    return cunumeric_mg_norm2(s.r[end])
end

function check_benchmark_correctness(b::NASMultiGrid, gs::GlobalSettings; mod=cuNumeric)
    state = only(initialize(b; mod))
    squared = run!(b, state)
    norm = sqrt(cuNumeric.@allowscalar squared[] / Float64(b.N)^3)
    return nas_mg_verified(b.class, norm) ? "pass" : "fail"
end
