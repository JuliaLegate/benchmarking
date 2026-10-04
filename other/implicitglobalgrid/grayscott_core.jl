# Array-generic pieces shared by the CUDA benchmark and CPU partition tests.
# coords are zero-based Cartesian MPI coordinates; local arrays include halos.
@views function igg_seed!(u, v, coords, seed_u, seed_v)
    ox, oy = coords[1] * (size(u, 1) - 2), coords[2] * (size(u, 2) - 2)
    sx = min(size(seed_u, 1) - ox, size(u, 1) - 2)
    sy = min(size(seed_u, 2) - oy, size(u, 2) - 2)
    if sx > 0 && sy > 0
        u[2:sx+1, 2:sy+1] .= seed_u[ox+1:ox+sx, oy+1:oy+sy]
        v[2:sx+1, 2:sy+1] .= seed_v[ox+1:ox+sx, oy+1:oy+sy]
    end
    return nothing
end

# Read only the old state; write both species once, with no grid intermediates.
@inline function igg_update_cell!(i, j, u, v, un, vn, p)
    @inbounds begin
        up, vp = u[i, j], v[i, j]
        lu = (u[i+1, j] - 2up + u[i-1, j]) / p.dx2 +
             (u[i, j+1] - 2up + u[i, j-1]) / p.dx2
        lv = (v[i+1, j] - 2vp + v[i-1, j]) / p.dx2 +
             (v[i, j+1] - 2vp + v[i, j-1]) / p.dx2
        uvv = up * vp * vp
        un[i, j] = up + p.dt * (p.cu * lu - uvv + p.f * (one(up) - up))
        vn[i, j] = vp + p.dt * (p.cv * lv + uvv - (p.f + p.k) * vp)
    end
    return nothing
end
