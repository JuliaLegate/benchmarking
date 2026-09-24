# Fixed grid spacing; an N × N grid covers [0, (N - 1) * DX]^2.
const DX = 1

# Zero Dirichlet boundaries and a five-point finite-difference Laplacian.
# The RHS writes du/dt in place. Materialized slices keep distributed arrays
# on their array backend instead of scalar-indexing a view during broadcast.
function heat!(du, u, κ, t)
    n = size(u, 1)
    fill!(du, zero(eltype(du)))
    interior = (κ / DX^2) .* (
        u[3:n, 2:(n - 1)] .+ u[1:(n - 2), 2:(n - 1)] .+
        u[2:(n - 1), 3:n] .+ u[2:(n - 1), 1:(n - 2)] .-
        4 .* u[2:(n - 1), 2:(n - 1)]
    )
    copyto!(view(du, 2:(n - 1), 2:(n - 1)), interior)
    return nothing
end
