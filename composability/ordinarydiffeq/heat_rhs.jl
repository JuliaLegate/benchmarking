# Fixed grid spacing; an N × N grid covers [0, (N - 1) * DX]^2.
const DX = 1

# Zero Dirichlet boundaries and a five-point finite-difference Laplacian.
# The RHS writes du/dt in place with whole-array fill and sliced broadcast.
function heat!(du, u, κ, t)
    n = size(u, 1)
    fill!(du, zero(eltype(du)))
    @views du[2:(n - 1), 2:(n - 1)] .= (κ / DX^2) .* (
        u[3:n, 2:(n - 1)] .+ u[1:(n - 2), 2:(n - 1)] .+
        u[2:(n - 1), 3:n] .+ u[2:(n - 1), 1:(n - 2)] .-
        4 .* u[2:(n - 1), 2:(n - 1)]
    )
    return nothing
end
