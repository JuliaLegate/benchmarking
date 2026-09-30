# Fixed grid spacing; an N × N grid covers [0, (N - 1) * DX]^2.
const DX = 1

# Zero Dirichlet boundaries and a five-point finite-difference Laplacian.
# The RHS writes du/dt in place. CUDA and cuNumeric use views of the five
# stencil windows. Dagger overrides heat_slice to keep distributed slices on
# its array backend; a view of a DArray falls back to scalar indexing.
heat_slice(u, rows, cols) = @views u[rows, cols]

function heat!(du, u, κ, t)
    n = size(u, 1)
    fill!(du, zero(eltype(du)))
    interior = (κ / DX^2) .* (
        heat_slice(u, 3:n, 2:(n - 1)) .+
        heat_slice(u, 1:(n - 2), 2:(n - 1)) .+
        heat_slice(u, 2:(n - 1), 3:n) .+
        heat_slice(u, 2:(n - 1), 1:(n - 2)) .-
        4 .* heat_slice(u, 2:(n - 1), 2:(n - 1))
    )
    copyto!(view(du, 2:(n - 1), 2:(n - 1)), interior)
    return nothing
end
