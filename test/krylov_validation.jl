module KrylovValidationTests
using Test, LinearAlgebra

# Load the actual validation function without importing GPU backends or
# launching the benchmark. Backend boundaries below are CPU test doubles.
source = Meta.parseall(read(joinpath(@__DIR__, "..", "composability", "krylov", "krylov.jl"), String))
validation = only(expr for expr in source.args if
    expr isa Expr && expr.head == :function &&
    expr.args[1] isa Expr && expr.args[1].head == :call &&
    expr.args[1].args[1] == :checked_solve!)
Core.eval(@__MODULE__, validation)

buffer_constructor = only(expr for expr in source.args if
    expr isa Expr && expr.head == :function &&
    expr.args[1] isa Expr && expr.args[1].head == :call &&
    expr.args[1].args[1] == :dense_operator_buffer)
Core.eval(@__MODULE__, buffer_constructor)

@testset "Dense operator host layout" begin
    for T in (Float32, Float64), n in (2, 7, 16), symmetric in (true, false)
        lower = fill(T(-0.3), n - 1)
        upper = fill(T(symmetric ? -0.3 : -0.8), n - 1)
        reference = Tridiagonal(lower, T.(range(2, 4; length=n)), upper)
        column_major = dense_operator_buffer(reference)
        row_major = dense_operator_buffer(reference; row_major=true)
        @test column_major == Matrix(reference)
        @test eltype(row_major) == T
        @test size(row_major) == size(reference)
        # Read the packed buffer using the row-major index used by Legate.
        reconstructed = [row_major[(i - 1) * n + j] for i in 1:n, j in 1:n]
        @test reconstructed == Matrix(reference)
    end
end

const TOL = 1e-5
const SYNCHRONIZED = Ref(false)
permitted_solve!(w, A, b) = (SYNCHRONIZED[] = false; (w.x, 12, w.solved))
sync(w) = (SYNCHRONIZED[] = true)
host_array(x) = (SYNCHRONIZED[] || error("Missing synchronization"); x)
correct_storage(x) = error("Final placement must not gate numerical validation")

@testset "Krylov independent residual validation" begin
    A = Tridiagonal(fill(-0.5, 7), collect(range(2.0, 4.0; length=8)), fill(-0.5, 7))
    b = [sin(i) + 1 for i in 1:8]
    x = A \ b
    check(value; solved=true) = checked_solve!((x=value, solved=solved), A, b, A, b)

    # A correct host vector passes even if the solver flag says otherwise.
    for solved in (true, false)
        iterations, residual = check(x; solved)
        @test iterations == 12
        @test residual <= TOL
    end
    @test last(check(Float32.(x))) <= TOL
    @test last(check((1 + TOL / 2) .* x)) <= TOL
    @test_throws ErrorException check((1 + 2TOL) .* x)

    # Reproduce a partially uncleared solution despite a successful solver flag.
    stale = copy(x)
    stale[1:4] .*= 2
    @test_throws ErrorException check(stale)
    for value in (NaN, Inf, -Inf)
        invalid = copy(x)
        invalid[1] = value
        @test_throws ErrorException check(invalid)
    end
    for invalid in (1.0, x[1:7], reshape(x, :, 1))
        @test_throws ErrorException check(invalid)
    end
end
end
