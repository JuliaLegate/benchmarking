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
