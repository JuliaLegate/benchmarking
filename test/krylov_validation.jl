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

layout = only(expr for expr in source.args if
    expr isa Expr && expr.head == :function &&
    expr.args[1] isa Expr && expr.args[1].head == :call &&
    expr.args[1].args[1] == :dagger_krylov_layout)
Core.eval(@__MODULE__, layout)

@testset "Dagger Krylov column strips" begin
    for n in (64, 67), gpus in (1, 2, 4, 8), splits in (1, 2, 4), solver in ("cg", "bicgstab")
        global N = n
        global GPUS = gpus
        global BLOCKS_PER_GPU = splits
        global SOLVER = solver
        lower = fill(solver == "cg" ? -0.5 : -0.3, n - 1)
        upper = fill(solver == "cg" ? -0.5 : -0.8, n - 1)
        A = Matrix(Tridiagonal(lower, collect(range(2, 4; length=n)), upper))
        x = sin.(1:n)
        blocks, owners = dagger_krylov_layout(A, collect(1:gpus))
        vblocks, vowners = dagger_krylov_layout(x, collect(1:gpus))
        @test blocks[1] == n
        @test size(owners, 1) == 1
        @test size(owners, 2) <= gpus * splits
        @test vec(owners) == vowners
        @test Set(vowners) == Set(1:gpus)
        @test issorted(vowners) # Contiguous groups, like Gray-Scott.
        @test vblocks == (blocks[2],)
        n == 64 && @test count(==(1), owners) == splits

        # Each matrix strip consumes the matching vector segment. Summing
        # full-height partial products must reproduce the original operator.
        y = zeros(n)
        for firstcol in 1:blocks[2]:n
            cols = firstcol:min(firstcol + blocks[2] - 1, n)
            y .+= A[:, cols] * x[cols]
        end
        @test y ≈ A * x
    end
end

const TOL = 1e-5
TUNING = false
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
    global TUNING = true
    @test last(check(0.9 .* x)) ≈ 0.1
    for invalid in (zero(x), 3 .* x, fill(NaN, length(x)), fill(Inf, length(x)))
        @test_throws ErrorException check(invalid)
    end
end
end
