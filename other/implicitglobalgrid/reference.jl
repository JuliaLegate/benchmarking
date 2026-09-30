# Evaluate the actual cuNumeric/CUDA step expression without loading its runtime.
# Keeping the reference in one place also makes changes to that benchmark visible
# to these checks instead of validating against a second copy of the equations.
module GrayScottReference
const source = joinpath(@__DIR__, "..", "..", "src", "benchmarks", "grayscott.jl")
const definition = only(filter(Meta.parseall(read(source, String)).args) do expr
    expr isa Expr && expr.head == :const && expr.args[1] isa Expr &&
        expr.args[1].head == :(=) && expr.args[1].args[1] == :GRAYSCOTT_STEP_BODY
end)
Core.eval(@__MODULE__, definition)
@eval step!(u, v, u_new, v_new, args) = $GRAYSCOTT_STEP_BODY
end
