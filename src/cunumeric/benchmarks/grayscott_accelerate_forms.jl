# Compare the four `@accelerate` scopes on one shared step.

abstract type AbstractGrayScottAccelerateForm{T} <: AbstractGrayScott{T} end

Base.@kwdef struct GrayScottFunctionAccelerated{T} <:
                   AbstractGrayScottAccelerateForm{T}
    N::Int
    M::Int
end

Base.@kwdef struct GrayScottBeginAccelerated{T} <: AbstractGrayScottAccelerateForm{T}
    N::Int
    M::Int
end

Base.@kwdef struct GrayScottLetAccelerated{T} <: AbstractGrayScottAccelerateForm{T}
    N::Int
    M::Int
end

Base.@kwdef struct GrayScottExpressionAccelerated{T} <:
                   AbstractGrayScottAccelerateForm{T}
    N::Int
    M::Int
end

name(::GrayScottFunctionAccelerated) = "grayscott_function_accelerated"
name(::GrayScottBeginAccelerated) = "grayscott_begin_accelerated"
name(::GrayScottLetAccelerated) = "grayscott_let_accelerated"
name(::GrayScottExpressionAccelerated) = "grayscott_expression_accelerated"

function cuda_runnable(b::AbstractGrayScottAccelerateForm{T}) where {T}
    return GrayScottBaseline{T}(; N=b.N, M=b.M)
end

function _define_grayscott_accelerated_step(type, form=:function; aggressive=false)
    body = deepcopy(GRAYSCOTT_STEP_BODY)
    signature = :(_gs_step!(b::$type, u, v, u_new, v_new, args::GSParams))
    definition = _define_accelerated_definition(signature, body, form; aggressive)
    return Core.eval(@__MODULE__, definition)
end

if CUNUMERIC_BENCH_ACCELERATE
    # `grayscott` merges the u/v updates into one launch; the form variants keep
    # the default so they compare scope forms only.
    _define_grayscott_accelerated_step(GrayScottAccelerated; aggressive=true)
    for (type, form) in (
        (GrayScottFunctionAccelerated, :function),
        (GrayScottBeginAccelerated, :begin),
        (GrayScottLetAccelerated, :let),
    )
        _define_grayscott_accelerated_step(type, form)
    end
end

# Expression form accelerates each assignment independently.
function accelerate_grayscott_rhs(body::Expr)
    statements = Any[]
    for statement in body.args
        if statement isa LineNumberNode
            push!(statements, statement)
        elseif statement isa Expr && statement.head === :(=)
            lhs, rhs = statement.args
            push!(statements, :($lhs = @accelerate $rhs))
        else
            error("Gray-Scott expression benchmark expected assignments; got $(repr(statement))")
        end
    end
    return Expr(:block, statements...)
end

let body = accelerate_grayscott_rhs(deepcopy(GRAYSCOTT_STEP_BODY))
    if CUNUMERIC_BENCH_ACCELERATE
        @eval function _gs_step!(
            b::GrayScottExpressionAccelerated, u, v, u_new, v_new, args::GSParams
        )
            $body
        end
    end
end

register_benchmark("grayscott_function_accelerated", GrayScottFunctionAccelerated)
register_benchmark("grayscott_begin_accelerated", GrayScottBeginAccelerated)
register_benchmark("grayscott_let_accelerated", GrayScottLetAccelerated)
register_benchmark("grayscott_expression_accelerated", GrayScottExpressionAccelerated)

benchmark_array_module(::Type{<:AbstractGrayScottAccelerateForm}) = cuNumeric
