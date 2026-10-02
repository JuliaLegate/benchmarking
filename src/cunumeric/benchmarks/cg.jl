Base.@kwdef struct ConjugateGradientBenchmark{T} <: AbstractConjugateGradient{T}
    N::Int
    M::Int = 1
    check_every::Int = 10
    max_iter::Int = 1000
end

name(::ConjugateGradientBenchmark) = "cg_plain"
register_benchmark("cg_plain", ConjugateGradientBenchmark)

Base.@kwdef struct ConjugateGradientAccelerated{T} <: AbstractConjugateGradient{T}
    N::Int
    M::Int = 1
    check_every::Int = 10
    max_iter::Int = 1000
end

name(::ConjugateGradientAccelerated) = "cg"
register_benchmark("cg", ConjugateGradientAccelerated)

if CUNUMERIC_BENCH_ACCELERATE
    let body = deepcopy(CG_STEP_BODY)
        signature = :(
            cg_step!(
                b::ConjugateGradientAccelerated{T}, x, r, p, Ap, lower, diagonal, upper, rho
            ) where {T}
        )
        @eval $(_define_accelerated_definition(signature, body))
    end
end

benchmark_array_module(::Type{<:Union{ConjugateGradientBenchmark,ConjugateGradientAccelerated}}) = cuNumeric
