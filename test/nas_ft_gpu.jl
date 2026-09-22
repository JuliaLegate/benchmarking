# Run with --project=environments/cunumeric and
# LEGATE_AUTO_CONFIG=0 LEGATE_CONFIG="--gpus=1 --cpus=1 --fbmem=12288".
using Test, cuNumeric, AbstractFFTs
include("../src/core.jl")
include("../src/benchmarks/nas/ft.jl")
include("../src/cunumeric/benchmarks/nas/ft.jl")

function verify_ft_trial(b)
    state = only(initialize(b))
    try
        for _ in 1:2
            result = run!(b, state)
            cuNumeric.issue_execution_fence(; block=true)
            got = ComplexF64[cuNumeric.@allowscalar(x[]) for x in result]
            @test nas_ft_verified(b.class, got)
            cleanup_result!(b, result, state)
        end
    finally
        cleanup!(b, state)
    end
end

@testset "cuNumeric NAS FT buffer reuse" begin
    for cls in (isempty(ARGS) ? ["S", "W", "B"] : ARGS)
        @testset "class $cls" begin
            p = nas_ft_parameters(cls)
            b = NASFourierTransform{Float64}(; N=p.nx, M=p.ny, class=cls)
            for _ in 1:3
                GC.gc(true)
                verify_ft_trial(b)
            end
        end
    end
end
