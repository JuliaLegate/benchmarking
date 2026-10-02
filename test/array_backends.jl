# CPU-only checks of the catalogs loaded by the actual array workers.
module CUDAOnlyCatalog
using LinearAlgebra
include("../src/core.jl")
include_benchmarks(:cudajl)
end

module ArrayBackendSeparation
using Test, LinearAlgebra, Random
include("../src/core.jl")
include_benchmarks(:cunumeric)
const CUNUMERIC_CATALOG = copy(BENCHMARKS)
# Load the second backend into the same module to exercise shared dispatch.
include("../src/cuda/benchmarks.jl")
include("../src/models.jl")
include("../src/memory.jl")

@testset "Backend catalogs and shared sizing" begin
    cuda = parentmodule(@__MODULE__).CUDAOnlyCatalog
    expected = Set(k for k in keys(CUNUMERIC_CATALOG) if supports_benchmark(CUDAJLModel(), k))
    @test Set(keys(cuda.BENCHMARKS)) == expected
    @test cuda.BENCHMARKS["cg"] === cuda.CUDACG
    @test cuda.BENCHMARKS["grayscott"] === cuda.CUDAGrayScott
    @test !isdefined(cuda, :cuNumeric)
    @test !isdefined(cuda, :ConjugateGradientAccelerated)
    @test !isdefined(cuda, :GrayScottAccelerated)
    for key in expected
        B, C = CUNUMERIC_CATALOG[key], BENCHMARKS[key]
        @test B !== C
        @test startswith(string(nameof(C)), "CUDA")
        T = startswith(key, "nas_") ? Float64 : Float32
        N, M = startswith(key, "nas_") ? class_dims(B, Dict()) :
            key in ("cg", "cg_plain") ? (32, 1) : (32, 8)
        b, c = build_benchmark(B, T, N, M), build_benchmark(C, T, N, M)
        @test name(b) == name(c) == key
        @test dims(b) == dims(c)
        @test data(b) == data(c)
        @test total_flops(b) == total_flops(c)
        @test total_space(b) == total_space(c)
        @test estimate_scaling(b, 1) == estimate_scaling(c, 1)
        @test typeof(correctness_problem(c)) === typeof(c)
        context = MemoryContext(; model=:cudajl, workspace_bytes=0)
        @test peak_bytes(memory_estimate(b, context)) == peak_bytes(memory_estimate(c, context))
        if key in ("gemm", "montecarlo", "grayscott", "grayscott_plain",
                   "dmd_baseline", "poisson_fft", "tensor_projection3", "tensor_contract4")
            @test fit_one_gpu(B, T; budget=2^20, M_hint=8) ==
                  fit_one_gpu(C, T; budget=2^20, M_hint=8)
        end
    end
end

@testset "CG backend types share the same solve" begin
    for (B, C) in ((ConjugateGradientAccelerated, CUDACG),
                   (ConjugateGradientBenchmark, CUDACGPlain)),
        T in (Float32, Float64), every in (1, 4, 30)
        b, c = B{T}(; N=17, check_every=every, max_iter=60),
               C{T}(; N=17, check_every=every, max_iter=60)
        bs, cs = only(initialize(b; mod=Base)), only(initialize(c; mod=Base))
        @test which(run!, (typeof(b), typeof(bs))) === which(run!, (typeof(c), typeof(cs)))
        @test run!(b, bs) == run!(c, cs)
        @test bs.x == cs.x
        A = Tridiagonal(ones(T, 16), fill(T(4), 17), ones(T, 16))
        @test cs.x ≈ A \ fill(T(0.5), 17) rtol=(T === Float32 ? 3e-5 : 3e-8)
    end
end

@testset "Gray-Scott backend types share timesteps" begin
    for (B, C) in ((GrayScottAccelerated, CUDAGrayScott),
                   (GrayScottBaseline, CUDAGrayScottPlain)), T in (Float32, Float64)
        b, c = B{T}(; N=17, M=13), C{T}(; N=17, M=13)
        bs = only(initialize(b; mod=Base, deterministic=true))
        cs = only(initialize(c; mod=Base, deterministic=true))
        @test which(run!, (typeof(b), typeof(bs))) === which(run!, (typeof(c), typeof(cs)))
        @test !fence_each_iteration(c)
        for _ in 1:5
            run!(b, bs)
            run!(c, cs)
            @test bs.u == cs.u
            @test bs.v == cs.v
        end
    end
end

@testset "Other shared array algorithms" begin
    Random.seed!(123)
    for T in (Float32, Float64)
        b, c = GEMM{T}(; N=8, M=4), CUDAGEMM{T}(; N=8, M=4)
        bs = correctness_seed(b)
        cs = deepcopy(bs)
        run!(b, bs...)
        run!(c, cs...)
        @test bs[1] == cs[1]
        x = T[0, 0.5, 1, 2, 5]
        @test run!(MonteCarloIntegration{T}(; n_samples=5), x) ==
              run!(CUDAMonteCarlo{T}(; n_samples=5), x)
        @test _montecarlo_mapreduce(CUDAMonteCarlo{T}(; n_samples=5), x) ≈
              (T(10)/length(x))*sum(exp(-v^2) for v in x)
        for (B, C) in ((TensorProjection3, CUDATensorProjection3),
                       (TensorContract4, CUDATensorContract4))
            b, c = B{T}(; N=3), C{T}(; N=3)
            bs = initialize(b; mod=Base)
            cs = deepcopy(bs)
            @test run!(b, bs...) == run!(c, cs...)
        end
        X = rand(T, 16, 5)
        b, c = DMDBaseline{T}(; N=16, M=5), CUDADMD{T}(; N=16, M=5)
        @test correctness_result(b, nothing, run!(b, copy(X))) ≈
              correctness_result(c, nothing, run!(c, copy(X)))
    end
end
end
