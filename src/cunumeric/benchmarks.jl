# Load backend-owned concrete types before runtime-specific implementations.
include(joinpath(@__DIR__, "benchmarks", "gemm.jl"))
include(joinpath(@__DIR__, "benchmarks", "cg.jl"))
include(joinpath(@__DIR__, "benchmarks", "grayscott.jl"))
include(joinpath(@__DIR__, "benchmarks", "dmd.jl"))
include(joinpath(@__DIR__, "benchmarks", "montecarlo.jl"))
include(joinpath(@__DIR__, "benchmarks", "poisson_fft.jl"))
include(joinpath(@__DIR__, "benchmarks", "tensor_contractions.jl"))
include(joinpath(@__DIR__, "benchmarks", "nas", "types.jl"))
if CUNUMERIC_BENCH_RUNTIME
    include(joinpath(@__DIR__, "benchmarks", "nas", "ep.jl"))
    include(joinpath(@__DIR__, "benchmarks", "nas", "ft.jl"))
    CUNUMERIC_BENCH_ACCELERATE && include(joinpath(@__DIR__, "benchmarks", "nas", "mg.jl"))
end
include(joinpath(@__DIR__, "benchmarks", "grayscott_accelerate_forms.jl"))
