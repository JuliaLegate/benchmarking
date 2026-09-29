# Sweep Dagger Gray-Scott block sizes on one GPU; hardcode the winner as
# DAGGER_GS_BLOCK in grayscott.jl.
#   LD_LIBRARY_PATH="" julia --project=environments/dagger src/dagger/benchmarks/grayscott_tune.jl <N>
include(joinpath(@__DIR__, "..", "..", "model_worker.jl"))

using CUDA: CUDA
using Dagger: Dagger
import Dagger: @stencil, Pad, Wrap
using Printf

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "grayscott.jl"))

const T = Float32
length(ARGS) == 1 || error("usage: grayscott_tune.jl <N>")
const N = parse(Int, ARGS[1])
# Powers of two from 256 below N, N/8, N/4, N/2, then one block covering the whole grid.
const BLOCKS = sort!(unique!([[2^k for k in 8:30 if 2^k < N]; [cld(N, d) for d in (8, 4, 2)]; N]))
const WARMUP = 2
const STEPS = 10

scope = Dagger.scope(; cuda_gpus=[1])
processors = sort!(collect(Dagger.compatible_processors(scope)); by=string)

results = map(BLOCKS) do block
    b = dagger_grayscott(T, N, N, 1, scope, processors; block)
    s = model_initialize(b)
    foreach(_ -> model_run!(b, s), 1:WARMUP)
    model_synchronize(b)
    t = @elapsed begin
        foreach(_ -> model_run!(b, s), 1:STEPS)
        model_synchronize(b)
    end
    ms = 1e3t/STEPS
    @printf("block=%5d  blocks=%6d  %10.3f ms/step\n", block, cld(N, block)^2, ms)
    s = nothing
    GC.gc()
    return block => ms
end

best = argmin(last, results)
println("best: DAGGER_GS_BLOCK = $(first(best))  ($(round(last(best); digits=3)) ms/step, N=$N)")
