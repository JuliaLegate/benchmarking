struct TimingProbe <: AbstractBenchmark{Float32}
    events::Vector{Symbol}
end
struct GrayScottTimingProbe <: AbstractGrayScott{Float32}
    events::Vector{Symbol}
end

struct ModelTimingProbe
    events::Vector{Symbol}
end
model_initialize(b::ModelTimingProbe) = (push!(b.events, :initialize); nothing)
model_run!(b::ModelTimingProbe, state) = push!(b.events, :run)
model_synchronize(b::ModelTimingProbe) = push!(b.events, :sync)
const TimingProbes = Union{TimingProbe,GrayScottTimingProbe}
initialize(b::TimingProbes; mod=Base) = (push!(b.events, :initialize); ())
run!(b::TimingProbes) = push!(b.events, :run)
cleanup!(b::TimingProbes) = push!(b.events, :cleanup)
cleanup_result!(b::TimingProbes, result, state...) = push!(b.events, :result_cleanup)
total_flops(::TimingProbes) = 6000
name(::TimingProbes) = "timing probe"

@testset "Iteration completion policy" begin
    for B in values(BENCHMARKS)
        b = build_benchmark(B, Float32, 32, 32)
        @test fence_each_iteration(b) == !(b isa AbstractGrayScott)
    end
    for B in (TimingProbe, GrayScottTimingProbe), warmup in (0, 2)
        events = Symbol[]
        b = B(events)
        ticks = Ref(0)
        clock() = (push!(events, :clock); ticks[] += 6000)
        synchronize() = push!(events, :sync)
        gs = GlobalSettings(; n_warmup=warmup, n_iter=3)
        step = fence_each_iteration(b) ? [:run, :sync] : [:run]
        expected = vcat([:initialize], repeat(vcat(step, [:result_cleanup]), warmup),
            [:sync, :clock], repeat(vcat(step, [:result_cleanup]), 2), step,
            [:sync, :clock, :result_cleanup, :cleanup])
        # Also exercises forwarding the backend callback through run_benchmark.
        result = run_benchmark(b, gs; mod=Base, clock, synchronize)
        @test events == expected
        @test result.times_ms == [2.0]
        @test result.gflops == [0.003]
    end
end

@testset "Single FT-style sample excludes final result cleanup" begin
    events = Symbol[]
    b = TimingProbe(events)
    clock() = (push!(events, :clock); length(filter(==(:clock), events)) * 1000)
    synchronize() = push!(events, :sync)
    gs = GlobalSettings(; n_warmup=0, n_iter=1)
    _trial(b, gs; mod=Base, clock, synchronize)
    @test events == [
        :initialize, :sync, :clock, :run, :sync, :sync, :clock,
        :result_cleanup, :cleanup,
    ]
end

@testset "Native model timing boundaries" begin
    events = Symbol[]
    probe = ModelTimingProbe(events)
    config = ModelWorkerConfig(
        1, "probe", Float32, "Float32", 8, 1, 3, 2, 1, false, 1, 6000.0
    )
    ticks = Ref(0)
    clock() = (push!(events, :clock); ticks[] += 6_000_000)
    time_ms, gflops = model_trial(probe, config; clock)
    @test events == [
        :initialize,
        :run, :sync,
        :run, :sync,
        :sync,
        :clock,
        :run, :sync,
        :run, :sync,
        :run, :sync,
        :sync,
        :clock,
    ]
    @test time_ms == 2.0
    @test gflops == 0.003
end

@testset "Trial cleanup on failure" begin
    events = Symbol[]
    b = TimingProbe(events)
    gs = GlobalSettings(; n_warmup=1, n_iter=1)
    @test_throws ErrorException _trial(b, gs; mod=Base,
        clock=()->error("unexpected timing"),
        synchronize=()->error("simulated execution failure"))
    @test events == [:initialize, :run, :cleanup]
end
