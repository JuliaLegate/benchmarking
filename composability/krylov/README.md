# Krylov solver composability

This standalone benchmark compares stock Krylov.jl with concise, equivalent
unpreconditioned CG and BiCGSTAB loops. `plain` uses separate vector updates;
`fused` combines updates using broadcasts in straight-line `@accelerate`
helpers. Each mode uses the same dense input matrix and convergence tolerance.
BiCGSTAB tests a nonsymmetric matrix. The Julia loop, dot products, GEMV calls,
and convergence checks remain outside `@accelerate`. cuNumeric runs permit
scalar fetches at convergence checks and promotion of `I`'s Bool coefficient
in Krylov's BiCGSTAB path.
The CG input preserves the dense matrix and right-hand side from the earlier
single-GPU Krylov comparison; the nonsymmetric BiCGSTAB case changes its two
off-diagonals.

Create the local environment with Julia 1.12 and a cuNumeric checkout containing
the Krylov extension (including `BicgstabWorkspace(A, b::NDArray)`). The
extension is currently on
[`codex/cg-benchmark-reproducer`](https://github.com/JuliaLegate/cuNumeric.jl/tree/codex/cg-benchmark-reproducer):

```sh
julia --project=composability/krylov -e 'using Pkg; Pkg.develop(path="/path/to/cuNumeric.jl"); Pkg.instantiate()'
```

If your cuNumeric installation uses local backend-library preferences, copy its
`LocalPreferences.toml` into `composability/krylov/`. On dubliner, that was necessary
to select the installed Legate and cuPyNumeric libraries. Keep the generated
`Manifest.toml` and preference file with your run record; neither is committed.

Run from the repository root, specifying one or more matrix dimensions:

```sh
BENCH_ELTYPE=Float32 bash composability/krylov/run.sh 8192 16384 32768
BENCH_ELTYPE=Float64 BENCH_BACKENDS=cuNumeric bash composability/krylov/run.sh 8192
```

The launcher runs each backend, solver, and mode in a fresh process and writes
per-case logs, a results CSV, and an environment record. Override `JULIA`,
`BENCH_OUTPUT`, or `LEGATE_CONFIG` as needed. On H100, begin with the default
cuBLAS workspace; set `CUBLAS_WORKSPACE_CONFIG` before launch if testing a
fixed workspace budget, and use the same setting for every backend.

Two warm-up solves precede five synchronized timed solves. Workspace allocation,
host-to-device transfer, host residual validation, compilation, and explicit GC
are outside the timer. Every mode must converge and pass an independently
computed Float64 relative residual check. Compare timings *within* a solver:
CG and BiCGSTAB use different matrix symmetries and iterations. The stock and
local implementations have the same mathematical recurrences, but their floating
point order and breakdown handling can differ; report iteration counts and
residuals alongside time. The `fused` label describes rewritten update
expressions; verify actual kernel counts with a profiler before attributing a
timing change to fusion.

For one GPU, the default Legate setting allows 22000 MiB framebuffer memory.
For multi-GPU runs, set `LEGATE_CONFIG` to the available GPUs and memory and use
`BENCH_BACKENDS=cuNumeric`. CUDA's `CuArray` case uses one GPU. Weak scaling
can use N proportional to sqrt(GPU count) to hold dense matrix bytes per GPU
approximately constant, but distribution and communication need profiling
before interpreting scaling results.
