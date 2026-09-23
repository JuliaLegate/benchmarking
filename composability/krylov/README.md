# Krylov solver composability

This standalone benchmark runs CG and BiCGSTAB on the same dense operator and
right-hand side using four single-GPU paths:

| Plot label | Solver call |
| --- | --- |
| CUDA | Stock `Krylov.cg!` / `Krylov.bicgstab!` on `CuArray` |
| Dagger | Stock Krylov solver on GPU-backed Dagger arrays |
| cuNumeric | Stock Krylov solver on `NDArray` |
| cuNumeric local | Concise local solver loop on `NDArray`, with `@accelerate` vector-update helpers |

The weak-scaling run uses Dagger, cuNumeric, and cuNumeric local. Dagger uses
square tiles and the requested GPU scope; cuNumeric uses Legate's `--gpus`
configuration. The Dagger checkout used for the original comparison calls CPU
BLAS from its tile GEMV fallback, so this benchmark adds a CuArray tile method
that forwards to GPU `mul!`. It does not change Krylov's solver code.

Create an environment with Julia 1.12. The cuNumeric checkout must include the
Krylov extension (including `BicgstabWorkspace(A, b::NDArray)`), currently on
[`codex/cg-benchmark-reproducer`](https://github.com/JuliaLegate/cuNumeric.jl/tree/codex/cg-benchmark-reproducer).
Use a Dagger checkout that includes its Krylov extension, or a release containing
that extension:

```sh
julia --project=composability/krylov -e 'using Pkg; Pkg.develop(path="/path/to/cuNumeric.jl"); Pkg.develop(path="/path/to/Dagger.jl"); Pkg.instantiate()'
```

If cuNumeric uses local backend-library preferences, copy its
`LocalPreferences.toml` into this environment. Preserve the resulting
`Manifest.toml` and preference file with the run record; neither is committed.

Set one or more explicit matrix dimensions for the single-GPU comparison:

```sh
BENCH_ELTYPE=Float32 BENCH_OUTPUT=results/krylov-single-f32 bash composability/krylov/run.sh single 8192 16384 32768
python composability/krylov/plot.py single results/krylov-single-f32/results.csv --output single.png
```

For weak scaling, set the **one-GPU** dimension followed by GPU counts. The
runner chooses `N(G) = round(N(1) × sqrt(G))`, keeping dense matrix elements per
GPU approximately constant. Every backend gets the same `N(G)` at each count:

```sh
BENCH_ELTYPE=Float64 BENCH_OUTPUT=results/krylov-weak-f64 bash composability/krylov/run.sh weak 8192 1 2 4 8
python composability/krylov/plot.py weak results/krylov-weak-f64/results.csv --output weak.png
```

Set `BENCH_SOLVERS=cg` or `BENCH_SOLVERS=bicgstab` to run one solver; the
default runs both. Set `BENCH_OUTPUT` for a fixed output directory; its
`results.csv`, per-case logs, and `environment.txt` are kept together. `BENCH_PROJECT` can point to an
already-instantiated equivalent environment (the default is this directory).
Set `JULIA`, `BENCH_THREADS`,
`BENCH_CPUS`, `BENCH_FBMEM`, `BENCH_SYSMEM`, or `BENCH_ZCMEM` for the machine.
Set `BENCH_TIMEOUT` to an optional per-case duration accepted by GNU `timeout`
(for example `10m`). Failed or timed-out cases keep their logs; completed CSV
rows remain plottable without hiding other backends or sizes.
`BENCH_FBMEM` is MiB per GPU and defaults to 22000; adjust it to the device.
The runner sets `LEGATE_CONFIG` separately for every GPU count. Set
`CUBLAS_WORKSPACE_CONFIG` externally, if desired, so every backend sees the same
setting. `plot.py` needs Python with Matplotlib.

Each case gets a fresh Julia process, two warm-up solves, and five synchronized
timed solves. Allocation, host-to-device transfer, compilation, explicit GC,
and an independent Float64 residual check are outside the timer. The CSV records
iteration count and residual as well as timing. Compare timings within each
solver: CG uses a symmetric tridiagonal source and BiCGSTAB uses a nonsymmetric
one; both timed operators are dense. The local loops use the same mathematical
recurrences but may differ from Krylov in floating-point order and breakdown
handling. Their `@accelerate` helpers express fused vector updates; use a
profiler before attributing a timing difference to kernel fusion.

The single-GPU and one-GPU weak-scaling cases can run on a one-GPU machine.
Multi-GPU placement, convergence, and timings need validation on the target
multi-GPU host before publication.
