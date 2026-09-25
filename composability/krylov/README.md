# Krylov solver composability

This standalone benchmark runs CG on the same dense operator and right-hand
side using four single-GPU paths:

| Plot label | Solver call |
| --- | --- |
| CUDA | Stock `Krylov.cg!` on `CuArray` |
| Dagger | Stock Krylov solver on GPU-backed Dagger arrays |
| cuNumeric | Stock Krylov solver on `NDArray` |
| cuNumeric local | Concise local solver loop on `NDArray`, with `@accelerate` vector-update helpers |

The weak-scaling run uses Dagger, cuNumeric, and cuNumeric local. Dagger uses
square tiles and the requested GPU scope; cuNumeric uses Legate's `--gpus`
configuration. The Dagger checkout used for the original comparison calls CPU
BLAS from its tile GEMV fallback, so this benchmark adds a CuArray tile method
that forwards to GPU `mul!`. It does not change Krylov's solver code.

Use Julia 1.13 and cuNumeric's `codex/ode-scalar-broadcast` branch, which
contains the Krylov extension. Keep the benchmark environment outside the
checkout:

```sh
mkdir -p /opt/bench-envs/krylov
cp composability/krylov/Project.toml /opt/bench-envs/krylov/Project.toml
julia --project=/opt/bench-envs/krylov -e 'using Pkg; Pkg.develop([PackageSpec(path="/opt/cuNumeric.jl"), PackageSpec(path="/opt/cuNumeric.jl/lib/CNPreferences")]); Pkg.instantiate()'
export BENCH_PROJECT=/opt/bench-envs/krylov
```

If cuNumeric uses local backend-library preferences, copy its
`LocalPreferences.toml` into this environment. Preserve the resulting
`Manifest.toml` and preference file with the run record; neither is committed.

Set one or more explicit matrix dimensions for the single-GPU comparison:

```sh
BENCH_ELTYPE=Float32 BENCH_OUTPUT=results/krylov-single-f32 bash composability/krylov/run.sh single 1024 2048 4096 8192 16384 32768 65536
```

For weak scaling, set the **one-GPU** dimension followed by GPU counts. The
runner chooses `N(G) = round(N(1) × sqrt(G))`, keeping dense matrix elements per
GPU approximately constant. Use the largest dimension that passed all four
single-GPU variants as `N(1)`; the single-GPU sweep writes it to `base_n.txt`
and stops after the first failed size. Every backend gets the same `N(G)` at each count:

```sh
BASE_N=65536 # largest common passing N on the one-H100 sweep
BENCH_ELTYPE=Float32 BENCH_OUTPUT=results/krylov-weak-f32 bash composability/krylov/run.sh weak "$BASE_N" 1 2 4 8
```

The selected H100 weak-scaling dimensions are `N=65536, 92682, 131072,
185364` at 1, 2, 4, and 8 GPUs, respectively. CUDA is included only in the
one-GPU size sweep; the weak run has Dagger, cuNumeric stock, and cuNumeric
local cases at every count.

The seven listed one-GPU dimensions all passed the residual and GPU-storage
checks on the single H100. The `N=1024`, `2048`, and `4096` measurements fill
in the low end while retaining `N=65536` as the maximum and weak baseline.

On a one-GPU machine, use `BENCH_DRY_RUN=1` with the weak command to write
`planned-cases.csv` for all four GPU counts without launching them. Run the
one-GPU point normally with `weak "$BASE_N" 1`.

The runner uses CG only. It writes `results.csv`, `timings.png`, per-case logs,
the package manifest, sampled GPU-memory log and peak summary, and `environment.txt` together. `BENCH_PROJECT` can point to an
already-instantiated equivalent environment (the default is this directory).
Set `JULIA`, `BENCH_THREADS`, or `BENCH_CPUS` for the machine.
Set `BENCH_TIMEOUT` to a per-case duration accepted by GNU `timeout`
(default `15m`). Failed or timed-out cases keep their logs; completed CSV
rows remain plottable without hiding other backends or sizes.
Legate auto-sizes its memory pool. The sampled `nvidia-smi` memory peak is
diagnostic; pool reservation can make it much larger than live array storage.
The runner sets `LEGATE_CONFIG` separately for every GPU count. Set
`CUBLAS_WORKSPACE_CONFIG` externally, if desired, so every backend sees the same
setting. `plot.py` needs Python with Matplotlib.
On an eight-GPU host, each case launches Julia with exactly its requested
number of visible GPUs: by default devices `0` through `G-1`, or the first
`G` entries of an existing `CUDA_VISIBLE_DEVICES` list. The selected mask is
recorded in `planned-cases.csv`, and Dagger verifies GPU-backed tile placement.

Each case gets a fresh Julia process, two warm-up solves, and five synchronized
timed solves. Allocation, host-to-device transfer, compilation, explicit GC,
and an independent Float64 residual check are outside the timer. The CSV records
iteration count, residual, mean time, standard error, and raw samples. The plot
shows mean time with standard-error bars. For weak scaling, each backend has a
horizontal ideal-time reference at its one-GPU mean. CG uses a symmetric
tridiagonal source; the timed operator is dense. The local loop uses the same mathematical
recurrences but may differ from Krylov in floating-point order and breakdown
handling. Their `@accelerate` helpers express fused vector updates; use a
profiler before attributing a timing difference to kernel fusion.

The single-GPU and one-GPU weak-scaling cases can run on a one-GPU machine.
Multi-GPU placement, convergence, and timings need validation on the target
multi-GPU host before publication.
