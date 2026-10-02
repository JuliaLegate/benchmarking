# Krylov solver composability

This standalone benchmark runs CG (default) or BiCGStab. For each solver,
all backends use the same dense operator and right-hand side. The first three
single-GPU paths run by default; `--local` adds the fourth:

| Plot label | Solver call |
| --- | --- |
| CUDA | Stock `Krylov.cg!` or `Krylov.bicgstab!` on `CuArray` |
| Dagger | Stock Krylov solver on GPU-backed Dagger arrays |
| cuNumeric | Stock Krylov solver on `NDArray` |
| cuNumeric local (opt-in) | Concise local solver loop on `NDArray`, with `@accelerate` vector-update helpers |

The weak-scaling run uses Dagger and cuNumeric stock; `--local` also includes
cuNumeric local. Dagger uses
square tiles and the requested GPU scope; cuNumeric uses Legate's `--gpus`
configuration. The Dagger checkout used for the original comparison calls CPU
BLAS from its tile GEMV fallback, so this benchmark adds a CuArray tile method
that forwards to GPU `mul!`. It does not change Krylov's solver code.

Use Julia 1.13 and a cuNumeric checkout containing the Krylov extension. From
the benchmarking repository root, initialize all benchmark environments with
the repository's standard setup script:

```sh
CUNUMERIC_SOURCE=/opt/cuNumeric.jl ./instantiate_projects.sh
```

Krylov uses the shared `environments/composability` project in both local runs
and Docker. Set `BENCH_PROJECT` to select an alternate instantiated project.

If cuNumeric uses local backend-library preferences, copy its
`LocalPreferences.toml` into this environment. Preserve the resulting
`Manifest.toml` and preference file with the run record; neither is committed.

Set one or more explicit matrix dimensions for the single-GPU comparison:

```sh
BENCH_ELTYPE=Float32 BENCH_OUTPUT=results/krylov-single-f32 bash composability/krylov/run.sh single 1024 2048 4096 8192 16384 32768 65536 81920 98304 114688
```

Select solvers with the unified runner's `--solvers` option or the shell
launcher's comma-separated `BENCH_SOLVERS` setting:

```sh
julia --project=. run_composability.jl --only=krylov --solvers=cg,bicgstab \
  --config=composability/sizes_141GB.toml --mode=both --gpus=1,2,4,8
BENCH_SOLVERS=bicgstab bash composability/krylov/run.sh single 1024 2048
BENCH_SOLVERS=cg,bicgstab bash composability/krylov/run.sh weak 1024 1 2 4 8

# Include cuNumeric local alongside all stock implementations:
julia --project=. run_composability.jl --only=krylov --solvers=cg,bicgstab --local
BENCH_SOLVERS=cg,bicgstab BENCH_LOCAL=1 bash composability/krylov/run.sh single 1024 2048
```

Both solvers use the selected config's `[krylov]` sizes and weak baseline.
CG measurements do not establish BiCGStab's memory or convergence limits.
The shell launcher defaults to `BENCH_LOCAL=0`. The unified runner controls
this setting through `--local`, regardless of an inherited `BENCH_LOCAL`.

For weak scaling, set the **one-GPU** dimension followed by GPU counts. The
runner chooses `N(G) = round(N(1) × sqrt(G))`, keeping dense matrix elements per
GPU approximately constant. Use the largest dimension that passed all enabled
single-GPU variants in `results.csv` as `N(1)` for each solver.
A backend failure retains the other backends' successful results and the sweep
continues to larger sizes. Failed cases retain logs, and the launcher returns
a nonzero exit status after collecting and plotting the remaining results.
The unified runner uses the config's `weak_base`.
Every backend gets the same `N(G)` at each count:

```sh
BASE_N=65536 # largest common passing N on the one-H100 sweep
BENCH_ELTYPE=Float32 BENCH_OUTPUT=results/krylov-weak-f32 bash composability/krylov/run.sh weak "$BASE_N" 1 2 4 8
```

The selected H100 weak-scaling dimensions are `N=65536, 92682, 131072,
185364` at 1, 2, 4, and 8 GPUs, respectively. CUDA is included only in the
one-GPU size sweep; the weak run has Dagger and cuNumeric stock cases at
every count, plus cuNumeric local when enabled.

The original seven one-GPU dimensions through `N=65536` all passed the
residual and GPU-storage checks on the single H100. The H100 preset now
extends through `N=114688`; the H200 preset adds `N=131072`. Individual
backends may fail at the larger sizes. The configured weak baseline remains
`N=65536` for H100 and uses the reported passing `N=131072` point for H200.

On a one-GPU machine, use `BENCH_DRY_RUN=1` with the weak command to write
`planned-cases.csv` for all four GPU counts without launching them. Run the
one-GPU point normally with `weak "$BASE_N" 1`.

The runner writes one `results.csv` with a `solver` column, per-case and per-sample logs,
the package manifest, sampled GPU-memory logs and peak summary, and
`environment.txt` together. `memory.csv` and `planned-cases.csv` identify
the solver, and log filenames include it. A single selected solver writes
`timings.png`; selecting both writes `timings-cg.png` and
`timings-bicgstab.png` for solvers with successful cases. `BENCH_PROJECT` can point to an
already-instantiated equivalent environment.
Set `JULIA`, `BENCH_THREADS`, or `BENCH_CPUS` for the machine.
Like ODE, the launcher has no time limit; `BENCH_TIMEOUT` is no longer used.
Every sample repeats startup, dense matrix construction, transfer, and warmup,
which are excluded from the timed solve.
Failed cases keep their logs; completed CSV
rows remain plottable without hiding other backends or sizes.
Legate auto-sizes its memory pool. The sampled `nvidia-smi` memory peak is
diagnostic; pool reservation can make it much larger than live array storage.
The runner sets `LEGATE_CONFIG` separately for every GPU count. Set
`CUBLAS_WORKSPACE_CONFIG` externally, if desired, so every backend sees the same
setting. `plot_results.jl` uses Plots.jl from `environments/composability`. Plotting runs after measurements:
a plotting failure returns a nonzero exit code but preserves the collected CSV.
Regenerate a weak-scaling plot without rerunning the benchmarks:

```bash
GKSwstype=100 julia --project=environments/composability \
  composability/krylov/plot_results.jl weak /path/to/results.csv --output /path/to/timings.png

# Select BiCGStab rows from the same CSV:
GKSwstype=100 julia --project=environments/composability \
  composability/krylov/plot_results.jl weak /path/to/results.csv \
  --solver=bicgstab --output /path/to/timings-bicgstab.png
```

Use `single` instead of `weak` for a single-GPU size sweep.
On an eight-GPU host, each case launches Julia with exactly its requested
number of visible GPUs: by default devices `0` through `G-1`, or the first
`G` entries of an existing `CUDA_VISIBLE_DEVICES` list. The selected mask is
recorded in `planned-cases.csv`, and Dagger verifies initial GPU-backed tile placement.

Solution correctness is determined by the independent relative residual
`norm(A*x - b) / norm(b)`, evaluated on the host in Float64 against the original
tridiagonal operator and right-hand side. The limit is `1e-5` for Float32 and
`1e-8` for Float64; non-finite residuals also fail. A solver convergence flag
alone cannot pass this check, and a correct solution is accepted regardless
of its final GPU placement or convergence flag. Each process checks its warmup
and its actual timed solution; validation does not run an additional solve.

Each case gets five fresh Julia sample processes by default (`BENCH_SAMPLES`
sets the count, at least two). Each process runs one warmup and one synchronized
timed solve, then exits before the next process starts. Both CG and BiCGStab,
single- and multi-GPU runs, and the optional local loops use this isolation.
[`run_samples.jl`](run_samples.jl) aggregates only after all samples pass;
each `*-sample-N.log` contains the worker PID and its individual result.
A case appears in `results.csv` only after all requested samples pass;
interrupted cases retain individual results in their sample logs.

For cuNumeric, setup constructs the dense operator directly in row-major host
storage and calls the same `cuNumeric.nda_attach_external` helper used by its
matrix constructor. This avoids a full dense `permutedims` and subsequent
`collect` copy in `NDArray(Matrix(reference))`; the attachment retains its host
buffer. Both CG and the nonsymmetric BiCGStab operator retain their original
entries. Other backends keep their usual column-major host inputs. No sparse
operator is used in the timed solve.

Allocation, host-to-device transfer, startup, the warmup, explicit GC, and an
independent Float64 residual check are outside the timer. Repeated startup and
warmups add wall-clock time. The CSV records the maximum iteration count and
residual across samples, mean time, standard error, and raw times. The plot
shows mean time with standard-error bars. For weak scaling, each backend has a
horizontal ideal-time reference at its one-GPU mean. CG uses a symmetric
tridiagonal source and BiCGStab uses a nonsymmetric tridiagonal source; the
timed operators are dense. The local loops use the same mathematical
recurrences but may differ from Krylov in floating-point order and breakdown
handling. Their `@accelerate` helpers express fused vector updates; use a
profiler before attributing a timing difference to kernel fusion.

The single-GPU and one-GPU weak-scaling cases can run on a one-GPU machine.
Multi-GPU placement, convergence, and timings need validation on the target
multi-GPU host before publication.
