# Composability benchmarks

Three workloads pass `CuArray`, Dagger `DArray`, and cuNumeric `NDArray` through
the same library code: [Krylov CG/BiCGStab](krylov/README.md),
[OrdinaryDiffEq heat](ordinarydiffeq/README.md), and
[Integrals + Optimization plume fit](integrals_optimization/README.md).
CUDA.jl is the one-GPU reference; Dagger and cuNumeric also run weak scaling.
For the paper run, follow [FINAL_RUN.md](FINAL_RUN.md).

## Setup

Use Julia 1.13. From the repository root:

```bash
export CUNUMERIC_SOURCE=/path/to/cuNumeric.jl   # only for a standalone checkout
./instantiate_projects.sh
```

All workloads share `environments/composability`, which pins Dagger's
`aot-schedulers-rebased` branch (version 0.22.5). Put machine-specific
`LocalPreferences.toml` there.

## Run

```sh
julia --project=. run_composability.jl                     # all workloads, one GPU
julia --project=. run_composability.jl --only=krylov --solvers=cg,bicgstab --mode=both
julia --project=. run_composability.jl --only=krylov,ordinarydiffeq --mode=multi --gpus=1,2,4,8
julia --project=. run_composability.jl --mode=both --dry-run   # preview commands
julia --project=. run_composability.jl --help
```

- `--only`: `krylov`, `ordinarydiffeq`, `integrals_optimization`, a list, or `all`.
- `--mode`: `single` (size sweep on one GPU), `multi` (weak scaling), or `both`.
- `--solvers`: Krylov `cg` (default), `bicgstab`, or both. `--local` adds the
  cuNumeric local implementations.
- `--models`: any of `cuda,dagger,cunumeric` (default all), e.g.
  `--models=cuda,cunumeric` to skip Dagger. CUDA.jl runs only in single mode.
- `--config`: `sizes_80GB.toml` (H100, default), `sizes_141GB.toml` (H200), or your own.
- `--output`: result root (default `results/composability-<run-id>`), with
  `single/<workload>` and `multi/<workload>` subdirectories.

A failed case keeps its log; the remaining cases still run, and the exit code
is nonzero. Existing results are never overwritten.

## Dagger block tuning

The root `tune_dagger.sh` launches both main and composability tuning. Its
default run includes all main benchmarks plus composability CG and heat.
To tune only composability at the actual multi-GPU sizes on 1, 2, 4, and 8 GPUs:

```sh
bash tune_dagger.sh --dry-run krylov_cg ordinarydiffeq
bash tune_dagger.sh krylov_cg ordinarydiffeq
DAGGER_TUNE_CONFIG=composability/sizes_80GB.toml bash tune_dagger.sh krylov_cg ordinarydiffeq
```

The tuner reads `weak_base` from `sizes_141GB.toml` (H200) by default, matching
`run_composability.jl --mode=multi --config=composability/sizes_141GB.toml`.
For each GPU count G, it uses
`N = round(weak_base * sqrt(G))`, exactly as the benchmark launchers do.
Set `DAGGER_TUNE_CONFIG` to the same preset/custom TOML file used for your
benchmark run; relative paths are resolved from the calling directory.
The chosen config is saved as `sizes.toml` alongside each case's logs.
This tunes the multi-GPU sizes, including G=1, rather than every entry in the
separate `single` size sweep. Each solver has its own call at the bottom of
`tune_dagger.sh`: comment out `tune "$g" krylov_cg` or
`tune "$g" ordinarydiffeq` to disable it. BiCGSTAB (`krylov_bicgstab`) and plume
(`integrals_optimization`) start commented out; uncomment either call to enable
it. `cg` selects the main CG benchmark, while `krylov_cg` selects Krylov.jl CG.
Change the single `GPUS=(1 2 4 8)` list to limit GPU counts for every workload.

Each candidate runs in a fresh Julia process with `COMPOSABILITY_TUNE=1`:
one untimed warmup followed by two timed runs. Both warmup and timed runs use
at most **5 Krylov iterations**, **5 heat steps**, or **5 optimizer iterations**,
instead of the regular limits of 200, 20, and 80. This mode overrides inherited
sample/iteration settings so a benchmark environment cannot make tuning expensive.
Heat uses the usual default timestep of 0.05 over a shorter trajectory.
Krylov checks finite residual reduction and plume checks finite loss improvement;
these short runs do not require full convergence or parameter recovery. Heat
retains its analytical accuracy check. Normal benchmark validation is unchanged.
Input construction, warmup, and process startup are excluded from timing.

The sweep tries `blocks_per_gpu = 1, 2, 4, 8, 16, 32, 64`, stopping when a
successful candidate is more than 4x slower than the best mean, as in the root
tuner. Failed candidates keep their logs and the sweep continues with a nonzero
final status. The Julia helper `composability/tune.jl` appends mean timings to
`tunes/composability/<name>.csv` and winners to `<name>-best.csv`, using hyphens
in filenames (for example, `krylov-cg.csv`). Raw samples remain in per-run
directories under `tunes/composability/logs/`. Existing CSVs are appended to;
logs are never overwritten. Set `DAGGER_TUNE_OUTPUT` to override the composability
output directory (relative paths are relative to the repository root).
`--dry-run` uses Julia's standard
library to read the config and prints commands without creating files,
loading GPU packages, or running benchmarks.

Apply a winner through `DAGGER_BLOCKS_PER_GPU`, which defaults to 1 in regular
composability runs. Heat and plume use row slabs of height
`cld(N, GPUs * blocks_per_gpu)`, assigned round-robin to GPUs. Krylov uses that
same side length for square matrix tiles and matching vector blocks, so its
matrix tile count grows quadratically with this factor.

## Problem sizes

Each config section sets the one-GPU sweep and the weak-scaling base, with
`N(G) = round(weak_base * sqrt(G))`:

```toml
[krylov]
single = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
weak_base = 65536
```

N is the dense matrix dimension for CG and the grid side length for heat and
plume.

| Workload | 80 GB max | 80 GB weak_base | 141 GB max | 141 GB weak_base |
| --- | ---: | ---: | ---: | ---: |
| Krylov CG | 114,688 | 65,536 | 131,072 | 131,072 |
| OrdinaryDiffEq heat | 32,768 | 16,384 | 32,768 | 32,768 |
| Integrals + Optimization plume | 8,192 | 8,192 | 10,240 | 10,240 |

The CG matrix is built on the host first: the H200 weak base needs about
64 GiB at one GPU and 512 GiB at eight. Validate one-GPU sizes on every
backend before raising `weak_base`.

## Workload launchers

`run_composability.jl` calls these directly; use them for custom sizes. Shared
shell helpers are in [`common.sh`](common.sh), and Krylov and ODE run each
sample (one warmup plus one timed solve) in a fresh Julia process through
[`process_samples.jl`](process_samples.jl).

```sh
bash composability/krylov/run.sh single 1024 2048
bash composability/ordinarydiffeq/run_benchmark.sh weak 128 1 2
bash composability/integrals_optimization/run_benchmark.sh single 32 128
```

Set `BENCH_DRY_RUN=1`, `ODE_DRY_RUN=1`, or `INTOPT_DRY_RUN=1` to write
`planned-cases.csv` without running. Each result directory holds `results.csv`
(raw samples plus the correctness metric), per-case logs, the
GPU mask, and a copy of the manifest; plots recompute means and standard errors
from the samples.
