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
