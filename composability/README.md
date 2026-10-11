# Composability benchmarks

Library code run unchanged on `CuArray`, Dagger `DArray` and cuNumeric `NDArray`.
CUDA.jl is the one-GPU reference; Dagger and cuNumeric also run weak scaling.

| Workload | Library code | Guide |
|---|---|---|
| `krylov` | Krylov.jl CG / BiCGStab on a dense matrix | [krylov](krylov/README.md) |
| `ordinarydiffeq` | OrdinaryDiffEq.jl 2D heat equation | [ordinarydiffeq](ordinarydiffeq/README.md) |
| `integrals_optimization` | Integrals.jl + Optimization.jl plume fit | [integrals_optimization](integrals_optimization/README.md) |

## Setup

```bash
./deps-install/instantiate_projects.sh
```

All workloads share `environments/composability` (Julia 1.13). Put
machine-specific `LocalPreferences.toml` there.

## Run

```bash
julia --project=. run_composability.jl                            # all workloads, one GPU
julia --project=. run_composability.jl --only=krylov --mode=multi --gpus=1,2,4,8
julia --project=. run_composability.jl --only=krylov --solvers=cg,bicgstab --local
julia --project=. run_composability.jl --mode=both --dry-run      # print the commands
```

| Option | Values |
|---|---|
| `--only` | `krylov`, `ordinarydiffeq`, `integrals_optimization`, a list, or `all` |
| `--mode` | `single` (size sweep on one GPU), `multi` (weak scaling), `both` |
| `--models` | any of `cuda,dagger,cunumeric`; CUDA.jl runs only in single mode |
| `--solvers`, `--local` | Krylov solvers; `--local` adds cuNumeric's local CG loop |
| `--config` | `sizes_80GB.toml` (H100, default), `sizes_141GB.toml` (H200) |
| `--output` | result root; runs go to `single/<workload>` and `multi/<workload>` |

Failed cases keep their logs and the rest still run. Existing results are never
overwritten.

## Problem sizes

Each config section sets the one-GPU sweep and the weak-scaling base;
`N(G) = round(weak_base * sqrt(G))`.

```toml
[krylov]
single = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
weak_base = 65536
```

N is the matrix dimension for Krylov and the grid side for heat and plume.

## Launcher

`run_composability.jl` calls [`run.sh`](run.sh), which can also be run directly:

```bash
bash composability/run.sh krylov single 1024 2048
bash composability/run.sh ordinarydiffeq weak 128 1 2
```

Settings use the workload's prefix (`BENCH_`, `ODE_`, `INTOPT_`):

| Setting | Meaning |
|---|---|
| `<P>_OUTPUT` | result directory |
| `<P>_BACKENDS` | e.g. `"CuArray Dagger cuNumeric"` |
| `<P>_SAMPLES` | samples per case (default 5) |
| `<P>_DRY_RUN=1` | write `planned-cases.csv` only |

Each case uses the first G visible GPUs. Krylov and ODE run every sample in a
fresh Julia process ([`run_samples.jl`](run_samples.jl)). Each result directory
holds `results.csv`, `timings.png`, per-case logs, `metadata.txt` and the
package manifest.

## Dagger tuning

```bash
bash scripts/tune_dagger.sh krylov_cg ordinarydiffeq
```

Tunes `blocks_per_gpu` at the weak-scaling sizes from `sizes_141GB.toml`
(override with `DAGGER_TUNE_CONFIG`), using short runs (5 iterations or steps).
Results go to `results/tunes/composability/`. Apply a winner with
`DAGGER_BLOCKS_PER_GPU` (default 1).

## Plots

The paper figure reads the CSVs listed in
[`configs/plots/composability.toml`](../configs/plots/composability.toml):

```bash
julia --project=. plotter/plot_composability.jl combined
julia --project=. plotter/plot_composability.jl --help    # per-workload figures
```
