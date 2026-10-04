# JuliaLegate benchmarks

GPU benchmarks across cuNumeric.jl, cuPyNumeric, CUDA.jl, JACC.jl and Dagger.jl.
Each model runs in its own process and environment, with correctness checks
and per-trial mean time and throughput.

## Layout

| Path | Contents |
|---|---|
| `run.jl`, `src/` | Main harness: planning, workers, timing, results |
| `run_composability.jl`, `composability/` | Krylov CG, OrdinaryDiffEq heat, Integrals + Optimization ([guide](composability/README.md)) |
| `other/implicitglobalgrid/` | ImplicitGlobalGrid Gray–Scott ([guide](other/implicitglobalgrid/README.md)) |
| `nas/` | NAS EP/FT/MG notes ([guide](nas/README.md)) |
| `configs/` | `single_gpu/`, `multi_gpu/` runs; `plots/` figure configs |
| `scripts/` | `run_benchmark.sh` (per-worker launcher), `tune_dagger.sh` |
| `deps-install/` | Environment, cuPyNumeric and IGG setup |
| `plotter/`, `plot_all.sh` | Result and paper plots |
| `loc-analysis/` | Code complexity counts ([guide](loc-analysis/README.md)) |
| `docker/` | Benchmark container and its CI ([guide](docker/README.md)) |

## Setup

```bash
./deps-install/instantiate_projects.sh     # Julia envs + IGG's Conda MPI env
./deps-install/install_cupynumeric.sh      # cuPyNumeric Conda env
```

As cuNumeric.jl's `benchmark/` submodule, setup develops the parent checkout;
standalone, set `CUNUMERIC_SOURCE=/path/to/cuNumeric.jl`. Use Julia 1.13.
`CUNUMERIC_BENCH_JULIA` and `CUNUMERIC_BENCH_CONDA` pick the Julia and Conda
executables.

## Run

Everything in the paper:

```bash
./run_all.sh
```

Individual runs:

```bash
julia --project=. run.jl --config=configs/single_gpu/smoke.toml   # quick check
julia --project=. run.jl --config=configs/multi_gpu/grayscott.toml --fusion=both
julia --project=. run.jl --only=gemm --models=cunumeric,dagger --dry-run
julia --project=. run_composability.jl --only=krylov --mode=multi --config=composability/sizes_141GB.toml
bash other/implicitglobalgrid/run_weak_scaling.sh
```

`--only` and `--models` take comma-separated lists, `--fusion` is `on`, `off`
or `both`, and `--verbose` prints backend details. `run.jl` defaults to
`configs/multi_gpu/all.toml`.

## Configure

Configs are TOML. `[Global]` values apply to every benchmark block:

```toml
[Global]
models = ["cunumeric", "cupynumeric", "cudajl", "jacc", "dagger"]
n_warmup = 2
n_iter = 10
n_trial = 5

[[montecarlo]]
T = "Float32"
gpus = [1, 2, 4, 8]
N = [7537741000, 15075482000, 30150964000, 60301928000]
```

`gpus`, `cpus`, `N` and `M` zip by position; `T` and `fusion` sweep
independently. With `auto_size = true`, an omitted size fills `mem_frac` of the
smallest GPU. NAS blocks take their sizes from `kwargs.class`. GEMM needs
per-model scratch bounds under `[workspace.<benchmark>]`.

## Tune Dagger

```bash
bash scripts/tune_dagger.sh                  # every enabled benchmark
bash scripts/tune_dagger.sh grayscott cg     # selected ones
```

Tunes are written to `results/tunes/`. Dagger runs use the fastest
`blocks_per_gpu` tuned for each benchmark and GPU count, or 1 if none exists.

## Results and plots

Each run writes CSVs and a manifest to `results/<run-id>/` and plots to
`plots/<run-id>/`. Timings include synchronization but not setup or warmup.

```bash
julia --project=. plotter/plot_results.jl results/<run-id> --format=pdf
./plot_all.sh    # paper figures from configs/plots/ -> plots/
```

## Support

Every model runs every benchmark on 1–8 GPUs, except CUDA.jl, which is
single-GPU only.

## Tests

[Setup CI](.github/workflows/setup-tests.yml) instantiates every environment
on Julia 1.13.1 without a GPU. Run it locally with:

```bash
CUNUMERIC_SOURCE=/path/to/cuNumeric.jl julia --startup-file=no --project=. test/instantiate_projects.jl
```
