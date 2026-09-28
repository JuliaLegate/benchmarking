# JuliaLegate benchmarks

## Composability

The [composability setup and running guide](composability/README.md) covers Krylov CG,
the OrdinaryDiffEq heat equation, and the Integrals.jl + Optimization.jl gas
plume fit. These workloads run independently of the benchmark orchestrator.

This directory runs the same GPU benchmarks across cuNumeric.jl, cuPyNumeric,
CUDA.jl, JACC.jl, and Dagger.jl. Each model runs in an isolated process and
environment. Runs include correctness checks, trial progress, mean time, and
mean throughput with trial standard deviations.

## Setup

Instantiate the Julia environments once, including the shared environment for
Krylov, OrdinaryDiffEq, and Integrals + Optimization:

```bash
./instantiate_projects.sh
```

When used as cuNumeric.jl's `benchmark/` submodule, this develops the parent
checkout and its CNPreferences package. For a standalone clone:

```bash
CUNUMERIC_SOURCE=/path/to/cuNumeric.jl ./instantiate_projects.sh
```

Set `CUNUMERIC_BENCH_JULIA` to select a different Julia executable.
Use Julia 1.13 for the composability workloads. All three share the versioned
[`environments/composability/Project.toml`](environments/composability/Project.toml).
Setup and launchers use this directory directly, including inside the container.
Dagger is pinned to the registered `0.22.5` release in both the composability
and Dagger environments. Setup also releases old Dagger branch pins in existing
Dagger manifests; rerun `./instantiate_projects.sh` after updating.
These workloads do not use JACC.
`BENCH_PROJECT`, `ODE_PROJECT`, and `INTOPT_PROJECT` remain available as
per-workload launcher overrides. Launchers also accept `JULIA`, which takes
precedence over `CUNUMERIC_BENCH_JULIA`.

Existing checkouts/images using the old per-workload environments must rerun
`./instantiate_projects.sh`. Apply any machine-specific `LocalPreferences.toml`
to the shared environment; old manifests and preferences are not migrated.

In cuNumeric.jl, initialize the pinned harness with
`git submodule update --init --recursive`. Publish benchmark changes here first,
then commit the updated `benchmark` pointer in cuNumeric.jl. The benchmark
container remains a derived layer on the existing cuNumeric base image;
[container builds and opt-in CI](docker/README.md) live in this repository.

cuPyNumeric also needs its conda environment:

```bash
./install_cupynumeric.sh
```

Set `CUNUMERIC_BENCH_CONDA` if `conda` is not on `PATH`, or
`CUPYNUMERIC_ENV` to use an existing environment.

## Run composability benchmarks

Run composability benchmarks through their unified entry point. It defaults to
all three workloads with fixed Float32 sizes for an 80 GB H100:

```bash
julia --project=. run_composability.jl
julia --project=. run_composability.jl --only=krylov,ordinarydiffeq
julia --project=. run_composability.jl --mode=multi --gpus=1,2,4,8
julia --project=. run_composability.jl --mode=both --output=results/composability-paper
julia --project=. run_composability.jl --only=integrals_optimization --dry-run
```

Set N in [`composability/sizes.toml`](composability/sizes.toml): `single` lists
the one-GPU dimensions and `weak_base` sets N for the one-GPU weak-scaling
baseline. Use `--config=/path/to/sizes.toml` to select a separate config.

`--mode=single` runs size sweeps on one GPU; `--mode=multi` runs weak scaling.
`--only=all` is the default. `--dry-run` previews commands without requiring a
GPU or initialized environment. Use `--help` for all options and the
[composability guide](composability/README.md) for sizes and smoke checks.

## Run the benchmark suite

Use the smoke test for a quick end-to-end check:

```bash
julia --project=. run.jl --config=configs/single_gpu/smoke.toml
```

Run the configured benchmark suite with:

```bash
julia --project=. run.jl
```

Useful filters:

```bash
julia --project=. run.jl --only=montecarlo
julia --project=. run.jl --only=gemm --models=cunumeric,cudajl,jacc,dagger
julia --project=. run.jl --only=grayscott --fusion=both
julia --project=. run.jl --only=montecarlo --dry-run
```

`--only` and `--models` accept comma-separated values. `--fusion` accepts
`on`, `off`, or `both`. Use `--verbose` for backend details.

Run the focused cuNumeric, cuPyNumeric, and Dagger Gray-Scott weak-scaling
check on 1, 2, 4, and 8 GPUs with:

```bash
julia --project=. run.jl --config=configs/multi_gpu/grayscott.toml --verbose
```

## Setup tests and CI

[Environment setup tests](.github/workflows/setup-tests.yml) run on every pull
request update and pushes to `main`, using **Julia 1.13.1** on a hosted Ubuntu
runner. The job checks out `JuliaLegate/cuNumeric.jl` at `main`, runs the setup
path and composability CLI tests, then runs the real `instantiate_projects.sh`
in fresh copies of all project directories. It verifies the generated
manifests, local cuNumeric/CNPreferences paths, and released Dagger version.

The job disables automatic precompilation and GPU runtime startup. It tests
installation rather than GPU execution; it does not need a GPU or build the
benchmark container. Downloaded packages may be cached, but project manifests
are always created afresh. Logs, tested commit IDs, and resolved project files
are uploaded as `setup-julia-1.13.1`, including on failure.

Run the same integration test locally on Linux with Julia 1.13.1:

```bash
CUNUMERIC_SOURCE=/path/to/cuNumeric.jl \
  julia --startup-file=no --project=. test/instantiate_projects.jl
```

It leaves your benchmark environments unchanged and saves diagnostics under
`results/instantiate-tests/`. The cuNumeric checkout should be clean, without
stale local manifests or machine-specific preferences.

## Configure

Benchmarks are declared in TOML configs under `configs/`: `single_gpu/` holds
one-GPU runs (smoke test, CG, NAS) and `multi_gpu/` holds weak-scaling sweeps.
`run.jl` defaults to `configs/multi_gpu/all.toml`; pass another with `--config=`.
Global values are inherited by each benchmark block:

```toml
[Global]
models = ["cunumeric", "cupynumeric", "cudajl", "jacc", "dagger"]
n_warmup = 2
n_iter = 10
n_trial = 5
check_correctness = true
auto_size = true
mem_frac = 0.75

[[montecarlo]]
T = "Float32"
gpus = 1
cpus = 1
```

Set `N` and `M` explicitly for fixed problem sizes. When `auto_size = true`, an
omitted dimension is selected from `mem_frac` of the smallest visible GPU.
NAS benchmarks take their fixed `N` and `M` from `kwargs.class`, so omit them.
A `class` list zips with `gpus` (see `configs/multi_gpu/nas_*_{strong,weak}.toml`).
Each NAS iteration is one complete, separately fenced class run (e.g. all 20 MG
V-cycles), so `n_iter` averages whole runs.
`T` and `fusion` form independent sweeps; `gpus`, `cpus`, `N`, and `M` are
zipped by position. A benchmark block may override `models`, `n_warmup`,
`n_iter`, or `n_trial`.

For JACC and Dagger, `run_benchmark.sh` restricts each worker to the requested
GPU count. It selects the first `gpus` entries from an existing
`CUDA_VISIBLE_DEVICES` scheduler mask, or uses logical devices starting at zero
when no mask is provided.

Native-library benchmarks such as GEMM require verified per-model scratch-space
bounds under `[workspace.<benchmark>]`; the planner reports any missing bound.

The model registry only schedules benchmark/model pairs with a runnable native
implementation. CUDA.jl is single-GPU; the other models may support multiple
GPUs depending on the workload.

### Benchmark support

Each cell shows **1 GPU / multiple GPUs**. ✅ is supported, ⚠ is supported with
a known limitation, and — is not supported by the harness.

| Benchmark | cuNumeric.jl | cuPyNumeric | CUDA.jl | JACC | Dagger.jl |
|---|---:|---:|---:|---:|---:|
| Monte Carlo | ✅ / ✅ | ✅ / ✅ | ✅ / — | ✅ / ✅ | ✅ / ✅ |
| GEMM | ✅ / ✅ | ✅ / ✅ | ✅ / — | ✅ / ✅ | ✅ / ✅ |
| 2D Gray–Scott | ✅ / ✅ | ✅ / ✅ | ✅ / — | ✅ / — | ✅ / ⚠ |
| Conjugate gradient | ✅ / ✅ | ✅ / ✅ | ✅ / — | ✅ / ✅ | ✅ / ⚠ |
| NAS embarrassingly parallel | ✅ / ✅ | ✅ / ✅ | ✅ / — | ✅ / ✅ | ✅ / ✅ |
| NAS Fourier transform | ✅ / ✅ | ✅ / ✅ | ✅ / — | ✅ / ⚠ | ⚠ / ⚠ |
| NAS multigrid | ✅ / ✅ | ✅ / ✅ | ✅ / — | ✅ / ⚠ | ✅ / ⚠ |

The Dagger Gray–Scott implementation is correct on multiple GPUs, but Dagger's
current fused stencil path transfers whole neighboring chunks before slicing
their halos, causing poor scaling. Dagger CG uses the high-level distributed
array API and has passed distributed CPU correctness testing; its multi-GPU CUDA
path still needs validation in the benchmark container. JACC Gray–Scott remains
single-GPU pending working two-dimensional ghost exchange. The table covers the
default benchmark forms; cuNumeric-specific accelerated forms and `cg_plain`
are comparison variants rather than separate workloads.

`montecarlo` uses cuNumeric's fused mapped reduction. The cuNumeric-only
`montecarlo_naive` variant materializes the broadcasted integrand before its
ordinary reduction for an explicit implementation comparison. Run both with
`julia --project=. run.jl --config=configs/multi_gpu/montecarlo.toml`.

NAS EP reproduces the official 46-bit RNG sequence and verification sums; every
model except CUDA.jl partitions the independent streams across GPUs.

NAS FT runs a complete official class per timed sample. cuNumeric, cuPyNumeric,
and JACC split the 3-D FFT into slab passes (2-D over x/y, then 1-D over z)
that each partition across GPUs; Dagger uses its distributed FFT. JACC uses
cuFFT per GPU, since JACC has no FFT. Dagger's cross-GPU checksum aggregation
is untimed.

NAS MG runs the official periodic multigrid V-cycle and verifies its final L2
norm. Its exact sparse RNG-generated right-hand side is setup outside timing,
matching NPB-GPU. Dagger uses distributed periodic stencils, but currently
constructs a tuple-valued distributed array for interpolation and temporary
arrays for restriction; its GPU scaling still needs measurement. An optional
CUDA.jl separable-array point uses cuNumeric's three-axis transfer algorithm.

JACC FT and MG use `JACC.Multi` at every GPU count; their multi-GPU paths pass
on simulated devices but still need validation on multiple GPUs. See
`nas/README.md` for details.

## Results

Each run writes CSV files and a manifest to `results/<run-id>/`, then writes
plots to `plots/<run-id>/`. The manifest records resolved dimensions, memory
estimates, package versions, and worker status.

Timed iterations include synchronization but exclude harness initialization and
warmup. NAS FT's specified RNG/index-map setup occurs inside `run!` and is
therefore timed as part of each complete FT sample.
Each trial reports its mean milliseconds per iteration and throughput; the
final summary reports the mean and standard deviation across trials. Most
benchmarks use GFLOP/s, while NAS EP follows NPB and reports G random numbers/s.

To plot existing CSV files:

```bash
julia --project=. plot_results.jl results/<run-id>
```

### Conjugate gradient

`cg` is the default variant and runs on every model; on cuNumeric it applies
`@accelerate` to each update. `cg_plain` is the cuNumeric variant without
`@accelerate`, for the accelerate comparison. The generic solver is shared by the
array workers (`src/benchmarks/cg.jl`); JACC and Dagger have native versions.

```bash
julia --project=. run.jl --config=configs/single_gpu/cg.toml
```

For the 9-million-elements-per-GPU weak-scaling run:

```bash
julia --project=. run.jl --config=configs/multi_gpu/cg.toml
```

Dagger CG uses distributed arrays and its native stencil, broadcast, and
reduction operations across all selected GPUs.

Set solver controls per entry, e.g. `kwargs = { check_every = 10, max_iter = 1000 }`.
The problem is `tridiag(1,4,1) x = 1/2` from `x = 0`. Each solve checks convergence
(and syncs the residual to the host) only when `k % check_every == 0` or
`k == max_iter`, and fails if it does not converge (relative tolerance 1e-8 for
Float64, 1e-5 for Float32); use `max_iter = 1` for a single-update comparison.

`n_iter` counts complete solves per trial. GFLOP/s uses 15N FLOPs per executed
iteration, counted by a one-time CPU replay (~30 s at N=10^8). Auto-sizing depends on
`max_iter`, so keep N fixed when comparing check intervals. The config pins
N=65,536 (set N=100,000,000 for the paper-sized workload); JACC additionally
requires N divisible by the GPU count. Validate the JACC partition kernels with
the selected GPUs visible:

```bash
julia --project=environments/jacc test/jacc_cg.jl
```
