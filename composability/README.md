# Composability benchmarks

For the eight-H100 smoke checks, final seven-point one-GPU sweeps, and full
weak-scaling run, follow the [final runbook](FINAL_RUN.md).

The three workloads are [Krylov CG](krylov/README.md),
[OrdinaryDiffEq heat](ordinarydiffeq/README.md), and
[Integrals.jl + Optimization.jl plume calibration](integrals_optimization/README.md).
They pass `CuArray`, Dagger `DArray`, and cuNumeric `NDArray` to the same
workload operations. Each workload has its own one-GPU launcher, correctness
checks, raw synchronized timing samples, and mean-time plot with standard
error bars. CUDA.jl is the one-GPU reference; Dagger and cuNumeric also have
weak-scaling launchers. Krylov runs stock CG on all three backends and local
CG on cuNumeric.

## Setup

Use Julia 1.13 on Linux with the installed cuNumeric/Legate libraries. Krylov
plotting also needs Python with Matplotlib. From the benchmarking repository
root, initialize the shared environment:

```bash
# For a standalone benchmarking checkout, point to your cuNumeric checkout.
export CUNUMERIC_SOURCE=/path/to/cuNumeric.jl
./instantiate_projects.sh
```

When benchmarking is cuNumeric.jl's `benchmark/` submodule, setup can infer the
parent checkout. The benchmark container performs setup during its build.
`environments/composability` includes all three workloads, Julia plotting
dependencies, and the optional ODE integrator smoke checks.

Local runs and Docker both use `environments/composability` directly.
The composability project pins Dagger to the registered `0.22.5` release.
JACC is not used by these workloads. Individual launchers still accept
`BENCH_PROJECT`, `ODE_PROJECT`, and `INTOPT_PROJECT` for alternate environments.
Rerun setup when upgrading from the old separate environments and apply
machine-specific `LocalPreferences.toml` to the shared environment.
`COMPOSABILITY_ENV_ROOT` is no longer used.
The former per-workload `setup.jl` files and Krylov project have been removed;
`instantiate_projects.sh` is the only setup entry point. The
[setup integration test and PR CI](../README.md#setup-tests-and-ci) exercise the
shared environment on Julia 1.13.1.

## Run

From the repository root, use `run_composability.jl`. It defaults to all three
workloads in single-GPU mode:

```sh
# All benchmarks on one GPU
julia --project=. run_composability.jl

# One benchmark on one GPU
julia --project=. run_composability.jl --only=krylov

# Selected benchmarks with multi-GPU weak scaling
julia --project=. run_composability.jl --only=krylov,ordinarydiffeq --mode=multi --gpus=1,2,4,8

# All benchmarks in both modes, with a chosen result directory
julia --project=. run_composability.jl --only=all --mode=both --output=results/composability-paper

# Preview without launching GPU workers, or show all options
julia --project=. run_composability.jl --mode=both --dry-run
julia --project=. run_composability.jl --help
```

`--only` accepts `krylov`, `ordinarydiffeq`, `integrals_optimization`, a
comma-separated selection, or `all`. `--mode=single` runs size sweeps on one
GPU. `--mode=multi` runs weak scaling at the specified GPU counts (default
`1,2,4,8`, including the one-GPU baseline). `--mode=both` runs both sequentially.
Supported counts are `1`, `2`, `4`, and `8`.

Single-GPU runs use the CG and heat sizes in `sizes.toml` and plume sizes
`32, 128, 512, 1024, 2048, 4096, 6144, 8192`. These are Float32 presets for an
80 GB H100; edit `sizes.toml` or pass `--config` for custom sizes. Each mode
saves its size config with the results. Multi-GPU runs also generate
`weak_scaling_plan.csv` from that config.

The default result root is `results/composability-<run-id>`; `--output` selects
another root, relative to the calling directory. Results are grouped under
`single/<workload>` and `multi/<workload>`. Existing result CSVs are protected
before either mode starts. A failed run returns a nonzero exit code and retains
its logs; the remaining workloads and modes are still attempted. Launcher progress
and errors stream to the terminal. If startup fails before any cases run, the
runner also prints the end of `environment.txt` (Krylov) or `metadata.txt` (other
workloads). Header-only CSVs indicate that no measurements were collected.

`--dry-run` prints the selected workloads, GPU counts, output paths, and launcher
commands without starting any process or writing files. It does not need Bash,
GPU access, or instantiated packages. Normal runs need the setup described above.
The workers use the current Julia executable unless `JULIA` or
`CUNUMERIC_BENCH_JULIA` selects another one. Workload settings such as
`BENCH_SAMPLES`, `ODE_SAMPLES`, `INTOPT_SAMPLES`, and project overrides remain
available. CLI selections control the workloads, output root, and dry-run mode.

## Set N (problem sizes)

Edit [`sizes.toml`](sizes.toml) to change N for any workload. This is the single
source of sizes for the unified runner; no Julia code or CSV needs editing.
Each workload has two settings:

```toml
[krylov]
single = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
weak_base = 65536
```

- `single` lists dimensions N for the one-GPU sweep, in increasing order.
- `weak_base` sets the one-GPU dimension for weak scaling. The runner uses
  `N(G) = round(weak_base * sqrt(G))` for G GPUs.

For CG, N is the dense matrix dimension. For heat and plume fitting, it is the
side length of the two-dimensional grid/image. The default `weak_base` values
are 65,536 for CG, 16,384 for heat, and 8,192 for plume fitting. The defaults
retain the original H100 sweeps; choose smaller values for smaller GPUs or
quick checks. Single-GPU values must be unique and increasing; N must be at
least 2 for CG and 4 for heat/plume.

To keep the defaults intact, copy the config and select your copy:

```sh
cp composability/sizes.toml my-sizes.toml
# Edit single and weak_base in my-sizes.toml, then preview the commands.
julia --project=. run_composability.jl --config=my-sizes.toml --mode=both --dry-run
julia --project=. run_composability.jl --config=my-sizes.toml --mode=both
```

A custom config only needs sections for the workloads selected by `--only`.
The results include a `sizes.toml` snapshot for each mode. The generated
`multi/weak_scaling_plan.csv` records the corresponding dimensions at 1, 2, 4,
and 8 GPUs; it is an output record, not a configuration file.

## Workload launchers

`run_composability.jl` selects sizes and directly calls the three workload
launchers. They handle process isolation, timeouts, GPU memory sampling, result
collection, and plotting. They remain useful for custom sizes and small smoke
checks; there are no separate shell entry points for the full suite.

```sh
bash composability/krylov/run.sh single 1024 2048
bash composability/ordinarydiffeq/run_benchmark.sh weak 128 1 2
bash composability/integrals_optimization/run_benchmark.sh single 32 128
```

For metadata validation without running a solver, use `BENCH_DRY_RUN=1`,
`ODE_DRY_RUN=1`, or `INTOPT_DRY_RUN=1` with the corresponding workload launcher.
This requires the initialized environment and GPU metadata tools. Each launcher
writes `planned-cases.csv`, environment metadata, and a copy of its Julia
manifest. Use the unified CLI's `--dry-run` for a preview without those dependencies.

The one-GPU points
have been measured separately. The two-, four-, and eight-GPU points require
correctness and GPU-placement checks on that machine before interpreting
timings. A failed case keeps its log and does not enter the successful CSV or
plot. Each result row includes raw samples and the numerical check; plots
recompute mean and sample-standard-error from those samples.
