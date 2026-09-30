# Composability benchmarks

For the eight-H100 smoke checks, one-GPU sweeps, and full
weak-scaling run, follow the [final runbook](FINAL_RUN.md).

The three workloads are [Krylov CG](krylov/README.md),
[OrdinaryDiffEq heat](ordinarydiffeq/README.md), and
[Integrals.jl + Optimization.jl plume calibration](integrals_optimization/README.md).
They pass `CuArray`, Dagger `DArray`, and cuNumeric `NDArray` to the same
workload operations. Each workload has its own one-GPU launcher, correctness
checks, raw synchronized timing samples, and mean-time plot with standard
error bars. CUDA.jl is the one-GPU reference; Dagger and cuNumeric also have
weak-scaling launchers. Krylov runs stock CG or BiCGStab on all three backends;
`--local` also enables local implementations on cuNumeric. CG is the default.

## Setup

Use Julia 1.13 on Linux with the installed cuNumeric/Legate libraries. All three
workloads use Plots.jl from the shared environment. From the benchmarking
repository root, initialize it:

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
The composability project selects Dagger's `aot-schedulers-rebased` branch through
`[sources]`, with compatibility restricted to version `0.22.5`.
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

# Both Krylov solvers, with single-GPU sweeps and multi-GPU weak scaling
julia --project=. run_composability.jl --only=krylov --solvers=cg,bicgstab --mode=both

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

`--solvers=cg` (default), `--solvers=bicgstab`, or `--solvers=cg,bicgstab`
selects the Krylov solvers in any mode. Include `krylov` in `--only` when
passing this option; other workloads are unaffected. Both solvers use the
same `[krylov]` sizes and `weak_base` from the selected config. Existing
large-size measurements are for CG; validate BiCGStab independently.

Stock implementations run by default. Add `--local` to include cuNumeric
local implementations alongside cuNumeric stock for every selected solver
and GPU mode. Single-GPU sweeps include CUDA and Dagger stock; multi-GPU runs
include Dagger stock. `--local` requires `krylov` in `--only`.

Each Krylov result directory contains one `results.csv` with a `solver`
column. A single selected solver writes `timings.png`; selecting both writes
`timings-cg.png` and `timings-bicgstab.png`. Case and memory logs identify
the solver. A failed backend or solver does not prevent the remaining cases
or the other solver's plot.

All three workloads use the Float32 preset in `sizes_80GB.toml` by default.
Select `--config=composability/sizes_141GB.toml` for a 141 GB H200, or pass a
custom config path. Each mode saves its size config with the results.
Multi-GPU runs also generate `weak_scaling_plan.csv` from that config.

The default result root is `results/composability-<run-id>`; `--output` selects
another root, relative to the calling directory. Results are grouped under
`single/<workload>` and `multi/<workload>`. Existing result CSVs are protected
before either mode starts. A failed case retains its logs and the successful
results from other backends; the remaining backends, sizes, workloads, and modes
are still attempted. The run returns a nonzero exit code if any case fails. Launcher progress
and errors stream to the terminal. If startup fails before any cases run, the
runner also prints the end of `environment.txt` (Krylov) or `metadata.txt` (other
workloads). Header-only CSVs indicate that no measurements were collected.

`--dry-run` prints the selected workloads, GPU counts, output paths, and launcher
commands without starting any process or writing files. It does not need Bash,
GPU access, or instantiated packages. Normal runs need the setup described above.
The workers use the current Julia executable unless `JULIA` or
`CUNUMERIC_BENCH_JULIA` selects another one. The runner passes its Julia package
depots to workers explicitly, so `--startup-file=no` does not hide packages
installed in a depot added by the container's startup file. Workload settings such as
`BENCH_SAMPLES`, `ODE_SAMPLES`, `INTOPT_SAMPLES`, and project overrides remain
available. CLI selections control the workloads, Krylov solvers, output root,
and dry-run mode. The unified runner sets `BENCH_SOLVERS` from `--solvers`
and `BENCH_LOCAL` from `--local`. For the Krylov shell launcher, set
`BENCH_SOLVERS` directly and use `BENCH_LOCAL=1` to add local implementations.

## Set N (problem sizes)

Choose [`sizes_80GB.toml`](sizes_80GB.toml) for an 80 GB H100 (default) or
[`sizes_141GB.toml`](sizes_141GB.toml) for a 141 GB H200. The selected config
sets N for all workloads; no Julia code or CSV needs editing. Each workload
has two settings:

```toml
[krylov]
single = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
weak_base = 65536
```

- `single` lists dimensions N for the one-GPU sweep, in increasing order.
- `weak_base` sets the one-GPU dimension for weak scaling. The runner uses
  `N(G) = round(weak_base * sqrt(G))` for G GPUs.

For CG, N is the dense matrix dimension. For heat and plume fitting, it is the
side length of the two-dimensional grid/image. The maximum single-GPU N and
`weak_base` values are:

| Workload | 80 GB maximum | 80 GB weak_base | 141 GB maximum | 141 GB weak_base |
| --- | ---: | ---: | ---: | ---: |
| Krylov CG | 114,688 | 65,536 | 131,072 | 131,072 |
| OrdinaryDiffEq heat | 32,768 | 16,384 | 32,768 | 32,768 |
| Integrals + Optimization plume | 8,192 | 8,192 | 10,240 | 10,240 |

The H100 Krylov preset includes 81,920, 98,304, and 114,688 to cover the prior
H100 comparison. The 141 GB preset omits 98,304 to match the measured H200
sweep and adds 131,072. Krylov keeps successful
backend results and continues to larger sizes when another backend fails;
failures retain logs and still produce a nonzero exit status. The configured
Krylov weak-scaling baseline remains 65,536 for H100 and uses the reported
passing 131,072 point for H200.

Both heat sweeps stop at 32,768; the H200 65,536 point proved too large and
was removed. All heat sweep dimensions are powers of two. Heat `weak_base`
remains 16,384 for H100 and is 32,768 for H200. Even if every
backend fails at a size, the launcher keeps earlier results and continues.
If no case succeeds, it retains the CSV header and failure logs and skips
plotting. Any failed case still produces a nonzero final exit status.

The H200 CG weak baseline stores a 64 GiB Float32 matrix at one GPU and
approximately 512 GiB at eight GPUs. The benchmark first constructs this
matrix on the host, so host memory must also accommodate it and temporary
copies before GPU distribution.

The H200 plume preset appends an estimated endpoint at 1.25 times the H100
maximum: approximately 1.5625 times the array storage for this quadratic
workload, below the 141/80 = 1.7625 capacity ratio.
Validate the selected single-GPU sizes on every backend before using them
for weak scaling; raising a sweep maximum does not automatically raise
`weak_base`:

```sh
julia --project=. run_composability.jl --config=composability/sizes_141GB.toml --dry-run
julia --project=. run_composability.jl --config=composability/sizes_141GB.toml --mode=single
# After validating the one-GPU sizes:
julia --project=. run_composability.jl --config=composability/sizes_141GB.toml --mode=multi --gpus=1,2,4,8
```

Choose smaller values for smaller GPUs or quick checks. Single-GPU values must
be unique and increasing; N must be at least 2 for CG and 4 for heat/plume.

To keep the defaults intact, copy the config and select your copy:

```sh
cp composability/sizes_80GB.toml my-sizes.toml
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
