# Final H100 benchmark run

Run CG and the heat equation on an otherwise idle eight-H100 (80 GB each)
machine. Use Julia 1.13 and clean, pinned checkouts of benchmarking `main`
and cuNumeric's `codex/ode-scalar-broadcast` branch. Keep the checkouts at
`/opt/benchmarking-composability` and `/opt/cuNumeric.jl`, or set
`CUNUMERIC_SOURCE` to the actual cuNumeric checkout. Record the chosen commit
SHAs before running; every launcher also saves them with its results.

The benchmark container initializes the shared composability environment
during its build through `instantiate_projects.sh`. On a bare host, run
`./instantiate_projects.sh` once from
the benchmarking checkout. Preserve its `Manifest.toml` and
`LocalPreferences.toml` files with the results, and verify that preferences
point to the installed cuNumeric/Legate libraries. Check
`nvidia-smi -L` shows eight H100s and that no other job is using them.
Check host and container memory limits as well: the `G=8` dense CG matrix
contains about 128 GiB of Float32 values and is first constructed on the
host, before GPU distribution. Allow substantial room for transfer and
temporary copies; 512 GiB of available host memory is a sensible target.

The launchers select exactly `G` devices for each case. If
`CUDA_VISIBLE_DEVICES` is already a comma-separated scheduler allocation,
they use its first `G` entries; otherwise they use `0` through `G-1`.
`planned-cases.csv` records the mask. All runs below are sequential; use a
fresh output root for each final attempt.

```bash
cd /opt/benchmarking-composability
export CUNUMERIC_SOURCE=/opt/cuNumeric.jl
export ODE_PROJECT="$PWD/environments/composability" # also used below for plotting
run_id=$(date -u +%Y%m%dT%H%M%SZ)
results=/opt/bench-results/final-$run_id
mkdir -p "$results"
git rev-parse HEAD > "$results/benchmark-commit.txt"
git -C "$CUNUMERIC_SOURCE" rev-parse HEAD > "$results/cunumeric-commit.txt"
julia --version > "$results/julia-version.txt"
nvidia-smi -L > "$results/gpus.txt"
free -h > "$results/host-memory.txt"
```

First test the multi-GPU launch and correctness paths at inexpensive sizes.
The two-sample smoke runs are diagnostic and do not go into the paper plots.
Inspect each `results.csv`, `planned-cases.csv`, and any failure logs before
running full sizes. Dagger checks that every requested GPU owns CuArray
chunks; the solver checks numerical convergence or exact-solution error.

```bash
BENCH_SAMPLES=2 BENCH_OUTPUT="$results/cg-smoke" \
  bash composability/krylov/run.sh weak 1024 2 4 8
ODE_SAMPLES=2 ODE_OUTPUT="$results/ode-smoke" \
  bash composability/ordinarydiffeq/run_benchmark.sh weak 128 2 4 8
```

Then produce the final seven-point one-GPU plots. Each case uses two warmups
and five synchronized timed solves by default. These one-GPU measurements
should be on the same machine, code, and Julia environments as the weak run.
The largest passing dimensions remain the weak-scaling baselines.

```bash
julia --project=. run_composability.jl --only=krylov,ordinarydiffeq \
  --mode=single --output="$results"
```

Finally run weak scaling at `1, 2, 4, 8` GPUs. The versioned
`sizes.toml` sets the one-GPU baselines for `N(G) = round(N(1) * sqrt(G))`: CG uses
`65536, 92682, 131072, 185364`; heat uses `16384, 23170, 32768, 46341`.
Select these two workloads explicitly for the final run.

```bash
julia --project=. run_composability.jl --only=krylov,ordinarydiffeq \
  --mode=multi --gpus=1,2,4,8 --output="$results"
```

Each result directory retains the raw samples, correctness metric, case
logs, exact package manifest, code commits, GPU mask, and memory diagnostics.
Plotters recompute means and standard errors from the samples. A failed case
retains its log but does not contribute a successful plotted point. Verify
GPU placement and correctness on the eight-GPU host before interpreting its
multi-GPU timing. Keep the complete result root, including the smoke runs.

For vector figures after the successful run:

```bash
GKSwstype=100 julia --project=environments/composability \
  composability/krylov/plot_results.jl single "$results/single/krylov/results.csv" \
  --output "$results/single/krylov/timings.svg"
julia --project="$ODE_PROJECT" \
  composability/ordinarydiffeq/plot_results.jl single \
  "$results/single/ordinarydiffeq/results.csv" "$results/single/ordinarydiffeq/timings.svg"
GKSwstype=100 julia --project=environments/composability \
  composability/krylov/plot_results.jl weak "$results/multi/krylov/results.csv" \
  --output "$results/multi/krylov/timings.svg"
julia --project="$ODE_PROJECT" \
  composability/ordinarydiffeq/plot_results.jl weak \
  "$results/multi/ordinarydiffeq/results.csv" \
  "$results/multi/ordinarydiffeq/timings.svg"
```
