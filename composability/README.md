# Composability benchmarks

The three workloads are [Krylov CG](krylov/README.md),
[OrdinaryDiffEq heat](ordinarydiffeq/README.md), and
[Integrals.jl + Optimization.jl plume calibration](integrals_optimization/README.md).
They pass `CuArray`, Dagger `DArray`, and cuNumeric `NDArray` to the same
workload operations. Each workload has its own one-GPU launcher, correctness
checks, raw synchronized timing samples, and mean-time plot with standard
error bars. CUDA.jl is the one-GPU reference; Dagger and cuNumeric also have
weak-scaling launchers. Krylov runs stock CG on all three backends and local
CG on cuNumeric.

`weak_scaling_plan.csv` is the versioned run sheet. Its `n_1`, `n_2`, `n_4`,
and `n_8` columns give the exact dimensions passed by each launcher. All
dimensions follow `N(G) = round(N(1) * sqrt(G))`. The dense CG matrix and the
two-dimensional heat grid and plume image thereby have about the same number
of elements per GPU. The one-GPU baseline for each workload came from its
common passing one-GPU sweep, and the launchers' one-GPU weak points were
exercised on the single-H100 machine.

Run all three workloads on the eight-GPU H100 machine from this combined
branch, after making the three external Julia environments as described in
the workload READMEs. The wrapper uses `/opt/bench-envs/krylov`,
`/opt/bench-envs/ode`, and `/opt/bench-envs/intopt` by default; override them
with `BENCH_PROJECT`, `ODE_PROJECT`, and `INTOPT_PROJECT` when necessary.
It records the run sheet with the output and invokes the individual launchers:

```sh
COMPOSABILITY_OUTPUT_ROOT=/opt/bench-results/composability-weak \
  bash composability/run_weak_scaling.sh 1 2 4 8
```

For command and metadata validation without running a solver, set
`COMPOSABILITY_DRY_RUN=1`. Each launcher then writes `planned-cases.csv`, its
environment metadata, and a copy of its Julia manifest. The one-GPU points
have been measured separately. The two-, four-, and eight-GPU points require
correctness and GPU-placement checks on that machine before interpreting
timings. A failed case keeps its log and does not enter the successful CSV or
plot. Each result row includes raw samples and the numerical check; plots
recompute mean and sample-standard-error from those samples.
