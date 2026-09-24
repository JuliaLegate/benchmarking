# Integrals.jl + Optimization.jl: gas plume calibration

This benchmark fits the peak concentration and spatial width of a gas plume
from synthetic images in several spectral bands. More gas absorbs more light.
At image pixel `(i,j)` and band `c`, the forward model is

```text
concentration[i,j] = background + amplitude * exp(-distance[i,j]^2 / width^2)
measurement[c,i,j] = ∫ response[c](λ) * exp(-absorption(λ) * concentration[i,j]) dλ + noise
```

The plume center and background are known; `amplitude` and `width` are fitted.
The absorption line and band responses are smooth Gaussians on a dimensionless
wavelength interval. Observations use twice the quadrature order used during
fitting and include small seeded noise. This is a physically interpretable
synthetic inverse problem, not a model of a particular gas or instrument.

[`model.jl`](model.jl) defines the integrand and calls `Integrals.jl`'s
Gauss-Legendre quadrature. `OptimizationOptimJL.NelderMead()` fits the two
positive parameters without gradients; log parameters enforce positivity.
`Optimization.jl` owns a two-element host parameter vector. Each objective
evaluation builds a backend concentration image, runs `Integrals.jl` on that
image for every band, and reduces the error to one scalar loss. The expensive
image arithmetic and integrals use `CuArray` or `NDArray` storage. The scalar
loss reaches the optimizer through ordinary `allowautofetch` conversion; no
`all(isfinite, NDArray)` shim or `CNBool` conversion is defined.

The default comparison is single-GPU CUDA.jl versus cuNumeric. Each backend
and size runs in its own Julia process with the same model, initial parameters,
observations, quadrature, solver, and correctness checks. The optional `cpu`
backend supports small smoke tests. Dagger is not included because this
Optimization.jl objective path has not been validated for it.

## Setup and run

On a Linux GPU machine with cuNumeric installed, create an environment outside
the benchmarking repository. The setup develops the specified cuNumeric
checkout and its CNPreferences package:

```sh
export CUNUMERIC_SOURCE=/path/to/cuNumeric.jl
julia --startup-file=no composability/integrals_optimization/setup.jl "$HOME/integrals-opt-env"
export INTOPT_PROJECT="$HOME/integrals-opt-env"
```

Apply machine-specific cuNumeric `LocalPreferences.toml` settings to that
environment if needed. Keep its `Manifest.toml` with benchmark results. Start
with the default cuBLAS workspace on H100 and use one setting for both backends:

```sh
unset CUBLAS_WORKSPACE_CONFIG
bash composability/integrals_optimization/run_benchmark.sh 32 128 256
```

The launcher sets `LEGATE_AUTO_CONFIG=0` and
`LEGATE_CONFIG="--gpus 1 --cpus 4"` by default, so cuNumeric and CUDA.jl
both use one GPU. Override either variable for a different machine setup;
the effective settings are recorded in `metadata.txt`.

The shell arguments are image dimensions `N`; each case has `N × N` pixels.
Start with `32` for correctness. The launcher writes `results.csv`,
`timings.png`, `metadata.txt`, and per-case logs under a timestamped results
directory. Set `INTOPT_OUTPUT=/path/to/results` to choose one. It exits nonzero
when a requested case fails and retains the failure log. The plot compares
complete `Optimization.solve` time against `N`, with minimum/maximum error bars
around the median.

`INTOPT_ELTYPE=Float32`, `INTOPT_BANDS=4`, `INTOPT_ORDER=12`,
`INTOPT_ITERS=80`, `INTOPT_SAMPLES=3`, and `INTOPT_NOISE=0.001` are the defaults.
`INTOPT_ITERS` is a maximum; the CSV records the actual objective evaluation
count. For a CPU smoke run:

```sh
INTOPT_BACKENDS=cpu INTOPT_SAMPLES=1 \
  bash composability/integrals_optimization/run_benchmark.sh 32
```

The timed region is one complete optimizer solve with a fresh copy of the
initial parameters. Synthetic observation generation, transfers, and a warmup
solve are outside timing. There is no per-iteration `GC.gc()` call. cuNumeric's
`get_time_nanoseconds()` and CUDA's `synchronize()` delimit the timed solve.
The result is checked against the known plume parameters after timing.

## A30X validation (2026-09-23)

On dubliner, single-sample Float32 runs at `N=16`, `128`, and `256` passed for
CUDA.jl and cuNumeric; the `N=16` CPU case also passed. All backends recovered
the same two parameters to within 0.4% relative error, and the optimizer used
32 objective evaluations. The cuNumeric runs used one configured GPU and no
benchmark-specific cuNumeric method. These are correctness smoke runs, not
publication-grade timings; cuNumeric was slower than CUDA.jl at these sizes.

A standalone `Integrals.jl` call returned an `NDArray` result at `N=1024`, but
the complete derivative-free cuNumeric solve did not finish within a 180-second
time limit. The corresponding CUDA.jl solve completed. Investigate the
repeated-objective scaling before using this workload to claim a performance
win at large image sizes.
