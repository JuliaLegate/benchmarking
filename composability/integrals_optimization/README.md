# Integrals.jl + Optimization.jl: absorption image fit

This prototype recovers a synthetic gas concentration image from measurements
in several spectral bands. More gas absorbs more light. For pixel `(i,j)` and
band `c`, the forward model is

```text
measurement[c,i,j] = ∫ response[c](λ) * exp(-absorption(λ) * concentration[i,j]) dλ + noise
```

The absorption line and band responses are smooth Gaussians on a dimensionless
wavelength interval. The image is a smooth plume, and observations include
small seeded noise. Observations use twice the quadrature order used during
fitting. This is a physically interpretable synthetic inverse problem, not a
model of a particular gas or instrument.

[`model.jl`](model.jl) contains the integrand, the call to `Integrals.jl`'s
Gauss–Legendre quadrature, the mean-squared-error objective, and its gradient.
The gradient is the same integral with one extra factor of `-absorption(λ)`.
[`benchmark.jl`](benchmark.jl) gives the concentration image to
`Optimization.jl`; `OptimizationOptimisers.Adam` updates the entire backend
array. This is a gradient-based array fit because derivative-free methods are
not practical for one unknown per pixel. The benchmark's repeated solves are
timing samples; the optimization iteration loop belongs to the solver.

The default backends are single-GPU CUDA.jl `CuArray` and cuNumeric `NDArray`.
Each backend and size runs in its own Julia process. The same model, initial
image, data generation, quadrature, iteration count, and correctness checks
are used for both. The optional `cpu` backend is useful for small smoke tests.
Dagger is not included because an `Optimization.jl` array-state path has not
been validated for it. GPU validation status is summarized below; no
speedup is claimed.

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
with the default cuBLAS workspace on H100 and use one workspace setting for
both backends in a comparison:

```sh
unset CUBLAS_WORKSPACE_CONFIG
bash composability/integrals_optimization/run_benchmark.sh 32 128 256
```

The launcher sets `LEGATE_AUTO_CONFIG=0` and
`LEGATE_CONFIG="--gpus 1 --cpus 4"` by default, so cuNumeric and CUDA.jl
both use one GPU. Override either variable for a different machine setup;
the effective settings are recorded in `metadata.txt`.

The shell arguments are image dimensions `N`; each case has `N × N` pixels.
Start with `32` for correctness before larger allocations. The launcher
writes `results.csv`, `timings.png`, `metadata.txt`, and per-case logs under a
timestamped results directory. Set `INTOPT_OUTPUT=/path/to/results` to choose
one. It exits nonzero when a requested case fails and retains the failure log.
The plot compares complete `Optimization.solve` time against `N`, with
minimum/maximum error bars around the median.

`INTOPT_ELTYPE=Float64`, `INTOPT_BANDS=4`, `INTOPT_ORDER=12`,
`INTOPT_ITERS=40`, `INTOPT_SAMPLES=3`, `INTOPT_RATE=0.05`, and
`INTOPT_NOISE=0.001` control the experiment; the listed values are defaults
except for `INTOPT_ELTYPE`, which defaults to `Float32`. For a CPU smoke run:

```sh
INTOPT_BACKENDS=cpu INTOPT_SAMPLES=1 \
  bash composability/integrals_optimization/run_benchmark.sh 32
```

The timed region is one complete optimizer solve with a fresh copy of the
initial image. Synthetic observation generation, transfers, and a warmup solve
are outside timing. There is no per-iteration `GC.gc()` call. cuNumeric's
`get_time_nanoseconds()` and CUDA's `synchronize()` delimit the timed solve.
The optimizer needs a host scalar loss and a finite-gradient decision each
iteration; `allowautofetch` scopes those cuNumeric scalar decisions. The full
array state, integrand evaluations, and gradients stay on their backend; the
script rejects an optimizer result on the wrong backend. The final image is
transferred to host after timing for a relative-error check against the known
synthetic plume.

## Prototype compatibility note

The current `OptimizationOptimisers` wrapper calls `all(isfinite, gradient)`
on every step. cuNumeric implements `all` for a Boolean `NDArray` but not that
two-argument predicate call. [`benchmark.jl`](benchmark.jl) includes one
explicit prototype method that broadcasts `isfinite` and reduces the result;
this must be moved into cuNumeric and tested there before presenting GPU
numbers as unmodified package composability. The `NDArray` solve path passed
on an A30X; optimizer moment storage has not been directly inspected.

## A30X validation (2026-09-23)

On dubliner, both CUDA.jl and cuNumeric passed single-sample Float32 runs at
`N=32`, `128`, and `256` with four spectral bands, order-12 quadrature, and
40 Adam iterations. Their final losses and recovered images agreed. Nsight
Systems recorded CUDA kernels from the cuNumeric solve, confirming that its
Legate tasks executed on the GPU. These exploratory timings are not sufficient
for a paper performance claim; cuNumeric was slower at all three sizes.

The CUDA.jl `N=1024` case finished, but the cuNumeric case was interrupted
after more than five minutes without a result. The interrupt stack was inside
`Integrals.jl`'s Gauss-Legendre path during the optimizer gradient. Investigate
this scaling problem before using large image sizes or claiming a GPU win.
