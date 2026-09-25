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
image arithmetic and integrals use `CuArray`, Dagger `DArray`, or `NDArray` storage. The scalar
loss reaches the optimizer through ordinary `allowautofetch` conversion; no
`all(isfinite, NDArray)` shim or `CNBool` conversion is defined.

The default single-GPU comparison is CUDA.jl, Dagger, and cuNumeric. Each backend
and size runs in its own Julia process with the same model, initial parameters,
observations, quadrature, solver, and correctness checks. The optional `cpu`
backend supports small smoke tests. Dagger's `Integrals.solve` path must pass
the same checks before it contributes a timing.

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
with the default cuBLAS workspace on H100 and use one setting for every backend:

```sh
unset CUBLAS_WORKSPACE_CONFIG
INTOPT_OUTPUT=/opt/bench-results/intopt-single bash composability/integrals_optimization/run_benchmark.sh single 32 128 256
```

The launcher enables Legate auto configuration, so it sizes the framebuffer
pool for the available GPU. The setting is recorded in `metadata.txt`.

The shell arguments are image dimensions `N`; each case has `N × N` pixels.
Start with `32` for correctness. The launcher writes `results.csv`,
`timings.png`, `metadata.txt`, and per-case logs under a timestamped results
directory, along with the package manifest and sampled GPU-memory peak. Set `INTOPT_OUTPUT=/path/to/results` to choose one. It exits nonzero
when a requested case fails and retains the failure log. The plot compares
mean complete `Optimization.solve` time against `N`, with standard-error bars.

For weak scaling, choose the largest `N` that passed all three single-GPU
variants as `N(1)`. The runner uses `N(G) = round(N(1)√G)` and runs Dagger and
cuNumeric at each count. The one-GPU point can be validated here; 2, 4, and 8
GPUs require the later machine. The weak-scaling plot shows mean time versus
GPU count with standard-error bars and a horizontal ideal reference per backend.
The complete single-GPU sweep writes `N(1)` to `base_n.txt` and stops after
the first failed size.

```sh
BASE_N=8192 # largest common passing one-GPU N in the H100 sweep below
INTOPT_OUTPUT=/opt/bench-results/intopt-weak bash composability/integrals_optimization/run_benchmark.sh weak "$BASE_N" 1 2 4 8
```

On the one-GPU H100, set `INTOPT_DRY_RUN=1` to write `planned-cases.csv` for
all GPU counts, then run `weak "$BASE_N" 1` normally. The launcher records
sampled GPU memory as a diagnostic; the reading includes Legate's reserved pool.

`INTOPT_ELTYPE=Float32`, `INTOPT_BANDS=4`, `INTOPT_ORDER=12`,
`INTOPT_ITERS=80`, `INTOPT_SAMPLES=5`, and `INTOPT_NOISE=0.001` are the defaults.
`INTOPT_ITERS` is a maximum; the CSV records the actual objective evaluation
count. For a CPU smoke run:

```sh
INTOPT_BACKENDS=cpu INTOPT_SAMPLES=2 \
  bash composability/integrals_optimization/run_benchmark.sh single 32
```

The timed region is one complete optimizer solve with a fresh copy of the
initial parameters. Synthetic observation generation, transfers, and two warmup
solves are outside timing. There is no per-iteration `GC.gc()` call. cuNumeric's
`get_time_nanoseconds()` and CUDA's `synchronize()` delimit the timed solve.
The result is checked against the known plume parameters after timing.

## Single-H100 sweep (2026-09-24)

CUDA.jl, Dagger, and cuNumeric each passed `N=32, 128, 512, 1024, 2048,
4096, 6144, 8192` on one H100. Every case used two warmups and five
synchronized complete-solve samples, recovered both parameters within
0.095% relative error, and kept image arrays on its intended GPU backend.
All variants used 32 objective evaluations. Selected mean times, in seconds,
with standard errors are:

| N | CUDA.jl | Dagger | cuNumeric |
| ---: | ---: | ---: | ---: |
| 4096 | 0.802 ± 0.014 | 6.450 ± 0.162 | 0.842 ± 0.009 |
| 6144 | 1.733 ± 0.001 | 10.850 ± 0.368 | 15.855 ± 4.256 |
| 8192 | 3.051 ± 0.001 | 20.663 ± 0.316 | 41.545 ± 0.693 |

The `N=8192` point is the largest common passing size and the weak-scaling
baseline. Its exact planned dimensions for 1, 2, 4, and 8 GPUs are `8192`,
`11585`, `16384`, and `23170`. The cuNumeric slowdown above `N=4096` is
real in the measured samples; these data do not support a claim that its
large-size curve converges with CUDA.jl. Sampled `nvidia-smi` memory reflects
Legate's automatically reserved pool and is retained as a diagnostic, not
used as a live-allocation threshold.

An instrumented `N=8192` run using Julia's `@timed` around two complete
cuNumeric solves measured 40.237 and 40.489 seconds per solve, of which
39.421 and 39.677 seconds were Julia GC time. The cuNumeric allocation
heuristic calls `GC.gc` when predicted device or host use crosses its soft
threshold. This isolates collection as the immediate cause of the slowdown;
it does not establish whether the predicted pressure, the collector, or
deferred Legate-array frees should be changed. The main plot uses the
unmodified benchmark and cuNumeric settings.

## A30X validation (2026-09-23)

On dubliner, single-sample Float32 runs at `N=16`, `128`, and `256` passed for
CUDA.jl and cuNumeric; the `N=16` CPU case also passed. All backends recovered
the same two parameters to within 0.4% relative error, and the optimizer used
32 objective evaluations. The cuNumeric runs used one configured GPU and no
benchmark-specific cuNumeric method. An Nsight Systems trace of the pushed
`N=16` run recorded cuPyNumeric CUDA kernels. These are correctness smoke runs, not
publication-grade timings; cuNumeric was slower than CUDA.jl at these sizes.

The initial `N=1024` cuNumeric run timed out after 180 seconds because its
manual Legate configuration reserved only 256 MiB of framebuffer memory on
the 24 GiB A30X. cuNumeric's 80% memory threshold then triggered about 51
Julia collections per objective evaluation; later evaluations spent about
8–9 seconds almost entirely in GC. With Legate auto configuration, the full
`N=1024` solve passed with 32 objective evaluations. A cuNumeric-only run took
0.73 seconds; the default paired run took 0.91 seconds for cuNumeric and 0.16
seconds for CUDA.jl. An explicit 16 GiB framebuffer setting also passed at
0.70 seconds. These are diagnostic single runs, not publication-grade timings.
