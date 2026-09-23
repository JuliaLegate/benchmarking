# OrdinaryDiffEq on cuNumeric: 2D heat equation

[`heat.jl`](heat.jl) is the user-facing example: create an `NDArray`, pass it
to `ODEProblem`, and call SciML's `solve`. It solves a five-point 2D heat
equation with `OrdinaryDiffEqLowStorageRK.CarpenterKennedy2N54`. There is no
explicit synchronization or garbage collection in the example.
The example and benchmark both include [`heat_rhs.jl`](heat_rhs.jl), so they
use the same stencil implementation.
The grid spacing is fixed at `DX = 1`, so a case with grid dimension `N`
represents the physical square `[0, (N-1)DX]²`. The five-point Laplacian
includes the `1/DX²` factor for the second spatial derivatives. Increasing
`N` expands the domain at fixed resolution, diffusivity, and final time.

[`benchmark_heat.jl`](benchmark_heat.jl) compares the same RHS and solver on
`Array`, `CuArray`, and `NDArray`, with an optional Dagger `DArray` probe. Its
initial state is a discrete sine
eigenmode, so the final state has an independent exact reference. It asserts
that the solution keeps its input array backend and reports median, minimum,
maximum, and relative error. [`run_benchmark.sh`](run_benchmark.sh) runs each
backend and size in a fresh Julia process, saves logs and CSV results, and
uses [`plot_results.jl`](plot_results.jl) to make a PNG comparison plot.

Status: the `heat.jl` N=128 cuNumeric example completed on one H100 with
Julia 1.13 and returned an `NDArray{Float32,2}`. The N=128 cuNumeric benchmark
completed with `ODE_SAMPLES=1` and relative error `4.50e-7`; the N=128 CuArray
benchmark also completed. Larger comparisons remain unrun; no speedup is
claimed from these smoke tests.

The [SciML solver documentation](https://docs.sciml.ai/OrdinaryDiffEq/stable/explicit/LowStorageRK/)
describes this as a fixed-step, fourth-order low-storage method. We set
`williamson_condition=false` because its fused RHS optimization only supports
ordinary `Array` storage. cuNumeric's existing sliced-broadcast tests exercise
the stencil operations used here.
The problem uses `FullSpecialize` so SciML accepts both the initial NDArray
and solver work arrays when their storage-parent types differ. This fixed-step
example disables SciML's per-step instability scan, which scalar-iterates
custom arrays; the benchmark independently checks the final state against
the heat equation's known eigenmode solution.
The stencil computes `du/dt = κ Δ_h u`; the explicit integrator evaluates
this RHS at its stage states and advances it through 20 time steps by default.
This benchmark does not exercise an implicit integrator or its linear solves.

On a Linux GPU host with cuNumeric installed, use Julia 1.12 and create an
environment outside the benchmarking repository:

```sh
export CUNUMERIC_SOURCE=/path/to/cuNumeric.jl
julia --startup-file=no composability/ordinarydiffeq/setup.jl "$HOME/ode-smoke-env"
export ODE_PROJECT="$HOME/ode-smoke-env"
```

The setup develops the specified cuNumeric checkout and its CNPreferences
package. Keep the resulting `Manifest.toml` with any results.
Apply machine-specific cuNumeric `LocalPreferences.toml` settings to the new
environment if needed. Start with the default cuBLAS workspace on H100:

```sh
unset CUBLAS_WORKSPACE_CONFIG
julia --startup-file=no --project="$ODE_PROJECT" composability/ordinarydiffeq/heat.jl
```

To compare single-GPU CUDA.jl and cuNumeric after the example works, pass the
grid dimensions you want to test (each problem has `N × N` elements):

```sh
bash composability/ordinarydiffeq/run_benchmark.sh 128 1024 4096
```

The launcher writes `results.csv`, `timings.png`, `metadata.txt`, and one log
per case under a timestamped `composability/ordinarydiffeq/results-*` directory.
Set `ODE_OUTPUT=/path/to/results` to choose the directory. The plot shows
complete solve time against N, with median of the timed solves and min/max
error bars. It includes every backend that produced a valid result. The
launcher exits nonzero if any requested case fails and keeps its log.

The default backends are `CuArray cuNumeric`. For an optional single-GPU
Dagger `DArray` check and comparison, install Dagger into the same environment
and request it explicitly:

```sh
ODE_INSTALL_DAGGER=1 CUNUMERIC_SOURCE=/path/to/cuNumeric.jl \
  julia --startup-file=no composability/ordinarydiffeq/setup.jl "$ODE_PROJECT"
ODE_BACKENDS="CuArray cuNumeric Dagger" \
  bash composability/ordinarydiffeq/run_benchmark.sh 128 1024 4096
```

The Dagger run uses one CUDA worker and one GPU chunk. It is a composability
probe: an unmodified OrdinaryDiffEq solve must complete, retain GPU-backed
`DArray` storage, and pass the same numerical check before it contributes a
CSV row. This path has not been run on a GPU yet. `ODE_BACKENDS="cpu CuArray
cuNumeric"` adds a host baseline; use small N for `cpu`.

`ODE_ELTYPE=Float64`, `ODE_STEPS=20`, and `ODE_SAMPLES=5` control precision,
fixed time steps, and timed solves. Transfers and initial-state construction
are outside timing. Each measurement is a complete `solve` call, including
solver setup and cache allocation, which can take multiple internal stages
per time step. The cuNumeric clock, `cuNumeric.get_time_nanoseconds()`, blocks
on preceding Legate work. The CuArray and Dagger paths synchronize before
reading the host clock. The stencil does not use GEMV, so the A30 CG cuBLAS
workspace result does not predict this benchmark. Start with the default
workspace on H100; if you change it for diagnostics, use one setting across
the backends in each comparison.

Start with the `128` correctness case before large allocations. Any solver
failure or host storage fallback exits nonzero; retain the error and package
versions. There is no cuNumeric-specific extension in this experiment.
