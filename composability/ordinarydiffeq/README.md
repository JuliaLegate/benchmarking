# OrdinaryDiffEq on cuNumeric: 2D heat equation

[`heat.jl`](heat.jl) is the user-facing example: create an `NDArray`, pass it
to `ODEProblem`, and call SciML's `solve`. It solves a five-point 2D heat
equation with `OrdinaryDiffEqLowStorageRK.CarpenterKennedy2N54`. There is no
explicit synchronization or garbage collection in the example.

[`benchmark_heat.jl`](benchmark_heat.jl) compares the same RHS and solver on
`Array`, `CuArray`, and `NDArray`. Its initial state is a discrete sine
eigenmode, so the final state has an independent exact reference. It asserts
that the solution keeps its input array backend and reports median, minimum,
maximum, and relative error.

Status: the Julia files parse, and the stencil's discrete eigenmode identity
was checked on CPU. An OrdinaryDiffEq solve and the GPU backends have not yet
run; no speedup is claimed until those results exist.

The [SciML solver documentation](https://docs.sciml.ai/OrdinaryDiffEq/stable/explicit/LowStorageRK/)
describes this as a fixed-step, fourth-order low-storage method. We set
`williamson_condition=false` because its fused RHS optimization only supports
ordinary `Array` storage. cuNumeric's existing sliced-broadcast tests exercise
the stencil operations used here.

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

To compare GPU backends after the example works:

```sh
for backend in CuArray cuNumeric; do
  julia --startup-file=no --project="$ODE_PROJECT" \
    composability/ordinarydiffeq/benchmark_heat.jl "$backend" 128 1024 4096
done
```

The benchmark also accepts `cpu` for a host baseline, for example at `N=128`.
Run each backend in its own Julia process. `ODE_ELTYPE=Float64`, `ODE_STEPS=20`,
and `ODE_SAMPLES=5` control precision, fixed time steps, and timed solves.
Transfers and initial-state construction are outside timing. The benchmark uses
`cuNumeric.get_time_nanoseconds()` for cuNumeric; that clock blocks on preceding
Legate work, so it needs no separate execution fence. It synchronizes CUDA
before reading the host clock for the CuArray case. A timing includes solver
setup and cache allocation. The work
is stencil and broadcast based, so no GEMV/cuBLAS result
from the CG benchmark should be extrapolated to it. If changing the workspace
for diagnostics, use the same setting for every backend in a comparison.

Start with the `128` correctness case before large allocations. Any solver
failure or host storage fallback exits nonzero; retain the error and package
versions. There is no cuNumeric-specific extension in this experiment.
