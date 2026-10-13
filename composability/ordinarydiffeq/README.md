# OrdinaryDiffEq heat equation

A 2D five-point heat equation solved with SciML's `solve` and the fixed-step
`CarpenterKennedy2N54` method, on `CuArray`, `NDArray` and Dagger `DArray`.

| File | Contents |
|---|---|
| [`heat.jl`](heat.jl) | Minimal user-facing example on `NDArray` |
| [`heat_rhs.jl`](heat_rhs.jl) | Stencil shared by every file |
| [`benchmark_heat.jl`](benchmark_heat.jl) | Benchmark worker |
| [`integrator_smoke.jl`](integrator_smoke.jl) | Same problem with RK4, Tsit5, Vern7 (fixed and adaptive) |

## Run

```bash
julia --project=environments/composability composability/ordinarydiffeq/heat.jl
bash composability/run.sh ordinarydiffeq single 128 512 1024
bash composability/run.sh ordinarydiffeq weak 16384 1 2 4 8
```

`ODE_ELTYPE` (Float32), `ODE_STEPS` (20) and `ODE_SAMPLES` (5) tune the run.

## What is measured

- One complete `solve` per sample, each in a fresh process, including solver
  setup. Initial state, transfer and one warmup are untimed.
- The initial state is a sine eigenmode, so the result is checked against the
  exact solution: relative error below `1e-4` (Float32) or `1e-7` (Float64).

## Other integrators

```bash
julia --project=environments/composability composability/ordinarydiffeq/integrator_smoke.jl 128
ODE_ADAPTIVE=1 julia --project=environments/composability composability/ordinarydiffeq/integrator_smoke.jl 128
```

Adaptive stepping wraps `solve` in `cuNumeric.allowfetch()` for the host-side
step-size decisions.
