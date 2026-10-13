# Integrals + Optimization plume fit

Fits a gas plume's peak concentration and width to synthetic multi-band images.
Optimization.jl (Nelder–Mead) drives the fit; each objective evaluation builds
the concentration image on the GPU and integrates every band with Integrals.jl.

```text
concentration[i,j] = background + amplitude * exp(-distance[i,j]^2 / width^2)
measurement[c,i,j] = ∫ response[c](λ) * exp(-absorption(λ) * concentration[i,j]) dλ + noise
```

The model is in [`model.jl`](model.jl); the benchmark worker is
[`benchmark.jl`](benchmark.jl).

## Run

```bash
bash composability/run.sh integrals_optimization single 32 128 256
bash composability/run.sh integrals_optimization weak 8192 1 2 4 8
INTOPT_BACKENDS=cpu INTOPT_SAMPLES=2 bash composability/run.sh integrals_optimization single 32
```

| Setting | Default |
|---|---|
| `INTOPT_ELTYPE` | Float32 |
| `INTOPT_BANDS`, `INTOPT_ORDER` | 4 bands, quadrature order 12 |
| `INTOPT_ITERS` | 80 (maximum optimizer iterations) |
| `INTOPT_SAMPLES` | 5 |
| `INTOPT_TIMEOUT` | 15m per case |

## What is measured

- One complete `Optimization.solve` per sample. Observation generation,
  transfers and two warmup solves are untimed.
- The fitted parameters are checked against the known plume after timing.
