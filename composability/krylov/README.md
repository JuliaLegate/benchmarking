# Krylov solvers

Krylov.jl CG (default) or BiCGStab on one dense matrix and right-hand side.

| Label | Solver |
|---|---|
| CUDA | stock `Krylov.cg!` / `bicgstab!` on `CuArray` |
| Dagger | stock solver on GPU-backed Dagger arrays |
| cuNumeric | stock solver on `NDArray` |
| cuNumeric local | local CG/BiCGStab loop with `@accelerate` updates (`--local`) |

## Run

```bash
julia --project=. run_composability.jl --only=krylov --solvers=cg,bicgstab --mode=both
bash composability/run.sh krylov single 1024 2048
BENCH_SOLVERS=cg,bicgstab BENCH_LOCAL=1 bash composability/run.sh krylov weak 65536 1 2 4 8
```

`BENCH_ELTYPE` (Float32), `BENCH_THREADS` and `BENCH_CPUS` tune the run.

## What is measured

- One complete solve per sample, five samples per case, each in a fresh process.
  Matrix construction, transfer and one warmup are untimed.
- A result passes when the residual `norm(A*x - b) / norm(b)`, checked in
  Float64 on the host, is at most `1e-5` (Float32) or `1e-8` (Float64).
- Dagger splits the matrix into column strips; `DAGGER_BLOCKS_PER_GPU` sets
  strips per GPU.

## Plot

```bash
julia --project=environments/composability composability/krylov/plot_results.jl weak results.csv timings.png
```

With several solvers in the CSV it writes `timings-cg.png` and `timings-bicgstab.png`.
