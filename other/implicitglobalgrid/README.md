# ImplicitGlobalGrid Gray–Scott

Port of [`julia-con/models/diffeq/grayscott.jl`](https://github.com/JuliaLegate/benchmarking/blob/julia-con/models/diffeq/grayscott.jl)
to ImplicitGlobalGrid (IGG 0.17, CUDA 5), one MPI rank per GPU.

## Setup

`./instantiate_projects.sh` installs `environments/implicitglobalgrid`. Then, once:

```bash
bash other/implicitglobalgrid/setup_mpi.sh
```

This creates the Conda `igg-mpi` environment (Open MPI 5 + UCX) and points
MPIPreferences at it. `IGG_MPI_PREFIX` moves it, `IGG_CUDA_VERSION` picks its
CUDA (default `13.0`), and `IGG_LOCAL_CUDA=1` uses a local CUDA toolkit.

## Run

```bash
# GPUS N [N_ITER=10] [N_WARMUP=5] [N_TRIALS=5]
bash other/implicitglobalgrid/run_benchmark.sh 4 14000 10 5 5
```

`N` is the global square domain, excluding halos; it is split across the MPI
process grid and must divide evenly. Set `IGG_OUTPUT=<run dir>` to also write
harness-compatible rows to `<run dir>/Float32/grayscott_igg.csv` (throughput in
G cell updates/s, the harness's Gray-Scott unit) for plotting with
`configs/plots/figures.toml`.

Weak-scaling sweep, matching the harness's Gray-Scott sizes:

```bash
export IGG_OUTPUT=results/implicitglobalgrid JULIA_NUM_THREADS=8
GPUS=(1 2 4 8); SIZES=(28000 39600 56000 79200)
mkdir -p "$IGG_OUTPUT"
for i in "${!GPUS[@]}"; do
    bash other/implicitglobalgrid/run_benchmark.sh "${GPUS[$i]}" "${SIZES[$i]}" 50 2 5 \
        2>&1 | tee "$IGG_OUTPUT/igg-${GPUS[$i]}gpu-N${SIZES[$i]}.log" || break
done
```

## Notes

- Each trial uses fresh arrays and `N_WARMUP` untimed steps; its time is the
  slowest rank's elapsed time over `N_ITER` steps.
- Halos travel through CUDA-aware MPI by default; `IGG_CUDAAWARE_MPI=0` stages
  them through the host.
- Physics matches [`src/benchmarks/grayscott.jl`](../../src/benchmarks/grayscott.jl),
  but initial conditions and boundaries follow the original script, so
  trajectories differ from the harness's.
