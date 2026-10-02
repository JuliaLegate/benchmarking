# ImplicitGlobalGrid Gray–Scott

Port of [`julia-con/models/diffeq/grayscott.jl`](https://github.com/JuliaLegate/benchmarking/blob/julia-con/models/diffeq/grayscott.jl)
to ImplicitGlobalGrid (IGG 0.17, CUDA 5), one MPI rank per GPU.

## Setup

`./instantiate_projects.sh` installs `environments/implicitglobalgrid`, creates
or updates the Conda `igg-mpi` environment (Open MPI 5 + UCX), and points
MPIPreferences at it. Conda must be on `PATH`, or selected with
`CUNUMERIC_BENCH_CONDA=/path/to/conda`.

To install or refresh only IGG's environment:

```bash
bash other/implicitglobalgrid/setup_igg.sh
```

`IGG_MPI_PREFIX` selects the Conda environment's location, `IGG_CUDA_VERSION`
picks its CUDA (default `CUDA_VERSION_MAJOR_MINOR`, or `13.0` when unset), and
`IGG_LOCAL_CUDA=1` uses a local CUDA toolkit. These settings also apply when
running `instantiate_projects.sh`.

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

Run the weak-scaling sweep on 1, 2, 4, and 8 GPUs with global sizes 28000,
39600, 56000, and 79200, matching the harness's Gray-Scott config:

```bash
# [N_ITER=50] [N_WARMUP=2] [N_TRIALS=5]
bash other/implicitglobalgrid/run_weak_scaling.sh

# Choose another output directory or iteration counts.
IGG_OUTPUT=results/igg-run2 bash other/implicitglobalgrid/run_weak_scaling.sh 50 2 5
```

The script defaults to eight Julia threads and `IGG_CUDAAWARE_MPI=1`, unless
those variables are already set. It unsets the MPI flag on exit, including
failure; running it with `bash` leaves the caller's environment unchanged.
It writes a log for each GPU count and appends trial rows to
`$IGG_OUTPUT/Float32/grayscott_igg.csv`. The default output directory is
`results/implicitglobalgrid`, relative to the repository root. A failed run
stops the sweep and returns a nonzero exit status.

## Notes

- Each trial uses fresh arrays and `N_WARMUP` untimed steps; its time is the
  slowest rank's elapsed time over `N_ITER` steps.
- Halos travel through CUDA-aware MPI by default; `IGG_CUDAAWARE_MPI=0` stages
  them through the host.
- Physics matches [`src/benchmarks/grayscott.jl`](../../src/benchmarks/grayscott.jl),
  but initial conditions and boundaries follow the original script, so
  trajectories differ from the harness's.
