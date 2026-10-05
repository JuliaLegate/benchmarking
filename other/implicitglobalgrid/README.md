# ImplicitGlobalGrid Gray–Scott

Port of [`julia-con/models/diffeq/grayscott.jl`](https://github.com/JuliaLegate/benchmarking/blob/julia-con/models/diffeq/grayscott.jl)
to ImplicitGlobalGrid (IGG 0.17, CUDA 5), one MPI rank per GPU.

## Setup

`./deps-install/instantiate_projects.sh` sets this up. To install or refresh
only IGG (Julia environment plus the Conda `igg-mpi` Open MPI 5 + UCX environment):

```bash
bash deps-install/setup_igg.sh
```

Conda must be on `PATH` or set with `CUNUMERIC_BENCH_CONDA`. `IGG_MPI_PREFIX`
moves the Conda environment, `IGG_CUDA_VERSION` picks its CUDA (default 13.0),
and `IGG_LOCAL_CUDA=1` uses a local toolkit.

## Run

```bash
# GPUS N [N_ITER=10] [N_WARMUP=5] [N_TRIALS=5]
bash other/implicitglobalgrid/run_benchmark.sh 4 14000

# Weak scaling, sizes of configs/multi_gpu/grayscott.toml (8 GPUs: N=79198)
# [N_ITER=50] [N_WARMUP=2] [N_TRIALS=5]
bash other/implicitglobalgrid/run_weak_scaling.sh
```

`N` is the global array size, as in the harness: the outer ring holds periodic
copies, so `N-2` cells per dimension are updated, and `N-2` must split evenly
across the process grid. With `IGG_OUTPUT` set, rows are appended to
`$IGG_OUTPUT/Float32/grayscott_igg.csv` in the harness's format. The sweep
defaults it to `results/implicitglobalgrid` and also saves one log per GPU
count. `IGG_VERBOSE=1` prints startup progress to locate hangs.

## Method

- Periodic on both axes, with the same initialization recipe as the JACC and
  Dagger reference (`u = 1`, `v = 0`, one random square patch).
- One CUDA kernel updates both species; halos are exchanged after each step,
  without overlapping communication and computation.
- Halos use CUDA-aware MPI; `IGG_CUDAAWARE_MPI=0` stages them through the host.
- Each trial reports the slowest rank's time over `N_ITER` steps. CSV
  correctness is `skipped`; `test/igg_grayscott.jl` checks the update and
  initialization on the CPU.
