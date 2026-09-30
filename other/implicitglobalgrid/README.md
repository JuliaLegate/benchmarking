# ImplicitGlobalGrid Gray–Scott

Port of [`julia-con/models/diffeq/grayscott.jl`](https://github.com/JuliaLegate/benchmarking/blob/julia-con/models/diffeq/grayscott.jl),
keeping its initialization, in-place updates, and default IGG halo exchange.

`./instantiate_projects.sh` installs `environments/implicitglobalgrid` with
CUDA, ImplicitGlobalGrid, MPI, Random, and Printf. CUDA 5 is required by IGG 0.17.

From the repository root:

```bash
# GPUS N [STEPS=10] [WARMUP=5]
bash other/implicitglobalgrid/run_benchmark.sh 4 14000 10 5
```

`N` is the **per-GPU square size**, including the one-cell halos, as in the
original script. IGG chooses the process layout and prints the resulting global
dimensions; the combined domain need not be square. Each rank updates `(N-2)^2`
cells per step. The launcher uses the environment's MPI with one rank per GPU.
IGG selects GPUs by node-local rank, respecting `CUDA_VISIBLE_DEVICES`.
Set `JULIA` or `CUNUMERIC_BENCH_JULIA` to choose the Julia executable.

Allocation and warmup are outside timing. GPU synchronization and MPI barriers
bound the measured steps; rank zero reports the maximum elapsed time across
ranks. Throughput is cell updates/s, not FLOP/s.

The reaction terms and five-point Laplacian match the existing
[`src/benchmarks/grayscott.jl`](../../src/benchmarks/grayscott.jl), but the original
script has different initial conditions (zeros with a random patch on each
rank), a Float64 `dt`, and nonperiodic outer boundaries. The cuNumeric/CUDA
Float32 benchmark starts `u` at one, uses a global `min(150,N)` seed patch and
Float32 `dt`, and copies periodic borders from the previous step. Those
differences are preserved here, so the full trajectories are not identical.
