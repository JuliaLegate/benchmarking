# ImplicitGlobalGrid Gray–Scott

Port of [`julia-con/models/diffeq/grayscott.jl`](https://github.com/JuliaLegate/benchmarking/blob/julia-con/models/diffeq/grayscott.jl),
keeping its initialization, in-place updates, and default IGG halo exchange.

`./instantiate_projects.sh` installs `environments/implicitglobalgrid` with
CUDA, ImplicitGlobalGrid, MPI, Random, Printf, and Statistics. CUDA 5 is required
by IGG 0.17.

From the repository root:

```bash
# GPUS N [N_ITER=10] [N_WARMUP=5] [N_TRIALS=5]
bash other/implicitglobalgrid/run_benchmark.sh 4 14000 10 5 5
```

A sweep pairs each GPU count with one local size:

```bash
GPUS=(1 2 4 8)
SIZES=(14000 14000 14000 14000)
N_WARMUP=5
N_ITER=10
N_TRIALS=5

for i in "${!GPUS[@]}"; do
    bash other/implicitglobalgrid/run_benchmark.sh \
        "${GPUS[$i]}" "${SIZES[$i]}" "$N_ITER" "$N_WARMUP" "$N_TRIALS"
done
```

The launcher also accepts `N_ITER`, `N_WARMUP`, and `N_TRIALS` as environment
variables; explicit positional arguments take precedence.

`N` is the **per-GPU square size**, including the one-cell halos, as in the
original script. IGG chooses the process layout and prints the resulting global
dimensions; the combined domain need not be square. Each rank updates `(N-2)^2`
cells per step. The launcher uses the environment's MPI with one rank per GPU.
IGG selects GPUs by node-local rank, respecting `CUDA_VISIBLE_DEVICES`.
Set `JULIA` or `CUNUMERIC_BENCH_JULIA` to choose the Julia executable.

Each trial starts with fresh arrays, runs `N_WARMUP` untimed steps, and measures
`N_ITER` steps. Allocation and warmup are outside timing. GPU synchronization
and MPI barriers bound the measured steps. Each trial's time is the maximum
elapsed time across ranks divided by `N_ITER`.

Rank zero reports the mean, sample standard deviation, and standard error of
the mean (`stddev / sqrt(N_TRIALS)`) across the trial averages. With one trial,
standard deviation and standard error are unavailable (`NaN`). These describe
timing variability, not numerical solution error. Throughput is cell updates/s,
not FLOP/s.

The reaction terms and five-point Laplacian match the existing
[`src/benchmarks/grayscott.jl`](../../src/benchmarks/grayscott.jl), but the original
script has different initial conditions (zeros with a random patch on each
rank), a Float64 `dt`, and nonperiodic outer boundaries. The cuNumeric/CUDA
Float32 benchmark starts `u` at one, uses a global `min(150,N)` seed patch and
Float32 `dt`, and copies periodic borders from the previous step. Those
differences are preserved here, so the full trajectories are not identical.
