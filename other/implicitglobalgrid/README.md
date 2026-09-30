# ImplicitGlobalGrid Gray–Scott

Port of [`julia-con/models/diffeq/grayscott.jl`](https://github.com/JuliaLegate/benchmarking/blob/julia-con/models/diffeq/grayscott.jl),
keeping its initialization, in-place updates, and default IGG halo exchange.

`./instantiate_projects.sh` installs `environments/implicitglobalgrid` with
CUDA, ImplicitGlobalGrid, MPI, MPIPreferences, Random, Printf, and Statistics.
CUDA 5 is required by IGG 0.17.

From the repository root, set up MPI once, then run the benchmark:

```bash
bash other/implicitglobalgrid/setup_mpi.sh

# GPUS N [N_ITER=10] [N_WARMUP=5] [N_TRIALS=5]
bash other/implicitglobalgrid/run_benchmark.sh 4 14000 10 5 5
```

Setup creates (or updates) the `igg-mpi` environment under Conda's base
`envs` directory with conda-forge Open MPI 5 and UCX. It configures this Julia
project's `libmpi.so` and `mpiexec` through MPIPreferences, then resolves and
instantiates the project in a fresh Julia process. The benchmark container
already includes Conda; elsewhere, install Conda first.

Both scripts activate this environment and deactivate it on exit, including
on failure. Run them with `bash`, as above, so your calling shell is unchanged.
Set `IGG_MPI_PREFIX` to the same absolute installation path for both scripts
to use a different location. `CUNUMERIC_BENCH_CONDA` selects the Conda executable.

Setup selects Conda's CUDA version using `IGG_CUDA_VERSION`, then the container's
`CUDA_VERSION_MAJOR_MINOR`, defaulting to `13.0`. It leaves CUDA.jl's existing
toolkit selection unchanged. The container uses downloaded CUDA artifacts;
the old `local_toolkit=true` setting requires a complete local CUDA toolkit.
To use that old setting on a machine where the toolkit is installed and on
the library/tool search paths:

```bash
IGG_LOCAL_CUDA=1 bash other/implicitglobalgrid/setup_mpi.sh
```

This setting persists in the project's ignored `LocalPreferences.toml`, as
does the MPI configuration. Re-run setup if the MPI installation moves.

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

The launcher enables Conda Open MPI's CUDA support with
`OMPI_MCA_opal_cuda_support=true` and defaults `IGG_CUDAAWARE_MPI=1` for
GPU-buffer communication. It checks that Julia is configured for the active
Conda MPI installation before starting workers. Set `IGG_CUDAAWARE_MPI=0`
to use IGG's host-staged halo transfers instead. Before launching the benchmark,
a separate Julia process calls `MPI.Init()`, prints `MPI.has_cuda()`, and
finalizes MPI. If CUDA-aware transfers are enabled and the check returns
`false`, the launcher stops before starting workers. This reports MPI's CUDA
support; it does not test GPU-buffer transfers between ranks.
Open MPI's root-run flags
are set when running as root inside a container. See the
[IGG documentation](https://github.com/eth-cscs/ImplicitGlobalGrid.jl#cuda-awarerocm-aware-mpi-support)
and [Conda Open MPI instructions](https://github.com/conda-forge/openmpi-feedstock/blob/main/recipe/post-link-cuda.sh).

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
