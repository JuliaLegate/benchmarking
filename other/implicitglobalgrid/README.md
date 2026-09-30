# ImplicitGlobalGrid Gray–Scott

Standalone Float32 CUDA/MPI benchmark, ported from
[`julia-con/models/diffeq/grayscott.jl`](https://github.com/JuliaLegate/benchmarking/blob/julia-con/models/diffeq/grayscott.jl).
Each MPI rank uses one GPU. It runs independently of `run.jl`.

## Setup and run

`./instantiate_projects.sh` installs the environment with the other benchmarks.
To install just this environment:

```bash
julia --project=environments/implicitglobalgrid -e 'using Pkg; Pkg.instantiate()'
```

From the repository root (the launcher also works from other directories):

```bash
# GPUS N [STEPS=10] [WARMUP=5]
bash other/implicitglobalgrid/run_benchmark.sh 4 28000 10 5

# Small deterministic comparison against the existing cuNumeric/CUDA step.
bash other/implicitglobalgrid/run_benchmark.sh 4 32 5 2 --check
```

`N` is the **global N × N matrix size**, including its physical outer border;
communication halos are extra. The launcher chooses the most compact equal
rectangular tiling that divides N, with at least two physical cells per local
axis. For example, 4 GPUs use 2 × 2 tiles and 8 GPUs use 2 × 4 tiles. Incompatible
sizes fail rather than silently changing the workload. Only the global matrix
must be square.

The shell launcher targets one node and honors `CUDA_VISIBLE_DEVICES`, using
the first requested devices in their existing order. `JULIA` overrides
`CUNUMERIC_BENCH_JULIA`, which overrides `julia`. MPI.jl selects the matching
MPI launcher from this environment; a separate `mpiexecjl` install is unnecessary.
ImplicitGlobalGrid selects each GPU using the node-local MPI rank.

The only direct dependencies are CUDA, ImplicitGlobalGrid, MPI, Printf, and
Random. CUDA is constrained to major version 5 because ImplicitGlobalGrid 0.17's
[compatibility bounds](https://github.com/eth-cscs/ImplicitGlobalGrid.jl/blob/v0.17.1/Project.toml)
do not yet include CUDA 6. Host-staged MPI transfers work by default. If your
configured MPI supports CUDA-aware communication, opt in with
`IGG_CUDAAWARE_MPI=1`; see [MPI.jl configuration](https://juliaparallel.org/MPI.jl/stable/configuration/).

## Workload comparison

The reference is [`src/benchmarks/grayscott.jl`](../../src/benchmarks/grayscott.jl),
used by cuNumeric and CUDA. The port matches its five-point Laplacian,
reaction terms, forward-Euler update, Float32 constants (`dx=1`, `dt=0.2`,
`c_u=1`, `c_v=0.3`, `f=0.03`, `k=0.06`), and double-buffer swaps. It also matches
the initial ones/zeros fields with a random patch in `1:min(150,N)` on both axes.
The random patch uses a fixed host seed for partition-independent initialization;
random draws need not equal another backend's RNG stream.

The current cuNumeric/CUDA reference updates only `2:N-1` and copies the
**previous** step's opposite interior into the outer border. Row copies overwrite
corner values after column copies. This is deliberately preserved, including at
partition edges. Two communication halo cells per side allow access to the
opposite second physical cell. IGG's periodic transport exchanges the new local
fields after each step; it does not change that physical boundary rule.

The original script instead started both fields at zero, seeded a tenth of each
local domain, used a Float64 `dt`, updated in place, and did not enable periodic
MPI boundaries. Allocation was inside its timer and its reported FLOP count
was incorrect. Those differences are corrected here. Dagger/JACC currently use
a different boundary convention that evolves every physical cell periodically;
this port is not numerically equivalent to those boundaries.

Warmup and allocation are excluded from timing. Timesteps form one continuous
trajectory. Each step includes GPU synchronization and halo exchange; the final
time is the maximum elapsed time across ranks divided by the measured steps.
Rank zero prints a CSV header and result, plus human-readable timings. Throughput
is `(N-2)^2` interior cell updates per step, reported in billions of updates/s,
not FLOP/s. Scratch space includes four interior-sized work arrays in addition
to the four field buffers and IGG communication buffers.

## Verification

The array-only test uses the actual existing `GRAYSCOTT_STEP_BODY`, checks the
full fields (including borders/corners), and exercises multiple steps, tilings,
and initial conditions without GPU packages:

```bash
julia --startup-file=no test/implicitglobalgrid.jl
```

Real MPI/IGG communication can also be checked without GPUs:

```bash
julia --project=environments/implicitglobalgrid test/implicitglobalgrid_mpi.jl
```

Both checks run in setup CI. `--check` on the shell launcher runs the production
GPU path with the shared deterministic initial conditions and compares the full
result after warmup plus measured steps to the existing CPU step (N ≤ 512).
