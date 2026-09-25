# OrdinaryDiffEq on cuNumeric: 2D heat equation

[`heat.jl`](heat.jl) is the user-facing example: create an `NDArray`, pass it
to `ODEProblem`, and call SciML's `solve`. It solves a five-point 2D heat
equation with `OrdinaryDiffEqLowStorageRK.CarpenterKennedy2N54`. There is no
explicit synchronization or garbage collection in the example.
The example, benchmark, and integrator smoke check all include
[`heat_rhs.jl`](heat_rhs.jl) and use the same stencil implementation.
The grid spacing is fixed at `DX = 1`, so a case with grid dimension `N`
represents the physical square `[0, (N-1)DX]²`. The five-point Laplacian
includes the `1/DX²` factor for the second spatial derivatives. Increasing
`N` expands the domain at fixed resolution, diffusivity, and final time.

[`benchmark_heat.jl`](benchmark_heat.jl) compares the same RHS and solver on
`CuArray`, `NDArray`, and GPU-backed Dagger `DArray`. Its
initial state is a discrete sine
eigenmode, so the final state has an independent exact reference. It asserts
that the solution keeps its input array backend and reports mean time, standard
error, raw samples, and relative error. [`run_benchmark.sh`](run_benchmark.sh) runs each
backend and size in a fresh Julia process, saves logs and CSV results, and
uses [`plot_results.jl`](plot_results.jl) to make a PNG comparison plot.

The `heat.jl` N=128 cuNumeric example completed on one H100 with Julia 1.13
and returned an `NDArray{Float32,2}`. A five-sample H100 sweep at 20 fixed
steps passed CUDA.jl, Dagger, and cuNumeric at `N=128, 512, 2048, 4096,
8192, 16384`, with relative exact-solution errors below `1e-4` and GPU-backed
outputs. At `N=16384`, mean complete-solve times were 1.112 s for CUDA.jl,
8.169 s for Dagger, and 1.285 s for cuNumeric. Each standard error comes
from the sample standard deviation divided by the square root of five.

The [SciML solver documentation](https://docs.sciml.ai/OrdinaryDiffEq/stable/explicit/LowStorageRK/)
describes this as a fixed-step, fourth-order low-storage method. We set
`williamson_condition=false` because its fused RHS optimization only supports
ordinary `Array` storage.

The common heat RHS evaluates the five-point stencil on five shifted interior
slices, then copies the result into the derivative array. `heat_slice` uses
`@views` for CuArray and NDArray inputs so ranged indexing does not make a
full array copy. Dagger overrides `heat_slice` to use materialized distributed
slices: a view of a `DArray` hits scalar `getindex`, which the benchmark
disables and checks at `N=128`.
The earlier version materialized CUDA slices and took 8.029 s at `N=16384`;
the view-backed version takes 1.112 s. Keep results from these two code
versions separate.
The problem uses `FullSpecialize` so SciML accepts both the initial NDArray
and solver work arrays when their storage-parent types differ. This fixed-step
example disables SciML's per-step instability scan, which scalar-iterates
custom arrays; the benchmark independently checks the final state against
the heat equation's known eigenmode solution.
The stencil computes `du/dt = κ Δ_h u`; the explicit integrator evaluates
this RHS at its stage states and advances it through 20 time steps by default.
This benchmark does not exercise an implicit integrator or its linear solves.

On a Linux GPU host with cuNumeric installed, use Julia 1.13 and create an
environment outside the benchmarking repository:

```sh
export CUNUMERIC_SOURCE=/path/to/cuNumeric.jl
julia --startup-file=no composability/ordinarydiffeq/setup.jl "$HOME/ode-smoke-env"
export ODE_PROJECT="$HOME/ode-smoke-env"
```

The setup develops the specified cuNumeric checkout and its CNPreferences
package. Keep the resulting `Manifest.toml` with any results.
Apply machine-specific cuNumeric `LocalPreferences.toml` settings to the new
environment if needed. Start with the default cuBLAS workspace on H100:

```sh
unset CUBLAS_WORKSPACE_CONFIG
julia --startup-file=no --project="$ODE_PROJECT" composability/ordinarydiffeq/heat.jl
```

For the single-GPU comparison, pass grid dimensions (each problem has `N × N`
elements):

```sh
ODE_OUTPUT=/opt/bench-results/ode-single bash composability/ordinarydiffeq/run_benchmark.sh single 128 512 2048 4096 8192 16384
```

The launcher writes `results.csv`, `timings.png`, `metadata.txt`, and one log
per case under a timestamped `composability/ordinarydiffeq/results-*` directory.
It lets Legate auto-size its framebuffer pool on each GPU.
Set `ODE_OUTPUT=/path/to/results` to choose the directory. The plot shows
mean complete solve time against N with standard-error bars. It includes every
backend that produced a valid result. The
launcher exits nonzero if any requested case fails and keeps its log.

The default single-GPU backends are `CuArray Dagger cuNumeric`. The setup
installs Dagger. The Dagger variant checks GPU-backed chunks and uses one chunk
per requested GPU. For weak scaling, choose a dimension that passed all three
single-GPU backends, amortizes launch overhead, and leaves enough memory and
time headroom for all GPU counts. A complete single-GPU sweep writes its
largest common passing size to `base_n.txt` and stops after the first failed
size; record the selected practical baseline separately if it is smaller.

```sh
BASE_N=16384 # replace with the selected common passing single-GPU N
ODE_OUTPUT=/opt/bench-results/ode-weak bash composability/ordinarydiffeq/run_benchmark.sh weak "$BASE_N" 1 2 4 8
```

For the one-H100 run, the selected dimensions at 1, 2, 4, and 8 GPUs are
`N=16384, 23170, 32768, 46341`. The weak run includes Dagger and cuNumeric.

On the one-GPU H100, set `ODE_DRY_RUN=1` to write `planned-cases.csv` for all
GPU counts, then run `weak "$BASE_N" 1` normally. The launcher records sampled
GPU memory as a diagnostic; the reading includes Legate's reserved pool.

The weak run uses Dagger and cuNumeric. It sets `N(G) = round(N(1)√G)` and
checks CUDA chunk placement, backend retention, and the same exact-solution
reference at every GPU count. Multi-GPU execution must be verified on the
eight-GPU host. The one-GPU point can be checked here. The weak-scaling plot
shows mean time versus GPU count, with standard errors and a horizontal ideal
time reference for each backend.

`ODE_ELTYPE=Float64`, `ODE_STEPS=20`, and `ODE_SAMPLES=5` control precision,
fixed time steps, and timed solves. Transfers and initial-state construction
are outside timing. Each measurement is a complete `solve` call, including
solver setup and cache allocation, which can take multiple internal stages
per time step. The cuNumeric clock, `cuNumeric.get_time_nanoseconds()`, blocks
on preceding Legate work. The CuArray and Dagger paths synchronize before
reading the host clock. The stencil does not use GEMV, so the A30 CG cuBLAS
workspace result does not predict this benchmark. Start with the default
workspace on H100; if you change it for diagnostics, use one setting across
the backends in each comparison.

Start with the `128` correctness case before large allocations. Any solver
failure or host storage fallback exits nonzero; retain the error and package
versions. The launcher saves the Manifest, sampled GPU-memory peak, and metadata with the results and
uses a 15-minute per-case timeout by default. There is no cuNumeric-specific
extension in this experiment.

## Changing the time integrator

[`integrator_smoke.jl`](integrator_smoke.jl) runs the same cuNumeric heat problem
with CarpenterKennedy2N54, RK4, Tsit5, and Vern7 in fixed-step mode, and with
RK4, Tsit5, and Vern7 in adaptive mode. It checks the returned NDArray and
the final state against the discrete eigenmode solution. All passed at
`N=128` and `N=4096` on one H100 (Float32, Julia 1.13). Install the
additional solvers in a separate ODE environment:

```sh
export ODE_PROJECT=/path/to/ode-integrators-env
ODE_INSTALL_INTEGRATORS=1 CUNUMERIC_SOURCE=/path/to/cuNumeric.jl \
  julia --startup-file=no composability/ordinarydiffeq/setup.jl "$ODE_PROJECT"
julia --project="$ODE_PROJECT" composability/ordinarydiffeq/integrator_smoke.jl 128
ODE_ADAPTIVE=1 julia --project="$ODE_PROJECT" \
  composability/ordinarydiffeq/integrator_smoke.jl 128
ODE_ADAPTIVE=1 ODE_TIMED_SAMPLES=5 julia --project="$ODE_PROJECT" \
  composability/ordinarydiffeq/integrator_smoke.jl 4096
```

The adaptive case scopes `cuNumeric.allowautofetch()` around `solve`, because
the step-size controller makes host-side scalar decisions. The timed benchmark
above continues to use the fixed-step CarpenterKennedy2N54 method.

With five synchronized samples at `N=4096`, cuNumeric mean solve times
(milliseconds ± standard error) were:

| Method | Fixed, 20 steps | Adaptive |
| --- | ---: | ---: |
| CarpenterKennedy2N54 | 120.2 ± 1.2 | — |
| RK4 | 65.3 ± 0.7 | 83.4 ± 5.5 (6 accepted steps) |
| Tsit5 | 97.2 ± 1.1 | 85.4 ± 21.1 (5 accepted steps) |
| Vern7 | 163.6 ± 0.7 | 96.1 ± 4.0 (6 accepted steps) |

Adaptive Tsit5 also passed the same GPU storage and exact-solution checks on
CuArray and Dagger at `N=128` and `N=4096`. At `N=4096`, its five-sample
means were 30.0 ± 5.2 ms on CuArray and 430.7 ± 142.7 ms on Dagger. The
adaptive Tsit5 measurements had large outliers on all three backends. They run
fewer RHS evaluations and introduce host-side step decisions, so keep the
20-step low-storage method as the main weak-scaling workload. Adaptive Tsit5
is a useful supplementary composability result; a performance comparison
would need more samples and a separate accuracy and tolerance study.
