# LOC analysis

Compares the code each model needs for the 7 shared benchmarks.

```bash
./install_scc.sh                        # scc v4.0.0 into opt/scc (needs --uloc/--cognitive)
python3 loc-analysis/analyze_model_loc.py   # writes loc-analysis/results/
```

Counts come from scc only. Before counting, temporary copies are formatted with
JuliaFormatter and black at an unbounded margin (one statement per line), so
hand wrapping doesn't change the counts. JuliaFormatter comes from
`loc-analysis/Project.toml`, black from the Python that runs the script.

Output: `summary.csv`, `overall_metrics.csv`, and `report.md`. The report gives
reductions for cuNumeric.jl vs CUDA.jl, JACC.jl, Dagger.jl and cuPyNumeric.

## Refs

`refs/<benchmark>/<model>.{jl,py}.ref` are hand-extracted from `src/`. Each is a
standalone `init` + `run!`, and mirrors the benchmarked implementation. The
external CUDA C++ references use `cuda.cu.ref` (see below).

Included: imports, data placement/distribution, and everything the timed region
calls (kernels, halos, reductions). JACC's multi-GPU path is included.
CUDA.jl is single-GPU.

Excluded:
- harness interface, sizing, correctness checks, validation errors
- timing-only syncs, tuning knobs
- workload-sharing metaprogramming (`@accelerate` is written out instead)
- test mocks (JACC `ops`)
- single-GPU-only fast paths
- library-gap workarounds (JACC `ArrayPart` `size`, cuNumeric rank-3 slicing)
- class tables
- RNG input generators (NPB LCG, FT initial field, MG RHS): calls count,
  definitions don't

Update the refs when `src/` changes.

## CUDA C++ references

`refs/nas_{ep,ft,mg}/cuda.cu.ref` are curated from
[GMAP/NPB-GPU's CUDA implementations](https://github.com/GMAP/NPB-GPU/tree/3f12d84920ee315ab00ef283717c1e74b68f4d00/CUDA),
pinned to commit `3f12d84920ee315ab00ef283717c1e74b68f4d00`. Each file retains its
upstream MIT license and attribution. These are non-executable comparison
excerpts: initialization remains in globals, `main`/`run`, and the original
setup functions. Function order, GPU kernels, indexing, launch wrappers, and
the main algorithm are preserved rather than rewritten to match the Julia API.
Consecutive declarations with the same type specifiers are grouped on one line,
preserving each declarator, initializer, and their order (including pointer stars).

The same scope exclusions above apply. In particular:

- Forward declarations, CPU warm-up computations, verification tables, debug
  output, timers, result reporting, and final teardown are removed. The default
  dynamic allocation path is retained, with unused host buffers removed.
- Device selection and launch-configuration validation are omitted. Block sizes
  use the values in the pinned upstream `CUDA/config/gpu.config`; grid and shared
  memory calculations and algorithm-dependent launch branches remain.
- RNG calls remain, but generator definitions are omitted. EP's inline jump and
  starting-seed calculations are represented by `nas_ep_batch_jump()` and
  `nas_ep_start_seed(kk, an)`. FT's plane-seed calculation is represented by
  `nas_ft_plane_starts(starts, NX, NY, NZ)`; the host-to-device seed copy and
  initial-field kernel launch remain, while that generator kernel's body is
  omitted. MG retains the `zran3` RHS call without its generator/sorting helpers.
  These helper names denote intentionally external operations, not new
  implementations supplied by this repo.
- `npbparams.hpp` denotes externally supplied problem dimensions and iteration
  counts; generated class tables are not vendored. MG receives its smoother
  coefficients as the `c` argument to `run`. FT includes the complex type and
  arithmetic macros it uses from upstream `common/npb-CPP.hpp`; unrelated common
  utilities and the legacy double-atomic compatibility fallback are excluded.

EP retains its rejection sampling, histogram, sums, and GPU atomic reductions.
The post-timing host aggregation and its buffers are excluded, matching the
existing EP refs' partial-result scope. EP keeps upstream `MK=16` (the Julia refs
use `MK=8`) and produces block partials rather than individual batch partials.

FT retains the custom FFT kernels and their helpers, including FFT coefficient
initialization, transposes, evolution, and checksum reduction. The existing
Julia/Python refs use FFT libraries, so a later comparison includes that
implementation difference. OpenMP initialization dispatch and its device
synchronization remain; timing-only synchronization between ordered FFT stages
and after evolution is omitted. EP's atomic output arrays and FT's checksum
array are explicitly zeroed during setup, making their required initial state
visible even though upstream only allocated those buffers.

MG retains level layout, host initialization, transfers, the GPU V-cycle,
periodic halos, restriction, interpolation, smoothing, residuals, and norm
reductions. Its upstream norm computes both normalized L2 and infinity norms;
the existing refs return a squared-L2 sum. The unused host residual buffer and
copy are omitted; device residual storage is filled by the retained kernels.

### FT with cuFFT

`refs/nas_ft/cuda_cufft.cu.ref` is a second, locally adapted variant of
`cuda.cu.ref`. It replaces the manual FFT stages with cuFFT; the original
reference remains available. It retains the index map, initial-field RNG call,
evolution, checksum, initialization, and grouped declarations. Manual FFT
coefficient tables, scratch buffers, transpose kernels, logarithm helpers, and
their launch settings are removed. OpenMP initialization retains its two
remaining tasks; the FFT coefficient-generation task is gone.

One `CUFFT_Z2Z` plan is created during setup and reused, with workspace managed
by cuFFT. `cufftPlan3d(&fft_plan, NZ, NY, NX, CUFFT_Z2Z)` matches the original
`x + y*NX + z*NX*NY` layout. Buffers use `cufftDoubleComplex` directly (aliased
as `dcomplex`), with `.x`/`.y` fields; checksum shared memory uses the same type.
No casts between unrelated complex structs or conversion kernels are needed.

The initial NAS `+1` transform maps to `CUFFT_INVERSE`, and the iterative NAS
`-1` transform maps to `CUFFT_FORWARD`: these names reflect cuFFT's exponent
sign convention. Both are unnormalized, so the existing checksum division by
`NTOTAL` remains the only scaling. The first execution writes `u1_device` to
`u0_device`; iterative executions transform `u1_device` in place, preserving
the evolving spectrum in `u0_device`. See NVIDIA's
[cuFFT API and transform conventions](https://docs.nvidia.com/cuda/cufft/index.html).

As with the other excerpts, omitted generator bodies, declarations, size
headers, error handling, and final teardown (including `cufftDestroy`) are
outside the comparison. This is a source reference, not a GPU-tested or
performance-tested implementation.

These files are reference-only additions. The analysis script still selects
only the existing Julia/Python variants; CUDA language/formatting support and
metric runs are outside this change.
