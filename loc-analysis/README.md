# LOC analysis

Compares the code each model needs for the 7 shared benchmarks. CUDA C++ covers
EP/FT/MG, and its cuFFT variant covers FT only.

```bash
python -m pip install -r loc-analysis/requirements.txt
python loc-analysis/analyze_model_loc.py --scc-bin /path/to/scc
python loc-analysis/plot_model_loc.py
```

The analysis runner requires Python 3.10+ and an existing scc v4 executable.
Julia refs also require Julia with the JuliaFormatter environment in this
directory's `Project.toml`; Python refs require Black. Set `CUNUMERIC_BENCH_JULIA`
to override the Julia executable. The existing Julia setup instantiates that
environment when formatting.

`--scc-bin` accepts a full path, including a Windows `scc.exe` path. Without it,
the runner uses `opt/scc/bin/scc` if present, otherwise `scc` on PATH. The existing
`deps-install/install_scc.sh` remains an optional Linux installer; the runner does not install scc.

The existing JuliaFormatter and Black workflow is preserved: temporary Julia
and Python copies are formatted at a 10,000-column margin before counting.
Checked-in Julia/Python refs are unchanged. CUDA C++ refs are copied byte-for-byte
to temporary `.cu` files for language detection and counted as checked in.
CUDA-only analysis does not invoke JuliaFormatter or Black.

The four C++ refs have been preformatted with clang-format 20.1.8 using
`clang-format.yaml` (10,000-column margin, attached braces, expanded short
control-flow blocks, preserved grouped declarations). Their comments and license
headers have moved out of the measured files; full licensing and attribution are
in [refs/README.md](refs/README.md). clang-format and Matplotlib are pinned in
`requirements.txt` for maintenance and plotting; the analysis runner does not
invoke clang-format.

To reformat C++ refs after future edits, from the repository root in PowerShell:

```powershell
$cudaRefs = Get-ChildItem loc-analysis/refs -Recurse -Filter "cuda*.cu.ref"
clang-format --style=file:loc-analysis/clang-format.yaml -i $cudaRefs.FullName
```

To count only the CUDA refs on Windows:

```powershell
python loc-analysis/analyze_model_loc.py --variants cuda cuda_cufft --scc-bin "C:\path\to\scc.exe" --output-dir loc-analysis/results/cuda
```

Omit `--variants` to include all models. Only explicitly unsupported pairs are
skipped; a missing expected reference is an error. Overall
tables show each model's benchmark count, and reductions use shared benchmarks
only (three for CUDA C++, one for cuFFT).

Output: `summary.csv`, `overall_metrics.csv`, and `report.md`. The report gives
reductions for cuNumeric.jl vs CUDA.jl, JACC.jl, Dagger.jl, cuPyNumeric, and the
CUDA C++ variants. scc's complexity scores are heuristic estimates and should
not be treated as exact cyclomatic complexity or direct cross-language rankings.
scc v4.0.0's ULOC includes comment/license text and unique blank lines, so it can
exceed SLOC; overall ULOC here is the sum of per-file values, not a globally
deduplicated count. See [scc's metric definitions](https://github.com/boyter/scc/tree/v4.0.0#unique-lines-of-code-uloc).

## Bar charts

`plot_model_loc.py` reads `summary.csv` and writes grouped bar charts as both
PNG and PDF, defaulting to SLOC and ULOC:

```bash
python loc-analysis/plot_model_loc.py --input loc-analysis/results/summary.csv
python loc-analysis/plot_model_loc.py --benchmarks nas_ep nas_ft nas_mg
```

Charts go to a `plots` directory beside the input CSV, or to `--output-dir`.
`--variants` filters implementations, and `--metrics sloc uloc complexity`
selects metrics (complexity is opt-in because it is not a cross-language ranking).
A missing implementation is marked with a × in its series color, never an artificial zero.
Colors match the benchmark plots in `plot_results.jl`. The grey CUDA C++ series
uses the EP/MG CUDA references and the cuFFT reference for FT; the manual FT
implementation remains in the CSV but is excluded from plots. `--variants cuda`
selects this combined C++ series.
For CUDA-only output, use `--input loc-analysis/results/cuda/summary.csv`.
No GPU or benchmark execution is involved.

### Condensed comparisons

```bash
python loc-analysis/plot_model_comparison.py
```

This writes the `comparison_pooled` grouped bar chart (PNG/PDF) and `comparisons.csv` alongside the
other plots. Each group compares cuNumeric.jl with one backend, using only their
shared benchmarks (seven for the Julia/Python backends, three for CUDA C++).
The C++ group uses cuFFT for FT. `--benchmarks nas_ep nas_ft nas_mg` restricts all
groups to the same NAS subset when a common comparison set is preferred.

All three metrics use `100 * (sum(backend_count) / sum(cuNumeric_count) - 1)`.
Positive percentages mean the other backend has higher counts than cuNumeric.jl;
negative means lower. These are changes in summed counts, not mean percentages.
Zero complexity counts are included without adjustment or pseudocounts.
A zero cuNumeric total has an undefined percentage and is displayed as N/A.
Complexity remains an scc heuristic whose language-specific rules limit
cross-language interpretation. Full benchmark coverage and totals are recorded
in `comparisons.csv`; plot colors match the benchmark palette.

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
pinned to commit `3f12d84920ee315ab00ef283717c1e74b68f4d00`. The upstream MIT license and attribution are preserved in
[refs/README.md](refs/README.md), outside the measured C++ files. These are non-executable comparison
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

These are source-comparison references only; the analysis script counts them without compiling or executing any benchmark.
