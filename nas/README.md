# NAS benchmarks

EP, FT, and MG from GMAP/NPB-GPU at commit
`3f12d84920ee315ab00ef283717c1e74b68f4d00` (license:
[`THIRD_PARTY_LICENSE.md`](THIRD_PARTY_LICENSE.md)). Don't change the commit
without re-validating the official verification values. Backends live in
`src/<model>/benchmarks/nas/`.

## Running

The class sets the problem size; N and M are derived from it.

| Config | What it runs |
| --- | --- |
| `configs/single_gpu/nas_{ep,ft,mg}.toml` | Class B, one GPU, all models |
| `configs/single_gpu/nas_{ep,mg}_compare.toml` | Adds CUDA.jl/JACC variants (below) |
| `configs/multi_gpu/nas_*_strong.toml` | Class B on 1, 2, 4, 8 GPUs |
| `configs/multi_gpu/nas_*_weak.toml` | `class = [...]` zipped with `gpus` (A–D for FT, B–E otherwise) |

```sh
julia --project=. run.jl --config=configs/single_gpu/nas_ep.toml
```

Every config uses Float64 and `n_iter = 1`; use `n_trial` for repeats. Each
run is verified against the official NPB values after timing.

NPB classes don't grow in step with GPU count, so per-GPU work varies along a
weak-scaling ladder. The efficiency plot uses throughput, h(P) / (P · h(1)).

## Multi-GPU support

| | EP | FT | MG |
| --- | --- | --- | --- |
| cuNumeric, cuPyNumeric | partitioned | slab-partitioned FFT | partitioned |
| Dagger | partitioned | distributed FFT | partitioned |
| JACC | partitioned | 1 GPU | `JACC.Multi` z-slabs |
| CUDA.jl | 1 GPU | 1 GPU | 1 GPU |

cuNumeric and cuPyNumeric's FFT task broadcasts every transformed axis
(`src/ndarray/detail/fft.jl` in cuNumeric.jl), so a single 3-D FFT call runs
as one task. FT therefore does two slab passes, a 2-D FFT over the last two
axes and then a 1-D FFT over the first; each pass splits across GPUs along the
axes it does not transform, and Legate moves data between them.

## Fairness

These are the same problems, not identical kernels or certified NPB scores.
Throughput differences include algorithm choices, not only runtime overhead.
Each adapter's header lists its limitations.

## EP

One sample generates the class's `2^(m+1)` random numbers with NPB's 46-bit
LCG, applies the Gaussian transform, and produces the 10-bin histogram and
`sx`/`sy` sums. All models use `MK = 8` (NPB allows changing it; results are
identical), giving 256 pairs per independent stream. Final aggregation is
untimed.

- **cuNumeric**: broadcasts the stream function into one `NDArray{NASEPPartial}`.
- **cuPyNumeric**: array skip-ahead in slabs of up to `2^24` pairs.
  `CUPYNUMERIC_NAS_EP_IMPL=recurrence` selects the older stepwise version.
- **CUDA.jl, JACC**: per-stream kernels by default; `*_NAS_EP_IMPL=broadcast`
  (or `nas_ep_compare.toml`) runs the array-broadcast variant.
- **Dagger**: broadcasts the stream function over a GPU `DArray`.

EP produces two plots: `nas_ep_high_level_*_scaling.png` (array/broadcast
APIs) and `nas_ep_explicit_kernels_*_scaling.png` (CUDA.jl and JACC kernels).
cuNumeric saves to `nas_ep_cunumeric_struct.csv`.

## FT

One sample generates the initial field, runs one forward 3-D FFT, then for each
of the class's `NITER` iterations evolves the spectrum, runs an inverse FFT,
and takes the 1024-point checksum. All of this is timed; verification (relative
tolerance `1e-12`) is not.

- **CUDA.jl, JACC**: device RNG and cuFFT (JACC has no FFT API).
- **cuNumeric**: host RNG, attached upload, unnormalized inverse with a
  pre-scaled full-volume checksum mask.
- **cuPyNumeric**: host RNG, `fftn`/`ifftn`, checksum via `take`.
- **Dagger**: host RNG, distributed FFT, full-volume mask; cross-GPU checksum
  aggregation is untimed.

cuNumeric GPU regression (S/W/B under a 12 GiB framebuffer cap):

```sh
LEGATE_AUTO_CONFIG=0 LEGATE_CONFIG="--gpus=1 --cpus=1 --fbmem=12288" \
  julia --project=environments/cunumeric test/nas_ft_gpu.jl
```

## MG

One sample clears the hierarchy, computes the initial residual, runs the class's
fixed number of V-cycles (periodic exchange, 27-point residual, restriction,
interpolation, smoother), and computes initial and final L2 sums of squares.
The impulse search is setup; the square root and verification (relative
tolerance `1e-8`) are untimed. NPB's Linf norm is omitted.

- **CUDA.jl**: direct kernels; the `separable` variant (`nas_mg_compare.toml`)
  uses cuNumeric's three-axis restriction/interpolation decomposition.
- **JACC**: `JACC.Multi` z-slabs on every GPU count (`mg_multi.jl`): ghost
  planes via `sync_ghost_elems!` (host-staged), small levels replicated per GPU,
  custom GPU-to-GPU copies for the periodic wrap and the slab-to-replicated
  gather. `JACC_NAS_MG_IMPL=single` runs the original single-GPU kernels.
- **cuNumeric, cuPyNumeric, Dagger**: distributed arrays. Dagger's restriction
  evaluates the full fine grid, and it combines per-slab norms after timing.
