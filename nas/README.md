# NAS benchmarks

EP, FT and MG from GMAP/NPB-GPU at commit
`3f12d84920ee315ab00ef283717c1e74b68f4d00` (license:
[`THIRD_PARTY_LICENSE.md`](THIRD_PARTY_LICENSE.md)). Re-validate the official
verification values before changing the commit. Classes and references are in
`src/nas/`; adapters are in `src/<model>/benchmarks/nas/`.

These are the same problems, not identical kernels or certified NPB scores.
Each adapter's header lists its deviations.

## Running

| Config | Runs |
| --- | --- |
| `configs/single_gpu/nas_{ep,ft,mg}.toml` | Class B, one GPU |
| `configs/single_gpu/nas_mg_compare.toml` | Adds CUDA.jl's `separable` MG |
| `configs/multi_gpu/nas_*_weak.toml` | Weak scaling on 1, 2, 4, 8 GPUs |
| `configs/multi_gpu/nas_*_strong.toml` | Class B on 1, 2, 4, 8 GPUs |

```sh
julia --project=. run.jl --config=configs/multi_gpu/nas_ft_weak.toml
```

All configs use Float64 and `n_iter = 1` (one full class run per sample).
Official classes don't grow with GPU count, so weak scaling adds sizes named
`<class>.<k>` that keep the work per GPU roughly fixed:

| | 1 GPU | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| EP | B | B.2 | C | C.8 |
| FT | A | A.2 | B.4 | B.8 |
| MG | B | B.2 | B.4 | C |

Custom sizes have no NAS reference and report `skipped` correctness, except
FT B.4 (class B's grid for 6 iterations), which checks B's first 6 checksums.

## EP

Each sample runs NPB's 46-bit LCG and Gaussian transform for `2^(m+1)` numbers,
producing the 10-bin histogram and `sx`/`sy` sums. Every model uses `MK = 8`
(256 pairs per stream) and splits streams across GPUs. Final aggregation is
untimed.

- **cuNumeric**: broadcasts the stream function into one `NDArray` of struct
  partials (`nas_ep_cunumeric_struct.csv`).
- **cuPyNumeric**: array skip-ahead; materializes larger intermediates.
- **CUDA.jl, Dagger**: broadcast over a `CuArray` / GPU `DArray`.
- **JACC**: `JACC.Multi.parallel_for` over the streams.

## FT

The initial field and twiddle are built before timing (NPB times them; here
host vs device RNG would dominate). Each sample copies the field on the device,
runs one forward 3-D FFT, then for each of `NITER` iterations evolves the
spectrum, inverse transforms it and takes the 1024-point checksum.
Verification (relative tolerance `1e-12`) is untimed.

- **cuNumeric**: host RNG, one `fft!`/`bfft!` per 3-D transform, masked
  checksum reduction.
- **cuPyNumeric**: host RNG, one `fftn`/`ifftn` per transform, checksum via
  `take`.
- **CUDA.jl**: device RNG and cuFFT.
- **JACC**: `JACC.Multi` slabs with cuFFT per GPU (JACC has no FFT): z-slabs
  for the x/y FFTs, y-slabs for the z FFT, and a GPU-to-GPU all-to-all between
  them.
- **Dagger**: host RNG, Dagger's distributed FFT; cross-GPU checksum
  aggregation is untimed.

## MG

Each sample clears the hierarchy, computes the initial residual, runs the
class's V-cycles (periodic exchange, 27-point residual, restriction,
interpolation, smoother) and the L2 sums of squares. The sparse right-hand
side is built on the host before timing, as in NPB-GPU; verification
(relative tolerance `1e-8`) is untimed and NPB's Linf norm is omitted.

- **cuNumeric**: views and broadcasts with separable transfer passes;
  `@accelerate` on every operator.
- **cuPyNumeric**: views and array expressions with separable transfer passes.
- **CUDA.jl**: fused `CuArray` broadcasts; `separable` uses cuNumeric's
  three-axis transfer decomposition.
- **JACC**: `JACC.Multi` z-slabs with ghost planes; small levels replicated
  on every GPU.
- **Dagger**: distributed periodic stencils; restriction filters the whole
  fine grid, and norm aggregation is untimed.
