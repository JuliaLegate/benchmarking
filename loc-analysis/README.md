# LOC analysis

Counts the code each model needs for the 7 shared benchmarks with
[scc](https://github.com/boyter/scc) v4. CUDA C++ (NPB-GPU) covers EP, FT and
MG; its cuFFT variant covers FT only.

```bash
python -m pip install -r loc-analysis/requirements.txt
python loc-analysis/analyze_model_loc.py --scc-bin /path/to/scc   # -> results/
python loc-analysis/plot_model_loc.py                             # bar charts
python loc-analysis/plot_model_comparison.py                      # vs cuNumeric.jl
```

Needs Python 3.10+, scc (`--scc-bin`, else `opt/scc/bin/scc`, else `PATH`;
`deps-install/install_scc.sh` installs it), Black, and Julia with this
directory's JuliaFormatter environment. Every script takes `--benchmarks`;
the first two also take `--variants`.

## Method

- Julia and Python refs are formatted to a 10,000-column margin before counting,
  so line counts don't depend on wrapping. CUDA refs are preformatted with
  `clang-format.yaml`.
- `results/` gets `summary.csv`, `overall_metrics.csv` and `report.md`.
  Reductions only use benchmarks both models implement.
- Comparisons report `100 * (sum(backend) / sum(cuNumeric) - 1)`; positive
  means more code than cuNumeric.jl.
- scc's complexity is a heuristic, not exact cyclomatic complexity, and
  compares poorly across languages. ULOC can exceed SLOC.
- Plots mark a missing implementation with ×, not zero. The CUDA C++ series
  uses cuFFT for FT.

## Refs

`refs/<benchmark>/<model>.{jl,py}.ref` are standalone `init` + `run!`
extracts of `src/`; update them when `src/` changes. They include imports,
data placement and everything the timed region calls (kernels, halos,
reductions, JACC's multi-GPU path). They exclude:

- harness interface, sizing, correctness checks and tuning knobs (written at
  the simplest default, e.g. one Dagger chunk per GPU)
- timing-only syncs (setup waits, waits a call already implies), setup-only
  frees, and single-GPU-only fast paths; syncs needed for correctness stay
- metaprogramming (the plain `@accelerate` macro stands in for its generator)
  and test mocks
- library-gap workarounds (e.g. Dagger's Greedy scheduler, cuPyNumeric's
  aliasing copies, JACC's flat launches) and class tables; JACC's cross-device
  copy helpers (`multi_copy!`, `multi_upload!`) are treated as library calls
- RNG generator definitions (their calls count)
- comments

Every model does the same work: each finishes the same reductions, and code
the source writes as a temporary or an `if` stays in that form.

`refs/nas_{ep,ft,mg}/cuda.cu.ref` are excerpts of NPB-GPU's CUDA code under
the same rules: kernels, launches and the algorithm are kept, while
verification, timers, reporting and teardown are removed. EP keeps upstream
`MK=16`; FT keeps its hand-written FFT. `nas_ft/cuda_cufft.cu.ref` is a local
adaptation that swaps those FFT stages for one reused cuFFT plan. None of the
CUDA refs are compiled or run. Provenance and license are in
[refs/README.md](refs/README.md).
