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
standalone `init` + `run!`, and mirrors the benchmarked implementation.

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
