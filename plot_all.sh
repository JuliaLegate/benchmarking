#!/bin/bash
# Every paper figure: each configs/plots/grid*.toml -> plots/<name>/, then
# figures, complexity, tunes and size sweeps, and last the composability figure
# (its CSVs are under configs/plots/composability.toml's results_root).
set -euo pipefail
cd "$(dirname "$0")"
julia=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
for config in configs/plots/grid*.toml; do
    "$julia" --project=. plotter/plot_grid.jl --config="$config" --out="plots/$(basename "$config" .toml)"
done
"$julia" --project=. plotter/plot_figures.jl
"$julia" --project=. plotter/plot_complexity.jl
"$julia" --project=. plotter/plot_tunes.jl
"$julia" --project=. plotter/plot_sweeps.jl
"$julia" --project=. plotter/plot_composability.jl combined
