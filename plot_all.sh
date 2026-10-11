#!/bin/bash
# Every paper figure, from results/paper (written by run_all.sh).
set -euo pipefail
cd "$(dirname "$0")"
julia=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
"$julia" --project=. plotter/plot_grid.jl --config=configs/plots/grid.toml --out=plots/grid
"$julia" --project=. plotter/plot_figures.jl
"$julia" --project=. plotter/plot_complexity.jl
"$julia" --project=. plotter/plot_tunes.jl
"$julia" --project=. plotter/plot_sweeps.jl
"$julia" --project=. plotter/plot_composability.jl combined
