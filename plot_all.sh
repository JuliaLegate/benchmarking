#!/bin/bash
# Grid + speedup summary (configs/plots/grid.toml), then standalone figures
# (configs/plots/figures.toml).
set -euo pipefail
cd "$(dirname "$0")"
julia=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
"$julia" --project=. plot_grid.jl
"$julia" --project=. plot_figures.jl
