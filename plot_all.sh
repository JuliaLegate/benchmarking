#!/bin/bash
# Each configs/plots/grid*.toml -> plots/<name>/ (grid + speedup summary),
# then the standalone figures in configs/plots/figures.toml -> plots/figures/,
# the complexity plots -> plots/complexity/ and the Dagger tunes -> plots/tunes/.
set -euo pipefail
cd "$(dirname "$0")"
julia=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
for config in configs/plots/grid*.toml; do
    [[ $config == *.example.toml ]] && continue
    "$julia" --project=. plot_grid.jl --config="$config" --out="plots/$(basename "$config" .toml)"
done
"$julia" --project=. plot_figures.jl
"$julia" --project=. plot_complexity.jl
"$julia" --project=. plot_tunes.jl
