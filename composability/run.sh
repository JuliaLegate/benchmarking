#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
julia_bin=${JULIA:-julia}
export BENCH_ELTYPE=${BENCH_ELTYPE:-Float32}
export LEGATE_AUTO_CONFIG=${LEGATE_AUTO_CONFIG:-0}
export LEGATE_CONFIG=${LEGATE_CONFIG:-"--gpus 1 --cpus 2 --fbmem 22000 --sysmem 65536 --zcmem 1024"}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if (( $# == 0 )); then set -- 8192; fi
output=${BENCH_OUTPUT:-"$PWD/composability-$(date +%Y%m%d-%H%M%S)-$$"}
mkdir -p "$output"
csv="$output/results.csv"
printf '%s\n' 'backend,solver,mode,eltype,n,iterations,median_ms,min_ms,max_ms,relative_residual,samples_ms' > "$csv"
{
    git -C "$script_dir" rev-parse HEAD
    "$julia_bin" --version
    nvidia-smi
    printf 'CUBLAS_WORKSPACE_CONFIG=%s\nLEGATE_CONFIG=%s\n' "${CUBLAS_WORKSPACE_CONFIG:-<default>}" "$LEGATE_CONFIG"
    "$julia_bin" --startup-file=no --project="$script_dir" -e 'using Pkg; Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)'
} > "$output/environment.txt" 2>&1
failed=0
for n in "$@"; do
    [[ "$n" =~ ^[0-9]+$ ]] && (( n > 1 )) || { echo "Invalid N: $n" >&2; exit 2; }
    for backend in ${BENCH_BACKENDS:-"cuNumeric CuArray"}; do
        for solver in cg bicgstab; do
            for mode in stock plain fused; do
                log="$output/$BENCH_ELTYPE-$backend-$solver-$mode-$n.log"
                if "$julia_bin" -t4 --startup-file=no --project="$script_dir" \
                    "$script_dir/krylov.jl" "$backend" "$solver" "$mode" "$n" > "$log" 2>&1; then
                    sed -n 's/^RESULT,//p' "$log" >> "$csv"
                else
                    echo "Failed: $backend $solver $mode N=$n ($log)" >&2
                    failed=1
                fi
            done
        done
    done
done
echo "Results: $csv"
exit "$failed"
