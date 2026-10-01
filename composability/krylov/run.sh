#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../common.sh"
julia_bin=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
project=${BENCH_PROJECT:-"$script_dir/../../environments/composability"}
export BENCH_ELTYPE=${BENCH_ELTYPE:-Float32}
export BENCH_SAMPLES=${BENCH_SAMPLES:-5}
export BENCH_SOLVERS=${BENCH_SOLVERS-cg}
case "$BENCH_SOLVERS" in
    cg|bicgstab|cg,bicgstab|bicgstab,cg) ;;
    *) echo "BENCH_SOLVERS must be cg, bicgstab, or a comma-separated list without duplicates" >&2; exit 2 ;;
esac
IFS=, read -r -a solvers <<< "$BENCH_SOLVERS"
export BENCH_LOCAL=${BENCH_LOCAL-0}
[[ $BENCH_LOCAL == 0 || $BENCH_LOCAL == 1 ]] || { echo "BENCH_LOCAL must be 0 or 1" >&2; exit 2; }
[[ $BENCH_ELTYPE == Float32 || $BENCH_ELTYPE == Float64 ]] || { echo "BENCH_ELTYPE must be Float32 or Float64" >&2; exit 2; }
export LEGATE_AUTO_CONFIG=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

usage() {
    echo "Usage: $0 single N [N ...] | weak BASE_N GPU_COUNT [GPU_COUNT ...]" >&2
    echo "Select solvers with BENCH_SOLVERS=cg, bicgstab, or cg,bicgstab (default: cg)." >&2
    echo "Set BENCH_LOCAL=1 to also run the cuNumeric local implementations (default: 0)." >&2
    exit 2
}
(( $# >= 2 )) || usage
experiment=$1; shift
[[ $experiment == single || $experiment == weak ]] || usage
if [[ $experiment == weak ]]; then (( $# >= 2 )) || usage; base_n=$1; shift; fi
for arg in "$@" "${base_n:-2}"; do
    [[ $arg =~ ^[0-9]+$ ]] && (( arg > 0 )) || usage
done

output=${BENCH_OUTPUT:-"$PWD/krylov-$experiment-$(date +%Y%m%d-%H%M%S)-$$"}
mkdir -p "$output"
csv="$output/results.csv"
echo 'experiment,base_n,backend,solver,mode,eltype,gpus,n,iterations,mean_ms,stderr_ms,median_ms,min_ms,max_ms,relative_residual,samples_ms' > "$csv"
echo 'backend,solver,gpus,n,peak_gpu_memory_mib' > "$output/memory.csv"
echo 'backend,solver,mode,gpus,n,legate_config,cuda_visible_devices' > "$output/planned-cases.csv"
{
    printf 'benchmark_commit=%s\n' "$(git -C "$script_dir" rev-parse HEAD)"
    printf 'cunumeric_commit=%s\n' "$(git -C "${CUNUMERIC_SOURCE:-/opt/cuNumeric.jl}" rev-parse HEAD)"
    "$julia_bin" --version
    nvidia-smi
    printf 'CUBLAS_WORKSPACE_CONFIG=%s\nBENCH_ELTYPE=%s\nBENCH_SOLVERS=%s\nBENCH_LOCAL=%s\nBENCH_SAMPLES=%s\nBENCH_CPUS=%s\nBENCH_TIMEOUT=%s\nLEGATE_AUTO_CONFIG=%s\n' \
        "${CUBLAS_WORKSPACE_CONFIG:-<default>}" "$BENCH_ELTYPE" "$BENCH_SOLVERS" "$BENCH_LOCAL" "$BENCH_SAMPLES" "${BENCH_CPUS:-2}" \
        "${BENCH_TIMEOUT:-15m}" "$LEGATE_AUTO_CONFIG"
} > "$output/environment.txt" 2>&1
cp "$project/Manifest.toml" "$output/Manifest.toml"
[[ ! -f "$project/LocalPreferences.toml" ]] || cp "$project/LocalPreferences.toml" "$output/LocalPreferences.toml"

failed=0
run_case() {
    local backend=$1 solver=$2 mode=$3 n=$4 gpus=$5
    local log="$output/$BENCH_ELTYPE-$backend-$solver-$mode-$gpus-$n.log"
    local gpu_mask
    gpu_mask=$(gpu_mask_for_count "$gpus")
    export BENCH_GPUS=$gpus
    export LEGATE_CONFIG="--gpus $gpus --cpus ${BENCH_CPUS:-2}"
    printf '%s,%s,%s,%s,%s,%s,"%s"\n' "$backend" "$solver" "$mode" "$gpus" "$n" "$LEGATE_CONFIG" "$gpu_mask" >> "$output/planned-cases.csv"
    echo "Running $backend $solver $mode: G=$gpus N=$n"
    [[ ${BENCH_DRY_RUN:-0} != 1 ]] || return 0
    local memory_log="$output/gpu-memory-$BENCH_ELTYPE-$backend-$solver-$mode-$gpus-$n.log"
    local result peak
    if result=$(CUDA_VISIBLE_DEVICES="$gpu_mask" run_logged_case "$log" "$memory_log" \
        timeout --signal=TERM --kill-after=30s "${BENCH_TIMEOUT:-15m}" \
        "$julia_bin" -t"${BENCH_THREADS:-4}" --startup-file=no --project="$project" \
        "$script_dir/krylov.jl" "$backend" "$solver" "$mode" "$n"); then
        printf '%s,%s,%s\n' "$experiment" "${base_n:-}" "$result" >> "$csv"
    else
        echo "Failed: $backend $solver $mode, G=$gpus N=$n ($log)" >&2
        failed=1
        size_failed=1
    fi
    peak=$(read_peak_memory "$memory_log") || failed=1
    printf '%s,%s,%s,%s,%s\n' "$backend-$mode" "$solver" "$gpus" "$n" "$peak" >> "$output/memory.csv"
}

for value in "$@"; do
    if [[ $experiment == single ]]; then
        gpus=1; n=$value
    else
        gpus=$value
        n=$(awk -v base="$base_n" -v g="$gpus" 'BEGIN { printf "%.0f", base * sqrt(g) }')
    fi
    (( n > 1 )) || usage
    for solver in "${solvers[@]}"; do
        size_failed=0
        if [[ $experiment == single ]]; then
            run_case CuArray "$solver" stock "$n" "$gpus"
        fi
        run_case Dagger "$solver" stock "$n" "$gpus"
        run_case cuNumeric "$solver" stock "$n" "$gpus"
        if [[ $BENCH_LOCAL == 1 ]]; then
            run_case cuNumeric "$solver" local "$n" "$gpus"
        fi
        if [[ $experiment == single ]]; then
            if [[ $size_failed -ne 0 ]]; then
                echo "Some backends failed for $solver at N=$n; keeping successful results and continuing the sweep" >&2
            elif [[ ${BENCH_DRY_RUN:-0} != 1 ]]; then
                printf '%s\n' "$n" > "$output/base_n-$solver.txt"
                if [[ ${#solvers[@]} == 1 ]]; then
                    printf '%s\n' "$n" > "$output/base_n.txt"
                fi
            fi
        fi
    done
done
echo "Results: $csv"
for solver in "${solvers[@]}"; do
    # A solver with no successful cases must not prevent the other plot.
    if ! awk -F, -v solver="$solver" 'NR > 1 && $4 == solver { found=1 } END { exit !found }' "$csv"; then
        continue
    fi
    image="$output/timings-$solver.png"
    [[ ${#solvers[@]} != 1 ]] || image="$output/timings.png"
    if ! GKSwstype=100 "$julia_bin" --startup-file=no --project="$project" \
        "$script_dir/plot_results.jl" "$experiment" "$csv" --solver="$solver" --output "$image"; then
        echo "Plot generation failed for $solver; benchmark data is saved in $csv. See the Julia error above." >&2
        failed=1
    fi
done
exit "$failed"
