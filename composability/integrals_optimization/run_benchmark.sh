#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../common.sh"
INTOPT_PROJECT=${INTOPT_PROJECT:-"$script_dir/../../environments/composability"}

usage() {
    echo "Usage: $0 single N [N ...] | weak BASE_N GPU_COUNT [GPU_COUNT ...]" >&2
    exit 2
}
[[ $# -ge 2 ]] || usage
experiment=$1; shift
[[ $experiment == single || $experiment == weak ]] || usage
if [[ $experiment == weak ]]; then
    [[ $# -ge 2 ]] || usage
    base_n=$1; shift
    [[ $base_n =~ ^[1-9][0-9]*$ ]] && (( base_n >= 4 )) || usage
fi
for value in "$@"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || usage
    if [[ $experiment == single ]]; then
        (( value >= 4 )) || usage
    else
        [[ $value == 1 || $value == 2 || $value == 4 || $value == 8 ]] || usage
    fi
done

julia_bin=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
output=${INTOPT_OUTPUT:-"$script_dir/results-$experiment-$(date +%Y%m%d-%H%M%S)-$$"}
mkdir -p "$output"
csv="$output/results.csv"
echo 'experiment,base_n,backend,eltype,gpus,N,bands,order,maxiters,objective_evals,mean_ms,stderr_ms,median_ms,min_ms,max_ms,initial_loss,final_loss,parameter_error,samples_ms' > "$csv"
echo 'backend,gpus,N,legate_config,cuda_visible_devices' > "$output/planned-cases.csv"
export INTOPT_SAMPLES=${INTOPT_SAMPLES:-5}
export LEGATE_AUTO_CONFIG=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

if [[ $experiment == single ]]; then
    backends=(CuArray Dagger cuNumeric)
else
    backends=(Dagger cuNumeric)
fi
if [[ -n ${INTOPT_BACKENDS:-} ]]; then
    read -r -a backends <<< "$INTOPT_BACKENDS"
fi
[[ ${#backends[@]} -gt 0 ]] || usage
for backend in "${backends[@]}"; do
    case "$backend" in
        cpu|CuArray|Dagger|cuNumeric) ;;
        *) echo "Unknown backend '$backend'" >&2; exit 2 ;;
    esac
    [[ $experiment == single || ( $backend != CuArray && $backend != cpu ) ]] || {
        echo "$backend is single GPU only" >&2; exit 2;
    }
done

{
    printf 'benchmark_commit=%s\n' "$(git -C "$script_dir/../.." rev-parse HEAD)"
    printf 'cunumeric_commit=%s\n' "$(git -C "${CUNUMERIC_SOURCE:-/opt/cuNumeric.jl}" rev-parse HEAD)"
    printf 'julia=%s\n' "$("$julia_bin" --version)"
    printf 'experiment=%s\nbase_n=%s\nbackends=%s\nvalues=%s\n' "$experiment" "${base_n:-}" "${backends[*]}" "$*"
    printf 'eltype=%s\nbands=%s\norder=%s\nmaxiters=%s\nsamples=%s\nnoise=%s\ntimeout=%s\n' \
        "${INTOPT_ELTYPE:-Float32}" "${INTOPT_BANDS:-4}" "${INTOPT_ORDER:-12}" \
        "${INTOPT_ITERS:-80}" "$INTOPT_SAMPLES" "${INTOPT_NOISE:-0.001}" "${INTOPT_TIMEOUT:-15m}"
    printf 'CUBLAS_WORKSPACE_CONFIG=%s\nLEGATE_AUTO_CONFIG=%s\n' \
        "${CUBLAS_WORKSPACE_CONFIG:-<default>}" "$LEGATE_AUTO_CONFIG"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} > "$output/metadata.txt" 2>&1
cp "$INTOPT_PROJECT/Manifest.toml" "$output/Manifest.toml"
[[ ! -f "$INTOPT_PROJECT/LocalPreferences.toml" ]] || cp "$INTOPT_PROJECT/LocalPreferences.toml" "$output/LocalPreferences.toml"

status=0
for value in "$@"; do
    size_failed=0
    if [[ $experiment == single ]]; then
        gpus=1; n=$value
    else
        gpus=$value
        n=$(awk -v base="$base_n" -v g="$gpus" 'BEGIN { printf "%.0f", base * sqrt(g) }')
    fi
    export INTOPT_GPUS=$gpus
    export LEGATE_CONFIG="--gpus $gpus --cpus ${INTOPT_CPUS:-4}"
    gpu_mask=$(gpu_mask_for_count "$gpus")
    for backend in "${backends[@]}"; do
        printf '%s,%s,%s,%s,"%s"\n' "$backend" "$gpus" "$n" "$LEGATE_CONFIG" "$gpu_mask" >> "$output/planned-cases.csv"
        if [[ ${INTOPT_DRY_RUN:-0} == 1 ]]; then
            echo "Planned $backend G=$gpus N=$n"
            continue
        fi
        log="$output/$backend-$gpus-$n.log"
        echo "Running $backend G=$gpus N=$n"
        if result=$(CUDA_VISIBLE_DEVICES="$gpu_mask" run_logged_case "$log" \
            timeout --signal=TERM --kill-after=30s "${INTOPT_TIMEOUT:-15m}" \
            "$julia_bin" --startup-file=no --project="$INTOPT_PROJECT" \
            "$script_dir/benchmark.jl" "$backend" "$n"); then
            printf '%s,%s,%s\n' "$experiment" "${base_n:-}" "$result" >> "$csv"
        else
            echo "$backend G=$gpus N=$n failed; see $log" >&2; status=1; size_failed=1
        fi
    done
    if [[ $experiment == single ]]; then
        if [[ $size_failed -ne 0 ]]; then
            echo "Some backends failed at N=$n; keeping successful results and continuing the sweep" >&2
        elif [[ ${INTOPT_DRY_RUN:-0} != 1 && "${backends[*]}" == 'CuArray Dagger cuNumeric' ]]; then
            printf '%s\n' "$n" > "$output/base_n.txt"
        fi
    fi
done

if [[ $(wc -l < "$csv") -gt 1 ]]; then
    if ! GKSwstype=100 "$julia_bin" --startup-file=no --project="$INTOPT_PROJECT" \
        "$script_dir/plot_results.jl" "$experiment" "$csv" "$output/timings.png"; then
        echo "Plot generation failed" >&2; status=1
    fi
fi
echo "Results: $csv"
[[ ! -f "$output/timings.png" ]] || echo "Plot: $output/timings.png"
exit "$status"
