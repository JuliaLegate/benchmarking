#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../common.sh"
ODE_PROJECT=${ODE_PROJECT:-"$script_dir/../../environments/composability"}

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
output=${ODE_OUTPUT:-"$script_dir/results-$experiment-$(date +%Y%m%d-%H%M%S)-$$"}
mkdir -p "$output"
csv="$output/results.csv"
echo 'experiment,base_n,backend,eltype,gpus,N,steps,mean_ms,stderr_ms,median_ms,min_ms,max_ms,relative_error,samples_ms' > "$csv"
echo 'backend,gpus,N,peak_gpu_memory_mib' > "$output/memory.csv"
echo 'backend,gpus,N,legate_config,cuda_visible_devices' > "$output/planned-cases.csv"
export ODE_SAMPLES=${ODE_SAMPLES:-5}
export LEGATE_AUTO_CONFIG=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

if [[ $experiment == single ]]; then
    backends=(CuArray Dagger cuNumeric)
else
    backends=(Dagger cuNumeric)
fi
if [[ -n ${ODE_BACKENDS:-} ]]; then
    read -r -a backends <<< "$ODE_BACKENDS"
fi
[[ ${#backends[@]} -gt 0 ]] || usage
for backend in "${backends[@]}"; do
    case "$backend" in
        CuArray|Dagger|cuNumeric) ;;
        *) echo "Unknown GPU backend '$backend'" >&2; exit 2 ;;
    esac
    [[ $experiment == single || $backend != CuArray ]] || { echo "CuArray is single GPU only" >&2; exit 2; }
done

{
    printf 'benchmark_commit=%s\n' "$(git -C "$script_dir/../.." rev-parse HEAD)"
    printf 'cunumeric_commit=%s\n' "$(git -C "${CUNUMERIC_SOURCE:-/opt/cuNumeric.jl}" rev-parse HEAD)"
    printf 'julia=%s\n' "$("$julia_bin" --version)"
    printf 'experiment=%s\nbase_n=%s\nbackends=%s\nvalues=%s\n' "$experiment" "${base_n:-}" "${backends[*]}" "$*"
    printf 'eltype=%s\nsteps=%s\nsamples=%s\nsample_isolation=process\nwarmups_per_sample=1\ntimeout=none\n' "${ODE_ELTYPE:-Float32}" "${ODE_STEPS:-20}" "$ODE_SAMPLES"
    printf 'CUBLAS_WORKSPACE_CONFIG=%s\nLEGATE_AUTO_CONFIG=%s\n' \
        "${CUBLAS_WORKSPACE_CONFIG:-<default>}" "$LEGATE_AUTO_CONFIG"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} > "$output/metadata.txt" 2>&1
cp "$ODE_PROJECT/Manifest.toml" "$output/Manifest.toml"
[[ ! -f "$ODE_PROJECT/LocalPreferences.toml" ]] || cp "$ODE_PROJECT/LocalPreferences.toml" "$output/LocalPreferences.toml"

status=0
for value in "$@"; do
    size_failed=0
    if [[ $experiment == single ]]; then
        gpus=1; n=$value
    else
        gpus=$value
        n=$(awk -v base="$base_n" -v g="$gpus" 'BEGIN { printf "%.0f", base * sqrt(g) }')
    fi
    export ODE_GPUS=$gpus
    export LEGATE_CONFIG="--gpus $gpus --cpus ${ODE_CPUS:-4}"
    gpu_mask=$(gpu_mask_for_count "$gpus")
    for backend in "${backends[@]}"; do
        printf '%s,%s,%s,%s,"%s"\n' "$backend" "$gpus" "$n" "$LEGATE_CONFIG" "$gpu_mask" >> "$output/planned-cases.csv"
        if [[ ${ODE_DRY_RUN:-0} == 1 ]]; then
            echo "Planned $backend G=$gpus N=$n"
            continue
        fi
        log="$output/$backend-$gpus-$n.log"
        echo "Running $backend G=$gpus N=$n"
        memory_log="$output/gpu-memory-$backend-$gpus-$n.log"
        # The CPU-only coordinator waits for each fresh sample process to exit.
        # No time limit: a case runs until completion, failure, or cancellation.
        if result=$(CUDA_VISIBLE_DEVICES="$gpu_mask" run_logged_case "$log" "$memory_log" \
            "$julia_bin" --startup-file=no --project="$ODE_PROJECT" \
            "$script_dir/run_samples.jl" "$log" "$backend" "$n"); then
            printf '%s,%s,%s\n' "$experiment" "${base_n:-}" "$result" >> "$csv"
        else
            echo "$backend G=$gpus N=$n failed; continuing the sweep" >&2
            status=1; size_failed=1
        fi
        peak=$(read_peak_memory "$memory_log") || status=1
        printf '%s,%s,%s,%s\n' "$backend" "$gpus" "$n" "$peak" >> "$output/memory.csv"
    done
    if [[ $experiment == single ]]; then
        if [[ $size_failed -ne 0 ]]; then
            echo "Some backends failed at N=$n; keeping successful results and continuing the sweep" >&2
        elif [[ ${ODE_DRY_RUN:-0} != 1 && "${backends[*]}" == 'CuArray Dagger cuNumeric' ]]; then
            printf '%s\n' "$n" > "$output/base_n.txt"
        fi
    fi
done

if [[ $(wc -l < "$csv") -gt 1 ]]; then
    if ! GKSwstype=100 "$julia_bin" --startup-file=no --project="$ODE_PROJECT" \
        "$script_dir/plot_results.jl" "$experiment" "$csv" "$output/timings.png"; then
        echo "Plot generation failed" >&2; status=1
    fi
fi
echo "Results: $csv"
[[ ! -f "$output/timings.png" ]] || echo "Plot: $output/timings.png"
exit "$status"
