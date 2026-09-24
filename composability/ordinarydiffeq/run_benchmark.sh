#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: ODE_PROJECT=/path/to/env $0 single N [N ...] | weak BASE_N GPU_COUNT [GPU_COUNT ...]" >&2
    exit 2
}
[[ $# -ge 2 ]] || usage
: "${ODE_PROJECT:?Set ODE_PROJECT to the environment created by setup.jl}"
experiment=$1; shift
[[ $experiment == single || $experiment == weak ]] || usage
if [[ $experiment == weak ]]; then
    [[ $# -ge 2 ]] || usage
    base_n=$1; shift
    [[ $base_n =~ ^[1-9][0-9]*$ ]] && (( base_n >= 4 )) || usage
fi
for value in "$@"; do
    rows_before=$(wc -l < "$csv")
    [[ $value =~ ^[1-9][0-9]*$ ]] || usage
    if [[ $experiment == single ]]; then
        (( value >= 4 )) || usage
    else
        [[ $value == 1 || $value == 2 || $value == 4 || $value == 8 ]] || usage
    fi
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
julia_bin=${JULIA:-julia}
output=${ODE_OUTPUT:-"$script_dir/results-$experiment-$(date +%Y%m%d-%H%M%S)-$$"}
mkdir -p "$output"
csv="$output/results.csv"
echo 'experiment,base_n,backend,eltype,gpus,N,steps,mean_ms,stderr_ms,median_ms,min_ms,max_ms,relative_error,samples_ms' > "$csv"
echo 'backend,gpus,N,peak_gpu_memory_mib' > "$output/memory.csv"
echo 'backend,gpus,N,legate_config' > "$output/planned-cases.csv"
export ODE_SAMPLES=${ODE_SAMPLES:-5}
export LEGATE_AUTO_CONFIG=${LEGATE_AUTO_CONFIG:-1}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

read -r -a backends <<< "${ODE_BACKENDS:-$( [[ $experiment == single ]] && echo 'CuArray Dagger cuNumeric' || echo 'Dagger cuNumeric' )}"
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
    printf 'eltype=%s\nsteps=%s\nsamples=%s\ntimeout=%s\nGPU_MEMORY_LIMIT_MIB=%s\n' "${ODE_ELTYPE:-Float32}" "${ODE_STEPS:-20}" "$ODE_SAMPLES" "${ODE_TIMEOUT:-15m}" "${GPU_MEMORY_LIMIT_MIB:-61440}"
    printf 'CUBLAS_WORKSPACE_CONFIG=%s\nLEGATE_AUTO_CONFIG=%s\n' "${CUBLAS_WORKSPACE_CONFIG:-<default>}" "$LEGATE_AUTO_CONFIG"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$julia_bin" --startup-file=no --project="$ODE_PROJECT" -e 'using Pkg; Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)'
} > "$output/metadata.txt" 2>&1
cp "$ODE_PROJECT/Manifest.toml" "$output/Manifest.toml"
[[ ! -f "$ODE_PROJECT/LocalPreferences.toml" ]] || cp "$ODE_PROJECT/LocalPreferences.toml" "$output/LocalPreferences.toml"

status=0
for value in "$@"; do
    if [[ $experiment == single ]]; then
        gpus=1; n=$value
    else
        gpus=$value
        n=$(awk -v base="$base_n" -v g="$gpus" 'BEGIN { printf "%.0f", base * sqrt(g) }')
    fi
    export ODE_GPUS=$gpus
    export LEGATE_CONFIG="--gpus $gpus --cpus ${ODE_CPUS:-4}"
    for backend in "${backends[@]}"; do
        printf '%s,%s,%s,%s\n' "$backend" "$gpus" "$n" "$LEGATE_CONFIG" >> "$output/planned-cases.csv"
        if [[ ${ODE_DRY_RUN:-0} == 1 ]]; then
            echo "Planned $backend G=$gpus N=$n"
            continue
        fi
        log="$output/$backend-$gpus-$n.log"
        echo "Running $backend G=$gpus N=$n"
        memory_log="$output/gpu-memory-$backend-$gpus-$n.log"
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop-ms=250 > "$memory_log" 2>&1 &
        monitor_pid=$!
        result_line=""
        if timeout --signal=TERM --kill-after=30s "${ODE_TIMEOUT:-15m}" \
            "$julia_bin" --startup-file=no --project="$ODE_PROJECT" \
            "$script_dir/benchmark_heat.jl" "$backend" "$n" > "$log" 2>&1; then
            line=$(grep '^RESULT,' "$log" | tail -n 1 || true)
            if [[ -n $line ]]; then
                result_line="$experiment,${base_n:-},${line#RESULT,}"
            else
                echo "No RESULT row in $log" >&2; status=1
            fi
        else
            echo "$backend G=$gpus N=$n failed; see $log" >&2; status=1
        fi
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
        peak=$(awk '$1 ~ /^[0-9]+$/ && $1 > peak { peak=$1 } END { print peak+0 }' "$memory_log")
        printf '%s,%s,%s,%s\n' "$backend" "$gpus" "$n" "$peak" >> "$output/memory.csv"
        if [[ $experiment == single && $peak -gt ${GPU_MEMORY_LIMIT_MIB:-61440} ]]; then
            echo "Memory limit exceeded for $backend G=$gpus N=$n: $peak MiB" >&2; status=1
        elif [[ -n $result_line ]]; then
            printf '%s\n' "$result_line" >> "$csv"
        fi
    done
    if [[ $experiment == single ]]; then
        if [[ $status -ne 0 ]]; then
            head -n "$rows_before" "$csv" > "$csv.tmp" && mv "$csv.tmp" "$csv"
            echo "Stopping size sweep at failed N=$n; see retained logs" >&2
            break
        fi
        if [[ ${ODE_DRY_RUN:-0} != 1 && "${backends[*]}" == 'CuArray Dagger cuNumeric' ]]; then
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
