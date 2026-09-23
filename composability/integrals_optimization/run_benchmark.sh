#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: INTOPT_PROJECT=/path/to/env bash composability/integrals_optimization/run_benchmark.sh N [N ...]" >&2
    exit 2
fi
: "${INTOPT_PROJECT:?Set INTOPT_PROJECT to the environment created by setup.jl}"

# Match CUDA.jl's single-GPU run by default. Callers can override either setting.
export LEGATE_AUTO_CONFIG="${LEGATE_AUTO_CONFIG:-0}"
export LEGATE_CONFIG="${LEGATE_CONFIG:---gpus 1 --cpus 4}"

for n in "$@"; do
    if ! [[ $n =~ ^[1-9][0-9]*$ ]] || (( 10#$n < 4 )); then
        echo "Each N must be an integer at least 4; got '$n'" >&2
        exit 2
    fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
julia_bin="${JULIA:-julia}"
output="${INTOPT_OUTPUT:-$script_dir/results-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$output"
csv="$output/results.csv"
printf 'backend,eltype,N,bands,order,iters,median_ms,min_ms,max_ms,initial_loss,final_loss,relative_error,samples_ms\n' > "$csv"

read -r -a backends <<< "${INTOPT_BACKENDS:-CuArray cuNumeric}"
if [[ ${#backends[@]} -eq 0 ]]; then
    echo "INTOPT_BACKENDS must name at least one backend" >&2
    exit 2
fi
for backend in "${backends[@]}"; do
    case "$backend" in
        cpu|CuArray|cuNumeric) ;;
        *) echo "Unknown backend '$backend'" >&2; exit 2 ;;
    esac
done

{
    printf 'git_commit=%s\n' "$(git -C "$script_dir/../.." rev-parse HEAD)"
    printf 'julia=%s\n' "$("$julia_bin" --version)"
    printf 'eltype=%s\nbands=%s\norder=%s\niters=%s\nsamples=%s\nrate=%s\nnoise=%s\n' \
        "${INTOPT_ELTYPE:-Float32}" "${INTOPT_BANDS:-4}" "${INTOPT_ORDER:-12}" \
        "${INTOPT_ITERS:-40}" "${INTOPT_SAMPLES:-3}" \
        "${INTOPT_RATE:-0.05}" "${INTOPT_NOISE:-0.001}"
    printf 'CUBLAS_WORKSPACE_CONFIG=%s\n' "${CUBLAS_WORKSPACE_CONFIG:-<default>}"
    printf 'LEGATE_AUTO_CONFIG=%s\n' "$LEGATE_AUTO_CONFIG"
    printf 'LEGATE_CONFIG=%s\n' "$LEGATE_CONFIG"
    printf 'backends=%s\nsizes=%s\n' "${backends[*]}" "$*"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    fi
} > "$output/metadata.txt"

status=0
for backend in "${backends[@]}"; do
    for n in "$@"; do
        log="$output/${backend}-${n}.log"
        echo "Running $backend N=$n"
        if "$julia_bin" --startup-file=no --project="$INTOPT_PROJECT" \
                "$script_dir/benchmark.jl" "$backend" "$n" > "$log" 2>&1; then
            line="$(grep '^RESULT,' "$log" | tail -n 1 || true)"
            if [[ -n $line ]]; then
                printf '%s\n' "${line#RESULT,}" >> "$csv"
            else
                echo "No RESULT row in $log" >&2
                status=1
            fi
        else
            echo "$backend N=$n failed; see $log" >&2
            status=1
        fi
    done
done

if [[ $(wc -l < "$csv") -gt 1 ]]; then
    if ! GKSwstype=100 "$julia_bin" --startup-file=no --project="$INTOPT_PROJECT" \
            "$script_dir/plot_results.jl" "$csv" "$output/timings.png"; then
        echo "Plot generation failed" >&2
        status=1
    fi
fi

echo "Results: $csv"
[[ -f "$output/timings.png" ]] && echo "Plot: $output/timings.png"
exit "$status"
