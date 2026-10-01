#!/usr/bin/env bash
# Usage: run_benchmark.sh single N [N ...] | weak BASE_N GPUS [GPUS ...]
# ODE_BACKENDS picks backends (default CuArray Dagger cuNumeric; CuArray only for single).
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../common.sh"
julia_bin=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
project=${ODE_PROJECT:-"$script_dir/../../environments/composability"}
export ODE_SAMPLES=${ODE_SAMPLES:-5}
export LEGATE_AUTO_CONFIG=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

experiment=${1:?Usage: run_benchmark.sh single N... | weak BASE_N GPUS...}; shift
[[ $experiment == weak ]] && { base_n=$1; shift; }
read -r -a backends <<< "${ODE_BACKENDS:-CuArray Dagger cuNumeric}"
output=${ODE_OUTPUT:-"$script_dir/results-$experiment-$(date +%Y%m%d-%H%M%S)-$$"}
mkdir -p "$output"
csv=$output/results.csv
echo 'experiment,base_n,backend,eltype,gpus,N,steps,mean_ms,stderr_ms,median_ms,min_ms,max_ms,relative_error,samples_ms' > "$csv"
echo 'backend,gpus,N,peak_gpu_memory_mib' > "$output/memory.csv"
echo 'backend,gpus,N,legate_config,cuda_visible_devices' > "$output/planned-cases.csv"
{
    echo "benchmark_commit=$(git -C "$script_dir/../.." rev-parse HEAD)"
    echo "cunumeric_commit=$(git -C "${CUNUMERIC_SOURCE:-/opt/cuNumeric.jl}" rev-parse HEAD)"
    echo "julia=$("$julia_bin" --version)"
    echo "experiment=$experiment base_n=${base_n:-} backends=${backends[*]} values=$*"
    echo "sample_isolation=process warmups_per_sample=1 timeout=none"
    env | grep -E '^(ODE_|LEGATE_|CUBLAS_)' | sort
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} > "$output/metadata.txt" 2>&1
cp "$project/Manifest.toml" "$output/"
[[ ! -f $project/LocalPreferences.toml ]] || cp "$project/LocalPreferences.toml" "$output/"

status=0
for value in "$@"; do
    if [[ $experiment == single ]]; then gpus=1 n=$value
    else gpus=$value n=$(awk -v b="$base_n" -v g="$gpus" 'BEGIN { printf "%.0f", b * sqrt(g) }'); fi
    mask=$(gpu_mask_for_count "$gpus")
    export ODE_GPUS=$gpus LEGATE_CONFIG="--gpus $gpus --cpus ${ODE_CPUS:-4}"
    for backend in "${backends[@]}"; do
        [[ $backend != CuArray || $experiment == single ]] || continue
        echo "$backend,$gpus,$n,$LEGATE_CONFIG,\"$mask\"" >> "$output/planned-cases.csv"
        echo "Running $backend G=$gpus N=$n"
        [[ ${ODE_DRY_RUN:-0} != 1 ]] || continue
        name=$backend-$gpus-$n
        # One fresh Julia process per sample; no time limit.
        if result=$(CUDA_VISIBLE_DEVICES=$mask run_logged_case "$output/$name.log" "$output/gpu-memory-$name.log" \
            "$julia_bin" --startup-file=no --project="$project" \
            "$script_dir/run_samples.jl" "$output/$name.log" "$backend" "$n"); then
            echo "$experiment,${base_n:-},$result" >> "$csv"
        else
            echo "Failed: $backend G=$gpus N=$n; continuing the sweep" >&2
            status=1
        fi
        peak=$(read_peak_memory "$output/gpu-memory-$name.log") || status=1
        echo "$backend,$gpus,$n,$peak" >> "$output/memory.csv"
    done
done

echo "Results: $csv"
if (( $(wc -l < "$csv") > 1 )); then
    GKSwstype=100 "$julia_bin" --startup-file=no --project="$project" \
        "$script_dir/plot_results.jl" "$experiment" "$csv" "$output/timings.png" ||
        { echo "Plot generation failed; data is in $csv" >&2; status=1; }
fi
exit "$status"
