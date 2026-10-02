#!/usr/bin/env bash
# Usage: run.sh single N [N ...] | weak BASE_N GPUS [GPUS ...]
# BENCH_SOLVERS=cg|bicgstab|cg,bicgstab (default cg); BENCH_LOCAL=1 adds cuNumeric local.
# BENCH_BACKENDS picks backends (default CuArray Dagger cuNumeric; CuArray only for single).
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../common.sh"
julia_bin=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
project=${BENCH_PROJECT:-"$script_dir/../../environments/composability"}
export BENCH_ELTYPE=${BENCH_ELTYPE:-Float32} BENCH_SAMPLES=${BENCH_SAMPLES:-5}
export BENCH_SOLVERS=${BENCH_SOLVERS-cg} BENCH_LOCAL=${BENCH_LOCAL-0}
export LEGATE_AUTO_CONFIG=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
case $BENCH_SOLVERS in
    cg|bicgstab|cg,bicgstab|bicgstab,cg) ;;
    *) echo "BENCH_SOLVERS must be cg, bicgstab, or cg,bicgstab" >&2; exit 2 ;;
esac
[[ $BENCH_LOCAL == [01] ]] || { echo "BENCH_LOCAL must be 0 or 1" >&2; exit 2; }
IFS=, read -r -a solvers <<< "$BENCH_SOLVERS"
read -r -a backends <<< "${BENCH_BACKENDS:-CuArray Dagger cuNumeric}"

experiment=${1:?Usage: run.sh single N... | weak BASE_N GPUS...}; shift
[[ $experiment == weak ]] && { base_n=$1; shift; }
output=${BENCH_OUTPUT:-"$PWD/krylov-$experiment-$(date +%Y%m%d-%H%M%S)-$$"}
mkdir -p "$output"
csv=$output/results.csv
echo 'experiment,base_n,backend,solver,mode,eltype,gpus,n,iterations,mean_ms,stderr_ms,median_ms,min_ms,max_ms,relative_residual,samples_ms' > "$csv"
echo 'backend,solver,gpus,n,peak_gpu_memory_mib' > "$output/memory.csv"
echo 'backend,solver,mode,gpus,n,legate_config,cuda_visible_devices' > "$output/planned-cases.csv"
{
    echo "benchmark_commit=$(git -C "$script_dir" rev-parse HEAD)"
    echo "cunumeric_commit=$(git -C "${CUNUMERIC_SOURCE:-/opt/cuNumeric.jl}" rev-parse HEAD)"
    "$julia_bin" --version
    nvidia-smi
    echo "sample_isolation=process warmups_per_sample=1 timeout=none"
    env | grep -E '^(BENCH_|LEGATE_|CUBLAS_)' | sort
} > "$output/environment.txt" 2>&1
cp "$project/Manifest.toml" "$output/"
[[ ! -f $project/LocalPreferences.toml ]] || cp "$project/LocalPreferences.toml" "$output/"

failed=0
run_case() {  # backend solver mode n gpus
    local name=$BENCH_ELTYPE-$1-$2-$3-$5-$4 mask result peak
    mask=$(gpu_mask_for_count "$5")
    export BENCH_GPUS=$5 LEGATE_CONFIG="--gpus $5 --cpus ${BENCH_CPUS:-2}"
    echo "$1,$2,$3,$5,$4,$LEGATE_CONFIG,\"$mask\"" >> "$output/planned-cases.csv"
    echo "Running $1 $2 $3: G=$5 N=$4"
    [[ ${BENCH_DRY_RUN:-0} != 1 ]] || return 0
    # Each fresh process rebuilds the dense input and warms up before timing.
    # Like ODE, allow all samples to finish without a case-wide time limit.
    if result=$(CUDA_VISIBLE_DEVICES=$mask run_logged_case "$output/$name.log" "$output/gpu-memory-$name.log" \
        "$julia_bin" -t"${BENCH_THREADS:-4}" --startup-file=no --project="$project" \
        "$script_dir/run_samples.jl" "$output/$name.log" "$1" "$2" "$3" "$4"); then
        echo "$experiment,${base_n:-},$result" >> "$csv"
    else
        echo "Failed: $1 $2 $3 G=$5 N=$4; continuing the sweep" >&2
        failed=1
    fi
    peak=$(read_peak_memory "$output/gpu-memory-$name.log") || failed=1
    echo "$1-$3,$2,$5,$4,$peak" >> "$output/memory.csv"
}

for value in "$@"; do
    if [[ $experiment == single ]]; then gpus=1 n=$value
    else gpus=$value n=$(awk -v b="$base_n" -v g="$gpus" 'BEGIN { printf "%.0f", b * sqrt(g) }'); fi
    for solver in "${solvers[@]}"; do
        for backend in "${backends[@]}"; do
            [[ $backend == CuArray && $experiment != single ]] ||
                run_case "$backend" "$solver" stock "$n" "$gpus"
        done
        [[ $BENCH_LOCAL != 1 ]] || run_case cuNumeric "$solver" local "$n" "$gpus"
    done
done

echo "Results: $csv"
for solver in "${solvers[@]}"; do
    awk -F, -v s="$solver" 'NR > 1 && $4 == s { found=1 } END { exit !found }' "$csv" || continue
    image=$output/timings-$solver.png
    (( ${#solvers[@]} > 1 )) || image=$output/timings.png
    GKSwstype=100 "$julia_bin" --startup-file=no --project="$project" \
        "$script_dir/plot_results.jl" "$experiment" "$csv" --solver="$solver" --output "$image" ||
        { echo "Plot generation failed for $solver; data is in $csv" >&2; failed=1; }
done
exit "$failed"
