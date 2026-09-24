#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
julia_bin=${JULIA:-julia}
project=${BENCH_PROJECT:-$script_dir}
export BENCH_ELTYPE=${BENCH_ELTYPE:-Float32}
export BENCH_SAMPLES=${BENCH_SAMPLES:-5}
[[ $BENCH_ELTYPE == Float32 || $BENCH_ELTYPE == Float64 ]] || { echo "BENCH_ELTYPE must be Float32 or Float64" >&2; exit 2; }
export LEGATE_AUTO_CONFIG=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

usage() {
    echo "Usage: $0 single N [N ...] | weak BASE_N GPU_COUNT [GPU_COUNT ...]" >&2
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
{
    printf 'benchmark_commit=%s\n' "$(git -C "$script_dir" rev-parse HEAD)"
    printf 'cunumeric_commit=%s\n' "$(git -C "${CUNUMERIC_SOURCE:-/opt/cuNumeric.jl}" rev-parse HEAD)"
    "$julia_bin" --version
    nvidia-smi
    printf 'CUBLAS_WORKSPACE_CONFIG=%s\nBENCH_ELTYPE=%s\nBENCH_SOLVERS=cg\nBENCH_SAMPLES=%s\nBENCH_CPUS=%s\nBENCH_FBMEM=%s\nBENCH_SYSMEM=%s\nBENCH_ZCMEM=%s\nBENCH_TIMEOUT=%s\n' \
        "${CUBLAS_WORKSPACE_CONFIG:-<default>}" "$BENCH_ELTYPE" "$BENCH_SAMPLES" "${BENCH_CPUS:-2}" \
        "${BENCH_FBMEM:-61440}" "${BENCH_SYSMEM:-65536}" "${BENCH_ZCMEM:-1024}" "${BENCH_TIMEOUT:-15m}"
    "$julia_bin" --startup-file=no --project="$project" -e 'using Pkg; Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)'
} > "$output/environment.txt" 2>&1
cp "$project/Manifest.toml" "$output/Manifest.toml"
[[ ! -f "$project/LocalPreferences.toml" ]] || cp "$project/LocalPreferences.toml" "$output/LocalPreferences.toml"

failed=0
run_case() {
    local backend=$1 solver=$2 mode=$3 n=$4 gpus=$5
    local -a time_limit=()
    time_limit=(timeout --signal=TERM --kill-after=30s "${BENCH_TIMEOUT:-15m}")
    local log="$output/$BENCH_ELTYPE-$backend-$solver-$mode-$gpus-$n.log"
    export BENCH_GPUS=$gpus
    export LEGATE_CONFIG="--gpus $gpus --cpus ${BENCH_CPUS:-2} --fbmem ${BENCH_FBMEM:-61440} --sysmem ${BENCH_SYSMEM:-65536} --zcmem ${BENCH_ZCMEM:-1024}"
    echo "Running $backend $solver $mode: G=$gpus N=$n"
    if "${time_limit[@]}" "$julia_bin" -t"${BENCH_THREADS:-4}" --startup-file=no --project="$project" \
        "$script_dir/krylov.jl" "$backend" "$solver" "$mode" "$n" > "$log" 2>&1; then
        if [[ $(grep -c '^RESULT,' "$log") == 1 ]]; then
            sed -n "s/^RESULT,/$experiment,${base_n:-},/p" "$log" >> "$csv"
        else
            echo "Missing or duplicate RESULT: $log" >&2
            failed=1
        fi
    else
        echo "Failed: $backend $solver $mode, G=$gpus N=$n ($log)" >&2
        failed=1
    fi
}

for value in "$@"; do
    if [[ $experiment == single ]]; then
        gpus=1; n=$value
    else
        gpus=$value
        n=$(awk -v base="$base_n" -v g="$gpus" 'BEGIN { printf "%.0f", base * sqrt(g) }')
    fi
    (( n > 1 )) || usage
    if [[ $experiment == single ]]; then
        run_case CuArray cg stock "$n" "$gpus"
    fi
    run_case Dagger cg stock "$n" "$gpus"
    run_case cuNumeric cg stock "$n" "$gpus"
    run_case cuNumeric cg local "$n" "$gpus"
done
echo "Results: $csv"
if [[ $(wc -l < "$csv") -gt 1 ]]; then
    python3 "$script_dir/plot.py" "$experiment" "$csv" --output "$output/timings.png" || failed=1
fi
exit "$failed"
