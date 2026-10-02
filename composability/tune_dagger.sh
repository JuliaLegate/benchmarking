#!/usr/bin/env bash
# ./composability/tune_dagger.sh [--dry-run] [workload ...]
# Workloads: krylov, ordinarydiffeq, integrals_optimization (default: all).
# Small Float32 problems; each candidate gets one warmup + two timed runs,
# each capped at five iterations/steps by COMPOSABILITY_TUNE=1.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
source common.sh || exit 1
JULIA=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
CPUS=8
GPUS=(1 2 4 8)
BLOCKS=(1 2 4 8 16 32 64)
DRY=0
[[ ${1:-} != --dry-run ]] || { DRY=1; shift; }
ONLY=("$@")
FAILED=()
for name in "${ONLY[@]}"; do
    case $name in krylov|ordinarydiffeq|integrals_optimization) ;;
        *) echo "Unknown workload: $name" >&2; exit 2 ;;
    esac
done
OUTPUT=${DAGGER_TUNE_OUTPUT:-"$PWD/tunes/$(date +%Y%m%d-%H%M%S)-$$"}
if (( ! DRY )); then
    mkdir -p "$(dirname "$OUTPUT")" && mkdir "$OUTPUT" || exit 1
    echo 'name,gpus,n,blocks_per_gpu,mean_ms' | tee "$OUTPUT/results.csv" > "$OUTPUT/best.csv"
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# tune <gpus> <workload> <label> <N> <env prefix> <worker> <mean column> [worker args]
tune() {
    local gpus=$1 workload=$2 name=$3 n=$4 prefix=$5 worker=$6 column=$7
    shift 7
    [[ ${#ONLY[@]} -gt 0 && ! " ${ONLY[*]} " =~ " $workload " ]] && return
    local mask blocks log result ms best= winner= row
    mask=$(gpu_mask_for_count "$gpus") || { FAILED+=("$name G=$gpus (GPU mask)"); return; }
    for blocks in "${BLOCKS[@]}"; do
        echo "==> $name G=$gpus N=$n blocks_per_gpu=$blocks"
        local -a cmd=(env "CUDA_VISIBLE_DEVICES=$mask" "DAGGER_BLOCKS_PER_GPU=$blocks"
            COMPOSABILITY_TUNE=1 "${prefix}_GPUS=$gpus" "${prefix}_ELTYPE=Float32"
            "$JULIA" --startup-file=no --project=../environments/composability --threads=$CPUS
            "$worker" Dagger "$@" "$n")
        if (( DRY )); then printf '  %q' "${cmd[@]}"; printf '\n'; continue; fi
        log=$OUTPUT/$name-g$gpus-b$blocks.log
        if result=$(run_logged_case "$log" "${cmd[@]}") &&
           ms=$(awk -F, -v c="$column" '{
               if ($c !~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ || $c+0 <= 0) exit 1
               print $c
           }' <<< "$result"); then
            row="$name,$gpus,$n,$blocks,$ms"
            echo "$row" >> "$OUTPUT/results.csv"
            if [[ -z $best ]] || awk -v a="$ms" -v b="$best" 'BEGIN { exit !(a < b) }'; then
                best=$ms winner=$row
            fi
            echo "    $ms ms/run"
            # Same stop rule as the root tuner.
            awk -v a="$ms" -v b="$best" 'BEGIN { exit !(a > 4*b) }' && break
        else
            FAILED+=("$name G=$gpus blocks=$blocks ($log)")
        fi
    done
    if [[ -n $winner ]]; then
        echo "$winner" >> "$OUTPUT/best.csv"
        echo "best (name,gpus,n,blocks_per_gpu,mean_ms): $winner"
    fi
}

for g in "${GPUS[@]}"; do
    tune "$g" krylov krylov-cg 4096 BENCH krylov/krylov.jl 8 cg stock
    tune "$g" krylov krylov-bicgstab 4096 BENCH krylov/krylov.jl 8 bicgstab stock
    tune "$g" ordinarydiffeq heat 1024 ODE ordinarydiffeq/benchmark_heat.jl 6
    tune "$g" integrals_optimization plume 512 INTOPT integrals_optimization/benchmark.jl 9
done
if (( ${#FAILED[@]} )); then
    echo 'Failed tunes:'; printf '  %s\n' "${FAILED[@]}"; exit 1
fi
(( DRY )) || echo "All tunes finished; timings and winners in $OUTPUT/"
exit 0
