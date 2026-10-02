#!/usr/bin/env bash
# Tune small, fixed-size composability problems on 1, 2, 4, and 8 GPUs.
# Like ../tune_dagger.sh, keep successful timings and continue after failures.
set -uo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh" || exit 2

usage() {
    cat <<'EOF'
Usage: bash composability/tune_dagger.sh [--dry-run] [workload ...]
Workloads: krylov ordinarydiffeq integrals_optimization (default: all)

DAGGER_TUNE_GPUS       GPU counts (default: "1 2 4 8")
DAGGER_TUNE_BLOCKS     Increasing block factors (default: "1 2 4 8 16 32 64")
DAGGER_TUNE_SAMPLES    Timed solves after warmup per candidate (default: 3)
DAGGER_TUNE_THREADS    Julia threads (default: 8)
DAGGER_TUNE_KRYLOV_N   Dense matrix dimension (default: 4096)
DAGGER_TUNE_ODE_N      Heat grid side (default: 1024)
DAGGER_TUNE_INTOPT_N   Plume grid side (default: 512)
DAGGER_TUNE_OUTPUT     New output directory (default: composability/tunes/<run-id>)
BENCH_SOLVERS         Krylov solvers: cg, bicgstab, cg,bicgstab (default: cg,bicgstab)
JULIA / CUNUMERIC_BENCH_JULIA override Julia. Existing workload settings
(e.g. ODE_STEPS, INTOPT_ITERS, *_ELTYPE) still apply. Requires the instantiated
environments/composability project. Stops a sweep after a candidate is >4x
slower than its best successful mean. --dry-run needs neither Julia nor GPUs.
EOF
}

dry=0
workloads=()
for arg in "$@"; do
    case $arg in
        --dry-run) dry=1 ;;
        -h|--help) usage; exit 0 ;;
        krylov|ordinarydiffeq|integrals_optimization) workloads+=("$arg") ;;
        *) echo "Unknown workload/option: $arg" >&2; usage >&2; exit 2 ;;
    esac
done
(( ${#workloads[@]} )) || workloads=(krylov ordinarydiffeq integrals_optimization)
read -r -a gpus_list <<< "${DAGGER_TUNE_GPUS-1 2 4 8}"
read -r -a blocks_list <<< "${DAGGER_TUNE_BLOCKS-1 2 4 8 16 32 64}"
samples=${DAGGER_TUNE_SAMPLES-3}
threads=${DAGGER_TUNE_THREADS-8}
krylov_n=${DAGGER_TUNE_KRYLOV_N-4096}
ode_n=${DAGGER_TUNE_ODE_N-1024}
intopt_n=${DAGGER_TUNE_INTOPT_N-512}
solvers_arg=${BENCH_SOLVERS-cg,bicgstab}
case $solvers_arg in
    cg|bicgstab|cg,bicgstab|bicgstab,cg) ;;
    *) echo "Invalid BENCH_SOLVERS: $solvers_arg" >&2; exit 2 ;;
esac
IFS=, read -r -a krylov_solvers <<< "$solvers_arg"
for value in "$samples" "$threads" "$krylov_n" "$ode_n" "$intopt_n" "${blocks_list[@]}"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { echo "Expected a positive integer: $value" >&2; exit 2; }
done
(( samples >= 2 && krylov_n >= 2 && ode_n >= 4 && intopt_n >= 4 &&
   ${#gpus_list[@]} > 0 && ${#blocks_list[@]} > 0 )) || {
    echo "Need >=2 samples, valid problem sizes, and nonempty GPU/block lists" >&2; exit 2;
}
previous=0
for blocks in "${blocks_list[@]}"; do
    (( blocks > previous )) || { echo "Block factors must be increasing and unique" >&2; exit 2; }
    previous=$blocks
done
for gpus in "${gpus_list[@]}"; do
    [[ $gpus == 1 || $gpus == 2 || $gpus == 4 || $gpus == 8 ]] || {
        echo "GPU counts must be 1, 2, 4, or 8" >&2; exit 2;
    }
done

julia_bin=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
project=$script_dir/../environments/composability
output=${DAGGER_TUNE_OUTPUT:-"$script_dir/tunes/$(date +%Y%m%d-%H%M%S)-$$"}
header='workload,solver,eltype,n,gpus,blocks_per_gpu,mean_ms,median_ms,min_ms,max_ms,samples_ms'
if (( ! dry )); then
    mkdir -p -- "$(dirname -- "$output")" && mkdir -- "$output" || exit 2
    printf '%s\n' "$header" > "$output/results.csv"
    printf '%s\n' "$header" > "$output/best.csv"
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
failed=()

tune() {
    local workload=$1 solver=$2 n=$3 gpus=$4 prefix=$5 worker=$6
    shift 6
    local mask blocks log result timing eltype mean rest best_mean= best_row= best_blocks= row
    local -a cmd
    if ! mask=$(gpu_mask_for_count "$gpus"); then
        failed+=("$workload $solver G=$gpus (GPU mask)")
        return
    fi
    for blocks in "${blocks_list[@]}"; do
        # Avoid requesting more row partitions than there are rows.
        (( gpus * blocks <= n )) || break
        cmd=(env "CUDA_VISIBLE_DEVICES=$mask" "DAGGER_BLOCKS_PER_GPU=$blocks"
            "${prefix}_GPUS=$gpus" "${prefix}_SAMPLES=$samples"
            "$julia_bin" --startup-file=no --project="$project" --threads="$threads"
            "$script_dir/$worker" Dagger "$@" "$n")
        echo "==> $workload $solver N=$n G=$gpus blocks_per_gpu=$blocks"
        if (( dry )); then printf '  %q' "${cmd[@]}"; printf '\n'; continue; fi
        log=$output/$workload-$solver-g$gpus-b$blocks.log
        if result=$(run_logged_case "$log" "${cmd[@]}"); then
            # Workers publish RESULT only after numerical validation. Normalize
            # their different schemas, excluding warmup and process startup.
            timing=$(printf '%s\n' "$result" | awk -F, -v w="$workload" \
                -v s="$solver" -v g="$gpus" -v n="$n" -v count="$samples" '
                function positive(x) { return x ~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ && x+0 > 0 }
                {
                    if (w == "krylov") { t=4; m=8; fields=14 }
                    else if (w == "ordinarydiffeq") { t=2; m=6; fields=12 }
                    else { t=2; m=9; fields=17 }
                    if (NF != fields || $1 != "Dagger" || $(t+1) != g || $(t+2) != n) exit 1
                    if (w == "krylov" && ($2 != s || $3 != "stock")) exit 1
                    if (!positive($m) || !positive($(m+2)) || !positive($(m+3)) || !positive($(m+4))) exit 1
                    if (split($NF, times, ";") != count) exit 1
                    for (i in times) if (!positive(times[i])) exit 1
                    print $t "," $m "," $(m+2) "," $(m+3) "," $(m+4) "," $NF
                }') || timing=
            if [[ -n $timing ]]; then
                IFS=, read -r eltype mean rest <<< "$timing"
                row="$workload,$solver,$eltype,$n,$gpus,$blocks,$mean,$rest"
                printf '%s\n' "$row" >> "$output/results.csv"
                if [[ -z $best_mean ]] || awk -v a="$mean" -v b="$best_mean" 'BEGIN { exit !(a < b) }'; then
                    best_mean=$mean best_row=$row best_blocks=$blocks
                fi
                echo "    mean=$mean ms"
                if awk -v a="$mean" -v b="$best_mean" 'BEGIN { exit !(a > 4*b) }'; then
                    echo "    Stopping: more than 4x the best mean"
                    break
                fi
                continue
            fi
            echo "Invalid timing result: $log" >&2
        fi
        failed+=("$workload $solver G=$gpus blocks=$blocks ($log)")
    done
    if [[ -n $best_row ]]; then
        printf '%s\n' "$best_row" >> "$output/best.csv"
        echo "best: $workload $solver G=$gpus N=$n blocks_per_gpu=$best_blocks ($best_mean ms/solve)"
    elif (( ! dry )); then
        failed+=("$workload $solver G=$gpus: no successful candidates")
    fi
}

for workload in "${workloads[@]}"; do
    for gpus in "${gpus_list[@]}"; do
        case $workload in
            krylov)
                for solver in "${krylov_solvers[@]}"; do
                    tune "$workload" "$solver" "$krylov_n" "$gpus" BENCH krylov/krylov.jl "$solver" stock
                done ;;
            ordinarydiffeq)
                tune "$workload" heat "$ode_n" "$gpus" ODE ordinarydiffeq/benchmark_heat.jl ;;
            integrals_optimization)
                tune "$workload" plume "$intopt_n" "$gpus" INTOPT integrals_optimization/benchmark.jl ;;
        esac
    done
done
if (( ${#failed[@]} )); then
    printf 'Failed tunes:\n'; printf '  %s\n' "${failed[@]}"; exit 1
fi
(( dry )) || echo "All tunes finished; timings: $output/results.csv; winners: $output/best.csv"
exit 0
