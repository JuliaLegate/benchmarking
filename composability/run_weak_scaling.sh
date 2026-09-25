#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: COMPOSABILITY_OUTPUT_ROOT=/path/to/results $0 GPU_COUNT [GPU_COUNT ...]" >&2
    exit 2
}
[[ $# -ge 1 ]] || usage
for g in "$@"; do
    [[ $g == 1 || $g == 2 || $g == 4 || $g == 8 ]] || usage
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
plan="$script_dir/weak_scaling_plan.csv"
[[ -f $plan ]] || { echo "Missing $plan" >&2; exit 2; }
read -r -a workloads <<< "${COMPOSABILITY_WORKLOADS:-krylov ordinarydiffeq}"
[[ ${#workloads[@]} -gt 0 ]] || usage
declare -A selected=()
for workload in "${workloads[@]}"; do
    case $workload in
        krylov|ordinarydiffeq|integrals_optimization) ;;
        *) echo "Unknown workload: $workload" >&2; exit 2 ;;
    esac
    [[ ! ${selected[$workload]+exists} ]] || {
        echo "Duplicate workload: $workload" >&2
        exit 2
    }
    selected[$workload]=1
done
output_root=${COMPOSABILITY_OUTPUT_ROOT:-"$script_dir/results-weak-$(date +%Y%m%d-%H%M%S)-$$"}
for workload in "${workloads[@]}"; do
    [[ ! -e "$output_root/$workload/results.csv" ]] || {
        echo "Existing results would be overwritten: $output_root/$workload" >&2
        exit 2
    }
done
mkdir -p "$output_root"
cp "$plan" "$output_root/weak_scaling_plan.csv"

plan_field() {
    awk -F, -v workload="$1" -v column="$2" \
        '$1 == workload { print $column }' "$plan"
}

for workload in "${workloads[@]}"; do
    base_n=$(plan_field "$workload" 2)
    [[ $base_n =~ ^[1-9][0-9]*$ ]] || {
        echo "Missing base N for $workload in $plan" >&2
        exit 2
    }
    for g in "$@"; do
        case $g in
            1) column=3 ;;
            2) column=4 ;;
            4) column=5 ;;
            8) column=6 ;;
        esac
        expected=$(plan_field "$workload" "$column")
        actual=$(awk -v base="$base_n" -v count="$g" \
            'BEGIN { printf "%.0f", base * sqrt(count) }')
        [[ $expected == "$actual" ]] || {
            echo "$workload G=$g: plan has N=$expected, launcher would use N=$actual" >&2
            exit 2
        }
    done
done

dry_run=${COMPOSABILITY_DRY_RUN:-0}
[[ $dry_run == 0 || $dry_run == 1 ]] || usage
status=0
for workload in "${workloads[@]}"; do
    echo "$workload weak scaling: N(1)=$(plan_field "$workload" 2)"
    case $workload in
        krylov)
            if ! BENCH_PROJECT="${BENCH_PROJECT:-/opt/bench-envs/krylov}" \
                BENCH_OUTPUT="$output_root/krylov" BENCH_DRY_RUN="$dry_run" \
                bash "$script_dir/krylov/run.sh" weak \
                "$(plan_field krylov 2)" "$@"; then
                status=1
            fi
            ;;
        ordinarydiffeq)
            if ! ODE_PROJECT="${ODE_PROJECT:-/opt/bench-envs/ode}" \
                ODE_OUTPUT="$output_root/ordinarydiffeq" ODE_DRY_RUN="$dry_run" \
                bash "$script_dir/ordinarydiffeq/run_benchmark.sh" weak \
                "$(plan_field ordinarydiffeq 2)" "$@"; then
                status=1
            fi
            ;;
        integrals_optimization)
            if ! INTOPT_PROJECT="${INTOPT_PROJECT:-/opt/bench-envs/intopt}" \
                INTOPT_OUTPUT="$output_root/integrals_optimization" \
                INTOPT_DRY_RUN="$dry_run" \
                bash "$script_dir/integrals_optimization/run_benchmark.sh" weak \
                "$(plan_field integrals_optimization 2)" "$@"; then
                status=1
            fi
            ;;
    esac
done

echo "Results: $output_root"
exit "$status"
