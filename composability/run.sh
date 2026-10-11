#!/usr/bin/env bash
# Run one composability workload sweep, one fresh process per case.
#   run.sh WORKLOAD single N [N ...] | weak BASE_N GPUS [GPUS ...]
# WORKLOAD is krylov, ordinarydiffeq, or integrals_optimization; each reads
# settings with its own prefix (BENCH_, ODE_, INTOPT_), e.g. BENCH_OUTPUT,
# BENCH_BACKENDS, BENCH_SAMPLES. Usually launched by run_composability.jl.
set -euo pipefail
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage() { sed -n 3p "$0" >&2; exit 2; }
(( $# >= 3 )) || usage
workload=$1 experiment=$2; shift 2
case $experiment in
    single) ;;
    weak) base_n=$1; shift ;;
    *) usage ;;
esac

case $workload in
    krylov)
        P=BENCH cpus=2
        header=experiment,base_n,backend,solver,mode,eltype,gpus,n,iterations,mean_ms,stderr_ms,median_ms,min_ms,max_ms,relative_residual,samples_ms ;;
    ordinarydiffeq)
        P=ODE cpus=4
        header=experiment,base_n,backend,eltype,gpus,N,steps,mean_ms,stderr_ms,median_ms,min_ms,max_ms,relative_error,samples_ms ;;
    integrals_optimization)
        P=INTOPT cpus=4
        header=experiment,base_n,backend,eltype,gpus,N,bands,order,maxiters,objective_evals,mean_ms,stderr_ms,median_ms,min_ms,max_ms,initial_loss,final_loss,parameter_error,samples_ms ;;
    *) usage ;;
esac
setting() { local name=${P}_$1; printf '%s' "${!name-$2}"; }  # $P_<name>, or default $2

julia=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
project=$(setting PROJECT "$dir/../environments/composability")
output=$(setting OUTPUT "$PWD/$workload-$experiment-$(date +%Y%m%d-%H%M%S)-$$")
read -r -a backends <<< "$(setting BACKENDS "CuArray Dagger cuNumeric")"
export "${P}_SAMPLES=$(setting SAMPLES 5)" LEGATE_AUTO_CONFIG=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# Krylov runs each solver in stock mode, plus cuNumeric's local CG with BENCH_LOCAL=1.
cases=()
for backend in "${backends[@]}"; do
    if [[ $workload == krylov ]]; then
        IFS=, read -r -a solvers <<< "${BENCH_SOLVERS:-cg}"
        for solver in "${solvers[@]}"; do cases+=("$backend $solver stock"); done
    else
        cases+=("$backend")
    fi
done
if [[ $workload == krylov && ${BENCH_LOCAL:-0} == 1 ]]; then
    for solver in "${solvers[@]}"; do cases+=("cuNumeric $solver local"); done
fi

mkdir -p "$output"
csv=$output/results.csv
echo "$header" > "$csv"
echo 'case,gpus,n,legate_config,cuda_visible_devices' > "$output/planned-cases.csv"
{
    echo "benchmark_commit=$(git -C "$dir" rev-parse HEAD)"
    echo "cunumeric_commit=$(git -C "${CUNUMERIC_SOURCE:-/opt/cuNumeric.jl}" rev-parse HEAD)"
    echo "workload=$workload experiment=$experiment base_n=${base_n:-} values=$*"
    "$julia" --version
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    env | grep -E "^(${P}_|LEGATE_|CUBLAS_)" | sort
} > "$output/metadata.txt" 2>&1
cp "$project/Manifest.toml" "$output/"
[[ ! -f $project/LocalPreferences.toml ]] || cp "$project/LocalPreferences.toml" "$output/"

# First COUNT visible GPUs, as a CUDA_VISIBLE_DEVICES mask.
gpu_mask() {
    local -a devices
    if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
        IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    else
        devices=($(seq 0 $(($1 - 1))))
    fi
    (( ${#devices[@]} >= $1 )) || { echo "fewer than $1 GPUs visible" >&2; return 1; }
    local IFS=,; echo "${devices[*]:0:$1}"
}

status=0
for value in "$@"; do
    if [[ $experiment == single ]]; then
        gpus=1 n=$value
    else
        gpus=$value n=$(awk -v b="$base_n" -v g="$gpus" 'BEGIN { printf "%.0f", b * sqrt(g) }')
    fi
    mask=$(gpu_mask "$gpus")
    export "${P}_GPUS=$gpus" LEGATE_CONFIG="--gpus $gpus --cpus $(setting CPUS $cpus)"
    for case in "${cases[@]}"; do
        read -r -a c <<< "$case"
        [[ ${c[0]} != CuArray || $experiment == single ]] || continue  # CUDA.jl is single-GPU only
        echo "$case,$gpus,$n,$LEGATE_CONFIG,\"$mask\"" >> "$output/planned-cases.csv"
        echo "Running $case: G=$gpus N=$n"
        [[ $(setting DRY_RUN 0) != 1 ]] || continue
        log=$output/${case// /-}-$gpus-$n.log
        # Krylov and ODE run each sample in a fresh process; IntOpt samples in one, with a timeout.
        if [[ $workload == integrals_optimization ]]; then
            cmd=(timeout --kill-after=30s "${INTOPT_TIMEOUT:-15m}" "$julia" --startup-file=no
                --project="$project" "$dir/integrals_optimization/benchmark.jl" "${c[@]}" "$n")
        else
            cmd=("$julia" --startup-file=no --project="$project" "$dir/run_samples.jl"
                "$workload" "$log" "${c[@]}" "$n")
        fi
        if CUDA_VISIBLE_DEVICES=$mask "${cmd[@]}" > "$log" 2>&1 &&
            row=$(grep '^RESULT,' "$log") && [[ $(wc -l <<< "$row") == 1 ]]; then
            echo "$experiment,${base_n:-},${row#RESULT,}" >> "$csv"
        else
            echo "Failed: $case G=$gpus N=$n; last lines of $log:" >&2
            tail -n 20 "$log" >&2
            status=1
        fi
    done
done

echo "Results: $csv"
if (( $(wc -l < "$csv") > 1 )); then
    GKSwstype=100 "$julia" --startup-file=no --project="$project" \
        "$dir/$workload/plot_results.jl" "$experiment" "$csv" "$output/timings.png" ||
        { echo "Plot failed; data is in $csv" >&2; status=1; }
fi
exit "$status"
