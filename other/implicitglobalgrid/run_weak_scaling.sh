#!/usr/bin/env bash
# Run the same 1/2/4/8-GPU sizes as configs/multi_gpu/grayscott.toml.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    echo "Usage: bash $0 [N_ITER=50] [N_WARMUP=2] [N_TRIALS=5]" >&2
    echo "Runs 1/2/4/8 GPUs with global N=28000/39600/56000/79200." >&2
    echo "IGG_OUTPUT selects the output directory (default: results/implicitglobalgrid)." >&2
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (( $# > 3 )); then usage; exit 2; fi
n_iter=${1:-${N_ITER:-50}}
n_warmup=${2:-${N_WARMUP:-2}}
n_trials=${3:-${N_TRIALS:-5}}
for value in "$n_iter" "$n_trials"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
done
[[ $n_warmup =~ ^(0|[1-9][0-9]*)$ ]] || { usage; exit 2; }

cd "$script_dir/../.."
export IGG_OUTPUT=${IGG_OUTPUT:-results/implicitglobalgrid}
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-8}
export IGG_CUDAAWARE_MPI=${IGG_CUDAAWARE_MPI:-1}
trap 'unset IGG_CUDAAWARE_MPI' EXIT

gpus=(1 2 4 8)
sizes=(28000 39600 56000 79200)
mkdir -p "$IGG_OUTPUT"
for i in "${!gpus[@]}"; do
    bash "$script_dir/run_benchmark.sh" "${gpus[$i]}" "${sizes[$i]}" "$n_iter" "$n_warmup" "$n_trials" \
        2>&1 | tee "$IGG_OUTPUT/igg-${gpus[$i]}gpu-N${sizes[$i]}.log"
done
