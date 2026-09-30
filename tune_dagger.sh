#!/bin/bash
# Tune Dagger blocks_per_gpu at every weak- and strong-scaling point in
# configs/multi_gpu; each run appends to tunes/<name>.csv.
#   ./tune_dagger.sh [benchmark...]   e.g. ./tune_dagger.sh grayscott nas_ft
# Weak scaling: gpus[i] runs N[i] -> tunes/<name>.csv.
# Strong scaling: one size on every GPU count -> tunes/<name>-strong.csv.
# Failed tunes are listed at the end; the script keeps going.

set -uo pipefail
cd "$(dirname "$0")"

JULIA=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
CPUS=8
ONLY=("$@")
FAILED=()

# tune <gpus> <name> <T> <N> <M> [kwargs TOML]
tune() {
    local gpus=$1 name=$2
    [[ ${#ONLY[@]} -gt 0 && ! " ${ONLY[*]} " =~ " $name " ]] && return
    echo
    echo "==> $name, $gpus GPU(s): ${*:3}"
    bash run_benchmark.sh --model=dagger --gpus="$gpus" --cpus=$CPUS -- \
        "$JULIA" --project=environments/dagger --threads=$CPUS src/dagger/tune.jl "${@:2}" ||
        FAILED+=("$*")
}

GPUS=(1 2 4 8)

# Weak scaling. gemm/montecarlo are auto-sized in all.toml; these are its paper sizes.
N=(20000 25200 31752 40000)
for i in "${!GPUS[@]}"; do tune "${GPUS[i]}" gemm Float32 "${N[i]}" "${N[i]}"; done
N=(1000000 2000000 4000000 8000000)
for i in "${!GPUS[@]}"; do tune "${GPUS[i]}" montecarlo Float32 "${N[i]}" 1; done
N=(24000 33944 48000 67888)
for i in "${!GPUS[@]}"; do tune "${GPUS[i]}" grayscott Float32 "${N[i]}" "${N[i]}"; done
N=(9000000 18000000 36000000 72000000)
for i in "${!GPUS[@]}"; do
    tune "${GPUS[i]}" cg Float64 "${N[i]}" 1 $'check_every = 10\nmax_iter = 1000'
done
# NAS: N/M are fixed by the class.
CLASS=(B B.2 C C.8); N=(2147483648 4294967296 8589934592 17179869184)
for i in "${!GPUS[@]}"; do
    tune "${GPUS[i]}" nas_ep Float64 "${N[i]}" 1 "class = \"${CLASS[i]}\""
done
CLASS=(A A.2 B.4 B.8); N=(256 256 512 512); M=(256 256 256 512)
for i in "${!GPUS[@]}"; do
    tune "${GPUS[i]}" nas_ft Float64 "${N[i]}" "${M[i]}" "class = \"${CLASS[i]}\""
done
CLASS=(B B.2 B.4 C); N=(256 256 256 512); M=(256 256 512 512)
for i in "${!GPUS[@]}"; do
    tune "${GPUS[i]}" nas_mg Float64 "${N[i]}" "${M[i]}" "class = \"${CLASS[i]}\""
done

# Strong scaling: NAS class B on every GPU count, 1 GPU included so each file
# stands alone.
export DAGGER_TUNE_TAG=strong
for g in "${GPUS[@]}"; do
    tune "$g" nas_ep Float64 2147483648 1 'class = "B"'
    tune "$g" nas_ft Float64 512 256 'class = "B"'
    tune "$g" nas_mg Float64 256 256 'class = "B"'
done
unset DAGGER_TUNE_TAG

echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All tunes finished; results in tunes/."
else
    echo "Failed tunes:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
