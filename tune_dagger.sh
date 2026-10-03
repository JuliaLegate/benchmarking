#!/bin/bash
# Tune main and composability benchmarks. Comment out calls below to disable them.
#   ./tune_dagger.sh [--dry-run] [benchmark...]
#   ./tune_dagger.sh krylov_cg ordinarydiffeq
# Main results append to tunes/<name>.csv; composability uses tunes/composability/.
# Weak scaling: gpus[i] runs N[i] -> tunes/<name>.csv.
# Strong scaling: one size on every GPU count -> tunes/<name>-strong.csv.
# Failed tunes are listed at the end; the script keeps going.

set -uo pipefail
if [[ -n ${DAGGER_TUNE_CONFIG:-} ]]; then
    DAGGER_TUNE_CONFIG=$(realpath "$DAGGER_TUNE_CONFIG") || exit 1
    export DAGGER_TUNE_CONFIG
fi
cd "$(dirname "$0")" || exit 1

JULIA=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
CPUS=8
GPUS=(1 2 4 8)
DRY=0
[[ ${1:-} != --dry-run ]] || { DRY=1; shift; }
ONLY=("$@")
FAILED=()

# Main: tune <gpus> <name> <T> <N> <M> [kwargs TOML]
# Composability: tune <gpus> <name> (sizes come from DAGGER_TUNE_CONFIG).
tune() {
    local gpus=$1 name=$2
    [[ ${#ONLY[@]} -gt 0 && ! " ${ONLY[*]} " =~ " $name " ]] && return
    local project=environments/dagger worker=src/dagger/tune.jl
    local -a extra=() julia_flags=() cmd
    case $name in
        krylov_cg|krylov_bicgstab|ordinarydiffeq|integrals_optimization)
            project=environments/composability worker=composability/tune.jl
            julia_flags=(--startup-file=no)
            (( ! DRY )) || extra=(--dry-run) ;;
    esac
    echo
    echo "==> $name, $gpus GPU(s): ${*:3}"
    cmd=(bash run_benchmark.sh --model=dagger --gpus="$gpus" --cpus=$CPUS --
        "$JULIA" "${julia_flags[@]}" --project="$project" --threads=$CPUS
        "$worker" "${@:2}" "${extra[@]}")
    if (( DRY )) && [[ $worker == src/dagger/tune.jl ]]; then
        printf '  %q' "${cmd[@]}"; printf '\n'; return
    fi
    "${cmd[@]}" || FAILED+=("$*")
}

# Weak scaling, matching configs/multi_gpu/*.toml.
N=(43408 54688 68904 86816)
for i in "${!GPUS[@]}"; do tune "${GPUS[i]}" gemm Float32 "${N[i]}" "${N[i]}"; done
N=(7537741000 15075482000 30150964000 60301928000)
for i in "${!GPUS[@]}"; do tune "${GPUS[i]}" montecarlo Float32 "${N[i]}" 1; done
N=(28000 39600 56000 79200)
for i in "${!GPUS[@]}"; do tune "${GPUS[i]}" grayscott Float32 "${N[i]}" "${N[i]}"; done
N=(90000000 180000000 360000000 720000000)
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

# Composability: H200 weak_base sizes, one warmup + two five-iteration runs.
# Keep each call on its own line so individual solvers are easy to toggle.
for g in "${GPUS[@]}"; do
    tune "$g" krylov_cg
    # tune "$g" krylov_bicgstab
    tune "$g" ordinarydiffeq
    # tune "$g" integrals_optimization
done

echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All tunes finished; results in tunes/."
else
    echo "Failed tunes:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
