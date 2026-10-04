#!/bin/bash
# Run every paper benchmark.
#   ./run_all.sh [run.jl args...]   e.g. ./run_all.sh --verbose
# Failed runs are listed at the end; the script keeps going.

set -uo pipefail
cd "$(dirname "$0")"

JULIA=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
MULTI=configs/multi_gpu
EXTRA=("$@")
FAILED=()

run() {
    echo
    echo "==> $JULIA --project=. run.jl $* ${EXTRA[*]}"
    "$JULIA" --project=. run.jl "$@" "${EXTRA[@]}" || FAILED+=("$*")
}

# Weak scaling: the seven main benchmarks.
run --config=$MULTI/gemm.toml
run --config=$MULTI/montecarlo.toml
run --config=$MULTI/grayscott.toml
run --config=$MULTI/cg.toml
run --config=$MULTI/nas_ep_weak.toml
run --config=$MULTI/nas_ft_weak.toml
run --config=$MULTI/nas_mg_weak.toml

# Variants: Gray-Scott @accelerate forms, Monte Carlo fused vs naive.
run --config=$MULTI/grayscott_forms.toml
run --config=$MULTI/montecarlo_forms.toml

# Strong scaling: NAS (not used in the paper).
# run --config=$MULTI/nas_ep_strong.toml
# run --config=$MULTI/nas_ft_strong.toml
# run --config=$MULTI/nas_mg_strong.toml

# ImplicitGlobalGrid Gray-Scott, same weak-scaling sizes as grayscott.toml.
echo
echo "==> bash other/implicitglobalgrid/run_weak_scaling.sh"
bash other/implicitglobalgrid/run_weak_scaling.sh || FAILED+=("implicitglobalgrid")

composability() {
    local args=("$@" --config=${COMPOSABILITY_CONFIG:-composability/sizes_141GB.toml})
    echo
    echo "==> $JULIA --project=. run_composability.jl ${args[*]}"
    "$JULIA" --project=. run_composability.jl "${args[@]}" || FAILED+=("composability ${args[*]}")
}
STAMP=$(date +%Y%m%d-%H%M%S)

# Composability single GPU: OrdinaryDiffEq heat and Krylov CG (with local CG).
composability --only=ordinarydiffeq,krylov --solvers=cg --local --mode=single \
    --models=cuda,dagger,cunumeric --output=results/composability-single-$STAMP

# Composability weak scaling: Krylov CG.
composability --only=krylov --solvers=cg --mode=multi --gpus=1,2,4,8 \
    --models=${COMPOSABILITY_MODELS:-dagger,cunumeric} --output=results/composability-multi-$STAMP

echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All runs finished."
else
    echo "Failed runs:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
