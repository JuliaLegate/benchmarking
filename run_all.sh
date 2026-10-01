#!/bin/bash
# Run every benchmark config: main weak scaling, variants, NAS strong scaling,
# then composability weak scaling (COMPOSABILITY_CONFIG picks the size file).
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
run --config=$MULTI/all.toml --only=gemm
run --config=$MULTI/all.toml --only=montecarlo
run --config=$MULTI/grayscott.toml
run --config=$MULTI/cg.toml
run --config=$MULTI/nas_ep_weak.toml
run --config=$MULTI/nas_ft_weak.toml
run --config=$MULTI/nas_mg_weak.toml

# Variants: Gray-Scott @accelerate forms, Monte Carlo fused vs naive.
run --config=$MULTI/grayscott_forms.toml
run --config=$MULTI/montecarlo_forms.toml

# Strong scaling: NAS.
run --config=$MULTI/nas_ep_strong.toml
run --config=$MULTI/nas_ft_strong.toml
run --config=$MULTI/nas_mg_strong.toml

# Composability weak scaling: OrdinaryDiffEq heat and Krylov CG.
# COMPOSABILITY_MODELS picks models (default cuda,cunumeric: no Dagger).
COMPOSABILITY=(--only=ordinarydiffeq,krylov --solvers=cg --mode=multi --gpus=1,2,4,8
    --models=${COMPOSABILITY_MODELS:-cuda,cunumeric}
    --config=${COMPOSABILITY_CONFIG:-composability/sizes_141GB.toml}
    --output=results/composability-multi-$(date +%Y%m%d-%H%M%S))
echo
echo "==> $JULIA --project=. run_composability.jl ${COMPOSABILITY[*]}"
"$JULIA" --project=. run_composability.jl "${COMPOSABILITY[@]}" || FAILED+=("composability ${COMPOSABILITY[*]}")

echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All runs finished."
else
    echo "Failed runs:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
