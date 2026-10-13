#!/bin/bash
# Run every paper benchmark. Results go to results/paper/<name>, which plot_all.sh reads.
#   ./run_all.sh [run.jl args...]   e.g. ./run_all.sh --verbose
# Failed runs are listed at the end; the script keeps going.

set -uo pipefail
cd "$(dirname "$0")"

JULIA=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
OUT=results/paper
MULTI=configs/multi_gpu
SINGLE=configs/single_gpu
EXTRA=("$@")
FAILED=()

# run <name> <run.jl args...>
run() {
    local name=$1; shift
    echo
    echo "==> $name: run.jl $*"
    "$JULIA" --project=. run.jl "$@" --output=$OUT/$name "${EXTRA[@]}" || FAILED+=("$name")
}

composability() {
    echo
    echo "==> composability: $*"
    "$JULIA" --project=. run_composability.jl "$@" --output=$OUT/composability \
        --config=composability/sizes_141GB.toml || FAILED+=("composability $*")
}

# Dagger tunes first: Dagger runs read their block counts from results/tunes/.
bash scripts/tune_dagger.sh || FAILED+=("tune_dagger")
mkdir -p $OUT && cp -r results/tunes $OUT/

# Weak scaling: the seven main benchmarks.
run gemm       --config=$MULTI/gemm.toml
run montecarlo --config=$MULTI/montecarlo.toml
run grayscott  --config=$MULTI/grayscott.toml
run cg         --config=$MULTI/cg.toml
run nas_ep     --config=$MULTI/nas_ep_weak.toml
run nas_ft     --config=$MULTI/nas_ft_weak.toml
run nas_mg     --config=$MULTI/nas_mg_weak.toml

# Variants: Gray-Scott @accelerate forms, Monte Carlo fused vs naive,
# and unfused Gray-Scott for the IGG comparison.
run grayscott_forms   --config=$MULTI/grayscott_forms.toml
run montecarlo_forms  --config=$MULTI/montecarlo_forms.toml
run grayscott_unfused --config=$MULTI/grayscott.toml --models=cunumeric --fusion=off

# Single-GPU size sweeps.
run sweep_grayscott  --config=$SINGLE/grayscott_forms.toml
run sweep_montecarlo --config=$SINGLE/montecarlo_forms.toml

# Strong scaling: NAS (not used in the paper).
# run nas_ep_strong --config=$MULTI/nas_ep_strong.toml
# run nas_ft_strong --config=$MULTI/nas_ft_strong.toml
# run nas_mg_strong --config=$MULTI/nas_mg_strong.toml

# ImplicitGlobalGrid Gray-Scott, same weak-scaling sizes as grayscott.toml.
IGG_OUTPUT=$OUT/implicitglobalgrid bash other/implicitglobalgrid/run_weak_scaling.sh ||
    FAILED+=("implicitglobalgrid")

# Gray-Scott comparison figure: CSVs from three of the runs above.
mkdir -p $OUT/grayscott_compare/Float32
cp $OUT/grayscott/Float32/grayscott_cunumeric.csv \
   $OUT/grayscott/Float32/grayscott_cupynumeric.csv \
   $OUT/grayscott_unfused/Float32/grayscott_cunumeric_nofusion.csv \
   $OUT/implicitglobalgrid/Float32/grayscott_igg.csv \
   $OUT/grayscott_compare/Float32/ || FAILED+=("grayscott_compare")

# Composability: single GPU (ODE heat, Krylov CG), then weak scaling.
composability --only=ordinarydiffeq,krylov --solvers=cg --local --mode=single --models=cuda,dagger,cunumeric
composability --only=krylov --solvers=cg --mode=multi --gpus=1,2,4,8 --models=dagger,cunumeric
composability --only=ordinarydiffeq --mode=multi --gpus=1,2,4,8 --models=cunumeric

echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All runs finished: $OUT"
else
    echo "Failed runs:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
