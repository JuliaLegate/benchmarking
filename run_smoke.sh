#!/bin/bash
# Smoke test of run_all.sh: same runs at small sizes, still 1/2/4/8 GPUs.
# Results go to results/smoke/<name>. Dagger tuning is skipped (default blocks).
#   ./run_smoke.sh [run.jl args...]   e.g. ./run_smoke.sh --verbose
# Failed runs are listed at the end; the script keeps going.

set -uo pipefail
cd "$(dirname "$0")"

JULIA=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
OUT=results/smoke
SMOKE=configs/smoke
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
        --config=$SMOKE/composability.toml || FAILED+=("composability $*")
}

mkdir -p $OUT

# Weak scaling.
run gemm       --config=$SMOKE/gemm.toml
run montecarlo --config=$SMOKE/montecarlo.toml
run grayscott  --config=$SMOKE/grayscott.toml
run cg         --config=$SMOKE/cg.toml
run nas_ep     --config=$SMOKE/nas_ep.toml
run nas_ft     --config=$SMOKE/nas_ft.toml
run nas_mg     --config=$SMOKE/nas_mg.toml

# Variants.
run grayscott_forms   --config=$SMOKE/grayscott_forms.toml
run montecarlo_forms  --config=$SMOKE/montecarlo_forms.toml
run grayscott_unfused --config=$SMOKE/grayscott.toml --models=cunumeric --fusion=off

# Single-GPU size sweeps.
run sweep_grayscott  --config=$SMOKE/sweep_grayscott.toml
run sweep_montecarlo --config=$SMOKE/sweep_montecarlo.toml

# ImplicitGlobalGrid Gray-Scott, same sizes as smoke/grayscott.toml.
export IGG_OUTPUT=$OUT/implicitglobalgrid
mkdir -p $IGG_OUTPUT
gpus=(1 2 4 8); sizes=(1026 1450 2050 2898)
for i in "${!gpus[@]}"; do
    echo
    echo "==> implicitglobalgrid: ${gpus[i]} GPU(s), N=${sizes[i]}"
    JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-8} IGG_CUDAAWARE_MPI=${IGG_CUDAAWARE_MPI:-1} \
        bash other/implicitglobalgrid/run_benchmark.sh "${gpus[i]}" "${sizes[i]}" 5 1 2 ||
        FAILED+=("implicitglobalgrid ${gpus[i]}")
done
unset IGG_OUTPUT

# Gray-Scott comparison CSVs.
mkdir -p $OUT/grayscott_compare/Float32
cp $OUT/grayscott/Float32/grayscott_cunumeric.csv \
   $OUT/grayscott/Float32/grayscott_cupynumeric.csv \
   $OUT/grayscott_unfused/Float32/grayscott_cunumeric_nofusion.csv \
   $OUT/implicitglobalgrid/Float32/grayscott_igg.csv \
   $OUT/grayscott_compare/Float32/ || FAILED+=("grayscott_compare")

# Composability.
composability --only=ordinarydiffeq,krylov --solvers=cg --local --mode=single --models=cuda,dagger,cunumeric
composability --only=krylov --solvers=cg --mode=multi --gpus=1,2,4,8 --models=dagger,cunumeric
composability --only=ordinarydiffeq --mode=multi --gpus=1,2,4,8 --models=cunumeric

echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All smoke runs finished: $OUT"
else
    echo "Failed runs:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
