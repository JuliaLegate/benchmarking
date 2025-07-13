WARMUP=1
NUM_TRIALS=20

CUNUMERIC_GRAY="bash run_benchmark.sh cunumeric grayscott"
DIFFEQ_GRAY="bash run_benchmark.sh diffeq grayscott"

GPUS_LIST=(1 2 4 8)

julia grayscott/grayscott_nval.jl > grayscott/nval.sh
source grayscott/nval.sh

SIZES=("${small[@]}")  
# SIZES=("${large[@]}")  

touch grayscott/grayscott.csv
expected_header="model,gpus,n,m,mean_time_ms,gflops"
if ! head -n 1 grayscott/grayscott.csv | grep -qx "$expected_header"; then
    sed -i "1i $expected_header" grayscott/grayscott.csv
fi

for i in "${!GPUS_LIST[@]}"; do
    gpus="${GPUS_LIST[$i]}"
    read -r N M <<< "${SIZES[$i]}"
    args=(--gpus "$gpus" "$N" "$N" "$NUM_TRIALS" "$WARMUP")

    $CUNUMERIC_GRAY "${args[@]}"
    $DIFFEQ_GRAY "${args[@]}"
done