WARMUP=1
NUM_TRIALS=20

BENCHMARK=gemm
FOLDER="benchmarks/$BENCHMARK"
CSV="$FOLDER/$BENCHMARK.csv"


CUNUMERIC_GEMM="bash run_benchmark.sh cunumeric $BENCHMARK"
CUDA_GEMM="bash run_benchmark.sh cuda $BENCHMARK"

GPUS_LIST=(1 2 4 8)

julia $FOLDER/nval.jl > $FOLDER/nval.sh
source $FOLDER/nval.sh

SIZES=("${small[@]}")  
# SIZES=("${large[@]}")  

touch $CSV
expected_header="model,gpus,n,m,mean_time_ms,gflops"
if ! head -n 1 $CSV | grep -qx "$expected_header"; then
    sed -i "1i $expected_header" $CSV
fi

for i in "${!GPUS_LIST[@]}"; do
    gpus="${GPUS_LIST[$i]}"
    read -r N M <<< "${SIZES[$i]}"
    args=(--gpus "$gpus" "$N" "$M" "$NUM_TRIALS" "$WARMUP")

    $CUNUMERIC_GEMM "${args[@]}"
    $CUDA_GEMM "${args[@]}"
done