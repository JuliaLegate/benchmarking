WARMUP=1
NUM_TRIALS=20

CUNUMERIC_GEMM="bash run_benchmark.sh cunumeric matmul"
CUDA_GEMM="bash run_benchmark.sh cuda matmul"

GPUS_LIST=(1 2 4 8)

julia gemm/gemm_nval.jl > gemm/nval.sh
source gemm/nval.sh

# SIZES=("${small[@]}")  
SIZES=("${large[@]}")  

touch gemm/gemm.csv
expected_header="model,gpus,n,m,mean_time_ms,gflops"
if ! head -n 1 gemm/gemm.csv | grep -qx "$expected_header"; then
    sed -i "1i $expected_header" gemm/gemm.csv
fi

for i in "${!GPUS_LIST[@]}"; do
    gpus="${GPUS_LIST[$i]}"
    read -r N M <<< "${SIZES[$i]}"
    args=(--gpus "$gpus" "$N" "$M" "$NUM_TRIALS" "$WARMUP")

    $CUNUMERIC_GEMM "${args[@]}"
    $CUDA_GEMM "${args[@]}"
done