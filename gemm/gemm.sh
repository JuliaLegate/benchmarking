WARMUP=1
NUM_TRIALS=10

CUNUMERIC_GEMM="bash run_benchmark.sh cunumeric matmul"
CUDA_GEMM="bash run_benchmark.sh cuda matmul"

GPUS_LIST=(1 2 4 8)

# declare -a SIZES=(
#   "48000 48000"
#   "48000 96000"
#   "96000 96000"
#   "96000 192000"
# )

declare -a SIZES=(
  "12288 24576"
  "16384 92682"
  "24576 49152"
  "32768 65536"
)


touch gemm/gemm.csv
expected_header="model,gpus,n,m,mean_time_ms,gflops"
if ! head -n 1 gemm/gemm.csv | grep -qx "$expected_header"; then
    sed -i "1i $expected_header" gemm/gemm.csv
fi

for i in "${!GPUS_LIST[@]}"; do
    gpus="${GPUS_LIST[$i]}"
    read -r N M <<< "${SIZES[$i]}"
    args=(--gpus "$gpus" "$N" "$M" "$NUM_TRIALS" "$WARMUP")

    # $CUNUMERIC_GEMM "${args[@]}"
    $CUDA_GEMM "${args[@]}"
done