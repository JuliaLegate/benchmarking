NUM_ITERS=20

BENCHMARK=montecarlo
FOLDER="benchmarks/$BENCHMARK"
CSV="$FOLDER/$BENCHMARK.csv"

CUNUMERIC_MC="bash run_benchmark.sh cunumeric $BENCHMARK"
CUPYNUMERIC_MC="bash run_benchmark.sh cupynumeric $BENCHMARK"

GPUS_LIST=(1 2 4 8)

BASE_SIZE=3000000000 # 1 billion
SIZES=()

# Loop through GPUS_LIST and calculate sizes
for size in "${GPUS_LIST[@]}"
do
    SIZES+=($(( BASE_SIZE * size )))
done


touch $CSV
expected_header="model,gpus,n,m,mean_time_ms,gflops"
if ! head -n 1 $CSV | grep -qx "$expected_header"; then
    sed -i "1i $expected_header" $CSV
fi

for i in "${!GPUS_LIST[@]}"; do
    gpus="${GPUS_LIST[$i]}"
    read -r N M <<< "${SIZES[$i]}"

    args=(--gpus "$gpus" "$N" "$NUM_ITERS")

    $CUNUMERIC_MC "${args[@]}"
    $CUPYNUMERIC_MC "${args[@]}"
 
done