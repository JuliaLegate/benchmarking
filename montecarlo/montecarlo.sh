
NUM_ITERS=20

CUNUMERIC_MC="bash run_benchmark.sh cunumeric montecarlo"
CUPYNUMERIC_MC="bash run_benchmark.sh cupynumeric montecarlo"

GPUS_LIST=(1 2 4 8)

BASE_SIZE=3000000000 # 1 billion
SIZES=()

# Loop through GPUS_LIST and calculate sizes
for size in "${GPUS_LIST[@]}"
do
    SIZES+=($(( BASE_SIZE * size )))
done


touch montecarlo/mc.csv
expected_header="model,gpus,n,m,mean_time_ms,gflops"
if ! head -n 1 montecarlo/montecarlo.csv | grep -qx "$expected_header"; then
    sed -i "1i $expected_header" montecarlo/montecarlo.csv
fi

for i in "${!GPUS_LIST[@]}"; do
    gpus="${GPUS_LIST[$i]}"
    read -r N M <<< "${SIZES[$i]}"

    args=(--gpus "$gpus" "$N" "$NUM_ITERS")

    $CUNUMERIC_MC "${args[@]}"
    $CUPYNUMERIC_MC "${args[@]}"
 
done