NUM_ITERS=1000

BENCHMARK=fusion
FOLDER="benchmarks/$BENCHMARK"
CSV="$FOLDER/$BENCHMARK.csv"

CUNUMERIC_FUSION="bash run_benchmark.sh cunumeric $BENCHMARK"

GPUS_LIST=(1 2 4 8)
# GPUS_LIST=(1)

julia $FOLDER/nval.jl > $FOLDER/nval.sh
source $FOLDER/nval.sh

# SIZES=("${test[@]}")  
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

    args=(--gpus "$gpus" "$N" "$NUM_ITERS")

    $CUNUMERIC_FUSION "${args[@]}" 1 # use operator kernel fusion with legate+cuda.jl
    $CUNUMERIC_FUSION "${args[@]}" 0 # no fusion with cunumeric.jl
done