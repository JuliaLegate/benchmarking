export IGG_CUDAAWARE_MPI=1

NUM_ITERS=1000

BENCHMARK=grayscott
FOLDER="benchmarks/$BENCHMARK"
CSV="$FOLDER/$BENCHMARK.csv"

CUNUMERIC_GRAY_SCOPE="bash run_benchmark.sh cunumeric ${BENCHMARK}_scoping"
CUNUMERIC_GRAY_GC="bash run_benchmark.sh cunumeric ${BENCHMARK}_gc"
CUNUMERIC_GRAY="bash run_benchmark.sh cunumeric $BENCHMARK"
CUPYNUMERIC_GRAY="bash run_benchmark.sh cupynumeric $BENCHMARK"
DIFFEQ_GRAY="bash run_benchmark.sh diffeq $BENCHMARK --diffeq"

GPUS_LIST=(1 2 4 8)
# GC_LIST=(1 2 3 4 5)
GC_LIST=(6)


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

    args=(--gpus "$gpus" "$N" "$M" "$NUM_ITERS")

    # $CUPYNUMERIC_GRAY "${args[@]}"
    $CUNUMERIC_GRAY "${args[@]}"
    # $DIFFEQ_GRAY "${args[@]}"

    $CUNUMERIC_GRAY_SCOPE "${args[@]}"

    for gc in "${GC_LIST[@]}"; do
        $CUNUMERIC_GRAY_GC "${args[@]}" "${gc}"
    done
done