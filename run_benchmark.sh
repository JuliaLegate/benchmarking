#!/bin/bash

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <programming_model> <benchmark> [--gpus <num_gpus>] [--cpus <num_cpus>] [--config-as-args] [extra_args...]"
    exit 1
fi
FILEEXT=".jl"
MODEL=$1
BENCHMARK=$2
shift 2  # Move past MODEL and BENCHMARK

MACHINE="dub"

# Defaults
GPUS=1
CPUS=1
CONFIG_AS_ARGS=0
EXTRA_ARGS=()

# Parse optional flags and extra arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS=$2
            shift 2
            ;;
        --cpus)
            CPUS=$2
            shift 2
            ;;
        --diffeq)
            DIFFEQ_CONDITION=1
            shift 1
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ $DIFFEQ_CONDITION -eq 1 ]]; then
    DIFFEQ="mpiexec -n $GPUS"
fi

RUNNER="julia --project='models/$MODEL'"


if [[ "$MODEL" == "cupynumeric" ]]; then
    FILEEXT=".py"
    RUNNER=python3
fi

# Check for script existence
if [[ ! -f "models/$MODEL/$BENCHMARK$FILEEXT" ]]; then
    echo "Error: File $MODEL/$BENCHMARK$FILEEXT does not exist."
    exit 1
fi

# Validate CPU and GPU inputs
if [[ $GPUS -lt 0 ]]; then
    echo "Invalid GPU count: $GPUS"
    exit 1
fi

if [[ $CPUS -lt 0 ]]; then
    echo "Invalid CPU count: $CPUS"
    exit 1
fi

# Configure runtime
if [[ "$MODEL" == "cunumeric" || "$MODEL" == "cupynumeric" ]]; then
    export LEGATE_AUTO_CONFIG=0
    if [[ "$MACHINE" == "nvidia-brev" ]]; then 
        export LEGATE_CONFIG="--cpus=$CPUS --gpus=$GPUS --omps=0 --ompthreads=0 --utility=8 --sysmem=774026  --numamem=0 --fbmem=76596 --zcmem=128 --regmem=0"
    else
        export LEGATE_CONFIG="--cpus=$CPUS --gpus=$GPUS --omps=0 --ompthreads=0 --utility=2 --sysmem=256 --numamem=19029 --fbmem=7569 --zcmem=128 --regmem=0"
    fi 
    export LEGATE_SHOW_CONFIG=0
fi

printf "\n"
echo "Running: $MODEL/$BENCHMARK$FILEEXT with $CPUS CPUs and $GPUS GPUs"
CMD="$DIFFEQ $RUNNER models/$MODEL/$BENCHMARK$FILEEXT $GPUS ${EXTRA_ARGS[@]}"
printf "Running: %s\n" "$CMD"
eval "$CMD"