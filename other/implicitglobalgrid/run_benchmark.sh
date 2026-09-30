#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project="$script_dir/../../environments/implicitglobalgrid"
julia_bin=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}

usage() {
    echo "Usage: $0 GPUS N [STEPS=10] [WARMUP=5]" >&2
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (( $# < 2 || $# > 4 )); then usage; exit 2; fi
gpus=$1; n=$2; steps=${3:-10}; warmup=${4:-5}
for value in "$gpus" "$n" "$steps"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
done
[[ $warmup =~ ^(0|[1-9][0-9]*)$ ]] && (( n >= 4 )) || { usage; exit 2; }

# Use this environment's MPI; IGG selects one GPU per node-local rank.
exec "$julia_bin" --startup-file=no --project="$project" -e '
    using MPI
    project = dirname(Base.active_project())
    worker = ARGS[1]
    gpus = parse(Int, ARGS[2])
    command = `$(MPI.mpiexec()) -n $gpus $(Base.julia_cmd()) --startup-file=no --project=$project $worker $(ARGS[2:end])`
    process = run(ignorestatus(command))
    exit(success(process) ? 0 : 1)
' "$script_dir/grayscott.jl" "$gpus" "$n" "$steps" "$warmup"
