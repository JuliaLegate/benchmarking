#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project="$script_dir/../../environments/implicitglobalgrid"

usage() {
    echo "Usage: $0 GPUS N [N_ITER=10] [N_WARMUP=5] [N_TRIALS=5]" >&2
    echo "N is the global square domain size, excluding halo cells." >&2
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (( $# < 2 || $# > 5 )); then usage; exit 2; fi
gpus=$1; n=$2
N_ITER=${3:-${N_ITER:-10}}
N_WARMUP=${4:-${N_WARMUP:-5}}
N_TRIALS=${5:-${N_TRIALS:-5}}
for value in "$gpus" "$n" "$N_ITER" "$N_TRIALS"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
done
[[ $N_WARMUP =~ ^(0|[1-9][0-9]*)$ ]] && (( n >= 4 )) || { usage; exit 2; }

julia_bin=$(command -v "${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}")
conda_bin=${CUNUMERIC_BENCH_CONDA:-${CONDA_EXE:-conda}}
conda_base=$("$conda_bin" info --base)
mpi_prefix=${IGG_MPI_PREFIX:-$conda_base/envs/igg-mpi}
if [[ ! -d "$mpi_prefix/conda-meta" ]]; then
    echo "Run $script_dir/setup_mpi.sh first to install and configure MPI." >&2
    exit 1
fi
source "$conda_base/etc/profile.d/conda.sh"
conda activate "$mpi_prefix"
trap 'conda deactivate' EXIT

# IGG_OUTPUT=<run dir>: append harness rows to <run dir>/Float32/grayscott_igg.csv.
[[ -z ${IGG_OUTPUT:-} ]] || export IGG_CSV="$IGG_OUTPUT/Float32/grayscott_igg.csv"
export OMPI_MCA_opal_cuda_support=true
export IGG_CUDAAWARE_MPI=${IGG_CUDAAWARE_MPI:-1}
if (( EUID == 0 )); then
    export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi

# Use this environment's MPI; IGG selects one GPU per node-local rank.
"$julia_bin" --startup-file=no --project="$project" -e '
    using MPIPreferences
    MPIPreferences.binary == "system" &&
        MPIPreferences.System.libmpi == joinpath(ENV["CONDA_PREFIX"], "lib", "libmpi.so") ||
        error("Run setup_mpi.sh to configure Julia for the active Conda MPI environment")
    using MPI
    project = dirname(Base.active_project())
    worker = ARGS[1]
    gpus = parse(Int, ARGS[2])
    command = `$(MPI.mpiexec()) -n $gpus $(Base.julia_cmd()) --startup-file=no --project=$project $worker $(ARGS[2:end])`
    process = run(ignorestatus(command))
    exit(success(process) ? 0 : 1)
' "$script_dir/grayscott.jl" "$gpus" "$n" "$N_ITER" "$N_WARMUP" "$N_TRIALS"
