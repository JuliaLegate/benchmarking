#!/usr/bin/env bash
# Install IGG's Julia environment and configure its Conda MPI/CUDA dependencies.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project="$script_dir/../environments/implicitglobalgrid"
julia_bin=$(command -v "${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}")
conda_bin=${CUNUMERIC_BENCH_CONDA:-${CONDA_EXE:-conda}}
command -v "$conda_bin" >/dev/null 2>&1 || {
    echo "Conda is required for IGG. Add it to PATH or set CUNUMERIC_BENCH_CONDA." >&2
    exit 1
}
conda_base=$("$conda_bin" info --base)
mpi_prefix=${IGG_MPI_PREFIX:-$conda_base/envs/igg-mpi}
cuda_version=${IGG_CUDA_VERSION:-${CUDA_VERSION_MAJOR_MINOR:-13.0}}

# Re-running setup updates the existing environment without replacing it.
conda_action=create
[[ ! -d "$mpi_prefix/conda-meta" ]] || conda_action=install
"$conda_bin" "$conda_action" -y --prefix "$mpi_prefix" --override-channels \
    -c conda-forge 'openmpi=5' ucx "cuda-version=$cuda_version"

source "$conda_base/etc/profile.d/conda.sh"
conda activate "$mpi_prefix"
trap 'conda deactivate' EXIT

"$julia_bin" --startup-file=no --project="$project" -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
"$julia_bin" --startup-file=no --project="$project" -e '
    using MPIPreferences
    prefix = ENV["CONDA_PREFIX"]
    MPIPreferences.use_system_binary(
        library_names=[joinpath(prefix, "lib", "libmpi.so")],
        extra_paths=[joinpath(prefix, "lib")],
        mpiexec=joinpath(prefix, "bin", "mpiexec"),
    )
'

# Match the old setup only when a complete local CUDA toolkit is installed.
if [[ ${IGG_LOCAL_CUDA:-0} == 1 ]]; then
    "$julia_bin" --startup-file=no --project="$project" -e \
        'using CUDA; CUDA.set_runtime_version!(local_toolkit=true)'
fi

# MPI and CUDA preferences take effect in a fresh Julia process.
"$julia_bin" --startup-file=no --project="$project" -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
echo "IGG setup complete. Run $script_dir/../other/implicitglobalgrid/run_benchmark.sh GPUS N [N_ITER] [N_WARMUP] [N_TRIALS]."
