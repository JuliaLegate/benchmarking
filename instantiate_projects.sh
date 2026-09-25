#!/usr/bin/env bash
set -euo pipefail

benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
julia_bin="${CUNUMERIC_BENCH_JULIA:-julia}"

source_dir="${CUNUMERIC_SOURCE:-$benchmark_dir/..}"
if [[ ! -f "$source_dir/Project.toml" || ! -f "$source_dir/lib/CNPreferences/Project.toml" ]]; then
    echo "Set CUNUMERIC_SOURCE to a cuNumeric.jl checkout (including lib/CNPreferences)." >&2
    exit 1
fi
source_dir="$(cd -- "$source_dir" && pwd)"
export CUNUMERIC_SOURCE="$source_dir"
composability_env_root="${COMPOSABILITY_ENV_ROOT:-$benchmark_dir/environments}"
mkdir -p "$composability_env_root"
composability_env_root="$(cd -- "$composability_env_root" && pwd)"

cd "$benchmark_dir"

echo "Checking Julia package registry"
"$julia_bin" --startup-file=no -e 'using Pkg; isempty(Pkg.Registry.reachable_registries()) && Pkg.Registry.add("General")'

echo "Instantiating the benchmark orchestrator"
"$julia_bin" --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

for environment in cuda jacc dagger; do
    echo "Instantiating environments/$environment"
    "$julia_bin" --project="environments/$environment" \
        -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
done

echo "Setting JACC backend to cuda"
"$julia_bin" --project="environments/jacc" \
    -e 'import Pkg; using JACC; JACC.set_backend("cuda")' # restart julia

echo "Developing local packages and instantiating environments/cunumeric"
"$julia_bin" --project="environments/cunumeric" \
    -e 'using Pkg; Pkg.develop([PackageSpec(path=ARGS[1]), PackageSpec(path=ARGS[2])]); Pkg.instantiate()' \
    "$source_dir" "$source_dir/lib/CNPreferences"

echo "Instantiating Krylov composability environment"
mkdir -p "$composability_env_root/krylov"
cp composability/krylov/Project.toml "$composability_env_root/krylov/Project.toml"
"$julia_bin" --project="$composability_env_root/krylov" \
    -e 'using Pkg; Pkg.develop([PackageSpec(path=ARGS[1]), PackageSpec(path=ARGS[2])]); Pkg.instantiate()' \
    "$source_dir" "$source_dir/lib/CNPreferences"

for workload in ordinarydiffeq integrals_optimization; do
    echo "Instantiating $workload composability environment"
    "$julia_bin" --startup-file=no "composability/$workload/setup.jl" \
        "$composability_env_root/$workload"
done
