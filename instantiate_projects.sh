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

cd "$benchmark_dir"

echo "Checking Julia package registry"
"$julia_bin" --startup-file=no -e 'using Pkg; isempty(Pkg.Registry.reachable_registries()) && Pkg.Registry.add("General")'

echo "Instantiating the benchmark orchestrator"
"$julia_bin" --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

for environment in cuda jacc dagger; do
    echo "Instantiating environments/$environment"
    "$julia_bin" --project="environments/$environment" -e '
        using Pkg
        Pkg.resolve()
        # Existing manifests may still track the old Dagger development branch.
        for (uuid, info) in Pkg.dependencies()
            if info.name == "Dagger" && info.is_direct_dep
                # Pkg.free first unpins a pinned repo; a second call releases it.
                info.is_pinned && Pkg.free(PackageSpec(uuid=uuid))
                !info.is_tracking_registry && Pkg.free(PackageSpec(uuid=uuid))
            end
        end
        Pkg.instantiate()
    '
done

echo "Installing ImplicitGlobalGrid's Conda MPI environment and Julia packages"
JULIA="$julia_bin" bash "$benchmark_dir/other/implicitglobalgrid/setup_igg.sh"

echo "Setting JACC backend to cuda"
"$julia_bin" --project="environments/jacc" \
    -e 'import Pkg; using JACC; JACC.set_backend("cuda")' # restart julia

echo "Developing local packages and instantiating environments/cunumeric"
"$julia_bin" --project="environments/cunumeric" \
    -e 'using Pkg; Pkg.develop([PackageSpec(path=ARGS[1]), PackageSpec(path=ARGS[2])]); Pkg.instantiate()' \
    "$source_dir" "$source_dir/lib/CNPreferences"

echo "Developing local packages and instantiating environments/composability"
"$julia_bin" --project="environments/composability" \
    -e 'using Pkg; Pkg.develop([PackageSpec(path=ARGS[1]), PackageSpec(path=ARGS[2])]); Pkg.instantiate()' \
    "$source_dir" "$source_dir/lib/CNPreferences"
