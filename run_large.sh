#!/bin/bash
# Run the large (140 GB / H200) weak-scaling configs in configs/multi_gpu/large.
#   ./run_large.sh [run.jl args...]   e.g. ./run_large.sh --verbose
# Failed runs are listed at the end; the script keeps going.

set -uo pipefail
cd "$(dirname "$0")"

JULIA=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
LARGE=configs/multi_gpu/large
EXTRA=("$@")
FAILED=()

run() {
    echo
    echo "==> $JULIA --project=. run.jl $* ${EXTRA[*]}"
    "$JULIA" --project=. run.jl "$@" "${EXTRA[@]}" || FAILED+=("$*")
}

run --config=$LARGE/grayscott.toml
run --config=$LARGE/cg.toml
run --config=$LARGE/nas_ep_weak.toml
run --config=$LARGE/nas_ft_weak.toml

echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All runs finished."
else
    echo "Failed runs:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
