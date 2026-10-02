# Shared shell helpers for the composability launchers.

gpu_mask_for_count() {
    local count=$1 i mask
    local -a devices=()
    if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
        [[ -n $CUDA_VISIBLE_DEVICES ]] || { echo "CUDA_VISIBLE_DEVICES is empty" >&2; return 2; }
        IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
        (( ${#devices[@]} >= count )) || {
            echo "CUDA_VISIBLE_DEVICES has fewer than $count devices" >&2
            return 2
        }
    else
        for ((i=0; i<count; i++)); do devices+=("$i"); done
    fi
    printf -v mask '%s,' "${devices[@]:0:count}"
    printf '%s\n' "${mask%,}"
}

# Run one case, save its output, and return its single RESULT payload on stdout.
# Call this in an if/else so a failed case does not stop the sweep.
run_logged_case() {
    local log=$1 exit_code=0 row
    shift
    "$@" > "$log" 2>&1 || exit_code=$?

    if [[ $exit_code == 0 ]]; then
        if row=$(awk '/^RESULT,/ { row=$0; count++ } END { if (count != 1) exit 1; print row }' "$log" 2>/dev/null); then
            printf '%s\n' "${row#RESULT,}"
            return 0
        fi
        echo "Missing or duplicate RESULT: $log" >&2
    else
        echo "Case failed with exit status $exit_code: $log" >&2
    fi
    if [[ -s $log ]]; then
        echo "Last 20 lines of $log:" >&2
        tail -n 20 "$log" >&2 || true
    fi
    return 1
}
