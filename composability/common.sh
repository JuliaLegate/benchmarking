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
    local log=$1 memory_log=$2 exit_code=0 row
    shift 2
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop-ms=250 > "$memory_log" 2>&1 &
    local monitor_pid=$!
    "$@" > "$log" 2>&1 || exit_code=$?
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true

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

# Missing telemetry must never discard a valid timing row. Return no value and
# a failure status so callers can record a blank peak and continue the sweep.
read_peak_memory() {
    local log=$1 peak
    if peak=$(awk '$1 ~ /^[0-9]+$/ { seen=1; if ($1 > peak) peak=$1 } END { if (seen) print peak+0 }' "$log" 2>/dev/null) && [[ -n $peak ]]; then
        printf '%s\n' "$peak"
        return 0
    fi
    echo "Memory samples unavailable: $log; keeping timing results and continuing" >&2
    return 1
}
