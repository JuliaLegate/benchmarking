#!/usr/bin/env bash
# CPU-only checks of the combined tuning launcher; Julia/GPU execution is mocked.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
export JULIA="$tmp/julia" TUNE_TEST_CALLS="$tmp/calls"
export CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0
unset DAGGER_TUNE_CONFIG DAGGER_TUNE_TAG
cat > "$JULIA" <<'EOF'
#!/usr/bin/env bash
set -eu
[[ $1 != --startup-file=no ]] || shift
[[ $CUNUMERIC_BENCH_ACTIVE_MODEL == dagger && $2 == --threads=8 ]]
worker=$3 name=$4 gpus=$CUNUMERIC_BENCH_GPUS
case $worker in
    src/dagger/tune.jl) [[ $1 == --project=environments/dagger ]] ;;
    composability/tune.jl) [[ $1 == --project=environments/composability ]] ;;
    *) exit 9 ;;
esac
case $gpus in
    1) mask=7 ;; 2) mask=7,6 ;; 4) mask=7,6,5,4 ;; 8) mask=7,6,5,4,3,2,1,0 ;;
esac
[[ $CUDA_VISIBLE_DEVICES == "$mask" ]]
dry=
[[ ${@: -1} != --dry-run ]] || dry=--dry-run
printf '%s|%s|%s|%s|%s|%s\n' "$name" "$gpus" "$worker" "${DAGGER_TUNE_TAG:-}" "${DAGGER_TUNE_CONFIG:-}" "$dry" >> "$TUNE_TEST_CALLS"
[[ ${TUNE_TEST_FAIL:-0} != 1 || $name != ordinarydiffeq || $gpus != 2 ]]
EOF
chmod +x "$JULIA"

bash "$root/scripts/tune_dagger.sh" > "$tmp/all.log"
[[ $(wc -l < "$TUNE_TEST_CALLS") == 48 ]] # 28 weak + 12 strong + 8 composability.
[[ $(grep -c '|composability/tune.jl|' "$TUNE_TEST_CALLS") == 8 ]]
! grep -Eq '^(krylov_bicgstab|integrals_optimization)\|' "$TUNE_TEST_CALLS"

: > "$TUNE_TEST_CALLS"
bash "$root/scripts/tune_dagger.sh" krylov_cg ordinarydiffeq > "$tmp/selected.log"
[[ $(wc -l < "$TUNE_TEST_CALLS") == 8 ]]
[[ $(grep -c '^krylov_cg|' "$TUNE_TEST_CALLS") == 4 ]]
[[ $(grep -c '^ordinarydiffeq|' "$TUNE_TEST_CALLS") == 4 ]]

: > "$TUNE_TEST_CALLS"
bash "$root/scripts/tune_dagger.sh" --dry-run cg > "$tmp/main-dry.log"
[[ ! -s $TUNE_TEST_CALLS ]]
[[ $(grep -c '^==> cg,' "$tmp/main-dry.log") == 4 ]]
(cd "$root"; DAGGER_TUNE_CONFIG=composability/sizes_80GB.toml \
    bash scripts/tune_dagger.sh --dry-run ordinarydiffeq) > "$tmp/comp-dry.log"
[[ $(wc -l < "$TUNE_TEST_CALLS") == 4 ]]
[[ $(grep -c '|--dry-run$' "$TUNE_TEST_CALLS") == 4 ]]
grep -Fq "$(realpath "$root/composability/sizes_80GB.toml")" "$TUNE_TEST_CALLS"

: > "$TUNE_TEST_CALLS"
if TUNE_TEST_FAIL=1 bash "$root/scripts/tune_dagger.sh" ordinarydiffeq > "$tmp/failed.log" 2>&1; then exit 1; fi
[[ $(wc -l < "$TUNE_TEST_CALLS") == 4 ]] # Later GPU counts survive a failed case.
grep -q 'Failed tunes:' "$tmp/failed.log"
echo 'Combined tuning launcher checks passed'
