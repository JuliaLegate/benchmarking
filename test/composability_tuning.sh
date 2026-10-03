#!/usr/bin/env bash
# CPU-only integration checks (Julia stdlib required): bash test/composability_tuning.sh
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
export REAL_JULIA=${JULIA:-${CUNUMERIC_BENCH_JULIA:-julia}}
export JULIA="$tmp/mock-julia"
unset DAGGER_TUNE_CONFIG
export DAGGER_TUNE_OUTPUT="$tmp/results"
export CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0
cat > "$JULIA" <<'EOF'
#!/usr/bin/env bash
set -eu
# Exercise real TOML parsing; mock only the GPU workers.
if [[ ${2:-} == -e ]]; then exec "$REAL_JULIA" "$@"; fi
worker=$4
[[ $1 == --startup-file=no && $2 == --project=* && $3 == --threads=8 ]]
[[ $5 == Dagger ]]
[[ $COMPOSABILITY_TUNE == 1 ]]
blocks=$DAGGER_BLOCKS_PER_GPU
case $blocks in 1) mean=10 ;; 2) mean=5 ;; 4) mean=30 ;; *) mean=50 ;; esac
case $worker in
    krylov/krylov.jl)
        g=$BENCH_GPUS n=$8
        row="Dagger,$6,stock,Float32,$g,$n,5,$mean,0,$mean,$mean,$mean,0.001,$mean;$mean"
        ;;
    ordinarydiffeq/benchmark_heat.jl)
        g=$ODE_GPUS n=$6
        row="Dagger,Float32,$g,$n,5,$mean,0,$mean,$mean,$mean,0.000001,$mean;$mean"
        ;;
    integrals_optimization/benchmark.jl)
        g=$INTOPT_GPUS n=$6
        row="Dagger,Float32,$g,$n,4,12,5,10,$mean,0,$mean,$mean,$mean,1,0.001,0.2,$mean;$mean"
        ;;
    *) exit 9 ;;
esac
case $g in
    1) mask=7 ;; 2) mask=7,6 ;; 4) mask=7,6,5,4 ;; 8) mask=7,6,5,4,3,2,1,0 ;;
esac
[[ $CUDA_VISIBLE_DEVICES == "$mask" ]]
if [[ ${MOCK_FAIL:-0} == 1 && $blocks == 2 ]]; then echo 'mock worker failure'; exit 7; fi
if [[ ${MOCK_BAD:-0} == 1 && $blocks == 2 ]]; then row=malformed; fi
echo "RESULT,$row"
if [[ ${MOCK_DUPLICATE:-0} == 1 && $blocks == 2 ]]; then echo "RESULT,$row"; fi
EOF
chmod +x "$JULIA"

bash "$root/composability/tune_dagger.sh" --dry-run > "$tmp/dry.log"
[[ ! -e $DAGGER_TUNE_OUTPUT ]]
[[ $(grep -c '^==>' "$tmp/dry.log") == 84 ]]
for pair in '1 130000' '2 183848' '4 260000' '8 367696'; do
    read -r g n <<< "$pair"
    grep -q "^==> krylov-cg G=$g N=$n " "$tmp/dry.log"
done
for pair in '1 32768' '2 46341' '4 65536' '8 92682'; do
    read -r g n <<< "$pair"
    grep -q "^==> heat G=$g N=$n " "$tmp/dry.log"
done
# Config paths are relative to the caller, even though the tuner changes directory.
(cd "$root"; DAGGER_TUNE_CONFIG=composability/sizes_80GB.toml \
    bash composability/tune_dagger.sh --dry-run) > "$tmp/h100.log"
grep -q '^==> krylov-cg G=8 N=185364 ' "$tmp/h100.log"
grep -q '^==> heat G=8 N=46341 ' "$tmp/h100.log"
bash "$root/composability/tune_dagger.sh" > "$tmp/run.log"
cmp "$root/composability/sizes_141GB.toml" "$DAGGER_TUNE_OUTPUT/sizes.toml"
[[ $(wc -l < "$DAGGER_TUNE_OUTPUT/results.csv") == 37 ]]
[[ $(wc -l < "$DAGGER_TUNE_OUTPUT/best.csv") == 13 ]]
awk -F, 'NR > 1 && ($4 != 2 || $5 != 5) { exit 1 }' "$DAGGER_TUNE_OUTPUT/best.csv"
# A second run must leave the original files intact.
if bash "$root/composability/tune_dagger.sh" > "$tmp/repeat.log" 2>&1; then exit 1; fi
[[ $(wc -l < "$DAGGER_TUNE_OUTPUT/results.csv") == 37 ]]

# Failure and malformed/duplicate results cannot win; later candidates still run.
for fault in MOCK_FAIL MOCK_BAD MOCK_DUPLICATE; do
    export DAGGER_TUNE_OUTPUT="$tmp/$fault"
    if env "$fault=1" bash "$root/composability/tune_dagger.sh" ordinarydiffeq > "$tmp/$fault.log" 2>&1; then exit 1; fi
    [[ $(wc -l < "$DAGGER_TUNE_OUTPUT/results.csv") == 13 ]]
    [[ $(wc -l < "$DAGGER_TUNE_OUTPUT/best.csv") == 5 ]]
    awk -F, 'NR > 1 && $4 != 1 { exit 1 }' "$DAGGER_TUNE_OUTPUT/best.csv"
done

if bash "$root/composability/tune_dagger.sh" --dry-run unknown > "$tmp/invalid.log" 2>&1; then exit 1; fi
if CUDA_VISIBLE_DEVICES=0 bash "$root/composability/tune_dagger.sh" --dry-run ordinarydiffeq > "$tmp/mask.log" 2>&1; then exit 1; fi
echo 'Composability tuning checks passed'
