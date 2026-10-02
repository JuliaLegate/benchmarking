#!/usr/bin/env bash
# CPU-only integration checks: bash test/composability_tuning.sh
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
export JULIA="$tmp/mock-julia"
export DAGGER_TUNE_OUTPUT="$tmp/results"
export CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0
export DAGGER_TUNE_BLOCKS='1 2 4'
export DAGGER_TUNE_GPUS='1 2 4 8'
export DAGGER_TUNE_SAMPLES=3
export BENCH_SOLVERS=cg,bicgstab
cat > "$JULIA" <<'EOF'
#!/usr/bin/env bash
set -eu
worker=$4
[[ $1 == --startup-file=no && $2 == --project=* && $3 == --threads=8 ]]
[[ $5 == Dagger ]]
blocks=$DAGGER_BLOCKS_PER_GPU
case $blocks in 1) mean=10 ;; 2) mean=5 ;; 4) mean=30 ;; esac
case $worker in
    */krylov.jl)
        g=$BENCH_GPUS n=$8 count=$BENCH_SAMPLES
        row="Dagger,$6,stock,Float32,$g,$n,12,$mean,0,$mean,$mean,$mean,0.000001,$mean;$mean;$mean"
        [[ $n == 4096 ]] ;;
    */benchmark_heat.jl)
        g=$ODE_GPUS n=$6 count=$ODE_SAMPLES
        row="Dagger,Float32,$g,$n,20,$mean,0,$mean,$mean,$mean,0.000001,$mean;$mean;$mean"
        [[ $n == 1024 ]] ;;
    */benchmark.jl)
        g=$INTOPT_GPUS n=$6 count=$INTOPT_SAMPLES
        row="Dagger,Float32,$g,$n,4,12,80,90,$mean,0,$mean,$mean,$mean,1,0.001,0.01,$mean;$mean;$mean"
        [[ $n == 512 ]] ;;
    *) exit 9 ;;
esac
[[ $count == 3 ]]
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
[[ $(grep -c '^==>' "$tmp/dry.log") == 48 ]]
env -u DAGGER_TUNE_BLOCKS bash "$root/composability/tune_dagger.sh" --dry-run > "$tmp/default.log"
[[ $(grep -c '^==>' "$tmp/default.log") == 112 ]]
bash "$root/composability/tune_dagger.sh" > "$tmp/run.log"
[[ $(wc -l < "$DAGGER_TUNE_OUTPUT/results.csv") == 49 ]]
[[ $(wc -l < "$DAGGER_TUNE_OUTPUT/best.csv") == 17 ]]
awk -F, 'NR > 1 && ($6 != 2 || $7 != 5) { exit 1 }' "$DAGGER_TUNE_OUTPUT/best.csv"
# A second run must leave the original files intact.
if bash "$root/composability/tune_dagger.sh" > "$tmp/repeat.log" 2>&1; then exit 1; fi
[[ $(wc -l < "$DAGGER_TUNE_OUTPUT/results.csv") == 49 ]]

# Failure and malformed/duplicate results cannot win; later candidates still run.
for fault in MOCK_FAIL MOCK_BAD MOCK_DUPLICATE; do
    export DAGGER_TUNE_OUTPUT="$tmp/$fault"
    if env "$fault=1" bash "$root/composability/tune_dagger.sh" ordinarydiffeq > "$tmp/$fault.log" 2>&1; then exit 1; fi
    [[ $(wc -l < "$DAGGER_TUNE_OUTPUT/results.csv") == 9 ]]
    [[ $(wc -l < "$DAGGER_TUNE_OUTPUT/best.csv") == 5 ]]
    awk -F, 'NR > 1 && $6 != 1 { exit 1 }' "$DAGGER_TUNE_OUTPUT/best.csv"
done

# The 4x stop suppresses later candidates without marking a failure.
export DAGGER_TUNE_OUTPUT="$tmp/early-stop" DAGGER_TUNE_BLOCKS='1 2 4 8'
bash "$root/composability/tune_dagger.sh" ordinarydiffeq > "$tmp/early.log"
[[ $(wc -l < "$DAGGER_TUNE_OUTPUT/results.csv") == 13 ]]
for invalid in 'DAGGER_TUNE_GPUS=3' 'DAGGER_TUNE_BLOCKS=2 1' 'DAGGER_TUNE_SAMPLES=1' 'BENCH_SOLVERS=unknown'; do
    if env "$invalid" bash "$root/composability/tune_dagger.sh" --dry-run > "$tmp/invalid.log" 2>&1; then exit 1; fi
done
if CUDA_VISIBLE_DEVICES=0 bash "$root/composability/tune_dagger.sh" --dry-run ordinarydiffeq > "$tmp/mask.log" 2>&1; then exit 1; fi
echo 'Composability tuning checks passed'
