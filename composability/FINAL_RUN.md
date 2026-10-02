# Final H100 benchmark run

CG and the heat equation on an idle eight-H100 (80 GB) machine, with Julia 1.13,
benchmarking `main`, and cuNumeric's `codex/ode-scalar-broadcast` branch.

## Before running

- Set up with `./instantiate_projects.sh` (the container does this at build time).
- Check `nvidia-smi -L` shows eight idle H100s.
- The `G=8` CG matrix (~128 GiB Float32) is built on the host first; aim for
  512 GiB of free host memory.
- Each case uses the first `G` entries of `CUDA_VISIBLE_DEVICES`, or `0..G-1`.

```bash
cd /opt/benchmarking-composability
export CUNUMERIC_SOURCE=/opt/cuNumeric.jl
results=/opt/bench-results/final-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$results"
nvidia-smi -L > "$results/gpus.txt"; free -h > "$results/host-memory.txt"
```

## 1. Smoke test multi-GPU

Check each `results.csv`, `planned-cases.csv`, and failure log before going on.
These runs are not plotted.

```bash
BENCH_SAMPLES=2 BENCH_OUTPUT="$results/cg-smoke" \
  bash composability/krylov/run.sh weak 1024 2 4 8
ODE_SAMPLES=2 ODE_OUTPUT="$results/ode-smoke" \
  bash composability/ordinarydiffeq/run_benchmark.sh weak 128 2 4 8
```

## 2. One-GPU sweeps

```bash
julia --project=. run_composability.jl --only=krylov,ordinarydiffeq \
  --mode=single --output="$results"
```

## 3. Weak scaling

`sizes_80GB.toml` gives CG `N = 65536, 92682, 131072, 185364` and heat
`N = 16384, 23170, 32768, 46341` at 1, 2, 4, 8 GPUs.

```bash
julia --project=. run_composability.jl --only=krylov,ordinarydiffeq \
  --mode=multi --gpus=1,2,4,8 --output="$results"
```

Keep the whole result root. Commits, manifests, GPU masks, and case logs
are saved with each run.

## Vector figures

```bash
for mode in single:single multi:weak; do
  dir=${mode%%:*}; kind=${mode##*:}
  GKSwstype=100 julia --project=environments/composability composability/krylov/plot_results.jl \
    $kind "$results/$dir/krylov/results.csv" --output "$results/$dir/krylov/timings.svg"
  julia --project=environments/composability composability/ordinarydiffeq/plot_results.jl \
    $kind "$results/$dir/ordinarydiffeq/results.csv" "$results/$dir/ordinarydiffeq/timings.svg"
done
```
