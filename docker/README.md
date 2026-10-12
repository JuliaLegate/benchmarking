# Benchmark container

## Use the published image

```bash
docker pull ghcr.io/julialegate/benchmarking:latest
docker run --rm -it --gpus=all ghcr.io/julialegate/benchmarking:latest
```

Inside the container you are in the benchmark checkout with every Julia
environment, the cuPyNumeric Conda env, and IGG already set up:

```bash
julia --project=. run.jl --config=configs/single_gpu/smoke.toml   # quick check
bash other/implicitglobalgrid/run_weak_scaling.sh                  # IGG, needs 8 GPUs
```

To pick up newer commits without rebuilding:

```bash
git pull --ff-only
./deps-install/instantiate_projects.sh   # only if Project.toml files changed
```

## Build it yourself

Run from the repo root on a clean checkout whose commit is pushed to GitHub:

```bash
docker build -f docker/Dockerfile \
  --build-arg BASE_IMAGE=ghcr.io/julialegate/cunumeric.jl:dev \
  --build-arg BENCHMARK_REF="$(git branch --show-current)" \
  --build-arg BENCHMARK_COMMIT="$(git rev-parse HEAD)" \
  -t cunumeric:benchmark .
```

`BASE_IMAGE` is any cuNumeric.jl image; it is never rebuilt here.

## Build in CI

Either push a commit with `[benchmark-container]` in its message, or run the
**Benchmark container build and push** workflow manually (optionally with a
base image tag). Images are pushed as:

- `ghcr.io/julialegate/benchmarking:latest`
- `ghcr.io/julialegate/benchmarking:<commit>-benchmark-<run-id>`

To update cuNumeric.jl too: build its image first (`[container]` in
cuNumeric.jl), then run this workflow manually with that image tag.
