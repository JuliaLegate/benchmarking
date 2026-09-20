# Benchmark container

From the benchmarking repository root:

```bash
docker build -f docker/Dockerfile \
  --build-arg BASE_IMAGE=ghcr.io/julialegate/cunumeric.jl:dev \
  -t cunumeric:benchmark .
docker run --rm --gpus=all cunumeric:benchmark \
  julia --project=. run.jl --config=benchmarks_smoke.toml
```

CI is opt-in: push a commit containing `[benchmark-container]` in this repo,
or run **Benchmark container build and push** manually. Manual runs accept an
existing base image tag/digest; commit-triggered runs use `cunumeric.jl:dev`.
The base image is never rebuilt here. The Conda CUDA override comes from it.

Images are published as `ghcr.io/julialegate/benchmarking:dev-benchmark` and
`<commit>-benchmark-<run-id>`. Existing cuNumeric.jl image tags are left untouched.
The repository needs a `self-hosted, linux, x64` runner with Docker access.

To refresh both layers, first build `[container]` in cuNumeric.jl, then launch
this workflow with the resulting immutable base image tag. Putting both markers
in one cuNumeric.jl commit no longer builds both images.
