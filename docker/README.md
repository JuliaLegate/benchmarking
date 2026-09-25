# Benchmark container

From the benchmarking repository root:

```bash
docker build -f docker/Dockerfile \
  --build-arg BASE_IMAGE=ghcr.io/julialegate/cunumeric.jl:dev \
  --build-arg BENCHMARK_REF="$(git branch --show-current)" \
  --build-arg BENCHMARK_COMMIT="$(git rev-parse HEAD)" \
  -t cunumeric:benchmark .
docker run --rm --gpus=all cunumeric:benchmark \
  julia --project=. run.jl --config=configs/single_gpu/smoke.toml
```

The image build runs `instantiate_projects.sh`, so the composability Julia
environments are already initialized at `/opt/bench-envs/krylov`,
`/opt/bench-envs/ordinarydiffeq`, and
`/opt/bench-envs/integrals_optimization`. The composability launchers select
them automatically. Build from a cuNumeric base image containing the Krylov
extension and ODE scalar-broadcast support before running those workloads.

The image also contains a clean Git checkout of the exact benchmark commit
used for the build. From the image's default working directory,
`git pull --ff-only` can fetch later commits on the same branch. The build
requires a clean checkout and the two `BENCHMARK_*` build arguments above;
the CI workflow supplies them automatically. If package projects change after
pulling, rerun `./instantiate_projects.sh` to update the Julia environments.

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
