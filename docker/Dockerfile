ARG BASE_IMAGE=ghcr.io/julialegate/cunumeric.jl:dev
FROM ${BASE_IMAGE} AS benchmark-deps

ARG MINIFORGE_VERSION=25.3.1-0
ENV CONDA_DIR=/opt/conda
ENV PATH="${CONDA_DIR}/bin:${PATH}"
ENV CUNUMERIC_BENCH_CONDA="${CONDA_DIR}/bin/conda"
ENV CUNUMERIC_SOURCE=/opt/cuNumeric.jl
ENV COMPOSABILITY_ENV_ROOT=/opt/bench-envs

COPY docker/profile.sh /etc/profile.d/cunumeric.sh
COPY docker/entrypoint.sh /usr/local/bin/cunumeric-entrypoint
RUN chmod +x /usr/local/bin/cunumeric-entrypoint

RUN wget --no-verbose \
    "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && conda clean --all --yes

WORKDIR /opt/cuNumeric.jl/benchmark
COPY Project.toml ./Project.toml
COPY environments/cuda/Project.toml environments/cuda/Project.toml
COPY environments/cunumeric/Project.toml environments/cunumeric/Project.toml
COPY environments/dagger/Project.toml environments/dagger/Project.toml
COPY environments/jacc/Project.toml environments/jacc/Project.toml
COPY composability/krylov/Project.toml composability/krylov/Project.toml
COPY composability/ordinarydiffeq/setup.jl composability/ordinarydiffeq/setup.jl
COPY composability/integrals_optimization/setup.jl composability/integrals_optimization/setup.jl
COPY instantiate_projects.sh install_cupynumeric.sh ./

ARG DEPENDENCY_REFRESH=0
RUN echo "Benchmark dependency refresh: ${DEPENDENCY_REFRESH}" \
    && julia -e 'using Pkg; Pkg.Registry.add("General")' \
    && JULIA_PKG_PRECOMPILE_AUTO=0 CUNUMERIC_BENCH_PRECOMPILE=0 ./instantiate_projects.sh \
    && rm -rf /depot/registries /depot/scratchspaces /depot/logs

# Docker builds do not expose the host NVIDIA driver, so Conda cannot discover
# its __cuda virtual package. Solve against the same CUDA version as the base
# image instead of pretending CUDA is unavailable.
RUN CONDA_OVERRIDE_CUDA="${CUDA_VERSION_MAJOR_MINOR}" ./install_cupynumeric.sh \
    && conda clean --all --yes

# The cuNumeric TensorOperations extension links the CUDA Legate library while
# precompiling. The driver stub is available through the base image's env file;
# runtime startup itself must remain disabled because builds have no GPU.
RUN source /etc/profile.d/cunumeric.sh \
    && for project in . environments/cuda environments/dagger; do \
        julia --project="$project" -e 'using Pkg; Pkg.precompile()'; \
    done \
    && julia --project=environments/jacc \
        -e 'using Pkg; Pkg.precompile(); import JACC; JACC.@init_backend' \
    && LEGATE_SKIP_RUNTIME=true julia --project=environments/cunumeric \
        -e 'using Pkg; Pkg.precompile()' \
    && for project in "$COMPOSABILITY_ENV_ROOT"/krylov "$COMPOSABILITY_ENV_ROOT"/ordinarydiffeq "$COMPOSABILITY_ENV_ROOT"/integrals_optimization; do \
        LEGATE_SKIP_RUNTIME=true julia --project="$project" -e 'using Pkg; Pkg.precompile()'; \
    done \
    && rm -rf /depot/registries /depot/scratchspaces /depot/logs

FROM benchmark-deps AS benchmark

COPY . /opt/cuNumeric.jl/benchmark/

# Add clean public Git metadata for the exact source copied above. Never copy
# the build-context .git directory: CI checkouts may contain an auth header.
ARG BENCHMARK_REF=main
ARG BENCHMARK_COMMIT
RUN test -n "$BENCHMARK_COMMIT" \
    && test ! -e .git \
    && git clone --quiet --no-checkout --depth 1 --branch "$BENCHMARK_REF" \
        https://github.com/JuliaLegate/benchmarking.git /tmp/benchmark-git \
    && git -C /tmp/benchmark-git fetch --quiet --depth 1 origin "$BENCHMARK_COMMIT" \
    && mv /tmp/benchmark-git/.git .git \
    && git config core.autocrlf input \
    && git reset --mixed --quiet "$BENCHMARK_COMMIT" \
    && git branch --set-upstream-to="origin/$BENCHMARK_REF" \
    && git diff --quiet \
    && git diff --cached --quiet \
    && rm -rf /tmp/benchmark-git

# Pkg.develop records absolute [sources] paths in Project.toml, which the COPY
# above replaced with the committed (source-free) file. Re-record the image's
# cuNumeric.jl checkout and hide that local edit from git status. The profile
# supplies the CUDA driver stub, since builds have no GPU.
RUN source /etc/profile.d/cunumeric.sh \
    && julia -e 'using Pkg; Pkg.Registry.add("General")' \
    && julia --project=environments/cunumeric -e \
        'using Pkg; Pkg.develop([PackageSpec(path=ARGS[1]), PackageSpec(path=ARGS[2])])' \
        "$CUNUMERIC_SOURCE" "$CUNUMERIC_SOURCE/lib/CNPreferences" \
    && git update-index --skip-worktree environments/cunumeric/Project.toml \
    && LEGATE_SKIP_RUNTIME=true julia --project=environments/cunumeric -e 'import cuNumeric' \
    && rm -rf /depot/registries /depot/scratchspaces /depot/logs

LABEL org.opencontainers.image.title="cuNumeric.jl benchmark suite" \
    org.opencontainers.image.source="https://github.com/JuliaLegate/benchmarking" \
    org.opencontainers.image.revision="$BENCHMARK_COMMIT" \
    org.opencontainers.image.description="Benchmark-ready cuNumeric.jl image with CUDA.jl, JACC.jl, Dagger.jl, and cuPyNumeric"

ENTRYPOINT ["/usr/local/bin/cunumeric-entrypoint"]
CMD ["/bin/bash"]
WORKDIR /opt/cuNumeric.jl/benchmark
