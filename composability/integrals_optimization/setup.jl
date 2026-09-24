# Create a standalone environment for the Integrals + Optimization probe.
using Pkg

length(ARGS) == 1 || error("Usage: CUNUMERIC_SOURCE=/path/to/cuNumeric.jl julia setup.jl /path/to/env")
source = get(ENV, "CUNUMERIC_SOURCE", "")
isfile(joinpath(source, "Project.toml")) &&
    isfile(joinpath(source, "lib", "CNPreferences", "Project.toml")) ||
    error("CUNUMERIC_SOURCE must name a cuNumeric.jl checkout")

Pkg.activate(abspath(ARGS[1]))
Pkg.develop([
    PackageSpec(path=abspath(source)),
    PackageSpec(path=abspath(joinpath(source, "lib", "CNPreferences"))),
])
Pkg.add([
    PackageSpec(name="Integrals"),
    PackageSpec(name="FastGaussQuadrature"),
    PackageSpec(name="Optimization"),
    PackageSpec(name="OptimizationOptimJL"),
    PackageSpec(name="CUDA", version="6.4.0"),
    PackageSpec(name="Plots"),
])
Pkg.precompile()
Pkg.status()
