# Create a standalone environment for the OrdinaryDiffEq composition probe.
using Pkg

length(ARGS) == 1 || error("Usage: CUNUMERIC_SOURCE=/path/to/cuNumeric.jl julia setup.jl /path/to/ode-env")
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
    PackageSpec(name="OrdinaryDiffEqLowStorageRK"),
    PackageSpec(name="SciMLBase"),
    PackageSpec(name="CUDA", version="6.4.0"),
])
Pkg.precompile()
Pkg.status()
