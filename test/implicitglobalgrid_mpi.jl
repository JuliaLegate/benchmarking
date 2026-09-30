# Run with --project=environments/implicitglobalgrid. Uses real MPI/IGG halos
# and the production worker on CPU, so it also runs on GPU-free setup CI.
using Test, MPI

if !isempty(ARGS) && first(ARGS) == "--worker"
    include(joinpath(@__DIR__, "..", "other", "implicitglobalgrid", "grayscott.jl"))
    main(ARGS[2:end]; cpu=true)
else
    project = dirname(Base.active_project())
    @testset "Real ImplicitGlobalGrid MPI halo exchange" begin
        for ranks in (1, 2, 4), n in (8, 160)
            cmd = `$(MPI.mpiexec()) -n $ranks $(Base.julia_cmd()) --startup-file=no --project=$project $(@__FILE__) --worker $ranks $n 5 2 --check`
            @test success(pipeline(cmd; stdout, stderr))
        end
    end
end
