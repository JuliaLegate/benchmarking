# JACC.Multi z-slabs on every GPU count; see mg_multi.jl. The common harness
# times initial zeroing and L2 sum-of-squares, but omits NPB's Linf norm; see
# nas/README.md.

include(joinpath(@__DIR__, "..", "..", "..", "nas", "mg.jl"))
include(joinpath(@__DIR__, "mg_multi.jl"))

struct JACCNASMG
    class::String
    N::Int
    M::Int
end

function model_build_nas_mg(config::ModelWorkerConfig)
    config.T === Float64 || error("NAS MG requires Float64")
    JACC.Multi.ndev() == config.gpus || error(
        "JACC sees $(JACC.Multi.ndev()) GPU(s), but this run requested $(config.gpus)"
    )
    class = uppercase(string(get(config.kwargs, :class, "S")))
    nx, ny, _ = nas_mg_dims(nas_mg_parameters(class))
    (config.N, config.M) == (nx, ny) || error(
        "NAS MG class $class requires N=$nx, M=$ny"
    )
    return JACCNASMG(class, config.N, config.M)
end

model_initialize(b::JACCNASMG) = jacc_multi_mg(JACCMultiOps(), b.class)

model_run!(b::JACCNASMG, s::JACCMultiMG) = mgm_run!(s, b.class)
# JACC.Multi launches and copies synchronize every device, so the default
# no-op model_synchronize applies.

function model_check_correctness(b::JACCNASMG, config)
    p = nas_mg_parameters(b.class)
    sumsq = model_run!(b, model_initialize(b))
    return nas_mg_status(b.class, sqrt(sumsq/Float64(prod(nas_mg_dims(p)))))
end

function model_correctness_context(b::JACCNASMG, config)
    return (; reference="NPB-GPU", dims=nas_mg_dims(nas_mg_parameters(b.class)))
end
