# JACC.Multi slabs on every GPU count; see ft_multi.jl.

include(joinpath(@__DIR__, "..", "..", "..", "nas", "ft.jl"))
include(joinpath(@__DIR__, "ft_multi.jl"))

struct JACCNASFT
    class::String
    N::Int
    M::Int
end

function model_build_nas_ft(config::ModelWorkerConfig)
    config.T === Float64 || error("NAS FT requires Float64")
    JACC.Multi.ndev() == config.gpus || error(
        "JACC sees $(JACC.Multi.ndev()) GPU(s), but this run requested $(config.gpus)"
    )
    class = uppercase(string(get(config.kwargs, :class, "S")))
    p = nas_ft_parameters(class)
    (config.N, config.M) == (p.nx, p.ny) || error(
        "NAS FT class $class requires N=$(p.nx), M=$(p.ny)"
    )
    return JACCNASFT(class, config.N, config.M)
end

model_initialize(b::JACCNASFT) = jacc_multi_ft(JACCMultiOps(), b.class)
model_run!(::JACCNASFT, s::JACCMultiFT) = ftm_run!(s)
# JACC.Multi launches, copies, and per-part FFTs synchronize every device, so
# the default no-op model_synchronize applies.

function model_check_correctness(b::JACCNASFT, config)
    return nas_ft_status(b.class, model_run!(b, model_initialize(b)))
end

function model_correctness_context(b::JACCNASFT, config)
    return (; reference="NPB-GPU", dims=(b.N, b.M, nas_ft_parameters(b.class).nz))
end
