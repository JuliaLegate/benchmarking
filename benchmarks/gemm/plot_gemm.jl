using CSV, DataFrames, Plots
include("../../plotting.jl")

function large_params()
    csv = "gemm/gemm_large.csv"
    title = "GEMM Weak Scaling [Large Test]"
    file = "gemm/gemm_weak_scaling_large.png"
    return csv, title, file
end

function small_params()
    csv = "gemm/gemm.csv"
    title = "GEMM Weak Scaling [Small Test]"
    file = "gemm/gemm_weak_scaling_small.png"
    return csv, title, file
end

function large_params_eff()
    csv = "gemm/gemm_large.csv"
    title = "GEMM Weak Scaling Efficiency [Large Test]"
    file = "gemm/gemm_weak_scaling_efficiency_large.png"
    return csv, title, file
end

function small_params_eff()
    csv = "gemm/gemm.csv"
    title = "GEMM Weak Scaling Efficiency [Small Test]"
    file = "gemm/gemm_weak_scaling_efficiency_small.png"
    return csv, title, file
end

make_weak_plot(large_params()...)
make_weak_plot(small_params()...)

make_efficiency_plot(large_params_eff()...)
make_efficiency_plot(small_params_eff()...)
