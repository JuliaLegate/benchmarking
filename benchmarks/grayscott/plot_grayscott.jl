using CSV, DataFrames, Plots
include("../plotting.jl")

function large_params()
    csv = "grayscott/grayscott_large.csv"
    title = "grayscott Weak Scaling [Large Test]"
    file = "grayscott/grayscott_weak_scaling_large.png"
    return csv, title, file
end

function small_params()
    csv = "grayscott/grayscott.csv"
    title = "grayscott Weak Scaling [Small Test]"
    file = "grayscott/grayscott_weak_scaling_small.png"
    return csv, title, file
end

function large_params_eff()
    csv = "grayscott/grayscott_large.csv"
    title = "grayscott Weak Scaling Efficiency [Large Test]"
    file = "grayscott/grayscott_weak_scaling_efficiency_large.png"
    return csv, title, file
end

function small_params_eff()
    csv = "grayscott/grayscott.csv"
    title = "grayscott Weak Scaling Efficiency [Small Test]"
    file = "grayscott/grayscott_weak_scaling_efficiency_small.png"
    return csv, title, file
end


function gc_params()
    csv = "grayscott/grayscott_gc.csv"
    title = "grayscott GC Weak Scaling [Small Test]"
    file = "grayscott/grayscott_gc_weak_scaling_small.png"
    return csv, title, file
end

function gc_params_eff()
    csv = "grayscott/grayscott_gc.csv"
    title = "grayscott GC Weak Scaling Efficiency [Small Test]"
    file = "grayscott/grayscott_gc_weak_scaling_efficiency_small.png"
    return csv, title, file
end

# make_weak_plot(large_params()...)
make_weak_plot(small_params()...)
make_weak_plot(gc_params()...)


# make_efficiency_plot(large_params_eff()...)
make_efficiency_plot(small_params_eff()...)
make_efficiency_plot(gc_params_eff()...)