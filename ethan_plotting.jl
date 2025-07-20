using CSV, DataFrames, CairoMakie

const NVIDIA_GREEN = "#76B900"
const JULIA_PURPLE = "#9558B2"
const PYTHON_BLUE = " #4B8BBE"
const PINK = "#f03e62"
const DARK_PINK = "#ed0534"
const LIGHT_PINK = "#f08b9f"

const ORANGE = "#ff8c00"
const GOLD = "#ffca38"

function make_efficiency_plot(csv, file, colors)
    df = CSV.read(csv, DataFrame)
    sort!(df, [:model, :gpus])

    # Compute efficiency = T(1) / T(Np) for each model
    df_eff = DataFrame()
    for g in groupby(df, :model)
        t1 = g.mean_time_ms[g.gpus .== 1][1]
        eff = t1 ./ g.mean_time_ms
        df_tmp = DataFrame(model=g.model, gpus=g.gpus, efficiency=eff)
        append!(df_eff, df_tmp)
    end

    size_in_inches = (3, 2.25)
    dpi = 300
    size_in_pixels = size_in_inches .* dpi

    efficiency_fig = Figure(resolution = size_in_pixels);
    ax = Axis(efficiency_fig[1,1], xlabel = "Number of GPUs", ylabel = "Parallel Efficiency",
        ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
        xticks = [1, 2, 4, 8], xgridvisible = false, ygridvisible = false, xscale = log2,
        xticksmirrored = true, yticksmirrored = true, xticklabelpad = 4, xtickalign=1, ytickalign = 1)

    ylims!(0, 1.2)

    plots = []
    labels = []

    for (i,g) in enumerate(groupby(df_eff, :model))
        sort!(g, :gpus)
        label = g.model[1]
        marker = (label == "cuNumeric.jl") ? :star6 : :circle
        overdraw = (label == "cuNumeric.jl") ? true : false
        s = scatterlines!(g.gpus, g.efficiency, marker = marker, overdraw = overdraw, strokewidth = 2, linewidth = 2,
                            markersize = 35, color = colors[i], strokecolor = :black)
        push!(plots, s)
        push!(labels, g.model[1])
    end

    axislegend(ax, plots, labels, position = :lb,
                 patchlabelgap = 12, labelsize = 35, framevisible = false)

    save(file, efficiency_fig)
    
    println("Saved weak scaling efficiency plot to $(file)")
end 


function make_weak_plot(csv, file, colors, markers; fp32_peak = 19500.0)
    df = CSV.read(csv, DataFrame)
    # Compute total problem size and per-GPU size
    df.size = df.n .* df.m
    df.size_per_gpu = df.size ./ df.gpus
    df.gflops_per_gpu = df.gflops ./ df.gpus

    # Sort so GPUs increase left-to-right
    sort!(df, [:model, :gpus])

    size_in_inches = (3, 2.25)
    dpi = 300
    size_in_pixels = size_in_inches .* dpi

    gflops_fig = Figure(resolution = size_in_pixels);
    ax = Axis(gflops_fig[1,1], xlabel = "Number of GPUs", ylabel = "GFLOPS / GPU",
        ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
        xticks = sort(unique(df.gpus)), yticks = ([1e2,1e3, 1e4, 1e5], ["10²","10³", "10⁴", "10⁵"]),
        xgridvisible = false, ygridvisible = false, xscale = log2,
        yscale = log10, xticksmirrored = true, yticksmirrored = true, xticklabelpad = 4, xtickalign=1, ytickalign = 1)
    ylims!(1,5e5)

    plots = Any[]
    labels = []

    
    h = hlines!(fp32_peak, xmin = 0.0, xmax = 100.0, color = :black, linestyle = (:dash, :loose), linewidth = 3)

    push!(plots, h)
    push!(labels, "FP32 Theoretical Bandwidth")


    # Plot one line per model
    for (i,g) in enumerate(groupby(df, :model))
        label = g.model[1]
        sort!(g, :gpus)

        marker = (label == "cuNumeric.jl") ? :star6 : markers[i]
        overdraw = (label == "cuNumeric.jl") ? true : false

        s = scatterlines!(g.gpus, g.gflops_per_gpu, marker = marker, overdraw = overdraw, strokewidth = 2, linewidth = 2,
                            markersize = 35, color = colors[i], strokecolor = :black)
        push!(plots, s)
        push!(labels, g.model[1])
    end

    axislegend(ax, plots, labels, position = :rt,
                 patchlabelgap = 12, labelsize = 30, framevisible = false)

    save(file, gflops_fig)
    println("Saved plot to $(file)")
end


function make_efficiency_plot_grayscott(csv, file, colors, markers)
    df = CSV.read(csv, DataFrame)
    sort!(df, [:model, :gpus])

    # Compute efficiency = T(1) / T(Np) for each model
    df_eff = DataFrame()
    for g in groupby(df, :model)
        t1 = g.mean_time_ms[g.gpus .== 1][1]
        eff = t1 ./ g.mean_time_ms
        df_tmp = DataFrame(model=g.model, gpus=g.gpus, efficiency=eff)
        append!(df_eff, df_tmp)
    end

    size_in_inches = (3, 2.25)
    dpi = 300
    size_in_pixels = size_in_inches .* dpi

    efficiency_fig = Figure(resolution = size_in_pixels);
    ax = Axis(efficiency_fig[1,1], xlabel = "Number of GPUs", ylabel = "Parallel Efficiency",
        ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
        xticks = [1, 2, 4, 8], xgridvisible = false, ygridvisible = false, xscale = log2,
        xticksmirrored = true, yticksmirrored = true, xticklabelpad = 4, xtickalign=1, ytickalign = 1)

    ylims!(0, 1.2)

    plots = []
    labels = []

    for (i,g) in enumerate(groupby(df_eff, :model))
        sort!(g, :gpus)
        label = g.model[1]
        marker = (label == "cuNumeric.jl") ? :star6 : :circle
        overdraw = (label == "cuNumeric.jl") ? true : false
        s = scatterlines!(g.gpus, g.efficiency, marker = marker, overdraw = overdraw, strokewidth = 2, linewidth = 2,
                            markersize = 35, color = colors[i], strokecolor = :black)
        push!(plots, s)
        push!(labels, g.model[1])
    end

    axislegend(ax, plots[2:end], labels[2:end], "cuNumeric.jl",
         position = :lb, patchlabelgap = 12, labelsize = 30, framevisible = false,
         labelhalign = :center, colgap = 25, titlesize = 30)

    axislegend(ax, [plots[1]], [labels[1]], position = :rb, 
                labelsize = 30, framevisible = false, patchlabelgap = 12)

    save(file, efficiency_fig)
    
    println("Saved weak scaling efficiency plot to $(file)")
end 


# make_weak_plot_grayscott(grayscott_csv, grascott_weak, [PYTHON_BLUE, DARK_PINK, PINK, LIGHT_PINK], [:circle, :star6, :star6, :star6])

function make_weak_plot_grayscott(csv, file, colors, markers; fp32_peak = 19500.0)
    df = CSV.read(csv, DataFrame)
    # Compute total problem size and per-GPU size
    df.size = df.n .* df.m
    df.size_per_gpu = df.size ./ df.gpus
    df.gflops_per_gpu = df.gflops ./ df.gpus

    # Sort so GPUs increase left-to-right
    sort!(df, [:model, :gpus])

    size_in_inches = (3, 2.25)
    dpi = 300
    size_in_pixels = size_in_inches .* dpi

    gflops_fig = Figure(resolution = size_in_pixels);
    ax = Axis(gflops_fig[1,1], xlabel = "Number of GPUs", ylabel = "GFLOPS / GPU",
        ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
        xticks = sort(unique(df.gpus)), yticks = ([1e2,1e3, 1e4, 1e5], ["10²","10³", "10⁴", "10⁵"]),
        xgridvisible = false, ygridvisible = false, xscale = log2,
        yscale = log10, xticksmirrored = true, yticksmirrored = true, xticklabelpad = 4, xtickalign=1, ytickalign = 1)
    ylims!(1,5e5)

    plots = Any[]
    labels = []

    h = hlines!(fp32_peak, xmin = 0.0, xmax = 100.0, color = :black, linestyle = (:dash, :loose), linewidth = 3)

    push!(plots, h)
    push!(labels, "FP32 Theoretical Bandwidth")

    # Plot one line per model
    for (i,g) in enumerate(groupby(df, :model))
        label = g.model[1]
        sort!(g, :gpus)

        marker = (label == "cuNumeric.jl") ? :star6 : markers[i]
        overdraw = (label == "cuNumeric.jl") ? true : false

        s = scatterlines!(g.gpus, g.gflops_per_gpu, marker = marker, overdraw = overdraw, strokewidth = 2, linewidth = 2,
                            markersize = 35, color = colors[i], strokecolor = :black)
        push!(plots, s)
        push!(labels, g.model[1])
    end

    axislegend(ax, plots[3:end],labels[3:end],  "cuNumeric.jl",
         position = :lb, patchlabelgap = 12, labelsize = 30, framevisible = false,
         labelhalign = :center, colgap = 25, titlesize = 30)

    axislegend(ax, reverse(plots[1:2]), reverse(labels[1:2]),
        position = :rb, labelsize = 30, framevisible = false, patchlabelgap = 12)

    save(file, gflops_fig)
    println("Saved plot to $(file)")
end


function make_efficiency_plot_mc(csv, file, colors, markers)
    df = CSV.read(csv, DataFrame)
    sort!(df, [:model, :gpus])

    # Compute efficiency = T(1) / T(Np) for each model
    df_eff = DataFrame()
    for g in groupby(df, :model)
        t1 = g.mean_time_ms[g.gpus .== 1][1]
        eff = t1 ./ g.mean_time_ms
        df_tmp = DataFrame(model=g.model, gpus=g.gpus, efficiency=eff)
        append!(df_eff, df_tmp)
    end

    size_in_inches = (3, 2.25)
    dpi = 300
    size_in_pixels = size_in_inches .* dpi

    efficiency_fig = Figure(resolution = size_in_pixels);
    ax = Axis(efficiency_fig[1,1], xlabel = "Number of GPUs", ylabel = "Parallel Efficiency",
        ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
        xticks = [1, 2, 4, 8], xgridvisible = false, ygridvisible = false, xscale = log2,
        xticksmirrored = true, yticksmirrored = true, xticklabelpad = 4, xtickalign=1, ytickalign = 1)

    ylims!(0, 1.2)

    plots = []
    labels = []

    for (i,g) in enumerate(groupby(df_eff, :model))
        sort!(g, :gpus)
        label = g.model[1]
        if label != "CUDA.jl" # no scaling for this one
            marker = (label == "cuNumeric.jl") ? :star6 : markers[i]
            fill_color = (label == "cuPyNumeric") ? :transparent : colors[i]
            sz = (label == "cuPyNumeric") ? 26 : 40
            overdraw = (label == "cuNumeric.jl") ? true : false
            s = scatterlines!(g.gpus, g.efficiency, marker = marker, overdraw = overdraw, strokewidth = 2, linewidth = 2,
                                markersize = sz, color = colors[i], strokecolor = :black)
            push!(plots, s)
            push!(labels, g.model[1])
        end
    end

    axislegend(ax, plots, labels,
         position = :lb, patchlabelgap = 12, labelsize = 30, framevisible = false,
         labelhalign = :center, colgap = 25, titlesize = 30)


    save(file, efficiency_fig)
    
    println("Saved weak scaling efficiency plot to $(file)")
end


function make_weak_plot_mc(csv, file, colors, markers)
    df = CSV.read(csv, DataFrame)
    # Compute total problem size and per-GPU size
    df.size = df.n .* df.m
    df.size_per_gpu = df.size ./ df.gpus
    df.gflops_per_gpu = df.gflops ./ df.gpus

    # Sort so GPUs increase left-to-right
    sort!(df, [:model, :gpus])

    size_in_inches = (3, 2.25)
    dpi = 300
    size_in_pixels = size_in_inches .* dpi

    gflops_fig = Figure(resolution = size_in_pixels);
    ax = Axis(gflops_fig[1,1], xlabel = "Number of GPUs", ylabel = "c ⨯ GFLOPs / GPU",
        ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
        xticks = sort(unique(df.gpus)), yticks = ([1e1, 1e2,1e3], ["10¹", "10²","10³"]),
        xgridvisible = false, ygridvisible = false, xscale = log2,
        yscale = log10, xticksmirrored = true, yticksmirrored = true, xticklabelpad = 4, xtickalign=1, ytickalign = 1)
    ylims!(1,1e4)

    plots = Any[]
    labels = []


    # Plot one line per model
    for (i,g) in enumerate(groupby(df, :model))
        label = g.model[1]
        sort!(g, :gpus)

        marker = (label == "cuNumeric.jl") ? :star6 : markers[i]
        overdraw = (label == "cuNumeric.jl") ? true : false
        sz = (label == "cuPyNumeric") ? 26 : 40

        s = scatterlines!(g.gpus, g.gflops_per_gpu, marker = marker, overdraw = overdraw, strokewidth = 2, linewidth = 2,
                            markersize = sz, color = colors[i], strokecolor = :black)
        push!(plots, s)
        push!(labels, g.model[1])
    end

    axislegend(ax, plots, labels, position = :rt,
                 patchlabelgap = 12, labelsize = 30, framevisible = false)

    save(file, gflops_fig)
    println("Saved plot to $(file)")
end
