using CSV, DataFrames, Plots

function make_efficiency_plot(csv, title, file)
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

    # Plot efficiency
    plt_eff = plot(; 
        xlabel = "Number of GPUs", 
        ylabel = "Scaling Efficiency [T(1) / T(Np)]",
        title = title,
        xticks = [1, 2, 4, 8],
        legend = :bottomleft,
        dpi = 300,
        xscale = :log2,
        ylims = (0, 1.2)
    )

    for g in groupby(df_eff, :model)
        label = g.model[1]
        sort!(g, :gpus)
        plot!(plt_eff, g.gpus, g.efficiency;
            label=label,
            lw=2,
            marker=:circle,
            xticks=([1,2,4,8], ["1","2","4","8"])
        )
    end

    savefig(plt_eff, file)
    println("Saved weak scaling efficiency plot to $(file)")
end 


function make_weak_plot(csv, title, file)
    df = CSV.read(csv, DataFrame)
    # Compute total problem size and per-GPU size
    df.size = df.n .* df.m
    df.size_per_gpu = df.size ./ df.gpus
    df.gflops_per_gpu = df.gflops ./ df.gpus

    # Sort so GPUs increase left-to-right
    sort!(df, [:model, :gpus])

    plt = plot(; 
        xlabel = "Number of GPUs", 
        ylabel = "GFLOPS / GPU",
        title = title,
        xticks = sort(unique(df.gpus)),
        legend = :bottomright,
        dpi = 300
    )

    # Plot one line per model
    for g in groupby(df, :model)
        label = g.model[1]
        sort!(g, :gpus)
        plot!(plt, g.gpus, g.gflops_per_gpu;
            label=label,
            lw=2,
            marker=:circle,
            xscale=:log2,
            yscale=:log10,
            ylims=(1, 1e5),
            xticks = ([1, 2, 4, 8], ["1", "2", "4", "8"]))
        end

    savefig(plt, file)
    println("Saved plot to $(file)")
end