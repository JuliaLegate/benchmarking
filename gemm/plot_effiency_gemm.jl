using CSV, DataFrames, Plots

function large_params()
    csv = "gemm/gemm_large.csv"
    title = "GEMM Weak Scaling Efficiency [Large Test]"
    file = "gemm/gemm_weak_scaling_efficiency_large.png"
    return csv, title, file
end

function small_params()
    csv = "gemm/gemm.csv"
    title = "GEMM Weak Scaling Efficiency [Small Test]"
    file = "gemm/gemm_weak_scaling_efficiency_small.png"
    return csv, title, file
end


function make_plot(csv, title, file )
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
        legend = :bottomright,
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

make_plot(large_params()...)
make_plot(small_params()...)
