using CSV, DataFrames, Plots


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


function make_plot(type::String)
    if type=="large" 
        csv, title, file = large_params()
    else
        csv, title, file = small_params()
    end

    df = CSV.read(csv, DataFrame)
    nr_gpus = length(df.gpus)
    # Compute total problem size and per-GPU size
    df.size = df.n .* df.m
    df.size_per_gpu = df.size ./ df.gpus

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
        plot!(plt, g.gpus, g.gflops / nr_gpus;
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


make_plot("large")
make_plot("small")
