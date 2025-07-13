using CSV, DataFrames, Plots

# Load data
df = CSV.read("gemm/gemm.csv", DataFrame)

nr_gpus = length(df.gpus)

# Compute total problem size and per-GPU size
df.size = df.n .* df.m
df.size_per_gpu = df.size ./ df.gpus

# Sort so GPUs increase left-to-right
sort!(df, [:model, :gpus])

plt = plot(; 
    xlabel = "Number of GPUs", 
    ylabel = "GFLOPS",
    title = "GEMM Weak Scaling Performance [Small Test]",
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
        yscale=:log10,
        ylims=(1, 1e5))
end

savefig(plt, "gemm/gemm_weak_scaling_small.png")
println("Saved plot to gemm_weak_scaling_small.png")