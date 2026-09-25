using Plots
using Statistics: mean, std

length(ARGS) == 3 || error("Usage: julia plot_results.jl {single|weak} results.csv timings.png")
experiment, csv_path, image_path = ARGS
experiment in ("single", "weak") || error("Unknown experiment $experiment")

rows = NamedTuple[]
for (i, line) in enumerate(eachline(csv_path))
    i == 1 && continue
    fields = split(line, ',')
    length(fields) == 14 || error("Malformed result row $i")
    fields[1] == experiment || continue
    samples = parse.(Float64, split(fields[14], ';'))
    length(samples) >= 2 && all(isfinite, samples) && all(>(0), samples) ||
        error("Invalid timing samples in row $i")
    push!(rows, (
        base_n=fields[2], backend=fields[3], eltype=fields[4],
        gpus=parse(Int, fields[5]), n=parse(Int, fields[6]),
        steps=parse(Int, fields[7]), mean=mean(samples),
        stderr=std(samples) / sqrt(length(samples)),
    ))
end
isempty(rows) && error("No $experiment results in $csv_path")
length(unique((r.eltype, r.steps) for r in rows)) == 1 ||
    error("Mixed precision or step counts in one plot")
experiment == "single" || length(unique(row.base_n for row in rows)) == 1 ||
    error("Mixed weak-scaling base N in one plot")

xs = sort!(unique(experiment == "single" ? row.n : row.gpus for row in rows))
figure = plot(;
    xlabel=experiment == "single" ? "Grid dimension N (N × N)" : "GPUs",
    ylabel="Mean complete solve (ms) ± standard error",
    title="OrdinaryDiffEq heat equation — $(first(rows).eltype), $(first(rows).steps) steps" *
          (experiment == "weak" ? ", N(1)=$(first(rows).base_n)" : ""),
    xscale=:log2, yscale=:log10, xticks=(xs, string.(xs)),
    legend=:topleft, linewidth=2, markersize=5, size=(900, 550),
    left_margin=18Plots.mm, bottom_margin=8Plots.mm,
)
for backend in ("CuArray", "Dagger", "cuNumeric")
    subset = sort!(filter(row -> row.backend == backend, rows);
                   by=row -> experiment == "single" ? row.n : row.gpus)
    isempty(subset) && continue
    x = [experiment == "single" ? row.n : row.gpus for row in subset]
    plot!(figure, x, [row.mean for row in subset];
          yerror=[row.stderr for row in subset], label=backend, marker=:circle)
    if experiment == "weak" && first(x) == 1
        hline!(figure, [first(subset).mean];
               linestyle=:dash, alpha=0.4, label="$backend ideal")
    end
end
savefig(figure, image_path)
println("Saved $image_path")
