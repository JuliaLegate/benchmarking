using Plots

length(ARGS) == 2 || error("Usage: julia plot_results.jl results.csv timings.png")
csv_path, image_path = ARGS
rows = NamedTuple[]
for (i, line) in enumerate(eachline(csv_path))
    i == 1 && continue
    fields = split(line, ',')
    length(fields) == 13 || error("Malformed result row $i")
    push!(rows, (
        backend=fields[1], eltype=fields[2], n=parse(Int, fields[3]),
        bands=parse(Int, fields[4]), order=parse(Int, fields[5]),
        iters=parse(Int, fields[6]), median=parse(Float64, fields[7]),
        minimum=parse(Float64, fields[8]), maximum=parse(Float64, fields[9]),
    ))
end
isempty(rows) && error("No results in $csv_path")
length(unique((r.eltype, r.bands, r.order, r.iters) for r in rows)) == 1 ||
    error("Mixed precision or solver settings in one plot")

sizes = sort!(unique(row.n for row in rows))
settings = first(rows)
figure = plot(;
    xlabel="Image dimension N (N × N pixels)", ylabel="Complete optimization (ms)",
    title="Absorption image fit — $(settings.eltype), $(settings.bands) bands, $(settings.iters) Adam steps",
    xscale=:log10, yscale=:log10, xticks=(sizes, string.(sizes)),
    legend=:topleft, linewidth=2, markersize=5, size=(900, 550),
)
for backend in ("CuArray", "cuNumeric", "cpu")
    subset = sort!(filter(row -> row.backend == backend, rows); by=row -> row.n)
    isempty(subset) && continue
    medians = [row.median for row in subset]
    lows = [row.median - row.minimum for row in subset]
    highs = [row.maximum - row.median for row in subset]
    plot!(figure, [row.n for row in subset], medians;
          yerror=(lows, highs), label=backend, marker=:circle)
end
savefig(figure, image_path)
println("Saved $image_path")
