using Plots

length(ARGS) == 2 || error("Usage: julia plot_results.jl results.csv timings.png")
csv_path, image_path = ARGS
rows = NamedTuple[]
for (i, line) in enumerate(eachline(csv_path))
    i == 1 && continue
    fields = split(line, ',')
    length(fields) == 9 || error("Malformed result row $i")
    push!(rows, (
        backend=fields[1], eltype=fields[2], n=parse(Int, fields[3]),
        steps=parse(Int, fields[4]), median=parse(Float64, fields[5]),
        minimum=parse(Float64, fields[6]), maximum=parse(Float64, fields[7]),
    ))
end
isempty(rows) && error("No results in $csv_path")
length(unique(row.eltype for row in rows)) == 1 || error("Mixed element types in one plot")
length(unique(row.steps for row in rows)) == 1 || error("Mixed step counts in one plot")

sizes = sort!(unique(row.n for row in rows))
figure = plot(;
    xlabel="Grid dimension N (N × N)", ylabel="Complete solve (ms)",
    title="OrdinaryDiffEq heat equation — $(first(rows).eltype), $(first(rows).steps) steps",
    xscale=:log10, yscale=:log10, xticks=(sizes, string.(sizes)),
    legend=:topleft, linewidth=2, markersize=5, size=(900, 550),
)
for backend in ("CuArray", "cuNumeric", "Dagger", "cpu")
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
