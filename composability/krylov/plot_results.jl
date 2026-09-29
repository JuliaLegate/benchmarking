module KrylovPlot

using Plots
using Statistics: mean, std

const COLORS = Dict("CUDA" => "#4c78a8", "Dagger" => "#f58518",
    "cuNumeric" => "#54a24b", "cuNumeric local" => "#b279a2")

function read_results(experiment, paths)
    experiment in ("single", "weak") || error("Unknown experiment $experiment")
    expected = experiment == "single" ? collect(keys(COLORS)) : ["Dagger", "cuNumeric", "cuNumeric local"]
    rows = NamedTuple[]
    for path in paths
        open(path) do stream
            for (i, line) in enumerate(eachline(stream))
                i == 1 && continue
                fields = split(line, ',')
                length(fields) == 16 || error("Malformed result row $i in $path")
                fields[1] == experiment && fields[4] == "cg" || continue
                fields[3] in expected || error("Unexpected backend: $(fields[3])")
                samples = parse.(Float64, split(fields[16], ';'))
                length(samples) >= 2 && all(isfinite, samples) && all(>(0), samples) ||
                    error("Invalid timing samples in row $i of $path")
                push!(rows, (base_n=fields[2], backend=fields[3], eltype=fields[6],
                    gpus=parse(Int, fields[7]), n=parse(Int, fields[8]),
                    mean=mean(samples), stderr=std(samples) / sqrt(length(samples))))
            end
        end
    end
    isempty(rows) && error("No CG results for $experiment")
    length(unique(row.eltype for row in rows)) == 1 || error("Mixed precision in one plot")
    experiment == "single" || length(unique(row.base_n for row in rows)) == 1 ||
        error("Mixed weak-scaling base N in one plot")
    allunique((row.backend, experiment == "single" ? row.n : row.gpus) for row in rows) ||
        error("Duplicate backend/size point")
    return rows
end

function plot_results(experiment, paths, image_path)
    rows = read_results(experiment, paths)
    xs = experiment == "single" ? sort!(unique(row.n for row in rows)) : [1, 2, 4, 8]
    subtitle = first(rows).eltype * (experiment == "weak" ? ", N(1)=$(first(rows).base_n); N(G)≈N(1)√G" : "")
    figure = plot(;
        xlabel=experiment == "single" ? "Matrix dimension N" : "GPUs",
        ylabel="Mean complete solve (ms) ± standard error",
        title="CG — $experiment GPU scaling — $subtitle",
        xscale=:log2, yscale=:log10, xticks=(xs, string.(xs)),
        legend=:topleft, linewidth=2, markersize=5, size=(900, 550),
        left_margin=18Plots.mm, bottom_margin=8Plots.mm,
    )
    for backend in ("CUDA", "Dagger", "cuNumeric", "cuNumeric local")
        subset = sort!(filter(row -> row.backend == backend, rows);
            by=row -> experiment == "single" ? row.n : row.gpus)
        isempty(subset) && continue
        x = [experiment == "single" ? row.n : row.gpus for row in subset]
        plot!(figure, x, [row.mean for row in subset];
            yerror=[row.stderr for row in subset], label=backend, marker=:circle, color=COLORS[backend])
        if experiment == "weak" && first(x) == 1
            hline!(figure, [first(subset).mean];
                linestyle=:dash, alpha=0.4, label="$backend ideal", color=COLORS[backend])
        end
    end
    savefig(figure, image_path)
    println("Saved $image_path")
end

function main(args=ARGS)
    length(args) >= 4 && args[end-1] == "--output" ||
        error("Usage: julia plot_results.jl {single|weak} results.csv [results.csv ...] --output timings.png")
    plot_results(first(args), args[2:end-2], last(args))
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    KrylovPlot.main()
end
