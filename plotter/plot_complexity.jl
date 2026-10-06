#!/usr/bin/env julia
# Complexity vs performance: one figure per GPU count, one panel per metric,
# a shaded hull per model. Config: configs/plots/complexity.toml.

using TOML

# Series, colors, layout and legend come from the grid plots.
include(joinpath(@__DIR__, "plot_grid.jl"))

const COMPLEXITY_METRICS = Dict(
    "cyclomatic" => (column="scc_complexity", name="Cyclomatic complexity"),
    "uloc" => (column="scc_uloc", name="ULOC"),
    "sloc" => (column="scc_code", name="SLOC"),
)

# Series label => loc-analysis variant; other series are skipped.
const LOC_VARIANTS = Dict(
    CUNUMERIC_NAME => "cunumeric", CUPYNUMERIC_NAME => "cupynumeric", "CUDA.jl" => "cudajl",
    "JACC.jl" => "jacc", "Dagger.jl" => "dagger",
)

const COLOR_IDEAL = "#b3261e"   # not a model color, unlike the gray y = 1 line

# plot_grid.jl's flags with this plot's defaults.
parse_complexity_args(args) = parse_grid_args(
    vcat(["--config=configs/plots/complexity.toml", "--out=plots/complexity"], args))

# (benchmark, variant) => column => count.
function load_loc(path)
    lines = filter(!isempty ∘ strip, readlines(path))
    header = split(first(lines), ',')
    columns = [m.column for m in values(COMPLEXITY_METRICS)]
    return Dict(begin
            f = Dict(zip(header, split(line, ',')))
            (f["benchmark"], f["variant"]) => Dict(c => parse(Int, f[c]) for c in columns)
        end for line in lines[2:end])
end

# One point per (benchmark, model); rel = fastest time / this time.
function complexity_points(panels, loc, gpus, hide)
    points = []
    for p in panels
        timed = [(s, x.t) for s in p.series if haskey(LOC_VARIANTS, s.label) && !(s.label in hide)
                 for x in s.agg if x.gpus == gpus]
        isempty(timed) && continue
        fastest = minimum(last, timed)
        for (s, t) in timed
            key = (p.benchmark, LOC_VARIANTS[s.label])
            haskey(loc, key) || error("no LOC row for $(join(key, " / ")) in the loc summary")
            push!(points, (series=s, benchmark=p.benchmark, t, rel=fastest / t, counts=loc[key]))
        end
    end
    return points
end

# Per model, in plot order: its series, x (count in `column`) and y (rel) values.
function model_groups(points, column)
    return map(unique(pt.series.label for pt in points)) do label
        own = filter(pt -> pt.series.label == label, points)
        (series=first(own).series, xs=[pt.counts[column] for pt in own], ys=[pt.rel for pt in own])
    end
end

# Convex hull (monotone chain).
function convex_hull(pts)
    pts = sort(unique(pts))
    length(pts) <= 2 && return pts
    cross(o, a, b) = (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])
    function half(ps)
        h = eltype(ps)[]
        for q in ps
            while length(h) >= 2 && cross(h[end - 1], h[end], q) <= 0
                pop!(h)
            end
            push!(h, q)
        end
        return h
    end
    lower, upper = half(pts), half(reverse(pts))
    return vcat(lower[1:(end - 1)], upper[1:(end - 1)])
end

function hull!(p, g, st)
    h = convex_hull(collect(zip(g.xs, g.ys)))
    if length(h) >= 3
        plot!(p, Shape(first.(h), last.(h)); fillcolor=g.series.color, fillalpha=0.16,
            linecolor=g.series.color, linealpha=0.55, lw=0.8st.k, label="")
    elseif length(h) == 2
        plot!(p, first.(h), last.(h); color=g.series.color, alpha=0.3, lw=3st.lw, label="")
    end
    return p
end

# Drawn behind the points, so a point at the ideal stays visible.
function ideal_point!(p, st)
    scatter!(p, [0], [1.0]; marker=:star5, ms=2.5st.ms, color=:white, msc=COLOR_IDEAL,
        markerstrokewidth=2.4st.k, label="")
    annotate!(p, 0, 1.0, text("   Ideal", st.tick, COLOR_IDEAL, :left, :bottom))
    return p
end

# Left is simpler; higher is faster (1 = fastest model on that benchmark).
# At most four round ticks, so narrow panels don't crowd them. With capped points,
# the edge tick shows their largest value instead.
function complexity_xticks(hi, clipped)
    ticks = collect(0:first(filter(t -> hi / t <= 4, [m * 10.0^k for k in 0:6 for m in (1, 2, 5)])):hi)
    isempty(clipped) && return ticks
    labels = compact_tick.(ticks)
    ticks[end] == hi ? (labels[end] = string(maximum(first, clipped))) :
        (push!(ticks, hi); push!(labels, string(maximum(first, clipped))))
    return (ticks, labels)
end

# `xmax` caps the axis: points past it sit on the edge, which is labeled with their value.
function complexity_panel(points, metric; show_mean, ideal, first_col, fix, st, xmax=nothing)
    m = COMPLEXITY_METRICS[metric]
    groups = model_groups(points, m.column)
    hi = something(xmax, maximum(maximum(g.xs) for g in groups))
    clipped = [(x, y, g.series) for g in groups for (x, y) in zip(g.xs, g.ys) if x > hi]
    # Means use the true counts; hulls and points use the capped ones.
    means = [(mean(g.xs), mean(g.ys)) for g in groups]
    groups = [merge(g, (; xs=min.(g.xs, hi))) for g in groups]
    p = plot(;
        xlabel=m.name, ylabel=first_col ? "Relative perf." : "",
        # Just left of 0, so markers at 0 show whole.
        xlims=(-0.045hi, 1.08hi), ylims=(0, ideal ? 1.15 : 1.08), widen=false,
        xticks=complexity_xticks(hi, clipped),
        # The panels share the 0..1 y scale, so only the first labels it.
        xformatter=compact_tick, yformatter=first_col ? compact_tick : (_ -> ""),
        framestyle=:box, legend=false,
        tickfontsize=st.tick, guidefontsize=st.guide,
        left_margin=first_col ? (fix.ticks + fix.guide) * Plots.px + 3st.k * Plots.mm :
                    -4Plots.mm,   # GR keeps room for the (empty) tick labels
        bottom_margin=(fix.bottom + fix.guide) * Plots.px,
        top_margin=1Plots.mm, right_margin=1Plots.mm,
    )
    hline!(p, [1.0]; color=IDEALCOL, lw=1.0st.k, label="")
    ideal && ideal_point!(p, st)
    # Clouds, then points, then means on top.
    foreach(g -> hull!(p, g, st), groups)
    for g in groups
        scatter!(p, g.xs, g.ys; color=g.series.color, marker=g.series.marker,
            ms=0.65st.ms, msc=:white, markerstrokewidth=0.4st.k, label="")
    end
    # Axis break just before the capped edge.
    !isempty(clipped) && annotate!(p, 0.9hi, 0, text("//", st.tick, :black, :center))
    for (g, (mx, my)) in zip(groups, means)
        show_mean || break
        scatter!(p, [mx], [my]; color=g.series.color, marker=g.series.marker,
            ms=2.0st.ms, msc=:black, markerstrokewidth=2st.k, label="")
    end
    return p
end

# One row of panels with the legend centered underneath.
function complexity_figure(points, metrics; show_mean, ideal, panel_w, panel_h, st, legend_cols=nothing,
        xmax=nothing)
    draw(i; first_col, fix, _...) =
        complexity_panel(points, metrics[i]; show_mean, ideal, first_col, fix, st, xmax)
    legend_series = unique(s -> s.label, [pt.series for pt in points])
    return grid_layout(draw, length(metrics), legend_series, length(metrics);
        panel_w, panel_h, st, center=true, legend_cols)
end

function write_points_csv(path, points, gpus)
    columns = [COMPLEXITY_METRICS[m].column for m in sort(collect(keys(COMPLEXITY_METRICS)))]
    open(path, "w") do io
        println(io, join(["gpus", "benchmark", "model", "time_ms", "relative_performance", columns...], ","))
        for pt in points
            row = [gpus, pt.benchmark, pt.series.label, pt.t, round(pt.rel; digits=4)]
            println(io, join([row; [pt.counts[c] for c in columns]], ","))
        end
    end
end

# Each model's mean count and mean relative performance, simplest first.
function model_means(points, metric)
    groups = model_groups(points, COMPLEXITY_METRICS[metric].column)
    rows = [(label=g.series.label, x=mean(g.xs), y=mean(g.ys)) for g in groups]
    return sort(rows; by=r -> r.x)
end

r3(v) = round(v; digits=3)

function complexity_summary(by_gpus, metrics)
    lines = ["# Complexity vs performance summary", "",
        "Per model: mean count over benchmarks (lower = simpler) and mean relative " *
        "performance (fastest time / model time; 1 = fastest).", ""]
    perf = Dict{String,Vector{Float64}}()
    for (g, points) in by_gpus, metric in metrics
        name = COMPLEXITY_METRICS[metric].name
        push!(lines, "## $g GPU, $name", "",
            "| Model | Mean $name | Mean relative performance |", "|---|---:|---:|")
        for r in model_means(points, metric)
            push!(lines, "| $(r.label) | $(round(r.x; digits=1)) | $(r3(r.y)) |")
            metric == first(metrics) && push!(get!(perf, r.label, Float64[]), r.y)
        end
        push!(lines, "")
    end
    push!(lines, "## Mean relative performance over $(join(first.(by_gpus), "/")) GPUs", "",
        "| Model | Mean relative performance | GPU counts |", "|---|---:|---:|")
    for (label, ys) in sort(collect(perf); by=kv -> -mean(kv[2]))
        push!(lines, "| $label | $(r3(mean(ys))) | $(length(ys)) |")
    end
    return join(lines, "\n")
end

function complexity_main(args=ARGS)
    cfg = parse_complexity_args(args)
    raw = TOML.parsefile(cfg.config)
    resolve(p) = isabspath(p) ? p : joinpath(BENCH_ROOT, p)
    grid_raw = TOML.parsefile(resolve(get(raw, "grid", "configs/plots/grid.toml")))
    loc = load_loc(resolve(get(raw, "loc", "loc-analysis/results/summary.csv")))
    metrics = string.(get(raw, "metrics", ["cyclomatic", "sloc"]))
    for m in metrics
        haskey(COMPLEXITY_METRICS, m) ||
            error("unknown metric $m; use $(join(keys(COMPLEXITY_METRICS), ", "))")
    end
    hide = Set(display_name.(string.(get(raw, "hide", String[]))))
    show_mean = get(raw, "mean", true)
    ideal = get(raw, "ideal", true)
    legend_cols = get(raw, "legend_columns", nothing)
    xmax = get(raw, "xmax", nothing)
    panel_w, panel_h, st = grid_dimensions(raw, length(metrics))
    format = get(raw, "format", "png")
    panels = [(series=panel_series(p), benchmark=p["benchmark"]) for p in get(grid_raw, "panel", [])]
    isempty(panels) && error("no [[panel]] entries in the grid config")

    mkpath(cfg.out_dir)
    by_gpus = []
    for g in get(raw, "gpus", [1, 2, 4, 8])
        points = complexity_points(panels, loc, g, hide)
        isempty(points) && (println("no results at $g GPUs; skipped"); continue)
        out = joinpath(cfg.out_dir, "complexity_$(g)gpu.$(format)")
        savefig(complexity_figure(points, metrics; show_mean, ideal, panel_w, panel_h, st, legend_cols, xmax), out)
        println("wrote $out")
        write_points_csv(joinpath(cfg.out_dir, "complexity_$(g)gpu.csv"), points, g)
        push!(by_gpus, g => points)
    end
    out = joinpath(cfg.out_dir, "complexity_summary.md")
    write(out, complexity_summary(by_gpus, metrics))
    println("wrote $out")
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    complexity_main()
end
