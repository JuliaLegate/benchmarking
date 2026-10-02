#!/usr/bin/env julia
# Complexity vs performance: one figure per GPU count, one panel per metric,
# a shaded hull per model. Config: configs/plots/complexity.toml.

using TOML

# Series, colors, layout and legend come from the grid plots.
include(joinpath(@__DIR__, "plot_grid.jl"))

const COMPLEXITY_METRICS = Dict(
    "cyclomatic" => (column="scc_complexity", xlabel="Cyclomatic complexity"),
    "uloc" => (column="scc_uloc", xlabel="ULOC"),
    "sloc" => (column="scc_code", xlabel="SLOC"),
)

# Series label => loc-analysis variant; other series are skipped.
const LOC_VARIANTS = Dict(
    "cuNumeric.jl" => "cunumeric", "cuPyNumeric" => "cupynumeric", "CUDA.jl" => "cudajl",
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
    scatter!(p, [0], [1.0]; marker=:star5, ms=2st.ms, color=:white, msc=COLOR_IDEAL,
        markerstrokewidth=2.4st.k, label="")
    annotate!(p, 0, 1.0, text("  Ideal", st.tick, COLOR_IDEAL, :left, :bottom))
    return p
end

function complexity_panel(points, metric; show_mean, ideal, first_col, fix, st)
    m = COMPLEXITY_METRICS[metric]
    groups = model_groups(points, m.column)
    hi = maximum(maximum(g.xs) for g in groups)
    p = plot(;
        xlabel=m.xlabel, ylabel=first_col ? "Relative performance" : "",
        # Just left of 0, so markers at 0 show whole.
        xlims=(-0.02hi, 1.08hi), ylims=(0, 1.08), widen=false,
        xformatter=compact_tick, yformatter=compact_tick, framestyle=:box, legend=false,
        tickfontsize=st.tick, guidefontsize=st.guide,
        left_margin=(fix.ticks + (first_col ? fix.guide : 0)) * Plots.px +
                    (first_col ? 3st.k * Plots.mm : 0Plots.mm),
        bottom_margin=(fix.bottom + fix.guide) * Plots.px,
        top_margin=1Plots.mm, right_margin=2Plots.mm,
    )
    hline!(p, [1.0]; color=IDEALCOL, ls=:dashdot, lw=1.4st.k, label="")
    ideal && ideal_point!(p, st)
    # Clouds, then points, then means on top.
    foreach(g -> hull!(p, g, st), groups)
    for g in groups
        scatter!(p, g.xs, g.ys; color=g.series.color, marker=g.series.marker,
            ms=0.65st.ms, msc=:white, markerstrokewidth=0.4st.k, label="")
    end
    for g in groups
        show_mean || break
        scatter!(p, [mean(g.xs)], [mean(g.ys)]; color=g.series.color, marker=g.series.marker,
            ms=1.6st.ms, msc=:black, markerstrokewidth=2st.k, label="")
    end
    return p
end

# One row of panels with the legend centered underneath.
function complexity_figure(points, metrics; show_mean, ideal, panel_w, panel_h, st)
    width = panel_w * length(metrics)
    rows_legend = legend_rows(unique(s -> s.label, [pt.series for pt in points]), width, st)
    legend_h = legend_dims(st).row * length(rows_legend) + 4
    height = panel_h + legend_h
    fix = gr_margin_fix(width, height, st)
    panels = [complexity_panel(points, m; show_mean, ideal, first_col=i == 1, fix, st)
              for (i, m) in enumerate(metrics)]
    return plot(plot(panels...; layout=grid(1, length(metrics))),
        grid_legend(rows_legend, width, st; center=true);
        layout=grid(2, 1; heights=[panel_h, legend_h] ./ height),
        size=(width, height), dpi=200, background_color=:white)
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

# Distance from each model's mean to the ideal (0, 1), with x scaled by the
# panel's largest count and y already 0..1. Lower is better.
function ideal_distances(points, metric)
    groups = model_groups(points, COMPLEXITY_METRICS[metric].column)
    hi = maximum(maximum(g.xs) for g in groups)
    rows = [(label=g.series.label, x=mean(g.xs), y=mean(g.ys),
             distance=hypot(mean(g.xs) / hi, 1 - mean(g.ys))) for g in groups]
    return sort(rows; by=r -> r.distance)
end

r3(v) = round(v; digits=3)

function complexity_summary(by_gpus, metrics)
    lines = ["# Complexity vs performance summary", "",
        "Distance from each model's mean point to the ideal (no complexity, fastest), " *
        "with x scaled to 0–1 by the panel's largest count. Lower is better.", ""]
    overall = Dict{Tuple{String,String},Vector{Float64}}()
    for (g, points) in by_gpus, metric in metrics
        xlabel = COMPLEXITY_METRICS[metric].xlabel
        push!(lines, "## $g GPU, $xlabel", "",
            "| Rank | Model | Mean $xlabel | Mean relative performance | Distance |",
            "|---:|---|---:|---:|---:|")
        for (i, r) in enumerate(ideal_distances(points, metric))
            push!(lines, "| $i | $(r.label) | $(round(r.x; digits=2)) | $(r3(r.y)) | $(r3(r.distance)) |")
            push!(get!(overall, (metric, r.label), Float64[]), r.distance)
        end
        push!(lines, "")
    end
    for metric in metrics
        rows = sort([(label, mean(d), length(d)) for ((m, label), d) in overall if m == metric];
            by=r -> r[2])
        push!(lines, "## Overall, $(COMPLEXITY_METRICS[metric].xlabel) " *
            "(mean distance over $(join(first.(by_gpus), "/")) GPUs)", "",
            "| Rank | Model | Mean distance | GPU counts |", "|---:|---|---:|---:|")
        for (i, (label, d, n)) in enumerate(rows)
            push!(lines, "| $i | $label | $(r3(d)) | $n |")
        end
        push!(lines, "")
    end
    return join(lines, "\n")
end

function complexity_main(args=ARGS)
    cfg = parse_complexity_args(args)
    raw = TOML.parsefile(cfg.config)
    resolve(p) = isabspath(p) ? p : joinpath(@__DIR__, p)
    grid_raw = TOML.parsefile(resolve(get(raw, "grid", "configs/plots/grid.toml")))
    loc = load_loc(resolve(get(raw, "loc", "loc-analysis/results/summary.csv")))
    metrics = string.(get(raw, "metrics", ["cyclomatic", "sloc"]))
    for m in metrics
        haskey(COMPLEXITY_METRICS, m) ||
            error("unknown metric $m; use $(join(keys(COMPLEXITY_METRICS), ", "))")
    end
    hide = Set(string.(get(raw, "hide", String[])))
    show_mean = get(raw, "mean", true)
    ideal = get(raw, "ideal", true)
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
        savefig(complexity_figure(points, metrics; show_mean, ideal, panel_w, panel_h, st), out)
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
