#!/usr/bin/env julia
# Grid of benchmark panels, one figure per metric (throughput, time, efficiency).
# Which runs feed which panel is set in configs/plots/grid.toml.

using TOML

# Series loading and colors come from the single-benchmark plots.
include(joinpath(@__DIR__, "plot_results.jl"))

const METRICS = Dict(
    "throughput" => (ylabel="Throughput",),
    "time" => (ylabel="Time/step (ms)",),
    "efficiency" => (ylabel="Efficiency",),
)

function parse_grid_args(args)
    config = joinpath(@__DIR__, "configs", "plots", "grid.toml")
    out_dir = joinpath(@__DIR__, "plots", "grid")
    for arg in args
        if startswith(arg, "--config=")
            config = last(split(arg, "="; limit=2))
        elseif startswith(arg, "--out=")
            out_dir = last(split(arg, "="; limit=2))
        else
            error("unknown argument: $arg")
        end
    end
    config = isabspath(config) ? config : joinpath(@__DIR__, config)
    out_dir = isabspath(out_dir) ? out_dir : joinpath(@__DIR__, out_dir)
    return (; config, out_dir)
end

# A run directory holds its CSVs under <T>/; accept either level.
function csv_dir(path)
    path = isabspath(path) ? path : joinpath(@__DIR__, path)
    isdir(path) || error("results directory not found: $path")
    any(endswith(".csv"), readdir(path)) && return path
    subdirs = filter(d -> any(endswith(".csv"), readdir(d)),
        filter(isdir, readdir(path; join=true)))
    length(subdirs) == 1 ||
        error("expected one CSV subdirectory in $path, found $(length(subdirs))")
    return only(subdirs)
end

function panel_series(panel)
    group = panel["benchmark"]
    members = String[string(m) for m in get(panel, "members", [group])]
    hide = Set(string.(get(panel, "hide", String[])))
    series = []
    seen = Set{String}()
    for dir in aslist(panel["results"])
        dir = csv_dir(dir)
        found = group == "nas_ep" && members == ["nas_ep"] ?
            ep_series(dir) : group_series(dir, group, members)
        for s in found
            (s.label in seen || s.label in hide) && continue
            push!(series, s)
            push!(seen, s.label)
        end
    end
    isempty(series) && error("panel $group: no series found in $(panel["results"])")
    validate_series_sizes(series)
    return series
end

function efficiency(s)
    i1 = findfirst(x -> x.gpus == 1, s.agg)
    i1 === nothing && return nothing
    base = s.agg[i1].h
    return [x.h / (x.gpus * base) for x in s.agg]
end

# Short tick labels (20k, 1.5k, 320) keep the left margin narrow.
function compact_tick(v)
    v == 0 && return "0"
    n, suffix = abs(v) >= 1e3 ? (v / 1e3, "k") : (v, "")
    t = string(round(n; sigdigits=3))
    return (endswith(t, ".0") ? t[1:(end - 2)] : t) * suffix
end

text_px(pt) = 1.4 * pt * 100 / 72   # GR line height; Plots' px is 1/100 inch

# All sizes scale with the tick font, so proportions hold at print width.
function grid_style(font_size; legend_scale=1.0)
    k = font_size / 11
    return (tick=font_size, guide=font_size + 1, title=font_size + 2,
            # Int: Plots reads a Float text size as a rotation angle.
            legend=round(Int, legend_scale * (font_size - 1)), legend_k=legend_scale * k,
            lw=2.4k, ms=6k, k)
end

# GR sizes margins as if text were normalized per side, but it is normalized
# to the longer side, so the shorter side loses room (wide grids clip axis
# labels, tall ones y ticks). Add the missing short/long fraction back.
function gr_margin_fix(width, height, st)
    sv, sh = height / max(width, height), width / max(width, height)
    tick_w = 5 * 0.45 * text_px(st.tick)   # "0.75", "31.6"
    return (top=(1 - sv) * text_px(st.title),
            bottom=(1 - sv) * text_px(st.tick),
            guide=(1 - sv) * text_px(st.guide),
            ticks=(1 - sh) * tick_w)
end

function grid_line!(p, s, y, st; kw...)
    return plot!(p, getfield.(s.agg, :gpus), y; color=s.color, lw=st.lw, ls=s.ls,
        marker=s.marker, ms=st.ms, msc=s.color, markerstrokewidth=0.6st.k,
        label=s.label, kw...)
end

# Axis labels only on the outer edge of the grid; every panel shares them.
function panel_plot(series, metric; title, log_values, first_col, last_row, bottom_row, fix, st)
    gpus = sort(unique(x.gpus for s in series for x in s.agg))
    p = plot(;
        title, xlabel=bottom_row ? "GPUs" : "",
        ylabel=first_col ? METRICS[metric].ylabel : "",
        # GPU ticks are shared, so only the bottom panel of each column labels them.
        xscale=:log2, xticks=(gpus, last_row ? string.(gpus) : fill("", length(gpus))),
        xlims=(minimum(gpus) / 1.15, maximum(gpus) * 1.15), widen=false,
        framestyle=:box, legend=false,
        tickfontsize=st.tick, guidefontsize=st.guide, titlefontsize=st.title,
        titlefontfamily="DejaVuSans-Bold",   # TTF bold of the default font; GR built-ins mis-size
        left_margin=(fix.ticks + (first_col ? fix.guide : 0)) * Plots.px +
                    (first_col ? 3st.k * Plots.mm : 0Plots.mm),
        bottom_margin=last_row ? (fix.bottom + (bottom_row ? fix.guide : 0)) * Plots.px :
                      -1Plots.mm,
        # GR adds 2mm on every side; above the title that is only white space.
        top_margin=fix.top * Plots.px - 2Plots.mm, right_margin=2Plots.mm,
    )
    if metric == "efficiency"
        effs = filter(!isnothing, efficiency.(series))
        hi = max(1.0, maximum(maximum, effs; init=0.0))
        plot!(p; ylims=positive_ylim(hi; pad=0.12))
        hline!(p, [1.0]; color=IDEALCOL, ls=:dashdot, lw=1.4st.k, label="")
        for s in series
            e = efficiency(s)
            e === nothing || grid_line!(p, s, e, st)
        end
        return p
    end
    y, e = metric == "throughput" ? (:h, :hsd) : (:t, :tsd)
    ylims = log_values ?
        (series_ymin_positive(series, y, e) / 1.5, series_ymax(series, y, e) * 1.5) :
        positive_ylim(series_ymax(series, y, e); pad=0.2)
    plot!(p; ylims, yscale=log_values ? :log10 : :identity)
    plot!(p; yformatter=compact_tick)
    for s in series
        grid_line!(p, s, getfield.(s.agg, y), st; yerror=getfield.(s.agg, e))
    end
    return p
end

# Hand-drawn legend table in canvas pixels; Plots' multi-column legend drops
# entries when short on height.
legend_dims(st) = (swatch=34st.legend_k, char=0.8st.legend, gap=10st.legend_k, row=2.7st.legend)
# GR padding around an axis-less subplot; excluding it keeps px ≈ plot units.
const LEGEND_INSET_PX = 90

entry_px(s, d) = d.swatch + 4 + d.char * length(s.label)

# Each column is as wide as its own longest entry (rows fill left to right).
function legend_col_widths(series, cols, d)
    return [maximum(entry_px(s, d) for s in series[c:cols:end]) + d.gap for c in 1:cols]
end

function legend_rows(series, width, st)
    d = legend_dims(st)
    cols = something(findlast(c -> sum(legend_col_widths(series, c, d)) <= width - LEGEND_INSET_PX,
        1:length(series)), 1)
    # Same row count, entries spread evenly (5 -> 3 + 2, not 4 + 1).
    cols = cld(length(series), cld(length(series), cols))
    return [series[i:min(i + cols - 1, end)] for i in 1:cols:length(series)]
end

# `slot_h` set: the legend fills an empty grid slot, rows hanging from the top.
# `shift` moves the rows up by that fraction of the legend's height.
# `center` centers the columns horizontally.
function grid_legend(rows, width, st; slot_h=nothing, shift=0.0, center=false)
    d = legend_dims(st)
    col_x = cumsum([0.0; legend_col_widths(reduce(vcat, rows), length(first(rows)), d)])
    center && (col_x .+= max(0, (width - LEGEND_INSET_PX - col_x[end]) / 2))
    pl = plot(; framestyle=:none, grid=false, ticks=false, legend=false,
        xlims=(0, width - LEGEND_INSET_PX), ylims=(0, 1), widen=false,
        # Cancel GR's 2mm padding; at the canvas bottom, stop short so rounding
        # cannot push the viewport off the canvas (GR then draws it elsewhere).
        margin=0Plots.mm, top_margin=-2Plots.mm, bottom_margin=-1.9Plots.mm)
    for (r, row) in enumerate(rows)
        # In a slot, GR padding shrinks the height; spread rows instead.
        step = slot_h === nothing ? 1 / length(rows) : min(1 / length(rows), 1.6d.row / slot_h)
        y = 1 - (r - 0.5) * step + shift
        for (c, s) in enumerate(row)
            x = col_x[c]
            scale = st.legend_k / st.k
            plot!(pl, [x, x + d.swatch], [y, y]; color=s.color, lw=0.9scale * st.lw, ls=s.ls,
                label="")
            scatter!(pl, [x + d.swatch / 2], [y]; color=s.color, marker=s.marker,
                ms=0.65scale * st.ms, msc=s.color, markerstrokewidth=0.5st.legend_k, label="")
            annotate!(pl, x + d.swatch + 4, y, text(s.label, st.legend, :black, :left))
        end
    end
    return pl
end

function grid_figure(panels, metric, columns; panel_w, panel_h, st)
    rows = cld(length(panels), columns)
    # Same label always has the same style, so one shared legend covers all panels.
    legend_series = unique(s -> s.label, [s for p in panels for s in p.series])
    width = panel_w * columns
    # An empty grid slot holds the legend; otherwise it gets a row underneath.
    in_slot = rows * columns > length(panels)
    rows_legend = legend_rows(legend_series, in_slot ? panel_w : width, st)
    # No figure title: the y-axis label already names the metric.
    legend_h = in_slot ? 0 : legend_dims(st).row * length(rows_legend) + 4
    height = rows * panel_h + legend_h
    fix = gr_margin_fix(width, height, st)

    plots = Any[panel_plot(p.series, metric; title=p.title, log_values=p.log,
                    first_col=(i - 1) % columns == 0,
                    last_row=i > length(panels) - columns,
                    bottom_row=i > (rows - 1) * columns, fix, st)
                for (i, p) in enumerate(panels)]
    in_slot && push!(plots, grid_legend(rows_legend, panel_w, st; slot_h=panel_h))
    for _ in (length(plots) + 1):(rows * columns)
        push!(plots, plot(; framestyle=:none))
    end
    # One grid for all panels, so margins align per column (same panel widths).
    fig_kw = (size=(width, height), dpi=200, background_color=:white)
    in_slot && return plot(plots...; layout=grid(rows, columns), fig_kw...)
    body = plot(plots...; layout=grid(rows, columns))
    return plot(body, grid_legend(rows_legend, width, st);
        layout=grid(2, 1; heights=[rows * panel_h, legend_h] ./ height), fig_kw...)
end

# cuNumeric.jl speedups for the paper text: (label, GPU counts compared).
# Speedup = reference time / cuNumeric.jl time at the same GPU count; panels
# already guarantee equal problem sizes per GPU count.
const SPEEDUP_REFERENCES = (
    ("cuPyNumeric", :all), ("JACC.jl", :all), ("Dagger.jl", :all), ("CUDA.jl", 1),
)

geomean(x) = exp(sum(log, x) / length(x))
fmt_x(x) = string(round(x; sigdigits=3), "×")

function speedups(panel, reference, gpus)
    find(label) = findfirst(s -> s.label == label, panel.series)
    i, j = find("cuNumeric.jl"), find(reference)
    (i === nothing || j === nothing) && return nothing
    ours = Dict(x.gpus => x.t for x in panel.series[i].agg)
    theirs = Dict(x.gpus => x.t for x in panel.series[j].agg)
    shared = sort([g for g in keys(ours) if haskey(theirs, g) && (gpus === :all || g == gpus)])
    isempty(shared) && return nothing
    return [(gpus=g, speedup=theirs[g] / ours[g]) for g in shared]
end

# Per benchmark: geomean over shared GPU counts; overall: geomean of those.
function speedup_summary(panels)
    lines = ["# cuNumeric.jl speedup summary", ""]
    overall = Dict{String,Any}()
    for (reference, gpus) in SPEEDUP_REFERENCES
        scope = gpus === :all ? "all shared GPU counts" : "$gpus GPU"
        rows = [(p.title, s) for p in panels for s in (speedups(p, reference, gpus),) if s !== nothing]
        push!(lines, "## vs $reference ($scope)", "")
        if isempty(rows)
            push!(lines, "No $reference results in these panels.", "")
            continue
        end
        push!(lines, "| Benchmark | GPUs | Speedup |", "|---|---|---:|")
        per_bench = Float64[]
        for (title, s) in rows
            g = geomean(getfield.(s, :speedup))
            push!(per_bench, g)
            detail = join(["$(x.gpus): $(fmt_x(x.speedup))" for x in s], ", ")
            push!(lines, "| $title | $detail | $(fmt_x(g)) |")
        end
        overall[reference] = (geomean=geomean(per_bench), n=length(rows),
            lo=minimum(per_bench), hi=maximum(per_bench))
        o = overall[reference]
        push!(lines, "", "Geomean over $(o.n) benchmarks: **$(fmt_x(o.geomean))** " *
            "(per-benchmark range $(fmt_x(o.lo))–$(fmt_x(o.hi))).", "")
    end
    return join(lines, "\n")
end

# Panel size and text style from a plot config's style keys.
function grid_dimensions(raw, columns)
    panel_w, panel_h = get(raw, "panel_size", [400, 300])
    font_size = get(raw, "font_size", 11)
    # `width_in`: canvas units are points, drawn at 2x (GR lays out small
    # canvases poorly), so scaled to width_in, text prints at font_size pt.
    if haskey(raw, "width_in")
        w = 144raw["width_in"] / columns
        panel_w, panel_h = w, panel_h * w / panel_w
        font_size *= 2
    end
    return panel_w, panel_h, grid_style(font_size; legend_scale=get(raw, "legend_scale", 1.0))
end

function grid_main(args=ARGS)
    cfg = parse_grid_args(args)
    isfile(cfg.config) || error("grid config not found: $(cfg.config); " *
        "copy configs/plots/grid.example.toml to configs/plots/grid.toml and fill in run ids")
    raw = TOML.parsefile(cfg.config)
    metrics = string.(get(raw, "metrics", ["throughput", "time", "efficiency"]))
    for m in metrics
        haskey(METRICS, m) || error("unknown metric $m; use $(join(keys(METRICS), ", "))")
    end
    columns = min(get(raw, "columns", 2), length(get(raw, "panel", [])))
    panel_w, panel_h, st = grid_dimensions(raw, columns)
    format = get(raw, "format", "png")
    panels = map(get(raw, "panel", [])) do p
        (series=panel_series(p),
         title=get(p, "title", group_title(p["benchmark"])),
         log=get(p, "log", false))
    end
    isempty(panels) && error("no [[panel]] entries in $(cfg.config)")

    mkpath(cfg.out_dir)
    for m in metrics
        out = joinpath(cfg.out_dir, "grid_$(m).$(format)")
        fig = grid_figure(panels, m, columns; panel_w, panel_h, st)
        savefig(fig, out)
        println("wrote $out")
    end
    summary = speedup_summary(panels)
    out = joinpath(cfg.out_dir, "speedup_summary.md")
    write(out, summary)
    println("wrote $out")
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    grid_main()
end
