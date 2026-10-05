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
    config = joinpath(BENCH_ROOT, "configs", "plots", "grid.toml")
    out_dir = joinpath(BENCH_ROOT, "plots", "grid")
    for arg in args
        if startswith(arg, "--config=")
            config = last(split(arg, "="; limit=2))
        elseif startswith(arg, "--out=")
            out_dir = last(split(arg, "="; limit=2))
        else
            error("unknown argument: $arg")
        end
    end
    config = isabspath(config) ? config : joinpath(BENCH_ROOT, config)
    out_dir = isabspath(out_dir) ? out_dir : joinpath(BENCH_ROOT, out_dir)
    return (; config, out_dir)
end

# A run directory holds its CSVs under <T>/; accept either level.
function csv_dir(path)
    path = isabspath(path) ? path : joinpath(BENCH_ROOT, path)
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
    hide = Set(display_name.(string.(get(panel, "hide", String[]))))
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

# Hollow markers for every series first, then all lines and error bars over them:
# GR can't draw a see-through marker fill, so this keeps every line visible
# through the markers. `items` are (series, y, yerror or nothing).
function grid_lines!(p, items, st)
    for (s, y, _) in items
        hollow_marker!(p, getfield.(s.agg, :gpus), y, s, st.ms)
    end
    for (s, y, err) in items
        grid_line!(p, s, y, st; yerror=err)
    end
    return p
end

# A colored marker with a smaller white one on top. GR's PDF output scales marker
# outlines (markerstrokewidth) with canvas height, so tall figures got near-solid
# markers; filled markers keep the same size in every figure.
function hollow_marker!(p, x, y, s, ms)
    scatter!(p, x, y; marker=s.marker, ms, color=s.color, markerstrokewidth=0, label="")
    return scatter!(p, x, y; marker=s.marker, ms=0.6ms, color=:white, markerstrokewidth=0, label="")
end

# One series' line and thin error bars (Plots would draw them at the line width).
function grid_line!(p, s, y, st; yerror=nothing, kw...)
    x = getfield.(s.agg, :gpus)
    plot!(p, x, y; color=s.color, lw=st.lw, ls=s.ls, label=s.label, kw...)
    yerror === nothing && return p
    lo, hi = yerror isa Tuple ? yerror : (yerror, yerror)
    cap, w = 2^0.05, 0.4st.lw
    for (xi, yi, l, h) in zip(x, y, lo, hi)
        bottom = max(yi - l, 1e-3yi)   # stays positive on log axes
        plot!(p, [xi, xi], [bottom, yi + h]; color=s.color, lw=w, label="")
        for yc in (bottom, yi + h)
            plot!(p, [xi / cap, xi * cap], [yc, yc]; color=s.color, lw=w, label="")
        end
    end
    return p
end

# Axis labels only on the outer edge of the grid; every panel shares them.
function panel_plot(series, metric; title, log_values, zero_log=false, split=nothing, split_pad=0.05, gridlines=true,
    first_col, last_row, bottom_row, fix, st)
    gpus = sort(unique(x.gpus for s in series for x in s.agg))
    p = plot(;
        title, xlabel=bottom_row ? "GPUs" : "",
        ylabel=first_col ? METRICS[metric].ylabel : "",
        # GPU ticks are shared, so only the bottom panel of each column labels them.
        xscale=:log2, xticks=(gpus, last_row ? string.(gpus) : fill("", length(gpus))),
        xlims=(minimum(gpus) / 1.15, maximum(gpus) * 1.15), widen=false,
        framestyle=:box, legend=false,
        # Light lines at every tick (on unless a config sets gridlines = false).
        grid=gridlines, gridcolor=:gray, gridalpha=0.25, gridlinewidth=0.6st.k, gridstyle=:solid,
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
        grid_lines!(p, [(s, efficiency(s), nothing) for s in series if efficiency(s) !== nothing], st)
        return p
    end
    y, e = metric == "throughput" ? (:h, :hsd) : (:t, :tsd)
    if zero_log
        hi = 1.5series_ymax(series, y, e)
        # Transform the data and error-bar endpoints, while labeling real values.
        exponent = floor(Int, log10(hi))
        ticks = filter(v -> log1p(v) / log1p(hi) >= 0.18,
            10.0 .^ collect(min(0, exponent):exponent))
        isempty(ticks) && push!(ticks, hi)
        ticks = [0.0; ticks]
        plot!(p; ylims=(0, log1p(hi)), yscale=:identity,
            yticks=(log1p.(ticks), compact_tick.(ticks)))
        grid_lines!(p, map(series) do s
            v, err = getfield.(s.agg, y), getfield.(s.agg, e)
            (s, log1p.(v), (log1p.(v) .- log1p.(max.(v .- err, 0)),
                log1p.(v .+ err) .- log1p.(v)))
        end, st)
        return p
    end
    ylims = log_values ?
        (series_ymin_positive(series, y, e) / 1.5, series_ymax(series, y, e) * 1.5) :
        positive_ylim(series_ymax(series, y, e); pad=0.2)
    plot!(p; ylims, yscale=log_values ? :log10 : :identity)
    plot!(p; yformatter=compact_tick)
    above = [getfield(x, y) for s in series for x in s.agg if split !== nothing && getfield(x, y) > split]
    (log_values || isempty(above)) || return split_panel(p, series, y, e, split, gpus, st; pad=split_pad)
    grid_lines!(p, [(s, getfield.(s.agg, y), getfield.(s.agg, e)) for s in series], st)
    return p
end

# Broken y axis in one panel: 0..split fills the bottom 70%, split..max is
# compressed into the top 30%, and a mark on the left axis shows the break.
function split_panel(p, series, y, e, split, gpus, st; frac=0.7, pad=0.05)
    hi = series_ymax(series, y, e) * (1 + pad)
    # Data never lands in the gap: 0..split ends at its bottom, split..max starts at its top.
    gap = 0.045
    t(v) = v <= split ? (frac - gap) * v / split :
        frac + gap + (1 - frac - gap) * (v - split) / (hi - split)
    step(lo, hi, n) = first(filter(s -> (hi - lo) / s <= n, [m * 10.0^k for k in 0:9 for m in (1, 2, 5)]))
    below = collect(0:step(0, split, 4):(0.999split))
    s_hi = step(split, hi, 3)
    ticks = [below; collect((ceil(split / s_hi) * s_hi):s_hi:hi)]
    plot!(p; ylims=(0, 1), yticks=(t.(ticks), compact_tick.(ticks)))
    grid_lines!(p, map(series) do s
        v, err = getfield.(s.agg, y), getfield.(s.agg, e)
        (s, t.(v), (t.(v) .- t.(max.(v .- err, 0)), t.(v .+ err) .- t.(v)))
    end, st)
    # GR draws the frame over everything, so a split panel draws its own: full
    # top and bottom lines, left and right lines cut at the break with // marks,
    # and inward tick marks like the :box frame of the other panels.
    plot!(p; framestyle=:grid)
    # The drawn frame sits at the usual x limits; the plot area extends a little
    # past it so the outer half of each slash isn't clipped.
    (x0, x1), lw = (minimum(gpus) / 1.15, maximum(gpus) * 1.15), 0.7st.k
    edge(xs, ys) = plot!(p, xs, ys; color=INK, lw, label="")
    tick = 2^(0.02 * log2(x1 / x0))   # inward tick length on the log2 x axis
    slash = 2^0.08
    plot!(p; xlims=(x0 / slash^1.2, x1 * slash^1.2))
    edge([x0, x1], [0, 0]); edge([x0, x1], [1, 1])
    for (x, inward) in ((x0, tick), (x1, 1 / tick))
        edge([x, x], [0, frac - gap]); edge([x, x], [frac + gap, 1])
        # Each slash is centered on the axis line, at the start and the end of the gap.
        for dy in (-gap, gap)
            plot!(p, [x / slash, x * slash], [frac + dy - 0.025, frac + dy + 0.025];
                color=INK, lw=1.8lw, label="")
        end
        for v in t.(ticks)
            edge([x, x * inward], [v, v])
        end
    end
    for g in gpus
        edge([g, g], [0, 0.03]); edge([g, g], [0.97, 1])
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

# `pad` adds that many px between columns (for fonts the width estimate runs short on).
function legend_rows(series, width, st; pad=0)
    d = legend_dims(st)
    d = merge(d, (; gap=d.gap + pad))
    cols = something(findlast(c -> sum(legend_col_widths(series, c, d)) <= width - LEGEND_INSET_PX,
        1:length(series)), 1)
    # Same row count, entries spread evenly (5 -> 3 + 2, not 4 + 1).
    cols = cld(length(series), cld(length(series), cols))
    return [series[i:min(i + cols - 1, end)] for i in 1:cols:length(series)]
end

# `slot_h` set: the legend fills an empty grid slot, rows hanging from the top.
# `shift` moves the rows up by that fraction of the legend's height.
# `center` centers the columns horizontally; `justify` spreads one row across the width. `nrows` spaces rows as if there
# were that many (room for extra lines, like a legend title).
function grid_legend(rows, width, st; slot_h=nothing, shift=0.0, center=false, nrows=length(rows), pad=0,
        justify=false, char=nothing, errorbars=false)
    d = legend_dims(st)
    # `char`: per-character width override, where the default estimate runs short.
    d = merge(d, (; gap=d.gap + pad, char=something(char, d.char)))
    col_x = cumsum([0.0; legend_col_widths(reduce(vcat, rows), length(first(rows)), d)])
    center && (col_x .+= max(0, (width - LEGEND_INSET_PX - col_x[end]) / 2))
    # `justify` (one row): spread the entries over the full width, equal gaps between.
    if justify && length(rows) == 1 && length(only(rows)) > 1
        w = [entry_px(s, d) for s in only(rows)]
        extra = max(0, (width - LEGEND_INSET_PX - sum(w)) / (length(w) - 1))
        col_x = cumsum([0.0; w .+ extra])
    end
    pl = plot(; framestyle=:none, grid=false, ticks=false, legend=false,
        xlims=(0, width - LEGEND_INSET_PX), ylims=(0, 1), widen=false,
        # Cancel GR's 2mm padding; at the canvas bottom, stop short so rounding
        # cannot push the viewport off the canvas (GR then draws it elsewhere).
        margin=0Plots.mm, top_margin=-2Plots.mm, bottom_margin=-1.9Plots.mm)
    for (r, row) in enumerate(rows)
        # In a slot, GR padding shrinks the height; spread rows instead.
        step = slot_h === nothing ? 1 / nrows : min(1 / nrows, 1.6d.row / slot_h)
        y = 1 - (r - 0.5) * step + shift
        for (c, s) in enumerate(row)
            x = col_x[c]
            scale = st.legend_k / st.k
            plot!(pl, [x, x + d.swatch], [y, y]; color=s.color, lw=0.9scale * st.lw, ls=s.ls,
                label="")
            hollow_marker!(pl, [x + d.swatch / 2], [y], s, 0.65scale * st.ms)
            if errorbars
                # Illustrative whiskers extend beyond the marker, even when
                # the measured errors in the panels are too small to see.
                mid, cap, err = x + d.swatch / 2, 0.10d.swatch, 0.32step
                plot!(pl, [mid, mid], [y - err, y + err];
                    color=s.color, lw=0.6scale * st.lw, label="")
                for endpoint in (y - err, y + err)
                    plot!(pl, [mid - cap, mid + cap], [endpoint, endpoint];
                        color=s.color, lw=0.6scale * st.lw, label="")
                end
            end
            annotate!(pl, x + d.swatch + 4, y, text(s.label, st.legend, :black, :left))
        end
    end
    return pl
end

# `n` panels in `columns` columns plus a shared legend. `make_panel(i; first_col,
# last_row, bottom_row, fix)` draws panel i. `center` centers a legend row.
function grid_layout(make_panel, n, legend_series, columns; panel_w, panel_h, st, center=false)
    rows = cld(n, columns)
    width = panel_w * columns
    # An empty grid slot holds the legend; otherwise it gets a row underneath.
    in_slot = rows * columns > n
    rows_legend = legend_rows(legend_series, in_slot ? panel_w : width, st)
    # No figure title: the y-axis label already names the metric.
    legend_h = in_slot ? 0 : legend_dims(st).row * length(rows_legend) + 4
    height = rows * panel_h + legend_h
    fix = gr_margin_fix(width, height, st)

    plots = Any[make_panel(i; first_col=(i - 1) % columns == 0, last_row=i > n - columns,
                    bottom_row=i > (rows - 1) * columns, fix)
                for i in 1:n]
    in_slot && push!(plots, grid_legend(rows_legend, panel_w, st; slot_h=panel_h))
    for _ in (length(plots) + 1):(rows * columns)
        push!(plots, plot(; framestyle=:none))
    end
    # One grid for all panels, so margins align per column (same panel widths).
    fig_kw = (size=(width, height), dpi=200, background_color=:white)
    in_slot && return plot(plots...; layout=grid(rows, columns), fig_kw...)
    body = plot(plots...; layout=grid(rows, columns))
    return plot(body, grid_legend(rows_legend, width, st; center);
        layout=grid(2, 1; heights=[rows * panel_h, legend_h] ./ height), fig_kw...)
end

# Panel `note`: { text, model (colors it like that model's series), at = [x, y] as
# fractions of the panel, default bottom right }.
function panel_note!(p, note, log_y, st)
    note === nothing && return p
    model = display_name(get(note, "model", ""))
    color = something(get(note, "color", nothing),
        get(Dict(f[2] => f[3] for f in REF_FAMILIES), model, nothing),
        model == CUNUMERIC_NAME ? COLOR_CUNUMERIC : INK)
    fx, fy = get(note, "at", [0.97, 0.02])
    at(lims, f, log) = log ? exp2(log2(lims[1]) + f * (log2(lims[2]) - log2(lims[1]))) :
                       lims[1] + f * (lims[2] - lims[1])
    y = log_y ? exp10(log10(ylims(p)[1]) + fy * (log10(ylims(p)[2]) - log10(ylims(p)[1]))) :
        at(ylims(p), fy, false)
    annotate!(p, at(xlims(p), fx, true), y, text(note["text"], st.tick, color, fx > 0.5 ? :right : :left, :bottom))
    return p
end

function grid_figure(panels, metric, columns; panel_w, panel_h, st, gridlines=true)
    # Same label always has the same style, so one shared legend covers all panels.
    legend_series = unique(s -> s.label, [s for p in panels for s in p.series])
    draw(i; kw...) = panel_note!(panel_plot(panels[i].series, metric; title=panels[i].title,
            log_values=panels[i].log, split=get(get(panels[i], :split, Dict()), metric, nothing),
            gridlines, st, kw...),
        get(panels[i], :note, nothing), panels[i].log && metric != "efficiency", st)
    return grid_layout(draw, length(panels), legend_series, columns; panel_w, panel_h, st)
end

# cuNumeric speedups for the paper text: (label, GPU counts compared).
# Speedup = reference time / cuNumeric time at the same GPU count; panels
# already guarantee equal problem sizes per GPU count.
const SPEEDUP_REFERENCES = (
    (CUPYNUMERIC_NAME, :all), ("JACC.jl", :all), ("Dagger.jl", :all), ("CUDA.jl", 1),
)

geomean(x) = exp(sum(log, x) / length(x))
fmt_x(x) = string(round(x; sigdigits=3), "×")

function speedups(panel, reference, gpus)
    find(label) = findfirst(s -> s.label == label, panel.series)
    i, j = find(CUNUMERIC_NAME), find(reference)
    (i === nothing || j === nothing) && return nothing
    ours = Dict(x.gpus => x.t for x in panel.series[i].agg)
    theirs = Dict(x.gpus => x.t for x in panel.series[j].agg)
    shared = sort([g for g in keys(ours) if haskey(theirs, g) && (gpus === :all || g == gpus)])
    isempty(shared) && return nothing
    return [(gpus=g, speedup=theirs[g] / ours[g]) for g in shared]
end

# Per benchmark: geomean over shared GPU counts; overall: geomean of those.
function speedup_summary(panels)
    lines = ["# $CUNUMERIC_NAME speedup summary", ""]
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
    isfile(cfg.config) || error("grid config not found: $(cfg.config)")
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
         log=get(p, "log", false),
         split=get(p, "split", Dict()),
         note=get(p, "note", nothing))
    end
    isempty(panels) && error("no [[panel]] entries in $(cfg.config)")

    mkpath(cfg.out_dir)
    for m in metrics
        out = joinpath(cfg.out_dir, "grid_$(m).$(format)")
        fig = grid_figure(panels, m, columns; panel_w, panel_h, st,
            gridlines=get(raw, "gridlines", true))
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
