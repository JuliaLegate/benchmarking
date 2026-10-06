#!/usr/bin/env julia
# Standalone figures from configs/plots/figures.toml, in the grid's style.
#   julia --project=. plotter/plot_figures.jl [--config=PATH] [--out=DIR]

include(joinpath(@__DIR__, "plot_grid.jl"))

const FUSION_TAG = Dict("on" => " (fused)", "off" => " (unfused)", "both" => "")

# One row of metric panels with a shared legend.
function metric_row(series, metrics, title; unit, panel_w, panel_h, st)
    width = panel_w * length(metrics)
    rows_legend = legend_rows(series, width, st)
    legend_h = legend_dims(st).row * length(rows_legend) + 4
    height = panel_h + legend_h
    fix = gr_margin_fix(width, height, st)
    # No title: panels are titled by metric instead of a y-label.
    by_metric = isempty(title)
    plots = [panel_plot(series, m; title=by_metric ? METRICS[m].ylabel :
                     i == cld(length(metrics), 2) ? title : "",
                 ylabel=m == "throughput" ? unit : nothing,
                 log_values=false, first_col=!by_metric, last_row=true, bottom_row=true, fix, st)
             for (i, m) in enumerate(metrics)]
    by_metric || foreach(p -> plot!(p; right_margin=2.5st.k * Plots.mm), plots[1:(end - 1)])
    by_metric && plot!(plots[end]; right_margin=4st.k * Plots.mm)
    # Lift a single legend row toward the x labels.
    shift = length(rows_legend) == 1 ? 0.17 : 0.0
    return plot(plot(plots...; layout=grid(1, length(metrics))), grid_legend(rows_legend, width, st; shift);
        layout=grid(2, 1; heights=[panel_h, legend_h] ./ height),
        size=(width, height), dpi=200, background_color=:white)
end

# One panel with the legend stacked on its right, under optional title lines.
function legend_right(series, metric; unit, panel_w, panel_h, st, legend_title=String[])
    width = 2panel_w
    d = legend_dims(st)
    widest = max(maximum(s -> entry_px(s, d), series),
        maximum(l -> d.char * length(l), legend_title; init=0.0))
    legend_frac = clamp((widest + d.gap + 30) / width, 0.25, 0.5)
    fix = gr_margin_fix(width, panel_h, st)
    p = panel_plot(series, metric; title="", log_values=false,
        ylabel=metric == "throughput" ? unit : nothing,
        first_col=true, last_row=true, bottom_row=true, fix, st)
    plot!(p; top_margin=2Plots.mm)
    rows = length(series) + length(legend_title)
    step = min(1 / rows, 1.6d.row / panel_h)
    legend = grid_legend([[s] for s in series], legend_frac * width + 0.7LEGEND_INSET_PX, st;
        slot_h=panel_h, shift=-step * length(legend_title), nrows=rows)
    for (k, line) in enumerate(legend_title)
        # Int size: Plots reads a Float as an angle.
        annotate!(legend, 0, 1 - (k - 0.5) * step, text(line, round(Int, 0.85st.legend), :black, :left))
    end
    return plot(p, legend; layout=grid(1, 2; widths=[1 - legend_frac, legend_frac]),
        size=(width, panel_h), dpi=200, background_color=:white)
end

# Light (low) to dark (high): gold, orange, red, purple, blue, navy.
const VALUE_GRADIENT = cgrad([colorant"#E3A008", colorant"#F28E2B", colorant"#E15759",
    colorant"#C0307A", colorant"#7B3294", colorant"#3A4FB4", colorant"#0B1F4F"])

# One panel colored by a per-series value (`vals`), a color bar, and a legend row
# below. Distinct values are evenly spaced on the gradient; ties share a color.
function colorbar_figure(series, metric, vals, cbar_label; unit, log_values=false, panel_w, panel_h, st)
    width = 2panel_w
    series = sort(series; by=s -> get(vals, s.label, Inf))
    tick_vals = unique(get(vals, s.label, NaN) for s in series)
    m = length(tick_vals)
    tick_pos = m == 1 ? [0.0] : collect(range(0, 1; length=m))
    at = Dict(zip(tick_vals, tick_pos))
    series = [merge(s, (; color=get(VALUE_GRADIENT, at[get(vals, s.label, NaN)]), ls=:solid))
              for s in series]
    lst = merge(st, (; legend=st.legend - 1, legend_k=0.7st.legend_k))
    d = legend_dims(lst)
    legend_h = 0.8d.row
    # Taller, for the color bar's title.
    height = 1.15panel_h + legend_h
    fix = gr_margin_fix(width, height, st)
    p = panel_plot(series, metric; title="", log_values,
        ylabel=metric == "throughput" ? unit : nothing,
        first_col=true, last_row=true, bottom_row=true, fix, st)
    # Room for the legend under the x label.
    bottom = (fix.bottom + fix.guide + legend_h) * Plots.px
    plot!(p; top_margin=2Plots.mm, bottom_margin=bottom)
    ys = range(0, 1; length=200)
    cbar = heatmap([0, 1], ys, repeat(collect(ys), 1, 2); c=VALUE_GRADIENT, clims=(0, 1),
        colorbar=false, legend=false, xticks=false, ymirror=true, widen=false,
        ylims=(0, 1), framestyle=:box, yticks=(tick_pos, compact_tick.(tick_vals)),
        yguide=cbar_label, yguidefontsize=st.tick, tickfontsize=st.tick,
        left_margin=-2Plots.mm, top_margin=2Plots.mm,
        right_margin=1.2text_px(st.tick) * Plots.px,
        bottom_margin=bottom)
    f = plot(p, cbar; layout=grid(1, 2; widths=[0.9, 0.1]), size=(width, height), dpi=200,
        background_color=:white)
    legend_row!(f, series, width, height, legend_h, lst, d;
        below_axis=1.15 * (text_px(st.tick) + text_px(st.guide)))
    return f
end

# Legend row in an inset under subplot 1, spanning the figure: a stacked subplot
# would share the panel's margins and get only its plot width.
function legend_row!(f, series, width, height, legend_h, st, d; below_axis)
    Plots.prepare_output(f)
    pa = Plots.plotarea(f[1])
    l, t, w, h = (x.value for x in (Plots.left(pa), Plots.top(pa), Plots.width(pa), Plots.height(pa)))
    mm(px) = px * 25.4 / 100   # Plots px are 1/100 inch
    edge = 2.0                 # mm
    strip_mm = mm(width) - 2edge
    top = t + h + mm(below_axis)
    plot!(f; inset=(1, bbox((edge - l) / w, (top - t) / h, strip_mm / w, mm(legend_h) / h,
            :left, :top)),
        subplot=3, framestyle=:none, grid=false, ticks=false, legend=false,
        xlims=(0, strip_mm * 100 / 25.4), ylims=(0, 1), widen=false,
        background_color_subplot=:transparent)
    strip = strip_mm * 100 / 25.4
    char = 0.85st.legend   # px per character
    widths = [d.swatch + 4 + char * length(s.label) for s in series]
    gap = 16.0   # px
    x = max(0, (strip - sum(widths) - gap * (length(widths) - 1)) / 2)
    scale = st.legend_k / st.k
    for (s, wi) in zip(series, widths)
        plot!(f, [x, x + d.swatch], [0.5, 0.5]; subplot=3, color=s.color,
            lw=0.9scale * st.lw, label="")
        hollow_marker!(f, [x + d.swatch / 2], [0.5], s, 0.65scale * st.ms; subplot=3)
        annotate!(f, x + d.swatch + 4, 0.5, text(s.label, st.legend, :black, :left); subplot=3)
        x += wi + gap
    end
    return f
end

# `panels`: one panel per entry ({ config, results, fusion, metric, title, log }),
# each from its own results, with one shared legend (`legend_columns` wide).
function combined_figure(f, sizing)
    panels = map(f["panels"]) do pf
        cfg = isabspath(pf["config"]) ? pf["config"] : joinpath(BENCH_ROOT, pf["config"])
        group, members = first(parse_plot_groups(cfg))
        series = group_series(csv_dir(pf["results"]), group, members; fusion=get(pf, "fusion", "both"))
        validate_series_sizes(series)
        # Short dashes: GR's long ones are hard to read over nearby solid lines.
        series = [merge(s, (; ls=s.ls == :dash ? :dot : s.ls)) for s in series]
        (; series, metric=get(pf, "metric", "time"), title=get(pf, "title", group_title(group)),
            unit=throughput_unit(group), log=get(pf, "log", false))
    end
    legend_series = unique(s -> s.label, [s for p in panels for s in p.series])
    sz = sizing(length(panels))
    draw(i; first_col, last_row, bottom_row, fix) = panel_plot(panels[i].series, panels[i].metric;
        title=panels[i].title, log_values=panels[i].log, first_col=true, last_row, bottom_row, fix, st=sz.st,
        ylabel=panels[i].metric == "throughput" ? panels[i].unit : METRICS[panels[i].metric].ylabel)
    return grid_layout(draw, length(panels), legend_series, length(panels);
        panel_w=sz.panel_w, panel_h=sz.panel_h, st=sz.st, center=true,
        legend_cols=get(f, "legend_columns", nothing))
end

function figures_main(args=ARGS)
    config = joinpath(BENCH_ROOT, "configs", "plots", "figures.toml")
    out_dir = joinpath(BENCH_ROOT, "plots", "figures")
    for arg in args
        if startswith(arg, "--config=")
            config = last(split(arg, "="; limit=2))
        elseif startswith(arg, "--out=")
            out_dir = last(split(arg, "="; limit=2))
        else
            error("unknown argument: $arg")
        end
    end
    isfile(config) || error("figures config not found: $config")
    raw = TOML.parsefile(config)
    metrics = string.(get(raw, "metrics", ["throughput", "time", "efficiency"]))
    format = get(raw, "format", "pdf")
    # With width_in, a row prints at that width (as in the grid).
    function sizing(columns)
        w, h = get(raw, "panel_size", [400, 300])
        font_size = get(raw, "font_size", 11)
        if haskey(raw, "width_in")
            w, h = 144raw["width_in"] / columns, h * (144raw["width_in"] / columns) / w
            font_size *= 2
        end
        return (; panel_w=w, panel_h=h,
            st=grid_style(font_size; legend_scale=get(raw, "legend_scale", 1.0)))
    end
    mkpath(out_dir)
    for f in get(raw, "figure", [])
        if haskey(f, "panels")
            out = joinpath(out_dir, "$(f["name"]).$(format)")
            savefig(scale_open_markers!(combined_figure(f, sizing)), out)
            println("wrote $out")
            continue
        end
        dir = csv_dir(f["results"])
        cfg_path = isabspath(f["config"]) ? f["config"] : joinpath(BENCH_ROOT, f["config"])
        for (group, members) in parse_plot_groups(cfg_path)
            panels = map(aslist(get(f, "fusion", "both"))) do fusion
                series = group_series(dir, group, members; fusion)
                relabel = Dict(display_name(k) => v for (k, v) in get(get(f, "labels", Dict()), fusion, Dict()))
                series = [merge(s, (; label=get(relabel, s.label, s.label))) for s in series]
                # Sort by the first number in a label; others keep their order.
                labelnum(s) = (m = match(r"\d+(\.\d+)?", s.label); m === nothing ? Inf : parse(Float64, m.match))
                sort!(series; by=labelnum)
                validate_series_sizes(series)
                base = get(f, "title", group_title(group))
                tag = FUSION_TAG[fusion]
                title = isempty(base) && !isempty(tag) ? uppercasefirst(strip(tag, [' ', '(', ')'])) : base * tag
                (; series, title, log=false)
            end
            all(p -> isempty(p.series), panels) && continue
            name = get(f, "name", group)
            if length(panels) == 1
                row = string.(get(f, "metrics", metrics))
                fig = metric_row(only(panels).series, row, only(panels).title;
                    unit=throughput_unit(group), sizing(length(row))...)
                out = joinpath(out_dir, "$(name).$(format)")
                savefig(scale_open_markers!(fig), out)
                println("wrote $out")
                continue
            end
            for (fusion, p) in zip(aslist(get(f, "fusion", "both")), panels), m in metrics
                isempty(p.series) && continue
                fig = haskey(f, "colorbar") ?
                    colorbar_figure(p.series, m, f["colorbar"]["values"], f["colorbar"]["label"];
                        unit=throughput_unit(group), log_values=m in aslist(get(f, "log", String[])),
                        sizing(2)...) :
                    legend_right(p.series, m; unit=throughput_unit(group), sizing(2)...,
                        legend_title=string.(aslist(get(f, "legend_title", String[]))))
                tag = Dict("on" => "_fused", "off" => "_unfused", "both" => "")[fusion]
                out = joinpath(out_dir, "$(name)_$(m)$(tag).$(format)")
                savefig(scale_open_markers!(fig), out)
                println("wrote $out")
            end
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    figures_main()
end
