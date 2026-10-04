#!/usr/bin/env julia
# Standalone figures listed in configs/plots/figures.toml, drawn in the grid's
# style. Several fusion settings: one file per metric and setting, a single
# untitled panel with the legend stacked on its right. One setting: one file,
# one panel per metric.
#   julia --project=. plot_figures.jl [--config=PATH] [--out=DIR]

include(joinpath(@__DIR__, "plot_grid.jl"))

const FUSION_TAG = Dict("on" => " (fused)", "off" => " (unfused)", "both" => "")

# One row of metric panels sharing a legend; the title sits over the middle panel.
function metric_row(series, metrics, title; panel_w, panel_h, st)
    width = panel_w * length(metrics)
    rows_legend = legend_rows(series, width, st)
    legend_h = legend_dims(st).row * length(rows_legend) + 4
    height = panel_h + legend_h
    fix = gr_margin_fix(width, height, st)
    # No figure title: each panel is titled by its metric instead of a rotated
    # y-label, which keeps the panels wide. Otherwise the title sits mid-row.
    by_metric = isempty(title)
    plots = [panel_plot(series, m; title=by_metric ? METRICS[m].ylabel :
                     i == cld(length(metrics), 2) ? title : "",
                 log_values=false, first_col=!by_metric, last_row=true, bottom_row=true, fix, st)
             for (i, m) in enumerate(metrics)]
    # With rotated y-labels, leave a gap before the next panel's label.
    by_metric || foreach(p -> plot!(p; right_margin=2.5st.k * Plots.mm), plots[1:(end - 1)])
    # Centered panel titles can run past the last panel's right edge.
    by_metric && plot!(plots[end]; right_margin=4st.k * Plots.mm)
    # One legend row: lift it toward the x-labels (the slot's top quarter is blank).
    shift = length(rows_legend) == 1 ? 0.17 : 0.0
    return plot(plot(plots...; layout=grid(1, length(metrics))), grid_legend(rows_legend, width, st; shift);
        layout=grid(2, 1; heights=[panel_h, legend_h] ./ height),
        size=(width, height), dpi=200, background_color=:white)
end

# One untitled panel with the legend stacked down its right side.
# One untitled panel with the legend stacked down its right side; `legend_title`
# lines sit above the entries.
function legend_right(series, metric; panel_w, panel_h, st, legend_title=String[])
    width = 2panel_w   # a full row's width: the panel plus the legend column
    # Legend column as wide as its longest entry or title line needs.
    d = legend_dims(st)
    widest = max(maximum(s -> entry_px(s, d), series),
        maximum(l -> d.char * length(l), legend_title; init=0.0))
    legend_frac = clamp((widest + d.gap + 30) / width, 0.25, 0.5)
    fix = gr_margin_fix(width, panel_h, st)
    p = panel_plot(series, metric; title="", log_values=false,
        first_col=true, last_row=true, bottom_row=true, fix, st)
    plot!(p; top_margin=2Plots.mm)
    # Title lines take the first rows; entries shift down below them.
    rows = length(series) + length(legend_title)
    step = min(1 / rows, 1.6d.row / panel_h)
    # grid_legend subtracts a full-width row's padding; a narrow column has less.
    legend = grid_legend([[s] for s in series], legend_frac * width + 0.7LEGEND_INSET_PX, st;
        slot_h=panel_h, shift=-step * length(legend_title), nrows=rows)
    for (k, line) in enumerate(legend_title)
        # A bit smaller than the entries. Int: Plots reads a Float text size as an angle.
        annotate!(legend, 0, 1 - (k - 0.5) * step, text(line, round(Int, 0.85st.legend), :black, :left))
    end
    return plot(p, legend; layout=grid(1, 2; widths=[1 - legend_frac, legend_frac]),
        size=(width, panel_h), dpi=200, background_color=:white)
end

function figures_main(args=ARGS)
    config = joinpath(@__DIR__, "configs", "plots", "figures.toml")
    out_dir = joinpath(@__DIR__, "plots", "figures")
    for arg in args
        if startswith(arg, "--config=")
            config = last(split(arg, "="; limit=2))
        elseif startswith(arg, "--out=")
            out_dir = last(split(arg, "="; limit=2))
        else
            error("unknown argument: $arg")
        end
    end
    isfile(config) || error("$config not found; copy configs/plots/figures.example.toml")
    raw = TOML.parsefile(config)
    metrics = string.(get(raw, "metrics", ["throughput", "time", "efficiency"]))
    format = get(raw, "format", "pdf")
    # Per-figure sizing; with width_in, the row is that printed width (as in the grid).
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
        dir = csv_dir(f["results"])
        cfg_path = isabspath(f["config"]) ? f["config"] : joinpath(@__DIR__, f["config"])
        for (group, members) in parse_plot_groups(cfg_path)
            panels = map(aslist(get(f, "fusion", "both"))) do fusion
                series = group_series(dir, group, members; fusion)
                # Optional per-setting legend text: labels.<fusion>.<label> = "new label".
                relabel = get(get(f, "labels", Dict()), fusion, Dict())
                series = [merge(s, (; label=get(relabel, s.label, s.label))) for s in series]
                # Legend ordered by the first number in each label, lowest first;
                # labels without one keep their order.
                labelnum(s) = (m = match(r"\d+(\.\d+)?", s.label); m === nothing ? Inf : parse(Float64, m.match))
                sort!(series; by=labelnum)
                validate_series_sizes(series)
                base = get(f, "title", group_title(group))
                tag = FUSION_TAG[fusion]
                # An empty title with a fusion setting leaves just "Fused" / "Unfused".
                title = isempty(base) && !isempty(tag) ? uppercasefirst(strip(tag, [' ', '(', ')'])) : base * tag
                (; series, title, log=false)
            end
            all(p -> isempty(p.series), panels) && continue
            name = get(f, "name", group)
            if length(panels) == 1
                row = string.(get(f, "metrics", metrics))
                fig = metric_row(only(panels).series, row, only(panels).title; sizing(length(row))...)
                out = joinpath(out_dir, "$(name).$(format)")
                savefig(fig, out)
                println("wrote $out")
                continue
            end
            for (fusion, p) in zip(aslist(get(f, "fusion", "both")), panels), m in metrics
                isempty(p.series) && continue
                fig = legend_right(p.series, m; sizing(2)...,
                    legend_title=string.(aslist(get(f, "legend_title", String[]))))
                tag = Dict("on" => "_fused", "off" => "_unfused", "both" => "")[fusion]
                out = joinpath(out_dir, "$(name)_$(m)$(tag).$(format)")
                savefig(fig, out)
                println("wrote $out")
            end
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    figures_main()
end
