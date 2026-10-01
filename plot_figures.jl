#!/usr/bin/env julia
# Standalone figures listed in configs/plots/figures.toml, drawn in the grid's
# style. Several fusion settings: one file per metric, one panel per setting.
# One setting: one file, one panel per metric.
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
    plots = [panel_plot(series, m; title=i == cld(length(metrics), 2) ? title : "",
                 log_values=false, first_col=true, last_row=true, bottom_row=true, fix, st)
             for (i, m) in enumerate(metrics)]
    return plot(plot(plots...; layout=grid(1, length(metrics))), grid_legend(rows_legend, width, st);
        layout=grid(2, 1; heights=[panel_h, legend_h] ./ height),
        size=(width, height), dpi=200, background_color=:white)
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
    panel_w, panel_h = get(raw, "panel_size", [400, 300])
    st = grid_style(get(raw, "font_size", 11); legend_scale=get(raw, "legend_scale", 1.0))
    format = get(raw, "format", "pdf")
    mkpath(out_dir)
    for f in get(raw, "figure", [])
        dir = csv_dir(f["results"])
        cfg_path = isabspath(f["config"]) ? f["config"] : joinpath(@__DIR__, f["config"])
        for (group, members) in parse_plot_groups(cfg_path)
            panels = map(aslist(get(f, "fusion", "both"))) do fusion
                series = group_series(dir, group, members; fusion)
                validate_series_sizes(series)
                (; series, title=group_title(group) * FUSION_TAG[fusion], log=false)
            end
            all(p -> isempty(p.series), panels) && continue
            name = get(f, "name", group)
            if length(panels) == 1
                fig = metric_row(only(panels).series, metrics, only(panels).title; panel_w, panel_h, st)
                out = joinpath(out_dir, "$(name).$(format)")
                savefig(fig, out)
                println("wrote $out")
                continue
            end
            for m in metrics
                fig = grid_figure(panels, m, length(panels); panel_w, panel_h, st)
                out = joinpath(out_dir, "$(name)_$(m).$(format)")
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
