#!/usr/bin/env julia
# Standalone figures listed in configs/plots/figures.toml, drawn in the grid's
# style: one file per metric, one panel per fusion setting, side by side.
#   julia --project=. plot_figures.jl [--config=PATH] [--out=DIR]

include(joinpath(@__DIR__, "plot_grid.jl"))

const FUSION_TAG = Dict("on" => " (fused)", "off" => " (unfused)", "both" => "")

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
            for m in metrics
                fig = grid_figure(panels, m, length(panels); panel_w, panel_h, st)
                out = joinpath(out_dir, "$(group)_$(m).$(format)")
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
