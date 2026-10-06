#!/usr/bin/env julia
# Single-GPU size sweeps: one figure per metric, one panel per benchmark,
# x = problem size. Config: configs/plots/sweeps.toml.

using TOML

# Series, colors, layout and legend come from the grid plots.
include(joinpath(@__DIR__, "plot_grid.jl"))

parse_sweep_args(args) = parse_grid_args(
    vcat(["--config=configs/plots/sweeps.toml", "--out=plots/sweeps"], args))

function sweep_series(panel)
    group = panel["benchmark"]
    members = String[string(m) for m in get(panel, "members", [group])]
    dir = csv_dir(panel["results"])
    series = group_series(dir, group, members; fusion=get(panel, "fusion", "both"), by=:N)
    # Reference models are saved under the group name (e.g. grayscott_CUDA.jl.csv).
    have = Set(s.label for s in series)
    append!(series, filter(s -> !(s.label in have), overlay_refs(dir, [group]; by=:N)))
    relabel = get(panel, "labels", Dict())
    series = [merge(s, (; label=get(relabel, s.label, s.label))) for s in series]
    hide = Set(string.(get(panel, "hide", String[])))
    filter!(s -> !(s.label in hide), series)
    isempty(series) && error("panel $group: no series found in $(panel["results"])")
    all(s -> all(x -> x.gpus == 1, s.agg), series) || error("panel $group: not a single-GPU sweep")
    return series
end

# Log x (sizes span decades), linear y; every panel labels its own y unit.
function sweep_panel(panel, metric; first_col, last_row, bottom_row, fix, st)
    y = metric == "throughput" ? :h : :t
    p = plot(;
        title=panel.title, xlabel=bottom_row ? panel.xlabel : "",
        ylabel=metric == "throughput" ? panel.unit : METRICS[metric].ylabel,
        xscale=:log10, yformatter=compact_tick, framestyle=:box, legend=false,
        ylims=positive_ylim(maximum(maximum(getfield.(s.agg, y)) for s in panel.series); pad=0.1),
        grid=true, gridcolor=:gray, gridalpha=0.25, gridlinewidth=0.6st.k, gridstyle=:solid,
        tickfontsize=st.tick, guidefontsize=st.guide, titlefontsize=st.title,
        titlefontfamily="DejaVuSans-Bold",
        left_margin=(fix.ticks + fix.guide) * Plots.px + 3st.k * Plots.mm,
        bottom_margin=(fix.bottom + (bottom_row ? fix.guide : 0)) * Plots.px,
        # Room for the last x tick label (10^10), which is centered on the edge.
        top_margin=fix.top * Plots.px - 2Plots.mm, right_margin=4st.k * Plots.mm,
    )
    # Dashed (unfused) last: they track MosaicPIE closely and would hide under it.
    for s in sort(panel.series; by=s -> s.ls != :solid)
        xs, ys = getfield.(s.agg, :N), getfield.(s.agg, y)
        hollow_marker!(p, xs, ys, s, st.ms)
        plot!(p, xs, ys; color=s.color, lw=st.lw, ls=s.ls, label=s.label)
    end
    return p
end

function sweeps_main(args=ARGS)
    cfg = parse_sweep_args(args)
    raw = TOML.parsefile(cfg.config)
    columns = min(get(raw, "columns", 2), length(get(raw, "panel", [])))
    panel_w, panel_h, st = grid_dimensions(raw, columns)
    format = get(raw, "format", "png")
    panels = map(get(raw, "panel", [])) do p
        (series=sweep_series(p), title=get(p, "title", group_title(p["benchmark"])),
         xlabel=get(p, "xlabel", "N"), unit=throughput_unit(p["benchmark"]))
    end
    isempty(panels) && error("no [[panel]] entries in $(cfg.config)")
    legend_series = unique(s -> s.label, [s for p in panels for s in p.series])
    # One style per label across panels (a relabeled series takes the legend's style).
    # Dashes become dots: GR's dashes are too long to read over the solid lines.
    legend_series = [merge(s, (; ls=s.ls == :dash ? :dot : s.ls)) for s in legend_series]
    style = Dict(s.label => s for s in legend_series)
    panels = [merge(p, (; series=[merge(s, (; color=style[s.label].color,
        marker=style[s.label].marker, ls=style[s.label].ls)) for s in p.series])) for p in panels]
    mkpath(cfg.out_dir)
    for m in string.(get(raw, "metrics", ["time", "throughput"]))
        draw(i; kw...) = sweep_panel(panels[i], m; st, kw...)
        out = joinpath(cfg.out_dir, "sweep_$(m).$(format)")
        fig = grid_layout(draw, length(panels), legend_series, columns; panel_w, panel_h, st)
        savefig(scale_open_markers!(fig), out)
        println("wrote $out")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    sweeps_main()
end
