#!/usr/bin/env julia
# Dagger blocks_per_gpu sensitivity: one panel per benchmark, one box per setting
# spanning the tuned GPU counts. Config: configs/plots/tunes.toml.

using TOML

# Layout, styles and legend come from the grid plots; the tune parser from the Dagger worker.
include(joinpath(@__DIR__, "plot_grid.jl"))
include(joinpath(@__DIR__, "src", "dagger", "tunes.jl"))

# Okabe-Ito (colorblind-safe), plus a distinct marker per GPU count.
const GPU_STYLES = Dict(1 => ("#E69F00", :circle), 2 => ("#56B4E9", :rect),
    4 => ("#009E73", :utriangle), 8 => ("#D55E00", :diamond))

parse_tunes_args(args) = parse_grid_args(
    vcat(["--config=configs/plots/tunes.toml", "--out=plots/tunes"], args))

# benchmark => [(gpus, blocks_per_gpu => speedup over 1 block per GPU)].
function tune_speedups(tunes)
    out = Dict{String,Vector{Tuple{Int,Dict{Int,Float64}}}}()
    for ((name, gpus, _), splits) in tunes
        haskey(splits, 1) || continue
        push!(get!(out, name, []), (gpus, Dict(b => splits[1] / ms for (b, ms) in splits)))
    end
    foreach(v -> sort!(v; by=first), values(out))
    return out
end

function gpu_series(g)
    color, marker = get(GPU_STYLES, g, (INK, :circle))
    return (label=g == 1 ? "1 GPU" : "$g GPUs", color, marker, ls=:solid)
end

function box!(p, x, v, st; w)
    q1, med, q3 = quantile(v, (0.25, 0.5, 0.75))
    lo, hi = extrema(v)
    plot!(p, [x, x], [lo, q1]; color=INK, lw=0.8st.k, label="")
    plot!(p, [x, x], [q3, hi]; color=INK, lw=0.8st.k, label="")
    plot!(p, [x - w / 2, x + w / 2], [lo, lo]; color=INK, lw=0.8st.k, label="")
    plot!(p, [x - w / 2, x + w / 2], [hi, hi]; color=INK, lw=0.8st.k, label="")
    plot!(p, Shape([x - w, x + w, x + w, x - w], [q1, q1, q3, q3]); fillcolor="#eeeeee",
        linecolor=INK, lw=0.8st.k, label="")
    plot!(p, [x - w, x + w], [med, med]; color=INK, lw=1.6st.k, label="")
    return p
end

function tunes_panel(name, runs; first_col, last_row, bottom_row, fix, st)
    # 1 block/GPU is the baseline (always 1.0, the dashed line), so it gets no column.
    blocks = sort(unique(b for (_, s) in runs for b in keys(s) if b > 1))
    xs = log2.(blocks)
    all = [v for (_, s) in runs for v in values(s)]
    p = plot(;
        title=group_title(name), xlabel=bottom_row ? "Blocks per GPU" : "",
        ylabel=first_col ? "Speedup" : "",
        xticks=(xs, string.(blocks)), xlims=(minimum(xs) - 0.6, maximum(xs) + 0.6),
        # Same 0.5 steps and one decimal everywhere, so tick labels (and y labels) line up.
        ylims=(max(0, floor(2minimum(all)) / 2 - 0.1), ceil(2maximum(all)) / 2 + 0.1),
        yticks=0:0.5:ceil(2maximum(all)) / 2, yformatter=v -> string(round(v; digits=1)),
        widen=false,
        framestyle=:box, legend=false,
        tickfontsize=st.tick, guidefontsize=st.guide, titlefontsize=st.title,
        titlefontfamily="DejaVuSans-Bold",
        left_margin=(fix.ticks + (first_col ? fix.guide : 0)) * Plots.px +
                    (first_col ? 3st.k * Plots.mm : 0Plots.mm),
        bottom_margin=(fix.bottom + (bottom_row ? fix.guide : 0)) * Plots.px,
        top_margin=fix.top * Plots.px - 2Plots.mm, right_margin=2Plots.mm,
    )
    hline!(p, [1.0]; color=IDEALCOL, ls=:dashdot, lw=1.4st.k, label="")
    # Box widths and dot offsets are a share of the x range, so they look the same in every panel.
    span = maximum(xs) - minimum(xs) + 1.2
    for (b, x) in zip(blocks, xs)
        v = [s[b] for (_, s) in runs if haskey(s, b)]
        length(v) > 1 && box!(p, x, v, st; w=0.045span)
    end
    # One dot per GPU count, nudged apart so they don't hide each other.
    for (i, (g, s)) in enumerate(runs)
        b = sort(filter(>(1), collect(keys(s))))
        dx = (i - (length(runs) + 1) / 2) * 0.013span
        style = gpu_series(g)
        scatter!(p, log2.(b) .+ dx, [s[k] for k in b]; color=style.color, marker=style.marker,
            ms=0.55st.ms, msc=:white, markerstrokewidth=0.3st.k, label="")
    end
    return p
end

function tunes_summary(speedups, names)
    lines = ["# Dagger blocks_per_gpu sensitivity", "",
        "Speedup over the default (1 block per GPU) from the tune sweep; above 1 means " *
        "more blocks helped.", "",
        "| Benchmark | GPUs | Best blocks/GPU | Best speedup | Worst blocks/GPU | Worst speedup |",
        "|---|---:|---:|---:|---:|---:|"]
    r3(v) = round(v; digits=3)
    for name in names, (g, s) in get(speedups, name, [])
        best, worst = argmax(last, collect(s)), argmin(last, collect(s))
        push!(lines, "| $(group_title(name)) | $g | $(first(best)) | $(r3(last(best)))× | " *
            "$(first(worst)) | $(r3(last(worst)))× |")
    end
    return join(lines, "\n") * "\n"
end

function tunes_main(args=ARGS)
    cfg = parse_tunes_args(args)
    raw = TOML.parsefile(cfg.config)
    dir = get(raw, "tunes", "results-brev/tunes")
    speedups = tune_speedups(read_dagger_tunes(isabspath(dir) ? dir : joinpath(@__DIR__, dir)))
    isempty(speedups) && error("no tune results with a 1 block/GPU baseline in $dir")
    names = filter(n -> haskey(speedups, n), string.(get(raw, "benchmarks", sort(collect(keys(speedups))))))
    columns = min(get(raw, "columns", 2), length(names))
    panel_w, panel_h, st = grid_dimensions(raw, columns)
    legend_series = [gpu_series(g) for g in sort(unique(g for n in names for (g, _) in speedups[n]))]
    draw(i; kw...) = tunes_panel(names[i], speedups[names[i]]; st, kw...)

    mkpath(cfg.out_dir)
    out = joinpath(cfg.out_dir, "tunes.$(get(raw, "format", "png"))")
    savefig(grid_layout(draw, length(names), legend_series, columns; panel_w, panel_h, st), out)
    println("wrote $out")
    out = joinpath(cfg.out_dir, "tunes_summary.md")
    write(out, tunes_summary(speedups, names))
    println("wrote $out")
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    tunes_main()
end
