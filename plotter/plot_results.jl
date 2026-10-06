#!/usr/bin/env julia
# Generate weak- or strong-scaling plots from benchmark CSVs.
# Each figure is one `[plot.groups]` entry (or a singleton [[benchmark]]).

using Plots
using Statistics

# The benchmark root (this file lives in plotter/).
const BENCH_ROOT = dirname(@__DIR__)

include(joinpath(BENCH_ROOT, "src", "parse_benchmarks.jl"))
include(joinpath(@__DIR__, "names.jl"))

# GPU nodes often have no display; 100 = PNG. Override with GKSwstype if needed.
get!(ENV, "GKSwstype", "100")
gr()
default(;
    fontfamily="sans-serif",
    fg=:black,
    fg_text=:black,
    fg_axis=:black,
    fg_border=:black,
    grid=false,
    gridalpha=0,
    gridlinewidth=0,
    minorgrid=false,
    legend=false,
)

function parse_args(args)
    results_dir = "results"
    out_dir = nothing
    output_suffix = nothing
    fusion = "both"
    format = "png"
    config = joinpath(BENCH_ROOT, "configs", "multi_gpu", "all.toml")

    for arg in args
        if startswith(arg, "--out=")
            out_dir = last(split(arg, "="; limit=2))
        elseif startswith(arg, "--suffix=")
            output_suffix = last(split(arg, "="; limit=2))
        elseif startswith(arg, "--config=")
            config = last(split(arg, "="; limit=2))
        elseif startswith(arg, "--format=")
            format = last(split(arg, "="; limit=2))
        elseif startswith(arg, "--fusion=")
            fusion = last(split(arg, "="; limit=2))
            fusion in ("on", "off", "both") || error("--fusion must be on, off, or both")
        else
            results_dir = arg
        end
    end
    results_dir = isabspath(results_dir) ? results_dir : joinpath(BENCH_ROOT, results_dir)
    config = isabspath(config) ? config : joinpath(BENCH_ROOT, config)
    if out_dir === nothing
        out_dir = if basename(normpath(results_dir)) == "results"
            joinpath(BENCH_ROOT, "plots")
        else
            joinpath(BENCH_ROOT, "plots", basename(normpath(results_dir)))
        end
    else
        out_dir = isabspath(out_dir) ? out_dir : joinpath(BENCH_ROOT, out_dir)
    end
    # One figure per fusion setting: default file suffix _fused / _unfused.
    if output_suffix === nothing
        output_suffix = Dict("on" => "_fused", "off" => "_unfused", "both" => "")[fusion]
    end
    return (; results_dir, out_dir, output_suffix, config, fusion, format)
end

# Fixed across every figure so GEMM / Gray-Scott / DMD read as one set.
const COLOR_CUNUMERIC = "#2a78d6"
const COLOR_CUPYNUMERIC = "#eb6834"
const COLOR_CUDA = "#1a7f37"
const COLOR_CUTENSOR = "#0d7377"
const COLOR_JACC = "#8e44ad"
const COLOR_DAGGER = "#c49a00"
const COLOR_IGG = "#6d4c41"
const MARKER_CUNUMERIC = :circle
const MARKER_CUPYNUMERIC = :rect
const MARKER_CUDA = :utriangle
const MARKER_CUTENSOR = :star5
const MARKER_JACC = :diamond
const MARKER_DAGGER = :hexagon
const MARKER_IGG = :pentagon

# Extra cuNumeric variants (Gray-Scott forms, DMD accelerated). Avoid the
# reference orange/green so CUDA.jl and cuPyNumeric stay unique.
const VARIANT_COLORS = [COLOR_CUNUMERIC, "#7b2d8e", "#b8860b", "#3d5a80", "#a23b72", "#2f6f4e"]
const VARIANT_MARKERS = [:circle, :diamond, :hexagon, :dtriangle, :star4, :pentagon]

const REF_FAMILIES = [
    ("cupynumeric", CUPYNUMERIC_NAME, COLOR_CUPYNUMERIC, MARKER_CUPYNUMERIC),
    ("CUDA.jl", "CUDA.jl", COLOR_CUDA, MARKER_CUDA),
    ("CUDA.jl_separable", "CUDA.jl (separable arrays)", COLOR_CUDA, :dtriangle),
    ("tensoroperations_cuda", "TensorOperations.jl / cuTENSOR", COLOR_CUTENSOR, MARKER_CUTENSOR),
    ("jacc", "JACC.jl", COLOR_JACC, MARKER_JACC),
    ("dagger", "Dagger.jl", COLOR_DAGGER, MARKER_DAGGER),
    ("igg", "ImplicitGlobalGrid.jl", COLOR_IGG, MARKER_IGG),
]

const INK = "#111111"
const IDEALCOL = "#6e6e6e"

const GROUP_TITLES = Dict(
    "grayscott" => "Gray-Scott",
    "dmd" => "DMD",
    "gemm" => "GEMM",
    "cg" => "CG",
    "poisson_fft" => "Poisson FFT",
    "montecarlo" => "Monte Carlo",
    "tensor_projection3" => "Tensor projection (3-mode)",
    "tensor_contract4" => "Tensor contraction (rank-4)",
    "nas_ep" => "NAS EP",
    "nas_ft" => "NAS FT",
    "nas_mg" => "NAS MG",
)

include(joinpath(BENCH_ROOT, "src", "result_rows.jl"))

_all_rows(runs) = reduce(vcat, runs; init=Row[])

function make_series(label, color, marker, ls, runs; by=:gpus)
    isempty(runs) && return nothing
    return (label=label, color=color, marker=marker, ls=ls,
        agg=aggregate(_all_rows(runs); by))
end

function load_csv_series(results_dir, bench, key, label, color, marker, ls; by=:gpus)
    path = joinpath(results_dir, "$(bench)_$(key).csv")
    isfile(path) || return nothing
    runs = load_runs(path)
    isempty(runs) && return nothing
    return make_series(label, color, marker, ls, runs; by)
end

function group_title(group)
    return get(GROUP_TITLES, group, titlecase(replace(group, '_' => ' ')))
end

# Throughput unit, as the harness reports it (`throughput_label`). Gray-Scott counts
# grid points updated, not FLOPs.
function throughput_unit(benchmark)
    startswith(benchmark, "nas_ep") && return "G random numbers/s"
    startswith(benchmark, "grayscott") && return "G cells/s"
    return "GFLOP/s"
end

# Variants are cuNumeric-only, so every label names the model.
function variant_label(group, member)
    member == group && return nothing
    prefix = group * "_"
    stem = startswith(member, prefix) ? member[(length(prefix) + 1):end] : member
    return replace(stem, '_' => ' ')
end

function cunumeric_series_label(group, member, fused)
    notes = filter(!isnothing, [variant_label(group, member), fused ? nothing : "unfused"])
    return isempty(notes) ? CUNUMERIC_NAME : "$CUNUMERIC_NAME ($(join(notes, ", ")))"
end

function overlay_refs(results_dir, members; by=:gpus)
    series = []
    seen = Set{String}()
    order = unique!(vcat([plot_baseline(members)], members))
    for (key, label, color, marker) in REF_FAMILIES
        key in seen && continue
        for member in order
            s = load_csv_series(results_dir, member, key, label, color, marker, :solid; by)
            s === nothing && continue
            push!(series, s)
            push!(seen, key)
            break
        end
    end
    return series
end

function group_series(results_dir, group, members; fusion="both", by=:gpus)
    series = []
    n_members = length(members)
    for (i, member) in enumerate(members)
        color, marker = if n_members == 1
            COLOR_CUNUMERIC, MARKER_CUNUMERIC
        else
            VARIANT_COLORS[mod1(i, length(VARIANT_COLORS))],
            VARIANT_MARKERS[mod1(i, length(VARIANT_MARKERS))]
        end
        for (key, fused, ls) in (
            ("cunumeric", true, :solid),
            ("cunumeric_nofusion", false, :dash),
        )
            fusion == "both" || fused == (fusion == "on") || continue
            label = cunumeric_series_label(group, member, fused)
            # One fusion setting per figure: name just the form, all lines solid.
            if fusion != "both" && n_members > 1
                stem = something(variant_label(group, member), CUNUMERIC_NAME)
                label = replace(stem, " accelerated" => "", "expression" => "expr")
                ls = :solid
            end
            s = load_csv_series(results_dir, member, key, label, color, marker, ls; by)
            s === nothing || push!(series, s)
        end
    end
    append!(series, overlay_refs(results_dir, members; by))
    return filter(!isnothing, series)
end

# cuNumeric saves EP as `cunumeric_struct`; otherwise the usual model series.
function ep_series(results_dir)
    entries = (
        ("cunumeric_struct", CUNUMERIC_NAME, COLOR_CUNUMERIC, MARKER_CUNUMERIC, :solid),
        ("dagger", "Dagger.jl", COLOR_DAGGER, MARKER_DAGGER, :solid),
        ("cupynumeric", CUPYNUMERIC_NAME, COLOR_CUPYNUMERIC, MARKER_CUPYNUMERIC, :solid),
        ("CUDA.jl", "CUDA.jl", COLOR_CUDA, MARKER_CUDA, :solid),
        ("jacc", "JACC.jl", COLOR_JACC, MARKER_JACC, :solid),
    )
    series = []
    for (key, label, color, marker, ls) in entries
        s = load_csv_series(results_dir, "nas_ep", key, label, color, marker, ls)
        s === nothing || push!(series, s)
    end
    return series
end

function addline!(p, s, y; kw...)
    return plot!(p, getfield.(s.agg, :gpus), y; color=s.color,
        lw=2.6, ls=s.ls, marker=s.marker, ms=7, msc=s.color, markerstrokewidth=0.7,
        label=s.label, kw...)
end

# Long variant labels ("function accelerated, unfused") need wider columns.
function legend_cols(series)
    longest = maximum(s -> length(s.label), series; init=0)
    return min(max(length(series), 1), longest > 30 ? 3 : longest <= 12 ? 5 : 4)
end

function build_legend(series)
    n = length(series)
    cols = legend_cols(series)
    rows = cld(n, cols)
    slot = min(0.32, 0.92 / cols)
    x0 = (1 - cols * slot) / 2
    step = rows > 1 ? min(0.36, 0.8 / (rows - 1)) : 0.0
    y0 = 0.50 + step * (rows - 1) / 2

    pl = plot(;
        framestyle=:none, grid=false, ticks=false, legend=false,
        xlims=(0, 1), ylims=(0, 1), widen=false,
        left_margin=0Plots.mm, right_margin=0Plots.mm,
        top_margin=0Plots.mm, bottom_margin=0Plots.mm,
        background_color=:transparent,
    )
    # Pin the coordinate system so later scatter/annotate cannot rescale it.
    scatter!(pl, [0.0, 1.0], [0.0, 1.0]; ms=0, msw=0, mc=:white, label="")

    for (i, s) in enumerate(series)
        r, c = divrem(i - 1, cols)
        x = x0 + c * slot
        y = y0 - r * step
        plot!(pl, [x, x + 0.028], [y, y]; color=s.color, lw=2.8, ls=s.ls, label="")
        scatter!(pl, [x + 0.014], [y]; color=s.color, marker=s.marker,
            ms=7, msc=s.color, markerstrokewidth=0.6, label="")
        annotate!(pl, x + 0.036, y, text(s.label, 12, :black, :left))
    end
    plot!(pl; xlims=(0, 1), ylims=(0, 1), widen=false)
    return pl
end

function series_ymax(series, yfield, efield)
    m = 0.0
    for s in series, x in s.agg
        m = max(m, getfield(x, yfield) + getfield(x, efield))
    end
    return m
end

function series_ymin_positive(series, yfield, efield)
    values = [max(eps(Float64), getfield(x, yfield) - getfield(x, efield))
              for s in series for x in s.agg]
    return minimum(values)
end

function positive_ylim(hi; pad=0.18)
    hi > 0 || return (0, 1)
    return (0, hi * (1 + pad))
end

# Strong = one size at every GPU count. Efficiency is throughput-based either way.
function scaling_kind(series)
    sweep = any(length(s.agg) > 1 for s in series)
    fixed = all(length(unique((x.N, x.M) for x in s.agg)) == 1 for s in series)
    return sweep && fixed ? "strong" : "weak"
end

function weak_scaling_figure(series; plot_title, log_values=false, unit="GFLOP/s")
    kind = scaling_kind(series)
    common = (
        xscale=:log2, xticks=([1, 2, 4, 8], ["1", "2", "4", "8"]), xlabel="GPUs",
        # Light grid lines, matching plot_grid.jl.
        framestyle=:box, grid=true, gridcolor=:gray, gridalpha=0.25, gridlinewidth=0.8,
        gridstyle=:solid, minorgrid=false,
        foreground_color_axis=:black, foreground_color_border=:black,
        foreground_color_text=:black, foreground_color_guide=:black,
        tickfontcolor=:black, guidefontcolor=:black, titlefontcolor=:black,
        tickfontsize=14, guidefontsize=16, titlefontsize=16,
        xtickfontsize=14, ytickfontsize=14,
        xguidefontsize=16, yguidefontsize=16,
        legend=false, xlims=(0.85, 9.4), widen=false,
        left_margin=10Plots.mm, right_margin=6Plots.mm,
        top_margin=5Plots.mm, bottom_margin=10Plots.mm,
    )

    throughput_limits = log_values ?
        (series_ymin_positive(series, :h, :hsd) / 1.5,
         series_ymax(series, :h, :hsd) * 1.5) :
        positive_ylim(series_ymax(series, :h, :hsd); pad=0.28)
    time_limits = log_values ?
        (series_ymin_positive(series, :t, :tsd) / 1.5,
         series_ymax(series, :t, :tsd) * 1.5) :
        positive_ylim(series_ymax(series, :t, :tsd); pad=0.28)
    p1 = plot(; ylabel="Throughput ($unit" * (log_values ? ", log scale)" : ")"),
        title="Throughput", yscale=log_values ? :log10 : :identity,
        ylims=throughput_limits, common...)
    for s in series
        addline!(p1, s, getfield.(s.agg, :h); yerror=getfield.(s.agg, :hsd))
    end

    p2 = plot(; ylabel=log_values ? "Time / step (ms, log scale)" : "Time / step (ms)",
        title="Time per step", yscale=log_values ? :log10 : :identity,
        ylims=time_limits,
        common..., left_margin=28Plots.mm, yguidefontsize=15)
    for s in series
        addline!(p2, s, getfield.(s.agg, :t); yerror=getfield.(s.agg, :tsd))
    end

    efficiencies = Float64[]
    for s in series
        i1 = findfirst(x -> x.gpus == 1, s.agg)
        i1 === nothing && continue
        base = s.agg[i1].h
        append!(efficiencies, [x.h / (x.gpus * base) for x in s.agg])
    end
    p3 = plot(; ylabel="Parallel efficiency", title="$(titlecase(kind))-scaling efficiency",
        ylims=positive_ylim(
            max(1.0, isempty(efficiencies) ? 0.0 : maximum(efficiencies)); pad=0.12
        ),
        common..., left_margin=16Plots.mm)
    hline!(p3, [1.0]; color=IDEALCOL, ls=:dashdot, lw=1.6, label="")
    for s in series
        i1 = findfirst(x -> x.gpus == 1, s.agg)
        i1 === nothing && continue
        base = s.agg[i1].h
        addline!(p3, s, [x.h / (x.gpus * base) for x in s.agg])
    end

    nrows = cld(length(series), legend_cols(series))
    layout = if nrows > 2
        @layout([grid(1, 3); leg{0.24h}])
    elseif nrows > 1
        @layout([grid(1, 3); leg{0.16h}])
    else
        @layout([grid(1, 3); leg{0.14h}])
    end
    return plot(
        p1, p2, p3, build_legend(series);
        layout,
        size=(1760, nrows > 2 ? 780 : nrows > 1 ? 680 : 620), dpi=220, plot_title,
        plot_titlefontsize=20, plot_titlefontcolor=:black,
        background_color=:white,
    )
end

function main(args=ARGS)
    cfg = parse_args(args)
    if !isdir(cfg.results_dir)
        println("no results directory at $(cfg.results_dir)")
        return nothing
    end
    isfile(cfg.config) || error("plot config not found: $(cfg.config)")

    mkpath(cfg.out_dir)
    for (group, members) in parse_plot_groups(cfg.config)
        if group == "nas_ep" && members == ["nas_ep"]
            series = ep_series(cfg.results_dir)
            isempty(series) && continue
            validate_series_sizes(series)
            kind = scaling_kind(series)
            # Implementations span orders of magnitude; log axes keep all visible.
            fig = weak_scaling_figure(
                series; plot_title="NAS EP — $kind scaling", log_values=true,
                unit=throughput_unit("nas_ep"),
            )
            out = joinpath(cfg.out_dir, "nas_ep_$(kind)_scaling$(cfg.output_suffix).$(cfg.format)")
            savefig(fig, out)
            println("wrote $out")
            continue
        end
        series = group_series(cfg.results_dir, group, members; cfg.fusion)
        isempty(series) && continue
        validate_series_sizes(series)
        kind = scaling_kind(series)
        fig = weak_scaling_figure(
            series; plot_title=group_title(group) * " — $kind scaling" *
                Dict("on" => " (fused)", "off" => " (unfused)", "both" => "")[cfg.fusion],
            unit=throughput_unit(group),
        )
        out = joinpath(cfg.out_dir, "$(group)_$(kind)_scaling$(cfg.output_suffix).$(cfg.format)")
        savefig(fig, out)
        println("wrote $out")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
