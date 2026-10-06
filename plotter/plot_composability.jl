#!/usr/bin/env julia
# Assemble composability figures from independently selected backend CSVs.
#   julia --project=. plotter/plot_composability.jl {krylov|ode} --cuda=PATH ...

module ComposabilityPlot

include(joinpath(@__DIR__, "plot_grid.jl"))

const INPUT_KEYS = ("cuda", "cunumeric-single", "cunumeric-multi", "dagger-single", "dagger-multi")
const BACKENDS = (
    (key="cuda", backend="CUDA", label="CUDA.jl", color=COLOR_CUDA, marker=MARKER_CUDA),
    (key="cunumeric", backend="cuNumeric", label=CUNUMERIC_NAME, color=COLOR_CUNUMERIC, marker=MARKER_CUNUMERIC),
    (key="dagger", backend="Dagger", label="Dagger.jl", color=COLOR_DAGGER, marker=MARKER_DAGGER),
)

# Select only this source's backend: a CSV can also contain older runs of the
# other backends, or the local Krylov implementation, which must not leak in.
function read_source(path, workload, experiment, backend)
    isfile(path) || error("results CSV not found: $path")
    rows = NamedTuple[]
    open(path) do io
        eof(io) && error("empty results CSV: $path")
        header = split(strip(readline(io)), ',')
        required = ["experiment", "base_n", "backend", "eltype", "gpus", "samples_ms",
            workload == "krylov" ? "n" : "N"]
        append!(required, workload == "krylov" ? ["solver", "mode"] : ["steps"])
        all(k -> k in header, required) || error("missing $workload columns in $path")
        columns = Dict(k => i for (i, k) in enumerate(header))
        for (line_number, line) in enumerate(eachline(io))
            isempty(strip(line)) && continue
            fields = split(line, ',')
            length(fields) == length(header) || error("malformed row $(line_number + 1) in $path")
            value(k) = strip(fields[columns[k]])
            value("experiment") == experiment || continue
            source_backend = workload == "ode" && backend == "CUDA" ? "CuArray" : backend
            value("backend") == source_backend || continue
            if workload == "krylov"
                value("solver") == "cg" && value("mode") == "stock" || continue
            end
            samples = parse.(Float64, split(value("samples_ms"), ';'))
            length(samples) >= 2 && all(isfinite, samples) && all(>(0), samples) ||
                error("invalid timing samples in row $(line_number + 1) of $path")
            n = parse(Int, value(workload == "krylov" ? "n" : "N"))
            gpus = parse(Int, value("gpus"))
            n > 0 && gpus > 0 || error("nonpositive size or GPU count in $path")
            experiment == "single" && gpus != 1 && error("single-GPU row has $gpus GPUs in $path")
            base_n = experiment == "weak" ? parse(Int, value("base_n")) : nothing
            base_n === nothing || base_n > 0 || error("nonpositive weak-scaling base N in $path")
            steps = workload == "ode" ? parse(Int, value("steps")) : nothing
            steps === nothing || steps > 0 || error("nonpositive step count in $path")
            # ODE samples time the full fixed-step solve, including setup.
            # Normalize each sample before computing its mean and standard deviation.
            workload == "ode" && (samples ./= steps)
            push!(rows, (; backend, n, gpus, base_n, steps, eltype=value("eltype"),
                t=mean(samples), tsd=std(samples)))
        end
    end
    isempty(rows) && error("no $workload $experiment $backend results in $path")
    return rows
end

function load_panel(workload, experiment, inputs)
    workload in ("krylov", "ode") || error("unknown workload: $workload")
    experiment in ("single", "weak") || error("unknown experiment: $experiment")
    series, rows = [], NamedTuple[]
    dagger_multi_missing = workload == "ode" && experiment == "weak" && !haskey(inputs, "dagger-multi")
    for b in BACKENDS
        experiment == "weak" && b.key == "cuda" && continue
        key = b.key == "cuda" ? "cuda" : b.key * (experiment == "single" ? "-single" : "-multi")
        subset = if haskey(inputs, key)
            read_source(inputs[key], workload, experiment, b.backend)
        elseif dagger_multi_missing && b.key == "dagger" && haskey(inputs, "dagger-single") && !isempty(rows)
            # Reuse only the measured one-GPU solve at the weak-scaling base N.
            base_n = first(rows).base_n
            single = read_source(inputs["dagger-single"], workload, "single", b.backend)
            baseline = filter(r -> r.n == base_n, single)
            length(baseline) == 1 || error("expected one single-GPU Dagger result at N=$base_n for the ODE multi-GPU baseline")
            [merge(only(baseline), (; base_n))]
        else
            continue
        end
        x(r) = experiment == "single" ? r.n : r.gpus
        sort!(subset; by=x)
        allunique(x.(subset)) || error("duplicate $key size/GPU point")
        append!(rows, subset)
        # The shared panel helpers use `gpus` as their x coordinate, also for N.
        agg = [(gpus=x(r), t=r.t, tsd=r.tsd) for r in subset]
        push!(series, (; b.label, b.color, b.marker, ls=:solid, agg))
    end
    isempty(rows) && return nothing
    if experiment == "weak" && haskey(inputs, "cuda")
        # CUDA runs on one GPU only; reuse its largest measured single-GPU case.
        b = first(BACKENDS)
        single = read_source(inputs["cuda"], workload, "single", b.backend)
        largest_n = maximum(r.n for r in single)
        base_n = first(rows).base_n
        largest_n == base_n || error("largest single-GPU CUDA size N=$largest_n does not match $workload weak-scaling base N=$base_n")
        baseline = filter(r -> r.n == largest_n, single)
        length(baseline) == 1 || error("expected one single-GPU CUDA result at N=$largest_n")
        r = merge(only(baseline), (; base_n))
        pushfirst!(rows, r)
        agg = [(gpus=r.gpus, t=r.t, tsd=r.tsd)]
        pushfirst!(series, (; b.label, b.color, b.marker, ls=:solid, agg))
    end
    length(unique((r.eltype, r.steps) for r in rows)) == 1 ||
        error("mixed precision or ODE step counts in $workload $experiment")
    if experiment == "weak"
        length(unique(r.base_n for r in rows)) == 1 || error("mixed weak-scaling base N")
        for g in unique(r.gpus for r in rows)
            length(unique(r.n for r in rows if r.gpus == g)) == 1 ||
                error("different problem sizes at $g GPUs")
        end
    end
    return (; series, rows, dagger_multi_missing)
end

function composability_panel(workload, experiment, panel; first_col, last_col, show_title=true, fix, st)
    (; series, rows) = panel
    title = show_title ? (experiment == "single" ? "Single GPU" : "Multi-GPU") : ""
    # Reuse the paper grid's broken-axis styling for the slow Dagger baseline.
    split_at = workload == "ode" ? 8000.0 / first(rows).steps : Inf
    split = workload == "ode" && experiment == "weak" && any(r -> r.t > split_at, rows) ? split_at : nothing
    # PIE.jl drawn last, over the other models (the legend keeps its own order).
    series = sort(series; by=s -> startswith(s.label, CUNUMERIC_NAME))
    p = panel_plot(series, "time"; title, split, split_pad=0.50,
        log_values=false, zero_log=experiment == "single",
        first_col, last_row=true, bottom_row=true, fix, st)
    xlabel = experiment == "single" ?
        (workload == "krylov" ? "Matrix dimension N" : "Grid dimension N (N × N)") : "GPUs"
    ylabel = workload == "ode" ? "Time/step (ms)" : "Time to Solve (ms)"
    plot!(p; xlabel, ylabel=first_col ? ylabel : "",
        right_margin=(last_col ? 2.5st.k : 0) * Plots.mm)
    # GR's bundled fonts on Windows do not include the grid's DejaVu TTF.
    Sys.iswindows() && plot!(p; titlefontfamily="Helvetica Bold")
    if experiment == "single"
        xs = sort!(unique(r.n for r in rows))
        # Power-of-two labels; measured N values stay at their actual positions.
        lo, hi = floor(Int, log2(first(xs))), ceil(Int, log2(last(xs)))
        stride = max(1, cld(hi - lo, 5))
        exponents = unique([collect(lo:stride:hi); hi])
        if length(exponents) > 2 && exponents[end] - exponents[end - 1] < stride
            deleteat!(exponents, length(exponents) - 1)
        end
        superscript(n) = join(('⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹')[d + 1]
            for d in reverse(digits(n)))
        plot!(p; xticks=(2.0 .^ exponents, ["2" * superscript(n) for n in exponents]),
            xlims=(2.0^lo / 1.15, 2.0^hi * 1.15))
    end
    hi = last(Plots.ylims(p))
    if experiment == "weak" && split === nothing
        # Explicit ticks keep zero visible even when all timings are far from zero.
        target = hi / 5
        magnitude = 10.0^floor(log10(target))
        step = first(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= target)
        plot!(p; ylims=(0, hi), yticks=collect(0:step:hi))
    end
    if panel.dagger_multi_missing
        lo_x, hi_x = Plots.xlims(p)
        # Dark ochre keeps the Dagger hue readable as small text on white.
        annotate!(p, exp(log(lo_x) + 0.90 * log(hi_x / lo_x)), 0.84hi,
            text("Dagger.jl: >1 GPU\nintractable", st.legend - 1, "#806400", :right, :top))
    end
    return p
end

function composability_sizing(columns)
    # Match plot_figures.jl: width_in applies to the entire row, not each panel.
    sizing = TOML.parsefile(joinpath(BENCH_ROOT, "configs", "plots", "figures.toml"))
    panel_w, panel_h = get(sizing, "panel_size", [400, 300])
    font_size = get(sizing, "font_size", 11)
    if haskey(sizing, "width_in")
        scaled_w = 144sizing["width_in"] / columns
        panel_h *= scaled_w / panel_w
        panel_w = scaled_w
        font_size *= 2
    end
    width = panel_w * columns
    st = grid_style(font_size; legend_scale=get(sizing, "legend_scale", 1.0))
    # Two levels of headings share the configured base size in this compact grid.
    st = merge(st, (; title=st.tick))
    return (; width, panel_h, st)
end

function composability_figure(workload, panels)
    (; width, panel_h, st) = composability_sizing(length(panels))
    series = unique(s -> s.label, [s for (_, panel) in panels for s in panel.series])
    legend = legend_rows(series, width, st)
    legend_h = legend_dims(st).row * length(legend)
    body_h = panel_h + legend_h
    title_h = 1.5st.title
    height = body_h + title_h
    title = workload == "krylov" ? "Krylov.jl CG" : "OrdinaryDiffEq.jl 2D Heat Diffusion"
    fix = gr_margin_fix(width, height, st)
    plots = [composability_panel(workload, mode, panel;
                 first_col=i == 1, last_col=i == length(panels), fix, st)
             for (i, (mode, panel)) in enumerate(panels)]
    body = plot(plots...; layout=grid(1, length(plots)))
    return plot(body, grid_legend(legend, width, st; center=true, errorbars=true,
        shift=length(legend) == 1 ? 0.10 : 0.0);
        layout=grid(2, 1; heights=[panel_h, legend_h] ./ body_h),
        size=(width, round(Int, height)), dpi=200, background_color=:white,
        plot_title=title, plot_titlefontsize=st.title, plot_titlevspan=title_h / height,
        plot_titlefontfamily=Sys.iswindows() ? "Helvetica Bold" : "DejaVuSans-Bold")
end

function combined_figure(workloads)
    (; width, panel_h, st) = composability_sizing(2)
    series = unique(s -> s.label, [s for (_, panels) in workloads for (_, panel) in panels for s in panel.series])
    legend = legend_rows(series, width, st)
    legend_h = legend_dims(st).row * length(legend)
    heading_h = 1.5st.title
    height = length(workloads) * (heading_h + panel_h) + legend_h
    fix = gr_margin_fix(width, height, st)
    sections, heights = [], Float64[]
    for (row, (workload, panels)) in enumerate(workloads)
        title = workload == "krylov" ? "Krylov.jl CG" : "OrdinaryDiffEq.jl 2D Heat Diffusion"
        heading = plot(; framestyle=:none, grid=false, ticks=false, legend=false,
            xlims=(0, 1), ylims=(0, 1), margin=0Plots.mm)
        family = Sys.iswindows() ? "Helvetica Bold" : "DejaVuSans-Bold"
        annotate!(heading, 0.5, 0.5, text(title, st.title, family, :black, :center))
        plots = [composability_panel(workload, mode, panel;
                     first_col=i == 1, last_col=i == 2, show_title=row == 1, fix, st)
                 for (i, (mode, panel)) in enumerate(panels)]
        push!(sections, heading, plot(plots...; layout=grid(1, 2)))
        append!(heights, [heading_h, panel_h])
    end
    push!(sections, grid_legend(legend, width, st; center=true, errorbars=true,
        shift=length(legend) == 1 ? 0.10 : 0.0))
    push!(heights, legend_h)
    return plot(sections...; layout=grid(length(sections), 1; heights=heights ./ height),
        size=(width, round(Int, height)), dpi=200, background_color=:white)
end

function combined_main(args)
    config = joinpath(BENCH_ROOT, "configs", "plots", "composability.toml")
    root = nothing
    out_dir, format = joinpath(BENCH_ROOT, "plots", "composability"), "pdf"
    for arg in args
        startswith(arg, "--") && occursin('=', arg) || error("expected --option=value, got $arg")
        key, value = split(arg[3:end], '='; limit=2)
        isempty(value) && error("empty --$key")
        if key == "config"
            config = value
        elseif key == "results-root"
            root = value
        elseif key == "out"
            out_dir = value
        elseif key == "format"
            format = value
        else
            error("unknown combined argument: $arg")
        end
    end
    format in ("pdf", "png", "svg") || error("--format must be pdf, png, or svg")
    raw = TOML.parsefile(config)
    root = something(root, get(raw, "results_root", "results/composability"))
    root = isabspath(root) ? root : joinpath(BENCH_ROOT, root)
    workloads = map(("krylov", "ode")) do workload
        sources = raw[workload]
        all(k -> k in INPUT_KEYS, keys(sources)) || error("unknown input key in [$workload]")
        inputs = Dict(k => isabspath(v) ? v : joinpath(root, v) for (k, v) in sources)
        panels = [(mode, load_panel(workload, mode, inputs)) for mode in ("single", "weak")]
        all(p -> last(p) !== nothing, panels) || error("combined figure requires single and multi-GPU $workload inputs")
        return (workload, panels)
    end
    figure = combined_figure(workloads)
    mkpath(out_dir)
    out = joinpath(out_dir, "composability.$format")
    save_plot(figure, out)
    println("wrote $out")
    return nothing
end

function usage(io=stdout)
    println(io, """
    Usage: julia --project=. plotter/plot_composability.jl {krylov|ode} [options]
           julia --project=. plotter/plot_composability.jl combined [--config=PATH] [--results-root=DIR] [--out=DIR] [--format=pdf|png|svg]

      --cuda=PATH               Single-GPU CUDA results.csv (also supplies the multi-GPU baseline)
      --cunumeric-single=PATH   Single-GPU cuNumeric results.csv
      --cunumeric-multi=PATH    Weak-scaling cuNumeric results.csv
      --dagger-single=PATH      Single-GPU Dagger results.csv
      --dagger-multi=PATH       Weak-scaling Dagger results.csv
      --out=DIR                 Output directory (default: plots/composability)
      --format=pdf|png|svg      Output format (default: pdf)

    Paths may contain other backends; only the requested backend is selected.
    CUDA's largest single-GPU case is shown at one GPU in multi-GPU panels;
    its size must match the weak-scaling base N. ODE selects CUDA's CuArray rows.
    Combined mode reads configs/plots/composability.toml for the CSV paths and
    writes one 2×2 figure with workload rows, scaling columns, and a shared legend.
    One titled figure per workload: Single GPU on the left, Multi-GPU on the right.
    Omitted scaling modes are skipped. Single uses log(1 + time); all y axes start at zero.
    ODE samples are divided by their CSV step count to report average ms per step.
    Multi-GPU ODE splits the time axis at 8000 / steps ms per step when needed.
    Means and sample standard deviations are recomputed from samples_ms; Krylov uses stock CG.
    """)
end

function main(args=ARGS)
    if isempty(args) || "--help" in args || "-h" in args
        usage()
        return nothing
    end
    workload = first(args)
    workload == "combined" && return combined_main(args[2:end])
    workload in ("krylov", "ode") || error("expected krylov or ode, got $workload")
    inputs = Dict{String,String}()
    out_dir, format = joinpath(BENCH_ROOT, "plots", "composability"), "pdf"
    for arg in args[2:end]
        startswith(arg, "--") && occursin('=', arg) || error("expected --option=value, got $arg")
        key, value = split(arg[3:end], '='; limit=2)
        isempty(value) && error("empty --$key")
        if key in INPUT_KEYS
            haskey(inputs, key) && error("duplicate --$key")
            inputs[key] = value
        elseif key == "out"
            out_dir = value
        elseif key == "format"
            format = value
        else
            error("unknown argument: $arg")
        end
    end
    isempty(inputs) && error("supply at least one results.csv path; see --help")
    format in ("pdf", "png", "svg") || error("--format must be pdf, png, or svg")
    # Validate both modes before writing the combined figure.
    panels = [(mode, load_panel(workload, mode, inputs)) for mode in ("single", "weak")]
    filter!(p -> last(p) !== nothing, panels)
    figure = composability_figure(workload, panels)
    mkpath(out_dir)
    out = joinpath(out_dir, "$(workload).$(format)")
    save_plot(figure, out)
    println("wrote $out")
    for (mode, panel) in panels
        println("  $mode: $(first(panel.rows).eltype)" *
            (workload == "ode" ? ", $(first(panel.rows).steps) steps" : "") *
            (mode == "weak" ? ", N(1)=$(first(panel.rows).base_n)" : ""))
    end
    return nothing
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    ComposabilityPlot.main()
end
