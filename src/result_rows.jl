struct Row
    gpus::Int
    N::Int
    M::Int
    time_ms::Float64
    thr::Float64
end

function load_runs(path)
    rows = Row[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        f = split(line, ',')
        length(f) == 8 || error("Invalid result row in $path")
        push!(
            rows,
            Row(
                parse(Int, f[2]),
                parse(Int, f[3]),
                parse(Int, f[4]),
                parse(Float64, f[6]),
                parse(Float64, f[7]),
            ),
        )
    end
    isempty(rows) && return Vector{Row}[]
    runs = [Row[]]
    for (i, r) in enumerate(rows)
        i > 1 && r.gpus < rows[i - 1].gpus && push!(runs, Row[])
        push!(runs[end], r)
    end
    return runs
end

# One point per GPU count, or per problem size with `by=:N` (single-GPU sweeps).
function aggregate(rows; by=:gpus)
    groups = Dict{Int,Vector{Row}}()
    for r in rows
        push!(get!(groups, getfield(r, by), Row[]), r)
    end
    for (k, rs) in groups
        length(unique((r.gpus, r.N, r.M) for r in rs)) == 1 ||
            error("Cannot combine different dimensions at $by=$k; select one invocation")
    end
    sd(x) = length(x)>1 ? std(x) : 0.0
    return [
        (gpus=first(groups[k]).gpus, N=first(groups[k]).N, M=first(groups[k]).M,
            t=mean(getfield.(groups[k], :time_ms)), tsd=sd(getfield.(groups[k], :time_ms)),
            h=mean(getfield.(groups[k], :thr)), hsd=sd(getfield.(groups[k], :thr))) for
        k in sort(collect(keys(groups)))
    ]
end

# IGG is exempt: its 8-GPU run uses N=79198 (N-2 must split across its process grid).
function validate_series_sizes(series)
    sizes = Dict{Int,Tuple{Int,Int}}()
    for s in series, r in s.agg
        s.label == "ImplicitGlobalGrid.jl" && continue
        previous = get!(sizes, r.gpus, (r.N, r.M))
        previous == (r.N, r.M) ||
            error("Comparison series use different dimensions at $(r.gpus) GPUs")
    end
    return nothing
end
