using TOML

# tunes/*.csv (written by tune_dagger.sh) as (benchmark, gpus, class) => blocks_per_gpu => ms,
# whatever N/M was tuned. When a key was tuned at several sizes, the most recently tuned
# size wins; within a size, a split's latest non-failing row wins.
function read_dagger_tunes(dir)
    row = r"^([^,\n]+),([^,]+),[^,]+,(\d+),(\d+),(\d+),(\"(?:[^\"]|\"\")*\"|[^,]*),(\d+),\d+,\d+,([\d.]+),(?:pass|skipped)$"m
    # key => (N, M) => (latest timestamp, split => ms)
    tunes = Dict{Tuple{String,Int,String},Dict{Tuple{Int,Int},Tuple{String,Dict{Int,Float64}}}}()
    isdir(dir) || return Dict{keytype(tunes),Dict{Int,Float64}}()
    for file in filter(endswith(".csv"), readdir(dir; join=true)), m in eachmatch(row, read(file, String))
        stamp, name, N, M, gpus, kwargs, split, ms = m.captures
        kwargs = startswith(kwargs, '"') ? replace(kwargs[2:end-1], "\"\"" => "\"") : kwargs
        class = string(get(TOML.parse(kwargs), "class", ""))
        sizes = get!(tunes, (name, parse(Int, gpus), class), Dict())
        latest, splits = get(sizes, (parse(Int, N), parse(Int, M)), ("", Dict{Int,Float64}()))
        splits[parse(Int, split)] = parse(Float64, ms)
        sizes[(parse(Int, N), parse(Int, M))] = (max(latest, stamp), splits)
    end
    return Dict(key => last(argmax(first, values(sizes))) for (key, sizes) in tunes)
end
