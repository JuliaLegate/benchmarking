# Model names in every plot; loc-analysis reads these two lines too.
const CUNUMERIC_NAME = "cuNumeric.jl"
const CUPYNUMERIC_NAME = "cuPyNumeric"

# Config files may name a model by its default label, alone or leading a series
# label ("cuNumeric.jl (naive)"); map that to the configured one.
const NAME_ALIASES = Dict("cuNumeric.jl" => CUNUMERIC_NAME, "cuNumeric" => CUNUMERIC_NAME,
    "cuPyNumeric" => CUPYNUMERIC_NAME)
function display_name(label)
    m = match(r"^(\S+)(.*)$"s, label)
    m === nothing && return label
    return get(NAME_ALIASES, m[1], m[1]) * m[2]
end
