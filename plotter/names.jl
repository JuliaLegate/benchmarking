# Model names in every plot; loc-analysis reads these two lines too.
const CUNUMERIC_NAME = "PIE.jl"
const CUPYNUMERIC_NAME = "MosaicPIE"

# Config files may name a model by its default label; map that to the configured one.
const NAME_ALIASES = Dict("cuNumeric.jl" => CUNUMERIC_NAME, "cuNumeric" => CUNUMERIC_NAME,
    "cuPyNumeric" => CUPYNUMERIC_NAME)
display_name(label) = get(NAME_ALIASES, label, label)
