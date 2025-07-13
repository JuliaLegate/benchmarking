# N^2 / P --> constant

function calc(baseline_N, label)
    gpu_counts = [1, 2, 4, 8]

    println("# $label test")
    println("declare -a $label=(")
    for P in gpu_counts
        N = round(Int, baseline_N * P^(1/2))
        println("  \"$N $(2 * N)\"")
    end
    println(")")
    println()
end

small = 10000  # N for 1 GPU
large = 80000

calc(large, "large")
calc(small, "small")