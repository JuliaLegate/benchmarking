# 2 x N^2 x M / P --> constant
# eh we will restrict to NxN
# N^3 / P --> constant

function calc(baseline_N, label)
    gpu_counts = [1, 2, 4, 8]

    println("# $label test")
    println("declare -a $label=(")
    for P in gpu_counts
        N = round(Int, baseline_N * P^(1/3))
        println("  \"$N $N\"")
    end
    println(")")
    println()
end

test  = 1000  # N for 1 GPU
small = 20000  
large = 45000

calc(large, "large")
calc(small, "small")
calc(test, "test")