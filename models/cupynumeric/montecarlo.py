# python equivalent of gray-scott.jl to test the GC problem

import cupynumeric as np
from legate.timing import time
import csv
import sys 
import gc


def do_work(x, N):
    # time in same scope to avoid GC overhead
    gc.disable()
    start_time = time()
    res = (np.float32(10.0) / N) * np.sum(np.exp(-np.square(x)))
    end_time = time()
    gc.enable()
    return end_time - start_time

def mc_integration(N, n_steps, n_warmup):

    x = (np.float32(10.0)*np.random.rand(N)) - np.float32(5.0)

    for i in range(n_warmup):
        _ = do_work(x, N)

    times = []
    for i in range(n_steps):
        times.append(do_work(x, N))

    return np.sum(times)

def total_flops(N):
    return N    

gpus = int(sys.argv[1])
N = int(sys.argv[2])
n_samples = int(sys.argv[3])
warmup=2

print(f"[cuPyNumeric] Monte-Carlo Integration benchmark on {N} elements for {n_samples} iterations")

elapsed_time = mc_integration(N, n_samples, warmup)

total_time_ms = elapsed_time/1000.0
mean_time_ms = total_time_ms / n_samples
gflops = total_flops(N) / (mean_time_ms * 1e6) # GFLOP is 1e9

print(f"[cuPyNumeric] Mean Run Time: {mean_time_ms} ms")
print(f"[cuPyNumeric] FLOPS: {gflops} GFLOPS")


with open("./benchmarks/montecarlo/montecarlo.csv", "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["cuPyNumeric", gpus, N, 1, f"{mean_time_ms:.6f}", f"{gflops:.6f}"])