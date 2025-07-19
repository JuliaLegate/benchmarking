# python equivalent of gray-scott.jl to test the GC problem

import cupynumeric as np
from legate.timing import time
import csv
import sys 


def integrand(x):
    return np.mean(np.exp(np.square(x)))

def mc_integration(N, n_steps, n_warmup):

    x = (np.float32(5.0)*np.random.rand(N, dtype = np.float32)) - np.float32(10.0)

    for i in range(n_warmup):
        res = integrand(x)

    start_time = time()
    for i in range(n_steps):
        res = integrand(x)
    end_time = time()

    return end_time - start_time

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
gflops = total_flops(N, M) / (mean_time_ms * 1e6) # GFLOP is 1e9

print(f"[cuPyNumeric] Mean Run Time: {mean_time_ms} ms")
print(f"[cuPyNumeric] FLOPS: {gflops} GFLOPS")


with open("./montecarlo/mc.csv", "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["cuPyNumeric", gpus, N, M, f"{mean_time_ms:.6f}", f"{gflops:.6f}"])