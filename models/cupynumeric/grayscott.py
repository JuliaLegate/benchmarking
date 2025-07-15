# python equivalent of gray-scott.jl to test the GC problem

import cupynumeric as np
from legate.timing import time
import csv
import sys 

def greyScottSys(u, v, dx, dt, c_u, c_v, f, k):
    # u,v are arrays 
    # dx,dt are space and time steps
    # c_u, c_v, f, k are constant paramaters
     #create new u array
    u_new = np.zeros_like(u)
    v_new = np.zeros_like(v)
    
    #calculate F_u and F_v functions
    F_u = (-u[1:-1,1:-1]*(v[1:-1,1:-1]**2)) + f*(1-u[1:-1,1:-1])
    F_v = (u[1:-1,1:-1]*(v[1:-1,1:-1]**2)) - (f+k)*v[1:-1,1:-1]
    
    # 2-D Laplacian of f using array slicing, excluding boundaries
    # For an N x N array f, f_lap is the N-1 x N-1 array in the "middle"
    u_lap = (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dx**2\
            + (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / dx**2
    v_lap = (v[2:,1:-1] - 2*v[1:-1,1:-1] + v[:-2,1:-1]) / dx**2\
            + (v[1:-1,2:] - 2*v[1:-1,1:-1] + v[1:-1,:-2]) / dx**2

    # Forward-Euler time step for all points except the boundaries
    u_new[1:-1,1:-1] = ((c_u * u_lap) + F_u)*dt + u[1:-1,1:-1]
    v_new[1:-1,1:-1] = ((c_v * v_lap) + F_v)*dt + v[1:-1,1:-1]

    # Apply periodic boundary conditions
    u_new[:,0] = u[:,-2]
    u_new[:,-1] = u[:,1]
    u_new[0,:] = u[-2,:]
    u_new[-1,:] = u[1,:]
    v_new[:,0] = v[:,-2]
    v_new[:,-1] = v[:,1]
    v_new[0,:] = v[-2,:]
    v_new[-1,:] = v[1,:]
    return u_new, v_new


def grayscott(N, M, n_steps):
    dx = 1
    dt = dx/5
    u = np.ones((N,M))
    v = np.zeros((N,M))
    u[:150,:150] = np.random.rand(150,150)
    v[:150,:150] = np.random.rand(150,150)

    c_u = 1
    c_v = 0.3
    f = 0.03
    k = 0.06

    for i in range(n_steps):
        greyScottSys(u, v, dx, dt, c_u, c_v, f, k)

def total_flops(N, T):
    return N * N * T # O(N^2 * T)     

gpus = int(sys.argv[1])
N = int(sys.argv[2])
M = int(sys.argv[3])
n_samples = int(sys.argv[4])


print(f"[cuPyNumeric] GrayScott benchmark on {N}x{M} matricies for {n_samples} iterations")


start_time = time()
grayscott(N, N, n_samples)
end_time = time()

total_time_ms = (end_time - start_time)/1000.0
mean_time_ms = total_time_ms / n_samples
gflops = total_flops(N, M) / (mean_time_ms * 1e6) # GFLOP is 1e9

print(f"[cuPyNumeric] Mean Run Time: {mean_time_ms} ms")
print(f"[cuPyNumeric] FLOPS: {gflops} GFLOPS")


with open("./grayscott/grayscott.csv", "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["cupynumeric", gpus, N, M, f"{mean_time_ms:.6f}", f"{gflops:.6f}"])