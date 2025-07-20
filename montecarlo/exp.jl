using CUDA

A = CUDA.rand(Float32, 10000)
y = exp.(A)