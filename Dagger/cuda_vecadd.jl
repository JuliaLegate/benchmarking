using Dagger, CUDA
using KernelAbstractions

## note this doesn't work. todo: fix this

# Define a kernel function
@kernel function vector_add!(c, a, b)
    i = @index(Global, Linear)
    c[i] = a[i] + b[i]
end

# Run on GPU
# Note: GPU arrays must be marked @mutable or use Datadeps to ensure mutability
gpu_scope = Dagger.scope(cuda_gpu=:)
a = Dagger.@mutable CUDA.rand(1000)
b = Dagger.@mutable CUDA.rand(1000)
c = Dagger.@mutable CUDA.zeros(1000)
result = Dagger.with_options(;scope=gpu_scope) do
    fetch(Dagger.@spawn Dagger.Kernel(vector_add!)(c, a, b; ndrange=length(c)))
    # Synchronize the GPU
    Dagger.gpu_synchronize(:CUDA)
end
