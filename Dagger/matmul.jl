using CUDA
using Dagger

function init_dagger(N, bs)
  @assert div(N, bs) == 0 "N must be multiple of block size"
  DA = rand(Blocks(bs, bs), Float32, N, N)
  DB = rand(Blocks(bs, bs), Float32, N, N)
  DC = zeros(Blocks(bs,bs), Float32, N, N)
  return DA, DB, DC
end

N = ARGS[1]
bs = ARGS[2]
times = []
for i in eachindex(devices())
  DA, DB, DC = init_dagger(N, bs)
  t = @belapsed seconds=1000 samples=N begin
    Dagger.with_options(;scope=Dagger.scope(worker=1, cuda_gpus=$i)) do
      mul!($DC, $DA, $DB)
    end
  end
  push!(times, t)
end

println(times)
