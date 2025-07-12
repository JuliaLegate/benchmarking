using CUDA
using Dagger
using BenchmarkTools

function init_dagger(N, M, bs)
  DA = rand(Blocks(bs, bs), Float32, N, M)
  DB = rand(Blocks(bs, bs), Float32, N, M)
  DC = zeros(Blocks(bs,bs), Float32, N, N)
  return DA, DB, DC
end

N = ARGS[1]
M = ARGS[2]
bs = ARGS[3]
times = []
for i in eachindex(devices())
  DA, DB, DC = init_dagger(N, M, bs)
  t = @belapsed seconds=1000 samples=N begin
    Dagger.with_options(;scope=Dagger.scope(worker=1, cuda_gpus=$i)) do
      # mul!($DC, $DA, $DB)
      mul!(DC, DA, DB)
    end
  end
  push!(times, t)
end

println(times)
