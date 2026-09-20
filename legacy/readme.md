Only works from base project dir
```bash
bash benchmarks/gemm/gemm.sh
```

to plot
```julia
julia --project=. benchmarks/gemm/plot_gemm.jl 
```

## Note
Due to cuNumeric being unregistered, you may need to copy the Manifest.toml from the ```$HOME``` directory of the container to the cunumeric folder in ```\models```