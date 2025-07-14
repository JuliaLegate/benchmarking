#!/usr/bin/env bash

# Install Open MPI v4.1.5
cd $HOME

wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
tar -xvf openmpi-4.1.5.tar.gz
cd openmpi-4.1.5
./configure --prefix=$HOME/.local --with-cuda=/usr/local/cuda
make all -j
make install

export PATH=$HOME/.local/bin:$HOME/.juliaup/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib

cd $HOME
curl -fsSL https://install.julialang.org | bash -s -- --default-channel 1.10 --yes

git clone https://github.com/JuliaLegate/juliacon-benchmarking.git
cd juliacon-benchmarking/models/diffeq

julia --project=. -e 'using Pkg; Pkg.add("MPIPreferences")'
julia --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary(library_names="/home/ubuntu/.local/lib/libmpi.so", extra_paths=["/home/ubuntu/.local/lib/"])'
julia --project=. -e "using CUDA; CUDA.set_runtime_version!(local_toolkit=true)"
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate();'