#!/usr/bin/env bash


sudo apt-get install libpmix2 libpmix-dev
# Install Open MPI v4.1.5
cd $HOME && \
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz && \
    tar -xvf openmpi-4.1.5.tar.gz && \
    cd openmpi-4.1.5 && \
    ./configure --prefix=$HOME/.local --enable-mpi-cxx  --with-pmix=internal --with-cuda=/usr/local/cuda && \
    make all -j && \
    make install

# Install CMAKE v3.30
cd $HOME && \
    wget https://github.com/Kitware/CMake/releases/download/v3.30.7/cmake-3.30.7-linux-x86_64.sh --no-check-certificate && \
    sh cmake-3.30.7-linux-x86_64.sh --skip-license --prefix=$HOME/.local

# Install Julia
cd $HOME
curl -fsSL https://install.julialang.org | bash -s -- --default-channel 1.10 --yes

# setup paths for installed software
export PATH=$HOME/.local/bin:$HOME/.juliaup/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Get our benchmarking stuff
git clone https://github.com/JuliaLegate/juliacon-benchmarking.git

cd $HOME/juliacon-benchmarking/models/cunumeric
# necessary to build cupynumeric from source
export LEGATE_DEVELOP_MODE=1
# use system MPI that we installed above
export JULIA_MPI_PATH="/home/ubuntu/.local/lib/"

threads=$(($(nproc) / 2))
export JULIA_NUM_THREADS=$threads
rm Project.toml # cunumeric and legate are unregistered. we will build a Project.toml from scratch

julia --project=. -e 'using Pkg; Pkg.add("MPIPreferences")'
julia --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary(library_names=["/home/ubuntu/.local/lib/libmpi.so", "/home/ubuntu/.local/lib/libmpi_cxx.so"], extra_paths=["/home/ubuntu/.local/lib/"])'
julia --project=. -e 'using Pkg; Pkg.add("CUDA")'
julia --project=. -e "using CUDA; CUDA.set_runtime_version!(local_toolkit=true)"

# Install Legate.jl and cuNumeric.jl
julia --project=. -e 'using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/Legate.jl", rev = "main")'
julia --project=. -e 'using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/cuNumeric.jl", rev = "cuda-jl-tasking")'
julia --project=. -e 'using Pkg; Pkg.build()'

# conda install for cupynumeric
mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh --no-check-certificate && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm ~/miniconda3/miniconda.sh && \
    source ~/miniconda3/bin/activate
# install cupynumeric
conda init bash && \
    source ~/.bashrc && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    CONDA_OVERRIDE_CUDA="12.4" conda create -n myenv -c conda-forge -c legate/label/rc cupynumeric=25.05.00.rc3 -y && \
    conda activate myenv

# Setup implicit global grid
cd $HOME/juliacon-benchmarking/models/diffeq
julia --project=. -e 'using Pkg; Pkg.add("MPIPreferences")'
julia --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary(library_names=["/home/ubuntu/.local/lib/libmpi.so", "/home/ubuntu/.local/lib/libmpi_cxx.so"], extra_paths=["/home/ubuntu/.local/lib/"])'
julia --project=. -e "using CUDA; CUDA.set_runtime_version!(local_toolkit=true)"
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate();'