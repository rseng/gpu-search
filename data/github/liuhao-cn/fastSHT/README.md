# https://github.com/liuhao-cn/fastSHT

```console
compile.sh:if [[ ${1} == *"GPU=off"* ]]; then
compile.sh:    cmake .. -DGPU=on
compile.sh:    gfortran -E -cpp -DGPU=True ../src/sht_data_module.f90 > src/sht_data_module.f90
compile.sh:    gfortran -E -cpp -DGPU=True ../src/sht_main.f90 > src/sht_main.f90
compile.sh:    LDFLAGS="-cpp -Mcuda -acc -Mcudalib=cublas -fPIC -lcufft -pgf90libs -mp -lpthread -lm -ldl -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread" f2py-f90wrap --fcompiler=nv -c -m _fastSHT f90wrap_*.f90 --f90flags="-cpp -acc -Mcud\
compile.sh:a -Mcudalib=cublas -pgf90libs -mp -lpthread -lm -ldl" *.a
configure.sh:p2=gpu
configure.sh:    --gpu-skip)
configure.sh:        p2=--gpu-skip
configure.sh:    --gpu-skip)
configure.sh:        p2=--gpu-skip
configure.sh:    --gpu-skip)
configure.sh:	    ./compile.sh -DGPU=off
configure.sh: 	    echo 'deb [trusted=yes] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
configure.sh:	    sed -i '1 i\export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/compilers/bin/:$PATH"' ~/.bashrc
configure.sh:	        sed -i '1 i\export LD_PRELOAD=:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.so:/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/REDIST/compilers/lib/libomp.so' ~/.bashrc
configure.sh:	    ./compile.sh -DGPU=on
README.md:fastSHT is a fast toolkit for doing spherical harmonic transforms on a large number of spherical maps. It converts massive SHT operations to a BLAS level 3 problem and uses the highly optimized matrix multiplication toolkit to accelerate the computation. GPU acceleration is supported and can be very effective. The core code is written in Fortran, but a Python wrapper is provided and recommended.
README.md:To ensure a precise result, fastSHT uses double precision floating numbers (FP64) by default, which prefers GPU hardware with high FP64 performance. Therefore, the current best choice is the NVIDIA A100 (till Aug-2022), which provides a full-speed FP64 computation with its tensor cores (same performance for double and single precisions).
README.md:Fortran compiler `ifort` is recommended for the CPU version, and `nvfortran` is required for the GPU version. The main dependencies and their versions are listed below:
README.md:`Nvidia HPC SDK (22.3 or 22.7)`
README.md:A quick installation can be done by the steps below, which is a full installation with both CPU and GPU (without docker).
README.md:The default behavior of `configure.sh` is a full installation of Intel ONE API and NVIDIA HPC SDK, and the following options are also supported:
README.md:# skip the installation of NVIDIA HPC SDK, and therefore disable the GPU support.
README.md:configure.sh --gpu-skip
README.md:The default behavior of `compile.sh` is to compile the code with GPU support, and the following option is supported:
README.md:# compile without the GPU support.
README.md:./compile.sh -DGPU=off
README.md:sudo docker build -t fastsht:gpu .
README.md:sudo docker pull rectaflex/intel_nvidia_sdk
README.md:sudo docker image tag rectaflex/intel_nvidia_sdk fastsht:gpu
README.md:### Step-2: install the Nvidia container runtime
README.md:When the docker image is prepared, one needs to install the Nvidia container runtime by
README.md:curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey |   sudo apt-key add -
README.md:curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list |   sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
README.md:sudo apt-get install nvidia-container-runtime
README.md:sudo docker run -it -v /home:/home --gpus all fastsht:gpu
README.md:which makes all GPUs available in docker and also makes `/home` available, so if one clones the fastSHT repository to `/home` on the host machine, it will be available in docker.
README.md:If no GPU is going to be employed, one can stop here and jump to compilation.
README.md:#### step-4) Continue with GPU support and install the nvidia hpc sdk:
README.md:echo 'deb [trusted=yes] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
README.md:sed -i '1 i\export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/compilers/bin/:$PATH"' ~/.bashrc
README.md:The GPU version will be compiled by default. Use the following command to compile the CPU version:
README.md:./compile.sh -DGPU=off
README.md:or the following command to compile the GPU version:
README.md:### The case with GPU
README.md:/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/REDIST/compilers/lib/libomp.so
README.md:`/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/REDIST/compilers/lib/libomp.so`
README.md:where 22.3 or 22.7 depends on the nvidia hpc sdk version. Because the GPU version uses a different omp library.
CMakeLists.txt:option(GPU "GPU" OFF)
CMakeLists.txt:if(INTEL AND NOT GPU)
CMakeLists.txt:if(GPU)
CMakeLists.txt:message("Will compile with GPU (supported by openacc)")
CMakeLists.txt:set(CMAKE_Fortran_FLAGS "-fPIC -O3 -fopenmp -Mcuda -Mcudalib=cublas -lcufft -acc -ta=tesla -fast -cpp -DGPU=TRUE")
scripts/benchmark.py:from numba import cuda
scripts/benchmark.py:    #alms = numba.cuda.pinned_array((nsim, lmax+1, lmax+1), dtype=np.double, strides=None, order='F')
scripts/nproc.py:from numba import cuda
docker/Dockerfile:# install nvidia sdk
docker/Dockerfile:    && wget https://developer.download.nvidia.com/hpc-sdk/22.7/nvhpc_2022_227_Linux_x86_64_cuda_11.7.tar.gz \
docker/Dockerfile:    && tar xpzf nvhpc_2022_227_Linux_x86_64_cuda_11.7.tar.gz \
docker/Dockerfile:    && rm nvhpc_2022_227_Linux_x86_64_cuda_11.7.tar.gz \
docker/Dockerfile:    && nvhpc_2022_227_Linux_x86_64_cuda_11.7/install \
docker/Dockerfile:    && rm -rf nvhpc_2022_227_Linux_x86_64_cuda_11.7/ \
docker/Dockerfile:    && sed -i '1 i\export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/compilers/bin/:$PATH"' ~/.bashrc \
docker/Dockerfile:    && sed -i '1 i\export LD_PRELOAD=:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.so:/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/REDIST/compilers/lib/libomp.so' ~/.bashrc \
SHT.py:from numba import cuda
SHT.py:                alms = numba.cuda.pinned_array((self.nsim, self.lmax+1, self.lmax+1), dtype=np.double, strides=None, order='F')
SHT.py:                maps = numba.cuda.pinned_array((self.npix, self.nsim), dtype=np.double, strides=None, order='F')
SHT.py:                almEs = numba.cuda.pinned_array((self.nsim, self.lmax+1, self.lmax+1), dtype=np.double, strides=None, order='F')
SHT.py:                almBs = numba.cuda.pinned_array((self.nsim, self.lmax+1, self.lmax+1), dtype=np.double, strides=None, order='F')
SHT.py:                almEs = numba.cuda.pinned_array((self.nsim, self.lmax+1, self.lmax+1), dtype=np.double, strides=None, order='F')
SHT.py:                almBs = numba.cuda.pinned_array((self.nsim, self.lmax+1, self.lmax+1), dtype=np.double, strides=None, order='F')
SHT.py:            Q = numba.cuda.pinned_array((self.npix, self.nsim), dtype=np.double, strides=None, order='F')
SHT.py:            U = numba.cuda.pinned_array((self.npix, self.nsim), dtype=np.double, strides=None, order='F')
SHT.py:    # get alms from GPU (only use it when gpu is enabled)
SHT.py:                alms = numba.cuda.pinned_array((self.nsim, self.lmax+1, self.lmax+1), dtype=np.double, strides=None, order='F')
src/sht_data_init.f90:#ifdef GPU
src/sht_data_init.f90:#ifdef GPU
src/sht_data_init.f90:#ifdef GPU
src/sht_data_init.f90:#ifdef GPU
src/sht_data_init.f90:#ifdef GPU
src/sht_data_init.f90:#ifdef GPU
src/sht_data_init.f90:#ifdef GPU
src/sht_data_init.f90:#ifdef GPU
src/sht_data_init.f90:        stat = cudaStreamCreate(cu_streams(m))
src/sht_data_init.f90:#ifdef GPU
src/sht_data_init.f90:  use cudafor
src/sht_data_init.f90:#ifdef GPU
src/sht_data_init.f90:#ifdef GPU
src/sht_data_init.f90:#ifdef GPU
src/sht_data_init.f90:#ifdef GPU
src/fft_mapping.f90:#ifdef GPU
src/fft_mapping.f90:  use cudafor
src/fft_mapping.f90:#ifdef GPU
src/fft_mapping.f90:#ifdef GPU
src/fft_mapping.f90:    i = cudaDeviceSynchronize()
src/fft_mapping.f90:#ifndef GPU
src/fft_mapping.f90:#ifndef GPU
src/fft_mapping.f90:#ifndef GPU
src/fft_mapping.f90:#ifndef GPU
src/fft_mapping.f90:#ifdef GPU
src/fft_mapping.f90:    use cudafor
src/fft_mapping.f90:#ifdef GPU
src/fft_mapping.f90:#ifdef GPU
src/fft_mapping.f90:    stat = cudaDeviceSynchronize()
src/fft_mapping.f90:#ifndef GPU
src/fft_mapping.f90:#ifndef GPU
src/fft_mapping.f90:#ifdef GPU
src/fft_mapping.f90:    use cudafor
src/fft_mapping.f90:#ifdef GPU
src/fft_mapping.f90:#ifdef GPU
src/fft_mapping.f90:    stat = cudaDeviceSynchronize()
src/fft_mapping.f90:#ifndef GPU
src/fft_mapping.f90:#ifndef GPU
src/sht_core.f90:#ifdef GPU
src/sht_core.f90:    use cudafor
src/sht_core.f90:    integer(cuda_stream_kind):: stream
src/sht_core.f90:#ifdef GPU
src/sht_core.f90:    use cudafor
src/sht_core.f90:    integer(cuda_stream_kind):: stream
src/sht_core.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:  use cudafor
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:    use cudafor
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:    use cudafor
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:  use cudafor
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:         stat = cudaDeviceSynchronize()
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:  use cudafor
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:  use cudafor
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:  use cudafor
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:! IMPORTANT: to save memory for the GPU code,
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:    use cudafor
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:    use cudafor
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:    use cudafor
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:    use cudafor
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:    use cudafor
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:    use cudafor
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_main.f90:#ifdef GPU
src/sht_data_module.f90:#ifdef GPU
src/sht_data_module.f90:  use cudafor
src/sht_data_module.f90:#ifdef GPU
src/sht_data_module.f90:    integer(cuda_stream_kind), dimension(:), allocatable :: cu_streams
src/compute_fft.f90:#ifdef GPU
src/compute_fft.f90:    use cudafor
src/compute_fft.f90:        !stat = cudaStreamCreate(cu_streams(i))
src/compute_fft.f90:        stat = cudaMemcpyAsync(d_fft(1:nsim,h_i1_arr(i)), fft(1:nsim,h_i1_arr(i)), nsim*nfft, cu_streams(i))
src/compute_fft.f90:        stat = cudaMemcpyAsync(d_fft(1:nsim,h_i1_arr(j)), fft(1:nsim,h_i1_arr(j)), nsim*nfft, cu_streams(i))
src/compute_fft.f90:       !stat = cudaStreamCreate(cu_streams(i))
src/compute_fft.f90:       stat = cudaMemcpyAsync(d_fft(1:nsim,h_i1_arr(i)), fft(1:nsim,h_i1_arr(i)), nsim*nfft, cu_streams(i))
src/compute_fft.f90:#ifdef GPU
src/compute_fft.f90:    stat = cudaDeviceSynchronize()
src/compute_fft.f90:#ifdef GPU
src/compute_fft.f90:  use cudafor
src/compute_fft.f90:#ifdef GPU
src/compute_fft.f90:       stat = cudaMemcpyAsync(fft(1:nsim,h_i1_arr(i)), d_fft(1:nsim,h_i1_arr(i)), nsim*nfft, cu_streams(i))
src/compute_fft.f90:       stat = cudaMemcpyAsync(fft(1:nsim,h_i1_arr(j)), d_fft(1:nsim,h_i1_arr(j)), nsim*nfft, cu_streams(i))
src/compute_fft.f90:       stat = cudaMemcpyAsync(fft(1:nsim,h_i1_arr(i)), d_fft(1:nsim,h_i1_arr(i)), nsim*4*nside, cu_streams(i))
src/compute_fft.f90:             stat = cudaStreamQuery(cu_streams(i))
src/compute_fft.f90:#ifdef GPU
src/compute_fft.f90:#ifdef GPU
src/compute_fft.f90:#ifdef GPU

```
