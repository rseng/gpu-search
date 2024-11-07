# https://github.com/plumed/plumed2

```console
plugins/pycv/PythonFunction.cpp:GPU) can be performed via Google's [JAX
plugins/pycv/PythonCVInterface.cpp:GPU) can be performed via Google's [JAX
plugins/pycv/PythonCVInterface.cpp:GPU) can be performed via Google's [JAX
plugins/pycv/README.md: - example if you have a cuda12 compatible device (a wheel for cuda will be installed alongside jax):
plugins/pycv/README.md:`pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
plugins/pycv/README.md: - example if you have a cuda12 compatible device, and **cuda already installed on your system**:
plugins/pycv/README.md:`pip install "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
plugins/pycv/README.md:- JAX's GPU/TPU offloading are not 100% testes.
plugins/cudaCoord/Coordination.cuh:   cudaOnPlumed is free software: you can redistribute it and/or modify
plugins/cudaCoord/Coordination.cuh:   cudaOnPlumed is distributed in the hope that it will be useful,
plugins/cudaCoord/Coordination.cuh:#include "cuda_runtime.h"
plugins/cudaCoord/Coordination.cuh:namespace GPU {
plugins/cudaCoord/Coordination.cuh:__device__ __forceinline__ calculateFloat pcuda_fastpow (calculateFloat base,
plugins/cudaCoord/Coordination.cuh:template <typename calculateFloat> __device__ calculateFloat pcuda_eps() {
plugins/cudaCoord/Coordination.cuh:template <> constexpr __device__ float pcuda_eps<float>() {
plugins/cudaCoord/Coordination.cuh:template <> constexpr __device__ double pcuda_eps<double>() {
plugins/cudaCoord/Coordination.cuh:pcuda_Rational (const calculateFloat rdist,
plugins/cudaCoord/Coordination.cuh:    calculateFloat rNdist = pcuda_fastpow (rdist, NN - 1);
plugins/cudaCoord/Coordination.cuh:    if (rdist > (1. - pcuda_eps<calculateFloat>()) &&
plugins/cudaCoord/Coordination.cuh:        rdist < (1 + pcuda_eps<calculateFloat>())) {
plugins/cudaCoord/Coordination.cuh:      calculateFloat rNdist = pcuda_fastpow (rdist, NN - 1);
plugins/cudaCoord/Coordination.cuh:      calculateFloat rMdist = pcuda_fastpow (rdist, MM - 1);
plugins/cudaCoord/Coordination.cuh:__global__ void getpcuda_Rational (const calculateFloat *rdists,
plugins/cudaCoord/Coordination.cuh:    res[i] = pcuda_Rational (rdists[i], NN, MM, dfunc[i]);
plugins/cudaCoord/Coordination.cuh:    result = pcuda_Rational (
plugins/cudaCoord/Coordination.cuh:} // namespace GPU
plugins/cudaCoord/nvcc-mklib.sh:if [[ ${SILENT_CUDA_COMPILATION} ]]; then
plugins/cudaCoord/Readme.md:# Coordination in Cuda
plugins/cudaCoord/Readme.md:`CUDACOORDINATION` and `CUDACOORDINATIONFLOAT` depend on [CCCL](https://github.com/NVIDIA/cccl) which is automatically fetched by the cuda compiler (if you use nvcc, you have access to the CCCL headers).
plugins/cudaCoord/Readme.md:The files `cudaHelpers.cuh` and `cudaHelpers.cu` contains a few support functions for helping in interfacing `PLMD::Vector` and `PLMD::Tensor` with Cuda's thrust,
plugins/cudaCoord/Readme.md:along with the reduction functions baked with Cuda's cub building blocks and their drivers.
plugins/cudaCoord/Readme.md:`CUDACOORDINATION` and `CUDACOORDINATIONFLOAT` work more or less as the standard `COORDINATION`, except from:
plugins/cudaCoord/Readme.md: - The GPU device needs to be explicitly selected
plugins/cudaCoord/cudaHelpers.cuh:   cudaOnPlumed is free software: you can redistribute it and/or modify
plugins/cudaCoord/cudaHelpers.cuh:   cudaOnPlumed is distributed in the hope that it will be useful,
plugins/cudaCoord/cudaHelpers.cuh:#ifndef __PLUMED_cuda_helpers_cuh
plugins/cudaCoord/cudaHelpers.cuh:#define __PLUMED_cuda_helpers_cuh
plugins/cudaCoord/cudaHelpers.cuh:namespace CUDAHELPERS {
plugins/cudaCoord/cudaHelpers.cuh:/// @brief a interface to help in the data I/O to the GPU
plugins/cudaCoord/cudaHelpers.cuh:/// gpu to a PLMD container
plugins/cudaCoord/cudaHelpers.cuh:inline void plmdDataFromGPU (thrust::device_vector<double> &dvmem,
plugins/cudaCoord/cudaHelpers.cuh:  cudaMemcpy (data.ptr,
plugins/cudaCoord/cudaHelpers.cuh:              cudaMemcpyDeviceToHost);
plugins/cudaCoord/cudaHelpers.cuh:/// data from the gpu to a PLMD container
plugins/cudaCoord/cudaHelpers.cuh:inline void plmdDataFromGPU (thrust::device_vector<double> &dvmem,
plugins/cudaCoord/cudaHelpers.cuh:                             cudaStream_t stream) {
plugins/cudaCoord/cudaHelpers.cuh:  cudaMemcpyAsync (data.ptr,
plugins/cudaCoord/cudaHelpers.cuh:                   cudaMemcpyDeviceToHost,
plugins/cudaCoord/cudaHelpers.cuh:/// data from a PLMD container to the gpu
plugins/cudaCoord/cudaHelpers.cuh:inline void plmdDataToGPU (thrust::device_vector<double> &dvmem,
plugins/cudaCoord/cudaHelpers.cuh:  cudaMemcpy (thrust::raw_pointer_cast (dvmem.data()),
plugins/cudaCoord/cudaHelpers.cuh:              cudaMemcpyHostToDevice);
plugins/cudaCoord/cudaHelpers.cuh:/// data from a PLMD container to the gpu
plugins/cudaCoord/cudaHelpers.cuh:inline void plmdDataToGPU (thrust::device_vector<double> &dvmem,
plugins/cudaCoord/cudaHelpers.cuh:                           cudaStream_t stream) {
plugins/cudaCoord/cudaHelpers.cuh:  cudaMemcpyAsync (thrust::raw_pointer_cast (dvmem.data()),
plugins/cudaCoord/cudaHelpers.cuh:                   cudaMemcpyHostToDevice,
plugins/cudaCoord/cudaHelpers.cuh:/// gpu to a PLMD container
plugins/cudaCoord/cudaHelpers.cuh:/// @param dvmem the cuda interface to the data on the device
plugins/cudaCoord/cudaHelpers.cuh:inline void plmdDataFromGPU (thrust::device_vector<float> &dvmem,
plugins/cudaCoord/cudaHelpers.cuh:                             cudaStream_t = 0) {
plugins/cudaCoord/cudaHelpers.cuh:  cudaMemcpy (tempMemory.data(),
plugins/cudaCoord/cudaHelpers.cuh:              cudaMemcpyDeviceToHost);
plugins/cudaCoord/cudaHelpers.cuh:/// data from a PLMD container to the gpu
plugins/cudaCoord/cudaHelpers.cuh:inline void plmdDataToGPU (thrust::device_vector<float> &dvmem,
plugins/cudaCoord/cudaHelpers.cuh:  cudaMemcpy (thrust::raw_pointer_cast (dvmem.data()),
plugins/cudaCoord/cudaHelpers.cuh:              cudaMemcpyHostToDevice);
plugins/cudaCoord/cudaHelpers.cuh:/// data from a PLMD container to the gpu
plugins/cudaCoord/cudaHelpers.cuh:inline void plmdDataToGPU (thrust::device_vector<float> &dvmem,
plugins/cudaCoord/cudaHelpers.cuh:                           cudaStream_t stream) {
plugins/cudaCoord/cudaHelpers.cuh:  cudaMemcpyAsync (thrust::raw_pointer_cast (dvmem.data()),
plugins/cudaCoord/cudaHelpers.cuh:                   cudaMemcpyHostToDevice,
plugins/cudaCoord/cudaHelpers.cuh:/// @brief copies data to the GPU, using thrust::device_vector as interface
plugins/cudaCoord/cudaHelpers.cuh:inline void plmdDataToGPU (thrust::device_vector<T> &dvmem, Y &data) {
plugins/cudaCoord/cudaHelpers.cuh:  plmdDataToGPU (dvmem, DataInterface (data));
plugins/cudaCoord/cudaHelpers.cuh:/// @brief async version of plmdDataToGPU
plugins/cudaCoord/cudaHelpers.cuh:plmdDataToGPU (thrust::device_vector<T> &dvmem, Y &data, cudaStream_t stream) {
plugins/cudaCoord/cudaHelpers.cuh:  plmdDataToGPU (dvmem, DataInterface (data), stream);
plugins/cudaCoord/cudaHelpers.cuh:/// @brief copies data from the GPU, using thrust::device_vector as interface
plugins/cudaCoord/cudaHelpers.cuh:inline void plmdDataFromGPU (thrust::device_vector<T> &dvmem, Y &data) {
plugins/cudaCoord/cudaHelpers.cuh:  plmdDataFromGPU (dvmem, DataInterface (data));
plugins/cudaCoord/cudaHelpers.cuh:/// @brief async version of plmdDataFromGPU
plugins/cudaCoord/cudaHelpers.cuh:inline void plmdDataFromGPU (thrust::device_vector<T> &dvmem,
plugins/cudaCoord/cudaHelpers.cuh:                             cudaStream_t stream) {
plugins/cudaCoord/cudaHelpers.cuh:  plmdDataFromGPU (dvmem, DataInterface (data), stream);
plugins/cudaCoord/cudaHelpers.cuh:                      cudaStream_t stream) {
plugins/cudaCoord/cudaHelpers.cuh:                    cudaStream_t stream) {
plugins/cudaCoord/cudaHelpers.cuh:                      cudaStream_t stream) {
plugins/cudaCoord/cudaHelpers.cuh:                    cudaStream_t stream) {
plugins/cudaCoord/cudaHelpers.cuh:} // namespace CUDAHELPERS
plugins/cudaCoord/cudaHelpers.cuh:#endif //__PLUMED_cuda_helpers_cuh
plugins/cudaCoord/nvcc-MakeFile.sh:if [[ ${SILENT_CUDA_COMPILATION} ]]; then
plugins/cudaCoord/nvcc-MakeFile.sh:#tested with nvcc with :"Build cuda_11.7.r11.7/compiler.31442593_0"
plugins/cudaCoord/nvcc-MakeFile.sh:#and tested with nvcc with :"Build cuda 11.8
plugins/cudaCoord/Makefile:#tested with nvcc with :"Build cuda_11.7.r11.7/compiler.31442593_0"
plugins/cudaCoord/Makefile:OBJS = Coordination.o cudaHelpers.o
plugins/cudaCoord/Makefile:all: CudaCoordination.$(SOEXT)
plugins/cudaCoord/Makefile:CudaCoordination.$(SOEXT): $(OBJS)
plugins/cudaCoord/Makefile:	@rm -fv $(OBJS) CudaCoordination.$(SOEXT)
plugins/cudaCoord/cudaHelpers.cu:   cudaOnPlumed is free software: you can redistribute it and/or modify
plugins/cudaCoord/cudaHelpers.cu:   cudaOnPlumed is distributed in the hope that it will be useful,
plugins/cudaCoord/cudaHelpers.cu:#include "cudaHelpers.cuh"
plugins/cudaCoord/cudaHelpers.cu:namespace CUDAHELPERS {
plugins/cudaCoord/cudaHelpers.cu:  const size_t nnToGPU =
plugins/cudaCoord/cudaHelpers.cu:  // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
plugins/cudaCoord/cudaHelpers.cu:  const size_t expectedTotalThreads = ceil (nnToGPU / log2 (runningThreads));
plugins/cudaCoord/cudaHelpers.cu:} // namespace CUDAHELPERS
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/plumed.dat:gpu:    CUDACOORDINATIONFLOAT PAIR GROUPA=1-107 GROUPB=2-108 R_0=1 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/plumed.dat:gpu512: CUDACOORDINATIONFLOAT PAIR GROUPA=1-107 GROUPB=2-108 R_0=1 THREADS=512 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/plumed.dat:gpu256: CUDACOORDINATIONFLOAT PAIR GROUPA=1-107 GROUPB=2-108 R_0=1 THREADS=256 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/plumed.dat:gpu128: CUDACOORDINATIONFLOAT PAIR GROUPA=1-107 GROUPB=2-108 R_0=1 THREADS=128 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/plumed.dat:gpu64:  CUDACOORDINATIONFLOAT PAIR GROUPA=1-107 GROUPB=2-108 R_0=1 THREADS=64 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-float-nopbc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.5f
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/plumed.dat:gpu:    CUDACOORDINATIONFLOAT PAIR GROUPA=1-107 GROUPB=2-108 R_0=0.4
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/plumed.dat:gpu512: CUDACOORDINATIONFLOAT PAIR GROUPA=1-107 GROUPB=2-108 R_0=0.4 THREADS=512
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/plumed.dat:gpu256: CUDACOORDINATIONFLOAT PAIR GROUPA=1-107 GROUPB=2-108 R_0=0.4 THREADS=256
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/plumed.dat:gpu128: CUDACOORDINATIONFLOAT PAIR GROUPA=1-107 GROUPB=2-108 R_0=0.4 THREADS=128
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/plumed.dat:gpu64:  CUDACOORDINATIONFLOAT PAIR GROUPA=1-107 GROUPB=2-108 R_0=0.4 THREADS=64
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/plumed.dat:PRINT ARG=gpu,cpu FILE=colvars FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatestPair/rt-float-orthobpc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.5f
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/plumed.dat:gpu:    CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 NN=6 R_0=0.4
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/plumed.dat:gpu512: CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 NN=6 R_0=0.4 THREADS=512
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/plumed.dat:gpu256: CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 NN=6 R_0=0.4 THREADS=256
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/plumed.dat:gpu128: CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 NN=6 R_0=0.4 THREADS=128
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/plumed.dat:gpu64:  CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 NN=6 R_0=0.4 THREADS=64
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/plumed.dat:PRINT ARG=gpu,cpu FILE=colvars FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbcMM_2NN/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.5f
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/plumed.dat:gpu:    CUDACOORDINATION PAIR GROUPA=1-107,1-107 GROUPB=2-108,2-108 R_0=1 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/plumed.dat:gpu512: CUDACOORDINATION PAIR GROUPA=1-107,1-107 GROUPB=2-108,2-108 R_0=1 THREADS=512 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/plumed.dat:gpu256: CUDACOORDINATION PAIR GROUPA=1-107,1-107 GROUPB=2-108,2-108 R_0=1 THREADS=256 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/plumed.dat:gpu128: CUDACOORDINATION PAIR GROUPA=1-107,1-107 GROUPB=2-108,2-108 R_0=1 THREADS=128 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/plumed.dat:gpu64:  CUDACOORDINATION PAIR GROUPA=1-107,1-107 GROUPB=2-108,2-108 R_0=1 THREADS=64 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/plumed.dat:diff:    CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/plumed.dat:diff64:  CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/plumed.dat:PRINT ARG=gpu,cpu FILE=colvars FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatestPair/rt-double-multiple-nopbc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.4f
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/plumed.dat:gpu:    CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 NN=6 MM=14 R_0=0.4
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/plumed.dat:gpu512: CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 NN=6 MM=14 R_0=0.4 THREADS=512
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/plumed.dat:gpu256: CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 NN=6 MM=14 R_0=0.4 THREADS=256
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/plumed.dat:gpu128: CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 NN=6 MM=14 R_0=0.4 THREADS=128
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/plumed.dat:gpu64:  CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 NN=6 MM=14 R_0=0.4 THREADS=64
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/plumed.dat:PRINT ARG=gpu,cpu FILE=colvars FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatestPair/rt-double-orthopbc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.5f
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/plumed.dat:gpu:    CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 R_0=1 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/plumed.dat:gpu512: CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 R_0=1 THREADS=512 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/plumed.dat:gpu256: CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 R_0=1 THREADS=256 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/plumed.dat:gpu128: CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 R_0=1 THREADS=128 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/plumed.dat:gpu64:  CUDACOORDINATION PAIR GROUPA=1-107 GROUPB=2-108 R_0=1 THREADS=64 NOPBC
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/plumed.dat:#PRINT ARG=gpu,cpu FILE=colvars FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatestPair/rt-double-nopbc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.4f
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/plumed.dat:gpu: CUDACOORDINATIONFLOAT GROUPA=1-108 R_0=1 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/plumed.dat:gpu512: CUDACOORDINATIONFLOAT GROUPA=1-108 R_0=1 THREADS=512 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/plumed.dat:gpu256: CUDACOORDINATIONFLOAT GROUPA=1-108 R_0=1 THREADS=256 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/plumed.dat:gpu128: CUDACOORDINATIONFLOAT GROUPA=1-108 R_0=1 THREADS=128 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/plumed.dat:gpu64: CUDACOORDINATIONFLOAT GROUPA=1-108 R_0=1 THREADS=64 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-float-nopbc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.5f
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/plumed.dat:gpu: CUDACOORDINATIONFLOAT GROUPA=@mdatoms     NN=6 R_0=0.4
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/plumed.dat:gpu512: CUDACOORDINATIONFLOAT GROUPA=@mdatoms  NN=6 R_0=0.4 THREADS=512
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/plumed.dat:gpu256: CUDACOORDINATIONFLOAT GROUPA=@mdatoms  NN=6 R_0=0.4 THREADS=256
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/plumed.dat:gpu128: CUDACOORDINATIONFLOAT GROUPA=@mdatoms  NN=6 R_0=0.4 THREADS=128
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/plumed.dat:gpu64:  CUDACOORDINATIONFLOAT GROUPA=@mdatoms  NN=6 R_0=0.4 THREADS=64
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/plumed.dat:PRINT ARG=gpu,cpu FILE=colvars FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatest/rt-float-orthobpc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.5f
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/plumed.dat:gpu: CUDACOORDINATION GROUPA=@mdatoms     NN=6 R_0=0.4
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/plumed.dat:gpu512: CUDACOORDINATION GROUPA=@mdatoms  NN=6 R_0=0.4 THREADS=512
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/plumed.dat:gpu256: CUDACOORDINATION GROUPA=@mdatoms  NN=6 R_0=0.4 THREADS=256
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/plumed.dat:gpu128: CUDACOORDINATION GROUPA=@mdatoms  NN=6 R_0=0.4 THREADS=128
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/plumed.dat:gpu64:  CUDACOORDINATION GROUPA=@mdatoms  NN=6 R_0=0.4 THREADS=64
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/plumed.dat:PRINT ARG=gpu,cpu FILE=colvars FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbcMM_2NN/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.5f
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/plumed.dat:gpu: CUDACOORDINATION GROUPA=1-108,1-108  R_0=1 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/plumed.dat:gpu512: CUDACOORDINATION GROUPA=1-108,1-108  R_0=1 THREADS=512 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/plumed.dat:gpu256: CUDACOORDINATION GROUPA=1-108,1-108  R_0=1 THREADS=256 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/plumed.dat:gpu128: CUDACOORDINATION GROUPA=1-108,1-108  R_0=1 THREADS=128 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/plumed.dat:gpu64: CUDACOORDINATION GROUPA=1-108,1-108  R_0=1 THREADS=64 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/plumed.dat:#PRINT ARG=gpu,cpu FILE=colvar FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatest/rt-double-multiple-nopbc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.4f
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/plumed.dat:gpu: CUDACOORDINATION GROUPA=@mdatoms     NN=6 MM=14 R_0=0.4
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/plumed.dat:gpu512: CUDACOORDINATION GROUPA=@mdatoms  NN=6 MM=14 R_0=0.4 THREADS=512
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/plumed.dat:gpu256: CUDACOORDINATION GROUPA=@mdatoms  NN=6 MM=14 R_0=0.4 THREADS=256
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/plumed.dat:gpu128: CUDACOORDINATION GROUPA=@mdatoms  NN=6 MM=14 R_0=0.4 THREADS=128
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/plumed.dat:gpu64:  CUDACOORDINATION GROUPA=@mdatoms  NN=6 MM=14 R_0=0.4 THREADS=64
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/plumed.dat:PRINT ARG=gpu,cpu FILE=colvars FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatest/rt-double-orthopbc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.5f
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/plumed.dat:gpu: CUDACOORDINATION GROUPA=1-108  R_0=1.1 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/plumed.dat:gpu512: CUDACOORDINATION GROUPA=1-108  R_0=1.1 THREADS=512 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/plumed.dat:gpu256: CUDACOORDINATION GROUPA=1-108  R_0=1.1 THREADS=256 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/plumed.dat:gpu128: CUDACOORDINATION GROUPA=1-108  R_0=1.1 THREADS=128 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/plumed.dat:gpu64: CUDACOORDINATION GROUPA=1-108  R_0=1.1 THREADS=64 NOPBC
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/plumed.dat:#PRINT ARG=gpu,cpu FILE=colvars FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatest/rt-double-nopbc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.4f
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/plumed.dat:gpu: CUDACOORDINATIONFLOAT GROUPA=1-108 GROUPB=1-108 R_0=1 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/plumed.dat:gpu512: CUDACOORDINATIONFLOAT GROUPA=1-108 GROUPB=1-108 R_0=1 THREADS=512 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/plumed.dat:gpu256: CUDACOORDINATIONFLOAT GROUPA=1-108 GROUPB=1-108 R_0=1 THREADS=256 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/plumed.dat:gpu128: CUDACOORDINATIONFLOAT GROUPA=1-108 GROUPB=1-108 R_0=1 THREADS=128 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/plumed.dat:gpu64: CUDACOORDINATIONFLOAT GROUPA=1-108 GROUPB=1-108 R_0=1 THREADS=64 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-float-nopbc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.5f
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/plumed.dat:gpu: CUDACOORDINATIONFLOAT GROUPA=@mdatoms GROUPB=@mdatoms     NN=6 R_0=0.4
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/plumed.dat:gpu512: CUDACOORDINATIONFLOAT GROUPA=@mdatoms GROUPB=@mdatoms  NN=6 R_0=0.4 THREADS=512
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/plumed.dat:gpu256: CUDACOORDINATIONFLOAT GROUPA=@mdatoms GROUPB=@mdatoms  NN=6 R_0=0.4 THREADS=256
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/plumed.dat:gpu128: CUDACOORDINATIONFLOAT GROUPA=@mdatoms GROUPB=@mdatoms  NN=6 R_0=0.4 THREADS=128
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/plumed.dat:gpu64:  CUDACOORDINATIONFLOAT GROUPA=@mdatoms GROUPB=@mdatoms  NN=6 R_0=0.4 THREADS=64
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/plumed.dat:PRINT ARG=gpu,cpu FILE=colvars FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatestWB/rt-float-orthobpc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.5f
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/plumed.dat:gpu: CUDACOORDINATION GROUPA=@mdatoms GROUPB=@mdatoms     NN=6 R_0=0.4
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/plumed.dat:gpu512: CUDACOORDINATION GROUPA=@mdatoms GROUPB=@mdatoms  NN=6 R_0=0.4 THREADS=512
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/plumed.dat:gpu256: CUDACOORDINATION GROUPA=@mdatoms GROUPB=@mdatoms  NN=6 R_0=0.4 THREADS=256
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/plumed.dat:gpu128: CUDACOORDINATION GROUPA=@mdatoms GROUPB=@mdatoms  NN=6 R_0=0.4 THREADS=128
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/plumed.dat:gpu64:  CUDACOORDINATION GROUPA=@mdatoms GROUPB=@mdatoms  NN=6 R_0=0.4 THREADS=64
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/plumed.dat:PRINT ARG=gpu,cpu FILE=colvars FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbcMM_2NN/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.5f
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/plumed.dat:gpu: CUDACOORDINATION GROUPA=1-108,1-108 GROUPB=1-108,1-108  R_0=1 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/plumed.dat:gpu512: CUDACOORDINATION GROUPA=1-108,1-108 GROUPB=1-108,1-108  R_0=1 THREADS=512 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/plumed.dat:gpu256: CUDACOORDINATION GROUPA=1-108,1-108 GROUPB=1-108,1-108  R_0=1 THREADS=256 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/plumed.dat:gpu128: CUDACOORDINATION GROUPA=1-108,1-108 GROUPB=1-108,1-108  R_0=1 THREADS=128 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/plumed.dat:gpu64: CUDACOORDINATION GROUPA=1-108,1-108 GROUPB=1-108,1-108  R_0=1 THREADS=64 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/plumed.dat:#PRINT ARG=gpu,cpu FILE=colvar FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatestWB/rt-double-multiple-nopbc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.4f
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/plumed.dat:gpu: CUDACOORDINATION GROUPA=@mdatoms GROUPB=@mdatoms     NN=6 MM=14 R_0=0.4
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/plumed.dat:gpu512: CUDACOORDINATION GROUPA=@mdatoms GROUPB=@mdatoms  NN=6 MM=14 R_0=0.4 THREADS=512
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/plumed.dat:gpu256: CUDACOORDINATION GROUPA=@mdatoms GROUPB=@mdatoms  NN=6 MM=14 R_0=0.4 THREADS=256
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/plumed.dat:gpu128: CUDACOORDINATION GROUPA=@mdatoms GROUPB=@mdatoms  NN=6 MM=14 R_0=0.4 THREADS=128
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/plumed.dat:gpu64:  CUDACOORDINATION GROUPA=@mdatoms GROUPB=@mdatoms  NN=6 MM=14 R_0=0.4 THREADS=64
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/plumed.dat:PRINT ARG=gpu,cpu FILE=colvars FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatestWB/rt-double-orthopbc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.5f
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/config:    echo '#! FIELDS time parameter cpu-gpu'
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/deriv_delta.reference:#! FIELDS time parameter cpu-gpu
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/plumed.dat:LOAD FILE=../../../../CudaCoordination.so
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/plumed.dat:gpu: CUDACOORDINATION GROUPA=1-108 GROUPB=1-108  R_0=1.1 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/plumed.dat:gpu512: CUDACOORDINATION GROUPA=1-108 GROUPB=1-108  R_0=1.1 THREADS=512 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/plumed.dat:gpu256: CUDACOORDINATION GROUPA=1-108 GROUPB=1-108  R_0=1.1 THREADS=256 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/plumed.dat:gpu128: CUDACOORDINATION GROUPA=1-108 GROUPB=1-108  R_0=1.1 THREADS=128 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/plumed.dat:gpu64: CUDACOORDINATION GROUPA=1-108 GROUPB=1-108  R_0=1.1 THREADS=64 NOPBC
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/plumed.dat:diff: CUSTOM ARG=gpu,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/plumed.dat:diff512: CUSTOM ARG=gpu512,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/plumed.dat:diff256: CUSTOM ARG=gpu256,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/plumed.dat:diff128: CUSTOM ARG=gpu128,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/plumed.dat:diff64: CUSTOM ARG=gpu64,cpu FUNC=y-x PERIODIC=NO
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/plumed.dat:#PRINT ARG=gpu,cpu FILE=colvars FMT=%8.4f STRIDE=1
plugins/cudaCoord/regtest/cudatestWB/rt-double-nopbc/plumed.dat:DUMPDERIVATIVES ARG=gpu,cpu FILE=deriv FMT=%8.4f
plugins/cudaCoord/Coordination.cu:   cudaOnPlumed is free software: you can redistribute it and/or modify
plugins/cudaCoord/Coordination.cu:   cudaOnPlumed is distributed in the hope that it will be useful,
plugins/cudaCoord/Coordination.cu:#include "cudaHelpers.cuh"
plugins/cudaCoord/Coordination.cu:#include "cuda_runtime.h"
plugins/cudaCoord/Coordination.cu://+PLUMEDOC COLVAR CUDACOORDINATION
plugins/cudaCoord/Coordination.cu:Calculate coordination numbers. Like coordination, but on nvdia gpu and with
plugins/cudaCoord/Coordination.cu:CUDACOORDINATION can be invoked with CUDACOORDINATIONFLOAT, but that version
plugins/cudaCoord/Coordination.cu:desktop-based Nvidia cards.
plugins/cudaCoord/Coordination.cu:CUDACOORDINATION GROUPA=group R_0=0.3
plugins/cudaCoord/Coordination.cu:template <typename calculateFloat> class CudaCoordination : public Colvar {
plugins/cudaCoord/Coordination.cu:  /// the pointer to the coordinates on the GPU
plugins/cudaCoord/Coordination.cu:  thrust::device_vector<calculateFloat> cudaPositions;
plugins/cudaCoord/Coordination.cu:  /// the pointer to the nn list on the GPU
plugins/cudaCoord/Coordination.cu:  thrust::device_vector<calculateFloat> cudaCoordination;
plugins/cudaCoord/Coordination.cu:  thrust::device_vector<calculateFloat> cudaDerivatives;
plugins/cudaCoord/Coordination.cu:  thrust::device_vector<calculateFloat> cudaVirial;
plugins/cudaCoord/Coordination.cu:  thrust::device_vector<unsigned> cudaTrueIndexes;
plugins/cudaCoord/Coordination.cu:  cudaStream_t streamDerivatives;
plugins/cudaCoord/Coordination.cu:  cudaStream_t streamVirial;
plugins/cudaCoord/Coordination.cu:  cudaStream_t streamCoordination;
plugins/cudaCoord/Coordination.cu:  PLMD::GPU::rationalSwitchParameters<calculateFloat> switchingParameters;
plugins/cudaCoord/Coordination.cu:  PLMD::GPU::ortoPBCs<calculateFloat> myPBC;
plugins/cudaCoord/Coordination.cu:  void setUpPermanentGPUMemory();
plugins/cudaCoord/Coordination.cu:  explicit CudaCoordination (const ActionOptions &);
plugins/cudaCoord/Coordination.cu:  virtual ~CudaCoordination();
plugins/cudaCoord/Coordination.cu:using CudaCoordination_d = CudaCoordination<double>;
plugins/cudaCoord/Coordination.cu:using CudaCoordination_f = CudaCoordination<float>;
plugins/cudaCoord/Coordination.cu:PLUMED_REGISTER_ACTION (CudaCoordination_d, "CUDACOORDINATION")
plugins/cudaCoord/Coordination.cu:PLUMED_REGISTER_ACTION (CudaCoordination_f, "CUDACOORDINATIONFLOAT")
plugins/cudaCoord/Coordination.cu:void CudaCoordination<calculateFloat>::setUpPermanentGPUMemory() {
plugins/cudaCoord/Coordination.cu:  cudaPositions.resize (3 * nat);
plugins/cudaCoord/Coordination.cu:  cudaDerivatives.resize (3 * nat);
plugins/cudaCoord/Coordination.cu:  cudaTrueIndexes.resize (nat);
plugins/cudaCoord/Coordination.cu:  cudaTrueIndexes = trueIndexes;
plugins/cudaCoord/Coordination.cu:void CudaCoordination<calculateFloat>::registerKeywords (Keywords &keys) {
plugins/cudaCoord/Coordination.cu:CudaCoordination<calculateFloat>::~CudaCoordination() {
plugins/cudaCoord/Coordination.cu:  cudaStreamDestroy (streamDerivatives);
plugins/cudaCoord/Coordination.cu:  cudaStreamDestroy (streamVirial);
plugins/cudaCoord/Coordination.cu:  cudaStreamDestroy (streamCoordination);
plugins/cudaCoord/Coordination.cu:void CudaCoordination<calculateFloat>::calculate() {
plugins/cudaCoord/Coordination.cu:  /***************************copying data on the GPU**************************/
plugins/cudaCoord/Coordination.cu:  CUDAHELPERS::plmdDataToGPU (cudaPositions, positions, streamDerivatives);
plugins/cudaCoord/Coordination.cu:  /***************************copying data on the GPU**************************/
plugins/cudaCoord/Coordination.cu:  cudaDeviceSynchronize();
plugins/cudaCoord/Coordination.cu:  CUDAHELPERS::plmdDataFromGPU (cudaDerivatives, deriv, streamDerivatives);
plugins/cudaCoord/Coordination.cu:    size_t runningThreads = CUDAHELPERS::threadsPerBlock (
plugins/cudaCoord/Coordination.cu:    CUDAHELPERS::doReductionND<dataperthread> (
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaVirial.data()),
plugins/cudaCoord/Coordination.cu:    CUDAHELPERS::doReduction1D<dataperthread> (
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaCoordination.data()),
plugins/cudaCoord/Coordination.cu:      CUDAHELPERS::plmdDataFromGPU (
plugins/cudaCoord/Coordination.cu:      reductionMemoryVirial.swap (cudaVirial);
plugins/cudaCoord/Coordination.cu:      reductionMemoryCoord.swap (cudaCoordination);
plugins/cudaCoord/Coordination.cu:  if (reductionMemoryCoord.size() > cudaCoordination.size())
plugins/cudaCoord/Coordination.cu:    reductionMemoryCoord.swap (cudaCoordination);
plugins/cudaCoord/Coordination.cu:  if (reductionMemoryVirial.size() > cudaVirial.size())
plugins/cudaCoord/Coordination.cu:    reductionMemoryVirial.swap (cudaVirial);
plugins/cudaCoord/Coordination.cu:  cudaDeviceSynchronize();
plugins/cudaCoord/Coordination.cu:    PLMD::GPU::invData<T> const pbc) {
plugins/cudaCoord/Coordination.cu:    return PLMD::GPU::pbcClamp (val * pbc.inv) * pbc.val;
plugins/cudaCoord/Coordination.cu:              const PLMD::GPU::rationalSwitchParameters<calculateFloat>
plugins/cudaCoord/Coordination.cu:              const PLMD::GPU::ortoPBCs<calculateFloat> myPBC,
plugins/cudaCoord/Coordination.cu:size_t CudaCoordination<calculateFloat>::doSelf() {
plugins/cudaCoord/Coordination.cu:  size_t nat = cudaPositions.size() / 3;
plugins/cudaCoord/Coordination.cu:  /**********************allocating the memory on the GPU**********************/
plugins/cudaCoord/Coordination.cu:  cudaCoordination.resize (nat);
plugins/cudaCoord/Coordination.cu:  cudaVirial.resize (nat * 9);
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaPositions.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaTrueIndexes.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaCoordination.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaDerivatives.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaVirial.data()));
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaPositions.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaTrueIndexes.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaCoordination.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaDerivatives.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaVirial.data()));
plugins/cudaCoord/Coordination.cu:              const PLMD::GPU::rationalSwitchParameters<calculateFloat>
plugins/cudaCoord/Coordination.cu:              const PLMD::GPU::ortoPBCs<calculateFloat> myPBC,
plugins/cudaCoord/Coordination.cu:              const PLMD::GPU::rationalSwitchParameters<calculateFloat>
plugins/cudaCoord/Coordination.cu:              const PLMD::GPU::ortoPBCs<calculateFloat> myPBC,
plugins/cudaCoord/Coordination.cu:size_t CudaCoordination<calculateFloat>::doDual() {
plugins/cudaCoord/Coordination.cu:  /**********************allocating the memory on the GPU**********************/
plugins/cudaCoord/Coordination.cu:  cudaCoordination.resize (atomsInA);
plugins/cudaCoord/Coordination.cu:  cudaVirial.resize (atomsInA * 9);
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaPositions.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaPositions.data()) + 3 * atomsInA,
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaTrueIndexes.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaCoordination.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaDerivatives.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaVirial.data()));
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaPositions.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaPositions.data()) + 3 * atomsInA,
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaTrueIndexes.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaDerivatives.data()) + 3 * atomsInA);
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaPositions.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaPositions.data()) + 3 * atomsInA,
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaTrueIndexes.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaCoordination.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaDerivatives.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaVirial.data()));
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaPositions.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaPositions.data()) + 3 * atomsInA,
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaTrueIndexes.data()),
plugins/cudaCoord/Coordination.cu:                   thrust::raw_pointer_cast (cudaDerivatives.data()) + 3 * atomsInA);
plugins/cudaCoord/Coordination.cu:  const PLMD::GPU::rationalSwitchParameters<calculateFloat>
plugins/cudaCoord/Coordination.cu:  const PLMD::GPU::ortoPBCs<calculateFloat> myPBC,
plugins/cudaCoord/Coordination.cu:size_t CudaCoordination<calculateFloat>::doPair() {
plugins/cudaCoord/Coordination.cu:  size_t couples = cudaPositions.size() / 6;
plugins/cudaCoord/Coordination.cu:  /**********************allocating the memory on the GPU**********************/
plugins/cudaCoord/Coordination.cu:  cudaCoordination.resize (neededThreads);
plugins/cudaCoord/Coordination.cu:  cudaVirial.resize (neededThreads * 9);
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaPositions.data()),
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaPositions.data()) + 3 * couples,
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaTrueIndexes.data()),
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaCoordination.data()),
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaDerivatives.data()),
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaDerivatives.data()) + 3 * couples,
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaVirial.data()));
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaPositions.data()),
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaPositions.data()) + 3 * couples,
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaTrueIndexes.data()),
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaCoordination.data()),
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaDerivatives.data()),
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaDerivatives.data()) + 3 * couples,
plugins/cudaCoord/Coordination.cu:      thrust::raw_pointer_cast (cudaVirial.data()));
plugins/cudaCoord/Coordination.cu:CudaCoordination<calculateFloat>::CudaCoordination (const ActionOptions &ao)
plugins/cudaCoord/Coordination.cu:  { // loading data to the GPU
plugins/cudaCoord/Coordination.cu:      PLMD::GPU::getpcuda_Rational<<<1, 2>>> (
plugins/cudaCoord/Coordination.cu:  cudaStreamCreate (&streamDerivatives);
plugins/cudaCoord/Coordination.cu:  cudaStreamCreate (&streamVirial);
plugins/cudaCoord/Coordination.cu:  cudaStreamCreate (&streamCoordination);
plugins/cudaCoord/Coordination.cu:  setUpPermanentGPUMemory();
plugins/cudaCoord/Coordination.cu:  cudaFuncAttributes attr;
plugins/cudaCoord/Coordination.cu:      cudaFuncGetAttributes (&attr, &getSelfCoord<true, calculateFloat>);
plugins/cudaCoord/Coordination.cu:      cudaFuncGetAttributes (&attr, &getSelfCoord<false, calculateFloat>);
plugins/cudaCoord/Coordination.cu:      cudaFuncGetAttributes (&attr, &getDerivDual<true, calculateFloat>);
plugins/cudaCoord/Coordination.cu:      cudaFuncGetAttributes (&attr, &getCoordDual<true, calculateFloat>);
plugins/cudaCoord/Coordination.cu:      cudaFuncGetAttributes (&attr, &getDerivDual<false, calculateFloat>);
plugins/cudaCoord/Coordination.cu:      cudaFuncGetAttributes (&attr, &getCoordDual<false, calculateFloat>);
plugins/cudaCoord/Coordination.cu:      cudaFuncGetAttributes (&attr, &getCoordPair<true, calculateFloat>);
plugins/cudaCoord/Coordination.cu:      cudaFuncGetAttributes (&attr, &getCoordPair<false, calculateFloat>);
plugins/cudaCoord/Coordination.cu:  log << "GPU info:\n"
plugins/cudaCoord/Coordination.cu:  // cudaFuncGetAttributes (&attr, &getSelfCoord<true, calculateFloat>);
plugins/cudaCoord/Coordination.cu:  // cudaFuncGetAttributes ( &attr, &getSelfCoord<false,calculateFloat> );
plugins/cudaCoord/Coordination.cu:  // cudaFuncGetAttributes ( &attr,
plugins/cudaCoord/Coordination.cu:  // &CUDAHELPERS::reduction1DKernel<calculateFloat, 128, 4> ); std::cout<<
user-doc/METATENSORMOD.md:# architecture (Apple Silicon, arm64), CUDA versions, and newer versions of
user-doc/METATENSORMOD.md:# alternatively if you have a CUDA-enabled GPU, you can use the corresponding
user-doc/METATENSORMOD.md:# pre-built library (here for CUDA 12.1):
user-doc/tutorials/aa-masterclass-21-7.txt:This might be different when using a GPU (and indeed the increment in the wallclock time should be smaller).
user-doc/tutorials/a-masterclass-22-01.txt:We also remember that gpu could be called with the `-gpu_id` flag and proper resource management can be controlle with the various `-ntomp`, 
user-doc/tutorials/hrex.txt:Notice that when you run with GPUs acceptance could be different from 1.0.
user-doc/tutorials/hrex.txt:made with GPUs are typically not reproducible to machine precision. For a large system, even
user-doc/tutorials/hrex.txt:- Choose neighbor list update (nstlist) that divides replex. Notice that running with GPUs
user-doc/tutorials/a-masterclass-22-10.txt:on your machine  might lead to much better performance. In addition, using a GPU will also make your simulations
user-doc/tutorials/a-masterclass-22-10.txt:For everything else, use the same settings you will use in production (ideally, same number of processes per replica, same GPU settings, etc).
user-doc/tutorials/a-masterclass-22-10.txt:**Notice that this might be expected to fail if you use a GPU.**
user-doc/tutorials/performance-optimization.txt:This will run a simulation for 500 steps, without using any GPU-acceleration, and with 12 OpenMP 
user-doc/tutorials/performance-optimization.txt:when using the GPU.
user-doc/tutorials/performance-optimization.txt:\subsection performance-optimization-2g Running GROMACS on the GPU
user-doc/tutorials/performance-optimization.txt:Let's see what happens using the GPU. In this case the first couple of thousands steps are kind of different since
user-doc/tutorials/performance-optimization.txt:GROMACS tries to optimize the GPU load, so we should run a longer simulation to estimate the simulation speed.
user-doc/tutorials/performance-optimization.txt:Let's compare the timings with/without GPU and with/without PLUMED using the following commands
user-doc/tutorials/performance-optimization.txt:| GPU | PLUMED | Wall t (s) | PLUMED time (s) |
user-doc/tutorials/performance-optimization.txt:PLUMED is not running on the GPU, so the total time spent in PLUMED is roughly the same both with and without GPU
user-doc/tutorials/performance-optimization.txt:However, when GROMACS runs using the GPU the load balancing transfers part of the load to the GPU.
user-doc/Installation.md:If you have a GPU, you might want to use the CUDA-accelerated version of LibTorch. For example, the following script downloads the <a href="https://download.pytorch.org/libtorch/cu117/libtorch-shared-with-deps-2.0.0%2Bcu117.zip"> `libtorch-shared-with-deps-2.0.0%2Bcu117.zip`</a> (2.0.0, GPU, Cuda 11.7, pre-cxx11 ABI binary).
user-doc/Installation.md:In both CPU and GPU cases, the location of the include and library files need to be exported in the environment:
user-doc/Installation.md:- `--enable-libtorch` will first try first to link the CUDA-enabled library and if it does not found it will try to link the CPU-only version.
user-doc/Installation.md:- To verify that the linking of LibTorch is succesful, one should look at the output of the configure commands: `checking libtorch[cpu/cuda] [without extra libs/with -ltorch_cpu ... ]`.  If any of these commands are succesfull, it will return `... yes`. Otherwise, the configure will display a warning (and not an error!) that says: `configure: WARNING: cannot enable __PLUMED_HAS_LIBTORCH`. In this case, it is recommended to examine the output of the above commands in the config.log file to understand the reason (e.g. it cannot find the required libraries).
user-doc/Performances.md:- \subpage GMXGPU 
user-doc/Performances.md:\page GMXGPU GROMACS and PLUMED with GPU
user-doc/Performances.md:your CPU and your GPU (either using CUDA or OpenCL for newer versions of
user-doc/Performances.md:performed on the GPU while long-range and bonded interactions are at the
user-doc/Performances.md:interactions GROMACS can optimize the balance between GPU/CPU loading 
user-doc/Performances.md:range interactions. This means that the CPU/GPU balance will be optimized 
user-doc/Performances.md:on the GPU or GROMACS + PLUMED can be different, try to change the number
user-doc/Performances.md:can use the same GPU:
user-doc/Performances.md:i.e. if you have 4 cores and 2 GPU you can:
user-doc/Performances.md:- use 2 MPI/2GPU/2OPENMP:
user-doc/Performances.md:mpiexec -np 2 gmx_mpi mdrun -nb gpu -ntomp 2 -pin on -gpu_id 01
user-doc/Performances.md:- use 4 MPI/2GPU:
user-doc/Performances.md:mpiexec -np 4 gmx_mpi mdrun -nb gpu -ntomp 1 -pin on -gpu_id 0011
user-doc/Performances.md:mpiexec -np 2 gmx_mpi mdrun -nb gpu -ntomp 2 -pin on -gpu_id 01
user-doc/spelling_words.dict:CUDA
user-doc/spelling_words.dict:OpenCL
user-doc/spelling_words.dict:GPUs
user-doc/Introduction.md:- [GPUMD](https://gpumd.org/)
user-doc/ISDB.md:It is highly recommened to install the CUDA version of LibTorch to calculate \ref EMMIVOX efficiently on the GPU.
configure.ac:PLUMED_CONFIG_ENABLE([af_cuda],[search for arrayfire_cuda],[no])
configure.ac:  PLUMED_CHECK_PACKAGE([arrayfire.h],[af_is_double],[__PLUMED_HAS_ARRAYFIRE],[afopencl])
configure.ac:  PLUMED_CHECK_PACKAGE([arrayfire.h],[af_is_double],[__PLUMED_HAS_ARRAYFIRE_OCL],[afopencl])
configure.ac:if test "$af_cuda" = true ; then
configure.ac:  PLUMED_CHECK_PACKAGE([arrayfire.h],[af_is_double],[__PLUMED_HAS_ARRAYFIRE],[afcuda])
configure.ac:  PLUMED_CHECK_PACKAGE([arrayfire.h],[af_is_double],[__PLUMED_HAS_ARRAYFIRE_CUDA],[afcuda])
configure.ac:  # CUDA and CPU libtorch libs have different libraries
configure.ac:  # first test CUDA program
configure.ac:  PLUMED_CHECK_CXX_PACKAGE([libtorch_cuda],[
configure.ac:    #include <torch/cuda.h>
configure.ac:      std::cerr << "CUDA is available: " << torch::cuda::is_available() << std::endl;
configure.ac:      device = torch::kCUDA;
configure.ac:  ], [__PLUMED_HAS_LIBTORCH], [ torch_cpu c10 c10_cuda torch_cuda ], [true])
configure.ac:  # AC_MSG_NOTICE([CUDA-enabled libtorch not found (or devices not available), trying with CPU version.])
patches/namd-2.12.diff: 	$(CUDALIB) \
patches/namd-2.12.diff: #include "DeviceCUDA.h"
patches/namd-2.12.diff: #ifdef NAMD_CUDA
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/ewald/pme_coordinate_receiver_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/ewald/pme_pp_comm_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/nbnxm/nbnxm_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/timing/gpu_timing.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gpuforcereduction.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static const bool c_disableAlternatingWait = (getenv("GMX_DISABLE_ALTERNATING_GPU_WAIT") != nullptr);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   bool                  useGpuPmePpComms,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   bool                  receivePmeForceToGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gmx_pme_receive_f(fr->pmePpCommGpu.get(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                      useGpuPmePpComms,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                      receivePmeForceToGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* GPU kernel launch overhead is already timed separately */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (!nbv->useGpu())
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Launch the prepare_step and spread stages of PME GPU.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param[in]  useMdGpuGraph        Whether MD GPU Graph is in use.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static inline void launchPmeGpuSpread(gmx_pme_t*            pmedata,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                      GpuEventSynchronizer* xReadyOnDevice,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                      bool                  useMdGpuGraph,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    wallcycle_start(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_prepare_computation(pmedata, box, wcycle, stepWork);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool                           useGpuDirectComm         = false;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gmx::PmeCoordinateReceiverGpu* pmeCoordinateReceiverGpu = nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_launch_spread(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            pmedata, xReadyOnDevice, wcycle, lambdaQ, useGpuDirectComm, pmeCoordinateReceiverGpu, useMdGpuGraph);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Launch the FFT and gather stages of PME GPU
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void launchPmeGpuFftAndGather(gmx_pme_t*               pmedata,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_launch_complex_transforms(pmedata, wcycle, stepWork);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_launch_gather(pmedata, wcycle, lambdaQ);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * Blocks until PME GPU tasks are completed, and gets the output forces and virial/energy
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void pmeGpuWaitAndReduce(gmx_pme_t*               pme,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_wait_and_reduce(pme, stepWork, wcycle, forceWithVirial, enerd, lambdaQ);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: *  Polling wait for either of the PME or nonbonded GPU tasks.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * Instead of a static order in waiting for GPU tasks, this function
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * one of the reductions, regardless of the GPU task completion order.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void alternatePmeNbGpuWaitReduce(nonbonded_verlet_t* nbv,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool isPmeGpuDone = false;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool isNbGpuDone  = false;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gmx::ArrayRef<const gmx::RVec> pmeGpuForces;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    while (!isPmeGpuDone || !isNbGpuDone)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!isPmeGpuDone)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GpuTaskCompletion completionType =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    (isNbGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            isPmeGpuDone = pme_gpu_try_finish_task(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!isNbGpuDone)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GpuTaskCompletion completionType =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    (isPmeGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            isNbGpuDone = Nbnxm::gpu_try_finish_task(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    nbv->gpu_nbv,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (isNbGpuDone)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        && (domainWork.haveCpuLocalForceWork || !stepWork.useGpuFBufferOps
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            || (havePpDomainDecomposition && !stepWork.useGpuFHalo)))
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    domainWork.haveGpuBondedWork =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            ((fr.listedForcesGpu != nullptr) && fr.listedForcesGpu->haveInteractions());
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuXBufferOps || simulationWork.useGpuFBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        GMX_ASSERT(simulationWork.useGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    flags.useGpuXBufferOps = simulationWork.useGpuXBufferOps && !flags.doNeighborSearch;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    flags.useGpuFBufferOps = simulationWork.useGpuFBufferOps && !flags.computeVirial;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    flags.useGpuPmeFReduction =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            flags.computeSlowForces && flags.useGpuFBufferOps
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            && (simulationWork.haveGpuPmeOnPpRank() || simulationWork.useGpuPmePpCommunication);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    flags.useGpuXHalo              = simulationWork.useGpuHaloExchange && !flags.doNeighborSearch;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    flags.useGpuFHalo              = simulationWork.useGpuHaloExchange && flags.useGpuFBufferOps;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    flags.haveGpuPmeOnThisRank     = simulationWork.haveGpuPmeOnPpRank() && flags.computeSlowForces;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:             && !(flags.computeVirial || simulationWork.useGpuNonbonded || flags.haveGpuPmeOnThisRank));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // On NS steps, the buffer is cleared in stateGpu->reinit, no need to clear it twice.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    flags.clearGpuFBufferEarly =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            flags.useGpuFHalo && !domainWork.haveCpuLocalForceWork && !flags.doNeighborSearch;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/* \brief Launch end-of-step GPU tasks: buffer clearing and rolling pruning.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void launchGpuEndOfStepTasks(nonbonded_verlet_t*               nbv,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                    gmx::ListedForcesGpu*             listedForcesGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork.simulationWork.useGpuNonbonded && runScheduleWork.stepWork.computeNonbondedForces)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:         * clear kernel launches can leave the GPU idle while it could be running
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (nbv->isDynamicPruningStepGpu(step))
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            nbv->dispatchPruneKernelGpu(step);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* now clear the GPU outputs while we finish the step on the CPU */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        Nbnxm::gpu_clear_outputs(nbv->gpu_nbv, runScheduleWork.stepWork.computeVirial);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork.stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pme_gpu_reinit_computation(pmedata, runScheduleWork.simulationWork.useMdGpuGraph, wcycle);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork.domainWork.haveGpuBondedWork && runScheduleWork.stepWork.computeEnergy)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        listedForcesGpu->waitAccumulateEnergyTerms(enerd);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        listedForcesGpu->clearEnergies();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Compute the number of times the "local coordinates ready on device" GPU event will be used as a synchronization point.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * When some work is offloaded to GPU, force calculation should wait for the atom coordinates to
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * or from the GPU integration at the end of the previous step.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param pmeSendCoordinatesFromGpu Whether peer-to-peer communication is used for PME coordinates.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                                          bool pmeSendCoordinatesFromGpu)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (pmeSendCoordinatesFromGpu)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                       "GPU PME PP communications require having a separate PME rank");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // Event is consumed by gmx_pme_send_coordinates for GPU PME PP Communications
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // Event is consumed by launchPmeGpuSpread
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.computeNonbondedForces && stepWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // Event is consumed by convertCoordinatesGpu
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.useGpuXHalo)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // Event is consumed by communicateGpuHaloCoordinates
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.clearGpuFBufferEarly && simulationWork.useGpuUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Compute the number of times the "local forces ready on device" GPU event will be used as a synchronization point.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param useOrEmulateGpuNb Whether GPU non-bonded calculations are used or emulated.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param alternateGpuWait Whether alternating wait/reduce scheme is used.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                                          bool useOrEmulateGpuNb,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                                          bool alternateGpuWait)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool eventUsedInGpuForceReduction =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:             || (simulationWork.havePpDomainDecomposition && !simulationWork.useGpuHaloExchange));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool gpuForceReductionUsed = useOrEmulateGpuNb && !alternateGpuWait && stepWork.useGpuFBufferOps
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (gpuForceReductionUsed && eventUsedInGpuForceReduction)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool gpuForceHaloUsed = simulationWork.havePpDomainDecomposition && stepWork.computeForces
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                            && stepWork.useGpuFHalo;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (gpuForceHaloUsed)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Setup for the local GPU force reduction:
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] stateGpu            GPU state propagator object
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] gpuForceReduction   GPU force reduction object
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] pmePpCommGpu        PME-PP GPU communication object
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void setupLocalGpuForceReduction(const gmx::MdrunScheduleWorkload* runScheduleWork,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        gmx::StatePropagatorDataGpu*      stateGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        gmx::GpuForceReduction*           gpuForceReduction,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        gmx::PmePpCommGpu*                pmePpCommGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "GPU force reduction is not compatible with MTS");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // (re-)initialize local GPU force reduction
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gpuForceReduction->reinit(stateGpu->getForces(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                              stateGpu->fReducedOnDevice(AtomLocality::Local));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gpuForceReduction->registerNbnxmForce(Nbnxm::gpu_get_f(nbv->gpu_nbv));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GpuEventSynchronizer*   pmeSynchronizer     = nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork->simulationWork.haveGpuPmeOnPpRank())
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pmeForcePtr = pme_gpu_get_device_f(pmedata);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            pmeSynchronizer     = pme_gpu_get_f_ready_synchronizer(pmedata);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    else if (runScheduleWork->simulationWork.useGpuPmePpCommunication)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pmeForcePtr = pmePpCommGpu->getGpuForceStagingPtr();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                pmeSynchronizer = pmePpCommGpu->getForcesReadySynchronizer();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        gpuForceReduction->registerRvecForce(pmeForcePtr);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!runScheduleWork->simulationWork.useGpuPmePpCommunication || GMX_THREAD_MPI)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GMX_ASSERT(pmeSynchronizer != nullptr, "PME force ready cuda event should not be NULL");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            gpuForceReduction->addDependency(pmeSynchronizer);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            && !runScheduleWork->simulationWork.useGpuHaloExchange))
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::Local));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork->simulationWork.useGpuHaloExchange)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        gpuForceReduction->addDependency(dd->gpuHaloExchange[0][0]->getForcesReadyOnDeviceEvent());
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Setup for the non-local GPU force reduction:
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] stateGpu            GPU state propagator object
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] gpuForceReduction   GPU force reduction object
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void setupNonLocalGpuForceReduction(const gmx::MdrunScheduleWorkload* runScheduleWork,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                           gmx::StatePropagatorDataGpu*      stateGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                           gmx::GpuForceReduction*           gpuForceReduction,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // (re-)initialize non-local GPU force reduction
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gpuForceReduction->reinit(stateGpu->getForces(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                              stateGpu->fReducedOnDevice(AtomLocality::NonLocal));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gpuForceReduction->registerNbnxmForce(Nbnxm::gpu_get_f(nbv->gpu_nbv));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::NonLocal));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gmx::StatePropagatorDataGpu* stateGpu = fr->stateGpu;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (fr->listedForcesGpu)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            fr->listedForcesGpu->updateHaveInteractions(top->idef);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.doNeighborSearch && gmx::needStateGpu(simulationWork))
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->reinit(mdatoms->homenr,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.doNeighborSearch && simulationWork.haveGpuPmeOnPpRank())
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        GMX_ASSERT(gmx::needStateGpu(simulationWork), "StatePropagatorDataGpu is needed");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // TODO: This should be moved into PME setup function ( pme_gpu_prepare_computation(...) )
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pme_gpu_set_device_x(fr->pmedata, stateGpu->getCoordinates());
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            (stepWork.haveGpuPmeOnThisRank || simulationWork.useGpuXBufferOps || simulationWork.useGpuUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    ? stateGpu->getCoordinatesReadyOnDeviceEvent(AtomLocality::Local, simulationWork, stepWork)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.clearGpuFBufferEarly)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // GPU Force halo exchange will set a subset of local atoms with remote non-local data.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // which is satisfied when localXReadyOnDevice has been marked for GPU update case.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        GpuEventSynchronizer* dependency = simulationWork.useGpuUpdate ? localXReadyOnDevice : nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->clearForcesOnGpu(AtomLocality::Local, dependency);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    const bool pmeSendCoordinatesFromGpu =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            simulationWork.useGpuPmePpCommunication && !(stepWork.doNeighborSearch);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    const bool reinitGpuPmePpComms =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            simulationWork.useGpuPmePpCommunication && (stepWork.doNeighborSearch);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GMX_ASSERT(simulationWork.useGpuHaloExchange
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                       == ((cr->dd != nullptr) && (!cr->dd->gpuHaloExchange[0].empty())),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "The GPU halo exchange is active, but it has not been constructed.");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool gmx_used_in_debug haveCopiedXFromGpu = false;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // Copy coordinate from the GPU if update is on the GPU and there
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->copyCoordinatesFromGpu(x.unpaddedArrayRef(), AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        haveCopiedXFromGpu = true;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.haveGpuPmeOnThisRank || stepWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        GMX_ASSERT(stateGpu != nullptr, "stateGpu should not be null");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        simulationWork, stepWork, pmeSendCoordinatesFromGpu);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!simulationWork.useGpuUpdate || stepWork.doNeighborSearch)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        else if (simulationWork.useGpuUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            stateGpu->setXUpdatedOnDeviceEventExpectedConsumptionCount(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!pmeSendCoordinatesFromGpu && !stepWork.doNeighborSearch && simulationWork.useGpuUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 simulationWork.useGpuPmePpCommunication,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 reinitGpuPmePpComms,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 pmeSendCoordinatesFromGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 stepWork.useGpuPmeFReduction,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 pmeSendCoordinatesFromGpu ? localXReadyOnDevice : nullptr,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 simulationWork.useMdGpuGraph,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        launchPmeGpuSpread(fr->pmedata,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                           simulationWork.useMdGpuGraph,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* initialize the GPU nbnxm atom data and bonded data structures */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (simulationWork.useGpuNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // Note: cycle counting only nononbondeds, GPU listed forces counts internally
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            Nbnxm::gpu_init_atomdata(nbv->gpu_nbv, nbv->nbat.get());
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (fr->listedForcesGpu)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                 * interactions to the GPU, where the grid order is
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        nbv->getGridIndices(), top->idef, Nbnxm::gpuGetNBAtomData(nbv->gpu_nbv));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (simulationWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            nbv->atomdata_init_copy_x_to_nbat_x_gpu();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (simulationWork.useGpuFBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            setupLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        stateGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        fr->gpuForceReduction[gmx::AtomLocality::Local].get(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        fr->pmePpCommGpu.get(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                setupNonLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                               stateGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                               fr->gpuForceReduction[gmx::AtomLocality::NonLocal].get(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GMX_ASSERT(stateGpu, "stateGpu should be valid when buffer ops are offloaded");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            nbv->convertCoordinatesGpu(AtomLocality::Local, stateGpu->getCoordinates(), localXReadyOnDevice);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (simulationWork.useGpuUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                GMX_ASSERT(stateGpu, "need a valid stateGpu object");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuNonbonded && (stepWork.computeNonbondedForces || domainWork.haveGpuBondedWork))
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        ddBalanceRegionHandler.openBeforeForceComputationGpu();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        Nbnxm::gpu_upload_shiftvec(nbv->gpu_nbv, nbv->nbat.get());
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!stepWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            Nbnxm::gpu_copy_xq_to_gpu(nbv->gpu_nbv, nbv->nbat.get(), AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // with X buffer ops offloaded to the GPU on all but the search steps
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (domainWork.haveGpuBondedWork && !simulationWork.havePpDomainDecomposition)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* launch local nonbonded work on GPU */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // In PME GPU and mixed mode we launch FFT / gather after the
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // X copy/transform to allow overlap as well as after the GPU NB
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        launchPmeGpuFftAndGather(fr->pmedata,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // TODO refactor this GPU halo exchange re-initialisation
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // to location in do_md where GPU halo exchange is
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // constructed at partitioning, after above stateGpu
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (simulationWork.useGpuHaloExchange)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                reinitGpuHaloExchange(*cr, stateGpu->getCoordinates(), stateGpu->getForces());
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GpuEventSynchronizer* gpuCoordinateHaloLaunched = nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.useGpuXHalo)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                gpuCoordinateHaloLaunched = communicateGpuHaloCoordinates(*cr, box, localXReadyOnDevice);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->copyCoordinatesFromGpu(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                            x.unpaddedArrayRef(), AtomLocality::NonLocal, gpuCoordinateHaloLaunched);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (simulationWork.useGpuUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                            (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (!stepWork.useGpuXHalo)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(), AtomLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                GpuEventSynchronizer* xReadyOnDeviceEvent = stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        AtomLocality::NonLocal, simulationWork, stepWork, gpuCoordinateHaloLaunched);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (stepWork.useGpuXHalo && domainWork.haveCpuNonLocalForceWork)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    /* We already enqueued an event for Gpu Halo exchange completion into the
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                nbv->convertCoordinatesGpu(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        AtomLocality::NonLocal, stateGpu->getCoordinates(), xReadyOnDeviceEvent);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (simulationWork.useGpuNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (!stepWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                Nbnxm::gpu_copy_xq_to_gpu(nbv->gpu_nbv, nbv->nbat.get(), AtomLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (domainWork.haveGpuBondedWork)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            /* launch non-local nonbonded tasks on GPU */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuNonbonded && stepWork.computeNonbondedForces)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            Nbnxm::gpu_launch_cpyback(nbv->gpu_nbv, nbv->nbat.get(), stepWork, AtomLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        Nbnxm::gpu_launch_cpyback(nbv->gpu_nbv, nbv->nbat.get(), stepWork, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (domainWork.haveGpuBondedWork && stepWork.computeEnergy)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            fr->listedForcesGpu->launchEnergyTransfer();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // For the rest of the CPU tasks that depend on GPU-update produced coordinates,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                || (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    const bool useOrEmulateGpuNb = simulationWork.useGpuNonbonded || fr->nbv->emulateGpu();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (!useOrEmulateGpuNb)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.useGpuXHalo && domainWork.haveCpuNonLocalForceWork)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:         * Happens here on the CPU both with and without GPU.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.computeNonbondedForces && !useOrEmulateGpuNb)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            (stepWork.haveGpuPmeOnThisRank || needToReceivePmeResultsFromSeparateRank);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:     * GPU we must wait for the PME calculation (dhdl) results to finish before sampling the
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            pmeGpuWaitAndReduce(fr->pmedata,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   simulationWork.useGpuPmePpCommunication,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   stepWork.useGpuPmeFReduction,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.havePpDomainDecomposition && stepWork.computeForces && stepWork.useGpuFHalo
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(), AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFBufferOps),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFHalo),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "The schedule below does not allow for nonbonded MTS with GPU halo exchange");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // Will store the amount of cycles spent waiting for the GPU that
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    float cycles_wait_gpu = 0;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (useOrEmulateGpuNb && stepWork.computeNonbondedForces)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (simulationWork.useGpuNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                cycles_wait_gpu += Nbnxm::gpu_wait_finish_task(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        nbv->gpu_nbv,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.useGpuFBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                fr->gpuForceReduction[gmx::AtomLocality::NonLocal]->execute();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (!stepWork.useGpuFHalo)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    // copy from GPU input for dd_move_f()
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->copyForcesFromGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (fr->nbv->emulateGpu() && stepWork.computeVirial)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // With both nonbonded and PME offloaded a GPU on the same rank, we use
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    const bool alternateGpuWait = (!c_disableAlternatingWait && stepWork.haveGpuPmeOnThisRank
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   && simulationWork.useGpuNonbonded && !simulationWork.havePpDomainDecomposition
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   && !stepWork.useGpuFBufferOps && !needEarlyPmeResults);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            simulationWork, domainWork, stepWork, useOrEmulateGpuNb, alternateGpuWait);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // If expectedLocalFReadyOnDeviceConsumptionCount == 0, stateGpu can be uninitialized
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->setFReadyOnDeviceEventExpectedConsumptionCount(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:         * If we use a GPU this will overlap with GPU work, so in that case
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.useGpuFHalo)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                gmx::FixedCapacityVector<GpuEventSynchronizer*, 2> gpuForceHaloDependencies;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (domainWork.haveCpuLocalForceWork || stepWork.clearGpuFBufferEarly)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    gpuForceHaloDependencies.push_back(stateGpu->fReadyOnDevice(AtomLocality::Local));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                gpuForceHaloDependencies.push_back(stateGpu->fReducedOnDevice(AtomLocality::NonLocal));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                communicateGpuHaloForces(*cr, accumulateForces, &gpuForceHaloDependencies);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (stepWork.useGpuFBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->waitForcesReadyOnHost(AtomLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (alternateGpuWait)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        alternatePmeNbGpuWaitReduce(fr->nbv.get(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (!alternateGpuWait && stepWork.haveGpuPmeOnThisRank && !needEarlyPmeResults)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pmeGpuWaitAndReduce(fr->pmedata,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* Wait for local GPU NB outputs on the non-alternating wait path */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (!alternateGpuWait && stepWork.computeNonbondedForces && simulationWork.useGpuNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* Measured overhead on CUDA and OpenCL with(out) GPU sharing
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        const float gpuWaitApiOverheadMargin = 2e6F; /* cycles */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        const float waitCycles               = Nbnxm::gpu_wait_finish_task(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                nbv->gpu_nbv,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            DdBalanceRegionWaitedForGpu waitedForGpu = DdBalanceRegionWaitedForGpu::yes;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.computeForces && waitCycles <= gpuWaitApiOverheadMargin)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                waitedForGpu = DdBalanceRegionWaitedForGpu::no;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            ddBalanceRegionHandler.closeAfterForceComputationGpu(cycles_wait_gpu, waitedForGpu);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (fr->nbv->emulateGpu())
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // If on GPU PME-PP comms path, receive forces from PME before GPU buffer ops
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (needToReceivePmeResultsFromSeparateRank && simulationWork.useGpuPmePpCommunication && !needEarlyPmeResults)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                               simulationWork.useGpuPmePpCommunication,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                               stepWork.useGpuPmeFReduction,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* Do the nonbonded GPU (or emulation) force buffer reduction
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFBufferOps),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (useOrEmulateGpuNb && !alternateGpuWait)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.useGpuFBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // - copy is not perfomed if GPU force halo exchange is active, because it would overwrite the result
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (domainWork.haveLocalForceContribInCpuBuffer && !stepWork.useGpuFHalo)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                stateGpu->copyForcesToGpu(forceWithShift, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                fr->gpuForceReduction[gmx::AtomLocality::Local]->execute();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (!simulationWork.useGpuUpdate
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                || (simulationWork.useGpuUpdate && haveDDAtomOrdering(*cr) && simulationWork.useCpuPmePpCommunication)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    /* We have previously issued force reduction on the GPU, but we will
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                stateGpu->copyForcesFromGpu(forceWithShift, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->setFReadyOnDeviceEventExpectedConsumptionCount(AtomLocality::Local, 1);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    launchGpuEndOfStepTasks(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            nbv, fr->listedForcesGpu.get(), fr->pmedata, enerd, *runScheduleWork, step, wcycle);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // TODO refactor this and unify with above GPU PME-PP / GPU update path call to the same function
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                               simulationWork.useGpuPmePpCommunication,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* In case we don't have constraints and are using GPUs, the next balancing
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/ewald/pme_coordinate_receiver_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/ewald/pme_pp_comm_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/nbnxm/nbnxm_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/timing/gpu_timing.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gpuforcereduction.h"
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:static const bool c_disableAlternatingWait = (getenv("GMX_DISABLE_ALTERNATING_GPU_WAIT") != nullptr);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                   bool                  useGpuPmePpComms,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                   bool                  receivePmeForceToGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    gmx_pme_receive_f(fr->pmePpCommGpu.get(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                      useGpuPmePpComms,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                      receivePmeForceToGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    /* GPU kernel launch overhead is already timed separately */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (!nbv->useGpu())
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Launch the prepare_step and spread stages of PME GPU.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param[in]  useMdGpuGraph        Whether MD GPU Graph is in use.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:static inline void launchPmeGpuSpread(gmx_pme_t*            pmedata,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                      GpuEventSynchronizer* xReadyOnDevice,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                      bool                  useMdGpuGraph,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    wallcycle_start(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_prepare_computation(pmedata, box, wcycle, stepWork);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    bool                           useGpuDirectComm         = false;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    gmx::PmeCoordinateReceiverGpu* pmeCoordinateReceiverGpu = nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_launch_spread(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            pmedata, xReadyOnDevice, wcycle, lambdaQ, useGpuDirectComm, pmeCoordinateReceiverGpu, useMdGpuGraph);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Launch the FFT and gather stages of PME GPU
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:static void launchPmeGpuFftAndGather(gmx_pme_t*               pmedata,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_launch_complex_transforms(pmedata, wcycle, stepWork);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_launch_gather(pmedata, wcycle, lambdaQ);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * Blocks until PME GPU tasks are completed, and gets the output forces and virial/energy
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:static void pmeGpuWaitAndReduce(gmx_pme_t*               pme,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_wait_and_reduce(pme, stepWork, wcycle, forceWithVirial, enerd, lambdaQ);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: *  Polling wait for either of the PME or nonbonded GPU tasks.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * Instead of a static order in waiting for GPU tasks, this function
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * one of the reductions, regardless of the GPU task completion order.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:static void alternatePmeNbGpuWaitReduce(nonbonded_verlet_t* nbv,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    bool isPmeGpuDone = false;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    bool isNbGpuDone  = false;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    gmx::ArrayRef<const gmx::RVec> pmeGpuForces;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    while (!isPmeGpuDone || !isNbGpuDone)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (!isPmeGpuDone)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            GpuTaskCompletion completionType =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    (isNbGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            isPmeGpuDone = pme_gpu_try_finish_task(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (!isNbGpuDone)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            GpuTaskCompletion completionType =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    (isPmeGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            isNbGpuDone = Nbnxm::gpu_try_finish_task(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    nbv->gpu_nbv,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (isNbGpuDone)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        && (domainWork.haveCpuLocalForceWork || !stepWork.useGpuFBufferOps
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            || (havePpDomainDecomposition && !stepWork.useGpuFHalo)))
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    domainWork.haveGpuBondedWork =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            ((fr.listedForcesGpu != nullptr) && fr.listedForcesGpu->haveInteractions());
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuXBufferOps || simulationWork.useGpuFBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        GMX_ASSERT(simulationWork.useGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    flags.useGpuXBufferOps = simulationWork.useGpuXBufferOps && !flags.doNeighborSearch;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    flags.useGpuFBufferOps = simulationWork.useGpuFBufferOps && !flags.computeVirial;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    flags.useGpuPmeFReduction =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            flags.computeSlowForces && flags.useGpuFBufferOps
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            && (simulationWork.haveGpuPmeOnPpRank() || simulationWork.useGpuPmePpCommunication);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    flags.useGpuXHalo              = simulationWork.useGpuHaloExchange && !flags.doNeighborSearch;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    flags.useGpuFHalo              = simulationWork.useGpuHaloExchange && flags.useGpuFBufferOps;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    flags.haveGpuPmeOnThisRank     = simulationWork.haveGpuPmeOnPpRank() && flags.computeSlowForces;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:             && !(flags.computeVirial || simulationWork.useGpuNonbonded || flags.haveGpuPmeOnThisRank));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    // On NS steps, the buffer is cleared in stateGpu->reinit, no need to clear it twice.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    flags.clearGpuFBufferEarly =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            flags.useGpuFHalo && !domainWork.haveCpuLocalForceWork && !flags.doNeighborSearch;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:/* \brief Launch end-of-step GPU tasks: buffer clearing and rolling pruning.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:static void launchGpuEndOfStepTasks(nonbonded_verlet_t*               nbv,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                    gmx::ListedForcesGpu*             listedForcesGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork.simulationWork.useGpuNonbonded && runScheduleWork.stepWork.computeNonbondedForces)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:         * clear kernel launches can leave the GPU idle while it could be running
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (nbv->isDynamicPruningStepGpu(step))
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            nbv->dispatchPruneKernelGpu(step);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        /* now clear the GPU outputs while we finish the step on the CPU */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        Nbnxm::gpu_clear_outputs(nbv->gpu_nbv, runScheduleWork.stepWork.computeVirial);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork.stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        pme_gpu_reinit_computation(pmedata, runScheduleWork.simulationWork.useMdGpuGraph, wcycle);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork.domainWork.haveGpuBondedWork && runScheduleWork.stepWork.computeEnergy)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        listedForcesGpu->waitAccumulateEnergyTerms(enerd);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        listedForcesGpu->clearEnergies();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Compute the number of times the "local coordinates ready on device" GPU event will be used as a synchronization point.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * When some work is offloaded to GPU, force calculation should wait for the atom coordinates to
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * or from the GPU integration at the end of the previous step.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param pmeSendCoordinatesFromGpu Whether peer-to-peer communication is used for PME coordinates.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                                          bool pmeSendCoordinatesFromGpu)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (pmeSendCoordinatesFromGpu)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                       "GPU PME PP communications require having a separate PME rank");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            // Event is consumed by gmx_pme_send_coordinates for GPU PME PP Communications
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            // Event is consumed by launchPmeGpuSpread
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.computeNonbondedForces && stepWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            // Event is consumed by convertCoordinatesGpu
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.useGpuXHalo)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        // Event is consumed by communicateGpuHaloCoordinates
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.clearGpuFBufferEarly && simulationWork.useGpuUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Compute the number of times the "local forces ready on device" GPU event will be used as a synchronization point.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param useOrEmulateGpuNb Whether GPU non-bonded calculations are used or emulated.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param alternateGpuWait Whether alternating wait/reduce scheme is used.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                                          bool useOrEmulateGpuNb,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                                          bool alternateGpuWait)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    bool eventUsedInGpuForceReduction =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:             || (simulationWork.havePpDomainDecomposition && !simulationWork.useGpuHaloExchange));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    bool gpuForceReductionUsed = useOrEmulateGpuNb && !alternateGpuWait && stepWork.useGpuFBufferOps
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (gpuForceReductionUsed && eventUsedInGpuForceReduction)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    bool gpuForceHaloUsed = simulationWork.havePpDomainDecomposition && stepWork.computeForces
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                            && stepWork.useGpuFHalo;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (gpuForceHaloUsed)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Setup for the local GPU force reduction:
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] stateGpu            GPU state propagator object
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] gpuForceReduction   GPU force reduction object
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] pmePpCommGpu        PME-PP GPU communication object
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:static void setupLocalGpuForceReduction(const gmx::MdrunScheduleWorkload* runScheduleWork,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                        gmx::StatePropagatorDataGpu*      stateGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                        gmx::GpuForceReduction*           gpuForceReduction,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                        gmx::PmePpCommGpu*                pmePpCommGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:               "GPU force reduction is not compatible with MTS");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    // (re-)initialize local GPU force reduction
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    gpuForceReduction->reinit(stateGpu->getForces(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                              stateGpu->fReducedOnDevice(AtomLocality::Local));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    gpuForceReduction->registerNbnxmForce(Nbnxm::gpu_get_f(nbv->gpu_nbv));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    GpuEventSynchronizer*   pmeSynchronizer     = nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork->simulationWork.haveGpuPmeOnPpRank())
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        pmeForcePtr = pme_gpu_get_device_f(pmedata);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            pmeSynchronizer     = pme_gpu_get_f_ready_synchronizer(pmedata);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    else if (runScheduleWork->simulationWork.useGpuPmePpCommunication)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        pmeForcePtr = pmePpCommGpu->getGpuForceStagingPtr();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                pmeSynchronizer = pmePpCommGpu->getForcesReadySynchronizer();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        gpuForceReduction->registerRvecForce(pmeForcePtr);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (!runScheduleWork->simulationWork.useGpuPmePpCommunication || GMX_THREAD_MPI)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            GMX_ASSERT(pmeSynchronizer != nullptr, "PME force ready cuda event should not be NULL");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            gpuForceReduction->addDependency(pmeSynchronizer);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            && !runScheduleWork->simulationWork.useGpuHaloExchange))
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::Local));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork->simulationWork.useGpuHaloExchange)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        gpuForceReduction->addDependency(dd->gpuHaloExchange[0][0]->getForcesReadyOnDeviceEvent());
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Setup for the non-local GPU force reduction:
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] stateGpu            GPU state propagator object
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] gpuForceReduction   GPU force reduction object
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:static void setupNonLocalGpuForceReduction(const gmx::MdrunScheduleWorkload* runScheduleWork,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                           gmx::StatePropagatorDataGpu*      stateGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                           gmx::GpuForceReduction*           gpuForceReduction,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    // (re-)initialize non-local GPU force reduction
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    gpuForceReduction->reinit(stateGpu->getForces(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                              stateGpu->fReducedOnDevice(AtomLocality::NonLocal));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    gpuForceReduction->registerNbnxmForce(Nbnxm::gpu_get_f(nbv->gpu_nbv));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::NonLocal));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    gmx::StatePropagatorDataGpu* stateGpu = fr->stateGpu;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (fr->listedForcesGpu)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            fr->listedForcesGpu->updateHaveInteractions(top->idef);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.doNeighborSearch && gmx::needStateGpu(simulationWork))
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->reinit(mdatoms->homenr,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.doNeighborSearch && simulationWork.haveGpuPmeOnPpRank())
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        GMX_ASSERT(gmx::needStateGpu(simulationWork), "StatePropagatorDataGpu is needed");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        // TODO: This should be moved into PME setup function ( pme_gpu_prepare_computation(...) )
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        pme_gpu_set_device_x(fr->pmedata, stateGpu->getCoordinates());
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            (stepWork.haveGpuPmeOnThisRank || simulationWork.useGpuXBufferOps || simulationWork.useGpuUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    ? stateGpu->getCoordinatesReadyOnDeviceEvent(AtomLocality::Local, simulationWork, stepWork)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.clearGpuFBufferEarly)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        // GPU Force halo exchange will set a subset of local atoms with remote non-local data.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        // which is satisfied when localXReadyOnDevice has been marked for GPU update case.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        GpuEventSynchronizer* dependency = simulationWork.useGpuUpdate ? localXReadyOnDevice : nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->clearForcesOnGpu(AtomLocality::Local, dependency);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    const bool pmeSendCoordinatesFromGpu =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            simulationWork.useGpuPmePpCommunication && !(stepWork.doNeighborSearch);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    const bool reinitGpuPmePpComms =
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            simulationWork.useGpuPmePpCommunication && (stepWork.doNeighborSearch);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    GMX_ASSERT(simulationWork.useGpuHaloExchange
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                       == ((cr->dd != nullptr) && (!cr->dd->gpuHaloExchange[0].empty())),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:               "The GPU halo exchange is active, but it has not been constructed.");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    bool gmx_used_in_debug haveCopiedXFromGpu = false;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    // Copy coordinate from the GPU if update is on the GPU and there
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->copyCoordinatesFromGpu(x.unpaddedArrayRef(), AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        haveCopiedXFromGpu = true;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.haveGpuPmeOnThisRank || stepWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        GMX_ASSERT(stateGpu != nullptr, "stateGpu should not be null");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                        simulationWork, stepWork, pmeSendCoordinatesFromGpu);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (!simulationWork.useGpuUpdate || stepWork.doNeighborSearch)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        else if (simulationWork.useGpuUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            stateGpu->setXUpdatedOnDeviceEventExpectedConsumptionCount(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (!pmeSendCoordinatesFromGpu && !stepWork.doNeighborSearch && simulationWork.useGpuUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                 simulationWork.useGpuPmePpCommunication,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                 reinitGpuPmePpComms,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                 pmeSendCoordinatesFromGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                 stepWork.useGpuPmeFReduction,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                 pmeSendCoordinatesFromGpu ? localXReadyOnDevice : nullptr,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                 simulationWork.useMdGpuGraph,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        launchPmeGpuSpread(fr->pmedata,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                           simulationWork.useMdGpuGraph,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        /* initialize the GPU nbnxm atom data and bonded data structures */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (simulationWork.useGpuNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            // Note: cycle counting only nononbondeds, GPU listed forces counts internally
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            Nbnxm::gpu_init_atomdata(nbv->gpu_nbv, nbv->nbat.get());
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (fr->listedForcesGpu)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                 * interactions to the GPU, where the grid order is
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                        nbv->getGridIndices(), top->idef, Nbnxm::gpuGetNBAtomData(nbv->gpu_nbv));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (simulationWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            nbv->atomdata_init_copy_x_to_nbat_x_gpu();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (simulationWork.useGpuFBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            setupLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                        stateGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                        fr->gpuForceReduction[gmx::AtomLocality::Local].get(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                        fr->pmePpCommGpu.get(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                setupNonLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                               stateGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                               fr->gpuForceReduction[gmx::AtomLocality::NonLocal].get(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            GMX_ASSERT(stateGpu, "stateGpu should be valid when buffer ops are offloaded");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            nbv->convertCoordinatesGpu(AtomLocality::Local, stateGpu->getCoordinates(), localXReadyOnDevice);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (simulationWork.useGpuUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                GMX_ASSERT(stateGpu, "need a valid stateGpu object");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuNonbonded && (stepWork.computeNonbondedForces || domainWork.haveGpuBondedWork))
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        ddBalanceRegionHandler.openBeforeForceComputationGpu();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        Nbnxm::gpu_upload_shiftvec(nbv->gpu_nbv, nbv->nbat.get());
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (!stepWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            Nbnxm::gpu_copy_xq_to_gpu(nbv->gpu_nbv, nbv->nbat.get(), AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        // with X buffer ops offloaded to the GPU on all but the search steps
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (domainWork.haveGpuBondedWork && !simulationWork.havePpDomainDecomposition)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        /* launch local nonbonded work on GPU */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        // In PME GPU and mixed mode we launch FFT / gather after the
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        // X copy/transform to allow overlap as well as after the GPU NB
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        launchPmeGpuFftAndGather(fr->pmedata,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            // TODO refactor this GPU halo exchange re-initialisation
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            // to location in do_md where GPU halo exchange is
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            // constructed at partitioning, after above stateGpu
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (simulationWork.useGpuHaloExchange)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                reinitGpuHaloExchange(*cr, stateGpu->getCoordinates(), stateGpu->getForces());
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            GpuEventSynchronizer* gpuCoordinateHaloLaunched = nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.useGpuXHalo)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                gpuCoordinateHaloLaunched = communicateGpuHaloCoordinates(*cr, box, localXReadyOnDevice);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->copyCoordinatesFromGpu(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                            x.unpaddedArrayRef(), AtomLocality::NonLocal, gpuCoordinateHaloLaunched);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                if (simulationWork.useGpuUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                            (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                if (!stepWork.useGpuXHalo)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(), AtomLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                GpuEventSynchronizer* xReadyOnDeviceEvent = stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                        AtomLocality::NonLocal, simulationWork, stepWork, gpuCoordinateHaloLaunched);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                if (stepWork.useGpuXHalo && domainWork.haveCpuNonLocalForceWork)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    /* We already enqueued an event for Gpu Halo exchange completion into the
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                nbv->convertCoordinatesGpu(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                        AtomLocality::NonLocal, stateGpu->getCoordinates(), xReadyOnDeviceEvent);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (simulationWork.useGpuNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (!stepWork.useGpuXBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                Nbnxm::gpu_copy_xq_to_gpu(nbv->gpu_nbv, nbv->nbat.get(), AtomLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (domainWork.haveGpuBondedWork)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            /* launch non-local nonbonded tasks on GPU */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuNonbonded && stepWork.computeNonbondedForces)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            Nbnxm::gpu_launch_cpyback(nbv->gpu_nbv, nbv->nbat.get(), stepWork, AtomLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        Nbnxm::gpu_launch_cpyback(nbv->gpu_nbv, nbv->nbat.get(), stepWork, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (domainWork.haveGpuBondedWork && stepWork.computeEnergy)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            fr->listedForcesGpu->launchEnergyTransfer();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    // For the rest of the CPU tasks that depend on GPU-update produced coordinates,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                || (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    const bool useOrEmulateGpuNb = simulationWork.useGpuNonbonded || fr->nbv->emulateGpu();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (!useOrEmulateGpuNb)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.useGpuXHalo && domainWork.haveCpuNonLocalForceWork)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:         * Happens here on the CPU both with and without GPU.
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.computeNonbondedForces && !useOrEmulateGpuNb)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            (stepWork.haveGpuPmeOnThisRank || needToReceivePmeResultsFromSeparateRank);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:     * GPU we must wait for the PME calculation (dhdl) results to finish before sampling the
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            pmeGpuWaitAndReduce(fr->pmedata,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                   simulationWork.useGpuPmePpCommunication,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                   stepWork.useGpuPmeFReduction,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.havePpDomainDecomposition && stepWork.computeForces && stepWork.useGpuFHalo
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(), AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFBufferOps),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFHalo),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:               "The schedule below does not allow for nonbonded MTS with GPU halo exchange");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    // Will store the amount of cycles spent waiting for the GPU that
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    float cycles_wait_gpu = 0;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (useOrEmulateGpuNb && stepWork.computeNonbondedForces)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (simulationWork.useGpuNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                cycles_wait_gpu += Nbnxm::gpu_wait_finish_task(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                        nbv->gpu_nbv,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.useGpuFBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                fr->gpuForceReduction[gmx::AtomLocality::NonLocal]->execute();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                if (!stepWork.useGpuFHalo)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    // copy from GPU input for dd_move_f()
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->copyForcesFromGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (fr->nbv->emulateGpu() && stepWork.computeVirial)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    // With both nonbonded and PME offloaded a GPU on the same rank, we use
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    const bool alternateGpuWait = (!c_disableAlternatingWait && stepWork.haveGpuPmeOnThisRank
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                   && simulationWork.useGpuNonbonded && !simulationWork.havePpDomainDecomposition
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                                   && !stepWork.useGpuFBufferOps && !needEarlyPmeResults);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            simulationWork, domainWork, stepWork, useOrEmulateGpuNb, alternateGpuWait);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    // If expectedLocalFReadyOnDeviceConsumptionCount == 0, stateGpu can be uninitialized
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->setFReadyOnDeviceEventExpectedConsumptionCount(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:         * If we use a GPU this will overlap with GPU work, so in that case
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.useGpuFHalo)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                gmx::FixedCapacityVector<GpuEventSynchronizer*, 2> gpuForceHaloDependencies;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                if (domainWork.haveCpuLocalForceWork || stepWork.clearGpuFBufferEarly)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    gpuForceHaloDependencies.push_back(stateGpu->fReadyOnDevice(AtomLocality::Local));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                gpuForceHaloDependencies.push_back(stateGpu->fReducedOnDevice(AtomLocality::NonLocal));
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                communicateGpuHaloForces(*cr, accumulateForces, &gpuForceHaloDependencies);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                if (stepWork.useGpuFBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->waitForcesReadyOnHost(AtomLocality::NonLocal);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (alternateGpuWait)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        alternatePmeNbGpuWaitReduce(fr->nbv.get(),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (!alternateGpuWait && stepWork.haveGpuPmeOnThisRank && !needEarlyPmeResults)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        pmeGpuWaitAndReduce(fr->pmedata,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    /* Wait for local GPU NB outputs on the non-alternating wait path */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (!alternateGpuWait && stepWork.computeNonbondedForces && simulationWork.useGpuNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        /* Measured overhead on CUDA and OpenCL with(out) GPU sharing
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        const float gpuWaitApiOverheadMargin = 2e6F; /* cycles */
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        const float waitCycles               = Nbnxm::gpu_wait_finish_task(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                nbv->gpu_nbv,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            DdBalanceRegionWaitedForGpu waitedForGpu = DdBalanceRegionWaitedForGpu::yes;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.computeForces && waitCycles <= gpuWaitApiOverheadMargin)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                waitedForGpu = DdBalanceRegionWaitedForGpu::no;
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            ddBalanceRegionHandler.closeAfterForceComputationGpu(cycles_wait_gpu, waitedForGpu);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (fr->nbv->emulateGpu())
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    // If on GPU PME-PP comms path, receive forces from PME before GPU buffer ops
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (needToReceivePmeResultsFromSeparateRank && simulationWork.useGpuPmePpCommunication && !needEarlyPmeResults)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                               simulationWork.useGpuPmePpCommunication,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                               stepWork.useGpuPmeFReduction,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    /* Do the nonbonded GPU (or emulation) force buffer reduction
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFBufferOps),
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (useOrEmulateGpuNb && !alternateGpuWait)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.useGpuFBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            // - copy is not perfomed if GPU force halo exchange is active, because it would overwrite the result
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (domainWork.haveLocalForceContribInCpuBuffer && !stepWork.useGpuFHalo)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                stateGpu->copyForcesToGpu(forceWithShift, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                fr->gpuForceReduction[gmx::AtomLocality::Local]->execute();
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (!simulationWork.useGpuUpdate
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                || (simulationWork.useGpuUpdate && haveDDAtomOrdering(*cr) && simulationWork.useCpuPmePpCommunication)
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    /* We have previously issued force reduction on the GPU, but we will
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                stateGpu->copyForcesFromGpu(forceWithShift, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->setFReadyOnDeviceEventExpectedConsumptionCount(AtomLocality::Local, 1);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    launchGpuEndOfStepTasks(
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:            nbv, fr->listedForcesGpu.get(), fr->pmedata, enerd, *runScheduleWork, step, wcycle);
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    // TODO refactor this and unify with above GPU PME-PP / GPU update path call to the same function
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:                               simulationWork.useGpuPmePpCommunication,
patches/gromacs-2023.5.diff/src/gromacs/mdlib/sim_util.cpp:    /* In case we don't have constraints and are using GPUs, the next balancing
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/ewald/pme_gpu_program.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/ewald/pme_pp_comm_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/gpu_utils/device_stream_manager.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/gpu_utils/gpueventsynchronizer_helpers.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/mdlib/gpuforcereduction.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/mdlib/mdgraph_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/taskassignment/decidegpuusage.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/taskassignment/usergpuids.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/timing/gpu_timing.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp: * the GPU communication flags are set to false in non-tMPI and non-CUDA builds.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp: * \param[in]  useGpuForNonbonded   True if the nonbonded task is offloaded in this run.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                         const bool           useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    devFlags.enableGpuBufferOps = (GMX_GPU_CUDA || GMX_GPU_SYCL) && useGpuForNonbonded
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                  && (getenv("GMX_USE_GPU_BUFFER_OPS") != nullptr);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    if (getenv("GMX_CUDA_GRAPH") != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (GMX_HAVE_CUDA_GRAPH_SUPPORT)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            devFlags.enableCudaGraphs = true;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "GMX_CUDA_GRAPH environment variable is detected. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "The experimental CUDA Graphs feature will be used if run conditions "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            devFlags.enableCudaGraphs = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "GMX_CUDA_GRAPH environment variable is detected, "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "but the CUDA version in use is below the minumum requirement (11.1). "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "CUDA Graphs will be disabled.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    // Flag use to enable GPU-aware MPI depenendent features such PME GPU decomposition
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    // GPU-aware MPI is marked available if it has been detected by GROMACS or detection fails but
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    devFlags.canUseGpuAwareMpi = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    // Direct GPU comm path is being used with GPU-aware MPI
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    // make sure underlying MPI implementation is GPU-aware
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    if (GMX_LIB_MPI && (GMX_GPU_CUDA || GMX_GPU_SYCL))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        // Allow overriding the detection for GPU-aware MPI
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        GpuAwareMpiStatus gpuAwareMpiStatus = checkMpiCudaAwareSupport();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        const bool        forceGpuAwareMpi  = gpuAwareMpiStatus == GpuAwareMpiStatus::Forced;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        const bool haveDetectedGpuAwareMpi  = gpuAwareMpiStatus == GpuAwareMpiStatus::Supported;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (getenv("GMX_FORCE_CUDA_AWARE_MPI") != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "GMX_FORCE_CUDA_AWARE_MPI environment variable is inactive. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "Please use GMX_FORCE_GPU_AWARE_MPI instead.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        devFlags.canUseGpuAwareMpi = haveDetectedGpuAwareMpi || forceGpuAwareMpi;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (getenv("GMX_ENABLE_DIRECT_GPU_COMM") != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            if (!haveDetectedGpuAwareMpi && forceGpuAwareMpi)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                // GPU-aware support not detected in MPI library but, user has forced it's use
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                "This run has forced use of 'GPU-aware MPI'. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                "However, GROMACS cannot determine if underlying MPI is GPU-aware. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                "Check the GROMACS install guide for recommendations for GPU-aware "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                "GMX_FORCE_GPU_AWARE_MPI environment variable.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            if (devFlags.canUseGpuAwareMpi)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                "GMX_ENABLE_DIRECT_GPU_COMM environment variable detected, "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                "enabling direct GPU communication using GPU-aware MPI.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                "GPU-aware MPI was not detected, will not use direct GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                "for GPU-aware support. If you are certain about GPU-aware support "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                "GMX_FORCE_GPU_AWARE_MPI environment variable.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        else if (haveDetectedGpuAwareMpi)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            // GPU-aware MPI was detected, let the user know that using it may improve performance
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "GPU-aware MPI detected, but by default GROMACS will not "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "make use the direct GPU communication capabilities of MPI. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "the GMX_ENABLE_DIRECT_GPU_COMM environment variable.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (getenv("GMX_FORCE_GPU_AWARE_MPI") != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            // Cannot force use of GPU-aware MPI in this build configuration
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "A CUDA or SYCL build with an external MPI library is required in "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "order to benefit from GMX_FORCE_GPU_AWARE_MPI. That environment "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    if (devFlags.enableGpuBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                        "This run uses the 'GPU buffer ops' feature, enabled by the "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                        "GMX_USE_GPU_BUFFER_OPS environment variable.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    // PME decomposition is supported only with CUDA-backend in mixed mode
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    // CUDA-backend also needs GPU-aware MPI support for decomposition to work
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    const bool pmeGpuDecompositionRequested =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            (pmeRunMode == PmeRunMode::GPU || pmeRunMode == PmeRunMode::Mixed)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    const bool pmeGpuDecompositionSupported =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            (devFlags.canUseGpuAwareMpi && (GMX_GPU_CUDA || GMX_GPU_SYCL)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:             && ((pmeRunMode == PmeRunMode::GPU && (GMX_USE_Heffte || GMX_USE_cuFFTMp))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    const bool forcePmeGpuDecomposition = getenv("GMX_GPU_PME_DECOMPOSITION") != nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    if (pmeGpuDecompositionSupported && pmeGpuDecompositionRequested)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        // PME decomposition is supported only when it is forced using GMX_GPU_PME_DECOMPOSITION
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (forcePmeGpuDecomposition)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "This run has requested the 'GPU PME decomposition' feature, enabled "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "by the GMX_GPU_PME_DECOMPOSITION environment variable. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                      "Multiple PME tasks were required to run on GPUs, "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                      "Use GMX_GPU_PME_DECOMPOSITION environment variable to enable it.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    if (!pmeGpuDecompositionSupported && pmeGpuDecompositionRequested)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (GMX_GPU_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                      "PME tasks were required to run on more than one CUDA-devices. To enable "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                      "use MPI with CUDA-aware support and build GROMACS with cuFFTMp support.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                    "PME tasks were required to run on GPUs, but that is not implemented with "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    devFlags.enableGpuPmeDecomposition =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            forcePmeGpuDecomposition && pmeGpuDecompositionRequested && pmeGpuDecompositionSupported;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                  bool                           makeGpuPairList,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                (makeGpuPairList ? ListSetupType::Gpu : ListSetupType::CpuSimdWhenSupported);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                fplog, cr, ir, nstlist_cmdline, &mtop, box, effectiveAtomDensity.value(), makeGpuPairList, cpuinfo);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:/*! \brief Return whether GPU acceleration of nonbondeds is supported with the given settings.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:static bool gpuAccelerationOfNonbondedIsUseful(const MDLogger&   mdlog,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    bool        gpuIsUseful = true;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        /* The GPU code does not support more than one energy group.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:         * If the user requested GPUs explicitly, a fatal error is given later.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        gpuIsUseful = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                    "Multiple energy groups is not implemented for GPUs, falling back to the CPU. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                    "For better performance, run on the GPU without energy groups and then do "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    /* There are resource handling issues in the GPU code paths with MTS on anything else than only
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        gpuIsUseful = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                "Multiple time stepping is only supported with GPUs when MTS is only applied to %s "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        gpuIsUseful = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        warning     = "TPI is not implemented for GPUs.";
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    if (!gpuIsUseful && issueWarning)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    return gpuIsUseful;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    else if (strncmp(optionString, "gpu", 3) == 0)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        returnValue = TaskTarget::Gpu;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        auto* nbnxn_gpu_timings =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                (nbv != nullptr && nbv->useGpu()) ? Nbnxm::gpu_get_timings(nbv->gpu_nbv) : nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        gmx_wallclock_gpu_pme_t pme_gpu_timings = {};
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (pme_gpu_task_enabled(pme))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            pme_gpu_get_timings(pme, &pme_gpu_timings);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                        nbnxn_gpu_timings,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                        &pme_gpu_timings);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    EmulateGpuNonbonded emulateGpuNonbonded =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            (getenv("GMX_EMULATE_GPU") != nullptr ? EmulateGpuNonbonded::Yes : EmulateGpuNonbonded::No);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    std::vector<int> userGpuTaskAssignment;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        userGpuTaskAssignment = parseUserTaskAssignmentString(hw_opt.userGpuTaskAssignment);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        bool useGpuForNonbonded = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        bool useGpuForPme       = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            // the number of GPUs to choose the number of ranks.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            auto canUseGpuForNonbonded = buildSupportsNonbondedOnGpu(nullptr);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            useGpuForNonbonded         = decideWhetherToUseGpusForNonbondedWithThreadMpi(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                    userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                    emulateGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                    canUseGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                    gpuAccelerationOfNonbondedIsUseful(mdlog, *inputrec, GMX_THREAD_MPI, doRerun),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            useGpuForPme = decideWhetherToUseGpusForPmeWithThreadMpi(useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                                     userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                useGpuForPme,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    // Note that when bonded interactions run on a GPU they always run
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuForNonbonded = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuForPme       = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuForBonded    = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuForUpdate    = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    bool gpusWereDetected   = hwinfo_->ngpu_compatible_tot > 0;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        // It's possible that there are different numbers of GPUs on
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        auto canUseGpuForNonbonded = buildSupportsNonbondedOnGpu(nullptr);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        useGpuForNonbonded         = decideWhetherToUseGpusForNonbonded(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                emulateGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                canUseGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                gpuAccelerationOfNonbondedIsUseful(mdlog, *inputrec, !GMX_THREAD_MPI, doRerun),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                gpusWereDetected);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        useGpuForPme    = decideWhetherToUseGpusForPme(useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                    userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                    gpusWereDetected);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        useGpuForBonded = decideWhetherToUseGpusForBonded(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                useGpuForNonbonded, useGpuForPme, bondedTarget, *inputrec, mtop, domdecOptions.numPmeRanks, gpusWereDetected);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    const PmeRunMode pmeRunMode = determinePmeRunMode(useGpuForPme, pmeFftTarget, *inputrec);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            mdlog, useGpuForNonbonded, pmeRunMode, cr->sizeOfDefaultCommunicator, domdecOptions.numPmeRanks);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                              updateTarget == TaskTarget::Gpu);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                || (!useGpuForNonbonded && usingFullElectrostatics(inputrec->coulombtype)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                          useGpuForNonbonded || (emulateGpuNonbonded == EmulateGpuNonbonded::Yes),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        useGpuForUpdate = decideWhetherToUseGpuForUpdate(useDomainDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                         useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                         gpusWereDetected,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    const bool canUseDirectGpuComm = decideWhetherDirectGpuCommunicationCanBeUsed(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuDirectHalo = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    if (useGpuForNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        // domdecOptions.numPmeRanks == -1 results in 0 separate PME ranks when useGpuForNonbonded is true.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        useGpuDirectHalo = decideWhetherToUseGpuForHalo(havePPDomainDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                        useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                        canUseDirectGpuComm,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        // The DD builder will disable useGpuDirectHalo if the Y or Z component of any domain is
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        // smaller than twice the communication distance, since GPU-direct communication presently
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        // perform well on multiple GPUs in any case, but it is important that our core functionality
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        // (in particular for testing) does not break depending on GPU direct communication being enabled.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                useGpuForPme,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                useGpuForUpdate,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                useGpuDirectHalo,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                devFlags.enableGpuPmeDecomposition);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    GpuTaskAssignments gpuTaskAssignments = GpuTaskAssignmentsBuilder::build(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            useGpuForPme,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    DeviceInformation* deviceInfo = gpuTaskAssignments.initDevice(&deviceId);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        // TODO Pass the GPU streams to ddBuilder to use in buffer
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    const bool useGpuPmeDecomposition = numPmeDomains.x * numPmeDomains.y > 1 && useGpuForPme;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    GMX_RELEASE_ASSERT(!useGpuPmeDecomposition || devFlags.enableGpuPmeDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                       "GPU PME decomposition works only in the cases where it is supported");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuForBonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuForUpdate,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuDirectHalo,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                              canUseDirectGpuComm,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuPmeDecomposition);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    if (runScheduleWork.simulationWork.useGpuDirectCommunication && GMX_GPU_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        // Don't enable event counting with GPU Direct comm, see #3988.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        gmx::internal::disableCudaEventConsumptionCounting();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    if (isSimulationMainRank && GMX_GPU_SYCL)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        bool                      haveAnyGpuWork = simWorkload.useGpuPme || simWorkload.useGpuBonded
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                              || simWorkload.useGpuNonbonded || simWorkload.useGpuUpdate;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (haveAnyGpuWork)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            "\nNOTE: SYCL GPU support in GROMACS, and the compilers, libraries,\n"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    gpuTaskAssignments.reportGpuUsage(mdlog, printHostName, pmeRunMode, runScheduleWork.simulationWork);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        const bool useGpuTiming = decideGpuTimingsUsage();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                *deviceInfo, runScheduleWork.simulationWork, useGpuTiming);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        gpuTaskAssignments.logPerformanceHints(mdlog, numAvailableDevices);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    check_resource_division_efficiency(hwinfo_, gpuTaskAssignments.thisRankHasAnyGpuTask(), cr, mdlog);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    // Enable Peer access between GPUs where available
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    // any of the GPU communication features are active.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        && (runScheduleWork.simulationWork.useGpuHaloExchange
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            || runScheduleWork.simulationWork.useGpuPmePpCommunication))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        setupGpuDevicePeerAccess(gpuTaskAssignments.deviceIdsAssigned(), mdlog);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    const bool thisRankHasPmeGpuTask = gpuTaskAssignments.thisRankHasPmeGpuTask();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (runScheduleWork.simulationWork.useGpuPmePpCommunication && !thisRankHasDuty(cr, DUTY_PME))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                    "GPU device stream manager should be valid in order to use PME-PP direct "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                    "GPU PP-PME stream should be valid in order to use GPU PME-PP direct "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            fr->pmePpCommGpu = std::make_unique<gmx::PmePpCommGpu>(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                runScheduleWork.simulationWork.useGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        // TODO: Move the logic below to a GPU bonded builder
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (runScheduleWork.simulationWork.useGpuBonded)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                               "GPU device stream manager should be valid in order to use GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            fr->listedForcesGpu = std::make_unique<ListedForcesGpu>(mtop.ffparams,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        mdAtoms = makeMDAtoms(fplog, mtop, *inputrec, thisRankHasPmeGpuTask);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (globalState && thisRankHasPmeGpuTask)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            // PME on GPU without DD or on a separate PME rank, and because the local state pointer
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    PmeGpuProgramStorage pmeGpuProgram;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    if (thisRankHasPmeGpuTask)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                "GPU device stream manager should be initialized in order to use GPU for PME.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                           "GPU device should be initialized in order to use GPU for PME.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        pmeGpuProgram = buildPmeGpuProgram(deviceStreamManager->context());
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                GMX_RELEASE_ASSERT(!runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                   "Device stream manager should be valid in order to use GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                        !runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                        "GPU PME stream should be valid in order to use GPU version of PME.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                const DeviceContext* deviceContext = runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                        runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                                       pmeGpuProgram.get(),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (runScheduleWork.simulationWork.useGpuFBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            fr->gpuForceReduction[gmx::AtomLocality::Local] = std::make_unique<gmx::GpuForceReduction>(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            fr->gpuForceReduction[gmx::AtomLocality::NonLocal] = std::make_unique<gmx::GpuForceReduction>(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            if (runScheduleWork.simulationWork.useMdGpuGraph)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                        std::make_unique<gmx::MdGpuGraph>(*fr->deviceStreamManager,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                        std::make_unique<gmx::MdGpuGraph>(*fr->deviceStreamManager,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        std::unique_ptr<gmx::StatePropagatorDataGpu> stateGpu;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (gpusWereDetected && gmx::needStateGpu(runScheduleWork.simulationWork))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            GpuApiCallBehavior transferKind =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            ? GpuApiCallBehavior::Async
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                            : GpuApiCallBehavior::Sync;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                               "GPU device stream manager should be initialized to use GPU.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            stateGpu = std::make_unique<gmx::StatePropagatorDataGpu>(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                    *deviceStreamManager, transferKind, pme_gpu_get_block_size(fr->pmedata), wcycle.get());
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            fr->stateGpu = stateGpu.get();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:          /* set GPU device id */
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:             plumed_cmd(plumedmain,"setGpuDeviceId", &deviceId);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:          if(useGpuForUpdate) {
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                        "This simulation is resident on GPU (-update gpu)\n"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        if (fr->pmePpCommGpu)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            // destroy object since it is no longer required. (This needs to be done while the GPU context still exists.)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            fr->pmePpCommGpu.reset();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:                    runScheduleWork.simulationWork.useGpuPmePpCommunication,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    // before we destroy the GPU context(s)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    // Pinned buffers are associated with contexts in CUDA.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    // As soon as we destroy GPU contexts after mdrunner() exits, these lines should go.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    mdModules_.reset(nullptr); // destruct force providers here as they might also use the GPU
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    fr.reset(nullptr);         // destruct forcerec before gpu
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        /* stop the GPU profiler (only CUDA) */
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        stopGpuProfiler();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:     * destroying the CUDA context as some tMPI ranks may be sharing
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:     * GPU and context.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:     * This is not a concern in OpenCL where we use one context per rank.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:     * Note: it is safe to not call the barrier on the ranks which do not use GPU,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:     * Note that this function needs to be called even if GPUs are not used
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:     * in this run because the PME ranks have no knowledge of whether GPUs
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:     * that it's not needed anymore (with a shared GPU run).
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    const bool haveDetectedOrForcedCudaAwareMpi =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:            (gmx::checkMpiCudaAwareSupport() == gmx::GpuAwareMpiStatus::Supported
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:             || gmx::checkMpiCudaAwareSupport() == gmx::GpuAwareMpiStatus::Forced);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:    if (!haveDetectedOrForcedCudaAwareMpi)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        // Don't reset GPU in case of GPU-AWARE MPI
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp:        // UCX creates GPU buffers which are cleaned-up as part of MPI_Finalize()
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:    // which compatible GPUs are availble for use, or to select a GPU
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:        hw_opt.userGpuTaskAssignment = userGpuTaskAssignment;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:        const char* env = getenv("GMX_GPU_ID");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:                gmx_fatal(FARGS, "GMX_GPU_ID and -gpu_id can not be used at the same time");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:        env = getenv("GMX_GPUTASKS");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:            if (!hw_opt.userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:                gmx_fatal(FARGS, "GMX_GPUTASKS and -gputasks can not be used at the same time");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:            hw_opt.userGpuTaskAssignment = env;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:        if (!hw_opt.devicesSelectedByUser.empty() && !hw_opt.userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:            gmx_fatal(FARGS, "-gpu_id and -gputasks cannot be used at the same time");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:    // which compatible GPUs are availble for use, or to select a GPU
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:        hw_opt.userGpuTaskAssignment = userGpuTaskAssignment;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:        const char* env = getenv("GMX_GPU_ID");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:                gmx_fatal(FARGS, "GMX_GPU_ID and -gpu_id can not be used at the same time");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:        env = getenv("GMX_GPUTASKS");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:            if (!hw_opt.userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:                gmx_fatal(FARGS, "GMX_GPUTASKS and -gputasks can not be used at the same time");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:            hw_opt.userGpuTaskAssignment = env;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:        if (!hw_opt.devicesSelectedByUser.empty() && !hw_opt.userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:            gmx_fatal(FARGS, "-gpu_id and -gputasks cannot be used at the same time");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* nbpu_opt_choices[5]    = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* pme_opt_choices[5]     = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* pme_fft_opt_choices[5] = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* bonded_opt_choices[5]  = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* update_opt_choices[5]  = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* userGpuTaskAssignment  = "";
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:        { "-gpu_id",
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:          "List of unique GPU device IDs available to use" },
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:        { "-gputasks",
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:          { &userGpuTaskAssignment },
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:          "List of GPU device IDs, mapping each task on a node to a device. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:          "Optimize PME load between PP/PME ranks or GPU/CPU" },
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* nbpu_opt_choices[5]    = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* pme_opt_choices[5]     = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* pme_fft_opt_choices[5] = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* bonded_opt_choices[5]  = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* update_opt_choices[5]  = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* userGpuTaskAssignment  = "";
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:        { "-gpu_id",
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:          "List of unique GPU device IDs available to use" },
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:        { "-gputasks",
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:          { &userGpuTaskAssignment },
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:          "List of GPU device IDs, mapping each task on a node to a device. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:          "Optimize PME load between PP/PME ranks or GPU/CPU" },
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/ewald/pme_gpu_program.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/ewald/pme_pp_comm_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/gpu_utils/device_stream_manager.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/gpu_utils/gpueventsynchronizer_helpers.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/mdlib/gpuforcereduction.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/mdlib/mdgraph_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/taskassignment/decidegpuusage.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/taskassignment/usergpuids.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/timing/gpu_timing.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed: * the GPU communication flags are set to false in non-tMPI and non-CUDA builds.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed: * \param[in]  useGpuForNonbonded   True if the nonbonded task is offloaded in this run.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                         const bool           useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    devFlags.enableGpuBufferOps = (GMX_GPU_CUDA || GMX_GPU_SYCL) && useGpuForNonbonded
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                  && (getenv("GMX_USE_GPU_BUFFER_OPS") != nullptr);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (getenv("GMX_CUDA_GRAPH") != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (GMX_HAVE_CUDA_GRAPH_SUPPORT)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            devFlags.enableCudaGraphs = true;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "GMX_CUDA_GRAPH environment variable is detected. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "The experimental CUDA Graphs feature will be used if run conditions "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            devFlags.enableCudaGraphs = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "GMX_CUDA_GRAPH environment variable is detected, "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "but the CUDA version in use is below the minumum requirement (11.1). "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "CUDA Graphs will be disabled.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Flag use to enable GPU-aware MPI depenendent features such PME GPU decomposition
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // GPU-aware MPI is marked available if it has been detected by GROMACS or detection fails but
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    devFlags.canUseGpuAwareMpi = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Direct GPU comm path is being used with GPU-aware MPI
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // make sure underlying MPI implementation is GPU-aware
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (GMX_LIB_MPI && (GMX_GPU_CUDA || GMX_GPU_SYCL))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // Allow overriding the detection for GPU-aware MPI
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        GpuAwareMpiStatus gpuAwareMpiStatus = checkMpiCudaAwareSupport();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        const bool        forceGpuAwareMpi  = gpuAwareMpiStatus == GpuAwareMpiStatus::Forced;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        const bool haveDetectedGpuAwareMpi  = gpuAwareMpiStatus == GpuAwareMpiStatus::Supported;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (getenv("GMX_FORCE_CUDA_AWARE_MPI") != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "GMX_FORCE_CUDA_AWARE_MPI environment variable is inactive. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "Please use GMX_FORCE_GPU_AWARE_MPI instead.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        devFlags.canUseGpuAwareMpi = haveDetectedGpuAwareMpi || forceGpuAwareMpi;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (getenv("GMX_ENABLE_DIRECT_GPU_COMM") != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            if (!haveDetectedGpuAwareMpi && forceGpuAwareMpi)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                // GPU-aware support not detected in MPI library but, user has forced it's use
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "This run has forced use of 'GPU-aware MPI'. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "However, GROMACS cannot determine if underlying MPI is GPU-aware. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "Check the GROMACS install guide for recommendations for GPU-aware "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "GMX_FORCE_GPU_AWARE_MPI environment variable.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            if (devFlags.canUseGpuAwareMpi)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "GMX_ENABLE_DIRECT_GPU_COMM environment variable detected, "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "enabling direct GPU communication using GPU-aware MPI.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "GPU-aware MPI was not detected, will not use direct GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "for GPU-aware support. If you are certain about GPU-aware support "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "GMX_FORCE_GPU_AWARE_MPI environment variable.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        else if (haveDetectedGpuAwareMpi)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // GPU-aware MPI was detected, let the user know that using it may improve performance
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "GPU-aware MPI detected, but by default GROMACS will not "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "make use the direct GPU communication capabilities of MPI. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "the GMX_ENABLE_DIRECT_GPU_COMM environment variable.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (getenv("GMX_FORCE_GPU_AWARE_MPI") != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // Cannot force use of GPU-aware MPI in this build configuration
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "A CUDA or SYCL build with an external MPI library is required in "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "order to benefit from GMX_FORCE_GPU_AWARE_MPI. That environment "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (devFlags.enableGpuBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        "This run uses the 'GPU buffer ops' feature, enabled by the "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        "GMX_USE_GPU_BUFFER_OPS environment variable.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // PME decomposition is supported only with CUDA-backend in mixed mode
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // CUDA-backend also needs GPU-aware MPI support for decomposition to work
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool pmeGpuDecompositionRequested =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            (pmeRunMode == PmeRunMode::GPU || pmeRunMode == PmeRunMode::Mixed)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool pmeGpuDecompositionSupported =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            (devFlags.canUseGpuAwareMpi && (GMX_GPU_CUDA || GMX_GPU_SYCL)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:             && ((pmeRunMode == PmeRunMode::GPU && (GMX_USE_Heffte || GMX_USE_cuFFTMp))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool forcePmeGpuDecomposition = getenv("GMX_GPU_PME_DECOMPOSITION") != nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (pmeGpuDecompositionSupported && pmeGpuDecompositionRequested)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // PME decomposition is supported only when it is forced using GMX_GPU_PME_DECOMPOSITION
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (forcePmeGpuDecomposition)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "This run has requested the 'GPU PME decomposition' feature, enabled "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "by the GMX_GPU_PME_DECOMPOSITION environment variable. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                      "Multiple PME tasks were required to run on GPUs, "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                      "Use GMX_GPU_PME_DECOMPOSITION environment variable to enable it.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (!pmeGpuDecompositionSupported && pmeGpuDecompositionRequested)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (GMX_GPU_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                      "PME tasks were required to run on more than one CUDA-devices. To enable "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                      "use MPI with CUDA-aware support and build GROMACS with cuFFTMp support.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "PME tasks were required to run on GPUs, but that is not implemented with "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    devFlags.enableGpuPmeDecomposition =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            forcePmeGpuDecomposition && pmeGpuDecompositionRequested && pmeGpuDecompositionSupported;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                  bool                           makeGpuPairList,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                (makeGpuPairList ? ListSetupType::Gpu : ListSetupType::CpuSimdWhenSupported);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                fplog, cr, ir, nstlist_cmdline, &mtop, box, effectiveAtomDensity.value(), makeGpuPairList, cpuinfo);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:/*! \brief Return whether GPU acceleration of nonbondeds is supported with the given settings.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:static bool gpuAccelerationOfNonbondedIsUseful(const MDLogger&   mdlog,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool        gpuIsUseful = true;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        /* The GPU code does not support more than one energy group.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:         * If the user requested GPUs explicitly, a fatal error is given later.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gpuIsUseful = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "Multiple energy groups is not implemented for GPUs, falling back to the CPU. "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "For better performance, run on the GPU without energy groups and then do "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    /* There are resource handling issues in the GPU code paths with MTS on anything else than only
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gpuIsUseful = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                "Multiple time stepping is only supported with GPUs when MTS is only applied to %s "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gpuIsUseful = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        warning     = "TPI is not implemented for GPUs.";
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (!gpuIsUseful && issueWarning)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    return gpuIsUseful;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    else if (strncmp(optionString, "gpu", 3) == 0)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        returnValue = TaskTarget::Gpu;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        auto* nbnxn_gpu_timings =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                (nbv != nullptr && nbv->useGpu()) ? Nbnxm::gpu_get_timings(nbv->gpu_nbv) : nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gmx_wallclock_gpu_pme_t pme_gpu_timings = {};
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (pme_gpu_task_enabled(pme))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            pme_gpu_get_timings(pme, &pme_gpu_timings);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        nbnxn_gpu_timings,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        &pme_gpu_timings);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    EmulateGpuNonbonded emulateGpuNonbonded =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            (getenv("GMX_EMULATE_GPU") != nullptr ? EmulateGpuNonbonded::Yes : EmulateGpuNonbonded::No);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    std::vector<int> userGpuTaskAssignment;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        userGpuTaskAssignment = parseUserTaskAssignmentString(hw_opt.userGpuTaskAssignment);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        bool useGpuForNonbonded = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        bool useGpuForPme       = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // the number of GPUs to choose the number of ranks.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            auto canUseGpuForNonbonded = buildSupportsNonbondedOnGpu(nullptr);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            useGpuForNonbonded         = decideWhetherToUseGpusForNonbondedWithThreadMpi(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    emulateGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    canUseGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    gpuAccelerationOfNonbondedIsUseful(mdlog, *inputrec, GMX_THREAD_MPI, doRerun),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            useGpuForPme = decideWhetherToUseGpusForPmeWithThreadMpi(useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                                     userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                useGpuForPme,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Note that when bonded interactions run on a GPU they always run
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuForNonbonded = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuForPme       = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuForBonded    = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuForUpdate    = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool gpusWereDetected   = hwinfo_->ngpu_compatible_tot > 0;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // It's possible that there are different numbers of GPUs on
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        auto canUseGpuForNonbonded = buildSupportsNonbondedOnGpu(nullptr);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuForNonbonded         = decideWhetherToUseGpusForNonbonded(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                emulateGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                canUseGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                gpuAccelerationOfNonbondedIsUseful(mdlog, *inputrec, !GMX_THREAD_MPI, doRerun),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                gpusWereDetected);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuForPme    = decideWhetherToUseGpusForPme(useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                    userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                    gpusWereDetected);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuForBonded = decideWhetherToUseGpusForBonded(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuForNonbonded, useGpuForPme, bondedTarget, *inputrec, mtop, domdecOptions.numPmeRanks, gpusWereDetected);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const PmeRunMode pmeRunMode = determinePmeRunMode(useGpuForPme, pmeFftTarget, *inputrec);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            mdlog, useGpuForNonbonded, pmeRunMode, cr->sizeOfDefaultCommunicator, domdecOptions.numPmeRanks);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              updateTarget == TaskTarget::Gpu);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                || (!useGpuForNonbonded && usingFullElectrostatics(inputrec->coulombtype)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                          useGpuForNonbonded || (emulateGpuNonbonded == EmulateGpuNonbonded::Yes),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuForUpdate = decideWhetherToUseGpuForUpdate(useDomainDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                         useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                         gpusWereDetected,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool canUseDirectGpuComm = decideWhetherDirectGpuCommunicationCanBeUsed(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuDirectHalo = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (useGpuForNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // domdecOptions.numPmeRanks == -1 results in 0 separate PME ranks when useGpuForNonbonded is true.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuDirectHalo = decideWhetherToUseGpuForHalo(havePPDomainDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                        useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                        canUseDirectGpuComm,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // The DD builder will disable useGpuDirectHalo if the Y or Z component of any domain is
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // smaller than twice the communication distance, since GPU-direct communication presently
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // perform well on multiple GPUs in any case, but it is important that our core functionality
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // (in particular for testing) does not break depending on GPU direct communication being enabled.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuForPme,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuForUpdate,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuDirectHalo,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                devFlags.enableGpuPmeDecomposition);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    GpuTaskAssignments gpuTaskAssignments = GpuTaskAssignmentsBuilder::build(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            useGpuForPme,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    DeviceInformation* deviceInfo = gpuTaskAssignments.initDevice(&deviceId);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // TODO Pass the GPU streams to ddBuilder to use in buffer
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool useGpuPmeDecomposition = numPmeDomains.x * numPmeDomains.y > 1 && useGpuForPme;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    GMX_RELEASE_ASSERT(!useGpuPmeDecomposition || devFlags.enableGpuPmeDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                       "GPU PME decomposition works only in the cases where it is supported");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuForBonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuForUpdate,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuDirectHalo,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              canUseDirectGpuComm,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuPmeDecomposition);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (runScheduleWork.simulationWork.useGpuDirectCommunication && GMX_GPU_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // Don't enable event counting with GPU Direct comm, see #3988.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gmx::internal::disableCudaEventConsumptionCounting();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (isSimulationMainRank && GMX_GPU_SYCL)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        bool                      haveAnyGpuWork = simWorkload.useGpuPme || simWorkload.useGpuBonded
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                              || simWorkload.useGpuNonbonded || simWorkload.useGpuUpdate;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (haveAnyGpuWork)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "\nNOTE: SYCL GPU support in GROMACS, and the compilers, libraries,\n"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    gpuTaskAssignments.reportGpuUsage(mdlog, printHostName, pmeRunMode, runScheduleWork.simulationWork);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        const bool useGpuTiming = decideGpuTimingsUsage();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                *deviceInfo, runScheduleWork.simulationWork, useGpuTiming);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gpuTaskAssignments.logPerformanceHints(mdlog, numAvailableDevices);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    check_resource_division_efficiency(hwinfo_, gpuTaskAssignments.thisRankHasAnyGpuTask(), cr, mdlog);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Enable Peer access between GPUs where available
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // any of the GPU communication features are active.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        && (runScheduleWork.simulationWork.useGpuHaloExchange
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            || runScheduleWork.simulationWork.useGpuPmePpCommunication))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        setupGpuDevicePeerAccess(gpuTaskAssignments.deviceIdsAssigned(), mdlog);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool thisRankHasPmeGpuTask = gpuTaskAssignments.thisRankHasPmeGpuTask();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (runScheduleWork.simulationWork.useGpuPmePpCommunication && !thisRankHasDuty(cr, DUTY_PME))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "GPU device stream manager should be valid in order to use PME-PP direct "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "GPU PP-PME stream should be valid in order to use GPU PME-PP direct "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->pmePpCommGpu = std::make_unique<gmx::PmePpCommGpu>(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                runScheduleWork.simulationWork.useGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // TODO: Move the logic below to a GPU bonded builder
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (runScheduleWork.simulationWork.useGpuBonded)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                               "GPU device stream manager should be valid in order to use GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->listedForcesGpu = std::make_unique<ListedForcesGpu>(mtop.ffparams,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        mdAtoms = makeMDAtoms(fplog, mtop, *inputrec, thisRankHasPmeGpuTask);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (globalState && thisRankHasPmeGpuTask)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // PME on GPU without DD or on a separate PME rank, and because the local state pointer
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    PmeGpuProgramStorage pmeGpuProgram;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (thisRankHasPmeGpuTask)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                "GPU device stream manager should be initialized in order to use GPU for PME.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                           "GPU device should be initialized in order to use GPU for PME.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        pmeGpuProgram = buildPmeGpuProgram(deviceStreamManager->context());
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                GMX_RELEASE_ASSERT(!runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                   "Device stream manager should be valid in order to use GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        !runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        "GPU PME stream should be valid in order to use GPU version of PME.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                const DeviceContext* deviceContext = runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                       pmeGpuProgram.get(),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (runScheduleWork.simulationWork.useGpuFBufferOps)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->gpuForceReduction[gmx::AtomLocality::Local] = std::make_unique<gmx::GpuForceReduction>(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->gpuForceReduction[gmx::AtomLocality::NonLocal] = std::make_unique<gmx::GpuForceReduction>(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            if (runScheduleWork.simulationWork.useMdGpuGraph)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        std::make_unique<gmx::MdGpuGraph>(*fr->deviceStreamManager,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        std::make_unique<gmx::MdGpuGraph>(*fr->deviceStreamManager,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        std::unique_ptr<gmx::StatePropagatorDataGpu> stateGpu;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (gpusWereDetected && gmx::needStateGpu(runScheduleWork.simulationWork))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            GpuApiCallBehavior transferKind =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            ? GpuApiCallBehavior::Async
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            : GpuApiCallBehavior::Sync;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                               "GPU device stream manager should be initialized to use GPU.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            stateGpu = std::make_unique<gmx::StatePropagatorDataGpu>(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    *deviceStreamManager, transferKind, pme_gpu_get_block_size(fr->pmedata), wcycle.get());
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->stateGpu = stateGpu.get();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (fr->pmePpCommGpu)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // destroy object since it is no longer required. (This needs to be done while the GPU context still exists.)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->pmePpCommGpu.reset();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    runScheduleWork.simulationWork.useGpuPmePpCommunication,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // before we destroy the GPU context(s)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Pinned buffers are associated with contexts in CUDA.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // As soon as we destroy GPU contexts after mdrunner() exits, these lines should go.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    mdModules_.reset(nullptr); // destruct force providers here as they might also use the GPU
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    fr.reset(nullptr);         // destruct forcerec before gpu
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        /* stop the GPU profiler (only CUDA) */
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        stopGpuProfiler();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * destroying the CUDA context as some tMPI ranks may be sharing
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * GPU and context.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * This is not a concern in OpenCL where we use one context per rank.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * Note: it is safe to not call the barrier on the ranks which do not use GPU,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * Note that this function needs to be called even if GPUs are not used
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * in this run because the PME ranks have no knowledge of whether GPUs
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * that it's not needed anymore (with a shared GPU run).
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool haveDetectedOrForcedCudaAwareMpi =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            (gmx::checkMpiCudaAwareSupport() == gmx::GpuAwareMpiStatus::Supported
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:             || gmx::checkMpiCudaAwareSupport() == gmx::GpuAwareMpiStatus::Forced);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (!haveDetectedOrForcedCudaAwareMpi)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // Don't reset GPU in case of GPU-AWARE MPI
patches/gromacs-2023.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // UCX creates GPU buffers which are cleaned-up as part of MPI_Finalize()
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/gpu_utils/device_stream_manager.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/mdlib/mdgraph_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/mdlib/update_constrain_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    /* PME load balancing data for GPU kernels */
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    const bool  useGpuForPme       = simulationWork.useGpuPme;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    const bool  useGpuForNonbonded = simulationWork.useGpuNonbonded;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    const bool  useGpuForUpdate    = simulationWork.useGpuUpdate;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                                 useGpuForPme);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                   (simulationWork.useGpuFBufferOps || useGpuForUpdate) ? PinningPolicy::PinnedIfSupported
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    std::unique_ptr<UpdateConstrainGpu> integrator;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    StatePropagatorDataGpu* stateGpu = fr->stateGpu;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    if (useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "groups if using GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "SHAKE is not supported with GPU update.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        GMX_RELEASE_ASSERT(useGpuForPme || (useGpuForNonbonded && simulationWork.useGpuXBufferOps),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "the GPU to use GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Only the md integrator is supported with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                "Nose-Hoover temperature coupling is not supported with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                "with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Virtual sites are not supported with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Essential dynamics is not supported with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Constraints pulling is not supported with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Orientation restraints are not supported with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                "Free energy perturbation of masses and constraints are not supported with the GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    .appendText("Updating coordinates and applying constraints on the GPU.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            GMX_LOG(mdlog.info).asParagraph().appendText("Updating coordinates on the GPU.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Device stream manager should be initialized in order to use GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                "Update stream should be initialized in order to use GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        integrator = std::make_unique<UpdateConstrainGpu>(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        stateGpu->setXUpdatedOnDeviceEvent(integrator->xUpdatedOnDeviceEvent());
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    if (useGpuForPme || simulationWork.useGpuXBufferOps || useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    if (useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:     * Disable PME tuning with GPU PME decomposition */
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                && ir->cutoff_scheme != CutoffScheme::Group && !simulationWork.useGpuPmeDecomposition);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                &pme_loadbal, cr, mdlog, *ir, state->box, *fr->ic, *fr->nbv, fr->pmedata, fr->nbv->useGpu());
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    bool usedMdGpuGraphLastStep = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate && !bFirstStep)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            /* PME grid + cut-off optimization with GPUs or PME nodes */
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           simulationWork.useGpuPmePpCommunication);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        // On search steps, when doing the update on the GPU, copy
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (useGpuForUpdate && bNS && !bFirstStep && !bExchanged)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (usedMdGpuGraphLastStep)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                // Wait on coordinates produced from GPU graph
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->waitCoordinatesUpdatedOnDevice();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->copyVelocitiesFromGpu(state->v, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            // the GPU Update object should be informed
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate && (bMainState || bExchanged))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        // Allocate or re-size GPU halo exchange object, if necessary
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (bNS && simulationWork.havePpDomainDecomposition && simulationWork.useGpuHaloExchange)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                               "GPU device manager has to be initialized to use GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            constructGpuHaloExchange(*cr, *fr->deviceStreamManager, wcycle);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        MdGpuGraph* mdGraph = simulationWork.useMdGpuGraph ? fr->mdGraph[step % 2].get() : nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (simulationWork.useMdGpuGraph)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                mdGraph->setUsedGraphLastStep(usedMdGpuGraphLastStep);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                bool canUseMdGpuGraphThisStep =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                if (mdGraph->captureThisStep(canUseMdGpuGraphThisStep))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    mdGraph->startRecord(stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (!simulationWork.useMdGpuGraph || mdGraph->graphIsCapturingThisStep()
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            // Copy coordinate from the GPU for the output/checkpointing if the update is offloaded
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate && !bNS && !runScheduleWork->domainWork.haveCpuLocalForceWork
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate && !bNS
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->copyVelocitiesFromGpu(state->v, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            // Copy forces for the output if the forces were reduced on the GPU (not the case on virial steps)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            // and update is offloaded hence forces are kept on the GPU for update and have not been
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            //       when the forces are ready on the GPU -- the same synchronizer should be used as the one
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            //       prior to GPU update.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (runScheduleWork->stepWork.useGpuFBufferOps
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                && (simulationWork.useGpuUpdate && !vsite) && do_per_step(step, ir->nstfout))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->copyForcesFromGpu(f.view().force(), AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (!useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                GMX_ASSERT(!useGpuForUpdate, "GPU update is not supported with VVAK integrator.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                if (useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        integrator->set(stateGpu->getCoordinates(),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                                        stateGpu->getVelocities(),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                                        stateGpu->getForces(),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        // Copy data to the GPU after buffers might have been reinitialized
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->copyVelocitiesToGpu(state->v, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    // Copy x to the GPU unless we have already transferred in do_force().
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    // We transfer in do_force() if a GPU force task requires x (PME or x buffer ops).
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    if (!(runScheduleWork->stepWork.haveGpuPmeOnThisRank
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                          || runScheduleWork->stepWork.useGpuXBufferOps))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->copyCoordinatesToGpu(state->x, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->consumeCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    if ((simulationWork.useGpuPme && simulationWork.useCpuPmePpCommunication)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        || (!runScheduleWork->stepWork.useGpuFBufferOps))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        // rest of the forces computed on the GPU, so the final forces have to be
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        // copied back to the GPU. Or the buffer ops were not offloaded this step,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->copyForcesToGpu(f.view().force(), AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    integrator->integrate(stateGpu->getLocalForcesReadyOnDeviceEvent(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (simulationWork.useMdGpuGraph)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            GMX_ASSERT((mdGraph != nullptr), "MD GPU graph does not exist.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                // update): with PME tuning, since the GPU kernels
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            usedMdGpuGraphLastStep = mdGraph->useGraphThisStep();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->copyVelocitiesFromGpu(state->v, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    if (useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->resetCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->copyCoordinatesToGpu(state->x, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->waitCoordinatesCopiedToDevice(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                            stateGpu->copyVelocitiesToGpu(state->v, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        const bool scaleCoordinates = !useGpuForUpdate || bDoReplEx;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (useGpuForUpdate
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        // any run that uses GPUs must be at least offloading nonbondeds
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        const bool usingGpu = simulationWork.useGpuNonbonded;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (usingGpu)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            // ensure that GPU errors do not propagate between MD steps
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        pme_loadbal_done(pme_loadbal, fplog, mdlog, fr->nbv->useGpu());
patches/gromacs-2023.5.diff/src/gromacs/mdrun/rerun.cpp.preplumed:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/rerun.cpp.preplumed:                                 runScheduleWork->simulationWork.useGpuPme);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/rerun.cpp:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/rerun.cpp:                                 runScheduleWork->simulationWork.useGpuPme);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/gpu_utils/device_stream_manager.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/mdlib/mdgraph_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/mdlib/update_constrain_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:    /* PME load balancing data for GPU kernels */
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:    const bool  useGpuForPme       = simulationWork.useGpuPme;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:    const bool  useGpuForNonbonded = simulationWork.useGpuNonbonded;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:    const bool  useGpuForUpdate    = simulationWork.useGpuUpdate;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                                 useGpuForPme);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                   (simulationWork.useGpuFBufferOps || useGpuForUpdate) ? PinningPolicy::PinnedIfSupported
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:    std::unique_ptr<UpdateConstrainGpu> integrator;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:    StatePropagatorDataGpu* stateGpu = fr->stateGpu;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:    if (useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                           "groups if using GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                           "SHAKE is not supported with GPU update.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        GMX_RELEASE_ASSERT(useGpuForPme || (useGpuForNonbonded && simulationWork.useGpuXBufferOps),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                           "the GPU to use GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                           "Only the md integrator is supported with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                "Nose-Hoover temperature coupling is not supported with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                "with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                           "Virtual sites are not supported with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                           "Essential dynamics is not supported with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                           "Constraints pulling is not supported with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                           "Orientation restraints are not supported with the GPU update.\n");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                "Free energy perturbation of masses and constraints are not supported with the GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                    .appendText("Updating coordinates and applying constraints on the GPU.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            GMX_LOG(mdlog.info).asParagraph().appendText("Updating coordinates on the GPU.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                           "Device stream manager should be initialized in order to use GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                "Update stream should be initialized in order to use GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        integrator = std::make_unique<UpdateConstrainGpu>(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        stateGpu->setXUpdatedOnDeviceEvent(integrator->xUpdatedOnDeviceEvent());
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:    if (useGpuForPme || simulationWork.useGpuXBufferOps || useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:    if (useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:     * Disable PME tuning with GPU PME decomposition */
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                && ir->cutoff_scheme != CutoffScheme::Group && !simulationWork.useGpuPmeDecomposition);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                &pme_loadbal, cr, mdlog, *ir, state->box, *fr->ic, *fr->nbv, fr->pmedata, fr->nbv->useGpu());
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:    bool usedMdGpuGraphLastStep = false;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate && !bFirstStep)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            /* PME grid + cut-off optimization with GPUs or PME nodes */
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                           simulationWork.useGpuPmePpCommunication);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        // On search steps, when doing the update on the GPU, copy
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        if (useGpuForUpdate && bNS && !bFirstStep && !bExchanged)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            if (usedMdGpuGraphLastStep)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                // Wait on coordinates produced from GPU graph
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                stateGpu->waitCoordinatesUpdatedOnDevice();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->copyVelocitiesFromGpu(state->v, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            // the GPU Update object should be informed
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate && (bMainState || bExchanged))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        // Allocate or re-size GPU halo exchange object, if necessary
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        if (bNS && simulationWork.havePpDomainDecomposition && simulationWork.useGpuHaloExchange)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                               "GPU device manager has to be initialized to use GPU "
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            constructGpuHaloExchange(*cr, *fr->deviceStreamManager, wcycle);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        MdGpuGraph* mdGraph = simulationWork.useMdGpuGraph ? fr->mdGraph[step % 2].get() : nullptr;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        if (simulationWork.useMdGpuGraph)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                mdGraph->setUsedGraphLastStep(usedMdGpuGraphLastStep);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                bool canUseMdGpuGraphThisStep =
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                if (mdGraph->captureThisStep(canUseMdGpuGraphThisStep))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                    mdGraph->startRecord(stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        if (!simulationWork.useMdGpuGraph || mdGraph->graphIsCapturingThisStep()
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            // Copy coordinate from the GPU for the output/checkpointing if the update is offloaded
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate && !bNS && !runScheduleWork->domainWork.haveCpuLocalForceWork
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate && !bNS
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                stateGpu->copyVelocitiesFromGpu(state->v, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            // Copy forces for the output if the forces were reduced on the GPU (not the case on virial steps)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            // and update is offloaded hence forces are kept on the GPU for update and have not been
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            //       when the forces are ready on the GPU -- the same synchronizer should be used as the one
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            //       prior to GPU update.
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            if (runScheduleWork->stepWork.useGpuFBufferOps
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                && (simulationWork.useGpuUpdate && !vsite) && do_per_step(step, ir->nstfout))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                stateGpu->copyForcesFromGpu(f.view().force(), AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            if (!useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                GMX_ASSERT(!useGpuForUpdate, "GPU update is not supported with VVAK integrator.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                if (useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                        integrator->set(stateGpu->getCoordinates(),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                                        stateGpu->getVelocities(),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                                        stateGpu->getForces(),
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                        // Copy data to the GPU after buffers might have been reinitialized
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->copyVelocitiesToGpu(state->v, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                    // Copy x to the GPU unless we have already transferred in do_force().
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                    // We transfer in do_force() if a GPU force task requires x (PME or x buffer ops).
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                    if (!(runScheduleWork->stepWork.haveGpuPmeOnThisRank
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                          || runScheduleWork->stepWork.useGpuXBufferOps))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->copyCoordinatesToGpu(state->x, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->consumeCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                    if ((simulationWork.useGpuPme && simulationWork.useCpuPmePpCommunication)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                        || (!runScheduleWork->stepWork.useGpuFBufferOps))
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                        // rest of the forces computed on the GPU, so the final forces have to be
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                        // copied back to the GPU. Or the buffer ops were not offloaded this step,
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->copyForcesToGpu(f.view().force(), AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                    integrator->integrate(stateGpu->getLocalForcesReadyOnDeviceEvent(
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        if (simulationWork.useMdGpuGraph)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            GMX_ASSERT((mdGraph != nullptr), "MD GPU graph does not exist.");
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                // update): with PME tuning, since the GPU kernels
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            usedMdGpuGraphLastStep = mdGraph->useGraphThisStep();
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->copyVelocitiesFromGpu(state->v, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                    if (useGpuForUpdate)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->resetCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->copyCoordinatesToGpu(state->x, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->waitCoordinatesCopiedToDevice(AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:                            stateGpu->copyVelocitiesToGpu(state->v, AtomLocality::Local);
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        const bool scaleCoordinates = !useGpuForUpdate || bDoReplEx;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        if (useGpuForUpdate
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        // any run that uses GPUs must be at least offloading nonbondeds
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        const bool usingGpu = simulationWork.useGpuNonbonded;
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        if (usingGpu)
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:            // ensure that GPU errors do not propagate between MD steps
patches/gromacs-2023.5.diff/src/gromacs/mdrun/md.cpp:        pme_loadbal_done(pme_loadbal, fplog, mdlog, fr->nbv->useGpu());
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:if (GMX_CLANG_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    include(gmxClangCudaUtils)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:set_property(GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:add_subdirectory(gpu_utils)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:# Mark some shared GPU implementation files to compile with CUDA if needed
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    get_property(CUDA_SOURCES GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    set_source_files_properties(${CUDA_SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        # needed as we need to include cufftmp include path before CUDA include path
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        cuda_include_directories(${cuFFTMp_INCLUDE_DIR})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    # Work around FindCUDA that prevents using target_link_libraries()
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    if (NOT GMX_CLANG_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        gmx_cuda_add_library(libgromacs ${LIBGROMACS_SOURCES})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        target_link_libraries(libgromacs PRIVATE ${CUDA_CUFFT_LIBRARIES})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:if(GMX_GPU_FFT_VKFFT)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:if(GMX_GPU_FFT_ROCFFT)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_FFT_CLFFT)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    if (NOT GMX_GPU_OPENCL)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        message(FATAL_ERROR "clFFT is only supported in OpenCL builds")
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:"An OpenCL build was requested with Visual Studio compiler, but GROMACS
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:clFFT to help with building for OpenCL, but that clFFT has not yet been
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:requires. Thus for now, OpenCL is not available with MSVC and the internal
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:installing a clFFT package, use VkFFT by setting -DGMX_GPU_FFT_LIBRARY=VkFFT,
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:# CUDA runtime headers
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_CUDA AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        set(GMX_CUDA_CLANG_FLAGS "${GMX_CUDA_CLANG_FLAGS} ${_compile_flag}")
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    if (GMX_CLANG_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:            get_source_file_property(_cuda_source_format ${_file} CUDA_SOURCE_PROPERTY_FORMAT)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:            if ("${_ext}" STREQUAL ".cu" OR _cuda_source_format)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:                gmx_compile_cuda_file_with_clang(${_file})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        get_property(CUDA_SOURCES GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        set_source_files_properties(${CUDA_SOURCES} PROPERTIES COMPILE_FLAGS ${GMX_CUDA_CLANG_FLAGS})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_SYCL)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:                      ${OpenCL_LIBRARIES}
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:                      $<BUILD_INTERFACE:gpu_utils>
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:# Technically, the user could want to do this for an OpenCL build
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:# using the CUDA runtime, but currently there's no reason to want to
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:if (INSTALL_CUDART_LIB) #can be set manual by user
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    if (GMX_GPU_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        foreach(CUDA_LIB ${CUDA_LIBRARIES})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:            string(REGEX MATCH "cudart" IS_CUDART ${CUDA_LIB})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:            if(IS_CUDART) #libcuda should not be installed
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:                file(GLOB CUDA_LIBS ${CUDA_LIB}*)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:                install(FILES ${CUDA_LIBS} DESTINATION
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        message(WARNING "INSTALL_CUDART_LIB only makes sense when configuring for CUDA support")
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:if(GMX_GPU_OPENCL)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        gpu_utils/vectype_ops.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        gpu_utils/device_utils.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        DESTINATION ${GMX_INSTALL_OCLDIR}/gromacs/gpu_utils
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernels.cl
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernel.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernel_pruneonly.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernels.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernels_fastgen.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernels_fastgen_add_twincut.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernel_utils.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_consts.h
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        DESTINATION ${GMX_INSTALL_OCLDIR}/gromacs/nbnxm/opencl
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        ewald/pme_gpu_calculate_splines.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:        ewald/pme_gpu_types.h
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_CLANG_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    include(gmxClangCudaUtils)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:set_property(GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:add_subdirectory(gpu_utils)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:# Mark some shared GPU implementation files to compile with CUDA if needed
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    get_property(CUDA_SOURCES GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    set_source_files_properties(${CUDA_SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        # needed as we need to include cufftmp include path before CUDA include path
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        cuda_include_directories(${cuFFTMp_INCLUDE_DIR})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    # Work around FindCUDA that prevents using target_link_libraries()
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    if (NOT GMX_CLANG_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        gmx_cuda_add_library(libgromacs ${LIBGROMACS_SOURCES})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        target_link_libraries(libgromacs PRIVATE ${CUDA_CUFFT_LIBRARIES})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:if(GMX_GPU_FFT_VKFFT)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:if(GMX_GPU_FFT_ROCFFT)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_FFT_CLFFT)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    if (NOT GMX_GPU_OPENCL)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        message(FATAL_ERROR "clFFT is only supported in OpenCL builds")
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:"An OpenCL build was requested with Visual Studio compiler, but GROMACS
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:clFFT to help with building for OpenCL, but that clFFT has not yet been
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:requires. Thus for now, OpenCL is not available with MSVC and the internal
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:installing a clFFT package, use VkFFT by setting -DGMX_GPU_FFT_LIBRARY=VkFFT,
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:# CUDA runtime headers
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_CUDA AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        set(GMX_CUDA_CLANG_FLAGS "${GMX_CUDA_CLANG_FLAGS} ${_compile_flag}")
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    if (GMX_CLANG_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:            get_source_file_property(_cuda_source_format ${_file} CUDA_SOURCE_PROPERTY_FORMAT)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:            if ("${_ext}" STREQUAL ".cu" OR _cuda_source_format)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:                gmx_compile_cuda_file_with_clang(${_file})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        get_property(CUDA_SOURCES GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        set_source_files_properties(${CUDA_SOURCES} PROPERTIES COMPILE_FLAGS ${GMX_CUDA_CLANG_FLAGS})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_SYCL)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:                      ${OpenCL_LIBRARIES}
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:                      $<BUILD_INTERFACE:gpu_utils>
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:# Technically, the user could want to do this for an OpenCL build
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:# using the CUDA runtime, but currently there's no reason to want to
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (INSTALL_CUDART_LIB) #can be set manual by user
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    if (GMX_GPU_CUDA)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        foreach(CUDA_LIB ${CUDA_LIBRARIES})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:            string(REGEX MATCH "cudart" IS_CUDART ${CUDA_LIB})
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:            if(IS_CUDART) #libcuda should not be installed
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:                file(GLOB CUDA_LIBS ${CUDA_LIB}*)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:                install(FILES ${CUDA_LIBS} DESTINATION
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        message(WARNING "INSTALL_CUDART_LIB only makes sense when configuring for CUDA support")
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:if(GMX_GPU_OPENCL)
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        gpu_utils/vectype_ops.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        gpu_utils/device_utils.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        DESTINATION ${GMX_INSTALL_OCLDIR}/gromacs/gpu_utils
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernels.cl
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernel.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernel_pruneonly.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernels.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernels_fastgen.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernels_fastgen_add_twincut.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernel_utils.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_consts.h
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        DESTINATION ${GMX_INSTALL_OCLDIR}/gromacs/nbnxm/opencl
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        ewald/pme_gpu_calculate_splines.clh
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:        ewald/pme_gpu_types.h
patches/gromacs-2023.5.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed: * \brief Defines functionality for deciding whether tasks will run on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#include "gromacs/taskassignment/decidegpuusage.h"
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#include "gromacs/mdlib/update_constrain_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        "When you use mdrun -gputasks, %s must be set to non-default "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#if GMX_GPU
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        " If you simply want to restrict which GPUs are used, then it is "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        "better to use mdrun -gpu_id. Otherwise, setting the "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#    if GMX_GPU_CUDA
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        "CUDA_VISIBLE_DEVICES"
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#    elif GMX_GPU_OPENCL
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // OpenCL standard, but the only current relevant case for GROMACS
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // is AMD OpenCL, which offers this variable.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        "GPU_DEVICE_ORDINAL"
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#    elif GMX_GPU_SYCL && GMX_SYCL_DPCPP
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#    elif GMX_GPU_SYCL && GMX_SYCL_HIPSYCL
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // Not true if we use hipSYCL over CUDA or IntelLLVM, but in that case the user probably
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // https://rocmdocs.amd.com/en/latest/Other_Solutions/Other-Solutions.html#hip-environment-variables
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:constexpr bool c_gpuBuildSyclWithoutGpuFft =
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        (GMX_GPU_SYCL != 0) && (GMX_GPU_FFT_MKL == 0) && (GMX_GPU_FFT_ROCFFT == 0)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        && (GMX_GPU_FFT_VKFFT == 0) && (GMX_GPU_FFT_DBFFT == 0); // NOLINT(misc-redundant-expression)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpusForNonbondedWithThreadMpi(const TaskTarget        nonbondedTarget,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                                     const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                                     const EmulateGpuNonbonded emulateGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                                     const bool buildSupportsNonbondedOnGpu,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                                     const bool nonbondedOnGpuIsUseful,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // First, exclude all cases where we can't run NB on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (nonbondedTarget == TaskTarget::Cpu || emulateGpuNonbonded == EmulateGpuNonbonded::Yes
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        || !nonbondedOnGpuIsUseful || !buildSupportsNonbondedOnGpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // If the user required NB on GPUs, we issue an error later.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // We now know that NB on GPUs makes sense, if we have any.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Because this is thread-MPI, we already know about the GPUs that
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // If we get here, then the user permitted or required GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:static bool decideWhetherToUseGpusForPmeFft(const TaskTarget pmeFftTarget)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                           || (pmeFftTarget == TaskTarget::Auto && c_gpuBuildSyclWithoutGpuFft);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:static bool canUseGpusForPme(const bool        useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    errorReasons.startContext("Cannot compute PME interactions on a GPU, because:");
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    errorReasons.appendIf(!useGpuForNonbonded, "Nonbonded interactions must also run on GPUs.");
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    errorReasons.appendIf(!pme_gpu_supports_build(&tempString), tempString);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    errorReasons.appendIf(!pme_gpu_supports_input(inputrec, &tempString), tempString);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!decideWhetherToUseGpusForPmeFft(pmeFftTarget))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        errorReasons.appendIf(!pme_gpu_mixed_mode_supports_input(inputrec, &tempString), tempString);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (pmeTarget == TaskTarget::Gpu && errorMessage != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpusForPmeWithThreadMpi(const bool              useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                               const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // First, exclude all cases where we can't run PME on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!canUseGpusForPme(useGpuForNonbonded, pmeTarget, pmeFftTarget, inputrec, nullptr))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // PME can't run on a GPU. If the user required that, we issue an error later.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // We now know that PME on GPUs might make sense, if we have any.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "PME tasks were required to run on GPUs with multiple ranks "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // Follow the user's choice of GPU task assignment, if we
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // can. Checking that their IDs are for compatible GPUs comes
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // PME on GPUs is only supported in a single case
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                        "When you run mdrun -pme gpu -gputasks, you must supply a PME-enabled .tpr "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Because this is thread-MPI, we already know about the GPUs that
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "PME tasks were required to run on GPUs, but that is not implemented with "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // PME can run well on a GPU shared with NB, and we permit
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // We have a single separate PME rank, that can use a GPU
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // run well on a GPU shared with NB, and we permit mdrun to
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // default to it if there is only one GPU available.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Not enough support for PME on GPUs for anything else
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpusForNonbonded(const TaskTarget          nonbondedTarget,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                        const std::vector<int>&   userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                        const EmulateGpuNonbonded emulateGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                        const bool                buildSupportsNonbondedOnGpu,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                        const bool                nonbondedOnGpuIsUseful,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                        const bool                gpusWereDetected)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "A GPU task assignment was specified, but nonbonded interactions were "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!buildSupportsNonbondedOnGpu && nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                "Nonbonded interactions on the GPU were requested with -nb gpu, "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                "but the GROMACS binary has been built without GPU support. "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                "Either run without selecting GPU options, or recompile GROMACS "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                "with GPU support enabled"));
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // TODO refactor all these TaskTarget::Gpu checks into one place?
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (emulateGpuNonbonded == EmulateGpuNonbonded::Yes)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "Nonbonded interactions on the GPU were required, which is inconsistent "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    InconsistentInputError("GPU ID usage was specified, as was GPU emulation. Make "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!nonbondedOnGpuIsUseful)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "Nonbonded interactions on the GPU were required, but not supported for these "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "simulation settings. Change your settings, or do not require using GPUs."));
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // We still don't know whether it is an error if no GPUs are found
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // If we get here, then the user permitted GPUs, which we should
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    return buildSupportsNonbondedOnGpu && gpusWereDetected;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpusForPme(const bool              useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                  const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                  const bool              gpusWereDetected)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!canUseGpusForPme(useGpuForNonbonded, pmeTarget, pmeFftTarget, inputrec, &message))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "A GPU task assignment was specified, but PME interactions were "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "PME tasks were required to run on GPUs with multiple ranks "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // We still don't know whether it is an error if no GPUs are found
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // If we get here, then the user permitted GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // PME can run well on a single GPU shared with NB when there
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // detected GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        return gpusWereDetected;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // We have a single separate PME rank, that can use a GPU
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        return gpusWereDetected;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Not enough support for PME on GPUs for anything else
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:PmeRunMode determinePmeRunMode(const bool useGpuForPme, const TaskTarget& pmeFftTarget, const t_inputrec& inputrec)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (useGpuForPme)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (c_gpuBuildSyclWithoutGpuFft && pmeFftTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                      "GROMACS is built without SYCL GPU FFT library. Please do not use -pmefft "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                      "gpu.");
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (!decideWhetherToUseGpusForPmeFft(pmeFftTarget))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:            return PmeRunMode::GPU;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (pmeFftTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                      "Assigning FFTs to GPU requires PME to be assigned to GPU as well. With PME "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpusForBonded(bool              useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                     bool              useGpuForPme,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                     bool              gpusWereDetected)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!buildSupportsListedForcesGpu(&errorMessage))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!inputSupportsListedForcesGpu(inputrec, mtop, &errorMessage))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!useGpuForNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "Bonded interactions on the GPU were required, but this requires that "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "short-ranged non-bonded interactions are also run on the GPU. Change "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "your settings, or do not require using GPUs."));
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // We still don't know whether it is an error if no GPUs are
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // If we get here, then the user permitted GPUs, which we should
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // choose separate PME ranks when nonBonded are assigned to the GPU.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                     || (usingPmeOrEwald(inputrec.coulombtype) && !useGpuForPme
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    return gpusWereDetected && usingOurCpuForPmeOrEwald;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpuForUpdate(const bool           isDomainDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                    const bool           useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                    const bool           gpusWereDetected,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Flag to set if we do not want to log the error with `-update auto` (e.g., for non-GPU build)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:            errorMessage += "With separate PME rank(s), PME must run on the GPU.\n";
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Using the GPU-version of update if:
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // 1. PME is on the GPU (there should be a copy of coordinates on GPU for PME spread) or inactive, or
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // 2. Non-bonded interactions are on the GPU.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if ((pmeRunMode == PmeRunMode::CPU || pmeRunMode == PmeRunMode::None) && !useGpuForNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                "Either PME or short-ranged non-bonded interaction tasks must run on the GPU.\n";
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!gpusWereDetected)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        errorMessage += "Compatible GPUs must have been found.\n";
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!(GMX_GPU_CUDA || GMX_GPU_SYCL))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        errorMessage += "Only CUDA and SYCL builds are supported.\n";
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // does not support it, the actual CUDA LINCS code does support it
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!UpdateConstrainGpu::isNumCoupledConstraintsSupported(mtop))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                "The number of coupled constraints is higher than supported in the GPU LINCS "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (hasAnyConstraints && !UpdateConstrainGpu::areConstraintsSupported())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        errorMessage += "Chosen GPU implementation does not support constraints.\n";
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // There is a known bug with frozen atoms and GPU update, see Issue #3920.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                            "Update task can not run on the GPU, because the following "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        else if (updateTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "Update task on the GPU was required,\n"
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    return (updateTarget == TaskTarget::Gpu
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherDirectGpuCommunicationCanBeUsed(const DevelopmentFeatureFlags& devFlags,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    const bool buildSupportsDirectGpuComm = (GMX_GPU_CUDA || GMX_GPU_SYCL) && GMX_MPI;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!buildSupportsDirectGpuComm)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Direct GPU communication is presently turned off due to insufficient testing
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    const bool enableDirectGpuComm = (getenv("GMX_ENABLE_DIRECT_GPU_COMM") != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                     || (getenv("GMX_GPU_DD_COMMS") != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                     || (getenv("GMX_GPU_PME_PP_COMMS") != nullptr);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (GMX_THREAD_MPI && GMX_GPU_SYCL && enableDirectGpuComm)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                        "GMX_ENABLE_DIRECT_GPU_COMM environment variable detected, "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    errorReasons.startContext("GPU direct communication can not be activated because:");
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    bool runAndGpuSupportDirectGpuComm = (runUsesCompatibleFeatures && enableDirectGpuComm);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    bool canUseDirectGpuCommWithThreadMpi =
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:            (runAndGpuSupportDirectGpuComm && GMX_THREAD_MPI && !GMX_GPU_SYCL);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // GPU-aware MPI case off by default, can be enabled with dev flag
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Note: GMX_DISABLE_DIRECT_GPU_COMM already taken into account in devFlags.enableDirectGpuCommWithMpi
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    bool canUseDirectGpuCommWithMpi = (runAndGpuSupportDirectGpuComm && GMX_LIB_MPI
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                       && devFlags.canUseGpuAwareMpi && enableDirectGpuComm);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    return canUseDirectGpuCommWithThreadMpi || canUseDirectGpuCommWithMpi;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpuForHalo(bool                 havePPDomainDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                  bool                 useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                  bool                 canUseDirectGpuComm,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!canUseDirectGpuComm || !havePPDomainDecomposition || !useGpuForNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    errorReasons.startContext("GPU halo exchange will not be activated because:");
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \brief Declares functionality for deciding whether tasks will run on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:#ifndef GMX_TASKASSIGNMENT_DECIDEGPUUSAGE_H
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:#define GMX_TASKASSIGNMENT_DECIDEGPUUSAGE_H
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    Gpu
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed://! Help pass GPU-emulation parameters with type safety.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:enum class EmulateGpuNonbonded : bool
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    //! Do not emulate GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    //! Do emulate GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    bool enableGpuBufferOps = false;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    //! True if the GPU-aware MPI can be used for GPU direct communication feature
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    bool canUseGpuAwareMpi = false;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    //! True if GPU PME-decomposition is enabled
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    bool enableGpuPmeDecomposition = false;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    //! True if CUDA Graphs are enabled
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    bool enableCudaGraphs = false;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * nonbonded tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * The number of GPU tasks and devices influences both the choice of
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in] userGpuTaskAssignment        The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in] emulateGpuNonbonded          Whether we will emulate GPU calculation of nonbonded
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in] buildSupportsNonbondedOnGpu  Whether GROMACS was built with GPU support.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in] nonbondedOnGpuIsUseful       Whether computing nonbonded interactions on a GPU is
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether the simulation will run nonbonded tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpusForNonbondedWithThreadMpi(TaskTarget              nonbondedTarget,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                                     const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                                     EmulateGpuNonbonded     emulateGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                                     bool buildSupportsNonbondedOnGpu,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                                     bool nonbondedOnGpuIsUseful,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * PME tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * The number of GPU tasks and devices influences both the choice of
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForNonbonded        Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  numDevicesToUse           The number of compatible GPUs that the user permitted us to use.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  userGpuTaskAssignment     The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether the simulation will run PME tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpusForPmeWithThreadMpi(bool                    useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                               const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * is known. But we need to know if nonbonded will run on GPUs for
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * user requires GPUs for the tasks of that duty, then it will be an
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * decideWhetherToUseGpusForNonbondedWithThreadMpi() and
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * decideWhetherToUseGpusForPmeWithThreadMpi() to help determine
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  userGpuTaskAssignment       The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  emulateGpuNonbonded         Whether we will emulate GPU calculation of nonbonded interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  buildSupportsNonbondedOnGpu Whether GROMACS was build with GPU support.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  nonbondedOnGpuIsUseful      Whether computing nonbonded interactions on a GPU is useful for this calculation.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  gpusWereDetected            Whether compatible GPUs were detected on any node.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether the simulation will run nonbonded and PME tasks, respectively, on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpusForNonbonded(TaskTarget              nonbondedTarget,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                        const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                        EmulateGpuNonbonded     emulateGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                        bool                    buildSupportsNonbondedOnGpu,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                        bool                    nonbondedOnGpuIsUseful,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                        bool                    gpusWereDetected);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * different types on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * is known. But we need to know if nonbonded will run on GPUs for
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * user requires GPUs for the tasks of that duty, then it will be an
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * decideWhetherToUseGpusForNonbondedWithThreadMpi() and
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * decideWhetherToUseGpusForPmeWithThreadMpi() to help determine
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForNonbonded        Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  userGpuTaskAssignment     The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  gpusWereDetected          Whether compatible GPUs were detected on any node.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether the simulation will run nonbonded and PME tasks, respectively, on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpusForPme(bool                    useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                  const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                  bool                    gpusWereDetected);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * Given the PME task assignment in \p useGpuForPme and the user-provided
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \note Aborts the run upon incompatible values of \p useGpuForPme and \p pmeFftTarget.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForPme              PME task assignment, true if PME task is mapped to the GPU.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:PmeRunMode determinePmeRunMode(bool useGpuForPme, const TaskTarget& pmeFftTarget, const t_inputrec& inputrec);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:/*! \brief Decide whether the simulation will try to run bonded tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForNonbonded        Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForPme              Whether GPUs will be used for PME interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  gpusWereDetected          Whether compatible GPUs were detected on any node.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether the simulation will run bondeded tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpusForBonded(bool              useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                     bool              useGpuForPme,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                     bool              gpusWereDetected);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:/*! \brief Decide whether to use GPU for update.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  pmeRunMode                   PME running mode: CPU, GPU or mixed.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForNonbonded           Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  updateTarget                 User choice for running simulation on GPU.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  gpusWereDetected             Whether compatible GPUs were detected on any node.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether complete simulation can be run on GPU.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpuForUpdate(bool                 isDomainDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                    bool                 useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                    bool                 gpusWereDetected,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:/*! \brief Decide whether direct GPU communication can be used.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * Takes into account the build type which determines feature support as well as GPU
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * development feature flags, determines whether this run can use direct GPU communication.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  devFlags                     GPU development / experimental feature flags.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether the MPI-parallel runs can use direct GPU communication.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherDirectGpuCommunicationCanBeUsed(const DevelopmentFeatureFlags& devFlags,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:/*! \brief Decide whether to use GPU for halo exchange.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForNonbonded           Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  canUseDirectGpuComm          Whether direct GPU communication can be used.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether halo exchange can be run on GPU.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpuForHalo(bool                 havePPDomainDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                  bool                 useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                  bool                 canUseDirectGpuComm,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \brief Declares functionality for deciding whether tasks will run on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:#ifndef GMX_TASKASSIGNMENT_DECIDEGPUUSAGE_H
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:#define GMX_TASKASSIGNMENT_DECIDEGPUUSAGE_H
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    Gpu
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h://! Help pass GPU-emulation parameters with type safety.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:enum class EmulateGpuNonbonded : bool
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    //! Do not emulate GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    //! Do emulate GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    bool enableGpuBufferOps = false;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    //! True if the GPU-aware MPI can be used for GPU direct communication feature
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    bool canUseGpuAwareMpi = false;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    //! True if GPU PME-decomposition is enabled
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    bool enableGpuPmeDecomposition = false;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    //! True if CUDA Graphs are enabled
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    bool enableCudaGraphs = false;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * nonbonded tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * The number of GPU tasks and devices influences both the choice of
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in] userGpuTaskAssignment        The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in] emulateGpuNonbonded          Whether we will emulate GPU calculation of nonbonded
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in] buildSupportsNonbondedOnGpu  Whether GROMACS was built with GPU support.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in] nonbondedOnGpuIsUseful       Whether computing nonbonded interactions on a GPU is
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether the simulation will run nonbonded tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpusForNonbondedWithThreadMpi(TaskTarget              nonbondedTarget,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                                     const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                                     EmulateGpuNonbonded     emulateGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                                     bool buildSupportsNonbondedOnGpu,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                                     bool nonbondedOnGpuIsUseful,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * PME tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * The number of GPU tasks and devices influences both the choice of
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForNonbonded        Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  numDevicesToUse           The number of compatible GPUs that the user permitted us to use.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  userGpuTaskAssignment     The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether the simulation will run PME tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpusForPmeWithThreadMpi(bool                    useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                               const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * is known. But we need to know if nonbonded will run on GPUs for
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * user requires GPUs for the tasks of that duty, then it will be an
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * decideWhetherToUseGpusForNonbondedWithThreadMpi() and
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * decideWhetherToUseGpusForPmeWithThreadMpi() to help determine
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  userGpuTaskAssignment       The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  emulateGpuNonbonded         Whether we will emulate GPU calculation of nonbonded interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  buildSupportsNonbondedOnGpu Whether GROMACS was build with GPU support.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  nonbondedOnGpuIsUseful      Whether computing nonbonded interactions on a GPU is useful for this calculation.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  gpusWereDetected            Whether compatible GPUs were detected on any node.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether the simulation will run nonbonded and PME tasks, respectively, on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpusForNonbonded(TaskTarget              nonbondedTarget,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                        const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                        EmulateGpuNonbonded     emulateGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                        bool                    buildSupportsNonbondedOnGpu,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                        bool                    nonbondedOnGpuIsUseful,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                        bool                    gpusWereDetected);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * different types on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * is known. But we need to know if nonbonded will run on GPUs for
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * user requires GPUs for the tasks of that duty, then it will be an
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * decideWhetherToUseGpusForNonbondedWithThreadMpi() and
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * decideWhetherToUseGpusForPmeWithThreadMpi() to help determine
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForNonbonded        Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  userGpuTaskAssignment     The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  gpusWereDetected          Whether compatible GPUs were detected on any node.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether the simulation will run nonbonded and PME tasks, respectively, on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpusForPme(bool                    useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                  const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                  bool                    gpusWereDetected);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * Given the PME task assignment in \p useGpuForPme and the user-provided
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \note Aborts the run upon incompatible values of \p useGpuForPme and \p pmeFftTarget.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForPme              PME task assignment, true if PME task is mapped to the GPU.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:PmeRunMode determinePmeRunMode(bool useGpuForPme, const TaskTarget& pmeFftTarget, const t_inputrec& inputrec);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:/*! \brief Decide whether the simulation will try to run bonded tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForNonbonded        Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForPme              Whether GPUs will be used for PME interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  gpusWereDetected          Whether compatible GPUs were detected on any node.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether the simulation will run bondeded tasks on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpusForBonded(bool              useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                     bool              useGpuForPme,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                     bool              gpusWereDetected);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:/*! \brief Decide whether to use GPU for update.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  pmeRunMode                   PME running mode: CPU, GPU or mixed.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForNonbonded           Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  updateTarget                 User choice for running simulation on GPU.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  gpusWereDetected             Whether compatible GPUs were detected on any node.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether complete simulation can be run on GPU.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpuForUpdate(bool                 isDomainDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                    bool                 useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                    bool                 gpusWereDetected,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:/*! \brief Decide whether direct GPU communication can be used.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * Takes into account the build type which determines feature support as well as GPU
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * development feature flags, determines whether this run can use direct GPU communication.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  devFlags                     GPU development / experimental feature flags.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether the MPI-parallel runs can use direct GPU communication.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherDirectGpuCommunicationCanBeUsed(const DevelopmentFeatureFlags& devFlags,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:/*! \brief Decide whether to use GPU for halo exchange.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForNonbonded           Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  canUseDirectGpuComm          Whether direct GPU communication can be used.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether halo exchange can be run on GPU.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpuForHalo(bool                 havePPDomainDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                  bool                 useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                  bool                 canUseDirectGpuComm,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp: * \brief Defines functionality for deciding whether tasks will run on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#include "gromacs/taskassignment/decidegpuusage.h"
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#include "gromacs/mdlib/update_constrain_gpu.h"
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        "When you use mdrun -gputasks, %s must be set to non-default "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#if GMX_GPU
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        " If you simply want to restrict which GPUs are used, then it is "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        "better to use mdrun -gpu_id. Otherwise, setting the "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#    if GMX_GPU_CUDA
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        "CUDA_VISIBLE_DEVICES"
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#    elif GMX_GPU_OPENCL
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // OpenCL standard, but the only current relevant case for GROMACS
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // is AMD OpenCL, which offers this variable.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        "GPU_DEVICE_ORDINAL"
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#    elif GMX_GPU_SYCL && GMX_SYCL_DPCPP
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#    elif GMX_GPU_SYCL && GMX_SYCL_HIPSYCL
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // Not true if we use hipSYCL over CUDA or IntelLLVM, but in that case the user probably
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // https://rocmdocs.amd.com/en/latest/Other_Solutions/Other-Solutions.html#hip-environment-variables
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:constexpr bool c_gpuBuildSyclWithoutGpuFft =
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        (GMX_GPU_SYCL != 0) && (GMX_GPU_FFT_MKL == 0) && (GMX_GPU_FFT_ROCFFT == 0)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        && (GMX_GPU_FFT_VKFFT == 0) && (GMX_GPU_FFT_DBFFT == 0); // NOLINT(misc-redundant-expression)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpusForNonbondedWithThreadMpi(const TaskTarget        nonbondedTarget,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                                     const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                                     const EmulateGpuNonbonded emulateGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                                     const bool buildSupportsNonbondedOnGpu,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                                     const bool nonbondedOnGpuIsUseful,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // First, exclude all cases where we can't run NB on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (nonbondedTarget == TaskTarget::Cpu || emulateGpuNonbonded == EmulateGpuNonbonded::Yes
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        || !nonbondedOnGpuIsUseful || !buildSupportsNonbondedOnGpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // If the user required NB on GPUs, we issue an error later.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // We now know that NB on GPUs makes sense, if we have any.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Because this is thread-MPI, we already know about the GPUs that
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // If we get here, then the user permitted or required GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:static bool decideWhetherToUseGpusForPmeFft(const TaskTarget pmeFftTarget)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                           || (pmeFftTarget == TaskTarget::Auto && c_gpuBuildSyclWithoutGpuFft);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:static bool canUseGpusForPme(const bool        useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    errorReasons.startContext("Cannot compute PME interactions on a GPU, because:");
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    errorReasons.appendIf(!useGpuForNonbonded, "Nonbonded interactions must also run on GPUs.");
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    errorReasons.appendIf(!pme_gpu_supports_build(&tempString), tempString);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    errorReasons.appendIf(!pme_gpu_supports_input(inputrec, &tempString), tempString);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!decideWhetherToUseGpusForPmeFft(pmeFftTarget))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        errorReasons.appendIf(!pme_gpu_mixed_mode_supports_input(inputrec, &tempString), tempString);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (pmeTarget == TaskTarget::Gpu && errorMessage != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpusForPmeWithThreadMpi(const bool              useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                               const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // First, exclude all cases where we can't run PME on GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!canUseGpusForPme(useGpuForNonbonded, pmeTarget, pmeFftTarget, inputrec, nullptr))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // PME can't run on a GPU. If the user required that, we issue an error later.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // We now know that PME on GPUs might make sense, if we have any.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "PME tasks were required to run on GPUs with multiple ranks "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // Follow the user's choice of GPU task assignment, if we
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // can. Checking that their IDs are for compatible GPUs comes
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // PME on GPUs is only supported in a single case
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                        "When you run mdrun -pme gpu -gputasks, you must supply a PME-enabled .tpr "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Because this is thread-MPI, we already know about the GPUs that
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "PME tasks were required to run on GPUs, but that is not implemented with "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // PME can run well on a GPU shared with NB, and we permit
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // We have a single separate PME rank, that can use a GPU
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // run well on a GPU shared with NB, and we permit mdrun to
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // default to it if there is only one GPU available.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Not enough support for PME on GPUs for anything else
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpusForNonbonded(const TaskTarget          nonbondedTarget,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                        const std::vector<int>&   userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                        const EmulateGpuNonbonded emulateGpuNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                        const bool                buildSupportsNonbondedOnGpu,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                        const bool                nonbondedOnGpuIsUseful,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                        const bool                gpusWereDetected)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "A GPU task assignment was specified, but nonbonded interactions were "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!buildSupportsNonbondedOnGpu && nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                "Nonbonded interactions on the GPU were requested with -nb gpu, "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                "but the GROMACS binary has been built without GPU support. "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                "Either run without selecting GPU options, or recompile GROMACS "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                "with GPU support enabled"));
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // TODO refactor all these TaskTarget::Gpu checks into one place?
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (emulateGpuNonbonded == EmulateGpuNonbonded::Yes)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "Nonbonded interactions on the GPU were required, which is inconsistent "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    InconsistentInputError("GPU ID usage was specified, as was GPU emulation. Make "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!nonbondedOnGpuIsUseful)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "Nonbonded interactions on the GPU were required, but not supported for these "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "simulation settings. Change your settings, or do not require using GPUs."));
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // We still don't know whether it is an error if no GPUs are found
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // If we get here, then the user permitted GPUs, which we should
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    return buildSupportsNonbondedOnGpu && gpusWereDetected;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpusForPme(const bool              useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                  const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                  const bool              gpusWereDetected)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!canUseGpusForPme(useGpuForNonbonded, pmeTarget, pmeFftTarget, inputrec, &message))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "A GPU task assignment was specified, but PME interactions were "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "PME tasks were required to run on GPUs with multiple ranks "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // We still don't know whether it is an error if no GPUs are found
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // If we get here, then the user permitted GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // PME can run well on a single GPU shared with NB when there
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // detected GPUs.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        return gpusWereDetected;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // We have a single separate PME rank, that can use a GPU
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        return gpusWereDetected;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Not enough support for PME on GPUs for anything else
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:PmeRunMode determinePmeRunMode(const bool useGpuForPme, const TaskTarget& pmeFftTarget, const t_inputrec& inputrec)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (useGpuForPme)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (c_gpuBuildSyclWithoutGpuFft && pmeFftTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                      "GROMACS is built without SYCL GPU FFT library. Please do not use -pmefft "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                      "gpu.");
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (!decideWhetherToUseGpusForPmeFft(pmeFftTarget))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:            return PmeRunMode::GPU;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (pmeFftTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                      "Assigning FFTs to GPU requires PME to be assigned to GPU as well. With PME "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpusForBonded(bool              useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                     bool              useGpuForPme,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                     bool              gpusWereDetected)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!buildSupportsListedForcesGpu(&errorMessage))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!inputSupportsListedForcesGpu(inputrec, mtop, &errorMessage))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!useGpuForNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "Bonded interactions on the GPU were required, but this requires that "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "short-ranged non-bonded interactions are also run on the GPU. Change "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "your settings, or do not require using GPUs."));
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // We still don't know whether it is an error if no GPUs are
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // If we get here, then the user permitted GPUs, which we should
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // choose separate PME ranks when nonBonded are assigned to the GPU.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                     || (usingPmeOrEwald(inputrec.coulombtype) && !useGpuForPme
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    return gpusWereDetected && usingOurCpuForPmeOrEwald;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpuForUpdate(const bool           isDomainDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                    const bool           useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                    const bool           gpusWereDetected,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Flag to set if we do not want to log the error with `-update auto` (e.g., for non-GPU build)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:            errorMessage += "With separate PME rank(s), PME must run on the GPU.\n";
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Using the GPU-version of update if:
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // 1. PME is on the GPU (there should be a copy of coordinates on GPU for PME spread) or inactive, or
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // 2. Non-bonded interactions are on the GPU.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if ((pmeRunMode == PmeRunMode::CPU || pmeRunMode == PmeRunMode::None) && !useGpuForNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                "Either PME or short-ranged non-bonded interaction tasks must run on the GPU.\n";
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!gpusWereDetected)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        errorMessage += "Compatible GPUs must have been found.\n";
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!(GMX_GPU_CUDA || GMX_GPU_SYCL))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        errorMessage += "Only CUDA and SYCL builds are supported.\n";
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // does not support it, the actual CUDA LINCS code does support it
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!UpdateConstrainGpu::isNumCoupledConstraintsSupported(mtop))
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                "The number of coupled constraints is higher than supported in the GPU LINCS "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (hasAnyConstraints && !UpdateConstrainGpu::areConstraintsSupported())
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        errorMessage += "Chosen GPU implementation does not support constraints.\n";
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // There is a known bug with frozen atoms and GPU update, see Issue #3920.
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                            "Update task can not run on the GPU, because the following "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        else if (updateTarget == TaskTarget::Gpu)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "Update task on the GPU was required,\n"
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    return (updateTarget == TaskTarget::Gpu
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherDirectGpuCommunicationCanBeUsed(const DevelopmentFeatureFlags& devFlags,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    const bool buildSupportsDirectGpuComm = (GMX_GPU_CUDA || GMX_GPU_SYCL) && GMX_MPI;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!buildSupportsDirectGpuComm)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Direct GPU communication is presently turned off due to insufficient testing
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    const bool enableDirectGpuComm = (getenv("GMX_ENABLE_DIRECT_GPU_COMM") != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                     || (getenv("GMX_GPU_DD_COMMS") != nullptr)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                     || (getenv("GMX_GPU_PME_PP_COMMS") != nullptr);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (GMX_THREAD_MPI && GMX_GPU_SYCL && enableDirectGpuComm)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                        "GMX_ENABLE_DIRECT_GPU_COMM environment variable detected, "
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    errorReasons.startContext("GPU direct communication can not be activated because:");
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    bool runAndGpuSupportDirectGpuComm = (runUsesCompatibleFeatures && enableDirectGpuComm);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    bool canUseDirectGpuCommWithThreadMpi =
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:            (runAndGpuSupportDirectGpuComm && GMX_THREAD_MPI && !GMX_GPU_SYCL);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // GPU-aware MPI case off by default, can be enabled with dev flag
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Note: GMX_DISABLE_DIRECT_GPU_COMM already taken into account in devFlags.enableDirectGpuCommWithMpi
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    bool canUseDirectGpuCommWithMpi = (runAndGpuSupportDirectGpuComm && GMX_LIB_MPI
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                       && devFlags.canUseGpuAwareMpi && enableDirectGpuComm);
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    return canUseDirectGpuCommWithThreadMpi || canUseDirectGpuCommWithMpi;
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpuForHalo(bool                 havePPDomainDecomposition,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                  bool                 useGpuForNonbonded,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                  bool                 canUseDirectGpuComm,
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!canUseDirectGpuComm || !havePPDomainDecomposition || !useGpuForNonbonded)
patches/gromacs-2023.5.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    errorReasons.startContext("GPU halo exchange will not be activated because:");
patches/qespresso-7.2.diff/PW/src/forces.f90.preplumed:#if defined(__CUDA)
patches/qespresso-7.2.diff/PW/src/forces.f90.preplumed:#if defined(__CUDA)
patches/qespresso-7.2.diff/PW/src/forces.f90.preplumed:  IF (ierr .ne. 0) CALL infomsg('forces', 'Cannot reset GPU buffers! Some buffers still locked.')
patches/qespresso-7.2.diff/PW/src/forces.f90.preplumed:#if defined(__CUDA)
patches/qespresso-7.2.diff/PW/src/forces.f90.preplumed:  IF (ierr .ne. 0) CALL errore('forces', 'Cannot reset GPU buffers! Buffers still locked: ', abs(ierr))
patches/qespresso-7.2.diff/PW/src/run_pwscf.f90:     IF ( ierr .ne. 0 ) CALL infomsg( 'run_pwscf', 'Cannot reset GPU buffers! Some buffers still locked.' )
patches/qespresso-7.2.diff/PW/src/forces.f90:#if defined(__CUDA)
patches/qespresso-7.2.diff/PW/src/forces.f90:#if defined(__CUDA)
patches/qespresso-7.2.diff/PW/src/forces.f90:  IF (ierr .ne. 0) CALL infomsg('forces', 'Cannot reset GPU buffers! Some buffers still locked.')
patches/qespresso-7.2.diff/PW/src/forces.f90:#if defined(__CUDA)
patches/qespresso-7.2.diff/PW/src/forces.f90:  IF (ierr .ne. 0) CALL errore('forces', 'Cannot reset GPU buffers! Buffers still locked: ', abs(ierr))
patches/qespresso-7.2.diff/PW/src/run_pwscf.f90.preplumed:     IF ( ierr .ne. 0 ) CALL infomsg( 'run_pwscf', 'Cannot reset GPU buffers! Some buffers still locked.' )
patches/qespresso-7.2.diff/Modules/Makefile:# GPU versions of modules
patches/qespresso-7.2.diff/Modules/Makefile:  wavefunctions_gpu.o \
patches/qespresso-7.2.diff/Modules/Makefile:  becmod_gpu.o \
patches/qespresso-7.2.diff/Modules/Makefile:  becmod_subs_gpu.o \
patches/qespresso-7.2.diff/Modules/Makefile:  cuda_subroutines.o \
patches/qespresso-7.2.diff/Modules/Makefile:  random_numbers_gpu.o
patches/qespresso-7.2.diff/Modules/Makefile.preplumed:# GPU versions of modules
patches/qespresso-7.2.diff/Modules/Makefile.preplumed:  wavefunctions_gpu.o \
patches/qespresso-7.2.diff/Modules/Makefile.preplumed:  becmod_gpu.o \
patches/qespresso-7.2.diff/Modules/Makefile.preplumed:  becmod_subs_gpu.o \
patches/qespresso-7.2.diff/Modules/Makefile.preplumed:  cuda_subroutines.o \
patches/qespresso-7.2.diff/Modules/Makefile.preplumed:  random_numbers_gpu.o
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:  USE control_flags,     ONLY : use_gpu
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:#if defined(__CUDA)
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:  IF (ierr .ne. 0) CALL infomsg('forces', 'Cannot reset GPU buffers! Some buffers still locked.')
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:  IF (.not. use_gpu) CALL force_us( forcenl )
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:  IF (      use_gpu) CALL force_us_gpu( forcenl )
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:  IF (.not. use_gpu) & ! On the CPU
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:  IF (      use_gpu) THEN ! On the GPU
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:     ! move these data to the GPU
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:     CALL force_lc_gpu( nat, tau, ityp, alat, omega, ngm, ngl, igtongl_d, &
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:  IF (.not. use_gpu) CALL force_cc( forcecc )
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:  IF (      use_gpu) CALL force_cc_gpu( forcecc )
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:  IF (.not. use_gpu) THEN
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:     IF ( lda_plus_u .AND. U_projection.NE.'pseudo' ) CALL force_hub_gpu( forceh )
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:  IF (ierr .ne. 0) CALL errore('forces', 'Cannot reset GPU buffers! Buffers still locked: ', abs(ierr))
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:  IF ( .not. use_gpu ) CALL force_corr( forcescc )
patches/qespresso-7.0.diff/PW/src/forces.f90.preplumed:  IF (       use_gpu ) CALL force_corr_gpu( forcescc )
patches/qespresso-7.0.diff/PW/src/run_pwscf.f90:     IF ( ierr .ne. 0 ) CALL infomsg( 'run_pwscf', 'Cannot reset GPU buffers! Some buffers still locked.' )
patches/qespresso-7.0.diff/PW/src/forces.f90:  USE control_flags,     ONLY : use_gpu
patches/qespresso-7.0.diff/PW/src/forces.f90:#if defined(__CUDA)
patches/qespresso-7.0.diff/PW/src/forces.f90:  IF (ierr .ne. 0) CALL infomsg('forces', 'Cannot reset GPU buffers! Some buffers still locked.')
patches/qespresso-7.0.diff/PW/src/forces.f90:  IF (.not. use_gpu) CALL force_us( forcenl )
patches/qespresso-7.0.diff/PW/src/forces.f90:  IF (      use_gpu) CALL force_us_gpu( forcenl )
patches/qespresso-7.0.diff/PW/src/forces.f90:  IF (.not. use_gpu) & ! On the CPU
patches/qespresso-7.0.diff/PW/src/forces.f90:  IF (      use_gpu) THEN ! On the GPU
patches/qespresso-7.0.diff/PW/src/forces.f90:     ! move these data to the GPU
patches/qespresso-7.0.diff/PW/src/forces.f90:     CALL force_lc_gpu( nat, tau, ityp, alat, omega, ngm, ngl, igtongl_d, &
patches/qespresso-7.0.diff/PW/src/forces.f90:  IF (.not. use_gpu) CALL force_cc( forcecc )
patches/qespresso-7.0.diff/PW/src/forces.f90:  IF (      use_gpu) CALL force_cc_gpu( forcecc )
patches/qespresso-7.0.diff/PW/src/forces.f90:  IF (.not. use_gpu) THEN
patches/qespresso-7.0.diff/PW/src/forces.f90:     IF ( lda_plus_u .AND. U_projection.NE.'pseudo' ) CALL force_hub_gpu( forceh )
patches/qespresso-7.0.diff/PW/src/forces.f90:  IF (ierr .ne. 0) CALL errore('forces', 'Cannot reset GPU buffers! Buffers still locked: ', abs(ierr))
patches/qespresso-7.0.diff/PW/src/forces.f90:  IF ( .not. use_gpu ) CALL force_corr( forcescc )
patches/qespresso-7.0.diff/PW/src/forces.f90:  IF (       use_gpu ) CALL force_corr_gpu( forcescc )
patches/qespresso-7.0.diff/PW/src/run_pwscf.f90.preplumed:     IF ( ierr .ne. 0 ) CALL infomsg( 'run_pwscf', 'Cannot reset GPU buffers! Some buffers still locked.' )
patches/qespresso-7.0.diff/Modules/Makefile:# GPU versions of modules
patches/qespresso-7.0.diff/Modules/Makefile:  wavefunctions_gpu.o \
patches/qespresso-7.0.diff/Modules/Makefile:  becmod_gpu.o \
patches/qespresso-7.0.diff/Modules/Makefile:  becmod_subs_gpu.o \
patches/qespresso-7.0.diff/Modules/Makefile:  cuda_subroutines.o \
patches/qespresso-7.0.diff/Modules/Makefile:  random_numbers_gpu.o
patches/qespresso-7.0.diff/Modules/Makefile.preplumed:# GPU versions of modules
patches/qespresso-7.0.diff/Modules/Makefile.preplumed:  wavefunctions_gpu.o \
patches/qespresso-7.0.diff/Modules/Makefile.preplumed:  becmod_gpu.o \
patches/qespresso-7.0.diff/Modules/Makefile.preplumed:  becmod_subs_gpu.o \
patches/qespresso-7.0.diff/Modules/Makefile.preplumed:  cuda_subroutines.o \
patches/qespresso-7.0.diff/Modules/Makefile.preplumed:  random_numbers_gpu.o
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/ewald/pme_coordinate_receiver_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/ewald/pme_pp_comm_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/nbnxm/nbnxm_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/timing/gpu_timing.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gpuforcereduction.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static const bool c_disableAlternatingWait = (getenv("GMX_DISABLE_ALTERNATING_GPU_WAIT") != nullptr);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   bool                  useGpuPmePpComms,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   bool                  receivePmeForceToGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gmx_pme_receive_f(fr->pmePpCommGpu.get(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                      useGpuPmePpComms,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                      receivePmeForceToGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* GPU kernel launch overhead is already timed separately */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (!nbv->useGpu())
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Launch the prepare_step and spread stages of PME GPU.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static inline void launchPmeGpuSpread(gmx_pme_t*            pmedata,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                      GpuEventSynchronizer* xReadyOnDevice,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_prepare_computation(pmedata, box, wcycle, stepWork);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool                           useGpuDirectComm         = false;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gmx::PmeCoordinateReceiverGpu* pmeCoordinateReceiverGpu = nullptr;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_launch_spread(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            pmedata, xReadyOnDevice, wcycle, lambdaQ, useGpuDirectComm, pmeCoordinateReceiverGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Launch the FFT and gather stages of PME GPU
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void launchPmeGpuFftAndGather(gmx_pme_t*               pmedata,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_launch_complex_transforms(pmedata, wcycle, stepWork);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_launch_gather(pmedata, wcycle, lambdaQ);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: *  Polling wait for either of the PME or nonbonded GPU tasks.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * Instead of a static order in waiting for GPU tasks, this function
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * one of the reductions, regardless of the GPU task completion order.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void alternatePmeNbGpuWaitReduce(nonbonded_verlet_t* nbv,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool isPmeGpuDone = false;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool isNbGpuDone  = false;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gmx::ArrayRef<const gmx::RVec> pmeGpuForces;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    while (!isPmeGpuDone || !isNbGpuDone)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!isPmeGpuDone)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GpuTaskCompletion completionType =
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    (isNbGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            isPmeGpuDone = pme_gpu_try_finish_task(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!isNbGpuDone)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GpuTaskCompletion completionType =
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    (isPmeGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            isNbGpuDone = Nbnxm::gpu_try_finish_task(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    nbv->gpu_nbv,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (isNbGpuDone)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        && (domainWork.haveCpuLocalForceWork || !stepWork.useGpuFBufferOps
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            || (havePpDomainDecomposition && !stepWork.useGpuFHalo)))
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    domainWork.haveGpuBondedWork =
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            ((fr.listedForcesGpu != nullptr) && fr.listedForcesGpu->haveInteractions());
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuXBufferOps || simulationWork.useGpuFBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        GMX_ASSERT(simulationWork.useGpuNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    flags.useGpuXBufferOps = simulationWork.useGpuXBufferOps && !flags.doNeighborSearch;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    flags.useGpuFBufferOps       = simulationWork.useGpuFBufferOps && !flags.computeVirial;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    const bool rankHasGpuPmeTask = simulationWork.useGpuPme && !simulationWork.haveSeparatePmeRank;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    flags.useGpuPmeFReduction    = flags.computeSlowForces && flags.useGpuFBufferOps
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                && (rankHasGpuPmeTask || simulationWork.useGpuPmePpCommunication);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    flags.useGpuXHalo              = simulationWork.useGpuHaloExchange && !flags.doNeighborSearch;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    flags.useGpuFHalo              = simulationWork.useGpuHaloExchange && flags.useGpuFBufferOps;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    flags.haveGpuPmeOnThisRank     = rankHasGpuPmeTask && flags.computeSlowForces;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:             && !(flags.computeVirial || simulationWork.useGpuNonbonded || flags.haveGpuPmeOnThisRank));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/* \brief Launch end-of-step GPU tasks: buffer clearing and rolling pruning.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void launchGpuEndOfStepTasks(nonbonded_verlet_t*               nbv,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                    gmx::ListedForcesGpu*             listedForcesGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork.simulationWork.useGpuNonbonded && runScheduleWork.stepWork.computeNonbondedForces)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:         * clear kernel launches can leave the GPU idle while it could be running
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (nbv->isDynamicPruningStepGpu(step))
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            nbv->dispatchPruneKernelGpu(step);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* now clear the GPU outputs while we finish the step on the CPU */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        Nbnxm::gpu_clear_outputs(nbv->gpu_nbv, runScheduleWork.stepWork.computeVirial);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork.stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pme_gpu_reinit_computation(pmedata, wcycle);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork.domainWork.haveGpuBondedWork && runScheduleWork.stepWork.computeEnergy)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        listedForcesGpu->waitAccumulateEnergyTerms(enerd);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        listedForcesGpu->clearEnergies();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Compute the number of times the "local coordinates ready on device" GPU event will be used as a synchronization point.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * When some work is offloaded to GPU, force calculation should wait for the atom coordinates to
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * or from the GPU integration at the end of the previous step.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param pmeSendCoordinatesFromGpu Whether peer-to-peer communication is used for PME coordinates.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                                          bool pmeSendCoordinatesFromGpu)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (pmeSendCoordinatesFromGpu)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                       "GPU PME PP communications require having a separate PME rank");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // Event is consumed by gmx_pme_send_coordinates for GPU PME PP Communications
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // Event is consumed by launchPmeGpuSpread
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.computeNonbondedForces && stepWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // Event is consumed by convertCoordinatesGpu
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.useGpuXHalo)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // Event is consumed by communicateGpuHaloCoordinates
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Setup for the local GPU force reduction:
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] stateGpu            GPU state propagator object
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] gpuForceReduction   GPU force reduction object
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] pmePpCommGpu        PME-PP GPU communication object
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void setupLocalGpuForceReduction(const gmx::MdrunScheduleWorkload* runScheduleWork,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        gmx::StatePropagatorDataGpu*      stateGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        gmx::GpuForceReduction*           gpuForceReduction,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        gmx::PmePpCommGpu*                pmePpCommGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "GPU force reduction is not compatible with MTS");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // (re-)initialize local GPU force reduction
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gpuForceReduction->reinit(stateGpu->getForces(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                              stateGpu->fReducedOnDevice(AtomLocality::Local));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gpuForceReduction->registerNbnxmForce(Nbnxm::gpu_get_f(nbv->gpu_nbv));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GpuEventSynchronizer*   pmeSynchronizer     = nullptr;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork->simulationWork.useGpuPme && !runScheduleWork->simulationWork.haveSeparatePmeRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pmeForcePtr = pme_gpu_get_device_f(pmedata);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            pmeSynchronizer     = pme_gpu_get_f_ready_synchronizer(pmedata);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    else if (runScheduleWork->simulationWork.useGpuPmePpCommunication)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pmeForcePtr = pmePpCommGpu->getGpuForceStagingPtr();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                pmeSynchronizer = pmePpCommGpu->getForcesReadySynchronizer();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        gpuForceReduction->registerRvecForce(pmeForcePtr);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!runScheduleWork->simulationWork.useGpuPmePpCommunication || GMX_THREAD_MPI)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GMX_ASSERT(pmeSynchronizer != nullptr, "PME force ready cuda event should not be NULL");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            gpuForceReduction->addDependency(pmeSynchronizer);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            && !runScheduleWork->simulationWork.useGpuHaloExchange))
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::Local));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork->simulationWork.useGpuHaloExchange)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        gpuForceReduction->addDependency(dd->gpuHaloExchange[0][0]->getForcesReadyOnDeviceEvent());
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Setup for the non-local GPU force reduction:
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] stateGpu            GPU state propagator object
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] gpuForceReduction   GPU force reduction object
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void setupNonLocalGpuForceReduction(const gmx::MdrunScheduleWorkload* runScheduleWork,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                           gmx::StatePropagatorDataGpu*      stateGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                           gmx::GpuForceReduction*           gpuForceReduction,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // (re-)initialize non-local GPU force reduction
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gpuForceReduction->reinit(stateGpu->getForces(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                              stateGpu->fReducedOnDevice(AtomLocality::NonLocal));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gpuForceReduction->registerNbnxmForce(Nbnxm::gpu_get_f(nbv->gpu_nbv));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::NonLocal));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gmx::StatePropagatorDataGpu* stateGpu = fr->stateGpu;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.doNeighborSearch && gmx::needStateGpu(simulationWork))
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->reinit(mdatoms->homenr,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // TODO: This should be moved into PME setup function ( pme_gpu_prepare_computation(...) )
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            pme_gpu_set_device_x(fr->pmedata, stateGpu->getCoordinates());
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.useGpuFHalo && !runScheduleWork->domainWork.haveCpuLocalForceWork && !stepWork.doNeighborSearch)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // GPU Force halo exchange will set a subset of local atoms with remote non-local data
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // On NS steps, the buffer could have already cleared in stateGpu->reinit.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->clearForcesOnGpu(AtomLocality::Local,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    const bool pmeSendCoordinatesFromGpu =
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            simulationWork.useGpuPmePpCommunication && !(stepWork.doNeighborSearch);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    const bool reinitGpuPmePpComms =
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            simulationWork.useGpuPmePpCommunication && (stepWork.doNeighborSearch);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    auto* localXReadyOnDevice = (stepWork.haveGpuPmeOnThisRank || simulationWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        ? stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GMX_ASSERT(simulationWork.useGpuHaloExchange
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                       == ((cr->dd != nullptr) && (!cr->dd->gpuHaloExchange[0].empty())),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "The GPU halo exchange is active, but it has not been constructed.");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool gmx_used_in_debug haveCopiedXFromGpu = false;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // Copy coordinate from the GPU if update is on the GPU and there
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->copyCoordinatesFromGpu(x.unpaddedArrayRef(), AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        haveCopiedXFromGpu = true;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.haveGpuPmeOnThisRank || stepWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        GMX_ASSERT(stateGpu != nullptr, "stateGpu should not be null");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        simulationWork, stepWork, pmeSendCoordinatesFromGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!simulationWork.useGpuUpdate || stepWork.doNeighborSearch)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        else if (simulationWork.useGpuUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            stateGpu->setXUpdatedOnDeviceEventExpectedConsumptionCount(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!pmeSendCoordinatesFromGpu && !stepWork.doNeighborSearch && simulationWork.useGpuUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 simulationWork.useGpuPmePpCommunication,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 reinitGpuPmePpComms,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 pmeSendCoordinatesFromGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 stepWork.useGpuPmeFReduction,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        launchPmeGpuSpread(fr->pmedata,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* initialize the GPU nbnxm atom data and bonded data structures */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (simulationWork.useGpuNonbonded)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // Note: cycle counting only nononbondeds, GPU listed forces counts internally
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            Nbnxm::gpu_init_atomdata(nbv->gpu_nbv, nbv->nbat.get());
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (fr->listedForcesGpu)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                 * interactions to the GPU, where the grid order is
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        Nbnxm::gpu_get_xq(nbv->gpu_nbv),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        Nbnxm::gpu_get_f(nbv->gpu_nbv),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        Nbnxm::gpu_get_fshift(nbv->gpu_nbv));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // Need to run after the GPU-offload bonded interaction lists
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (simulationWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            nbv->atomdata_init_copy_x_to_nbat_x_gpu();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (simulationWork.useGpuFBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            setupLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        stateGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        fr->gpuForceReduction[gmx::AtomLocality::Local].get(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        fr->pmePpCommGpu.get(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                setupNonLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                               stateGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                               fr->gpuForceReduction[gmx::AtomLocality::NonLocal].get(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GMX_ASSERT(stateGpu, "stateGpu should be valid when buffer ops are offloaded");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            nbv->convertCoordinatesGpu(AtomLocality::Local, stateGpu->getCoordinates(), localXReadyOnDevice);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (simulationWork.useGpuUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                GMX_ASSERT(stateGpu, "need a valid stateGpu object");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuNonbonded && (stepWork.computeNonbondedForces || domainWork.haveGpuBondedWork))
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        ddBalanceRegionHandler.openBeforeForceComputationGpu();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        Nbnxm::gpu_upload_shiftvec(nbv->gpu_nbv, nbv->nbat.get());
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!stepWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            Nbnxm::gpu_copy_xq_to_gpu(nbv->gpu_nbv, nbv->nbat.get(), AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // with X buffer ops offloaded to the GPU on all but the search steps
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (domainWork.haveGpuBondedWork && !simulationWork.havePpDomainDecomposition)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* launch local nonbonded work on GPU */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // In PME GPU and mixed mode we launch FFT / gather after the
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // X copy/transform to allow overlap as well as after the GPU NB
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        launchPmeGpuFftAndGather(fr->pmedata,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // TODO refactor this GPU halo exchange re-initialisation
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // to location in do_md where GPU halo exchange is
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // constructed at partitioning, after above stateGpu
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (simulationWork.useGpuHaloExchange)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                reinitGpuHaloExchange(*cr, stateGpu->getCoordinates(), stateGpu->getForces());
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GpuEventSynchronizer* gpuCoordinateHaloLaunched = nullptr;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.useGpuXHalo)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                gpuCoordinateHaloLaunched = communicateGpuHaloCoordinates(*cr, box, localXReadyOnDevice);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->copyCoordinatesFromGpu(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                            x.unpaddedArrayRef(), AtomLocality::NonLocal, gpuCoordinateHaloLaunched);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (simulationWork.useGpuUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                            (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (!stepWork.useGpuXHalo)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(), AtomLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                nbv->convertCoordinatesGpu(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        stateGpu->getCoordinates(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                AtomLocality::NonLocal, simulationWork, stepWork, gpuCoordinateHaloLaunched));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (simulationWork.useGpuNonbonded)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (!stepWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                wallcycle_start(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                Nbnxm::gpu_copy_xq_to_gpu(nbv->gpu_nbv, nbv->nbat.get(), AtomLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (domainWork.haveGpuBondedWork)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            /* launch non-local nonbonded tasks on GPU */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuNonbonded && stepWork.computeNonbondedForces)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            Nbnxm::gpu_launch_cpyback(nbv->gpu_nbv, nbv->nbat.get(), stepWork, AtomLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        Nbnxm::gpu_launch_cpyback(nbv->gpu_nbv, nbv->nbat.get(), stepWork, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (domainWork.haveGpuBondedWork && stepWork.computeEnergy)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            fr->listedForcesGpu->launchEnergyTransfer();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // For the rest of the CPU tasks that depend on GPU-update produced coordinates,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                || (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    const bool useOrEmulateGpuNb = simulationWork.useGpuNonbonded || fr->nbv->emulateGpu();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (!useOrEmulateGpuNb)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.useGpuXHalo && (domainWork.haveCpuBondedWork || domainWork.haveFreeEnergyWork))
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:         * Happens here on the CPU both with and without GPU.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.computeNonbondedForces && !useOrEmulateGpuNb)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            (stepWork.haveGpuPmeOnThisRank || needToReceivePmeResultsFromSeparateRank);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:     * GPU we must wait for the PME calculation (dhdl) results to finish before sampling the
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            pme_gpu_wait_and_reduce(fr->pmedata,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   simulationWork.useGpuPmePpCommunication,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   stepWork.useGpuPmeFReduction,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.havePpDomainDecomposition && stepWork.computeForces && stepWork.useGpuFHalo
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(), AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFBufferOps),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFHalo),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "The schedule below does not allow for nonbonded MTS with GPU halo exchange");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // Will store the amount of cycles spent waiting for the GPU that
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    float cycles_wait_gpu = 0;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (useOrEmulateGpuNb && stepWork.computeNonbondedForces)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (simulationWork.useGpuNonbonded)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                cycles_wait_gpu += Nbnxm::gpu_wait_finish_task(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        nbv->gpu_nbv,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.useGpuFBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                fr->gpuForceReduction[gmx::AtomLocality::NonLocal]->execute();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (!stepWork.useGpuFHalo)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    // copy from GPU input for dd_move_f()
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->copyForcesFromGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (fr->nbv->emulateGpu() && stepWork.computeVirial)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:         * If we use a GPU this will overlap with GPU work, so in that case
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.useGpuFHalo)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                gmx::FixedCapacityVector<GpuEventSynchronizer*, 2> gpuForceHaloDependencies;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                gpuForceHaloDependencies.push_back(stateGpu->fReadyOnDevice(AtomLocality::Local));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                gpuForceHaloDependencies.push_back(stateGpu->fReducedOnDevice(AtomLocality::NonLocal));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                communicateGpuHaloForces(*cr, accumulateForces, &gpuForceHaloDependencies);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (stepWork.useGpuFBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->waitForcesReadyOnHost(AtomLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // With both nonbonded and PME offloaded a GPU on the same rank, we use
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool alternateGpuWait = (!c_disableAlternatingWait && stepWork.haveGpuPmeOnThisRank
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                             && simulationWork.useGpuNonbonded && !simulationWork.havePpDomainDecomposition
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                             && !stepWork.useGpuFBufferOps && !needEarlyPmeResults);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (alternateGpuWait)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        alternatePmeNbGpuWaitReduce(fr->nbv.get(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (!alternateGpuWait && stepWork.haveGpuPmeOnThisRank && !needEarlyPmeResults)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pme_gpu_wait_and_reduce(fr->pmedata,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* Wait for local GPU NB outputs on the non-alternating wait path */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (!alternateGpuWait && stepWork.computeNonbondedForces && simulationWork.useGpuNonbonded)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* Measured overhead on CUDA and OpenCL with(out) GPU sharing
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        const float gpuWaitApiOverheadMargin = 2e6F; /* cycles */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        const float waitCycles               = Nbnxm::gpu_wait_finish_task(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                nbv->gpu_nbv,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            DdBalanceRegionWaitedForGpu waitedForGpu = DdBalanceRegionWaitedForGpu::yes;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.computeForces && waitCycles <= gpuWaitApiOverheadMargin)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                waitedForGpu = DdBalanceRegionWaitedForGpu::no;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            ddBalanceRegionHandler.closeAfterForceComputationGpu(cycles_wait_gpu, waitedForGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (fr->nbv->emulateGpu())
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // If on GPU PME-PP comms path, receive forces from PME before GPU buffer ops
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (needToReceivePmeResultsFromSeparateRank && simulationWork.useGpuPmePpCommunication && !needEarlyPmeResults)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                               simulationWork.useGpuPmePpCommunication,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                               stepWork.useGpuPmeFReduction,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* Do the nonbonded GPU (or emulation) force buffer reduction
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFBufferOps),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (useOrEmulateGpuNb && !alternateGpuWait)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.useGpuFBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // - copy is not perfomed if GPU force halo exchange is active, because it would overwrite the result
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (domainWork.haveLocalForceContribInCpuBuffer && !stepWork.useGpuFHalo)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                stateGpu->copyForcesToGpu(forceWithShift, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                fr->gpuForceReduction[gmx::AtomLocality::Local]->execute();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (!simulationWork.useGpuUpdate
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                || (simulationWork.useGpuUpdate && haveDDAtomOrdering(*cr) && simulationWork.useCpuPmePpCommunication)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    /* We have previously issued force reduction on the GPU, but we will
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                stateGpu->copyForcesFromGpu(forceWithShift, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    launchGpuEndOfStepTasks(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            nbv, fr->listedForcesGpu.get(), fr->pmedata, enerd, *runScheduleWork, step, wcycle);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // TODO refactor this and unify with above GPU PME-PP / GPU update path call to the same function
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                               simulationWork.useGpuPmePpCommunication,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* In case we don't have constraints and are using GPUs, the next balancing
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/ewald/pme_coordinate_receiver_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/ewald/pme_pp_comm_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/nbnxm/nbnxm_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/timing/gpu_timing.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:#include "gpuforcereduction.h"
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:static const bool c_disableAlternatingWait = (getenv("GMX_DISABLE_ALTERNATING_GPU_WAIT") != nullptr);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                   bool                  useGpuPmePpComms,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                   bool                  receivePmeForceToGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    gmx_pme_receive_f(fr->pmePpCommGpu.get(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                      useGpuPmePpComms,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                      receivePmeForceToGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    /* GPU kernel launch overhead is already timed separately */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (!nbv->useGpu())
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Launch the prepare_step and spread stages of PME GPU.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:static inline void launchPmeGpuSpread(gmx_pme_t*            pmedata,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                      GpuEventSynchronizer* xReadyOnDevice,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_prepare_computation(pmedata, box, wcycle, stepWork);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    bool                           useGpuDirectComm         = false;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    gmx::PmeCoordinateReceiverGpu* pmeCoordinateReceiverGpu = nullptr;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_launch_spread(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            pmedata, xReadyOnDevice, wcycle, lambdaQ, useGpuDirectComm, pmeCoordinateReceiverGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Launch the FFT and gather stages of PME GPU
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:static void launchPmeGpuFftAndGather(gmx_pme_t*               pmedata,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_launch_complex_transforms(pmedata, wcycle, stepWork);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_launch_gather(pmedata, wcycle, lambdaQ);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp: *  Polling wait for either of the PME or nonbonded GPU tasks.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp: * Instead of a static order in waiting for GPU tasks, this function
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp: * one of the reductions, regardless of the GPU task completion order.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:static void alternatePmeNbGpuWaitReduce(nonbonded_verlet_t* nbv,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    bool isPmeGpuDone = false;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    bool isNbGpuDone  = false;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    gmx::ArrayRef<const gmx::RVec> pmeGpuForces;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    while (!isPmeGpuDone || !isNbGpuDone)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (!isPmeGpuDone)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            GpuTaskCompletion completionType =
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                    (isNbGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            isPmeGpuDone = pme_gpu_try_finish_task(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (!isNbGpuDone)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            GpuTaskCompletion completionType =
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                    (isPmeGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            isNbGpuDone = Nbnxm::gpu_try_finish_task(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                    nbv->gpu_nbv,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (isNbGpuDone)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        && (domainWork.haveCpuLocalForceWork || !stepWork.useGpuFBufferOps
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            || (havePpDomainDecomposition && !stepWork.useGpuFHalo)))
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    domainWork.haveGpuBondedWork =
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            ((fr.listedForcesGpu != nullptr) && fr.listedForcesGpu->haveInteractions());
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuXBufferOps || simulationWork.useGpuFBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        GMX_ASSERT(simulationWork.useGpuNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    flags.useGpuXBufferOps = simulationWork.useGpuXBufferOps && !flags.doNeighborSearch;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    flags.useGpuFBufferOps       = simulationWork.useGpuFBufferOps && !flags.computeVirial;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    const bool rankHasGpuPmeTask = simulationWork.useGpuPme && !simulationWork.haveSeparatePmeRank;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    flags.useGpuPmeFReduction    = flags.computeSlowForces && flags.useGpuFBufferOps
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                && (rankHasGpuPmeTask || simulationWork.useGpuPmePpCommunication);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    flags.useGpuXHalo              = simulationWork.useGpuHaloExchange && !flags.doNeighborSearch;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    flags.useGpuFHalo              = simulationWork.useGpuHaloExchange && flags.useGpuFBufferOps;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    flags.haveGpuPmeOnThisRank     = rankHasGpuPmeTask && flags.computeSlowForces;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:             && !(flags.computeVirial || simulationWork.useGpuNonbonded || flags.haveGpuPmeOnThisRank));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:/* \brief Launch end-of-step GPU tasks: buffer clearing and rolling pruning.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:static void launchGpuEndOfStepTasks(nonbonded_verlet_t*               nbv,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                    gmx::ListedForcesGpu*             listedForcesGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork.simulationWork.useGpuNonbonded && runScheduleWork.stepWork.computeNonbondedForces)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:         * clear kernel launches can leave the GPU idle while it could be running
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (nbv->isDynamicPruningStepGpu(step))
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            nbv->dispatchPruneKernelGpu(step);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        /* now clear the GPU outputs while we finish the step on the CPU */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        Nbnxm::gpu_clear_outputs(nbv->gpu_nbv, runScheduleWork.stepWork.computeVirial);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork.stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        pme_gpu_reinit_computation(pmedata, wcycle);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork.domainWork.haveGpuBondedWork && runScheduleWork.stepWork.computeEnergy)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        listedForcesGpu->waitAccumulateEnergyTerms(enerd);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        listedForcesGpu->clearEnergies();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Compute the number of times the "local coordinates ready on device" GPU event will be used as a synchronization point.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp: * When some work is offloaded to GPU, force calculation should wait for the atom coordinates to
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp: * or from the GPU integration at the end of the previous step.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param pmeSendCoordinatesFromGpu Whether peer-to-peer communication is used for PME coordinates.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                                          bool pmeSendCoordinatesFromGpu)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (pmeSendCoordinatesFromGpu)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                       "GPU PME PP communications require having a separate PME rank");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            // Event is consumed by gmx_pme_send_coordinates for GPU PME PP Communications
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            // Event is consumed by launchPmeGpuSpread
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.computeNonbondedForces && stepWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            // Event is consumed by convertCoordinatesGpu
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.useGpuXHalo)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        // Event is consumed by communicateGpuHaloCoordinates
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Setup for the local GPU force reduction:
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] stateGpu            GPU state propagator object
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] gpuForceReduction   GPU force reduction object
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] pmePpCommGpu        PME-PP GPU communication object
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:static void setupLocalGpuForceReduction(const gmx::MdrunScheduleWorkload* runScheduleWork,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                        gmx::StatePropagatorDataGpu*      stateGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                        gmx::GpuForceReduction*           gpuForceReduction,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                        gmx::PmePpCommGpu*                pmePpCommGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:               "GPU force reduction is not compatible with MTS");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    // (re-)initialize local GPU force reduction
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    gpuForceReduction->reinit(stateGpu->getForces(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                              stateGpu->fReducedOnDevice(AtomLocality::Local));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    gpuForceReduction->registerNbnxmForce(Nbnxm::gpu_get_f(nbv->gpu_nbv));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    GpuEventSynchronizer*   pmeSynchronizer     = nullptr;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork->simulationWork.useGpuPme && !runScheduleWork->simulationWork.haveSeparatePmeRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        pmeForcePtr = pme_gpu_get_device_f(pmedata);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            pmeSynchronizer     = pme_gpu_get_f_ready_synchronizer(pmedata);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    else if (runScheduleWork->simulationWork.useGpuPmePpCommunication)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        pmeForcePtr = pmePpCommGpu->getGpuForceStagingPtr();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                pmeSynchronizer = pmePpCommGpu->getForcesReadySynchronizer();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        gpuForceReduction->registerRvecForce(pmeForcePtr);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (!runScheduleWork->simulationWork.useGpuPmePpCommunication || GMX_THREAD_MPI)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            GMX_ASSERT(pmeSynchronizer != nullptr, "PME force ready cuda event should not be NULL");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            gpuForceReduction->addDependency(pmeSynchronizer);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            && !runScheduleWork->simulationWork.useGpuHaloExchange))
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::Local));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork->simulationWork.useGpuHaloExchange)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        gpuForceReduction->addDependency(dd->gpuHaloExchange[0][0]->getForcesReadyOnDeviceEvent());
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Setup for the non-local GPU force reduction:
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] stateGpu            GPU state propagator object
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] gpuForceReduction   GPU force reduction object
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:static void setupNonLocalGpuForceReduction(const gmx::MdrunScheduleWorkload* runScheduleWork,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                           gmx::StatePropagatorDataGpu*      stateGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                           gmx::GpuForceReduction*           gpuForceReduction,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    // (re-)initialize non-local GPU force reduction
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    gpuForceReduction->reinit(stateGpu->getForces(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                              stateGpu->fReducedOnDevice(AtomLocality::NonLocal));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    gpuForceReduction->registerNbnxmForce(Nbnxm::gpu_get_f(nbv->gpu_nbv));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::NonLocal));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    gmx::StatePropagatorDataGpu* stateGpu = fr->stateGpu;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.doNeighborSearch && gmx::needStateGpu(simulationWork))
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->reinit(mdatoms->homenr,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            // TODO: This should be moved into PME setup function ( pme_gpu_prepare_computation(...) )
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            pme_gpu_set_device_x(fr->pmedata, stateGpu->getCoordinates());
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.useGpuFHalo && !runScheduleWork->domainWork.haveCpuLocalForceWork && !stepWork.doNeighborSearch)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        // GPU Force halo exchange will set a subset of local atoms with remote non-local data
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        // On NS steps, the buffer could have already cleared in stateGpu->reinit.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->clearForcesOnGpu(AtomLocality::Local,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                   stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    const bool pmeSendCoordinatesFromGpu =
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            simulationWork.useGpuPmePpCommunication && !(stepWork.doNeighborSearch);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    const bool reinitGpuPmePpComms =
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            simulationWork.useGpuPmePpCommunication && (stepWork.doNeighborSearch);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    auto* localXReadyOnDevice = (stepWork.haveGpuPmeOnThisRank || simulationWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                        ? stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    GMX_ASSERT(simulationWork.useGpuHaloExchange
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                       == ((cr->dd != nullptr) && (!cr->dd->gpuHaloExchange[0].empty())),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:               "The GPU halo exchange is active, but it has not been constructed.");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    bool gmx_used_in_debug haveCopiedXFromGpu = false;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    // Copy coordinate from the GPU if update is on the GPU and there
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->copyCoordinatesFromGpu(x.unpaddedArrayRef(), AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        haveCopiedXFromGpu = true;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.haveGpuPmeOnThisRank || stepWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        GMX_ASSERT(stateGpu != nullptr, "stateGpu should not be null");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                        simulationWork, stepWork, pmeSendCoordinatesFromGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (!simulationWork.useGpuUpdate || stepWork.doNeighborSearch)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        else if (simulationWork.useGpuUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            stateGpu->setXUpdatedOnDeviceEventExpectedConsumptionCount(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (!pmeSendCoordinatesFromGpu && !stepWork.doNeighborSearch && simulationWork.useGpuUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                 simulationWork.useGpuPmePpCommunication,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                 reinitGpuPmePpComms,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                 pmeSendCoordinatesFromGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                 stepWork.useGpuPmeFReduction,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        launchPmeGpuSpread(fr->pmedata,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        /* initialize the GPU nbnxm atom data and bonded data structures */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (simulationWork.useGpuNonbonded)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            // Note: cycle counting only nononbondeds, GPU listed forces counts internally
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            Nbnxm::gpu_init_atomdata(nbv->gpu_nbv, nbv->nbat.get());
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (fr->listedForcesGpu)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                 * interactions to the GPU, where the grid order is
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                        Nbnxm::gpu_get_xq(nbv->gpu_nbv),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                        Nbnxm::gpu_get_f(nbv->gpu_nbv),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                        Nbnxm::gpu_get_fshift(nbv->gpu_nbv));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        // Need to run after the GPU-offload bonded interaction lists
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (simulationWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            nbv->atomdata_init_copy_x_to_nbat_x_gpu();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (simulationWork.useGpuFBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            setupLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                        stateGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                        fr->gpuForceReduction[gmx::AtomLocality::Local].get(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                        fr->pmePpCommGpu.get(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                setupNonLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                               stateGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                               fr->gpuForceReduction[gmx::AtomLocality::NonLocal].get(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            GMX_ASSERT(stateGpu, "stateGpu should be valid when buffer ops are offloaded");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            nbv->convertCoordinatesGpu(AtomLocality::Local, stateGpu->getCoordinates(), localXReadyOnDevice);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (simulationWork.useGpuUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                GMX_ASSERT(stateGpu, "need a valid stateGpu object");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuNonbonded && (stepWork.computeNonbondedForces || domainWork.haveGpuBondedWork))
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        ddBalanceRegionHandler.openBeforeForceComputationGpu();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        Nbnxm::gpu_upload_shiftvec(nbv->gpu_nbv, nbv->nbat.get());
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (!stepWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            Nbnxm::gpu_copy_xq_to_gpu(nbv->gpu_nbv, nbv->nbat.get(), AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        // with X buffer ops offloaded to the GPU on all but the search steps
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (domainWork.haveGpuBondedWork && !simulationWork.havePpDomainDecomposition)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        /* launch local nonbonded work on GPU */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        // In PME GPU and mixed mode we launch FFT / gather after the
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        // X copy/transform to allow overlap as well as after the GPU NB
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        launchPmeGpuFftAndGather(fr->pmedata,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            // TODO refactor this GPU halo exchange re-initialisation
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            // to location in do_md where GPU halo exchange is
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            // constructed at partitioning, after above stateGpu
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (simulationWork.useGpuHaloExchange)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                reinitGpuHaloExchange(*cr, stateGpu->getCoordinates(), stateGpu->getForces());
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            GpuEventSynchronizer* gpuCoordinateHaloLaunched = nullptr;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.useGpuXHalo)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                gpuCoordinateHaloLaunched = communicateGpuHaloCoordinates(*cr, box, localXReadyOnDevice);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->copyCoordinatesFromGpu(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                            x.unpaddedArrayRef(), AtomLocality::NonLocal, gpuCoordinateHaloLaunched);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                if (simulationWork.useGpuUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                    GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                            (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                if (!stepWork.useGpuXHalo)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(), AtomLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                nbv->convertCoordinatesGpu(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                        stateGpu->getCoordinates(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                        stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                AtomLocality::NonLocal, simulationWork, stepWork, gpuCoordinateHaloLaunched));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (simulationWork.useGpuNonbonded)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (!stepWork.useGpuXBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                wallcycle_start(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                Nbnxm::gpu_copy_xq_to_gpu(nbv->gpu_nbv, nbv->nbat.get(), AtomLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (domainWork.haveGpuBondedWork)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            /* launch non-local nonbonded tasks on GPU */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuNonbonded && stepWork.computeNonbondedForces)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            Nbnxm::gpu_launch_cpyback(nbv->gpu_nbv, nbv->nbat.get(), stepWork, AtomLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        Nbnxm::gpu_launch_cpyback(nbv->gpu_nbv, nbv->nbat.get(), stepWork, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (domainWork.haveGpuBondedWork && stepWork.computeEnergy)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            fr->listedForcesGpu->launchEnergyTransfer();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    // For the rest of the CPU tasks that depend on GPU-update produced coordinates,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                || (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    const bool useOrEmulateGpuNb = simulationWork.useGpuNonbonded || fr->nbv->emulateGpu();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (!useOrEmulateGpuNb)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.useGpuXHalo && (domainWork.haveCpuBondedWork || domainWork.haveFreeEnergyWork))
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:         * Happens here on the CPU both with and without GPU.
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.computeNonbondedForces && !useOrEmulateGpuNb)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            (stepWork.haveGpuPmeOnThisRank || needToReceivePmeResultsFromSeparateRank);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:     * GPU we must wait for the PME calculation (dhdl) results to finish before sampling the
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            pme_gpu_wait_and_reduce(fr->pmedata,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                   simulationWork.useGpuPmePpCommunication,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                                   stepWork.useGpuPmeFReduction,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.havePpDomainDecomposition && stepWork.computeForces && stepWork.useGpuFHalo
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(), AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFBufferOps),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFHalo),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:               "The schedule below does not allow for nonbonded MTS with GPU halo exchange");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    // Will store the amount of cycles spent waiting for the GPU that
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    float cycles_wait_gpu = 0;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (useOrEmulateGpuNb && stepWork.computeNonbondedForces)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (simulationWork.useGpuNonbonded)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                cycles_wait_gpu += Nbnxm::gpu_wait_finish_task(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                        nbv->gpu_nbv,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.useGpuFBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                fr->gpuForceReduction[gmx::AtomLocality::NonLocal]->execute();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                if (!stepWork.useGpuFHalo)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                    // copy from GPU input for dd_move_f()
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->copyForcesFromGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (fr->nbv->emulateGpu() && stepWork.computeVirial)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:         * If we use a GPU this will overlap with GPU work, so in that case
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.useGpuFHalo)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                gmx::FixedCapacityVector<GpuEventSynchronizer*, 2> gpuForceHaloDependencies;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                gpuForceHaloDependencies.push_back(stateGpu->fReadyOnDevice(AtomLocality::Local));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                gpuForceHaloDependencies.push_back(stateGpu->fReducedOnDevice(AtomLocality::NonLocal));
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                communicateGpuHaloForces(*cr, accumulateForces, &gpuForceHaloDependencies);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                if (stepWork.useGpuFBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->waitForcesReadyOnHost(AtomLocality::NonLocal);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    // With both nonbonded and PME offloaded a GPU on the same rank, we use
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    bool alternateGpuWait = (!c_disableAlternatingWait && stepWork.haveGpuPmeOnThisRank
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                             && simulationWork.useGpuNonbonded && !simulationWork.havePpDomainDecomposition
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                             && !stepWork.useGpuFBufferOps && !needEarlyPmeResults);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (alternateGpuWait)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        alternatePmeNbGpuWaitReduce(fr->nbv.get(),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (!alternateGpuWait && stepWork.haveGpuPmeOnThisRank && !needEarlyPmeResults)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        pme_gpu_wait_and_reduce(fr->pmedata,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    /* Wait for local GPU NB outputs on the non-alternating wait path */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (!alternateGpuWait && stepWork.computeNonbondedForces && simulationWork.useGpuNonbonded)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        /* Measured overhead on CUDA and OpenCL with(out) GPU sharing
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        const float gpuWaitApiOverheadMargin = 2e6F; /* cycles */
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        const float waitCycles               = Nbnxm::gpu_wait_finish_task(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                nbv->gpu_nbv,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            DdBalanceRegionWaitedForGpu waitedForGpu = DdBalanceRegionWaitedForGpu::yes;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.computeForces && waitCycles <= gpuWaitApiOverheadMargin)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                waitedForGpu = DdBalanceRegionWaitedForGpu::no;
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            ddBalanceRegionHandler.closeAfterForceComputationGpu(cycles_wait_gpu, waitedForGpu);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (fr->nbv->emulateGpu())
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    // If on GPU PME-PP comms path, receive forces from PME before GPU buffer ops
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (needToReceivePmeResultsFromSeparateRank && simulationWork.useGpuPmePpCommunication && !needEarlyPmeResults)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                               simulationWork.useGpuPmePpCommunication,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                               stepWork.useGpuPmeFReduction,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    /* Do the nonbonded GPU (or emulation) force buffer reduction
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFBufferOps),
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    if (useOrEmulateGpuNb && !alternateGpuWait)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.useGpuFBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            // - copy is not perfomed if GPU force halo exchange is active, because it would overwrite the result
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (domainWork.haveLocalForceContribInCpuBuffer && !stepWork.useGpuFHalo)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                stateGpu->copyForcesToGpu(forceWithShift, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                fr->gpuForceReduction[gmx::AtomLocality::Local]->execute();
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            if (!simulationWork.useGpuUpdate
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                || (simulationWork.useGpuUpdate && haveDDAtomOrdering(*cr) && simulationWork.useCpuPmePpCommunication)
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                    /* We have previously issued force reduction on the GPU, but we will
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                stateGpu->copyForcesFromGpu(forceWithShift, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    launchGpuEndOfStepTasks(
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:            nbv, fr->listedForcesGpu.get(), fr->pmedata, enerd, *runScheduleWork, step, wcycle);
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    // TODO refactor this and unify with above GPU PME-PP / GPU update path call to the same function
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:                               simulationWork.useGpuPmePpCommunication,
patches/gromacs-2022.5.diff/src/gromacs/mdlib/sim_util.cpp:    /* In case we don't have constraints and are using GPUs, the next balancing
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/ewald/pme_gpu_program.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/ewald/pme_pp_comm_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/gpu_utils/device_stream_manager.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/mdlib/gpuforcereduction.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/taskassignment/decidegpuusage.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/taskassignment/usergpuids.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/timing/gpu_timing.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp: * the GPU communication flags are set to false in non-tMPI and non-CUDA builds.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp: * \param[in]  useGpuForNonbonded   True if the nonbonded task is offloaded in this run.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                         const bool           useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    devFlags.enableGpuBufferOps = (GMX_GPU_CUDA || GMX_GPU_SYCL) && useGpuForNonbonded
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                  && (getenv("GMX_USE_GPU_BUFFER_OPS") != nullptr);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    devFlags.forceGpuUpdateDefault = (getenv("GMX_FORCE_UPDATE_DEFAULT_GPU") != nullptr) || GMX_FAHCORE;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    // Flag use to enable GPU-aware MPI depenendent features such PME GPU decomposition
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    // GPU-aware MPI is marked available if it has been detected by GROMACS or detection fails but
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    devFlags.canUseGpuAwareMpi = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    // Direct GPU comm path is being used with GPU-aware MPI
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    // make sure underlying MPI implementation is GPU-aware
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    if (GMX_LIB_MPI && GMX_GPU_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        // Allow overriding the detection for GPU-aware MPI
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        GpuAwareMpiStatus gpuAwareMpiStatus = checkMpiCudaAwareSupport();
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        const bool        forceGpuAwareMpi  = gpuAwareMpiStatus == GpuAwareMpiStatus::Forced;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        const bool haveDetectedGpuAwareMpi  = gpuAwareMpiStatus == GpuAwareMpiStatus::Supported;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        if (getenv("GMX_FORCE_CUDA_AWARE_MPI") != nullptr)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                            "GMX_FORCE_CUDA_AWARE_MPI environment variable is inactive. "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                            "Please use GMX_FORCE_GPU_AWARE_MPI instead.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        devFlags.canUseGpuAwareMpi = haveDetectedGpuAwareMpi || forceGpuAwareMpi;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        if (getenv("GMX_ENABLE_DIRECT_GPU_COMM") != nullptr)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            if (!haveDetectedGpuAwareMpi && forceGpuAwareMpi)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                // GPU-aware support not detected in MPI library but, user has forced it's use
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                "This run has forced use of 'GPU-aware MPI', ie. 'CUDA-aware MPI'. "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                "However, GROMACS cannot determine if underlying MPI is GPU-aware. "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                "GROMACS recommends use of latest OpenMPI version for GPU-aware "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                "GMX_FORCE_GPU_AWARE_MPI environment variable.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            if (devFlags.canUseGpuAwareMpi)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                "GMX_ENABLE_DIRECT_GPU_COMM environment variable detected, "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                "enabling direct GPU communication using GPU-aware MPI.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                "GPU-aware MPI was not detected, will not use direct GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                "for GPU-aware support. If you are certain about GPU-aware support "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                "GMX_FORCE_GPU_AWARE_MPI environment variable. Note that such "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                "support is often called \"CUDA-aware MPI.\"");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        else if (haveDetectedGpuAwareMpi)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            // GPU-aware MPI was detected, let the user know that using it may improve performance
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                            "GPU-aware MPI detected, but by default GROMACS will not "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                            "make use the direct GPU communication capabilities of MPI. "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                            "the GMX_ENABLE_DIRECT_GPU_COMM environment variable.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        if (getenv("GMX_FORCE_GPU_AWARE_MPI") != nullptr)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            // Cannot force use of GPU-aware MPI in this build configuration
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                            "A CUDA build with an external MPI library is required in order to "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                            "benefit from GMX_FORCE_GPU_AWARE_MPI. That environment variable is "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    if (devFlags.enableGpuBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                        "This run uses the 'GPU buffer ops' feature, enabled by the "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                        "GMX_USE_GPU_BUFFER_OPS environment variable.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    if (devFlags.forceGpuUpdateDefault)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                        "This run will default to '-update gpu' as requested by the "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                        "GMX_FORCE_UPDATE_DEFAULT_GPU environment variable.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    // PME decomposition is supported only with CUDA-backend in mixed mode
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    // CUDA-backend also needs GPU-aware MPI support for decomposition to work
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    const bool pmeGpuDecompositionRequested =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            (pmeRunMode == PmeRunMode::GPU || pmeRunMode == PmeRunMode::Mixed)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    const bool pmeGpuDecompositionSupported =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            (devFlags.canUseGpuAwareMpi && pmeRunMode == PmeRunMode::Mixed);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    const bool forcePmeGpuDecomposition = getenv("GMX_GPU_PME_DECOMPOSITION") != nullptr;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    if (pmeGpuDecompositionSupported && pmeGpuDecompositionRequested)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        // PME decomposition is supported only when it is forced using GMX_GPU_PME_DECOMPOSITION
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        if (forcePmeGpuDecomposition)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                            "This run has requested the 'GPU PME decomposition' feature, enabled "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                            "by the GMX_GPU_PME_DECOMPOSITION environment variable. "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                      "Multiple PME tasks were required to run on GPUs, "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                      "Use GMX_GPU_PME_DECOMPOSITION environment variable to enable it.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    if (!pmeGpuDecompositionSupported && pmeGpuDecompositionRequested)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                  "PME tasks were required to run on GPUs, but that is not implemented with "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    devFlags.enableGpuPmeDecomposition =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            forcePmeGpuDecomposition && pmeGpuDecompositionRequested && pmeGpuDecompositionSupported;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                  bool                makeGpuPairList,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                (makeGpuPairList ? ListSetupType::Gpu : ListSetupType::CpuSimdWhenSupported);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        increaseNstlist(fplog, cr, ir, nstlist_cmdline, &mtop, box, makeGpuPairList, cpuinfo);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:/*! \brief Return whether GPU acceleration of nonbondeds is supported with the given settings.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:static bool gpuAccelerationOfNonbondedIsUseful(const MDLogger&   mdlog,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    bool        gpuIsUseful = true;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        /* The GPU code does not support more than one energy group.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:         * If the user requested GPUs explicitly, a fatal error is given later.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        gpuIsUseful = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                    "Multiple energy groups is not implemented for GPUs, falling back to the CPU. "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                    "For better performance, run on the GPU without energy groups and then do "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    /* There are resource handling issues in the GPU code paths with MTS on anything else than only
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        gpuIsUseful = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                "Multiple time stepping is only supported with GPUs when MTS is only applied to %s "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        gpuIsUseful = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        warning     = "TPI is not implemented for GPUs.";
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    if (!gpuIsUseful && issueWarning)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    return gpuIsUseful;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    else if (strncmp(optionString, "gpu", 3) == 0)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        returnValue = TaskTarget::Gpu;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        auto* nbnxn_gpu_timings =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                (nbv != nullptr && nbv->useGpu()) ? Nbnxm::gpu_get_timings(nbv->gpu_nbv) : nullptr;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        gmx_wallclock_gpu_pme_t pme_gpu_timings = {};
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        if (pme_gpu_task_enabled(pme))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            pme_gpu_get_timings(pme, &pme_gpu_timings);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                        nbnxn_gpu_timings,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                        &pme_gpu_timings);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    EmulateGpuNonbonded emulateGpuNonbonded =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            (getenv("GMX_EMULATE_GPU") != nullptr ? EmulateGpuNonbonded::Yes : EmulateGpuNonbonded::No);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    std::vector<int> userGpuTaskAssignment;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        userGpuTaskAssignment = parseUserTaskAssignmentString(hw_opt.userGpuTaskAssignment);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        bool useGpuForNonbonded = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        bool useGpuForPme       = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            // the number of GPUs to choose the number of ranks.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            auto canUseGpuForNonbonded = buildSupportsNonbondedOnGpu(nullptr);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            useGpuForNonbonded         = decideWhetherToUseGpusForNonbondedWithThreadMpi(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                    userGpuTaskAssignment,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                    emulateGpuNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                    canUseGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                    gpuAccelerationOfNonbondedIsUseful(mdlog, *inputrec, GMX_THREAD_MPI, doRerun),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            useGpuForPme = decideWhetherToUseGpusForPmeWithThreadMpi(useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                                     userGpuTaskAssignment,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                useGpuForPme,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    // Note that when bonded interactions run on a GPU they always run
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuForNonbonded = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuForPme       = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuForBonded    = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuForUpdate    = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    bool gpusWereDetected   = hwinfo_->ngpu_compatible_tot > 0;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        // It's possible that there are different numbers of GPUs on
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        auto canUseGpuForNonbonded = buildSupportsNonbondedOnGpu(nullptr);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        useGpuForNonbonded         = decideWhetherToUseGpusForNonbonded(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                userGpuTaskAssignment,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                emulateGpuNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                canUseGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                gpuAccelerationOfNonbondedIsUseful(mdlog, *inputrec, !GMX_THREAD_MPI, doRerun),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                gpusWereDetected);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        useGpuForPme    = decideWhetherToUseGpusForPme(useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                    userGpuTaskAssignment,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                    gpusWereDetected);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        useGpuForBonded = decideWhetherToUseGpusForBonded(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                useGpuForNonbonded, useGpuForPme, bondedTarget, *inputrec, mtop, domdecOptions.numPmeRanks, gpusWereDetected);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    const PmeRunMode pmeRunMode = determinePmeRunMode(useGpuForPme, pmeFftTarget, *inputrec);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            mdlog, useGpuForNonbonded, pmeRunMode, cr->sizeOfDefaultCommunicator, domdecOptions.numPmeRanks);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                || (!useGpuForNonbonded && EEL_FULL(inputrec->coulombtype) && useDDWithSingleRank != 0)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                          useGpuForNonbonded || (emulateGpuNonbonded == EmulateGpuNonbonded::Yes),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        useGpuForUpdate = decideWhetherToUseGpuForUpdate(useDomainDecomposition,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                         useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                         gpusWereDetected,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    const bool canUseDirectGpuComm = decideWhetherDirectGpuCommunicationCanBeUsed(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuDirectHalo = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    if (useGpuForNonbonded)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        // domdecOptions.numPmeRanks == -1 results in 0 separate PME ranks when useGpuForNonbonded is true.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        useGpuDirectHalo = decideWhetherToUseGpuForHalo(havePPDomainDecomposition,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                        useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                        canUseDirectGpuComm,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        // The DD builder will disable useGpuDirectHalo if the Y or Z component of any domain is
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        // smaller than twice the communication distance, since GPU-direct communication presently
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        // perform well on multiple GPUs in any case, but it is important that our core functionality
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        // (in particular for testing) does not break depending on GPU direct communication being enabled.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                useGpuForPme,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                useGpuForUpdate,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                &useGpuDirectHalo,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                devFlags.enableGpuPmeDecomposition);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    GpuTaskAssignments gpuTaskAssignments = GpuTaskAssignmentsBuilder::build(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            userGpuTaskAssignment,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            useGpuForPme,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    DeviceInformation* deviceInfo = gpuTaskAssignments.initDevice(&deviceId);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        // TODO Pass the GPU streams to ddBuilder to use in buffer
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    const bool useGpuPmeDecomposition = numPmeDomains.x * numPmeDomains.y > 1 && useGpuForPme;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    GMX_RELEASE_ASSERT(!useGpuPmeDecomposition || devFlags.enableGpuPmeDecomposition,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                       "GPU PME decomposition works only in the cases where it is supported");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuForBonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuForUpdate,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuDirectHalo,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                              canUseDirectGpuComm,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuPmeDecomposition);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    if (isSimulationMasterRank && GMX_GPU_SYCL)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        bool                      haveAnyGpuWork = simWorkload.useGpuPme || simWorkload.useGpuBonded
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                              || simWorkload.useGpuNonbonded || simWorkload.useGpuUpdate;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        if (haveAnyGpuWork)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                            "\nNOTE: SYCL GPU support in GROMACS is still new and less tested than "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    gpuTaskAssignments.reportGpuUsage(mdlog, printHostName, pmeRunMode, runScheduleWork.simulationWork);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        const bool useGpuTiming = decideGpuTimingsUsage();
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                *deviceInfo, havePPDomainDecomposition(cr), runScheduleWork.simulationWork, useGpuTiming);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        gpuTaskAssignments.logPerformanceHints(mdlog, numAvailableDevices);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    check_resource_division_efficiency(hwinfo_, gpuTaskAssignments.thisRankHasAnyGpuTask(), cr, mdlog);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    // Enable Peer access between GPUs where available
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    // any of the GPU communication features are active.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        && (runScheduleWork.simulationWork.useGpuHaloExchange
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            || runScheduleWork.simulationWork.useGpuPmePpCommunication))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        setupGpuDevicePeerAccess(gpuTaskAssignments.deviceIdsAssigned(), mdlog);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    const bool thisRankHasPmeGpuTask = gpuTaskAssignments.thisRankHasPmeGpuTask();
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        if (runScheduleWork.simulationWork.useGpuPmePpCommunication && !thisRankHasDuty(cr, DUTY_PME))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                    "GPU device stream manager should be valid in order to use PME-PP direct "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                    "GPU PP-PME stream should be valid in order to use GPU PME-PP direct "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            fr->pmePpCommGpu = std::make_unique<gmx::PmePpCommGpu>(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                        runScheduleWork.simulationWork.useGpuNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        // TODO: Move the logic below to a GPU bonded builder
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        if (runScheduleWork.simulationWork.useGpuBonded)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                               "GPU device stream manager should be valid in order to use GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            fr->listedForcesGpu = std::make_unique<ListedForcesGpu>(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        mdAtoms = makeMDAtoms(fplog, mtop, *inputrec, thisRankHasPmeGpuTask);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        if (globalState && thisRankHasPmeGpuTask)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            // PME on GPU without DD or on a separate PME rank, and because the local state pointer
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    PmeGpuProgramStorage pmeGpuProgram;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    if (thisRankHasPmeGpuTask)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                "GPU device stream manager should be initialized in order to use GPU for PME.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                           "GPU device should be initialized in order to use GPU for PME.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        pmeGpuProgram = buildPmeGpuProgram(deviceStreamManager->context());
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                GMX_RELEASE_ASSERT(!runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                   "Device stream manager should be valid in order to use GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                        !runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                        "GPU PME stream should be valid in order to use GPU version of PME.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                const DeviceContext* deviceContext = runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                        runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                                       pmeGpuProgram.get(),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        if (runScheduleWork.simulationWork.useGpuFBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            fr->gpuForceReduction[gmx::AtomLocality::Local] = std::make_unique<gmx::GpuForceReduction>(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            fr->gpuForceReduction[gmx::AtomLocality::NonLocal] = std::make_unique<gmx::GpuForceReduction>(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        std::unique_ptr<gmx::StatePropagatorDataGpu> stateGpu;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        if (gpusWereDetected && gmx::needStateGpu(runScheduleWork.simulationWork))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            GpuApiCallBehavior transferKind =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                            ? GpuApiCallBehavior::Async
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                            : GpuApiCallBehavior::Sync;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                               "GPU device stream manager should be initialized to use GPU.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            stateGpu = std::make_unique<gmx::StatePropagatorDataGpu>(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                    *deviceStreamManager, transferKind, pme_gpu_get_block_size(fr->pmedata), wcycle.get());
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            fr->stateGpu = stateGpu.get();
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:          /* set GPU device id */
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:             plumed_cmd(plumedmain,"setGpuDeviceId", &deviceId);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:          if(useGpuForUpdate) {
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                        "This simulation is resident on GPU (-update gpu)\n"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        if (fr->pmePpCommGpu)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            // destroy object since it is no longer required. (This needs to be done while the GPU context still exists.)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            fr->pmePpCommGpu.reset();
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:                    runScheduleWork.simulationWork.useGpuPmePpCommunication,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    // before we destroy the GPU context(s)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    // Pinned buffers are associated with contexts in CUDA.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    // As soon as we destroy GPU contexts after mdrunner() exits, these lines should go.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    mdModules_.reset(nullptr); // destruct force providers here as they might also use the GPU
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    fr.reset(nullptr);         // destruct forcerec before gpu
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        /* stop the GPU profiler (only CUDA) */
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        stopGpuProfiler();
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:     * destroying the CUDA context as some tMPI ranks may be sharing
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:     * GPU and context.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:     * This is not a concern in OpenCL where we use one context per rank.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:     * Note: it is safe to not call the barrier on the ranks which do not use GPU,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:     * Note that this function needs to be called even if GPUs are not used
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:     * in this run because the PME ranks have no knowledge of whether GPUs
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:     * that it's not needed anymore (with a shared GPU run).
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    const bool haveDetectedOrForcedCudaAwareMpi =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:            (gmx::checkMpiCudaAwareSupport() == gmx::GpuAwareMpiStatus::Supported
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:             || gmx::checkMpiCudaAwareSupport() == gmx::GpuAwareMpiStatus::Forced);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:    if (!haveDetectedOrForcedCudaAwareMpi)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        // Don't reset GPU in case of GPU-AWARE MPI
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp:        // UCX creates GPU buffers which are cleaned-up as part of MPI_Finalize()
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:    // which compatible GPUs are availble for use, or to select a GPU
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:        hw_opt.userGpuTaskAssignment = userGpuTaskAssignment;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:        const char* env = getenv("GMX_GPU_ID");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:                gmx_fatal(FARGS, "GMX_GPU_ID and -gpu_id can not be used at the same time");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:        env = getenv("GMX_GPUTASKS");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:            if (!hw_opt.userGpuTaskAssignment.empty())
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:                gmx_fatal(FARGS, "GMX_GPUTASKS and -gputasks can not be used at the same time");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:            hw_opt.userGpuTaskAssignment = env;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:        if (!hw_opt.devicesSelectedByUser.empty() && !hw_opt.userGpuTaskAssignment.empty())
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:            gmx_fatal(FARGS, "-gpu_id and -gputasks cannot be used at the same time");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:    // which compatible GPUs are availble for use, or to select a GPU
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:        hw_opt.userGpuTaskAssignment = userGpuTaskAssignment;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:        const char* env = getenv("GMX_GPU_ID");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:                gmx_fatal(FARGS, "GMX_GPU_ID and -gpu_id can not be used at the same time");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:        env = getenv("GMX_GPUTASKS");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:            if (!hw_opt.userGpuTaskAssignment.empty())
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:                gmx_fatal(FARGS, "GMX_GPUTASKS and -gputasks can not be used at the same time");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:            hw_opt.userGpuTaskAssignment = env;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:        if (!hw_opt.devicesSelectedByUser.empty() && !hw_opt.userGpuTaskAssignment.empty())
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:            gmx_fatal(FARGS, "-gpu_id and -gputasks cannot be used at the same time");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* nbpu_opt_choices[5]    = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* pme_opt_choices[5]     = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* pme_fft_opt_choices[5] = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* bonded_opt_choices[5]  = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* update_opt_choices[5]  = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* userGpuTaskAssignment  = "";
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:        { "-gpu_id",
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:          "List of unique GPU device IDs available to use" },
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:        { "-gputasks",
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:          { &userGpuTaskAssignment },
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:          "List of GPU device IDs, mapping each PP task on each node to a device" },
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:          "Optimize PME load between PP/PME ranks or GPU/CPU" },
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* nbpu_opt_choices[5]    = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* pme_opt_choices[5]     = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* pme_fft_opt_choices[5] = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* bonded_opt_choices[5]  = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* update_opt_choices[5]  = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* userGpuTaskAssignment  = "";
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:        { "-gpu_id",
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:          "List of unique GPU device IDs available to use" },
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:        { "-gputasks",
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:          { &userGpuTaskAssignment },
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:          "List of GPU device IDs, mapping each PP task on each node to a device" },
patches/gromacs-2022.5.diff/src/gromacs/mdrun/legacymdrunoptions.h:          "Optimize PME load between PP/PME ranks or GPU/CPU" },
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/ewald/pme_gpu_program.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/ewald/pme_pp_comm_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/gpu_utils/device_stream_manager.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/mdlib/gpuforcereduction.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/taskassignment/decidegpuusage.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/taskassignment/usergpuids.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/timing/gpu_timing.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed: * the GPU communication flags are set to false in non-tMPI and non-CUDA builds.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed: * \param[in]  useGpuForNonbonded   True if the nonbonded task is offloaded in this run.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                         const bool           useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    devFlags.enableGpuBufferOps = (GMX_GPU_CUDA || GMX_GPU_SYCL) && useGpuForNonbonded
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                  && (getenv("GMX_USE_GPU_BUFFER_OPS") != nullptr);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    devFlags.forceGpuUpdateDefault = (getenv("GMX_FORCE_UPDATE_DEFAULT_GPU") != nullptr) || GMX_FAHCORE;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Flag use to enable GPU-aware MPI depenendent features such PME GPU decomposition
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // GPU-aware MPI is marked available if it has been detected by GROMACS or detection fails but
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    devFlags.canUseGpuAwareMpi = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Direct GPU comm path is being used with GPU-aware MPI
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // make sure underlying MPI implementation is GPU-aware
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (GMX_LIB_MPI && GMX_GPU_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // Allow overriding the detection for GPU-aware MPI
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        GpuAwareMpiStatus gpuAwareMpiStatus = checkMpiCudaAwareSupport();
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        const bool        forceGpuAwareMpi  = gpuAwareMpiStatus == GpuAwareMpiStatus::Forced;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        const bool haveDetectedGpuAwareMpi  = gpuAwareMpiStatus == GpuAwareMpiStatus::Supported;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (getenv("GMX_FORCE_CUDA_AWARE_MPI") != nullptr)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "GMX_FORCE_CUDA_AWARE_MPI environment variable is inactive. "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "Please use GMX_FORCE_GPU_AWARE_MPI instead.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        devFlags.canUseGpuAwareMpi = haveDetectedGpuAwareMpi || forceGpuAwareMpi;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (getenv("GMX_ENABLE_DIRECT_GPU_COMM") != nullptr)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            if (!haveDetectedGpuAwareMpi && forceGpuAwareMpi)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                // GPU-aware support not detected in MPI library but, user has forced it's use
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "This run has forced use of 'GPU-aware MPI', ie. 'CUDA-aware MPI'. "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "However, GROMACS cannot determine if underlying MPI is GPU-aware. "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "GROMACS recommends use of latest OpenMPI version for GPU-aware "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "GMX_FORCE_GPU_AWARE_MPI environment variable.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            if (devFlags.canUseGpuAwareMpi)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "GMX_ENABLE_DIRECT_GPU_COMM environment variable detected, "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "enabling direct GPU communication using GPU-aware MPI.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "GPU-aware MPI was not detected, will not use direct GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "for GPU-aware support. If you are certain about GPU-aware support "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "GMX_FORCE_GPU_AWARE_MPI environment variable. Note that such "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "support is often called \"CUDA-aware MPI.\"");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        else if (haveDetectedGpuAwareMpi)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // GPU-aware MPI was detected, let the user know that using it may improve performance
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "GPU-aware MPI detected, but by default GROMACS will not "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "make use the direct GPU communication capabilities of MPI. "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "the GMX_ENABLE_DIRECT_GPU_COMM environment variable.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (getenv("GMX_FORCE_GPU_AWARE_MPI") != nullptr)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // Cannot force use of GPU-aware MPI in this build configuration
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "A CUDA build with an external MPI library is required in order to "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "benefit from GMX_FORCE_GPU_AWARE_MPI. That environment variable is "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (devFlags.enableGpuBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        "This run uses the 'GPU buffer ops' feature, enabled by the "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        "GMX_USE_GPU_BUFFER_OPS environment variable.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (devFlags.forceGpuUpdateDefault)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        "This run will default to '-update gpu' as requested by the "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        "GMX_FORCE_UPDATE_DEFAULT_GPU environment variable.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // PME decomposition is supported only with CUDA-backend in mixed mode
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // CUDA-backend also needs GPU-aware MPI support for decomposition to work
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool pmeGpuDecompositionRequested =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            (pmeRunMode == PmeRunMode::GPU || pmeRunMode == PmeRunMode::Mixed)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool pmeGpuDecompositionSupported =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            (devFlags.canUseGpuAwareMpi && pmeRunMode == PmeRunMode::Mixed);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool forcePmeGpuDecomposition = getenv("GMX_GPU_PME_DECOMPOSITION") != nullptr;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (pmeGpuDecompositionSupported && pmeGpuDecompositionRequested)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // PME decomposition is supported only when it is forced using GMX_GPU_PME_DECOMPOSITION
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (forcePmeGpuDecomposition)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "This run has requested the 'GPU PME decomposition' feature, enabled "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "by the GMX_GPU_PME_DECOMPOSITION environment variable. "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                      "Multiple PME tasks were required to run on GPUs, "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                      "Use GMX_GPU_PME_DECOMPOSITION environment variable to enable it.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (!pmeGpuDecompositionSupported && pmeGpuDecompositionRequested)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                  "PME tasks were required to run on GPUs, but that is not implemented with "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    devFlags.enableGpuPmeDecomposition =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            forcePmeGpuDecomposition && pmeGpuDecompositionRequested && pmeGpuDecompositionSupported;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                  bool                makeGpuPairList,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                (makeGpuPairList ? ListSetupType::Gpu : ListSetupType::CpuSimdWhenSupported);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        increaseNstlist(fplog, cr, ir, nstlist_cmdline, &mtop, box, makeGpuPairList, cpuinfo);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:/*! \brief Return whether GPU acceleration of nonbondeds is supported with the given settings.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:static bool gpuAccelerationOfNonbondedIsUseful(const MDLogger&   mdlog,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool        gpuIsUseful = true;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        /* The GPU code does not support more than one energy group.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:         * If the user requested GPUs explicitly, a fatal error is given later.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gpuIsUseful = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "Multiple energy groups is not implemented for GPUs, falling back to the CPU. "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "For better performance, run on the GPU without energy groups and then do "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    /* There are resource handling issues in the GPU code paths with MTS on anything else than only
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gpuIsUseful = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                "Multiple time stepping is only supported with GPUs when MTS is only applied to %s "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gpuIsUseful = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        warning     = "TPI is not implemented for GPUs.";
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (!gpuIsUseful && issueWarning)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    return gpuIsUseful;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    else if (strncmp(optionString, "gpu", 3) == 0)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        returnValue = TaskTarget::Gpu;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        auto* nbnxn_gpu_timings =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                (nbv != nullptr && nbv->useGpu()) ? Nbnxm::gpu_get_timings(nbv->gpu_nbv) : nullptr;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gmx_wallclock_gpu_pme_t pme_gpu_timings = {};
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (pme_gpu_task_enabled(pme))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            pme_gpu_get_timings(pme, &pme_gpu_timings);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        nbnxn_gpu_timings,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        &pme_gpu_timings);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    EmulateGpuNonbonded emulateGpuNonbonded =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            (getenv("GMX_EMULATE_GPU") != nullptr ? EmulateGpuNonbonded::Yes : EmulateGpuNonbonded::No);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    std::vector<int> userGpuTaskAssignment;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        userGpuTaskAssignment = parseUserTaskAssignmentString(hw_opt.userGpuTaskAssignment);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        bool useGpuForNonbonded = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        bool useGpuForPme       = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // the number of GPUs to choose the number of ranks.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            auto canUseGpuForNonbonded = buildSupportsNonbondedOnGpu(nullptr);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            useGpuForNonbonded         = decideWhetherToUseGpusForNonbondedWithThreadMpi(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    userGpuTaskAssignment,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    emulateGpuNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    canUseGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    gpuAccelerationOfNonbondedIsUseful(mdlog, *inputrec, GMX_THREAD_MPI, doRerun),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            useGpuForPme = decideWhetherToUseGpusForPmeWithThreadMpi(useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                                     userGpuTaskAssignment,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                useGpuForPme,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Note that when bonded interactions run on a GPU they always run
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuForNonbonded = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuForPme       = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuForBonded    = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuForUpdate    = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool gpusWereDetected   = hwinfo_->ngpu_compatible_tot > 0;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // It's possible that there are different numbers of GPUs on
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        auto canUseGpuForNonbonded = buildSupportsNonbondedOnGpu(nullptr);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuForNonbonded         = decideWhetherToUseGpusForNonbonded(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                userGpuTaskAssignment,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                emulateGpuNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                canUseGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                gpuAccelerationOfNonbondedIsUseful(mdlog, *inputrec, !GMX_THREAD_MPI, doRerun),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                gpusWereDetected);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuForPme    = decideWhetherToUseGpusForPme(useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                    userGpuTaskAssignment,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                    gpusWereDetected);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuForBonded = decideWhetherToUseGpusForBonded(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuForNonbonded, useGpuForPme, bondedTarget, *inputrec, mtop, domdecOptions.numPmeRanks, gpusWereDetected);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const PmeRunMode pmeRunMode = determinePmeRunMode(useGpuForPme, pmeFftTarget, *inputrec);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            mdlog, useGpuForNonbonded, pmeRunMode, cr->sizeOfDefaultCommunicator, domdecOptions.numPmeRanks);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                || (!useGpuForNonbonded && EEL_FULL(inputrec->coulombtype) && useDDWithSingleRank != 0)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                          useGpuForNonbonded || (emulateGpuNonbonded == EmulateGpuNonbonded::Yes),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuForUpdate = decideWhetherToUseGpuForUpdate(useDomainDecomposition,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                         useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                         gpusWereDetected,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool canUseDirectGpuComm = decideWhetherDirectGpuCommunicationCanBeUsed(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuDirectHalo = false;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (useGpuForNonbonded)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // domdecOptions.numPmeRanks == -1 results in 0 separate PME ranks when useGpuForNonbonded is true.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuDirectHalo = decideWhetherToUseGpuForHalo(havePPDomainDecomposition,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                        useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                        canUseDirectGpuComm,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // The DD builder will disable useGpuDirectHalo if the Y or Z component of any domain is
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // smaller than twice the communication distance, since GPU-direct communication presently
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // perform well on multiple GPUs in any case, but it is important that our core functionality
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // (in particular for testing) does not break depending on GPU direct communication being enabled.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuForPme,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuForUpdate,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                &useGpuDirectHalo,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                devFlags.enableGpuPmeDecomposition);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    GpuTaskAssignments gpuTaskAssignments = GpuTaskAssignmentsBuilder::build(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            userGpuTaskAssignment,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            useGpuForPme,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    DeviceInformation* deviceInfo = gpuTaskAssignments.initDevice(&deviceId);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // TODO Pass the GPU streams to ddBuilder to use in buffer
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool useGpuPmeDecomposition = numPmeDomains.x * numPmeDomains.y > 1 && useGpuForPme;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    GMX_RELEASE_ASSERT(!useGpuPmeDecomposition || devFlags.enableGpuPmeDecomposition,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                       "GPU PME decomposition works only in the cases where it is supported");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuForNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuForBonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuForUpdate,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuDirectHalo,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              canUseDirectGpuComm,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuPmeDecomposition);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (isSimulationMasterRank && GMX_GPU_SYCL)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        bool                      haveAnyGpuWork = simWorkload.useGpuPme || simWorkload.useGpuBonded
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                              || simWorkload.useGpuNonbonded || simWorkload.useGpuUpdate;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (haveAnyGpuWork)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "\nNOTE: SYCL GPU support in GROMACS is still new and less tested than "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    gpuTaskAssignments.reportGpuUsage(mdlog, printHostName, pmeRunMode, runScheduleWork.simulationWork);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        const bool useGpuTiming = decideGpuTimingsUsage();
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                *deviceInfo, havePPDomainDecomposition(cr), runScheduleWork.simulationWork, useGpuTiming);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gpuTaskAssignments.logPerformanceHints(mdlog, numAvailableDevices);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    check_resource_division_efficiency(hwinfo_, gpuTaskAssignments.thisRankHasAnyGpuTask(), cr, mdlog);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Enable Peer access between GPUs where available
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // any of the GPU communication features are active.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        && (runScheduleWork.simulationWork.useGpuHaloExchange
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            || runScheduleWork.simulationWork.useGpuPmePpCommunication))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        setupGpuDevicePeerAccess(gpuTaskAssignments.deviceIdsAssigned(), mdlog);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool thisRankHasPmeGpuTask = gpuTaskAssignments.thisRankHasPmeGpuTask();
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (runScheduleWork.simulationWork.useGpuPmePpCommunication && !thisRankHasDuty(cr, DUTY_PME))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "GPU device stream manager should be valid in order to use PME-PP direct "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "GPU PP-PME stream should be valid in order to use GPU PME-PP direct "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->pmePpCommGpu = std::make_unique<gmx::PmePpCommGpu>(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                        runScheduleWork.simulationWork.useGpuNonbonded,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // TODO: Move the logic below to a GPU bonded builder
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (runScheduleWork.simulationWork.useGpuBonded)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                               "GPU device stream manager should be valid in order to use GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->listedForcesGpu = std::make_unique<ListedForcesGpu>(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        mdAtoms = makeMDAtoms(fplog, mtop, *inputrec, thisRankHasPmeGpuTask);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (globalState && thisRankHasPmeGpuTask)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // PME on GPU without DD or on a separate PME rank, and because the local state pointer
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    PmeGpuProgramStorage pmeGpuProgram;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (thisRankHasPmeGpuTask)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                "GPU device stream manager should be initialized in order to use GPU for PME.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                           "GPU device should be initialized in order to use GPU for PME.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        pmeGpuProgram = buildPmeGpuProgram(deviceStreamManager->context());
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                GMX_RELEASE_ASSERT(!runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                   "Device stream manager should be valid in order to use GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        !runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        "GPU PME stream should be valid in order to use GPU version of PME.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                const DeviceContext* deviceContext = runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                       pmeGpuProgram.get(),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (runScheduleWork.simulationWork.useGpuFBufferOps)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->gpuForceReduction[gmx::AtomLocality::Local] = std::make_unique<gmx::GpuForceReduction>(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->gpuForceReduction[gmx::AtomLocality::NonLocal] = std::make_unique<gmx::GpuForceReduction>(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        std::unique_ptr<gmx::StatePropagatorDataGpu> stateGpu;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (gpusWereDetected && gmx::needStateGpu(runScheduleWork.simulationWork))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            GpuApiCallBehavior transferKind =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            ? GpuApiCallBehavior::Async
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            : GpuApiCallBehavior::Sync;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                               "GPU device stream manager should be initialized to use GPU.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            stateGpu = std::make_unique<gmx::StatePropagatorDataGpu>(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    *deviceStreamManager, transferKind, pme_gpu_get_block_size(fr->pmedata), wcycle.get());
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->stateGpu = stateGpu.get();
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (fr->pmePpCommGpu)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // destroy object since it is no longer required. (This needs to be done while the GPU context still exists.)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->pmePpCommGpu.reset();
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    runScheduleWork.simulationWork.useGpuPmePpCommunication,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // before we destroy the GPU context(s)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Pinned buffers are associated with contexts in CUDA.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // As soon as we destroy GPU contexts after mdrunner() exits, these lines should go.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    mdModules_.reset(nullptr); // destruct force providers here as they might also use the GPU
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    fr.reset(nullptr);         // destruct forcerec before gpu
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        /* stop the GPU profiler (only CUDA) */
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        stopGpuProfiler();
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * destroying the CUDA context as some tMPI ranks may be sharing
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * GPU and context.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * This is not a concern in OpenCL where we use one context per rank.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * Note: it is safe to not call the barrier on the ranks which do not use GPU,
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * Note that this function needs to be called even if GPUs are not used
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * in this run because the PME ranks have no knowledge of whether GPUs
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:     * that it's not needed anymore (with a shared GPU run).
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool haveDetectedOrForcedCudaAwareMpi =
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:            (gmx::checkMpiCudaAwareSupport() == gmx::GpuAwareMpiStatus::Supported
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:             || gmx::checkMpiCudaAwareSupport() == gmx::GpuAwareMpiStatus::Forced);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (!haveDetectedOrForcedCudaAwareMpi)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // Don't reset GPU in case of GPU-AWARE MPI
patches/gromacs-2022.5.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // UCX creates GPU buffers which are cleaned-up as part of MPI_Finalize()
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/gpu_utils/device_stream_manager.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/mdlib/update_constrain_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    /* PME load balancing data for GPU kernels */
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    const bool  useGpuForPme       = simulationWork.useGpuPme;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    const bool  useGpuForNonbonded = simulationWork.useGpuNonbonded;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    const bool  useGpuForUpdate    = simulationWork.useGpuUpdate;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                                 useGpuForPme);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                   (simulationWork.useGpuFBufferOps || useGpuForUpdate) ? PinningPolicy::PinnedIfSupported
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    std::unique_ptr<UpdateConstrainGpu> integrator;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    StatePropagatorDataGpu* stateGpu = fr->stateGpu;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    if (useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "groups if using GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "SHAKE is not supported with GPU update.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        GMX_RELEASE_ASSERT(useGpuForPme || (useGpuForNonbonded && simulationWork.useGpuXBufferOps),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "the GPU to use GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Only the md integrator is supported with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                "Nose-Hoover temperature coupling is not supported with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                "with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Virtual sites are not supported with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Essential dynamics is not supported with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Constraints pulling is not supported with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Orientation restraints are not supported with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                "Free energy perturbation of masses and constraints are not supported with the GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    .appendText("Updating coordinates and applying constraints on the GPU.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            GMX_LOG(mdlog.info).asParagraph().appendText("Updating coordinates on the GPU.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Device stream manager should be initialized in order to use GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                "Update stream should be initialized in order to use GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        integrator = std::make_unique<UpdateConstrainGpu>(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        stateGpu->setXUpdatedOnDeviceEvent(integrator->xUpdatedOnDeviceEvent());
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    if (useGpuForPme || simulationWork.useGpuXBufferOps || useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:    if (useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:     * Disable PME tuning with GPU PME decomposition */
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                && ir->cutoff_scheme != CutoffScheme::Group && !simulationWork.useGpuPmeDecomposition);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                &pme_loadbal, cr, mdlog, *ir, state->box, *fr->ic, *fr->nbv, fr->pmedata, fr->nbv->useGpu());
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate && !bFirstStep)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            /* PME grid + cut-off optimization with GPUs or PME nodes */
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                           simulationWork.useGpuPmePpCommunication);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        // On search steps, when doing the update on the GPU, copy
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (useGpuForUpdate && bNS && !bFirstStep && !bExchanged)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->copyVelocitiesFromGpu(state->v, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            // the GPU Update object should be informed
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate && (bMasterState || bExchanged))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        // Allocate or re-size GPU halo exchange object, if necessary
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (bNS && simulationWork.havePpDomainDecomposition && simulationWork.useGpuHaloExchange)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                               "GPU device manager has to be initialized to use GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            constructGpuHaloExchange(*cr, *fr->deviceStreamManager, wcycle);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        // Copy coordinate from the GPU for the output/checkpointing if the update is offloaded and
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (useGpuForUpdate && !bNS && !runScheduleWork->domainWork.haveCpuLocalForceWork
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (useGpuForUpdate && !bNS
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->copyVelocitiesFromGpu(state->v, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        // Copy forces for the output if the forces were reduced on the GPU (not the case on virial steps)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        // and update is offloaded hence forces are kept on the GPU for update and have not been
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        //       when the forces are ready on the GPU -- the same synchronizer should be used as the one
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        //       prior to GPU update.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (runScheduleWork->stepWork.useGpuFBufferOps && (simulationWork.useGpuUpdate && !vsite)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->copyForcesFromGpu(f.view().force(), AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (!useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            GMX_ASSERT(!useGpuForUpdate, "GPU update is not supported with VVAK integrator.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    integrator->set(stateGpu->getCoordinates(),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                                    stateGpu->getVelocities(),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                                    stateGpu->getForces(),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    // Copy data to the GPU after buffers might have been reinitialized
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->copyVelocitiesToGpu(state->v, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                // Copy x to the GPU unless we have already transferred in do_force().
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                // We transfer in do_force() if a GPU force task requires x (PME or x buffer ops).
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                if (!(runScheduleWork->stepWork.haveGpuPmeOnThisRank
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                      || runScheduleWork->stepWork.useGpuXBufferOps))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->copyCoordinatesToGpu(state->x, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->consumeCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                if ((simulationWork.useGpuPme && simulationWork.useCpuPmePpCommunication)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    || (!runScheduleWork->stepWork.useGpuFBufferOps))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    // rest of the forces computed on the GPU, so the final forces have to be copied
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    // back to the GPU. Or the buffer ops were not offloaded this step, so the
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->copyForcesToGpu(f.view().force(), AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                integrator->integrate(stateGpu->getLocalForcesReadyOnDeviceEvent(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->copyVelocitiesFromGpu(state->v, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                    if (useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->resetCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->copyCoordinatesToGpu(state->x, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->waitCoordinatesCopiedToDevice(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:                            stateGpu->copyVelocitiesToGpu(state->v, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        bool scaleCoordinates = !useGpuForUpdate || bDoReplEx;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (useGpuForUpdate
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp.preplumed:        pme_loadbal_done(pme_loadbal, fplog, mdlog, fr->nbv->useGpu());
patches/gromacs-2022.5.diff/src/gromacs/mdrun/rerun.cpp.preplumed:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/rerun.cpp.preplumed:                                 runScheduleWork->simulationWork.useGpuPme);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/rerun.cpp:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/rerun.cpp:                                 runScheduleWork->simulationWork.useGpuPme);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/gpu_utils/device_stream_manager.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/mdlib/update_constrain_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:    /* PME load balancing data for GPU kernels */
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:    const bool  useGpuForPme       = simulationWork.useGpuPme;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:    const bool  useGpuForNonbonded = simulationWork.useGpuNonbonded;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:    const bool  useGpuForUpdate    = simulationWork.useGpuUpdate;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                                 useGpuForPme);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                   (simulationWork.useGpuFBufferOps || useGpuForUpdate) ? PinningPolicy::PinnedIfSupported
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:    std::unique_ptr<UpdateConstrainGpu> integrator;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:    StatePropagatorDataGpu* stateGpu = fr->stateGpu;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:    if (useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                           "groups if using GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                           "SHAKE is not supported with GPU update.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        GMX_RELEASE_ASSERT(useGpuForPme || (useGpuForNonbonded && simulationWork.useGpuXBufferOps),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                           "the GPU to use GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                           "Only the md integrator is supported with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                "Nose-Hoover temperature coupling is not supported with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                "with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                           "Virtual sites are not supported with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                           "Essential dynamics is not supported with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                           "Constraints pulling is not supported with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                           "Orientation restraints are not supported with the GPU update.\n");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                "Free energy perturbation of masses and constraints are not supported with the GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    .appendText("Updating coordinates and applying constraints on the GPU.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            GMX_LOG(mdlog.info).asParagraph().appendText("Updating coordinates on the GPU.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                           "Device stream manager should be initialized in order to use GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                "Update stream should be initialized in order to use GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        integrator = std::make_unique<UpdateConstrainGpu>(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        stateGpu->setXUpdatedOnDeviceEvent(integrator->xUpdatedOnDeviceEvent());
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:    if (useGpuForPme || simulationWork.useGpuXBufferOps || useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:    if (useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:     * Disable PME tuning with GPU PME decomposition */
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                && ir->cutoff_scheme != CutoffScheme::Group && !simulationWork.useGpuPmeDecomposition);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                &pme_loadbal, cr, mdlog, *ir, state->box, *fr->ic, *fr->nbv, fr->pmedata, fr->nbv->useGpu());
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate && !bFirstStep)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            /* PME grid + cut-off optimization with GPUs or PME nodes */
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                           simulationWork.useGpuPmePpCommunication);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        // On search steps, when doing the update on the GPU, copy
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        if (useGpuForUpdate && bNS && !bFirstStep && !bExchanged)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->copyVelocitiesFromGpu(state->v, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            // the GPU Update object should be informed
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate && (bMasterState || bExchanged))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        // Allocate or re-size GPU halo exchange object, if necessary
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        if (bNS && simulationWork.havePpDomainDecomposition && simulationWork.useGpuHaloExchange)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                               "GPU device manager has to be initialized to use GPU "
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            constructGpuHaloExchange(*cr, *fr->deviceStreamManager, wcycle);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        // Copy coordinate from the GPU for the output/checkpointing if the update is offloaded and
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        if (useGpuForUpdate && !bNS && !runScheduleWork->domainWork.haveCpuLocalForceWork
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        if (useGpuForUpdate && !bNS
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->copyVelocitiesFromGpu(state->v, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        // Copy forces for the output if the forces were reduced on the GPU (not the case on virial steps)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        // and update is offloaded hence forces are kept on the GPU for update and have not been
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        //       when the forces are ready on the GPU -- the same synchronizer should be used as the one
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        //       prior to GPU update.
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        if (runScheduleWork->stepWork.useGpuFBufferOps && (simulationWork.useGpuUpdate && !vsite)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->copyForcesFromGpu(f.view().force(), AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        if (!useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            GMX_ASSERT(!useGpuForUpdate, "GPU update is not supported with VVAK integrator.");
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    integrator->set(stateGpu->getCoordinates(),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                                    stateGpu->getVelocities(),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                                    stateGpu->getForces(),
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    // Copy data to the GPU after buffers might have been reinitialized
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->copyVelocitiesToGpu(state->v, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                // Copy x to the GPU unless we have already transferred in do_force().
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                // We transfer in do_force() if a GPU force task requires x (PME or x buffer ops).
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                if (!(runScheduleWork->stepWork.haveGpuPmeOnThisRank
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                      || runScheduleWork->stepWork.useGpuXBufferOps))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->copyCoordinatesToGpu(state->x, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->consumeCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                if ((simulationWork.useGpuPme && simulationWork.useCpuPmePpCommunication)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    || (!runScheduleWork->stepWork.useGpuFBufferOps))
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    // rest of the forces computed on the GPU, so the final forces have to be copied
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    // back to the GPU. Or the buffer ops were not offloaded this step, so the
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->copyForcesToGpu(f.view().force(), AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                integrator->integrate(stateGpu->getLocalForcesReadyOnDeviceEvent(
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->copyCoordinatesFromGpu(state->x, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->copyVelocitiesFromGpu(state->v, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                    if (useGpuForUpdate)
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->resetCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->copyCoordinatesToGpu(state->x, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->waitCoordinatesCopiedToDevice(AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:                            stateGpu->copyVelocitiesToGpu(state->v, AtomLocality::Local);
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        bool scaleCoordinates = !useGpuForUpdate || bDoReplEx;
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        if (useGpuForUpdate
patches/gromacs-2022.5.diff/src/gromacs/mdrun/md.cpp:        pme_loadbal_done(pme_loadbal, fplog, mdlog, fr->nbv->useGpu());
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:if (GMX_CLANG_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    include(gmxClangCudaUtils)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:set_property(GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:add_subdirectory(gpu_utils)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:# Mark some shared GPU implementation files to compile with CUDA if needed
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    get_property(CUDA_SOURCES GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    set_source_files_properties(${CUDA_SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    # Work around FindCUDA that prevents using target_link_libraries()
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    if (NOT GMX_CLANG_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        gmx_cuda_add_library(libgromacs ${LIBGROMACS_SOURCES})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    target_link_libraries(libgromacs PRIVATE ${CUDA_CUFFT_LIBRARIES})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_OPENCL)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:"An OpenCL build was requested with Visual Studio compiler, but GROMACS
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:clFFT to help with building for OpenCL, but that clFFT has not yet been
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:requires. Thus for now, OpenCL is not available with MSVC and the internal
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:# CUDA runtime headers
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_CUDA AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        set(GMX_CUDA_CLANG_FLAGS "${GMX_CUDA_CLANG_FLAGS} ${_compile_flag}")
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    if (GMX_CLANG_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:            get_source_file_property(_cuda_source_format ${_file} CUDA_SOURCE_PROPERTY_FORMAT)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:            if ("${_ext}" STREQUAL ".cu" OR _cuda_source_format)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:                gmx_compile_cuda_file_with_clang(${_file})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        get_property(CUDA_SOURCES GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        set_source_files_properties(${CUDA_SOURCES} PROPERTIES COMPILE_FLAGS ${GMX_CUDA_CLANG_FLAGS})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_SYCL)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:                      ${OpenCL_LIBRARIES}
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:                      $<BUILD_INTERFACE:gpu_utils>
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:# Technically, the user could want to do this for an OpenCL build
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:# using the CUDA runtime, but currently there's no reason to want to
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:if (INSTALL_CUDART_LIB) #can be set manual by user
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    if (GMX_GPU_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        foreach(CUDA_LIB ${CUDA_LIBRARIES})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:            string(REGEX MATCH "cudart" IS_CUDART ${CUDA_LIB})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:            if(IS_CUDART) #libcuda should not be installed
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:                file(GLOB CUDA_LIBS ${CUDA_LIB}*)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:                install(FILES ${CUDA_LIBS} DESTINATION
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        message(WARNING "INSTALL_CUDART_LIB only makes sense when configuring for CUDA support")
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:if(GMX_GPU_OPENCL)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        gpu_utils/vectype_ops.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        gpu_utils/device_utils.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        DESTINATION ${GMX_INSTALL_OCLDIR}/gromacs/gpu_utils
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernels.cl
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernel.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernel_pruneonly.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernels.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernels_fastgen.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernels_fastgen_add_twincut.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernel_utils.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_consts.h
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        DESTINATION ${GMX_INSTALL_OCLDIR}/gromacs/nbnxm/opencl
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        ewald/pme_gpu_calculate_splines.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:        ewald/pme_gpu_types.h
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_CLANG_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    include(gmxClangCudaUtils)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:set_property(GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:add_subdirectory(gpu_utils)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:# Mark some shared GPU implementation files to compile with CUDA if needed
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    get_property(CUDA_SOURCES GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    set_source_files_properties(${CUDA_SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    # Work around FindCUDA that prevents using target_link_libraries()
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    if (NOT GMX_CLANG_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        gmx_cuda_add_library(libgromacs ${LIBGROMACS_SOURCES})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    target_link_libraries(libgromacs PRIVATE ${CUDA_CUFFT_LIBRARIES})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_OPENCL)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:"An OpenCL build was requested with Visual Studio compiler, but GROMACS
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:clFFT to help with building for OpenCL, but that clFFT has not yet been
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:requires. Thus for now, OpenCL is not available with MSVC and the internal
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:# CUDA runtime headers
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_CUDA AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        set(GMX_CUDA_CLANG_FLAGS "${GMX_CUDA_CLANG_FLAGS} ${_compile_flag}")
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    if (GMX_CLANG_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:            get_source_file_property(_cuda_source_format ${_file} CUDA_SOURCE_PROPERTY_FORMAT)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:            if ("${_ext}" STREQUAL ".cu" OR _cuda_source_format)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:                gmx_compile_cuda_file_with_clang(${_file})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        get_property(CUDA_SOURCES GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        set_source_files_properties(${CUDA_SOURCES} PROPERTIES COMPILE_FLAGS ${GMX_CUDA_CLANG_FLAGS})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_SYCL)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:                      ${OpenCL_LIBRARIES}
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:                      $<BUILD_INTERFACE:gpu_utils>
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:# Technically, the user could want to do this for an OpenCL build
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:# using the CUDA runtime, but currently there's no reason to want to
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:if (INSTALL_CUDART_LIB) #can be set manual by user
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    if (GMX_GPU_CUDA)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        foreach(CUDA_LIB ${CUDA_LIBRARIES})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:            string(REGEX MATCH "cudart" IS_CUDART ${CUDA_LIB})
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:            if(IS_CUDART) #libcuda should not be installed
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:                file(GLOB CUDA_LIBS ${CUDA_LIB}*)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:                install(FILES ${CUDA_LIBS} DESTINATION
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        message(WARNING "INSTALL_CUDART_LIB only makes sense when configuring for CUDA support")
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:if(GMX_GPU_OPENCL)
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        gpu_utils/vectype_ops.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        gpu_utils/device_utils.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        DESTINATION ${GMX_INSTALL_OCLDIR}/gromacs/gpu_utils
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernels.cl
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernel.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernel_pruneonly.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernels.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernels_fastgen.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernels_fastgen_add_twincut.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernel_utils.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_consts.h
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        DESTINATION ${GMX_INSTALL_OCLDIR}/gromacs/nbnxm/opencl
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        ewald/pme_gpu_calculate_splines.clh
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:        ewald/pme_gpu_types.h
patches/gromacs-2022.5.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/namd-2.14.diff: 	$(CUDALIB) \
patches/namd-2.14.diff: #include "DeviceCUDA.h"
patches/namd-2.14.diff: #ifdef NAMD_CUDA
patches/namd-2.13.diff: 	$(CUDALIB) \
patches/namd-2.13.diff: #include "DeviceCUDA.h"
patches/namd-2.13.diff: #ifdef NAMD_CUDA
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/ewald/pme_coordinate_receiver_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/ewald/pme_pp_comm_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/nbnxm/nbnxm_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gromacs/timing/gpu_timing.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:#include "gpuforcereduction.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static const bool c_disableAlternatingWait = (getenv("GMX_DISABLE_ALTERNATING_GPU_WAIT") != nullptr);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   bool                  useGpuPmePpComms,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   bool                  receivePmeForceToGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gmx_pme_receive_f(fr->pmePpCommGpu.get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                      useGpuPmePpComms,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                      receivePmeForceToGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* GPU kernel launch overhead is already timed separately */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (!nbv->useGpu())
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Launch the prepare_step and spread stages of PME GPU.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param[in]  useMdGpuGraph        Whether MD GPU Graph is in use.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static inline void launchPmeGpuSpread(gmx_pme_t*            pmedata,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                      GpuEventSynchronizer* xReadyOnDevice,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                      bool                  useMdGpuGraph,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    wallcycle_start(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_prepare_computation(pmedata, box, wcycle, stepWork);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool                           useGpuDirectComm         = false;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gmx::PmeCoordinateReceiverGpu* pmeCoordinateReceiverGpu = nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_launch_spread(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            pmedata, xReadyOnDevice, wcycle, lambdaQ, useGpuDirectComm, pmeCoordinateReceiverGpu, useMdGpuGraph);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Launch the FFT and gather stages of PME GPU
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void launchPmeGpuFftAndGather(gmx_pme_t*               pmedata,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_launch_complex_transforms(pmedata, wcycle, stepWork);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_launch_gather(pmedata, wcycle, lambdaQ, stepWork.computeVirial);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * Blocks until PME GPU tasks are completed, and gets the output forces and virial/energy
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void pmeGpuWaitAndReduce(gmx_pme_t*               pme,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    pme_gpu_wait_and_reduce(pme, stepWork, wcycle, forceWithVirial, enerd, lambdaQ);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: *  Polling wait for either of the PME or nonbonded GPU tasks.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * Instead of a static order in waiting for GPU tasks, this function
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * one of the reductions, regardless of the GPU task completion order.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void alternatePmeNbGpuWaitReduce(nonbonded_verlet_t* nbv,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool isPmeGpuDone = false;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool isNbGpuDone  = false;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gmx::ArrayRef<const gmx::RVec> pmeGpuForces;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    while (!isPmeGpuDone || !isNbGpuDone)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!isPmeGpuDone)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GpuTaskCompletion completionType =
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    (isNbGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            isPmeGpuDone = pme_gpu_try_finish_task(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!isNbGpuDone)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GpuTaskCompletion completionType =
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    (isPmeGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // To get the wcycle call count right, when in GpuTaskCompletion::Check mode,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // GpuTaskCompletion::Wait mode the timing is expected to be done in the caller.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_start_nocount(wcycle, WallCycleCounter::WaitGpuNbL);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            isNbGpuDone = Nbnxm::gpu_try_finish_task(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    nbv->gpuNbv(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_stop(wcycle, WallCycleCounter::WaitGpuNbL);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (isNbGpuDone)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                wallcycle_increment_event_count(wcycle, WallCycleCounter::WaitGpuNbL);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        && (domainWork.haveCpuLocalForceWork || !stepWork.useGpuFBufferOps
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            || (havePpDomainDecomposition && !stepWork.useGpuFHalo)))
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/* \brief Launch end-of-step GPU tasks: buffer clearing and rolling pruning.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void launchGpuEndOfStepTasks(nonbonded_verlet_t*               nbv,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                    gmx::ListedForcesGpu*             listedForcesGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork.simulationWork.useGpuNonbonded && runScheduleWork.stepWork.computeNonbondedForces)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:         * clear kernel launches can leave the GPU idle while it could be running
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (nbv->isDynamicPruningStepGpu(step))
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            nbv->dispatchPruneKernelGpu(step);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* now clear the GPU outputs while we finish the step on the CPU */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        Nbnxm::gpu_clear_outputs(nbv->gpuNbv(), runScheduleWork.stepWork.computeVirial);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork.stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        bool gpuGraphWithSeparatePmeRank = false;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pme_gpu_reinit_computation(pmedata, gpuGraphWithSeparatePmeRank, wcycle);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork.domainWork.haveGpuBondedWork && runScheduleWork.stepWork.computeEnergy)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        listedForcesGpu->waitAccumulateEnergyTerms(enerd);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        listedForcesGpu->clearEnergies();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Compute the number of times the "local coordinates ready on device" GPU event will be used as a synchronization point.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * When some work is offloaded to GPU, force calculation should wait for the atom coordinates to
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * or from the GPU integration at the end of the previous step.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param pmeSendCoordinatesFromGpu Whether peer-to-peer communication is used for PME coordinates.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                                          bool pmeSendCoordinatesFromGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (pmeSendCoordinatesFromGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                       "GPU PME PP communications require having a separate PME rank");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // Event is consumed by gmx_pme_send_coordinates for GPU PME PP Communications
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // Event is consumed by launchPmeGpuSpread
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.computeNonbondedForces && stepWork.useGpuXBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // Event is consumed by convertCoordinatesGpu
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.useGpuXHalo)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // Event is consumed by communicateGpuHaloCoordinates
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.clearGpuFBufferEarly && simulationWork.useGpuUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Compute the number of times the "local forces ready on device" GPU event will be used as a synchronization point.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param useOrEmulateGpuNb Whether GPU non-bonded calculations are used or emulated.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param alternateGpuWait Whether alternating wait/reduce scheme is used.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                                          bool useOrEmulateGpuNb,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                                          bool alternateGpuWait)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool eventUsedInGpuForceReduction =
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:             || (simulationWork.havePpDomainDecomposition && !simulationWork.useGpuHaloExchange));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool gpuForceReductionUsed = useOrEmulateGpuNb && !alternateGpuWait && stepWork.useGpuFBufferOps
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (gpuForceReductionUsed && eventUsedInGpuForceReduction)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool gpuForceHaloUsed = simulationWork.havePpDomainDecomposition && stepWork.computeForces
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                            && stepWork.useGpuFHalo;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (gpuForceHaloUsed)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Setup for the local GPU force reduction:
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] stateGpu            GPU state propagator object
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] gpuForceReduction   GPU force reduction object
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] pmePpCommGpu        PME-PP GPU communication object
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void setupLocalGpuForceReduction(const gmx::MdrunScheduleWorkload& runScheduleWork,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        gmx::StatePropagatorDataGpu*      stateGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        gmx::GpuForceReduction*           gpuForceReduction,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        gmx::PmePpCommGpu*                pmePpCommGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "GPU force reduction is not compatible with MTS");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // (re-)initialize local GPU force reduction
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gpuForceReduction->reinit(stateGpu->getForces(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                              stateGpu->fReducedOnDevice(AtomLocality::Local));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gpuForceReduction->registerNbnxmForce(Nbnxm::gpu_get_f(nbv->gpuNbv()));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GpuEventSynchronizer*   pmeSynchronizer     = nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork.simulationWork.haveGpuPmeOnPpRank())
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pmeForcePtr = pme_gpu_get_device_f(pmedata);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            pmeSynchronizer     = pme_gpu_get_f_ready_synchronizer(pmedata);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    else if (runScheduleWork.simulationWork.useGpuPmePpCommunication)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pmeForcePtr = pmePpCommGpu->getGpuForceStagingPtr();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            pmeSynchronizer = pmePpCommGpu->getForcesReadySynchronizer();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        gpuForceReduction->registerRvecForce(pmeForcePtr);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            DeviceBuffer<uint64_t> forcesReadyNvshmemFlags = pmePpCommGpu->getGpuForcesSyncObj();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            gpuForceReduction->registerForcesReadyNvshmemFlags(forcesReadyNvshmemFlags);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!runScheduleWork.simulationWork.useGpuPmePpCommunication || GMX_THREAD_MPI)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GMX_ASSERT(pmeSynchronizer != nullptr, "PME force ready cuda event should not be NULL");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            gpuForceReduction->addDependency(pmeSynchronizer);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            && !runScheduleWork.simulationWork.useGpuHaloExchange))
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::Local));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (runScheduleWork.simulationWork.useGpuHaloExchange)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        gpuForceReduction->addDependency(dd->gpuHaloExchange[0][0]->getForcesReadyOnDeviceEvent());
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:/*! \brief Setup for the non-local GPU force reduction:
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] stateGpu            GPU state propagator object
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed: * \param [in] gpuForceReduction   GPU force reduction object
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:static void setupNonLocalGpuForceReduction(const gmx::MdrunScheduleWorkload& runScheduleWork,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                           gmx::StatePropagatorDataGpu*      stateGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                           gmx::GpuForceReduction*           gpuForceReduction,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // (re-)initialize non-local GPU force reduction
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gpuForceReduction->reinit(stateGpu->getForces(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                              stateGpu->fReducedOnDevice(AtomLocality::NonLocal));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gpuForceReduction->registerNbnxmForce(Nbnxm::gpu_get_f(nbv->gpuNbv()));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::NonLocal));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gmx::StatePropagatorDataGpu* stateGpu = fr->stateGpu;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (gmx::needStateGpu(simulationWork))
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->reinit(mdatoms.homenr,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.haveGpuPmeOnPpRank())
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        GMX_ASSERT(gmx::needStateGpu(simulationWork), "StatePropagatorDataGpu is needed");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // TODO: This should be moved into PME setup function ( pme_gpu_prepare_computation(...) )
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pme_gpu_set_device_x(fr->pmedata, stateGpu->getCoordinates());
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* initialize the GPU nbnxm atom data and bonded data structures */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // Note: cycle counting only nononbondeds, GPU listed forces counts internally
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        Nbnxm::gpu_init_atomdata(nbv->gpuNbv(), &nbv->nbat());
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (fr->listedForcesGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:             * interactions to the GPU, where the grid order is
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    nbv->getGridIndices(), top.idef, Nbnxm::gpuGetNBAtomData(nbv->gpuNbv()));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuXBufferOpsWhenAllowed)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        nbv->atomdata_init_copy_x_to_nbat_x_gpu();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuFBufferOpsWhenAllowed)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // with MPI, direct GPU communication, and separate PME ranks we need
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        bool delaySetupLocalGpuForceReduction = GMX_MPI && simulationWork.useGpuPmePpCommunication;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!delaySetupLocalGpuForceReduction)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            setupLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        stateGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        fr->gpuForceReduction[gmx::AtomLocality::Local].get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        fr->pmePpCommGpu.get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            setupNonLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                           stateGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                           fr->gpuForceReduction[gmx::AtomLocality::NonLocal].get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // TODO refactor this GPU halo exchange re-initialisation
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // to location in do_md where GPU halo exchange is
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // constructed at partitioning, after above stateGpu
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (simulationWork.useGpuHaloExchange)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            reinitGpuHaloExchange(*cr, stateGpu->getCoordinates(), stateGpu->getForces());
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    gmx::StatePropagatorDataGpu* stateGpu = fr->stateGpu;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    const bool pmeSendCoordinatesFromGpu =
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            simulationWork.useGpuPmePpCommunication && !stepWork.doNeighborSearch;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    auto* localXReadyOnDevice = (stepWork.haveGpuPmeOnThisRank || stepWork.useGpuXBufferOps
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 || simulationWork.useGpuUpdate || pmeSendCoordinatesFromGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        ? stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.clearGpuFBufferEarly)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // GPU Force halo exchange will set a subset of local atoms with remote non-local data.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // which is satisfied when localXReadyOnDevice has been marked for GPU update case.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        GpuEventSynchronizer* dependency = simulationWork.useGpuUpdate ? localXReadyOnDevice : nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->clearForcesOnGpu(AtomLocality::Local, dependency);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GMX_ASSERT(simulationWork.useGpuHaloExchange
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                       == ((cr->dd != nullptr) && (!cr->dd->gpuHaloExchange[0].empty())),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "The GPU halo exchange is active, but it has not been constructed.");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    bool gmx_used_in_debug haveCopiedXFromGpu = false;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // Copy coordinate from the GPU if update is on the GPU and there
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->copyCoordinatesFromGpu(x.unpaddedArrayRef(), AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        haveCopiedXFromGpu = true;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.haveGpuPmeOnThisRank || stepWork.useGpuXBufferOps || pmeSendCoordinatesFromGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        GMX_ASSERT(stateGpu != nullptr, "stateGpu should not be null");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        simulationWork, stepWork, pmeSendCoordinatesFromGpu);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!simulationWork.useGpuUpdate || stepWork.doNeighborSearch)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        else if (simulationWork.useGpuUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            stateGpu->setXUpdatedOnDeviceEventExpectedConsumptionCount(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!pmeSendCoordinatesFromGpu && !stepWork.doNeighborSearch && simulationWork.useGpuUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        const bool reinitGpuPmePpComms =
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                simulationWork.useGpuPmePpCommunication && stepWork.doNeighborSearch;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 simulationWork.useGpuPmePpCommunication,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 reinitGpuPmePpComms,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 pmeSendCoordinatesFromGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 stepWork.useGpuPmeFReduction,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 pmeSendCoordinatesFromGpu ? localXReadyOnDevice : nullptr,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                 simulationWork.useMdGpuGraph,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuFBufferOpsWhenAllowed && stepWork.doNeighborSearch)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // with MPI, direct GPU communication, and separate PME ranks we need
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        bool doSetupLocalGpuForceReduction = GMX_MPI && simulationWork.useGpuPmePpCommunication;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (doSetupLocalGpuForceReduction)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            setupLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        stateGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        fr->gpuForceReduction[gmx::AtomLocality::Local].get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                        fr->pmePpCommGpu.get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        launchPmeGpuSpread(fr->pmedata,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                           simulationWork.useMdGpuGraph,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.useGpuXBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GMX_ASSERT(stateGpu, "stateGpu should be valid when buffer ops are offloaded");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            nbv->convertCoordinatesGpu(AtomLocality::Local, stateGpu->getCoordinates(), localXReadyOnDevice);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (simulationWork.useGpuUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                GMX_ASSERT(stateGpu, "need a valid stateGpu object");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuNonbonded && (stepWork.computeNonbondedForces || domainWork.haveGpuBondedWork))
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        ddBalanceRegionHandler.openBeforeForceComputationGpu();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        Nbnxm::gpu_upload_shiftvec(nbv->gpuNbv(), &nbv->nbat());
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (!stepWork.useGpuXBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            Nbnxm::gpu_copy_xq_to_gpu(nbv->gpuNbv(), &nbv->nbat(), AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // with X buffer ops offloaded to the GPU on all but the search steps
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (domainWork.haveGpuBondedWork && !simulationWork.havePpDomainDecomposition)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* launch local nonbonded work on GPU */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // In PME GPU and mixed mode we launch FFT / gather after the
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        // X copy/transform to allow overlap as well as after the GPU NB
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        launchPmeGpuFftAndGather(fr->pmedata,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GpuEventSynchronizer* gpuCoordinateHaloLaunched = nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.useGpuXHalo)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                gpuCoordinateHaloLaunched = communicateGpuHaloCoordinates(*cr, box, localXReadyOnDevice);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->copyCoordinatesFromGpu(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                            x.unpaddedArrayRef(), AtomLocality::NonLocal, gpuCoordinateHaloLaunched);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (simulationWork.useGpuUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                            (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.useGpuXBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (!stepWork.useGpuXHalo)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(), AtomLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                GpuEventSynchronizer* xReadyOnDeviceEvent = stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        AtomLocality::NonLocal, simulationWork, stepWork, gpuCoordinateHaloLaunched);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (stepWork.useGpuXHalo && domainWork.haveCpuNonLocalForceWork)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    /* We already enqueued an event for Gpu Halo exchange completion into the
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                nbv->convertCoordinatesGpu(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        AtomLocality::NonLocal, stateGpu->getCoordinates(), xReadyOnDeviceEvent);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (simulationWork.useGpuNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (!stepWork.useGpuXBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                Nbnxm::gpu_copy_xq_to_gpu(nbv->gpuNbv(), &nbv->nbat(), AtomLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (domainWork.haveGpuBondedWork)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            /* launch non-local nonbonded tasks on GPU */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuNonbonded && stepWork.computeNonbondedForces)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            Nbnxm::gpu_launch_cpyback(nbv->gpuNbv(), &nbv->nbat(), stepWork, AtomLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        Nbnxm::gpu_launch_cpyback(nbv->gpuNbv(), &nbv->nbat(), stepWork, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (domainWork.haveGpuBondedWork && stepWork.computeEnergy)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            fr->listedForcesGpu->launchEnergyTransfer();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // For the rest of the CPU tasks that depend on GPU-update produced coordinates,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                || (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:     * with independent GPU work (integration/constraints, x D2H copy).
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    const bool useOrEmulateGpuNb = simulationWork.useGpuNonbonded || fr->nbv->emulateGpu();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (!useOrEmulateGpuNb)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.useGpuXHalo && domainWork.haveCpuNonLocalForceWork)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:         * Happens here on the CPU both with and without GPU.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (stepWork.computeNonbondedForces && !useOrEmulateGpuNb)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            (stepWork.haveGpuPmeOnThisRank || needToReceivePmeResultsFromSeparateRank);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:     * GPU we must wait for the PME calculation (dhdl) results to finish before sampling the
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            pmeGpuWaitAndReduce(fr->pmedata,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   simulationWork.useGpuPmePpCommunication,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   stepWork.useGpuPmeFReduction,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (simulationWork.havePpDomainDecomposition && stepWork.computeForces && stepWork.useGpuFHalo
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(), AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFBufferOps),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFHalo),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "The schedule below does not allow for nonbonded MTS with GPU halo exchange");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // Will store the amount of cycles spent waiting for the GPU that
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    float cycles_wait_gpu = 0;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (useOrEmulateGpuNb && stepWork.computeNonbondedForces)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (simulationWork.useGpuNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                cycles_wait_gpu += Nbnxm::gpu_wait_finish_task(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                        nbv->gpuNbv(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.useGpuFBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                fr->gpuForceReduction[gmx::AtomLocality::NonLocal]->execute();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (!stepWork.useGpuFHalo)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    // copy from GPU input for dd_move_f()
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->copyForcesFromGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (fr->nbv->emulateGpu() && stepWork.computeVirial)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // With both nonbonded and PME offloaded a GPU on the same rank, we use
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    const bool alternateGpuWait = (!c_disableAlternatingWait && stepWork.haveGpuPmeOnThisRank
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   && simulationWork.useGpuNonbonded && !simulationWork.havePpDomainDecomposition
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                                   && !stepWork.useGpuFBufferOps && !needEarlyPmeResults);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            simulationWork, domainWork, stepWork, useOrEmulateGpuNb, alternateGpuWait);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // If expectedLocalFReadyOnDeviceConsumptionCount == 0, stateGpu can be uninitialized
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->setFReadyOnDeviceEventExpectedConsumptionCount(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:         * If we use a GPU this will overlap with GPU work, so in that case
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.useGpuFHalo)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                gmx::FixedCapacityVector<GpuEventSynchronizer*, 2> gpuForceHaloDependencies;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (domainWork.haveCpuLocalForceWork || stepWork.clearGpuFBufferEarly)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    gpuForceHaloDependencies.push_back(stateGpu->fReadyOnDevice(AtomLocality::Local));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                gpuForceHaloDependencies.push_back(stateGpu->fReducedOnDevice(AtomLocality::NonLocal));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                communicateGpuHaloForces(*cr, accumulateForces, &gpuForceHaloDependencies);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                if (stepWork.useGpuFBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->waitForcesReadyOnHost(AtomLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (alternateGpuWait)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        alternatePmeNbGpuWaitReduce(fr->nbv.get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (!alternateGpuWait && stepWork.haveGpuPmeOnThisRank && !needEarlyPmeResults)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        pmeGpuWaitAndReduce(fr->pmedata,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* Wait for local GPU NB outputs on the non-alternating wait path */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (!alternateGpuWait && stepWork.computeNonbondedForces && simulationWork.useGpuNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        /* Measured overhead on CUDA and OpenCL with(out) GPU sharing
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        const float gpuWaitApiOverheadMargin = 2e6F; /* cycles */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        const float waitCycles               = Nbnxm::gpu_wait_finish_task(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                nbv->gpuNbv(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            DdBalanceRegionWaitedForGpu waitedForGpu = DdBalanceRegionWaitedForGpu::yes;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (stepWork.computeForces && waitCycles <= gpuWaitApiOverheadMargin)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                waitedForGpu = DdBalanceRegionWaitedForGpu::no;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            ddBalanceRegionHandler.closeAfterForceComputationGpu(cycles_wait_gpu, waitedForGpu);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (fr->nbv->emulateGpu())
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // If on GPU PME-PP comms path, receive forces from PME before GPU buffer ops
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (needToReceivePmeResultsFromSeparateRank && simulationWork.useGpuPmePpCommunication && !needEarlyPmeResults)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                               simulationWork.useGpuPmePpCommunication,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                               stepWork.useGpuPmeFReduction,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* Do the nonbonded GPU (or emulation) force buffer reduction
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFBufferOps),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    if (useOrEmulateGpuNb && !alternateGpuWait)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        if (stepWork.useGpuFBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            // - copy is not perfomed if GPU force halo exchange is active, because it would overwrite the result
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (domainWork.haveLocalForceContribInCpuBuffer && !stepWork.useGpuFHalo)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                stateGpu->copyForcesToGpu(forceWithShift, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                fr->gpuForceReduction[gmx::AtomLocality::Local]->execute();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            if (!simulationWork.useGpuUpdate
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                || (simulationWork.useGpuUpdate && haveDDAtomOrdering(*cr) && simulationWork.useCpuPmePpCommunication)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    /* We have previously issued force reduction on the GPU, but we will
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                stateGpu->copyForcesFromGpu(forceWithShift, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:        stateGpu->setFReadyOnDeviceEventExpectedConsumptionCount(AtomLocality::Local, 1);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    launchGpuEndOfStepTasks(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:            nbv, fr->listedForcesGpu.get(), fr->pmedata, enerd, runScheduleWork, step, wcycle);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // TODO refactor this and unify with above GPU PME-PP / GPU update path call to the same function
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:                               simulationWork.useGpuPmePpCommunication,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp.preplumed:    /* In case we don't have constraints and are using GPUs, the next balancing
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/ewald/pme_coordinate_receiver_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/ewald/pme_pp_comm_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/nbnxm/nbnxm_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:#include "gromacs/timing/gpu_timing.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:#include "gpuforcereduction.h"
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:static const bool c_disableAlternatingWait = (getenv("GMX_DISABLE_ALTERNATING_GPU_WAIT") != nullptr);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                   bool                  useGpuPmePpComms,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                   bool                  receivePmeForceToGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    gmx_pme_receive_f(fr->pmePpCommGpu.get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                      useGpuPmePpComms,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                      receivePmeForceToGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    /* GPU kernel launch overhead is already timed separately */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (!nbv->useGpu())
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Launch the prepare_step and spread stages of PME GPU.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * \param[in]  useMdGpuGraph        Whether MD GPU Graph is in use.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:static inline void launchPmeGpuSpread(gmx_pme_t*            pmedata,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                      GpuEventSynchronizer* xReadyOnDevice,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                      bool                  useMdGpuGraph,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    wallcycle_start(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_prepare_computation(pmedata, box, wcycle, stepWork);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    bool                           useGpuDirectComm         = false;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    gmx::PmeCoordinateReceiverGpu* pmeCoordinateReceiverGpu = nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_launch_spread(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            pmedata, xReadyOnDevice, wcycle, lambdaQ, useGpuDirectComm, pmeCoordinateReceiverGpu, useMdGpuGraph);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Launch the FFT and gather stages of PME GPU
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:static void launchPmeGpuFftAndGather(gmx_pme_t*               pmedata,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_launch_complex_transforms(pmedata, wcycle, stepWork);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_launch_gather(pmedata, wcycle, lambdaQ, stepWork.computeVirial);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * Blocks until PME GPU tasks are completed, and gets the output forces and virial/energy
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:static void pmeGpuWaitAndReduce(gmx_pme_t*               pme,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    pme_gpu_wait_and_reduce(pme, stepWork, wcycle, forceWithVirial, enerd, lambdaQ);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: *  Polling wait for either of the PME or nonbonded GPU tasks.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * Instead of a static order in waiting for GPU tasks, this function
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * one of the reductions, regardless of the GPU task completion order.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:static void alternatePmeNbGpuWaitReduce(nonbonded_verlet_t* nbv,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    bool isPmeGpuDone = false;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    bool isNbGpuDone  = false;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    gmx::ArrayRef<const gmx::RVec> pmeGpuForces;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    while (!isPmeGpuDone || !isNbGpuDone)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (!isPmeGpuDone)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            GpuTaskCompletion completionType =
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    (isNbGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            isPmeGpuDone = pme_gpu_try_finish_task(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (!isNbGpuDone)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            GpuTaskCompletion completionType =
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    (isPmeGpuDone) ? GpuTaskCompletion::Wait : GpuTaskCompletion::Check;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            // To get the wcycle call count right, when in GpuTaskCompletion::Check mode,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            // GpuTaskCompletion::Wait mode the timing is expected to be done in the caller.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_start_nocount(wcycle, WallCycleCounter::WaitGpuNbL);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            isNbGpuDone = Nbnxm::gpu_try_finish_task(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    nbv->gpuNbv(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_stop(wcycle, WallCycleCounter::WaitGpuNbL);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            if (isNbGpuDone)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                wallcycle_increment_event_count(wcycle, WallCycleCounter::WaitGpuNbL);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        && (domainWork.haveCpuLocalForceWork || !stepWork.useGpuFBufferOps
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            || (havePpDomainDecomposition && !stepWork.useGpuFHalo)))
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:/* \brief Launch end-of-step GPU tasks: buffer clearing and rolling pruning.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:static void launchGpuEndOfStepTasks(nonbonded_verlet_t*               nbv,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                    gmx::ListedForcesGpu*             listedForcesGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork.simulationWork.useGpuNonbonded && runScheduleWork.stepWork.computeNonbondedForces)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:         * clear kernel launches can leave the GPU idle while it could be running
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (nbv->isDynamicPruningStepGpu(step))
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            nbv->dispatchPruneKernelGpu(step);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        /* now clear the GPU outputs while we finish the step on the CPU */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        Nbnxm::gpu_clear_outputs(nbv->gpuNbv(), runScheduleWork.stepWork.computeVirial);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork.stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start_nocount(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        bool gpuGraphWithSeparatePmeRank = false;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        pme_gpu_reinit_computation(pmedata, gpuGraphWithSeparatePmeRank, wcycle);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::PmeGpuMesh);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork.domainWork.haveGpuBondedWork && runScheduleWork.stepWork.computeEnergy)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        listedForcesGpu->waitAccumulateEnergyTerms(enerd);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        listedForcesGpu->clearEnergies();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Compute the number of times the "local coordinates ready on device" GPU event will be used as a synchronization point.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * When some work is offloaded to GPU, force calculation should wait for the atom coordinates to
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * or from the GPU integration at the end of the previous step.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * \param pmeSendCoordinatesFromGpu Whether peer-to-peer communication is used for PME coordinates.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                                          bool pmeSendCoordinatesFromGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (pmeSendCoordinatesFromGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                       "GPU PME PP communications require having a separate PME rank");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            // Event is consumed by gmx_pme_send_coordinates for GPU PME PP Communications
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            // Event is consumed by launchPmeGpuSpread
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.computeNonbondedForces && stepWork.useGpuXBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            // Event is consumed by convertCoordinatesGpu
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.useGpuXHalo)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        // Event is consumed by communicateGpuHaloCoordinates
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.clearGpuFBufferEarly && simulationWork.useGpuUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Compute the number of times the "local forces ready on device" GPU event will be used as a synchronization point.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * \param useOrEmulateGpuNb Whether GPU non-bonded calculations are used or emulated.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * \param alternateGpuWait Whether alternating wait/reduce scheme is used.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                                          bool useOrEmulateGpuNb,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                                          bool alternateGpuWait)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    bool eventUsedInGpuForceReduction =
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:             || (simulationWork.havePpDomainDecomposition && !simulationWork.useGpuHaloExchange));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    bool gpuForceReductionUsed = useOrEmulateGpuNb && !alternateGpuWait && stepWork.useGpuFBufferOps
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (gpuForceReductionUsed && eventUsedInGpuForceReduction)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    bool gpuForceHaloUsed = simulationWork.havePpDomainDecomposition && stepWork.computeForces
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                            && stepWork.useGpuFHalo;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (gpuForceHaloUsed)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Setup for the local GPU force reduction:
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] stateGpu            GPU state propagator object
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] gpuForceReduction   GPU force reduction object
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] pmePpCommGpu        PME-PP GPU communication object
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:static void setupLocalGpuForceReduction(const gmx::MdrunScheduleWorkload& runScheduleWork,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                        gmx::StatePropagatorDataGpu*      stateGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                        gmx::GpuForceReduction*           gpuForceReduction,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                        gmx::PmePpCommGpu*                pmePpCommGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:               "GPU force reduction is not compatible with MTS");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    // (re-)initialize local GPU force reduction
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    gpuForceReduction->reinit(stateGpu->getForces(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                              stateGpu->fReducedOnDevice(AtomLocality::Local));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    gpuForceReduction->registerNbnxmForce(Nbnxm::gpu_get_f(nbv->gpuNbv()));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    GpuEventSynchronizer*   pmeSynchronizer     = nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork.simulationWork.haveGpuPmeOnPpRank())
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        pmeForcePtr = pme_gpu_get_device_f(pmedata);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            pmeSynchronizer     = pme_gpu_get_f_ready_synchronizer(pmedata);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    else if (runScheduleWork.simulationWork.useGpuPmePpCommunication)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        pmeForcePtr = pmePpCommGpu->getGpuForceStagingPtr();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            pmeSynchronizer = pmePpCommGpu->getForcesReadySynchronizer();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        gpuForceReduction->registerRvecForce(pmeForcePtr);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            DeviceBuffer<uint64_t> forcesReadyNvshmemFlags = pmePpCommGpu->getGpuForcesSyncObj();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            gpuForceReduction->registerForcesReadyNvshmemFlags(forcesReadyNvshmemFlags);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (!runScheduleWork.simulationWork.useGpuPmePpCommunication || GMX_THREAD_MPI)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            GMX_ASSERT(pmeSynchronizer != nullptr, "PME force ready cuda event should not be NULL");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            gpuForceReduction->addDependency(pmeSynchronizer);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            && !runScheduleWork.simulationWork.useGpuHaloExchange))
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::Local));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (runScheduleWork.simulationWork.useGpuHaloExchange)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        gpuForceReduction->addDependency(dd->gpuHaloExchange[0][0]->getForcesReadyOnDeviceEvent());
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:/*! \brief Setup for the non-local GPU force reduction:
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] stateGpu            GPU state propagator object
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp: * \param [in] gpuForceReduction   GPU force reduction object
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:static void setupNonLocalGpuForceReduction(const gmx::MdrunScheduleWorkload& runScheduleWork,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                           gmx::StatePropagatorDataGpu*      stateGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                           gmx::GpuForceReduction*           gpuForceReduction,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    // (re-)initialize non-local GPU force reduction
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    gpuForceReduction->reinit(stateGpu->getForces(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                              stateGpu->fReducedOnDevice(AtomLocality::NonLocal));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    gpuForceReduction->registerNbnxmForce(Nbnxm::gpu_get_f(nbv->gpuNbv()));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        gpuForceReduction->addDependency(stateGpu->fReadyOnDevice(AtomLocality::NonLocal));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    gmx::StatePropagatorDataGpu* stateGpu = fr->stateGpu;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (gmx::needStateGpu(simulationWork))
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->reinit(mdatoms.homenr,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.haveGpuPmeOnPpRank())
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        GMX_ASSERT(gmx::needStateGpu(simulationWork), "StatePropagatorDataGpu is needed");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        // TODO: This should be moved into PME setup function ( pme_gpu_prepare_computation(...) )
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        pme_gpu_set_device_x(fr->pmedata, stateGpu->getCoordinates());
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    /* initialize the GPU nbnxm atom data and bonded data structures */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        // Note: cycle counting only nononbondeds, GPU listed forces counts internally
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        Nbnxm::gpu_init_atomdata(nbv->gpuNbv(), &nbv->nbat());
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (fr->listedForcesGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:             * interactions to the GPU, where the grid order is
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            fr->listedForcesGpu->updateInteractionListsAndDeviceBuffers(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    nbv->getGridIndices(), top.idef, Nbnxm::gpuGetNBAtomData(nbv->gpuNbv()));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuXBufferOpsWhenAllowed)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        nbv->atomdata_init_copy_x_to_nbat_x_gpu();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuFBufferOpsWhenAllowed)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        // with MPI, direct GPU communication, and separate PME ranks we need
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        bool delaySetupLocalGpuForceReduction = GMX_MPI && simulationWork.useGpuPmePpCommunication;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (!delaySetupLocalGpuForceReduction)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            setupLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                        stateGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                        fr->gpuForceReduction[gmx::AtomLocality::Local].get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                        fr->pmePpCommGpu.get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            setupNonLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                           stateGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                           fr->gpuForceReduction[gmx::AtomLocality::NonLocal].get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        /* Note that with a GPU the launch overhead of the list transfer is not timed separately */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        nbv->setupGpuShortRangeWork(fr->listedForcesGpu.get(), InteractionLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        // TODO refactor this GPU halo exchange re-initialisation
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        // to location in do_md where GPU halo exchange is
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        // constructed at partitioning, after above stateGpu
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (simulationWork.useGpuHaloExchange)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            reinitGpuHaloExchange(*cr, stateGpu->getCoordinates(), stateGpu->getForces());
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    gmx::StatePropagatorDataGpu* stateGpu = fr->stateGpu;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    const bool pmeSendCoordinatesFromGpu =
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            simulationWork.useGpuPmePpCommunication && !stepWork.doNeighborSearch;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    auto* localXReadyOnDevice = (stepWork.haveGpuPmeOnThisRank || stepWork.useGpuXBufferOps
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                 || simulationWork.useGpuUpdate || pmeSendCoordinatesFromGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                        ? stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.clearGpuFBufferEarly)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        // GPU Force halo exchange will set a subset of local atoms with remote non-local data.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        // which is satisfied when localXReadyOnDevice has been marked for GPU update case.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        GpuEventSynchronizer* dependency = simulationWork.useGpuUpdate ? localXReadyOnDevice : nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->clearForcesOnGpu(AtomLocality::Local, dependency);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    GMX_ASSERT(simulationWork.useGpuHaloExchange
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                       == ((cr->dd != nullptr) && (!cr->dd->gpuHaloExchange[0].empty())),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:               "The GPU halo exchange is active, but it has not been constructed.");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    bool gmx_used_in_debug haveCopiedXFromGpu = false;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    // Copy coordinate from the GPU if update is on the GPU and there
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->copyCoordinatesFromGpu(x.unpaddedArrayRef(), AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        haveCopiedXFromGpu = true;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.haveGpuPmeOnThisRank || stepWork.useGpuXBufferOps || pmeSendCoordinatesFromGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        GMX_ASSERT(stateGpu != nullptr, "stateGpu should not be null");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                        simulationWork, stepWork, pmeSendCoordinatesFromGpu);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (!simulationWork.useGpuUpdate || stepWork.doNeighborSearch)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        else if (simulationWork.useGpuUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            stateGpu->setXUpdatedOnDeviceEventExpectedConsumptionCount(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (!pmeSendCoordinatesFromGpu && !stepWork.doNeighborSearch && simulationWork.useGpuUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        const bool reinitGpuPmePpComms =
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                simulationWork.useGpuPmePpCommunication && stepWork.doNeighborSearch;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                 simulationWork.useGpuPmePpCommunication,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                 reinitGpuPmePpComms,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                 pmeSendCoordinatesFromGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                 stepWork.useGpuPmeFReduction,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                 pmeSendCoordinatesFromGpu ? localXReadyOnDevice : nullptr,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                 simulationWork.useMdGpuGraph,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuFBufferOpsWhenAllowed && stepWork.doNeighborSearch)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        // with MPI, direct GPU communication, and separate PME ranks we need
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        bool doSetupLocalGpuForceReduction = GMX_MPI && simulationWork.useGpuPmePpCommunication;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (doSetupLocalGpuForceReduction)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            setupLocalGpuForceReduction(runScheduleWork,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                        stateGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                        fr->gpuForceReduction[gmx::AtomLocality::Local].get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                        fr->pmePpCommGpu.get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        launchPmeGpuSpread(fr->pmedata,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                           simulationWork.useMdGpuGraph,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.useGpuXBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            GMX_ASSERT(stateGpu, "stateGpu should be valid when buffer ops are offloaded");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            nbv->convertCoordinatesGpu(AtomLocality::Local, stateGpu->getCoordinates(), localXReadyOnDevice);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            if (simulationWork.useGpuUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                GMX_ASSERT(stateGpu, "need a valid stateGpu object");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuNonbonded && (stepWork.computeNonbondedForces || domainWork.haveGpuBondedWork))
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        ddBalanceRegionHandler.openBeforeForceComputationGpu();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        Nbnxm::gpu_upload_shiftvec(nbv->gpuNbv(), &nbv->nbat());
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (!stepWork.useGpuXBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            Nbnxm::gpu_copy_xq_to_gpu(nbv->gpuNbv(), &nbv->nbat(), AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        // with X buffer ops offloaded to the GPU on all but the search steps
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (domainWork.haveGpuBondedWork && !simulationWork.havePpDomainDecomposition)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        /* launch local nonbonded work on GPU */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        // In PME GPU and mixed mode we launch FFT / gather after the
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        // X copy/transform to allow overlap as well as after the GPU NB
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        launchPmeGpuFftAndGather(fr->pmedata,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            GpuEventSynchronizer* gpuCoordinateHaloLaunched = nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.useGpuXHalo)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                gpuCoordinateHaloLaunched = communicateGpuHaloCoordinates(*cr, box, localXReadyOnDevice);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->copyCoordinatesFromGpu(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                            x.unpaddedArrayRef(), AtomLocality::NonLocal, gpuCoordinateHaloLaunched);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                if (simulationWork.useGpuUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                            (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.useGpuXBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                if (!stepWork.useGpuXHalo)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->copyCoordinatesToGpu(x.unpaddedArrayRef(), AtomLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                GpuEventSynchronizer* xReadyOnDeviceEvent = stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                        AtomLocality::NonLocal, simulationWork, stepWork, gpuCoordinateHaloLaunched);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                if (stepWork.useGpuXHalo && domainWork.haveCpuNonLocalForceWork)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    /* We already enqueued an event for Gpu Halo exchange completion into the
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                nbv->convertCoordinatesGpu(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                        AtomLocality::NonLocal, stateGpu->getCoordinates(), xReadyOnDeviceEvent);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (simulationWork.useGpuNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            if (!stepWork.useGpuXBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                wallcycle_start(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                Nbnxm::gpu_copy_xq_to_gpu(nbv->gpuNbv(), &nbv->nbat(), AtomLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            if (domainWork.haveGpuBondedWork)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                fr->listedForcesGpu->setPbcAndlaunchKernel(fr->pbcType, box, fr->bMolPBC, stepWork);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            /* launch non-local nonbonded tasks on GPU */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_sub_start(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuNonbonded && stepWork.computeNonbondedForces)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_start_nocount(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_start_nocount(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            Nbnxm::gpu_launch_cpyback(nbv->gpuNbv(), &nbv->nbat(), stepWork, AtomLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        Nbnxm::gpu_launch_cpyback(nbv->gpuNbv(), &nbv->nbat(), stepWork, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_sub_stop(wcycle, WallCycleSubCounter::LaunchGpuNonBonded);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (domainWork.haveGpuBondedWork && stepWork.computeEnergy)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            fr->listedForcesGpu->launchEnergyTransfer();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        wallcycle_stop(wcycle, WallCycleCounter::LaunchGpuPp);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    // For the rest of the CPU tasks that depend on GPU-update produced coordinates,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.useGpuUpdate && !stepWork.doNeighborSearch)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                || (stepWork.computePmeOnSeparateRank && !pmeSendCoordinatesFromGpu);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            GMX_ASSERT(haveCopiedXFromGpu,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:     * with independent GPU work (integration/constraints, x D2H copy).
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    const bool useOrEmulateGpuNb = simulationWork.useGpuNonbonded || fr->nbv->emulateGpu();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (!useOrEmulateGpuNb)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.useGpuXHalo && domainWork.haveCpuNonLocalForceWork)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->waitCoordinatesReadyOnHost(AtomLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:         * Happens here on the CPU both with and without GPU.
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (stepWork.computeNonbondedForces && !useOrEmulateGpuNb)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            (stepWork.haveGpuPmeOnThisRank || needToReceivePmeResultsFromSeparateRank);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:     * GPU we must wait for the PME calculation (dhdl) results to finish before sampling the
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.haveGpuPmeOnThisRank)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            pmeGpuWaitAndReduce(fr->pmedata,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                   simulationWork.useGpuPmePpCommunication,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                   stepWork.useGpuPmeFReduction,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (simulationWork.havePpDomainDecomposition && stepWork.computeForces && stepWork.useGpuFHalo
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(), AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFBufferOps),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFHalo),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:               "The schedule below does not allow for nonbonded MTS with GPU halo exchange");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    // Will store the amount of cycles spent waiting for the GPU that
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    float cycles_wait_gpu = 0;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (useOrEmulateGpuNb && stepWork.computeNonbondedForces)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            if (simulationWork.useGpuNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                cycles_wait_gpu += Nbnxm::gpu_wait_finish_task(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                        nbv->gpuNbv(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.useGpuFBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->copyForcesToGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                fr->gpuForceReduction[gmx::AtomLocality::NonLocal]->execute();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                if (!stepWork.useGpuFHalo)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    // copy from GPU input for dd_move_f()
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->copyForcesFromGpu(forceOutMtsLevel0.forceWithShiftForces().force(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            if (fr->nbv->emulateGpu() && stepWork.computeVirial)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    // With both nonbonded and PME offloaded a GPU on the same rank, we use
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    const bool alternateGpuWait = (!c_disableAlternatingWait && stepWork.haveGpuPmeOnThisRank
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                   && simulationWork.useGpuNonbonded && !simulationWork.havePpDomainDecomposition
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                                   && !stepWork.useGpuFBufferOps && !needEarlyPmeResults);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            simulationWork, domainWork, stepWork, useOrEmulateGpuNb, alternateGpuWait);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    // If expectedLocalFReadyOnDeviceConsumptionCount == 0, stateGpu can be uninitialized
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->setFReadyOnDeviceEventExpectedConsumptionCount(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:         * If we use a GPU this will overlap with GPU work, so in that case
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.useGpuFHalo)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                gmx::FixedCapacityVector<GpuEventSynchronizer*, 2> gpuForceHaloDependencies;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                if (domainWork.haveCpuLocalForceWork || stepWork.clearGpuFBufferEarly)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    gpuForceHaloDependencies.push_back(stateGpu->fReadyOnDevice(AtomLocality::Local));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                gpuForceHaloDependencies.push_back(stateGpu->fReducedOnDevice(AtomLocality::NonLocal));
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                communicateGpuHaloForces(*cr, accumulateForces, &gpuForceHaloDependencies);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                if (stepWork.useGpuFBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->waitForcesReadyOnHost(AtomLocality::NonLocal);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (alternateGpuWait)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        alternatePmeNbGpuWaitReduce(fr->nbv.get(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (!alternateGpuWait && stepWork.haveGpuPmeOnThisRank && !needEarlyPmeResults)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        pmeGpuWaitAndReduce(fr->pmedata,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    /* Wait for local GPU NB outputs on the non-alternating wait path */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (!alternateGpuWait && stepWork.computeNonbondedForces && simulationWork.useGpuNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        /* Measured overhead on CUDA and OpenCL with(out) GPU sharing
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        const float gpuWaitApiOverheadMargin = 2e6F; /* cycles */
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        const float waitCycles               = Nbnxm::gpu_wait_finish_task(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                nbv->gpuNbv(),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            DdBalanceRegionWaitedForGpu waitedForGpu = DdBalanceRegionWaitedForGpu::yes;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            if (stepWork.computeForces && waitCycles <= gpuWaitApiOverheadMargin)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                waitedForGpu = DdBalanceRegionWaitedForGpu::no;
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            ddBalanceRegionHandler.closeAfterForceComputationGpu(cycles_wait_gpu, waitedForGpu);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (fr->nbv->emulateGpu())
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    // If on GPU PME-PP comms path, receive forces from PME before GPU buffer ops
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (needToReceivePmeResultsFromSeparateRank && simulationWork.useGpuPmePpCommunication && !needEarlyPmeResults)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                               simulationWork.useGpuPmePpCommunication,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                               stepWork.useGpuPmeFReduction,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    /* Do the nonbonded GPU (or emulation) force buffer reduction
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    GMX_ASSERT(!(nonbondedAtMtsLevel1 && stepWork.useGpuFBufferOps),
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:               "The schedule below does not allow for nonbonded MTS with GPU buffer ops");
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    if (useOrEmulateGpuNb && !alternateGpuWait)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        if (stepWork.useGpuFBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            // - copy is not perfomed if GPU force halo exchange is active, because it would overwrite the result
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            if (domainWork.haveLocalForceContribInCpuBuffer && !stepWork.useGpuFHalo)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                stateGpu->copyForcesToGpu(forceWithShift, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                fr->gpuForceReduction[gmx::AtomLocality::Local]->execute();
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            if (!simulationWork.useGpuUpdate
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                || (simulationWork.useGpuUpdate && haveDDAtomOrdering(*cr) && simulationWork.useCpuPmePpCommunication)
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    /* We have previously issued force reduction on the GPU, but we will
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                    stateGpu->consumeForcesReducedOnDeviceEvent(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                stateGpu->copyForcesFromGpu(forceWithShift, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:        stateGpu->setFReadyOnDeviceEventExpectedConsumptionCount(AtomLocality::Local, 1);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    launchGpuEndOfStepTasks(
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:            nbv, fr->listedForcesGpu.get(), fr->pmedata, enerd, runScheduleWork, step, wcycle);
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    // TODO refactor this and unify with above GPU PME-PP / GPU update path call to the same function
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    // When running free energy perturbations steered by AWH and calculating PME on GPU,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:                               simulationWork.useGpuPmePpCommunication,
patches/gromacs-2024.3.diff/src/gromacs/mdlib/sim_util.cpp:    /* In case we don't have constraints and are using GPUs, the next balancing
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/ewald/pme_gpu_program.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/ewald/pme_pp_comm_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/gpu_utils/device_stream_manager.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/gpu_utils/gpueventsynchronizer_helpers.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/gpu_utils/nvshmem_utils.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/mdlib/gpuforcereduction.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/mdlib/mdgraph_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/taskassignment/decidegpuusage.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/taskassignment/usergpuids.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:#include "gromacs/timing/gpu_timing.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp: * the GPU communication flags are set to false in non-tMPI and non-CUDA builds.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp: * \param[in]  useGpuForNonbonded   True if the nonbonded task is offloaded in this run.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp: * \param[in]  gpuAwareMpiStatus  Minimum level of GPU-aware MPI support across all ranks
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                         const bool           useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                         gmx::GpuAwareMpiStatus gpuAwareMpiStatus)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    devFlags.enableGpuBufferOps = (GMX_GPU_CUDA || GMX_GPU_SYCL) && useGpuForNonbonded
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                  && (getenv("GMX_USE_GPU_BUFFER_OPS") != nullptr);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    if (getenv("GMX_CUDA_GRAPH") != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        if (GMX_HAVE_GPU_GRAPH_SUPPORT)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            devFlags.enableCudaGraphs = true;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            "GMX_CUDA_GRAPH environment variable is detected. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            "The experimental CUDA Graphs feature will be used if run conditions "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            devFlags.enableCudaGraphs = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            if (GMX_GPU_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                errorReason = "the CUDA version in use is below the minimum requirement (11.1)";
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                errorReason = "GROMACS is built without CUDA";
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            "GMX_CUDA_GRAPH environment variable is detected, but %s. GPU Graphs "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    // Flag use to enable GPU-aware MPI depenendent features such PME GPU decomposition
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    // GPU-aware MPI is marked available if it has been detected by GROMACS or detection fails but
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    devFlags.canUseGpuAwareMpi = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    // Direct GPU comm path is being used with GPU-aware MPI
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    // make sure underlying MPI implementation is GPU-aware
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    if (GMX_LIB_MPI && (GMX_GPU_CUDA || GMX_GPU_SYCL))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // Allow overriding the detection for GPU-aware MPI
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        if (getenv("GMX_FORCE_CUDA_AWARE_MPI") != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            "GMX_FORCE_CUDA_AWARE_MPI environment variable is inactive. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            "Please use GMX_FORCE_GPU_AWARE_MPI instead.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        devFlags.canUseGpuAwareMpi = (gpuAwareMpiStatus == gmx::GpuAwareMpiStatus::Supported
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                      || gpuAwareMpiStatus == gmx::GpuAwareMpiStatus::Forced);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        if (getenv("GMX_ENABLE_DIRECT_GPU_COMM") != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            if (gpuAwareMpiStatus == gmx::GpuAwareMpiStatus::Forced)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                // GPU-aware support not detected in MPI library but, user has forced its use
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                "This run has forced use of 'GPU-aware MPI'. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                "However, GROMACS cannot determine if underlying MPI is GPU-aware. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                "Check the GROMACS install guide for recommendations for GPU-aware "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                "GMX_FORCE_GPU_AWARE_MPI environment variable.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            if (devFlags.canUseGpuAwareMpi)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                "GMX_ENABLE_DIRECT_GPU_COMM environment variable detected, "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                "enabling direct GPU communication using GPU-aware MPI.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                "GPU-aware MPI was not detected, will not use direct GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                "for GPU-aware support. If you are certain about GPU-aware support "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                "GMX_FORCE_GPU_AWARE_MPI environment variable.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        else if (gpuAwareMpiStatus == gmx::GpuAwareMpiStatus::Supported)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            // GPU-aware MPI was detected, let the user know that using it may improve performance
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            "GPU-aware MPI detected, but by default GROMACS will not "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            "make use the direct GPU communication capabilities of MPI. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            "the GMX_ENABLE_DIRECT_GPU_COMM environment variable.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        if (getenv("GMX_FORCE_GPU_AWARE_MPI") != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            // Cannot force use of GPU-aware MPI in this build configuration
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            "A CUDA or SYCL build with an external MPI library is required in "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            "order to benefit from GMX_FORCE_GPU_AWARE_MPI. That environment "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    if (devFlags.enableGpuBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                        "This run uses the 'GPU buffer ops' feature, enabled by the "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                        "GMX_USE_GPU_BUFFER_OPS environment variable.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    // PME decomposition is supported only with CUDA or SYCL and also
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    // needs GPU-aware MPI support for it to work.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    const bool pmeGpuDecompositionRequested =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            (pmeRunMode == PmeRunMode::GPU || pmeRunMode == PmeRunMode::Mixed)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    const bool pmeGpuDecompositionSupported =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            (devFlags.canUseGpuAwareMpi && (GMX_GPU_CUDA || GMX_GPU_SYCL)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:             && ((pmeRunMode == PmeRunMode::GPU && (GMX_USE_Heffte || GMX_USE_cuFFTMp))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    const bool forcePmeGpuDecomposition = getenv("GMX_GPU_PME_DECOMPOSITION") != nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    if (pmeGpuDecompositionSupported && pmeGpuDecompositionRequested)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // PME decomposition is supported only when it is forced using GMX_GPU_PME_DECOMPOSITION
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        if (forcePmeGpuDecomposition)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            "This run has requested the 'GPU PME decomposition' feature, enabled "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            "by the GMX_GPU_PME_DECOMPOSITION environment variable. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                      "Multiple PME tasks were required to run on GPUs, "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                      "Use GMX_GPU_PME_DECOMPOSITION environment variable to enable it.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    if (!pmeGpuDecompositionSupported && pmeGpuDecompositionRequested)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        if (GMX_GPU_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                      "PME tasks were required to run on more than one CUDA-devices. To enable "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                      "use MPI with CUDA-aware support and build GROMACS with cuFFTMp support.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                    "PME tasks were required to run on GPUs, but that is not implemented with "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    devFlags.enableGpuPmeDecomposition =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            forcePmeGpuDecomposition && pmeGpuDecompositionRequested && pmeGpuDecompositionSupported;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                  bool                           makeGpuPairList,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                (makeGpuPairList ? ListSetupType::Gpu : ListSetupType::CpuSimdWhenSupported);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                fplog, cr, ir, nstlist_cmdline, &mtop, box, effectiveAtomDensity.value(), makeGpuPairList, cpuinfo);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:/*! \brief Return whether GPU acceleration of nonbondeds is supported with the given settings.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:static bool gpuAccelerationOfNonbondedIsUseful(const MDLogger&   mdlog,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    bool        gpuIsUseful = true;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        /* The GPU code does not support more than one energy group.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:         * If the user requested GPUs explicitly, a fatal error is given later.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        gpuIsUseful = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                    "Multiple energy groups is not implemented for GPUs, falling back to the CPU. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                    "For better performance, run on the GPU without energy groups and then do "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    /* There are resource handling issues in the GPU code paths with MTS on anything else than only
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        gpuIsUseful = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                "Multiple time stepping is only supported with GPUs when MTS is only applied to %s "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        gpuIsUseful = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        warning     = "TPI is not implemented for GPUs.";
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    if (!gpuIsUseful && issueWarning)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    return gpuIsUseful;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    else if (strncmp(optionString, "gpu", 3) == 0)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        returnValue = TaskTarget::Gpu;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        auto* nbnxn_gpu_timings =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                (nbv != nullptr && nbv->useGpu()) ? Nbnxm::gpu_get_timings(nbv->gpuNbv()) : nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        gmx_wallclock_gpu_pme_t pme_gpu_timings = {};
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        if (pme_gpu_task_enabled(pme))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            pme_gpu_get_timings(pme, &pme_gpu_timings);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                        nbnxn_gpu_timings,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                        &pme_gpu_timings);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    EmulateGpuNonbonded emulateGpuNonbonded =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            (getenv("GMX_EMULATE_GPU") != nullptr ? EmulateGpuNonbonded::Yes : EmulateGpuNonbonded::No);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    std::vector<int> userGpuTaskAssignment;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        userGpuTaskAssignment = parseUserTaskAssignmentString(hw_opt.userGpuTaskAssignment);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        bool useGpuForNonbonded = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        bool useGpuForPme       = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            // the number of GPUs to choose the number of ranks.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            auto canUseGpuForNonbonded = buildSupportsNonbondedOnGpu(nullptr);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            useGpuForNonbonded         = decideWhetherToUseGpusForNonbondedWithThreadMpi(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                    userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                    emulateGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                    canUseGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                    gpuAccelerationOfNonbondedIsUseful(mdlog, *inputrec, GMX_THREAD_MPI, doRerun),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            useGpuForPme = decideWhetherToUseGpusForPmeWithThreadMpi(useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                                     userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                useGpuForPme,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    // Note that when bonded interactions run on a GPU they always run
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuForNonbonded = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuForPme       = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuForBonded    = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuForUpdate    = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    bool gpusWereDetected   = hwinfo_->ngpu_compatible_tot > 0;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // It's possible that there are different numbers of GPUs on
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        auto canUseGpuForNonbonded = buildSupportsNonbondedOnGpu(nullptr);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        useGpuForNonbonded         = decideWhetherToUseGpusForNonbonded(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                emulateGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                canUseGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                gpuAccelerationOfNonbondedIsUseful(mdlog, *inputrec, !GMX_THREAD_MPI, doRerun),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                gpusWereDetected);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        useGpuForPme    = decideWhetherToUseGpusForPme(useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                    userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                    gpusWereDetected);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        useGpuForBonded = decideWhetherToUseGpusForBonded(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                useGpuForNonbonded, useGpuForPme, bondedTarget, *inputrec, mtop, domdecOptions.numPmeRanks, gpusWereDetected);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    const PmeRunMode pmeRunMode = determinePmeRunMode(useGpuForPme, pmeFftTarget, *inputrec);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    // We are using the minimal supported level of GPU-aware MPI
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                                       useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                                       hwinfo_->minGpuAwareMpiStatus);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                              updateTarget == TaskTarget::Gpu);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                || (!useGpuForNonbonded && usingFullElectrostatics(inputrec->coulombtype)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                          useGpuForNonbonded || (emulateGpuNonbonded == EmulateGpuNonbonded::Yes),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        useGpuForUpdate = decideWhetherToUseGpuForUpdate(useDomainDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                         useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                         gpusWereDetected,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    const bool canUseDirectGpuComm = decideWhetherDirectGpuCommunicationCanBeUsed(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    bool useGpuDirectHalo = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    if (useGpuForNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // domdecOptions.numPmeRanks == -1 results in 0 separate PME ranks when useGpuForNonbonded is true.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        useGpuDirectHalo = decideWhetherToUseGpuForHalo(havePPDomainDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                        useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                        canUseDirectGpuComm,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // The DD builder will disable useGpuDirectHalo if the Y or Z component of any domain is
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // smaller than twice the communication distance, since GPU-direct communication presently
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // perform well on multiple GPUs in any case, but it is important that our core functionality
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // (in particular for testing) does not break depending on GPU direct communication being enabled.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                useGpuForPme,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                useGpuForUpdate,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                useGpuDirectHalo,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                devFlags.enableGpuPmeDecomposition);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    GpuTaskAssignments gpuTaskAssignments = GpuTaskAssignmentsBuilder::build(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            useGpuForPme,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    DeviceInformation* deviceInfo = gpuTaskAssignments.initDevice(&deviceId);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // TODO Pass the GPU streams to ddBuilder to use in buffer
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    const bool useGpuPmeDecomposition = numPmeDomains.x * numPmeDomains.y > 1 && useGpuForPme;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    GMX_RELEASE_ASSERT(!useGpuPmeDecomposition || devFlags.enableGpuPmeDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                       "GPU PME decomposition works only in the cases where it is supported");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuForBonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuForUpdate,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuDirectHalo,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                              canUseDirectGpuComm,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                                              useGpuPmeDecomposition);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    if (runScheduleWork.simulationWork.useGpuDirectCommunication && GMX_GPU_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // Don't enable event counting with GPU Direct comm, see #3988.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        gmx::internal::disableCudaEventConsumptionCounting();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    if (isSimulationMainRank && GMX_GPU_SYCL)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        bool                      haveAnyGpuWork = simWorkload.useGpuPme || simWorkload.useGpuBonded
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                              || simWorkload.useGpuNonbonded || simWorkload.useGpuUpdate;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        if (haveAnyGpuWork)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            "\nNOTE: SYCL GPU support in GROMACS, and the compilers, libraries,\n"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    gpuTaskAssignments.reportGpuUsage(mdlog, printHostName, pmeRunMode, runScheduleWork.simulationWork);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        const bool useGpuTiming = decideGpuTimingsUsage();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                *deviceInfo, runScheduleWork.simulationWork, useGpuTiming);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        gpuTaskAssignments.logPerformanceHints(mdlog, numAvailableDevices);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    check_resource_division_efficiency(hwinfo_, gpuTaskAssignments.thisRankHasAnyGpuTask(), cr, mdlog);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    // Enable Peer access between GPUs where available
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    // any of the GPU communication features are active.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        && (runScheduleWork.simulationWork.useGpuHaloExchange
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            || runScheduleWork.simulationWork.useGpuPmePpCommunication))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        setupGpuDevicePeerAccess(gpuTaskAssignments.deviceIdsAssigned(), mdlog);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    const bool thisRankHasPmeGpuTask = gpuTaskAssignments.thisRankHasPmeGpuTask();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        if (runScheduleWork.simulationWork.useGpuPmePpCommunication && !thisRankHasDuty(cr, DUTY_PME))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                    "GPU device stream manager should be valid in order to use PME-PP direct "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                    "GPU PP-PME stream should be valid in order to use GPU PME-PP direct "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            fr->pmePpCommGpu = std::make_unique<gmx::PmePpCommGpu>(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                runScheduleWork.simulationWork.useGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // TODO: Move the logic below to a GPU bonded builder
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        if (runScheduleWork.simulationWork.useGpuBonded)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                               "GPU device stream manager should be valid in order to use GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            fr->listedForcesGpu =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                    std::make_unique<ListedForcesGpu>(mtop.ffparams,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        mdAtoms = makeMDAtoms(fplog, mtop, *inputrec, thisRankHasPmeGpuTask);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        if (globalState && thisRankHasPmeGpuTask)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            // PME on GPU without DD or on a separate PME rank, and because the local state pointer
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    PmeGpuProgramStorage pmeGpuProgram;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:    if (thisRankHasPmeGpuTask)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                "GPU device stream manager should be initialized in order to use GPU for PME.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                           "GPU device should be initialized in order to use GPU for PME.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        pmeGpuProgram = buildPmeGpuProgram(deviceStreamManager->context());
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                GMX_RELEASE_ASSERT(!runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                   "Device stream manager should be valid in order to use GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                        !runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                        "GPU PME stream should be valid in order to use GPU version of PME.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                const DeviceContext* deviceContext = runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                        runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                       pmeGpuProgram.get(),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            if (runScheduleWork.simulationWork.useGpuFBufferOpsWhenAllowed)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                fr->gpuForceReduction[gmx::AtomLocality::Local] = std::make_unique<gmx::GpuForceReduction>(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                    fr->gpuForceReduction[gmx::AtomLocality::NonLocal] =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            std::make_unique<gmx::GpuForceReduction>(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                if (runScheduleWork.simulationWork.useMdGpuGraph)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            std::make_unique<gmx::MdGpuGraph>(*fr->deviceStreamManager,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                            std::make_unique<gmx::MdGpuGraph>(*fr->deviceStreamManager,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            std::unique_ptr<gmx::StatePropagatorDataGpu> stateGpu;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            if (gpusWereDetected && gmx::needStateGpu(runScheduleWork.simulationWork))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                GpuApiCallBehavior transferKind =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                ? GpuApiCallBehavior::Async
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                : GpuApiCallBehavior::Sync;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                                   "GPU device stream manager should be initialized to use GPU.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                stateGpu = std::make_unique<gmx::StatePropagatorDataGpu>(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                        *deviceStreamManager, transferKind, pme_gpu_get_block_size(fr->pmedata), wcycle.get());
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                fr->stateGpu = stateGpu.get();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:          /* set GPU device id */
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:             plumed_cmd(plumedmain,"setGpuDeviceId", &deviceId);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:          if(useGpuForUpdate) {
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                        "This simulation is resident on GPU (-update gpu)\n"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            if (fr->pmePpCommGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                // destroy object since it is no longer required. (This needs to be done while the GPU context still exists.)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                fr->pmePpCommGpu.reset();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                        runScheduleWork.simulationWork.useGpuPmePpCommunication,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            /* stop the GPU profiler (only CUDA);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            stopGpuProfiler();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // before we destroy the GPU context(s)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // Pinned buffers are associated with contexts in CUDA.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        // As soon as we destroy GPU contexts after mdrunner() exits, these lines should go.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        mdModules_.reset(nullptr); // destruct force providers here as they might also use the GPU
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        fr.reset(nullptr);         // destruct forcerec before gpu
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:         * destroying the CUDA context as some tMPI ranks may be sharing
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:         * GPU and context.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:         * This is not a concern in OpenCL where we use one context per rank.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:         * Note: it is safe to not call the barrier on the ranks which do not use GPU,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:         * Note that this function needs to be called even if GPUs are not used
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:         * in this run because the PME ranks have no knowledge of whether GPUs
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:         * that it's not needed anymore (with a shared GPU run).
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        const bool haveDetectedOrForcedCudaAwareMpi =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                (gmx::checkMpiCudaAwareSupport() == gmx::GpuAwareMpiStatus::Supported
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:                 || gmx::checkMpiCudaAwareSupport() == gmx::GpuAwareMpiStatus::Forced);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:        if (!haveDetectedOrForcedCudaAwareMpi)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            // Don't reset GPU in case of GPU-AWARE MPI
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp:            // UCX creates GPU buffers which are cleaned-up as part of MPI_Finalize()
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:    // which compatible GPUs are availble for use, or to select a GPU
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:        hw_opt.userGpuTaskAssignment = userGpuTaskAssignment;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:        const char* env = getenv("GMX_GPU_ID");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:                gmx_fatal(FARGS, "GMX_GPU_ID and -gpu_id can not be used at the same time");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:        env = getenv("GMX_GPUTASKS");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:            if (!hw_opt.userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:                gmx_fatal(FARGS, "GMX_GPUTASKS and -gputasks can not be used at the same time");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:            hw_opt.userGpuTaskAssignment = env;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:        if (!hw_opt.devicesSelectedByUser.empty() && !hw_opt.userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp:            gmx_fatal(FARGS, "-gpu_id and -gputasks cannot be used at the same time");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:    // which compatible GPUs are availble for use, or to select a GPU
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:        hw_opt.userGpuTaskAssignment = userGpuTaskAssignment;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:        const char* env = getenv("GMX_GPU_ID");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:                gmx_fatal(FARGS, "GMX_GPU_ID and -gpu_id can not be used at the same time");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:        env = getenv("GMX_GPUTASKS");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:            if (!hw_opt.userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:                gmx_fatal(FARGS, "GMX_GPUTASKS and -gputasks can not be used at the same time");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:            hw_opt.userGpuTaskAssignment = env;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:        if (!hw_opt.devicesSelectedByUser.empty() && !hw_opt.userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.cpp.preplumed:            gmx_fatal(FARGS, "-gpu_id and -gputasks cannot be used at the same time");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* nbpu_opt_choices[5]    = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* pme_opt_choices[5]     = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* pme_fft_opt_choices[5] = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* bonded_opt_choices[5]  = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* update_opt_choices[5]  = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:    const char* userGpuTaskAssignment  = "";
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:        { "-gpu_id",
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:          "List of unique GPU device IDs available to use" },
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:        { "-gputasks",
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:          { &userGpuTaskAssignment },
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:          "List of GPU device IDs, mapping each task on a node to a device. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h.preplumed:          "Optimize PME load between PP/PME ranks or GPU/CPU" },
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* nbpu_opt_choices[5]    = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* pme_opt_choices[5]     = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* pme_fft_opt_choices[5] = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* bonded_opt_choices[5]  = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* update_opt_choices[5]  = { nullptr, "auto", "cpu", "gpu", nullptr };
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h:    const char* userGpuTaskAssignment  = "";
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h:        { "-gpu_id",
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h:          "List of unique GPU device IDs available to use" },
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h:        { "-gputasks",
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h:          { &userGpuTaskAssignment },
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h:          "List of GPU device IDs, mapping each task on a node to a device. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/legacymdrunoptions.h:          "Optimize PME load between PP/PME ranks or GPU/CPU" },
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/ewald/pme_gpu_program.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/ewald/pme_pp_comm_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/gpu_utils/device_stream_manager.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/gpu_utils/gpueventsynchronizer_helpers.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/gpu_utils/nvshmem_utils.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/mdlib/gpuforcereduction.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/mdlib/mdgraph_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/taskassignment/decidegpuusage.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/taskassignment/usergpuids.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:#include "gromacs/timing/gpu_timing.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed: * the GPU communication flags are set to false in non-tMPI and non-CUDA builds.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed: * \param[in]  useGpuForNonbonded   True if the nonbonded task is offloaded in this run.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed: * \param[in]  gpuAwareMpiStatus  Minimum level of GPU-aware MPI support across all ranks
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                         const bool           useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                         gmx::GpuAwareMpiStatus gpuAwareMpiStatus)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    devFlags.enableGpuBufferOps = (GMX_GPU_CUDA || GMX_GPU_SYCL) && useGpuForNonbonded
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                  && (getenv("GMX_USE_GPU_BUFFER_OPS") != nullptr);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (getenv("GMX_CUDA_GRAPH") != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (GMX_HAVE_GPU_GRAPH_SUPPORT)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            devFlags.enableCudaGraphs = true;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "GMX_CUDA_GRAPH environment variable is detected. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "The experimental CUDA Graphs feature will be used if run conditions "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            devFlags.enableCudaGraphs = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            if (GMX_GPU_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                errorReason = "the CUDA version in use is below the minimum requirement (11.1)";
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                errorReason = "GROMACS is built without CUDA";
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "GMX_CUDA_GRAPH environment variable is detected, but %s. GPU Graphs "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Flag use to enable GPU-aware MPI depenendent features such PME GPU decomposition
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // GPU-aware MPI is marked available if it has been detected by GROMACS or detection fails but
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    devFlags.canUseGpuAwareMpi = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Direct GPU comm path is being used with GPU-aware MPI
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // make sure underlying MPI implementation is GPU-aware
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (GMX_LIB_MPI && (GMX_GPU_CUDA || GMX_GPU_SYCL))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // Allow overriding the detection for GPU-aware MPI
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (getenv("GMX_FORCE_CUDA_AWARE_MPI") != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "GMX_FORCE_CUDA_AWARE_MPI environment variable is inactive. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "Please use GMX_FORCE_GPU_AWARE_MPI instead.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        devFlags.canUseGpuAwareMpi = (gpuAwareMpiStatus == gmx::GpuAwareMpiStatus::Supported
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                      || gpuAwareMpiStatus == gmx::GpuAwareMpiStatus::Forced);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (getenv("GMX_ENABLE_DIRECT_GPU_COMM") != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            if (gpuAwareMpiStatus == gmx::GpuAwareMpiStatus::Forced)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                // GPU-aware support not detected in MPI library but, user has forced its use
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "This run has forced use of 'GPU-aware MPI'. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "However, GROMACS cannot determine if underlying MPI is GPU-aware. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "Check the GROMACS install guide for recommendations for GPU-aware "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "GMX_FORCE_GPU_AWARE_MPI environment variable.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            if (devFlags.canUseGpuAwareMpi)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "GMX_ENABLE_DIRECT_GPU_COMM environment variable detected, "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "enabling direct GPU communication using GPU-aware MPI.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "GPU-aware MPI was not detected, will not use direct GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "for GPU-aware support. If you are certain about GPU-aware support "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                "GMX_FORCE_GPU_AWARE_MPI environment variable.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        else if (gpuAwareMpiStatus == gmx::GpuAwareMpiStatus::Supported)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // GPU-aware MPI was detected, let the user know that using it may improve performance
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "GPU-aware MPI detected, but by default GROMACS will not "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "make use the direct GPU communication capabilities of MPI. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "the GMX_ENABLE_DIRECT_GPU_COMM environment variable.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (getenv("GMX_FORCE_GPU_AWARE_MPI") != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // Cannot force use of GPU-aware MPI in this build configuration
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "A CUDA or SYCL build with an external MPI library is required in "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "order to benefit from GMX_FORCE_GPU_AWARE_MPI. That environment "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (devFlags.enableGpuBufferOps)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        "This run uses the 'GPU buffer ops' feature, enabled by the "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        "GMX_USE_GPU_BUFFER_OPS environment variable.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // PME decomposition is supported only with CUDA or SYCL and also
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // needs GPU-aware MPI support for it to work.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool pmeGpuDecompositionRequested =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            (pmeRunMode == PmeRunMode::GPU || pmeRunMode == PmeRunMode::Mixed)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool pmeGpuDecompositionSupported =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            (devFlags.canUseGpuAwareMpi && (GMX_GPU_CUDA || GMX_GPU_SYCL)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:             && ((pmeRunMode == PmeRunMode::GPU && (GMX_USE_Heffte || GMX_USE_cuFFTMp))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool forcePmeGpuDecomposition = getenv("GMX_GPU_PME_DECOMPOSITION") != nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (pmeGpuDecompositionSupported && pmeGpuDecompositionRequested)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // PME decomposition is supported only when it is forced using GMX_GPU_PME_DECOMPOSITION
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (forcePmeGpuDecomposition)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "This run has requested the 'GPU PME decomposition' feature, enabled "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "by the GMX_GPU_PME_DECOMPOSITION environment variable. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                      "Multiple PME tasks were required to run on GPUs, "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                      "Use GMX_GPU_PME_DECOMPOSITION environment variable to enable it.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (!pmeGpuDecompositionSupported && pmeGpuDecompositionRequested)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (GMX_GPU_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                      "PME tasks were required to run on more than one CUDA-devices. To enable "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                      "use MPI with CUDA-aware support and build GROMACS with cuFFTMp support.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "PME tasks were required to run on GPUs, but that is not implemented with "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    devFlags.enableGpuPmeDecomposition =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            forcePmeGpuDecomposition && pmeGpuDecompositionRequested && pmeGpuDecompositionSupported;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                  bool                           makeGpuPairList,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                (makeGpuPairList ? ListSetupType::Gpu : ListSetupType::CpuSimdWhenSupported);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                fplog, cr, ir, nstlist_cmdline, &mtop, box, effectiveAtomDensity.value(), makeGpuPairList, cpuinfo);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:/*! \brief Return whether GPU acceleration of nonbondeds is supported with the given settings.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:static bool gpuAccelerationOfNonbondedIsUseful(const MDLogger&   mdlog,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool        gpuIsUseful = true;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        /* The GPU code does not support more than one energy group.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:         * If the user requested GPUs explicitly, a fatal error is given later.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gpuIsUseful = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "Multiple energy groups is not implemented for GPUs, falling back to the CPU. "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "For better performance, run on the GPU without energy groups and then do "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    /* There are resource handling issues in the GPU code paths with MTS on anything else than only
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gpuIsUseful = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                "Multiple time stepping is only supported with GPUs when MTS is only applied to %s "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gpuIsUseful = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        warning     = "TPI is not implemented for GPUs.";
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (!gpuIsUseful && issueWarning)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    return gpuIsUseful;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    else if (strncmp(optionString, "gpu", 3) == 0)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        returnValue = TaskTarget::Gpu;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        auto* nbnxn_gpu_timings =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                (nbv != nullptr && nbv->useGpu()) ? Nbnxm::gpu_get_timings(nbv->gpuNbv()) : nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gmx_wallclock_gpu_pme_t pme_gpu_timings = {};
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (pme_gpu_task_enabled(pme))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            pme_gpu_get_timings(pme, &pme_gpu_timings);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        nbnxn_gpu_timings,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        &pme_gpu_timings);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    EmulateGpuNonbonded emulateGpuNonbonded =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            (getenv("GMX_EMULATE_GPU") != nullptr ? EmulateGpuNonbonded::Yes : EmulateGpuNonbonded::No);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    std::vector<int> userGpuTaskAssignment;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        userGpuTaskAssignment = parseUserTaskAssignmentString(hw_opt.userGpuTaskAssignment);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        bool useGpuForNonbonded = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        bool useGpuForPme       = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // the number of GPUs to choose the number of ranks.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            auto canUseGpuForNonbonded = buildSupportsNonbondedOnGpu(nullptr);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            useGpuForNonbonded         = decideWhetherToUseGpusForNonbondedWithThreadMpi(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    emulateGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    canUseGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    gpuAccelerationOfNonbondedIsUseful(mdlog, *inputrec, GMX_THREAD_MPI, doRerun),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            useGpuForPme = decideWhetherToUseGpusForPmeWithThreadMpi(useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                                     userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                useGpuForPme,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Note that when bonded interactions run on a GPU they always run
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuForNonbonded = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuForPme       = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuForBonded    = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuForUpdate    = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool gpusWereDetected   = hwinfo_->ngpu_compatible_tot > 0;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // It's possible that there are different numbers of GPUs on
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        auto canUseGpuForNonbonded = buildSupportsNonbondedOnGpu(nullptr);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuForNonbonded         = decideWhetherToUseGpusForNonbonded(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                emulateGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                canUseGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                gpuAccelerationOfNonbondedIsUseful(mdlog, *inputrec, !GMX_THREAD_MPI, doRerun),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                gpusWereDetected);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuForPme    = decideWhetherToUseGpusForPme(useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                    userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                    gpusWereDetected);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuForBonded = decideWhetherToUseGpusForBonded(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuForNonbonded, useGpuForPme, bondedTarget, *inputrec, mtop, domdecOptions.numPmeRanks, gpusWereDetected);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const PmeRunMode pmeRunMode = determinePmeRunMode(useGpuForPme, pmeFftTarget, *inputrec);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // We are using the minimal supported level of GPU-aware MPI
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                                       useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                                       hwinfo_->minGpuAwareMpiStatus);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              updateTarget == TaskTarget::Gpu);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                || (!useGpuForNonbonded && usingFullElectrostatics(inputrec->coulombtype)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                          useGpuForNonbonded || (emulateGpuNonbonded == EmulateGpuNonbonded::Yes),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuForUpdate = decideWhetherToUseGpuForUpdate(useDomainDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                         useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                         gpusWereDetected,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool canUseDirectGpuComm = decideWhetherDirectGpuCommunicationCanBeUsed(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    bool useGpuDirectHalo = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (useGpuForNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // domdecOptions.numPmeRanks == -1 results in 0 separate PME ranks when useGpuForNonbonded is true.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        useGpuDirectHalo = decideWhetherToUseGpuForHalo(havePPDomainDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                        useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                        canUseDirectGpuComm,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // The DD builder will disable useGpuDirectHalo if the Y or Z component of any domain is
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // smaller than twice the communication distance, since GPU-direct communication presently
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // perform well on multiple GPUs in any case, but it is important that our core functionality
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // (in particular for testing) does not break depending on GPU direct communication being enabled.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuForPme,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuForUpdate,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                useGpuDirectHalo,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                devFlags.enableGpuPmeDecomposition);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    GpuTaskAssignments gpuTaskAssignments = GpuTaskAssignmentsBuilder::build(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            useGpuForPme,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    DeviceInformation* deviceInfo = gpuTaskAssignments.initDevice(&deviceId);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // TODO Pass the GPU streams to ddBuilder to use in buffer
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool useGpuPmeDecomposition = numPmeDomains.x * numPmeDomains.y > 1 && useGpuForPme;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    GMX_RELEASE_ASSERT(!useGpuPmeDecomposition || devFlags.enableGpuPmeDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                       "GPU PME decomposition works only in the cases where it is supported");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuForBonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuForUpdate,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuDirectHalo,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              canUseDirectGpuComm,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                                              useGpuPmeDecomposition);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (runScheduleWork.simulationWork.useGpuDirectCommunication && GMX_GPU_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // Don't enable event counting with GPU Direct comm, see #3988.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gmx::internal::disableCudaEventConsumptionCounting();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (isSimulationMainRank && GMX_GPU_SYCL)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        bool                      haveAnyGpuWork = simWorkload.useGpuPme || simWorkload.useGpuBonded
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                              || simWorkload.useGpuNonbonded || simWorkload.useGpuUpdate;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (haveAnyGpuWork)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            "\nNOTE: SYCL GPU support in GROMACS, and the compilers, libraries,\n"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    gpuTaskAssignments.reportGpuUsage(mdlog, printHostName, pmeRunMode, runScheduleWork.simulationWork);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        const bool useGpuTiming = decideGpuTimingsUsage();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                *deviceInfo, runScheduleWork.simulationWork, useGpuTiming);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        gpuTaskAssignments.logPerformanceHints(mdlog, numAvailableDevices);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    check_resource_division_efficiency(hwinfo_, gpuTaskAssignments.thisRankHasAnyGpuTask(), cr, mdlog);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // Enable Peer access between GPUs where available
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    // any of the GPU communication features are active.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        && (runScheduleWork.simulationWork.useGpuHaloExchange
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            || runScheduleWork.simulationWork.useGpuPmePpCommunication))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        setupGpuDevicePeerAccess(gpuTaskAssignments.deviceIdsAssigned(), mdlog);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    const bool thisRankHasPmeGpuTask = gpuTaskAssignments.thisRankHasPmeGpuTask();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (runScheduleWork.simulationWork.useGpuPmePpCommunication && !thisRankHasDuty(cr, DUTY_PME))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "GPU device stream manager should be valid in order to use PME-PP direct "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    "GPU PP-PME stream should be valid in order to use GPU PME-PP direct "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->pmePpCommGpu = std::make_unique<gmx::PmePpCommGpu>(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                runScheduleWork.simulationWork.useGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // TODO: Move the logic below to a GPU bonded builder
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (runScheduleWork.simulationWork.useGpuBonded)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                               "GPU device stream manager should be valid in order to use GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            fr->listedForcesGpu =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    std::make_unique<ListedForcesGpu>(mtop.ffparams,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        mdAtoms = makeMDAtoms(fplog, mtop, *inputrec, thisRankHasPmeGpuTask);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (globalState && thisRankHasPmeGpuTask)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // PME on GPU without DD or on a separate PME rank, and because the local state pointer
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    PmeGpuProgramStorage pmeGpuProgram;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:    if (thisRankHasPmeGpuTask)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                "GPU device stream manager should be initialized in order to use GPU for PME.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                           "GPU device should be initialized in order to use GPU for PME.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        pmeGpuProgram = buildPmeGpuProgram(deviceStreamManager->context());
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                GMX_RELEASE_ASSERT(!runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                   "Device stream manager should be valid in order to use GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        !runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        "GPU PME stream should be valid in order to use GPU version of PME.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                const DeviceContext* deviceContext = runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        runScheduleWork.simulationWork.useGpuPme
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                       pmeGpuProgram.get(),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            if (runScheduleWork.simulationWork.useGpuFBufferOpsWhenAllowed)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                fr->gpuForceReduction[gmx::AtomLocality::Local] = std::make_unique<gmx::GpuForceReduction>(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                    fr->gpuForceReduction[gmx::AtomLocality::NonLocal] =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            std::make_unique<gmx::GpuForceReduction>(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                if (runScheduleWork.simulationWork.useMdGpuGraph)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            std::make_unique<gmx::MdGpuGraph>(*fr->deviceStreamManager,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                            std::make_unique<gmx::MdGpuGraph>(*fr->deviceStreamManager,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            std::unique_ptr<gmx::StatePropagatorDataGpu> stateGpu;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            if (gpusWereDetected && gmx::needStateGpu(runScheduleWork.simulationWork))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                GpuApiCallBehavior transferKind =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                ? GpuApiCallBehavior::Async
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                : GpuApiCallBehavior::Sync;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                                   "GPU device stream manager should be initialized to use GPU.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                stateGpu = std::make_unique<gmx::StatePropagatorDataGpu>(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        *deviceStreamManager, transferKind, pme_gpu_get_block_size(fr->pmedata), wcycle.get());
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                fr->stateGpu = stateGpu.get();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            if (fr->pmePpCommGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                // destroy object since it is no longer required. (This needs to be done while the GPU context still exists.)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                fr->pmePpCommGpu.reset();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                        runScheduleWork.simulationWork.useGpuPmePpCommunication,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            /* stop the GPU profiler (only CUDA);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            stopGpuProfiler();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // before we destroy the GPU context(s)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // Pinned buffers are associated with contexts in CUDA.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        // As soon as we destroy GPU contexts after mdrunner() exits, these lines should go.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        mdModules_.reset(nullptr); // destruct force providers here as they might also use the GPU
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        fr.reset(nullptr);         // destruct forcerec before gpu
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:         * destroying the CUDA context as some tMPI ranks may be sharing
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:         * GPU and context.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:         * This is not a concern in OpenCL where we use one context per rank.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:         * Note: it is safe to not call the barrier on the ranks which do not use GPU,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:         * Note that this function needs to be called even if GPUs are not used
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:         * in this run because the PME ranks have no knowledge of whether GPUs
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:         * that it's not needed anymore (with a shared GPU run).
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        const bool haveDetectedOrForcedCudaAwareMpi =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                (gmx::checkMpiCudaAwareSupport() == gmx::GpuAwareMpiStatus::Supported
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:                 || gmx::checkMpiCudaAwareSupport() == gmx::GpuAwareMpiStatus::Forced);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:        if (!haveDetectedOrForcedCudaAwareMpi)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // Don't reset GPU in case of GPU-AWARE MPI
patches/gromacs-2024.3.diff/src/gromacs/mdrun/runner.cpp.preplumed:            // UCX creates GPU buffers which are cleaned-up as part of MPI_Finalize()
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/gpu_utils/device_stream_manager.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/mdlib/mdgraph_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/mdlib/update_constrain_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:    /* PME load balancing data for GPU kernels */
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:    const bool  useGpuForPme       = simulationWork.useGpuPme;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:    const bool  useGpuForNonbonded = simulationWork.useGpuNonbonded;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:    const bool  useGpuForUpdate    = simulationWork.useGpuUpdate;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                                 useGpuForPme);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                   (simulationWork.useGpuFBufferOpsWhenAllowed || useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:    std::unique_ptr<UpdateConstrainGpu> integrator;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:    StatePropagatorDataGpu* stateGpu = fr_->stateGpu;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:    if (useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "groups if using GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "SHAKE is not supported with GPU update.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        GMX_RELEASE_ASSERT(useGpuForPme || (useGpuForNonbonded && simulationWork.useGpuXBufferOpsWhenAllowed),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "the GPU to use GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Only the md integrator is supported with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                "Nose-Hoover temperature coupling is not supported with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                "with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Virtual sites are not supported with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Essential dynamics is not supported with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Constraints pulling is not supported with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Orientation restraints are not supported with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                "Free energy perturbation of masses and constraints are not supported with the GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                    .appendText("Updating coordinates and applying constraints on the GPU.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            GMX_LOG(mdLog_.info).asParagraph().appendText("Updating coordinates on the GPU.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                           "Device stream manager should be initialized in order to use GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                "Update stream should be initialized in order to use GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        integrator = std::make_unique<UpdateConstrainGpu>(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        stateGpu->setXUpdatedOnDeviceEvent(integrator->xUpdatedOnDeviceEvent());
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:    if (useGpuForPme || simulationWork.useGpuXBufferOpsWhenAllowed || useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:    if (useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:     * Disable PME tuning with GPU PME decomposition */
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                && ir->cutoff_scheme != CutoffScheme::Group && !simulationWork.useGpuPmeDecomposition);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                &pme_loadbal, cr_, mdLog_, *ir, state_->box, *fr_->ic, *fr_->nbv, fr_->pmedata, fr_->nbv->useGpu());
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:    bool usedMdGpuGraphLastStep = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate && !bFirstStep)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->copyCoordinatesFromGpu(state_->x, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            /* PME grid + cut-off optimization with GPUs or PME nodes */
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                           simulationWork.useGpuPmePpCommunication);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        // On search steps, when doing the update on the GPU, copy
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (useGpuForUpdate && bNS && !bFirstStep && !bExchanged)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (usedMdGpuGraphLastStep)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                // Wait on coordinates produced from GPU graph
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->waitCoordinatesUpdatedOnDevice();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->copyVelocitiesFromGpu(state_->v, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->copyCoordinatesFromGpu(state_->x, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            // the GPU Update object should be informed
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate && (bMainState || bExchanged))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        // Allocate or re-size GPU halo exchange object, if necessary
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (bNS && simulationWork.havePpDomainDecomposition && simulationWork.useGpuHaloExchange)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                               "GPU device manager has to be initialized to use GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            constructGpuHaloExchange(*cr_, *fr_->deviceStreamManager, wallCycleCounters_);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (fr_->listedForcesGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                fr_->listedForcesGpu->updateHaveInteractions(top_->idef);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        MdGpuGraph* mdGraph = simulationWork.useMdGpuGraph ? fr_->mdGraph[step % 2].get() : nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (simulationWork.useMdGpuGraph)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                mdGraph->setUsedGraphLastStep(usedMdGpuGraphLastStep);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                bool canUseMdGpuGraphThisStep =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                if (mdGraph->captureThisStep(canUseMdGpuGraphThisStep))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                    mdGraph->startRecord(stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (!simulationWork.useMdGpuGraph || mdGraph->graphIsCapturingThisStep()
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            // Copy coordinate from the GPU for the output/checkpointing if the update is offloaded
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate && !bNS && !runScheduleWork_->domainWork.haveCpuLocalForceWork
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->copyCoordinatesFromGpu(state_->x, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate && !bNS
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->copyVelocitiesFromGpu(state_->v, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            // Copy forces for the output if the forces were reduced on the GPU (not the case on virial steps)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            // and update is offloaded hence forces are kept on the GPU for update and have not been
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            //       when the forces are ready on the GPU -- the same synchronizer should be used as the one
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            //       prior to GPU update.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (runScheduleWork_->stepWork.useGpuFBufferOps
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                && (simulationWork.useGpuUpdate && !virtualSites_) && do_per_step(step, ir->nstfout))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->copyForcesFromGpu(f.view().force(), AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (!useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                GMX_ASSERT(!useGpuForUpdate, "GPU update is not supported with VVAK integrator.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                if (useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                        integrator->set(stateGpu->getCoordinates(),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                                        stateGpu->getVelocities(),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                                        stateGpu->getForces(),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                        // Copy data to the GPU after buffers might have been reinitialized
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->copyVelocitiesToGpu(state_->v, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                    // Copy x to the GPU unless we have already transferred in do_force().
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                    // We transfer in do_force() if a GPU force task requires x (PME or x buffer ops).
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                    if (!(runScheduleWork_->stepWork.haveGpuPmeOnThisRank
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                          || runScheduleWork_->stepWork.useGpuXBufferOps))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->copyCoordinatesToGpu(state_->x, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->consumeCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                    if ((simulationWork.useGpuPme && simulationWork.useCpuPmePpCommunication)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                        || (!runScheduleWork_->stepWork.useGpuFBufferOps))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                        // rest of the forces computed on the GPU, so the final forces have to be
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                        // copied back to the GPU. Or the buffer ops were not offloaded this step,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->copyForcesToGpu(f.view().force(), AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                            stateGpu->getLocalForcesReadyOnDeviceEvent(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (simulationWork.useMdGpuGraph)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            GMX_ASSERT((mdGraph != nullptr), "MD GPU graph does not exist.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                // update): with PME tuning, since the GPU kernels
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            usedMdGpuGraphLastStep = mdGraph->useGraphThisStep();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            if (useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->copyCoordinatesFromGpu(state_->x, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->copyVelocitiesFromGpu(state_->v, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                    stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                    if (useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->resetCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->copyCoordinatesToGpu(state_->x, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                        stateGpu->waitCoordinatesCopiedToDevice(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:                            stateGpu->copyVelocitiesToGpu(state_->v, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        const bool scaleCoordinates = !useGpuForUpdate || bDoReplEx;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (useGpuForUpdate
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        // any run that uses GPUs must be at least offloading nonbondeds
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        const bool usingGpu = simulationWork.useGpuNonbonded;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        if (usingGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:            // ensure that GPU errors do not propagate between MD steps
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp.preplumed:        pme_loadbal_done(pme_loadbal, fpLog_, mdLog_, fr_->nbv->useGpu());
patches/gromacs-2024.3.diff/src/gromacs/mdrun/rerun.cpp.preplumed:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/rerun.cpp.preplumed:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/rerun.cpp.preplumed:                                 runScheduleWork_->simulationWork.useGpuPme);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/rerun.cpp.preplumed:            if (fr_->listedForcesGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/rerun.cpp.preplumed:                fr_->listedForcesGpu->updateHaveInteractions(top_->idef);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/minimize.cpp.preplumed:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/minimize.cpp.preplumed:        if (fr->listedForcesGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/minimize.cpp.preplumed:            fr->listedForcesGpu->updateHaveInteractions(top->idef);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/rerun.cpp:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/rerun.cpp:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/rerun.cpp:                                 runScheduleWork_->simulationWork.useGpuPme);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/rerun.cpp:            if (fr_->listedForcesGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/rerun.cpp:                fr_->listedForcesGpu->updateHaveInteractions(top_->idef);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/minimize.cpp:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/minimize.cpp:        if (fr->listedForcesGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/minimize.cpp:            fr->listedForcesGpu->updateHaveInteractions(top->idef);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/domdec/gpuhaloexchange.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/gpu_utils/device_stream_manager.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/gpu_utils/gpu_utils.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/mdlib/mdgraph_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/mdlib/update_constrain_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/mdtypes/state_propagator_data_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:#include "gromacs/nbnxm/gpu_data_mgmt.h"
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:    /* PME load balancing data for GPU kernels */
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:    const bool  useGpuForPme       = simulationWork.useGpuPme;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:    const bool  useGpuForNonbonded = simulationWork.useGpuNonbonded;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:    const bool  useGpuForUpdate    = simulationWork.useGpuUpdate;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                                 useGpuForPme);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                   (simulationWork.useGpuFBufferOpsWhenAllowed || useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:    std::unique_ptr<UpdateConstrainGpu> integrator;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:    StatePropagatorDataGpu* stateGpu = fr_->stateGpu;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:    if (useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                           "groups if using GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                           "SHAKE is not supported with GPU update.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        GMX_RELEASE_ASSERT(useGpuForPme || (useGpuForNonbonded && simulationWork.useGpuXBufferOpsWhenAllowed),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                           "the GPU to use GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                           "Only the md integrator is supported with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                "Nose-Hoover temperature coupling is not supported with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                "with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                           "Virtual sites are not supported with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                           "Essential dynamics is not supported with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                           "Constraints pulling is not supported with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                           "Orientation restraints are not supported with the GPU update.\n");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                "Free energy perturbation of masses and constraints are not supported with the GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                    .appendText("Updating coordinates and applying constraints on the GPU.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            GMX_LOG(mdLog_.info).asParagraph().appendText("Updating coordinates on the GPU.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                           "Device stream manager should be initialized in order to use GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                "Update stream should be initialized in order to use GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        integrator = std::make_unique<UpdateConstrainGpu>(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        stateGpu->setXUpdatedOnDeviceEvent(integrator->xUpdatedOnDeviceEvent());
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:    if (useGpuForPme || simulationWork.useGpuXBufferOpsWhenAllowed || useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:    if (useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:     * Disable PME tuning with GPU PME decomposition */
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                && ir->cutoff_scheme != CutoffScheme::Group && !simulationWork.useGpuPmeDecomposition);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                &pme_loadbal, cr_, mdLog_, *ir, state_->box, *fr_->ic, *fr_->nbv, fr_->pmedata, fr_->nbv->useGpu());
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:    bool usedMdGpuGraphLastStep = false;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate && !bFirstStep)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                stateGpu->copyCoordinatesFromGpu(state_->x, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            /* PME grid + cut-off optimization with GPUs or PME nodes */
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                           simulationWork.useGpuPmePpCommunication);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        // On search steps, when doing the update on the GPU, copy
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        if (useGpuForUpdate && bNS && !bFirstStep && !bExchanged)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            if (usedMdGpuGraphLastStep)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                // Wait on coordinates produced from GPU graph
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                stateGpu->waitCoordinatesUpdatedOnDevice();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            stateGpu->copyVelocitiesFromGpu(state_->v, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            stateGpu->copyCoordinatesFromGpu(state_->x, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            // the GPU Update object should be informed
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate && (bMainState || bExchanged))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        // Allocate or re-size GPU halo exchange object, if necessary
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        if (bNS && simulationWork.havePpDomainDecomposition && simulationWork.useGpuHaloExchange)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                               "GPU device manager has to be initialized to use GPU "
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            constructGpuHaloExchange(*cr_, *fr_->deviceStreamManager, wallCycleCounters_);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            if (fr_->listedForcesGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                fr_->listedForcesGpu->updateHaveInteractions(top_->idef);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        MdGpuGraph* mdGraph = simulationWork.useMdGpuGraph ? fr_->mdGraph[step % 2].get() : nullptr;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        if (simulationWork.useMdGpuGraph)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                mdGraph->setUsedGraphLastStep(usedMdGpuGraphLastStep);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                bool canUseMdGpuGraphThisStep =
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                if (mdGraph->captureThisStep(canUseMdGpuGraphThisStep))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                    mdGraph->startRecord(stateGpu->getCoordinatesReadyOnDeviceEvent(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        if (!simulationWork.useMdGpuGraph || mdGraph->graphIsCapturingThisStep()
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            // Copy coordinate from the GPU for the output/checkpointing if the update is offloaded
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate && !bNS && !runScheduleWork_->domainWork.haveCpuLocalForceWork
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                stateGpu->copyCoordinatesFromGpu(state_->x, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate && !bNS
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                stateGpu->copyVelocitiesFromGpu(state_->v, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            // Copy forces for the output if the forces were reduced on the GPU (not the case on virial steps)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            // and update is offloaded hence forces are kept on the GPU for update and have not been
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            //       when the forces are ready on the GPU -- the same synchronizer should be used as the one
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            //       prior to GPU update.
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            if (runScheduleWork_->stepWork.useGpuFBufferOps
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                && (simulationWork.useGpuUpdate && !virtualSites_) && do_per_step(step, ir->nstfout))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                stateGpu->copyForcesFromGpu(f.view().force(), AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            if (!useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                GMX_ASSERT(!useGpuForUpdate, "GPU update is not supported with VVAK integrator.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                if (useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                        integrator->set(stateGpu->getCoordinates(),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                                        stateGpu->getVelocities(),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                                        stateGpu->getForces(),
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                        // Copy data to the GPU after buffers might have been reinitialized
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->copyVelocitiesToGpu(state_->v, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                    // Copy x to the GPU unless we have already transferred in do_force().
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                    // We transfer in do_force() if a GPU force task requires x (PME or x buffer ops).
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                    if (!(runScheduleWork_->stepWork.haveGpuPmeOnThisRank
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                          || runScheduleWork_->stepWork.useGpuXBufferOps))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->copyCoordinatesToGpu(state_->x, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->consumeCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                    if ((simulationWork.useGpuPme && simulationWork.useCpuPmePpCommunication)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                        || (!runScheduleWork_->stepWork.useGpuFBufferOps))
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                        // rest of the forces computed on the GPU, so the final forces have to be
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                        // copied back to the GPU. Or the buffer ops were not offloaded this step,
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->copyForcesToGpu(f.view().force(), AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                            stateGpu->getLocalForcesReadyOnDeviceEvent(
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        if (simulationWork.useMdGpuGraph)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            GMX_ASSERT((mdGraph != nullptr), "MD GPU graph does not exist.");
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                // update): with PME tuning, since the GPU kernels
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            usedMdGpuGraphLastStep = mdGraph->useGraphThisStep();
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            if (useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->copyCoordinatesFromGpu(state_->x, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->copyVelocitiesFromGpu(state_->v, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                    stateGpu->waitVelocitiesReadyOnHost(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                    if (useGpuForUpdate)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->resetCoordinatesCopiedToDeviceEvent(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->copyCoordinatesToGpu(state_->x, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                        stateGpu->waitCoordinatesCopiedToDevice(AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:                            stateGpu->copyVelocitiesToGpu(state_->v, AtomLocality::Local);
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        const bool scaleCoordinates = !useGpuForUpdate || bDoReplEx;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        if (useGpuForUpdate
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        // any run that uses GPUs must be at least offloading nonbondeds
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        const bool usingGpu = simulationWork.useGpuNonbonded;
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        if (usingGpu)
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:            // ensure that GPU errors do not propagate between MD steps
patches/gromacs-2024.3.diff/src/gromacs/mdrun/md.cpp:        pme_loadbal_done(pme_loadbal, fpLog_, mdLog_, fr_->nbv->useGpu());
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:if (GMX_CLANG_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    include(gmxClangCudaUtils)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:set_property(GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:add_subdirectory(gpu_utils)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:# Mark some shared GPU implementation files to compile with CUDA if needed
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    get_property(CUDA_SOURCES GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    set_source_files_properties(${CUDA_SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        # needed as we need to include cufftmp include path before CUDA include path
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        cuda_include_directories(${cuFFTMp_INCLUDE_DIR})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    # Work around FindCUDA that prevents using target_link_libraries()
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    if (NOT GMX_CLANG_CUDA AND NOT GMX_NVSHMEM)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        gmx_cuda_add_library(libgromacs ${LIBGROMACS_SOURCES})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        target_link_libraries(libgromacs PRIVATE ${CUDA_CUFFT_LIBRARIES})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        set_target_properties(libgromacs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        set_target_properties(libgromacs PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        set_target_properties(libgromacs PROPERTIES CUDA_ARCHITECTURES "${GMX_NVSHMEM_LINK_ARCHS}")
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        # We need to PUBLIC link to the stub libraries nvml/cuda to WAR an issue
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        target_link_libraries(libgromacs PUBLIC ${GMX_CUDA_DRV_LIB} ${GMX_NVIDIA_ML_LIB})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:if(GMX_GPU_FFT_ROCFFT)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_FFT_CLFFT)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    if (NOT GMX_GPU_OPENCL)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        message(FATAL_ERROR "clFFT is only supported in OpenCL builds")
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:"An OpenCL build was requested with Visual Studio compiler, but GROMACS
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:clFFT to help with building for OpenCL, but that clFFT has not yet been
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:requires. Thus for now, OpenCL is not available with MSVC and the internal
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:installing a clFFT package, use VkFFT by setting -DGMX_GPU_FFT_LIBRARY=VkFFT,
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:# CUDA runtime headers
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_CUDA AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        set(GMX_CUDA_CLANG_FLAGS "${GMX_CUDA_CLANG_FLAGS} ${_compile_flag}")
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    if (GMX_CLANG_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:            get_source_file_property(_cuda_source_format ${_file} CUDA_SOURCE_PROPERTY_FORMAT)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:            if ("${_ext}" STREQUAL ".cu" OR _cuda_source_format)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:                gmx_compile_cuda_file_with_clang(${_file})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        get_property(CUDA_SOURCES GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        set_source_files_properties(${CUDA_SOURCES} PROPERTIES COMPILE_FLAGS ${GMX_CUDA_CLANG_FLAGS})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:if (GMX_GPU_SYCL)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:                      ${OpenCL_LIBRARIES}
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:                      $<BUILD_INTERFACE:gpu_utils>
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:# Technically, the user could want to do this for an OpenCL build
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:# using the CUDA runtime, but currently there's no reason to want to
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:if (INSTALL_CUDART_LIB) #can be set manual by user
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    if (GMX_GPU_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        foreach(CUDA_LIB ${CUDA_LIBRARIES})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:            string(REGEX MATCH "cudart" IS_CUDART ${CUDA_LIB})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:            if(IS_CUDART) #libcuda should not be installed
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:                file(GLOB CUDA_LIBS ${CUDA_LIB}*)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:                install(FILES ${CUDA_LIBS} DESTINATION
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        message(WARNING "INSTALL_CUDART_LIB only makes sense when configuring for CUDA support")
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:if(GMX_GPU_OPENCL)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        gpu_utils/vectype_ops.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        gpu_utils/device_utils.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        DESTINATION ${GMX_INSTALL_OCLDIR}/gromacs/gpu_utils
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernels.cl
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernel.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernel_pruneonly.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernels.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernels_fastgen.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernels_fastgen_add_twincut.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_kernel_utils.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        nbnxm/opencl/nbnxm_ocl_consts.h
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        DESTINATION ${GMX_INSTALL_OCLDIR}/gromacs/nbnxm/opencl
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        ewald/pme_gpu_calculate_splines.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:        ewald/pme_gpu_types.h
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_CLANG_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    include(gmxClangCudaUtils)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:set_property(GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:add_subdirectory(gpu_utils)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:# Mark some shared GPU implementation files to compile with CUDA if needed
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    get_property(CUDA_SOURCES GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    set_source_files_properties(${CUDA_SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        # needed as we need to include cufftmp include path before CUDA include path
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        cuda_include_directories(${cuFFTMp_INCLUDE_DIR})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    # Work around FindCUDA that prevents using target_link_libraries()
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    if (NOT GMX_CLANG_CUDA AND NOT GMX_NVSHMEM)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        gmx_cuda_add_library(libgromacs ${LIBGROMACS_SOURCES})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        target_link_libraries(libgromacs PRIVATE ${CUDA_CUFFT_LIBRARIES})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        set_target_properties(libgromacs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        set_target_properties(libgromacs PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        set_target_properties(libgromacs PROPERTIES CUDA_ARCHITECTURES "${GMX_NVSHMEM_LINK_ARCHS}")
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        # We need to PUBLIC link to the stub libraries nvml/cuda to WAR an issue
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        target_link_libraries(libgromacs PUBLIC ${GMX_CUDA_DRV_LIB} ${GMX_NVIDIA_ML_LIB})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:if(GMX_GPU_FFT_ROCFFT)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_FFT_CLFFT)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    if (NOT GMX_GPU_OPENCL)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        message(FATAL_ERROR "clFFT is only supported in OpenCL builds")
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:"An OpenCL build was requested with Visual Studio compiler, but GROMACS
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:clFFT to help with building for OpenCL, but that clFFT has not yet been
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:requires. Thus for now, OpenCL is not available with MSVC and the internal
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:installing a clFFT package, use VkFFT by setting -DGMX_GPU_FFT_LIBRARY=VkFFT,
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:# CUDA runtime headers
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_CUDA AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        set(GMX_CUDA_CLANG_FLAGS "${GMX_CUDA_CLANG_FLAGS} ${_compile_flag}")
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    if (GMX_CLANG_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:            get_source_file_property(_cuda_source_format ${_file} CUDA_SOURCE_PROPERTY_FORMAT)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:            if ("${_ext}" STREQUAL ".cu" OR _cuda_source_format)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:                gmx_compile_cuda_file_with_clang(${_file})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        get_property(CUDA_SOURCES GLOBAL PROPERTY CUDA_SOURCES)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        set_source_files_properties(${CUDA_SOURCES} PROPERTIES COMPILE_FLAGS ${GMX_CUDA_CLANG_FLAGS})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:if (GMX_GPU_SYCL)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:                      ${OpenCL_LIBRARIES}
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:                      $<BUILD_INTERFACE:gpu_utils>
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:# Technically, the user could want to do this for an OpenCL build
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:# using the CUDA runtime, but currently there's no reason to want to
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:if (INSTALL_CUDART_LIB) #can be set manual by user
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    if (GMX_GPU_CUDA)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        foreach(CUDA_LIB ${CUDA_LIBRARIES})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:            string(REGEX MATCH "cudart" IS_CUDART ${CUDA_LIB})
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:            if(IS_CUDART) #libcuda should not be installed
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:                file(GLOB CUDA_LIBS ${CUDA_LIB}*)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:                install(FILES ${CUDA_LIBS} DESTINATION
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        message(WARNING "INSTALL_CUDART_LIB only makes sense when configuring for CUDA support")
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:if(GMX_GPU_OPENCL)
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        gpu_utils/vectype_ops.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        gpu_utils/device_utils.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        DESTINATION ${GMX_INSTALL_OCLDIR}/gromacs/gpu_utils
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernels.cl
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernel.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernel_pruneonly.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernels.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernels_fastgen.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernels_fastgen_add_twincut.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_kernel_utils.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        nbnxm/opencl/nbnxm_ocl_consts.h
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        DESTINATION ${GMX_INSTALL_OCLDIR}/gromacs/nbnxm/opencl
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    file(GLOB OPENCL_INSTALLED_FILES
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        ewald/pme_gpu_calculate_splines.clh
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:        ewald/pme_gpu_types.h
patches/gromacs-2024.3.diff/src/gromacs/CMakeLists.txt.preplumed:    install(FILES ${OPENCL_INSTALLED_FILES}
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed: * \brief Defines functionality for deciding whether tasks will run on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#include "gromacs/taskassignment/decidegpuusage.h"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#include "gromacs/mdlib/update_constrain_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        "When you use mdrun -gputasks, %s must be set to non-default "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#if GMX_GPU
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        " If you simply want to restrict which GPUs are used, then it is "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        "better to use mdrun -gpu_id. Otherwise, setting the "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#    if GMX_GPU_CUDA
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        "CUDA_VISIBLE_DEVICES"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#    elif GMX_GPU_OPENCL
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // OpenCL standard, but the only current relevant case for GROMACS
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // is AMD OpenCL, which offers this variable.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        "GPU_DEVICE_ORDINAL"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#    elif GMX_GPU_SYCL && GMX_SYCL_DPCPP
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#    elif GMX_GPU_SYCL && GMX_SYCL_HIPSYCL && GMX_HIPSYCL_HAVE_HIP_TARGET
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:#    elif GMX_GPU_SYCL && GMX_SYCL_HIPSYCL && GMX_HIPSYCL_HAVE_CUDA_TARGET
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        "CUDA_VISIBLE_DEVIES"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:constexpr bool c_gpuBuildSyclWithoutGpuFft =
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        (GMX_GPU_SYCL != 0) && (GMX_GPU_FFT_MKL == 0) && (GMX_GPU_FFT_ROCFFT == 0)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        && (GMX_GPU_FFT_VKFFT == 0) && (GMX_GPU_FFT_BBFFT == 0); // NOLINT(misc-redundant-expression)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpusForNonbondedWithThreadMpi(const TaskTarget        nonbondedTarget,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                                     const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                                     const EmulateGpuNonbonded emulateGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                                     const bool buildSupportsNonbondedOnGpu,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                                     const bool nonbondedOnGpuIsUseful,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // First, exclude all cases where we can't run NB on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (nonbondedTarget == TaskTarget::Cpu || emulateGpuNonbonded == EmulateGpuNonbonded::Yes
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        || !nonbondedOnGpuIsUseful || binaryReproducibilityRequested || !buildSupportsNonbondedOnGpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // If the user required NB on GPUs, we issue an error later.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // We now know that NB on GPUs makes sense, if we have any.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Because this is thread-MPI, we already know about the GPUs that
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // If we get here, then the user permitted or required GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:static bool decideWhetherToUseGpusForPmeFft(const TaskTarget pmeFftTarget)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                           || (pmeFftTarget == TaskTarget::Auto && c_gpuBuildSyclWithoutGpuFft);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:static bool canUseGpusForPme(const bool        useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    errorReasons.startContext("Cannot compute PME interactions on a GPU, because:");
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    errorReasons.appendIf(!useGpuForNonbonded, "Nonbonded interactions must also run on GPUs.");
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    errorReasons.appendIf(!pme_gpu_supports_build(&tempString), tempString);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    errorReasons.appendIf(!pme_gpu_supports_input(inputrec, &tempString), tempString);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!decideWhetherToUseGpusForPmeFft(pmeFftTarget))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        errorReasons.appendIf(!pme_gpu_mixed_mode_supports_input(inputrec, &tempString), tempString);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (pmeTarget == TaskTarget::Gpu && errorMessage != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpusForPmeWithThreadMpi(const bool              useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                               const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // First, exclude all cases where we can't run PME on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!canUseGpusForPme(useGpuForNonbonded, pmeTarget, pmeFftTarget, inputrec, nullptr))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // PME can't run on a GPU. If the user required that, we issue an error later.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // We now know that PME on GPUs might make sense, if we have any.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "PME tasks were required to run on GPUs with multiple ranks "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // Follow the user's choice of GPU task assignment, if we
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // can. Checking that their IDs are for compatible GPUs comes
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // PME on GPUs is only supported in a single case
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                        "When you run mdrun -pme gpu -gputasks, you must supply a PME-enabled .tpr "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Because this is thread-MPI, we already know about the GPUs that
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "PME tasks were required to run on GPUs, but that is not implemented with "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // PME can run well on a GPU shared with NB, and we permit
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // We have a single separate PME rank, that can use a GPU
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // run well on a GPU shared with NB, and we permit mdrun to
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // default to it if there is only one GPU available.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Not enough support for PME on GPUs for anything else
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpusForNonbonded(const TaskTarget          nonbondedTarget,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                        const std::vector<int>&   userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                        const EmulateGpuNonbonded emulateGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                        const bool                buildSupportsNonbondedOnGpu,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                        const bool                nonbondedOnGpuIsUseful,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                        const bool                gpusWereDetected)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "A GPU task assignment was specified, but nonbonded interactions were "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!buildSupportsNonbondedOnGpu && nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                "Nonbonded interactions on the GPU were requested with -nb gpu, "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                "but the GROMACS binary has been built without GPU support. "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                "Either run without selecting GPU options, or recompile GROMACS "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                "with GPU support enabled"));
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // TODO refactor all these TaskTarget::Gpu checks into one place?
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (emulateGpuNonbonded == EmulateGpuNonbonded::Yes)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "Nonbonded interactions on the GPU were required, which is inconsistent "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    InconsistentInputError("GPU ID usage was specified, as was GPU emulation. Make "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!nonbondedOnGpuIsUseful)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "Nonbonded interactions on the GPU were required, but not supported for these "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "simulation settings. Change your settings, or do not require using GPUs."));
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "Nonbonded interactions on the GPU and binary reprocibility were required. "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // We still don't know whether it is an error if no GPUs are found
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // If we get here, then the user permitted GPUs, which we should
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    return buildSupportsNonbondedOnGpu && gpusWereDetected;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpusForPme(const bool              useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                  const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                  const bool              gpusWereDetected)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!canUseGpusForPme(useGpuForNonbonded, pmeTarget, pmeFftTarget, inputrec, &message))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "A GPU task assignment was specified, but PME interactions were "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "PME tasks were required to run on GPUs with multiple ranks "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // We still don't know whether it is an error if no GPUs are found
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // If we get here, then the user permitted GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // PME can run well on a single GPU shared with NB when there
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // detected GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        return gpusWereDetected;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // We have a single separate PME rank, that can use a GPU
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        return gpusWereDetected;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Not enough support for PME on GPUs for anything else
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:PmeRunMode determinePmeRunMode(const bool useGpuForPme, const TaskTarget& pmeFftTarget, const t_inputrec& inputrec)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (useGpuForPme)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (c_gpuBuildSyclWithoutGpuFft && pmeFftTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                      "GROMACS is built without SYCL GPU FFT library. Please do not use -pmefft "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                      "gpu.");
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (!decideWhetherToUseGpusForPmeFft(pmeFftTarget))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:            return PmeRunMode::GPU;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (pmeFftTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                      "Assigning FFTs to GPU requires PME to be assigned to GPU as well. With PME "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpusForBonded(bool              useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                     bool              useGpuForPme,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                     bool              gpusWereDetected)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!buildSupportsListedForcesGpu(&errorMessage))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!inputSupportsListedForcesGpu(inputrec, mtop, &errorMessage))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!useGpuForNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "Bonded interactions on the GPU were required, but this requires that "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "short-ranged non-bonded interactions are also run on the GPU. Change "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "your settings, or do not require using GPUs."));
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // We still don't know whether it is an error if no GPUs are
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // If we get here, then the user permitted GPUs, which we should
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // choose separate PME ranks when nonBonded are assigned to the GPU.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                     || (usingPmeOrEwald(inputrec.coulombtype) && !useGpuForPme
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    return gpusWereDetected && usingOurCpuForPmeOrEwald;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpuForUpdate(const bool           isDomainDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                    const bool           useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                    const bool           gpusWereDetected,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Flag to set if we do not want to log the error with `-update auto` (e.g., for non-GPU build)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:            errorMessage += "With separate PME rank(s), PME must run on the GPU.\n";
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Using the GPU-version of update if:
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // 1. PME is on the GPU (there should be a copy of coordinates on GPU for PME spread) or inactive, or
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // 2. Non-bonded interactions are on the GPU.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if ((pmeRunMode == PmeRunMode::CPU || pmeRunMode == PmeRunMode::None) && !useGpuForNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                "Either PME or short-ranged non-bonded interaction tasks must run on the GPU.\n";
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!gpusWereDetected)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        errorMessage += "Compatible GPUs must have been found.\n";
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!(GMX_GPU_CUDA || GMX_GPU_SYCL))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        errorMessage += "Only CUDA and SYCL builds are supported.\n";
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // does not support it, the actual CUDA LINCS code does support it
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!UpdateConstrainGpu::isNumCoupledConstraintsSupported(mtop))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                "The number of coupled constraints is higher than supported in the GPU LINCS "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (hasAnyConstraints && !UpdateConstrainGpu::areConstraintsSupported())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        errorMessage += "Chosen GPU implementation does not support constraints.\n";
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        // There is a known bug with frozen atoms and GPU update, see Issue #3920.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                            "Update task can not run on the GPU, because the following "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:        else if (updateTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                    "Update task on the GPU was required,\n"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    return (updateTarget == TaskTarget::Gpu
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherDirectGpuCommunicationCanBeUsed(const DevelopmentFeatureFlags& devFlags,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    const bool buildSupportsDirectGpuComm = (GMX_GPU_CUDA || GMX_GPU_SYCL) && GMX_MPI;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!buildSupportsDirectGpuComm)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Direct GPU communication is presently turned off due to insufficient testing
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    const bool enableDirectGpuComm = (getenv("GMX_ENABLE_DIRECT_GPU_COMM") != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                     || (getenv("GMX_GPU_DD_COMMS") != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                     || (getenv("GMX_GPU_PME_PP_COMMS") != nullptr);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (GMX_THREAD_MPI && GMX_GPU_SYCL && enableDirectGpuComm)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                        "GMX_ENABLE_DIRECT_GPU_COMM environment variable detected, "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    errorReasons.startContext("GPU direct communication can not be activated because:");
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    bool runAndGpuSupportDirectGpuComm = (runUsesCompatibleFeatures && enableDirectGpuComm);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    bool canUseDirectGpuCommWithThreadMpi =
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:            (runAndGpuSupportDirectGpuComm && GMX_THREAD_MPI && !GMX_GPU_SYCL);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // GPU-aware MPI case off by default, can be enabled with dev flag
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    // Note: GMX_DISABLE_DIRECT_GPU_COMM already taken into account in devFlags.enableDirectGpuCommWithMpi
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    bool canUseDirectGpuCommWithMpi = (runAndGpuSupportDirectGpuComm && GMX_LIB_MPI
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                       && devFlags.canUseGpuAwareMpi && enableDirectGpuComm);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    return canUseDirectGpuCommWithThreadMpi || canUseDirectGpuCommWithMpi;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:bool decideWhetherToUseGpuForHalo(bool                 havePPDomainDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                  bool                 useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:                                  bool                 canUseDirectGpuComm,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    if (!canUseDirectGpuComm || !havePPDomainDecomposition || !useGpuForNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp.preplumed:    errorReasons.startContext("GPU halo exchange will not be activated because:");
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \brief Declares functionality for deciding whether tasks will run on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:#ifndef GMX_TASKASSIGNMENT_DECIDEGPUUSAGE_H
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:#define GMX_TASKASSIGNMENT_DECIDEGPUUSAGE_H
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    Gpu
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed://! Help pass GPU-emulation parameters with type safety.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:enum class EmulateGpuNonbonded : bool
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    //! Do not emulate GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    //! Do emulate GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    bool enableGpuBufferOps = false;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    //! True if the GPU-aware MPI can be used for GPU direct communication feature
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    bool canUseGpuAwareMpi = false;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    //! True if GPU PME-decomposition is enabled
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    bool enableGpuPmeDecomposition = false;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    //! True if CUDA Graphs are enabled
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    bool enableCudaGraphs = false;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:    //! True if NVSHMEM can be used for GPU communication
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * nonbonded tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * The number of GPU tasks and devices influences both the choice of
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in] userGpuTaskAssignment        The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in] emulateGpuNonbonded          Whether we will emulate GPU calculation of nonbonded
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in] buildSupportsNonbondedOnGpu  Whether GROMACS was built with GPU support.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in] nonbondedOnGpuIsUseful       Whether computing nonbonded interactions on a GPU is
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether the simulation will run nonbonded tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpusForNonbondedWithThreadMpi(TaskTarget              nonbondedTarget,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                                     const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                                     EmulateGpuNonbonded     emulateGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                                     bool buildSupportsNonbondedOnGpu,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                                     bool nonbondedOnGpuIsUseful,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * PME tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * The number of GPU tasks and devices influences both the choice of
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForNonbonded        Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  numDevicesToUse           The number of compatible GPUs that the user permitted us to use.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  userGpuTaskAssignment     The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether the simulation will run PME tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpusForPmeWithThreadMpi(bool                    useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                               const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * is known. But we need to know if nonbonded will run on GPUs for
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * user requires GPUs for the tasks of that duty, then it will be an
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * decideWhetherToUseGpusForNonbondedWithThreadMpi() and
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * decideWhetherToUseGpusForPmeWithThreadMpi() to help determine
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  userGpuTaskAssignment       The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  emulateGpuNonbonded         Whether we will emulate GPU calculation of nonbonded interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  buildSupportsNonbondedOnGpu Whether GROMACS was build with GPU support.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  nonbondedOnGpuIsUseful      Whether computing nonbonded interactions on a GPU is useful for this calculation.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  gpusWereDetected            Whether compatible GPUs were detected on any node.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether the simulation will run nonbonded and PME tasks, respectively, on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpusForNonbonded(TaskTarget              nonbondedTarget,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                        const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                        EmulateGpuNonbonded     emulateGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                        bool                    buildSupportsNonbondedOnGpu,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                        bool                    nonbondedOnGpuIsUseful,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                        bool                    gpusWereDetected);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * different types on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * is known. But we need to know if nonbonded will run on GPUs for
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * user requires GPUs for the tasks of that duty, then it will be an
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * decideWhetherToUseGpusForNonbondedWithThreadMpi() and
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * decideWhetherToUseGpusForPmeWithThreadMpi() to help determine
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForNonbonded        Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  userGpuTaskAssignment     The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  gpusWereDetected          Whether compatible GPUs were detected on any node.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether the simulation will run nonbonded and PME tasks, respectively, on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpusForPme(bool                    useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                  const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                  bool                    gpusWereDetected);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * Given the PME task assignment in \p useGpuForPme and the user-provided
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \note Aborts the run upon incompatible values of \p useGpuForPme and \p pmeFftTarget.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForPme              PME task assignment, true if PME task is mapped to the GPU.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:PmeRunMode determinePmeRunMode(bool useGpuForPme, const TaskTarget& pmeFftTarget, const t_inputrec& inputrec);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:/*! \brief Decide whether the simulation will try to run bonded tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForNonbonded        Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForPme              Whether GPUs will be used for PME interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  gpusWereDetected          Whether compatible GPUs were detected on any node.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether the simulation will run bondeded tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpusForBonded(bool              useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                     bool              useGpuForPme,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                     bool              gpusWereDetected);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:/*! \brief Decide whether to use GPU for update.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  pmeRunMode                   PME running mode: CPU, GPU or mixed.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForNonbonded           Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  updateTarget                 User choice for running simulation on GPU.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  gpusWereDetected             Whether compatible GPUs were detected on any node.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether complete simulation can be run on GPU.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpuForUpdate(bool                 isDomainDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                    bool                 useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                    bool                 gpusWereDetected,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:/*! \brief Decide whether direct GPU communication can be used.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * Takes into account the build type which determines feature support as well as GPU
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * development feature flags, determines whether this run can use direct GPU communication.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  devFlags                     GPU development / experimental feature flags.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether the MPI-parallel runs can use direct GPU communication.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherDirectGpuCommunicationCanBeUsed(const DevelopmentFeatureFlags& devFlags,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:/*! \brief Decide whether to use GPU for halo exchange.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  useGpuForNonbonded           Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \param[in]  canUseDirectGpuComm          Whether direct GPU communication can be used.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed: * \returns    Whether halo exchange can be run on GPU.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:bool decideWhetherToUseGpuForHalo(bool                 havePPDomainDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                  bool                 useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h.preplumed:                                  bool                 canUseDirectGpuComm,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \brief Declares functionality for deciding whether tasks will run on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:#ifndef GMX_TASKASSIGNMENT_DECIDEGPUUSAGE_H
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:#define GMX_TASKASSIGNMENT_DECIDEGPUUSAGE_H
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    Gpu
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h://! Help pass GPU-emulation parameters with type safety.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:enum class EmulateGpuNonbonded : bool
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    //! Do not emulate GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    //! Do emulate GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    bool enableGpuBufferOps = false;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    //! True if the GPU-aware MPI can be used for GPU direct communication feature
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    bool canUseGpuAwareMpi = false;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    //! True if GPU PME-decomposition is enabled
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    bool enableGpuPmeDecomposition = false;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    //! True if CUDA Graphs are enabled
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    bool enableCudaGraphs = false;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:    //! True if NVSHMEM can be used for GPU communication
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * nonbonded tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * The number of GPU tasks and devices influences both the choice of
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in] userGpuTaskAssignment        The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in] emulateGpuNonbonded          Whether we will emulate GPU calculation of nonbonded
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in] buildSupportsNonbondedOnGpu  Whether GROMACS was built with GPU support.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in] nonbondedOnGpuIsUseful       Whether computing nonbonded interactions on a GPU is
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether the simulation will run nonbonded tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpusForNonbondedWithThreadMpi(TaskTarget              nonbondedTarget,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                                     const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                                     EmulateGpuNonbonded     emulateGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                                     bool buildSupportsNonbondedOnGpu,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                                     bool nonbondedOnGpuIsUseful,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * PME tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * The number of GPU tasks and devices influences both the choice of
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForNonbonded        Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  numDevicesToUse           The number of compatible GPUs that the user permitted us to use.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  userGpuTaskAssignment     The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether the simulation will run PME tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpusForPmeWithThreadMpi(bool                    useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                               const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * is known. But we need to know if nonbonded will run on GPUs for
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * user requires GPUs for the tasks of that duty, then it will be an
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * decideWhetherToUseGpusForNonbondedWithThreadMpi() and
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * decideWhetherToUseGpusForPmeWithThreadMpi() to help determine
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  userGpuTaskAssignment       The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  emulateGpuNonbonded         Whether we will emulate GPU calculation of nonbonded interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  buildSupportsNonbondedOnGpu Whether GROMACS was build with GPU support.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  nonbondedOnGpuIsUseful      Whether computing nonbonded interactions on a GPU is useful for this calculation.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  gpusWereDetected            Whether compatible GPUs were detected on any node.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether the simulation will run nonbonded and PME tasks, respectively, on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpusForNonbonded(TaskTarget              nonbondedTarget,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                        const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                        EmulateGpuNonbonded     emulateGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                        bool                    buildSupportsNonbondedOnGpu,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                        bool                    nonbondedOnGpuIsUseful,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                        bool                    gpusWereDetected);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * different types on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * is known. But we need to know if nonbonded will run on GPUs for
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * user requires GPUs for the tasks of that duty, then it will be an
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * decideWhetherToUseGpusForNonbondedWithThreadMpi() and
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * decideWhetherToUseGpusForPmeWithThreadMpi() to help determine
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForNonbonded        Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  userGpuTaskAssignment     The user-specified assignment of GPU tasks to device IDs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  gpusWereDetected          Whether compatible GPUs were detected on any node.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether the simulation will run nonbonded and PME tasks, respectively, on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpusForPme(bool                    useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                  const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                  bool                    gpusWereDetected);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * Given the PME task assignment in \p useGpuForPme and the user-provided
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \note Aborts the run upon incompatible values of \p useGpuForPme and \p pmeFftTarget.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForPme              PME task assignment, true if PME task is mapped to the GPU.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:PmeRunMode determinePmeRunMode(bool useGpuForPme, const TaskTarget& pmeFftTarget, const t_inputrec& inputrec);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:/*! \brief Decide whether the simulation will try to run bonded tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForNonbonded        Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForPme              Whether GPUs will be used for PME interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  gpusWereDetected          Whether compatible GPUs were detected on any node.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether the simulation will run bondeded tasks on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpusForBonded(bool              useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                     bool              useGpuForPme,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                     bool              gpusWereDetected);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:/*! \brief Decide whether to use GPU for update.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  pmeRunMode                   PME running mode: CPU, GPU or mixed.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForNonbonded           Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  updateTarget                 User choice for running simulation on GPU.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  gpusWereDetected             Whether compatible GPUs were detected on any node.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether complete simulation can be run on GPU.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpuForUpdate(bool                 isDomainDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                    bool                 useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                    bool                 gpusWereDetected,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:/*! \brief Decide whether direct GPU communication can be used.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * Takes into account the build type which determines feature support as well as GPU
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * development feature flags, determines whether this run can use direct GPU communication.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  devFlags                     GPU development / experimental feature flags.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether the MPI-parallel runs can use direct GPU communication.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherDirectGpuCommunicationCanBeUsed(const DevelopmentFeatureFlags& devFlags,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:/*! \brief Decide whether to use GPU for halo exchange.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  useGpuForNonbonded           Whether GPUs will be used for nonbonded interactions.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \param[in]  canUseDirectGpuComm          Whether direct GPU communication can be used.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h: * \returns    Whether halo exchange can be run on GPU.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:bool decideWhetherToUseGpuForHalo(bool                 havePPDomainDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                  bool                 useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/include/gromacs/taskassignment/decidegpuusage.h:                                  bool                 canUseDirectGpuComm,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp: * \brief Defines functionality for deciding whether tasks will run on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#include "gromacs/taskassignment/decidegpuusage.h"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#include "gromacs/listed_forces/listed_forces_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#include "gromacs/mdlib/update_constrain_gpu.h"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        "When you use mdrun -gputasks, %s must be set to non-default "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#if GMX_GPU
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        " If you simply want to restrict which GPUs are used, then it is "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        "better to use mdrun -gpu_id. Otherwise, setting the "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#    if GMX_GPU_CUDA
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        "CUDA_VISIBLE_DEVICES"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#    elif GMX_GPU_OPENCL
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // OpenCL standard, but the only current relevant case for GROMACS
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // is AMD OpenCL, which offers this variable.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        "GPU_DEVICE_ORDINAL"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#    elif GMX_GPU_SYCL && GMX_SYCL_DPCPP
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#    elif GMX_GPU_SYCL && GMX_SYCL_HIPSYCL && GMX_HIPSYCL_HAVE_HIP_TARGET
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:#    elif GMX_GPU_SYCL && GMX_SYCL_HIPSYCL && GMX_HIPSYCL_HAVE_CUDA_TARGET
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        "CUDA_VISIBLE_DEVIES"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:constexpr bool c_gpuBuildSyclWithoutGpuFft =
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        (GMX_GPU_SYCL != 0) && (GMX_GPU_FFT_MKL == 0) && (GMX_GPU_FFT_ROCFFT == 0)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        && (GMX_GPU_FFT_VKFFT == 0) && (GMX_GPU_FFT_BBFFT == 0); // NOLINT(misc-redundant-expression)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpusForNonbondedWithThreadMpi(const TaskTarget        nonbondedTarget,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                                     const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                                     const EmulateGpuNonbonded emulateGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                                     const bool buildSupportsNonbondedOnGpu,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                                     const bool nonbondedOnGpuIsUseful,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // First, exclude all cases where we can't run NB on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (nonbondedTarget == TaskTarget::Cpu || emulateGpuNonbonded == EmulateGpuNonbonded::Yes
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        || !nonbondedOnGpuIsUseful || binaryReproducibilityRequested || !buildSupportsNonbondedOnGpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // If the user required NB on GPUs, we issue an error later.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // We now know that NB on GPUs makes sense, if we have any.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Because this is thread-MPI, we already know about the GPUs that
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // If we get here, then the user permitted or required GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:static bool decideWhetherToUseGpusForPmeFft(const TaskTarget pmeFftTarget)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                           || (pmeFftTarget == TaskTarget::Auto && c_gpuBuildSyclWithoutGpuFft);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:static bool canUseGpusForPme(const bool        useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    errorReasons.startContext("Cannot compute PME interactions on a GPU, because:");
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    errorReasons.appendIf(!useGpuForNonbonded, "Nonbonded interactions must also run on GPUs.");
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    errorReasons.appendIf(!pme_gpu_supports_build(&tempString), tempString);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    errorReasons.appendIf(!pme_gpu_supports_input(inputrec, &tempString), tempString);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!decideWhetherToUseGpusForPmeFft(pmeFftTarget))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        errorReasons.appendIf(!pme_gpu_mixed_mode_supports_input(inputrec, &tempString), tempString);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (pmeTarget == TaskTarget::Gpu && errorMessage != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpusForPmeWithThreadMpi(const bool              useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                               const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // First, exclude all cases where we can't run PME on GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!canUseGpusForPme(useGpuForNonbonded, pmeTarget, pmeFftTarget, inputrec, nullptr))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // PME can't run on a GPU. If the user required that, we issue an error later.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // We now know that PME on GPUs might make sense, if we have any.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "PME tasks were required to run on GPUs with multiple ranks "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // Follow the user's choice of GPU task assignment, if we
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // can. Checking that their IDs are for compatible GPUs comes
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // PME on GPUs is only supported in a single case
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                        "When you run mdrun -pme gpu -gputasks, you must supply a PME-enabled .tpr "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Because this is thread-MPI, we already know about the GPUs that
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "PME tasks were required to run on GPUs, but that is not implemented with "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // PME can run well on a GPU shared with NB, and we permit
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // We have a single separate PME rank, that can use a GPU
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // run well on a GPU shared with NB, and we permit mdrun to
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // default to it if there is only one GPU available.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Not enough support for PME on GPUs for anything else
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpusForNonbonded(const TaskTarget          nonbondedTarget,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                        const std::vector<int>&   userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                        const EmulateGpuNonbonded emulateGpuNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                        const bool                buildSupportsNonbondedOnGpu,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                        const bool                nonbondedOnGpuIsUseful,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                        const bool                gpusWereDetected)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "A GPU task assignment was specified, but nonbonded interactions were "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!buildSupportsNonbondedOnGpu && nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                "Nonbonded interactions on the GPU were requested with -nb gpu, "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                "but the GROMACS binary has been built without GPU support. "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                "Either run without selecting GPU options, or recompile GROMACS "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                "with GPU support enabled"));
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // TODO refactor all these TaskTarget::Gpu checks into one place?
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (emulateGpuNonbonded == EmulateGpuNonbonded::Yes)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "Nonbonded interactions on the GPU were required, which is inconsistent "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    InconsistentInputError("GPU ID usage was specified, as was GPU emulation. Make "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!nonbondedOnGpuIsUseful)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "Nonbonded interactions on the GPU were required, but not supported for these "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "simulation settings. Change your settings, or do not require using GPUs."));
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "Nonbonded interactions on the GPU and binary reprocibility were required. "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (nonbondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // We still don't know whether it is an error if no GPUs are found
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // If we get here, then the user permitted GPUs, which we should
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    return buildSupportsNonbondedOnGpu && gpusWereDetected;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpusForPme(const bool              useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                  const std::vector<int>& userGpuTaskAssignment,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                  const bool              gpusWereDetected)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!canUseGpusForPme(useGpuForNonbonded, pmeTarget, pmeFftTarget, inputrec, &message))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "A GPU task assignment was specified, but PME interactions were "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "PME tasks were required to run on GPUs with multiple ranks "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!userGpuTaskAssignment.empty())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // Specifying -gputasks requires specifying everything.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // We still don't know whether it is an error if no GPUs are found
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (pmeTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // If we get here, then the user permitted GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // PME can run well on a single GPU shared with NB when there
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // detected GPUs.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        return gpusWereDetected;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // We have a single separate PME rank, that can use a GPU
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        return gpusWereDetected;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Not enough support for PME on GPUs for anything else
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:PmeRunMode determinePmeRunMode(const bool useGpuForPme, const TaskTarget& pmeFftTarget, const t_inputrec& inputrec)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (useGpuForPme)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (c_gpuBuildSyclWithoutGpuFft && pmeFftTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                      "GROMACS is built without SYCL GPU FFT library. Please do not use -pmefft "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                      "gpu.");
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (!decideWhetherToUseGpusForPmeFft(pmeFftTarget))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:            return PmeRunMode::GPU;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (pmeFftTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                      "Assigning FFTs to GPU requires PME to be assigned to GPU as well. With PME "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpusForBonded(bool              useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                     bool              useGpuForPme,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                     bool              gpusWereDetected)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!buildSupportsListedForcesGpu(&errorMessage))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!inputSupportsListedForcesGpu(inputrec, mtop, &errorMessage))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!useGpuForNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "Bonded interactions on the GPU were required, but this requires that "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "short-ranged non-bonded interactions are also run on the GPU. Change "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "your settings, or do not require using GPUs."));
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (bondedTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // We still don't know whether it is an error if no GPUs are
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // If we get here, then the user permitted GPUs, which we should
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // choose separate PME ranks when nonBonded are assigned to the GPU.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                     || (usingPmeOrEwald(inputrec.coulombtype) && !useGpuForPme
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    return gpusWereDetected && usingOurCpuForPmeOrEwald;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpuForUpdate(const bool           isDomainDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                    const bool           useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                    const bool           gpusWereDetected,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Flag to set if we do not want to log the error with `-update auto` (e.g., for non-GPU build)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:            errorMessage += "With separate PME rank(s), PME must run on the GPU.\n";
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Using the GPU-version of update if:
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // 1. PME is on the GPU (there should be a copy of coordinates on GPU for PME spread) or inactive, or
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // 2. Non-bonded interactions are on the GPU.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if ((pmeRunMode == PmeRunMode::CPU || pmeRunMode == PmeRunMode::None) && !useGpuForNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                "Either PME or short-ranged non-bonded interaction tasks must run on the GPU.\n";
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!gpusWereDetected)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        errorMessage += "Compatible GPUs must have been found.\n";
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!(GMX_GPU_CUDA || GMX_GPU_SYCL))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        errorMessage += "Only CUDA and SYCL builds are supported.\n";
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // does not support it, the actual CUDA LINCS code does support it
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!UpdateConstrainGpu::isNumCoupledConstraintsSupported(mtop))
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                "The number of coupled constraints is higher than supported in the GPU LINCS "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (hasAnyConstraints && !UpdateConstrainGpu::areConstraintsSupported())
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        errorMessage += "Chosen GPU implementation does not support constraints.\n";
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        // There is a known bug with frozen atoms and GPU update, see Issue #3920.
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                            "Update task can not run on the GPU, because the following "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:        else if (updateTarget == TaskTarget::Gpu)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                    "Update task on the GPU was required,\n"
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    return (updateTarget == TaskTarget::Gpu
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherDirectGpuCommunicationCanBeUsed(const DevelopmentFeatureFlags& devFlags,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    const bool buildSupportsDirectGpuComm = (GMX_GPU_CUDA || GMX_GPU_SYCL) && GMX_MPI;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!buildSupportsDirectGpuComm)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Direct GPU communication is presently turned off due to insufficient testing
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    const bool enableDirectGpuComm = (getenv("GMX_ENABLE_DIRECT_GPU_COMM") != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                     || (getenv("GMX_GPU_DD_COMMS") != nullptr)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                     || (getenv("GMX_GPU_PME_PP_COMMS") != nullptr);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (GMX_THREAD_MPI && GMX_GPU_SYCL && enableDirectGpuComm)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                        "GMX_ENABLE_DIRECT_GPU_COMM environment variable detected, "
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    errorReasons.startContext("GPU direct communication can not be activated because:");
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    bool runAndGpuSupportDirectGpuComm = (runUsesCompatibleFeatures && enableDirectGpuComm);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    bool canUseDirectGpuCommWithThreadMpi =
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:            (runAndGpuSupportDirectGpuComm && GMX_THREAD_MPI && !GMX_GPU_SYCL);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // GPU-aware MPI case off by default, can be enabled with dev flag
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    // Note: GMX_DISABLE_DIRECT_GPU_COMM already taken into account in devFlags.enableDirectGpuCommWithMpi
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    bool canUseDirectGpuCommWithMpi = (runAndGpuSupportDirectGpuComm && GMX_LIB_MPI
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                       && devFlags.canUseGpuAwareMpi && enableDirectGpuComm);
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    return canUseDirectGpuCommWithThreadMpi || canUseDirectGpuCommWithMpi;
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:bool decideWhetherToUseGpuForHalo(bool                 havePPDomainDecomposition,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                  bool                 useGpuForNonbonded,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:                                  bool                 canUseDirectGpuComm,
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    if (!canUseDirectGpuComm || !havePPDomainDecomposition || !useGpuForNonbonded)
patches/gromacs-2024.3.diff/src/gromacs/taskassignment/decidegpuusage.cpp:    errorReasons.startContext("GPU halo exchange will not be activated because:");
regtest/metatensor/rt-basic/cv.py:    supported_devices=["cpu", "mps", "cuda"],
regtest/metatensor/rt-basic/cv.py:    supported_devices=["cpu", "mps", "cuda"],
regtest/metatensor/rt-basic/plumed.dat:      # DEVICE=cuda
regtest/metatensor/rt-basic/plumed.dat:      # DEVICE=cuda
regtest/metatensor/rt-basic/plumed.dat:      # DEVICE=cuda
regtest/metatensor/rt-basic/plumed.dat:      # DEVICE=cuda
regtest/isdb/rt-saxs-gpu/plumed.dat:GPU
configure:enable_af_cuda
configure:  --enable-af_cuda        enable search for arrayfire_cuda, default: no
configure:af_cuda=
configure:# Check whether --enable-af_cuda was given.
configure:if test "${enable_af_cuda+set}" = set; then :
configure:  enableval=$enable_af_cuda; case "${enableval}" in
configure:             (yes) af_cuda=true ;;
configure:             (no)  af_cuda=false ;;
configure:             (*)   as_fn_error $? "wrong argument to --enable-af_cuda" "$LINENO" 5 ;;
configure:             (yes) af_cuda=true ;;
configure:             (no)  af_cuda=false ;;
configure:for ac_lib in '' afopencl; do
configure:for ac_lib in '' afopencl; do
configure:if test "$af_cuda" = true ; then
configure:for ac_lib in '' afcuda; do
configure:    __PLUMED_HAS_ARRAYFIRE_CUDA=no
configure:for ac_lib in '' afcuda; do
configure:       $as_echo "#define __PLUMED_HAS_ARRAYFIRE_CUDA 1" >>confdefs.h
configure:       __PLUMED_HAS_ARRAYFIRE_CUDA=yes
configure:       { $as_echo "$as_me:${as_lineno-$LINENO}: WARNING: cannot enable __PLUMED_HAS_ARRAYFIRE_CUDA" >&5
configure:$as_echo "$as_me: WARNING: cannot enable __PLUMED_HAS_ARRAYFIRE_CUDA" >&2;}
configure:  # CUDA and CPU libtorch libs have different libraries
configure:  # first test CUDA program
configure:      testlibs=" torch_cpu c10 c10_cuda torch_cuda "
configure:    { $as_echo "$as_me:${as_lineno-$LINENO}: checking libtorch_cuda without extra libs" >&5
configure:$as_echo_n "checking libtorch_cuda without extra libs... " >&6; }
configure:    #include <torch/cuda.h>
configure:      std::cerr << "CUDA is available: " << torch::cuda::is_available() << std::endl;
configure:      device = torch::kCUDA;
configure:        { $as_echo "$as_me:${as_lineno-$LINENO}: checking libtorch_cuda with $all_LIBS" >&5
configure:$as_echo_n "checking libtorch_cuda with $all_LIBS... " >&6; }
configure:    #include <torch/cuda.h>
configure:      std::cerr << "CUDA is available: " << torch::cuda::is_available() << std::endl;
configure:      device = torch::kCUDA;
configure:          { $as_echo "$as_me:${as_lineno-$LINENO}: checking libtorch_cuda with -l$testlib" >&5
configure:$as_echo_n "checking libtorch_cuda with -l$testlib... " >&6; }
configure:    #include <torch/cuda.h>
configure:      std::cerr << "CUDA is available: " << torch::cuda::is_available() << std::endl;
configure:      device = torch::kCUDA;
configure:  # AC_MSG_NOTICE([CUDA-enabled libtorch not found (or devices not available), trying with CPU version.])
CHANGES/v2.5.md:  - \ref SAXS includes a GPU implementation based on ArrayFire (need to be linked at compile time) that can be activated with GPU
CHANGES/v2.5.md:- plumed can be compiled with ArrayFire to enable for gpu code. \ref SAXS collective variable is available as part of the isdb module to provide an example of a gpu implementation for a CV
CHANGES/v2.0.md:  using GPUs.
CHANGES/v2.10.md:  - A prototype of the \ref COORDINATION collective variable with limited functionalities is now included in a CUDA implementation which is orders of magnitude faster
CHANGES/v2.10.md:    that can be linked against the proper CUDA libraries and loaded at runtime with \ref LOAD. Documentation about how to install and use this feature can be found in directory
CHANGES/v2.10.md:    `plugins/cudaCoord`.
CHANGES/v2.8.md:- Updated GROMACS patches to warn about the joint use of update gpu and plumed: needs patch to be reapplied 
src/metatensor/metatensor.cpp:#include <torch/cuda.h>
src/metatensor/metatensor.cpp:        } else if (device == "cuda") {
src/metatensor/metatensor.cpp:            if (torch::cuda::is_available()) {
src/metatensor/metatensor.cpp:                available_devices.push_back(torch::Device("cuda"));
src/metatensor/metatensor.cpp:            } else if (device.is_cuda() && requested_device == "cuda") {
src/molfile/molfile_plugin.h:    * kernel-bypass direct I/O or using GPU-Direct Storage APIs,
src/core/PlumedMain.h:/// GpuDevice Identifier
src/core/PlumedMain.h:  int gpuDeviceId=-1;
src/core/PlumedMain.h:/// Get the value of the gpuDeviceId
src/core/PlumedMain.h:  int getGpuDeviceId() const ;
src/core/PlumedMain.h:int PlumedMain::getGpuDeviceId() const {
src/core/PlumedMain.h:  return gpuDeviceId;
src/core/PlumedMain.cpp:      case cmd_setGpuDeviceId:
src/core/PlumedMain.cpp:          if(id>=0) gpuDeviceId=id;
src/isdb/SAXS.cpp:#ifdef __PLUMED_HAS_ARRAYFIRE_CUDA
src/isdb/SAXS.cpp:#include <cuda_runtime.h>
src/isdb/SAXS.cpp:#include <af/cuda.h>
src/isdb/SAXS.cpp:#include <af/opencl.h>
src/isdb/SAXS.cpp:By default SAXS is calculated using Debye on CPU, by adding the GPU flag it is possible to solve the equation on
src/isdb/SAXS.cpp:a GPU if the ARRAYFIRE libraries are installed and correctly linked.
src/isdb/SAXS.cpp:By default SANS is calculated using Debye on CPU, by adding the GPU flag it is possible to solve the equation on a
src/isdb/SAXS.cpp:GPU if the ARRAYFIRE libraries are installed and correctly linked.
src/isdb/SAXS.cpp:  bool gpu;
src/isdb/SAXS.cpp:  void calculate_gpu(std::vector<Vector> &pos, std::vector<Vector> &deriv);
src/isdb/SAXS.cpp:  keys.add("compulsory","DEVICEID","-1","Identifier of the GPU to be used");
src/isdb/SAXS.cpp:  keys.addFlag("GPU",false,"Calculate SAXS using ARRAYFIRE on an accelerator device");
src/isdb/SAXS.cpp:  gpu(false),
src/isdb/SAXS.cpp:  parseFlag("GPU",gpu);
src/isdb/SAXS.cpp:  if(gpu) error("To use the GPU mode PLUMED must be compiled with ARRAYFIRE");
src/isdb/SAXS.cpp:  if(gpu&&comm.Get_rank()==0) {
src/isdb/SAXS.cpp:    if(deviceid==-1) deviceid=plumed.getGpuDeviceId();
src/isdb/SAXS.cpp:#ifdef  __PLUMED_HAS_ARRAYFIRE_CUDA
src/isdb/SAXS.cpp:  if(gpu && resolution) error("Resolution function is not supported in GPUs");
src/isdb/SAXS.cpp:  if(!gpu) {
src/isdb/SAXS.cpp:void SAXS::calculate_gpu(std::vector<Vector> &pos, std::vector<Vector> &deriv)
src/isdb/SAXS.cpp:  // on gpu only the master rank run the calculation
src/isdb/SAXS.cpp:#ifdef  __PLUMED_HAS_ARRAYFIRE_CUDA
src/isdb/SAXS.cpp:            if(!gpu) {
src/isdb/SAXS.cpp:        if(!gpu) {
src/isdb/SAXS.cpp:            if(!gpu) {
src/isdb/SAXS.cpp:        if(!gpu) {
src/isdb/SAXS.cpp:  if(gpu) calculate_gpu(beads_pos, bd_deriv);
src/isdb/EMMIVox.cpp:  bool gpu_;
src/isdb/EMMIVox.cpp:  torch::Tensor ovmd_gpu_;
src/isdb/EMMIVox.cpp:  torch::Tensor ovmd_der_gpu_;
src/isdb/EMMIVox.cpp:  torch::Tensor ismin_gpu_;
src/isdb/EMMIVox.cpp:  torch::Tensor ovdd_gpu_;
src/isdb/EMMIVox.cpp:  torch::Tensor Map_m_gpu_;
src/isdb/EMMIVox.cpp:  torch::Tensor pref_gpu_;
src/isdb/EMMIVox.cpp:  torch::Tensor invs2_gpu_;
src/isdb/EMMIVox.cpp:  torch::Tensor nl_id_gpu_;
src/isdb/EMMIVox.cpp:  torch::Tensor nl_im_gpu_;
src/isdb/EMMIVox.cpp:  torch::Tensor pref_nl_gpu_;
src/isdb/EMMIVox.cpp:  torch::Tensor invs2_nl_gpu_;
src/isdb/EMMIVox.cpp:  torch::Tensor Map_m_nl_gpu_;
src/isdb/EMMIVox.cpp:  void prepare_gpu();
src/isdb/EMMIVox.cpp:  void push_auxiliary_gpu();
src/isdb/EMMIVox.cpp:  void update_gpu();
src/isdb/EMMIVox.cpp:  keys.addFlag("GPU",false,"calculate EMMIVOX on GPU with Libtorch");
src/isdb/EMMIVox.cpp:  eps_(0.0001), mapstride_(0), gpu_(false)
src/isdb/EMMIVox.cpp:  // use GPU?
src/isdb/EMMIVox.cpp:  parseFlag("GPU",gpu_);
src/isdb/EMMIVox.cpp:  if (gpu_ && torch::cuda::is_available()) {
src/isdb/EMMIVox.cpp:    device_t_ = torch::kCUDA;
src/isdb/EMMIVox.cpp:    gpu_ = false;
src/isdb/EMMIVox.cpp:  if(gpu_) {log.printf("  running on GPU \n");}
src/isdb/EMMIVox.cpp:  // prepare gpu stuff: map centers, data, and error
src/isdb/EMMIVox.cpp:  prepare_gpu();
src/isdb/EMMIVox.cpp:void EMMIVOX::prepare_gpu()
src/isdb/EMMIVox.cpp:  ismin_gpu_ = torch::from_blob(ismin_.data(), {nd}, torch::kFloat64).to(torch::kFloat32).to(device_t_);
src/isdb/EMMIVox.cpp:  ovdd_gpu_  = torch::from_blob(ovdd_.data(),  {nd}, torch::kFloat64).to(torch::kFloat32).to(device_t_);
src/isdb/EMMIVox.cpp:  std::vector<double> Map_m_gpu(3*nd);
src/isdb/EMMIVox.cpp:    Map_m_gpu[i]      = Map_m_[i][0];
src/isdb/EMMIVox.cpp:    Map_m_gpu[i+nd]   = Map_m_[i][1];
src/isdb/EMMIVox.cpp:    Map_m_gpu[i+2*nd] = Map_m_[i][2];
src/isdb/EMMIVox.cpp:  Map_m_gpu_ = torch::from_blob(Map_m_gpu.data(), {3,nd}, torch::kFloat64).clone().to(torch::kFloat32).to(device_t_);
src/isdb/EMMIVox.cpp:  // push to GPU
src/isdb/EMMIVox.cpp:  push_auxiliary_gpu();
src/isdb/EMMIVox.cpp:void EMMIVOX::push_auxiliary_gpu()
src/isdb/EMMIVox.cpp:  // 2) initialize gpu tensors
src/isdb/EMMIVox.cpp:  pref_gpu_  = torch::from_blob(pref.data(),  {5,natoms}, torch::kFloat64).clone().to(torch::kFloat32).to(device_t_);
src/isdb/EMMIVox.cpp:  invs2_gpu_ = torch::from_blob(invs2.data(), {5,natoms}, torch::kFloat64).clone().to(torch::kFloat32).to(device_t_);
src/isdb/EMMIVox.cpp:// update auxiliary lists (to update pref_gpu_, invs2_gpu_, and cut_ on CPU/GPU)
src/isdb/EMMIVox.cpp:// update neighbor list (new cut_ + update pref_nl_gpu_ and invs2_nl_gpu_ on GPU)
src/isdb/EMMIVox.cpp:  // transfer data to gpu
src/isdb/EMMIVox.cpp:  update_gpu();
src/isdb/EMMIVox.cpp:void EMMIVOX::update_gpu()
src/isdb/EMMIVox.cpp:  nl_id_gpu_ = torch::from_blob(nl_id.data(), {nl_size}, torch::kInt32).clone().to(device_t_);
src/isdb/EMMIVox.cpp:  nl_im_gpu_ = torch::from_blob(nl_im.data(), {nl_size}, torch::kInt32).clone().to(device_t_);
src/isdb/EMMIVox.cpp:  // now we need to create pref_nl_gpu_ [5,nl_size]
src/isdb/EMMIVox.cpp:  pref_nl_gpu_  = torch::index_select(pref_gpu_,1,nl_im_gpu_);
src/isdb/EMMIVox.cpp:  // and invs2_nl_gpu_ [5,nl_size]
src/isdb/EMMIVox.cpp:  invs2_nl_gpu_ = torch::index_select(invs2_gpu_,1,nl_im_gpu_);
src/isdb/EMMIVox.cpp:  // and Map_m_nl_gpu_ [3,nl_size]
src/isdb/EMMIVox.cpp:  Map_m_nl_gpu_ = torch::index_select(Map_m_gpu_,1,nl_id_gpu_);
src/isdb/EMMIVox.cpp:// calculate forward model on gpu
src/isdb/EMMIVox.cpp:  // transfer positions to pos_gpu [3,natoms]
src/isdb/EMMIVox.cpp:  torch::Tensor pos_gpu = torch::from_blob(posg.data(), {3,natoms}, torch::kFloat64).to(torch::kFloat32).to(device_t_);
src/isdb/EMMIVox.cpp:  // create pos_nl_gpu_ [3,nl_size]
src/isdb/EMMIVox.cpp:  torch::Tensor pos_nl_gpu = torch::index_select(pos_gpu,1,nl_im_gpu_);
src/isdb/EMMIVox.cpp:  torch::Tensor md = Map_m_nl_gpu_ - pos_nl_gpu;
src/isdb/EMMIVox.cpp:  torch::Tensor ov = pref_nl_gpu_ * torch::exp(-0.5 * md2 * invs2_nl_gpu_);
src/isdb/EMMIVox.cpp:  ovmd_der_gpu_ = invs2_nl_gpu_ * ov;
src/isdb/EMMIVox.cpp:  ovmd_gpu_ = torch::zeros({nd}, options);
src/isdb/EMMIVox.cpp:  ovmd_gpu_.index_add_(0, nl_id_gpu_, ov);
src/isdb/EMMIVox.cpp:  ovmd_der_gpu_ = md * torch::sum(ovmd_der_gpu_,0);
src/isdb/EMMIVox.cpp:    // communicate ovmd_gpu_ to CPU [1, nd]
src/isdb/EMMIVox.cpp:    torch::Tensor ovmd_cpu = ovmd_gpu_.detach().to(torch::kCPU).to(torch::kFloat64);
src/isdb/EMMIVox.cpp:    ovmd_gpu_ = torch::from_blob(ovmd_.data(), {nd}, torch::kFloat64).to(torch::kFloat32).to(device_t_);
src/isdb/EMMIVox.cpp:    // communicate ovmd_gpu_ to CPU [1, nd]
src/isdb/EMMIVox.cpp:    torch::Tensor ovmd_cpu = ovmd_gpu_.detach().to(torch::kCPU).to(torch::kFloat64);
src/isdb/EMMIVox.cpp:  torch::Tensor dev = scale_ * ovmd_gpu_ + offset_ - ovdd_gpu_;
src/isdb/EMMIVox.cpp:  torch::Tensor errf = torch::erf( dev * inv_sqrt2_ * ismin_gpu_ );
src/isdb/EMMIVox.cpp:  torch::Tensor ene = 0.5 * ( errf / dev * zeros + torch::logical_not(zeros) * sqrt2_pi_ *  ismin_gpu_);
src/isdb/EMMIVox.cpp:  torch::Tensor d_der = -kbt_ * zeros * ( sqrt2_pi_ * torch::exp( -0.5 * dev * dev * ismin_gpu_ * ismin_gpu_ ) * ismin_gpu_ / errf - 1.0 / dev );
src/isdb/EMMIVox.cpp:  torch::Tensor der_gpu = torch::index_select(d_der,0,nl_id_gpu_);
src/isdb/EMMIVox.cpp:  // multiply by ovmd_der_gpu_ and scale [3, nl_size]
src/isdb/EMMIVox.cpp:  der_gpu = ovmd_der_gpu_ * scale_ * der_gpu;
src/isdb/EMMIVox.cpp:  torch::Tensor atoms_der_gpu = torch::zeros({3,natoms}, options);
src/isdb/EMMIVox.cpp:  atoms_der_gpu.index_add_(1, nl_im_gpu_, der_gpu);
src/isdb/EMMIVox.cpp:  torch::Tensor atom_der_cpu = atoms_der_gpu.detach().to(torch::kCPU).to(torch::kFloat64);

```
