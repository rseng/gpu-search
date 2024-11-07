# https://github.com/citiususc/veryfasttree

```console
libs/boost-core/include/boost/config/detail/select_compiler_config.hpp:#if defined __CUDACC__
libs/boost-core/include/boost/config/detail/select_compiler_config.hpp://  NVIDIA CUDA C++ compiler for GPU
libs/boost-core/include/boost/config/detail/suffix.hpp:// Set some default values GPU support
libs/boost-core/include/boost/config/detail/suffix.hpp:#  ifndef BOOST_GPU_ENABLED
libs/boost-core/include/boost/config/detail/suffix.hpp:#  define BOOST_GPU_ENABLED
libs/boost-core/include/boost/config/detail/suffix.hpp:#    if defined(__CUDACC__)
libs/boost-core/include/boost/config/compiler/gcc.hpp:#if !defined(__CUDACC__)
libs/boost-core/include/boost/config/compiler/gcc.hpp:// doesn't actually support __int128 as of CUDA_VERSION=7500
libs/boost-core/include/boost/config/compiler/gcc.hpp:#if defined(__CUDACC__)
libs/boost-core/include/boost/config/compiler/gcc.hpp:// Nevertheless, as of CUDA 7.5, using __float128 with the host
libs/boost-core/include/boost/config/compiler/nvcc.hpp://  NVIDIA CUDA C++ compiler setup
libs/boost-core/include/boost/config/compiler/nvcc.hpp:#  define BOOST_COMPILER "NVIDIA CUDA C++ Compiler"
libs/boost-core/include/boost/config/compiler/nvcc.hpp:#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && defined(__CUDACC_VER_BUILD__)
libs/boost-core/include/boost/config/compiler/nvcc.hpp:#  define BOOST_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 1000000 + __CUDACC_VER_MINOR__ * 10000 + __CUDACC_VER_BUILD__)
libs/boost-core/include/boost/config/compiler/nvcc.hpp:// We don't really know what the CUDA version is, but it's definitely before 7.5:
libs/boost-core/include/boost/config/compiler/nvcc.hpp:#  define BOOST_CUDA_VERSION 7000000
libs/boost-core/include/boost/config/compiler/nvcc.hpp:// NVIDIA Specific support
libs/boost-core/include/boost/config/compiler/nvcc.hpp:// BOOST_GPU_ENABLED : Flag a function or a method as being enabled on the host and device
libs/boost-core/include/boost/config/compiler/nvcc.hpp:#define BOOST_GPU_ENABLED __host__ __device__
libs/boost-core/include/boost/config/compiler/nvcc.hpp:// A bug in version 7.0 of CUDA prevents use of variadic templates in some occasions
libs/boost-core/include/boost/config/compiler/nvcc.hpp:#if BOOST_CUDA_VERSION < 7050000
libs/boost-core/include/boost/config/compiler/nvcc.hpp:#if (BOOST_CUDA_VERSION > 8000000) && (BOOST_CUDA_VERSION < 8010000)
libs/boost-core/include/boost/config/compiler/nvcc.hpp:// CUDA (8.0) has no constexpr support in msvc mode:
libs/boost-core/include/boost/config/compiler/nvcc.hpp:#if defined(_MSC_VER) && (BOOST_CUDA_VERSION < 9000000)
libs/boost-core/include/boost/config/compiler/nvcc.hpp:#ifdef __CUDACC__
libs/boost-core/include/boost/config/compiler/nvcc.hpp:#if (BOOST_CUDA_VERSION >= 8000000) && (BOOST_CUDA_VERSION < 8010000)
libs/boost-core/include/boost/config/compiler/intel.hpp:#if defined(__CUDACC__)
libs/boost-core/include/boost/config/compiler/pgi.hpp://  Copyright 2017, NVIDIA CORPORATION.
libs/boost-core/include/boost/config/compiler/clang.hpp:// doesn't actually support __int128 as of CUDA_VERSION=7500
libs/boost-core/include/boost/config/compiler/clang.hpp:#if defined(__CUDACC__)
libs/boost-core/include/boost/core/swap.hpp:  BOOST_GPU_ENABLED
libs/boost-core/include/boost/core/swap.hpp:  BOOST_GPU_ENABLED
libs/boost-core/include/boost/core/swap.hpp:  BOOST_GPU_ENABLED
libs/boost-core/include/boost/core/empty_value.hpp:#elif defined(BOOST_CLANG) && !defined(__CUDACC__)
main.cpp:    app.add_set_ignore_case("-ext", options.extension, {"AUTO", "NONE", "SSE", "SSE3", "AVX", "AVX2", "AVX512", "CUDA"},
main.cpp:                            "Available: AUTO(default), NONE, SSE, SSE3 , AVX, AVX2, AVX512 or CUDA")->type_name("name")
README.md:    - Nvidia CUDA GPU computing support (experimental)
README.md:* CUDA Toolkit (CUDA only)
README.md:    // enable/disable CUDA
README.md:    USE_CUDA:BOOL=OFF
README.md:    // change CUDA Architecture
README.md:    CUDA_ARCH:STRING=80
README.md:    - **CUDA**: (Experimental) Arithmetic operations are performed using NVIDIA CUDA.
CMakeLists.txt:option(USE_CUDA "enable/disable CUDA" OFF)
CMakeLists.txt:set(CUDA_ARCH "80" CACHE STRING "change CUDA Architecture")
CMakeLists.txt:        src/impl/VeryFastTreeDoubleCuda.cpp
CMakeLists.txt:        src/impl/VeryFastTreeFloatCuda.cpp
CMakeLists.txt:if (USE_CUDA)
CMakeLists.txt:    message("WARNING!! CUDA is only a experimental feature")
CMakeLists.txt:    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
CMakeLists.txt:    set(USE_CUDA 1)
CMakeLists.txt:    enable_language(CUDA)
CMakeLists.txt:    find_package(CUDAToolkit)
CMakeLists.txt:            src/operations/CudaOperations.h
CMakeLists.txt:            src/operations/CudaOperations.cu
CMakeLists.txt:    set(LIBRARIES ${LIBRARIES} CUDA::cublas)
CMakeLists.txt:    set(USE_CUDA 0)
CMakeLists.txt:    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W2 /O2 /EHsc /bigobj /DUSE_CUDA=${USE_CUDA}")
CMakeLists.txt:    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -DUSE_CUDA=${USE_CUDA}")
man/VeryFastTree.1:AVX2, AVX512 or CUDA
man/VeryFastTree.1:Available: AUTO(default), NONE, SSE, SSE3, AVX, AVX2, AVX512 or CUDA
src/operations/CudaOperations.cu:#include "CudaOperations.h"
src/operations/CudaOperations.cu:        cudaSetDevice(omp_get_thread_num() % nDevices);
src/operations/CudaOperations.cu:veryfasttree::CudaOperations<Precision>::CudaOperations(){
src/operations/CudaOperations.cu:    cudaGetDeviceCount(&this->nDevices);
src/operations/CudaOperations.cu:veryfasttree::CudaOperations<Precision>::vector_multiply(numeric_t f1[], numeric_t f2[], int64_t n, numeric_t fOut[]) {
src/operations/CudaOperations.cu:Precision veryfasttree::CudaOperations<Precision>::vector_multiply_sum(numeric_t f1[], numeric_t f2[], int64_t n) {
src/operations/CudaOperations.cu:Precision veryfasttree::CudaOperations<Precision>::vector_multiply3_sum(numeric_t f1[], numeric_t f2[], numeric_t f3[],
src/operations/CudaOperations.cu:veryfasttree::CudaOperations<Precision>::vector_dot_product_rot(numeric_t f1[], numeric_t f2[], numeric_t fBy[],
src/operations/CudaOperations.cu:void veryfasttree::CudaOperations<Precision>::vector_add(numeric_t fTot[], numeric_t fAdd[], int64_t n) {
src/operations/CudaOperations.cu:Precision veryfasttree::CudaOperations<Precision>::vector_sum(numeric_t f1[], int64_t n) {
src/operations/CudaOperations.cu:void veryfasttree::CudaOperations<Precision>::vector_multiply_by(numeric_t f[], numeric_t fBy, int64_t n,
src/operations/CudaOperations.cu:void veryfasttree::CudaOperations<Precision>::vector_add_mult(numeric_t fTot[], numeric_t fAdd[], numeric_t weight,
src/operations/CudaOperations.cu:void veryfasttree::CudaOperations<float>::
src/operations/CudaOperations.cu:void veryfasttree::CudaOperations<double>::
src/operations/CudaOperations.cu:void veryfasttree::CudaOperations<float>::fastexp(numeric_t fTot[], int64_t n, int lvl) {
src/operations/CudaOperations.cu:void veryfasttree::CudaOperations<double>::fastexp(numeric_t fTot[], int64_t n, int lvl) {
src/operations/CudaOperations.cu:class veryfasttree::CudaOperations<float>;
src/operations/CudaOperations.cu:template void veryfasttree::CudaOperations<float>::matrix_by_vector4(float mat[][4], float vec[], float out[]);
src/operations/CudaOperations.cu:template void veryfasttree::CudaOperations<float>::matrix_by_vector4(float mat[][20], float vec[], float out[]);
src/operations/CudaOperations.cu:class veryfasttree::CudaOperations<double>;
src/operations/CudaOperations.cu:template void veryfasttree::CudaOperations<double>::matrix_by_vector4(double mat[][4], double vec[], double out[]);
src/operations/CudaOperations.cu:template void veryfasttree::CudaOperations<double>::matrix_by_vector4(double mat[][20], double vec[], double out[]);
src/operations/CudaOperations.cu:void veryfasttree::configCuda(Options& options){
src/operations/CudaOperations.cu:    cudaGetDeviceCount(&nDevices);
src/operations/CudaOperations.cu:        throw std::runtime_error("No CUDA compatible device found");
src/operations/CudaOperations.h:#ifndef VERYFASTTREE_CUDAOPERATION_H
src/operations/CudaOperations.h:#define VERYFASTTREE_CUDAOPERATION_H
src/operations/CudaOperations.h:    class CudaOperations {
src/operations/CudaOperations.h:        CudaOperations();
src/operations/CudaOperations.h:    void configCuda(Options& options);
src/operations/CudaOperations.h:#endif //VERYFASTTREE_CUDAOPERATION_H
src/VeryFastTree.cpp:#if USE_CUDA
src/VeryFastTree.cpp:#include "operations/CudaOperations.h"
src/VeryFastTree.cpp:extern template class veryfasttree::VeyFastTreeImpl<float, veryfasttree::CudaOperations>;
src/VeryFastTree.cpp:extern template class veryfasttree::VeyFastTreeImpl<double, veryfasttree::CudaOperations>;
src/VeryFastTree.cpp:            #if USE_CUDA
src/VeryFastTree.cpp:    else if (options.extension == "CUDA") {
src/VeryFastTree.cpp:        configCuda(options);
src/VeryFastTree.cpp:            VeyFastTreeImpl<double, CudaOperations>(options, in, out, log).run();
src/VeryFastTree.cpp:            VeyFastTreeImpl<float, CudaOperations>(options, in, out, log).run();
src/impl/VeryFastTreeFloatCuda.cpp:#if USE_CUDA
src/impl/VeryFastTreeFloatCuda.cpp:#include "../operations/CudaOperations.h"
src/impl/VeryFastTreeFloatCuda.cpp:template class veryfasttree::VeyFastTreeImpl<float, veryfasttree::CudaOperations>;
src/impl/VeryFastTreeDoubleCuda.cpp:#if USE_CUDA
src/impl/VeryFastTreeDoubleCuda.cpp:#include "../operations/CudaOperations.h"
src/impl/VeryFastTreeDoubleCuda.cpp:template class veryfasttree::VeyFastTreeImpl<double, veryfasttree::CudaOperations>;
src/Constants.h:                #if USE_CUDA
src/Constants.h:                ", CUDA"

```
