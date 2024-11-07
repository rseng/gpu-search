# https://github.com/FEniCS/basix

```console
cpp/basix/mdspan.hpp:#ifndef _MDSPAN_HAS_CUDA
cpp/basix/mdspan.hpp:#  if defined(__CUDACC__)
cpp/basix/mdspan.hpp:#    define _MDSPAN_HAS_CUDA __CUDACC__
cpp/basix/mdspan.hpp:#  if (!defined(__NVCC__) || (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ * 10 >= 1170)) && \
cpp/basix/mdspan.hpp:#if defined(_MDSPAN_HAS_CUDA) || defined(_MDSPAN_HAS_HIP) || defined(_MDSPAN_HAS_SYCL)
cpp/basix/mdspan.hpp:#  if defined(_MDSPAN_HAS_CUDA) || defined(_MDSPAN_HAS_HIP)
cpp/basix/mdspan.hpp:// In CUDA defaulted functions do not need host device markup
cpp/basix/mdspan.hpp:#if defined(_MDSPAN_HAS_CUDA) || defined(_MDSPAN_HAS_HIP)
cpp/basix/mdspan.hpp:  /* Might need this on NVIDIA?
cpp/basix/mdspan.hpp:    #if !defined(_MDSPAN_HAS_HIP) && !defined(_MDSPAN_HAS_CUDA)
cpp/basix/mdspan.hpp:// Depending on the CUDA and GCC version we need both the builtin
cpp/basix/mdspan.hpp:      #ifdef __CUDA_ARCH__
cpp/basix/mdspan.hpp:#if defined(__NVCC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
cpp/basix/mdspan.hpp:  // Even with CUDA_ARCH protection this thing warns about calling host function
cpp/basix/mdspan.hpp:      #ifdef __CUDA_ARCH__
cpp/basix/mdspan.hpp:// Depending on the CUDA and GCC version we need both the builtin
cpp/basix/mdspan.hpp:#ifdef __CUDA_ARCH__
cpp/basix/mdspan.hpp:#if defined(__NVCC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
cpp/basix/mdspan.hpp:// the issue But Clang-CUDA also doesn't accept the use of deduction guide so
cpp/basix/mdspan.hpp:// disable it for CUDA altogether
cpp/basix/mdspan.hpp:#if defined(_MDSPAN_HAS_HIP) || defined(_MDSPAN_HAS_CUDA)
cpp/basix/mdspan.hpp:#if defined(__NVCC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
cpp/basix/mdspan.hpp:      // Compilers: CUDA 11.2 with GCC 9.1
cpp/basix/mdspan.hpp:// the issue But Clang-CUDA also doesn't accept the use of deduction guide so
cpp/basix/mdspan.hpp:// disable it for CUDA alltogether
cpp/basix/mdspan.hpp:#if defined(_MDSPAN_HAS_HIP) || defined(_MDSPAN_HAS_CUDA)
cpp/basix/mdspan.hpp:#if defined(__NVCC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
cpp/basix/mdspan.hpp:#if defined(__NVCC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
cpp/basix/mdspan.hpp:// the issue But Clang-CUDA also doesn't accept the use of deduction guide so
cpp/basix/mdspan.hpp:// disable it for CUDA altogether
cpp/basix/mdspan.hpp:#if defined(_MDSPAN_HAS_HIP) || defined(_MDSPAN_HAS_CUDA)
cpp/basix/mdspan.hpp:#if defined(__NVCC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
cpp/basix/mdspan.hpp:// the issue But Clang-CUDA also doesn't accept the use of deduction guide so
cpp/basix/mdspan.hpp:// disable it for CUDA alltogether
cpp/basix/mdspan.hpp:#if defined(_MDSPAN_HAS_HIP) || defined(_MDSPAN_HAS_CUDA)
cpp/basix/mdspan.hpp:#if defined(__NVCC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
cpp/basix/mdspan.hpp:     (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ * 10) < 1120)
cpp/basix/mdspan.hpp:#ifdef __CUDA_ARCH__

```
