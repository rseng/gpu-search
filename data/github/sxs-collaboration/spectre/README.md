# https://github.com/sxs-collaboration/spectre

```console
docs/DevGuide/PerformanceGuidelines.md:- Reduce memory allocations. On all modern hardware (many core CPUs, GPUs, and
docs/Installation/BuildSystem.md:Kokkos will use to configure itself. For example, to enable CUDA support you
docs/Installation/BuildSystem.md:must pass `-D Kokkos_ENABLE_CUDA=ON`. See the
docs/Installation/BuildSystem.md:## Nvidia Compiler
docs/Installation/BuildSystem.md:If you are using CUDA to compile for Nvidia GPUs but do not have the target GPU
docs/Installation/BuildSystem.md:on the system you are compiling on then you must also tell CMake what CUDA
docs/Installation/BuildSystem.md:architecture to use. You can do this by passing `-D CMAKE_CUDA_ARCHITECTURES=80`
docs/Installation/BuildSystem.md:to CMake. You must choose the architecture that will be compatible with the GPU
docs/Installation/BuildSystem.md:version by Nvidia and can be viewed
docs/Installation/BuildSystem.md:[here](https://developer.nvidia.com/cuda-gpus).
external/brigand/include/brigand/brigand.hpp:#if defined(__CUDACC__)
external/brigand/include/brigand/brigand.hpp:#define BRIGAND_COMP_CUDA
external/brigand/include/brigand/brigand.hpp:#if defined(BRIGAND_COMP_MSVC_2013) || defined(BRIGAND_COMP_CUDA) || defined(BRIGAND_COMP_INTEL) || (defined(_LIBCPP_VERSION) && __cplusplus < 201402L)
external/brigand/include/brigand/brigand.hpp:#if defined(BRIGAND_COMP_MSVC_2013) || defined(BRIGAND_COMP_CUDA) || defined(BRIGAND_COMP_INTEL) || (defined(_LIBCPP_VERSION) && __cplusplus < 201402L)
tests/Unit/ControlSystem/Systems/Test_Size.cpp:// The Nvidia compiler crashes if we define these lists inside the MockComponent
tools/CharmModulePatches/src/Parallel/Algorithms/AlgorithmArray_v7.0.0.def.h.patch:   CkRegisterMessagePupFn(epidx, (CkMessagePupFn)MessageType::ckDebugPup);
tools/CharmModulePatches/src/Parallel/Algorithms/AlgorithmArray_v7.0.1.def.h.patch:   CkRegisterMessagePupFn(epidx, (CkMessagePupFn)MessageType::ckDebugPup);
cmake/SetupKokkos.cmake:  if(Kokkos_ENABLE_CUDA)
cmake/SetupKokkos.cmake:    set(CMAKE_CUDA_STANDARD 20)
cmake/SetupKokkos.cmake:    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
cmake/SetupKokkos.cmake:    enable_language(CUDA)
cmake/SetupKokkos.cmake:    find_package(CUDAToolkit REQUIRED)
cmake/SetupKokkos.cmake:    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL
cmake/SetupKokkos.cmake:      "Enable lambda expressions in CUDA")
cmake/SetupKokkos.cmake:      AND Kokkos_ENABLE_CUDA
cmake/SetupKokkos.cmake:      -Xcudafe;"--diag_suppress=186,191,554,1301,1305,2189,3060">
cmake/SetupKokkos.cmake:      AND Kokkos_ENABLE_CUDA
cmake/SetupKokkos.cmake:    if (${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}
cmake/SetupKokkos.cmake:      message(STATUS "CUDA 12.1 is only partially supported by Clang")
cmake/SetupKokkos.cmake:        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-cuda-version>
src/Utilities/BlazeExceptions.hpp:#ifdef __CUDA_ARCH__
src/Utilities/BlazeExceptions.hpp:// When building for Nvidia GPUs we need to disable the use of vector
src/Utilities/ErrorHandling/Error.hpp:#ifdef __CUDA_ARCH__
src/Utilities/ErrorHandling/Error.hpp:#if defined(__clang__) && defined(__CUDA__)
src/Utilities/ErrorHandling/Error.hpp:#ifdef __CUDA_ARCH__
src/Utilities/ErrorHandling/Error.hpp:#if defined(__clang__) && defined(__CUDA__)
src/Utilities/ErrorHandling/Error.hpp:#ifdef __CUDA_ARCH__
src/Utilities/ErrorHandling/Error.hpp:#if defined(__clang__) && defined(__CUDA__)
src/Utilities/Simd/Simd.hpp:  // The nvcc compiler's built-in __sincos is for GPU code, not CPU code. In
src/Utilities/Simd/Simd.hpp:  // the case that we are running on a GPU (__CUDA_ARCH__ is defined) or we
src/Utilities/Simd/Simd.hpp:#if (defined(__CUDACC__) && defined(__CUDA_ARCH__)) or (not defined(__CUDACC__))
src/Utilities/Simd/Simd.hpp:  // The nvcc compiler's built-in __sincos is for GPU code, not CPU code. In
src/Utilities/Simd/Simd.hpp:  // the case that we are running on a GPU (__CUDA_ARCH__ is defined) or we
src/Utilities/Simd/Simd.hpp:#if (defined(__CUDACC__) && defined(__CUDA_ARCH__)) or (not defined(__CUDACC__))
src/DataStructures/Variables.hpp:#ifdef __CUDACC__
src/DataStructures/Variables.hpp:#ifndef __CUDACC__
src/DataStructures/Variables.hpp:#ifdef __CUDACC__
src/DataStructures/Variables.hpp:#ifndef __CUDACC__
src/Parallel/ArrayCollection/DgElementArrayMember.hpp: * nodegroup, but that is mostly of interest when using GPUs.
src/IO/H5/Cce.cpp:#ifdef __CUDACC__

```
