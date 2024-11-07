# https://github.com/BioMedIA/MIRTK

```console
Modules/Common/include/mirtk/Config.h:// CUDA
Modules/Common/include/mirtk/Config.h:#  if __CUDACC__
Modules/Common/include/mirtk/Config.h:#  if __CUDACC__
Modules/Common/include/mirtk/Config.h:#  if __CUDACC__
Modules/Common/include/mirtk/Parallel.h:/// Enable/disable GPU acceleration
Modules/Common/include/mirtk/Parallel.h:MIRTK_Common_EXPORT extern bool use_gpu;
Modules/Common/include/mirtk/Parallel.h:/// Debugging level of GPU code
Modules/Common/include/mirtk/Parallel.h:MIRTK_Common_EXPORT extern int debug_gpu;
Modules/Common/include/mirtk/Profiling.h:// GPU Profiling
Modules/Common/include/mirtk/Profiling.h:///   // launch CUDA kernel here
Modules/Common/include/mirtk/Profiling.h:           cudaEvent_t e_start, e_stop;                                        \
Modules/Common/include/mirtk/Profiling.h:           CudaSafeCall( cudaEventCreate(&e_start) );                          \
Modules/Common/include/mirtk/Profiling.h:           CudaSafeCall( cudaEventCreate(&e_stop) );                           \
Modules/Common/include/mirtk/Profiling.h:           CudaSafeCall( cudaEventRecord(e_start, 0) )
Modules/Common/include/mirtk/Profiling.h:#  define MIRTKCU_RESET_TIMING()   CudaSafeCall( cudaEventRecord(e_start, 0) )
Modules/Common/include/mirtk/Profiling.h:///   // launch CUDA kernel here
Modules/Common/include/mirtk/Profiling.h:///   CudaSafeCall( cudaDeviceSynchronize() );
Modules/Common/include/mirtk/Profiling.h:       CudaSafeCall( cudaEventRecord(e_stop, 0) );                             \
Modules/Common/include/mirtk/Profiling.h:       CudaSafeCall( cudaEventSynchronize(e_stop) );                           \
Modules/Common/include/mirtk/Profiling.h:       CudaSafeCall( cudaEventElapsedTime(&t_elapsed, e_start, e_stop) );      \
Modules/Common/include/mirtk/Profiling.h:///   // launch CUDA kernel here
Modules/Common/include/mirtk/Profiling.h:       CudaSafeCall( cudaEventRecord(e_stop, 0) );                             \
Modules/Common/include/mirtk/Profiling.h:       CudaSafeCall( cudaEventSynchronize(e_stop) );                           \
Modules/Common/include/mirtk/Profiling.h:       CudaSafeCall( cudaEventElapsedTime(&t_elapsed, e_start, e_stop) );      \
Modules/Common/include/mirtk/Profiling.h:       CudaSafeCall( cudaEventDestroy(e_start) );                              \
Modules/Common/include/mirtk/Profiling.h:       CudaSafeCall( cudaEventDestroy(e_stop) );                               \
Modules/Common/include/mirtk/Profiling.h:       oss << section << " [GPU]";                                             \
Modules/Common/include/mirtk/Profiling.h:///   // launch CUDA kernel here
Modules/Common/include/mirtk/Profiling.h:///   // launch CUDA kernel here
Modules/Common/include/mirtk/Profiling.h:       CudaSafeCall( cudaEventDestroy(e_start) );                              \
Modules/Common/include/mirtk/Profiling.h:       CudaSafeCall( cudaEventDestroy(e_stop) );                               \
Modules/Common/include/mirtk/CutilMath.h: * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
Modules/Common/include/mirtk/CutilMath.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
Modules/Common/include/mirtk/CutilMath.h:    (float3, float4 etc.) since these are not provided as standard by CUDA.
Modules/Common/include/mirtk/CutilMath.h:    This is part of the CUTIL library and is not supported by NVIDIA.
Modules/Common/include/mirtk/CutilMath.h:#if !defined(__CUDACC__)
Modules/Common/include/mirtk/CutilMath.h:#include "mirtk/CudaRuntime.h"
Modules/Common/include/mirtk/CutilMath.h:// host implementations of CUDA functions
Modules/Common/include/mirtk/CutilMath.h:#if !defined(__CUDACC__)
Modules/Common/include/mirtk/CutilMath.h:#endif // !defined(__CUDACC__)
Modules/Common/include/mirtk/CudaRuntime.h:#ifndef MIRTK_CudaRuntime_H
Modules/Common/include/mirtk/CudaRuntime.h:#define MIRTK_CudaRuntime_H
Modules/Common/include/mirtk/CudaRuntime.h:#ifdef HAVE_CUDA
Modules/Common/include/mirtk/CudaRuntime.h:#  include <cuda_runtime.h>
Modules/Common/include/mirtk/CudaRuntime.h:#endif // MIRTK_CudaRuntime_H
Modules/Common/include/mirtk/Cuda.h:#ifndef MIRTK_Cuda_H
Modules/Common/include/mirtk/Cuda.h:#define MIRTK_Cuda_H
Modules/Common/include/mirtk/Cuda.h:#ifdef HAVE_CUDA
Modules/Common/include/mirtk/Cuda.h:#  include <cuda.h>
Modules/Common/src/Profiling.cc:// Default: Use seconds for CPU time and milliseconds for GPU time
Modules/Common/src/Profiling.cc:#ifdef USE_CUDA
Modules/Common/src/Profiling.cc:  out << "  -profile-unit <msecs|secs>   Unit of time measurements. (default: secs [CPU], msecs [GPU])" << endl;
Modules/Common/src/Parallel.cc:// Default: Disable GPU acceleration
Modules/Common/src/Parallel.cc:bool use_gpu = false;
Modules/Common/src/Parallel.cc:// Default: No debugging of GPU code
Modules/Common/src/Parallel.cc:int debug_gpu = 0;
Modules/Common/src/Parallel.cc:  else if (strcmp(arg, "-gpu")     == 0) _option = "-gpu";
Modules/Common/src/Parallel.cc:    use_gpu = false;
Modules/Common/src/Parallel.cc:  } else if (OPTION("-gpu")) {
Modules/Common/src/Parallel.cc:#ifdef USE_CUDA
Modules/Common/src/Parallel.cc:    use_gpu = true;
Modules/Common/src/Parallel.cc:    cerr << "WARNING: Program compiled without GPU support using CUDA." << endl;
Modules/Common/src/Parallel.cc:    use_gpu = false;
Modules/Common/src/Parallel.cc:#if defined(HAVE_TBB) || defined(USE_CUDA)
Modules/Common/src/Parallel.cc:#ifdef USE_CUDA
Modules/Common/src/Parallel.cc:  out << "  -gpu                         Enable  GPU acceleration."   << (use_gpu ? " (default)" : "") << endl;
Modules/Common/src/Parallel.cc:  out << "  -cpu                         Disable GPU acceleration."   << (use_gpu ? "" : " (default)") << endl;
Modules/Common/src/Parallel.cc:#endif // HAVE_TBB || USE_CUDA
Modules/Common/src/CMakeLists.txt:  Cuda.h
Modules/Common/src/CMakeLists.txt:  CudaRuntime.h
Modules/Image/src/GenericImage.cc:  //            of host and device memory in CUGenericImage used by CUDA code.
Modules/Image/src/GenericImage.cc:  //            of host and device memory in CUGenericImage used by CUDA code.
Modules/Image/src/GenericImage.cc:  //            of host and device memory in CUGenericImage used by CUDA code.
Modules/Image/src/GenericImage.cc:  //            of host and device memory in CUGenericImage used by CUDA code.
Modules/Image/src/GenericImage.cc:  //            of host and device memory in CUGenericImage used by CUDA code.
Modules/Image/src/GenericImage.cc:  //            of host and device memory in CUGenericImage used by CUDA code.
Modules/Image/src/GenericImage.cc:  //            of host and device memory in CUGenericImage used by CUDA code.
Modules/Image/src/GenericImage.cc:  //            of host and device memory in CUGenericImage used by CUDA code.
Modules/Image/src/GenericImage.cc:  //            of host and device memory in CUGenericImage used by CUDA code.
Modules/Image/src/GenericImage.cc:  //            of host and device memory in CUGenericImage used by CUDA code.
Modules/Image/src/GenericImage.cc:  //            of host and device memory in CUGenericImage used by CUDA code.
Modules/Image/src/GenericImage.cc:  //            of host and device memory in CUGenericImage used by CUDA code.
Modules/Numerics/include/mirtk/Matrix.h:// Conversion to CUDA vector types
Modules/IO/src/meta/metaImage.cxx:#include <csignal>    /* sigprocmask */
Documentation/CMakeLists.txt:    mirtkCuda.h
Documentation/CMakeLists.txt:    mirtkCudaRuntime.h

```
