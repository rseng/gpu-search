# https://github.com/dib-lab/TheGreatGenotyper

```console
external-libraries/eigen/test/gpu_common.h:#ifndef EIGEN_TEST_GPU_COMMON_H
external-libraries/eigen/test/gpu_common.h:#define EIGEN_TEST_GPU_COMMON_H
external-libraries/eigen/test/gpu_common.h:  #include <cuda.h>
external-libraries/eigen/test/gpu_common.h:  #include <cuda_runtime.h>
external-libraries/eigen/test/gpu_common.h:  #include <cuda_runtime_api.h>
external-libraries/eigen/test/gpu_common.h:#define EIGEN_USE_GPU
external-libraries/eigen/test/gpu_common.h:#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
external-libraries/eigen/test/gpu_common.h:#if !defined(__CUDACC__) && !defined(__HIPCC__)
external-libraries/eigen/test/gpu_common.h:void run_on_gpu_meta_kernel(const Kernel ker, int n, const Input* in, Output* out)
external-libraries/eigen/test/gpu_common.h:void run_on_gpu(const Kernel& ker, int n, const Input& in, Output& out)
external-libraries/eigen/test/gpu_common.h:  gpuMalloc((void**)(&d_in),  in_bytes);
external-libraries/eigen/test/gpu_common.h:  gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/test/gpu_common.h:  gpuMemcpy(d_in,  in.data(),  in_bytes,  gpuMemcpyHostToDevice);
external-libraries/eigen/test/gpu_common.h:  gpuMemcpy(d_out, out.data(), out_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/test/gpu_common.h:  gpuDeviceSynchronize();
external-libraries/eigen/test/gpu_common.h:  hipLaunchKernelGGL(HIP_KERNEL_NAME(run_on_gpu_meta_kernel<Kernel,
external-libraries/eigen/test/gpu_common.h:  run_on_gpu_meta_kernel<<<Grids,Blocks>>>(ker, n, d_in, d_out);
external-libraries/eigen/test/gpu_common.h:  gpuDeviceSynchronize();
external-libraries/eigen/test/gpu_common.h:  gpuMemcpy(const_cast<typename Input::Scalar*>(in.data()),  d_in,  in_bytes,  gpuMemcpyDeviceToHost);
external-libraries/eigen/test/gpu_common.h:  gpuMemcpy(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost);
external-libraries/eigen/test/gpu_common.h:  gpuFree(d_in);
external-libraries/eigen/test/gpu_common.h:  gpuFree(d_out);
external-libraries/eigen/test/gpu_common.h:void run_and_compare_to_gpu(const Kernel& ker, int n, const Input& in, Output& out)
external-libraries/eigen/test/gpu_common.h:  Input  in_ref,  in_gpu;
external-libraries/eigen/test/gpu_common.h:  Output out_ref, out_gpu;
external-libraries/eigen/test/gpu_common.h:  #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
external-libraries/eigen/test/gpu_common.h:  in_ref = in_gpu = in;
external-libraries/eigen/test/gpu_common.h:  out_ref = out_gpu = out;
external-libraries/eigen/test/gpu_common.h:  run_on_gpu(ker, n, in_gpu, out_gpu);
external-libraries/eigen/test/gpu_common.h:  #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
external-libraries/eigen/test/gpu_common.h:  VERIFY_IS_APPROX(in_ref, in_gpu);
external-libraries/eigen/test/gpu_common.h:  VERIFY_IS_APPROX(out_ref, out_gpu);
external-libraries/eigen/test/gpu_common.h:    #if defined(__CUDA_ARCH__)
external-libraries/eigen/test/gpu_common.h:    info[0] = int(__CUDA_ARCH__ +0);
external-libraries/eigen/test/gpu_common.h:void ei_test_init_gpu()
external-libraries/eigen/test/gpu_common.h:  gpuDeviceProp_t deviceProp;
external-libraries/eigen/test/gpu_common.h:  gpuGetDeviceProperties(&deviceProp, device);
external-libraries/eigen/test/gpu_common.h:  run_on_gpu(compile_time_device_info(),10,dummy,info);
external-libraries/eigen/test/gpu_common.h:  std::cout << "GPU compile-time info:\n";
external-libraries/eigen/test/gpu_common.h:  #ifdef EIGEN_CUDACC
external-libraries/eigen/test/gpu_common.h:  std::cout << "  EIGEN_CUDACC:                 " << int(EIGEN_CUDACC) << "\n";
external-libraries/eigen/test/gpu_common.h:  #ifdef EIGEN_CUDACC_VER
external-libraries/eigen/test/gpu_common.h:  std::cout << "  EIGEN_CUDACC_VER:             " << int(EIGEN_CUDACC_VER) << "\n";
external-libraries/eigen/test/gpu_common.h:  std::cout << "  EIGEN_CUDA_ARCH:             " << info[0] << "\n";  
external-libraries/eigen/test/gpu_common.h:  std::cout << "GPU device info:\n";
external-libraries/eigen/test/gpu_common.h:#endif // EIGEN_TEST_GPU_COMMON_H
external-libraries/eigen/test/CMakeLists.txt:# CUDA unit tests
external-libraries/eigen/test/CMakeLists.txt:option(EIGEN_TEST_CUDA "Enable CUDA support in unit tests" OFF)
external-libraries/eigen/test/CMakeLists.txt:option(EIGEN_TEST_CUDA_CLANG "Use clang instead of nvcc to compile the CUDA tests" OFF)
external-libraries/eigen/test/CMakeLists.txt:if(EIGEN_TEST_CUDA_CLANG AND NOT CMAKE_CXX_COMPILER MATCHES "clang")
external-libraries/eigen/test/CMakeLists.txt:  message(WARNING "EIGEN_TEST_CUDA_CLANG is set, but CMAKE_CXX_COMPILER does not appear to be clang.")
external-libraries/eigen/test/CMakeLists.txt:if(EIGEN_TEST_CUDA)
external-libraries/eigen/test/CMakeLists.txt:find_package(CUDA 5.0)
external-libraries/eigen/test/CMakeLists.txt:if(CUDA_FOUND)
external-libraries/eigen/test/CMakeLists.txt:  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
external-libraries/eigen/test/CMakeLists.txt:    set(CUDA_NVCC_FLAGS "-ccbin ${CMAKE_C_COMPILER}" CACHE STRING "nvcc flags" FORCE)
external-libraries/eigen/test/CMakeLists.txt:  if(EIGEN_TEST_CUDA_CLANG)
external-libraries/eigen/test/CMakeLists.txt:   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 --cuda-gpu-arch=sm_30")
external-libraries/eigen/test/CMakeLists.txt:  ei_add_test(gpu_basic)
external-libraries/eigen/test/CMakeLists.txt:endif(CUDA_FOUND)
external-libraries/eigen/test/CMakeLists.txt:endif(EIGEN_TEST_CUDA)
external-libraries/eigen/test/CMakeLists.txt:  set(HIP_PATH "/opt/rocm/hip" CACHE STRING "Path to the HIP installation.")
external-libraries/eigen/test/CMakeLists.txt:	ei_add_test(gpu_basic)
external-libraries/eigen/test/gpu_basic.cu:// workaround issue between gcc >= 4.7 and cuda 5.5
external-libraries/eigen/test/gpu_basic.cu:#include "gpu_common.h"
external-libraries/eigen/test/gpu_basic.cu:EIGEN_DECLARE_TEST(gpu_basic)
external-libraries/eigen/test/gpu_basic.cu:  ei_test_init_gpu();
external-libraries/eigen/test/gpu_basic.cu:  #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(coeff_wise<Vector3f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(coeff_wise<Array44f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(replicate<Array4f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(replicate<Array33f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(redux<Array4f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(redux<Matrix3f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(prod_test<Matrix3f,Matrix3f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(prod_test<Matrix4f,Vector4f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(diagonal<Matrix3f,Vector3f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(diagonal<Matrix4f,Vector4f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(matrix_inverse<Matrix2f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(matrix_inverse<Matrix3f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(matrix_inverse<Matrix4f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(eigenvalues_direct<Matrix3f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(eigenvalues_direct<Matrix2f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  // These subtests compiles only with nvcc and fail with HIPCC and clang-cuda
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(eigenvalues<Matrix4f>(), nthreads, in, out) );
external-libraries/eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(eigenvalues<Matrix6f>(), nthreads, in, out) );
external-libraries/eigen/test/half_float.cpp:#include <Eigen/src/Core/arch/GPU/Half.h>
external-libraries/eigen/test/main.h:// Same for cuda_fp16.h
external-libraries/eigen/test/main.h:#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
external-libraries/eigen/test/main.h:#define EIGEN_TEST_CUDACC_VER  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
external-libraries/eigen/test/main.h:#elif defined(__CUDACC_VER__)
external-libraries/eigen/test/main.h:#define EIGEN_TEST_CUDACC_VER __CUDACC_VER__
external-libraries/eigen/test/main.h:#define EIGEN_TEST_CUDACC_VER 0
external-libraries/eigen/test/main.h:#if EIGEN_TEST_CUDACC_VER >= 70500
external-libraries/eigen/test/main.h:#include <cuda_fp16.h>
external-libraries/eigen/test/main.h:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && !defined(__SYCL_DEVICE_ONLY__)
external-libraries/eigen/test/main.h:  #elif !defined(__CUDACC__) && !defined(__HIPCC__) && !defined(__SYCL_DEVICE_ONLY__) // EIGEN_DEBUG_ASSERTS
external-libraries/eigen/test/main.h:  #if !defined(__CUDACC__) && !defined(__HIPCC__) && !defined(__SYCL_DEVICE_ONLY__)
external-libraries/eigen/Eigen/Core:// We need cuda_runtime.h/hip_runtime.h to ensure that
external-libraries/eigen/Eigen/Core:#if defined(EIGEN_CUDACC)
external-libraries/eigen/Eigen/Core:  #include <cuda_runtime.h>
external-libraries/eigen/Eigen/Core:#if defined(EIGEN_HAS_CUDA_FP16) || defined(EIGEN_HAS_HIP_FP16)
external-libraries/eigen/Eigen/Core:  #define EIGEN_HAS_GPU_FP16
external-libraries/eigen/Eigen/Core:#include "src/Core/arch/GPU/Half.h"
external-libraries/eigen/Eigen/Core:#include "src/Core/arch/GPU/PacketMathHalf.h"
external-libraries/eigen/Eigen/Core:#include "src/Core/arch/GPU/TypeCasting.h"
external-libraries/eigen/Eigen/Core:#if defined EIGEN_VECTORIZE_GPU
external-libraries/eigen/Eigen/Core:  #include "src/Core/arch/GPU/PacketMath.h"
external-libraries/eigen/Eigen/Core:  #include "src/Core/arch/GPU/MathFunctions.h"
external-libraries/eigen/Eigen/Core:// on CUDA devices
external-libraries/eigen/Eigen/Core:#ifdef EIGEN_CUDACC
external-libraries/eigen/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
external-libraries/eigen/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
external-libraries/eigen/Eigen/src/Core/util/Memory.h:#if ! defined EIGEN_ALLOCA && ! defined EIGEN_GPU_COMPILE_PHASE
external-libraries/eigen/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/Eigen/src/Core/util/Meta.h: #if defined(EIGEN_CUDA_ARCH)
external-libraries/eigen/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
external-libraries/eigen/Eigen/src/Core/util/Meta.h:    return CUDART_MAX_NORMAL_F;
external-libraries/eigen/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
external-libraries/eigen/Eigen/src/Core/util/Meta.h:    return CUDART_INF_F;
external-libraries/eigen/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
external-libraries/eigen/Eigen/src/Core/util/Meta.h:    return CUDART_NAN_F;
external-libraries/eigen/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
external-libraries/eigen/Eigen/src/Core/util/Meta.h:    return CUDART_INF;
external-libraries/eigen/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
external-libraries/eigen/Eigen/src/Core/util/Meta.h:    return CUDART_NAN;
external-libraries/eigen/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/Eigen/src/Core/util/Meta.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) || (!defined(EIGEN_CUDA_ARCH) && defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
external-libraries/eigen/Eigen/src/Core/util/Meta.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) || (!defined(EIGEN_CUDA_ARCH) && defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// Detect GPU compilers and architectures
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// Note that this also makes EIGEN_CUDACC and EIGEN_HIPCC mutually exclusive
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#if defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:  // Means the compiler is either nvcc or clang with CUDA enabled
external-libraries/eigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDACC __CUDACC__
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#if defined(__CUDA_ARCH__) && !defined(EIGEN_NO_CUDA)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDA_ARCH __CUDA_ARCH__
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// Starting with CUDA 9 the composite __CUDACC_VER__ is not available.
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDACC_VER  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#elif defined(__CUDACC_VER__)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDACC_VER __CUDACC_VER__
external-libraries/eigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDACC_VER 0
external-libraries/eigen/Eigen/src/Core/util/Macros.h:  // Means the compiler is HIPCC (analogous to EIGEN_CUDACC, but for HIP)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:    // analogous to EIGEN_CUDA_ARCH, but for HIP
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// Unify CUDA/HIPCC
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC) || defined(EIGEN_HIPCC)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// If either EIGEN_CUDACC or EIGEN_HIPCC is defined, then define EIGEN_GPUCC
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#define EIGEN_GPUCC
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// EIGEN_CUDACC implies the CUDA compiler and is used to tweak Eigen code for use in CUDA kernels
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// In most cases the same tweaks are required to the Eigen code to enable in both the HIP and CUDA kernels.
external-libraries/eigen/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDACC) || defined(EIGEN_HIPCC)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// For cases where the tweak is specific to CUDA, the code should be guarded with
external-libraries/eigen/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDACC)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIP_DEVICE_COMPILE)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// If either EIGEN_CUDA_ARCH or EIGEN_HIP_DEVICE_COMPILE is defined, then define EIGEN_GPU_COMPILE_PHASE
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#define EIGEN_GPU_COMPILE_PHASE
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// GPU compilers (HIPCC, NVCC) typically do two passes over the source code,
external-libraries/eigen/Eigen/src/Core/util/Macros.h://   + another to compile the source for the "device" (ie. GPU)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// EIGEN_CUDA_ARCH implies the device compilation phase in CUDA
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// In most cases, the "host" / "device" specific code is the same for both HIP and CUDA
external-libraries/eigen/Eigen/src/Core/util/Macros.h://       #if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/Eigen/src/Core/util/Macros.h://       #if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIP_DEVICE_COMPILE)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// For cases where the tweak is specific to CUDA, the code should be guarded with
external-libraries/eigen/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDA_ARCH)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:  && (!defined(__NVCC__) || !EIGEN_ARCH_ARM_OR_ARM64 || (EIGEN_CUDACC_VER >= 80000) )
external-libraries/eigen/Eigen/src/Core/util/Macros.h:  #if defined(EIGEN_CUDACC)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:    #if EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && (EIGEN_COMP_CLANG || EIGEN_CUDACC_VER >= 70500))
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC) && EIGEN_HAS_CONSTEXPR
external-libraries/eigen/Eigen/src/Core/util/Macros.h:    #ifdef __CUDACC_RELAXED_CONSTEXPR__
external-libraries/eigen/Eigen/src/Core/util/Macros.h:  #elif defined(__clang__) && defined(__CUDA__) && __has_feature(cxx_relaxed_constexpr)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// GPU stuff
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// Disable some features when compiling with GPU compilers (NVCC/clang-cuda/SYCL/HIPCC)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC) || defined(__SYCL_DEVICE_ONLY__) || defined(EIGEN_HIPCC)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// All functions callable from CUDA/HIP code must be qualified with __device__
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#ifdef EIGEN_GPUCC
external-libraries/eigen/Eigen/src/Core/util/Macros.h:// When compiling CUDA/HIP device code with NVCC or HIPCC
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#if (defined(EIGEN_CUDA_ARCH) && defined(__NVCC__)) || defined(EIGEN_HIP_DEVICE_COMPILE)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC < 1900 || EIGEN_CUDACC_VER>0)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:  // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(EIGEN_CUDA_ARCH) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL) && !defined(EIGEN_HIP_DEVICE_COMPILE)
external-libraries/eigen/Eigen/src/Core/util/Macros.h:#  if defined(EIGEN_CUDA_ARCH)
external-libraries/eigen/Eigen/src/Core/util/ConfigureVectorization.h:#if (defined EIGEN_CUDACC)
external-libraries/eigen/Eigen/src/Core/util/ConfigureVectorization.h:#if defined EIGEN_CUDACC
external-libraries/eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #define EIGEN_VECTORIZE_GPU
external-libraries/eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #if EIGEN_CUDACC_VER >= 70500
external-libraries/eigen/Eigen/src/Core/util/ConfigureVectorization.h:    #define EIGEN_HAS_CUDA_FP16
external-libraries/eigen/Eigen/src/Core/util/ConfigureVectorization.h:#if defined(EIGEN_HAS_CUDA_FP16)
external-libraries/eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #include <cuda_fp16.h>
external-libraries/eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #define EIGEN_VECTORIZE_GPU
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:  #if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:  #if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:  #if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if (!defined(EIGEN_GPUCC) || defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC)) 
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if (!defined(EIGEN_GPUCC)) && EIGEN_FAST_MATH && (!defined(__SYCL_DEVICE_ONLY__))
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef EIGEN_GPUCC
external-libraries/eigen/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
external-libraries/eigen/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
external-libraries/eigen/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
external-libraries/eigen/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
external-libraries/eigen/Eigen/src/Core/GeneralProduct.h:    #ifndef EIGEN_GPU_COMPILE_PHASE
external-libraries/eigen/Eigen/src/Core/GenericPacketMath.h:#elif defined(EIGEN_CUDA_ARCH)
external-libraries/eigen/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
external-libraries/eigen/Eigen/src/Core/GenericPacketMath.h:#if !defined(EIGEN_GPUCC)
external-libraries/eigen/Eigen/src/Core/ProductEvaluators.h:#ifndef EIGEN_GPUCC
external-libraries/eigen/Eigen/src/Core/ProductEvaluators.h:#ifndef EIGEN_GPUCC
external-libraries/eigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
external-libraries/eigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
external-libraries/eigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#ifndef EIGEN_PACKET_MATH_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_PACKET_MATH_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // EIGEN_PACKET_MATH_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:// type Eigen::half (inheriting either from CUDA's or HIP's __half struct) with
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:// to disk and the likes), but fast on GPUs.
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#ifndef EIGEN_HALF_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#define EIGEN_HALF_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if !defined(EIGEN_HAS_GPU_FP16)
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:// Make our own __half_raw definition that is similar to CUDA's.
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#elif defined(EIGEN_HAS_CUDA_FP16)
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h: #if defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER < 90000
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:// In CUDA < 9.0, __half is the equivalent of CUDA 9's __half_raw
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h: #endif // defined(EIGEN_HAS_CUDA_FP16)
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if defined(EIGEN_HAS_GPU_FP16)
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h: #elif defined(EIGEN_HAS_CUDA_FP16)
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:  #if (defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER >= 90000)
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if !defined(EIGEN_HAS_GPU_FP16)
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#elif defined(EIGEN_HAS_CUDA_FP16)
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:  // Note that EIGEN_CUDACC_VER is set to 0 even when compiling with HIP, so (EIGEN_CUDACC_VER < 90000) is true even for HIP!
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:  // So keeping this within #if defined(EIGEN_HAS_CUDA_FP16) is needed
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h: #if defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER < 90000  
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if defined(EIGEN_HAS_GPU_FP16)
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h: #elif defined(EIGEN_HAS_CUDA_FP16)
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:  #if defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER >= 90000
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER >= 90000
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER >= 90000
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:// Definitions for CPUs and older HIP+CUDA, mostly working through conversion
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && EIGEN_CUDACC_VER >= 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (EIGEN_CUDACC_VER >= 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:  #if (EIGEN_CUDACC_VER < 90000) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if (defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/Eigen/src/Core/arch/GPU/Half.h:#endif // EIGEN_HALF_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#ifndef EIGEN_PACKET_MATH_HALF_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#define EIGEN_PACKET_MATH_HALF_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDACC) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else  // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if EIGEN_CUDA_ARCH >= 350
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else  // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if EIGEN_CUDA_ARCH >= 350
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else  // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if EIGEN_CUDA_ARCH >= 530
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else  // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if EIGEN_CUDA_ARCH >= 530
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else  // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if EIGEN_CUDA_ARCH >= 530
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else  // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if EIGEN_CUDA_ARCH >= 530
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else  // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if EIGEN_CUDA_ARCH >= 530
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else  // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if EIGEN_CUDA_ARCH >= 530
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else  // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if EIGEN_CUDA_ARCH >= 530
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else  // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if EIGEN_CUDA_ARCH >= 530
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else  // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if EIGEN_CUDA_ARCH >= 530
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#else  // EIGEN_CUDA_ARCH
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if EIGEN_CUDA_ARCH >= 530
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#if (EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/PacketMathHalf.h:#endif // EIGEN_PACKET_MATH_HALF_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
external-libraries/eigen/Eigen/src/Core/arch/GPU/MathFunctions.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
external-libraries/eigen/Eigen/src/Core/arch/GPU/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/TypeCasting.h:#define EIGEN_TYPE_CASTING_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/GPU/TypeCasting.h:    #if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/TypeCasting.h:    #if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/TypeCasting.h:    #if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/TypeCasting.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
external-libraries/eigen/Eigen/src/Core/arch/GPU/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_GPU_H
external-libraries/eigen/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
external-libraries/eigen/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
external-libraries/eigen/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(EIGEN_CUDACC) && defined(EIGEN_USE_GPU)
external-libraries/eigen/Eigen/src/Core/arch/CUDA/Complex.h:// building for CUDA to avoid non-constexpr methods.
external-libraries/eigen/Eigen/src/Core/arch/CUDA/Complex.h:#endif // EIGEN_COMPLEX_CUDA_H
external-libraries/eigen/Eigen/src/Core/arch/SYCL/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
external-libraries/eigen/Eigen/src/Core/arch/SYCL/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_CUDA_H
external-libraries/eigen/Eigen/src/Core/arch/SYCL/InteropHeaders.h:// Make sure this is only available when targeting a GPU: we don't want to
external-libraries/eigen/Eigen/src/Core/arch/HIP/hcc/math_constants.h: *  HIP equivalent of the CUDA header of the same name
external-libraries/eigen/Eigen/src/SVD/BDCSVD.h:#if !defined(EIGEN_GPUCC)
external-libraries/eigen/doc/Manual.dox:  - \subpage TopicCUDA
external-libraries/eigen/doc/PreprocessorDirectives.dox: - \b \c EIGEN_NO_CUDA - disables CUDA support when defined. Might be useful in .cu files for which Eigen is used on the host only,
external-libraries/eigen/doc/UsingNVCC.dox:/** \page TopicCUDA Using Eigen in CUDA kernels
external-libraries/eigen/doc/UsingNVCC.dox:Staring from CUDA 5.5 and Eigen 3.3, it is possible to use Eigen's matrices, vectors, and arrays for fixed size within CUDA kernels. This is especially useful when working on numerous but small problems. By default, when Eigen's headers are included within a .cu file compiled by nvcc most Eigen's functions and methods are prefixed by the \c __device__ \c __host__ keywords making them callable from both host and device code.
external-libraries/eigen/doc/UsingNVCC.dox:This support can be disabled by defining \c EIGEN_NO_CUDA before including any Eigen's header.
external-libraries/eigen/doc/UsingNVCC.dox:    // workaround issue between gcc >= 4.7 and cuda 5.5
external-libraries/eigen/doc/UsingNVCC.dox: - On 64bits system Eigen uses \c long \c int as the default type for indexes and sizes. On CUDA device, it would make sense to default to 32 bits \c int.
external-libraries/eigen/doc/UsingNVCC.dox:   However, to keep host and CUDA code compatible, this cannot be done automatically by %Eigen, and the user is thus required to define \c EIGEN_DEFAULT_DENSE_INDEX_TYPE to \c int throughout his code (or only for CUDA code if there is no interaction between host and CUDA code through %Eigen's object).
external-libraries/eigen/CMakeLists.txt:set(EIGEN_CUDA_COMPUTE_ARCH 30 CACHE STRING "The CUDA compute architecture level to target when compiling CUDA code")
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  DataType* gpu_in_data = static_cast<DataType*>(sycl_device.allocate(tensor.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  DataType* gpu_out_data =static_cast<DataType*>(sycl_device.allocate(reversed_tensor.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> >  in_gpu(gpu_in_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> >  out_gpu(gpu_out_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, tensor.data(),(tensor.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.reverse(dim_rev);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyDeviceToHost(reversed_tensor.data(), gpu_out_data, reversed_tensor.dimensions().TotalSize()*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.reverse(dim_rev);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyDeviceToHost(reversed_tensor.data(), gpu_out_data, reversed_tensor.dimensions().TotalSize()*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.reverse(dim_rev);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyDeviceToHost(reversed_tensor.data(), gpu_out_data, reversed_tensor.dimensions().TotalSize()*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  DataType* gpu_in_data = static_cast<DataType*>(sycl_device.allocate(tensor.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  DataType* gpu_out_data_expected =static_cast<DataType*>(sycl_device.allocate(expected.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  DataType* gpu_out_data_result =static_cast<DataType*>(sycl_device.allocate(result.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> >  in_gpu(gpu_in_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> >  out_gpu_expected(gpu_out_data_expected, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> >  out_gpu_result(gpu_out_data_result, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, tensor.data(),(tensor.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:    out_gpu_expected.reverse(dim_rev).device(sycl_device) = in_gpu;
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:    out_gpu_expected.device(sycl_device) = in_gpu.reverse(dim_rev);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyDeviceToHost(expected.data(), gpu_out_data_expected, expected.dimensions().TotalSize()*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:        out_gpu_result.slice(dst_slice_start, dst_slice_dim).reverse(dim_rev).device(sycl_device) =
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:          in_gpu.slice(src_slice_start, src_slice_dim);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:      out_gpu_result.slice(dst_slice_start, dst_slice_dim).device(sycl_device) =
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:          in_gpu.slice(src_slice_start, src_slice_dim).reverse(dim_rev);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyDeviceToHost(result.data(), gpu_out_data_result, result.dimensions().TotalSize()*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_out_data_result, result.data(),(result.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:       out_gpu_result.slice(dst_slice_start, dst_slice_dim).reverse(dim_rev).device(sycl_device) =
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:           in_gpu.slice(dst_slice_start, dst_slice_dim);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:       out_gpu_result.slice(dst_slice_start, dst_slice_dim).device(sycl_device) =
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:           in_gpu.reverse(dim_rev).slice(dst_slice_start, dst_slice_dim);
external-libraries/eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyDeviceToHost(result.data(), gpu_out_data_result, result.dimensions().TotalSize()*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(left.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(right.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_in1(gpu_in1_data, leftRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_in2(gpu_in2_data, rightRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in1_data, left.data(),(left.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, right.data(),(right.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_out_data1 =  static_cast<DataType*>(sycl_device.allocate(concatenation1.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_out1(gpu_out_data1, concatenation1.dimensions());
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  gpu_out1.device(sycl_device) =gpu_in1.concatenate(gpu_in2, 0);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyDeviceToHost(concatenation1.data(), gpu_out_data1,(concatenation1.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_out_data1);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_out_data2 =  static_cast<DataType*>(sycl_device.allocate(concatenation2.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_out2(gpu_out_data2, concatenation2.dimensions());
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  gpu_out2.device(sycl_device) =gpu_in1.concatenate(gpu_in2, 1);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyDeviceToHost(concatenation2.data(), gpu_out_data2,(concatenation2.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_out_data2);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_out_data3 =  static_cast<DataType*>(sycl_device.allocate(concatenation3.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_out3(gpu_out_data3, concatenation3.dimensions());
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  gpu_out3.device(sycl_device) =gpu_in1.concatenate(gpu_in2, 2);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyDeviceToHost(concatenation3.data(), gpu_out_data3,(concatenation3.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_out_data3);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_in1_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_in2_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(left.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(right.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(result.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>> gpu_in1(gpu_in1_data, leftRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>> gpu_in2(gpu_in2_data, rightRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>> gpu_out(gpu_out_data, resRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in1_data, left.data(),(left.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, right.data(),(right.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_out_data, result.data(),(result.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp: gpu_in1.concatenate(gpu_in2, 0).device(sycl_device) =gpu_out;
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp: sycl_device.memcpyDeviceToHost(left.data(), gpu_in1_data,(left.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp: sycl_device.memcpyDeviceToHost(right.data(), gpu_in2_data,(right.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_in1_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_in2_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_tensor(d_tensor, tensor_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_no_stride(d_no_stride, tensor_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_stride(d_stride, stride_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  gpu_no_stride.device(sycl_device)=gpu_tensor.stride(strides);
external-libraries/eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  gpu_stride.device(sycl_device)=gpu_tensor.stride(strides);
external-libraries/eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_tensor(d_tensor, tensor_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_no_stride(d_no_stride, stride_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_stride(d_stride, stride_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  gpu_stride.stride(strides).device(sycl_device)=gpu_tensor;
external-libraries/eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  gpu_no_stride.stride(strides).device(sycl_device)=gpu_tensor.stride(no_strides);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:#define EIGEN_USE_GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Type* gpu_in_ptr = static_cast<Type*>(gpu_device.allocate(in_bytes));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Type* gpu_out_ptr = static_cast<Type*>(gpu_device.allocate(out_bytes));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  TensorMap<Tensor<Type, 2, DataLayout> > in_gpu(gpu_in_ptr, num_rows, num_cols);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  TensorMap<Tensor<Type, 0, DataLayout> > out_gpu(gpu_out_ptr);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  out_gpu.device(gpu_device) = in_gpu.sum();
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Tensor<Type, 0, DataLayout> full_redux_gpu;
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  // Check that the CPU and GPU reductions return the same result.
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_device.deallocate(gpu_in_ptr);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_device.deallocate(gpu_out_ptr);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::GpuDevice dev(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 3, DataLayout> > gpu_in(in_data, dim_x, dim_y, dim_z);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 2, DataLayout> > gpu_out(out_data, dim_y, dim_z);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_out.device(dev) = gpu_in.sum(red_axis);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_out.device(dev) += gpu_in.sum(red_axis);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Tensor<Type, 2, DataLayout> redux_gpu(dim_y, dim_z);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  dev.memcpyDeviceToHost(redux_gpu.data(), out_data, gpu_out.size()*sizeof(Type));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  // Check that the CPU and GPU reductions return the same result.
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  for (int i = 0; i < gpu_out.size(); ++i) {
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:    VERIFY_IS_APPROX(2*redux(i), redux_gpu(i));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::GpuDevice dev(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 3, DataLayout> > gpu_in(in_data, dim_x, dim_y, dim_z);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 2, DataLayout> > gpu_out(out_data, dim_x, dim_y);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_out.device(dev) = gpu_in.sum(red_axis);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_out.device(dev) += gpu_in.sum(red_axis);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Tensor<Type, 2, DataLayout> redux_gpu(dim_x, dim_y);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  dev.memcpyDeviceToHost(redux_gpu.data(), out_data, gpu_out.size()*sizeof(Type));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  // Check that the CPU and GPU reductions return the same result.
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  for (int i = 0; i < gpu_out.size(); ++i) {
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:    VERIFY_IS_APPROX(2*redux(i), redux_gpu(i));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_reduction_gpu) {
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  DataType* gpu_data_single_voxel_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  TensorMap<Tensor<DataType, 6, DataLayout,IndexType>> gpu_single_voxel_patch_col_major(gpu_data_single_voxel_patch_col_major, patchColMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  gpu_single_voxel_patch_col_major.device(sycl_device)=gpu_col_major.extract_volume_patches(1, 1, 1);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_voxel_patch_col_major.data(), gpu_data_single_voxel_patch_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  DataType* gpu_data_single_voxel_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  TensorMap<Tensor<DataType, 6, RowMajor,IndexType>> gpu_single_voxel_patch_row_major(gpu_data_single_voxel_patch_row_major, patchRowMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  gpu_single_voxel_patch_row_major.device(sycl_device)=gpu_row_major.extract_volume_patches(1, 1, 1);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_voxel_patch_row_major.data(), gpu_data_single_voxel_patch_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp: sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_voxel_patch_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_voxel_patch_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    TensorMap<Tensor<DataType, 5, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    TensorMap<Tensor<DataType, 5, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    DataType* gpu_data_entire_volume_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    TensorMap<Tensor<DataType, 6, DataLayout,IndexType>> gpu_entire_volume_patch_col_major(gpu_data_entire_volume_patch_col_major, patchColMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    gpu_entire_volume_patch_col_major.device(sycl_device)=gpu_col_major.extract_volume_patches(patch_z, patch_y, patch_x);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    sycl_device.memcpyDeviceToHost(entire_volume_patch_col_major.data(), gpu_data_entire_volume_patch_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  DataType* gpu_data_entire_volume_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  TensorMap<Tensor<DataType, 6, RowMajor,IndexType>> gpu_entire_volume_patch_row_major(gpu_data_entire_volume_patch_row_major, patchRowMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  gpu_entire_volume_patch_row_major.device(sycl_device)=gpu_row_major.extract_volume_patches(patch_z, patch_y, patch_x);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(entire_volume_patch_row_major.data(), gpu_data_entire_volume_patch_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_entire_volume_patch_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_entire_volume_patch_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_input(d_input, input_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout, IndexType> > gpu_kernel(d_kernel, kernel_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_result(d_result, result_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims3);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_input(d_input, input_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_kernel(d_kernel, kernel_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_result(d_result, result_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims3);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_input(d_input, input_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_kernel(d_kernel, kernel_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_result(d_result, result_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims3);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_input(d_input, input_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout, IndexType> > gpu_kernel(d_kernel, kernel_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_result(d_result, result_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims3);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout,IndexType> > gpu_input(d_input, input_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout,IndexType> > gpu_kernel(d_kernel, kernel_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout,IndexType> > gpu_result(d_result, result_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_input(d_input, input_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_kernel(d_kernel, kernel_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_valid(d_valid, valid.dimensions());
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_valid.device(sycl_device)=gpu_input.pad(padding).convolve(gpu_kernel, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_same(d_same, same.dimensions());
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_same.device(sycl_device)=gpu_input.pad(padding).convolve(gpu_kernel, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_full(d_full, full.dimensions());
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_full.device(sycl_device)=gpu_input.pad(padding).convolve(gpu_kernel, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_input(d_input, input_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_kernel(d_kernel, kernel_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_result(d_result, result.dimensions());
external-libraries/eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_result.device(sycl_device)=gpu_input.stride(stride_of_3).convolve(gpu_kernel, dims).stride(stride_of_2);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(in1.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(out1.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu1(gpu_data1, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu2(gpu_data2, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, in1.data(),(in1.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data2, in1.data(),(in1.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu1.device(sycl_device) = gpu1 * 3.14f;
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu2.device(sycl_device) = gpu2 * 2.7f;
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out1.data(), gpu_data1,(out1.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out2.data(), gpu_data1,(out2.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out3.data(), gpu_data2,(out3.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_data1);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_data2);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType* gpu_data  = static_cast<DataType*>(sycl_device.allocate(in1.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 1, DataLayout, IndexType>> gpu1(gpu_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data, in1.data(),(in1.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_data, out.size()*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(in1.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(in2.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType * gpu_in3_data  = static_cast<DataType*>(sycl_device.allocate(in3.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(out.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_in1(gpu_in1_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_in2(gpu_in2_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_in3(gpu_in3_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_out(gpu_out_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_in1.device(sycl_device) = gpu_in1.constant(1.2f);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(in1.data(), gpu_in1_data ,(in1.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 * 1.2f;
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data ,(out.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in2.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 * gpu_in2;
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 + gpu_in2;
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 * gpu_in1;
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) =  gpu_in1 * gpu_in1.constant(3.14f) + gpu_in2 * gpu_in2.constant(2.7f);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(),gpu_out_data,(out.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in3_data, in3.data(),(in3.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) =(gpu_in1 > gpu_in1.constant(0.5f)).select(gpu_in2, gpu_in3);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_in1_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_in2_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_in3_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:    Scalar1* gpu_in_data  = static_cast<Scalar1*>(sycl_device.allocate(in.size()*sizeof(Scalar1)));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:    Scalar2 * gpu_out_data =  static_cast<Scalar2*>(sycl_device.allocate(out.size()*sizeof(Scalar2)));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:    TensorMap<Tensor<Scalar1, 1, DataLayout, IndexType>> gpu_in(gpu_in_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:    TensorMap<Tensor<Scalar2, 1, DataLayout, IndexType>> gpu_out(gpu_out_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.size())*sizeof(Scalar1));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:    gpu_out.device(sycl_device) = gpu_in. template cast<Scalar2>();
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data, out.size()*sizeof(Scalar2));
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:    sycl_device.deallocate(gpu_in_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_sycl.cpp:    sycl_device.deallocate(gpu_out_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:#define EIGEN_USE_GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:#include <Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:void test_gpu_cumsum(int m_size, int k_size, int n_size)
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  Tensor<float, 3, DataLayout> t_result_gpu(m_size, k_size, n_size);
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpuMalloc((void**)(&d_t_input), t_input_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpuMalloc((void**)(&d_t_result), t_result_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpuMemcpy(d_t_input, t_input.data(), t_input_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:      gpu_t_input(d_t_input, Eigen::array<int, 3>(m_size, k_size, n_size));
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:      gpu_t_result(d_t_result, Eigen::array<int, 3>(m_size, k_size, n_size));
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpu_t_result.device(gpu_device) = gpu_t_input.cumsum(1);
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpuMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, gpuMemcpyDeviceToHost);
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:    if (fabs(t_result(i) - t_result_gpu(i)) < 1e-4f) {
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), 1e-4f)) {
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:              << " vs " <<  t_result_gpu(i) << std::endl;
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpuFree((void*)d_t_input);
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpuFree((void*)d_t_result);
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_scan_gpu)
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  CALL_SUBTEST_1(test_gpu_cumsum<ColMajor>(128, 128, 128));
external-libraries/eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  CALL_SUBTEST_2(test_gpu_cumsum<RowMajor>(128, 128, 128));
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:#define EIGEN_USE_GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:#include <Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:void test_gpu_random_uniform()
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  gpu_out.device(gpu_device) = gpu_out.random();
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:void test_gpu_random_normal()
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  gpu_out.device(gpu_device) = gpu_out.random(gen);
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_random_gpu)
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  CALL_SUBTEST(test_gpu_random_uniform());
external-libraries/eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  CALL_SUBTEST(test_gpu_random_normal());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:#define EIGEN_USE_GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_nullary() {
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in1), tensor_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in2), tensor_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in1, in1.data(), tensor_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in2, in2.data(), tensor_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_in1(
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_in2(
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_in1.device(gpu_device) = gpu_in1.constant(3.14f);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_in2.device(gpu_device) = gpu_in2.random();
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(new1.data(), d_in1, tensor_bytes, gpuMemcpyDeviceToHost,
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(new2.data(), d_in2, tensor_bytes, gpuMemcpyDeviceToHost,
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in1);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in2);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_elementwise_small() {
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in1), in1_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in2), in2_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in2, in2.data(), in2_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in2(
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_out(
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost,
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in1);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in2);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_elementwise()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in1), in1_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in2), in2_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in3), in3_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in2, in2.data(), in2_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in3, in3.data(), in3_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in3(d_in3, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2 * gpu_in3;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in1);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in2);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in3);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_props() {
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in1), in1_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_out(
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = (gpu_in1.isnan)();
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost,
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in1);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_reduction()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in1), in1_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_in1(d_in1, 72,53,97,113);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in1.maximum(reduction_axis);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in1);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_contraction()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  // a 15 SM GK110 GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_t_left), t_left_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_t_right), t_right_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_t_result), t_result_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_t_left, t_left.data(), t_left_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_t_right, t_right.data(), t_right_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_t_left(d_t_left, 6, 50, 3, 31);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_t_right(d_t_right, 3, 31, 7, 20, 1);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_t_result(d_t_result, 6, 50, 7, 20, 1);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(t_result.data(), d_t_result, t_result_bytes, gpuMemcpyDeviceToHost);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_t_left);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_t_right);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_t_result);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_convolution_1d()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_input), input_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_kernel), kernel_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_input(d_input, 74,37,11,137);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, DataLayout> > gpu_kernel(d_kernel, 4);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_out(d_out, 74,34,11,137);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_input);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_kernel);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_convolution_inner_dim_col_major_1d()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_input), input_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_kernel), kernel_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, ColMajor> > gpu_input(d_input,74,9,11,7);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, ColMajor> > gpu_kernel(d_kernel,4);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, ColMajor> > gpu_out(d_out,71,9,11,7);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_input);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_kernel);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_convolution_inner_dim_row_major_1d()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_input), input_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_kernel), kernel_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, RowMajor> > gpu_input(d_input, 7,9,11,74);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, RowMajor> > gpu_kernel(d_kernel, 4);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, RowMajor> > gpu_out(d_out, 7,9,11,71);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_input);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_kernel);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_convolution_2d()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_input), input_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_kernel), kernel_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_input(d_input,74,37,11,137);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2, DataLayout> > gpu_kernel(d_kernel,3,4);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_out(d_out,74,35,8,137);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_input);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_kernel);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_convolution_3d()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_input), input_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_kernel), kernel_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;    
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_input(d_input,74,37,11,137,17);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3, DataLayout> > gpu_kernel(d_kernel,3,4,2);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_out(d_out,74,35,8,136,17);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_input);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_kernel);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_lgamma(const Scalar stddev)
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in, in.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in.lgamma();
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_digamma()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in, in.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 7);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 7);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in.digamma();
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_zeta()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_x), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_q), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_x, in_x.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_q, in_q.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 6);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_q(d_in_q, 6);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 6);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in_x.zeta(gpu_in_q);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_x);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_q);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_polygamma()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_x), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_n), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_x, in_x.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_n, in_n.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 7);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_n(d_in_n, 7);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 7);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in_n.polygamma(gpu_in_x);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_x);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_n);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_igamma()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMalloc((void**)(&d_a), bytes) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMalloc((void**)(&d_x), bytes) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMalloc((void**)(&d_out), bytes) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_a, a.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_x, x.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_a(d_a, 6, 6);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_x(d_x, 6, 6);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 6, 6);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_a.igamma(gpu_x);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_a);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_x);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_igammac()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_a), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_x), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_a, a.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_x, x.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_a(d_a, 6, 6);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_x(d_x, 6, 6);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 6, 6);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_a.igammac(gpu_x);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_a);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_x);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_erf(const Scalar stddev)
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMalloc((void**)(&d_in), bytes) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMalloc((void**)(&d_out), bytes) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in, in.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in.erf();
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_erfc(const Scalar stddev)
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in, in.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in.erfc();
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_betainc()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_x), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_a), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_b), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_x, in_x.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_a, in_a.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_b, in_b.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 125);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_a(d_in_a, 125);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_b(d_in_b, 125);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 125);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = betainc(gpu_in_a, gpu_in_b, gpu_in_x);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_x);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_a);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_b);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_i0e()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in, in_x.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 21);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 21);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in.i0e();
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_i1e()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in, in_x.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 21);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 21);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in.i1e();
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_igamma_der_a()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_a), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_x), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_a, in_a.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_x, in_x.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_a(d_a, 30);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_x(d_x, 30);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 30);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_a.igamma_der_a(gpu_x);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_a);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_x);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_gamma_sample_der_alpha()
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_alpha), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_sample), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_alpha, in_alpha.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_sample, in_sample.data(), bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_alpha(d_alpha, 30);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_sample(d_sample, 30);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 30);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_alpha.gamma_sample_der_alpha(gpu_sample);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_alpha);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_sample);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_gpu)
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_1(test_gpu_nullary());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_1(test_gpu_elementwise_small());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_1(test_gpu_elementwise());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_1(test_gpu_props());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_1(test_gpu_reduction());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_2(test_gpu_contraction<ColMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_2(test_gpu_contraction<RowMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_1d<ColMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_1d<RowMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_inner_dim_col_major_1d());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_inner_dim_row_major_1d());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_2d<ColMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_2d<RowMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_3d<ColMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_3d<RowMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<float>(1.0f));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<float>(100.0f));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<float>(0.01f));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<float>(0.001f));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<double>(1.0));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<double>(100.0));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<double>(0.01));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<double>(0.001));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<float>(1.0f));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<float>(100.0f));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<float>(0.01f));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<float>(0.001f));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<float>(1.0f));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  // CALL_SUBTEST(test_gpu_erfc<float>(100.0f));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<float>(5.0f)); // GPU erfc lacks precision for large inputs
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<float>(0.01f));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<float>(0.001f));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<double>(1.0));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<double>(100.0));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<double>(0.01));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<double>(0.001));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<double>(1.0));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  // CALL_SUBTEST(test_gpu_erfc<double>(100.0));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<double>(5.0)); // GPU erfc lacks precision for large inputs
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<double>(0.01));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<double>(0.001));
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_digamma<float>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_digamma<double>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_polygamma<float>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_polygamma<double>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_zeta<float>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_zeta<double>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_igamma<float>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_igammac<float>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_igamma<double>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_igammac<double>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_betainc<float>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_betainc<double>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_i0e<float>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_i0e<double>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_i1e<float>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_i1e<double>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_i1e<float>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_i1e<double>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_igamma_der_a<float>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_igamma_der_a<double>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_gamma_sample_der_alpha<float>());
external-libraries/eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_gamma_sample_der_alpha<double>());
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(padded.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  TensorMap<Tensor<DataType, 4,DataLayout,IndexType>> gpu1(gpu_data1, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  TensorMap<Tensor<DataType, 4,DataLayout,IndexType>> gpu2(gpu_data2, padedtensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  gpu2.device(sycl_device)=gpu1.pad(paddings);
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.memcpyDeviceToHost(padded.data(), gpu_data2,(padded.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.deallocate(gpu_data1);
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.deallocate(gpu_data2);
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(result.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  TensorMap<Tensor<DataType, 4,DataLayout,IndexType>> gpu1(gpu_data1, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  TensorMap<Tensor<DataType, 2,DataLayout,IndexType>> gpu2(gpu_data2, reshape_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  gpu2.device(sycl_device)=gpu1.pad(paddings).reshape(reshape_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.memcpyDeviceToHost(result.data(), gpu_data2,(result.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.deallocate(gpu_data1);
external-libraries/eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.deallocate(gpu_data2);
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  DataType * gpu_in_data  = static_cast<DataType*>(sycl_device.allocate(input.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  DataType * gpu_out_data  = static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  TensorMap<TensorFixedSize<DataType, Sizes<2, 3, 5, 7>, DataLayout, IndexType>> gpu_in(gpu_in_data, in_range);
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>> gpu_out(gpu_out_data, out_range);
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, input.data(),(input.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in.broadcast(broadcasts);
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  DataType * gpu_in_data  = static_cast<DataType*>(sycl_device.allocate(input.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  DataType * gpu_out_data  = static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>>  gpu_in(gpu_in_data, in_range);
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>> gpu_out(gpu_out_data, out_range);
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, input.data(),(input.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in.broadcast(broadcasts);
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip1  = static_cast<DataType*>(sycl_device.allocate(chip1TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip1(gpu_data_chip1, chip1TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip1.device(sycl_device)=gpu_tensor.template chip<0l>(1l);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip1.data(), gpu_data_chip1, chip1TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip2  = static_cast<DataType*>(sycl_device.allocate(chip2TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip2(gpu_data_chip2, chip2TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip2.device(sycl_device)=gpu_tensor.template chip<1l>(1l);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip2.data(), gpu_data_chip2, chip2TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip3  = static_cast<DataType*>(sycl_device.allocate(chip3TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip3(gpu_data_chip3, chip3TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip3.device(sycl_device)=gpu_tensor.template chip<2l>(2l);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip3.data(), gpu_data_chip3, chip3TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip4  = static_cast<DataType*>(sycl_device.allocate(chip4TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip4(gpu_data_chip4, chip4TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip4.device(sycl_device)=gpu_tensor.template chip<3l>(5l);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip4.data(), gpu_data_chip4, chip4TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip5  = static_cast<DataType*>(sycl_device.allocate(chip5TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip5(gpu_data_chip5, chip5TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip5.device(sycl_device)=gpu_tensor.template chip<4l>(7l);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip5.data(), gpu_data_chip5, chip5TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip1);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip2);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip3);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip4);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip5);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip1  = static_cast<DataType*>(sycl_device.allocate(chip1TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip1(gpu_data_chip1, chip1TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip1.device(sycl_device)=gpu_tensor.chip(1l,0l);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip1.data(), gpu_data_chip1, chip1TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip2  = static_cast<DataType*>(sycl_device.allocate(chip2TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip2(gpu_data_chip2, chip2TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip2.device(sycl_device)=gpu_tensor.chip(1l,1l);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip2.data(), gpu_data_chip2, chip2TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip3  = static_cast<DataType*>(sycl_device.allocate(chip3TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip3(gpu_data_chip3, chip3TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip3.device(sycl_device)=gpu_tensor.chip(2l,2l);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip3.data(), gpu_data_chip3, chip3TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip4  = static_cast<DataType*>(sycl_device.allocate(chip4TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip4(gpu_data_chip4, chip4TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip4.device(sycl_device)=gpu_tensor.chip(5l,3l);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip4.data(), gpu_data_chip4, chip4TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip5  = static_cast<DataType*>(sycl_device.allocate(chip5TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip5(gpu_data_chip5, chip5TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip5.device(sycl_device)=gpu_tensor.chip(7l,4l);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip5.data(), gpu_data_chip5, chip5TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip1);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip2);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip3);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip4);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip5);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip1  = static_cast<DataType*>(sycl_device.allocate(chip1TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_tensor1  = static_cast<DataType*>(sycl_device.allocate(chip1TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip1(gpu_data_chip1, chip1TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_tensor1(gpu_data_tensor1, chip1TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor1, tensor1.data(), chip1TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip1.device(sycl_device)=gpu_tensor.template chip<0l>(0l) + gpu_tensor1;
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip1.data(), gpu_data_chip1, chip1TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_tensor2  = static_cast<DataType*>(sycl_device.allocate(chip2TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip2  = static_cast<DataType*>(sycl_device.allocate(chip2TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout,IndexType>> gpu_tensor2(gpu_data_tensor2, chip2TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout,IndexType>> gpu_chip2(gpu_data_chip2, chip2TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor2, tensor2.data(), chip2TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip2.device(sycl_device)=gpu_tensor.template chip<0l>(0l).template chip<1l>(2l) + gpu_tensor2;
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip2.data(), gpu_data_chip2, chip2TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor1);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip1);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor2);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip2);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input1  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input2  = static_cast<DataType*>(sycl_device.allocate(input2TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_input1(gpu_data_input1, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input2(gpu_data_input2, input2TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input1, input1.data(), tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.device(sycl_device)=gpu_input1;
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input2, input2.data(), input2TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.template chip<0l>(1l).device(sycl_device)=gpu_input2;
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.device(sycl_device)=gpu_input1;
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input3  = static_cast<DataType*>(sycl_device.allocate(input3TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input3(gpu_data_input3, input3TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input3, input3.data(), input3TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.template chip<1l>(1l).device(sycl_device)=gpu_input3;
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.device(sycl_device)=gpu_input1;
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input4  = static_cast<DataType*>(sycl_device.allocate(input4TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input4(gpu_data_input4, input4TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input4, input4.data(), input4TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.template chip<2l>(3l).device(sycl_device)=gpu_input4;
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.device(sycl_device)=gpu_input1;
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input5  = static_cast<DataType*>(sycl_device.allocate(input5TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input5(gpu_data_input5, input5TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input5, input5.data(), input5TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.template chip<3l>(4l).device(sycl_device)=gpu_input5;
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.device(sycl_device)=gpu_input1;
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input6  = static_cast<DataType*>(sycl_device.allocate(input6TensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input6(gpu_data_input6, input6TensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input6, input6.data(), input6TensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.template chip<4l>(5l).device(sycl_device)=gpu_input6;
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.device(sycl_device)=gpu_input1;
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input7  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_input7(gpu_data_input7, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input7, input7.data(), tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.chip(0l,0l).device(sycl_device)=gpu_input7.chip(0l,0l);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input1);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input2);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input3);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input4);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input5);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input6);
external-libraries/eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input7);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:#define EIGEN_USE_GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:// Context for evaluation on GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:struct GPUContext {
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  GPUContext(const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in1, Eigen::TensorMap<Eigen::Tensor<float, 3> >& in2, Eigen::TensorMap<Eigen::Tensor<float, 3> >& out) : in1_(in1), in2_(in2), out_(out), gpu_device_(&stream_) {
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuMalloc((void**)(&kernel_1d_), 2*sizeof(float)) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuMemcpy(kernel_1d_, kernel_1d_val, 2*sizeof(float), gpuMemcpyHostToDevice) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuMalloc((void**)(&kernel_2d_), 4*sizeof(float)) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuMemcpy(kernel_2d_, kernel_2d_val, 4*sizeof(float), gpuMemcpyHostToDevice) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuMalloc((void**)(&kernel_3d_), 8*sizeof(float)) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuMemcpy(kernel_3d_, kernel_3d_val, 8*sizeof(float), gpuMemcpyHostToDevice) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  ~GPUContext() {
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuFree(kernel_1d_) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuFree(kernel_2d_) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuFree(kernel_3d_) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  const Eigen::GpuDevice& device() const { return gpu_device_; }
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::GpuStreamDevice stream_;
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::GpuDevice gpu_device_;
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:void test_gpu() {
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  gpuMalloc((void**)(&d_in1), in1_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  gpuMalloc((void**)(&d_in2), in2_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  gpuMemcpy(d_in2, in2.data(), in2_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, 40,50,70);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, 40,50,70);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, 40,50,70);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  GPUContext context(gpu_in1, gpu_in2, gpu_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpy(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpy(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpy(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpy(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, context.device().stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuStreamSynchronize(context.device().stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, context.device().stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuStreamSynchronize(context.device().stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, context.device().stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuStreamSynchronize(context.device().stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_device.cu:  CALL_SUBTEST_2(test_gpu());
external-libraries/eigen/unsupported/test/CMakeLists.txt:find_package(CUDA 7.0)
external-libraries/eigen/unsupported/test/CMakeLists.txt:if(CUDA_FOUND AND EIGEN_TEST_CUDA)
external-libraries/eigen/unsupported/test/CMakeLists.txt:  # in the CUDA runtime
external-libraries/eigen/unsupported/test/CMakeLists.txt:  message(STATUS "Flags used to compile cuda code: " ${CMAKE_CXX_FLAGS})
external-libraries/eigen/unsupported/test/CMakeLists.txt:    set(CUDA_NVCC_FLAGS "-ccbin ${CMAKE_C_COMPILER}" CACHE STRING "nvcc flags" FORCE)
external-libraries/eigen/unsupported/test/CMakeLists.txt:  if(EIGEN_TEST_CUDA_CLANG)
external-libraries/eigen/unsupported/test/CMakeLists.txt:   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 --cuda-gpu-arch=sm_${EIGEN_CUDA_COMPUTE_ARCH}")
external-libraries/eigen/unsupported/test/CMakeLists.txt:  set(EIGEN_CUDA_RELAXED_CONSTEXPR "--expt-relaxed-constexpr")
external-libraries/eigen/unsupported/test/CMakeLists.txt:  if (${CUDA_VERSION} STREQUAL "7.0")
external-libraries/eigen/unsupported/test/CMakeLists.txt:    set(EIGEN_CUDA_RELAXED_CONSTEXPR "--relaxed-constexpr")
external-libraries/eigen/unsupported/test/CMakeLists.txt:    set(EIGEN_CUDA_CXX11_FLAG "-std=c++11")
external-libraries/eigen/unsupported/test/CMakeLists.txt:    set(EIGEN_CUDA_CXX11_FLAG "")
external-libraries/eigen/unsupported/test/CMakeLists.txt:  set(CUDA_NVCC_FLAGS  "${EIGEN_CUDA_CXX11_FLAG} ${EIGEN_CUDA_RELAXED_CONSTEXPR} -arch compute_${EIGEN_CUDA_COMPUTE_ARCH} -Xcudafe \"--display_error_number\" ${CUDA_NVCC_FLAGS}")
external-libraries/eigen/unsupported/test/CMakeLists.txt:  cuda_include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/include")
external-libraries/eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_complex_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_complex_cwise_ops_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_reduction_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_argmax_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_cast_float16_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_scan_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:  if (${EIGEN_CUDA_COMPUTE_ARCH} GREATER 29)
external-libraries/eigen/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_contract_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_of_float16_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:  if (${EIGEN_CUDA_COMPUTE_ARCH} GREATER 34)
external-libraries/eigen/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_random_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:  set(HIP_PATH "/opt/rocm/hip" CACHE STRING "Path to the HIP installation.")
external-libraries/eigen/unsupported/test/CMakeLists.txt:	# ei_add_test(cxx11_tensor_complex_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:	# ei_add_test(cxx11_tensor_complex_cwise_ops_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_reduction_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_argmax_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_cast_float16_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_scan_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_contract_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_of_float16_gpu)
external-libraries/eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_random_gpu)
external-libraries/eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  DataType* gpu_in_data  = static_cast<DataType*>(sycl_device.allocate(in.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  sycl_device.memset(gpu_in_data, 1, in.size()*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  sycl_device.memcpyDeviceToHost(in.data(), gpu_in_data, in.size()*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  DataType* gpu_data = static_cast<DataType*>(sycl_device.allocate(sizeDim1*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  sycl_device.memset(gpu_data, 1, sizeDim1*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  TensorMap<Tensor<DataType, 1, DataLayout,IndexType>> in(gpu_data, tensorDims);
external-libraries/eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  TensorMap<Tensor<DataType, 1, DataLayout,IndexType>> out(gpu_data, tensorDims);
external-libraries/eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  sycl_device.deallocate(gpu_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor1.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  TensorMap<Tensor<DataType, 3, ColMajor, IndexType>> gpu1(gpu_data1, tensorColRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  TensorMap<Tensor<DataType, 3, RowMajor, IndexType>> gpu2(gpu_data2, tensorRowRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor1.data(),(tensor1.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  gpu2.device(sycl_device)=gpu1.swap_layout();
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor2.data(), gpu_data2,(tensor2.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.deallocate(gpu_data1);
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.deallocate(gpu_data2);
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor1.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  TensorMap<Tensor<DataType, 3, ColMajor, IndexType>> gpu1(gpu_data1, tensorColRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  TensorMap<Tensor<DataType, 3, RowMajor, IndexType>> gpu2(gpu_data2, tensorRowRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor1.data(),(tensor1.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  gpu2.swap_layout().device(sycl_device)=gpu1;
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor2.data(), gpu_data2,(tensor2.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.deallocate(gpu_data1);
external-libraries/eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.deallocate(gpu_data2);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:#define EIGEN_USE_GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:void test_gpu_simple_argmax()
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuMalloc((void**)(&d_in), in_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuMalloc((void**)(&d_out_max), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuMalloc((void**)(&d_out_min), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuMemcpy(d_in, in.data(), in_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<double, 3, Layout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 3>(72,53,97));
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 1, Layout>, Aligned > gpu_out_max(d_out_max, Eigen::array<DenseIndex, 1>(1));
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 1, Layout>, Aligned > gpu_out_min(d_out_min, Eigen::array<DenseIndex, 1>(1));
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpu_out_max.device(gpu_device) = gpu_in.argmax();
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpu_out_min.device(gpu_device) = gpu_in.argmin();
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  assert(gpuMemcpyAsync(out_max.data(), d_out_max, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  assert(gpuMemcpyAsync(out_min.data(), d_out_min, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuFree(d_in);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuFree(d_out_max);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuFree(d_out_min);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:void test_gpu_argmax_dim()
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMalloc((void**)(&d_in), in_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMemcpy(d_in, tensor.data(), in_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 4>(2, 3, 5, 7));
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout>, Aligned > gpu_out(d_out, out_shape);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpu_out.device(gpu_device) = gpu_in.argmax(dim);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuMemcpyAsync(tensor_arg.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMemcpy(d_in, tensor.data(), in_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpu_out.device(gpu_device) = gpu_in.argmax(dim);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuMemcpyAsync(tensor_arg.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuFree(d_in);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:void test_gpu_argmin_dim()
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMalloc((void**)(&d_in), in_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMalloc((void**)(&d_out), out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMemcpy(d_in, tensor.data(), in_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 4>(2, 3, 5, 7));
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout>, Aligned > gpu_out(d_out, out_shape);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpu_out.device(gpu_device) = gpu_in.argmin(dim);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuMemcpyAsync(tensor_arg.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMemcpy(d_in, tensor.data(), in_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpu_out.device(gpu_device) = gpu_in.argmin(dim);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuMemcpyAsync(tensor_arg.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuFree(d_in);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_argmax_gpu)
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  CALL_SUBTEST_1(test_gpu_simple_argmax<RowMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  CALL_SUBTEST_1(test_gpu_simple_argmax<ColMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  CALL_SUBTEST_2(test_gpu_argmax_dim<RowMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  CALL_SUBTEST_2(test_gpu_argmax_dim<ColMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  CALL_SUBTEST_3(test_gpu_argmin_dim<RowMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  CALL_SUBTEST_3(test_gpu_argmin_dim<ColMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:#define EIGEN_USE_GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:void test_gpu_contraction(int m_size, int k_size, int n_size)
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  // a 15 SM GK110 GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  Tensor<float, 2, DataLayout> t_result_gpu(m_size, n_size);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMalloc((void**)(&d_t_left), t_left_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMalloc((void**)(&d_t_right), t_right_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMalloc((void**)(&d_t_result), t_result_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMemcpy(d_t_left, t_left.data(), t_left_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMemcpy(d_t_right, t_right.data(), t_right_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      gpu_t_left(d_t_left, Eigen::array<int, 2>(m_size, k_size));
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      gpu_t_right(d_t_right, Eigen::array<int, 2>(k_size, n_size));
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      gpu_t_result(d_t_result, Eigen::array<int, 2>(m_size, n_size));
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, gpuMemcpyDeviceToHost);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    if (fabs(t_result(i) - t_result_gpu(i)) < 1e-4f) {
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), 1e-4f)) {
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:              << " vs " <<  t_result_gpu(i) << std::endl;
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuFree((void*)d_t_left);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuFree((void*)d_t_right);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuFree((void*)d_t_result);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  // a 15 SM GK110 GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  Tensor<float, 0, DataLayout> t_result_gpu;
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMalloc((void**)(&d_t_left), t_left_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMalloc((void**)(&d_t_right), t_right_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMalloc((void**)(&d_t_result), t_result_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMemcpy(d_t_left, t_left.data(), t_left_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMemcpy(d_t_right, t_right.data(), t_right_bytes, gpuMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      gpu_t_left(d_t_left, m_size, k_size);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      gpu_t_right(d_t_right, k_size, n_size);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      gpu_t_result(d_t_result);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, gpuMemcpyDeviceToHost);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  if (fabs(t_result() - t_result_gpu()) > 1e-4f &&
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      !Eigen::internal::isApprox(t_result(), t_result_gpu(), 1e-4f)) {
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:              << " vs " <<  t_result_gpu() << std::endl;
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuFree((void*)d_t_left);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuFree((void*)d_t_right);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuFree((void*)d_t_result);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:void test_gpu_contraction_m() {
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    test_gpu_contraction<ColMajor>(k, 128, 128);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    test_gpu_contraction<RowMajor>(k, 128, 128);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:void test_gpu_contraction_k() {
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    test_gpu_contraction<ColMajor>(128, k, 128);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    test_gpu_contraction<RowMajor>(128, k, 128);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:void test_gpu_contraction_n() {
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    test_gpu_contraction<ColMajor>(128, 128, k);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    test_gpu_contraction<RowMajor>(128, 128, k);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:void test_gpu_contraction_sizes() {
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:        test_gpu_contraction<DataLayout>(m_sizes[i], n_sizes[j], k_sizes[k]);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_contract_gpu)
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_1(test_gpu_contraction<ColMajor>(128, 128, 128));
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_1(test_gpu_contraction<RowMajor>(128, 128, 128));
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_2(test_gpu_contraction_m<ColMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_3(test_gpu_contraction_m<RowMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_4(test_gpu_contraction_k<ColMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_5(test_gpu_contraction_k<RowMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_6(test_gpu_contraction_n<ColMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_7(test_gpu_contraction_n<RowMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_8(test_gpu_contraction_sizes<ColMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_9(test_gpu_contraction_sizes<RowMajor>());
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  // a 15 SM GK110 GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Tensor<DataType, 2, DataLayout, IndexType> t_result_gpu(m_size, n_size);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_left(d_t_left, left_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_right(d_t_right, right_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_result(d_t_result, result_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result, t_result_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:    if (static_cast<DataType>(fabs(t_result(i) - t_result_gpu(i))) < error_threshold) {
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), error_threshold)) {
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:              << " vs " <<  t_result_gpu(i) << std::endl;
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Tensor<DataType, 2, DataLayout, IndexType> t_result_gpu(res_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_left(d_t_left, left_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_right(d_t_right, right_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_result(d_t_result, res_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result, t_result_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:    if (static_cast<DataType>(fabs(t_result(i) - t_result_gpu(i))) < error_threshold) {
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), error_threshold)) {
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:              << " vs " <<  t_result_gpu(i) << std::endl;
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  // a 15 SM GK110 GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Tensor<DataType, 0, DataLayout, IndexType> t_result_gpu;
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_left(d_t_left, left_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_right(d_t_right, right_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 0, DataLayout, IndexType> > gpu_t_result(d_t_result);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result, t_result_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  if (static_cast<DataType>(fabs(t_result() - t_result_gpu())) > error_threshold &&
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:      !Eigen::internal::isApprox(t_result(), t_result_gpu(), error_threshold)) {
external-libraries/eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:              << " vs " <<  t_result_gpu() << std::endl;
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:#define EIGEN_USE_GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_numext() {
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  bool* d_res_half = (bool*)gpu_device.allocate(num_elem * sizeof(bool));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  bool* d_res_float = (bool*)gpu_device.allocate(num_elem * sizeof(bool));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_res_half(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_res_float(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float.unaryExpr(Eigen::internal::scalar_isnan_op<float>());
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = gpu_float.cast<Eigen::half>().unaryExpr(Eigen::internal::scalar_isnan_op<Eigen::half>());
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(bool));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(bool));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:#ifdef EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_conversion() {
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_conv = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_half(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_conv(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float.device(gpu_device) = gpu_float.random();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_half.device(gpu_device) = gpu_float.cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_conv.device(gpu_device) = gpu_half.cast<float>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(initial.data(), d_float, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(final.data(), d_conv, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_half);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_conv);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_unary() {
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_half = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float.abs();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().cast<float>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_elementwise() {
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_half = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float1(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float2(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float1.device(gpu_device) = gpu_float1.random();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float2.device(gpu_device) = gpu_float2.random();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = (gpu_float1 + gpu_float2) * gpu_float1;
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = ((gpu_float1.cast<Eigen::half>() + gpu_float2.cast<Eigen::half>()) * gpu_float1.cast<Eigen::half>()).cast<float>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float1);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float2);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_trancendental() {
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float3 = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res1_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res1_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res2_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res2_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res3_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res3_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float1(d_float1, num_elem);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float2(d_float2, num_elem);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float3(d_float3, num_elem);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res1_half(d_res1_half, num_elem);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res1_float(d_res1_float, num_elem);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res2_half(d_res2_half, num_elem);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res2_float(d_res2_float, num_elem);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res3_half(d_res3_half, num_elem);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res3_float(d_res3_float, num_elem);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res4_half(d_res3_half, num_elem);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res4_float(d_res3_float, num_elem);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float1.device(gpu_device) = gpu_float1.random() - gpu_float1.constant(0.5f);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float2.device(gpu_device) = gpu_float2.random() + gpu_float1.constant(0.5f);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float3.device(gpu_device) = gpu_float3.random();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res1_float.device(gpu_device) = gpu_float1.exp().cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res2_float.device(gpu_device) = gpu_float2.log().cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res3_float.device(gpu_device) = gpu_float3.log1p().cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res4_float.device(gpu_device) = gpu_float3.expm1().cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res1_half.device(gpu_device) = gpu_float1.cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res1_half.device(gpu_device) = gpu_res1_half.exp();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res2_half.device(gpu_device) = gpu_float2.cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res2_half.device(gpu_device) = gpu_res2_half.log();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res3_half.device(gpu_device) = gpu_float3.cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res3_half.device(gpu_device) = gpu_res3_half.log1p();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res3_half.device(gpu_device) = gpu_float3.cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res3_half.device(gpu_device) = gpu_res3_half.expm1();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(input1.data(), d_float1, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(input2.data(), d_float2, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(input3.data(), d_float3, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec1.data(), d_res1_half, num_elem*sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec1.data(), d_res1_float, num_elem*sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec2.data(), d_res2_half, num_elem*sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec2.data(), d_res2_float, num_elem*sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec3.data(), d_res3_half, num_elem*sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec3.data(), d_res3_float, num_elem*sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float1);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float2);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float3);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res1_half);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res1_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res2_half);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res2_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res3_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res3_half);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_contractions() {
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2>, Eigen::Aligned> gpu_res_half(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2>, Eigen::Aligned> gpu_res_float(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float1.device(gpu_device) = gpu_float1.random() - gpu_float1.constant(0.5f);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float2.device(gpu_device) = gpu_float2.random() - gpu_float2.constant(0.5f);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float1.contract(gpu_float2, dims).cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().contract(gpu_float2.cast<Eigen::half>(), dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float1);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float2);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_reductions(int size1, int size2, int redux) {
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(result_size * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(result_size * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res_half(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res_float(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float1.device(gpu_device) = gpu_float1.random() * 2.0f;
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float2.device(gpu_device) = gpu_float2.random() * 2.0f;
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float1.sum(redux_dim).cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().sum(redux_dim);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, result_size*sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, result_size*sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float1);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float2);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_reductions() {
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  test_gpu_reductions<void>(13, 13, 0);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  test_gpu_reductions<void>(13, 13, 1);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  test_gpu_reductions<void>(35, 36, 0);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  test_gpu_reductions<void>(35, 36, 1);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  test_gpu_reductions<void>(36, 35, 0);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  test_gpu_reductions<void>(36, 35, 1);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_full_reductions() {
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(1 * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(1 * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 0>, Eigen::Aligned> gpu_res_half(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 0>, Eigen::Aligned> gpu_res_float(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float1.device(gpu_device) = gpu_float1.random();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float2.device(gpu_device) = gpu_float2.random();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float1.sum().cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().sum();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float1.maximum().cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().maximum();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float1);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float2);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_forced_evals() {
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_half1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_half2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half1(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu: Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Unaligned> gpu_res_half2(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float.abs();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half1.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().eval().cast<float>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half2.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().broadcast(no_bcast).eval().cast<float>();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec1.data(), d_res_half1, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec2.data(), d_res_half1, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half1);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half2);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_of_float16_gpu)
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_1(test_gpu_numext<void>());
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:#ifdef EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_1(test_gpu_conversion<void>());
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_1(test_gpu_unary<void>());
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_1(test_gpu_elementwise<void>());
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_1(test_gpu_trancendental<void>());
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_2(test_gpu_contractions<void>());
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_3(test_gpu_reductions<void>());
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_4(test_gpu_full_reductions<void>());
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_5(test_gpu_forced_evals<void>());
external-libraries/eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  std::cout << "Half floats are not supported by this version of gpu: skipping the test" << std::endl;
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:#define EIGEN_USE_GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:void test_gpu_conversion() {
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  Eigen::half* d_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  float* d_conv = (float*)gpu_device.allocate(num_elem * sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_half(
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_conv(
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.memcpyHostToDevice(d_float, floats.data(), num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_half.device(gpu_device) = gpu_float.cast<Eigen::half>();
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_conv.device(gpu_device) = gpu_half.cast<float>();
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(initial.data(), d_float, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(final.data(), d_conv, num_elem*sizeof(float));
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.deallocate(d_float);
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.deallocate(d_half);
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.deallocate(d_conv);
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_cast_float16_gpu)
external-libraries/eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  CALL_SUBTEST(test_gpu_conversion());
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<DataType, 0, DataLayout, IndexType> full_redux_gpu;
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_in_data = static_cast<DataType*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_out_data =(DataType*)sycl_device.allocate(sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout, IndexType> >  in_gpu(gpu_in_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 0, DataLayout, IndexType> >  out_gpu(gpu_out_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.mean();
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data, sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<DataType, 0, DataLayout, IndexType> full_redux_gpu;
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_in_data = static_cast<DataType*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_out_data =(DataType*)sycl_device.allocate(sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout, IndexType> >  in_gpu(gpu_in_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 0, DataLayout, IndexType> >  out_gpu(gpu_out_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.minimum();
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data, sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<DataType, 2, DataLayout, IndexType> redux_gpu(reduced_tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_in_data = static_cast<DataType*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_out_data = static_cast<DataType*>(sycl_device.allocate(redux_gpu.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType> >  in_gpu(gpu_in_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout, IndexType> >  out_gpu(gpu_out_data, reduced_tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.maximum(red_axis);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(redux_gpu.data(), gpu_out_data, redux_gpu.dimensions().TotalSize()*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:      VERIFY_IS_APPROX(redux_gpu(j,k), redux(j,k));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<DataType, 2, DataLayout, IndexType> redux_gpu(reduced_tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_in_data = static_cast<DataType*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_out_data = static_cast<DataType*>(sycl_device.allocate(redux_gpu.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType> >  in_gpu(gpu_in_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout, IndexType> >  out_gpu(gpu_out_data, reduced_tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.sum(red_axis);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(redux_gpu.data(), gpu_out_data, redux_gpu.dimensions().TotalSize()*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:      VERIFY_IS_APPROX(redux_gpu(j,k), redux(j,k));
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  DataType* gpu_data_no_stride  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_no_stride(gpu_data_no_stride, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  gpu_no_stride.device(sycl_device)=gpu_tensor.inflate(strides);
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  sycl_device.memcpyDeviceToHost(no_stride.data(), gpu_data_no_stride, tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  DataType* gpu_data_inflated  = static_cast<DataType*>(sycl_device.allocate(inflatedTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>> gpu_inflated(gpu_data_inflated, inflatedTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  gpu_inflated.device(sycl_device)=gpu_tensor.inflate(strides);
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  sycl_device.memcpyDeviceToHost(inflated.data(), gpu_data_inflated, inflatedTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor);
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  sycl_device.deallocate(gpu_data_no_stride);
external-libraries/eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  sycl_device.deallocate(gpu_data_inflated);
external-libraries/eigen/unsupported/test/openglsupport.cpp:    #ifdef GLEW_ARB_gpu_shader_fp64
external-libraries/eigen/unsupported/test/openglsupport.cpp:    if(GLEW_ARB_gpu_shader_fp64)
external-libraries/eigen/unsupported/test/openglsupport.cpp:      #ifdef GL_ARB_gpu_shader_fp64
external-libraries/eigen/unsupported/test/openglsupport.cpp:      std::cerr << "Warning: GLEW_ARB_gpu_shader_fp64 was not tested\n";
external-libraries/eigen/unsupported/test/openglsupport.cpp:      std::cerr << "Warning: GLEW_ARB_gpu_shader_fp64 was not tested\n";
external-libraries/eigen/unsupported/test/cxx11_tensor_of_strings.cpp:  // Beware: none of this is likely to ever work on a GPU.
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, Layout, DenseIndex> > gpu_in(d_in, Eigen::array<DenseIndex, 3>{{2,2,2}});
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 0, Layout, DenseIndex> > gpu_out_max(d_out_max);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 0, Layout, DenseIndex> > gpu_out_min(d_out_min);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:  gpu_out_max.device(sycl_device) = gpu_in.argmax();
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:  gpu_out_min.device(sycl_device) = gpu_in.argmin();
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, DenseIndex> > gpu_in(d_in, Eigen::array<DenseIndex, 4>{{sizeDim0,sizeDim1,sizeDim2,sizeDim3}});
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout, DenseIndex> > gpu_out(d_out, out_shape);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    gpu_out.device(sycl_device) = gpu_in.argmax(dim);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    gpu_out.device(sycl_device) = gpu_in.argmax(dim);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, DenseIndex> > gpu_in(d_in, Eigen::array<DenseIndex, 4>{{sizeDim0,sizeDim1,sizeDim2,sizeDim3}});
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout, DenseIndex> > gpu_out(d_out, out_shape);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    gpu_out.device(sycl_device) = gpu_in.argmin(dim);
external-libraries/eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    gpu_out.device(sycl_device) = gpu_in.argmin(dim);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_single_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_single_patch_col_major(gpu_data_single_patch_col_major, patchColMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_single_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(1, 1);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_patch_col_major.data(), gpu_data_single_patch_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_single_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_single_patch_row_major(gpu_data_single_patch_row_major, patchRowMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_single_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(1, 1);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_patch_row_major.data(), gpu_data_single_patch_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_entire_image_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_entire_image_patch_col_major(gpu_data_entire_image_patch_col_major, patchColMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_entire_image_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(3, 5);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(entire_image_patch_col_major.data(), gpu_data_entire_image_patch_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_entire_image_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_entire_image_patch_row_major(gpu_data_entire_image_patch_row_major, patchRowMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_entire_image_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(3, 5);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(entire_image_patch_row_major.data(), gpu_data_entire_image_patch_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_twod_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_twod_patch_col_major(gpu_data_twod_patch_col_major, patchColMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_twod_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(2, 2);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(twod_patch_col_major.data(), gpu_data_twod_patch_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_twod_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_twod_patch_row_major(gpu_data_twod_patch_row_major, patchRowMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_twod_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(2, 2);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(twod_patch_row_major.data(), gpu_data_twod_patch_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_patch_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_patch_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_entire_image_patch_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_entire_image_patch_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_twod_patch_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_twod_patch_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_result_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_result_col_major(gpu_data_result_col_major, patchColMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_result_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(ksize, ksize, stride, stride, 1, 1, PADDING_VALID);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(result_col_major.data(), gpu_data_result_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_result_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_result_row_major(gpu_data_result_row_major, patchRowMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_result_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(ksize, ksize, stride, stride, 1, 1, PADDING_VALID);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(result_row_major.data(), gpu_data_result_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_result_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_result_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_col_major.device(sycl_device)=gpu_col_major.constant(11.0f);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor_col_major.data(), gpu_data_col_major, (tensor_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_row_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_result_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_result_col_major(gpu_data_result_col_major, patchColMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_result_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(ksize, ksize, stride, stride, 1, 1, PADDING_VALID);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(result_col_major.data(), gpu_data_result_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_result_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_result_row_major(gpu_data_result_row_major, patchRowMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_result_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(ksize, ksize, stride, stride, 1, 1, PADDING_VALID);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(result_row_major.data(), gpu_data_result_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:DataType* gpu_data_result_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_result_col_major(gpu_data_result_col_major, patchColMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:gpu_result_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(ksize, ksize, stride, stride, PADDING_SAME);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:sycl_device.memcpyDeviceToHost(result_col_major.data(), gpu_data_result_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_result_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_result_row_major(gpu_data_result_row_major, patchRowMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_result_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(ksize, ksize, stride, stride, PADDING_SAME);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(result_row_major.data(), gpu_data_result_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 3, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 3, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_row_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_single_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_single_patch_col_major(gpu_data_single_patch_col_major, patchColMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_single_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(1, 1);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_patch_col_major.data(), gpu_data_single_patch_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_single_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, RowMajor,IndexType>> gpu_single_patch_row_major(gpu_data_single_patch_row_major, patchRowMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_single_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(1, 1);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_patch_row_major.data(), gpu_data_single_patch_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_entire_image_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_entire_image_patch_col_major(gpu_data_entire_image_patch_col_major, patchColMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_entire_image_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(3, 5);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(entire_image_patch_col_major.data(), gpu_data_entire_image_patch_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:DataType* gpu_data_entire_image_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:TensorMap<Tensor<DataType, 4, RowMajor,IndexType>> gpu_entire_image_patch_row_major(gpu_data_entire_image_patch_row_major, patchRowMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:gpu_entire_image_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(3, 5);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:sycl_device.memcpyDeviceToHost(entire_image_patch_row_major.data(), gpu_data_entire_image_patch_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_twod_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_twod_patch_col_major(gpu_data_twod_patch_col_major, patchColMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_twod_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(2, 2);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(twod_patch_col_major.data(), gpu_data_twod_patch_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_twod_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, RowMajor,IndexType>> gpu_twod_patch_row_major(gpu_data_twod_patch_row_major, patchRowMajorTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_twod_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(2, 2);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(twod_patch_row_major.data(), gpu_data_twod_patch_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_patch_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_patch_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_entire_image_patch_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_entire_image_patch_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_twod_patch_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_twod_patch_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_l_in_col_major  = static_cast<DataType*>(sycl_device.allocate(l_in_col_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_l_in_col_major(gpu_data_l_in_col_major, tensorColMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_l_in_col_major, l_in_col_major.data(),(l_in_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_l_out_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_l_out_col_major(gpu_data_l_out_col_major, patchTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_col_major.device(sycl_device)=gpu_l_in_col_major.extract_image_patches(11, 11);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_col_major.data(), gpu_data_l_out_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_l_out_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_l_out_row_major(gpu_data_l_out_row_major, patchTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_row_major.device(sycl_device)=gpu_l_in_col_major.swap_layout().extract_image_patches(11, 11);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_row_major.data(), gpu_data_l_out_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_in_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_in_col_major  = static_cast<DataType*>(sycl_device.allocate(l_in_col_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>>gpu_l_in_col_major_resize1(gpu_data_l_in_col_major, tensorColMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_out_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>>gpu_l_out_col_major_resize1(gpu_data_l_out_col_major, patchTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_l_in_col_major, l_in_col_major.data(),(l_in_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_col_major_resize1.device(sycl_device)=gpu_l_in_col_major_resize1.extract_image_patches(9, 9);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_col_major.data(), gpu_data_l_out_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_out_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>>gpu_l_out_row_major_resize1(gpu_data_l_out_row_major, patchTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_row_major_resize1.device(sycl_device)=gpu_l_in_col_major_resize1.swap_layout().extract_image_patches(9, 9);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_row_major.data(), gpu_data_l_out_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_in_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_in_col_major  = static_cast<DataType*>(sycl_device.allocate(l_in_col_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>>gpu_l_in_col_major_resize2(gpu_data_l_in_col_major, tensorColMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_out_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>>gpu_l_out_col_major_resize2(gpu_data_l_out_col_major, patchTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_l_in_col_major, l_in_col_major.data(),(l_in_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_col_major_resize2.device(sycl_device)=gpu_l_in_col_major_resize2.extract_image_patches(7, 7);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_col_major.data(), gpu_data_l_out_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_out_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>>gpu_l_out_row_major_resize2(gpu_data_l_out_row_major, patchTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_row_major_resize2.device(sycl_device)=gpu_l_in_col_major_resize2.swap_layout().extract_image_patches(7, 7);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_row_major.data(), gpu_data_l_out_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_in_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_in_col_major  = static_cast<DataType*>(sycl_device.allocate(l_in_col_major.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>>gpu_l_in_col_major_resize3(gpu_data_l_in_col_major, tensorColMajorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_out_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>>gpu_l_out_col_major_resize3(gpu_data_l_out_col_major, patchTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_l_in_col_major, l_in_col_major.data(),(l_in_col_major.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_col_major_resize3.device(sycl_device)=gpu_l_in_col_major_resize3.extract_image_patches(3, 3);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_col_major.data(), gpu_data_l_out_col_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_out_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>>gpu_l_out_row_major_resize3(gpu_data_l_out_row_major, patchTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_row_major_resize3.device(sycl_device)=gpu_l_in_col_major_resize3.swap_layout().extract_image_patches(3, 3);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_row_major.data(), gpu_data_l_out_row_major, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_in_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_col_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_row_major);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:#define EIGEN_USE_GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:void test_cuda_nullary() {
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaMalloc((void**)(&d_in1), complex_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaMalloc((void**)(&d_in2), complex_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaMalloc((void**)(&d_out2), float_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaMemcpy(d_in1, in1.data(), complex_bytes, cudaMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaMemcpy(d_in2, in2.data(), complex_bytes, cudaMemcpyHostToDevice);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1, 0, int>, Eigen::Aligned> gpu_in1(
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1, 0, int>, Eigen::Aligned> gpu_in2(
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_out2(
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_in1.device(gpu_device) = gpu_in1.constant(std::complex<float>(3.14f, 2.7f));
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_out2.device(gpu_device) = gpu_in2.abs();
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  assert(cudaMemcpyAsync(new1.data(), d_in1, complex_bytes, cudaMemcpyDeviceToHost,
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:                         gpu_device.stream()) == cudaSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  assert(cudaMemcpyAsync(new2.data(), d_out2, float_bytes, cudaMemcpyDeviceToHost,
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:                         gpu_device.stream()) == cudaSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaFree(d_in1);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaFree(d_in2);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaFree(d_out2);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:static void test_cuda_sum_reductions() {
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  std::complex<float>* gpu_in_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(in_bytes));
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  std::complex<float>* gpu_out_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(out_bytes));
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  TensorMap<Tensor<std::complex<float>, 2> > in_gpu(gpu_in_ptr, num_rows, num_cols);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  TensorMap<Tensor<std::complex<float>, 0> > out_gpu(gpu_out_ptr);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  out_gpu.device(gpu_device) = in_gpu.sum();
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Tensor<std::complex<float>, 0> full_redux_gpu;
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  // Check that the CPU and GPU reductions return the same result.
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.deallocate(gpu_in_ptr);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.deallocate(gpu_out_ptr);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:static void test_cuda_mean_reductions() {
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  std::complex<float>* gpu_in_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(in_bytes));
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  std::complex<float>* gpu_out_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(out_bytes));
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  TensorMap<Tensor<std::complex<float>, 2> > in_gpu(gpu_in_ptr, num_rows, num_cols);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  TensorMap<Tensor<std::complex<float>, 0> > out_gpu(gpu_out_ptr);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  out_gpu.device(gpu_device) = in_gpu.mean();
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Tensor<std::complex<float>, 0> full_redux_gpu;
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  // Check that the CPU and GPU reductions return the same result.
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.deallocate(gpu_in_ptr);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.deallocate(gpu_out_ptr);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:static void test_cuda_product_reductions() {
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  std::complex<float>* gpu_in_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(in_bytes));
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  std::complex<float>* gpu_out_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(out_bytes));
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  TensorMap<Tensor<std::complex<float>, 2> > in_gpu(gpu_in_ptr, num_rows, num_cols);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  TensorMap<Tensor<std::complex<float>, 0> > out_gpu(gpu_out_ptr);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  out_gpu.device(gpu_device) = in_gpu.prod();
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Tensor<std::complex<float>, 0> full_redux_gpu;
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.synchronize();
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  // Check that the CPU and GPU reductions return the same result.
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.deallocate(gpu_in_ptr);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.deallocate(gpu_out_ptr);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  CALL_SUBTEST(test_cuda_nullary());
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  CALL_SUBTEST(test_cuda_sum_reductions());
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  CALL_SUBTEST(test_cuda_mean_reductions());
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  CALL_SUBTEST(test_cuda_product_reductions());
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:#define EIGEN_USE_GPU
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:void test_cuda_complex_cwise_ops() {
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  cudaMalloc((void**)(&d_in1), complex_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  cudaMalloc((void**)(&d_in2), complex_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  cudaMalloc((void**)(&d_out), complex_bytes);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  Eigen::GpuStreamDevice stream;
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_in1(
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_in2(
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_out(
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  gpu_in1.device(gpu_device) = gpu_in1.constant(a);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  gpu_in2.device(gpu_device) = gpu_in2.constant(b);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:        gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:        gpu_out.device(gpu_device) = gpu_in1 - gpu_in2;
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:        gpu_out.device(gpu_device) = gpu_in1 * gpu_in2;
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:        gpu_out.device(gpu_device) = gpu_in1 / gpu_in2;
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:        gpu_out.device(gpu_device) = -gpu_in1;
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:    assert(cudaMemcpyAsync(actual.data(), d_out, complex_bytes, cudaMemcpyDeviceToHost,
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:                           gpu_device.stream()) == cudaSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  cudaFree(d_in1);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  cudaFree(d_in2);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  cudaFree(d_out);
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  CALL_SUBTEST(test_cuda_complex_cwise_ops<float>());
external-libraries/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  CALL_SUBTEST(test_cuda_complex_cwise_ops<double>());
external-libraries/eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(buffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(buffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu1(gpu_data1, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu2(gpu_data2, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(), buffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  gpu2.device(sycl_device)=gpu1.shuffle(shuffles);
external-libraries/eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  sycl_device.memcpyDeviceToHost(no_shuffle.data(), gpu_data2, buffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(buffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  TensorMap<Tensor<DataType, 4,DataLayout,IndexType>> gpu3(gpu_data3, tensorrangeShuffle);
external-libraries/eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  gpu3.device(sycl_device)=gpu1.shuffle(shuffles);
external-libraries/eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  sycl_device.memcpyDeviceToHost(shuffle.data(), gpu_data3, buffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data = static_cast<SCALAR *>(                                  \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu(gpu_data, tensorRange);          \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data, in.data(),                        \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_out, out.data(),                   \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    gpu_out.device(sycl_device) OPERATOR gpu.FUNC();                           \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data);                                          \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_out);                                      \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_out, out.data(),                   \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    gpu_out.device(sycl_device) OPERATOR gpu_out.FUNC();                       \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_out);                                      \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data = static_cast<SCALAR *>(                                  \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    bool *gpu_data_out =                                                       \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu(gpu_data, tensorRange);          \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<bool, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);    \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data, in.data(),                        \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    gpu_out.device(sycl_device) = gpu.FUNC();                                  \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data);                                          \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_out);                                      \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_1 = static_cast<SCALAR *>(                                \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_2 = static_cast<SCALAR *>(                                \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_1(gpu_data_1, tensorRange);      \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_2(gpu_data_2, tensorRange);      \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_1, in_1.data(),                    \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_2, in_2.data(),                    \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    gpu_out.device(sycl_device) = gpu_1.FUNC(gpu_2);                           \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_1);                                        \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_2);                                        \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_out);                                      \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_1 = static_cast<SCALAR *>(                                \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_2 = static_cast<SCALAR *>(                                \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_1(gpu_data_1, tensorRange);      \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_2(gpu_data_2, tensorRange);      \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_1, in_1.data(),                    \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_2, in_2.data(),                    \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    gpu_out.device(sycl_device) = gpu_1 OPERATOR gpu_2;                        \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_1);                                        \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_2);                                        \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_out);                                      \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_1 = static_cast<SCALAR *>(                                \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_1(gpu_data_1, tensorRange);      \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_1, in_1.data(),                    \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    gpu_out.device(sycl_device) = gpu_1 OPERATOR 2;                            \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_1);                                        \
external-libraries/eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_out);                                      \
external-libraries/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(in2.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_in1(gpu_in1_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_in2(gpu_in2_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_out(gpu_out_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in1_data, in1.data(),(in1.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in2.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  gpu_out.device(sycl_device) =(gpu_in1 + gpu_in2).eval() * gpu_in2;
external-libraries/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.deallocate(gpu_in1_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.deallocate(gpu_in2_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  DataType* gpu_data_no_patch  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_no_patch(gpu_data_no_patch, patchTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  gpu_no_patch.device(sycl_device)=gpu_tensor.extract_patches(patch_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(no_patch.data(), gpu_data_no_patch, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  DataType* gpu_data_single_patch  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_single_patch(gpu_data_single_patch, patchTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  gpu_single_patch.device(sycl_device)=gpu_tensor.extract_patches(patch_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_patch.data(), gpu_data_single_patch, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  DataType* gpu_data_twod_patch  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_twod_patch(gpu_data_twod_patch, patchTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  gpu_twod_patch.device(sycl_device)=gpu_tensor.extract_patches(patch_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(twod_patch.data(), gpu_data_twod_patch, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  DataType* gpu_data_threed_patch  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_threed_patch(gpu_data_threed_patch, patchTensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  gpu_threed_patch.device(sycl_device)=gpu_tensor.extract_patches(patch_dims);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(threed_patch.data(), gpu_data_threed_patch, patchTensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_no_patch);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_patch);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_twod_patch);
external-libraries/eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_threed_patch);
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  DataType* gpu_data_vec  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  DataType* gpu_data_result  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  TensorMap<Tensor<DataType, 1, DataLayout,IndexType>> gpu_vec(gpu_data_vec, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  TensorMap<Tensor<DataType, 1, DataLayout,IndexType>> gpu_result(gpu_data_result, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_vec, vec.data(), tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  gpu_result.device(sycl_device)=gpu_vec.generate(Generator1D());
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  sycl_device.memcpyDeviceToHost(result.data(), gpu_data_result, tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  DataType* gpu_data_matrix  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  DataType* gpu_data_result  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout,IndexType>> gpu_matrix(gpu_data_matrix, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout,IndexType>> gpu_result(gpu_data_result, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_matrix, matrix.data(), tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  gpu_result.device(sycl_device)=gpu_matrix.generate(Generator2D());
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  sycl_device.memcpyDeviceToHost(result.data(), gpu_data_result, tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  DataType* gpu_data_matrix  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  DataType* gpu_data_result  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout,IndexType>> gpu_matrix(gpu_data_matrix, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout,IndexType>> gpu_result(gpu_data_result, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_matrix, matrix.data(), tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  gpu_result.device(sycl_device)=gpu_matrix.generate(gaussian_gen);
external-libraries/eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  sycl_device.memcpyDeviceToHost(result.data(), gpu_data_result, tensorBuffSize);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor1.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(tensor3.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data4  = static_cast<DataType*>(sycl_device.allocate(tensor4.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu1(gpu_data1, dim1);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 3,DataLayout, IndexType>> gpu2(gpu_data2, dim2);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 2,DataLayout, IndexType>> gpu3(gpu_data3, dim3);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 2,DataLayout, IndexType>> gpu4(gpu_data4, dim4);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor1.data(),(tensor1.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu2.device(sycl_device)=gpu1.reshape(dim2);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor2.data(), gpu_data2,(tensor1.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu3.device(sycl_device)=gpu1.reshape(dim3);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor3.data(), gpu_data3,(tensor3.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu4.device(sycl_device)=gpu1.reshape(dim2).reshape(dim4);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor4.data(), gpu_data4,(tensor4.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data1);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data2);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data3);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data4);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2d.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(tensor5d.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap< Tensor<DataType, 3, DataLayout, IndexType> > gpu1(gpu_data1, dim1);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap< Tensor<DataType, 2, DataLayout, IndexType> > gpu2(gpu_data2, dim2);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap< Tensor<DataType, 5, DataLayout, IndexType> > gpu3(gpu_data3, dim3);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu2.reshape(dim1).device(sycl_device)=gpu1;
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor2d.data(), gpu_data2,(tensor2d.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu3.reshape(dim1).device(sycl_device)=gpu1;
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor5d.data(), gpu_data3,(tensor5d.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data1);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data2);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data3);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(slice1.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu1(gpu_data1, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu2(gpu_data2, slice1_range);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu2.device(sycl_device)=gpu1.slice(indices, sizes);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(slice1.data(), gpu_data2,(slice1.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(slice2.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu3(gpu_data3, slice2_range);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu3.device(sycl_device)=gpu1.slice(indices2, sizes2);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(slice2.data(), gpu_data3,(slice2.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data1);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data2);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data3);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(slice.size()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 2,DataLayout,IndexType>> gpu1(gpu_data1, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 2,DataLayout,IndexType>> gpu2(gpu_data2, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 2,DataLayout,IndexType>> gpu3(gpu_data3, sliceRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu2.device(sycl_device)=gpu1;
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data3, slice.data(),(slice.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu1.slice(indicesStart,lengths).device(sycl_device)=gpu3;
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu2.stridedSlice(indicesStart,indicesStop,strides).device(sycl_device)=gpu3;
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data1,(tensor.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor2.data(), gpu_data2,(tensor2.size())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data1);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data2);
external-libraries/eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data3);
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  TensorType gpu_in1(gpu_in1_data, tensorRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  TensorType gpu_out(gpu_out_data, tensorResultRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in1_data, in1.data(),(in1.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1.customOp(InsertZeros<TensorType>());
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(in2.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  TensorType gpu_in1(gpu_in1_data, tensorRange1);
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  TensorType gpu_in2(gpu_in2_data, tensorRange2);
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  TensorType gpu_out(gpu_out_data, tensorResultRange);
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in1_data, in1.data(),(in1.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in2.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1.customOp(gpu_in2, BatchMatMul<TensorType>());
external-libraries/eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));
external-libraries/eigen/unsupported/Eigen/CXX11/Tensor:#ifdef EIGEN_USE_GPU
external-libraries/eigen/unsupported/Eigen/CXX11/Tensor:    #include <cuda_runtime.h>
external-libraries/eigen/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorDeviceGpu.h"
external-libraries/eigen/unsupported/Eigen/CXX11/Tensor:#ifndef gpu_assert
external-libraries/eigen/unsupported/Eigen/CXX11/Tensor:#define gpu_assert(x)
external-libraries/eigen/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorReductionGpu.h"
external-libraries/eigen/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorContractionGpu.h"
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:  gpu_assert(threadIdx.z == 0);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:// GPU: the evaluation of the expression is offloaded to a GPU.
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#if defined(EIGEN_USE_GPU)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:class TensorExecutor<Expression, GpuDevice, Vectorizable, Tileable> {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:  static void run(const Expression& expr, const GpuDevice& device);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#if defined(EIGEN_GPUCC)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:EIGEN_STRONG_INLINE void TensorExecutor<Expression, GpuDevice, Vectorizable, Tileable>::run(
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const Expression& expr, const GpuDevice& device) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:  TensorEvaluator<Expression, GpuDevice> evaluator(expr, device);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const int block_size = device.maxGpuThreadsPerBlock();
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const int max_blocks = device.getNumGpuMultiProcessors() *
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:                           device.maxGpuThreadsPerMultiProcessor() / block_size;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    LAUNCH_GPU_KERNEL(
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:        (EigenMetaKernel<TensorEvaluator<Expression, GpuDevice>, StorageIndex>),
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#endif  // EIGEN_GPUCC
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#endif  // EIGEN_USE_GPU
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef EIGEN_GPU_COMPILE_PHASE
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef EIGEN_GPU_COMPILE_PHASE
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__SIZEOF_INT128__) && !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(__SYCL_DEVICE_ONLY__)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef EIGEN_GPU_COMPILE_PHASE
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#if !defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(__SYCL_DEVICE_ONLY__)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device, return the amount of shared memory available.
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(__SYCL_DEVICE_ONLY__)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#if !defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    return EIGEN_CUDA_ARCH / 100;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_GPU_H
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_GPU_H
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:#if defined(EIGEN_USE_GPU) && defined(EIGEN_GPUCC)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:#if defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER < 90000)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    #if defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER < 90000)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType, OutputKernelType>, GpuDevice> :
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    public TensorContractionEvaluatorBase<TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType, OutputKernelType>, GpuDevice> > {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:                "GPU tensor contraction does not support output kernels.");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:  typedef GpuDevice Device;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:  typedef typename PacketType<CoeffReturnType, GpuDevice>::type PacketReturnType;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    static void Run(const LhsMapper& lhs, const RhsMapper& rhs, const OutputMapper& output, Index m, Index n, Index k, const GpuDevice& device) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    LAUNCH_GPU_KERNEL((EigenContractionKernel<Scalar, Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    static void Run(const LhsMapper& lhs, const RhsMapper& rhs, const OutputMapper& output, Index m, Index n, Index k, const GpuDevice& device) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:        LAUNCH_GPU_KERNEL((EigenFloatContractionKernel16x16<Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:        LAUNCH_GPU_KERNEL((EigenFloatContractionKernel<Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    setGpuSharedMemConfig(hipSharedMemBankSizeEightByte);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    setGpuSharedMemConfig(cudaSharedMemBankSizeEightByte);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:#endif // EIGEN_USE_GPU and EIGEN_GPUCC
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_GPU_H
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceSycl.h://get_devices returns all the available opencl devices. Either use device_selector or exclude devices that computecpp does not support (AMD OpenCL for CPU  and intel GPU)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceSycl.h:    (device.is_gpu() && platform_name.find("intel")!=std::string::npos);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceSycl.h:  /// buffer with map_allocator on the gpu in parallel. At the end of the function call the destination buffer would be destroyed and the data
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceSycl.h:    // OpenCL doesn't have such concept
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceSycl.h:    // OpenCL doesn't have such concept
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#if defined(EIGEN_CXX11_TENSOR_GPU_HIP_CUDA_DEFINES_H)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuStream_t
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuDeviceProp_t 
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuError_t
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuSuccess
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuErrorNotReady
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuGetDeviceCount
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuGetErrorString
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuGetDeviceProperties
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuStreamDefault
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuGetDevice
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuSetDevice
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMalloc
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuFree
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMemsetAsync
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMemcpyAsync
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMemcpyDeviceToDevice
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMemcpyDeviceToHost
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMemcpyHostToDevice
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuStreamQuery
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuSharedMemConfig
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuDeviceSetSharedMemConfig
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuStreamSynchronize
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuDeviceSynchronize
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMemcpy
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef EIGEN_CXX11_TENSOR_GPU_HIP_CUDA_DEFINES_H
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#endif // EIGEN_CXX11_TENSOR_GPU_HIP_CUDA_DEFINES_H
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_GPU_H
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_GPU_H
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#if defined(EIGEN_USE_GPU) && defined(EIGEN_GPUCC)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:// Full reducers for GPU, don't vectorize for now
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:// Reducer function that enables multiple gpu thread to safely accumulate at the same
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:// attempts to update it with the new value. If in the meantime another gpu thread
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#if (defined(EIGEN_HIP_DEVICE_COMPILE) && defined(__HIP_ARCH_HAS_WARP_SHUFFLE__)) || (EIGEN_CUDA_ARCH >= 300)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(0 && "Wordsize not supported");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#else // EIGEN_CUDA_ARCH >= 300
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  gpu_assert(0 && "Shouldn't be called on unsupported device");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_CUDA_ARCH >= 300
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#if (defined(EIGEN_HIP_DEVICE_COMPILE) && defined(__HIP_ARCH_HAS_WARP_SHUFFLE__)) || (EIGEN_CUDA_ARCH >= 300)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#else // EIGEN_CUDA_ARCH >= 300
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  gpu_assert(0 && "Shouldn't be called on unsupported device");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_CUDA_ARCH >= 300
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#if (defined(EIGEN_HIP_DEVICE_COMPILE) && defined(__HIP_ARCH_HAS_WARP_SHUFFLE__)) || (EIGEN_CUDA_ARCH >= 300)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  #elif defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER < 90000
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#else // EIGEN_CUDA_ARCH >= 300
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  gpu_assert(0 && "Shouldn't be called on unsupported device");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_CUDA_ARCH >= 300
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  #elif defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER < 90000
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static void run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(false && "Should only be called on doubles, floats and half floats");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    LAUNCH_GPU_KERNEL((FullReductionKernel<block_size, num_per_thread, Self, Op, Index>),
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static void run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(false && "Should not be called since there is no packet accessor");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      LAUNCH_GPU_KERNEL((ReductionInitFullReduxKernelHalfFloat<Self, Op, Index>),
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    LAUNCH_GPU_KERNEL((FullReductionKernelHalfFloat<block_size, num_per_thread, Self, Op, Index>),
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      LAUNCH_GPU_KERNEL((ReductionCleanupKernelHalfFloat<Op>),
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:struct FullReducer<Self, Op, GpuDevice, Vectorizable> {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#else // EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(HasOptimizedImplementation && "Should only be called on doubles, floats or half floats");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    FullReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#if (defined(EIGEN_HIP_DEVICE_COMPILE) && defined(__HIP_ARCH_HAS_WARP_SHUFFLE__)) || (EIGEN_CUDA_ARCH >= 300)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      #elif defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER < 90000
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#else // EIGEN_CUDA_ARCH >= 300
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  gpu_assert(0 && "Shouldn't be called on unsupported device");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_CUDA_ARCH >= 300
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      #elif defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER < 90000
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static EIGEN_DEVICE_FUNC bool run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index, typename Self::Index) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(false && "Should only be called to reduce doubles, floats and half floats on a gpu device");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    const int max_blocks = device.getNumGpuMultiProcessors() *
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:                           device.maxGpuThreadsPerMultiProcessor() / block_size;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      const int max_blocks = device.getNumGpuMultiProcessors() *
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:                           device.maxGpuThreadsPerMultiProcessor() / 1024;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      LAUNCH_GPU_KERNEL((ReductionInitKernel<OutputType, Index>),
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    LAUNCH_GPU_KERNEL((InnerReductionKernel<num_per_thread, Self, Op, Index>),
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static bool run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index, typename Self::Index) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(false && "Should not be called since there is no packet accessor");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    const int max_blocks = device.getNumGpuMultiProcessors() *
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:                           device.maxGpuThreadsPerMultiProcessor() / block_size;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      const int max_blocks = device.getNumGpuMultiProcessors() *
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:                           device.maxGpuThreadsPerMultiProcessor() / 1024;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      LAUNCH_GPU_KERNEL((ReductionInitKernelHalfFloat<Self, Op, Index>),
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    LAUNCH_GPU_KERNEL((InnerReductionKernelHalfFloat<num_per_thread, Self, Op, Index>),
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:struct InnerReducer<Self, Op, GpuDevice> {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#else // EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(HasOptimizedImplementation && "Should only be called on doubles, floats or half floats");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    return InnerReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs_to_reduce, num_preserved_vals);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:struct OuterReducer<Self, Op, GpuDevice> {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    //          (in the cxx11_tensor_reduction_gpu test)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(false && "Should only be called to reduce doubles or floats on a gpu device");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, float* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    const int max_blocks = device.getNumGpuMultiProcessors() *
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:                           device.maxGpuThreadsPerMultiProcessor() / block_size;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      const int max_blocks = device.getNumGpuMultiProcessors() *
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:                             device.maxGpuThreadsPerMultiProcessor() / 1024;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      LAUNCH_GPU_KERNEL((ReductionInitKernel<float, Index>),
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    LAUNCH_GPU_KERNEL((OuterReductionKernel<num_per_thread, Self, Op, Index>),
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // defined(EIGEN_USE_GPU) && defined(EIGEN_GPUCC)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_GPU_H
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDevice.h:  * on the specified computing 'device' (GPU, thread pool, ...)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDevice.h:  *    C.device(EIGEN_GPU) = A + B;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#warning "Deprecated header file, please either include the main Eigen/CXX11/Tensor header or the respective TensorReductionGpu.h file"
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#include "TensorReductionGpu.h"
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:GPU using cuda.  Additional implementations may be added later.
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:GpuDevice.
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:#### Evaluating On GPU
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:You need to create a GPU device but you also need to explicitly allocate the
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:memory for tensors with cuda.
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:*   On GPUs only floating point values are properly tested and optimized for.
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:*   Complex and integer values are known to be broken on GPUs. If you try to use
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:// It is very expensive to start the memcpy kernel on GPU: we therefore only
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:#ifdef EIGEN_USE_GPU
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:template <typename Index> struct MemcpyTriggerForSlicing<Index, GpuDevice>  {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:  EIGEN_DEVICE_FUNC MemcpyTriggerForSlicing(const GpuDevice&) { }
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:// It is very expensive to start the memcpy kernel on GPU: we therefore only
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:// For CUDA packet types when using a GpuDevice
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:#if defined(EIGEN_USE_GPU) && defined(EIGEN_HAS_GPU_FP16)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:struct PacketType<half, GpuDevice> {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:// Can't use std::pairs on cuda devices
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorSycl.h:// this is used to change the address space type in tensor map for GPU
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorFFT.h:  * Improve the performance on GPU
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:struct GpuDevice;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:struct IsVectorizable<GpuDevice, Expression> {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:  static const bool value = TensorEvaluator<Expression, GpuDevice>::PacketAccess &&
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:                            TensorEvaluator<Expression, GpuDevice>::IsAligned;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#warning "Deprecated header file, please either include the main Eigen/CXX11/Tensor header or the respective TensorContractionGpu.h file"
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#include "TensorContractionGpu.h"
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMacros.h:#if !defined(EIGEN_GPUCC)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#if defined(EIGEN_USE_GPU) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_GPU_H)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_GPU_H
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:// This header file container defines fo gpu* macros which will resolve to
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:// their equivalent hip* or cuda* versions depending on the compiler in use
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#include "TensorGpuHipCudaDefines.h"
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:static const int kGpuScratchSize = 1024;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:// This defines an interface that GPUDevice can take to use
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:// HIP / CUDA streams underneath.
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  virtual const gpuStream_t& stream() const = 0;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  virtual const gpuDeviceProp_t& deviceProperties() const = 0;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:static gpuDeviceProp_t* m_deviceProperties;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpuError_t status = gpuGetDeviceCount(&num_devices);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      if (status != gpuSuccess) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:        std::cerr << "Failed to get the number of GPU devices: "
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:                  << gpuGetErrorString(status)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:        gpu_assert(status == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      m_deviceProperties = new gpuDeviceProp_t[num_devices];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:        status = gpuGetDeviceProperties(&m_deviceProperties[i], i);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:        if (status != gpuSuccess) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:          std::cerr << "Failed to initialize GPU device #"
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:                    << gpuGetErrorString(status)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:          gpu_assert(status == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:static const gpuStream_t default_stream = gpuStreamDefault;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:class GpuStreamDevice : public StreamInterface {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  GpuStreamDevice() : stream_(&default_stream), scratch_(NULL), semaphore_(NULL) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuGetDevice(&device_);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  GpuStreamDevice(int device) : stream_(&default_stream), device_(device), scratch_(NULL), semaphore_(NULL) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  // assumes that the stream is associated to the current gpu device.
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  GpuStreamDevice(const gpuStream_t* stream, int device = -1)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpuGetDevice(&device_);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpuError_t err = gpuGetDeviceCount(&num_devices);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpu_assert(err == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpu_assert(device < num_devices);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  virtual ~GpuStreamDevice() {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  const gpuStream_t& stream() const { return *stream_; }
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  const gpuDeviceProp_t& deviceProperties() const {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err = gpuSetDevice(device_);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    err = gpuMalloc(&result, num_bytes);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(result != NULL);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err = gpuSetDevice(device_);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(buffer != NULL);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    err = gpuFree(buffer);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      scratch_ = allocate(kGpuScratchSize + sizeof(unsigned int));
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      char* scratch = static_cast<char*>(scratchpad()) + kGpuScratchSize;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpuError_t err = gpuMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpu_assert(err == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  const gpuStream_t* stream_;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:struct GpuDevice {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  explicit GpuDevice(const StreamInterface* stream) : stream_(stream), max_blocks_(INT_MAX) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  explicit GpuDevice(const StreamInterface* stream, int num_blocks) : stream_(stream), max_blocks_(num_blocks) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  EIGEN_STRONG_INLINE const gpuStream_t& stream() const {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#ifndef EIGEN_GPU_COMPILE_PHASE
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err = gpuMemcpyAsync(dst, src, n, gpuMemcpyDeviceToDevice,
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err =
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:        gpuMemcpyAsync(dst, src, n, gpuMemcpyHostToDevice, stream_->stream());
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err =
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:        gpuMemcpyAsync(dst, src, n, gpuMemcpyDeviceToHost, stream_->stream());
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#ifndef EIGEN_GPU_COMPILE_PHASE
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err = gpuMemsetAsync(buffer, c, n, stream_->stream());
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    // there is no l3 cache on hip/cuda devices.
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#if defined(EIGEN_GPUCC) && !defined(EIGEN_GPU_COMPILE_PHASE)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err = gpuStreamSynchronize(stream_->stream());
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    if (err != gpuSuccess) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      std::cerr << "Error detected in GPU stream: "
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:                << gpuGetErrorString(err)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpu_assert(err == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(false && "The default device should be used instead to generate kernel code");
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  EIGEN_STRONG_INLINE int getNumGpuMultiProcessors() const {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  EIGEN_STRONG_INLINE int maxGpuThreadsPerBlock() const {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  EIGEN_STRONG_INLINE int maxGpuThreadsPerMultiProcessor() const {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  // This function checks if the GPU runtime recorded an error for the
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#ifdef EIGEN_GPUCC
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t error = gpuStreamQuery(stream_->stream());
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    return (error == gpuSuccess) || (error == gpuErrorNotReady);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#define LAUNCH_GPU_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)             \
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  gpu_assert(hipGetLastError() == hipSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#define LAUNCH_GPU_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)             \
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  gpu_assert(cudaGetLastError() == cudaSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#ifdef EIGEN_GPUCC
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:static EIGEN_DEVICE_FUNC inline void setGpuSharedMemConfig(gpuSharedMemConfig config) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#ifndef EIGEN_GPU_COMPILE_PHASE
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  gpuError_t status = gpuDeviceSetSharedMemConfig(config);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  gpu_assert(status == gpuSuccess);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:// undefine all the gpu* macros we defined at the beginning of the file
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#include "TensorGpuHipCudaUndefines.h"
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_GPU_H
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionSycl.h:    /// This one is used to collect all the reduced value of shared memory as we don't have global barrier on GPU. Once it is saved we can
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionSycl.h:    /// This one is used to collect all the reduced value of shared memory as we don't have global barrier on GPU. Once it is saved we can
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:// clang is incompatible with the CUDA syntax wrt making a kernel a class friend,
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(__clang__) && (defined(__CUDA__) || defined(__HIP__))
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && (defined(EIGEN_GPUCC))
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_HAS_GPU_FP16)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:        ((RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) ||
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:         !RunningOnGPU))) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    else if (RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    if ((RunningOnSycl || RunningFullReduction || RunningOnGPU) && m_result) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    if (RunningOnGPU && m_result) {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && (defined(EIGEN_GPUCC))
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_HAS_GPU_FP16)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && (defined(EIGEN_GPUCC))
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:  static const bool RunningOnGPU = internal::is_same<Device, Eigen::GpuDevice>::value;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:static const bool RunningOnGPU = false;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:  static const bool RunningOnGPU = false;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:    const size_t plane_tensor_offset =indexMapper.mapCudaInputPlaneToTensorInputOffset(itemID.get_global(1));
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:      const size_t tensor_index  =  plane_tensor_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i + first_input_start);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:      const size_t tensor_index = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(itemID.get_global(1))
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:      +indexMapper.mapCudaOutputKernelToTensorOutputOffset(itemID.get_local(0) + first_output_start);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:    const size_t plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(itemID.get_global(2));
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:        const size_t tensor_index  = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i + first_x_input_start, j+ first_y_input_start );
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:      const size_t tensor_index = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(itemID.get_global(2))
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:      +indexMapper.mapCudaOutputKernelToTensorOutputOffset(itemID.get_local(0) + fitst_x_output_start, itemID.get_local(1) + fitst_y_output_start);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:      const size_t plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:            const size_t tensor_index  = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i + first_x_input_start, j+ first_y_input_start , k+ first_z_input_start );
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:        const size_t tensor_index = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:        +indexMapper.mapCudaOutputKernelToTensorOutputOffset(itemID.get_local(0) + fitst_x_output_start, itemID.get_local(1) + fitst_y_output_start, itemID.get_local(2) + fitst_z_output_start );
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:          gpu_assert(static_cast<unsigned long>(shared_mem) <= m_device.sharedMemPerBlock());
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:          gpu_assert(static_cast<unsigned long>(shared_mem) <= m_device.sharedMemPerBlock());
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:          gpu_assert(static_cast<unsigned long>(shared_mem) <= m_device.sharedMemPerBlock());
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#if defined(EIGEN_USE_GPU) && !defined(EIGEN_CXX11_TENSOR_GPU_HIP_CUDA_DEFINES_H)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define EIGEN_CXX11_TENSOR_GPU_HIP_CUDA_DEFINES_H
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:// There is code in the Tensorflow codebase that will define EIGEN_USE_GPU,  but
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:// for some reason gets sent to the gcc/host compiler instead of the gpu/nvcc/hipcc compiler
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:// When compiling such files, gcc will end up trying to pick up the CUDA headers by 
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:// default (see the code within "unsupported/Eigen/CXX11/Tensor" that is guarded by EIGEN_USE_GPU)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:// This will obsviously not work when trying to compile tensorflow on a system with no CUDA
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStream_t hipStream_t
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuDeviceProp_t hipDeviceProp_t
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuError_t hipError_t
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuSuccess hipSuccess
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuErrorNotReady hipErrorNotReady
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetDeviceCount hipGetDeviceCount
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetErrorString hipGetErrorString
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetDeviceProperties hipGetDeviceProperties
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStreamDefault hipStreamDefault
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetDevice hipGetDevice
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuSetDevice hipSetDevice
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMalloc hipMalloc
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuFree hipFree
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemsetAsync hipMemsetAsync
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyAsync hipMemcpyAsync
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStreamQuery hipStreamQuery
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuSharedMemConfig hipSharedMemConfig
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuDeviceSetSharedMemConfig hipDeviceSetSharedMemConfig
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStreamSynchronize hipStreamSynchronize
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuDeviceSynchronize hipDeviceSynchronize
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpy hipMemcpy
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStream_t cudaStream_t
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuDeviceProp_t cudaDeviceProp
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuError_t cudaError_t
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuSuccess cudaSuccess
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuErrorNotReady cudaErrorNotReady
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetDeviceCount cudaGetDeviceCount
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetErrorString cudaGetErrorString
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetDeviceProperties cudaGetDeviceProperties
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStreamDefault cudaStreamDefault
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetDevice cudaGetDevice
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuSetDevice cudaSetDevice
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMalloc cudaMalloc
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuFree cudaFree
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemsetAsync cudaMemsetAsync
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyAsync cudaMemcpyAsync
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStreamQuery cudaStreamQuery
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuSharedMemConfig cudaSharedMemConfig
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuDeviceSetSharedMemConfig cudaDeviceSetSharedMemConfig
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStreamSynchronize cudaStreamSynchronize
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuDeviceSynchronize cudaDeviceSynchronize
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpy cudaMemcpy
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#if defined(EIGEN_HIP_DEVICE_COMPILE) || (defined(EIGEN_CUDACC) && (EIGEN_CUDACC_VER==0))
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:// clang-cuda and HIPCC do not support the use of assert on the GPU side.
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpu_assert(COND)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpu_assert(COND) assert(COND)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#endif  // EIGEN_CXX11_TENSOR_GPU_HIP_CUDA_DEFINES_H
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#warning "Deprecated header file, please either include the main Eigen/CXX11/Tensor header or the respective TensorDeviceGpu.h file"
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#include "TensorDeviceGpu.h"
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    array<Index, NumDims> gpuInputDimensions;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    array<Index, NumDims> gpuOutputDimensions;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      gpuInputDimensions[index] = input_dims[indices[i]];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      gpuOutputDimensions[index] = dimensions[indices[i]];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        gpuInputDimensions[written] = input_dims[i];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        gpuOutputDimensions[written] = dimensions[i];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuInputStrides[i] =
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_gpuInputStrides[i - 1] * gpuInputDimensions[i - 1];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuOutputStrides[i] =
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_gpuOutputStrides[i - 1] * gpuOutputDimensions[i - 1];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuInputStrides[i] = 1;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuOutputStrides[i] = 1;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuInputStrides[i] =
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_gpuInputStrides[i + 1] * gpuInputDimensions[i + 1];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuOutputStrides[i] =
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_gpuOutputStrides[i + 1] * gpuOutputDimensions[i + 1];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuInputStrides[i] = 1;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuOutputStrides[i] = 1;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuInputPlaneToTensorInputOffset(Index p) const {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_gpuInputStrides[d];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_gpuInputStrides[d];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_gpuInputStrides[d];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_gpuInputStrides[d];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuOutputPlaneToTensorOutputOffset(Index p) const {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_gpuOutputStrides[d];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_gpuOutputStrides[d];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_gpuOutputStrides[d];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_gpuOutputStrides[d];
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuInputKernelToTensorInputOffset(Index i) const {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuOutputKernelToTensorOutputOffset(Index i) const {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuInputKernelToTensorInputOffset(Index i, Index j) const {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuOutputKernelToTensorOutputOffset(Index i, Index j) const {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuInputKernelToTensorInputOffset(Index i, Index j, Index k) const {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuOutputKernelToTensorOutputOffset(Index i, Index j, Index k) const {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  array<Index, NumDims> m_gpuInputStrides;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  array<Index, NumDims> m_gpuOutputStrides;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:// Use an optimized implementation of the evaluation code for GPUs whenever possible.
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:#if defined(EIGEN_USE_GPU) && defined(EIGEN_GPUCC)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapGpuInputPlaneToTensorInputOffset(p);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const int tensor_index = plane_input_offset + indexMapper.mapGpuInputKernelToTensorInputOffset(i+first_x);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapGpuOutputPlaneToTensorOutputOffset(p);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const int tensor_index = plane_output_offset + indexMapper.mapGpuOutputKernelToTensorOutputOffset(i+first_x);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapGpuInputPlaneToTensorInputOffset(p);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const int tensor_index = plane_input_offset + indexMapper.mapGpuInputKernelToTensorInputOffset(i+first_x, j+first_y);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapGpuOutputPlaneToTensorOutputOffset(p);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const int tensor_index = plane_output_offset + indexMapper.mapGpuOutputKernelToTensorOutputOffset(i+first_x, j+first_y);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapGpuInputPlaneToTensorInputOffset(p);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          const int tensor_index = plane_input_offset + indexMapper.mapGpuInputKernelToTensorInputOffset(i+first_x, j+first_y, k+first_z);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapGpuOutputPlaneToTensorOutputOffset(p);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          const int tensor_index = plane_output_offset + indexMapper.mapGpuOutputKernelToTensorOutputOffset(i+first_x, j+first_y, k+first_z);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:struct TensorEvaluator<const TensorConvolutionOp<Indices, InputArgType, KernelArgType>, GpuDevice>
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  static const int NumDims =  internal::array_size<typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions>::value;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  typedef typename TensorEvaluator<KernelArgType, GpuDevice>::Dimensions KernelDimensions;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    IsAligned = TensorEvaluator<InputArgType, GpuDevice>::IsAligned & TensorEvaluator<KernelArgType, GpuDevice>::IsAligned,
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    Layout = TensorEvaluator<InputArgType, GpuDevice>::Layout,
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const GpuDevice& device)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<InputArgType, GpuDevice>::Layout) == static_cast<int>(TensorEvaluator<KernelArgType, GpuDevice>::Layout)), YOU_MADE_A_PROGRAMMING_MISTAKE);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions& input_dims = m_inputImpl.dimensions();
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const typename TensorEvaluator<KernelArgType, GpuDevice>::Dimensions& kernel_dims = m_kernelImpl.dimensions();
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  typedef typename PacketType<CoeffReturnType, GpuDevice>::type PacketReturnType;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const bool PacketAccess = internal::IsVectorizable<GpuDevice, KernelArgType>::value;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      internal::TensorExecutor<const EvalTo, GpuDevice, PacketAccess>::run(evalToTmp, m_device);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    typedef typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions InputDims;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int maxThreadsPerBlock = m_device.maxGpuThreadsPerBlock();
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int maxBlocksPerProcessor = m_device.maxGpuThreadsPerMultiProcessor() / maxThreadsPerBlock;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int numMultiProcessors = m_device.getNumGpuMultiProcessors();
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        gpu_assert(shared_mem <= maxSharedMem);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_GPU_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, 4, data);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_GPU_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, 7, data);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_GPU_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, kernel_size, data);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        gpu_assert(shared_mem <= maxSharedMem);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_GPU_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4, 7>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 4, 7, data);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_GPU_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 4, kernel_size_y, data);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_GPU_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7, 4>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 7, 4, data);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_GPU_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 7, kernel_size_y, data);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_GPU_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, Dynamic, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, kernel_size_x, kernel_size_y, data);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        gpu_assert(shared_mem <= maxSharedMem);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        LAUNCH_GPU_KERNEL((EigenConvolutionKernel3D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, numZ, maxZ, kernel_size_x, kernel_size_y, kernel_size_z, data);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  TensorEvaluator<InputArgType, GpuDevice> m_inputImpl;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  TensorEvaluator<KernelArgType, GpuDevice> m_kernelImpl;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  const GpuDevice& m_device;
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorBroadcasting.h:        #ifdef EIGEN_GPU_COMPILE_PHASE
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorBroadcasting.h:        // See PR 437: on NVIDIA P100 and K20m we observed a x3-4 speed up by enforcing
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorBroadcasting.h:        #ifdef EIGEN_GPU_COMPILE_PHASE
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:#if defined(EIGEN_USE_GPU) && (defined(EIGEN_GPUCC))
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:// GPU implementation of scan
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:struct ScanLauncher<Self, Reducer, GpuDevice> {
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:     LAUNCH_GPU_KERNEL((ScanKernel<Self, Reducer>), num_blocks, block_size, 0, self.device(), self, total_size, data);
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:#endif  // EIGEN_USE_GPU && (EIGEN_GPUCC)
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:// Use the texture cache on CUDA devices whenever possible
external-libraries/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
external-libraries/eigen/unsupported/Eigen/CXX11/src/util/EmulateArray.h:// Moreover, CUDA doesn't support the STL containers, so we use our own instead.
external-libraries/eigen/unsupported/Eigen/CXX11/src/util/EmulateArray.h:#if (__cplusplus <= 199711L && EIGEN_COMP_MSVC < 1900) || defined(EIGEN_GPUCC) || defined(EIGEN_AVOID_STL_ARRAY)
external-libraries/eigen/unsupported/Eigen/CXX11/src/util/EmulateArray.h:// The compiler supports c++11, and we're not targeting cuda: use std::array as Eigen::array
external-libraries/eigen/unsupported/Eigen/CXX11/src/util/MaxSizeVector.h:  * std::vector is not an option (e.g. on GPU or when compiling using
external-libraries/eigen/unsupported/Eigen/SpecialFunctions:#if defined EIGEN_VECTORIZE_GPU
external-libraries/eigen/unsupported/Eigen/SpecialFunctions:  #include "src/SpecialFunctions/arch/GPU/GpuSpecialFunctions.h"
external-libraries/eigen/unsupported/Eigen/OpenGLSupport:#ifdef GL_ARB_gpu_shader_fp64
external-libraries/eigen/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__) 
external-libraries/eigen/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__) 
external-libraries/eigen/unsupported/Eigen/src/SpecialFunctions/arch/GPU/GpuSpecialFunctions.h:#ifndef EIGEN_GPU_SPECIALFUNCTIONS_H
external-libraries/eigen/unsupported/Eigen/src/SpecialFunctions/arch/GPU/GpuSpecialFunctions.h:#define EIGEN_GPU_SPECIALFUNCTIONS_H
external-libraries/eigen/unsupported/Eigen/src/SpecialFunctions/arch/GPU/GpuSpecialFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
external-libraries/eigen/unsupported/Eigen/src/SpecialFunctions/arch/GPU/GpuSpecialFunctions.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
external-libraries/eigen/unsupported/Eigen/src/SpecialFunctions/arch/GPU/GpuSpecialFunctions.h:#endif // EIGEN_GPU_SPECIALFUNCTIONS_H
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:#define EIGEN_USE_GPU
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:#include <cuda.h>
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:#include <cuda_runtime.h>
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:#define BM_FuncGPU(FUNC)                                                       \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, float> suite(device, N);                  \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    cudaDeviceSynchronize();                                                   \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(memcpy);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(typeCasting);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(random);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(slicing);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(rowChip);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(colChip);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(shuffling);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(padding);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(striding);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(broadcasting);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(coeffWiseOp);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(algebraicFunc);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(transcendentalFunc);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(rowReduction);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(colReduction);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(fullReduction);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:#define BM_FuncWithInputDimsGPU(FUNC, D1, D2, D3)                              \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, float> suite(device, D1, D2, D3);         \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    cudaDeviceSynchronize();                                                   \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, N, N);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithInputDimsGPU(contraction, 64, N, N);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, 64, N);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, N, 64);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:#define BM_FuncWithKernelDimsGPU(FUNC, DIM1, DIM2)                             \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, float> suite(device, N);                  \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    cudaDeviceSynchronize();                                                   \
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 1);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 1, 7);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 4);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 4, 7);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 64);
external-libraries/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 64, 7);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:#define BM_FuncGPU(FUNC)                                                       \
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:    cl::sycl::gpu_selector selector;                                           \
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(memcpy);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(typeCasting);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(slicing);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(rowChip);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(colChip);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(shuffling);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(padding);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(striding);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(broadcasting);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(coeffWiseOp);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(algebraicFunc);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(transcendentalFunc);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(rowReduction);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(colReduction);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(fullReduction);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:#define BM_FuncWithInputDimsGPU(FUNC, D1, D2, D3)                              \
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:    cl::sycl::gpu_selector selector;                                           \
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithInputDimsGPU(contraction, N, N, N);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithInputDimsGPU(contraction, 64, N, N);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithInputDimsGPU(contraction, N, 64, N);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithInputDimsGPU(contraction, N, N, 64);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:#define BM_FuncWithKernelDimsGPU(FUNC, DIM1, DIM2)                             \
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:    cl::sycl::gpu_selector selector;                                           \
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithKernelDimsGPU(convolution, 7, 1);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithKernelDimsGPU(convolution, 1, 7);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithKernelDimsGPU(convolution, 7, 4);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithKernelDimsGPU(convolution, 4, 7);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithKernelDimsGPU(convolution, 7, 64);
external-libraries/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithKernelDimsGPU(convolution, 64, 7);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#define EIGEN_USE_GPU
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#include <cuda.h>
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#include <cuda_runtime.h>
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#define BM_FuncGPU(FUNC)                                                       \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, Eigen::half> suite(device, N);            \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    cudaDeviceSynchronize();                                                   \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(memcpy);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(typeCasting);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu://BM_FuncGPU(random);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(slicing);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(rowChip);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(colChip);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(shuffling);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(padding);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(striding);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(broadcasting);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(coeffWiseOp);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(algebraicFunc);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(transcendentalFunc);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(rowReduction);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(colReduction);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(fullReduction);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#define BM_FuncWithInputDimsGPU(FUNC, D1, D2, D3)                              \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, Eigen::half> suite(device, D1, D2, D3);   \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    cudaDeviceSynchronize();                                                   \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, N, N);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithInputDimsGPU(contraction, 64, N, N);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, 64, N);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, N, 64);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#define BM_FuncWithKernelDimsGPU(FUNC, DIM1, DIM2)                             \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, Eigen::half> suite(device, N);            \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    cudaDeviceSynchronize();                                                   \
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 1);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 1, 7);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 4);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 4, 7);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 64);
external-libraries/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 64, 7);
external-libraries/eigen/bench/tensors/tensor_benchmarks.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
external-libraries/eigen/bench/tensors/tensor_benchmarks.h:    if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
external-libraries/eigen/bench/tensors/README:The first part is a generic suite, in which each benchmark comes in 2 flavors: one that runs on CPU, and one that runs on GPU.
external-libraries/eigen/bench/tensors/README:To compile the floating point GPU benchmarks, simply call:
external-libraries/eigen/bench/tensors/README:nvcc tensor_benchmarks_gpu.cu benchmark_main.cc -I ../../ -std=c++11 -O2 -DNDEBUG -use_fast_math -ftz=true -arch compute_35 -o benchmarks_gpu
external-libraries/eigen/bench/tensors/README:We also provide a version of the generic GPU tensor benchmarks that uses half floats (aka fp16) instead of regular floats. To compile these benchmarks, simply call the command line below. You'll need a recent GPU that supports compute capability 5.3 or higher to run them and nvcc 7.5 or higher to compile the code.
external-libraries/eigen/bench/tensors/README:nvcc tensor_benchmarks_fp16_gpu.cu benchmark_main.cc -I ../../ -std=c++11 -O2 -DNDEBUG -use_fast_math -ftz=true -arch compute_53 -o benchmarks_fp16_gpu
external-libraries/eigen/bench/tensors/README:clang++ -O3 tensor_benchmarks_sycl_include_headers.cc -pthread -I ../../ -I  {ComputeCpp_ROOT}/include/ -L  {ComputeCpp_ROOT}/lib/ -lComputeCpp -lOpenCL -D_GLIBCXX_USE_CXX11_ABI=0 -DEIGEN_USE_SYCL=1 -std=c++11 benchmark_main.o -o tensor_benchmark_sycl
external-libraries/eigen/cmake/FindPastix.cmake:#   - STARPU_CUDA: to activate detection of StarPU with CUDA
external-libraries/eigen/cmake/FindPastix.cmake:set(PASTIX_LOOK_FOR_STARPU_CUDA OFF)
external-libraries/eigen/cmake/FindPastix.cmake:    if (${component} STREQUAL "STARPU_CUDA")
external-libraries/eigen/cmake/FindPastix.cmake:      # means we look for PaStiX with StarPU + CUDA
external-libraries/eigen/cmake/FindPastix.cmake:      set(PASTIX_LOOK_FOR_STARPU_CUDA ON)
external-libraries/eigen/cmake/FindPastix.cmake:  if (PASTIX_LOOK_FOR_STARPU_CUDA)
external-libraries/eigen/cmake/FindPastix.cmake:    list(APPEND STARPU_COMPONENT_LIST "CUDA")
external-libraries/eigen/cmake/FindPastix.cmake:  # CUDA
external-libraries/eigen/cmake/FindPastix.cmake:  if (PASTIX_LOOK_FOR_STARPU_CUDA AND CUDA_FOUND)
external-libraries/eigen/cmake/FindPastix.cmake:    if (CUDA_INCLUDE_DIRS)
external-libraries/eigen/cmake/FindPastix.cmake:      list(APPEND REQUIRED_INCDIRS "${CUDA_INCLUDE_DIRS}")
external-libraries/eigen/cmake/FindPastix.cmake:    foreach(libdir ${CUDA_LIBRARY_DIRS})
external-libraries/eigen/cmake/FindPastix.cmake:    list(APPEND REQUIRED_LIBS "${CUDA_CUBLAS_LIBRARIES};${CUDA_LIBRARIES}")
external-libraries/eigen/cmake/FindPastix.cmake:	"Have you tried with COMPONENTS (MPI/SEQ, STARPU, STARPU_CUDA, SCOTCH, PTSCOTCH, METIS)? "
external-libraries/eigen/cmake/EigenTesting.cmake:    elseif(EIGEN_TEST_CUDA_CLANG)
external-libraries/eigen/cmake/EigenTesting.cmake:      if(CUDA_64_BIT_DEVICE_CODE AND (EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64"))
external-libraries/eigen/cmake/EigenTesting.cmake:        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
external-libraries/eigen/cmake/EigenTesting.cmake:        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib")
external-libraries/eigen/cmake/EigenTesting.cmake:      set(CUDA_CLANG_LINK_LIBRARIES "cudart_static" "cuda" "dl" "pthread")
external-libraries/eigen/cmake/EigenTesting.cmake:      set(CUDA_CLANG_LINK_LIBRARIES ${CUDA_CLANG_LINK_LIBRARIES} "rt")
external-libraries/eigen/cmake/EigenTesting.cmake:      target_link_libraries(${targetname} ${CUDA_CLANG_LINK_LIBRARIES})
external-libraries/eigen/cmake/EigenTesting.cmake:        cuda_add_executable(${targetname} ${filename} OPTIONS ${ARGV2})
external-libraries/eigen/cmake/EigenTesting.cmake:        cuda_add_executable(${targetname} ${filename})
external-libraries/eigen/cmake/EigenTesting.cmake:    if(EIGEN_TEST_CUDA)
external-libraries/eigen/cmake/EigenTesting.cmake:      if(EIGEN_TEST_CUDA_CLANG)
external-libraries/eigen/cmake/EigenTesting.cmake:        message(STATUS "CUDA:              ON (using clang)")
external-libraries/eigen/cmake/EigenTesting.cmake:        message(STATUS "CUDA:              ON (using nvcc)")
external-libraries/eigen/cmake/EigenTesting.cmake:      message(STATUS "CUDA:              OFF")
external-libraries/eigen/cmake/FindBLAS.cmake:##  ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic
external-libraries/eigen/cmake/FindBLAS.cmake:      ((BLA_VENDOR STREQUAL "ACML_GPU") AND (NOT BLAS_ACML_GPU_LIB_DIRS)))
external-libraries/eigen/cmake/FindBLAS.cmake:      file( GLOB _ACML_GPU_ROOT "C:/AMD/acml*/GPGPUexamples" )
external-libraries/eigen/cmake/FindBLAS.cmake:      file( GLOB _ACML_GPU_ROOT "/opt/acml*/GPGPUexamples" )
external-libraries/eigen/cmake/FindBLAS.cmake:    list(GET _ACML_GPU_ROOT 0 _ACML_GPU_ROOT)
external-libraries/eigen/cmake/FindBLAS.cmake:  elseif( BLA_VENDOR STREQUAL "ACML_GPU" )
external-libraries/eigen/cmake/FindBLAS.cmake:    foreach( BLAS_ACML_GPU_LIB_DIRS ${_ACML_GPU_LIB_DIRS})
external-libraries/eigen/cmake/FindBLAS.cmake:	"" "acml;acml_mv;CALBLAS" "" ${BLAS_ACML_GPU_LIB_DIRS}
external-libraries/eigen/cmake/FindBLAS.cmake:	"Intel( older versions of mkl 32 and 64 bit), ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
external-libraries/eigen/cmake/FindBLAS.cmake:	"Intel( older versions of mkl 32 and 64 bit), ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
external-libraries/eigen/cmake/FindBLASEXT.cmake:    ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
external-libraries/eigen/cmake/FindBLASEXT.cmake:    "\n   ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
external-libraries/eigen/cmake/FindComputeCpp.cmake:# Find OpenCL package
external-libraries/eigen/cmake/FindComputeCpp.cmake:find_package(OpenCL REQUIRED)
external-libraries/eigen/cmake/FindComputeCpp.cmake:                        PUBLIC ${OpenCL_LIBRARIES})
external-libraries/eigen/cmake/FindTriSYCL.cmake:option(TRISYCL_OPENCL "triSYCL OpenCL interoperability mode" OFF)
external-libraries/eigen/cmake/FindTriSYCL.cmake:mark_as_advanced(TRISYCL_OPENCL)
external-libraries/eigen/cmake/FindTriSYCL.cmake:# Find OpenCL package
external-libraries/eigen/cmake/FindTriSYCL.cmake:if(TRISYCL_OPENCL)
external-libraries/eigen/cmake/FindTriSYCL.cmake:  find_package(OpenCL REQUIRED)
external-libraries/eigen/cmake/FindTriSYCL.cmake:    $<$<BOOL:${TRISYCL_OPENCL}>:${OpenCL_INCLUDE_DIRS}>
external-libraries/eigen/cmake/FindTriSYCL.cmake:    $<$<BOOL:${TRISYCL_OPENCL}>:${BOOST_COMPUTE_INCPATH}>)
external-libraries/eigen/cmake/FindTriSYCL.cmake:    $<$<BOOL:${TRISYCL_OPENCL}>:${OpenCL_LIBRARIES}>
external-libraries/eigen/cmake/FindTriSYCL.cmake:    $<$<BOOL:${TRISYCL_OPENCL}>:TRISYCL_OPENCL>
external-libraries/eigen/demos/opengl/gpuhelper.cpp:#include "gpuhelper.h"
external-libraries/eigen/demos/opengl/gpuhelper.cpp:GpuHelper gpu;
external-libraries/eigen/demos/opengl/gpuhelper.cpp:GpuHelper::GpuHelper()
external-libraries/eigen/demos/opengl/gpuhelper.cpp:GpuHelper::~GpuHelper()
external-libraries/eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::pushProjectionMode2D(ProjectionMode2D pm)
external-libraries/eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::popProjectionMode2D(void)
external-libraries/eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::drawVector(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect /* = 50.*/)
external-libraries/eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::drawVectorBox(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect)
external-libraries/eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::drawUnitCube(void)
external-libraries/eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::drawUnitSphere(int level)
external-libraries/eigen/demos/opengl/quaternion_demo.h:#include "gpuhelper.h"
external-libraries/eigen/demos/opengl/CMakeLists.txt:  set(quaternion_demo_SRCS  gpuhelper.cpp icosphere.cpp camera.cpp trackball.cpp quaternion_demo.cpp)
external-libraries/eigen/demos/opengl/camera.cpp:#include "gpuhelper.h"
external-libraries/eigen/demos/opengl/camera.cpp:  gpu.loadMatrix(projectionMatrix(),GL_PROJECTION);
external-libraries/eigen/demos/opengl/camera.cpp:  gpu.loadMatrix(viewMatrix().matrix(),GL_MODELVIEW);
external-libraries/eigen/demos/opengl/gpuhelper.h:#ifndef EIGEN_GPUHELPER_H
external-libraries/eigen/demos/opengl/gpuhelper.h:#define EIGEN_GPUHELPER_H
external-libraries/eigen/demos/opengl/gpuhelper.h:class GpuHelper
external-libraries/eigen/demos/opengl/gpuhelper.h:    GpuHelper();
external-libraries/eigen/demos/opengl/gpuhelper.h:    ~GpuHelper();
external-libraries/eigen/demos/opengl/gpuhelper.h:extern GpuHelper gpu;
external-libraries/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::setMatrixTarget(GLenum matrixTarget)
external-libraries/eigen/demos/opengl/gpuhelper.h:void GpuHelper::multMatrix(const Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
external-libraries/eigen/demos/opengl/gpuhelper.h:void GpuHelper::loadMatrix(
external-libraries/eigen/demos/opengl/gpuhelper.h:void GpuHelper::loadMatrix(const Eigen::Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
external-libraries/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::pushMatrix(GLenum matrixTarget)
external-libraries/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::pushMatrix(const Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
external-libraries/eigen/demos/opengl/gpuhelper.h:void GpuHelper::pushMatrix(
external-libraries/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::popMatrix(GLenum matrixTarget)
external-libraries/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, uint nofElement)
external-libraries/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, const std::vector<uint>* pIndexes)
external-libraries/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, uint start, uint end)
external-libraries/eigen/demos/opengl/gpuhelper.h:#endif // EIGEN_GPUHELPER_H
external-libraries/eigen/demos/opengl/quaternion_demo.cpp:        gpu.pushMatrix(GL_MODELVIEW);
external-libraries/eigen/demos/opengl/quaternion_demo.cpp:        gpu.multMatrix(t.matrix(),GL_MODELVIEW);
external-libraries/eigen/demos/opengl/quaternion_demo.cpp:        gpu.popMatrix(GL_MODELVIEW);
external-libraries/eigen/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitX(), Color(1,0,0,1));
external-libraries/eigen/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitY(), Color(0,1,0,1));
external-libraries/eigen/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitZ(), Color(0,0,1,1));
external-libraries/spdlog/include/spdlog/fmt/bundled/format-inl.h:  // Chosen instead of std::abort to satisfy Clang in CUDA mode during device
external-libraries/spdlog/include/spdlog/fmt/bundled/format.h:#  define FMT_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__)
external-libraries/spdlog/include/spdlog/fmt/bundled/format.h:#  define FMT_CUDA_VERSION 0
external-libraries/spdlog/include/spdlog/fmt/bundled/format.h:// EDG based compilers (Intel, NVIDIA, Elbrus, etc), GCC and MSVC support UDLs.
external-libraries/spdlog/CMakeLists.txt:        target_compile_options(spdlog PUBLIC $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/wd4251
external-libraries/sdsl-lite/external/googletest/googlemock/test/gmock-matchers_test.cc:TEST(MatcherInterfaceTest, CanBeImplementedUsingPublishedAPI) {
external-libraries/sdsl-lite/external/googletest/googletest/include/gtest/internal/gtest-port.h:// with a TR1 tuple implementation.  NVIDIA's CUDA NVCC compiler
external-libraries/sdsl-lite/external/googletest/googletest/include/gtest/internal/gtest-port.h:# if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000) \
external-libraries/asio/asio/include/asio/signal_set.hpp: * POSIX allows signals to be blocked using functions such as @c sigprocmask()
external-libraries/asio/asio/include/asio/basic_signal_set.hpp: * POSIX allows signals to be blocked using functions such as @c sigprocmask()
external-libraries/asio/asio/src/doc/reference.qbk:POSIX allows signals to be blocked using functions such as `sigprocmask()` and `pthread_sigmask()`. For signals to be delivered, programs must ensure that any signals registered using [link asio.reference.signal_set `signal_set`] objects are unblocked in at least one thread. 
external-libraries/asio/asio/src/examples/cpp03/ssl/ca.pem:c4MLyUpdAoGBAOxTtGDpeF6U4s+GPuOCzHCwKQyzfOyCL/UTZv1UJX7Kn1FYycJH
external-libraries/asio/asio/src/examples/cpp11/ssl/ca.pem:c4MLyUpdAoGBAOxTtGDpeF6U4s+GPuOCzHCwKQyzfOyCL/UTZv1UJX7Kn1FYycJH
external-libraries/caches/deps/googletest/googlemock/test/gmock-matchers_test.cc:TEST(MatcherInterfaceTest, CanBeImplementedUsingPublishedAPI) {
external-libraries/folly/CMake/GenPkgConfig.cmake:      "<COMPILE_LANG_AND_ID:CUDA,NVIDIA>" "<COMPILE_LANGUAGE:CUDA>"
external-libraries/folly/folly/Memory.h: *      using T = foobar::FooBarAsyncClient;
external-libraries/folly/folly/Memory.h: *      using T = foobar::FooBarAsyncClient;
external-libraries/folly/folly/experimental/coro/test/AsyncPipeTest.cpp:TEST(BoundedAsyncPipeTest, BlockingPublisherCanceledOnDestroy) {
external-libraries/folly/folly/experimental/coro/test/AsyncPipeTest.cpp:TEST(BoundedAsyncPipeTest, BlockingPublisherCancelsWithParent) {
external-libraries/folly/folly/experimental/coro/test/AsyncPipeTest.cpp:TEST(BoundedAsyncPipeTest, ClosingPublisherEndsConsumer) {
external-libraries/folly/folly/experimental/coro/test/AsyncPipeTest.cpp:TEST(BoundedAsyncPipeTest, ClosingPublisherWithException) {
external-libraries/folly/folly/experimental/RelaxedConcurrentPriorityQueue.h:            blockingPushImpl();
external-libraries/folly/folly/experimental/RelaxedConcurrentPriorityQueue.h:          blockingPushImpl();
external-libraries/folly/folly/experimental/RelaxedConcurrentPriorityQueue.h:  void blockingPushImpl() {
external-libraries/folly/folly/experimental/symbolizer/ElfCache.cpp:  sigprocmask(SIG_SETMASK, &newsigs, &oldsigs);
external-libraries/folly/folly/experimental/symbolizer/ElfCache.cpp:  SCOPE_EXIT { sigprocmask(SIG_SETMASK, &oldsigs, nullptr); };
external-libraries/folly/folly/test/SubprocessTestParentDeathHelper.cpp:  CHECK_ERR(sigprocmask(SIG_BLOCK, &sigs, nullptr));
external-libraries/folly/folly/io/coro/test/TransportTest.cpp:TEST_F(TransportTest, AsyncClientAndServer) {
external-libraries/folly/build/fbcode_builder/CMake/FBThriftCppLibrary.cmake:      "${output_dir}/gen-cpp2/${service}AsyncClient.h"
external-libraries/folly/build/fbcode_builder/CMake/FBThriftCppLibrary.cmake:      "${output_dir}/gen-cpp2/${service}AsyncClient.cpp"
external-libraries/jsoncpp/include/json/value.h:// Workaround for bug in the NVIDIAs CUDA 9.1 nvcc compiler
external-libraries/googletest/googlemock/test/gmock-matchers-comparisons_test.cc:TEST(MatcherInterfaceTest, CanBeImplementedUsingPublishedAPI) {
external-libraries/benchmark/docs/user_guide.md:An example use case for this is benchmarking GPU execution (e.g. OpenCL
external-libraries/benchmark/docs/user_guide.md:or CUDA kernels, OpenGL or Vulkan or Direct3D draw calls), which cannot
external-libraries/benchmark/include/benchmark/benchmark.h:  // If a benchmark must measure time manually (e.g. if GPU execution time is

```
