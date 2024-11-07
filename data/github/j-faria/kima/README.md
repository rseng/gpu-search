# https://github.com/j-faria/kima

```console
eigen/test/gpu_common.h:#ifndef EIGEN_TEST_GPU_COMMON_H
eigen/test/gpu_common.h:#define EIGEN_TEST_GPU_COMMON_H
eigen/test/gpu_common.h:  #include <cuda.h>
eigen/test/gpu_common.h:  #include <cuda_runtime.h>
eigen/test/gpu_common.h:  #include <cuda_runtime_api.h>
eigen/test/gpu_common.h:#define EIGEN_USE_GPU
eigen/test/gpu_common.h:#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
eigen/test/gpu_common.h:#if !defined(__CUDACC__) && !defined(__HIPCC__)
eigen/test/gpu_common.h:void run_on_gpu_meta_kernel(const Kernel ker, int n, const Input* in, Output* out)
eigen/test/gpu_common.h:void run_on_gpu(const Kernel& ker, int n, const Input& in, Output& out)
eigen/test/gpu_common.h:  gpuMalloc((void**)(&d_in),  in_bytes);
eigen/test/gpu_common.h:  gpuMalloc((void**)(&d_out), out_bytes);
eigen/test/gpu_common.h:  gpuMemcpy(d_in,  in.data(),  in_bytes,  gpuMemcpyHostToDevice);
eigen/test/gpu_common.h:  gpuMemcpy(d_out, out.data(), out_bytes, gpuMemcpyHostToDevice);
eigen/test/gpu_common.h:  gpuDeviceSynchronize();
eigen/test/gpu_common.h:  hipLaunchKernelGGL(HIP_KERNEL_NAME(run_on_gpu_meta_kernel<Kernel,
eigen/test/gpu_common.h:  run_on_gpu_meta_kernel<<<Grids,Blocks>>>(ker, n, d_in, d_out);
eigen/test/gpu_common.h:  gpuDeviceSynchronize();
eigen/test/gpu_common.h:  gpuMemcpy(const_cast<typename Input::Scalar*>(in.data()),  d_in,  in_bytes,  gpuMemcpyDeviceToHost);
eigen/test/gpu_common.h:  gpuMemcpy(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost);
eigen/test/gpu_common.h:  gpuFree(d_in);
eigen/test/gpu_common.h:  gpuFree(d_out);
eigen/test/gpu_common.h:void run_and_compare_to_gpu(const Kernel& ker, int n, const Input& in, Output& out)
eigen/test/gpu_common.h:  Input  in_ref,  in_gpu;
eigen/test/gpu_common.h:  Output out_ref, out_gpu;
eigen/test/gpu_common.h:  #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
eigen/test/gpu_common.h:  in_ref = in_gpu = in;
eigen/test/gpu_common.h:  out_ref = out_gpu = out;
eigen/test/gpu_common.h:  run_on_gpu(ker, n, in_gpu, out_gpu);
eigen/test/gpu_common.h:  #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
eigen/test/gpu_common.h:  VERIFY_IS_APPROX(in_ref, in_gpu);
eigen/test/gpu_common.h:  VERIFY_IS_APPROX(out_ref, out_gpu);
eigen/test/gpu_common.h:    #if defined(__CUDA_ARCH__)
eigen/test/gpu_common.h:    info[0] = int(__CUDA_ARCH__ +0);
eigen/test/gpu_common.h:void ei_test_init_gpu()
eigen/test/gpu_common.h:  gpuDeviceProp_t deviceProp;
eigen/test/gpu_common.h:  gpuGetDeviceProperties(&deviceProp, device);
eigen/test/gpu_common.h:  run_on_gpu(compile_time_device_info(),10,dummy,info);
eigen/test/gpu_common.h:  std::cout << "GPU compile-time info:\n";
eigen/test/gpu_common.h:  #ifdef EIGEN_CUDACC
eigen/test/gpu_common.h:  std::cout << "  EIGEN_CUDACC:                 " << int(EIGEN_CUDACC) << "\n";
eigen/test/gpu_common.h:  #ifdef EIGEN_CUDA_SDK_VER
eigen/test/gpu_common.h:  std::cout << "  EIGEN_CUDA_SDK_VER:             " << int(EIGEN_CUDA_SDK_VER) << "\n";
eigen/test/gpu_common.h:  std::cout << "  EIGEN_CUDA_ARCH:             " << info[0] << "\n";  
eigen/test/gpu_common.h:  std::cout << "GPU device info:\n";
eigen/test/gpu_common.h:#endif // EIGEN_TEST_GPU_COMMON_H
eigen/test/CMakeLists.txt:# CUDA unit tests
eigen/test/CMakeLists.txt:option(EIGEN_TEST_CUDA "Enable CUDA support in unit tests" OFF)
eigen/test/CMakeLists.txt:option(EIGEN_TEST_CUDA_CLANG "Use clang instead of nvcc to compile the CUDA tests" OFF)
eigen/test/CMakeLists.txt:if(EIGEN_TEST_CUDA_CLANG AND NOT CMAKE_CXX_COMPILER MATCHES "clang")
eigen/test/CMakeLists.txt:  message(WARNING "EIGEN_TEST_CUDA_CLANG is set, but CMAKE_CXX_COMPILER does not appear to be clang.")
eigen/test/CMakeLists.txt:if(EIGEN_TEST_CUDA)
eigen/test/CMakeLists.txt:find_package(CUDA 5.0)
eigen/test/CMakeLists.txt:if(CUDA_FOUND)
eigen/test/CMakeLists.txt:  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
eigen/test/CMakeLists.txt:    set(CUDA_NVCC_FLAGS "-ccbin ${CMAKE_C_COMPILER}" CACHE STRING "nvcc flags" FORCE)
eigen/test/CMakeLists.txt:  if(EIGEN_TEST_CUDA_CLANG)
eigen/test/CMakeLists.txt:    string(APPEND CMAKE_CXX_FLAGS " --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")
eigen/test/CMakeLists.txt:    foreach(GPU IN LISTS EIGEN_CUDA_COMPUTE_ARCH)
eigen/test/CMakeLists.txt:      string(APPEND CMAKE_CXX_FLAGS " --cuda-gpu-arch=sm_${GPU}")
eigen/test/CMakeLists.txt:  ei_add_test(gpu_basic)
eigen/test/CMakeLists.txt:endif(CUDA_FOUND)
eigen/test/CMakeLists.txt:endif(EIGEN_TEST_CUDA)
eigen/test/CMakeLists.txt:  set(HIP_PATH "/opt/rocm/hip" CACHE STRING "Path to the HIP installation.")
eigen/test/CMakeLists.txt:	ei_add_test(gpu_basic)
eigen/test/gpu_basic.cu:// workaround issue between gcc >= 4.7 and cuda 5.5
eigen/test/gpu_basic.cu:#include "gpu_common.h"
eigen/test/gpu_basic.cu:EIGEN_DECLARE_TEST(gpu_basic)
eigen/test/gpu_basic.cu:  ei_test_init_gpu();
eigen/test/gpu_basic.cu:  #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(coeff_wise<Vector3f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(coeff_wise<Array44f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(replicate<Array4f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(replicate<Array33f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(redux<Array4f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(redux<Matrix3f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(prod_test<Matrix3f,Matrix3f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(prod_test<Matrix4f,Vector4f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(diagonal<Matrix3f,Vector3f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(diagonal<Matrix4f,Vector4f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(matrix_inverse<Matrix2f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(matrix_inverse<Matrix3f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(matrix_inverse<Matrix4f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(eigenvalues_direct<Matrix3f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(eigenvalues_direct<Matrix2f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  // These subtests compiles only with nvcc and fail with HIPCC and clang-cuda
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(eigenvalues<Matrix4f>(), nthreads, in, out) );
eigen/test/gpu_basic.cu:  CALL_SUBTEST( run_and_compare_to_gpu(eigenvalues<Matrix6f>(), nthreads, in, out) );
eigen/test/main.h:// Same for cuda_fp16.h
eigen/test/main.h:#if defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
eigen/test/main.h:  // Means the compiler is either nvcc or clang with CUDA enabled
eigen/test/main.h:  #define EIGEN_CUDACC __CUDACC__
eigen/test/main.h:#if defined(EIGEN_CUDACC)
eigen/test/main.h:#include <cuda.h>
eigen/test/main.h:  #define EIGEN_CUDA_SDK_VER (CUDA_VERSION * 10)
eigen/test/main.h:  #define EIGEN_CUDA_SDK_VER 0
eigen/test/main.h:#if EIGEN_CUDA_SDK_VER >= 70500
eigen/test/main.h:#include <cuda_fp16.h>
eigen/test/main.h:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && !defined(__SYCL_DEVICE_ONLY__)
eigen/test/main.h:  #elif !defined(__CUDACC__) && !defined(__HIPCC__) && !defined(SYCL_DEVICE_ONLY) // EIGEN_DEBUG_ASSERTS
eigen/test/main.h:  #if !defined(__CUDACC__) && !defined(__HIPCC__) && !defined(SYCL_DEVICE_ONLY)
eigen/Eigen/Core:// We need cuda_runtime.h/hip_runtime.h to ensure that
eigen/Eigen/Core:#if defined(EIGEN_CUDACC)
eigen/Eigen/Core:  #include <cuda_runtime.h>
eigen/Eigen/Core:#if defined(EIGEN_HAS_CUDA_FP16) || defined(EIGEN_HAS_HIP_FP16)
eigen/Eigen/Core:  #define EIGEN_HAS_GPU_FP16
eigen/Eigen/Core:#if defined EIGEN_VECTORIZE_GPU
eigen/Eigen/Core:  #include "src/Core/arch/GPU/PacketMath.h"
eigen/Eigen/Core:  #include "src/Core/arch/GPU/MathFunctions.h"
eigen/Eigen/Core:  #include "src/Core/arch/GPU/TypeCasting.h"
eigen/Eigen/Core:// on CUDA devices
eigen/Eigen/Core:#ifdef EIGEN_CUDACC
eigen/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
eigen/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
eigen/Eigen/src/Core/util/Memory.h:#if ! defined EIGEN_ALLOCA && ! defined EIGEN_GPU_COMPILE_PHASE
eigen/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/Eigen/src/Core/util/Meta.h: #if defined(EIGEN_CUDA_ARCH)
eigen/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
eigen/Eigen/src/Core/util/Meta.h:    return CUDART_MAX_NORMAL_F;
eigen/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
eigen/Eigen/src/Core/util/Meta.h:    return CUDART_INF_F;
eigen/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
eigen/Eigen/src/Core/util/Meta.h:    return CUDART_NAN_F;
eigen/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
eigen/Eigen/src/Core/util/Meta.h:    return CUDART_INF;
eigen/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
eigen/Eigen/src/Core/util/Meta.h:    return CUDART_NAN;
eigen/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/Eigen/src/Core/util/Meta.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) || (!defined(EIGEN_CUDA_ARCH) && defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
eigen/Eigen/src/Core/util/Meta.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) || (!defined(EIGEN_CUDA_ARCH) && defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
eigen/Eigen/src/Core/util/StaticAssert.h:        GPU_TENSOR_CONTRACTION_DOES_NOT_SUPPORT_OUTPUT_KERNELS=1
eigen/Eigen/src/Core/util/Macros.h:#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
eigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_COMP_NVCC  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
eigen/Eigen/src/Core/util/Macros.h:#elif defined(__CUDACC_VER__)
eigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_COMP_NVCC __CUDACC_VER__
eigen/Eigen/src/Core/util/Macros.h:// Detect GPU compilers and architectures
eigen/Eigen/src/Core/util/Macros.h:// Note that this also makes EIGEN_CUDACC and EIGEN_HIPCC mutually exclusive
eigen/Eigen/src/Core/util/Macros.h:#if defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
eigen/Eigen/src/Core/util/Macros.h:  // Means the compiler is either nvcc or clang with CUDA enabled
eigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDACC __CUDACC__
eigen/Eigen/src/Core/util/Macros.h:#if defined(__CUDA_ARCH__) && !defined(EIGEN_NO_CUDA)
eigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDA_ARCH __CUDA_ARCH__
eigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC)
eigen/Eigen/src/Core/util/Macros.h:#include <cuda.h>
eigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDA_SDK_VER (CUDA_VERSION * 10)
eigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDA_SDK_VER 0
eigen/Eigen/src/Core/util/Macros.h:  // Means the compiler is HIPCC (analogous to EIGEN_CUDACC, but for HIP)
eigen/Eigen/src/Core/util/Macros.h:    // analogous to EIGEN_CUDA_ARCH, but for HIP
eigen/Eigen/src/Core/util/Macros.h:// Unify CUDA/HIPCC
eigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC) || defined(EIGEN_HIPCC)
eigen/Eigen/src/Core/util/Macros.h:// If either EIGEN_CUDACC or EIGEN_HIPCC is defined, then define EIGEN_GPUCC
eigen/Eigen/src/Core/util/Macros.h:#define EIGEN_GPUCC
eigen/Eigen/src/Core/util/Macros.h:// EIGEN_CUDACC implies the CUDA compiler and is used to tweak Eigen code for use in CUDA kernels
eigen/Eigen/src/Core/util/Macros.h:// In most cases the same tweaks are required to the Eigen code to enable in both the HIP and CUDA kernels.
eigen/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDACC) || defined(EIGEN_HIPCC)
eigen/Eigen/src/Core/util/Macros.h:// For cases where the tweak is specific to CUDA, the code should be guarded with
eigen/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDACC)
eigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIP_DEVICE_COMPILE)
eigen/Eigen/src/Core/util/Macros.h:// If either EIGEN_CUDA_ARCH or EIGEN_HIP_DEVICE_COMPILE is defined, then define EIGEN_GPU_COMPILE_PHASE
eigen/Eigen/src/Core/util/Macros.h:#define EIGEN_GPU_COMPILE_PHASE
eigen/Eigen/src/Core/util/Macros.h:// GPU compilers (HIPCC, NVCC) typically do two passes over the source code,
eigen/Eigen/src/Core/util/Macros.h://   + another to compile the source for the "device" (ie. GPU)
eigen/Eigen/src/Core/util/Macros.h:// EIGEN_CUDA_ARCH implies the device compilation phase in CUDA
eigen/Eigen/src/Core/util/Macros.h:// In most cases, the "host" / "device" specific code is the same for both HIP and CUDA
eigen/Eigen/src/Core/util/Macros.h://       #if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/Eigen/src/Core/util/Macros.h://       #if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIP_DEVICE_COMPILE)
eigen/Eigen/src/Core/util/Macros.h:// For cases where the tweak is specific to CUDA, the code should be guarded with
eigen/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDA_ARCH)
eigen/Eigen/src/Core/util/Macros.h:  #if defined(EIGEN_CUDACC)
eigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC) && EIGEN_HAS_CONSTEXPR
eigen/Eigen/src/Core/util/Macros.h:    #ifdef __CUDACC_RELAXED_CONSTEXPR__
eigen/Eigen/src/Core/util/Macros.h:  #elif defined(__clang__) && defined(__CUDA__) && __has_feature(cxx_relaxed_constexpr)
eigen/Eigen/src/Core/util/Macros.h:// GPU stuff
eigen/Eigen/src/Core/util/Macros.h:// Disable some features when compiling with GPU compilers (NVCC/clang-cuda/SYCL/HIPCC)
eigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC) || defined(SYCL_DEVICE_ONLY) || defined(EIGEN_HIPCC)
eigen/Eigen/src/Core/util/Macros.h:// All functions callable from CUDA/HIP code must be qualified with __device__
eigen/Eigen/src/Core/util/Macros.h:#elif defined(EIGEN_GPUCC) 
eigen/Eigen/src/Core/util/Macros.h:// When compiling CUDA/HIP device code with NVCC or HIPCC
eigen/Eigen/src/Core/util/Macros.h:#if (defined(EIGEN_CUDA_ARCH) && defined(__NVCC__)) || defined(EIGEN_HIP_DEVICE_COMPILE)
eigen/Eigen/src/Core/util/Macros.h:  // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
eigen/Eigen/src/Core/util/Macros.h:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(EIGEN_CUDA_ARCH) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL) && !defined(EIGEN_HIP_DEVICE_COMPILE)
eigen/Eigen/src/Core/util/Macros.h:#  if defined(EIGEN_CUDA_ARCH)
eigen/Eigen/src/Core/util/ConfigureVectorization.h:#if (defined EIGEN_CUDACC)
eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/util/ConfigureVectorization.h:    // GPU code is always vectorized and requires memory alignment for
eigen/Eigen/src/Core/util/ConfigureVectorization.h:#if !(defined(EIGEN_DONT_VECTORIZE) || defined(EIGEN_GPUCC))
eigen/Eigen/src/Core/util/ConfigureVectorization.h:#if defined EIGEN_CUDACC
eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #define EIGEN_VECTORIZE_GPU
eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #if EIGEN_CUDA_SDK_VER >= 70500
eigen/Eigen/src/Core/util/ConfigureVectorization.h:    #define EIGEN_HAS_CUDA_FP16
eigen/Eigen/src/Core/util/ConfigureVectorization.h:#if defined(EIGEN_HAS_CUDA_FP16)
eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #include <cuda_runtime_api.h>
eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #include <cuda_fp16.h>
eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #define EIGEN_VECTORIZE_GPU
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/Eigen/src/Core/MathFunctions.h:  #if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/Eigen/src/Core/MathFunctions.h:  #if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/Eigen/src/Core/MathFunctions.h:  #if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/Eigen/src/Core/MathFunctions.h:#if (!defined(EIGEN_GPUCC) || defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC)) 
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if (!defined(EIGEN_GPUCC)) && EIGEN_FAST_MATH && !defined(SYCL_DEVICE_ONLY)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef EIGEN_GPUCC
eigen/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
eigen/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
eigen/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
eigen/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
eigen/Eigen/src/Core/GeneralProduct.h:    #ifndef EIGEN_GPU_COMPILE_PHASE
eigen/Eigen/src/Core/GenericPacketMath.h:#elif defined(EIGEN_CUDA_ARCH)
eigen/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
eigen/Eigen/src/Core/GenericPacketMath.h:#if !defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/ProductEvaluators.h:#ifndef EIGEN_GPUCC
eigen/Eigen/src/Core/ProductEvaluators.h:#ifndef EIGEN_GPUCC
eigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
eigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
eigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#ifndef EIGEN_PACKET_MATH_GPU_H
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_PACKET_MATH_GPU_H
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:// We need to distinguish ‘clang as the CUDA compiler’ from ‘clang as the host compiler,
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIP_DEVICE_COMPILE) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#endif  // EIGEN_CUDA_ARCH || defined(EIGEN_HIP_DEVICE_COMPILE)
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDACC) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:  (defined(EIGEN_HAS_CUDA_FP16) && defined(__clang__) && defined(__CUDA__))
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if !defined(EIGEN_CUDA_ARCH) && !defined(EIGEN_HIP_DEVICE_COMPILE)
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if !defined(EIGEN_CUDA_ARCH) && !defined(EIGEN_HIP_DEVICE_COMPILE)
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 350
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 350
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530) || \
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // EIGEN_PACKET_MATH_GPU_H
eigen/Eigen/src/Core/arch/GPU/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_GPU_H
eigen/Eigen/src/Core/arch/GPU/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_GPU_H
eigen/Eigen/src/Core/arch/GPU/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
eigen/Eigen/src/Core/arch/GPU/MathFunctions.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
eigen/Eigen/src/Core/arch/GPU/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_GPU_H
eigen/Eigen/src/Core/arch/GPU/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_GPU_H
eigen/Eigen/src/Core/arch/GPU/TypeCasting.h:#define EIGEN_TYPE_CASTING_GPU_H
eigen/Eigen/src/Core/arch/GPU/TypeCasting.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
eigen/Eigen/src/Core/arch/GPU/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_GPU_H
eigen/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(EIGEN_CUDACC) && defined(EIGEN_USE_GPU)
eigen/Eigen/src/Core/arch/CUDA/Complex.h:// building for CUDA to avoid non-constexpr methods.
eigen/Eigen/src/Core/arch/CUDA/Complex.h:#endif // EIGEN_COMPLEX_CUDA_H
eigen/Eigen/src/Core/arch/SYCL/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
eigen/Eigen/src/Core/arch/SYCL/InteropHeaders.h:// Make sure this is only available when targeting a GPU: we don't want to
eigen/Eigen/src/Core/arch/HIP/hcc/math_constants.h: *  HIP equivalent of the CUDA header of the same name
eigen/Eigen/src/Core/arch/Default/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
eigen/Eigen/src/Core/arch/Default/Half.h:// type Eigen::half (inheriting either from CUDA's or HIP's __half struct) with
eigen/Eigen/src/Core/arch/Default/Half.h:// to disk and the likes), but fast on GPUs.
eigen/Eigen/src/Core/arch/Default/Half.h:#if !defined(EIGEN_HAS_GPU_FP16)
eigen/Eigen/src/Core/arch/Default/Half.h:// Make our own __half_raw definition that is similar to CUDA's.
eigen/Eigen/src/Core/arch/Default/Half.h:#elif defined(EIGEN_HAS_CUDA_FP16)
eigen/Eigen/src/Core/arch/Default/Half.h: #if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER < 90000
eigen/Eigen/src/Core/arch/Default/Half.h:// In CUDA < 9.0, __half is the equivalent of CUDA 9's __half_raw
eigen/Eigen/src/Core/arch/Default/Half.h: #endif // defined(EIGEN_HAS_CUDA_FP16)
eigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16)
eigen/Eigen/src/Core/arch/Default/Half.h: #elif defined(EIGEN_HAS_CUDA_FP16)
eigen/Eigen/src/Core/arch/Default/Half.h:  #if (defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER >= 90000)
eigen/Eigen/src/Core/arch/Default/Half.h:#if !defined(EIGEN_HAS_GPU_FP16)
eigen/Eigen/src/Core/arch/Default/Half.h:#elif defined(EIGEN_HAS_CUDA_FP16)
eigen/Eigen/src/Core/arch/Default/Half.h:  // Note that EIGEN_CUDA_SDK_VER is set to 0 even when compiling with HIP, so
eigen/Eigen/src/Core/arch/Default/Half.h:  // (EIGEN_CUDA_SDK_VER < 90000) is true even for HIP!  So keeping this within
eigen/Eigen/src/Core/arch/Default/Half.h:  // #if defined(EIGEN_HAS_CUDA_FP16) is needed
eigen/Eigen/src/Core/arch/Default/Half.h:  #if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER < 90000
eigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16)
eigen/Eigen/src/Core/arch/Default/Half.h: #elif defined(EIGEN_HAS_CUDA_FP16)
eigen/Eigen/src/Core/arch/Default/Half.h:  #if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER >= 90000
eigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && \
eigen/Eigen/src/Core/arch/Default/Half.h:     EIGEN_CUDA_ARCH >= 530) ||                                  \
eigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER >= 90000
eigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER >= 90000
eigen/Eigen/src/Core/arch/Default/Half.h:// We need to distinguish ‘clang as the CUDA compiler’ from ‘clang as the host compiler,
eigen/Eigen/src/Core/arch/Default/Half.h:#if EIGEN_COMP_CLANG && defined(EIGEN_CUDACC)
eigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_HAS_NATIVE_FP16)
eigen/Eigen/src/Core/arch/Default/Half.h:// Definitions for CPUs and older HIP+CUDA, mostly working through conversion
eigen/Eigen/src/Core/arch/Default/Half.h:#if defined(__clang__) && defined(__CUDA__)
eigen/Eigen/src/Core/arch/Default/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
eigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
eigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
eigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
eigen/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530) || \
eigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && EIGEN_CUDA_SDK_VER >= 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
eigen/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530) || \
eigen/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300) || \
eigen/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300) || \
eigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
eigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
eigen/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
eigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
eigen/Eigen/src/Core/arch/Default/Half.h:  #if (EIGEN_CUDA_SDK_VER < 90000) || \
eigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350) || \
eigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/Eigen/src/Core/arch/Default/TypeCasting.h:    #if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
eigen/Eigen/src/Core/arch/Default/TypeCasting.h:    #if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
eigen/Eigen/src/Core/arch/Default/TypeCasting.h:    #if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
eigen/Eigen/src/SVD/BDCSVD.h:#if !defined(EIGEN_GPUCC)
eigen/doc/Manual.dox:  - \subpage TopicCUDA
eigen/doc/PreprocessorDirectives.dox: - \b \c EIGEN_NO_CUDA - disables CUDA support when defined. Might be useful in .cu files for which Eigen is used on the host only,
eigen/doc/UsingNVCC.dox:/** \page TopicCUDA Using Eigen in CUDA kernels
eigen/doc/UsingNVCC.dox:Staring from CUDA 5.5 and Eigen 3.3, it is possible to use Eigen's matrices, vectors, and arrays for fixed size within CUDA kernels. This is especially useful when working on numerous but small problems. By default, when Eigen's headers are included within a .cu file compiled by nvcc most Eigen's functions and methods are prefixed by the \c __device__ \c __host__ keywords making them callable from both host and device code.
eigen/doc/UsingNVCC.dox:This support can be disabled by defining \c EIGEN_NO_CUDA before including any Eigen's header.
eigen/doc/UsingNVCC.dox:    // workaround issue between gcc >= 4.7 and cuda 5.5
eigen/doc/UsingNVCC.dox: - On 64bits system Eigen uses \c long \c int as the default type for indexes and sizes. On CUDA device, it would make sense to default to 32 bits \c int.
eigen/doc/UsingNVCC.dox:   However, to keep host and CUDA code compatible, this cannot be done automatically by %Eigen, and the user is thus required to define \c EIGEN_DEFAULT_DENSE_INDEX_TYPE to \c int throughout his code (or only for CUDA code if there is no interaction between host and CUDA code through %Eigen's object).
eigen/CMakeLists.txt:set(EIGEN_CUDA_COMPUTE_ARCH 30 CACHE STRING "The CUDA compute architecture level to target when compiling CUDA code")
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  DataType* gpu_in_data = static_cast<DataType*>(sycl_device.allocate(tensor.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  DataType* gpu_out_data =static_cast<DataType*>(sycl_device.allocate(reversed_tensor.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> >  in_gpu(gpu_in_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> >  out_gpu(gpu_out_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, tensor.data(),(tensor.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.reverse(dim_rev);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyDeviceToHost(reversed_tensor.data(), gpu_out_data, reversed_tensor.dimensions().TotalSize()*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.reverse(dim_rev);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyDeviceToHost(reversed_tensor.data(), gpu_out_data, reversed_tensor.dimensions().TotalSize()*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.reverse(dim_rev);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyDeviceToHost(reversed_tensor.data(), gpu_out_data, reversed_tensor.dimensions().TotalSize()*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  DataType* gpu_in_data = static_cast<DataType*>(sycl_device.allocate(tensor.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  DataType* gpu_out_data_expected =static_cast<DataType*>(sycl_device.allocate(expected.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  DataType* gpu_out_data_result =static_cast<DataType*>(sycl_device.allocate(result.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> >  in_gpu(gpu_in_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> >  out_gpu_expected(gpu_out_data_expected, tensorRange);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType> >  out_gpu_result(gpu_out_data_result, tensorRange);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, tensor.data(),(tensor.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:    out_gpu_expected.reverse(dim_rev).device(sycl_device) = in_gpu;
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:    out_gpu_expected.device(sycl_device) = in_gpu.reverse(dim_rev);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyDeviceToHost(expected.data(), gpu_out_data_expected, expected.dimensions().TotalSize()*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:        out_gpu_result.slice(dst_slice_start, dst_slice_dim).reverse(dim_rev).device(sycl_device) =
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:          in_gpu.slice(src_slice_start, src_slice_dim);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:      out_gpu_result.slice(dst_slice_start, dst_slice_dim).device(sycl_device) =
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:          in_gpu.slice(src_slice_start, src_slice_dim).reverse(dim_rev);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyDeviceToHost(result.data(), gpu_out_data_result, result.dimensions().TotalSize()*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_out_data_result, result.data(),(result.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:       out_gpu_result.slice(dst_slice_start, dst_slice_dim).reverse(dim_rev).device(sycl_device) =
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:           in_gpu.slice(dst_slice_start, dst_slice_dim);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:       out_gpu_result.slice(dst_slice_start, dst_slice_dim).device(sycl_device) =
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:           in_gpu.reverse(dim_rev).slice(dst_slice_start, dst_slice_dim);
eigen/unsupported/test/cxx11_tensor_reverse_sycl.cpp:  sycl_device.memcpyDeviceToHost(result.data(), gpu_out_data_result, result.dimensions().TotalSize()*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(left.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(right.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_in1(gpu_in1_data, leftRange);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_in2(gpu_in2_data, rightRange);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in1_data, left.data(),(left.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, right.data(),(right.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_out_data1 =  static_cast<DataType*>(sycl_device.allocate(concatenation1.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_out1(gpu_out_data1, concatenation1.dimensions());
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  gpu_out1.device(sycl_device) =gpu_in1.concatenate(gpu_in2, 0);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyDeviceToHost(concatenation1.data(), gpu_out_data1,(concatenation1.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_out_data1);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_out_data2 =  static_cast<DataType*>(sycl_device.allocate(concatenation2.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_out2(gpu_out_data2, concatenation2.dimensions());
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  gpu_out2.device(sycl_device) =gpu_in1.concatenate(gpu_in2, 1);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyDeviceToHost(concatenation2.data(), gpu_out_data2,(concatenation2.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_out_data2);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_out_data3 =  static_cast<DataType*>(sycl_device.allocate(concatenation3.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_out3(gpu_out_data3, concatenation3.dimensions());
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  gpu_out3.device(sycl_device) =gpu_in1.concatenate(gpu_in2, 2);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyDeviceToHost(concatenation3.data(), gpu_out_data3,(concatenation3.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_out_data3);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_in1_data);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_in2_data);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(left.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(right.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(result.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>> gpu_in1(gpu_in1_data, leftRange);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>> gpu_in2(gpu_in2_data, rightRange);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>> gpu_out(gpu_out_data, resRange);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in1_data, left.data(),(left.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, right.data(),(right.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_out_data, result.data(),(result.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp: gpu_in1.concatenate(gpu_in2, 0).device(sycl_device) =gpu_out;
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp: sycl_device.memcpyDeviceToHost(left.data(), gpu_in1_data,(left.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp: sycl_device.memcpyDeviceToHost(right.data(), gpu_in2_data,(right.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_in1_data);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_in2_data);
eigen/unsupported/test/cxx11_tensor_concatenation_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_tensor(d_tensor, tensor_dims);
eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_no_stride(d_no_stride, tensor_dims);
eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_stride(d_stride, stride_dims);
eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  gpu_no_stride.device(sycl_device)=gpu_tensor.stride(strides);
eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  gpu_stride.device(sycl_device)=gpu_tensor.stride(strides);
eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_tensor(d_tensor, tensor_dims);
eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_no_stride(d_no_stride, stride_dims);
eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, IndexType> > gpu_stride(d_stride, stride_dims);
eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  gpu_stride.stride(strides).device(sycl_device)=gpu_tensor;
eigen/unsupported/test/cxx11_tensor_striding_sycl.cpp:  gpu_no_stride.stride(strides).device(sycl_device)=gpu_tensor.stride(no_strides);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:#define EIGEN_USE_GPU
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Type* gpu_in_ptr = static_cast<Type*>(gpu_device.allocate(in_bytes));
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Type* gpu_out_ptr = static_cast<Type*>(gpu_device.allocate(out_bytes));
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  TensorMap<Tensor<Type, 2, DataLayout> > in_gpu(gpu_in_ptr, num_rows, num_cols);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  TensorMap<Tensor<Type, 0, DataLayout> > out_gpu(gpu_out_ptr);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  out_gpu.device(gpu_device) = in_gpu.sum();
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Tensor<Type, 0, DataLayout> full_redux_gpu;
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  // Check that the CPU and GPU reductions return the same result.
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_device.deallocate(gpu_in_ptr);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_device.deallocate(gpu_out_ptr);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::GpuDevice dev(&stream);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 3, DataLayout> > gpu_in(in_data, dim_x, dim_y, dim_z);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 2, DataLayout> > gpu_out(out_data, dim_y, dim_z);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_out.device(dev) = gpu_in.sum(red_axis);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_out.device(dev) += gpu_in.sum(red_axis);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Tensor<Type, 2, DataLayout> redux_gpu(dim_y, dim_z);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  dev.memcpyDeviceToHost(redux_gpu.data(), out_data, gpu_out.size()*sizeof(Type));
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  // Check that the CPU and GPU reductions return the same result.
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  for (int i = 0; i < gpu_out.size(); ++i) {
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:    VERIFY_IS_APPROX(2*redux(i), redux_gpu(i));
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::GpuDevice dev(&stream);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 3, DataLayout> > gpu_in(in_data, dim_x, dim_y, dim_z);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 2, DataLayout> > gpu_out(out_data, dim_x, dim_y);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_out.device(dev) = gpu_in.sum(red_axis);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  gpu_out.device(dev) += gpu_in.sum(red_axis);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  Tensor<Type, 2, DataLayout> redux_gpu(dim_x, dim_y);
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  dev.memcpyDeviceToHost(redux_gpu.data(), out_data, gpu_out.size()*sizeof(Type));
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  // Check that the CPU and GPU reductions return the same result.
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:  for (int i = 0; i < gpu_out.size(); ++i) {
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:    VERIFY_IS_APPROX(2*redux(i), redux_gpu(i));
eigen/unsupported/test/cxx11_tensor_reduction_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_reduction_gpu) {
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  DataType* gpu_data_single_voxel_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  TensorMap<Tensor<DataType, 6, DataLayout,IndexType>> gpu_single_voxel_patch_col_major(gpu_data_single_voxel_patch_col_major, patchColMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  gpu_single_voxel_patch_col_major.device(sycl_device)=gpu_col_major.extract_volume_patches(1, 1, 1);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_voxel_patch_col_major.data(), gpu_data_single_voxel_patch_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  DataType* gpu_data_single_voxel_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  TensorMap<Tensor<DataType, 6, RowMajor,IndexType>> gpu_single_voxel_patch_row_major(gpu_data_single_voxel_patch_row_major, patchRowMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  gpu_single_voxel_patch_row_major.device(sycl_device)=gpu_row_major.extract_volume_patches(1, 1, 1);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_voxel_patch_row_major.data(), gpu_data_single_voxel_patch_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp: sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_col_major);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_row_major);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_voxel_patch_col_major);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_voxel_patch_row_major);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    TensorMap<Tensor<DataType, 5, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    TensorMap<Tensor<DataType, 5, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    DataType* gpu_data_entire_volume_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    TensorMap<Tensor<DataType, 6, DataLayout,IndexType>> gpu_entire_volume_patch_col_major(gpu_data_entire_volume_patch_col_major, patchColMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    gpu_entire_volume_patch_col_major.device(sycl_device)=gpu_col_major.extract_volume_patches(patch_z, patch_y, patch_x);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:    sycl_device.memcpyDeviceToHost(entire_volume_patch_col_major.data(), gpu_data_entire_volume_patch_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  DataType* gpu_data_entire_volume_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  TensorMap<Tensor<DataType, 6, RowMajor,IndexType>> gpu_entire_volume_patch_row_major(gpu_data_entire_volume_patch_row_major, patchRowMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  gpu_entire_volume_patch_row_major.device(sycl_device)=gpu_row_major.extract_volume_patches(patch_z, patch_y, patch_x);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(entire_volume_patch_row_major.data(), gpu_data_entire_volume_patch_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_col_major);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_row_major);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_entire_volume_patch_col_major);
eigen/unsupported/test/cxx11_tensor_volume_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_entire_volume_patch_row_major);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_input(d_input, input_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout, IndexType> > gpu_kernel(d_kernel, kernel_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_result(d_result, result_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims3);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_input(d_input, input_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_kernel(d_kernel, kernel_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_result(d_result, result_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims3);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_input(d_input, input_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_kernel(d_kernel, kernel_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > gpu_result(d_result, result_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims3);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_input(d_input, input_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout, IndexType> > gpu_kernel(d_kernel, kernel_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_result(d_result, result_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims3);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout,IndexType> > gpu_input(d_input, input_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout,IndexType> > gpu_kernel(d_kernel, kernel_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout,IndexType> > gpu_result(d_result, result_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_result.device(sycl_device)=gpu_input.convolve(gpu_kernel, dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_input(d_input, input_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_kernel(d_kernel, kernel_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_valid(d_valid, valid.dimensions());
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_valid.device(sycl_device)=gpu_input.pad(padding).convolve(gpu_kernel, dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_same(d_same, same.dimensions());
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_same.device(sycl_device)=gpu_input.pad(padding).convolve(gpu_kernel, dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_full(d_full, full.dimensions());
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_full.device(sycl_device)=gpu_input.pad(padding).convolve(gpu_kernel, dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_input(d_input, input_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_kernel(d_kernel, kernel_dims);
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 1, DataLayout,IndexType> > gpu_result(d_result, result.dimensions());
eigen/unsupported/test/cxx11_tensor_convolution_sycl.cpp:  gpu_result.device(sycl_device)=gpu_input.stride(stride_of_3).convolve(gpu_kernel, dims).stride(stride_of_2);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(in1.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(out1.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu1(gpu_data1, tensorRange);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu2(gpu_data2, tensorRange);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, in1.data(),(in1.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data2, in1.data(),(in1.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu1.device(sycl_device) = gpu1 * 3.14f;
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu2.device(sycl_device) = gpu2 * 2.7f;
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out1.data(), gpu_data1,(out1.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out2.data(), gpu_data1,(out2.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out3.data(), gpu_data2,(out3.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_data1);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_data2);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType* gpu_data  = static_cast<DataType*>(sycl_device.allocate(in1.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 1, DataLayout, IndexType>> gpu1(gpu_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data, in1.data(),(in1.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_data, out.size()*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_data);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(in1.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(in2.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType * gpu_in3_data  = static_cast<DataType*>(sycl_device.allocate(in3.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(out.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_in1(gpu_in1_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_in2(gpu_in2_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_in3(gpu_in3_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_out(gpu_out_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_in1.device(sycl_device) = gpu_in1.constant(1.2f);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(in1.data(), gpu_in1_data ,(in1.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 * 1.2f;
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data ,(out.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in2.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 * gpu_in2;
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 + gpu_in2;
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 * gpu_in1;
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) =  gpu_in1 * gpu_in1.constant(3.14f) + gpu_in2 * gpu_in2.constant(2.7f);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(),gpu_out_data,(out.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in3_data, in3.data(),(in3.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) =(gpu_in1 > gpu_in1.constant(0.5f)).select(gpu_in2, gpu_in3);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_in1_data);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_in2_data);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_in3_data);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:    Scalar1* gpu_in_data  = static_cast<Scalar1*>(sycl_device.allocate(in.size()*sizeof(Scalar1)));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:    Scalar2 * gpu_out_data =  static_cast<Scalar2*>(sycl_device.allocate(out.size()*sizeof(Scalar2)));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:    TensorMap<Tensor<Scalar1, 1, DataLayout, IndexType>> gpu_in(gpu_in_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:    TensorMap<Tensor<Scalar2, 1, DataLayout, IndexType>> gpu_out(gpu_out_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.size())*sizeof(Scalar1));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:    gpu_out.device(sycl_device) = gpu_in. template cast<Scalar2>();
eigen/unsupported/test/cxx11_tensor_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data, out.size()*sizeof(Scalar2));
eigen/unsupported/test/cxx11_tensor_sycl.cpp:    sycl_device.deallocate(gpu_in_data);
eigen/unsupported/test/cxx11_tensor_sycl.cpp:    sycl_device.deallocate(gpu_out_data);
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:#define EIGEN_USE_GPU
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:#include <Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:void test_gpu_cumsum(int m_size, int k_size, int n_size)
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  Tensor<float, 3, DataLayout> t_result_gpu(m_size, k_size, n_size);
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpuMalloc((void**)(&d_t_input), t_input_bytes);
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpuMalloc((void**)(&d_t_result), t_result_bytes);
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpuMemcpy(d_t_input, t_input.data(), t_input_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:      gpu_t_input(d_t_input, Eigen::array<int, 3>(m_size, k_size, n_size));
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:      gpu_t_result(d_t_result, Eigen::array<int, 3>(m_size, k_size, n_size));
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpu_t_result.device(gpu_device) = gpu_t_input.cumsum(1);
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpuMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, gpuMemcpyDeviceToHost);
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:    if (fabs(t_result(i) - t_result_gpu(i)) < 1e-4f) {
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), 1e-4f)) {
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:              << " vs " <<  t_result_gpu(i) << std::endl;
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpuFree((void*)d_t_input);
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  gpuFree((void*)d_t_result);
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_scan_gpu)
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  CALL_SUBTEST_1(test_gpu_cumsum<ColMajor>(128, 128, 128));
eigen/unsupported/test/cxx11_tensor_scan_gpu.cu:  CALL_SUBTEST_2(test_gpu_cumsum<RowMajor>(128, 128, 128));
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:#define EIGEN_USE_GPU
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:#include <Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:void test_gpu_random_uniform()
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  gpu_out.device(gpu_device) = gpu_out.random();
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:void test_gpu_random_normal()
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  gpu_out.device(gpu_device) = gpu_out.random(gen);
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_random_gpu)
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  CALL_SUBTEST(test_gpu_random_uniform());
eigen/unsupported/test/cxx11_tensor_random_gpu.cu:  CALL_SUBTEST(test_gpu_random_normal());
eigen/unsupported/test/cxx11_tensor_gpu.cu:#define EIGEN_USE_GPU
eigen/unsupported/test/cxx11_tensor_gpu.cu:#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
eigen/unsupported/test/cxx11_tensor_gpu.cu:#define EIGEN_GPU_TEST_C99_MATH  EIGEN_HAS_CXX11
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_nullary() {
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in1), tensor_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in2), tensor_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in1, in1.data(), tensor_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in2, in2.data(), tensor_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_in1(
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_in2(
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_in1.device(gpu_device) = gpu_in1.constant(3.14f);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_in2.device(gpu_device) = gpu_in2.random();
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(new1.data(), d_in1, tensor_bytes, gpuMemcpyDeviceToHost,
eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(new2.data(), d_in2, tensor_bytes, gpuMemcpyDeviceToHost,
eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in1);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in2);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_elementwise_small() {
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in1), in1_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in2), in2_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in2, in2.data(), in2_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in2(
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_out(
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost,
eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in1);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in2);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_elementwise()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in1), in1_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in2), in2_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in3), in3_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in2, in2.data(), in2_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in3, in3.data(), in3_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in3(d_in3, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2 * gpu_in3;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in1);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in2);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in3);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_props() {
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in1), in1_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_out(
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = (gpu_in1.isnan)();
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost,
eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in1);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_reduction()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in1), in1_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_in1(d_in1, 72,53,97,113);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in1.maximum(reduction_axis);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in1);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_contraction()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  // a 15 SM GK110 GPU
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_t_left), t_left_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_t_right), t_right_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_t_result), t_result_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_t_left, t_left.data(), t_left_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_t_right, t_right.data(), t_right_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_t_left(d_t_left, 6, 50, 3, 31);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_t_right(d_t_right, 3, 31, 7, 20, 1);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_t_result(d_t_result, 6, 50, 7, 20, 1);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(t_result.data(), d_t_result, t_result_bytes, gpuMemcpyDeviceToHost);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_t_left);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_t_right);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_t_result);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_convolution_1d()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_input), input_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_kernel), kernel_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_input(d_input, 74,37,11,137);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, DataLayout> > gpu_kernel(d_kernel, 4);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_out(d_out, 74,34,11,137);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_input);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_kernel);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_convolution_inner_dim_col_major_1d()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_input), input_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_kernel), kernel_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, ColMajor> > gpu_input(d_input,74,9,11,7);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, ColMajor> > gpu_kernel(d_kernel,4);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, ColMajor> > gpu_out(d_out,71,9,11,7);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_input);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_kernel);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_convolution_inner_dim_row_major_1d()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_input), input_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_kernel), kernel_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, RowMajor> > gpu_input(d_input, 7,9,11,74);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, RowMajor> > gpu_kernel(d_kernel, 4);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, RowMajor> > gpu_out(d_out, 7,9,11,71);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_input);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_kernel);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_convolution_2d()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_input), input_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_kernel), kernel_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_input(d_input,74,37,11,137);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2, DataLayout> > gpu_kernel(d_kernel,3,4);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_out(d_out,74,35,8,137);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_input);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_kernel);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_convolution_3d()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_input), input_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_kernel), kernel_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_input, input.data(), input_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_kernel, kernel.data(), kernel_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;    
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_input(d_input,74,37,11,137,17);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3, DataLayout> > gpu_kernel(d_kernel,3,4,2);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_out(d_out,74,35,8,136,17);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_input);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_kernel);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:#if EIGEN_GPU_TEST_C99_MATH
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_lgamma(const Scalar stddev)
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in, in.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in.lgamma();
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_digamma()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in, in.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 7);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 7);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in.digamma();
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_zeta()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_x), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_q), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_x, in_x.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_q, in_q.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 6);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_q(d_in_q, 6);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 6);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in_x.zeta(gpu_in_q);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_x);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_q);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_polygamma()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_x), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_n), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_x, in_x.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_n, in_n.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 7);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_n(d_in_n, 7);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 7);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in_n.polygamma(gpu_in_x);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_x);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_n);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_igamma()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMalloc((void**)(&d_a), bytes) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMalloc((void**)(&d_x), bytes) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMalloc((void**)(&d_out), bytes) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_a, a.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_x, x.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_a(d_a, 6, 6);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_x(d_x, 6, 6);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 6, 6);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_a.igamma(gpu_x);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_a);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_x);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_igammac()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_a), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_x), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_a, a.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_x, x.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_a(d_a, 6, 6);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_x(d_x, 6, 6);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 6, 6);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_a.igammac(gpu_x);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_a);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_x);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:#if EIGEN_GPU_TEST_C99_MATH
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_erf(const Scalar stddev)
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMalloc((void**)(&d_in), bytes) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMalloc((void**)(&d_out), bytes) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in, in.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in.erf();
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_erfc(const Scalar stddev)
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in, in.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in.erfc();
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_ndtri()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_x), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_x, in_x.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 6);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 6);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in_x.ndtri();
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_x);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_betainc()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_x), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_a), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in_b), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_x, in_x.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_a, in_a.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in_b, in_b.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 125);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_a(d_in_a, 125);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_b(d_in_b, 125);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 125);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = betainc(gpu_in_a, gpu_in_b, gpu_in_x);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_x);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_a);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in_b);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_i0e()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in, in_x.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 21);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 21);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in.bessel_i0e();
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_i1e()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_in), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_in, in_x.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 21);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 21);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_in.bessel_i1e();
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_in);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_igamma_der_a()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_a), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_x), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_a, in_a.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_x, in_x.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_a(d_a, 30);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_x(d_x, 30);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 30);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_a.igamma_der_a(gpu_x);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_a);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_x);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:void test_gpu_gamma_sample_der_alpha()
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_alpha), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_sample), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMalloc((void**)(&d_out), bytes);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_alpha, in_alpha.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuMemcpy(d_sample, in_sample.data(), bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_alpha(d_alpha, 30);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_sample(d_sample, 30);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 30);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpu_out.device(gpu_device) = gpu_alpha.gamma_sample_der_alpha(gpu_sample);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuMemcpyAsync(out.data(), d_out, bytes, gpuMemcpyDeviceToHost,
eigen/unsupported/test/cxx11_tensor_gpu.cu:                         gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_alpha);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_sample);
eigen/unsupported/test/cxx11_tensor_gpu.cu:  gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_gpu)
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_1(test_gpu_nullary());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_1(test_gpu_elementwise_small());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_1(test_gpu_elementwise());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_1(test_gpu_props());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_1(test_gpu_reduction());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_2(test_gpu_contraction<ColMajor>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_2(test_gpu_contraction<RowMajor>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_1d<ColMajor>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_1d<RowMajor>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_inner_dim_col_major_1d());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_inner_dim_row_major_1d());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_2d<ColMajor>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_2d<RowMajor>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_3d<ColMajor>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_3(test_gpu_convolution_3d<RowMajor>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:#if EIGEN_GPU_TEST_C99_MATH
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<float>(1.0f));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<float>(100.0f));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<float>(0.01f));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<float>(0.001f));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<double>(1.0));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<double>(100.0));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<double>(0.01));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_lgamma<double>(0.001));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<float>(1.0f));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<float>(100.0f));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<float>(0.01f));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<float>(0.001f));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<float>(1.0f));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  // CALL_SUBTEST(test_gpu_erfc<float>(100.0f));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<float>(5.0f)); // GPU erfc lacks precision for large inputs
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<float>(0.01f));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<float>(0.001f));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<double>(1.0));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<double>(100.0));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<double>(0.01));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erf<double>(0.001));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<double>(1.0));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  // CALL_SUBTEST(test_gpu_erfc<double>(100.0));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<double>(5.0)); // GPU erfc lacks precision for large inputs
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<double>(0.01));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_4(test_gpu_erfc<double>(0.001));
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_ndtri<float>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_ndtri<double>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_digamma<float>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_digamma<double>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_polygamma<float>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_polygamma<double>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_zeta<float>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_zeta<double>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_igamma<float>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_igammac<float>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_igamma<double>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_5(test_gpu_igammac<double>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_betainc<float>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_betainc<double>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_i0e<float>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_i0e<double>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_i1e<float>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_i1e<double>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_i1e<float>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_i1e<double>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_igamma_der_a<float>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_igamma_der_a<double>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_gamma_sample_der_alpha<float>());
eigen/unsupported/test/cxx11_tensor_gpu.cu:  CALL_SUBTEST_6(test_gpu_gamma_sample_der_alpha<double>());
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(padded.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  TensorMap<Tensor<DataType, 4,DataLayout,IndexType>> gpu1(gpu_data1, tensorRange);
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  TensorMap<Tensor<DataType, 4,DataLayout,IndexType>> gpu2(gpu_data2, padedtensorRange);
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  gpu2.device(sycl_device)=gpu1.pad(paddings);
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.memcpyDeviceToHost(padded.data(), gpu_data2,(padded.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.deallocate(gpu_data1);
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.deallocate(gpu_data2);
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(result.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  TensorMap<Tensor<DataType, 4,DataLayout,IndexType>> gpu1(gpu_data1, tensorRange);
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  TensorMap<Tensor<DataType, 2,DataLayout,IndexType>> gpu2(gpu_data2, reshape_dims);
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  gpu2.device(sycl_device)=gpu1.pad(paddings).reshape(reshape_dims);
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.memcpyDeviceToHost(result.data(), gpu_data2,(result.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.deallocate(gpu_data1);
eigen/unsupported/test/cxx11_tensor_padding_sycl.cpp:  sycl_device.deallocate(gpu_data2);
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  DataType * gpu_in_data  = static_cast<DataType*>(sycl_device.allocate(input.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  DataType * gpu_out_data  = static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  TensorMap<TensorFixedSize<DataType, Sizes<2, 3, 5, 7>, DataLayout, IndexType>> gpu_in(gpu_in_data, in_range);
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>> gpu_out(gpu_out_data, out_range);
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, input.data(),(input.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in.broadcast(broadcasts);
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  DataType * gpu_in_data  = static_cast<DataType*>(sycl_device.allocate(input.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  DataType * gpu_out_data  = static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>>  gpu_in(gpu_in_data, in_range);
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>> gpu_out(gpu_out_data, out_range);
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, input.data(),(input.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in.broadcast(broadcasts);
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip1  = static_cast<DataType*>(sycl_device.allocate(chip1TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip1(gpu_data_chip1, chip1TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip1.device(sycl_device)=gpu_tensor.template chip<0l>(1l);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip1.data(), gpu_data_chip1, chip1TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip2  = static_cast<DataType*>(sycl_device.allocate(chip2TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip2(gpu_data_chip2, chip2TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip2.device(sycl_device)=gpu_tensor.template chip<1l>(1l);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip2.data(), gpu_data_chip2, chip2TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip3  = static_cast<DataType*>(sycl_device.allocate(chip3TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip3(gpu_data_chip3, chip3TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip3.device(sycl_device)=gpu_tensor.template chip<2l>(2l);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip3.data(), gpu_data_chip3, chip3TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip4  = static_cast<DataType*>(sycl_device.allocate(chip4TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip4(gpu_data_chip4, chip4TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip4.device(sycl_device)=gpu_tensor.template chip<3l>(5l);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip4.data(), gpu_data_chip4, chip4TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip5  = static_cast<DataType*>(sycl_device.allocate(chip5TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip5(gpu_data_chip5, chip5TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip5.device(sycl_device)=gpu_tensor.template chip<4l>(7l);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip5.data(), gpu_data_chip5, chip5TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip1);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip2);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip3);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip4);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip5);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip1  = static_cast<DataType*>(sycl_device.allocate(chip1TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip1(gpu_data_chip1, chip1TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip1.device(sycl_device)=gpu_tensor.chip(1l,0l);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip1.data(), gpu_data_chip1, chip1TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip2  = static_cast<DataType*>(sycl_device.allocate(chip2TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip2(gpu_data_chip2, chip2TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip2.device(sycl_device)=gpu_tensor.chip(1l,1l);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip2.data(), gpu_data_chip2, chip2TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip3  = static_cast<DataType*>(sycl_device.allocate(chip3TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip3(gpu_data_chip3, chip3TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip3.device(sycl_device)=gpu_tensor.chip(2l,2l);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip3.data(), gpu_data_chip3, chip3TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip4  = static_cast<DataType*>(sycl_device.allocate(chip4TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip4(gpu_data_chip4, chip4TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip4.device(sycl_device)=gpu_tensor.chip(5l,3l);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip4.data(), gpu_data_chip4, chip4TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip5  = static_cast<DataType*>(sycl_device.allocate(chip5TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip5(gpu_data_chip5, chip5TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip5.device(sycl_device)=gpu_tensor.chip(7l,4l);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip5.data(), gpu_data_chip5, chip5TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip1);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip2);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip3);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip4);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip5);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip1  = static_cast<DataType*>(sycl_device.allocate(chip1TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_tensor1  = static_cast<DataType*>(sycl_device.allocate(chip1TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip1(gpu_data_chip1, chip1TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_tensor1(gpu_data_tensor1, chip1TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor1, tensor1.data(), chip1TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip1.device(sycl_device)=gpu_tensor.template chip<0l>(0l) + gpu_tensor1;
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip1.data(), gpu_data_chip1, chip1TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_tensor2  = static_cast<DataType*>(sycl_device.allocate(chip2TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_chip2  = static_cast<DataType*>(sycl_device.allocate(chip2TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout,IndexType>> gpu_tensor2(gpu_data_tensor2, chip2TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout,IndexType>> gpu_chip2(gpu_data_chip2, chip2TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor2, tensor2.data(), chip2TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_chip2.device(sycl_device)=gpu_tensor.template chip<0l>(0l).template chip<1l>(2l) + gpu_tensor2;
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(chip2.data(), gpu_data_chip2, chip2TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor1);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip1);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor2);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_chip2);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input1  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input2  = static_cast<DataType*>(sycl_device.allocate(input2TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_input1(gpu_data_input1, tensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input2(gpu_data_input2, input2TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input1, input1.data(), tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.device(sycl_device)=gpu_input1;
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input2, input2.data(), input2TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.template chip<0l>(1l).device(sycl_device)=gpu_input2;
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.device(sycl_device)=gpu_input1;
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input3  = static_cast<DataType*>(sycl_device.allocate(input3TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input3(gpu_data_input3, input3TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input3, input3.data(), input3TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.template chip<1l>(1l).device(sycl_device)=gpu_input3;
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.device(sycl_device)=gpu_input1;
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input4  = static_cast<DataType*>(sycl_device.allocate(input4TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input4(gpu_data_input4, input4TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input4, input4.data(), input4TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.template chip<2l>(3l).device(sycl_device)=gpu_input4;
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.device(sycl_device)=gpu_input1;
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input5  = static_cast<DataType*>(sycl_device.allocate(input5TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input5(gpu_data_input5, input5TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input5, input5.data(), input5TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.template chip<3l>(4l).device(sycl_device)=gpu_input5;
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.device(sycl_device)=gpu_input1;
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input6  = static_cast<DataType*>(sycl_device.allocate(input6TensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input6(gpu_data_input6, input6TensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input6, input6.data(), input6TensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.template chip<4l>(5l).device(sycl_device)=gpu_input6;
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.device(sycl_device)=gpu_input1;
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  DataType* gpu_data_input7  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_input7(gpu_data_input7, tensorRange);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_input7, input7.data(), tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  gpu_tensor.chip(0l,0l).device(sycl_device)=gpu_input7.chip(0l,0l);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input1);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input2);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input3);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input4);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input5);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input6);
eigen/unsupported/test/cxx11_tensor_chipping_sycl.cpp:  sycl_device.deallocate(gpu_data_input7);
eigen/unsupported/test/cxx11_tensor_device.cu:#define EIGEN_USE_GPU
eigen/unsupported/test/cxx11_tensor_device.cu:#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
eigen/unsupported/test/cxx11_tensor_device.cu:// Context for evaluation on GPU
eigen/unsupported/test/cxx11_tensor_device.cu:struct GPUContext {
eigen/unsupported/test/cxx11_tensor_device.cu:  GPUContext(const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in1, Eigen::TensorMap<Eigen::Tensor<float, 3> >& in2, Eigen::TensorMap<Eigen::Tensor<float, 3> >& out) : in1_(in1), in2_(in2), out_(out), gpu_device_(&stream_) {
eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuMalloc((void**)(&kernel_1d_), 2*sizeof(float)) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuMemcpy(kernel_1d_, kernel_1d_val, 2*sizeof(float), gpuMemcpyHostToDevice) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuMalloc((void**)(&kernel_2d_), 4*sizeof(float)) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuMemcpy(kernel_2d_, kernel_2d_val, 4*sizeof(float), gpuMemcpyHostToDevice) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuMalloc((void**)(&kernel_3d_), 8*sizeof(float)) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuMemcpy(kernel_3d_, kernel_3d_val, 8*sizeof(float), gpuMemcpyHostToDevice) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:  ~GPUContext() {
eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuFree(kernel_1d_) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuFree(kernel_2d_) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:    assert(gpuFree(kernel_3d_) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:  const Eigen::GpuDevice& device() const { return gpu_device_; }
eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::GpuStreamDevice stream_;
eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::GpuDevice gpu_device_;
eigen/unsupported/test/cxx11_tensor_device.cu:void test_gpu() {
eigen/unsupported/test/cxx11_tensor_device.cu:  gpuMalloc((void**)(&d_in1), in1_bytes);
eigen/unsupported/test/cxx11_tensor_device.cu:  gpuMalloc((void**)(&d_in2), in2_bytes);
eigen/unsupported/test/cxx11_tensor_device.cu:  gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_device.cu:  gpuMemcpy(d_in1, in1.data(), in1_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_device.cu:  gpuMemcpy(d_in2, in2.data(), in2_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, 40,50,70);
eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, 40,50,70);
eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, 40,50,70);
eigen/unsupported/test/cxx11_tensor_device.cu:  GPUContext context(gpu_in1, gpu_in2, gpu_out);
eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpy(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpy(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpy(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpy(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, context.device().stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuStreamSynchronize(context.device().stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, context.device().stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuStreamSynchronize(context.device().stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, context.device().stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:  assert(gpuStreamSynchronize(context.device().stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_device.cu:  CALL_SUBTEST_2(test_gpu());
eigen/unsupported/test/CMakeLists.txt:find_package(CUDA 7.0)
eigen/unsupported/test/CMakeLists.txt:if(CUDA_FOUND AND EIGEN_TEST_CUDA)
eigen/unsupported/test/CMakeLists.txt:  # in the CUDA runtime
eigen/unsupported/test/CMakeLists.txt:  message(STATUS "Flags used to compile cuda code: " ${CMAKE_CXX_FLAGS})
eigen/unsupported/test/CMakeLists.txt:    set(CUDA_NVCC_FLAGS "-ccbin ${CMAKE_C_COMPILER}" CACHE STRING "nvcc flags" FORCE)
eigen/unsupported/test/CMakeLists.txt:  if(EIGEN_TEST_CUDA_CLANG)
eigen/unsupported/test/CMakeLists.txt:    string(APPEND CMAKE_CXX_FLAGS " --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")
eigen/unsupported/test/CMakeLists.txt:    foreach(ARCH IN LISTS EIGEN_CUDA_COMPUTE_ARCH)
eigen/unsupported/test/CMakeLists.txt:        string(APPEND CMAKE_CXX_FLAGS " --cuda-gpu-arch=sm_${ARCH}")
eigen/unsupported/test/CMakeLists.txt:  set(EIGEN_CUDA_RELAXED_CONSTEXPR "--expt-relaxed-constexpr")
eigen/unsupported/test/CMakeLists.txt:  if (${CUDA_VERSION} STREQUAL "7.0")
eigen/unsupported/test/CMakeLists.txt:    set(EIGEN_CUDA_RELAXED_CONSTEXPR "--relaxed-constexpr")
eigen/unsupported/test/CMakeLists.txt:    set(EIGEN_CUDA_CXX11_FLAG "-std=c++11")
eigen/unsupported/test/CMakeLists.txt:    set(EIGEN_CUDA_CXX11_FLAG "")
eigen/unsupported/test/CMakeLists.txt:  foreach(ARCH IN LISTS EIGEN_CUDA_COMPUTE_ARCH)
eigen/unsupported/test/CMakeLists.txt:  set(CUDA_NVCC_FLAGS  "${EIGEN_CUDA_CXX11_FLAG} ${EIGEN_CUDA_RELAXED_CONSTEXPR} -Xcudafe \"--display_error_number\" ${NVCC_ARCH_FLAGS} ${CUDA_NVCC_FLAGS}")
eigen/unsupported/test/CMakeLists.txt:  cuda_include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/include")
eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_complex_gpu)
eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_complex_cwise_ops_gpu)
eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_reduction_gpu)
eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_argmax_gpu)
eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_cast_float16_gpu)
eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_scan_gpu)
eigen/unsupported/test/CMakeLists.txt:  set(EIGEN_CUDA_OLDEST_COMPUTE_ARCH 9999)
eigen/unsupported/test/CMakeLists.txt:  foreach(ARCH IN LISTS EIGEN_CUDA_COMPUTE_ARCH)
eigen/unsupported/test/CMakeLists.txt:    if(${ARCH} LESS ${EIGEN_CUDA_OLDEST_COMPUTE_ARCH})
eigen/unsupported/test/CMakeLists.txt:      set(EIGEN_CUDA_OLDEST_COMPUTE_ARCH ${ARCH})
eigen/unsupported/test/CMakeLists.txt:  if (${EIGEN_CUDA_OLDEST_COMPUTE_ARCH} GREATER 29)
eigen/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_gpu)
eigen/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_contract_gpu)
eigen/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_of_float16_gpu)
eigen/unsupported/test/CMakeLists.txt:  if (${EIGEN_CUDA_OLDEST_COMPUTE_ARCH} GREATER 34)
eigen/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_random_gpu)
eigen/unsupported/test/CMakeLists.txt:  set(HIP_PATH "/opt/rocm/hip" CACHE STRING "Path to the HIP installation.")
eigen/unsupported/test/CMakeLists.txt:	# ei_add_test(cxx11_tensor_complex_gpu)
eigen/unsupported/test/CMakeLists.txt:	# ei_add_test(cxx11_tensor_complex_cwise_ops_gpu)
eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_reduction_gpu)
eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_argmax_gpu)
eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_cast_float16_gpu)
eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_scan_gpu)
eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_gpu)
eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_contract_gpu)
eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_of_float16_gpu)
eigen/unsupported/test/CMakeLists.txt:	ei_add_test(cxx11_tensor_random_gpu)
eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  DataType* gpu_in_data  = static_cast<DataType*>(sycl_device.allocate(in.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  sycl_device.memset(gpu_in_data, 1, in.size()*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  sycl_device.memcpyDeviceToHost(in.data(), gpu_in_data, in.size()*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  DataType* gpu_data = static_cast<DataType*>(sycl_device.allocate(sizeDim1*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  sycl_device.memset(gpu_data, 1, sizeDim1*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  TensorMap<Tensor<DataType, 1, DataLayout,IndexType>> in(gpu_data, tensorDims);
eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  TensorMap<Tensor<DataType, 1, DataLayout,IndexType>> out(gpu_data, tensorDims);
eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  sycl_device.deallocate(gpu_data);
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor1.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  TensorMap<Tensor<DataType, 3, ColMajor, IndexType>> gpu1(gpu_data1, tensorColRange);
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  TensorMap<Tensor<DataType, 3, RowMajor, IndexType>> gpu2(gpu_data2, tensorRowRange);
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor1.data(),(tensor1.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  gpu2.device(sycl_device)=gpu1.swap_layout();
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor2.data(), gpu_data2,(tensor2.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.deallocate(gpu_data1);
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.deallocate(gpu_data2);
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor1.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  TensorMap<Tensor<DataType, 3, ColMajor, IndexType>> gpu1(gpu_data1, tensorColRange);
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  TensorMap<Tensor<DataType, 3, RowMajor, IndexType>> gpu2(gpu_data2, tensorRowRange);
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor1.data(),(tensor1.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  gpu2.swap_layout().device(sycl_device)=gpu1;
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor2.data(), gpu_data2,(tensor2.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.deallocate(gpu_data1);
eigen/unsupported/test/cxx11_tensor_layout_swap_sycl.cpp:  sycl_device.deallocate(gpu_data2);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:#define EIGEN_USE_GPU
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:void test_gpu_simple_argmax()
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuMalloc((void**)(&d_in), in_bytes);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuMalloc((void**)(&d_out_max), out_bytes);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuMalloc((void**)(&d_out_min), out_bytes);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuMemcpy(d_in, in.data(), in_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<double, 3, Layout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 3>(72,53,97));
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 1, Layout>, Aligned > gpu_out_max(d_out_max, Eigen::array<DenseIndex, 1>(1));
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 1, Layout>, Aligned > gpu_out_min(d_out_min, Eigen::array<DenseIndex, 1>(1));
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpu_out_max.device(gpu_device) = gpu_in.argmax();
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpu_out_min.device(gpu_device) = gpu_in.argmin();
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  assert(gpuMemcpyAsync(out_max.data(), d_out_max, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  assert(gpuMemcpyAsync(out_min.data(), d_out_min, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuFree(d_in);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuFree(d_out_max);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  gpuFree(d_out_min);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:void test_gpu_argmax_dim()
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMalloc((void**)(&d_in), in_bytes);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMemcpy(d_in, tensor.data(), in_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 4>(2, 3, 5, 7));
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout>, Aligned > gpu_out(d_out, out_shape);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpu_out.device(gpu_device) = gpu_in.argmax(dim);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuMemcpyAsync(tensor_arg.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMemcpy(d_in, tensor.data(), in_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpu_out.device(gpu_device) = gpu_in.argmax(dim);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuMemcpyAsync(tensor_arg.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuFree(d_in);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:void test_gpu_argmin_dim()
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMalloc((void**)(&d_in), in_bytes);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMalloc((void**)(&d_out), out_bytes);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMemcpy(d_in, tensor.data(), in_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 4>(2, 3, 5, 7));
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout>, Aligned > gpu_out(d_out, out_shape);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpu_out.device(gpu_device) = gpu_in.argmin(dim);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuMemcpyAsync(tensor_arg.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuMemcpy(d_in, tensor.data(), in_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpu_out.device(gpu_device) = gpu_in.argmin(dim);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuMemcpyAsync(tensor_arg.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuFree(d_in);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:    gpuFree(d_out);
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_argmax_gpu)
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  CALL_SUBTEST_1(test_gpu_simple_argmax<RowMajor>());
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  CALL_SUBTEST_1(test_gpu_simple_argmax<ColMajor>());
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  CALL_SUBTEST_2(test_gpu_argmax_dim<RowMajor>());
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  CALL_SUBTEST_2(test_gpu_argmax_dim<ColMajor>());
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  CALL_SUBTEST_3(test_gpu_argmin_dim<RowMajor>());
eigen/unsupported/test/cxx11_tensor_argmax_gpu.cu:  CALL_SUBTEST_3(test_gpu_argmin_dim<ColMajor>());
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:#define EIGEN_USE_GPU
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:void test_gpu_contraction(int m_size, int k_size, int n_size)
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  // a 15 SM GK110 GPU
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  Tensor<float, 2, DataLayout> t_result_gpu(m_size, n_size);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMalloc((void**)(&d_t_left), t_left_bytes);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMalloc((void**)(&d_t_right), t_right_bytes);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMalloc((void**)(&d_t_result), t_result_bytes);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMemcpy(d_t_left, t_left.data(), t_left_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMemcpy(d_t_right, t_right.data(), t_right_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      gpu_t_left(d_t_left, Eigen::array<int, 2>(m_size, k_size));
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      gpu_t_right(d_t_right, Eigen::array<int, 2>(k_size, n_size));
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      gpu_t_result(d_t_result, Eigen::array<int, 2>(m_size, n_size));
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, gpuMemcpyDeviceToHost);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    if (fabs(t_result(i) - t_result_gpu(i)) < 1e-4f) {
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), 1e-4f)) {
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:              << " vs " <<  t_result_gpu(i) << std::endl;
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuFree((void*)d_t_left);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuFree((void*)d_t_right);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuFree((void*)d_t_result);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  // a 15 SM GK110 GPU
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  Tensor<float, 0, DataLayout> t_result_gpu;
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMalloc((void**)(&d_t_left), t_left_bytes);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMalloc((void**)(&d_t_right), t_right_bytes);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMalloc((void**)(&d_t_result), t_result_bytes);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMemcpy(d_t_left, t_left.data(), t_left_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMemcpy(d_t_right, t_right.data(), t_right_bytes, gpuMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      gpu_t_left(d_t_left, m_size, k_size);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      gpu_t_right(d_t_right, k_size, n_size);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      gpu_t_result(d_t_result);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, gpuMemcpyDeviceToHost);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  if (fabs(t_result() - t_result_gpu()) > 1e-4f &&
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:      !Eigen::internal::isApprox(t_result(), t_result_gpu(), 1e-4f)) {
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:              << " vs " <<  t_result_gpu() << std::endl;
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuFree((void*)d_t_left);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuFree((void*)d_t_right);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  gpuFree((void*)d_t_result);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:void test_gpu_contraction_m() {
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    test_gpu_contraction<ColMajor>(k, 128, 128);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    test_gpu_contraction<RowMajor>(k, 128, 128);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:void test_gpu_contraction_k() {
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    test_gpu_contraction<ColMajor>(128, k, 128);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    test_gpu_contraction<RowMajor>(128, k, 128);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:void test_gpu_contraction_n() {
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    test_gpu_contraction<ColMajor>(128, 128, k);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:    test_gpu_contraction<RowMajor>(128, 128, k);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:void test_gpu_contraction_sizes() {
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:        test_gpu_contraction<DataLayout>(m_sizes[i], n_sizes[j], k_sizes[k]);
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_contract_gpu)
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_1(test_gpu_contraction<ColMajor>(128, 128, 128));
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_1(test_gpu_contraction<RowMajor>(128, 128, 128));
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_2(test_gpu_contraction_m<ColMajor>());
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_3(test_gpu_contraction_m<RowMajor>());
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_4(test_gpu_contraction_k<ColMajor>());
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_5(test_gpu_contraction_k<RowMajor>());
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_6(test_gpu_contraction_n<ColMajor>());
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_7(test_gpu_contraction_n<RowMajor>());
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_8(test_gpu_contraction_sizes<ColMajor>());
eigen/unsupported/test/cxx11_tensor_contract_gpu.cu:  CALL_SUBTEST_9(test_gpu_contraction_sizes<RowMajor>());
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  // a 15 SM GK110 GPU
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Tensor<DataType, 2, DataLayout, IndexType> t_result_gpu(m_size, n_size);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_left(d_t_left, left_dims);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_right(d_t_right, right_dims);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_result(d_t_result, result_dims);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result, t_result_bytes);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:    if (static_cast<DataType>(fabs(t_result(i) - t_result_gpu(i))) < error_threshold) {
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), error_threshold)) {
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:              << " vs " <<  t_result_gpu(i) << std::endl;
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Tensor<DataType, 2, DataLayout, IndexType> t_result_gpu(res_dims);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_left(d_t_left, left_dims);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_right(d_t_right, right_dims);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_result(d_t_result, res_dims);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result, t_result_bytes);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:    if (static_cast<DataType>(fabs(t_result(i) - t_result_gpu(i))) < error_threshold) {
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), error_threshold)) {
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:              << " vs " <<  t_result_gpu(i) << std::endl;
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  // a 15 SM GK110 GPU
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Tensor<DataType, 0, DataLayout, IndexType> t_result_gpu;
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_left(d_t_left, left_dims);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_right(d_t_right, right_dims);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 0, DataLayout, IndexType> > gpu_t_result(d_t_result);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result, t_result_bytes);
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:  if (static_cast<DataType>(fabs(t_result() - t_result_gpu())) > error_threshold &&
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:      !Eigen::internal::isApprox(t_result(), t_result_gpu(), error_threshold)) {
eigen/unsupported/test/cxx11_tensor_contract_sycl.cpp:              << " vs " <<  t_result_gpu() << std::endl;
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:#define EIGEN_USE_GPU
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_numext() {
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  bool* d_res_half = (bool*)gpu_device.allocate(num_elem * sizeof(bool));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  bool* d_res_float = (bool*)gpu_device.allocate(num_elem * sizeof(bool));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_res_half(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_res_float(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float.unaryExpr(Eigen::internal::scalar_isnan_op<float>());
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = gpu_float.cast<Eigen::half>().unaryExpr(Eigen::internal::scalar_isnan_op<Eigen::half>());
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(bool));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(bool));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:#ifdef EIGEN_HAS_GPU_FP16
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_conversion() {
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_conv = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_half(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_conv(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float.device(gpu_device) = gpu_float.random();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_half.device(gpu_device) = gpu_float.cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_conv.device(gpu_device) = gpu_half.cast<float>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(initial.data(), d_float, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(final.data(), d_conv, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_half);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_conv);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_unary() {
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_half = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float.abs();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().cast<float>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_elementwise() {
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_half = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float1(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float2(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float1.device(gpu_device) = gpu_float1.random();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float2.device(gpu_device) = gpu_float2.random();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = (gpu_float1 + gpu_float2) * gpu_float1;
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = ((gpu_float1.cast<Eigen::half>() + gpu_float2.cast<Eigen::half>()) * gpu_float1.cast<Eigen::half>()).cast<float>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float1);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float2);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_trancendental() {
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float3 = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res1_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res1_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res2_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res2_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res3_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res3_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float1(d_float1, num_elem);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float2(d_float2, num_elem);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float3(d_float3, num_elem);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res1_half(d_res1_half, num_elem);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res1_float(d_res1_float, num_elem);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res2_half(d_res2_half, num_elem);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res2_float(d_res2_float, num_elem);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res3_half(d_res3_half, num_elem);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res3_float(d_res3_float, num_elem);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res4_half(d_res3_half, num_elem);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res4_float(d_res3_float, num_elem);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float1.device(gpu_device) = gpu_float1.random() - gpu_float1.constant(0.5f);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float2.device(gpu_device) = gpu_float2.random() + gpu_float1.constant(0.5f);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float3.device(gpu_device) = gpu_float3.random();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res1_float.device(gpu_device) = gpu_float1.exp().cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res2_float.device(gpu_device) = gpu_float2.log().cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res3_float.device(gpu_device) = gpu_float3.log1p().cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res4_float.device(gpu_device) = gpu_float3.expm1().cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res1_half.device(gpu_device) = gpu_float1.cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res1_half.device(gpu_device) = gpu_res1_half.exp();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res2_half.device(gpu_device) = gpu_float2.cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res2_half.device(gpu_device) = gpu_res2_half.log();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res3_half.device(gpu_device) = gpu_float3.cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res3_half.device(gpu_device) = gpu_res3_half.log1p();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res3_half.device(gpu_device) = gpu_float3.cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res3_half.device(gpu_device) = gpu_res3_half.expm1();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(input1.data(), d_float1, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(input2.data(), d_float2, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(input3.data(), d_float3, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec1.data(), d_res1_half, num_elem*sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec1.data(), d_res1_float, num_elem*sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec2.data(), d_res2_half, num_elem*sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec2.data(), d_res2_float, num_elem*sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec3.data(), d_res3_half, num_elem*sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec3.data(), d_res3_float, num_elem*sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float1);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float2);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float3);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res1_half);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res1_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res2_half);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res2_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res3_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res3_half);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_contractions() {
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2>, Eigen::Aligned> gpu_res_half(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2>, Eigen::Aligned> gpu_res_float(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float1.device(gpu_device) = gpu_float1.random() - gpu_float1.constant(0.5f);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float2.device(gpu_device) = gpu_float2.random() - gpu_float2.constant(0.5f);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float1.contract(gpu_float2, dims).cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().contract(gpu_float2.cast<Eigen::half>(), dims);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float1);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float2);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_reductions(int size1, int size2, int redux) {
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(result_size * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(result_size * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res_half(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res_float(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float1.device(gpu_device) = gpu_float1.random() * 2.0f;
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float2.device(gpu_device) = gpu_float2.random() * 2.0f;
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float1.sum(redux_dim).cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().sum(redux_dim);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, result_size*sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, result_size*sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float1);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float2);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_reductions() {
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  test_gpu_reductions<void>(13, 13, 0);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  test_gpu_reductions<void>(13, 13, 1);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  test_gpu_reductions<void>(35, 36, 0);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  test_gpu_reductions<void>(35, 36, 1);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  test_gpu_reductions<void>(36, 35, 0);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  test_gpu_reductions<void>(36, 35, 1);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_full_reductions() {
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(1 * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(1 * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 0>, Eigen::Aligned> gpu_res_half(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 0>, Eigen::Aligned> gpu_res_float(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float1.device(gpu_device) = gpu_float1.random();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float2.device(gpu_device) = gpu_float2.random();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float1.sum().cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().sum();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float1.maximum().cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().maximum();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float1);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float2);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:void test_gpu_forced_evals() {
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_half1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_half2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half1(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu: Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Unaligned> gpu_res_half2(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_float.device(gpu_device) = gpu_float.abs();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half1.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().eval().cast<float>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_res_half2.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().broadcast(no_bcast).eval().cast<float>();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec1.data(), d_res_half1, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(half_prec2.data(), d_res_half1, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half1);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_half2);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  gpu_device.deallocate(d_res_float);
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_of_float16_gpu)
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_1(test_gpu_numext<void>());
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:#ifdef EIGEN_HAS_GPU_FP16
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_1(test_gpu_conversion<void>());
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_1(test_gpu_unary<void>());
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_1(test_gpu_elementwise<void>());
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_1(test_gpu_trancendental<void>());
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_2(test_gpu_contractions<void>());
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_3(test_gpu_reductions<void>());
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_4(test_gpu_full_reductions<void>());
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  CALL_SUBTEST_5(test_gpu_forced_evals<void>());
eigen/unsupported/test/cxx11_tensor_of_float16_gpu.cu:  std::cout << "Half floats are not supported by this version of gpu: skipping the test" << std::endl;
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:#define EIGEN_USE_GPU
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:void test_gpu_conversion() {
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  Eigen::half* d_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  float* d_conv = (float*)gpu_device.allocate(num_elem * sizeof(float));
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_half(
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_conv(
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.memcpyHostToDevice(d_float, floats.data(), num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_half.device(gpu_device) = gpu_float.cast<Eigen::half>();
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_conv.device(gpu_device) = gpu_half.cast<float>();
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(initial.data(), d_float, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.memcpyDeviceToHost(final.data(), d_conv, num_elem*sizeof(float));
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.deallocate(d_float);
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.deallocate(d_half);
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  gpu_device.deallocate(d_conv);
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:EIGEN_DECLARE_TEST(cxx11_tensor_cast_float16_gpu)
eigen/unsupported/test/cxx11_tensor_cast_float16_gpu.cu:  CALL_SUBTEST(test_gpu_conversion());
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<DataType, 0, DataLayout, IndexType> full_redux_gpu;
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_in_data = static_cast<DataType*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_out_data =(DataType*)sycl_device.allocate(sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout, IndexType> >  in_gpu(gpu_in_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 0, DataLayout, IndexType> >  out_gpu(gpu_out_data);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.mean();
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data, sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<DataType, 0, DataLayout, IndexType> full_redux_gpu;
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_in_data = static_cast<DataType*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_out_data =(DataType*)sycl_device.allocate(sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout, IndexType> >  in_gpu(gpu_in_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 0, DataLayout, IndexType> >  out_gpu(gpu_out_data);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.minimum();
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data, sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<DataType, 2, DataLayout, IndexType> redux_gpu(reduced_tensorRange);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_in_data = static_cast<DataType*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_out_data = static_cast<DataType*>(sycl_device.allocate(redux_gpu.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType> >  in_gpu(gpu_in_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout, IndexType> >  out_gpu(gpu_out_data, reduced_tensorRange);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.maximum(red_axis);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(redux_gpu.data(), gpu_out_data, redux_gpu.dimensions().TotalSize()*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:      VERIFY_IS_APPROX(redux_gpu(j,k), redux(j,k));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<DataType, 2, DataLayout, IndexType> redux_gpu(reduced_tensorRange);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_in_data = static_cast<DataType*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  DataType* gpu_out_data = static_cast<DataType*>(sycl_device.allocate(redux_gpu.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 3, DataLayout, IndexType> >  in_gpu(gpu_in_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout, IndexType> >  out_gpu(gpu_out_data, reduced_tensorRange);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.sum(red_axis);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(redux_gpu.data(), gpu_out_data, redux_gpu.dimensions().TotalSize()*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:      VERIFY_IS_APPROX(redux_gpu(j,k), redux(j,k));
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  DataType* gpu_data_no_stride  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_no_stride(gpu_data_no_stride, tensorRange);
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  gpu_no_stride.device(sycl_device)=gpu_tensor.inflate(strides);
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  sycl_device.memcpyDeviceToHost(no_stride.data(), gpu_data_no_stride, tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  DataType* gpu_data_inflated  = static_cast<DataType*>(sycl_device.allocate(inflatedTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>> gpu_inflated(gpu_data_inflated, inflatedTensorRange);
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  gpu_inflated.device(sycl_device)=gpu_tensor.inflate(strides);
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  sycl_device.memcpyDeviceToHost(inflated.data(), gpu_data_inflated, inflatedTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor);
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  sycl_device.deallocate(gpu_data_no_stride);
eigen/unsupported/test/cxx11_tensor_inflation_sycl.cpp:  sycl_device.deallocate(gpu_data_inflated);
eigen/unsupported/test/openglsupport.cpp:    #ifdef GLEW_ARB_gpu_shader_fp64
eigen/unsupported/test/openglsupport.cpp:    if(GLEW_ARB_gpu_shader_fp64)
eigen/unsupported/test/openglsupport.cpp:      #ifdef GL_ARB_gpu_shader_fp64
eigen/unsupported/test/openglsupport.cpp:      std::cerr << "Warning: GLEW_ARB_gpu_shader_fp64 was not tested\n";
eigen/unsupported/test/openglsupport.cpp:      std::cerr << "Warning: GLEW_ARB_gpu_shader_fp64 was not tested\n";
eigen/unsupported/test/cxx11_tensor_of_strings.cpp:  // Beware: none of this is likely to ever work on a GPU.
eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, Layout, DenseIndex> > gpu_in(d_in, Eigen::array<DenseIndex, 3>{{2,2,2}});
eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 0, Layout, DenseIndex> > gpu_out_max(d_out_max);
eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 0, Layout, DenseIndex> > gpu_out_min(d_out_min);
eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:  gpu_out_max.device(sycl_device) = gpu_in.argmax();
eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:  gpu_out_min.device(sycl_device) = gpu_in.argmin();
eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, DenseIndex> > gpu_in(d_in, Eigen::array<DenseIndex, 4>{{sizeDim0,sizeDim1,sizeDim2,sizeDim3}});
eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout, DenseIndex> > gpu_out(d_out, out_shape);
eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    gpu_out.device(sycl_device) = gpu_in.argmax(dim);
eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    gpu_out.device(sycl_device) = gpu_in.argmax(dim);
eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    Eigen::TensorMap<Eigen::Tensor<DataType, 4, DataLayout, DenseIndex> > gpu_in(d_in, Eigen::array<DenseIndex, 4>{{sizeDim0,sizeDim1,sizeDim2,sizeDim3}});
eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout, DenseIndex> > gpu_out(d_out, out_shape);
eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    gpu_out.device(sycl_device) = gpu_in.argmin(dim);
eigen/unsupported/test/cxx11_tensor_argmax_sycl.cpp:    gpu_out.device(sycl_device) = gpu_in.argmin(dim);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_single_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_single_patch_col_major(gpu_data_single_patch_col_major, patchColMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_single_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(1, 1);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_patch_col_major.data(), gpu_data_single_patch_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_single_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_single_patch_row_major(gpu_data_single_patch_row_major, patchRowMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_single_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(1, 1);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_patch_row_major.data(), gpu_data_single_patch_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_entire_image_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_entire_image_patch_col_major(gpu_data_entire_image_patch_col_major, patchColMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_entire_image_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(3, 5);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(entire_image_patch_col_major.data(), gpu_data_entire_image_patch_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_entire_image_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_entire_image_patch_row_major(gpu_data_entire_image_patch_row_major, patchRowMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_entire_image_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(3, 5);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(entire_image_patch_row_major.data(), gpu_data_entire_image_patch_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_twod_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_twod_patch_col_major(gpu_data_twod_patch_col_major, patchColMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_twod_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(2, 2);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(twod_patch_col_major.data(), gpu_data_twod_patch_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_twod_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_twod_patch_row_major(gpu_data_twod_patch_row_major, patchRowMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_twod_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(2, 2);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(twod_patch_row_major.data(), gpu_data_twod_patch_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_row_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_patch_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_patch_row_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_entire_image_patch_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_entire_image_patch_row_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_twod_patch_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_twod_patch_row_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_result_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_result_col_major(gpu_data_result_col_major, patchColMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_result_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(ksize, ksize, stride, stride, 1, 1, PADDING_VALID);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(result_col_major.data(), gpu_data_result_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_result_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_result_row_major(gpu_data_result_row_major, patchRowMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_result_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(ksize, ksize, stride, stride, 1, 1, PADDING_VALID);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(result_row_major.data(), gpu_data_result_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_row_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_result_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_result_row_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_col_major.device(sycl_device)=gpu_col_major.constant(11.0f);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor_col_major.data(), gpu_data_col_major, (tensor_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_row_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_result_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_result_col_major(gpu_data_result_col_major, patchColMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_result_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(ksize, ksize, stride, stride, 1, 1, PADDING_VALID);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(result_col_major.data(), gpu_data_result_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_result_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_result_row_major(gpu_data_result_row_major, patchRowMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_result_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(ksize, ksize, stride, stride, 1, 1, PADDING_VALID);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(result_row_major.data(), gpu_data_result_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:DataType* gpu_data_result_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_result_col_major(gpu_data_result_col_major, patchColMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:gpu_result_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(ksize, ksize, stride, stride, PADDING_SAME);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:sycl_device.memcpyDeviceToHost(result_col_major.data(), gpu_data_result_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_result_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_result_row_major(gpu_data_result_row_major, patchRowMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_result_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(ksize, ksize, stride, stride, PADDING_SAME);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(result_row_major.data(), gpu_data_result_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 3, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 3, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_row_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_single_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_single_patch_col_major(gpu_data_single_patch_col_major, patchColMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_single_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(1, 1);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_patch_col_major.data(), gpu_data_single_patch_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_single_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, RowMajor,IndexType>> gpu_single_patch_row_major(gpu_data_single_patch_row_major, patchRowMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_single_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(1, 1);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_patch_row_major.data(), gpu_data_single_patch_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_entire_image_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_entire_image_patch_col_major(gpu_data_entire_image_patch_col_major, patchColMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_entire_image_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(3, 5);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(entire_image_patch_col_major.data(), gpu_data_entire_image_patch_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:DataType* gpu_data_entire_image_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:TensorMap<Tensor<DataType, 4, RowMajor,IndexType>> gpu_entire_image_patch_row_major(gpu_data_entire_image_patch_row_major, patchRowMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:gpu_entire_image_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(3, 5);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:sycl_device.memcpyDeviceToHost(entire_image_patch_row_major.data(), gpu_data_entire_image_patch_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_twod_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_twod_patch_col_major(gpu_data_twod_patch_col_major, patchColMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_twod_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(2, 2);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(twod_patch_col_major.data(), gpu_data_twod_patch_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_twod_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, RowMajor,IndexType>> gpu_twod_patch_row_major(gpu_data_twod_patch_row_major, patchRowMajorTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_twod_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(2, 2);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(twod_patch_row_major.data(), gpu_data_twod_patch_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_row_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_patch_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_patch_row_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_entire_image_patch_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_entire_image_patch_row_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_twod_patch_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_twod_patch_row_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_l_in_col_major  = static_cast<DataType*>(sycl_device.allocate(l_in_col_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_l_in_col_major(gpu_data_l_in_col_major, tensorColMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_l_in_col_major, l_in_col_major.data(),(l_in_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_l_out_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_l_out_col_major(gpu_data_l_out_col_major, patchTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_col_major.device(sycl_device)=gpu_l_in_col_major.extract_image_patches(11, 11);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_col_major.data(), gpu_data_l_out_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  DataType* gpu_data_l_out_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_l_out_row_major(gpu_data_l_out_row_major, patchTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_row_major.device(sycl_device)=gpu_l_in_col_major.swap_layout().extract_image_patches(11, 11);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_row_major.data(), gpu_data_l_out_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_in_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_in_col_major  = static_cast<DataType*>(sycl_device.allocate(l_in_col_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>>gpu_l_in_col_major_resize1(gpu_data_l_in_col_major, tensorColMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_out_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>>gpu_l_out_col_major_resize1(gpu_data_l_out_col_major, patchTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_l_in_col_major, l_in_col_major.data(),(l_in_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_col_major_resize1.device(sycl_device)=gpu_l_in_col_major_resize1.extract_image_patches(9, 9);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_col_major.data(), gpu_data_l_out_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_row_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_out_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>>gpu_l_out_row_major_resize1(gpu_data_l_out_row_major, patchTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_row_major_resize1.device(sycl_device)=gpu_l_in_col_major_resize1.swap_layout().extract_image_patches(9, 9);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_row_major.data(), gpu_data_l_out_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_in_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_in_col_major  = static_cast<DataType*>(sycl_device.allocate(l_in_col_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>>gpu_l_in_col_major_resize2(gpu_data_l_in_col_major, tensorColMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_out_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>>gpu_l_out_col_major_resize2(gpu_data_l_out_col_major, patchTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_l_in_col_major, l_in_col_major.data(),(l_in_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_col_major_resize2.device(sycl_device)=gpu_l_in_col_major_resize2.extract_image_patches(7, 7);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_col_major.data(), gpu_data_l_out_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_row_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_out_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>>gpu_l_out_row_major_resize2(gpu_data_l_out_row_major, patchTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_row_major_resize2.device(sycl_device)=gpu_l_in_col_major_resize2.swap_layout().extract_image_patches(7, 7);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_row_major.data(), gpu_data_l_out_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_in_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_in_col_major  = static_cast<DataType*>(sycl_device.allocate(l_in_col_major.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>>gpu_l_in_col_major_resize3(gpu_data_l_in_col_major, tensorColMajorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_out_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>>gpu_l_out_col_major_resize3(gpu_data_l_out_col_major, patchTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_l_in_col_major, l_in_col_major.data(),(l_in_col_major.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_col_major_resize3.device(sycl_device)=gpu_l_in_col_major_resize3.extract_image_patches(3, 3);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_col_major.data(), gpu_data_l_out_col_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_row_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_data_l_out_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>>gpu_l_out_row_major_resize3(gpu_data_l_out_row_major, patchTensorRange);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  gpu_l_out_row_major_resize3.device(sycl_device)=gpu_l_in_col_major_resize3.swap_layout().extract_image_patches(3, 3);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(l_out_row_major.data(), gpu_data_l_out_row_major, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_in_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_col_major);
eigen/unsupported/test/cxx11_tensor_image_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_l_out_row_major);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:#define EIGEN_USE_GPU
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:void test_cuda_nullary() {
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaMalloc((void**)(&d_in1), complex_bytes);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaMalloc((void**)(&d_in2), complex_bytes);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaMalloc((void**)(&d_out2), float_bytes);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaMemcpy(d_in1, in1.data(), complex_bytes, cudaMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaMemcpy(d_in2, in2.data(), complex_bytes, cudaMemcpyHostToDevice);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1, 0, int>, Eigen::Aligned> gpu_in1(
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1, 0, int>, Eigen::Aligned> gpu_in2(
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_out2(
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_in1.device(gpu_device) = gpu_in1.constant(std::complex<float>(3.14f, 2.7f));
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_out2.device(gpu_device) = gpu_in2.abs();
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  assert(cudaMemcpyAsync(new1.data(), d_in1, complex_bytes, cudaMemcpyDeviceToHost,
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:                         gpu_device.stream()) == cudaSuccess);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  assert(cudaMemcpyAsync(new2.data(), d_out2, float_bytes, cudaMemcpyDeviceToHost,
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:                         gpu_device.stream()) == cudaSuccess);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaFree(d_in1);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaFree(d_in2);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  cudaFree(d_out2);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:static void test_cuda_sum_reductions() {
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  std::complex<float>* gpu_in_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(in_bytes));
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  std::complex<float>* gpu_out_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(out_bytes));
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  TensorMap<Tensor<std::complex<float>, 2> > in_gpu(gpu_in_ptr, num_rows, num_cols);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  TensorMap<Tensor<std::complex<float>, 0> > out_gpu(gpu_out_ptr);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  out_gpu.device(gpu_device) = in_gpu.sum();
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Tensor<std::complex<float>, 0> full_redux_gpu;
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  // Check that the CPU and GPU reductions return the same result.
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.deallocate(gpu_in_ptr);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.deallocate(gpu_out_ptr);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:static void test_cuda_mean_reductions() {
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  std::complex<float>* gpu_in_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(in_bytes));
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  std::complex<float>* gpu_out_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(out_bytes));
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  TensorMap<Tensor<std::complex<float>, 2> > in_gpu(gpu_in_ptr, num_rows, num_cols);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  TensorMap<Tensor<std::complex<float>, 0> > out_gpu(gpu_out_ptr);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  out_gpu.device(gpu_device) = in_gpu.mean();
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Tensor<std::complex<float>, 0> full_redux_gpu;
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  // Check that the CPU and GPU reductions return the same result.
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.deallocate(gpu_in_ptr);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.deallocate(gpu_out_ptr);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:static void test_cuda_product_reductions() {
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  std::complex<float>* gpu_in_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(in_bytes));
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  std::complex<float>* gpu_out_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(out_bytes));
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  TensorMap<Tensor<std::complex<float>, 2> > in_gpu(gpu_in_ptr, num_rows, num_cols);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  TensorMap<Tensor<std::complex<float>, 0> > out_gpu(gpu_out_ptr);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  out_gpu.device(gpu_device) = in_gpu.prod();
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  Tensor<std::complex<float>, 0> full_redux_gpu;
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.synchronize();
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  // Check that the CPU and GPU reductions return the same result.
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.deallocate(gpu_in_ptr);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  gpu_device.deallocate(gpu_out_ptr);
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  CALL_SUBTEST(test_cuda_nullary());
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  CALL_SUBTEST(test_cuda_sum_reductions());
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  CALL_SUBTEST(test_cuda_mean_reductions());
eigen/unsupported/test/cxx11_tensor_complex_gpu.cu:  CALL_SUBTEST(test_cuda_product_reductions());
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:#define EIGEN_USE_GPU
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:void test_cuda_complex_cwise_ops() {
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  cudaMalloc((void**)(&d_in1), complex_bytes);
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  cudaMalloc((void**)(&d_in2), complex_bytes);
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  cudaMalloc((void**)(&d_out), complex_bytes);
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  Eigen::GpuStreamDevice stream;
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  Eigen::GpuDevice gpu_device(&stream);
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_in1(
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_in2(
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_out(
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  gpu_in1.device(gpu_device) = gpu_in1.constant(a);
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  gpu_in2.device(gpu_device) = gpu_in2.constant(b);
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:        gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:        gpu_out.device(gpu_device) = gpu_in1 - gpu_in2;
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:        gpu_out.device(gpu_device) = gpu_in1 * gpu_in2;
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:        gpu_out.device(gpu_device) = gpu_in1 / gpu_in2;
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:        gpu_out.device(gpu_device) = -gpu_in1;
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:    assert(cudaMemcpyAsync(actual.data(), d_out, complex_bytes, cudaMemcpyDeviceToHost,
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:                           gpu_device.stream()) == cudaSuccess);
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  cudaFree(d_in1);
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  cudaFree(d_in2);
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  cudaFree(d_out);
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  CALL_SUBTEST(test_cuda_complex_cwise_ops<float>());
eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_gpu.cu:  CALL_SUBTEST(test_cuda_complex_cwise_ops<double>());
eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(buffSize));
eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(buffSize));
eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu1(gpu_data1, tensorRange);
eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu2(gpu_data2, tensorRange);
eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(), buffSize);
eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  gpu2.device(sycl_device)=gpu1.shuffle(shuffles);
eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  sycl_device.memcpyDeviceToHost(no_shuffle.data(), gpu_data2, buffSize);
eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(buffSize));
eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  TensorMap<Tensor<DataType, 4,DataLayout,IndexType>> gpu3(gpu_data3, tensorrangeShuffle);
eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  gpu3.device(sycl_device)=gpu1.shuffle(shuffles);
eigen/unsupported/test/cxx11_tensor_shuffling_sycl.cpp:  sycl_device.memcpyDeviceToHost(shuffle.data(), gpu_data3, buffSize);
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data = static_cast<SCALAR *>(                                  \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu(gpu_data, tensorRange);          \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data, in.data(),                        \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_out, out.data(),                   \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    gpu_out.device(sycl_device) OPERATOR gpu.FUNC();                           \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data);                                          \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_out);                                      \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_out, out.data(),                   \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    gpu_out.device(sycl_device) OPERATOR gpu_out.FUNC();                       \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_out);                                      \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data = static_cast<SCALAR *>(                                  \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    bool *gpu_data_out =                                                       \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu(gpu_data, tensorRange);          \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<bool, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);    \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data, in.data(),                        \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    gpu_out.device(sycl_device) = gpu.FUNC();                                  \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data);                                          \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_out);                                      \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_1 = static_cast<SCALAR *>(                                \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_2 = static_cast<SCALAR *>(                                \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_1(gpu_data_1, tensorRange);      \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_2(gpu_data_2, tensorRange);      \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_1, in_1.data(),                    \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_2, in_2.data(),                    \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    gpu_out.device(sycl_device) = gpu_1.FUNC(gpu_2);                           \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_1);                                        \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_2);                                        \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_out);                                      \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_1 = static_cast<SCALAR *>(                                \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_2 = static_cast<SCALAR *>(                                \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_1(gpu_data_1, tensorRange);      \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_2(gpu_data_2, tensorRange);      \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_1, in_1.data(),                    \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_2, in_2.data(),                    \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    gpu_out.device(sycl_device) = gpu_1 OPERATOR gpu_2;                        \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_1);                                        \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_2);                                        \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_out);                                      \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_1 = static_cast<SCALAR *>(                                \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_1(gpu_data_1, tensorRange);      \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyHostToDevice(gpu_data_1, in_1.data(),                    \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    gpu_out.device(sycl_device) = gpu_1 OPERATOR 2;                            \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_1);                                        \
eigen/unsupported/test/cxx11_tensor_builtins_sycl.cpp:    sycl_device.deallocate(gpu_data_out);                                      \
eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(in2.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_in1(gpu_in1_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_in2(gpu_in2_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType>> gpu_out(gpu_out_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in1_data, in1.data(),(in1.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in2.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  gpu_out.device(sycl_device) =(gpu_in1 + gpu_in2).eval() * gpu_in2;
eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.deallocate(gpu_in1_data);
eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.deallocate(gpu_in2_data);
eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  DataType* gpu_data_no_patch  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_no_patch(gpu_data_no_patch, patchTensorRange);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  gpu_no_patch.device(sycl_device)=gpu_tensor.extract_patches(patch_dims);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(no_patch.data(), gpu_data_no_patch, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  DataType* gpu_data_single_patch  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_single_patch(gpu_data_single_patch, patchTensorRange);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  gpu_single_patch.device(sycl_device)=gpu_tensor.extract_patches(patch_dims);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(single_patch.data(), gpu_data_single_patch, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  DataType* gpu_data_twod_patch  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_twod_patch(gpu_data_twod_patch, patchTensorRange);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  gpu_twod_patch.device(sycl_device)=gpu_tensor.extract_patches(patch_dims);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(twod_patch.data(), gpu_data_twod_patch, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  DataType* gpu_data_threed_patch  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_threed_patch(gpu_data_threed_patch, patchTensorRange);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  gpu_threed_patch.device(sycl_device)=gpu_tensor.extract_patches(patch_dims);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.memcpyDeviceToHost(threed_patch.data(), gpu_data_threed_patch, patchTensorBuffSize);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_tensor);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_no_patch);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_single_patch);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_twod_patch);
eigen/unsupported/test/cxx11_tensor_patch_sycl.cpp:  sycl_device.deallocate(gpu_data_threed_patch);
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  DataType* gpu_data_vec  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  DataType* gpu_data_result  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  TensorMap<Tensor<DataType, 1, DataLayout,IndexType>> gpu_vec(gpu_data_vec, tensorRange);
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  TensorMap<Tensor<DataType, 1, DataLayout,IndexType>> gpu_result(gpu_data_result, tensorRange);
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_vec, vec.data(), tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  gpu_result.device(sycl_device)=gpu_vec.generate(Generator1D());
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  sycl_device.memcpyDeviceToHost(result.data(), gpu_data_result, tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  DataType* gpu_data_matrix  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  DataType* gpu_data_result  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout,IndexType>> gpu_matrix(gpu_data_matrix, tensorRange);
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout,IndexType>> gpu_result(gpu_data_result, tensorRange);
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_matrix, matrix.data(), tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  gpu_result.device(sycl_device)=gpu_matrix.generate(Generator2D());
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  sycl_device.memcpyDeviceToHost(result.data(), gpu_data_result, tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  DataType* gpu_data_matrix  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  DataType* gpu_data_result  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout,IndexType>> gpu_matrix(gpu_data_matrix, tensorRange);
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  TensorMap<Tensor<DataType, 2, DataLayout,IndexType>> gpu_result(gpu_data_result, tensorRange);
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data_matrix, matrix.data(), tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  gpu_result.device(sycl_device)=gpu_matrix.generate(gaussian_gen);
eigen/unsupported/test/cxx11_tensor_generator_sycl.cpp:  sycl_device.memcpyDeviceToHost(result.data(), gpu_data_result, tensorBuffSize);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor1.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(tensor3.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data4  = static_cast<DataType*>(sycl_device.allocate(tensor4.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu1(gpu_data1, dim1);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 3,DataLayout, IndexType>> gpu2(gpu_data2, dim2);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 2,DataLayout, IndexType>> gpu3(gpu_data3, dim3);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 2,DataLayout, IndexType>> gpu4(gpu_data4, dim4);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor1.data(),(tensor1.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu2.device(sycl_device)=gpu1.reshape(dim2);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor2.data(), gpu_data2,(tensor1.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu3.device(sycl_device)=gpu1.reshape(dim3);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor3.data(), gpu_data3,(tensor3.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu4.device(sycl_device)=gpu1.reshape(dim2).reshape(dim4);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor4.data(), gpu_data4,(tensor4.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data1);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data2);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data3);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data4);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2d.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(tensor5d.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap< Tensor<DataType, 3, DataLayout, IndexType> > gpu1(gpu_data1, dim1);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap< Tensor<DataType, 2, DataLayout, IndexType> > gpu2(gpu_data2, dim2);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap< Tensor<DataType, 5, DataLayout, IndexType> > gpu3(gpu_data3, dim3);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu2.reshape(dim1).device(sycl_device)=gpu1;
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor2d.data(), gpu_data2,(tensor2d.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu3.reshape(dim1).device(sycl_device)=gpu1;
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor5d.data(), gpu_data3,(tensor5d.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data1);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data2);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data3);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(slice1.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu1(gpu_data1, tensorRange);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu2(gpu_data2, slice1_range);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu2.device(sycl_device)=gpu1.slice(indices, sizes);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(slice1.data(), gpu_data2,(slice1.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(slice2.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu3(gpu_data3, slice2_range);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu3.device(sycl_device)=gpu1.slice(indices2, sizes2);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(slice2.data(), gpu_data3,(slice2.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data1);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data2);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data3);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(slice.size()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 2,DataLayout,IndexType>> gpu1(gpu_data1, tensorRange);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 2,DataLayout,IndexType>> gpu2(gpu_data2, tensorRange);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  TensorMap<Tensor<DataType, 2,DataLayout,IndexType>> gpu3(gpu_data3, sliceRange);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu2.device(sycl_device)=gpu1;
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_data3, slice.data(),(slice.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu1.slice(indicesStart,lengths).device(sycl_device)=gpu3;
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  gpu2.stridedSlice(indicesStart,indicesStop,strides).device(sycl_device)=gpu3;
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data1,(tensor.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.memcpyDeviceToHost(tensor2.data(), gpu_data2,(tensor2.size())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data1);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data2);
eigen/unsupported/test/cxx11_tensor_morphing_sycl.cpp:  sycl_device.deallocate(gpu_data3);
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  TensorType gpu_in1(gpu_in1_data, tensorRange);
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  TensorType gpu_out(gpu_out_data, tensorResultRange);
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in1_data, in1.data(),(in1.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1.customOp(InsertZeros<TensorType>());
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(in2.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  TensorType gpu_in1(gpu_in1_data, tensorRange1);
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  TensorType gpu_in2(gpu_in2_data, tensorRange2);
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  TensorType gpu_out(gpu_out_data, tensorResultRange);
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in1_data, in1.data(),(in1.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in2.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1.customOp(gpu_in2, BatchMatMul<TensorType>());
eigen/unsupported/test/cxx11_tensor_custom_op_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));
eigen/unsupported/Eigen/CXX11/Tensor:#ifdef EIGEN_USE_GPU
eigen/unsupported/Eigen/CXX11/Tensor:    #include <cuda_runtime.h>
eigen/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorDeviceGpu.h"
eigen/unsupported/Eigen/CXX11/Tensor:#ifndef gpu_assert
eigen/unsupported/Eigen/CXX11/Tensor:#define gpu_assert(x)
eigen/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorReductionGpu.h"
eigen/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorContractionGpu.h"
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:  gpu_assert(threadIdx.z == 0);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:    //the same for all the thread. As unlike CUDA, the thread.ID, BlockID, etc is not a global function.
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:    // and only  available on the Operator() function (which is called on the GPU).
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:    // Thus for CUDA (((CLOCK  + global_thread_id)* 6364136223846793005ULL) + 0xda3e39cb94b95bdbULL) is passed to each thread 
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:    // similar to CUDA Therefore, the thread Id injection is not available at this stage. 
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:    //to the seed and construct the unique m_state per thead similar to cuda.  
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:      // The (i * 6364136223846793005ULL) is the remaining part of the PCG_XSH_RS_state on the GPU side
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:    //the same for all the thread. As unlike CUDA, the thread.ID, BlockID, etc is not a global function.
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:    // and only  available on the Operator() function (which is called on the GPU).
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:    //to the seed and construct the unique m_state per thead similar to cuda.  
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:// GPU: the evaluation of the expression is offloaded to a GPU.
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#if defined(EIGEN_USE_GPU)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:class TensorExecutor<Expression, GpuDevice, Vectorizable, Tiling> {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:  static void run(const Expression& expr, const GpuDevice& device);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#if defined(EIGEN_GPUCC)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:EIGEN_STRONG_INLINE void TensorExecutor<Expression, GpuDevice, Vectorizable, Tiling>::run(
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const Expression& expr, const GpuDevice& device) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:  TensorEvaluator<Expression, GpuDevice> evaluator(expr, device);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const int block_size = device.maxGpuThreadsPerBlock();
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const int max_blocks = device.getNumGpuMultiProcessors() *
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:                           device.maxGpuThreadsPerMultiProcessor() / block_size;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    LAUNCH_GPU_KERNEL(
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:        (EigenMetaKernel<TensorEvaluator<Expression, GpuDevice>, StorageIndex>),
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#endif  // EIGEN_GPUCC
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#endif  // EIGEN_USE_GPU
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef EIGEN_GPU_COMPILE_PHASE
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef EIGEN_GPU_COMPILE_PHASE
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__SIZEOF_INT128__) && !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(SYCL_DEVICE_ONLY)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef EIGEN_GPU_COMPILE_PHASE
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#if !defined(EIGEN_GPU_COMPILE_PHASE)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(SYCL_DEVICE_ONLY)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device, return the amount of shared memory available.
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(SYCL_DEVICE_ONLY)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#if !defined(EIGEN_GPU_COMPILE_PHASE)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    return EIGEN_CUDA_ARCH / 100;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_GPU_H
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_GPU_H
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:#if defined(EIGEN_USE_GPU) && defined(EIGEN_GPUCC)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:#if defined(EIGEN_HIPCC) || (defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER < 90000)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    #if defined(EIGEN_HIPCC) || (defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER < 90000)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType, OutputKernelType>, GpuDevice> :
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    public TensorContractionEvaluatorBase<TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType, OutputKernelType>, GpuDevice> > {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:  typedef GpuDevice Device;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:  typedef typename PacketType<CoeffReturnType, GpuDevice>::type PacketReturnType;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:                          GPU_TENSOR_CONTRACTION_DOES_NOT_SUPPORT_OUTPUT_KERNELS);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    static void Run(const LhsMapper& lhs, const RhsMapper& rhs, const OutputMapper& output, Index m, Index n, Index k, const GpuDevice& device) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    LAUNCH_GPU_KERNEL((EigenContractionKernel<Scalar, Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    static void Run(const LhsMapper& lhs, const RhsMapper& rhs, const OutputMapper& output, Index m, Index n, Index k, const GpuDevice& device) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:        LAUNCH_GPU_KERNEL((EigenFloatContractionKernel16x16<Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:        LAUNCH_GPU_KERNEL((EigenFloatContractionKernel<Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    setGpuSharedMemConfig(hipSharedMemBankSizeEightByte);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:    setGpuSharedMemConfig(cudaSharedMemBankSizeEightByte);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:#endif // EIGEN_USE_GPU and EIGEN_GPUCC
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h:#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_GPU_H
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceSycl.h:// All devices (even AMD CPU with intel OpenCL runtime) that support OpenCL and 
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceSycl.h:    // OpenCL doesnot have such concept
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceSycl.h:    // OpenCL doesnot have such concept
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#if defined(EIGEN_CXX11_TENSOR_GPU_HIP_CUDA_DEFINES_H)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuStream_t
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuDeviceProp_t 
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuError_t
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuSuccess
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuErrorNotReady
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuGetDeviceCount
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuGetErrorString
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuGetDeviceProperties
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuStreamDefault
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuGetDevice
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuSetDevice
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMalloc
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuFree
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMemsetAsync
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMemcpyAsync
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMemcpyDeviceToDevice
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMemcpyDeviceToHost
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMemcpyHostToDevice
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuStreamQuery
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuSharedMemConfig
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuDeviceSetSharedMemConfig
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuStreamSynchronize
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuDeviceSynchronize
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef gpuMemcpy
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#undef EIGEN_CXX11_TENSOR_GPU_HIP_CUDA_DEFINES_H
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaUndefines.h:#endif // EIGEN_CXX11_TENSOR_GPU_HIP_CUDA_DEFINES_H
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_GPU_H
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_GPU_H
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#if defined(EIGEN_USE_GPU) && defined(EIGEN_GPUCC)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:// Full reducers for GPU, don't vectorize for now
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:// Reducer function that enables multiple gpu thread to safely accumulate at the same
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:// attempts to update it with the new value. If in the meantime another gpu thread
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#if (defined(EIGEN_HIP_DEVICE_COMPILE) && defined(__HIP_ARCH_HAS_WARP_SHUFFLE__)) || (EIGEN_CUDA_ARCH >= 300)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(0 && "Wordsize not supported");
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#else // EIGEN_CUDA_ARCH >= 300
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  gpu_assert(0 && "Shouldn't be called on unsupported device");
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_CUDA_ARCH >= 300
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#if (defined(EIGEN_HIP_DEVICE_COMPILE) && defined(__HIP_ARCH_HAS_WARP_SHUFFLE__)) || (EIGEN_CUDA_ARCH >= 300)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#else // EIGEN_CUDA_ARCH >= 300
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  gpu_assert(0 && "Shouldn't be called on unsupported device");
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_CUDA_ARCH >= 300
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#if (defined(EIGEN_HIP_DEVICE_COMPILE) && defined(__HIP_ARCH_HAS_WARP_SHUFFLE__)) || (EIGEN_CUDA_ARCH >= 300)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  #elif defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER < 90000
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#else // EIGEN_CUDA_ARCH >= 300
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  gpu_assert(0 && "Shouldn't be called on unsupported device");
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_CUDA_ARCH >= 300
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  #elif defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER < 90000
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static void run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(false && "Should only be called on doubles, floats and half floats");
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    LAUNCH_GPU_KERNEL((FullReductionKernel<block_size, num_per_thread, Self, Op, Index>),
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static void run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(false && "Should not be called since there is no packet accessor");
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      LAUNCH_GPU_KERNEL((ReductionInitFullReduxKernelHalfFloat<Self, Op, Index>),
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    LAUNCH_GPU_KERNEL((FullReductionKernelHalfFloat<block_size, num_per_thread, Self, Op, Index>),
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      LAUNCH_GPU_KERNEL((ReductionCleanupKernelHalfFloat<Op>),
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:struct FullReducer<Self, Op, GpuDevice, Vectorizable> {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#else // EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(HasOptimizedImplementation && "Should only be called on doubles, floats or half floats");
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    FullReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#if (defined(EIGEN_HIP_DEVICE_COMPILE) && defined(__HIP_ARCH_HAS_WARP_SHUFFLE__)) || (EIGEN_CUDA_ARCH >= 300)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      #elif defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER < 90000
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#else // EIGEN_CUDA_ARCH >= 300
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  gpu_assert(0 && "Shouldn't be called on unsupported device");
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_CUDA_ARCH >= 300
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      #elif defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER < 90000
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static EIGEN_DEVICE_FUNC bool run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index, typename Self::Index) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(false && "Should only be called to reduce doubles, floats and half floats on a gpu device");
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    const int max_blocks = device.getNumGpuMultiProcessors() *
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:                           device.maxGpuThreadsPerMultiProcessor() / block_size;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      const int max_blocks = device.getNumGpuMultiProcessors() *
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:                           device.maxGpuThreadsPerMultiProcessor() / 1024;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      LAUNCH_GPU_KERNEL((ReductionInitKernel<OutputType, Index>),
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    LAUNCH_GPU_KERNEL((InnerReductionKernel<num_per_thread, Self, Op, Index>),
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static bool run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index, typename Self::Index) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(false && "Should not be called since there is no packet accessor");
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    const int max_blocks = device.getNumGpuMultiProcessors() *
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:                           device.maxGpuThreadsPerMultiProcessor() / block_size;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      LAUNCH_GPU_KERNEL((ReductionInitKernelHalfFloat<Self, Op, Index>),
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    LAUNCH_GPU_KERNEL((InnerReductionKernelHalfFloat<num_per_thread, Self, Op, Index>),
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:struct InnerReducer<Self, Op, GpuDevice> {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#ifdef EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#else // EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_HAS_GPU_FP16
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(HasOptimizedImplementation && "Should only be called on doubles, floats or half floats");
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    return InnerReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs_to_reduce, num_preserved_vals);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:struct OuterReducer<Self, Op, GpuDevice> {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    //          (in the cxx11_tensor_reduction_gpu test)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    gpu_assert(false && "Should only be called to reduce doubles or floats on a gpu device");
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, float* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    const int max_blocks = device.getNumGpuMultiProcessors() *
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:                           device.maxGpuThreadsPerMultiProcessor() / block_size;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      const int max_blocks = device.getNumGpuMultiProcessors() *
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:                             device.maxGpuThreadsPerMultiProcessor() / 1024;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:      LAUNCH_GPU_KERNEL((ReductionInitKernel<float, Index>),
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:    LAUNCH_GPU_KERNEL((OuterReductionKernel<num_per_thread, Self, Op, Index>),
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // defined(EIGEN_USE_GPU) && defined(EIGEN_GPUCC)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:#endif // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_GPU_H
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDevice.h:  * on the specified computing 'device' (GPU, thread pool, ...)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDevice.h:  *    C.device(EIGEN_GPU) = A + B;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#warning "Deprecated header file, please either include the main Eigen/CXX11/Tensor header or the respective TensorReductionGpu.h file"
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#include "TensorReductionGpu.h"
eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:GPU using cuda.  Additional implementations may be added later.
eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:GpuDevice.
eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:#### Evaluating On GPU
eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:You need to create a GPU device but you also need to explicitly allocate the
eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:memory for tensors with cuda.
eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:*   On GPUs only floating point values are properly tested and optimized for.
eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:*   Complex and integer values are known to be broken on GPUs. If you try to use
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:// It is very expensive to start the memcpy kernel on GPU: we therefore only
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:#ifdef EIGEN_USE_GPU
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:template <typename Index, bool BlockAccess> struct MemcpyTriggerForSlicing<Index, GpuDevice, BlockAccess>  {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:  EIGEN_DEVICE_FUNC MemcpyTriggerForSlicing(const GpuDevice&) { }
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:// It is very expensive to start the memcpy kernel on GPU: we therefore only
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:// For CUDA packet types when using a GpuDevice
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:#if defined(EIGEN_USE_GPU) && defined(EIGEN_HAS_GPU_FP16)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:struct PacketType<half, GpuDevice> {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:// Can't use std::pairs on cuda devices
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorSycl.h:// this is used to change the address space type in tensor map for GPU
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorFFT.h:  * Improve the performance on GPU
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:struct GpuDevice;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:struct IsVectorizable<GpuDevice, Expression> {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:  static const bool value = TensorEvaluator<Expression, GpuDevice>::PacketAccess &&
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:                            TensorEvaluator<Expression, GpuDevice>::IsAligned;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#warning "Deprecated header file, please either include the main Eigen/CXX11/Tensor header or the respective TensorContractionGpu.h file"
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#include "TensorContractionGpu.h"
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMacros.h:#if !defined(EIGEN_GPUCC)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#if defined(EIGEN_USE_GPU) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_GPU_H)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_GPU_H
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:// This header file container defines fo gpu* macros which will resolve to
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:// their equivalent hip* or cuda* versions depending on the compiler in use
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#include "TensorGpuHipCudaDefines.h"
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:static const int kGpuScratchSize = 1024;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:// This defines an interface that GPUDevice can take to use
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:// HIP / CUDA streams underneath.
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  virtual const gpuStream_t& stream() const = 0;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  virtual const gpuDeviceProp_t& deviceProperties() const = 0;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:static gpuDeviceProp_t* m_deviceProperties;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpuError_t status = gpuGetDeviceCount(&num_devices);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      if (status != gpuSuccess) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:        std::cerr << "Failed to get the number of GPU devices: "
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:                  << gpuGetErrorString(status)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:        gpu_assert(status == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      m_deviceProperties = new gpuDeviceProp_t[num_devices];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:        status = gpuGetDeviceProperties(&m_deviceProperties[i], i);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:        if (status != gpuSuccess) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:          std::cerr << "Failed to initialize GPU device #"
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:                    << gpuGetErrorString(status)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:          gpu_assert(status == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:static const gpuStream_t default_stream = gpuStreamDefault;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:class GpuStreamDevice : public StreamInterface {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  GpuStreamDevice() : stream_(&default_stream), scratch_(NULL), semaphore_(NULL) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuGetDevice(&device_);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  GpuStreamDevice(int device) : stream_(&default_stream), device_(device), scratch_(NULL), semaphore_(NULL) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  // assumes that the stream is associated to the current gpu device.
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  GpuStreamDevice(const gpuStream_t* stream, int device = -1)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpuGetDevice(&device_);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpuError_t err = gpuGetDeviceCount(&num_devices);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpu_assert(err == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpu_assert(device < num_devices);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  virtual ~GpuStreamDevice() {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  const gpuStream_t& stream() const { return *stream_; }
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  const gpuDeviceProp_t& deviceProperties() const {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err = gpuSetDevice(device_);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    err = gpuMalloc(&result, num_bytes);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(result != NULL);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err = gpuSetDevice(device_);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(buffer != NULL);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    err = gpuFree(buffer);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      scratch_ = allocate(kGpuScratchSize + sizeof(unsigned int));
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      char* scratch = static_cast<char*>(scratchpad()) + kGpuScratchSize;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpuError_t err = gpuMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpu_assert(err == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  const gpuStream_t* stream_;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:struct GpuDevice {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  explicit GpuDevice(const StreamInterface* stream) : stream_(stream), max_blocks_(INT_MAX) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  explicit GpuDevice(const StreamInterface* stream, int num_blocks) : stream_(stream), max_blocks_(num_blocks) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  EIGEN_STRONG_INLINE const gpuStream_t& stream() const {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#ifndef EIGEN_GPU_COMPILE_PHASE
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err = gpuMemcpyAsync(dst, src, n, gpuMemcpyDeviceToDevice,
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err =
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:        gpuMemcpyAsync(dst, src, n, gpuMemcpyHostToDevice, stream_->stream());
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err =
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:        gpuMemcpyAsync(dst, src, n, gpuMemcpyDeviceToHost, stream_->stream());
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#ifndef EIGEN_GPU_COMPILE_PHASE
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err = gpuMemsetAsync(buffer, c, n, stream_->stream());
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(err == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    // there is no l3 cache on hip/cuda devices.
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#ifndef EIGEN_GPU_COMPILE_PHASE
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t err = gpuStreamSynchronize(stream_->stream());
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    if (err != gpuSuccess) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      std::cerr << "Error detected in GPU stream: "
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:                << gpuGetErrorString(err)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:      gpu_assert(err == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpu_assert(false && "The default device should be used instead to generate kernel code");
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  EIGEN_STRONG_INLINE int getNumGpuMultiProcessors() const {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  EIGEN_STRONG_INLINE int maxGpuThreadsPerBlock() const {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  EIGEN_STRONG_INLINE int maxGpuThreadsPerMultiProcessor() const {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  // This function checks if the GPU runtime recorded an error for the
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#ifdef EIGEN_GPUCC
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    gpuError_t error = gpuStreamQuery(stream_->stream());
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:    return (error == gpuSuccess) || (error == gpuErrorNotReady);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#define LAUNCH_GPU_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)             \
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  gpu_assert(hipGetLastError() == hipSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#define LAUNCH_GPU_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)             \
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  gpu_assert(cudaGetLastError() == cudaSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#ifdef EIGEN_GPUCC
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:static EIGEN_DEVICE_FUNC inline void setGpuSharedMemConfig(gpuSharedMemConfig config) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#ifndef EIGEN_GPU_COMPILE_PHASE
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  gpuError_t status = gpuDeviceSetSharedMemConfig(config);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:  gpu_assert(status == gpuSuccess);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:// undefine all the gpu* macros we defined at the beginning of the file
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#include "TensorGpuHipCudaUndefines.h"
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h:#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_GPU_H
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionSycl.h:    /// This one is used to collect all the reduced value of shared memory as we don't have global barrier on GPU. Once it is saved we can
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionSycl.h:    /// This one is used to collect all the reduced value of shared memory as we don't have global barrier on GPU. Once it is saved we can
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:// clang is incompatible with the CUDA syntax wrt making a kernel a class friend,
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(__clang__) && (defined(__CUDA__) || defined(__HIP__))
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && (defined(EIGEN_GPUCC))
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_HAS_GPU_FP16)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:        ((RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) ||
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:         !RunningOnGPU))) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    else if ((RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) || (RunningOnSycl)) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    if (( RunningFullReduction || RunningOnGPU) && m_result ) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    if (RunningOnGPU && m_result) {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && (defined(EIGEN_GPUCC))
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_HAS_GPU_FP16)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && (defined(EIGEN_GPUCC))
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:  static const bool RunningOnGPU = internal::is_same<Device, Eigen::GpuDevice>::value;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:static const bool RunningOnGPU = false;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:  static const bool RunningOnGPU = false;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:    const size_t plane_tensor_offset =indexMapper.mapCudaInputPlaneToTensorInputOffset(itemID.get_global(1));
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:      const size_t tensor_index  =  plane_tensor_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i + first_input_start);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:      const size_t tensor_index = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(itemID.get_global(1))
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:      +indexMapper.mapCudaOutputKernelToTensorOutputOffset(itemID.get_local(0) + first_output_start);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:    const size_t plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(itemID.get_global(2));
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:        const size_t tensor_index  = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i + first_x_input_start, j+ first_y_input_start );
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:      const size_t tensor_index = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(itemID.get_global(2))
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:      +indexMapper.mapCudaOutputKernelToTensorOutputOffset(itemID.get_local(0) + fitst_x_output_start, itemID.get_local(1) + fitst_y_output_start);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:      const size_t plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:            const size_t tensor_index  = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i + first_x_input_start, j+ first_y_input_start , k+ first_z_input_start );
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:        const size_t tensor_index = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:        +indexMapper.mapCudaOutputKernelToTensorOutputOffset(itemID.get_local(0) + fitst_x_output_start, itemID.get_local(1) + fitst_y_output_start, itemID.get_local(2) + fitst_z_output_start );
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:          gpu_assert(static_cast<unsigned long>(shared_mem) <= m_device.sharedMemPerBlock());
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:          gpu_assert(static_cast<unsigned long>(shared_mem) <= m_device.sharedMemPerBlock());
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolutionSycl.h:          gpu_assert(static_cast<unsigned long>(shared_mem) <= m_device.sharedMemPerBlock());
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#if defined(EIGEN_USE_GPU) && !defined(EIGEN_CXX11_TENSOR_GPU_HIP_CUDA_DEFINES_H)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define EIGEN_CXX11_TENSOR_GPU_HIP_CUDA_DEFINES_H
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:// There is code in the Tensorflow codebase that will define EIGEN_USE_GPU,  but
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:// for some reason gets sent to the gcc/host compiler instead of the gpu/nvcc/hipcc compiler
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:// When compiling such files, gcc will end up trying to pick up the CUDA headers by 
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:// default (see the code within "unsupported/Eigen/CXX11/Tensor" that is guarded by EIGEN_USE_GPU)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:// This will obsviously not work when trying to compile tensorflow on a system with no CUDA
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStream_t hipStream_t
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuDeviceProp_t hipDeviceProp_t
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuError_t hipError_t
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuSuccess hipSuccess
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuErrorNotReady hipErrorNotReady
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetDeviceCount hipGetDeviceCount
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetErrorString hipGetErrorString
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetDeviceProperties hipGetDeviceProperties
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStreamDefault hipStreamDefault
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetDevice hipGetDevice
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuSetDevice hipSetDevice
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMalloc hipMalloc
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuFree hipFree
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemsetAsync hipMemsetAsync
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyAsync hipMemcpyAsync
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStreamQuery hipStreamQuery
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuSharedMemConfig hipSharedMemConfig
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuDeviceSetSharedMemConfig hipDeviceSetSharedMemConfig
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStreamSynchronize hipStreamSynchronize
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuDeviceSynchronize hipDeviceSynchronize
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpy hipMemcpy
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStream_t cudaStream_t
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuDeviceProp_t cudaDeviceProp
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuError_t cudaError_t
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuSuccess cudaSuccess
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuErrorNotReady cudaErrorNotReady
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetDeviceCount cudaGetDeviceCount
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetErrorString cudaGetErrorString
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetDeviceProperties cudaGetDeviceProperties
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStreamDefault cudaStreamDefault
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuGetDevice cudaGetDevice
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuSetDevice cudaSetDevice
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMalloc cudaMalloc
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuFree cudaFree
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemsetAsync cudaMemsetAsync
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyAsync cudaMemcpyAsync
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStreamQuery cudaStreamQuery
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuSharedMemConfig cudaSharedMemConfig
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuDeviceSetSharedMemConfig cudaDeviceSetSharedMemConfig
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuStreamSynchronize cudaStreamSynchronize
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuDeviceSynchronize cudaDeviceSynchronize
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpuMemcpy cudaMemcpy
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:// gpu_assert can be overridden
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#ifndef gpu_assert
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:// HIPCC do not support the use of assert on the GPU side.
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpu_assert(COND)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#define gpu_assert(COND) assert(COND)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#endif // gpu_assert
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h:#endif  // EIGEN_CXX11_TENSOR_GPU_HIP_CUDA_DEFINES_H
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#warning "Deprecated header file, please either include the main Eigen/CXX11/Tensor header or the respective TensorDeviceGpu.h file"
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#include "TensorDeviceGpu.h"
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    array<Index, NumDims> gpuInputDimensions;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    array<Index, NumDims> gpuOutputDimensions;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      gpuInputDimensions[index] = input_dims[indices[i]];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      gpuOutputDimensions[index] = dimensions[indices[i]];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        gpuInputDimensions[written] = input_dims[i];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        gpuOutputDimensions[written] = dimensions[i];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuInputStrides[i] =
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_gpuInputStrides[i - 1] * gpuInputDimensions[i - 1];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuOutputStrides[i] =
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_gpuOutputStrides[i - 1] * gpuOutputDimensions[i - 1];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuInputStrides[i] = 1;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuOutputStrides[i] = 1;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuInputStrides[i] =
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_gpuInputStrides[i + 1] * gpuInputDimensions[i + 1];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuOutputStrides[i] =
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_gpuOutputStrides[i + 1] * gpuOutputDimensions[i + 1];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuInputStrides[i] = 1;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_gpuOutputStrides[i] = 1;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuInputPlaneToTensorInputOffset(Index p) const {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_gpuInputStrides[d];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_gpuInputStrides[d];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_gpuInputStrides[d];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_gpuInputStrides[d];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuOutputPlaneToTensorOutputOffset(Index p) const {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_gpuOutputStrides[d];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_gpuOutputStrides[d];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_gpuOutputStrides[d];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_gpuOutputStrides[d];
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuInputKernelToTensorInputOffset(Index i) const {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuOutputKernelToTensorOutputOffset(Index i) const {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuInputKernelToTensorInputOffset(Index i, Index j) const {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuOutputKernelToTensorOutputOffset(Index i, Index j) const {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuInputKernelToTensorInputOffset(Index i, Index j, Index k) const {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapGpuOutputKernelToTensorOutputOffset(Index i, Index j, Index k) const {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  array<Index, NumDims> m_gpuInputStrides;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  array<Index, NumDims> m_gpuOutputStrides;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:// Use an optimized implementation of the evaluation code for GPUs whenever possible.
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:#if defined(EIGEN_USE_GPU) && defined(EIGEN_GPUCC)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapGpuInputPlaneToTensorInputOffset(p);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const int tensor_index = plane_input_offset + indexMapper.mapGpuInputKernelToTensorInputOffset(i+first_x);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapGpuOutputPlaneToTensorOutputOffset(p);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const int tensor_index = plane_output_offset + indexMapper.mapGpuOutputKernelToTensorOutputOffset(i+first_x);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapGpuInputPlaneToTensorInputOffset(p);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const int tensor_index = plane_input_offset + indexMapper.mapGpuInputKernelToTensorInputOffset(i+first_x, j+first_y);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapGpuOutputPlaneToTensorOutputOffset(p);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const int tensor_index = plane_output_offset + indexMapper.mapGpuOutputKernelToTensorOutputOffset(i+first_x, j+first_y);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapGpuInputPlaneToTensorInputOffset(p);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          const int tensor_index = plane_input_offset + indexMapper.mapGpuInputKernelToTensorInputOffset(i+first_x, j+first_y, k+first_z);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapGpuOutputPlaneToTensorOutputOffset(p);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          const int tensor_index = plane_output_offset + indexMapper.mapGpuOutputKernelToTensorOutputOffset(i+first_x, j+first_y, k+first_z);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:struct TensorEvaluator<const TensorConvolutionOp<Indices, InputArgType, KernelArgType>, GpuDevice>
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  static const int NumDims =  internal::array_size<typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions>::value;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  typedef typename TensorEvaluator<KernelArgType, GpuDevice>::Dimensions KernelDimensions;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    IsAligned = TensorEvaluator<InputArgType, GpuDevice>::IsAligned & TensorEvaluator<KernelArgType, GpuDevice>::IsAligned,
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    Layout = TensorEvaluator<InputArgType, GpuDevice>::Layout,
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const GpuDevice& device)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<InputArgType, GpuDevice>::Layout) == static_cast<int>(TensorEvaluator<KernelArgType, GpuDevice>::Layout)), YOU_MADE_A_PROGRAMMING_MISTAKE);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions& input_dims = m_inputImpl.dimensions();
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const typename TensorEvaluator<KernelArgType, GpuDevice>::Dimensions& kernel_dims = m_kernelImpl.dimensions();
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  typedef typename PacketType<CoeffReturnType, GpuDevice>::type PacketReturnType;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const bool PacketAccess = internal::IsVectorizable<GpuDevice, KernelArgType>::value;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      internal::TensorExecutor<const EvalTo, GpuDevice, PacketAccess>::run(evalToTmp, m_device);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    typedef typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions InputDims;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int maxThreadsPerBlock = m_device.maxGpuThreadsPerBlock();
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int maxBlocksPerProcessor = m_device.maxGpuThreadsPerMultiProcessor() / maxThreadsPerBlock;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int numMultiProcessors = m_device.getNumGpuMultiProcessors();
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        gpu_assert(shared_mem <= maxSharedMem);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_GPU_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, 4, data);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_GPU_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, 7, data);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_GPU_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, kernel_size, data);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        gpu_assert(shared_mem <= maxSharedMem);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_GPU_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4, 7>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 4, 7, data);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_GPU_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 4, kernel_size_y, data);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_GPU_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7, 4>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 7, 4, data);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_GPU_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 7, kernel_size_y, data);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_GPU_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, Dynamic, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, kernel_size_x, kernel_size_y, data);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        gpu_assert(shared_mem <= maxSharedMem);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        LAUNCH_GPU_KERNEL((EigenConvolutionKernel3D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, numZ, maxZ, kernel_size_x, kernel_size_y, kernel_size_z, data);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  TensorEvaluator<InputArgType, GpuDevice> m_inputImpl;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  TensorEvaluator<KernelArgType, GpuDevice> m_kernelImpl;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  const GpuDevice& m_device;
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorBroadcasting.h:        #ifdef EIGEN_GPU_COMPILE_PHASE
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorBroadcasting.h:        // See PR 437: on NVIDIA P100 and K20m we observed a x3-4 speed up by enforcing
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorBroadcasting.h:        #ifdef EIGEN_GPU_COMPILE_PHASE
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:#if defined(EIGEN_USE_GPU) && (defined(EIGEN_GPUCC))
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:// GPU implementation of scan
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:struct ScanLauncher<Self, Reducer, GpuDevice> {
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:     LAUNCH_GPU_KERNEL((ScanKernel<Self, Reducer>), num_blocks, block_size, 0, self.device(), self, total_size, data);
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:#endif  // EIGEN_USE_GPU && (EIGEN_GPUCC)
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:// Use the texture cache on CUDA devices whenever possible
eigen/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
eigen/unsupported/Eigen/CXX11/src/util/EmulateArray.h:// Moreover, CUDA doesn't support the STL containers, so we use our own instead.
eigen/unsupported/Eigen/CXX11/src/util/EmulateArray.h:#if (__cplusplus <= 199711L && EIGEN_COMP_MSVC < 1900) || defined(EIGEN_GPUCC) || defined(EIGEN_AVOID_STL_ARRAY)
eigen/unsupported/Eigen/CXX11/src/util/EmulateArray.h:// The compiler supports c++11, and we're not targeting cuda: use std::array as Eigen::array
eigen/unsupported/Eigen/CXX11/src/util/MaxSizeVector.h:  * std::vector is not an option (e.g. on GPU or when compiling using
eigen/unsupported/Eigen/SpecialFunctions:#if defined EIGEN_VECTORIZE_GPU
eigen/unsupported/Eigen/SpecialFunctions:  #include "src/SpecialFunctions/arch/GPU/GpuSpecialFunctions.h"
eigen/unsupported/Eigen/OpenGLSupport:#ifdef GL_ARB_gpu_shader_fp64
eigen/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__)
eigen/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__)
eigen/unsupported/Eigen/src/SpecialFunctions/arch/GPU/GpuSpecialFunctions.h:#ifndef EIGEN_GPU_SPECIALFUNCTIONS_H
eigen/unsupported/Eigen/src/SpecialFunctions/arch/GPU/GpuSpecialFunctions.h:#define EIGEN_GPU_SPECIALFUNCTIONS_H
eigen/unsupported/Eigen/src/SpecialFunctions/arch/GPU/GpuSpecialFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
eigen/unsupported/Eigen/src/SpecialFunctions/arch/GPU/GpuSpecialFunctions.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
eigen/unsupported/Eigen/src/SpecialFunctions/arch/GPU/GpuSpecialFunctions.h:#endif // EIGEN_GPU_SPECIALFUNCTIONS_H
eigen/bench/tensors/tensor_benchmarks_gpu.cu:#define EIGEN_USE_GPU
eigen/bench/tensors/tensor_benchmarks_gpu.cu:#include <cuda.h>
eigen/bench/tensors/tensor_benchmarks_gpu.cu:#include <cuda_runtime.h>
eigen/bench/tensors/tensor_benchmarks_gpu.cu:#define BM_FuncGPU(FUNC)                                                       \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, float> suite(device, N);                  \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:    cudaDeviceSynchronize();                                                   \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(memcpy);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(typeCasting);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(random);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(slicing);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(rowChip);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(colChip);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(shuffling);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(padding);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(striding);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(broadcasting);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(coeffWiseOp);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(algebraicFunc);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(transcendentalFunc);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(rowReduction);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(colReduction);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(fullReduction);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:#define BM_FuncWithInputDimsGPU(FUNC, D1, D2, D3)                              \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, float> suite(device, D1, D2, D3);         \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:    cudaDeviceSynchronize();                                                   \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, N, N);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithInputDimsGPU(contraction, 64, N, N);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, 64, N);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, N, 64);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:#define BM_FuncWithKernelDimsGPU(FUNC, DIM1, DIM2)                             \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, float> suite(device, N);                  \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:    cudaDeviceSynchronize();                                                   \
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 1);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 1, 7);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 4);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 4, 7);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 64);
eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 64, 7);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:#define BM_FuncGPU(FUNC)                                                       \
eigen/bench/tensors/tensor_benchmarks_sycl.cc:    cl::sycl::gpu_selector selector;                                           \
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(memcpy);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(typeCasting);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(slicing);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(rowChip);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(colChip);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(shuffling);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(padding);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(striding);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(broadcasting);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(coeffWiseOp);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(algebraicFunc);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(transcendentalFunc);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(rowReduction);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(colReduction);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(fullReduction);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:#define BM_FuncWithInputDimsGPU(FUNC, D1, D2, D3)                              \
eigen/bench/tensors/tensor_benchmarks_sycl.cc:    cl::sycl::gpu_selector selector;                                           \
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithInputDimsGPU(contraction, N, N, N);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithInputDimsGPU(contraction, 64, N, N);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithInputDimsGPU(contraction, N, 64, N);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithInputDimsGPU(contraction, N, N, 64);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:#define BM_FuncWithKernelDimsGPU(FUNC, DIM1, DIM2)                             \
eigen/bench/tensors/tensor_benchmarks_sycl.cc:    cl::sycl::gpu_selector selector;                                           \
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithKernelDimsGPU(convolution, 7, 1);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithKernelDimsGPU(convolution, 1, 7);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithKernelDimsGPU(convolution, 7, 4);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithKernelDimsGPU(convolution, 4, 7);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithKernelDimsGPU(convolution, 7, 64);
eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncWithKernelDimsGPU(convolution, 64, 7);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#define EIGEN_USE_GPU
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#include <cuda.h>
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#include <cuda_runtime.h>
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#define BM_FuncGPU(FUNC)                                                       \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, Eigen::half> suite(device, N);            \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    cudaDeviceSynchronize();                                                   \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(memcpy);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(typeCasting);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu://BM_FuncGPU(random);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(slicing);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(rowChip);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(colChip);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(shuffling);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(padding);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(striding);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(broadcasting);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(coeffWiseOp);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(algebraicFunc);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(transcendentalFunc);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(rowReduction);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(colReduction);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(fullReduction);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#define BM_FuncWithInputDimsGPU(FUNC, D1, D2, D3)                              \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, Eigen::half> suite(device, D1, D2, D3);   \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    cudaDeviceSynchronize();                                                   \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, N, N);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithInputDimsGPU(contraction, 64, N, N);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, 64, N);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, N, 64);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#define BM_FuncWithKernelDimsGPU(FUNC, DIM1, DIM2)                             \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, Eigen::half> suite(device, N);            \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    cudaDeviceSynchronize();                                                   \
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 1);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 1, 7);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 4);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 4, 7);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 64);
eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 64, 7);
eigen/bench/tensors/tensor_benchmarks.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
eigen/bench/tensors/tensor_benchmarks.h:    if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
eigen/bench/tensors/README:The first part is a generic suite, in which each benchmark comes in 2 flavors: one that runs on CPU, and one that runs on GPU.
eigen/bench/tensors/README:To compile the floating point GPU benchmarks, simply call:
eigen/bench/tensors/README:nvcc tensor_benchmarks_gpu.cu benchmark_main.cc -I ../../ -std=c++11 -O2 -DNDEBUG -use_fast_math -ftz=true -arch compute_35 -o benchmarks_gpu
eigen/bench/tensors/README:We also provide a version of the generic GPU tensor benchmarks that uses half floats (aka fp16) instead of regular floats. To compile these benchmarks, simply call the command line below. You'll need a recent GPU that supports compute capability 5.3 or higher to run them and nvcc 7.5 or higher to compile the code.
eigen/bench/tensors/README:nvcc tensor_benchmarks_fp16_gpu.cu benchmark_main.cc -I ../../ -std=c++11 -O2 -DNDEBUG -use_fast_math -ftz=true -arch compute_53 -o benchmarks_fp16_gpu
eigen/bench/tensors/README:clang++ -O3 tensor_benchmarks_sycl_include_headers.cc -pthread -I ../../ -I  {ComputeCpp_ROOT}/include/ -L  {ComputeCpp_ROOT}/lib/ -lComputeCpp -lOpenCL -D_GLIBCXX_USE_CXX11_ABI=0 -DEIGEN_USE_SYCL=1 -std=c++11 benchmark_main.o -o tensor_benchmark_sycl
eigen/cmake/FindPastix.cmake:#   - STARPU_CUDA: to activate detection of StarPU with CUDA
eigen/cmake/FindPastix.cmake:set(PASTIX_LOOK_FOR_STARPU_CUDA OFF)
eigen/cmake/FindPastix.cmake:    if (${component} STREQUAL "STARPU_CUDA")
eigen/cmake/FindPastix.cmake:      # means we look for PaStiX with StarPU + CUDA
eigen/cmake/FindPastix.cmake:      set(PASTIX_LOOK_FOR_STARPU_CUDA ON)
eigen/cmake/FindPastix.cmake:  if (PASTIX_LOOK_FOR_STARPU_CUDA)
eigen/cmake/FindPastix.cmake:    list(APPEND STARPU_COMPONENT_LIST "CUDA")
eigen/cmake/FindPastix.cmake:  # CUDA
eigen/cmake/FindPastix.cmake:  if (PASTIX_LOOK_FOR_STARPU_CUDA AND CUDA_FOUND)
eigen/cmake/FindPastix.cmake:    if (CUDA_INCLUDE_DIRS)
eigen/cmake/FindPastix.cmake:      list(APPEND REQUIRED_INCDIRS "${CUDA_INCLUDE_DIRS}")
eigen/cmake/FindPastix.cmake:    foreach(libdir ${CUDA_LIBRARY_DIRS})
eigen/cmake/FindPastix.cmake:    list(APPEND REQUIRED_LIBS "${CUDA_CUBLAS_LIBRARIES};${CUDA_LIBRARIES}")
eigen/cmake/FindPastix.cmake:	"Have you tried with COMPONENTS (MPI/SEQ, STARPU, STARPU_CUDA, SCOTCH, PTSCOTCH, METIS)? "
eigen/cmake/EigenTesting.cmake:    elseif(EIGEN_TEST_CUDA_CLANG)
eigen/cmake/EigenTesting.cmake:      if(CUDA_64_BIT_DEVICE_CODE AND (EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64"))
eigen/cmake/EigenTesting.cmake:        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
eigen/cmake/EigenTesting.cmake:        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib")
eigen/cmake/EigenTesting.cmake:      set(CUDA_CLANG_LINK_LIBRARIES "cudart_static" "cuda" "dl" "pthread")
eigen/cmake/EigenTesting.cmake:      set(CUDA_CLANG_LINK_LIBRARIES ${CUDA_CLANG_LINK_LIBRARIES} "rt")
eigen/cmake/EigenTesting.cmake:      target_link_libraries(${targetname} ${CUDA_CLANG_LINK_LIBRARIES})
eigen/cmake/EigenTesting.cmake:        cuda_add_executable(${targetname} ${filename} OPTIONS ${ARGV2})
eigen/cmake/EigenTesting.cmake:        cuda_add_executable(${targetname} ${filename})
eigen/cmake/EigenTesting.cmake:    if(EIGEN_TEST_CUDA)
eigen/cmake/EigenTesting.cmake:      if(EIGEN_TEST_CUDA_CLANG)
eigen/cmake/EigenTesting.cmake:        message(STATUS "CUDA:              ON (using clang)")
eigen/cmake/EigenTesting.cmake:        message(STATUS "CUDA:              ON (using nvcc)")
eigen/cmake/EigenTesting.cmake:      message(STATUS "CUDA:              OFF")
eigen/cmake/FindBLAS.cmake:##  ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic
eigen/cmake/FindBLAS.cmake:      ((BLA_VENDOR STREQUAL "ACML_GPU") AND (NOT BLAS_ACML_GPU_LIB_DIRS)))
eigen/cmake/FindBLAS.cmake:      file( GLOB _ACML_GPU_ROOT "C:/AMD/acml*/GPGPUexamples" )
eigen/cmake/FindBLAS.cmake:      file( GLOB _ACML_GPU_ROOT "/opt/acml*/GPGPUexamples" )
eigen/cmake/FindBLAS.cmake:    list(GET _ACML_GPU_ROOT 0 _ACML_GPU_ROOT)
eigen/cmake/FindBLAS.cmake:  elseif( BLA_VENDOR STREQUAL "ACML_GPU" )
eigen/cmake/FindBLAS.cmake:    foreach( BLAS_ACML_GPU_LIB_DIRS ${_ACML_GPU_LIB_DIRS})
eigen/cmake/FindBLAS.cmake:	"" "acml;acml_mv;CALBLAS" "" ${BLAS_ACML_GPU_LIB_DIRS}
eigen/cmake/FindBLAS.cmake:	"Intel( older versions of mkl 32 and 64 bit), ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
eigen/cmake/FindBLAS.cmake:	"Intel( older versions of mkl 32 and 64 bit), ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
eigen/cmake/FindBLASEXT.cmake:    ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
eigen/cmake/FindBLASEXT.cmake:    "\n   ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
eigen/cmake/FindComputeCpp.cmake:# Find OpenCL package
eigen/cmake/FindComputeCpp.cmake:find_package(OpenCL REQUIRED)
eigen/cmake/FindComputeCpp.cmake:                        PUBLIC ${OpenCL_LIBRARIES})
eigen/cmake/FindTriSYCL.cmake:option(TRISYCL_OPENCL "triSYCL OpenCL interoperability mode" OFF)
eigen/cmake/FindTriSYCL.cmake:mark_as_advanced(TRISYCL_OPENCL)
eigen/cmake/FindTriSYCL.cmake:# Find OpenCL package
eigen/cmake/FindTriSYCL.cmake:if(TRISYCL_OPENCL)
eigen/cmake/FindTriSYCL.cmake:  find_package(OpenCL REQUIRED)
eigen/cmake/FindTriSYCL.cmake:    $<$<BOOL:${TRISYCL_OPENCL}>:${OpenCL_INCLUDE_DIRS}>
eigen/cmake/FindTriSYCL.cmake:    $<$<BOOL:${TRISYCL_OPENCL}>:${BOOST_COMPUTE_INCPATH}>)
eigen/cmake/FindTriSYCL.cmake:    $<$<BOOL:${TRISYCL_OPENCL}>:${OpenCL_LIBRARIES}>
eigen/cmake/FindTriSYCL.cmake:    $<$<BOOL:${TRISYCL_OPENCL}>:TRISYCL_OPENCL>
eigen/demos/opengl/gpuhelper.cpp:#include "gpuhelper.h"
eigen/demos/opengl/gpuhelper.cpp:GpuHelper gpu;
eigen/demos/opengl/gpuhelper.cpp:GpuHelper::GpuHelper()
eigen/demos/opengl/gpuhelper.cpp:GpuHelper::~GpuHelper()
eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::pushProjectionMode2D(ProjectionMode2D pm)
eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::popProjectionMode2D(void)
eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::drawVector(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect /* = 50.*/)
eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::drawVectorBox(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect)
eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::drawUnitCube(void)
eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::drawUnitSphere(int level)
eigen/demos/opengl/quaternion_demo.h:#include "gpuhelper.h"
eigen/demos/opengl/CMakeLists.txt:  set(quaternion_demo_SRCS  gpuhelper.cpp icosphere.cpp camera.cpp trackball.cpp quaternion_demo.cpp)
eigen/demos/opengl/camera.cpp:#include "gpuhelper.h"
eigen/demos/opengl/camera.cpp:  gpu.loadMatrix(projectionMatrix(),GL_PROJECTION);
eigen/demos/opengl/camera.cpp:  gpu.loadMatrix(viewMatrix().matrix(),GL_MODELVIEW);
eigen/demos/opengl/gpuhelper.h:#ifndef EIGEN_GPUHELPER_H
eigen/demos/opengl/gpuhelper.h:#define EIGEN_GPUHELPER_H
eigen/demos/opengl/gpuhelper.h:class GpuHelper
eigen/demos/opengl/gpuhelper.h:    GpuHelper();
eigen/demos/opengl/gpuhelper.h:    ~GpuHelper();
eigen/demos/opengl/gpuhelper.h:extern GpuHelper gpu;
eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::setMatrixTarget(GLenum matrixTarget)
eigen/demos/opengl/gpuhelper.h:void GpuHelper::multMatrix(const Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
eigen/demos/opengl/gpuhelper.h:void GpuHelper::loadMatrix(
eigen/demos/opengl/gpuhelper.h:void GpuHelper::loadMatrix(const Eigen::Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::pushMatrix(GLenum matrixTarget)
eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::pushMatrix(const Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
eigen/demos/opengl/gpuhelper.h:void GpuHelper::pushMatrix(
eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::popMatrix(GLenum matrixTarget)
eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, uint nofElement)
eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, const std::vector<uint>* pIndexes)
eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, uint start, uint end)
eigen/demos/opengl/gpuhelper.h:#endif // EIGEN_GPUHELPER_H
eigen/demos/opengl/quaternion_demo.cpp:        gpu.pushMatrix(GL_MODELVIEW);
eigen/demos/opengl/quaternion_demo.cpp:        gpu.multMatrix(t.matrix(),GL_MODELVIEW);
eigen/demos/opengl/quaternion_demo.cpp:        gpu.popMatrix(GL_MODELVIEW);
eigen/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitX(), Color(1,0,0,1));
eigen/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitY(), Color(0,1,0,1));
eigen/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitZ(), Color(0,0,1,1));
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:// Handle NVCC/CUDA/SYCL
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:#if defined(__CUDACC__) || defined(__SYCL_DEVICE_ONLY__)
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:  // Do not try asserts on CUDA and SYCL!
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:  // All functions callable from CUDA code must be qualified with __device__
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:  #ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:    // Do not try to vectorize on CUDA and SYCL!
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:// When compiling CUDA device code with NVCC, pull in math functions from the
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:#if defined(__CUDA_ARCH__) && defined(__NVCC__)
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL)
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:#if defined __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:  #define EIGEN_VECTORIZE_CUDA
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:  #if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:    #define EIGEN_HAS_CUDA_FP16
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:#if defined EIGEN_HAS_CUDA_FP16
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:  #include <cuda_fp16.h>
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:#include "src/Core/arch/CUDA/Half.h"
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:#include "src/Core/arch/CUDA/PacketMathHalf.h"
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:#include "src/Core/arch/CUDA/TypeCasting.h"
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:#if defined EIGEN_VECTORIZE_CUDA
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:  #include "src/Core/arch/CUDA/PacketMath.h"
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:  #include "src/Core/arch/CUDA/MathFunctions.h"
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:// on CUDA devices
celerite/cpp/lib/eigen_3.3.3/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Meta.h:  static float (max)() { return CUDART_MAX_NORMAL_F; }
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Meta.h:  static float infinity() { return CUDART_INF_F; }
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Meta.h:  static float quiet_NaN() { return CUDART_NAN_F; }
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Meta.h:  static double infinity() { return CUDART_INF; }
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Meta.h:  static double quiet_NaN() { return CUDART_NAN; }
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Macros.h:  && ( !defined(__NVCC__) || !EIGEN_ARCH_ARM_OR_ARM64 || (defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000) )
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Macros.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Macros.h:#if EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && defined(__CUDACC_VER__) && (EIGEN_COMP_CLANG || __CUDACC_VER__ >= 70500))
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Macros.h:#if (defined __CUDACC__)
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Macros.h:#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC < 1900 ||  defined(__CUDACC_VER__)) // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/util/Macros.h:#  ifdef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifndef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#if (!defined(__CUDACC__)) && EIGEN_FAST_MATH
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/MatrixBase.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/GeneralProduct.h:#ifndef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/GeneralProduct.h:#endif // __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/GenericPacketMath.h:#ifdef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/GenericPacketMath.h:#ifndef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMath.h:#ifndef EIGEN_PACKET_MATH_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMath.h:#define EIGEN_PACKET_MATH_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMath.h:#endif // EIGEN_PACKET_MATH_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:// type Eigen::half (inheriting from CUDA's __half struct) with
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:// to disk and the likes), but fast on GPUs.
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#ifndef EIGEN_HALF_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#define EIGEN_HALF_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#if !defined(EIGEN_HAS_CUDA_FP16)
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:// Make our own __half definition that is similar to CUDA's.
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:  #if !defined(EIGEN_HAS_CUDA_FP16)
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:// Definitions for CPUs and older CUDA, mostly working through conversion
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__)
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Half.h:#endif // EIGEN_HALF_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#ifndef EIGEN_PACKET_MATH_HALF_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#define EIGEN_PACKET_MATH_HALF_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 530
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#endif // EIGEN_PACKET_MATH_HALF_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#define EIGEN_TYPE_CASTING_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Complex.h:// building for CUDA to avoid non-constexpr methods.
celerite/cpp/lib/eigen_3.3.3/Eigen/src/Core/arch/CUDA/Complex.h:#endif // EIGEN_COMPLEX_CUDA_H
celerite/cpp/lib/eigen_3.3.3/Eigen/src/SVD/BDCSVD.h:#ifndef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  float * gpu_in1_data  = static_cast<float*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  float * gpu_in2_data  = static_cast<float*>(sycl_device.allocate(in2.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  float * gpu_in3_data  = static_cast<float*>(sycl_device.allocate(in3.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  float * gpu_out_data =  static_cast<float*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<float, 3>> gpu_in1(gpu_in1_data, tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<float, 3>> gpu_in2(gpu_in2_data, tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<float, 3>> gpu_in3(gpu_in3_data, tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<float, 3>> gpu_out(gpu_out_data, tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_in1.device(sycl_device) = gpu_in1.constant(1.2f);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(in1.data(), gpu_in1_data ,(in1.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 * 1.2f;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data ,(out.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in2.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 * gpu_in2;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 + gpu_in2;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 * gpu_in1;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) =  gpu_in1 * gpu_in1.constant(3.14f) + gpu_in2 * gpu_in2.constant(2.7f);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(),gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in3_data, in3.data(),(in3.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) =(gpu_in1 > gpu_in1.constant(0.5f)).select(gpu_in2, gpu_in3);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_in1_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_in2_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_in3_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_sycl.cpp:  cl::sycl::gpu_selector s;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_random_cuda
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:#define EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:#include <cuda_fp16.h>
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:void test_cuda_random_uniform()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  gpu_out.device(gpu_device) = gpu_out.random();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:void test_cuda_random_normal()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  gpu_out.device(gpu_device) = gpu_out.random(gen);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:void test_cxx11_tensor_random_cuda()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  CALL_SUBTEST(test_cuda_random_uniform());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_random_cuda.cu:  CALL_SUBTEST(test_cuda_random_normal());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_scan_cuda
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:#define EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:#include <cuda_fp16.h>
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:void test_cuda_cumsum(int m_size, int k_size, int n_size)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:  Tensor<float, 3, DataLayout> t_result_gpu(m_size, k_size, n_size);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:  cudaMalloc((void**)(&d_t_input), t_input_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:  cudaMalloc((void**)(&d_t_result), t_result_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:  cudaMemcpy(d_t_input, t_input.data(), t_input_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:      gpu_t_input(d_t_input, Eigen::array<int, 3>(m_size, k_size, n_size));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:      gpu_t_result(d_t_result, Eigen::array<int, 3>(m_size, k_size, n_size));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:  gpu_t_result.device(gpu_device) = gpu_t_input.cumsum(1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:  cudaMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, cudaMemcpyDeviceToHost);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:    if (fabs(t_result(i) - t_result_gpu(i)) < 1e-4f) {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), 1e-4f)) {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:              << " vs " <<  t_result_gpu(i) << std::endl;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:  cudaFree((void*)d_t_input);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:  cudaFree((void*)d_t_result);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:void test_cxx11_tensor_scan_cuda()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:  CALL_SUBTEST_1(test_cuda_cumsum<ColMajor>(128, 128, 128));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_scan_cuda.cu:  CALL_SUBTEST_2(test_cuda_cumsum<RowMajor>(128, 128, 128));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_cast_float16_cuda
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:#define EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:#include <cuda_fp16.h>
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:void test_cuda_conversion() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  Eigen::half* d_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  float* d_conv = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_half(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_conv(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.memcpyHostToDevice(d_float, floats.data(), num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_half.device(gpu_device) = gpu_float.cast<Eigen::half>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_conv.device(gpu_device) = gpu_half.cast<float>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(initial.data(), d_float, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(final.data(), d_conv, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.synchronize();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.deallocate(d_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.deallocate(d_half);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.deallocate(d_conv);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:void test_cxx11_tensor_cast_float16_cuda()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  CALL_SUBTEST(test_cuda_conversion());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  float * gpu_in_data  = static_cast<float*>(sycl_device.allocate(input.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  float * gpu_out_data  = static_cast<float*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  TensorMap<Tensor<float, 4>>  gpu_in(gpu_in_data, in_range);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  TensorMap<Tensor<float, 4>> gpu_out(gpu_out_data, out_range);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, input.data(),(input.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in.broadcast(broadcasts);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  cl::sycl::gpu_selector s;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_of_float16_cuda
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:#define EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:#include <cuda_fp16.h>
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_numext() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  bool* d_res_half = (bool*)gpu_device.allocate(num_elem * sizeof(bool));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  bool* d_res_float = (bool*)gpu_device.allocate(num_elem * sizeof(bool));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_res_half(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_res_float(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float.unaryExpr(Eigen::internal::scalar_isnan_op<float>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = gpu_float.cast<Eigen::half>().unaryExpr(Eigen::internal::scalar_isnan_op<Eigen::half>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(bool));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(bool));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:#ifdef EIGEN_HAS_CUDA_FP16
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_conversion() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_conv = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_half(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_conv(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float.device(gpu_device) = gpu_float.random();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_half.device(gpu_device) = gpu_float.cast<Eigen::half>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_conv.device(gpu_device) = gpu_half.cast<float>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(initial.data(), d_float, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(final.data(), d_conv, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_half);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_conv);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_unary() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_half = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float.abs();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().cast<float>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_elementwise() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_half = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float1(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float2(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float1.device(gpu_device) = gpu_float1.random();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float2.device(gpu_device) = gpu_float2.random();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = (gpu_float1 + gpu_float2) * gpu_float1;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = ((gpu_float1.cast<Eigen::half>() + gpu_float2.cast<Eigen::half>()) * gpu_float1.cast<Eigen::half>()).cast<float>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float2);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_trancendental() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float3 = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res1_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res1_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res2_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res2_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res3_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res3_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float1(d_float1, num_elem);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float2(d_float2, num_elem);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float3(d_float3, num_elem);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res1_half(d_res1_half, num_elem);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res1_float(d_res1_float, num_elem);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res2_half(d_res2_half, num_elem);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res2_float(d_res2_float, num_elem);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res3_half(d_res3_half, num_elem);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res3_float(d_res3_float, num_elem);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float1.device(gpu_device) = gpu_float1.random() - gpu_float1.constant(0.5f);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float2.device(gpu_device) = gpu_float2.random() + gpu_float1.constant(0.5f);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float3.device(gpu_device) = gpu_float3.random();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res1_float.device(gpu_device) = gpu_float1.exp().cast<Eigen::half>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res2_float.device(gpu_device) = gpu_float2.log().cast<Eigen::half>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res3_float.device(gpu_device) = gpu_float3.log1p().cast<Eigen::half>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res1_half.device(gpu_device) = gpu_float1.cast<Eigen::half>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res1_half.device(gpu_device) = gpu_res1_half.exp();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res2_half.device(gpu_device) = gpu_float2.cast<Eigen::half>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res2_half.device(gpu_device) = gpu_res2_half.log();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res3_half.device(gpu_device) = gpu_float3.cast<Eigen::half>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res3_half.device(gpu_device) = gpu_res3_half.log1p();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(input1.data(), d_float1, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(input2.data(), d_float2, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(input3.data(), d_float3, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec1.data(), d_res1_half, num_elem*sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec1.data(), d_res1_float, num_elem*sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec2.data(), d_res2_half, num_elem*sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec2.data(), d_res2_float, num_elem*sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec3.data(), d_res3_half, num_elem*sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec3.data(), d_res3_float, num_elem*sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float2);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float3);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res1_half);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res1_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res2_half);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res2_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res3_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res3_half);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_contractions() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2>, Eigen::Aligned> gpu_res_half(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2>, Eigen::Aligned> gpu_res_float(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float1.device(gpu_device) = gpu_float1.random() - gpu_float1.constant(0.5f);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float2.device(gpu_device) = gpu_float2.random() - gpu_float2.constant(0.5f);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float1.contract(gpu_float2, dims).cast<Eigen::half>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().contract(gpu_float2.cast<Eigen::half>(), dims);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float2);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_reductions(int size1, int size2, int redux) {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(result_size * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(result_size * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res_half(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res_float(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float1.device(gpu_device) = gpu_float1.random() * 2.0f;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float2.device(gpu_device) = gpu_float2.random() * 2.0f;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float1.sum(redux_dim).cast<Eigen::half>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().sum(redux_dim);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, result_size*sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, result_size*sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float2);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_reductions() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  test_cuda_reductions<void>(13, 13, 0);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  test_cuda_reductions<void>(13, 13, 1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  test_cuda_reductions<void>(35, 36, 0);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  test_cuda_reductions<void>(35, 36, 1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  test_cuda_reductions<void>(36, 35, 0);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  test_cuda_reductions<void>(36, 35, 1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_full_reductions() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(1 * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(1 * sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 0>, Eigen::Aligned> gpu_res_half(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 0>, Eigen::Aligned> gpu_res_float(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float1.device(gpu_device) = gpu_float1.random();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float2.device(gpu_device) = gpu_float2.random();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float1.sum().cast<Eigen::half>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().sum();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float1.maximum().cast<Eigen::half>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().maximum();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, sizeof(Eigen::half));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float2);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_forced_evals() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_half1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_half2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half1(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu: Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Unaligned> gpu_res_half2(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float.abs();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half1.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().eval().cast<float>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half2.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().broadcast(no_bcast).eval().cast<float>();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec1.data(), d_res_half1, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec2.data(), d_res_half1, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half2);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cxx11_tensor_of_float16_cuda()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_1(test_cuda_numext<void>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:#ifdef EIGEN_HAS_CUDA_FP16
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_1(test_cuda_conversion<void>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_1(test_cuda_unary<void>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_1(test_cuda_elementwise<void>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_1(test_cuda_trancendental<void>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_2(test_cuda_contractions<void>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_3(test_cuda_reductions<void>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_4(test_cuda_full_reductions<void>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_5(test_cuda_forced_evals<void>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  std::cout << "Half floats are not supported by this version of cuda: skipping the test" << std::endl;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_cuda
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:#define EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:#include <cuda_fp16.h>
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:void test_cuda_contraction(int m_size, int k_size, int n_size)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  // a 15 SM GK110 GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  Tensor<float, 2, DataLayout> t_result_gpu(m_size, n_size);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMalloc((void**)(&d_t_left), t_left_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMalloc((void**)(&d_t_right), t_right_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMalloc((void**)(&d_t_result), t_result_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMemcpy(d_t_left, t_left.data(), t_left_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMemcpy(d_t_right, t_right.data(), t_right_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:      gpu_t_left(d_t_left, Eigen::array<int, 2>(m_size, k_size));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:      gpu_t_right(d_t_right, Eigen::array<int, 2>(k_size, n_size));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:      gpu_t_result(d_t_result, Eigen::array<int, 2>(m_size, n_size));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, cudaMemcpyDeviceToHost);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:    if (fabs(t_result(i) - t_result_gpu(i)) < 1e-4f) {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), 1e-4f)) {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:              << " vs " <<  t_result_gpu(i) << std::endl;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaFree((void*)d_t_left);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaFree((void*)d_t_right);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaFree((void*)d_t_result);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  // a 15 SM GK110 GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  Tensor<float, 0, DataLayout> t_result_gpu;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMalloc((void**)(&d_t_left), t_left_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMalloc((void**)(&d_t_right), t_right_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMalloc((void**)(&d_t_result), t_result_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMemcpy(d_t_left, t_left.data(), t_left_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMemcpy(d_t_right, t_right.data(), t_right_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:      gpu_t_left(d_t_left, m_size, k_size);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:      gpu_t_right(d_t_right, k_size, n_size);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:      gpu_t_result(d_t_result);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, cudaMemcpyDeviceToHost);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  if (fabs(t_result() - t_result_gpu()) > 1e-4f &&
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:      !Eigen::internal::isApprox(t_result(), t_result_gpu(), 1e-4f)) {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:              << " vs " <<  t_result_gpu() << std::endl;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaFree((void*)d_t_left);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaFree((void*)d_t_right);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaFree((void*)d_t_result);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:void test_cuda_contraction_m() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:    test_cuda_contraction<ColMajor>(k, 128, 128);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:    test_cuda_contraction<RowMajor>(k, 128, 128);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:void test_cuda_contraction_k() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:    test_cuda_contraction<ColMajor>(128, k, 128);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:    test_cuda_contraction<RowMajor>(128, k, 128);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:void test_cuda_contraction_n() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:    test_cuda_contraction<ColMajor>(128, 128, k);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:    test_cuda_contraction<RowMajor>(128, 128, k);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:void test_cuda_contraction_sizes() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:        test_cuda_contraction<DataLayout>(m_sizes[i], n_sizes[j], k_sizes[k]);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:void test_cxx11_tensor_cuda()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_1(test_cuda_contraction<ColMajor>(128, 128, 128));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_1(test_cuda_contraction<RowMajor>(128, 128, 128));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_2(test_cuda_contraction_m<ColMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_3(test_cuda_contraction_m<RowMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_4(test_cuda_contraction_k<ColMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_5(test_cuda_contraction_k<RowMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_6(test_cuda_contraction_n<ColMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_7(test_cuda_contraction_n<RowMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_8(test_cuda_contraction_sizes<ColMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_9(test_cuda_contraction_sizes<RowMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:#define EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:#include <cuda_fp16.h>
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:// Context for evaluation on GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:struct GPUContext {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  GPUContext(const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in1, Eigen::TensorMap<Eigen::Tensor<float, 3> >& in2, Eigen::TensorMap<Eigen::Tensor<float, 3> >& out) : in1_(in1), in2_(in2), out_(out), gpu_device_(&stream_) {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:    assert(cudaMalloc((void**)(&kernel_1d_), 2*sizeof(float)) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:    assert(cudaMemcpy(kernel_1d_, kernel_1d_val, 2*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:    assert(cudaMalloc((void**)(&kernel_2d_), 4*sizeof(float)) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:    assert(cudaMemcpy(kernel_2d_, kernel_2d_val, 4*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:    assert(cudaMalloc((void**)(&kernel_3d_), 8*sizeof(float)) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:    assert(cudaMemcpy(kernel_3d_, kernel_3d_val, 8*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  ~GPUContext() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:    assert(cudaFree(kernel_1d_) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:    assert(cudaFree(kernel_2d_) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:    assert(cudaFree(kernel_3d_) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  const Eigen::GpuDevice& device() const { return gpu_device_; }
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  Eigen::CudaStreamDevice stream_;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  Eigen::GpuDevice gpu_device_;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:void test_gpu() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  cudaMalloc((void**)(&d_in1), in1_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  cudaMalloc((void**)(&d_in2), in2_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, 40,50,70);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, 40,50,70);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, 40,50,70);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  GPUContext context(gpu_in1, gpu_in2, gpu_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, context.device().stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  assert(cudaStreamSynchronize(context.device().stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, context.device().stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  assert(cudaStreamSynchronize(context.device().stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, context.device().stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  assert(cudaStreamSynchronize(context.device().stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device.cu:  CALL_SUBTEST_2(test_gpu());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:find_package(CUDA 7.0)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:if(CUDA_FOUND AND EIGEN_TEST_CUDA)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  # in the CUDA runtime
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  message(STATUS "Flags used to compile cuda code: " ${CMAKE_CXX_FLAGS})
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:    set(CUDA_NVCC_FLAGS "-ccbin ${CMAKE_C_COMPILER}" CACHE STRING "nvcc flags" FORCE)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  if(EIGEN_TEST_CUDA_CLANG)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 --cuda-gpu-arch=sm_${EIGEN_CUDA_COMPUTE_ARCH}")
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  set(EIGEN_CUDA_RELAXED_CONSTEXPR "--expt-relaxed-constexpr")
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  if (${CUDA_VERSION} STREQUAL "7.0")
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:    set(EIGEN_CUDA_RELAXED_CONSTEXPR "--relaxed-constexpr")
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:    set(EIGEN_CUDA_CXX11_FLAG "-std=c++11")
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:    set(EIGEN_CUDA_CXX11_FLAG "")
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  set(CUDA_NVCC_FLAGS  "${EIGEN_CUDA_CXX11_FLAG} ${EIGEN_CUDA_RELAXED_CONSTEXPR} -arch compute_${EIGEN_CUDA_COMPUTE_ARCH} -Xcudafe \"--display_error_number\" ${CUDA_NVCC_FLAGS}")
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  cuda_include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/include")
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_complex_cuda)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_complex_cwise_ops_cuda)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_reduction_cuda)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_argmax_cuda)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_cast_float16_cuda)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_scan_cuda)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  if (${EIGEN_CUDA_COMPUTE_ARCH} GREATER 29)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_cuda)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_contract_cuda)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_of_float16_cuda)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:  if (${EIGEN_CUDA_COMPUTE_ARCH} GREATER 34)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_random_cuda)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_device_sycl.cpp:  cl::sycl::gpu_selector s;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:#define EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:#include <cuda_fp16.h>
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:void test_cuda_complex_cwise_ops() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  cudaMalloc((void**)(&d_in1), complex_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  cudaMalloc((void**)(&d_in2), complex_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  cudaMalloc((void**)(&d_out), complex_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_in1(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_in2(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_out(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  gpu_in1.device(gpu_device) = gpu_in1.constant(a);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  gpu_in2.device(gpu_device) = gpu_in2.constant(b);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:        gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:        gpu_out.device(gpu_device) = gpu_in1 - gpu_in2;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:        gpu_out.device(gpu_device) = gpu_in1 * gpu_in2;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:        gpu_out.device(gpu_device) = gpu_in1 / gpu_in2;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:    assert(cudaMemcpyAsync(actual.data(), d_out, complex_bytes, cudaMemcpyDeviceToHost,
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:                           gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  cudaFree(d_in1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  cudaFree(d_in2);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  CALL_SUBTEST(test_cuda_complex_cwise_ops<float>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  CALL_SUBTEST(test_cuda_complex_cwise_ops<double>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:#define EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:#include <cuda_fp16.h>
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:void test_cuda_nullary() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaMalloc((void**)(&d_in1), complex_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaMalloc((void**)(&d_in2), complex_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaMalloc((void**)(&d_out2), float_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaMemcpy(d_in1, in1.data(), complex_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaMemcpy(d_in2, in2.data(), complex_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1, 0, int>, Eigen::Aligned> gpu_in1(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1, 0, int>, Eigen::Aligned> gpu_in2(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_out2(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_in1.device(gpu_device) = gpu_in1.constant(std::complex<float>(3.14f, 2.7f));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_out2.device(gpu_device) = gpu_in2.abs();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  assert(cudaMemcpyAsync(new1.data(), d_in1, complex_bytes, cudaMemcpyDeviceToHost,
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:                         gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  assert(cudaMemcpyAsync(new2.data(), d_out2, float_bytes, cudaMemcpyDeviceToHost,
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:                         gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaFree(d_in1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaFree(d_in2);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaFree(d_out2);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:static void test_cuda_sum_reductions() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  std::complex<float>* gpu_in_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(in_bytes));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  std::complex<float>* gpu_out_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(out_bytes));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  TensorMap<Tensor<std::complex<float>, 2> > in_gpu(gpu_in_ptr, num_rows, num_cols);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  TensorMap<Tensor<std::complex<float>, 0> > out_gpu(gpu_out_ptr);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  out_gpu.device(gpu_device) = in_gpu.sum();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  Tensor<std::complex<float>, 0> full_redux_gpu;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.synchronize();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  // Check that the CPU and GPU reductions return the same result.
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.deallocate(gpu_in_ptr);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.deallocate(gpu_out_ptr);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:static void test_cuda_product_reductions() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  std::complex<float>* gpu_in_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(in_bytes));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  std::complex<float>* gpu_out_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(out_bytes));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  TensorMap<Tensor<std::complex<float>, 2> > in_gpu(gpu_in_ptr, num_rows, num_cols);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  TensorMap<Tensor<std::complex<float>, 0> > out_gpu(gpu_out_ptr);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  out_gpu.device(gpu_device) = in_gpu.prod();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  Tensor<std::complex<float>, 0> full_redux_gpu;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.synchronize();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  // Check that the CPU and GPU reductions return the same result.
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.deallocate(gpu_in_ptr);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.deallocate(gpu_out_ptr);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  CALL_SUBTEST(test_cuda_nullary());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  CALL_SUBTEST(test_cuda_sum_reductions());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_complex_cuda.cu:  CALL_SUBTEST(test_cuda_product_reductions());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<float, 0> full_redux_gpu;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  float* gpu_in_data = static_cast<float*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  float* gpu_out_data =(float*)sycl_device.allocate(sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<float, 2> >  in_gpu(gpu_in_data, tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<float, 0> >  out_gpu(gpu_out_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.sum();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data, sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<float, 2> redux_gpu(reduced_tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  float* gpu_in_data = static_cast<float*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  float* gpu_out_data = static_cast<float*>(sycl_device.allocate(redux_gpu.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<float, 3> >  in_gpu(gpu_in_data, tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<float, 2> >  out_gpu(gpu_out_data, reduced_tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.sum(red_axis);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(redux_gpu.data(), gpu_out_data, redux_gpu.dimensions().TotalSize()*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:      VERIFY_IS_APPROX(redux_gpu(j,k), redux(j,k));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<float, 2> redux_gpu(reduced_tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  float* gpu_in_data = static_cast<float*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  float* gpu_out_data = static_cast<float*>(sycl_device.allocate(redux_gpu.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<float, 3> >  in_gpu(gpu_in_data, tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<float, 2> >  out_gpu(gpu_out_data, reduced_tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.sum(red_axis);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(redux_gpu.data(), gpu_out_data, redux_gpu.dimensions().TotalSize()*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:      VERIFY_IS_APPROX(redux_gpu(j,k), redux(j,k));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  cl::sycl::gpu_selector s;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_cuda
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:#define EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:#include <cuda_fp16.h>
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:void test_cuda_simple_argmax()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaMalloc((void**)(&d_in), in_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaMalloc((void**)(&d_out_max), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaMalloc((void**)(&d_out_min), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaMemcpy(d_in, in.data(), in_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<double, 3, Layout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 3>(72,53,97));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 1, Layout>, Aligned > gpu_out_max(d_out_max, Eigen::array<DenseIndex, 1>(1));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 1, Layout>, Aligned > gpu_out_min(d_out_min, Eigen::array<DenseIndex, 1>(1));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  gpu_out_max.device(gpu_device) = gpu_in.argmax();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  gpu_out_min.device(gpu_device) = gpu_in.argmin();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  assert(cudaMemcpyAsync(out_max.data(), d_out_max, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  assert(cudaMemcpyAsync(out_min.data(), d_out_min, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaFree(d_in);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaFree(d_out_max);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaFree(d_out_min);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:void test_cuda_argmax_dim()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMalloc((void**)(&d_in), in_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMemcpy(d_in, tensor.data(), in_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 4>(2, 3, 5, 7));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout>, Aligned > gpu_out(d_out, out_shape);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    gpu_out.device(gpu_device) = gpu_in.argmax(dim);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaMemcpyAsync(tensor_arg.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMemcpy(d_in, tensor.data(), in_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    gpu_out.device(gpu_device) = gpu_in.argmax(dim);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaMemcpyAsync(tensor_arg.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaFree(d_in);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:void test_cuda_argmin_dim()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMalloc((void**)(&d_in), in_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMemcpy(d_in, tensor.data(), in_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 4>(2, 3, 5, 7));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout>, Aligned > gpu_out(d_out, out_shape);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    gpu_out.device(gpu_device) = gpu_in.argmin(dim);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaMemcpyAsync(tensor_arg.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMemcpy(d_in, tensor.data(), in_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    gpu_out.device(gpu_device) = gpu_in.argmin(dim);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaMemcpyAsync(tensor_arg.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaFree(d_in);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:void test_cxx11_tensor_cuda()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  CALL_SUBTEST_1(test_cuda_simple_argmax<RowMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  CALL_SUBTEST_1(test_cuda_simple_argmax<ColMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  CALL_SUBTEST_2(test_cuda_argmax_dim<RowMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  CALL_SUBTEST_2(test_cuda_argmax_dim<ColMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  CALL_SUBTEST_3(test_cuda_argmin_dim<RowMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_argmax_cuda.cu:  CALL_SUBTEST_3(test_cuda_argmin_dim<ColMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/openglsupport.cpp:    #ifdef GLEW_ARB_gpu_shader_fp64
celerite/cpp/lib/eigen_3.3.3/unsupported/test/openglsupport.cpp:    if(GLEW_ARB_gpu_shader_fp64)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/openglsupport.cpp:      #ifdef GL_ARB_gpu_shader_fp64
celerite/cpp/lib/eigen_3.3.3/unsupported/test/openglsupport.cpp:      std::cerr << "Warning: GLEW_ARB_gpu_shader_fp64 was not tested\n";
celerite/cpp/lib/eigen_3.3.3/unsupported/test/openglsupport.cpp:      std::cerr << "Warning: GLEW_ARB_gpu_shader_fp64 was not tested\n";
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_of_strings.cpp:  // Beware: none of this is likely to ever work on a GPU.
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_cuda
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:#define EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:#include <cuda_fp16.h>
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_nullary() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in1), tensor_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in2), tensor_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in1, in1.data(), tensor_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in2, in2.data(), tensor_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_in1(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_in2(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_in1.device(gpu_device) = gpu_in1.constant(3.14f);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_in2.device(gpu_device) = gpu_in2.random();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(new1.data(), d_in1, tensor_bytes, cudaMemcpyDeviceToHost,
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:                         gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(new2.data(), d_in2, tensor_bytes, cudaMemcpyDeviceToHost,
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:                         gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in2);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_elementwise_small() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in1), in1_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in2), in2_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in2(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_out(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost,
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:                         gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in2);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_elementwise()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in1), in1_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in2), in2_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in3), in3_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in3, in3.data(), in3_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in3(d_in3, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2 * gpu_in3;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in2);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in3);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_props() {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in1), in1_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_out(
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = (gpu_in1.isnan)();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost,
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:                         gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_reduction()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in1), in1_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_in1(d_in1, 72,53,97,113);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in1.maximum(reduction_axis);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_contraction()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  // a 15 SM GK110 GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_t_left), t_left_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_t_right), t_right_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_t_result), t_result_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_t_left, t_left.data(), t_left_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_t_right, t_right.data(), t_right_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_t_left(d_t_left, 6, 50, 3, 31);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_t_right(d_t_right, 3, 31, 7, 20, 1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_t_result(d_t_result, 6, 50, 7, 20, 1);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(t_result.data(), d_t_result, t_result_bytes, cudaMemcpyDeviceToHost);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_t_left);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_t_right);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_t_result);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_convolution_1d()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_input), input_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_kernel), kernel_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_input(d_input, 74,37,11,137);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, DataLayout> > gpu_kernel(d_kernel, 4);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_out(d_out, 74,34,11,137);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_input);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_kernel);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_convolution_inner_dim_col_major_1d()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_input), input_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_kernel), kernel_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, ColMajor> > gpu_input(d_input,74,9,11,7);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, ColMajor> > gpu_kernel(d_kernel,4);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, ColMajor> > gpu_out(d_out,71,9,11,7);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_input);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_kernel);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_convolution_inner_dim_row_major_1d()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_input), input_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_kernel), kernel_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, RowMajor> > gpu_input(d_input, 7,9,11,74);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, RowMajor> > gpu_kernel(d_kernel, 4);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, RowMajor> > gpu_out(d_out, 7,9,11,71);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_input);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_kernel);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_convolution_2d()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_input), input_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_kernel), kernel_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_input(d_input,74,37,11,137);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2, DataLayout> > gpu_kernel(d_kernel,3,4);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_out(d_out,74,35,8,137);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_input);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_kernel);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_convolution_3d()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_input), input_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_kernel), kernel_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;    
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_input(d_input,74,37,11,137,17);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3, DataLayout> > gpu_kernel(d_kernel,3,4,2);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_out(d_out,74,35,8,136,17);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_input);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_kernel);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_lgamma(const Scalar stddev)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in.lgamma();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_digamma()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 7);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 7);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in.digamma();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_zeta()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_x), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_q), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_x, in_x.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_q, in_q.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 6);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_q(d_in_q, 6);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 6);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in_x.zeta(gpu_in_q);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_x);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_q);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_polygamma()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_x), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_n), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_x, in_x.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_n, in_n.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 7);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_n(d_in_n, 7);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 7);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in_n.polygamma(gpu_in_x);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_x);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_n);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_igamma()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMalloc((void**)(&d_a), bytes) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMalloc((void**)(&d_x), bytes) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMalloc((void**)(&d_out), bytes) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_a(d_a, 6, 6);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_x(d_x, 6, 6);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 6, 6);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_a.igamma(gpu_x);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_a);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_x);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_igammac()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_a), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_x), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_a(d_a, 6, 6);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_x(d_x, 6, 6);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 6, 6);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_a.igammac(gpu_x);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_a);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_x);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_erf(const Scalar stddev)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMalloc((void**)(&d_in), bytes) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMalloc((void**)(&d_out), bytes) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in.erf();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_erfc(const Scalar stddev)
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in.erfc();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_betainc()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_x), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_a), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_b), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_x, in_x.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_a, in_a.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_b, in_b.data(), bytes, cudaMemcpyHostToDevice);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 125);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_a(d_in_a, 125);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_b(d_in_b, 125);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 125);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = betainc(gpu_in_a, gpu_in_b, gpu_in_x);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_x);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_a);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_b);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:void test_cxx11_tensor_cuda()
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_1(test_cuda_nullary());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_1(test_cuda_elementwise_small());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_1(test_cuda_elementwise());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_1(test_cuda_props());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_1(test_cuda_reduction());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_2(test_cuda_contraction<ColMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_2(test_cuda_contraction<RowMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_1d<ColMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_1d<RowMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_inner_dim_col_major_1d());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_inner_dim_row_major_1d());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_2d<ColMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_2d<RowMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_3d<ColMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_3d<RowMajor>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<float>(1.0f));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<float>(100.0f));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<float>(0.01f));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<float>(0.001f));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<double>(1.0));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<double>(100.0));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<double>(0.01));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<double>(0.001));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<float>(1.0f));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<float>(100.0f));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<float>(0.01f));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<float>(0.001f));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<float>(1.0f));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  // CALL_SUBTEST(test_cuda_erfc<float>(100.0f));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<float>(5.0f)); // CUDA erfc lacks precision for large inputs
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<float>(0.01f));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<float>(0.001f));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<double>(1.0));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<double>(100.0));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<double>(0.01));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<double>(0.001));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<double>(1.0));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  // CALL_SUBTEST(test_cuda_erfc<double>(100.0));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<double>(5.0)); // CUDA erfc lacks precision for large inputs
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<double>(0.01));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<double>(0.001));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_digamma<float>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_digamma<double>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_polygamma<float>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_polygamma<double>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_zeta<float>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_zeta<double>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_igamma<float>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_igammac<float>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_igamma<double>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_igammac<double>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_6(test_cuda_betainc<float>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_6(test_cuda_betainc<double>());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  float * gpu_in1_data  = static_cast<float*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  float * gpu_in2_data  = static_cast<float*>(sycl_device.allocate(in2.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  float * gpu_out_data =  static_cast<float*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(float)));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<float, 3>> gpu_in1(gpu_in1_data, tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<float, 3>> gpu_in2(gpu_in2_data, tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<float, 3>> gpu_out(gpu_out_data, tensorRange);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in1_data, in1.data(),(in1.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in1.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  gpu_out.device(sycl_device) =(gpu_in1 + gpu_in2).eval() * gpu_in2;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.deallocate(gpu_in1_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.deallocate(gpu_in2_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  cl::sycl::gpu_selector s;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_reduction_cuda
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:#define EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:#include <cuda_fp16.h>
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Type* gpu_in_ptr = static_cast<Type*>(gpu_device.allocate(in_bytes));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Type* gpu_out_ptr = static_cast<Type*>(gpu_device.allocate(out_bytes));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  TensorMap<Tensor<Type, 2, DataLayout> > in_gpu(gpu_in_ptr, num_rows, num_cols);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  TensorMap<Tensor<Type, 0, DataLayout> > out_gpu(gpu_out_ptr);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  out_gpu.device(gpu_device) = in_gpu.sum();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Tensor<Type, 0, DataLayout> full_redux_gpu;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_device.synchronize();
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  // Check that the CPU and GPU reductions return the same result.
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_device.deallocate(gpu_in_ptr);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_device.deallocate(gpu_out_ptr);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::GpuDevice dev(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 3, DataLayout> > gpu_in(in_data, dim_x, dim_y, dim_z);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 2, DataLayout> > gpu_out(out_data, dim_y, dim_z);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_out.device(dev) = gpu_in.sum(red_axis);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_out.device(dev) += gpu_in.sum(red_axis);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Tensor<Type, 2, DataLayout> redux_gpu(dim_y, dim_z);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  dev.memcpyDeviceToHost(redux_gpu.data(), out_data, gpu_out.size()*sizeof(Type));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  // Check that the CPU and GPU reductions return the same result.
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  for (int i = 0; i < gpu_out.size(); ++i) {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:    VERIFY_IS_APPROX(2*redux(i), redux_gpu(i));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::CudaStreamDevice stream;
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::GpuDevice dev(&stream);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 3, DataLayout> > gpu_in(in_data, dim_x, dim_y, dim_z);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 2, DataLayout> > gpu_out(out_data, dim_x, dim_y);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_out.device(dev) = gpu_in.sum(red_axis);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_out.device(dev) += gpu_in.sum(red_axis);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Tensor<Type, 2, DataLayout> redux_gpu(dim_x, dim_y);
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  dev.memcpyDeviceToHost(redux_gpu.data(), out_data, gpu_out.size()*sizeof(Type));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  // Check that the CPU and GPU reductions return the same result.
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:  for (int i = 0; i < gpu_out.size(); ++i) {
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:    VERIFY_IS_APPROX(2*redux(i), redux_gpu(i));
celerite/cpp/lib/eigen_3.3.3/unsupported/test/cxx11_tensor_reduction_cuda.cu:void test_cxx11_tensor_reduction_cuda() {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/Tensor:#ifdef EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/Tensor:#include <cuda_runtime.h>
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorDeviceCuda.h"
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorReductionCuda.h"
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorContractionCuda.h"
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:#ifdef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:// GPU: the evaluation of the expression is offloaded to a GPU.
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#if defined(EIGEN_USE_GPU)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:class TensorExecutor<Expression, GpuDevice, Vectorizable> {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:  static void run(const Expression& expr, const GpuDevice& device);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#if defined(__CUDACC__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:inline void TensorExecutor<Expression, GpuDevice, Vectorizable>::run(
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const Expression& expr, const GpuDevice& device) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:  TensorEvaluator<Expression, GpuDevice> evaluator(expr, device);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const int block_size = device.maxCudaThreadsPerBlock();
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    LAUNCH_CUDA_KERNEL(
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:        (EigenMetaKernel<TensorEvaluator<Expression, GpuDevice>, Index>),
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#endif  // __CUDACC__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#endif  // EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__CUDA_ARCH__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__CUDA_ARCH__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__SIZEOF_INT128__) && !defined(__CUDA_ARCH__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device, return the amount of shared memory available.
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    return __CUDA_ARCH__ / 100;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDevice.h:  * on the specified computing 'device' (GPU, thread pool, ...)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDevice.h:  *    C.device(EIGEN_GPU) = A + B;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:// Full reducers for GPU, don't vectorize for now
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:// Reducer function that enables multiple cuda thread to safely accumulate at the same
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:// attempts to update it with the new value. If in the meantime another cuda thread
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((FullReductionKernel<block_size, num_per_thread, Self, Op, Index>),
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitFullReduxKernelHalfFloat<Self, Op, Index>),
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((FullReductionKernelHalfFloat<block_size, num_per_thread, Self, Op, Index>),
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionCleanupKernelHalfFloat<Op>),
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:struct FullReducer<Self, Op, GpuDevice, Vectorizable> {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    FullReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static EIGEN_DEVICE_FUNC bool run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index, typename Self::Index) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    assert(false && "Should only be called to reduce doubles, floats and half floats on a gpu device");
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      const int max_blocks = device.getNumCudaMultiProcessors() *
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / 1024;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitKernel<OutputType, Index>),
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((InnerReductionKernel<num_per_thread, Self, Op, Index>),
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index, typename Self::Index) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      const int max_blocks = device.getNumCudaMultiProcessors() *
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / 1024;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitKernelHalfFloat<Self, Op, Index>),
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((InnerReductionKernelHalfFloat<num_per_thread, Self, Op, Index>),
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:struct InnerReducer<Self, Op, GpuDevice> {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    return InnerReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs_to_reduce, num_preserved_vals);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:struct OuterReducer<Self, Op, GpuDevice> {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    assert(false && "Should only be called to reduce doubles or floats on a gpu device");
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, float* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      const int max_blocks = device.getNumCudaMultiProcessors() *
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                             device.maxCudaThreadsPerMultiProcessor() / 1024;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitKernel<float, Index>),
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((OuterReductionKernel<num_per_thread, Self, Op, Index>),
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#endif // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/README.md:GPU using cuda.  Additional implementations may be added later.
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/README.md:GpuDevice.
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/README.md:#### Evaluating On GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/README.md:You need to create a GPU device but you also need to explicitly allocate the
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/README.md:memory for tensors with cuda.
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/README.md:*   On GPUs only floating point values are properly tested and optimized for.
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/README.md:*   Complex and integer values are known to be broken on GPUs. If you try to use
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:// It is very expensive to start the memcpy kernel on GPU: we therefore only
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:#ifdef EIGEN_USE_GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:template <typename Index> struct MemcpyTriggerForSlicing<Index, GpuDevice>  {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:  EIGEN_DEVICE_FUNC MemcpyTriggerForSlicing(const GpuDevice&) { }
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:// For CUDA packet types when using a GpuDevice
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__) && defined(EIGEN_HAS_CUDA_FP16)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:struct PacketType<half, GpuDevice> {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:// Can't use std::pairs on cuda devices
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorSycl.h:// this is used to change the address space type in tensor map for GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorFFT.h:  * Improve the performance on GPU
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:struct GpuDevice;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:struct IsVectorizable<GpuDevice, Expression> {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:  static const bool value = TensorEvaluator<Expression, GpuDevice>::PacketAccess &&
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:                            TensorEvaluator<Expression, GpuDevice>::IsAligned;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, GpuDevice> :
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    public TensorContractionEvaluatorBase<TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, GpuDevice> > {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:  typedef GpuDevice Device;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:  typedef typename PacketType<CoeffReturnType, GpuDevice>::type PacketReturnType;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    static void Run(const LhsMapper& lhs, const RhsMapper& rhs, const OutputMapper& output, Index m, Index n, Index k, const GpuDevice& device) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    LAUNCH_CUDA_KERNEL((EigenContractionKernel<Scalar, Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    static void Run(const LhsMapper& lhs, const RhsMapper& rhs, const OutputMapper& output, Index m, Index n, Index k, const GpuDevice& device) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:        LAUNCH_CUDA_KERNEL((EigenFloatContractionKernel16x16<Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:        LAUNCH_CUDA_KERNEL((EigenFloatContractionKernel<Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    setCudaSharedMemConfig(cudaSharedMemBankSizeEightByte);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#endif // EIGEN_USE_GPU and __CUDACC__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorMacros.h:#ifndef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionSycl.h:    /// This one is used to collect all the reduced value of shared memory as we dont have global barrier on GPU. Once it is saved we can
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionSycl.h:    /// This one is used to collect all the reduced value of shared memory as we dont have global barrier on GPU. Once it is saved we can
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#ifdef EIGEN_HAS_CUDA_FP16
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:        ((RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) ||
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:         !RunningOnGPU))) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    else if (RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    if ((RunningOnSycl || RunningFullReduction || RunningOnGPU) && m_result) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    if (RunningOnGPU && m_result) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#ifdef EIGEN_HAS_CUDA_FP16
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:  static const bool RunningOnGPU = internal::is_same<Device, Eigen::GpuDevice>::value;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:static const bool RunningOnGPU = false;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:  static const bool RunningOnGPU = false;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#if defined(EIGEN_USE_GPU) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static const int kCudaScratchSize = 1024;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:// This defines an interface that GPUDevice can take to use
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:// CUDA streams underneath.
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  virtual const cudaStream_t& stream() const = 0;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  virtual const cudaDeviceProp& deviceProperties() const = 0;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static cudaDeviceProp* m_deviceProperties;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaError_t status = cudaGetDeviceCount(&num_devices);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      if (status != cudaSuccess) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        std::cerr << "Failed to get the number of CUDA devices: "
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:                  << cudaGetErrorString(status)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        assert(status == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      m_deviceProperties = new cudaDeviceProp[num_devices];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        status = cudaGetDeviceProperties(&m_deviceProperties[i], i);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        if (status != cudaSuccess) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:          std::cerr << "Failed to initialize CUDA device #"
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:                    << cudaGetErrorString(status)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:          assert(status == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static const cudaStream_t default_stream = cudaStreamDefault;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:class CudaStreamDevice : public StreamInterface {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  CudaStreamDevice() : stream_(&default_stream), scratch_(NULL), semaphore_(NULL) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaGetDevice(&device_);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  CudaStreamDevice(int device) : stream_(&default_stream), device_(device), scratch_(NULL), semaphore_(NULL) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  // assumes that the stream is associated to the current gpu device.
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  CudaStreamDevice(const cudaStream_t* stream, int device = -1)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaGetDevice(&device_);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaError_t err = cudaGetDeviceCount(&num_devices);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      assert(err == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  virtual ~CudaStreamDevice() {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  const cudaStream_t& stream() const { return *stream_; }
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  const cudaDeviceProp& deviceProperties() const {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaSetDevice(device_);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    err = cudaMalloc(&result, num_bytes);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaSetDevice(device_);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    err = cudaFree(buffer);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      scratch_ = allocate(kCudaScratchSize + sizeof(unsigned int));
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      char* scratch = static_cast<char*>(scratchpad()) + kCudaScratchSize;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaError_t err = cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      assert(err == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  const cudaStream_t* stream_;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:struct GpuDevice {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  explicit GpuDevice(const StreamInterface* stream) : stream_(stream), max_blocks_(INT_MAX) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  explicit GpuDevice(const StreamInterface* stream, int num_blocks) : stream_(stream), max_blocks_(num_blocks) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE const cudaStream_t& stream() const {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifndef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice,
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err =
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream_->stream());
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err =
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream_->stream());
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifndef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaMemsetAsync(buffer, c, n, stream_->stream());
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    // there is no l3 cache on cuda devices.
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaStreamSynchronize(stream_->stream());
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    if (err != cudaSuccess) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      std::cerr << "Error detected in CUDA stream: "
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:                << cudaGetErrorString(err)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      assert(err == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE int getNumCudaMultiProcessors() const {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE int maxCudaThreadsPerBlock() const {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE int maxCudaThreadsPerMultiProcessor() const {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  // This function checks if the CUDA runtime recorded an error for the
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t error = cudaStreamQuery(stream_->stream());
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    return (error == cudaSuccess) || (error == cudaErrorNotReady);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#define LAUNCH_CUDA_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)             \
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  assert(cudaGetLastError() == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifdef __CUDACC__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static EIGEN_DEVICE_FUNC inline void setCudaSharedMemConfig(cudaSharedMemConfig config) {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifndef __CUDA_ARCH__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  cudaError_t status = cudaDeviceSetSharedMemConfig(config);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  assert(status == cudaSuccess);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    array<Index, NumDims> cudaInputDimensions;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    array<Index, NumDims> cudaOutputDimensions;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      cudaInputDimensions[index] = input_dims[indices[i]];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      cudaOutputDimensions[index] = dimensions[indices[i]];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        cudaInputDimensions[written] = input_dims[i];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        cudaOutputDimensions[written] = dimensions[i];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] =
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaInputStrides[i - 1] * cudaInputDimensions[i - 1];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] =
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaOutputStrides[i - 1] * cudaOutputDimensions[i - 1];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] = 1;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] = 1;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] =
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaInputStrides[i + 1] * cudaInputDimensions[i + 1];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] =
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaOutputStrides[i + 1] * cudaOutputDimensions[i + 1];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] = 1;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] = 1;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputPlaneToTensorInputOffset(Index p) const {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaInputStrides[d];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaInputStrides[d];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaInputStrides[d];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaInputStrides[d];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputPlaneToTensorOutputOffset(Index p) const {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaOutputStrides[d];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaOutputStrides[d];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaOutputStrides[d];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaOutputStrides[d];
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputKernelToTensorInputOffset(Index i) const {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputKernelToTensorOutputOffset(Index i) const {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputKernelToTensorInputOffset(Index i, Index j) const {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputKernelToTensorOutputOffset(Index i, Index j) const {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputKernelToTensorInputOffset(Index i, Index j, Index k) const {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputKernelToTensorOutputOffset(Index i, Index j, Index k) const {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  array<Index, NumDims> m_cudaInputStrides;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  array<Index, NumDims> m_cudaOutputStrides;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:// Use an optimized implementation of the evaluation code for GPUs whenever possible.
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const int tensor_index = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i+first_x);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const int tensor_index = plane_output_offset + indexMapper.mapCudaOutputKernelToTensorOutputOffset(i+first_x);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const int tensor_index = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i+first_x, j+first_y);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const int tensor_index = plane_output_offset + indexMapper.mapCudaOutputKernelToTensorOutputOffset(i+first_x, j+first_y);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          const int tensor_index = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i+first_x, j+first_y, k+first_z);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          const int tensor_index = plane_output_offset + indexMapper.mapCudaOutputKernelToTensorOutputOffset(i+first_x, j+first_y, k+first_z);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:struct TensorEvaluator<const TensorConvolutionOp<Indices, InputArgType, KernelArgType>, GpuDevice>
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  static const int NumDims =  internal::array_size<typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions>::value;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  typedef typename TensorEvaluator<KernelArgType, GpuDevice>::Dimensions KernelDimensions;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    IsAligned = TensorEvaluator<InputArgType, GpuDevice>::IsAligned & TensorEvaluator<KernelArgType, GpuDevice>::IsAligned,
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    Layout = TensorEvaluator<InputArgType, GpuDevice>::Layout,
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const GpuDevice& device)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<InputArgType, GpuDevice>::Layout) == static_cast<int>(TensorEvaluator<KernelArgType, GpuDevice>::Layout)), YOU_MADE_A_PROGRAMMING_MISTAKE);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions& input_dims = m_inputImpl.dimensions();
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const typename TensorEvaluator<KernelArgType, GpuDevice>::Dimensions& kernel_dims = m_kernelImpl.dimensions();
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  typedef typename PacketType<CoeffReturnType, GpuDevice>::type PacketReturnType;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const bool PacketAccess = internal::IsVectorizable<GpuDevice, KernelArgType>::value;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      internal::TensorExecutor<const EvalTo, GpuDevice, PacketAccess>::run(evalToTmp, m_device);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    typedef typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions InputDims;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int maxThreadsPerBlock = m_device.maxCudaThreadsPerBlock();
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int maxBlocksPerProcessor = m_device.maxCudaThreadsPerMultiProcessor() / maxThreadsPerBlock;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int numMultiProcessors = m_device.getNumCudaMultiProcessors();
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, 4, data);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, 7, data);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, kernel_size, data);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4, 7>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 4, 7, data);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 4, kernel_size_y, data);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7, 4>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 7, 4, data);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 7, kernel_size_y, data);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, Dynamic, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, kernel_size_x, kernel_size_y, data);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        LAUNCH_CUDA_KERNEL((EigenConvolutionKernel3D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, numZ, maxZ, kernel_size_x, kernel_size_y, kernel_size_z, data);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  TensorEvaluator<InputArgType, GpuDevice> m_inputImpl;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  TensorEvaluator<KernelArgType, GpuDevice> m_kernelImpl;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  const GpuDevice& m_device;
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:// GPU implementation of scan
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:struct ScanLauncher<Self, Reducer, GpuDevice> {
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:     LAUNCH_CUDA_KERNEL((ScanKernel<Self, Reducer>), num_blocks, block_size, 0, self.device(), self, total_size, data);
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:#endif  // EIGEN_USE_GPU && __CUDACC__
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:// Use the texture cache on CUDA devices whenever possible
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/util/EmulateArray.h:// Moreover, CUDA doesn't support the STL containers, so we use our own instead.
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/util/EmulateArray.h:#if (__cplusplus <= 199711L && EIGEN_COMP_MSVC < 1900) || defined(__CUDACC__) || defined(EIGEN_AVOID_STL_ARRAY)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/util/EmulateArray.h:// The compiler supports c++11, and we're not targetting cuda: use std::array as Eigen::array
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/CXX11/src/util/MaxSizeVector.h:  * std::vector is not an option (e.g. on GPU or when compiling using
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/SpecialFunctions:#if defined EIGEN_VECTORIZE_CUDA
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/SpecialFunctions:  #include "src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h"
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/OpenGLSupport:#ifdef GL_ARB_gpu_shader_fp64
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:#if !defined(__CUDA_ARCH__) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:#if !defined(__CUDA_ARCH__) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:    if (x == inf) return zero;  // std::isinf crashes on CUDA
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#ifndef EIGEN_CUDA_SPECIALFUNCTIONS_H
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#define EIGEN_CUDA_SPECIALFUNCTIONS_H
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
celerite/cpp/lib/eigen_3.3.3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#endif // EIGEN_CUDA_SPECIALFUNCTIONS_H

```
