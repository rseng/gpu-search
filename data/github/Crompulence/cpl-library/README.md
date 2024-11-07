# https://github.com/Crompulence/cpl-library

```console
test/gtests/packd_gtest/gtest/gtest.h:// with a TR1 tuple implementation.  NVIDIA's CUDA NVCC compiler
test/gtests/packd_gtest/gtest/gtest.h:# if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000) \
Makefile:	BUILDPPROCMACROS += JSON_SUPPORT
Makefile:ifdef BUILDPPROCMACROS
Makefile:	$(F90) $(FFLAGS) -D$(BUILDPPROCMACROS) -c $(fbindsrcfile) -o $(fbindobjfile)
Makefile:ifdef BUILDPPROCMACROS
Makefile:	$(F90) $(FFLAGS) -D$(BUILDPPROCMACROS) -c $(cbindsrcFfile) -o $(cbindobjFfile)
Makefile:ifdef BUILDPPROCMACROS
Makefile:	$(CPP) $(CFLAGS) -D$(BUILDPPROCMACROS) -c $(cbindsrcCfile) -o $(cbindobjCfile)
Makefile:ifdef BUILDPPROCMACROS
Makefile:	$(CPP) $(CFLAGS) -D$(BUILDPPROCMACROS) -I$(cbinddir) -c $(cppbindsrcfiles) -o $(cppbindobjfiles)
src/utils/overlap/eigen/test/cuda_basic.cu:// workaround issue between gcc >= 4.7 and cuda 5.5
src/utils/overlap/eigen/test/cuda_basic.cu:#define EIGEN_TEST_FUNC cuda_basic
src/utils/overlap/eigen/test/cuda_basic.cu:#include <cuda.h>
src/utils/overlap/eigen/test/cuda_basic.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
src/utils/overlap/eigen/test/cuda_basic.cu:#include <cuda_fp16.h>
src/utils/overlap/eigen/test/cuda_basic.cu:#include "cuda_common.h"
src/utils/overlap/eigen/test/cuda_basic.cu:void test_cuda_basic()
src/utils/overlap/eigen/test/cuda_basic.cu:  ei_test_init_cuda();
src/utils/overlap/eigen/test/cuda_basic.cu:  #ifndef __CUDA_ARCH__
src/utils/overlap/eigen/test/cuda_basic.cu:  CALL_SUBTEST( run_and_compare_to_cuda(coeff_wise<Vector3f>(), nthreads, in, out) );
src/utils/overlap/eigen/test/cuda_basic.cu:  CALL_SUBTEST( run_and_compare_to_cuda(coeff_wise<Array44f>(), nthreads, in, out) );
src/utils/overlap/eigen/test/cuda_basic.cu:  CALL_SUBTEST( run_and_compare_to_cuda(replicate<Array4f>(), nthreads, in, out) );
src/utils/overlap/eigen/test/cuda_basic.cu:  CALL_SUBTEST( run_and_compare_to_cuda(replicate<Array33f>(), nthreads, in, out) );
src/utils/overlap/eigen/test/cuda_basic.cu:  CALL_SUBTEST( run_and_compare_to_cuda(redux<Array4f>(), nthreads, in, out) );
src/utils/overlap/eigen/test/cuda_basic.cu:  CALL_SUBTEST( run_and_compare_to_cuda(redux<Matrix3f>(), nthreads, in, out) );
src/utils/overlap/eigen/test/cuda_basic.cu:  CALL_SUBTEST( run_and_compare_to_cuda(prod_test<Matrix3f,Matrix3f>(), nthreads, in, out) );
src/utils/overlap/eigen/test/cuda_basic.cu:  CALL_SUBTEST( run_and_compare_to_cuda(prod_test<Matrix4f,Vector4f>(), nthreads, in, out) );
src/utils/overlap/eigen/test/cuda_basic.cu:  CALL_SUBTEST( run_and_compare_to_cuda(diagonal<Matrix3f,Vector3f>(), nthreads, in, out) );
src/utils/overlap/eigen/test/cuda_basic.cu:  CALL_SUBTEST( run_and_compare_to_cuda(diagonal<Matrix4f,Vector4f>(), nthreads, in, out) );
src/utils/overlap/eigen/test/cuda_basic.cu:  CALL_SUBTEST( run_and_compare_to_cuda(eigenvalues<Matrix3f>(), nthreads, in, out) );
src/utils/overlap/eigen/test/cuda_basic.cu:  CALL_SUBTEST( run_and_compare_to_cuda(eigenvalues<Matrix2f>(), nthreads, in, out) );
src/utils/overlap/eigen/test/CMakeLists.txt:# CUDA unit tests
src/utils/overlap/eigen/test/CMakeLists.txt:option(EIGEN_TEST_CUDA "Enable CUDA support in unit tests" OFF)
src/utils/overlap/eigen/test/CMakeLists.txt:option(EIGEN_TEST_CUDA_CLANG "Use clang instead of nvcc to compile the CUDA tests" OFF)
src/utils/overlap/eigen/test/CMakeLists.txt:if(EIGEN_TEST_CUDA_CLANG AND NOT CMAKE_CXX_COMPILER MATCHES "clang")
src/utils/overlap/eigen/test/CMakeLists.txt:  message(WARNING "EIGEN_TEST_CUDA_CLANG is set, but CMAKE_CXX_COMPILER does not appear to be clang.")
src/utils/overlap/eigen/test/CMakeLists.txt:if(EIGEN_TEST_CUDA)
src/utils/overlap/eigen/test/CMakeLists.txt:find_package(CUDA 5.0)
src/utils/overlap/eigen/test/CMakeLists.txt:if(CUDA_FOUND)
src/utils/overlap/eigen/test/CMakeLists.txt:  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
src/utils/overlap/eigen/test/CMakeLists.txt:    set(CUDA_NVCC_FLAGS "-ccbin ${CMAKE_C_COMPILER}" CACHE STRING "nvcc flags" FORCE)
src/utils/overlap/eigen/test/CMakeLists.txt:  if(EIGEN_TEST_CUDA_CLANG)
src/utils/overlap/eigen/test/CMakeLists.txt:   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 --cuda-gpu-arch=sm_30")
src/utils/overlap/eigen/test/CMakeLists.txt:  cuda_include_directories(${CMAKE_CURRENT_BINARY_DIR})
src/utils/overlap/eigen/test/CMakeLists.txt:  ei_add_test(cuda_basic)
src/utils/overlap/eigen/test/CMakeLists.txt:endif(CUDA_FOUND)
src/utils/overlap/eigen/test/CMakeLists.txt:endif(EIGEN_TEST_CUDA)
src/utils/overlap/eigen/test/cuda_common.h:#ifndef EIGEN_TEST_CUDA_COMMON_H
src/utils/overlap/eigen/test/cuda_common.h:#define EIGEN_TEST_CUDA_COMMON_H
src/utils/overlap/eigen/test/cuda_common.h:#include <cuda.h>
src/utils/overlap/eigen/test/cuda_common.h:#include <cuda_runtime.h>
src/utils/overlap/eigen/test/cuda_common.h:#include <cuda_runtime_api.h>
src/utils/overlap/eigen/test/cuda_common.h:#ifndef __CUDACC__
src/utils/overlap/eigen/test/cuda_common.h:void run_on_cuda_meta_kernel(const Kernel ker, int n, const Input* in, Output* out)
src/utils/overlap/eigen/test/cuda_common.h:void run_on_cuda(const Kernel& ker, int n, const Input& in, Output& out)
src/utils/overlap/eigen/test/cuda_common.h:  cudaMalloc((void**)(&d_in),  in_bytes);
src/utils/overlap/eigen/test/cuda_common.h:  cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/test/cuda_common.h:  cudaMemcpy(d_in,  in.data(),  in_bytes,  cudaMemcpyHostToDevice);
src/utils/overlap/eigen/test/cuda_common.h:  cudaMemcpy(d_out, out.data(), out_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/test/cuda_common.h:  cudaThreadSynchronize();
src/utils/overlap/eigen/test/cuda_common.h:  run_on_cuda_meta_kernel<<<Grids,Blocks>>>(ker, n, d_in, d_out);
src/utils/overlap/eigen/test/cuda_common.h:  cudaThreadSynchronize();
src/utils/overlap/eigen/test/cuda_common.h:  cudaMemcpy(const_cast<typename Input::Scalar*>(in.data()),  d_in,  in_bytes,  cudaMemcpyDeviceToHost);
src/utils/overlap/eigen/test/cuda_common.h:  cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);
src/utils/overlap/eigen/test/cuda_common.h:  cudaFree(d_in);
src/utils/overlap/eigen/test/cuda_common.h:  cudaFree(d_out);
src/utils/overlap/eigen/test/cuda_common.h:void run_and_compare_to_cuda(const Kernel& ker, int n, const Input& in, Output& out)
src/utils/overlap/eigen/test/cuda_common.h:  Input  in_ref,  in_cuda;
src/utils/overlap/eigen/test/cuda_common.h:  Output out_ref, out_cuda;
src/utils/overlap/eigen/test/cuda_common.h:  #ifndef __CUDA_ARCH__
src/utils/overlap/eigen/test/cuda_common.h:  in_ref = in_cuda = in;
src/utils/overlap/eigen/test/cuda_common.h:  out_ref = out_cuda = out;
src/utils/overlap/eigen/test/cuda_common.h:  run_on_cuda(ker, n, in_cuda, out_cuda);
src/utils/overlap/eigen/test/cuda_common.h:  #ifndef __CUDA_ARCH__
src/utils/overlap/eigen/test/cuda_common.h:  VERIFY_IS_APPROX(in_ref, in_cuda);
src/utils/overlap/eigen/test/cuda_common.h:  VERIFY_IS_APPROX(out_ref, out_cuda);
src/utils/overlap/eigen/test/cuda_common.h:void ei_test_init_cuda()
src/utils/overlap/eigen/test/cuda_common.h:  cudaDeviceProp deviceProp;
src/utils/overlap/eigen/test/cuda_common.h:  cudaGetDeviceProperties(&deviceProp, device);
src/utils/overlap/eigen/test/cuda_common.h:  std::cout << "CUDA device info:\n";
src/utils/overlap/eigen/test/cuda_common.h:#endif // EIGEN_TEST_CUDA_COMMON_H
src/utils/overlap/eigen/test/half_float.cpp:#include <Eigen/src/Core/arch/CUDA/Half.h>
src/utils/overlap/eigen/test/main.h:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__)
src/utils/overlap/eigen/test/main.h:  #elif !defined(__CUDACC__) // EIGEN_DEBUG_ASSERTS
src/utils/overlap/eigen/test/main.h:  #if !defined(__CUDACC__)
src/utils/overlap/eigen/Eigen/Core:// Handle NVCC/CUDA/SYCL
src/utils/overlap/eigen/Eigen/Core:#if defined(__CUDACC__) || defined(__SYCL_DEVICE_ONLY__)
src/utils/overlap/eigen/Eigen/Core:  // Do not try asserts on CUDA and SYCL!
src/utils/overlap/eigen/Eigen/Core:  // All functions callable from CUDA code must be qualified with __device__
src/utils/overlap/eigen/Eigen/Core:  #ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/Core:    // Do not try to vectorize on CUDA and SYCL!
src/utils/overlap/eigen/Eigen/Core:// When compiling CUDA device code with NVCC, pull in math functions from the
src/utils/overlap/eigen/Eigen/Core:#if defined(__CUDA_ARCH__) && defined(__NVCC__)
src/utils/overlap/eigen/Eigen/Core:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL)
src/utils/overlap/eigen/Eigen/Core:#if defined __CUDACC__
src/utils/overlap/eigen/Eigen/Core:  #define EIGEN_VECTORIZE_CUDA
src/utils/overlap/eigen/Eigen/Core:  #if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
src/utils/overlap/eigen/Eigen/Core:    #define EIGEN_HAS_CUDA_FP16
src/utils/overlap/eigen/Eigen/Core:#if defined EIGEN_HAS_CUDA_FP16
src/utils/overlap/eigen/Eigen/Core:  #include <cuda_fp16.h>
src/utils/overlap/eigen/Eigen/Core:#include "src/Core/arch/CUDA/Half.h"
src/utils/overlap/eigen/Eigen/Core:#include "src/Core/arch/CUDA/PacketMathHalf.h"
src/utils/overlap/eigen/Eigen/Core:#include "src/Core/arch/CUDA/TypeCasting.h"
src/utils/overlap/eigen/Eigen/Core:#if defined EIGEN_VECTORIZE_CUDA
src/utils/overlap/eigen/Eigen/Core:  #include "src/Core/arch/CUDA/PacketMath.h"
src/utils/overlap/eigen/Eigen/Core:  #include "src/Core/arch/CUDA/MathFunctions.h"
src/utils/overlap/eigen/Eigen/Core:// on CUDA devices
src/utils/overlap/eigen/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
src/utils/overlap/eigen/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
src/utils/overlap/eigen/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
src/utils/overlap/eigen/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
src/utils/overlap/eigen/Eigen/src/Core/util/Meta.h:  static float (max)() { return CUDART_MAX_NORMAL_F; }
src/utils/overlap/eigen/Eigen/src/Core/util/Meta.h:  static float infinity() { return CUDART_INF_F; }
src/utils/overlap/eigen/Eigen/src/Core/util/Meta.h:  static float quiet_NaN() { return CUDART_NAN_F; }
src/utils/overlap/eigen/Eigen/src/Core/util/Meta.h:  static double infinity() { return CUDART_INF; }
src/utils/overlap/eigen/Eigen/src/Core/util/Meta.h:  static double quiet_NaN() { return CUDART_NAN; }
src/utils/overlap/eigen/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
src/utils/overlap/eigen/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
src/utils/overlap/eigen/Eigen/src/Core/util/Macros.h:  && ( !defined(__NVCC__) || !EIGEN_ARCH_ARM_OR_ARM64 || (defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000) )
src/utils/overlap/eigen/Eigen/src/Core/util/Macros.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/util/Macros.h:#if EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && defined(__CUDACC_VER__) && (EIGEN_COMP_CLANG || __CUDACC_VER__ >= 70500))
src/utils/overlap/eigen/Eigen/src/Core/util/Macros.h:#if (defined __CUDACC__)
src/utils/overlap/eigen/Eigen/src/Core/util/Macros.h:#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC < 1900 ||  defined(__CUDACC_VER__)) // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
src/utils/overlap/eigen/Eigen/src/Core/util/Macros.h:#  ifdef __CUDA_ARCH__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifndef __CUDA_ARCH__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#if (!defined(__CUDACC__)) && EIGEN_FAST_MATH
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
src/utils/overlap/eigen/Eigen/src/Core/MatrixBase.h:#ifdef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
src/utils/overlap/eigen/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
src/utils/overlap/eigen/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
src/utils/overlap/eigen/Eigen/src/Core/GeneralProduct.h:#ifndef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/GeneralProduct.h:#endif // __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/GenericPacketMath.h:#ifdef __CUDA_ARCH__
src/utils/overlap/eigen/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
src/utils/overlap/eigen/Eigen/src/Core/GenericPacketMath.h:#ifndef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
src/utils/overlap/eigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
src/utils/overlap/eigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
src/utils/overlap/eigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#ifndef EIGEN_PACKET_MATH_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#define EIGEN_PACKET_MATH_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#endif // EIGEN_PACKET_MATH_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:// type Eigen::half (inheriting from CUDA's __half struct) with
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:// to disk and the likes), but fast on GPUs.
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#ifndef EIGEN_HALF_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#define EIGEN_HALF_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#if !defined(EIGEN_HAS_CUDA_FP16)
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:// Make our own __half definition that is similar to CUDA's.
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:  #if !defined(EIGEN_HAS_CUDA_FP16)
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:// Definitions for CPUs and older CUDA, mostly working through conversion
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__)
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Half.h:#endif // EIGEN_HALF_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#ifndef EIGEN_PACKET_MATH_HALF_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#define EIGEN_PACKET_MATH_HALF_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 530
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#endif // EIGEN_PACKET_MATH_HALF_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/MathFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:#define EIGEN_TYPE_CASTING_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Complex.h:// building for CUDA to avoid non-constexpr methods.
src/utils/overlap/eigen/Eigen/src/Core/arch/CUDA/Complex.h:#endif // EIGEN_COMPLEX_CUDA_H
src/utils/overlap/eigen/Eigen/src/SVD/BDCSVD.h:#ifndef __CUDACC__
src/utils/overlap/eigen/doc/Manual.dox:  - \subpage TopicCUDA
src/utils/overlap/eigen/doc/UsingNVCC.dox:/** \page TopicCUDA Using Eigen in CUDA kernels
src/utils/overlap/eigen/doc/UsingNVCC.dox:Staring from CUDA 5.0, the CUDA compiler, \c nvcc, is able to properly parse %Eigen's code (almost).
src/utils/overlap/eigen/doc/UsingNVCC.dox:A few adaptations of the %Eigen's code already allows to use some parts of %Eigen in your own CUDA kernels.
src/utils/overlap/eigen/doc/UsingNVCC.dox:To this end you need the devel branch of %Eigen, CUDA 5.0 or greater with GCC.
src/utils/overlap/eigen/doc/UsingNVCC.dox:    // workaround issue between gcc >= 4.7 and cuda 5.5
src/utils/overlap/eigen/doc/UsingNVCC.dox: - On 64bits system Eigen uses \c long \c int as the default type for indexes and sizes. On CUDA device, it would make sense to default to 32 bits \c int.
src/utils/overlap/eigen/doc/UsingNVCC.dox:   However, to keep host and CUDA code compatible, this cannot be done automatically by %Eigen, and the user is thus required to define \c EIGEN_DEFAULT_DENSE_INDEX_TYPE to \c int throughout his code (or only for CUDA code if there is no interaction between host and CUDA code through %Eigen's object).
src/utils/overlap/eigen/CMakeLists.txt:set(EIGEN_CUDA_COMPUTE_ARCH 30 CACHE STRING "The CUDA compute architecture level to target when compiling CUDA code")
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  float * gpu_in1_data  = static_cast<float*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  float * gpu_in2_data  = static_cast<float*>(sycl_device.allocate(in2.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  float * gpu_in3_data  = static_cast<float*>(sycl_device.allocate(in3.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  float * gpu_out_data =  static_cast<float*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<float, 3>> gpu_in1(gpu_in1_data, tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<float, 3>> gpu_in2(gpu_in2_data, tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<float, 3>> gpu_in3(gpu_in3_data, tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  TensorMap<Tensor<float, 3>> gpu_out(gpu_out_data, tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_in1.device(sycl_device) = gpu_in1.constant(1.2f);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(in1.data(), gpu_in1_data ,(in1.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 * 1.2f;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data ,(out.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in2.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 * gpu_in2;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 + gpu_in2;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in1 * gpu_in1;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) =  gpu_in1 * gpu_in1.constant(3.14f) + gpu_in2 * gpu_in2.constant(2.7f);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(),gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in3_data, in3.data(),(in3.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  gpu_out.device(sycl_device) =(gpu_in1 > gpu_in1.constant(0.5f)).select(gpu_in2, gpu_in3);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_in1_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_in2_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_in3_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_sycl.cpp:  cl::sycl::gpu_selector s;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_random_cuda
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:#define EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:#include <cuda_fp16.h>
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:void test_cuda_random_uniform()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  gpu_out.device(gpu_device) = gpu_out.random();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:void test_cuda_random_normal()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  gpu_out.device(gpu_device) = gpu_out.random(gen);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:void test_cxx11_tensor_random_cuda()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  CALL_SUBTEST(test_cuda_random_uniform());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_random_cuda.cu:  CALL_SUBTEST(test_cuda_random_normal());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_scan_cuda
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:#define EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:#include <cuda_fp16.h>
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:void test_cuda_cumsum(int m_size, int k_size, int n_size)
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:  Tensor<float, 3, DataLayout> t_result_gpu(m_size, k_size, n_size);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:  cudaMalloc((void**)(&d_t_input), t_input_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:  cudaMalloc((void**)(&d_t_result), t_result_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:  cudaMemcpy(d_t_input, t_input.data(), t_input_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:      gpu_t_input(d_t_input, Eigen::array<int, 3>(m_size, k_size, n_size));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:      gpu_t_result(d_t_result, Eigen::array<int, 3>(m_size, k_size, n_size));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:  gpu_t_result.device(gpu_device) = gpu_t_input.cumsum(1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:  cudaMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, cudaMemcpyDeviceToHost);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:    if (fabs(t_result(i) - t_result_gpu(i)) < 1e-4f) {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), 1e-4f)) {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:              << " vs " <<  t_result_gpu(i) << std::endl;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:  cudaFree((void*)d_t_input);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:  cudaFree((void*)d_t_result);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:void test_cxx11_tensor_scan_cuda()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:  CALL_SUBTEST_1(test_cuda_cumsum<ColMajor>(128, 128, 128));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_scan_cuda.cu:  CALL_SUBTEST_2(test_cuda_cumsum<RowMajor>(128, 128, 128));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_cast_float16_cuda
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:#define EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:#include <cuda_fp16.h>
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:void test_cuda_conversion() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  Eigen::half* d_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  float* d_conv = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_half(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_conv(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.memcpyHostToDevice(d_float, floats.data(), num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_half.device(gpu_device) = gpu_float.cast<Eigen::half>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_conv.device(gpu_device) = gpu_half.cast<float>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(initial.data(), d_float, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(final.data(), d_conv, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.synchronize();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.deallocate(d_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.deallocate(d_half);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  gpu_device.deallocate(d_conv);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:void test_cxx11_tensor_cast_float16_cuda()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cast_float16_cuda.cu:  CALL_SUBTEST(test_cuda_conversion());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  float * gpu_in_data  = static_cast<float*>(sycl_device.allocate(input.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  float * gpu_out_data  = static_cast<float*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  TensorMap<Tensor<float, 4>>  gpu_in(gpu_in_data, in_range);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  TensorMap<Tensor<float, 4>> gpu_out(gpu_out_data, out_range);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, input.data(),(input.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  gpu_out.device(sycl_device) = gpu_in.broadcast(broadcasts);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_broadcast_sycl.cpp:  cl::sycl::gpu_selector s;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_of_float16_cuda
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:#define EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:#include <cuda_fp16.h>
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_numext() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  bool* d_res_half = (bool*)gpu_device.allocate(num_elem * sizeof(bool));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  bool* d_res_float = (bool*)gpu_device.allocate(num_elem * sizeof(bool));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_res_half(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_res_float(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float.unaryExpr(Eigen::internal::scalar_isnan_op<float>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = gpu_float.cast<Eigen::half>().unaryExpr(Eigen::internal::scalar_isnan_op<Eigen::half>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(bool));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(bool));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:#ifdef EIGEN_HAS_CUDA_FP16
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_conversion() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_conv = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_half(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_conv(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float.device(gpu_device) = gpu_float.random();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_half.device(gpu_device) = gpu_float.cast<Eigen::half>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_conv.device(gpu_device) = gpu_half.cast<float>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(initial.data(), d_float, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(final.data(), d_conv, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_half);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_conv);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_unary() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_half = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float.abs();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().cast<float>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_elementwise() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_half = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float1(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float2(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float1.device(gpu_device) = gpu_float1.random();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float2.device(gpu_device) = gpu_float2.random();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = (gpu_float1 + gpu_float2) * gpu_float1;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = ((gpu_float1.cast<Eigen::half>() + gpu_float2.cast<Eigen::half>()) * gpu_float1.cast<Eigen::half>()).cast<float>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float2);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_trancendental() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float3 = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res1_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res1_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res2_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res2_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res3_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res3_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float1(d_float1, num_elem);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float2(d_float2, num_elem);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float3(d_float3, num_elem);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res1_half(d_res1_half, num_elem);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res1_float(d_res1_float, num_elem);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res2_half(d_res2_half, num_elem);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res2_float(d_res2_float, num_elem);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res3_half(d_res3_half, num_elem);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res3_float(d_res3_float, num_elem);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float1.device(gpu_device) = gpu_float1.random() - gpu_float1.constant(0.5f);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float2.device(gpu_device) = gpu_float2.random() + gpu_float1.constant(0.5f);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float3.device(gpu_device) = gpu_float3.random();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res1_float.device(gpu_device) = gpu_float1.exp().cast<Eigen::half>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res2_float.device(gpu_device) = gpu_float2.log().cast<Eigen::half>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res3_float.device(gpu_device) = gpu_float3.log1p().cast<Eigen::half>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res1_half.device(gpu_device) = gpu_float1.cast<Eigen::half>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res1_half.device(gpu_device) = gpu_res1_half.exp();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res2_half.device(gpu_device) = gpu_float2.cast<Eigen::half>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res2_half.device(gpu_device) = gpu_res2_half.log();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res3_half.device(gpu_device) = gpu_float3.cast<Eigen::half>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res3_half.device(gpu_device) = gpu_res3_half.log1p();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(input1.data(), d_float1, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(input2.data(), d_float2, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(input3.data(), d_float3, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec1.data(), d_res1_half, num_elem*sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec1.data(), d_res1_float, num_elem*sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec2.data(), d_res2_half, num_elem*sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec2.data(), d_res2_float, num_elem*sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec3.data(), d_res3_half, num_elem*sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec3.data(), d_res3_float, num_elem*sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float2);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float3);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res1_half);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res1_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res2_half);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res2_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res3_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res3_half);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_contractions() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(num_elem * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2>, Eigen::Aligned> gpu_res_half(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2>, Eigen::Aligned> gpu_res_float(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float1.device(gpu_device) = gpu_float1.random() - gpu_float1.constant(0.5f);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float2.device(gpu_device) = gpu_float2.random() - gpu_float2.constant(0.5f);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float1.contract(gpu_float2, dims).cast<Eigen::half>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().contract(gpu_float2.cast<Eigen::half>(), dims);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, num_elem*sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float2);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_reductions(int size1, int size2, int redux) {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(result_size * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(result_size * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res_half(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1>, Eigen::Aligned> gpu_res_float(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float1.device(gpu_device) = gpu_float1.random() * 2.0f;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float2.device(gpu_device) = gpu_float2.random() * 2.0f;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float1.sum(redux_dim).cast<Eigen::half>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().sum(redux_dim);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, result_size*sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, result_size*sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float2);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_reductions() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  test_cuda_reductions<void>(13, 13, 0);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  test_cuda_reductions<void>(13, 13, 1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  test_cuda_reductions<void>(35, 36, 0);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  test_cuda_reductions<void>(35, 36, 1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  test_cuda_reductions<void>(36, 35, 0);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  test_cuda_reductions<void>(36, 35, 1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_full_reductions() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res_half = (Eigen::half*)gpu_device.allocate(1 * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::half* d_res_float = (Eigen::half*)gpu_device.allocate(1 * sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float1(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2>, Eigen::Aligned> gpu_float2(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 0>, Eigen::Aligned> gpu_res_half(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Eigen::half, 0>, Eigen::Aligned> gpu_res_float(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float1.device(gpu_device) = gpu_float1.random();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float2.device(gpu_device) = gpu_float2.random();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float1.sum().cast<Eigen::half>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().sum();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float1.maximum().cast<Eigen::half>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half.device(gpu_device) = gpu_float1.cast<Eigen::half>().maximum();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec.data(), d_res_half, sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, sizeof(Eigen::half));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float2);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cuda_forced_evals() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_half1 = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_half2 = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  float* d_res_float = (float*)gpu_device.allocate(num_elem * sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_float(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_half1(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu: Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Unaligned> gpu_res_half2(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_res_float(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_float.device(gpu_device) = gpu_float.random() - gpu_float.constant(0.5f);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_float.device(gpu_device) = gpu_float.abs();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half1.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().eval().cast<float>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_res_half2.device(gpu_device) = gpu_float.cast<Eigen::half>().abs().broadcast(no_bcast).eval().cast<float>();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec1.data(), d_res_half1, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(half_prec2.data(), d_res_half1, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.memcpyDeviceToHost(full_prec.data(), d_res_float, num_elem*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.synchronize();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_half2);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  gpu_device.deallocate(d_res_float);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:void test_cxx11_tensor_of_float16_cuda()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_1(test_cuda_numext<void>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:#ifdef EIGEN_HAS_CUDA_FP16
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_1(test_cuda_conversion<void>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_1(test_cuda_unary<void>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_1(test_cuda_elementwise<void>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_1(test_cuda_trancendental<void>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_2(test_cuda_contractions<void>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_3(test_cuda_reductions<void>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_4(test_cuda_full_reductions<void>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  CALL_SUBTEST_5(test_cuda_forced_evals<void>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_float16_cuda.cu:  std::cout << "Half floats are not supported by this version of cuda: skipping the test" << std::endl;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_cuda
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:#define EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:#include <cuda_fp16.h>
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:void test_cuda_contraction(int m_size, int k_size, int n_size)
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  // a 15 SM GK110 GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  Tensor<float, 2, DataLayout> t_result_gpu(m_size, n_size);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMalloc((void**)(&d_t_left), t_left_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMalloc((void**)(&d_t_right), t_right_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMalloc((void**)(&d_t_result), t_result_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMemcpy(d_t_left, t_left.data(), t_left_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMemcpy(d_t_right, t_right.data(), t_right_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:      gpu_t_left(d_t_left, Eigen::array<int, 2>(m_size, k_size));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:      gpu_t_right(d_t_right, Eigen::array<int, 2>(k_size, n_size));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:      gpu_t_result(d_t_result, Eigen::array<int, 2>(m_size, n_size));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, cudaMemcpyDeviceToHost);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:    if (fabs(t_result(i) - t_result_gpu(i)) < 1e-4f) {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), 1e-4f)) {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:              << " vs " <<  t_result_gpu(i) << std::endl;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaFree((void*)d_t_left);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaFree((void*)d_t_right);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaFree((void*)d_t_result);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  // a 15 SM GK110 GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  Tensor<float, 0, DataLayout> t_result_gpu;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMalloc((void**)(&d_t_left), t_left_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMalloc((void**)(&d_t_right), t_right_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMalloc((void**)(&d_t_result), t_result_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMemcpy(d_t_left, t_left.data(), t_left_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMemcpy(d_t_right, t_right.data(), t_right_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:      gpu_t_left(d_t_left, m_size, k_size);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:      gpu_t_right(d_t_right, k_size, n_size);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:      gpu_t_result(d_t_result);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, cudaMemcpyDeviceToHost);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  if (fabs(t_result() - t_result_gpu()) > 1e-4f &&
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:      !Eigen::internal::isApprox(t_result(), t_result_gpu(), 1e-4f)) {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:              << " vs " <<  t_result_gpu() << std::endl;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaFree((void*)d_t_left);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaFree((void*)d_t_right);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  cudaFree((void*)d_t_result);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:void test_cuda_contraction_m() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:    test_cuda_contraction<ColMajor>(k, 128, 128);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:    test_cuda_contraction<RowMajor>(k, 128, 128);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:void test_cuda_contraction_k() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:    test_cuda_contraction<ColMajor>(128, k, 128);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:    test_cuda_contraction<RowMajor>(128, k, 128);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:void test_cuda_contraction_n() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:    test_cuda_contraction<ColMajor>(128, 128, k);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:    test_cuda_contraction<RowMajor>(128, 128, k);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:void test_cuda_contraction_sizes() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:        test_cuda_contraction<DataLayout>(m_sizes[i], n_sizes[j], k_sizes[k]);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:void test_cxx11_tensor_cuda()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_1(test_cuda_contraction<ColMajor>(128, 128, 128));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_1(test_cuda_contraction<RowMajor>(128, 128, 128));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_2(test_cuda_contraction_m<ColMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_3(test_cuda_contraction_m<RowMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_4(test_cuda_contraction_k<ColMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_5(test_cuda_contraction_k<RowMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_6(test_cuda_contraction_n<ColMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_7(test_cuda_contraction_n<RowMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_8(test_cuda_contraction_sizes<ColMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_contract_cuda.cu:  CALL_SUBTEST_9(test_cuda_contraction_sizes<RowMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:#define EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:#include <cuda_fp16.h>
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:// Context for evaluation on GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:struct GPUContext {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  GPUContext(const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in1, Eigen::TensorMap<Eigen::Tensor<float, 3> >& in2, Eigen::TensorMap<Eigen::Tensor<float, 3> >& out) : in1_(in1), in2_(in2), out_(out), gpu_device_(&stream_) {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(cudaMalloc((void**)(&kernel_1d_), 2*sizeof(float)) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(cudaMemcpy(kernel_1d_, kernel_1d_val, 2*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(cudaMalloc((void**)(&kernel_2d_), 4*sizeof(float)) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(cudaMemcpy(kernel_2d_, kernel_2d_val, 4*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(cudaMalloc((void**)(&kernel_3d_), 8*sizeof(float)) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(cudaMemcpy(kernel_3d_, kernel_3d_val, 8*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  ~GPUContext() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(cudaFree(kernel_1d_) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(cudaFree(kernel_2d_) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:    assert(cudaFree(kernel_3d_) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  const Eigen::GpuDevice& device() const { return gpu_device_; }
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::CudaStreamDevice stream_;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::GpuDevice gpu_device_;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:void test_gpu() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  cudaMalloc((void**)(&d_in1), in1_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  cudaMalloc((void**)(&d_in2), in2_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, 40,50,70);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, 40,50,70);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, 40,50,70);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  GPUContext context(gpu_in1, gpu_in2, gpu_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, context.device().stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(cudaStreamSynchronize(context.device().stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, context.device().stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(cudaStreamSynchronize(context.device().stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, context.device().stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  assert(cudaStreamSynchronize(context.device().stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device.cu:  CALL_SUBTEST_2(test_gpu());
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:find_package(CUDA 7.0)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:if(CUDA_FOUND AND EIGEN_TEST_CUDA)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  # in the CUDA runtime
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  message(STATUS "Flags used to compile cuda code: " ${CMAKE_CXX_FLAGS})
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:    set(CUDA_NVCC_FLAGS "-ccbin ${CMAKE_C_COMPILER}" CACHE STRING "nvcc flags" FORCE)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  if(EIGEN_TEST_CUDA_CLANG)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 --cuda-gpu-arch=sm_${EIGEN_CUDA_COMPUTE_ARCH}")
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  set(EIGEN_CUDA_RELAXED_CONSTEXPR "--expt-relaxed-constexpr")
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  if (${CUDA_VERSION} STREQUAL "7.0")
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:    set(EIGEN_CUDA_RELAXED_CONSTEXPR "--relaxed-constexpr")
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:    set(EIGEN_CUDA_CXX11_FLAG "-std=c++11")
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:    set(EIGEN_CUDA_CXX11_FLAG "")
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  set(CUDA_NVCC_FLAGS  "${EIGEN_CUDA_CXX11_FLAG} ${EIGEN_CUDA_RELAXED_CONSTEXPR} -arch compute_${EIGEN_CUDA_COMPUTE_ARCH} -Xcudafe \"--display_error_number\" ${CUDA_NVCC_FLAGS}")
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  cuda_include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/include")
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_complex_cuda)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_complex_cwise_ops_cuda)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_reduction_cuda)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_argmax_cuda)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_cast_float16_cuda)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  ei_add_test(cxx11_tensor_scan_cuda)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  if (${EIGEN_CUDA_COMPUTE_ARCH} GREATER 29)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_cuda)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_contract_cuda)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_of_float16_cuda)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:  if (${EIGEN_CUDA_COMPUTE_ARCH} GREATER 34)
src/utils/overlap/eigen/unsupported/test/CMakeLists.txt:    ei_add_test(cxx11_tensor_random_cuda)
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_device_sycl.cpp:  cl::sycl::gpu_selector s;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:#define EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:#include <cuda_fp16.h>
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:void test_cuda_complex_cwise_ops() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  cudaMalloc((void**)(&d_in1), complex_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  cudaMalloc((void**)(&d_in2), complex_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  cudaMalloc((void**)(&d_out), complex_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_in1(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_in2(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, 1, 0, int>, Eigen::Aligned> gpu_out(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  gpu_in1.device(gpu_device) = gpu_in1.constant(a);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  gpu_in2.device(gpu_device) = gpu_in2.constant(b);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:        gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:        gpu_out.device(gpu_device) = gpu_in1 - gpu_in2;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:        gpu_out.device(gpu_device) = gpu_in1 * gpu_in2;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:        gpu_out.device(gpu_device) = gpu_in1 / gpu_in2;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:    assert(cudaMemcpyAsync(actual.data(), d_out, complex_bytes, cudaMemcpyDeviceToHost,
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:                           gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  cudaFree(d_in1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  cudaFree(d_in2);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  CALL_SUBTEST(test_cuda_complex_cwise_ops<float>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cwise_ops_cuda.cu:  CALL_SUBTEST(test_cuda_complex_cwise_ops<double>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:#define EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:#include <cuda_fp16.h>
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:void test_cuda_nullary() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaMalloc((void**)(&d_in1), complex_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaMalloc((void**)(&d_in2), complex_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaMalloc((void**)(&d_out2), float_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaMemcpy(d_in1, in1.data(), complex_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaMemcpy(d_in2, in2.data(), complex_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1, 0, int>, Eigen::Aligned> gpu_in1(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<std::complex<float>, 1, 0, int>, Eigen::Aligned> gpu_in2(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_out2(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_in1.device(gpu_device) = gpu_in1.constant(std::complex<float>(3.14f, 2.7f));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_out2.device(gpu_device) = gpu_in2.abs();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  assert(cudaMemcpyAsync(new1.data(), d_in1, complex_bytes, cudaMemcpyDeviceToHost,
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:                         gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  assert(cudaMemcpyAsync(new2.data(), d_out2, float_bytes, cudaMemcpyDeviceToHost,
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:                         gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaFree(d_in1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaFree(d_in2);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  cudaFree(d_out2);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:static void test_cuda_sum_reductions() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  std::complex<float>* gpu_in_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(in_bytes));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  std::complex<float>* gpu_out_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(out_bytes));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  TensorMap<Tensor<std::complex<float>, 2> > in_gpu(gpu_in_ptr, num_rows, num_cols);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  TensorMap<Tensor<std::complex<float>, 0> > out_gpu(gpu_out_ptr);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  out_gpu.device(gpu_device) = in_gpu.sum();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  Tensor<std::complex<float>, 0> full_redux_gpu;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.synchronize();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  // Check that the CPU and GPU reductions return the same result.
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.deallocate(gpu_in_ptr);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.deallocate(gpu_out_ptr);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:static void test_cuda_product_reductions() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  std::complex<float>* gpu_in_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(in_bytes));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  std::complex<float>* gpu_out_ptr = static_cast<std::complex<float>*>(gpu_device.allocate(out_bytes));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  TensorMap<Tensor<std::complex<float>, 2> > in_gpu(gpu_in_ptr, num_rows, num_cols);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  TensorMap<Tensor<std::complex<float>, 0> > out_gpu(gpu_out_ptr);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  out_gpu.device(gpu_device) = in_gpu.prod();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  Tensor<std::complex<float>, 0> full_redux_gpu;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.synchronize();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  // Check that the CPU and GPU reductions return the same result.
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.deallocate(gpu_in_ptr);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  gpu_device.deallocate(gpu_out_ptr);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  CALL_SUBTEST(test_cuda_nullary());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  CALL_SUBTEST(test_cuda_sum_reductions());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_complex_cuda.cu:  CALL_SUBTEST(test_cuda_product_reductions());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<float, 0> full_redux_gpu;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  float* gpu_in_data = static_cast<float*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  float* gpu_out_data =(float*)sycl_device.allocate(sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<float, 2> >  in_gpu(gpu_in_data, tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<float, 0> >  out_gpu(gpu_out_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.sum();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_data, sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  VERIFY_IS_APPROX(full_redux_gpu(), full_redux());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<float, 2> redux_gpu(reduced_tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  float* gpu_in_data = static_cast<float*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  float* gpu_out_data = static_cast<float*>(sycl_device.allocate(redux_gpu.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<float, 3> >  in_gpu(gpu_in_data, tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<float, 2> >  out_gpu(gpu_out_data, reduced_tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.sum(red_axis);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(redux_gpu.data(), gpu_out_data, redux_gpu.dimensions().TotalSize()*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:      VERIFY_IS_APPROX(redux_gpu(j,k), redux(j,k));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  Tensor<float, 2> redux_gpu(reduced_tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  float* gpu_in_data = static_cast<float*>(sycl_device.allocate(in.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  float* gpu_out_data = static_cast<float*>(sycl_device.allocate(redux_gpu.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<float, 3> >  in_gpu(gpu_in_data, tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  TensorMap<Tensor<float, 2> >  out_gpu(gpu_out_data, reduced_tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in_data, in.data(),(in.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  out_gpu.device(sycl_device) = in_gpu.sum(red_axis);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.memcpyDeviceToHost(redux_gpu.data(), gpu_out_data, redux_gpu.dimensions().TotalSize()*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  // Check that the CPU and GPU reductions return the same result.
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:      VERIFY_IS_APPROX(redux_gpu(j,k), redux(j,k));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_in_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_sycl.cpp:  cl::sycl::gpu_selector s;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_cuda
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:#define EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:#include <cuda_fp16.h>
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:void test_cuda_simple_argmax()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaMalloc((void**)(&d_in), in_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaMalloc((void**)(&d_out_max), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaMalloc((void**)(&d_out_min), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaMemcpy(d_in, in.data(), in_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<double, 3, Layout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 3>(72,53,97));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 1, Layout>, Aligned > gpu_out_max(d_out_max, Eigen::array<DenseIndex, 1>(1));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 1, Layout>, Aligned > gpu_out_min(d_out_min, Eigen::array<DenseIndex, 1>(1));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  gpu_out_max.device(gpu_device) = gpu_in.argmax();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  gpu_out_min.device(gpu_device) = gpu_in.argmin();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  assert(cudaMemcpyAsync(out_max.data(), d_out_max, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  assert(cudaMemcpyAsync(out_min.data(), d_out_min, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaFree(d_in);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaFree(d_out_max);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  cudaFree(d_out_min);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:void test_cuda_argmax_dim()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMalloc((void**)(&d_in), in_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMemcpy(d_in, tensor.data(), in_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 4>(2, 3, 5, 7));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout>, Aligned > gpu_out(d_out, out_shape);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    gpu_out.device(gpu_device) = gpu_in.argmax(dim);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaMemcpyAsync(tensor_arg.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMemcpy(d_in, tensor.data(), in_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    gpu_out.device(gpu_device) = gpu_in.argmax(dim);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaMemcpyAsync(tensor_arg.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaFree(d_in);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:void test_cuda_argmin_dim()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMalloc((void**)(&d_in), in_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMemcpy(d_in, tensor.data(), in_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 4>(2, 3, 5, 7));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout>, Aligned > gpu_out(d_out, out_shape);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    gpu_out.device(gpu_device) = gpu_in.argmin(dim);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaMemcpyAsync(tensor_arg.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaMemcpy(d_in, tensor.data(), in_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    gpu_out.device(gpu_device) = gpu_in.argmin(dim);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaMemcpyAsync(tensor_arg.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaFree(d_in);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:    cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:void test_cxx11_tensor_cuda()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  CALL_SUBTEST_1(test_cuda_simple_argmax<RowMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  CALL_SUBTEST_1(test_cuda_simple_argmax<ColMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  CALL_SUBTEST_2(test_cuda_argmax_dim<RowMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  CALL_SUBTEST_2(test_cuda_argmax_dim<ColMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  CALL_SUBTEST_3(test_cuda_argmin_dim<RowMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_argmax_cuda.cu:  CALL_SUBTEST_3(test_cuda_argmin_dim<ColMajor>());
src/utils/overlap/eigen/unsupported/test/openglsupport.cpp:    #ifdef GLEW_ARB_gpu_shader_fp64
src/utils/overlap/eigen/unsupported/test/openglsupport.cpp:    if(GLEW_ARB_gpu_shader_fp64)
src/utils/overlap/eigen/unsupported/test/openglsupport.cpp:      #ifdef GL_ARB_gpu_shader_fp64
src/utils/overlap/eigen/unsupported/test/openglsupport.cpp:      std::cerr << "Warning: GLEW_ARB_gpu_shader_fp64 was not tested\n";
src/utils/overlap/eigen/unsupported/test/openglsupport.cpp:      std::cerr << "Warning: GLEW_ARB_gpu_shader_fp64 was not tested\n";
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_of_strings.cpp:  // Beware: none of this is likely to ever work on a GPU.
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_cuda
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:#define EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:#include <cuda_fp16.h>
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_nullary() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in1), tensor_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in2), tensor_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in1, in1.data(), tensor_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in2, in2.data(), tensor_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_in1(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, 0, int>, Eigen::Aligned> gpu_in2(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_in1.device(gpu_device) = gpu_in1.constant(3.14f);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_in2.device(gpu_device) = gpu_in2.random();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(new1.data(), d_in1, tensor_bytes, cudaMemcpyDeviceToHost,
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:                         gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(new2.data(), d_in2, tensor_bytes, cudaMemcpyDeviceToHost,
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:                         gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in2);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_elementwise_small() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in1), in1_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in2), in2_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in2(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_out(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost,
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:                         gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in2);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_elementwise()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in1), in1_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in2), in2_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in3), in3_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in3, in3.data(), in3_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in3(d_in3, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, Eigen::array<Eigen::DenseIndex, 3>(72,53,97));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in1 + gpu_in2 * gpu_in3;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in2);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in3);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_props() {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in1), in1_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<bool, 1>, Eigen::Aligned> gpu_out(
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = (gpu_in1.isnan)();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost,
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:                         gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_reduction()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in1), in1_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4> > gpu_in1(d_in1, 72,53,97,113);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in1.maximum(reduction_axis);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_contraction()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  // a 15 SM GK110 GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_t_left), t_left_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_t_right), t_right_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_t_result), t_result_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_t_left, t_left.data(), t_left_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_t_right, t_right.data(), t_right_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_t_left(d_t_left, 6, 50, 3, 31);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_t_right(d_t_right, 3, 31, 7, 20, 1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_t_result(d_t_result, 6, 50, 7, 20, 1);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(t_result.data(), d_t_result, t_result_bytes, cudaMemcpyDeviceToHost);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_t_left);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_t_right);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_t_result);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_convolution_1d()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_input), input_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_kernel), kernel_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_input(d_input, 74,37,11,137);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, DataLayout> > gpu_kernel(d_kernel, 4);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_out(d_out, 74,34,11,137);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_input);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_kernel);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_convolution_inner_dim_col_major_1d()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_input), input_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_kernel), kernel_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, ColMajor> > gpu_input(d_input,74,9,11,7);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, ColMajor> > gpu_kernel(d_kernel,4);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, ColMajor> > gpu_out(d_out,71,9,11,7);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_input);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_kernel);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_convolution_inner_dim_row_major_1d()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_input), input_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_kernel), kernel_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, RowMajor> > gpu_input(d_input, 7,9,11,74);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 1, RowMajor> > gpu_kernel(d_kernel, 4);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, RowMajor> > gpu_out(d_out, 7,9,11,71);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_input);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_kernel);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_convolution_2d()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_input), input_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_kernel), kernel_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_input(d_input,74,37,11,137);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 2, DataLayout> > gpu_kernel(d_kernel,3,4);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout> > gpu_out(d_out,74,35,8,137);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_input);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_kernel);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_convolution_3d()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_input), input_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_kernel), kernel_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;    
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_input(d_input,74,37,11,137,17);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 3, DataLayout> > gpu_kernel(d_kernel,3,4,2);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<float, 5, DataLayout> > gpu_out(d_out,74,35,8,136,17);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_input.convolve(gpu_kernel, dims);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_input);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_kernel);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_lgamma(const Scalar stddev)
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in.lgamma();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_digamma()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in(d_in, 7);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 7);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in.digamma();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_zeta()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_x), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_q), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_x, in_x.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_q, in_q.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 6);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_q(d_in_q, 6);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 6);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in_x.zeta(gpu_in_q);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_x);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_q);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_polygamma()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_x), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_n), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_x, in_x.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_n, in_n.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 7);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_n(d_in_n, 7);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 7);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in_n.polygamma(gpu_in_x);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_x);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_n);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_igamma()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMalloc((void**)(&d_a), bytes) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMalloc((void**)(&d_x), bytes) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMalloc((void**)(&d_out), bytes) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_a(d_a, 6, 6);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_x(d_x, 6, 6);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 6, 6);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_a.igamma(gpu_x);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_a);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_x);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_igammac()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_a), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_x), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_a(d_a, 6, 6);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_x(d_x, 6, 6);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 6, 6);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_a.igammac(gpu_x);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_a);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_x);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_erf(const Scalar stddev)
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMalloc((void**)(&d_in), bytes) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMalloc((void**)(&d_out), bytes) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in.erf();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_erfc(const Scalar stddev)
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_in(d_in, 72, 97);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, 72, 97);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = gpu_in.erfc();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cuda_betainc()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_x), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_a), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_in_b), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMalloc((void**)(&d_out), bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_x, in_x.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_a, in_a.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaMemcpy(d_in_b, in_b.data(), bytes, cudaMemcpyHostToDevice);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_x(d_in_x, 125);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_a(d_in_a, 125);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_in_b(d_in_b, 125);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Scalar, 1> > gpu_out(d_out, 125);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  gpu_out.device(gpu_device) = betainc(gpu_in_a, gpu_in_b, gpu_in_x);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_x);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_a);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_in_b);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  cudaFree(d_out);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:void test_cxx11_tensor_cuda()
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_1(test_cuda_nullary());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_1(test_cuda_elementwise_small());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_1(test_cuda_elementwise());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_1(test_cuda_props());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_1(test_cuda_reduction());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_2(test_cuda_contraction<ColMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_2(test_cuda_contraction<RowMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_1d<ColMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_1d<RowMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_inner_dim_col_major_1d());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_inner_dim_row_major_1d());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_2d<ColMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_2d<RowMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_3d<ColMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_3(test_cuda_convolution_3d<RowMajor>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<float>(1.0f));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<float>(100.0f));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<float>(0.01f));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<float>(0.001f));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<double>(1.0));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<double>(100.0));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<double>(0.01));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_lgamma<double>(0.001));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<float>(1.0f));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<float>(100.0f));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<float>(0.01f));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<float>(0.001f));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<float>(1.0f));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  // CALL_SUBTEST(test_cuda_erfc<float>(100.0f));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<float>(5.0f)); // CUDA erfc lacks precision for large inputs
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<float>(0.01f));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<float>(0.001f));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<double>(1.0));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<double>(100.0));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<double>(0.01));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erf<double>(0.001));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<double>(1.0));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  // CALL_SUBTEST(test_cuda_erfc<double>(100.0));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<double>(5.0)); // CUDA erfc lacks precision for large inputs
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<double>(0.01));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_4(test_cuda_erfc<double>(0.001));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_digamma<float>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_digamma<double>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_polygamma<float>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_polygamma<double>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_zeta<float>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_zeta<double>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_igamma<float>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_igammac<float>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_igamma<double>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_5(test_cuda_igammac<double>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_6(test_cuda_betainc<float>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_cuda.cu:  CALL_SUBTEST_6(test_cuda_betainc<double>());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  float * gpu_in1_data  = static_cast<float*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  float * gpu_in2_data  = static_cast<float*>(sycl_device.allocate(in2.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  float * gpu_out_data =  static_cast<float*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(float)));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<float, 3>> gpu_in1(gpu_in1_data, tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<float, 3>> gpu_in2(gpu_in2_data, tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  Eigen::TensorMap<Eigen::Tensor<float, 3>> gpu_out(gpu_out_data, tensorRange);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in1_data, in1.data(),(in1.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in1.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  gpu_out.device(sycl_device) =(gpu_in1 + gpu_in2).eval() * gpu_in2;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(float));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.deallocate(gpu_in1_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.deallocate(gpu_in2_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  sycl_device.deallocate(gpu_out_data);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_forced_eval_sycl.cpp:  cl::sycl::gpu_selector s;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:#define EIGEN_TEST_FUNC cxx11_tensor_reduction_cuda
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:#define EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:#include <cuda_fp16.h>
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::GpuDevice gpu_device(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Type* gpu_in_ptr = static_cast<Type*>(gpu_device.allocate(in_bytes));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Type* gpu_out_ptr = static_cast<Type*>(gpu_device.allocate(out_bytes));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  TensorMap<Tensor<Type, 2, DataLayout> > in_gpu(gpu_in_ptr, num_rows, num_cols);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  TensorMap<Tensor<Type, 0, DataLayout> > out_gpu(gpu_out_ptr);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  out_gpu.device(gpu_device) = in_gpu.sum();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Tensor<Type, 0, DataLayout> full_redux_gpu;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_device.synchronize();
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  // Check that the CPU and GPU reductions return the same result.
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_device.deallocate(gpu_in_ptr);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_device.deallocate(gpu_out_ptr);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::GpuDevice dev(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 3, DataLayout> > gpu_in(in_data, dim_x, dim_y, dim_z);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 2, DataLayout> > gpu_out(out_data, dim_y, dim_z);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_out.device(dev) = gpu_in.sum(red_axis);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_out.device(dev) += gpu_in.sum(red_axis);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Tensor<Type, 2, DataLayout> redux_gpu(dim_y, dim_z);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  dev.memcpyDeviceToHost(redux_gpu.data(), out_data, gpu_out.size()*sizeof(Type));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  // Check that the CPU and GPU reductions return the same result.
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  for (int i = 0; i < gpu_out.size(); ++i) {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:    VERIFY_IS_APPROX(2*redux(i), redux_gpu(i));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::CudaStreamDevice stream;
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::GpuDevice dev(&stream);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 3, DataLayout> > gpu_in(in_data, dim_x, dim_y, dim_z);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Eigen::TensorMap<Eigen::Tensor<Type, 2, DataLayout> > gpu_out(out_data, dim_x, dim_y);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_out.device(dev) = gpu_in.sum(red_axis);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  gpu_out.device(dev) += gpu_in.sum(red_axis);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  Tensor<Type, 2, DataLayout> redux_gpu(dim_x, dim_y);
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  dev.memcpyDeviceToHost(redux_gpu.data(), out_data, gpu_out.size()*sizeof(Type));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  // Check that the CPU and GPU reductions return the same result.
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:  for (int i = 0; i < gpu_out.size(); ++i) {
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:    VERIFY_IS_APPROX(2*redux(i), redux_gpu(i));
src/utils/overlap/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu:void test_cxx11_tensor_reduction_cuda() {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/Tensor:#ifdef EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/Eigen/CXX11/Tensor:#include <cuda_runtime.h>
src/utils/overlap/eigen/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorDeviceCuda.h"
src/utils/overlap/eigen/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorReductionCuda.h"
src/utils/overlap/eigen/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorContractionCuda.h"
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:#ifdef __CUDA_ARCH__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:// GPU: the evaluation of the expression is offloaded to a GPU.
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#if defined(EIGEN_USE_GPU)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:class TensorExecutor<Expression, GpuDevice, Vectorizable> {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:  static void run(const Expression& expr, const GpuDevice& device);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#if defined(__CUDACC__)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:inline void TensorExecutor<Expression, GpuDevice, Vectorizable>::run(
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const Expression& expr, const GpuDevice& device) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:  TensorEvaluator<Expression, GpuDevice> evaluator(expr, device);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const int block_size = device.maxCudaThreadsPerBlock();
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    LAUNCH_CUDA_KERNEL(
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:        (EigenMetaKernel<TensorEvaluator<Expression, GpuDevice>, Index>),
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#endif  // __CUDACC__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#endif  // EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef __CUDA_ARCH__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef __CUDA_ARCH__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__CUDA_ARCH__)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__CUDA_ARCH__)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__SIZEOF_INT128__) && !defined(__CUDA_ARCH__)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef __CUDA_ARCH__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device, return the amount of shared memory available.
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    return __CUDA_ARCH__ / 100;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDevice.h:  * on the specified computing 'device' (GPU, thread pool, ...)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDevice.h:  *    C.device(EIGEN_GPU) = A + B;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:// Full reducers for GPU, don't vectorize for now
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:// Reducer function that enables multiple cuda thread to safely accumulate at the same
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:// attempts to update it with the new value. If in the meantime another cuda thread
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((FullReductionKernel<block_size, num_per_thread, Self, Op, Index>),
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitFullReduxKernelHalfFloat<Self, Op, Index>),
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((FullReductionKernelHalfFloat<block_size, num_per_thread, Self, Op, Index>),
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionCleanupKernelHalfFloat<Op>),
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:struct FullReducer<Self, Op, GpuDevice, Vectorizable> {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    FullReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static EIGEN_DEVICE_FUNC bool run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index, typename Self::Index) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    assert(false && "Should only be called to reduce doubles, floats and half floats on a gpu device");
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      const int max_blocks = device.getNumCudaMultiProcessors() *
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / 1024;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitKernel<OutputType, Index>),
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((InnerReductionKernel<num_per_thread, Self, Op, Index>),
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index, typename Self::Index) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      const int max_blocks = device.getNumCudaMultiProcessors() *
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / 1024;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitKernelHalfFloat<Self, Op, Index>),
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((InnerReductionKernelHalfFloat<num_per_thread, Self, Op, Index>),
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:struct InnerReducer<Self, Op, GpuDevice> {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    return InnerReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs_to_reduce, num_preserved_vals);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:struct OuterReducer<Self, Op, GpuDevice> {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    assert(false && "Should only be called to reduce doubles or floats on a gpu device");
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, float* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      const int max_blocks = device.getNumCudaMultiProcessors() *
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                             device.maxCudaThreadsPerMultiProcessor() / 1024;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitKernel<float, Index>),
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((OuterReductionKernel<num_per_thread, Self, Op, Index>),
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#endif // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:GPU using cuda.  Additional implementations may be added later.
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:GpuDevice.
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:#### Evaluating On GPU
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:You need to create a GPU device but you also need to explicitly allocate the
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:memory for tensors with cuda.
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:*   On GPUs only floating point values are properly tested and optimized for.
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/README.md:*   Complex and integer values are known to be broken on GPUs. If you try to use
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:// It is very expensive to start the memcpy kernel on GPU: we therefore only
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:#ifdef EIGEN_USE_GPU
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:template <typename Index> struct MemcpyTriggerForSlicing<Index, GpuDevice>  {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:  EIGEN_DEVICE_FUNC MemcpyTriggerForSlicing(const GpuDevice&) { }
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:// For CUDA packet types when using a GpuDevice
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__) && defined(EIGEN_HAS_CUDA_FP16)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:struct PacketType<half, GpuDevice> {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:// Can't use std::pairs on cuda devices
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorSycl.h:// this is used to change the address space type in tensor map for GPU
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorFFT.h:  * Improve the performance on GPU
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:struct GpuDevice;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:struct IsVectorizable<GpuDevice, Expression> {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:  static const bool value = TensorEvaluator<Expression, GpuDevice>::PacketAccess &&
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:                            TensorEvaluator<Expression, GpuDevice>::IsAligned;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, GpuDevice> :
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    public TensorContractionEvaluatorBase<TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, GpuDevice> > {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:  typedef GpuDevice Device;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:  typedef typename PacketType<CoeffReturnType, GpuDevice>::type PacketReturnType;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    static void Run(const LhsMapper& lhs, const RhsMapper& rhs, const OutputMapper& output, Index m, Index n, Index k, const GpuDevice& device) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    LAUNCH_CUDA_KERNEL((EigenContractionKernel<Scalar, Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    static void Run(const LhsMapper& lhs, const RhsMapper& rhs, const OutputMapper& output, Index m, Index n, Index k, const GpuDevice& device) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:        LAUNCH_CUDA_KERNEL((EigenFloatContractionKernel16x16<Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:        LAUNCH_CUDA_KERNEL((EigenFloatContractionKernel<Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    setCudaSharedMemConfig(cudaSharedMemBankSizeEightByte);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#endif // EIGEN_USE_GPU and __CUDACC__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorMacros.h:#ifndef __CUDACC__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionSycl.h:    /// This one is used to collect all the reduced value of shared memory as we dont have global barrier on GPU. Once it is saved we can
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReductionSycl.h:    /// This one is used to collect all the reduced value of shared memory as we dont have global barrier on GPU. Once it is saved we can
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#ifdef EIGEN_HAS_CUDA_FP16
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:        ((RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) ||
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:         !RunningOnGPU))) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    else if (RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    if ((RunningOnSycl || RunningFullReduction || RunningOnGPU) && m_result) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    if (RunningOnGPU && m_result) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#ifdef EIGEN_HAS_CUDA_FP16
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:  static const bool RunningOnGPU = internal::is_same<Device, Eigen::GpuDevice>::value;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:static const bool RunningOnGPU = false;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:  static const bool RunningOnGPU = false;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#if defined(EIGEN_USE_GPU) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static const int kCudaScratchSize = 1024;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:// This defines an interface that GPUDevice can take to use
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:// CUDA streams underneath.
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  virtual const cudaStream_t& stream() const = 0;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  virtual const cudaDeviceProp& deviceProperties() const = 0;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static cudaDeviceProp* m_deviceProperties;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaError_t status = cudaGetDeviceCount(&num_devices);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      if (status != cudaSuccess) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        std::cerr << "Failed to get the number of CUDA devices: "
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:                  << cudaGetErrorString(status)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        assert(status == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      m_deviceProperties = new cudaDeviceProp[num_devices];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        status = cudaGetDeviceProperties(&m_deviceProperties[i], i);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        if (status != cudaSuccess) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:          std::cerr << "Failed to initialize CUDA device #"
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:                    << cudaGetErrorString(status)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:          assert(status == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static const cudaStream_t default_stream = cudaStreamDefault;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:class CudaStreamDevice : public StreamInterface {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  CudaStreamDevice() : stream_(&default_stream), scratch_(NULL), semaphore_(NULL) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaGetDevice(&device_);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  CudaStreamDevice(int device) : stream_(&default_stream), device_(device), scratch_(NULL), semaphore_(NULL) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  // assumes that the stream is associated to the current gpu device.
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  CudaStreamDevice(const cudaStream_t* stream, int device = -1)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaGetDevice(&device_);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaError_t err = cudaGetDeviceCount(&num_devices);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      assert(err == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  virtual ~CudaStreamDevice() {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  const cudaStream_t& stream() const { return *stream_; }
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  const cudaDeviceProp& deviceProperties() const {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaSetDevice(device_);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    err = cudaMalloc(&result, num_bytes);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaSetDevice(device_);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    err = cudaFree(buffer);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      scratch_ = allocate(kCudaScratchSize + sizeof(unsigned int));
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      char* scratch = static_cast<char*>(scratchpad()) + kCudaScratchSize;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaError_t err = cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      assert(err == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  const cudaStream_t* stream_;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:struct GpuDevice {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  explicit GpuDevice(const StreamInterface* stream) : stream_(stream), max_blocks_(INT_MAX) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  explicit GpuDevice(const StreamInterface* stream, int num_blocks) : stream_(stream), max_blocks_(num_blocks) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE const cudaStream_t& stream() const {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifndef __CUDA_ARCH__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice,
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err =
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream_->stream());
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err =
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream_->stream());
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifndef __CUDA_ARCH__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaMemsetAsync(buffer, c, n, stream_->stream());
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    // there is no l3 cache on cuda devices.
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaStreamSynchronize(stream_->stream());
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    if (err != cudaSuccess) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      std::cerr << "Error detected in CUDA stream: "
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:                << cudaGetErrorString(err)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      assert(err == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE int getNumCudaMultiProcessors() const {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE int maxCudaThreadsPerBlock() const {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE int maxCudaThreadsPerMultiProcessor() const {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  // This function checks if the CUDA runtime recorded an error for the
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifdef __CUDACC__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t error = cudaStreamQuery(stream_->stream());
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    return (error == cudaSuccess) || (error == cudaErrorNotReady);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#define LAUNCH_CUDA_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)             \
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  assert(cudaGetLastError() == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifdef __CUDACC__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static EIGEN_DEVICE_FUNC inline void setCudaSharedMemConfig(cudaSharedMemConfig config) {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifndef __CUDA_ARCH__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  cudaError_t status = cudaDeviceSetSharedMemConfig(config);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  assert(status == cudaSuccess);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    array<Index, NumDims> cudaInputDimensions;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    array<Index, NumDims> cudaOutputDimensions;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      cudaInputDimensions[index] = input_dims[indices[i]];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      cudaOutputDimensions[index] = dimensions[indices[i]];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        cudaInputDimensions[written] = input_dims[i];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        cudaOutputDimensions[written] = dimensions[i];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] =
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaInputStrides[i - 1] * cudaInputDimensions[i - 1];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] =
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaOutputStrides[i - 1] * cudaOutputDimensions[i - 1];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] = 1;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] = 1;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] =
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaInputStrides[i + 1] * cudaInputDimensions[i + 1];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] =
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaOutputStrides[i + 1] * cudaOutputDimensions[i + 1];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] = 1;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] = 1;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputPlaneToTensorInputOffset(Index p) const {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaInputStrides[d];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaInputStrides[d];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaInputStrides[d];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaInputStrides[d];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputPlaneToTensorOutputOffset(Index p) const {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaOutputStrides[d];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaOutputStrides[d];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaOutputStrides[d];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaOutputStrides[d];
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputKernelToTensorInputOffset(Index i) const {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputKernelToTensorOutputOffset(Index i) const {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputKernelToTensorInputOffset(Index i, Index j) const {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputKernelToTensorOutputOffset(Index i, Index j) const {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputKernelToTensorInputOffset(Index i, Index j, Index k) const {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputKernelToTensorOutputOffset(Index i, Index j, Index k) const {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  array<Index, NumDims> m_cudaInputStrides;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  array<Index, NumDims> m_cudaOutputStrides;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:// Use an optimized implementation of the evaluation code for GPUs whenever possible.
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const int tensor_index = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i+first_x);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const int tensor_index = plane_output_offset + indexMapper.mapCudaOutputKernelToTensorOutputOffset(i+first_x);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const int tensor_index = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i+first_x, j+first_y);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const int tensor_index = plane_output_offset + indexMapper.mapCudaOutputKernelToTensorOutputOffset(i+first_x, j+first_y);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          const int tensor_index = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i+first_x, j+first_y, k+first_z);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          const int tensor_index = plane_output_offset + indexMapper.mapCudaOutputKernelToTensorOutputOffset(i+first_x, j+first_y, k+first_z);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:struct TensorEvaluator<const TensorConvolutionOp<Indices, InputArgType, KernelArgType>, GpuDevice>
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  static const int NumDims =  internal::array_size<typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions>::value;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  typedef typename TensorEvaluator<KernelArgType, GpuDevice>::Dimensions KernelDimensions;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    IsAligned = TensorEvaluator<InputArgType, GpuDevice>::IsAligned & TensorEvaluator<KernelArgType, GpuDevice>::IsAligned,
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    Layout = TensorEvaluator<InputArgType, GpuDevice>::Layout,
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const GpuDevice& device)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<InputArgType, GpuDevice>::Layout) == static_cast<int>(TensorEvaluator<KernelArgType, GpuDevice>::Layout)), YOU_MADE_A_PROGRAMMING_MISTAKE);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions& input_dims = m_inputImpl.dimensions();
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const typename TensorEvaluator<KernelArgType, GpuDevice>::Dimensions& kernel_dims = m_kernelImpl.dimensions();
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  typedef typename PacketType<CoeffReturnType, GpuDevice>::type PacketReturnType;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const bool PacketAccess = internal::IsVectorizable<GpuDevice, KernelArgType>::value;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      internal::TensorExecutor<const EvalTo, GpuDevice, PacketAccess>::run(evalToTmp, m_device);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    typedef typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions InputDims;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int maxThreadsPerBlock = m_device.maxCudaThreadsPerBlock();
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int maxBlocksPerProcessor = m_device.maxCudaThreadsPerMultiProcessor() / maxThreadsPerBlock;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int numMultiProcessors = m_device.getNumCudaMultiProcessors();
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, 4, data);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, 7, data);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, kernel_size, data);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4, 7>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 4, 7, data);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 4, kernel_size_y, data);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7, 4>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 7, 4, data);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 7, kernel_size_y, data);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, Dynamic, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, kernel_size_x, kernel_size_y, data);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        LAUNCH_CUDA_KERNEL((EigenConvolutionKernel3D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, numZ, maxZ, kernel_size_x, kernel_size_y, kernel_size_z, data);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  TensorEvaluator<InputArgType, GpuDevice> m_inputImpl;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  TensorEvaluator<KernelArgType, GpuDevice> m_kernelImpl;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  const GpuDevice& m_device;
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:// GPU implementation of scan
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:struct ScanLauncher<Self, Reducer, GpuDevice> {
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:     LAUNCH_CUDA_KERNEL((ScanKernel<Self, Reducer>), num_blocks, block_size, 0, self.device(), self, total_size, data);
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:#endif  // EIGEN_USE_GPU && __CUDACC__
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:// Use the texture cache on CUDA devices whenever possible
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/util/EmulateArray.h:// Moreover, CUDA doesn't support the STL containers, so we use our own instead.
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/util/EmulateArray.h:#if (__cplusplus <= 199711L && EIGEN_COMP_MSVC < 1900) || defined(__CUDACC__) || defined(EIGEN_AVOID_STL_ARRAY)
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/util/EmulateArray.h:// The compiler supports c++11, and we're not targetting cuda: use std::array as Eigen::array
src/utils/overlap/eigen/unsupported/Eigen/CXX11/src/util/MaxSizeVector.h:  * std::vector is not an option (e.g. on GPU or when compiling using
src/utils/overlap/eigen/unsupported/Eigen/SpecialFunctions:#if defined EIGEN_VECTORIZE_CUDA
src/utils/overlap/eigen/unsupported/Eigen/SpecialFunctions:  #include "src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h"
src/utils/overlap/eigen/unsupported/Eigen/OpenGLSupport:#ifdef GL_ARB_gpu_shader_fp64
src/utils/overlap/eigen/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:#if !defined(__CUDA_ARCH__) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__)
src/utils/overlap/eigen/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:#if !defined(__CUDA_ARCH__) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__)
src/utils/overlap/eigen/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:    if (x == inf) return zero;  // std::isinf crashes on CUDA
src/utils/overlap/eigen/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#ifndef EIGEN_CUDA_SPECIALFUNCTIONS_H
src/utils/overlap/eigen/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#define EIGEN_CUDA_SPECIALFUNCTIONS_H
src/utils/overlap/eigen/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
src/utils/overlap/eigen/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
src/utils/overlap/eigen/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#endif // EIGEN_CUDA_SPECIALFUNCTIONS_H
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:#define EIGEN_USE_GPU
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:#include <cuda.h>
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:#include <cuda_runtime.h>
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:#define BM_FuncGPU(FUNC)                                                       \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, float> suite(device, N);                  \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    cudaDeviceSynchronize();                                                   \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(memcpy);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(typeCasting);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(random);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(slicing);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(rowChip);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(colChip);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(shuffling);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(padding);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(striding);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(broadcasting);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(coeffWiseOp);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(algebraicFunc);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(transcendentalFunc);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(rowReduction);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(colReduction);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncGPU(fullReduction);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:#define BM_FuncWithInputDimsGPU(FUNC, D1, D2, D3)                              \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, float> suite(device, D1, D2, D3);         \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    cudaDeviceSynchronize();                                                   \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, N, N);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithInputDimsGPU(contraction, 64, N, N);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, 64, N);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, N, 64);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:#define BM_FuncWithKernelDimsGPU(FUNC, DIM1, DIM2)                             \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, float> suite(device, N);                  \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:    cudaDeviceSynchronize();                                                   \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 1);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 1, 7);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 4);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 4, 7);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 64);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 64, 7);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_sycl.cc:#define BM_FuncGPU(FUNC)                                       \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_sycl.cc:    cl::sycl::queue q = sycl_queue<cl::sycl::gpu_selector>();  \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(broadcasting);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_sycl.cc:BM_FuncGPU(coeffWiseOp);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#define EIGEN_USE_GPU
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#include <cuda.h>
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#include <cuda_runtime.h>
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#define BM_FuncGPU(FUNC)                                                       \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, Eigen::half> suite(device, N);            \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    cudaDeviceSynchronize();                                                   \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(memcpy);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(typeCasting);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu://BM_FuncGPU(random);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(slicing);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(rowChip);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(colChip);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(shuffling);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(padding);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(striding);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(broadcasting);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(coeffWiseOp);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(algebraicFunc);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(transcendentalFunc);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(rowReduction);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(colReduction);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncGPU(fullReduction);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#define BM_FuncWithInputDimsGPU(FUNC, D1, D2, D3)                              \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, Eigen::half> suite(device, D1, D2, D3);   \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    cudaDeviceSynchronize();                                                   \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, N, N);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithInputDimsGPU(contraction, 64, N, N);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, 64, N);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithInputDimsGPU(contraction, N, N, 64);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:#define BM_FuncWithKernelDimsGPU(FUNC, DIM1, DIM2)                             \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::CudaStreamDevice stream;                                            \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    Eigen::GpuDevice device(&stream);                                          \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    BenchmarkSuite<Eigen::GpuDevice, Eigen::half> suite(device, N);            \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:    cudaDeviceSynchronize();                                                   \
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 1);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 1, 7);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 4);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 4, 7);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 7, 64);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks_fp16_gpu.cu:BM_FuncWithKernelDimsGPU(convolution, 64, 7);
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
src/utils/overlap/eigen/bench/tensors/tensor_benchmarks.h:    if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
src/utils/overlap/eigen/bench/tensors/README:The first part is a generic suite, in which each benchmark comes in 2 flavors: one that runs on CPU, and one that runs on GPU.
src/utils/overlap/eigen/bench/tensors/README:To compile the floating point GPU benchmarks, simply call:
src/utils/overlap/eigen/bench/tensors/README:nvcc tensor_benchmarks_gpu.cu benchmark_main.cc -I ../../ -std=c++11 -O2 -DNDEBUG -use_fast_math -ftz=true -arch compute_35 -o benchmarks_gpu
src/utils/overlap/eigen/bench/tensors/README:We also provide a version of the generic GPU tensor benchmarks that uses half floats (aka fp16) instead of regular floats. To compile these benchmarks, simply call the command line below. You'll need a recent GPU that supports compute capability 5.3 or higher to run them and nvcc 7.5 or higher to compile the code.
src/utils/overlap/eigen/bench/tensors/README:nvcc tensor_benchmarks_fp16_gpu.cu benchmark_main.cc -I ../../ -std=c++11 -O2 -DNDEBUG -use_fast_math -ftz=true -arch compute_53 -o benchmarks_fp16_gpu
src/utils/overlap/eigen/bench/tensors/README:clang++-3.7 -include tensor_benchmarks_sycl.sycl benchmark_main.cc tensor_benchmarks_sycl.cc -pthread -I ../../ -I {ComputeCpp_ROOT}/include/ -L {ComputeCpp_ROOT}/lib/ -lComputeCpp -lOpenCL -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -o tensor_benchmark_sycl
src/utils/overlap/eigen/cmake/FindPastix.cmake:#   - STARPU_CUDA: to activate detection of StarPU with CUDA
src/utils/overlap/eigen/cmake/FindPastix.cmake:set(PASTIX_LOOK_FOR_STARPU_CUDA OFF)
src/utils/overlap/eigen/cmake/FindPastix.cmake:    if (${component} STREQUAL "STARPU_CUDA")
src/utils/overlap/eigen/cmake/FindPastix.cmake:      # means we look for PaStiX with StarPU + CUDA
src/utils/overlap/eigen/cmake/FindPastix.cmake:      set(PASTIX_LOOK_FOR_STARPU_CUDA ON)
src/utils/overlap/eigen/cmake/FindPastix.cmake:  if (PASTIX_LOOK_FOR_STARPU_CUDA)
src/utils/overlap/eigen/cmake/FindPastix.cmake:    list(APPEND STARPU_COMPONENT_LIST "CUDA")
src/utils/overlap/eigen/cmake/FindPastix.cmake:  # CUDA
src/utils/overlap/eigen/cmake/FindPastix.cmake:  if (PASTIX_LOOK_FOR_STARPU_CUDA AND CUDA_FOUND)
src/utils/overlap/eigen/cmake/FindPastix.cmake:    if (CUDA_INCLUDE_DIRS)
src/utils/overlap/eigen/cmake/FindPastix.cmake:      list(APPEND REQUIRED_INCDIRS "${CUDA_INCLUDE_DIRS}")
src/utils/overlap/eigen/cmake/FindPastix.cmake:    foreach(libdir ${CUDA_LIBRARY_DIRS})
src/utils/overlap/eigen/cmake/FindPastix.cmake:    list(APPEND REQUIRED_LIBS "${CUDA_CUBLAS_LIBRARIES};${CUDA_LIBRARIES}")
src/utils/overlap/eigen/cmake/FindPastix.cmake:	"Have you tried with COMPONENTS (MPI/SEQ, STARPU, STARPU_CUDA, SCOTCH, PTSCOTCH, METIS)? "
src/utils/overlap/eigen/cmake/EigenTesting.cmake:    if(EIGEN_TEST_CUDA_CLANG)
src/utils/overlap/eigen/cmake/EigenTesting.cmake:      if(CUDA_64_BIT_DEVICE_CODE)
src/utils/overlap/eigen/cmake/EigenTesting.cmake:        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
src/utils/overlap/eigen/cmake/EigenTesting.cmake:        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib")
src/utils/overlap/eigen/cmake/EigenTesting.cmake:      target_link_libraries(${targetname} "cudart_static" "cuda" "dl" "rt" "pthread")
src/utils/overlap/eigen/cmake/EigenTesting.cmake:        cuda_add_executable(${targetname} ${filename} OPTIONS ${ARGV2})
src/utils/overlap/eigen/cmake/EigenTesting.cmake:        cuda_add_executable(${targetname} ${filename})
src/utils/overlap/eigen/cmake/EigenTesting.cmake:    if(EIGEN_TEST_CUDA)
src/utils/overlap/eigen/cmake/EigenTesting.cmake:      if(EIGEN_TEST_CUDA_CLANG)
src/utils/overlap/eigen/cmake/EigenTesting.cmake:        message(STATUS "CUDA:              ON (using clang)")
src/utils/overlap/eigen/cmake/EigenTesting.cmake:        message(STATUS "CUDA:              ON (using nvcc)")
src/utils/overlap/eigen/cmake/EigenTesting.cmake:      message(STATUS "CUDA:              OFF")
src/utils/overlap/eigen/cmake/FindBLAS.cmake:##  ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic
src/utils/overlap/eigen/cmake/FindBLAS.cmake:      ((BLA_VENDOR STREQUAL "ACML_GPU") AND (NOT BLAS_ACML_GPU_LIB_DIRS)))
src/utils/overlap/eigen/cmake/FindBLAS.cmake:      file( GLOB _ACML_GPU_ROOT "C:/AMD/acml*/GPGPUexamples" )
src/utils/overlap/eigen/cmake/FindBLAS.cmake:      file( GLOB _ACML_GPU_ROOT "/opt/acml*/GPGPUexamples" )
src/utils/overlap/eigen/cmake/FindBLAS.cmake:    list(GET _ACML_GPU_ROOT 0 _ACML_GPU_ROOT)
src/utils/overlap/eigen/cmake/FindBLAS.cmake:  elseif( BLA_VENDOR STREQUAL "ACML_GPU" )
src/utils/overlap/eigen/cmake/FindBLAS.cmake:    foreach( BLAS_ACML_GPU_LIB_DIRS ${_ACML_GPU_LIB_DIRS})
src/utils/overlap/eigen/cmake/FindBLAS.cmake:	"" "acml;acml_mv;CALBLAS" "" ${BLAS_ACML_GPU_LIB_DIRS}
src/utils/overlap/eigen/cmake/FindBLAS.cmake:	"Intel( older versions of mkl 32 and 64 bit), ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
src/utils/overlap/eigen/cmake/FindBLAS.cmake:	"Intel( older versions of mkl 32 and 64 bit), ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
src/utils/overlap/eigen/cmake/FindBLASEXT.cmake:    ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
src/utils/overlap/eigen/cmake/FindBLASEXT.cmake:    "\n   ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
src/utils/overlap/eigen/cmake/FindComputeCpp.cmake:# Find OpenCL package
src/utils/overlap/eigen/cmake/FindComputeCpp.cmake:find_package(OpenCL REQUIRED)
src/utils/overlap/eigen/cmake/FindComputeCpp.cmake:                        PUBLIC ${OpenCL_LIBRARIES})
src/utils/overlap/eigen/demos/opengl/gpuhelper.cpp:#include "gpuhelper.h"
src/utils/overlap/eigen/demos/opengl/gpuhelper.cpp:GpuHelper gpu;
src/utils/overlap/eigen/demos/opengl/gpuhelper.cpp:GpuHelper::GpuHelper()
src/utils/overlap/eigen/demos/opengl/gpuhelper.cpp:GpuHelper::~GpuHelper()
src/utils/overlap/eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::pushProjectionMode2D(ProjectionMode2D pm)
src/utils/overlap/eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::popProjectionMode2D(void)
src/utils/overlap/eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::drawVector(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect /* = 50.*/)
src/utils/overlap/eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::drawVectorBox(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect)
src/utils/overlap/eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::drawUnitCube(void)
src/utils/overlap/eigen/demos/opengl/gpuhelper.cpp:void GpuHelper::drawUnitSphere(int level)
src/utils/overlap/eigen/demos/opengl/quaternion_demo.h:#include "gpuhelper.h"
src/utils/overlap/eigen/demos/opengl/CMakeLists.txt:  set(quaternion_demo_SRCS  gpuhelper.cpp icosphere.cpp camera.cpp trackball.cpp quaternion_demo.cpp)
src/utils/overlap/eigen/demos/opengl/camera.cpp:#include "gpuhelper.h"
src/utils/overlap/eigen/demos/opengl/camera.cpp:  gpu.loadMatrix(projectionMatrix(),GL_PROJECTION);
src/utils/overlap/eigen/demos/opengl/camera.cpp:  gpu.loadMatrix(viewMatrix().matrix(),GL_MODELVIEW);
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:#ifndef EIGEN_GPUHELPER_H
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:#define EIGEN_GPUHELPER_H
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:class GpuHelper
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:    GpuHelper();
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:    ~GpuHelper();
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:extern GpuHelper gpu;
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::setMatrixTarget(GLenum matrixTarget)
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:void GpuHelper::multMatrix(const Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:void GpuHelper::loadMatrix(
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:void GpuHelper::loadMatrix(const Eigen::Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::pushMatrix(GLenum matrixTarget)
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::pushMatrix(const Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:void GpuHelper::pushMatrix(
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::popMatrix(GLenum matrixTarget)
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, uint nofElement)
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, const std::vector<uint>* pIndexes)
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, uint start, uint end)
src/utils/overlap/eigen/demos/opengl/gpuhelper.h:#endif // EIGEN_GPUHELPER_H
src/utils/overlap/eigen/demos/opengl/quaternion_demo.cpp:        gpu.pushMatrix(GL_MODELVIEW);
src/utils/overlap/eigen/demos/opengl/quaternion_demo.cpp:        gpu.multMatrix(t.matrix(),GL_MODELVIEW);
src/utils/overlap/eigen/demos/opengl/quaternion_demo.cpp:        gpu.popMatrix(GL_MODELVIEW);
src/utils/overlap/eigen/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitX(), Color(1,0,0,1));
src/utils/overlap/eigen/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitY(), Color(0,1,0,1));
src/utils/overlap/eigen/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitZ(), Color(0,0,1,1));

```
