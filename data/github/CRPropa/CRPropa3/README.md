# https://github.com/CRPropa/CRPropa3

```console
libs/gtest/include/gtest/internal/gtest-port.h:// with a TR1 tuple implementation.  NVIDIA's CUDA NVCC compiler
libs/gtest/include/gtest/internal/gtest-port.h:# if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000) \
libs/eigen3/Eigen/Core:// Handle NVCC/CUDA/SYCL
libs/eigen3/Eigen/Core:#if defined(__CUDACC__) || defined(__SYCL_DEVICE_ONLY__)
libs/eigen3/Eigen/Core:  // Do not try asserts on CUDA and SYCL!
libs/eigen3/Eigen/Core:  // All functions callable from CUDA code must be qualified with __device__
libs/eigen3/Eigen/Core:  #ifdef __CUDACC__
libs/eigen3/Eigen/Core:    // Do not try to vectorize on CUDA and SYCL!
libs/eigen3/Eigen/Core:// When compiling CUDA device code with NVCC, pull in math functions from the
libs/eigen3/Eigen/Core:#if defined(__CUDA_ARCH__) && defined(__NVCC__)
libs/eigen3/Eigen/Core:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL)
libs/eigen3/Eigen/Core:#if defined __CUDACC__
libs/eigen3/Eigen/Core:  #define EIGEN_VECTORIZE_CUDA
libs/eigen3/Eigen/Core:  #if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
libs/eigen3/Eigen/Core:    #define EIGEN_HAS_CUDA_FP16
libs/eigen3/Eigen/Core:#if defined EIGEN_HAS_CUDA_FP16
libs/eigen3/Eigen/Core:  #include <cuda_fp16.h>
libs/eigen3/Eigen/Core:#include "src/Core/arch/CUDA/Half.h"
libs/eigen3/Eigen/Core:#include "src/Core/arch/CUDA/PacketMathHalf.h"
libs/eigen3/Eigen/Core:#include "src/Core/arch/CUDA/TypeCasting.h"
libs/eigen3/Eigen/Core:#if defined EIGEN_VECTORIZE_CUDA
libs/eigen3/Eigen/Core:  #include "src/Core/arch/CUDA/PacketMath.h"
libs/eigen3/Eigen/Core:  #include "src/Core/arch/CUDA/MathFunctions.h"
libs/eigen3/Eigen/Core:// on CUDA devices
libs/eigen3/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
libs/eigen3/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
libs/eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
libs/eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
libs/eigen3/Eigen/src/Core/util/Meta.h:  static float (max)() { return CUDART_MAX_NORMAL_F; }
libs/eigen3/Eigen/src/Core/util/Meta.h:  static float infinity() { return CUDART_INF_F; }
libs/eigen3/Eigen/src/Core/util/Meta.h:  static float quiet_NaN() { return CUDART_NAN_F; }
libs/eigen3/Eigen/src/Core/util/Meta.h:  static double infinity() { return CUDART_INF; }
libs/eigen3/Eigen/src/Core/util/Meta.h:  static double quiet_NaN() { return CUDART_NAN; }
libs/eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
libs/eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
libs/eigen3/Eigen/src/Core/util/Macros.h:  && ( !defined(__NVCC__) || !EIGEN_ARCH_ARM_OR_ARM64 || (defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000) )
libs/eigen3/Eigen/src/Core/util/Macros.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/util/Macros.h:#if EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && defined(__CUDACC_VER__) && (EIGEN_COMP_CLANG || __CUDACC_VER__ >= 70500))
libs/eigen3/Eigen/src/Core/util/Macros.h:#if (defined __CUDACC__)
libs/eigen3/Eigen/src/Core/util/Macros.h:#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC < 1900 ||  defined(__CUDACC_VER__)) // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
libs/eigen3/Eigen/src/Core/util/Macros.h:#  ifdef __CUDA_ARCH__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
libs/eigen3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
libs/eigen3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
libs/eigen3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifndef __CUDA_ARCH__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#if (!defined(__CUDACC__)) && EIGEN_FAST_MATH
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
libs/eigen3/Eigen/src/Core/MatrixBase.h:#ifdef __CUDACC__
libs/eigen3/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
libs/eigen3/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
libs/eigen3/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
libs/eigen3/Eigen/src/Core/GeneralProduct.h:#ifndef __CUDACC__
libs/eigen3/Eigen/src/Core/GeneralProduct.h:#endif // __CUDACC__
libs/eigen3/Eigen/src/Core/GenericPacketMath.h:#ifdef __CUDA_ARCH__
libs/eigen3/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
libs/eigen3/Eigen/src/Core/GenericPacketMath.h:#ifndef __CUDACC__
libs/eigen3/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
libs/eigen3/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
libs/eigen3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
libs/eigen3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
libs/eigen3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#ifndef EIGEN_PACKET_MATH_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#define EIGEN_PACKET_MATH_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#endif // EIGEN_PACKET_MATH_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// type Eigen::half (inheriting from CUDA's __half struct) with
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// to disk and the likes), but fast on GPUs.
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#ifndef EIGEN_HALF_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#define EIGEN_HALF_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if !defined(EIGEN_HAS_CUDA_FP16)
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Make our own __half definition that is similar to CUDA's.
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:  #if !defined(EIGEN_HAS_CUDA_FP16)
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Definitions for CPUs and older CUDA, mostly working through conversion
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__)
libs/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#endif // EIGEN_HALF_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#ifndef EIGEN_PACKET_MATH_HALF_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#define EIGEN_PACKET_MATH_HALF_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 530
libs/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#endif // EIGEN_PACKET_MATH_HALF_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
libs/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
libs/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#define EIGEN_TYPE_CASTING_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
libs/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
libs/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
libs/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
libs/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
libs/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
libs/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:// building for CUDA to avoid non-constexpr methods.
libs/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#endif // EIGEN_COMPLEX_CUDA_H
doc/pages/Basic-concepts.md:MPI, and GPU computing. These require a specific program layout and are

```
