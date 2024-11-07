# https://github.com/bbuchfink/diamond

```console
src/lib/atomic_wait/atomic_wait:Copyright (c) 2019, NVIDIA Corporation
src/lib/atomic_wait/atomic_wait: * CUDA: default to timed backoff, fallback to spin. (This is not all checked in in this tree.)
src/lib/atomic_wait/atomic_wait:    #ifndef __CUDACC__
src/lib/atomic_wait/latch:Copyright (c) 2019, NVIDIA Corporation
src/lib/atomic_wait/barrier:Copyright (c) 2019, NVIDIA Corporation
src/lib/atomic_wait/semaphore:Copyright (c) 2019, NVIDIA Corporation
src/lib/Eigen/Core:#if defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
src/lib/Eigen/Core:  #define EIGEN_CUDACC __CUDACC__
src/lib/Eigen/Core:#if defined(__CUDA_ARCH__) && !defined(EIGEN_NO_CUDA)
src/lib/Eigen/Core:  #define EIGEN_CUDA_ARCH __CUDA_ARCH__
src/lib/Eigen/Core:#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
src/lib/Eigen/Core:#define EIGEN_CUDACC_VER  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
src/lib/Eigen/Core:#elif defined(__CUDACC_VER__)
src/lib/Eigen/Core:#define EIGEN_CUDACC_VER __CUDACC_VER__
src/lib/Eigen/Core:#define EIGEN_CUDACC_VER 0
src/lib/Eigen/Core:// Handle NVCC/CUDA/SYCL
src/lib/Eigen/Core:#if defined(__CUDACC__) || defined(__SYCL_DEVICE_ONLY__)
src/lib/Eigen/Core:  // Do not try asserts on CUDA and SYCL!
src/lib/Eigen/Core:  // All functions callable from CUDA code must be qualified with __device__
src/lib/Eigen/Core:  #ifdef __CUDACC__
src/lib/Eigen/Core:    // Do not try to vectorize on CUDA and SYCL!
src/lib/Eigen/Core:    // We need cuda_runtime.h to ensure that that EIGEN_USING_STD_MATH macro
src/lib/Eigen/Core:    #include <cuda_runtime.h>
src/lib/Eigen/Core:// When compiling CUDA device code with NVCC, pull in math functions from the
src/lib/Eigen/Core:#if defined(__CUDA_ARCH__) && defined(__NVCC__)
src/lib/Eigen/Core:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL)
src/lib/Eigen/Core:#if defined __CUDACC__
src/lib/Eigen/Core:  #define EIGEN_VECTORIZE_CUDA
src/lib/Eigen/Core:  #if EIGEN_CUDACC_VER >= 70500
src/lib/Eigen/Core:    #define EIGEN_HAS_CUDA_FP16
src/lib/Eigen/Core:#if defined EIGEN_HAS_CUDA_FP16
src/lib/Eigen/Core:  #include <cuda_fp16.h>
src/lib/Eigen/Core:#include "src/Core/arch/CUDA/Half.h"
src/lib/Eigen/Core:#include "src/Core/arch/CUDA/PacketMathHalf.h"
src/lib/Eigen/Core:#include "src/Core/arch/CUDA/TypeCasting.h"
src/lib/Eigen/Core:#if defined EIGEN_VECTORIZE_CUDA
src/lib/Eigen/Core:  #include "src/Core/arch/CUDA/PacketMath.h"
src/lib/Eigen/Core:  #include "src/Core/arch/CUDA/MathFunctions.h"
src/lib/Eigen/Core:// on CUDA devices
src/lib/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
src/lib/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
src/lib/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
src/lib/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
src/lib/Eigen/src/Core/util/Meta.h:  static float (max)() { return CUDART_MAX_NORMAL_F; }
src/lib/Eigen/src/Core/util/Meta.h:  static float infinity() { return CUDART_INF_F; }
src/lib/Eigen/src/Core/util/Meta.h:  static float quiet_NaN() { return CUDART_NAN_F; }
src/lib/Eigen/src/Core/util/Meta.h:  static double infinity() { return CUDART_INF; }
src/lib/Eigen/src/Core/util/Meta.h:  static double quiet_NaN() { return CUDART_NAN; }
src/lib/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
src/lib/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
src/lib/Eigen/src/Core/util/Macros.h:  && (!defined(__NVCC__) || !EIGEN_ARCH_ARM_OR_ARM64 || (EIGEN_CUDACC_VER >= 80000) )
src/lib/Eigen/src/Core/util/Macros.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/util/Macros.h:#if EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && (EIGEN_COMP_CLANG || EIGEN_CUDACC_VER >= 70500))
src/lib/Eigen/src/Core/util/Macros.h:#if (defined __CUDACC__)
src/lib/Eigen/src/Core/util/Macros.h:#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC < 1900 || EIGEN_CUDACC_VER>0)
src/lib/Eigen/src/Core/util/Macros.h:  // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
src/lib/Eigen/src/Core/util/Macros.h:#  ifdef __CUDA_ARCH__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
src/lib/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
src/lib/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
src/lib/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
src/lib/Eigen/src/Core/MathFunctions.h:#ifndef __CUDA_ARCH__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#if (!defined(__CUDACC__)) && EIGEN_FAST_MATH
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef __CUDACC__
src/lib/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
src/lib/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
src/lib/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
src/lib/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
src/lib/Eigen/src/Core/GeneralProduct.h:    #ifndef EIGEN_CUDA_ARCH
src/lib/Eigen/src/Core/GenericPacketMath.h:#ifdef __CUDA_ARCH__
src/lib/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
src/lib/Eigen/src/Core/GenericPacketMath.h:#ifndef __CUDACC__
src/lib/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
src/lib/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
src/lib/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
src/lib/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
src/lib/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
src/lib/Eigen/src/Core/arch/CUDA/PacketMath.h:#ifndef EIGEN_PACKET_MATH_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/PacketMath.h:#define EIGEN_PACKET_MATH_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
src/lib/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
src/lib/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
src/lib/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
src/lib/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
src/lib/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
src/lib/Eigen/src/Core/arch/CUDA/PacketMath.h:#endif // EIGEN_PACKET_MATH_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
src/lib/Eigen/src/Core/arch/CUDA/Half.h:// type Eigen::half (inheriting from CUDA's __half struct) with
src/lib/Eigen/src/Core/arch/CUDA/Half.h:// to disk and the likes), but fast on GPUs.
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#ifndef EIGEN_HALF_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#define EIGEN_HALF_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if !defined(EIGEN_HAS_CUDA_FP16)
src/lib/Eigen/src/Core/arch/CUDA/Half.h:// Make our own __half_raw definition that is similar to CUDA's.
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#elif defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER < 90000
src/lib/Eigen/src/Core/arch/CUDA/Half.h:// In CUDA < 9.0, __half is the equivalent of CUDA 9's __half_raw
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER >= 90000
src/lib/Eigen/src/Core/arch/CUDA/Half.h:  #if !defined(EIGEN_HAS_CUDA_FP16) || (defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER < 90000)
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER >= 90000
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
src/lib/Eigen/src/Core/arch/CUDA/Half.h:// Definitions for CPUs and older CUDA, mostly working through conversion
src/lib/Eigen/src/Core/arch/CUDA/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && EIGEN_CUDACC_VER >= 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300
src/lib/Eigen/src/Core/arch/CUDA/Half.h:  #if EIGEN_CUDACC_VER < 90000
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_CUDA_ARCH)
src/lib/Eigen/src/Core/arch/CUDA/Half.h:#endif // EIGEN_HALF_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#ifndef EIGEN_PACKET_MATH_HALF_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#define EIGEN_PACKET_MATH_HALF_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530
src/lib/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#endif // EIGEN_PACKET_MATH_HALF_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
src/lib/Eigen/src/Core/arch/CUDA/MathFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
src/lib/Eigen/src/Core/arch/CUDA/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/TypeCasting.h:#define EIGEN_TYPE_CASTING_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
src/lib/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
src/lib/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
src/lib/Eigen/src/Core/arch/CUDA/TypeCasting.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
src/lib/Eigen/src/Core/arch/CUDA/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
src/lib/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
src/lib/Eigen/src/Core/arch/CUDA/Complex.h:// building for CUDA to avoid non-constexpr methods.
src/lib/Eigen/src/Core/arch/CUDA/Complex.h:#endif // EIGEN_COMPLEX_CUDA_H
src/lib/Eigen/src/SVD/BDCSVD.h:#ifndef __CUDACC__

```
