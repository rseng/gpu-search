# https://github.com/pierrexyz/cbird

```console
cnest/Eigen/Core:#if defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
cnest/Eigen/Core:  #define EIGEN_CUDACC __CUDACC__
cnest/Eigen/Core:#if defined(__CUDA_ARCH__) && !defined(EIGEN_NO_CUDA)
cnest/Eigen/Core:  #define EIGEN_CUDA_ARCH __CUDA_ARCH__
cnest/Eigen/Core:#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
cnest/Eigen/Core:#define EIGEN_CUDACC_VER  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
cnest/Eigen/Core:#elif defined(__CUDACC_VER__)
cnest/Eigen/Core:#define EIGEN_CUDACC_VER __CUDACC_VER__
cnest/Eigen/Core:#define EIGEN_CUDACC_VER 0
cnest/Eigen/Core:// Handle NVCC/CUDA/SYCL
cnest/Eigen/Core:#if defined(__CUDACC__) || defined(__SYCL_DEVICE_ONLY__)
cnest/Eigen/Core:  // Do not try asserts on CUDA and SYCL!
cnest/Eigen/Core:  // All functions callable from CUDA code must be qualified with __device__
cnest/Eigen/Core:  #ifdef __CUDACC__
cnest/Eigen/Core:    // Do not try to vectorize on CUDA and SYCL!
cnest/Eigen/Core:// When compiling CUDA device code with NVCC, pull in math functions from the
cnest/Eigen/Core:#if defined(__CUDA_ARCH__) && defined(__NVCC__)
cnest/Eigen/Core:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL)
cnest/Eigen/Core:#if defined __CUDACC__
cnest/Eigen/Core:  #define EIGEN_VECTORIZE_CUDA
cnest/Eigen/Core:  #if EIGEN_CUDACC_VER >= 70500
cnest/Eigen/Core:    #define EIGEN_HAS_CUDA_FP16
cnest/Eigen/Core:#if defined EIGEN_HAS_CUDA_FP16
cnest/Eigen/Core:  #include <cuda_fp16.h>
cnest/Eigen/Core:#include "src/Core/arch/CUDA/Half.h"
cnest/Eigen/Core:#include "src/Core/arch/CUDA/PacketMathHalf.h"
cnest/Eigen/Core:#include "src/Core/arch/CUDA/TypeCasting.h"
cnest/Eigen/Core:#if defined EIGEN_VECTORIZE_CUDA
cnest/Eigen/Core:  #include "src/Core/arch/CUDA/PacketMath.h"
cnest/Eigen/Core:  #include "src/Core/arch/CUDA/MathFunctions.h"
cnest/Eigen/Core:// on CUDA devices
cnest/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
cnest/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
cnest/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
cnest/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
cnest/Eigen/src/Core/util/Meta.h:  static float (max)() { return CUDART_MAX_NORMAL_F; }
cnest/Eigen/src/Core/util/Meta.h:  static float infinity() { return CUDART_INF_F; }
cnest/Eigen/src/Core/util/Meta.h:  static float quiet_NaN() { return CUDART_NAN_F; }
cnest/Eigen/src/Core/util/Meta.h:  static double infinity() { return CUDART_INF; }
cnest/Eigen/src/Core/util/Meta.h:  static double quiet_NaN() { return CUDART_NAN; }
cnest/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
cnest/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
cnest/Eigen/src/Core/util/Macros.h:  && (!defined(__NVCC__) || !EIGEN_ARCH_ARM_OR_ARM64 || (EIGEN_CUDACC_VER >= 80000) )
cnest/Eigen/src/Core/util/Macros.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/util/Macros.h:#if EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && (EIGEN_COMP_CLANG || EIGEN_CUDACC_VER >= 70500))
cnest/Eigen/src/Core/util/Macros.h:#if (defined __CUDACC__)
cnest/Eigen/src/Core/util/Macros.h:#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC < 1900 || EIGEN_CUDACC_VER>0)
cnest/Eigen/src/Core/util/Macros.h:  // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
cnest/Eigen/src/Core/util/Macros.h:#  ifdef __CUDA_ARCH__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
cnest/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
cnest/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
cnest/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
cnest/Eigen/src/Core/MathFunctions.h:#ifndef __CUDA_ARCH__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#if (!defined(__CUDACC__)) && EIGEN_FAST_MATH
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef __CUDACC__
cnest/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
cnest/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
cnest/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
cnest/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
cnest/Eigen/src/Core/GeneralProduct.h:    #ifndef EIGEN_CUDA_ARCH
cnest/Eigen/src/Core/GenericPacketMath.h:#ifdef __CUDA_ARCH__
cnest/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
cnest/Eigen/src/Core/GenericPacketMath.h:#ifndef __CUDACC__
cnest/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
cnest/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
cnest/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
cnest/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
cnest/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
cnest/Eigen/src/Core/arch/CUDA/PacketMath.h:#ifndef EIGEN_PACKET_MATH_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/PacketMath.h:#define EIGEN_PACKET_MATH_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
cnest/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
cnest/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
cnest/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
cnest/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
cnest/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
cnest/Eigen/src/Core/arch/CUDA/PacketMath.h:#endif // EIGEN_PACKET_MATH_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
cnest/Eigen/src/Core/arch/CUDA/Half.h:// type Eigen::half (inheriting from CUDA's __half struct) with
cnest/Eigen/src/Core/arch/CUDA/Half.h:// to disk and the likes), but fast on GPUs.
cnest/Eigen/src/Core/arch/CUDA/Half.h:#ifndef EIGEN_HALF_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/Half.h:#define EIGEN_HALF_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if !defined(EIGEN_HAS_CUDA_FP16)
cnest/Eigen/src/Core/arch/CUDA/Half.h:// Make our own __half definition that is similar to CUDA's.
cnest/Eigen/src/Core/arch/CUDA/Half.h:  #if !defined(EIGEN_HAS_CUDA_FP16)
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/Half.h:// Definitions for CPUs and older CUDA, mostly working through conversion
cnest/Eigen/src/Core/arch/CUDA/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && EIGEN_CUDACC_VER >= 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
cnest/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__)
cnest/Eigen/src/Core/arch/CUDA/Half.h:#endif // EIGEN_HALF_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#ifndef EIGEN_PACKET_MATH_HALF_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#define EIGEN_PACKET_MATH_HALF_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530
cnest/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#endif // EIGEN_PACKET_MATH_HALF_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
cnest/Eigen/src/Core/arch/CUDA/MathFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
cnest/Eigen/src/Core/arch/CUDA/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/TypeCasting.h:#define EIGEN_TYPE_CASTING_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
cnest/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
cnest/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
cnest/Eigen/src/Core/arch/CUDA/TypeCasting.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
cnest/Eigen/src/Core/arch/CUDA/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
cnest/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
cnest/Eigen/src/Core/arch/CUDA/Complex.h:// building for CUDA to avoid non-constexpr methods.
cnest/Eigen/src/Core/arch/CUDA/Complex.h:#endif // EIGEN_COMPLEX_CUDA_H
cnest/Eigen/src/SVD/BDCSVD.h:#ifndef __CUDACC__

```