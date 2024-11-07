# https://github.com/Griffan/VerifyBamID

```console
Eigen/Core:// Handle NVCC/CUDA/SYCL
Eigen/Core:#if defined(__CUDACC__) || defined(__SYCL_DEVICE_ONLY__)
Eigen/Core:  // Do not try asserts on CUDA and SYCL!
Eigen/Core:  // All functions callable from CUDA code must be qualified with __device__
Eigen/Core:  #ifdef __CUDACC__
Eigen/Core:    // Do not try to vectorize on CUDA and SYCL!
Eigen/Core:// When compiling CUDA device code with NVCC, pull in math functions from the
Eigen/Core:#if defined(__CUDA_ARCH__) && defined(__NVCC__)
Eigen/Core:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL)
Eigen/Core:#if defined __CUDACC__
Eigen/Core:  #define EIGEN_VECTORIZE_CUDA
Eigen/Core:  #if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
Eigen/Core:    #define EIGEN_HAS_CUDA_FP16
Eigen/Core:#if defined EIGEN_HAS_CUDA_FP16
Eigen/Core:  #include <cuda_fp16.h>
Eigen/Core:#include "src/Core/arch/CUDA/Half.h"
Eigen/Core:#include "src/Core/arch/CUDA/PacketMathHalf.h"
Eigen/Core:#include "src/Core/arch/CUDA/TypeCasting.h"
Eigen/Core:#if defined EIGEN_VECTORIZE_CUDA
Eigen/Core:  #include "src/Core/arch/CUDA/PacketMath.h"
Eigen/Core:  #include "src/Core/arch/CUDA/MathFunctions.h"
Eigen/Core:// on CUDA devices
Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
Eigen/src/Core/util/Meta.h:  static float (max)() { return CUDART_MAX_NORMAL_F; }
Eigen/src/Core/util/Meta.h:  static float infinity() { return CUDART_INF_F; }
Eigen/src/Core/util/Meta.h:  static float quiet_NaN() { return CUDART_NAN_F; }
Eigen/src/Core/util/Meta.h:  static double infinity() { return CUDART_INF; }
Eigen/src/Core/util/Meta.h:  static double quiet_NaN() { return CUDART_NAN; }
Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
Eigen/src/Core/util/Macros.h:  && ( !defined(__NVCC__) || !EIGEN_ARCH_ARM_OR_ARM64 || (defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000) )
Eigen/src/Core/util/Macros.h:#ifdef __CUDACC__
Eigen/src/Core/util/Macros.h:#if EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && defined(__CUDACC_VER__) && (EIGEN_COMP_CLANG || __CUDACC_VER__ >= 70500))
Eigen/src/Core/util/Macros.h:#if (defined __CUDACC__)
Eigen/src/Core/util/Macros.h:#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC < 1900 ||  __CUDACC_VER__) // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
Eigen/src/Core/util/Macros.h:#  ifdef __CUDA_ARCH__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
Eigen/src/Core/MathFunctions.h:#ifndef __CUDA_ARCH__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#if (!defined(__CUDACC__)) && EIGEN_FAST_MATH
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef __CUDACC__
Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
Eigen/src/Core/MatrixBase.h:#ifdef __CUDACC__
Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
Eigen/src/Core/GeneralProduct.h:#ifndef __CUDACC__
Eigen/src/Core/GeneralProduct.h:#endif // __CUDACC__
Eigen/src/Core/GenericPacketMath.h:#ifdef __CUDA_ARCH__
Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
Eigen/src/Core/GenericPacketMath.h:#ifndef __CUDACC__
Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
Eigen/src/Core/arch/CUDA/PacketMath.h:#ifndef EIGEN_PACKET_MATH_CUDA_H
Eigen/src/Core/arch/CUDA/PacketMath.h:#define EIGEN_PACKET_MATH_CUDA_H
Eigen/src/Core/arch/CUDA/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Eigen/src/Core/arch/CUDA/PacketMath.h:#endif // EIGEN_PACKET_MATH_CUDA_H
Eigen/src/Core/arch/CUDA/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
Eigen/src/Core/arch/CUDA/Half.h:// type Eigen::half (inheriting from CUDA's __half struct) with
Eigen/src/Core/arch/CUDA/Half.h:// to disk and the likes), but fast on GPUs.
Eigen/src/Core/arch/CUDA/Half.h:#ifndef EIGEN_HALF_CUDA_H
Eigen/src/Core/arch/CUDA/Half.h:#define EIGEN_HALF_CUDA_H
Eigen/src/Core/arch/CUDA/Half.h:#if !defined(EIGEN_HAS_CUDA_FP16)
Eigen/src/Core/arch/CUDA/Half.h:// Make our own __half definition that is similar to CUDA's.
Eigen/src/Core/arch/CUDA/Half.h:  #if !defined(EIGEN_HAS_CUDA_FP16)
Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/Half.h:// Definitions for CPUs and older CUDA, mostly working through conversion
Eigen/src/Core/arch/CUDA/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/Half.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__)
Eigen/src/Core/arch/CUDA/Half.h:#endif // EIGEN_HALF_CUDA_H
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#ifndef EIGEN_PACKET_MATH_HALF_CUDA_H
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#define EIGEN_PACKET_MATH_HALF_CUDA_H
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 530
Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#endif // EIGEN_PACKET_MATH_HALF_CUDA_H
Eigen/src/Core/arch/CUDA/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_CUDA_H
Eigen/src/Core/arch/CUDA/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_CUDA_H
Eigen/src/Core/arch/CUDA/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
Eigen/src/Core/arch/CUDA/MathFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
Eigen/src/Core/arch/CUDA/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_CUDA_H
Eigen/src/Core/arch/CUDA/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_CUDA_H
Eigen/src/Core/arch/CUDA/TypeCasting.h:#define EIGEN_TYPE_CASTING_CUDA_H
Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen/src/Core/arch/CUDA/TypeCasting.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen/src/Core/arch/CUDA/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_CUDA_H
Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
Eigen/src/Core/arch/CUDA/Complex.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
Eigen/src/Core/arch/CUDA/Complex.h:// building for CUDA to avoid non-constexpr methods.
Eigen/src/Core/arch/CUDA/Complex.h:#endif // EIGEN_COMPLEX_CUDA_H
Eigen/src/SVD/BDCSVD.h:#ifndef __CUDACC__

```
