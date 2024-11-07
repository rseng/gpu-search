# https://github.com/vdemichev/DiaNN

```console
eigen/Eigen/Core:#if defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
eigen/Eigen/Core:  #define EIGEN_CUDACC __CUDACC__
eigen/Eigen/Core:#if defined(__CUDA_ARCH__) && !defined(EIGEN_NO_CUDA)
eigen/Eigen/Core:  #define EIGEN_CUDA_ARCH __CUDA_ARCH__
eigen/Eigen/Core:#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
eigen/Eigen/Core:#define EIGEN_CUDACC_VER  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
eigen/Eigen/Core:#elif defined(__CUDACC_VER__)
eigen/Eigen/Core:#define EIGEN_CUDACC_VER __CUDACC_VER__
eigen/Eigen/Core:#define EIGEN_CUDACC_VER 0
eigen/Eigen/Core:// Handle NVCC/CUDA/SYCL
eigen/Eigen/Core:#if defined(__CUDACC__) || defined(__SYCL_DEVICE_ONLY__)
eigen/Eigen/Core:  // Do not try asserts on CUDA and SYCL!
eigen/Eigen/Core:  // All functions callable from CUDA code must be qualified with __device__
eigen/Eigen/Core:  #ifdef __CUDACC__
eigen/Eigen/Core:    // Do not try to vectorize on CUDA and SYCL!
eigen/Eigen/Core:    // We need cuda_runtime.h to ensure that that EIGEN_USING_STD_MATH macro
eigen/Eigen/Core:    #include <cuda_runtime.h>
eigen/Eigen/Core:// When compiling CUDA device code with NVCC, pull in math functions from the
eigen/Eigen/Core:#if defined(__CUDA_ARCH__) && defined(__NVCC__)
eigen/Eigen/Core:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL)
eigen/Eigen/Core:#if defined __CUDACC__
eigen/Eigen/Core:  #define EIGEN_VECTORIZE_CUDA
eigen/Eigen/Core:  #if EIGEN_CUDACC_VER >= 70500
eigen/Eigen/Core:    #define EIGEN_HAS_CUDA_FP16
eigen/Eigen/Core:#if defined EIGEN_HAS_CUDA_FP16
eigen/Eigen/Core:  #include <cuda_fp16.h>
eigen/Eigen/Core:#include "src/Core/arch/CUDA/Half.h"
eigen/Eigen/Core:#include "src/Core/arch/CUDA/PacketMathHalf.h"
eigen/Eigen/Core:#include "src/Core/arch/CUDA/TypeCasting.h"
eigen/Eigen/Core:#if defined EIGEN_VECTORIZE_CUDA
eigen/Eigen/Core:  #include "src/Core/arch/CUDA/PacketMath.h"
eigen/Eigen/Core:  #include "src/Core/arch/CUDA/MathFunctions.h"
eigen/Eigen/Core:// on CUDA devices
eigen/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
eigen/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
eigen/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
eigen/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
eigen/Eigen/src/Core/util/Meta.h:  static float (max)() { return CUDART_MAX_NORMAL_F; }
eigen/Eigen/src/Core/util/Meta.h:  static float infinity() { return CUDART_INF_F; }
eigen/Eigen/src/Core/util/Meta.h:  static float quiet_NaN() { return CUDART_NAN_F; }
eigen/Eigen/src/Core/util/Meta.h:  static double infinity() { return CUDART_INF; }
eigen/Eigen/src/Core/util/Meta.h:  static double quiet_NaN() { return CUDART_NAN; }
eigen/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
eigen/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
eigen/Eigen/src/Core/util/Macros.h:  && (!defined(__NVCC__) || !EIGEN_ARCH_ARM_OR_ARM64 || (EIGEN_CUDACC_VER >= 80000) )
eigen/Eigen/src/Core/util/Macros.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/util/Macros.h:#if EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && (EIGEN_COMP_CLANG || EIGEN_CUDACC_VER >= 70500))
eigen/Eigen/src/Core/util/Macros.h:#if (defined __CUDACC__)
eigen/Eigen/src/Core/util/Macros.h:#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC < 1900 || EIGEN_CUDACC_VER>0)
eigen/Eigen/src/Core/util/Macros.h:  // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
eigen/Eigen/src/Core/util/Macros.h:#  ifdef __CUDA_ARCH__
eigen/Eigen/src/Core/util/ConfigureVectorization.h:#if (defined EIGEN_CUDACC)
eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #if defined(EIGEN_GPUCC)
eigen/Eigen/src/Core/util/ConfigureVectorization.h:    // GPU code is always vectorized and requires memory alignment for
eigen/Eigen/src/Core/util/ConfigureVectorization.h:#if !(defined(EIGEN_DONT_VECTORIZE) || defined(EIGEN_GPUCC))
eigen/Eigen/src/Core/util/ConfigureVectorization.h:#if defined(__F16C__) && (!defined(EIGEN_GPUCC) && (!defined(EIGEN_COMP_CLANG) || EIGEN_COMP_CLANG>=380))
eigen/Eigen/src/Core/util/ConfigureVectorization.h:#if defined EIGEN_CUDACC
eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #define EIGEN_VECTORIZE_GPU
eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #if EIGEN_CUDA_SDK_VER >= 70500
eigen/Eigen/src/Core/util/ConfigureVectorization.h:    #define EIGEN_HAS_CUDA_FP16
eigen/Eigen/src/Core/util/ConfigureVectorization.h:#if defined(EIGEN_HAS_CUDA_FP16)
eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #include <cuda_runtime_api.h>
eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #include <cuda_fp16.h>
eigen/Eigen/src/Core/util/ConfigureVectorization.h:  #define EIGEN_VECTORIZE_GPU
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
eigen/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
eigen/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
eigen/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
eigen/Eigen/src/Core/MathFunctions.h:#ifndef __CUDA_ARCH__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#if (!defined(__CUDACC__)) && EIGEN_FAST_MATH
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef __CUDACC__
eigen/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
eigen/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
eigen/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
eigen/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
eigen/Eigen/src/Core/GeneralProduct.h:    #ifndef EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/GenericPacketMath.h:#ifdef __CUDA_ARCH__
eigen/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
eigen/Eigen/src/Core/GenericPacketMath.h:#ifndef __CUDACC__
eigen/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
eigen/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
eigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
eigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
eigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#ifndef EIGEN_PACKET_MATH_GPU_H
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_PACKET_MATH_GPU_H
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:// We need to distinguish ‘clang as the CUDA compiler’ from ‘clang as the host compiler,
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#endif  // EIGEN_CUDA_ARCH || defined(EIGEN_HIP_DEVICE_COMPILE)
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:// Packet4h2 must be defined in the macro without EIGEN_CUDA_ARCH, meaning
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDACC)) || \
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:  (defined(EIGEN_HAS_CUDA_FP16) && defined(__clang__) && defined(__CUDA__))
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if !defined(EIGEN_CUDA_ARCH) && !defined(EIGEN_HIPCC)
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if !defined(EIGEN_CUDA_ARCH) && !defined(EIGEN_HIPCC)
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
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 350
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:// the implementation of GPU half reduction.
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else  // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#else // EIGEN_CUDA_ARCH
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // defined(EIGEN_CUDA_ARCH)
eigen/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDACC)
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
eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#ifndef EIGEN_PACKET_MATH_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#define EIGEN_PACKET_MATH_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
eigen/Eigen/src/Core/arch/CUDA/PacketMath.h:#endif // EIGEN_PACKET_MATH_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
eigen/Eigen/src/Core/arch/CUDA/Half.h:// type Eigen::half (inheriting from CUDA's __half struct) with
eigen/Eigen/src/Core/arch/CUDA/Half.h:// to disk and the likes), but fast on GPUs.
eigen/Eigen/src/Core/arch/CUDA/Half.h:#ifndef EIGEN_HALF_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/Half.h:#define EIGEN_HALF_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if !defined(EIGEN_HAS_CUDA_FP16)
eigen/Eigen/src/Core/arch/CUDA/Half.h:// Make our own __half_raw definition that is similar to CUDA's.
eigen/Eigen/src/Core/arch/CUDA/Half.h:#elif defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER < 90000
eigen/Eigen/src/Core/arch/CUDA/Half.h:// In CUDA < 9.0, __half is the equivalent of CUDA 9's __half_raw
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER >= 90000
eigen/Eigen/src/Core/arch/CUDA/Half.h:  #if !defined(EIGEN_HAS_CUDA_FP16) || (defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER < 90000)
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDACC_VER) && EIGEN_CUDACC_VER >= 90000
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/CUDA/Half.h:// Definitions for CPUs and older CUDA, mostly working through conversion
eigen/Eigen/src/Core/arch/CUDA/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && EIGEN_CUDACC_VER >= 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if EIGEN_CUDACC_VER >= 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300
eigen/Eigen/src/Core/arch/CUDA/Half.h:  #if EIGEN_CUDACC_VER < 90000
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
eigen/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_CUDA_ARCH)
eigen/Eigen/src/Core/arch/CUDA/Half.h:#endif // EIGEN_HALF_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#ifndef EIGEN_PACKET_MATH_HALF_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#define EIGEN_PACKET_MATH_HALF_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if EIGEN_CUDACC_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530
eigen/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#endif // EIGEN_PACKET_MATH_HALF_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
eigen/Eigen/src/Core/arch/CUDA/MathFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
eigen/Eigen/src/Core/arch/CUDA/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:#define EIGEN_TYPE_CASTING_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
eigen/Eigen/src/Core/arch/CUDA/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
eigen/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
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
eigen/Eigen/src/SVD/BDCSVD.h:#ifndef __CUDACC__
MiniDNN/include/external/sparsepp/spp_config.h:    // doesn't actually support __int128 as of CUDA_VERSION=7500
MiniDNN/include/external/sparsepp/spp_config.h:    #if defined(__CUDACC__)
MiniDNN/include/external/sparsepp/spp_config.h:    // Nevertheless, as of CUDA 7.5, using __float128 with the host
Third-party/Arrow LICENSE.txt:Copyright (c) 2018-2020      NVIDIA CORPORATION. All rights reserved.

```
