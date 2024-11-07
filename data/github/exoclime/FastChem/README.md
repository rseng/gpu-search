# https://github.com/exoclime/FastChem

```console
_ext/Eigen/Core:// We need cuda_runtime.h/hip_runtime.h to ensure that
_ext/Eigen/Core:#if defined(EIGEN_CUDACC)
_ext/Eigen/Core:  #include <cuda_runtime.h>
_ext/Eigen/Core:#if defined(EIGEN_COMP_ICC) && defined(EIGEN_GPU_COMPILE_PHASE) \
_ext/Eigen/Core:#if defined(EIGEN_HAS_CUDA_FP16) || defined(EIGEN_HAS_HIP_FP16)
_ext/Eigen/Core:  #define EIGEN_HAS_GPU_FP16
_ext/Eigen/Core:#if defined(EIGEN_HAS_CUDA_BF16) || defined(EIGEN_HAS_HIP_BF16)
_ext/Eigen/Core:  #define EIGEN_HAS_GPU_BF16
_ext/Eigen/Core:#if defined EIGEN_VECTORIZE_GPU
_ext/Eigen/Core:  #include "src/Core/arch/GPU/PacketMath.h"
_ext/Eigen/Core:  #include "src/Core/arch/GPU/MathFunctions.h"
_ext/Eigen/Core:  #include "src/Core/arch/GPU/TypeCasting.h"
_ext/Eigen/Core:// on CUDA devices
_ext/Eigen/Core:#ifdef EIGEN_CUDACC
_ext/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
_ext/Eigen/src/Core/AssignEvaluator.h:#ifndef EIGEN_GPU_COMPILE_PHASE
_ext/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
_ext/Eigen/src/Core/util/Memory.h:#if ! defined EIGEN_ALLOCA && ! defined EIGEN_GPU_COMPILE_PHASE
_ext/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/util/Meta.h: #if defined(EIGEN_CUDA_ARCH)
_ext/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE) && !EIGEN_HAS_CXX11
_ext/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
_ext/Eigen/src/Core/util/Meta.h:    return CUDART_MAX_NORMAL_F;
_ext/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
_ext/Eigen/src/Core/util/Meta.h:    return CUDART_INF_F;
_ext/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
_ext/Eigen/src/Core/util/Meta.h:    return CUDART_NAN_F;
_ext/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
_ext/Eigen/src/Core/util/Meta.h:    return CUDART_INF;
_ext/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
_ext/Eigen/src/Core/util/Meta.h:    return CUDART_NAN;
_ext/Eigen/src/Core/util/Meta.h:#endif // defined(EIGEN_GPU_COMPILE_PHASE) && !EIGEN_HAS_CXX11
_ext/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE) && !EIGEN_HAS_CXX11
_ext/Eigen/src/Core/util/Meta.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) || (!defined(EIGEN_CUDA_ARCH) && defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
_ext/Eigen/src/Core/util/Meta.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) || (!defined(EIGEN_CUDA_ARCH) && defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
_ext/Eigen/src/Core/util/StaticAssert.h:        GPU_TENSOR_CONTRACTION_DOES_NOT_SUPPORT_OUTPUT_KERNELS=1,
_ext/Eigen/src/Core/util/Macros.h:#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
_ext/Eigen/src/Core/util/Macros.h:  #define EIGEN_COMP_NVCC  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
_ext/Eigen/src/Core/util/Macros.h:#elif defined(__CUDACC_VER__)
_ext/Eigen/src/Core/util/Macros.h:  #define EIGEN_COMP_NVCC __CUDACC_VER__
_ext/Eigen/src/Core/util/Macros.h:// Detect GPU compilers and architectures
_ext/Eigen/src/Core/util/Macros.h:// Note that this also makes EIGEN_CUDACC and EIGEN_HIPCC mutually exclusive
_ext/Eigen/src/Core/util/Macros.h:#if defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
_ext/Eigen/src/Core/util/Macros.h:  // Means the compiler is either nvcc or clang with CUDA enabled
_ext/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDACC __CUDACC__
_ext/Eigen/src/Core/util/Macros.h:#if defined(__CUDA_ARCH__) && !defined(EIGEN_NO_CUDA)
_ext/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDA_ARCH __CUDA_ARCH__
_ext/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC)
_ext/Eigen/src/Core/util/Macros.h:#include <cuda.h>
_ext/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDA_SDK_VER (CUDA_VERSION * 10)
_ext/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDA_SDK_VER 0
_ext/Eigen/src/Core/util/Macros.h:  // Means the compiler is HIPCC (analogous to EIGEN_CUDACC, but for HIP)
_ext/Eigen/src/Core/util/Macros.h:    // analogous to EIGEN_CUDA_ARCH, but for HIP
_ext/Eigen/src/Core/util/Macros.h:  // For HIP (ROCm 3.5 and higher), we need to explicitly set the launch_bounds attribute
_ext/Eigen/src/Core/util/Macros.h:  // specified. This results in failures on the HIP platform, for cases when a GPU kernel
_ext/Eigen/src/Core/util/Macros.h:  // couple of ROCm releases (compiler will go back to using 1024 value as the default)
_ext/Eigen/src/Core/util/Macros.h:// Unify CUDA/HIPCC
_ext/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC) || defined(EIGEN_HIPCC)
_ext/Eigen/src/Core/util/Macros.h:// If either EIGEN_CUDACC or EIGEN_HIPCC is defined, then define EIGEN_GPUCC
_ext/Eigen/src/Core/util/Macros.h:#define EIGEN_GPUCC
_ext/Eigen/src/Core/util/Macros.h:// EIGEN_CUDACC implies the CUDA compiler and is used to tweak Eigen code for use in CUDA kernels
_ext/Eigen/src/Core/util/Macros.h:// In most cases the same tweaks are required to the Eigen code to enable in both the HIP and CUDA kernels.
_ext/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDACC) || defined(EIGEN_HIPCC)
_ext/Eigen/src/Core/util/Macros.h:// For cases where the tweak is specific to CUDA, the code should be guarded with
_ext/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDACC)
_ext/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIP_DEVICE_COMPILE)
_ext/Eigen/src/Core/util/Macros.h:// If either EIGEN_CUDA_ARCH or EIGEN_HIP_DEVICE_COMPILE is defined, then define EIGEN_GPU_COMPILE_PHASE
_ext/Eigen/src/Core/util/Macros.h:#define EIGEN_GPU_COMPILE_PHASE
_ext/Eigen/src/Core/util/Macros.h:// GPU compilers (HIPCC, NVCC) typically do two passes over the source code,
_ext/Eigen/src/Core/util/Macros.h://   + another to compile the source for the "device" (ie. GPU)
_ext/Eigen/src/Core/util/Macros.h:// EIGEN_CUDA_ARCH implies the device compilation phase in CUDA
_ext/Eigen/src/Core/util/Macros.h:// In most cases, the "host" / "device" specific code is the same for both HIP and CUDA
_ext/Eigen/src/Core/util/Macros.h://       #if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/util/Macros.h://       #if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIP_DEVICE_COMPILE)
_ext/Eigen/src/Core/util/Macros.h:// For cases where the tweak is specific to CUDA, the code should be guarded with
_ext/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDA_ARCH)
_ext/Eigen/src/Core/util/Macros.h:#if EIGEN_HAS_CXX11 && !defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/util/Macros.h:  #if defined(EIGEN_CUDACC)
_ext/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC) && EIGEN_HAS_CONSTEXPR
_ext/Eigen/src/Core/util/Macros.h:    #ifdef __CUDACC_RELAXED_CONSTEXPR__
_ext/Eigen/src/Core/util/Macros.h:  #elif defined(__clang__) && defined(__CUDA__) && __has_feature(cxx_relaxed_constexpr)
_ext/Eigen/src/Core/util/Macros.h:#if (EIGEN_COMP_MSVC || EIGEN_COMP_ICC) && !defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/util/Macros.h:// GPU stuff
_ext/Eigen/src/Core/util/Macros.h:// Disable some features when compiling with GPU compilers (NVCC/clang-cuda/SYCL/HIPCC)
_ext/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC) || defined(SYCL_DEVICE_ONLY) || defined(EIGEN_HIPCC)
_ext/Eigen/src/Core/util/Macros.h:// All functions callable from CUDA/HIP code must be qualified with __device__
_ext/Eigen/src/Core/util/Macros.h:#elif defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/util/Macros.h:// When compiling CUDA/HIP device code with NVCC or HIPCC
_ext/Eigen/src/Core/util/Macros.h:#if (defined(EIGEN_CUDA_ARCH) && defined(__NVCC__)) || defined(EIGEN_HIP_DEVICE_COMPILE)
_ext/Eigen/src/Core/util/Macros.h:  // For older MSVC versions, as well as 1900 && CUDA 8, using the base operator is necessary,
_ext/Eigen/src/Core/util/Macros.h:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(EIGEN_CUDA_ARCH) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL) && !defined(EIGEN_HIP_DEVICE_COMPILE)
_ext/Eigen/src/Core/util/Macros.h:#  if defined(EIGEN_CUDA_ARCH)
_ext/Eigen/src/Core/util/ConfigureVectorization.h:#if (defined EIGEN_CUDACC)
_ext/Eigen/src/Core/util/ConfigureVectorization.h:  #if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/util/ConfigureVectorization.h:    // GPU code is always vectorized and requires memory alignment for
_ext/Eigen/src/Core/util/ConfigureVectorization.h:#if !(defined(EIGEN_DONT_VECTORIZE) || defined(EIGEN_GPUCC))
_ext/Eigen/src/Core/util/ConfigureVectorization.h:#if defined(__F16C__) && (!defined(EIGEN_GPUCC) && (!defined(EIGEN_COMP_CLANG) || EIGEN_COMP_CLANG>=380))
_ext/Eigen/src/Core/util/ConfigureVectorization.h:#if defined EIGEN_CUDACC
_ext/Eigen/src/Core/util/ConfigureVectorization.h:  #define EIGEN_VECTORIZE_GPU
_ext/Eigen/src/Core/util/ConfigureVectorization.h:  #if EIGEN_CUDA_SDK_VER >= 70500
_ext/Eigen/src/Core/util/ConfigureVectorization.h:    #define EIGEN_HAS_CUDA_FP16
_ext/Eigen/src/Core/util/ConfigureVectorization.h:#if defined(EIGEN_HAS_CUDA_FP16)
_ext/Eigen/src/Core/util/ConfigureVectorization.h:  #include <cuda_runtime_api.h>
_ext/Eigen/src/Core/util/ConfigureVectorization.h:  #include <cuda_fp16.h>
_ext/Eigen/src/Core/util/ConfigureVectorization.h:  #define EIGEN_VECTORIZE_GPU
_ext/Eigen/src/Core/MathFunctions.h:// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/MathFunctions.h:// GPU, and correctly handles special cases (unlike MSVC).
_ext/Eigen/src/Core/MathFunctions.h:  #if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/MathFunctions.h:  #if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/MathFunctions.h:  #if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/MathFunctions.h:#if (!defined(EIGEN_GPUCC) || defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
_ext/Eigen/src/Core/MathFunctions.h:#if !defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:// HIP and CUDA do not support long double.
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if (!defined(EIGEN_GPUCC)) && EIGEN_FAST_MATH && !defined(SYCL_DEVICE_ONLY)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/functors/UnaryFunctors.h:#ifndef EIGEN_GPU_COMPILE_PHASE
_ext/Eigen/src/Core/functors/UnaryFunctors.h:#endif  // #ifndef EIGEN_GPU_COMPILE_PHASE
_ext/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef EIGEN_GPUCC
_ext/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
_ext/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
_ext/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
_ext/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
_ext/Eigen/src/Core/GeneralProduct.h:    #ifndef EIGEN_GPU_COMPILE_PHASE
_ext/Eigen/src/Core/GenericPacketMath.h:#elif defined(EIGEN_CUDA_ARCH)
_ext/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
_ext/Eigen/src/Core/GenericPacketMath.h:#if !defined(EIGEN_GPUCC)
_ext/Eigen/src/Core/ProductEvaluators.h:#ifndef EIGEN_GPUCC
_ext/Eigen/src/Core/ProductEvaluators.h:#ifndef EIGEN_GPUCC
_ext/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
_ext/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
_ext/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#ifndef EIGEN_PACKET_MATH_GPU_H
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_PACKET_MATH_GPU_H
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_HIP_DEVICE_COMPILE) || (defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_GPU_HAS_LDG 1
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if (defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_CUDA_HAS_FP16_ARITHMETIC 1
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_HIP_DEVICE_COMPILE) || defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_GPU_HAS_FP16_ARITHMETIC 1
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:// We need to distinguish ‘clang as the CUDA compiler’ from ‘clang as the host compiler,
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:// Packet4h2 must be defined in the macro without EIGEN_CUDA_ARCH, meaning
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_HAS_CUDA_FP16) || defined(EIGEN_HAS_HIP_FP16)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:// #if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)) || \
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#elif defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:// the implementation of GPU half reduction.
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:// #endif // defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // defined(EIGEN_HAS_CUDA_FP16) || defined(EIGEN_HAS_HIP_FP16)
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#undef EIGEN_GPU_HAS_LDG
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#undef EIGEN_CUDA_HAS_FP16_ARITHMETIC
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#undef EIGEN_GPU_HAS_FP16_ARITHMETIC
_ext/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // EIGEN_PACKET_MATH_GPU_H
_ext/Eigen/src/Core/arch/GPU/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_GPU_H
_ext/Eigen/src/Core/arch/GPU/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_GPU_H
_ext/Eigen/src/Core/arch/GPU/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
_ext/Eigen/src/Core/arch/GPU/MathFunctions.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
_ext/Eigen/src/Core/arch/GPU/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_GPU_H
_ext/Eigen/src/Core/arch/GPU/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_GPU_H
_ext/Eigen/src/Core/arch/GPU/TypeCasting.h:#define EIGEN_TYPE_CASTING_GPU_H
_ext/Eigen/src/Core/arch/GPU/TypeCasting.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
_ext/Eigen/src/Core/arch/GPU/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_GPU_H
_ext/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
_ext/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
_ext/Eigen/src/Core/arch/CUDA/Complex.h:// operators and functors for complex types when building for CUDA to enable
_ext/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(EIGEN_CUDACC) && defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/arch/CUDA/Complex.h:#endif  // EIGEN_CUDACC && EIGEN_GPU_COMPILE_PHASE
_ext/Eigen/src/Core/arch/CUDA/Complex.h:#endif  // EIGEN_COMPLEX_CUDA_H
_ext/Eigen/src/Core/arch/SYCL/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
_ext/Eigen/src/Core/arch/SYCL/InteropHeaders.h:// Make sure this is only available when targeting a GPU: we don't want to
_ext/Eigen/src/Core/arch/HIP/hcc/math_constants.h: *  HIP equivalent of the CUDA header of the same name
_ext/Eigen/src/Core/arch/Default/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
_ext/Eigen/src/Core/arch/Default/Half.h:// type Eigen::half (inheriting either from CUDA's or HIP's __half struct) with
_ext/Eigen/src/Core/arch/Default/Half.h:// to disk and the likes), but fast on GPUs.
_ext/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16) || defined(EIGEN_HAS_ARM64_FP16_SCALAR_ARITHMETIC)
_ext/Eigen/src/Core/arch/Default/Half.h:// When compiling with GPU support, the "__half_raw" base class as well as
_ext/Eigen/src/Core/arch/Default/Half.h:// some other routines are defined in the GPU compiler header files
_ext/Eigen/src/Core/arch/Default/Half.h:// (cuda_fp16.h, hip_fp16.h), and they are not tagged constexpr
_ext/Eigen/src/Core/arch/Default/Half.h:// GPU support. Hence the need to disable EIGEN_CONSTEXPR when building
_ext/Eigen/src/Core/arch/Default/Half.h:// Eigen with GPU support
_ext/Eigen/src/Core/arch/Default/Half.h:// This is required because of a quirk in the way TensorFlow GPU builds are done.
_ext/Eigen/src/Core/arch/Default/Half.h:// When compiling TensorFlow source code with GPU support, files that
_ext/Eigen/src/Core/arch/Default/Half.h://  * contain GPU kernels (i.e. *.cu.cc files) are compiled via hipcc
_ext/Eigen/src/Core/arch/Default/Half.h://  * do not contain GPU kernels ( i.e. *.cc files) are compiled via gcc (typically)
_ext/Eigen/src/Core/arch/Default/Half.h:#if !defined(EIGEN_HAS_GPU_FP16) || !defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/arch/Default/Half.h:// Make our own __half_raw definition that is similar to CUDA's.
_ext/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_GPU_FP16) && !defined(EIGEN_GPU_COMPILE_PHASE))
_ext/Eigen/src/Core/arch/Default/Half.h:#elif defined(EIGEN_HAS_CUDA_FP16)
_ext/Eigen/src/Core/arch/Default/Half.h:  #if EIGEN_CUDA_SDK_VER < 90000
_ext/Eigen/src/Core/arch/Default/Half.h:    // In CUDA < 9.0, __half is the equivalent of CUDA 9's __half_raw
_ext/Eigen/src/Core/arch/Default/Half.h:  #endif // defined(EIGEN_HAS_CUDA_FP16)
_ext/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16)
_ext/Eigen/src/Core/arch/Default/Half.h: #elif defined(EIGEN_HAS_CUDA_FP16)
_ext/Eigen/src/Core/arch/Default/Half.h:  #if EIGEN_CUDA_SDK_VER >= 90000
_ext/Eigen/src/Core/arch/Default/Half.h:#if !defined(EIGEN_HAS_GPU_FP16) || !defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/arch/Default/Half.h:  // * when compiling without GPU support enabled
_ext/Eigen/src/Core/arch/Default/Half.h:  // * during host compile phase when compiling with GPU support enabled
_ext/Eigen/src/Core/arch/Default/Half.h:#elif defined(EIGEN_HAS_CUDA_FP16)
_ext/Eigen/src/Core/arch/Default/Half.h:  // Note that EIGEN_CUDA_SDK_VER is set to 0 even when compiling with HIP, so
_ext/Eigen/src/Core/arch/Default/Half.h:  // (EIGEN_CUDA_SDK_VER < 90000) is true even for HIP!  So keeping this within
_ext/Eigen/src/Core/arch/Default/Half.h:  // #if defined(EIGEN_HAS_CUDA_FP16) is needed
_ext/Eigen/src/Core/arch/Default/Half.h:  #if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER < 90000
_ext/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16)
_ext/Eigen/src/Core/arch/Default/Half.h: #elif defined(EIGEN_HAS_CUDA_FP16)
_ext/Eigen/src/Core/arch/Default/Half.h:  #if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER >= 90000
_ext/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16) && !defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && \
_ext/Eigen/src/Core/arch/Default/Half.h:     EIGEN_CUDA_ARCH >= 530) ||                                  \
_ext/Eigen/src/Core/arch/Default/Half.h:// fp16 type since GPU halfs are rather different from native CPU halfs.
_ext/Eigen/src/Core/arch/Default/Half.h:// TODO: Rename to something like EIGEN_HAS_NATIVE_GPU_FP16
_ext/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER >= 90000
_ext/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER >= 90000
_ext/Eigen/src/Core/arch/Default/Half.h:// We need to distinguish ‘clang as the CUDA compiler’ from ‘clang as the host compiler,
_ext/Eigen/src/Core/arch/Default/Half.h:#if EIGEN_COMP_CLANG && defined(EIGEN_CUDACC)
_ext/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_HAS_NATIVE_FP16)
_ext/Eigen/src/Core/arch/Default/Half.h:// Definitions for CPUs and older HIP+CUDA, mostly working through conversion
_ext/Eigen/src/Core/arch/Default/Half.h:#if defined(__clang__) && defined(__CUDA__)
_ext/Eigen/src/Core/arch/Default/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
_ext/Eigen/src/Core/arch/Default/Half.h:  // Fortunately, since we need to disable EIGEN_CONSTEXPR for GPU anyway, we can get out
_ext/Eigen/src/Core/arch/Default/Half.h:  // of this catch22 by having separate bodies for GPU / non GPU
_ext/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16)
_ext/Eigen/src/Core/arch/Default/Half.h:  // HIP/CUDA/Default have a member 'x' of type uint16_t.
_ext/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
_ext/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
_ext/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
_ext/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530) || \
_ext/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && EIGEN_CUDA_SDK_VER >= 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
_ext/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530) || \
_ext/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300) || \
_ext/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300) || \
_ext/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
_ext/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
_ext/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16) || defined(EIGEN_HAS_ARM64_FP16_SCALAR_ARITHMETIC)
_ext/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
_ext/Eigen/src/Core/arch/Default/Half.h:// The __shfl* functions are only valid on HIP or _CUDA_ARCH_ >= 300.
_ext/Eigen/src/Core/arch/Default/Half.h://   CUDA defines them for (__CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__))
_ext/Eigen/src/Core/arch/Default/Half.h:// HIP and CUDA prior to SDK 9.0 define
_ext/Eigen/src/Core/arch/Default/Half.h:// CUDA since 9.0 deprecates those and instead defines
_ext/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_CUDACC) && (!defined(EIGEN_CUDA_ARCH) || EIGEN_CUDA_ARCH >= 300)) \
_ext/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && EIGEN_CUDA_SDK_VER >= 90000
_ext/Eigen/src/Core/arch/Default/Half.h:#else // HIP or CUDA SDK < 9.0
_ext/Eigen/src/Core/arch/Default/Half.h:#endif // HIP vs CUDA
_ext/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_CUDACC) && (!defined(EIGEN_CUDA_ARCH) || EIGEN_CUDA_ARCH >= 350)) \
_ext/Eigen/src/Core/arch/Default/BFloat16.h:// We need to distinguish ‘clang as the CUDA compiler’ from ‘clang as the host compiler,
_ext/Eigen/src/Core/arch/Default/BFloat16.h:#if EIGEN_COMP_CLANG && defined(EIGEN_CUDACC)
_ext/Eigen/src/Core/arch/Default/BFloat16.h:#if defined(EIGEN_HAS_CUDA_BF16) && defined(EIGEN_HAS_NATIVE_BF16)
_ext/Eigen/src/Core/arch/Default/BFloat16.h:#if EIGEN_COMP_CLANG && defined(EIGEN_CUDACC)
_ext/Eigen/src/Core/arch/Default/BFloat16.h:#if (defined(EIGEN_HAS_CUDA_BF16) && defined(EIGEN_HAS_HIP_BF16))
_ext/Eigen/src/Core/arch/Default/BFloat16.h:#if (defined(EIGEN_HAS_CUDA_BF16) && defined(EIGEN_HAS_HIP_BF16))
_ext/Eigen/src/Core/arch/Default/TypeCasting.h:    #if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
_ext/Eigen/src/Core/arch/Default/TypeCasting.h:    #if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
_ext/Eigen/src/Core/arch/Default/TypeCasting.h:    #if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \

```
