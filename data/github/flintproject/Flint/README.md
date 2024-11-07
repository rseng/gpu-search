# https://github.com/flintproject/Flint

```console
source/include/Eigen/Core:// We need cuda_runtime.h/hip_runtime.h to ensure that
source/include/Eigen/Core:#if defined(EIGEN_CUDACC)
source/include/Eigen/Core:  #include <cuda_runtime.h>
source/include/Eigen/Core:#if defined(EIGEN_COMP_ICC) && defined(EIGEN_GPU_COMPILE_PHASE) \
source/include/Eigen/Core:#if defined(EIGEN_HAS_CUDA_FP16) || defined(EIGEN_HAS_HIP_FP16)
source/include/Eigen/Core:  #define EIGEN_HAS_GPU_FP16
source/include/Eigen/Core:#if defined(EIGEN_HAS_CUDA_BF16) || defined(EIGEN_HAS_HIP_BF16)
source/include/Eigen/Core:  #define EIGEN_HAS_GPU_BF16
source/include/Eigen/Core:#if defined EIGEN_VECTORIZE_GPU
source/include/Eigen/Core:  #include "src/Core/arch/GPU/PacketMath.h"
source/include/Eigen/Core:  #include "src/Core/arch/GPU/MathFunctions.h"
source/include/Eigen/Core:  #include "src/Core/arch/GPU/TypeCasting.h"
source/include/Eigen/Core:// on CUDA devices
source/include/Eigen/Core:#ifdef EIGEN_CUDACC
source/include/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
source/include/Eigen/src/Core/AssignEvaluator.h:#ifndef EIGEN_GPU_COMPILE_PHASE
source/include/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
source/include/Eigen/src/Core/util/Memory.h:#if ! defined EIGEN_ALLOCA && ! defined EIGEN_GPU_COMPILE_PHASE
source/include/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/util/Meta.h: #if defined(EIGEN_CUDA_ARCH)
source/include/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE) && !EIGEN_HAS_CXX11
source/include/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
source/include/Eigen/src/Core/util/Meta.h:    return CUDART_MAX_NORMAL_F;
source/include/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
source/include/Eigen/src/Core/util/Meta.h:    return CUDART_INF_F;
source/include/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
source/include/Eigen/src/Core/util/Meta.h:    return CUDART_NAN_F;
source/include/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
source/include/Eigen/src/Core/util/Meta.h:    return CUDART_INF;
source/include/Eigen/src/Core/util/Meta.h:  #if defined(EIGEN_CUDA_ARCH)
source/include/Eigen/src/Core/util/Meta.h:    return CUDART_NAN;
source/include/Eigen/src/Core/util/Meta.h:#endif // defined(EIGEN_GPU_COMPILE_PHASE) && !EIGEN_HAS_CXX11
source/include/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE) && !EIGEN_HAS_CXX11
source/include/Eigen/src/Core/util/Meta.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) || (!defined(EIGEN_CUDA_ARCH) && defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
source/include/Eigen/src/Core/util/Meta.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) || (!defined(EIGEN_CUDA_ARCH) && defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
source/include/Eigen/src/Core/util/StaticAssert.h:        GPU_TENSOR_CONTRACTION_DOES_NOT_SUPPORT_OUTPUT_KERNELS=1,
source/include/Eigen/src/Core/util/Macros.h:#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
source/include/Eigen/src/Core/util/Macros.h:  #define EIGEN_COMP_NVCC  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
source/include/Eigen/src/Core/util/Macros.h:#elif defined(__CUDACC_VER__)
source/include/Eigen/src/Core/util/Macros.h:  #define EIGEN_COMP_NVCC __CUDACC_VER__
source/include/Eigen/src/Core/util/Macros.h:// Detect GPU compilers and architectures
source/include/Eigen/src/Core/util/Macros.h:// Note that this also makes EIGEN_CUDACC and EIGEN_HIPCC mutually exclusive
source/include/Eigen/src/Core/util/Macros.h:#if defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
source/include/Eigen/src/Core/util/Macros.h:  // Means the compiler is either nvcc or clang with CUDA enabled
source/include/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDACC __CUDACC__
source/include/Eigen/src/Core/util/Macros.h:#if defined(__CUDA_ARCH__) && !defined(EIGEN_NO_CUDA)
source/include/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDA_ARCH __CUDA_ARCH__
source/include/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC)
source/include/Eigen/src/Core/util/Macros.h:#include <cuda.h>
source/include/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDA_SDK_VER (CUDA_VERSION * 10)
source/include/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDA_SDK_VER 0
source/include/Eigen/src/Core/util/Macros.h:  // Means the compiler is HIPCC (analogous to EIGEN_CUDACC, but for HIP)
source/include/Eigen/src/Core/util/Macros.h:    // analogous to EIGEN_CUDA_ARCH, but for HIP
source/include/Eigen/src/Core/util/Macros.h:  // For HIP (ROCm 3.5 and higher), we need to explicitly set the launch_bounds attribute
source/include/Eigen/src/Core/util/Macros.h:  // specified. This results in failures on the HIP platform, for cases when a GPU kernel
source/include/Eigen/src/Core/util/Macros.h:  // couple of ROCm releases (compiler will go back to using 1024 value as the default)
source/include/Eigen/src/Core/util/Macros.h:// Unify CUDA/HIPCC
source/include/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC) || defined(EIGEN_HIPCC)
source/include/Eigen/src/Core/util/Macros.h:// If either EIGEN_CUDACC or EIGEN_HIPCC is defined, then define EIGEN_GPUCC
source/include/Eigen/src/Core/util/Macros.h:#define EIGEN_GPUCC
source/include/Eigen/src/Core/util/Macros.h:// EIGEN_CUDACC implies the CUDA compiler and is used to tweak Eigen code for use in CUDA kernels
source/include/Eigen/src/Core/util/Macros.h:// In most cases the same tweaks are required to the Eigen code to enable in both the HIP and CUDA kernels.
source/include/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDACC) || defined(EIGEN_HIPCC)
source/include/Eigen/src/Core/util/Macros.h:// For cases where the tweak is specific to CUDA, the code should be guarded with
source/include/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDACC)
source/include/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIP_DEVICE_COMPILE)
source/include/Eigen/src/Core/util/Macros.h:// If either EIGEN_CUDA_ARCH or EIGEN_HIP_DEVICE_COMPILE is defined, then define EIGEN_GPU_COMPILE_PHASE
source/include/Eigen/src/Core/util/Macros.h:#define EIGEN_GPU_COMPILE_PHASE
source/include/Eigen/src/Core/util/Macros.h:// GPU compilers (HIPCC, NVCC) typically do two passes over the source code,
source/include/Eigen/src/Core/util/Macros.h://   + another to compile the source for the "device" (ie. GPU)
source/include/Eigen/src/Core/util/Macros.h:// EIGEN_CUDA_ARCH implies the device compilation phase in CUDA
source/include/Eigen/src/Core/util/Macros.h:// In most cases, the "host" / "device" specific code is the same for both HIP and CUDA
source/include/Eigen/src/Core/util/Macros.h://       #if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/util/Macros.h://       #if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIP_DEVICE_COMPILE)
source/include/Eigen/src/Core/util/Macros.h:// For cases where the tweak is specific to CUDA, the code should be guarded with
source/include/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDA_ARCH)
source/include/Eigen/src/Core/util/Macros.h:#if EIGEN_HAS_CXX11 && !defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/util/Macros.h:  #if defined(EIGEN_CUDACC)
source/include/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC) && EIGEN_HAS_CONSTEXPR
source/include/Eigen/src/Core/util/Macros.h:    #ifdef __CUDACC_RELAXED_CONSTEXPR__
source/include/Eigen/src/Core/util/Macros.h:  #elif defined(__clang__) && defined(__CUDA__) && __has_feature(cxx_relaxed_constexpr)
source/include/Eigen/src/Core/util/Macros.h:#if (EIGEN_COMP_MSVC || EIGEN_COMP_ICC) && !defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/util/Macros.h:// GPU stuff
source/include/Eigen/src/Core/util/Macros.h:// Disable some features when compiling with GPU compilers (NVCC/clang-cuda/SYCL/HIPCC)
source/include/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC) || defined(SYCL_DEVICE_ONLY) || defined(EIGEN_HIPCC)
source/include/Eigen/src/Core/util/Macros.h:// All functions callable from CUDA/HIP code must be qualified with __device__
source/include/Eigen/src/Core/util/Macros.h:#elif defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/util/Macros.h:// When compiling CUDA/HIP device code with NVCC or HIPCC
source/include/Eigen/src/Core/util/Macros.h:#if (defined(EIGEN_CUDA_ARCH) && defined(__NVCC__)) || defined(EIGEN_HIP_DEVICE_COMPILE)
source/include/Eigen/src/Core/util/Macros.h:  // For older MSVC versions, as well as 1900 && CUDA 8, using the base operator is necessary,
source/include/Eigen/src/Core/util/Macros.h:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(EIGEN_CUDA_ARCH) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL) && !defined(EIGEN_HIP_DEVICE_COMPILE)
source/include/Eigen/src/Core/util/Macros.h:#  if defined(EIGEN_CUDA_ARCH)
source/include/Eigen/src/Core/util/ConfigureVectorization.h:#if (defined EIGEN_CUDACC)
source/include/Eigen/src/Core/util/ConfigureVectorization.h:  #if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/util/ConfigureVectorization.h:    // GPU code is always vectorized and requires memory alignment for
source/include/Eigen/src/Core/util/ConfigureVectorization.h:#if !(defined(EIGEN_DONT_VECTORIZE) || defined(EIGEN_GPUCC))
source/include/Eigen/src/Core/util/ConfigureVectorization.h:#if defined(__F16C__) && (!defined(EIGEN_GPUCC) && (!defined(EIGEN_COMP_CLANG) || EIGEN_COMP_CLANG>=380))
source/include/Eigen/src/Core/util/ConfigureVectorization.h:#if defined EIGEN_CUDACC
source/include/Eigen/src/Core/util/ConfigureVectorization.h:  #define EIGEN_VECTORIZE_GPU
source/include/Eigen/src/Core/util/ConfigureVectorization.h:  #if EIGEN_CUDA_SDK_VER >= 70500
source/include/Eigen/src/Core/util/ConfigureVectorization.h:    #define EIGEN_HAS_CUDA_FP16
source/include/Eigen/src/Core/util/ConfigureVectorization.h:#if defined(EIGEN_HAS_CUDA_FP16)
source/include/Eigen/src/Core/util/ConfigureVectorization.h:  #include <cuda_runtime_api.h>
source/include/Eigen/src/Core/util/ConfigureVectorization.h:  #include <cuda_fp16.h>
source/include/Eigen/src/Core/util/ConfigureVectorization.h:  #define EIGEN_VECTORIZE_GPU
source/include/Eigen/src/Core/MathFunctions.h:// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/MathFunctions.h:// GPU, and correctly handles special cases (unlike MSVC).
source/include/Eigen/src/Core/MathFunctions.h:  #if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/MathFunctions.h:  #if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/MathFunctions.h:  #if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/MathFunctions.h:#if (!defined(EIGEN_GPUCC) || defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
source/include/Eigen/src/Core/MathFunctions.h:#if !defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:// HIP and CUDA do not support long double.
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if (!defined(EIGEN_GPUCC)) && EIGEN_FAST_MATH && !defined(SYCL_DEVICE_ONLY)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/functors/UnaryFunctors.h:#ifndef EIGEN_GPU_COMPILE_PHASE
source/include/Eigen/src/Core/functors/UnaryFunctors.h:#endif  // #ifndef EIGEN_GPU_COMPILE_PHASE
source/include/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef EIGEN_GPUCC
source/include/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
source/include/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
source/include/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
source/include/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
source/include/Eigen/src/Core/GeneralProduct.h:    #ifndef EIGEN_GPU_COMPILE_PHASE
source/include/Eigen/src/Core/GenericPacketMath.h:#elif defined(EIGEN_CUDA_ARCH)
source/include/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
source/include/Eigen/src/Core/GenericPacketMath.h:#if !defined(EIGEN_GPUCC)
source/include/Eigen/src/Core/ProductEvaluators.h:#ifndef EIGEN_GPUCC
source/include/Eigen/src/Core/ProductEvaluators.h:#ifndef EIGEN_GPUCC
source/include/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
source/include/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
source/include/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#ifndef EIGEN_PACKET_MATH_GPU_H
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_PACKET_MATH_GPU_H
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_HIP_DEVICE_COMPILE) || (defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_GPU_HAS_LDG 1
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if (defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_CUDA_HAS_FP16_ARITHMETIC 1
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_HIP_DEVICE_COMPILE) || defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_GPU_HAS_FP16_ARITHMETIC 1
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:// We need to distinguish ‘clang as the CUDA compiler’ from ‘clang as the host compiler,
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:// Packet4h2 must be defined in the macro without EIGEN_CUDA_ARCH, meaning
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_HAS_CUDA_FP16) || defined(EIGEN_HAS_HIP_FP16)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:// #if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)) || \
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#elif defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:// the implementation of GPU half reduction.
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:// #endif // defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // defined(EIGEN_HAS_CUDA_FP16) || defined(EIGEN_HAS_HIP_FP16)
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#undef EIGEN_GPU_HAS_LDG
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#undef EIGEN_CUDA_HAS_FP16_ARITHMETIC
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#undef EIGEN_GPU_HAS_FP16_ARITHMETIC
source/include/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // EIGEN_PACKET_MATH_GPU_H
source/include/Eigen/src/Core/arch/GPU/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_GPU_H
source/include/Eigen/src/Core/arch/GPU/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_GPU_H
source/include/Eigen/src/Core/arch/GPU/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
source/include/Eigen/src/Core/arch/GPU/MathFunctions.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
source/include/Eigen/src/Core/arch/GPU/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_GPU_H
source/include/Eigen/src/Core/arch/GPU/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_GPU_H
source/include/Eigen/src/Core/arch/GPU/TypeCasting.h:#define EIGEN_TYPE_CASTING_GPU_H
source/include/Eigen/src/Core/arch/GPU/TypeCasting.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
source/include/Eigen/src/Core/arch/GPU/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_GPU_H
source/include/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
source/include/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
source/include/Eigen/src/Core/arch/CUDA/Complex.h:// operators and functors for complex types when building for CUDA to enable
source/include/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(EIGEN_CUDACC) && defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/arch/CUDA/Complex.h:#endif  // EIGEN_CUDACC && EIGEN_GPU_COMPILE_PHASE
source/include/Eigen/src/Core/arch/CUDA/Complex.h:#endif  // EIGEN_COMPLEX_CUDA_H
source/include/Eigen/src/Core/arch/SYCL/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
source/include/Eigen/src/Core/arch/SYCL/InteropHeaders.h:// Make sure this is only available when targeting a GPU: we don't want to
source/include/Eigen/src/Core/arch/HIP/hcc/math_constants.h: *  HIP equivalent of the CUDA header of the same name
source/include/Eigen/src/Core/arch/Default/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
source/include/Eigen/src/Core/arch/Default/Half.h:// type Eigen::half (inheriting either from CUDA's or HIP's __half struct) with
source/include/Eigen/src/Core/arch/Default/Half.h:// to disk and the likes), but fast on GPUs.
source/include/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16) || defined(EIGEN_HAS_ARM64_FP16_SCALAR_ARITHMETIC)
source/include/Eigen/src/Core/arch/Default/Half.h:// When compiling with GPU support, the "__half_raw" base class as well as
source/include/Eigen/src/Core/arch/Default/Half.h:// some other routines are defined in the GPU compiler header files
source/include/Eigen/src/Core/arch/Default/Half.h:// (cuda_fp16.h, hip_fp16.h), and they are not tagged constexpr
source/include/Eigen/src/Core/arch/Default/Half.h:// GPU support. Hence the need to disable EIGEN_CONSTEXPR when building
source/include/Eigen/src/Core/arch/Default/Half.h:// Eigen with GPU support
source/include/Eigen/src/Core/arch/Default/Half.h:// This is required because of a quirk in the way TensorFlow GPU builds are done.
source/include/Eigen/src/Core/arch/Default/Half.h:// When compiling TensorFlow source code with GPU support, files that
source/include/Eigen/src/Core/arch/Default/Half.h://  * contain GPU kernels (i.e. *.cu.cc files) are compiled via hipcc
source/include/Eigen/src/Core/arch/Default/Half.h://  * do not contain GPU kernels ( i.e. *.cc files) are compiled via gcc (typically)
source/include/Eigen/src/Core/arch/Default/Half.h:#if !defined(EIGEN_HAS_GPU_FP16) || !defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/arch/Default/Half.h:// Make our own __half_raw definition that is similar to CUDA's.
source/include/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_GPU_FP16) && !defined(EIGEN_GPU_COMPILE_PHASE))
source/include/Eigen/src/Core/arch/Default/Half.h:#elif defined(EIGEN_HAS_CUDA_FP16)
source/include/Eigen/src/Core/arch/Default/Half.h:  #if EIGEN_CUDA_SDK_VER < 90000
source/include/Eigen/src/Core/arch/Default/Half.h:    // In CUDA < 9.0, __half is the equivalent of CUDA 9's __half_raw
source/include/Eigen/src/Core/arch/Default/Half.h:  #endif // defined(EIGEN_HAS_CUDA_FP16)
source/include/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16)
source/include/Eigen/src/Core/arch/Default/Half.h: #elif defined(EIGEN_HAS_CUDA_FP16)
source/include/Eigen/src/Core/arch/Default/Half.h:  #if EIGEN_CUDA_SDK_VER >= 90000
source/include/Eigen/src/Core/arch/Default/Half.h:#if !defined(EIGEN_HAS_GPU_FP16) || !defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/arch/Default/Half.h:  // * when compiling without GPU support enabled
source/include/Eigen/src/Core/arch/Default/Half.h:  // * during host compile phase when compiling with GPU support enabled
source/include/Eigen/src/Core/arch/Default/Half.h:#elif defined(EIGEN_HAS_CUDA_FP16)
source/include/Eigen/src/Core/arch/Default/Half.h:  // Note that EIGEN_CUDA_SDK_VER is set to 0 even when compiling with HIP, so
source/include/Eigen/src/Core/arch/Default/Half.h:  // (EIGEN_CUDA_SDK_VER < 90000) is true even for HIP!  So keeping this within
source/include/Eigen/src/Core/arch/Default/Half.h:  // #if defined(EIGEN_HAS_CUDA_FP16) is needed
source/include/Eigen/src/Core/arch/Default/Half.h:  #if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER < 90000
source/include/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16)
source/include/Eigen/src/Core/arch/Default/Half.h: #elif defined(EIGEN_HAS_CUDA_FP16)
source/include/Eigen/src/Core/arch/Default/Half.h:  #if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER >= 90000
source/include/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16) && !defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && \
source/include/Eigen/src/Core/arch/Default/Half.h:     EIGEN_CUDA_ARCH >= 530) ||                                  \
source/include/Eigen/src/Core/arch/Default/Half.h:// fp16 type since GPU halfs are rather different from native CPU halfs.
source/include/Eigen/src/Core/arch/Default/Half.h:// TODO: Rename to something like EIGEN_HAS_NATIVE_GPU_FP16
source/include/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER >= 90000
source/include/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER >= 90000
source/include/Eigen/src/Core/arch/Default/Half.h:// We need to distinguish ‘clang as the CUDA compiler’ from ‘clang as the host compiler,
source/include/Eigen/src/Core/arch/Default/Half.h:#if EIGEN_COMP_CLANG && defined(EIGEN_CUDACC)
source/include/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_HAS_NATIVE_FP16)
source/include/Eigen/src/Core/arch/Default/Half.h:// Definitions for CPUs and older HIP+CUDA, mostly working through conversion
source/include/Eigen/src/Core/arch/Default/Half.h:#if defined(__clang__) && defined(__CUDA__)
source/include/Eigen/src/Core/arch/Default/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
source/include/Eigen/src/Core/arch/Default/Half.h:  // Fortunately, since we need to disable EIGEN_CONSTEXPR for GPU anyway, we can get out
source/include/Eigen/src/Core/arch/Default/Half.h:  // of this catch22 by having separate bodies for GPU / non GPU
source/include/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16)
source/include/Eigen/src/Core/arch/Default/Half.h:  // HIP/CUDA/Default have a member 'x' of type uint16_t.
source/include/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
source/include/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
source/include/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
source/include/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530) || \
source/include/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && EIGEN_CUDA_SDK_VER >= 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
source/include/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530) || \
source/include/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300) || \
source/include/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300) || \
source/include/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
source/include/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
source/include/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16) || defined(EIGEN_HAS_ARM64_FP16_SCALAR_ARITHMETIC)
source/include/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
source/include/Eigen/src/Core/arch/Default/Half.h:// The __shfl* functions are only valid on HIP or _CUDA_ARCH_ >= 300.
source/include/Eigen/src/Core/arch/Default/Half.h://   CUDA defines them for (__CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__))
source/include/Eigen/src/Core/arch/Default/Half.h:// HIP and CUDA prior to SDK 9.0 define
source/include/Eigen/src/Core/arch/Default/Half.h:// CUDA since 9.0 deprecates those and instead defines
source/include/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_CUDACC) && (!defined(EIGEN_CUDA_ARCH) || EIGEN_CUDA_ARCH >= 300)) \
source/include/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && EIGEN_CUDA_SDK_VER >= 90000
source/include/Eigen/src/Core/arch/Default/Half.h:#else // HIP or CUDA SDK < 9.0
source/include/Eigen/src/Core/arch/Default/Half.h:#endif // HIP vs CUDA
source/include/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_CUDACC) && (!defined(EIGEN_CUDA_ARCH) || EIGEN_CUDA_ARCH >= 350)) \
source/include/Eigen/src/Core/arch/Default/BFloat16.h:// We need to distinguish ‘clang as the CUDA compiler’ from ‘clang as the host compiler,
source/include/Eigen/src/Core/arch/Default/BFloat16.h:#if EIGEN_COMP_CLANG && defined(EIGEN_CUDACC)
source/include/Eigen/src/Core/arch/Default/BFloat16.h:#if defined(EIGEN_HAS_CUDA_BF16) && defined(EIGEN_HAS_NATIVE_BF16)
source/include/Eigen/src/Core/arch/Default/BFloat16.h:#if EIGEN_COMP_CLANG && defined(EIGEN_CUDACC)
source/include/Eigen/src/Core/arch/Default/BFloat16.h:#if (defined(EIGEN_HAS_CUDA_BF16) && defined(EIGEN_HAS_HIP_BF16))
source/include/Eigen/src/Core/arch/Default/BFloat16.h:#if (defined(EIGEN_HAS_CUDA_BF16) && defined(EIGEN_HAS_HIP_BF16))
source/include/Eigen/src/Core/arch/Default/TypeCasting.h:    #if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
source/include/Eigen/src/Core/arch/Default/TypeCasting.h:    #if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
source/include/Eigen/src/Core/arch/Default/TypeCasting.h:    #if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \

```
