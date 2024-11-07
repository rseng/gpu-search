# https://github.com/PCMSolver/pcmsolver

```console
external/eigen3/include/eigen3/Eigen/Core:// Handle NVCC/CUDA/SYCL
external/eigen3/include/eigen3/Eigen/Core:#if defined(__CUDACC__) || defined(__SYCL_DEVICE_ONLY__)
external/eigen3/include/eigen3/Eigen/Core:  // Do not try asserts on CUDA and SYCL!
external/eigen3/include/eigen3/Eigen/Core:  // All functions callable from CUDA code must be qualified with __device__
external/eigen3/include/eigen3/Eigen/Core:  #ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/Core:    // Do not try to vectorize on CUDA and SYCL!
external/eigen3/include/eigen3/Eigen/Core:// When compiling CUDA device code with NVCC, pull in math functions from the
external/eigen3/include/eigen3/Eigen/Core:#if defined(__CUDA_ARCH__) && defined(__NVCC__)
external/eigen3/include/eigen3/Eigen/Core:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL)
external/eigen3/include/eigen3/Eigen/Core:#if defined __CUDACC__
external/eigen3/include/eigen3/Eigen/Core:  #define EIGEN_VECTORIZE_CUDA
external/eigen3/include/eigen3/Eigen/Core:  #if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
external/eigen3/include/eigen3/Eigen/Core:    #define EIGEN_HAS_CUDA_FP16
external/eigen3/include/eigen3/Eigen/Core:#if defined EIGEN_HAS_CUDA_FP16
external/eigen3/include/eigen3/Eigen/Core:  #include <cuda_fp16.h>
external/eigen3/include/eigen3/Eigen/Core:#include "src/Core/arch/CUDA/Half.h"
external/eigen3/include/eigen3/Eigen/Core:#include "src/Core/arch/CUDA/PacketMathHalf.h"
external/eigen3/include/eigen3/Eigen/Core:#include "src/Core/arch/CUDA/TypeCasting.h"
external/eigen3/include/eigen3/Eigen/Core:#if defined EIGEN_VECTORIZE_CUDA
external/eigen3/include/eigen3/Eigen/Core:  #include "src/Core/arch/CUDA/PacketMath.h"
external/eigen3/include/eigen3/Eigen/Core:  #include "src/Core/arch/CUDA/MathFunctions.h"
external/eigen3/include/eigen3/Eigen/Core:// on CUDA devices
external/eigen3/include/eigen3/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
external/eigen3/include/eigen3/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
external/eigen3/include/eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
external/eigen3/include/eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
external/eigen3/include/eigen3/Eigen/src/Core/util/Meta.h:  static float (max)() { return CUDART_MAX_NORMAL_F; }
external/eigen3/include/eigen3/Eigen/src/Core/util/Meta.h:  static float infinity() { return CUDART_INF_F; }
external/eigen3/include/eigen3/Eigen/src/Core/util/Meta.h:  static float quiet_NaN() { return CUDART_NAN_F; }
external/eigen3/include/eigen3/Eigen/src/Core/util/Meta.h:  static double infinity() { return CUDART_INF; }
external/eigen3/include/eigen3/Eigen/src/Core/util/Meta.h:  static double quiet_NaN() { return CUDART_NAN; }
external/eigen3/include/eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
external/eigen3/include/eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
external/eigen3/include/eigen3/Eigen/src/Core/util/Macros.h:  && ( !defined(__NVCC__) || !EIGEN_ARCH_ARM_OR_ARM64 || (defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000) )
external/eigen3/include/eigen3/Eigen/src/Core/util/Macros.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/util/Macros.h:#if EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && defined(__CUDACC_VER__) && (EIGEN_COMP_CLANG || __CUDACC_VER__ >= 70500))
external/eigen3/include/eigen3/Eigen/src/Core/util/Macros.h:#if (defined __CUDACC__)
external/eigen3/include/eigen3/Eigen/src/Core/util/Macros.h:#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC < 1900 ||  __CUDACC_VER__) // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
external/eigen3/include/eigen3/Eigen/src/Core/util/Macros.h:#  ifdef __CUDA_ARCH__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifndef __CUDA_ARCH__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#if (!defined(__CUDACC__)) && EIGEN_FAST_MATH
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
external/eigen3/include/eigen3/Eigen/src/Core/MatrixBase.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
external/eigen3/include/eigen3/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
external/eigen3/include/eigen3/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
external/eigen3/include/eigen3/Eigen/src/Core/GeneralProduct.h:#ifndef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/GeneralProduct.h:#endif // __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/GenericPacketMath.h:#ifdef __CUDA_ARCH__
external/eigen3/include/eigen3/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
external/eigen3/include/eigen3/Eigen/src/Core/GenericPacketMath.h:#ifndef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
external/eigen3/include/eigen3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
external/eigen3/include/eigen3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
external/eigen3/include/eigen3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#ifndef EIGEN_PACKET_MATH_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#define EIGEN_PACKET_MATH_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#endif // EIGEN_PACKET_MATH_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// type Eigen::half (inheriting from CUDA's __half struct) with
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// to disk and the likes), but fast on GPUs.
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#ifndef EIGEN_HALF_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#define EIGEN_HALF_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if !defined(EIGEN_HAS_CUDA_FP16)
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Make our own __half definition that is similar to CUDA's.
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:  #if !defined(EIGEN_HAS_CUDA_FP16)
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Definitions for CPUs and older CUDA, mostly working through conversion
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__)
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#endif // EIGEN_HALF_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#ifndef EIGEN_PACKET_MATH_HALF_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#define EIGEN_PACKET_MATH_HALF_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 530
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#endif // EIGEN_PACKET_MATH_HALF_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#define EIGEN_TYPE_CASTING_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:// building for CUDA to avoid non-constexpr methods.
external/eigen3/include/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#endif // EIGEN_COMPLEX_CUDA_H
external/eigen3/include/eigen3/Eigen/src/SVD/BDCSVD.h:#ifndef __CUDACC__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/Tensor:#ifdef EIGEN_USE_GPU
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/Tensor:#include <cuda_runtime.h>
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorDeviceCuda.h"
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorReductionCuda.h"
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorContractionCuda.h"
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:#ifdef __CUDA_ARCH__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:// GPU: the evaluation of the expression is offloaded to a GPU.
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#if defined(EIGEN_USE_GPU)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:class TensorExecutor<Expression, GpuDevice, Vectorizable> {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:  static void run(const Expression& expr, const GpuDevice& device);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#if defined(__CUDACC__)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:inline void TensorExecutor<Expression, GpuDevice, Vectorizable>::run(
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const Expression& expr, const GpuDevice& device) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:  TensorEvaluator<Expression, GpuDevice> evaluator(expr, device);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const int block_size = device.maxCudaThreadsPerBlock();
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    LAUNCH_CUDA_KERNEL(
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:        (EigenMetaKernel<TensorEvaluator<Expression, GpuDevice>, Index>),
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#endif  // __CUDACC__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#endif  // EIGEN_USE_GPU
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef __CUDA_ARCH__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef __CUDA_ARCH__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__CUDA_ARCH__)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__CUDA_ARCH__)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__SIZEOF_INT128__) && !defined(__CUDA_ARCH__)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef __CUDA_ARCH__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device, return the amount of shared memory available.
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    return __CUDA_ARCH__ / 100;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDevice.h:  * on the specified computing 'device' (GPU, thread pool, ...)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDevice.h:  *    C.device(EIGEN_GPU) = A + B;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:// Full reducers for GPU, don't vectorize for now
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:// Reducer function that enables multiple cuda thread to safely accumulate at the same
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:// attempts to update it with the new value. If in the meantime another cuda thread
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((FullReductionKernel<block_size, num_per_thread, Self, Op, Index>),
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitFullReduxKernelHalfFloat<Self, Op, Index>),
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((FullReductionKernelHalfFloat<block_size, num_per_thread, Self, Op, Index>),
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionCleanupKernelHalfFloat<Op>),
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:struct FullReducer<Self, Op, GpuDevice, Vectorizable> {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    FullReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static EIGEN_DEVICE_FUNC bool run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index, typename Self::Index) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    assert(false && "Should only be called to reduce doubles, floats and half floats on a gpu device");
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      const int max_blocks = device.getNumCudaMultiProcessors() *
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / 1024;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitKernel<OutputType, Index>),
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((InnerReductionKernel<num_per_thread, Self, Op, Index>),
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index, typename Self::Index) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      const int max_blocks = device.getNumCudaMultiProcessors() *
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / 1024;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitKernelHalfFloat<Self, Op, Index>),
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((InnerReductionKernelHalfFloat<num_per_thread, Self, Op, Index>),
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:struct InnerReducer<Self, Op, GpuDevice> {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    return InnerReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs_to_reduce, num_preserved_vals);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:struct OuterReducer<Self, Op, GpuDevice> {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    assert(false && "Should only be called to reduce doubles or floats on a gpu device");
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, float* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      const int max_blocks = device.getNumCudaMultiProcessors() *
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                             device.maxCudaThreadsPerMultiProcessor() / 1024;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitKernel<float, Index>),
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((OuterReductionKernel<num_per_thread, Self, Op, Index>),
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#endif // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:// It is very expensive to start the memcpy kernel on GPU: we therefore only
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:#ifdef EIGEN_USE_GPU
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:template <typename Index> struct MemcpyTriggerForSlicing<Index, GpuDevice>  {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:  EIGEN_DEVICE_FUNC MemcpyTriggerForSlicing(const GpuDevice&) { }
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:// For CUDA packet types when using a GpuDevice
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__) && defined(EIGEN_HAS_CUDA_FP16)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:struct PacketType<half, GpuDevice> {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:// Can't use std::pairs on cuda devices
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorSycl.h:// this is used to change the address space type in tensor map for GPU
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorFFT.h:  * Improve the performance on GPU
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:struct GpuDevice;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:struct IsVectorizable<GpuDevice, Expression> {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:  static const bool value = TensorEvaluator<Expression, GpuDevice>::PacketAccess &&
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:                            TensorEvaluator<Expression, GpuDevice>::IsAligned;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, GpuDevice> :
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    public TensorContractionEvaluatorBase<TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, GpuDevice> > {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:  typedef GpuDevice Device;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:  typedef typename PacketType<CoeffReturnType, GpuDevice>::type PacketReturnType;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    static void Run(const LhsMapper& lhs, const RhsMapper& rhs, const OutputMapper& output, Index m, Index n, Index k, const GpuDevice& device) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    LAUNCH_CUDA_KERNEL((EigenContractionKernel<Scalar, Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    static void Run(const LhsMapper& lhs, const RhsMapper& rhs, const OutputMapper& output, Index m, Index n, Index k, const GpuDevice& device) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:        LAUNCH_CUDA_KERNEL((EigenFloatContractionKernel16x16<Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:        LAUNCH_CUDA_KERNEL((EigenFloatContractionKernel<Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    setCudaSharedMemConfig(cudaSharedMemBankSizeEightByte);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#endif // EIGEN_USE_GPU and __CUDACC__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMacros.h:#ifndef __CUDACC__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionSycl.h:    /// This one is used to collect all the reduced value of shared memory as we dont have global barrier on GPU. Once it is saved we can
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionSycl.h:    /// This one is used to collect all the reduced value of shared memory as we dont have global barrier on GPU. Once it is saved we can
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#ifdef EIGEN_HAS_CUDA_FP16
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:        ((RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) ||
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:         !RunningOnGPU))) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    else if (RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    if ((RunningOnSycl || RunningFullReduction || RunningOnGPU) && m_result) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    if (RunningOnGPU && m_result) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#ifdef EIGEN_HAS_CUDA_FP16
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:  static const bool RunningOnGPU = internal::is_same<Device, Eigen::GpuDevice>::value;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:static const bool RunningOnGPU = false;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:  static const bool RunningOnGPU = false;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#if defined(EIGEN_USE_GPU) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static const int kCudaScratchSize = 1024;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:// This defines an interface that GPUDevice can take to use
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:// CUDA streams underneath.
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  virtual const cudaStream_t& stream() const = 0;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  virtual const cudaDeviceProp& deviceProperties() const = 0;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static cudaDeviceProp* m_deviceProperties;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaError_t status = cudaGetDeviceCount(&num_devices);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      if (status != cudaSuccess) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        std::cerr << "Failed to get the number of CUDA devices: "
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:                  << cudaGetErrorString(status)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        assert(status == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      m_deviceProperties = new cudaDeviceProp[num_devices];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        status = cudaGetDeviceProperties(&m_deviceProperties[i], i);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        if (status != cudaSuccess) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:          std::cerr << "Failed to initialize CUDA device #"
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:                    << cudaGetErrorString(status)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:          assert(status == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static const cudaStream_t default_stream = cudaStreamDefault;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:class CudaStreamDevice : public StreamInterface {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  CudaStreamDevice() : stream_(&default_stream), scratch_(NULL), semaphore_(NULL) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaGetDevice(&device_);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  CudaStreamDevice(int device) : stream_(&default_stream), device_(device), scratch_(NULL), semaphore_(NULL) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  // assumes that the stream is associated to the current gpu device.
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  CudaStreamDevice(const cudaStream_t* stream, int device = -1)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaGetDevice(&device_);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaError_t err = cudaGetDeviceCount(&num_devices);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      assert(err == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  virtual ~CudaStreamDevice() {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  const cudaStream_t& stream() const { return *stream_; }
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  const cudaDeviceProp& deviceProperties() const {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaSetDevice(device_);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    err = cudaMalloc(&result, num_bytes);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaSetDevice(device_);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    err = cudaFree(buffer);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      scratch_ = allocate(kCudaScratchSize + sizeof(unsigned int));
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      char* scratch = static_cast<char*>(scratchpad()) + kCudaScratchSize;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaError_t err = cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      assert(err == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  const cudaStream_t* stream_;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:struct GpuDevice {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  explicit GpuDevice(const StreamInterface* stream) : stream_(stream), max_blocks_(INT_MAX) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  explicit GpuDevice(const StreamInterface* stream, int num_blocks) : stream_(stream), max_blocks_(num_blocks) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE const cudaStream_t& stream() const {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifndef __CUDA_ARCH__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice,
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err =
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream_->stream());
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err =
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream_->stream());
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifndef __CUDA_ARCH__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaMemsetAsync(buffer, c, n, stream_->stream());
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    // there is no l3 cache on cuda devices.
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaStreamSynchronize(stream_->stream());
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    if (err != cudaSuccess) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      std::cerr << "Error detected in CUDA stream: "
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:                << cudaGetErrorString(err)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      assert(err == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE int getNumCudaMultiProcessors() const {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE int maxCudaThreadsPerBlock() const {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE int maxCudaThreadsPerMultiProcessor() const {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  // This function checks if the CUDA runtime recorded an error for the
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t error = cudaStreamQuery(stream_->stream());
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    return (error == cudaSuccess) || (error == cudaErrorNotReady);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#define LAUNCH_CUDA_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)             \
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  assert(cudaGetLastError() == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifdef __CUDACC__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static EIGEN_DEVICE_FUNC inline void setCudaSharedMemConfig(cudaSharedMemConfig config) {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifndef __CUDA_ARCH__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  cudaError_t status = cudaDeviceSetSharedMemConfig(config);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  assert(status == cudaSuccess);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    array<Index, NumDims> cudaInputDimensions;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    array<Index, NumDims> cudaOutputDimensions;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      cudaInputDimensions[index] = input_dims[indices[i]];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      cudaOutputDimensions[index] = dimensions[indices[i]];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        cudaInputDimensions[written] = input_dims[i];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        cudaOutputDimensions[written] = dimensions[i];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] =
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaInputStrides[i - 1] * cudaInputDimensions[i - 1];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] =
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaOutputStrides[i - 1] * cudaOutputDimensions[i - 1];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] = 1;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] = 1;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] =
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaInputStrides[i + 1] * cudaInputDimensions[i + 1];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] =
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaOutputStrides[i + 1] * cudaOutputDimensions[i + 1];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] = 1;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] = 1;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputPlaneToTensorInputOffset(Index p) const {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaInputStrides[d];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaInputStrides[d];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaInputStrides[d];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaInputStrides[d];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputPlaneToTensorOutputOffset(Index p) const {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaOutputStrides[d];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaOutputStrides[d];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaOutputStrides[d];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaOutputStrides[d];
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputKernelToTensorInputOffset(Index i) const {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputKernelToTensorOutputOffset(Index i) const {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputKernelToTensorInputOffset(Index i, Index j) const {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputKernelToTensorOutputOffset(Index i, Index j) const {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputKernelToTensorInputOffset(Index i, Index j, Index k) const {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputKernelToTensorOutputOffset(Index i, Index j, Index k) const {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  array<Index, NumDims> m_cudaInputStrides;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  array<Index, NumDims> m_cudaOutputStrides;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:// Use an optimized implementation of the evaluation code for GPUs whenever possible.
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const int tensor_index = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i+first_x);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const int tensor_index = plane_output_offset + indexMapper.mapCudaOutputKernelToTensorOutputOffset(i+first_x);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const int tensor_index = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i+first_x, j+first_y);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const int tensor_index = plane_output_offset + indexMapper.mapCudaOutputKernelToTensorOutputOffset(i+first_x, j+first_y);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          const int tensor_index = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i+first_x, j+first_y, k+first_z);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          const int tensor_index = plane_output_offset + indexMapper.mapCudaOutputKernelToTensorOutputOffset(i+first_x, j+first_y, k+first_z);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:struct TensorEvaluator<const TensorConvolutionOp<Indices, InputArgType, KernelArgType>, GpuDevice>
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  static const int NumDims =  internal::array_size<typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions>::value;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  typedef typename TensorEvaluator<KernelArgType, GpuDevice>::Dimensions KernelDimensions;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    IsAligned = TensorEvaluator<InputArgType, GpuDevice>::IsAligned & TensorEvaluator<KernelArgType, GpuDevice>::IsAligned,
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    Layout = TensorEvaluator<InputArgType, GpuDevice>::Layout,
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const GpuDevice& device)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<InputArgType, GpuDevice>::Layout) == static_cast<int>(TensorEvaluator<KernelArgType, GpuDevice>::Layout)), YOU_MADE_A_PROGRAMMING_MISTAKE);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions& input_dims = m_inputImpl.dimensions();
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const typename TensorEvaluator<KernelArgType, GpuDevice>::Dimensions& kernel_dims = m_kernelImpl.dimensions();
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  typedef typename PacketType<CoeffReturnType, GpuDevice>::type PacketReturnType;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const bool PacketAccess = internal::IsVectorizable<GpuDevice, KernelArgType>::value;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      internal::TensorExecutor<const EvalTo, GpuDevice, PacketAccess>::run(evalToTmp, m_device);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    typedef typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions InputDims;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int maxThreadsPerBlock = m_device.maxCudaThreadsPerBlock();
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int maxBlocksPerProcessor = m_device.maxCudaThreadsPerMultiProcessor() / maxThreadsPerBlock;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int numMultiProcessors = m_device.getNumCudaMultiProcessors();
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, 4, data);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, 7, data);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, kernel_size, data);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4, 7>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 4, 7, data);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 4, kernel_size_y, data);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7, 4>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 7, 4, data);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 7, kernel_size_y, data);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, Dynamic, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, kernel_size_x, kernel_size_y, data);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        LAUNCH_CUDA_KERNEL((EigenConvolutionKernel3D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, numZ, maxZ, kernel_size_x, kernel_size_y, kernel_size_z, data);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  TensorEvaluator<InputArgType, GpuDevice> m_inputImpl;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  TensorEvaluator<KernelArgType, GpuDevice> m_kernelImpl;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  const GpuDevice& m_device;
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:// GPU implementation of scan
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:struct ScanLauncher<Self, Reducer, GpuDevice> {
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:     LAUNCH_CUDA_KERNEL((ScanKernel<Self, Reducer>), num_blocks, block_size, 0, self.device(), self, total_size, data);
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:#endif  // EIGEN_USE_GPU && __CUDACC__
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:// Use the texture cache on CUDA devices whenever possible
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/util/EmulateArray.h:// Moreover, CUDA doesn't support the STL containers, so we use our own instead.
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/util/EmulateArray.h:#if (__cplusplus <= 199711L && EIGEN_COMP_MSVC < 1900) || defined(__CUDACC__) || defined(EIGEN_AVOID_STL_ARRAY)
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/util/EmulateArray.h:// The compiler supports c++11, and we're not targetting cuda: use std::array as Eigen::array
external/eigen3/include/eigen3/unsupported/Eigen/CXX11/src/util/MaxSizeVector.h:  * std::vector is not an option (e.g. on GPU or when compiling using
external/eigen3/include/eigen3/unsupported/Eigen/SpecialFunctions:#if defined EIGEN_VECTORIZE_CUDA
external/eigen3/include/eigen3/unsupported/Eigen/SpecialFunctions:  #include "src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h"
external/eigen3/include/eigen3/unsupported/Eigen/OpenGLSupport:#ifdef GL_ARB_gpu_shader_fp64
external/eigen3/include/eigen3/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:#if !defined(__CUDA_ARCH__) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__)
external/eigen3/include/eigen3/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:#if !defined(__CUDA_ARCH__) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__)
external/eigen3/include/eigen3/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:    if (x == inf) return zero;  // std::isinf crashes on CUDA
external/eigen3/include/eigen3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#ifndef EIGEN_CUDA_SPECIALFUNCTIONS_H
external/eigen3/include/eigen3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#define EIGEN_CUDA_SPECIALFUNCTIONS_H
external/eigen3/include/eigen3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
external/eigen3/include/eigen3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
external/eigen3/include/eigen3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#endif // EIGEN_CUDA_SPECIALFUNCTIONS_H
doc/cloc_tools/cloc.pl:                             Pascal/PHP; Lisp/OpenCL; Lisp/Julia; Perl/Prolog)
doc/cloc_tools/cloc.pl:                $language eq "Lisp/OpenCL"                      or
doc/cloc_tools/cloc.pl:    if (!$language or $language =~ /^(Lisp|OpenCL)$/i) {
doc/cloc_tools/cloc.pl:        push @{$extensions{'OpenCL'}}, "cl";
doc/cloc_tools/cloc.pl:        delete $extensions{'Lisp/OpenCL'};
doc/cloc_tools/cloc.pl:            } elsif ($Language_by_Extension{$extension} eq 'Lisp/OpenCL') {
doc/cloc_tools/cloc.pl:                return Lisp_or_OpenCL($full_file, $rh_Err, $raa_errors);
doc/cloc_tools/cloc.pl:            'cl'          => 'Lisp/OpenCL'           ,
doc/cloc_tools/cloc.pl:            'cu'          => 'CUDA'                  ,
doc/cloc_tools/cloc.pl:            'cuh'         => 'CUDA'                  , # CUDA header file
doc/cloc_tools/cloc.pl:    'CUDA'               => [
doc/cloc_tools/cloc.pl:    'Lisp/OpenCL'        => [ [ 'die' ,          ], ], # never called
doc/cloc_tools/cloc.pl:    'OpenCL'             => [
doc/cloc_tools/cloc.pl:    'CUDA'                         =>   1.00,
doc/cloc_tools/cloc.pl:    'OpenCL'                       => 1.50,
doc/cloc_tools/cloc.pl:        "Lisp/OpenCL"                       => 1,
doc/cloc_tools/cloc.pl:sub Lisp_or_OpenCL {                         # {{{1
doc/cloc_tools/cloc.pl:    print "-> Lisp_or_OpenCL\n" if $opt_v > 2;
doc/cloc_tools/cloc.pl:        $lang = "OpenCL";
doc/cloc_tools/cloc.pl:    print "<- Lisp_or_OpenCL\n" if $opt_v > 2;

```
