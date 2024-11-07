# https://github.com/COMBINE-lab/salmon

```console
tests/catch.hpp:#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__) && !defined(__LCC__)
tests/catch.hpp:#  if !defined(__ibmxl__) && !defined(__CUDACC__)
tests/catch.hpp:        // Code accompanying the article "Approximating the erfinv function" in GPU Computing Gems, Volume 2
include/parallel_hashmap/phmap_config.h:        (defined(__CUDACC__) && __CUDACC_VER_MAJOR__ >= 9) ||                \
include/parallel_hashmap/phmap_config.h:        (defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__))
include/parallel_hashmap/phmap_config.h:    #elif defined(__CUDACC__)
include/parallel_hashmap/phmap_config.h:        #if __CUDACC_VER__ >= 70000
include/parallel_hashmap/phmap_config.h:        #endif  // __CUDACC_VER__ >= 70000
include/parallel_hashmap/phmap_config.h:    #endif  // defined(__CUDACC__)
include/eigen3/Eigen/Core:// Handle NVCC/CUDA/SYCL
include/eigen3/Eigen/Core:#if defined(__CUDACC__) || defined(__SYCL_DEVICE_ONLY__)
include/eigen3/Eigen/Core:  // Do not try asserts on CUDA and SYCL!
include/eigen3/Eigen/Core:  // All functions callable from CUDA code must be qualified with __device__
include/eigen3/Eigen/Core:  #ifdef __CUDACC__
include/eigen3/Eigen/Core:    // Do not try to vectorize on CUDA and SYCL!
include/eigen3/Eigen/Core:// When compiling CUDA device code with NVCC, pull in math functions from the
include/eigen3/Eigen/Core:#if defined(__CUDA_ARCH__) && defined(__NVCC__)
include/eigen3/Eigen/Core:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL)
include/eigen3/Eigen/Core:#if defined __CUDACC__
include/eigen3/Eigen/Core:  #define EIGEN_VECTORIZE_CUDA
include/eigen3/Eigen/Core:  #if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
include/eigen3/Eigen/Core:    #define EIGEN_HAS_CUDA_FP16
include/eigen3/Eigen/Core:#if defined EIGEN_HAS_CUDA_FP16
include/eigen3/Eigen/Core:  #include <cuda_fp16.h>
include/eigen3/Eigen/Core:#include "src/Core/arch/CUDA/Half.h"
include/eigen3/Eigen/Core:#include "src/Core/arch/CUDA/PacketMathHalf.h"
include/eigen3/Eigen/Core:#include "src/Core/arch/CUDA/TypeCasting.h"
include/eigen3/Eigen/Core:#if defined EIGEN_VECTORIZE_CUDA
include/eigen3/Eigen/Core:  #include "src/Core/arch/CUDA/PacketMath.h"
include/eigen3/Eigen/Core:  #include "src/Core/arch/CUDA/MathFunctions.h"
include/eigen3/Eigen/Core:// on CUDA devices
include/eigen3/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
include/eigen3/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
include/eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
include/eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
include/eigen3/Eigen/src/Core/util/Meta.h:  static float (max)() { return CUDART_MAX_NORMAL_F; }
include/eigen3/Eigen/src/Core/util/Meta.h:  static float infinity() { return CUDART_INF_F; }
include/eigen3/Eigen/src/Core/util/Meta.h:  static float quiet_NaN() { return CUDART_NAN_F; }
include/eigen3/Eigen/src/Core/util/Meta.h:  static double infinity() { return CUDART_INF; }
include/eigen3/Eigen/src/Core/util/Meta.h:  static double quiet_NaN() { return CUDART_NAN; }
include/eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
include/eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
include/eigen3/Eigen/src/Core/util/Macros.h:  && ( !defined(__NVCC__) || !EIGEN_ARCH_ARM_OR_ARM64 || (defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000) )
include/eigen3/Eigen/src/Core/util/Macros.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/util/Macros.h:#if EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && defined(__CUDACC_VER__) && (EIGEN_COMP_CLANG || __CUDACC_VER__ >= 70500))
include/eigen3/Eigen/src/Core/util/Macros.h:#if (defined __CUDACC__)
include/eigen3/Eigen/src/Core/util/Macros.h:#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC < 1900 ||  defined(__CUDACC_VER__)) // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
include/eigen3/Eigen/src/Core/util/Macros.h:#  ifdef __CUDA_ARCH__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
include/eigen3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
include/eigen3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
include/eigen3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifndef __CUDA_ARCH__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#if (!defined(__CUDACC__)) && EIGEN_FAST_MATH
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
include/eigen3/Eigen/src/Core/MatrixBase.h:#ifdef __CUDACC__
include/eigen3/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
include/eigen3/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
include/eigen3/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
include/eigen3/Eigen/src/Core/GeneralProduct.h:#ifndef __CUDACC__
include/eigen3/Eigen/src/Core/GeneralProduct.h:#endif // __CUDACC__
include/eigen3/Eigen/src/Core/GenericPacketMath.h:#ifdef __CUDA_ARCH__
include/eigen3/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
include/eigen3/Eigen/src/Core/GenericPacketMath.h:#ifndef __CUDACC__
include/eigen3/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
include/eigen3/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
include/eigen3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
include/eigen3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
include/eigen3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#ifndef EIGEN_PACKET_MATH_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#define EIGEN_PACKET_MATH_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#endif // EIGEN_PACKET_MATH_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// type Eigen::half (inheriting from CUDA's __half struct) with
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// to disk and the likes), but fast on GPUs.
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#ifndef EIGEN_HALF_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#define EIGEN_HALF_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if !defined(EIGEN_HAS_CUDA_FP16)
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Make our own __half definition that is similar to CUDA's.
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:  #if !defined(EIGEN_HAS_CUDA_FP16)
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Definitions for CPUs and older CUDA, mostly working through conversion
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__)
include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:#endif // EIGEN_HALF_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#ifndef EIGEN_PACKET_MATH_HALF_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#define EIGEN_PACKET_MATH_HALF_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 530
include/eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#endif // EIGEN_PACKET_MATH_HALF_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
include/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
include/eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#define EIGEN_TYPE_CASTING_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
include/eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
include/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
include/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:// building for CUDA to avoid non-constexpr methods.
include/eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#endif // EIGEN_COMPLEX_CUDA_H
include/eigen3/Eigen/src/SVD/BDCSVD.h:#ifndef __CUDACC__
include/eigen3/unsupported/Eigen/CXX11/Tensor:#ifdef EIGEN_USE_GPU
include/eigen3/unsupported/Eigen/CXX11/Tensor:#include <cuda_runtime.h>
include/eigen3/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorDeviceCuda.h"
include/eigen3/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorReductionCuda.h"
include/eigen3/unsupported/Eigen/CXX11/Tensor:#include "src/Tensor/TensorContractionCuda.h"
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h:#ifdef __CUDA_ARCH__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:// GPU: the evaluation of the expression is offloaded to a GPU.
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#if defined(EIGEN_USE_GPU)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:class TensorExecutor<Expression, GpuDevice, Vectorizable> {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:  static void run(const Expression& expr, const GpuDevice& device);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#if defined(__CUDACC__)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:inline void TensorExecutor<Expression, GpuDevice, Vectorizable>::run(
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const Expression& expr, const GpuDevice& device) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:  TensorEvaluator<Expression, GpuDevice> evaluator(expr, device);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const int block_size = device.maxCudaThreadsPerBlock();
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:    LAUNCH_CUDA_KERNEL(
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:        (EigenMetaKernel<TensorEvaluator<Expression, GpuDevice>, Index>),
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#endif  // __CUDACC__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h:#endif  // EIGEN_USE_GPU
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef __CUDA_ARCH__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef __CUDA_ARCH__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__CUDA_ARCH__)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__CUDA_ARCH__)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#if defined(__SIZEOF_INT128__) && !defined(__CUDA_ARCH__)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h:#ifdef __CUDA_ARCH__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device, return the amount of shared memory available.
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:#ifndef __CUDA_ARCH__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    // Running on a CUDA device
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h:    return __CUDA_ARCH__ / 100;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDevice.h:  * on the specified computing 'device' (GPU, thread pool, ...)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDevice.h:  *    C.device(EIGEN_GPU) = A + B;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:// Full reducers for GPU, don't vectorize for now
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:// Reducer function that enables multiple cuda thread to safely accumulate at the same
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:// attempts to update it with the new value. If in the meantime another cuda thread
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((FullReductionKernel<block_size, num_per_thread, Self, Op, Index>),
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitFullReduxKernelHalfFloat<Self, Op, Index>),
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((FullReductionKernelHalfFloat<block_size, num_per_thread, Self, Op, Index>),
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionCleanupKernelHalfFloat<Op>),
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:struct FullReducer<Self, Op, GpuDevice, Vectorizable> {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    FullReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#if __CUDA_ARCH__ >= 300
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static EIGEN_DEVICE_FUNC bool run(const Self&, Op&, const GpuDevice&, OutputType*, typename Self::Index, typename Self::Index) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    assert(false && "Should only be called to reduce doubles, floats and half floats on a gpu device");
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      const int max_blocks = device.getNumCudaMultiProcessors() *
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / 1024;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitKernel<OutputType, Index>),
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((InnerReductionKernel<num_per_thread, Self, Op, Index>),
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self&, Op&, const GpuDevice&, half*, typename Self::Index, typename Self::Index) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, half* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      const int max_blocks = device.getNumCudaMultiProcessors() *
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / 1024;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitKernelHalfFloat<Self, Op, Index>),
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((InnerReductionKernelHalfFloat<num_per_thread, Self, Op, Index>),
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:struct InnerReducer<Self, Op, GpuDevice> {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#ifdef EIGEN_HAS_CUDA_FP16
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:       (internal::is_same<typename Self::CoeffReturnType, Eigen::half>::value && reducer_traits<Op, GpuDevice>::PacketAccess));
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    return InnerReductionLauncher<Self, Op, OutputType, reducer_traits<Op, GpuDevice>::PacketAccess>::run(self, reducer, device, output, num_coeffs_to_reduce, num_preserved_vals);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:struct OuterReducer<Self, Op, GpuDevice> {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  // Unfortunately nvidia doesn't support well exotic types such as complex,
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    assert(false && "Should only be called to reduce doubles or floats on a gpu device");
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:  static bool run(const Self& self, Op& reducer, const GpuDevice& device, float* output, typename Self::Index num_coeffs_to_reduce, typename Self::Index num_preserved_vals) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    const int max_blocks = device.getNumCudaMultiProcessors() *
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                           device.maxCudaThreadsPerMultiProcessor() / block_size;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      const int max_blocks = device.getNumCudaMultiProcessors() *
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:                             device.maxCudaThreadsPerMultiProcessor() / 1024;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:      LAUNCH_CUDA_KERNEL((ReductionInitKernel<float, Index>),
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:    LAUNCH_CUDA_KERNEL((OuterReductionKernel<num_per_thread, Self, Op, Index>),
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionCuda.h:#endif // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:// It is very expensive to start the memcpy kernel on GPU: we therefore only
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:#ifdef EIGEN_USE_GPU
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:template <typename Index> struct MemcpyTriggerForSlicing<Index, GpuDevice>  {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h:  EIGEN_DEVICE_FUNC MemcpyTriggerForSlicing(const GpuDevice&) { }
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:// For CUDA packet types when using a GpuDevice
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__) && defined(EIGEN_HAS_CUDA_FP16)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:struct PacketType<half, GpuDevice> {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h:// Can't use std::pairs on cuda devices
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorSycl.h:// this is used to change the address space type in tensor map for GPU
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorFFT.h:  * Improve the performance on GPU
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:struct GpuDevice;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:struct IsVectorizable<GpuDevice, Expression> {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:  static const bool value = TensorEvaluator<Expression, GpuDevice>::PacketAccess &&
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h:                            TensorEvaluator<Expression, GpuDevice>::IsAligned;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, GpuDevice> :
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    public TensorContractionEvaluatorBase<TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, GpuDevice> > {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:  typedef GpuDevice Device;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:  typedef typename PacketType<CoeffReturnType, GpuDevice>::type PacketReturnType;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    static void Run(const LhsMapper& lhs, const RhsMapper& rhs, const OutputMapper& output, Index m, Index n, Index k, const GpuDevice& device) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    LAUNCH_CUDA_KERNEL((EigenContractionKernel<Scalar, Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    static void Run(const LhsMapper& lhs, const RhsMapper& rhs, const OutputMapper& output, Index m, Index n, Index k, const GpuDevice& device) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:        LAUNCH_CUDA_KERNEL((EigenFloatContractionKernel16x16<Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:        LAUNCH_CUDA_KERNEL((EigenFloatContractionKernel<Index, LhsMapper, RhsMapper, OutputMapper>), num_blocks, block_size, 0, device, lhs, rhs, output, m, n, k);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:    setCudaSharedMemConfig(cudaSharedMemBankSizeEightByte);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#endif // EIGEN_USE_GPU and __CUDACC__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorContractionCuda.h:#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_CUDA_H
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorMacros.h:#ifndef __CUDACC__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionSycl.h:    /// This one is used to collect all the reduced value of shared memory as we dont have global barrier on GPU. Once it is saved we can
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionSycl.h:    /// This one is used to collect all the reduced value of shared memory as we dont have global barrier on GPU. Once it is saved we can
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#ifdef EIGEN_HAS_CUDA_FP16
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:        ((RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) ||
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:         !RunningOnGPU))) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    else if (RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    if ((RunningOnSycl || RunningFullReduction || RunningOnGPU) && m_result) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:    if (RunningOnGPU && m_result) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#ifdef EIGEN_HAS_CUDA_FP16
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:  static const bool RunningOnGPU = internal::is_same<Device, Eigen::GpuDevice>::value;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:static const bool RunningOnGPU = false;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h:  static const bool RunningOnGPU = false;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#if defined(EIGEN_USE_GPU) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static const int kCudaScratchSize = 1024;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:// This defines an interface that GPUDevice can take to use
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:// CUDA streams underneath.
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  virtual const cudaStream_t& stream() const = 0;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  virtual const cudaDeviceProp& deviceProperties() const = 0;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static cudaDeviceProp* m_deviceProperties;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaError_t status = cudaGetDeviceCount(&num_devices);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      if (status != cudaSuccess) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        std::cerr << "Failed to get the number of CUDA devices: "
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:                  << cudaGetErrorString(status)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        assert(status == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      m_deviceProperties = new cudaDeviceProp[num_devices];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        status = cudaGetDeviceProperties(&m_deviceProperties[i], i);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        if (status != cudaSuccess) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:          std::cerr << "Failed to initialize CUDA device #"
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:                    << cudaGetErrorString(status)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:          assert(status == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static const cudaStream_t default_stream = cudaStreamDefault;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:class CudaStreamDevice : public StreamInterface {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  CudaStreamDevice() : stream_(&default_stream), scratch_(NULL), semaphore_(NULL) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaGetDevice(&device_);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  CudaStreamDevice(int device) : stream_(&default_stream), device_(device), scratch_(NULL), semaphore_(NULL) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  // assumes that the stream is associated to the current gpu device.
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  CudaStreamDevice(const cudaStream_t* stream, int device = -1)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaGetDevice(&device_);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaError_t err = cudaGetDeviceCount(&num_devices);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      assert(err == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  virtual ~CudaStreamDevice() {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  const cudaStream_t& stream() const { return *stream_; }
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  const cudaDeviceProp& deviceProperties() const {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaSetDevice(device_);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    err = cudaMalloc(&result, num_bytes);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaSetDevice(device_);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    err = cudaFree(buffer);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      scratch_ = allocate(kCudaScratchSize + sizeof(unsigned int));
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      char* scratch = static_cast<char*>(scratchpad()) + kCudaScratchSize;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      cudaError_t err = cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      assert(err == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  const cudaStream_t* stream_;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:struct GpuDevice {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  explicit GpuDevice(const StreamInterface* stream) : stream_(stream), max_blocks_(INT_MAX) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  explicit GpuDevice(const StreamInterface* stream, int num_blocks) : stream_(stream), max_blocks_(num_blocks) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE const cudaStream_t& stream() const {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifndef __CUDA_ARCH__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice,
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err =
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream_->stream());
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err =
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:        cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream_->stream());
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifndef __CUDA_ARCH__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaMemsetAsync(buffer, c, n, stream_->stream());
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    assert(err == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    // there is no l3 cache on cuda devices.
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t err = cudaStreamSynchronize(stream_->stream());
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    if (err != cudaSuccess) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      std::cerr << "Error detected in CUDA stream: "
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:                << cudaGetErrorString(err)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:      assert(err == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE int getNumCudaMultiProcessors() const {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE int maxCudaThreadsPerBlock() const {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  EIGEN_STRONG_INLINE int maxCudaThreadsPerMultiProcessor() const {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  // This function checks if the CUDA runtime recorded an error for the
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifdef __CUDACC__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    cudaError_t error = cudaStreamQuery(stream_->stream());
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:    return (error == cudaSuccess) || (error == cudaErrorNotReady);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#define LAUNCH_CUDA_KERNEL(kernel, gridsize, blocksize, sharedmem, device, ...)             \
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  assert(cudaGetLastError() == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifdef __CUDACC__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:static EIGEN_DEVICE_FUNC inline void setCudaSharedMemConfig(cudaSharedMemConfig config) {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#ifndef __CUDA_ARCH__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  cudaError_t status = cudaDeviceSetSharedMemConfig(config);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:  assert(status == cudaSuccess);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h:#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_CUDA_H
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    array<Index, NumDims> cudaInputDimensions;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    array<Index, NumDims> cudaOutputDimensions;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      cudaInputDimensions[index] = input_dims[indices[i]];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      cudaOutputDimensions[index] = dimensions[indices[i]];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        cudaInputDimensions[written] = input_dims[i];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        cudaOutputDimensions[written] = dimensions[i];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] =
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaInputStrides[i - 1] * cudaInputDimensions[i - 1];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] =
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaOutputStrides[i - 1] * cudaOutputDimensions[i - 1];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] = 1;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] = 1;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] =
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaInputStrides[i + 1] * cudaInputDimensions[i + 1];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] =
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:              m_cudaOutputStrides[i + 1] * cudaOutputDimensions[i + 1];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaInputStrides[i] = 1;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          m_cudaOutputStrides[i] = 1;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputPlaneToTensorInputOffset(Index p) const {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaInputStrides[d];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaInputStrides[d];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaInputStrides[d];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaInputStrides[d];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputPlaneToTensorOutputOffset(Index p) const {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaOutputStrides[d];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaOutputStrides[d];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const Index idx = p / m_cudaOutputStrides[d];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        p -= idx * m_cudaOutputStrides[d];
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputKernelToTensorInputOffset(Index i) const {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputKernelToTensorOutputOffset(Index i) const {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputKernelToTensorInputOffset(Index i, Index j) const {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputKernelToTensorOutputOffset(Index i, Index j) const {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaInputKernelToTensorInputOffset(Index i, Index j, Index k) const {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Index mapCudaOutputKernelToTensorOutputOffset(Index i, Index j, Index k) const {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  array<Index, NumDims> m_cudaInputStrides;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  array<Index, NumDims> m_cudaOutputStrides;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:// Use an optimized implementation of the evaluation code for GPUs whenever possible.
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const int tensor_index = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i+first_x);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const int tensor_index = plane_output_offset + indexMapper.mapCudaOutputKernelToTensorOutputOffset(i+first_x);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const int tensor_index = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i+first_x, j+first_y);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        const int tensor_index = plane_output_offset + indexMapper.mapCudaOutputKernelToTensorOutputOffset(i+first_x, j+first_y);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_input_offset = indexMapper.mapCudaInputPlaneToTensorInputOffset(p);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          const int tensor_index = plane_input_offset + indexMapper.mapCudaInputKernelToTensorInputOffset(i+first_x, j+first_y, k+first_z);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int plane_output_offset = indexMapper.mapCudaOutputPlaneToTensorOutputOffset(p);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:          const int tensor_index = plane_output_offset + indexMapper.mapCudaOutputKernelToTensorOutputOffset(i+first_x, j+first_y, k+first_z);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:struct TensorEvaluator<const TensorConvolutionOp<Indices, InputArgType, KernelArgType>, GpuDevice>
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  static const int NumDims =  internal::array_size<typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions>::value;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  typedef typename TensorEvaluator<KernelArgType, GpuDevice>::Dimensions KernelDimensions;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    IsAligned = TensorEvaluator<InputArgType, GpuDevice>::IsAligned & TensorEvaluator<KernelArgType, GpuDevice>::IsAligned,
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    Layout = TensorEvaluator<InputArgType, GpuDevice>::Layout,
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const GpuDevice& device)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<InputArgType, GpuDevice>::Layout) == static_cast<int>(TensorEvaluator<KernelArgType, GpuDevice>::Layout)), YOU_MADE_A_PROGRAMMING_MISTAKE);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions& input_dims = m_inputImpl.dimensions();
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const typename TensorEvaluator<KernelArgType, GpuDevice>::Dimensions& kernel_dims = m_kernelImpl.dimensions();
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  typedef typename PacketType<CoeffReturnType, GpuDevice>::type PacketReturnType;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      const bool PacketAccess = internal::IsVectorizable<GpuDevice, KernelArgType>::value;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:      internal::TensorExecutor<const EvalTo, GpuDevice, PacketAccess>::run(evalToTmp, m_device);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    typedef typename TensorEvaluator<InputArgType, GpuDevice>::Dimensions InputDims;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int maxThreadsPerBlock = m_device.maxCudaThreadsPerBlock();
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int maxBlocksPerProcessor = m_device.maxCudaThreadsPerMultiProcessor() / maxThreadsPerBlock;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:    const int numMultiProcessors = m_device.getNumCudaMultiProcessors();
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, 4, data);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, 7, data);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel1D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, kernel_size, data);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4, 7>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 4, 7, data);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 4, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 4, kernel_size_y, data);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7, 4>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 7, 4, data);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:                LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, 7, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, 7, kernel_size_y, data);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:            LAUNCH_CUDA_KERNEL((EigenConvolutionKernel2D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims, Dynamic, Dynamic>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, kernel_size_x, kernel_size_y, data);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:        LAUNCH_CUDA_KERNEL((EigenConvolutionKernel3D<TensorEvaluator<InputArgType, GpuDevice>, Index, InputDims>), num_blocks, block_size, shared_mem, m_device, m_inputImpl, indexMapper, m_kernel, numP, numX, maxX, numY, maxY, numZ, maxZ, kernel_size_x, kernel_size_y, kernel_size_z, data);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  TensorEvaluator<InputArgType, GpuDevice> m_inputImpl;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  TensorEvaluator<KernelArgType, GpuDevice> m_kernelImpl;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h:  const GpuDevice& m_device;
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:// GPU implementation of scan
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:struct ScanLauncher<Self, Reducer, GpuDevice> {
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:     LAUNCH_CUDA_KERNEL((ScanKernel<Self, Reducer>), num_blocks, block_size, 0, self.device(), self, total_size, data);
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorScan.h:#endif  // EIGEN_USE_GPU && __CUDACC__
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:// Use the texture cache on CUDA devices whenever possible
include/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
include/eigen3/unsupported/Eigen/CXX11/src/util/EmulateArray.h:// Moreover, CUDA doesn't support the STL containers, so we use our own instead.
include/eigen3/unsupported/Eigen/CXX11/src/util/EmulateArray.h:#if (__cplusplus <= 199711L && EIGEN_COMP_MSVC < 1900) || defined(__CUDACC__) || defined(EIGEN_AVOID_STL_ARRAY)
include/eigen3/unsupported/Eigen/CXX11/src/util/EmulateArray.h:// The compiler supports c++11, and we're not targetting cuda: use std::array as Eigen::array
include/eigen3/unsupported/Eigen/CXX11/src/util/MaxSizeVector.h:  * std::vector is not an option (e.g. on GPU or when compiling using
include/eigen3/unsupported/Eigen/SpecialFunctions:#if defined EIGEN_VECTORIZE_CUDA
include/eigen3/unsupported/Eigen/SpecialFunctions:  #include "src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h"
include/eigen3/unsupported/Eigen/OpenGLSupport:#ifdef GL_ARB_gpu_shader_fp64
include/eigen3/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:#if !defined(__CUDA_ARCH__) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__)
include/eigen3/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:#if !defined(__CUDA_ARCH__) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__)
include/eigen3/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h:    if (x == inf) return zero;  // std::isinf crashes on CUDA
include/eigen3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#ifndef EIGEN_CUDA_SPECIALFUNCTIONS_H
include/eigen3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#define EIGEN_CUDA_SPECIALFUNCTIONS_H
include/eigen3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
include/eigen3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
include/eigen3/unsupported/Eigen/src/SpecialFunctions/arch/CUDA/CudaSpecialFunctions.h:#endif // EIGEN_CUDA_SPECIALFUNCTIONS_H

```
