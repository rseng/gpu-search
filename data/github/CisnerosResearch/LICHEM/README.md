# https://github.com/CisnerosResearch/LICHEM

```console
Makefile:GPUFLAGS=-fopenacc
Makefile:GPUDev:	title gpubin devtest manual stats compdone
Makefile:FLAGSGPU=$(CXXFLAGS) $(DEVFLAGS) $(GPUFLAGS) $(LDFLAGS) -I./src/ -I./include/
Makefile:gpubin:
Makefile:	echo "### Compiling the LICHEM GPU binary ###"; \
Makefile:	$(CXX) ./src/LICHEM.cpp -o $(INSTALLBIN)/lichem $(FLAGSGPU)
include/LICHEM_clibs.h:#ifdef _OPENACC
include/LICHEM_clibs.h: #pragma message("OpenACC is enabled.")
include/LICHEM_clibs.h: #include <openacc.h>
include/LICHEM_clibs.h: #pragma message("OpenACC is disabled.")
Eigen3/Eigen/Core:// Handle NVCC/CUDA
Eigen3/Eigen/Core:#ifdef __CUDACC__
Eigen3/Eigen/Core:  // Do not try asserts on CUDA!
Eigen3/Eigen/Core:  // Do not try to vectorize on CUDA!
Eigen3/Eigen/Core:  // All functions callable from CUDA code must be qualified with __device__
Eigen3/Eigen/Core:// When compiling CUDA device code with NVCC, pull in math functions from the
Eigen3/Eigen/Core:#if defined(__CUDA_ARCH__) && defined(__NVCC__)
Eigen3/Eigen/Core:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(EIGEN_EXCEPTIONS)
Eigen3/Eigen/Core:#if defined __CUDACC__
Eigen3/Eigen/Core:  #define EIGEN_VECTORIZE_CUDA
Eigen3/Eigen/Core:  #if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
Eigen3/Eigen/Core:    #define EIGEN_HAS_CUDA_FP16
Eigen3/Eigen/Core:#if defined EIGEN_HAS_CUDA_FP16
Eigen3/Eigen/Core:  #include <cuda_fp16.h>
Eigen3/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
Eigen3/Eigen/Core:#include "src/Core/arch/CUDA/Half.h"
Eigen3/Eigen/Core:#include "src/Core/arch/CUDA/PacketMathHalf.h"
Eigen3/Eigen/Core:#include "src/Core/arch/CUDA/TypeCasting.h"
Eigen3/Eigen/Core:#if defined EIGEN_VECTORIZE_CUDA
Eigen3/Eigen/Core:  #include "src/Core/arch/CUDA/PacketMath.h"
Eigen3/Eigen/Core:  #include "src/Core/arch/CUDA/MathFunctions.h"
Eigen3/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
Eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
Eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
Eigen3/Eigen/src/Core/util/Meta.h:  static float (max)() { return CUDART_MAX_NORMAL_F; }
Eigen3/Eigen/src/Core/util/Meta.h:  static float infinity() { return CUDART_INF_F; }
Eigen3/Eigen/src/Core/util/Meta.h:  static float quiet_NaN() { return CUDART_NAN_F; }
Eigen3/Eigen/src/Core/util/Meta.h:  static double infinity() { return CUDART_INF; }
Eigen3/Eigen/src/Core/util/Meta.h:  static double quiet_NaN() { return CUDART_NAN; }
Eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
Eigen3/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
Eigen3/Eigen/src/Core/util/Macros.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/util/Macros.h:#if EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && defined(__CUDACC_VER__) && (EIGEN_COMP_CLANG || __CUDACC_VER__ >= 70500))
Eigen3/Eigen/src/Core/util/Macros.h:#if (defined __CUDACC__)
Eigen3/Eigen/src/Core/util/Macros.h:#  ifdef __CUDA_ARCH__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
Eigen3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
Eigen3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
Eigen3/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifndef __CUDA_ARCH__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#if (!defined(__CUDACC__)) && EIGEN_FAST_MATH
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
Eigen3/Eigen/src/Core/MatrixBase.h:#ifdef __CUDACC__
Eigen3/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
Eigen3/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
Eigen3/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
Eigen3/Eigen/src/Core/GeneralProduct.h:#ifndef __CUDACC__
Eigen3/Eigen/src/Core/GeneralProduct.h:#endif // __CUDACC__
Eigen3/Eigen/src/Core/GenericPacketMath.h:#ifdef __CUDA_ARCH__
Eigen3/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
Eigen3/Eigen/src/Core/GenericPacketMath.h:#ifndef __CUDACC__
Eigen3/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
Eigen3/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
Eigen3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
Eigen3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
Eigen3/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
Eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#ifndef EIGEN_PACKET_MATH_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#define EIGEN_PACKET_MATH_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
Eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
Eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Eigen3/Eigen/src/Core/arch/CUDA/PacketMath.h:#endif // EIGEN_PACKET_MATH_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:// type Eigen::half (inheriting from CUDA's __half struct) with
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:// to disk and the likes), but fast on GPUs.
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#ifndef EIGEN_HALF_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#define EIGEN_HALF_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if !defined(EIGEN_HAS_CUDA_FP16)
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Make our own __half definition that is similar to CUDA's.
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:  #if !defined(EIGEN_HAS_CUDA_FP16)
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Definitions for CPUs and older CUDA, mostly working through conversion
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__)
Eigen3/Eigen/src/Core/arch/CUDA/Half.h:#endif // EIGEN_HALF_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#ifndef EIGEN_PACKET_MATH_HALF_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#define EIGEN_PACKET_MATH_HALF_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 530
Eigen3/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#endif // EIGEN_PACKET_MATH_HALF_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
Eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
Eigen3/Eigen/src/Core/arch/CUDA/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#define EIGEN_TYPE_CASTING_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Eigen3/Eigen/src/Core/arch/CUDA/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
Eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
Eigen3/Eigen/src/Core/arch/CUDA/Complex.h:// building for CUDA to avoid non-constexpr methods.
Eigen3/Eigen/src/Core/arch/CUDA/Complex.h:#endif // EIGEN_COMPLEX_CUDA_H
Eigen3/Eigen/src/SVD/BDCSVD.h:#ifndef __CUDACC__
src/makefile.in:GPUFLAGS=-fopenacc
src/makefile.in:GPUDev:	title gpubin devtest manual stats compdone
src/makefile.in:FLAGSGPU=$(CXXFLAGS) $(DEVFLAGS) $(GPUFLAGS) $(LDFLAGS) -I./src/ -I./include/
src/makefile.in:gpubin:	
src/makefile.in:	echo "### Compiling the LICHEM GPU binary ###"; \
src/makefile.in:	$(CXX) ./src/LICHEM.cpp -o $(INSTALLBIN)/lichem $(FLAGSGPU)

```
