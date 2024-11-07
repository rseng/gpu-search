# https://github.com/youngjookim/sdr

```console
Code/packages/tapkee-master/include/tapkee/routines/eigendecomposition.hpp:		if (strategy.is(HeterogeneousOpenCLStrategy))
Code/packages/tapkee-master/include/tapkee/routines/eigendecomposition.hpp:				return eigendecomposition_impl_arpack<DenseMatrix,GPUDenseMatrixOperation>
Code/packages/tapkee-master/include/tapkee/routines/eigendecomposition.hpp:				return eigendecomposition_impl_arpack<DenseMatrix,GPUDenseImplicitSquareMatrixOperation>
Code/packages/tapkee-master/include/tapkee/routines/eigendecomposition.hpp:		if (strategy.is(HeterogeneousOpenCLStrategy))
Code/packages/tapkee-master/include/tapkee/routines/eigendecomposition.hpp:				return eigendecomposition_impl_randomized<DenseMatrix,GPUDenseMatrixOperation>
Code/packages/tapkee-master/include/tapkee/routines/eigendecomposition.hpp:				return eigendecomposition_impl_randomized<DenseMatrix,GPUDenseImplicitSquareMatrixOperation>
Code/packages/tapkee-master/include/tapkee/routines/matrix_operations.hpp:struct GPUDenseImplicitSquareMatrixOperation
Code/packages/tapkee-master/include/tapkee/routines/matrix_operations.hpp:	GPUDenseImplicitSquareMatrixOperation(const DenseMatrix& matrix)
Code/packages/tapkee-master/include/tapkee/routines/matrix_operations.hpp:struct GPUDenseMatrixOperation
Code/packages/tapkee-master/include/tapkee/routines/matrix_operations.hpp:	GPUDenseMatrixOperation(const DenseMatrix& matrix)
Code/packages/tapkee-master/include/tapkee/defines/methods.hpp:	static const ComputationStrategy HeterogeneousOpenCLStrategy("OpenCL");
Code/packages/tapkee-master/include/tapkee/defines/keywords.hpp:			computation_strategy("computation strategy (cpu, cpu+gpu)", HomogeneousCPUStrategy);
Code/packages/tapkee-master/CMakeLists.txt:find_package(OpenCL)
Code/packages/tapkee-master/CMakeLists.txt:if (OPENCL_FOUND)
Code/packages/tapkee-master/CMakeLists.txt:		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPENCL_C_FLAGS}")
Code/packages/tapkee-master/CMakeLists.txt:		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENCL_CXX_FLAGS}")
Code/packages/tapkee-master/CMakeLists.txt:		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OPENCL_EXE_LINKER_FLAGS}")
Code/packages/tapkee-master/CMakeLists.txt:	target_link_libraries(tapkee OpenCL)
Code/packages/tapkee-master/CMakeLists.txt:	target_link_libraries(tapkee_library INTERFACE OpenCL)
Code/packages/tapkee-master/CMakeLists.txt:			target_link_libraries(test_${exe} OpenCL)
Code/packages/tapkee-master/src/cli/main.cpp:		"opencl, "
Code/packages/tapkee-master/src/cli/util.hpp:	if (!strcmp(str,"opencl"))
Code/packages/tapkee-master/src/cli/util.hpp:		return tapkee::HeterogeneousOpenCLStrategy;
Code/packages/tapkee-master/src/cmake/FindViennaCL.cmake:# VIENNACL_WITH_OPENCL
Code/packages/tapkee-master/src/cmake/FindViennaCL.cmake:option(VIENNACL_WITH_OPENCL "Use ViennaCL with OpenCL" YES)
Code/packages/tapkee-master/src/cmake/FindViennaCL.cmake:IF(VIENNACL_WITH_OPENCL)
Code/packages/tapkee-master/src/cmake/FindViennaCL.cmake:  find_package(OpenCL REQUIRED)
Code/packages/tapkee-master/src/cmake/FindViennaCL.cmake:ENDIF(VIENNACL_WITH_OPENCL)
Code/packages/tapkee-master/src/cmake/FindViennaCL.cmake:if(VIENNACL_WITH_OPENCL)
Code/packages/tapkee-master/src/cmake/FindViennaCL.cmake:  set(VIENNACL_INCLUDE_DIRS ${VIENNACL_INCLUDE_DIR} ${OPENCL_INCLUDE_DIRS})
Code/packages/tapkee-master/src/cmake/FindViennaCL.cmake:  set(VIENNACL_LIBRARIES ${OPENCL_LIBRARIES})
Code/packages/tapkee-master/src/cmake/FindViennaCL.cmake:  find_package_handle_standard_args(ViennaCL "ViennaCL not found!" VIENNACL_INCLUDE_DIR OPENCL_INCLUDE_DIRS OPENCL_LIBRARIES)
Code/packages/tapkee-master/src/cmake/FindViennaCL.cmake:else(VIENNACL_WITH_OPENCL)
Code/packages/tapkee-master/src/cmake/FindViennaCL.cmake:endif(VIENNACL_WITH_OPENCL)
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:#  This file taken from FindOpenCL project @ http://gitorious.com/findopencl
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:# - Try to find OpenCL
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:# This module tries to find an OpenCL implementation on your system. It supports
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:# AMD / ATI, Apple and NVIDIA implementations, but shoudl work, too.
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:#  OPENCL_FOUND        - system has OpenCL
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:#  OPENCL_INCLUDE_DIRS  - the OpenCL include directory
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:#  OPENCL_LIBRARIES    - link these to use OpenCL
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:SET (OPENCL_VERSION_STRING "0.1.0")
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:SET (OPENCL_VERSION_MAJOR 0)
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:SET (OPENCL_VERSION_MINOR 1)
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:SET (OPENCL_VERSION_PATCH 0)
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:  FIND_LIBRARY(OPENCL_LIBRARIES OpenCL DOC "OpenCL lib for OSX")
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:  FIND_PATH(OPENCL_INCLUDE_DIRS OpenCL/cl.h DOC "Include for OpenCL on OSX")
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:  FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS OpenCL/cl.hpp DOC "Include for OpenCL CPP bindings on OSX")
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	    FIND_PATH(OPENCL_INCLUDE_DIRS CL/cl.h)
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	    FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp)
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	    	SET(OPENCL_LIB_DIR "$ENV{ATISTREAMSDKROOT}/lib/x86_64")
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:			SET(OPENCL_LIB_DIR "$ENV{ATIINTERNALSTREAMSDKROOT}/lib/x86_64")
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	    	SET(OPENCL_LIB_DIR "$ENV{ATISTREAMSDKROOT}/lib/x86")
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	   		SET(OPENCL_LIB_DIR "$ENV{ATIINTERNALSTREAMSDKROOT}/lib/x86")
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	    # 64 or 32 bit NVIDIA library paths to the search:
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	    	FIND_LIBRARY(OPENCL_LIBRARIES OpenCL.lib ${OPENCL_LIB_DIR} $ENV{CUDA_LIB_PATH} $ENV{CUDA_PATH}/lib/x64)
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	    	FIND_LIBRARY(OPENCL_LIBRARIES OpenCL.lib ${OPENCL_LIB_DIR} $ENV{CUDA_LIB_PATH} $ENV{CUDA_PATH}/lib/Win32)
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	    GET_FILENAME_COMPONENT(_OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE)
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	    FIND_PATH(OPENCL_INCLUDE_DIRS CL/cl.h PATHS "${_OPENCL_INC_CAND}" $ENV{CUDA_INC_PATH} $ENV{CUDA_PATH}/include)
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	    FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS "${_OPENCL_INC_CAND}" $ENV{CUDA_INC_PATH} $ENV{CUDA_PATH}/include)
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:            FIND_LIBRARY(OPENCL_LIBRARIES OpenCL
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:            GET_FILENAME_COMPONENT(OPENCL_LIB_DIR ${OPENCL_LIBRARIES} PATH)
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:            GET_FILENAME_COMPONENT(_OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE)
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:            FIND_PATH(OPENCL_INCLUDE_DIRS CL/cl.h PATHS ${_OPENCL_INC_CAND} "/usr/local/cuda/include")
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:            FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS ${_OPENCL_INC_CAND} "/usr/local/cuda/include")
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:FIND_PACKAGE_HANDLE_STANDARD_ARGS( OpenCL DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS )
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:IF( _OPENCL_CPP_INCLUDE_DIRS )
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	SET( OPENCL_HAS_CPP_BINDINGS TRUE )
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	LIST( APPEND OPENCL_INCLUDE_DIRS ${_OPENCL_CPP_INCLUDE_DIRS} )
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:	LIST( REMOVE_DUPLICATES OPENCL_INCLUDE_DIRS )
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:ENDIF( _OPENCL_CPP_INCLUDE_DIRS )
Code/packages/tapkee-master/src/cmake/FindOpenCL.cmake:  OPENCL_INCLUDE_DIRS
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:// Handle NVCC/CUDA/SYCL
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:#if defined(__CUDACC__) || defined(__SYCL_DEVICE_ONLY__)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:  // Do not try asserts on CUDA and SYCL!
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:  // All functions callable from CUDA code must be qualified with __device__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:  #ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:    // Do not try to vectorize on CUDA and SYCL!
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:// When compiling CUDA device code with NVCC, pull in math functions from the
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:#if defined(__CUDA_ARCH__) && defined(__NVCC__)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:#if defined __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:  #define EIGEN_VECTORIZE_CUDA
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:  #if defined __CUDACC_VER__ && __CUDACC_VER__ >= 70500
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:    #define EIGEN_HAS_CUDA_FP16
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:#if defined EIGEN_HAS_CUDA_FP16
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:  #include <cuda_fp16.h>
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:#include "src/Core/arch/CUDA/Half.h"
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:#include "src/Core/arch/CUDA/PacketMathHalf.h"
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:#include "src/Core/arch/CUDA/TypeCasting.h"
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:#if defined EIGEN_VECTORIZE_CUDA
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:  #include "src/Core/arch/CUDA/PacketMath.h"
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:  #include "src/Core/arch/CUDA/MathFunctions.h"
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:// on CUDA devices
Code/packages/Eigen.3.3.3/build/native/include/Eigen/Core:#include "src/Core/arch/CUDA/Complex.h"
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Meta.h:  static float (max)() { return CUDART_MAX_NORMAL_F; }
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Meta.h:  static float infinity() { return CUDART_INF_F; }
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Meta.h:  static float quiet_NaN() { return CUDART_NAN_F; }
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Meta.h:  static double infinity() { return CUDART_INF; }
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Meta.h:  static double quiet_NaN() { return CUDART_NAN; }
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Meta.h:#if defined(__CUDA_ARCH__)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Macros.h:  && ( !defined(__NVCC__) || !EIGEN_ARCH_ARM_OR_ARM64 || (defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000) )
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Macros.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Macros.h:#if EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && defined(__CUDACC_VER__) && (EIGEN_COMP_CLANG || __CUDACC_VER__ >= 70500))
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Macros.h:#if (defined __CUDACC__)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Macros.h:#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC < 1900 ||  defined(__CUDACC_VER__)) // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/util/Macros.h:#  ifdef __CUDA_ARCH__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDA_ARCH__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:  #ifdef __CUDA_ARCH__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifndef __CUDA_ARCH__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#if (!defined(__CUDACC__)) && EIGEN_FAST_MATH
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MathFunctions.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/MatrixBase.h:#ifdef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/GeneralProduct.h:#ifndef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/GeneralProduct.h:#endif // __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/GenericPacketMath.h:#ifdef __CUDA_ARCH__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/GenericPacketMath.h:#ifndef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/ProductEvaluators.h:#ifndef __CUDACC__
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMath.h:#ifndef EIGEN_PACKET_MATH_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMath.h:#define EIGEN_PACKET_MATH_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMath.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMath.h:#endif // EIGEN_PACKET_MATH_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:// type Eigen::half (inheriting from CUDA's __half struct) with
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:// to disk and the likes), but fast on GPUs.
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#ifndef EIGEN_HALF_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#define EIGEN_HALF_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#if !defined(EIGEN_HAS_CUDA_FP16)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:// Make our own __half definition that is similar to CUDA's.
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:  #if !defined(EIGEN_HAS_CUDA_FP16)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:// Definitions for CPUs and older CUDA, mostly working through conversion
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#if defined(__CUDA_ARCH__)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Half.h:#endif // EIGEN_HALF_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#ifndef EIGEN_PACKET_MATH_HALF_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#define EIGEN_PACKET_MATH_HALF_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 350
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 530
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/PacketMathHalf.h:#endif // EIGEN_PACKET_MATH_HALF_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/MathFunctions.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/TypeCasting.h:#define EIGEN_TYPE_CASTING_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/TypeCasting.h:    #if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/TypeCasting.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Complex.h:#ifndef EIGEN_COMPLEX_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Complex.h:#define EIGEN_COMPLEX_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Complex.h:#if defined(__CUDACC__) && defined(EIGEN_USE_GPU)
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Complex.h:// building for CUDA to avoid non-constexpr methods.
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/Core/arch/CUDA/Complex.h:#endif // EIGEN_COMPLEX_CUDA_H
Code/packages/Eigen.3.3.3/build/native/include/Eigen/src/SVD/BDCSVD.h:#ifndef __CUDACC__

```
