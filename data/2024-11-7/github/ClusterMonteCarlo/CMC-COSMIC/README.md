# https://github.com/ClusterMonteCarlo/CMC-COSMIC

```console
docs/sphinx/source/install/index.rst:    module load intel/2021 python/intel21 cuda/11.1
cuda/Makefile:CUDA_GLOBAL_INSTALL = /usr/local
cuda/Makefile:INCLUDES  += -I. -I$(CUDA_GLOBAL_INSTALL)/include -I./common/inc -I/share/apps/gsl-1.9/include/ -I/share/apps/cfitsio/include/
cuda/Makefile:LIB	   = ${LIB} -L$(CUDA_GLOBAL_INSTALL)/lib -L./lib -L./common/lib -lcuda -lcudart -L/usr/lib
cuda/Makefile:#use of cuda in the program
cuda/Makefile:#CFLAGS 	+= -DUSE_CUDA
cuda/Makefile:CUDALIBFLAGS := -lcuda -lcudart -lGL -lGLU
cuda/Makefile:CUDAEXTRAS   := ./common/obj/release/bank_checker.cpp_o \
cuda/Makefile:all: cmc_cuda.cu_o
cuda/Makefile:	rm -f cmc_cuda.cu_o
cuda/common/cutil.h:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/cutil.h:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/cutil.h:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/cutil.h:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/cutil.h:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/cutil.h:/* CUda UTility Library */
cuda/common/cutil.h:    if( CUDA_SUCCESS != err) {                                               \
cuda/common/cutil.h:        fprintf(stderr, "Cuda driver error %x in file '%s' in line %i.\n",   \
cuda/common/cutil.h:    if( CUDA_SUCCESS != err) {                                               \
cuda/common/cutil.h:        fprintf(stderr, "Cuda driver error %x in file '%s' in line %i.\n",   \
cuda/common/cutil.h:#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
cuda/common/cutil.h:    cudaError err = call;                                                    \
cuda/common/cutil.h:    if( cudaSuccess != err) {                                                \
cuda/common/cutil.h:        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
cuda/common/cutil.h:                __FILE__, __LINE__, cudaGetErrorString( err) );              \
cuda/common/cutil.h:#  define CUDA_SAFE_CALL( call) do {                                         \
cuda/common/cutil.h:    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
cuda/common/cutil.h:    cudaError err = cudaThreadSynchronize();                                 \
cuda/common/cutil.h:    if( cudaSuccess != err) {                                                \
cuda/common/cutil.h:        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
cuda/common/cutil.h:                __FILE__, __LINE__, cudaGetErrorString( err) );              \
cuda/common/cutil.h:    //! Check for CUDA error
cuda/common/cutil.h:    cudaError_t err = cudaThreadSynchronize();                               \
cuda/common/cutil.h:    if( cudaSuccess != err) {                                                \
cuda/common/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
cuda/common/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
cuda/common/cutil.h:#  define CUDA_SAFE_CALL_NO_SYNC( call) call
cuda/common/cutil.h:#  define CUDA_SAFE_CALL( call) call
cuda/common/cutil.h:    CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));                \
cuda/common/cutil.h:        cudaDeviceProp deviceProp;                                           \
cuda/common/cutil.h:        CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, dev));   \
cuda/common/cutil.h:        fprintf(stderr, "There is no device supporting CUDA.\n");            \
cuda/common/cutil.h:        CUDA_SAFE_CALL(cudaSetDevice(dev));                                  \
cuda/common/cutil.h:    if (CUDA_SUCCESS == err)                                                 \
cuda/common/cutil.h:        fprintf(stderr, "There is no device supporting CUDA.\n");            \
cuda/common/Makefile:# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/Makefile:# This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/Makefile:# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/Makefile:# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/Makefile:# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/Makefile:# CUda UTility library build script
cuda/common/inc/stopwatch_linux.h: * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/inc/stopwatch_linux.h: * This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/inc/stopwatch_linux.h: * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/inc/stopwatch_linux.h: * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/inc/stopwatch_linux.h: * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/inc/stopwatch_linux.h:/* CUda UTility Library */
cuda/common/inc/cutil.h:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/inc/cutil.h:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/inc/cutil.h:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/inc/cutil.h:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/inc/cutil.h:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/inc/cutil.h:/* CUda UTility Library */
cuda/common/inc/cutil.h:    if( CUDA_SUCCESS != err) {                                               \
cuda/common/inc/cutil.h:        fprintf(stderr, "Cuda driver error %x in file '%s' in line %i.\n",   \
cuda/common/inc/cutil.h:    if( CUDA_SUCCESS != err) {                                               \
cuda/common/inc/cutil.h:        fprintf(stderr, "Cuda driver error %x in file '%s' in line %i.\n",   \
cuda/common/inc/cutil.h:#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
cuda/common/inc/cutil.h:    cudaError err = call;                                                    \
cuda/common/inc/cutil.h:    if( cudaSuccess != err) {                                                \
cuda/common/inc/cutil.h:        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
cuda/common/inc/cutil.h:                __FILE__, __LINE__, cudaGetErrorString( err) );              \
cuda/common/inc/cutil.h:#  define CUDA_SAFE_CALL( call) do {                                         \
cuda/common/inc/cutil.h:    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
cuda/common/inc/cutil.h:    cudaError err = cudaThreadSynchronize();                                 \
cuda/common/inc/cutil.h:    if( cudaSuccess != err) {                                                \
cuda/common/inc/cutil.h:        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
cuda/common/inc/cutil.h:                __FILE__, __LINE__, cudaGetErrorString( err) );              \
cuda/common/inc/cutil.h:    //! Check for CUDA error
cuda/common/inc/cutil.h:    cudaError_t err = cudaThreadSynchronize();                               \
cuda/common/inc/cutil.h:    if( cudaSuccess != err) {                                                \
cuda/common/inc/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
cuda/common/inc/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
cuda/common/inc/cutil.h:#  define CUDA_SAFE_CALL_NO_SYNC( call) call
cuda/common/inc/cutil.h:#  define CUDA_SAFE_CALL( call) call
cuda/common/inc/cutil.h:    CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));                \
cuda/common/inc/cutil.h:        cudaDeviceProp deviceProp;                                           \
cuda/common/inc/cutil.h:        CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, dev));   \
cuda/common/inc/cutil.h:        fprintf(stderr, "There is no device supporting CUDA.\n");            \
cuda/common/inc/cutil.h:        CUDA_SAFE_CALL(cudaSetDevice(dev));                                  \
cuda/common/inc/cutil.h:    if (CUDA_SUCCESS == err)                                                 \
cuda/common/inc/cutil.h:        fprintf(stderr, "There is no device supporting CUDA.\n");            \
cuda/common/inc/bank_checker.h: * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/inc/bank_checker.h: * This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/inc/bank_checker.h: * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/inc/bank_checker.h: * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/inc/bank_checker.h: * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/inc/GL/glew.h:/* --------------------- GL_EXT_gpu_program_parameters --------------------- */
cuda/common/inc/GL/glew.h:#ifndef GL_EXT_gpu_program_parameters
cuda/common/inc/GL/glew.h:#define GL_EXT_gpu_program_parameters 1
cuda/common/inc/GL/glew.h:#define GLEW_EXT_gpu_program_parameters GLEW_GET_VAR(__GLEW_EXT_gpu_program_parameters)
cuda/common/inc/GL/glew.h:#endif /* GL_EXT_gpu_program_parameters */
cuda/common/inc/GL/glew.h:/* --------------------------- GL_EXT_gpu_shader4 -------------------------- */
cuda/common/inc/GL/glew.h:#ifndef GL_EXT_gpu_shader4
cuda/common/inc/GL/glew.h:#define GL_EXT_gpu_shader4 1
cuda/common/inc/GL/glew.h:#define GLEW_EXT_gpu_shader4 GLEW_GET_VAR(__GLEW_EXT_gpu_shader4)
cuda/common/inc/GL/glew.h:#endif /* GL_EXT_gpu_shader4 */
cuda/common/inc/GL/glew.h:/* --------------------------- GL_NV_gpu_program4 -------------------------- */
cuda/common/inc/GL/glew.h:#ifndef GL_NV_gpu_program4
cuda/common/inc/GL/glew.h:#define GL_NV_gpu_program4 1
cuda/common/inc/GL/glew.h:#define GLEW_NV_gpu_program4 GLEW_GET_VAR(__GLEW_NV_gpu_program4)
cuda/common/inc/GL/glew.h:#endif /* GL_NV_gpu_program4 */
cuda/common/inc/GL/glew.h:GLEW_VAR_EXPORT GLboolean __GLEW_EXT_gpu_program_parameters;
cuda/common/inc/GL/glew.h:GLEW_VAR_EXPORT GLboolean __GLEW_EXT_gpu_shader4;
cuda/common/inc/GL/glew.h:GLEW_VAR_EXPORT GLboolean __GLEW_NV_gpu_program4;
cuda/common/inc/cutil_interop.h:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/inc/cutil_interop.h:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/inc/cutil_interop.h:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/inc/cutil_interop.h:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/inc/cutil_interop.h:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/inc/cutil_interop.h:/* CUda UTility Library :: additional functionality for graphics 
cuda/common/inc/stopwatch_base.inl:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/inc/stopwatch_base.inl:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/inc/stopwatch_base.inl:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/inc/stopwatch_base.inl:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/inc/stopwatch_base.inl:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/inc/stopwatch_base.inl:/* CUda UTility Library */
cuda/common/inc/error_checker.h:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/inc/error_checker.h:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/inc/error_checker.h:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/inc/error_checker.h:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/inc/error_checker.h:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/inc/error_checker.h:/* CUda UTility Library */
cuda/common/inc/cmd_arg_reader.h:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/inc/cmd_arg_reader.h:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/inc/cmd_arg_reader.h:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/inc/cmd_arg_reader.h:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/inc/cmd_arg_reader.h:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/inc/cmd_arg_reader.h:/* CUda UTility Library */
cuda/common/inc/exception.h:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/inc/exception.h:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/inc/exception.h:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/inc/exception.h:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/inc/exception.h:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/inc/exception.h:/* CUda UTility Library */
cuda/common/inc/stopwatch.h:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/inc/stopwatch.h:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/inc/stopwatch.h:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/inc/stopwatch.h:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/inc/stopwatch.h:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/inc/stopwatch.h:/* CUda UTility Library */
cuda/common/inc/stopwatch_base.h:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/inc/stopwatch_base.h:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/inc/stopwatch_base.h:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/inc/stopwatch_base.h:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/inc/stopwatch_base.h:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/inc/stopwatch_base.h:/* CUda UTility Library */
cuda/common/inc/param.h:  sgreen@nvidia.com 4/2001
cuda/common/common.mk:# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/common.mk:# This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/common.mk:# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/common.mk:# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/common.mk:# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/common.mk:CUDA_INSTALL_PATH := /usr/local
cuda/common/common.mk:INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include -I$(COMMONDIR)/inc
cuda/common/common.mk:LIB       := -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib -lcuda -lcudart ${OPENGLLIB} ${LIB}
cuda/common/common.mk:		CUDACCFLAGS += 
cuda/common/common.mk:# Add cudacc flags
cuda/common/common.mk:NVCCFLAGS += $(CUDACCFLAGS)
cuda/common/src/cutil.cpp:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/src/cutil.cpp:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/src/cutil.cpp:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/src/cutil.cpp:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/src/cutil.cpp:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/src/cutil.cpp:/* CUda UTility Library */
cuda/common/src/cutil.cpp:/* Credit: Cuda team for the PGM file reader / writer code. */
cuda/common/src/cutil.cpp:// includes, cuda
cuda/common/src/stopwatch_linux.cpp: * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/src/stopwatch_linux.cpp: * This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/src/stopwatch_linux.cpp: * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/src/stopwatch_linux.cpp: * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/src/stopwatch_linux.cpp: * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/src/stopwatch_linux.cpp:/* CUda UTility Library */
cuda/common/src/bank_checker.cpp: * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/src/bank_checker.cpp: * This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/src/bank_checker.cpp: * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/src/bank_checker.cpp: * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/src/bank_checker.cpp: * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/src/cmd_arg_reader.cpp:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/src/cmd_arg_reader.cpp:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/src/cmd_arg_reader.cpp:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/src/cmd_arg_reader.cpp:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/src/cmd_arg_reader.cpp:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/src/cmd_arg_reader.cpp:/* CUda UTility Library */
cuda/common/src/error_checker.cpp:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/src/error_checker.cpp:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/src/error_checker.cpp:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/src/error_checker.cpp:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/src/error_checker.cpp:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/src/error_checker.cpp:/* CUda UTility Library */
cuda/common/src/cutil_interop.cpp:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/src/cutil_interop.cpp:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/src/cutil_interop.cpp:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/src/cutil_interop.cpp:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/src/cutil_interop.cpp:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/src/cutil_interop.cpp:/* CUda UTility Library :: additional functionality for graphics
cuda/common/src/cutil_interop.cpp:    // definitions for lib nvidia-cfg
cuda/common/src/cutil_interop.cpp:    handle = dlopen( "libnvidia-cfg.so", RTLD_LAZY);
cuda/common/src/cutil_interop.cpp:        // no NVIDIA driver installed
cuda/common/src/cutil_interop.cpp:        fprintf( stderr, "Cannot find NVIDIA driver.\n" );
cuda/common/src/cutil_interop.cpp:        fprintf( stderr, "Graphics interoperability on multi GPU systems currently not supported.\n" );
cuda/common/src/stopwatch.cpp:* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
cuda/common/src/stopwatch.cpp:* This source code is subject to NVIDIA ownership rights under U.S. and 
cuda/common/src/stopwatch.cpp:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
cuda/common/src/stopwatch.cpp:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
cuda/common/src/stopwatch.cpp:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
cuda/common/src/stopwatch.cpp:/* CUda UTility Library */
cuda/make_cubin:nvcc -arch sm_13 cmc_cuda.cu -I./common/inc -cubin -I/share/apps/cfitsio/include -I/share/apps/gsl-1.9/include && less cmc_cuda.cubin
cuda/trash/cuda-sp.cu:#define _CUDA_MAIN_
cuda/trash/cuda-sp.cu:#include "cuda.h"
cuda/trash/cuda-sp.cu:// Found in CUDA examples / yanked from dsfun90 directly
cuda/trash/cuda-sp.cu://start up CUDA - needs to be called before anything else
cuda/trash/cuda-sp.cu:	printf("------------------ USING CUDA -------------------\n");
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_m, sizeof(float)*totalStars) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_m_, sizeof(float)*totalStars) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_r, sizeof(float)*totalStars) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_r_, sizeof(float)*totalStars) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_phi, sizeof(float)*totalStars) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_phi_, sizeof(float)*totalStars) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_E, sizeof(float)*totalStars) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_E_, sizeof(float)*totalStars) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_J, sizeof(float)*totalStars) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_J_, sizeof(float)*totalStars) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_kmin, sizeof(long)*totalStars) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_kmax, sizeof(long)*totalStars) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_ktemp, sizeof(long)*totalStars) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMemcpy( cu_m, m, sizeof(float)*totalStars, cudaMemcpyHostToDevice) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMemcpy( cu_m_, m_, sizeof(float)*totalStars, cudaMemcpyHostToDevice) );
cuda/trash/cuda-sp.cu:    CUDA_SAFE_CALL( cudaMemcpy( cu_r, r, sizeof(float)*totalStars, cudaMemcpyHostToDevice) );
cuda/trash/cuda-sp.cu:    CUDA_SAFE_CALL( cudaMemcpy( cu_r_, r, sizeof(float)*totalStars, cudaMemcpyHostToDevice) );
cuda/trash/cuda-sp.cu:    CUDA_SAFE_CALL( cudaMemcpy( cu_phi, phi, sizeof(float)*totalStars, cudaMemcpyHostToDevice) );
cuda/trash/cuda-sp.cu:    CUDA_SAFE_CALL( cudaMemcpy( cu_phi_, phi_, sizeof(float)*totalStars, cudaMemcpyHostToDevice) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMemcpy( cu_E, cE, sizeof(float)*totalStars, cudaMemcpyHostToDevice) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMemcpy( cu_E_, cE_, sizeof(float)*totalStars, cudaMemcpyHostToDevice) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMemcpy( cu_J, cJ, sizeof(float)*totalStars, cudaMemcpyHostToDevice) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaMemcpy( cu_J_, cJ_, sizeof(float)*totalStars, cudaMemcpyHostToDevice) );
cuda/trash/cuda-sp.cu:    CUDA_SAFE_CALL( cudaMemcpy( h_kmin, cu_kmin, sizeof(long)*totalStars, cudaMemcpyDeviceToHost) );
cuda/trash/cuda-sp.cu:    CUDA_SAFE_CALL( cudaMemcpy( h_kmax, cu_kmax, sizeof(long)*totalStars, cudaMemcpyDeviceToHost) );
cuda/trash/cuda-sp.cu:    CUDA_SAFE_CALL( cudaMemcpy( h_ktemp, cu_ktemp, sizeof(long)*totalStars, cudaMemcpyDeviceToHost) );
cuda/trash/cuda-sp.cu:	printf("------------------ USING CUDA -------------------\n");
cuda/trash/cuda-sp.cu:	cudaThreadSynchronize();
cuda/trash/cuda-sp.cu:	cudaThreadSynchronize();
cuda/trash/cuda-sp.cu:	printf("\tCUDA device time: %f\n", time);
cuda/trash/cuda-sp.cu:// clean up all the CUDA crud
cuda/trash/cuda-sp.cu:	printf("------------------ USING CUDA -------------------\n");
cuda/trash/cuda-sp.cu:    CUDA_SAFE_CALL( cudaFree(cu_m) );
cuda/trash/cuda-sp.cu:    CUDA_SAFE_CALL( cudaFree(cu_r) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaFree(cu_r_) );
cuda/trash/cuda-sp.cu:    CUDA_SAFE_CALL( cudaFree(cu_phi) );
cuda/trash/cuda-sp.cu:    CUDA_SAFE_CALL( cudaFree(cu_J) );
cuda/trash/cuda-sp.cu:    CUDA_SAFE_CALL( cudaFree(cu_E) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaFree(cu_kmin) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaFree(cu_kmax) );
cuda/trash/cuda-sp.cu:	CUDA_SAFE_CALL( cudaFree(cu_ktemp) );
cuda/trash/Makefile.bk:CUDA_GLOBAL_INSTALL := /usr/local
cuda/trash/Makefile.bk:INCLUDES  += -I. -I$(CUDA_GLOBAL_INSTALL)/include -I./$(CUDADIR)/common/inc
cuda/trash/Makefile.bk:LIB       := ${LIB} -L$(CUDA_GLOBAL_INSTALL)/lib -L./$(CUDADIR)/lib -L./$(CUDADIR)/common/lib -lcuda -lcudart -L/usr/lib
cuda/trash/Makefile.bk:#use of cuda in the program
cuda/trash/Makefile.bk:CFLAGS 		+= -DUSE_CUDA
cuda/trash/Makefile.bk:CUDALIBFLAGS := -lcuda -lcudart -lGL -lGLU
cuda/trash/Makefile.bk:CUDAEXTRAS 	:= 	cuda/common/obj/release/bank_checker.cpp_o \
cuda/trash/Makefile.bk:	cuda/common/obj/release/cmd_arg_reader.cpp_o \
cuda/trash/Makefile.bk:	cuda/common/obj/release/cutil.cpp_o \
cuda/trash/Makefile.bk:	cuda/common/obj/release/cutil_interop.cpp_o \
cuda/trash/Makefile.bk:	cuda/common/obj/release/error_checker.cpp_o \
cuda/trash/Makefile.bk:	cuda/common/obj/release/stopwatch.cpp_o \
cuda/trash/Makefile.bk:	cuda/common/obj/release/stopwatch_linux.cpp_o 
cuda/trash/Makefile.bk:all: cuda.cu_o
cuda/trash/Makefile.bk:OBJS += $(CUDADIR)/cuda.cu_o
cuda/trash/Makefile.bk:	rm cuda.cu_o
cuda/trash/cuda_extern.h:#ifndef _CUDA_MAIN_
cuda/trash/cuda_extern.h:#define _CUDA_EXTERN_ extern
cuda/trash/cuda_extern.h:#define _CUDA_EXTERN_
cuda/trash/cuda_extern.h:_CUDA_EXTERN_ long  *h_kmin;
cuda/trash/cuda_extern.h:_CUDA_EXTERN_ long  *h_kmax;
cuda/trash/cuda_extern.h:_CUDA_EXTERN_ long  *h_ktemp;
cuda/cmc_cuda.h:#include <cuda.h>
cuda/cmc_cuda.h:#ifndef _CUDA_MAIN_
cuda/cmc_cuda.h:#define _CUDA_EXTERN_ extern
cuda/cmc_cuda.h:#define _CUDA_EXTERN_
cuda/cmc_cuda.h:  All the _CUDA_EXTERN_ are host variables that need to be copied over
cuda/cmc_cuda.h:  to the __device__ variables that live in cmc_cuda.cu.  They are simply
cuda/cmc_cuda.h:_CUDA_EXTERN_ double *m;
cuda/cmc_cuda.h:_CUDA_EXTERN_ double *r;
cuda/cmc_cuda.h:_CUDA_EXTERN_ double *phi;
cuda/cmc_cuda.h:_CUDA_EXTERN_ double *cE;
cuda/cmc_cuda.h:_CUDA_EXTERN_ double *cJ;
cuda/cmc_cuda.h:_CUDA_EXTERN_ double *cf;
cuda/cmc_cuda.h:_CUDA_EXTERN_ long  *h_kmin;
cuda/cmc_cuda.h:_CUDA_EXTERN_ long  *h_kmax;
cuda/cmc_cuda.h:_CUDA_EXTERN_ long  *h_ktemp;
cuda/cmc_cuda.h:#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
cuda/cmc_cuda.h:     cudaError err = call;                                                   \
cuda/cmc_cuda.h:    if( cudaSuccess != err) {                                                \
cuda/cmc_cuda.h:        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
cuda/cmc_cuda.h:                __FILE__, __LINE__, cudaGetErrorString( err) );              \
cuda/cmc_cuda.h:#  define CUDA_SAFE_CALL( call) do {                                         \
cuda/cmc_cuda.h:    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
cuda/cmc_cuda.h:    cudaError err = cudaThreadSynchronize();                                 \
cuda/cmc_cuda.h:    if( cudaSuccess != err) {                                                \
cuda/cmc_cuda.h:        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
cuda/cmc_cuda.h:                __FILE__, __LINE__, cudaGetErrorString( err) );              \
cuda/cmc_cuda.h:    CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));                \
cuda/cmc_cuda.h:        cudaDeviceProp deviceProp;                                           \
cuda/cmc_cuda.h:        CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, dev));   \
cuda/cmc_cuda.h:        fprintf(stderr, "There is no device supporting CUDA.\n");            \
cuda/cmc_cuda.h:        CUDA_SAFE_CALL(cudaSetDevice(dev));                                  \
cuda/cmc_cuda.h:    if (CUDA_SUCCESS == err)                                                 \
cuda/cmc_cuda.h:        fprintf(stderr, "There is no device supporting CUDA.\n");            \
cuda/cmc_cuda.cu:#define _CUDA_MAIN_
cuda/cmc_cuda.cu:#include "cmc_cuda.h" 
cuda/cmc_cuda.cu:// Start up CUDA - needs to be called before anything else
cuda/cmc_cuda.cu:    printf("------------------ USING CUDA -------------------\n");
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_m, sizeof(double)*totalStars) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_r, sizeof(double)*totalStars) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_phi, sizeof(double)*totalStars) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_E, sizeof(double)*totalStars) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_J, sizeof(double)*totalStars) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_kmin, sizeof(long)*totalStars) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_kmax, sizeof(long)*totalStars) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMalloc( (void**) &cu_ktemp, sizeof(long)*totalStars) );
cuda/cmc_cuda.cu:    cudaError_t er = cudaGetLastError();
cuda/cmc_cuda.cu:    printf("\tCUDA's comments regarding init: %s\n", cudaGetErrorString(er));
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMemcpy( cu_m, m, sizeof(double)*totalStars, cudaMemcpyHostToDevice) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMemcpy( cu_r, r, sizeof(double)*totalStars, cudaMemcpyHostToDevice) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMemcpy( cu_phi, phi, sizeof(double)*totalStars, cudaMemcpyHostToDevice) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMemcpy( cu_E, cE, sizeof(double)*totalStars, cudaMemcpyHostToDevice) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMemcpy( cu_J, cJ, sizeof(double)*totalStars, cudaMemcpyHostToDevice) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMemcpy( h_kmin, cu_kmin, sizeof(long)*totalStars, cudaMemcpyDeviceToHost) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMemcpy( h_kmax, cu_kmax, sizeof(long)*totalStars, cudaMemcpyDeviceToHost) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaMemcpy( h_ktemp, cu_ktemp, sizeof(long)*totalStars, cudaMemcpyDeviceToHost) );
cuda/cmc_cuda.cu:    printf("------------------ USING CUDA -------------------\n");
cuda/cmc_cuda.cu:    //printf("\tCUDA memcpy time: %f sec\n", double(end-start)/(CLOCKS_PER_SEC));
cuda/cmc_cuda.cu:    cudaThreadSynchronize();
cuda/cmc_cuda.cu:    cudaThreadSynchronize();
cuda/cmc_cuda.cu:	    cudaError er = cudaGetLastError();
cuda/cmc_cuda.cu:	    printf("\t||\tCUDA says: %s\n", cudaGetErrorString(er));
cuda/cmc_cuda.cu:    cudaError er = cudaGetLastError();
cuda/cmc_cuda.cu:    printf("\tCUDA says: %s\n", cudaGetErrorString(er));
cuda/cmc_cuda.cu:// clean up all the CUDA crud
cuda/cmc_cuda.cu:    printf("------------------ USING CUDA -------------------\n");
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaFree(cu_m) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaFree(cu_r) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaFree(cu_phi) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaFree(cu_J) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaFree(cu_E) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaFree(cu_kmin) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaFree(cu_kmax) );
cuda/cmc_cuda.cu:    CUDA_SAFE_CALL( cudaFree(cu_ktemp) );
src/cmc/cmc_io.c:	//If we use the GPU code, we dont need the SEARCH_GRID. So commenting it out
src/cmc/cmc_evolution_thr.c:#ifdef USE_CUDA
src/cmc/cmc_evolution_thr.c:#include "cuda/cmc_cuda.h"
src/cmc/cmc_evolution_thr.c:#ifdef USE_CUDA
src/cmc/cmc_evolution_thr.c:#ifdef USE_CUDA
src/cmc/cmc.c:#ifdef USE_CUDA
src/cmc/cmc.c:#include "cuda/cmc_cuda.h"
src/cmc/cmc.c:	#ifdef USE_CUDA
src/cmc/cmc.c:#ifdef USE_CUDA

```
