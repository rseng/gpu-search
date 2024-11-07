# https://github.com/pkestene/xsmurf

```console
ChangeLog:	- use Nvidia GPU: in doc/template one can find a CUDA project that
ChangeLog:	  to allowing testing GPU computing.
doc/templates/testXsmurf_scalar2d/scr_gaussian.tcl:    set type xsm_fftw${useFftw}_nmaxsup${useNMaxSup}_gpu${useGPU}
doc/templates/testXsmurf_scalar2d/scr_gaussian.tcl:	if {$useGPU==1} {
doc/templates/testXsmurf_scalar2d/scr_gaussian.tcl:	    #set path2cuda /home/pkestene/install/nvidia/cuda/NVIDIA_CUDA_SDK1.1/bin/linux/release
doc/templates/testXsmurf_scalar2d/scr_gaussian.tcl:	    exec ${path2cuda}/cannyEdge2D_cuda -i ${imaIdf} -o ${baseDir}/${imaIdf}_${type}_max_${wavelet}/max -s $noct -v $nvox 
doc/templates/testXsmurf_scalar2d/parameters_gaussian.tcl.in:set useGPU 0
doc/templates/testXsmurf_scalar2d/parameters_gaussian.tcl.in:set path2cuda /home/pkestene/install/nvidia/cuda/NVIDIA_CUDA_SDK1.1/bin/linux/release
doc/templates/testXsmurf_scalar2d/parameters_mexican.tcl.in:set useGPU 0
doc/templates/testXsmurf_scalar2d/parameters_mexican.tcl.in:set path2cuda /home/pkestene/install/nvidia/cuda/NVIDIA_CUDA_SDK1.1/bin/linux/release
doc/templates/testXsmurf_scalar2d/scr_mexican.tcl:    set type xsm_fftw${useFftw}_nmaxsup${useNMaxSup}_gpu${useGPU}
doc/templates/testXsmurf_scalar2d/average_pf.tcl:    variable useGPU
doc/templates/testXsmurf_scalar2d/average_pf.tcl:    set type xsm_fftw${useFftw}_nmaxsup${useNMaxSup}_gpu${useGPU}
doc/templates/testXsmurf_scalar2d/README:If you have a recent Nvidia graphics card (GeForce 8 series), and the Cuda SDK
doc/templates/testXsmurf_scalar2d/README:installed in you path, you can try to use the GPU to compute the WTMM edges.
doc/templates/testXsmurf_scalar2d/README:directory $(top_of_xsmurf_sources)/doc/templates/gpu/cannyEdge2D in the 
doc/templates/testXsmurf_scalar2d/README:projects directory of your CUDA SDK, go into this directory and type make; this
doc/templates/testXsmurf_scalar2d/README:will install the binary executable cannyEdge2D_cuda in your CUDA SDK 
doc/templates/testXsmurf_scalar2d/README:You can then turn on the use of the GPU in xsmurf, by setting useGPU to 1, 
doc/templates/testXsmurf_scalar2d/README:purely CPU execution (usetGPU set to 0).
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu: * cannyEdge2D_cuda.c - program of 2D edges detection
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu: *                                        run on host and device (GPU)
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu: * programme test pour port vers GPU (NVIDIA/CUDA)
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:#include "cmd_cannyEdge2D_cuda.h"
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu: * ./cannyEdge2D_cuda -o edge -i image.xsm
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu: *  ../../bin/linux/release/cannyEdge2D_cuda -i image.xsm -o edge
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu: * ./cannyEdge2D_cuda --output edge --input image.xsm
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu: * CUDA header
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:#include "cannyEdge2D_cuda_kernel.cu"
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  /* cuda related variables */ 
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  cudaError_t res;
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:   * CUDA
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  // display CUDA device info
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:    cudaDeviceProp deviceProp;
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  res=cudaMallocHost((void **) &dataIn, Lx*Ly* sizeof(Complex));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaMalloc((void**)&deviceFourier, Lx*Ly*sizeof(Complex) ));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaMalloc((void**)&deviceGradx, Lx*Ly*sizeof(Complex) ));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaMalloc((void**)&deviceGrady, Lx*Ly*sizeof(Complex) ));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaMemcpy(deviceFourier, dataIn, Lx*Ly*sizeof(Complex), cudaMemcpyHostToDevice));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaMalloc((void**)&deviceMod, Lx*Ly*sizeof(float) ));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaMalloc((void**)&deviceArg, Lx*Ly*sizeof(float) ));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaMalloc((void**)&deviceMaxima, lx*ly*sizeof(float) ));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  cudaMallocHost((void **) &maximaMask, Lx*Ly* sizeof(float));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  cudaMallocHost((void **) &arg, Lx*Ly* sizeof(float));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:      CUDA_SAFE_CALL(cudaMemcpy(deviceGradx, deviceFourier, Lx*Ly*sizeof(Complex), cudaMemcpyDeviceToDevice));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:      CUDA_SAFE_CALL(cudaMemcpy(deviceGrady, deviceFourier, Lx*Ly*sizeof(Complex), cudaMemcpyDeviceToDevice));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:      CUDA_SAFE_CALL(cudaMemcpy(maximaMask,deviceMaxima, Lx*Ly*sizeof(float), cudaMemcpyDeviceToHost));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:      CUDA_SAFE_CALL(cudaMemcpy(arg,deviceArg, Lx*Ly*sizeof(float), cudaMemcpyDeviceToHost));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaFree(deviceFourier));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaFree(deviceGradx));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaFree(deviceGrady));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaFree(deviceMod));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaFree(deviceArg));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaFree(deviceMaxima));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaFreeHost(maximaMask));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaFreeHost(arg));
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.cu:  CUDA_SAFE_CALL(cudaFreeHost(dataIn)); 
doc/templates/gpu/cannyEdge2D/reduction_kernel.cu: * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
doc/templates/gpu/cannyEdge2D/reduction_kernel.cu: * This source code is subject to NVIDIA ownership rights under U.S. and 
doc/templates/gpu/cannyEdge2D/reduction_kernel.cu: * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
doc/templates/gpu/cannyEdge2D/reduction_kernel.cu: * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
doc/templates/gpu/cannyEdge2D/reduction_kernel.cu: * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
doc/templates/gpu/cannyEdge2D/reduction_kernel.cu:   operator.  This operator is very expensive on GPUs, and the interleaved 
doc/templates/gpu/cannyEdge2D/cmd_cannyEdge2D_cuda.c:  gengetopt -i cannyEdge2D_cuda.ggo -F cmd_cannyEdge2D_cuda --long-help -u 
doc/templates/gpu/cannyEdge2D/cmd_cannyEdge2D_cuda.c:#include "cmd_cannyEdge2D_cuda.h"
doc/templates/gpu/cannyEdge2D/cmd_cannyEdge2D_cuda.c:const char *gengetopt_args_info_usage = "Usage: cannyEdge2D_cuda -oSTRING|--output=STRING -iSTRING|--input=STRING \n         [-h|--help] [-V|--version] [-bINT|--blocksize=INT] \n         [-sINT|--octave=INT] [-vINT|--vox=INT]  [FILES]...";
doc/templates/gpu/cannyEdge2D/reduction.cu:    int gpu_result = 0.0f;
doc/templates/gpu/cannyEdge2D/reduction.cu:    CUDA_SAFE_CALL( cudaMalloc((void**) &d_odata, numBlocks*sizeof(int)) );
doc/templates/gpu/cannyEdge2D/reduction.cu:    gpu_result = 0.0f;
doc/templates/gpu/cannyEdge2D/reduction.cu:    // sum partial block sums on GPU
doc/templates/gpu/cannyEdge2D/reduction.cu:    CUDA_SAFE_CALL( cudaMemcpy( &gpu_result, d_odata, sizeof(int), cudaMemcpyDeviceToHost) );
doc/templates/gpu/cannyEdge2D/reduction.cu:    //gpu_result = d_odata[0];
doc/templates/gpu/cannyEdge2D/reduction.cu:    CUDA_SAFE_CALL(cudaFree(d_odata));
doc/templates/gpu/cannyEdge2D/reduction.cu:    return gpu_result;
doc/templates/gpu/cannyEdge2D/cmd_cannyEdge2D_cuda.h:/* cmd_cannyEdge2D_cuda.h */
doc/templates/gpu/cannyEdge2D/cmd_cannyEdge2D_cuda.h:#ifndef CMD_CANNYEDGE2D_CUDA_H
doc/templates/gpu/cannyEdge2D/cmd_cannyEdge2D_cuda.h:#define CMD_CANNYEDGE2D_CUDA_H
doc/templates/gpu/cannyEdge2D/cmd_cannyEdge2D_cuda.h:#define CMDLINE_PARSER_PACKAGE "cannyEdge2D_cuda"
doc/templates/gpu/cannyEdge2D/cmd_cannyEdge2D_cuda.h:#endif /* CMD_CANNYEDGE2D_CUDA_H */
doc/templates/gpu/cannyEdge2D/cannyEdge2D_cuda.ggo:package "cannyEdge2D_cuda"       # don't use package if you're using automake
doc/templates/gpu/cannyEdge2D/Makefile:EXECUTABLE	:= cannyEdge2D_cuda
doc/templates/gpu/cannyEdge2D/Makefile:# Cuda source files (compiled with cudacc)
doc/templates/gpu/cannyEdge2D/Makefile:CUFILES		:= cannyEdge2D_cuda.cu
doc/templates/gpu/cannyEdge2D/Makefile:# CUDA dependency files
doc/templates/gpu/cannyEdge2D/Makefile:CU_DEPS         := cannyEdge2D_cuda_kernel.cu reduction.cu reduction_kernel.cu
doc/templates/gpu/cannyEdge2D/Makefile:CFILES		:= cmd_cannyEdge2D_cuda.c fft_utils.c
doc/templates/gpu/cannyEdge2D/readme.pierre:taille / CPU   / GPU  / acceleration
doc/templates/gpu/cannyEdge2D/cannyEdge2D_host.c: *                      run on host (nothing on GPU)
doc/templates/gpu/cannyEdge2D/cannyEdge2D_host.c: * programme test pour port vers GPU (NVIDIA/CUDA)
doc/templates/gpu/cannyEdge2D/README:# This directory contains a CUDA project for computing 2D Canny edge
doc/templates/gpu/cannyEdge2D/README:# on NVIDIA GPU.
doc/templates/gpu/cannyEdge2D/README:# To compile, copy this directory into NVIDIA's SDK projects sub-directory
doc/templates/gpu/cannyEdge2D/README:# x8 to x11 by running this Canny edge detector on the GPU instead of the CPU.
doc/templates/gpu/cannyEdge2D/README:# to cannyEdge2D_cuda (located in your CUDA SDK binary sub-dir).

```
