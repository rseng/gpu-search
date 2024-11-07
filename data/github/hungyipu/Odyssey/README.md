# https://github.com/hungyipu/Odyssey

```console
Makefile:#before run the Makefile: module load cuda/8.0.44
Makefile:CUDA_PATH = /usr/local/cuda
Makefile:CUDA      = nvcc
Makefile:#CUDAFLAGS = -arch=compute_20 -c
Makefile:CUDAFLAGS = -c 
Makefile:#-Wno-deprecated-gpu-targets
Makefile:	@${CPP} ${CFLAGS} -I. -I${CUDA_PATH}/include ${SRC_PATH}/main.cpp  -o main.cpp.o
Makefile:	@${CPP} ${CFLAGS} -I. -I${CUDA_PATH}/include ${SRC_PATH}/task1.cpp -o task1.cpp.o
Makefile:	@${CPP} ${CFLAGS} -I. -I${CUDA_PATH}/include ${SRC_PATH}/task2.cpp -o task2.cpp.o  
Makefile:	@${CUDA} ${CUDAFLAGS} -I. -I${CUDA_PATH}/include ${SRC_PATH}/Odyssey.cu -o Odyssey.cu.o
Makefile:	@${CPP} -o exec *.o -L${CUDA_PATH}/lib64 -lcudart
README.md:# Odyssey: a GPU-based GRRT code
README.md:Odyssey is a public, GPU-based General Relativistic Radiative Transfer (GRRT) code for computing images and/or spectra in Kerr metric, which described the spacetime aroung a rotating black hole. Implemented in CUDA C/C++, Odyssey is based on the ray-tracing algorithm presented in [Fuerst & Wu (2004)](http://adsabs.harvard.edu/abs/2004A%26A...424..733F), and radiative transfer formulation described in [Younsi, Wu, & Fuerst. (2012)](http://adsabs.harvard.edu/abs/2012A%26A...545A..13Y).
README.md:assign parameters (black hole spin, inclinaiton angle, image size, observed frequency, CUDA configuration...), call Task, then save CUDA computed result to output file<br />
README.md:define functions for setting up CUDA computation for Task1, including `setDims()`, `PRE()`, `GPUcompute()`, and `AFTER()`<br />
README.md:define functions for setting up CUDA computation for Task2<br />
README.md: describe job details of each specific Tasks, such as `__global__ GPU_task1work()`, `__global__ GPU_task1work()`. Computation result will return to **main.cpp**<br />
README.md: and variables which will be saved in the GPU global memory during computation<br />
README.md:|---set CUDA configuration `setDims()`<br />
README.md:|---perform the [*for-loop* for GRRT](https://github.com/hungyipu/Odyssey/wiki/How-Odyssey-Works) `GPUcompute()`<br />
README.md:|---copy memory form device to host and free CUDA memory `AFTER()`<br />
README.md:By calling `GPUcompute()`, the parallel computation will be performed according to the job-detials described inside `__global__ GPU_task1work()` in **Odyssey.cu**.<br />
README.md:`__global__ GPU_task1work()`:<br />
README.md:["Odyssey: A Public GPU-based Code for General-relativistic Radiative Transfer in Kerr Spacetime"](http://adsabs.harvard.edu/abs/2016ApJ...820..105P)<br /> 
src/Odyssey.cu:    This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
src/Odyssey.cu:    "Odyssey: A PUBLIC GPU-BASED CODE FOR GENERAL-RELATIVISTIC RADIATIVE TRANSFER 
src/Odyssey.cu:#include <cuda.h>
src/Odyssey.cu:#include <cuda_runtime.h>
src/Odyssey.cu:__global__ void GPU_task1work(double* ResultsPixel, double* VariablesIn, int GridIdxX, int GridIdxY)
src/Odyssey.cu:void GPU_assigntask1(double* ResultsPixel, double* VariablesIn, int GridIdxX, int GridIdxY,
src/Odyssey.cu:	GPU_task1work<<<GridDim, BlockDim>>>(ResultsPixel, VariablesIn, GridIdxX, GridIdxY);
src/Odyssey.cu:	cudaThreadSynchronize();
src/Odyssey.cu:__global__ void GPU_task2work(double* ResultsPixel, double* VariablesIn, int GridIdxX, int GridIdxY)
src/Odyssey.cu:void GPU_assigntask2(double* ResultsPixel, double* VariablesIn, int GridIdxX, int GridIdxY,
src/Odyssey.cu:	GPU_task2work<<<GridDim, BlockDim>>>(ResultsPixel, VariablesIn, GridIdxX, GridIdxY);
src/Odyssey.cu:	cudaThreadSynchronize();
src/main.cpp:    This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
src/main.cpp:    "Odyssey: A PUBLIC GPU-BASED CODE FOR GENERAL-RELATIVISTIC RADIATIVE TRANSFER 
src/main.cpp:#include <cuda.h>
src/main.cpp:		//assign CUDA congfiguration
src/main.cpp:				mission.GPUCompute(GridIdxX, GridIdxY);
src/main.cpp:		//copy memory form device to host and free CUDA memory
src/main.cpp:		//assign CUDA congfigurations
src/main.cpp:				mission.GPUCompute(GridIdxX, GridIdxY);
src/main.cpp:		//copy memory form device to host and free CUDA memory
src/main.cpp:	cudaSetDevice(0);	
src/Odyssey_def_fun.h:    This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
src/Odyssey_def_fun.h:    "Odyssey: A PUBLIC GPU-BASED CODE FOR GENERAL-RELATIVISTIC RADIATIVE TRANSFER 
src/Odyssey_def.h:    This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
src/Odyssey_def.h:    "Odyssey: A PUBLIC GPU-BASED CODE FOR GENERAL-RELATIVISTIC RADIATIVE TRANSFER 
src/Odyssey_def.h:#include <cuda.h>
src/Odyssey_def.h:#include <cuda_runtime.h>
src/task1.cpp:    This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
src/task1.cpp:    "Odyssey: A PUBLIC GPU-BASED CODE FOR GENERAL-RELATIVISTIC RADIATIVE TRANSFER 
src/task1.cpp:	    cudaMalloc(&d_ResultsPixel           , sizeof(double) * mSize * mSize * 3);
src/task1.cpp:		cudaMalloc(&d_VariablesIn            , sizeof(double) * VarINNUM);
src/task1.cpp:		cudaMemcpy(d_VariablesIn		    , VariablesIn	 , sizeof(double) * VarINNUM	, cudaMemcpyHostToDevice);
src/task1.cpp:		cudaMemcpy(ResultHit, d_ResultsPixel, sizeof(double) * mSize * mSize * 3, cudaMemcpyDeviceToHost);
src/task1.cpp:		cudaFree(d_ResultsPixel);
src/task1.cpp:		cudaFree(d_VariablesIn);
src/task1.cpp:	void GPU_assigntask1(double* ResultsPixel, double* VariablesIn, int GridIdxX, int GridIdxY,
src/task1.cpp:	void mission1::GPUCompute(int GridIdxX, int GridIdxY)
src/task1.cpp:		GPU_assigntask1(d_ResultsPixel, d_VariablesIn, GridIdxX, GridIdxY,
src/task2.cpp:    This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
src/task2.cpp:    "Odyssey: A PUBLIC GPU-BASED CODE FOR GENERAL-RELATIVISTIC RADIATIVE TRANSFER 
src/task2.cpp:	    cudaMalloc(&d_ResultsPixel           , sizeof(double) * mSize * mSize * 3);
src/task2.cpp:		cudaMalloc(&d_VariablesIn        , sizeof(double) * VarINNUM);
src/task2.cpp:		cudaMemcpy(d_VariablesIn	 , VariablesIn	 , sizeof(double) * VarINNUM	, cudaMemcpyHostToDevice);
src/task2.cpp:		cudaMemcpy(ResultHit, d_ResultsPixel, sizeof(double) * mSize * mSize * 3, cudaMemcpyDeviceToHost);
src/task2.cpp:		cudaFree(d_ResultsPixel);
src/task2.cpp:		cudaFree(d_VariablesIn);
src/task2.cpp:	void GPU_assigntask2(double* ResultsPixel, double* VariablesIn, int GridIdxX, int GridIdxY,
src/task2.cpp:	void mission2::GPUCompute(int GridIdxX, int GridIdxY)
src/task2.cpp:		GPU_assigntask2(d_ResultsPixel, d_VariablesIn, GridIdxX, GridIdxY,
src/task2.h:    This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
src/task2.h:    "Odyssey: A PUBLIC GPU-BASED CODE FOR GENERAL-RELATIVISTIC RADIATIVE TRANSFER 
src/task2.h:#include <cuda.h>
src/task2.h:#include <cuda_runtime.h>
src/task2.h:		void GPUCompute(int GridIdxX, int GridIdxY);
src/task1.h:    This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
src/task1.h:    "Odyssey: A PUBLIC GPU-BASED CODE FOR GENERAL-RELATIVISTIC RADIATIVE TRANSFER 
src/task1.h:#include <cuda.h>
src/task1.h:#include <cuda_runtime.h>
src/task1.h:		void GPUCompute(int GridIdxX, int GridIdxY);

```
