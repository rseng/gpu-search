# https://github.com/StephenJHardy/SL1M

```console
radioL1cuda.h:#ifndef RADIOL1CUDA_H
radioL1cuda.h:#define RADIOL1CUDA_H
radioL1cuda.h:namespace radiocuda
radioL1cuda.h:  // Documnetation to their interfaces is found in radioL1cuda.cu
radioL1cuda.h:#endif // #ifndef RADIOL1CUDA_H
radioL1cuda.cu:#include "radioL1cuda.h"
radioL1cuda.cu:#include <cuda.h>
radioL1cuda.cu:// this is the block size for partitioning the visibilities and intensities calculation on the GPU
radioL1cuda.cu:CUDA kernels for calculating the matrix multiplications and transpose multiplications for the 
radioL1cuda.cu:C++ functions for calling the above CUDA kernels, managing memory and multiprocessing across multiple CUDA devices
radioL1cuda.cu:namespace radiocuda
radioL1cuda.cu:// Section 1 - CUDA kernels
radioL1cuda.cu:   Cuda kernel for calculating real intensities from visibilities by multiplying by the
radioL1cuda.cu:   Profiling this code shows 99% utilisation of the GPU.
radioL1cuda.cu:   Cuda kernel for calculating complex visibilities from intensities by multiplying by the
radioL1cuda.cu:   Cuda kernel for calculating real intensities from visibilities by multiplying by the
radioL1cuda.cu:   Cuda kernel for calculating complex visibilities from complex intensities by multiplying by the
radioL1cuda.cu:   Cuda kernel for calculating real intensities from visibilities by multiplying by the
radioL1cuda.cu:   Profiling this code shows 99% utilisation of the GPU.
radioL1cuda.cu:   Cuda kernel for calculating complex visibilities from intensities by multiplying by the
radioL1cuda.cu:   Cuda kernel for calculating complex intensities from visibilities by multiplying by the
radioL1cuda.cu:   Cuda kernel for calculating complex visibilities from complex intensities by multiplying by the
radioL1cuda.cu:// Section 2 - C++ function to call CUDA kernels
radioL1cuda.cu:   Code is structured to make use of 2 GPUs as per AWS GPU instances.
radioL1cuda.cu:   To make it go faster we use CUDA to do the basic calculation. EC2 GPU instances have
radioL1cuda.cu:   a different GPU. Number of threads used by should be set to 2 globally before this
radioL1cuda.cu:  int splitpt = us.size(); // half the calculation goes on each GPU card
radioL1cuda.cu:	int res1 = cudaSetDevice(0);
radioL1cuda.cu:	// extract raw pointers from the device vectors to pass into cuda call
radioL1cuda.cu:	// call the appropriate CUDA kernel
radioL1cuda.cu:// We run this part on the first GPU selected using the thread number
radioL1cuda.cu:	int res1 = cudaSetDevice(0);
radioL1cuda.cu:	// get raw pointers to cuda memory to pass into the cuda function
radioL1cuda.cu:	int res1 = cudaSetDevice(0);
radioL1cuda.cu:	int res1 = cudaSetDevice(omp_get_thread_num());
radioL1cuda.cu:// We run this part on the first GPU selected using the thread number
radioL1cuda.cu:	int res1 = cudaSetDevice(0);
radioL1cuda.cu:	// get raw pointers to cuda memory to pass into the cuda function
radioL1cuda.cu:	int res1 = cudaSetDevice(0);
radioL1cuda.cu:	int res1 = cudaSetDevice(0);
radioL1cuda.cu:	int res1 = cudaSetDevice(0);
radioL1cuda.cu:// These are thrust functors that allow for simple operations on host or device vectors - allows cuda processing without explicitly writing cuda kernels
radioL1cuda.cu:      maxEV = radiocuda::calculateMaxEigenvalueAndVector(100,xsf,ysf,ssf,usf,vsf,wsf,evr,evi);
radioL1cuda.cu:      maxEV = radiocuda::calculateMaxEigenvalueAndVector(100,xsf,ysf,usf,vsf,wsf,evr,evi);
radioL1cuda.cu:    iters = radiocuda::gaussL1Minimisation(lambda,L,maxiters,usf,vsf,wsf,chvsre,chvsim,xsf,ysf,ssf,imout,l1err,l2err);
radioL1cuda.cu:    iters = radiocuda::deltaL1Minimisation(lambda,L,maxiters,usf,vsf,wsf,chvsre,chvsim,xsf,ysf,imout,l1err,l2err);
radioL1cuda.cu:  cudaMalloc((void**)&data, sizeof(cufftComplex)*sz*sz);
radioL1cuda.cu:  cudaMemcpy2D(tmp_d,2*sizeof(tmp_d[0]),rpr, 1*sizeof(rpr[0]),sizeof(rpr[0]),sz*sz,cudaMemcpyHostToDevice);
radioL1cuda.cu:  cudaMemcpy2D(tmp_d+1,2*sizeof(tmp_d[0]),ipr, 1*sizeof(ipr[0]),sizeof(ipr[0]),sz*sz,cudaMemcpyHostToDevice);
radioL1cuda.cu:  cudaMemcpy2D(resr,sizeof(resr[0]),tmp_d,2*sizeof(tmp_d[0]),sizeof(tmp_d[0]),sz*sz,cudaMemcpyDeviceToHost);
radioL1cuda.cu:  cudaFree(data);
Makefile:sl1m: sl1m.o radioL1cuda.o readvises.o
Makefile:	nvcc $(LDFLAGS) -o sl1m sl1m.o radioL1cuda.o readvises.o $(LDLIBS) 
Makefile:sl1m.o: sl1m.cu radioL1cuda.h config.h readvises.h
Makefile:radioL1cuda.o: radioL1cuda.cu radioL1cuda.h config.h
Makefile:	nvcc $(CPPFLAGS) -c radioL1cuda.cu
README.md:The code can currently only be run on a machine with 2 CUDA devices, such as an Amazon GPU instance.
README.md:The easiest way to exercise the code is to spin up a GPU instance on Amazon using the the machine image: ami-2ac85b43 
sl1m.cu:#include <cuda.h>
sl1m.cu:#include "radioL1cuda.h"
sl1m.cu:void configure2devicecuda()
sl1m.cu:	 cudaSetDevice(omp_get_thread_num());
sl1m.cu:	 cudaSetDevice(omp_get_thread_num());
sl1m.cu:      radiocuda::calculateInitialConditionsOnGrid(usf,vsf,wsf,chvsre,chvsim,pixelSize,sz,xinit);
sl1m.cu:    radiocuda::SynthesisL1Minimisation(usf,vsf,wsf,chvsre,chvsim,xsf,ysf,ssf,xinit,xout,maxEV,lambda,maxIters);

```
