# https://github.com/mbejger/polgraw-allsky

```console
search/spotlight/src/lib/yeppp-1.0.0/library/sources/library/CpuArm.cpp:			VendorNVidia,
search/spotlight/src/lib/yeppp-1.0.0/library/sources/library/CpuArm.cpp:				case VendorNVidia:
search/spotlight/src/lib/yeppp-1.0.0/library/sources/library/CpuArm.cpp:					return YEP_MAKE_CONSTANT_STRING("nVidia");
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && !defined(__CUDA_ARCH__)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(__CUDA_ARCH__)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#define YEP_NVIDIA_COMPILER
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(__CUDA_ARCH__)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#define YEP_CUDA_GPU
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(__OPENCL_VERSION__)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#define YEP_OPENCL_DEVICE
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:		#define YEP_OPENCL_CPU
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#elif defined(__GPU__)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:		#define YEP_OPENCL_GPU
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_CUDA_GPU)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#if __CUDA_ARCH__ >= 130
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_OPENCL_DEVICE)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_CUDA_GPU) && (__CUDA_ARCH__ >= 200)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_OPENCL_DEVICE)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_CUDA_GPU)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#if defined(YEP_INTEL_COMPILER) || defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_ARM_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_GCC_COMPATIBLE_COMPILER) || defined(YEP_ARM_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#if defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: *  @note	This module can be used from C, C++, or CUDA.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when nVidia CUDA compiler (nvcc) is used for compilation of GPU-specific code.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox:#define YEP_NVIDIA_COMPILER
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target architecture is a CUDA-enabled GPU.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox:#define YEP_CUDA_GPU
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports misaligned memory access.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports single-precision floating-point operations in hardware.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports double-precision floating-point operations in hardware.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports fused multiply-add instructions in single precision.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports fused multiply-add instructions in double precision.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Supported compilers currently include only Microsoft compiler, Intel compiler, ARM compiler, GCC, Clang, all C99-compatible compilers, and nVidia CUDA compiler.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Supported compilers currently include only Microsoft compiler, Intel compiler, GCC, Clang, and nVidia CUDA compiler.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @warning	For CUDA devices with compute capability 1.x compiler is not guaranteed to honour YEP_NOINLINE specifier.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Supported compilers currently include only Microsoft compiler, Intel compiler, GCC, Clang, all C99-compatible compilers, all C++ compilers, and nVidia CUDA compiler.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Supported compilers currently include only Microsoft compiler, Intel compiler, GCC, Clang, ARM compiler, and nVidia CUDA compiler.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Marks the function as a device function in CUDA. Has no effect in C and C++.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Expands to __device__ when compiled by CUDA compiler for a device. Expands to nothing in all other cases.
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepLibrary.h:		 *  @details	This counter is supported on Intel Sandy Bridge and Ivy Bridge processors, and estimates the energy (in Joules) consumed by power plane 1 (includes GPU cores). */
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepLibrary.h:		 *  @details	This counter is supported on Intel Sandy Bridge and Ivy Bridge processors, and estimates the average power (in Watts) consumed by power plane 1 (includes GPU cores).
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_GCC_COMPATIBLE_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER) && defined(YEP_PROCESSOR_SUPPORTS_DOUBLE_PRECISION_FMA_INSTRUCTIONS)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER) && defined(YEP_PROCESSOR_SUPPORTS_SINGLE_PRECISION_FMA_INSTRUCTIONS)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER) && defined(YEP_CUDA_GPU)
search/spotlight/src/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER) && defined(YEP_CUDA_GPU)
search/network/src-gpu/spline_z.cu:#include <cuda.h>
search/network/src-gpu/spline_z.cu:#include <cuda_runtime_api.h>
search/network/src-gpu/spline_z.cu:void gpu_interp(cufftDoubleComplex *cu_y, int N, 
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaMemset(cu_B, 0, sizeof(cufftDoubleComplex)*(N+1))); //set values to zero
search/network/src-gpu/spline_z.cu:  CudaCheckError();
search/network/src-gpu/spline_z.cu:  CudaCheckError();
search/network/src-gpu/spline_z.cu:  CudaCheckError();
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaMallocHost((void**)&d, sizeof(cufftDoubleComplex)*(N-1)) );
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaMallocHost((void**)&du, sizeof(cufftDoubleComplex)*(N-1)) );
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaMallocHost((void**)&dl, sizeof(cufftDoubleComplex)*(N-1)) );
search/network/src-gpu/spline_z.cu:  //copy to gpu
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaMalloc((void**)cu_d, sizeof(cufftDoubleComplex)*(N-1)));
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaMemcpy(*cu_d, d, sizeof(cufftDoubleComplex)*(N-1), cudaMemcpyHostToDevice));
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaMalloc((void**)cu_dl, sizeof(cufftDoubleComplex)*(N-1)));
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaMemcpy(*cu_dl, dl, sizeof(cufftDoubleComplex)*(N-1), cudaMemcpyHostToDevice));
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaMalloc((void**)cu_du, sizeof(cufftDoubleComplex)*(N-1)));
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaMemcpy(*cu_du, du, sizeof(cufftDoubleComplex)*(N-1), cudaMemcpyHostToDevice));
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaMalloc((void**)cu_B, sizeof(cufftDoubleComplex)*(N+1)));
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaFreeHost(d) );
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaFreeHost(du) );
search/network/src-gpu/spline_z.cu:  CudaSafeCall( cudaFreeHost(dl) );
search/network/src-gpu/jobcore.cu:#include "cuda_error.h"
search/network/src-gpu/jobcore.cu:  //allocate vector for FStat_gpu
search/network/src-gpu/jobcore.cu:  CudaSafeCall ( cudaMalloc((void**)&aux->mu_t_d, sizeof(FLOAT_TYPE)*blocks) );
search/network/src-gpu/jobcore.cu:  CudaSafeCall ( cudaMalloc((void**)&aux->mu_d, sizeof(FLOAT_TYPE)*nav_blocks) );
search/network/src-gpu/jobcore.cu:    modvir_gpu(sinalt, cosalt, sindelt, cosdelt, 
search/network/src-gpu/jobcore.cu:    CudaCheckError();
search/network/src-gpu/jobcore.cu:    cudaDeviceSynchronize();
search/network/src-gpu/jobcore.cu:    CudaCheckError();
search/network/src-gpu/jobcore.cu:    CudaCheckError();
search/network/src-gpu/jobcore.cu:    gpu_interp(fft_arr->xa_d,       // input data
search/network/src-gpu/jobcore.cu:    gpu_interp(fft_arr->xb_d,       // input data
search/network/src-gpu/jobcore.cu:    CudaCheckError();
search/network/src-gpu/jobcore.cu:  CudaCheckError();
search/network/src-gpu/jobcore.cu:  cudaMemcpyToSymbol(maa_d, &_maa, sizeof(double), 0, cudaMemcpyHostToDevice);
search/network/src-gpu/jobcore.cu:  cudaMemcpyToSymbol(mbb_d, &_mbb, sizeof(double), 0, cudaMemcpyHostToDevice);
search/network/src-gpu/jobcore.cu:  CudaCheckError();  
search/network/src-gpu/jobcore.cu:      cudaDeviceSynchronize();
search/network/src-gpu/jobcore.cu:      CudaCheckError();
search/network/src-gpu/jobcore.cu:      CudaCheckError();
search/network/src-gpu/jobcore.cu:#define GPUFSTAT
search/network/src-gpu/jobcore.cu:#ifdef GPUFSTAT
search/network/src-gpu/jobcore.cu:	FStat_gpu(F_d+sett->nmin, sett->nmax - sett->nmin, NAV, aux->mu_d, aux->mu_t_d);
search/network/src-gpu/jobcore.cu:	FStat_gpu_simple(F_d + sett->nmin, sett->nmax - sett->nmin, NAVFSTAT);
search/network/src-gpu/jobcore.cu:      CudaSafeCall ( cudaMemcpy(F, F_d, 2*sett->nfft*sizeof(FLOAT_TYPE), cudaMemcpyDeviceToHost));
search/network/src-gpu/jobcore.cu:      FILE *f1 = fopen("fstat-gpu.dat", "w");
search/network/src-gpu/jobcore.cu:      printf("wrote fstat-gpu.dat | ss=%d  \n", ss);
search/network/src-gpu/jobcore.cu:void modvir_gpu(double sinal, double cosal, double sindel, double cosdel,
search/network/src-gpu/jobcore.cu:void FStat_gpu_simple(FLOAT_TYPE *F_d, int nfft, int nav) {
search/network/src-gpu/jobcore.cu:  CudaCheckError();
search/network/src-gpu/jobcore.cu:#ifdef GPUFSTAT
search/network/src-gpu/jobcore.cu:void FStat_gpu(FLOAT_TYPE *F_d, int N, int nav, FLOAT_TYPE *mu_d, FLOAT_TYPE *mu_t_d) {
search/network/src-gpu/jobcore.cu:  //    CudaSafeCall ( cudaMalloc((void**)&cu_mu_t, sizeof(float)*blocks) );
search/network/src-gpu/jobcore.cu:  //    CudaSafeCall ( cudaMalloc((void**)&cu_mu, sizeof(float)*nav_blocks) );
search/network/src-gpu/jobcore.cu:  CudaCheckError();
search/network/src-gpu/jobcore.cu:  CudaCheckError();
search/network/src-gpu/jobcore.cu:  CudaCheckError();
search/network/src-gpu/test/testrun.sh:../gwsearch-gpu -data ../../../../testdata/test-data-network-injection -ident 205 -label J0322+0441 -band 000 -fpo 1391.3 -r ../../../../testdata/range-J0322+0441.txt
search/network/src-gpu/Makefile:TARGET = gwsearch-gpu
search/network/src-gpu/Makefile:CFLAGS = -DPREFIX="./candidates" -DTIMERS=3 -D$(SINCOS) -DVERBOSE -DCUDA_DEV=0
search/network/src-gpu/Makefile:CFLAGS += -I/usr/local/cuda/include/
search/network/src-gpu/Makefile:LDFLAGS = -L/usr/local/cuda/lib64
search/network/src-gpu/Makefile:	 -Xlinker -Bdynamic -lcufft -lcuda -lcudart -lcusparse -lcublas -lc -lrt -lm
search/network/src-gpu/jobcore.h:void FStat_gpu_simple(FLOAT_TYPE *F_d, int nfft, int nav);
search/network/src-gpu/jobcore.h:void FStat_gpu(FLOAT_TYPE *F_d, int N, int nav, FLOAT_TYPE *mu_d, FLOAT_TYPE *mu_t_d);
search/network/src-gpu/init.cu:#include "cuda_error.h"
search/network/src-gpu/init.cu:#include <cuda_runtime_api.h>
search/network/src-gpu/init.cu:    /// mapped memory works for CUDART_VERSION >= 2020
search/network/src-gpu/init.cu:    CudaSafeCall( cudaHostAlloc((void **)&(ifo[i].sig.xDat), sett->N*sizeof(double), 
search/network/src-gpu/init.cu:				cudaHostAllocMapped) );
search/network/src-gpu/init.cu:    CudaSafeCall( cudaHostGetDevicePointer((void **)&(ifo[i].sig.xDat_d), 
search/network/src-gpu/init.cu:    CudaSafeCall( cudaHostAlloc((void **)&(ifo[i].sig.DetSSB), 3*sett->N*sizeof(double), 
search/network/src-gpu/init.cu:				cudaHostAllocMapped) );
search/network/src-gpu/init.cu:    CudaSafeCall( cudaHostGetDevicePointer((void **)&(ifo[i].sig.DetSSB_d), 
search/network/src-gpu/init.cu:    CudaSafeCall( cudaMalloc((void**)&ifo[i].sig.xDatma_d,
search/network/src-gpu/init.cu:    CudaSafeCall( cudaMalloc((void**)&ifo[i].sig.xDatmb_d, 
search/network/src-gpu/init.cu:    CudaSafeCall( cudaMalloc((void**)&(ifo[i].sig.aa_d), 
search/network/src-gpu/init.cu:    CudaSafeCall( cudaMalloc((void**)&(ifo[i].sig.bb_d), 
search/network/src-gpu/init.cu:    CudaSafeCall( cudaMalloc((void**)&(ifo[i].sig.shft_d), 
search/network/src-gpu/init.cu:    CudaSafeCall( cudaMalloc((void**)&(ifo[i].sig.shftf_d), 
search/network/src-gpu/init.cu:  CudaSafeCall ( cudaMalloc((void **)F_d, 2*sett->nfft*sizeof(double)));
search/network/src-gpu/init.cu:  CudaSafeCall( cudaMalloc((void**)&(aux_arr->t2_d),
search/network/src-gpu/init.cu:  CudaSafeCall( cudaMalloc((void**)&(aux_arr->cosmodf_d), 
search/network/src-gpu/init.cu:  CudaSafeCall( cudaMalloc((void**)&(aux_arr->sinmodf_d), 
search/network/src-gpu/init.cu:  CudaSafeCall( cudaMalloc((void**)&(aux_arr->tshift_d),
search/network/src-gpu/init.cu:  CudaSafeCall ( cudaMalloc((void **)&fft_arr->xa_d, 2*fft_arr->arr_len*sizeof(cufftDoubleComplex)) );
search/network/src-gpu/init.cu:  CudaSafeCall ( cudaMalloc((void **)&fft_arr->xa_d, 2*fft_arr->arr_len*sizeof(cufftDoubleComplex)) );
search/network/src-gpu/init.cu:    CudaSafeCall( cudaFreeHost(ifo[i].sig.xDat) );
search/network/src-gpu/init.cu:    CudaSafeCall( cudaFreeHost(ifo[i].sig.DetSSB) );
search/network/src-gpu/init.cu:    CudaSafeCall( cudaFree(ifo[i].sig.xDatma_d) );
search/network/src-gpu/init.cu:    CudaSafeCall( cudaFree(ifo[i].sig.xDatmb_d) );
search/network/src-gpu/init.cu:    CudaSafeCall( cudaFree(ifo[i].sig.aa_d) );
search/network/src-gpu/init.cu:    CudaSafeCall( cudaFree(ifo[i].sig.bb_d) );
search/network/src-gpu/init.cu:    CudaSafeCall( cudaFree(ifo[i].sig.shft_d) );
search/network/src-gpu/init.cu:    CudaSafeCall( cudaFree(ifo[i].sig.shftf_d) );
search/network/src-gpu/init.cu:  CudaSafeCall( cudaFree(aux->cosmodf_d) );
search/network/src-gpu/init.cu:  CudaSafeCall( cudaFree(aux->sinmodf_d) );
search/network/src-gpu/init.cu:  CudaSafeCall( cudaFree(aux->t2_d) );
search/network/src-gpu/init.cu:  CudaSafeCall( cudaFree(F_d) );
search/network/src-gpu/init.cu:  CudaSafeCall( cudaFree(fft_arr->xa_d) );
search/network/src-gpu/init.cu:  Initialize CUDA: cuinit
search/network/src-gpu/init.cu:  - sets cuda device to (in priority order): cdev, 0 
search/network/src-gpu/init.cu:  cudaDeviceProp deviceProp;
search/network/src-gpu/init.cu:  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
search/network/src-gpu/init.cu:    printf("ERROR: cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
search/network/src-gpu/init.cu:    printf("ERROR: There is no device supporting CUDA\n");
search/network/src-gpu/init.cu:  printf("__________________________________CUDA devices___________________________________\n");
search/network/src-gpu/init.cu:    cudaGetDeviceProperties(&deviceProp, dev);
search/network/src-gpu/init.cu:      cudaSetDevice(cdev);
search/network/src-gpu/init.cu:  cudaSetDeviceFlags(cudaDeviceMapHost);
search/network/src-gpu/init.cu:  cudaThreadSynchronize();
search/network/src-gpu/settings.c:// replaced by modvir_gpu in jobcore.cu
search/network/src-gpu/kernels.cu:  cudaMemcpyToSymbol(amod_d, amod_coeff_tmp, sizeof(Ampl_mod_coeff)*nifo, 
search/network/src-gpu/kernels.cu:		     0, cudaMemcpyHostToDevice);
search/network/src-gpu/kernels.cu:    // no need for this on gpu
search/network/src-gpu/cuda_error.h:#ifndef __CUDA_ERROR_H__
search/network/src-gpu/cuda_error.h:#define __CUDA_ERROR_H__
search/network/src-gpu/cuda_error.h:#define CUDA_ERROR_CHECK
search/network/src-gpu/cuda_error.h:#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
search/network/src-gpu/cuda_error.h:#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
search/network/src-gpu/cuda_error.h:inline void __cudaSafeCall( cudaError err, const char *file, const int line )
search/network/src-gpu/cuda_error.h:#ifdef CUDA_ERROR_CHECK
search/network/src-gpu/cuda_error.h:    if ( cudaSuccess != err )
search/network/src-gpu/cuda_error.h:        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
search/network/src-gpu/cuda_error.h:                 file, line, cudaGetErrorString( err ) );
search/network/src-gpu/cuda_error.h:inline void __cudaCheckError( const char *file, const int line )
search/network/src-gpu/cuda_error.h:#ifdef CUDA_ERROR_CHECK
search/network/src-gpu/cuda_error.h:    cudaError err = cudaGetLastError();
search/network/src-gpu/cuda_error.h:    if ( cudaSuccess != err )
search/network/src-gpu/cuda_error.h:        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
search/network/src-gpu/cuda_error.h:                 file, line, cudaGetErrorString( err ) );
search/network/src-gpu/cuda_error.h:    err = cudaDeviceSynchronize();
search/network/src-gpu/cuda_error.h:    if( cudaSuccess != err )
search/network/src-gpu/cuda_error.h:        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
search/network/src-gpu/cuda_error.h:                 file, line, cudaGetErrorString( err ) );
search/network/src-gpu/spline_z.h:#include <cuda.h>
search/network/src-gpu/spline_z.h:#include <cuda_runtime_api.h>
search/network/src-gpu/spline_z.h:#include "cuda_error.h"
search/network/src-gpu/spline_z.h:void gpu_interp(cufftDoubleComplex *cu_y, int Np, double *cu_new_x, cufftDoubleComplex *cu_new_y,
search/network/src-gpu/runme6d:# Sample run, input data from "../../../testdata/test-gpu-cpu6d/"
search/network/src-gpu/runme6d:./gwsearch-gpu -data ../../../testdata/test-gpu-cpu6d/ -output . -ident 001 -band 0666 -dt 2 --nocheckpoint -range range.dat -usedet H1L1
search/network/src-gpu/runme:# Sample run, input data from "../../../testdata/test-gpu-cpu/"
search/network/src-gpu/runme:./gwsearch-gpu -data ../../../testdata/test-gpu-cpu/ -output . -ident 001 -band 0666 -dt 2 --nocheckpoint -range range.dat -usedet H1L1
search/network/src-gpu/.gitignore:gwsearch-gpu
search/network/src-gpu/main.c:  /* init CUDA device,
search/network/src-gpu/main.c:     CUDA_DEV is set in Makefile */
search/network/src-gpu/main.c:  if (cuinit(CUDA_DEV) == -1) {
search/network/src-gpu/main.c:    printf("\nGPU device initialization error!\n");
search/network/src-gpu/settings.h:void modvir_gpu(
search/network/src-openmp/lib/yeppp-1.0.0/library/sources/library/CpuArm.cpp:			VendorNVidia,
search/network/src-openmp/lib/yeppp-1.0.0/library/sources/library/CpuArm.cpp:				case VendorNVidia:
search/network/src-openmp/lib/yeppp-1.0.0/library/sources/library/CpuArm.cpp:					return YEP_MAKE_CONSTANT_STRING("nVidia");
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && !defined(__CUDA_ARCH__)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(__CUDA_ARCH__)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#define YEP_NVIDIA_COMPILER
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(__CUDA_ARCH__)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#define YEP_CUDA_GPU
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(__OPENCL_VERSION__)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#define YEP_OPENCL_DEVICE
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:		#define YEP_OPENCL_CPU
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#elif defined(__GPU__)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:		#define YEP_OPENCL_GPU
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_CUDA_GPU)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#if __CUDA_ARCH__ >= 130
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_OPENCL_DEVICE)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_CUDA_GPU) && (__CUDA_ARCH__ >= 200)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_OPENCL_DEVICE)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_CUDA_GPU)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#if defined(YEP_INTEL_COMPILER) || defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_ARM_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_GCC_COMPATIBLE_COMPILER) || defined(YEP_ARM_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: *  @note	This module can be used from C, C++, or CUDA.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when nVidia CUDA compiler (nvcc) is used for compilation of GPU-specific code.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox:#define YEP_NVIDIA_COMPILER
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target architecture is a CUDA-enabled GPU.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox:#define YEP_CUDA_GPU
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports misaligned memory access.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports single-precision floating-point operations in hardware.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports double-precision floating-point operations in hardware.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports fused multiply-add instructions in single precision.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports fused multiply-add instructions in double precision.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Supported compilers currently include only Microsoft compiler, Intel compiler, ARM compiler, GCC, Clang, all C99-compatible compilers, and nVidia CUDA compiler.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Supported compilers currently include only Microsoft compiler, Intel compiler, GCC, Clang, and nVidia CUDA compiler.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @warning	For CUDA devices with compute capability 1.x compiler is not guaranteed to honour YEP_NOINLINE specifier.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Supported compilers currently include only Microsoft compiler, Intel compiler, GCC, Clang, all C99-compatible compilers, all C++ compilers, and nVidia CUDA compiler.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Supported compilers currently include only Microsoft compiler, Intel compiler, GCC, Clang, ARM compiler, and nVidia CUDA compiler.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Marks the function as a device function in CUDA. Has no effect in C and C++.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Expands to __device__ when compiled by CUDA compiler for a device. Expands to nothing in all other cases.
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepLibrary.h:		 *  @details	This counter is supported on Intel Sandy Bridge and Ivy Bridge processors, and estimates the energy (in Joules) consumed by power plane 1 (includes GPU cores). */
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepLibrary.h:		 *  @details	This counter is supported on Intel Sandy Bridge and Ivy Bridge processors, and estimates the average power (in Watts) consumed by power plane 1 (includes GPU cores).
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_GCC_COMPATIBLE_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER) && defined(YEP_PROCESSOR_SUPPORTS_DOUBLE_PRECISION_FMA_INSTRUCTIONS)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER) && defined(YEP_PROCESSOR_SUPPORTS_SINGLE_PRECISION_FMA_INSTRUCTIONS)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER) && defined(YEP_CUDA_GPU)
search/network/src-openmp/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER) && defined(YEP_CUDA_GPU)
search/network/Makefile:SUBDIRS = src-cpu src-gpu
search/network/doc/readme_search-gpu.txt:   ./gwsearch-gpu -d . -o ./candidates -i 42 -b 271 -a L1 --whitenoise
search/network/README.md:[one detector](https://github.com/mbejger/polgraw-allsky/tree/one-detector) and [GPU](https://github.com/mbejger/polgraw-allsky/tree/gpu-current) 
search/network/src-cpu/lib/yeppp-1.0.0/library/sources/library/CpuArm.cpp:			VendorNVidia,
search/network/src-cpu/lib/yeppp-1.0.0/library/sources/library/CpuArm.cpp:				case VendorNVidia:
search/network/src-cpu/lib/yeppp-1.0.0/library/sources/library/CpuArm.cpp:					return YEP_MAKE_CONSTANT_STRING("nVidia");
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && !defined(__CUDA_ARCH__)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(__CUDA_ARCH__)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#define YEP_NVIDIA_COMPILER
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(__CUDA_ARCH__)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#define YEP_CUDA_GPU
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(__OPENCL_VERSION__)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#define YEP_OPENCL_DEVICE
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:		#define YEP_OPENCL_CPU
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#elif defined(__GPU__)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:		#define YEP_OPENCL_GPU
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_CUDA_GPU)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:	#if __CUDA_ARCH__ >= 130
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_OPENCL_DEVICE)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_CUDA_GPU) && (__CUDA_ARCH__ >= 200)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_OPENCL_DEVICE)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_CUDA_GPU)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#if defined(YEP_INTEL_COMPILER) || defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_ARM_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#elif defined(YEP_GCC_COMPATIBLE_COMPILER) || defined(YEP_ARM_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: *  @note	This module can be used from C, C++, or CUDA.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when nVidia CUDA compiler (nvcc) is used for compilation of GPU-specific code.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox:#define YEP_NVIDIA_COMPILER
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target architecture is a CUDA-enabled GPU.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox:#define YEP_CUDA_GPU
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports misaligned memory access.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports single-precision floating-point operations in hardware.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports double-precision floating-point operations in hardware.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports fused multiply-add instructions in single precision.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Defined only when the target processor (either CPU or GPU) supports fused multiply-add instructions in double precision.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Supported compilers currently include only Microsoft compiler, Intel compiler, ARM compiler, GCC, Clang, all C99-compatible compilers, and nVidia CUDA compiler.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Supported compilers currently include only Microsoft compiler, Intel compiler, GCC, Clang, and nVidia CUDA compiler.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @warning	For CUDA devices with compute capability 1.x compiler is not guaranteed to honour YEP_NOINLINE specifier.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Supported compilers currently include only Microsoft compiler, Intel compiler, GCC, Clang, all C99-compatible compilers, all C++ compilers, and nVidia CUDA compiler.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Supported compilers currently include only Microsoft compiler, Intel compiler, GCC, Clang, ARM compiler, and nVidia CUDA compiler.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @brief	Marks the function as a device function in CUDA. Has no effect in C and C++.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepPredefines.dox: * @details	Expands to __device__ when compiled by CUDA compiler for a device. Expands to nothing in all other cases.
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepLibrary.h:		 *  @details	This counter is supported on Intel Sandy Bridge and Ivy Bridge processors, and estimates the energy (in Joules) consumed by power plane 1 (includes GPU cores). */
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepLibrary.h:		 *  @details	This counter is supported on Intel Sandy Bridge and Ivy Bridge processors, and estimates the average power (in Watts) consumed by power plane 1 (includes GPU cores).
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_GCC_COMPATIBLE_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_GNU_COMPILER) || defined(YEP_CLANG_COMPILER) || defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER) && defined(YEP_PROCESSOR_SUPPORTS_DOUBLE_PRECISION_FMA_INSTRUCTIONS)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER) && defined(YEP_PROCESSOR_SUPPORTS_SINGLE_PRECISION_FMA_INSTRUCTIONS)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#if defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER) && defined(YEP_CUDA_GPU)
search/network/src-cpu/lib/yeppp-1.0.0/library/headers/yepBuiltin.h:#elif defined(YEP_NVIDIA_COMPILER) && defined(YEP_CUDA_GPU)
search/one-detector/src-gpu/struct.h:#include <cuda.h>
search/one-detector/src-gpu/struct.h:#include <cuda_runtime_api.h>
search/one-detector/src-gpu/struct.h:	//auxiliary arrays used in modvir_gpu
search/one-detector/src-gpu/spline_z.cu:#include <cuda.h>
search/one-detector/src-gpu/spline_z.cu:#include <cuda_runtime_api.h>
search/one-detector/src-gpu/spline_z.cu:void gpu_interp(cufftDoubleComplex *cu_y, int N, double *cu_new_x, cufftDoubleComplex *cu_new_y,
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaMemset(cu_B, 0, sizeof(cufftDoubleComplex)*(N+1))); //set values to zero
search/one-detector/src-gpu/spline_z.cu:	CudaCheckError();
search/one-detector/src-gpu/spline_z.cu:	CudaCheckError();
search/one-detector/src-gpu/spline_z.cu:	CudaCheckError();
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaMallocHost((void**)&d, sizeof(cufftDoubleComplex)*(N-1)) );
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaMallocHost((void**)&du, sizeof(cufftDoubleComplex)*(N-1)) );
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaMallocHost((void**)&dl, sizeof(cufftDoubleComplex)*(N-1)) );
search/one-detector/src-gpu/spline_z.cu:	//copy to gpu
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaMalloc((void**)cu_d, sizeof(cufftDoubleComplex)*(N-1)));
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaMemcpy(*cu_d, d, sizeof(cufftDoubleComplex)*(N-1), cudaMemcpyHostToDevice));
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaMalloc((void**)cu_dl, sizeof(cufftDoubleComplex)*(N-1)));
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaMemcpy(*cu_dl, dl, sizeof(cufftDoubleComplex)*(N-1), cudaMemcpyHostToDevice));
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaMalloc((void**)cu_du, sizeof(cufftDoubleComplex)*(N-1)));
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaMemcpy(*cu_du, du, sizeof(cufftDoubleComplex)*(N-1), cudaMemcpyHostToDevice));
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaMalloc((void**)cu_B, sizeof(cufftDoubleComplex)*(N+1)));
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaFreeHost(d) );
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaFreeHost(du) );
search/one-detector/src-gpu/spline_z.cu:	CudaSafeCall( cudaFreeHost(dl) );
search/one-detector/src-gpu/jobcore.cu:#include <cuda.h>
search/one-detector/src-gpu/jobcore.cu:#include <cuda_runtime_api.h>
search/one-detector/src-gpu/jobcore.cu:  //allocate vector for FStat_gpu
search/one-detector/src-gpu/jobcore.cu:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_mu_t, sizeof(FLOAT_TYPE)*blocks) );
search/one-detector/src-gpu/jobcore.cu:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_mu, sizeof(FLOAT_TYPE)*nav_blocks) );
search/one-detector/src-gpu/jobcore.cu:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_o_aa, sizeof(double)*modvir_blocks));
search/one-detector/src-gpu/jobcore.cu:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_o_bb, sizeof(double)*modvir_blocks));
search/one-detector/src-gpu/jobcore.cu:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_o_aa2, sizeof(double)*modvir_blocks));
search/one-detector/src-gpu/jobcore.cu:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_o_bb2, sizeof(double)*modvir_blocks));
search/one-detector/src-gpu/jobcore.cu:  cudaFree(arr->cu_mu);
search/one-detector/src-gpu/jobcore.cu:  cudaFree(arr->cu_mu_t);
search/one-detector/src-gpu/jobcore.cu:  CudaSafeCall ( cudaFree(arr->cu_o_aa) );
search/one-detector/src-gpu/jobcore.cu:  CudaSafeCall ( cudaFree(arr->cu_o_bb) );
search/one-detector/src-gpu/jobcore.cu:  CudaSafeCall ( cudaFree(arr->cu_o_aa2) );
search/one-detector/src-gpu/jobcore.cu:  CudaSafeCall ( cudaFree(arr->cu_o_bb2) );
search/one-detector/src-gpu/jobcore.cu:  modvir_gpu(sinalt, cosalt, sindelt, cosdelt, sett->sphir,
search/one-detector/src-gpu/jobcore.cu:  CudaCheckError();
search/one-detector/src-gpu/jobcore.cu:  cudaDeviceSynchronize();
search/one-detector/src-gpu/jobcore.cu:  CudaCheckError();
search/one-detector/src-gpu/jobcore.cu:  CudaCheckError();
search/one-detector/src-gpu/jobcore.cu:  gpu_interp(arr->cu_xa,	//input data
search/one-detector/src-gpu/jobcore.cu:  gpu_interp(arr->cu_xb,	//input data
search/one-detector/src-gpu/jobcore.cu:	CudaCheckError();
search/one-detector/src-gpu/jobcore.cu:	CudaCheckError();
search/one-detector/src-gpu/jobcore.cu:	CudaCheckError();
search/one-detector/src-gpu/jobcore.cu:	CudaCheckError();
search/one-detector/src-gpu/jobcore.cu:      CudaCheckError();
search/one-detector/src-gpu/jobcore.cu:	FStat_gpu(cu_F+sett->nmin, sett->nmax - sett->nmin, NAV, arr->cu_mu, arr->cu_mu_t);
search/one-detector/src-gpu/jobcore.cu:      cudaMemset(arr->cu_cand_count, 0, sizeof(int));
search/one-detector/src-gpu/jobcore.cu:      cudaMemcpy(&cand_count, arr->cu_cand_count, sizeof(int), cudaMemcpyDeviceToHost);
search/one-detector/src-gpu/jobcore.cu:	cudaMemcpy(arr->cu_cand_buffer + *cand_buffer_count * NPAR,
search/one-detector/src-gpu/jobcore.cu:		   cudaMemcpyDeviceToDevice);
search/one-detector/src-gpu/jobcore.cu:  CudaSafeCall
search/one-detector/src-gpu/jobcore.cu:    ( cudaMemcpy(cand_buffer, cu_cand_buffer, sizeof(FLOAT_TYPE)*NPAR * (*cand_buffer_count),
search/one-detector/src-gpu/jobcore.cu:		 cudaMemcpyDeviceToHost)
search/one-detector/src-gpu/jobcore.cu:void modvir_gpu (double sinal, double cosal, double sindel, double cosdel,
search/one-detector/src-gpu/jobcore.cu:	CudaCheckError();
search/one-detector/src-gpu/jobcore.cu:		CudaCheckError();
search/one-detector/src-gpu/jobcore.cu:	CudaSafeCall ( cudaMemcpy(&s_a, outa, sizeof(double), cudaMemcpyDeviceToHost));
search/one-detector/src-gpu/jobcore.cu:	CudaSafeCall ( cudaMemcpy(&s_b, outb, sizeof(double), cudaMemcpyDeviceToHost));
search/one-detector/src-gpu/jobcore.cu:  //	printf("Sa, Sb: %e %e (GPU)\n", s_a, s_b);
search/one-detector/src-gpu/jobcore.cu:  CudaCheckError();
search/one-detector/src-gpu/jobcore.cu:void FStat_gpu(FLOAT_TYPE *cu_F, int N, int nav, float *cu_mu, float *cu_mu_t) {
search/one-detector/src-gpu/jobcore.cu:  //	CudaSafeCall ( cudaMalloc((void**)&cu_mu_t, sizeof(float)*blocks) );
search/one-detector/src-gpu/jobcore.cu:  //	CudaSafeCall ( cudaMalloc((void**)&cu_mu, sizeof(float)*nav_blocks) );
search/one-detector/src-gpu/jobcore.cu:  CudaCheckError();
search/one-detector/src-gpu/jobcore.cu:  CudaCheckError();
search/one-detector/src-gpu/jobcore.cu:  CudaCheckError();
search/one-detector/src-gpu/Makefile:CFLAGS = -O3 -g -I/usr/local/cuda/include/
search/one-detector/src-gpu/Makefile:LDFLAGS = -L/usr/local/cuda/lib -L/usr/local/cuda/lib64
search/one-detector/src-gpu/Makefile:LIBS = -static -lgsl -lgslcblas -Wl,-Bdynamic -lcufft -lcuda -lcudart -lcusparse -lcublas -lc -lrt -lm
search/one-detector/src-gpu/Makefile:TARGET = gwsearch-gpu
search/one-detector/src-gpu/jobcore.h:			FLOAT_TYPE *cu_F,			// F-stat on GPU
search/one-detector/src-gpu/jobcore.h:void modvir_gpu (double sinal, double cosal, double sindel, double cosdel,
search/one-detector/src-gpu/jobcore.h:void FStat_gpu(FLOAT_TYPE *cu_F, int N, int nav, FLOAT_TYPE *cu_mu, FLOAT_TYPE *cu_mu_t);
search/one-detector/src-gpu/settings.c:#include "cuda_error.h"
search/one-detector/src-gpu/settings.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_sinmodf, sizeof(double)*N));
search/one-detector/src-gpu/settings.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_cosmodf, sizeof(double)*N));
search/one-detector/src-gpu/settings.c:  CudaSafeCall ( cudaMemcpy(arr->cu_sinmodf, arr->sinmodf, sizeof(double)*N, cudaMemcpyHostToDevice));
search/one-detector/src-gpu/settings.c:  CudaSafeCall ( cudaMemcpy(arr->cu_cosmodf, arr->cosmodf, sizeof(double)*N, cudaMemcpyHostToDevice));
search/one-detector/src-gpu/kernels.cu:  cudaMemcpyToSymbol(cu_c, amod_coeff_tmp, sizeof(double)*9);
search/one-detector/src-gpu/cuda_error.h:#ifndef __CUDA_ERROR_H__
search/one-detector/src-gpu/cuda_error.h:#define __CUDA_ERROR_H__
search/one-detector/src-gpu/cuda_error.h:#define CUDA_ERROR_CHECK
search/one-detector/src-gpu/cuda_error.h:#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
search/one-detector/src-gpu/cuda_error.h:#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
search/one-detector/src-gpu/cuda_error.h:inline void __cudaSafeCall( cudaError err, const char *file, const int line )
search/one-detector/src-gpu/cuda_error.h:#ifdef CUDA_ERROR_CHECK
search/one-detector/src-gpu/cuda_error.h:    if ( cudaSuccess != err )
search/one-detector/src-gpu/cuda_error.h:        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
search/one-detector/src-gpu/cuda_error.h:                 file, line, cudaGetErrorString( err ) );
search/one-detector/src-gpu/cuda_error.h:inline void __cudaCheckError( const char *file, const int line )
search/one-detector/src-gpu/cuda_error.h:#ifdef CUDA_ERROR_CHECK
search/one-detector/src-gpu/cuda_error.h:    cudaError err = cudaGetLastError();
search/one-detector/src-gpu/cuda_error.h:    if ( cudaSuccess != err )
search/one-detector/src-gpu/cuda_error.h:        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
search/one-detector/src-gpu/cuda_error.h:                 file, line, cudaGetErrorString( err ) );
search/one-detector/src-gpu/cuda_error.h:    err = cudaDeviceSynchronize();
search/one-detector/src-gpu/cuda_error.h:    if( cudaSuccess != err )
search/one-detector/src-gpu/cuda_error.h:        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
search/one-detector/src-gpu/cuda_error.h:                 file, line, cudaGetErrorString( err ) );
search/one-detector/src-gpu/spline_z.h:#include <cuda.h>
search/one-detector/src-gpu/spline_z.h:#include <cuda_runtime_api.h>
search/one-detector/src-gpu/spline_z.h:#include "cuda_error.h"
search/one-detector/src-gpu/spline_z.h:void gpu_interp(cufftDoubleComplex *cu_y, int Np, double *cu_new_x, cufftDoubleComplex *cu_new_y,
search/one-detector/src-gpu/init.c:#include "cuda_error.h"
search/one-detector/src-gpu/init.c:	CudaSafeCall( cudaMallocHost((void**)&arr->xDat, sizeof(double)*sett->N));
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_xDat, sizeof(double)*sett->N));
search/one-detector/src-gpu/init.c:	CudaSafeCall( cudaMallocHost((void**)&arr->DetSSB, sizeof(double)*3*sett->N) );
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_DetSSB, sizeof(double)*3*sett->N));
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)cu_F, sizeof(FLOAT_TYPE)*sett->fftpad*sett->nfft));
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMemset(*cu_F, 0, sizeof(FLOAT_TYPE)*sett->fftpad*sett->nfft));
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMemcpy(arr->cu_xDat, arr->xDat, sizeof(double)*sett->N, cudaMemcpyHostToDevice));
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMemcpy(arr->cu_DetSSB, arr->DetSSB, sizeof(double)*sett->N*3, cudaMemcpyHostToDevice));
search/one-detector/src-gpu/init.c:  CudaSafeCall( cudaMallocHost((void**)&arr->aa, sizeof(double)*sett->N) );
search/one-detector/src-gpu/init.c:  CudaSafeCall( cudaMallocHost((void**)&arr->bb, sizeof(double)*sett->N) );
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_aa, sizeof(double)*sett->nfft));
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_bb, sizeof(double)*sett->nfft));
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_shft, sizeof(double)*sett->N));
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_shftf, sizeof(double)*sett->N));
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_tshift, sizeof(double)*sett->N));
search/one-detector/src-gpu/init.c:  CudaSafeCall (cudaMalloc((void**)&arr->cu_cand_params, sizeof(FLOAT_TYPE)*arr->cand_params_size));
search/one-detector/src-gpu/init.c:  CudaSafeCall (cudaMalloc((void**)&arr->cu_cand_buffer, sizeof(FLOAT_TYPE)*arr->cand_buffer_size));
search/one-detector/src-gpu/init.c:  CudaSafeCall (cudaMalloc((void**)&arr->cu_cand_count, sizeof(int)));
search/one-detector/src-gpu/init.c:	CudaSafeCall( cudaMallocHost((void**)&arr->cand_buffer, sizeof(FLOAT_TYPE)*arr->cand_buffer_size) );
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_xa, arr->arr_len*sizeof(cufftDoubleComplex)) );
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_xb, arr->arr_len*sizeof(cufftDoubleComplex)) );
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_xar, arr->arr_len*sizeof(cufftDoubleComplex)) );
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_xbr, arr->arr_len*sizeof(cufftDoubleComplex)) );
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_xa_f, arr->arr_len*sizeof(COMPLEX_TYPE)) );
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_xb_f, arr->arr_len*sizeof(COMPLEX_TYPE)) );
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_xar_f, arr->arr_len*sizeof(COMPLEX_TYPE)) );
search/one-detector/src-gpu/init.c:  CudaSafeCall ( cudaMalloc((void**)&arr->cu_xbr_f, arr->arr_len*sizeof(COMPLEX_TYPE)) );
search/one-detector/src-gpu/init.c:    CudaSafeCall ( cudaMalloc((void**)&arr->cu_xa2_f, sett->nfft*sizeof(COMPLEX_TYPE)) );
search/one-detector/src-gpu/init.c:    CudaSafeCall ( cudaMalloc((void**)&arr->cu_xb2_f, sett->nfft*sizeof(COMPLEX_TYPE)) );
search/one-detector/src-gpu/init.c:	CudaSafeCall( cudaFreeHost(arr->xDat) );
search/one-detector/src-gpu/init.c:  CudaSafeCall( cudaFreeHost(arr->DetSSB) );
search/one-detector/src-gpu/init.c:	CudaSafeCall( cudaFreeHost(arr->aa) );
search/one-detector/src-gpu/init.c:  CudaSafeCall( cudaFreeHost(arr->bb) );
search/one-detector/src-gpu/init.c:  CudaSafeCall( cudaFreeHost(arr->cand_buffer) );
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_xa);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_xb);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_xar);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_xbr);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_xa_f);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_xb_f);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_xar_f);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_xbr_f);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_xDat);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_aa);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_bb);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_shft);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_shftf);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_tshift);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_DetSSB);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_d);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_dl);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_du);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_B);
search/one-detector/src-gpu/init.c:  cudaFree(cu_F);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_sinmodf);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_cosmodf);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_cand_buffer);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_cand_params);
search/one-detector/src-gpu/init.c:  cudaFree(arr->cu_cand_count);
search/one-detector/src-gpu/init.c:    cudaFree(arr->cu_xa2_f);
search/one-detector/src-gpu/init.c:    cudaFree(arr->cu_xb2_f);
search/one-detector/src-gpu/.gitignore:gwsearch-gpu
search/one-detector/src-gpu/settings.h:#include <cuda.h>
search/one-detector/src-gpu/settings.h:#include <cuda_runtime_api.h>
search/one-detector/Makefile:SUBDIRS = src src-gpu
search/one-detector/doc/readme_search-gpu.txt:   ./gwsearch-gpu -d . -o ./candidates -i 42 -b 271 -a L1 --whitenoise

```
