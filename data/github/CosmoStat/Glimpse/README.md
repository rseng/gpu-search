# https://github.com/CosmoStat/Glimpse

```console
README.md:## GP-GPU
README.md:Reconstructing a 3D field is very computationally demanding, and using a GPU is higly recommended to speed up the reconstruction. If an installation of CUDA can be
README.md:detected on your system, CMake will automatically compile GPU specific code to
README.md:Glimpse uses peer-to-peer memory transfer between GPUs on a multi-gpu system. You
README.md:may choose at runtime which GPUs to use in your system by providing the -g option:
README.md:This option will use only GPUs 1 and 2 even if more are installed on the system. To
README.md:get a list of CUDA capable devices on your system, use the deviceQuery command
README.md:provided with CUDA.
CMakeLists.txt:		src/gpu_utils.c)
CMakeLists.txt:find_package(CUDA)
CMakeLists.txt:if(${CUDA_FOUND})
CMakeLists.txt:    message("Compiling CUDA accelerated reconstruction code, with 3D support")
CMakeLists.txt:    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_ACC")
CMakeLists.txt:    set(CUDA_NVCC_FLAGS
CMakeLists.txt:    ${CUDA_NVCC_FLAGS};
CMakeLists.txt:    cuda_add_executable(glimpse src/glimpse.cpp ${GLIMPSE_SRC} src/spg.cu src/spg.cpp)
CMakeLists.txt:    cuda_add_cufft_to_target(glimpse)
CMakeLists.txt:else(${CUDA_FOUND})
CMakeLists.txt:    message("Compiling without CUDA acceleration")
CMakeLists.txt:endif(${CUDA_FOUND})
src/glimpse.cpp:#include "gpu_utils.h"
src/glimpse.cpp:    // List of GPU indices
src/glimpse.cpp:#ifdef CUDA_ACC
src/glimpse.cpp:    ("gpu,g", po::value< std::string >(), "comma separated list of GPUs to use (e.g: -g 0,1)")
src/glimpse.cpp:    // In case of GPU acceleration, parse the list of GPUs
src/glimpse.cpp:#ifdef CUDA_ACC
src/glimpse.cpp:    if (vm.count("gpu")) {
src/glimpse.cpp:        // Check that none of the requested GPU id is larger than nGPU
src/glimpse.cpp:        int nGpu;
src/glimpse.cpp:        int whichGPUs[MAX_GPUS];
src/glimpse.cpp:        cudaGetDeviceCount(&nGpu);
src/glimpse.cpp:        boost::split(strs,vm["gpu"].as<std::string>(),boost::is_any_of(",;"));
src/glimpse.cpp:        if(IDlist.size() > MAX_GPUS){
src/glimpse.cpp:            cout << "ERROR: Requested more GPUs than maximum number;"<< endl;
src/glimpse.cpp:            cout << "Maximum size of GPU array " << MAX_GPUS << endl;
src/glimpse.cpp:        if(IDlist.size() > nGpu){
src/glimpse.cpp:            cout << "ERROR: Requested more GPUs than available;"<< endl;
src/glimpse.cpp:            cout << "Number of GPUs available: " << nGpu << endl;
src/glimpse.cpp:            if(IDlist[i] >= nGpu){
src/glimpse.cpp:                cout << "ERROR: Requested GPU id not available;"<< endl;
src/glimpse.cpp:                cout << "Maximum GPU id : " << nGpu - 1 << endl;
src/glimpse.cpp:            whichGPUs[i] = IDlist[i];
src/glimpse.cpp:        setWhichGPUs( IDlist.size(), whichGPUs);
src/glimpse.cpp:    // If CUDA is available, give the option to reconstruct in 3D
src/wavelet_transform.cpp:    // Allocate batch wavelet transform either using fftw or CUDA
src/wavelet_transform.cpp:#ifdef CUDA_ACC
src/wavelet_transform.cpp:    // Look for the number of available GPUs
src/wavelet_transform.cpp:    getDeviceCount(&nGPU);
src/wavelet_transform.cpp:    getGPUs(whichGPUs);
src/wavelet_transform.cpp:    std::cout << "Performing wavelet transform using " << nGPU << " GPUs" <<std::endl; 
src/wavelet_transform.cpp:    // 2 cases: Single GPU or Multiple GPUs
src/wavelet_transform.cpp:    if(nGPU > 1){
src/wavelet_transform.cpp:        ret =cufftXtSetGPUs(fft_plan , nGPU, whichGPUs);
src/wavelet_transform.cpp:        if(ret != 0) std::cout <<"set gpus" << ret << std::endl;
src/wavelet_transform.cpp:        // Select GPU
src/wavelet_transform.cpp:        cudaSetDevice(whichGPUs[0]);
src/wavelet_transform.cpp:        cudaMalloc(&d_frame, sizeof(cufftComplex)*nlp*nframes*npix*npix);
src/wavelet_transform.cpp:#ifdef CUDA_ACC
src/wavelet_transform.cpp:    if(nGPU>1){
src/wavelet_transform.cpp:     cudaFree(d_frame);
src/wavelet_transform.cpp:#ifdef CUDA_ACC
src/wavelet_transform.cpp:    if(nGPU>1){
src/wavelet_transform.cpp:        cudaMemcpy(d_frame, fft_frame, sizeof(cufftComplex)* npix*npix*nlp*nframes, cudaMemcpyHostToDevice);
src/wavelet_transform.cpp:        cudaMemcpy(fft_frame, d_frame, sizeof(cufftComplex)* npix*npix*nlp*nframes, cudaMemcpyDeviceToHost);
src/wavelet_transform.cpp:#ifdef CUDA_ACC
src/wavelet_transform.cpp:    if(nGPU>1){
src/wavelet_transform.cpp:        cudaMemcpy(d_frame, fft_frame, sizeof(cufftComplex)* npix*npix*nlp*nframes, cudaMemcpyHostToDevice);
src/wavelet_transform.cpp:        cudaMemcpy(fft_frame, d_frame, sizeof(cufftComplex)* npix*npix*nlp*nframes, cudaMemcpyDeviceToHost);
src/wavelet_transform.h:#ifdef CUDA_ACC
src/wavelet_transform.h:#include <cuda_runtime.h>
src/wavelet_transform.h:#include "gpu_utils.h"
src/wavelet_transform.h:#ifdef CUDA_ACC
src/wavelet_transform.h:    cudaLibXtDesc *d_frameXt;
src/wavelet_transform.h:    int nGPU;
src/wavelet_transform.h:    int whichGPUs[MAX_GPUS];
src/wavelet_transform.h:    size_t worksize[MAX_GPUS];
src/spg.cpp:    // Look for the number of available GPUs
src/spg.cpp:    getDeviceCount ( &nGPU );
src/spg.cpp:    getGPUs ( whichGPUs );
src/spg.cpp:    std::cout << "Running SPG algorithm on " << nGPU << " GPUs" << std::endl;
src/spg.cpp:    // Create device pointer arrays to hold the memory space for each GPUs
src/spg.cpp:    d_x     = ( float ** ) malloc ( sizeof ( float * ) * nGPU );
src/spg.cpp:    d_u     = ( float ** ) malloc ( sizeof ( float * ) * nGPU );
src/spg.cpp:    d_u_pos = ( float ** ) malloc ( sizeof ( float * ) * nGPU );
src/spg.cpp:    d_w     = ( float ** ) malloc ( sizeof ( float * ) * nGPU );
src/spg.cpp:    // Stride between coefficients processed by different GPUs
src/spg.cpp:    coeff_stride = ( unsigned long * ) malloc ( sizeof ( unsigned long ) * nGPU );
src/spg.cpp:    coeff_stride_pos = ( unsigned long * ) malloc ( sizeof ( unsigned long ) * nGPU );
src/spg.cpp:    // Memory allocation for all GPUs
src/spg.cpp:    for ( int i = 0; i < nGPU; i++ ) {
src/spg.cpp:        // Set strided data for each GPU, leaving the last GPU to handle any extra coefficients
src/spg.cpp:        coeff_stride[i] = npix * npix * nframes / nGPU;
src/spg.cpp:        coeff_stride_pos[i] = npix * npix / nGPU;
src/spg.cpp:        if ( i == ( nGPU - 1 ) ) {
src/spg.cpp:            coeff_stride[i] += npix * npix * nframes % nGPU;
src/spg.cpp:            coeff_stride_pos[i] += npix * npix % nGPU;
src/spg.cpp:        // Select GPU
src/spg.cpp:        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );
src/spg.cpp:        checkCudaErrors ( cudaMalloc ( ( void ** ) &d_x[i],     sizeof ( float ) * coeff_stride[i] * nz ) );
src/spg.cpp:        checkCudaErrors ( cudaMalloc ( ( void ** ) &d_u[i],     sizeof ( float ) * coeff_stride[i] * nz ) );
src/spg.cpp:        checkCudaErrors ( cudaMalloc ( ( void ** ) &d_u_pos[i],     sizeof ( float ) * coeff_stride_pos[i] * nz ) );
src/spg.cpp:        checkCudaErrors ( cudaMalloc ( ( void ** ) &d_w[i],     sizeof ( float ) * coeff_stride[i] * nz ) );
src/spg.cpp:        checkCudaErrors ( cudaMemset ( d_x[i],    0, sizeof ( float ) * coeff_stride[i] * nz ) );
src/spg.cpp:        checkCudaErrors ( cudaMemset ( d_u[i],    0, sizeof ( float ) * coeff_stride[i] * nz ) );
src/spg.cpp:        checkCudaErrors ( cudaMemset ( d_u_pos[i],    0, sizeof ( float ) * coeff_stride_pos[i] * nz ) );
src/spg.cpp:        checkCudaErrors ( cudaMemset ( d_w[i],    0, sizeof ( float ) * coeff_stride[i] * nz ) );
src/spg.cpp:        // Store the l1_weights for each GPU    
src/spg.cpp:        checkCudaErrors (cudaMemcpy2DAsync(d_w[i], coeff_stride[i]*sizeof ( float ), &l1_weights[i * coeff_stride[0]], npix * npix * nframes * sizeof ( float ), coeff_stride[i]*sizeof ( float ), nz, cudaMemcpyHostToDevice ) );
src/spg.cpp:    // Memory allocation for all GPUs
src/spg.cpp:    for ( int i = 0; i < nGPU; i++ ) {
src/spg.cpp:        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );
src/spg.cpp:    for ( int i = 0; i < nGPU; i++ ) {
src/spg.cpp:        checkCudaErrors ( cudaFree ( d_x[i] ) );
src/spg.cpp:        checkCudaErrors ( cudaFree ( d_u[i] ) );
src/spg.cpp:        checkCudaErrors ( cudaFree ( d_u_pos[i] ) );
src/spg.cpp:        checkCudaErrors ( cudaFree ( d_w[i] ) );
src/spg.cpp:    for ( int i = 0; i < nGPU; i++ ) {
src/spg.cpp:        // Select GPU
src/spg.cpp:        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );
src/spg.cpp:        checkCudaErrors ( cudaMemcpy2DAsync ( d_x[i], coeff_stride_pos[i]*sizeof ( float ), &delta[i * coeff_stride_pos[0]], npix * npix * sizeof ( float ), coeff_stride_pos[i]*sizeof ( float ), nz, cudaMemcpyHostToDevice ) );
src/spg.cpp:    for ( int i = 0; i < nGPU; i++ ) {
src/spg.cpp:        // Select GPU
src/spg.cpp:        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );
src/spg.cpp:    // Wait for all GPUs to be done
src/spg.cpp:    for ( int i = 0; i < nGPU; i++ ) {
src/spg.cpp:        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );
src/spg.cpp:        checkCudaErrors ( cudaMemcpy2DAsync ( &delta[i * coeff_stride_pos[0]], npix * npix * sizeof ( float ), d_x[i], coeff_stride_pos[i]*sizeof ( float ), coeff_stride_pos[i]*sizeof ( float ), nz, cudaMemcpyDeviceToHost ) );
src/spg.cpp:    for ( int i = 0; i < nGPU; i++ ) {
src/spg.cpp:        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );
src/spg.cpp:        checkCudaErrors ( cudaDeviceSynchronize() );
src/spg.cpp:        checkCudaErrors ( cudaPeekAtLastError() );
src/spg.cpp:    for ( int i = 0; i < nGPU; i++ ) {
src/spg.cpp:        // Select GPU
src/spg.cpp:        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );
src/spg.cpp:        checkCudaErrors ( cudaMemcpy2DAsync ( d_x[i], coeff_stride[i]*sizeof ( float ), &alpha[i * coeff_stride[0]], npix * npix * nframes * sizeof ( float ), coeff_stride[i]*sizeof ( float ), nz, cudaMemcpyHostToDevice ) );
src/spg.cpp:    for ( int i = 0; i < nGPU; i++ ) {
src/spg.cpp:        // Select GPU
src/spg.cpp:        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );
src/spg.cpp:    // Wait for all GPUs to be done
src/spg.cpp:    for ( int i = 0; i < nGPU; i++ ) {
src/spg.cpp:        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );
src/spg.cpp:        checkCudaErrors ( cudaMemcpy2DAsync ( &alpha[i * coeff_stride[0]], npix * npix * nframes * sizeof ( float ), d_x[i], coeff_stride[i]*sizeof ( float ), coeff_stride[i] * sizeof ( float ), nz, cudaMemcpyDeviceToHost ) );
src/spg.cpp:    for ( int i = 0; i < nGPU; i++ ) {
src/spg.cpp:        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );
src/spg.cpp:        checkCudaErrors ( cudaDeviceSynchronize() );
src/spg.cpp:        checkCudaErrors ( cudaPeekAtLastError() );
src/spg.cpp:    // Memory allocation for all GPUs
src/spg.cpp:    for ( int i = 0; i < nGPU; i++ ) {
src/spg.cpp:        // Store the l1_weights for each GPU       
src/spg.cpp:        checkCudaErrors (cudaMemcpy2DAsync(d_w[i], coeff_stride[i]*sizeof ( float ), &l1_weights[i * coeff_stride[0]], npix * npix * nframes * sizeof ( float ), coeff_stride[i]*sizeof ( float ), nz, cudaMemcpyHostToDevice ) );
src/spg.cpp:    // Wait for all GPUs to be done
src/spg.cpp:    for ( int i = 0; i < nGPU; i++ ) {
src/spg.cpp:        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );
src/spg.cpp:        checkCudaErrors ( cudaDeviceSynchronize() );
src/spg.cu:#include <cuda.h>
src/spg.cu:    cudaMemcpyToSymbol ( P,   p, nz*nz*sizeof ( float ) );
src/spg.cu:    cudaMemcpyToSymbol ( PP, pp, nz*nz*sizeof ( float ) );
src/gpu_utils.c:#include "gpu_utils.h"
src/gpu_utils.c:int gpuIDs[MAX_GPUS];
src/gpu_utils.c:int gpuCount = 0;
src/helper_timer.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
src/helper_timer.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
src/density_reconstruction.cpp:    #ifdef CUDA_ACC
src/density_reconstruction.cpp:#ifdef CUDA_ACC
src/density_reconstruction.cpp:    // Retrieve which GPUs to use
src/density_reconstruction.cpp:    getDeviceCount(&nGPU);
src/density_reconstruction.cpp:    getGPUs(whichGPUs);
src/density_reconstruction.cpp:    if (nGPU > 1) {
src/density_reconstruction.cpp:        ret = cufftXtSetGPUs(fft_plan , nGPU, whichGPUs);
src/density_reconstruction.cpp:        if (ret != 0) std::cout << "set gpus" << ret << std::endl;
src/density_reconstruction.cpp:        cudaMalloc(&d_frame, sizeof(cufftComplex)*nlp * npix * npix);
src/density_reconstruction.cpp:#ifdef CUDA_ACC
src/density_reconstruction.cpp:    if (nGPU > 1) {
src/density_reconstruction.cpp:        cudaFree(d_frame);
src/density_reconstruction.cpp:#ifdef CUDA_ACC
src/density_reconstruction.cpp:    if (nGPU > 1) {
src/density_reconstruction.cpp:        cudaMemcpy(d_frame, input, sizeof(cufftComplex)* npix * npix * nlp, cudaMemcpyHostToDevice);
src/density_reconstruction.cpp:        cudaMemcpy(output, d_frame, sizeof(cufftComplex)* npix * npix * nlp, cudaMemcpyDeviceToHost);
src/density_reconstruction.cpp:#ifdef CUDA_ACC
src/density_reconstruction.cpp:    if (nGPU > 1) {
src/density_reconstruction.cpp:        cudaMemcpy(d_frame, input, sizeof(cufftComplex)* npix * npix * nlp, cudaMemcpyHostToDevice);
src/density_reconstruction.cpp:        cudaMemcpy(output, d_frame, sizeof(cufftComplex)* npix * npix * nlp, cudaMemcpyDeviceToHost);
src/density_reconstruction.cpp:#ifdef CUDA_ACC
src/density_reconstruction.cpp:#ifdef CUDA_ACC
src/density_reconstruction.cpp:#ifdef CUDA_ACC
src/density_reconstruction.h:#ifdef CUDA_ACC
src/density_reconstruction.h:#include "gpu_utils.h"
src/density_reconstruction.h:#ifdef CUDA_ACC
src/density_reconstruction.h:#ifdef CUDA_ACC
src/density_reconstruction.h:    int nGPU;
src/density_reconstruction.h:    int whichGPUs[MAX_GPUS];
src/density_reconstruction.h:    size_t worksize[MAX_GPUS];
src/density_reconstruction.h:    cudaLibXtDesc *d_frameXt;
src/spg.h:#include <cuda_runtime.h>
src/spg.h:#include <cuda.h>
src/spg.h:#include "gpu_utils.h"
src/spg.h:    int nGPU;
src/spg.h:    int whichGPUs[MAX_GPUS];
src/spg.h:    // Stride between coefficients proccessed by different GPUs
src/spg.h:    /* Initialise cuda accelerated SPG algorithm for evaluating simple
src/gpu_utils.h:#ifndef GPU_UTILS_H
src/gpu_utils.h:#define GPU_UTILS_H
src/gpu_utils.h:#define MAX_GPUS 64
src/gpu_utils.h://TODO: Implement a cleaner way to select GPUs
src/gpu_utils.h:extern int gpuIDs[MAX_GPUS];
src/gpu_utils.h:extern int gpuCount;
src/gpu_utils.h:// Sets the number and IDs of GPUs to use
src/gpu_utils.h:static void setWhichGPUs(int count, int* whichGPUs){
src/gpu_utils.h:    gpuCount = count;
src/gpu_utils.h:        gpuIDs[i] = whichGPUs[i];
src/gpu_utils.h:// Get the number of GPUs to use
src/gpu_utils.h:    count[0] = gpuCount; 
src/gpu_utils.h:// Sets the gpus index in the provided array, 
src/gpu_utils.h:static void getGPUs(int* whichGPUs){
src/gpu_utils.h:    for(i=0; i < gpuCount; i++){
src/gpu_utils.h:        whichGPUs[i] = gpuIDs[i];
src/spg.cuh:#include <cuda_runtime.h>
src/spg.cuh:#include <cuda.h>
src/spg.cuh:#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
src/spg.cuh:inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
src/spg.cuh:    if (code != cudaSuccess) {
src/spg.cuh:        std::cerr << "CUDA Error : " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;

```
