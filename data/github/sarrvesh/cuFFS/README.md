# https://github.com/sarrvesh/cuFFS

```console
README.md:# CUDA-accelerated Fast Faraday Synthesis (cuFFS)
README.md:* [nvcc](docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) (Need both the driver and the toolkit)
CHANGELOG:V0.3     Code tested using a single GPU on the Bracewell cluster.
CMakeLists.txt:find_package(CUDA REQUIRED)
CMakeLists.txt:# Build the GPU code
CMakeLists.txt:LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_50")
CMakeLists.txt:cuda_add_executable(rmsynthesis ${RMSYNTHESIS})
parsetFile:// How many GPUs to use?
parsetFile:nGPU = 1;
parsetFile:qCubeName = "/home/sarrvesh/Work/RMSynth_GPU/test_wsrt/q.rot.fits";
parsetFile:uCubeName = "/home/sarrvesh/Work/RMSynth_GPU/test_wsrt/u.rot.fits";
parsetFile:freqFileName = "/home/sarrvesh/Work/RMSynth_GPU/test_wsrt/freqTable";
build.sh:#Rootdir of CUDA Toolkit
build.sh:CUDA_PATH="/home/sarrvesh/Documents/cuda"
build.sh:nvcc -O3 -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -I${HDF5_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -L${CUDA_PATH}/lib64/ -c src/rmsynthesis/devices.cu -lhdf5 -gencode $NVCC_FLAGS
build.sh:nvcc -O3 -I${CUDA_PATH}/include/ -I${LIB_CONFIG_PATH}/include/ -I${CFITSIO_PATH}/include/ -I${HDF5_PATH}/include/ -L${LIB_CONFIG_PATH}/lib/ -L/${CFITSIO_PATH}/lib/ -L${CUDA_PATH}/lib64/ -L${HDF5_PATH}/lib/ -o rmsynthesis rmsynthesis.o devices.o fileaccess.o inputparser.o rmsf.o -lconfig -lcfitsio -lcudart -lm -lhdf5 -lhdf5_hl -gencode $NVCC_FLAGS
src/rmsynthesis/structures.h:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis/structures.h:    int nGPU;
src/rmsynthesis/structures.h:/* Structure to store useful GPU device information */
src/rmsynthesis/devices.cu:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis/devices.cu:#include<cuda_runtime.h>
src/rmsynthesis/devices.cu:#include<cuda.h>
src/rmsynthesis/devices.cu:* Check if CUDA ERROR flag has been set. If raised, print 
src/rmsynthesis/devices.cu:void checkCudaError() {
src/rmsynthesis/devices.cu:    cudaError_t errorID = cudaGetLastError();
src/rmsynthesis/devices.cu:    if(errorID != cudaSuccess) {
src/rmsynthesis/devices.cu:        printf("\nERROR: %s", cudaGetErrorString(errorID));
src/rmsynthesis/devices.cu:* Check for valid CUDA supported devices. If detected, 
src/rmsynthesis/devices.cu:    struct cudaDeviceProp deviceProp;
src/rmsynthesis/devices.cu:    struct deviceInfoList *gpuList;
src/rmsynthesis/devices.cu:    cudaDeviceReset();
src/rmsynthesis/devices.cu:    cudaGetDeviceCount(&deviceCount);
src/rmsynthesis/devices.cu:    checkCudaError();
src/rmsynthesis/devices.cu:        printf("\nError: Could not detect CUDA supported GPU(s)\n\n");
src/rmsynthesis/devices.cu:    printf("INFO: Detected %d CUDA-supported GPU(s)\n", deviceCount);
src/rmsynthesis/devices.cu:    /* Store useful information about each GPU in a structure array */
src/rmsynthesis/devices.cu:    gpuList = (deviceInfoList *)malloc(deviceCount * 
src/rmsynthesis/devices.cu:        cudaSetDevice(dev);
src/rmsynthesis/devices.cu:        cudaGetDeviceProperties(&deviceProp, dev);
src/rmsynthesis/devices.cu:        checkCudaError();
src/rmsynthesis/devices.cu:        gpuList[dev].deviceID    = dev;
src/rmsynthesis/devices.cu:        gpuList[dev].globalMem   = deviceProp.totalGlobalMem;
src/rmsynthesis/devices.cu:        gpuList[dev].constantMem = deviceProp.totalConstMem;
src/rmsynthesis/devices.cu:        gpuList[dev].sharedMemPerBlock = deviceProp.sharedMemPerBlock;
src/rmsynthesis/devices.cu:        gpuList[dev].maxThreadPerMP = deviceProp.maxThreadsPerMultiProcessor;
src/rmsynthesis/devices.cu:        gpuList[dev].maxThreadPerBlock = deviceProp.maxThreadsPerBlock;
src/rmsynthesis/devices.cu:        gpuList[dev].threadBlockSize[0] = deviceProp.maxThreadsDim[0];
src/rmsynthesis/devices.cu:        gpuList[dev].threadBlockSize[1] = deviceProp.maxThreadsDim[1];
src/rmsynthesis/devices.cu:        gpuList[dev].threadBlockSize[2] = deviceProp.maxThreadsDim[2];
src/rmsynthesis/devices.cu:        gpuList[dev].warpSize           = deviceProp.warpSize;
src/rmsynthesis/devices.cu:        gpuList[dev].nSM                = deviceProp.multiProcessorCount;
src/rmsynthesis/devices.cu:        printf("\n\tGlobal memory: %f MB", gpuList[dev].globalMem/MEGA);
src/rmsynthesis/devices.cu:        printf("\n\tShared memory: %f kB", gpuList[dev].sharedMemPerBlock/KILO);
src/rmsynthesis/devices.cu:        printf("\n\tMax threads per block: %d", gpuList[dev].maxThreadPerBlock);
src/rmsynthesis/devices.cu:        printf("\n\tMax threads per MP: %d", gpuList[dev].maxThreadPerMP);
src/rmsynthesis/devices.cu:    return(gpuList);
src/rmsynthesis/devices.cu:* Select the best GPU device
src/rmsynthesis/devices.cu:int getBestDevice(struct deviceInfoList *gpuList, int nDevices) {
src/rmsynthesis/devices.cu:        maxMem = gpuList[dev].globalMem;
src/rmsynthesis/devices.cu:            if(maxMem < gpuList[i].globalMem) { 
src/rmsynthesis/devices.cu:                maxMem = gpuList[i].globalMem;
src/rmsynthesis/devices.cu:* Copy GPU device information of selectedDevice from gpuList 
src/rmsynthesis/devices.cu:struct deviceInfoList copySelectedDeviceInfo(struct deviceInfoList *gpuList, 
src/rmsynthesis/devices.cu:    selectedDeviceInfo.deviceID           = gpuList[i].deviceID;
src/rmsynthesis/devices.cu:    selectedDeviceInfo.globalMem          = gpuList[i].globalMem;
src/rmsynthesis/devices.cu:    selectedDeviceInfo.constantMem        = gpuList[i].constantMem;
src/rmsynthesis/devices.cu:    selectedDeviceInfo.sharedMemPerBlock  = gpuList[i].sharedMemPerBlock;
src/rmsynthesis/devices.cu:    selectedDeviceInfo.maxThreadPerMP     = gpuList[i].maxThreadPerMP;
src/rmsynthesis/devices.cu:    selectedDeviceInfo.maxThreadPerBlock  = gpuList[i].maxThreadPerBlock;
src/rmsynthesis/devices.cu:    selectedDeviceInfo.threadBlockSize[0] = gpuList[i].threadBlockSize[0];
src/rmsynthesis/devices.cu:    selectedDeviceInfo.threadBlockSize[1] = gpuList[i].threadBlockSize[1];
src/rmsynthesis/devices.cu:    selectedDeviceInfo.threadBlockSize[2] = gpuList[i].threadBlockSize[2];
src/rmsynthesis/devices.cu:    selectedDeviceInfo.warpSize           = gpuList[i].warpSize;
src/rmsynthesis/devices.cu:    selectedDeviceInfo.nSM                = gpuList[i].nSM;
src/rmsynthesis/devices.cu:* GPU accelerated RM Synthesis function
src/rmsynthesis/devices.cu:    cudaMalloc(&d_lambdaDiff2, sizeof(*lambdaDiff2)*params->qAxisLen3);
src/rmsynthesis/devices.cu:    cudaMalloc(&d_phiAxis, sizeof(*(params->phiAxis))*inOptions->nPhi);
src/rmsynthesis/devices.cu:    cudaMalloc(&d_qImageArray, nInElements*sizeof(*qImageArray));
src/rmsynthesis/devices.cu:    cudaMalloc(&d_uImageArray, nInElements*sizeof(*uImageArray));
src/rmsynthesis/devices.cu:    cudaMalloc(&d_qPhi, nOutElements*sizeof(*qPhi));
src/rmsynthesis/devices.cu:    cudaMalloc(&d_uPhi, nOutElements*sizeof(*uPhi));
src/rmsynthesis/devices.cu:    cudaMalloc(&d_pPhi, nOutElements*sizeof(*pPhi));
src/rmsynthesis/devices.cu:    checkCudaError();
src/rmsynthesis/devices.cu:    cudaMemcpy(d_lambdaDiff2, lambdaDiff2, 
src/rmsynthesis/devices.cu:               sizeof(*lambdaDiff2)*params->qAxisLen3, cudaMemcpyHostToDevice);
src/rmsynthesis/devices.cu:    checkCudaError();
src/rmsynthesis/devices.cu:    cudaMemcpy(d_phiAxis, params->phiAxis, 
src/rmsynthesis/devices.cu:               cudaMemcpyHostToDevice);
src/rmsynthesis/devices.cu:    checkCudaError();
src/rmsynthesis/devices.cu:    //cudaEventRecord(totStart);
src/rmsynthesis/devices.cu:       cudaMemcpy(d_qImageArray, qImageArray, 
src/rmsynthesis/devices.cu:                  cudaMemcpyHostToDevice);
src/rmsynthesis/devices.cu:       cudaMemcpy(d_uImageArray, uImageArray, 
src/rmsynthesis/devices.cu:                  cudaMemcpyHostToDevice);
src/rmsynthesis/devices.cu:       cudaThreadSynchronize();
src/rmsynthesis/devices.cu:       cudaMemcpy(qPhi, d_qPhi, nOutElements*sizeof(*qPhi), cudaMemcpyDeviceToHost);
src/rmsynthesis/devices.cu:       cudaMemcpy(uPhi, d_uPhi, nOutElements*sizeof(*qPhi), cudaMemcpyDeviceToHost);
src/rmsynthesis/devices.cu:       cudaMemcpy(pPhi, d_pPhi, nOutElements*sizeof(*qPhi), cudaMemcpyDeviceToHost);
src/rmsynthesis/devices.cu:    cudaFree(d_qImageArray); cudaFree(d_uImageArray);
src/rmsynthesis/devices.cu:    cudaFree(d_qPhi); cudaFree(d_uPhi); cudaFree(d_pPhi);
src/rmsynthesis/devices.cu:    free(lambdaDiff2); cudaFree(d_lambdaDiff2);
src/rmsynthesis/devices.cu:    cudaFree(d_phiAxis);
src/rmsynthesis/devices.h:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis/devices.h:int getBestDevice(struct deviceInfoList *gpuList, int nDevices);
src/rmsynthesis/devices.h:struct deviceInfoList copySelectedDeviceInfo(struct deviceInfoList *gpuList,  
src/rmsynthesis/devices.h:void checkCudaError(void);
src/rmsynthesis/devices.h:void getGpuAllocForP(int *blockSize, int *threadSize, long *nFrames, 
src/rmsynthesis/rmsynthesis.c:rmsynthesis.c: A GPU based implementation of RM Synthesis.
src/rmsynthesis/rmsynthesis.c:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis/rmsynthesis.c:    struct deviceInfoList *gpuList;
src/rmsynthesis/rmsynthesis.c:    /* Retreive information about all connected GPU devices */
src/rmsynthesis/rmsynthesis.c:    gpuList = getDeviceInformation(&nDevices);
src/rmsynthesis/rmsynthesis.c:    selectedDevice = getBestDevice(gpuList, nDevices);
src/rmsynthesis/rmsynthesis.c:    cudaSetDevice(selectedDevice);
src/rmsynthesis/rmsynthesis.c:    checkCudaError();
src/rmsynthesis/rmsynthesis.c:    selectedDeviceInfo = copySelectedDeviceInfo(gpuList, selectedDevice);
src/rmsynthesis/rmsynthesis.c:    free(gpuList);
src/rmsynthesis/rmsynthesis.c:    cudaDeviceReset();
src/rmsynthesis/inputparser.c:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis/inputparser.c:    if(! config_lookup_int(&cfg, "nGPU", &inOptions.nGPU)) {
src/rmsynthesis/inputparser.c:        printf("INFO: 'nGPU' undefined in parset. Will use 1 device.\n");
src/rmsynthesis/inputparser.c:        inOptions.nGPU = 1;
src/rmsynthesis/inputparser.h:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis/version.h:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis/constants.h:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis/fileaccess.h:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis/rmsf.c:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis/rmsf.h:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis/fileaccess.c:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis_cpu/cpu_fileaccess.c:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis_cpu/cpu_rmsf.h:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis_cpu/cpu_fileaccess.h:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis_cpu/parsetFile:qCubeName = "/home/sarrvesh/Work/RMSynth_GPU/test_wsrt/cube-Q.fits";
src/rmsynthesis_cpu/parsetFile:uCubeName = "/home/sarrvesh/Work/RMSynth_GPU/test_wsrt/cube-U.fits";
src/rmsynthesis_cpu/parsetFile:freqFileName = "/home/sarrvesh/Work/RMSynth_GPU/test_wsrt/freqTable";
src/rmsynthesis_cpu/cpu_rmsf.c:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis_cpu/cpu_rmclean.c:Correspondence concerning RMSynth_GPU should be addressed to: 
src/rmsynthesis_cpu/cpu_rmclean.h:Correspondence concerning RMSynth_GPU should be addressed to: 

```
