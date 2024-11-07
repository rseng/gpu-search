# https://github.com/NLeSC/eAstroViz

```console
CITATION.cff:  directory LOFAR-source. Finally, we have developed a GPU
CITATION.cff:  masters project. The code is in the directory GPU-source.
README.md:Finally, we have developed a GPU prototype version of the code as well.
README.md:The code is in the directory GPU-source.
GPU-source/cuPrintf.cuh: * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
GPU-source/cuPrintf.cuh: * Please refer to the NVIDIA end user license agreement (EULA) associated
GPU-source/cuPrintf.cuh:                cudaPrintfInit();
GPU-source/cuPrintf.cuh:                cudaPrintfDisplay(stdout, true);
GPU-source/cuPrintf.cuh:                cudaPrintfEnd();
GPU-source/cuPrintf.cuh://      kernels unless cudaPrintfInit() is called again.
GPU-source/cuPrintf.cuh://      cudaPrintfInit
GPU-source/cuPrintf.cuh://      file or buffer size needs to be changed, call cudaPrintfEnd()
GPU-source/cuPrintf.cuh://      before re-calling cudaPrintfInit().
GPU-source/cuPrintf.cuh://      The default size for the buffer is 1 megabyte. For CUDA
GPU-source/cuPrintf.cuh://              cudaSuccess if all is well.
GPU-source/cuPrintf.cuh:extern "C" cudaError_t cudaPrintfInit(size_t bufferLen=1048576);   // 1-meg - that's enough for 4096 printfs by all threads put together
GPU-source/cuPrintf.cuh://      cudaPrintfEnd
GPU-source/cuPrintf.cuh://      Cleans up all memories allocated by cudaPrintfInit().
GPU-source/cuPrintf.cuh://      Call this at exit, or before calling cudaPrintfInit() again.
GPU-source/cuPrintf.cuh:extern "C" void cudaPrintfEnd();
GPU-source/cuPrintf.cuh://      cudaPrintfDisplay
GPU-source/cuPrintf.cuh://              cudaSuccess if all is well.
GPU-source/cuPrintf.cuh:extern "C" cudaError_t cudaPrintfDisplay(void *outputFP=NULL, bool showThreadID=false);
GPU-source/Data_reader.cu:Device_array_pointers* allocate_cuda_memory(Data_info* h_data_info){
GPU-source/Data_reader.cu:  cudaError_t cErr;
GPU-source/Data_reader.cu:  if( (cErr = cudaMalloc(&result->data, sizeof(float) * (nr_times * nr_subbands *
GPU-source/Data_reader.cu:        nr_channels * nr_polarizations))) != cudaSuccess){
GPU-source/Data_reader.cu:   fprintf(stderr, "%s\n", cudaGetErrorString(cErr));
GPU-source/Data_reader.cu:  if( (cErr = cudaMalloc(&flagged, sizeof(unsigned char) * (nr_times * nr_subbands *
GPU-source/Data_reader.cu:        nr_channels * nr_polarizations))) != cudaSuccess){
GPU-source/Data_reader.cu:   fprintf(stderr, "%s\n", cudaGetErrorString(cErr));
GPU-source/Data_reader.cu:  if( (cErr = cudaMalloc(&initial_flagged, sizeof(unsigned char) * (nr_times * nr_subbands *
GPU-source/Data_reader.cu:        nr_channels * nr_polarizations))) != cudaSuccess){
GPU-source/Data_reader.cu:   fprintf(stderr, "%s\n", cudaGetErrorString(cErr));
GPU-source/Data_reader.cu:  if( (cErr = cudaMemcpy(result->data, linear_data, sizeof(float) * (nr_times * nr_subbands *
GPU-source/Data_reader.cu:        nr_channels * nr_polarizations), cudaMemcpyHostToDevice)) !=
GPU-source/Data_reader.cu:        cudaSuccess){
GPU-source/Data_reader.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(cErr));
GPU-source/Data_reader.cu:  if( (cErr = cudaMemcpy(flagged, linear_flagged, sizeof(unsigned char) * (nr_times * nr_subbands *
GPU-source/Data_reader.cu:        nr_channels * nr_polarizations ), cudaMemcpyHostToDevice)) !=
GPU-source/Data_reader.cu:        cudaSuccess){
GPU-source/Data_reader.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(cErr));
GPU-source/Data_reader.cu:  if( (cErr = cudaMemcpy(initial_flagged, linear_initial_flagged, sizeof(unsigned char) * (nr_times * nr_subbands *
GPU-source/Data_reader.cu:        nr_channels * nr_polarizations ), cudaMemcpyHostToDevice)) !=
GPU-source/Data_reader.cu:        cudaSuccess){
GPU-source/Data_reader.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(cErr));
GPU-source/Data_reader.cu:Device_array_pointers* malloc_cuda_memory(Data_info* h_data_info, Device_data** d_data){
GPU-source/Data_reader.cu:  cudaError_t cErr;
GPU-source/Data_reader.cu:  if((cErr = cudaMalloc(&(*d_data), sizeof(Device_data))) != cudaSuccess){
GPU-source/Data_reader.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(cErr));
GPU-source/Data_reader.cu:  if((cErr = cudaMemcpy(*d_data, h_data, sizeof(Device_data),
GPU-source/Data_reader.cu:          cudaMemcpyHostToDevice)) != cudaSuccess){
GPU-source/Data_reader.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(cErr));
GPU-source/Data_reader.cu:  return allocate_cuda_memory(h_data_info);
GPU-source/cuPrintf.cu: * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
GPU-source/cuPrintf.cu: * Please refer to the NVIDIA end user license agreement (EULA) associated
GPU-source/cuPrintf.cu: *      the host side - but only after a cudaDeviceSynchronize() on the host.
GPU-source/cuPrintf.cu:                cudaPrintfInit();
GPU-source/cuPrintf.cu:                cudaPrintfDisplay(stdout, true);
GPU-source/cuPrintf.cu:                cudaPrintfEnd();
GPU-source/cuPrintf.cu: *      arguments to cudaPrintfInit() and cudaPrintfDisplay();
GPU-source/cuPrintf.cu:#if __CUDA_ARCH__ > 100      // Atomics only used with > sm_10 architecture
GPU-source/cuPrintf.cu:#if __CUDA_ARCH__ == 100
GPU-source/cuPrintf.cu:        // (so that cudaPrintfDisplay() below will work). This is only run once.
GPU-source/cuPrintf.cu://  cudaPrintfDisplay() below so we can handle the SM_10 architecture
GPU-source/cuPrintf.cu:        cudaMemcpy(printfbuf_local, bufptr, CUPRINTF_MAX_LEN, cudaMemcpyDeviceToHost);
GPU-source/cuPrintf.cu:            cudaMemset(bufptr, 0, CUPRINTF_MAX_LEN);
GPU-source/cuPrintf.cu://  cudaPrintfInit
GPU-source/cuPrintf.cu:extern "C" cudaError_t cudaPrintfInit(size_t bufferLen)
GPU-source/cuPrintf.cu:    if (cudaMalloc((void **)&printfbuf_device, printfbuf_len) != cudaSuccess)
GPU-source/cuPrintf.cu:        return cudaErrorInitializationError;
GPU-source/cuPrintf.cu:    cudaMemset(printfbuf_device, 0, printfbuf_len);
GPU-source/cuPrintf.cu:    cudaMemcpyToSymbol(restrictRules, &restrict, sizeof(restrict));
GPU-source/cuPrintf.cu:    cudaMemcpyToSymbol(globalPrintfBuffer, &printfbuf_device, sizeof(char *));
GPU-source/cuPrintf.cu:    cudaMemcpyToSymbol(printfBufferPtr, &printfbuf_device, sizeof(char *));
GPU-source/cuPrintf.cu:    cudaMemcpyToSymbol(printfBufferLength, &printfbuf_len, sizeof(printfbuf_len));
GPU-source/cuPrintf.cu:    return cudaSuccess;
GPU-source/cuPrintf.cu://  cudaPrintfEnd
GPU-source/cuPrintf.cu:extern "C" void cudaPrintfEnd()
GPU-source/cuPrintf.cu:    cudaFree(printfbuf_device);
GPU-source/cuPrintf.cu://  cudaPrintfDisplay
GPU-source/cuPrintf.cu:extern "C" cudaError_t cudaPrintfDisplay(void *outputFP, bool showThreadID)
GPU-source/cuPrintf.cu:        return cudaErrorMissingConfiguration;
GPU-source/cuPrintf.cu:    cudaMemcpy(&magic, printfbuf_device, sizeof(unsigned short), cudaMemcpyDeviceToHost);
GPU-source/cuPrintf.cu:            cudaMemcpy(&hdr, blockptr, sizeof(hdr), cudaMemcpyDeviceToHost);
GPU-source/cuPrintf.cu:        cudaMemcpyFromSymbol(&printfbuf_end, printfBufferPtr, sizeof(char *));
GPU-source/cuPrintf.cu:        cudaMemset(printfbuf_device, 0, printfbuf_len);
GPU-source/cuPrintf.cu:    return cudaSuccess;
GPU-source/flagger.c:void allocate_cuda_memory(Data_info* d_data_info){
GPU-source/flagger.c:  data = (float****) cudaMalloc(sizeof(float***) * nr_times);
GPU-source/flagger.c:  flagged = (unsigned char***) cudaMalloc(sizeof(unsigned char **) * nr_times);
GPU-source/flagger.c:  initial_flagged = (unsigned char***) cudaMalloc(sizeof(unsigned char **) * nr_times);
GPU-source/flagger.c:    data[i] = (float ***) cudaMalloc(sizeof(float**) * nr_subbands);
GPU-source/flagger.c:    flagged[i] = (unsigned char **) cudaMalloc(sizeof(unsigned char *) * nr_subbands);
GPU-source/flagger.c:    initial_flagged[i] = (unsigned char **) cudaMalloc(sizeof(unsigned char *) 
GPU-source/flagger.c:      data[i][j] = (float **) cudaMalloc(sizeof(float*) * nr_polarizations);
GPU-source/flagger.c:        data[i][j][k] = (float *) cudaMalloc(sizeof(float) * nr_channels);
GPU-source/flagger.c:        cudaMemset(data[i][j][k], 0, sizeof(float) * nr_channels);
GPU-source/flagger.c:      flagged[i][j] = (unsigned char*) cudaMalloc(sizeof(unsigned char) * nr_channels);
GPU-source/flagger.c:      initial_flagged[i][j] = (unsigned char*) cudaMalloc(sizeof(unsigned char) *
GPU-source/flagger.c:      cudaMemset(flagged[i][j], 0, sizeof(unsigned char) * nr_channels);
GPU-source/flagger.c:      cudaMemset(initial_flagged[i][j], 0, sizeof(unsigned char) * nr_channels);
GPU-source/flagger.c:void malloc_cuda_memory(Data_info* h_data_info, Data_info* d_data_info){
GPU-source/flagger.c:  d_data_info = cudaMalloc(sizeof(Data_info));
GPU-source/flagger.c:  allocate_cuda_memory(d_data_info);
GPU-source/flagger.c:  malloc_cuda_memory(&h_data_info, &d_data_info);
GPU-source/flagger.c:    fprintf(stderr, "Something went wrong allocating CUDA memory");
GPU-source/Makefile:CUDA_LDFLAGS = -lrt -lcudart
GPU-source/Makefile:CUDA_LIBS = -L"$(CUDA_INSTALL_PATH)/lib64/"
GPU-source/Makefile:	$(CC) $(CFLAGS) flagger.o Data_reader.o timer.o -o flagger $(CUDA_LDFLAGS) $(CUDA_LIBS)
GPU-source/Makefile:	$(CC) $(CFLAGS) -c flagger.cu $(CUDA_LDFLAGS) $(CUDA_LIBS)
GPU-source/Makefile:	$(CC) $(CFLAGS) -c Data_reader.cu $(CUDA_LDFLAGS) $(CUDA_LIBS)
GPU-source/Data_reader.h:Device_array_pointers* malloc_cuda_memory(Data_info* h_data_info, Device_data** d_data);
GPU-source/flagger.cu:#include <cuda.h>
GPU-source/flagger.cu:  cudaError_t devRetVal;
GPU-source/flagger.cu:  devRetVal = cudaMalloc(&(*d_flags), (size_t)(data_size *  sizeof(unsigned char)));
GPU-source/flagger.cu:  if(devRetVal != cudaSuccess){
GPU-source/flagger.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  devRetVal = cudaMalloc(&(*d_nr_flagged), (size_t)(nr_blocks *  sizeof(unsigned int)));
GPU-source/flagger.cu:  if(devRetVal != cudaSuccess){
GPU-source/flagger.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  devRetVal = cudaMemset(*d_nr_flagged, 0, (size_t)(nr_blocks *  sizeof(unsigned int)));
GPU-source/flagger.cu:  if(devRetVal != cudaSuccess){
GPU-source/flagger.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  devRetVal = cudaMemset(*d_flags, 0, (size_t)(data_size * sizeof(unsigned
GPU-source/flagger.cu:  if(devRetVal != cudaSuccess){
GPU-source/flagger.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  cudaError_t devRetVal;
GPU-source/flagger.cu:  devRetVal = cudaMalloc(&(*d_nr_flagged), (size_t)(nr_blocks *  sizeof(unsigned int)));
GPU-source/flagger.cu:  if(devRetVal != cudaSuccess){
GPU-source/flagger.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  cudaError_t devRetVal;
GPU-source/flagger.cu:  if( (devRetVal = cudaGetLastError()) != cudaSuccess){
GPU-source/flagger.cu:        cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  cudaError_t devRetVal;
GPU-source/flagger.cu:  if( (devRetVal = cudaGetLastError()) != cudaSuccess){
GPU-source/flagger.cu:        cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  cudaError_t devRetVal;
GPU-source/flagger.cu:  devRetVal = cudaMalloc(&d_values_reduced, (size_t)((data_size /
GPU-source/flagger.cu:  if(devRetVal != cudaSuccess){
GPU-source/flagger.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  if( (devRetVal = cudaGetLastError()) != cudaSuccess){
GPU-source/flagger.cu:        cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  cudaError_t devRetVal;
GPU-source/flagger.cu:  devRetVal = cudaMalloc(&d_values_reduced, (size_t)((data_size /
GPU-source/flagger.cu:  if(devRetVal != cudaSuccess){
GPU-source/flagger.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  if( (devRetVal = cudaGetLastError()) != cudaSuccess){
GPU-source/flagger.cu:        cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  cudaError_t devRetVal;
GPU-source/flagger.cu:  if( (devRetVal = cudaGetLastError()) != cudaSuccess){
GPU-source/flagger.cu:        cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  cudaError_t devRetVal;
GPU-source/flagger.cu:  if( (devRetVal = cudaGetLastError()) != cudaSuccess){
GPU-source/flagger.cu:        cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  cudaError_t devRetVal;
GPU-source/flagger.cu:  ptrs = malloc_cuda_memory(h_data_info, &d_data);
GPU-source/flagger.cu:  cudaPrintfInit();
GPU-source/flagger.cu:  cudaPrintfDisplay(stdout, true);
GPU-source/flagger.cu:  cudaPrintfEnd();
GPU-source/flagger.cu:  cudaDeviceSynchronize();
GPU-source/flagger.cu:  if( (devRetVal = cudaGetLastError()) != cudaSuccess){
GPU-source/flagger.cu:        cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  if( (devRetVal = cudaMemcpy(h_flags, d_flags,
GPU-source/flagger.cu:          cudaMemcpyDeviceToHost))
GPU-source/flagger.cu:        != cudaSuccess){
GPU-source/flagger.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  if( (devRetVal = cudaMemcpy(h_values, ptrs->data,
GPU-source/flagger.cu:          cudaMemcpyDeviceToHost))
GPU-source/flagger.cu:        != cudaSuccess){
GPU-source/flagger.cu:    fprintf(stderr, "%s\n", cudaGetErrorString(devRetVal));
GPU-source/flagger.cu:  cudaDeviceReset();
GPU-source/flagger.cu:  cudaFree(d_flags);
GPU-source/flagger.cu:  cudaFree(d_nr_flagged);

```
