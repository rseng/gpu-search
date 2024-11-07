# https://github.com/lab-medvedeva/GADES-main

```console
Dockerfile:FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
R/mtrx.R:    gpu_loaded <- tryCatch({
R/mtrx.R:        .C("check_gpu", PACKAGE="mtrx")
R/mtrx.R:        message("GPU mode enabled")
R/mtrx.R:        message("GPU Package not installed. You can use only CPU version of the package")
R/mtrx.R:    assign("gpu_loaded", gpu_loaded, envir = parent.env(environment()))
R/mtrx.R:#' Function to process batch from shared objects for GPU.
R/mtrx.R:#' @param type "gpu" or "cpu".
R/mtrx.R:mtrx_distance <- function(a, filename = "", batch_size = 1000, metric = "kendall",type="gpu", sparse = F, write=F)
R/mtrx.R:  if (type == "gpu" && !GADES:::gpu_loaded) {
R/mtrx.R:    stop("GPU not loaded. Please, install CUDA Toolkit and with your setup")
R/mtrx.R:            if (type=="gpu") {
README.md:## GADES - GPU-Assisted Distance Estimation Software
README.md:* (Optional) CUDA 11+
README.md:### Docker image start CUDA
README.md:Please, install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) first.
README.md:docker run --name gades --gpus all -it akhtyamovpavel/gades-gpu
README.md:This command builds code of the library using CMake, checks GPU and install package using CPU+GPU or only CPU code.
README.md:dist.matrix <- mtrx_distance(mtx, batch_size = 5000, metric = 'kendall', type='gpu', sparse=F, write=T)
README.md:### Sparse mode - GPU
README.md:dist.matrix <- mtrx_distance(mtx, batch_size = 5000, metric = 'kendall', type='gpu', sparse=T, write=T)
CMakeLists.txt:option(WITH_CUDA "Install With CUDA" ON)
CMakeLists.txt:if (WITH_CUDA)
CMakeLists.txt:    message("Building with CUDA required")
CMakeLists.txt:    find_package(CUDA REQUIRED)
CMakeLists.txt:    if(CUDA_FOUND)
CMakeLists.txt:        cuda_add_library(
CMakeLists.txt:    endif(CUDA_FOUND)
CMakeLists.txt:endif(WITH_CUDA)
DESCRIPTION:Title: GADES - GPU-Assisted Distance Estimation Software
test.R:        library.dynam('mtrx', package = 'HobotnicaGPU', lib.loc = NULL)
test.R:       library.dynam('mtrx_cpu', package = 'HobotnicaGPU', lib.loc = NULL)
test.R:#.C("check_gpu", PACKAGE = "mtrx")
test.R:library(HobotnicaGPU)
test.R:    if (method == 'GPU') {
test.R:        distMatrix_mtrx <- mtrx_distance(data, batch_size = batch_size , metric = metric,type="gpu",sparse=sparse, filename=filename)
man/mtrx_distance.Rd:  type = "gpu"
man/mtrx_distance.Rd:\item{type}{"gpu" or "cpu".}
man/process_batch.Rd:\title{Function to process batch from shared objects for GPU.}
man/process_batch.Rd:Function to process batch from shared objects for GPU.
src/main.cu:#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
src/main.cu:inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
src/main.cu:   if (code != cudaSuccess) 
src/main.cu:      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
src/main.cu:__global__ void Rkendall_gpu_atomic_float(float* array, const int n, const int m, unsigned int* result) {
src/main.cu:__global__ void Reuclidean_gpu_atomic_float(float* array, const int n, const int m, float* result) {
src/main.cu:__global__ void Rkendall_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, unsigned int* result) {
src/main.cu:__global__ void Reuclidean_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, float* result) {
src/main.cu:__global__ void RpearsonCorr_gpu_atomic_float_same_block(
src/main.cu:__global__ void RpearsonCorr_gpu_atomic_float_different_blocks(float* array, float* array2, const int n, const int m, const int m_b, float* scalar_prod,float *x_norm, float* y_norm ){
src/main.cu:extern "C" bool check_gpu() {
src/main.cu:    cudaDeviceProp deviceProp;
src/main.cu:	cudaGetDeviceProperties(&deviceProp, 0);
src/main.cu:    cudaEvent_t start;
src/main.cu:    cudaEventCreate(&start);
src/main.cu:    cudaEventRecord(start);
src/main.cu:    cudaEventSynchronize(start);
src/main.cu:  cudaEvent_t start, stop1, stop2, stop3;
src/main.cu:  cudaMalloc(&d_array, array_size * sizeof(float));
src/main.cu:  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:  cudaMalloc(&d_result, (*m) * (*m) * sizeof(unsigned int));
src/main.cu:  cudaMemset(d_result, 0, (*m) * (*m) * sizeof(unsigned int));
src/main.cu:  Rkendall_gpu_atomic_float<<<BLOCKS, THREADS>>>(d_array, *n, *m, d_result);
src/main.cu:  cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:  cudaFree(d_result);
src/main.cu:  cudaFree(d_array);
src/main.cu:  //cudaEventRecord(stop3);
src/main.cu:  //cudaEventSynchronize(stop3);
src/main.cu:  //cudaEventElapsedTime(&milliseconds, start, stop3);
src/main.cu:  cudaMalloc(&d_array, array_size * sizeof(float));
src/main.cu:  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:  cudaMalloc(&d_result, (*m) * (*m) * sizeof(float));
src/main.cu:  cudaMemset(d_result, 0, (*m) * (*m) * sizeof(float));
src/main.cu:  Reuclidean_gpu_atomic_float<<<blocks_in_row, threads>>>(d_array, *n, *m, d_result);
src/main.cu:  cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:  cudaFree(d_result);
src/main.cu:  cudaFree(d_array);
src/main.cu:  cudaMalloc(&d_array, array_size * sizeof(float));
src/main.cu:  cudaMalloc(&d_array2, array_size * sizeof(float));
src/main.cu:  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:  cudaMemcpy(d_array2, array2_new, array2_size * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:  cudaMalloc(&d_result, (*m) * (*m_b) * sizeof(unsigned int));
src/main.cu:  cudaMemset(d_result, 0, (*m) * (*m_b) * sizeof(unsigned int));
src/main.cu:  Rkendall_gpu_atomic_float_different_blocks<<<BLOCKS, THREADS>>>(d_array, d_array2, *n, *m, *m_b, d_result);
src/main.cu:  cudaMemcpy(h_result, d_result, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:  cudaFree(d_result);
src/main.cu:  cudaFree(d_array);
src/main.cu:  cudaFree(d_array2);
src/main.cu:  cudaMalloc(&d_array, array_size * sizeof(float));
src/main.cu:  cudaMalloc(&d_array2, array_size * sizeof(float));
src/main.cu:  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:  cudaMemcpy(d_array2, array2_new, array2_size * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:  cudaMalloc(&d_result, (*m) * (*m_b) * sizeof(float));
src/main.cu:  cudaMemset(d_result, 0, (*m) * (*m_b) * sizeof(float));
src/main.cu:  Reuclidean_gpu_atomic_float_different_blocks<<<blocks_in_row, threads>>>(d_array, d_array2, *n, *m, *m_b, d_result);
src/main.cu:  cudaMemcpy(h_result, d_result, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:  cudaFree(d_result);
src/main.cu:  cudaFree(d_array);
src/main.cu:  cudaFree(d_array2);
src/main.cu:  cudaMalloc(&d_array, array_size * sizeof(float));
src/main.cu:  // cudaMalloc(&d_array2, array_size * sizeof(float));
src/main.cu:  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:  // cudaMemcpy(d_array2, array2_new, array2_size * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:  cudaMalloc(&d_result, (*m) * (*m) * sizeof(float)); 
src/main.cu:  cudaMemset(d_result, 0, (*m) * (*m) * sizeof(float));
src/main.cu:  cudaMalloc(&d_x_norm_result, (*m) * (*m) * sizeof(float)); 
src/main.cu:  cudaMemset(d_x_norm_result, 0, (*m) * (*m) * sizeof(float));
src/main.cu:  cudaMalloc(&d_y_norm_result, (*m) * (*m) * sizeof(float)); 
src/main.cu:  cudaMemset(d_y_norm_result, 0, (*m) * (*m) * sizeof(float));
src/main.cu:  RpearsonCorr_gpu_atomic_float_same_block<<<blocks, threads>>>(
src/main.cu:  cudaMemcpy(h_result, d_result, (*m) * (*m) * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:  cudaFree(d_result);
src/main.cu:  cudaFree(d_x_norm_result);
src/main.cu:  cudaFree(d_y_norm_result);
src/main.cu:  cudaFree(d_array);
src/main.cu:  cudaMalloc(&d_array, array_size * sizeof(float));
src/main.cu:  cudaMalloc(&d_array2, array2_size * sizeof(float));
src/main.cu:  cudaMemcpy(d_array, array_new, array_size * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:  cudaMemcpy(d_array2, array2_new, array2_size * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu://  cudaMalloc(&d_result, (*m) * (*m_b) * sizeof(float)); 
src/main.cu://  cudaMemset(d_result, 0, (*m) * (*m_b) * sizeof(float));
src/main.cu:  cudaMalloc(&scalar, (*m) * (*m_b) * sizeof(float)); 
src/main.cu:  cudaMemset(scalar, 0, (*m) * (*m_b) * sizeof(float));
src/main.cu:  cudaMalloc(&prod1, (*m) * (*m_b) * sizeof(float)); 
src/main.cu:  cudaMemset(prod1, 0, (*m) * (*m_b) * sizeof(float));
src/main.cu:  cudaMalloc(&prod2, (*m) * (*m_b) * sizeof(float)); 
src/main.cu:  cudaMemset(prod2, 0, (*m) * (*m_b) * sizeof(float));
src/main.cu:  RpearsonCorr_gpu_atomic_float_different_blocks<<<blocks_in_row, threads>>>(d_array,d_array2, *n, *m,*m_b, scalar,prod1,prod2);
src/main.cu:  cudaMemcpy(h_scalar, scalar, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:  cudaMemcpy(h_prod1, prod1, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:  cudaMemcpy(h_prod2, prod2, (*m) * (*m_b) * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:  cudaFree(scalar);
src/main.cu:  cudaFree(prod2);
src/main.cu:  cudaFree(prod1);
src/main.cu:  cudaFree(d_array);
src/main.cu:  cudaFree(d_array2);
src/main.cu:__global__ void ReuclideanSparse_gpu_atomic_float_same_block(
src/main.cu:    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
src/main.cu:    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
src/main.cu:    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
src/main.cu:    cudaMalloc(&d_result, columns * columns * sizeof(float));
src/main.cu:    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemset(d_result, 0, columns * columns * sizeof(float));
src/main.cu:    ReuclideanSparse_gpu_atomic_float_same_block<<<blocksPerGrid, threadsPerBlock>>>(
src/main.cu:    gpuErrchk( cudaPeekAtLastError() );
src/main.cu:    gpuErrchk( cudaPeekAtLastError() );
src/main.cu:    cudaMemcpy(float_result, d_result, columns * columns * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:    gpuErrchk( cudaPeekAtLastError() );
src/main.cu:    cudaFree(d_a_index);
src/main.cu:    cudaFree(d_a_positions);
src/main.cu:    cudaFree(d_a_values);
src/main.cu:    cudaFree(d_result);
src/main.cu:    gpuErrchk(cudaPeekAtLastError());
src/main.cu:__global__ void ReuclideanSparse_gpu_atomic_float_different_blocks(
src/main.cu:    gpuErrchk(cudaPeekAtLastError());
src/main.cu:    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
src/main.cu:    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
src/main.cu:    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
src/main.cu:    cudaMalloc(&d_b_index, num_elements_b_int * sizeof(int));  // Use the same size as 'a' since 'b' is not used in this version
src/main.cu:    cudaMalloc(&d_b_positions, (rows + 1) * sizeof(int));       // Use the same size as 'a' since 'b' is not used in this version
src/main.cu:    cudaMalloc(&d_b_values, num_elements_b_int * sizeof(float)); // Use the same size as 'a' since 'b' is not used in this version
src/main.cu:    cudaMalloc(&d_result, columns * columns_b * sizeof(float));
src/main.cu:    gpuErrchk(cudaPeekAtLastError());
src/main.cu:    gpuErrchk(cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice));
src/main.cu:    gpuErrchk(cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
src/main.cu:    gpuErrchk(cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice));
src/main.cu:    gpuErrchk(cudaMemcpy(d_b_index, b_index, num_elements_b_int * sizeof(int), cudaMemcpyHostToDevice));  // Use the same data for 'b' as they are not used
src/main.cu:    gpuErrchk(cudaMemcpy(d_b_positions, b_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));  // Use the same data for 'b' as they are not used
src/main.cu:    gpuErrchk(cudaMemcpy(d_b_values, b_values, num_elements_b_int * sizeof(float), cudaMemcpyHostToDevice)); // Use the same data for 'b' as they are not used
src/main.cu:    gpuErrchk(cudaMemset(d_result, 0, columns * columns_b * sizeof(float)));
src/main.cu:    gpuErrchk(cudaPeekAtLastError());
src/main.cu:    ReuclideanSparse_gpu_atomic_float_different_blocks<<<blocksPerGrid, threadsPerBlock>>>(
src/main.cu:    gpuErrchk( cudaPeekAtLastError() );
src/main.cu:    cudaMemcpy(float_result, d_result, columns * columns_b * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:    gpuErrchk( cudaPeekAtLastError() );
src/main.cu:    cudaFree(d_a_index);
src/main.cu:    cudaFree(d_a_positions);
src/main.cu:    cudaFree(d_a_values);
src/main.cu:    cudaFree(d_b_index);
src/main.cu:    cudaFree(d_b_positions);
src/main.cu:    cudaFree(d_b_values);
src/main.cu:    cudaFree(d_result);
src/main.cu:__global__ void RpearsonSparseCorr_gpu_atomic_float_different_blocks(
src/main.cu:    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
src/main.cu:    cudaMalloc(&d_b_values, num_elements_b_int * sizeof(float));
src/main.cu:    cudaMalloc(&d_float_result, columns * columns_b * sizeof(float));
src/main.cu:    cudaMalloc(&d_squares_a, columns * sizeof(float));
src/main.cu:    cudaMalloc(&d_squares_b, columns_b * sizeof(float));
src/main.cu:    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
src/main.cu:    cudaMalloc(&d_b_positions, (rows + 1) * sizeof(int));
src/main.cu:    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
src/main.cu:    cudaMalloc(&d_b_index, num_elements_b_int * sizeof(int));
src/main.cu:    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_b_values, b_values, num_elements_b_int * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_float_result, float_result, columns * columns_b * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_squares_a, squares_a, columns * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_squares_b, squares_b, columns_b * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_b_index, b_index, num_elements_b_int * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_b_positions, b_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    RpearsonSparseCorr_gpu_atomic_float_different_blocks<<<blocksPerGrid, threadsPerBlock>>>(
src/main.cu:    gpuErrchk( cudaPeekAtLastError() );
src/main.cu:    cudaMemcpy(float_result, d_float_result, columns * columns_b * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:    cudaMemcpy(squares_a, d_squares_a, columns * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:    cudaMemcpy(squares_b, d_squares_b, columns_b * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:    cudaFree(d_a_values);
src/main.cu:    cudaFree(d_b_values);
src/main.cu:    cudaFree(d_float_result);
src/main.cu:    cudaFree(d_squares_a);
src/main.cu:    cudaFree(d_squares_b);
src/main.cu:    cudaFree(d_a_index);
src/main.cu:    cudaFree(d_b_index);
src/main.cu:    cudaFree(d_a_positions);
src/main.cu:    cudaFree(d_b_positions);
src/main.cu:__global__ void RpearsonSparseCorr_gpu_atomic_float_same_block(
src/main.cu:    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
src/main.cu:    cudaMalloc(&d_float_result, columns * columns * sizeof(float));
src/main.cu:    cudaMalloc(&d_squares, columns * sizeof(float));
src/main.cu:    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
src/main.cu:    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
src/main.cu:    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_float_result, float_result, columns * columns * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_squares, squares, columns * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    RpearsonSparseCorr_gpu_atomic_float_same_block<<<blocksPerGrid, threadsPerBlock>>>(
src/main.cu:    gpuErrchk( cudaPeekAtLastError() );
src/main.cu:    gpuErrchk( cudaPeekAtLastError() );
src/main.cu:    cudaMemcpy(float_result, d_float_result, columns * columns * sizeof(float), cudaMemcpyDeviceToHost);
src/main.cu:    gpuErrchk( cudaPeekAtLastError() );
src/main.cu:    cudaFree(d_a_values);
src/main.cu:    cudaFree(d_float_result);
src/main.cu:    cudaFree(d_squares);
src/main.cu:    cudaFree(d_a_index);
src/main.cu:    cudaFree(d_a_positions);
src/main.cu:__global__ void RkendallSparseCorr_gpu_atomic_float_same_block(
src/main.cu:    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
src/main.cu:    cudaMalloc(&d_concordant, columns * columns * sizeof(int));
src/main.cu:    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
src/main.cu:    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
src/main.cu:    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(
src/main.cu:      cudaMemcpyHostToDevice
src/main.cu:    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    RkendallSparseCorr_gpu_atomic_float_same_block<<<BLOCKS, THREADS>>>(
src/main.cu:    cudaMemcpy(h_concordant, d_concordant, columns * columns * sizeof(int), cudaMemcpyDeviceToHost);
src/main.cu:    cudaFree(d_a_index);
src/main.cu:    cudaFree(d_a_positions);
src/main.cu:    cudaFree(d_a_values);
src/main.cu:    cudaFree(d_concordant);
src/main.cu:__global__ void RkendallSparseCorr_gpu_atomic_float_different_blocks(
src/main.cu:    cudaMalloc(&d_a_values, num_elements_a_int * sizeof(float));
src/main.cu:    cudaMalloc(&d_b_values, num_elements_b_int * sizeof(float));
src/main.cu:    cudaMalloc(&d_a_positions, (rows + 1) * sizeof(int));
src/main.cu:    cudaMalloc(&d_b_positions, (rows + 1) * sizeof(int));
src/main.cu:    cudaMalloc(&d_a_index, num_elements_a_int * sizeof(int));
src/main.cu:    cudaMalloc(&d_b_index, num_elements_b_int * sizeof(int));
src/main.cu:    cudaMalloc(&d_concordant, columns * columns_b * sizeof(int));
src/main.cu:    cudaMemcpy(d_a_values, a_values, num_elements_a_int * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_b_values, b_values, num_elements_b_int * sizeof(float), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(
src/main.cu:      cudaMemcpyHostToDevice
src/main.cu:    cudaMemcpy(d_a_index, a_index, num_elements_a_int * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_a_positions, a_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_b_index, b_index, num_elements_b_int * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    cudaMemcpy(d_b_positions, b_positions, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
src/main.cu:    RkendallSparseCorr_gpu_atomic_float_different_blocks<<<BLOCKS, THREADS>>>(
src/main.cu:    cudaMemcpy(h_concordant, d_concordant, columns * columns_b * sizeof(int), cudaMemcpyDeviceToHost);
src/main.cu:    cudaFree(d_a_values);
src/main.cu:    cudaFree(d_concordant);
src/main.cu:    cudaFree(d_a_index);
src/main.cu:    cudaFree(d_a_positions);
src/main.cu:    cudaFree(d_b_index);
src/main.cu:    cudaFree(d_b_positions);
src/main.cu:    cudaFree(d_b_values);

```
