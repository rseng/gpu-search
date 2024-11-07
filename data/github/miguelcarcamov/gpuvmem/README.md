# https://github.com/miguelcarcamov/gpuvmem

```console
Dockerfile:FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
Dockerfile:ENV PATH /usr/local/cuda/bin${PATH:+:${PATH}}
Dockerfile:ENV LD_LIBRARY_PATH /usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
Dockerfile:# Install CUDA samples
Dockerfile:RUN cd /usr/local/cuda && \
Dockerfile:    git clone --single-branch --branch v12.4 https://github.com/NVIDIA/cuda-samples.git samples && \
Dockerfile:RUN echo "Hello there from gpuvmem base image"
Dockerfile:LABEL org.opencontainers.image.source="https://github.com/miguelcarcamov/gpuvmem"
_config.yml:url: https://gpuvmem.github.io
environment.yml:name: gpuvmem-env
Dockerfile.prod:FROM ghcr.io/miguelcarcamov/gpuvmem:base
Dockerfile.prod:RUN echo "Installing GPUVMEM"
Dockerfile.prod:    git clone -b "${BRANCH_NAME}" https://github.com/miguelcarcamov/gpuvmem.git
Dockerfile.prod:RUN cd gpuvmem && \
Dockerfile.prod:RUN echo "Hello there! from GPUVMEM production image"
Dockerfile.prod:LABEL org.opencontainers.image.source="https://github.com/miguelcarcamov/gpuvmem"
README.md:   <img src="https://github.com/miguelcarcamov/gpuvmem/wiki/images/logos/logo2.png" height="400">
README.md:- Wiki: <https://github.com/miguelcarcamov/gpuvmem/wiki>
README.md:If you use GPUVMEM for your research please do not forget to cite Cárcamo et al.
README.md:   title = "Multi-GPU maximum entropy image synthesis for radio astronomy",
README.md:   keywords = "Maximum entropy, GPU, ALMA, Inverse problem, Radio interferometry, Image synthesis"
README.md:5. Download or clone gpuvmem.
README.md:6. To compile GPUVMEM you will need:
README.md:   - CUDA 9, 9.1, 9.2, 10.0 and 11.0. Remember to add binaries and libraries to the **PATH** and **LD_LIBRARY_PATH** environment variables, respectively.
README.md:   docker pull ghcr.io/miguelcarcamov/gpuvmem:latest
README.md:   cd gpuvmem
README.md:# Use GPUVMEM
README.md:Usage: `./bin/gpuvmem [options]`
README.md:      -G --gpus [default: 0]
README.md:          Index of the GPU/s you are going to use separated by a comma
README.md:          GPU block X Size for image/Fourier plane (Needs to be pow of 2)
README.md:          GPU block Y Size for image/Fourier plane (Needs to be pow of 2)
README.md:          GPU block V Size for visibilities (Needs to be pow of 2)
README.md:          Runs gpuvmem with no positivity restrictions on the images
README.md:- CUDA version [e.g. 9]
README.md:- gpuvmem Version [e.g. 22]
include/classes/objectivefunction.cuh:    checkCudaErrors(cudaMemset(dphi, 0, sizeof(float) * M * N * image_count));
include/classes/objectivefunction.cuh:    checkCudaErrors(cudaMemcpy(xi, dphi, sizeof(float) * M * N * image_count,
include/classes/objectivefunction.cuh:                               cudaMemcpyDeviceToDevice));
include/classes/objectivefunction.cuh:    checkCudaErrors(cudaMalloc((void**)&dphi, sizeof(float) * M * N * I));
include/classes/objectivefunction.cuh:    checkCudaErrors(cudaMemset(dphi, 0, sizeof(float) * M * N * I));
include/classes/virtualimageprocessor.cuh:#include <cuda_runtime.h>
include/classes/synthesizer.cuh:#endif  // GPUVMEM_SYNTHESIZER_CUH
include/classes/optimizer.cuh:  __host__ virtual void allocateMemoryGpu() = 0;
include/classes/optimizer.cuh:  __host__ virtual void deallocateMemoryGpu() = 0;
include/classes/io.cuh:                          bool isInGPU){};
include/classes/io.cuh:                          bool isInGPU){};
include/classes/io.cuh:                          bool isInGPU){};
include/classes/io.cuh:                          bool isInGPU){};
include/classes/io.cuh:                                 bool isInGPU){};
include/classes/io.cuh:                                 bool isInGPU){};
include/classes/io.cuh:                                 bool isInGPU){};
include/classes/io.cuh:                                 bool isInGPU){};
include/classes/io.cuh:                                 bool isInGPU){};
include/classes/io.cuh:                                       bool isInGPU){};
include/classes/io.cuh:                                              bool isInGPU){};
include/classes/io.cuh:                                   bool isInGPU){};
include/classes/io.cuh:                                   bool isInGPU){};
include/classes/io.cuh:                                                bool isInGPU){};
include/classes/io.cuh:                                   bool isInGPU){};
include/classes/io.cuh:                                 bool isInGPU){};
include/classes/io.cuh:                                 bool isInGPU){};
include/classes/io.cuh:                                 bool isInGPU){};
include/classes/ckernel.cuh:  __host__ virtual float* getGCFGPU() { return this->gcf->getGPUKernel(); };
include/classes/ckernel.cuh:  __host__ virtual void printGCFGPU() { return this->gcf->printGPUCKernel(); };
include/classes/ckernel.cuh:    this->freeGPUKernel();
include/classes/ckernel.cuh:      this->gcf->freeGPUKernel();
include/classes/ckernel.cuh:  __host__ int getGPUID() { return this->gpu_id; };
include/classes/ckernel.cuh:  __host__ float* getGPUKernel() { return this->gpu_kernel; };
include/classes/ckernel.cuh:  __host__ void setGPUID(int gpu_id) { this->gpu_id = gpu_id; };
include/classes/ckernel.cuh:  __host__ void printGPUCKernel() {
include/classes/ckernel.cuh:      this->ioImageHandler->printImage(this->getGPUKernel(), "ckernel_gpu.fits",
include/classes/ckernel.cuh:  __host__ void printGPUGCF() {
include/classes/ckernel.cuh:          this->gcf->getGPUKernel(), "ckernel_gpu.fits", "", 0, 0, 1.0f,
include/classes/ckernel.cuh:  __host__ void freeGPUKernel() { cudaFree(this->gpu_kernel); };
include/classes/ckernel.cuh:  int gpu_id = 0;
include/classes/ckernel.cuh:  float* gpu_kernel;
include/classes/ckernel.cuh:    cudaSetDevice(this->gpu_id);
include/classes/ckernel.cuh:    checkCudaErrors(
include/classes/ckernel.cuh:        cudaMalloc(&this->gpu_kernel, sizeof(float) * this->m_times_n));
include/classes/ckernel.cuh:    checkCudaErrors(
include/classes/ckernel.cuh:        cudaMemset(this->gpu_kernel, 0, sizeof(float) * this->m_times_n));
include/classes/ckernel.cuh:  __host__ void copyKerneltoGPU() {
include/classes/ckernel.cuh:    cudaSetDevice(this->gpu_id);
include/classes/ckernel.cuh:    checkCudaErrors(cudaMemcpy(this->gpu_kernel, this->kernel.data(),
include/classes/ckernel.cuh:                               cudaMemcpyHostToDevice));
include/classes/fi.cuh:#include <cuda_runtime.h>
include/classes/fi.cuh:    cudaFree(device_S);
include/classes/fi.cuh:    cudaFree(device_DS);
include/classes/fi.cuh:    checkCudaErrors(cudaMalloc((void**)&device_S, sizeof(float) * M * N));
include/classes/fi.cuh:    checkCudaErrors(cudaMemset(device_S, 0, sizeof(float) * M * N));
include/classes/fi.cuh:    checkCudaErrors(cudaMalloc((void**)&device_DS, sizeof(float) * M * N));
include/classes/fi.cuh:    checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
include/framework.cuh:typedef struct varsPerGPU {
include/framework.cuh:} varsPerGPU;
include/framework.cuh:  std::string gpus;
include/longnam.h:#define fits_read_grppar_usht ffggpui
include/longnam.h:#define fits_read_grppar_ulng ffggpuj
include/longnam.h:#define fits_read_grppar_uint ffggpuk
include/longnam.h:#define fits_write_grppar_usht ffpgpui
include/longnam.h:#define fits_write_grppar_ulng ffpgpuj
include/longnam.h:#define fits_write_grppar_uint ffpgpuk
include/frprmn.cuh:  __host__ void allocateMemoryGpu();
include/frprmn.cuh:  __host__ void deallocateMemoryGpu();
include/iofits.cuh:                  bool isInGPU) override;
include/iofits.cuh:                  bool isInGPU) override;
include/iofits.cuh:                  bool isInGPU) override;
include/iofits.cuh:                  bool isInGPU) override;
include/iofits.cuh:                         bool isInGPU) override;
include/iofits.cuh:                         bool isInGPU) override;
include/iofits.cuh:                         bool isInGPU) override;
include/iofits.cuh:                         bool isInGPU) override;
include/iofits.cuh:                         bool isInGPU) override;
include/iofits.cuh:                               bool isInGPU) override;
include/iofits.cuh:                                      bool isInGPU) override;
include/iofits.cuh:                           bool isInGPU) override;
include/iofits.cuh:                           bool isInGPU) override;
include/iofits.cuh:                                        bool isInGPU) override;
include/iofits.cuh:                           bool isInGPU) override;
include/iofits.cuh:                         bool isInGPU) override;
include/iofits.cuh:                         bool isInGPU) override;
include/iofits.cuh:                         bool isInGPU) override;
include/lbfgs.cuh:  __host__ void allocateMemoryGpu();
include/lbfgs.cuh:  __host__ void deallocateMemoryGpu();
include/functions.cuh:                                          int num_gpus,
include/functions.cuh:                                          int firstgpu,
include/functions.cuh:                         int num_gpus,
include/functions.cuh:                         int firstgpu,
include/functions.cuh:__host__ void initFFT(varsPerGPU* vars_gpu,
include/functions.cuh:                      int firstgpu,
include/functions.cuh:                      int num_gpus);
include/functions.cuh:__global__ void do_griddingGPU(float3* uvw,
include/functions.cuh:__global__ void degriddingGPU(double3* uvw,
include/MSFITSIO.cuh:#include <cuda.h>
include/MSFITSIO.cuh:#include <helper_cuda.h>
include/MSFITSIO.cuh:const float PI = CUDART_PI_F;
include/MSFITSIO.cuh:const double PI_D = CUDART_PI;
include/MSFITSIO.cuh:                          int num_gpus,
include/MSFITSIO.cuh:                          int firstgpu);
include/MSFITSIO.cuh:                        bool isInGPU);
include/MSFITSIO.cuh:                                    bool isInGPU);
CMakeLists.txt:project(gpuvmem LANGUAGES C CXX CUDA)
CMakeLists.txt:find_package(CUDAToolkit REQUIRED)
CMakeLists.txt:option(MEMORY_DEBUG "This sets the GDB debug for CUDA code")
CMakeLists.txt:       "This option accelerate CUDA math functions decreasing precision")
CMakeLists.txt:  add_executable(gpuvmem ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu)
CMakeLists.txt:  target_sources(gpuvmem PRIVATE ${SOURCE_FILES})
CMakeLists.txt:  target_compile_features(gpuvmem PUBLIC cxx_std_11)
CMakeLists.txt:    set_target_properties(gpuvmem PROPERTIES RUNTIME_OUTPUT_DIRECTORY
CMakeLists.txt:    set_target_properties(gpuvmem PROPERTIES RUNTIME_OUTPUT_DIRECTORY
CMakeLists.txt:    gpuvmem
CMakeLists.txt:           ${CUDAToolkit_LIBRARY_ROOT}/samples/common/inc)
CMakeLists.txt:  get_target_property(TEMP gpuvmem COMPILE_FLAGS)
CMakeLists.txt:  # Find CUDA architecture
CMakeLists.txt:  include(FindCUDA/select_compute_arch)
CMakeLists.txt:  cuda_detect_installed_gpus(INSTALLED_GPU_CCS_1)
CMakeLists.txt:  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
CMakeLists.txt:  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
CMakeLists.txt:  string(REPLACE "." "" INSTALLED_GPU_CCS_4 "${INSTALLED_GPU_CCS_3}")
CMakeLists.txt:  string(REPLACE "+PTX" "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_4}")
CMakeLists.txt:  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
CMakeLists.txt:  set_target_properties(gpuvmem PROPERTIES CUDA_ARCHITECTURES
CMakeLists.txt:                                           "${CUDA_ARCH_LIST}")
CMakeLists.txt:  message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
CMakeLists.txt:  message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
CMakeLists.txt:  message(STATUS "CUDA Path: ${CUDAToolkit_LIBRARY_ROOT}")
CMakeLists.txt:      gpuvmem
CMakeLists.txt:                 CUDA_SEPARABLE_COMPILATION ON
CMakeLists.txt:                 CUDA_STANDARD 11)
CMakeLists.txt:    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
CMakeLists.txt:        gpuvmem
CMakeLists.txt:                   CUDA_SEPARABLE_COMPILATION ON
CMakeLists.txt:                   CUDA_STANDARD 11)
CMakeLists.txt:      set(CMAKE_CUDA_FLAGS
CMakeLists.txt:          "${CMAKE_CUDA_FLAGS} -Xptxas -O3 -Xcompiler -fopenmp")
CMakeLists.txt:        gpuvmem
CMakeLists.txt:                   CUDA_SEPARABLE_COMPILATION ON
CMakeLists.txt:                   CUDA_STANDARD 11)
CMakeLists.txt:      set(CMAKE_CUDA_FLAGS
CMakeLists.txt:          "${CMAKE_CUDA_FLAGS} -Xptxas -O3 -Xcompiler -fopenmp")
CMakeLists.txt:  set(ALL_CUDA_LIBRARIES ${CUDA_cuda_LIBRARY} ${CUDA_cudart_LIBRARY}
CMakeLists.txt:                         ${CUDA_cufft_LIBRARY})
CMakeLists.txt:  target_link_libraries(gpuvmem m stdc++ gomp ${ALL_CUDA_LIBRARIES}
CMakeLists.txt:  install(TARGETS gpuvmem DESTINATION bin)
CMakeLists.txt:           ${BINARY_DIR}/gpuvmem ${TEST_DIRECTORY}/antennae)
CMakeLists.txt:  add_test(co65 bash ${TEST_DIRECTORY}/co65/test.sh ${BINARY_DIR}/gpuvmem
CMakeLists.txt:  add_test(freq78 bash ${TEST_DIRECTORY}/FREQ78/test.sh ${BINARY_DIR}/gpuvmem
CMakeLists.txt:  add_test(m87 bash ${TEST_DIRECTORY}/M87/test.sh ${BINARY_DIR}/gpuvmem
CMakeLists.txt:           ${BINARY_DIR}/gpuvmem ${TEST_DIRECTORY}/selfcalband9)
.gitignore:*.gpu
environment_cudatoolkit.yml:name: gpuvmem-cudatoolkit-env
environment_cudatoolkit.yml:- nvidia
environment_cudatoolkit.yml:- cuda-toolkit
environment_cudatoolkit.yml:- cuda-samples
src/directioncosines.cu:   Additionally, this program uses some NVIDIA routines whose copyright is held
src/directioncosines.cu:   by NVIDIA end user license agreement (EULA).
src/chi2.cu:  checkCudaErrors(
src/chi2.cu:      cudaMalloc((void**)&result_dchi2, sizeof(float) * M * N * image_count));
src/chi2.cu:  checkCudaErrors(
src/chi2.cu:      cudaMemset(result_dchi2, 0, sizeof(float) * M * N * image_count));
src/chi2.cu:  checkCudaErrors(
src/chi2.cu:      cudaMemset(result_dchi2, 0, sizeof(float) * M * N * image_count));
src/chi2.cu:    checkCudaErrors(
src/chi2.cu:        cudaMemset(device_dphi, 0, sizeof(float) * M * N * image_count));
src/chi2.cu:    checkCudaErrors(cudaMemcpy(device_dphi, result_dchi2,
src/chi2.cu:                               cudaMemcpyDeviceToDevice));
src/entropy.cu:  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
src/entropy.cu:  cudaFree(this->device_S);
src/entropy.cu:  cudaFree(this->device_DS);
src/gridding.cu:extern int num_gpus;
src/imageProcessor.cu:    checkCudaErrors(
src/imageProcessor.cu:        cudaMalloc((void**)&chain, sizeof(float) * M * N * image_count));
src/imageProcessor.cu:    checkCudaErrors(cudaMemset(chain, 0, sizeof(float) * M * N * image_count));
src/main.cu:   Additionally, this program uses some NVIDIA routines whose copyright is held
src/main.cu:   by NVIDIA end user license agreement (EULA).
src/main.cu:int num_gpus;
src/main.cu:   This is a function that runs gpuvmem and calculates new regularization values
src/main.cu:std::vector<float> runGpuvmem(std::vector<float> args,
src/main.cu:  ////CHECK FOR AVAILABLE GPUs
src/main.cu:  cudaGetDeviceCount(&num_gpus);
src/main.cu:      "gpuvmem Copyright (C) 2016-2020  Miguel Carcamo, Pablo Roman, Simon "
src/main.cu:  if (num_gpus < 1) {
src/main.cu:    printf("No CUDA capable devices were detected\n");
src/main.cu:  sy->setDevice();  // This routine sends the data to GPU memory
src/main.cu:     std::vector<float> final_lambdas = fixedPointOpt(lambdas, &runGpuvmem,
src/totalvariation.cu:  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
src/totalvariation.cu:  cudaFree(this->device_S);
src/totalvariation.cu:  cudaFree(this->device_DS);
src/brent.cu:   Additionally, this program uses some NVIDIA routines whose copyright is held
src/brent.cu:   by NVIDIA end user license agreement (EULA).
src/l1norm.cu:  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
src/l1norm.cu:  cudaFree(this->device_S);
src/l1norm.cu:  cudaFree(this->device_DS);
src/sinc2D.cu:  this->copyKerneltoGPU();
src/sinc2D.cu:  this->copyKerneltoGPU();
src/quadraticpenalization.cu:  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
src/quadraticpenalization.cu:  cudaFree(this->device_S);
src/quadraticpenalization.cu:  cudaFree(this->device_DS);
src/rvgs.cu:   Additionally, this program uses some NVIDIA routines whose copyright is held
src/rvgs.cu:   by NVIDIA end user license agreement (EULA).
src/frprmn.cu:   Additionally, this program uses some NVIDIA routines whose copyright is held
src/frprmn.cu:   by NVIDIA end user license agreement (EULA).
src/frprmn.cu:  cudaFree(device_gg_vector);  \
src/frprmn.cu:  cudaFree(device_dgg_vector); \
src/frprmn.cu:  cudaFree(xi);                \
src/frprmn.cu:  cudaFree(device_h);          \
src/frprmn.cu:  cudaFree(device_g);          \
src/frprmn.cu:  cudaFree(temp);
src/frprmn.cu:__host__ void ConjugateGradient::allocateMemoryGpu() {
src/frprmn.cu:  checkCudaErrors(cudaMalloc((void**)&device_g,
src/frprmn.cu:  checkCudaErrors(
src/frprmn.cu:      cudaMemset(device_g, 0, sizeof(float) * M * N * image->getImageCount()));
src/frprmn.cu:  checkCudaErrors(cudaMalloc((void**)&device_h,
src/frprmn.cu:  checkCudaErrors(
src/frprmn.cu:      cudaMemset(device_h, 0, sizeof(float) * M * N * image->getImageCount()));
src/frprmn.cu:  checkCudaErrors(
src/frprmn.cu:      cudaMalloc((void**)&xi, sizeof(float) * M * N * image->getImageCount()));
src/frprmn.cu:  checkCudaErrors(
src/frprmn.cu:      cudaMemset(xi, 0, sizeof(float) * M * N * image->getImageCount()));
src/frprmn.cu:  checkCudaErrors(cudaMalloc((void**)&temp,
src/frprmn.cu:  checkCudaErrors(
src/frprmn.cu:      cudaMemset(temp, 0, sizeof(float) * M * N * image->getImageCount()));
src/frprmn.cu:  checkCudaErrors(cudaMalloc((void**)&device_gg_vector, sizeof(float) * M * N));
src/frprmn.cu:  checkCudaErrors(cudaMemset(device_gg_vector, 0, sizeof(float) * M * N));
src/frprmn.cu:  checkCudaErrors(
src/frprmn.cu:      cudaMalloc((void**)&device_dgg_vector, sizeof(float) * M * N));
src/frprmn.cu:  checkCudaErrors(cudaMemset(device_dgg_vector, 0, sizeof(float) * M * N));
src/frprmn.cu:__host__ void ConjugateGradient::deallocateMemoryGpu(){FREEALL};
src/frprmn.cu:  allocateMemoryGpu();
src/frprmn.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/frprmn.cu:      deallocateMemoryGpu();
src/frprmn.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/frprmn.cu:      deallocateMemoryGpu();
src/frprmn.cu:    checkCudaErrors(cudaMemset(device_gg_vector, 0, sizeof(float) * M * N));
src/frprmn.cu:    checkCudaErrors(cudaMemset(device_dgg_vector, 0, sizeof(float) * M * N));
src/frprmn.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/frprmn.cu:      deallocateMemoryGpu();
src/frprmn.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/frprmn.cu:  deallocateMemoryGpu();
src/complexOps.cu:  float phase = atan2f(c.y, c.x) * 180.0f / CUDART_PI_F;
src/complexOps.cu:  double phase = atan2(c.y, c.x) * 180.0 / CUDART_PI;
src/mfs.cu:int multigpu, firstgpu, reg_term, total_visibilities, image_count,
src/mfs.cu:extern int num_gpus;
src/mfs.cu:varsPerGPU* vars_gpu;
src/mfs.cu:inline bool IsGPUCapableP2P(cudaDeviceProp* pProp) {
src/mfs.cu:  cudaGetDeviceCount(&num_gpus);
src/mfs.cu:  cudaDeviceProp dprop[num_gpus];
src/mfs.cu:  printf("Number of CUDA devices:\t%d\n", num_gpus);
src/mfs.cu:  for (int i = 0; i < num_gpus; i++) {
src/mfs.cu:    checkCudaErrors(cudaGetDeviceProperties(&dprop[i], i));
src/mfs.cu:    printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i,
src/mfs.cu:           dprop[i].name, (IsGPUCapableP2P(&dprop[i]) ? "IS " : "NOT"));
src/mfs.cu:  multigpu = 0;
src/mfs.cu:  firstgpu = 0;
src/mfs.cu:  int count_gpus;
src/mfs.cu:  string_values = countAndSeparateStrings(variables.gpus, ",");
src/mfs.cu:  count_gpus = string_values.size();
src/mfs.cu:  if (count_gpus == 0) {
src/mfs.cu:    multigpu = 0;
src/mfs.cu:    firstgpu = 0;
src/mfs.cu:  } else if (count_gpus == 1) {
src/mfs.cu:    multigpu = 0;
src/mfs.cu:    firstgpu = std::stoi(string_values[0]);
src/mfs.cu:    multigpu = count_gpus;
src/mfs.cu:    firstgpu = std::stoi(string_values[0]);
src/mfs.cu:  this->ckernel->setGPUID(firstgpu);
src/mfs.cu:  if (multigpu < 0 || multigpu > num_gpus) {
src/mfs.cu:        "ERROR. NUMBER OF GPUS CANNOT BE NEGATIVE OR GREATER THAN THE NUMBER "
src/mfs.cu:        "OF GPUS\n");
src/mfs.cu:    if (multigpu == 0) {
src/mfs.cu:      num_gpus = 1;
src/mfs.cu:        printf("ONLY ONE FREQUENCY. CHANGING NUMBER OF GPUS TO 1\n");
src/mfs.cu:        num_gpus = 1;
src/mfs.cu:        num_gpus = multigpu;
src/mfs.cu:  int total_gpus;
src/mfs.cu:  cudaGetDeviceCount(&total_gpus);
src/mfs.cu:  if (firstgpu > total_gpus - 1 || firstgpu < 0) {
src/mfs.cu:    printf("ERROR. The selected GPU ID does not exist\n");
src/mfs.cu:    printf("Number of CUDA devices and threads: %d\n", num_gpus);
src/mfs.cu:  // Check peer access if there is more than 1 GPU
src/mfs.cu:  if (num_gpus > 1) {
src/mfs.cu:    for (int i = firstgpu + 1; i < firstgpu + num_gpus; i++) {
src/mfs.cu:      cudaDeviceProp dprop0, dpropX;
src/mfs.cu:      cudaGetDeviceProperties(&dprop0, firstgpu);
src/mfs.cu:      cudaGetDeviceProperties(&dpropX, i);
src/mfs.cu:      cudaDeviceCanAccessPeer(&canAccessPeer0_x, firstgpu, i);
src/mfs.cu:      cudaDeviceCanAccessPeer(&canAccessPeerx_0, i, firstgpu);
src/mfs.cu:            "> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n",
src/mfs.cu:            dprop0.name, firstgpu, dpropX.name, i,
src/mfs.cu:            "> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n",
src/mfs.cu:            dpropX.name, i, dprop0.name, firstgpu,
src/mfs.cu:        printf("Number of GPUs: %d\n", num_gpus);
src/mfs.cu:        printf("Two or more SM 2.0 class GPUs are required for %s to run.\n",
src/mfs.cu:        printf("Support for UVA requires a GPU with SM 2.0 capabilities.\n");
src/mfs.cu:            "Peer to Peer access is not available between GPU%d <-> GPU%d, "
src/mfs.cu:        cudaSetDevice(firstgpu);
src/mfs.cu:          printf("Granting access from %d to %d...\n", firstgpu, i);
src/mfs.cu:        cudaDeviceEnablePeerAccess(i, 0);
src/mfs.cu:        cudaSetDevice(i);
src/mfs.cu:          printf("Granting access from %d to %d...\n", i, firstgpu);
src/mfs.cu:        cudaDeviceEnablePeerAccess(firstgpu, 0);
src/mfs.cu:          printf("Checking GPU %d and GPU %d for UVA capabilities...\n",
src/mfs.cu:                 firstgpu, i);
src/mfs.cu:          printf("> %s (GPU%d) supports UVA: %s\n", dprop0.name, firstgpu,
src/mfs.cu:          printf("> %s (GPU%d) supports UVA: %s\n", dpropX.name, i,
src/mfs.cu:            printf("Both GPUs can support UVA, enabling...\n");
src/mfs.cu:              "At least one of the two GPUs does NOT support UVA, waiving "
src/mfs.cu:  vars_gpu = (varsPerGPU*)malloc(num_gpus * sizeof(varsPerGPU));
src/mfs.cu:      cudaSetDevice(firstgpu);
src/mfs.cu:      checkCudaErrors(cudaMalloc((void**)&datasets[d].fields[f].atten_image,
src/mfs.cu:      checkCudaErrors(cudaMemset(datasets[d].fields[f].atten_image, 0,
src/mfs.cu:        cudaSetDevice((i % num_gpus) + firstgpu);
src/mfs.cu:          checkCudaErrors(cudaMalloc(
src/mfs.cu:          checkCudaErrors(cudaMalloc(
src/mfs.cu:          checkCudaErrors(cudaMalloc(
src/mfs.cu:          checkCudaErrors(cudaMalloc(
src/mfs.cu:          checkCudaErrors(cudaMalloc(
src/mfs.cu:          checkCudaErrors(cudaMemcpy(
src/mfs.cu:              cudaMemcpyHostToDevice));
src/mfs.cu:          checkCudaErrors(cudaMemcpy(
src/mfs.cu:              cudaMemcpyHostToDevice));
src/mfs.cu:          checkCudaErrors(cudaMemcpy(
src/mfs.cu:              cudaMemcpyHostToDevice));
src/mfs.cu:          checkCudaErrors(cudaMemset(
src/mfs.cu:          checkCudaErrors(cudaMemset(
src/mfs.cu:  printf("gpuvmem estimated beam size: %e x %e (arcsec) / %lf (degrees)\n",
src/mfs.cu:  ////////////////////////////////////////////////CUDA MEMORY ALLOCATION FOR
src/mfs.cu:  for (int g = 0; g < num_gpus; g++) {
src/mfs.cu:    cudaSetDevice((g % num_gpus) + firstgpu);
src/mfs.cu:    checkCudaErrors(
src/mfs.cu:        cudaMalloc(&vars_gpu[g].device_V, sizeof(cufftComplex) * M * N));
src/mfs.cu:    checkCudaErrors(
src/mfs.cu:        cudaMalloc(&vars_gpu[g].device_I_nu, sizeof(cufftComplex) * M * N));
src/mfs.cu:    checkCudaErrors(
src/mfs.cu:        cudaMalloc(&vars_gpu[g].device_chi2, sizeof(float) * max_number_vis));
src/mfs.cu:    checkCudaErrors(
src/mfs.cu:        cudaMalloc(&vars_gpu[g].device_dchi2, sizeof(float) * M * N));
src/mfs.cu:    checkCudaErrors(
src/mfs.cu:        cudaMemset(vars_gpu[g].device_V, 0, sizeof(cufftComplex) * M * N));
src/mfs.cu:    checkCudaErrors(
src/mfs.cu:        cudaMemset(vars_gpu[g].device_I_nu, 0, sizeof(cufftComplex) * M * N));
src/mfs.cu:    checkCudaErrors(
src/mfs.cu:        cudaMemset(vars_gpu[g].device_chi2, 0, sizeof(float) * max_number_vis));
src/mfs.cu:    checkCudaErrors(
src/mfs.cu:        cudaMemset(vars_gpu[g].device_dchi2, 0, sizeof(float) * M * N));
src/mfs.cu:  cudaSetDevice(firstgpu);
src/mfs.cu:  checkCudaErrors(
src/mfs.cu:      cudaMalloc((void**)&device_Image, sizeof(float) * M * N * image_count));
src/mfs.cu:  checkCudaErrors(
src/mfs.cu:      cudaMemset(device_Image, 0, sizeof(float) * M * N * image_count));
src/mfs.cu:  checkCudaErrors(cudaMemcpy(device_Image, host_I,
src/mfs.cu:                             cudaMemcpyHostToDevice));
src/mfs.cu:  checkCudaErrors(
src/mfs.cu:      cudaMalloc((void**)&device_noise_image, sizeof(float) * M * N));
src/mfs.cu:  checkCudaErrors(cudaMemset(device_noise_image, 0, sizeof(float) * M * N));
src/mfs.cu:  checkCudaErrors(
src/mfs.cu:      cudaMalloc((void**)&device_weight_image, sizeof(float) * M * N));
src/mfs.cu:  checkCudaErrors(cudaMemset(device_weight_image, 0, sizeof(float) * M * N));
src/mfs.cu:    checkCudaErrors(
src/mfs.cu:        cudaMalloc((void**)&device_distance_image, sizeof(float) * M * N));
src/mfs.cu:  initFFT(vars_gpu, M, N, firstgpu, num_gpus);
src/mfs.cu:#pragma omp parallel for schedule(static, 1) num_threads(num_gpus)
src/mfs.cu:        int gpu_idx = i % num_gpus;
src/mfs.cu:        cudaSetDevice(gpu_idx + firstgpu);
src/mfs.cu:        int gpu_id = -1;
src/mfs.cu:        cudaGetDevice(&gpu_id);
src/mfs.cu:            checkCudaErrors(cudaDeviceSynchronize());
src/mfs.cu:    cudaSetDevice(firstgpu);
src/mfs.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/mfs.cu:  cudaSetDevice(firstgpu);
src/mfs.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/mfs.cu:        checkCudaErrors(cudaDeviceSynchronize());
src/mfs.cu:  checkCudaErrors(cudaMemcpy2D(host_weight_image, sizeof(float),
src/mfs.cu:                               sizeof(float), M * N, cudaMemcpyDeviceToHost));
src/mfs.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/mfs.cu:  checkCudaErrors(cudaMemcpy2D(host_noise_image, sizeof(float),
src/mfs.cu:                               M * N, cudaMemcpyDeviceToHost));
src/mfs.cu:    checkCudaErrors(cudaMemcpy2D(device_noise_image, sizeof(float),
src/mfs.cu:                                 M * N, cudaMemcpyHostToDevice));
src/mfs.cu:    checkCudaErrors(cudaMemcpy2D(
src/mfs.cu:        sizeof(float), M * N, cudaMemcpyDeviceToDevice));
src/mfs.cu:  cudaFree(device_weight_image);
src/mfs.cu:    cudaFree(device_distance_image);
src/mfs.cu:      cudaFree(datasets[d].fields[f].atten_image);
src/mfs.cu:        cudaSetDevice((i % num_gpus) + firstgpu);
src/mfs.cu:          checkCudaErrors(cudaMemset(
src/mfs.cu:          checkCudaErrors(cudaMemset(
src/mfs.cu:  for (int g = 0; g < num_gpus; g++) {
src/mfs.cu:    cudaSetDevice((g % num_gpus) + firstgpu);
src/mfs.cu:    checkCudaErrors(
src/mfs.cu:        cudaMemset(vars_gpu[g].device_V, 0, sizeof(cufftComplex) * M * N));
src/mfs.cu:    checkCudaErrors(
src/mfs.cu:        cudaMemset(vars_gpu[g].device_I_nu, 0, sizeof(cufftComplex) * M * N));
src/mfs.cu:  cudaSetDevice(firstgpu);
src/mfs.cu:  checkCudaErrors(cudaMemcpy(device_Image, host_I,
src/mfs.cu:                             cudaMemcpyHostToDevice));
src/mfs.cu:      // num_gpus,
src/mfs.cu:      //            firstgpu, variables.blockSizeV, M, N, this->ckernel);
src/mfs.cu:                                  num_gpus, firstgpu, variables.blockSizeV);
src/mfs.cu:    modelToHost(datasets[d].fields, datasets[d].data, num_gpus, firstgpu);
src/mfs.cu:  cudaSetDevice(firstgpu);
src/mfs.cu:        cudaSetDevice((i % num_gpus) + firstgpu);
src/mfs.cu:          cudaFree(datasets[d].fields[f].device_visibilities[i][s].uvw);
src/mfs.cu:          cudaFree(datasets[d].fields[f].device_visibilities[i][s].weight);
src/mfs.cu:          cudaFree(datasets[d].fields[f].device_visibilities[i][s].Vr);
src/mfs.cu:          cudaFree(datasets[d].fields[f].device_visibilities[i][s].Vm);
src/mfs.cu:          cudaFree(datasets[d].fields[f].device_visibilities[i][s].Vo);
src/mfs.cu:  for (int g = 0; g < num_gpus; g++) {
src/mfs.cu:    cudaSetDevice((g % num_gpus) + firstgpu);
src/mfs.cu:    cufftDestroy(vars_gpu[g].plan);
src/mfs.cu:  cudaSetDevice(firstgpu);
src/mfs.cu:  cudaFree(device_Image);
src/mfs.cu:  for (int g = 0; g < num_gpus; g++) {
src/mfs.cu:    cudaSetDevice((g % num_gpus) + firstgpu);
src/mfs.cu:    cudaFree(vars_gpu[g].device_V);
src/mfs.cu:    cudaFree(vars_gpu[g].device_I_nu);
src/mfs.cu:  cudaSetDevice(firstgpu);
src/mfs.cu:  cudaFree(device_noise_image);
src/mfs.cu:  cudaFree(device_dphi);
src/mfs.cu:  cudaFree(device_dchi2_total);
src/mfs.cu:  cudaFree(device_dS);
src/mfs.cu:  cudaFree(device_S);
src/mfs.cu:  if (num_gpus > 1) {
src/mfs.cu:    for (int i = firstgpu + 1; i < num_gpus + firstgpu; i++) {
src/mfs.cu:      cudaSetDevice(firstgpu);
src/mfs.cu:      cudaDeviceDisablePeerAccess(i);
src/mfs.cu:      cudaSetDevice(i);
src/mfs.cu:      cudaDeviceDisablePeerAccess(firstgpu);
src/mfs.cu:    for (int i = 0; i < num_gpus; i++) {
src/mfs.cu:      cudaSetDevice((i % num_gpus) + firstgpu);
src/mfs.cu:      cudaDeviceReset();
src/linmin.cu:   Additionally, this program uses some NVIDIA routines whose copyright is held
src/linmin.cu:   by NVIDIA end user license agreement (EULA).
src/linmin.cu:                     float (*func)(float*))  // p and xi are in GPU
src/linmin.cu:  checkCudaErrors(
src/linmin.cu:      cudaMalloc((void**)&device_pcom, sizeof(float) * M * N * image_count));
src/linmin.cu:  checkCudaErrors(
src/linmin.cu:      cudaMemset(device_pcom, 0, sizeof(float) * M * N * image_count));
src/linmin.cu:  checkCudaErrors(
src/linmin.cu:      (cudaMalloc((void**)&device_xicom, sizeof(float) * M * N * image_count)));
src/linmin.cu:  checkCudaErrors(
src/linmin.cu:      cudaMemset(device_xicom, 0, sizeof(float) * M * N * image_count));
src/linmin.cu:  checkCudaErrors(cudaMemcpy(device_pcom, p,
src/linmin.cu:                             cudaMemcpyDeviceToDevice));
src/linmin.cu:  checkCudaErrors(cudaMemcpy(device_xicom, xi,
src/linmin.cu:                             cudaMemcpyDeviceToDevice));
src/linmin.cu:  // GPU MUL AND ADD
src/linmin.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/linmin.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/linmin.cu:       checkCudaErrors(cudaDeviceSynchronize());*/
src/linmin.cu:  cudaFree(device_xicom);
src/linmin.cu:  cudaFree(device_pcom);
src/gl1norm.cu:  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * M * N));
src/gl1norm.cu:  checkCudaErrors(
src/gl1norm.cu:      cudaMemcpy(this->prior, prior.data(), M * N, cudaMemcpyHostToDevice));
src/gl1norm.cu:  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * M * N));
src/gl1norm.cu:  checkCudaErrors(
src/gl1norm.cu:      cudaMemcpy(this->prior, prior.data(), M * N, cudaMemcpyHostToDevice));
src/gl1norm.cu:  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * M * N));
src/gl1norm.cu:  checkCudaErrors(
src/gl1norm.cu:      cudaMemcpy(this->prior, prior.data(), M * N, cudaMemcpyHostToDevice));
src/gl1norm.cu:  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * M * N));
src/gl1norm.cu:  checkCudaErrors(
src/gl1norm.cu:      cudaMemcpy(this->prior, prior.data(), M * N, cudaMemcpyHostToDevice));
src/gl1norm.cu:  cudaFree(this->prior);
src/gl1norm.cu:  cudaFree(this->prior);
src/gl1norm.cu:  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
src/gl1norm.cu:  cudaFree(this->device_S);
src/gl1norm.cu:  cudaFree(this->device_DS);
src/mnbrak.cu:   Additionally, this program uses some NVIDIA routines whose copyright is held
src/mnbrak.cu:   by NVIDIA end user license agreement (EULA).
src/gaussian2D.cu:  this->copyKerneltoGPU();
src/gaussian2D.cu:  this->copyKerneltoGPU();
src/gaussian2D.cu:  this->copyKerneltoGPU();
src/gaussian2D.cu:  this->copyKerneltoGPU();
src/MSFITSIO.cu:   Additionally, this program uses some NVIDIA routines whose copyright is held
src/MSFITSIO.cu:   by NVIDIA end user license agreement (EULA).
src/MSFITSIO.cu:                        bool isInGPU) {
src/MSFITSIO.cu:                  "Number of iteration in gpuvmem software", &status);
src/MSFITSIO.cu:                  "Changed by gpuvmem", &status);
src/MSFITSIO.cu:  fits_update_key(fpointer, TFLOAT, "EQUINOX", &equinox, "Changed by gpuvmem",
src/MSFITSIO.cu:  fits_update_key(fpointer, TDOUBLE, "CRVAL1", &ra_center, "Changed by gpuvmem",
src/MSFITSIO.cu:                  "Changed by gpuvmem", &status);
src/MSFITSIO.cu:  if (isInGPU) {
src/MSFITSIO.cu:    checkCudaErrors(cudaMemcpy(host_IFITS, &I[offset], sizeof(float) * M * N,
src/MSFITSIO.cu:                               cudaMemcpyDeviceToHost));
src/MSFITSIO.cu:                                    bool isInGPU) {
src/MSFITSIO.cu:                  "Number of iteration in gpuvmem software", &status);
src/MSFITSIO.cu:  if (isInGPU) {
src/MSFITSIO.cu:    checkCudaErrors(cudaMemcpy2D(host_IFITS, sizeof(cufftComplex), I,
src/MSFITSIO.cu:                                 M * N, cudaMemcpyDeviceToHost));
src/MSFITSIO.cu:  printf("GPUVMEM is reading %s data column\n", data_column.c_str());
src/MSFITSIO.cu:  printf("GPUVMEM is reading %s data column\n", data_column.c_str());
src/MSFITSIO.cu:                          int num_gpus,
src/MSFITSIO.cu:                          int firstgpu) {
src/MSFITSIO.cu:      cudaSetDevice((i % num_gpus) + firstgpu);
src/MSFITSIO.cu:        checkCudaErrors(
src/MSFITSIO.cu:            cudaMemcpy(fields[f].visibilities[i][s].Vm.data(),
src/MSFITSIO.cu:                       cudaMemcpyDeviceToHost));
src/MSFITSIO.cu:        column_name, "created by gpuvmem"));
src/gentropy.cu:  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * M * N));
src/gentropy.cu:  checkCudaErrors(
src/gentropy.cu:      cudaMemcpy(this->prior, prior.data(), M * N, cudaMemcpyHostToDevice));
src/gentropy.cu:  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * M * N));
src/gentropy.cu:  checkCudaErrors(
src/gentropy.cu:      cudaMemcpy(this->prior, prior.data(), M * N, cudaMemcpyHostToDevice));
src/gentropy.cu:  checkCudaErrors(cudaMalloc((void**)&this->prior, sizeof(float) * M * N));
src/gentropy.cu:  checkCudaErrors(
src/gentropy.cu:      cudaMemcpy(this->prior, prior.data(), M * N, cudaMemcpyHostToDevice));
src/gentropy.cu:  cudaFree(this->prior);
src/gentropy.cu:  cudaFree(this->prior);
src/gentropy.cu:  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
src/gentropy.cu:  cudaFree(this->device_S);
src/gentropy.cu:  cudaFree(this->device_DS);
src/gaussianSinc2D.cu:  this->copyKerneltoGPU();
src/gaussianSinc2D.cu:  this->copyKerneltoGPU();
src/rngs.cu:   Additionally, this program uses some NVIDIA routines whose copyright is held
src/rngs.cu:   by NVIDIA end user license agreement (EULA).
src/functions.cu:   Additionally, this program uses some NVIDIA routines whose copyright is held
src/functions.cu:   by NVIDIA end user license agreement (EULA).
src/functions.cu:extern int iterations, iter, image_count, status_mod_in, flag_opt, num_gpus,
src/functions.cu:    multigpu, firstgpu, reg_term;
src/functions.cu:extern cufftHandle plan1GPU;
src/functions.cu:extern varsPerGPU* vars_gpu;
src/functions.cu:  if (num_gpus > 1) {
src/functions.cu:    for (int i = firstgpu + 1; i < firstgpu + num_gpus; i++) {
src/functions.cu:      cudaSetDevice(firstgpu);
src/functions.cu:      cudaDeviceDisablePeerAccess(i);
src/functions.cu:      cudaSetDevice(i);
src/functions.cu:      cudaDeviceDisablePeerAccess(firstgpu);
src/functions.cu:    for (int i = 0; i < num_gpus; i++) {
src/functions.cu:      cudaSetDevice((i % num_gpus) + firstgpu);
src/functions.cu:      cudaDeviceReset();
src/functions.cu:  flags.Var(variables.gpus, 'G', "gpus", std::string("0"),
src/functions.cu:            "Index of the GPU/s you are going to use separated by a comma");
src/functions.cu:            "GPU block X Size for image/Fourier plane (Needs to be pow of 2)");
src/functions.cu:            "GPU block Y Size for image/Fourier plane (Needs to be pow of 2)");
src/functions.cu:            "GPU block V Size for visibilities (Needs to be pow of 2)");
src/functions.cu:             "Runs gpuvmem with no positivity restrictions on the images",
src/functions.cu:             "Modify Measurement Set WEIGHT column with gpuvmem weights",
src/functions.cu:  cudaDeviceProp prop;
src/functions.cu:  checkCudaErrors(cudaGetDevice(&device));
src/functions.cu:  checkCudaErrors(cudaGetDeviceProperties(&prop, device));
src/functions.cu:  checkCudaErrors(cudaMalloc((void**)&d_odata, blocks * sizeof(T)));
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  checkCudaErrors(
src/functions.cu:      cudaMemcpy(h_odata, d_odata, blocks * sizeof(T), cudaMemcpyDeviceToHost));
src/functions.cu:  cudaFree(d_odata);
src/functions.cu:  checkCudaErrors(cudaMalloc((void**)&d_odata, blocks * sizeof(float)));
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  checkCudaErrors(cudaMemcpy(h_odata, d_odata, blocks * sizeof(float),
src/functions.cu:                             cudaMemcpyDeviceToHost));
src/functions.cu:  cudaFree(d_odata);
src/functions.cu:  checkCudaErrors(cudaMalloc((void**)&d_odata, blocks * sizeof(float)));
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  checkCudaErrors(cudaMemcpy(h_odata, d_odata, blocks * sizeof(float),
src/functions.cu:                             cudaMemcpyDeviceToHost));
src/functions.cu:  cudaFree(d_odata);
src/functions.cu:    checkCudaErrors(cudaMalloc((void**)&kernel_device, sizeof(T) * M * N));
src/functions.cu:    checkCudaErrors(cudaMemcpy(kernel_device, kernel, sizeof(T) * M * N,
src/functions.cu:                               cudaMemcpyHostToDevice));
src/functions.cu:    checkCudaErrors(cudaMalloc((void**)&data_device, sizeof(TD) * M * N));
src/functions.cu:    checkCudaErrors(cudaMemcpy(data_device, data, sizeof(TD) * M * N,
src/functions.cu:                               cudaMemcpyHostToDevice));
src/functions.cu:  checkCudaErrors(
src/functions.cu:      cudaMalloc((void**)&kernel_complex_device, sizeof(TD) * M * N));
src/functions.cu:  checkCudaErrors(cudaMemset(kernel_complex_device, 0, sizeof(TD) * M * N));
src/functions.cu:  checkCudaErrors(cudaMemcpy(kernel_complex_device, kernel_device,
src/functions.cu:                             sizeof(T) * M * N, cudaMemcpyDeviceToDevice));
src/functions.cu:  checkCudaErrors(cudaMalloc((void**)&padded_kernel_complex,
src/functions.cu:  checkCudaErrors(
src/functions.cu:      cudaMemset(padded_kernel_complex, 0, sizeof(TD) * padding_M * padding_N));
src/functions.cu:  checkCudaErrors(cudaMalloc((void**)&padded_data_complex,
src/functions.cu:  checkCudaErrors(
src/functions.cu:      cudaMemset(padded_data_complex, 0, sizeof(TD) * padding_M * padding_N));
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  checkCudaErrors(cudaMalloc((void**)&data_spectrum_device,
src/functions.cu:  checkCudaErrors(cudaMalloc((void**)&kernel_spectrum_device,
src/functions.cu:  checkCudaErrors(cufftPlan2d(&fftPlan, padding_M, padding_N, CUFFT_R2C));
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:    checkCudaErrors(cudaMemcpy(data, data_device, sizeof(TD) * M * N,
src/functions.cu:                               cudaMemcpyDeviceToHost));
src/functions.cu:    checkCudaErrors(cudaMemcpy(result_host, data_device, sizeof(TD) * M * N,
src/functions.cu:                               cudaMemcpyDeviceToHost));
src/functions.cu:  // Free GPU MEMORY
src/functions.cu:  checkCudaErrors(cudaFree(kernel_device));
src/functions.cu:  checkCudaErrors(cudaFree(kernel_complex_device));
src/functions.cu:  checkCudaErrors(cudaFree(padded_kernel_complex));
src/functions.cu:  checkCudaErrors(cudaFree(padded_data_complex));
src/functions.cu:  checkCudaErrors(cudaFree(data_spectrum_device));
src/functions.cu:  checkCudaErrors(cudaFree(kernel_spectrum_device));
src/functions.cu:  checkCudaErrors(cudaFree(padded_data_complex));
src/functions.cu:  checkCudaErrors(cudaFree(data_device));
src/functions.cu:  checkCudaErrors(cufftDestroy(fftPlan));
src/functions.cu:#if (__CUDA_ARCH__ >= 300)
src/functions.cu:  cudaDeviceProp dprop;
src/functions.cu:  checkCudaErrors(cudaGetDevice(&device));
src/functions.cu:  checkCudaErrors(cudaGetDeviceProperties(&dprop, device));
src/functions.cu:                                          int num_gpus,
src/functions.cu:                                          int firstgpu,
src/functions.cu:#pragma omp parallel for schedule(static, 1) num_threads(num_gpus)
src/functions.cu:      int gpu_idx = i % num_gpus;
src/functions.cu:      cudaSetDevice(gpu_idx + firstgpu);
src/functions.cu:      int gpu_id = -1;
src/functions.cu:      cudaGetDevice(&gpu_id);
src/functions.cu:        checkCudaErrors(
src/functions.cu:            cudaMalloc(&fields[f].device_visibilities[i][s].Vm,
src/functions.cu:        checkCudaErrors(
src/functions.cu:            cudaMemset(fields[f].device_visibilities[i][s].Vm, 0,
src/functions.cu:        checkCudaErrors(
src/functions.cu:            cudaMalloc(&fields[f].device_visibilities[i][s].Vr,
src/functions.cu:        checkCudaErrors(
src/functions.cu:            cudaMemset(fields[f].device_visibilities[i][s].Vr, 0,
src/functions.cu:        checkCudaErrors(cudaMalloc(
src/functions.cu:        checkCudaErrors(
src/functions.cu:            cudaMalloc(&fields[f].device_visibilities[i][s].Vo,
src/functions.cu:        checkCudaErrors(cudaMalloc(
src/functions.cu:        checkCudaErrors(cudaMemcpy(
src/functions.cu:            cudaMemcpyHostToDevice));
src/functions.cu:        checkCudaErrors(cudaMemcpy(
src/functions.cu:            cudaMemcpyHostToDevice));
src/functions.cu:        checkCudaErrors(cudaMemcpy(
src/functions.cu:            cudaMemcpyHostToDevice));
src/functions.cu:        checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  for (int g = 0; g < num_gpus; g++) {
src/functions.cu:    cudaSetDevice((g % num_gpus) + firstgpu);
src/functions.cu:    checkCudaErrors(
src/functions.cu:        cudaMalloc(&vars_gpu[g].device_chi2, sizeof(float) * max_number_vis));
src/functions.cu:    checkCudaErrors(
src/functions.cu:        cudaMemset(vars_gpu[g].device_chi2, 0, sizeof(float) * max_number_vis));
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:                         int num_gpus,
src/functions.cu:                         int firstgpu,
src/functions.cu:  modelToHost(fields, data, num_gpus, firstgpu);
src/functions.cu:      num_gpus, std::vector<cufftComplex>(M * N));
src/functions.cu:#pragma omp parallel for schedule(static, 1) num_threads(num_gpus)
src/functions.cu:      int gpu_idx = i % num_gpus;
src/functions.cu:      cudaSetDevice(gpu_idx + firstgpu);
src/functions.cu:      int gpu_id = -1;
src/functions.cu:      cudaGetDevice(&gpu_id);
src/functions.cu:            gridded_visibilities[gpu_idx], fields[f].visibilities[i][s].Vm,
src/functions.cu:        Model visibilities and original (u,v) positions to GPU.
src/functions.cu:        checkCudaErrors(
src/functions.cu:            cudaMalloc(&fields[f].device_visibilities[i][s].Vm,
src/functions.cu:        checkCudaErrors(
src/functions.cu:            cudaMemset(fields[f].device_visibilities[i][s].Vm, 0,
src/functions.cu:        checkCudaErrors(cudaMalloc(
src/functions.cu:        checkCudaErrors(
src/functions.cu:            cudaMalloc(&fields[f].device_visibilities[i][s].Vo,
src/functions.cu:        checkCudaErrors(cudaMalloc(
src/functions.cu:        checkCudaErrors(cudaMemcpy(
src/functions.cu:            vars_gpu[gpu_idx].device_V, gridded_visibilities[gpu_idx].data(),
src/functions.cu:            sizeof(cufftComplex) * M * N, cudaMemcpyHostToDevice));
src/functions.cu:        checkCudaErrors(cudaMemcpy(
src/functions.cu:            cudaMemcpyHostToDevice));
src/functions.cu:        checkCudaErrors(
src/functions.cu:            cudaMemcpy(fields[f].device_visibilities[i][s].Vo,
src/functions.cu:                       cudaMemcpyHostToDevice));
src/functions.cu:        checkCudaErrors(cudaMemcpy(
src/functions.cu:            cudaMemcpyHostToDevice));
src/functions.cu:        checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:            fields[f].device_visibilities[i][s].Vm, vars_gpu[gpu_idx].device_V,
src/functions.cu:        // degriddingGPU<<< fields[f].device_visibilities[i][s].numBlocksUV,
src/functions.cu:        //             vars_gpu[gpu_idx].device_V, ckernel->getGPUKernel(),
src/functions.cu:        checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:__host__ void initFFT(varsPerGPU* vars_gpu,
src/functions.cu:                      int firstgpu,
src/functions.cu:                      int num_gpus) {
src/functions.cu:  for (int g = 0; g < num_gpus; g++) {
src/functions.cu:    cudaSetDevice((g % num_gpus) + firstgpu);
src/functions.cu:    checkCudaErrors(cufftPlan2d(&vars_gpu[g].plan, N, M, CUFFT_C2C));
src/functions.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  checkCudaErrors(cufftExecC2C(plan, (cufftComplex*)input_data,
src/functions.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:__global__ void do_griddingGPU(float3* uvw,
src/functions.cu:__global__ void degriddingGPU(double3* uvw,
src/functions.cu:#if (__CUDA_ARCH__ >= 300)
src/functions.cu:#if (__CUDA_ARCH__ >= 300)
src/functions.cu:#if (__CUDA_ARCH__ >= 300)
src/functions.cu:#if (__CUDA_ARCH__ >= 300)
src/functions.cu:#if (__CUDA_ARCH__ >= 300)
src/functions.cu:#if (__CUDA_ARCH__ >= 300)
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:#pragma omp parallel for schedule(static, 1) num_threads(num_gpus) \
src/functions.cu:        int gpu_idx = i % num_gpus;
src/functions.cu:        cudaSetDevice(gpu_idx + firstgpu);
src/functions.cu:        int gpu_id = -1;
src/functions.cu:        cudaGetDevice(&gpu_id);
src/functions.cu:        ip->calculateInu(vars_gpu[gpu_idx].device_I_nu, I,
src/functions.cu:        ip->apply_beam(vars_gpu[gpu_idx].device_I_nu,
src/functions.cu:              vars_gpu[gpu_idx].device_I_nu, ip->getCKernel()->getGCFGPU(), N);
src/functions.cu:          checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:        FFT2D(vars_gpu[gpu_idx].device_V, vars_gpu[gpu_idx].device_I_nu,
src/functions.cu:              vars_gpu[gpu_idx].plan, M, N, CUFFT_INVERSE, false);
src/functions.cu:            vars_gpu[gpu_idx].device_V, M, N,
src/functions.cu:        checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:              checkCudaErrors(cudaMemset(vars_gpu[gpu_idx].device_chi2, 0,
src/functions.cu:                  vars_gpu[gpu_idx].device_V,
src/functions.cu:              checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:              checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:                  vars_gpu[gpu_idx].device_chi2,
src/functions.cu:              checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:                  vars_gpu[gpu_idx].device_chi2,
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:#pragma omp parallel for schedule(static, 1) num_threads(num_gpus) \
src/functions.cu:        int gpu_idx = i % num_gpus;
src/functions.cu:        cudaSetDevice(gpu_idx + firstgpu);
src/functions.cu:        int gpu_id = -1;
src/functions.cu:        cudaGetDevice(&gpu_id);
src/functions.cu:        ip->calculateInu(vars_gpu[gpu_idx].device_I_nu, I,
src/functions.cu:        ip->apply_beam(vars_gpu[gpu_idx].device_I_nu,
src/functions.cu:              vars_gpu[gpu_idx].device_I_nu, ip->getCKernel()->getGCFGPU(), N);
src/functions.cu:          checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:        FFT2D(vars_gpu[gpu_idx].device_V, vars_gpu[gpu_idx].device_I_nu,
src/functions.cu:              vars_gpu[gpu_idx].plan, M, N, CUFFT_INVERSE, false);
src/functions.cu:            vars_gpu[gpu_idx].device_V, M, N,
src/functions.cu:        checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:              checkCudaErrors(cudaMemset(vars_gpu[gpu_idx].device_chi2, 0,
src/functions.cu:                  vars_gpu[gpu_idx].device_V,
src/functions.cu:              checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:              checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:                  vars_gpu[gpu_idx].device_chi2,
src/functions.cu:              checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:                  vars_gpu[gpu_idx].device_chi2,
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:#pragma omp parallel for schedule(static, 1) num_threads(num_gpus)
src/functions.cu:        int gpu_idx = i % num_gpus;
src/functions.cu:        cudaSetDevice(gpu_idx + firstgpu);
src/functions.cu:        int gpu_id = -1;
src/functions.cu:        cudaGetDevice(&gpu_id);
src/functions.cu:              checkCudaErrors(cudaMemset(vars_gpu[gpu_idx].device_dchi2, 0,
src/functions.cu:                    device_noise_image, ip->getCKernel()->getGCFGPU(),
src/functions.cu:                    vars_gpu[gpu_idx].device_dchi2,
src/functions.cu:                    device_noise_image, vars_gpu[gpu_idx].device_dchi2,
src/functions.cu:              // vars_gpu[gpu_idx].device_dchi2,
src/functions.cu:              checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:                      vars_gpu[gpu_idx].device_dchi2, I,
src/functions.cu:                      vars_gpu[gpu_idx].device_dchi2, I,
src/functions.cu:                checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  cudaSetDevice(firstgpu);
src/functions.cu:  checkCudaErrors(cudaMalloc((void**)&errors,
src/functions.cu:  checkCudaErrors(
src/functions.cu:      cudaMemset(errors, 0, sizeof(float) * M * N * image->getImageCount()));
src/functions.cu:#pragma omp parallel for private(sum_weights) num_threads(num_gpus) \
src/functions.cu:        int gpu_idx = i % num_gpus;
src/functions.cu:        cudaSetDevice(gpu_idx + firstgpu);
src/functions.cu:        int gpu_id = -1;
src/functions.cu:        cudaGetDevice(&gpu_id);
src/functions.cu:                checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:                checkCudaErrors(cudaDeviceSynchronize());
src/functions.cu:  checkCudaErrors(cudaDeviceSynchronize());
src/pswf_12D.cu:  this->copyKerneltoGPU();
src/pswf_12D.cu:  this->copyKerneltoGPU();
src/pswf_12D.cu:  this->copyKerneltoGPU();
src/pswf_12D.cu:  this->copyKerneltoGPU();
src/iofits.cu:                        bool isInGPU) {
src/iofits.cu:            equinox, isInGPU);
src/iofits.cu:                        bool isInGPU) {
src/iofits.cu:            this->dec, this->frame, this->equinox, isInGPU);
src/iofits.cu:                        bool isInGPU) {
src/iofits.cu:            fg_scale, M, N, ra_center, dec_center, frame, equinox, isInGPU);
src/iofits.cu:                        bool isInGPU) {
src/iofits.cu:            isInGPU);
src/iofits.cu:                               bool isInGPU) {
src/iofits.cu:            fg_scale, M, N, ra_center, dec_center, frame, equinox, isInGPU);
src/iofits.cu:                               bool isInGPU) {
src/iofits.cu:            equinox, isInGPU);
src/iofits.cu:                               bool isInGPU) {
src/iofits.cu:            this->ra, this->dec, this->frame, this->equinox, isInGPU);
src/iofits.cu:                               bool isInGPU) {
src/iofits.cu:            this->dec, this->frame, this->equinox, isInGPU);
src/iofits.cu:                               bool isInGPU) {
src/iofits.cu:            this->frame, this->equinox, isInGPU);
src/iofits.cu:                                     bool isInGPU) {
src/iofits.cu:            this->equinox, isInGPU);
src/iofits.cu:                                            bool isInGPU) {
src/iofits.cu:            this->frame, this->equinox, isInGPU);
src/iofits.cu:                                 bool isInGPU) {
src/iofits.cu:            isInGPU);
src/iofits.cu:                                 bool isInGPU) {
src/iofits.cu:            this->dec, this->frame, this->equinox, isInGPU);
src/iofits.cu:                                              bool isInGPU) {
src/iofits.cu:            this->equinox, isInGPU);
src/iofits.cu:                                 bool isInGPU) {
src/iofits.cu:            M, N, ra_center, dec_center, frame, equinox, isInGPU);
src/iofits.cu:                               bool isInGPU) {
src/iofits.cu:                        iteration, fg_scale, M, N, option, isInGPU);
src/iofits.cu:                               bool isInGPU) {
src/iofits.cu:                        option, isInGPU);
src/iofits.cu:                               bool isInGPU) {
src/iofits.cu:                        option, isInGPU);
src/lbfgs.cu:   Additionally, this program uses some NVIDIA routines whose copyright is held
src/lbfgs.cu:   by NVIDIA end user license agreement (EULA).
src/lbfgs.cu:  cudaFree(d_y);    \
src/lbfgs.cu:  cudaFree(d_s);    \
src/lbfgs.cu:  cudaFree(xi);     \
src/lbfgs.cu:  cudaFree(xi_old); \
src/lbfgs.cu:  cudaFree(p_old);  \
src/lbfgs.cu:  cudaFree(norm_vector);
src/lbfgs.cu:__host__ void LBFGS::allocateMemoryGpu() {
src/lbfgs.cu:  checkCudaErrors(cudaMalloc(
src/lbfgs.cu:  checkCudaErrors(
src/lbfgs.cu:      cudaMemset(d_y, 0, sizeof(float) * M * N * K * image->getImageCount()));
src/lbfgs.cu:  checkCudaErrors(cudaMalloc(
src/lbfgs.cu:  checkCudaErrors(
src/lbfgs.cu:      cudaMemset(d_s, 0, sizeof(float) * M * N * K * image->getImageCount()));
src/lbfgs.cu:  checkCudaErrors(cudaMalloc((void**)&p_old,
src/lbfgs.cu:  checkCudaErrors(
src/lbfgs.cu:      cudaMemset(p_old, 0, sizeof(float) * M * N * image->getImageCount()));
src/lbfgs.cu:  checkCudaErrors(
src/lbfgs.cu:      cudaMalloc((void**)&xi, sizeof(float) * M * N * image->getImageCount()));
src/lbfgs.cu:  checkCudaErrors(
src/lbfgs.cu:      cudaMemset(xi, 0, sizeof(float) * M * N * image->getImageCount()));
src/lbfgs.cu:  checkCudaErrors(cudaMalloc((void**)&xi_old,
src/lbfgs.cu:  checkCudaErrors(
src/lbfgs.cu:      cudaMemset(xi_old, 0, sizeof(float) * M * N * image->getImageCount()));
src/lbfgs.cu:  checkCudaErrors(cudaMalloc((void**)&norm_vector,
src/lbfgs.cu:  checkCudaErrors(cudaMemset(norm_vector, 0,
src/lbfgs.cu:__host__ void LBFGS::deallocateMemoryGpu(){FREEALL};
src/lbfgs.cu:  allocateMemoryGpu();
src/lbfgs.cu:  // checkCudaErrors(cudaMemcpy(p_old, image->getImage(),
src/lbfgs.cu:  // sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));
src/lbfgs.cu:  // checkCudaErrors(cudaMemcpy(xi_old, xi,
src/lbfgs.cu:  // sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));
src/lbfgs.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/lbfgs.cu:    checkCudaErrors(cudaMemcpy(p_old, image->getImage(),
src/lbfgs.cu:                               cudaMemcpyDeviceToDevice));
src/lbfgs.cu:    checkCudaErrors(cudaMemcpy(xi_old, xi,
src/lbfgs.cu:                               cudaMemcpyDeviceToDevice));
src/lbfgs.cu:      deallocateMemoryGpu();
src/lbfgs.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/lbfgs.cu:      deallocateMemoryGpu();
src/lbfgs.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/lbfgs.cu:  deallocateMemoryGpu();
src/lbfgs.cu:  checkCudaErrors(cudaMalloc((void**)&aux_vector, sizeof(float) * M * N));
src/lbfgs.cu:  checkCudaErrors(
src/lbfgs.cu:      cudaMalloc((void**)&d_q, sizeof(float) * M * N * image->getImageCount()));
src/lbfgs.cu:  checkCudaErrors(
src/lbfgs.cu:      cudaMalloc((void**)&d_r, sizeof(float) * M * N * image->getImageCount()));
src/lbfgs.cu:  checkCudaErrors(cudaMemset(aux_vector, 0, sizeof(float) * M * N));
src/lbfgs.cu:  checkCudaErrors(
src/lbfgs.cu:      cudaMemset(d_r, 0, sizeof(float) * M * N * image->getImageCount()));
src/lbfgs.cu:  checkCudaErrors(cudaMemcpy(d_q, xi,
src/lbfgs.cu:                             cudaMemcpyDeviceToDevice));
src/lbfgs.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/lbfgs.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/lbfgs.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/lbfgs.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/lbfgs.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/lbfgs.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/lbfgs.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/lbfgs.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/lbfgs.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/lbfgs.cu:    checkCudaErrors(cudaDeviceSynchronize());
src/lbfgs.cu:  checkCudaErrors(cudaMemcpy(xi, d_r,
src/lbfgs.cu:                             cudaMemcpyDeviceToDevice));
src/lbfgs.cu:  cudaFree(aux_vector);
src/lbfgs.cu:  cudaFree(d_q);
src/lbfgs.cu:  cudaFree(d_r);
src/f1dim.cu:   Additionally, this program uses some NVIDIA routines whose copyright is held
src/f1dim.cu:   by NVIDIA end user license agreement (EULA).
src/f1dim.cu:  checkCudaErrors(
src/f1dim.cu:      cudaMalloc((void**)&device_xt, sizeof(float) * M * N * image_count));
src/f1dim.cu:  checkCudaErrors(
src/f1dim.cu:      cudaMemset(device_xt, 0, sizeof(float) * M * N * image_count));
src/f1dim.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/f1dim.cu:      checkCudaErrors(cudaDeviceSynchronize());
src/f1dim.cu:       checkCudaErrors(cudaDeviceSynchronize());*/
src/f1dim.cu:  cudaFree(device_xt);
src/totalsquaredvariation.cu:  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
src/totalsquaredvariation.cu:  cudaFree(this->device_S);
src/totalsquaredvariation.cu:  cudaFree(this->device_DS);
src/laplacian.cu:  checkCudaErrors(cudaMemset(device_DS, 0, sizeof(float) * M * N));
src/laplacian.cu:  cudaFree(this->device_S);
src/laplacian.cu:  cudaFree(this->device_DS);
src/pillBox2D.cu:  this->copyKerneltoGPU();
src/pillBox2D.cu:  this->copyKerneltoGPU();
src/pillBox2D.cu:  this->copyKerneltoGPU();
src/pillBox2D.cu:  this->copyKerneltoGPU();

```
