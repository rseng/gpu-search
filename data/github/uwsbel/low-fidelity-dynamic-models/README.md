# https://github.com/uwsbel/low-fidelity-dynamic-models

```console
README.md:Using CMake, the user can choose to build the models to execute on the CPU or NVIDIA GPU cards. The CPU models are implemented in C++, whereas the GPU models utilize CUDA. A Python API is also available, provided through SWIG-wrapped C++ models.
README.md:2. **GPU Optimization for Scalability**: The GPU models are adept at parallel simulations of multiple vehicles. The 18 DOF GPU model, for example, can simulate 300,000 vehicles in real-time on an NVIDIA A100 GPU. Note: The GPU models are only available for Nvidia GPUs.
README.md:  - **11dof-gpu**: GPU version for the 11 DOF wheeled vehicle model. See [README](./wheeled_vehicle_models/11dof-gpu/README.md) for use instructions.
README.md:  - **18dof-gpu**: GPU version for the 18 DOF wheeled vehicle model. See [README](./wheeled_vehicle_models/18dof-gpu/README.md) for use instructions.
README.md:  - **24dof-gpu**: GPU version for the 24 DOF wheeled vehicle model. See [README](./wheeled_vehicle_models/24dof-gpu/README.md) for use instructions.
paper/paper_results/README.md:This directory contains scripts and tools necessary to reproduce the results presented in our paper. By following the instructions below, you will be able to run benchmarks on both CPU and GPU, and generate plots illustrating the performance characteristics discussed in the paper.
paper/paper_results/README.md:- GPU drivers and CUDA Toolkit (if you wish to run GPU benchmarks)
paper/paper_results/README.md:    The script will first ask if you want to build and run GPU benchmarks. Type y for yes or n for no, then press Enter. Based on your input, the script will either build with or without GPU benchmark support and then proceed to run the CPU and GPU benchmarks.
paper/paper_results/README.md:    Note: Running GPU benchmarks requires a CUDA-compatible GPU and appropriate drivers and toolkit installed.
paper/paper_results/CMakeLists.txt:# GPU tests dependencies
paper/paper_results/CMakeLists.txt:option(BUILD_GPU_TESTS "Enable building of GPU benchmarks" OFF)
paper/paper_results/CMakeLists.txt:if(BUILD_GPU_TESTS)
paper/paper_results/CMakeLists.txt:  enable_language(CUDA)
paper/paper_results/CMakeLists.txt:  find_package(CUDA 8.0 REQUIRED)
paper/paper_results/CMakeLists.txt:  message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
paper/paper_results/CMakeLists.txt:  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
paper/paper_results/CMakeLists.txt:  set(GPU_MODEL_FILES
paper/paper_results/CMakeLists.txt:  ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh
paper/paper_results/CMakeLists.txt:  ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu
paper/paper_results/CMakeLists.txt:  ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh
paper/paper_results/CMakeLists.txt:  ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu
paper/paper_results/CMakeLists.txt:  ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh
paper/paper_results/CMakeLists.txt:  ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu
paper/paper_results/CMakeLists.txt:  set(GPU_UTILS_FILES
paper/paper_results/CMakeLists.txt:      ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/utils/utils_gpu.cuh
paper/paper_results/CMakeLists.txt:      ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/utils/utils_gpu.cu
paper/paper_results/CMakeLists.txt:  set(GPU_SOLVER_FILES
paper/paper_results/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh
paper/paper_results/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu
paper/paper_results/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh
paper/paper_results/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu
paper/paper_results/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh
paper/paper_results/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu
paper/paper_results/CMakeLists.txt:  set(GPU_TEST_PROGRAMS
paper/paper_results/CMakeLists.txt:    18dof_gpu_bench
paper/paper_results/CMakeLists.txt:    11dof_gpu_bench
paper/paper_results/CMakeLists.txt:    24dof_gpu_bench
paper/paper_results/CMakeLists.txt:  include_directories(${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/18dof-gpu)
paper/paper_results/CMakeLists.txt:  include_directories(${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/11dof-gpu)
paper/paper_results/CMakeLists.txt:  include_directories(${CMAKE_SOURCE_DIR}/../../wheeled_vehicle_models/24dof-gpu)
paper/paper_results/CMakeLists.txt:  # Build GPU Tests
paper/paper_results/CMakeLists.txt:  foreach(PROGRAM ${GPU_TEST_PROGRAMS})
paper/paper_results/CMakeLists.txt:    add_executable(${PROGRAM} ${CMAKE_SOURCE_DIR}/${PROGRAM}.cu ${GPU_UTILS_FILES} ${GPU_MODEL_FILES} ${GPU_SOLVER_FILES})
paper/paper_results/CMakeLists.txt:    # Enable separate compilation for CUDA
paper/paper_results/CMakeLists.txt:    set_property(TARGET ${PROGRAM} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
paper/paper_results/CMakeLists.txt:    # Set architecure based on this link - https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
paper/paper_results/CMakeLists.txt:    set_target_properties(${PROGRAM} PROPERTIES CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90")
paper/paper_results/11dof_gpu_bench.cu:#include <cuda.h>
paper/paper_results/11dof_gpu_bench.cu:#include <cuda_runtime.h>
paper/paper_results/11dof_gpu_bench.cu:#include "dof11_halfImplicit_gpu.cuh"
paper/paper_results/11dof_gpu_bench.cu:using namespace d11GPU;
paper/paper_results/11dof_gpu_bench.cu:    d11SolverHalfImplicitGPU solver(num_vehicles);
paper/paper_results/11dof_gpu_bench.cu:    cudaEvent_t start, stop;
paper/paper_results/11dof_gpu_bench.cu:    cudaEventCreate(&start);
paper/paper_results/11dof_gpu_bench.cu:    cudaEventCreate(&stop);
paper/paper_results/11dof_gpu_bench.cu:    cudaEventRecord(start);
paper/paper_results/11dof_gpu_bench.cu:    cudaEventRecord(stop);
paper/paper_results/11dof_gpu_bench.cu:    cudaEventSynchronize(stop);
paper/paper_results/11dof_gpu_bench.cu:    cudaEventElapsedTime(&milliseconds, start, stop);
paper/paper_results/plotting/gpuScalePlot.py:    print("usage: python3 gpuScalePlot <save>")
paper/paper_results/plotting/gpuScalePlot.py:    fig.savefig("./images/gpuScale.png", format='png', facecolor='w', dpi=600)
paper/paper_results/bench.sh:# Ask the user if they want to build GPU tests
paper/paper_results/bench.sh:read -p "Do you want to build GPU Benchmarks? (y/n): " answer
paper/paper_results/bench.sh:    echo "Building with GPU Benchmarks..."
paper/paper_results/bench.sh:    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU_TESTS=ON ..
paper/paper_results/bench.sh:    # Run the GPU benchmarks
paper/paper_results/bench.sh:    echo "Running GPU Benchmarks..."
paper/paper_results/bench.sh:    ./11dof_gpu_bench 1024 16 > ../data/LFDM/11dof.out
paper/paper_results/bench.sh:    ./11dof_gpu_bench 2048 16 >> ../data/LFDM/11dof.out
paper/paper_results/bench.sh:    ./11dof_gpu_bench 4096 16 >> ../data/LFDM/11dof.out
paper/paper_results/bench.sh:    ./11dof_gpu_bench 8192 16 >> ../data/LFDM/11dof.out
paper/paper_results/bench.sh:    ./11dof_gpu_bench 16384 16 >> ../data/LFDM/11dof.out
paper/paper_results/bench.sh:    ./11dof_gpu_bench 32768 16 >> ../data/LFDM/11dof.out
paper/paper_results/bench.sh:    ./11dof_gpu_bench 65536 16 >> ../data/LFDM/11dof.out
paper/paper_results/bench.sh:    ./11dof_gpu_bench 131072 16 >> ../data/LFDM/11dof.out
paper/paper_results/bench.sh:    ./11dof_gpu_bench 262144 16 >> ../data/LFDM/11dof.out
paper/paper_results/bench.sh:    ./11dof_gpu_bench 393216 16 >> ../data/LFDM/11dof.out
paper/paper_results/bench.sh:    ./11dof_gpu_bench 524288 16 >> ../data/LFDM/11dof.out
paper/paper_results/bench.sh:    ./18dof_gpu_bench 1024 16 > ../data/LFDM/18dof.out
paper/paper_results/bench.sh:    ./18dof_gpu_bench 2048 16 >> ../data/LFDM/18dof.out
paper/paper_results/bench.sh:    ./18dof_gpu_bench 4096 16 >> ../data/LFDM/18dof.out
paper/paper_results/bench.sh:    ./18dof_gpu_bench 8192 16 >> ../data/LFDM/18dof.out
paper/paper_results/bench.sh:    ./18dof_gpu_bench 16384 16 >> ../data/LFDM/18dof.out
paper/paper_results/bench.sh:    ./18dof_gpu_bench 32768 16 >> ../data/LFDM/18dof.out
paper/paper_results/bench.sh:    ./18dof_gpu_bench 65536 16 >> ../data/LFDM/18dof.out
paper/paper_results/bench.sh:    ./18dof_gpu_bench 131072 16 >> ../data/LFDM/18dof.out
paper/paper_results/bench.sh:    ./18dof_gpu_bench 262144 16 >> ../data/LFDM/18dof.out
paper/paper_results/bench.sh:    ./18dof_gpu_bench 393216 16 >> ../data/LFDM/18dof.out
paper/paper_results/bench.sh:    ./18dof_gpu_bench 524288 16 >> ../data/LFDM/18dof.out
paper/paper_results/bench.sh:    ./24dof_gpu_bench 1024 16 > ../data/LFDM/24dof.out
paper/paper_results/bench.sh:    ./24dof_gpu_bench 2048 16 >> ../data/LFDM/24dof.out
paper/paper_results/bench.sh:    ./24dof_gpu_bench 4096 16 >> ../data/LFDM/24dof.out
paper/paper_results/bench.sh:    ./24dof_gpu_bench 8192 16 >> ../data/LFDM/24dof.out
paper/paper_results/bench.sh:    ./24dof_gpu_bench 16384 16 >> ../data/LFDM/24dof.out
paper/paper_results/bench.sh:    ./24dof_gpu_bench 32768 16 >> ../data/LFDM/24dof.out
paper/paper_results/bench.sh:    ./24dof_gpu_bench 65536 16 >> ../data/LFDM/24dof.out
paper/paper_results/bench.sh:    ./24dof_gpu_bench 131072 16 >> ../data/LFDM/24dof.out
paper/paper_results/bench.sh:    ./24dof_gpu_bench 262144 16 >> ../data/LFDM/24dof.out
paper/paper_results/bench.sh:    ./24dof_gpu_bench 393216 16 >> ../data/LFDM/24dof.out
paper/paper_results/bench.sh:    ./24dof_gpu_bench 524288 16 >> ../data/LFDM/24dof.out
paper/paper_results/bench.sh:    echo "Building without GPU Benchmarks..."
paper/paper_results/bench.sh:    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU_TESTS=OFF ..
paper/paper_results/18dof_gpu_bench.cu:#include <cuda.h>
paper/paper_results/18dof_gpu_bench.cu:#include <cuda_runtime.h>
paper/paper_results/18dof_gpu_bench.cu:#include "dof18_halfImplicit_gpu.cuh"
paper/paper_results/18dof_gpu_bench.cu:using namespace d18GPU;
paper/paper_results/18dof_gpu_bench.cu:    d18SolverHalfImplicitGPU solver(num_vehicles);
paper/paper_results/18dof_gpu_bench.cu:    cudaEvent_t start, stop;
paper/paper_results/18dof_gpu_bench.cu:    cudaEventCreate(&start);
paper/paper_results/18dof_gpu_bench.cu:    cudaEventCreate(&stop);
paper/paper_results/18dof_gpu_bench.cu:    cudaEventRecord(start);
paper/paper_results/18dof_gpu_bench.cu:    cudaEventRecord(stop);
paper/paper_results/18dof_gpu_bench.cu:    cudaEventSynchronize(stop);
paper/paper_results/18dof_gpu_bench.cu:    cudaEventElapsedTime(&milliseconds, start, stop);
paper/paper_results/24dof_gpu_bench.cu:#include <cuda.h>
paper/paper_results/24dof_gpu_bench.cu:#include <cuda_runtime.h>
paper/paper_results/24dof_gpu_bench.cu:#include "dof24_halfImplicit_gpu.cuh"
paper/paper_results/24dof_gpu_bench.cu:using namespace d24GPU;
paper/paper_results/24dof_gpu_bench.cu:    d24SolverHalfImplicitGPU solver(num_vehicles);
paper/paper_results/24dof_gpu_bench.cu:    cudaEvent_t start, stop;
paper/paper_results/24dof_gpu_bench.cu:    cudaEventCreate(&start);
paper/paper_results/24dof_gpu_bench.cu:    cudaEventCreate(&stop);
paper/paper_results/24dof_gpu_bench.cu:    cudaEventRecord(start);
paper/paper_results/24dof_gpu_bench.cu:    cudaEventRecord(stop);
paper/paper_results/24dof_gpu_bench.cu:    cudaEventSynchronize(stop);
paper/paper_results/24dof_gpu_bench.cu:    cudaEventElapsedTime(&milliseconds, start, stop);
paper/paper.md:  - CUDA
paper/paper.md:Lower Fidelity Dynamic Models (LFDM) is a library of dynamics models for on-road wheeled vehicles that is written in C++ and CUDA and wrapped to Python using SWIG. Each model is described as a set of Ordinary Differential Equations (ODEs) that take a driver input - a normalized throttle between 0 and 1, a normalized steering between -1 and 1 (with -1 representing max steering toward a left turn), and a normalized braking input between 0 and 1, and subsequently advance the state of the vehicle (its position and velocity) forward in time.
paper/paper.md:The LFDM library contains three dynamic vehicle models, differentiated by their Degrees of Freedom (DoF) counts: 11 DoF, 18 DoF, and 24 DoF. Each can be run on the CPU or an NVIDIA GPU.
paper/paper.md:In this paper, we present a Statement of need, describing in what applications these models are most useful. In the LFDM Accuracy section, we present a comparison of the LFDM library's accuracy with the High-Fidelity Vehicle Model, Chrono::Vehicle [@Serban:2019]. We then demonstrate in the LFDM Speed and Scaling section that the LFDMs, while closely matching the accuracy of Chrono::Vehicle, operate approximately 3000 times faster. Additionally, by utilizing the GPU version of the models, it is possible to simulate about 300,000 vehicles in real-time, i.e., simulating one second of dynamics for 300,000 vehicles takes only one real-world second. Further details on the model formulation are available in Chapter 2 of @huzaifa:2023. Therein, the users can find the actual equations of motion, for each of the three models.
paper/paper.md:3. To the best of our knowledge, there is currently no open-source software capable of executing large-scale, parallel simulations of on-road vehicle dynamics on GPUs. LFDM bridges this gap, facilitating the real-time simulation of nearly 300,000 vehicles. This capability significantly enhances the potential for large-scale reinforcement learning and comprehensive traffic simulations.
paper/paper.md:Further, the GPU version of the LFDMs enables large-scale parallel simulation, which comes into play in  Reinforcement Learning and traffic simulation. As shown in \autoref{fig:gpu_scale}, around 330,000 11DoF vehicle models can be simulated on an NVIDIA A100 GPU with an RTF of 1.
paper/paper.md:![Scaling analysis of the GPU versions of the LFDMs shows that about 330,000 11 DoF vehicles can be simulated in Real-Time.\label{fig:gpu_scale}](images/gpuScale.png){ width=60% }
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:#include "dof11_gpu.cuh"
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:using namespace d11GPU;
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ double d11GPU::driveTorque(const VehicleParam* v_params, const double throttle, const double motor_speed) {
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::vehToTireTransform(TMeasyState* tiref_st,
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::vehToTireTransform(TMeasyNrState* tiref_st,
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::tireToVehTransform(TMeasyState* tiref_st,
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::tireToVehTransform(TMeasyNrState* tiref_st,
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ __host__ void d11GPU::tireInit(TMeasyParam* t_params) {
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::tireInit(TMeasyNrParam* t_params) {
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:d11GPU::tmxy_combined(double* f, double* fos, double s, double df0, double sm, double fm, double ss, double fs) {
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::computeCombinedCoulombForce(double* fx,
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::computeTireLoads(double* loads,
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::computeTireLoads(double* loads,
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::differentialSplit(double torque,
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::computeTireRHS(TMeasyState* t_states,
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::computeTireRHS(TMeasyNrState* t_states,
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::computePowertrainRHS(VehicleState* v_states,
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::computePowertrainRHS(VehicleState* v_states,
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__device__ void d11GPU::computeVehRHS(VehicleState* v_states,
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__host__ void d11GPU::setVehParamsJSON(VehicleParam& v_params, const char* fileName) {
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._steerMap, sizeof(MapEntry) * steerMapSize));
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:    CHECK_CUDA_ERROR(
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:        cudaMallocManaged((void**)&v_params._gearRatios, sizeof(double) * noGears));  // assign the memory for the gears
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._shiftMap, sizeof(MapEntry) * noGears));
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._powertrainMap, sizeof(MapEntry) * powertrainMapSize));
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._lossesMap, sizeof(MapEntry) * lossesMapSize));
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._CFmap, sizeof(MapEntry) * CFmapSize));
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._TRmap, sizeof(MapEntry) * TRmapSize));
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__host__ void d11GPU::setTireParamsJSON(TMeasyParam& t_params, const char* fileName) {
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__host__ void d11GPU::setTireParamsJSON(TMeasyNrParam& t_params, const char* fileName) {
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__host__ double d11GPU::GetTireMaxLoad(unsigned int li) {
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__host__ void d11GPU::GuessTruck80Par(unsigned int li,          // tire load index
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__host__ void d11GPU::GuessTruck80Par(double tireLoad,   // tire load index
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__host__ void d11GPU::GuessPassCar70Par(unsigned int li,          // tire load index
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cu:__host__ void d11GPU::GuessPassCar70Par(double tireLoad,            // tire load index
wheeled_vehicle_models/11dof-gpu/README.md:The 11 DOF GPU model is theoretically exactly the same as the CPU model, in that it solves the same equations of motions. However, the GPU version of the model has solvers that support solving multiple sets of these equations of motions on different GPU threads parallely. The GPU models however have a more limited functionallity when compared to the CPU models. For instance:
wheeled_vehicle_models/11dof-gpu/README.md:- The GPU models only support the Half-Implicit integrator
wheeled_vehicle_models/11dof-gpu/README.md:- Because Sundials is not supported, the GPU models do not support Forward Sensitivity Analysis (FSA)
wheeled_vehicle_models/11dof-gpu/README.md:- The GPU models do not provide system RHS Jacobians
wheeled_vehicle_models/11dof-gpu/README.md:- The GPU models are not wrapped to Python
wheeled_vehicle_models/11dof-gpu/README.md:However, the GPU model can simultaneously simulate upto 300,000 vehicles in real-time at a time step of $1e-3$ when benchmarked on an Nvidia A100 GPU. See chapter 6 [here](https://uwmadison.box.com/s/2tsvr4adbrzklle30z0twpu2nlzvlayc) for more details.
wheeled_vehicle_models/11dof-gpu/README.md:The model parameters and inputs are provided in the same way as the CPU model. The repository also contains [demos](./demos/) that describe how the GPU models can be used.
wheeled_vehicle_models/11dof-gpu/README.md:See [here](../README.md#how-do-i-use-the-models) for a general description of how to run the demos. For the 11 DOF model, the demos are placed in the [demos](./demos) folder. The demos are built in the [build](../README.md#generate) folder. The demos require command line arguments specifying the input controls for the vehicles. Additionally, some of the GPU demos require information about the number of vehicles desired to be simulated and the number of threads per block to be launched. We recommend using `32` for the number of threads per block. For a description of how to create these control input files, or how to use the default control input files provided, see [here](../11dof/README.md#how-do-i-provide-driver-inputs).
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:project(d11dof_gpu LANGUAGES CXX CUDA)
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:enable_language(CUDA)
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:find_package(CUDA 8.0 REQUIRED)
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:    set(CUDA_NVCC_FLAGS_DEBUG "-G -g" CACHE STRING "NVCC debug flags" FORCE)
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/dof11_gpu.cuh
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/dof11_gpu.cu
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/utils_gpu.cuh
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/utils_gpu.cu
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:# Only half implicit solver available for GPU
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:  ${CMAKE_CURRENT_SOURCE_DIR}/dof11_halfImplicit_gpu.cuh
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:  ${CMAKE_CURRENT_SOURCE_DIR}/dof11_halfImplicit_gpu.cu
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:    demo_hmmwv_11gpu
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:    demo_hmmwv_varControls_11gpu
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:    demo_hmmwv_step_11gpu
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:    demo_hmmwv_controlsFunctor_11gpu
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:    # Enable separate compilation for CUDA
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:    set_property(TARGET ${PROGRAM} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:    # Set architecure based on this link - https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
wheeled_vehicle_models/11dof-gpu/CMakeLists.txt:    set_target_properties(${PROGRAM} PROPERTIES CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90")
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:#ifndef DOF11_HALFIMPLICIT_GPU_CUH
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:#define DOF11_HALFIMPLICIT_GPU_CUH
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:#include "dof11_gpu.cuh"
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:class d11SolverHalfImplicitGPU {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:    d11SolverHalfImplicitGPU(unsigned int total_num_vehicles);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:    ~d11SolverHalfImplicitGPU();
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:    /// @brief Set the time for which the GPU kernel simulates without a sync between the vehicles
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:    /// The simulation proceeds by multiple launches of the GPU kernel for m_kernel_sim_time duration. This is mainly
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:    /// @brief Sets the threads per block for the GPU kernel. Defaults to 32
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:    __host__ void Initialize(d11GPU::VehicleState& vehicle_states,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                             d11GPU::TMeasyState& tire_states_F,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                             d11GPU::TMeasyState& tire_states_R,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:    __host__ void Initialize(d11GPU::VehicleState& vehicle_states,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                             d11GPU::TMeasyNrState& tire_states_F,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                             d11GPU::TMeasyNrState& tire_states_R,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:    __host__ d11GPU::SimState GetSimState(unsigned int vehicle_index);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:    d11GPU::SimData* m_sim_data;          ///< Simulation data for all the vehicles. See d11GPU::SimData for more info
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:    d11GPU::SimDataNr* m_sim_data_nr;     ///< Simulation data but with the TMeasyNr tire for all the vehicles. See
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                                          // d11GPU::SimDataNr for more info
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:    d11GPU::SimState* m_sim_states;       ///< Simulation states for all the vehicles. See d11GPU::SimState for more
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:    d11GPU::SimStateNr* m_sim_states_nr;  ///< Simulation states but with the TMeasyNr tire for all the vehicles. See
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                                          // d11GPU::SimStateNr for more info
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:            CHECK_CUDA_ERROR(cudaMalloc((void**)&m_device_response, m_device_size));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:            cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:            if (err != cudaSuccess) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:            cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:            if (err != cudaSuccess) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                          d11GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                          d11GPU::SimStateNr* sim_states_nr);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                          d11GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                          d11GPU::SimStateNr* sim_states_nr);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                          d11GPU::SimData* sim_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                          d11GPU::SimState* sim_states);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                          d11GPU::SimData* sim_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                          d11GPU::SimState* sim_states);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                       d11GPU::SimData* sim_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                       d11GPU::SimState* sim_states);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                       d11GPU::SimData* sim_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                       d11GPU::SimState* sim_states,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                       d11GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                       d11GPU::SimStateNr* sim_states_nr);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                       d11GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                       d11GPU::SimStateNr* sim_states_nr,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                          d11GPU::SimData* sim_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                          d11GPU::SimState* sim_states) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:            d11GPU::VehicleState& v_states = sim_states[vehicle_id]._veh_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:            d11GPU::VehicleParam& veh_param = sim_data[vehicle_id]._veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:            d11GPU::TMeasyState& tiref_st = sim_states[vehicle_id]._tiref_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:            d11GPU::TMeasyState& tirer_st = sim_states[vehicle_id]._tirer_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                          d11GPU::SimDataNr* sim_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                          d11GPU::SimStateNr* sim_states) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:            d11GPU::VehicleState& v_states = sim_states[vehicle_id]._veh_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:            d11GPU::VehicleParam& veh_param = sim_data[vehicle_id]._veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:            d11GPU::TMeasyNrState& tiref_st = sim_states[vehicle_id]._tiref_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:            d11GPU::TMeasyNrState& tirer_st = sim_states[vehicle_id]._tirer_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                       d11GPU::SimData* sim_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                       d11GPU::SimState* sim_states,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:        d11GPU::VehicleParam& veh_param = sim_data[vehicle_index]._veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:        d11GPU::VehicleState& veh_state = sim_states[vehicle_index]._veh_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:        d11GPU::TMeasyParam& tireTM_param = sim_data[vehicle_index]._tireTM_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:        d11GPU::TMeasyState& tireTMf_state = sim_states[vehicle_index]._tiref_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:        d11GPU::TMeasyState& tireTMr_state = sim_states[vehicle_index]._tirer_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                       d11GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:                       d11GPU::SimStateNr* sim_states_nr,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:        d11GPU::VehicleParam& veh_param = sim_data_nr[vehicle_index]._veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:        d11GPU::VehicleState& veh_state = sim_states_nr[vehicle_index]._veh_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:        d11GPU::TMeasyNrParam& tireTMNr_param = sim_data_nr[vehicle_index]._tireTMNr_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:        d11GPU::TMeasyNrState& tireTMNrf_state = sim_states_nr[vehicle_index]._tiref_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cuh:        d11GPU::TMeasyNrState& tireTMNrr_state = sim_states_nr[vehicle_index]._tirer_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:#include "dof11_gpu.cuh"
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:#include "dof11_halfImplicit_gpu.cuh"
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:using namespace d11GPU;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:__host__ d11SolverHalfImplicitGPU::d11SolverHalfImplicitGPU(unsigned int total_num_vehicles)
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data, sizeof(d11GPU::SimData) * m_total_num_vehicles));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data_nr, sizeof(d11GPU::SimDataNr) * m_total_num_vehicles));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_states, sizeof(d11GPU::SimState) * m_total_num_vehicles));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_states_nr, sizeof(d11GPU::SimStateNr) * m_total_num_vehicles));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    int deviceId = 0;         // Assume we are using GPU 0
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    cudaSetDevice(deviceId);  // Set the device
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:__host__ d11SolverHalfImplicitGPU::~d11SolverHalfImplicitGPU() {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaFree(m_sim_data);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaFree(m_sim_states);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaFree(m_sim_data_nr);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaFree(m_sim_states_nr);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    cudaFree(m_device_response);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:__host__ void d11SolverHalfImplicitGPU::Construct(const std::string& vehicle_params_file,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    cudaFree(m_sim_data_nr);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    cudaFree(m_sim_states_nr);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them up
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    d11GPU::VehicleParam veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    d11GPU::TMeasyParam tire_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data[i]._driver_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data, sizeof(m_sim_data[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                                          0));  // move the simData onto the GPU
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:__host__ void d11SolverHalfImplicitGPU::Construct(const std::string& vehicle_params_file,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaFree(m_sim_data_nr);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaFree(m_sim_states_nr);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        d11GPU::VehicleParam veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        d11GPU::TMeasyParam tire_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data[i]._driver_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data, sizeof(m_sim_data[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                                              0));  // move the simData onto the GPU
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaFree(m_sim_data);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaFree(m_sim_states);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        d11GPU::VehicleParam veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        d11GPU::TMeasyNrParam tire_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data_nr[i]._driver_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data_nr, sizeof(m_sim_data_nr[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                                              0));  // move the simData onto the GPU
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:__host__ void d11SolverHalfImplicitGPU::Construct(const std::string& vehicle_params_file,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    cudaFree(m_sim_data_nr);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    cudaFree(m_sim_states_nr);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them up
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    d11GPU::VehicleParam veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    d11GPU::TMeasyParam tire_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data, sizeof(m_sim_data[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                                          0));  // move the simData onto the GPU
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:__host__ void d11SolverHalfImplicitGPU::Construct(const std::string& vehicle_params_file,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaFree(m_sim_data_nr);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaFree(m_sim_states_nr);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        d11GPU::VehicleParam veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        d11GPU::TMeasyParam tire_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data, sizeof(m_sim_data[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                                              0));  // move the simData onto the GPU
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaFree(m_sim_data);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaFree(m_sim_states);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        d11GPU::VehicleParam veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        d11GPU::TMeasyNrParam tire_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data_nr, sizeof(m_sim_data_nr[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                                              0));  // move the simData onto the GPU
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:void d11SolverHalfImplicitGPU::Initialize(d11GPU::VehicleState& vehicle_states,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                                          d11GPU::TMeasyState& tire_states_F,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                                          d11GPU::TMeasyState& tire_states_R,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_states, sizeof(SimState) * m_vehicle_count_tracker_states,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                                          0));  // move the simState onto the GPU
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:void d11SolverHalfImplicitGPU::Initialize(d11GPU::VehicleState& vehicle_states,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                                          d11GPU::TMeasyNrState& tire_states_F,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                                          d11GPU::TMeasyNrState& tire_states_R,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_states_nr, sizeof(SimState) * m_vehicle_count_tracker_states,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                                          0));  // move the simState onto the GPU
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:__host__ void d11SolverHalfImplicitGPU::SetOutput(const std::string& output_file,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMalloc((void**)&m_device_response, m_device_size));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:__host__ void d11SolverHalfImplicitGPU::Solve() {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMalloc((void**)&m_device_response, m_device_size));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            if (err != cudaSuccess) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            if (err != cudaSuccess) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            CHECK_CUDA_ERROR(cudaMemcpy(m_host_response + filled_response, m_device_response, m_device_size,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                                        cudaMemcpyDeviceToHost));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:__host__ double d11SolverHalfImplicitGPU::SolveStep(double t,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMalloc((void**)&m_device_response, m_device_size));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        if (err != cudaSuccess) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        if (err != cudaSuccess) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:__host__ void d11SolverHalfImplicitGPU::Write(double t, unsigned int time_steps_to_write) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:__host__ void d11SolverHalfImplicitGPU::WriteToFile() {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:__host__ SimState d11SolverHalfImplicitGPU::GetSimState(unsigned int vehicle_index) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        // Copy the specific SimState from the GPU to the host
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaMemcpy(&host_state, &m_sim_states[vehicle_index], sizeof(SimState), cudaMemcpyDeviceToHost);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:        cudaMemcpy(&host_state, &m_sim_states_nr[vehicle_index], sizeof(SimState), cudaMemcpyDeviceToHost);
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                       d11GPU::SimData* sim_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                       d11GPU::SimState* sim_states,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                       d11GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                       d11GPU::SimStateNr* sim_states_nr) {  // Get the vehicle index
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                       d11GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                       d11GPU::SimStateNr* sim_states_nr,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                          d11GPU::SimData* sim_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                          d11GPU::SimState* sim_states) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::VehicleState& v_states = sim_states[vehicle_id]._veh_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::VehicleParam& veh_param = sim_data[vehicle_id]._veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::TMeasyState& tiref_st = sim_states[vehicle_id]._tiref_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::TMeasyState& tirer_st = sim_states[vehicle_id]._tirer_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                          d11GPU::SimData* sim_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                          d11GPU::SimState* sim_states) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::VehicleState& v_states = sim_states[vehicle_id]._veh_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::VehicleParam& veh_param = sim_data[vehicle_id]._veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::TMeasyState& tiref_st = sim_states[vehicle_id]._tiref_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::TMeasyState& tirer_st = sim_states[vehicle_id]._tirer_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                          d11GPU::SimDataNr* sim_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                          d11GPU::SimStateNr* sim_states) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::VehicleState& v_states = sim_states[vehicle_id]._veh_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::VehicleParam& veh_param = sim_data[vehicle_id]._veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::TMeasyNrState& tiref_st = sim_states[vehicle_id]._tiref_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::TMeasyNrState& tirer_st = sim_states[vehicle_id]._tirer_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                          d11GPU::SimDataNr* sim_data,
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:                          d11GPU::SimStateNr* sim_states) {
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::VehicleState& v_states = sim_states[vehicle_id]._veh_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::VehicleParam& veh_param = sim_data[vehicle_id]._veh_param;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::TMeasyNrState& tiref_st = sim_states[vehicle_id]._tiref_state;
wheeled_vehicle_models/11dof-gpu/dof11_halfImplicit_gpu.cu:            d11GPU::TMeasyNrState& tirer_st = sim_states[vehicle_id]._tirer_state;
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:#ifndef DOF11_GPU_CUH
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:#define DOF11_GPU_CUH
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:#include "utils_gpu.cuh"
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:namespace d11GPU {
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// http://www.tmeasy.de/. It is important to note that this implementation within the d11GPU namespace is exactly the
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// same as the implementation in the d11 namespace. The only difference is that the d11GPU namespace is meant to be
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// used on the GPU where standard library functions are not available.
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// http://www.tmeasy.de/. It is important to note that this implementation within the d11GPU namespace is exactly the
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// same as the implementation in the d11 namespace. The only difference is that the d11GPU namespace is meant to be
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// used on the GPU where standard library functions are not available.
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// important to note that this implementation within the d11GPU namespace is exactly the same as the implementation in
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// the d11 namespace. The only difference is that the d11GPU namespace is meant to be used on the GPU where standard
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// important to note that this implementation within the d11GPU namespace is exactly the same as the implementation in
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// the d11 namespace. The only difference is that the d11GPU namespace is meant to be used on the GPU where standard
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// note that this implementation within the d11GPU namespace is exactly the same as the implementation in the d11
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// namespace. The only difference is that the d11GPU namespace is meant to be used on the GPU where standard library
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:    int _steerMapSize;  //!< Size of the steering map - this variable is unique to the GPU implementation as we need to
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:///  d11GPU namespace is exactly the same as the implementation in the d11 namespace. The only difference is that the
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:///  d11GPU namespace is meant to be used on the GPU where standard library functions are not available.
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:// Data structures that help handling multiple vehicles as needed in GPU version
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// stores all the "data" required to simulate 1 vehicle on the GPU. This is something largely the user does not have to
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:        cudaFree(_driver_data);  // Assumes _driver_data was allocated with new[]
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:        cudaFree(_driver_data);  // Assumes _driver_data was allocated with new[]
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// stores all the "states" of 1 vehicle simulated on the GPU. The user can use this to get the states of the vehicle at
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:/// any point in time during the simulation through the solver (see d11SolverHalfImplicitGPU class)
wheeled_vehicle_models/11dof-gpu/dof11_gpu.cuh:}  // namespace d11GPU
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:// Since the Half Implicit solver is the only one supported for the GPU models, that is what is used here.
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:#include <cuda.h>
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:#include "dof11_halfImplicit_gpu.cuh"
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:using namespace d11GPU;
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:    std::string inputPath = "../../11dof-gpu/data/input/" + file_name + ".txt";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:    std::string vehParamsJSON = "../../11dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:    std::string tireParamsJSON = "../../11dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:    d11SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:    cudaEvent_t start, stop;
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:    cudaEventCreate(&start);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:    cudaEventCreate(&stop);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:    cudaEventRecord(start);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:    cudaEventRecord(stop);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:    cudaEventSynchronize(stop);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_step_11gpu.cu:    cudaEventElapsedTime(&milliseconds, start, stop);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_11gpu.cu:// same driver inputs on the GPU. Since the Half Implicit solver is the only one supported for the GPU models,
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_11gpu.cu:#include <cuda.h>
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_11gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_11gpu.cu:#include "dof11_halfImplicit_gpu.cuh"
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_11gpu.cu:using namespace d11GPU;
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_11gpu.cu:    std::string driver_file = "../../11dof-gpu/data/input/" + file_name + ".txt";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_11gpu.cu:    std::string outputBasePath = "../../11dof-gpu/data/output/";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_11gpu.cu:    std::string vehParamsJSON = "../../11dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_11gpu.cu:    std::string tireParamsJSON = "../../11dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_11gpu.cu:    d11SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:// for the GPU models, that is what is used here.
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:#include <cuda.h>
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:#include "dof11_halfImplicit_gpu.cuh"
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:using namespace d11GPU;
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:    std::string driver_file_1 = "../../11dof-gpu/data/input/" + file_name_1 + ".txt";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:    std::string driver_file_2 = "../../11dof-gpu/data/input/" + file_name_2 + ".txt";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:    std::string vehParamsJSON = "../../11dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:    std::string tireParamsJSON = "../../11dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:    d11SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:    cudaEvent_t start, stop;
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:    cudaEventCreate(&start);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:    cudaEventCreate(&stop);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:    cudaEventRecord(start);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:    cudaEventRecord(stop);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:    cudaEventSynchronize(stop);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_varControls_11gpu.cu:    cudaEventElapsedTime(&milliseconds, start, stop);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:// Since the Half Implicit solver is the only one supported for the GPU models, that is what is used here.
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:#include <cuda.h>
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:#include "dof11_halfImplicit_gpu.cuh"
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:using namespace d11GPU;
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:    std::string inputPath = "../../11dof-gpu/data/input/" + file_name + ".txt";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:    std::string vehParamsJSON = "../../11dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:    std::string tireParamsJSON = "../../11dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:    d11SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:    cudaEvent_t start, stop;
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:    cudaEventCreate(&start);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:    cudaEventCreate(&stop);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:    cudaEventRecord(start);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:    cudaEventRecord(stop);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:    cudaEventSynchronize(stop);
wheeled_vehicle_models/11dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_11gpu.cu:    cudaEventElapsedTime(&milliseconds, start, stop);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:#include "dof24_gpu.cuh"
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:#include "dof24_halfImplicit_gpu.cuh"
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:using namespace d24GPU;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:d24SolverHalfImplicitGPU::d24SolverHalfImplicitGPU(unsigned int total_num_vehicles)
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data, sizeof(d24GPU::SimData) * m_total_num_vehicles));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data_nr, sizeof(d24GPU::SimDataNr) * m_total_num_vehicles));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_states, sizeof(d24GPU::SimState) * m_total_num_vehicles));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_states_nr, sizeof(d24GPU::SimStateNr) * m_total_num_vehicles));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    int deviceId = 0;         // Assume we are using GPU 0
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    cudaSetDevice(deviceId);  // Set the device
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:__host__ d24SolverHalfImplicitGPU::~d24SolverHalfImplicitGPU() {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaFree(m_sim_data);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaFree(m_sim_states);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaFree(m_sim_data_nr);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaFree(m_sim_states_nr);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    cudaFree(m_device_response);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:__host__ void d24SolverHalfImplicitGPU::Construct(const std::string& vehicle_params_file,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    cudaFree(m_sim_data_nr);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    cudaFree(m_sim_states_nr);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them up
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    d24GPU::VehicleParam veh_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    d24GPU::TMeasyParam tire_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    d24GPU::SuspensionParam sus_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data[i]._driver_data,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data, sizeof(m_sim_data[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                          0));  // move the simData onto the GPU
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:__host__ void d24SolverHalfImplicitGPU::Construct(const std::string& vehicle_params_file,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaFree(m_sim_data_nr);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaFree(m_sim_states_nr);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        d24GPU::VehicleParam veh_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        d24GPU::TMeasyParam tire_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        d24GPU::SuspensionParam sus_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:            CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data[i]._driver_data,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data, sizeof(m_sim_data[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                              0));  // move the simData onto the GPU
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaFree(m_sim_data);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaFree(m_sim_states);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        d24GPU::VehicleParam veh_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        d24GPU::TMeasyNrParam tire_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        d24GPU::SuspensionParam sus_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:            CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data_nr[i]._driver_data,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data_nr, sizeof(m_sim_data_nr[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                              0));  // move the simData onto the GPU
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:__host__ void d24SolverHalfImplicitGPU::Construct(const std::string& vehicle_params_file,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    cudaFree(m_sim_data_nr);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    cudaFree(m_sim_states_nr);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them up
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    d24GPU::VehicleParam veh_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    d24GPU::TMeasyParam tire_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    d24GPU::SuspensionParam sus_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data, sizeof(m_sim_data[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                          0));  // move the simData onto the GPU
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:__host__ void d24SolverHalfImplicitGPU::Construct(const std::string& vehicle_params_file,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaFree(m_sim_data_nr);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaFree(m_sim_states_nr);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        d24GPU::VehicleParam veh_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        d24GPU::TMeasyParam tire_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        d24GPU::SuspensionParam sus_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data, sizeof(m_sim_data[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                              0));  // move the simData onto the GPU
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaFree(m_sim_data);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaFree(m_sim_states);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        d24GPU::VehicleParam veh_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        d24GPU::TMeasyNrParam tire_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        d24GPU::SuspensionParam sus_param;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data_nr, sizeof(m_sim_data_nr[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                              0));  // move the simData onto the GPU
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:__host__ void d24SolverHalfImplicitGPU::Initialize(d24GPU::VehicleState& vehicle_states,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::TMeasyState& tire_states_LF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::TMeasyState& tire_states_RF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::TMeasyState& tire_states_LR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::TMeasyState& tire_states_RR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::SuspensionState& sus_states_LF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::SuspensionState& sus_states_RF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::SuspensionState& sus_states_LR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::SuspensionState& sus_states_RR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_states, sizeof(SimState) * m_vehicle_count_tracker_states,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                          0));  // move the simState onto the GPU
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:__host__ void d24SolverHalfImplicitGPU::Initialize(d24GPU::VehicleState& vehicle_states,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::TMeasyNrState& tire_states_LF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::TMeasyNrState& tire_states_RF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::TMeasyNrState& tire_states_LR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::TMeasyNrState& tire_states_RR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::SuspensionState& sus_states_LF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::SuspensionState& sus_states_RF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::SuspensionState& sus_states_LR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                                   d24GPU::SuspensionState& sus_states_RR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_states, sizeof(SimState) * m_vehicle_count_tracker_states,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                          0));  // move the simState onto the GPU
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:__host__ void d24SolverHalfImplicitGPU::SetOutput(const std::string& output_file,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMalloc((void**)&m_device_response, m_device_size));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:__host__ void d24SolverHalfImplicitGPU::Solve() {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMalloc((void**)&m_device_response, m_device_size));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:            cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:            if (err != cudaSuccess) {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:            cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:            if (err != cudaSuccess) {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:            CHECK_CUDA_ERROR(cudaMemcpy(m_host_response + filled_response, m_device_response, m_device_size,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                                        cudaMemcpyDeviceToHost));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:__host__ double d24SolverHalfImplicitGPU::SolveStep(double t,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMalloc((void**)&m_device_response, m_device_size));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        if (err != cudaSuccess) {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        if (err != cudaSuccess) {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:__host__ void d24SolverHalfImplicitGPU::Write(double t, unsigned int time_steps_to_write) {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:__host__ void d24SolverHalfImplicitGPU::WriteToFile() {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:__host__ SimState d24SolverHalfImplicitGPU::GetSimState(unsigned int vehicle_index) {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        // Copy the specific SimState from the GPU to the host
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaMemcpy(&host_state, &m_sim_states[vehicle_index], sizeof(SimState), cudaMemcpyDeviceToHost);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:        cudaMemcpy(&host_state, &m_sim_states_nr[vehicle_index], sizeof(SimState), cudaMemcpyDeviceToHost);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                       d24GPU::SimData* sim_data,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                       d24GPU::SimState* sim_states,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                       d24GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                       d24GPU::SimStateNr* sim_states_nr) {  // Get the vehicle index
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                       d24GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                       d24GPU::SimStateNr* sim_states_nr,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                          d24GPU::SimData* sim_data,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                          d24GPU::SimState* sim_states) {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                          d24GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                          d24GPU::SimStateNr* sim_states_nr) {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                          d24GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cu:                          d24GPU::SimStateNr* sim_states_nr) {
wheeled_vehicle_models/24dof-gpu/README.md:The 24 DOF GPU model is theoretically exactly the same as the CPU model, in that it solves the same equations of motions. However, the GPU version of the model has solvers that support solving multiple sets of these equations of motions on different GPU threads parallely. The GPU models however have a more limited functionallity when compared to the CPU models. For instance:
wheeled_vehicle_models/24dof-gpu/README.md:- The GPU models only support the Half-Implicit integrator
wheeled_vehicle_models/24dof-gpu/README.md:- Because Sundials is not supported, the GPU models do not support Forward Sensitivity Analysis (FSA)
wheeled_vehicle_models/24dof-gpu/README.md:- The GPU models do not provide system RHS Jacobians
wheeled_vehicle_models/24dof-gpu/README.md:- The GPU models are not wrapped to Python
wheeled_vehicle_models/24dof-gpu/README.md:However, the GPU model can simultaneously simulate upto 300,000 vehicles in real-time at a time step of $1e-3$ when benchmarked on an Nvidia A100 GPU. See chapter 6 [here](https://uwmadison.box.com/s/2tsvr4adbrzklle30z0twpu2nlzvlayc) for more details.
wheeled_vehicle_models/24dof-gpu/README.md:The model parameters and inputs are provided in the same way as the CPU model. The repository also contains [demos](./demos/) that describe how the GPU models can be used.
wheeled_vehicle_models/24dof-gpu/README.md:See [here](../README.md#how-do-i-use-the-models) for a general description of how to run the demos. For the 11 DOF model, the demos are placed in the [demos](./demos) folder. The demos are built in the [build](../README.md#generate) folder. The demos require command line arguments specifying the input controls for the vehicles. Additionally, some of the GPU demos require information about the number of vehicles desired to be simulated and the number of threads per block to be launched. We recommend using `32` for the number of threads per block. For a description of how to create these control input files, or how to use the default control input files provided, see [here](../24dof/README.md#how-do-i-provide-driver-inputs).
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:project(d24dof_gpu LANGUAGES CXX CUDA)
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:enable_language(CUDA)
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:find_package(CUDA 8.0 REQUIRED)
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:    set(CUDA_NVCC_FLAGS_DEBUG "-G -g" CACHE STRING "NVCC debug flags" FORCE)
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/dof24_gpu.cuh
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/dof24_gpu.cu
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/utils_gpu.cuh
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/utils_gpu.cu
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:# Only half implicit solver available for GPU
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:  ${CMAKE_CURRENT_SOURCE_DIR}/dof24_halfImplicit_gpu.cuh
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:  ${CMAKE_CURRENT_SOURCE_DIR}/dof24_halfImplicit_gpu.cu
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:    demo_hmmwv_24gpu
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:    demo_hmmwv_varControls_24gpu
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:    demo_hmmwv_step_24gpu
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:    demo_hmmwv_controlsFunctor_24gpu
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:    # Enable separate compilation for CUDA
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:    set_property(TARGET ${PROGRAM} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:    # Set architecure based on this link - https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
wheeled_vehicle_models/24dof-gpu/CMakeLists.txt:    set_target_properties(${PROGRAM} PROPERTIES CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90")
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:#ifndef DOF24_GPU_CUH
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:#define DOF24_GPU_CUH
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:#include "utils_gpu.cuh"
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:namespace d24GPU {
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// http://www.tmeasy.de/. It is important to note that this implementation within the d24GPU namespace is exactly
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// the same as the implementation in the d24 namespace. The only difference is that the d24GPU namespace is meant
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// to be used on the GPU where standard library functions are not available.
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// http://www.tmeasy.de/. It is important to note that this implementation within the d24GPU namespace is exactly the
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// same as the implementation in the d24 namespace. The only difference is that the d24GPU namespace is meant to be
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// used on the GPU where standard library functions are not available.
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// important to note that this implementation within the d24GPU namespace is exactly the same as the implementation in
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// the d24 namespace. The only difference is that the d24GPU namespace is meant to be used on the GPU where standard
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// important to note that this implementation within the d24GPU namespace is exactly the same as the implementation in
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// the d24 namespace. The only difference is that the d24GPU namespace is meant to be used on the GPU where standard
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// implementation within the d24GPU namespace is exactly the same as the implementation in the d24 namespace. The only
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// difference is that the d24GPU namespace is meant to be used on the GPU where standard library functions are not
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:    int _steerMapSize;  //!< Size of the steering map - this variable is unique to the GPU implementation as we need to
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:///  d24GPU namespace is exactly the same as the implementation in the d24 namespace. The only difference is that the
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:///  d24GPU namespace is meant to be used on the GPU where standard library functions are not available.
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// within the d24GPU namespace is exactly the same as the implementation in the d24 namespace. The only difference is
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// that the d24GPU namespace is meant to be used on the GPU where standard library functions are not available.
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:// Data structures that help handing multiple vehicles as needed in GPU version
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// stores all the "data" required to simulate 1 vehicle on the GPU. This is something largely the user does not have to
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:        cudaFree(_driver_data);  // Assumes _driver_data was allocated with new[]
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:        cudaFree(_driver_data);  // Assumes _driver_data was allocated with new[]
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// stores all the "states" of 1 vehicle simulated on the GPU. The user can use this to get the states of the vehicle at
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:/// any point in time during the simulation through the solver (see d24SolverHalfImplicitGPU class)
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cuh:}  // namespace d24GPU
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:#include "dof24_gpu.cuh"
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:using namespace d24GPU;
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__host__ void d24GPU::tireInit(TMeasyParam* t_params) {
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__host__ void d24GPU::tireInit(TMeasyNrParam* t_params) {
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__host__ void d24GPU::initializeTireSus(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__host__ void d24GPU::initializeTireSus(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::vehToSusTransform(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::vehToSusTransform(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::vehToTireTransform(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::vehToTireTransform(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::tireToVehTransform(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::tireToVehTransform(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::computeForcesThroughSus(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::computeForcesThroughSus(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:d24GPU::tmxy_combined(double* f, double* fos, double s, double df0, double sm, double fm, double ss, double fs) {
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::computeCombinedCoulombForce(double* fx,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::computeTireRHS(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::computeTireRHS(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::computeTireCompressionVelocity(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::computeTireCompressionVelocity(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::computeSusRHS(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::computeSusRHS(const VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ double d24GPU::driveTorque(const VehicleParam* v_params, const double throttle, const double motor_speed) {
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::differentialSplit(double torque,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::computePowertrainRHS(VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::computePowertrainRHS(VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::computeVehicleRHS(VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__device__ void d24GPU::computeVehicleRHS(VehicleState* v_states,
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__host__ void d24GPU::setTireParamsJSON(TMeasyParam& t_params, const char* fileName) {
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__host__ void d24GPU::setTireParamsJSON(TMeasyNrParam& t_params, const char* fileName) {
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:double d24GPU::GetTireMaxLoad(unsigned int li) {
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__host__ void d24GPU::GuessTruck80Par(unsigned int li,          // tire load index
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__host__ void d24GPU::GuessTruck80Par(double tireLoad,   // tire load index
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__host__ void d24GPU::GuessPassCar70Par(unsigned int li,          // tire load index
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__host__ void d24GPU::GuessPassCar70Par(double tireLoad,            // tire load index
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__host__ void d24GPU::setSuspensionParamsJSON(SuspensionParam& sus_params, const char* fileName) {
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:__host__ void d24GPU::setVehParamsJSON(VehicleParam& v_params, const char* fileName) {
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._steerMap, sizeof(MapEntry) * steerMapSize));
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:    CHECK_CUDA_ERROR(
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:        cudaMallocManaged((void**)&v_params._gearRatios, sizeof(double) * noGears));  // assign the memory for the gears
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._shiftMap, sizeof(MapEntry) * noGears));
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._powertrainMap, sizeof(MapEntry) * powertrainMapSize));
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._lossesMap, sizeof(MapEntry) * lossesMapSize));
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._CFmap, sizeof(MapEntry) * CFmapSize));
wheeled_vehicle_models/24dof-gpu/dof24_gpu.cu:        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._TRmap, sizeof(MapEntry) * TRmapSize));
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:// same driver inputs on the GPU. The Half Implicit solver, designed for GPU execution, is used.
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:// The API resembles the CPU version but includes specific GPU settings such as vehicle counts and threads per block.
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:#include <cuda.h>
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:#include "dof24_halfImplicit_gpu.cuh"
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:using namespace d24GPU;
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:    std::string inputPath = "../../24dof-gpu/data/input/" + file_name + ".txt";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:    std::string vehParamsJSON = "../../24dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:    std::string tireParamsJSON = "../../24dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:    std::string susParamsJSON = "../../24dof-gpu/data/json/HMMWV/suspension.json";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:    // Construct the solver with GPU-specific settings
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:    d24SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:    cudaEvent_t start, stop;
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:    cudaEventCreate(&start);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:    cudaEventCreate(&stop);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:    cudaEventRecord(start);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:    cudaEventRecord(stop);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:    cudaEventSynchronize(stop);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_step_24gpu.cu:    cudaEventElapsedTime(&milliseconds, start, stop);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:// The Half Implicit solver, which is designed for GPU models, is utilized here.
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:#include <cuda.h>
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:#include "dof24_halfImplicit_gpu.cuh"
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:using namespace d24GPU;
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    std::string inputPath1 = "../../24dof-gpu/data/input/" + file_name_1 + ".txt";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    std::string inputPath2 = "../../24dof-gpu/data/input/" + file_name_2 + ".txt";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    std::string vehParamsJSON = "../../24dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    std::string tireParamsJSON = "../../24dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    std::string susParamsJSON = "../../24dof-gpu/data/json/HMMWV/suspension.json";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    d24SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    std::string outputPath = "../../24dof-gpu/data/output/";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    cudaEvent_t start, stop;
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    cudaEventCreate(&start);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    cudaEventCreate(&stop);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    cudaEventRecord(start);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    cudaEventRecord(stop);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    cudaEventSynchronize(stop);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_varControls_24gpu.cu:    cudaEventElapsedTime(&milliseconds, start, stop);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_24gpu.cu:// all operating on the same driver inputs on the GPU. Since the Half Implicit solver is the
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_24gpu.cu:// only one supported for the GPU models, that is what is used here. The structure of the API
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_24gpu.cu:#include <cuda.h>
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_24gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_24gpu.cu:#include "dof24_halfImplicit_gpu.cuh"
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_24gpu.cu:using namespace d24GPU;
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_24gpu.cu:    std::string inputPath = "../../24dof-gpu/data/input/" + file_name + ".txt";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_24gpu.cu:    std::string outputPath = "../../24dof-gpu/data/output/";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_24gpu.cu:    std::string vehParamsJSON = "../../24dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_24gpu.cu:    std::string tireParamsJSON = "../../24dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_24gpu.cu:    std::string susParamsJSON = "../../24dof-gpu/data/json/HMMWV/suspension.json";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_24gpu.cu:    d24SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:// basis using the Half Implicit solver on GPU. It includes functionality for setting control inputs through a functor.
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:#include <cuda.h>
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:#include "dof24_halfImplicit_gpu.cuh"
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:using namespace d24GPU;
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:    std::string inputPath = "../../24dof-gpu/data/input/" + file_name + ".txt";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:    std::string vehParamsJSON = "../../24dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:    std::string tireParamsJSON = "../../24dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:    std::string susParamsJSON = "../../24dof-gpu/data/json/HMMWV/suspension.json";
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:    // Construct the solver with GPU-specific settings
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:    d24SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:    cudaEvent_t start, stop;
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:    cudaEventCreate(&start);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:    cudaEventCreate(&stop);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:    cudaEventRecord(start);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:    cudaEventRecord(stop);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:    cudaEventSynchronize(stop);
wheeled_vehicle_models/24dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_24gpu.cu:    cudaEventElapsedTime(&milliseconds, start, stop);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:#ifndef DOF24_HALFIMPLICIT_GPU_CUH
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:#define DOF24_HALFIMPLICIT_GPU_CUH
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:#include "dof24_gpu.cuh"
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:class d24SolverHalfImplicitGPU {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:    __host__ d24SolverHalfImplicitGPU(unsigned int num_vehicles);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:    __host__ ~d24SolverHalfImplicitGPU();
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:    /// @brief Set the time for which the GPU kernel simulates without a sync between the vehicles
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:    /// The simulation proceeds by multiple launches of the GPU kernel for m_kernel_sim_time duration. This is mainly
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:    /// @brief Sets the threads per block for the GPU kernel. Defaults to 32
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:    void Initialize(d24GPU::VehicleState& vehicle_states,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::TMeasyState& tire_states_LF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::TMeasyState& tire_states_RF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::TMeasyState& tire_states_LR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::TMeasyState& tire_states_RR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::SuspensionState& sus_states_LF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::SuspensionState& sus_states_RF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::SuspensionState& sus_states_LR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::SuspensionState& sus_states_RR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:    void Initialize(d24GPU::VehicleState& vehicle_states,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::TMeasyNrState& tire_states_LF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::TMeasyNrState& tire_states_RF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::TMeasyNrState& tire_states_LR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::TMeasyNrState& tire_states_RR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::SuspensionState& sus_states_LF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::SuspensionState& sus_states_RF,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::SuspensionState& sus_states_LR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                    d24GPU::SuspensionState& sus_states_RR,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:    __host__ d24GPU::SimState GetSimState(unsigned int vehicle_index);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:    d24GPU::SimData* m_sim_data;          ///< Simulation data for all the vehicles. See d24GPU::SimData for more info
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:    d24GPU::SimDataNr* m_sim_data_nr;     ///< Simulation data but with the TMeasyNr tire for all the vehicles. See
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                                          // d24GPU::SimDataNr for more info
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:    d24GPU::SimState* m_sim_states;       ///< Simulation states for all the vehicles. See d24GPU::SimState for more
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:    d24GPU::SimStateNr* m_sim_states_nr;  ///< Simulation states but with the TMeasyNr tire for all the vehicles. See
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                                          // d24GPU::SimStateNr for more info
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            CHECK_CUDA_ERROR(cudaMalloc((void**)&m_device_response, m_device_size));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            if (err != cudaSuccess) {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            if (err != cudaSuccess) {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                          d24GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                          d24GPU::SimStateNr* sim_states_nr);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                          d24GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                          d24GPU::SimStateNr* sim_states_nr);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                          d24GPU::SimData* sim_data,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                          d24GPU::SimState* sim_states);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                          d24GPU::SimData* sim_data,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                          d24GPU::SimState* sim_states);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                       d24GPU::SimData* sim_data,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                       d24GPU::SimState* sim_states);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                       d24GPU::SimData* sim_data,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                       d24GPU::SimState* sim_states,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                       d24GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                       d24GPU::SimStateNr* sim_states_nr);
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                       d24GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                       d24GPU::SimStateNr* sim_states_nr,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                          d24GPU::SimData* sim_data,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                          d24GPU::SimState* sim_states) {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::VehicleState& v_states = sim_states[vehicle_id]._v_states;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::VehicleParam& veh_param = sim_data[vehicle_id]._veh_params;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::TMeasyState& tirelf_st = sim_states[vehicle_id]._tirelf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::TMeasyState& tirerf_st = sim_states[vehicle_id]._tirerf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::TMeasyState& tirelr_st = sim_states[vehicle_id]._tirelr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::TMeasyState& tirerr_st = sim_states[vehicle_id]._tirerr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::SuspensionState& suslf_st = sim_states[vehicle_id]._suslf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::SuspensionState& susrf_st = sim_states[vehicle_id]._susrf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::SuspensionState& suslr_st = sim_states[vehicle_id]._suslr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::SuspensionState& susrr_st = sim_states[vehicle_id]._susrr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                          d24GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                          d24GPU::SimStateNr* sim_states_nr) {
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::VehicleState& v_states = sim_states_nr[vehicle_id]._v_states;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::VehicleParam& veh_param = sim_data_nr[vehicle_id]._veh_params;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::TMeasyNrState& tirelf_st = sim_states_nr[vehicle_id]._tirelf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::TMeasyNrState& tirerf_st = sim_states_nr[vehicle_id]._tirerf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::TMeasyNrState& tirelr_st = sim_states_nr[vehicle_id]._tirelr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::TMeasyNrState& tirerr_st = sim_states_nr[vehicle_id]._tirerr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::SuspensionState& suslf_st = sim_states_nr[vehicle_id]._suslf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::SuspensionState& susrf_st = sim_states_nr[vehicle_id]._susrf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::SuspensionState& suslr_st = sim_states_nr[vehicle_id]._suslr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:            d24GPU::SuspensionState& susrr_st = sim_states_nr[vehicle_id]._susrr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                       d24GPU::SimData* sim_data,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                       d24GPU::SimState* sim_states,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::VehicleParam& veh_params = sim_data[vehicle_index]._veh_params;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::VehicleState& v_states = sim_states[vehicle_index]._v_states;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::TMeasyParam& tireTM_params = sim_data[vehicle_index]._tireTM_params;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::TMeasyState& tireTMlf_st = sim_states[vehicle_index]._tirelf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::TMeasyState& tireTMrf_st = sim_states[vehicle_index]._tirerf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::TMeasyState& tireTMlr_st = sim_states[vehicle_index]._tirelr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::TMeasyState& tireTMrr_st = sim_states[vehicle_index]._tirerr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::SuspensionParam& sus_params = sim_data[vehicle_index]._sus_params;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::SuspensionState& suslf_st = sim_states[vehicle_index]._suslf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::SuspensionState& susrf_st = sim_states[vehicle_index]._susrf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::SuspensionState& suslr_st = sim_states[vehicle_index]._suslr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::SuspensionState& susrr_st = sim_states[vehicle_index]._susrr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                       d24GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:                       d24GPU::SimStateNr* sim_states_nr,
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::VehicleParam& veh_params = sim_data_nr[vehicle_index]._veh_params;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::VehicleState& v_states = sim_states_nr[vehicle_index]._v_states;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::TMeasyNrParam& tireTM_params = sim_data_nr[vehicle_index]._tireTMNr_params;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::TMeasyNrState& tireTMlf_st = sim_states_nr[vehicle_index]._tirelf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::TMeasyNrState& tireTMrf_st = sim_states_nr[vehicle_index]._tirerf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::TMeasyNrState& tireTMlr_st = sim_states_nr[vehicle_index]._tirelr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::TMeasyNrState& tireTMrr_st = sim_states_nr[vehicle_index]._tirerr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::SuspensionParam& sus_params = sim_data_nr[vehicle_index]._sus_params;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::SuspensionState& suslf_st = sim_states_nr[vehicle_index]._suslf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::SuspensionState& susrf_st = sim_states_nr[vehicle_index]._susrf_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::SuspensionState& suslr_st = sim_states_nr[vehicle_index]._suslr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:        d24GPU::SuspensionState& susrr_st = sim_states_nr[vehicle_index]._susrr_st;
wheeled_vehicle_models/24dof-gpu/dof24_halfImplicit_gpu.cuh:#endif  // DOF24_HALFIMPLICIT_GPU_CUH
wheeled_vehicle_models/tests/11dof_gpu.cu:#include <cuda.h>
wheeled_vehicle_models/tests/11dof_gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/tests/11dof_gpu.cu:#include "dof11_halfImplicit_gpu.cuh"
wheeled_vehicle_models/tests/11dof_gpu.cu:TEST(dof11_gpu, acc_test) {
wheeled_vehicle_models/tests/11dof_gpu.cu:    std::string driver_file = "../../11dof-gpu/data/input/acc3.txt";
wheeled_vehicle_models/tests/11dof_gpu.cu:    std::string vehParamsJSON = (char*)"../../11dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/tests/11dof_gpu.cu:    std::string tireParamsJSON = (char*)"../../11dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/tests/11dof_gpu.cu:    d11SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/tests/11dof_gpu.cu:    d11GPU::VehicleState veh_st;
wheeled_vehicle_models/tests/11dof_gpu.cu:    d11GPU::TMeasyState tiref_st;
wheeled_vehicle_models/tests/11dof_gpu.cu:    d11GPU::TMeasyState tirer_st;
wheeled_vehicle_models/tests/11dof_gpu.cu:    d11GPU::SimState sim_state_1 = solver.GetSimState(0);
wheeled_vehicle_models/tests/11dof_gpu.cu:TEST(dof11_gpu, variable_controls) {
wheeled_vehicle_models/tests/11dof_gpu.cu:    std::string driver_file_1 = "../../11dof-gpu/data/input/" + file_name_1 + ".txt";
wheeled_vehicle_models/tests/11dof_gpu.cu:    std::string driver_file_2 = "../../11dof-gpu/data/input/" + file_name_2 + ".txt";
wheeled_vehicle_models/tests/11dof_gpu.cu:    std::string vehParamsJSON = (char*)"../../11dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/tests/11dof_gpu.cu:    std::string tireParamsJSON = (char*)"../../11dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/tests/11dof_gpu.cu:    d11SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/tests/11dof_gpu.cu:    d11GPU::VehicleState veh_st;
wheeled_vehicle_models/tests/11dof_gpu.cu:    d11GPU::TMeasyState tiref_st;
wheeled_vehicle_models/tests/11dof_gpu.cu:    d11GPU::TMeasyState tirer_st;
wheeled_vehicle_models/tests/11dof_gpu.cu:    d11GPU::SimState sim_state_1 = solver.GetSimState(499);
wheeled_vehicle_models/tests/11dof_gpu.cu:    d11GPU::SimState sim_state_2 = solver.GetSimState(999);
wheeled_vehicle_models/tests/18dof_gpu.cu:#include <cuda.h>
wheeled_vehicle_models/tests/18dof_gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/tests/18dof_gpu.cu:#include "dof18_halfImplicit_gpu.cuh"
wheeled_vehicle_models/tests/18dof_gpu.cu:TEST(dof18_gpu, acc_test) {
wheeled_vehicle_models/tests/18dof_gpu.cu:    std::string driver_file = "../../18dof-gpu/data/input/acc3.txt";
wheeled_vehicle_models/tests/18dof_gpu.cu:    std::string vehParamsJSON = (char*)"../../18dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/tests/18dof_gpu.cu:    std::string tireParamsJSON = (char*)"../../18dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18GPU::VehicleState veh_st;
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18GPU::TMeasyState tirelf_st;
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18GPU::TMeasyState tirerf_st;
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18GPU::TMeasyState tirelr_st;
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18GPU::TMeasyState tirerr_st;
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18GPU::SimState sim_state_1 = solver.GetSimState(0);
wheeled_vehicle_models/tests/18dof_gpu.cu:TEST(dof18_gpu, variable_controls) {
wheeled_vehicle_models/tests/18dof_gpu.cu:    std::string driver_file_1 = "../../18dof-gpu/data/input/" + file_name_1 + ".txt";
wheeled_vehicle_models/tests/18dof_gpu.cu:    std::string driver_file_2 = "../../18dof-gpu/data/input/" + file_name_2 + ".txt";
wheeled_vehicle_models/tests/18dof_gpu.cu:    std::string vehParamsJSON = (char*)"../../18dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/tests/18dof_gpu.cu:    std::string tireParamsJSON = (char*)"../../18dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18GPU::VehicleState veh_st;
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18GPU::TMeasyState tirelf_st;
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18GPU::TMeasyState tirerf_st;
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18GPU::TMeasyState tirelr_st;
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18GPU::TMeasyState tirerr_st;
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18GPU::SimState sim_state_1 = solver.GetSimState(499);
wheeled_vehicle_models/tests/18dof_gpu.cu:    d18GPU::SimState sim_state_2 = solver.GetSimState(999);
wheeled_vehicle_models/tests/CMakeLists.txt:# GPU tests dependencies
wheeled_vehicle_models/tests/CMakeLists.txt:option(BUILD_GPU_TESTS "Enable building of GPU tests" OFF)
wheeled_vehicle_models/tests/CMakeLists.txt:if(BUILD_GPU_TESTS)
wheeled_vehicle_models/tests/CMakeLists.txt:  enable_language(CUDA)
wheeled_vehicle_models/tests/CMakeLists.txt:  find_package(CUDA 8.0 REQUIRED)
wheeled_vehicle_models/tests/CMakeLists.txt:  message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
wheeled_vehicle_models/tests/CMakeLists.txt:  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
wheeled_vehicle_models/tests/CMakeLists.txt:  set(GPU_MODEL_FILES
wheeled_vehicle_models/tests/CMakeLists.txt:  ${CMAKE_SOURCE_DIR}/../18dof-gpu/dof18_gpu.cuh
wheeled_vehicle_models/tests/CMakeLists.txt:  ${CMAKE_SOURCE_DIR}/../18dof-gpu/dof18_gpu.cu
wheeled_vehicle_models/tests/CMakeLists.txt:  ${CMAKE_SOURCE_DIR}/../11dof-gpu/dof11_gpu.cuh
wheeled_vehicle_models/tests/CMakeLists.txt:  ${CMAKE_SOURCE_DIR}/../11dof-gpu/dof11_gpu.cu
wheeled_vehicle_models/tests/CMakeLists.txt:  ${CMAKE_SOURCE_DIR}/../24dof-gpu/dof24_gpu.cuh
wheeled_vehicle_models/tests/CMakeLists.txt:  ${CMAKE_SOURCE_DIR}/../24dof-gpu/dof24_gpu.cu
wheeled_vehicle_models/tests/CMakeLists.txt:  set(GPU_UTILS_FILES
wheeled_vehicle_models/tests/CMakeLists.txt:      ${CMAKE_SOURCE_DIR}/../utils/utils_gpu.cuh
wheeled_vehicle_models/tests/CMakeLists.txt:      ${CMAKE_SOURCE_DIR}/../utils/utils_gpu.cu
wheeled_vehicle_models/tests/CMakeLists.txt:  set(GPU_SOLVER_FILES
wheeled_vehicle_models/tests/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/../18dof-gpu/dof18_halfImplicit_gpu.cuh
wheeled_vehicle_models/tests/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/../18dof-gpu/dof18_halfImplicit_gpu.cu
wheeled_vehicle_models/tests/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/../11dof-gpu/dof11_halfImplicit_gpu.cuh
wheeled_vehicle_models/tests/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/../11dof-gpu/dof11_halfImplicit_gpu.cu
wheeled_vehicle_models/tests/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/../24dof-gpu/dof24_halfImplicit_gpu.cuh
wheeled_vehicle_models/tests/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/../24dof-gpu/dof24_halfImplicit_gpu.cu
wheeled_vehicle_models/tests/CMakeLists.txt:  set(GPU_TEST_PROGRAMS
wheeled_vehicle_models/tests/CMakeLists.txt:    18dof_gpu
wheeled_vehicle_models/tests/CMakeLists.txt:    11dof_gpu
wheeled_vehicle_models/tests/CMakeLists.txt:    24dof_gpu
wheeled_vehicle_models/tests/CMakeLists.txt:  include_directories(${CMAKE_SOURCE_DIR}/../18dof-gpu)
wheeled_vehicle_models/tests/CMakeLists.txt:  include_directories(${CMAKE_SOURCE_DIR}/../11dof-gpu)
wheeled_vehicle_models/tests/CMakeLists.txt:  include_directories(${CMAKE_SOURCE_DIR}/../24dof-gpu)
wheeled_vehicle_models/tests/CMakeLists.txt:  # Build GPU Tests
wheeled_vehicle_models/tests/CMakeLists.txt:  foreach(PROGRAM ${GPU_TEST_PROGRAMS})
wheeled_vehicle_models/tests/CMakeLists.txt:    add_executable(${PROGRAM} ${CMAKE_SOURCE_DIR}/${PROGRAM}.cu ${GPU_UTILS_FILES} ${GPU_MODEL_FILES} ${GPU_SOLVER_FILES})
wheeled_vehicle_models/tests/CMakeLists.txt:    # Enable separate compilation for CUDA
wheeled_vehicle_models/tests/CMakeLists.txt:    set_property(TARGET ${PROGRAM} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
wheeled_vehicle_models/tests/CMakeLists.txt:    # Set architecure based on this link - https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
wheeled_vehicle_models/tests/CMakeLists.txt:    set_target_properties(${PROGRAM} PROPERTIES CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90")
wheeled_vehicle_models/tests/CMakeLists.txt:  gtest_discover_tests(18dof_gpu)
wheeled_vehicle_models/tests/CMakeLists.txt:  gtest_discover_tests(11dof_gpu)
wheeled_vehicle_models/tests/CMakeLists.txt:  gtest_discover_tests(24dof_gpu)
wheeled_vehicle_models/tests/24dof_gpu.cu:#include <cuda.h>
wheeled_vehicle_models/tests/24dof_gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/tests/24dof_gpu.cu:#include "dof24_halfImplicit_gpu.cuh"
wheeled_vehicle_models/tests/24dof_gpu.cu:TEST(dof24_gpu, acc_test) {
wheeled_vehicle_models/tests/24dof_gpu.cu:    std::string driver_file = "../../24dof-gpu/data/input/acc3.txt";
wheeled_vehicle_models/tests/24dof_gpu.cu:    std::string vehParamsJSON = (char*)"../../24dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/tests/24dof_gpu.cu:    std::string tireParamsJSON = (char*)"../../24dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/tests/24dof_gpu.cu:    std::string susParamsJSON = (char*)"../../24dof-gpu/data/json/HMMWV/suspension.json";
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::VehicleState veh_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::TMeasyState tirelf_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::TMeasyState tirerf_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::TMeasyState tirelr_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::TMeasyState tirerr_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::SuspensionState suslf_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::SuspensionState susrf_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::SuspensionState suslr_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::SuspensionState susrr_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::SimState sim_state_1 = solver.GetSimState(0);
wheeled_vehicle_models/tests/24dof_gpu.cu:TEST(dof24_gpu, variable_controls) {
wheeled_vehicle_models/tests/24dof_gpu.cu:    std::string driver_file_1 = "../../24dof-gpu/data/input/" + file_name_1 + ".txt";
wheeled_vehicle_models/tests/24dof_gpu.cu:    std::string driver_file_2 = "../../24dof-gpu/data/input/" + file_name_2 + ".txt";
wheeled_vehicle_models/tests/24dof_gpu.cu:    std::string vehParamsJSON = (char*)"../../24dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/tests/24dof_gpu.cu:    std::string tireParamsJSON = (char*)"../../24dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/tests/24dof_gpu.cu:    std::string susParamsJSON = (char*)"../../24dof-gpu/data/json/HMMWV/suspension.json";
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::VehicleState veh_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::TMeasyState tirelf_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::TMeasyState tirerf_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::TMeasyState tirelr_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::TMeasyState tirerr_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::SuspensionState suslf_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::SuspensionState susrf_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::SuspensionState suslr_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::SuspensionState susrr_st;
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::SimState sim_state_1 = solver.GetSimState(499);
wheeled_vehicle_models/tests/24dof_gpu.cu:    d24GPU::SimState sim_state_2 = solver.GetSimState(999);
wheeled_vehicle_models/README.md:The Library of Low-Fidelity Vehicle Models is a collection of vehicle models of three varying fidelities each with Nvidia GPU and CPU versions:
wheeled_vehicle_models/README.md:1. 11 Degree Of Freedom (DOF) Model - [CPU](./11dof/), [Nvidia GPU](./11dof-gpu/)
wheeled_vehicle_models/README.md:2. 18 Degree Of Freedom (DOF) Model - [CPU](./18dof/), [Nvidia GPU](./18dof-gpu/)
wheeled_vehicle_models/README.md:3. 24 Degree Of Freedom (DOF) Model - [CPU](./24dof/), [Nvidia GPU](./24dof-gpu/)
wheeled_vehicle_models/README.md:6. Nvidia GPU along with NVCC and CUDA Version 8.0 or higher (for the GPU versions of models). We recommend installing the [CUDA Toolkit](https://developer.nvidia.com/cuda-12-0-0-download-archive) to meet these dependencies
wheeled_vehicle_models/README.md:- `BUILD_11DOF_GPU` - Bool to switch on the building of the Nvidia GPU version of 11 DOF model. Default: `OFF`
wheeled_vehicle_models/README.md:- `BUILD_18DOF_GPU` - Bool to switch on the building of the Nvidia GPU version of 18 DOF model. Default: `OFF`
wheeled_vehicle_models/README.md:- `BUILD_24DOF_GPU` - Bool to switch on the building of the Nvidia GPU version of 24 DOF model. Default: `OFF`
wheeled_vehicle_models/README.md:#### Optional - GPU models
wheeled_vehicle_models/README.md:If you set the `ON` for the GPU models, you also have the choice to set the following CMake options - we however recommend leaving these in their default state
wheeled_vehicle_models/README.md:- `CMAKE_CUDA_ARCHITECTURES` - Set this to the CUDA Architecture in your machine for highly optimized code. See [here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) for more details. Default: `52`
wheeled_vehicle_models/README.md:- `CUDA_HOST_COMPILER` - Path to CUDA compiler. This can be left empty for now. Default: blank
wheeled_vehicle_models/README.md:- `CUDA_SDK_ROOT_DIR` - Path to CUDA SDK. Leave as is. Default: blank
wheeled_vehicle_models/README.md:- `CUDA_TOOLKIT_ROOT_DIR` - Path to the CUDA Toolkit. Should be automatically filled in. Default: `path_to_toolkit_on_your_machine`
wheeled_vehicle_models/README.md:- `CUDA_USE_STATIC_CUDA_RUNTIME` - Boolean for whether CUDA runtime is statically linked. Leave as is. Default: `ON`
wheeled_vehicle_models/README.md:- `CUDA_rt_LIBRARY` - Path to CUDA runtime library. Should be automatically filled in. Default: `path_to_runtime_library_on_your_machine`
wheeled_vehicle_models/11dof/dof11.h:///  d11GPU namespace is exactly the same as the implementation in the d11 namespace. The only difference is that the
wheeled_vehicle_models/11dof/dof11.h:///  d11GPU namespace is meant to be used on the GPU where standard library functions are not available.
wheeled_vehicle_models/CMakeLists.txt:option(BUILD_11DOF_GPU "Build the 11dof-gpu project" OFF)
wheeled_vehicle_models/CMakeLists.txt:option(BUILD_18DOF_GPU "Build the 18dof-gpu project" OFF)
wheeled_vehicle_models/CMakeLists.txt:option(BUILD_24DOF_GPU "Build the 24dof-gpu project" OFF)
wheeled_vehicle_models/CMakeLists.txt:# Enable CUDA language globally
wheeled_vehicle_models/CMakeLists.txt:if(BUILD_11DOF_GPU OR BUILD_18DOF_GPU OR BUILD_24DOF_GPU)
wheeled_vehicle_models/CMakeLists.txt:    enable_language(CUDA)
wheeled_vehicle_models/CMakeLists.txt:    find_package(CUDA 8.0 REQUIRED)
wheeled_vehicle_models/CMakeLists.txt:    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
wheeled_vehicle_models/CMakeLists.txt:    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
wheeled_vehicle_models/CMakeLists.txt:    set(CUDA_NVCC_FLAGS_DEBUG "-G -g" CACHE STRING "NVCC debug flags" FORCE)
wheeled_vehicle_models/CMakeLists.txt:if(BUILD_11DOF_GPU)
wheeled_vehicle_models/CMakeLists.txt:    add_subdirectory(11dof-gpu)
wheeled_vehicle_models/CMakeLists.txt:if(BUILD_18DOF_GPU)
wheeled_vehicle_models/CMakeLists.txt:    add_subdirectory(18dof-gpu)
wheeled_vehicle_models/CMakeLists.txt:if(BUILD_24DOF_GPU)
wheeled_vehicle_models/CMakeLists.txt:    add_subdirectory(24dof-gpu)
wheeled_vehicle_models/24dof/dof24.h:/// within the d24GPU namespace is exactly the same as the implementation in the d24 namespace. The only difference is
wheeled_vehicle_models/24dof/dof24.h:/// that the d24GPU namespace is meant to be used on the GPU where standard library functions are not available.
wheeled_vehicle_models/18dof/dof18.h:///  d18GPU namespace is exactly the same as the implementation in the d18 namespace. The only difference is that the
wheeled_vehicle_models/18dof/dof18.h:///  d18GPU namespace is meant to be used on the GPU where standard library functions are not available.
wheeled_vehicle_models/utils/utils_gpu.cuh:#include <cuda.h>
wheeled_vehicle_models/utils/utils_gpu.cuh:#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
wheeled_vehicle_models/utils/utils_gpu.cuh:    if (err != cudaSuccess) {
wheeled_vehicle_models/utils/utils_gpu.cuh:        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
wheeled_vehicle_models/utils/utils_gpu.cuh:        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
wheeled_vehicle_models/utils/utils_gpu.cuh:/// @brief Struct through which driver inputs can be provided to the library of vehicle models - GPU version
wheeled_vehicle_models/utils/utils_gpu.cu:#include "utils_gpu.cuh"
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:#include "dof18_gpu.cuh"
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:using namespace d18GPU;
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ double d18GPU::driveTorque(const VehicleParam* v_params, const double throttle, const double motor_speed) {
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ void d18GPU::differentialSplit(double torque,
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ void d18GPU::vehToTireTransform(TMeasyState* tirelf_st,
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ void d18GPU::vehToTireTransform(TMeasyNrState* tirelf_st,
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ void d18GPU::tireToVehTransform(TMeasyState* tirelf_st,
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ void d18GPU::tireToVehTransform(TMeasyNrState* tirelf_st,
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ __host__ void d18GPU::tireInit(TMeasyParam* t_params) {
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ __host__ void d18GPU::tireInit(TMeasyNrParam* t_params) {
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:d18GPU::tmxy_combined(double* f, double* fos, double s, double df0, double sm, double fm, double ss, double fs) {
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ void d18GPU::computeCombinedCoulombForce(double* fx,
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ void d18GPU::computeTireLoads(double* loads,
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ void d18GPU::computeTireLoads(double* loads,
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ void d18GPU::computeTireRHS(TMeasyState* t_states,
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ void d18GPU::computeTireRHS(TMeasyNrState* t_states,
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ void d18GPU::computePowertrainRHS(VehicleState* v_states,
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ void d18GPU::computePowertrainRHS(VehicleState* v_states,
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__device__ void d18GPU::computeVehRHS(VehicleState* v_states,
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__host__ void d18GPU::setVehParamsJSON(VehicleParam& v_params, const char* fileName) {
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._steerMap, sizeof(MapEntry) * steerMapSize));
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:    CHECK_CUDA_ERROR(
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:        cudaMallocManaged((void**)&v_params._gearRatios, sizeof(double) * noGears));  // assign the memory for the gears
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._shiftMap, sizeof(MapEntry) * noGears));
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._powertrainMap, sizeof(MapEntry) * powertrainMapSize));
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._lossesMap, sizeof(MapEntry) * lossesMapSize));
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._CFmap, sizeof(MapEntry) * CFmapSize));
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_params._TRmap, sizeof(MapEntry) * TRmapSize));
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__host__ void d18GPU::setTireParamsJSON(TMeasyParam& t_params, const char* fileName) {
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__host__ void d18GPU::setTireParamsJSON(TMeasyNrParam& t_params, const char* fileName) {
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__host__ double d18GPU::GetTireMaxLoad(unsigned int li) {
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__host__ void d18GPU::GuessTruck80Par(unsigned int li,          // tire load index
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__host__ void d18GPU::GuessTruck80Par(double tireLoad,   // tire load index
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__host__ void d18GPU::GuessPassCar70Par(unsigned int li,          // tire load index
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cu:__host__ void d18GPU::GuessPassCar70Par(double tireLoad,            // tire load index
wheeled_vehicle_models/18dof-gpu/README.md:The 18 DOF GPU model is theoretically exactly the same as the CPU model, in that it solves the same equations of motions. However, the GPU version of the model has solvers that support solving multiple sets of these equations of motions on different GPU threads parallely. The GPU models however have a more limited functionallity when compared to the CPU models. For instance:
wheeled_vehicle_models/18dof-gpu/README.md:- The GPU models only support the Half-Implicit integrator
wheeled_vehicle_models/18dof-gpu/README.md:- Because Sundials is not supported, the GPU models do not support Forward Sensitivity Analysis (FSA)
wheeled_vehicle_models/18dof-gpu/README.md:- The GPU models do not provide system RHS Jacobians
wheeled_vehicle_models/18dof-gpu/README.md:- The GPU models are not wrapped to Python
wheeled_vehicle_models/18dof-gpu/README.md:However, the GPU model can simultaneously simulate upto 300,000 vehicles in real-time at a time step of $1e-3$ when benchmarked on an Nvidia A100 GPU. See chapter 6 [here](https://uwmadison.box.com/s/2tsvr4adbrzklle30z0twpu2nlzvlayc) for more details.
wheeled_vehicle_models/18dof-gpu/README.md:The model parameters and inputs are provided in the same way as the CPU model. The repository also contains [demos](./demos/) that describe how the GPU models can be used.
wheeled_vehicle_models/18dof-gpu/README.md:See [here](../README.md#how-do-i-use-the-models) for a general description of how to run the demos. For the 11 DOF model, the demos are placed in the [demos](./demos) folder. The demos are built in the [build](../README.md#generate) folder. The demos require command line arguments specifying the input controls for the vehicles. Additionally, some of the GPU demos require information about the number of vehicles desired to be simulated and the number of threads per block to be launched. We recommend using `32` for the number of threads per block. For a description of how to create these control input files, or how to use the default control input files provided, see [here](../18dof/README.md#how-do-i-provide-driver-inputs).
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:project(d18dof_gpu LANGUAGES CXX CUDA)
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:enable_language(CUDA)
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:find_package(CUDA 8.0 REQUIRED)
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:    set(CUDA_NVCC_FLAGS_DEBUG "-G -g" CACHE STRING "NVCC debug flags" FORCE)
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/dof18_gpu.cuh
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/dof18_gpu.cu
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/utils_gpu.cuh
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/utils_gpu.cu
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:# Only half implicit solver available for GPU
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:  ${CMAKE_CURRENT_SOURCE_DIR}/dof18_halfImplicit_gpu.cuh
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:  ${CMAKE_CURRENT_SOURCE_DIR}/dof18_halfImplicit_gpu.cu
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:    demo_hmmwv_18gpu
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:    demo_hmmwv_varControls_18gpu
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:    demo_hmmwv_step_18gpu
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:    demo_hmmwv_controlsFunctor_18gpu
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:    # Enable separate compilation for CUDA
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:    set_property(TARGET ${PROGRAM} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:    # Set architecure based on this link - https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
wheeled_vehicle_models/18dof-gpu/CMakeLists.txt:    set_target_properties(${PROGRAM} PROPERTIES CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90")
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:#ifndef DOF18_HALFIMPLICIT_GPU_CUH
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:#define DOF18_HALFIMPLICIT_GPU_CUH
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:#include "dof18_gpu.cuh"
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:class d18SolverHalfImplicitGPU {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:    __host__ d18SolverHalfImplicitGPU(unsigned int total_num_vehicles);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:    __host__ ~d18SolverHalfImplicitGPU();
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:    /// @brief Set the time for which the GPU kernel simulates without a sync between the vehicles
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:    /// The simulation proceeds by multiple launches of the GPU kernel for m_kernel_sim_time duration. This is mainly
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:    /// @brief Sets the threads per block for the GPU kernel. Defaults to 32
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:    __host__ void Initialize(d18GPU::VehicleState& vehicle_states,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                             d18GPU::TMeasyState& tire_states_LF,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                             d18GPU::TMeasyState& tire_states_RF,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                             d18GPU::TMeasyState& tire_states_LR,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                             d18GPU::TMeasyState& tire_states_RR,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:    __host__ void Initialize(d18GPU::VehicleState& vehicle_states,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                             d18GPU::TMeasyNrState& tire_states_LF,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                             d18GPU::TMeasyNrState& tire_states_RF,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                             d18GPU::TMeasyNrState& tire_states_LR,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                             d18GPU::TMeasyNrState& tire_states_RR,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:    __host__ d18GPU::SimState GetSimState(unsigned int vehicle_index);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:    d18GPU::SimData* m_sim_data;          ///< Simulation data for all the vehicles. See d18GPU::SimData for more info
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:    d18GPU::SimDataNr* m_sim_data_nr;     ///< Simulation data but with the TMeasyNr tire for all the vehicles. See
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                                          // d18GPU::SimDataNr for more info
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:    d18GPU::SimState* m_sim_states;       ///< Simulation states for all the vehicles. See d18GPU::SimState for more
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:    d18GPU::SimStateNr* m_sim_states_nr;  ///< Simulation states but with the TMeasyNr tire for all the vehicles. See
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                                          // d18GPU::SimStateNr for more info
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            CHECK_CUDA_ERROR(cudaMalloc((void**)&m_device_response, m_device_size));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            if (err != cudaSuccess) {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            if (err != cudaSuccess) {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                          d18GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                          d18GPU::SimStateNr* sim_states_nr);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                          d18GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                          d18GPU::SimStateNr* sim_states_nr);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                          d18GPU::SimData* sim_data,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                          d18GPU::SimState* sim_states);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                          d18GPU::SimData* sim_data,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                          d18GPU::SimState* sim_states);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                       d18GPU::SimData* sim_data,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                       d18GPU::SimState* sim_states);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                       d18GPU::SimData* sim_data,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                       d18GPU::SimState* sim_states,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                       d18GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                       d18GPU::SimStateNr* sim_states_nr);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                       d18GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                       d18GPU::SimStateNr* sim_states_nr,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                          d18GPU::SimData* sim_data,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                          d18GPU::SimState* sim_states) {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            d18GPU::VehicleState& v_states = sim_states[vehicle_id]._veh_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            d18GPU::VehicleParam& veh_param = sim_data[vehicle_id]._veh_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            d18GPU::TMeasyState& tirelf_st = sim_states[vehicle_id]._tirelf_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            d18GPU::TMeasyState& tirerf_st = sim_states[vehicle_id]._tirerf_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            d18GPU::TMeasyState& tirelr_st = sim_states[vehicle_id]._tirelr_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            d18GPU::TMeasyState& tirerr_st = sim_states[vehicle_id]._tirerr_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                          d18GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                          d18GPU::SimStateNr* sim_states_nr) {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            d18GPU::VehicleState& v_states = sim_states_nr[vehicle_id]._veh_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            d18GPU::VehicleParam& veh_param = sim_data_nr[vehicle_id]._veh_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            d18GPU::TMeasyNrState& tirelf_st = sim_states_nr[vehicle_id]._tirelf_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            d18GPU::TMeasyNrState& tirerf_st = sim_states_nr[vehicle_id]._tirerf_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            d18GPU::TMeasyNrState& tirelr_st = sim_states_nr[vehicle_id]._tirelr_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:            d18GPU::TMeasyNrState& tirerr_st = sim_states_nr[vehicle_id]._tirerr_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                       d18GPU::SimData* sim_data,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                       d18GPU::SimState* sim_states,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::VehicleParam& veh_param = sim_data[vehicle_index]._veh_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::VehicleState& veh_state = sim_states[vehicle_index]._veh_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::TMeasyParam& tireTM_param = sim_data[vehicle_index]._tireTM_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::TMeasyState& tireTMlf_state = sim_states[vehicle_index]._tirelf_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::TMeasyState& tireTMrf_state = sim_states[vehicle_index]._tirerf_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::TMeasyState& tireTMlr_state = sim_states[vehicle_index]._tirelr_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::TMeasyState& tireTMrr_state = sim_states[vehicle_index]._tirerr_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                       d18GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:                       d18GPU::SimStateNr* sim_states_nr,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::VehicleParam& veh_param = sim_data_nr[vehicle_index]._veh_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::VehicleState& veh_state = sim_states_nr[vehicle_index]._veh_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::TMeasyNrParam& tireTMNr_param = sim_data_nr[vehicle_index]._tireTMNr_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::TMeasyNrState& tireTMNrlf_state = sim_states_nr[vehicle_index]._tirelf_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::TMeasyNrState& tireTMNrrf_state = sim_states_nr[vehicle_index]._tirerf_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::TMeasyNrState& tireTMNrlr_state = sim_states_nr[vehicle_index]._tirelr_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:        d18GPU::TMeasyNrState& tireTMNrrr_state = sim_states_nr[vehicle_index]._tirerr_state;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cuh:#endif  // DOF18_HALFIMPLICIT_GPU_CUH
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:#ifndef DOF18_GPU_CUH
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:#define DOF18_GPU_CUH
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:#include "utils_gpu.cuh"
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:namespace d18GPU {
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// http://www.tmeasy.de/. It is important to note that this implementation within the d18GPU namespace is exactly the
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// same as the implementation in the d18 namespace. The only difference is that the d18GPU namespace is meant to be
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// used on the GPU where standard library functions are not available.
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// http://www.tmeasy.de/. It is important to note that this implementation within the d18GPU namespace is exactly the
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// same as the implementation in the d18 namespace. The only difference is that the d18GPU namespace is meant to be
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// used on the GPU where standard library functions are not available.
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// important to note that this implementation within the d18GPU namespace is exactly the same as the implementation in
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// the d18 namespace. The only difference is that the d18GPU namespace is meant to be used on the GPU where standard
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// important to note that this implementation within the d18GPU namespace is exactly the same as the implementation in
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// the d18 namespace. The only difference is that the d18GPU namespace is meant to be used on the GPU where standard
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// note that this implementation within the d18GPU namespace is exactly the same as the implementation in the d18
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// namespace. The only difference is that the d18GPU namespace is meant to be used on the GPU where standard library
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:    int _steerMapSize;  //!< Size of the steering map - this variable is unique to the GPU implementation as we need to
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:///  d18GPU namespace is exactly the same as the implementation in the d18 namespace. The only difference is that the
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:///  d18GPU namespace is meant to be used on the GPU where standard library functions are not available.
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:// Data structures that help handing multiple vehicles as needed in GPU version
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// stores all the "data" required to simulate 1 vehicle on the GPU. This is something largely the user does not have to
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:        cudaFree(_driver_data);  // Assumes _driver_data was allocated with new[]
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:        cudaFree(_driver_data);  // Assumes _driver_data was allocated with new[]
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// stores all the "states" of 1 vehicle simulated on the GPU. The user can use this to get the states of the vehicle at
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:/// any point in time during the simulation through the solver (see d18SolverHalfImplicitGPU class)
wheeled_vehicle_models/18dof-gpu/dof18_gpu.cuh:}  // namespace d18GPU
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:#include "dof18_gpu.cuh"
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:#include "dof18_halfImplicit_gpu.cuh"
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:using namespace d18GPU;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ d18SolverHalfImplicitGPU::d18SolverHalfImplicitGPU(unsigned int total_num_vehicles)
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data, sizeof(d18GPU::SimData) * m_total_num_vehicles));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data_nr, sizeof(d18GPU::SimDataNr) * m_total_num_vehicles));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_states, sizeof(d18GPU::SimState) * m_total_num_vehicles));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_states_nr, sizeof(d18GPU::SimStateNr) * m_total_num_vehicles));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    int deviceId = 0;         // Assume we are using GPU 0
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    cudaSetDevice(deviceId);  // Set the device
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ d18SolverHalfImplicitGPU::~d18SolverHalfImplicitGPU() {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaFree(m_sim_data);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaFree(m_sim_states);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaFree(m_sim_data_nr);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaFree(m_sim_states_nr);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    cudaFree(m_device_response);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ void d18SolverHalfImplicitGPU::Construct(const std::string& vehicle_params_file,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    cudaFree(m_sim_data_nr);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    cudaFree(m_sim_states_nr);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them up
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    d18GPU::VehicleParam veh_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    d18GPU::TMeasyParam tire_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data[i]._driver_data,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data, sizeof(m_sim_data[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                          0));  // move the simData onto the GPU
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ void d18SolverHalfImplicitGPU::Construct(const std::string& vehicle_params_file,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaFree(m_sim_data_nr);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaFree(m_sim_states_nr);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        d18GPU::VehicleParam veh_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        d18GPU::TMeasyParam tire_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:            CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data[i]._driver_data,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data, sizeof(m_sim_data[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                              0));  // move the simData onto the GPU
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaFree(m_sim_data);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaFree(m_sim_states);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        d18GPU::VehicleParam veh_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        d18GPU::TMeasyNrParam tire_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:            CHECK_CUDA_ERROR(cudaMallocManaged((void**)&m_sim_data_nr[i]._driver_data,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data_nr, sizeof(m_sim_data_nr[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                              0));  // move the simData onto the GPU
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ void d18SolverHalfImplicitGPU::Construct(const std::string& vehicle_params_file,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    cudaFree(m_sim_data_nr);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    cudaFree(m_sim_states_nr);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them up
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    d18GPU::VehicleParam veh_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    d18GPU::TMeasyParam tire_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data, sizeof(m_sim_data[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                          0));  // move the simData onto the GPU
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ void d18SolverHalfImplicitGPU::Construct(const std::string& vehicle_params_file,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaFree(m_sim_data_nr);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaFree(m_sim_states_nr);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        d18GPU::VehicleParam veh_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        d18GPU::TMeasyParam tire_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data, sizeof(m_sim_data[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                              0));  // move the simData onto the GPU
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaFree(m_sim_data);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaFree(m_sim_states);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        // Since cudaMallocManaged does not call the constructor for non-POD types, we create cpu structs and fill them
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        d18GPU::VehicleParam veh_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        d18GPU::TMeasyNrParam tire_param;
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_data_nr, sizeof(m_sim_data_nr[0]) * m_vehicle_count_tracker_params,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                              0));  // move the simData onto the GPU
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ void d18SolverHalfImplicitGPU::Initialize(d18GPU::VehicleState& vehicle_states,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                                   d18GPU::TMeasyState& tire_states_LF,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                                   d18GPU::TMeasyState& tire_states_RF,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                                   d18GPU::TMeasyState& tire_states_LR,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                                   d18GPU::TMeasyState& tire_states_RR,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_states, sizeof(SimState) * m_vehicle_count_tracker_states,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                          0));  // move the simState onto the GPU
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ void d18SolverHalfImplicitGPU::Initialize(d18GPU::VehicleState& vehicle_states,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                                   d18GPU::TMeasyNrState& tire_states_LF,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                                   d18GPU::TMeasyNrState& tire_states_RF,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                                   d18GPU::TMeasyNrState& tire_states_LR,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                                   d18GPU::TMeasyNrState& tire_states_RR,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(m_sim_states_nr, sizeof(SimState) * m_vehicle_count_tracker_states,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                          0));  // move the simState onto the GPU
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ void d18SolverHalfImplicitGPU::SetOutput(const std::string& output_file,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:    CHECK_CUDA_ERROR(cudaMalloc((void**)&m_device_response, m_device_size));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ void d18SolverHalfImplicitGPU::Solve() {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMalloc((void**)&m_device_response, m_device_size));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:            cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:            if (err != cudaSuccess) {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:            cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:            if (err != cudaSuccess) {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:            CHECK_CUDA_ERROR(cudaMemcpy(m_host_response + filled_response, m_device_response, m_device_size,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                                        cudaMemcpyDeviceToHost));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ double d18SolverHalfImplicitGPU::SolveStep(double t,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        CHECK_CUDA_ERROR(cudaMalloc((void**)&m_device_response, m_device_size));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        if (err != cudaSuccess) {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaError_t err = cudaGetLastError();
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        if (err != cudaSuccess) {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ void d18SolverHalfImplicitGPU::Write(double t, unsigned int time_steps_to_write) {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ void d18SolverHalfImplicitGPU::WriteToFile() {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:__host__ SimState d18SolverHalfImplicitGPU::GetSimState(unsigned int vehicle_index) {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        // Copy the specific SimState from the GPU to the host
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaMemcpy(&host_state, &m_sim_states[vehicle_index], sizeof(SimState), cudaMemcpyDeviceToHost);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:        cudaMemcpy(&host_state, &m_sim_states_nr[vehicle_index], sizeof(SimState), cudaMemcpyDeviceToHost);
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                       d18GPU::SimData* sim_data,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                       d18GPU::SimState* sim_states,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                       d18GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                       d18GPU::SimStateNr* sim_states_nr,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                          d18GPU::SimData* sim_data,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                          d18GPU::SimState* sim_states) {
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                          d18GPU::SimDataNr* sim_data_nr,
wheeled_vehicle_models/18dof-gpu/dof18_halfImplicit_gpu.cu:                          d18GPU::SimStateNr* sim_states_nr) {
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:// Since the Half Implicit solver is the only one supported for the GPU models, that is what is used here.
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:#include <cuda.h>
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:#include "dof18_halfImplicit_gpu.cuh"
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:using namespace d18GPU;
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:    std::string inputPath = "../../18dof-gpu/data/input/" + file_name + ".txt";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:    std::string outputPath = "../../18dof-gpu/data/output/";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:    std::string vehParamsJSON = "../../18dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:    std::string tireParamsJSON = "../../18dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:    d18SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:    cudaEvent_t start, stop;
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:    cudaEventCreate(&start);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:    cudaEventCreate(&stop);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:    cudaEventRecord(start);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:    cudaEventRecord(stop);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:    cudaEventSynchronize(stop);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_controlsFunctor_18gpu.cu:    cudaEventElapsedTime(&milliseconds, start, stop);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_varControls_18gpu.cu:// for the GPU models, that is what is used here.
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_varControls_18gpu.cu:#include <cuda.h>
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_varControls_18gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_varControls_18gpu.cu:#include "dof18_halfImplicit_gpu.cuh"
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_varControls_18gpu.cu:using namespace d18GPU;
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_varControls_18gpu.cu:    std::string driver_file_1 = "../../18dof-gpu/data/input/" + file_name_1 + ".txt";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_varControls_18gpu.cu:    std::string driver_file_2 = "../../18dof-gpu/data/input/" + file_name_2 + ".txt";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_varControls_18gpu.cu:    std::string vehParamsJSON = "../../18dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_varControls_18gpu.cu:    std::string tireParamsJSON = "../../18dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_varControls_18gpu.cu:    d18SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:// Since the Half Implicit solver is the only one supported for the GPU models, that is what is used here.
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:#include <cuda.h>
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:#include "dof18_halfImplicit_gpu.cuh"
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:using namespace d18GPU;
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:    std::string inputPath = "../../18dof-gpu/data/input/" + file_name + ".txt";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:    std::string vehParamsJSON = "../../18dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:    std::string tireParamsJSON = "../../18dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:    d18SolverHalfImplicitGPU solver(num_vehicles);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:    cudaEvent_t start, stop;
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:    cudaEventCreate(&start);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:    cudaEventCreate(&stop);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:    cudaEventRecord(start);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:    cudaEventRecord(stop);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:    cudaEventSynchronize(stop);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_step_18gpu.cu:    cudaEventElapsedTime(&milliseconds, start, stop);
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_18gpu.cu:// same driver inputs on the GPU. Since the Half Implicit solver is the only one supported for the GPU models,
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_18gpu.cu:#include <cuda.h>
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_18gpu.cu:#include <cuda_runtime.h>
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_18gpu.cu:#include "dof18_halfImplicit_gpu.cuh"
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_18gpu.cu:using namespace d18GPU;
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_18gpu.cu:    std::string inputBasePath = "../../18dof-gpu/data/input/";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_18gpu.cu:    std::string outputBasePath = "../../18dof-gpu/data/output/";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_18gpu.cu:    std::string vehParamsJSON = "../../18dof-gpu/data/json/HMMWV/vehicle.json";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_18gpu.cu:    std::string tireParamsJSON = "../../18dof-gpu/data/json/HMMWV/tmeasy.json";
wheeled_vehicle_models/18dof-gpu/demos/HMMWV/demo_hmmwv_18gpu.cu:    d18SolverHalfImplicitGPU solver(num_vehicles);

```
