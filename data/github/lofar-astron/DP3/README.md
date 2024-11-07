# https://github.com/lofar-astron/DP3

```console
CMake/FindNVML.cmake:if(${CUDA_VERSION_STRING} VERSION_LESS "9.1")
CMake/FindNVML.cmake:    string(CONCAT ERROR_MSG "--> ARCHER: Current CUDA version "
CMake/FindNVML.cmake:                         ${CUDA_VERSION_STRING}
CMake/FindNVML.cmake:    set(NVML_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
CMake/FindNVML.cmake:    set(NVML_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
CMake/FindNVML.cmake:              PATHS "C:/Program Files/NVIDIA Corporation/NVSMI")
CMake/FindNVML.cmake:    set(NVML_NAMES nvidia-ml)
CMake/FindNVML.cmake:    set(NVML_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x86_64-linux-gnu")
CMake/FindNVML.cmake:    set(NVML_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
docs/schemas/IDGImager.yml:    symbols: CPU_OPTIMIZED, CUDA_GENERIC, CPU_REFERENCE, HYBRID
docs/schemas/IDGImager.yml:      Type of algorithm to do the gridding with IDG. The options are  `CPU_OPTIMIZED`, `CUDA_GENERIC`, `CPU_REFERENCE` or 
docs/schemas/IDGImager.yml:      `HYBRID`. Note that `CUDA_GENERIC` and `HYBRID` have to make use of a CUDA compatible GPU.
docs/schemas/DDECal.yml:  usegpu:
docs/schemas/DDECal.yml:      Use GPU solver. This is an experimental feature only available for the iterative
docs/schemas/DDECal.yml:      diagonal solver and requires DP3 to be built with BUILD_WITH_CUDA=1.
docs/schemas/DDECal.yml:      Setting this to true will cause the host buffers to be kept during the execution if used in combination with the GPU solver.
steps/IDGImager.cc:  } else if (idg_type.compare("CUDA_GENERIC") == 0) {
steps/IDGImager.cc:    return idg::api::Type::CUDA_GENERIC;
steps/IDGImager.cc:    return idg::api::Type::HYBRID_CUDA_CPU_OPTIMIZED;
steps/SagecalPredict.h:#if defined(HAVE_LIBDIRAC) || defined(HAVE_LIBDIRAC_CUDA)
steps/SagecalPredict.h:#if defined(HAVE_LIBDIRAC) || defined(HAVE_LIBDIRAC_CUDA)
steps/SagecalPredict.h:#endif /* HAVE_LIBDIRAC || HAVE_LIBDIRAC_CUDA */
steps/SagecalPredict.h:#if defined(HAVE_LIBDIRAC) || defined(HAVE_LIBDIRAC_CUDA)
steps/SagecalPredict.h:#endif /* HAVE_LIBDIRAC || HAVE_LIBDIRAC_CUDA */
steps/SagecalPredict.cc:#if defined(HAVE_LIBDIRAC) || defined(HAVE_LIBDIRAC_CUDA)
steps/SagecalPredict.cc:#endif /* HAVE_LIBDIRAC || HAVE_LIBDIRAC_CUDA */
steps/SagecalPredict.cc:#if defined(HAVE_LIBDIRAC) || defined(HAVE_LIBDIRAC_CUDA)
steps/SagecalPredict.cc:#endif /* HAVE_LIBDIRAC || HAVE_LIBDIRAC_CUDA */
steps/SagecalPredict.cc:#if defined(HAVE_LIBDIRAC) || defined(HAVE_LIBDIRAC_CUDA)
steps/SagecalPredict.cc:#endif /* HAVE_LIBDIRAC || HAVE_LIBDIRAC_CUDA */
steps/SagecalPredict.cc:#if defined(HAVE_LIBDIRAC) || defined(HAVE_LIBDIRAC_CUDA)
steps/SagecalPredict.cc:#ifdef HAVE_LIBDIRAC /* mutually exclusive with HAVE_LIBDIRAC_CUDA */
steps/SagecalPredict.cc:#ifdef HAVE_LIBDIRAC_CUDA /* mutually exclusive with HAVE_LIBDIRAC */
steps/SagecalPredict.cc:    predict_visibilities_multifreq_withbeam_gpu(
steps/SagecalPredict.cc:    predict_visibilities_withsol_withbeam_gpu(
steps/SagecalPredict.cc:#endif /* HAVE_LIBDIRAC_CUDA */
steps/SagecalPredict.cc:#endif /* HAVE_LIBDIRAC || HAVE_LIBDIRAC_CUDA */
steps/SagecalPredict.cc:#if defined(HAVE_LIBDIRAC) || defined(HAVE_LIBDIRAC_CUDA)
steps/SagecalPredict.cc:#endif /* HAVE_LIBDIRAC || HAVE_LIBDIRAC_CUDA */
steps/SagecalPredict.cc:#if defined(HAVE_LIBDIRAC) || defined(HAVE_LIBDIRAC_CUDA)
steps/SagecalPredict.cc:#endif /* HAVE_LIBDIRAC || HAVE_LIBDIRAC_CUDA */
steps/SagecalPredict.cc:#ifdef HAVE_LIBDIRAC_CUDA
steps/SagecalPredict.cc:  os << "SagecalPredict (GPU) " << name_ << '\n';
steps/SagecalPredict.cc:#if defined(HAVE_LIBDIRAC) || defined(HAVE_LIBDIRAC_CUDA)
steps/SagecalPredict.cc:#endif /* HAVE_LIBDIRAC || HAVE_LIBDIRAC_CUDA */
steps/SagecalPredict.cc:#if defined(HAVE_LIBDIRAC) || defined(HAVE_LIBDIRAC_CUDA)
steps/SagecalPredict.cc:#endif /* HAVE_LIBDIRAC || HAVE_LIBDIRAC_CUDA */
base/IDGConfiguration.h:      if (proxy == "cuda-generic") proxyType = idg::api::Type::CUDA_GENERIC;
base/IDGConfiguration.h:      if (proxy == "hybrid-cuda-cpu-optimized")
base/IDGConfiguration.h:        proxyType = idg::api::Type::HYBRID_CUDA_CPU_OPTIMIZED;
CMakeLists.txt:option(BUILD_WITH_CUDA "Build with CUDA support" FALSE)
CMakeLists.txt:if(BUILD_WITH_CUDA)
CMakeLists.txt:    LANGUAGES CUDA C CXX)
CMakeLists.txt:  set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
CMakeLists.txt:  set(CMAKE_CUDA_ARCHITECTURES
CMakeLists.txt:      CACHE STRING "Specify GPU architecture(s) to compile for")
CMakeLists.txt:  add_definitions(-DHAVE_CUDA_SOLVER)
CMakeLists.txt:  find_package(CUDAToolkit REQUIRED)
CMakeLists.txt:  # Necessary to find the cuda.h file in iterativediagonalsolver as a result of resolving SolverFactory
CMakeLists.txt:  include_directories(${CUDAToolkit_INCLUDE_DIRS})
CMakeLists.txt:    cudawrappers
CMakeLists.txt:    GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers.git
CMakeLists.txt:  FetchContent_MakeAvailable(cudawrappers)
CMakeLists.txt:if(BUILD_WITH_CUDA)
CMakeLists.txt:  target_link_libraries(DDECal cudawrappers::cu)
CMakeLists.txt:  # The cudasolvers library is built as static for two reasons
CMakeLists.txt:  # the cuda ecosystem
CMakeLists.txt:    CudaSolvers STATIC ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc
CMakeLists.txt:    CudaSolvers
CMakeLists.txt:    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
CMakeLists.txt:  target_link_libraries(CudaSolvers PUBLIC cudawrappers::cu CUDA::nvToolsExt
CMakeLists.txt:                                           CUDA::cudart_static xsimd xtensor)
CMakeLists.txt:    CudaSolvers
CMakeLists.txt:    PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
CMakeLists.txt:               CUDA_RESOLVE_DEVICE_SYMBOLS ON
CMakeLists.txt:               CUDA_SEPARABLE_COMPILATION ON
CMakeLists.txt:  install(TARGETS CudaSolvers)
CMakeLists.txt:  if(HAVE_CUDA)
CMakeLists.txt:    enable_language(CUDA)
CMakeLists.txt:    find_package(CUDA QUIET REQUIRED)
CMakeLists.txt:    message(STATUS "CUDA_LIBRARIES ............ = ${CUDA_LIBRARIES}")
CMakeLists.txt:    include_directories(${CUDA_INCLUDE_DIRS})
CMakeLists.txt:if(BUILD_WITH_CUDA)
CMakeLists.txt:  list(APPEND DP3_LIBRARIES CudaSolvers)
CMakeLists.txt:  if(HAVE_CUDA)
CMakeLists.txt:    # if we use libdirac with CUDA support, we enable a different preprocessor def
CMakeLists.txt:    add_definitions(-DHAVE_LIBDIRAC_CUDA)
CMakeLists.txt:    add_definitions(-DHAVE_CUDA)
CMakeLists.txt:        ${CUDA_LIBRARIES}
CMakeLists.txt:        ${CUDA_CUBLAS_LIBRARIES}
CMakeLists.txt:        ${CUDA_CUFFT_LIBRARIES}
CMakeLists.txt:        ${CUDA_cusolver_LIBRARY}
CMakeLists.txt:        ${CUDA_cudadevrt_LIBRARY}
CMakeLists.txt:  if(BUILD_WITH_CUDA)
CMakeLists.txt:    set_target_properties(unittests PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
CMakeLists.txt:  if(LIBDIRAC_FOUND AND NOT HAVE_CUDA)
ddecal/SolverFactory.cc:#if defined(HAVE_CUDA_SOLVER)
ddecal/SolverFactory.cc:#include "gain_solvers/IterativeDiagonalSolverCuda.h"
ddecal/SolverFactory.cc:#if defined(HAVE_CUDA_SOLVER)
ddecal/SolverFactory.cc:  if (settings.use_gpu) {
ddecal/SolverFactory.cc:          return std::make_unique<IterativeDiagonalSolverCuda>(
ddecal/SolverFactory.cc:        "usegpu=true, but no GPU implementation for solver algorithm is "
ddecal/SolverFactory.cc:  if (settings.use_gpu) {
ddecal/SolverFactory.cc:        "usegpu=true, but DP3 is built without CUDA support.");
ddecal/test/unit/tSolvers.cc:#if defined(HAVE_CUDA_SOLVER)
ddecal/test/unit/tSolvers.cc:#include "../../gain_solvers/IterativeDiagonalSolverCuda.h"
ddecal/test/unit/tSolvers.cc:#if defined(HAVE_CUDA_SOLVER)
ddecal/test/unit/tSolvers.cc:BOOST_FIXTURE_TEST_CASE(iterative_diagonal_cuda, SolverTester,
ddecal/test/unit/tSolvers.cc:  dp3::ddecal::IterativeDiagonalSolverCuda solver;
ddecal/test/unit/tSolvers.cc:BOOST_FIXTURE_TEST_CASE(iterative_diagonal_cuda_keep_buffers, SolverTester,
ddecal/test/unit/tSolvers.cc:  dp3::ddecal::IterativeDiagonalSolverCuda solver{true};
ddecal/Settings.cc:      use_gpu(GetBool("usegpu", 0)),
ddecal/Settings.h:  const bool use_gpu;
ddecal/Settings.h:  // for the GPU solver
ddecal/gain_solvers/kernels/IterativeDiagonal.h:#include <cuda_runtime.h>
ddecal/gain_solvers/kernels/IterativeDiagonal.h:#include <cudawrappers/cu.hpp>
ddecal/gain_solvers/kernels/IterativeDiagonal.h:void LaunchSubtractKernel(cudaStream_t stream, size_t n_directions,
ddecal/gain_solvers/kernels/IterativeDiagonal.h:    cudaStream_t stream, size_t n_antennas, size_t n_visibilities,
ddecal/gain_solvers/kernels/IterativeDiagonal.h:    cudaStream_t stream, size_t n_visibilities, size_t n_direction_solutions,
ddecal/gain_solvers/kernels/IterativeDiagonal.h:void LaunchStepKernel(cudaStream_t stream, size_t n_visibilities,
ddecal/gain_solvers/kernels/IterativeDiagonal.cu:    cudaStream_t stream, size_t n_visibilities, size_t n_direction_solutions,
ddecal/gain_solvers/kernels/IterativeDiagonal.cu:void LaunchSubtractKernel(cudaStream_t stream, size_t n_directions,
ddecal/gain_solvers/kernels/IterativeDiagonal.cu:        destination[pol] = {CUDART_NAN, CUDART_NAN};
ddecal/gain_solvers/kernels/IterativeDiagonal.cu:    cudaStream_t stream, size_t n_antennas, size_t n_visibilities,
ddecal/gain_solvers/kernels/IterativeDiagonal.cu:    if (distance > CUDART_PI)
ddecal/gain_solvers/kernels/IterativeDiagonal.cu:      distance = distance - 2.0 * CUDART_PI;
ddecal/gain_solvers/kernels/IterativeDiagonal.cu:    else if (distance < -CUDART_PI)
ddecal/gain_solvers/kernels/IterativeDiagonal.cu:      distance = distance + 2.0 * CUDART_PI;
ddecal/gain_solvers/kernels/IterativeDiagonal.cu:void LaunchStepKernel(cudaStream_t stream, size_t n_visibilities,
ddecal/gain_solvers/kernels/Complex.h: * https://forums.developer.nvidia.com/t/additional-cucomplex-functions-cucnorm-cucsqrt-cucexp-and-some-complex-double-functions/36892
ddecal/gain_solvers/kernels/Complex.h: * Cuda complex implementation of std::arg
ddecal/gain_solvers/kernels/Complex.h: * Cuda complex implementation of std::polar
ddecal/gain_solvers/kernels/Common.h:#include <cudawrappers/cu.hpp>
ddecal/gain_solvers/kernels/Common.h:/// cu::DeviceMemory references, while the GPU kernels require the actual
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:#include "IterativeDiagonalSolverCuda.h"
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:#include <cuda_runtime.h>
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:IterativeDiagonalSolverCuda::IterativeDiagonalSolverCuda(bool keep_buffers)
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:void IterativeDiagonalSolverCuda::AllocateGPUBuffers(const SolveData& data) {
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:  gpu_buffers_.numerator = std::make_unique<cu::DeviceMemory>(
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:  gpu_buffers_.denominator = std::make_unique<cu::DeviceMemory>(
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:    gpu_buffers_.antenna_pairs.emplace_back(
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:    gpu_buffers_.solution_map.emplace_back(
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:    gpu_buffers_.solutions.emplace_back(SizeOfSolutions(max_n_visibilities));
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:    gpu_buffers_.next_solutions.emplace_back(
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:    gpu_buffers_.model.emplace_back(
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:    gpu_buffers_.residual.emplace_back(SizeOfResidual(max_n_visibilities));
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:void IterativeDiagonalSolverCuda::AllocateHostBuffers(const SolveData& data) {
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:void IterativeDiagonalSolverCuda::DeallocateHostBuffers() {
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:void IterativeDiagonalSolverCuda::CopyHostToHost(
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:void IterativeDiagonalSolverCuda::CopyHostToDevice(size_t ch_block,
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:  cu::DeviceMemory& device_solution_map = gpu_buffers_.solution_map[buffer_id];
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:      gpu_buffers_.antenna_pairs[buffer_id];
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:  cu::DeviceMemory& device_model = gpu_buffers_.model[buffer_id];
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:  cu::DeviceMemory& device_residual = gpu_buffers_.residual[buffer_id];
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:  cu::DeviceMemory& device_solutions = gpu_buffers_.solutions[buffer_id];
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:void IterativeDiagonalSolverCuda::PostProcessing(
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:IterativeDiagonalSolver::SolveResult IterativeDiagonalSolverCuda::Solve(
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:  if (!gpu_buffers_initialized_) {
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:    AllocateGPUBuffers(data);
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:    gpu_buffers_initialized_ = true;
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:    nvtxRangeId_t nvts_range_gpu = nvtxRangeStart("GPU");
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:      // copied to the GPU and the host buffers could theoretically be reused.
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:      // is scheduled using a second set of GPU buffers.
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:        // set of GPU buffers, wait for the compute_finished event to be
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:                       NDirections(), gpu_buffers_.solution_map[buffer_id],
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:                       gpu_buffers_.solutions[buffer_id],
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:                       gpu_buffers_.next_solutions[buffer_id],
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:                       gpu_buffers_.residual[buffer_id],
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:                       gpu_buffers_.residual[2], gpu_buffers_.model[buffer_id],
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:                       gpu_buffers_.antenna_pairs[buffer_id],
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:                       *gpu_buffers_.numerator, *gpu_buffers_.denominator);
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:          gpu_buffers_.next_solutions[buffer_id],
ddecal/gain_solvers/IterativeDiagonalSolverCuda.cc:    nvtxRangeEnd(nvts_range_gpu);
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:#ifndef DDECAL_GAIN_SOLVERS_ITERATIVE_DIAGONAL_SOLVER_CUDA_H_
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:#define DDECAL_GAIN_SOLVERS_ITERATIVE_DIAGONAL_SOLVER_CUDA_H_
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:#include <cudawrappers/cu.hpp>
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:class IterativeDiagonalSolverCuda final : public SolverBase {
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:  IterativeDiagonalSolverCuda(bool keep_buffers = false);
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:  void AllocateGPUBuffers(const SolveData& data);
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:  /// If this variable if false gpu buffers are not initialized
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:  bool gpu_buffers_initialized_ = false;
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:   * GPUBuffers hold the GPU memory used in ::Solve()
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:   * The GPU memory is of type cu::DeviceMemory. This is a wrapper around a
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:   * plain CUdeviceptr, provided by the cudawrappers library.
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:   * ::AllocateGPUBuffers) and stored in a vector. There are three exceptions:
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:  struct GPUBuffers {
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:  } gpu_buffers_;
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:   * plain void*, provided by the cudawrappers library.
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:   * Using an extra host-to-cuda-host-memory copy and then a
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:   * cuda-host-memory-to-gpu copy is faster than a direct host-to-gpu copy.
ddecal/gain_solvers/IterativeDiagonalSolverCuda.h:#endif  // DDECAL_GAIN_SOLVERS_ITERATIVE_DIAGONAL_SOLVER_CUDA_H_

```
