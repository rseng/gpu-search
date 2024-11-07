# https://github.com/project-asgard/asgard

```console
doxygen/installation.md:* If you have Nvidia GPU ASGarD can take advantage of the [linear algebra libraries](https://developer.nvidia.com/cublas) and custom [CUDA kernels](https://developer.nvidia.com/cuda-zone)
CHANGELOG.md:- [x] single gpu capability for low-level code (merged 10 Sep 2019)
CHANGELOG.md:    - [x] CMake CUDA language capability (merged 03 Dec 2019)
CHANGELOG.md:    - [x] blas on single gpu
README.md:| unit/g++/cuda  | ![Build Status](https://codebuild.us-east-2.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiblkzVDBCNm95TkdzMTlRUzRGbU9SVm5SMlNTVjR2amQySG1jQ0cwNnZjQlBnbklvOGhBRzhaOUpLK3pHNjZYKzhsU1M2amR6OUkyQ2lCTWZuWGY5UTlnPSIsIml2UGFyYW1ldGVyU3BlYyI6Ijd2QSsxWmJRem9UTXgwQXIiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=develop) |
testing/automated/clang-format-check.sh:device/asgard_preconditioner_gpu.cpp
testing/automated/clang-format-check.sh:asgard_resources_cuda.tpp
testing/tests_general.hpp:#ifdef ASGARD_USE_CUDA
CMakeLists.txt:option (ASGARD_USE_CUDA "Optional CUDA support for asgard" OFF)
CMakeLists.txt:cmake_dependent_option (ASGARD_USE_GPU_MEM_LIMIT "Allow the ability to limit the GPU memory used by kronmult (can hurt performance)" OFF "ASGARD_USE_CUDA" OFF)
CMakeLists.txt:  if (ASGARD_USE_CUDA)
CMakeLists.txt:  if (NOT ASGARD_USE_CUDA)
CMakeLists.txt:if (ASGARD_USE_GPU_MEM_LIMIT AND NOT ASGARD_USE_CUDA)
CMakeLists.txt:  message(FATAL_ERROR " ASGARD_USE_GPU_MEM_LIMIT=ON requires ASGARD_USE_CUDA=ON")
CMakeLists.txt:if (ASGARD_USE_CUDA)
CMakeLists.txt:    # CUDA has to be enabled before libasgard is created
CMakeLists.txt:        if ("$ENV{CUDAARCHS}" STREQUAL "")
CMakeLists.txt:            # ENV{CUDAARCHS} is used to set CMAKE_CUDA_ARCHITECTURES
CMakeLists.txt:            set (CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "Architecture for the CUDA device.")
CMakeLists.txt:        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
CMakeLists.txt:CMAKE_CUDA_ARCHITECTURES or environment variable CUDAARCHS \
CMakeLists.txt:CMAKE_CUDA_ARCHITECTURES could be specified as empty or 'False', \
CMakeLists.txt:but then the appropriate CMAKE_CUDA_FLAGS must be set manually.")
CMakeLists.txt:    enable_language (CUDA)
CMakeLists.txt:    find_package (CUDAToolkit REQUIRED)
CMakeLists.txt:    set (ASGARD_NUM_GPU_THREADS "1024" CACHE STRING "Number of threads for GPU launch kernels")
CMakeLists.txt:    set (ASGARD_NUM_GPU_BLOCKS "300" CACHE STRING "Number of blocks for GPU launch kernels")
CMakeLists.txt:                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_glkronmult_gpu.cpp>
CMakeLists.txt:                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_preconditioner_gpu.cpp>
CMakeLists.txt:if (ASGARD_USE_CUDA)
CMakeLists.txt:                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_glkronmult_gpu.cpp
CMakeLists.txt:                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_preconditioner_gpu.cpp
CMakeLists.txt:                                 PROPERTIES LANGUAGE CUDA)
CMakeLists.txt:    target_compile_features (libasgard PUBLIC cuda_std_14)
CMakeLists.txt:                         $<$<COMPILE_LANGUAGE:CUDA>:-Wl,-rpath,${CMAKE_BINARY_DIR}>
CMakeLists.txt:                           CUDA::cudart
CMakeLists.txt:                           CUDA::cublas
CMakeLists.txt:                           CUDA::cusparse
CMakeLists.txt:      	if (ASGARD_USE_CUDA)
CMakeLists.txt:  foreach(_opt CMAKE_CXX_FLAGS ASGARD_PRECISIONS ASGARD_USE_OPENMP ASGARD_USE_MPI ASGARD_USE_CUDA ASGARD_USE_PYTHON ASGARD_IO_HIGHFIVE KRON_MODE_GLOBAL KRON_MODE_GLOBAL_BLOCK)
CMakeLists.txt:  if (ASGARD_USE_CUDA)
CMakeLists.txt:    foreach(_opt CMAKE_CUDA_COMPILER CMAKE_CUDA_FLAGS ASGARD_USE_GPU_MEM_LIMIT)
CMakeLists.txt:    message(STATUS "  ASGARD_USE_CUDA=${ASGARD_USE_CUDA}")
asgard-config.cmake:if ("@ASGARD_USE_CUDA@")
asgard-config.cmake:  set(CMAKE_CUDA_COMPILER "@CMAKE_CUDA_COMPILER@")
asgard-config.cmake:  enable_language (CUDA)
asgard-config.cmake:  find_package (CUDAToolkit REQUIRED)
asgard-config.cmake:set(asgard_CUDA_FOUND   "@ASGARD_USE_CUDA@")
asgard-config.cmake:foreach(_asgard_module OPENMP MPI CUDA)
src/asgard_build_info.hpp.in:#cmakedefine ASGARD_USE_CUDA
src/asgard_build_info.hpp.in:#cmakedefine ASGARD_USE_GPU_MEM_LIMIT
src/asgard_build_info.hpp.in:#define ASGARD_NUM_GPU_BLOCKS @ASGARD_NUM_GPU_BLOCKS@
src/asgard_build_info.hpp.in:#define ASGARD_NUM_GPU_THREADS @ASGARD_NUM_GPU_THREADS@
src/asgard_resources_cuda.tpp:#include <cuda_runtime.h>
src/asgard_resources_cuda.tpp:    if (cudaMalloc((void **)&ptr, num_elems * sizeof(P)) != cudaSuccess)
src/asgard_resources_cuda.tpp:      auto success = cudaMemset((void *)ptr, 0, num_elems * sizeof(P));
src/asgard_resources_cuda.tpp:      expect(success == cudaSuccess);
src/asgard_resources_cuda.tpp:    auto const success = cudaFree(ptr);
src/asgard_resources_cuda.tpp:    // returning a cudartUnloading error code.
src/asgard_resources_cuda.tpp:    expect((success == cudaSuccess) || (success == cudaErrorCudartUnloading));
src/asgard_resources_cuda.tpp:static constexpr cudaMemcpyKind
src/asgard_resources_cuda.tpp:getCudaMemcpyKind(resource destination, resource source)
src/asgard_resources_cuda.tpp:      return cudaMemcpyHostToHost;
src/asgard_resources_cuda.tpp:      return cudaMemcpyDeviceToHost;
src/asgard_resources_cuda.tpp:    return cudaMemcpyHostToDevice;
src/asgard_resources_cuda.tpp:    return cudaMemcpyDeviceToDevice;
src/asgard_resources_cuda.tpp:  cudaMemcpyKind constexpr kind = getCudaMemcpyKind(out, in);
src/asgard_resources_cuda.tpp:  auto const success            = cudaMemcpy(dest, source, num_elems * sizeof(P), kind);
src/asgard_resources_cuda.tpp:  expect(success == cudaSuccess);
src/asgard_resources_cuda.tpp:  cudaMemcpyKind constexpr kind = getCudaMemcpyKind(out, in);
src/asgard_resources_cuda.tpp:      cudaMemcpy2D(dest, dest_stride * sizeof(P), source,
src/asgard_resources_cuda.tpp:  expect(success == cudaSuccess);
src/asgard_distribution.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_distribution.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_basis_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_basis_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_basis_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_moment_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_resources_host.tpp:  throw std::runtime_error("calling allocate_device without CUDA");
src/asgard_resources_host.tpp:  throw std::runtime_error("calling delete_device without CUDA");
src/asgard_resources_host.tpp:  throw std::runtime_error("calling copy_to_device without CUDA");
src/asgard_resources_host.tpp:  throw std::runtime_error("calling copy_to_device without CUDA");
src/asgard_resources_host.tpp:  throw std::runtime_error("calling copy_to_host without CUDA");
src/asgard_solver.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_solver.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_solver.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_solver.cpp:void apply_diagonal_precond(gpu::vector<P> const &pc, P dt,
src/asgard_solver.cpp:  kronmult::gpu_precon_jacobi(pc.size(), dt, pc.data(), x.data());
src/asgard_solver.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_solver.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_solver.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_solver_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_solver_tests.cpp:  fk::vector<P> const mf_gpu_gmres = [&operator_matrices, &gold, &b, dt]() {
src/asgard_solver_tests.cpp:  rmse_comparison(gold, mf_gpu_gmres, tol_factor);
src/asgard_solver_tests.cpp:  fk::vector<P> const mf_gpu_bicgstab = [&operator_matrices, &gold, &b, dt]() {
src/asgard_solver_tests.cpp:  rmse_comparison(gold, mf_gpu_bicgstab, tol_factor);
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:  auto stat = cudaDeviceSynchronize();
src/asgard_lib_dispatch_tests.cpp:  REQUIRE(stat == cudaSuccess);
src/asgard_lib_dispatch_tests.cpp:  stat = cudaDeviceSynchronize();
src/asgard_lib_dispatch_tests.cpp:  REQUIRE(stat == cudaSuccess);
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_time_advance.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:#include <cuda_runtime.h>
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:      gpu_terms(num_terms);
src/asgard_kronmult_matrix.cpp:    gpu_terms[t] = terms[t].clone_onto_device();
src/asgard_kronmult_matrix.cpp:  auto gpu_elem = elem.clone_onto_device();
src/asgard_kronmult_matrix.cpp:      std::move(gpu_terms), std::move(gpu_elem), grid.row_start, grid.col_start,
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:    // CUDA case, split evenly since parallelism is per kron-product
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kronmult_matrix.cpp:    std::vector<fk::vector<int, mem_type::owner, resource::device>> gpu_iA(
src/asgard_kronmult_matrix.cpp:    std::vector<fk::vector<int, mem_type::owner, resource::device>> gpu_col(
src/asgard_kronmult_matrix.cpp:    std::vector<fk::vector<int, mem_type::owner, resource::device>> gpu_row(
src/asgard_kronmult_matrix.cpp:    for (size_t i = 0; i < gpu_iA.size(); i++)
src/asgard_kronmult_matrix.cpp:      gpu_iA[i]  = list_iA[i].clone_onto_device();
src/asgard_kronmult_matrix.cpp:      gpu_col[i] = list_col_indx[i].clone_onto_device();
src/asgard_kronmult_matrix.cpp:      gpu_row[i] = list_row_indx[i].clone_onto_device();
src/asgard_kronmult_matrix.cpp:      num_ints += int64_t{gpu_iA[i].size()} + int64_t{gpu_col[i].size()} +
src/asgard_kronmult_matrix.cpp:                  int64_t{gpu_row[i].size()};
src/asgard_kronmult_matrix.cpp:        std::move(gpu_row), std::move(gpu_col), std::move(gpu_iA),
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:    std::vector<fk::vector<P, mem_type::owner, resource::device>> gpu_terms(
src/asgard_kronmult_matrix.cpp:      gpu_terms[t] = terms[t].clone_onto_device();
src/asgard_kronmult_matrix.cpp:    mat.update_stored_coefficients(std::move(gpu_terms));
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kronmult_matrix.cpp:  // base_line_entries are the entries that must always be loaded in GPU memory
src/asgard_kronmult_matrix.cpp:    // assume all terms will be loaded into the GPU, as one IMEX flag or another
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA // split the patterns into threes
src/asgard_kronmult_matrix.cpp:#ifndef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:    preset_gpu_gkron(gpu::sparse_handle const &hndl, imex_flag const imex)
src/asgard_kronmult_matrix.cpp:  gpu_global[imex_indx] = kronmult::global_gpu_operations<precision>(
src/asgard_kronmult_matrix.cpp:  size_t buff_size = gpu_global[imex_indx].size_workspace();
src/asgard_kronmult_matrix.cpp:  for (auto &glb : gpu_global)
src/asgard_kronmult_matrix.cpp:  int64_t total = gpu_global[imex_indx].memory() / (1024 * 1024);
src/asgard_kronmult_matrix.cpp:      std::cout << "  GPU: " << (1 + total / 1024) << "GB\n";
src/asgard_kronmult_matrix.cpp:      std::cout << "  GPU: " << total << "MB\n";
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:        if (mat.gpu_global[imex_indx].empty_values(vid))
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:        mat.gpu_global[imex_indx].update_values(vid, gvals);
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:    mat.gpu_pre_con_.clear();
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:  precision const *gpux = (rec == resource::device) ? x : get_buffer<workspace::dev_x>();
src/asgard_kronmult_matrix.cpp:  precision *gpuy       = (rec == resource::device) ? y : get_buffer<workspace::dev_y>();
src/asgard_kronmult_matrix.cpp:    kronmult::set_gpu_buffer_to_zero(num_active_, gpuy);
src/asgard_kronmult_matrix.cpp:    lib_dispatch::scal<resource::device>(num_active_, beta, gpuy, 1);
src/asgard_kronmult_matrix.cpp:  fk::copy_on_device(get_buffer<workspace::pad_x>(), gpux, num_active_);
src/asgard_kronmult_matrix.cpp:  gpu_global[imex].execute(); // this is global kronmult
src/asgard_kronmult_matrix.cpp:                                       get_buffer<workspace::pad_y>(), 1, gpuy, 1);
src/asgard_kronmult_matrix.cpp:    fk::copy_to_host<precision>(y, gpuy, num_active_);
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_moment.cpp:    // create a sparse version of this matrix and put it on the GPU
src/asgard_moment.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_vector.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_vector.hpp:namespace gpu
src/asgard_vector.hpp: * \brief Simple container for GPU data, interoperable with std::vector
src/asgard_vector.hpp:} // namespace gpu
src/device/asgard_kronmult.cpp:#ifdef ASGARD_USE_CUDA
src/device/asgard_kronmult.cpp:void set_gpu_buffer_to_zero(int64_t num, T *x)
src/device/asgard_kronmult.cpp:  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_kronmult.cpp:  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
src/device/asgard_kronmult.cpp:template void set_gpu_buffer_to_zero(int64_t, double *);
src/device/asgard_kronmult.cpp:template void set_gpu_buffer_to_zero(int64_t, float *);
src/device/asgard_kronmult.cpp:#ifdef ASGARD_USE_CUDA
src/device/asgard_kronmult.cpp:#ifdef ASGARD_USE_CUDA
src/device/asgard_kronmult.cpp:    return ASGARD_GPU_WARP_SIZE;
src/device/asgard_kronmult.cpp:  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_kronmult.cpp:  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
src/device/asgard_kronmult.cpp:  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_kronmult.cpp:  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
src/device/asgard_kronmult.cpp:  constexpr int max_blocks = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_kronmult.cpp:      (n >= 8) ? ASGARD_NUM_GPU_THREADS / 2 : ASGARD_NUM_GPU_THREADS;
src/device/asgard_kronmult.cpp:  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_kronmult.cpp:  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
src/device/asgard_kronmult.cpp:  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_kronmult.cpp:                                  ? ASGARD_NUM_GPU_THREADS / 2
src/device/asgard_kronmult.cpp:                                  : ASGARD_NUM_GPU_THREADS;
src/device/asgard_kronmult.cpp:  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_kronmult.cpp:  constexpr int max_threads = ASGARD_NUM_GPU_THREADS / 2;
src/device/asgard_kronmult.cpp:void gpu_dense(int const dimensions, int const n, int const output_size,
src/device/asgard_kronmult.cpp:      throw std::runtime_error("kronmult unimplemented n for the gpu");
src/device/asgard_kronmult.cpp:      throw std::runtime_error("kronmult unimplemented n for the gpu");
src/device/asgard_kronmult.cpp:      throw std::runtime_error("kronmult unimplemented n for the gpu");
src/device/asgard_kronmult.cpp:      throw std::runtime_error("kronmult unimplemented n for the gpu");
src/device/asgard_kronmult.cpp:      throw std::runtime_error("kronmult unimplemented n for the gpu");
src/device/asgard_kronmult.cpp:      throw std::runtime_error("kronmult unimplemented n for the gpu");
src/device/asgard_kronmult.cpp:        "kronmult unimplemented number of dimensions for the gpu " +
src/device/asgard_kronmult.cpp:  cudaError_t err = cudaGetLastError(); // add
src/device/asgard_kronmult.cpp:  if (err != cudaSuccess)
src/device/asgard_kronmult.cpp:    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
src/device/asgard_kronmult.cpp:template void gpu_dense<double>(int const, int const, int const, int64_t const,
src/device/asgard_kronmult.cpp:template void gpu_dense<float>(int const, int const, int const, int64_t const,
src/device/asgard_kronmult_cycle2.hpp:  constexpr manual_sync sync_mode = (team_size > ASGARD_GPU_WARP_SIZE or
src/device/asgard_kronmult_cycle2.hpp:                                     ASGARD_GPU_WARP_SIZE % team_size != 0)
src/device/asgard_kronmult_cycle2.hpp:#if (CUDART_VERSION < 11070)
src/device/asgard_kronmult_cycle2.hpp:#if (CUDART_VERSION < 11070)
src/device/asgard_kronmult_cycle1.hpp:  constexpr manual_sync sync_mode = (team_size > ASGARD_GPU_WARP_SIZE or
src/device/asgard_kronmult_cycle1.hpp:                                     ASGARD_GPU_WARP_SIZE % team_size != 0)
src/device/asgard_kronmult_cycle1.hpp:  constexpr manual_sync sync_mode = (team_size > ASGARD_GPU_WARP_SIZE or
src/device/asgard_kronmult_cycle1.hpp:                                     ASGARD_GPU_WARP_SIZE % team_size != 0)
src/device/asgard_kronmult_cycle1.hpp:#if (CUDART_VERSION < 11070)
src/device/asgard_kronmult_cycle1.hpp:#if (CUDART_VERSION < 11070)
src/device/asgard_spkronmult_cycle1.hpp:  constexpr manual_sync sync_mode = (team_size > ASGARD_GPU_WARP_SIZE or
src/device/asgard_spkronmult_cycle1.hpp:                                     ASGARD_GPU_WARP_SIZE % team_size != 0)
src/device/asgard_spkronmult_cycle1.hpp:  constexpr manual_sync sync_mode = (team_size > ASGARD_GPU_WARP_SIZE or
src/device/asgard_spkronmult_cycle1.hpp:                                     ASGARD_GPU_WARP_SIZE % team_size != 0)
src/device/asgard_spkronmult_cycle1.hpp:#if (CUDART_VERSION < 11070)
src/device/asgard_cusparse.hpp:#ifdef ASGARD_USE_CUDA
src/device/asgard_cusparse.hpp:#include <cuda.h>
src/device/asgard_cusparse.hpp:#include <cuda_runtime.h>
src/device/asgard_cusparse.hpp:namespace asgard::gpu
src/device/asgard_cusparse.hpp: * \brief Helper template, converts float/double to CUDA_R_32F/64F
src/device/asgard_cusparse.hpp://! \brief Float specialization for CUDA_R_32F
src/device/asgard_cusparse.hpp:  //! \brief The corresponding cuda type
src/device/asgard_cusparse.hpp:  static const cudaDataType value = CUDA_R_32F;
src/device/asgard_cusparse.hpp://! \brief Float specialization for CUDA_R_64F
src/device/asgard_cusparse.hpp:  //! \brief The corresponding cuda type
src/device/asgard_cusparse.hpp:  static const cudaDataType value = CUDA_R_64F;
src/device/asgard_cusparse.hpp: * will be pushed onto the GPU device, so they can be used directly from
src/device/asgard_cusparse.hpp: * GPU memory with less synchronization.
src/device/asgard_cusparse.hpp: * by directly giving the number to the constructor of gpu::vector<T>.
src/device/asgard_cusparse.hpp: *   gpu::vector<T> buffer( sp_mat.size_workspace(cusp) );
src/device/asgard_cusparse.hpp:  //! \brief The cuda versions of the x/y vectors
src/device/asgard_cusparse.hpp:  gpu::vector<T> scale_factors_;
src/device/asgard_cusparse.hpp:} // namespace asgard::gpu
src/device/asgard_kronmult_cpu.cpp:#ifndef ASGARD_USE_CUDA // no need to compile for the CPU if CUDA is on
src/device/asgard_preconditioner_gpu.cpp:#ifdef ASGARD_USE_CUDA
src/device/asgard_preconditioner_gpu.cpp:__global__ void gpu_precon_jacobi(int64_t size, T dt, T const prec[], T x[])
src/device/asgard_preconditioner_gpu.cpp:void gpu_precon_jacobi(int64_t size, T dt, T const prec[], T x[])
src/device/asgard_preconditioner_gpu.cpp:  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_preconditioner_gpu.cpp:  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
src/device/asgard_preconditioner_gpu.cpp:  kernel::gpu_precon_jacobi<T><<<num_blocks, max_threads>>>(size, dt, prec, x);
src/device/asgard_preconditioner_gpu.cpp:template void gpu_precon_jacobi(int64_t size, double dt, double const prec[], double x[]);
src/device/asgard_preconditioner_gpu.cpp:template void gpu_precon_jacobi(int64_t size, float dt, float const prec[], float x[]);
src/device/asgard_glkronmult_gpu.cpp:#ifdef ASGARD_USE_CUDA
src/device/asgard_glkronmult_gpu.cpp:global_gpu_operations<precision>
src/device/asgard_glkronmult_gpu.cpp:::global_gpu_operations(gpu::sparse_handle const &hndl, int num_dimensions,
src/device/asgard_glkronmult_gpu.cpp:template struct global_gpu_operations<double>;
src/device/asgard_glkronmult_gpu.cpp:template struct global_gpu_operations<float>;
src/device/asgard_kronmult.hpp:void gpu_precon_jacobi(int64_t size, T dt, T const prec[], T x[]);
src/device/asgard_kronmult.hpp:#ifdef ASGARD_USE_CUDA
src/device/asgard_kronmult.hpp: * \brief Performs a batch of kronmult operations using a dense GPU matrix.
src/device/asgard_kronmult.hpp: * The arrays iA, vA, x and y are stored on the GPU device.
src/device/asgard_kronmult.hpp:void gpu_dense(int const dimensions, int const n, int const output_size,
src/device/asgard_kronmult.hpp: * \brief Sparse variant for the GPU.
src/device/asgard_kronmult.hpp:void gpu_sparse(int const dimensions, int const n, int const output_size,
src/device/asgard_kronmult.hpp:#ifdef ASGARD_USE_CUDA
src/device/asgard_kronmult.hpp: * The object will load the cpu data onto the gpu and will hold onto the vectors
src/device/asgard_kronmult.hpp:class global_gpu_operations
src/device/asgard_kronmult.hpp:  global_gpu_operations() : hndl_(nullptr), buffer_(nullptr)
src/device/asgard_kronmult.hpp:  global_gpu_operations(gpu::sparse_handle const &hndl, int num_dimensions,
src/device/asgard_kronmult.hpp:  //! \brief Set the work-buffer for the gpu sparse method
src/device/asgard_kronmult.hpp:  gpu::sparse_handle::htype hndl_;
src/device/asgard_kronmult.hpp:  std::vector<gpu::vector<int>> gpntr_;
src/device/asgard_kronmult.hpp:  std::vector<gpu::vector<int>> gindx_;
src/device/asgard_kronmult.hpp:  std::vector<gpu::vector<precision>> gvals_;
src/device/asgard_kronmult.hpp:  mutable std::vector<gpu::sparse_matrix<precision>> mats_;
src/device/asgard_spkronmult_cycle2.hpp:  constexpr manual_sync sync_mode = (team_size > ASGARD_GPU_WARP_SIZE or
src/device/asgard_spkronmult_cycle2.hpp:                                     ASGARD_GPU_WARP_SIZE % team_size != 0)
src/device/asgard_spkronmult_cycle2.hpp:#if (CUDART_VERSION < 11070)
src/device/asgard_kronmult_common.hpp:#ifdef ASGARD_USE_CUDA
src/device/asgard_kronmult_common.hpp:#include <cuda.h>
src/device/asgard_kronmult_common.hpp:#include <cuda_runtime.h>
src/device/asgard_kronmult_common.hpp:#define ASGARD_GPU_WARP_SIZE 32
src/device/asgard_kronmult_common.hpp: * \brief Computes the number of CUDA blocks.
src/device/asgard_kronmult_common.hpp:#ifdef ASGARD_USE_CUDA
src/device/asgard_kronmult_common.hpp:void set_gpu_buffer_to_zero(int64_t num, T *x);
src/device/asgard_kronmult_common.hpp://! \brief Specialization for the gpu case
src/device/asgard_kronmult_common.hpp:    set_gpu_buffer_to_zero(num, x);
src/device/asgard_kronmult_common.hpp:void set_buffer_to_zero(gpu::vector<T> &x)
src/device/asgard_kronmult_common.hpp:  set_gpu_buffer_to_zero(static_cast<int64_t>(x.size()), x.data());
src/device/asgard_spkronmult_cpu.cpp:#ifndef ASGARD_USE_CUDA // no need to compile for the CPU if CUDA is on
src/device/asgard_kronmult_cyclex.hpp:#if (CUDART_VERSION < 11070)
src/device/asgard_kronmult_cyclex.hpp:  constexpr manual_sync sync_mode = (team_size > ASGARD_GPU_WARP_SIZE or
src/device/asgard_kronmult_cyclex.hpp:                                     ASGARD_GPU_WARP_SIZE % team_size != 0)
src/device/asgard_spkronmult.cpp:#ifdef ASGARD_USE_CUDA
src/device/asgard_spkronmult.cpp:#ifdef ASGARD_USE_CUDA
src/device/asgard_spkronmult.cpp:    return ASGARD_GPU_WARP_SIZE;
src/device/asgard_spkronmult.cpp:  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_spkronmult.cpp:  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
src/device/asgard_spkronmult.cpp:  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_spkronmult.cpp:  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
src/device/asgard_spkronmult.cpp:  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_spkronmult.cpp:  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
src/device/asgard_spkronmult.cpp:  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_spkronmult.cpp:  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
src/device/asgard_spkronmult.cpp:  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_spkronmult.cpp:  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
src/device/asgard_spkronmult.cpp:  constexpr int max_blocks  = ASGARD_NUM_GPU_BLOCKS;
src/device/asgard_spkronmult.cpp:  constexpr int max_threads = ASGARD_NUM_GPU_THREADS;
src/device/asgard_spkronmult.cpp:void gpu_sparse(int const dimensions, int const n, int const output_size,
src/device/asgard_spkronmult.cpp:      throw std::runtime_error("kronmult unimplemented n for the gpu");
src/device/asgard_spkronmult.cpp:      throw std::runtime_error("kronmult unimplemented n for the gpu");
src/device/asgard_spkronmult.cpp:      throw std::runtime_error("kronmult unimplemented n for the gpu");
src/device/asgard_spkronmult.cpp:      throw std::runtime_error("kronmult unimplemented n for the gpu");
src/device/asgard_spkronmult.cpp:      throw std::runtime_error("kronmult unimplemented n for the gpu");
src/device/asgard_spkronmult.cpp:      throw std::runtime_error("kronmult unimplemented n for the gpu");
src/device/asgard_spkronmult.cpp:        "kronmult unimplemented number of dimensions for the gpu " +
src/device/asgard_spkronmult.cpp:template void gpu_sparse<double>(int const, int const, int const, int const,
src/device/asgard_spkronmult.cpp:template void gpu_sparse<float>(int const, int const, int const, int const,
src/device/asgard_spkronmult_cyclex.hpp:#if (CUDART_VERSION < 11070)
src/device/asgard_spkronmult_cyclex.hpp:  constexpr manual_sync sync_mode = (team_size > ASGARD_GPU_WARP_SIZE or
src/device/asgard_spkronmult_cyclex.hpp:                                     ASGARD_GPU_WARP_SIZE % team_size != 0)
src/asgard_tools.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_tools.hpp:#include <cuda.h>
src/asgard_tools.hpp:#include <cuda_runtime.h>
src/asgard_tools.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_tools.hpp:    cudaDeviceSynchronize(); // needed for accurate kronmult timing
src/asgard_fast_math.hpp:#ifndef ASGARD_USE_CUDA
src/asgard_fast_math.hpp: * Does not work with GPU vectors and does not check if the data is on the device.
src/asgard_fast_math.hpp: * Does not work with GPU vectors and does not check if the data is on the device.
src/asgard_time_advance_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_time_advance_tests.cpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_time_advance_tests.cpp:  cudaStream_t load_stream;
src/asgard_time_advance_tests.cpp:  cudaStreamCreate(&load_stream);
src/asgard_time_advance_tests.cpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_time_advance_tests.cpp:  cudaStreamDestroy(load_stream);
src/asgard_kronmult_matrix_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix_tests.cpp:      gpu_terms(num_terms);
src/asgard_kronmult_matrix_tests.cpp:    gpu_terms[t] = data->coefficients[t].clone_onto_device();
src/asgard_kronmult_matrix_tests.cpp:    terms_ptr[t] = gpu_terms[t].data();
src/asgard_kronmult_matrix_tests.cpp:  auto gpu_terms_ptr = terms_ptr.clone_onto_device();
src/asgard_kronmult_matrix_tests.cpp:      std::move(gpu_terms), std::move(elem), 0, 0, num_1d_blocks,
src/asgard_kronmult_matrix_tests.cpp:#ifndef ASGARD_USE_CUDA // test CPU kronmult only when CUDA is not enabled
src/asgard_kronmult_matrix_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix_tests.cpp:TEMPLATE_TEST_CASE("testing kronmult gpu 1d", "[gpu_sparse 1d]", test_precs)
src/asgard_kronmult_matrix_tests.cpp:TEMPLATE_TEST_CASE("testing kronmult gpu 1d", "[gpu_dense 1d]", test_precs)
src/asgard_kronmult_matrix_tests.cpp:TEMPLATE_TEST_CASE("testing kronmult gpu 2d", "[gpu_sparse 2d]", test_precs)
src/asgard_kronmult_matrix_tests.cpp:TEMPLATE_TEST_CASE("testing kronmult gpu 2d", "[gpu_dense 2d]", test_precs)
src/asgard_kronmult_matrix_tests.cpp:TEMPLATE_TEST_CASE("testing kronmult gpu 3d", "[gpu_sparse 3d]", test_precs)
src/asgard_kronmult_matrix_tests.cpp:TEMPLATE_TEST_CASE("testing kronmult gpu 3d", "[gpu_dense 3d]", test_precs)
src/asgard_kronmult_matrix_tests.cpp:TEMPLATE_TEST_CASE("testing kronmult gpu 4d", "[gpu_sparse 4d]", test_precs)
src/asgard_kronmult_matrix_tests.cpp:TEMPLATE_TEST_CASE("testing kronmult gpu 4d", "[gpu_dense 4d]", test_precs)
src/asgard_kronmult_matrix_tests.cpp:TEMPLATE_TEST_CASE("testing kronmult gpu 5d", "[gpu_sparse 5d]", test_precs)
src/asgard_kronmult_matrix_tests.cpp:TEMPLATE_TEST_CASE("testing kronmult gpu 5d", "[gpu_dense 5d]", test_precs)
src/asgard_kronmult_matrix_tests.cpp:TEMPLATE_TEST_CASE("testing kronmult gpu 6d", "[gpu_sparse 6d]", test_precs)
src/asgard_kronmult_matrix_tests.cpp:TEMPLATE_TEST_CASE("testing kronmult gpu 6d", "[gpu_dense 6d]", test_precs)
src/asgard_kronmult_matrix_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix_tests.cpp:  asgard::gpu::sparse_handle cusparse;
src/asgard_kronmult_matrix_tests.cpp:  asgard::gpu::vector<int> gpntr      = pntr;
src/asgard_kronmult_matrix_tests.cpp:  asgard::gpu::vector<int> gindx      = indx;
src/asgard_kronmult_matrix_tests.cpp:  asgard::gpu::vector<TestType> gvals = vals;
src/asgard_kronmult_matrix_tests.cpp:  asgard::gpu::vector<TestType> gx    = x;
src/asgard_kronmult_matrix_tests.cpp:  asgard::gpu::vector<TestType> gy(gx.size());
src/asgard_kronmult_matrix_tests.cpp:  asgard::gpu::sparse_matrix<TestType> mat(4, 4, indx.size(), gpntr.data(),
src/asgard_kronmult_matrix_tests.cpp:  asgard::gpu::vector<std::byte> work(work_size);
src/asgard_batch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_resources.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_resources.hpp:#include "asgard_resources_cuda.tpp"
src/asgard_sparse_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_time_advance.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_time_advance.cpp:#ifndef ASGARD_USE_CUDA
src/asgard_time_advance.cpp:#ifndef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:        "the GPU is enabled, so input vectors must have resource::device");
src/asgard_kronmult_matrix.hpp:        "the GPU is disabled, so input vectors must have resource::host");
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:   * desired sparse or dense mode and whether the CPU or GPU are being used.
src/asgard_kronmult_matrix.hpp:   * when running on the CPU and GPU respectively. The CPU format uses standard
src/asgard_kronmult_matrix.hpp:   * The GPU format row_indx.size() == col_indx.size() and each Kronecker
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:        "the GPU is enabled, so input vectors must have resource::device");
src/asgard_kronmult_matrix.hpp:        "the GPU is disabled, so input vectors must have resource::host");
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:   * \tparam multi_mode must be set to the host if using only the CPU or if CUDA
src/asgard_kronmult_matrix.hpp:   *         has out-of-core mode enabled, i.e., with ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kronmult_matrix.hpp:   *         loaded on the GPU and multi_mode must be set to device
src/asgard_kronmult_matrix.hpp:   *         for the CPU and device when CUDA is enabled
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kronmult_matrix.hpp:                         cudaStream_t stream)
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:      kronmult::gpu_dense(num_dimensions_, kron_size_, output_size(),
src/asgard_kronmult_matrix.hpp:        kronmult::gpu_sparse(num_dimensions_, kron_size_, output_size(),
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kronmult_matrix.hpp:        auto stats1 = cudaMemcpyAsync(load_buffer, list_iA[0].data(),
src/asgard_kronmult_matrix.hpp:                                      cudaMemcpyHostToDevice, load_stream);
src/asgard_kronmult_matrix.hpp:            cudaMemcpyAsync(load_buffer_rows, list_row_indx_[0].data(),
src/asgard_kronmult_matrix.hpp:                            cudaMemcpyHostToDevice, load_stream);
src/asgard_kronmult_matrix.hpp:            cudaMemcpyAsync(load_buffer_cols, list_col_indx_[0].data(),
src/asgard_kronmult_matrix.hpp:                            cudaMemcpyHostToDevice, load_stream);
src/asgard_kronmult_matrix.hpp:        expect(stats1 == cudaSuccess);
src/asgard_kronmult_matrix.hpp:        expect(stats2 == cudaSuccess);
src/asgard_kronmult_matrix.hpp:        expect(stats3 == cudaSuccess);
src/asgard_kronmult_matrix.hpp:          cudaStreamSynchronize(load_stream);
src/asgard_kronmult_matrix.hpp:            cudaStreamSynchronize(nullptr);
src/asgard_kronmult_matrix.hpp:            stats1 = cudaMemcpyAsync(load_buffer, list_iA[i + 1].data(),
src/asgard_kronmult_matrix.hpp:                                     cudaMemcpyHostToDevice, load_stream);
src/asgard_kronmult_matrix.hpp:                cudaMemcpyAsync(load_buffer_rows, list_row_indx_[i + 1].data(),
src/asgard_kronmult_matrix.hpp:                                cudaMemcpyHostToDevice, load_stream);
src/asgard_kronmult_matrix.hpp:                cudaMemcpyAsync(load_buffer_cols, list_col_indx_[i + 1].data(),
src/asgard_kronmult_matrix.hpp:                                cudaMemcpyHostToDevice, load_stream);
src/asgard_kronmult_matrix.hpp:            expect(stats1 == cudaSuccess);
src/asgard_kronmult_matrix.hpp:            expect(stats2 == cudaSuccess);
src/asgard_kronmult_matrix.hpp:            expect(stats3 == cudaSuccess);
src/asgard_kronmult_matrix.hpp:          // note that the first call to gpu_dense with the given output_size()
src/asgard_kronmult_matrix.hpp:          kronmult::gpu_sparse(num_dimensions_, kron_size_, output_size(),
src/asgard_kronmult_matrix.hpp:          kronmult::gpu_sparse(
src/asgard_kronmult_matrix.hpp:                  "CUDA not enabled, only resource::host is allowed for "
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:        "the GPU is enabled, so input vectors must have resource::device");
src/asgard_kronmult_matrix.hpp:        "the GPU is disabled, so input vectors must have resource::host");
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:        "the GPU is enabled, so input vectors must have resource::device");
src/asgard_kronmult_matrix.hpp:        "the GPU is disabled, so input vectors must have resource::host");
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:      if (gpu_pre_con_.empty())
src/asgard_kronmult_matrix.hpp:        gpu_pre_con_ = pre_con_;
src/asgard_kronmult_matrix.hpp:      return gpu_pre_con_;
src/asgard_kronmult_matrix.hpp:    static_assert(rec == resource::host, "GPU not enabled");
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kronmult_matrix.hpp:        "the GPU is enabled, the coefficient vectors have resource::device");
src/asgard_kronmult_matrix.hpp:        "the GPU memory usage has been limited, thus we are assuming that the "
src/asgard_kronmult_matrix.hpp:        "problem data will not fit in GPU memory and the index vectors must "
src/asgard_kronmult_matrix.hpp:                  "the GPU is enabled, the vectors have resource::device");
src/asgard_kronmult_matrix.hpp:        "the GPU is enabled, the coefficient vectors have resource::host");
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:  // indicates that the input vectors for single-call-mode will be on the GPU
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kronmult_matrix.hpp:  cudaStream_t load_stream;
src/asgard_kronmult_matrix.hpp:  // if memory is not limited, multiple vectors are all loaded on the GPU
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:  mutable gpu::vector<precision> gpu_pre_con_; // gpu copy
src/asgard_kronmult_matrix.hpp: * is available on the GPU device.
src/asgard_kronmult_matrix.hpp:// can be used with the cuSparse library (in CUDA mode)
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:    dev_x, // move CPU data to the GPU
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:  //! \brief The default vector to use for data, gpu::vector for gpu
src/asgard_kronmult_matrix.hpp:  using default_vector = gpu::vector<T>;
src/asgard_kronmult_matrix.hpp:  //! \brief On the GPU, we split the patterns into triples
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:  //! \brief Set the GPU side of the global kron, needed for the handle
src/asgard_kronmult_matrix.hpp:  void preset_gpu_gkron(gpu::sparse_handle const &hndl, imex_flag const imex);
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:      if (gpu_pre_con_.empty())
src/asgard_kronmult_matrix.hpp:        gpu_pre_con_ = pre_con_;
src/asgard_kronmult_matrix.hpp:      return gpu_pre_con_;
src/asgard_kronmult_matrix.hpp:    static_assert(rec == resource::host, "GPU not enabled");
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:  std::array<kronmult::global_gpu_operations<precision>, num_imex_variants> gpu_global;
src/asgard_kronmult_matrix.hpp:  mutable gpu::vector<precision> gpu_pre_con_; // gpu copy
src/asgard_kronmult_matrix.hpp:      // the buffers must be set before preset_gpu_gkron()
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:      kglobal.preset_gpu_gkron(sp_handle, entry);
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:        kglobal.preset_gpu_gkron(sp_handle, entry);
src/asgard_kronmult_matrix.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_matrix.hpp:  gpu::sparse_handle sp_handle;
src/asgard_kronmult_matrix.hpp:  gpu::vector<std::byte> gpu_sparse_buffer;
src/asgard_kronmult_matrix.hpp:    static_assert(rec == resource::host, "GPU not enabled");
src/asgard_sparse.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_sparse.hpp:#include <cuda_runtime.h>
src/asgard_program_options.cpp:-memory                  int        Memory limit for the GPU, applied to the earlier versions
src/asgard_program_options.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_program_options.cpp:  os << "GPU Acceleration         CUDA\n";
src/asgard_program_options.cpp:  os << "GPU Acceleration         Disabled\n";
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_fast_math_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch.cpp:// for calling cpu/gpu blas etc.
src/asgard_batch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_batch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_tensors_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_tensors_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_tensors_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_tensors_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_tensors_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_tensors_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_tensors_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_tensors_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_tensors_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_tensors_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_tensors_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_tensors_tests.cpp:TEMPLATE_TEST_CASE("gpu::vector", "[gpu::vector]", test_precs, int)
src/asgard_tensors_tests.cpp:    gpu::vector<TestType> gpu0; // make empty
src/asgard_tensors_tests.cpp:    REQUIRE(gpu0.size() == 0);
src/asgard_tensors_tests.cpp:    REQUIRE(gpu0.data() == nullptr);
src/asgard_tensors_tests.cpp:    REQUIRE(gpu0.empty());
src/asgard_tensors_tests.cpp:    gpu0.resize(10); // resize
src/asgard_tensors_tests.cpp:    REQUIRE(gpu0.size() == 10);
src/asgard_tensors_tests.cpp:    REQUIRE(gpu0.data() != nullptr);
src/asgard_tensors_tests.cpp:    REQUIRE(not gpu0.empty());
src/asgard_tensors_tests.cpp:    gpu0 = gpu::vector<TestType>(); // move-assign
src/asgard_tensors_tests.cpp:    REQUIRE(gpu0.size() == 0);
src/asgard_tensors_tests.cpp:    REQUIRE(gpu0.data() == nullptr);
src/asgard_tensors_tests.cpp:    gpu::vector<TestType> gpu1(cpu1); // copy construct (std::vector)
src/asgard_tensors_tests.cpp:    REQUIRE(data_match(cpu1, gpu1));
src/asgard_tensors_tests.cpp:    gpu::vector<TestType> gpu2(std::vector<TestType>{1, 2}); // move construct
src/asgard_tensors_tests.cpp:    REQUIRE(data_match(std::vector<TestType>{1, 2}, gpu2));
src/asgard_tensors_tests.cpp:    cpu2 = gpu0 = gpu2 = cpu1; // copy assignments
src/asgard_tensors_tests.cpp:    REQUIRE(data_match(cpu1, gpu2));
src/asgard_tensors_tests.cpp:    REQUIRE(data_match(gpu2, gpu0));
src/asgard_tensors_tests.cpp:    REQUIRE(data_match(gpu0, cpu2));
src/asgard_tensors_tests.cpp:    gpu0 = std::vector<TestType>{1, 2, 3, 4, 5, 6}; // move assign (std::vector)
src/asgard_tensors_tests.cpp:    REQUIRE(data_match(std::vector<TestType>{1, 2, 3, 4, 5, 6}, gpu0));
src/asgard_tensors_tests.cpp:    gpu1 = std::move(gpu0); // move assign
src/asgard_tensors_tests.cpp:    REQUIRE(gpu0.size() == 0);
src/asgard_tensors_tests.cpp:    REQUIRE(gpu0.data() == nullptr);
src/asgard_tensors_tests.cpp:    REQUIRE(data_match(std::vector<TestType>{1, 2, 3, 4, 5, 6}, gpu1));
src/asgard_tensors_tests.cpp:    gpu1.clear();
src/asgard_tensors_tests.cpp:    REQUIRE(gpu1.size() == 0);
src/asgard_tensors_tests.cpp:    REQUIRE(gpu1.data() == nullptr);
src/asgard_tensors_tests.cpp:    REQUIRE(gpu1.empty());
src/asgard_tensors_tests.cpp:    gpu0 = cpu1;
src/asgard_tensors_tests.cpp:    gpu::vector<TestType> gpu3(std::move(gpu0)); // move construct
src/asgard_tensors_tests.cpp:    REQUIRE(gpu0.empty());
src/asgard_tensors_tests.cpp:    REQUIRE(data_match(cpu1, gpu3));
src/asgard_resources_tests.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#include <cuda_runtime.h>
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:    auto success = cudaGetDeviceCount(&num_devices);
src/asgard_lib_dispatch.cpp:    expect(success == cudaSuccess);
src/asgard_lib_dispatch.cpp:    success = cudaSetDevice(local_rank);
src/asgard_lib_dispatch.cpp:    expect(success == cudaSuccess);
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:  if (cudaGetDeviceCount(&num_devices) != cudaSuccess)
src/asgard_lib_dispatch.cpp:    throw std::runtime_error("cannot read the number of GPUs");
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:        cudaMemcpy(y, x, n * sizeof(P), cudaMemcpyDeviceToDevice);
src/asgard_lib_dispatch.cpp:    expect(success == cudaSuccess);
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:    if (cudaMalloc((void **)&A_d, sizeof(P *)) != cudaSuccess)
src/asgard_lib_dispatch.cpp:    auto stat = cudaMemcpy(A_d, &A, sizeof(P *), cudaMemcpyHostToDevice);
src/asgard_lib_dispatch.cpp:    expect(stat == cudaSuccess);
src/asgard_lib_dispatch.cpp:    if (cudaMalloc((void **)&info_d, sizeof(int)) != cudaSuccess)
src/asgard_lib_dispatch.cpp:    stat = cudaMemcpy(&info, info_d, sizeof(int), cudaMemcpyDeviceToHost);
src/asgard_lib_dispatch.cpp:    expect(stat == cudaSuccess);
src/asgard_lib_dispatch.cpp:    stat = cudaFree(A_d);
src/asgard_lib_dispatch.cpp:    expect(stat == cudaSuccess);
src/asgard_lib_dispatch.cpp:    stat = cudaFree(info_d);
src/asgard_lib_dispatch.cpp:    expect(stat == cudaSuccess);
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:    if (cudaMalloc((void **)&A_d, sizeof(P *)) != cudaSuccess)
src/asgard_lib_dispatch.cpp:    if (cudaMalloc((void **)&work_d, sizeof(P *)) != cudaSuccess)
src/asgard_lib_dispatch.cpp:    if (cudaMalloc((void **)&info_d, sizeof(int)) != cudaSuccess)
src/asgard_lib_dispatch.cpp:    auto stat = cudaMemcpy(A_d, &A, sizeof(P *), cudaMemcpyHostToDevice);
src/asgard_lib_dispatch.cpp:    stat = cudaMemcpy(work_d, &work, sizeof(P *), cudaMemcpyHostToDevice);
src/asgard_lib_dispatch.cpp:    stat = cudaMemcpy(&info, info_d, sizeof(int), cudaMemcpyDeviceToHost);
src/asgard_lib_dispatch.cpp:    expect(stat == cudaSuccess);
src/asgard_lib_dispatch.cpp:    stat = cudaFree(A_d);
src/asgard_lib_dispatch.cpp:    expect(stat == cudaSuccess);
src/asgard_lib_dispatch.cpp:    stat = cudaFree(work_d);
src/asgard_lib_dispatch.cpp:    expect(stat == cudaSuccess);
src/asgard_lib_dispatch.cpp:    stat = cudaFree(info_d);
src/asgard_lib_dispatch.cpp:    expect(stat == cudaSuccess);
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:    if (cudaMalloc((void **)&a_d, list_size) != cudaSuccess)
src/asgard_lib_dispatch.cpp:    if (cudaMalloc((void **)&b_d, list_size) != cudaSuccess)
src/asgard_lib_dispatch.cpp:    if (cudaMalloc((void **)&c_d, list_size) != cudaSuccess)
src/asgard_lib_dispatch.cpp:    auto stat = cudaMemcpy(a_d, a, list_size, cudaMemcpyHostToDevice);
src/asgard_lib_dispatch.cpp:    expect(stat == cudaSuccess);
src/asgard_lib_dispatch.cpp:    stat = cudaMemcpy(b_d, b, list_size, cudaMemcpyHostToDevice);
src/asgard_lib_dispatch.cpp:    expect(stat == cudaSuccess);
src/asgard_lib_dispatch.cpp:    stat = cudaMemcpy(c_d, c, list_size, cudaMemcpyHostToDevice);
src/asgard_lib_dispatch.cpp:    expect(stat == cudaSuccess);
src/asgard_lib_dispatch.cpp:      auto const cuda_stat = cudaDeviceSynchronize();
src/asgard_lib_dispatch.cpp:      expect(cuda_stat == 0);
src/asgard_lib_dispatch.cpp:      auto const cuda_stat = cudaDeviceSynchronize();
src/asgard_lib_dispatch.cpp:      expect(cuda_stat == 0);
src/asgard_lib_dispatch.cpp:    stat = cudaFree(a_d);
src/asgard_lib_dispatch.cpp:    stat = cudaFree(b_d);
src/asgard_lib_dispatch.cpp:    stat = cudaFree(c_d);
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_lib_dispatch.cpp:    // set the correct cuda data type based on P=double or P=float
src/asgard_lib_dispatch.cpp:    cudaDataType_t constexpr fp_prec =
src/asgard_lib_dispatch.cpp:        std::is_same_v<P, double> ? CUDA_R_64F : CUDA_R_32F;
src/asgard_lib_dispatch.cpp:    auto success    = cudaMalloc(&sp_buffer, buffer_size);
src/asgard_lib_dispatch.cpp:    expect(success == cudaSuccess);
src/asgard_lib_dispatch.cpp:    success = cudaFree(sp_buffer);
src/asgard_lib_dispatch.cpp:    expect(success == cudaSuccess);
src/asgard_lib_dispatch.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_benchmark.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_kronmult_benchmark.cpp:        << " using CUDA backend\n"
src/asgard_io.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_io.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_io.hpp:#if defined(ASGARD_USE_CUDA)
src/asgard_io.hpp:  bool constexpr using_gpu = true;
src/asgard_io.hpp:  bool constexpr using_gpu = false;
src/asgard_io.hpp:  H5Easy::dump(file, "USING_GPU", using_gpu);
src/asgard_kron_operators.hpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kron_operators.hpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kron_operators.hpp:      auto status = cudaStreamDestroy(load_stream);
src/asgard_kron_operators.hpp:      expect(status == cudaSuccess);
src/asgard_kron_operators.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kron_operators.hpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kron_operators.hpp:        cudaStreamCreate(&load_stream);
src/asgard_kron_operators.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_kron_operators.hpp:#ifdef ASGARD_USE_GPU_MEM_LIMIT
src/asgard_kron_operators.hpp:  cudaStream_t load_stream;
src/asgard_moment.hpp:#ifdef ASGARD_USE_CUDA
src/asgard_basis.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_basis.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_basis.cpp:#ifdef ASGARD_USE_CUDA
src/asgard_basis.cpp:#ifdef ASGARD_USE_CUDA

```
