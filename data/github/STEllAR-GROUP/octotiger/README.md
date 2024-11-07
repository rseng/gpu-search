# https://github.com/STEllAR-GROUP/octotiger

```console
CITATION.cff:  affiliation: NVIDIA
doc/content/cmake-options.md:## GPGPU Integration
doc/content/cmake-options.md:* `OCTOTIGER_WITH_CUDA`: Enable [CUDA](https://docs.nvidia.com/cuda/) FMM kernels. The default value is `OFF`.
doc/conf.doxy.in:			 ../../src/cuda_util \
doc/conf.doxy.in:             ../../src/unitiger/hydro_impl/hydro_cuda \
README.md:#### Jenkins - All CPU / GPU node-level tests for the 8 major build configurations:
README.md:| CPU/GPU Tests with Kokkos, CUDA, HIP, SYCL | [![Build Status](https://rostam.cct.lsu.edu/jenkins/buildStatus/icon?job=Octo-Tiger+Node-Level%2Fmaster&config=nodelevel)](https://rostam.cct.lsu.edu/jenkins/job/Octo-Tiger%20Node-Level/job/master/) |
tools/docker/base_image/readme.md:This image exists for sake of facilitating testing non-CUDA Octo-Tiger builds
CMakeLists.txt:option(OCTOTIGER_WITH_CUDA "Enable CUDA kernels" OFF)
CMakeLists.txt:# Use workaround to make Intel GPUs work as intended
CMakeLists.txt:# May cause rare segfault during cleanup when using the KOKKOS SYCL backend on NVIDIA hardware
CMakeLists.txt:option(OCTOTIGER_WITH_INTEL_GPU_WORKAROUND "This activates a workaround that enables SYCL builds to run on Intel GPUs with Octotiger (but cause problems for SYCL builds with CUDA" OFF)
CMakeLists.txt:option(OCTOTIGER_WITH_FAST_FP_CONTRACT "Enable fp-contract=fast for CPU- and fmad for GPU kernels. Causes results of CPU and GPU to slightly differ (<1e-19) " OFF) # about 1e-20 in level 3 sod test
CMakeLists.txt:set(OCTOTIGER_CUDA_ARCH "sm_50"
CMakeLists.txt:  CACHE STRING "Compile for this CUDA arch" )
CMakeLists.txt:# (modifying CMAKE_CUDA_(CXX_)_FLAGS did not help, 
CMakeLists.txt:if(OCTOTIGER_WITH_CUDA)
CMakeLists.txt:    enable_language(CUDA)
CMakeLists.txt:    set(MY_CUDA_FLAGS -Xptxas=-fmad=true,-dlcm=cg,--opt-level=4)
CMakeLists.txt:    #string(APPEND HPX_CUDA_CLANG_FLAGS " -fast-math -Xcuda-ptxas -v")
CMakeLists.txt:    string(APPEND CUDA_NVCC_FLAGS ";-Xptxas=-v,-fmad=true,--opt-level=4")
CMakeLists.txt:  if(OCTOTIGER_WITH_CXX20 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12)
CMakeLists.txt:    message(FATAL_ERROR "CUDA builds with C++20 are not supported right now")
CMakeLists.txt:    message(FATAL_ERROR " CUDA cannot be used together with OCTOTIGER_WITH_LEGACY_Vc!")
CMakeLists.txt:  if(Kokkos_ENABLE_CUDA)
CMakeLists.txt:      enable_language(CUDA)
CMakeLists.txt:    src/cuda_util/cuda_scheduler.cpp
CMakeLists.txt:    src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp
CMakeLists.txt:    src/monopole_interactions/legacy/monopole_cuda_kernel.cpp
CMakeLists.txt:    src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp
CMakeLists.txt:    src/multipole_interactions/legacy/multipole_cuda_kernel.cpp
CMakeLists.txt:    src/unitiger/hydro_impl/hydro_boundary_exchange_cuda.cpp 
CMakeLists.txt:    octotiger/cuda_util/cuda_global_def.hpp
CMakeLists.txt:    octotiger/cuda_util/cuda_helper.hpp
CMakeLists.txt:    octotiger/cuda_util/cuda_scheduler.hpp
CMakeLists.txt:    octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp
CMakeLists.txt:    octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp
CMakeLists.txt:    octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp
CMakeLists.txt:    octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp
CMakeLists.txt:    octotiger/radiation/cuda_kernel.hpp
CMakeLists.txt:   src/unitiger/hydro_impl/flux_cuda_kernel.cpp
CMakeLists.txt:   src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp
CMakeLists.txt:   src/unitiger/hydro_impl/hydro_cuda_interface.cpp
CMakeLists.txt:    src/multipole_interactions/legacy/multipole_cuda_kernel.cpp
CMakeLists.txt:    src/monopole_interactions/legacy/monopole_cuda_kernel.cpp
CMakeLists.txt:    src/unitiger/hydro_impl/hydro_boundary_exchange_cuda.cpp)
CMakeLists.txt:   src/unitiger/hydro_impl/flux_cuda_kernel.cpp
CMakeLists.txt:   src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp)
CMakeLists.txt:if(OCTOTIGER_WITH_CUDA)
CMakeLists.txt:        src/unitiger/hydro_impl/hydro_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=fast --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH}")
CMakeLists.txt:        set_source_files_properties(src/multipole_interactions/multipole_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=fast --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH}")
CMakeLists.txt:        set_source_files_properties(src/multipole_interactions/monopole_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=fast --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH}")
CMakeLists.txt:        set_source_files_properties(src/unitiger/hydro_impl/hydro_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=fast --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH}")
CMakeLists.txt:        src/unitiger/hydro_impl/hydro_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=off --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH}")
CMakeLists.txt:        set_source_files_properties(src/multipole_interactions/multipole_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=off --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH}")
CMakeLists.txt:        set_source_files_properties(src/multipole_interactions/monopole_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=off --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH}")
CMakeLists.txt:        set_source_files_properties(src/unitiger/hydro_impl/hydro_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=off --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH}")
CMakeLists.txt:        set_source_files_properties(${cu_source_files} ${hydro_cu_source_files} src/unitiger/hydro_impl/hydro_kernel_interface.cpp src/multipole_interactions/multipole_kernel_interface.cpp src/monopole_interactions/monopole_kernel_interface.cpp PROPERTIES LANGUAGE CUDA)
CMakeLists.txt:        set_source_files_properties(${cu_source_files} ${hydro_cu_source_files} src/unitiger/hydro_impl/hydro_kernel_interface.cpp src/multipole_interactions/multipole_kernel_interface.cpp src/monopole_interactions/monopole_kernel_interface.cpp PROPERTIES LANGUAGE CUDA)
CMakeLists.txt:    #set_source_files_properties(${cu_source_files} ${hydro_cu_source_files} PROPERTIES LANGUAGE CUDA)
CMakeLists.txt:      # -DSTRICT_ANSI__ resolves the issue of clang trying to pull in float128 stuff in the standard libs for the GPU pass
CMakeLists.txt:      #set_source_files_properties(${cu_source_files} ${hydro_cu_source_files} PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=fast --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH} -D__STRICT_ANSI__")
CMakeLists.txt:          src/unitiger/hydro_impl/hydro_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=fast ${OCTOTIGER_ARCH_FLAG} --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH} -D__STRICT_ANSI__")
CMakeLists.txt:        set_source_files_properties(src/multipole_interactions/multipole_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=fast ${OCTOTIGER_ARCH_FLAG} --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH} -D__STRICT_ANSI__")
CMakeLists.txt:        set_source_files_properties(src/multipole_interactions/monopole_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=fast ${OCTOTIGER_ARCH_FLAG} --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH} -D__STRICT_ANSI__")
CMakeLists.txt:        set_source_files_properties(src/unitiger/hydro_impl/hydro_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=fast ${OCTOTIGER_ARCH_FLAG} --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH} -D__STRICT_ANSI__")
CMakeLists.txt:        src/unitiger/hydro_impl/hydro_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=off ${OCTOTIGER_ARCH_FLAG} --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH} -D__STRICT_ANSI__")
CMakeLists.txt:        set_source_files_properties(src/multipole_interactions/multipole_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=off ${OCTOTIGER_ARCH_FLAG} --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH} -D__STRICT_ANSI__")
CMakeLists.txt:        set_source_files_properties(src/multipole_interactions/monopole_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=off ${OCTOTIGER_ARCH_FLAG} --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH} -D__STRICT_ANSI__")
CMakeLists.txt:        set_source_files_properties(src/unitiger/hydro_impl/hydro_kernel_interface.cpp PROPERTIES COMPILE_FLAGS " -xcuda -fno-fast-math -ffp-contract=off ${OCTOTIGER_ARCH_FLAG} --cuda-gpu-arch=${OCTOTIGER_CUDA_ARCH} -D__STRICT_ANSI__")
CMakeLists.txt:        set_source_files_properties(${cu_source_files} ${hydro_cu_source_files} src/unitiger/hydro_impl/hydro_kernel_interface.cpp src/multipole_interactions/multipole_kernel_interface.cpp src/monopole_interactions/monopole_kernel_interface.cpp PROPERTIES LANGUAGE CUDA)
CMakeLists.txt:        set_source_files_properties(${cu_source_files} ${hydro_cu_source_files} src/unitiger/hydro_impl/hydro_kernel_interface.cpp src/multipole_interactions/multipole_kernel_interface.cpp src/monopole_interactions/monopole_kernel_interface.cpp PROPERTIES LANGUAGE CUDA)
CMakeLists.txt:  # Deal with the incompatibility of Kokkos+Cuda and Vc, HPX actions and general constexpr problems
CMakeLists.txt:    src/cuda_util/cuda_scheduler.cpp
CMakeLists.txt:    src/unitiger/hydro_impl/hydro_cuda_interface.cpp
CMakeLists.txt:    set_source_files_properties(${host_only_source_files} ${host_only_header_files} PROPERTIES COMPILE_FLAGS " -isystem ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} --host-only")
CMakeLists.txt:  target_compile_definitions(optionslib PUBLIC HPX_WITH_CUDA) # workaround for the kokkos hpx include
CMakeLists.txt:  target_compile_definitions(octolib PUBLIC HPX_WITH_CUDA) # workaround for the kokkos hpx include
CMakeLists.txt:  target_compile_definitions(hydrolib PUBLIC HPX_WITH_CUDA) # workaround for the kokkos hpx include
CMakeLists.txt:if(OCTOTIGER_WITH_CUDA)
CMakeLists.txt:  target_compile_definitions(octotiger PUBLIC OCTOTIGER_HAVE_CUDA)
CMakeLists.txt:  if(NOT OCTOTIGER_WITH_CUDA)
CMakeLists.txt:# Intel GPU workarounds...
CMakeLists.txt:if(OCTOTIGER_WITH_INTEL_GPU_WORKAROUND)
CMakeLists.txt:  message(WARNING " Compiling with SYCL Intel GPU workaround (might cause cleanup segfaults when using the SYCL CUDA backend)")
CMakeLists.txt:  target_compile_definitions(octolib PUBLIC OCTOTIGER_HAVE_INTEL_GPU_WORKAROUND)
CMakeLists.txt:  target_compile_definitions(hydrolib PUBLIC OCTOTIGER_HAVE_INTEL_GPU_WORKAROUND)
CMakeLists.txt:# Handle CUDA
CMakeLists.txt:if(OCTOTIGER_WITH_CUDA)
CMakeLists.txt:  message(INFO " CUDA support is turned on!")
CMakeLists.txt:  target_compile_definitions(octolib PUBLIC OCTOTIGER_HAVE_CUDA)
CMakeLists.txt:  target_compile_definitions(hydrolib PUBLIC OCTOTIGER_HAVE_CUDA)
CMakeLists.txt:  #set_property(TARGET octolib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
CMakeLists.txt:  #set_property(TARGET octolib PROPERTY INTERFACE_COMPILE_OPTIONS "${MY_CUDA_FLAGS}")
CMakeLists.txt:  #set_property(TARGET hydrolib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
CMakeLists.txt:  #set_property(TARGET hydrolib PROPERTY INTERFACE_COMPILE_OPTIONS "${MY_CUDA_FLAGS}")
octotiger/unitiger/physics_impl.hpp:/*** Reconstruct uses this - GPUize****/
octotiger/unitiger/physics_impl.hpp:/*** Reconstruct uses this - GPUize****/
octotiger/unitiger/physics.hpp:	/*** Reconstruct uses this - GPUize****/
octotiger/unitiger/physics.hpp:	/*** Reconstruct uses this - GPUize****/
octotiger/unitiger/hydro.hpp://#ifdef OCTOTIGER_WITH_CUDA
octotiger/unitiger/hydro.hpp:	const hydro::recon_type<NDIM>& reconstruct_cuda(hydro::state_type &U, const hydro::x_type&, safe_real);
octotiger/unitiger/hydro_impl/reconstruct.hpp:#include <octotiger/cuda_util/cuda_helper.hpp>
octotiger/unitiger/hydro_impl/reconstruct.hpp:#include <octotiger/cuda_util/cuda_scheduler.hpp>
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int number_dirs = 27;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int q_inx = INX + 2;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int q_inx2 = q_inx * q_inx;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int q_inx3 = q_inx * q_inx * q_inx;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int q_face_offset = number_dirs * q_inx3;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int u_face_offset = H_N3;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int x_offset = H_N3;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int q_dir_offset = q_inx3;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int am_offset = q_inx3;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int HR_DNX = H_NX * H_NX;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int HR_DNY = H_NX;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int HR_DNZ = 1;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int HR_DN0 = 0;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp://CUDA_GLOBAL_METHOD const int NDIR = 27;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int disc_offset = H_NX * H_NX * H_NX;
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int dir[27] = {
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const safe_real vw[27] = {
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int xloc[27][3] = {
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline double deg_pres(double x, double A_) {
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline int to_q_index(const int j, const int k, const int l) {
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline int flip(const int d) {
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline void cell_find_contact_discs_phase1(container_t &P,
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline void cell_find_contact_discs_phase2(
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline void cell_hydro_pre_recon(const_container_t& X, safe_real omega, bool angmom,
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline void make_monotone_simd(simd_t& ql, simd_t q0, simd_t& qr) {
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline simd_t minmod_simd(simd_t a, simd_t b) {
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline simd_t minmod_theta_simd(simd_t a, simd_t b, simd_t c) {
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline void cell_reconstruct_minmod_simd(double* __restrict__ combined_q,
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline void cell_reconstruct_ppm_simd(double *__restrict__ combined_q,
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline void cell_reconstruct_inner_loop_p1_simd(const size_t nf_,
octotiger/unitiger/hydro_impl/reconstruct_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline void cell_reconstruct_inner_loop_p2_simd(const safe_real omega,
octotiger/unitiger/hydro_impl/hydro_kernel_interface.hpp:    const size_t max_gpu_executor_queue_length);
octotiger/unitiger/hydro_impl/hydro_kernel_interface.hpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
octotiger/unitiger/hydro_impl/hydro_kernel_interface.hpp:timestep_t launch_hydro_cuda_kernels(const hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp:#include "octotiger/cuda_util/cuda_global_def.hpp"
octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp:#ifdef OCTOTIGER_HAVE_CUDA
octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp:#include <cuda_buffer_util.hpp>
octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp:#include <cuda_runtime.h>
octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp:#include "octotiger/cuda_util/cuda_helper.hpp"
octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp:using aggregated_executor_t = Aggregated_Executor<hpx::cuda::experimental::cuda_executor>::Executor_Slice;
octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp:void launch_reconstruct_cuda(
octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp:void launch_find_contact_discs_cuda(aggregated_executor_t& executor, double* combined_u, double *device_P, double* disc, double A_, double B_, double fgamma_, double de_switch_1, const int nf);
octotiger/unitiger/hydro_impl/reconstruct_kernel_interface.hpp:void launch_hydro_pre_recon_cuda(aggregated_executor_t& executor, 
octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp:#include "octotiger/cuda_util/cuda_global_def.hpp"
octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp:#ifdef OCTOTIGER_HAVE_CUDA
octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp:#include "octotiger/cuda_util/cuda_helper.hpp"
octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp:#include <cuda_buffer_util.hpp>
octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp:#include <cuda_runtime.h>
octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp:#include "octotiger/cuda_util/cuda_helper.hpp"
octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp:using aggregated_executor_t = Aggregated_Executor<hpx::cuda::experimental::cuda_executor>::Executor_Slice;
octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp:#if defined(OCTOTIGER_HAVE_CUDA) 
octotiger/unitiger/hydro_impl/flux_kernel_interface.hpp:void launch_flux_cuda_kernel_post(aggregated_executor_t& executor,
octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp:#ifdef OCTOTIGER_HAVE_CUDA
octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp:#include <cuda_buffer_util.hpp>
octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp:#include <cuda_runtime.h>
octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp:#include "octotiger/cuda_util/cuda_helper.hpp"
octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp:#ifdef OCTOTIGER_HAVE_CUDA
octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp:using aggregated_executor_t = Aggregated_Executor<hpx::cuda::experimental::cuda_executor>::Executor_Slice;
octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp:complete_hydro_amr_cuda_kernel(const double dx, const bool energy_only,
octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp:void launch_complete_hydro_amr_boundary_cuda(
octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp:void launch_complete_hydro_amr_boundary_cuda_post(
octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp:void launch_complete_hydro_amr_boundary_cuda_phase2_post(
octotiger/unitiger/hydro_impl/hydro_boundary_exchange.hpp:CUDA_GLOBAL_METHOD inline void complete_hydro_amr_boundary_inner_loop(const double dx, const bool energy_only,
octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp:#include "octotiger/cuda_util/cuda_global_def.hpp"
octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int faces[3][9] = {{12, 0, 3, 6, 9, 15, 18, 21, 24},
octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp:CUDA_CALLABLE_METHOD const double quad_weights[9] = {
octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int offset = 0;
octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int compressedH_DN[3] = {dimension_length * dimension_length, dimension_length, 1};
octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int dim_offset = dimension_length * dimension_length * dimension_length;
octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp:CUDA_CALLABLE_METHOD const int face_offset = 27 * dim_offset;
octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline int flip_dim(const int d, const int flip_dim) {
octotiger/unitiger/hydro_impl/flux_kernel_templates.hpp:CUDA_GLOBAL_METHOD inline simd_t cell_inner_flux_loop_simd(const double omega, const size_t nf_,
octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp:extern std::atomic<std::uint64_t> hydro_cuda_gpu_subgrids_processed;
octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp:extern std::atomic<std::uint64_t> hydro_cuda_gpu_aggregated_subgrids_launches;
octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp:extern std::atomic<std::uint64_t> hydro_kokkos_gpu_subgrids_processed;
octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp:extern std::atomic<std::uint64_t> hydro_kokkos_gpu_aggregated_subgrids_launches;
octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp:std::uint64_t hydro_cuda_gpu_subgrid_processed_performance_data(bool reset);
octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp:std::uint64_t hydro_cuda_gpu_aggregated_subgrids_launches_performance_data(bool reset);
octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp:std::uint64_t hydro_cuda_avg_aggregation_rate(bool reset);
octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp:std::uint64_t hydro_kokkos_gpu_subgrids_processed_performance_data(bool reset);
octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp:std::uint64_t hydro_kokkos_gpu_aggregated_subgrids_launches_performance_data(bool reset);
octotiger/unitiger/hydro_impl/hydro_performance_counters.hpp:std::uint64_t hydro_kokkos_gpu_avg_aggregation_rate(bool reset);
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:const storage& get_flux_device_masks(executor_t& exec2, const size_t gpu_id = 0) {
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:    /* if (agg_exec.parent.gpu_id == 1) */
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:        for (int gpu_id_loop = 0; gpu_id_loop < opts().number_gpus; gpu_id_loop++) {
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:                  round_robin_pool<executor_t>>(gpu_id_loop);
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:          masks.emplace_back(gpu_id_loop, NDIM * q_inx3);
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:          Kokkos::deep_copy(exec.instance(), masks[gpu_id_loop], tmp_masks);
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:    return masks[gpu_id];
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:        const size_t gpu_id = agg_exec.parent.gpu_id;
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:        const size_t gpu_id = agg_exec.parent.gpu_id;
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:        const size_t gpu_id = agg_exec.parent.gpu_id;
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:        const size_t gpu_id = agg_exec.parent.gpu_id;
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:        const size_t gpu_id = agg_exec.parent.gpu_id;
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:        const size_t gpu_id = agg_exec.parent.gpu_id;
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:        const size_t gpu_id = agg_exec.parent.gpu_id;
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:        const size_t gpu_id = agg_exec.parent.gpu_id;
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:              round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:            agg_exec.get_underlying_executor(), agg_exec.parent.gpu_id);
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:    octotiger::hydro::hydro_kokkos_gpu_subgrids_processed++;
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:        octotiger::hydro::hydro_kokkos_gpu_aggregated_subgrids_launches++;
octotiger/unitiger/hydro_impl/hydro_kokkos_kernel.hpp:      // Either handles the launches on the CPU or on the GPU depending on the passed executor
octotiger/unitiger/radiation/radiation_physics_impl.hpp:/*** Reconstruct uses this - GPUize****/
octotiger/unitiger/radiation/radiation_physics_impl.hpp:/*** Reconstruct uses this - GPUize****/
octotiger/unitiger/radiation/radiation_physics_impl.hpp:/*** Reconstruct uses this - GPUize****/
octotiger/unitiger/radiation/radiation_physics_impl.hpp:/*** Reconstruct uses this - GPUize****/
octotiger/unitiger/radiation/radiation_physics.hpp:	/*** Reconstruct uses this - GPUize****/
octotiger/unitiger/radiation/radiation_physics.hpp:	/*** Reconstruct uses this - GPUize****/
octotiger/unitiger/radiation/radiation_physics.hpp:	/*** Reconstruct uses this - GPUize****/
octotiger/unitiger/radiation/radiation_physics.hpp:	/*** Reconstruct uses this - GPUize****/
octotiger/unitiger/safe_real.hpp:#if defined(__CUDACC__) // cuda used? 
octotiger/unitiger/safe_real.hpp:#if __CUDA_ARCH__ == 0 // host code
octotiger/config/export_definitions.hpp:#elif defined(__NVCC__) || defined(__CUDACC__)
octotiger/options.hpp:	size_t number_gpus;
octotiger/options.hpp:	size_t executors_per_gpu;
octotiger/options.hpp:	size_t max_gpu_executor_queue_length;
octotiger/options.hpp:		arc & number_gpus;
octotiger/options.hpp:		arc & executors_per_gpu;
octotiger/options.hpp:		arc & max_gpu_executor_queue_length;
octotiger/aggregation_util.hpp:#include <cuda/std/tuple>
octotiger/aggregation_util.hpp:#if defined(HPX_CUDA_VERSION) && (HPX_CUDA_VERSION < 1202)
octotiger/aggregation_util.hpp:// cuda::std::tuple structured bindings are broken in CUDA < 1202
octotiger/aggregation_util.hpp:// See https://github.com/NVIDIA/libcudacxx/issues/316
octotiger/aggregation_util.hpp:// According to https://github.com/NVIDIA/libcudacxx/pull/317 the fix for this 
octotiger/aggregation_util.hpp:// which the following snippet does. This is only necessary for old CUDA versions
octotiger/aggregation_util.hpp:    struct tuple_element<_Ip, _CUDA_VSTD::tuple<_Tp...>> 
octotiger/aggregation_util.hpp:      : _CUDA_VSTD::tuple_element<_Ip, _CUDA_VSTD::tuple<_Tp...>> {};
octotiger/aggregation_util.hpp:    struct tuple_size<_CUDA_VSTD::tuple<_Tp...>> 
octotiger/aggregation_util.hpp:      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::tuple<_Tp...>> {};
octotiger/aggregation_util.hpp:CUDA_GLOBAL_METHOD typename Agg_view_t::view_type get_slice_subview(
octotiger/aggregation_util.hpp:CUDA_GLOBAL_METHOD auto map_views_to_slice(const Integer slice_id, const Integer max_slices,
octotiger/aggregation_util.hpp:        return cuda::std::tuple_cat(cuda::std::make_tuple(get_slice_subview(slice_id, max_slices, current_arg)),
octotiger/aggregation_util.hpp:        return cuda::std::make_tuple(get_slice_subview(slice_id, max_slices, current_arg));
octotiger/aggregation_util.hpp:CUDA_GLOBAL_METHOD auto map_views_to_slice(const Agg_executor_t& agg_exec, const Agg_view_t& current_arg,
octotiger/aggregation_util.hpp:    const size_t gpu_id = agg_exec.parent.gpu_id;
octotiger/aggregation_util.hpp:    auto launch_copy_lambda = [gpu_id](TargetView_t& target, SourceView_t& source,
octotiger/aggregation_util.hpp:              round_robin_pool<executor_t>>(gpu_id);
octotiger/aggregation_util.hpp:    const size_t gpu_id = agg_exec.parent.gpu_id;
octotiger/aggregation_util.hpp:    auto launch_copy_lambda = [gpu_id, elements_per_slice, number_slices](TargetView_t& target,
octotiger/aggregation_util.hpp:              round_robin_pool<executor_t>>(gpu_id);
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline void select_wrapper<double, bool>(
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline double max_wrapper<double>(const double& tmp1, const double& tmp2) {
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline double min_wrapper<double>(const double& tmp1, const double& tmp2) {
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline double sqrt_wrapper<double>(const double& tmp1) {
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline double pow_wrapper<double>(const double& tmp1, const double& tmp2) {
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline double copysign_wrapper<double>(const double& tmp1, const double& tmp2) {
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline double abs_wrapper<double>(const double& tmp1) {
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline double minmod_wrapper<double>(const double& a, const double& b) {
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline double minmod_theta_wrapper<double>(const double& a, const double& b, const double& c) {
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline double limiter_wrapper<double>(const double& a, const double& b) {
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline double asinh_wrapper<double>(const double& tmp1) {
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline bool skippable<double>(const double& tmp1) {
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline double load_value<double>(const double* __restrict__ data, const size_t index) {
octotiger/util/vec_scalar_host_wrapper.hpp:CUDA_GLOBAL_METHOD inline void store_value<double>(
octotiger/util/vec_base_wrapper.hpp:#include "octotiger/cuda_util/cuda_global_def.hpp"
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline void select_wrapper(
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline T max_wrapper(const T& tmp1, const T& tmp2) {
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline T min_wrapper(const T& tmp1, const T& tmp2) {
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline T sqrt_wrapper(const T& tmp1) {
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline T pow_wrapper(const T& tmp1, const double& tmp2) {
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline T asinh_wrapper(const T& tmp1) {
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline T copysign_wrapper(const T& tmp1, const T& tmp2) {
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline T abs_wrapper(const T& tmp1) {
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline T minmod_wrapper(const T& a, const T& b) {
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline T minmod_theta_wrapper(const T& a, const T& b, const T& c) {
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline T limiter_wrapper(const T& a, const T& b) {
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline bool skippable(const T& tmp1) {
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline T load_value(const double* __restrict__ data, const size_t index) {
octotiger/util/vec_base_wrapper.hpp:CUDA_GLOBAL_METHOD inline void store_value(
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline void select_wrapper<vc_type, mask_type>(
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline vc_type max_wrapper<vc_type>(const vc_type& tmp1, const vc_type& tmp2) {
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline vc_type min_wrapper<vc_type>(const vc_type& tmp1, const vc_type& tmp2) {
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline vc_type sqrt_wrapper<vc_type>(const vc_type& tmp1) {
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline vc_type asinh_wrapper<vc_type>(const vc_type& tmp1) {
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline vc_type copysign_wrapper<vc_type>(const vc_type& tmp1, const vc_type& tmp2) {
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline vc_type abs_wrapper<vc_type>(const vc_type& tmp1) {
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline vc_type minmod_wrapper<vc_type>(const vc_type& a, const vc_type& b) {
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline vc_type minmod_theta_wrapper<vc_type>(const vc_type& a, const vc_type& b, const vc_type& c) {
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline vc_type limiter_wrapper<vc_type>(const vc_type& a, const vc_type& b) {
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline vc_type load_value<vc_type>(const double* __restrict__ data, const size_t index) {
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline void store_value<vc_type>(
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline bool skippable<mask_type>(const mask_type& tmp1) {
octotiger/util/vec_vc_wrapper.hpp:CUDA_GLOBAL_METHOD inline vc_type pow_wrapper<vc_type>(const vc_type& tmp1, const double& tmp2) {
octotiger/common_kernel/kokkos_simd.hpp:#include "octotiger/cuda_util/cuda_global_def.hpp"
octotiger/common_kernel/kokkos_simd.hpp:#ifdef __CUDACC__ // hence: Use scalar when using nvcc
octotiger/common_kernel/kokkos_simd.hpp:// TODO Actually test with a non-cuda kokkos build and/or clang
octotiger/common_kernel/kokkos_simd.hpp:CUDA_GLOBAL_METHOD inline simd_t sqrt_with_serial_fallback(const simd_t input) {
octotiger/common_kernel/kokkos_simd.hpp:CUDA_GLOBAL_METHOD inline simd_t pow_with_serial_fallback(const simd_t input, const double exponent) {
octotiger/common_kernel/kokkos_simd.hpp:CUDA_GLOBAL_METHOD inline simd_t asinh_with_serial_fallback(const simd_t input) {
octotiger/common_kernel/kokkos_simd.hpp:CUDA_GLOBAL_METHOD inline simd_t copysign_with_serial_fallback(const simd_t input1, const simd_t input2) {
octotiger/common_kernel/kokkos_simd.hpp:CUDA_GLOBAL_METHOD inline simd_t abs(const simd_t input1) {
octotiger/common_kernel/gravity_performance_counters.hpp:extern std::atomic<std::uint64_t> p2p_kokkos_gpu_subgrids_launched;
octotiger/common_kernel/gravity_performance_counters.hpp:extern std::atomic<std::uint64_t> p2p_cuda_gpu_subgrids_launched;
octotiger/common_kernel/gravity_performance_counters.hpp:extern std::atomic<std::uint64_t> multipole_kokkos_gpu_subgrids_launched;
octotiger/common_kernel/gravity_performance_counters.hpp:extern std::atomic<std::uint64_t> multipole_cuda_gpu_subgrids_launched;
octotiger/common_kernel/gravity_performance_counters.hpp:std::uint64_t p2p_kokkos_gpu_subgrid_processed_performance_data(bool reset);
octotiger/common_kernel/gravity_performance_counters.hpp:std::uint64_t p2p_cuda_gpu_subgrid_processed_performance_data(bool reset);
octotiger/common_kernel/gravity_performance_counters.hpp:std::uint64_t multipole_kokkos_gpu_subgrid_processed_performance_data(bool reset);
octotiger/common_kernel/gravity_performance_counters.hpp:std::uint64_t multipole_cuda_gpu_subgrid_processed_performance_data(bool reset);
octotiger/common_kernel/kokkos_util.hpp:#if defined(KOKKOS_ENABLE_CUDA)
octotiger/common_kernel/kokkos_util.hpp:using kokkos_device_executor = hpx::kokkos::cuda_executor;
octotiger/common_kernel/kokkos_util.hpp:#ifdef KOKKOS_ENABLE_CUDA 
octotiger/common_kernel/kokkos_util.hpp:        std::is_same<hpx::kokkos::cuda_executor, typename std::remove_cv<T>::type>::value>
octotiger/common_kernel/kokkos_util.hpp:#ifdef KOKKOS_ENABLE_CUDA
octotiger/common_kernel/kokkos_util.hpp:#include <cuda_buffer_util.hpp>
octotiger/common_kernel/kokkos_util.hpp:#if defined(KOKKOS_ENABLE_CUDA)
octotiger/common_kernel/kokkos_util.hpp:using kokkos_um_device_array = Kokkos::View<T*, Kokkos::CudaSpace, Kokkos::MemoryUnmanaged>;
octotiger/common_kernel/kokkos_util.hpp:using kokkos_device_array = Kokkos::View<T*, Kokkos::CudaSpace>;
octotiger/common_kernel/kokkos_util.hpp:    recycler::recycle_allocator_cuda_device<T>, T>;
octotiger/common_kernel/kokkos_util.hpp:// Fallback without cuda
octotiger/common_kernel/kokkos_util.hpp:// NOTE: Must use the same layout to be able to use e.g. cudaMemcpyAsync
octotiger/common_kernel/kokkos_util.hpp:// NOTE: Must use the same layout to be able to use e.g. cudaMemcpyAsync
octotiger/common_kernel/kokkos_util.hpp:#if defined(KOKKOS_ENABLE_CUDA)
octotiger/common_kernel/kokkos_util.hpp:using kokkos_host_allocator = recycler::detail::cuda_pinned_allocator<T>;
octotiger/common_kernel/kokkos_util.hpp:using kokkos_device_allocator = recycler::detail::cuda_device_allocator<T>;
octotiger/common_kernel/kokkos_util.hpp:    Kokkos::CudaHostPinnedSpace, Kokkos::MemoryUnmanaged>;
octotiger/common_kernel/kokkos_util.hpp:    recycler::recycled_view<kokkos_um_pinned_array<T>, recycler::recycle_allocator_cuda_host<T>, T>;
octotiger/common_kernel/struct_of_array_data.hpp:#ifdef OCTOTIGER_HAVE_CUDA
octotiger/common_kernel/struct_of_array_data.hpp:#include "../cuda_util/cuda_helper.hpp"
octotiger/common_kernel/struct_of_array_data.hpp:#include <cuda_buffer_util.hpp>
octotiger/common_kernel/struct_of_array_data.hpp:#include "../cuda_util/cuda_helper.hpp"
octotiger/common_kernel/struct_of_array_data.hpp:#ifdef OCTOTIGER_HAVE_CUDA
octotiger/common_kernel/struct_of_array_data.hpp:    using cuda_expansion_buffer_t = struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING,
octotiger/common_kernel/struct_of_array_data.hpp:        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>;
octotiger/common_kernel/struct_of_array_data.hpp:    using cuda_space_vector_buffer_t =
octotiger/common_kernel/struct_of_array_data.hpp:            std::vector<real, recycler::recycle_allocator_cuda_host<real>>>;
octotiger/common_kernel/struct_of_array_data.hpp:    using cuda_expansion_result_buffer_t =
octotiger/common_kernel/struct_of_array_data.hpp:            std::vector<real, recycler::recycle_allocator_cuda_host<real>>>;
octotiger/common_kernel/struct_of_array_data.hpp:    using cuda_angular_result_t =
octotiger/common_kernel/struct_of_array_data.hpp:            std::vector<real, recycler::recycle_allocator_cuda_host<real>>>;
octotiger/common_kernel/struct_of_array_data.hpp:    using cuda_monopole_buffer_t =
octotiger/common_kernel/struct_of_array_data.hpp:        std::vector<real, recycler::recycle_allocator_cuda_host<real>>;
octotiger/common_kernel/struct_of_array_data.hpp:    using cuda_expansion_buffer_t = struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING,
octotiger/common_kernel/struct_of_array_data.hpp:    using cuda_space_vector_buffer_t =
octotiger/common_kernel/struct_of_array_data.hpp:    using cuda_expansion_result_buffer_t =
octotiger/common_kernel/struct_of_array_data.hpp:    using cuda_angular_result_t =
octotiger/common_kernel/struct_of_array_data.hpp:    using cuda_monopole_buffer_t =
octotiger/common_kernel/multiindex.hpp:#include "octotiger/cuda_util/cuda_global_def.hpp"
octotiger/common_kernel/multiindex.hpp:        CUDA_GLOBAL_METHOD multiindex(T x, T y, T z)
octotiger/common_kernel/multiindex.hpp:        CUDA_GLOBAL_METHOD explicit multiindex(const multiindex<U>& other) {
octotiger/common_kernel/multiindex.hpp:        CUDA_GLOBAL_METHOD multiindex() {
octotiger/common_kernel/multiindex.hpp:            constructor, we cannot use the class within cuda constant memory (and we want to) */
octotiger/common_kernel/multiindex.hpp:        CUDA_GLOBAL_METHOD inline double length() const {
octotiger/common_kernel/multiindex.hpp:        CUDA_GLOBAL_METHOD inline bool compare(multiindex& other) {
octotiger/common_kernel/multiindex.hpp:        CUDA_GLOBAL_METHOD inline bool operator == (const multiindex& other) const {
octotiger/common_kernel/multiindex.hpp:        CUDA_GLOBAL_METHOD void transform_coarse() {
octotiger/common_kernel/multiindex.hpp:    CUDA_GLOBAL_METHOD inline multiindex<> flat_index_to_multiindex_not_padded(
octotiger/common_kernel/multiindex.hpp:    CUDA_GLOBAL_METHOD inline multiindex<> flat_index_to_multiindex_padded(size_t flat_index) {
octotiger/common_kernel/multiindex.hpp:    CUDA_GLOBAL_METHOD inline T to_flat_index_padded(const multiindex<T>& m) {
octotiger/common_kernel/multiindex.hpp:    CUDA_GLOBAL_METHOD inline T to_inner_flat_index_not_padded(const multiindex<T>& m) {
octotiger/common_kernel/multiindex.hpp:    // This specialization is only required on cuda devices since T::value_type is not supported!
octotiger/common_kernel/multiindex.hpp:    CUDA_GLOBAL_METHOD inline void multiindex<int32_t>::transform_coarse() {
octotiger/common_kernel/multiindex.hpp:    CUDA_GLOBAL_METHOD inline int32_t distance_squared_reciprocal(
octotiger/interaction_types.hpp:COMMAND_LINE_ENUM(interaction_device_kernel_type,OFF,CUDA,HIP,KOKKOS_CUDA,KOKKOS_HIP,KOKKOS_SYCL);
octotiger/interaction_types.hpp:COMMAND_LINE_ENUM(amr_boundary_type,AMR_LEGACY,AMR_OPTIMIZED,AMR_CUDA);
octotiger/radiation/kernel_interface.hpp:#include "octotiger/radiation/cuda_kernel.hpp"
octotiger/radiation/cuda_kernel.hpp:#if OCTOTIGER_HAVE_CUDA && !defined(RADIATION_CUDA_KERNEL_HPP_)
octotiger/radiation/cuda_kernel.hpp:#define RADIATION_CUDA_KERNEL_HPP_
octotiger/radiation/cuda_kernel.hpp:    void radiation_cuda_kernel(integer const d, std::vector<real> const& rho,
octotiger/radiation/cuda_kernel.hpp:#endif    // RADIATION_CUDA_KERNEL_HPP_
octotiger/defs.hpp:#   if defined(__NVCC__) || defined(__CUDACC__)
octotiger/monopole_interactions/kernel/monopole_kernel_templates.hpp:#include "octotiger/cuda_util/cuda_global_def.hpp"
octotiger/monopole_interactions/kernel/monopole_kernel_templates.hpp:        CUDA_GLOBAL_METHOD inline void compute_monopole_interaction(const T& monopole,
octotiger/monopole_interactions/kernel/monopole_kernel_templates.hpp:        CUDA_GLOBAL_METHOD inline void compute_interaction_p2m_non_rho(
octotiger/monopole_interactions/kernel/monopole_kernel_templates.hpp:        CUDA_GLOBAL_METHOD inline void compute_interaction_p2m_rho(const T& d2, const T& d3,
octotiger/monopole_interactions/kernel/monopole_kernel_templates.hpp:        CUDA_GLOBAL_METHOD inline void compute_kernel_p2m_non_rho(
octotiger/monopole_interactions/kernel/monopole_kernel_templates.hpp:        CUDA_GLOBAL_METHOD inline void compute_kernel_p2m_rho(T (&X)[NDIM], T (&Y)[NDIM],
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:        const storage& get_device_masks(executor_t& exec2, const size_t gpu_id = 0) {
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                for (int gpu_id_loop = 0; gpu_id_loop < opts().number_gpus; gpu_id_loop++) {
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                          round_robin_pool<executor_t>>(gpu_id_loop);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                    stencil_masks.emplace_back(gpu_id_loop, FULL_STENCIL_SIZE);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                    Kokkos::deep_copy(exec.instance(), stencil_masks[gpu_id_loop], tmp);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            return stencil_masks[gpu_id];
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:        const storage& get_device_constants(executor_t& exec2, const size_t gpu_id = 0) {
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                for (int gpu_id_loop = 0; gpu_id_loop < opts().number_gpus; gpu_id_loop++) {
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                          round_robin_pool<executor_t>>(gpu_id_loop);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                    stencil_constants.emplace_back(gpu_id_loop, 4 * FULL_STENCIL_SIZE);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                    Kokkos::deep_copy(exec.instance(), stencil_constants[gpu_id_loop], tmp);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            return stencil_constants[gpu_id];
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            //   [monopoles, potential_expansions, devicemasks, dx, theta] CUDA_GLOBAL_METHOD(
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                    // Loop gets executed once on GPU (as we have multiple blocks to replace it)
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                const int gpu_id = agg_exec.parent.gpu_id;
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                    round_robin_pool<hpx::kokkos::executor<kokkos_backend_t>>>(gpu_id);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                //   [monopoles, potential_expansions, devicemasks, dx, theta] CUDA_GLOBAL_METHOD(
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                        // Loop gets executed once on GPU (as we have multiple blocks to replace it)
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            //   [monopoles, potential_expansions, devicemasks, dx, theta] CUDA_GLOBAL_METHOD(
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            const int gpu_id = agg_exec.parent.gpu_id;
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                    agg_exec.get_underlying_executor(), gpu_id);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                    agg_exec.get_underlying_executor(), gpu_id);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            auto gpu_id = device_id;
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                  round_robin_pool<executor_t>>(gpu_id);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                get_device_masks<device_buffer<int>, host_buffer<int>, executor_t>(exec, gpu_id);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                get_device_constants<device_buffer<double>, host_buffer<double>, executor_t>(exec, gpu_id);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> device_monopoles(gpu_id, NUMBER_LOCAL_MONOPOLE_VALUES);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> device_tmp_results(gpu_id, 
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> device_results(gpu_id, NUMBER_POT_EXPANSIONS_SMALL);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            auto gpu_id = device_id;
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                  round_robin_pool<executor_t>>(gpu_id);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                get_device_masks<device_buffer<int>, host_buffer<int>, executor_t>(exec, gpu_id);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                get_device_constants<device_buffer<double>, host_buffer<double>, executor_t>(exec, gpu_id);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> device_monopoles(gpu_id, NUMBER_LOCAL_MONOPOLE_VALUES);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> device_tmp_results(gpu_id, 
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> device_results(gpu_id, NUMBER_POT_EXPANSIONS_SMALL);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> device_center_of_masses_inner_cells(gpu_id, 
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> device_corrections(gpu_id, NUMBER_ANG_CORRECTIONS);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                    device_local_expansions_neighbors.emplace_back(gpu_id, (size + SOA_PADDING) * 20);
octotiger/monopole_interactions/kernel/kokkos_kernel.hpp:                    device_center_of_masses_neighbors.emplace_back(gpu_id, (size + SOA_PADDING) * 3);
octotiger/monopole_interactions/legacy/monopole_interaction_interface.hpp:            static OCTOTIGER_EXPORT size_t& cuda_launch_counter();
octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp:// will be used as fallback in non-cuda compilations
octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp:#include "octotiger/cuda_util/cuda_helper.hpp"
octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp:#include "octotiger/cuda_util/cuda_scheduler.hpp"
octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp:        /** Interface to calculate monopole monopole FMM interactions on either a cuda device or on
octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp:         * the cpu! It takes AoS data, transforms it into SoA data, moves it to the cuda device,
octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp:         * launches cuda kernel on a slot given by the scheduler, gets results and stores them in
octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp:         * the AoS arrays L and L_c. Load balancing between CPU and GPU is done by the scheduler
octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp:         * ../cuda_util/cuda_scheduler.hpp). If the scheduler detects that the cuda device is
octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp:        class cuda_monopole_interaction_interface : public monopole_interaction_interface
octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp:            cuda_monopole_interaction_interface();
octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp:            //     std::vector<real, cuda_pinned_allocator<real>>>
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:#include "octotiger/cuda_util/cuda_helper.hpp"
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:        __host__ void init_stencil(size_t gpu_id, std::unique_ptr<bool[]> stencil_masks,
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:        /*__global__ void cuda_p2p_interactions_kernel(
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:        __global__ void cuda_p2m_interaction_rho(
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:        __global__ void cuda_p2m_interaction_non_rho(
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:#ifdef OCTOTIGER_HAVE_CUDA
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:        void launch_p2p_cuda_kernel_post(
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:        void launch_p2m_rho_cuda_kernel_post(
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:        void launch_p2m_non_rho_cuda_kernel_post(
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/cuda_util/cuda_scheduler.hpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
octotiger/cuda_util/cuda_scheduler.hpp:#include "octotiger/cuda_util/cuda_helper.hpp"
octotiger/cuda_util/cuda_scheduler.hpp:    // Define sizes of CUDA buffers
octotiger/cuda_util/cuda_helper.hpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
octotiger/cuda_util/cuda_helper.hpp:#define CUDA_API_PER_THREAD_DEFAULT_STREAM
octotiger/cuda_util/cuda_helper.hpp:#include <hpx/async_cuda/cuda_executor.hpp>
octotiger/cuda_util/cuda_helper.hpp:#if defined(OCTOTIGER_HAVE_CUDA)
octotiger/cuda_util/cuda_helper.hpp:#include <cuda_runtime.h>
octotiger/cuda_util/cuda_helper.hpp://using pool_strategy = multi_gpu_round_robin_pool<hpx::cuda::experimental::cuda_executor, round_robin_pool<hpx::cuda::experimental::cuda_executor>>;
octotiger/cuda_util/cuda_helper.hpp:using pool_strategy = round_robin_pool<hpx::cuda::experimental::cuda_executor>;
octotiger/cuda_util/cuda_helper.hpp:using kokkos_strategy = round_robin_pool<hpx::cuda::experimental::cuda_executor>;
octotiger/cuda_util/cuda_global_def.hpp:#if defined(OCTOTIGER_HAVE_CUDA)
octotiger/cuda_util/cuda_global_def.hpp:#if defined(__CUDACC__)
octotiger/cuda_util/cuda_global_def.hpp:#if !defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
octotiger/cuda_util/cuda_global_def.hpp:#define CUDA_API_PER_THREAD_DEFAULT_STREAM
octotiger/cuda_util/cuda_global_def.hpp:#define CUDA_CALLABLE_METHOD __device__
octotiger/cuda_util/cuda_global_def.hpp:#define CUDA_GLOBAL_METHOD __host__ __device__
octotiger/cuda_util/cuda_global_def.hpp:#define CUDA_CALLABLE_METHOD
octotiger/cuda_util/cuda_global_def.hpp:#define CUDA_GLOBAL_METHOD
octotiger/cuda_util/cuda_global_def.hpp:#define CUDA_CALLABLE_METHOD __device__
octotiger/cuda_util/cuda_global_def.hpp:#define CUDA_GLOBAL_METHOD __host__ __device__
octotiger/cuda_util/cuda_global_def.hpp:#define CUDA_CALLABLE_METHOD
octotiger/cuda_util/cuda_global_def.hpp:#define CUDA_GLOBAL_METHOD
octotiger/sycl_initialization_guard.hpp:  // We encounter segfaults on Intel GPUs when running the normal kernels for the first time after
octotiger/sycl_initialization_guard.hpp:  // (presumably initializes something within the intel gpu runtime).
octotiger/sycl_initialization_guard.hpp:  // Somewhat of an ugly workaround but it does the trick and allows us to target Intel GPUs as
octotiger/sycl_initialization_guard.hpp:  /// Utility function working around segfault on Intel GPU. Initializes something within the runtime by runnning
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:#include "octotiger/cuda_util/cuda_global_def.hpp"
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:// Workaround to use sycl::sqrt with double types on GPU
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:#if defined(__clang__) // Clang can handle cuda device constexpr better
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:        CUDA_CALLABLE_METHOD const double factor[20] = {1.000000, 1.000000, 1.000000, 1.000000,
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:        CUDA_CALLABLE_METHOD const double factor_half[20] = {1.000000 / 2.0, 1.000000 / 2.0, 1.000000 / 2.0, 1.000000 /2.0 ,
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:        CUDA_CALLABLE_METHOD const double factor_sixth[20] = {1.000000 / 6.0, 1.000000 / 6.0, 1.000000 / 6.0, 1.000000 / 6.0,
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:        CUDA_CALLABLE_METHOD const double factor_half[20] = {factor[0] / 2.0, factor[1] / 2.0,
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:        CUDA_CALLABLE_METHOD const double factor_sixth[20] = {factor[0] / 6.0, factor[1] / 6.0,
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:        CUDA_GLOBAL_METHOD const inline T sqr(T const& val) noexcept {
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:        CUDA_GLOBAL_METHOD inline void compute_d_factors(T& d2, T& d3, T& X_00, T& X_11, T& X_22,
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:            // Workaround to use sycl::sqrt with double types on GPU
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:        CUDA_GLOBAL_METHOD inline void compute_interaction_multipole_non_rho(
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:        CUDA_GLOBAL_METHOD inline void compute_interaction_multipole_rho(const T& d2, const T& d3,
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:        CUDA_GLOBAL_METHOD inline void compute_kernel_rho(T (&X)[NDIM], T (&Y)[NDIM],
octotiger/multipole_interactions/kernel/compute_kernel_templates.hpp:        CUDA_GLOBAL_METHOD inline void compute_kernel_non_rho(T (&X)[NDIM], T (&Y)[NDIM],
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:        const storage& get_device_masks(executor_t& exec2, bool indicators, const size_t gpu_id = 0) {
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:                for (int gpu_id_loop = 0; gpu_id_loop < opts().number_gpus; gpu_id_loop++) {
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:                          round_robin_pool<executor_t>>(gpu_id_loop);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:                    stencil_masks.emplace_back(gpu_id_loop, FULL_STENCIL_SIZE);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:                    Kokkos::deep_copy(exec.instance(), stencil_masks[gpu_id_loop], tmp_masks);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:                    stencil_indicators.emplace_back(gpu_id_loop, FULL_STENCIL_SIZE);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:                    Kokkos::deep_copy(exec.instance(), stencil_indicators[gpu_id_loop], tmp_indicators);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:                return stencil_indicators[gpu_id];
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:                return stencil_masks[gpu_id];
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:            auto gpu_id = device_id;
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:                  round_robin_pool<executor_t>>(gpu_id);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:                get_device_masks<device_buffer<int>, host_buffer<int>, executor_t>(exec, false, gpu_id);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:                get_device_masks<device_buffer<int>, host_buffer<int>, executor_t>(exec, true, gpu_id);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> device_monopoles(gpu_id, NUMBER_LOCAL_MONOPOLE_VALUES);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> device_multipoles(gpu_id, NUMBER_LOCAL_EXPANSION_VALUES);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> device_centers(gpu_id, NUMBER_MASS_VALUES);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> device_expansions(gpu_id, NUMBER_POT_EXPANSIONS);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> device_corrections(gpu_id, NUMBER_ANG_CORRECTIONS);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> tmp_device_corrections(gpu_id, number_blocks * NUMBER_ANG_CORRECTIONS);
octotiger/multipole_interactions/kernel/kokkos_kernel.hpp:            device_buffer<double> tmp_device_expansions(gpu_id, 
octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp:#include "octotiger/multipole_interactions/legacy/multipole_interaction_interface.hpp"    // will be used as fallback in non-cuda compilations
octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp:#include "octotiger/cuda_util/cuda_helper.hpp"
octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp:#include "octotiger/cuda_util/cuda_scheduler.hpp"
octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp:        /** Interface to calculate multipole FMM interactions on either a cuda device or on the
octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp:         * cpu! It takes AoS data, transforms it into SoA data, moves it to the cuda device,
octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp:         * launches cuda kernel on a slot given by the scheduler, gets results and stores them in
octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp:         * the AoS arrays L and L_c. Load balancing between CPU and GPU is done by the scheduler (see
octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp:         * ../cuda_util/cuda_scheduler.hpp). If the scheduler detects that the cuda device is
octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp:        class cuda_multipole_interaction_interface : public multipole_interaction_interface
octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp:            cuda_multipole_interaction_interface();
octotiger/multipole_interactions/legacy/multipole_interaction_interface.hpp:            static OCTOTIGER_EXPORT size_t& cuda_launch_counter();
octotiger/multipole_interactions/legacy/multipole_interaction_interface.hpp:            static OCTOTIGER_EXPORT size_t& cuda_launch_counter_non_rho();
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:#include "octotiger/cuda_util/cuda_helper.hpp"
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:        __host__ void init_stencil(size_t gpu_id, std::unique_ptr<bool[]> multipole_stencil_masks,
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:        /*__global__ void cuda_multipole_interactions_kernel_rho(
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:        __global__ void cuda_multipole_interactions_kernel_root_rho(
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:        __global__ void cuda_multipole_interactions_kernel_non_rho(
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:        __global__ void cuda_multipole_interactions_kernel_root_non_rho(
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:#if defined(OCTOTIGER_HAVE_CUDA)
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:        void launch_multipole_rho_cuda_kernel_post(
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:        void launch_multipole_non_rho_cuda_kernel_post(
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:        void launch_multipole_root_rho_cuda_kernel_post(
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:        void launch_multipole_root_non_rho_cuda_kernel_post(
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
.gitignore:cmake-cuda-*
test_problems/sod/CMakeLists.txt:if(OCTOTIGER_WITH_CUDA)
test_problems/sod/CMakeLists.txt:  test_sod_scenario(test_problems.gpu.am_hydro_on.sod_cuda sod_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} OFF
test_problems/sod/CMakeLists.txt:  "--correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/sod/CMakeLists.txt:	test_sod_scenario(test_problems.gpu.am_hydro_off.sod_hip sod_hip_log.txt ${ini_scenario_filename} ${silo_reference_filename} OFF
test_problems/sod/CMakeLists.txt:    "--correct_am_hydro=off --number_gpus=1 --executors_per_gpu=8 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=HIP --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/sod/CMakeLists.txt:  if(OCTOTIGER_WITH_CUDA)
test_problems/sod/CMakeLists.txt:    test_sod_scenario(test_problems.gpu.am_hydro_on.sod_kokkos sod_kokkos_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} OFF
test_problems/sod/CMakeLists.txt:    "--correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/sod/CMakeLists.txt:	  test_sod_scenario(test_problems.gpu.am_hydro_off.sod_kokkos sod_kokkos_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} OFF
test_problems/sod/CMakeLists.txt:      "--correct_am_hydro=off --number_gpus=1 --executors_per_gpu=8 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=KOKKOS_HIP --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/sod/CMakeLists.txt:if(OCTOTIGER_WITH_CUDA)
test_problems/sod/CMakeLists.txt:  test_sod_scenario(test_problems.gpu.am_hydro_off.sod_cuda sod_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
test_problems/sod/CMakeLists.txt:  "--correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/sod/CMakeLists.txt:  if(OCTOTIGER_WITH_CUDA)
test_problems/sod/CMakeLists.txt:    test_sod_scenario(test_problems.gpu.am_hydro_off.sod_kokkos_cuda sod_kokkos_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
test_problems/sod/CMakeLists.txt:    "--correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/sod/CMakeLists.txt:    test_sod_scenario(test_problems.gpu.am_hydro_off.sod_kokkos_sycl sod_kokkos_sycl_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
test_problems/sod/CMakeLists.txt:      "--correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=KOKKOS_SYCL --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/sod/CMakeLists.txt:  if(OCTOTIGER_WITH_CUDA)
test_problems/sod/CMakeLists.txt:    test_sod_scenario(test_problems.gpu.am_hydro_off.sod_big_cuda sod_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
test_problems/sod/CMakeLists.txt:    "--correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/sod/CMakeLists.txt:      # Test combined cpu + gpu execution: Only makes sense when kernels use the same FP_Contract (otherwise there'll be an ~ 1e-20 error due to interactions between kernels executed on different devices)
test_problems/sod/CMakeLists.txt:    #   test_sod_scenario(test_problems.cpu_gpu.am_hydro_off.sod_big_cuda sod_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
test_problems/sod/CMakeLists.txt:    #   "--correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=4 --max_gpu_executor_queue_length=1 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=LEGACY")
test_problems/sod/CMakeLists.txt:    if(OCTOTIGER_WITH_CUDA)
test_problems/sod/CMakeLists.txt:      test_sod_scenario(test_problems.gpu.am_hydro_off.sod_big_kokkos_cuda sod_kokkos_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
test_problems/sod/CMakeLists.txt:      "--correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/sod/CMakeLists.txt:      # Test combined cpu + gpu execution: Only makes sense when kernels use the same FP_Contract (otherwise there'll be an ~ 1e-20 error due to interactions between kernels executed on different devices)
test_problems/sod/CMakeLists.txt:      #   test_sod_scenario(test_problems.cpu_gpu.am_hydro_off.sod_big_kokkos sod_kokkos_cuda_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
test_problems/sod/CMakeLists.txt:      #   "--correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=4 --max_gpu_executor_queue_length=1 --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=KOKKOS")
test_problems/sod/CMakeLists.txt:      test_sod_scenario(test_problems.gpu.am_hydro_off.sod_big_kokkos_sycl sod_kokkos_sycl_log.txt ${ini_scenario_filename} ${silo_reference_filename} ON
test_problems/sod/CMakeLists.txt:        "--correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=KOKKOS_SYCL --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/blast/CMakeLists.txt:if(OCTOTIGER_WITH_CUDA)
test_problems/blast/CMakeLists.txt:  test_blast_scenario(test_problems.gpu.am_hydro_on.blast_cuda blast_with_am_hydro_cuda_log.txt ${silo_scenario_filename}
test_problems/blast/CMakeLists.txt:    "  --correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/blast/CMakeLists.txt:#   test_blast_scenario(test_problems.cpu_gpu.am_hydro_on.blast_cuda blast_without_am_hydro_cuda_log.txt ${silo_scenario_filename}
test_problems/blast/CMakeLists.txt:#     " --correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=LEGACY")
test_problems/blast/CMakeLists.txt:  test_blast_scenario(test_problems.gpu.am_hydro_off.blast_hip blast_without_am_hydro_hip_log.txt ${silo_scenario_filename}
test_problems/blast/CMakeLists.txt:  " --correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=HIP --hydro_host_kernel_type=LEGACY")
test_problems/blast/CMakeLists.txt:  if(OCTOTIGER_WITH_CUDA)
test_problems/blast/CMakeLists.txt:    test_blast_scenario(test_problems.gpu.am_hydro_on.blast_kokkos blast_without_am_hydro_kokkos_cuda_log.txt ${silo_scenario_filename}
test_problems/blast/CMakeLists.txt:    " --correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/blast/CMakeLists.txt:  #   test_blast_scenario(test_problems.cpu_gpu.am_hydro_on.blast_kokkos blast_with_am_hydro_kokkos_cuda_log.txt ${silo_scenario_filename}
test_problems/blast/CMakeLists.txt:  #   "  --correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1 --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=KOKKOS")
test_problems/blast/CMakeLists.txt:    test_blast_scenario(test_problems.gpu.am_hydro_on.blast_kokkos blast_with_am_hydro_kokkos_hip_log.txt ${silo_scenario_filename}
test_problems/blast/CMakeLists.txt:    "  --correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=KOKKOS_HIP --hydro_host_kernel_type=KOKKOS")
test_problems/blast/CMakeLists.txt:if(OCTOTIGER_WITH_CUDA)
test_problems/blast/CMakeLists.txt:  test_blast_scenario(test_problems.gpu.am_hydro_off.blast_cuda blast_without_am_hydro_cuda_log.txt ${silo_scenario_filename}
test_problems/blast/CMakeLists.txt:  " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/blast/CMakeLists.txt:#   test_blast_scenario(test_problems.cpu_gpu.am_hydro_off.blast_cuda blast_without_am_hydro_cuda_log.txt ${silo_scenario_filename}
test_problems/blast/CMakeLists.txt:#   " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1 --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=LEGACY")
test_problems/blast/CMakeLists.txt:  test_blast_scenario(test_problems.gpu.am_hydro_off.blast_hip blast_without_am_hydro_hip_log.txt ${silo_scenario_filename}
test_problems/blast/CMakeLists.txt:  " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=HIP --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/blast/CMakeLists.txt:  if(OCTOTIGER_WITH_CUDA)
test_problems/blast/CMakeLists.txt:    # test_blast_scenario(test_problems.cpu_gpu.am_hydro_off.blast_kokkos blast_without_am_hydro_kokkos_cuda_log.txt ${silo_scenario_filename}
test_problems/blast/CMakeLists.txt:    # " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1 --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=KOKKOS")
test_problems/blast/CMakeLists.txt:    test_blast_scenario(test_problems.gpu.am_hydro_off.blast_kokkos blast_without_am_hydro_kokkos_cuda_log.txt ${silo_scenario_filename}
test_problems/blast/CMakeLists.txt:    " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/blast/CMakeLists.txt:    test_blast_scenario(test_problems.gpu.am_hydro_off.blast_kokkos blast_without_am_hydro_kokkos_hip_log.txt ${silo_scenario_filename}
test_problems/blast/CMakeLists.txt:   " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --hydro_device_kernel_type=KOKKOS_HIP --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/star/CMakeLists.txt:# TODO Add CUDA/HIP/KOKKOS tests as the kernels get ported...
test_problems/sphere/CMakeLists.txt:# Sphere - GPU plain CUDA
test_problems/sphere/CMakeLists.txt:if(OCTOTIGER_WITH_CUDA)
test_problems/sphere/CMakeLists.txt:  test_sphere_scenario(test_problems.gpu.sphere_cuda  sphere_cuda_log.txt ${silo_scenario_filename}
test_problems/sphere/CMakeLists.txt:    " --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA ")
test_problems/sphere/CMakeLists.txt:# Sphere - GPU plain HIP
test_problems/sphere/CMakeLists.txt:  test_sphere_scenario(test_problems.gpu.sphere_hip  sphere_hip_log.txt ${silo_scenario_filename}
test_problems/sphere/CMakeLists.txt:    " --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=HIP --multipole_device_kernel_type=HIP ")
test_problems/sphere/CMakeLists.txt:  if(OCTOTIGER_WITH_CUDA)
test_problems/sphere/CMakeLists.txt:    test_sphere_scenario(test_problems.gpu.sphere_kokkos_cuda  sphere_kokkos_cuda_log.txt ${silo_scenario_filename}
test_problems/sphere/CMakeLists.txt:      " --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA ")
test_problems/sphere/CMakeLists.txt:    test_sphere_scenario(test_problems.gpu.sphere_kokkos_hip  sphere_kokkos_hip_log.txt ${silo_scenario_filename}
test_problems/sphere/CMakeLists.txt:      " --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_HIP --multipole_device_kernel_type=KOKKOS_HIP ")
test_problems/sphere/CMakeLists.txt:    test_sphere_scenario(test_problems.gpu.sphere_kokkos_sycl  sphere_kokkos_sycl_log.txt ${silo_scenario_filename}
test_problems/sphere/CMakeLists.txt:      " --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_SYCL --multipole_device_kernel_type=KOKKOS_SYCL ")
test_problems/sphere/CMakeLists.txt:# TODO CPU+GPU tests?
test_problems/rotating_star/CMakeLists.txt:if(OCTOTIGER_WITH_CUDA)
test_problems/rotating_star/CMakeLists.txt:  test_rotating_star_scenario(test_problems.gpu.am_hydro_on.rotating_star_cuda rotating_star_am_hydro_on_cuda_log.txt ${silo_scenario_filename} 
test_problems/rotating_star/CMakeLists.txt:  "  --correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:    # combined cpu+gpu
test_problems/rotating_star/CMakeLists.txt:    test_rotating_star_scenario(test_problems.cpu_gpu.am_hydro_on.rotating_star_cuda rotating_star_am_hydro_on_cuda_log.txt ${silo_scenario_filename} 
test_problems/rotating_star/CMakeLists.txt:      "  --correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1 --monopole_host_kernel_type=VC --multipole_host_kernel_type=VC --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:  test_rotating_star_scenario(test_problems.gpu.am_hydro_on.rotating_star_hip rotating_star_am_hydro_on_hip_log.txt ${silo_scenario_filename} 
test_problems/rotating_star/CMakeLists.txt:    "  --correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=16 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=HIP --multipole_device_kernel_type=HIP --hydro_device_kernel_type=HIP --hydro_host_kernel_type=DEVICE_ONLY ")
test_problems/rotating_star/CMakeLists.txt:  if(OCTOTIGER_WITH_CUDA)
test_problems/rotating_star/CMakeLists.txt:    test_rotating_star_scenario(test_problems.gpu.am_hydro_on.rotating_star_kokkos rotating_star_am_hydro_on_kokkos_cuda_log.txt ${silo_scenario_filename} 
test_problems/rotating_star/CMakeLists.txt:    "  --correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:      # combined cpu+gpu
test_problems/rotating_star/CMakeLists.txt:      test_rotating_star_scenario(test_problems.cpu_gpu.am_hydro_on.rotating_star_kokkos_cuda rotating_star_am_hydro_on_kokkos_cuda_log.txt ${silo_scenario_filename} 
test_problems/rotating_star/CMakeLists.txt:        "  --correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1 --monopole_host_kernel_type=KOKKOS --multipole_host_kernel_type=KOKKOS --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:    test_rotating_star_scenario(test_problems.gpu.am_hydro_on.rotating_star_kokkos_hip rotating_star_am_hydro_on_kokkos_hip_log.txt ${silo_scenario_filename} 
test_problems/rotating_star/CMakeLists.txt:      "  --correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=16 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_HIP --multipole_device_kernel_type=KOKKOS_HIP --hydro_device_kernel_type=KOKKOS_HIP --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:    test_rotating_star_scenario(test_problems.gpu.am_hydro_on.rotating_star_kokkos_sycl rotating_star_am_hydro_on_kokkos_sycl_log.txt ${silo_scenario_filename} 
test_problems/rotating_star/CMakeLists.txt:      "  --correct_am_hydro=1 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_SYCL --multipole_device_kernel_type=KOKKOS_SYCL --hydro_device_kernel_type=KOKKOS_SYCL --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:if(OCTOTIGER_WITH_CUDA)
test_problems/rotating_star/CMakeLists.txt:  test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_cuda rotating_star_am_hydro_off_cuda_log.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:  " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:  test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_cuda_work_aggregation rotating_star_am_hydro_off_cuda_work_aggregation_log.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:  " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY --max_kernels_fused=8")
test_problems/rotating_star/CMakeLists.txt:  # combined cpu+gpu
test_problems/rotating_star/CMakeLists.txt:    test_rotating_star_scenario(test_problems.cpu_gpu.am_hydro_off.rotating_star_cuda rotating_star_am_hydro_off_cuda_log.txt ${silo_scenario_filename} 
test_problems/rotating_star/CMakeLists.txt:      "  --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1 --monopole_host_kernel_type=VC --multipole_host_kernel_type=VC --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:  test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_hip rotating_star_am_hydro_off_hip_log.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:    " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=HIP --multipole_device_kernel_type=HIP --hydro_device_kernel_type=HIP --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:    test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_hip_work_aggregation rotating_star_am_hydro_off_hip_work_aggregation_log.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:    " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=8 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=HIP --multipole_device_kernel_type=HIP --hydro_device_kernel_type=HIP --hydro_host_kernel_type=DEVICE_ONLY --max_kernels_fused=8")
test_problems/rotating_star/CMakeLists.txt:  if(OCTOTIGER_WITH_CUDA)
test_problems/rotating_star/CMakeLists.txt:    test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_kokkos_cuda rotating_star_am_hydro_off_kokkos_cuda_log.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:    " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:  test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_kokkos_cuda_work_aggregation rotating_star_am_hydro_off_kokkos_cuda_work_aggregation_log.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:  " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY --max_kernels_fused=8")
test_problems/rotating_star/CMakeLists.txt:      # combined cpu+gpu
test_problems/rotating_star/CMakeLists.txt:      test_rotating_star_scenario(test_problems.cpu_gpu.am_hydro_off.rotating_star_kokkos_cuda rotating_star_am_hydro_off_kokkos_cuda_log.txt ${silo_scenario_filename} 
test_problems/rotating_star/CMakeLists.txt:        "  --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1 --monopole_host_kernel_type=KOKKOS --multipole_host_kernel_type=KOKKOS --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:    test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_kokkos_hip rotating_star_am_hydro_off_kokkos_hip_log.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:      " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=16 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_HIP --multipole_device_kernel_type=KOKKOS_HIP --hydro_device_kernel_type=KOKKOS_HIP --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:      test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_kokkos_hip_work_aggregation rotating_star_am_hydro_off_kokkos_hip_work_aggregation_log.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:      " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=8 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_HIP --multipole_device_kernel_type=KOKKOS_HIP --hydro_device_kernel_type=KOKKOS_HIP --hydro_host_kernel_type=DEVICE_ONLY --max_kernels_fused=8")
test_problems/rotating_star/CMakeLists.txt:    test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_kokkos_sycl rotating_star_am_hydro_off_kokkos_sycl_log.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:      " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_SYCL --multipole_device_kernel_type=KOKKOS_SYCL --hydro_device_kernel_type=KOKKOS_SYCL --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:    test_rotating_star_scenario(test_problems.gpu.am_hydro_off.rotating_star_kokkos_sycl_work_aggregation rotating_star_am_hydro_off_kokkos_sycl_work_aggregation_log.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:    " --correct_am_hydro=0 --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_SYCL --multipole_device_kernel_type=KOKKOS_SYCL --hydro_device_kernel_type=KOKKOS_SYCL --hydro_host_kernel_type=DEVICE_ONLY --max_kernels_fused=8")
test_problems/rotating_star/CMakeLists.txt:  if(OCTOTIGER_WITH_CUDA)
test_problems/rotating_star/CMakeLists.txt:    test_rotating_star_scenario(test_problems.gpu.eos_wd.rotating_star_cuda rotating_star_eos_wd_cuda_log.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:      " --correct_am_hydro=0 --eos=WD --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=CUDA --multipole_device_kernel_type=CUDA --hydro_device_kernel_type=CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:    test_rotating_star_scenario(test_problems.gpu.eos_wd.rotating_star_hip rotating_star_eos_wd_hip.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:      " --correct_am_hydro=0 --eos=WD --number_gpus=1 --executors_per_gpu=16 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=HIP --multipole_device_kernel_type=HIP --hydro_device_kernel_type=HIP --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:  # TODO Add eos=wd CPU+GPU test
test_problems/rotating_star/CMakeLists.txt:    if(OCTOTIGER_WITH_CUDA)
test_problems/rotating_star/CMakeLists.txt:      test_rotating_star_scenario(test_problems.gpu.eos_wd.rotating_star_kokkos_cuda rotating_star_eos_wd_kokkos_cuda_log.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:        " --correct_am_hydro=0 --eos=WD --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_CUDA --multipole_device_kernel_type=KOKKOS_CUDA --hydro_device_kernel_type=KOKKOS_CUDA --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:      test_rotating_star_scenario(test_problems.gpu.eos_wd.rotating_star_kokkos_hip rotating_star_eos_wd_kokkos_hip.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:        " --correct_am_hydro=0 --eos=WD --number_gpus=1 --executors_per_gpu=16 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_HIP --multipole_device_kernel_type=KOKKOS_HIP --hydro_device_kernel_type=KOKKOS_HIP --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:      test_rotating_star_scenario(test_problems.gpu.eos_wd.rotating_star_kokkos_sycl rotating_star_eos_wd_kokkos_sycl.txt ${silo_scenario_filename}
test_problems/rotating_star/CMakeLists.txt:        " --correct_am_hydro=0 --eos=WD --number_gpus=1 --executors_per_gpu=32 --max_gpu_executor_queue_length=1024 --monopole_host_kernel_type=DEVICE_ONLY --multipole_host_kernel_type=DEVICE_ONLY --monopole_device_kernel_type=KOKKOS_SYCL --multipole_device_kernel_type=KOKKOS_SYCL --hydro_device_kernel_type=KOKKOS_SYCL --hydro_host_kernel_type=DEVICE_ONLY")
test_problems/rotating_star/CMakeLists.txt:  # TODO Add eos=wd CPU+GPU kokkos test
frontend/main.cpp:    // TODO Retest with newer ROCM versions (last tested with 5.4.6) and newer AMDGPUs (last tested with MI100).
frontend/main.cpp:    printf("WARNING: Using build without buffer recycling enabled. This will cause a major degradation of GPU performance !\n");
frontend/main.cpp:    // Touch all AMDGPUs before starting HPX. This initializes all GPUs before starting HPX
frontend/main.cpp:    // See bug https://github.com/ROCm-Developer-Tools/HIP/issues/3063
frontend/main.cpp:    for (size_t gpu_id = 0; gpu_id < numDevices; gpu_id++) {
frontend/main.cpp:      hipSetDevice(gpu_id);
frontend/main.cpp:      hipStream_t gpu1;
frontend/main.cpp:      hipStreamCreate(&gpu1);
frontend/main.cpp:      /* hipStreamDestroy(gpu1); */
frontend/main.cpp:         "Enable dedicated HPX thread pool for cuda/network polling using N threads");
frontend/main.cpp:    // TODO Retest with newer ROCM versions (last tested with 5.4.6) and newer AMDGPUs (last tested with MI100).
frontend/main.cpp:              << "This will cause a major degradation of GPU performance !\n";
frontend/frontend-helper.cpp:#ifdef OCTOTIGER_HAVE_CUDA
frontend/frontend-helper.cpp:#include "octotiger/cuda_util/cuda_helper.hpp"
frontend/frontend-helper.cpp:#include "octotiger/cuda_util/cuda_scheduler.hpp"
frontend/frontend-helper.cpp:#include "octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp"
frontend/frontend-helper.cpp:#include "octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp"
frontend/init_methods.cpp:#ifdef OCTOTIGER_HAVE_CUDA
frontend/init_methods.cpp:#include "octotiger/cuda_util/cuda_scheduler.hpp"
frontend/init_methods.cpp:#include "octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp"
frontend/init_methods.cpp:#include "octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp"
frontend/init_methods.cpp:#include <cuda_buffer_util.hpp>
frontend/init_methods.cpp:#ifdef OCTOTIGER_HAVE_CUDA
frontend/init_methods.cpp:#include "octotiger/cuda_util/cuda_helper.hpp"
frontend/init_methods.cpp:// In case we build without kokkos we want the cuda futures to default
frontend/init_methods.cpp:#ifndef HPX_KOKKOS_CUDA_FUTURE_TYPE
frontend/init_methods.cpp:#define HPX_KOKKOS_CUDA_FUTURE_TYPE 0
frontend/init_methods.cpp:    if (opts().executors_per_gpu > 0) {
frontend/init_methods.cpp:#if defined(OCTOTIGER_HAVE_CUDA) 
frontend/init_methods.cpp:      stream_pool::cleanup<hpx::cuda::experimental::cuda_executor, pool_strategy>();
frontend/init_methods.cpp:      stream_pool::cleanup<hpx::cuda::experimental::cuda_executor, pool_strategy>();
frontend/init_methods.cpp:#if defined(KOKKOS_ENABLE_CUDA)
frontend/init_methods.cpp:      stream_pool::cleanup<hpx::kokkos::cuda_executor, round_robin_pool<hpx::kokkos::cuda_executor>>();
frontend/init_methods.cpp:#if (defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)) && HPX_KOKKOS_CUDA_FUTURE_TYPE == 0 
frontend/init_methods.cpp:      std::cout << "Unregistering cuda polling on polling pool... " << std::endl;
frontend/init_methods.cpp:      hpx::cuda::experimental::detail::unregister_polling(hpx::resource::get_thread_pool("polling"));
frontend/init_methods.cpp:        std::cout << "Unregistering cuda polling..." << std::endl;
frontend/init_methods.cpp:        hpx::cuda::experimental::detail::unregister_polling(hpx::resource::get_thread_pool(0));
frontend/init_methods.cpp:    for (size_t gpu_id = 0; gpu_id < numDevices; gpu_id++) {
frontend/init_methods.cpp:      std::cerr << "Resetting HIP device " << gpu_id << "..." << std::endl;
frontend/init_methods.cpp:      hipSetDevice(gpu_id);
frontend/init_methods.cpp:    std::cout << "Check number of available GPUs..." << std::endl;
frontend/init_methods.cpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(KOKKOS_ENABLE_CUDA) 
frontend/init_methods.cpp:    cudaGetDeviceCount(&num_devices);
frontend/init_methods.cpp:    std::cout << "Found " << num_devices << " CUDA devices! " << std::endl;
frontend/init_methods.cpp:      if (opts().number_gpus > num_devices) {
frontend/init_methods.cpp:          std::cerr << "ERROR: Requested " << opts().number_gpus << " GPUs but only "
frontend/init_methods.cpp:      if (opts().number_gpus > recycler::max_number_gpus) {
frontend/init_methods.cpp:        std::cerr << "ERROR: Requested " << opts().number_gpus
frontend/init_methods.cpp:                  << " GPUs but CPPuddle was built with CPPUDDLE_WITH_MAX_NUMBER_GPUS="
frontend/init_methods.cpp:                  << recycler::max_number_gpus << std::endl;
frontend/init_methods.cpp:#if HPX_KOKKOS_CUDA_FUTURE_TYPE == 0
frontend/init_methods.cpp:#if (defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP) || defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))  
frontend/init_methods.cpp:      std::cout << "Registering HPX CUDA polling on polling pool..." << std::endl;
frontend/init_methods.cpp:      hpx::cuda::experimental::detail::register_polling(hpx::resource::get_thread_pool("polling"));
frontend/init_methods.cpp:      std::cout << "Registering HPX CUDA polling..." << std::endl;
frontend/init_methods.cpp:      hpx::cuda::experimental::detail::register_polling(hpx::resource::get_thread_pool(0));
frontend/init_methods.cpp:    std::cout << "Registered HPX CUDA polling!" << std::endl;
frontend/init_methods.cpp:    std::cout << "CPPuddle config: Max number GPUs: " << recycler::max_number_gpus << " devices!" << std::endl;
frontend/init_methods.cpp:#if defined(KOKKOS_ENABLE_CUDA)
frontend/init_methods.cpp:    stream_pool::set_device_selector<hpx::kokkos::cuda_executor,
frontend/init_methods.cpp:          round_robin_pool<hpx::kokkos::cuda_executor>>([](size_t gpu_id) {
frontend/init_methods.cpp:              cudaSetDevice(gpu_id);
frontend/init_methods.cpp:    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
frontend/init_methods.cpp:      stream_pool::init_executor_pool<hpx::kokkos::cuda_executor,
frontend/init_methods.cpp:          round_robin_pool<hpx::kokkos::cuda_executor>>(
frontend/init_methods.cpp:          gpu_id, opts().executors_per_gpu,
frontend/init_methods.cpp:    std::cout << "KOKKOS/CUDA is enabled!" << std::endl;
frontend/init_methods.cpp:          round_robin_pool<hpx::kokkos::hip_executor>>([](size_t gpu_id) {
frontend/init_methods.cpp:              hipSetDevice(gpu_id);
frontend/init_methods.cpp:    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
frontend/init_methods.cpp:          gpu_id, opts().executors_per_gpu,
frontend/init_methods.cpp:    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
frontend/init_methods.cpp:          gpu_id, opts().executors_per_gpu,
frontend/init_methods.cpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP) || defined(KOKKOS_ENABLE_SYCL)
frontend/init_methods.cpp:#if HPX_KOKKOS_CUDA_FUTURE_TYPE == 0 
frontend/init_methods.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
frontend/init_methods.cpp:    std::cout << "CUDA is enabled!" << std::endl;
frontend/init_methods.cpp:    stream_pool::set_device_selector<hpx::cuda::experimental::cuda_executor,
frontend/init_methods.cpp:          round_robin_pool<hpx::cuda::experimental::cuda_executor>>([](size_t gpu_id) {
frontend/init_methods.cpp:              cudaSetDevice(gpu_id);
frontend/init_methods.cpp:#if HPX_KOKKOS_CUDA_FUTURE_TYPE == 0 
frontend/init_methods.cpp:    std::cout << "CUDA with polling futures enabled!" << std::endl;
frontend/init_methods.cpp:    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
frontend/init_methods.cpp:      stream_pool::init_executor_pool<hpx::cuda::experimental::cuda_executor, pool_strategy>(gpu_id,
frontend/init_methods.cpp:          opts().executors_per_gpu, gpu_id, true);
frontend/init_methods.cpp:    std::cout << "CUDA with callback futures enabled!" << std::endl;
frontend/init_methods.cpp:    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
frontend/init_methods.cpp:      stream_pool::init_executor_pool<hpx::cuda::experimental::cuda_executor, pool_strategy>(gpu_id,
frontend/init_methods.cpp:          opts().executors_per_gpu, gpu_id, false);
frontend/init_methods.cpp:    stream_pool::set_device_selector<hpx::cuda::experimental::cuda_executor,
frontend/init_methods.cpp:          round_robin_pool<hpx::cuda::experimental::cuda_executor>>([](size_t gpu_id) {
frontend/init_methods.cpp:              hipSetDevice(gpu_id);
frontend/init_methods.cpp:#if HPX_KOKKOS_CUDA_FUTURE_TYPE == 0  // cuda in the name is correct
frontend/init_methods.cpp:    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
frontend/init_methods.cpp:      stream_pool::init_executor_pool<hpx::cuda::experimental::cuda_executor, pool_strategy>(gpu_id,
frontend/init_methods.cpp:          opts().executors_per_gpu, gpu_id, true);
frontend/init_methods.cpp:    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
frontend/init_methods.cpp:      stream_pool::init_executor_pool<hpx::cuda::experimental::cuda_executor, pool_strategy>(gpu_id,
frontend/init_methods.cpp:          opts().executors_per_gpu, gpu_id, false);
frontend/init_methods.cpp:    // CUDA host / device allocators -- also used by KOKKOS
frontend/init_methods.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
frontend/init_methods.cpp:            double, recycler::detail::cuda_pinned_allocator<double>>);
frontend/init_methods.cpp:            int, recycler::detail::cuda_pinned_allocator<int>>);
frontend/init_methods.cpp:            double, recycler::detail::cuda_device_allocator<double>>);
frontend/init_methods.cpp:            int, recycler::detail::cuda_device_allocator<int>>);
src/unitiger/hydro_impl/hydro_boundary_exchange_cuda.cpp://#include <__clang_cuda_builtin_vars.h>
src/unitiger/hydro_impl/hydro_boundary_exchange_cuda.cpp:#ifdef OCTOTIGER_HAVE_CUDA
src/unitiger/hydro_impl/hydro_boundary_exchange_cuda.cpp:complete_hydro_amr_cuda_kernel(const double *dx_global, const int *energy_only_global,
src/unitiger/hydro_impl/hydro_boundary_exchange_cuda.cpp:complete_hydro_amr_cuda_kernel_phase2(
src/unitiger/hydro_impl/hydro_boundary_exchange_cuda.cpp:void launch_complete_hydro_amr_boundary_cuda_post(
src/unitiger/hydro_impl/hydro_boundary_exchange_cuda.cpp:    cudaLaunchKernel<decltype(complete_hydro_amr_cuda_kernel)>,
src/unitiger/hydro_impl/hydro_boundary_exchange_cuda.cpp:    complete_hydro_amr_cuda_kernel, grid_spec, threads_per_block, args, 0);
src/unitiger/hydro_impl/hydro_boundary_exchange_cuda.cpp:void launch_complete_hydro_amr_boundary_cuda_phase2_post(
src/unitiger/hydro_impl/hydro_boundary_exchange_cuda.cpp:    cudaLaunchKernel<decltype(complete_hydro_amr_cuda_kernel_phase2)>,
src/unitiger/hydro_impl/hydro_boundary_exchange_cuda.cpp:    complete_hydro_amr_cuda_kernel_phase2, grid_spec, threads_per_block, args, 0);
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp://#include <__clang_cuda_builtin_vars.h>
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#include <hpx/modules/async_cuda.hpp>
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#include "octotiger/cuda_util/cuda_helper.hpp"
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#define cudaLaunchKernel hipLaunchKernel
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#define cudaMemcpy hipMemcpy
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#define cudaMemcpyAsync hipMemcpyAsync
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:__global__ void __launch_bounds__(64, 4) reconstruct_cuda_kernel_no_amc(double omega,
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:__global__ void reconstruct_cuda_kernel_no_amc(double omega,
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:__global__ void __launch_bounds__(64, 4) reconstruct_cuda_kernel(double omega, int nf_,
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:__global__ void reconstruct_cuda_kernel(const double omega, const int nf_,
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:    cudaStream_t const& stream) {
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:				hipLaunchKernelGGL(reconstruct_cuda_kernel, grid_spec, threads_per_block, 0, stream, omega, nf_,
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:				hipLaunchKernelGGL(reconstruct_cuda_kernel_no_amc, grid_spec, threads_per_block, 0, stream, omega, nf_,
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:void launch_reconstruct_cuda(
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:    static_assert(device_simd_t::size() == 1, "CUDA/HIP kernels expect scalar SIMD types");
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:      /* hpx::apply(executor.post(, cudaLaunchKernel<decltype(reconstruct_cuda_kernel)>, reconstruct_cuda_kernel, */
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:      executor.post(cudaLaunchKernel<decltype(reconstruct_cuda_kernel)>, reconstruct_cuda_kernel,
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:        executor.post(cudaLaunchKernel<decltype(reconstruct_cuda_kernel_no_amc)>,
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:            reconstruct_cuda_kernel_no_amc, grid_spec, threads_per_block, args, 0);
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:    cudaStream_t const& stream) {
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:    double* device_disc, double* device_P, double fgamma_, int ndir, cudaStream_t const& stream) {
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:void launch_find_contact_discs_cuda(
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:    /* hpx::apply(executor, cudaLaunchKernel<decltype(discs_phase1)>, discs_phase1, grid_spec_phase1, */
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:    executor.post(cudaLaunchKernel<decltype(discs_phase1)>, discs_phase1, grid_spec_phase1,
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:    /* hpx::apply(executor, cudaLaunchKernel<decltype(discs_phase2)>, discs_phase2, grid_spec_phase2, */
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:    executor.post(cudaLaunchKernel<decltype(discs_phase2)>, discs_phase2, grid_spec_phase2,
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:    hydro_pre_recon_cuda(double* __restrict__ device_X, safe_real omega, bool angmom,
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:    cudaStream_t const& stream) {
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:    hipLaunchKernelGGL(hydro_pre_recon_cuda, grid_spec, threads_per_block, 0, stream, device_X,
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:void launch_hydro_pre_recon_cuda(
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:    /* hpx::apply(executor, cudaLaunchKernel<decltype(hydro_pre_recon_cuda)>, hydro_pre_recon_cuda, grid_spec, */
src/unitiger/hydro_impl/reconstruct_cuda_kernel.cpp:    executor.post(cudaLaunchKernel<decltype(hydro_pre_recon_cuda)>, hydro_pre_recon_cuda, grid_spec,
src/unitiger/hydro_impl/hydro_performance_counters.cpp:std::atomic<std::uint64_t> hydro_cuda_gpu_subgrids_processed(0);
src/unitiger/hydro_impl/hydro_performance_counters.cpp:std::atomic<std::uint64_t> hydro_cuda_gpu_aggregated_subgrids_launches(0);
src/unitiger/hydro_impl/hydro_performance_counters.cpp:std::atomic<std::uint64_t> hydro_kokkos_gpu_subgrids_processed(0);
src/unitiger/hydro_impl/hydro_performance_counters.cpp:std::atomic<std::uint64_t> hydro_kokkos_gpu_aggregated_subgrids_launches(0);
src/unitiger/hydro_impl/hydro_performance_counters.cpp:std::uint64_t hydro_cuda_gpu_subgrid_processed_performance_data(bool reset) {
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        hydro_cuda_gpu_subgrids_processed = 0;
src/unitiger/hydro_impl/hydro_performance_counters.cpp:    return hydro_cuda_gpu_subgrids_processed;
src/unitiger/hydro_impl/hydro_performance_counters.cpp:std::uint64_t hydro_cuda_gpu_aggregated_subgrids_launches_performance_data(bool reset) {
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        hydro_cuda_gpu_aggregated_subgrids_launches = 0;
src/unitiger/hydro_impl/hydro_performance_counters.cpp:    return hydro_cuda_gpu_aggregated_subgrids_launches;
src/unitiger/hydro_impl/hydro_performance_counters.cpp:std::uint64_t hydro_cuda_avg_aggregation_rate(bool reset) {
src/unitiger/hydro_impl/hydro_performance_counters.cpp:    if (hydro_cuda_gpu_aggregated_subgrids_launches > 0)
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        return hydro_cuda_gpu_subgrids_processed / hydro_cuda_gpu_aggregated_subgrids_launches;
src/unitiger/hydro_impl/hydro_performance_counters.cpp:std::uint64_t hydro_kokkos_gpu_subgrids_processed_performance_data(bool reset) {
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        hydro_kokkos_gpu_subgrids_processed = 0;
src/unitiger/hydro_impl/hydro_performance_counters.cpp:    return hydro_kokkos_gpu_subgrids_processed;
src/unitiger/hydro_impl/hydro_performance_counters.cpp:std::uint64_t hydro_kokkos_gpu_aggregated_subgrids_launches_performance_data(bool reset) {
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        hydro_kokkos_gpu_aggregated_subgrids_launches = 0;
src/unitiger/hydro_impl/hydro_performance_counters.cpp:    return hydro_kokkos_gpu_aggregated_subgrids_launches;
src/unitiger/hydro_impl/hydro_performance_counters.cpp:std::uint64_t hydro_kokkos_gpu_avg_aggregation_rate(bool reset) {
src/unitiger/hydro_impl/hydro_performance_counters.cpp:    if (hydro_kokkos_gpu_aggregated_subgrids_launches > 0)
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        return hydro_kokkos_gpu_subgrids_processed / hydro_kokkos_gpu_aggregated_subgrids_launches;
src/unitiger/hydro_impl/hydro_performance_counters.cpp:    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/hydro_cuda",
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        &hydro_cuda_gpu_subgrid_processed_performance_data,
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        "Number of calls to the hydro_solver with CUDA. Each call handles one sub-grid for one "
src/unitiger/hydro_impl/hydro_performance_counters.cpp:    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/hydro_cuda_aggregated",
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        &hydro_cuda_gpu_aggregated_subgrids_launches_performance_data,
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        "Number of aggregated calls to the hydro_solver with CUDA. Each call handles one ore more "
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        "/octotiger/compute/gpu/hydro_cuda_aggregation_rate", &hydro_cuda_avg_aggregation_rate,
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        "Average number of hydro CUDA kernels per aggregated kernel call");
src/unitiger/hydro_impl/hydro_performance_counters.cpp:    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/hydro_kokkos",
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        &hydro_kokkos_gpu_subgrids_processed_performance_data,
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        "Number of calls to the hydro_solver with KOKKOS_GPU. Each call handles one sub-grid for "
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        "/octotiger/compute/gpu/hydro_kokkos_aggregated",
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        &hydro_kokkos_gpu_aggregated_subgrids_launches_performance_data,
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        "Number of aggregated calls to the hydro_solver with KOKKOS_GPU. Each call handles one ore "
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        "/octotiger/compute/gpu/hydro_kokkos_aggregation_rate",
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        &hydro_kokkos_gpu_avg_aggregation_rate,
src/unitiger/hydro_impl/hydro_performance_counters.cpp:        "Average number of hydro KOKKOS_GPU kernels per aggregated kernel call");
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:#include <cuda_buffer_util.hpp>
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:#include <cuda_runtime.h>
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:#include "octotiger/cuda_util/cuda_helper.hpp"
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:static const char hydro_cuda_kernel_identifier[] = "hydro_kernel_aggregator_cuda";
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:using hydro_cuda_agg_executor_pool = aggregation_pool<hydro_cuda_kernel_identifier,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:    hpx::cuda::experimental::cuda_executor, pool_strategy>;
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:using host_pinned_allocator = recycler::detail::cuda_pinned_allocator<T>;
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:using device_allocator = recycler::detail::cuda_device_allocator<T>;
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:using device_buffer_t = recycler::cuda_device_buffer<T>;
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:using aggregated_device_buffer_t = recycler::cuda_aggregated_device_buffer<T, Alloc>;
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:using host_buffer_t = std::vector<T, recycler::recycle_allocator_cuda_host<T>>;
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:using executor_t = hpx::cuda::experimental::cuda_executor;
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:using executor_t = hpx::cuda::experimental::cuda_executor;
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:#define cudaLaunchKernel hipLaunchKernel
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:#define cudaMemcpy hipMemcpy
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:#define cudaMemcpyAsync hipMemcpyAsync
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:    hydro_cuda_agg_executor_pool::init(number_aggregation_executors, max_slices,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:        executor_mode, opts().number_gpus);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:__host__ void init_gpu_masks(std::array<bool*, recycler::max_number_gpus>& masks) {
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:    for (size_t gpu_id = 0; gpu_id < opts().number_gpus; gpu_id++) {
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      masks[gpu_id] = recycler::detail::buffer_recycler::get<bool,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          typename recycler::recycle_allocator_cuda_device<bool>::underlying_allocator_type>(
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          NDIM * q_inx3, false, location_id, gpu_id);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      masks[gpu_id] = recycler::detail::buffer_recycler::get<bool,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          NDIM * q_inx3, false, location_id, gpu_id);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      cudaMemcpy(masks[gpu_id], masks_boost.data(), NDIM * q_inx3 * sizeof(bool), cudaMemcpyHostToDevice);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:__host__ bool* get_gpu_masks(const size_t gpu_id = 0) {
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:    static std::array<bool*, recycler::max_number_gpus> masks;
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:    hpx::call_once(flag1, init_gpu_masks, masks);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:    return masks[gpu_id];
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:timestep_t launch_hydro_cuda_kernels(const hydro_computer<NDIM, INX, physics<NDIM>>& hydro,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:    auto executor_slice_fut = hydro_cuda_agg_executor_pool::request_executor_slice();
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          "cuda_hydro_solver::convert_input")();
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      hpx::apply(exec_slice, cudaMemcpyAsync, device_u.device_side_buffer, combined_u.data(),
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          (hydro.get_nf() * H_N3 + 128) * sizeof(double) * number_slices, cudaMemcpyHostToDevice);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      hpx::apply(exec_slice, cudaMemcpyAsync, device_x.device_side_buffer, combined_x.data(),
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          (NDIM * q_inx3 + 128) * sizeof(double) * number_slices, cudaMemcpyHostToDevice);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      hpx::apply(exec_slice, cudaMemcpyAsync, device_disc_detect.device_side_buffer,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          cudaMemcpyHostToDevice);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      hpx::apply(exec_slice, cudaMemcpyAsync, device_smooth_field.device_side_buffer,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          cudaMemcpyHostToDevice);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      launch_find_contact_discs_cuda(exec_slice, device_u.device_side_buffer,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      hpx::apply(exec_slice, cudaMemcpyAsync, device_large_x.device_side_buffer,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          cudaMemcpyHostToDevice);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      launch_hydro_pre_recon_cuda(exec_slice, device_large_x.device_side_buffer, omega,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      hpx::apply(exec_slice, cudaMemcpyAsync, dx_device.device_side_buffer, dx_host.data(),
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          number_slices * sizeof(double), cudaMemcpyHostToDevice);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      launch_reconstruct_cuda(exec_slice, omega, hydro.get_nf(), hydro.get_angmom_index(),
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      const bool* masks = get_gpu_masks(exec_slice.parent.gpu_id);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      launch_flux_cuda_kernel_post(exec_slice, grid_spec, threads_per_block,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      hpx::apply(exec_slice, cudaMemcpyAsync, amax.data(), device_amax.device_side_buffer,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          cudaMemcpyDeviceToHost);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      hpx::apply(exec_slice, cudaMemcpyAsync, amax_indices.data(),
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          number_slices * number_blocks * NDIM * sizeof(int), cudaMemcpyDeviceToHost);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      hpx::apply(exec_slice, cudaMemcpyAsync, amax_d.data(), device_amax_d.device_side_buffer,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          number_slices * number_blocks * NDIM * sizeof(int), cudaMemcpyDeviceToHost);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:          hpx::async(exec_slice, cudaMemcpyAsync, f.data(), device_f.device_side_buffer,
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:              cudaMemcpyDeviceToHost);
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      octotiger::hydro::hydro_cuda_gpu_subgrids_processed++;
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:        octotiger::hydro::hydro_cuda_gpu_aggregated_subgrids_launches++;
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      /* auto max_lambda = launch_flux_cuda(executor, device_q.device_side_buffer, f, combined_x, */
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:      }, "cuda_hydro_solver::convert_output")();
src/unitiger/hydro_impl/hydro_cuda_interface.cpp:    }, "cuda_hydro_solver"));
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:#ifdef OCTOTIGER_HAVE_CUDA
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:static const char amr_cuda_kernel_identifier[] = "amr_kernel_aggregator_cuda";
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:using amr_cuda_agg_executor_pool = aggregation_pool<amr_cuda_kernel_identifier, hpx::cuda::experimental::cuda_executor,
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:    amr_cuda_agg_executor_pool::init(number_aggregation_executors, max_slices, executor_mode);
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:__host__ void launch_complete_hydro_amr_boundary_cuda(double dx, bool
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:      amr_cuda_agg_executor_pool::request_executor_slice();
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:                .template make_allocator<double, recycler::detail::cuda_pinned_allocator<double>>();
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:                .template make_allocator<double, recycler::detail::cuda_device_allocator<double>>();
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:            exec_slice.template make_allocator<int, recycler::detail::cuda_pinned_allocator<int>>();
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:            exec_slice.template make_allocator<int, recycler::detail::cuda_device_allocator<int>>();
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        recycler::cuda_aggregated_device_buffer<double, decltype(alloc_device_double)> device_uf(
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        recycler::cuda_aggregated_device_buffer<double, decltype(alloc_device_double)> device_ushad(
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        recycler::cuda_aggregated_device_buffer<int, decltype(alloc_device_int)> device_coarse(
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        recycler::cuda_aggregated_device_buffer<double, decltype(alloc_device_double)> device_xmin(
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        recycler::cuda_aggregated_device_buffer<int, decltype(alloc_device_int)> energy_only_device(
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        recycler::cuda_aggregated_device_buffer<double, decltype(alloc_device_double)> dx_device(
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        exec_slice.post(cudaMemcpyAsync, device_xmin.device_side_buffer, x_min.data(),
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:            number_slices * (NDIM) * sizeof(double), cudaMemcpyHostToDevice);
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        exec_slice.post(cudaMemcpyAsync, dx_device.device_side_buffer, dx_host.data(),
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:            number_slices * sizeof(double), cudaMemcpyHostToDevice);
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        exec_slice.post(cudaMemcpyAsync, energy_only_device.device_side_buffer,
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:            cudaMemcpyHostToDevice);
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        exec_slice.post(cudaMemcpyAsync, device_ushad.device_side_buffer, unified_ushad.data(),
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:            cudaMemcpyHostToDevice);
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        exec_slice.post(cudaMemcpyAsync, device_coarse.device_side_buffer, coarse.data(),
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:            number_slices * (HS_N3) * sizeof(int), cudaMemcpyHostToDevice);
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        launch_complete_hydro_amr_boundary_cuda_post(
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        /*     exec_slice.async(cudaMemcpyAsync, unified_uf.data(), device_uf.device_side_buffer, */
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        /*         cudaMemcpyDeviceToHost); */
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        recycler::cuda_aggregated_device_buffer<double, decltype(alloc_device_double)> device_u(
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:        launch_complete_hydro_amr_boundary_cuda_phase2_post(
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:            exec_slice.async(cudaMemcpyAsync, unified_u.data(), device_u.device_side_buffer,
src/unitiger/hydro_impl/hydro_boundary_exchange.cpp:                cudaMemcpyDeviceToHost);
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:#if defined(OCTOTIGER_HAVE_INTEL_GPU_WORKAROUND)
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:#pragma message "SYCL builds without OCTOTIGER_WITH_INTEL_GPU_WORKAROUND=ON may break on Intel GPUs"
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:#if defined(KOKKOS_ENABLE_CUDA)
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:using device_executor = hpx::kokkos::cuda_executor;
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:    if (opts().executors_per_gpu > 0) {
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:#if defined(KOKKOS_ENABLE_CUDA)
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:        hydro_kokkos_agg_executor_pool<hpx::kokkos::cuda_executor>::init(
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:            number_aggregation_executors, max_slices, executor_mode, opts().number_gpus);
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:            number_aggregation_executors, max_slices, executor_mode, opts().number_gpus);
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:            number_aggregation_executors, max_slices, executor_mode, opts().number_gpus);
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:#include <hpx/async_cuda/cuda_executor.hpp>
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:using device_executor_cuda = hpx::cuda::experimental::cuda_executor;
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:using device_pool_strategy_cuda = round_robin_pool<device_executor_cuda>;
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:using executor_interface_cuda_t = stream_interface<device_executor_cuda, device_pool_strategy_cuda>;
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:    const size_t max_gpu_executor_queue_length) {
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:        if (device_type == interaction_device_kernel_type::KOKKOS_CUDA ||
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:#if defined(OCTOTIGER_HAVE_KOKKOS) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)|| defined(KOKKOS_ENABLE_SYCL))
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:                    stream_pool::get_next_device_id<device_executor, device_pool_strategy>(opts().number_gpus);
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:                    max_gpu_executor_queue_length, device_id);
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:        if (device_type == interaction_device_kernel_type::CUDA) {
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:#ifdef OCTOTIGER_HAVE_CUDA
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:                    stream_pool::get_next_device_id<device_executor_cuda, device_pool_strategy_cuda>(opts().number_gpus);
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:                avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor,
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:                    device_pool_strategy_cuda>(max_gpu_executor_queue_length, device_id);
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:                max_lambda = launch_hydro_cuda_kernels(hydro, U, X, omega, device_id, F);
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:            std::cerr << "Trying to call Hydro CUDA device kernels in a non-CUDA build! "
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:                    stream_pool::get_next_device_id<device_executor_cuda, device_pool_strategy_cuda>(opts().number_gpus);
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:              avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor,
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:                  device_pool_strategy_cuda>(max_gpu_executor_queue_length, device_id);
src/unitiger/hydro_impl/hydro_kernel_interface.cpp:                max_lambda = launch_hydro_cuda_kernels(hydro, U, X, omega, device_id, F);
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:#define cudaLaunchKernel hipLaunchKernel
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:#define cudaMemcpy hipMemcpy
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:#define cudaMemcpyAsync hipMemcpyAsync
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:#define cudaError_t hipError_t
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:#include <hpx/modules/async_cuda.hpp>
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:#include "octotiger/cuda_util/cuda_helper.hpp"
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:#include <cuda_buffer_util.hpp>
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:#include <cuda_runtime.h>
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:__global__ void __launch_bounds__(128, 2) flux_cuda_kernel(const double* __restrict__ q_combined,
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:    const double de_switch_1, const int number_blocks, cudaStream_t const &stream) {
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:    hipLaunchKernelGGL(flux_cuda_kernel, grid_spec, threads_per_block, 0,
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:void launch_flux_cuda_kernel_post(
src/unitiger/hydro_impl/flux_cuda_kernel.cpp:    executor.post(cudaLaunchKernel<decltype(flux_cuda_kernel)>, flux_cuda_kernel, grid_spec,
src/common_kernel/gravity_performance_counters.cpp:std::atomic<std::uint64_t> p2p_kokkos_gpu_subgrids_launched(0);
src/common_kernel/gravity_performance_counters.cpp:std::atomic<std::uint64_t> p2p_cuda_gpu_subgrids_launched(0);
src/common_kernel/gravity_performance_counters.cpp:std::atomic<std::uint64_t> multipole_kokkos_gpu_subgrids_launched(0);
src/common_kernel/gravity_performance_counters.cpp:std::atomic<std::uint64_t> multipole_cuda_gpu_subgrids_launched(0);
src/common_kernel/gravity_performance_counters.cpp:std::uint64_t p2p_kokkos_gpu_subgrid_processed_performance_data(bool reset) {
src/common_kernel/gravity_performance_counters.cpp:        p2p_kokkos_gpu_subgrids_launched = 0;
src/common_kernel/gravity_performance_counters.cpp:    return p2p_kokkos_gpu_subgrids_launched;
src/common_kernel/gravity_performance_counters.cpp:std::uint64_t p2p_cuda_gpu_subgrid_processed_performance_data(bool reset) {
src/common_kernel/gravity_performance_counters.cpp:        p2p_cuda_gpu_subgrids_launched = 0;
src/common_kernel/gravity_performance_counters.cpp:    return p2p_cuda_gpu_subgrids_launched;
src/common_kernel/gravity_performance_counters.cpp:std::uint64_t multipole_kokkos_gpu_subgrid_processed_performance_data(bool reset) {
src/common_kernel/gravity_performance_counters.cpp:        multipole_kokkos_gpu_subgrids_launched = 0;
src/common_kernel/gravity_performance_counters.cpp:    return multipole_kokkos_gpu_subgrids_launched;
src/common_kernel/gravity_performance_counters.cpp:std::uint64_t multipole_cuda_gpu_subgrid_processed_performance_data(bool reset) {
src/common_kernel/gravity_performance_counters.cpp:        multipole_cuda_gpu_subgrids_launched = 0;
src/common_kernel/gravity_performance_counters.cpp:    return multipole_cuda_gpu_subgrids_launched;
src/common_kernel/gravity_performance_counters.cpp:    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/p2p_cuda",
src/common_kernel/gravity_performance_counters.cpp:        &p2p_cuda_gpu_subgrid_processed_performance_data,
src/common_kernel/gravity_performance_counters.cpp:        "Number of calls to the fmm solver (p2p) with CUDA. Each call handles one sub-grid for one "
src/common_kernel/gravity_performance_counters.cpp:    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/p2p_kokkos",
src/common_kernel/gravity_performance_counters.cpp:        &p2p_kokkos_gpu_subgrid_processed_performance_data,
src/common_kernel/gravity_performance_counters.cpp:        "Number of calls to the fmm solver (p2p) with KOKKOS_GPU. Each call handles one sub-grid for one "
src/common_kernel/gravity_performance_counters.cpp:    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/multipole_cuda",
src/common_kernel/gravity_performance_counters.cpp:        &multipole_cuda_gpu_subgrid_processed_performance_data,
src/common_kernel/gravity_performance_counters.cpp:        "Number of calls to the fmm solver (multipole) with CUDA. Each call handles one sub-grid for one "
src/common_kernel/gravity_performance_counters.cpp:    hpx::performance_counters::install_counter_type("/octotiger/compute/gpu/multipole_kokkos",
src/common_kernel/gravity_performance_counters.cpp:        &multipole_kokkos_gpu_subgrid_processed_performance_data,
src/common_kernel/gravity_performance_counters.cpp:        "Number of calls to the fmm solver (multipole) with KOKKOS_GPU. Each call handles one sub-grid for one "
src/node_server.cpp:// CUDA implementation supports optional execution on GPU, hence we need to check if a stram is available
src/node_server.cpp:#ifdef OCTOTIGER_HAVE_CUDA
src/node_server.cpp:		if (kernel_type == AMR_CUDA)
src/node_server.cpp:			/* avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor, */
src/node_server.cpp:			/* 		pool_strategy>(opts().max_gpu_executor_queue_length); */
src/node_server.cpp:		} else { // Run on GPU
src/node_server.cpp:			launch_complete_hydro_amr_boundary_cuda(dx, energy_only, grid_ptr->Ushad,
src/node_server.cpp:// None GPU build -> run on CPU
src/radiation/cuda_kernel.cu:#if defined(OCTOTIGER_HAVE_CUDA)
src/radiation/cuda_kernel.cu:#include "octotiger/radiation/cuda_kernel.hpp"
src/radiation/cuda_kernel.cu:__global__ void cuda_radiation_kernel() {}
src/radiation/cuda_kernel.cu:#endif    // OCTOTIGER_HAVE_CUDA
src/monopole_interactions/monopole_kernel_interface.cpp:#include "octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp"
src/monopole_interactions/monopole_kernel_interface.cpp:#if defined(OCTOTIGER_HAVE_INTEL_GPU_WORKAROUND)
src/monopole_interactions/monopole_kernel_interface.cpp:#pragma message "SYCL builds without OCTOTIGER_WITH_INTEL_GPU_WORKAROUND=ON may break on Intel GPUs"
src/monopole_interactions/monopole_kernel_interface.cpp:#if defined(KOKKOS_ENABLE_CUDA)
src/monopole_interactions/monopole_kernel_interface.cpp:using device_executor = hpx::kokkos::cuda_executor;
src/monopole_interactions/monopole_kernel_interface.cpp:    if (opts().executors_per_gpu > 0) {
src/monopole_interactions/monopole_kernel_interface.cpp:#if defined(KOKKOS_ENABLE_CUDA)
src/monopole_interactions/monopole_kernel_interface.cpp:        monopole_kokkos_agg_executor_pool<hpx::kokkos::cuda_executor>::init(
src/monopole_interactions/monopole_kernel_interface.cpp:            number_aggregation_executors, max_slices, executor_mode, opts().number_gpus);
src/monopole_interactions/monopole_kernel_interface.cpp:            number_aggregation_executors, max_slices, executor_mode, opts().number_gpus);
src/monopole_interactions/monopole_kernel_interface.cpp:            number_aggregation_executors, max_slices, executor_mode, opts().number_gpus);
src/monopole_interactions/monopole_kernel_interface.cpp:                if (device_type == interaction_device_kernel_type::KOKKOS_CUDA ||
src/monopole_interactions/monopole_kernel_interface.cpp:#if defined(OCTOTIGER_HAVE_KOKKOS) && (defined(KOKKOS_ENABLE_CUDA) || \
src/monopole_interactions/monopole_kernel_interface.cpp:                        stream_pool::get_next_device_id<device_executor, device_pool_strategy>(opts().number_gpus);
src/monopole_interactions/monopole_kernel_interface.cpp:                                opts().max_gpu_executor_queue_length, device_id);
src/monopole_interactions/monopole_kernel_interface.cpp:                        p2p_kokkos_gpu_subgrids_launched++;
src/monopole_interactions/monopole_kernel_interface.cpp:                if (device_type == interaction_device_kernel_type::CUDA) {
src/monopole_interactions/monopole_kernel_interface.cpp:#ifdef OCTOTIGER_HAVE_CUDA
src/monopole_interactions/monopole_kernel_interface.cpp:                    cuda_monopole_interaction_interface monopole_interactor{};
src/monopole_interactions/monopole_kernel_interface.cpp:                    p2p_cuda_gpu_subgrids_launched++;
src/monopole_interactions/monopole_kernel_interface.cpp:                    std::cerr << "Trying to call P2P CUDA kernel in a non-CUDA build! "
src/monopole_interactions/monopole_kernel_interface.cpp:                    cuda_monopole_interaction_interface monopole_interactor{};
src/monopole_interactions/monopole_kernel_interface.cpp:                    p2p_cuda_gpu_subgrids_launched++;
src/monopole_interactions/legacy/monopole_interaction_interface.cpp:        size_t& monopole_interaction_interface::cuda_launch_counter() {
src/monopole_interactions/legacy/monopole_interaction_interface.cpp:            static thread_local size_t cuda_launch_counter_ = 0;
src/monopole_interactions/legacy/monopole_interaction_interface.cpp:            return cuda_launch_counter_;
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:#include "octotiger/monopole_interactions/legacy/cuda_monopole_interaction_interface.hpp"
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:#include "octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp"
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:#include <cuda_buffer_util.hpp>
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:#include <cuda_runtime.h>
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:using device_buffer_t = recycler::cuda_device_buffer<T>;
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:using host_buffer_t = std::vector<T, recycler::recycle_allocator_cuda_host<T>>;
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:using executor_t = hpx::cuda::experimental::cuda_executor;
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:using executor_t = hpx::cuda::experimental::cuda_executor;
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:#define cudaLaunchKernel hipLaunchKernel
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:#define cudaMemcpy hipMemcpy
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:#define cudaMemcpyAsync hipMemcpyAsync
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                // executor.post(cudaLaunchKernel<decltype(cuda_p2m_interaction_rho)>,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                //     cuda_p2m_interaction_rho, grid_spec, threads_per_block, args, 0);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                launch_p2m_rho_cuda_kernel_post(executor, grid_spec, threads_per_block, args);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                // executor.post(cudaLaunchKernel<decltype(cuda_p2m_interaction_non_rho)>,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                //     cuda_p2m_interaction_non_rho, grid_spec, threads_per_block, args, 0);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                launch_p2m_non_rho_cuda_kernel_post(executor, grid_spec, threads_per_block, args);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:        cuda_monopole_interaction_interface::cuda_monopole_interaction_interface()
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:        void cuda_monopole_interaction_interface::compute_interactions(std::vector<real>& monopoles,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                stream_pool::get_next_device_id<hpx::cuda::experimental::cuda_executor,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                    pool_strategy>(opts().number_gpus);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                    pool_strategy>(opts().max_gpu_executor_queue_length, device_id);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                       "Use KOKKOS_CUDA instead or recompile with larger minimal theta!");
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                // run on CUDA device
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                cuda_launch_counter()++;
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy> executor{device_id};
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                cuda_expansion_result_buffer_t potential_expansions_SoA;
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                cuda_monopole_buffer_t local_monopoles(ENTRIES);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                    cudaMemcpyAsync, device_local_monopoles.device_side_buffer,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                    local_monopoles.data(), local_monopoles_size, cudaMemcpyHostToDevice);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                launch_p2p_cuda_kernel_post(executor, grid_spec, threads_per_block, args);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                      hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                          cudaMemsetAsync, device_erg_corrs.device_side_buffer, 0,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        cudaMemcpyAsync, center_of_masses_inner_cells.device_side_buffer,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        (INNER_CELLS + SOA_PADDING) * 3 * sizeof(double), cudaMemcpyHostToDevice);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>>
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>>
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>>
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>>
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>>
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        std::vector<real, recycler::recycle_allocator_cuda_host<real>>>>
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                    // Loop that launches p2m cuda kernels for appropriate neighbors
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    cudaMemcpyAsync,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    cudaMemcpyHostToDevice);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    cudaMemcpyAsync,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    cudaMemcpyHostToDevice);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    cudaMemcpyAsync,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    cudaMemcpyHostToDevice);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    cudaMemcpyAsync,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    cudaMemcpyHostToDevice);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    cudaMemcpyAsync,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    cudaMemcpyHostToDevice);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    cudaMemcpyAsync,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                                    cudaMemcpyHostToDevice);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                    cuda_angular_result_t angular_corrections_SoA;
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                            cudaMemcpyAsync, angular_corrections_SoA.get_pod(),
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                            cudaMemcpyDeviceToHost);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        cudaMemcpyAsync, potential_expansions_SoA.get_pod(), erg.device_side_buffer,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        potential_expansions_small_size, cudaMemcpyDeviceToHost);
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        cudaMemcpyAsync, potential_expansions_SoA.get_pod(), erg.device_side_buffer,
src/monopole_interactions/legacy/cuda_monopole_interaction_interface.cpp:                        potential_expansions_small_size, cudaMemcpyDeviceToHost);
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:#include "octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp"
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:#include "octotiger/cuda_util/cuda_scheduler.hpp"
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:#define cudaSetDevice hipSetDevice
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:#define cudaMemcpyToSymbol hipMemcpyToSymbol
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:        // Note: This renders the CUDA versions non-functional -- however there is a mechanism
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:        // in place which throws an error if a user tries to use the CUDA kernel in this configuration
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:        __host__ void init_stencil(size_t gpu_id, std::unique_ptr<bool[]> stencil_masks,
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            cudaSetDevice(gpu_id);
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            cudaMemcpyToSymbol(monopole_interactions::device_stencil_masks, stencil_masks.get(),
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            cudaMemcpyToSymbol(monopole_interactions::device_four_constants,
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:        __global__ void cuda_p2p_interactions_kernel(
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:        __global__ void __launch_bounds__(INX* INX, 2) cuda_p2p_interactions_kernel(
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:        __global__ void cuda_sum_p2p_results(
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            hipLaunchKernelGGL(cuda_p2p_interactions_kernel, grid_spec, threads_per_block,
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            hipLaunchKernelGGL(cuda_sum_p2p_results, grid_spec, threads_per_block,
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:        void launch_sum_p2p_results_post(stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            auto launch_function = cudaLaunchKernel<decltype(cuda_sum_p2p_results)>;
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:                cuda_sum_p2p_results, grid_spec, threads_per_block, args, 0);
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:        void launch_p2p_cuda_kernel_post(stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            auto launch_function = cudaLaunchKernel<decltype(cuda_p2p_interactions_kernel)>;
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:                cuda_p2p_interactions_kernel, grid_spec, threads_per_block, args, 0);
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            cuda_p2m_interaction_rho(const double* __restrict__ expansions_neighbors_soa,
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:        void launch_p2m_rho_cuda_kernel_post(stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:          auto launch_function = cudaLaunchKernel<decltype(cuda_p2m_interaction_rho)>;
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            cuda_p2m_interaction_rho, grid_spec, threads_per_block, args, 0);
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            cuda_p2m_interaction_non_rho(const double* __restrict__ expansions_neighbors_soa,
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:        void launch_p2m_non_rho_cuda_kernel_post(
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:          auto launch_function = cudaLaunchKernel<decltype(cuda_p2m_interaction_non_rho)>;
src/monopole_interactions/legacy/monopole_cuda_kernel.cpp:            cuda_p2m_interaction_non_rho, grid_spec, threads_per_block, args, 0);
src/options_processing.cpp:	("number_gpus", po::value<size_t>(&(opts().number_gpus))->default_value(size_t(0)), "cuda streams per HPX locality") //
src/options_processing.cpp:	("executors_per_gpu", po::value<size_t>(&(opts().executors_per_gpu))->default_value(size_t(0)), "cuda streams per GPU (per locality)") //
src/options_processing.cpp:	("max_gpu_executor_queue_length", po::value<size_t>(&(opts().max_gpu_executor_queue_length))->default_value(size_t(5)), "How many launches should be buffered before using the CPU") //
src/options_processing.cpp:("polling-threads", po::value<int>(&(opts().polling_threads))->default_value(0), "Enable dedicated HPX thread pool for cuda/network polling using N threads!") //
src/options_processing.cpp:	("root_node_on_device", po::value<bool>(&(opts().root_node_on_device))->default_value(true), "Offload root node gravity kernels to the GPU? May degrade performance given weak GPUs") //
src/options_processing.cpp:    if (opts().executors_per_gpu > 0 && opts().number_gpus == 0) {
src/options_processing.cpp:        opts().number_gpus = 1;
src/options_processing.cpp:		SHOW(number_gpus);
src/options_processing.cpp:		SHOW(executors_per_gpu);
src/options_processing.cpp:		SHOW(max_gpu_executor_queue_length);
src/options_processing.cpp:        if (opts().multipole_device_kernel_type == interaction_device_kernel_type::CUDA &&
src/options_processing.cpp:            << " multipole cuda device kernels cannot be mixed with the respective kokkos host kernel!" << std::endl
src/options_processing.cpp:            << "(or move to kokkos device kernel with --multipole_device_kernel_type=KOKKOS_CUDA)" << std::endl;
src/options_processing.cpp:        if (opts().monopole_device_kernel_type == interaction_device_kernel_type::CUDA &&
src/options_processing.cpp:            << " monopole cuda device kernels cannot be mixed with the respective kokkos host kernel!" << std::endl
src/options_processing.cpp:            << "(or move to kokkos device kernel with --monopole_device_kernel_type=KOKKOS_CUDA)" << std::endl;
src/options_processing.cpp:#ifndef OCTOTIGER_HAVE_CUDA
src/options_processing.cpp:        if (opts().monopole_device_kernel_type == interaction_device_kernel_type::CUDA) {
src/options_processing.cpp:            std::cerr << "Octotiger has been compiled without CUDA support!" 
src/options_processing.cpp:        if (opts().multipole_device_kernel_type == interaction_device_kernel_type::CUDA) {
src/options_processing.cpp:            std::cerr << "Octotiger has been compiled without CUDA support! " <<
src/options_processing.cpp:    if (opts().executors_per_gpu < 1 && (opts().monopole_device_kernel_type != OFF ||
src/options_processing.cpp:        std::cerr << "You have chosen an GPU kernel, however, you did not specify --executors_per_gpu > 0" << std::endl
src/options_processing.cpp:        << " Choose a different kernel or add at least one or more executors via --executors_per_gpu=X" << std::endl;
src/options_processing.cpp:        std::cerr << "minimum value for --max_kernels_fused is 1 when a GPU kernel is active!" << std::endl;
src/cuda_util/cuda_scheduler.cpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
src/cuda_util/cuda_scheduler.cpp:#include "octotiger/cuda_util/cuda_scheduler.hpp"
src/cuda_util/cuda_scheduler.cpp:#include "octotiger/monopole_interactions/legacy/monopole_cuda_kernel.hpp"
src/cuda_util/cuda_scheduler.cpp:#include "octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp"
src/cuda_util/cuda_scheduler.cpp:#ifdef OCTOTIGER_HAVE_CUDA
src/cuda_util/cuda_scheduler.cpp:    void cuda_error(cudaError_t err) {
src/cuda_util/cuda_scheduler.cpp:        if (err != cudaSuccess) {
src/cuda_util/cuda_scheduler.cpp:            temp << "CUDA function returned error code " << cudaGetErrorString(err);
src/cuda_util/cuda_scheduler.cpp:        std::size_t gpu_count = opts().number_gpus;
src/cuda_util/cuda_scheduler.cpp:        // Move data to constant memory, once per gpu
src/cuda_util/cuda_scheduler.cpp:        for (std::size_t gpu_id = 0; gpu_id < gpu_count; ++gpu_id) {
src/cuda_util/cuda_scheduler.cpp:            std::cout << "Init FMM GPU constants on device " << gpu_id << " ..." << std::endl;
src/cuda_util/cuda_scheduler.cpp:                gpu_id, std::move(stencil_masks), std::move(four_constants_tmp));
src/cuda_util/cuda_scheduler.cpp:            multipole_interactions::init_stencil(gpu_id, std::move(multipole_stencil_masks),
src/cuda_util/cuda_scheduler.cpp:            cudaDeviceSynchronize();
src/grid.cpp://#include "octotiger/unitiger/hydro_impl/hydro_cuda_interface.hpp"
src/grid.cpp:  const size_t device_queue_length = opts().max_gpu_executor_queue_length;
src/multipole_interactions/multipole_kernel_interface.cpp:#include "octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp"
src/multipole_interactions/multipole_kernel_interface.cpp:#if defined(OCTOTIGER_HAVE_INTEL_GPU_WORKAROUND)
src/multipole_interactions/multipole_kernel_interface.cpp:#pragma message "SYCL builds without OCTOTIGER_WITH_INTEL_GPU_WORKAROUND=ON may break on Intel GPUs"
src/multipole_interactions/multipole_kernel_interface.cpp:#if defined(KOKKOS_ENABLE_CUDA)
src/multipole_interactions/multipole_kernel_interface.cpp:using device_executor = hpx::kokkos::cuda_executor;
src/multipole_interactions/multipole_kernel_interface.cpp:                if (device_type == interaction_device_kernel_type::KOKKOS_CUDA ||
src/multipole_interactions/multipole_kernel_interface.cpp:#if defined(OCTOTIGER_HAVE_KOKKOS) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL))
src/multipole_interactions/multipole_kernel_interface.cpp:                        stream_pool::get_next_device_id<device_executor, device_pool_strategy>(opts().number_gpus);
src/multipole_interactions/multipole_kernel_interface.cpp:                                opts().max_gpu_executor_queue_length, device_id);
src/multipole_interactions/multipole_kernel_interface.cpp:                        octotiger::fmm::multipole_kokkos_gpu_subgrids_launched++;
src/multipole_interactions/multipole_kernel_interface.cpp:                if (device_type == interaction_device_kernel_type::CUDA) {
src/multipole_interactions/multipole_kernel_interface.cpp:#ifdef OCTOTIGER_HAVE_CUDA
src/multipole_interactions/multipole_kernel_interface.cpp:                    cuda_multipole_interaction_interface multipole_interactor{};
src/multipole_interactions/multipole_kernel_interface.cpp:                    octotiger::fmm::multipole_cuda_gpu_subgrids_launched++;
src/multipole_interactions/multipole_kernel_interface.cpp:                    std::cerr << "Trying to call multipole CUDA kernel in a non-CUDA build! "
src/multipole_interactions/multipole_kernel_interface.cpp:                    cuda_multipole_interaction_interface multipole_interactor{};
src/multipole_interactions/multipole_kernel_interface.cpp:                    octotiger::fmm::multipole_cuda_gpu_subgrids_launched++;
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#include "octotiger/multipole_interactions/legacy/cuda_multipole_interaction_interface.hpp"
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#include "octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp"
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#include <cuda_buffer_util.hpp>
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#include <cuda_runtime.h>
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:using device_buffer_t = recycler::cuda_device_buffer<T>;
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:using host_buffer_t = std::vector<T, recycler::recycle_allocator_cuda_host<T>>;
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:using executor_t = hpx::cuda::experimental::cuda_executor;
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:using executor_t = hpx::cuda::experimental::cuda_executor;
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#define cudaLaunchKernel hipLaunchKernel
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#define cudaMemcpy hipMemcpy
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#define cudaMemcpyAsync hipMemcpyAsync
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:        cuda_multipole_interaction_interface::cuda_multipole_interaction_interface()
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:        void cuda_multipole_interaction_interface::compute_multipole_interactions(
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                stream_pool::get_next_device_id<hpx::cuda::experimental::cuda_executor,
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                    pool_strategy>(opts().number_gpus);
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                avail = stream_pool::interface_available<hpx::cuda::experimental::cuda_executor,
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                    pool_strategy>(opts().max_gpu_executor_queue_length, device_id);
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:            } else {    // run on cuda device
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                    cuda_launch_counter()++;
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                    cuda_launch_counter_non_rho()++;
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy> executor{device_id};
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                cuda_monopole_buffer_t local_monopoles(ENTRIES);
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                cuda_expansion_buffer_t local_expansions_SoA;
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                cuda_space_vector_buffer_t center_of_masses_SoA;
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                cuda_expansion_result_buffer_t potential_expansions_SoA;
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                cuda_angular_result_t angular_corrections_SoA;
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                    hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                        cudaMemcpyAsync, device_local_monopoles.device_side_buffer,
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                        local_monopoles.data(), local_monopoles_size, cudaMemcpyHostToDevice);
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                    cudaMemcpyAsync, device_local_expansions.device_side_buffer,
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                    local_expansions_SoA.get_pod(), local_expansions_size, cudaMemcpyHostToDevice);
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                    cudaMemcpyAsync, device_centers.device_side_buffer,
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                    center_of_masses_SoA.get_pod(), center_of_masses_size, cudaMemcpyHostToDevice);
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                        launch_multipole_root_rho_cuda_kernel_post(
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                        hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                            cudaMemcpyAsync, angular_corrections_SoA.get_pod(),
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                            cudaMemcpyDeviceToHost);
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                        launch_multipole_root_non_rho_cuda_kernel_post(
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                        launch_multipole_rho_cuda_kernel_post(
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                        hpx::apply(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                            cudaMemcpyAsync, angular_corrections_SoA.get_pod(),
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                            cudaMemcpyDeviceToHost);
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:#if defined(OCTOTIGER_HAVE_CUDA)
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                        launch_multipole_non_rho_cuda_kernel_post(
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                auto fut = hpx::async(static_cast<hpx::cuda::experimental::cuda_executor>(executor),
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                    cudaMemcpyAsync, potential_expansions_SoA.get_pod(),
src/multipole_interactions/legacy/cuda_multipole_interaction_interface.cpp:                    cudaMemcpyDeviceToHost);
src/multipole_interactions/legacy/multipole_interaction_interface.cpp:        size_t& multipole_interaction_interface::cuda_launch_counter() {
src/multipole_interactions/legacy/multipole_interaction_interface.cpp:            static thread_local size_t cuda_launch_counter_ = 0;
src/multipole_interactions/legacy/multipole_interaction_interface.cpp:            return cuda_launch_counter_;
src/multipole_interactions/legacy/multipole_interaction_interface.cpp:        size_t& multipole_interaction_interface::cuda_launch_counter_non_rho() {
src/multipole_interactions/legacy/multipole_interaction_interface.cpp:            static thread_local size_t cuda_launch_counter_non_rho_ = 0;
src/multipole_interactions/legacy/multipole_interaction_interface.cpp:            return cuda_launch_counter_non_rho_;
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:#if defined(OCTOTIGER_HAVE_CUDA) || defined(OCTOTIGER_HAVE_HIP)
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:#include "octotiger/multipole_interactions/legacy/multipole_cuda_kernel.hpp"
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:#include "octotiger/cuda_util/cuda_scheduler.hpp"
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:#define cudaSetDevice hipSetDevice
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:#define cudaMemcpyToSymbol hipMemcpyToSymbol
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        __host__ void init_stencil(size_t gpu_id, std::unique_ptr<bool[]> multipole_stencil_masks,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            cudaSetDevice(gpu_id);
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            cudaMemcpyToSymbol(device_constant_stencil_masks, multipole_stencil_masks.get(),
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            cudaMemcpyToSymbol(device_stencil_indicator_const, multipole_indicators.get(),
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            cudaMemcpyToSymbol(device_constant_stencil_masks, stencil_masks, full_stencil_size);
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            cudaMemcpyToSymbol(device_stencil_indicator_const, indicator, indicator_size);
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        __global__ void cuda_multipole_interactions_kernel_rho(
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        __global__ void __launch_bounds__(INX* INX, 2) cuda_multipole_interactions_kernel_rho(
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        __global__ void cuda_sum_multipole_angular_corrections_results(
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            hipLaunchKernelGGL(cuda_multipole_interactions_kernel_rho, grid_spec, threads_per_block,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            hipLaunchKernelGGL(cuda_sum_multipole_angular_corrections_results, grid_spec,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        void launch_multipole_rho_cuda_kernel_post(
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            auto launch_function = cudaLaunchKernel<decltype(cuda_multipole_interactions_kernel_rho)>;
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:                cuda_multipole_interactions_kernel_rho, grid_spec, threads_per_block, args, 0);
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            auto launch_function = cudaLaunchKernel<decltype(cuda_sum_multipole_angular_corrections_results)>;
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:                cuda_sum_multipole_angular_corrections_results, grid_spec, threads_per_block, args,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        __global__ void cuda_multipole_interactions_kernel_root_rho(
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        __global__ void __launch_bounds__(INX* INX, 2) cuda_multipole_interactions_kernel_root_rho(
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            hipLaunchKernelGGL(cuda_multipole_interactions_kernel_root_rho, grid_spec,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        void launch_multipole_root_rho_cuda_kernel_post(
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            auto launch_function = cudaLaunchKernel<decltype(cuda_multipole_interactions_kernel_root_rho)>;
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:                cuda_multipole_interactions_kernel_root_rho, grid_spec, threads_per_block, args, 0);
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        __global__ void cuda_multipole_interactions_kernel_non_rho(
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        __global__ void __launch_bounds__(INX* INX, 2) cuda_multipole_interactions_kernel_non_rho(
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        __global__ void cuda_sum_multipole_potential_expansions_results(
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            hipLaunchKernelGGL(cuda_multipole_interactions_kernel_non_rho, grid_spec,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            hipLaunchKernelGGL(cuda_sum_multipole_potential_expansions_results, grid_spec,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        void launch_multipole_non_rho_cuda_kernel_post(
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            auto launch_function = cudaLaunchKernel<decltype(cuda_multipole_interactions_kernel_non_rho)>;
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:                cuda_multipole_interactions_kernel_non_rho, grid_spec, threads_per_block, args, 0);
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            auto launch_function = cudaLaunchKernel<decltype(cuda_sum_multipole_potential_expansions_results)>;
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:                cuda_sum_multipole_potential_expansions_results, grid_spec, threads_per_block, args,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        cuda_multipole_interactions_kernel_root_non_rho(const double* center_of_masses,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            hipLaunchKernelGGL(cuda_multipole_interactions_kernel_root_non_rho, grid_spec,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:        void launch_multipole_root_non_rho_cuda_kernel_post(
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            stream_interface<hpx::cuda::experimental::cuda_executor, pool_strategy>& executor,
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:            auto launch_function = cudaLaunchKernel<decltype(cuda_multipole_interactions_kernel_root_non_rho)>;
src/multipole_interactions/legacy/multipole_cuda_kernel.cpp:                cuda_multipole_interactions_kernel_root_non_rho, grid_spec, threads_per_block, args,

```
