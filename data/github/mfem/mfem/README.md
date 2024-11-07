# https://github.com/mfem/mfem

```console
miniapps/gslib/makefile:   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
miniapps/dpg/makefile:   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
miniapps/hooke/kernels/kernel_helpers.hpp:// Should be moved in backends/cuda/hip header files.
miniapps/hooke/kernels/kernel_helpers.hpp:#if defined(__CUDA_ARCH__)
miniapps/hooke/kernels/kernel_helpers.hpp: * @note TODO: Does not make use of shared memory on the GPU.
miniapps/tools/makefile:   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
miniapps/electromagnetics/makefile:   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
miniapps/nurbs/nurbs_ex24.cpp://               nurbs_ex24 -m ../../data/escher.mesh -pa -d cuda
miniapps/nurbs/nurbs_ex24.cpp://               nurbs_ex24 -m ../../data/escher.mesh -pa -d raja-cuda
miniapps/nurbs/nurbs_ex24.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
miniapps/nurbs/nurbs_ex24.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
miniapps/nurbs/nurbs_patch_ex1.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
miniapps/nurbs/nurbs_patch_ex1.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
miniapps/nurbs/nurbs_ex5.cpp://               nurbs_ex5 -m ../../data/escher.mesh -pa -d cuda
miniapps/nurbs/nurbs_ex5.cpp://               nurbs_ex5 -m ../../data/escher.mesh -pa -d raja-cuda
miniapps/nurbs/nurbs_ex5.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
miniapps/nurbs/nurbs_ex5.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
miniapps/nurbs/nurbs_solenoidal.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
miniapps/nurbs/nurbs_solenoidal.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
miniapps/nurbs/nurbs_ex3.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
miniapps/nurbs/nurbs_ex3.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
miniapps/common/makefile:   ifeq ($(MFEM_USE_CUDA),YES)
miniapps/common/makefile:      XLINKER = $(CUDA_XLINKER)
miniapps/common/CMakeLists.txt:if (MFEM_USE_CUDA)
miniapps/common/CMakeLists.txt:  set_property(SOURCE ${SRCS} PROPERTY LANGUAGE CUDA)
miniapps/navier/navier_solver.hpp: * High-order matrix-free incompressible flow solvers with GPU acceleration and
miniapps/performance/makefile:ifeq (YES,$$(MFEM_USE_CUDA))
miniapps/performance/makefile:ifneq (YES,$(MFEM_USE_CUDA))
miniapps/performance/CMakeLists.txt:  if (NOT MFEM_USE_CUDA)
miniapps/shifted/distance.cpp:   // Enable hardware devices such as GPUs, and programming models such as CUDA,
miniapps/shifted/makefile:   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
miniapps/shifted/CMakeLists.txt:    if (HYPRE_USING_CUDA OR HYPRE_USING_HIP)
miniapps/shifted/diffusion.cpp:#ifdef HYPRE_USING_GPU
miniapps/shifted/diffusion.cpp:        << "is NOT supported with the GPU version of hypre.\n\n";
miniapps/shifted/diffusion.cpp:   // Enable hardware devices such as GPUs, and programming models such as CUDA,
miniapps/solvers/block-solvers.cpp:#ifdef HYPRE_USING_GPU
miniapps/solvers/block-solvers.cpp:             << "is NOT supported with the GPU version of hypre.\n\n";
miniapps/solvers/plor_solvers.cpp://    mpirun -np 4 plor_solvers -m ../../data/fichera.mesh -fe h -d cuda
miniapps/solvers/plor_solvers.cpp://    mpirun -np 4 plor_solvers -m ../../data/fichera.mesh -fe n -d cuda
miniapps/solvers/plor_solvers.cpp://    mpirun -np 4 plor_solvers -m ../../data/fichera.mesh -fe r -d cuda
miniapps/solvers/plor_solvers.cpp://    mpirun -np 4 plor_solvers -m ../../data/fichera.mesh -fe l -d cuda
miniapps/solvers/CMakeLists.txt:    if (NOT (HYPRE_USING_CUDA OR HYPRE_USING_HIP))
miniapps/solvers/lor_elast.cpp://    precondition vector valued PDEs, such as elasticity, on GPUs.
miniapps/solvers/lor_elast.cpp://    prohibitive on GPUs. An effective preconditioning strategy for materials
miniapps/solvers/lor_elast.cpp://    but this is still prohibitive for high order discretizations on GPUs. This
miniapps/solvers/lor_elast.cpp://    matrix assembly are supported on GPUs.
miniapps/solvers/lor_elast.cpp://    latter may only work for order 1 on GPUs.
miniapps/solvers/lor_elast.cpp://       ./lor_elast --device cuda -m ../../data/beam-hex.mesh -l 4 -o 2 -pa
miniapps/solvers/lor_elast.cpp://       ./lor_elast --device cuda -m ../../data/beam-hex.mesh -l 4 -o 2 -pa -pv
miniapps/solvers/lor_elast.cpp://       ./lor_elast --device cuda -m ../../data/beam-hex.mesh -l 4 -o 2 -pa -ss
miniapps/solvers/lor_elast.cpp://       ./lor_elast --device cuda -m ../../data/beam-hex.mesh -l 4 -o 2 -pa -ca
miniapps/solvers/lor_elast.cpp://       ./lor_elast --device cuda -m ../../data/beam-hex.mesh -l 5 -ca
miniapps/solvers/lor_elast.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
miniapps/solvers/lor_elast.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
miniapps/solvers/lor_solvers.cpp://    lor_solvers -fe h -d cuda
miniapps/solvers/lor_solvers.cpp://    lor_solvers -fe n -d cuda
miniapps/solvers/lor_solvers.cpp://    lor_solvers -fe r -d cuda
miniapps/solvers/lor_solvers.cpp://    lor_solvers -fe l -d cuda
miniapps/solvers/README:on GPUs. The miniapp supports partial assembly, and the low-order refined
miniapps/solvers/README:preconditioner can be assembled entirely on the GPU.
miniapps/solvers/README:prohibitive on GPUs. An effective preconditioning strategy for materials
miniapps/solvers/README:but this is still prohibitive for high order discretizations on GPUs. This
miniapps/meshing/pmesh-optimizer.cpp://   Adapted discrete size 3D with PA on device (requires CUDA):
miniapps/meshing/pmesh-optimizer.cpp://   * mpirun -n 4 pmesh-optimizer -m cube.mesh -o 3 -rs 3 -mid 321 -tid 5 -ls 3 -nor -lc 0.1 -pa -d cuda
miniapps/meshing/pmesh-optimizer.cpp://     (requires CUDA):
miniapps/meshing/pmesh-optimizer.cpp://   * mpirun -np 4 pmesh-optimizer -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 3 -art 1 -bnd -qt 1 -qo 8 -d cuda
miniapps/meshing/mesh-optimizer.cpp://   Adapted discrete size 3D with PA on device (requires CUDA):
miniapps/meshing/mesh-optimizer.cpp://   * mesh-optimizer -m cube.mesh -o 3 -rs 3 -mid 321 -tid 5 -ls 3 -nor -lc 0.1 -pa -d cuda
miniapps/meshing/mesh-optimizer.cpp://     (requires CUDA):
miniapps/meshing/mesh-optimizer.cpp://   * mesh-optimizer -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 3 -art 1 -bnd -qt 1 -qo 8 -d cuda
miniapps/meshing/makefile:   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
miniapps/meshing/CMakeLists.txt:    if (HYPRE_USING_CUDA OR HYPRE_USING_HIP)
miniapps/meshing/pminimal-surface.cpp://               mpirun -np 4 pminimal-surface -d  cuda
miniapps/meshing/pminimal-surface.cpp://               mpirun -np 4 pminimal-surface -d  cuda -a
miniapps/meshing/pminimal-surface.cpp://               mpirun -np 4 pminimal-surface -d  cuda -c
miniapps/meshing/pminimal-surface.cpp://               mpirun -np 4 pminimal-surface -d  cuda -c -a
miniapps/meshing/pminimal-surface.cpp://               mpirun -np 4 pminimal-surface -d  cuda -no-pa
miniapps/meshing/pminimal-surface.cpp://               mpirun -np 4 pminimal-surface -d  cuda -no-pa -a
miniapps/meshing/pminimal-surface.cpp://               mpirun -np 4 pminimal-surface -d  cuda -no-pa -c
miniapps/meshing/pminimal-surface.cpp://               mpirun -np 4 pminimal-surface -d  cuda -no-pa -c -a
miniapps/meshing/minimal-surface.cpp://               minimal-surface -d  cuda
miniapps/meshing/minimal-surface.cpp://               minimal-surface -d  cuda -a
miniapps/meshing/minimal-surface.cpp://               minimal-surface -d  cuda -c
miniapps/meshing/minimal-surface.cpp://               minimal-surface -d  cuda -c -a
miniapps/meshing/minimal-surface.cpp://               minimal-surface -d  cuda -no-pa
miniapps/meshing/minimal-surface.cpp://               minimal-surface -d  cuda -no-pa -a
miniapps/meshing/minimal-surface.cpp://               minimal-surface -d  cuda -no-pa -c
miniapps/meshing/minimal-surface.cpp://               minimal-surface -d  cuda -no-pa -c -a
miniapps/meshing/pmesh-fitting.cpp:#ifdef HYPRE_USING_GPU
miniapps/meshing/pmesh-fitting.cpp:   cout << "\nThis miniapp is NOT supported with the GPU version of hypre.\n\n";
miniapps/toys/makefile:   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
config/test.mk:ifeq ($(MFEM_USE_CUDA),YES)
config/test.mk:.PHONY: test-par-YES-cuda test-par-NO-cuda test-ser-cuda test-par-cuda test-cuda
config/test.mk:test-par-YES: test-par-YES-cuda
config/test.mk:test-par-NO:  test-par-NO-cuda
config/test.mk:test-par-YES-cuda: test-par-cuda test-ser-cuda
config/test.mk:test-par-NO-cuda:  test-ser-cuda
config/test.mk:test-ser-cuda: $(SEQ_DEVICE_$(MFEM_TESTS):=-test-seq-cuda)
config/test.mk:test-par-cuda: $(PAR_DEVICE_$(MFEM_TESTS):=-test-par-cuda)
config/test.mk:test-cuda: test-par-$(MFEM_USE_MPI)-cuda clean-exec
config/XSDKDefaults.cmake:IF (DEFINED TPL_ENABLE_CUDA)
config/XSDKDefaults.cmake:  SET(MFEM_USE_CUDA ${TPL_ENABLE_CUDA} CACHE BOOL "Enable CUDA" FORCE)
config/defaults.cmake:option(MFEM_USE_CUDA "Enable CUDA" OFF)
config/defaults.cmake:# Set the target CUDA architecture
config/defaults.cmake:set(CUDA_ARCH "sm_60" CACHE STRING "Target CUDA architecture.")
config/defaults.cmake:# CUDA and HIP dependencies for HYPRE are handled in FindHYPRE.cmake.
config/config.hpp.in:// Build the NVIDIA GPU/CUDA-enabled version of the MFEM library.
config/config.hpp.in:// Requires a CUDA compiler (nvcc).
config/config.hpp.in:// #define MFEM_USE_CUDA
config/config.hpp.in:// Build the AMD GPU/HIP-enabled version of the MFEM library.
config/sample-runs.sh:       mfem_config+=" MFEM_USE_CUDA=YES MFEM_USE_OPENMP=YES"
config/sample-runs.sh:      if [ -n "${CUDA_ARCH}" ]; then
config/sample-runs.sh:         mfem_config+=" CUDA_ARCH=${CUDA_ARCH}"
config/docker/Dockerfile.base:FROM ghcr.io/rse-ops/cuda-ubuntu-20.04:cuda-11.0.3
config/cmake/MFEMConfig.cmake.in:set(MFEM_USE_CUDA @MFEM_USE_CUDA@)
config/cmake/modules/FindHYPRE.cmake:#   - HYPRE_USING_CUDA (internal)
config/cmake/modules/FindHYPRE.cmake:  if (HYPRE_USING_CUDA)
config/cmake/modules/FindHYPRE.cmake:    find_package(CUDAToolkit REQUIRED)
config/cmake/modules/FindHYPRE.cmake:  CHECK_BUILD HYPRE_USING_CUDA FALSE
config/cmake/modules/FindHYPRE.cmake:#undef HYPRE_USING_CUDA
config/cmake/modules/FindHYPRE.cmake:#ifndef HYPRE_USING_CUDA
config/cmake/modules/FindHYPRE.cmake:#error HYPRE is built without CUDA.
config/cmake/modules/FindHYPRE.cmake:if (HYPRE_FOUND AND HYPRE_USING_CUDA)
config/cmake/modules/FindHYPRE.cmake:  find_package(CUDAToolkit REQUIRED)
config/cmake/modules/FindHYPRE.cmake:  get_target_property(CUSPARSE_LIBRARIES CUDA::cusparse LOCATION)
config/cmake/modules/FindHYPRE.cmake:  get_target_property(CURAND_LIBRARIES CUDA::curand LOCATION)
config/cmake/modules/FindHYPRE.cmake:  get_target_property(CUBLAS_LIBRARIES CUDA::cublas LOCATION)
config/cmake/modules/FindSUNDIALS.cmake:  ADD_COMPONENT NVector_Cuda
config/cmake/modules/FindSUNDIALS.cmake:    "include" nvector/nvector_cuda.h "lib" sundials_nveccuda
config/cmake/modules/FindMAGMA.cmake:if (MAGMA_FOUND AND MFEM_USE_CUDA)
config/cmake/modules/FindMAGMA.cmake:  get_target_property(CUSPARSE_LIBRARIES CUDA::cusparse LOCATION)
config/cmake/modules/FindMAGMA.cmake:  get_target_property(CUBLAS_LIBRARIES CUDA::cublas LOCATION)
config/cmake/modules/FindAMGX.cmake:  list(APPEND AMGX_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/lib${lib}${CMAKE_SHARED_LIBRARY_SUFFIX})
config/cmake/modules/MfemCmakeUtilities.cmake:  if (MFEM_USE_CUDA)
config/cmake/modules/MfemCmakeUtilities.cmake:      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
config/cmake/modules/MfemCmakeUtilities.cmake:    # If CUDA is enabled, tag source files to be compiled with nvcc.
config/cmake/modules/MfemCmakeUtilities.cmake:    if (MFEM_USE_CUDA)
config/cmake/modules/MfemCmakeUtilities.cmake:      set_source_files_properties(${SRC_FILE} PROPERTIES LANGUAGE CUDA)
config/cmake/modules/MfemCmakeUtilities.cmake:  # If CUDA is enabled, tag source files to be compiled with nvcc.
config/cmake/modules/MfemCmakeUtilities.cmake:  if (MFEM_USE_CUDA)
config/cmake/modules/MfemCmakeUtilities.cmake:    set_source_files_properties(${MAIN_LIST} ${EXTRA_SOURCES_LIST} PROPERTIES LANGUAGE CUDA)
config/cmake/modules/MfemCmakeUtilities.cmake:      MFEM_USE_HIOP MFEM_USE_GSLIB MFEM_USE_CUDA MFEM_USE_HIP MFEM_USE_RAJA
config/cmake/modules/MfemCmakeUtilities.cmake:  # TODO: Add support for MFEM_USE_CUDA=YES
config/cmake/config.hpp.in:// Build the NVIDIA GPU/CUDA-enabled version of the MFEM library.
config/cmake/config.hpp.in:// Requires a CUDA compiler (nvcc).
config/cmake/config.hpp.in:#cmakedefine MFEM_USE_CUDA
config/cmake/config.hpp.in:// Build the AMD GPU/HIP-enabled version of the MFEM library.
config/config.hpp:#if (defined(MFEM_USE_CUDA) && defined(__CUDACC__)) || \
config/defaults.mk:# CUDA configuration options
config/defaults.mk:# If you set MFEM_USE_ENZYME=YES, CUDA_CXX has to be configured to use cuda with
config/defaults.mk:CUDA_CXX = nvcc
config/defaults.mk:CUDA_ARCH = sm_60
config/defaults.mk:CUDA_FLAGS = -x=cu --expt-extended-lambda -arch=$(CUDA_ARCH)
config/defaults.mk:# Prefixes for passing flags to the host compiler and linker when using CUDA_CXX
config/defaults.mk:CUDA_XCOMPILER = -Xcompiler=
config/defaults.mk:CUDA_XLINKER   = -Xlinker=
config/defaults.mk:# The HIP_ARCH option specifies the AMD GPU processor, similar to CUDA_ARCH. For
config/defaults.mk:MFEM_USE_CUDA          = NO
config/defaults.mk:# ROCM/HIP directory such that ROCM/HIP libraries like rocsparse and rocrand are
config/defaults.mk:# the form /opt/rocm-X.Y.Z which is called ROCM_PATH by hipconfig.
config/defaults.mk:      HIP_DIR := $(shell hipconfig --rocmpath 2> /dev/null)
config/defaults.mk:ifeq (YES,$(MFEM_USE_CUDA))
config/defaults.mk:   # This is only necessary when hypre is built with cuda:
config/defaults.mk:ifeq ($(MFEM_USE_CUDA),YES)
config/defaults.mk:   SUNDIALS_LIB += -lsundials_nveccuda
config/defaults.mk:# CUDA library configuration
config/defaults.mk:CUDA_OPT =
config/defaults.mk:CUDA_LIB = -lcusparse -lcublas
config/config.mk.in:MFEM_USE_CUDA          = @MFEM_USE_CUDA@
fem/tmop/tmop_pa_h3d.cpp:   // This kernel uses its own CUDA/ROCM limits: runtime values:
fem/tmop/tmop_pa_h3d.cpp:   if (Device::Allows(Backend::CUDA_MASK))
fem/tmop/tmop_pa_h3d.cpp:      // This kernel uses its own CUDA/ROCM limits: compile time values:
fem/tmop/tmop_pa_h3d.cpp:#if defined(__CUDA_ARCH__)
fem/tmop/tmop_pa_da3.cpp:   MFEM_VERIFY(MFEM_CUDA_BLOCKS==256,"");
fem/tmop/tmop_pa_da3.cpp:      MFEM_SHARED real_t min_size[MFEM_CUDA_BLOCKS];
fem/tmop/tmop_pa_da3.cpp:      MFEM_FOREACH_THREAD(t,x,MFEM_CUDA_BLOCKS) { min_size[t] = infinity; }
fem/tmop/tmop_pa_da3.cpp:      for (int wrk = MFEM_CUDA_BLOCKS >> 1; wrk > 0; wrk >>= 1)
fem/tmop/tmop_pa_da3.cpp:         MFEM_FOREACH_THREAD(t,x,MFEM_CUDA_BLOCKS)
fem/pfespace.hpp:   bool mpi_gpu_aware;
fem/dgmassinv.hpp:/// supports execution on device (GPU).
fem/pfespace.cpp:     mpi_gpu_aware(Device::GetGPUAwareMPI())
fem/pfespace.cpp:   if (mpi_gpu_aware) { MFEM_STREAM_SYNC; }
fem/pfespace.cpp:            auto send_buf = mpi_gpu_aware ? shr_buf.Read() : shr_buf.HostRead();
fem/pfespace.cpp:            auto recv_buf = mpi_gpu_aware ? ext_buf.Write() : ext_buf.HostWrite();
fem/pfespace.cpp:   if (mpi_gpu_aware) { MFEM_STREAM_SYNC; }
fem/pfespace.cpp:            auto send_buf = mpi_gpu_aware ? ext_buf.Read() : ext_buf.HostRead();
fem/pfespace.cpp:            auto recv_buf = mpi_gpu_aware ? shr_buf.Write() : shr_buf.HostWrite();
fem/pgridfunc.hpp:   //TODO: Use temporary memory to avoid CUDA malloc allocation cost.
fem/pgridfunc.cpp:   bool mpi_gpu_aware = Device::GetGPUAwareMPI();
fem/pgridfunc.cpp:   auto send_data_ptr = mpi_gpu_aware ? send_data.Read() : send_data.HostRead();
fem/pgridfunc.cpp:   auto face_nbr_data_ptr = mpi_gpu_aware ? face_nbr_data.Write() :
fem/pgridfunc.cpp:   if (mpi_gpu_aware) { MFEM_STREAM_SYNC; }
fem/lor/lor_batched.hpp:   /// @name GPU kernel functions
fem/integ/bilininteg_hcurl_kernels.hpp:      MFEM_ABORT_KERNEL("This kernel should only be used on GPU.");
fem/integ/bilininteg_hcurl_kernels.hpp:      MFEM_ABORT_KERNEL("This kernel should only be used on GPU.");
fem/integ/bilininteg_hcurl_kernels.hpp:      MFEM_ABORT_KERNEL("This kernel should only be used on GPU.");
fem/integ/bilininteg_mass_kernels.cpp:   if (!Device::Allows(Backend::OCCA_CUDA))
fem/integ/bilininteg_mass_kernels.cpp:      static occa_kernel_t OccaMassApply2D_gpu;
fem/integ/bilininteg_mass_kernels.cpp:      if (OccaMassApply2D_gpu.find(id) == OccaMassApply2D_gpu.end())
fem/integ/bilininteg_mass_kernels.cpp:         const occa::kernel MassApply2D_GPU =
fem/integ/bilininteg_mass_kernels.cpp:                                        "MassApply2D_GPU", props);
fem/integ/bilininteg_mass_kernels.cpp:         OccaMassApply2D_gpu.emplace(id, MassApply2D_GPU);
fem/integ/bilininteg_mass_kernels.cpp:      OccaMassApply2D_gpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
fem/integ/bilininteg_mass_kernels.cpp:   if (!Device::Allows(Backend::OCCA_CUDA))
fem/integ/bilininteg_mass_kernels.cpp:      static occa_kernel_t OccaMassApply3D_gpu;
fem/integ/bilininteg_mass_kernels.cpp:      if (OccaMassApply3D_gpu.find(id) == OccaMassApply3D_gpu.end())
fem/integ/bilininteg_mass_kernels.cpp:         const occa::kernel MassApply3D_GPU =
fem/integ/bilininteg_mass_kernels.cpp:                                        "MassApply3D_GPU", props);
fem/integ/bilininteg_mass_kernels.cpp:         OccaMassApply3D_gpu.emplace(id, MassApply3D_GPU);
fem/integ/bilininteg_mass_kernels.cpp:      OccaMassApply3D_gpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
fem/integ/bilininteg_elasticity_kernels.hpp:/// Mainly intended to be used for order 1 elements on gpus to enable
fem/integ/bilininteg_diffusion_kernels.cpp:   if (!Device::Allows(Backend::OCCA_CUDA))
fem/integ/bilininteg_diffusion_kernels.cpp:      static occa_kernel_t OccaDiffApply2D_gpu;
fem/integ/bilininteg_diffusion_kernels.cpp:      if (OccaDiffApply2D_gpu.find(id) == OccaDiffApply2D_gpu.end())
fem/integ/bilininteg_diffusion_kernels.cpp:         const occa::kernel DiffusionApply2D_GPU =
fem/integ/bilininteg_diffusion_kernels.cpp:                                        "DiffusionApply2D_GPU", props);
fem/integ/bilininteg_diffusion_kernels.cpp:         OccaDiffApply2D_gpu.emplace(id, DiffusionApply2D_GPU);
fem/integ/bilininteg_diffusion_kernels.cpp:      OccaDiffApply2D_gpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_D, o_X, o_Y);
fem/integ/bilininteg_diffusion_kernels.cpp:   if (!Device::Allows(Backend::OCCA_CUDA))
fem/integ/bilininteg_diffusion_kernels.cpp:      static occa_kernel_t OccaDiffApply3D_gpu;
fem/integ/bilininteg_diffusion_kernels.cpp:      if (OccaDiffApply3D_gpu.find(id) == OccaDiffApply3D_gpu.end())
fem/integ/bilininteg_diffusion_kernels.cpp:         const occa::kernel DiffusionApply3D_GPU =
fem/integ/bilininteg_diffusion_kernels.cpp:                                        "DiffusionApply3D_GPU", props);
fem/integ/bilininteg_diffusion_kernels.cpp:         OccaDiffApply3D_gpu.emplace(id, DiffusionApply3D_GPU);
fem/integ/bilininteg_diffusion_kernels.cpp:      OccaDiffApply3D_gpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_D, o_X, o_Y);
fem/occa.okl:@kernel void DiffusionApply2D_GPU(const int NE,
fem/occa.okl:@kernel void DiffusionApply3D_GPU(const int NE,
fem/occa.okl:@kernel void MassApply2D_GPU(const int NE,
fem/occa.okl:@kernel void MassApply3D_GPU(const int NE,
fem/nonlinearform.hpp:       support both CPU and GPU backends and utilize features such as fast
fem/nonlinearform.hpp:       also uses partial assembly with support for CPU and GPU backends.
fem/nonlinearform.hpp:       support both CPU and GPU backends and utilize features such as fast
fem/restriction.hpp:       emulate SetSubVector and its transpose on GPUs. This method is running on
fem/ceed/solvers/algebraic.cpp:   if (!Device::Allows(Backend::CUDA) || mem != CEED_MEM_DEVICE)
fem/ceed/solvers/algebraic.cpp:      if (ilevel == 0 && !Device::Allows(Backend::CUDA))
tests/unit/miniapps/test_tmop_pa.cpp:   if (HypreUsingGPU())
tests/unit/miniapps/test_tmop_pa.cpp:           << "is NOT supported with the GPU version of hypre.\n\n";
tests/unit/miniapps/test_sedov.cpp:   bool gpu_aware_mpi = false;
tests/unit/miniapps/test_sedov.cpp:   args.AddOption(&gpu_aware_mpi, "-gam", "--gpu-aware-mpi", "-no-gam",
tests/unit/miniapps/test_sedov.cpp:                  "--no-gpu-aware-mpi", "Enable GPU aware MPI communications.");
tests/unit/miniapps/test_sedov.cpp:   if (HypreUsingGPU() && !strcmp(MFEM_SEDOV_DEVICE, "debug"))
tests/unit/miniapps/test_sedov.cpp:           << "is NOT supported with the GPU version of hypre.\n\n";
tests/unit/catch.hpp:#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__) && !defined(__LCC__)
tests/unit/catch.hpp:#  if !defined(__ibmxl__) && !defined(__CUDACC__)
tests/unit/catch.hpp:        // Code accompanying the article "Approximating the erfinv function" in GPU Computing Gems, Volume 2
tests/unit/fem/test_assemblediagonalpa.cpp:          "[CUDA][PartialAssembly][AssembleDiagonal]")
tests/unit/fem/test_linearform_ext.cpp:TEST_CASE("Linear Form Extension", "[LinearFormExtension], [CUDA]")
tests/unit/fem/test_linearform_ext.cpp:TEST_CASE("H(div) Linear Form Extension", "[LinearFormExtension], [CUDA]")
tests/unit/fem/test_pa_coeff.cpp:          "[CUDA][PartialAssembly][Coefficient]")
tests/unit/fem/test_pa_coeff.cpp:          "[CUDA][PartialAssembly][Coefficient]")
tests/unit/fem/test_quadinterpolator.cpp:TEST_CASE("QuadratureInterpolator", "[QuadratureInterpolator][CUDA]")
tests/unit/fem/test_pa_idinterp.cpp:TEST_CASE("PAIdentityInterp", "[CUDA]")
tests/unit/fem/test_dgmassinv.cpp:TEST_CASE("DG Mass Inverse", "[CUDA]")
tests/unit/fem/test_assembly_levels.cpp:TEST_CASE("H1 Assembly Levels", "[AssemblyLevel], [PartialAssembly], [CUDA]")
tests/unit/fem/test_assembly_levels.cpp:TEST_CASE("L2 Assembly Levels", "[AssemblyLevel], [PartialAssembly], [CUDA]")
tests/unit/fem/test_assembly_levels.cpp:TEST_CASE("Serial H1 Full Assembly", "[AssemblyLevel], [CUDA]")
tests/unit/fem/test_assembly_levels.cpp:TEST_CASE("Parallel H1 Full Assembly", "[AssemblyLevel], [Parallel], [CUDA]")
tests/unit/fem/test_quadf_coef.cpp:TEST_CASE("Quadrature Function Integration", "[QuadratureFunction][CUDA]")
tests/unit/fem/test_fa_determinism.cpp:TEST_CASE("FA Determinism", "[PartialAssembly][CUDA]")
tests/unit/fem/test_lor_batched.cpp:TEST_CASE("LOR Batched H1", "[LOR][BatchedLOR][CUDA]")
tests/unit/fem/test_lor_batched.cpp:TEST_CASE("LOR Batched ND", "[LOR][BatchedLOR][CUDA]")
tests/unit/fem/test_lor_batched.cpp:TEST_CASE("LOR Batched RT", "[LOR][BatchedLOR][CUDA]")
tests/unit/fem/test_lor_batched.cpp:TEST_CASE("Parallel LOR Batched H1", "[LOR][BatchedLOR][Parallel][CUDA]")
tests/unit/fem/test_lor_batched.cpp:TEST_CASE("Parallel LOR Batched ND", "[LOR][BatchedLOR][Parallel][CUDA]")
tests/unit/fem/test_lor_batched.cpp:TEST_CASE("Parallel LOR Batched RT", "[LOR][BatchedLOR][Parallel][CUDA]")
tests/unit/fem/test_lor_batched.cpp:TEST_CASE("LOR AMS", "[LOR][BatchedLOR][AMS][Parallel][CUDA]")
tests/unit/fem/test_lor_batched.cpp:TEST_CASE("LOR ADS", "[LOR][BatchedLOR][ADS][Parallel][CUDA]")
tests/unit/fem/test_pa_kernels.cpp:TEST_CASE("PA VectorDivergence", "[PartialAssembly], [CUDA]")
tests/unit/fem/test_pa_kernels.cpp:TEST_CASE("PA Gradient", "[PartialAssembly], [CUDA]")
tests/unit/fem/test_pa_kernels.cpp:TEST_CASE("Nonlinear Convection", "[PartialAssembly], [NonlinearPA], [CUDA]")
tests/unit/fem/test_pa_kernels.cpp:TEST_CASE("PA Vector Mass", "[PartialAssembly], [VectorPA], [CUDA]")
tests/unit/fem/test_pa_kernels.cpp:TEST_CASE("PA Vector Diffusion", "[PartialAssembly], [VectorPA], [CUDA]")
tests/unit/fem/test_pa_kernels.cpp:TEST_CASE("PA Convection", "[PartialAssembly], [CUDA]")
tests/unit/fem/test_pa_kernels.cpp:TEST_CASE("PA Convection advanced", "[PartialAssembly], [MFEMData], [CUDA]")
tests/unit/fem/test_pa_kernels.cpp:TEST_CASE("PA Mass", "[PartialAssembly], [CUDA]")
tests/unit/fem/test_pa_kernels.cpp:TEST_CASE("PA Diffusion", "[PartialAssembly], [CUDA]")
tests/unit/fem/test_pa_kernels.cpp:TEST_CASE("PA Markers", "[PartialAssembly], [CUDA]")
tests/unit/fem/test_pa_kernels.cpp:TEST_CASE("PA Boundary Mass", "[PartialAssembly], [CUDA]")
tests/unit/fem/test_pa_kernels.cpp:TEST_CASE("PA DG Diffusion", "[PartialAssembly], [CUDA]")
tests/unit/fem/test_pa_kernels.cpp:TEST_CASE("Parallel PA DG Diffusion", "[PartialAssembly][Parallel][CUDA]")
tests/unit/fem/test_bilinearform.cpp:          "[CUDA]")
tests/unit/fem/test_pa_grad.cpp:TEST_CASE("PAGradient", "[CUDA]")
tests/unit/cunit_test_main.cpp:   std::cout << "\nThe serial CUDA unit tests are not supported in single"
tests/unit/cunit_test_main.cpp:   mfem::Device device("cuda");
tests/unit/cunit_test_main.cpp:   // Include only tests labeled with CUDA. Exclude parallel tests.
tests/unit/cunit_test_main.cpp:   return RunCatchSession(argc, argv, {"[CUDA]", "~[Parallel]"});
tests/unit/README.md:* `cunit_tests` if MFEM is compiled with CUDA support
tests/unit/README.md:* `sedov_tests_cpu`, `sedov_tests_debug` (and `sedov_tests_cuda` and
tests/unit/README.md:  `sedov_tests_cuda_uvm` if CUDA is enabled), testing a Sedov hydrodynamics case
tests/unit/README.md:* `tmop_pa_tests_cpu`, `tmop_pa_tests_debug` (and `tmop_pa_tests_cuda` if CUDA
tests/unit/README.md:* `[CUDA]`, which indicates that a test will be tested with the CUDA executables
tests/unit/README.md:  executables. `cunit_tests` will only run tests marked with `[CUDA]`, and its
tests/unit/README.md:  `[CUDA]` and `[Parallel]`.
tests/unit/pcunit_test_main.cpp:   std::cout << "\nThe parallel CUDA unit tests are not supported in single"
tests/unit/pcunit_test_main.cpp:   mfem::Device device("cuda");
tests/unit/pcunit_test_main.cpp:   // Include only tests that are labeled with both CUDA and Parallel.
tests/unit/pcunit_test_main.cpp:   return RunCatchSession(argc, argv, {"[CUDA]","[Parallel]"}, Root());
tests/unit/makefile:CUDA_MAIN_OBJ = cunit_test_main.o
tests/unit/makefile:PCUDA_MAIN_OBJ = pcunit_test_main.o
tests/unit/makefile:USE_CUDA := $(MFEM_USE_CUDA:NO=)
tests/unit/makefile:SEQ_SEDOV_TESTS += $(if $(USE_CUDA),sedov_tests_cuda)
tests/unit/makefile:SEQ_SEDOV_TESTS += $(if $(USE_CUDA),sedov_tests_cuda_uvm)
tests/unit/makefile:SEQ_SEDOV_CUDA_OBJ_FILES = $(if $(USE_CUDA),$(SEDOV_FILES:$(SRC)%.cpp=%.cuda.o))
tests/unit/makefile:SEQ_SEDOV_CUDA_UVM_OBJ_FILES = $(if $(USE_CUDA),$(SEDOV_FILES:$(SRC)%.cpp=%.cuda_uvm.o))
tests/unit/makefile:PAR_SEDOV_CUDA_OBJ_FILES = $(if $(USE_CUDA),$(SEDOV_FILES:$(SRC)%.cpp=%.pcuda.o))
tests/unit/makefile:PAR_SEDOV_CUDA_UVM_OBJ_FILES = $(if $(USE_CUDA),$(SEDOV_FILES:$(SRC)%.cpp=%.pcuda_uvm.o))
tests/unit/makefile:SEQ_TMOP_TESTS += $(if $(USE_CUDA),tmop_pa_tests_cuda)
tests/unit/makefile:# SEQ_TMOP_TESTS += $(if $(USE_CUDA),tmop_tests_cuda_uvm)
tests/unit/makefile:PAR_TMOP_TESTS += $(if $(USE_CUDA),ptmop_pa_tests_cuda)
tests/unit/makefile:SEQ_TMOP_CUDA_OBJ_FILES = $(if $(USE_CUDA),$(TMOP_FILES:$(SRC)%.cpp=%.cuda.o))
tests/unit/makefile:SEQ_TMOP_CUDA_UVM_OBJ_FILES = $(if $(USE_CUDA),$(TMOP_FILES:$(SRC)%.cpp=%.cuda_uvm.o))
tests/unit/makefile:PAR_TMOP_CUDA_OBJ_FILES = $(if $(USE_CUDA),$(TMOP_FILES:$(SRC)%.cpp=%.pcuda.o))
tests/unit/makefile:PAR_TMOP_CUDA_UVM_OBJ_FILES = $(if $(USE_CUDA),$(TMOP_FILES:$(SRC)%.cpp=%.pcuda_uvm.o))
tests/unit/makefile:SEQ_UNIT_TESTS = unit_tests $(if $(USE_CUDA),cunit_tests)
tests/unit/makefile:PAR_UNIT_TESTS = punit_tests $(if $(USE_CUDA),pcunit_tests)
tests/unit/makefile:cunit_tests: $(CUDA_MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
tests/unit/makefile:	$(CCC) $(CUDA_MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)
tests/unit/makefile:pcunit_tests: $(PCUDA_MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
tests/unit/makefile:	$(CCC) $(PCUDA_MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)
tests/unit/makefile:$(OBJECT_FILES) $(SEQ_MAIN_OBJ) $(PAR_MAIN_OBJ) $(CUDA_MAIN_OBJ) \
tests/unit/makefile: $(PCUDA_MAIN_OBJ) $(DEBUG_DEVICE_OBJ): %.o: $(SRC)%.cpp $(HEADER_FILES) \
tests/unit/makefile:$(eval $(call sedov_tests,cuda,CUDA,cuda))
tests/unit/makefile:$(eval $(call sedov_tests,cuda_uvm,CUDA_UVM,cuda:uvm))
tests/unit/makefile:$(eval $(call psedov_tests,cuda,CUDA,cuda))
tests/unit/makefile:$(eval $(call psedov_tests,cuda_uvm,CUDA_UVM,cuda:uvm))
tests/unit/makefile:$(eval $(call tmop_pa_tests,cuda,CUDA,cuda))
tests/unit/makefile:# $(eval $(call tmop_pa_tests,cuda_uvm,CUDA_UVM,cuda:uvm))
tests/unit/makefile:$(eval $(call ptmop_pa_tests,cuda,CUDA,cuda))
tests/unit/makefile:#$(eval $(call ptmop_pa_tests,cuda_uvm,CUDA_UVM,cuda:uvm))
tests/unit/makefile:ifeq ($(MFEM_USE_CUDA),YES)
tests/unit/makefile:	@$(call mfem-test,$<,, CEED Unit tests (cuda-ref),--device ceed-cuda:/gpu/cuda/ref $(MFEM_DATA_FLAG),SKIP-NO-VIS)
tests/unit/makefile:	@$(call mfem-test,$<,, CEED Unit tests (cuda-shared),--device ceed-cuda:/gpu/cuda/shared $(MFEM_DATA_FLAG),SKIP-NO-VIS)
tests/unit/makefile:	@$(call mfem-test,$<,, CEED Unit tests (cuda-gen),--device ceed-cuda:/gpu/cuda/gen $(MFEM_DATA_FLAG),SKIP-NO-VIS)
tests/unit/CMakeLists.txt:if (MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:                PROPERTY LANGUAGE CUDA)
tests/unit/CMakeLists.txt:# SERIAL CUDA TESTS: cunit_tests
tests/unit/CMakeLists.txt:# Create CUDA 'cunit_tests' executable and test
tests/unit/CMakeLists.txt:if (MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:   set_property(SOURCE ${CUNIT_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
tests/unit/CMakeLists.txt:#   sedov_tests_{cpu,debug,cuda,cuda_uvm}
tests/unit/CMakeLists.txt:#   tmop_pa_tests_{cpu,debug,cuda}
tests/unit/CMakeLists.txt:    if (MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:       set_property(SOURCE ${${NAME}_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
tests/unit/CMakeLists.txt:    endif(MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:    if (MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:       mfem_add_executable(${name}_tests_cuda ${${NAME}_TESTS_SRCS})
tests/unit/CMakeLists.txt:       target_compile_definitions(${name}_tests_cuda PUBLIC MFEM_${NAME}_DEVICE="cuda")
tests/unit/CMakeLists.txt:       target_link_libraries(${name}_tests_cuda mfem)
tests/unit/CMakeLists.txt:       add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} ${name}_tests_cuda)
tests/unit/CMakeLists.txt:           add_test(NAME ${name}_tests_cuda COMMAND ${name}_tests_cuda)
tests/unit/CMakeLists.txt:           mfem_add_executable(${name}_tests_cuda_uvm ${${NAME}_TESTS_SRCS})
tests/unit/CMakeLists.txt:           target_compile_definitions(${name}_tests_cuda_uvm PUBLIC
tests/unit/CMakeLists.txt:               MFEM_${NAME}_DEVICE="cuda:uvm")
tests/unit/CMakeLists.txt:           target_link_libraries(${name}_tests_cuda_uvm mfem)
tests/unit/CMakeLists.txt:               ${name}_tests_cuda_uvm)
tests/unit/CMakeLists.txt:               add_test(NAME ${name}_tests_cuda_uvm COMMAND ${name}_tests_cuda_uvm)
tests/unit/CMakeLists.txt:    endif(MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:#   ceed_tests, ceed_tests_cuda_{ref,shared,gen}
tests/unit/CMakeLists.txt:   if (MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:      set_property(SOURCE ${CEED_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
tests/unit/CMakeLists.txt:   endif(MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:   if (MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:      add_test(NAME ceed_tests_cuda_ref
tests/unit/CMakeLists.txt:               COMMAND ceed_tests --device ceed-cuda:/gpu/cuda/ref)
tests/unit/CMakeLists.txt:      add_test(NAME ceed_tests_cuda_shared
tests/unit/CMakeLists.txt:               COMMAND ceed_tests --device ceed-cuda:/gpu/cuda/shared)
tests/unit/CMakeLists.txt:      add_test(NAME ceed_tests_cuda_gen
tests/unit/CMakeLists.txt:               COMMAND ceed_tests --device ceed-cuda:/gpu/cuda/gen)
tests/unit/CMakeLists.txt:# PARALLEL CPU AND CUDA TESTS: {p,pc}unit_tests
tests/unit/CMakeLists.txt:   if (MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:      set_property(SOURCE punit_test_main.cpp PROPERTY LANGUAGE CUDA)
tests/unit/CMakeLists.txt:   if (MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:      set_property(SOURCE ${PCUNIT_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
tests/unit/CMakeLists.txt:#   psedov_tests_{cpu,debug,cuda,cuda_uvm}
tests/unit/CMakeLists.txt:#   ptmop_pa_tests_{cpu,cuda}
tests/unit/CMakeLists.txt:    if (MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:        set_property(SOURCE ${PAR_${NAME}_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
tests/unit/CMakeLists.txt:        # * HypreUsingGPU() is true; this is the same as: HYPRE_USING_GPU is
tests/unit/CMakeLists.txt:                 (HYPRE_USING_CUDA OR HYPRE_USING_HIP) AND
tests/unit/CMakeLists.txt:    if (MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:        list(APPEND backends cuda)
tests/unit/CMakeLists.txt:            list(APPEND backends cuda_uvm)
tests/unit/CMakeLists.txt:if (MFEM_USE_CUDA)
tests/unit/CMakeLists.txt:   set_property(SOURCE ${DEBUG_DEVICE_SRCS} PROPERTY LANGUAGE CUDA)
tests/unit/linalg/test_constrainedsolver.cpp:   if (HypreUsingGPU())
tests/unit/linalg/test_constrainedsolver.cpp:                << "is NOT supported with the GPU version of hypre.\n\n";
tests/unit/linalg/test_constrainedsolver.cpp:   if (HypreUsingGPU())
tests/unit/linalg/test_constrainedsolver.cpp:                << "is NOT supported with the GPU version of hypre.\n\n";
tests/unit/linalg/test_constrainedsolver.cpp:   if (HypreUsingGPU())
tests/unit/linalg/test_constrainedsolver.cpp:                << "is NOT supported with the GPU version of hypre.\n\n";
tests/unit/linalg/test_constrainedsolver.cpp:   if (HypreUsingGPU())
tests/unit/linalg/test_constrainedsolver.cpp:                << "is NOT supported with the GPU version of hypre.\n\n";
tests/unit/linalg/test_hypre_ilu.cpp:   if (HypreUsingGPU())
tests/unit/linalg/test_hypre_ilu.cpp:                << "is NOT supported with the GPU version of hypre.\n\n";
tests/unit/linalg/test_matrix_dense.cpp:          "[DenseMatrix][CUDA]")
tests/unit/linalg/test_matrix_dense.cpp:                           BatchedLinAlg::GPU_BLAS,
tests/unit/linalg/test_vector.cpp:TEST_CASE("Vector Sum", "[Vector],[CUDA]")
tests/unit/linalg/test_matrix_hypre.cpp:TEST_CASE("HypreParMatrixWrapConstructors-SyncChecks", "[Parallel], [CUDA]")
tests/unit/linalg/test_hypre_vector.cpp:// TODO: modify the tests here to support HYPRE_USING_GPU?
tests/unit/linalg/test_hypre_vector.cpp:#ifndef HYPRE_USING_GPU
tests/unit/linalg/test_hypre_vector.cpp:#endif // HYPRE_USING_GPU
tests/unit/linalg/test_direct_solvers.cpp:TEST_CASE("Serial Direct Solvers", "[CUDA]")
tests/unit/linalg/test_direct_solvers.cpp:TEST_CASE("Parallel Direct Solvers", "[Parallel], [CUDA]")
tests/unit/general/test_umpire_mem.cpp:#if defined(MFEM_USE_UMPIRE) && (defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
tests/unit/general/test_umpire_mem.cpp:#ifdef MFEM_USE_CUDA
tests/unit/general/test_umpire_mem.cpp:#include <cuda.h>
tests/unit/general/test_umpire_mem.cpp:constexpr const char * device_name = "cuda";
tests/unit/general/test_umpire_mem.cpp:#ifdef MFEM_USE_CUDA
tests/unit/general/test_umpire_mem.cpp:   auto err = cudaHostGetFlags(&flags, h_p);
tests/unit/general/test_umpire_mem.cpp:   cudaGetLastError(); // also resets last error
tests/unit/general/test_umpire_mem.cpp:   if (err == cudaSuccess) { return true; }
tests/unit/general/test_umpire_mem.cpp:   else if (err == cudaErrorInvalidValue) { return false; }
tests/unit/general/test_umpire_mem.cpp:#endif // MFEM_USE_UMPIRE && (MFEM_USE_CUDA || MFEM_USE_HIP)
tests/unit/general/test_mem.cpp:          "[CUDA]")
tests/benchmarks/bench_elasticity.cpp: * --benchmark_context=device=cuda
tests/benchmarks/bench_dg_amr.cpp:   * --benchmark_context=device=[cpu/cuda/hip]
tests/benchmarks/CMakeLists.txt:    if (MFEM_USE_CUDA)
tests/benchmarks/CMakeLists.txt:       set_property(SOURCE ${${NAME}_BENCH_SRCS} PROPERTY LANGUAGE CUDA)
tests/benchmarks/CMakeLists.txt:    endif(MFEM_USE_CUDA)
tests/benchmarks/CMakeLists.txt:    if (MFEM_USE_CUDA)
tests/benchmarks/CMakeLists.txt:        add_test(NAME bench_${name}_cuda
tests/benchmarks/CMakeLists.txt:                 COMMAND bench_${name} --benchmark_context=device=cuda)
tests/benchmarks/CMakeLists.txt:    endif(MFEM_USE_CUDA)
tests/benchmarks/bench_tmop.cpp: * --benchmark_context=device=cuda
tests/benchmarks/bench_assembly_levels.cpp:   * --benchmark_context=device=[cpu/cuda/hip]
doc/CodeDocumentation.dox: * <H3>Main GPU classes</H3>
doc/CodeDocumentation.dox: * - <a class="el" href="lor__elast_8cpp_source.html">LOR Elasticity</a>: solve linear elasticity with LOR preconditioning on GPUs
README.md:version 4.0, MFEM offers support for GPU acceleration, and programming models,
README.md:such as CUDA, HIP, OCCA, RAJA and OpenMP. MFEM-based applications require
CHANGELOG:GPU computing
CHANGELOG:- Added support for GPU-accelerated batched linear algebra (using cuBLAS,
CHANGELOG:- A new GPU kernel dispatch mechanism was introduced. Users can instantiate
CHANGELOG:GPU computing
CHANGELOG:- Added partial assembly and GPU support for the DG diffusion integrator.
CHANGELOG:- Efficient GPU-accelerated LOR assembly is now supported on surface meshes.
CHANGELOG:  MFEM's compute policy when hypre is built with GPU support. Requires version
CHANGELOG:  for linear elasticity on GPUs. See miniapps/solvers/lor_elast.
CHANGELOG:- VectorFEBoundaryFluxLFIntegrator is now supported on device/GPU.
CHANGELOG:  for GPU acceleration. Examples illustrating the solution of Darcy and grad-div
CHANGELOG:- Added support for assembling low-order-refined matrices using a GPU-enabled
CHANGELOG:  "batched" algorithm. The lor_solvers and plor_solvers now fully support GPU
CHANGELOG:  values at quadrature points (in particular for GPU/device kernels).
CHANGELOG:  including support for device/GPU acceleration.
CHANGELOG:  device acceleration, e.g. with NVIDIA and AMD GPUs. The p-adaptivity is
CHANGELOG:  the existing hypre + CUDA support, most of the MFEM examples and miniapps work
CHANGELOG:  parameters, particularly GPU-relevant options. Updated parallel example codes
CHANGELOG:- GPU-enabled partial (PA) and element (EA) assembly for discontinuous Galerkin
CHANGELOG:- Added 'double' atomicAdd implementation for previous versions of CUDA.
CHANGELOG:- Added support for AMG preconditioners on GPUs based on the hypre library
CHANGELOG:  The GPU preconditioners require that both hypre and MFEM are built with CUDA
CHANGELOG:  support. Hypre builds with CUDA and unified memory are also supported and
CHANGELOG:  can be used with `-d cuda:uvm` as a command-line option.
CHANGELOG:- The TMOP mesh optimization algorithms were extended to GPU:
CHANGELOG:- Added initial support for GPU-accelerated versions of PETSc that works with
CHANGELOG:  MFEM_USE_CUDA if PETSc has been configured with CUDA support. Examples 1 and 9
CHANGELOG:  in the examples/petsc directory have been modified to work with --device cuda.
CHANGELOG:- Added support for different modes of QuadratureInterpolator on GPU.
CHANGELOG:- Added HOST_PINNED MemoryType and a pinned host allocator for CUDA and HIP.
CHANGELOG:- Added matrix-free GPU-enabled implementations of GradientInterpolator and
CHANGELOG:- Extended `make test` to include GPU tests when MFEM is built with CUDA or HIP
CHANGELOG:  * CUDA >= 10.1.168
CHANGELOG:  * HYPRE >= 2.22.0 for CUDA support
CHANGELOG:  * PETSc >= 3.15.0 for CUDA support
CHANGELOG:- Added an Element Assembly mode compatible with GPU device execution for H1 and
CHANGELOG:- Added a Full Assembly mode compatible with GPU device execution. This assembly
CHANGELOG:- Added CUDA support for:
CHANGELOG:- Added a new solver class for simple integration with NVIDIA's multigrid
CHANGELOG:  solver may be configured to run with one GPU per MPI rank or with more MPI
CHANGELOG:  ranks than GPUs. In the latter case, matrices and vectors are consolidated to
CHANGELOG:  ranks communicating with the GPUs and the solution is then broadcasted.
CHANGELOG:  Although CUDA is required to build, the AmgX support is compatible with the
CHANGELOG:  partially based on: "AmgXWrapper: An interface between PETSc and the NVIDIA
CHANGELOG:  matrix-based and matrix-free discretizations with basic GPU capability, see
CHANGELOG:    to solve the Laplace problem with AMG preconditioning on GPUs.
CHANGELOG:Improved GPU capabilities
CHANGELOG:- Added initial support for AMD GPUs based on HIP: a C++ runtime API and kernel
CHANGELOG:  language that can run on both AMD and NVIDIA hardware.
CHANGELOG:  memory devices like NUMA and GPUs, see https://github.com/LLNL/Umpire.
CHANGELOG:- GPU acceleration is now available in 3 additional examples: 3, 9 and 24.
CHANGELOG:- Improved RAJA backend and multi-GPU MPI communications.
CHANGELOG:- Added a "debug" device designed specifically to aid in debugging GPU code by
CHANGELOG:  host <-> device transfers) without any GPU hardware.
CHANGELOG:- Added support for matrix-free diagonal smoothers on GPUs.
CHANGELOG:- The current list of available device backends is: "ceed-cuda", "occa-cuda",
CHANGELOG:  "raja-cuda", "cuda", "hip", "debug", "occa-omp", "raja-omp", "omp",
CHANGELOG:  * CUDA pointers, using cudaMalloc and HIP pointers, using hipMalloc,
CHANGELOG:  * Managed CUDA/HIP memory (UVM), using cudaMallocManaged/hipMallocManaged,
CHANGELOG:- This initial integration includes Mass and Diffusion integrators. libCEED GPU
CHANGELOG:  recommended to use the "cuda" build option to minimize memory transfers.
CHANGELOG:- Both CPU and GPU modes are available as MFEM device backends (ceed-cpu and
CHANGELOG:  ceed-cuda), using some of the best performing CPU and GPU backends from
CHANGELOG:- NOTE: The current default libCEED GPU backend (ceed-cuda) uses atomics and
CHANGELOG:- The support for matrix-free methods on both CPU and GPU devices based on a
CHANGELOG:  improved performance, particularly in high-order 3D runs on GPUs.
CHANGELOG:- Added support for Ginkgo, a high-performance linear algebra library for GPU
CHANGELOG:- In the enum classes MemoryType and MemoryClass, "CUDA" was renamed to "DEVICE"
CHANGELOG:  which now denotes either "CUDA" or "HIP" depending on the build configuration.
CHANGELOG:  In the same enum classes, "CUDA_UVM" was renamed to "MANAGED".
CHANGELOG:GPU support
CHANGELOG:- Added initial support for hardware devices, such as GPUs, and programming
CHANGELOG:  models, such as CUDA, OCCA, RAJA and OpenMP.
CHANGELOG:- The GPU/device support is based on MFEM's new backends and kernels working
CHANGELOG:  advantage of GPU acceleration with the backend selectable at runtime. Many of
CHANGELOG:- In addition to native CUDA kernels, the library currently supports OCCA, RAJA
CHANGELOG:  code. The list of current backends is: "occa-cuda", "raja-cuda", "cuda",
CHANGELOG:- GPU-related limitations:
CHANGELOG:  * Hypre preconditioners are not yet available in GPU mode, and in particular
CHANGELOG:  * Only constant coefficients are currently supported on GPUs.
CHANGELOG:  results when using devices such as CUDA and OpenMP.
mesh/face_nbr_geom.cpp:   bool mpi_gpu_aware = Device::GetGPUAwareMPI();
mesh/face_nbr_geom.cpp:   const auto send_data_ptr = mpi_gpu_aware ? send_data.Read() :
mesh/face_nbr_geom.cpp:   auto x_shared_ptr = mpi_gpu_aware ? x_shared.Write() : x_shared.HostWrite();
makefile:   make cuda
makefile:   make pcuda
makefile:make cuda
makefile:   A shortcut to configure and build the serial GPU/CUDA optimized version of the library.
makefile:make pcuda
makefile:   A shortcut to configure and build the parallel GPU/CUDA optimized version of the library.
makefile:   A shortcut to configure and build the serial GPU/CUDA debug version of the library.
makefile:   A shortcut to configure and build the parallel GPU/CUDA debug version of the library.
makefile:   A shortcut to configure and build the serial GPU/HIP optimized version of the library.
makefile:   A shortcut to configure and build the parallel GPU/HIP optimized version of the library.
makefile:   A shortcut to configure and build the serial GPU/HIP debug version of the library.
makefile:   A shortcut to configure and build the parallel GPU/HIP debug version of the library.
makefile: cuda hip pcuda phip cudebug hipdebug pcudebug phipdebug hpc style
makefile:ifeq ($(MFEM_USE_CUDA)$(MFEM_USE_HIP),NONO)
makefile:ifeq ($(MFEM_USE_CUDA),YES)
makefile:   MFEM_CXX ?= $(CUDA_CXX)
makefile:   CXXFLAGS += $(CUDA_FLAGS) -ccbin $(MFEM_HOST_CXX)
makefile:   XCOMPILER = $(CUDA_XCOMPILER)
makefile:   XLINKER   = $(CUDA_XLINKER)
makefile:   # CUDA_OPT and CUDA_LIB are added below
makefile:      $(error Incompatible config: MFEM_USE_CUDA can not be combined with MFEM_USE_HIP)
makefile:   # Compatibility test against MFEM_USE_CUDA
makefile:   ifeq ($(MFEM_USE_CUDA),YES)
makefile:      $(error Incompatible config: MFEM_USE_HIP can not be combined with MFEM_USE_CUDA)
makefile:   # MFEM_USE_OPENMP, MFEM_USE_CUDA, MFEM_USE_RAJA, MFEM_USE_OCCA
makefile:MFEM_DEPENDENCIES = $(MFEM_REQ_LIB_DEPS) LIBUNWIND OPENMP CUDA HIP
makefile: MFEM_USE_PUMI MFEM_USE_HIOP MFEM_USE_GSLIB MFEM_USE_CUDA MFEM_USE_HIP\
makefile:	debug pdebug cuda hip pcuda cudebug pcudebug hpc style check test unittest \
makefile:serial debug cuda hip cudebug hipdebug:           M_MPI=NO
makefile:parallel pdebug pcuda pcudebug phip phipdebug:    M_MPI=YES
makefile:serial parallel cuda pcuda hip phip:              M_DBG=NO
makefile:cuda pcuda cudebug pcudebug:                      M_CUDA=YES
makefile:cuda pcuda cudebug pcudebug:
makefile:	   MFEM_USE_CUDA=$(M_CUDA) $(MAKEOVERRIDES_SAVE)
makefile:	$(MAKE) -f $(THIS_MK) config MFEM_USE_MPI=YES MFEM_USE_CUDA=YES \
makefile:	$(info MFEM_USE_CUDA          = $(MFEM_USE_CUDA))
CMakeLists.txt:# Version 3.8 or newer is required for direct CUDA support.
CMakeLists.txt:if (MFEM_USE_CUDA)
CMakeLists.txt:      message(FATAL_ERROR " *** MFEM_USE_HIP cannot be combined with MFEM_USE_CUDA.")
CMakeLists.txt:   # Use ${CMAKE_CXX_COMPILER} as the cuda host compiler.
CMakeLists.txt:   if (NOT CMAKE_CUDA_HOST_COMPILER)
CMakeLists.txt:      set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
CMakeLists.txt:   enable_language(CUDA)
CMakeLists.txt:   set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING
CMakeLists.txt:      "CUDA standard to use.")
CMakeLists.txt:   set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL
CMakeLists.txt:      "Force the use of the chosen CUDA standard.")
CMakeLists.txt:   set(CMAKE_CUDA_EXTENSIONS OFF CACHE BOOL "Enable CUDA standard extensions.")
CMakeLists.txt:   set(CUDA_FLAGS "--expt-extended-lambda")
CMakeLists.txt:      set(CUDA_FLAGS "-arch=${CUDA_ARCH} ${CUDA_FLAGS}")
CMakeLists.txt:   elseif (NOT CMAKE_CUDA_ARCHITECTURES)
CMakeLists.txt:      string(REGEX REPLACE "^sm_" "" ARCH_NUMBER "${CUDA_ARCH}")
CMakeLists.txt:      if ("${CUDA_ARCH}" STREQUAL "sm_${ARCH_NUMBER}")
CMakeLists.txt:         set(CMAKE_CUDA_ARCHITECTURES "${ARCH_NUMBER}")
CMakeLists.txt:         message(FATAL_ERROR "Unknown CUDA_ARCH: ${CUDA_ARCH}")
CMakeLists.txt:      set(CUDA_ARCH "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
CMakeLists.txt:   message(STATUS "Using CUDA architecture: ${CUDA_ARCH}")
CMakeLists.txt:      set(CUDA_FLAGS "-ccbin=${CMAKE_CXX_COMPILER} ${CUDA_FLAGS}")
CMakeLists.txt:      set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${CMAKE_CXX_COMPILER})
CMakeLists.txt:   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_FLAGS}")
CMakeLists.txt:   find_package(CUDAToolkit REQUIRED)
CMakeLists.txt:   get_target_property(CUSPARSE_LIBRARIES CUDA::cusparse LOCATION)
CMakeLists.txt:   get_target_property(CUBLAS_LIBRARIES CUDA::cublas LOCATION)
CMakeLists.txt:    set(GPU_TARGETS "${HIP_ARCH}" CACHE STRING "HIP targets to compile for")
CMakeLists.txt:  if (ROCM_PATH)
CMakeLists.txt:    list(INSERT CMAKE_PREFIX_PATH 0 ${ROCM_PATH})
CMakeLists.txt:    if (MFEM_USE_CUDA)
CMakeLists.txt:      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
CMakeLists.txt:  if (MFEM_USE_CUDA)
CMakeLists.txt:    list(APPEND SUNDIALS_COMPONENTS NVector_Cuda)
CMakeLists.txt:if (MFEM_USE_CUDA)
CMakeLists.txt:  set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
INSTALL:MFEM also includes support for devices such as GPUs, and programming models such
INSTALL:as CUDA, HIP, OCCA, OpenMP and RAJA.
INSTALL:- CUDA support requires an NVIDIA GPU and an installation of the CUDA Toolkit
INSTALL:  https://developer.nvidia.com/cuda-toolkit
INSTALL:- HIP support requires an AMD GPU and an installation of the ROCm software stack
INSTALL:  https://rocmdocs.amd.com
INSTALL:  with (optionally) support for CUDA and OpenMP
INSTALL:CUDA build:
INSTALL:   make cuda -j 4
INSTALL:   (build for a specific compute capability: 'make cuda -j 4 CUDA_ARCH=sm_70')
INSTALL:   (build for a specific AMD GPU chip: 'make hip -j 4 HIP_ARCH=gfx900')
INSTALL:CUDA build:
INSTALL:   cmake <mfem-source-dir> -DMFEM_USE_CUDA=YES
INSTALL:   make cuda      -> Builds serial cuda optimized version of the library
INSTALL:   make pcuda     -> Builds parallel cuda optimized version of the library
INSTALL:   make cudebug   -> Builds serial cuda debug version of the library
INSTALL:   make pcudebug  -> Builds parallel cuda debug version of the library
INSTALL:   CUDA_CXX - The CUDA compiler, 'nvcc'
INSTALL:   iterative linear solvers and preconditioners with OpenMP, CUDA backends, see
INSTALL:   Enable MFEM functionality based on the AmgX multigrid library from NVIDIA.
INSTALL:   implementations that have been optimized for Nvidia and AMD GPUs.
INSTALL:   memory devices like NUMA and GPUs.
INSTALL:MFEM_USE_CUDA = YES/NO
INSTALL:   Enables support for CUDA devices in MFEM. CUDA is a parallel computing
INSTALL:   units (GPUs). The variable CUDA_ARCH is used to specify the CUDA compute
INSTALL:   capability used during compilation (by default, CUDA_ARCH=sm_60). When
INSTALL:   enabled, this option uses the CUDA_* build options, see below.
INSTALL:   NVIDIA GPUs. The variable HIP_ARCH is used to specify the AMD GPU processor
INSTALL:   model backends. When using RAJA built with CUDA support, CUDA support must be
INSTALL:   also enabled in MFEM, i.e. MFEM_USE_CUDA=YES must be set.
INSTALL:   GPU, FPGA) by providing an unified API for interacting with JIT-compiled
INSTALL:   backends. In order to use the OCCA CUDA backend, CUDA support must be enabled
INSTALL:   in MFEM as well, i.e. MFEM_USE_CUDA=YES must be set.
INSTALL:  Versions: HYPRE >= 2.10.0b  (HYPRE built without CUDA)
INSTALL:            HYPRE >= 2.22.1   (HYPRE built with CUDA)
INSTALL:            HYPRE >= 2.31.0   (runtime selectable HYPRE execution on CPU/GPU)
INSTALL:  When MFEM_USE_CUDA is enabled, only SUNDIALS v5.4.0+ is supported.
INSTALL:  If MFEM_USE_CUDA is enabled, we expect that SUNDIALS is built with support
INSTALL:  for CUDA.
INSTALL:            SUNDIALS >= 5.4.0 for CUDA support, and
INSTALL:  URL: https://github.com/NVIDIA/AMGX
INSTALL:- CUDA (optional), used when MFEM_USE_CUDA = YES.
INSTALL:  URL: https://developer.nvidia.com/cuda-toolkit
INSTALL:  Options: CUDA_CXX, CUDA_ARCH, CUDA_OPT, CUDA_LIB.
INSTALL:  Versions: CUDA >= 10.1.168.
INSTALL:  URL: https://rocmdocs.amd.com
INSTALL:Note: the option MFEM_USE_CUDA requires CMake version 3.8 or newer!
INSTALL:MFEM_USE_CUDA
linalg/hypre.cpp:#if defined(HYPRE_USING_GPU) && (MFEM_HYPRE_VERSION >= 23100)
linalg/hypre.cpp:   // https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html#gpu-supported-options
linalg/hypre.cpp:#ifdef HYPRE_USING_CUDA
linalg/hypre.cpp:   // Allocate hypre objects in GPU memory (default)
linalg/hypre.cpp:   // Use GPU-based random number generator (default)
linalg/hypre.cpp:   // HYPRE_SetUseGpuRand(1);
linalg/hypre.cpp:#if !defined(HYPRE_USING_GPU)
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:#if !defined(HYPRE_USING_GPU)
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:   if (HypreUsingGPU())
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:   if (HypreUsingGPU())
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:   if (HypreUsingGPU())
linalg/hypre.cpp:#if defined(HYPRE_USING_GPU)
linalg/hypre.cpp:      if (HypreUsingGPU())
linalg/hypre.cpp:#if defined(HYPRE_USING_GPU)
linalg/hypre.cpp:      if (HypreUsingGPU())
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:   if (HypreUsingGPU() && ParCSROwner && (diagOwner < 0 || offdOwner < 0))
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:   if (HypreUsingGPU())
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:   if (HypreUsingGPU())
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:   if (HypreUsingGPU())
linalg/hypre.cpp:   // HYPRE_USING_GPU is not defined for these versions of HYPRE
linalg/hypre.cpp:#elif defined(HYPRE_USING_GPU)
linalg/hypre.cpp:   if (HypreUsingGPU())
linalg/hypre.cpp:   if (!HypreUsingGPU())
linalg/hypre.cpp:#if defined(hypre_IntArrayData) && defined(HYPRE_USING_GPU)
linalg/hypre.cpp:      if (HypreUsingGPU())
linalg/hypre.cpp:#ifdef HYPRE_USING_GPU
linalg/hypre.cpp:   if (HypreUsingGPU())
linalg/hypre.cpp:      MFEM_ABORT("this method is not supported in hypre built with GPU support");
linalg/hypre.cpp:   const bool hypre_gpu = HypreUsingGPU();
linalg/hypre.cpp:   int amg_coarsen_type = hypre_gpu ? 8 : 10;
linalg/hypre.cpp:   int amg_agg_levels   = hypre_gpu ? 0 : 1;
linalg/hypre.cpp:   int amg_rlx_type     = hypre_gpu ? 18 : 8;
linalg/hypre.cpp:   int rlx_type         = hypre_gpu ? 1: 2;
linalg/hypre.cpp:   const bool hypre_gpu = HypreUsingGPU();
linalg/hypre.cpp:   int rlx_type         = hypre_gpu ? 1 : 2;
linalg/hypre.cpp:   int amg_coarsen_type = hypre_gpu ? 8 : 10;
linalg/hypre.cpp:   int amg_agg_levels   = hypre_gpu ? 0 : 1;
linalg/hypre.cpp:   int amg_rlx_type     = hypre_gpu ? 18 : 8;
linalg/hypre.cpp:   const bool is_device_ptr = HypreUsingGPU();
linalg/dtensor.hpp:#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
linalg/dtensor.hpp:#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
linalg/dtensor.hpp:/// A basic generic Tensor class, appropriate for use on the GPU
linalg/solvers.cpp:   // TODO: GPU/device implementation
linalg/hypre.hpp:#if defined(HYPRE_USING_GPU) && \
linalg/hypre.hpp:    !(defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
linalg/hypre.hpp:#error "Unsupported GPU build of HYPRE! Only CUDA and HIP builds are supported."
linalg/hypre.hpp:#if defined(HYPRE_USING_CUDA) && !defined(MFEM_USE_CUDA)
linalg/hypre.hpp:#error "MFEM_USE_CUDA=YES is required when HYPRE is built with CUDA!"
linalg/hypre.hpp:/// HYPRE_Init() and sets some GPU-relevant options at construction and 2) calls
linalg/hypre.hpp:   /// HYPRE's default will be used; if HYPRE is built for the GPU and the
linalg/hypre.hpp:   /// aforementioned variable is false then HYPRE will use the GPU even if MFEM
linalg/hypre.hpp:   /// This function is no-op if HYPRE is built without GPU support or the HYPRE
linalg/hypre.hpp:   /// This value is not used if HYPRE is build without GPU support or the HYPRE
linalg/hypre.hpp:   /// Set the default hypre global options (mostly GPU-relevant).
linalg/hypre.hpp:#if !defined(HYPRE_USING_GPU)
linalg/hypre.hpp:#else // HYPRE_USING_GPU is defined and MFEM_HYPRE_VERSION >= 23100
linalg/hypre.hpp:#if !defined(HYPRE_USING_GPU)
linalg/hypre.hpp:#else // HYPRE_USING_GPU is defined and MFEM_HYPRE_VERSION >= 23100
linalg/hypre.hpp:   //      when hypre is built with CUDA support, A->diag owns the "host"
linalg/hypre.hpp:   //  -2: used when hypre is built with CUDA support, A->diag owns the "hypre"
linalg/hypre.hpp:   /** When HYPRE is built for GPUs, this method will construct and store the
linalg/hypre.hpp:       for GPUs, this method is a no-op.
linalg/hypre.hpp:#if !defined(HYPRE_USING_GPU)
linalg/hypre.hpp:       running on GPU. */
linalg/hypre.hpp:      return HypreUsingGPU() ? l1Jacobi : l1GS;
linalg/dinvariants.hpp:#include "../general/cuda.hpp"
linalg/sundials.hpp:#if defined(MFEM_USE_CUDA) && ((SUNDIALS_VERSION_MAJOR == 5) && (SUNDIALS_VERSION_MINOR < 4))
linalg/sundials.hpp:#error MFEM requires SUNDIALS version 5.4.0 or newer when MFEM_USE_CUDA=TRUE!
linalg/sundials.hpp:#if defined(MFEM_USE_CUDA) && !defined(SUNDIALS_NVECTOR_CUDA)
linalg/sundials.hpp:#error MFEM_USE_CUDA=TRUE requires SUNDIALS to be built with CUDA support
linalg/sundials.hpp:#if defined(MFEM_USE_CUDA)
linalg/sundials.hpp:#include <sunmemory/sunmemory_cuda.h>
linalg/sundials.hpp:#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
linalg/sundials.hpp:// SUNMemory interface class (used when CUDA or HIP is enabled)
linalg/sundials.hpp:#else // MFEM_USE_CUDA || MFEM_USE_HIP
linalg/sundials.hpp:// Dummy SUNMemory interface class (used when CUDA or HIP is not enabled)
linalg/sundials.hpp:#endif // MFEM_USE_CUDA || MFEM_USE_HIP
linalg/sundials.hpp:   /** @param[in] use_device  If true, use the SUNDIALS CUDA or HIP N_Vector. */
linalg/sundials.hpp:       @param[in] use_device  If true, use the SUNDIALS CUDA or HIP N_Vector. */
linalg/sundials.hpp:#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
linalg/petsc.cpp:#if defined(MFEM_USE_CUDA) && defined(PETSC_HAVE_CUDA)
linalg/petsc.cpp:#define PETSC_VECDEVICE VECCUDA
linalg/petsc.cpp:#define VecDeviceGetArrayRead      VecCUDAGetArrayRead
linalg/petsc.cpp:#define VecDeviceGetArrayWrite     VecCUDAGetArrayWrite
linalg/petsc.cpp:#define VecDeviceGetArray          VecCUDAGetArray
linalg/petsc.cpp:#define VecDeviceRestoreArrayRead  VecCUDARestoreArrayRead
linalg/petsc.cpp:#define VecDeviceRestoreArrayWrite VecCUDARestoreArrayWrite
linalg/petsc.cpp:#define VecDeviceRestoreArray      VecCUDARestoreArray
linalg/petsc.cpp:#define VecDevicePlaceArray        VecCUDAPlaceArray
linalg/petsc.cpp:#define VecDeviceResetArray        VecCUDAResetArray
linalg/petsc.cpp:   // Tell PETSc to use the same CUDA or HIP device as MFEM:
linalg/petsc.cpp:   if (mfem::Device::Allows(mfem::Backend::CUDA_MASK))
linalg/petsc.cpp:      const char *opts = "-cuda_device";
linalg/petsc.cpp:      const char *opts = "-device_select_cuda";
linalg/petsc.cpp:                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
linalg/petsc.cpp:         ierr = __mfem_VecSetOffloadMask(x,PETSC_OFFLOAD_GPU); PCHKERRQ(x,ierr);
linalg/petsc.cpp:                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
linalg/petsc.cpp:         case PETSC_OFFLOAD_GPU:
linalg/petsc.cpp:                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
linalg/petsc.cpp:      else if (dv) { mask = PETSC_OFFLOAD_GPU; }
linalg/petsc.cpp:                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
linalg/petsc.cpp:                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
linalg/petsc.cpp:                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
linalg/petsc.cpp:                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
linalg/petsc.cpp:                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
linalg/petsc.cpp:         ierr = __mfem_VecSetOffloadMask(x,PETSC_OFFLOAD_GPU); PCHKERRQ(x,ierr);
linalg/petsc.cpp:                                    VECSEQCUDA,VECMPICUDA,VECSEQHIP,VECMPIHIP,
linalg/petsc.cpp:         ierr = __mfem_VecSetOffloadMask(x,PETSC_OFFLOAD_GPU); PCHKERRQ(x,ierr);
linalg/petsc.cpp:      if ((usedev && (mask != PETSC_OFFLOAD_GPU && mask != PETSC_OFFLOAD_BOTH)) ||
linalg/petsc.cpp:   if (!A || (!Device::Allows(Backend::CUDA_MASK) &&
linalg/amgxsolver.cpp:// Implementation of the MFEM wrapper for Nvidia's multigrid library, AmgX
linalg/amgxsolver.cpp://    AmgXWrapper: An interface between PETSc and the NVIDIA AmgX library.
linalg/amgxsolver.cpp:   InitExclusiveGPU(comm);
linalg/amgxsolver.cpp:   mpi_gpu_mode = "serial";
linalg/amgxsolver.cpp:void AmgXSolver::InitExclusiveGPU(const MPI_Comm &comm)
linalg/amgxsolver.cpp:   // Note that every MPI rank may talk to a GPU
linalg/amgxsolver.cpp:   mpi_gpu_mode = "mpi-gpu-exclusive";
linalg/amgxsolver.cpp:   gpuProc = 0;
linalg/amgxsolver.cpp:   MPI_Comm_dup(comm, &gpuWorld);
linalg/amgxsolver.cpp:   MPI_Comm_size(gpuWorld, &gpuWorldSize);
linalg/amgxsolver.cpp:   MPI_Comm_rank(gpuWorld, &myGpuWorldRank);
linalg/amgxsolver.cpp:// Initialize for MPI ranks > GPUs, all devices are visible to all of the MPI
linalg/amgxsolver.cpp:   mpi_gpu_mode = "mpi-teams";
linalg/amgxsolver.cpp:   // Only processes in gpuWorld are required to initialize AmgX
linalg/amgxsolver.cpp:   if (gpuProc == 0)
linalg/amgxsolver.cpp:   if (count == 1) { AMGX_SAFE_CALL(AMGX_resources_create(&rsrc, cfg, &gpuWorld, 1, &devID)); }
linalg/amgxsolver.cpp:// Groups MPI ranks into teams and assigns the roots to talk to GPUs
linalg/amgxsolver.cpp:   MPI_Comm_split(globalCpuWorld, gpuProc, 0, &gpuWorld);
linalg/amgxsolver.cpp:   // Get size and rank for the communicator corresponding to gpuWorld
linalg/amgxsolver.cpp:   if (gpuWorld != MPI_COMM_NULL)
linalg/amgxsolver.cpp:      MPI_Comm_set_name(gpuWorld, "gpuWorld");
linalg/amgxsolver.cpp:      MPI_Comm_size(gpuWorld, &gpuWorldSize);
linalg/amgxsolver.cpp:      MPI_Comm_rank(gpuWorld, &myGpuWorldRank);
linalg/amgxsolver.cpp:   else // for those that will not communicate with the GPU
linalg/amgxsolver.cpp:      gpuWorldSize = MPI_UNDEFINED;
linalg/amgxsolver.cpp:      myGpuWorldRank = MPI_UNDEFINED;
linalg/amgxsolver.cpp:   // Split local world into worlds corresponding to each CUDA device
linalg/amgxsolver.cpp:      gpuProc = 0;
linalg/amgxsolver.cpp:      MFEM_WARNING("CUDA devices on the node " << nodeName.c_str() <<
linalg/amgxsolver.cpp:      gpuProc = 0;
linalg/amgxsolver.cpp:         if (myLocalRank % (nBasic + 1) == 0) { gpuProc = 0; }
linalg/amgxsolver.cpp:         if ((myLocalRank - (nBasic+1)*nRemain) % nBasic == 0) { gpuProc = 0; }
linalg/amgxsolver.cpp:   // Assumes one GPU per MPI rank
linalg/amgxsolver.cpp:   if (mpi_gpu_mode=="mpi-gpu-exclusive")
linalg/amgxsolver.cpp:      SetMatrixMPIGPUExclusive(A, loc_A, loc_I, loc_J, update_mat);
linalg/amgxsolver.cpp:   // Assumes teams of MPI ranks are sharing a GPU
linalg/amgxsolver.cpp:   if (mpi_gpu_mode == "mpi-teams")
linalg/amgxsolver.cpp:   mfem_error("Unsupported MPI_GPU combination \n");
linalg/amgxsolver.cpp:void AmgXSolver::SetMatrixMPIGPUExclusive(const HypreParMatrix &A,
linalg/amgxsolver.cpp:   Array<int64_t> rowPart(gpuWorldSize+1); rowPart = 0.0;
linalg/amgxsolver.cpp:                 ,gpuWorld);
linalg/amgxsolver.cpp:   MPI_Barrier(gpuWorld);
linalg/amgxsolver.cpp:   rowPart[gpuWorldSize] = A.M();
linalg/amgxsolver.cpp:      MPI_Barrier(gpuWorld);
linalg/amgxsolver.cpp:   if (gpuProc == 0)
linalg/amgxsolver.cpp:      rowPart.SetSize(gpuWorldSize+1); rowPart=0;
linalg/amgxsolver.cpp:                    gpuWorld);
linalg/amgxsolver.cpp:      MPI_Barrier(gpuWorld);
linalg/amgxsolver.cpp:      MPI_Barrier(gpuWorld);
linalg/amgxsolver.cpp:         MPI_Barrier(gpuWorld);
linalg/amgxsolver.cpp:   if (mpi_gpu_mode != "mpi-teams")
linalg/amgxsolver.cpp:      if (mpi_gpu_mode != "serial")
linalg/amgxsolver.cpp:         MPI_Barrier(gpuWorld);
linalg/amgxsolver.cpp:   if (gpuWorld != MPI_COMM_NULL)
linalg/amgxsolver.cpp:      MPI_Barrier(gpuWorld);
linalg/amgxsolver.cpp:   // Only processes using GPU are required to destroy AmgX content
linalg/amgxsolver.cpp:   if (gpuProc == 0 || mpi_gpu_mode == "serial")
linalg/amgxsolver.cpp:      // destroy gpuWorld
linalg/amgxsolver.cpp:      if (mpi_gpu_mode != "serial")
linalg/amgxsolver.cpp:         MPI_Comm_free(&gpuWorld);
linalg/amgxsolver.cpp:   gpuProc = MPI_UNDEFINED;
linalg/vector.cpp:#ifdef MFEM_USE_CUDA
linalg/vector.cpp:   __shared__ real_t s_min[MFEM_CUDA_BLOCKS];
linalg/vector.cpp:static Array<real_t> cuda_reduce_buf;
linalg/vector.cpp:   const int tpb = MFEM_CUDA_BLOCKS;
linalg/vector.cpp:   const int blockSize = MFEM_CUDA_BLOCKS;
linalg/vector.cpp:   cuda_reduce_buf.SetSize(min_sz);
linalg/vector.cpp:   Memory<real_t> &buf = cuda_reduce_buf.GetMemory();
linalg/vector.cpp:   MFEM_GPU_CHECK(cudaGetLastError());
linalg/vector.cpp:   __shared__ real_t s_dot[MFEM_CUDA_BLOCKS];
linalg/vector.cpp:   const int tpb = MFEM_CUDA_BLOCKS;
linalg/vector.cpp:   const int blockSize = MFEM_CUDA_BLOCKS;
linalg/vector.cpp:   cuda_reduce_buf.SetSize(dot_sz, Device::GetDeviceMemoryType());
linalg/vector.cpp:   Memory<real_t> &buf = cuda_reduce_buf.GetMemory();
linalg/vector.cpp:   MFEM_GPU_CHECK(cudaGetLastError());
linalg/vector.cpp:#endif // MFEM_USE_CUDA
linalg/vector.cpp:   MFEM_GPU_CHECK(hipGetLastError());
linalg/vector.cpp:   MFEM_GPU_CHECK(hipGetLastError());
linalg/vector.cpp:#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP) || defined(MFEM_USE_OPENMP)
linalg/vector.cpp:#ifdef MFEM_USE_CUDA
linalg/vector.cpp:   if (Device::Allows(Backend::CUDA_MASK))
linalg/vector.cpp:#ifdef MFEM_USE_CUDA
linalg/vector.cpp:   if (Device::Allows(Backend::CUDA_MASK))
linalg/vector.cpp:#ifdef MFEM_USE_CUDA
linalg/vector.cpp:      if (Device::Allows(Backend::CUDA_MASK))
linalg/ginkgo.hpp:      /// CUDA GPU Executor.
linalg/ginkgo.hpp:      CUDA = 2,
linalg/ginkgo.hpp:      /// HIP GPU Executor.
linalg/ginkgo.hpp:    * In Ginkgo, GPU Executors must have an associated host Executor.
linalg/ginkgo.hpp:    * In Ginkgo, GPU Executors must have an associated host Executor.
linalg/ginkgo.hpp:    * for GPU backends.
linalg/ginkgo.hpp:    * CUDA, Ginkgo will choose the CudaExecutor with a default
linalg/ginkgo.hpp:    * Executor for GPU backends.
linalg/ginkgo.hpp:    * +    CudaExecutor specifies that the data should be stored and the
linalg/ginkgo.hpp:    *      operations executed on the NVIDIA GPU accelerator;
linalg/ginkgo.hpp:    *      operations executed on the GPU accelerator using HIP;
linalg/ginkgo.hpp:    * `gko::OmpExecutor`, `gko::CudaExecutor` and `gko::ReferenceExecutor`
linalg/ginkgo.hpp:    * `gko::OmpExecutor`, `gko::CudaExecutor` and `gko::ReferenceExecutor`
linalg/sparsemat.cpp:#if defined(MFEM_USE_CUDA)
linalg/sparsemat.cpp:#define MFEM_CUDA_or_HIP(stub) CUDA##stub
linalg/sparsemat.cpp:#define MFEM_GPUSPARSE_ALG CUSPARSE_SPMV_CSR_ALG1
linalg/sparsemat.cpp:#define MFEM_GPUSPARSE_ALG CUSPARSE_CSRMV_ALG1
linalg/sparsemat.cpp:#define MFEM_CUDA_or_HIP(stub) HIP##stub
linalg/sparsemat.cpp:#define MFEM_GPUSPARSE_ALG HIPSPARSE_CSRMV_ALG1
linalg/sparsemat.cpp:#endif // defined(MFEM_USE_CUDA)
linalg/sparsemat.cpp:#ifdef MFEM_USE_CUDA_OR_HIP
linalg/sparsemat.cpp:#endif // MFEM_USE_CUDA_OR_HIP
linalg/sparsemat.cpp:void SparseMatrix::InitGPUSparse()
linalg/sparsemat.cpp:#ifdef MFEM_USE_CUDA_OR_HIP
linalg/sparsemat.cpp:   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
linalg/sparsemat.cpp:      useGPUSparse=true;
linalg/sparsemat.cpp:      useGPUSparse=false;
linalg/sparsemat.cpp:#endif // MFEM_USE_CUDA_OR_HIP
linalg/sparsemat.cpp:void SparseMatrix::ClearGPUSparse()
linalg/sparsemat.cpp:#ifdef MFEM_USE_CUDA_OR_HIP
linalg/sparsemat.cpp:#if CUDA_VERSION >= 10010 || defined(MFEM_USE_HIP)
linalg/sparsemat.cpp:#endif // CUDA_VERSION >= 10010 || defined(MFEM_USE_HIP)
linalg/sparsemat.cpp:#endif // MFEM_USE_CUDA_OR_HIP
linalg/sparsemat.cpp:   InitGPUSparse();
linalg/sparsemat.cpp:   InitGPUSparse();
linalg/sparsemat.cpp:   InitGPUSparse();
linalg/sparsemat.cpp:   InitGPUSparse();
linalg/sparsemat.cpp:   InitGPUSparse();
linalg/sparsemat.cpp:   InitGPUSparse();
linalg/sparsemat.cpp:   ClearGPUSparse();
linalg/sparsemat.cpp:#ifdef MFEM_USE_CUDA_OR_HIP
linalg/sparsemat.cpp:   if ( Device::Allows( Backend::CUDA_MASK ))
linalg/sparsemat.cpp:#if defined(MFEM_USE_CUDA)
linalg/sparsemat.cpp:#endif // MFEM_USE_CUDA_OR_HIP
linalg/sparsemat.cpp:   if ((Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK)) && useGPUSparse)
linalg/sparsemat.cpp:#ifdef MFEM_USE_CUDA_OR_HIP
linalg/sparsemat.cpp:#if CUDA_VERSION >= 10010 || defined(MFEM_USE_HIP)
linalg/sparsemat.cpp:            MFEM_CUDA_or_HIP(_R_32F));
linalg/sparsemat.cpp:            MFEM_CUDA_or_HIP(_R_64F));
linalg/sparsemat.cpp:                                           MFEM_CUDA_or_HIP(_R_32F));
linalg/sparsemat.cpp:                                           MFEM_CUDA_or_HIP(_R_64F));
linalg/sparsemat.cpp:                                           MFEM_CUDA_or_HIP(_R_32F));
linalg/sparsemat.cpp:                                           MFEM_CUDA_or_HIP(_R_64F));
linalg/sparsemat.cpp:#endif // CUDA_VERSION >= 10010 || defined(MFEM_USE_HIP)
linalg/sparsemat.cpp:         MFEM_CUDA_or_HIP(_R_32F),
linalg/sparsemat.cpp:         MFEM_CUDA_or_HIP(_R_64F),
linalg/sparsemat.cpp:         MFEM_GPUSPARSE_ALG,
linalg/sparsemat.cpp:#if CUDA_VERSION >= 10010 || defined(MFEM_USE_HIP)
linalg/sparsemat.cpp:         MFEM_CUDA_or_HIP(_R_32F),
linalg/sparsemat.cpp:         MFEM_CUDA_or_HIP(_R_64F),
linalg/sparsemat.cpp:         MFEM_GPUSPARSE_ALG,
linalg/sparsemat.cpp:#endif // CUDA_VERSION >= 10010 || defined(MFEM_USE_HIP)
linalg/sparsemat.cpp:#endif // MFEM_USE_CUDA_OR_HIP
linalg/sparsemat.cpp:   ClearGPUSparse();
linalg/sparsemat.cpp:#ifdef MFEM_USE_CUDA_OR_HIP
linalg/sparsemat.cpp:   if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
linalg/sparsemat.cpp:#endif // MFEM_USE_CUDA_OR_HIP
linalg/CMakeLists.txt:  batched/gpu_blas.cpp
linalg/CMakeLists.txt:  batched/gpu_blas.hpp
linalg/sundials.cpp:#if defined(MFEM_USE_CUDA)
linalg/sundials.cpp:#include <nvector/nvector_cuda.h>
linalg/sundials.cpp:#if defined(MFEM_USE_CUDA)
linalg/sundials.cpp:#define SUN_Hip_OR_Cuda(X) X##_Cuda
linalg/sundials.cpp:#define SUN_HIP_OR_CUDA(X) X##_CUDA
linalg/sundials.cpp:#define SUN_Hip_OR_Cuda(X) X##_Hip
linalg/sundials.cpp:#define SUN_HIP_OR_CUDA(X) X##_HIP
linalg/sundials.cpp:#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
linalg/sundials.cpp:MFEM_DEPRECATED N_Vector SUN_Hip_OR_Cuda(N_VNewWithMemHelp)(sunindextype length,
linalg/sundials.cpp:   return SUN_Hip_OR_Cuda(N_VNewWithMemHelp)(length, use_managed_mem, helper);
linalg/sundials.cpp:#endif // MFEM_USE_CUDA || MFEM_USE_HIP
linalg/sundials.cpp:#if defined(MFEM_USE_MPI) && (defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
linalg/sundials.cpp:#endif // MFEM_USE_MPI && (MFEM_USE_CUDA || MFEM_USE_HIP)
linalg/sundials.cpp:#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
linalg/sundials.cpp:   h->ops->copy      = SUN_Hip_OR_Cuda(SUNMemoryHelper_Copy);
linalg/sundials.cpp:   h->ops->copyasync = SUN_Hip_OR_Cuda(SUNMemoryHelper_CopyAsync);
linalg/sundials.cpp:#endif // MFEM_USE_CUDA || MFEM_USE_HIP
linalg/sundials.cpp:#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
linalg/sundials.cpp:      case SUN_HIP_OR_CUDA(SUNDIALS_NVEC):
linalg/sundials.cpp:         SUN_Hip_OR_Cuda(N_VSetHostArrayPointer)(HostReadWrite(), local_x);
linalg/sundials.cpp:         SUN_Hip_OR_Cuda(N_VSetDeviceArrayPointer)(ReadWrite(), local_x);
linalg/sundials.cpp:         static_cast<SUN_Hip_OR_Cuda(N_VectorContent)>(GET_CONTENT(
linalg/sundials.cpp:#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
linalg/sundials.cpp:      case SUN_HIP_OR_CUDA(SUNDIALS_NVEC):
linalg/sundials.cpp:         double *h_ptr = SUN_Hip_OR_Cuda(N_VGetHostArrayPointer)(local_x);
linalg/sundials.cpp:         double *d_ptr = SUN_Hip_OR_Cuda(N_VGetDeviceArrayPointer)(local_x);
linalg/sundials.cpp:         size = SUN_Hip_OR_Cuda(N_VGetLength)(local_x);
linalg/sundials.cpp:#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
linalg/sundials.cpp:      x = SUN_Hip_OR_Cuda(N_VNewWithMemHelp)(0, UseManagedMemory(),
linalg/sundials.cpp:#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
linalg/sundials.cpp:         x = N_VMake_MPIPlusX(comm, SUN_Hip_OR_Cuda(N_VNewWithMemHelp)(0,
linalg/sundials.cpp:#endif // MFEM_USE_CUDA || MFEM_USE_HIP
linalg/sparsemat.hpp:   bool useGPUSparse = true; // Use cuSPARSE or hipSPARSE if available
linalg/sparsemat.hpp:   void InitGPUSparse();
linalg/sparsemat.hpp:#ifdef MFEM_USE_CUDA_OR_HIP
linalg/sparsemat.hpp:#if defined(MFEM_USE_CUDA)
linalg/sparsemat.hpp:#if CUDA_VERSION >= 10010
linalg/sparsemat.hpp:#else // CUDA_VERSION >= 10010
linalg/sparsemat.hpp:#endif // CUDA_VERSION >= 10010
linalg/sparsemat.hpp:#else // defined(MFEM_USE_CUDA)
linalg/sparsemat.hpp:#endif // defined(MFEM_USE_CUDA)
linalg/sparsemat.hpp:#endif // MFEM_USE_CUDA_OR_HIP
linalg/sparsemat.hpp:      InitGPUSparse();
linalg/sparsemat.hpp:       a CUDA or HIP backend.
linalg/sparsemat.hpp:   void UseGPUSparse(bool useGPUSparse_ = true) { useGPUSparse = useGPUSparse_;}
linalg/sparsemat.hpp:   /// Deprecated equivalent of UseGPUSparse().
linalg/sparsemat.hpp:   void UseCuSparse(bool useCuSparse_ = true) { UseGPUSparse(useCuSparse_); }
linalg/sparsemat.hpp:   void ClearGPUSparse();
linalg/sparsemat.hpp:   /// Deprecated equivalent of ClearGPUSparse().
linalg/sparsemat.hpp:   void ClearCuSparse() { ClearGPUSparse(); }
linalg/sparsemat.hpp:   /** For non-serial-CPU backends (e.g. GPU, OpenMP), multiplying by the
linalg/linalg.hpp:#include "batched/gpu_blas.hpp"
linalg/strumpack.cpp:::EnableGPU()
linalg/strumpack.cpp:   solver_->options().enable_gpu();
linalg/strumpack.cpp:::DisableGPU()
linalg/strumpack.cpp:   solver_->options().disable_gpu();
linalg/batched/gpu_blas.cpp:#include "gpu_blas.hpp"
linalg/batched/gpu_blas.cpp:#if defined(MFEM_USE_CUDA)
linalg/batched/gpu_blas.cpp:#define MFEM_GPUBLAS_PREFIX(stub) MFEM_CONCAT(MFEM_cu_or_hip(blas), S, stub)
linalg/batched/gpu_blas.cpp:#define MFEM_GPUBLAS_PREFIX(stub) MFEM_CONCAT(MFEM_cu_or_hip(blas), D, stub)
linalg/batched/gpu_blas.cpp:GPUBlas &GPUBlas::Instance()
linalg/batched/gpu_blas.cpp:   static GPUBlas instance;
linalg/batched/gpu_blas.cpp:GPUBlas::HandleType GPUBlas::Handle()
linalg/batched/gpu_blas.cpp:#ifndef MFEM_USE_CUDA_OR_HIP
linalg/batched/gpu_blas.cpp:GPUBlas::GPUBlas() { }
linalg/batched/gpu_blas.cpp:GPUBlas::~GPUBlas() { }
linalg/batched/gpu_blas.cpp:void GPUBlas::EnableAtomics() { }
linalg/batched/gpu_blas.cpp:void GPUBlas::DisableAtomics() { }
linalg/batched/gpu_blas.cpp:GPUBlas::GPUBlas()
linalg/batched/gpu_blas.cpp:   MFEM_VERIFY(status == MFEM_BLAS_SUCCESS, "Cannot initialize GPU BLAS.");
linalg/batched/gpu_blas.cpp:GPUBlas::~GPUBlas()
linalg/batched/gpu_blas.cpp:void GPUBlas::EnableAtomics()
linalg/batched/gpu_blas.cpp:   MFEM_VERIFY(status == MFEM_BLAS_SUCCESS, "GPU BLAS error.");
linalg/batched/gpu_blas.cpp:void GPUBlas::DisableAtomics()
linalg/batched/gpu_blas.cpp:   MFEM_VERIFY(status == MFEM_BLAS_SUCCESS, "GPU BLAS error.");
linalg/batched/gpu_blas.cpp:void GPUBlasBatchedLinAlg::AddMult(const DenseTensor &A, const Vector &x,
linalg/batched/gpu_blas.cpp:   const blasStatus_t status = MFEM_GPUBLAS_PREFIX(gemmStridedBatched)(
linalg/batched/gpu_blas.cpp:                                  GPUBlas::Handle(), op_A, op_B, m, k, n,
linalg/batched/gpu_blas.cpp:   MFEM_VERIFY(status == MFEM_BLAS_SUCCESS, "GPU BLAS error.");
linalg/batched/gpu_blas.cpp:void GPUBlasBatchedLinAlg::LUFactor(DenseTensor &A, Array<int> &P) const
linalg/batched/gpu_blas.cpp:   const blasStatus_t status = MFEM_GPUBLAS_PREFIX(getrfBatched)(
linalg/batched/gpu_blas.cpp:                                  GPUBlas::Handle(), n, d_A_ptrs, n, P.Write(),
linalg/batched/gpu_blas.cpp:void GPUBlasBatchedLinAlg::LUSolve(
linalg/batched/gpu_blas.cpp:   const blasStatus_t status = MFEM_GPUBLAS_PREFIX(getrsBatched)(
linalg/batched/gpu_blas.cpp:                                  GPUBlas::Handle(), MFEM_CU_or_HIP(BLAS_OP_N),
linalg/batched/gpu_blas.cpp:void GPUBlasBatchedLinAlg::Invert(DenseTensor &A) const
linalg/batched/gpu_blas.cpp:   status = MFEM_GPUBLAS_PREFIX(getrfBatched)(
linalg/batched/gpu_blas.cpp:               GPUBlas::Handle(), n, d_LU_ptrs, n, P.Write(),
linalg/batched/gpu_blas.cpp:   status = MFEM_GPUBLAS_PREFIX(getriBatched)(
linalg/batched/gpu_blas.cpp:               GPUBlas::Handle(), n, d_LU_ptrs, n, P.ReadWrite(), d_A_ptrs, n,
linalg/batched/batched.cpp:#include "gpu_blas.hpp"
linalg/batched/batched.cpp:   if (Device::Allows(mfem::Backend::CUDA_MASK | mfem::Backend::HIP_MASK))
linalg/batched/batched.cpp:#ifdef MFEM_USE_CUDA_OR_HIP
linalg/batched/batched.cpp:      backends[GPU_BLAS].reset(new GPUBlasBatchedLinAlg);
linalg/batched/batched.cpp:#elif defined(MFEM_USE_CUDA_OR_HIP)
linalg/batched/batched.cpp:      active_backend = GPU_BLAS;
linalg/batched/gpu_blas.hpp:#ifndef MFEM_GPU_BLAS_LINALG
linalg/batched/gpu_blas.hpp:#define MFEM_GPU_BLAS_LINALG
linalg/batched/gpu_blas.hpp:#if defined(MFEM_USE_CUDA)
linalg/batched/gpu_blas.hpp:/// If MFEM is compiled without CUDA or HIP, then this class has no effect.
linalg/batched/gpu_blas.hpp:class GPUBlas
linalg/batched/gpu_blas.hpp:#if defined(MFEM_USE_CUDA)
linalg/batched/gpu_blas.hpp:   GPUBlas(); ///< Create the handle.
linalg/batched/gpu_blas.hpp:   ~GPUBlas(); ///< Destroy the handle.
linalg/batched/gpu_blas.hpp:   static GPUBlas &Instance(); ///< Get the unique instnce.
linalg/batched/gpu_blas.hpp:#ifdef MFEM_USE_CUDA_OR_HIP
linalg/batched/gpu_blas.hpp:class GPUBlasBatchedLinAlg : public BatchedLinAlgBase
linalg/batched/gpu_blas.hpp:#endif // MFEM_USE_CUDA_OR_HIP
linalg/batched/gpu_blas.hpp:#endif // MFEM_GPU_BLAS_LINALG
linalg/batched/solver.hpp:/// time), but solving the system is more efficient in parallel (e.g. on GPUs).
linalg/batched/batched.hpp:/// using accelerated algorithms (GPU BLAS or MAGMA). Accessed using static
linalg/batched/batched.hpp:   /// order: MAGMA, GPU_BLAS, NATIVE.
linalg/batched/batched.hpp:      /// CUDA or HIP. Not available otherwise.
linalg/batched/batched.hpp:      GPU_BLAS,
linalg/batched/batched.hpp:   /// compiled with, and whether the the CUDA/HIP device is enabled.
linalg/slepc.cpp:   if (mfem::Device::Allows(mfem::Backend::CUDA_MASK))
linalg/slepc.cpp:      // Tell PETSc to use the same CUDA device as MFEM:
linalg/slepc.cpp:      ierr = PetscOptionsSetValue(NULL,"-cuda_device",
linalg/ginkgo.cpp:#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
linalg/ginkgo.cpp:      case GinkgoExecutor::CUDA:
linalg/ginkgo.cpp:         if (gko::CudaExecutor::get_num_devices() > 0)
linalg/ginkgo.cpp:#ifdef MFEM_USE_CUDA
linalg/ginkgo.cpp:            MFEM_GPU_CHECK(cudaGetDevice(&current_device));
linalg/ginkgo.cpp:               executor = gko::CudaExecutor::create(current_device,
linalg/ginkgo.cpp:               executor = gko::CudaExecutor::create(current_device,
linalg/ginkgo.cpp:            MFEM_ABORT("gko::CudaExecutor::get_num_devices() did not report "
linalg/ginkgo.cpp:            MFEM_GPU_CHECK(hipGetDevice(&current_device));
linalg/ginkgo.cpp:// related CPU Executor (only applicable to GPU backends).
linalg/ginkgo.cpp:      case GinkgoExecutor::CUDA:
linalg/ginkgo.cpp:         if (gko::CudaExecutor::get_num_devices() > 0)
linalg/ginkgo.cpp:#ifdef MFEM_USE_CUDA
linalg/ginkgo.cpp:            MFEM_GPU_CHECK(cudaGetDevice(&current_device));
linalg/ginkgo.cpp:               executor = gko::CudaExecutor::create(current_device,
linalg/ginkgo.cpp:               executor = gko::CudaExecutor::create(current_device,
linalg/ginkgo.cpp:            MFEM_ABORT("gko::CudaExecutor::get_num_devices() did not report "
linalg/ginkgo.cpp:            MFEM_GPU_CHECK(hipGetDevice(&current_device));
linalg/ginkgo.cpp:   if (mfem_device.Allows(Backend::CUDA_MASK))
linalg/ginkgo.cpp:      if (gko::CudaExecutor::get_num_devices() > 0)
linalg/ginkgo.cpp:#ifdef MFEM_USE_CUDA
linalg/ginkgo.cpp:         MFEM_GPU_CHECK(cudaGetDevice(&current_device));
linalg/ginkgo.cpp:            executor = gko::CudaExecutor::create(current_device,
linalg/ginkgo.cpp:            executor = gko::CudaExecutor::create(current_device,
linalg/ginkgo.cpp:         MFEM_ABORT("gko::CudaExecutor::get_num_devices() did not report "
linalg/ginkgo.cpp:         MFEM_GPU_CHECK(hipGetDevice(&current_device));
linalg/ginkgo.cpp:// applicable to GPU backends).
linalg/ginkgo.cpp:   if (mfem_device.Allows(Backend::CUDA_MASK))
linalg/ginkgo.cpp:      if (gko::CudaExecutor::get_num_devices() > 0)
linalg/ginkgo.cpp:#ifdef MFEM_USE_CUDA
linalg/ginkgo.cpp:         MFEM_GPU_CHECK(cudaGetDevice(&current_device));
linalg/ginkgo.cpp:            executor = gko::CudaExecutor::create(current_device,
linalg/ginkgo.cpp:            executor = gko::CudaExecutor::create(current_device,
linalg/ginkgo.cpp:         MFEM_ABORT("gko::CudaExecutor::get_num_devices() did not report "
linalg/ginkgo.cpp:         MFEM_GPU_CHECK(hipGetDevice(&current_device));
linalg/ginkgo.cpp:   // on CPU or GPU.
linalg/ginkgo.cpp:   // Additionally, if the logger is logging on the gpu, it is necessary to copy
linalg/ginkgo.cpp:   // on CPU or GPU.
linalg/amgxsolver.hpp:   MFEM wrapper for Nvidia's multigrid library, AmgX (github.com/NVIDIA/AMGX)
linalg/amgxsolver.hpp:   AmgX requires building MFEM with CUDA, and AMGX enabled. For distributed
linalg/amgxsolver.hpp:   CUDA is required for building, the AmgX solver is compatible with a MFEM CPU
linalg/amgxsolver.hpp:   Serial - Takes a SparseMatrix solves on a single GPU and assumes no MPI
linalg/amgxsolver.hpp:   Exclusive GPU - Takes a HypreParMatrix and assumes each MPI rank is paired
linalg/amgxsolver.hpp:   with an Nvidia GPU.
linalg/amgxsolver.hpp:   MPI ranks, and GPUs. Specifically, MPI ranks are grouped with GPUs and a
linalg/amgxsolver.hpp:   with exclusive GPU or MPI teams modes.
linalg/amgxsolver.hpp:      AmgXWrapper: An interface between PETSc and the NVIDIA AmgX library.
linalg/amgxsolver.hpp:      and verbosity. Pairs each MPI rank with one GPU.
linalg/amgxsolver.hpp:      and verbosity. Creates MPI teams around GPUs to support more ranks than
linalg/amgxsolver.hpp:      GPUs. Consolidates linear solver data to avoid multiple ranks sharing
linalg/amgxsolver.hpp:      GPUs. Requires specifying the number  of devices in each compute node as
linalg/amgxsolver.hpp:      GPU per rank after the solver configuration has been established,
linalg/amgxsolver.hpp:   void InitExclusiveGPU(const MPI_Comm &comm);
linalg/amgxsolver.hpp:   void SetMatrixMPIGPUExclusive(const HypreParMatrix &A,
linalg/amgxsolver.hpp:   // Number of local GPU devices used by AmgX.
linalg/amgxsolver.hpp:   // The ID of corresponding GPU device used by this MPI process.
linalg/amgxsolver.hpp:   int                     gpuProc = MPI_UNDEFINED;
linalg/amgxsolver.hpp:   MPI_Comm                gpuWorld;
linalg/amgxsolver.hpp:   int                     gpuWorldSize;
linalg/amgxsolver.hpp:   int                     myGpuWorldRank;
linalg/amgxsolver.hpp:   /// Set the ID of the corresponding GPU used by this process.
linalg/amgxsolver.hpp:   std::string mpi_gpu_mode;
linalg/strumpack.hpp:   /** @brief Enable GPU off-loading available if STRUMPACK was compiled with
linalg/strumpack.hpp:       CUDA.
linalg/strumpack.hpp:   void EnableGPU();
linalg/strumpack.hpp:   /** @brief Disable GPU off-loading available if STRUMPACK was compiled with
linalg/strumpack.hpp:       CUDA.
linalg/strumpack.hpp:   void DisableGPU();
linalg/operator.cpp:   // typically z and w are large vectors, so use the device (GPU) to perform
linalg/tensor.hpp:#if defined(__CUDACC__)
linalg/tensor.hpp:#if __CUDAVER__ >= 75000
linalg/tensor.hpp:#else  //__CUDACC__
examples/ex13p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex13p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex1p.cpp://               mpirun -np 4 ex1p -pa -d cuda
examples/ex1p.cpp://               mpirun -np 4 ex1p -fa -d cuda
examples/ex1p.cpp://               mpirun -np 4 ex1p -pa -d occa-cuda
examples/ex1p.cpp://             * mpirun -np 4 ex1p -pa -d ceed-cuda
examples/ex1p.cpp://               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared
examples/ex1p.cpp://               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/square-mixed.mesh
examples/ex1p.cpp://               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/fichera-mixed.mesh
examples/ex1p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex1p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex1p.cpp:      // Sort the matrix column indices when running on GPU or with OpenMP (i.e.
examples/ex4p.cpp://               mpirun -np 4 ex4p -m ../data/star.mesh -pa -d cuda
examples/ex4p.cpp://               mpirun -np 4 ex4p -m ../data/star.mesh -pa -d raja-cuda
examples/ex4p.cpp://               mpirun -np 4 ex4p -m ../data/beam-hex.mesh -pa -d cuda
examples/ex4p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex4p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex26.cpp://               ex26 -d cuda
examples/ex26.cpp://               ex26 -d raja-cuda
examples/ex26.cpp://               ex26 -d occa-cuda
examples/ex26.cpp://               ex26 -d ceed-cuda
examples/ex26.cpp://               ex26 -m ../data/beam-hex.mesh -d cuda
examples/ex26.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/ex26.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex5p.cpp://               mpirun -np 4 ex5p -m ../data/star.mesh -r 2 -pa -d cuda
examples/ex5p.cpp://               mpirun -np 4 ex5p -m ../data/star.mesh -r 2 -pa -d raja-cuda
examples/ex5p.cpp://               mpirun -np 4 ex5p -m ../data/beam-hex.mesh -pa -d cuda
examples/ex5p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex5p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex5.cpp://               ex5 -m ../data/star.mesh -pa -d cuda
examples/ex5.cpp://               ex5 -m ../data/star.mesh -pa -d raja-cuda
examples/ex5.cpp://               ex5 -m ../data/beam-hex.mesh -pa -d cuda
examples/ex5.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/ex5.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex34p.cpp://               mpirun -np 4 ex34p -o 2 -hex -pa -d cuda
examples/ex34p.cpp://               mpirun -np 4 ex34p -o 2 -no-pa -d cuda
examples/ex34p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex34p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex3p.cpp://               mpirun -np 4 ex3p -m ../data/star.mesh -pa -d cuda
examples/ex3p.cpp://               mpirun -np 4 ex3p -m ../data/star.mesh -no-pa -d cuda
examples/ex3p.cpp://               mpirun -np 4 ex3p -m ../data/star.mesh -pa -d raja-cuda
examples/ex3p.cpp://               mpirun -np 4 ex3p -m ../data/beam-hex.mesh -pa -d cuda
examples/ex3p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex3p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex22p.cpp://               mpirun -np 4 ex22p -m ../data/inline-quad.mesh -o 1 -p 1 -pa -d cuda
examples/ex22p.cpp://               mpirun -np 4 ex22p -m ../data/inline-hex.mesh -o 1 -p 2 -pa -d cuda
examples/ex22p.cpp://               mpirun -np 4 ex22p -m ../data/star.mesh -o 2 -sigma 10.0 -pa -d cuda
examples/ex22p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex22p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex7p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex7p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/amgx/ex1p.cpp://               mpirun -np 4 ex1p -d cuda
examples/amgx/ex1p.cpp:                  "--amgx-mpi-gpu-exclusive", "--amgx-mpi-gpu-exclusive",
examples/amgx/ex1p.cpp:                  "Create MPI teams when using AmgX to load balance between ranks and GPUs.");
examples/amgx/ex1p.cpp:   args.AddOption(&ndevices, "-nd","--gpus-per-node-in-teams-mode",
examples/amgx/ex1p.cpp:                  "Number of GPU devices per node (Only used if amgx_mpi_teams is true).");
examples/amgx/ex1p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/amgx/ex1p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/amgx/ex1p.cpp:         // Forms MPI teams to load balance between MPI ranks and GPUs
examples/amgx/ex1p.cpp:         // Assumes each MPI rank is paired with a GPU
examples/amgx/ex1p.cpp:         amgx.InitExclusiveGPU(MPI_COMM_WORLD);
examples/amgx/ex1.cpp://               ex1 -d cuda
examples/amgx/ex1.cpp://               ex1 --amgx-file multi_gs.json --amgx-solver -d cuda
examples/amgx/ex1.cpp://               ex1 --amgx-file precon.json --amgx-preconditioner -d cuda
examples/amgx/ex1.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/amgx/ex1.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/amgx/README:use of MFEM features based on NVIDIA's multigrid library AmgX.
examples/ex2p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex2p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex28p.cpp:#ifdef HYPRE_USING_GPU
examples/ex28p.cpp:        << "is NOT supported with the GPU version of hypre.\n\n";
examples/ex25.cpp://               ex25 -o 2 -f 8.0 -ref 3 -prob 4 -m ../data/inline-quad.mesh -pa -d cuda
examples/ex25.cpp://               ex25 -o 2 -f 2.0 -ref 1 -prob 4 -m ../data/inline-hex.mesh -pa -d cuda
examples/ex25.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/ex25.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/makefile:%-test-par-cuda: %
examples/makefile:	@$(call mfem-test,$<, $(RUN_MPI), Parallel CUDA example,-d cuda)
examples/makefile:%-test-seq-cuda: %
examples/makefile:	@$(call mfem-test,$<,, Serial CUDA example,-d cuda)
examples/makefile:ex14-test-seq-cuda: ex14
examples/makefile:	@$(call mfem-test,$<,, Serial CUDA example,-r 2 -pa -d cuda)
examples/makefile:ex14p-test-par-cuda: ex14p
examples/makefile:	@$(call mfem-test,$<, $(RUN_MPI), Parallel CUDA example,-rs 2 -rp 0 -pa -d cuda)
examples/CMakeLists.txt:if (HYPRE_USING_CUDA OR HYPRE_USING_HIP)
examples/CMakeLists.txt:  # Add CUDA/HIP tests.
examples/CMakeLists.txt:  if (MFEM_USE_CUDA)
examples/CMakeLists.txt:    set(MFEM_TEST_DEVICE "cuda")
examples/petsc/ex1p.cpp://               mpirun -np 4 ex1p -pa -d cuda --petscopts rc_ex1p_device
examples/petsc/ex1p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/petsc/ex1p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/petsc/ex4p.cpp:   // 2b. Enable hardware devices such as GPUs, and programming models such as
examples/petsc/ex4p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/petsc/ex5p.cpp:   // 2b. Enable hardware devices such as GPUs, and programming models such as
examples/petsc/ex5p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/petsc/rc_ex11p_lobpcg_device:#sor is not implemented for GPU, use jacobi
examples/petsc/ex10p.cpp:   // 2b. Enable hardware devices such as GPUs, and programming models such as
examples/petsc/ex10p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/petsc/ex11p.cpp:   // 2b. Enable hardware devices such as GPUs, and programming models such as
examples/petsc/ex11p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/petsc/rc_ex1p_deviceamg:#sor is not implemented for GPU, use jacobi
examples/petsc/ex3p.cpp:   // 2b. Enable hardware devices such as GPUs, and programming models such as
examples/petsc/ex3p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/petsc/ex2p.cpp:   // 2b. Enable hardware devices such as GPUs, and programming models such as
examples/petsc/ex2p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/petsc/makefile:TESTNAME_CUDA = Parallel CUDA PETSc example
examples/petsc/makefile:EX1_ARGS_CUDA         := -m ../../data/star.mesh --usepetsc --partial-assembly --device cuda --petscopts rc_ex1p_device
examples/petsc/makefile:EX1_ARGS_CUDAAMG      := -m ../../data/star.mesh --usepetsc --device cuda --petscopts rc_ex1p_deviceamg
examples/petsc/makefile:EX9_ES_ARGS_CUDA      := -m ../../data/periodic-hexagon.mesh --usepetsc --petscopts rc_ex9p_expl_device --no-step --partial-assembly --device cuda
examples/petsc/makefile:EX11_ARGS_LOBPCG_CUDA := -m ../../data/star.mesh --useslepc --slepcopts rc_ex11p_lobpcg_device --device cuda
examples/petsc/makefile:ifeq ($(MFEM_USE_CUDA),YES)
examples/petsc/makefile:	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME_CUDA),$(EX1_ARGS_CUDA))
examples/petsc/makefile:	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME_CUDA),$(EX1_ARGS_CUDAAMG))
examples/petsc/makefile:ifeq ($(MFEM_USE_CUDA),YES)
examples/petsc/makefile:	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME_CUDA),$(EX9_ES_ARGS_CUDA))
examples/petsc/makefile:ifeq ($(MFEM_USE_CUDA),YES)
examples/petsc/makefile:	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME_CUDA),$(EX11_ARGS_LOBPCG_CUDA))
examples/petsc/CMakeLists.txt:set(EX1_ARGS_CUDA    -m ../../data/star.mesh --usepetsc --partial-assembly --device cuda --petscopts rc_ex1p_device)
examples/petsc/CMakeLists.txt:set(EX1_ARGS_CUDAAMG -m ../../data/star.mesh --usepetsc --device cuda --petscopts rc_ex1p_deviceamg)
examples/petsc/CMakeLists.txt:set(EX9_ES_ARGS_CUDA -m ../../data/periodic-hexagon.mesh --usepetsc --petscopts rc_ex9p_expl_device --no-step --partial-assembly --device cuda)
examples/petsc/CMakeLists.txt:  set(EX11_ARGS_LOBPCG_CUDA -m ../../data/star.mesh --useslepc --slepcopts rc_ex11p_lobpcg_device --device cuda)
examples/petsc/CMakeLists.txt:  # CUDA/HIP tests
examples/petsc/CMakeLists.txt:  if (MFEM_USE_CUDA)
examples/petsc/CMakeLists.txt:      EX1_ARGS_CUDA EX1_ARGS_CUDAAMG EX9_ES_ARGS_CUDA)
examples/petsc/CMakeLists.txt:      list(APPEND TEST_OPTIONS_VARS EX11_ARGS_LOBPCG_CUDA)
examples/petsc/ex6p.cpp:   // 2b. Enable hardware devices such as GPUs, and programming models such as
examples/petsc/ex6p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex14.cpp://               ex14 -pa -r 2 -d cuda -o 3
examples/ex14.cpp://               ex14 -pa -r 2 -d cuda -o 3 -m ../data/fichera.mesh
examples/ex14.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/ex14.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex14p.cpp://               mpirun -np 4 ex14p -pa -rs 2 -rp 0 -d cuda -o 3
examples/ex14p.cpp://               mpirun -np 4 ex14p -pa -rs 2 -rp 0 -d cuda -m ../data/fichera.mesh -o 3
examples/ex24p.cpp://               mpirun -np 4 ex24p -m ../data/star.mesh -pa -d cuda
examples/ex24p.cpp://               mpirun -np 4 ex24p -m ../data/star.mesh -pa -d raja-cuda
examples/ex24p.cpp://               mpirun -np 4 ex24p -m ../data/beam-hex.mesh -pa -d cuda
examples/ex24p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex24p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex9p.cpp://    mpirun -np 4 ex9p -pa -m ../data/periodic-cube.mesh -d cuda
examples/ex9p.cpp://    mpirun -np 4 ex9p -ea -m ../data/periodic-cube.mesh -d cuda
examples/ex9p.cpp://    mpirun -np 4 ex9p -fa -m ../data/periodic-cube.mesh -d cuda
examples/ex9p.cpp://    mpirun -np 4 ex9p -pa -m ../data/amr-quad.mesh -p 1 -rp 1 -dt 0.002 -tf 9 -d cuda
examples/sundials/makefile:SERIAL_CUDA_NAME := Serial SUNDIALS CUDA example
examples/sundials/makefile:PARALLEL_CUDA_NAME := Parallel SUNDIALS CUDA example
examples/sundials/makefile:%-test-par-cuda: %
examples/sundials/makefile:	@$(call mfem-test,$<, $(RUN_MPI), $(PARALLEL_CUDA_NAME),-d cuda)
examples/sundials/makefile:%-test-seq-cuda: %
examples/sundials/makefile:	@$(call mfem-test,$<,, $(SERIAL_CUDA_NAME),-d cuda)
examples/sundials/makefile:ex9-test-seq-cuda: ex9
examples/sundials/makefile:	@$(call mfem-test,$<,, $(SERIAL_CUDA_NAME),-d cuda $(EX9_ARGS))
examples/sundials/makefile:ex9p-test-par-cuda: ex9p
examples/sundials/makefile:	@$(call mfem-test,$<, $(RUN_MPI), $(PARALLEL_CUDA_NAME),-d cuda \
examples/sundials/CMakeLists.txt:  # Add CUDA/HIP tests.
examples/sundials/CMakeLists.txt:  if (MFEM_USE_CUDA)
examples/sundials/CMakeLists.txt:    set(MFEM_TEST_DEVICE "cuda")
examples/sundials/ex9p.cpp://    mpirun -np 4 ex9p -pa -m ../../data/periodic-cube.mesh -d cuda
examples/sundials/ex9p.cpp://    mpirun -np 4 ex9p -ea -m ../../data/periodic-cube.mesh -d cuda
examples/sundials/ex9p.cpp://    mpirun -np 4 ex9p -fa -m ../../data/periodic-cube.mesh -d cuda
examples/sundials/ex9.cpp://    ex9 -pa -m ../../data/periodic-cube.mesh -d cuda
examples/sundials/ex9.cpp://    ex9 -ea -m ../../data/periodic-cube.mesh -d cuda
examples/sundials/ex9.cpp://    ex9 -fa -m ../../data/periodic-cube.mesh -d cuda
examples/ex1.cpp://               ex1 -pa -d cuda
examples/ex1.cpp://               ex1 -fa -d cuda
examples/ex1.cpp://               ex1 -pa -d raja-cuda
examples/ex1.cpp://               ex1 -pa -d occa-cuda
examples/ex1.cpp://             * ex1 -pa -d ceed-cuda
examples/ex1.cpp://               ex1 -pa -d ceed-cuda:/gpu/cuda/shared
examples/ex1.cpp://               ex1 -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/square-mixed.mesh
examples/ex1.cpp://               ex1 -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/fichera-mixed.mesh
examples/ex1.cpp://               ex1 -m ../data/beam-hex.mesh -pa -d cuda
examples/ex1.cpp://               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cuda:/gpu/cuda/ref
examples/ex1.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/ex1.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex1.cpp:      // Sort the matrix column indices when running on GPU or with OpenMP (i.e.
examples/ex4.cpp://               ex4 -m ../data/star.mesh -pa -d cuda
examples/ex4.cpp://               ex4 -m ../data/star.mesh -pa -d raja-cuda
examples/ex4.cpp://               ex4 -m ../data/beam-hex.mesh -pa -d cuda
examples/ex4.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/ex4.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex24.cpp://               ex24 -m ../data/star.mesh -pa -d cuda
examples/ex24.cpp://               ex24 -m ../data/star.mesh -pa -d raja-cuda
examples/ex24.cpp://               ex24 -m ../data/beam-hex.mesh -pa -d cuda
examples/ex24.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/ex24.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex6p.cpp://               mpirun -np 4 ex6p -pa -d cuda
examples/ex6p.cpp://               mpirun -np 4 ex6p -pa -d occa-cuda
examples/ex6p.cpp://             * mpirun -np 4 ex6p -pa -d ceed-cuda
examples/ex6p.cpp://               mpirun -np 4 ex6p -pa -d ceed-cuda:/gpu/cuda/shared
examples/ex6p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex6p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex25p.cpp://               mpirun -np 4 ex25p -o 1 -f 3.0 -rs 3 -rp 1 -prob 2 -pa -d cuda
examples/ex25p.cpp://               mpirun -np 4 ex25p -o 2 -f 1.0 -rs 1 -rp 1 -prob 3 -pa -d cuda
examples/ex25p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex25p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex19p.cpp:#ifdef HYPRE_USING_GPU
examples/ex19p.cpp:        << "is NOT supported with the GPU version of hypre.\n\n";
examples/ex19p.cpp:#if !defined(HYPRE_USING_GPU)
examples/ex19p.cpp:         // Not available yet when hypre is built with GPU support
examples/ex9.cpp://    ex9 -pa -m ../data/periodic-cube.mesh -d cuda
examples/ex9.cpp://    ex9 -ea -m ../data/periodic-cube.mesh -d cuda
examples/ex9.cpp://    ex9 -fa -m ../data/periodic-cube.mesh -d cuda
examples/ex9.cpp://    ex9 -pa -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9 -d cuda
examples/ex34.cpp://               ex34 -o 2 -pa -hex -d cuda
examples/ex34.cpp://               ex34 -o 2 -no-pa -d cuda
examples/ex34.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/ex34.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ginkgo/ex1.cpp://               ex1 -pa -d cuda
examples/ginkgo/ex1.cpp://               ex1 -pa -d raja-cuda
examples/ginkgo/ex1.cpp://               ex1 -pa -d occa-cuda
examples/ginkgo/ex1.cpp://               ex1 -m ../../data/beam-hex.mesh -pa -d cuda
examples/ginkgo/ex1.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/ginkgo/ex1.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ginkgo/README:for GPU and manycore nodes.
examples/caliper/ex1p.cpp://               mpirun -np 4 ex1p -pa -d cuda
examples/caliper/ex1p.cpp://               mpirun -np 4 ex1p -pa -d occa-cuda
examples/caliper/ex1p.cpp://             * mpirun -np 4 ex1p -pa -d ceed-cuda
examples/caliper/ex1p.cpp://               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared
examples/caliper/ex1p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/caliper/ex1p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/caliper/ex1.cpp://               ex1 -pa -d cuda
examples/caliper/ex1.cpp://               ex1 -pa -d raja-cuda
examples/caliper/ex1.cpp://               ex1 -pa -d occa-cuda
examples/caliper/ex1.cpp://             * ex1 -pa -d ceed-cuda
examples/caliper/ex1.cpp://               ex1 -pa -d ceed-cuda:/gpu/cuda/shared
examples/caliper/ex1.cpp://               ex1 -m ../data/beam-hex.mesh -pa -d cuda
examples/caliper/ex1.cpp://               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cuda:/gpu/cuda/ref
examples/caliper/ex1.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/caliper/ex1.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex22.cpp://               ex22 -m ../data/inline-quad.mesh -o 3 -p 1 -pa -d cuda
examples/ex22.cpp://               ex22 -m ../data/inline-hex.mesh -o 2 -p 2 -pa -d cuda
examples/ex22.cpp://               ex22 -m ../data/star.mesh -r 1 -o 2 -sigma 10.0 -pa -d cuda
examples/ex22.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/ex22.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex3.cpp://               ex3 -m ../data/star.mesh -pa -d cuda
examples/ex3.cpp://               ex3 -m ../data/star.mesh -pa -d raja-cuda
examples/ex3.cpp://               ex3 -m ../data/beam-hex.mesh -pa -d cuda
examples/ex3.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/ex3.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/superlu/ex1p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/superlu/ex1p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex6.cpp://               ex6 -pa -d cuda
examples/ex6.cpp://               ex6 -pa -d occa-cuda
examples/ex6.cpp://             * ex6 -pa -d ceed-cuda
examples/ex6.cpp://               ex6 -pa -d ceed-cuda:/gpu/cuda/shared
examples/ex6.cpp:   // 2. Enable hardware devices such as GPUs, and programming models such as
examples/ex6.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex35p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex35p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
examples/ex26p.cpp://               mpirun -np 4 ex26p -d cuda
examples/ex26p.cpp://               mpirun -np 4 ex26p -d occa-cuda
examples/ex26p.cpp://               mpirun -np 4 ex26p -d ceed-cuda
examples/ex26p.cpp:   // 3. Enable hardware devices such as GPUs, and programming models such as
examples/ex26p.cpp:   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
general/forall.hpp:struct DofQuadLimits_CUDA
general/forall.hpp:#if defined(__CUDA_ARCH__)
general/forall.hpp:using DofQuadLimits = internal::DofQuadLimits_CUDA;
general/forall.hpp:/// configured device (e.g. when the user has selected GPU execution at
general/forall.hpp:      if (Device::Allows(Backend::CUDA_MASK)) { Populate<internal::DofQuadLimits_CUDA>(); }
general/forall.hpp:   /// @a T should be one of DofQuadLimits_CUDA, DofQuadLimits_HIP, or
general/forall.hpp:#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
general/forall.hpp:// MFEM_GPU_FORALL: "parallel for" executed with CUDA or HIP based on the MFEM
general/forall.hpp:// build-time configuration (MFEM_USE_CUDA or MFEM_USE_HIP). If neither CUDA nor
general/forall.hpp:#if defined(MFEM_USE_CUDA)
general/forall.hpp:#define MFEM_GPU_FORALL(i, N,...) CuWrap1D(N, [=] MFEM_DEVICE      \
general/forall.hpp:#define MFEM_GPU_FORALL(i, N,...) HipWrap1D(N, [=] MFEM_DEVICE     \
general/forall.hpp:#define MFEM_GPU_FORALL(i, N,...) do { } while (false)
general/forall.hpp:// interfaces supporting RAJA, CUDA, OpenMP, and sequential backends.
general/forall.hpp:// MFEM_FORALL with a 2D CUDA block
general/forall.hpp:// MFEM_FORALL with a 3D CUDA block
general/forall.hpp:// MFEM_FORALL with a 3D CUDA block and grid
general/forall.hpp:/// RAJA Cuda and Hip backends
general/forall.hpp:#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)
general/forall.hpp:using cuda_launch_policy =
general/forall.hpp:   RAJA::LaunchPolicy<RAJA::cuda_launch_t<true>>;
general/forall.hpp:using cuda_teams_x =
general/forall.hpp:   RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;
general/forall.hpp:using cuda_threads_z =
general/forall.hpp:   RAJA::LoopPolicy<RAJA::cuda_thread_z_direct>;
general/forall.hpp:#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)
general/forall.hpp:template <const int BLOCKS = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:   RAJA::forall<RAJA::cuda_exec<BLOCKS,true>>(RAJA::RangeSegment(0,N),d_body);
general/forall.hpp:   launch<cuda_launch_policy>
general/forall.hpp:      loop<cuda_teams_x>(ctx, RangeSegment(0, G), [&] (const int n)
general/forall.hpp:         loop<cuda_threads_z>(ctx, RangeSegment(0, BZ), [&] (const int tz)
general/forall.hpp:   MFEM_GPU_CHECK(cudaGetLastError());
general/forall.hpp:   launch<cuda_launch_policy>
general/forall.hpp:      loop<cuda_teams_x>(ctx, RangeSegment(0, N), d_body);
general/forall.hpp:   MFEM_GPU_CHECK(cudaGetLastError());
general/forall.hpp:   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:   MFEM_GPU_CHECK(hipGetLastError());
general/forall.hpp:   MFEM_GPU_CHECK(hipGetLastError());
general/forall.hpp:   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:/// CUDA backend
general/forall.hpp:#ifdef MFEM_USE_CUDA
general/forall.hpp:template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:   MFEM_GPU_CHECK(cudaGetLastError());
general/forall.hpp:   MFEM_GPU_CHECK(cudaGetLastError());
general/forall.hpp:   MFEM_GPU_CHECK(cudaGetLastError());
general/forall.hpp:   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:#endif // MFEM_USE_CUDA
general/forall.hpp:   MFEM_GPU_CHECK(hipGetLastError());
general/forall.hpp:   MFEM_GPU_CHECK(hipGetLastError());
general/forall.hpp:   MFEM_GPU_CHECK(hipGetLastError());
general/forall.hpp:   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
general/forall.hpp:#if defined(MFEM_USE_RAJA) && defined(RAJA_ENABLE_CUDA)
general/forall.hpp:   // If Backend::RAJA_CUDA is allowed, use it
general/forall.hpp:   if (Device::Allows(Backend::RAJA_CUDA))
general/forall.hpp:#ifdef MFEM_USE_CUDA
general/forall.hpp:   // If Backend::CUDA is allowed, use it
general/forall.hpp:   if (Device::Allows(Backend::CUDA))
general/forall.hpp:// Function mfem::hypre_forall_gpu() similar to mfem::forall, but it always
general/forall.hpp:// executes on the GPU device that hypre was configured with at build time.
general/forall.hpp:#if defined(HYPRE_USING_GPU)
general/forall.hpp:inline void hypre_forall_gpu(int N, lambda &&body)
general/forall.hpp:#if defined(HYPRE_USING_CUDA)
general/forall.hpp:#error Unknown HYPRE GPU backend!
general/forall.hpp:// device, CPU or GPU, that hypre was configured with at build time (when the
general/forall.hpp:// HYPRE version is < 2.31.0) or at runtime (when HYPRE was configured with GPU
general/forall.hpp:#if !defined(HYPRE_USING_GPU)
general/forall.hpp:   hypre_forall_gpu(N, body);
general/forall.hpp:#else // HYPRE_USING_GPU is defined and MFEM_HYPRE_VERSION >= 23100
general/forall.hpp:   if (!HypreUsingGPU())
general/forall.hpp:      hypre_forall_gpu(N, body);
general/forall.hpp:// on the result of HypreUsingGPU().
general/forall.hpp:   return HypreUsingGPU() ? MemoryClass::DEVICE : MemoryClass::HOST;
general/mem_manager.hpp:   MANAGED,        /**< Managed memory; using CUDA or HIP *MallocManaged
general/mem_manager.hpp:   DEVICE,         ///< Device memory; using CUDA or HIP *Malloc and *Free
general/mem_manager.hpp:   /// memory type, e.g. CUDA (mt will not be HOST).
general/mem_manager.hpp:#if !defined(HYPRE_USING_GPU)
general/mem_manager.hpp:#else // HYPRE_USING_GPU is defined and MFEM_HYPRE_VERSION >= 23100
general/mem_manager.hpp:/// Return true if HYPRE is configured to use GPU
general/mem_manager.hpp:inline bool HypreUsingGPU()
general/mem_manager.hpp:#if !defined(HYPRE_USING_GPU)
general/mem_manager.hpp:#else // HYPRE_USING_GPU is defined and MFEM_HYPRE_VERSION >= 23100
general/mem_manager.hpp:#if !defined(HYPRE_USING_GPU)
general/mem_manager.hpp:         // When HYPRE_USING_GPU is defined and HYPRE < 2.31.0, we always
general/mem_manager.hpp:#else // HYPRE_USING_GPU is defined and MFEM_HYPRE_VERSION >= 23100
general/mem_manager.hpp:         MemoryManager::Exists() && HypreUsingGPU()
general/version.cpp:#ifdef MFEM_USE_CUDA
general/version.cpp:      "MFEM_USE_CUDA\n"
general/occa.cpp:#if defined(MFEM_USE_CUDA) && OCCA_CUDA_ENABLED
general/occa.cpp:#include <occa/modes/cuda/utils.hpp>
general/occa.cpp:#if defined(MFEM_USE_CUDA) && OCCA_CUDA_ENABLED
general/occa.cpp:   // If OCCA_CUDA is allowed, it will be used since it has the highest priority
general/occa.cpp:   if (Device::Allows(Backend::OCCA_CUDA))
general/occa.cpp:      return occa::cuda::wrapMemory(internal::occaDevice, ptr, bytes);
general/occa.cpp:#endif // MFEM_USE_CUDA && OCCA_CUDA_ENABLED
general/device.hpp:    memory space (e.g. GPUs) or share the memory space of the host (OpenMP). */
general/device.hpp:      /// [device] CUDA backend. Enabled when MFEM_USE_CUDA = YES.
general/device.hpp:      CUDA = 1 << 2,
general/device.hpp:      /** @brief [device] RAJA CUDA backend. Enabled when MFEM_USE_RAJA = YES
general/device.hpp:          and MFEM_USE_CUDA = YES. */
general/device.hpp:      RAJA_CUDA = 1 << 6,
general/device.hpp:      /** @brief [device] OCCA CUDA backend. Enabled when MFEM_USE_OCCA = YES
general/device.hpp:          and MFEM_USE_CUDA = YES. */
general/device.hpp:      OCCA_CUDA = 1 << 10,
general/device.hpp:      /** @brief [host] CEED CPU backend. GPU backends can still be used, but
general/device.hpp:      /** @brief [device] CEED CUDA backend working together with the CUDA
general/device.hpp:          backend. Enabled when MFEM_USE_CEED = YES and MFEM_USE_CUDA = YES.
general/device.hpp:          NOTE: The current default libCEED CUDA backend is non-deterministic! */
general/device.hpp:      CEED_CUDA = 1 << 12,
general/device.hpp:          transfers) without any GPU hardware. As 'DEBUG' is sometimes used
general/device.hpp:      /// Biwise-OR of all CUDA backends
general/device.hpp:      CUDA_MASK = CUDA | RAJA_CUDA | OCCA_CUDA | CEED_CUDA,
general/device.hpp:      CEED_MASK = CEED_CPU | CEED_CUDA | CEED_HIP,
general/device.hpp:      DEVICE_MASK = CUDA_MASK | HIP_MASK | DEBUG_DEVICE,
general/device.hpp:      RAJA_MASK = RAJA_CPU | RAJA_OMP | RAJA_CUDA | RAJA_HIP,
general/device.hpp:      OCCA_MASK = OCCA_CPU | OCCA_OMP | OCCA_CUDA
general/device.hpp:/** @brief The MFEM Device class abstracts hardware devices such as GPUs, as
general/device.hpp:    well as programming models such as CUDA, OCCA, RAJA and OpenMP. */
general/device.hpp:   int ngpu = -1; ///< Number of detected devices; -1: not initialized.
general/device.hpp:   bool mpi_gpu_aware = false;
general/device.hpp:       actual devices (e.g. GPU) to use.
general/device.hpp:         'ceed-cuda', 'occa-cuda', 'raja-cuda', 'cuda',
general/device.hpp:       * The backend 'occa-cuda' enables the 'cuda' backend unless 'raja-cuda'
general/device.hpp:       * The backend 'ceed-cuda' delegates to a libCEED CUDA backend the setup
general/device.hpp:         and evaluation of operators and enables the 'cuda' backend to avoid
general/device.hpp:   static inline bool IsConfigured() { return Get().ngpu >= 0; }
general/device.hpp:   /// Return true if an actual device (e.g. GPU) has been configured.
general/device.hpp:   static inline bool IsAvailable() { return Get().ngpu > 0; }
general/device.hpp:   static void SetGPUAwareMPI(const bool force = true)
general/device.hpp:   { Get().mpi_gpu_aware = force; }
general/device.hpp:   static bool GetGPUAwareMPI() { return Get().mpi_gpu_aware; }
general/cuda.cpp:// Internal debug option, useful for tracking CUDA allocations, deallocations
general/cuda.cpp:// #define MFEM_TRACK_CUDA_MEM
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:void mfem_cuda_error(cudaError_t err, const char *expr, const char *func,
general/cuda.cpp:   mfem::err << "\n\nCUDA error: (" << expr << ") failed with error:\n --> "
general/cuda.cpp:             << cudaGetErrorString(err)
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:   MFEM_GPU_CHECK(cudaMalloc(dptr, bytes));
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:   MFEM_GPU_CHECK(cudaMallocManaged(dptr, bytes));
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:   MFEM_GPU_CHECK(cudaMallocHost(ptr, bytes));
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:   MFEM_GPU_CHECK(cudaFree(dptr));
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:   MFEM_GPU_CHECK(cudaFreeHost(ptr));
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:   MFEM_GPU_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:   MFEM_GPU_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice));
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:   MFEM_GPU_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:   MFEM_GPU_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice));
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:   MFEM_GPU_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
general/cuda.cpp:#ifdef MFEM_TRACK_CUDA_MEM
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:   MFEM_GPU_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost));
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:   MFEM_GPU_CHECK(cudaGetLastError());
general/cuda.cpp:   int num_gpus = -1;
general/cuda.cpp:#ifdef MFEM_USE_CUDA
general/cuda.cpp:   MFEM_GPU_CHECK(cudaGetDeviceCount(&num_gpus));
general/cuda.cpp:   return num_gpus;
general/device.cpp:   Backend::CEED_CUDA, Backend::OCCA_CUDA, Backend::RAJA_CUDA, Backend::CUDA,
general/device.cpp:   "ceed-cuda", "occa-cuda", "raja-cuda", "cuda",
general/device.cpp:#ifdef MFEM_USE_CUDA
general/device.cpp:               || mem_backend == "cuda"
general/device.cpp:   Get().ngpu = -1;
general/device.cpp:   // OCCA_CUDA and CEED_CUDA need CUDA or RAJA_CUDA:
general/device.cpp:   if (Allows(Backend::OCCA_CUDA|Backend::CEED_CUDA) &&
general/device.cpp:       !Allows(Backend::RAJA_CUDA))
general/device.cpp:      Get().MarkBackend(Backend::CUDA);
general/device.cpp:#ifdef MFEM_USE_CUDA
general/device.cpp:static void DeviceSetup(const int dev, int &ngpu)
general/device.cpp:   ngpu = CuGetDeviceCount();
general/device.cpp:   MFEM_VERIFY(ngpu > 0, "No CUDA device found!");
general/device.cpp:   MFEM_GPU_CHECK(cudaSetDevice(dev));
general/device.cpp:static void CudaDeviceSetup(const int dev, int &ngpu)
general/device.cpp:#ifdef MFEM_USE_CUDA
general/device.cpp:   DeviceSetup(dev, ngpu);
general/device.cpp:   MFEM_CONTRACT_VAR(ngpu);
general/device.cpp:static void HipDeviceSetup(const int dev, int &ngpu)
general/device.cpp:   MFEM_GPU_CHECK(hipGetDeviceCount(&ngpu));
general/device.cpp:   MFEM_VERIFY(ngpu > 0, "No HIP device found!");
general/device.cpp:   MFEM_GPU_CHECK(hipSetDevice(dev));
general/device.cpp:   MFEM_CONTRACT_VAR(ngpu);
general/device.cpp:static void RajaDeviceSetup(const int dev, int &ngpu)
general/device.cpp:#ifdef MFEM_USE_CUDA
general/device.cpp:   if (ngpu <= 0) { DeviceSetup(dev, ngpu); }
general/device.cpp:   HipDeviceSetup(dev, ngpu);
general/device.cpp:   MFEM_CONTRACT_VAR(ngpu);
general/device.cpp:   const int cuda = Device::Allows(Backend::OCCA_CUDA);
general/device.cpp:   if (cpu + omp + cuda > 1)
general/device.cpp:   if (cuda)
general/device.cpp:#if OCCA_CUDA_ENABLED
general/device.cpp:      std::string mode("mode: 'CUDA', device_id : ");
general/device.cpp:      MFEM_ABORT("the OCCA CUDA backend requires OCCA built with CUDA!");
general/device.cpp:       strcmp(ceed_spec, "/gpu/hip"))
general/device.cpp:   MFEM_VERIFY(ngpu == -1, "the mfem::Device is already configured!");
general/device.cpp:   ngpu = 0;
general/device.cpp:#ifndef MFEM_USE_CUDA
general/device.cpp:   MFEM_VERIFY(!Allows(Backend::CUDA_MASK),
general/device.cpp:               "the CUDA backends require MFEM built with MFEM_USE_CUDA=YES");
general/device.cpp:   int ceed_cuda = Allows(Backend::CEED_CUDA);
general/device.cpp:   MFEM_VERIFY(ceed_cpu + ceed_cuda + ceed_hip <= 1,
general/device.cpp:   if (Allows(Backend::CUDA)) { CudaDeviceSetup(dev, ngpu); }
general/device.cpp:   if (Allows(Backend::HIP)) { HipDeviceSetup(dev, ngpu); }
general/device.cpp:   if (Allows(Backend::RAJA_CUDA) || Allows(Backend::RAJA_HIP))
general/device.cpp:   { RajaDeviceSetup(dev, ngpu); }
general/device.cpp:   if (Allows(Backend::CEED_CUDA))
general/device.cpp:         // NOTE: libCEED's /gpu/cuda/gen backend is non-deterministic!
general/device.cpp:         CeedDeviceSetup("/gpu/cuda/gen");
general/device.cpp:         CeedDeviceSetup("/gpu/hip");
general/device.cpp:   if (Allows(Backend::DEBUG_DEVICE)) { ngpu = 1; }
general/backends.hpp:#ifdef MFEM_USE_CUDA
general/backends.hpp:#include <cuda_runtime.h>
general/backends.hpp:#include <cuda.h>
general/backends.hpp:#include "cuda.hpp"
general/backends.hpp:#if defined(RAJA_ENABLE_CUDA) && !defined(MFEM_USE_CUDA)
general/backends.hpp:#error When RAJA is built with CUDA, MFEM_USE_CUDA=YES is required
general/backends.hpp:#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
general/backends.hpp:// MFEM_STREAM_SYNC is used for UVM and MPI GPU-Aware kernels
general/backends.hpp:#if !((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
general/backends.hpp:// 'double' and 'float' atomicAdd implementation for previous versions of CUDA
general/backends.hpp:#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
general/backends.hpp:#if ((defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)) || \
general/mem_manager.cpp:// Make sure Umpire is build with CUDA support if MFEM is built with it.
general/mem_manager.cpp:#if defined(MFEM_USE_CUDA) && !defined(UMPIRE_ENABLE_CUDA)
general/mem_manager.cpp:#error "CUDA is not enabled in Umpire!"
general/mem_manager.cpp:#ifdef MFEM_USE_CUDA
general/mem_manager.cpp:#ifdef MFEM_USE_CUDA
general/mem_manager.cpp:/// The CUDA device memory space
general/mem_manager.cpp:class CudaDeviceMemorySpace: public DeviceMemorySpace
general/mem_manager.cpp:   CudaDeviceMemorySpace(): DeviceMemorySpace() { }
general/mem_manager.cpp:/// The CUDA/HIP page-locked host memory space
general/mem_manager.cpp:#ifdef MFEM_USE_CUDA
general/mem_manager.cpp:#ifdef MFEM_USE_CUDA
general/mem_manager.cpp:   // Unlike cudaMemcpy(DtoD), hipMemcpy(DtoD) causes a host-side synchronization so
general/mem_manager.cpp:class UvmCudaMemorySpace : public DeviceMemorySpace
general/mem_manager.cpp:#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
general/mem_manager.cpp:#ifdef MFEM_USE_CUDA
general/mem_manager.cpp:#ifdef MFEM_USE_CUDA
general/mem_manager.cpp:      // Unlike cudaMemcpy(DtoD), hipMemcpy(DtoD) causes a host-side synchronization so
general/mem_manager.cpp:#ifdef MFEM_USE_CUDA
general/mem_manager.cpp:#endif // MFEM_USE_CUDA || MFEM_USE_HIP
general/mem_manager.cpp:#if defined(MFEM_USE_CUDA)
general/mem_manager.cpp:      device[static_cast<int>(MT::MANAGED)-shift] = new UvmCudaMemorySpace();
general/mem_manager.cpp:      device[static_cast<int>(MT::MANAGED)-shift] = new UvmCudaMemorySpace();
general/mem_manager.cpp:#if defined(MFEM_USE_CUDA)
general/mem_manager.cpp:            return new CudaDeviceMemorySpace();
general/mem_manager.cpp:#if defined(MFEM_USE_CUDA)
general/mem_manager.cpp:#if defined(MFEM_USE_CUDA)
general/mem_manager.cpp:   "cuda-uvm",
general/mem_manager.cpp:   "cuda",
general/mem_manager.cpp:#if defined(MFEM_USE_CUDA)
general/mem_manager.cpp:   "cuda-umpire",
general/mem_manager.cpp:   "cuda-umpire-2",
general/error.hpp:#if defined(__CUDA_ARCH__)
general/CMakeLists.txt:  cuda.cpp
general/CMakeLists.txt:  cuda.hpp
general/cuda.hpp:#ifndef MFEM_CUDA_HPP
general/cuda.hpp:#define MFEM_CUDA_HPP
general/cuda.hpp:// CUDA block size used by MFEM.
general/cuda.hpp:#define MFEM_CUDA_BLOCKS 256
general/cuda.hpp:#ifdef MFEM_USE_CUDA
general/cuda.hpp:#define MFEM_USE_CUDA_OR_HIP
general/cuda.hpp:#define MFEM_DEVICE_SYNC MFEM_GPU_CHECK(cudaDeviceSynchronize())
general/cuda.hpp:#define MFEM_STREAM_SYNC MFEM_GPU_CHECK(cudaStreamSynchronize(0))
general/cuda.hpp:// Define a CUDA error check macro, MFEM_GPU_CHECK(x), where x returns/is of
general/cuda.hpp:// type 'cudaError_t'. This macro evaluates 'x' and raises an error if the
general/cuda.hpp:// result is not cudaSuccess.
general/cuda.hpp:#define MFEM_GPU_CHECK(x) \
general/cuda.hpp:      cudaError_t err = (x); \
general/cuda.hpp:      if (err != cudaSuccess) \
general/cuda.hpp:         mfem_cuda_error(err, #x, _MFEM_FUNC_NAME, __FILE__, __LINE__); \
general/cuda.hpp:#endif // MFEM_USE_CUDA
general/cuda.hpp:#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
general/cuda.hpp:#ifdef MFEM_USE_CUDA
general/cuda.hpp:// Function used by the macro MFEM_GPU_CHECK.
general/cuda.hpp:void mfem_cuda_error(cudaError_t err, const char *expr, const char *func,
general/cuda.hpp:/// Check the error code returned by cudaGetLastError(), aborting on error.
general/cuda.hpp:/// Get the number of CUDA devices
general/cuda.hpp:#endif // MFEM_CUDA_HPP
general/hip.cpp:   MFEM_GPU_CHECK(hipMalloc(dptr, bytes));
general/hip.cpp:   MFEM_GPU_CHECK(hipMallocManaged(dptr, bytes));
general/hip.cpp:   MFEM_GPU_CHECK(hipHostMalloc(ptr, bytes, hipHostMallocDefault));
general/hip.cpp:   MFEM_GPU_CHECK(hipFree(dptr));
general/hip.cpp:   MFEM_GPU_CHECK(hipHostFree(ptr));
general/hip.cpp:   MFEM_GPU_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
general/hip.cpp:   MFEM_GPU_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice));
general/hip.cpp:   MFEM_GPU_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
general/hip.cpp:   MFEM_GPU_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice));
general/hip.cpp:   MFEM_GPU_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
general/hip.cpp:   MFEM_GPU_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost));
general/hip.cpp:   MFEM_GPU_CHECK(hipGetLastError());
general/hip.cpp:   int num_gpus = -1;
general/hip.cpp:   MFEM_GPU_CHECK(hipGetDeviceCount(&num_gpus));
general/hip.cpp:   return num_gpus;
general/occa.hpp:   return Device::Allows(Backend::OCCA_CUDA) ||
general/hip.hpp:#define MFEM_USE_CUDA_OR_HIP
general/hip.hpp:#define MFEM_DEVICE_SYNC MFEM_GPU_CHECK(hipDeviceSynchronize())
general/hip.hpp:#define MFEM_STREAM_SYNC MFEM_GPU_CHECK(hipStreamSynchronize(0))
general/hip.hpp:// Define a HIP error check macro, MFEM_GPU_CHECK(x), where x returns/is of
general/hip.hpp:#define MFEM_GPU_CHECK(x) \
general/hip.hpp:// Function used by the macro MFEM_GPU_CHECK.
CONTRIBUTING.md:#### GPU and general device support
CONTRIBUTING.md:GPU and multi-core CPU support is based on device kernels supporting different
CONTRIBUTING.md:backends (CUDA, OCCA, RAJA, OpenMP, etc.) and an internal lightweight
CONTRIBUTING.md:  + the [`cuda.hpp`](https://docs.mfem.org/html/cuda_8hpp.html) and [`occa.hpp`](https://docs.mfem.org/html/occa_8hpp.html) files

```
