# https://github.com/AMReX-Combustion/PeleLMeX

```console
.clang-tidy:  - key:             cppcoreguidelines-non-private-member-variables-in-classes.IgnoreClassesWithAllMemberVariablesBeingPublic
Tests/CMakeLists.txt:      if(PELE_ENABLE_CUDA)
Tests/CMakeLists.txt:        set(PELE_NP 2) # 1 rank per GPU on Eagle
Tests/CMakeLists.txt:      if(PELE_ENABLE_CUDA)
Tests/CMakeLists.txt:      if(PELE_ENABLE_CUDA)
Tests/CMakeLists.txt:        set(PELE_NP 2) # 1 rank per GPU on Eagle
CMake/SetSundialsOptions.cmake:set(ENABLE_CUDA ${PELE_ENABLE_CUDA})
CMake/SetSundialsOptions.cmake:if(ENABLE_CUDA)
CMake/SetSundialsOptions.cmake:  set(EXAMPLES_ENABLE_CUDA OFF)
CMake/SetPeleCompileFlags.cmake:if((NOT PELE_ENABLE_CUDA) AND (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$"))
CMake/SetAmrexHydroOptions.cmake:set(HYDRO_GPU_BACKEND ${AMReX_GPU_BACKEND} CACHE STRING "AMReX GPU type" FORCE)
CMake/BuildPeleExe.cmake:  if(PELE_ENABLE_CUDA)
CMake/BuildPeleExe.cmake:      set_source_files_properties(${PELE_SOURCES} PROPERTIES LANGUAGE CUDA)
CMake/BuildPeleExe.cmake:    set_target_properties(${pele_exe_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
CMake/BuildPeleExe.cmake:    target_compile_options(${pele_exe_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas --disable-optimizer-constants>)
CMake/BuildPelePhysicsLib.cmake:    if(PELE_ENABLE_CUDA)
CMake/BuildPelePhysicsLib.cmake:      target_link_libraries(${pele_physics_lib_name} PUBLIC sundials_nveccuda sundials_sunlinsolcusolversp sundials_sunmatrixcusparse)
CMake/SetAmrexOptions.cmake:if(PELE_ENABLE_CUDA)
CMake/SetAmrexOptions.cmake:  set(AMReX_GPU_BACKEND CUDA CACHE STRING "AMReX GPU type" FORCE)
CMake/SetAmrexOptions.cmake:  set(AMReX_GPU_BACKEND HIP CACHE STRING "AMReX GPU type" FORCE)
CMake/SetAmrexOptions.cmake:  set(AMReX_GPU_BACKEND SYCL CACHE STRING "AMReX GPU type" FORCE)
CMake/SetAmrexOptions.cmake:  set(AMReX_GPU_BACKEND NONE CACHE STRING "AMReX GPU type" FORCE)
Docs/sphinx/manual/Tutorials_TripleFlame.rst:   USE_CUDA        = FALSE
Docs/sphinx/manual/Tutorials_HotBubble.rst:   USE_CUDA        = FALSE
Docs/sphinx/manual/Tutorials_FlowPastCyl.rst:   USE_CUDA        = FALSE
Docs/sphinx/manual/Implementation.rst:compute-intensive kernels implemented as lambda functions to seamlessly run on CPU and various GPU backends through AMReX
Docs/sphinx/manual/Implementation.rst:    for (MFIter mfi(State,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Docs/sphinx/manual/Implementation.rst:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Docs/sphinx/manual/Implementation.rst:and CUDA, HIP or SYCL for heterogeneous architectures.
Docs/sphinx/manual/Implementation.rst:The reader is referred to `AMReX GPU documentation <https://amrex-codes.github.io/amrex/docs_html/GPU.html>`_ for more details on
Docs/sphinx/manual/LMeXControls.rst:Note that the last five parameters belong to the Reactor class of PelePhysics but are specified here for completeness. In particular, CVODE is the adequate choice of integrator to tackle PeleLMeX large time step sizes. Several linear solvers are available depending on whether or not GPU are employed: on CPU, `dense_direct` is a finite-difference direct solver, `denseAJ_direct` is an analytical-jacobian direct solver (preferred choice), `sparse_direct` is an analytical-jacobian sparse direct solver based on the KLU library and `GMRES` is a matrix-free iterative solver; on GPU `GMRES` is a matrix-free iterative solver (available on all the platforms), `sparse_direct` is a batched block-sparse direct solve based on NVIDIA's cuSparse (only with CUDA), `magma_direct` is a batched block-dense direct solve based on the MAGMA library (available with CUDA and HIP. Different `cvode.solve_type` should be tried before increasing the `cvode.max_substeps`.
Docs/sphinx/manual/Tutorials_BFSFlame.rst:   USE_CUDA        = FALSE
Docs/sphinx/manual/Tutorials_FlameSheet.rst:   USE_CUDA        = FALSE
Docs/sphinx/manual/Performances.rst:CUDA, HIP or SYCL, for Nvidia, AMD and Intel GPUs vendor, respectively. The actual performances
Docs/sphinx/manual/Performances.rst:Additionally, unless otherwise specified, all the tests on GPUs are conducted
Docs/sphinx/manual/Performances.rst:Perlmutter's `GPU nodes <https://docs.nersc.gov/systems/perlmutter/architecture/#gpu-nodes>`_ consists of a single AMD EPYC 7763 (Milan)
Docs/sphinx/manual/Performances.rst:CPU connected to 4 NVIDIA A100 GPUs. The `CPU nodes <https://docs.nersc.gov/systems/perlmutter/architecture/#cpu-nodes>`_ consists of
Docs/sphinx/manual/Performances.rst:two of the same AMD EPYC, 64-cores CPUs. When running on the GPU node, `PeleLMeX` will use 4 MPI ranks with each access to one A100, while
Docs/sphinx/manual/Performances.rst:leading to an initial cell count of 3.276 M, i.e. 0.8M/cells per GPU. The git hashes of `PeleLMeX` and its dependencies for
Docs/sphinx/manual/Performances.rst:of the stiff chemistry integration, especially on the GPU.
Docs/sphinx/manual/Performances.rst:each containing 2 Graphics Compute Dies (GCDs) for a total of 8 GCDs per node. When running with GPU acceleration, `PeleLMeX` will use 8 MPI ranks with each access to one GCD, while when running on flat MPI, we will use 64 MPI-ranks.
Docs/sphinx/manual/Performances.rst:leading to an initial cell count of 6.545 M, i.e. 0.8M/cells per GPU. The git hashes of `PeleLMeX` and its dependencies for
Docs/sphinx/manual/Performances.rst:of the stiff chemistry integration, especially on the GPU.
Docs/sphinx/manual/Performances.rst:Summit was launched in 2018 as the first DOE's fully GPU-accelerated platform.
Docs/sphinx/manual/Performances.rst:of a two IBM Power9 CPU connected to 6 NVIDIA V100 GPUs. When running with GPU acceleration, `PeleLMeX` will
Docs/sphinx/manual/Performances.rst:Note that in contrast with newer GPUs available on Perlmutter or Crusher, Summit's V100s only have 16GBs of
Docs/sphinx/manual/Performances.rst:memory which limit the number of cells/GPU. For this reason, the chemical linear solver used within Sundials is
Docs/sphinx/manual/Performances.rst:leading to an initial cell count of 0.819 M, i.e. 0.136M/cells per GPU. The git hashes of `PeleLMeX` and its dependencies for
Docs/sphinx/manual/Performances.rst:Crusher nodes (8 to 1024 GPUs) and a closer look at the scaling data shows that most of the
Docs/sphinx/manual/Model.rst:   * Parallelization using MPI+X approach, with X one of OpenMP, CUDA, HIP or SYCL
Build/cmake.sh:      -DPELE_ENABLE_CUDA:BOOL=OFF \
Build/cmake.sh:      -DAMReX_CUDA_ARCH=Volta \
Exec/Efield/PremBunsen3DKuhl/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Efield/PremBunsen3DKuhl/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/Efield/PremBunsen3DKuhl/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Efield/PremBunsen3DKuhl/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Efield/PremBunsen3DKuhl/GNUmakefile:USE_CUDA = FALSE
Exec/Efield/FlameSheetIons/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Efield/FlameSheetIons/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/Efield/FlameSheetIons/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Efield/FlameSheetIons/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/Efield/FlameSheetIons/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Efield/FlameSheetIons/GNUmakefile:USE_CUDA = FALSE
Exec/Efield/IonizedAirWave/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Efield/IonizedAirWave/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/Efield/IonizedAirWave/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Efield/IonizedAirWave/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/Efield/IonizedAirWave/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Efield/IonizedAirWave/GNUmakefile:USE_CUDA = FALSE
Exec/Production/ChallengeProblem/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/Production/ChallengeProblem/pelelmex_prob_parm.H:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> win_lo = {0.0, 0.0, 0.0};
Exec/Production/ChallengeProblem/pelelmex_prob_parm.H:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> win_hi = {0.0, 0.0, 0.0};
Exec/Production/ChallengeProblem/getLMeXChallengePB.sh:   module load cmake gcc cuda python
Exec/Production/ChallengeProblem/getLMeXChallengePB.sh:   module load PrgEnv-amd cmake rocm/4.5.0
Exec/Production/ChallengeProblem/getLMeXChallengePB.sh:   module load PrgEnv-amd cmake rocm/5.1.0 craype-accel-amd-gfx90a cray-libsci/21.08.1.2
Exec/Production/ChallengeProblem/getLMeXChallengePB.sh:   module load PrgEnv-cray cmake rocm/5.2.0 craype-x86-trento craype-accel-amd-gfx90a cray-libsci/21.08.1.2
Exec/Production/ChallengeProblem/getLMeXChallengePB.sh:   module unload craype-x86-naples rocm
Exec/Production/ChallengeProblem/getLMeXChallengePB.sh:   module load craype-x86-rome rocm cmake
Exec/Production/ChallengeProblem/getLMeXChallengePB.sh: AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Exec/Production/ChallengeProblem/getLMeXChallengePB.sh:   make -j 12 COMP=gcc USE_HIP=FALSE USE_CUDA=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=summit TPLrealclean
Exec/Production/ChallengeProblem/getLMeXChallengePB.sh:   make -j 12 COMP=gcc USE_HIP=FALSE USE_CUDA=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=summit TPL
Exec/Production/ChallengeProblem/getLMeXChallengePB.sh:   make -j 12 COMP=gnu USE_HIP=FALSE USE_CUDA=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=summit realclean
Exec/Production/ChallengeProblem/getLMeXChallengePB.sh:   make -j 12 COMP=gnu USE_HIP=FALSE USE_CUDA=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=summit
Exec/Production/ChallengeProblem/input.3d_Hypre:amrex.abort_on_out_of_gpu_memory = 1
Exec/Production/ChallengeProblem/README.md:module load PrgEnv-amd rocm/5.1.0 craype-accel-amd-gfx90a cray-libsci/21.08.1.2
Exec/Production/ChallengeProblem/README.md:module load cmake gcc cuda netlib-lapack
Exec/Production/ChallengeProblem/getLMeXChallengePBHypre.sh:   module load cmake gcc cuda python
Exec/Production/ChallengeProblem/getLMeXChallengePBHypre.sh:   module load PrgEnv-amd cmake rocm/4.5.0
Exec/Production/ChallengeProblem/getLMeXChallengePBHypre.sh:   module load PrgEnv-amd cmake rocm/5.1.0 craype-accel-amd-gfx90a cray-libsci/21.08.1.2
Exec/Production/ChallengeProblem/getLMeXChallengePBHypre.sh:   module load PrgEnv-cray cmake rocm/5.2.0 craype-x86-trento craype-accel-amd-gfx90a cray-libsci/21.08.1.2
Exec/Production/ChallengeProblem/getLMeXChallengePBHypre.sh:   module unload craype-x86-naples rocm
Exec/Production/ChallengeProblem/getLMeXChallengePBHypre.sh:   module load craype-x86-rome rocm cmake
Exec/Production/ChallengeProblem/getLMeXChallengePBHypre.sh:   ./configure --prefix=${MYPWD}/hypre/install/ --without-superlu --disable-bigint --without-openmp --with-cuda --enable-unified-memory --enable-curand --enable-cusolver --enable-cusparse --disable-cublas --enable-gpu-profiling --enable-shared
Exec/Production/ChallengeProblem/getLMeXChallengePBHypre.sh:   ./configure --prefix=${MYPWD}/hypre/install/ --with-gpu-arch=gfx90a --without-superlu --disable-bigint --without-openmp --with-hip --enable-rocsparse --enable-rocrand --enable-shared --with-MPI-lib-dirs=/opt/cray/pe/mpich/8.1.16/ofi/crayclang/10.0/lib /opt/cray/pe/mpich/8.1.16/gtl/lib --with-MPI-libs=mpi mpi_gtl_hsa --with-MPI-include=/opt/cray/pe/mpich/8.1.16/ofi/crayclang/10.0/include
Exec/Production/ChallengeProblem/getLMeXChallengePBHypre.sh:   ./configure --prefix=${MYPWD}/hypre/install/ --with-gpu-arch=gfx90a --without-superlu --disable-bigint --without-openmp --with-hip --enable-rocsparse --enable-rocrand --enable-shared --with-MPI-lib-dirs=/opt/cray/pe/mpich/8.1.16/ofi/crayclang/10.0/lib /opt/cray/pe/mpich/8.1.16/gtl/lib --with-MPI-libs=mpi mpi_gtl_hsa --with-MPI-include=/opt/cray/pe/mpich/8.1.16/ofi/crayclang/10.0/include
Exec/Production/ChallengeProblem/getLMeXChallengePBHypre.sh:   make -j 12 COMP=gcc USE_HIP=FALSE USE_CUDA=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=summit TPLrealclean
Exec/Production/ChallengeProblem/getLMeXChallengePBHypre.sh:   make -j 12 COMP=gcc USE_HIP=FALSE USE_CUDA=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=summit TPL
Exec/Production/ChallengeProblem/getLMeXChallengePBHypre.sh:   make -j 12 COMP=gnu USE_HIP=FALSE USE_CUDA=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=summit realclean
Exec/Production/ChallengeProblem/getLMeXChallengePBHypre.sh:   make -j 12 COMP=gnu USE_HIP=FALSE USE_CUDA=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=summit
Exec/Production/ChallengeProblem/pelelmex_prob.cpp:    amrex::Gpu::copy(
Exec/Production/ChallengeProblem/pelelmex_prob.cpp:      amrex::Gpu::hostToDevice, xarray.begin(), xarray.end(),
Exec/Production/ChallengeProblem/pelelmex_prob.cpp:    amrex::Gpu::copy(
Exec/Production/ChallengeProblem/pelelmex_prob.cpp:      amrex::Gpu::hostToDevice, xdiff.begin(), xdiff.end(),
Exec/Production/ChallengeProblem/pelelmex_prob.cpp:    amrex::Gpu::copy(
Exec/Production/ChallengeProblem/pelelmex_prob.cpp:      amrex::Gpu::hostToDevice, uinput.begin(), uinput.end(),
Exec/Production/ChallengeProblem/pelelmex_prob.cpp:    amrex::Gpu::copy(
Exec/Production/ChallengeProblem/pelelmex_prob.cpp:      amrex::Gpu::hostToDevice, vinput.begin(), vinput.end(),
Exec/Production/ChallengeProblem/pelelmex_prob.cpp:    amrex::Gpu::copy(
Exec/Production/ChallengeProblem/pelelmex_prob.cpp:      amrex::Gpu::hostToDevice, winput.begin(), winput.end(),
Exec/Production/ChallengeProblem/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/ChallengeProblem/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/ChallengeProblem/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/ChallengeProblem/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/ChallengeProblem/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/ChallengeProblem/PeleLMeX_EBUserDefined.H:AMREX_GPU_DEVICE
Exec/Production/ChallengeProblem/PeleLMeX_EBUserDefined.H:  // Aborting here (message will not show on GPUs)
Exec/Production/ChallengeProblem/PeleLMeX_EBUserDefined.H:AMREX_GPU_DEVICE
Exec/Production/ChallengeProblem/input.3d:amrex.abort_on_out_of_gpu_memory = 1
Exec/Production/ChallengeProblem/GNUmakefile:USE_CUDA = FALSE
Exec/Production/TripleFlame/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/Production/TripleFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/TripleFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/TripleFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/TripleFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/TripleFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/TripleFlame/GNUmakefile:USE_CUDA = FALSE
Exec/Production/SwirlFlowWallInteractions/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/Production/SwirlFlowWallInteractions/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/SwirlFlowWallInteractions/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/SwirlFlowWallInteractions/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/SwirlFlowWallInteractions/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/Production/SwirlFlowWallInteractions/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/SwirlFlowWallInteractions/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/SwirlFlowWallInteractions/GNUmakefile:USE_CUDA = FALSE
Exec/Production/NormalJet_OpenDomain/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/Production/NormalJet_OpenDomain/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/NormalJet_OpenDomain/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/NormalJet_OpenDomain/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/NormalJet_OpenDomain/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/NormalJet_OpenDomain/GNUmakefile:USE_CUDA = FALSE
Exec/Production/CounterFlowSpray/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/CounterFlowSpray/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/CounterFlowSpray/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/CounterFlowSpray/GNUmakefile:USE_CUDA = FALSE
Exec/Production/DiffBunsen2D/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/DiffBunsen2D/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/DiffBunsen2D/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/Production/DiffBunsen2D/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/DiffBunsen2D/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/Production/DiffBunsen2D/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/DiffBunsen2D/GNUmakefile:USE_CUDA = FALSE
Exec/Production/PremBunsen3D/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/PremBunsen3D/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/Production/PremBunsen3D/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/PremBunsen3D/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/PremBunsen3D/GNUmakefile:USE_CUDA = FALSE
Exec/Production/JetInCrossflow/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/Production/JetInCrossflow/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/JetInCrossflow/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/JetInCrossflow/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/Production/JetInCrossflow/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/JetInCrossflow/GNUmakefile:USE_CUDA = FALSE
Exec/Production/PremBunsen2D/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/PremBunsen2D/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/Production/PremBunsen2D/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/PremBunsen2D/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/PremBunsen2D/GNUmakefile:USE_CUDA = FALSE
Exec/Production/CounterFlow/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/CounterFlow/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/CounterFlow/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/Production/CounterFlow/GNUmakefile:USE_CUDA = FALSE
Exec/Make.PeleLMeX:ifeq ($(USE_CUDA),TRUE)
Exec/Make.PeleLMeX:	@cd $(PELE_PHYSICS_HOME)/ThirdParty && $(MAKE) $(MAKEFLAGS) sundials SUNDIALS_HOME=$(SUNDIALS_HOME) AMREX_HOME=$(AMREX_HOME) USE_CUDA=$(USE_CUDA) USE_HIP=$(USE_HIP) USE_SYCL=$(USE_SYCL) PELE_USE_MAGMA=$(PELE_USE_MAGMA) PELE_USE_KLU=$(PELE_USE_KLU) DEBUG=$(DEBUG) COMP=$(HOSTCC) NVCC=$(COMP)
Exec/Make.PeleLMeX:	cd $(PELE_PHYSICS_HOME)/ThirdParty; $(MAKE) $(MAKEFLAGS) SUNDIALS_HOME=$(SUNDIALS_HOME) AMREX_HOME=$(AMREX_HOME) USE_CUDA=$(USE_CUDA) USE_HIP=$(USE_HIP) USE_SYCL=$(USE_SYCL) PELE_USE_KLU=$(PELE_USE_KLU) PELE_USE_MAGMA=$(PELE_USE_MAGMA) DEBUG=$(DEBUG) COMP=$(HOSTCC) NVCC=$(COMP) clean
Exec/Make.PeleLMeX:	cd $(PELE_PHYSICS_HOME)/ThirdParty; $(MAKE) $(MAKEFLAGS) SUNDIALS_HOME=$(SUNDIALS_HOME) AMREX_HOME=$(AMREX_HOME) USE_CUDA=$(USE_CUDA) USE_HIP=$(USE_HIP) USE_SYCL=$(USE_SYCL) PELE_USE_KLU=$(PELE_USE_KLU) PELE_USE_MAGMA=$(PELE_USE_MAGMA) DEBUG=$(DEBUG) COMP=$(HOSTCC) NVCC=$(COMP) realclean
Exec/RegTests/EB_ODEQty/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/RegTests/EB_ODEQty/pelelmex_prob_parm.H:struct ProbParm : amrex::Gpu::Managed
Exec/RegTests/EB_ODEQty/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_ODEQty/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_ODEQty/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_ODEQty/PeleLMeX_ProblemSpecificFunctions.cpp:    *ext_src, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Exec/RegTests/EB_ODEQty/PeleLMeX_ProblemSpecificFunctions.cpp:  Gpu::streamSynchronize();
Exec/RegTests/EB_ODEQty/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/HotBubble/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/HotBubble/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/HotBubble/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/HotBubble/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/Unit/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/Unit/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/Unit/pelelmex_prob.H:  // amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/RegTests/Unit/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/TripleFlame/pelelmex_prob.H:AMREX_GPU_HOST_DEVICE
Exec/RegTests/TripleFlame/pelelmex_prob.H:AMREX_GPU_HOST_DEVICE
Exec/RegTests/TripleFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/TripleFlame/pelelmex_prob.H:AMREX_GPU_HOST_DEVICE
Exec/RegTests/TripleFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/TripleFlame/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/EnclosedFlame/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/RegTests/EnclosedFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EnclosedFlame/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/RegTests/EnclosedFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EnclosedFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EnclosedFlame/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/SprayTest/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/RegTests/SprayTest/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/SprayTest/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/SprayTest/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES> massfrac = {{0.0}};
Exec/RegTests/SprayTest/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/SprayTest/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/EB_BackwardStepFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_BackwardStepFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_BackwardStepFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_BackwardStepFlame/PeleLMeX_EBUserDefined.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_BackwardStepFlame/PeleLMeX_EBUserDefined.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_BackwardStepFlame/PeleLMeX_PatchFlowVariables.cpp:  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
Exec/RegTests/EB_BackwardStepFlame/PeleLMeX_PatchFlowVariables.cpp:  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
Exec/RegTests/EB_BackwardStepFlame/PeleLMeX_PatchFlowVariables.cpp:  for (amrex::MFIter mfi(a_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/RegTests/EB_BackwardStepFlame/PeleLMeX_PatchFlowVariables.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/RegTests/EB_BackwardStepFlame/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/EB_EnclosedFlame/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/RegTests/EB_EnclosedFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_EnclosedFlame/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/RegTests/EB_EnclosedFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_EnclosedFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_EnclosedFlame/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/PeriodicCases/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/RegTests/PeriodicCases/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/PeriodicCases/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/PeriodicCases/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/PeriodicCases/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/FlameSheet/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/RegTests/FlameSheet/inputs.3d_Dodecane:amrex.abort_on_out_of_gpu_memory = 1
Exec/RegTests/FlameSheet/inputs.3d_DodecaneQSS:amrex.abort_on_out_of_gpu_memory = 1
Exec/RegTests/FlameSheet/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/FlameSheet/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/RegTests/FlameSheet/pelelmex_prob.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM - 1> lpert{Lx};
Exec/RegTests/FlameSheet/pelelmex_prob.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM - 1> lpert{Lx, Ly};
Exec/RegTests/FlameSheet/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/FlameSheet/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/RegTests/FlameSheet/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/FlameSheet/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/TurbInflow/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/RegTests/TurbInflow/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/TurbInflow/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/TurbInflow/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/TurbInflow/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/HITDecay/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/RegTests/HITDecay/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/HITDecay/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/HITDecay/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/HITDecay/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/HITDecay/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/SootRadTest/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/RegTests/SootRadTest/pelelmex_prob_parm.H:  amrex::GpuArray<amrex::Real, NUM_SOOT_MOMENTS + 1> soot_vals;
Exec/RegTests/SootRadTest/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/SootRadTest/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES> massfrac = {{0.0}};
Exec/RegTests/SootRadTest/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {{0.0}};
Exec/RegTests/SootRadTest/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/SootRadTest/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {{0.0}};
Exec/RegTests/SootRadTest/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/SootRadTest/GNUmakefile:USE_CUDA        = FALSE
Exec/RegTests/EB_PipeFlow/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/RegTests/EB_PipeFlow/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_PipeFlow/pelelmex_prob.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> u;
Exec/RegTests/EB_PipeFlow/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_PipeFlow/pelelmex_prob.H:  // amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/RegTests/EB_PipeFlow/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_PipeFlow/PeleLMeX_EBUserDefined.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_PipeFlow/PeleLMeX_EBUserDefined.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_PipeFlow/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/EB_FlowPastCylinder/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/RegTests/EB_FlowPastCylinder/pelelmex_prob_parm.H:struct ProbParm : amrex::Gpu::Managed
Exec/RegTests/EB_FlowPastCylinder/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_FlowPastCylinder/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_FlowPastCylinder/pelelmex_prob.H:  // amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/RegTests/EB_FlowPastCylinder/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_FlowPastCylinder/PeleLMeX_EBUserDefined.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_FlowPastCylinder/PeleLMeX_EBUserDefined.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_FlowPastCylinder/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/EB_EnclosedVortex/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/RegTests/EB_EnclosedVortex/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_EnclosedVortex/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_EnclosedVortex/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EB_EnclosedVortex/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/TaylorGreen/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/RegTests/TaylorGreen/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/TaylorGreen/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/TaylorGreen/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/TaylorGreen/GNUmakefile:USE_CUDA = FALSE
Exec/RegTests/EnclosedInjection/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EnclosedInjection/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EnclosedInjection/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/RegTests/EnclosedInjection/GNUmakefile:USE_CUDA = FALSE
Exec/UnitTests/DodecaneLu/inputs.3d:amrex.abort_on_out_of_gpu_memory = 1
Exec/UnitTests/DodecaneLu/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/UnitTests/DodecaneLu/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/UnitTests/DodecaneLu/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/UnitTests/DodecaneLu/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/UnitTests/DodecaneLu/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/UnitTests/DodecaneLu/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/UnitTests/DodecaneLu/GNUmakefile:USE_CUDA = FALSE
Exec/UnitTests/EB_SphericalFlame/pelelmex_prob_parm.H:#include <AMReX_GpuMemory.H>
Exec/UnitTests/EB_SphericalFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/UnitTests/EB_SphericalFlame/pelelmex_prob.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
Exec/UnitTests/EB_SphericalFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/UnitTests/EB_SphericalFlame/pelelmex_prob.H:AMREX_GPU_DEVICE
Exec/UnitTests/EB_SphericalFlame/GNUmakefile:USE_CUDA = FALSE
README.md:a flexible tool to address research questions on platforms ranging from small workstations to the world's largest GPU-accelerated supercomputers.
README.md:Finally, when building with GPU support, CUDA >= 11 is required with NVIDIA GPUs and ROCm >= 5.2 is required with AMD GPUs.
README.md:Finally, make with: `make -j`, or if on macOS: `make -j COMP=llvm`. To clean the installation, use either `make clean` or `make realclean`. If running into compile errors after changing compile time options in PeleLMeX (e.g., the chemical mechanism), the first thing to try is to clean your build by running `make TPLrealclean && make realclean`, then try to rebuild the third party libraries and PeleLMeX with `make TPL && make -j`. See the [Tutorial](https://amrex-combustion.github.io/PeleLMeX/manual/html/Tutorials_HotBubble.html) for this case for instructions on how to compile with different options (for example, to compile without MPI support or to compile for GPUs) and how to run the code once compiled.
paper/paper.bib:   title     = {{Enabling GPU accelerated computing in the SUNDIALS time integration library}},
paper/paper.md:logical tiles spread across threads using OpenMP for multi-core CPU machines, or spread across GPU threads using CUDA/HIP/SYCL
paper/paper.md:on GPU-accelerated machines.
CMakeLists.txt:option(PELE_ENABLE_CUDA "Enable CUDA" OFF)
CMakeLists.txt:if(PELE_ENABLE_CUDA)
CMakeLists.txt:  enable_language(CUDA)
CMakeLists.txt:  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10.0")
CMakeLists.txt:    message(FATAL_ERROR "Your nvcc version is ${CMAKE_CUDA_COMPILER_VERSION} which is unsupported."
CMakeLists.txt:      "Please use CUDA toolkit version 10.0 or newer.")
CMakeLists.txt:  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
CMakeLists.txt:    set(CMAKE_CUDA_ARCHITECTURES 70)
Source/PeleLMeX_Setup.cpp:#ifdef AMREX_USE_GPU
Source/PeleLMeX_Setup.cpp:  Gpu::copy(Gpu::hostToDevice, prob_parm, prob_parm + 1, prob_parm_d);
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirState>> bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirState>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirState>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDens>> bndry_func_rho(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDens>> crse_bndry_func_rho(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDens>> fine_bndry_func_rho(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirSpec>> bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirSpec>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirSpec>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirTemp>> bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirTemp>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirTemp>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirPhiV>> bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirPhiV>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirPhiV>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:  PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirState>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:  PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirState>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:  PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:  PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:  PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:  PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:  PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:  PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:  PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> crse_bndry_func(
Source/PeleLMeX_BC.cpp:  PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> fine_bndry_func(
Source/PeleLMeX_BC.cpp:  PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirState>> bndry_func(
Source/PeleLMeX_BC.cpp:    amrex::Gpu::copy(
Source/PeleLMeX_BC.cpp:      amrex::Gpu::deviceToHost, probparmDD, probparmDD + 1, probparmDH);
Source/PeleLMeX_BC.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_BC.cpp:    for (MFIter mfi(a_vel, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_BC.cpp:    amrex::Gpu::copy(
Source/PeleLMeX_BC.cpp:      amrex::Gpu::hostToDevice, probparmDH, probparmDH + 1, probparmDD);
Source/PeleLMeX_BPatch.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_line_center;
Source/PeleLMeX_BPatch.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_circle_center;
Source/PeleLMeX_BPatch.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_rectangle_lo;
Source/PeleLMeX_BPatch.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_rectangle_hi;
Source/PeleLMeX_BPatch.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_circ_ann_center;
Source/PeleLMeX_BPatch.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_rect_ann_outer_lo;
Source/PeleLMeX_BPatch.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_rect_ann_outer_hi;
Source/PeleLMeX_BPatch.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_rect_ann_inner_lo;
Source/PeleLMeX_BPatch.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_rect_ann_inner_hi;
Source/PeleLMeX_BPatch.H:    AMREX_GPU_DEVICE
Source/PeleLMeX_BPatch.H:      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> point_coordinate,
Source/PeleLMeX_BPatch.H:        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> patch_rectangle_lo_touse =
Source/PeleLMeX_BPatch.H:        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> patch_rectangle_hi_touse =
Source/PeleLMeX_BPatch.H:        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
Source/PeleLMeX_BPatch.H:        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
Source/PeleLMeX_BPatch.H:        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
Source/PeleLMeX_BPatch.H:        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
Source/PeleLMeX_BPatch.H:      amrex::Gpu::copy(
Source/PeleLMeX_BPatch.H:        amrex::Gpu::hostToDevice, &m_bpdata_h, &m_bpdata_h + 1, m_bpdata_d);
Source/PeleLMeX_Plot.cpp:         cnt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_Plot.cpp:      Gpu::streamSynchronize();
Source/PeleLMeX_Plot.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Plot.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Plot.cpp:        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Plot.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Plot.cpp:        for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_Plot.cpp:            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Plot.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Plot.cpp:  for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Plot.cpp:      [=, eosparm = leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_BPatch.cpp:  amrex::Gpu::streamSynchronize();
Source/PeleLMeX_Reactions.cpp:  if (Gpu::notInLaunchRegion()) {
Source/PeleLMeX_Reactions.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Reactions.cpp:           extF_rhoH] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:           FrhoYe] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:#ifdef AMREX_USE_GPU
Source/PeleLMeX_Reactions.cpp:      amrex::Gpu::gpuStream()
Source/PeleLMeX_Reactions.cpp:           extF_rhoH] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:           extF_rhoY] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:#ifdef AMREX_USE_GPU
Source/PeleLMeX_Reactions.cpp:    Gpu::Device::streamSynchronize();
Source/PeleLMeX_Reactions.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Reactions.cpp:  for (MFIter mfi(ldataNew_p->state, amrex::TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_Reactions.cpp:       dt_inv] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Reactions.cpp:           dt_inv] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:  if (Gpu::notInLaunchRegion()) {
Source/PeleLMeX_Reactions.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Reactions.cpp:           extF_rhoH] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:           FrhoYe] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:#ifdef AMREX_USE_GPU
Source/PeleLMeX_Reactions.cpp:        amrex::Gpu::gpuStream()
Source/PeleLMeX_Reactions.cpp:      ParallelFor(bx, [fcl] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:      bx, [rhoY_o, rhoH_o] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:      [invmwt, nE_o, rhoYe_o] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:#ifdef AMREX_USE_GPU
Source/PeleLMeX_Reactions.cpp:    Gpu::Device::streamSynchronize();
Source/PeleLMeX_Reactions.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Reactions.cpp:  for (MFIter mfi(ldataNew_p->state, amrex::TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_Reactions.cpp:           dt_inv] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:           dt_inv] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Reactions.cpp:  for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Reactions.cpp:        [rhoYdot] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Reactions.cpp:             leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:             leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Reactions.cpp:    for (MFIter mfi(advData->Forcing[lev], TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_Reactions.cpp:             dtinv] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Reactions.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Reactions.cpp:    for (MFIter mfi(*a_HR, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Reactions.cpp:             leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFDeriveFunc.cpp:  amrex::GpuArray<amrex::Real, NUM_SPECIES> zk;
Source/Efield/PeleLMeX_EFDeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFDeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFDeriveFunc.cpp:  amrex::GpuArray<amrex::Real, NUM_SPECIES> zk;
Source/Efield/PeleLMeX_EFDeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFDeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFDeriveFunc.cpp:  amrex::GpuArray<amrex::Real, NUM_SPECIES> zk;
Source/Efield/PeleLMeX_EFDeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFDeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFDeriveFunc.cpp:  amrex::GpuArray<amrex::Real, NUM_SPECIES> zk;
Source/Efield/PeleLMeX_EFDeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EOS_Extension.H:AMREX_GPU_HOST_DEVICE
Source/Efield/PeleLMeX_EFUtils.cpp:  GpuArray<int, 3> blo = bx.loVect3d();
Source/Efield/PeleLMeX_EFUtils.cpp:  GpuArray<int, 3> bhi = bx.hiVect3d();
Source/Efield/PeleLMeX_EFUtils.cpp:         zk = zk] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFUtils.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFUtils.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/PeleLMeX_EFUtils.cpp:             lprobparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFUtils.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFUtils.cpp:    for (MFIter mfi(ldataR_p->I_R, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/PeleLMeX_EFUtils.cpp:        [YnEdot, nEdot, invmwt] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFUtils.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> bndry_func(
Source/Efield/PeleLMeX_EFUtils.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> crse_bndry_func(
Source/Efield/PeleLMeX_EFUtils.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirDummy>> fine_bndry_func(
Source/Efield/PeleLMeX_EFUtils.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirnE>> bndry_func(
Source/Efield/PeleLMeX_EFUtils.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirnE>> crse_bndry_func(
Source/Efield/PeleLMeX_EFUtils.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirnE>> fine_bndry_func(
Source/Efield/PeleLMeX_EFUtils.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirPhiV>> bndry_func(
Source/Efield/PeleLMeX_EFUtils.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirPhiV>> crse_bndry_func(
Source/Efield/PeleLMeX_EFUtils.cpp:    PhysBCFunct<GpuBndryFuncFab<PeleLMCCFillExtDirPhiV>> fine_bndry_func(
Source/Efield/PeleLMeX_EFPoisson.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFPoisson.cpp:    for (MFIter mfi(*rhsPoisson[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/PeleLMeX_EFPoisson.cpp:             zk = zk] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFTransport.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFTransport.cpp:    for (MFIter mfi(ldata_p->diffE_cc, TilingIfNotGPU()); mfi.isValid();
Source/Efield/PeleLMeX_EFTransport.cpp:           m_fixedKappaE] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFIonDrift.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFIonDrift.cpp:    for (MFIter mfi(mobH_cc, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/PeleLMeX_EFIonDrift.cpp:         mob_h] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/Efield/PeleLMeX_EFIonDrift.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFIonDrift.cpp:      for (MFIter mfi(mobH_ec[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/PeleLMeX_EFIonDrift.cpp:           Ud_Sp] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/Efield/PeleLMeX_EFIonDrift.cpp:      PhysBCFunct<GpuBndryFuncFab<umacFill>> crse_bndry_func(
Source/Efield/PeleLMeX_EFIonDrift.cpp:      Array<PhysBCFunct<GpuBndryFuncFab<umacFill>>, AMREX_SPACEDIM>
Source/Efield/PeleLMeX_EFIonDrift.cpp:      PhysBCFunct<GpuBndryFuncFab<umacFill>> fine_bndry_func(
Source/Efield/PeleLMeX_EFIonDrift.cpp:      Array<PhysBCFunct<GpuBndryFuncFab<umacFill>>, AMREX_SPACEDIM>
Source/Efield/PeleLMeX_EFIonDrift.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFIonDrift.cpp:    for (MFIter mfi(advData->umac[lev][idim], TilingIfNotGPU()); mfi.isValid();
Source/Efield/PeleLMeX_EFIonDrift.cpp:        [umac, Ud_Sp] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/Efield/LinOps/AMReX_MLCellABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
Source/Efield/LinOps/AMReX_MLCellABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLCellABecCecLap.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/LinOps/AMReX_MLCellABecCecLap.cpp:      for (MFIter mfi(*crse, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/LinOps/AMReX_MLCellABecCecLap.cpp:          [=] AMREX_GPU_HOST_DEVICE(Box const& b) -> ReduceTuple {
Source/Efield/LinOps/AMReX_MLCellABecCecLap.cpp:#ifdef AMREX_USE_GPU
Source/Efield/LinOps/AMReX_MLCellABecCecLap.cpp:        Gpu::inLaunchRegion() &&
Source/Efield/LinOps/AMReX_MLCellABecCecLap.cpp:          [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/Efield/LinOps/AMReX_MLCellABecCecLap.cpp:        Gpu::streamSynchronize();
Source/Efield/LinOps/AMReX_MLCellABecCecLap.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/LinOps/AMReX_MLCellABecCecLap.cpp:        for (MFIter mfi(*m_overset_mask[amrlev][mglev], TilingIfNotGPU());
Source/Efield/LinOps/AMReX_MLCellABecCecLap.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/LinOps/AMReX_MLCellABecCecLap.cpp:    for (MFIter mfi(*m_overset_mask[amrlev][0], TilingIfNotGPU());
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE void
Source/Efield/LinOps/AMReX_MLCellABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
Source/Efield/LinOps/AMReX_MLCellABecCecLap_3D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE void
Source/Efield/LinOps/AMReX_MLCellABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
Source/Efield/LinOps/AMReX_MLCellABecCecLap_1D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLap_2D_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE void
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:      for (MFIter mfi(a[mglev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:  for (MFIter mfi(out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:  for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:  if (Gpu::notInLaunchRegion())
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:    Gpu::AsyncArray<Array4<Real const>> aa(ha.data(), 2 * AMREX_SPACEDIM);
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:  if (Gpu::notInLaunchRegion())
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:  if (Gpu::notInLaunchRegion())
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:  if (Gpu::notInLaunchRegion())
Source/Efield/LinOps/AMReX_MLABecCecLaplacian.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EF.H:amrex::GpuArray<amrex::Real, NUM_SPECIES> zk;
Source/Efield/PeleLMeX_EFTimeStep.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFTimeStep.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/PeleLMeX_EFTimeStep.cpp:        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFTimeStep.cpp:        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFTimeStep.cpp:        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFTimeStep.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFTimeStep.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/PeleLMeX_EFTimeStep.cpp:        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFTimeStep.cpp:      [dx, cfl_lcl] AMREX_GPU_HOST_DEVICE(
Source/Efield/PeleLMeX_EFTimeStep.cpp:#if !defined(__CUDACC__) || (__CUDACC_VER_MAJOR__ != 9) || \
Source/Efield/PeleLMeX_EFTimeStep.cpp:  (__CUDACC_VER_MINOR__ != 2)
Source/Efield/PeleLMeX_EF_K.H:AMREX_GPU_HOST_DEVICE
Source/Efield/PeleLMeX_EF_K.H:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxinv,
Source/Efield/PeleLMeX_EF_K.H:AMREX_GPU_HOST_DEVICE
Source/Efield/PeleLMeX_EF_K.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES> a_zk,
Source/Efield/PeleLMeX_EF_K.H:AMREX_GPU_HOST_DEVICE
Source/Efield/PeleLMeX_EF_K.H:AMREX_GPU_HOST_DEVICE
Source/Efield/PeleLMeX_EF_K.H:AMREX_GPU_DEVICE
Source/Efield/PeleLMeX_EF_K.H:AMREX_GPU_HOST_DEVICE
Source/Efield/PeleLMeX_EF_K.H:AMREX_GPU_HOST_DEVICE
Source/Efield/PeleLMeX_EF_K.H:AMREX_GPU_HOST_DEVICE
Source/Efield/PeleLMeX_EF_K.H:  amrex::GpuArray<int, 3> const bxlo,
Source/Efield/PeleLMeX_EF_K.H:  amrex::GpuArray<int, 3> const bxhi,
Source/Efield/PeleLMeX_EF_K.H:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
Source/Efield/PeleLMeX_EF_K.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES> const a_zk,
Source/Efield/PeleLMeX_EF_K.H:AMREX_GPU_HOST_DEVICE
Source/Efield/PeleLMeX_EFNLSolve.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFNLSolve.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/PeleLMeX_EFNLSolve.cpp:             a_sstep] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFNLSolve.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFNLSolve.cpp:    for (MFIter mfi(ldataNLs_p->backgroundCharge, TilingIfNotGPU());
Source/Efield/PeleLMeX_EFNLSolve.cpp:         factor, zk = zk] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFNLSolve.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFNLSolve.cpp:    for (MFIter mfi(ldataNLs_p->nlResid, TilingIfNotGPU()); mfi.isValid();
Source/Efield/PeleLMeX_EFNLSolve.cpp:             scalLap] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFNLSolve.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFNLSolve.cpp:      for (MFIter mfi(ldataNLs_p->uEffnE[idim], TilingIfNotGPU());
Source/Efield/PeleLMeX_EFNLSolve.cpp:               kappa_e] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFNLSolve.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFNLSolve.cpp:    for (MFIter mfi(a_nE, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/PeleLMeX_EFNLSolve.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFNLSolve.cpp:    for (MFIter mfi(a_nE, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/PeleLMeX_EFNLSolve.cpp:          xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFNLSolve.cpp:          ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFNLSolve.cpp:          zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFNLSolve.cpp:        xbx, [u, xstate, xflux] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFNLSolve.cpp:        ybx, [v, ystate, yflux] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFNLSolve.cpp:        zbx, [w, zstate, zflux] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFNLSolve.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFNLSolve.cpp:    for (MFIter mfi(nEKe, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/PeleLMeX_EFNLSolve.cpp:              do_Schur] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFNLSolve.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFNLSolve.cpp:  for (MFIter mfi(ccMF, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/PeleLMeX_EFNLSolve.cpp:              edomain] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Efield/PeleLMeX_EFReactions.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Efield/PeleLMeX_EFReactions.cpp:  for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Efield/PeleLMeX_EFReactions.cpp:           nEdot] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Radiation.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Radiation.cpp:    for (amrex::MFIter mfi(*(m_extSource[lev]), amrex::TilingIfNotGPU());
Source/PeleLMeX_Radiation.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Radiation.cpp:    for (amrex::MFIter mfi(*(m_extSource[lev]), amrex::TilingIfNotGPU());
Source/PeleLMeX_Temporals.cpp:        [=] AMREX_GPU_DEVICE(
Source/PeleLMeX_Temporals.cpp:          int box_no, int i, int j, int k) noexcept -> GpuTuple<Real, Real> {
Source/PeleLMeX_Temporals.cpp:        [=] AMREX_GPU_DEVICE(
Source/PeleLMeX_Temporals.cpp:          int box_no, int i, int j, int k) noexcept -> GpuTuple<Real, Real> {
Source/PeleLMeX_Temporals.cpp:        [=] AMREX_GPU_DEVICE(
Source/PeleLMeX_Temporals.cpp:          int box_no, int i, int j, int k) noexcept -> GpuTuple<Real, Real> {
Source/PeleLMeX_Temporals.cpp:        [=] AMREX_GPU_DEVICE(
Source/PeleLMeX_Temporals.cpp:          int box_no, int i, int j, int k) noexcept -> GpuTuple<Real, Real> {
Source/PeleLMeX_Temporals.cpp:        [=] AMREX_GPU_DEVICE(
Source/PeleLMeX_Temporals.cpp:          int box_no, int i, int j, int k) noexcept -> GpuTuple<Real, Real> {
Source/PeleLMeX_Temporals.cpp:        [=] AMREX_GPU_DEVICE(
Source/PeleLMeX_Temporals.cpp:          int box_no, int i, int j, int k) noexcept -> GpuTuple<Real, Real> {
Source/PeleLMeX_Temporals.cpp:          [=] AMREX_GPU_DEVICE(
Source/PeleLMeX_Temporals.cpp:            int box_no, int i, int j, int k) noexcept -> GpuTuple<Real, Real> {
Source/PeleLMeX_Temporals.cpp:          [=] AMREX_GPU_DEVICE(
Source/PeleLMeX_Temporals.cpp:            int box_no, int i, int j, int k) noexcept -> GpuTuple<Real, Real> {
Source/PeleLMeX_Temporals.cpp:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> area;
Source/PeleLMeX_Temporals.cpp:          [=] AMREX_GPU_DEVICE(
Source/PeleLMeX_Temporals.cpp:            int box_no, int i, int j, int k) noexcept -> GpuTuple<Real, Real> {
Source/PeleLMeX_Temporals.cpp:            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> point_coordinates{
Source/PeleLMeX_ODEQty.cpp:      *m_extSource[lev], [state_arrs, ext_src_arrs, dt = m_dt] AMREX_GPU_DEVICE(
Source/PeleLMeX_ODEQty.cpp:    Gpu::streamSynchronize();
Source/PeleLMeX_Advection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Advection.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Advection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Advection.cpp:    for (MFIter mfi(ldataOld_p->state, TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_Advection.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Advection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Advection.cpp:    for (MFIter mfi(advData->Forcing[lev], TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_Advection.cpp:         leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Advection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Advection.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Advection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Advection.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Advection.cpp:            ebx, [rho_ed] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Advection.cpp:                  afrac] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Advection.cpp:            [rho_ed, rhoY_ed] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Advection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Advection.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Advection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Advection.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Advection.cpp:            ebx, [rhoHm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Advection.cpp:                  leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Advection.cpp:                  leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Advection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Advection.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Advection.cpp:      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_Advection.cpp:  Gpu::streamSynchronize();
Source/PeleLMeX_Advection.cpp:      advData->AofS[lev], [=, dt = m_dt] AMREX_GPU_DEVICE(
Source/PeleLMeX_Advection.cpp:  Gpu::streamSynchronize();
Source/PeleLMeX_Advection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Advection.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Advection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Advection.cpp:    for (MFIter mfi(ldataNew_p->state, TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_Advection.cpp:         dt = m_dt] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_TransportProp.cpp:        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_TransportProp.cpp:      Gpu::streamSynchronize();
Source/PeleLMeX_TransportProp.cpp:          [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_TransportProp.cpp:          [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_TransportProp.cpp:          [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_TransportProp.cpp:      Gpu::streamSynchronize();
Source/PeleLMeX_TransportProp.cpp:        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_TransportProp.cpp:  Gpu::streamSynchronize();
Source/PeleLMeX_TransportProp.cpp:    GpuArray<Real, NUM_SPECIES> mwt{0.0};
Source/PeleLMeX_TransportProp.cpp:      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_TransportProp.cpp:  Gpu::streamSynchronize();
Source/PeleLMeX_TransportProp.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_TransportProp.cpp:  for (MFIter mfi(beta_cc, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_TransportProp.cpp:              edomain] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_TransportProp.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_TransportProp.cpp:      for (MFIter mfi(beta_ec[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_TransportProp.cpp:          ebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Soot.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Soot.cpp:    for (MFIter mfi(*(m_extSource[lev]), TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_Soot.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Soot.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Soot.cpp:        gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Soot.cpp:          GpuArray<Real, NUM_SOOT_MOMENTS + 1> moments;
Source/PeleLMeX_Projection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Projection.cpp:      for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Projection.cpp:               dummy_dt] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Projection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Projection.cpp:      for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Projection.cpp:               dummy_dt] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Projection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Projection.cpp:      for (MFIter mfi(*rhoHalf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Projection.cpp:               a_dt] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Projection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Projection.cpp:      for (MFIter mfi(ldataNew_p->state, TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_Projection.cpp:           rho = m_rho] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Projection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Projection.cpp:        for (MFIter mfi(rhs_cc[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Projection.cpp:               m_closed_chamber] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Projection.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Projection.cpp:    for (MFIter mfi(ldata_p->gp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Projection.cpp:           gp_proj_arr] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Projection.cpp:                p_proj_arr] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Projection.cpp:           gp_proj_arr] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Projection.cpp:                p_proj_arr] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Projection.cpp:      [=, ncomp = a_mf.nComp()] AMREX_GPU_DEVICE(
Source/PeleLMeX_Projection.cpp:    Gpu::streamSynchronize();
Source/PeleLMeX_Projection.cpp:      [=, ncomp = a_mf.nComp()] AMREX_GPU_DEVICE(
Source/PeleLMeX_DeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:    bx, NUM_SPECIES, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:      bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:      bx, [=, rho = a_pelelm->m_rho] AMREX_GPU_DEVICE(
Source/PeleLMeX_DeriveFunc.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:       rho = a_pelelm->m_rho] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:       rho = a_pelelm->m_rho] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:  amrex::GpuArray<amrex::Real, NUM_SPECIES> fact_Bilger;
Source/PeleLMeX_DeriveFunc.cpp:         denom_inv] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 1> Cweights;
Source/PeleLMeX_DeriveFunc.cpp:    bx, [=, revert = a_pelelm->m_Crevert] AMREX_GPU_DEVICE(
Source/PeleLMeX_DeriveFunc.cpp:           ltransparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:         leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:         leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DeriveFunc.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_EB.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_EB.cpp:  for (MFIter mfi(a_tmpDiv, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_EB.cpp:            Box(scratch), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_EB.cpp:          bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_EB.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_EB.cpp:    for (MFIter mfi(a_imask, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_EB.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_EB.cpp:    for (MFIter mfi(mask_tmp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_EB.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_EB.cpp:  for (MFIter mfi(a_tmpDiv, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_EB.cpp:            Box(scratch), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_EB.cpp:          bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_EB.cpp:    Gpu::copy(
Source/PeleLMeX_EB.cpp:      Gpu::hostToDevice, coveredState_h.begin(), coveredState_h.end(),
Source/PeleLMeX_EB.cpp:    Gpu::copy(
Source/PeleLMeX_EB.cpp:      Gpu::hostToDevice, coveredState_h.begin(), coveredState_h.end(),
Source/PeleLMeX_EB.cpp:      for (MFIter mfi(ldataNew_p->state, TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_EB.cpp:  if (Gpu::notInLaunchRegion()) {
Source/PeleLMeX_EB.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_EB.cpp:        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_EB.cpp:  if (Gpu::notInLaunchRegion()) {
Source/PeleLMeX_EB.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_EB.cpp:        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_EB.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_EB.cpp:    for (MFIter mfi(*a_vel[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_EB.cpp:                 wmac_fab)] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_UMac.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_UMac.cpp:    for (MFIter mfi(advData->chi[lev], TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_UMac.cpp:                a_sdcIter] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_UMac.cpp:                a_sdcIter] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_UMac.cpp:                a_sdcIter] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_UMac.cpp:  PhysBCFunct<GpuBndryFuncFab<umacFill>> crse_bndry_func(
Source/PeleLMeX_UMac.cpp:  Array<PhysBCFunct<GpuBndryFuncFab<umacFill>>, AMREX_SPACEDIM> cbndyFuncArr = {
Source/PeleLMeX_UMac.cpp:  PhysBCFunct<GpuBndryFuncFab<umacFill>> fine_bndry_func(
Source/PeleLMeX_UMac.cpp:  Array<PhysBCFunct<GpuBndryFuncFab<umacFill>>, AMREX_SPACEDIM> fbndyFuncArr = {
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const gravity,
Source/PeleLMeX_K.H:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const gp0,
Source/PeleLMeX_K.H:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const /*dx*/,
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_HOST_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_HOST_DEVICE
Source/PeleLMeX_K.H:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dxinv,
Source/PeleLMeX_K.H:AMREX_GPU_HOST_DEVICE
Source/PeleLMeX_K.H:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dxinv,
Source/PeleLMeX_K.H:#ifndef AMREX_USE_GPU
Source/PeleLMeX_K.H:AMREX_GPU_HOST_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_HOST_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_HOST_DEVICE
Source/PeleLMeX_K.H:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM * AMREX_SPACEDIM> g2_ij;
Source/PeleLMeX_K.H:AMREX_GPU_HOST_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_HOST_DEVICE
Source/PeleLMeX_K.H:AMREX_GPU_HOST_DEVICE
Source/PeleLMeX_K.H:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> I{0.0};
Source/PeleLMeX_Init.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Init.cpp:    for (MFIter mfi(*m_signedDist0, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Init.cpp:        bx, [sd_cc, sd_nd] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Init.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Init.cpp:  for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Init.cpp:      bx, [=, m_incompressible = m_incompressible] AMREX_GPU_DEVICE(
Source/PeleLMeX_BCfill.H:  AMREX_GPU_HOST
Source/PeleLMeX_BCfill.H:  AMREX_GPU_DEVICE
Source/PeleLMeX_BCfill.H:  AMREX_GPU_HOST
Source/PeleLMeX_BCfill.H:  AMREX_GPU_DEVICE
Source/PeleLMeX_BCfill.H:  AMREX_GPU_HOST
Source/PeleLMeX_BCfill.H:  AMREX_GPU_DEVICE
Source/PeleLMeX_BCfill.H:  AMREX_GPU_HOST
Source/PeleLMeX_BCfill.H:  AMREX_GPU_DEVICE
Source/PeleLMeX_BCfill.H:  AMREX_GPU_HOST
Source/PeleLMeX_BCfill.H:  AMREX_GPU_DEVICE
Source/PeleLMeX_BCfill.H:  AMREX_GPU_HOST
Source/PeleLMeX_BCfill.H:  AMREX_GPU_DEVICE
Source/PeleLMeX_BCfill.H:  AMREX_GPU_HOST
Source/PeleLMeX_BCfill.H:  AMREX_GPU_DEVICE
Source/PeleLMeX_BCfill.H:  AMREX_GPU_HOST
Source/PeleLMeX_BCfill.H:  AMREX_GPU_DEVICE
Source/PeleLMeX_BCfill.H:  AMREX_GPU_DEVICE
Source/PeleLMeX_Eos.cpp:    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_Eos.cpp:  Gpu::streamSynchronize();
Source/PeleLMeX_Eos.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Eos.cpp:    for (MFIter mfi(ldata_p->divu, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Eos.cpp:          bx, [divu] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Eos.cpp:               leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Eos.cpp:           use_react, leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Eos.cpp:    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_Eos.cpp:  Gpu::streamSynchronize();
Source/PeleLMeX_Eos.cpp:  Gpu::streamSynchronize();
Source/PeleLMeX_Eos.cpp:    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_Eos.cpp:  Gpu::streamSynchronize();
Source/PeleLMeX_Eos.cpp:    *a_dPdt, [=, dt = m_dt, dpdt_fac = m_dpdtFactor] AMREX_GPU_DEVICE(
Source/PeleLMeX_Eos.cpp:  Gpu::streamSynchronize();
Source/PeleLMeX_Eos.cpp:      *ThetaHalft[lev], [=, pOld = m_pOld, pNew = m_pNew] AMREX_GPU_DEVICE(
Source/PeleLMeX_Eos.cpp:  Gpu::streamSynchronize();
Source/PeleLMeX_Eos.cpp:      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_Eos.cpp:  Gpu::streamSynchronize();
Source/PeleLMeX.H:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_gravity{
Source/PeleLMeX.H:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_background_gp{
Source/PeleLMeX.H:  amrex::Gpu::DeviceVector<amrex::Real> coveredState_d;
Source/PeleLMeX.H:  amrex::GpuArray<amrex::Real, NUM_SPECIES + 1> m_Cweights;
Source/PeleLMeX_Tagging.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Tagging.cpp:  for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Tagging.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) {
Source/PeleLMeX_Tagging.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Tagging.cpp:    for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Tagging.cpp:      amrex::ParallelFor(bx, [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) {
Source/PeleLMeX_Forces.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Forces.cpp:  for (MFIter mfi(*a_velForce, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Forces.cpp:         divTau_arr, force_arr] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Forces.cpp:             force_arr] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Forces.cpp:     ps_dir = m_ctrl_flameDir] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Forces.cpp:           [n]] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_Forces.cpp:      Gpu::streamSynchronize();
Source/PeleLMeX_FlowController.cpp:    Gpu::DeviceVector<Real> s_ext_v(NVAR);
Source/PeleLMeX_FlowController.cpp:       lpmfdata] AMREX_GPU_DEVICE(int /*i*/, int /*j*/, int /*k*/) noexcept {
Source/PeleLMeX_FlowController.cpp:    Gpu::copy(Gpu::deviceToHost, s_ext_v.begin(), s_ext_v.end(), s_ext.begin());
Source/PeleLMeX_FlowController.cpp:    Gpu::copy(Gpu::hostToDevice, fcdata_h, fcdata_h + 1, fcdata_d);
Source/PeleLMeX_FlowController.cpp:        [=] AMREX_GPU_HOST_DEVICE(
Source/PeleLMeX_FlowController.cpp:        [=] AMREX_GPU_HOST_DEVICE(
Source/PeleLMeX_Utils.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Utils.cpp:  for (MFIter mfi(a_divergence, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Utils.cpp:        [divergence] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Utils.cpp:             vol, scale] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Utils.cpp:             scale] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Utils.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Utils.cpp:  for (MFIter mfi(a_divergence, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Utils.cpp:        [divergence] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Utils.cpp:             scale] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Utils.cpp:               vol, scale] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Utils.cpp:               scale] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Utils.cpp:  const GpuArray<Real, AMREX_SPACEDIM> dx = Geom(lev).CellSizeArray();
Source/PeleLMeX_Utils.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Utils.cpp:  for (MFIter mfi(a_divergence, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Utils.cpp:        [divergence] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Utils.cpp:        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Utils.cpp:             scale] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Utils.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Utils.cpp:  for (MFIter mfi(a_divergence, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Utils.cpp:        bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Utils.cpp:          bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Utils.cpp:          bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Utils.cpp:      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_Utils.cpp:    Gpu::streamSynchronize();
Source/PeleLMeX_Utils.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Utils.cpp:        for (MFIter mfi(*m_coveredMask[lev], TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_Utils.cpp:              is.second, [mask] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Utils.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Utils.cpp:    for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Utils.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Utils.cpp:    for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Utils.cpp:        [vol, comp] AMREX_GPU_HOST_DEVICE(
Source/PeleLMeX_Utils.cpp:        [vol, comp] AMREX_GPU_HOST_DEVICE(
Source/PeleLMeX_Utils.cpp:        [comp] AMREX_GPU_HOST_DEVICE(
Source/PeleLMeX_Utils.cpp:        [comp] AMREX_GPU_HOST_DEVICE(
Source/PeleLMeX_Utils.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Utils.cpp:#ifdef AMREX_USE_GPU
Source/PeleLMeX_Utils.cpp:    if (Gpu::inLaunchRegion()) {
Source/PeleLMeX_Utils.cpp:        [=] AMREX_GPU_DEVICE(
Source/PeleLMeX_Utils.cpp:          int box_no, int i, int j, int k) noexcept -> GpuTuple<Real> {
Source/PeleLMeX_Utils.cpp:#ifdef AMREX_USE_GPU
Source/PeleLMeX_Utils.cpp:    if (Gpu::inLaunchRegion()) {
Source/PeleLMeX_Utils.cpp:        [=] AMREX_GPU_DEVICE(
Source/PeleLMeX_Utils.cpp:          int box_no, int i, int j, int k) noexcept -> GpuTuple<Real> {
Source/PeleLMeX_Utils.cpp:#ifdef AMREX_USE_GPU
Source/PeleLMeX_Utils.cpp:    if (Gpu::inLaunchRegion()) {
Source/PeleLMeX_Utils.cpp:        [=] AMREX_GPU_DEVICE(
Source/PeleLMeX_Utils.cpp:          int box_no, int i, int j, int k) noexcept -> GpuTuple<Real> {
Source/PeleLMeX_Utils.cpp:#ifdef AMREX_USE_GPU
Source/PeleLMeX_Utils.cpp:    if (Gpu::inLaunchRegion()) {
Source/PeleLMeX_Utils.cpp:        [=] AMREX_GPU_DEVICE(
Source/PeleLMeX_Utils.cpp:          int box_no, int i, int j, int k) noexcept -> GpuTuple<Real> {
Source/PeleLMeX_Utils.cpp:#ifdef AMREX_USE_GPU
Source/PeleLMeX_Utils.cpp:  Long free_mem_avail = Gpu::Device::freeMemAvailable() / (1024 * 1024);
Source/PeleLMeX_Utils.cpp:  Print() << "     [" << a_message << "] GPU mem. avail. (MB) "
Source/PeleLMeX_Utils.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Utils.cpp:  for (MFIter mfi(*a_signDist, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Utils.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Utils.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Utils.cpp:    for (MFIter mfi(*a_signDist, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Utils.cpp:      ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_EBUserDefined.H:AMREX_GPU_DEVICE
Source/PeleLMeX_EBUserDefined.H:  // Aborting here (message will not show on GPUs)
Source/PeleLMeX_EBUserDefined.H:AMREX_GPU_DEVICE
Source/PeleLMeX_Utils.H:amrex::Gpu::DeviceVector<T>
Source/PeleLMeX_Utils.H:  amrex::Gpu::DeviceVector<T> v_d(ncomp);
Source/PeleLMeX_Utils.H:#ifdef AMREX_USE_GPU
Source/PeleLMeX_Utils.H:  amrex::Gpu::htod_memcpy(v_d.data(), v.data(), sizeof(T) * ncomp);
Source/PeleLMeX_Diffusion.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Diffusion.cpp:    for (MFIter mfi(*a_spec[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Diffusion.cpp:                    bc_hi] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:               bc_hi] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:                bc_hi] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Diffusion.cpp:    for (MFIter mfi(Wbar[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Diffusion.cpp:              leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Diffusion.cpp:      for (MFIter mfi(*a_beta[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Diffusion.cpp:                  edomain] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:                    leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Diffusion.cpp:      for (MFIter mfi(*a_beta[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Diffusion.cpp:                  edomain] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:             edomain] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:             spsoretFlux_ar] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Diffusion.cpp:    for (MFIter mfi(Enth, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Diffusion.cpp:          gbx, [Hi_arr] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:                leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:                leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Diffusion.cpp:    for (MFIter mfi(Enth, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Diffusion.cpp:                 enth_ar] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Diffusion.cpp:    for (MFIter mfi(advData->Forcing[lev], TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_Diffusion.cpp:        [rhoY_o, fY, dt = m_dt] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Diffusion.cpp:      for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Diffusion.cpp:             flux_wbar] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Diffusion.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Diffusion.cpp:      for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Diffusion.cpp:             flux_soret] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Diffusion.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Diffusion.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Diffusion.cpp:           m_use_soret] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_Diffusion.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Diffusion.cpp:    for (MFIter mfi(ldataNew_p->state, TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_Diffusion.cpp:        bx, [=, dt = m_dt] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_Diffusion.cpp:  Gpu::streamSynchronize();
Source/PeleLMeX_Diffusion.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Diffusion.cpp:    for (MFIter mfi(advData->Forcing[lev], TilingIfNotGPU()); mfi.isValid();
Source/PeleLMeX_Diffusion.cpp:           m_closed_chamber] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_Timestep.cpp:        [dtfac, rhoMin] AMREX_GPU_HOST_DEVICE(
Source/PeleLMeX_Timestep.cpp:        [dtfac, rhoMin, dxinv] AMREX_GPU_HOST_DEVICE(
Source/PeleLMeX_Timestep.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_Timestep.cpp:    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_Timestep.cpp:             a_dt] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_ProblemSpecificFunctions.cpp:    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
Source/PeleLMeX_ProblemSpecificFunctions.cpp:  Gpu::streamSynchronize();
Source/PeleLMeX_DiffusionOp.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_DiffusionOp.cpp:    for (MFIter mfi(phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_DiffusionOp.cpp:         have_density] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_DiffusionOp.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_DiffusionOp.cpp:    for (MFIter mfi(phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_DiffusionOp.cpp:         have_density] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_DiffusionOp.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_DiffusionOp.cpp:    for (MFIter mfi(phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_DiffusionOp.cpp:         have_density] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_DiffusionOp.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_DiffusionOp.cpp:    for (MFIter mfi(phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_DiffusionOp.cpp:         have_density] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_DiffusionOp.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_DiffusionOp.cpp:    for (MFIter mfi(phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_DiffusionOp.cpp:         have_density] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_DiffusionOp.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_DiffusionOp.cpp:    for (MFIter mfi(phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_DiffusionOp.cpp:         have_density] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/PeleLMeX_DiffusionOp.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_DiffusionOp.cpp:      for (MFIter mfi(*a_divtau[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_DiffusionOp.cpp:          bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/PeleLMeX_DiffusionOp.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PeleLMeX_DiffusionOp.cpp:    for (MFIter mfi(rhs[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PeleLMeX_DiffusionOp.cpp:             ->m_incompressible] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Utils/RunScripts/OLCF/FlameSheet_DODQSS_Frontier.sh:cmd "module load cmake cray-python craype-x86-trento craype-accel-amd-gfx90a rocm/5.4.0"
Utils/RunScripts/OLCF/FlameSheet_DODQSS_Frontier.sh:ARGS="input.3d-regt geometry.prob_lo=0.0 0.0 0.0 geometry.prob_hi=0.016 0.016 0.016 amr.n_cell=64 64 64 amr.max_level=2 amr.n_error_buf=2 2 2 2 amr.max_grid_size=128 amr.blocking_factor=16 prob.P_mean=101325.0 prob.standoff=-.012 prob.pertmag=0.0004 prob.pertlength=0.008 peleLM.num_init_iter=1 peleLM.do_temporals=0 amr.max_step=16 amr.dt_shrink=1.0 amr.fixed_dt=2.5e-7 peleLM.v=3 cvode.solve_type=magma_direct amr.plot_int=-1 peleLM.diagnostics=xnormal peleLM.xnormal.int=500 amrex.abort_on_out_of_gpu_memory=1 amr.yH.value_greater=2.0e-5"
Utils/RunScripts/OLCF/FlameSheet_DODQSS_Frontier.sh:cmd "srun -N1 -n8 --gpus-per-node=8 --gpu-bind=closest ./PeleLMeX3d.hip.x86-trento.TPROF.MPI.HIP.ex ${ARGS}"
Utils/RunScripts/OLCF/FlameSheet_DRM19_Frontier.sh:cmd "module load cmake cray-python craype-x86-trento craype-accel-amd-gfx90a rocm/5.4.0"
Utils/RunScripts/OLCF/FlameSheet_DRM19_Frontier.sh:ARGS="input.3d-regt geometry.prob_lo=0.0 0.0 0.0 geometry.prob_hi=0.032 0.032 0.032 amr.n_cell=64 64 64 amr.max_level=3 amr.max_grid_size=128 amr.blocking_factor=16 prob.P_mean=101325.0 prob.standoff=-.023 prob.pertmag=0.00045 prob.pertlength=0.016 peleLM.num_init_iter=1 peleLM.do_temporals=0 amr.max_step=16 amr.dt_shrink=0.25 amr.fixed_dt=2.0e-6 peleLM.v=3 cvode.solve_type=magma_direct amr.plot_int=-1 peleLM.diagnostics=xnormal peleLM.xnormal.int=500 amrex.abort_on_out_of_gpu_memory=1"
Utils/RunScripts/OLCF/FlameSheet_DRM19_Frontier.sh:cmd "srun -N1 -n8 --gpus-per-node=8 --gpu-bind=closest ./PeleLMeX3d.hip.x86-trento.TPROF.MPI.HIP.ex ${ARGS}"
Utils/RunScripts/Scaling/WeakScaling/WeakScaling.py:            # Perlmutter: 4 CPU/GPU couples / nodes
Utils/RunScripts/Scaling/WeakScaling/WeakScaling.py:            # Crusher/Frontier: 8 CPU+GPU pair / nodes
Utils/RunScripts/Scaling/WeakScaling/WeakScaling.py:                    lineout = "srun -N{} -n{} -c7 --gpus-per-node=8 --gpu-bind=closest ./{} {}".format(case,case*8,args.exec,args.input_file)

```
