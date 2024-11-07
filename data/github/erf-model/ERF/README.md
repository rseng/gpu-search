# https://github.com/erf-model/ERF

```console
CMake/BuildERFExe.cmake:  if(ERF_ENABLE_CUDA)
CMake/BuildERFExe.cmake:      set_source_files_properties(${ERF_SOURCES} PROPERTIES LANGUAGE CUDA)
CMake/BuildERFExe.cmake:      message(STATUS "setting cuda for ${ERF_SOURCES}")
CMake/BuildERFExe.cmake:    LANGUAGE CUDA
CMake/BuildERFExe.cmake:    CUDA_SEPARABLE_COMPILATION ON
CMake/BuildERFExe.cmake:    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
CMake/BuildERFExe.cmake:  if(ERF_ENABLE_CUDA)
CMake/BuildERFExe.cmake:      set_source_files_properties(${ERF_SOURCES} PROPERTIES LANGUAGE CUDA)
CMake/BuildERFExe.cmake:      message(STATUS "setting cuda for ${ERF_SOURCES}")
CMake/BuildERFExe.cmake:    LANGUAGE CUDA
CMake/BuildERFExe.cmake:    CUDA_SEPARABLE_COMPILATION ON
CMake/BuildERFExe.cmake:    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
CMake/SetERFCompileFlags.cmake:if(ERF_ENABLE_CUDA)
CMake/SetERFCompileFlags.cmake:  list(APPEND ERF_CUDA_FLAGS "--expt-relaxed-constexpr")
CMake/SetERFCompileFlags.cmake:  list(APPEND ERF_CUDA_FLAGS "--expt-extended-lambda")
CMake/SetERFCompileFlags.cmake:  list(APPEND ERF_CUDA_FLAGS "--Wno-deprecated-gpu-targets")
CMake/SetERFCompileFlags.cmake:  list(APPEND ERF_CUDA_FLAGS "-m64")
CMake/SetERFCompileFlags.cmake:  if(ENABLE_CUDA_FASTMATH)
CMake/SetERFCompileFlags.cmake:    list(APPEND ERF_CUDA_FLAGS "--use_fast_math")
CMake/SetERFCompileFlags.cmake:  separate_arguments(ERF_CUDA_FLAGS)
CMake/SetERFCompileFlags.cmake:  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ERF_CUDA_FLAGS}>)
CMake/SetERFCompileFlags.cmake:  set(CMAKE_CUDA_FLAGS ${NVCC_ARCH_FLAGS})
CMake/SetERFCompileFlags.cmake:  set_cuda_architectures(AMReX_CUDA_ARCH)
CMake/SetERFCompileFlags.cmake:     CUDA_ARCHITECTURES "${AMREX_CUDA_ARCHS}"
CMake/SetERFCompileFlags.cmake:    LANGUAGE CUDA
CMake/SetERFCompileFlags.cmake:    CUDA_SEPARABLE_COMPILATION ON
CMake/SetERFCompileFlags.cmake:    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
CMake/SetAmrexOptions.cmake:set(AMReX_CUDA ${ERF_ENABLE_CUDA})
CMake/SetAmrexOptions.cmake:if(ERF_ENABLE_CUDA)
CMake/SetAmrexOptions.cmake:  set(AMReX_GPU_BACKEND CUDA CACHE STRING "AMReX GPU type" FORCE)
CMake/SetAmrexOptions.cmake:  set(AMReX_CUDA_WARN_CAPTURE_THIS OFF)
CMake/SetAmrexOptions.cmake:  set(AMReX_CUDA_ERROR_CAPTURE_THIS ON)
CMake/SetAmrexOptions.cmake:  set(AMReX_GPU_BACKEND HIP CACHE STRING "AMReX GPU type" FORCE)
CMake/SetAmrexOptions.cmake:  set(AMReX_GPU_BACKEND SYCL CACHE STRING "AMReX GPU type" FORCE)
Docs/sphinx_doc/Applications_Requirements.rst:Most importantly, WRF’s software architecture cannot effectively utilize Graphics Processing Units (GPUs), a requirement both for the increased computational expense of emerging applications, as well as for use of the next generation of high-performance computing (HPC) assets slated for rollout first at the DOE national laboratories, and eventually defining the majority of available scientific computing hardware. Moreover, no current code development efforts pursuing GPU-compatible architectures are addressing the downscaling functionality required of wind energy applications, with the other efforts pursuing either numerical weather prediction (NWP) on global-scale grids, or focusing on microscale simulation without including the critical mesoscale component defining the scales of energy input.
Docs/sphinx_doc/Applications_Requirements.rst:In short, ERF will provide a modern, flexible, and efficient GPU-capable software framework to supply critical atmospheric and environmental drivers of energy availability to the microscale wind plant environment, thereby enabling the significant advances in siting, design and operation required to support continued industry expansion into increasingly challenging environments and high penetration scenarios.
Docs/sphinx_doc/Applications_Requirements.rst:The ERF model is being designed for use by moderately skilled to advanced practitioners of computational fluid dynamics (CFD) codes such as OpenFOAM and WRF, working within the wind energy industry, national laboratories and university research groups, in applications involving numerical simulations of atmospheric flows, and the interactions of those flows with wind turbines, wind plants, and multiple interacting plants in regions of dense development. Beyond its applications, an additional goal of ERF is to serve as a bridge for users and developers of existing CFD and NWP codes to transition away from older legacy codes into a modern software architecture and programming paradigm that efficiently utilizes the next generation of GPU-accelerated HPC hardware, while providing opportunities for expanded applicability to modern wind energy research challenges that require the implementation and integration of new computational capabilities. ERF also targets user-developers who seek to contribute new code back to the ERF code base to improve and expand it, as has occurred within other open-source codes such as WRF and OpenFOAM.
Docs/sphinx_doc/Applications_Requirements.rst:A key application targeted by ERF is wind resource characterization, the assessment of the potential of a given site to produce power over time. Today’s commonly applied resource characterization approaches used in both mesoscale and microscale settings are necessarily of restricted fidelity due to the computational expense of higher-fidelity techniques exceeding industry resources in typical workflows. The computational burden of high fidelity will be significantly mitigated using ERF’s more efficient code architecture, algorithms, and ability to use GPU-accelerated HPC hardware, enabling improved resource characterization at all scales.
Docs/sphinx_doc/Applications_Requirements.rst:These above listed activities and phenomena define the key applications envisioned for the ERF model. However, there are two other flow simulation regimes of relevance to wind-energy that ERF is not intended to address. The first of these is weather forecasting. While ERF could, in principle, be extended to capture larger-scales of meteorological forcing, efforts are already underway elsewhere to create next-generation numerical weather NWP systems, operating at global scales, to capture the largest scales impacting weather system evolution, while also being designed to utilize GPUs for enhanced speed and efficiency. ERF will leverage these concurrent developments by interfacing with a data preprocessor to ingest forecast and analysis fields produced by these new larger-scale models. ERF will focus on the efficient downscaling of those solutions to footprints of relevance to wind energy applications, capturing the associated finer-scale mesoscale and turbulence features, as well as wind plant interactions, along the way.
Docs/sphinx_doc/Applications_Requirements.rst:1. Excellent Performance on Both CPU- and GPU-Based HPC Platforms
Docs/sphinx_doc/Applications_Requirements.rst:ERF must be able to run efficiently on both CPU- and GPU-based HPC platforms. This flexibility is required to support enhanced utilization of existing HPC architectures for which significant industry investments have been made, to serve as a vehicle to transition those users to next-generation platforms and programming paradigms, and to support current applications using GPU-accelerated hardware being rolled out today at leadership computing facilities (LCFs). To meet these use cases, ERF must compile and run on a variety of platforms, but also must be configurable for optimal performance on LCFs, including coupling with ExaWind tools on those platforms. Key metrics to assess adequate performance include superior scaling up to tens of thousands of cores on LCF systems, with several levels of mesh refinement, and solution accuracy that meets or exceeds that of legacy codes such as WRF and OpenFoam in similar applications. In addition to LCF machines at DOE labs, integration with new disk storage approaches (e.g., burst buffer at NERSC) should also be explored. Other platforms that would be desirable for ERF to utilize include emerging GPU-based small sized clusters, commodity desktops and laptops with GPU cards, and cloud resources, which are increasingly coming to replace industry-owned HPC resources at many wind energy companies.
Docs/sphinx_doc/Applications_Requirements.rst:For optimal utilization of GPU-based hardware, the ERF source code must be written in C++. However, ERF should also be able to incorporate legacy Fortran source code from other models. While Fortran code incorporated into the C++ code base will not result in optimal performance, it will allow for the rapid expansion of ERF’s capabilities, while providing a pathway to facilitate adoption of ERF by users and developers familiar with Fortran programming and legacy codes. Future development of ERF, including potential community development, can target the rewriting of desired Fortran modules into C++ for enhanced performance.
Docs/sphinx_doc/Applications_Requirements.rst:The ERF equation set will require a discretization strategy and numerical solution procedures amenable to optimization for different mesh spacings and applications. For ease of implementation and familiarity with users of other code bases, as well as ability to incorporate modules from other codes such as WRF, a finite-difference spatial discretization strategy with second-order accuracy should be employed for ERF. Options for higher-order spatial differences can be included as well, however those methods may not scale as well on GPU-based hardware.
Docs/sphinx_doc/Applications_Requirements.rst:ERF will utilize the AMReX adaptive mesh refinement framework for its computational mesh and refinement requirements. AMReX provides a flexible capability that can support all of ERF’s required mesh needs utilizing advanced data structures and memory management for robust and efficient data transfer and load balancing. Moreover, AMReX contains built-in abstractions to efficiently interface with a variety of CPU- and GPU-based HPC hardware. The continuing support of AMReX by the Exascale Computing Program makes AMReX an ideal choice for ERF.
Docs/sphinx_doc/building.rst:Building with GPU support may be done with CUDA, HIP, or SYCL.
Docs/sphinx_doc/building.rst:For CUDA, ERF requires versions >= 11.0. For HIP and SYCL, only the latest compilers are supported.
Docs/sphinx_doc/building.rst:   | USE_CUDA           | Whether to enable CUDA       | TRUE / FALSE     | FALSE       |
Docs/sphinx_doc/building.rst:      **At most one of USE_OMP, USE_CUDA, USE_HIP, USE_SYCL should be set to true.**
Docs/sphinx_doc/building.rst:   | ERF_ENABLE_CUDA           | Whether to enable CUDA       | TRUE / FALSE     | FALSE       |
Docs/sphinx_doc/building.rst:      **At most one of ERF_ENABLE_OMP, ERF_ENABLE_CUDA, ERF_ENABLE_HIP and ERF_ENABLE_SYCL should be set to true.**
Docs/sphinx_doc/building.rst:   module load cudatoolkit
Docs/sphinx_doc/building.rst:   make -j 4 COMP=gnu USE_MPI=TRUE USE_OMP=FALSE USE_CUDA=TRUE AMREX_HOME=/path_to_here/ERF/Submodules/AMReX
Docs/sphinx_doc/building.rst:             ## specify your allocation (with the _g) and that you want GPU nodes
Docs/sphinx_doc/building.rst:             #SBATCH -C gpu
Docs/sphinx_doc/building.rst:             ## we use the same number of MPI ranks per node as GPUs per node
Docs/sphinx_doc/building.rst:             #SBATCH --gpus-per-node=4
Docs/sphinx_doc/building.rst:             #SBATCH --gpu-bind=none
Docs/sphinx_doc/building.rst:             # pin to closest NIC to GPU
Docs/sphinx_doc/building.rst:             export MPICH_OFI_NIC_POLICY=GPU
Docs/sphinx_doc/building.rst:             # use GPU-aware MPI
Docs/sphinx_doc/building.rst:             #GPU_AWARE_MPI=""
Docs/sphinx_doc/building.rst:             GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"
Docs/sphinx_doc/building.rst:             # set ordering of CUDA visible devices inverse to local task IDs for optimal GPU-aware MPI
Docs/sphinx_doc/building.rst:               export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
Docs/sphinx_doc/building.rst:               ./ERF3d.gnu.MPI.CUDA.ex inputs_wrf_baseline max_step=100 ${GPU_AWARE_MPI}" \
Docs/sphinx_doc/building.rst:   module load cudatoolkit/12.2
Docs/sphinx_doc/building.rst:   cmake .. -DCMAKE_INSTALL_PREFIX=<path-to-kokkos-install-dir> -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DKokkos_ENABLE_CUDA=ON -DCMAKE_CXX_COMPILER=<full-path-to-kokkos-dir>/bin/nvcc_wrapper -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON -DKokkos_ARCH_PASCAL60=ON
Docs/sphinx_doc/building.rst:   cmake .. -DCMAKE_INSTALL_PREFIX=<path-to-amrex-install-dir> -DAMReX_GPU_BACKEND=CUDA -DAMReX_CUDA_ARCH=60 -DAMReX_MPI=OFF -DCMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64 -DAMReX_MPI=ON -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DMPI_INCLUDE_PATH=/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/include
Docs/sphinx_doc/building.rst:   cmake .. -DENABLE_CUDA=ON -DAMReX_ROOT=<full-path-to-amrex-install-dir> -DKokkos_ROOT=<full-path-to-kokkos-install-dir> -DCMAKE_CUDA_ARCHITECTURES=60 -DMPI_INCLUDE_PATH=/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/include
Docs/sphinx_doc/building.rst:Intel Xeon Sapphire Rapids nodes. It also contains a GPU partition with 4 Nvidia H100 GPUs per node.
Docs/sphinx_doc/building.rst:To run on GPUs on Kestrel, note that the machine has separate login nodes for GPU use and GPU jobs should only
Docs/sphinx_doc/building.rst:be started from GPU login nodes (accessed via ``kestrel-gpu.hpc.nrel.gov``). For compiling and running on GPUs,
Docs/sphinx_doc/building.rst:  module load cuda/12.3;
Docs/sphinx_doc/building.rst:  make realclean; make -j COMP=gnu USE_CUDA=TRUE
Docs/sphinx_doc/building.rst:When running on Kestrel, GPU node hours are charged allocation units (AUs) at 10 times the rate of CPU node hours.
Docs/sphinx_doc/building.rst:For ERF, the performance running on a Kestrel GPU node with 4 GPUs is typically 10-20x running on a CPU node
Docs/sphinx_doc/building.rst:with 96-104 MPI ranks per node, so the performance gain from on on GPUs is likely worth the higher charge
Docs/sphinx_doc/building.rst:or problems distributed across too many nodes (resulting in fewer than around 1 million cells/GPU),
Docs/sphinx_doc/building.rst:the compute capability of the GPUs may be unsaturated and the performance gain from running on GPUs
Docs/sphinx_doc/building.rst:is recommended. Otherwise, memory intensive operations such as CUDA compilation may fail. You can alternatively
Docs/sphinx_doc/index.rst:ERF is designed to run on machines from laptops to multicore CPU and hybrid CPU/GPU systems.
Docs/sphinx_doc/Performance.rst:GPU weak scaling
Docs/sphinx_doc/Performance.rst:The plot shows weak scaling of the ABL application with the Smagorinsky LES model using A100 GPUs on the Perlmutter system at NERSC.
Docs/sphinx_doc/Performance.rst:The domain size is **amr.n_cell = 128 128 512** for a **single GPU**; this is progressively scaled up to **2048 1024 512** for **128 GPUs**.
Docs/sphinx_doc/Performance.rst:This test uses all 4 GPUs per node with GPU-aware MPI communication and runs 100 time steps.
Docs/sphinx_doc/containers.rst:     1  FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04
Docs/sphinx_doc/containers.rst:    26    -DCMAKE_CUDA_ARCHITECTURES=80 \
Docs/sphinx_doc/containers.rst:    28    -DERF_ENABLE_CUDA=ON \
Docs/sphinx_doc/containers.rst:* Line 1 downloads a container base image from NVIDIA's container registry that contains the Ubuntu 22.04 operating system and CUDA 12.2.0
Docs/sphinx_doc/containers.rst:  #SBATCH --constraint=gpu
Docs/sphinx_doc/containers.rst:  srun -N 1 -n 4 -c 32 --ntasks-per-node=4 --gpus-per-node=4 ./device_wrapper \
Docs/sphinx_doc/containers.rst:  podman-hpc run --rm --mpi --gpu -v /pscratch/sd/u/user/erf/abl:/run -w /run erf:1.00 \
Docs/sphinx_doc/containers.rst:  /app/erf/ERF-24.06/MyBuild/Exec/ABL/erf_abl inputs_smagorinsky amrex.use_gpu_aware_mpi=0
Docs/sphinx_doc/containers.rst:      export CUDA_VISIBLE_DEVICES=$((3-$SLURM_LOCALID))
Docs/sphinx_doc/containers.rst:* ``--gpu`` enables NVIDIA GPU support
Docs/sphinx_doc/containers.rst:  #SBATCH --constraint=gpu
Docs/sphinx_doc/containers.rst:  #SBATCH --module=mpich,gpu  # for shifter, not podman-hpc; CPU-only MPI
Docs/sphinx_doc/containers.rst:  ##SBATCH --module=cuda-mpich  # for shifter, not podman-hpc; GPU-Aware MPI
Docs/sphinx_doc/containers.rst:  srun -N 1 -n 4 -c 32 --ntasks-per-node=4 --gpus-per-node=4 ./device_wrapper \
Docs/sphinx_doc/containers.rst:  /app/erf/ERF-24.06/MyBuild/Exec/ABL/erf_abl inputs_smagorinsky amrex.use_gpu_aware_mpi=0
Docs/sphinx_doc/RegressionTests.rst:Results from the nightly GPU tests can be found here: `GPU tests`_
Docs/sphinx_doc/RegressionTests.rst:.. _`GPU tests`: https://ccse.lbl.gov/pub/GpuRegressionTesting/ERF
Docs/sphinx_doc/Discretizations.rst:Sauer, J. A., & Muñoz-Esparza, D. (2020). The FastEddy® resident-GPU accelerated large-eddy simulation framework: Model formulation, dynamical-core validation and performance benchmarks. Journal of Advances in Modeling Earth Systems, 12, e2020MS002100. doi:10.1029/2020MS002100
Docs/sphinx_doc/Discretizations.rst:Ref: Muñoz-Esparza, D., Sauer, J. A., Jensen, A. A., Xue, L., & Grabowski, W. W. (2022). The FastEddy® resident-GPU accelerated large-eddy
Docs/main.dox:  of machines, from laptops to multicore CPU and hybrid CPU-GPU systems.
Build/cmake_cuda_SpecifyToolKit.sh:      -DERF_ENABLE_CUDA:BOOL=ON \
Build/cmake_cuda_SpecifyToolKit.sh:      -DCUDAToolkit_ROOT=/usr/local/cuda-12.4/bin \
Build/erf_containerfile:FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04
Build/erf_containerfile:  -DCMAKE_CUDA_ARCHITECTURES=80 \
Build/erf_containerfile:  -DERF_ENABLE_CUDA=ON \
Build/cmake_cuda.sh:      -DERF_ENABLE_CUDA:BOOL=ON \
Build/cmake_hip_crusher.sh:module load PrgEnv-gnu craype-accel-amd-gfx90a cray-mpich rocm cmake ccache ninja git
Exec/ABL/ERF_prob.cpp:  ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/ABL/ERF_prob.cpp:  ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/ABL/ERF_prob.cpp:  ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/ABL/ERF_prob.cpp:  ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/ABL/GNUmakefile:USE_CUDA = FALSE
Exec/ABL_input_sounding/ERF_prob.cpp:  ParallelFor(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/ABL_input_sounding/ERF_prob.cpp:  ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/ABL_input_sounding/ERF_prob.cpp:  ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/ABL_input_sounding/ERF_prob.cpp:  ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept
Exec/ABL_input_sounding/GNUmakefile:USE_CUDA = FALSE
Exec/SpongeTest/ERF_prob.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/SpongeTest/ERF_prob.cpp:    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/SpongeTest/ERF_prob.cpp:    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/SpongeTest/ERF_prob.cpp:    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/SpongeTest/ERF_prob.cpp:    amrex::Gpu::streamSynchronize();
Exec/SpongeTest/ERF_prob.cpp:        for ( amrex::MFIter mfi(z_phys_nd,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Exec/SpongeTest/ERF_prob.cpp:            ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int)
Exec/SpongeTest/GNUmakefile:USE_CUDA  = FALSE
Exec/DryRegTests/ScalarAdvDiff/ERF_prob.cpp:  ParallelFor(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/ScalarAdvDiff/ERF_prob.cpp:  ParallelFor(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/ScalarAdvDiff/ERF_prob.cpp:  ParallelFor(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/ScalarAdvDiff/ERF_prob.cpp:  ParallelFor(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/ScalarAdvDiff/ERF_prob.cpp:AMREX_GPU_DEVICE
Exec/DryRegTests/ScalarAdvDiff/ERF_prob.cpp:     const GpuArray<Real,AMREX_SPACEDIM> /*dx*/,
Exec/DryRegTests/ScalarAdvDiff/prob.cpp.convergence:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Exec/DryRegTests/ScalarAdvDiff/prob.cpp.convergence:    for ( MFIter mfi(rho_hse,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/DryRegTests/ScalarAdvDiff/prob.cpp.convergence:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/ScalarAdvDiff/prob.cpp.convergence:  amrex::ParallelFor(bx, [=, parms=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/ScalarAdvDiff/prob.cpp.convergence:  amrex::ParallelFor(xbx, [=, parms=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/ScalarAdvDiff/prob.cpp.convergence:  amrex::ParallelFor(ybx, [=, parms=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/ScalarAdvDiff/prob.cpp.convergence:  amrex::ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/ScalarAdvDiff/prob.cpp.convergence:AMREX_GPU_DEVICE
Exec/DryRegTests/ScalarAdvDiff/prob.cpp.convergence:     const GpuArray<Real,AMREX_SPACEDIM> /*dx*/,
Exec/DryRegTests/ScalarAdvDiff/GNUmakefile:USE_CUDA = FALSE
Exec/DryRegTests/Couette_Poiseuille/ERF_prob.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/Couette_Poiseuille/ERF_prob.cpp:        ParallelFor(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/DryRegTests/Couette_Poiseuille/ERF_prob.cpp:        ParallelFor(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/DryRegTests/Couette_Poiseuille/ERF_prob.cpp:        ParallelFor(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/DryRegTests/Couette_Poiseuille/ERF_prob.cpp:        ParallelFor(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/Couette_Poiseuille/ERF_prob.cpp:        ParallelFor(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/Couette_Poiseuille/ERF_prob.cpp:        ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/Couette_Poiseuille/GNUmakefile:USE_CUDA = FALSE
Exec/DryRegTests/Terrain2d_Cylinder/ERF_prob.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/Terrain2d_Cylinder/ERF_prob.cpp:    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/Terrain2d_Cylinder/ERF_prob.cpp:    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/Terrain2d_Cylinder/ERF_prob.cpp:    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/Terrain2d_Cylinder/ERF_prob.cpp:    amrex::Gpu::streamSynchronize();
Exec/DryRegTests/Terrain2d_Cylinder/ERF_prob.cpp:        for ( amrex::MFIter mfi(z_phys_nd,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Exec/DryRegTests/Terrain2d_Cylinder/ERF_prob.cpp:            ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int)
Exec/DryRegTests/Terrain2d_Cylinder/GNUmakefile:USE_CUDA  = FALSE
Exec/DryRegTests/EkmanSpiral/ERF_prob.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/EkmanSpiral/ERF_prob.cpp:        ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/DryRegTests/EkmanSpiral/ERF_prob.cpp:        ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/DryRegTests/EkmanSpiral/ERF_prob.cpp:        ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/DryRegTests/EkmanSpiral/GNUmakefile:USE_CUDA = FALSE
Exec/DryRegTests/TaylorGreenVortex/ERF_prob.cpp:  ParallelFor(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/DryRegTests/TaylorGreenVortex/ERF_prob.cpp:  ParallelFor(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/TaylorGreenVortex/ERF_prob.cpp:  ParallelFor(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/TaylorGreenVortex/ERF_prob.cpp:  ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/TaylorGreenVortex/GNUmakefile:USE_CUDA = FALSE
Exec/DryRegTests/WitchOfAgnesi/ERF_prob.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/WitchOfAgnesi/ERF_prob.cpp:    ParallelFor(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/WitchOfAgnesi/ERF_prob.cpp:    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/WitchOfAgnesi/ERF_prob.cpp:    amrex::GpuArray<Real, AMREX_SPACEDIM> dxInv;
Exec/DryRegTests/WitchOfAgnesi/ERF_prob.cpp:    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/WitchOfAgnesi/ERF_prob.cpp:    amrex::Gpu::streamSynchronize();
Exec/DryRegTests/WitchOfAgnesi/ERF_prob.cpp:        for ( amrex::MFIter mfi(z_phys_nd,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Exec/DryRegTests/WitchOfAgnesi/ERF_prob.cpp:            ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int)
Exec/DryRegTests/WitchOfAgnesi/GNUmakefile:USE_CUDA  = FALSE
Exec/DryRegTests/TurbulentInflow/ERF_prob.cpp:  ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/DryRegTests/TurbulentInflow/ERF_prob.cpp:  ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/DryRegTests/TurbulentInflow/ERF_prob.cpp:  ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/DryRegTests/TurbulentInflow/ERF_prob.cpp:  ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/DryRegTests/TurbulentInflow/GNUmakefile:USE_CUDA = FALSE
Exec/DryRegTests/Terrain3d_Hemisphere/ERF_prob.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/Terrain3d_Hemisphere/ERF_prob.cpp:    ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept
Exec/DryRegTests/Terrain3d_Hemisphere/ERF_prob.cpp:    ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept
Exec/DryRegTests/Terrain3d_Hemisphere/ERF_prob.cpp:    amrex::GpuArray<Real, AMREX_SPACEDIM> dxInv;
Exec/DryRegTests/Terrain3d_Hemisphere/ERF_prob.cpp:    ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept
Exec/DryRegTests/Terrain3d_Hemisphere/ERF_prob.cpp:    amrex::Gpu::streamSynchronize();
Exec/DryRegTests/Terrain3d_Hemisphere/ERF_prob.cpp:        for ( amrex::MFIter mfi(z_phys_nd,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Exec/DryRegTests/Terrain3d_Hemisphere/ERF_prob.cpp:            ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int)
Exec/DryRegTests/Terrain3d_Hemisphere/GNUmakefile:USE_CUDA  = FALSE
Exec/DryRegTests/WPS_Test/GNUmakefile:USE_CUDA = FALSE
Exec/DryRegTests/DensityCurrent/ERF_prob.cpp:      ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/DensityCurrent/ERF_prob.cpp:      ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/DensityCurrent/ERF_prob.cpp:  ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/DensityCurrent/ERF_prob.cpp:  ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/DensityCurrent/ERF_prob.cpp:  ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/DensityCurrent/ERF_prob.cpp:  amrex::Gpu::streamSynchronize();
Exec/DryRegTests/DensityCurrent/GNUmakefile:USE_CUDA = FALSE
Exec/DryRegTests/ParticlesOverWoA/ERF_prob.cpp:  ParallelFor(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/ParticlesOverWoA/ERF_prob.cpp:  ParallelFor(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/ParticlesOverWoA/ERF_prob.cpp:  ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/ParticlesOverWoA/ERF_prob.cpp:  amrex::GpuArray<Real, AMREX_SPACEDIM> dxInv;
Exec/DryRegTests/ParticlesOverWoA/ERF_prob.cpp:  ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/ParticlesOverWoA/ERF_prob.cpp:  amrex::Gpu::streamSynchronize();
Exec/DryRegTests/ParticlesOverWoA/ERF_prob.cpp:        for ( amrex::MFIter mfi(z_phys_nd,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Exec/DryRegTests/ParticlesOverWoA/ERF_prob.cpp:            ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int)
Exec/DryRegTests/ParticlesOverWoA/GNUmakefile:USE_CUDA  = FALSE
Exec/DryRegTests/StokesSecondProblem/ERF_prob.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/StokesSecondProblem/ERF_prob.cpp:    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/StokesSecondProblem/ERF_prob.cpp:    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/StokesSecondProblem/ERF_prob.cpp:    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/StokesSecondProblem/ERF_prob.cpp:    amrex::Gpu::streamSynchronize();
Exec/DryRegTests/StokesSecondProblem/GNUmakefile:USE_CUDA  = FALSE
Exec/DryRegTests/IsentropicVortex/ERF_prob.cpp:AMREX_GPU_DEVICE
Exec/DryRegTests/IsentropicVortex/ERF_prob.cpp:  ParallelFor(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/IsentropicVortex/ERF_prob.cpp:  ParallelFor(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/IsentropicVortex/ERF_prob.cpp:  ParallelFor(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/IsentropicVortex/ERF_prob.cpp:  ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DryRegTests/IsentropicVortex/ERF_prob.cpp:AMREX_GPU_DEVICE
Exec/DryRegTests/IsentropicVortex/ERF_prob.cpp:     const GpuArray<Real,AMREX_SPACEDIM> /*dx*/,
Exec/DryRegTests/IsentropicVortex/GNUmakefile:USE_CUDA = FALSE
Exec/LLJ/GNUmakefile:USE_CUDA = FALSE
Exec/WindFarmTests/EWP/ERF_prob.cpp:  ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/EWP/ERF_prob.cpp:  ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/EWP/ERF_prob.cpp:  ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/EWP/ERF_prob.cpp:  ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/EWP/GNUmakefile:USE_CUDA = FALSE
Exec/WindFarmTests/GeneralActuatorDisk/ERF_prob.cpp:  ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/GeneralActuatorDisk/ERF_prob.cpp:  ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/GeneralActuatorDisk/ERF_prob.cpp:  ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/GeneralActuatorDisk/ERF_prob.cpp:  ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/SimpleActuatorDisk/ERF_prob.cpp:  ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/SimpleActuatorDisk/ERF_prob.cpp:  ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/SimpleActuatorDisk/ERF_prob.cpp:  ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/SimpleActuatorDisk/ERF_prob.cpp:  ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/SimpleActuatorDisk/GNUmakefile:USE_CUDA = FALSE
Exec/WindFarmTests/AWAKEN/ERF_prob.cpp:  ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/AWAKEN/ERF_prob.cpp:  ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/AWAKEN/ERF_prob.cpp:  ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/AWAKEN/ERF_prob.cpp:  ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/WindFarmTests/AWAKEN/GNUmakefile:USE_CUDA = FALSE
Exec/DevTests/TemperatureSource/ERF_prob.H:        amrex::Gpu::DeviceVector<amrex::Real>& d_src,
Exec/DevTests/TemperatureSource/ERF_prob.cpp:  ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/DevTests/TemperatureSource/ERF_prob.cpp:  ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/DevTests/TemperatureSource/ERF_prob.cpp:  ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/DevTests/TemperatureSource/ERF_prob.cpp:  ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/DevTests/TemperatureSource/ERF_prob.cpp:    amrex::Gpu::DeviceVector<amrex::Real>& d_src,
Exec/DevTests/TemperatureSource/ERF_prob.cpp:    amrex::Gpu::copy(amrex::Gpu::hostToDevice, src.begin(), src.end(), d_src.begin());
Exec/DevTests/TemperatureSource/GNUmakefile:USE_CUDA = FALSE
Exec/DevTests/FlowInABox/ERF_prob.cpp:  ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept
Exec/DevTests/FlowInABox/ERF_prob.cpp:  ParallelFor(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/DevTests/FlowInABox/ERF_prob.cpp:  ParallelFor(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/DevTests/FlowInABox/ERF_prob.cpp:  ParallelFor(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/DevTests/FlowInABox/GNUmakefile:USE_CUDA = FALSE
Exec/DevTests/NoahMP/GNUmakefile:USE_CUDA = FALSE
Exec/DevTests/EB_Test/ERF_prob.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/EB_Test/ERF_prob.cpp:    ParallelFor(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/EB_Test/ERF_prob.cpp:    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/EB_Test/ERF_prob.cpp:    GpuArray<Real, AMREX_SPACEDIM> dxInv;
Exec/DevTests/EB_Test/ERF_prob.cpp:    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/EB_Test/ERF_prob.cpp:    amrex::Gpu::streamSynchronize();
Exec/DevTests/EB_Test/ERF_prob.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int)
Exec/DevTests/EB_Test/GNUmakefile:USE_CUDA  = FALSE
Exec/DevTests/TropicalCyclone/ERF_prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/TropicalCyclone/ERF_prob.cpp:    amrex::ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/TropicalCyclone/ERF_prob.cpp:    amrex::ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/TropicalCyclone/GNUmakefile:USE_CUDA = FALSE
Exec/DevTests/ABL_with_WW3/ERF_prob.cpp:  ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/DevTests/ABL_with_WW3/ERF_prob.cpp:  ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/DevTests/ABL_with_WW3/ERF_prob.cpp:  ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/DevTests/ABL_with_WW3/ERF_prob.cpp:  ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Exec/DevTests/ABL_with_WW3/GNUmakefile:USE_CUDA = FALSE
Exec/DevTests/LandSurfaceModel/GNUmakefile:USE_CUDA = FALSE
Exec/DevTests/MovingTerrain/ERF_prob.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/MovingTerrain/ERF_prob.cpp:  amrex::ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/MovingTerrain/ERF_prob.cpp:  amrex::ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/MovingTerrain/ERF_prob.cpp:  amrex::ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/MovingTerrain/ERF_prob.cpp:  amrex::Gpu::streamSynchronize();
Exec/DevTests/MovingTerrain/ERF_prob.cpp:        for (MFIter mfi(z_phys_nd,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/DevTests/MovingTerrain/ERF_prob.cpp:            ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int)
Exec/DevTests/MovingTerrain/GNUmakefile:USE_CUDA = FALSE
Exec/DevTests/Radiation/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Exec/DevTests/Radiation/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Exec/DevTests/Radiation/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Exec/DevTests/Radiation/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Exec/DevTests/Radiation/ERF_prob.cpp:    ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept
Exec/DevTests/Radiation/ERF_prob.cpp:    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/Radiation/ERF_prob.cpp:    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/DevTests/Radiation/ERF_prob.cpp:    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/DevTests/Radiation/ERF_prob.cpp:    amrex::Gpu::streamSynchronize();
Exec/DevTests/Radiation/GNUmakefile:USE_CUDA = FALSE
Exec/DevTests/MetGrid/GNUmakefile:USE_CUDA = FALSE
Exec/DevTests/MultiBlock/ERF_prob.cpp:  amrex::ParallelFor(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/MultiBlock/ERF_prob.cpp:  amrex::ParallelFor(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/MultiBlock/ERF_prob.cpp:  amrex::ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/MultiBlock/ERF_prob.cpp:  amrex::ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DevTests/MultiBlock/ERF_prob.cpp:  amrex::Gpu::streamSynchronize();
Exec/DevTests/MultiBlock/GNUmakefile:USE_CUDA = FALSE
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:            ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int) {
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:        amrex::Gpu::DeviceVector<Real> d_r(khi+2);
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_r.begin(), h_r.end(), d_r.begin());
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:          for ( MFIter mfi(rho_hse,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:              ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:   amrex::Gpu::DeviceVector<Real> d_r(khi+2);
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:   amrex::Gpu::DeviceVector<Real> d_p(khi+2);
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:   amrex::Gpu::DeviceVector<Real> d_t(khi+2);
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:   amrex::Gpu::DeviceVector<Real> d_q_v(khi+2);
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:   amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_r.begin(), h_r.end(), d_r.begin());
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:   amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_p.begin(), h_p.end(), d_p.begin());
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:   amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_t.begin(), h_t.end(), d_t.begin());
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:   amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_q_v.begin(), h_q_v.end(), d_q_v.begin());
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:  ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:  ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:  ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:  ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/MoistRegTests/SquallLine_2D/ERF_prob.cpp:  Gpu::streamSynchronize();
Exec/MoistRegTests/SquallLine_2D/GNUmakefile:USE_CUDA = FALSE
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:            ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int) {
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:        amrex::Gpu::DeviceVector<Real> d_r(khi+2);
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_r.begin(), h_r.end(), d_r.begin());
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:          for ( MFIter mfi(rho_hse,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:              ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:   amrex::Gpu::DeviceVector<Real> d_r(khi+2);
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:   amrex::Gpu::DeviceVector<Real> d_p(khi+2);
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:   amrex::Gpu::DeviceVector<Real> d_t(khi+2);
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:   amrex::Gpu::DeviceVector<Real> d_q_v(khi+2);
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:   amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_r.begin(), h_r.end(), d_r.begin());
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:   amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_p.begin(), h_p.end(), d_p.begin());
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:   amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_t.begin(), h_t.end(), d_t.begin());
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:   amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_q_v.begin(), h_q_v.end(), d_q_v.begin());
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:  ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:  ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:  ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:  ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/MoistRegTests/SuperCell_3D/ERF_prob.cpp:  Gpu::streamSynchronize();
Exec/MoistRegTests/SuperCell_3D/GNUmakefile:USE_CUDA = FALSE
Exec/MoistRegTests/Bubble/ERF_prob.H:    AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.H:    AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.H:    AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.H:    AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.H:    AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.H:    AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Exec/MoistRegTests/Bubble/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.cpp:AMREX_GPU_HOST_DEVICE
Exec/MoistRegTests/Bubble/ERF_prob.cpp:        Gpu::DeviceVector<Real> d_r(khi+2);
Exec/MoistRegTests/Bubble/ERF_prob.cpp:        Gpu::copyAsync(Gpu::hostToDevice, h_r.begin(), h_r.end(), d_r.begin());
Exec/MoistRegTests/Bubble/ERF_prob.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/MoistRegTests/Bubble/ERF_prob.cpp:          for ( MFIter mfi(rho_hse,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/MoistRegTests/Bubble/ERF_prob.cpp:              ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/MoistRegTests/Bubble/ERF_prob.cpp:            ParallelFor(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/MoistRegTests/Bubble/ERF_prob.cpp:            Gpu::DeviceVector<Real> d_r(khi+2);
Exec/MoistRegTests/Bubble/ERF_prob.cpp:            Gpu::DeviceVector<Real> d_p(khi+2);
Exec/MoistRegTests/Bubble/ERF_prob.cpp:            Gpu::DeviceVector<Real> d_t(khi+2);
Exec/MoistRegTests/Bubble/ERF_prob.cpp:            Gpu::DeviceVector<Real> d_q_v(khi+2);
Exec/MoistRegTests/Bubble/ERF_prob.cpp:            Gpu::copyAsync(Gpu::hostToDevice, h_r.begin(), h_r.end(), d_r.begin());
Exec/MoistRegTests/Bubble/ERF_prob.cpp:            Gpu::copyAsync(Gpu::hostToDevice, h_p.begin(), h_p.end(), d_p.begin());
Exec/MoistRegTests/Bubble/ERF_prob.cpp:            Gpu::copyAsync(Gpu::hostToDevice, h_t.begin(), h_t.end(), d_t.begin());
Exec/MoistRegTests/Bubble/ERF_prob.cpp:            Gpu::copyAsync(Gpu::hostToDevice, h_q_v.begin(), h_q_v.end(), d_q_v.begin());
Exec/MoistRegTests/Bubble/ERF_prob.cpp:            ParallelFor(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k)
Exec/MoistRegTests/Bubble/ERF_prob.cpp:            ParallelFor(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/MoistRegTests/Bubble/ERF_prob.cpp:    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/MoistRegTests/Bubble/ERF_prob.cpp:    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/MoistRegTests/Bubble/ERF_prob.cpp:    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/MoistRegTests/Bubble/ERF_prob.cpp:    Gpu::streamSynchronize();
Exec/MoistRegTests/Bubble/GNUmakefile:USE_CUDA  = FALSE
Exec/MoistRegTests/Bomex/ERF_prob.H:        amrex::Gpu::DeviceVector<amrex::Real>& d_src,
Exec/MoistRegTests/Bomex/ERF_prob.H:        amrex::Gpu::DeviceVector<amrex::Real>& d_qsrc,
Exec/MoistRegTests/Bomex/ERF_prob.H:        amrex::Gpu::DeviceVector<amrex::Real>& d_wbar,
Exec/MoistRegTests/Bomex/ERF_prob.H:        amrex::Gpu::DeviceVector<amrex::Real>& d_u_geos,
Exec/MoistRegTests/Bomex/ERF_prob.H:        amrex::Gpu::DeviceVector<amrex::Real>& d_v_geos,
Exec/MoistRegTests/Bomex/ERF_prob.cpp:    ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
Exec/MoistRegTests/Bomex/ERF_prob.cpp:    ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
Exec/MoistRegTests/Bomex/ERF_prob.cpp:  ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
Exec/MoistRegTests/Bomex/ERF_prob.cpp:  ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
Exec/MoistRegTests/Bomex/ERF_prob.cpp:                                  Gpu::DeviceVector<Real>& d_src,
Exec/MoistRegTests/Bomex/ERF_prob.cpp:    amrex::Gpu::copy(amrex::Gpu::hostToDevice, src.begin(), src.end(), d_src.begin());
Exec/MoistRegTests/Bomex/ERF_prob.cpp:                               Gpu::DeviceVector<Real>& d_qsrc,
Exec/MoistRegTests/Bomex/ERF_prob.cpp:    amrex::Gpu::copy(amrex::Gpu::hostToDevice, qsrc.begin(), qsrc.end(), d_qsrc.begin());
Exec/MoistRegTests/Bomex/ERF_prob.cpp:                              Gpu::DeviceVector<Real>& d_wbar,
Exec/MoistRegTests/Bomex/ERF_prob.cpp:    amrex::Gpu::copy(amrex::Gpu::hostToDevice, wbar.begin(), wbar.end(), d_wbar.begin());
Exec/MoistRegTests/Bomex/ERF_prob.cpp:                                     Gpu::DeviceVector<Real>& d_u_geos,
Exec/MoistRegTests/Bomex/ERF_prob.cpp:                                     Gpu::DeviceVector<Real>& d_v_geos,
Exec/MoistRegTests/Bomex/ERF_prob.cpp:    amrex::Gpu::copy(amrex::Gpu::hostToDevice, u_geos.begin(), u_geos.end(), d_u_geos.begin());
Exec/MoistRegTests/Bomex/ERF_prob.cpp:    amrex::Gpu::copy(amrex::Gpu::hostToDevice, v_geos.begin(), v_geos.end(), d_v_geos.begin());
Exec/MoistRegTests/Bomex/GNUmakefile:USE_CUDA = FALSE
paper/paper.md:or CUDA, HIP, or SYCL on GPU-accelerated systems.
paper/paper.md:ability to use GPU acceleration, which limits their ability to 
paper/paper.md:whether CPU-only or GPU-accelerated.  In addition, ERF is based on AMReX,
CMakeLists.txt:option(ERF_ENABLE_CUDA "Enable CUDA" OFF)
CMakeLists.txt:if(ERF_ENABLE_CUDA)
CMakeLists.txt:  enable_language(CUDA)
CMakeLists.txt:  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
CMakeLists.txt:    message(FATAL_ERROR "Your nvcc version is ${CMAKE_CUDA_COMPILER_VERSION} which is unsupported."
CMakeLists.txt:      "Please use CUDA toolkit version 11.0 or newer.")
CMakeLists.txt:#    if (ERF_ENABLE_CUDA AND (CMAKE_VERSION VERSION_LESS 3.20))
CMakeLists.txt:#      include(AMReX_SetupCUDA)
CMakeLists.txt:    if (ERF_ENABLE_CUDA)
CMakeLists.txt:      list(APPEND AMREX_COMPONENTS "CUDA")
CMakeLists.txt:    if (ERF_ENABLE_ROCM)
CMakeLists.txt:   # YAKL_ARCH can be CUDA, HIP, SYCL, OPENMP45, or empty
CMakeLists.txt:   if(ERF_ENABLE_CUDA)
CMakeLists.txt:      set(YAKL_ARCH "CUDA")
CMakeLists.txt:      # CUDA_FLAGS is set the same as ERF_CUDA_FLAGS
CMakeLists.txt:      string(APPEND YAKL_CUDA_FLAGS " -arch sm_70")
CMakeLists.txt:      if(ENABLE_CUDA_FASTMATH)
CMakeLists.txt:        string(APPEND YAKL_CUDA_FLAGS " --use_fast_math")
CMakeLists.txt:      set_cuda_architectures(AMReX_CUDA_ARCH)
CMakeLists.txt:      string(APPEND YAKL_HIP_FLAGS " -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip")
CMakeLists.txt:if (ERF_GPU_BACKEND STREQUAL "CUDA")
CMakeLists.txt:   setup_target_for_cuda_compilation(erf_api)
Source/SourceTerms/ERF_ApplySpongeZoneBCs_ReadFromFile.cpp:    ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
Source/SourceTerms/ERF_ApplySpongeZoneBCs_ReadFromFile.cpp:    ParallelFor(tby, [=] AMREX_GPU_DEVICE(int i, int j, int k)
Source/SourceTerms/ERF_moist_set_rhs.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_moist_set_rhs.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_moist_set_rhs.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_moist_set_rhs.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_moist_set_rhs.cpp:    ParallelFor(bx_xlo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_moist_set_rhs.cpp:    ParallelFor(bx_xhi, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_moist_set_rhs.cpp:    ParallelFor(bx_ylo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_moist_set_rhs.cpp:    ParallelFor(bx_yhi, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:#include <AMReX_GpuContainers.H>
Source/SourceTerms/ERF_make_sources.cpp:    const GpuArray<Real, AMREX_SPACEDIM> dxInv = geom.InvCellSizeArray();
Source/SourceTerms/ERF_make_sources.cpp:        Gpu::HostVector<    Real> r_plane_h(ncell);
Source/SourceTerms/ERF_make_sources.cpp:        Gpu::DeviceVector<  Real> r_plane_d(ncell);
Source/SourceTerms/ERF_make_sources.cpp:        Gpu::copyAsync(Gpu::hostToDevice, r_plane_h.begin(), r_plane_h.end(), r_plane_d.begin());
Source/SourceTerms/ERF_make_sources.cpp:        ParallelFor(ncell, [=] AMREX_GPU_DEVICE (int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:        Gpu::HostVector<    Real> t_plane_h(ncell);
Source/SourceTerms/ERF_make_sources.cpp:        Gpu::DeviceVector<  Real> t_plane_d(ncell);
Source/SourceTerms/ERF_make_sources.cpp:        Gpu::copyAsync(Gpu::hostToDevice, t_plane_h.begin(), t_plane_h.end(), t_plane_d.begin());
Source/SourceTerms/ERF_make_sources.cpp:        ParallelFor(ncell, [=] AMREX_GPU_DEVICE (int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:            Gpu::HostVector<  Real> qv_plane_h(ncell), qc_plane_h(ncell);
Source/SourceTerms/ERF_make_sources.cpp:            Gpu::DeviceVector<Real> qv_plane_d(ncell), qc_plane_d(ncell);
Source/SourceTerms/ERF_make_sources.cpp:            Gpu::copyAsync(Gpu::hostToDevice, qv_plane_h.begin(), qv_plane_h.end(), qv_plane_d.begin());
Source/SourceTerms/ERF_make_sources.cpp:            Gpu::copyAsync(Gpu::hostToDevice, qc_plane_h.begin(), qc_plane_h.end(), qc_plane_d.begin());
Source/SourceTerms/ERF_make_sources.cpp:            ParallelFor(ncell, [=] AMREX_GPU_DEVICE (int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/SourceTerms/ERF_make_sources.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_sources.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_buoyancy_utils.H:AMREX_GPU_DEVICE
Source/SourceTerms/ERF_buoyancy_utils.H:                        amrex::Real const& grav_gpu,
Source/SourceTerms/ERF_buoyancy_utils.H:    return (-grav_gpu * (theta_d_wface - theta_d_0_wface) / theta_d_0_wface * 0.5 * (r0_arr(i,j,k) + r0_arr(i,j,k-1)));
Source/SourceTerms/ERF_buoyancy_utils.H:AMREX_GPU_DEVICE
Source/SourceTerms/ERF_buoyancy_utils.H:                          amrex::Real const& grav_gpu,
Source/SourceTerms/ERF_buoyancy_utils.H:    return (-r0_q_avg * grav_gpu);
Source/SourceTerms/ERF_buoyancy_utils.H:AMREX_GPU_DEVICE
Source/SourceTerms/ERF_buoyancy_utils.H:                          amrex::Real const& grav_gpu,
Source/SourceTerms/ERF_buoyancy_utils.H:    return (-grav_gpu * (theta_v_wface - theta_v_0_wface) / theta_v_0_wface * 0.5 * (r0_arr(i,j,k) + r0_arr(i,j,k-1)));
Source/SourceTerms/ERF_buoyancy_utils.H:AMREX_GPU_DEVICE
Source/SourceTerms/ERF_buoyancy_utils.H:                      amrex::Real const& grav_gpu,
Source/SourceTerms/ERF_buoyancy_utils.H:    return (-r0_q_avg * grav_gpu);
Source/SourceTerms/ERF_buoyancy_utils.H:AMREX_GPU_DEVICE
Source/SourceTerms/ERF_buoyancy_utils.H:                amrex::Real const& grav_gpu,
Source/SourceTerms/ERF_buoyancy_utils.H:    return( grav_gpu * amrex::Real(0.5) * ( rhop_hi + rhop_lo ) );
Source/SourceTerms/ERF_buoyancy_utils.H:AMREX_GPU_DEVICE
Source/SourceTerms/ERF_buoyancy_utils.H:                amrex::Real const& grav_gpu,
Source/SourceTerms/ERF_buoyancy_utils.H:    return (-qavg * r0avg * grav_gpu);
Source/SourceTerms/ERF_buoyancy_utils.H:AMREX_GPU_DEVICE
Source/SourceTerms/ERF_buoyancy_utils.H:                amrex::Real const& grav_gpu,
Source/SourceTerms/ERF_buoyancy_utils.H:    return ( -qavg * r0avg * grav_gpu );
Source/SourceTerms/ERF_buoyancy_utils.H:AMREX_GPU_DEVICE
Source/SourceTerms/ERF_buoyancy_utils.H:                amrex::Real const& grav_gpu,
Source/SourceTerms/ERF_buoyancy_utils.H:    return (-qavg * r0avg * grav_gpu);
Source/SourceTerms/ERF_ApplySpongeZoneBCs.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/SourceTerms/ERF_ApplySpongeZoneBCs.cpp:    ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
Source/SourceTerms/ERF_ApplySpongeZoneBCs.cpp:    ParallelFor(tby, [=] AMREX_GPU_DEVICE(int i, int j, int k)
Source/SourceTerms/ERF_ApplySpongeZoneBCs.cpp:    ParallelFor(tbz, [=] AMREX_GPU_DEVICE(int i, int j, int k)
Source/SourceTerms/ERF_NumericalDiffusion.H:AMREX_GPU_DEVICE
Source/SourceTerms/ERF_NumericalDiffusion.cpp:    ParallelFor(bx, num_comp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int m) noexcept
Source/SourceTerms/ERF_NumericalDiffusion.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_NumericalDiffusion.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_buoyancy.cpp:#include <AMReX_GpuContainers.H>
Source/SourceTerms/ERF_make_buoyancy.cpp:    const GpuArray<Real,AMREX_SPACEDIM> grav_gpu{grav[0], grav[1], grav[2]};
Source/SourceTerms/ERF_make_buoyancy.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/SourceTerms/ERF_make_buoyancy.cpp:        for ( MFIter mfi(buoyancy,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/SourceTerms/ERF_make_buoyancy.cpp:                ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_buoyancy.cpp:                                                                   grav_gpu[2],
Source/SourceTerms/ERF_make_buoyancy.cpp:                ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_buoyancy.cpp:                                                                     grav_gpu[2],rv_over_rd,
Source/SourceTerms/ERF_make_buoyancy.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/SourceTerms/ERF_make_buoyancy.cpp:                for ( MFIter mfi(buoyancy,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/SourceTerms/ERF_make_buoyancy.cpp:                    ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_buoyancy.cpp:                        buoyancy_fab(i, j, k) = buoyancy_type1(i,j,k,n_q_dry,grav_gpu[2],r0_arr,cell_data);
Source/SourceTerms/ERF_make_buoyancy.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/SourceTerms/ERF_make_buoyancy.cpp:                for ( MFIter mfi(buoyancy,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/SourceTerms/ERF_make_buoyancy.cpp:                    ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_buoyancy.cpp:                                                                     grav_gpu[2],rd_over_cp,
Source/SourceTerms/ERF_make_buoyancy.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/SourceTerms/ERF_make_buoyancy.cpp:            for ( MFIter mfi(buoyancy,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/SourceTerms/ERF_make_buoyancy.cpp:                ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_buoyancy.cpp:                                                           grav_gpu[2],r0_arr,cell_data);
Source/SourceTerms/ERF_make_buoyancy.cpp:            Gpu::HostVector  <Real> rho_h(ncell), theta_h(ncell);
Source/SourceTerms/ERF_make_buoyancy.cpp:            Gpu::DeviceVector<Real> rho_d(ncell), theta_d(ncell);
Source/SourceTerms/ERF_make_buoyancy.cpp:            Gpu::copyAsync(Gpu::hostToDevice, rho_h.begin(), rho_h.end(), rho_d.begin());
Source/SourceTerms/ERF_make_buoyancy.cpp:            Gpu::copyAsync(Gpu::hostToDevice, theta_h.begin(), theta_h.end(), theta_d.begin());
Source/SourceTerms/ERF_make_buoyancy.cpp:            Gpu::HostVector  <Real> qv_h(ncell)    , qc_h(ncell)    , qp_h(ncell);
Source/SourceTerms/ERF_make_buoyancy.cpp:            Gpu::DeviceVector<Real> qv_d(ncell,0.0), qc_d(ncell,0.0), qp_d(ncell,0.0);
Source/SourceTerms/ERF_make_buoyancy.cpp:               Gpu::copyAsync(Gpu::hostToDevice,  qv_h.begin(), qv_h.end(), qv_d.begin());
Source/SourceTerms/ERF_make_buoyancy.cpp:                Gpu::copyAsync(Gpu::hostToDevice,  qc_h.begin(), qc_h.end(), qc_d.begin());
Source/SourceTerms/ERF_make_buoyancy.cpp:                Gpu::copyAsync(Gpu::hostToDevice,  qp_h.begin(), qp_h.end(), qp_d.begin());
Source/SourceTerms/ERF_make_buoyancy.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/SourceTerms/ERF_make_buoyancy.cpp:                for ( MFIter mfi(buoyancy,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/SourceTerms/ERF_make_buoyancy.cpp:                        ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_buoyancy.cpp:                            buoyancy_fab(i, j, k) = buoyancy_type2(i,j,k,n_qstate,grav_gpu[2],
Source/SourceTerms/ERF_make_buoyancy.cpp:                        ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_buoyancy.cpp:                            buoyancy_fab(i, j, k) = buoyancy_type4(i,j,k,n_qstate,grav_gpu[2],
Source/SourceTerms/ERF_make_buoyancy.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/SourceTerms/ERF_make_buoyancy.cpp:                for ( MFIter mfi(buoyancy,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/SourceTerms/ERF_make_buoyancy.cpp:                    ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_buoyancy.cpp:                        buoyancy_fab(i, j, k) = buoyancy_type3(i,j,k,n_qstate,grav_gpu[2],
Source/SourceTerms/ERF_add_thin_body_sources.cpp:#include <AMReX_GpuContainers.H>
Source/SourceTerms/ERF_add_thin_body_sources.cpp:#ifndef AMREX_USE_GPU
Source/SourceTerms/ERF_add_thin_body_sources.cpp:                ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/SourceTerms/ERF_add_thin_body_sources.cpp:#ifndef AMREX_USE_GPU
Source/SourceTerms/ERF_add_thin_body_sources.cpp:                ParallelFor(tby, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/SourceTerms/ERF_add_thin_body_sources.cpp:#ifndef AMREX_USE_GPU
Source/SourceTerms/ERF_add_thin_body_sources.cpp:                ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/SourceTerms/ERF_make_mom_sources.cpp:#include <AMReX_GpuContainers.H>
Source/SourceTerms/ERF_make_mom_sources.cpp:    const GpuArray<Real, AMREX_SPACEDIM> dxInv = geom.InvCellSizeArray();
Source/SourceTerms/ERF_make_mom_sources.cpp:        Gpu::HostVector<    Real> r_plane_h(ncell);
Source/SourceTerms/ERF_make_mom_sources.cpp:        Gpu::DeviceVector<  Real> r_plane_d(ncell);
Source/SourceTerms/ERF_make_mom_sources.cpp:        Gpu::copyAsync(Gpu::hostToDevice, r_plane_h.begin(), r_plane_h.end(), r_plane_d.begin());
Source/SourceTerms/ERF_make_mom_sources.cpp:        ParallelFor(ncell, [=] AMREX_GPU_DEVICE (int k) noexcept
Source/SourceTerms/ERF_make_mom_sources.cpp:        Gpu::HostVector<    Real> u_plane_h(u_ncell), v_plane_h(v_ncell);
Source/SourceTerms/ERF_make_mom_sources.cpp:        Gpu::DeviceVector<  Real> u_plane_d(u_ncell), v_plane_d(v_ncell);
Source/SourceTerms/ERF_make_mom_sources.cpp:        Gpu::copyAsync(Gpu::hostToDevice, u_plane_h.begin(), u_plane_h.end(), u_plane_d.begin());
Source/SourceTerms/ERF_make_mom_sources.cpp:        Gpu::copyAsync(Gpu::hostToDevice, v_plane_h.begin(), v_plane_h.end(), v_plane_d.begin());
Source/SourceTerms/ERF_make_mom_sources.cpp:        ParallelFor(u_ncell, [=] AMREX_GPU_DEVICE (int k) noexcept
Source/SourceTerms/ERF_make_mom_sources.cpp:        ParallelFor(v_ncell, [=] AMREX_GPU_DEVICE (int k) noexcept
Source/SourceTerms/ERF_make_mom_sources.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_mom_sources.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/SourceTerms/ERF_make_mom_sources.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/SourceTerms/ERF_make_mom_sources.cpp:            ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_mom_sources.cpp:            ParallelFor(tby, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_mom_sources.cpp:                ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_mom_sources.cpp:        ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_mom_sources.cpp:        ParallelFor(tby, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_mom_sources.cpp:        ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_mom_sources.cpp:            ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_mom_sources.cpp:            ParallelFor(tby, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/SourceTerms/ERF_make_mom_sources.cpp:                ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_mom_sources.cpp:                ParallelFor(tby, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_mom_sources.cpp:                ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_mom_sources.cpp:                ParallelFor(tby, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_mom_sources.cpp:            ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/SourceTerms/ERF_make_mom_sources.cpp:            ParallelFor(tby, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DataStructs/ERF_InputSoundingData.H:#include <AMReX_Gpu.H>
Source/DataStructs/ERF_InputSoundingData.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
Source/DataStructs/ERF_InputSoundingData.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
Source/DataStructs/ERF_InputSoundingData.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
Source/DataStructs/ERF_InputSoundingData.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
Source/DataStructs/ERF_InputSoundingData.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
Source/DataStructs/ERF_InputSoundingData.H:            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
Source/DataStructs/ERF_InputSoundingData.H:            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
Source/DataStructs/ERF_InputSoundingData.H:    amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> z_inp_sound_d, theta_inp_sound_d, qv_inp_sound_d, U_inp_sound_d, V_inp_sound_d;
Source/DataStructs/ERF_InputSoundingData.H:    amrex::Gpu::DeviceVector<amrex::Real> p_inp_sound_d, rho_inp_sound_d;
Source/DataStructs/ERF_AdvStruct.H:#include <AMReX_Gpu.H>
Source/DataStructs/ERF_SpongeStruct.H:#include <AMReX_Gpu.H>
Source/DataStructs/ERF_InputSpongeData.H:#include <AMReX_Gpu.H>
Source/DataStructs/ERF_DiffStruct.H:#include <AMReX_Gpu.H>
Source/DataStructs/ERF_TurbPertStruct.H:                   const amrex::GpuArray<amrex::Real,3> dx,
Source/DataStructs/ERF_TurbPertStruct.H:                ParallelFor(ubx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/DataStructs/ERF_TurbPertStruct.H:                ParallelForRNG(ubx, [=] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
Source/DataStructs/ERF_TurbPertStruct.H:        amrex::Gpu::DeviceVector<amrex::Real> avg_d(1,0.);
Source/DataStructs/ERF_TurbPertStruct.H:                ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), ubx, [=]
Source/DataStructs/ERF_TurbPertStruct.H:                AMREX_GPU_DEVICE(int i, int j, int k, amrex::Gpu::Handler const& handler) noexcept {
Source/DataStructs/ERF_TurbPertStruct.H:                    amrex::Gpu::deviceReduceSum(&avg[0], pert_cell(i,j,k)*norm, handler);
Source/DataStructs/ERF_TurbPertStruct.H:                amrex::Gpu::copy(amrex::Gpu::deviceToHost, avg_d.begin(), avg_d.end(), avg_h.begin());
Source/DataStructs/ERF_TurbPertStruct.H:               ParallelFor(ubx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/DataStructs/ERF_TurbPertStruct.H:        amrex::Gpu::DeviceVector<amrex::Real> avg_d(n_avg,0.);
Source/DataStructs/ERF_TurbPertStruct.H:                ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), ubx_u, [=]
Source/DataStructs/ERF_TurbPertStruct.H:                AMREX_GPU_DEVICE(int i, int j, int k, amrex::Gpu::Handler const& handler) noexcept {
Source/DataStructs/ERF_TurbPertStruct.H:                    amrex::Gpu::deviceReduceSum(&avg[0], xvel_arry(i,j,k)*norm, handler);
Source/DataStructs/ERF_TurbPertStruct.H:                ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), ubxSlab_lo, [=]
Source/DataStructs/ERF_TurbPertStruct.H:                AMREX_GPU_DEVICE(int i, int j, int k, amrex::Gpu::Handler const& handler) noexcept {
Source/DataStructs/ERF_TurbPertStruct.H:                    amrex::Gpu::deviceReduceSum(&avg[0], xvel_arry(i,j,k)*norm_lo, handler);
Source/DataStructs/ERF_TurbPertStruct.H:                ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), ubxSlab_hi, [=]
Source/DataStructs/ERF_TurbPertStruct.H:                AMREX_GPU_DEVICE(int i, int j, int k, amrex::Gpu::Handler const& handler) noexcept {
Source/DataStructs/ERF_TurbPertStruct.H:                    amrex::Gpu::deviceReduceSum(&avg[2], xvel_arry(i,j,k)*norm_hi, handler);
Source/DataStructs/ERF_TurbPertStruct.H:                ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), ubx_v, [=]
Source/DataStructs/ERF_TurbPertStruct.H:                AMREX_GPU_DEVICE(int i, int j, int k, amrex::Gpu::Handler const& handler) noexcept {
Source/DataStructs/ERF_TurbPertStruct.H:                    amrex::Gpu::deviceReduceSum(&avg[1], yvel_arry(i,j,k)*norm, handler);
Source/DataStructs/ERF_TurbPertStruct.H:                ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), ubxSlab_lo, [=]
Source/DataStructs/ERF_TurbPertStruct.H:                AMREX_GPU_DEVICE(int i, int j, int k, amrex::Gpu::Handler const& handler) noexcept {
Source/DataStructs/ERF_TurbPertStruct.H:                    amrex::Gpu::deviceReduceSum(&avg[1], yvel_arry(i,j,k)*norm_lo, handler);
Source/DataStructs/ERF_TurbPertStruct.H:                ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), ubxSlab_hi, [=]
Source/DataStructs/ERF_TurbPertStruct.H:                AMREX_GPU_DEVICE(int i, int j, int k, amrex::Gpu::Handler const& handler) noexcept {
Source/DataStructs/ERF_TurbPertStruct.H:                    amrex::Gpu::deviceReduceSum(&avg[3], yvel_arry(i,j,k)*norm_hi, handler);
Source/DataStructs/ERF_TurbPertStruct.H:        amrex::Gpu::copy(amrex::Gpu::deviceToHost, avg_d.begin(), avg_d.end(), avg_h.begin());
Source/DataStructs/ERF_DataStruct.H:#include <AMReX_Gpu.H>
Source/DataStructs/ERF_DataStruct.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> abl_pressure_grad;
Source/DataStructs/ERF_DataStruct.H:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> abl_geo_forcing;
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:#include <AMReX_GpuContainers.H>
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:#include <AMReX_GpuPrint.H>
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:                       const Gpu::DeviceVector<BCRec>& domain_bcs_type_d,
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:    const GpuArray<Real, AMREX_SPACEDIM> dxInv = geom.InvCellSizeArray();
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:    const GpuArray<Real,AMREX_SPACEDIM> grav_gpu{grav[0], grav[1], grav[2]};
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:    Vector<Real> max_scal(nvar, 1.0e34); Gpu::DeviceVector<Real> max_scal_d(nvar);
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:    Vector<Real> min_scal(nvar,-1.0e34); Gpu::DeviceVector<Real> min_scal_d(nvar);
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:            GpuTuple<Real,Real> mm = ParReduce(TypeList<ReduceOpMax,ReduceOpMin>{},
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:                [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:                -> GpuTuple<Real,Real>
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:    Gpu::copy(Gpu::hostToDevice, max_scal.begin(), max_scal.end(), max_scal_d.begin());
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:    Gpu::copy(Gpu::hostToDevice, min_scal.begin(), min_scal.end(), min_scal_d.begin());
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:        const GpuArray<const Array4<Real>, AMREX_SPACEDIM>
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:        const GpuArray<Array4<Real>, AMREX_SPACEDIM> flx_tmp_arr{{AMREX_D_DECL(tmpx,tmpy,tmpz)}};
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:            ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:#ifdef AMREX_USE_GPU
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:                    ParallelFor(gbxo_lo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:                    ParallelFor(gbxo_hi, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:                    ParallelFor(gbxo_mid, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:                    ParallelFor(gbxo_mid, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:                ParallelFor(gbxo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:                                       tm_arr, grav_gpu, bc_ptr_d, l_use_most);
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:                                       tm_arr, grav_gpu, bc_ptr_d, l_use_most);
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:        ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:        ParallelFor(tby, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:            ParallelFor(lo_x_dom_face, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:            ParallelFor(hi_x_dom_face, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:            ParallelFor(lo_y_dom_face, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:            ParallelFor(hi_y_dom_face, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:        ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:                ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int ) // bottom of box but not of domain
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:                ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int ) // top of box but not of domain
Source/TimeIntegration/ERF_slow_rhs_pre.cpp:            Gpu::streamSynchronize();
Source/TimeIntegration/ERF_fast_rhs_N.cpp:    const GpuArray<Real, AMREX_SPACEDIM> dxInv = geom.InvCellSizeArray();
Source/TimeIntegration/ERF_fast_rhs_N.cpp:    const GpuArray<Real,AMREX_SPACEDIM> grav_gpu{grav[0], grav[1], grav[2]};
Source/TimeIntegration/ERF_fast_rhs_N.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_fast_rhs_N.cpp:    for ( MFIter mfi(S_stage_data[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeIntegration/ERF_fast_rhs_N.cpp:        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_fast_rhs_N.cpp:        ParallelFor(gtbz, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_N.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_fast_rhs_N.cpp:    for ( MFIter mfi(S_stage_data[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeIntegration/ERF_fast_rhs_N.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_N.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_N.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_N.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_N.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_fast_rhs_N.cpp:        const GpuArray<const Array4<Real>, AMREX_SPACEDIM>
Source/TimeIntegration/ERF_fast_rhs_N.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_N.cpp:        // We set grav_gpu[2] to be the vector component which is negative
Source/TimeIntegration/ERF_fast_rhs_N.cpp:        Real halfg = std::abs(0.5 * grav_gpu[2]);
Source/TimeIntegration/ERF_fast_rhs_N.cpp:        ParallelFor(bx_shrunk_in_k, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_N.cpp:        ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int)
Source/TimeIntegration/ERF_fast_rhs_N.cpp:#ifdef AMREX_USE_GPU
Source/TimeIntegration/ERF_fast_rhs_N.cpp:            ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int)
Source/TimeIntegration/ERF_fast_rhs_N.cpp:            ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int)
Source/TimeIntegration/ERF_fast_rhs_N.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_fast_rhs_N.cpp:            Gpu::streamSynchronize();
Source/TimeIntegration/ERF_fast_rhs_N.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_fast_rhs_N.cpp:    for ( MFIter mfi(S_stage_data[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeIntegration/ERF_fast_rhs_N.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_fast_rhs_N.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_fast_rhs_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_TI_slow_headers.H:                      const amrex::Gpu::DeviceVector<amrex::BCRec>& domain_bcs_type_d,
Source/TimeIntegration/ERF_TI_slow_headers.H:                       const amrex::Gpu::DeviceVector<amrex::BCRec>& domain_bcs_type_d,
Source/TimeIntegration/ERF_TI_slow_headers.H:                       const amrex::Gpu::DeviceVector<amrex::BCRec>& domain_bcs_type_d,
Source/TimeIntegration/ERF_fast_rhs_T.cpp:    const GpuArray<Real, AMREX_SPACEDIM> dxInv = geom.InvCellSizeArray();
Source/TimeIntegration/ERF_fast_rhs_T.cpp:    const GpuArray<Real,AMREX_SPACEDIM> grav_gpu{grav[0], grav[1], grav[2]};
Source/TimeIntegration/ERF_fast_rhs_T.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_fast_rhs_T.cpp:    for ( MFIter mfi(S_stage_data[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeIntegration/ERF_fast_rhs_T.cpp:            ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_T.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_fast_rhs_T.cpp:    for ( MFIter mfi(S_stage_data[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_T.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_fast_rhs_T.cpp:    for ( MFIter mfi(S_stage_data[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_T.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_T.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        const GpuArray<const Array4<Real>, AMREX_SPACEDIM>
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_T.cpp:            ParallelFor(gbxo_lo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_T.cpp:            ParallelFor(gbxo_hi, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        ParallelFor(gbxo_mid, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        // We set grav_gpu[2] to be the vector component which is negative
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        Real halfg = std::abs(0.5 * grav_gpu[2]);
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        ParallelFor(bx_shrunk_in_k, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_T.cpp:#ifdef AMREX_USE_GPU
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int)
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_T.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_fast_rhs_T.cpp:            Gpu::streamSynchronize();
Source/TimeIntegration/ERF_TI_substep_fun.H:            for (MFIter mfi(*z_t_rk[level],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeIntegration/ERF_TI_substep_fun.H:                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_make_fast_coeffs.cpp:                       amrex::GpuArray<ERF_BC, AMREX_SPACEDIM*2> &phys_bc_type)
Source/TimeIntegration/ERF_make_fast_coeffs.cpp:    const GpuArray<Real, AMREX_SPACEDIM> dxInv = geom.InvCellSizeArray();
Source/TimeIntegration/ERF_make_fast_coeffs.cpp:    const GpuArray<Real,AMREX_SPACEDIM> grav_gpu{grav[0], grav[1], grav[2]};
Source/TimeIntegration/ERF_make_fast_coeffs.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_make_fast_coeffs.cpp:        // We set grav_gpu[2] to be the vector component which is negative
Source/TimeIntegration/ERF_make_fast_coeffs.cpp:        Real halfg = std::abs(0.5 * grav_gpu[2]);
Source/TimeIntegration/ERF_make_fast_coeffs.cpp:            ParallelFor(bx_shrunk_in_k, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_make_fast_coeffs.cpp:            ParallelFor(bx_shrunk_in_k, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_make_fast_coeffs.cpp:#ifdef AMREX_USE_GPU
Source/TimeIntegration/ERF_make_fast_coeffs.cpp:        ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int) {
Source/TimeIntegration/ERF_make_fast_coeffs.cpp:            ParallelFor(bx_shrunk_in_k, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_advance_dycore.cpp:        const GpuArray<Real, AMREX_SPACEDIM> dxInv = fine_geom.InvCellSizeArray();
Source/TimeIntegration/ERF_advance_dycore.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_TI_utils.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_TI_utils.H:      for (MFIter mfi(cons_state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeIntegration/ERF_TI_utils.H:          amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_TI_utils.H:          amrex::ParallelFor(gbx1, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:    const GpuArray<Real, AMREX_SPACEDIM> dxInv = geom.InvCellSizeArray();
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:    const GpuArray<Real,AMREX_SPACEDIM> grav_gpu{grav[0], grav[1], grav[2]};
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        const GpuArray<const Array4<Real>, AMREX_SPACEDIM>
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        ParallelFor(gbxo_lo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        ParallelFor(gbxo_hi, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        ParallelFor(gbxo_mid, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        // We set grav_gpu[2] to be the vector component which is negative
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        Real halfg = std::abs(0.5 * grav_gpu[2]);
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        ParallelFor(bx_shrunk_in_k, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:#ifdef AMREX_USE_GPU
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int)
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_fast_rhs_MT.cpp:            Gpu::streamSynchronize();
Source/TimeIntegration/ERF_TI_slow_rhs_fun.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_TI_slow_rhs_fun.H:            for (MFIter mfi(*z_t_rk[level],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeIntegration/ERF_TI_slow_rhs_fun.H:                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_TI_slow_rhs_fun.H:            const GpuArray<Real, AMREX_SPACEDIM> dxInv = fine_geom.InvCellSizeArray();
Source/TimeIntegration/ERF_TI_slow_rhs_fun.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_TI_slow_rhs_fun.H:           for ( MFIter mfi(*p0,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeIntegration/ERF_TI_slow_rhs_fun.H:               amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_TI_slow_rhs_fun.H:               amrex::ParallelFor(gbx2, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_make_tau_terms.cpp:#include <AMReX_GpuContainers.H>
Source/TimeIntegration/ERF_make_tau_terms.cpp:    const GpuArray<Real, AMREX_SPACEDIM> dxInv = geom.InvCellSizeArray();
Source/TimeIntegration/ERF_make_tau_terms.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_make_tau_terms.cpp:                ParallelFor(gbxo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_make_tau_terms.cpp:                ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_make_tau_terms.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_make_tau_terms.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_make_tau_terms.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_make_tau_terms.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_make_tau_terms.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_make_tau_terms.cpp:                ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_make_tau_terms.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeIntegration/ERF_make_tau_terms.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_make_tau_terms.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_make_tau_terms.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_make_tau_terms.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_TI_no_substep_fun.H:        const amrex::GpuArray<int, IntVars::NumTypes> scomp_fast = {0,0,0,0};
Source/TimeIntegration/ERF_TI_no_substep_fun.H:        const amrex::GpuArray<int, IntVars::NumTypes> ncomp_fast = {2,1,1,1};
Source/TimeIntegration/ERF_TI_no_substep_fun.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_TI_no_substep_fun.H:            for ( MFIter mfi(S_sum[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                Gpu::AsyncVector<Array4<Real> >  sold_d(n_data);
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                Gpu::AsyncVector<Array4<Real> >  ssum_d(n_data);
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                Gpu::AsyncVector<Array4<Real> > fslow_d(n_data);
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                Gpu::copy(Gpu::hostToDevice,  sold_h.begin(),  sold_h.end(),  sold_d.begin());
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                Gpu::copy(Gpu::hostToDevice,  ssum_h.begin(),  ssum_h.end(),  ssum_d.begin());
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                Gpu::copy(Gpu::hostToDevice, fslow_h.begin(), fslow_h.end(), fslow_d.begin());
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int nn) {
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int nn) {
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_TI_no_substep_fun.H:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_TI_fast_headers.H:                       amrex::GpuArray<ERF_BC, AMREX_SPACEDIM*2> &phys_bc_type);
Source/TimeIntegration/ERF_slow_rhs_post.cpp:                        const Gpu::DeviceVector<BCRec>& domain_bcs_type_d,
Source/TimeIntegration/ERF_slow_rhs_post.cpp:    const GpuArray<Real, AMREX_SPACEDIM> dxInv = geom.InvCellSizeArray();
Source/TimeIntegration/ERF_slow_rhs_post.cpp:    const GpuArray<Real,AMREX_SPACEDIM> grav_gpu{grav[0], grav[1], grav[2]};
Source/TimeIntegration/ERF_slow_rhs_post.cpp:    Vector<Real> max_scal(nvar, 1.0e34); Gpu::DeviceVector<Real> max_scal_d(nvar);
Source/TimeIntegration/ERF_slow_rhs_post.cpp:    Vector<Real> min_scal(nvar,-1.0e34); Gpu::DeviceVector<Real> min_scal_d(nvar);
Source/TimeIntegration/ERF_slow_rhs_post.cpp:            GpuTuple<Real,Real> mm = ParReduce(TypeList<ReduceOpMax,ReduceOpMin>{},
Source/TimeIntegration/ERF_slow_rhs_post.cpp:                [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
Source/TimeIntegration/ERF_slow_rhs_post.cpp:                -> GpuTuple<Real,Real>
Source/TimeIntegration/ERF_slow_rhs_post.cpp:    Gpu::copy(Gpu::hostToDevice, max_scal.begin(), max_scal.end(), max_scal_d.begin());
Source/TimeIntegration/ERF_slow_rhs_post.cpp:    Gpu::copy(Gpu::hostToDevice, min_scal.begin(), min_scal.end(), min_scal_d.begin());
Source/TimeIntegration/ERF_slow_rhs_post.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeIntegration/ERF_slow_rhs_post.cpp:      for ( MFIter mfi(S_data[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/TimeIntegration/ERF_slow_rhs_post.cpp:        const GpuArray<const Array4<Real>, AMREX_SPACEDIM>
Source/TimeIntegration/ERF_slow_rhs_post.cpp:        const GpuArray<Array4<Real>, AMREX_SPACEDIM> flx_tmp_arr{{AMREX_D_DECL(tmpx,tmpy,tmpz)}};
Source/TimeIntegration/ERF_slow_rhs_post.cpp:        const GpuArray<int, IntVars::NumTypes> scomp_slow = {  2,0,0,0};
Source/TimeIntegration/ERF_slow_rhs_post.cpp:        const GpuArray<int, IntVars::NumTypes> ncomp_slow = {nsv,0,0,0};
Source/TimeIntegration/ERF_slow_rhs_post.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int nn) {
Source/TimeIntegration/ERF_slow_rhs_post.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_slow_rhs_post.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_slow_rhs_post.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_slow_rhs_post.cpp:                                               tm_arr, grav_gpu, bc_ptr_d, use_most);
Source/TimeIntegration/ERF_slow_rhs_post.cpp:                                               tm_arr, grav_gpu, bc_ptr_d, use_most);
Source/TimeIntegration/ERF_slow_rhs_post.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int nn) noexcept {
Source/TimeIntegration/ERF_slow_rhs_post.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int nn) noexcept {
Source/TimeIntegration/ERF_slow_rhs_post.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept {
Source/TimeIntegration/ERF_slow_rhs_post.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_slow_rhs_post.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_slow_rhs_post.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/TimeIntegration/ERF_slow_rhs_post.cpp:            Gpu::streamSynchronize();
Source/TimeIntegration/ERF_ComputeTimestep.cpp:       [=] AMREX_GPU_HOST_DEVICE (Box const& b,
Source/TimeIntegration/ERF_ComputeTimestep.cpp:       [=] AMREX_GPU_HOST_DEVICE (Box const& b,
Source/TimeIntegration/ERF_ComputeTimestep.cpp:       [=] AMREX_GPU_HOST_DEVICE (Box const& b,
Source/Diffusion/ERF_DiffusionSrcForMom_T.cpp:                      const GpuArray<Real, AMREX_SPACEDIM>& dxInv,
Source/Diffusion/ERF_DiffusionSrcForMom_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Diffusion/ERF_DiffusionSrcForMom_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Diffusion/ERF_DiffusionSrcForMom_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Diffusion/ERF_ComputeStress_N.cpp:        ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_N.cpp:        ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_N.cpp:        ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_N.cpp:        ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_N.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_T.cpp:                         const GpuArray<Real, AMREX_SPACEDIM>& dxInv)
Source/Diffusion/ERF_ComputeStress_T.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_T.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_T.cpp:    ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:                        const GpuArray<Real, AMREX_SPACEDIM>& dxInv)
Source/Diffusion/ERF_ComputeStress_T.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_T.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_T.cpp:    ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStress_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStress_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp: * @param[in]  grav_gpu gravity vector
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:                        const GpuArray<Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:                        const GpuArray<Real,AMREX_SPACEDIM> grav_gpu,
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:    Real l_abs_g         = std::abs(grav_gpu[2]);
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:    Gpu::AsyncVector<Real> alpha_eff_d;
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:    Gpu::AsyncVector<int>  eddy_diff_idx_d,eddy_diff_idy_d,eddy_diff_idz_d;
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:    Gpu::copy(Gpu::hostToDevice, alpha_eff.begin()    , alpha_eff.end()    , alpha_eff_d.begin());
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:    Gpu::copy(Gpu::hostToDevice, eddy_diff_idx.begin(), eddy_diff_idx.end(), eddy_diff_idx_d.begin());
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:    Gpu::copy(Gpu::hostToDevice, eddy_diff_idy.begin(), eddy_diff_idy.end(), eddy_diff_idy_d.begin());
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:    Gpu::copy(Gpu::hostToDevice, eddy_diff_idz.begin(), eddy_diff_idz.end(), eddy_diff_idz_d.begin());
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(xbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(ybx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(zbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(xbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(ybx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(zbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(xbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(ybx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(zbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(xbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(ybx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(zbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_N.cpp:        ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeStrain_T.cpp:                 const BCRec* bc_ptr, const GpuArray<Real, AMREX_SPACEDIM>& dxInv,
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planecc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:    ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_T.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:                 const BCRec* bc_ptr, const GpuArray<Real, AMREX_SPACEDIM>& dxInv,
Source/Diffusion/ERF_ComputeStrain_N.cpp:        ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:        ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:        ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:        ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:        ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:        ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:    ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_ComputeStrain_N.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp: * @param[in]  grav_gpu gravity vector
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:                        const GpuArray<Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:                        const GpuArray<Real,AMREX_SPACEDIM> grav_gpu,
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:    Real l_abs_g         = std::abs(grav_gpu[2]);
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:    Gpu::AsyncVector<Real> alpha_eff_d;
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:    Gpu::AsyncVector<int>  eddy_diff_idx_d,eddy_diff_idy_d,eddy_diff_idz_d;
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:    Gpu::copy(Gpu::hostToDevice, alpha_eff.begin()    , alpha_eff.end()    , alpha_eff_d.begin());
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:    Gpu::copy(Gpu::hostToDevice, eddy_diff_idx.begin(), eddy_diff_idx.end(), eddy_diff_idx_d.begin());
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:    Gpu::copy(Gpu::hostToDevice, eddy_diff_idy.begin(), eddy_diff_idy.end(), eddy_diff_idy_d.begin());
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:    Gpu::copy(Gpu::hostToDevice, eddy_diff_idz.begin(), eddy_diff_idz.end(), eddy_diff_idz_d.begin());
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(xbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(ybx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(zbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(xbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(ybx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(zbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(xbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(ybx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(zbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(xbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(ybx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(zbx, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:      ParallelFor(planexy, num_comp, [=] AMREX_GPU_DEVICE (int i, int j, int , int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:    ParallelFor(zbx3, num_comp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:    ParallelFor(xbx, num_comp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:    ParallelFor(ybx, num_comp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_DiffusionSrcForState_T.cpp:        ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_DiffusionSrcForMom_N.cpp:                      const GpuArray<Real, AMREX_SPACEDIM>& dxInv,
Source/Diffusion/ERF_DiffusionSrcForMom_N.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Diffusion/ERF_DiffusionSrcForMom_N.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Diffusion/ERF_DiffusionSrcForMom_N.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Diffusion/ERF_EddyViscosity.H:AMREX_GPU_DEVICE
Source/Diffusion/ERF_Diffusion.H:                           const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv,
Source/Diffusion/ERF_Diffusion.H:                           const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv,
Source/Diffusion/ERF_Diffusion.H:                             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Diffusion/ERF_Diffusion.H:                             const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> grav_gpu,
Source/Diffusion/ERF_Diffusion.H:                             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv,
Source/Diffusion/ERF_Diffusion.H:                             const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> grav_gpu,
Source/Diffusion/ERF_Diffusion.H:                              const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv);
Source/Diffusion/ERF_Diffusion.H:                             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv);
Source/Diffusion/ERF_Diffusion.H:                     const amrex::BCRec* bc_ptr, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv,
Source/Diffusion/ERF_Diffusion.H:                     const amrex::BCRec* bc_ptr, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv,
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:    const GpuArray<Real, AMREX_SPACEDIM> cellSizeInv = geom.InvCellSizeArray();
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:      for (MFIter mfi(eddyViscosity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:          ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:        for ( MFIter mfi(eddyViscosity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:            ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:    Gpu::AsyncVector<Real> d_Factors; d_Factors.resize(Factors.size());
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:    Gpu::copy(Gpu::hostToDevice, Factors.begin(), Factors.end(), d_Factors.begin());
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:    for ( MFIter mfi(eddyViscosity,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:            ParallelFor(planex, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:            ParallelFor(planex, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:            ParallelFor(planey, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:            ParallelFor(planey, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:                ParallelFor(planex, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:                ParallelFor(planex, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:                ParallelFor(planey, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:                ParallelFor(planey, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:                   ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:                ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:                    ParallelFor(planez, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Diffusion/ERF_ComputeTurbulentViscosity.cpp:                 ParallelFor(planez, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_u   , amrex::Gpu::HostVector<amrex::Real>& h_avg_v,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_w   , amrex::Gpu::HostVector<amrex::Real>& h_avg_rho,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_th  , amrex::Gpu::HostVector<amrex::Real>& h_avg_ksgs,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_Kmv , amrex::Gpu::HostVector<amrex::Real>& h_avg_Khv,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_qv  , amrex::Gpu::HostVector<amrex::Real>& h_avg_qc,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_qr  ,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_wqv , amrex::Gpu::HostVector<amrex::Real>& h_avg_wqc,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_wqr ,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_qi  , amrex::Gpu::HostVector<amrex::Real>& h_avg_qs,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_qg  ,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_uu  , amrex::Gpu::HostVector<amrex::Real>& h_avg_uv,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_uw,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_vv  , amrex::Gpu::HostVector<amrex::Real>& h_avg_vw,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_ww,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_uth , amrex::Gpu::HostVector<amrex::Real>& h_avg_vth,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_wth, amrex::Gpu::HostVector<amrex::Real>& h_avg_thth,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_ku, amrex::Gpu::HostVector<amrex::Real>& h_avg_kv,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_kw,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_p,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_pu, amrex::Gpu::HostVector<amrex::Real>& h_avg_pv,
Source/ERF.H:                               amrex::Gpu::HostVector<amrex::Real>& h_avg_pw, amrex::Gpu::HostVector<amrex::Real>& h_avg_wthv);
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_u   , amrex::Gpu::HostVector<amrex::Real>& h_avg_v,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_w   , amrex::Gpu::HostVector<amrex::Real>& h_avg_rho,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_th  , amrex::Gpu::HostVector<amrex::Real>& h_avg_ksgs,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_Kmv , amrex::Gpu::HostVector<amrex::Real>& h_avg_Khv,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_qv  , amrex::Gpu::HostVector<amrex::Real>& h_avg_qc,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_qr  ,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_wqv , amrex::Gpu::HostVector<amrex::Real>& h_avg_wqc,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_wqr ,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_qi  , amrex::Gpu::HostVector<amrex::Real>& h_avg_qs,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_qg  ,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_uu  , amrex::Gpu::HostVector<amrex::Real>& h_avg_uv,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_uw,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_vv  , amrex::Gpu::HostVector<amrex::Real>& h_avg_vw,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_ww,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_uth , amrex::Gpu::HostVector<amrex::Real>& h_avg_vth,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_wth, amrex::Gpu::HostVector<amrex::Real>& h_avg_thth,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_ku, amrex::Gpu::HostVector<amrex::Real>& h_avg_kv,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_kw,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_p,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_pu, amrex::Gpu::HostVector<amrex::Real>& h_avg_pv,
Source/ERF.H:                                    amrex::Gpu::HostVector<amrex::Real>& h_avg_pw, amrex::Gpu::HostVector<amrex::Real>& h_avg_wthv);
Source/ERF.H:    void derive_stress_profiles (amrex::Gpu::HostVector<amrex::Real>& h_avg_tau11, amrex::Gpu::HostVector<amrex::Real>& h_avg_tau12,
Source/ERF.H:                                 amrex::Gpu::HostVector<amrex::Real>& h_avg_tau13, amrex::Gpu::HostVector<amrex::Real>& h_avg_tau22,
Source/ERF.H:                                 amrex::Gpu::HostVector<amrex::Real>& h_avg_tau23, amrex::Gpu::HostVector<amrex::Real>& h_avg_tau33,
Source/ERF.H:                                 amrex::Gpu::HostVector<amrex::Real>& h_avg_hfx3,  amrex::Gpu::HostVector<amrex::Real>& h_avg_q1fx3,
Source/ERF.H:                                 amrex::Gpu::HostVector<amrex::Real>& h_avg_q2fx3, amrex::Gpu::HostVector<amrex::Real>& h_avg_diss);
Source/ERF.H:    void derive_stress_profiles_stag (amrex::Gpu::HostVector<amrex::Real>& h_avg_tau11, amrex::Gpu::HostVector<amrex::Real>& h_avg_tau12,
Source/ERF.H:                                      amrex::Gpu::HostVector<amrex::Real>& h_avg_tau13, amrex::Gpu::HostVector<amrex::Real>& h_avg_tau22,
Source/ERF.H:                                      amrex::Gpu::HostVector<amrex::Real>& h_avg_tau23, amrex::Gpu::HostVector<amrex::Real>& h_avg_tau33,
Source/ERF.H:                                      amrex::Gpu::HostVector<amrex::Real>& h_avg_hfx3,  amrex::Gpu::HostVector<amrex::Real>& h_avg_q1fx3,
Source/ERF.H:                                      amrex::Gpu::HostVector<amrex::Real>& h_avg_q2fx3, amrex::Gpu::HostVector<amrex::Real>& h_avg_diss);
Source/ERF.H:    amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> xvel_bc_data;
Source/ERF.H:    amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> yvel_bc_data;
Source/ERF.H:    amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> zvel_bc_data;
Source/ERF.H:    amrex::Gpu::DeviceVector<amrex::BCRec> domain_bcs_type_d;
Source/ERF.H:    amrex::GpuArray<ERF_BC, AMREX_SPACEDIM*2> phys_bc_type;
Source/ERF.H:    amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real> > d_rhotheta_src;
Source/ERF.H:    amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real> > d_rhoqt_src;
Source/ERF.H:    amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real> > d_w_subsid;
Source/ERF.H:    amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real> > d_u_geos;
Source/ERF.H:    amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real> > d_v_geos;
Source/ERF.H:                                amrex::Gpu::DeviceVector<amrex::Real>& u_geos_d,
Source/ERF.H:                                amrex::Gpu::DeviceVector<amrex::Real>& v_geos_d,
Source/ERF.H:    amrex::Vector<amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real> > > d_rayleigh_ptrs;
Source/ERF.H:    amrex::Vector<amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real> > > d_sponge_ptrs;
Source/ERF.H:    amrex::Gpu::DeviceVector<amrex::Real> d_havg_density;
Source/ERF.H:    amrex::Gpu::DeviceVector<amrex::Real> d_havg_temperature;
Source/ERF.H:    amrex::Gpu::DeviceVector<amrex::Real> d_havg_pressure;
Source/ERF.H:    amrex::Gpu::DeviceVector<amrex::Real> d_havg_qv;
Source/ERF.H:    amrex::Gpu::DeviceVector<amrex::Real> d_havg_qc;
Source/ERF_read_waves.cpp:    for (MFIter mfi(*Hwave[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/ERF_read_waves.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/ERF_read_waves.cpp:    for (MFIter mfi(x_avg,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/ERF_read_waves.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/ERF_read_waves.cpp:    for (MFIter mfi(y_avg,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/ERF_read_waves.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/ERF_read_waves.cpp:    for (MFIter mfi(u_mag, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/ERF_read_waves.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Particles/ERFPCUtils.cpp:        [=] AMREX_GPU_DEVICE (  const ERFPC::ParticleTileType::ConstParticleTileDataType& ptr,
Source/Particles/ERFPCUtils.cpp:                [=] AMREX_GPU_DEVICE ( const ERFPC::ParticleType&, int)
Source/Particles/ERF_ParticleData.H:#include <AMReX_Gpu.H>
Source/Particles/ERFPCEvolve.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Particles/ERFPCEvolve.cpp:            //array of these pointers to pass to the GPU
Source/Particles/ERFPCEvolve.cpp:            GpuArray<Array4<const Real>, AMREX_SPACEDIM>
Source/Particles/ERFPCEvolve.cpp:            ParallelFor(n, [=] AMREX_GPU_DEVICE (int i)
Source/Particles/ERFPCEvolve.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Particles/ERFPCEvolve.cpp:        ParallelFor(n, [=] AMREX_GPU_DEVICE (int i)
Source/Particles/ERFPCInitializations.cpp:            ParallelFor(tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Particles/ERFPCInitializations.cpp:            ParallelFor(tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Particles/ERFPCInitializations.cpp:                                       [=] AMREX_GPU_DEVICE (int i) -> int { return in[i]; },
Source/Particles/ERFPCInitializations.cpp:                                       [=] AMREX_GPU_DEVICE (int i, int const &x) { out[i] = x; },
Source/Particles/ERFPCInitializations.cpp:            ParallelForRNG(tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k,
Source/Particles/ERFPCInitializations.cpp:            ParallelFor(tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Particles/ERFPCInitializations.cpp:            ParallelForRNG(tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k,
Source/Particles/ERFPCInitializations.cpp:            ParallelFor(tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Particles/ERFPC.H:    AMREX_GPU_HOST_DEVICE
Source/Particles/ERFPC.H:                                 amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Source/Particles/ERFPC.H:                                 amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
Source/Particles/ERFPC.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ERFPC.H:                              amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& a_plo,
Source/Particles/ERFPC.H:                              amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& a_dxi,
Source/Particles/ERFPC.H:        // public due to CUDA extended lambda capture rules
Source/Prob/ERF_init_constant_density_hse.H:    for ( amrex::MFIter mfi(rho_hse, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Prob/ERF_init_constant_density_hse.H:       ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Prob/ERF_init_density_hse_dry.H:            amrex::ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int)
Source/Prob/ERF_init_density_hse_dry.H:        amrex::Gpu::DeviceVector<amrex::Real> d_r(khi+2);
Source/Prob/ERF_init_density_hse_dry.H:        amrex::Gpu::DeviceVector<amrex::Real> d_p(khi+2);
Source/Prob/ERF_init_density_hse_dry.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_r.begin(), h_r.end(), d_r.begin());
Source/Prob/ERF_init_density_hse_dry.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_p.begin(), h_p.end(), d_p.begin());
Source/Prob/ERF_init_density_hse_dry.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Prob/ERF_init_density_hse_dry.H:          for ( amrex::MFIter mfi(rho_hse, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Prob/ERF_init_density_hse_dry.H:              amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/ERF_Derive.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/ERF_Derive.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/ERF_Derive.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/ERF_Derive.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/ERF_Derive.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/ERF_Derive.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/ERF_Derive.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/ERF_Derive.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForMom.cpp:                    const GpuArray<Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForMom.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Advection/ERF_AdvectionSrcForMom.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Advection/ERF_AdvectionSrcForMom.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForMom.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForMom.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForMom.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForMom.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForMom.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForMom_N.H:AMREX_GPU_DEVICE
Source/Advection/ERF_AdvectionSrcForMom_N.H:                       const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForMom_N.H:AMREX_GPU_DEVICE
Source/Advection/ERF_AdvectionSrcForMom_N.H:                       const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForMom_N.H:AMREX_GPU_DEVICE
Source/Advection/ERF_AdvectionSrcForMom_N.H:                       const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForMom_N.H:                             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForMom_N.H:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForMom_N.H:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForMom_N.H:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForMom_N.H:                          const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForMom_T.H:AMREX_GPU_DEVICE
Source/Advection/ERF_AdvectionSrcForMom_T.H:                     const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForMom_T.H:AMREX_GPU_DEVICE
Source/Advection/ERF_AdvectionSrcForMom_T.H:                     const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForMom_T.H:AMREX_GPU_DEVICE
Source/Advection/ERF_AdvectionSrcForMom_T.H:                     const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForMom_T.H:                           const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForMom_T.H:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForMom_T.H:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForMom_T.H:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForMom_T.H:                        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_Advection.H:                         const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_Advection.H:                         const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM>& flx_arr,
Source/Advection/ERF_Advection.H:                             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_Advection.H:                             const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM>& flx_arr,
Source/Advection/ERF_Advection.H:                             const amrex::GpuArray<      amrex::Array4<amrex::Real>, AMREX_SPACEDIM>& flx_tmp_arr,
Source/Advection/ERF_Advection.H:                         const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_Advection.H:                              const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv,
Source/Advection/ERF_Advection.H:                                    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_Advection.H:                                    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_Advection.H:                                    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_Advection.H:                                    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_Advection.H:AMREX_GPU_HOST_DEVICE
Source/Advection/ERF_Advection.H:AMREX_GPU_HOST_DEVICE
Source/Advection/ERF_AdvectionSrcForOpenBC.cpp:                              const GpuArray<Real, AMREX_SPACEDIM>& dxInv,
Source/Advection/ERF_AdvectionSrcForOpenBC.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForOpenBC.cpp:                                    const GpuArray<Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForOpenBC.cpp:    ParallelFor(bxx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForOpenBC.cpp:                                    const GpuArray<Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForOpenBC.cpp:    ParallelFor(bxy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForOpenBC.cpp:                                    const GpuArray<Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForOpenBC.cpp:    ParallelFor(bxz, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForOpenBC.cpp:                                    const GpuArray<Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForOpenBC.cpp:    ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Advection/ERF_AdvectionSrcForOpenBC.cpp:AMREX_GPU_HOST_DEVICE
Source/Advection/ERF_AdvectionSrcForState.cpp:                    const GpuArray<Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForState.cpp:                    const GpuArray<const Array4<Real>, AMREX_SPACEDIM>& flx_arr,
Source/Advection/ERF_AdvectionSrcForState.cpp:    ParallelFor(xbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForState.cpp:    ParallelFor(ybx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForState.cpp:    ParallelFor(zbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForState.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForState.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Advection/ERF_AdvectionSrcForState.cpp:                        const GpuArray<Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Advection/ERF_AdvectionSrcForState.cpp:                        const GpuArray<const Array4<Real>, AMREX_SPACEDIM>& flx_arr,
Source/Advection/ERF_AdvectionSrcForState.cpp:                        const GpuArray<      Array4<Real>, AMREX_SPACEDIM>& flx_tmp_arr,
Source/Advection/ERF_AdvectionSrcForState.cpp:        ParallelFor(xbx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Advection/ERF_AdvectionSrcForState.cpp:        ParallelFor(ybx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Advection/ERF_AdvectionSrcForState.cpp:        ParallelFor(zbx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Advection/ERF_AdvectionSrcForState.cpp:        // Copy flux data to flx_arr to avoid race condition on GPU
Source/Advection/ERF_AdvectionSrcForState.cpp:        ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Advection/ERF_AdvectionSrcForState.cpp:        ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Advection/ERF_AdvectionSrcForState.cpp:        // Copy back to flx_arr to avoid race condition on GPU
Source/Advection/ERF_AdvectionSrcForState.cpp:        ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Advection/ERF_AdvectionSrcForState.cpp:    ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Advection/ERF_AdvectionSrcForScalars.H:                               const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx_arr,
Source/Advection/ERF_AdvectionSrcForScalars.H:    amrex::ParallelFor(xbx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Advection/ERF_AdvectionSrcForScalars.H:    amrex::ParallelFor(ybx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Advection/ERF_AdvectionSrcForScalars.H:    amrex::ParallelFor(zbx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Advection/ERF_AdvectionSrcForScalars.H:                            const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx_arr,
Source/EB/ERF_FlowerIF.H:    : amrex::GPUable
Source/EB/ERF_FlowerIF.H:    AMREX_GPU_HOST_DEVICE inline
Source/EB/ERF_TerrainIF.H:    : amrex::GPUable
Source/EB/ERF_TerrainIF.H:    AMREX_GPU_HOST_DEVICE inline
Source/ERF.cpp:            for (MFIter mfi(vars_new[lev][Vars::cons], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/ERF.cpp:                    ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/ERF.cpp:                    ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/ERF.cpp:            for (MFIter mfi(vars_new[lev][Vars::cons], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/ERF.cpp:                    ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/ERF.cpp:                    ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/ERF.cpp:                    for (MFIter mfi(lev_new[Vars::cons], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/ERF.cpp:                        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/ERF.cpp:        d_rhotheta_src.resize(max_level+1, Gpu::DeviceVector<Real>(0));
Source/ERF.cpp:        d_u_geos.resize(max_level+1, Gpu::DeviceVector<Real>(0));
Source/ERF.cpp:        d_v_geos.resize(max_level+1, Gpu::DeviceVector<Real>(0));
Source/ERF.cpp:        d_rhoqt_src.resize(max_level+1, Gpu::DeviceVector<Real>(0));
Source/ERF.cpp:        d_w_subsid.resize(max_level+1, Gpu::DeviceVector<Real>(0));
Source/ERF.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/ERF.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/ERF.cpp:        Gpu::HostVector<Real> h_avg_qv          = sumToLine(mf,3,1,domain,zdir);
Source/ERF.cpp:        Gpu::HostVector<Real> h_avg_qc          = sumToLine(mf,4,1,domain,zdir);
Source/ERF.cpp:    Gpu::HostVector<Real> h_avg_density     = sumToLine(mf,0,1,domain,zdir);
Source/ERF.cpp:    Gpu::HostVector<Real> h_avg_temperature = sumToLine(mf,1,1,domain,zdir);
Source/ERF.cpp:    Gpu::HostVector<Real> h_avg_pressure    = sumToLine(mf,2,1,domain,zdir);
Source/ERF.cpp:    Gpu::copy(Gpu::hostToDevice, h_havg_density.begin(), h_havg_density.end(), d_havg_density.begin());
Source/ERF.cpp:    Gpu::copy(Gpu::hostToDevice, h_havg_temperature.begin(), h_havg_temperature.end(), d_havg_temperature.begin());
Source/ERF.cpp:    Gpu::copy(Gpu::hostToDevice, h_havg_pressure.begin(), h_havg_pressure.end(), d_havg_pressure.begin());
Source/ERF.cpp:        Gpu::copy(Gpu::hostToDevice, h_havg_qv.begin(), h_havg_qv.end(), d_havg_qv.begin());
Source/ERF.cpp:        Gpu::copy(Gpu::hostToDevice, h_havg_qc.begin(), h_havg_qc.end(), d_havg_qc.begin());
Source/ERF.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/ERF.cpp:        ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/ERF_make_new_level.cpp:        for ( MFIter mfi(*xflux_imask[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/ERF_make_new_level.cpp:        for ( MFIter mfi(*yflux_imask[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/ERF_make_new_level.cpp:        for ( MFIter mfi(*zflux_imask[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/ERF_Tagging.cpp:            for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Microphysics/Kessler/ERF_Kessler.cpp:        for ( MFIter mfi(*tabs,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Microphysics/Kessler/ERF_Kessler.cpp:            // Expose for GPU
Source/Microphysics/Kessler/ERF_Kessler.cpp:            ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Microphysics/Kessler/ERF_Kessler.cpp:        for ( MFIter mfi(fz, TilingIfNotGPU()); mfi.isValid(); ++mfi ){
Source/Microphysics/Kessler/ERF_Kessler.cpp:            ParallelFor(tbz, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Microphysics/Kessler/ERF_Kessler.cpp:        GpuTuple<Real> max = ParReduce(TypeList<ReduceOpMax>{},
Source/Microphysics/Kessler/ERF_Kessler.cpp:                                       [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
Source/Microphysics/Kessler/ERF_Kessler.cpp:                                       -> GpuTuple<Real>
Source/Microphysics/Kessler/ERF_Kessler.cpp:            for ( MFIter mfi(*tabs, TilingIfNotGPU()); mfi.isValid(); ++mfi ){
Source/Microphysics/Kessler/ERF_Kessler.cpp:                ParallelFor(tbz, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Microphysics/Kessler/ERF_Kessler.cpp:                ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Microphysics/Kessler/ERF_Kessler.cpp:        for ( MFIter mfi(*tabs,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Microphysics/Kessler/ERF_Kessler.cpp:            // Expose for GPU
Source/Microphysics/Kessler/ERF_Kessler.cpp:            ParallelFor(box3d, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Microphysics/Kessler/ERF_Update_Kessler.cpp:    for ( MFIter mfi(cons,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Microphysics/Kessler/ERF_Update_Kessler.cpp:        ParallelFor( box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Microphysics/Kessler/ERF_Init_Kessler.cpp:#include <AMReX_GpuContainers.H>
Source/Microphysics/Kessler/ERF_Init_Kessler.cpp:        ParallelFor( box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Microphysics/SAM/ERF_SAM.H:    AMREX_GPU_HOST_DEVICE
Source/Microphysics/SAM/ERF_Update_SAM.cpp:    for ( MFIter mfi(cons,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Microphysics/SAM/ERF_Update_SAM.cpp:        ParallelFor( box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Microphysics/SAM/ERF_IceFall.cpp:        ParallelFor(box3d, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Microphysics/SAM/ERF_IceFall.cpp:    GpuTuple<Real> max = ParReduce(TypeList<ReduceOpMax>{},
Source/Microphysics/SAM/ERF_IceFall.cpp:                         [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
Source/Microphysics/SAM/ERF_IceFall.cpp:                         -> GpuTuple<Real>
Source/Microphysics/SAM/ERF_IceFall.cpp:            ParallelFor(tbz, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Microphysics/SAM/ERF_IceFall.cpp:            ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Microphysics/SAM/ERF_Cloud_SAM.cpp:    for ( MFIter mfi(*(mic_fab_vars[MicVar::tabs]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Microphysics/SAM/ERF_Cloud_SAM.cpp:        ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Microphysics/SAM/ERF_Precip.cpp:    for ( MFIter mfi(*(mic_fab_vars[MicVar::tabs]),TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Microphysics/SAM/ERF_Precip.cpp:        ParallelFor(box3d, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Microphysics/SAM/ERF_Init_SAM.cpp:#include <AMReX_GpuContainers.H>
Source/Microphysics/SAM/ERF_Init_SAM.cpp:        ParallelFor( box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Microphysics/SAM/ERF_Init_SAM.cpp:    Gpu::HostVector<Real> rho_h(ncell), theta_h(ncell), qv_h(ncell);
Source/Microphysics/SAM/ERF_Init_SAM.cpp:    Gpu::DeviceVector<Real> rho_d(ncell), theta_d(ncell), qv_d(ncell);
Source/Microphysics/SAM/ERF_Init_SAM.cpp:    Gpu::copyAsync(Gpu::hostToDevice, rho_h.begin(), rho_h.end(), rho_d.begin());
Source/Microphysics/SAM/ERF_Init_SAM.cpp:    Gpu::copyAsync(Gpu::hostToDevice, theta_h.begin(), theta_h.end(), theta_d.begin());
Source/Microphysics/SAM/ERF_Init_SAM.cpp:    Gpu::copyAsync(Gpu::hostToDevice, qv_h.begin(), qv_h.end(), qv_d.begin());
Source/Microphysics/SAM/ERF_Init_SAM.cpp:    Gpu::streamSynchronize();
Source/Microphysics/SAM/ERF_Init_SAM.cpp:    ParallelFor(nlev, [=] AMREX_GPU_DEVICE (int k) noexcept
Source/Microphysics/SAM/ERF_Init_SAM.cpp:    ParallelFor(nlev, [=] AMREX_GPU_DEVICE (int k) noexcept
Source/Microphysics/SAM/ERF_PrecipFall.cpp:    for (MFIter mfi(fz, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Microphysics/SAM/ERF_PrecipFall.cpp:        ParallelFor(box3d, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Microphysics/SAM/ERF_PrecipFall.cpp:    GpuTuple<Real> max = ParReduce(TypeList<ReduceOpMax>{},
Source/Microphysics/SAM/ERF_PrecipFall.cpp:                         [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
Source/Microphysics/SAM/ERF_PrecipFall.cpp:                         -> GpuTuple<Real>
Source/Microphysics/SAM/ERF_PrecipFall.cpp:            ParallelFor(tbz, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Microphysics/SAM/ERF_PrecipFall.cpp:            ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/ERF_prob_common.H:                             amrex::Gpu::DeviceVector<amrex::Real>& d_src,
Source/ERF_prob_common.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice, src.begin(), src.end(), d_src.begin());
Source/ERF_prob_common.H:                             amrex::Gpu::DeviceVector<amrex::Real>& d_qsrc,
Source/ERF_prob_common.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice, qsrc.begin(), qsrc.end(), d_qsrc.begin());
Source/ERF_prob_common.H:                         amrex::Gpu::DeviceVector<amrex::Real>& d_wbar,
Source/ERF_prob_common.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice, wbar.begin(), wbar.end(), d_wbar.begin());
Source/ERF_prob_common.H:                         amrex::Gpu::DeviceVector<amrex::Real>& d_u_geos,
Source/ERF_prob_common.H:                         amrex::Gpu::DeviceVector<amrex::Real>& d_v_geos,
Source/ERF_prob_common.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice, u_geos.begin(), u_geos.end(), d_u_geos.begin());
Source/ERF_prob_common.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice, v_geos.begin(), v_geos.end(), d_v_geos.begin());
Source/ERF_prob_common.H:            for ( amrex::MFIter mfi(z_phys_nd, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/ERF_prob_common.H:                    ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int) {
Source/ERF_prob_common.H:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int) {
Source/ERF_prob_common.H:        amrex::Gpu::HostVector<amrex::Real> m_xterrain,m_yterrain,m_zterrain;
Source/ERF_prob_common.H:        // Copy data to the GPU
Source/ERF_prob_common.H:        amrex::Gpu::DeviceVector<amrex::Real> d_xterrain(nnode),d_yterrain(nnode),d_zterrain(nnode);
Source/ERF_prob_common.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_xterrain.begin(), m_xterrain.end(), d_xterrain.begin());
Source/ERF_prob_common.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_yterrain.begin(), m_yterrain.end(), d_yterrain.begin());
Source/ERF_prob_common.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_zterrain.begin(), m_zterrain.end(), d_zterrain.begin());
Source/ERF_prob_common.H:        for (amrex::MFIter mfi(z_phys_nd,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/ERF_prob_common.H:            amrex::ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
Source/ERF_prob_common.H:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/LandSurfaceModel/SLM/ERF_SLM.cpp:    // Expose for GPU copy
Source/LandSurfaceModel/SLM/ERF_SLM.cpp:        ParallelFor( box2d, [=] AMREX_GPU_DEVICE (int i, int j, int )
Source/LandSurfaceModel/SLM/ERF_SLM.cpp:    // Expose for GPU copy
Source/LandSurfaceModel/SLM/ERF_SLM.cpp:        ParallelFor( box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/LandSurfaceModel/SLM/ERF_SLM.cpp:    // Expose for GPU copy
Source/LandSurfaceModel/SLM/ERF_SLM.cpp:        ParallelFor( box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/LandSurfaceModel/MM5/ERF_MM5.cpp:    // Expose for GPU copy
Source/LandSurfaceModel/MM5/ERF_MM5.cpp:        ParallelFor( box2d, [=] AMREX_GPU_DEVICE (int i, int j, int )
Source/LandSurfaceModel/MM5/ERF_MM5.cpp:    // Expose for GPU copy
Source/LandSurfaceModel/MM5/ERF_MM5.cpp:        ParallelFor( box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/LandSurfaceModel/MM5/ERF_MM5.cpp:    // Expose for GPU copy
Source/LandSurfaceModel/MM5/ERF_MM5.cpp:        ParallelFor( box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Radiation/ERF_Rrtmgp.H: * code using C++ YAKL for CUDA, HiP and SYCL application by E3SM ECP team, the C++ version
Source/Radiation/ERF_Radiation.H: * code using C++ YAKL for CUDA, HiP and SYCL application by E3SM ECP team, the C++ version
Source/Radiation/ERF_Modal_aero_wateruptake.H:#include <AMReX_GpuComplex.H>
Source/Radiation/ERF_Modal_aero_wateruptake.H:            amrex::GpuComplex<real> cx4[4];
Source/Radiation/ERF_Modal_aero_wateruptake.H:                amrex::GpuComplex<real> cx3[3];
Source/Radiation/ERF_Modal_aero_wateruptake.H:    void makoh_cubic (amrex::GpuComplex<real> cx[],
Source/Radiation/ERF_Modal_aero_wateruptake.H:        auto ci = amrex::GpuComplex<real>(0., 1.);
Source/Radiation/ERF_Modal_aero_wateruptake.H:        auto sqrt3 = amrex::GpuComplex<real>(std::sqrt(3.), 0.);
Source/Radiation/ERF_Modal_aero_wateruptake.H:            cx[0] = amrex::GpuComplex<real>(std::pow(-p0, third), 0.);
Source/Radiation/ERF_Modal_aero_wateruptake.H:            auto q = amrex::GpuComplex<real>(p1/3., 0.);
Source/Radiation/ERF_Modal_aero_wateruptake.H:            auto r = amrex::GpuComplex<real>(p0/2., 0.);
Source/Radiation/ERF_Modal_aero_wateruptake.H:    void makoh_quartic (amrex::GpuComplex<real> cx[],
Source/Radiation/ERF_Modal_aero_wateruptake.H:        auto crad = amrex::sqrt(amrex::GpuComplex<real>(r*r+q*q*q, 0.));
Source/Radiation/ERF_Modal_aero_wateruptake.H:            cx[0] = amrex::pow(amrex::GpuComplex<real>(-p1, 0.0), third);
Source/Radiation/ERF_Radiation.cpp: * code using C++ YAKL for CUDA, HiP and SYCL application by E3SM ECP team, the C++ version
Source/Radiation/ERF_Radiation.cpp:#include <AMReX_GpuContainers.H>
Source/Radiation/ERF_Radiation.cpp:        ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Radiation/ERF_Radiation.cpp:        amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Radiation/ERF_Radiation.cpp:            amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Radiation/ERF_Radiation.cpp:            amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Radiation/ERF_Radiation.cpp:        amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Radiation/ERF_Radiation.cpp:        amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/IO/ERF_Write1DProfiles_stag.cpp:        Gpu::HostVector<Real> h_avg_u, h_avg_v, h_avg_w;
Source/IO/ERF_Write1DProfiles_stag.cpp:        Gpu::HostVector<Real> h_avg_rho, h_avg_th, h_avg_ksgs, h_avg_Kmv, h_avg_Khv;
Source/IO/ERF_Write1DProfiles_stag.cpp:        Gpu::HostVector<Real> h_avg_qv, h_avg_qc, h_avg_qr, h_avg_wqv, h_avg_wqc, h_avg_wqr, h_avg_qi, h_avg_qs, h_avg_qg;
Source/IO/ERF_Write1DProfiles_stag.cpp:        Gpu::HostVector<Real> h_avg_wthv;
Source/IO/ERF_Write1DProfiles_stag.cpp:        Gpu::HostVector<Real> h_avg_uth, h_avg_vth, h_avg_wth, h_avg_thth;
Source/IO/ERF_Write1DProfiles_stag.cpp:        Gpu::HostVector<Real> h_avg_uu, h_avg_uv, h_avg_uw, h_avg_vv, h_avg_vw, h_avg_ww;
Source/IO/ERF_Write1DProfiles_stag.cpp:        Gpu::HostVector<Real> h_avg_uiuiu, h_avg_uiuiv, h_avg_uiuiw;
Source/IO/ERF_Write1DProfiles_stag.cpp:        Gpu::HostVector<Real> h_avg_p, h_avg_pu, h_avg_pv, h_avg_pw;
Source/IO/ERF_Write1DProfiles_stag.cpp:        Gpu::HostVector<Real> h_avg_tau11, h_avg_tau12, h_avg_tau13, h_avg_tau22, h_avg_tau23, h_avg_tau33;
Source/IO/ERF_Write1DProfiles_stag.cpp:        Gpu::HostVector<Real> h_avg_sgshfx, h_avg_sgsq1fx, h_avg_sgsq2fx, h_avg_sgsdiss; // only output tau_{theta,w} and epsilon for now
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_u   , Gpu::HostVector<Real>& h_avg_v  , Gpu::HostVector<Real>& h_avg_w,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_rho , Gpu::HostVector<Real>& h_avg_th , Gpu::HostVector<Real>& h_avg_ksgs,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_Kmv , Gpu::HostVector<Real>& h_avg_Khv,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_qv  , Gpu::HostVector<Real>& h_avg_qc , Gpu::HostVector<Real>& h_avg_qr,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_wqv , Gpu::HostVector<Real>& h_avg_wqc, Gpu::HostVector<Real>& h_avg_wqr,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_qi  , Gpu::HostVector<Real>& h_avg_qs , Gpu::HostVector<Real>& h_avg_qg,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_uu  , Gpu::HostVector<Real>& h_avg_uv , Gpu::HostVector<Real>& h_avg_uw,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_vv  , Gpu::HostVector<Real>& h_avg_vw , Gpu::HostVector<Real>& h_avg_ww,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_uth , Gpu::HostVector<Real>& h_avg_vth, Gpu::HostVector<Real>& h_avg_wth,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_thth,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_uiuiu, Gpu::HostVector<Real>& h_avg_uiuiv, Gpu::HostVector<Real>& h_avg_uiuiw,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_p,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_pu  , Gpu::HostVector<Real>& h_avg_pv , Gpu::HostVector<Real>& h_avg_pw,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                Gpu::HostVector<Real>& h_avg_wthv)
Source/IO/ERF_Write1DProfiles_stag.cpp:    for ( MFIter mfi(mf_cons,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Write1DProfiles_stag.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Write1DProfiles_stag.cpp:        ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Write1DProfiles_stag.cpp:        for ( MFIter mfi(mf_cons,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Write1DProfiles_stag.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Write1DProfiles_stag.cpp:            ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Write1DProfiles_stag.cpp:ERF::derive_stress_profiles_stag (Gpu::HostVector<Real>& h_avg_tau11, Gpu::HostVector<Real>& h_avg_tau12,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                  Gpu::HostVector<Real>& h_avg_tau13, Gpu::HostVector<Real>& h_avg_tau22,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                  Gpu::HostVector<Real>& h_avg_tau23, Gpu::HostVector<Real>& h_avg_tau33,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                  Gpu::HostVector<Real>& h_avg_hfx3,  Gpu::HostVector<Real>& h_avg_q1fx3,
Source/IO/ERF_Write1DProfiles_stag.cpp:                                  Gpu::HostVector<Real>& h_avg_q2fx3, Gpu::HostVector<Real>& h_avg_diss)
Source/IO/ERF_Write1DProfiles_stag.cpp:    for ( MFIter mfi(mf_out,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Write1DProfiles_stag.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Write1DProfiles_stag.cpp:        ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_WriteBndryPlanes.H:#include "AMReX_Gpu.H"
Source/IO/ERF_ReadFromMetgrid.cpp:    ParallelFor(uubx, [=] AMREX_GPU_DEVICE (int , int , int )
Source/IO/ERF_ReadFromMetgrid.cpp:    ParallelFor(vvbx, [=] AMREX_GPU_DEVICE (int , int , int )
Source/IO/ERF_WriteBndryPlanes.cpp:#include "AMReX_Gpu.H"
Source/IO/ERF_WriteBndryPlanes.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/IO/ERF_WriteBndryPlanes.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept
Source/IO/ERF_WriteBndryPlanes.cpp:            for (MFIter mfi(Temp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_WriteBndryPlanes.cpp:            for (MFIter mfi(Temp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_WriteBndryPlanes.cpp:            for (MFIter mfi(Temp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_WriteBndryPlanes.cpp:                for (MFIter mfi(Temp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_WriteBndryPlanes.cpp:                for (MFIter mfi(Temp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Checkpoint.cpp:            for (MFIter mfi(base_state[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Checkpoint.cpp:                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/IO/ERF_NCWpsFile.H:#ifdef AMREX_USE_GPU
Source/IO/ERF_NCWpsFile.H:#ifdef AMREX_USE_GPU
Source/IO/ERF_NCWpsFile.H:#ifdef AMREX_USE_GPU
Source/IO/ERF_NCWpsFile.H:        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
Source/IO/ERF_console_io.cpp:        << "  GPU              :: "
Source/IO/ERF_console_io.cpp:#ifdef AMREX_USE_GPU
Source/IO/ERF_console_io.cpp:#if defined(AMREX_USE_CUDA)
Source/IO/ERF_console_io.cpp:        << "(Backend: CUDA)"
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:                for (MFIter mfi(dmf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:                for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:                for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:             for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:             for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:                for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:                for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:                for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:                for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:                for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:                for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for (MFIter mfi(mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(xbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for (MFIter mfi(mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(xbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/IO/ERF_Plotfile.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_Plotfile.cpp:            for ( MFIter mfi(mf[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Plotfile.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Plotfile.cpp:            for (MFIter mfi(mf_nd[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/IO/ERF_Plotfile.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/IO/ERF_ReadFromWRFBdy.cpp:#ifdef AMREX_USE_GPU
Source/IO/ERF_ReadFromWRFBdy.cpp:            ParallelFor(bx_u, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/IO/ERF_ReadFromWRFBdy.cpp:            ParallelFor(bx_v, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/IO/ERF_ReadFromWRFBdy.cpp:            ParallelFor(bx_t, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/IO/ERF_ReadFromWRFBdy.cpp:            ParallelFor(bx_qv, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/IO/ERF_SampleData.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_SampleData.H:                for (amrex::MFIter mfi(mf_cc_data, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/IO/ERF_SampleData.H:                    amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_SampleData.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/IO/ERF_SampleData.H:                for (amrex::MFIter mfi(mf_cc_data, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/IO/ERF_SampleData.H:                    amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_NCColumnFile.cpp:  amrex::Gpu::DeviceVector<Real> d_column_data(nheights*3, 0.0);
Source/IO/ERF_NCColumnFile.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/IO/ERF_NCColumnFile.cpp:        Gpu::Atomic::Add(&(ucol[idx_vec]), tmpx);
Source/IO/ERF_NCColumnFile.cpp:        Gpu::Atomic::Add(&(vcol[idx_vec]), tmpy);
Source/IO/ERF_NCColumnFile.cpp:        Gpu::Atomic::Add(&(thetacol[idx_vec]), tmpt);
Source/IO/ERF_NCColumnFile.cpp:  amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_column_data.begin(),
Source/IO/ERF_Write1DProfiles.cpp:        Gpu::HostVector<Real> h_avg_u, h_avg_v, h_avg_w;
Source/IO/ERF_Write1DProfiles.cpp:        Gpu::HostVector<Real> h_avg_rho, h_avg_th, h_avg_ksgs, h_avg_Kmv, h_avg_Khv;
Source/IO/ERF_Write1DProfiles.cpp:        Gpu::HostVector<Real> h_avg_qv, h_avg_qc, h_avg_qr, h_avg_wqv, h_avg_wqc, h_avg_wqr, h_avg_qi, h_avg_qs, h_avg_qg;
Source/IO/ERF_Write1DProfiles.cpp:        Gpu::HostVector<Real> h_avg_wthv;
Source/IO/ERF_Write1DProfiles.cpp:        Gpu::HostVector<Real> h_avg_uth, h_avg_vth, h_avg_wth, h_avg_thth;
Source/IO/ERF_Write1DProfiles.cpp:        Gpu::HostVector<Real> h_avg_uu, h_avg_uv, h_avg_uw, h_avg_vv, h_avg_vw, h_avg_ww;
Source/IO/ERF_Write1DProfiles.cpp:        Gpu::HostVector<Real> h_avg_uiuiu, h_avg_uiuiv, h_avg_uiuiw;
Source/IO/ERF_Write1DProfiles.cpp:        Gpu::HostVector<Real> h_avg_p, h_avg_pu, h_avg_pv, h_avg_pw;
Source/IO/ERF_Write1DProfiles.cpp:        Gpu::HostVector<Real> h_avg_tau11, h_avg_tau12, h_avg_tau13, h_avg_tau22, h_avg_tau23, h_avg_tau33;
Source/IO/ERF_Write1DProfiles.cpp:        Gpu::HostVector<Real> h_avg_sgshfx, h_avg_sgsq1fx, h_avg_sgsq2fx, h_avg_sgsdiss; // only output tau_{theta,w} and epsilon for now
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_u   , Gpu::HostVector<Real>& h_avg_v  , Gpu::HostVector<Real>& h_avg_w,
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_rho , Gpu::HostVector<Real>& h_avg_th , Gpu::HostVector<Real>& h_avg_ksgs,
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_Kmv , Gpu::HostVector<Real>& h_avg_Khv,
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_qv  , Gpu::HostVector<Real>& h_avg_qc , Gpu::HostVector<Real>& h_avg_qr,
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_wqv , Gpu::HostVector<Real>& h_avg_wqc, Gpu::HostVector<Real>& h_avg_wqr,
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_qi  , Gpu::HostVector<Real>& h_avg_qs , Gpu::HostVector<Real>& h_avg_qg,
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_uu  , Gpu::HostVector<Real>& h_avg_uv , Gpu::HostVector<Real>& h_avg_uw,
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_vv  , Gpu::HostVector<Real>& h_avg_vw , Gpu::HostVector<Real>& h_avg_ww,
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_uth , Gpu::HostVector<Real>& h_avg_vth, Gpu::HostVector<Real>& h_avg_wth,
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_thth,
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_uiuiu, Gpu::HostVector<Real>& h_avg_uiuiv, Gpu::HostVector<Real>& h_avg_uiuiw,
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_p,
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_pu  , Gpu::HostVector<Real>& h_avg_pv , Gpu::HostVector<Real>& h_avg_pw,
Source/IO/ERF_Write1DProfiles.cpp:                          Gpu::HostVector<Real>& h_avg_wthv)
Source/IO/ERF_Write1DProfiles.cpp:    Gpu::DeviceVector<Real> d_avg_u(hu_size, Real(0.0));
Source/IO/ERF_Write1DProfiles.cpp:    Gpu::DeviceVector<Real> d_avg_v(hu_size, Real(0.0));
Source/IO/ERF_Write1DProfiles.cpp:    Gpu::DeviceVector<Real> d_avg_w(hu_size, Real(0.0));
Source/IO/ERF_Write1DProfiles.cpp:    Gpu::copy(Gpu::hostToDevice, h_avg_u.begin(), h_avg_u.end(), d_avg_u.begin());
Source/IO/ERF_Write1DProfiles.cpp:    Gpu::copy(Gpu::hostToDevice, h_avg_v.begin(), h_avg_v.end(), d_avg_v.begin());
Source/IO/ERF_Write1DProfiles.cpp:    Gpu::copy(Gpu::hostToDevice, h_avg_w.begin(), h_avg_w.end(), d_avg_w.begin());
Source/IO/ERF_Write1DProfiles.cpp:    for ( MFIter mfi(mf_cons,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Write1DProfiles.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Write1DProfiles.cpp:        for ( MFIter mfi(mf_cons,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Write1DProfiles.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_Write1DProfiles.cpp:ERF::derive_stress_profiles (Gpu::HostVector<Real>& h_avg_tau11, Gpu::HostVector<Real>& h_avg_tau12,
Source/IO/ERF_Write1DProfiles.cpp:                             Gpu::HostVector<Real>& h_avg_tau13, Gpu::HostVector<Real>& h_avg_tau22,
Source/IO/ERF_Write1DProfiles.cpp:                             Gpu::HostVector<Real>& h_avg_tau23, Gpu::HostVector<Real>& h_avg_tau33,
Source/IO/ERF_Write1DProfiles.cpp:                             Gpu::HostVector<Real>& h_avg_hfx3,  Gpu::HostVector<Real>& h_avg_q1fx3,
Source/IO/ERF_Write1DProfiles.cpp:                             Gpu::HostVector<Real>& h_avg_q2fx3, Gpu::HostVector<Real>& h_avg_diss)
Source/IO/ERF_Write1DProfiles.cpp:    for ( MFIter mfi(mf_out,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_Write1DProfiles.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/IO/ERF_ReadFromWRFInput.cpp:    ParallelFor(uubx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/IO/ERF_ReadFromWRFInput.cpp:    ParallelFor(vvbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/IO/ERF_ReadFromWRFInput.cpp:    ParallelFor(wwbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/IO/ERF_ReadBndryPlanes.cpp:#include "AMReX_Gpu.H"
Source/IO/ERF_ReadBndryPlanes.cpp:    GpuArray<GpuArray<Real, AMREX_SPACEDIM*2>, AMREX_SPACEDIM+NBCVAR_max> l_bc_extdir_vals_d;
Source/IO/ERF_ReadBndryPlanes.cpp:                bx, ncomp_for_bc, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/IO/ERF_ReadBndryPlanes.cpp:                        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_ReadBndryPlanes.cpp:                        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_ReadBndryPlanes.cpp:                        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_ReadBndryPlanes.cpp:                        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_ReadBndryPlanes.cpp:                        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_ReadBndryPlanes.cpp:                        bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/IO/ERF_WriteScalarProfiles.cpp:        for ( MFIter mfi(pert_dens,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/IO/ERF_WriteScalarProfiles.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/IO/ERF_WriteScalarProfiles.cpp:    Gpu::HostVector<Real> h_avg_ustar; h_avg_ustar.resize(1);
Source/IO/ERF_WriteScalarProfiles.cpp:    Gpu::HostVector<Real> h_avg_tstar; h_avg_tstar.resize(1);
Source/IO/ERF_WriteScalarProfiles.cpp:    Gpu::HostVector<Real> h_avg_olen; h_avg_olen.resize(1);
Source/IO/ERF_WriteScalarProfiles.cpp:         [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> int
Source/IO/ERF_WriteScalarProfiles.cpp:    if (ParallelDescriptor::UseGpuAwareMpi()) {
Source/IO/ERF_WriteScalarProfiles.cpp:        Gpu::PinnedVector<int> hv(numpts);
Source/IO/ERF_WriteScalarProfiles.cpp:        Gpu::copyAsync(Gpu::deviceToHost, p, p+numpts, hv.data());
Source/IO/ERF_WriteScalarProfiles.cpp:        Gpu::streamSynchronize();
Source/IO/ERF_WriteScalarProfiles.cpp:        Gpu::copyAsync(Gpu::hostToDevice, hv.data(), hv.data()+numpts, p);
Source/IO/ERF_WriteScalarProfiles.cpp:         [=] AMREX_GPU_DEVICE (Long i) -> Long {
Source/IO/ERF_WriteScalarProfiles.cpp:    for (MFIter mfi(tmp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/IO/ERF_WriteScalarProfiles.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/IO/ERF_WriteScalarProfiles.cpp:    ParallelFor(fine_mask, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k) noexcept
Source/IO/ERF_ReadBndryPlanes.H:#include "AMReX_Gpu.H"
Source/Initialization/ERF_init_from_wrfinput.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Initialization/ERF_init_from_wrfinput.cpp:    for ( MFIter mfi(*(lat_m[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/Initialization/ERF_init_from_wrfinput.cpp:        ParallelFor(gtbx, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Initialization/ERF_init_from_wrfinput.cpp:    for ( MFIter mfi(*(lon_m[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/Initialization/ERF_init_from_wrfinput.cpp:        ParallelFor(gtbx, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Initialization/ERF_init_from_wrfinput.cpp:    for ( MFIter mfi(*(lmask_lev[lev][0]), TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/Initialization/ERF_init_from_wrfinput.cpp:        ParallelFor(gtbx, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Initialization/ERF_init_from_wrfinput.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Initialization/ERF_init_from_wrfinput.cpp:    for ( MFIter mfi(*mapfac_u[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Initialization/ERF_init_from_wrfinput.cpp:        for ( MFIter mfi(lev_new[Vars::cons], TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Initialization/ERF_init_from_wrfinput.cpp:        for ( MFIter mfi(lev_new[Vars::cons], TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Initialization/ERF_init_from_wrfinput.cpp:        ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_wrfinput.cpp:        Gpu::HostVector  <Real> MaxMax_h(2,-1.0e16);
Source/Initialization/ERF_init_from_wrfinput.cpp:        Gpu::DeviceVector<Real> MaxMax_d(2);
Source/Initialization/ERF_init_from_wrfinput.cpp:        Gpu::copy(Gpu::hostToDevice, MaxMax_h.begin(), MaxMax_h.end(), MaxMax_d.begin());
Source/Initialization/ERF_init_from_wrfinput.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_wrfinput.cpp:            amrex::Gpu::Atomic::Max(&(mm_d[0]),z_calc);
Source/Initialization/ERF_init_from_wrfinput.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_wrfinput.cpp:            amrex::Gpu::Atomic::Max(&(mm_d[1]),z_calc);
Source/Initialization/ERF_init_from_wrfinput.cpp:        Gpu::copy(Gpu::deviceToHost, MaxMax_d.begin(), MaxMax_d.end(), MaxMax_h.begin());
Source/Initialization/ERF_init_from_wrfinput.cpp:        ParallelFor(z_phys_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Initialization/ERF_init_sponge.cpp:        Gpu::copy(Gpu::hostToDevice, h_sponge_ptrs[lev][Sponge::ubar_sponge].begin(), h_sponge_ptrs[lev][Sponge::ubar_sponge].end(),
Source/Initialization/ERF_init_sponge.cpp:        Gpu::copy(Gpu::hostToDevice, h_sponge_ptrs[lev][Sponge::vbar_sponge].begin(), h_sponge_ptrs[lev][Sponge::vbar_sponge].end(),
Source/Initialization/ERF_init_from_metgrid.cpp:#ifdef AMREX_USE_GPU
Source/Initialization/ERF_init_from_metgrid.cpp:    // Make sure this lives on CPU and GPU
Source/Initialization/ERF_init_from_metgrid.cpp:    for ( MFIter mfi(lev_new[Vars::cons], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/Initialization/ERF_init_from_metgrid.cpp:            for ( MFIter mfi(*(sst_lev[lev][it]), TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/Initialization/ERF_init_from_metgrid.cpp:                ParallelFor(gtbx, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:            for ( MFIter mfi(*(lmask_lev[lev][it]), TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/Initialization/ERF_init_from_metgrid.cpp:                ParallelFor(gtbx, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:    for ( MFIter mfi(*(lat_m[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/Initialization/ERF_init_from_metgrid.cpp:        ParallelFor(gtbx, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:    for ( MFIter mfi(*(lon_m[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/Initialization/ERF_init_from_metgrid.cpp:        ParallelFor(gtbx, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Initialization/ERF_init_from_metgrid.cpp:    for ( MFIter mfi(lev_new[Vars::cons], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/Initialization/ERF_init_from_metgrid.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Initialization/ERF_init_from_metgrid.cpp:    for ( MFIter mfi(*mapfac_u[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/Initialization/ERF_init_from_metgrid.cpp:    for ( MFIter mfi(lev_new[Vars::cons], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/Initialization/ERF_init_from_metgrid.cpp:#ifndef AMREX_USE_GPU
Source/Initialization/ERF_init_from_metgrid.cpp:            ParallelFor(xlo_plane, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:            ParallelFor(xhi_plane, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:            ParallelFor(ylo_plane, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:            ParallelFor(yhi_plane, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:        ParallelFor(bx2d, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:        ParallelFor(bx2d, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:        ParallelFor(bx2d, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:                ParallelFor(bx2d, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:    Gpu::DeviceVector<int>flag_psfc_d(flag_psfc.size());
Source/Initialization/ERF_init_from_metgrid.cpp:    Gpu::copy(Gpu::hostToDevice, flag_psfc.begin(), flag_psfc.end(), flag_psfc_d.begin());
Source/Initialization/ERF_init_from_metgrid.cpp:#ifdef AMREX_USE_GPU
Source/Initialization/ERF_init_from_metgrid.cpp:    // Expose for copy to GPU
Source/Initialization/ERF_init_from_metgrid.cpp:        ParallelFor(valid_bx2d, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:        ParallelFor(valid_bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:        ParallelFor(valid_bx2d, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Initialization/ERF_init_from_metgrid.cpp:#ifndef AMREX_USE_GPU
Source/Initialization/ERF_init_from_metgrid.cpp:#ifndef AMREX_USE_GPU
Source/Initialization/ERF_init_from_metgrid.cpp:#ifndef AMREX_USE_GPU
Source/Initialization/ERF_Metgrid_utils.H:AMREX_GPU_DEVICE
Source/Initialization/ERF_Metgrid_utils.H:AMREX_GPU_DEVICE
Source/Initialization/ERF_init1d.cpp:        ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int)
Source/Initialization/ERF_init_bcs.cpp: * so it is available for GPU kernels.
Source/Initialization/ERF_init_bcs.cpp:    // NOTE: Gpu:copy is a wrapper to htod_memcpy (GPU) or memcpy (CPU) and is a blocking comm
Source/Initialization/ERF_init_bcs.cpp:    Gpu::copy(Gpu::hostToDevice, domain_bcs_type.begin(), domain_bcs_type.end(), domain_bcs_type_d.begin());
Source/Initialization/ERF_init_bcs.cpp:        Gpu::copy(Gpu::hostToDevice, u_inp.begin(), u_inp.end(), xvel_bc_data[lev].begin());
Source/Initialization/ERF_init_bcs.cpp:        Gpu::copy(Gpu::hostToDevice, v_inp.begin(), v_inp.end(), yvel_bc_data[lev].begin());
Source/Initialization/ERF_init_bcs.cpp:        Gpu::copy(Gpu::hostToDevice, w_inp.begin(), w_inp.end(), zvel_bc_data[lev].begin());
Source/Initialization/ERF_init_geowind.cpp:                                Gpu::DeviceVector<Real>& u_geos_d,
Source/Initialization/ERF_init_geowind.cpp:                                Gpu::DeviceVector<Real>& v_geos_d,
Source/Initialization/ERF_init_geowind.cpp:    Gpu::copy(Gpu::hostToDevice, u_geos.begin(), u_geos.end(), u_geos_d.begin());
Source/Initialization/ERF_init_geowind.cpp:    Gpu::copy(Gpu::hostToDevice, v_geos.begin(), v_geos.end(), v_geos_d.begin());
Source/Initialization/ERF_init_rayleigh.cpp:            Gpu::copy(Gpu::hostToDevice, h_rayleigh_ptrs[lev][n].begin(), h_rayleigh_ptrs[lev][n].end(),
Source/Initialization/ERF_init_rayleigh.cpp:        Gpu::copy(Gpu::hostToDevice, h_rayleigh_ptrs[lev][Rayleigh::ubar].begin(), h_rayleigh_ptrs[lev][Rayleigh::ubar].end(),
Source/Initialization/ERF_init_rayleigh.cpp:        Gpu::copy(Gpu::hostToDevice, h_rayleigh_ptrs[lev][Rayleigh::vbar].begin(), h_rayleigh_ptrs[lev][Rayleigh::vbar].end(),
Source/Initialization/ERF_init_rayleigh.cpp:        Gpu::copy(Gpu::hostToDevice, h_rayleigh_ptrs[lev][Rayleigh::wbar].begin(), h_rayleigh_ptrs[lev][Rayleigh::wbar].end(),
Source/Initialization/ERF_init_rayleigh.cpp:        Gpu::copy(Gpu::hostToDevice, h_rayleigh_ptrs[lev][Rayleigh::thetabar].begin(), h_rayleigh_ptrs[lev][Rayleigh::thetabar].end(),
Source/Initialization/ERF_init_custom.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Initialization/ERF_init_TurbPert.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Initialization/ERF_init_uniform.cpp:    for (MFIter mfi(lev_new[Vars::cons], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Initialization/ERF_init_from_input_sounding.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Initialization/ERF_init_from_input_sounding.cpp:    for (MFIter mfi(lev_new[Vars::cons], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Initialization/ERF_init_from_input_sounding.cpp:    ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Initialization/ERF_init_from_input_sounding.cpp:    ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Initialization/ERF_init_from_input_sounding.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Initialization/ERF_init_from_input_sounding.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Initialization/ERF_init_from_input_sounding.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/Initialization/ERF_init_from_hse.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Initialization/ERF_init_from_hse.cpp:        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/PBL/ERF_ComputeDiffusivityYSU.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PBL/ERF_ComputeDiffusivityYSU.cpp:        for ( MFIter mfi(eddyViscosity,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PBL/ERF_ComputeDiffusivityYSU.cpp:            ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/PBL/ERF_ComputeDiffusivityYSU.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/PBL/ERF_ComputeDiffusivityYSU.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/PBL/ERF_MYNNStruct.H:    AMREX_GPU_DEVICE
Source/PBL/ERF_MYNNStruct.H:    AMREX_GPU_DEVICE
Source/PBL/ERF_MYNNStruct.H:    AMREX_GPU_DEVICE
Source/PBL/ERF_MYNNStruct.H:    AMREX_GPU_DEVICE
Source/PBL/ERF_PBLModels.H:AMREX_GPU_DEVICE
Source/PBL/ERF_PBLModels.H:AMREX_GPU_DEVICE
Source/PBL/ERF_PBLModels.H:                       const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/PBL/ERF_ComputeDiffusivityMYNN25.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/PBL/ERF_ComputeDiffusivityMYNN25.cpp:    for ( MFIter mfi(eddyViscosity,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/PBL/ERF_ComputeDiffusivityMYNN25.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/PBL/ERF_ComputeDiffusivityMYNN25.cpp:                Gpu::Atomic::Add(&qint(i,j,0,0), Zval*qvel(i,j,k)*dz*fac);
Source/PBL/ERF_ComputeDiffusivityMYNN25.cpp:                Gpu::Atomic::Add(&qint(i,j,0,1),      qvel(i,j,k)*dz*fac);
Source/PBL/ERF_ComputeDiffusivityMYNN25.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/PBL/ERF_ComputeDiffusivityMYNN25.cpp:                Gpu::Atomic::Add(&qint(i,j,0,0), Zval*qvel(i,j,k)*fac);
Source/PBL/ERF_ComputeDiffusivityMYNN25.cpp:                Gpu::Atomic::Add(&qint(i,j,0,1),      qvel(i,j,k)*fac);
Source/PBL/ERF_ComputeDiffusivityMYNN25.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/PBL/ERF_PBLHeight.H:    AMREX_GPU_HOST
Source/PBL/ERF_PBLHeight.H:            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> amrex::Real
Source/PBL/ERF_PBLHeight.H:                ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/PBL/ERF_PBLHeight.H:                ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/PBL/ERF_PBLHeight.H:                ParallelFor(gtbxlow, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/PBL/ERF_PBLHeight.H:                ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/PBL/ERF_PBLHeight.H:            ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
Source/Utils/ERF_Wstar.H:AMREX_GPU_DEVICE
Source/Utils/ERF_Wstar.H:AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation.H:AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation.H:AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation.H:AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation.H:AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation.H:AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation.H:AMREX_GPU_DEVICE
Source/Utils/ERF_PlaneAverage.H:#include "AMReX_Gpu.H"
Source/Utils/ERF_PlaneAverage.H:#include "AMReX_GpuContainers.H"
Source/Utils/ERF_PlaneAverage.H:    void line_average (int comp, amrex::Gpu::HostVector<amrex::Real>& l_vec);
Source/Utils/ERF_PlaneAverage.H:PlaneAverage::line_average (int comp, amrex::Gpu::HostVector<amrex::Real>& l_vec)
Source/Utils/ERF_PlaneAverage.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Utils/ERF_PlaneAverage.H:    for (amrex::MFIter mfi(mfab, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Utils/ERF_PlaneAverage.H:        amrex::ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), pbx, [=]
Source/Utils/ERF_PlaneAverage.H:                    AMREX_GPU_DEVICE( int p_i, int p_j, int p_k,
Source/Utils/ERF_PlaneAverage.H:                                      amrex::Gpu::Handler const& handler) noexcept
Source/Utils/ERF_PlaneAverage.H:                            //       This more performant than Gpu::Atomic::Add.
Source/Utils/ERF_PlaneAverage.H:                            amrex::Gpu::deviceReduceSum(&line_avg[ncomp * ind + n],
Source/Utils/ERF_DirectionSelector.H:#include "AMReX_Gpu.H"
Source/Utils/ERF_DirectionSelector.H:    [[nodiscard]] AMREX_GPU_HOST_DEVICE static int getIndx (int i, int, int) { return i; }
Source/Utils/ERF_DirectionSelector.H:    [[nodiscard]] AMREX_GPU_HOST_DEVICE static int getIndx (int, int j, int) { return j; }
Source/Utils/ERF_DirectionSelector.H:    [[nodiscard]] AMREX_GPU_HOST_DEVICE static int getIndx (int, int, int k) { return k; }
Source/Utils/ERF_DirectionSelector.H:AMREX_GPU_HOST_DEVICE amrex::Box
Source/Utils/ERF_DirectionSelector.H:AMREX_GPU_HOST_DEVICE amrex::Box
Source/Utils/ERF_Microphysics_Utils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Microphysics_Utils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Microphysics_Utils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Microphysics_Utils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Microphysics_Utils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Microphysics_Utils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Microphysics_Utils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Microphysics_Utils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Microphysics_Utils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Microphysics_Utils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Microphysics_Utils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Microphysics_Utils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Microphysics_Utils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Thetav.H:AMREX_GPU_DEVICE
Source/Utils/ERF_Time_Avg_Vel.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_Time_Avg_Vel.cpp:    for ( MFIter mfi(*(vel_t_avg),TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_Time_Avg_Vel.cpp:        ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Utils/ERF_TerrainMetrics.H:                        amrex::GpuArray<ERF_BC, AMREX_SPACEDIM*2>& phys_bc_type);
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                           const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                       const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                       const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                       const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                              const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                              const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                              const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv)
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv)
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv)
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv,
Source/Utils/ERF_TerrainMetrics.H:AMREX_GPU_DEVICE
Source/Utils/ERF_TerrainMetrics.H:                      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv,
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Sat_methods.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Utils.H:AMREX_GPU_HOST
Source/Utils/ERF_Utils.H:    amrex::ParallelFor(bx_xlo, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/ERF_Utils.H:    bx_xhi, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/ERF_Utils.H:    amrex::ParallelFor(bx_ylo, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/ERF_Utils.H:    bx_yhi, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/ERF_Utils.H:AMREX_GPU_HOST
Source/Utils/ERF_Utils.H:    amrex::ParallelFor(bx_xlo, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/ERF_Utils.H:    bx_xhi, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/ERF_Utils.H:    amrex::ParallelFor(bx_ylo, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/ERF_Utils.H:    bx_yhi, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/ERF_Utils.H:AMREX_GPU_HOST
Source/Utils/ERF_Utils.H:    for (amrex::MFIter mfi(dst,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_Utils.H:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_Utils.H:AMREX_GPU_HOST
Source/Utils/ERF_Utils.H:    for (amrex::MFIter mfi(dst,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_Utils.H:            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_TileNoZ.H:    if (amrex::TilingIfNotGPU()) {
Source/Utils/ERF_Orbit.H:AMREX_GPU_HOST
Source/Utils/ERF_Orbit.H:AMREX_GPU_HOST
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO_Z.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_HSE_utils.H:    AMREX_GPU_HOST_DEVICE
Source/Utils/ERF_HSE_utils.H:    AMREX_GPU_HOST_DEVICE
Source/Utils/ERF_HSE_utils.H:    AMREX_GPU_HOST_DEVICE
Source/Utils/ERF_InteriorGhostCells.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_InteriorGhostCells.cpp:        for ( MFIter mfi(S_cur_data[ivar_idx],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Utils/ERF_InteriorGhostCells.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_InteriorGhostCells.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_InteriorGhostCells.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_InteriorGhostCells.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_InteriorGhostCells.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_InteriorGhostCells.cpp:            for ( MFIter mfi(S_old_data[ivar_idx],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Utils/ERF_InteriorGhostCells.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_InteriorGhostCells.cpp:            for ( MFIter mfi(S_cur_data[ivar_idx],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Utils/ERF_InteriorGhostCells.cpp:                ParallelFor(tbx_xlo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_InteriorGhostCells.cpp:                ParallelFor(tbx_xhi, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_InteriorGhostCells.cpp:                ParallelFor(tbx_ylo, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_InteriorGhostCells.cpp:                ParallelFor(tbx_yhi, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_InteriorGhostCells.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_InteriorGhostCells.cpp:            for ( MFIter mfi(fmf_p,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_InteriorGhostCells.cpp:                ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_InteriorGhostCells.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_InteriorGhostCells.cpp:            for ( MFIter mfi(fmf_p,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_InteriorGhostCells.cpp:                ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_InteriorGhostCells.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_InteriorGhostCells.cpp:            for ( MFIter mfi(fmf_p,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_InteriorGhostCells.cpp:                ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_InteriorGhostCells.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_InteriorGhostCells.cpp:        for ( MFIter mfi(rhs,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_InteriorGhostCells.cpp:            ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_InteriorGhostCells.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_InteriorGhostCells.cpp:        for ( MFIter mfi(fmf_p,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_InteriorGhostCells.cpp:            ParallelFor(vbx, num_var, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/ERF_MomentumToVelocity.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Utils/ERF_MomentumToVelocity.cpp:    for ( MFIter mfi(density,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_MomentumToVelocity.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_MomentumToVelocity.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_MomentumToVelocity.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_MomentumToVelocity.cpp:            ParallelFor(makeSlab(tbx,0,domain.smallEnd(0)), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_MomentumToVelocity.cpp:            ParallelFor(makeSlab(tbx,0,domain.bigEnd(0)+1), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_MomentumToVelocity.cpp:            ParallelFor(makeSlab(tby,1,domain.smallEnd(1)), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_MomentumToVelocity.cpp:            ParallelFor(makeSlab(tby,1,domain.bigEnd(1)+1), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_AverageDown.cpp:      for (MFIter mfi(vars_new[lev][Vars::cons], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Utils/ERF_AverageDown.cpp:            ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/ERF_AverageDown.cpp:            ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/ERF_AverageDown.cpp:      for (MFIter mfi(vars_new[lev][Vars::cons], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Utils/ERF_AverageDown.cpp:            ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/ERF_AverageDown.cpp:            ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/ERF_PoissonSolve.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_PoissonSolve.cpp:    for (MFIter mfi(phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_PoissonSolve.cpp:        ParallelFor(xbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_PoissonSolve.cpp:        ParallelFor(ybx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_PoissonSolve.cpp:        ParallelFor(zbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_PoissonSolve.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_PoissonSolve.cpp:    for (MFIter mfi(mom_mf[Vars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_PoissonSolve.cpp:                ParallelFor(makeSlab(bx,0,dom_lo.x), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_PoissonSolve.cpp:                ParallelFor(makeSlab(bx,0,dom_lo.x), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_PoissonSolve.cpp:                ParallelFor(makeSlab(bx,1,dom_lo.y), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_PoissonSolve.cpp:                ParallelFor(makeSlab(bx,1,dom_lo.y), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_PoissonSolve.cpp:            ParallelFor(makeSlab(bx,2,dom_lo.z), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_PoissonSolve.cpp:                ParallelFor(makeSlab(bx,0,dom_hi.x), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_PoissonSolve.cpp:                ParallelFor(makeSlab(bx,0,dom_hi.x), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_PoissonSolve.cpp:                ParallelFor(makeSlab(bx,1,dom_hi.y), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_PoissonSolve.cpp:                ParallelFor(makeSlab(bx,1,dom_hi.y), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_PoissonSolve.cpp:            ParallelFor(makeSlab(bx,2,dom_hi.z), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/ERF_TerrainMetrics.cpp:                   GpuArray<ERF_BC, AMREX_SPACEDIM*2>& phys_bc_type)
Source/Utils/ERF_TerrainMetrics.cpp:                ParallelFor(makeSlab(bx,0,1), [=] AMREX_GPU_DEVICE (int , int j, int k)
Source/Utils/ERF_TerrainMetrics.cpp:                ParallelFor(makeSlab(bx,0,1), [=] AMREX_GPU_DEVICE (int , int j, int k)
Source/Utils/ERF_TerrainMetrics.cpp:                ParallelFor(makeSlab(bx,1,1), [=] AMREX_GPU_DEVICE (int i, int  , int k)
Source/Utils/ERF_TerrainMetrics.cpp:                ParallelFor(makeSlab(bx,1,1), [=] AMREX_GPU_DEVICE (int i, int  , int k)
Source/Utils/ERF_TerrainMetrics.cpp:    for ( MFIter mfi(z_phys_nd, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Utils/ERF_TerrainMetrics.cpp:            ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int ) {
Source/Utils/ERF_TerrainMetrics.cpp:    for ( MFIter mfi(z_phys_nd, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Utils/ERF_TerrainMetrics.cpp:        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_TerrainMetrics.cpp:    Gpu::DeviceVector<Real> z_levels_d;
Source/Utils/ERF_TerrainMetrics.cpp:    Gpu::copy(Gpu::hostToDevice, z_levels_h.begin(), z_levels_h.end(), z_levels_d.begin());
Source/Utils/ERF_TerrainMetrics.cpp:        for ( MFIter mfi(z_phys_nd, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Utils/ERF_TerrainMetrics.cpp:            ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Utils/ERF_TerrainMetrics.cpp:        for ( MFIter mfi(z_phys_nd, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Utils/ERF_TerrainMetrics.cpp:                ParallelFor(makeSlab(gbx,2,0), [=] AMREX_GPU_DEVICE (int i, int j, int)
Source/Utils/ERF_TerrainMetrics.cpp:                    [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int) noexcept
Source/Utils/ERF_TerrainMetrics.cpp:                        -> GpuTuple<Real>
Source/Utils/ERF_TerrainMetrics.cpp:                    [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int) noexcept
Source/Utils/ERF_TerrainMetrics.cpp:                        -> GpuTuple<Real>
Source/Utils/ERF_TerrainMetrics.cpp:            for ( MFIter mfi(z_phys_nd, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Utils/ERF_TerrainMetrics.cpp:                ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int) {
Source/Utils/ERF_TerrainMetrics.cpp:            Gpu::streamSynchronize();
Source/Utils/ERF_TerrainMetrics.cpp:            for ( MFIter mfi(z_phys_nd, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Utils/ERF_TerrainMetrics.cpp:                ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Utils/ERF_TerrainMetrics.cpp:            for ( MFIter mfi(z_phys_nd, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Utils/ERF_TerrainMetrics.cpp:                ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Utils/ERF_TerrainMetrics.cpp:                ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Utils/ERF_TerrainMetrics.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_TerrainMetrics.cpp:    for ( MFIter mfi(detJ_cc, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Utils/ERF_TerrainMetrics.cpp:        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Utils/ERF_TerrainMetrics.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_TerrainMetrics.cpp:    for ( MFIter mfi(ax, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Utils/ERF_TerrainMetrics.cpp:        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Utils/ERF_TerrainMetrics.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_TerrainMetrics.cpp:    for ( MFIter mfi(ay, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Utils/ERF_TerrainMetrics.cpp:        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Utils/ERF_TerrainMetrics.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/ERF_TerrainMetrics.cpp:    for ( MFIter mfi(z_phys_cc, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Utils/ERF_TerrainMetrics.cpp:        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Utils/ERF_Water_vapor_saturation.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Water_vapor_saturation.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Water_vapor_saturation.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Water_vapor_saturation.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Water_vapor_saturation.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Water_vapor_saturation.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Water_vapor_saturation.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Water_vapor_saturation.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Water_vapor_saturation.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Water_vapor_saturation.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Water_vapor_saturation.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Interpolation_1D.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_VelocityToMomentum.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Utils/ERF_VelocityToMomentum.cpp:    for ( MFIter mfi(density,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_VelocityToMomentum.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_VelocityToMomentum.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_VelocityToMomentum.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_VelocityToMomentum.cpp:            ParallelFor(makeSlab(tbx,0,domain.smallEnd(0)), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_VelocityToMomentum.cpp:            ParallelFor(makeSlab(tbx,0,domain.bigEnd(0)+1), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_VelocityToMomentum.cpp:            ParallelFor(makeSlab(tby,1,domain.smallEnd(1)), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_VelocityToMomentum.cpp:            ParallelFor(makeSlab(tby,1,domain.bigEnd(1)+1), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Utils/ERF_PoissonSolve_tb.cpp:            for (MFIter mfi(rhs[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_PoissonSolve_tb.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Utils/ERF_PoissonSolve_tb.cpp:        //        for (MFIter mfi(rhs[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/ERF_PoissonSolve_tb.cpp:        //            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Utils/ERF_Orbit.cpp:            ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
Source/Utils/ERF_EOS.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_EOS.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_EOS.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_EOS.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_EOS.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_EOS.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_EOS.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_EOS.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_EOS.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_EOS.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_EOS.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_EOS.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/ERF_Interpolation_WENO.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_WENO.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_ParFunctions.H:        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int) noexcept
Source/Utils/ERF_ParFunctions.H:            -> amrex::GpuTuple<amrex::Real>
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/Utils/ERF_Interpolation_UPW.H:    AMREX_GPU_DEVICE
Source/WindFarmParametrization/EWP/ERF_AdvanceEWP.cpp:    for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/WindFarmParametrization/EWP/ERF_AdvanceEWP.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/WindFarmParametrization/EWP/ERF_AdvanceEWP.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/WindFarmParametrization/EWP/ERF_AdvanceEWP.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/WindFarmParametrization/EWP/ERF_AdvanceEWP.cpp:  Gpu::DeviceVector<Real> d_wind_speed(wind_speed.size());
Source/WindFarmParametrization/EWP/ERF_AdvanceEWP.cpp:  Gpu::DeviceVector<Real> d_thrust_coeff(thrust_coeff.size());
Source/WindFarmParametrization/EWP/ERF_AdvanceEWP.cpp:  Gpu::copy(Gpu::hostToDevice, wind_speed.begin(), wind_speed.end(), d_wind_speed.begin());
Source/WindFarmParametrization/EWP/ERF_AdvanceEWP.cpp:  Gpu::copy(Gpu::hostToDevice, thrust_coeff.begin(), thrust_coeff.end(), d_thrust_coeff.begin());
Source/WindFarmParametrization/EWP/ERF_AdvanceEWP.cpp:  for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/WindFarmParametrization/EWP/ERF_AdvanceEWP.cpp:        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/WindFarmParametrization/Fitch/ERF_AdvanceFitch.cpp:AMREX_GPU_DEVICE
Source/WindFarmParametrization/Fitch/ERF_AdvanceFitch.cpp:AMREX_GPU_DEVICE
Source/WindFarmParametrization/Fitch/ERF_AdvanceFitch.cpp:    for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/WindFarmParametrization/Fitch/ERF_AdvanceFitch.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/WindFarmParametrization/Fitch/ERF_AdvanceFitch.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/WindFarmParametrization/Fitch/ERF_AdvanceFitch.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/WindFarmParametrization/Fitch/ERF_AdvanceFitch.cpp:  Gpu::DeviceVector<Real> d_wind_speed(wind_speed.size());
Source/WindFarmParametrization/Fitch/ERF_AdvanceFitch.cpp:  Gpu::DeviceVector<Real> d_thrust_coeff(thrust_coeff.size());
Source/WindFarmParametrization/Fitch/ERF_AdvanceFitch.cpp:  Gpu::copy(Gpu::hostToDevice, wind_speed.begin(), wind_speed.end(), d_wind_speed.begin());
Source/WindFarmParametrization/Fitch/ERF_AdvanceFitch.cpp:  Gpu::copy(Gpu::hostToDevice, thrust_coeff.begin(), thrust_coeff.end(), d_thrust_coeff.begin());
Source/WindFarmParametrization/Fitch/ERF_AdvanceFitch.cpp:  for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/WindFarmParametrization/Fitch/ERF_AdvanceFitch.cpp:        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/WindFarmParametrization/Fitch/ERF_AdvanceFitch.cpp:                 //amrex::Gpu::Atomic::Add(sum_area, A_ijk);
Source/WindFarmParametrization/ERF_InitWindFarm.cpp:    amrex::Gpu::DeviceVector<Real> d_xloc(xloc.size());
Source/WindFarmParametrization/ERF_InitWindFarm.cpp:    amrex::Gpu::DeviceVector<Real> d_yloc(yloc.size());
Source/WindFarmParametrization/ERF_InitWindFarm.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, xloc.begin(), xloc.end(), d_xloc.begin());
Source/WindFarmParametrization/ERF_InitWindFarm.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, yloc.begin(), yloc.end(), d_yloc.begin());
Source/WindFarmParametrization/ERF_InitWindFarm.cpp:    for ( MFIter mfi(mf_Nturb,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/WindFarmParametrization/ERF_InitWindFarm.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/WindFarmParametrization/ERF_InitWindFarm.cpp:    amrex::Gpu::DeviceVector<Real> d_xloc(xloc.size());
Source/WindFarmParametrization/ERF_InitWindFarm.cpp:    amrex::Gpu::DeviceVector<Real> d_yloc(yloc.size());
Source/WindFarmParametrization/ERF_InitWindFarm.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, xloc.begin(), xloc.end(), d_xloc.begin());
Source/WindFarmParametrization/ERF_InitWindFarm.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, yloc.begin(), yloc.end(), d_yloc.begin());
Source/WindFarmParametrization/ERF_InitWindFarm.cpp:    for ( MFIter mfi(mf_SMark,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/WindFarmParametrization/ERF_InitWindFarm.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:     Gpu::DeviceVector<Real> d_freestream_velocity(xloc.size());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:     Gpu::DeviceVector<Real> d_freestream_phi(yloc.size());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:     Gpu::DeviceVector<Real> d_disk_cell_count(yloc.size());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:     Gpu::copy(Gpu::hostToDevice, freestream_velocity.begin(), freestream_velocity.end(), d_freestream_velocity.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:     Gpu::copy(Gpu::hostToDevice, freestream_phi.begin(), freestream_phi.end(), d_freestream_phi.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:     Gpu::copy(Gpu::hostToDevice, disk_cell_count.begin(), disk_cell_count.end(), d_disk_cell_count.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:     for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:        ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:                Gpu::Atomic::Add(&d_freestream_velocity_ptr[turb_index],std::pow(u_vel(i,j,k)*u_vel(i,j,k) + v_vel(i,j,k)*v_vel(i,j,k),0.5));
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:                Gpu::Atomic::Add(&d_disk_cell_count_ptr[turb_index],1.0);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:                Gpu::Atomic::Add(&d_freestream_phi_ptr[turb_index],phi);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::copy(Gpu::deviceToHost, d_freestream_velocity.begin(), d_freestream_velocity.end(), freestream_velocity.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::copy(Gpu::deviceToHost, d_freestream_phi.begin(), d_freestream_phi.end(), freestream_phi.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::copy(Gpu::deviceToHost, d_disk_cell_count.begin(), d_disk_cell_count.end(), disk_cell_count.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:AMREX_GPU_DEVICE
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:AMREX_GPU_DEVICE
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::DeviceVector<Real> d_xloc(xloc.size());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::DeviceVector<Real> d_yloc(yloc.size());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::copy(Gpu::hostToDevice, xloc.begin(), xloc.end(), d_xloc.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::copy(Gpu::hostToDevice, yloc.begin(), yloc.end(), d_yloc.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::DeviceVector<Real> d_freestream_velocity(nturbs);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::DeviceVector<Real> d_disk_cell_count(nturbs);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::copy(Gpu::hostToDevice, freestream_velocity.begin(), freestream_velocity.end(), d_freestream_velocity.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::copy(Gpu::hostToDevice, disk_cell_count.begin(), disk_cell_count.end(), d_disk_cell_count.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::DeviceVector<Real>    d_bld_rad_loc(n_bld_sections);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::DeviceVector<Real>    d_bld_twist(n_bld_sections);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::DeviceVector<Real>    d_bld_chord(n_bld_sections);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::copy(Gpu::hostToDevice, bld_rad_loc.begin(), bld_rad_loc.end(), d_bld_rad_loc.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::copy(Gpu::hostToDevice, bld_twist.begin(), bld_twist.end(), d_bld_twist.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::copy(Gpu::hostToDevice, bld_chord.begin(), bld_chord.end(), d_bld_chord.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Vector<Gpu::DeviceVector<Real>> d_bld_airfoil_aoa(n_bld_sections);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Vector<Gpu::DeviceVector<Real>> d_bld_airfoil_Cl(n_bld_sections);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Vector<Gpu::DeviceVector<Real>> d_bld_airfoil_Cd(n_bld_sections);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:        Gpu::copy(Gpu::hostToDevice, bld_airfoil_aoa[i].begin(), bld_airfoil_aoa[i].end(), d_bld_airfoil_aoa[i].begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:        Gpu::copy(Gpu::hostToDevice, bld_airfoil_Cl[i].begin(), bld_airfoil_Cl[i].end(), d_bld_airfoil_Cl[i].begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:        Gpu::copy(Gpu::hostToDevice, bld_airfoil_Cd[i].begin(), bld_airfoil_Cd[i].end(), d_bld_airfoil_Cd[i].begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::AsyncArray<Real*> aoa(hp_bld_airfoil_aoa.data(), n_bld_sections);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::AsyncArray<Real*> Cl(hp_bld_airfoil_Cl.data(), n_bld_sections);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::AsyncArray<Real*> Cd(hp_bld_airfoil_Cd.data(), n_bld_sections);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::DeviceVector<Real> d_velocity(n_spec_extra);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::DeviceVector<Real> d_rotor_RPM(n_spec_extra);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::DeviceVector<Real> d_blade_pitch(n_spec_extra);
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::copy(Gpu::hostToDevice, velocity.begin(), velocity.end(), d_velocity.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::copy(Gpu::hostToDevice, rotor_RPM.begin(), rotor_RPM.end(), d_rotor_RPM.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    Gpu::copy(Gpu::hostToDevice, blade_pitch.begin(), blade_pitch.end(), d_blade_pitch.begin());
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:    for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/WindFarmParametrization/GeneralActuatorDisk/ERF_AdvanceGeneralAD.cpp:        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/WindFarmParametrization/Null/ERF_NullWindFarm.H:#include <AMReX_Gpu.H>
Source/WindFarmParametrization/Null/ERF_NullWindFarm.H:static AMREX_GPU_DEVICE
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:    for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:     Gpu::DeviceVector<Real> d_freestream_velocity(xloc.size());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:     Gpu::DeviceVector<Real> d_freestream_phi(yloc.size());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:     Gpu::DeviceVector<Real> d_disk_cell_count(yloc.size());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:     Gpu::copy(Gpu::hostToDevice, freestream_velocity.begin(), freestream_velocity.end(), d_freestream_velocity.begin());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:     Gpu::copy(Gpu::hostToDevice, freestream_phi.begin(), freestream_phi.end(), d_freestream_phi.begin());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:     Gpu::copy(Gpu::hostToDevice, disk_cell_count.begin(), disk_cell_count.end(), d_disk_cell_count.begin());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:     for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:        ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:                Gpu::Atomic::Add(&d_freestream_velocity_ptr[turb_index],std::pow(u_vel(i,j,k)*u_vel(i,j,k) + v_vel(i,j,k)*v_vel(i,j,k),0.5));
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:                Gpu::Atomic::Add(&d_disk_cell_count_ptr[turb_index],1.0);
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:                Gpu::Atomic::Add(&d_freestream_phi_ptr[turb_index],phi);
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:    Gpu::copy(Gpu::deviceToHost, d_freestream_velocity.begin(), d_freestream_velocity.end(), freestream_velocity.begin());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:    Gpu::copy(Gpu::deviceToHost, d_freestream_phi.begin(), d_freestream_phi.end(), freestream_phi.begin());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:    Gpu::copy(Gpu::deviceToHost, d_disk_cell_count.begin(), d_disk_cell_count.end(), disk_cell_count.begin());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:    Gpu::DeviceVector<Real> d_xloc(xloc.size());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:    Gpu::DeviceVector<Real> d_yloc(yloc.size());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:    Gpu::copy(Gpu::hostToDevice, xloc.begin(), xloc.end(), d_xloc.begin());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:    Gpu::copy(Gpu::hostToDevice, yloc.begin(), yloc.end(), d_yloc.begin());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:     Gpu::DeviceVector<Real> d_freestream_velocity(nturbs);
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:     Gpu::DeviceVector<Real> d_freestream_phi(nturbs);
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:     Gpu::DeviceVector<Real> d_disk_cell_count(nturbs);
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:     Gpu::copy(Gpu::hostToDevice, freestream_velocity.begin(), freestream_velocity.end(), d_freestream_velocity.begin());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:     Gpu::copy(Gpu::hostToDevice, freestream_phi.begin(), freestream_phi.end(), d_freestream_phi.begin());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:     Gpu::copy(Gpu::hostToDevice, disk_cell_count.begin(), disk_cell_count.end(), d_disk_cell_count.begin());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:    Gpu::DeviceVector<Real> d_wind_speed(wind_speed.size());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:    Gpu::DeviceVector<Real> d_thrust_coeff(thrust_coeff.size());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:    Gpu::copy(Gpu::hostToDevice, wind_speed.begin(), wind_speed.end(), d_wind_speed.begin());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:    Gpu::copy(Gpu::hostToDevice, thrust_coeff.begin(), thrust_coeff.end(), d_thrust_coeff.begin());
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:    for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/WindFarmParametrization/SimpleActuatorDisk/ERF_AdvanceSimpleAD.cpp:        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/BoundaryConditions/ERF_PhysBCFunct.H:                         const amrex::Gpu::DeviceVector<amrex::BCRec>& domain_bcs_type_d,
Source/BoundaryConditions/ERF_PhysBCFunct.H:                                   const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> dxInv,
Source/BoundaryConditions/ERF_PhysBCFunct.H:    amrex::Gpu::DeviceVector<amrex::BCRec> m_domain_bcs_type_d;
Source/BoundaryConditions/ERF_PhysBCFunct.H:                      const amrex::Gpu::DeviceVector<amrex::BCRec>& domain_bcs_type_d,
Source/BoundaryConditions/ERF_PhysBCFunct.H:                                   const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> dxInv,
Source/BoundaryConditions/ERF_PhysBCFunct.H:    amrex::Gpu::DeviceVector<amrex::BCRec> m_domain_bcs_type_d;
Source/BoundaryConditions/ERF_PhysBCFunct.H:                      const amrex::Gpu::DeviceVector<amrex::BCRec>& domain_bcs_type_d,
Source/BoundaryConditions/ERF_PhysBCFunct.H:                                   const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> dxInv,
Source/BoundaryConditions/ERF_PhysBCFunct.H:    amrex::Gpu::DeviceVector<amrex::BCRec> m_domain_bcs_type_d;
Source/BoundaryConditions/ERF_PhysBCFunct.H:                      const amrex::Gpu::DeviceVector<amrex::BCRec>& domain_bcs_type_d,
Source/BoundaryConditions/ERF_PhysBCFunct.H:                                  const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> dxInv,
Source/BoundaryConditions/ERF_PhysBCFunct.H:                                   const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> dxInv,
Source/BoundaryConditions/ERF_PhysBCFunct.H:    amrex::Gpu::DeviceVector<amrex::BCRec> m_domain_bcs_type_d;
Source/BoundaryConditions/ERF_PhysBCFunct.H:                         const amrex::Gpu::DeviceVector<amrex::BCRec>& domain_bcs_type_d,
Source/BoundaryConditions/ERF_PhysBCFunct.H:    amrex::Gpu::DeviceVector<amrex::BCRec> m_domain_bcs_type_d;
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:    Gpu::DeviceVector<BCRec> bcrs_d(1);
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:    Gpu::copyAsync(Gpu::hostToDevice, bcrs.begin(), bcrs.end(), bcrs_d.begin());
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:    GpuArray<GpuArray<Real, AMREX_SPACEDIM*2>, 1> l_bc_extdir_vals_d;
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:    Gpu::streamSynchronize();
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:                                                 const GpuArray<Real,AMREX_SPACEDIM> dxInv,
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:    Gpu::DeviceVector<BCRec> bcrs_d(1);
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:    Gpu::copyAsync(Gpu::hostToDevice, bcrs.begin(), bcrs.end(), bcrs_d.begin());
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:    GpuArray<GpuArray<Real, AMREX_SPACEDIM*2>, 1> l_bc_extdir_vals_d;
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:            ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_yvel.cpp:    Gpu::streamSynchronize();
Source/BoundaryConditions/ERF_BoundaryConditions_realbdy.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_BoundaryConditions_realbdy.cpp:                for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/BoundaryConditions/ERF_BoundaryConditions_realbdy.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_realbdy.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_realbdy.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_realbdy.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_realbdy.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_BoundaryConditions_realbdy.cpp:                for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/BoundaryConditions/ERF_BoundaryConditions_realbdy.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_realbdy.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_realbdy.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_realbdy.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:    Gpu::DeviceVector<BCRec> bcrs_d(ncomp);
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:    Gpu::copyAsync(Gpu::hostToDevice, bcrs.begin(), bcrs.end(), bcrs_d.begin());
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:            bx_xlo, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:            bx_xhi, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:            bx_ylo, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:            bx_yhi, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:            bx_xlo, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:            bx_xhi, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:            bx_ylo, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:            bx_yhi, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:    Gpu::streamSynchronize();
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:        bx_zlo1, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:        bx_zlo, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_basestate.cpp:        bx_zhi, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_FillBdyCCVels.cpp:        for (MFIter mfi(mf_cc_vel[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/BoundaryConditions/ERF_FillBdyCCVels.cpp:                    ParallelFor(makeSlab(bx,0,0), [=] AMREX_GPU_DEVICE(int , int j, int k) noexcept
Source/BoundaryConditions/ERF_FillBdyCCVels.cpp:                    ParallelFor(makeSlab(bx,0,0), [=] AMREX_GPU_DEVICE(int , int j, int k) noexcept
Source/BoundaryConditions/ERF_FillBdyCCVels.cpp:                    ParallelFor(makeSlab(bx,1,0), [=] AMREX_GPU_DEVICE(int i, int  , int k) noexcept
Source/BoundaryConditions/ERF_FillBdyCCVels.cpp:                    ParallelFor(makeSlab(bx,1,0), [=] AMREX_GPU_DEVICE(int i, int , int k) noexcept
Source/BoundaryConditions/ERF_FillBdyCCVels.cpp:                    ParallelFor(makeSlab(bx,2,0), [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
Source/BoundaryConditions/ERF_FillBdyCCVels.cpp:                    ParallelFor(makeSlab(bx,2,0), [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:                                                const GpuArray<Real,AMREX_SPACEDIM> dxInv,
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:    Gpu::DeviceVector<BCRec> bcrs_w_d(1);
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:    Gpu::copyAsync(Gpu::hostToDevice, bcrs_w.begin(), bcrs_w.end(), bcrs_w_d.begin());
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:    GpuArray<GpuArray<Real, AMREX_SPACEDIM*2>,1> l_bc_extdir_vals_d;
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:    Gpu::streamSynchronize();
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:                                                 const GpuArray<Real,AMREX_SPACEDIM> dxInv,
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:    GpuArray<GpuArray<Real, AMREX_SPACEDIM*2>,1> l_bc_extdir_vals_d;
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:            ParallelFor(makeSlab(bx,2,dom_lo.z), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:            ParallelFor(makeSlab(bx,2,dom_lo.z), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:            ParallelFor(makeSlab(bx,2,dom_hi.z+1), [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:            ParallelFor(makeSlab(bx,2,dom_hi.z+1), [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_zvel.cpp:    Gpu::streamSynchronize();
Source/BoundaryConditions/ERF_ABLMost.cpp:        ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_ABLMost.cpp:        // Expose for GPU
Source/BoundaryConditions/ERF_ABLMost.cpp:                ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_ABLMost.cpp:                    ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_ABLMost.cpp:                ParallelFor(xb2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_ABLMost.cpp:                ParallelFor(yb2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_ABLMost.cpp:        ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_ABLMost.cpp:        ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_ABLMost.cpp:            Gpu::HostVector<Real> m_x,m_y,m_z0;
Source/BoundaryConditions/ERF_ABLMost.cpp:            // Copy data to the GPU
Source/BoundaryConditions/ERF_ABLMost.cpp:            Gpu::DeviceVector<Real> d_x(nnode),d_y(nnode),d_z0(nnode);
Source/BoundaryConditions/ERF_ABLMost.cpp:            Gpu::copy(Gpu::hostToDevice, m_x.begin(), m_x.end(), d_x.begin());
Source/BoundaryConditions/ERF_ABLMost.cpp:            Gpu::copy(Gpu::hostToDevice, m_y.begin(), m_y.end(), d_y.begin());
Source/BoundaryConditions/ERF_ABLMost.cpp:            Gpu::copy(Gpu::hostToDevice, m_z0.begin(), m_z0.end(), d_z0.begin());
Source/BoundaryConditions/ERF_ABLMost.cpp:            ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
Source/BoundaryConditions/ERF_ABLMost.cpp:                       bcr, 0, 0, RunOn::Gpu);
Source/BoundaryConditions/ERF_FillPatcher.cpp:            ParallelFor(com_bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_FillPatcher.cpp:            AMREX_HOST_DEVICE_PARALLEL_FOR_3D_FLAG(RunOn::Gpu,fbx,i,j,k,
Source/BoundaryConditions/ERF_FillPatcher.cpp:            AMREX_HOST_DEVICE_PARALLEL_FOR_3D_FLAG(RunOn::Gpu,fbx,i,j,k,
Source/BoundaryConditions/ERF_FillPatcher.cpp:            AMREX_HOST_DEVICE_PARALLEL_FOR_3D_FLAG(RunOn::Gpu,fbx,i,j,k,
Source/BoundaryConditions/ERF_FillPatcher.cpp:            AMREX_HOST_DEVICE_PARALLEL_FOR_3D_FLAG(RunOn::Gpu,fbx,i,j,k,
Source/BoundaryConditions/ERF_FillPatcher.cpp:            AMREX_HOST_DEVICE_PARALLEL_FOR_3D_FLAG(RunOn::Gpu,fbx,i,j,k,
Source/BoundaryConditions/ERF_FillPatcher.cpp:            AMREX_HOST_DEVICE_PARALLEL_FOR_3D_FLAG(RunOn::Gpu,fbx,i,j,k,
Source/BoundaryConditions/ERF_FillPatcher.cpp:        bool run_on_gpu = Gpu::inLaunchRegion();
Source/BoundaryConditions/ERF_FillPatcher.cpp:        amrex::ignore_unused(run_on_gpu);
Source/BoundaryConditions/ERF_FillPatcher.cpp:#ifdef AMREX_USE_GPU
Source/BoundaryConditions/ERF_FillPatcher.cpp:        AsyncArray<BCRec> async_bcr(bcr.data(), (run_on_gpu) ? ncomp : 0);
Source/BoundaryConditions/ERF_FillPatcher.cpp:        BCRec const* bcrp = (run_on_gpu) ? async_bcr.data() : bcr.data();
Source/BoundaryConditions/ERF_FillPatcher.cpp:        AMREX_HOST_DEVICE_PARALLEL_FOR_4D_FLAG(RunOn::Gpu, cslope_bx, ncomp, i, j, k, n,
Source/BoundaryConditions/ERF_FillPatcher.cpp:        AMREX_HOST_DEVICE_PARALLEL_FOR_4D_FLAG(RunOn::Gpu, fbx, ncomp, i, j, k, n,
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_HOST_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_HOST_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTStress.H:    AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_BoundaryConditions_bndryreg.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_BoundaryConditions_bndryreg.cpp:                bx_xlo, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/ERF_BoundaryConditions_bndryreg.cpp:                bx_xhi, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/ERF_BoundaryConditions_bndryreg.cpp:               bx_ylo, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/ERF_BoundaryConditions_bndryreg.cpp:                bx_yhi, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/ERF_MOSTAverage.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_MOSTAverage.cpp:        ParallelFor(ubx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:        ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(npbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:            ParallelFor(npbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:            ParallelFor(npbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:            ParallelFor(npbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:    // GPU array to accumulate averages into
Source/BoundaryConditions/ERF_MOSTAverage.cpp:    Gpu::DeviceVector<Real> pavg(plane_average.size(), 0.0);
Source/BoundaryConditions/ERF_MOSTAverage.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(Gpu::KernelInfo().setReduction(true), pbx, [=]
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                AMREX_GPU_DEVICE(int i, int j, int k, Gpu::Handler const& handler) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                    Gpu::deviceReduceSum(&plane_avg[imf], val, handler);
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(Gpu::KernelInfo().setReduction(true), pbx, [=]
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                AMREX_GPU_DEVICE(int i, int j, int k, Gpu::Handler const& handler) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                    Gpu::deviceReduceSum(&plane_avg[imf], val, handler);
Source/BoundaryConditions/ERF_MOSTAverage.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(Gpu::KernelInfo().setReduction(true), pbx, [=]
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                AMREX_GPU_DEVICE(int i, int j, int k, Gpu::Handler const& handler) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                    Gpu::deviceReduceSum(&plane_avg[iavg], val, handler);
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(Gpu::KernelInfo().setReduction(true), pbx, [=]
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                AMREX_GPU_DEVICE(int i, int j, int k, Gpu::Handler const& handler) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                    Gpu::deviceReduceSum(&plane_avg[iavg], val, handler);
Source/BoundaryConditions/ERF_MOSTAverage.cpp:        Gpu::copy(Gpu::deviceToDevice, pavg.begin() + 2, pavg.begin() + 3,
Source/BoundaryConditions/ERF_MOSTAverage.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(Gpu::KernelInfo().setReduction(true), pbx, [=]
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                AMREX_GPU_DEVICE(int i, int j, int k, Gpu::Handler const& handler) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                    Gpu::deviceReduceSum(&plane_avg[iavg], val, handler);
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(Gpu::KernelInfo().setReduction(true), pbx, [=]
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                AMREX_GPU_DEVICE(int i, int j, int k, Gpu::Handler const& handler) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                    Gpu::deviceReduceSum(&plane_avg[iavg], val, handler);
Source/BoundaryConditions/ERF_MOSTAverage.cpp:    Gpu::copy(Gpu::deviceToHost, pavg.begin(), pavg.end(), plane_average.begin());
Source/BoundaryConditions/ERF_MOSTAverage.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(pbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(pbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(pbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(pbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(pbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(pbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_MOSTAverage.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_MOSTAverage.cpp:                ParallelFor(gpbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:    GpuArray<GpuArray<Real, AMREX_SPACEDIM*2>,NBCVAR_max> l_bc_extdir_vals_d;
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:    Gpu::DeviceVector<BCRec> bcrs_d(ncomp);
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:    Gpu::copyAsync(Gpu::hostToDevice, bcrs.begin(), bcrs.end(), bcrs_d.begin());
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:            bx_xlo, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:            bx_xhi, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:            bx_ylo, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:            bx_yhi, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:            bx_xlo, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:            bx_xhi, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:            bx_ylo, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:            bx_yhi, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:    Gpu::streamSynchronize();
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:                                                    const GpuArray<Real,AMREX_SPACEDIM> dxInv,
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:    GpuArray<GpuArray<Real, AMREX_SPACEDIM*2>,NBCVAR_max> l_bc_extdir_vals_d;
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:    GpuArray<GpuArray<Real, AMREX_SPACEDIM*2>,NBCVAR_max> l_bc_neumann_vals_d;
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:    Gpu::DeviceVector<BCRec> bcrs_d(icomp+ncomp);
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:    Gpu::copyAsync(Gpu::hostToDevice, bcrs.begin(), bcrs.end(), bcrs_d.begin());
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:            bx_zlo, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:            bx_zhi, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:            bx_zlo, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:            bx_zhi, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:                    ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_cons.cpp:    Gpu::streamSynchronize();
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:    Gpu::DeviceVector<BCRec> bcrs_d(1);
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:    Gpu::copyAsync(Gpu::hostToDevice, bcrs.begin(), bcrs.end(), bcrs_d.begin());
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:    GpuArray<GpuArray<Real, AMREX_SPACEDIM*2>,1> l_bc_extdir_vals_d;
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:    Gpu::streamSynchronize();
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:                                                 const GpuArray<Real,AMREX_SPACEDIM> dxInv,
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:    Gpu::DeviceVector<BCRec> bcrs_d(1);
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:    Gpu::copyAsync(Gpu::hostToDevice, bcrs.begin(), bcrs.end(), bcrs_d.begin());
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:    GpuArray<GpuArray<Real, AMREX_SPACEDIM*2>,1> l_bc_extdir_vals_d;
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:            ParallelFor(xybx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/ERF_BoundaryConditions_xvel.cpp:    Gpu::streamSynchronize();
Source/BoundaryConditions/ERF_MOSTAverage.H:#include <AMReX_Gpu.H>
Source/BoundaryConditions/ERF_MOSTAverage.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/BoundaryConditions/ERF_MOSTAverage.H:                                    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& plo,
Source/BoundaryConditions/ERF_MOSTAverage.H:                                    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxi,
Source/BoundaryConditions/ERF_ABLMost.H:#ifdef AMREX_USE_GPU
Source/BoundaryConditions/ERF_ABLMost.H:                                         [=] AMREX_GPU_HOST_DEVICE (amrex::Box const& bx, amrex::Array4<int const> const& lm_arr) -> int
Source/BoundaryConditions/ERF_MOSTRoughness.H:AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_MOSTRoughness.H:AMREX_GPU_DEVICE
Source/BoundaryConditions/ERF_PhysBCFunct.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_PhysBCFunct.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_PhysBCFunct.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_PhysBCFunct.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/ERF_PhysBCFunct.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
CHANGES:    -- Fix debug GPU with particles (#1422)

```
