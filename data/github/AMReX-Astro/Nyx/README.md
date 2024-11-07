# https://github.com/AMReX-Astro/Nyx

```console
Docs/sphinx_documentation/source/getting_started/BuildingGMake.rst:   All executables should work with MPI+CUDA by setting ``USE_MPI=TRUE USE_OMP=FALSE USE_CUDA=TRUE``.
Docs/sphinx_documentation/source/getting_started/BuildingGMake.rst:      The flag ``USE_FUSED`` tells the Nyx compile whether you compiled Sundials with fused cuda kernels. The default assumption is that non-cuda Nyx compiles set ``USE_FUSED=FALSE`` to match Sundials being built without fused cuda kernels.
Docs/sphinx_documentation/source/getting_started/BuildingGMake.rst:      Starting with Sundials version 5.7.0, set ``USE_SUNDIALS_SUNMEMORY=TRUE`` to compile the optional Sundials SunMemory to AMReX Arena interface for GPU memory reuse.
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:   | CMAKE\_CUDA\    | User-defined CUDA flags      | valid CUDA       | None        |
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:   | Nyx\_GPU\_      | On-node, accelerated GPU \   | NONE             | NONE,SYCL,\ |
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:   | BACKEND         | backend                      |                  | CUDA,HIP    |
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:For GPU builds, Nyx relies on the `AMReX GPU build infrastructure <https://amrex-codes.github.io/amrex/docs_html/GPU.html#building-with-cmake>`_
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:. The target architecture to build for can be specified via the AMReX configuration option ``-DAMReX_CUDA_ARCH=<target-architecture>``,
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:or by defining the *environmental variable* ``AMREX_CUDA_ARCH`` (all caps). If no GPU architecture is specified,
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:CMake will try to determine which GPU is supported by the system.
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:GPU build
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:To compile on the GPU nodes in Cori, you first need to purge your modules, most of which won't work on the GPU nodes
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:    > module load cgpu gcc/7.3.0 cuda/11.1.1 openmpi/4.0.3 cmake/3.14.4
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:Then, you need to use slurm to request access to a GPU node:
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:    > salloc -N 1 -t 02:00:00 -c 10 -C gpu -A m1759 --gres=gpu:8 --exclusive
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:This reservers an entire GPU node for your job. Note that you can’t cross-compile for the GPU nodes - you have to log on to one and then build your software.
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:Finally, navigate to the base of the Nyx repository and compile in GPU mode:
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:    > cmake -DNyx_GPU_BACKEND=CUDA -DAMReX_CUDA_ARCH=Volta -DCMAKE_CXX_COMPILER=g++ ..
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:For more information about GPU nodes in Cori -- `<https://docs-dev.nersc.gov/cgpu/>`_
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:To build Nyx for GPUs, you need to load cuda module:
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:    > module load cuda/11.0.3
Docs/sphinx_documentation/source/getting_started/BuildingCMake.rst:    > cmake -DNyx_GPU_BACKEND=CUDA -DAMReX_CUDA_ARCH=Volta -DCMAKE_C_COMPILER=$(which gcc)  -DCMAKE_CXX_COMPILER=$(which g++)   -DCMAKE_CUDA_HOST_COMPILER=$(which g++)  -DCMAKE_CUDA_ARCHITECTURES=70 ..
Docs/sphinx_documentation/source/getting_started/NyxSundials.rst:   To install with cuda and openmp support:
Docs/sphinx_documentation/source/getting_started/NyxSundials.rst:      -DCMAKE_CUDA_HOST_COMPILER=$(which g++)    \
Docs/sphinx_documentation/source/getting_started/NyxSundials.rst:      -DCMAKE_CUDA_FLAGS="-DSUNDIALS_DEBUG_CUDA_LASTERROR" \
Docs/sphinx_documentation/source/getting_started/NyxSundials.rst:      -DCUDA_ENABLE=ON  \
Docs/sphinx_documentation/source/getting_started/NyxSundials.rst:      -DCUDA_ARCH=sm_70 ../
Docs/sphinx_documentation/source/getting_started/NyxSundials.rst:   To install with openmp and no cuda support:
Docs/sphinx_documentation/source/getting_started/NyxSundials.rst:      -DCMAKE_CUDA_HOST_COMPILER=$(which g++)    \
Docs/sphinx_documentation/source/getting_started/NyxSundials.rst:      -DCUDA_ENABLE=OFF  \
Docs/sphinx_documentation/source/getting_started/NyxSundials.rst:   To install with HIP support (with ROCm 4.5):
Docs/sphinx_documentation/source/getting_started/NyxSundials.rst:      -DCUDA_ENABLE=OFF  \
Docs/sphinx_documentation/source/getting_started/NyxSundials.rst:      -DCUDA_ENABLE=OFF  \
Docs/sphinx_documentation/source/getting_started/NyxSundials.rst:#. ``CUDA_ARCH`` must be set to the appropriate value for the GPU being targeted
Docs/sphinx_documentation/source/getting_started/NyxSundials.rst:      spack install sundials+cuda+openmp
Docs/sphinx_documentation/source/getting_started/amrex_basics.rst:(this feature is off when running on GPU). 
Docs/sphinx_documentation/source/getting_started/amrex_basics.rst:   for ( amrex::MFIter mfi(mf, TilingIfNotGpu()); mfi.isValid(); ++mfi ) { ... }
Docs/sphinx_documentation/source/NyxPreface.rst:and CUDA/HIP/DPC++ on hybrid CPU/GPU architectures.
Docs/sphinx_documentation/source/NightlyTests.rst:These tests are also run on an NVIDIA GPU and those results can be found at https://ccse.lbl.gov/pub/GpuRegressionTesting/Nyx.
Docs/sphinx_documentation/source/NyxHeatCool.rst:- ``nyx.use_sundials_fused`` which when non-zero uses Sundials's GPU fused operations (which are mathematically equivalent, but reduces GPU kernel launch time overhead)
Docs/sphinx_documentation/source/NyxHeatCool.rst:- ``nyx.sundials_alloc_type`` which has up to 5 different vector memory allocation strategies and only affects executables built for GPUs
Docs/paper/paper.md:(exposing coarse-grained parallelism) or CUDA/HIP/DPC++ to spread the work across
Docs/paper/paper.md:GPU threads on GPU-based machines (fine-grained parallelism).  All of
Docs/paper/paper.md:the core physics can run on GPUs and have been shown to scale well.
Docs/paper/paper.md:for both CPUs and GPUs; additionally, we implement our parallel loops
Docs/paper/paper.md:GPU thread). This strategy is similar to the way the Kokkos [@Kokkos] and
Util/Converters/assign_particle_vels/NyxParticles.cpp:            amrex::Gpu::LaunchSafeGuard lsg(particle_launch_ics);
Util/Converters/assign_particle_vels/NyxParticles.cpp:            amrex::Gpu::LaunchSafeGuard lsg(particle_launch_ics);
Util/Converters/assign_particle_vels/NyxParticles.cpp:            amrex::Gpu::Device::synchronize();
Util/Converters/assign_particle_vels/NyxParticles.cpp:            amrex::Gpu::LaunchSafeGuard lsg(particle_launch_ics);
Util/Converters/assign_particle_vels/NyxParticles.cpp:            amrex::Gpu::LaunchSafeGuard lsg(particle_launch_ics);
Util/Converters/assign_particle_vels/NyxParticles.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Util/Converters/assign_particle_vels/NyxParticles.cpp:        for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Util/Converters/assign_particle_vels/NyxParticles.cpp:            GpuArray<amrex::Real,max_prob_param> prob_param;
Util/Converters/assign_particle_vels/NyxParticles.cpp://                             bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Util/Converters/assign_particle_vels/NyxParticles.cpp:        amrex::Gpu::Device::streamSynchronize();
Util/Converters/assign_particle_vels/NyxParticles.cpp:          amrex::Gpu::Device::streamSynchronize();
Util/Converters/assign_particle_vels/NyxParticles.cpp:    amrex::Gpu::Device::streamSynchronize();
Util/Converters/assign_particle_vels/test_script_summit_jobindex.sh:module load gcc/10.2.0 cuda/11.2.0 hdf5 python
Util/Converters/assign_particle_vels/test_script_summit_jobindex.sh:make -j USE_CUDA=FALSE USE_OMP=FALSE SUNDIALS_ROOT=../../subprojects/sundials/instdir/ AMREX_HOME=../../subprojects/amrex/ USE_HEATCOOL=FALSE
Util/Converters/assign_particle_vels/test_script_summit_jobindex.sh:make -j AMREX_HOME=../../../subprojects/amrex/ USE_CUDA=FALSE USE_MPI=TRUE
Util/Converters/assign_particle_vels/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Util/Converters/assign_particle_vels/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Util/Converters/assign_particle_vels/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Util/Converters/assign_particle_vels/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Util/Converters/assign_particle_vels/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Util/Converters/assign_particle_vels/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Util/Converters/assign_particle_vels/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Util/Converters/assign_particle_vels/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Util/Converters/assign_particle_vels/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/ParticleMeshTest/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/ParticleMeshTest/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/ParticleMeshTest/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/ParticleMeshTest/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/ParticleMeshTest/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/ParticleMeshTest/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/ParticleMeshTest/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/ParticleMeshTest/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/ParticleMeshTest/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DrivenTurbulence/Nyx_sources.cpp:    amrex::ParallelFor(bx, QVAR, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Exec/DrivenTurbulence/Nyx_sources.cpp:	amrex::Gpu::streamSynchronize();
Exec/DrivenTurbulence/Nyx_sources.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/DrivenTurbulence/Nyx_sources.cpp:        amrex::Gpu::streamSynchronize();
Exec/DrivenTurbulence/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& /*prob_param*/)
Exec/DrivenTurbulence/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/DrivenTurbulence/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/DrivenTurbulence/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/DrivenTurbulence/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/DrivenTurbulence/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/DrivenTurbulence/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/DrivenTurbulence/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/DrivenTurbulence/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/HeatCoolTests/example_setup.sh:	-DCMAKE_CUDA_HOST_COMPILER="$(which g++)"    \
Exec/HeatCoolTests/example_setup.sh:	-DCUDA_ENABLE=OFF  \
Exec/HeatCoolTests/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/HeatCoolTests/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/HeatCoolTests/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/HeatCoolTests/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/HeatCoolTests/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/HeatCoolTests/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/HeatCoolTests/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/HeatCoolTests/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/HeatCoolTests/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/HeatCoolTests/GNUmakefile:USE_CUDA = FALSE
Exec/HeatCoolTests/GNUmakefile:USE_FUSED ?= $(USE_CUDA)
Exec/LyA/GNUmakefile.summit:ASCENT_HOME ?=/gpfs/alpine/world-shared/csc340/software/ascent/0.5.2-pre/summit/cuda/gnu/ascent-install
Exec/LyA/GNUmakefile.summit:SUNDIALS_ROOT=${WORLDWORK}/ast160/software/sundials-cuda11.0.3/instdir
Exec/LyA/GNUmakefile.summit:SUNDIALS_ROOT = /gpfs/alpine/world-shared/ast160/software/sundials-5.6.0-cuda10.1.168/instdir/
Exec/LyA/GNUmakefile.summit:SUNDIALS_ROOT=${WORLDWORK}/csc308/software/sundials-cuda11.0.3-gcc6.4.0/instdir
Exec/LyA/GNUmakefile.summit:SUNDIALS_ROOT = /gpfs/alpine/world-shared/csc308/software/sundials-cuda10.1.168-gcc6.4.0/instdir/
Exec/LyA/GNUmakefile.summit:USE_CUDA = TRUE
Exec/LyA/GNUmakefile.summit:USE_FUSED ?= $(USE_CUDA)
Exec/LyA/GNUmakefile.summit:#ifneq ($(USE_CUDA),TRUE)
Exec/LyA/GNUmakefile.summit:                       -lrover_mpi $(DRAY_LINK_RPATH) $(DRAY_LIB_FLAGS) $(ASCENT_VTKH_MPI_LIB_FLAGS) $(ASCENT_VTKM_LIB_FLAGS) $(ASCENT_CONDUIT_MPI_LIB_FLAGS) $(ASCENT_MFEM_LIB_FLAGS) $(ASCENT_PYTHON_LIBS) $(ASCENT_OPENMP_LINK_FLAGS) -L $(ASCENT_CUDA_LIB_FLAGS)
Exec/LyA/inputs.summit:#Didn't exhaustively test tiling on with cuda
Exec/LyA/inputs.summit:#Only plm is implemented in cuda
Exec/LyA/inputs.summit:#cuda
Exec/LyA/inputs.summit:#cuda
Exec/LyA/inputs.summit:#Gpu runs should use 128 or 256 here
Exec/LyA/GNUmakefile.ascent:ASCENT_HOME ?=/global/cfs/cdirs/alpine/software/ascent/current/perlmutter/cuda/gnu/ascent-install/
Exec/LyA/GNUmakefile.ascent:USE_CUDA = FALSE
Exec/LyA/GNUmakefile.ascent:USE_FUSED ?= $(USE_CUDA)
Exec/LyA/GNUmakefile.ascent:##ifneq ($(USE_CUDA),TRUE)
Exec/LyA/GNUmakefile.ascent:#                       -lrover_mpi $(DRAY_LINK_RPATH) $(DRAY_LIB_FLAGS) $(ASCENT_VTKH_MPI_LIB_FLAGS) $(ASCENT_VTKM_LIB_FLAGS) $(ASCENT_CONDUIT_MPI_LIB_FLAGS) $(ASCENT_MFEM_LIB_FLAGS) $(ASCENT_PYTHON_LIBS) $(ASCENT_OPENMP_LINK_FLAGS) -L $(ASCENT_CUDA_LIB_FLAGS)
Exec/LyA/GNUmakefile.ascent:##                       -lrover_mpi $(DRAY_LINK_RPATH) $(DRAY_MPI_LIB_FLAGS) $(ASCENT_VTKH_MPI_LIB_FLAGS) $(ASCENT_VTKM_LIB_FLAGS) $(ASCENT_CONDUIT_MPI_LIB_FLAGS) $(ASCENT_MFEM_LIB_FLAGS) $(ASCENT_FIDES_LIB_FLAGS) $(ASCENT_ADIOS2_LIB_FLAGS) $(ASCENT_PMT_LIB_FLAGS) $(ASCENT_BABELFLOW_LIB_FLAGS) $(ASCENT_OCCA_LIB_FLAGS) $(ASCENT_GENTEN_LIB_FLAGS) $(ASCENT_UMPIRE_LIB_FLAGS) $(ASCENT_PYTHON_LIBS) $(ASCENT_OPENMP_LINK_FLAGS) -L $(ASCENT_CUDA_LIB_FLAGS)
Exec/LyA/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/LyA/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/LyA/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/LyA/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/LyA/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/LyA/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/LyA/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/LyA/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/LyA/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/LyA/test_spock_hip_omp.sh:module load PrgEnv-gnu craype-accel-amd-gfx908 rocm
Exec/LyA/test_spock_hip_omp.sh:#make -j USE_HIP=TRUE USE_OMP=TRUE SUNDIALS_ROOT=../../subprojects/sundials/instdir-rocm/ AMREX_HOME=../../subprojects/amrex/ 
Exec/LyA/test_spock_hip_omp.sh:export omp=4; export OMP_NUM_THREADS=${omp}; srun -n 4 -c${omp} --gpus-per-task=1 --gpu-bind=closest ./Nyx3d.hip.x86-rome.TPROF.MPI.OMP.HIP.ex inputs nyx.binary_particle_file= ${WORLDWORK}/ast160/ICs/256sss_20mpc.nyx amr.n_cell=256 256 256 amr.max_grid_size=64 particles.n_readers=1 particles.nreaders=1 max_step=5 amrex.max_gpu_streams=${omp}
Exec/LyA/test_spock_hip_omp.sh:#srun -n 16 --ntasks-per-node=4 ./Nyx3d.hip.x86-rome.TPROF.MPI.HIP.ex inputs.scaling.768 max_step=5 max_step=50 nyx.minimize_memory=1 nyx.shrink_to_fit=1 amrex.max_gpu_streams=8  amrex.the_arena_init_size=1000 | tee out_768_128_16_scal8.txt
Exec/LyA/test_spock_hip_omp.sh:#srun -n 16 --ntasks-per-node=4 ./Nyx3d.hip.x86-rome.TPROF.MPI.HIP.ex inputs.scaling.1024 max_step=5 max_step=50 nyx.minimize_memory=1 nyx.shrink_to_fit=1 amrex.max_gpu_streams=1 amr.regrid_on_restart=0 nyx.v=2 particles.v=2 amr.v=3 gravity.v=3 | tee out_1024_128_16_scaling.txt
Exec/LyA/test_spock_hip_omp.sh:#srun -n 16 --ntasks-per-node=4 ./Nyx3d.hip.x86-rome.TPROF.MPI.HIP.ex inputs.scaling.2048 max_step=5 max_step=50 nyx.minimize_memory=1 nyx.shrink_to_fit=1 amrex.max_gpu_streams=1 | tee out_2048_128_16.txt
Exec/LyA/test_spock_hip_omp.sh:#srun -n 16 --ntasks-per-node=4 ./Nyx3d.hip.x86-rome.TPROF.MPI.HIP.ex inputs max_step=5 nyx.binary_particle_file= 1024s_20mpc.nyx amr.n_cell=1024 1024 1024 amr.max_grid_size=128 nyx.minimize_memory=1 nyx.shrink_to_fit=1 amrex.max_gpu_streams=1 | tee out_1024_128_16.txt
Exec/LyA/GNUmakefile:USE_CUDA = FALSE
Exec/LyA/GNUmakefile:USE_FUSED ?= $(USE_CUDA)
Exec/AMR-density/CMakeLists.txt:set(_input_files inputs inputs.cuda inputs_nohydro.rt inputs.rt)
Exec/AMR-density/GNUmakefile.cuda:#This install location assumes modules: pgi/19.10 cuda/10.1.243
Exec/AMR-density/GNUmakefile.cuda:USE_CUDA = TRUE
Exec/AMR-density/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& /*prob_param*/)
Exec/AMR-density/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/AMR-density/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>&   prob_param  )
Exec/AMR-density/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& /*prob_param*/)
Exec/AMR-density/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/AMR-density/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/AMR-density/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/AMR-density/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/AMR-density/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/AMR-density/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/AMR-density/inputs.cuda:#Didn't exhaustively test tiling on with cuda
Exec/AMR-density/inputs.cuda:#Only plm is implemented in cuda
Exec/AMR-density/inputs.cuda:#cuda
Exec/AMR-density/inputs.cuda:#cuda
Exec/AMR-density/inputs.cuda:#Gpu runs should use 128 or 256 here
Exec/AMR-density/GNUmakefile:USE_CUDA = FALSE
Exec/AMR-density/GNUmakefile:USE_FUSED ?= $(USE_CUDA)
Exec/SantaBarbara/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/SantaBarbara/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/SantaBarbara/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/SantaBarbara/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/SantaBarbara/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/SantaBarbara/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/SantaBarbara/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/SantaBarbara/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/SantaBarbara/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/ParticleFilterTest/Nyx.cpp:#ifndef AMREX_USE_GPU
Exec/ParticleFilterTest/Nyx.cpp:int Nyx::sundials_atomic_reductions = -1; // CUDA and HIP only
Exec/ParticleFilterTest/Nyx.cpp:#ifdef AMREX_USE_CUDA
Exec/ParticleFilterTest/Nyx.cpp:int Nyx::sundials_atomic_reductions = -1; // CUDA and HIP only
Exec/ParticleFilterTest/Nyx.cpp:#ifndef AMREX_USE_GPU
Exec/ParticleFilterTest/Nyx.cpp:#ifndef AMREX_USE_GPU
Exec/ParticleFilterTest/Nyx.cpp:            "\nSuggested default for currently compiled CPU / GPU: nyx.hydro_tile_size="<<
Exec/ParticleFilterTest/Nyx.cpp:            "\nSuggested default for currently compiled CPU / GPU: nyx.sundials_tile_size="<<
Exec/ParticleFilterTest/Nyx.cpp:    amrex::Gpu::Device::streamSynchronize();
Exec/ParticleFilterTest/Nyx.cpp:              [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& u) -> Real
Exec/ParticleFilterTest/Nyx.cpp:#if !defined(__CUDACC__) || (__CUDACC_VER_MAJOR__ != 9) || (__CUDACC_VER_MINOR__ != 2)
Exec/ParticleFilterTest/Nyx.cpp:                  amrex::Real dt_gpu = std::numeric_limits<amrex::Real>::max();
Exec/ParticleFilterTest/Nyx.cpp:                  amrex::Real dt_gpu = 1.e37;
Exec/ParticleFilterTest/Nyx.cpp:                            dt_gpu = amrex::min(dt_gpu,amrex::min(dt1,amrex::min(dt2,dt3)));
Exec/ParticleFilterTest/Nyx.cpp:                  return dt_gpu;
Exec/ParticleFilterTest/Nyx.cpp:    amrex::Gpu::streamSynchronize();
Exec/ParticleFilterTest/Nyx.cpp:    amrex::Gpu::streamSynchronize();
Exec/ParticleFilterTest/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/Nyx.cpp:                  for (MFIter mfi(S_new_lev,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/ParticleFilterTest/Nyx.cpp:                       AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Exec/ParticleFilterTest/Nyx.cpp:    amrex::Gpu::streamSynchronize();
Exec/ParticleFilterTest/Nyx.cpp:    amrex::Gpu::streamSynchronize();
Exec/ParticleFilterTest/Nyx.cpp:    amrex::Gpu::Device::streamSynchronize();
Exec/ParticleFilterTest/Nyx.cpp:        amrex::Gpu::Device::streamSynchronize();
Exec/ParticleFilterTest/Nyx.cpp:        amrex::Gpu::streamSynchronize();
Exec/ParticleFilterTest/Nyx.cpp:    amrex::Gpu::Device::streamSynchronize();
Exec/ParticleFilterTest/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/Nyx.cpp:    for (MFIter mfi(S_old,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/ParticleFilterTest/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/Nyx.cpp:        for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/ParticleFilterTest/Nyx.cpp:            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Exec/ParticleFilterTest/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/Nyx.cpp:  for (MFIter mfi(S,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/ParticleFilterTest/Nyx.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Exec/ParticleFilterTest/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/Nyx.cpp:        for (MFIter mfi(*derive_dat,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/ParticleFilterTest/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/Nyx.cpp:        for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/ParticleFilterTest/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/Nyx.cpp:    for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/ParticleFilterTest/Nyx.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/ParticleFilterTest/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/Nyx.cpp:    for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/ParticleFilterTest/Nyx.cpp:          amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/ParticleFilterTest/Nyx.cpp:    amrex::Gpu::synchronize();
Exec/ParticleFilterTest/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/Nyx.cpp:          for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/ParticleFilterTest/Nyx.cpp:            amrex::Gpu::synchronize();
Exec/ParticleFilterTest/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/Nyx.cpp:        for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/ParticleFilterTest/Nyx.cpp:    amrex::Gpu::synchronize();
Exec/ParticleFilterTest/Nyx.cpp:#ifdef AMREX_USE_GPU
Exec/ParticleFilterTest/Nyx.cpp:    if (Gpu::inLaunchRegion())
Exec/ParticleFilterTest/Nyx.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Exec/ParticleFilterTest/Nyx.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion() && !system::regtest_reduction)               \
Exec/ParticleFilterTest/Nyx.cpp:#ifdef AMREX_USE_GPU
Exec/ParticleFilterTest/Nyx.cpp:    if (Gpu::inLaunchRegion())
Exec/ParticleFilterTest/Nyx.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Exec/ParticleFilterTest/Nyx.cpp:#pragma omp parallel  if (amrex::Gpu::notInLaunchRegion())               \
Exec/ParticleFilterTest/DarkMatterParticleContainer.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE void update_dm_particle_single (amrex::ParticleContainer<1+AMREX_SPACEDIM, 0>::SuperParticleType&  p,
Exec/ParticleFilterTest/DarkMatterParticleContainer.H:                                                                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Exec/ParticleFilterTest/DarkMatterParticleContainer.H:                                                                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
Exec/ParticleFilterTest/DarkMatterParticleContainer.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE void update_dm_particle_move_single (amrex::ParticleContainer<1+AMREX_SPACEDIM, 0>::SuperParticleType&  p,
Exec/ParticleFilterTest/DarkMatterParticleContainer.H:                                                                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Exec/ParticleFilterTest/DarkMatterParticleContainer.H:                                                                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
Exec/ParticleFilterTest/DarkMatterParticleContainer.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE void store_dm_particle_single (amrex::ParticleContainer<1+AMREX_SPACEDIM, 0>::SuperParticleType&  p,
Exec/ParticleFilterTest/DarkMatterParticleContainer.H:                                                                  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Exec/ParticleFilterTest/DarkMatterParticleContainer.H:                                                                  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& phi,
Exec/ParticleFilterTest/DarkMatterParticleContainer.H:                                                                  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
Exec/ParticleFilterTest/NyxParticles.cpp:            amrex::Gpu::LaunchSafeGuard lsg(particle_launch_ics);
Exec/ParticleFilterTest/NyxParticles.cpp:            amrex::Gpu::LaunchSafeGuard lsg(particle_launch_ics);
Exec/ParticleFilterTest/NyxParticles.cpp:            amrex::Gpu::Device::synchronize();
Exec/ParticleFilterTest/NyxParticles.cpp:            amrex::Gpu::LaunchSafeGuard lsg(particle_launch_ics);
Exec/ParticleFilterTest/NyxParticles.cpp:            amrex::Gpu::LaunchSafeGuard lsg(particle_launch_ics);
Exec/ParticleFilterTest/NyxParticles.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/NyxParticles.cpp:        for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/ParticleFilterTest/NyxParticles.cpp:            GpuArray<amrex::Real,max_prob_param> prob_param;
Exec/ParticleFilterTest/NyxParticles.cpp://                             bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Exec/ParticleFilterTest/NyxParticles.cpp:        amrex::Gpu::Device::streamSynchronize();
Exec/ParticleFilterTest/NyxParticles.cpp:          amrex::Gpu::Device::streamSynchronize();
Exec/ParticleFilterTest/NyxParticles.cpp:    amrex::Gpu::Device::streamSynchronize();
Exec/ParticleFilterTest/NyxParticleContainer.H:    virtual void RedistributeGPU   (int lev_min              = 0,
Exec/ParticleFilterTest/NyxParticleContainer.H:                                     [=] AMREX_GPU_HOST_DEVICE (const PType& p) -> amrex::Real
Exec/ParticleFilterTest/NyxParticleContainer.H:      //        amrex::Gpu::synchronize();
Exec/ParticleFilterTest/NyxParticleContainer.H:        //        amrex::Gpu::synchronize();
Exec/ParticleFilterTest/NyxParticleContainer.H:    virtual void RedistributeGPU   (int /*lev_min              = 0*/,
Exec/ParticleFilterTest/NyxParticleContainer.H:        amrex::Gpu::synchronize();
Exec/ParticleFilterTest/NyxParticleContainer.H:            ::RedistributeGPU(lev_minal, lev_maxal, nGrowal, local);
Exec/ParticleFilterTest/NyxParticleContainer.H:    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> adxi{dxi[0]/a,dxi[1]/a,dxi[2]/a};
Exec/ParticleFilterTest/NyxParticleContainer.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(min:dt)
Exec/ParticleFilterTest/NyxParticleContainer.H:                [=] AMREX_GPU_DEVICE (const int i) -> ReduceTuple {
Exec/ParticleFilterTest/NyxParticleContainer.H:                           [=] AMREX_GPU_HOST_DEVICE ( long i)
Exec/ParticleFilterTest/NyxParticleContainer.H:   amrex::Gpu::streamSynchronize();
Exec/ParticleFilterTest/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/ParticleFilterTest/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/ParticleFilterTest/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/ParticleFilterTest/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/ParticleFilterTest/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/ParticleFilterTest/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/ParticleFilterTest/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/ParticleFilterTest/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/ParticleFilterTest/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    GpuArray<Real, AMREX_SPACEDIM> m_plo, m_phi, m_center;
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    ShellFilter (const GpuArray<Real, AMREX_SPACEDIM>& plo,
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                 const GpuArray<Real, AMREX_SPACEDIM>& phi,
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                 const GpuArray<Real, AMREX_SPACEDIM>& center,
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    AMREX_GPU_HOST_DEVICE
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    AMREX_GPU_HOST_DEVICE
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    GpuArray<Real, AMREX_SPACEDIM> m_plo, m_phi, m_center;
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    ShellStoreFilter (const GpuArray<Real, AMREX_SPACEDIM>& plo,
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                 const GpuArray<Real, AMREX_SPACEDIM>& phi,
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                 const GpuArray<Real, AMREX_SPACEDIM>& center,
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    AMREX_GPU_HOST_DEVICE
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    Gpu::DeviceVector<Index> mask_vec(np);
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    Gpu::DeviceVector<Index> offsets(np);
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    Gpu::exclusive_scan(mask, mask+np, offsets.begin());
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    Gpu::copyAsync(Gpu::deviceToHost, mask+np-1, mask + np, &last_mask);
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    Gpu::copyAsync(Gpu::deviceToHost, offsets.data()+np-1, offsets.data()+np, &last_offset);
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    const GpuArray<Real,AMREX_SPACEDIM> plo = Geom(lev).ProbLoArray();
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    const GpuArray<Real,AMREX_SPACEDIM> phi=geom_test.ProbHiArray();
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    const GpuArray<Real,AMREX_SPACEDIM> center({AMREX_D_DECL((phi[0]-plo[0])*0.5,(phi[1]-plo[1])*0.5,(phi[2]-plo[2])*0.5)});
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                           [=] AMREX_GPU_HOST_DEVICE ( long i)
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:        Gpu::Device::streamSynchronize();
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:#ifdef AMREX_USE_GPU
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                           [=] AMREX_GPU_HOST_DEVICE ( long i)
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                           [=] AMREX_GPU_HOST_DEVICE ( long i)
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:            Gpu::streamSynchronize();
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    const GpuArray<Real,AMREX_SPACEDIM> plo = Geom(lev).ProbLoArray();
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                           [=] AMREX_GPU_HOST_DEVICE ( long i)
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& phi,
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    const GpuArray<Real,AMREX_SPACEDIM> center({AMREX_D_DECL((phi[0]-plo[0])*0.5,(phi[1]-plo[1])*0.5,(phi[2]-plo[2])*0.5)});
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:                                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Exec/ParticleFilterTest/DarkMatterParticleContainer.cpp:    for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/ParticleFilterTest/GNUmakefile:USE_CUDA = FALSE
Exec/ParticleFilterTest/GNUmakefile:USE_FUSED ?= $(USE_CUDA)
Exec/MiniSB/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& /*prob_param*/)
Exec/MiniSB/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/MiniSB/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/MiniSB/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/MiniSB/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/MiniSB/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/MiniSB/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/MiniSB/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/MiniSB/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/ParticleOnlyTest/inputs.load:#Gpu runs should use 128 or 256 here
Exec/ParticleOnlyTest/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/HydroTests/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/HydroTests/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/HydroTests/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/HydroTests/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/HydroTests/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/HydroTests/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/HydroTests/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/HydroTests/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/HydroTests/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/HydroTests/GNUmakefile:USE_CUDA = FALSE
Exec/Make.Nyx:ifeq ($(USE_CUDA), TRUE)
Exec/Make.Nyx:ifeq ($(USE_CUDA),TRUE)
Exec/Make.Nyx:  LIBRARIES += -L$(SUNDIALS_LIB_DIR) -lsundials_nveccuda
Exec/Make.Nyx:ifneq ($(USE_GPU),TRUE)
Exec/Make.Nyx:ifeq ($(USE_CUDA),TRUE)
Exec/Make.Nyx:     LIBRARIES += -L$(SUNDIALS_LIB_DIR) -lsundials_cvode_fused_cuda
Exec/GravityTests/zeldovich_dm/build/Prob.H:static void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/GravityTests/zeldovich_dm/build/Prob.H:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/GravityTests/zeldovich_dm/build/Prob.H:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/GravityTests/zeldovich_dm/build/Prob.H:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/GravityTests/zeldovich_dm/build/Prob.H:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/GravityTests/zeldovich_dm/build/GNUmakefile:USE_CUDA = FALSE
Exec/Henson/sample-analysis/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/Henson/sample-analysis/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/Henson/sample-analysis/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/Henson/sample-analysis/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/Henson/sample-analysis/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/Henson/sample-analysis/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/Henson/sample-analysis/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
Exec/Henson/sample-analysis/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/Henson/sample-analysis/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
Exec/Henson/sample-analysis/GNUmakefile:USE_CUDA = FALSE
Exec/Henson/particles.chai:var procmap = ProcMap()
Exec/Henson/particles.chai:if(procmap.group() == "world")
Exec/Henson/particles.chai:    var nyx       = load("./Nyx3d.${compiler}.ex inputs", procmap)
Exec/Henson/particles.chai:    var particles = load("../Henson/extract-particles/extract-particles3d.${compiler}.ex", procmap)
Exec/Henson/extract-particles/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/Henson/extract-particles/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/Henson/extract-particles/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/Henson/extract-particles/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/Henson/extract-particles/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/Henson/extract-particles/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/Henson/extract-particles/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
Exec/Henson/extract-particles/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/Henson/extract-particles/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
Exec/Henson/extract-particles/GNUmakefile:USE_CUDA = FALSE
Exec/Henson/all.chai:var procmap = ProcMap()
Exec/Henson/all.chai:if(procmap.group() == "world")
Exec/Henson/all.chai:    var nyx       = load("./Nyx3d.${compiler}.ex inputs", procmap)
Exec/Henson/all.chai:    var particles = load("../Henson/extract-particles/extract-particles3d.${compiler}.ex", procmap)
Exec/Henson/all.chai:    var info      = load("../Henson/amr-info/amr-info3d.${compiler}.ex", procmap)
Exec/Henson/all.chai:    var analysis  = load("../Henson/sample-analysis/sample-analysis3d.${compiler}.ex", procmap)
Exec/Henson/amr-info.chai:var procmap = ProcMap()
Exec/Henson/amr-info.chai:if(procmap.group() == "world")
Exec/Henson/amr-info.chai:    var nyx   = load("./Nyx3d.${compiler}.ex inputs", procmap)
Exec/Henson/amr-info.chai:    var info  = load("../Henson/amr-info/amr-info3d.${compiler}.ex", procmap)
Exec/Henson/simple.chai:var procmap = ProcMap()
Exec/Henson/simple.chai:if(procmap.group() == "world")
Exec/Henson/simple.chai:    var nyx      = load("./Nyx3d.${compiler}.ex inputs", procmap)
Exec/Henson/simple.chai:    var analysis = load("../Henson/sample-analysis/sample-analysis3d.${compiler}.ex", procmap)
Exec/Henson/amr-info/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/Henson/amr-info/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/Henson/amr-info/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/Henson/amr-info/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/Henson/amr-info/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/Henson/amr-info/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/Henson/amr-info/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
Exec/Henson/amr-info/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/Henson/amr-info/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
Exec/Henson/amr-info/GNUmakefile:USE_CUDA = FALSE
Exec/Henson/nyx-only.chai:var procmap = ProcMap()
Exec/Henson/nyx-only.chai:if(procmap.group() == "world")
Exec/Henson/nyx-only.chai:    var nyx       = load("./Nyx3d.${compiler}.ex inputs", procmap)
Exec/Scaling/run_short.summit:module load cuda/9.1.85
Exec/Scaling/run_short.summit:#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/summit/cuda/9.1.85/lib64
Exec/Scaling/run_short.summit:#export MPICH_RDMA_ENABLED_CUDA=1
Exec/Scaling/run_short.summit:#EXE="./main3d.pgi.MPI.CUDA.ex"
Exec/Scaling/run_short.summit:# n = tasks (MPI), g = gpus/task, c = threads/task, a = task/resource
Exec/Scaling/run_short.summit:#${JSRUN} --smpiargs="-gpu" cuda-memcheck ${EXE} ${INPUTS} &> memcheck.${LSB_JOBID}.txt
Exec/Scaling/run_short.summit:#${JSRUN} --smpiargs="-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks" cuda-memcheck ${EXE} ${INPUTS} &> memcheck.${LSB_JOBID}.txt
Exec/Scaling/run_short.summit:${JSRUN} --smpiargs="-gpu" ${EXE} ${INPUTS} >& out.${LSB_JOBID}
Exec/Scaling/run_short.summit:#${JSRUN} --smpiargs="-gpu" nvprof ${EXE} ${INPUTS} &> nvprof.${LSB_JOBID}.txt
Exec/Scaling/run_cudashort.summit:module load cuda/9.1.85
Exec/Scaling/run_cudashort.summit:#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/summit/cuda/9.1.85/lib64
Exec/Scaling/run_cudashort.summit:#export MPICH_RDMA_ENABLED_CUDA=1
Exec/Scaling/run_cudashort.summit:EXE="./Nyx3d.pgi.MPI.CUDA.ex"
Exec/Scaling/run_cudashort.summit:#EXE="./main3d.pgi.MPI.CUDA.ex"
Exec/Scaling/run_cudashort.summit:# n = tasks (MPI), g = gpus/task, c = threads/task, a = task/resource
Exec/Scaling/run_cudashort.summit:#${JSRUN} --smpiargs="-gpu" cuda-memcheck ${EXE} ${INPUTS} &> memcheck.${LSB_JOBID}.txt
Exec/Scaling/run_cudashort.summit:#${JSRUN} --smpiargs="-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks" cuda-memcheck ${EXE} ${INPUTS} &> memcheck.${LSB_JOBID}.txt
Exec/Scaling/run_cudashort.summit:${JSRUN} --smpiargs="-gpu" ${EXE} ${INPUTS} >& out.${LSB_JOBID}
Exec/Scaling/run_cudashort.summit:#${JSRUN} --smpiargs="-gpu" nvprof ${EXE} ${INPUTS} &> nvprof.${LSB_JOBID}.txt
Exec/Scaling/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/Scaling/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/Scaling/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/Scaling/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/Scaling/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/Scaling/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/Scaling/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/Scaling/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/Scaling/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/Scaling/GNUmakefile:USE_CUDA = FALSE
Exec/Scaling/GNUmakefile:USE_FUSED ?= $(USE_CUDA)
Exec/LyA_Neutrinos/GNUmakefile.cuda:USE_CUDA = TRUE
Exec/LyA_Neutrinos/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/LyA_Neutrinos/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/LyA_Neutrinos/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/LyA_Neutrinos/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/LyA_Neutrinos/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/LyA_Neutrinos/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/LyA_Neutrinos/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/LyA_Neutrinos/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/LyA_Neutrinos/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/LyA_Neutrinos/GNUmakefile:USE_CUDA = FALSE
Exec/LyA_Neutrinos/GNUmakefile:USE_FUSED ?= $(USE_CUDA)
Exec/LyA_Neutrinos/GNUmakefile:ifeq ($(USE_CUDA),TRUE)
Exec/LyA_Neutrinos/GNUmakefile:     LIBRARIES += -L$(SUNDIALS_ROOT)/lib -lsundials_cvode_fused_cuda
Exec/AMR-zoom/Prob.cpp:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/AMR-zoom/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/AMR-zoom/Prob.cpp:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/AMR-zoom/Prob.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Exec/AMR-zoom/Prob.cpp:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Exec/AMR-zoom/Prob.cpp:                          const GpuArray<Real,max_prob_param>& prob_param)
Exec/AMR-zoom/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/AMR-zoom/Prob.cpp:                                const GpuArray<Real,max_prob_param>& prob_param)
Exec/AMR-zoom/Prob.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Exec/AMR-zoom/GNUmakefile:USE_CUDA = FALSE
Exec/AMR-zoom/GNUmakefile:USE_FUSED ?= $(USE_CUDA)
Exec/AMR-zoom/GNUmakefile:ifeq ($(USE_CUDA),TRUE)
Exec/AMR-zoom/GNUmakefile:     LIBRARIES += -L$(SUNDIALS_ROOT)/lib -lsundials_cvode_fused_cuda
README.md:CUDA/HIP/DPC++ on hybrid CPU/GPU architectures.
README.md:of up to 2,097,152 on NERSC's Cori-KNL. With Cuda implementation, it was run on up to
README.md:13,824 GPUs on OLCF's Summit.
README.md:OpenMP 4.5 or higher, Cuda 9 or higher, or HIP-Clang.
README.md:$ module load gcc/6.4.0 cuda/11.0.3
README.md:$ make -j 12 USE_CUDA=TRUE
CMakeLists.txt:# GPU backends    =============================================================
CMakeLists.txt:set(Nyx_GPU_BACKEND_VALUES NONE SYCL CUDA HIP)
CMakeLists.txt:set(Nyx_GPU_BACKEND NONE CACHE STRING "On-node, accelerated GPU backend: <NONE,SYCL,CUDA,HIP>")
CMakeLists.txt:set_property(CACHE Nyx_GPU_BACKEND PROPERTY STRINGS ${Nyx_GPU_BACKEND_VALUES})
CMakeLists.txt:if (NOT Nyx_GPU_BACKEND IN_LIST Nyx_GPU_BACKEND_VALUES)
CMakeLists.txt:   message(FATAL_ERROR "Nyx_GPU_BACKEND=${Nyx_GPU_BACKEND} is not allowed."
CMakeLists.txt:      " Must be one of ${Nyx_GPU_BACKEND_VALUES}")
CMakeLists.txt:if (NOT Nyx_GPU_BACKEND STREQUAL NONE)
CMakeLists.txt:   message( STATUS "   Nyx_GPU_BACKEND = ${Nyx_GPU_BACKEND}")
CMakeLists.txt:   "Nyx_GPU_BACKEND STREQUAL NONE" OFF)
CMakeLists.txt:   "NOT Nyx_GPU_BACKEND STREQUAL SYCL" OFF)
CMakeLists.txt:   if (Nyx_GPU_BACKEND STREQUAL "CUDA")
CMakeLists.txt:   if (Nyx_GPU_BACKEND STREQUAL "HIP")
CMakeLists.txt:if (Nyx_GPU_BACKEND STREQUAL CUDA)
CMakeLists.txt:    enable_language(CUDA)
Source/Particle/DarkMatterParticleContainer.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE void update_dm_particle_single (amrex::ParticleContainer<4, 0>::SuperParticleType&  p,
Source/Particle/DarkMatterParticleContainer.H:                                                                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Source/Particle/DarkMatterParticleContainer.H:                                                                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
Source/Particle/NyxParticles.cpp:            amrex::Gpu::LaunchSafeGuard lsg(particle_launch_ics);
Source/Particle/NyxParticles.cpp:            amrex::Gpu::LaunchSafeGuard lsg(particle_launch_ics);
Source/Particle/NyxParticles.cpp:            amrex::Gpu::Device::synchronize();
Source/Particle/NyxParticles.cpp:            amrex::Gpu::LaunchSafeGuard lsg(particle_launch_ics);
Source/Particle/NyxParticles.cpp:            amrex::Gpu::LaunchSafeGuard lsg(particle_launch_ics);
Source/Particle/NyxParticles.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Particle/NyxParticles.cpp:        for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Particle/NyxParticles.cpp:            GpuArray<amrex::Real,max_prob_param> prob_param;
Source/Particle/NyxParticles.cpp://                             bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Particle/NyxParticles.cpp:        amrex::Gpu::Device::streamSynchronize();
Source/Particle/NyxParticles.cpp:          amrex::Gpu::Device::streamSynchronize();
Source/Particle/NyxParticles.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Particle/AGNParticleContainer.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particle/AGNParticleContainer.H:                                     amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Source/Particle/AGNParticleContainer.H:                                     amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
Source/Particle/NyxParticleContainer.H:    virtual void RedistributeGPU   (int lev_min              = 0,
Source/Particle/NyxParticleContainer.H:                                     [=] AMREX_GPU_HOST_DEVICE (const PType& p) -> amrex::Real
Source/Particle/NyxParticleContainer.H:      //        amrex::Gpu::synchronize();
Source/Particle/NyxParticleContainer.H:        //        amrex::Gpu::synchronize();
Source/Particle/NyxParticleContainer.H:    virtual void RedistributeGPU   (int /*lev_min              = 0*/,
Source/Particle/NyxParticleContainer.H:        amrex::Gpu::synchronize();
Source/Particle/NyxParticleContainer.H:            ::RedistributeGPU(lev_minal, lev_maxal, nGrowal, local);
Source/Particle/NyxParticleContainer.H:    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> adxi{dxi[0]/a,dxi[1]/a,dxi[2]/a};
Source/Particle/NyxParticleContainer.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(min:dt)
Source/Particle/NyxParticleContainer.H:                [=] AMREX_GPU_DEVICE (const int i) -> ReduceTuple {
Source/Particle/NyxParticleContainer.H:                           [=] AMREX_GPU_HOST_DEVICE ( long i)
Source/Particle/NyxParticleContainer.H:   amrex::Gpu::streamSynchronize();
Source/Particle/NeutrinoParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Particle/NeutrinoParticleContainer.cpp:    for (MFIter mfi(*mf_pointer,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Particle/NeutrinoParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Particle/NeutrinoParticleContainer.cpp:        for (MFIter mfi(*mf_pointer,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Particle/AGNParticleContainer.cpp:    const GpuArray<Real,AMREX_SPACEDIM> plo = Geom(lev).ProbLoArray();
Source/Particle/AGNParticleContainer.cpp:                               [=] AMREX_GPU_HOST_DEVICE ( long i)
Source/Particle/AGNParticleContainer.cpp:    const GpuArray<Real,AMREX_SPACEDIM> plo = Geom(lev).ProbLoArray();
Source/Particle/AGNParticleContainer.cpp:                               [=] AMREX_GPU_HOST_DEVICE ( long i)
Source/Particle/AGNParticleContainer.cpp:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particle/AGNParticleContainer.cpp:                            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Source/Particle/AGNParticleContainer.cpp:                            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
Source/Particle/NeutrinoParticles_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particle/NeutrinoParticles_K.H:                                        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Source/Particle/NeutrinoParticles_K.H:                                        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi)
Source/Particle/NeutrinoParticles_K.H:                amrex::Gpu::Atomic::Add(&rho(i+ii-1, j+jj-1, k+kk-1, 0),
Source/Particle/NeutrinoParticles_K.H:                    amrex::Gpu::Atomic::Add(&rho(i+ii-1, j+jj-1, k+kk-1, comp),
Source/Particle/NeutrinoParticles_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particle/NeutrinoParticles_K.H:                                     amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Source/Particle/NeutrinoParticles_K.H:                                     amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
Source/Particle/NeutrinoParticles_K.H:                                     amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& pdxi)
Source/Particle/DarkMatterParticleContainer.cpp:    const GpuArray<Real,AMREX_SPACEDIM> plo = Geom(lev).ProbLoArray();
Source/Particle/DarkMatterParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Particle/DarkMatterParticleContainer.cpp:                           [=] AMREX_GPU_HOST_DEVICE ( long i)
Source/Particle/DarkMatterParticleContainer.cpp:                           [=] AMREX_GPU_HOST_DEVICE ( long i)
Source/Particle/DarkMatterParticleContainer.cpp:            Gpu::streamSynchronize();
Source/Particle/DarkMatterParticleContainer.cpp:    const GpuArray<Real,AMREX_SPACEDIM> plo = Geom(lev).ProbLoArray();
Source/Particle/DarkMatterParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Particle/DarkMatterParticleContainer.cpp:                           [=] AMREX_GPU_HOST_DEVICE ( long i)
Source/Particle/DarkMatterParticleContainer.cpp:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particle/DarkMatterParticleContainer.cpp:                                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Source/Particle/DarkMatterParticleContainer.cpp:                                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
Source/Particle/DarkMatterParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Particle/DarkMatterParticleContainer.cpp:    for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/Nyx.cpp:#ifndef AMREX_USE_GPU
Source/Driver/Nyx.cpp:int Nyx::sundials_atomic_reductions = -1; // CUDA and HIP only
Source/Driver/Nyx.cpp:#ifdef AMREX_USE_CUDA
Source/Driver/Nyx.cpp:int Nyx::sundials_atomic_reductions = -1; // CUDA and HIP only
Source/Driver/Nyx.cpp:#ifndef AMREX_USE_GPU
Source/Driver/Nyx.cpp:#ifndef AMREX_USE_GPU
Source/Driver/Nyx.cpp:            "\nSuggested default for currently compiled CPU / GPU: nyx.hydro_tile_size="<<
Source/Driver/Nyx.cpp:            "\nSuggested default for currently compiled CPU / GPU: nyx.sundials_tile_size="<<
Source/Driver/Nyx.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Driver/Nyx.cpp:              [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& u) -> Real
Source/Driver/Nyx.cpp:#if !defined(__CUDACC__) || (__CUDACC_VER_MAJOR__ != 9) || (__CUDACC_VER_MINOR__ != 2)
Source/Driver/Nyx.cpp:                  amrex::Real dt_gpu = std::numeric_limits<amrex::Real>::max();
Source/Driver/Nyx.cpp:                  amrex::Real dt_gpu = 1.e37;
Source/Driver/Nyx.cpp:                            dt_gpu = amrex::min(dt_gpu,amrex::min(dt1,amrex::min(dt2,dt3)));
Source/Driver/Nyx.cpp:                  return dt_gpu;
Source/Driver/Nyx.cpp:    amrex::Gpu::streamSynchronize();
Source/Driver/Nyx.cpp:    amrex::Gpu::streamSynchronize();
Source/Driver/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Driver/Nyx.cpp:                  for (MFIter mfi(S_new_lev,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/Nyx.cpp:                       AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Driver/Nyx.cpp:    amrex::Gpu::streamSynchronize();
Source/Driver/Nyx.cpp:    amrex::Gpu::streamSynchronize();
Source/Driver/Nyx.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Driver/Nyx.cpp:        amrex::Gpu::Device::streamSynchronize();
Source/Driver/Nyx.cpp:        amrex::Gpu::streamSynchronize();
Source/Driver/Nyx.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Driver/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Driver/Nyx.cpp:    for (MFIter mfi(S_old,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Driver/Nyx.cpp:        for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/Nyx.cpp:            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Driver/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Driver/Nyx.cpp:  for (MFIter mfi(S,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/Nyx.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Driver/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Driver/Nyx.cpp:        for (MFIter mfi(*derive_dat,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Driver/Nyx.cpp:        for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Driver/Nyx.cpp:    for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/Nyx.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Driver/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Driver/Nyx.cpp:    for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/Nyx.cpp:          amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Driver/Nyx.cpp:    amrex::Gpu::synchronize();
Source/Driver/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Driver/Nyx.cpp:          for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/Nyx.cpp:            amrex::Gpu::synchronize();
Source/Driver/Nyx.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Driver/Nyx.cpp:        for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/Nyx.cpp:    amrex::Gpu::synchronize();
Source/Driver/Nyx.cpp:#ifdef AMREX_USE_GPU
Source/Driver/Nyx.cpp:    if (Gpu::inLaunchRegion())
Source/Driver/Nyx.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/Driver/Nyx.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion() && !system::regtest_reduction)               \
Source/Driver/Nyx.cpp:#ifdef AMREX_USE_GPU
Source/Driver/Nyx.cpp:    if (Gpu::inLaunchRegion())
Source/Driver/Nyx.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/Driver/Nyx.cpp:#pragma omp parallel  if (amrex::Gpu::notInLaunchRegion())               \
Source/Driver/sum_utils.cpp:#ifndef AMREX_USE_GPU
Source/Driver/sum_utils.cpp:        for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/sum_utils.cpp:            AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/Driver/sum_utils.cpp:#ifndef AMREX_USE_GPU
Source/Driver/sum_utils.cpp:        for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/sum_utils.cpp:            AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/Driver/sum_utils.cpp:#ifndef AMREX_USE_GPU
Source/Driver/sum_utils.cpp:    for (MFIter mfi(*mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/sum_utils.cpp:        AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/Driver/sum_utils.cpp:#ifndef AMREX_USE_GPU
Source/Driver/sum_utils.cpp:        for (MFIter mfi(*mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/sum_utils.cpp:            AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/Driver/sum_utils.cpp:#ifndef AMREX_USE_GPU
Source/Driver/sum_utils.cpp:        for (MFIter mfi(*mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/sum_utils.cpp:            AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/Driver/sum_utils.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Driver/sum_utils.cpp:    for (MFIter mfi(*fine_mask,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Driver/sum_integrated_quantities.cpp:    amrex::Gpu::synchronize();
Source/Driver/sum_integrated_quantities.cpp:    amrex::Gpu::synchronize();
Source/Driver/sum_integrated_quantities.cpp:       amrex::Gpu::synchronize();
Source/Driver/Prob.H:void prob_param_special_fill(amrex::GpuArray<amrex::Real,max_prob_param>& /*prob_param*/);
Source/Driver/Prob.H:                   const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param);
Source/Driver/Prob.H:                         const amrex::GpuArray<amrex::Real,max_prob_param>& prob_param);
Source/Driver/Prob.H:                          const GpuArray<Real,max_prob_param>& prob_param);
Source/Driver/Prob.H:                                const GpuArray<Real,max_prob_param>& prob_param);
Source/Driver/Prob.H:static void prob_param_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param)
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/ParticleDerive.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/ParticleDerive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/ParticleDerive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/ParticleDerive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/ParticleDerive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/DerivedQuantities/ParticleDerive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/CMakeLists.txt:   if (Nyx_GPU_BACKEND STREQUAL CUDA)
Source/CMakeLists.txt:     target_link_libraries(nyxcore PUBLIC SUNDIALS::nveccuda)
Source/CMakeLists.txt:     target_link_libraries(nyxcore PUBLIC SUNDIALS::cvode_fused_cuda)
Source/CMakeLists.txt:   if (Nyx_GPU_BACKEND STREQUAL HIP)
Source/CMakeLists.txt:   if (Nyx_GPU_BACKEND STREQUAL SYCL)
Source/CMakeLists.txt:#                               Nyx CUDA and buildInfo                       #
Source/CMakeLists.txt:if (Nyx_GPU_BACKEND STREQUAL "CUDA")
Source/CMakeLists.txt:   setup_target_for_cuda_compilation( nyxcore )
Source/TimeStep/Nyx_update_state_with_sources.cpp:    for (amrex::MFIter mfi(S_new, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) 
Source/TimeStep/Nyx_update_state_with_sources.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
Source/TimeStep/Nyx_update_state_with_sources.cpp:    for (amrex::MFIter mfi(S_new, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) 
Source/TimeStep/Nyx_update_state_with_sources.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
Source/TimeStep/Nyx_enforce_minimum_density.H:AMREX_GPU_DEVICE
Source/TimeStep/Nyx_enforce_minimum_density.H:AMREX_GPU_DEVICE
Source/TimeStep/Nyx_enforce_minimum_density.H:AMREX_GPU_DEVICE
Source/TimeStep/Nyx_sources.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeStep/Nyx_sources.cpp:    for (MFIter mfi(S_old,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeStep/Nyx_sources.cpp:        amrex::Gpu::streamSynchronize();
Source/TimeStep/Nyx_sources.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeStep/Nyx_sources.cpp:    for (MFIter mfi(S_old,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeStep/Nyx_correct_gsrc.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeStep/Nyx_correct_gsrc.cpp:    for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeStep/Nyx_correct_gsrc.cpp:        AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/TimeStep/Nyx_time_center_sources.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeStep/Nyx_time_center_sources.cpp:    for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeStep/Nyx_time_center_sources.cpp:          AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/TimeStep/Nyx_enforce_minimum_density.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeStep/Nyx_enforce_minimum_density.cpp:        for (MFIter mfi(hydro_source,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeStep/Nyx_enforce_minimum_density.cpp:            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/TimeStep/Nyx_enforce_minimum_density.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeStep/Nyx_enforce_minimum_density.cpp:    for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeStep/Nyx_enforce_minimum_density.cpp:         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/TimeStep/Nyx_enforce_minimum_density.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeStep/Nyx_enforce_minimum_density.cpp:        for (MFIter mfi(Sborder,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeStep/Nyx_enforce_minimum_density.cpp:            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/TimeStep/Nyx_enforce_minimum_density.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/TimeStep/Nyx_enforce_minimum_density.cpp:        for (MFIter mfi(Sborder,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/TimeStep/Nyx_enforce_minimum_density.cpp:            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Hydro/Hydro.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())               \
Source/Hydro/Hydro.cpp:      const auto hydro_MFItInfo = MFItInfo().EnableTiling(hydro_tile_size).SetNumStreams(Nyx::minimize_memory ? 1 : Gpu::numGpuStreams());
Source/Hydro/Hydro.cpp:      for ( amrex::MFIter mfi(S_new, hydro_MFItInfo); mfi.isValid(); Nyx::minimize_memory ? Gpu::Device::streamSynchronize() : Gpu::Device::synchronize(), ++mfi)
Source/Hydro/Hydro.cpp:        amrex::GpuArray<amrex::FArrayBox, AMREX_SPACEDIM> flux;
Source/Hydro/Hydro.cpp:        // Get Arrays to pass to the gpu.
Source/Hydro/Hydro.cpp:        amrex::ParallelFor(qbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
Source/Hydro/Hydro.cpp:        amrex::ParallelFor(qbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
Source/Hydro/Hydro.cpp:        const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM>
Source/Hydro/Hydro.cpp:  const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
Source/Hydro/Hydro.cpp:  amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> qec_arr
Source/Hydro/Hydro.cpp:  amrex::ParallelFor(bxg2, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
Source/Hydro/Hydro.cpp:  const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
Source/Hydro/Hydro.cpp:    const GpuArray<Real,AMREX_SPACEDIM>  area{ del[1]*del[2], del[0]*del[2], del[0]*del[1] };
Source/Hydro/Hydro.cpp:    amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Hydro.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Hydro.cpp:  const GpuArray<Real,8> a_fact {a_half_inv,a_new_inv,a_new_inv,a_new_inv,
Source/Hydro/Hydro.cpp:    amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
Source/Hydro/Godunov.cpp:// Host function to call gpu hydro functions
Source/Hydro/Godunov.cpp:      amrex::ParallelFor(bxg2, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:    xflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:    yflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:    zflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:  amrex::ParallelFor(txbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:    txfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:  amrex::ParallelFor(tybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:    tyfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:  amrex::ParallelFor(tzbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:    tzfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:  amrex::ParallelFor(tyzbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:  amrex::ParallelFor(xfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:  amrex::ParallelFor(txzbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:  amrex::ParallelFor(yfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:  amrex::ParallelFor(txybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:  amrex::ParallelFor(zfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Godunov.cpp:  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Hydro.H:AMREX_GPU_DEVICE
Source/Hydro/Hydro.H:AMREX_GPU_DEVICE
Source/Hydro/Hydro.H:AMREX_GPU_DEVICE
Source/Hydro/Hydro.H:AMREX_GPU_DEVICE
Source/Hydro/Hydro.H:AMREX_GPU_DEVICE
Source/Hydro/Hydro.H:  const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
Source/Hydro/Hydro.H:  const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
Source/Hydro/Hydro.H:  const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
Source/Hydro/PLM.H:  of PeleC cuda. As per the convention of AMReX, inlined functions are defined
Source/Hydro/PLM.H:AMREX_GPU_DEVICE
Source/Hydro/PLM.H:AMREX_GPU_DEVICE
Source/Hydro/PLM.H:AMREX_GPU_DEVICE
Source/Hydro/PLM.H:AMREX_GPU_DEVICE
Source/Hydro/trace_ppm.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Hydro/PPM.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Hydro/PPM.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Hydro/PPM.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Hydro/Godunov.H:  of PeleC cuda. As per the convention of AMReX, inlined functions are defined
Source/Hydro/Godunov.H:AMREX_GPU_DEVICE
Source/Hydro/Godunov.H:AMREX_GPU_DEVICE
Source/Hydro/Godunov.H:AMREX_GPU_DEVICE
Source/Hydro/Godunov.H:AMREX_GPU_DEVICE
Source/Hydro/Godunov.H:AMREX_GPU_DEVICE
Source/Hydro/Godunov.H:AMREX_GPU_DEVICE
Source/Hydro/Godunov.H:AMREX_GPU_DEVICE
Source/Hydro/Godunov.H:AMREX_GPU_DEVICE
Source/Hydro/Godunov.H:AMREX_GPU_DEVICE
Source/Hydro/Godunov.H:AMREX_GPU_DEVICE
Source/Hydro/Godunov.H:AMREX_GPU_DEVICE
Source/Hydro/Godunov.H:AMREX_GPU_DEVICE
Source/Hydro/Godunov.H:// Designed for CPU or GPU
Source/Hydro/Godunov.H:AMREX_GPU_DEVICE
Source/Hydro/Utilities.cpp:AMREX_GPU_DEVICE
Source/Hydro/Utilities.cpp:AMREX_GPU_DEVICE
Source/Hydro/Utilities.cpp:AMREX_GPU_HOST_DEVICE
Source/Hydro/Utilities.cpp:AMREX_GPU_DEVICE
Source/Hydro/Utilities.cpp:    GpuArray<Real, QVAR> uR;
Source/Hydro/Utilities.cpp:    GpuArray<Real, QVAR> uL;
Source/Hydro/Utilities.cpp:    GpuArray<Real, QVAR> qR;
Source/Hydro/Utilities.cpp:    GpuArray<Real, QVAR> qL;
Source/Hydro/Utilities.cpp:    GpuArray<int, 3> idxL;
Source/Hydro/Utilities.cpp:    GpuArray<Real, QVAR> fluxL;
Source/Hydro/Utilities.cpp:    GpuArray<Real, QVAR> fluxR;
Source/Hydro/Utilities.cpp:    GpuArray<Real, QVAR> fluxLF;
Source/Hydro/Riemann.H:AMREX_GPU_DEVICE
Source/Hydro/strang_hydro.cpp:      amrex::Gpu::Device::streamSynchronize();
Source/Hydro/strang_hydro.cpp:    //assume user-provided source is not CUDA
Source/Hydro/strang_hydro.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Hydro/strang_hydro.cpp:        amrex::Gpu::Device::streamSynchronize();
Source/Hydro/strang_hydro.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Hydro/strang_hydro.cpp:    for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Hydro/strang_hydro.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Hydro/strang_hydro.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Hydro/strang_hydro.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Hydro/sdc_hydro.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Hydro/Utilities.H:AMREX_GPU_DEVICE
Source/Hydro/Utilities.H:AMREX_GPU_DEVICE
Source/Hydro/Utilities.H:AMREX_GPU_DEVICE
Source/Hydro/Utilities.H:    box, Ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/Hydro/Utilities.H:  amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Utilities.H:    box, Ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
Source/Hydro/Utilities.H:  amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Hydro/Utilities.H:AMREX_GPU_HOST_DEVICE
Source/HeatCool/integrate_state_with_source_3d.cpp:#if (defined(_OPENMP) && !defined(AMREX_USE_GPU))
Source/HeatCool/integrate_state_with_source_3d.cpp:#ifdef AMREX_USE_CUDA
Source/HeatCool/integrate_state_with_source_3d.cpp:#include <nvector/nvector_cuda.h>
Source/HeatCool/integrate_state_with_source_3d.cpp:  //#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/HeatCool/integrate_state_with_source_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_with_source_3d.cpp:  const auto tiling = (TilingIfNotGPU() && sundials_use_tiling) ? MFItInfo().EnableTiling(sundials_tile_size) : MFItInfo();
Source/HeatCool/integrate_state_with_source_3d.cpp:  const auto tiling = (TilingIfNotGPU() && sundials_use_tiling) ? MFItInfo().EnableTiling(sundials_tile_size) : MFItInfo();
Source/HeatCool/integrate_state_with_source_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_with_source_3d.cpp:      amrex::Gpu::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_with_source_3d.cpp:#ifdef AMREX_USE_CUDA
Source/HeatCool/integrate_state_with_source_3d.cpp:      auto currentstream = amrex::Gpu::Device::gpuStream();
Source/HeatCool/integrate_state_with_source_3d.cpp:      //     long N = ((tbx.numPts()+AMREX_GPU_NCELLS_PER_THREAD-1)/AMREX_GPU_NCELLS_PER_THREAD);
Source/HeatCool/integrate_state_with_source_3d.cpp:      //     SUNCudaThreadDirectExecPolicy stream_exec_policy(AMREX_GPU_MAX_THREADS, currentstream);
Source/HeatCool/integrate_state_with_source_3d.cpp:      //     SUNCudaGridStrideExecPolicy grid_exec_policy(AMREX_GPU_MAX_THREADS, std::max((N + AMREX_GPU_MAX_THREADS - 1) / AMREX_GPU_MAX_THREADS, static_cast<Long>(1)), currentstream);
Source/HeatCool/integrate_state_with_source_3d.cpp:      //     SUNCudaBlockReduceExecPolicy reduce_exec_policy(AMREX_GPU_MAX_THREADS, std::max((N + AMREX_GPU_MAX_THREADS - 1) / AMREX_GPU_MAX_THREADS, static_cast<Long>(1)), currentstream);
Source/HeatCool/integrate_state_with_source_3d.cpp:      SUNCudaExecPolicy* stream_exec_policy = new SUNCudaThreadDirectExecPolicy(256, currentstream);
Source/HeatCool/integrate_state_with_source_3d.cpp:      SUNCudaExecPolicy* reduce_exec_policy;
Source/HeatCool/integrate_state_with_source_3d.cpp:        reduce_exec_policy = new SUNCudaBlockReduceAtomicExecPolicy(256, 0, currentstream);
Source/HeatCool/integrate_state_with_source_3d.cpp:        reduce_exec_policy = new SUNCudaBlockReduceExecPolicy(256, 0, currentstream);
Source/HeatCool/integrate_state_with_source_3d.cpp:                  u = N_VNewWithMemHelp_Cuda(neq, 1, *amrex::sundials::The_SUNMemory_Helper(), *amrex::sundials::The_Sundials_Context());  /* Allocate u vector */
Source/HeatCool/integrate_state_with_source_3d.cpp:                  u = N_VNewManaged_Cuda(neq, *amrex::sundials::The_Sundials_Context());  /* Allocate u vector */
Source/HeatCool/integrate_state_with_source_3d.cpp:                  u = N_VNew_Cuda(neq, *amrex::sundials::The_Sundials_Context());  /* Allocate u vector */
Source/HeatCool/integrate_state_with_source_3d.cpp:                amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:                u = N_VMakeManaged_Cuda(neq,dptr, *amrex::sundials::The_Sundials_Context());  /* Allocate u vector */
Source/HeatCool/integrate_state_with_source_3d.cpp:                e_orig = N_VMakeManaged_Cuda(neq,eptr, *amrex::sundials::The_Sundials_Context());  /* Allocate u vector */
Source/HeatCool/integrate_state_with_source_3d.cpp:                N_VSetKernelExecPolicy_Cuda(e_orig, stream_exec_policy, reduce_exec_policy);
Source/HeatCool/integrate_state_with_source_3d.cpp:                N_VSetKernelExecPolicy_Cuda(u, stream_exec_policy, reduce_exec_policy);
Source/HeatCool/integrate_state_with_source_3d.cpp:                abstol_vec = N_VMakeManaged_Cuda(neq,abstol_ptr, *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:                N_VSetKernelExecPolicy_Cuda(abstol_vec, stream_exec_policy, reduce_exec_policy);
Source/HeatCool/integrate_state_with_source_3d.cpp:                abstol_achieve_vec = N_VMakeManaged_Cuda(neq,abstol_achieve_ptr, *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:                N_VSetKernelExecPolicy_Cuda(abstol_achieve_vec, stream_exec_policy, reduce_exec_policy);
Source/HeatCool/integrate_state_with_source_3d.cpp:                T_vec = N_VMakeManaged_Cuda(neq, T_vode, *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:                ne_vec = N_VMakeManaged_Cuda(neq, ne_vode, *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:                rho_vec = N_VMakeManaged_Cuda(neq, rho_vode, *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:                rho_init_vec = N_VMakeManaged_Cuda(neq, rho_init_vode, *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:                rho_src_vec = N_VMakeManaged_Cuda(neq, rho_src_vode, *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:                rhoe_src_vec = N_VMakeManaged_Cuda(neq, rhoe_src_vode, *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:                e_src_vec = N_VMakeManaged_Cuda(neq, e_src_vode, *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:                IR_vec = N_VMakeManaged_Cuda(neq, IR_vode, *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:                N_VSetKernelExecPolicy_Cuda(T_vec, stream_exec_policy, reduce_exec_policy);
Source/HeatCool/integrate_state_with_source_3d.cpp:                N_VSetKernelExecPolicy_Cuda(ne_vec, stream_exec_policy, reduce_exec_policy);
Source/HeatCool/integrate_state_with_source_3d.cpp:                N_VSetKernelExecPolicy_Cuda(rho_vec, stream_exec_policy, reduce_exec_policy);
Source/HeatCool/integrate_state_with_source_3d.cpp:                N_VSetKernelExecPolicy_Cuda(rho_init_vec, stream_exec_policy, reduce_exec_policy);
Source/HeatCool/integrate_state_with_source_3d.cpp:                N_VSetKernelExecPolicy_Cuda(rho_src_vec, stream_exec_policy, reduce_exec_policy);
Source/HeatCool/integrate_state_with_source_3d.cpp:                N_VSetKernelExecPolicy_Cuda(rhoe_src_vec, stream_exec_policy, reduce_exec_policy);
Source/HeatCool/integrate_state_with_source_3d.cpp:                N_VSetKernelExecPolicy_Cuda(e_src_vec, stream_exec_policy, reduce_exec_policy);
Source/HeatCool/integrate_state_with_source_3d.cpp:                N_VSetKernelExecPolicy_Cuda(IR_vec, stream_exec_policy, reduce_exec_policy);
Source/HeatCool/integrate_state_with_source_3d.cpp:                amrex::Gpu::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:            u = N_VNewWithMemHelp_Cuda(neq, /*use_managed_mem=*/true, *amrex::sundials::The_SUNMemory_Helper(), *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:            u = N_VNewWithMemHelp_Cuda(neq, /*use_managed_mem=*/false, *amrex::sundials::The_SUNMemory_Helper(), *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:      N_VSetKernelExecPolicy_Cuda(u, stream_exec_policy, reduce_exec_policy);
Source/HeatCool/integrate_state_with_source_3d.cpp:      amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:      auto currentstream = amrex::Gpu::Device::gpuStream();
Source/HeatCool/integrate_state_with_source_3d.cpp:      //     long N = ((tbx.numPts()+AMREX_GPU_NCELLS_PER_THREAD-1)/AMREX_GPU_NCELLS_PER_THREAD);
Source/HeatCool/integrate_state_with_source_3d.cpp:      //     SUNHipThreadDirectExecPolicy stream_exec_policy(AMREX_GPU_MAX_THREADS, currentstream);
Source/HeatCool/integrate_state_with_source_3d.cpp:      //     SUNHipGridStrideExecPolicy grid_exec_policy(AMREX_GPU_MAX_THREADS, std::max((N + AMREX_GPU_MAX_THREADS - 1) / AMREX_GPU_MAX_THREADS, static_cast<Long>(1)), currentstream);
Source/HeatCool/integrate_state_with_source_3d.cpp:      //     SUNHipBlockReduceExecPolicy reduce_exec_policy(AMREX_GPU_MAX_THREADS, std::max((N + AMREX_GPU_MAX_THREADS - 1) / AMREX_GPU_MAX_THREADS, static_cast<Long>(1)), currentstream);
Source/HeatCool/integrate_state_with_source_3d.cpp:                // might need a cuda analog to setting exec policy
Source/HeatCool/integrate_state_with_source_3d.cpp:                amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:// might need a cuda analog to setting exec policy
Source/HeatCool/integrate_state_with_source_3d.cpp:      amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:      auto currentstream = amrex::Gpu::Device::streamQueue();
Source/HeatCool/integrate_state_with_source_3d.cpp:      //     long N = ((tbx.numPts()+AMREX_GPU_NCELLS_PER_THREAD-1)/AMREX_GPU_NCELLS_PER_THREAD);
Source/HeatCool/integrate_state_with_source_3d.cpp:      //     SUNSyclThreadDirectExecPolicy stream_exec_policy(AMREX_GPU_MAX_THREADS);
Source/HeatCool/integrate_state_with_source_3d.cpp:      //     SUNSyclGridStrideExecPolicy grid_exec_policy(AMREX_GPU_MAX_THREADS, std::max((N + AMREX_GPU_MAX_THREADS - 1) / AMREX_GPU_MAX_THREADS, static_cast<Long>(1)));
Source/HeatCool/integrate_state_with_source_3d.cpp:      //     SUNSyclBlockReduceExecPolicy reduce_exec_policy(AMREX_GPU_MAX_THREADS, std::max((N + AMREX_GPU_MAX_THREADS - 1) / AMREX_GPU_MAX_THREADS, static_cast<Long>(1)));
Source/HeatCool/integrate_state_with_source_3d.cpp:                // might need a cuda analog to setting exec policy
Source/HeatCool/integrate_state_with_source_3d.cpp:                amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:              u = N_VNewWithMemHelp_Sycl(neq, /*use_managed_mem=*/true, *amrex::sundials::The_SUNMemory_Helper(), &amrex::Gpu::Device::streamQueue(), *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:         //              u = N_VNewWithMemHelp_Sycl(neq, /*use_managed_mem=*/true, S, &amrex::Gpu::Device::streamQueue(), *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:              u = N_VNewWithMemHelp_Sycl(neq, /*use_managed_mem=*/false, *amrex::sundials::The_SUNMemory_Helper(), &amrex::Gpu::Device::streamQueue(), *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:         //              u = N_VNewWithMemHelp_Sycl(neq, /*use_managed_mem=*/false, S, &amrex::Gpu::Device::streamQueue(), *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_with_source_3d.cpp:// might need a cuda analog to setting exec policy
Source/HeatCool/integrate_state_with_source_3d.cpp:#else  /* else for ndef AMREX_USE_GPU */
Source/HeatCool/integrate_state_with_source_3d.cpp:#endif /* end AMREX_USE_GPU if */
Source/HeatCool/integrate_state_with_source_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_with_source_3d.cpp:      amrex::Gpu::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:      amrex::Gpu::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_with_source_3d.cpp:      amrex::ParallelFor ( tbx, [=] AMREX_GPU_DEVICE (int i,int j,int k)
Source/HeatCool/integrate_state_with_source_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_with_source_3d.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:      amrex::Gpu::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:            amrex::Gpu::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:            amrex::Gpu::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:            //                              amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:            amrex::Gpu::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_with_source_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_with_source_3d.cpp:            amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:            amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:    amrex::Gpu::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:#ifdef AMREX_USE_CUDA
Source/HeatCool/integrate_state_with_source_3d.cpp:#if defined(AMREX_USE_GPU)
Source/HeatCool/integrate_state_with_source_3d.cpp:    amrex::Gpu::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_with_source_3d.cpp:  amrex::Gpu::streamSynchronize();
Source/HeatCool/integrate_state_with_source_3d.cpp:  amrex::Gpu::streamSynchronize();
Source/HeatCool/f_rhs.H:AMREX_GPU_DEVICE
Source/HeatCool/f_rhs.H:AMREX_GPU_DEVICE
Source/HeatCool/f_rhs_struct.H:#ifdef AMREX_USE_GPU
Source/HeatCool/f_rhs_struct.H:    amrex::Gpu::htod_memcpy(f_rhs_data,&f_rhs_data_host,sizeof(RhsData));
Source/HeatCool/f_rhs_struct.H:AMREX_GPU_DEVICE
Source/HeatCool/f_rhs_struct.H:AMREX_GPU_DEVICE
Source/HeatCool/f_rhs_struct.H:AMREX_GPU_DEVICE
Source/HeatCool/f_rhs_struct.H:AMREX_GPU_DEVICE
Source/HeatCool/f_rhs_struct.H:AMREX_GPU_DEVICE
Source/HeatCool/f_rhs_struct.H:AMREX_GPU_DEVICE
Source/HeatCool/f_rhs_struct.H:#ifndef AMREX_USE_GPU
Source/HeatCool/f_rhs_struct.H:#ifndef AMREX_USE_GPU
Source/HeatCool/integrate_state_vec_3d.cpp:#ifndef AMREX_USE_GPU
Source/HeatCool/integrate_state_vec_3d.cpp:#ifdef AMREX_USE_CUDA
Source/HeatCool/integrate_state_vec_3d.cpp:#include <nvector/nvector_cuda.h>
Source/HeatCool/integrate_state_vec_3d.cpp:  bool tiling = (sundials_use_tiling && TilingIfNotGPU());
Source/HeatCool/integrate_state_vec_3d.cpp:      amrex::Gpu::streamSynchronize();
Source/HeatCool/integrate_state_vec_3d.cpp:#ifdef AMREX_USE_CUDA
Source/HeatCool/integrate_state_vec_3d.cpp:                        cudaStream_t currentStream = amrex::Gpu::Device::cudaStream();
Source/HeatCool/integrate_state_vec_3d.cpp:                        //     long N = ((tbx.numPts()+AMREX_GPU_NCELLS_PER_THREAD-1)/AMREX_GPU_NCELLS_PER_THREAD);
Source/HeatCool/integrate_state_vec_3d.cpp:                        //     SUNCudaThreadDirectExecPolicy stream_exec_policy(AMREX_GPU_MAX_THREADS, currentStream);
Source/HeatCool/integrate_state_vec_3d.cpp:                        //     SUNCudaGridStrideExecPolicy grid_exec_policy(AMREX_GPU_MAX_THREADS, std::max((N + AMREX_GPU_MAX_THREADS - 1) / AMREX_GPU_MAX_THREADS, static_cast<Long>(1)), currentStream);
Source/HeatCool/integrate_state_vec_3d.cpp:                        //     SUNCudaBlockReduceExecPolicy reduce_exec_policy(AMREX_GPU_MAX_THREADS, std::max((N + AMREX_GPU_MAX_THREADS - 1) / AMREX_GPU_MAX_THREADS, static_cast<Long>(1)), currentStream);
Source/HeatCool/integrate_state_vec_3d.cpp:                        SUNCudaThreadDirectExecPolicy stream_exec_policy(256, currentStream);
Source/HeatCool/integrate_state_vec_3d.cpp:                        SUNCudaBlockReduceExecPolicy reduce_exec_policy(256, 0, currentStream);
Source/HeatCool/integrate_state_vec_3d.cpp:                            u = N_VNewWithMemHelp_Cuda(neq, 1, *amrex::sundials::The_SUNMemory_Helper(),
Source/HeatCool/integrate_state_vec_3d.cpp:                            u = N_VNewManaged_Cuda(neq, *amrex::sundials::The_Sundials_Context());  /* Allocate u vector */
Source/HeatCool/integrate_state_vec_3d.cpp:                          dptr=N_VGetDeviceArrayPointer_Cuda(u);
Source/HeatCool/integrate_state_vec_3d.cpp:                            e_orig = N_VNewWithMemHelp_Cuda(neq, 1, *amrex::sundials::The_SUNMemory_Helper(),
Source/HeatCool/integrate_state_vec_3d.cpp:                            e_orig = N_VNewManaged_Cuda(neq, *amrex::sundials::The_Sundials_Context());  /* Allocate u vector */
Source/HeatCool/integrate_state_vec_3d.cpp:                          eptr=N_VGetDeviceArrayPointer_Cuda(e_orig);
Source/HeatCool/integrate_state_vec_3d.cpp:                          N_VSetKernelExecPolicy_Cuda(e_orig, &stream_exec_policy, &reduce_exec_policy);
Source/HeatCool/integrate_state_vec_3d.cpp:                          N_VSetKernelExecPolicy_Cuda(u, &stream_exec_policy, &reduce_exec_policy);
Source/HeatCool/integrate_state_vec_3d.cpp:                            Data = N_VNewWithMemHelp_Cuda(4*neq, 1, *amrex::sundials::The_SUNMemory_Helper(),
Source/HeatCool/integrate_state_vec_3d.cpp:                            Data = N_VNewManaged_Cuda(4*neq, *amrex::sundials::The_Sundials_Context());  /* Allocate u vector */
Source/HeatCool/integrate_state_vec_3d.cpp:                          rparh = N_VGetDeviceArrayPointer_Cuda(Data);
Source/HeatCool/integrate_state_vec_3d.cpp:                          N_VSetKernelExecPolicy_Cuda(Data, &stream_exec_policy, &reduce_exec_policy);
Source/HeatCool/integrate_state_vec_3d.cpp:                            abstol_vec = N_VNewWithMemHelp_Cuda(neq, 1, *amrex::sundials::The_SUNMemory_Helper(),
Source/HeatCool/integrate_state_vec_3d.cpp:                            abstol_vec = N_VNewManaged_Cuda(neq, *amrex::sundials::The_Sundials_Context());  /* Allocate u vector */
Source/HeatCool/integrate_state_vec_3d.cpp:                          abstol_ptr = N_VGetDeviceArrayPointer_Cuda(abstol_vec);
Source/HeatCool/integrate_state_vec_3d.cpp:                          N_VSetKernelExecPolicy_Cuda(abstol_vec, &stream_exec_policy, &reduce_exec_policy);
Source/HeatCool/integrate_state_vec_3d.cpp:                          amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_vec_3d.cpp:                          u = N_VMakeManaged_Cuda(neq,dptr, *amrex::sundials::The_Sundials_Context());  /* Allocate u vector */
Source/HeatCool/integrate_state_vec_3d.cpp:                          e_orig = N_VMakeManaged_Cuda(neq,eptr, *amrex::sundials::The_Sundials_Context());  /* Allocate u vector */
Source/HeatCool/integrate_state_vec_3d.cpp:                          N_VSetKernelExecPolicy_Cuda(e_orig, &stream_exec_policy, &reduce_exec_policy);
Source/HeatCool/integrate_state_vec_3d.cpp:                          N_VSetKernelExecPolicy_Cuda(u, &stream_exec_policy, &reduce_exec_policy);
Source/HeatCool/integrate_state_vec_3d.cpp:                          Data = N_VMakeManaged_Cuda(4*neq,rparh, *amrex::sundials::The_Sundials_Context());  // Allocate u vector
Source/HeatCool/integrate_state_vec_3d.cpp:                          N_VSetKernelExecPolicy_Cuda(Data, &stream_exec_policy, &reduce_exec_policy);
Source/HeatCool/integrate_state_vec_3d.cpp:                          abstol_vec = N_VMakeManaged_Cuda(neq,abstol_ptr, *amrex::sundials::The_Sundials_Context());
Source/HeatCool/integrate_state_vec_3d.cpp:                          N_VSetKernelExecPolicy_Cuda(abstol_vec, &stream_exec_policy, &reduce_exec_policy);
Source/HeatCool/integrate_state_vec_3d.cpp:                          amrex::Gpu::streamSynchronize();
Source/HeatCool/integrate_state_vec_3d.cpp:#if (defined(_OPENMP) && !defined(AMREX_USE_GPU))
Source/HeatCool/integrate_state_vec_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_vec_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_vec_3d.cpp:                                amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_vec_3d.cpp:                                amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_vec_3d.cpp:                                //                              amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_vec_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_vec_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_vec_3d.cpp:                                amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_vec_3d.cpp:                                amrex::Gpu::Device::streamSynchronize();
Source/HeatCool/integrate_state_vec_3d.cpp:#ifdef AMREX_USE_CUDA
Source/HeatCool/integrate_state_vec_3d.cpp:    bool tiling = (sundials_use_tiling && TilingIfNotGPU());
Source/HeatCool/integrate_state_vec_3d.cpp:#ifdef AMREX_USE_GPU
Source/HeatCool/integrate_state_vec_3d.cpp:  amrex::Gpu::streamSynchronize();
Source/Forcing/Forcing.cpp:#ifdef AMREX_USE_GPU
Source/Forcing/Forcing.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
Source/Forcing/Forcing.cpp:    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/EOS/eos_hc.H:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE
Source/EOS/eos_hc.H:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void ion_n_device(AtomicRates* atomic_rates, const int JH, const int JHe,
Source/EOS/eos_hc.H:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE
Source/EOS/eos_hc.H:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void nyx_eos_T_given_Re_device(
Source/EOS/eos_hc.H:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void nyx_eos_T_given_Re_device(
Source/EOS/eos_hc.H:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void nyx_eos_given_RT(
Source/EOS/reset_internal_e.H:AMREX_GPU_DEVICE
Source/EOS/eos_adiabatic.H:  AMREX_GPU_DEVICE void fort_nyx_eos_T_given_Re_device(int JH, int JHe, Real* T, Real* Ne, Real R,Real e,Real comoving_a);
Source/EOS/eos_adiabatic.H:  AMREX_GPU_DEVICE void fort_nyx_eos_given_RT(Real* e, Real* P, Real R, Real T, Real Ne,Real comoving_a);
Source/EOS/eos_adiabatic.H:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void nyx_eos_T_given_Re_device(
Source/EOS/eos_adiabatic.H:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE void nyx_eos_given_RT(
Source/EOS/atomic_rates.H:#ifdef AMREX_USE_GPU
Source/EOS/atomic_rates.H:    amrex::Gpu::htod_memcpy(atomic_rates_glob,&atomic_rates_host,sizeof(AtomicRates));
Source/EOS/atomic_rates.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Gravity/Gravity.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Gravity/Gravity.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Gravity/Gravity.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Gravity/Gravity.cpp:    const GpuArray<Real,AMREX_SPACEDIM>  area{ dx[1]*dx[2], dx[0]*dx[2], dx[0]*dx[1] };
Source/Gravity/Gravity.cpp:    amrex::Gpu::Device::synchronize();
Source/Gravity/Gravity.cpp:    amrex::Gpu::LaunchSafeGuard lsg(false);
Source/Gravity/Gravity.cpp:        amrex::Gpu::Device::streamSynchronize();
Source/Gravity/Gravity.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Gravity/Gravity.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Gravity/Gravity.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Gravity/Gravity.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Gravity/Gravity.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Gravity/Gravity.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Gravity/Gravity.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Gravity/Gravity.cpp:          for (MFIter mfi(rhs,TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/Initialization/bc_fill.cpp:    AMREX_GPU_DEVICE
Source/Initialization/bc_fill.cpp:  AMREX_GPU_DEVICE
Source/Initialization/bc_fill.cpp:    GpuBndryFuncFab<NyxFillExtDir> gpu_bndry_func(NyxFillExtDir{});
Source/Initialization/bc_fill.cpp:    gpu_bndry_func(bx,data,dcomp,numcomp,geom,time,bcr,bcomp,scomp);
Source/Initialization/Nyx_initdata.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Initialization/Nyx_initdata.cpp:            for (MFIter mfi(D_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Initialization/Nyx_initdata.cpp:                               bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/Initialization/Nyx_initdata.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Initialization/Nyx_initdata.cpp:            for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Initialization/Nyx_initdata.cpp:                GpuArray<amrex::Real,max_prob_param> prob_param;
Source/Initialization/Nyx_initdata.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Initialization/Nyx_initdata.cpp:            for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Initialization/Nyx_initdata.cpp:                GpuArray<amrex::Real,max_prob_param> prob_param;
Source/Initialization/Nyx_initdata.cpp:    amrex::Gpu::Device::synchronize();
Source/Initialization/Nyx_initdata.cpp:    amrex::Gpu::Device::synchronize();
Source/Initialization/Nyx_initdata.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Initialization/Nyx_initdata.cpp:        for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Initialization/Nyx_initdata.cpp:          AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/Initialization/Nyx_initdata.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Initialization/Nyx_initdata.cpp:    for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Initialization/Nyx_initdata.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Initialization/Nyx_initdata.cpp:        amrex::Gpu::Device::streamSynchronize();
Source/Initialization/Nyx_setup.cpp:    amrex::Gpu::streamSynchronize();
Source/Initialization/Nyx_setup.cpp:    bndryfunc.setRunOnGPU(true);  // I promise the bc function will launch gpu kernels.
Source/Initialization/Nyx_setup.cpp:    bndryfunc.setRunOnGPU(true);  // I promise the bc function will launch gpu kernels.
cmake/NyxSetupAMReX.cmake:   if (Nyx_GPU_BACKEND STREQUAL CUDA)
cmake/NyxSetupAMReX.cmake:      list(APPEND AMREX_REQUIRED_COMPONENTS CUDA)
cmake/NyxSetupAMReX.cmake:   elseif (Nyx_GPU_BACKEND STREQUAL HIP)
cmake/NyxSetupAMReX.cmake:   elseif (Nyx_GPU_BACKEND STREQUAL SYCL)
cmake/NyxSetupAMReX.cmake:   # We load this here so we have the CUDA helper functions
cmake/NyxSetupAMReX.cmake:   if (ENABLE_CUDA)
cmake/NyxSetupAMReX.cmake:   set(AMReX_GPU_BACKEND          ${Nyx_GPU_BACKEND}         CACHE INTERNAL "" )
cmake/NyxSetupAMReX.cmake:   # If CUDA is required, enable the language BEFORE adding the AMReX directory
cmake/NyxSetupAMReX.cmake:   # Since AMReX_SetupCUDA has an include guard, it will be included only once here.
cmake/NyxSetupAMReX.cmake:   # The reason for enabling CUDA before adding the AMReX subdirectory is that
cmake/NyxSetupAMReX.cmake:   # the top-most directory needs to setup the CUDA language before a CUDA-enabled target
cmake/NyxSetupAMReX.cmake:   # it will not setup CUDA here!
cmake/NyxSetupAMReX.cmake:   if(Nyx_GPU_BACKEND STREQUAL CUDA)
cmake/NyxSetupAMReX.cmake:      include(AMReX_SetupCUDA)
cmake/NyxSetupExecutable.cmake:   if (Nyx_GPU_BACKEND STREQUAL "CUDA")
cmake/NyxSetupExecutable.cmake:      setup_target_for_cuda_compilation( ${_exe_name} )
cmake/NyxSetupSUNDIALS.cmake:   if (Nyx_GPU_BACKEND STREQUAL CUDA)
cmake/NyxSetupSUNDIALS.cmake:     set(ENABLE_CUDA                  ON                      CACHE INTERNAL "" )
cmake/NyxSetupSUNDIALS.cmake:      set(ENABLE_CUDA                  OFF                     CACHE INTERNAL "" )
cmake/NyxSetupSUNDIALS.cmake:      if (Nyx_GPU_BACKEND STREQUAL HIP)
cmake/NyxSetupSUNDIALS.cmake:      if (Nyx_GPU_BACKEND STREQUAL SYCL)
cmake/NyxSetupSUNDIALS.cmake:   if (Nyx_GPU_BACKEND STREQUAL CUDA)
cmake/NyxSetupSUNDIALS.cmake:      add_library(SUNDIALS::nveccuda ALIAS sundials_nveccuda_shared)
cmake/NyxSetupSUNDIALS.cmake:      add_library(SUNDIALS::cvode_fused_cuda ALIAS sundials_cvode_fused_cuda_shared)
cmake/NyxSetupSUNDIALS.cmake:      get_target_property(SUNDIALS_INCLUDES sundials_nveccuda_shared SOURCES)
cmake/NyxSetupSUNDIALS.cmake:      sundials_nveccuda_shared PROPERTIES PUBLIC_HEADER "${SUNDIALS_INCLUDES}")
cmake/NyxSetupSUNDIALS.cmake:   if (Nyx_GPU_BACKEND STREQUAL HIP)
cmake/NyxSetupSUNDIALS.cmake:   if (Nyx_GPU_BACKEND STREQUAL SYCL)
cmake/NyxSetupSUNDIALS.cmake:   if (Nyx_GPU_BACKEND STREQUAL CUDA)
cmake/NyxSetupSUNDIALS.cmake:      add_library(SUNDIALS::nveccuda ALIAS sundials_nveccuda_static)
cmake/NyxSetupSUNDIALS.cmake:      add_library(SUNDIALS::cvode_fused_cuda ALIAS sundials_cvode_fused_cuda_static)
cmake/NyxSetupSUNDIALS.cmake:      get_target_property(SUNDIALS_INCLUDES sundials_nveccuda_static SOURCES)
cmake/NyxSetupSUNDIALS.cmake:      sundials_nveccuda_static PROPERTIES PUBLIC_HEADER "${SUNDIALS_INCLUDES}")
cmake/NyxSetupSUNDIALS.cmake:   if (Nyx_GPU_BACKEND STREQUAL HIP)
cmake/NyxSetupSUNDIALS.cmake:   if (Nyx_GPU_BACKEND STREQUAL SYCL)

```
