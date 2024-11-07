# https://github.com/AMReX-Astro/Castro

```console
Docs/source/creating_a_problem.rst:The variables will all be initialized for the GPU as well.
Docs/source/creating_a_problem.rst:    can make the data managed for GPUs.
Docs/source/getting_started.rst: * A C++17 (or later) compiler (for GCC, we need >= 9.0 for CUDA compilation)
Docs/source/getting_started.rst:For running in parallel, an MPI library is required.  For running on GPUs:
Docs/source/getting_started.rst:* CUDA 11 or later is required for NVIDIA GPUs
Docs/source/getting_started.rst:* ROCM 4.5 or later is required for AMD GPUs
Docs/source/getting_started.rst:   with the proprietary Nvidia graphics driver. A fix for this is
Docs/source/mpi_plus_x.rst:Running Options: CPUs and GPUs
Docs/source/mpi_plus_x.rst:CPU-based computing and CUDA is used for GPUs.
Docs/source/mpi_plus_x.rst:Running on GPUs
Docs/source/mpi_plus_x.rst:Castro's compute kernels can run on GPUs and this is the preferred way
Docs/source/mpi_plus_x.rst:to run on supercomputers with GPUs.  The exact same compute kernels
Docs/source/mpi_plus_x.rst:are used on GPUs as on CPUs.
Docs/source/mpi_plus_x.rst:   Almost all of Castro runs on GPUs, with the main exception being
Docs/source/mpi_plus_x.rst:When using GPUs, almost all of the computing is done on the GPUs.  In
Docs/source/mpi_plus_x.rst:GPU thread, to take advantage of the massive parallelism.  The
Docs/source/mpi_plus_x.rst:advantage of GPUs, so entire simulations can be run on the GPU.
Docs/source/mpi_plus_x.rst:Castro / AMReX have an option to use managed memory for the GPU --
Docs/source/mpi_plus_x.rst:By default, Castro will abort if it runs out of GPU memory.  You can
Docs/source/mpi_plus_x.rst:disable this via ``amrex.abort_on_out_of_gpu_memory=0`` -- together
Docs/source/mpi_plus_x.rst:swapped off of the GPU to make more room available.  This is not
Docs/source/mpi_plus_x.rst:recommended -- oversubscribing the GPU memory will severely impact
Docs/source/mpi_plus_x.rst:GPU memory during the hydro advance.  To alleviate this, Castro can
Docs/source/mpi_plus_x.rst:you to run a problem on a smaller number of GPUs if the hydro
Docs/source/mpi_plus_x.rst:NVIDIA GPUs
Docs/source/mpi_plus_x.rst:With NVIDIA GPUs, we use MPI+CUDA, compiled with GCC and the NVIDIA compilers.
Docs/source/mpi_plus_x.rst:  USE_CUDA = TRUE
Docs/source/mpi_plus_x.rst:   For recent GPUs, like the NVIDIA RTX 4090, you may need to change
Docs/source/mpi_plus_x.rst:   the default CUDA architecture.  This can be done by adding:
Docs/source/mpi_plus_x.rst:      CUDA_ARCH=89
Docs/source/mpi_plus_x.rst:   CUDA 11.2 and later can do link time optimization.  This can
Docs/source/mpi_plus_x.rst:      CUDA_LTO=TRUE
Docs/source/mpi_plus_x.rst:AMD GPUs
Docs/source/mpi_plus_x.rst:For AMD GPUs, we use MPI+HIP, compiled with the ROCm compilers.
Docs/source/mpi_plus_x.rst:Printing Warnings from GPU Kernels
Docs/source/mpi_plus_x.rst:.. index:: USE_GPU_PRINTF
Docs/source/mpi_plus_x.rst:triggering a retry in the process).  On GPUs, printing from a kernel
Docs/source/mpi_plus_x.rst:wrapping them in ``#ifndef AMREX_USE_GPU``.
Docs/source/mpi_plus_x.rst:However, for debugging GPU runs, sometimes we want to see these
Docs/source/mpi_plus_x.rst:warnings.  The build option ``USE_GPU_PRINTF=TRUE`` will enable these
Docs/source/mpi_plus_x.rst:(by setting the preprocessor flag ``ALLOW_GPU_PRINTF``).
Docs/source/mpi_plus_x.rst:   Not every warning has been enabled for GPUs.
Docs/source/faq.rst:#. *When I try to use Amrvis with the Nvidia driver, all I see is
Docs/source/faq.rst:           Driver         "nvidia"
Docs/source/faq.rst:           VendorName     "NVIDIA Corporation"
Docs/source/faq.rst:   by running ``nvidia-xconfig`` first.
Docs/source/index.rst:   gpu
Docs/source/gpu.rst:GPU Programming Model
Docs/source/gpu.rst:CPUs and GPUs have separate memory, which means that working on both
Docs/source/gpu.rst:the memory on the host and that on the GPU.
Docs/source/gpu.rst:In Castro, the core design when running on GPUs is that all of the compute
Docs/source/gpu.rst:should be done on the GPU.
Docs/source/gpu.rst:When we compile with ``USE_CUDA=TRUE`` or ``USE_HIP=TRUE``, AMReX will allocate
Docs/source/gpu.rst:a pool of memory on the GPUs and all of the ``StateData`` will be stored there.
Docs/source/gpu.rst:As long as we then do all of the computation on the GPUs, then we don't need
Docs/source/gpu.rst:value in the GPU kernel, the GPU gets access to the pointer to the
Docs/source/gpu.rst:Most AMReX functions will work on the data directly on the GPU (like
Docs/source/gpu.rst:   For a thorough discussion of how the AMReX GPU offloading works
Docs/source/gpu.rst:The main exception for all data being on the GPUs all the time are the
Docs/source/Introduction.rst:  * parallelization via MPI + OpenMP (CPUs), MPI + CUDA (NVIDIA GPUs), or MPI + HIP (AMD GPUs)
Docs/source/Introduction.rst:is done (both on CPU and GPU).
Docs/source/radiation.rst:``USE_MPI``, ``USE_OMP``, and ``USE_CUDA``.
Docs/source/radiation.rst:As an example, to build Hypre on Summit with MPI and CUDA, you
Docs/source/radiation.rst:   CUDA_HOME=$OLCF_CUDA_ROOT HYPRE_CUDA_SM=70 CXX=mpicxx CC=mpicc FC=mpifort ./configure --prefix=/path/to/Hypre/install --with-MPI --with-cuda --enable-unified-memory
Docs/source/radiation.rst:``USE_MPI=TRUE`` and ``USE_CUDA=TRUE``.
Docs/source/build_system.rst:Parallelization and GPUs
Docs/source/build_system.rst:.. index:: USE_MPI, USE_OMP, USE_CUDA, USE_HIP
Docs/source/build_system.rst:The following parameters control how work is divided across nodes, cores, and GPUs.
Docs/source/build_system.rst:  * ``USE_CUDA``: compile with NVIDIA GPU support using CUDA.
Docs/source/build_system.rst:  * ``USE_HIP``: compile with AMD GPU support using HIP.
Docs/source/Preface.rst:The Castro GPU strategy and performance was described in:
Util/model_parser/model_parser.H:AMREX_INLINE AMREX_GPU_HOST_DEVICE
Util/model_parser/model_parser.H:AMREX_INLINE AMREX_GPU_HOST_DEVICE
Util/model_parser/model_parser.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Util/model_parser/model_parser_data.H:#include <AMReX_GpuContainers.H>
Util/model_parser/model_parser_data.H:    extern AMREX_GPU_MANAGED int npts;
Util/model_parser/model_parser_data.H:    extern AMREX_GPU_MANAGED bool initialized;
Util/model_parser/model_parser_data.H:    extern AMREX_GPU_MANAGED amrex::Array1D<initial_model_t, 0, NUM_MODELS-1> profile;
Util/model_parser/model_parser_data.cpp:    AMREX_GPU_MANAGED int npts;
Util/model_parser/model_parser_data.cpp:    AMREX_GPU_MANAGED bool initialized;
Util/model_parser/model_parser_data.cpp:    AMREX_GPU_MANAGED amrex::Array1D<initial_model_t, 0, NUM_MODELS-1> profile;
Util/scripts/diag_parser.py:- if compiled with GPU support, this will have two additional integer fields at
Util/scripts/diag_parser.py:  the end with size `datwidth` for the GPU memory usage
Util/scripts/diag_parser.py:                dtypes[6] = int  # maximum gpu memory used
Util/scripts/diag_parser.py:                dtypes[7] = int  # minimum gpu memory free
Util/scripts/write_probdata.py:                        fout.write(f"  extern AMREX_GPU_MANAGED {p.get_cxx_decl()} {p.name}[NumSpec];\n\n")
Util/scripts/write_probdata.py:                        fout.write(f"  extern AMREX_GPU_MANAGED {p.get_cxx_decl()} {p.name}[{p.size}];\n\n")
Util/scripts/write_probdata.py:                    fout.write(f"  extern AMREX_GPU_MANAGED {p.get_cxx_decl()} {p.name};\n\n")
Util/scripts/write_probdata.py:                        fout.write(f"  AMREX_GPU_MANAGED {p.get_cxx_decl()} problem::{p.name}[NumSpec];\n\n")
Util/scripts/write_probdata.py:                        fout.write(f"  AMREX_GPU_MANAGED {p.get_cxx_decl()} problem::{p.name}[{p.size}];\n\n")
Util/scripts/write_probdata.py:                    fout.write(f"  AMREX_GPU_MANAGED {p.get_cxx_decl()} problem::{p.name};\n\n")
Util/code_checker/clang_static_analysis.py:            if 'ignoring #pragma gpu box' not in m.group(1):
CHANGES.md:  * more GPU error printing (#2944)
CHANGES.md:    as GPU-hours instead of CPU-hours when running on GPUs (#2930)
CHANGES.md:  * We can now output warnings when running on GPUs if you build
CHANGES.md:    with `USE_GPU_PRINTF=TRUE`(#2923, #2928)
CHANGES.md:  * Fix an issue with large kernel sizes with ROCm in the reduction code
CHANGES.md:     can help limit the amount of memory used in GPU builds. (#2153)
CHANGES.md:     22.04 release was extended for GPU builds, as noted below.) However,
CHANGES.md:   * We now abort on GPUs if species do not sum to 1 (#2099)
CHANGES.md:     MPI ranks in a GPU build could result in an incorrect gravitational
CHANGES.md:   * Compiling with the PGI compiler is no longer a requirement for the CUDA build of Castro.
CHANGES.md:     networks implemented only in Fortran  will not be usable on GPUs, and eventually
CHANGES.md:   * The CUDA build no longer has a requirement that amr.blocking_factor
CHANGES.md:     may now result in race conditions and correctness issues in the CUDA
CHANGES.md:   * A CUDA illegal memory access error in Poisson gravity and diffusion
CHANGES.md:     prevented their use in lambda-capture functions on GPUs.  Now the
CHANGES.md:   * AMReX provides CpuBndryFuncFab and GpuBndryFuncFab which are very
CHANGES.md:   * Fixed a bug in the nuclear burning timestep estimator when on GPUs
CHANGES.md:     This is a wrapper for the EOS that must be used for CUDA
CHANGES.md:     places that don't run on the GPU. (#693)
CHANGES.md:     as managed for CUDA, and adds the ability to output the values to
CHANGES.md:   * The job_info file now reports the number of GPUs being used.
CHANGES.md:   * Fix CUDA compilation
CHANGES.md:     on the GPU.
CHANGES.md:     the GPU.  Since the tags_and_untags routine in AMReX is not
CHANGES.md:     GPU-accelerated, we opt to directly fill in the TagBoxArray in
CHANGES.md:   * fixed a bug in the CUDA version of the MOL integrator  * it was
CHANGES.md:     to make them GPU friendly
CHANGES.md:     to GPUs.
CHANGES.md:   * unified some CUDA hydro solver code with the method-of-lines code
CHANGES.md:     to GPUs with CUDA
CHANGES.md:   * A new GPU (CUDA) hydrodynamics solver (based on the
CHANGES.md:     requires the "gpu" branch of AMReX.
CHANGES.md:     at runtime.  This change is needed for the GPU port.
CHANGES.md:     clean-ups for GPU acceleration in the future.
CHANGES.md:   * start of some code cleaning for eventual GPU offload support
CITATION.md:## GPUs and scaling
CITATION.md:For GPU performance, please cite:
Exec/radiation_tests/Rad2Tshock/problem_bc_fill.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/Rad2Tshock/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/Rad2Tshock/problem_initialize_state_data.H:#ifndef AMREX_USE_GPU
Exec/radiation_tests/Rad2Tshock/problem_initialize_rad_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/Rad2Tshock/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS+1> const& xnu,
Exec/radiation_tests/Rad2Tshock/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& nugroup,
Exec/radiation_tests/Rad2Tshock/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& dnugroup,
Exec/radiation_tests/Rad2Tshock/problem_initialize_rad_data.H:#ifndef AMREX_USE_GPU
Exec/radiation_tests/RadShestakovBolstad/problem_emissivity.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadShestakovBolstad/problem_emissivity.H:                         const GpuArray<Real, NGROUPS>& nu,
Exec/radiation_tests/RadShestakovBolstad/problem_emissivity.H:                         const GpuArray<Real, NGROUPS+1>& xnu,
Exec/radiation_tests/RadShestakovBolstad/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadShestakovBolstad/problem_initialize_rad_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadShestakovBolstad/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS+1> const& xnu,
Exec/radiation_tests/RadShestakovBolstad/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& nugroup,
Exec/radiation_tests/RadShestakovBolstad/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& dnugroup,
Exec/radiation_tests/RadSuOlsonMG/problem_emissivity.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSuOlsonMG/problem_emissivity.H:                         const GpuArray<Real, NGROUPS>& nu,
Exec/radiation_tests/RadSuOlsonMG/problem_emissivity.H:                         const GpuArray<Real, NGROUPS+1>& xnu,
Exec/radiation_tests/RadSuOlsonMG/problem_rad_source.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSuOlsonMG/problem_rad_source.H:    GpuArray<Real, 3> loc;
Exec/radiation_tests/RadSuOlsonMG/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSuOlsonMG/problem_initialize_rad_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSuOlsonMG/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS+1> const& xnu,
Exec/radiation_tests/RadSuOlsonMG/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& nugroup,
Exec/radiation_tests/RadSuOlsonMG/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& dnugroup,
Exec/radiation_tests/RadFront/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadFront/problem_initialize_rad_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadFront/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS+1> const& xnu,
Exec/radiation_tests/RadFront/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& nugroup,
Exec/radiation_tests/RadFront/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& dnugroup,
Exec/radiation_tests/RadThermalWave/Problem_Derive.cpp:  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/radiation_tests/RadThermalWave/Problem_Derive.cpp:  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/radiation_tests/RadThermalWave/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadThermalWave/problem_initialize_rad_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadThermalWave/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS+1> const& xnu,
Exec/radiation_tests/RadThermalWave/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& nugroup,
Exec/radiation_tests/RadThermalWave/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& dnugroup,
Exec/radiation_tests/RadSourceTest/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSourceTest/problem_initialize_rad_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSourceTest/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS+1> const& xnu,
Exec/radiation_tests/RadSourceTest/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& nugroup,
Exec/radiation_tests/RadSourceTest/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& dnugroup,
Exec/radiation_tests/RadBreakout/filt_prim.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadBreakout/filt_prim.H:#ifndef AMREX_USE_GPU
Exec/radiation_tests/RadBreakout/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadBreakout/problem_initialize_rad_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadBreakout/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS+1> const& xnu,
Exec/radiation_tests/RadBreakout/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& nugroup,
Exec/radiation_tests/RadBreakout/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& dnugroup,
Exec/radiation_tests/RadSphere/Problem_Derive.cpp:  GpuArray<Real, NGROUPS> nugroup = {0.0};
Exec/radiation_tests/RadSphere/Problem_Derive.cpp:  GpuArray<Real, NGROUPS> dnugroup = {0.0};
Exec/radiation_tests/RadSphere/Problem_Derive.cpp:  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/radiation_tests/RadSphere/problem_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSphere/problem_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSphere/problem_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSphere/problem_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSphere/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSphere/problem_initialize_rad_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSphere/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS+1> const& xnu,
Exec/radiation_tests/RadSphere/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& nugroup,
Exec/radiation_tests/RadSphere/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& dnugroup,
Exec/radiation_tests/RadBlastWave/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadBlastWave/problem_initialize_rad_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadBlastWave/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS+1> const& xnu,
Exec/radiation_tests/RadBlastWave/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& nugroup,
Exec/radiation_tests/RadBlastWave/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& dnugroup,
Exec/radiation_tests/RadSuOlson/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSuOlson/problem_initialize_rad_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/radiation_tests/RadSuOlson/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS+1> const& xnu,
Exec/radiation_tests/RadSuOlson/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& nugroup,
Exec/radiation_tests/RadSuOlson/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& dnugroup,
Exec/mhd_tests/RT/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/RT/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/OrszagTang/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/OrszagTang/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/species/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/species/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/LoopAdvection/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/LoopAdvection/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/LoopAdvection/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/Alfven/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/Alfven/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/MagnetosonicWaves/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/MagnetosonicWaves/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/FastRarefaction/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/FastRarefaction/problem_initialize_state_data.H:#ifndef AMREX_USE_GPU
Exec/mhd_tests/FastRarefaction/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/BrioWu/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/BrioWu/problem_initialize_state_data.H:#ifndef AMREX_USE_GPU
Exec/mhd_tests/BrioWu/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/DaiWoodward/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/mhd_tests/DaiWoodward/problem_initialize_state_data.H:#ifndef AMREX_USE_GPU
Exec/mhd_tests/DaiWoodward/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/Make.Castro:ifeq ($(USE_CUDA),TRUE)
Exec/Make.Castro:  CUDA_VERBOSE = FALSE
Exec/Make.Castro:ifeq ($(USE_GPU),TRUE)
Exec/Make.Castro:  # when using GPUs. Throw an error to prevent this case.
Exec/Make.Castro:    $(error OpenMP is not supported by Castro when building with GPU support)
Exec/Make.Castro:ifeq ($(USE_GPU_PRINTF),TRUE)
Exec/Make.Castro:  DEFINES += -DALLOW_GPU_PRINTF
Exec/gravity_tests/hse_convergence/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/hse_convergence_general/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/StarGrav/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/StarGrav/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/StarGrav/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/StarGrav/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/advecting_white_dwarf/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/advecting_white_dwarf/problem_initialize_state_data.H:    GpuArray<Real, 3> loc;
Exec/gravity_tests/evrard_collapse/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/uniform_cube/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/uniform_cube/Prob.cpp:        for (MFIter mfi(*phiGrav, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/gravity_tests/uniform_cube/Prob.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Exec/gravity_tests/hydrostatic_adjust/Problem_Derive.cpp:using RealVector = amrex::Gpu::ManagedVector<amrex::Real>;
Exec/gravity_tests/hydrostatic_adjust/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/gravity_tests/hydrostatic_adjust/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/gravity_tests/hydrostatic_adjust/problem_bc_fill.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/hydrostatic_adjust/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/hydrostatic_adjust/problem_source.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/hydrostatic_adjust/problem_source.H:#ifndef AMREX_USE_GPU
Exec/gravity_tests/hydrostatic_adjust/problem_source.H:#ifndef AMREX_USE_GPU
Exec/gravity_tests/uniform_sphere/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/uniform_sphere/Prob.cpp:        for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/gravity_tests/uniform_sphere/Prob.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/gravity_tests/uniform_sphere/Prob.cpp:        for (MFIter mfi(*phiGrav, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/gravity_tests/uniform_sphere/Prob.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Exec/gravity_tests/DustCollapse/problem_bc_fill.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/gravity_tests/DustCollapse/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/double_bubble/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/Vortices_LWAcoustics/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/RT/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/double_mach_reflection/problem_bc_fill.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/double_mach_reflection/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/Sedov/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/Sod_stellar/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/Sod_stellar/problem_initialize_state_data.H:#ifndef AMREX_USE_GPU
Exec/hydro_tests/KH/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/KH/problem_initialize_state_data.H:#ifndef AMREX_USE_GPU
Exec/hydro_tests/toy_convect/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/toy_convect/problem_tagging.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/toy_convect/problem_source.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/riemann_2d/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/oddeven/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/rotating_torus/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/rotating_torus/problem_initialize_state_data.H:    GpuArray<Real, 3> loc;
Exec/hydro_tests/rotating_torus/problem_initialize_state_data.H:    GpuArray<Real, 3> vel = {0.0};
Exec/hydro_tests/rotating_torus/problem_tagging.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/rotating_torus/problem_tagging.H:    GpuArray<Real, 3> loc;
Exec/hydro_tests/Sod/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/Sod/problem_initialize_state_data.H:#ifndef AMREX_USE_GPU
Exec/hydro_tests/gamma_law_bubble/Problem_Derive.cpp:using RealVector = amrex::Gpu::ManagedVector<amrex::Real>;
Exec/hydro_tests/gamma_law_bubble/Problem_Derive.cpp:  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/hydro_tests/gamma_law_bubble/Problem_Derive.cpp:  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/hydro_tests/gamma_law_bubble/Problem_Derive.cpp:  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/hydro_tests/gamma_law_bubble/Problem_Derive.cpp:  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/hydro_tests/gamma_law_bubble/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/gamma_law_bubble/prob_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/gamma_law_bubble/prob_util.H:                        const int npts, const GpuArray<Real,AMREX_SPACEDIM>& dx) {
Exec/hydro_tests/acoustic_pulse_general/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/Noh/problem_bc_fill.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/Noh/problem_bc_fill.H:    GpuArray<Real, 3> loc;
Exec/hydro_tests/Noh/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/Noh/problem_initialize_state_data.H:    GpuArray<Real, 3> loc;
Exec/hydro_tests/test_convect/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/test_convect/problem_source.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/gresho_vortex/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/acoustic_pulse/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/hydro_tests/acoustic_pulse/problem_initialize_state_data.H:#ifndef AMREX_USE_GPU
Exec/unit_tests/diffusion_test/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/diffusion_test/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/unit_tests/diffusion_test/prob_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/unit_tests/particles_test/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/reacting_tests/toy_flame/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/reacting_tests/toy_flame/Prob.cpp:    for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/reacting_tests/toy_flame/Prob.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Exec/reacting_tests/toy_flame/Prob.cpp:    for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/reacting_tests/toy_flame/Prob.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Exec/reacting_tests/bubble_convergence/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/reacting_tests/reacting_convergence/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/reacting_tests/reacting_bubble/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/reacting_tests/reacting_bubble/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/reacting_tests/reacting_bubble/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/reacting_tests/reacting_bubble/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/reacting_tests/reacting_bubble/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/reacting_tests/nse_test/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/flame/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/flame/Prob.cpp:    for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/science/flame/Prob.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Exec/science/flame/Prob.cpp:    for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Exec/science/flame/Prob.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Exec/science/wdmerger/wdmerger_data.H:    extern AMREX_GPU_MANAGED amrex::Real mdot_P;
Exec/science/wdmerger/wdmerger_data.H:    extern AMREX_GPU_MANAGED amrex::Real mdot_S;
Exec/science/wdmerger/wdmerger_data.H:    extern AMREX_GPU_MANAGED amrex::Real rad_P[7];
Exec/science/wdmerger/wdmerger_data.H:    extern AMREX_GPU_MANAGED amrex::Real rad_S[7];
Exec/science/wdmerger/wdmerger_data.H:    extern AMREX_GPU_MANAGED amrex::Real vol_P[7];
Exec/science/wdmerger/wdmerger_data.H:    extern AMREX_GPU_MANAGED amrex::Real vol_S[7];
Exec/science/wdmerger/wdmerger_data.H:    extern AMREX_GPU_MANAGED amrex::Real rho_avg_P;
Exec/science/wdmerger/wdmerger_data.H:    extern AMREX_GPU_MANAGED amrex::Real rho_avg_S;
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> vel{dat(i,j,k,1) / rho, dat(i,j,k,2) / rho, dat(i,j,k,3) / rho};
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> inertial_vel{vel};
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> vel{dat(i,j,k,1) / rho, dat(i,j,k,2) / rho, dat(i,j,k,3) / rho};
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> inertial_vel{vel};
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> vel{dat(i,j,k,1) / rho, dat(i,j,k,2) / rho, dat(i,j,k,3) / rho};
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> inertial_vel{vel};
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> vel{dat(i,j,k,1) / rho, dat(i,j,k,2) / rho, dat(i,j,k,3) / rho};
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> inertial_vel{vel};
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> angular_vel;
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> vel{dat(i,j,k,1) / rho, dat(i,j,k,2) / rho, dat(i,j,k,3) / rho};
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> inertial_vel{vel};
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> angular_vel;
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> vel{dat(i,j,k,1) / rho, dat(i,j,k,2) / rho, dat(i,j,k,3) / rho};
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> inertial_vel{vel};
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> angular_vel;
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> mom{dat(i,j,k,1), dat(i,j,k,2), dat(i,j,k,3)};
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> inertial_mom{mom};
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> mom{dat(i,j,k,1), dat(i,j,k,2), dat(i,j,k,3)};
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> inertial_mom{mom};
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> mom{dat(i,j,k,1), dat(i,j,k,2), dat(i,j,k,3)};
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> inertial_mom{mom};
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/wdmerger/tests/he_double_det/inputs_pakmor_simp_sdc:# GPU option
Exec/science/wdmerger/tests/he_double_det/inputs_pakmor_strang:# GPU option
Exec/science/wdmerger/wdmerger_data.cpp:AMREX_GPU_MANAGED Real wdmerger::mdot_P = 0.0;
Exec/science/wdmerger/wdmerger_data.cpp:AMREX_GPU_MANAGED Real wdmerger::mdot_S = 0.0;
Exec/science/wdmerger/wdmerger_data.cpp:AMREX_GPU_MANAGED Real wdmerger::rad_P[7] = { 0.0 };
Exec/science/wdmerger/wdmerger_data.cpp:AMREX_GPU_MANAGED Real wdmerger::rad_S[7] = { 0.0 };
Exec/science/wdmerger/wdmerger_data.cpp:AMREX_GPU_MANAGED Real wdmerger::vol_P[7] = { 0.0 };
Exec/science/wdmerger/wdmerger_data.cpp:AMREX_GPU_MANAGED Real wdmerger::vol_S[7] = { 0.0 };
Exec/science/wdmerger/wdmerger_data.cpp:AMREX_GPU_MANAGED Real wdmerger::rho_avg_P = 0.0;
Exec/science/wdmerger/wdmerger_data.cpp:AMREX_GPU_MANAGED Real wdmerger::rho_avg_S = 0.0;
Exec/science/wdmerger/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/wdmerger/problem_initialize_state_data.H:    GpuArray<Real, 3> loc;
Exec/science/wdmerger/problem_initialize_state_data.H:        GpuArray<Real, 3> rot_loc = {loc[0], loc[1], loc[2]};
Exec/science/wdmerger/problem_initialize_state_data.H:        GpuArray<Real, 3> vel;
Exec/science/wdmerger/wdmerger_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/wdmerger/wdmerger_util.H:    GpuArray<Real, 3> loc;
Exec/science/wdmerger/Prob.cpp:        GpuArray<bool, 3> symm_bound_lo{false};
Exec/science/wdmerger/Prob.cpp:        GpuArray<bool, 3> symm_bound_hi{false};
Exec/science/wdmerger/Prob.cpp:        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/wdmerger/Prob.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Exec/science/wdmerger/Prob.cpp:                GpuArray<Real, 3> r;
Exec/science/wdmerger/Prob.cpp:                GpuArray<Real, 3> rSymmetric{r[0], r[1], r[2]};
Exec/science/wdmerger/Prob.cpp:                GpuArray<Real, 3> momSymmetric{xmom(i,j,k) * maskFactor,
Exec/science/wdmerger/Prob.cpp:      for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/wdmerger/Prob.cpp:          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Exec/science/wdmerger/Prob.cpp:            for (MFIter mfi(*force[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/wdmerger/Prob.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k)
Exec/science/wdmerger/Prob.cpp:        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/wdmerger/Prob.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Exec/science/wdmerger/Prob.cpp:                GpuArray<Real, 3> dF;
Exec/science/wdmerger/Prob.cpp:    GpuArray<Real, 3> L1, L2, L3;
Exec/science/wdmerger/Prob.cpp:        for (MFIter mfi(phi_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/wdmerger/Prob.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Exec/science/wdmerger/Prob.cpp:                GpuArray<Real, 3> r;
Exec/science/wdmerger/Prob.cpp:        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/wdmerger/Prob.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Exec/science/wdmerger/Prob.cpp:                GpuArray<Real, 3> r;
Exec/science/wdmerger/problem_source.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/wdmerger/problem_source.H:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/problem_source.H:        GpuArray<Real, 3> mom;
Exec/science/wdmerger/problem_source.H:        GpuArray<Real, 3> hybrid_mom;
Exec/science/wdmerger/problem_source.H:        GpuArray<Real, 3> Sr;
Exec/science/wdmerger/problem_source.H:        GpuArray<Real, 3> acceleration = {0.0};
Exec/science/wdmerger/problem_source.H:            GpuArray<Real, 3> v = {0.0_rt};
Exec/science/wdmerger/problem_source.H:            GpuArray<Real, 3> v = {0.0_rt};
Exec/science/wdmerger/problem_source.H:        GpuArray<Real, 3> Sr = {0.0};
Exec/science/wdmerger/problem_source.H:        GpuArray<Real, 3> loc;
Exec/science/wdmerger/problem_source.H:        GpuArray<Real, 3> hybrid_Sr = {0.0};
Exec/science/wdmerger/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/wdmerger/problem_initialize_mhd_data.H:    GpuArray<Real, 3> loc;
Exec/science/wdmerger/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/wdmerger/problem_initialize_mhd_data.H:    GpuArray<Real, 3> loc;
Exec/science/wdmerger/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/wdmerger/scaling/frontier/frontier-128nodes.slurm:#SBATCH --gpus-per-task=1
Exec/science/wdmerger/scaling/frontier/frontier-128nodes.slurm:#SBATCH --gpu-bind=closest
Exec/science/wdmerger/scaling/frontier/frontier-128nodes.slurm:srun -n${TOTAL_NMPI} -N${SLURM_JOB_NUM_NODES} --ntasks-per-node=8 --gpus-per-task=1 ./$EXEC $INPUTS ${restartString}
Exec/science/wdmerger/scaling/frontier/frontier-16nodes.slurm:#SBATCH --gpus-per-task=1
Exec/science/wdmerger/scaling/frontier/frontier-16nodes.slurm:#SBATCH --gpu-bind=closest
Exec/science/wdmerger/scaling/frontier/frontier-16nodes.slurm:srun -n${TOTAL_NMPI} -N${SLURM_JOB_NUM_NODES} --ntasks-per-node=8 --gpus-per-task=1 ./$EXEC $INPUTS ${restartString}
Exec/science/wdmerger/scaling/frontier/frontier_256base_20240709.txt:# Run with ROCm 6.0 and:
Exec/science/wdmerger/scaling/frontier/frontier-64nodes.slurm:#SBATCH --gpus-per-task=1
Exec/science/wdmerger/scaling/frontier/frontier-64nodes.slurm:#SBATCH --gpu-bind=closest
Exec/science/wdmerger/scaling/frontier/frontier-64nodes.slurm:srun -n${TOTAL_NMPI} -N${SLURM_JOB_NUM_NODES} --ntasks-per-node=8 --gpus-per-task=1 ./$EXEC $INPUTS ${restartString}
Exec/science/wdmerger/scaling/frontier/frontier_1024base_20240709.txt:# Run with ROCm 6.0 and:
Exec/science/wdmerger/scaling/frontier/inputs_scaling:# GPU option
Exec/science/wdmerger/scaling/frontier/frontier_512base_20240709.txt:# Run with ROCm 6.0 and:
Exec/science/wdmerger/scaling/frontier/frontier-32nodes.slurm:#SBATCH --gpus-per-task=1
Exec/science/wdmerger/scaling/frontier/frontier-32nodes.slurm:#SBATCH --gpu-bind=closest
Exec/science/wdmerger/scaling/frontier/frontier-32nodes.slurm:srun -n${TOTAL_NMPI} -N${SLURM_JOB_NUM_NODES} --ntasks-per-node=8 --gpus-per-task=1 ./$EXEC $INPUTS ${restartString}
Exec/science/wdmerger/scaling/frontier/frontier-256nodes.slurm:#SBATCH --gpus-per-task=1
Exec/science/wdmerger/scaling/frontier/frontier-256nodes.slurm:#SBATCH --gpu-bind=closest
Exec/science/wdmerger/scaling/frontier/frontier-256nodes.slurm:srun -n${TOTAL_NMPI} -N${SLURM_JOB_NUM_NODES} --ntasks-per-node=8 --gpus-per-task=1 ./$EXEC $INPUTS ${restartString}
Exec/science/wdmerger/scaling/frontier/frontier-512nodes.slurm:#SBATCH --gpus-per-task=1
Exec/science/wdmerger/scaling/frontier/frontier-512nodes.slurm:#SBATCH --gpu-bind=closest
Exec/science/wdmerger/scaling/frontier/frontier-512nodes.slurm:srun -n${TOTAL_NMPI} -N${SLURM_JOB_NUM_NODES} --ntasks-per-node=8 --gpus-per-task=1 ./$EXEC $INPUTS ${restartString}
Exec/science/planet/problem_initialize_rad_state.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/planet/problem_initialize_rad_state.H:                                  GpuArray<Real, NGROUPS+1> const& xnu,
Exec/science/planet/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/planet/problem_source.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/massive_star/inputs_3d.nse:# GPU options
Exec/science/massive_star/analysis/andes-slice.submit:#SBATCH -p gpu
Exec/science/massive_star/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/massive_star/problem_initialize_state_data.H:        GpuArray<Real, 27> alpha{};
Exec/science/massive_star/problem_initialize_state_data.H:        GpuArray<Real, 27> beta{};
Exec/science/massive_star/problem_initialize_state_data.H:        GpuArray<Real, 27> gamma{};
Exec/science/massive_star/problem_initialize_state_data.H:        GpuArray<Real, 27> phix{};
Exec/science/massive_star/problem_initialize_state_data.H:        GpuArray<Real, 27> phiy{};
Exec/science/massive_star/problem_initialize_state_data.H:        GpuArray<Real, 27> phiz{};
Exec/science/massive_star/problem_initialize_state_data.H:        GpuArray<Real, 27> normk{};
Exec/science/massive_star/job_scripts/summit_gpu.submit:CASTRO=./Castro2d.gnu.MPI.CUDA.ex
Exec/science/massive_star/job_scripts/summit_gpu.submit:# number of nodes * 6 gpu per node
Exec/science/massive_star/job_scripts/summit_gpu.submit:n_gpu=1
Exec/science/massive_star/job_scripts/summit_gpu.submit:module load cuda/11.2.0
Exec/science/massive_star/job_scripts/summit_gpu.submit:jsrun -n $n_mpi -r $n_rs_per_node -c $n_cores -a 1 -g $n_gpu $CASTRO $INPUTS ${restartString}
Exec/science/circular_det/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/celldet/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/celldet/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/celldet/GNUmakefile:#GPU_COMPATIBLE_PROBLEM = TRUE
Exec/science/xrb_layered/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/xrb_layered/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/xrb_layered/problem_tagging.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/flame_wave/inputs_He/inputs.He.nonsquare.static.1000Hz.pslope.cool:# GPU options
Exec/science/flame_wave/Problem_Derive.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Exec/science/flame_wave/analysis/vis_3d/andes-slice.submit:#SBATCH -p gpu
Exec/science/flame_wave/analysis/vis_3d/andes.submit:#SBATCH -p gpu
Exec/science/flame_wave/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/flame_wave/inputs_H_He/inputs.H_He.nonsquare.static.1000Hz.pslope.cool:# GPU options
Exec/science/flame_wave/problem_tagging.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/flame_wave/problem_tagging.H:    GpuArray<Real, 3> loc;
Exec/science/flame_wave/job_scripts/summit/summit_gpu.submit:CASTRO=./Castro2d.pgi.MPI.CUDA.ex
Exec/science/flame_wave/job_scripts/summit/summit_gpu.submit:# number of nodes * 6 gpu per node
Exec/science/flame_wave/job_scripts/summit/summit_gpu.submit:n_gpu=1
Exec/science/flame_wave/job_scripts/summit/summit_gpu.submit:jsrun -n $n_mpi -r $n_rs_per_node -c $n_cores -a 1 -g $n_gpu $CASTRO $INPUTS ${restartString}
Exec/science/flame_wave/job_scripts/summit/summit_3d_512.submit:#BSUB -J flame_gpu
Exec/science/flame_wave/job_scripts/summit/summit_3d_512.submit:#BSUB -o flame_gpu.%J
Exec/science/flame_wave/job_scripts/summit/summit_3d_512.submit:#BSUB -e flame_gpu.%J
Exec/science/flame_wave/job_scripts/summit/summit_3d_512.submit:module load cuda/11.2.0
Exec/science/flame_wave/job_scripts/summit/summit_3d_512.submit:Castro_ex=./Castro3d.gnu.TPROF.MPI.CUDA.ex
Exec/science/flame_wave/job_scripts/summit/summit_3d_512.submit:n_mpi=3072 # nodes * 6 gpu per node
Exec/science/flame_wave/job_scripts/summit/summit_3d_512.submit:n_gpu=1
Exec/science/flame_wave/job_scripts/summit/summit_3d_512.submit:jsrun -n $n_mpi -r $n_rs_per_node -c $n_cores -a 1 -g $n_gpu $Castro_ex $inputs_file ${restartString}
Exec/science/flame_wave/job_scripts/summit/chainbsub.sh:  aout=`bsub -w "ended(${oldjob})" summit_gpu.submit`
Exec/science/flame_wave/inputs.1d:# GPU options
Exec/science/flame_wave/scaling/frontier/frontier-scaling-rkc-2024-07-04.txt:# nodes  rocm      mag_grid_size   avg time /   std dev
Exec/science/flame_wave/scaling/frontier/frontier-scaling-rkc-2024-07-04.txt:# ROCm to get around some compiler bugs, so that might explain some
Exec/science/flame_wave/scaling/frontier/frontier-scaling-2024-07-04.txt:# nodes  rocm      mag_grid_size   avg time /   std dev
Exec/science/flame_wave/scaling/frontier/frontier-scaling-2024-07-04.txt:# ROCm to get around some compiler bugs, so that might explain some
Exec/science/flame_wave/scaling/frontier/frontier-scaling-2024-07-04-subch_simple.txt:# nodes  rocm      mag_grid_size   avg time /   std dev
Exec/science/flame_wave/scaling/frontier/frontier-scaling-2024-07-04-subch_simple.txt:#  48      6.0          128      --- crashes due to not enough GPU memory ---
Exec/science/flame_wave/scaling/frontier/frontier-scaling-rkc-2023-05-31.txt:# ROCm 5.3.0
Exec/science/flame_wave/scaling/frontier/frontier-scaling-rkc-2023-05-31.txt:# nodes  rocm      mag_grid_size   avg time /   std dev
Exec/science/flame_wave/scaling/frontier/frontier-scaling-2024-08-21.txt:# nodes  rocm      mag_grid_size   avg time /   std dev
Exec/science/flame_wave/scaling/frontier/frontier-scaling-2024-08-21.txt:# ROCm to get around some compiler bugs, so that might explain some
Exec/science/flame_wave/scaling/frontier/frontier_scaling.py:            ls="None", marker="x", label="Frontier (ROCm 6.0)")
Exec/science/flame_wave/scaling/frontier/frontier_scaling.py:            ls="None", marker="x", label="Frontier (ROCm 6.0; RKC integrator)")
Exec/science/flame_wave/scaling/frontier/frontier_scaling.py:            ls="None", marker="^", label="Summit (CUDA 11.4)")
Exec/science/flame_wave/scaling/frontier/frontier_scaling.py:            ls="None", marker="o", label="Frontier (ROCm 6.0; big network)")
Exec/science/flame_wave/scaling/frontier/frontier-scaling-2023-04-06.txt:# ROCm 5.3.0
Exec/science/flame_wave/scaling/frontier/frontier-scaling-2023-04-06.txt:# nodes  rocm      mag_grid_size   avg time /   std dev
Exec/science/flame_wave/scaling/frontier/frontier-scaling-2023-04-06.txt:# nodes  rocm      mag_grid_size   avg time /   std dev
Exec/science/flame_wave/scaling/3d_science/summit_gpu_512.submit:#BSUB -J flame_gpu
Exec/science/flame_wave/scaling/3d_science/summit_gpu_512.submit:#BSUB -o flame_gpu.512.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_512.submit:#BSUB -e flame_gpu.512.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_512.submit:Castro_ex=./Castro3d.gnu.MPI.CUDA.ex
Exec/science/flame_wave/scaling/3d_science/summit_gpu_512.submit:n_mpi=3072 # ~ nodes * 6 gpu per node
Exec/science/flame_wave/scaling/3d_science/summit_gpu_512.submit:n_gpu=1
Exec/science/flame_wave/scaling/3d_science/summit_gpu_512.submit:module load cuda/11.4.0
Exec/science/flame_wave/scaling/3d_science/summit_gpu_512.submit:jsrun -n $n_mpi -c $n_cores -a 1 -g $n_gpu $Castro_ex $inputs_file max_step=25
Exec/science/flame_wave/scaling/3d_science/summit_gpu_342.submit:#BSUB -J flame_gpu
Exec/science/flame_wave/scaling/3d_science/summit_gpu_342.submit:#BSUB -o flame_gpu.342.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_342.submit:#BSUB -e flame_gpu.342.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_342.submit:Castro_ex=./Castro3d.gnu.MPI.CUDA.ex
Exec/science/flame_wave/scaling/3d_science/summit_gpu_342.submit:n_mpi=2048 # ~ nodes * 6 gpu per node
Exec/science/flame_wave/scaling/3d_science/summit_gpu_342.submit:n_gpu=1
Exec/science/flame_wave/scaling/3d_science/summit_gpu_342.submit:module load cuda/11.4.0
Exec/science/flame_wave/scaling/3d_science/summit_gpu_342.submit:jsrun -n $n_mpi -c $n_cores -a 1 -g $n_gpu $Castro_ex $inputs_file max_step=25
Exec/science/flame_wave/scaling/3d_science/summit_gpu_768.submit:#BSUB -J flame_gpu
Exec/science/flame_wave/scaling/3d_science/summit_gpu_768.submit:#BSUB -o flame_gpu.768.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_768.submit:#BSUB -e flame_gpu.768.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_768.submit:Castro_ex=./Castro3d.gnu.MPI.CUDA.ex
Exec/science/flame_wave/scaling/3d_science/summit_gpu_768.submit:n_mpi=4608 # ~ nodes * 6 gpu per node
Exec/science/flame_wave/scaling/3d_science/summit_gpu_768.submit:n_gpu=1
Exec/science/flame_wave/scaling/3d_science/summit_gpu_768.submit:module load cuda/11.4.0
Exec/science/flame_wave/scaling/3d_science/summit_gpu_768.submit:jsrun -n $n_mpi -c $n_cores -a 1 -g $n_gpu $Castro_ex $inputs_file max_step=25
Exec/science/flame_wave/scaling/3d_science/summit_gpu_683.submit:#BSUB -J flame_gpu
Exec/science/flame_wave/scaling/3d_science/summit_gpu_683.submit:#BSUB -o flame_gpu.683.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_683.submit:#BSUB -e flame_gpu.683.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_683.submit:Castro_ex=./Castro3d.gnu.MPI.CUDA.ex
Exec/science/flame_wave/scaling/3d_science/summit_gpu_683.submit:n_mpi=4096 # ~ nodes * 6 gpu per node
Exec/science/flame_wave/scaling/3d_science/summit_gpu_683.submit:n_gpu=1
Exec/science/flame_wave/scaling/3d_science/summit_gpu_683.submit:module load cuda/11.4.0
Exec/science/flame_wave/scaling/3d_science/summit_gpu_683.submit:jsrun -n $n_mpi -c $n_cores -a 1 -g $n_gpu $Castro_ex $inputs_file max_step=25
Exec/science/flame_wave/scaling/3d_science/summit_gpu_1366.submit:#BSUB -J flame_gpu
Exec/science/flame_wave/scaling/3d_science/summit_gpu_1366.submit:#BSUB -o flame_gpu.1366.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_1366.submit:#BSUB -e flame_gpu.1366.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_1366.submit:Castro_ex=./Castro3d.gnu.MPI.CUDA.ex
Exec/science/flame_wave/scaling/3d_science/summit_gpu_1366.submit:n_mpi=8192 # ~ nodes * 6 gpu per node
Exec/science/flame_wave/scaling/3d_science/summit_gpu_1366.submit:n_gpu=1
Exec/science/flame_wave/scaling/3d_science/summit_gpu_1366.submit:module load cuda/11.4.0
Exec/science/flame_wave/scaling/3d_science/summit_gpu_1366.submit:jsrun -n $n_mpi -c $n_cores -a 1 -g $n_gpu $Castro_ex $inputs_file max_step=25
Exec/science/flame_wave/scaling/3d_science/summit_gpu_256.submit:#BSUB -J flame_gpu
Exec/science/flame_wave/scaling/3d_science/summit_gpu_256.submit:#BSUB -o flame_gpu.256.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_256.submit:#BSUB -e flame_gpu.256.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_256.submit:Castro_ex=./Castro3d.gnu.MPI.CUDA.ex
Exec/science/flame_wave/scaling/3d_science/summit_gpu_256.submit:n_mpi=1536 # ~ nodes * 6 gpu per node
Exec/science/flame_wave/scaling/3d_science/summit_gpu_256.submit:n_gpu=1
Exec/science/flame_wave/scaling/3d_science/summit_gpu_256.submit:module load cuda/11.4.0
Exec/science/flame_wave/scaling/3d_science/summit_gpu_256.submit:jsrun -n $n_mpi -c $n_cores -a 1 -g $n_gpu $Castro_ex $inputs_file max_step=25
Exec/science/flame_wave/scaling/3d_science/summit_gpu_192.submit:#BSUB -J flame_gpu
Exec/science/flame_wave/scaling/3d_science/summit_gpu_192.submit:#BSUB -o flame_gpu.192.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_192.submit:#BSUB -e flame_gpu.192.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_192.submit:Castro_ex=./Castro3d.gnu.MPI.CUDA.ex
Exec/science/flame_wave/scaling/3d_science/summit_gpu_192.submit:n_mpi=1152 # ~ nodes * 6 gpu per node
Exec/science/flame_wave/scaling/3d_science/summit_gpu_192.submit:n_gpu=1
Exec/science/flame_wave/scaling/3d_science/summit_gpu_192.submit:module load cuda/11.4.0
Exec/science/flame_wave/scaling/3d_science/summit_gpu_192.submit:jsrun -n $n_mpi -c $n_cores -a 1 -g $n_gpu $Castro_ex $inputs_file max_step=25
Exec/science/flame_wave/scaling/3d_science/summit_gpu_171.submit:#BSUB -J flame_gpu
Exec/science/flame_wave/scaling/3d_science/summit_gpu_171.submit:#BSUB -o flame_gpu.171.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_171.submit:#BSUB -e flame_gpu.171.%J
Exec/science/flame_wave/scaling/3d_science/summit_gpu_171.submit:Castro_ex=./Castro3d.gnu.MPI.CUDA.ex
Exec/science/flame_wave/scaling/3d_science/summit_gpu_171.submit:n_mpi=1024 # ~ nodes * 6 gpu per node
Exec/science/flame_wave/scaling/3d_science/summit_gpu_171.submit:n_gpu=1
Exec/science/flame_wave/scaling/3d_science/summit_gpu_171.submit:module load cuda/11.4.0
Exec/science/flame_wave/scaling/3d_science/summit_gpu_171.submit:jsrun -n $n_mpi -c $n_cores -a 1 -g $n_gpu $Castro_ex $inputs_file max_step=25
Exec/science/subch_planar/Problem_Derive.cpp:using RealVector = amrex::Gpu::ManagedVector<amrex::Real>;
Exec/science/subch_planar/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/subch_planar/Problem_Derive.cpp:#ifndef AMREX_USE_GPU
Exec/science/subch_planar/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/subch_planar/Problem_Derive.cpp:#ifndef AMREX_USE_GPU
Exec/science/subch_planar/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/subch_planar/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/subch_planar/Problem_Derive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/science/subch_planar/Problem_Derive.cpp:#ifndef AMREX_USE_GPU
Exec/science/subch_planar/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/Detonation/problem_bc_fill.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/Detonation/problem_bc_fill.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/Detonation/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/Detonation/problem_source.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/xrb_mixed/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/xrb_mixed/problem_tagging.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/subchandra/analysis/GNUmakefile:USE_CUDA = FALSE
Exec/science/subchandra/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/nova/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/nova/GNUmakefile:USE_CUDA         = FALSE
Exec/science/convective_flame/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/convective_flame/problem_tagging.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/flame_tube/inputs_He/inputs.He.25cm.static.pslope:# GPU options
Exec/science/flame_tube/analysis/vis_3d/andes-slice.submit:#SBATCH -p gpu
Exec/science/flame_tube/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/flame_tube/problem_tagging.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/flame_tube/problem_tagging.H:    GpuArray<Real, 3> loc;
Exec/science/bwp-rad/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/bwp-rad/problem_initialize_rad_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Exec/science/bwp-rad/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS+1> const& xnu,
Exec/science/bwp-rad/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& nugroup,
Exec/science/bwp-rad/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& dnugroup,
Exec/Make.auto_source:ifeq ($(USE_CUDA), TRUE)
Exec/Make.auto_source:   CUDA_FLAGS := --CUDA_VERSION "$(nvcc_version)"
Exec/Make.auto_source:          $(CUDA_FLAGS) \
Exec/scf_tests/single_star/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
README.md:with MPI + OpenMP for CPUs and MPI + CUDA for NVIDIA GPUs and MPI + HIP for
README.md:AMD GPUs.
paper/paper.bib:	abstract = {We describe the AMReX suite of astrophysics codes and their application to modeling problems in stellar astrophysics. Maestro is tuned to efficiently model subsonic convective flows while Castro models the highly compressible flows associated with stellar explosions. Both are built on the block-structured adaptive mesh refinement library AMReX. Together, these codes enable a thorough investigation of stellar phenomena, including Type Ia supernovae and X-ray bursts. We describe these science applications and the approach we are taking to make these codes performant on current and future many-core and GPU-based architectures.}
paper/paper.md:  - name: NVIDIA Corporation
paper/paper.md:(exposing coarse-grained parallelism) or CUDA to spread the work across
paper/paper.md:GPU threads on GPU-based machines (fine-grained parallelism).  All of
paper/paper.md:the core physics can run on GPUs and has been shown to scale well to
paper/paper.md:thousands of GPUs [@castro_2019] and hundreds of thousands of CPU cores
paper/paper.md:for both CPUs and GPUs, and implement our parallel loops in an abstraction
paper/paper.md:GPU thread). This strategy is similar to the way the Kokkos [@Kokkos] and
paper/paper.md:CPUs and GPUs for all solvers -- achieves performance portability as a core
paper/paper.md:also thank NVIDIA Corporation for the donation of a Titan X Pascal and
paper/paper.md:Titan V used in this research.  The GPU development of Castro
paper/paper.md:benefited greatly from numerous GPU hackathons arranged by OLCF.
Source/gravity/Gravity.cpp:AMREX_GPU_MANAGED Real multipole::volumeFactor;
Source/gravity/Gravity.cpp:AMREX_GPU_MANAGED Real multipole::parityFactor;
Source/gravity/Gravity.cpp:AMREX_GPU_MANAGED Real multipole::rmax;
Source/gravity/Gravity.cpp:AMREX_GPU_MANAGED Array1D<bool, 0, 2> multipole::doSymmetricAddLo;
Source/gravity/Gravity.cpp:AMREX_GPU_MANAGED Array1D<bool, 0, 2> multipole::doSymmetricAddHi;
Source/gravity/Gravity.cpp:AMREX_GPU_MANAGED bool multipole::doSymmetricAdd;
Source/gravity/Gravity.cpp:AMREX_GPU_MANAGED Array1D<bool, 0, 2> multipole::doReflectionLo;
Source/gravity/Gravity.cpp:AMREX_GPU_MANAGED Array1D<bool, 0, 2> multipole::doReflectionHi;
Source/gravity/Gravity.cpp:AMREX_GPU_MANAGED Array2D<Real, 0, multipole::lnum_max, 0, multipole::lnum_max> multipole::factArray;
Source/gravity/Gravity.cpp:AMREX_GPU_MANAGED Array1D<Real, 0, multipole::lnum_max> multipole::parity_q0;
Source/gravity/Gravity.cpp:AMREX_GPU_MANAGED Array2D<Real, 0, multipole::lnum_max, 0, multipole::lnum_max> multipole::parity_qC_qS;
Source/gravity/Gravity.cpp:                        GpuArray<Real, AMREX_SPACEDIM> dx,
Source/gravity/Gravity.cpp:                        GpuArray<Real, AMREX_SPACEDIM> problo,
Source/gravity/Gravity.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/gravity/Gravity.cpp:    for (MFIter mfi(Rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Gravity.cpp:    for (MFIter mfi(Rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Gravity.cpp:    for (MFIter mfi(grav_vector, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Gravity.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/gravity/Gravity.cpp:            GpuArray<Real, 3> loc;
Source/gravity/Gravity.cpp:#ifndef AMREX_USE_GPU
Source/gravity/Gravity.cpp:    GpuArray<Real, 3> dx, problo;
Source/gravity/Gravity.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/gravity/Gravity.cpp:#ifndef AMREX_USE_GPU
Source/gravity/Gravity.cpp:                            Gpu::Atomic::Add(&radial_mass_ptr[index], vol_frac * u(i,j,k,URHO));
Source/gravity/Gravity.cpp:                            Gpu::Atomic::Add(&radial_vol_ptr[index], vol_frac);
Source/gravity/Gravity.cpp:                            Gpu::Atomic::Add(&radial_pres_ptr[index], vol_frac * eos_state.p);
Source/gravity/Gravity.cpp:            for (MFIter mfi(source, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Gravity.cpp:                amrex::ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), bx,
Source/gravity/Gravity.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::Gpu::Handler const& handler) noexcept
Source/gravity/Gravity.cpp:    if (!ParallelDescriptor::UseGpuAwareMpi()) {
Source/gravity/Gravity.cpp:    Gpu::synchronize();
Source/gravity/Gravity.cpp:    if (!ParallelDescriptor::UseGpuAwareMpi()) {
Source/gravity/Gravity.cpp:        if (!ParallelDescriptor::UseGpuAwareMpi()) {
Source/gravity/Gravity.cpp:        Gpu::synchronize();
Source/gravity/Gravity.cpp:        if (!ParallelDescriptor::UseGpuAwareMpi()) {
Source/gravity/Gravity.cpp:    for (MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Gravity.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/gravity/Gravity.cpp:    GpuArray<Real, 3> bc_dx;
Source/gravity/Gravity.cpp:    GpuArray<Real, 3> problo;
Source/gravity/Gravity.cpp:    GpuArray<Real, 3> probhi;
Source/gravity/Gravity.cpp:            priv_bcXYLo[tid]->setVal<RunOn::Gpu>(0.0);
Source/gravity/Gravity.cpp:            priv_bcXYHi[tid]->setVal<RunOn::Gpu>(0.0);
Source/gravity/Gravity.cpp:            priv_bcXZLo[tid]->setVal<RunOn::Gpu>(0.0);
Source/gravity/Gravity.cpp:            priv_bcXZHi[tid]->setVal<RunOn::Gpu>(0.0);
Source/gravity/Gravity.cpp:            priv_bcYZLo[tid]->setVal<RunOn::Gpu>(0.0);
Source/gravity/Gravity.cpp:            priv_bcYZHi[tid]->setVal<RunOn::Gpu>(0.0);
Source/gravity/Gravity.cpp:            for (MFIter mfi(source, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Gravity.cpp:                GpuArray<bool, 3> doSymmetricAddLo {false};
Source/gravity/Gravity.cpp:                GpuArray<bool, 3> doSymmetricAddHi {false};
Source/gravity/Gravity.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/gravity/Gravity.cpp:                    GpuArray<Real, 3> loc, locb;
Source/gravity/Gravity.cpp:                            Gpu::Atomic::Add(&bcXYLo_arr(l,m,0), dbc);
Source/gravity/Gravity.cpp:                            Gpu::Atomic::Add(&bcXYHi_arr(l,m,0), dbc);
Source/gravity/Gravity.cpp:                            Gpu::Atomic::Add(&bcXZLo_arr(l,0,n), dbc);
Source/gravity/Gravity.cpp:                            Gpu::Atomic::Add(&bcXZHi_arr(l,0,n), dbc);
Source/gravity/Gravity.cpp:                            Gpu::Atomic::Add(&bcYZLo_arr(0,m,n), dbc);
Source/gravity/Gravity.cpp:                            Gpu::Atomic::Add(&bcYZHi_arr(0,m,n), dbc);
Source/gravity/Gravity.cpp:    for (MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Gravity.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/gravity/Gravity.cpp:    for (MFIter mfi(Rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Gravity.cpp:    for (MFIter mfi(cc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Gravity.cpp:        for (MFIter mfi(*edges[idir], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Gravity.cpp:    for (MFIter mfi(grav_vector, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Gravity.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/gravity/Gravity.cpp:        [=] AMREX_GPU_DEVICE (int i) noexcept
Source/gravity/Gravity.cpp:            for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Gravity.cpp:        if (!ParallelDescriptor::UseGpuAwareMpi()) {
Source/gravity/Gravity.cpp:            Gpu::prefetchToHost(radial_mass[lev].begin(), radial_mass[lev].end());
Source/gravity/Gravity.cpp:            Gpu::prefetchToHost(radial_vol[lev].begin(), radial_vol[lev].end());
Source/gravity/Gravity.cpp:            Gpu::prefetchToHost(radial_pres[lev].begin(), radial_pres[lev].end());
Source/gravity/Gravity.cpp:        if (!ParallelDescriptor::UseGpuAwareMpi()) {
Source/gravity/Gravity.cpp:            Gpu::prefetchToDevice(radial_mass[lev].begin(), radial_mass[lev].end());
Source/gravity/Gravity.cpp:            Gpu::prefetchToDevice(radial_vol[lev].begin(), radial_vol[lev].end());
Source/gravity/Gravity.cpp:            Gpu::prefetchToDevice(radial_pres[lev].begin(), radial_pres[lev].end());
Source/gravity/Gravity.cpp:            [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
Source/gravity/Gravity.cpp:    [=] AMREX_GPU_DEVICE (int i) noexcept
Source/gravity/Gravity.cpp:            [=] AMREX_GPU_DEVICE (int i) noexcept
Source/gravity/Gravity.cpp:        [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
Source/gravity/Gravity.cpp:    [=] AMREX_GPU_DEVICE (int i) noexcept
Source/gravity/Gravity.cpp:            [=] AMREX_GPU_DEVICE (int i) noexcept
Source/gravity/Gravity.cpp:    [=] AMREX_GPU_DEVICE (int i) noexcept
Source/gravity/Gravity.cpp:    [=] AMREX_GPU_DEVICE (int i) noexcept
Source/gravity/Gravity.cpp:            [=] AMREX_GPU_DEVICE (int i) noexcept
Source/gravity/Gravity.cpp:    [=] AMREX_GPU_DEVICE (int i) noexcept
Source/gravity/Gravity.cpp:        [=] AMREX_GPU_DEVICE (int i) -> Real
Source/gravity/Gravity.cpp:        [=] AMREX_GPU_DEVICE (int i, Real const& mass_encl_local)
Source/gravity/Gravity_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/gravity/Gravity_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/gravity/Gravity_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/gravity/Gravity_util.H:AMREX_GPU_DEVICE AMREX_INLINE
Source/gravity/Gravity_util.H:                   amrex::Gpu::Handler const& handler,
Source/gravity/Gravity_util.H:            amrex::Gpu::deviceReduceSum(&qL0(l,0,n), dQL0, handler);
Source/gravity/Gravity_util.H:            amrex::Gpu::deviceReduceSum(&qU0(l,0,n), dQU0, handler);
Source/gravity/Gravity_util.H:                amrex::Gpu::deviceReduceSum(&qLC(l,m,n), dQLC, handler);
Source/gravity/Gravity_util.H:                amrex::Gpu::deviceReduceSum(&qLS(l,m,n), dQLS, handler);
Source/gravity/Gravity_util.H:                amrex::Gpu::deviceReduceSum(&qUC(l,m,n), dQUC, handler);
Source/gravity/Gravity_util.H:                amrex::Gpu::deviceReduceSum(&qUS(l,m,n), dQUS, handler);
Source/gravity/Gravity_util.H:AMREX_GPU_DEVICE AMREX_INLINE
Source/gravity/Gravity_util.H:                             const GpuArray<Real, AMREX_SPACEDIM>& problo,
Source/gravity/Gravity_util.H:                             const GpuArray<Real, AMREX_SPACEDIM>& probhi,
Source/gravity/Gravity_util.H:                             amrex::Gpu::Handler const& handler)
Source/gravity/Gravity_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/gravity/Gravity_util.H:Real direct_sum_symmetric_add(const GpuArray<Real, 3>& loc, const GpuArray<Real, 3>& locb,
Source/gravity/Gravity_util.H:                              const GpuArray<Real, 3>& problo, const GpuArray<Real, 3>& probhi,
Source/gravity/Gravity_util.H:                              const GpuArray<bool, 3>& doSymmetricAddLo, const GpuArray<bool, 3>& doSymmetricAddHi)
Source/gravity/Castro_gravity.cpp:    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Castro_gravity.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/gravity/Castro_gravity.cpp:            GpuArray<Real, NUM_STATE> snew;
Source/gravity/Castro_gravity.cpp:            GpuArray<Real, NSRC> src;
Source/gravity/Castro_gravity.cpp:            GpuArray<Real, 3> Sr;
Source/gravity/Castro_gravity.cpp:            GpuArray<Real, 3> loc;
Source/gravity/Castro_gravity.cpp:            GpuArray<Real, 3> hybrid_src;
Source/gravity/Castro_gravity.cpp:    GpuArray<Real, 3> dx;
Source/gravity/Castro_gravity.cpp:        for (MFIter mfi(state_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Castro_gravity.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/gravity/Castro_gravity.cpp:                GpuArray<Real, NSRC> src{};
Source/gravity/Castro_gravity.cpp:                GpuArray<Real, NUM_STATE> snew{};
Source/gravity/Castro_gravity.cpp:                GpuArray<Real, 3> vold;
Source/gravity/Castro_gravity.cpp:                GpuArray<Real, 3> Sr_old;
Source/gravity/Castro_gravity.cpp:                GpuArray<Real, 3> vnew;
Source/gravity/Castro_gravity.cpp:                GpuArray<Real, 3> Sr_new;
Source/gravity/Castro_gravity.cpp:                GpuArray<Real, 3> Srcorr;
Source/gravity/Castro_gravity.cpp:                GpuArray<Real, 3> loc;
Source/gravity/Castro_gravity.cpp:                GpuArray<Real, 3> hybrid_src;
Source/gravity/Castro_gravity.cpp:                    GpuArray<Real, 3> g;
Source/gravity/Castro_pointmass.cpp:        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/gravity/Castro_pointmass.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/gravity/Castro_pointmass.cpp:            for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/gravity/Castro_pointmass.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/gravity/binary.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/gravity/binary.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/gravity/binary.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/gravity/binary.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/gravity/binary.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/gravity/binary.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/gravity/binary.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/gravity/binary.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/gravity/binary.H:                          GpuArray<Real, 3>& L1, GpuArray<Real, 3>& L2, GpuArray<Real, 3>& L3)
Source/gravity/binary.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/gravity/Gravity.H:// This vector can be accessed on the GPU.
Source/gravity/Gravity.H:using RealVector = amrex::Gpu::ManagedVector<amrex::Real>;
Source/gravity/Gravity.H:    extern AMREX_GPU_MANAGED amrex::Real volumeFactor;
Source/gravity/Gravity.H:    extern AMREX_GPU_MANAGED amrex::Real parityFactor;
Source/gravity/Gravity.H:    extern AMREX_GPU_MANAGED amrex::Real rmax;
Source/gravity/Gravity.H:    extern AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> doSymmetricAddLo;
Source/gravity/Gravity.H:    extern AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> doSymmetricAddHi;
Source/gravity/Gravity.H:    extern AMREX_GPU_MANAGED bool doSymmetricAdd;
Source/gravity/Gravity.H:    extern AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> doReflectionLo;
Source/gravity/Gravity.H:    extern AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> doReflectionHi;
Source/gravity/Gravity.H:    extern AMREX_GPU_MANAGED amrex::Array2D<amrex::Real, 0, lnum_max, 0, lnum_max> factArray;
Source/gravity/Gravity.H:    extern AMREX_GPU_MANAGED amrex::Array1D<amrex::Real, 0, lnum_max> parity_q0;
Source/gravity/Gravity.H:    extern AMREX_GPU_MANAGED amrex::Array2D<amrex::Real, 0, lnum_max, 0, lnum_max> parity_qC_qS;
Source/gravity/Gravity.H:                      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
Source/gravity/Gravity.H:                      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> problo,
Source/sources/Castro_sources.cpp:    for (MFIter mfi(Sborder, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/sources/Castro_sources.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/sources/Castro_sponge.cpp:    for (MFIter mfi(state_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/sources/Castro_sponge.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/sources/Castro_sponge.cpp:    GpuArray<Real, 3> r;
Source/sources/Castro_sponge.cpp:    GpuArray<Real, 3> Sr;
Source/sources/Castro_sponge.cpp:    GpuArray<Real, 3> sponge_target_velocity = {sponge_target_x_velocity,
Source/sources/Castro_sponge.cpp:    GpuArray<Real, 3> Sr_hybrid;
Source/sources/Castro_geom.cpp:  for (MFIter mfi(geom_src, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/sources/Castro_geom.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/sources/Castro_geom.cpp:  for (MFIter mfi(geom_src, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/sources/Castro_geom.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/sources/Castro_thermo.cpp:  for (MFIter mfi(thermo_src, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/sources/Castro_thermo.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/hlld.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/mhd_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/mhd/mhd_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/mhd_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/electric.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/electric.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/electric.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/Castro_mhd.cpp:      for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/mhd/Castro_mhd.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/Castro_mhd.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/Castro_mhd.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/Castro_mhd.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/mhd/Castro_mhd.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/mhd/Castro_mhd.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/mhd/Castro_mhd.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/Castro_mhd.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/Castro_mhd.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/Castro_mhd.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/mhd_plm.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/mhd_ppm.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/mhd_util.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/mhd/mhd_util.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/mhd/mhd_util.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/mhd/mhd_util.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/mhd/mhd_util.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/mhd/mhd_util.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/mhd/mhd_eigen.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/mhd/mhd_eigen.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/mhd/mhd_eigen.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/mhd/mhd_eigen.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/mhd/mhd_eigen.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/mhd/mhd_eigen.H:#ifndef AMREX_USE_GPU
Source/mhd/mhd_eigen.H:#ifndef AMREX_USE_GPU
Source/mhd/ct_upwind.cpp:  GpuArray<Real, 3> dx;
Source/mhd/ct_upwind.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/ct_upwind.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/ct_upwind.cpp:  GpuArray<Real, 3> dx;
Source/mhd/ct_upwind.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/mhd/ct_upwind.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/diffusion/diffusion_util.cpp:#include <AMReX_Gpu.H>
Source/diffusion/diffusion_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/diffusion/diffusion_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/diffusion/Diffusion.cpp:    for (MFIter mfi(cc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/diffusion/Diffusion.cpp:    for (MFIter mfi(cc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/diffusion/Castro_diffusion.cpp:           for (MFIter mfi(grown_state, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/problems/Castro_bc_fill_nd.cpp:    GpuBndryFuncFab<CastroGenericFill> gpu_bndry_func(CastroGenericFill{});
Source/problems/Castro_bc_fill_nd.cpp:    gpu_bndry_func(bx, data, dcomp, numcomp, geom, time, bcr_noinflow, bcomp, scomp);
Source/problems/Castro_bc_fill_nd.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/problems/problem_emissivity.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/problems/problem_emissivity.H:                         const GpuArray<Real, NGROUPS>& nu,
Source/problems/problem_emissivity.H:                         const GpuArray<Real, NGROUPS+1>& xnu,
Source/problems/problem_rad_source.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/problems/ambient.cpp:AMREX_GPU_MANAGED amrex::Real ambient::ambient_state[NUM_STATE];
Source/problems/Problem_Derive.cpp:    // need to explicitly synchronize after GPU kernels.
Source/problems/Castro_problem_source.cpp:    for (MFIter mfi(ext_src, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/problems/Castro_problem_source.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/problems/problem_bc_fill.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/problems/problem_initialize_state_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/problems/problem_tagging.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/problems/ambient_fill.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/problems/ambient_fill.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/problems/problem_source.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/problems/problem_initialize_rad_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/problems/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS+1> const& xnu,
Source/problems/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& nugroup,
Source/problems/problem_initialize_rad_data.H:                                  GpuArray<Real, NGROUPS> const& dnugroup,
Source/problems/hse_fill.cpp:            // single thread on the GPU
Source/problems/hse_fill.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/problems/hse_fill.cpp:#ifndef AMREX_USE_GPU
Source/problems/hse_fill.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/problems/hse_fill.cpp:#ifndef AMREX_USE_GPU
Source/problems/hse_fill.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/problems/hse_fill.cpp:#ifndef AMREX_USE_GPU
Source/problems/hse_fill.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/problems/hse_fill.cpp:#ifndef AMREX_USE_GPU
Source/problems/hse_fill.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/problems/hse_fill.cpp:#ifndef AMREX_USE_GPU
Source/problems/hse_fill.cpp:#ifndef AMREX_USE_GPU
Source/problems/problem_initialize_mhd_data.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/problems/ambient.H:    extern AMREX_GPU_MANAGED amrex::Real ambient_state[NUM_STATE];
Source/sdc/sdc_newton_solve.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/sdc/sdc_newton_solve.H:    GpuArray<Real, NUM_STATE> R_full;
Source/sdc/sdc_newton_solve.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/sdc/sdc_newton_solve.H:                 GpuArray<Real, NUM_STATE> const& U_old,
Source/sdc/sdc_newton_solve.H:                 GpuArray<Real, NUM_STATE> & U_new,
Source/sdc/sdc_newton_solve.H:                 GpuArray<Real, NUM_STATE> const& C,
Source/sdc/sdc_newton_solve.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/sdc/sdc_newton_solve.H:                     GpuArray<Real, NUM_STATE> const& U_old,
Source/sdc/sdc_newton_solve.H:                     GpuArray<Real, NUM_STATE>& U_new,
Source/sdc/sdc_newton_solve.H:                     GpuArray<Real, NUM_STATE> const& C,
Source/sdc/sdc_newton_solve.H:    GpuArray<Real, NUM_STATE> U_begin;
Source/sdc/Make.package:ifneq ($(USE_GPU), TRUE)
Source/sdc/vode_rhs_true_sdc.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/sdc/vode_rhs_true_sdc.H:               GpuArray<Real, NUM_STATE> const& U_old,
Source/sdc/vode_rhs_true_sdc.H:               GpuArray<Real, NUM_STATE>& U_new,
Source/sdc/vode_rhs_true_sdc.H:               GpuArray<Real, NUM_STATE> const& C,
Source/sdc/Castro_sdc.H:#ifndef AMREX_USE_GPU
Source/sdc/Castro_sdc.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/sdc/Castro_sdc.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/sdc/Castro_sdc.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/sdc/Castro_sdc.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/sdc/Castro_sdc.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/sdc/Castro_sdc.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/sdc/Castro_sdc.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/sdc/Castro_sdc.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/sdc/sdc_react_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/sdc/sdc_react_util.H:                         GpuArray<Real, NUM_STATE>& R) {
Source/sdc/sdc_react_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/sdc/sdc_react_util.H:    GpuArray<Real, NUM_STATE> R_arr;
Source/sdc/sdc_react_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/sdc/sdc_react_util.H:#ifndef AMREX_USE_GPU
Source/sdc/sdc_util.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/sdc/sdc_util.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/sdc/Castro_sdc_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/sdc/Castro_sdc_util.H:    GpuArray<Real, NumSpec> xn;
Source/sdc/Castro_sdc_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/sdc/Castro_sdc_util.H:          GpuArray<Real, NUM_STATE> const& U_old,
Source/sdc/Castro_sdc_util.H:          GpuArray<Real, NUM_STATE>& U_new,
Source/sdc/Castro_sdc_util.H:          GpuArray<Real, NUM_STATE> const& C,
Source/sdc/Castro_sdc_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/sdc/Castro_sdc_util.H:    GpuArray<Real, NUM_STATE> U_old_zone;
Source/sdc/Castro_sdc_util.H:    GpuArray<Real, NUM_STATE> U_new_zone;
Source/sdc/Castro_sdc_util.H:    GpuArray<Real, NUM_STATE> C_zone;
Source/sdc/Castro_sdc_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/sdc/Castro_sdc_util.H:    GpuArray<Real, NUM_STATE> U_old;
Source/sdc/Castro_sdc_util.H:    GpuArray<Real, NUM_STATE> U_new;
Source/sdc/Castro_sdc_util.H:    GpuArray<Real, NUM_STATE> R_full;
Source/sdc/Castro_sdc_util.H:    GpuArray<Real, NUM_STATE> C_zone;
Source/sdc/Castro_sdc_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/sdc/Castro_sdc_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/rotation/Castro_rotation.cpp:    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/rotation/Castro_rotation.cpp:        for (MFIter mfi(state_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/rotation/Rotation.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/rotation/Rotation.cpp:    GpuArray<Real, 3> r;
Source/rotation/rotation_sources.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/rotation/rotation_sources.cpp:    GpuArray<Real, 3> loc;
Source/rotation/rotation_sources.cpp:    GpuArray<Real, 3> v;
Source/rotation/rotation_sources.cpp:    GpuArray<Real, 3> linear_momentum;
Source/rotation/rotation_sources.cpp:    GpuArray<Real, 3> hybrid_source;
Source/rotation/rotation_sources.cpp:#ifndef AMREX_USE_GPU
Source/rotation/rotation_sources.cpp:  GpuArray<Real, 3> dx;
Source/rotation/rotation_sources.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/rotation/rotation_sources.cpp:    GpuArray<Real, 3> loc;
Source/rotation/rotation_sources.cpp:    GpuArray<Real, 3> vold;
Source/rotation/rotation_sources.cpp:    GpuArray<Real, 3> vnew;
Source/rotation/rotation_sources.cpp:    GpuArray<Real, 3> hybrid_source;
Source/rotation/rotation_sources.cpp:    GpuArray<Real, 3> linear_momentum;
Source/rotation/rotation_sources.cpp:              GpuArray<Real, 3> temp_vel{};
Source/rotation/rotation_sources.cpp:#ifndef AMREX_USE_GPU
Source/rotation/Rotation.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/rotation/Rotation.H:GpuArray<Real, 3> get_omega()
Source/rotation/Rotation.H:    GpuArray<Real, 3> omega = {0.0_rt};
Source/rotation/Rotation.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/rotation/Rotation.H:rotational_acceleration(GpuArray<Real, 3>& r, GpuArray<Real, 3>& v,
Source/rotation/Rotation.H:  GpuArray<Real, 3> omega_cross_v;
Source/rotation/Rotation.H:      GpuArray<Real, 3> omega_cross_r;
Source/rotation/Rotation.H:      GpuArray<Real, 3> omega_cross_omega_cross_r;
Source/rotation/Rotation.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/rotation/Rotation.H:rotational_potential(GpuArray<Real, 3>& r) {
Source/rotation/Rotation.H:      GpuArray<Real, 3> omega_cross_r;
Source/rotation/Rotation.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/rotation/Rotation.H:                                 const Real time, GpuArray<Real, 3>& v) {
Source/rotation/Rotation.H:  GpuArray<Real, 3> loc;
Source/rotation/Rotation.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/rotation/Rotation.H:                                 const Real time, GpuArray<Real, 3>& v) {
Source/rotation/Rotation.H:  GpuArray<Real, 3> loc;
Source/rotation/Rotation.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/rotation/Rotation.H:GpuArray<Real, 3> inertial_rotation(const GpuArray<Real, 3>& vec, Real time)
Source/rotation/Rotation.H:    GpuArray<Real, 3> vec_i{};
Source/hydro/edge_util.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/edge_util.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/ppm.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/ppm.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/ppm.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/ppm.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/ppm.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/riemann_type.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/riemann_type.H:#ifndef AMREX_USE_GPU
Source/hydro/riemann_type.H:#ifndef AMREX_USE_GPU
Source/hydro/Castro_hydro.cpp:    for (MFIter mfi(u, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/hydro/Castro_hydro.cpp:#ifndef AMREX_USE_GPU
Source/hydro/Castro_hydro.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Source/hydro/Castro_hydro.cpp:#ifndef AMREX_USE_GPU
Source/hydro/Castro_hydro.cpp:#ifndef AMREX_USE_GPU
Source/hydro/Castro_hybrid.cpp:    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/hydro/Castro_hybrid.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_hybrid.cpp:            GpuArray<Real, 3> loc;
Source/hydro/Castro_hybrid.cpp:    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/hydro/Castro_hybrid.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_hybrid.cpp:            GpuArray<Real, 3> loc;
Source/hydro/Castro_hybrid.cpp:            GpuArray<Real, 3> linear_mom;
Source/hydro/Castro_hybrid.cpp:            GpuArray<Real, 3> hybrid_mom;
Source/hydro/Castro_hybrid.cpp:    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/hydro/Castro_hybrid.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_hybrid.cpp:            GpuArray<Real, 3> loc;
Source/hydro/Castro_hybrid.cpp:            GpuArray<Real, 3> hybrid_mom;
Source/hydro/Castro_hybrid.cpp:            GpuArray<Real, 3> linear_mom;
Source/hydro/Castro_mol_hydro.cpp:  GpuArray<int, 3> domain_lo = geom.Domain().loVect3d();
Source/hydro/Castro_mol_hydro.cpp:  GpuArray<int, 3> domain_hi = geom.Domain().hiVect3d();
Source/hydro/Castro_mol_hydro.cpp:    // the asynchronous case, usually on GPUs).
Source/hydro/Castro_mol_hydro.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_mol_hydro.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_mol_hydro.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_mol_hydro.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_mol_hydro.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_mol_hydro.cpp:#ifndef AMREX_USE_GPU
Source/hydro/Castro_mol_hydro.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/Castro_mol_hydro.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/Castro_mol_hydro.cpp:            GpuArray<bool, AMREX_SPACEDIM> lo_periodic;
Source/hydro/Castro_mol_hydro.cpp:            GpuArray<bool, AMREX_SPACEDIM> hi_periodic;
Source/hydro/Castro_mol_hydro.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/Castro_mol_hydro.cpp:              [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/Castro_mol_hydro.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/Castro_mol_hydro.cpp:              [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/Castro_mol_hydro.cpp:#endif   // AMREX_USE_GPU
Source/hydro/Castro_mol_hydro.cpp:              [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_mol_hydro.cpp:#ifndef AMREX_USE_GPU
Source/hydro/Castro_mol_hydro.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_mol_hydro.cpp:          GpuArray<Real, 3> loc;
Source/hydro/Castro_mol_hydro.cpp:              [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/riemann_2shock_solvers.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/riemann_2shock_solvers.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/riemann_2shock_solvers.H:                    bool& converged, amrex::GpuArray<amrex::Real, riemann_constants::PSTAR_BISECT_FACTOR * riemann_constants::HISTORY_SIZE>& pstar_hist_extra) {
Source/hydro/riemann_2shock_solvers.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/hydro/riemann_2shock_solvers.H:    #ifndef AMREX_USE_GPU
Source/hydro/riemann_2shock_solvers.H:      amrex::GpuArray<amrex::Real, riemann_constants::HISTORY_SIZE> pstar_hist;
Source/hydro/riemann_2shock_solvers.H:    #ifndef AMREX_USE_GPU
Source/hydro/riemann_2shock_solvers.H:    #ifndef AMREX_USE_GPU
Source/hydro/riemann_2shock_solvers.H:              // we don't store the history if we are in CUDA, so
Source/hydro/riemann_2shock_solvers.H:    #ifndef AMREX_USE_GPU
Source/hydro/riemann_2shock_solvers.H:              amrex::GpuArray<amrex::Real, riemann_constants::PSTAR_BISECT_FACTOR * riemann_constants::HISTORY_SIZE> pstar_hist_extra;
Source/hydro/riemann_2shock_solvers.H:    #ifndef AMREX_USE_GPU
Source/hydro/riemann_2shock_solvers.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/hydro/trace_ppm.cpp:#ifndef AMREX_USE_GPU
Source/hydro/trace_ppm.cpp:#ifndef AMREX_USE_GPU
Source/hydro/trace_ppm.cpp:  GpuArray<int, NQSRC> do_source_trace;
Source/hydro/trace_ppm.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/trace_ppm.cpp:#ifndef AMREX_USE_GPU
Source/hydro/trace_ppm.cpp:#ifndef AMREX_USE_GPU
Source/hydro/trace_ppm.cpp:#ifndef AMREX_USE_GPU
Source/hydro/trace_ppm.cpp:#ifndef AMREX_USE_GPU
Source/hydro/trace_ppm.cpp:#ifndef AMREX_USE_GPU
Source/hydro/trace_ppm.cpp:#ifndef AMREX_USE_GPU
Source/hydro/trace_plm.cpp:#ifndef AMREX_USE_GPU
Source/hydro/trace_plm.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/trans.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/trans.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_center_average.cpp:                         GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {
Source/hydro/fourth_center_average.cpp:  GpuArray<bool, AMREX_SPACEDIM> lo_periodic;
Source/hydro/fourth_center_average.cpp:  GpuArray<bool, AMREX_SPACEDIM> hi_periodic;
Source/hydro/fourth_center_average.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/fourth_center_average.cpp:                                  GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {
Source/hydro/fourth_center_average.cpp:  GpuArray<bool, AMREX_SPACEDIM> lo_periodic;
Source/hydro/fourth_center_average.cpp:  GpuArray<bool, AMREX_SPACEDIM> hi_periodic;
Source/hydro/fourth_center_average.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_center_average.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_center_average.cpp:                         GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {
Source/hydro/fourth_center_average.cpp:  GpuArray<bool, AMREX_SPACEDIM> lo_periodic;
Source/hydro/fourth_center_average.cpp:  GpuArray<bool, AMREX_SPACEDIM> hi_periodic;
Source/hydro/fourth_center_average.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_center_average.cpp:                            GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {
Source/hydro/fourth_center_average.cpp:  GpuArray<bool, AMREX_SPACEDIM> lo_periodic;
Source/hydro/fourth_center_average.cpp:  GpuArray<bool, AMREX_SPACEDIM> hi_periodic;
Source/hydro/fourth_center_average.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/fourth_center_average.cpp:                             GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {
Source/hydro/fourth_center_average.cpp:                               GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {
Source/hydro/fourth_center_average.cpp:  GpuArray<bool, AMREX_SPACEDIM> lo_periodic;
Source/hydro/fourth_center_average.cpp:  GpuArray<bool, AMREX_SPACEDIM> hi_periodic;
Source/hydro/fourth_center_average.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_center_average.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_center_average.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/fourth_center_average.H:                  GpuArray<bool, AMREX_SPACEDIM> const& lo_periodic, GpuArray<bool, AMREX_SPACEDIM> const& hi_periodic,
Source/hydro/fourth_center_average.H:                  GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {
Source/hydro/fourth_center_average.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/fourth_center_average.H:                GpuArray<bool, AMREX_SPACEDIM> const& lo_periodic, GpuArray<bool, AMREX_SPACEDIM> const& hi_periodic,
Source/hydro/fourth_center_average.H:                GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {
Source/hydro/Castro_ctu.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/Castro_ctu.cpp:#ifndef AMREX_USE_GPU
Source/hydro/Castro_ctu.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_order.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_order.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_order.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_order.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_order.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_order.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_order.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_order.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_order.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_order.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/fourth_order.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_hydro.H:                          GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);
Source/hydro/Castro_hydro.H:                                   GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);
Source/hydro/Castro_hydro.H:                          GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);
Source/hydro/Castro_hydro.H:                             GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);
Source/hydro/Castro_hydro.H:                              GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);
Source/hydro/Castro_hydro.H:                                GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);
Source/hydro/advection_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/hydro/advection_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/hydro/advection_util.H:#ifndef AMREX_USE_GPU
Source/hydro/hybrid.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/hybrid.H:void linear_to_hybrid(const GpuArray<Real, 3>& loc,
Source/hydro/hybrid.H:                      const GpuArray<Real, 3>& linear_mom,
Source/hydro/hybrid.H:                      GpuArray<Real, 3>& hybrid_mom)
Source/hydro/hybrid.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/hybrid.H:void hybrid_to_linear(const GpuArray<Real, 3>& loc,
Source/hydro/hybrid.H:                      const GpuArray<Real, 3>& hybrid_mom,
Source/hydro/hybrid.H:                      GpuArray<Real, 3>& linear_mom)
Source/hydro/hybrid.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/hybrid.H:void set_hybrid_momentum_source(const GpuArray<Real, 3>& loc,
Source/hydro/hybrid.H:                                const GpuArray<Real, 3>& linear_source,
Source/hydro/hybrid.H:                                GpuArray<Real, 3>& hybrid_source)
Source/hydro/hybrid.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/hybrid.H:void compute_hybrid_flux(const GpuArray<Real, NGDNV>& state, const GeometryData& geomdata,
Source/hydro/hybrid.H:                         GpuArray<Real, NUM_STATE>& flux, bool cell_centered = false)
Source/hydro/hybrid.H:    GpuArray<Real, 3> loc;
Source/hydro/hybrid.H:    GpuArray<Real, 3> linear_mom;
Source/hydro/hybrid.H:    GpuArray<Real, 3> hybrid_mom;
Source/hydro/Castro_ctu_rad.cpp:  GpuArray<Real, NGROUPS> Erscale = {0.0};
Source/hydro/Castro_ctu_rad.cpp:  GpuArray<Real, NGROUPS> dlognu = {0.0};
Source/hydro/Castro_ctu_rad.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k, int g) noexcept
Source/hydro/Castro_ctu_rad.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_ctu_rad.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/hydro/reconstruction.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/reconstruction.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/reconstruction.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/reconstruction.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/reconstruction.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/riemann_solvers.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/hydro/riemann_solvers.H:    GpuArray<Real, NGDNV> qgdnv_zone;
Source/hydro/riemann_solvers.H:    GpuArray<Real, NUM_STATE> F_zone;
Source/hydro/riemann_solvers.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/hydro/riemann_solvers.H:              GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {
Source/hydro/riemann_solvers.H:#ifndef AMREX_USE_GPU
Source/hydro/advection_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/advection_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/advection_util.cpp:#ifndef AMREX_USE_GPU
Source/hydro/advection_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/advection_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/advection_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/advection_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/advection_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/advection_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k, int g) noexcept
Source/hydro/advection_util.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/advection_util.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/advection_util.cpp:#ifndef AMREX_USE_GPU
Source/hydro/advection_util.cpp:      GpuArray<Real, 3> loc;
Source/hydro/advection_util.cpp:      GpuArray<Real, 3> linear_mom;
Source/hydro/advection_util.cpp:      GpuArray<Real, 3> hybrid_mom;
Source/hydro/advection_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/advection_util.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_mol.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/Castro_mol.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_mol.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/Castro_mol.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/Castro_mol.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/Castro_mol.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_mol.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_mol.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_mol.cpp:    GpuArray<Real, NGDNV> qgdnv_zone;
Source/hydro/Castro_mol.cpp:    GpuArray<Real, NUM_STATE> F_zone;
Source/hydro/Castro_mol.cpp:  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_ctu_hydro.cpp:#ifdef AMREX_USE_GPU
Source/hydro/Castro_ctu_hydro.cpp:  // Our strategy for launching work on GPUs in the hydro is incompatible with OpenMP,
Source/hydro/Castro_ctu_hydro.cpp:#if defined(AMREX_USE_OMP) && defined(AMREX_USE_GPU)
Source/hydro/Castro_ctu_hydro.cpp:  amrex::Error("USE_OMP=TRUE and USE_GPU=TRUE are not concurrently supported in Castro");
Source/hydro/Castro_ctu_hydro.cpp:#ifdef AMREX_USE_GPU
Source/hydro/Castro_ctu_hydro.cpp:    // the asynchronous case, usually on GPUs).
Source/hydro/Castro_ctu_hydro.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_ctu_hydro.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_ctu_hydro.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_ctu_hydro.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_ctu_hydro.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_ctu_hydro.cpp:          GpuArray<Real, 3> loc;
Source/hydro/Castro_ctu_hydro.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_ctu_hydro.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/Castro_ctu_hydro.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/hydro/Castro_ctu_hydro.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_ctu_hydro.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/Castro_ctu_hydro.cpp:#ifdef AMREX_USE_GPU
Source/hydro/Castro_ctu_hydro.cpp:                  Gpu::synchronize();
Source/hydro/Castro_ctu_hydro.cpp:                  Gpu::synchronize();
Source/hydro/slope.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/slope.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/riemann.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/hydro/riemann.cpp:#ifndef AMREX_USE_GPU
Source/hydro/HLL_solvers.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/HLL_solvers.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/HLL_solvers.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/hydro/HLL_solvers.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/hydro/HLL_solvers.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/hydro/HLL_solvers.H:         GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {
Source/hydro/flatten.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/driver/Castro.H:#ifndef AMREX_USE_GPU
Source/driver/Castro.H:/// Returns true if we're on the GPU and the total
Source/driver/Castro.H:/// memory on this level oversubscribes GPU memory.
Source/driver/Castro.H:#ifdef AMREX_USE_GPU
Source/driver/Castro.H:                static_cast<Long>(amrex::Gpu::Device::totalGlobalMem()));
Source/driver/Castro.H:/// for keeping track of the amount of CPU or GPU time used -- this will persist
Source/driver/Castro_generic_fill.H:#ifdef AMREX_USE_GPU
Source/driver/Castro_generic_fill.H:    AMREX_GPU_DEVICE
Source/driver/Castro_generic_fill.cpp:    GpuBndryFuncFab<CastroGenericFill> gpu_bndry_func(CastroGenericFill{});
Source/driver/Castro_generic_fill.cpp:    gpu_bndry_func(bx, data, dcomp, numcomp, geom, time, bcr, bcomp, scomp);
Source/driver/sum_utils.cpp:    for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/sum_utils.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Source/driver/sum_utils.cpp:    for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/sum_utils.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Source/driver/sum_utils.cpp:    for (MFIter mfi(mf1, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/sum_utils.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Source/driver/sum_utils.cpp:    for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/sum_utils.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Source/driver/sum_utils.cpp:    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/driver/sum_utils.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Source/driver/sum_utils.cpp:            GpuArray<Real, 3> r;
Source/driver/sum_utils.cpp:            GpuArray<Real, 3> pos{r};
Source/driver/sum_utils.cpp:            GpuArray<Real, 3> vel;
Source/driver/sum_utils.cpp:            GpuArray<Real, 3> inertial_vel{vel};
Source/driver/sum_utils.cpp:            GpuArray<Real, 3> g;
Source/driver/sum_utils.cpp:            GpuArray<Real, 3> inertial_g{g};
Source/driver/parse_castro_params.py:    pf.write("#include <AMReX_Gpu.H>\n")
Source/driver/Castro_util.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/driver/Castro_util.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/driver/Castro_util.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/driver/Castro_util.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/driver/Castro_util.H:              GeometryData const& geomdata, GpuArray<Real, 3>& loc,
Source/driver/Castro_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/driver/Castro_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/driver/Make.package:ifneq ($(USE_GPU), TRUE)
Source/driver/main.cpp:        if (!pp.contains("abort_on_out_of_gpu_memory")) {
Source/driver/main.cpp:            // Abort if we run out of GPU memory.
Source/driver/main.cpp:            pp.add("abort_on_out_of_gpu_memory", true);
Source/driver/Derive.cpp:    // need to explicitly synchronize after GPU kernels.
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Derive.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:#ifndef AMREX_USE_GPU
Source/driver/Castro.cpp:#ifndef AMREX_USE_GPU
Source/driver/Castro.cpp:#ifndef AMREX_USE_GPU
Source/driver/Castro.cpp:    // SDC does not support GPUs yet
Source/driver/Castro.cpp:#ifdef AMREX_USE_GPU
Source/driver/Castro.cpp:        amrex::Error("SDC is currently not enabled on GPUs.");
Source/driver/Castro.cpp:#ifndef AMREX_USE_GPU
Source/driver/Castro.cpp:    // activity and GPU page faults that we're uninterested in.
Source/driver/Castro.cpp:    Gpu::Device::profilerStop();
Source/driver/Castro.cpp:       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/driver/Castro.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/Castro.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
Source/driver/Castro.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/Castro.cpp:           [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/driver/Castro.cpp:       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/driver/Castro.cpp:         [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:#ifndef AMREX_USE_GPU
Source/driver/Castro.cpp:#ifndef AMREX_USE_GPU
Source/driver/Castro.cpp:             [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:      for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/driver/Castro.cpp:          GpuArray<Real, NGROUPS+1> xnu_pass = {0.0};
Source/driver/Castro.cpp:          GpuArray<Real, NGROUPS> nugroup_pass = {0.0};
Source/driver/Castro.cpp:          GpuArray<Real, NGROUPS> dnugroup_pass = {0.0};
Source/driver/Castro.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:    Gpu::Device::profilerStart();
Source/driver/Castro.cpp:        for (MFIter mfi(crse_state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/driver/Castro.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/Castro.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/driver/Castro.cpp:#ifndef AMREX_USE_GPU
Source/driver/Castro.cpp:#elif defined(ALLOW_GPU_PRINTF)
Source/driver/Castro.cpp:    for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/Castro.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/driver/Castro.cpp:    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/driver/Castro.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/driver/Castro.cpp:#ifndef AMREX_USE_GPU
Source/driver/Castro.cpp:#ifndef AMREX_USE_GPU
Source/driver/Castro.cpp:#elif defined(ALLOW_GPU_PRINTF)
Source/driver/Castro.cpp:#ifdef ALLOW_GPU_PRINTF
Source/driver/Castro.cpp:    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/driver/Castro.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:        for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/Castro.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:        for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/driver/Castro.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:    for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/driver/Castro.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:    for (MFIter mfi(State, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/Castro.cpp:  for (MFIter mfi(State, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/Castro.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:  for (MFIter mfi(State, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/Castro.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/driver/Castro.cpp:  for (MFIter mfi(State, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/Castro.cpp:      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
Source/driver/Castro.cpp:#ifdef AMREX_USE_GPU
Source/driver/Castro.cpp:        // if we're running on a GPU. This helps us alleviate
Source/driver/Castro.cpp:        // pressure on the GPU memory, at the slight cost of
Source/driver/Castro.cpp:#ifdef AMREX_USE_GPU
Source/driver/Castro.cpp:#ifdef AMREX_USE_GPU
Source/driver/Castro_setup.cpp:  stateBndryFunc.setRunOnGPU(true);
Source/driver/Castro_setup.cpp:  genericBndryFunc.setRunOnGPU(true);
Source/driver/timestep.cpp:  [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> GpuTuple<ValLocPair<Real, IntVect>>
Source/driver/timestep.cpp:  [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> GpuTuple<ValLocPair<Real, IntVect>>
Source/driver/timestep.cpp:  [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> GpuTuple<ValLocPair<Real, IntVect>>
Source/driver/timestep.cpp:    [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> GpuTuple<ValLocPair<Real, IntVect>>
Source/driver/timestep.cpp:    for (MFIter mfi(stateMF, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/timestep.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Source/driver/timestep.cpp:        Gpu::synchronize();
Source/driver/MGutils.cpp:             GpuArray<Real, AMREX_SPACEDIM> dx,
Source/driver/MGutils.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/MGutils.cpp:             GpuArray<Real, AMREX_SPACEDIM> dx,
Source/driver/MGutils.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/MGutils.cpp:               GpuArray<Real, AMREX_SPACEDIM> dx,
Source/driver/MGutils.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/MGutils.cpp:                  GpuArray<Real, AMREX_SPACEDIM> dx,
Source/driver/MGutils.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/MGutils.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro_advance_sdc.cpp:#ifndef AMREX_USE_GPU
Source/driver/sum_integrated_quantities.cpp:        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/driver/sum_integrated_quantities.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/driver/sum_integrated_quantities.cpp:        // Calculate GPU memory consumption.
Source/driver/sum_integrated_quantities.cpp:#ifdef AMREX_USE_GPU
Source/driver/sum_integrated_quantities.cpp:        Long gpu_size_free_MB = Gpu::Device::freeMemAvailable() / (1024 * 1024);
Source/driver/sum_integrated_quantities.cpp:        ParallelDescriptor::ReduceLongMin(gpu_size_free_MB, ParallelDescriptor::IOProcessorNumber());
Source/driver/sum_integrated_quantities.cpp:        Long gpu_size_used_MB = (Gpu::Device::totalGlobalMem() - Gpu::Device::freeMemAvailable()) / (1024 * 1024);
Source/driver/sum_integrated_quantities.cpp:        ParallelDescriptor::ReduceLongMax(gpu_size_used_MB, ParallelDescriptor::IOProcessorNumber());
Source/driver/sum_integrated_quantities.cpp:#ifdef AMREX_USE_GPU
Source/driver/sum_integrated_quantities.cpp:                header << std::setw(datwidth) << "  MAXIMUM GPU MEMORY USED"; ++n;
Source/driver/sum_integrated_quantities.cpp:                header << std::setw(datwidth) << "  MINIMUM GPU MEMORY FREE"; ++n;
Source/driver/sum_integrated_quantities.cpp:#ifdef AMREX_USE_GPU
Source/driver/sum_integrated_quantities.cpp:            log << std::setw(datwidth)                                    << gpu_size_used_MB;
Source/driver/sum_integrated_quantities.cpp:            log << std::setw(datwidth)                                    << gpu_size_free_MB;
Source/driver/_cpp_parameters:# In GPU builds, the hydro advance typically results in a large amount of extra
Source/driver/_cpp_parameters:# efficiency. If you want to constrain the code's GPU memory footprint at the expense
Source/driver/Castro_advance.cpp:#ifndef AMREX_USE_GPU
Source/driver/Castro_advance.cpp:#endif // AMREX_USE_GPU
Source/driver/Castro_advance.cpp:            for (MFIter mfi(S_old, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/Castro_advance.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/MGutils.H:             GpuArray<Real, AMREX_SPACEDIM> dx,
Source/driver/MGutils.H:             GpuArray<Real, AMREX_SPACEDIM> dx,
Source/driver/MGutils.H:               GpuArray<Real, AMREX_SPACEDIM> dx,
Source/driver/MGutils.H:                  GpuArray<Real, AMREX_SPACEDIM> dx,
Source/driver/Castro_io.cpp:       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/driver/Castro_io.cpp:               [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/driver/Castro_io.cpp:#ifdef AMREX_USE_GPU
Source/driver/Castro_io.cpp:  jobInfoFile << "GPU time used since start of simulation (GPU-hours): " <<
Source/driver/Castro_io.cpp:#ifdef AMREX_USE_GPU
Source/driver/Castro_io.cpp:  // same type of GPU.
Source/driver/Castro_io.cpp:  jobInfoFile << "GPU Information:       " << "\n";
Source/driver/Castro_io.cpp:  jobInfoFile << "GPU model name: " << Gpu::Device::deviceName() << "\n";
Source/driver/Castro_io.cpp:  jobInfoFile << "Number of GPUs used: " << Gpu::Device::numDevicesUsed() << "\n";
Source/driver/Castro_io.cpp:#ifdef AMREX_USE_CUDA
Source/driver/Castro_io.cpp:  jobInfoFile << "CUDA version:  " << buildInfoGetCUDAVersion() << "\n";
Source/driver/math.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/driver/math.H:cross_product(amrex::GpuArray<amrex::Real, 3> const& a,
Source/driver/math.H:              amrex::GpuArray<amrex::Real, 3> const& b,
Source/driver/math.H:              amrex::GpuArray<amrex::Real, 3>& c) {
Source/driver/math.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/rad_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/rad_util.H:        GpuArray<Real, 3> loc;
Source/radiation/rad_util.H:       GpuArray<Real, 3> loc;
Source/radiation/rad_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/rad_util.H:        GpuArray<Real, 3> loc;
Source/radiation/rad_util.H:        GpuArray<Real, 3> loc;
Source/radiation/rad_util.H:            GpuArray<Real, 3> loc;
Source/radiation/rad_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/rad_util.H:            GpuArray<Real, 3> loc;
Source/radiation/rad_util.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/radiation/rad_util.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/radiation/rad_util.H:#ifndef AMREX_USE_GPU
Source/radiation/rad_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/rad_util.H:#ifndef AMREX_USE_GPU
Source/radiation/RadSolve.cpp:    for (MFIter mfi(cc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/radiation/RadSolve.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadSolve.cpp:  for (MFIter mfi(fkp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/RadSolve.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadSolve.cpp:          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadSolve.cpp:    for (MFIter mfi(lambda[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/RadSolve.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadSolve.cpp:        for (MFIter mfi(dcoefs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/RadSolve.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadSolve.cpp:  for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/RadSolve.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadSolve.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadSolve.cpp:      for (MFIter mfi(Flux[n], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/RadSolve.cpp:          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadSolve.cpp:          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadSolve.cpp:          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/radiation/RadSolve.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadSolve.cpp:  for (MFIter mfi(lambda, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/RadSolve.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadSolve.cpp:  for (MFIter mfi(kpp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/RadSolve.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadSolve.cpp:  for (MFIter ri(rhs, TilingIfNotGPU()); ri.isValid(); ++ri) {
Source/radiation/RadSolve.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/SGRadSolver.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreABec.H:               amrex::Array4<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM+1>> const& mat,
Source/radiation/HypreABec.H:               amrex::Array4<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM+1>> const& mat,
Source/radiation/HypreABec.H:              amrex::Array4<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM+1>> const& mat,
Source/radiation/HypreABec.H:               amrex::Array4<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM+1>> const& mat,
Source/radiation/RadDerive.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadHydro.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/RadHydro.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/RadHydro.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/radiation/RadHydro.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/blackbody.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/blackbody.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/blackbody.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/blackbody.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/blackbody.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/blackbody.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/trace_ppm_rad.cpp:#ifndef AMREX_USE_GPU
Source/radiation/trace_ppm_rad.cpp:  GpuArray<int, NQSRC> do_source_trace;
Source/radiation/trace_ppm_rad.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreABec.cpp:  Gpu::synchronize();
Source/radiation/HypreABec.cpp:                        Array4<GpuArray<Real, AMREX_SPACEDIM+1>> const& mat,
Source/radiation/HypreABec.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreABec.cpp:    Gpu::synchronize();
Source/radiation/HypreABec.cpp:                        Array4<GpuArray<Real, AMREX_SPACEDIM+1>> const& mat,
Source/radiation/HypreABec.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreABec.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreABec.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreABec.cpp:    Gpu::synchronize();
Source/radiation/HypreABec.cpp:                       Array4<GpuArray<Real, AMREX_SPACEDIM+1>> const& mat,
Source/radiation/HypreABec.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreABec.cpp:    Gpu::synchronize();
Source/radiation/HypreABec.cpp:                        Array4<GpuArray<Real, AMREX_SPACEDIM+1>> const& mat,
Source/radiation/HypreABec.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreABec.cpp:    Gpu::synchronize();
Source/radiation/HypreABec.cpp:  BaseFab<GpuArray<Real, size>> matfab; // AoS indexing
Source/radiation/HypreABec.cpp:    Gpu::synchronize();
Source/radiation/HypreABec.cpp:  Gpu::synchronize();
Source/radiation/HypreABec.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreABec.cpp:    Gpu::synchronize();
Source/radiation/HypreABec.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreABec.cpp:    Gpu::synchronize();
Source/radiation/HypreABec.cpp:      Gpu::streamSynchronize();
Source/radiation/HypreABec.cpp:    Gpu::streamSynchronize();
Source/radiation/HypreABec.cpp:    Gpu::streamSynchronize();
Source/radiation/HypreABec.cpp:    Gpu::streamSynchronize();
Source/radiation/HypreABec.cpp:  Gpu::synchronize();
Source/radiation/HypreABec.cpp:  Gpu::synchronize();
Source/radiation/HypreABec.cpp:    Gpu::synchronize();
Source/radiation/HypreABec.cpp:  Gpu::synchronize();
Source/radiation/HypreABec.cpp:  Gpu::synchronize();
Source/radiation/filt_prim.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/MGFLD.cpp:  for (MFIter mfi(Er_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/MGFLD.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Source/radiation/MGFLD.cpp:  for (MFIter mfi(rhoe_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/MGFLD.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Source/radiation/MGFLD.cpp:    for (MFIter mfi(kpp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/MGFLD.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:    for (MFIter mfi(rho, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/MGFLD.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:  for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/MGFLD.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:  for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/MGFLD.cpp:      GpuArray<Real, NGROUPS> nugroup_loc;
Source/radiation/MGFLD.cpp:      GpuArray<Real, NGROUPS+1> xnu_loc;
Source/radiation/MGFLD.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:  for (MFIter mfi(spec, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/MGFLD.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:  for (MFIter mfi(acoefs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/MGFLD.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:      for (MFIter mfi(spec, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/MGFLD.cpp:          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:              [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:  for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/MGFLD.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:  for (MFIter mfi(spec, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/MGFLD.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:    for (MFIter mfi(Er_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/MGFLD.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Source/radiation/MGFLD.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:            Gpu::synchronize();
Source/radiation/MGFLD.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:            Gpu::synchronize();
Source/radiation/MGFLD.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:        Gpu::synchronize();
Source/radiation/MGFLD.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:        Gpu::synchronize();
Source/radiation/MGFLD.cpp:  GpuArray<Real, NGROUPS> nugroup_loc;
Source/radiation/MGFLD.cpp:  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:  Gpu::synchronize();
Source/radiation/MGFLD.cpp:    GpuArray<Real, NGROUPS> nugroup_loc;
Source/radiation/MGFLD.cpp:    for (MFIter mfi(kappa_r, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/MGFLD.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:    Gpu::synchronize();
Source/radiation/MGFLD.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/MGFLD.cpp:    for (MFIter ri(rhs, TilingIfNotGPU()); ri.isValid(); ++ri)
Source/radiation/MGFLD.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:    for (MFIter mfi(exch, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/Radiation.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:        for (MFIter mfi(eta, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/Radiation.cpp:                [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:  for (MFIter mfi(eta, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/Radiation.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Source/radiation/Radiation.cpp:  for (MFIter mfi(eta,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/Radiation.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Source/radiation/Radiation.cpp:          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
Source/radiation/Radiation.cpp:    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/Radiation.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:    Gpu::synchronize();
Source/radiation/Radiation.cpp:    for (MFIter si(state,TilingIfNotGPU()); si.isValid(); ++si) {
Source/radiation/Radiation.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/Radiation.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:      for(MFIter mfi(kappa_r, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/radiation/Radiation.cpp:          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:  for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/Radiation.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:  for (MFIter mfi(kappa_r, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/Radiation.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:  Gpu::synchronize();
Source/radiation/Radiation.cpp:      for (MFIter mfi(R[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/Radiation.cpp:              [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:        for (MFIter mfi(lambda[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/Radiation.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:            Gpu::synchronize();
Source/radiation/Radiation.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:    for (MFIter mfi(dcf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/radiation/Radiation.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/Radiation.cpp:      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/fluxlimiter.H:AMREX_GPU_HOST_DEVICE inline
Source/radiation/fluxlimiter.H:#ifndef AMREX_USE_GPU
Source/radiation/fluxlimiter.H:#ifndef AMREX_USE_GPU
Source/radiation/fluxlimiter.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/fluxlimiter.H:#ifndef AMREX_USE_GPU
Source/radiation/HypreMultiABec.cpp:                           Array4<GpuArray<Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
Source/radiation/HypreMultiABec.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreMultiABec.cpp:    Gpu::synchronize();
Source/radiation/HypreMultiABec.cpp:                           Array4<GpuArray<Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
Source/radiation/HypreMultiABec.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreMultiABec.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreMultiABec.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreMultiABec.cpp:    Gpu::synchronize();
Source/radiation/HypreMultiABec.cpp:                       Array4<GpuArray<Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
Source/radiation/HypreMultiABec.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreMultiABec.cpp:    Gpu::synchronize();
Source/radiation/HypreMultiABec.cpp:                        Array4<GpuArray<Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
Source/radiation/HypreMultiABec.cpp:    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreMultiABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreMultiABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreMultiABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreMultiABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreMultiABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreMultiABec.cpp:#ifndef AMREX_USE_GPU
Source/radiation/HypreMultiABec.cpp:    Gpu::synchronize();
Source/radiation/HypreMultiABec.cpp:  BaseFab<GpuArray<Real, size>> matfab; // AoS indexing
Source/radiation/HypreExtMultiABec.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreExtMultiABec.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreExtMultiABec.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreExtMultiABec.cpp:            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/HypreExtMultiABec.cpp:      Gpu::streamSynchronize();
Source/radiation/HypreExtMultiABec.cpp:    Gpu::streamSynchronize();
Source/radiation/HypreMultiABec.H:             amrex::Array4<amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
Source/radiation/HypreMultiABec.H:             amrex::Array4<amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
Source/radiation/HypreMultiABec.H:              amrex::Array4<amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
Source/radiation/HypreMultiABec.H:               amrex::Array4<amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/filter.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/radiation/RadPlotvar.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadPlotvar.cpp:    GpuArray<Real, NGROUPS> dlognu = {0.0};
Source/radiation/RadPlotvar.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/radiation/RadPlotvar.cpp:    GpuArray<Real, NGROUPS> dlognu = {0.0};
Source/radiation/RadPlotvar.cpp:        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Source/reactions/Castro_react_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/reactions/Castro_react_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/reactions/Castro_react_util.H:okay_to_burn(GpuArray<Real, NUM_STATE> const& state) {
Source/reactions/Castro_react_util.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/reactions/Castro_react.cpp:#if defined(AMREX_USE_GPU)
Source/reactions/Castro_react.cpp:    Gpu::Buffer<int> d_num_failed({0});
Source/reactions/Castro_react.cpp:    for (MFIter mfi(s, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/reactions/Castro_react.cpp:#if defined(AMREX_USE_GPU)
Source/reactions/Castro_react.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/reactions/Castro_react.cpp:#if defined(AMREX_USE_GPU)
Source/reactions/Castro_react.cpp:                Gpu::Atomic::Add(p_num_failed, burn_failed);
Source/reactions/Castro_react.cpp:        Gpu::streamSynchronize(); // otherwise HIP may fail to allocate the necessary resources.
Source/reactions/Castro_react.cpp:#ifdef ALLOW_GPU_PRINTF
Source/reactions/Castro_react.cpp:#if defined(AMREX_USE_GPU)
Source/reactions/Castro_react.cpp:#if defined(AMREX_USE_GPU)
Source/reactions/Castro_react.cpp:    Gpu::Buffer<int> d_num_failed({0});
Source/reactions/Castro_react.cpp:    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/reactions/Castro_react.cpp:#if defined(AMREX_USE_GPU)
Source/reactions/Castro_react.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/reactions/Castro_react.cpp:#if defined(AMREX_USE_GPU)
Source/reactions/Castro_react.cpp:                Gpu::Atomic::Add(p_num_failed, burn_failed);
Source/reactions/Castro_react.cpp:        Gpu::streamSynchronize(); // otherwise HIP may fail to allocate the necessary resources.
Source/reactions/Castro_react.cpp:#ifdef ALLOW_GPU_PRINTF
Source/reactions/Castro_react.cpp:#if defined(AMREX_USE_GPU)
Source/reactions/sdc_cons_to_burn.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/reactions/sdc_cons_to_burn.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/reactions/sdc_cons_to_burn.H:copy_cons_to_burn_type(GpuArray<Real, NUM_STATE> const& state,
Source/scf/scf_relax.cpp:    GpuArray<Real, 3> scf_r_A = {problem::center[0], problem::center[1], problem::center[2]};
Source/scf/scf_relax.cpp:    GpuArray<Real, 3> scf_r_B = {problem::center[0], problem::center[1], problem::center[2]};
Source/scf/scf_relax.cpp:        for (MFIter mfi(state_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/scf/scf_relax.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/scf/scf_relax.cpp:            for (MFIter mfi((*psi[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/scf/scf_relax.cpp:            for (MFIter mfi((*phi[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/scf/scf_relax.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/scf/scf_relax.cpp:            for (MFIter mfi((*phi[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/scf/scf_relax.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/scf/scf_relax.cpp:                    GpuArray<Real, 3> r = {0.0};
Source/scf/scf_relax.cpp:            for (MFIter mfi((*phi[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/scf/scf_relax.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/scf/scf_relax.cpp:                    GpuArray<Real, 3> r = {0.0};
Source/scf/scf_relax.cpp:            for (MFIter mfi((*state_vec[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/scf/scf_relax.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/scf/scf_relax.cpp:            for (MFIter mfi((*state_vec[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/scf/scf_relax.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/scf/scf_relax.cpp:                    GpuArray<Real, 3> r = {0.0};

```
