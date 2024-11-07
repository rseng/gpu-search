# https://github.com/BenWibking/quokka

```console
regression/quokka-tests.ini:[Sedov-GPU]
regression/quokka-tests.ini:cmakeSetupOpts = -DAMReX_GPU_BACKEND=CUDA
regression/quokka-tests.ini:[ShockCloud-GPU]
regression/quokka-tests.ini:cmakeSetupOpts = -DAMReX_GPU_BACKEND=CUDA
regression/quokka-tests.ini:[RandomBlast-GPU]
regression/quokka-tests.ini:cmakeSetupOpts = -DAMReX_GPU_BACKEND=CUDA
docs/docs/about.md:Quokka is a high-resolution shock capturing AMR radiation hydrodynamics code using the AMReX library [@AMReX_JOSS] to provide patch-based adaptive mesh functionality. We take advantage of the C++ loop abstractions in AMReX in order to run with high performance on either CPUs, NVIDIA GPUs, or AMD GPUs.
docs/docs/about.md:The code is written in modern C++17, using MPI for distributed-memory parallelism, with the AMReX GPU abstraction compiling as either native CUDA code or native HIP code when GPU support is enabled.
docs/docs/running_on_hpc_clusters.md:Use the `openmpi/4.1.4` module (or newer), and build with `gcc/system` or `gcc/11.1.0`, and use `cuda/11.7.0` (or newer).
docs/docs/insitu_analysis.md:      [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
docs/docs/insitu_analysis.md:2.  Make sure there are entries listed for `hdf5`, `cuda`, and `openmpi` in your `~/.spack/packages.yaml` file.
docs/docs/insitu_analysis.md:4.  Run `spack fetch --dependencies ascent@develop+cuda+vtkh~fortran~shared cuda_arch=70 ^conduit~parmetis~fortran`
docs/docs/insitu_analysis.md:5.  On a dedicated compute node, run `spack install ascent@develop+cuda+vtkh~fortran~shared cuda_arch=70 ^conduit~parmetis~fortran`
docs/docs/insitu_analysis.md:For A100 GPUs, change the above lines to ``cuda_arch=80``. Currently, it's not possible to [build for both GPU models at the same time](https://github.com/Alpine-DAV/ascent/issues/950#issuecomment-1153243232).
docs/docs/debugging.md:    This is when the code accesses an array with an index that corresponds to an element that doesn't actually exist. In C++, this causes the computer to access a memory location that is completely unrelated to the array that you intended to access. In a CPU code, this usually causes a silently incorrect result, but on GPU, this may actually cause the simulation to crash. However, if you are accessing an `amrex::Array4` object *and you have compiled in Debug mode*, then AMReX will issue an error message when this occurs. There is a significant performance cost to this error checking, so it does not occur when compiled in Release mode.
docs/docs/debugging.md:-   accessing a host variable from the GPU
docs/docs/debugging.md:    The second most common type of bug encountered in Quokka is accessing a host variable (i.e., a variable that can only be accessed from code that runs on the CPU) from code running on the GPU (i.e., within a `ParallelFor`). Sometimes the compiler will detect this situation and print an error message, but often this will only present an issue when actuallly running the code -- for instance, this can happen when the GPU code tries to dereference a pointer to an address in CPU memory. In that case, the only way to debug this error is to run Quokka under `cuda-gdb` (or, on AMD GPUs, `rocgdb`).
docs/docs/debugging.md:## How to debug on GPUs
docs/docs/debugging.md:The best way to debug on GPUs is to\... not debug on GPUs. That is, it is always easier to instead debug the problem on a CPU-only run. GPU debugging is very painful and itself quite buggy. This is unfortunately true for all GPU vendors.
docs/docs/debugging.md:-   Build Quokka without GPU support but with `-DCMAKE_BUILD_TYPE=Debug` and re-run. If there are any array out-of-bounds errors, it will stop and report exactly which array is being accessed out-of-bounds and what the indices are. The only downside is that Quokka will run very slowly in this mode.
docs/docs/debugging.md:-   Build Quokka without GPU support but with `-DCMAKE_BUILD_TYPE=Release -DENABLE_ASAN=ON`. This turns on the AddressSanitizer, which checks for out-of-bounds array accesses and other memory bugs. This is faster than the previous method, but it produces less informative error messages (e.g., no array indices).
docs/docs/debugging.md:    -   This method may produce a lot of messages about memory leaks, [which are not necessarily bugs](https://stackoverflow.com/a/654766), and should not cause GPU crashes. These messages [can be disabled](https://stackoverflow.com/questions/51060801/how-to-suppress-leaksanitizer-report-when-running-under-fsanitize-address) if you are looking for, e.g., out-of-bounds array accesses, which is a class of bug that can cause a GPU crash.
docs/docs/debugging.md:-   On AMD GPUs, there is a [GPU-aware AddressSanitizer](https://rocm.docs.amd.com/en/latest/understand/using_gpu_sanitizer.html#compiling-for-address-sanitizer). Currently, enabling this requires manually changing the compiler flags.
docs/docs/debugging.md:## How to actually debug on GPUs
docs/docs/debugging.md:-   downsize the simulation to fit on a single GPU
docs/docs/debugging.md:-   start the simulation on an NVIDIA GPU from within CUDA-GDB (see the [CUDA-GDB documentation](https://docs.nvidia.com/cuda/cuda-gdb/index.html) and [slides](https://www.olcf.ornl.gov/wp-content/uploads/2021/06/cuda_training_series_cuda_debugging.pdf)).
docs/docs/debugging.md:-   hope CUDA-GDB does not itself crash
docs/docs/debugging.md:-   hope CUDA-GDB produces a useful error message that you can analyze
docs/docs/debugging.md:NVIDIA also provides the `compute-sanitizer` tool that is essentially the equivalent of AddressSanitizer (see the [ComputeSanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)). Unfortunately, it does not work as reliably as AddressSanitizer, and may itself crash while attempting to debug a GPU program.
docs/docs/debugging.md:For AMD GPUs, you have to use the AMD-provided debugger `rocgdb`. A tutorial its use is available [here](https://www.olcf.ornl.gov/wp-content/uploads/2021/04/rocgdb_hipmath_ornl_2021_v2.pdf).
docs/docs/debugging.md:AMD also provides a GPU-aware AddressSanitizer that can be enabled when building Quokka. Currently, the compiler flags must be manually modified in order to enable this. For details, see its [documentation](https://rocm.docs.amd.com/en/latest/understand/using_gpu_sanitizer.html#compiling-for-address-sanitizer).
docs/docs/debugging.md:## GPU kernel asynchronicity
docs/docs/debugging.md:**By default, GPU kernels launch asynchronously, i.e., execution of CPU code continues before the kernel starts on the GPU. This can cause synchronization problems if there is an implicit assumption about the order of operations with respect to CPU and GPU code.**
docs/docs/debugging.md:-   `CUDA_LAUNCH_BLOCKING=1` on NVIDIA GPUs, or
docs/docs/debugging.md:-   `HIP_LAUNCH_BLOCKING=1` on AMD GPUs.
docs/docs/debugging.md:This will cause the CPU to wait until the GPU kernel execution is complete before continuing past the call to `ParallelFor`.
docs/docs/debugging.md:For more details, refer to the [AMReX GPU debugging guide](https://amrex-codes.github.io/amrex/docs_html/Debugging.html#basic-gpu-debugging).
docs/docs/debugging.md:If you have tried *all* of the above steps, then you have to resort to adding `printf` statements within the GPU code. Note that `printf` inside GPU code is different from the CPU-side `printf` function, as explained in the [NVIDIA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#formatted-output).
docs/docs/error_checking.md:-   `AMREX_ALWAYS_ASSERT`: Always works on CPU. **Works on GPU only if "-DNDEBUG" is NOT added to the compiler flags. Note that CMake adds "-DNDEBUG" by default when "CMAKE_BUILD_TYPE=Release".** (See this [GitHub discussion](https://github.com/AMReX-Codes/amrex/discussions/2648) for details.)
docs/docs/error_checking.md:Because the default CMake flags added in Release mode causes ``AMREX_ALWAYS_ASSERT`` not to function in GPU code, ``amrex::Abort`` is the best option to use if you want to abort a GPU kernel.
docs/docs/error_checking.md:``amrex::Abort`` requires additional GPU register usage, so it should be used sparingly. The best strategy for error handling is often to set a value in an array that indicates an iterative solve failed in a given cell. (This is what Castro does for its nuclear burning networks.)
docs/docs/error_checking.md:For more details, see the [AMReX documentation on assertions and error checking](https://amrex-codes.github.io/amrex/docs_html/GPU.html#assertions-and-error-checking).
docs/docs/performance.md:-   Understand what a [GPU kernel](https://en.wikipedia.org/wiki/Compute_kernel) is. (For reference, consult these [notes](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/kernel_sm).)
docs/docs/performance.md:-   Know that calling ``amrex::ParallelFor`` launches a GPU kernel (when GPU support is enabled at compile time).
docs/docs/performance.md:## GPU hardware characteristics
docs/docs/performance.md:GPUs have hardware design features that make their performance characteristics significantly different from CPUs. In practice, two factors dominate GPU performance behavior:
docs/docs/performance.md:-   *Kernel launch latency:* this is a fundamental hardware characteristic of GPUs. It takes several microseconds (typically 3-10 microseconds, but it can vary depending on the compute kernel, the GPU hardware, the CPU hardware, and the driver) to launch a GPU kernel (i.e., to start running the code within an ``amrex::ParallelFor`` on the GPU). In practice, latency is generally longer for AMD and Intel GPUs.
docs/docs/performance.md:-   *Register pressure:* the number of registers per thread available for use by a given kernel is limited to the size of the GPU register file divided by the number of threads. If a kernel needs more registers than are available in the register file, the compiler will "spill" registers to memory, which will then make the kernel run very slowly. Alternatively, the number of concurrent threads can be reduced, which increases the number of registers available per thread.
docs/docs/performance.md:    -   For more details, see these [AMD website notes](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/) and OLCF [training materials](https://www.olcf.ornl.gov/wp-content/uploads/Intro_Register_pressure_ORNL_20220812_2083.pdf).
docs/docs/performance.md:A traditional rule of thumb for CPU-based MPI codes is that communication latency often limits performance when scaling to large number of CPU cores (or, equivalently, MPI ranks). We have found that this is *not* the case for Quokka when running on GPU nodes (by, e.g., adding additional dummy variables to the state arrays).
docs/docs/performance.md:-   GPU node performance is about 10x faster than CPU node performance, whereas network bandwidth is only 2-4x larger on GPU nodes compared to CPU nodes. The network bandwidth to compute ratio is therefore *lower* on GPU nodes than on CPU nodes.
docs/docs/performance.md:-   GPU kernel launch latency (3-10 microseconds) is often larger than the minimum MPI message latency (i.e., the latency for small messages to travel between nodes) of 2-3 microseconds.
docs/docs/performance.md:    -   *However,* combining multiple kernels can increase register pressure, which can decrease performance. There is no real way to know a priori whether there will be a net performance gain or loss without trying it out. The strategy that yields the best performance may be different for GPUs from different vendors!
docs/docs/performance.md:    -   *However,* this may increase the time lost due to kernel launch latency. This is an engineering trade-off that must be determined by performance measurements on the GPU hardware. This trade-off may be different on GPUs from different vendors!
docs/docs/performance.md:-   In order to decrease register pressure, avoid using ``printf``, ``assert``, and ``amrex::Abort`` in GPU code . All of these functions require using additional registers that could instead be allocated to the useful computations does in a kernel. This may require a significant code rewrite to handle errors in a different way. (You should *not* just ignore errors, e.g. in an iterative solver.)
docs/docs/performance.md:-   *Experts only:* Manually tune the number of GPU threads per block on a kernel-by-kernel basis. This can reduce register pressure by allowing each thread to use more registers. Note that this is an advanced optimization and should only be done with careful performance measurements done on multiple GPUs. The [AMReX documentation](https://amrex-codes.github.io/amrex/docs_html/GPU.html#gpu-block-size) provides guidance on how to do this.
docs/docs/references.bib:        title = "{QUOKKA: a code for two-moment AMR radiation hydrodynamics on GPUs}",
docs/docs/references.bib:        title = "{A novel numerical method for mixed-frame multigroup radiation-hydrodynamics with GPU acceleration implemented in the QUOKKA code}",
docs/docs/installation.md:**By default, Quokka compiles itself only for CPUs. If you want to run Quokka on GPUs, see the section "Running on GPUs" below.**
docs/docs/installation.md:## Running on GPUs
docs/docs/installation.md:By default, Quokka compiles itself to run only on CPUs. Quokka can run on either NVIDIA or AMD GPUs. Consult the sub-sections below for the build instructions for a given GPU vendor.
docs/docs/installation.md:### NVIDIA GPUs
docs/docs/installation.md:If you want to run on NVIDIA GPUs, re-build Quokka as shown below. (*CUDA >= 11.7 is required. Quokka is only supported on Volta V100 GPUs or newer models. Your MPI library* **must** *support CUDA-aware MPI.*)
docs/docs/installation.md:    cmake .. -DCMAKE_BUILD_TYPE=Release -DAMReX_GPU_BACKEND=CUDA -DAMReX_SPACEDIM=3 -G Ninja
docs/docs/installation.md:**All GPUs on a node must be visible from each MPI rank on the node for efficient GPU-aware MPI communication to take place via CUDA IPC.** When using the SLURM job scheduler, this means that `--gpu-bind` should be set to `none`.
docs/docs/installation.md:Note that 1D problems can run very slowly on GPUs due to a lack of sufficient parallelism. To run the test suite in a reasonable amount of time, you may wish to exclude the matter-energy exchange tests, e.g.:
docs/docs/installation.md:### AMD GPUs *(experimental, use at your own risk)*
docs/docs/installation.md:Compile with `-DAMReX_GPU_BACKEND=HIP`. Requires ROCm 5.2.0 or newer. Your MPI library **must** support GPU-aware MPI for AMD GPUs. Quokka has been tested on MI100 and MI250X GPUs, but there are known compiler issues that affect the correctness of simulation results (see <https://github.com/quokka-astro/quokka/issues/394> and <https://github.com/quokka-astro/quokka/issues/447>).
docs/docs/installation.md:### Intel GPUs *(does not compile)*
docs/docs/installation.md:Due to limitations in the Intel GPU programming model, Quokka currently cannot be compiled for Intel GPUs. (See <https://github.com/quokka-astro/quokka/issues/619> for the technical details.)
docs/docs/citation.md:    title = "{QUOKKA: a code for two-moment AMR radiation hydrodynamics on GPUs}",
docs/docs/index.md:Quokka is a two-moment radiation hydrodynamics code that uses the piecewise-parabolic method, with AMR and subcycling in time. Runs on CPUs (MPI+vectorized) or NVIDIA GPUs (MPI+CUDA) with a single-source codebase. Written in C++17. (100% Fortran-free.)
docs/docs/index.md:We use the AMReX library [@AMReX_JOSS] to provide patch-based adaptive mesh functionality. We take advantage of the C++ loop abstractions in AMReX in order to run with high performance on either CPUs or GPUs.
tests/blast_unigrid_256.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/StarCluster.in:amr.max_grid_size   = 64    # at least 128 for GPUs
tests/benchmark_unigrid_256.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/blast_unigrid_256_cpu.in:amr.max_grid_size   = 64   # at least 128 for GPUs
tests/SphericalCollapse.in:amr.max_grid_size   = 64    # at least 128 for GPUs
tests/blast_unigrid_128.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/radhydro_shell.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/radhydro_shell_amr.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/radhydro_shell_1024.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/StarCluster_AMR.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/benchmark_unigrid_1024.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/radhydro_shell_256.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/blast_amr_maxlev2.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/blast_unigrid_128_cpu.in:amr.max_grid_size   = 64    # at least 128 for GPUs
tests/benchmark_unigrid_4096.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/radhydro_shell_512.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/PopIII.in:amr.max_grid_size   = 128    # at least 128 for GPUs
tests/blast_32.in:amr.max_grid_size   = 64    # at least 128 for GPUs
tests/BinaryOrbit.in:amr.max_grid_size   = 32    # at least 128 for GPUs
tests/blast_unigrid_512.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/blast_unigrid_128_regression.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/blast_unigrid_2048.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/radhydro_shell_amr_512.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/blast_unigrid_1024.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/radhydro_shell_2048.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/benchmark_unigrid_512.in:amr.max_grid_size   = 128   # at least 128 for GPUs
tests/benchmark_unigrid_2048.in:amr.max_grid_size   = 128   # at least 128 for GPUs
CITATION.cff:  title: 'Quokka: A code for two-moment AMR radiation hydrodynamics on GPUs'
README.md:Quokka is a two-moment radiation hydrodynamics code that uses the piecewise-parabolic method, with AMR and subcycling in time. Runs on CPUs (MPI+vectorized) or NVIDIA GPUs (MPI+CUDA) with a single-source codebase. Written in C++17. (100% Fortran-free.)
README.md:* MPI library with GPU-aware support (OpenMPI, MPICH, or Cray MPI)
README.md:* CUDA 11.7+ (optional, for NVIDIA GPUs)
README.md:* ROCm 5.2.0+ (optional, for AMD GPUs)
README.md:* ADIOS2 2.9+ with GPU-aware support (optional, for writing terabyte-sized or larger outputs)
paper/performance_a100.csv:nodes,gpus,mzones,mzones_per_gpu,mzones_per_gpu_ideal,GPU_FillBoundary,GPU_scaling,node_scaling,size,problem_type
paper/quokka_v1.tex:\title[Two-moment AMR radiation hydrodynamics on GPUs]{\textsc{Quokka}: A code for two-moment AMR radiation hydrodynamics on GPUs}
paper/quokka_v1.tex:    We present \quokka, a new subcycling-in-time, block-structured adaptive mesh refinement (AMR) radiation hydrodynamics code optimised for graphics processing units (GPUs). \quokka~solves the equations of hydrodynamics with the piecewise parabolic method (PPM) in a method-of-lines formulation, and handles radiative transfer via the variable Eddington tensor (VET) radiation moment equations with a local closure. In order to maximise GPU performance, we combine explicit-in-time evolution of the radiation moment equations with the reduced speed-of-light approximation. We show results for a wide range of test problems for hydrodynamics, radiation, and coupled radiation hydrodynamics. On uniform grids in 3D, we achieve a peak of 93 million hydrodynamic updates per second per GPU, and 22 million radiation hydrodynamic updates per second per GPU. For radiation hydrodynamics problems on uniform grids in 3D, our code also scales from 4 GPUs to 256 GPUs with an efficiency of 80 per cent. The code is publicly released under an open-source license on \faGithub\href{https://github.com/BenWibking/quokka-code}{GitHub}.
paper/quokka_v1.tex:The unique feature of \quokka~is that it has been designed from the ground up to run efficiently on graphics processing units (GPUs). This design goal motivated our choice of both algorithms and low-level implementation details. While \quokka~is not the first GPU hydrodynamics code in astrophysics (others include \textsc{Gamer}, \citealt{Schive10a, Schive18a}, \textsc{Cholla}, \citealt{Schneider15a}, \textsc{Castro}, \citealt{Almgren_2020}, and \textsc{ARK-RT}, \citealt{Bloch_2021}), nor even the first AMR GPU code, it is the first to feature two-moment AMR RHD on GPUs.
paper/quokka_v1.tex:Bringing RHD to GPUs creates some unique challenges. Contemporary compute nodes are often limited by data bandwidth, both in terms of moving data between main memory and the CPU or GPU, and in terms of moving data between CPUs or GPUs. For this reason, implicit methods generally have poor scalability, due to the need for global communications during an implicit solve (see, e.g., Appendix E of \citealt{Skinner_2019}). This imbalance between computation and communication is magnified on GPUs. Likewise, robust implicit methods require iterative sparse matrix solvers, which achieve lower peak efficiency on GPUs compared to CPUs due to their heavy use of indirect addressing and highly branching control flow. These considerations motivate our choice of an explicit RSLA method. They also motivate our choice of time integration strategy, which as we detail below has been designed to maximize computation (and therefore minimize the relative amount of communication) on each hydrodynamic timestep. We show that, with this strategy, we are able to achieve update computation rates of $>90$ million zone updates per second per GPU for pure HD, and $>20$ million for RHD. We also achieve $\ge 80\%$ parallel efficiency out to 256 GPUs. This combination of performance and scaling makes \quokka~substantially faster than any other public RHD code.
paper/quokka_v1.tex:We solve the radiation transport subsystem (\autoref{eq:rad_energy}--\autoref{eq:rad_flux}, again omitting the terms on the right-hand side) in a similar method-of-lines fashion. Our approach is most similar to that of \cite{Skinner_2019}, who also evolve the radiation moment equations with a time-explicit method-of-lines approach; however, they do not use either PPM reconstruction or a reduced speed of light. Because even with the RSLA the signal speed for the radiation subsystem is substantially larger than for the hydrodynamic subsystem, we evolve the former explicitly in time with several radiation timesteps per hydrodynamic timestep. In the regime of applicability of the RSLA, this approach allows a much more computationally efficient solution to the radiation moment equations, due to the fact that explicit methods have a greater arithmetic intensity per byte of data, have simple memory access patterns and control flows (compared to implicit solvers), and do not require global communication across the computational domain in order to advance the solution in time. All these features are greatly beneficial on GPUs, where the ratio of floating-point arithmetic performance to memory bandwidth is typically greater than on CPUs.
paper/quokka_v1.tex:where $x_0 = 0.5$, $y_0 = 0.5$, $\tilde y = |y - y_0| - 0.25$, the shearing layer thickness $L = 0.01$, $\sigma = 0.2$, and perturbation amplitude $A = 0.01$. The initial pressure is uniform with $P = 2.5$ and we adopt an adiabatic index $\gamma = 1.4$. We enable AMR, with cells tagged for refinement if the relative density gradient on either side of the cell in either direction exceeds $0.2$, and we allow up to four levels of refinement on top of a base grid size of $2048^2$. Thus the peak resolution of the calculation is $32,768^2$. Each local AMR grid has a uniform size of $128^2$. We evolve the system to $t = 1.5$ with a CFL number of $0.4$, and show the resulting numerical solution in \autoref{fig:kh_zoom}. We are able to carry out this calculation on a single GPU in $\sim 4.5$ hours of wallclock time. While there appears to be no converged solution to this problem without explicit dissipation, we find that our hydrodynamic solver is able to resolve the Kelvin-Helmholz rolls with very little dissipation and with significant small-scale structure caused by secondary instabilities, as expected for inviscid simulations \citep{Lecoanet_2016}. There are no visible artifacts at resolution boundaries.
paper/quokka_v1.tex:We next present our results for the so-called Liska-Wendroff implosion test \citep{Hui_1999,Liska_2003}. This problem consists of the square domain $[0, 0.3]^2$, with an inner region $x+y \leq 0.15$ and an exterior region where $x + y > 0.15$ for an ideal gas with adiabatic index $\gamma = 1.4$. The inner region has initial density $\rho = 0.125$ and pressure $P = 0.14$ and the outer region begins with density $\rho = 1$ and pressure $P = 1$. We simulate the subsequent evolution to $t=2.5$ on a uniform grid of $1024^2$ cells with reflecting boundary conditions with a CFL number of $0.4$. These initial conditions lead to a shock directed toward the origin, which is then reflected many times by the upper and right walls before finally converging in a jet traveling away from the origin along the diagonal $x=y$, as shown in \autoref{fig:implosion}. \cite{Liska_2003} note that only codes that discretely preserve symmetry between x- and y-directions successfully produce the jet. In order to recover the jet in \quokka, we found it necessary to code the RK2-SSP integrator so that the fluxes in the x- and y-direction are added in an exactly symmetrical manner for each stage of the update. Additionally, when running the problem on NVIDIA GPUs, we preserve this symmetry only if we disable fused multiply-add (FMA) operations via the \texttt{nvcc} compiler option \texttt{fmad=false}, since the compiler otherwise breaks the symmetry expressed in the source code between the x- and y-direction fluxes. With this compiler option, \quokka~exactly preserves symmetry along the diagonal and successfully recovers the jet.
paper/quokka_v1.tex:The entire motivation for \quokka~is to achieve high performance on RHD problems run on GPUs. We therefore next test the performance and scaling of the code. All the tests we present were performed on the Gadi supercomputer at the National Computational Infrastructure\footnote{\url{https://nci.org.au/our-systems/hpc-systems}}, using the gpuvolta nodes. Each node has 2 24-core Intel Xeon Platinum 8268 (Cascade Lake) 2.9 GHz CPUs and 4 Nvidia Tesla Volta V100-SXM2-32GB GPUs. Nodes are coupled via HDR InfiniBand in a Dragonfly+ topology.
paper/quokka_v1.tex:We first demonstrate that \textsc{Quokka} has excellent parallel scaling efficiency when keeping the number of computational cells fixed per GPU (referred to as \emph{weak scaling}). For our first test of weak scaling, we show the scalability of the hydrodynamics solver on uniform grids, disabling mesh refinement and radiation. We simulate a Sedov-Taylor blast wave \citep{Sedov_1959,Taylor_1946} in a 3D periodic box on the domain $[-1, 1]$ in each coordinate direction. The initial conditions consist of a spherical region of high pressure $P = 10$ for radii $r < 0.1$ and low pressure $P = 0.1$ for $r \ge 0.1$, with a uniform density of $\rho = 1$ and zero velocity, for an ideal gas with adiabatic index $\gamma = 5/3$.
paper/quokka_v1.tex:We run with a varying number of GPUs with two $256^3$ grids per GPU, increasing the resolution of our simulation as we extend to greater numbers of GPUs. However, a power-of-two resolution increase does not easily map onto a jump from one GPU to four GPUs, so the single-GPU simulation only uses a grid size of $256^3$. The grid size of the simulations therefore ranges from $256^3$ (for 1 GPU) to $2048^3$ (for 256 GPUs). We set the \textsc{AMReX} domain decomposition parameters \texttt{blocking\_factor} and \texttt{max\_grid\_size} to a value of $128$, leading the computational grid to be decomposed into arrays of size $128^3$. (We also tested local grid sizes of $256^3$ but found only a few per cent performance improvement on this problem.) We use one MPI rank per GPU for all simulations. The CFL number is $0.25$ and we evolve for $100$ timesteps for each simulation. We assess performance by counting the total number of cell-updates and dividing by the number of GPUs in order to obtain the performance figure-of-merit in the units of 1 million cells (or zones) per timestep per GPU per second (Mzones/GPU/s).  We report the results in \autoref{table:weak_hydro_scaling}. 
paper/quokka_v1.tex:We find a $\approx 30\%$ drop in performance per GPU when going from 1 GPU to 4 GPUs, corresponding to using all 4 GPUs on a single node of the compute cluster. We hypothesize that this is due to the limited communication bandwidth between GPUs on a node, which is limited here by the bandwidth of the PCI-Express bus. Similar scaling behavior is observed when running the \textsc{K-Athena} hydrodynamics code on GPUs \citep{Grete_2019}, which uses an entirely different GPU programming framework and domain decomposition implementation. Nor is the phenomenon unique to GPU codes: \citet{Stone_2020} report a similar decrease in performance when going from one CPU to all the CPUs on a node for \textsc{Athena++}, which they also attribute to limitations of memory bandwidth. Regardless of the origin of the slow down, we observe very little further degradation in performance per GPU when going from 1 node (4 GPUs) to 64 nodes (256 GPUs), yielding a parallel efficiency of 88 per cent on 64 nodes when compared to running on 1 node.  We could not run on larger numbers of GPU nodes due to job size limitations, but we expect scaling to continue to thousands of GPUs based on the parallel scaling observed for other GPU hydrodynamics codes based on \textsc{AMReX}, such as \textsc{Castro} \citep{Almgren_2020}.
paper/quokka_v1.tex:    Nodes & GPUs & Mzones/GPU/s & Scaling efficiency & Grid size\\\hline
paper/quokka_v1.tex:        {weak_scaling_hydro.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\gpufill,6=\scaling,7=\scalingnode,8=\size}
paper/quokka_v1.tex:        {\nodes & \gpus & \mzonespergpu & \scalingnode & $\size$ \\}
paper/quokka_v1.tex:    \caption{Weak scaling efficiency for hydrodynamics as a function of the number of GPUs for a Sedov blast wave with periodic boundary conditions.}
paper/quokka_v1.tex:We next test the scaling behaviour for full radiation hydrodynamics solver on uniform grids. \autoref{table:weak_radhydro_scaling} lists the performance per GPU and parallel efficiency measured with respect to single-node performance for the radiation-driven shell test problem run for $50$ timesteps. Since we have many radiation substeps per hydrodynamic step (set here to 10; see \autoref{sssec:sync}), the performance metric in units of Mzones/GPU/s is lower by a factor comparable to but somewhat smaller than the number of radiation substeps per hydro step; a single radiation update is slightly less costly than a single hydrodynamic update. In this case, we observe a steeper drop in performance when going from 1 GPU to 4 GPUs (approximately a factor of 2). The lower parallel efficiency is not surprising, since \emph{each} radiation substep requires communicating boundary conditions between grids, so the amount of inter-GPU communication per hydro timestep increases significantly for radiation hydrodynamics. Nonetheless, as is the case for hydrodynamics, there is little additional performance penalty when scaling from 1 node to 64 nodes. We measure a parallel efficiency in this case of $80$ per cent.
paper/quokka_v1.tex:Finally, we point out that absolute speed of \quokka~is excellent. Comparison between CPU and GPU codes is non-trivial, since it obviously depends on the CPU-to-GPU ratio on a particular compute platform. However, it is worth pointing out that \quokka's update rate per core (normalised by the number of CPU cores per compute node) for \textit{radiation}-hydrodynamics on GPU is comparable to or better than \textsc{Athena++}'s for \textit{hydrodynamics} on CPU.
paper/quokka_v1.tex:Nodes & GPUs & Mzones/GPU/s & Scaling efficiency & Grid size\\\hline
paper/quokka_v1.tex:    {weak_scaling_radhydro.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\gpufill,6=\scaling,7=\scalingnode,8=\size}
paper/quokka_v1.tex:    {\nodes & \gpus & \mzonespergpu & \scalingnode & $\size$ \\}
paper/quokka_v1.tex:\caption{Weak scaling efficiency for radiation hydrodynamics as a function of the number of GPUs for the radiation-driven shell test (\autoref{section:shell}) with periodic boundary conditions.}
paper/quokka_v1.tex:Many applications of interest will seek to minimize either the total runtime of the simulation or the total node-hours used for a simulation for a problem of a fixed size. Additionally, most applications we are interested in will benefit from or require the use of AMR.  We therefore test the ability of \textsc{Quokka} to scale an AMR radiation hydrodynamic simulation of fixed size to larger numbers of GPUs in order to either minimize total runtime or total node-hours (referred to as \emph{strong scaling}). For this test, we initialize the radiation-driven shell problem (\autoref{section:shell}) on a base grid of $256^3$ cells with two levels of mesh refinement based on the relative gradient in the gas density. We run each simulation for 50 timesteps, with a CFL number of $0.3$ and PLM reconstruction for both hydrodynamics and radiation. We set the \textsc{AMReX} domain decomposition parameters \texttt{blocking\_factor} set to 32 and \texttt{max\_grid\_size} set to a value of $128$, so that all grids are between $32^3$ and $128^3$ in size, with possible non-cubic grids at intermediate sizes. The number of GPUs used for each simulation is varied, scaling from 1 node (4 GPUs) to 8 nodes (32 GPUs). This is a particularly stringest test, since the level-by-level AMR timestepping requires that each level be computed separately, limiting the amount of parallelism that can be distributed across GPUs. There is also additional communication overhead when AMR is enabled compared to a single-level uniform grid simulation. We show the scaling results in \autoref{table:strong_scaling}. Comparing \autoref{table:weak_radhydro_scaling} and \autoref{table:strong_scaling}, the performance per GPU for a single node is lower than that of a uniform grid simulation by $\approx 50$ per cent. (A similar, although somewhat smaller, overhead when enabling AMR is also observed with CPU codes, e.g., \textsc{Athena++}; \citealt{Stone_2020}). The scaling efficiency is reasonable for 2 and 4 nodes (67 per cent for 4 nodes), but drops significantly at 8 nodes to 54 per cent parallel efficiency. We hypothesize that this is due to the small number of cell-updates per GPU once 32 GPUs are in use for this problem (approximately $211^3 \approx 9.4 \times 10^6$ cells/GPU). We find that performance on a single GPU is significantly diminished for uniform-grid problems smaller than $256^3$, so this performance drop may be largely due to the inability to use all GPU hardware threads when the amount of work per GPU is small. Similar GPU performance behavior is observed when running \textsc{K-Athena} on GPUs for varying problem sizes per GPU \citep{Grete_2019}. This effect is also magnified by the sequential nature of the level-by-level timestepping. For level $l=1$, the number of cells per GPU drops below $256^3$ for 8 GPUs, and for level $l=2$, it drops below $256^3$ for 16 GPUs. High scaling efficiency is obtained before reaching these thresholds, so it appears that reasonable performance on GPUs may be obtained with AMR when all refinement levels have at least $256^3$ cells per GPU on average. In general, obtaining the best possible GPU performance may require an adjustment to the mesh refinement parameters usually used when running on CPUs. For self-gravitating problems, scaling may be aided by the self-similar nature of gravitational collapse, leading to an approximately equal number of cells on each refinement level for appropriate refinement criteria (see discussion in \citealt{Stone_2020}).
paper/quokka_v1.tex:Nodes & GPUs & Mzones/GPU/s & $\left<\frac{\text{Cells}}{\text{GPU}}\right>$ & \begin{tabular}{@{}r@{}}Scaling \\ efficiency\end{tabular} & Speedup\\\hline
paper/quokka_v1.tex:    {strong_scaling.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\gpufill,6=\scaling,7=\cellspergpu,8=\speedup}
paper/quokka_v1.tex:    {\nodes & \gpus & \mzonespergpu & ${\cellspergpu}^3$ & \scaling & {\speedup}x \\}
paper/quokka_v1.tex:    Nodes & GPUs & $\left<\frac{\text{Cells}}{\text{GPU}}\right>_{l=0}$ & $\left<\frac{\text{Cells}}{\text{GPU}}\right>_{l=1}$ & $\left<\frac{\text{Cells}}{\text{GPU}}\right>_{l=2}$ \\\hline
paper/quokka_v1.tex:        {strong_scaling.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\gpufill,6=\scaling,7=\cellspergpu,8=\speedup,9=\cellszero,10=\cellsone,11=\cellstwo}
paper/quokka_v1.tex:        {\nodes & \gpus & ${\cellszero}^3$ & ${\cellsone}^3$ & ${\cellstwo}^3$ \\}
paper/quokka_v1.tex:\caption{Strong scaling efficiency for radiation hydrodynamics as a function of the number of GPUs for the radiation-driven shell test (\autoref{section:shell}) with periodic boundary conditions on a base grid of $256^3$ cells and 2 levels of refinement. The number of cells per GPU is computed as an average over all timesteps.}
paper/quokka_v1.tex:In the near future, we plan to add support for self-gravity with \textsc{AMReX}'s geometric multigrid solver for GPUs \citep{AMReX_JOSS}, sink and star particles for star cluster simulations (e.g., \citealt{Krumholz04a, Offner09a}), and optically-thin line cooling for the interstellar medium. These additions will enable simulations of the interstellar medium, galactic winds, and star clusters, among others.
paper/quokka_v1.tex:Radiation hydrodynamics codes like ours will enable the widespread use of more accurate radiation transport methods and an ever-greater dynamic range in both space and time. As we approach the era of exascale supercomputers, we see a bright future for AMR radiation hydrodynamics on GPU architectures.
paper/quokka_diff.tex:\title[Two-moment AMR radiation hydrodynamics on GPUs]{\textsc{Quokka}: A code for two-moment AMR radiation hydrodynamics on GPUs}
paper/quokka_diff.tex:    We present \quokka, a new subcycling-in-time, block-structured adaptive mesh refinement (AMR) radiation hydrodynamics code optimised for graphics processing units (GPUs). \quokka~solves the equations of hydrodynamics with the piecewise parabolic method (PPM) in a method-of-lines formulation, and handles radiative transfer via the variable Eddington tensor (VET) radiation moment equations with a local closure. \DIFaddbegin \DIFadd{We use the AMReX library to handle the adaptive mesh management. }\DIFaddend In order to maximise GPU performance, we combine explicit-in-time evolution of the radiation moment equations with the reduced speed-of-light approximation. We show results for a wide range of test problems for hydrodynamics, radiation, and coupled radiation hydrodynamics. On uniform grids in 3D \DIFdelbegin \DIFdel{, we achieve a peak of 93 }\DIFdelend \DIFaddbegin \DIFadd{on a single GPU, our code achieves  $>250$ }\DIFaddend million hydrodynamic updates per second \DIFdelbegin \DIFdel{per GPU, and 22 }\DIFdelend \DIFaddbegin \DIFadd{and almost $40$ }\DIFaddend million radiation hydrodynamic updates per second\DIFdelbegin \DIFdel{per GPU}\DIFdelend . For radiation hydrodynamics problems on uniform grids in 3D, our code \DIFdelbegin \DIFdel{also }\DIFdelend scales from 4 GPUs to 256 GPUs with an efficiency of \DIFdelbegin \DIFdel{80 }\DIFdelend \DIFaddbegin \DIFadd{76 }\DIFaddend per cent. The code is publicly released under an open-source license on \faGithub\href{https://github.com/BenWibking/quokka-code}{GitHub}.
paper/quokka_diff.tex:The unique feature of \quokka~is that it has been designed from the ground up to run efficiently on graphics processing units (GPUs). This design goal motivated our choice of both algorithms and low-level implementation details. While \quokka~is not the first GPU hydrodynamics code in astrophysics (others include \textsc{Gamer}, \citealt{Schive10a, Schive18a}, \textsc{Cholla}, \citealt{Schneider15a}, \textsc{Castro}, \citealt{Almgren_2020}, and \textsc{ARK-RT}, \citealt{Bloch_2021}), nor even the first AMR GPU code, it is the first to feature two-moment AMR RHD on GPUs.
paper/quokka_diff.tex:Bringing RHD to GPUs creates some unique challenges. Contemporary compute nodes are often limited by data bandwidth, both in terms of moving data between main memory and the CPU or GPU, and in terms of moving data between CPUs or GPUs. For this reason, implicit methods generally have poor scalability, due to the need for global communications during an implicit solve (see, e.g., Appendix E of \citealt{Skinner_2019}). This imbalance between computation and communication is magnified on GPUs. Likewise, robust implicit methods require iterative sparse matrix solvers, which achieve lower peak efficiency on GPUs compared to CPUs due to their heavy use of indirect addressing and highly branching control flow. These considerations motivate our choice of an explicit RSLA method. They also motivate our choice of time integration strategy, which as we detail below has been designed to maximize computation (and therefore minimize the relative amount of communication) on each hydrodynamic timestep. We show that, with this strategy, we are able to achieve update computation rates of \DIFdelbegin \DIFdel{$>90$ }\DIFdelend \DIFaddbegin \DIFadd{$>250$ }\DIFaddend million zone updates per second per GPU for pure HD, and \DIFdelbegin \DIFdel{$>20$ }\DIFdelend \DIFaddbegin \DIFadd{nearly $40$ }\DIFaddend million for RHD. We also achieve \DIFdelbegin \DIFdel{$\ge 80\%$ parallel efficiency }\DIFdelend \DIFaddbegin \DIFadd{$\ge 75\%$ parallel efficiency (compared to single-node performance) }\DIFaddend out to 256 GPUs. This combination of performance and scaling makes \quokka~substantially faster than any other public RHD code.
paper/quokka_diff.tex:We solve the radiation transport subsystem (\autoref{eq:rad_energy}--\autoref{eq:rad_flux}, again omitting the terms on the right-hand side) in a similar method-of-lines fashion. Our approach is most similar to that of \cite{Skinner_2019}, who also evolve the radiation moment equations with a time-explicit method-of-lines approach; however, they do not use either PPM reconstruction or a reduced speed of light. Because even with the RSLA the signal speed for the radiation subsystem is substantially larger than for the hydrodynamic subsystem, we evolve the former explicitly in time with several radiation timesteps per hydrodynamic timestep. In the regime of applicability of the RSLA, this approach allows a much more computationally efficient solution to the radiation moment equations, due to the fact that explicit methods have a greater arithmetic intensity per byte of data, have simple memory access patterns and control flows (compared to implicit solvers), and do not require global communication across the computational domain in order to advance the solution in time. All these features are greatly beneficial on GPUs, where the ratio of floating-point arithmetic performance to memory bandwidth is typically greater than on CPUs.
paper/quokka_diff.tex:where $x_0 = 0.5$, $y_0 = 0.5$, $\tilde y = |y - y_0| - 0.25$, the shearing layer thickness $L = 0.01$, $\sigma = 0.2$, and perturbation amplitude $A = 0.01$. The initial pressure is uniform with $P = 2.5$ and we adopt an adiabatic index $\gamma = 1.4$. We enable AMR, with cells tagged for refinement if the relative density gradient on either side of the cell in either direction exceeds $0.2$, and we allow up to four levels of refinement on top of a base grid size of $2048^2$. Thus the peak resolution of the calculation is $32,768^2$. Each local AMR grid has a uniform size of $128^2$. We evolve the system to $t = 1.5$ with a CFL number of $0.4$, and show the resulting numerical solution in \autoref{fig:kh_zoom}. We are able to carry out this calculation on a single GPU in $\sim 4.5$ hours of wallclock time. While there appears to be no converged solution to this problem without explicit dissipation, we find that our hydrodynamic solver is able to resolve the Kelvin-Helmholz rolls with very little dissipation and with significant small-scale structure caused by secondary instabilities, as expected for inviscid simulations \citep{Lecoanet_2016}. There are no visible artifacts at resolution boundaries.
paper/quokka_diff.tex:We next present our results for the so-called Liska-Wendroff implosion test \citep{Hui_1999,Liska_2003}. This problem consists of the square domain $[0, 0.3]^2$, with an inner region $x+y \leq 0.15$ and an exterior region where $x + y > 0.15$ for an ideal gas with adiabatic index $\gamma = 1.4$. The inner region has initial density $\rho = 0.125$ and pressure $P = 0.14$ and the outer region begins with density $\rho = 1$ and pressure $P = 1$. We simulate the subsequent evolution to $t=2.5$ on a uniform grid of $1024^2$ cells with reflecting boundary conditions with a CFL number of $0.4$. These initial conditions lead to a shock directed toward the origin, which is then reflected many times by the upper and right walls before finally converging in a jet traveling away from the origin along the diagonal $x=y$, as shown in \autoref{fig:implosion}. \cite{Liska_2003} note that only codes that discretely preserve symmetry between x- and y-directions successfully produce the jet. In order to recover the jet in \quokka, we found it necessary to code the RK2-SSP integrator so that the fluxes in the x- and y-direction are added in an exactly symmetrical manner for each stage of the update. Additionally, when running the problem on NVIDIA GPUs, we preserve this symmetry only if we disable fused multiply-add (FMA) operations via the \texttt{nvcc} compiler option \texttt{fmad=false}, since the compiler otherwise breaks the symmetry expressed in the source code between the x- and y-direction fluxes. With this compiler option, \quokka~exactly preserves symmetry along the diagonal and successfully recovers the jet.
paper/quokka_diff.tex:The entire motivation for \quokka~is to achieve high performance on RHD problems run on GPUs. We therefore next test the performance and scaling of the code. All the tests we present were performed on the Gadi supercomputer at the National Computational Infrastructure\footnote{\url{https://nci.org.au/our-systems/hpc-systems}}, using the gpuvolta nodes. Each node has 2 24-core Intel Xeon Platinum 8268 (Cascade Lake) 2.9 GHz CPUs and 4 Nvidia Tesla Volta V100-SXM2-32GB GPUs \DIFaddbegin \DIFadd{connected to each other in an all-to-all topology with NVLink 2.0}\DIFaddend . Nodes are coupled via HDR InfiniBand in a Dragonfly+ topology.
paper/quokka_diff.tex:We first demonstrate that \textsc{Quokka} has excellent parallel scaling efficiency when keeping the number of computational cells fixed per GPU (referred to as \emph{weak scaling}). For our first test of weak scaling, we show the scalability of the hydrodynamics solver on uniform grids, disabling mesh refinement and radiation. We simulate a Sedov-Taylor blast wave \citep{Sedov_1959,Taylor_1946} in a 3D periodic box on the domain $[-1, 1]$ in each coordinate direction. The initial conditions consist of a spherical region of high pressure $P = 10$ for radii $r < 0.1$ and low pressure $P = 0.1$ for $r \ge 0.1$, with a uniform density of $\rho = 1$ and zero velocity, for an ideal gas with adiabatic index $\gamma = 5/3$.
paper/quokka_diff.tex:We run with a varying number of GPUs with two $256^3$ grids per GPU, increasing the resolution of our simulation as we extend to greater numbers of GPUs. However, a power-of-two resolution increase does not easily map onto a jump from one GPU to four GPUs, so the single-GPU simulation only uses a grid size of $256^3$. The grid size of the simulations therefore ranges from $256^3$ (for 1 GPU) to $2048^3$ (for 256 GPUs). We set the \textsc{AMReX} domain decomposition parameters \texttt{blocking\_factor} and \texttt{max\_grid\_size} to a value of $128$, leading the computational grid to be decomposed into arrays of size $128^3$. (We also tested local grid sizes of $256^3$ but found only a few per cent performance improvement on this problem.) We use one MPI rank per GPU for all simulations. The CFL number is $0.25$ and we evolve for $100$ timesteps for each simulation. We assess performance by counting the total number of cell-updates and dividing by the number of GPUs in order to obtain the performance figure-of-merit in the units of 1 million cells (or zones) per timestep per GPU per second (Mzones/GPU/s).  We report the results in \autoref{table:weak_hydro_scaling}.
paper/quokka_diff.tex:We find a \DIFdelbegin \DIFdel{$\approx 30\%$ }\DIFdelend \DIFaddbegin \DIFadd{$\approx 40\%$ }\DIFaddend drop in performance per GPU when going from 1 GPU to 4 GPUs, corresponding to using all 4 GPUs on a single node of the compute cluster. We hypothesize that this is due to the limited communication bandwidth between GPUs on a node\DIFdelbegin \DIFdel{, which is limited here by the bandwidth of the PCI-Express bus. Similar scaling behavior is observed when running the }\textsc{\DIFdel{K-Athena}} %DIFAUXCMD
paper/quokka_diff.tex:\DIFdel{hydrodynamics code on GPUs \mbox{%DIFAUXCMD
paper/quokka_diff.tex:, which uses an entirely different GPU programming framework and domain decomposition implementation. Nor is the phenomenon unique to GPU codes: }\DIFdelend \DIFaddbegin \DIFadd{. For intra-node scaling on CPUs, }\DIFaddend \citet{Stone_2020} report a similar decrease in performance when going from one CPU to all the CPUs on a node for \textsc{Athena++}, which they \DIFdelbegin \DIFdel{also }\DIFdelend attribute to limitations of memory bandwidth. \DIFdelbegin \DIFdel{Regardless of the origin of the slow down, we observe very little further degradation }\DIFdelend \DIFaddbegin \DIFadd{However, significantly different scaling behavior is observed when running the }\textsc{\DIFadd{K-Athena}} \DIFadd{hydrodynamics code on GPUs \mbox{%DIFAUXCMD
paper/quokka_diff.tex:on the Summit supercomputer}\footnote{\url{https://www.olcf.ornl.gov/olcf-resources/compute-systems/summit/}}\DIFadd{, finding a $99$ per cent weak scaling efficiency going from 1 GPU to 6 GPUs on a single node, so there may be some inefficiency in our current GPU-to-GPU communication method. We find that using CUDA-aware MPI does not improve performance for our code. However, we observe only a modest drop }\DIFaddend in performance per GPU when going from 1 node (4 GPUs) to 64 nodes (256 GPUs), yielding a parallel efficiency of \DIFdelbegin \DIFdel{88 }\DIFdelend \DIFaddbegin \DIFadd{83 }\DIFaddend per cent on 64 nodes when compared to running on 1 node.  We could not run on larger numbers of GPU nodes due to job size limitations, but we expect scaling to continue to thousands of GPUs based on the parallel scaling observed for other GPU hydrodynamics codes based on \textsc{AMReX}, such as \textsc{Castro} \citep{Almgren_2020}.
paper/quokka_diff.tex:\DIFaddbegin \DIFadd{In }\autoref{table:weak_hydro_scaling_plm}\DIFadd{, we show the same performance numbers as in }\autoref{table:weak_hydro_scaling}\DIFadd{, but using PLM reconstruction for each simulation instead of PPM reconstruction. We find that the performance improves significantly on a single GPU, going from $113$ million zone-updates per second to $158$ million zone-updates per second. However, communication overheads limit the relative performance improvement when using large numbers of nodes, as the $64$-node case goes from $59$ million zone-updates per GPU per second using PPM to only $65$ million zone-updates per GPU per second using PLM. Since the computations on each local grid are less expensive with PLM but the communication costs remain the same, the scaling efficiency decreases slightly as well, from $83$ per cent to $76$ per cent.
paper/quokka_diff.tex:%DIFDELCMD <     Nodes & GPUs & Mzones/GPU/s & Scaling efficiency & Grid size\\\hline
paper/quokka_diff.tex:%DIFDELCMD <         {weak_scaling_hydro.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\gpufill,6=\scaling,7=\scalingnode,8=\size}
paper/quokka_diff.tex:%DIFDELCMD <         {\nodes & \gpus & \mzonespergpu & \scalingnode & $\size$ \\}
paper/quokka_diff.tex:        Nodes   & GPUs  & Mzones/GPU/s  & Scaling efficiency (\%) & Grid size                         \\\hline
paper/quokka_diff.tex:        {weak_scaling_hydro.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\mzonespergpuideal,6=\gpufill,7=\scaling,8=\scalingnode,9=\size}
paper/quokka_diff.tex:        {\nodes & \gpus & \mzonespergpu & \scalingnode       & $\size$ \\}
paper/quokka_diff.tex:    \DIFaddendFL \caption{Weak scaling efficiency for hydrodynamics \DIFaddbeginFL \DIFaddFL{with PPM reconstruction }\DIFaddendFL as a function of the number of GPUs for a Sedov blast wave with periodic boundary conditions.}
paper/quokka_diff.tex:        Nodes   & GPUs  & Mzones/GPU/s  & Scaling efficiency (\%) & Grid size                         \\\hline
paper/quokka_diff.tex:        {weak_scaling_hydro_plm.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\scaling,6=\scalingnode,7=\size}
paper/quokka_diff.tex:        {\nodes & \gpus & \mzonespergpu & \scalingnode       & $\size$ \\}
paper/quokka_diff.tex:    \caption{\DIFaddFL{Weak scaling efficiency for hydrodynamics with PLM reconstruction as a function of the number of GPUs for a Sedov blast wave with periodic boundary conditions.}}
paper/quokka_diff.tex:\DIFaddend We next test the scaling behaviour for full radiation hydrodynamics solver on uniform grids. \autoref{table:weak_radhydro_scaling} lists the performance per GPU and parallel efficiency measured with respect to single-node performance for the radiation-driven shell test problem run for $50$ timesteps. Since we have many radiation substeps per hydrodynamic step (set here to 10; see \autoref{sssec:sync}), the performance metric in units of Mzones/GPU/s is lower by a factor comparable to but somewhat smaller than the number of radiation substeps per hydro step; a single radiation update is slightly less costly than a single hydrodynamic update. In this case, we observe a steeper drop in performance when going from 1 GPU to 4 GPUs (approximately a factor of 2). The lower parallel efficiency is not surprising, since \emph{each} radiation substep requires communicating boundary conditions between grids, so the amount of inter-GPU communication per hydro timestep increases significantly for radiation hydrodynamics. Nonetheless, as is the case for hydrodynamics, there is little additional performance penalty when scaling from 1 node to 64 nodes. We measure a parallel efficiency in this case of \DIFdelbegin \DIFdel{$80$ }\DIFdelend \DIFaddbegin \DIFadd{$76$ }\DIFaddend per cent.
paper/quokka_diff.tex:\DIFaddbegin \DIFadd{in }\autoref{table:scaling_a100}\DIFadd{, we list the performance metrics of the code on both the Sedov problem and the radiation-driven shell problem running on a compute node with newer NVIDIA A100 GPUs. Since we only have access to a limited number of these GPUs, we only show performance data for a single GPU and a single compute node (4 GPUs). The single GPU case achieves $254$ million hydrodynamic zone-updates per second using PPM, making Quokka, as far as we are aware, the fastest PPM hydrodynamics code that currently exists. On the radiation-driven shell problem, the code achieves $39$ million radiation hydrodynamic zone-updates per second. In both cases, the performance per GPU drops by a factor of approximately 2 when using all 4 GPUs on the node. This is almost entirely due to the time spent communicating boundary conditions, as shown in the table (`B.C. fill time', which denotes the percentage of total wall time spent filling ghost cells for each local grid). If communication of boundary data and computation over the local grids could be perfectly overlapped, the parallel efficiency going from 1 GPU to 4 GPUs would be $> 99$ per cent for hydrodynamics and $> 96$ for radiation hydrodynamics.
paper/quokka_diff.tex:\DIFaddend Finally, we point out that absolute speed of \quokka~is excellent. Comparison between CPU and GPU codes is non-trivial, since it obviously depends on the CPU-to-GPU ratio on a particular compute platform. However, it is worth pointing out that \quokka's update rate per core (normalised by the number of CPU cores per compute node) for \textit{radiation}-hydrodynamics on GPU is comparable to or better than \textsc{Athena++}'s for \textit{hydrodynamics} on CPU.
paper/quokka_diff.tex:%DIFDELCMD < Nodes & GPUs & Mzones/GPU/s & Scaling efficiency & Grid size\\\hline
paper/quokka_diff.tex:%DIFDELCMD <     {weak_scaling_radhydro.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\gpufill,6=\scaling,7=\scalingnode,8=\size}
paper/quokka_diff.tex:%DIFDELCMD <     {\nodes & \gpus & \mzonespergpu & \scalingnode & $\size$ \\}
paper/quokka_diff.tex:        Nodes   & GPUs  & Mzones/GPU/s  & Scaling efficiency (\%) & Grid size                         \\\hline
paper/quokka_diff.tex:        {weak_scaling_radhydro.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\mzonespergpuideal,6=\gpufill,7=\scaling,8=\scalingnode,9=\size}
paper/quokka_diff.tex:        {\nodes & \gpus & \mzonespergpu & \scalingnode       & $\size$ \\}
paper/quokka_diff.tex:    \DIFaddendFL \caption{Weak scaling efficiency for radiation hydrodynamics as a function of the number of GPUs for the radiation-driven shell test (\autoref{section:shell}) with periodic boundary conditions.}
paper/quokka_diff.tex:        GPUs  & Mzones/GPU/s  & B.C. fill time (\%) & Grid size & Problem                         \\\hline
paper/quokka_diff.tex:        {performance_a100.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\mzonespergpuideal,6=\gpufill,7=\scaling,8=\scalingnode,9=\size,10=\problemtype}
paper/quokka_diff.tex:        {\gpus & \mzonespergpu & \gpufill & $\size$       & \problemtype \\}
paper/quokka_diff.tex:    \caption{\DIFaddFL{Performance for both Sedov and radiating shell tests as a function of the number of GPUs for a single node with 4 NVIDIA A100 GPUs.}}
paper/quokka_diff.tex:Many applications of interest will seek to minimize either the total runtime of the simulation or the total node-hours used for a simulation for a problem of a fixed size. Additionally, most applications we are interested in will benefit from or require the use of AMR.  We therefore test the ability of \textsc{Quokka} to scale an AMR radiation hydrodynamic simulation of fixed size to larger numbers of GPUs in order to either minimize total runtime or total node-hours (referred to as \emph{strong scaling}). For this test, we initialize the radiation-driven shell problem (\autoref{section:shell}) on a base grid of $256^3$ cells with two levels of mesh refinement based on the relative gradient in the gas density. We run each simulation for 50 timesteps, with a CFL number of $0.3$ and PLM reconstruction for both hydrodynamics and radiation. We set the \textsc{AMReX} domain decomposition parameters \texttt{blocking\_factor} set to 32 and \texttt{max\_grid\_size} set to a value of $128$, so that all grids are between $32^3$ and $128^3$ in size, with possible non-cubic grids at intermediate sizes. The number of GPUs used for each simulation is varied, scaling from 1 node (4 GPUs) to 8 nodes (32 GPUs). This is a particularly stringest test, since the level-by-level AMR timestepping requires that each level be computed separately, limiting the amount of parallelism that can be distributed across GPUs. There is also additional communication overhead when AMR is enabled compared to a single-level uniform grid simulation. We show the scaling results in \autoref{table:strong_scaling}. Comparing \autoref{table:weak_radhydro_scaling} and \autoref{table:strong_scaling}, the performance per GPU for a single node is lower than that of a uniform grid simulation by $\approx 50$ per cent. (A similar, although somewhat smaller, overhead when enabling AMR is also observed with CPU codes, e.g., \textsc{Athena++}; \citealt{Stone_2020}). The scaling efficiency is reasonable for 2 and 4 nodes (\DIFdelbegin \DIFdel{67 }\DIFdelend \DIFaddbegin \DIFadd{66 }\DIFaddend per cent for 4 nodes), but drops significantly at 8 nodes to \DIFdelbegin \DIFdel{54 }\DIFdelend \DIFaddbegin \DIFadd{53 }\DIFaddend per cent parallel efficiency. We hypothesize that this is due to the small number of cell-updates per GPU once 32 GPUs are in use for this problem (approximately $211^3 \approx 9.4 \times 10^6$ cells/GPU). We find that performance on a single GPU is significantly diminished for uniform-grid problems smaller than $256^3$, so this performance drop may be largely due to the inability to use all GPU hardware threads when the amount of work per GPU is small. Similar GPU performance behavior is observed when running \textsc{K-Athena} on GPUs for varying problem sizes per GPU \citep{Grete_2019}. This effect is also magnified by the sequential nature of the level-by-level timestepping. For level $l=1$, the number of cells per GPU drops below $256^3$ for 8 GPUs, and for level $l=2$, it drops below $256^3$ for 16 GPUs. High scaling efficiency is obtained before reaching these thresholds, so it appears that reasonable performance on GPUs may be obtained with AMR when all refinement levels have at least $256^3$ cells per GPU on average. In general, obtaining the best possible GPU performance may require an adjustment to the mesh refinement parameters usually used when running on CPUs. For self-gravitating problems, scaling may be aided by the self-similar nature of gravitational collapse, leading to an approximately equal number of cells on each refinement level for appropriate refinement criteria (see discussion in \citealt{Stone_2020}).
paper/quokka_diff.tex:        Nodes   & GPUs  & Mzones/GPU/s  & $\left<\frac{\text{Cells}}{\text{GPU}}\right>$ & \begin{tabular}{@{}r@{}}Scaling \\ efficiency\end{tabular} & Speedup                               \\\hline
paper/quokka_diff.tex:        {strong_scaling.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\mzonespergpuideal,6=\gpufill,7=\scaling,8=\cellspergpu,9=\speedup,10=\cellszero,11=\cellsone,12=\cellstwo}
paper/quokka_diff.tex:        {\nodes & \gpus & \mzonespergpu & ${\cellspergpu}^3$                             & \scaling                   & {\speedup}x \\}
paper/quokka_diff.tex:        Nodes   & GPUs  & $\left<\frac{\text{Cells}}{\text{GPU}}\right>_{l=0}$ & $\left<\frac{\text{Cells}}{\text{GPU}}\right>_{l=1}$ & $\left<\frac{\text{Cells}}{\text{GPU}}\right>_{l=2}$ \\\hline
paper/quokka_diff.tex:        {strong_scaling.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\mzonespergpuideal,6=\gpufill,7=\scaling,8=\cellspergpu,9=\speedup,10=\cellszero,11=\cellsone,12=\cellstwo}
paper/quokka_diff.tex:        {\nodes & \gpus & ${\cellszero}^3$                                     & ${\cellsone}^3$                                      & ${\cellstwo}^3$ \\}
paper/quokka_diff.tex:    \DIFaddendFL \caption{Strong scaling efficiency for radiation hydrodynamics as a function of the number of GPUs for the radiation-driven shell test (\autoref{section:shell}) with periodic boundary conditions on a base grid of $256^3$ cells and 2 levels of refinement. The number of cells per GPU is computed as an average over all timesteps.}
paper/quokka_diff.tex:In the near future, we plan to add support for self-gravity with \textsc{AMReX}'s geometric multigrid solver for GPUs \citep{AMReX_JOSS}, sink and star particles for star cluster simulations (e.g., \citealt{Krumholz04a, Offner09a}), and optically-thin line cooling for the interstellar medium. These additions will enable simulations of the interstellar medium, galactic winds, and star clusters, among others.
paper/quokka_diff.tex:Radiation hydrodynamics codes like ours will enable the widespread use of more accurate radiation transport methods and an ever-greater dynamic range in both space and time. As we approach the era of exascale supercomputers, we see a bright future for AMR radiation hydrodynamics on GPU architectures.
paper/strong_scaling.csv:nodes,gpus,mzones,mzones_per_gpu,mzones_per_gpu_ideal,GPU_FillBoundary,GPU_scaling,average_cells_per_gpu,speedup,cells_per_gpu_0,cells_per_gpu_1,cells_per_gpu_2
paper/weak_scaling_hydro.csv:nodes,gpus,mzones,mzones_per_gpu,mzones_per_gpu_ideal,GPU_FillBoundary,GPU_scaling,GPU_scaling_1node,size
paper/weak_scaling_hydro_plm.csv:nodes,gpus,mzones,mzones_per_gpu,GPU_scaling,GPU_scaling_1node,size
paper/weak_scaling_radhydro.csv:nodes,gpus,mzones,mzones_per_gpu,mzones_per_gpu_ideal,GPU_FillBoundary,GPU_scaling,GPU_scaling_1node,size
paper/quokka.tex:\title[Two-moment AMR radiation hydrodynamics on GPUs]{\textsc{Quokka}: A code for two-moment AMR radiation hydrodynamics on GPUs}
paper/quokka.tex:    We present \quokka, a new subcycling-in-time, block-structured adaptive mesh refinement (AMR) radiation hydrodynamics code optimised for graphics processing units (GPUs). \quokka~solves the equations of hydrodynamics with the piecewise parabolic method (PPM) in a method-of-lines formulation, and handles radiative transfer via the variable Eddington tensor (VET) radiation moment equations with a local closure. We use the AMReX library to handle the adaptive mesh management. In order to maximise GPU performance, we combine explicit-in-time evolution of the radiation moment equations with the reduced speed-of-light approximation. We show results for a wide range of test problems for hydrodynamics, radiation, and coupled radiation hydrodynamics. On uniform grids in 3D on a single GPU, our code achieves  $>250$ million hydrodynamic updates per second and almost $40$ million radiation hydrodynamic updates per second. For radiation hydrodynamics problems on uniform grids in 3D, our code scales from 4 GPUs to 256 GPUs with an efficiency of 76 per cent. The code is publicly released under an open-source license on \faGithub\href{https://github.com/BenWibking/quokka-code}{GitHub}.
paper/quokka.tex:The unique feature of \quokka~is that it has been designed from the ground up to run efficiently on graphics processing units (GPUs). This design goal motivated our choice of both algorithms and low-level implementation details. While \quokka~is not the first GPU hydrodynamics code in astrophysics (others include \textsc{Gamer}, \citealt{Schive10a, Schive18a}, \textsc{Cholla}, \citealt{Schneider15a}, \textsc{Castro}, \citealt{Almgren_2020}, and \textsc{ARK-RT}, \citealt{Bloch_2021}), nor even the first AMR GPU code, it is the first to feature two-moment AMR RHD on GPUs.
paper/quokka.tex:Bringing RHD to GPUs creates some unique challenges. Contemporary compute nodes are often limited by data bandwidth, both in terms of moving data between main memory and the CPU or GPU, and in terms of moving data between CPUs or GPUs. For this reason, implicit methods generally have poor scalability, due to the need for global communications during an implicit solve (see, e.g., Appendix E of \citealt{Skinner_2019}). This imbalance between computation and communication is magnified on GPUs. Likewise, robust implicit methods require iterative sparse matrix solvers, which achieve lower peak efficiency on GPUs compared to CPUs due to their heavy use of indirect addressing and highly branching control flow. These considerations motivate our choice of an explicit RSLA method. They also motivate our choice of time integration strategy, which as we detail below has been designed to maximize computation (and therefore minimize the relative amount of communication) on each hydrodynamic timestep. We show that, with this strategy, we are able to achieve update computation rates of $>250$ million zone updates per second per GPU for pure HD, and nearly $40$ million for RHD. We also achieve $\ge 75\%$ parallel efficiency (compared to single-node performance) out to 256 GPUs. This combination of performance and scaling makes \quokka~substantially faster than any other public RHD code.
paper/quokka.tex:We solve the radiation transport subsystem (\autoref{eq:rad_energy}--\autoref{eq:rad_flux}, again omitting the terms on the right-hand side) in a similar method-of-lines fashion. Our approach is most similar to that of \cite{Skinner_2019}, who also evolve the radiation moment equations with a time-explicit method-of-lines approach; however, they do not use either PPM reconstruction or a reduced speed of light. Because even with the RSLA the signal speed for the radiation subsystem is substantially larger than for the hydrodynamic subsystem, we evolve the former explicitly in time with several radiation timesteps per hydrodynamic timestep. In the regime of applicability of the RSLA, this approach allows a much more computationally efficient solution to the radiation moment equations, due to the fact that explicit methods have a greater arithmetic intensity per byte of data, have simple memory access patterns and control flows (compared to implicit solvers), and do not require global communication across the computational domain in order to advance the solution in time. All these features are greatly beneficial on GPUs, where the ratio of floating-point arithmetic performance to memory bandwidth is typically greater than on CPUs.
paper/quokka.tex:where $x_0 = 0.5$, $y_0 = 0.5$, $\tilde y = |y - y_0| - 0.25$, the shearing layer thickness $L = 0.01$, $\sigma = 0.2$, and perturbation amplitude $A = 0.01$. The initial pressure is uniform with $P = 2.5$ and we adopt an adiabatic index $\gamma = 1.4$. We enable AMR, with cells tagged for refinement if the relative density gradient on either side of the cell in either direction exceeds $0.2$, and we allow up to four levels of refinement on top of a base grid size of $2048^2$. Thus the peak resolution of the calculation is $32,768^2$. Each local AMR grid has a uniform size of $128^2$. We evolve the system to $t = 1.5$ with a CFL number of $0.4$, and show the resulting numerical solution in \autoref{fig:kh_zoom}. We are able to carry out this calculation on a single GPU in $\sim 4.5$ hours of wallclock time. While there appears to be no converged solution to this problem without explicit dissipation, we find that our hydrodynamic solver is able to resolve the Kelvin-Helmholz rolls with very little dissipation and with significant small-scale structure caused by secondary instabilities, as expected for inviscid simulations \citep{Lecoanet_2016}. There are no visible artifacts at resolution boundaries.
paper/quokka.tex:We next present our results for the so-called Liska-Wendroff implosion test \citep{Hui_1999,Liska_2003}. This problem consists of the square domain $[0, 0.3]^2$, with an inner region $x+y \leq 0.15$ and an exterior region where $x + y > 0.15$ for an ideal gas with adiabatic index $\gamma = 1.4$. The inner region has initial density $\rho = 0.125$ and pressure $P = 0.14$ and the outer region begins with density $\rho = 1$ and pressure $P = 1$. We simulate the subsequent evolution to $t=2.5$ on a uniform grid of $1024^2$ cells with reflecting boundary conditions with a CFL number of $0.4$. These initial conditions lead to a shock directed toward the origin, which is then reflected many times by the upper and right walls before finally converging in a jet traveling away from the origin along the diagonal $x=y$, as shown in \autoref{fig:implosion}. \cite{Liska_2003} note that only codes that discretely preserve symmetry between x- and y-directions successfully produce the jet. In order to recover the jet in \quokka, we found it necessary to code the RK2-SSP integrator so that the fluxes in the x- and y-direction are added in an exactly symmetrical manner for each stage of the update. Additionally, when running the problem on NVIDIA GPUs, we preserve this symmetry only if we disable fused multiply-add (FMA) operations via the \texttt{nvcc} compiler option \texttt{fmad=false}, since the compiler otherwise breaks the symmetry expressed in the source code between the x- and y-direction fluxes. With this compiler option, \quokka~exactly preserves symmetry along the diagonal and successfully recovers the jet.
paper/quokka.tex:The entire motivation for \quokka~is to achieve high performance on RHD problems run on GPUs. We therefore next test the performance and scaling of the code. All the tests we present were performed on the Gadi supercomputer at the National Computational Infrastructure\footnote{\url{https://nci.org.au/our-systems/hpc-systems}}, using the gpuvolta nodes. Each node has 2 24-core Intel Xeon Platinum 8268 (Cascade Lake) 2.9 GHz CPUs and 4 Nvidia Tesla Volta V100-SXM2-32GB GPUs connected to each other in an all-to-all topology with NVLink 2.0. Nodes are coupled via HDR InfiniBand in a Dragonfly+ topology.
paper/quokka.tex:We first demonstrate that \textsc{Quokka} has excellent parallel scaling efficiency when keeping the number of computational cells fixed per GPU (referred to as \emph{weak scaling}). For our first test of weak scaling, we show the scalability of the hydrodynamics solver on uniform grids, disabling mesh refinement and radiation. We simulate a Sedov-Taylor blast wave \citep{Sedov_1959,Taylor_1946} in a 3D periodic box on the domain $[-1, 1]$ in each coordinate direction. The initial conditions consist of a spherical region of high pressure $P = 10$ for radii $r < 0.1$ and low pressure $P = 0.1$ for $r \ge 0.1$, with a uniform density of $\rho = 1$ and zero velocity, for an ideal gas with adiabatic index $\gamma = 5/3$.
paper/quokka.tex:We run with a varying number of GPUs with two $256^3$ grids per GPU, increasing the resolution of our simulation as we extend to greater numbers of GPUs. However, a power-of-two resolution increase does not easily map onto a jump from one GPU to four GPUs, so the single-GPU simulation only uses a grid size of $256^3$. The grid size of the simulations therefore ranges from $256^3$ (for 1 GPU) to $2048^3$ (for 256 GPUs). We set the \textsc{AMReX} domain decomposition parameters \texttt{blocking\_factor} and \texttt{max\_grid\_size} to a value of $128$, leading the computational grid to be decomposed into arrays of size $128^3$. (We also tested local grid sizes of $256^3$ but found only a few per cent performance improvement on this problem.) We use one MPI rank per GPU for all simulations. The CFL number is $0.25$ and we evolve for $100$ timesteps for each simulation. We assess performance by counting the total number of cell-updates and dividing by the number of GPUs in order to obtain the performance figure-of-merit in the units of 1 million cells (or zones) per timestep per GPU per second (Mzones/GPU/s).  We report the results in \autoref{table:weak_hydro_scaling}.
paper/quokka.tex:We find a $\approx 40\%$ drop in performance per GPU when going from 1 GPU to 4 GPUs, corresponding to using all 4 GPUs on a single node of the compute cluster. We hypothesize that this is due to the limited communication bandwidth between GPUs on a node. For intra-node scaling on CPUs, \citet{Stone_2020} report a similar decrease in performance when going from one CPU to all the CPUs on a node for \textsc{Athena++}, which they attribute to limitations of memory bandwidth. However, significantly different scaling behavior is observed when running the \textsc{K-Athena} hydrodynamics code on GPUs \citep{Grete_2019} on the Summit supercomputer\footnote{\url{https://www.olcf.ornl.gov/olcf-resources/compute-systems/summit/}}, finding a $99$ per cent weak scaling efficiency going from 1 GPU to 6 GPUs on a single node, so there may be some inefficiency in our current GPU-to-GPU communication method. We find that using CUDA-aware MPI does not improve performance for our code. However, we observe only a modest drop in performance per GPU when going from 1 node (4 GPUs) to 64 nodes (256 GPUs), yielding a parallel efficiency of 83 per cent on 64 nodes when compared to running on 1 node.  We could not run on larger numbers of GPU nodes due to job size limitations, but we expect scaling to continue to thousands of GPUs based on the parallel scaling observed for other GPU hydrodynamics codes based on \textsc{AMReX}, such as \textsc{Castro} \citep{Almgren_2020}.
paper/quokka.tex:In \autoref{table:weak_hydro_scaling_plm}, we show the same performance numbers as in \autoref{table:weak_hydro_scaling}, but using PLM reconstruction for each simulation instead of PPM reconstruction. We find that the performance improves significantly on a single GPU, going from $113$ million zone-updates per second to $158$ million zone-updates per second. However, communication overheads limit the relative performance improvement when using large numbers of nodes, as the $64$-node case goes from $59$ million zone-updates per GPU per second using PPM to only $65$ million zone-updates per GPU per second using PLM. Since the computations on each local grid are less expensive with PLM but the communication costs remain the same, the scaling efficiency decreases slightly as well, from $83$ per cent to $76$ per cent.
paper/quokka.tex:        Nodes   & GPUs  & Mzones/GPU/s  & Scaling efficiency (\%) & Grid size                         \\\hline
paper/quokka.tex:        {weak_scaling_hydro.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\mzonespergpuideal,6=\gpufill,7=\scaling,8=\scalingnode,9=\size}
paper/quokka.tex:        {\nodes & \gpus & \mzonespergpu & \scalingnode       & $\size$ \\}
paper/quokka.tex:    \caption{Weak scaling efficiency for hydrodynamics with PPM reconstruction as a function of the number of GPUs for a Sedov blast wave with periodic boundary conditions.}
paper/quokka.tex:        Nodes   & GPUs  & Mzones/GPU/s  & Scaling efficiency (\%) & Grid size                         \\\hline
paper/quokka.tex:        {weak_scaling_hydro_plm.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\scaling,6=\scalingnode,7=\size}
paper/quokka.tex:        {\nodes & \gpus & \mzonespergpu & \scalingnode       & $\size$ \\}
paper/quokka.tex:    \caption{Weak scaling efficiency for hydrodynamics with PLM reconstruction as a function of the number of GPUs for a Sedov blast wave with periodic boundary conditions.}
paper/quokka.tex:We next test the scaling behaviour for full radiation hydrodynamics solver on uniform grids. \autoref{table:weak_radhydro_scaling} lists the performance per GPU and parallel efficiency measured with respect to single-node performance for the radiation-driven shell test problem run for $50$ timesteps. Since we have many radiation substeps per hydrodynamic step (set here to 10; see \autoref{sssec:sync}), the performance metric in units of Mzones/GPU/s is lower by a factor comparable to but somewhat smaller than the number of radiation substeps per hydro step; a single radiation update is slightly less costly than a single hydrodynamic update. In this case, we observe a steeper drop in performance when going from 1 GPU to 4 GPUs (approximately a factor of 2). The lower parallel efficiency is not surprising, since \emph{each} radiation substep requires communicating boundary conditions between grids, so the amount of inter-GPU communication per hydro timestep increases significantly for radiation hydrodynamics. Nonetheless, as is the case for hydrodynamics, there is little additional performance penalty when scaling from 1 node to 64 nodes. We measure a parallel efficiency in this case of $76$ per cent.
paper/quokka.tex:in \autoref{table:scaling_a100}, we list the performance metrics of the code on both the Sedov problem and the radiation-driven shell problem running on a compute node with newer NVIDIA A100 GPUs. Since we only have access to a limited number of these GPUs, we only show performance data for a single GPU and a single compute node (4 GPUs). The single GPU case achieves $254$ million hydrodynamic zone-updates per second using PPM, making Quokka, as far as we are aware, the fastest PPM hydrodynamics code that currently exists. On the radiation-driven shell problem, the code achieves $39$ million radiation hydrodynamic zone-updates per second. In both cases, the performance per GPU drops by a factor of approximately 2 when using all 4 GPUs on the node. This is almost entirely due to the time spent communicating boundary conditions, as shown in the table (`B.C. fill time', which denotes the percentage of total wall time spent filling ghost cells for each local grid). If communication of boundary data and computation over the local grids could be perfectly overlapped, the parallel efficiency going from 1 GPU to 4 GPUs would be $> 99$ per cent for hydrodynamics and $> 96$ for radiation hydrodynamics.
paper/quokka.tex:Finally, we point out that absolute speed of \quokka~is excellent. Comparison between CPU and GPU codes is non-trivial, since it obviously depends on the CPU-to-GPU ratio on a particular compute platform. However, it is worth pointing out that \quokka's update rate per core (normalised by the number of CPU cores per compute node) for \textit{radiation}-hydrodynamics on GPU is comparable to or better than \textsc{Athena++}'s for \textit{hydrodynamics} on CPU.
paper/quokka.tex:        Nodes   & GPUs  & Mzones/GPU/s  & Scaling efficiency (\%) & Grid size                         \\\hline
paper/quokka.tex:        {weak_scaling_radhydro.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\mzonespergpuideal,6=\gpufill,7=\scaling,8=\scalingnode,9=\size}
paper/quokka.tex:        {\nodes & \gpus & \mzonespergpu & \scalingnode       & $\size$ \\}
paper/quokka.tex:    \caption{Weak scaling efficiency for radiation hydrodynamics as a function of the number of GPUs for the radiation-driven shell test (\autoref{section:shell}) with periodic boundary conditions.}
paper/quokka.tex:        GPUs  & Mzones/GPU/s  & B.C. fill time (\%) & Grid size & Problem                         \\\hline
paper/quokka.tex:        {performance_a100.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\mzonespergpuideal,6=\gpufill,7=\scaling,8=\scalingnode,9=\size,10=\problemtype}
paper/quokka.tex:        {\gpus & \mzonespergpu & \gpufill & $\size$       & \problemtype \\}
paper/quokka.tex:    \caption{Performance for both Sedov and radiating shell tests as a function of the number of GPUs for a single node with 4 NVIDIA A100 GPUs.}
paper/quokka.tex:Many applications of interest will seek to minimize either the total runtime of the simulation or the total node-hours used for a simulation for a problem of a fixed size. Additionally, most applications we are interested in will benefit from or require the use of AMR.  We therefore test the ability of \textsc{Quokka} to scale an AMR radiation hydrodynamic simulation of fixed size to larger numbers of GPUs in order to either minimize total runtime or total node-hours (referred to as \emph{strong scaling}). For this test, we initialize the radiation-driven shell problem (\autoref{section:shell}) on a base grid of $256^3$ cells with two levels of mesh refinement based on the relative gradient in the gas density. We run each simulation for 50 timesteps, with a CFL number of $0.3$ and PLM reconstruction for both hydrodynamics and radiation. We set the \textsc{AMReX} domain decomposition parameters \texttt{blocking\_factor} set to 32 and \texttt{max\_grid\_size} set to a value of $128$, so that all grids are between $32^3$ and $128^3$ in size, with possible non-cubic grids at intermediate sizes. The number of GPUs used for each simulation is varied, scaling from 1 node (4 GPUs) to 8 nodes (32 GPUs). This is a particularly stringest test, since the level-by-level AMR timestepping requires that each level be computed separately, limiting the amount of parallelism that can be distributed across GPUs. There is also additional communication overhead when AMR is enabled compared to a single-level uniform grid simulation. We show the scaling results in \autoref{table:strong_scaling}. Comparing \autoref{table:weak_radhydro_scaling} and \autoref{table:strong_scaling}, the performance per GPU for a single node is lower than that of a uniform grid simulation by $\approx 50$ per cent. (A similar, although somewhat smaller, overhead when enabling AMR is also observed with CPU codes, e.g., \textsc{Athena++}; \citealt{Stone_2020}). The scaling efficiency is reasonable for 2 and 4 nodes (66 per cent for 4 nodes), but drops significantly at 8 nodes to 53 per cent parallel efficiency. We hypothesize that this is due to the small number of cell-updates per GPU once 32 GPUs are in use for this problem (approximately $211^3 \approx 9.4 \times 10^6$ cells/GPU). We find that performance on a single GPU is significantly diminished for uniform-grid problems smaller than $256^3$, so this performance drop may be largely due to the inability to use all GPU hardware threads when the amount of work per GPU is small. Similar GPU performance behavior is observed when running \textsc{K-Athena} on GPUs for varying problem sizes per GPU \citep{Grete_2019}. This effect is also magnified by the sequential nature of the level-by-level timestepping. For level $l=1$, the number of cells per GPU drops below $256^3$ for 8 GPUs, and for level $l=2$, it drops below $256^3$ for 16 GPUs. High scaling efficiency is obtained before reaching these thresholds, so it appears that reasonable performance on GPUs may be obtained with AMR when all refinement levels have at least $256^3$ cells per GPU on average. In general, obtaining the best possible GPU performance may require an adjustment to the mesh refinement parameters usually used when running on CPUs. For self-gravitating problems, scaling may be aided by the self-similar nature of gravitational collapse, leading to an approximately equal number of cells on each refinement level for appropriate refinement criteria (see discussion in \citealt{Stone_2020}).
paper/quokka.tex:        Nodes   & GPUs  & Mzones/GPU/s  & $\left<\frac{\text{Cells}}{\text{GPU}}\right>$ & \begin{tabular}{@{}r@{}}Scaling \\ efficiency\end{tabular} & Speedup                               \\\hline
paper/quokka.tex:        {strong_scaling.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\mzonespergpuideal,6=\gpufill,7=\scaling,8=\cellspergpu,9=\speedup,10=\cellszero,11=\cellsone,12=\cellstwo}
paper/quokka.tex:        {\nodes & \gpus & \mzonespergpu & ${\cellspergpu}^3$                             & \scaling                   & {\speedup}x \\}
paper/quokka.tex:        Nodes   & GPUs  & $\left<\frac{\text{Cells}}{\text{GPU}}\right>_{l=0}$ & $\left<\frac{\text{Cells}}{\text{GPU}}\right>_{l=1}$ & $\left<\frac{\text{Cells}}{\text{GPU}}\right>_{l=2}$ \\\hline
paper/quokka.tex:        {strong_scaling.csv}{1=\nodes,2=\gpus,3=\mzones,4=\mzonespergpu,5=\mzonespergpuideal,6=\gpufill,7=\scaling,8=\cellspergpu,9=\speedup,10=\cellszero,11=\cellsone,12=\cellstwo}
paper/quokka.tex:        {\nodes & \gpus & ${\cellszero}^3$                                     & ${\cellsone}^3$                                      & ${\cellstwo}^3$ \\}
paper/quokka.tex:    \caption{Strong scaling efficiency for radiation hydrodynamics as a function of the number of GPUs for the radiation-driven shell test (\autoref{section:shell}) with periodic boundary conditions on a base grid of $256^3$ cells and 2 levels of refinement. The number of cells per GPU is computed as an average over all timesteps.}
paper/quokka.tex:In the near future, we plan to add support for self-gravity with \textsc{AMReX}'s geometric multigrid solver for GPUs \citep{AMReX_JOSS}, sink and star particles for star cluster simulations (e.g., \citealt{Krumholz04a, Offner09a}), and optically-thin line cooling for the interstellar medium. These additions will enable simulations of the interstellar medium, galactic winds, and star clusters, among others.
paper/quokka.tex:Radiation hydrodynamics codes like ours will enable the widespread use of more accurate radiation transport methods and an ever-greater dynamic range in both space and time. As we approach the era of exascale supercomputers, we see a bright future for AMR radiation hydrodynamics on GPU architectures.
paper/abstract.txt:We present Quokka, a new subcycling-in-time, block-structured adaptive mesh refinement (AMR) radiation hydrodynamics code optimised for graphics processing units (GPUs). Quokka solves the equations of hydrodynamics with the piecewise parabolic method (PPM) in a method-of-lines formulation, and handles radiative transfer via the variable Eddington tensor (VET) radiation moment equations with a local closure. We use the AMReX library to handle the adaptive mesh management. In order to maximise GPU performance, we combine explicit-in-time evolution of the radiation moment equations with the reduced speed-of-light approximation. We show results for a wide range of test problems for hydrodynamics, radiation, and coupled radiation hydrodynamics. On uniform grids in 3D on a single GPU, our code achieves > 250 million hydrodynamic updates per second and almost 40 million radiation hydrodynamic updates per second. For radiation hydrodynamics problems on uniform grids in 3D, our code scales from 4 GPUs to 256 GPUs with an efficiency of 76 per cent. The code is publicly released under an open-source license on GitHub.
paper/quokka.bib:  title         = {{GAMER-2: a GPU-accelerated adaptive mesh refinement code - accuracy, performance, and scalability}},
CMakeLists.txt:option(DISABLE_FMAD "Disable fused multiply-add instructions on GPU (on/off)" ON)
CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
CMakeLists.txt:  enable_language(CUDA)
CMakeLists.txt:  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.7)
CMakeLists.txt:    message(FATAL_ERROR "You must use CUDA version 11.7 or newer to compile Quokka. All previous CUDA versions have compiler bugs that cause Quokka to crash.")
CMakeLists.txt:  set(CMAKE_CUDA_ARCHITECTURES 70 80 CACHE STRING "")
CMakeLists.txt:    include(AMReX_SetupCUDA)
CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
scripts/gpu_wrapper_sedov_1024.sh:#hpcrun -e gpu=nvidia --trace ./build/src/test_hydro3d_blast tests/blast_unigrid_1024.in
scripts/gpu_wrapper_sedov_2048.sh:#hpcrun -e gpu=nvidia --trace ./build/src/test_hydro3d_blast tests/blast_unigrid_2048.in
scripts/gpu_wrapper_amr_shell_stream.sh:./build/src/RadhydroShell/test_radhydro3d_shell tests/radhydro_shell_amr.in amrex.max_gpu_streams=1
scripts/summit-64node.bsub:#   https://docs.olcf.ornl.gov/systems/summit_user_guide.html#cuda-aware-mpi
scripts/summit-64node.bsub:# GPU-aware MPI does NOT work on Summit!! You MUST disable it by adding: amrex.use_gpu_aware_mpi=0
scripts/summit-64node.bsub:jsrun -r 6 -a 1 -g 1 -c 7 -l GPU-CPU -d packed -b rs build_summit/src/HydroBlast3D/test_hydro3d_blast tests/benchmark_unigrid_1024.in amrex.use_gpu_aware_mpi=0
scripts/shell-1gpu.pbs:#PBS -N quokka_GPU_1gpu
scripts/shell-1gpu.pbs:#PBS -q gpuvolta
scripts/shell-1gpu.pbs:#PBS -l ngpus=4
scripts/shell-1gpu.pbs:# (only use 1 GPU -- for scaling tests only)
scripts/shell-1gpu.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_shell_256.sh > shell_1gpu.log
scripts/cpu-sedov-64nodes.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_sedov_2048.sh > sedov_64nodes.log
scripts/amr-shell-1node-1stream.pbs:#PBS -N quokka_GPU_1node_1stream
scripts/amr-shell-1node-1stream.pbs:#PBS -q gpuvolta
scripts/amr-shell-1node-1stream.pbs:#PBS -l ngpus=4
scripts/amr-shell-1node-1stream.pbs:MPI_OPTIONS="-np $PBS_NGPUS --map-by numa:SPAN --bind-to numa --mca pml ucx"
scripts/amr-shell-1node-1stream.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_amr_shell_stream.sh > amr_1node_1stream.log
scripts/amr-shell-8nodes.pbs:#PBS -N quokka_GPU_8nodes
scripts/amr-shell-8nodes.pbs:#PBS -q gpuvolta
scripts/amr-shell-8nodes.pbs:#PBS -l ngpus=32
scripts/amr-shell-8nodes.pbs:MPI_OPTIONS="-np $PBS_NGPUS --map-by numa:SPAN --bind-to numa --mca pml ucx"
scripts/amr-shell-8nodes.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_amr_shell.sh > amr_8nodes.log
scripts/setonix-1node.submit:#SBATCH -A pawsey0807-gpu
scripts/setonix-1node.submit:#SBATCH -p gpu
scripts/setonix-1node.submit:#SBATCH --gpus-per-node=8
scripts/setonix-1node.submit:# always run with GPU-aware MPI
scripts/setonix-1node.submit:export MPICH_GPU_SUPPORT_ENABLED=1
scripts/setonix-1node.submit:# use correct NIC-to-GPU binding
scripts/setonix-1node.submit:      0) GPU=4;;
scripts/setonix-1node.submit:      1) GPU=5;;
scripts/setonix-1node.submit:      2) GPU=2;;
scripts/setonix-1node.submit:      3) GPU=3;;
scripts/setonix-1node.submit:      4) GPU=6;;
scripts/setonix-1node.submit:      5) GPU=7;;
scripts/setonix-1node.submit:      6) GPU=0;;
scripts/setonix-1node.submit:      7) GPU=1;;
scripts/setonix-1node.submit:    export ROCR_VISIBLE_DEVICES=\$((GPU));
scripts/setonix-gpu.profile:# NOTE: CCE and ROCm versions must match according to this table:
scripts/setonix-gpu.profile:#  https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#compatible-compiler-rocm-toolchain-versions
scripts/setonix-gpu.profile:module load rocm/5.5.3 # matches cce/16 clang version
scripts/setonix-gpu.profile:# GPU-aware MPI
scripts/setonix-gpu.profile:export MPICH_GPU_SUPPORT_ENABLED=1
scripts/setonix-gpu.profile:# optimize ROCm/HIP compilation for MI250X
scripts/setonix-gpu.profile:export CFLAGS="-I${ROCM_PATH}/include"
scripts/setonix-gpu.profile:export CXXFLAGS="-I${ROCM_PATH}/include -mllvm -amdgpu-function-calls=true"
scripts/sedov-1node.pbs:#PBS -N quokka_GPU_1node
scripts/sedov-1node.pbs:#PBS -q gpuvolta
scripts/sedov-1node.pbs:#PBS -l ngpus=4
scripts/sedov-1node.pbs:MPI_OPTIONS="-np $PBS_NGPUS --map-by numa:SPAN --bind-to numa --mca pml ucx"
scripts/sedov-1node.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_sedov_512.sh > sedov_1node.log
scripts/amr-shell-2nodes.pbs:#PBS -N quokka_GPU_2nodes
scripts/amr-shell-2nodes.pbs:#PBS -q gpuvolta
scripts/amr-shell-2nodes.pbs:#PBS -l ngpus=8
scripts/amr-shell-2nodes.pbs:MPI_OPTIONS="-np $PBS_NGPUS --map-by numa:SPAN --bind-to numa --mca pml ucx"
scripts/amr-shell-2nodes.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_amr_shell.sh > amr_2nodes.log
scripts/setonix-8nodes.submit:#SBATCH -A pawsey0807-gpu
scripts/setonix-8nodes.submit:#SBATCH -p gpu
scripts/setonix-8nodes.submit:#SBATCH --gpus-per-node=8
scripts/setonix-8nodes.submit:# always run with GPU-aware MPI
scripts/setonix-8nodes.submit:export MPICH_GPU_SUPPORT_ENABLED=1
scripts/setonix-8nodes.submit:# use correct NIC-to-GPU binding
scripts/setonix-8nodes.submit:      0) GPU=4;;
scripts/setonix-8nodes.submit:      1) GPU=5;;
scripts/setonix-8nodes.submit:      2) GPU=2;;
scripts/setonix-8nodes.submit:      3) GPU=3;;
scripts/setonix-8nodes.submit:      4) GPU=6;;
scripts/setonix-8nodes.submit:      5) GPU=7;;
scripts/setonix-8nodes.submit:      6) GPU=0;;
scripts/setonix-8nodes.submit:      7) GPU=1;;
scripts/setonix-8nodes.submit:    export ROCR_VISIBLE_DEVICES=\$((GPU));
scripts/setonix-64nodes.submit:#SBATCH -A pawsey0807-gpu
scripts/setonix-64nodes.submit:#SBATCH -p gpu
scripts/setonix-64nodes.submit:#SBATCH --gpus-per-node=8
scripts/setonix-64nodes.submit:# always run with GPU-aware MPI
scripts/setonix-64nodes.submit:export MPICH_GPU_SUPPORT_ENABLED=1
scripts/setonix-64nodes.submit:# use correct NIC-to-GPU binding
scripts/setonix-64nodes.submit:      0) GPU=4;;
scripts/setonix-64nodes.submit:      1) GPU=5;;
scripts/setonix-64nodes.submit:      2) GPU=2;;
scripts/setonix-64nodes.submit:      3) GPU=3;;
scripts/setonix-64nodes.submit:      4) GPU=6;;
scripts/setonix-64nodes.submit:      5) GPU=7;;
scripts/setonix-64nodes.submit:      6) GPU=0;;
scripts/setonix-64nodes.submit:      7) GPU=1;;
scripts/setonix-64nodes.submit:    export ROCR_VISIBLE_DEVICES=\$((GPU));
scripts/mi100-1gpu.submit:#SBATCH --partition=gpuMI100x8
scripts/mi100-1gpu.submit:#SBATCH --account=cvz-delta-gpu
scripts/mi100-1gpu.submit:#SBATCH --gpus-per-task=1
scripts/mi100-1gpu.submit:#SBATCH --gpu-bind=closest,1
scripts/summit.profile:module load cuda/11.7.1
scripts/summit.profile:export Ascent_DIR=/sw/summit/ums/ums010/ascent/0.8.0_warpx/summit/cuda/gnu/ascent-install/
scripts/summit.profile:# optimize CUDA compilation for V100
scripts/summit.profile:export AMREX_CUDA_ARCH=7.0
scripts/summit.profile:export CUDACXX=$(which nvcc)
scripts/summit.profile:export CUDAHOSTCXX=$(which g++)
scripts/blast_1gpu.submit:#SBATCH --partition=gpuA100x4
scripts/blast_1gpu.submit:#SBATCH --account=cvz-delta-gpu
scripts/blast_1gpu.submit:#SBATCH --gpus-per-task=1
scripts/blast_1gpu.submit:#SBATCH --gpu-bind=none
scripts/blast_1gpu.submit:module load cuda/11.7.0
scripts/blast_1gpu.submit:GPU_AWARE_MPI=""
scripts/blast_1gpu.submit:nvidia-smi topo -m
scripts/blast_1gpu.submit:# 	GPU0	GPU1	GPU2	GPU3	mlx5_0	CPU Affinity	NUMA Affinity
scripts/blast_1gpu.submit:# GPU0	 X 	NV4	NV4	NV4	SYS	48-63	3
scripts/blast_1gpu.submit:# GPU1	NV4	 X 	NV4	NV4	SYS	32-47	2
scripts/blast_1gpu.submit:# GPU2	NV4	NV4	 X 	NV4	SYS	16-31	1
scripts/blast_1gpu.submit:# GPU3	NV4	NV4	NV4	 X 	PHB	0-15	0
scripts/blast_1gpu.submit:    export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
scripts/blast_1gpu.submit:    ${EXE} ${INPUTS} ${GPU_AWARE_MPI}"
scripts/shell-64nodes.pbs:#PBS -N quokka_GPU_64nodes
scripts/shell-64nodes.pbs:#PBS -q gpuvolta
scripts/shell-64nodes.pbs:#PBS -l ngpus=256
scripts/shell-64nodes.pbs:MPI_OPTIONS="-np $PBS_NGPUS --map-by numa:SPAN --bind-to numa --mca pml ucx -x HCOLL_ENABLE_MCAST=0"
scripts/shell-64nodes.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_shell_2048.sh > shell_64nodes.log
scripts/sedov-8nodes-mpich.pbs:#PBS -N quokka_GPU_8nodes
scripts/sedov-8nodes-mpich.pbs:#PBS -q gpuvolta
scripts/sedov-8nodes-mpich.pbs:#PBS -l ngpus=32
scripts/sedov-8nodes-mpich.pbs:#export MPICH_RDMA_ENABLED_CUDA=1
scripts/sedov-8nodes-mpich.pbs:#export MPICH_GPU_SUPPORT_ENABLED=1
scripts/sedov-8nodes-mpich.pbs:MPI_OPTIONS="-np $PBS_NGPUS -bind-to numa -gpus-per-proc=1"
scripts/sedov-8nodes-mpich.pbs:mpiexec $MPI_OPTIONS ./build/src/HydroBlast3D/test_hydro3d_blast tests/blast_unigrid_1024.in amrex.async_out=1 amrex.use_gpu_aware_mpi=0 > mpich_8nodes.log
scripts/moth.profile:module load rocm/6.0.0
scripts/moth.profile:# optimize ROCm/HIP compilation for MI210
scripts/moth.profile:export CFLAGS="-I${ROCM_PATH}/include"
scripts/moth.profile:export CXXFLAGS="-I${ROCM_PATH}/include"
scripts/moth.profile:export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64"
scripts/summit-1node.bsub:#   https://docs.olcf.ornl.gov/systems/summit_user_guide.html#cuda-aware-mpi
scripts/summit-1node.bsub:# GPU-aware MPI does NOT work on Summit!! You MUST disable it by adding: amrex.use_gpu_aware_mpi=0
scripts/summit-1node.bsub:jsrun -r 6 -a 1 -g 1 -c 7 -l GPU-CPU -d packed -b rs build_summit/src/HydroBlast3D/test_hydro3d_blast tests/benchmark_unigrid_256.in amrex.use_gpu_aware_mpi=0
scripts/shell-1node.pbs:#PBS -N quokka_GPU_1node
scripts/shell-1node.pbs:#PBS -q gpuvolta
scripts/shell-1node.pbs:#PBS -l ngpus=4
scripts/shell-1node.pbs:MPI_OPTIONS="-np $PBS_NGPUS --map-by numa:SPAN --bind-to numa --mca pml ucx"
scripts/shell-1node.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_shell_512.sh > shell_1node.log
scripts/amr-shell-4nodes.pbs:#PBS -N quokka_GPU_4nodes
scripts/amr-shell-4nodes.pbs:#PBS -q gpuvolta
scripts/amr-shell-4nodes.pbs:#PBS -l ngpus=16
scripts/amr-shell-4nodes.pbs:MPI_OPTIONS="-np $PBS_NGPUS --map-by numa:SPAN --bind-to numa --mca pml ucx"
scripts/amr-shell-4nodes.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_amr_shell.sh > amr_4nodes.log
scripts/blast_1node.submit:#SBATCH --partition=gpuA100x4
scripts/blast_1node.submit:#SBATCH --account=cvz-delta-gpu
scripts/blast_1node.submit:#SBATCH --gpus-per-task=1
scripts/blast_1node.submit:#SBATCH --gpu-bind=none
scripts/blast_1node.submit:module load cuda/11.7.0
scripts/blast_1node.submit:GPU_AWARE_MPI=""
scripts/blast_1node.submit:nvidia-smi topo -m
scripts/blast_1node.submit:# 	GPU0	GPU1	GPU2	GPU3	mlx5_0	CPU Affinity	NUMA Affinity
scripts/blast_1node.submit:# GPU0	 X 	NV4	NV4	NV4	SYS	48-63	3
scripts/blast_1node.submit:# GPU1	NV4	 X 	NV4	NV4	SYS	32-47	2
scripts/blast_1node.submit:# GPU2	NV4	NV4	 X 	NV4	SYS	16-31	1
scripts/blast_1node.submit:# GPU3	NV4	NV4	NV4	 X 	PHB	0-15	0
scripts/blast_1node.submit:    export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
scripts/blast_1node.submit:    ${EXE} ${INPUTS} ${GPU_AWARE_MPI}"
scripts/moth-sanitizer.profile:module load rocm/6.0.0
scripts/moth-sanitizer.profile:## for GPU ASAN on MI210 GPUs:
scripts/moth-sanitizer.profile:export LD_LIBRARY_PATH=/opt/rocm-6.0.0/llvm/lib/clang/17.0.0/lib/linux:$LD_LIBRARY_PATH
scripts/moth-sanitizer.profile:export CFLAGS="-I${ROCM_PATH}/include"
scripts/moth-sanitizer.profile:export CXXFLAGS="-fsanitize=address -shared-libsan -g -I${ROCM_PATH}/include"
scripts/moth-sanitizer.profile:export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64"
scripts/shell-8nodes.pbs:#PBS -N quokka_GPU_8nodes
scripts/shell-8nodes.pbs:#PBS -q gpuvolta
scripts/shell-8nodes.pbs:#PBS -l ngpus=32
scripts/shell-8nodes.pbs:MPI_OPTIONS="-np $PBS_NGPUS --map-by numa:SPAN --bind-to numa --mca pml ucx"
scripts/shell-8nodes.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_shell_1024.sh > shell_8nodes.log
scripts/summit-512node.bsub:#   https://docs.olcf.ornl.gov/systems/summit_user_guide.html#cuda-aware-mpi
scripts/summit-512node.bsub:# GPU-aware MPI does NOT work on Summit!! You MUST disable it by adding: amrex.use_gpu_aware_mpi=0
scripts/summit-512node.bsub:jsrun -r 6 -a 1 -g 1 -c 7 -l GPU-CPU -d packed -b rs build_summit/src/HydroBlast3D/test_hydro3d_blast tests/benchmark_unigrid_2048.in amrex.use_gpu_aware_mpi=0
scripts/sedov-64nodes-mpich.pbs:#PBS -N quokka_GPU_64nodes
scripts/sedov-64nodes-mpich.pbs:#PBS -q gpuvolta
scripts/sedov-64nodes-mpich.pbs:#PBS -l ngpus=256
scripts/sedov-64nodes-mpich.pbs:#export MPICH_RDMA_ENABLED_CUDA=1
scripts/sedov-64nodes-mpich.pbs:#export MPICH_GPU_SUPPORT_ENABLED=1
scripts/sedov-64nodes-mpich.pbs:MPI_OPTIONS="-np $PBS_NGPUS -bind-to numa -gpus-per-proc=1"
scripts/sedov-64nodes-mpich.pbs:mpiexec $MPI_OPTIONS ./build/src/HydroBlast3D/test_hydro3d_blast tests/blast_unigrid_2048.in amrex.async_out=1 amrex.use_gpu_aware_mpi=0 > mpich_64nodes.log
scripts/sedov-1gpu.pbs:#PBS -N quokka_GPU_1gpu
scripts/sedov-1gpu.pbs:#PBS -q gpuvolta
scripts/sedov-1gpu.pbs:#PBS -l ngpus=4
scripts/sedov-1gpu.pbs:# (only use 1 GPU -- for scaling tests only)
scripts/sedov-1gpu.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_sedov_256.sh > sedov_1gpu.log
scripts/sedov-64nodes.pbs:#PBS -N quokka_GPU_64nodes
scripts/sedov-64nodes.pbs:#PBS -q gpuvolta
scripts/sedov-64nodes.pbs:#PBS -l ngpus=256
scripts/sedov-64nodes.pbs:MPI_OPTIONS="-np $PBS_NGPUS --map-by numa:SPAN --bind-to numa --mca pml ucx -x HCOLL_ENABLE_MCAST=0"
scripts/sedov-64nodes.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_sedov_2048.sh > sedov_64nodes.log
scripts/summit-4096node.bsub:#   https://docs.olcf.ornl.gov/systems/summit_user_guide.html#cuda-aware-mpi
scripts/summit-4096node.bsub:# GPU-aware MPI does NOT work on Summit!! You MUST disable it by adding: amrex.use_gpu_aware_mpi=0
scripts/summit-4096node.bsub:jsrun -r 6 -a 1 -g 1 -c 7 -l GPU-CPU -d packed -b rs build_summit/src/HydroBlast3D/test_hydro3d_blast tests/benchmark_unigrid_4096.in amrex.use_gpu_aware_mpi=0
scripts/amr-shell-1node.pbs:#PBS -N quokka_GPU_1node
scripts/amr-shell-1node.pbs:#PBS -q gpuvolta
scripts/amr-shell-1node.pbs:#PBS -l ngpus=4
scripts/amr-shell-1node.pbs:MPI_OPTIONS="-np $PBS_NGPUS --map-by numa:SPAN --bind-to numa --mca pml ucx"
scripts/amr-shell-1node.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_amr_shell.sh > amr_1node.log
scripts/sedov-8nodes.pbs:#PBS -N quokka_GPU_8nodes
scripts/sedov-8nodes.pbs:#PBS -q gpuvolta
scripts/sedov-8nodes.pbs:#PBS -l ngpus=32
scripts/sedov-8nodes.pbs:MPI_OPTIONS="-np $PBS_NGPUS --map-by numa:SPAN --bind-to numa --mca pml ucx"
scripts/sedov-8nodes.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_sedov_1024.sh > sedov_8nodes.log
scripts/mi100-1node.submit:#SBATCH --partition=gpuMI100x8
scripts/mi100-1node.submit:#SBATCH --account=cvz-delta-gpu
scripts/mi100-1node.submit:#SBATCH --gpus-per-task=1
scripts/mi100-1node.submit:#SBATCH --gpu-bind=closest,1
scripts/sedov-1node-mpich.pbs:#PBS -N quokka_GPU_1node
scripts/sedov-1node-mpich.pbs:#PBS -q gpuvolta
scripts/sedov-1node-mpich.pbs:#PBS -l ngpus=4
scripts/sedov-1node-mpich.pbs:#export MPICH_RDMA_ENABLED_CUDA=1
scripts/sedov-1node-mpich.pbs:#export MPICH_GPU_SUPPORT_ENABLED=1
scripts/sedov-1node-mpich.pbs:MPI_OPTIONS="-np $PBS_NGPUS -bind-to numa -gpus-per-proc=1"
scripts/sedov-1node-mpich.pbs:mpiexec $MPI_OPTIONS ./build/src/HydroBlast3D/test_hydro3d_blast tests/blast_unigrid_512.in amrex.async_out=1 amrex.use_gpu_aware_mpi=0 > mpich_1node.log
scripts/summit-8node.bsub:#   https://docs.olcf.ornl.gov/systems/summit_user_guide.html#cuda-aware-mpi
scripts/summit-8node.bsub:# GPU-aware MPI does NOT work on Summit!! You MUST disable it by adding: amrex.use_gpu_aware_mpi=0
scripts/summit-8node.bsub:jsrun -r 6 -a 1 -g 1 -c 7 -l GPU-CPU -d packed -b rs build_summit/src/HydroBlast3D/test_hydro3d_blast tests/benchmark_unigrid_512.in amrex.use_gpu_aware_mpi=0
scripts/crusher-1node.submit:#SBATCH --gpus-per-task=1
scripts/crusher-1node.submit:#SBATCH --gpu-bind=closest
scripts/crusher-1node.submit:# module load cpe/22.08 craype-accel-amd-gfx90a rocm/5.2.0 cray-mpich cce/14.0.2 cray-hdf5
scripts/crusher-1node.submit:# always run with GPU-aware MPI
scripts/crusher-1node.submit:export MPICH_GPU_SUPPORT_ENABLED=1
scripts/crusher-1node.submit:# use correct NIC-to-GPU binding
scripts/cpu-sedov-8nodes.pbs:mpirun $MPI_OPTIONS ./scripts/gpu_wrapper_sedov_1024.sh > sedov_8nodes.log
cmake/avatar_nvhpc_cuda.cmake:set(CMAKE_CUDA_COMPILER "nvcc" CACHE PATH "")
cmake/avatar_nvhpc_cuda.cmake:set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
cmake/avatar_nvhpc_cuda.cmake:set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "")
cmake/gadi_oneapi_cuda.cmake:set(CMAKE_CUDA_COMPILER "nvcc" CACHE PATH "")
cmake/gadi_oneapi_cuda.cmake:set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
cmake/gadi_oneapi_cuda.cmake:set(CMAKE_CUDA_ARCHITECTURES 70 80 CACHE STRING "")
cmake/ubuntu_aocc_cuda.cmake:set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
cmake/delta_gcc_cuda.cmake:set(CMAKE_CUDA_COMPILER "nvcc" CACHE PATH "")
cmake/delta_gcc_cuda.cmake:set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
cmake/delta_gcc_cuda.cmake:set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "")
cmake/crusher.cmake:# module load cpe/22.08 cmake/3.23.2 craype-accel-amd-gfx90a rocm/5.2.0 cray-mpich cce/14.0.2 cray-hdf5
cmake/crusher.cmake:# (this must be set to use GPU-aware MPI)
cmake/crusher.cmake:# export MPICH_GPU_SUPPORT_ENABLED=1
cmake/crusher.cmake:set(AMReX_GPU_BACKEND HIP CACHE STRING "")
cmake/crusher.cmake:set(AMREX_GPUS_PER_NODE 8 CACHE STRING "")
cmake/gadi_gcc_cuda.cmake:set(CMAKE_CUDA_COMPILER "nvcc" CACHE PATH "")
cmake/gadi_gcc_cuda.cmake:set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
cmake/gadi_gcc_cuda.cmake:set(CMAKE_CUDA_ARCHITECTURES 70 80 CACHE STRING "")
cmake/gadi_gcc_cuda.cmake:set(AMREX_GPUS_PER_SOCKET 2 CACHE STRING "")
cmake/gadi_gcc_cuda.cmake:set(AMREX_GPUS_PER_NODE 4 CACHE STRING "")
cmake/avatar_llvm_cuda.cmake:set(CMAKE_CUDA_COMPILER "clang" CACHE PATH "")
cmake/avatar_llvm_cuda.cmake:set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
cmake/avatar_llvm_cuda.cmake:set(AMReX_CUDA_FASTMATH OFF CACHE BOOL "")
cmake/avatar_llvm_cuda.cmake:set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "")
cmake/avatar_oneapi_cuda.cmake:set(CMAKE_CUDA_COMPILER "nvcc" CACHE PATH "")
cmake/avatar_oneapi_cuda.cmake:set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
cmake/avatar_oneapi_cuda.cmake:set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "")
cmake/setonix.cmake:set(AMReX_GPU_BACKEND HIP CACHE STRING "")
cmake/gadi_llvm_cuda.cmake:set(CMAKE_CUDA_COMPILER "clang" CACHE PATH "")
cmake/gadi_llvm_cuda.cmake:set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
cmake/gadi_llvm_cuda.cmake:set(CMAKE_CUDA_ARCHITECTURES 70 80 CACHE STRING "")
cmake/avatar_gcc_cuda.cmake:##   Ascent_DIR=../../ascent/install cmake -C ../cmake/avatar_gcc_cuda.cmake .. -G Ninja
cmake/avatar_gcc_cuda.cmake:set(CMAKE_CUDA_COMPILER "nvcc" CACHE PATH "")
cmake/avatar_gcc_cuda.cmake:set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
cmake/avatar_gcc_cuda.cmake:set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "")
cmake/delta_gcc_hip.cmake:set(AMReX_GPU_BACKEND HIP CACHE STRING "")
cmake/delta_gcc_hip.cmake:set(AMREX_GPUS_PER_SOCKET 4 CACHE STRING "")
cmake/delta_gcc_hip.cmake:set(AMREX_GPUS_PER_NODE 8 CACHE STRING "")
cmake/summit.cmake:set(CMAKE_CUDA_COMPILER "nvcc" CACHE PATH "")
cmake/summit.cmake:set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
cmake/summit.cmake:set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "")
src/.clang-tidy:  - key:             cppcoreguidelines-non-private-member-variables-in-classes.IgnoreClassesWithAllMemberVariablesBeingPublic
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 4> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 4> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 4> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 4> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 4> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 4> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 5> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 5> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 5> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 5> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 5> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 5> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 8> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 8> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 8> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 8> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 8> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 8> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 10> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 10> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 10> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 10> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 10> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 10> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 13> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 13> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 13> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 13> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 13> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 13> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 15> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 15> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 15> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 15> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 15> const &
src/math/gauss.hpp:	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 15> const &
src/math/gauss.hpp:	template <class F> AMREX_GPU_DEVICE static auto integrate(F f, Real *pL1 = nullptr) -> decltype(f(Real(0.0)))
src/math/gauss.hpp:	template <class F> AMREX_GPU_DEVICE static auto integrate(F f, Real a, Real b, Real *pL1 = nullptr) -> decltype(f(Real(0.0)))
src/math/root_finding.hpp:#include "AMReX_GpuQualifiers.H"
src/math/root_finding.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE eps_tolerance() { eps = 4 * std::numeric_limits<T>::epsilon(); }
src/math/root_finding.hpp:	AMREX_GPU_HOST_DEVICE
src/math/root_finding.hpp:	AMREX_GPU_HOST_DEVICE
src/math/root_finding.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool operator()(const T &a, const T &b)
src/math/root_finding.hpp:template <class F, class T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void bracket(F f, T &a, T &b, T c, T &fa, T &fb, T &d, T &fd)
src/math/root_finding.hpp:template <class T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T safe_div(T num, T denom, T r)
src/math/root_finding.hpp:template <class T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T secant_interpolate(const T &a, const T &b, const T &fa, const T &fb)
src/math/root_finding.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T quadratic_interpolate(const T &a, const T &b, T const &d, const T &fa, const T &fb, T const &fd, unsigned count)
src/math/root_finding.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T cubic_interpolate(const T &a, const T &b, const T &d, const T &e, const T &fa, const T &fb, const T &fd, const T &fe)
src/math/root_finding.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE std::pair<T, T> toms748_solve(F f, const T &ax, const T &bx, const T &fax, const T &fbx, Tol tol, int &max_iter)
src/math/root_finding.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE std::pair<T, T> toms748_solve(F f, const T &ax, const T &bx, Tol tol, int &max_iter)
src/math/interpolate.hpp:#include "AMReX_GpuQualifiers.H"
src/math/interpolate.hpp:AMREX_GPU_HOST_DEVICE int64_t binary_search_with_guess(double key, const double *arr, int64_t len, int64_t guess);
src/math/interpolate.hpp:AMREX_GPU_HOST_DEVICE void interpolate_arrays(double *x, double *y, int len, double *arr_x, double *arr_y, int arr_len);
src/math/interpolate.hpp:AMREX_GPU_HOST_DEVICE double interpolate_value(double x, double const *arr_x, double const *arr_y, int arr_len);
src/math/quadrature.hpp:AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto kernel_wendland_c2(const amrex::Real r) -> amrex::Real
src/math/quadrature.hpp:AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto quad_3d(F &&f, amrex::Real x0, amrex::Real x1, amrex::Real y0, amrex::Real y1, amrex::Real z0, amrex::Real z1)
src/math/quadrature.hpp:	auto integrand = [=] AMREX_GPU_DEVICE(amrex::Real z) {
src/math/quadrature.hpp:		return quad_2d([=] AMREX_GPU_DEVICE(amrex::Real x, amrex::Real y) { return f(x, y, z); }, x0, x1, y0, y1);
src/math/quadrature.hpp:template <typename F> AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto quad_2d(F &&f, amrex::Real x0, amrex::Real x1, amrex::Real y0, amrex::Real y1) -> amrex::Real
src/math/quadrature.hpp:	auto integrand = [=] AMREX_GPU_DEVICE(amrex::Real y) { return quad_1d([=] AMREX_GPU_DEVICE(amrex::Real x) { return f(x, y); }, x0, x1); };
src/math/quadrature.hpp:template <typename F> AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto quad_1d(F &&f, amrex::Real x0, amrex::Real x1) -> amrex::Real
src/math/Interpolate2D.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate2d(double x, double y, amrex::Table1D<const double> const &xv, amrex::Table1D<const double> const &yv,
src/math/interpolate.cpp:AMREX_GPU_HOST_DEVICE
src/math/interpolate.cpp:AMREX_GPU_HOST_DEVICE
src/math/interpolate.cpp:AMREX_GPU_HOST_DEVICE
src/math/ODEIntegrate.hpp:#include "AMReX_GpuQualifiers.H"
src/math/ODEIntegrate.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto rk12_single_step(F &&rhs, Real t0, quokka::valarray<Real, N> const &y, Real dt, quokka::valarray<Real, N> &ynew,
src/math/ODEIntegrate.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto rk23_single_step(F &&rhs, Real t0, quokka::valarray<Real, N> const &y, Real dt, quokka::valarray<Real, N> &ynew,
src/math/ODEIntegrate.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto error_norm(quokka::valarray<Real, N> const &y0, quokka::valarray<Real, N> const &yerr, Real reltol,
src/math/ODEIntegrate.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void rk_adaptive_integrate(F &&rhs, Real t0, quokka::valarray<Real, N> &y0, Real t1, void *user_data, Real reltol,
src/math/FastMath.hpp:#include "AMReX_GpuQualifiers.H"
src/math/FastMath.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto fastlg(const double x) -> double
src/math/FastMath.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto fastpow2(const double x) -> double
src/math/FastMath.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto lg(const double x) -> double
src/math/FastMath.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto pow2(const double x) -> double { return fastpow2(x); }
src/math/FastMath.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto log10(const double x) -> double
src/math/FastMath.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto pow10(const double x) -> double
src/math/FastMath.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto log10(const double x) -> double { return std::log10(x); }
src/math/FastMath.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto pow10(const double x) -> double { return std::pow(10., x); }
src/math/math_impl.hpp:/// \brief Implements functions for various math operations on GPU not supported by CUDA C++
src/math/math_impl.hpp:#include "AMReX_GpuQualifiers.H"
src/math/math_impl.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto clamp(double v, double lo, double hi) -> double { return (v < lo) ? lo : (hi < v) ? hi : v; }
src/math/math_impl.hpp:template <typename T> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto sgn(T val) -> int { return (T(0) < val) - (val < T(0)); }
src/cooling/GrackleLikeCooling.cpp:auto grackle_tables::const_tables() const -> grackleGpuConstTables
src/cooling/GrackleLikeCooling.cpp:	grackleGpuConstTables tables{log_nH->const_table(),
src/cooling/TabulatedCooling.hpp:#include "AMReX_GpuQualifiers.H"
src/cooling/TabulatedCooling.hpp:struct cloudyGpuConstTables {
src/cooling/TabulatedCooling.hpp:	[[nodiscard]] auto const_tables() const -> cloudyGpuConstTables;
src/cooling/TabulatedCooling.hpp:	cloudyGpuConstTables tables;
src/cooling/TabulatedCooling.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto cloudy_cooling_function(Real const rho, Real const T, cloudyGpuConstTables const &tables) -> Real
src/cooling/TabulatedCooling.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeEgasFromTgas(double rho, double Tgas, double gamma, cloudyGpuConstTables const &tables) -> Real
src/cooling/TabulatedCooling.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeTgasFromEgas(double rho, double Egas, double gamma, cloudyGpuConstTables const &tables) -> Real
src/cooling/TabulatedCooling.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeCoolingLength(double rho, double Egas, double gamma, cloudyGpuConstTables const &tables) -> Real
src/cooling/TabulatedCooling.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeMMW(double rho, double Egas, double gamma, cloudyGpuConstTables const &tables) -> Real
src/cooling/TabulatedCooling.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int
src/cooling/TabulatedCooling.hpp:	cloudyGpuConstTables const &tables = udata->tables;
src/cooling/TabulatedCooling.hpp:		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/cooling/TabulatedCooling.cpp:auto cloudy_tables::const_tables() const -> cloudyGpuConstTables
src/cooling/TabulatedCooling.cpp:	cloudyGpuConstTables tables{log_nH->const_table(),
src/cooling/CloudyDataReader.hpp:#include "AMReX_GpuContainers.H"
src/cooling/CloudyDataReader.hpp:	std::vector<amrex::Gpu::PinnedVector<double>> grid_parametersVec;
src/cooling/CloudyDataReader.hpp:	amrex::Gpu::PinnedVector<double> heating_dataVec;
src/cooling/CloudyDataReader.hpp:	amrex::Gpu::PinnedVector<double> cooling_dataVec;
src/cooling/CloudyDataReader.hpp:	amrex::Gpu::PinnedVector<double> mmw_dataVec;
src/cooling/GrackleDataReader.cpp:		amrex::GpuArray<int, 3> lo{0, 0, 0};
src/cooling/GrackleDataReader.cpp:		amrex::GpuArray<int, 3> hi{static_cast<int>(my_cloudy.grid_dimension[2]), static_cast<int>(my_cloudy.grid_dimension[1]),
src/cooling/GrackleDataReader.cpp:		amrex::GpuArray<int, 3> lo{0, 0, 0};
src/cooling/GrackleDataReader.cpp:		amrex::GpuArray<int, 3> hi{static_cast<int>(my_cloudy.grid_dimension[2]), static_cast<int>(my_cloudy.grid_dimension[1]),
src/cooling/GrackleDataReader.cpp:		amrex::GpuArray<int, 3> lo{0, 0, 0};
src/cooling/GrackleDataReader.cpp:		amrex::GpuArray<int, 3> hi{static_cast<int>(my_cloudy.grid_dimension[2]), static_cast<int>(my_cloudy.grid_dimension[1]),
src/cooling/CloudyDataReader.cpp:#include "AMReX_GpuContainers.H"
src/cooling/CloudyDataReader.cpp:		my_cloudy.grid_parametersVec[q] = amrex::Gpu::PinnedVector<double>(my_cloudy.grid_dimension[q]);
src/cooling/CloudyDataReader.cpp:		my_cloudy.cooling_dataVec = amrex::Gpu::PinnedVector<double>(my_cloudy.data_size);
src/cooling/CloudyDataReader.cpp:		amrex::GpuArray<int, 2> const lo{0, 0};
src/cooling/CloudyDataReader.cpp:		amrex::GpuArray<int, 2> const hi{static_cast<int>(my_cloudy.grid_dimension[1]), static_cast<int>(my_cloudy.grid_dimension[0])};
src/cooling/CloudyDataReader.cpp:		my_cloudy.heating_dataVec = amrex::Gpu::PinnedVector<double>(my_cloudy.data_size);
src/cooling/CloudyDataReader.cpp:		amrex::GpuArray<int, 2> const lo{0, 0};
src/cooling/CloudyDataReader.cpp:		amrex::GpuArray<int, 2> const hi{static_cast<int>(my_cloudy.grid_dimension[1]), static_cast<int>(my_cloudy.grid_dimension[0])};
src/cooling/CloudyDataReader.cpp:		my_cloudy.mmw_dataVec = amrex::Gpu::PinnedVector<double>(my_cloudy.data_size);
src/cooling/CloudyDataReader.cpp:		amrex::GpuArray<int, 2> const lo{0, 0};
src/cooling/CloudyDataReader.cpp:		amrex::GpuArray<int, 2> const hi{static_cast<int>(my_cloudy.grid_dimension[1]), static_cast<int>(my_cloudy.grid_dimension[0])};
src/cooling/GrackleLikeCooling.hpp:#include "AMReX_GpuQualifiers.H"
src/cooling/GrackleLikeCooling.hpp:struct grackleGpuConstTables {
src/cooling/GrackleLikeCooling.hpp:	[[nodiscard]] auto const_tables() const -> grackleGpuConstTables;
src/cooling/GrackleLikeCooling.hpp:	grackleGpuConstTables tables;
src/cooling/GrackleLikeCooling.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto cloudy_cooling_function(Real const rho, Real const T, grackleGpuConstTables const &tables) -> Real
src/cooling/GrackleLikeCooling.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeEgasFromTgas(double rho, double Tgas, double gamma, grackleGpuConstTables const &tables) -> Real
src/cooling/GrackleLikeCooling.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeTgasFromEgas(double rho, double Egas, double gamma, grackleGpuConstTables const &tables) -> Real
src/cooling/GrackleLikeCooling.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int
src/cooling/GrackleLikeCooling.hpp:	grackleGpuConstTables const &tables = udata->tables;
src/cooling/GrackleLikeCooling.hpp:		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:AMREX_GPU_MANAGED double kappa0 = 100.;	 // NOLINT
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:AMREX_GPU_MANAGED double v0_adv = 1.0e6; // NOLINT
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:// AMREX_GPU_MANAGED double max_time = 4.8e-5; // max_time = 2.0 * width / v1;
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:AMREX_GPU_HOST_DEVICE
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:AMREX_GPU_HOST_DEVICE
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim.geom[0].ProbLoArray();
src/problems/RadhydroPulseGrey/test_radhydro_pulse_grey.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = sim.geom[0].ProbHiArray();
src/problems/RadhydroPulseGrey/CMakeLists.txt:  if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroPulseGrey/CMakeLists.txt:      setup_target_for_cuda_compilation(test_radhydro_pulse_grey)
src/problems/RadhydroPulseGrey/CMakeLists.txt:  endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroPulseMGconst/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroPulseMGconst/CMakeLists.txt:		setup_target_for_cuda_compilation(test_radhydro_pulse_MG_const_kappa)
src/problems/RadhydroPulseMGconst/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:// constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_{0., inf};
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:// constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_{1e15, 1e17, 1e19};
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_{1e15, 1e16, 1e17, 1e18, 1e19};
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:// constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_{1.00000000e+15, 1.15478198e+15, 1.33352143e+15, 1.53992653e+15,
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:AMREX_GPU_HOST_DEVICE
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:AMREX_GPU_HOST_DEVICE
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SGProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real { return kappa0; }
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SGProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:RadSystem<MGproblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> const /* rad_boundaries */, const double /* rho */,
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:							   const double /* Tgas */) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	amrex::GpuArray<double, nGroups_ + 1> exponents{};
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	amrex::GpuArray<double, nGroups_ + 1> kappa_lower{};
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> const exponents_and_values{exponents, kappa_lower};
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim2.geom[0].ProbLoArray();
src/problems/RadhydroPulseMGconst/test_radhydro_pulse_MG_const_kappa.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = sim2.geom[0].ProbHiArray();
src/problems/HydroRichtmeyerMeshkov/test_hydro2d_rm.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/HydroRichtmeyerMeshkov/test_hydro2d_rm.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/HydroRichtmeyerMeshkov/test_hydro2d_rm.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroRichtmeyerMeshkov/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroRichtmeyerMeshkov/CMakeLists.txt:        setup_target_for_cuda_compilation(test_hydro2d_rm)
src/problems/Advection2D/test_advection2d.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto exactSolutionAtIndex(int i, int j, amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_lo,
src/problems/Advection2D/test_advection2d.cpp:							      amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_hi,
src/problems/Advection2D/test_advection2d.cpp:							      amrex::GpuArray<Real, AMREX_SPACEDIM> const &dx) -> Real
src/problems/Advection2D/test_advection2d.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/Advection2D/test_advection2d.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/Advection2D/test_advection2d.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/Advection2D/test_advection2d.cpp:			   [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) { state_cc(i, j, k, n) = exactSolutionAtIndex(i, j, prob_lo, prob_hi, dx); });
src/problems/Advection2D/test_advection2d.cpp:void AdvectionSimulation<SquareProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/problems/Advection2D/test_advection2d.cpp:								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
src/problems/Advection2D/test_advection2d.cpp:								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi)
src/problems/Advection2D/test_advection2d.cpp:				   [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) { state(i, j, k, n) = exactSolutionAtIndex(i, j, prob_lo, prob_hi, dx); });
src/problems/Advection2D/test_advection2d.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/Advection2D/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/Advection2D/CMakeLists.txt:        setup_target_for_cuda_compilation(test_advection2d)
src/problems/Cooling/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/Cooling/CMakeLists.txt:        setup_target_for_cuda_compilation(test_cooling)
src/problems/Cooling/CMakeLists.txt:    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/Cooling/test_cooling.cpp:#include "AMReX_GpuDevice.H"
src/problems/Cooling/test_cooling.cpp:	// Copy data to GPU memory
src/problems/Cooling/test_cooling.cpp:	amrex::Gpu::streamSynchronize();
src/problems/Cooling/test_cooling.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/Cooling/test_cooling.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/Cooling/test_cooling.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/Cooling/test_cooling.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/Cooling/test_cooling.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/Cooling/test_cooling.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/RadhydroShock/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroShock/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radhydro_shock)
src/problems/RadhydroShock/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroShock/test_radhydro_shock.cpp:template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<ShockProblem>::ComputePlanckOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
src/problems/RadhydroShock/test_radhydro_shock.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShockProblem>::ComputeFluxMeanOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
src/problems/RadhydroShock/test_radhydro_shock.cpp:template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<ShockProblem>::ComputeEddingtonFactor(double /*f*/) -> double
src/problems/RadhydroShock/test_radhydro_shock.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadhydroShock/test_radhydro_shock.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/RadhydroShock/test_radhydro_shock.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/RadhydroShock/test_radhydro_shock.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/RadhydroShock/test_radhydro_shock.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroShock/test_radhydro_shock.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadMarshakDustPE/test_radiation_marshak_dust_and_PE.cpp:AMREX_GPU_MANAGED double kappa1 = NAN; // dust opacity at IR
src/problems/RadMarshakDustPE/test_radiation_marshak_dust_and_PE.cpp:AMREX_GPU_MANAGED double kappa2 = NAN; // dust opacity at FUV
src/problems/RadMarshakDustPE/test_radiation_marshak_dust_and_PE.cpp:static constexpr amrex::GpuArray<double, n_group_ + 1> radBoundaries_{1e-10, 30, 1e4};
src/problems/RadMarshakDustPE/test_radiation_marshak_dust_and_PE.cpp:	static constexpr amrex::GpuArray<double, n_group_ + 1> radBoundaries = radBoundaries_;
src/problems/RadMarshakDustPE/test_radiation_marshak_dust_and_PE.cpp:AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::DefinePhotoelectricHeatingE1Derivative(amrex::Real const /*temperature*/,
src/problems/RadMarshakDustPE/test_radiation_marshak_dust_and_PE.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
src/problems/RadMarshakDustPE/test_radiation_marshak_dust_and_PE.cpp:RadSystem<MarshakProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
src/problems/RadMarshakDustPE/test_radiation_marshak_dust_and_PE.cpp:								const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
src/problems/RadMarshakDustPE/test_radiation_marshak_dust_and_PE.cpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
src/problems/RadMarshakDustPE/test_radiation_marshak_dust_and_PE.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadMarshakDustPE/test_radiation_marshak_dust_and_PE.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadMarshakDustPE/test_radiation_marshak_dust_and_PE.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/RadMarshakDustPE/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadMarshakDustPE/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radiation_marshak_dust_PE)
src/problems/RadMarshakDustPE/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadTube/test_radiation_tube.cpp:	static constexpr amrex::GpuArray<double, Physics_Traits<TubeProblem>::nGroups + 1> radBoundaries{0.01 * T0, 3.3 * T0, 1000. * T0}; // Kelvin
src/problems/RadTube/test_radiation_tube.cpp:	// static constexpr amrex::GpuArray<double, Physics_Traits<TubeProblem>::nGroups + 1> radBoundaries{0.01 * T0, 1000. * T0}; // Kelvin
src/problems/RadTube/test_radiation_tube.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
src/problems/RadTube/test_radiation_tube.cpp:RadSystem<TubeProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
src/problems/RadTube/test_radiation_tube.cpp:							     const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
src/problems/RadTube/test_radiation_tube.cpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
src/problems/RadTube/test_radiation_tube.cpp:	amrex::Gpu::DeviceVector<double> x_arr_g;
src/problems/RadTube/test_radiation_tube.cpp:	amrex::Gpu::DeviceVector<double> rho_arr_g;
src/problems/RadTube/test_radiation_tube.cpp:	amrex::Gpu::DeviceVector<double> Pgas_arr_g;
src/problems/RadTube/test_radiation_tube.cpp:	amrex::Gpu::DeviceVector<double> Erad_arr_g;
src/problems/RadTube/test_radiation_tube.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, x_arr.begin(), x_arr.end(), userData_.x_arr_g.begin());
src/problems/RadTube/test_radiation_tube.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, rho_arr.begin(), rho_arr.end(), userData_.rho_arr_g.begin());
src/problems/RadTube/test_radiation_tube.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Pgas_arr.begin(), Pgas_arr.end(), userData_.Pgas_arr_g.begin());
src/problems/RadTube/test_radiation_tube.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Erad_arr.begin(), Erad_arr.end(), userData_.Erad_arr_g.begin());
src/problems/RadTube/test_radiation_tube.cpp:	amrex::Gpu::streamSynchronizeAll();
src/problems/RadTube/test_radiation_tube.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/RadTube/test_radiation_tube.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadTube/test_radiation_tube.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RadTube/test_radiation_tube.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadTube/test_radiation_tube.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/RadTube/test_radiation_tube.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/RadTube/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadTube/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radiation_tube)
src/problems/RadTube/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroVacuum/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroVacuum/CMakeLists.txt:    setup_target_for_cuda_compilation(test_hydro_vacuum)
src/problems/HydroVacuum/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroVacuum/test_hydro_vacuum.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/HydroVacuum/test_hydro_vacuum.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/HydroVacuum/test_hydro_vacuum.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroVacuum/test_hydro_vacuum.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/HydroVacuum/test_hydro_vacuum.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/HydroVacuum/test_hydro_vacuum.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/HydroVacuum/test_hydro_vacuum.cpp:void QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/problems/HydroVacuum/test_hydro_vacuum.cpp:								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
src/problems/HydroVacuum/test_hydro_vacuum.cpp:	amrex::Gpu::DeviceVector<double> rho_g(density_exact_interp.size());
src/problems/HydroVacuum/test_hydro_vacuum.cpp:	amrex::Gpu::DeviceVector<double> vx_g(velocity_exact_interp.size());
src/problems/HydroVacuum/test_hydro_vacuum.cpp:	amrex::Gpu::DeviceVector<double> P_g(pressure_exact_interp.size());
src/problems/HydroVacuum/test_hydro_vacuum.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, density_exact_interp.begin(), density_exact_interp.end(), rho_g.begin());
src/problems/HydroVacuum/test_hydro_vacuum.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, velocity_exact_interp.begin(), velocity_exact_interp.end(), vx_g.begin());
src/problems/HydroVacuum/test_hydro_vacuum.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, pressure_exact_interp.begin(), pressure_exact_interp.end(), P_g.begin());
src/problems/HydroVacuum/test_hydro_vacuum.cpp:	amrex::Gpu::streamSynchronizeAll();
src/problems/HydroVacuum/test_hydro_vacuum.cpp:		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RadhydroShockMultigroup/test_radhydro_shock_multigroup.cpp:	static constexpr amrex::GpuArray<double, Physics_Traits<ShockProblem>::nGroups + 1> radBoundaries{1.00000000e+15, 1.00000000e+16, 1.00000000e+17,
src/problems/RadhydroShockMultigroup/test_radhydro_shock_multigroup.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
src/problems/RadhydroShockMultigroup/test_radhydro_shock_multigroup.cpp:RadSystem<ShockProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double rho, const double /*Tgas*/)
src/problems/RadhydroShockMultigroup/test_radhydro_shock_multigroup.cpp:    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
src/problems/RadhydroShockMultigroup/test_radhydro_shock_multigroup.cpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
src/problems/RadhydroShockMultigroup/test_radhydro_shock_multigroup.cpp:template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<ShockProblem>::ComputeEddingtonFactor(double /*f*/) -> double
src/problems/RadhydroShockMultigroup/test_radhydro_shock_multigroup.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadhydroShockMultigroup/test_radhydro_shock_multigroup.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/RadhydroShockMultigroup/test_radhydro_shock_multigroup.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/RadhydroShockMultigroup/test_radhydro_shock_multigroup.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/RadhydroShockMultigroup/test_radhydro_shock_multigroup.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroShockMultigroup/test_radhydro_shock_multigroup.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroShockMultigroup/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroShockMultigroup/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radhydro_shock_multigroup)
src/problems/RadhydroShockMultigroup/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadDustMG/test_rad_dust_MG.cpp:	static constexpr amrex::GpuArray<double, Physics_Traits<DustProblem>::nGroups + 1> radBoundaries{1.0e-3, 0.1, 1.0, 10.0, 1.0e3};
src/problems/RadDustMG/test_rad_dust_MG.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
src/problems/RadDustMG/test_rad_dust_MG.cpp:RadSystem<DustProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double rho, const double /*Tgas*/)
src/problems/RadDustMG/test_rad_dust_MG.cpp:    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
src/problems/RadDustMG/test_rad_dust_MG.cpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
src/problems/RadDustMG/test_rad_dust_MG.cpp:AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputeThermalRadiationMultiGroup(amrex::Real temperature,
src/problems/RadDustMG/test_rad_dust_MG.cpp:										     amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
src/problems/RadDustMG/test_rad_dust_MG.cpp:AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputeThermalRadiationTempDerivativeMultiGroup(amrex::Real temperature,
src/problems/RadDustMG/test_rad_dust_MG.cpp:												   amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
src/problems/RadDustMG/test_rad_dust_MG.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadDustMG/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadDustMG/CMakeLists.txt:    setup_target_for_cuda_compilation(test_rad_dust_MG)
src/problems/RadDustMG/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/FCQuantities/test_fc_quantities.cpp:AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/problems/FCQuantities/test_fc_quantities.cpp:					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
src/problems/FCQuantities/test_fc_quantities.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/FCQuantities/test_fc_quantities.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/FCQuantities/test_fc_quantities.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/FCQuantities/test_fc_quantities.cpp:		    indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { state(i, j, k, MHDSystem<FCQuantities>::bfield_index) = 1.0 + (i % 2); });
src/problems/FCQuantities/test_fc_quantities.cpp:		    indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { state(i, j, k, MHDSystem<FCQuantities>::bfield_index) = 2.0 + (j % 2); });
src/problems/FCQuantities/test_fc_quantities.cpp:		    indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { state(i, j, k, MHDSystem<FCQuantities>::bfield_index) = 3.0 + (k % 2); });
src/problems/FCQuantities/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/FCQuantities/CMakeLists.txt:    setup_target_for_cuda_compilation(test_fc_quantities)
src/problems/RadMarshakVaytet/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadMarshakVaytet/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radiation_marshak_Vaytet)
src/problems/RadMarshakVaytet/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:// opacity (Vaytet et al. Sec 3.2.3) constexpr int n_groups_ = 6; constexpr amrex::GpuArray<double, n_groups_ + 1> group_edges_ = {0.3e12, 0.3e14, 0.6e14,
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:// 0.9e14, 1.2e14, 1.5e14, 1.5e16}; constexpr amrex::GpuArray<double, n_groups_> group_opacities_ = {1000., 750., 500., 250., 10., 10.};
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:constexpr amrex::GpuArray<double, n_groups_ + 1> group_edges_ = []() constexpr {
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:		return amrex::GpuArray<double, 3>{6.0e10, 6.0e12, 6.0e14};
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:		return amrex::GpuArray<double, 5>{6.0e10, 6.0e11, 6.0e12, 6.0e13, 6.0e14};
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:		return amrex::GpuArray<double, 9>{6.0000000e+10, 1.8973666e+11, 6.0000000e+11, 1.8973666e+12, 6.0000000e+12,
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:		return amrex::GpuArray<double, 17>{6.00000000e+10, 1.06696765e+11, 1.89736660e+11, 3.37404795e+11, 6.00000000e+11, 1.06696765e+12,
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:		return amrex::GpuArray<double, 65>{
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:		return amrex::GpuArray<double, 129>{
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:constexpr amrex::GpuArray<double, n_groups_> group_opacities_{};
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = group_edges_;
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:RadSystem<SuOlsonProblemCgs>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, const double /*rho*/,
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:								   const double Tgas) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/RadMarshakVaytet/test_radiation_marshak_Vaytet.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroShockCGS/test_radhydro_shock_cgs.cpp:template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<ShockProblem>::ComputePlanckOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
src/problems/RadhydroShockCGS/test_radhydro_shock_cgs.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShockProblem>::ComputeFluxMeanOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
src/problems/RadhydroShockCGS/test_radhydro_shock_cgs.cpp:template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<ShockProblem>::ComputeEddingtonFactor(double /*f*/) -> double
src/problems/RadhydroShockCGS/test_radhydro_shock_cgs.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadhydroShockCGS/test_radhydro_shock_cgs.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/RadhydroShockCGS/test_radhydro_shock_cgs.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/RadhydroShockCGS/test_radhydro_shock_cgs.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/RadhydroShockCGS/test_radhydro_shock_cgs.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroShockCGS/test_radhydro_shock_cgs.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroShockCGS/test_radhydro_shock_cgs.cpp:	//     [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroShockCGS/test_radhydro_shock_cgs.cpp:	//     [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroShockCGS/test_radhydro_shock_cgs.cpp:	//     [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroShockCGS/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroShockCGS/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radhydro_shock_cgs)
src/problems/RadhydroShockCGS/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/PopIII/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/PopIII/CMakeLists.txt:        setup_target_for_cuda_compilation(popiii)
src/problems/PopIII/CMakeLists.txt:    # AMR test only works on Setonix because Gadi and avatar do not have enough memory per GPU
src/problems/PopIII/popiii.cpp:		// copy to GPU
src/problems/PopIII/popiii.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/PopIII/popiii.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/PopIII/popiii.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/PopIII/popiii.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/PopIII/popiii.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/PopIII/popiii.cpp:			amrex::GpuArray<Real, Physics_Traits<PopIII>::numMassScalars> massScalars = RadSystem<PopIII>::ComputeMassScalars(state, i, j, k);
src/problems/PopIII/popiii.cpp:		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/PopIII/popiii.cpp:			amrex::GpuArray<Real, Physics_Traits<PopIII>::numMassScalars> massScalars = RadSystem<PopIII>::ComputeMassScalars(state[bx], i, j, k);
src/problems/PopIII/popiii.cpp:		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/PopIII/popiii.cpp:		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/PopIII/popiii.cpp:		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/PopIII/popiii.cpp:			amrex::GpuArray<Real, Physics_Traits<PopIII>::numMassScalars> massScalars = RadSystem<PopIII>::ComputeMassScalars(state[bx], i, j, k);
src/problems/AdvectionSemiellipse/test_advection_semiellipse.cpp:AMREX_GPU_DEVICE void ComputeExactSolution(int i, int j, int k, int n, amrex::Array4<amrex::Real> const &exact_arr,
src/problems/AdvectionSemiellipse/test_advection_semiellipse.cpp:					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
src/problems/AdvectionSemiellipse/test_advection_semiellipse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/AdvectionSemiellipse/test_advection_semiellipse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/AdvectionSemiellipse/test_advection_semiellipse.cpp:	amrex::ParallelFor(indexRange, ncomp_cc, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) { ComputeExactSolution(i, j, k, n, state_cc, dx, prob_lo); });
src/problems/AdvectionSemiellipse/test_advection_semiellipse.cpp:void AdvectionSimulation<SemiellipseProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/problems/AdvectionSemiellipse/test_advection_semiellipse.cpp:								       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
src/problems/AdvectionSemiellipse/test_advection_semiellipse.cpp:								       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi)
src/problems/AdvectionSemiellipse/test_advection_semiellipse.cpp:				   [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept { ComputeExactSolution(i, j, k, n, stateExact, dx, prob_lo); });
src/problems/AdvectionSemiellipse/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/AdvectionSemiellipse/CMakeLists.txt:    setup_target_for_cuda_compilation(test_advection_se)
src/problems/AdvectionSemiellipse/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadPulse/test_radiation_pulse.cpp:AMREX_GPU_HOST_DEVICE
src/problems/RadPulse/test_radiation_pulse.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadPulse/test_radiation_pulse.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadPulse/test_radiation_pulse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/RadPulse/test_radiation_pulse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadPulse/test_radiation_pulse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/RadPulse/test_radiation_pulse.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadPulse/test_radiation_pulse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim.geom[0].ProbLoArray();
src/problems/RadPulse/test_radiation_pulse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = sim.geom[0].ProbHiArray();
src/problems/RadPulse/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadPulse/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radiation_pulse)
src/problems/RadPulse/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroUniformAdvecting/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroUniformAdvecting/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radhydro_uniform_advecting)
src/problems/RadhydroUniformAdvecting/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroUniformAdvecting/test_radhydro_uniform_advecting.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadhydroUniformAdvecting/test_radhydro_uniform_advecting.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadhydroUniformAdvecting/test_radhydro_uniform_advecting.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroShocktube/test_hydro_shocktube.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/HydroShocktube/test_hydro_shocktube.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/HydroShocktube/test_hydro_shocktube.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroShocktube/test_hydro_shocktube.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/HydroShocktube/test_hydro_shocktube.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/HydroShocktube/test_hydro_shocktube.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/HydroShocktube/test_hydro_shocktube.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/HydroShocktube/test_hydro_shocktube.cpp:void QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/problems/HydroShocktube/test_hydro_shocktube.cpp:								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
src/problems/HydroShocktube/test_hydro_shocktube.cpp:	amrex::Gpu::DeviceVector<double> rho_g(density_exact_interp.size());
src/problems/HydroShocktube/test_hydro_shocktube.cpp:	amrex::Gpu::DeviceVector<double> vx_g(velocity_exact_interp.size());
src/problems/HydroShocktube/test_hydro_shocktube.cpp:	amrex::Gpu::DeviceVector<double> P_g(pressure_exact_interp.size());
src/problems/HydroShocktube/test_hydro_shocktube.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, density_exact_interp.begin(), density_exact_interp.end(), rho_g.begin());
src/problems/HydroShocktube/test_hydro_shocktube.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, velocity_exact_interp.begin(), velocity_exact_interp.end(), vx_g.begin());
src/problems/HydroShocktube/test_hydro_shocktube.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, pressure_exact_interp.begin(), pressure_exact_interp.end(), P_g.begin());
src/problems/HydroShocktube/test_hydro_shocktube.cpp:	amrex::Gpu::streamSynchronizeAll();
src/problems/HydroShocktube/test_hydro_shocktube.cpp:		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/HydroShocktube/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroShocktube/CMakeLists.txt:    setup_target_for_cuda_compilation(test_hydro_shocktube)
src/problems/HydroShocktube/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadStreamingY/test_radiation_streaming_y.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadStreamingY/test_radiation_streaming_y.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadStreamingY/test_radiation_streaming_y.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadStreamingY/test_radiation_streaming_y.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadStreamingY/test_radiation_streaming_y.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/RadStreamingY/test_radiation_streaming_y.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/RadStreamingY/CMakeLists.txt:	if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadStreamingY/CMakeLists.txt:			setup_target_for_cuda_compilation(test_radiation_streaming_y)
src/problems/RadStreamingY/CMakeLists.txt:	endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroQuirk/test_quirk.cpp:#include "AMReX_GpuAsyncArray.H"
src/problems/HydroQuirk/test_quirk.cpp:#include "AMReX_GpuQualifiers.H"
src/problems/HydroQuirk/test_quirk.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/HydroQuirk/test_quirk.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/HydroQuirk/test_quirk.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroQuirk/test_quirk.cpp:			amrex::GpuArray<int, 3> box_lo = bx.loVect3d();
src/problems/HydroQuirk/test_quirk.cpp:		amrex::launch(bx, [=] AMREX_GPU_DEVICE(amrex::Box const &tbx) {
src/problems/HydroQuirk/test_quirk.cpp:			amrex::GpuArray<int, 3> const idx = tbx.loVect3d();
src/problems/HydroQuirk/test_quirk.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/HydroQuirk/test_quirk.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/HydroQuirk/test_quirk.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/HydroQuirk/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroQuirk/CMakeLists.txt:        setup_target_for_cuda_compilation(test_quirk)
src/problems/BinaryOrbitCIC/binary_orbit.cpp:#include "AMReX_GpuContainers.H"
src/problems/BinaryOrbitCIC/binary_orbit.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/BinaryOrbitCIC/binary_orbit.cpp:		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
src/problems/BinaryOrbitCIC/binary_orbit.cpp:				amrex::Gpu::copy(amrex::Gpu::deviceToHost, pData, pData + np, pData_h.begin()); // NOLINT
src/problems/BinaryOrbitCIC/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/BinaryOrbitCIC/CMakeLists.txt:        setup_target_for_cuda_compilation(binary_orbit)
src/problems/HydroKelvinHelmholz/test_hydro2d_kh.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/HydroKelvinHelmholz/test_hydro2d_kh.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/HydroKelvinHelmholz/test_hydro2d_kh.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/HydroKelvinHelmholz/test_hydro2d_kh.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/HydroKelvinHelmholz/test_hydro2d_kh.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/HydroKelvinHelmholz/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroKelvinHelmholz/CMakeLists.txt:        setup_target_for_cuda_compilation(test_hydro2d_kh)
src/problems/RadMarshakDust/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadMarshakDust/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radiation_marshak_dust)
src/problems/RadMarshakDust/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:AMREX_GPU_MANAGED double kappa1 = NAN; // dust opacity at IR
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:AMREX_GPU_MANAGED double kappa2 = NAN; // dust opacity at FUV
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:// static constexpr amrex::GpuArray<double, n_group_ + 1> radBoundaries_{1e-10, 1e4};
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:static constexpr amrex::GpuArray<double, n_group_ + 1> radBoundaries_{1e-10, 100, 1e4};
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:	static constexpr amrex::GpuArray<double, n_group_ + 1> radBoundaries = radBoundaries_;
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:RadSystem<MarshakProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:								const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadMarshakDust/test_radiation_marshak_dust.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/RadTophat/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadTophat/CMakeLists.txt:        setup_target_for_cuda_compilation(test_radiation_tophat)
src/problems/RadTophat/test_radiation_tophat.cpp:template <> AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto RadSystem<TophatProblem>::ComputePlanckOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
src/problems/RadTophat/test_radiation_tophat.cpp:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto RadSystem<TophatProblem>::ComputeFluxMeanOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
src/problems/RadTophat/test_radiation_tophat.cpp:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto
src/problems/RadTophat/test_radiation_tophat.cpp:						std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/) -> double
src/problems/RadTophat/test_radiation_tophat.cpp:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto
src/problems/RadTophat/test_radiation_tophat.cpp:						std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/) -> double
src/problems/RadTophat/test_radiation_tophat.cpp:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto
src/problems/RadTophat/test_radiation_tophat.cpp:						      std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/) -> double
src/problems/RadTophat/test_radiation_tophat.cpp:template <> AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto RadSystem<TophatProblem>::ComputeEddingtonFactor(const double f_in) -> double
src/problems/RadTophat/test_radiation_tophat.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadTophat/test_radiation_tophat.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/RadTophat/test_radiation_tophat.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/RadTophat/test_radiation_tophat.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadTophat/test_radiation_tophat.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroShuOsher/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroShuOsher/CMakeLists.txt:    setup_target_for_cuda_compilation(test_hydro_shuosher)
src/problems/HydroShuOsher/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:void QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:	amrex::Gpu::DeviceVector<double> rho_g(density_exact_interp.size());
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:	amrex::Gpu::DeviceVector<double> vx_g(velocity_exact_interp.size());
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:	amrex::Gpu::DeviceVector<double> P_g(pressure_exact_interp.size());
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, density_exact_interp.begin(), density_exact_interp.end(), rho_g.begin());
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, velocity_exact_interp.begin(), velocity_exact_interp.end(), vx_g.begin());
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, pressure_exact_interp.begin(), pressure_exact_interp.end(), P_g.begin());
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:	amrex::Gpu::streamSynchronizeAll();
src/problems/HydroShuOsher/test_hydro_shuosher.cpp:		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RadStreaming/test_radiation_streaming.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadStreaming/test_radiation_streaming.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadStreaming/test_radiation_streaming.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadStreaming/test_radiation_streaming.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadStreaming/test_radiation_streaming.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/RadStreaming/test_radiation_streaming.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/RadStreaming/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadStreaming/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radiation_streaming)
src/problems/RadStreaming/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:AMREX_GPU_MANAGED double kappa0 = 100.;	 // NOLINT
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:AMREX_GPU_MANAGED double v0_adv = 1.0e6; // NOLINT
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:// AMREX_GPU_MANAGED double max_time = 4.8e-5; // max_time = 2.0 * width / v1;
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:AMREX_GPU_HOST_DEVICE
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:AMREX_GPU_HOST_DEVICE
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim2.geom[0].ProbLoArray();
src/problems/RadhydroPulse/test_radhydro_pulse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = sim2.geom[0].ProbHiArray();
src/problems/RadhydroPulse/CMakeLists.txt:  if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroPulse/CMakeLists.txt:      setup_target_for_cuda_compilation(test_radhydro_pulse)
src/problems/RadhydroPulse/CMakeLists.txt:  endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/StarCluster/star_cluster.cpp:		// copy to GPU
src/problems/StarCluster/star_cluster.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/StarCluster/star_cluster.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/StarCluster/star_cluster.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/StarCluster/star_cluster.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/StarCluster/star_cluster.cpp:	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/StarCluster/star_cluster.cpp:		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/StarCluster/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/StarCluster/CMakeLists.txt:        setup_target_for_cuda_compilation(star_cluster)
src/problems/StarCluster/CMakeLists.txt:    # AMR test only works on Setonix because Gadi and avatar do not have enough memory per GPU
src/problems/Advection/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/Advection/CMakeLists.txt:    setup_target_for_cuda_compilation(test_advection)
src/problems/Advection/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/Advection/test_advection.cpp:AMREX_GPU_DEVICE void ComputeExactSolution(int i, int j, int k, int n, amrex::Array4<amrex::Real> const &exact_arr,
src/problems/Advection/test_advection.cpp:					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
src/problems/Advection/test_advection.cpp:					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi)
src/problems/Advection/test_advection.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/Advection/test_advection.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/Advection/test_advection.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/Advection/test_advection.cpp:			   [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) { ComputeExactSolution(i, j, k, n, state_cc, dx, prob_lo, prob_hi); });
src/problems/Advection/test_advection.cpp:void AdvectionSimulation<SawtoothProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/problems/Advection/test_advection.cpp:								    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
src/problems/Advection/test_advection.cpp:								    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi)
src/problems/Advection/test_advection.cpp:		amrex::ParallelFor(indexRange, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
src/problems/ODEIntegration/test_ode.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto user_rhs(Real t, quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int;
src/problems/ODEIntegration/test_ode.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto cooling_function(Real const rho, Real const T) -> Real
src/problems/ODEIntegration/test_ode.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int
src/problems/ODEIntegration/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/ODEIntegration/CMakeLists.txt:    setup_target_for_cuda_compilation(test_ode)
src/problems/ODEIntegration/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RayleighTaylor3D/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RayleighTaylor3D/CMakeLists.txt:        setup_target_for_cuda_compilation(test_hydro3d_rt)
src/problems/RayleighTaylor3D/test_hydro3d_rt.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/RayleighTaylor3D/test_hydro3d_rt.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RayleighTaylor3D/test_hydro3d_rt.cpp:	amrex::ParallelForRNG(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const &rng) noexcept {
src/problems/RayleighTaylor3D/test_hydro3d_rt.cpp:	amrex::ParallelFor(state_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/RayleighTaylor3D/test_hydro3d_rt.cpp:	amrex::Gpu::streamSynchronize();
src/problems/RayleighTaylor3D/test_hydro3d_rt.cpp:	amrex::ParallelFor(state_new_cc_[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/RayleighTaylor3D/test_hydro3d_rt.cpp:	amrex::Gpu::streamSynchronize();
src/problems/RayleighTaylor3D/test_hydro3d_rt.cpp:		auto profile = computeAxisAlignedProfile(axis, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) {
src/problems/HydroBlast3D/test_hydro3d_blast.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/HydroBlast3D/test_hydro3d_blast.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/HydroBlast3D/test_hydro3d_blast.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/HydroBlast3D/test_hydro3d_blast.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroBlast3D/test_hydro3d_blast.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/HydroBlast3D/test_hydro3d_blast.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = geom[0].CellSizeArray();
src/problems/HydroBlast3D/test_hydro3d_blast.cpp:		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroBlast3D/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroBlast3D/CMakeLists.txt:        setup_target_for_cuda_compilation(test_hydro3d_blast)
src/problems/HydroHighMach/test_hydro_highmach.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/HydroHighMach/test_hydro_highmach.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/HydroHighMach/test_hydro_highmach.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroHighMach/test_hydro_highmach.cpp:void QuokkaSimulation<HighMachProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<Real, AMREX_SPACEDIM> const &dx,
src/problems/HydroHighMach/test_hydro_highmach.cpp:								 amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_lo)
src/problems/HydroHighMach/test_hydro_highmach.cpp:	amrex::Gpu::HostVector<double> d_interp(x.size());
src/problems/HydroHighMach/test_hydro_highmach.cpp:	amrex::Gpu::HostVector<double> vx_interp(x.size());
src/problems/HydroHighMach/test_hydro_highmach.cpp:	amrex::Gpu::HostVector<double> P_interp(x.size());
src/problems/HydroHighMach/test_hydro_highmach.cpp:	amrex::Gpu::DeviceVector<double> rho_g(d_interp.size());
src/problems/HydroHighMach/test_hydro_highmach.cpp:	amrex::Gpu::DeviceVector<double> vx_g(vx_interp.size());
src/problems/HydroHighMach/test_hydro_highmach.cpp:	amrex::Gpu::DeviceVector<double> P_g(P_interp.size());
src/problems/HydroHighMach/test_hydro_highmach.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, d_interp.begin(), d_interp.end(), rho_g.begin());
src/problems/HydroHighMach/test_hydro_highmach.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, vx_interp.begin(), vx_interp.end(), vx_g.begin());
src/problems/HydroHighMach/test_hydro_highmach.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, P_interp.begin(), P_interp.end(), P_g.begin());
src/problems/HydroHighMach/test_hydro_highmach.cpp:	amrex::Gpu::streamSynchronizeAll();
src/problems/HydroHighMach/test_hydro_highmach.cpp:		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/HydroHighMach/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroHighMach/CMakeLists.txt:    setup_target_for_cuda_compilation(test_hydro_highmach)
src/problems/HydroHighMach/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/PrimordialChem/test_primordial_chem.cpp:#include "AMReX_GpuDevice.H"
src/problems/PrimordialChem/test_primordial_chem.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/PrimordialChem/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/PrimordialChem/CMakeLists.txt:    setup_target_for_cuda_compilation(test_primordial_chem)
src/problems/PrimordialChem/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroPulseMGint/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroPulseMGint/CMakeLists.txt:		setup_target_for_cuda_compilation(test_radhydro_pulse_MG_int)
src/problems/RadhydroPulseMGint/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_ = []() constexpr {
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:		return amrex::GpuArray<double, 3>{1e15, 1e17, 1e19};
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:		return amrex::GpuArray<double, 5>{1e15, 1e16, 1e17, 1e18, 1e19};
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:		return amrex::GpuArray<double, 9>{1e15, 3.16e15, 1e16, 3.16e16, 1e17, 3.16e17, 1e18, 3.16e18, 1e19};
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:		return amrex::GpuArray<double, 17>{1.00000000e+15, 1.77827941e+15, 3.16227766e+15, 5.62341325e+15, 1.00000000e+16, 1.77827941e+16,
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:		return amrex::GpuArray<double, 33>{1.00000000e+15, 1.33352143e+15, 1.77827941e+15, 2.37137371e+15, 3.16227766e+15, 4.21696503e+15,
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:		return amrex::GpuArray<double, 65>{
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:constexpr int64_t max_timesteps = 1e2; // to make 3D test run fast on GPUs
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:AMREX_GPU_HOST_DEVICE
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:AMREX_GPU_HOST_DEVICE
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:AMREX_GPU_HOST_DEVICE
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:RadSystem<MGProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> const rad_boundaries, const double rho, const double Tgas)
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	amrex::GpuArray<double, nGroups_ + 1> exponents{};
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	amrex::GpuArray<double, nGroups_ + 1> kappa_lower{};
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> const exponents_and_values{exponents, kappa_lower};
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ExactProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ExactProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim.geom[0].ProbLoArray();
src/problems/RadhydroPulseMGint/test_radhydro_pulse_MG_int.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = sim.geom[0].ProbHiArray();
src/problems/RadhydroShell/test_radhydro_shell.cpp:void RadSystem<ShellProblem>::SetRadEnergySource(array_t &radEnergy, const amrex::Box &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/problems/RadhydroShell/test_radhydro_shell.cpp:						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
src/problems/RadhydroShell/test_radhydro_shell.cpp:						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
src/problems/RadhydroShell/test_radhydro_shell.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RadhydroShell/test_radhydro_shell.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShellProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadhydroShell/test_radhydro_shell.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShellProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadhydroShell/test_radhydro_shell.cpp:amrex::Gpu::HostVector<double> r_arr;
src/problems/RadhydroShell/test_radhydro_shell.cpp:amrex::Gpu::HostVector<double> Erad_arr;
src/problems/RadhydroShell/test_radhydro_shell.cpp:amrex::Gpu::HostVector<double> Frad_arr;
src/problems/RadhydroShell/test_radhydro_shell.cpp:amrex::Gpu::DeviceVector<double> r_arr_g;
src/problems/RadhydroShell/test_radhydro_shell.cpp:amrex::Gpu::DeviceVector<double> Erad_arr_g;
src/problems/RadhydroShell/test_radhydro_shell.cpp:amrex::Gpu::DeviceVector<double> Frad_arr_g;
src/problems/RadhydroShell/test_radhydro_shell.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, r_arr.begin(), r_arr.end(), r_arr_g.begin());
src/problems/RadhydroShell/test_radhydro_shell.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Erad_arr.begin(), Erad_arr.end(), Erad_arr_g.begin());
src/problems/RadhydroShell/test_radhydro_shell.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Frad_arr.begin(), Frad_arr.end(), Frad_arr_g.begin());
src/problems/RadhydroShell/test_radhydro_shell.cpp:	amrex::Gpu::streamSynchronizeAll();
src/problems/RadhydroShell/test_radhydro_shell.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/RadhydroShell/test_radhydro_shell.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroShell/test_radhydro_shell.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/RadhydroShell/test_radhydro_shell.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RadhydroShell/test_radhydro_shell.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto vec_dot_r(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vec, int i, int j, int k) -> amrex::Real
src/problems/RadhydroShell/test_radhydro_shell.cpp:  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 =
src/problems/RadhydroShell/test_radhydro_shell.cpp:          [=] AMREX_GPU_DEVICE(amrex::Box const &bx,
src/problems/RadhydroShell/test_radhydro_shell.cpp:              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vec{
src/problems/RadhydroShell/test_radhydro_shell.cpp:          [=] AMREX_GPU_DEVICE(amrex::Box const &bx,
src/problems/RadhydroShell/test_radhydro_shell.cpp:              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vec{
src/problems/RadhydroShell/test_radhydro_shell.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RadhydroShell/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroShell/CMakeLists.txt:        setup_target_for_cuda_compilation(test_radhydro3d_shell)
src/problems/HydroSMS/test_hydro_sms.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/HydroSMS/test_hydro_sms.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/HydroSMS/test_hydro_sms.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroSMS/test_hydro_sms.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/HydroSMS/test_hydro_sms.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/HydroSMS/test_hydro_sms.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/HydroSMS/test_hydro_sms.cpp:void QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/problems/HydroSMS/test_hydro_sms.cpp:								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
src/problems/HydroSMS/test_hydro_sms.cpp:	amrex::Gpu::DeviceVector<double> rho_g(density_exact.size());
src/problems/HydroSMS/test_hydro_sms.cpp:	amrex::Gpu::DeviceVector<double> vx_g(velocity_exact.size());
src/problems/HydroSMS/test_hydro_sms.cpp:	amrex::Gpu::DeviceVector<double> P_g(pressure_exact.size());
src/problems/HydroSMS/test_hydro_sms.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, density_exact.begin(), density_exact.end(), rho_g.begin());
src/problems/HydroSMS/test_hydro_sms.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, velocity_exact.begin(), velocity_exact.end(), vx_g.begin());
src/problems/HydroSMS/test_hydro_sms.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, pressure_exact.begin(), pressure_exact.end(), P_g.begin());
src/problems/HydroSMS/test_hydro_sms.cpp:	amrex::Gpu::streamSynchronizeAll();
src/problems/HydroSMS/test_hydro_sms.cpp:		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/HydroSMS/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroSMS/CMakeLists.txt:    setup_target_for_cuda_compilation(test_hydro_sms)
src/problems/HydroSMS/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:AMREX_GPU_MANAGED double kappa0 = 500.;	 // NOLINT
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:AMREX_GPU_MANAGED double v0_adv = 3.0e7; // NOLINT
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:// AMREX_GPU_MANAGED double max_time = 4.8e-6;
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:AMREX_GPU_HOST_DEVICE
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:AMREX_GPU_HOST_DEVICE
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim2.geom[0].ProbLoArray();
src/problems/RadhydroPulseDyn/test_radhydro_pulse_dyn.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = sim2.geom[0].ProbHiArray();
src/problems/RadhydroPulseDyn/CMakeLists.txt:  if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroPulseDyn/CMakeLists.txt:      setup_target_for_cuda_compilation(test_radhydro_pulse_dyn)
src/problems/RadhydroPulseDyn/CMakeLists.txt:  endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/NSCBC/vortex.cpp:#include "AMReX_GpuDevice.H"
src/problems/NSCBC/vortex.cpp:AMREX_GPU_MANAGED amrex::Real T_ref = NAN;				      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/NSCBC/vortex.cpp:AMREX_GPU_MANAGED amrex::Real P_ref = NAN;				      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/NSCBC/vortex.cpp:AMREX_GPU_MANAGED amrex::Real u0 = NAN;					      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/NSCBC/vortex.cpp:AMREX_GPU_MANAGED amrex::Real v0 = NAN;					      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/NSCBC/vortex.cpp:AMREX_GPU_MANAGED amrex::Real w0 = NAN;					      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/NSCBC/vortex.cpp:AMREX_GPU_MANAGED amrex::GpuArray<Real, HydroSystem<Vortex>::nscalars_> s0{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/NSCBC/vortex.cpp:	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/NSCBC/vortex.cpp:	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/NSCBC/vortex.cpp:	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/NSCBC/vortex.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/NSCBC/vortex.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<Vortex>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
src/problems/NSCBC/channel.cpp:#include "AMReX_GpuDevice.H"
src/problems/NSCBC/channel.cpp:#if 0 // workaround AMDGPU compiler bug
src/problems/NSCBC/channel.cpp:AMREX_GPU_MANAGED amrex::Real Tgas0 = NAN;							// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/NSCBC/channel.cpp:AMREX_GPU_MANAGED amrex::Real P_outflow = NAN;							// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/NSCBC/channel.cpp:AMREX_GPU_MANAGED amrex::Real u_inflow = NAN;							// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/NSCBC/channel.cpp:AMREX_GPU_MANAGED amrex::Real v_inflow = NAN;							// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/NSCBC/channel.cpp:AMREX_GPU_MANAGED amrex::Real w_inflow = NAN;							// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/NSCBC/channel.cpp:AMREX_GPU_MANAGED amrex::GpuArray<Real, Physics_Traits<Channel>::numPassiveScalars> s_inflow{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/NSCBC/channel.cpp:#if 0												// workaround AMDGPU compiler bug
src/problems/NSCBC/channel.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/NSCBC/channel.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<Channel>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
src/problems/NSCBC/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/NSCBC/CMakeLists.txt:    setup_target_for_cuda_compilation(test_channel_flow)
src/problems/NSCBC/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/NSCBC/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/NSCBC/CMakeLists.txt:        setup_target_for_cuda_compilation(test_vortex)
src/problems/NSCBC/CMakeLists.txt:    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/PassiveScalar/test_scalars.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/PassiveScalar/test_scalars.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/PassiveScalar/test_scalars.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/PassiveScalar/test_scalars.cpp:void QuokkaSimulation<ScalarProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<Real, AMREX_SPACEDIM> const &dx,
src/problems/PassiveScalar/test_scalars.cpp:							       amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_lo)
src/problems/PassiveScalar/test_scalars.cpp:		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/PassiveScalar/test_scalars.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/PassiveScalar/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/PassiveScalar/CMakeLists.txt:    setup_target_for_cuda_compilation(test_scalars)
src/problems/PassiveScalar/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadLineCoolingMG/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadLineCoolingMG/CMakeLists.txt:    setup_target_for_cuda_compilation(test_rad_line_cooling_MG)
src/problems/RadLineCoolingMG/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadLineCoolingMG/test_rad_line_cooling_MG.cpp:constexpr amrex::GpuArray<double, 5> rad_boundaries_ = {1.00000000e-03, 1.77827941e-02, 3.16227766e-01, 5.62341325e+00, 1.00000000e+02};
src/problems/RadLineCoolingMG/test_rad_line_cooling_MG.cpp:	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
src/problems/RadLineCoolingMG/test_rad_line_cooling_MG.cpp:AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefinePhotoelectricHeatingE1Derivative(amrex::Real const /*temperature*/,
src/problems/RadLineCoolingMG/test_rad_line_cooling_MG.cpp:AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefineNetCoolingRate(amrex::Real const temperature, amrex::Real const /*num_density*/)
src/problems/RadLineCoolingMG/test_rad_line_cooling_MG.cpp:AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefineNetCoolingRateTempDerivative(amrex::Real const /*temperature*/, amrex::Real const /*num_density*/)
src/problems/RadLineCoolingMG/test_rad_line_cooling_MG.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefineCosmicRayHeatingRate(amrex::Real const /*num_density*/) -> double
src/problems/RadLineCoolingMG/test_rad_line_cooling_MG.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
src/problems/RadLineCoolingMG/test_rad_line_cooling_MG.cpp:RadSystem<CoolingProblemMG>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
src/problems/RadLineCoolingMG/test_rad_line_cooling_MG.cpp:								  const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
src/problems/RadLineCoolingMG/test_rad_line_cooling_MG.cpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
src/problems/RadLineCoolingMG/test_rad_line_cooling_MG.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadMarshakCGS/test_radiation_marshak_cgs.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SuOlsonProblemCgs>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadMarshakCGS/test_radiation_marshak_cgs.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SuOlsonProblemCgs>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadMarshakCGS/test_radiation_marshak_cgs.cpp:AMREX_GPU_HOST_DEVICE auto quokka::EOS<SuOlsonProblemCgs>::ComputeTgasFromEint(const double /*rho*/, const double Egas,
src/problems/RadMarshakCGS/test_radiation_marshak_cgs.cpp:									       std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
src/problems/RadMarshakCGS/test_radiation_marshak_cgs.cpp:AMREX_GPU_HOST_DEVICE auto quokka::EOS<SuOlsonProblemCgs>::ComputeEintFromTgas(const double /*rho*/, const double Tgas,
src/problems/RadMarshakCGS/test_radiation_marshak_cgs.cpp:									       std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
src/problems/RadMarshakCGS/test_radiation_marshak_cgs.cpp:AMREX_GPU_HOST_DEVICE auto
src/problems/RadMarshakCGS/test_radiation_marshak_cgs.cpp:							  std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/) -> double
src/problems/RadMarshakCGS/test_radiation_marshak_cgs.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadMarshakCGS/test_radiation_marshak_cgs.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadMarshakCGS/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadMarshakCGS/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radiation_marshak_cgs)
src/problems/RadMarshakCGS/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RayleighTaylor2D/test_hydro2d_rt.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/RayleighTaylor2D/test_hydro2d_rt.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RayleighTaylor2D/test_hydro2d_rt.cpp:	amrex::ParallelForRNG(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const &rng) noexcept {
src/problems/RayleighTaylor2D/test_hydro2d_rt.cpp:	amrex::ParallelFor(state_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/RayleighTaylor2D/test_hydro2d_rt.cpp:	amrex::Gpu::streamSynchronize();
src/problems/RayleighTaylor2D/test_hydro2d_rt.cpp:	amrex::ParallelFor(state_new_cc_[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/RayleighTaylor2D/test_hydro2d_rt.cpp:	amrex::Gpu::streamSynchronize();
src/problems/RayleighTaylor2D/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RayleighTaylor2D/CMakeLists.txt:        setup_target_for_cuda_compilation(test_hydro2d_rt)
src/problems/SphericalCollapse/spherical_collapse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/SphericalCollapse/spherical_collapse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/SphericalCollapse/spherical_collapse.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/SphericalCollapse/spherical_collapse.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/SphericalCollapse/spherical_collapse.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/SphericalCollapse/spherical_collapse.cpp:		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
src/problems/SphericalCollapse/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/SphericalCollapse/CMakeLists.txt:        setup_target_for_cuda_compilation(spherical_collapse)
src/problems/RadMatterCouplingRSLA/test_radiation_matter_coupling_rsla.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CouplingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadMatterCouplingRSLA/test_radiation_matter_coupling_rsla.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CouplingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadMatterCouplingRSLA/test_radiation_matter_coupling_rsla.cpp:AMREX_GPU_HOST_DEVICE auto quokka::EOS<CouplingProblem>::ComputeTgasFromEint(const double /*rho*/, const double Egas,
src/problems/RadMatterCouplingRSLA/test_radiation_matter_coupling_rsla.cpp:									     std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
src/problems/RadMatterCouplingRSLA/test_radiation_matter_coupling_rsla.cpp:AMREX_GPU_HOST_DEVICE auto quokka::EOS<CouplingProblem>::ComputeEintFromTgas(const double /*rho*/, const double Tgas,
src/problems/RadMatterCouplingRSLA/test_radiation_matter_coupling_rsla.cpp:									     std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
src/problems/RadMatterCouplingRSLA/test_radiation_matter_coupling_rsla.cpp:AMREX_GPU_HOST_DEVICE auto
src/problems/RadMatterCouplingRSLA/test_radiation_matter_coupling_rsla.cpp:							std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/) -> double
src/problems/RadMatterCouplingRSLA/test_radiation_matter_coupling_rsla.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadMatterCouplingRSLA/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadMatterCouplingRSLA/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radiation_matter_coupling_rsla)
src/problems/RadMatterCouplingRSLA/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadForce/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadForce/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radiation_force)
src/problems/RadForce/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadForce/test_radiation_force.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<TubeProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real { return 0.; }
src/problems/RadForce/test_radiation_force.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<TubeProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadForce/test_radiation_force.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RadForce/test_radiation_force.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadForce/test_radiation_force.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/RadLineCooling/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadLineCooling/CMakeLists.txt:    setup_target_for_cuda_compilation(test_rad_line_cooling)
src/problems/RadLineCooling/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadLineCooling/test_rad_line_cooling.cpp:AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::DefineNetCoolingRate(amrex::Real const temperature, amrex::Real const /*num_density*/)
src/problems/RadLineCooling/test_rad_line_cooling.cpp:AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::DefineNetCoolingRateTempDerivative(amrex::Real const /*temperature*/, amrex::Real const /*num_density*/)
src/problems/RadLineCooling/test_rad_line_cooling.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::DefineCosmicRayHeatingRate(amrex::Real const /*num_density*/) -> double
src/problems/RadLineCooling/test_rad_line_cooling.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<CoolingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadLineCooling/test_rad_line_cooling.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadLineCooling/test_rad_line_cooling.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
src/problems/RadLineCooling/test_rad_line_cooling.cpp:RadSystem<CoolingProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
src/problems/RadLineCooling/test_rad_line_cooling.cpp:								const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
src/problems/RadLineCooling/test_rad_line_cooling.cpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
src/problems/RadLineCooling/test_rad_line_cooling.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadDust/test_rad_dust.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputePlanckOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
src/problems/RadDust/test_rad_dust.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadDust/test_rad_dust.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputeThermalRadiationSingleGroup(amrex::Real temperature) -> amrex::Real
src/problems/RadDust/test_rad_dust.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputeThermalRadiationTempDerivativeSingleGroup(amrex::Real /*temperature*/) -> amrex::Real
src/problems/RadDust/test_rad_dust.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadDust/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadDust/CMakeLists.txt:    setup_target_for_cuda_compilation(test_rad_dust)
src/problems/RadDust/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/ShockCloud/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/ShockCloud/CMakeLists.txt:        setup_target_for_cuda_compilation(shock_cloud)
src/problems/ShockCloud/CMakeLists.txt:    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/ShockCloud/cloud.cpp:#include "AMReX_GpuDevice.H"
src/problems/ShockCloud/cloud.cpp:#include "AMReX_GpuQualifiers.H"
src/problems/ShockCloud/cloud.cpp:AMREX_GPU_MANAGED Real rho0 = NAN;	     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/ShockCloud/cloud.cpp:AMREX_GPU_MANAGED Real rho1 = NAN;	     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/ShockCloud/cloud.cpp:AMREX_GPU_MANAGED Real P0 = NAN;	     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/ShockCloud/cloud.cpp:AMREX_GPU_MANAGED Real R_cloud = NAN;	     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/ShockCloud/cloud.cpp:AMREX_GPU_MANAGED Real cloud_relpos_x = 0.5; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/ShockCloud/cloud.cpp:AMREX_GPU_MANAGED Real shock_crossing_time = 0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/ShockCloud/cloud.cpp:AMREX_GPU_MANAGED Real rho_wind = 0;		// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/ShockCloud/cloud.cpp:AMREX_GPU_MANAGED Real v_wind = 0;		// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/ShockCloud/cloud.cpp:AMREX_GPU_MANAGED Real P_wind = 0;		// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/ShockCloud/cloud.cpp:AMREX_GPU_MANAGED Real delta_vx = 0;		// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
src/problems/ShockCloud/cloud.cpp:	amrex::GpuArray<Real, AMREX_SPACEDIM> const dx = grid.dx_;
src/problems/ShockCloud/cloud.cpp:	amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = grid.prob_lo_;
src/problems/ShockCloud/cloud.cpp:	amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi = grid.prob_hi_;
src/problems/ShockCloud/cloud.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/ShockCloud/cloud.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<ShockCloud>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
src/problems/ShockCloud/cloud.cpp:		amrex::GpuArray<amrex::Real, HydroSystem<ShockCloud>::nscalars_> scalars{0, 0, rho};
src/problems/ShockCloud/cloud.cpp:			amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) noexcept {
src/problems/ShockCloud/cloud.cpp:		amrex::Gpu::streamSynchronizeAll();
src/problems/ShockCloud/cloud.cpp:		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/ShockCloud/cloud.cpp:		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/ShockCloud/cloud.cpp:		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/ShockCloud/cloud.cpp:		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/ShockCloud/cloud.cpp:		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/ShockCloud/cloud.cpp:		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/ShockCloud/cloud.cpp:		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/ShockCloud/cloud.cpp:		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/ShockCloud/cloud.cpp:		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/ShockCloud/cloud.cpp:		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/ShockCloud/cloud.cpp:	amrex::Gpu::streamSynchronizeAll();
src/problems/ShockCloud/cloud.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto ComputeCellTemp(int i, int j, int k, amrex::Array4<const Real> const &state, amrex::Real gamma,
src/problems/ShockCloud/cloud.cpp:							 quokka::TabulatedCooling::cloudyGpuConstTables const &tables)
src/problems/ShockCloud/cloud.cpp:	const Real M_cl_1e4 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	const Real M_cl_8000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	const Real M_cl_9000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	const Real M_cl_11000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	const Real M_cl_12000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	const Real origM_cl_1e4 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	const Real origM_cl_8000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	const Real origM_cl_9000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	const Real origM_cl_11000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	const Real origM_cl_12000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	const Real M_cl_scalar_01 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	const Real M_cl_scalar_01_09 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	const Real M_cl_fraction_01 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	const Real M_cl_fraction_01_09 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
src/problems/ShockCloud/cloud.cpp:	amrex::GpuArray<Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
src/problems/ShockCloud/cloud.cpp:	amrex::ParallelFor(state_new_cc_[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/problems/ShockCloud/cloud.cpp:	amrex::Gpu::streamSynchronize();
src/problems/RadMatterCoupling/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadMatterCoupling/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radiation_matter_coupling)
src/problems/RadMatterCoupling/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadMatterCoupling/test_radiation_matter_coupling.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CouplingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadMatterCoupling/test_radiation_matter_coupling.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CouplingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadMatterCoupling/test_radiation_matter_coupling.cpp:AMREX_GPU_HOST_DEVICE auto quokka::EOS<CouplingProblem>::ComputeTgasFromEint(const double /*rho*/, const double Egas,
src/problems/RadMatterCoupling/test_radiation_matter_coupling.cpp:									     std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
src/problems/RadMatterCoupling/test_radiation_matter_coupling.cpp:AMREX_GPU_HOST_DEVICE auto quokka::EOS<CouplingProblem>::ComputeEintFromTgas(const double /*rho*/, const double Tgas,
src/problems/RadMatterCoupling/test_radiation_matter_coupling.cpp:									     std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
src/problems/RadMatterCoupling/test_radiation_matter_coupling.cpp:AMREX_GPU_HOST_DEVICE auto
src/problems/RadMatterCoupling/test_radiation_matter_coupling.cpp:							std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/) -> double
src/problems/RadMatterCoupling/test_radiation_matter_coupling.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroShocktubeCMA/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroShocktubeCMA/CMakeLists.txt:    setup_target_for_cuda_compilation(test_hydro_shocktube_cma)
src/problems/HydroShocktubeCMA/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroShocktubeCMA/test_hydro_shocktube_cma.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
src/problems/HydroShocktubeCMA/test_hydro_shocktube_cma.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
src/problems/HydroShocktubeCMA/test_hydro_shocktube_cma.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroShocktubeCMA/test_hydro_shocktube_cma.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/HydroShocktubeCMA/test_hydro_shocktube_cma.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/HydroShocktubeCMA/test_hydro_shocktube_cma.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/HydroShocktubeCMA/test_hydro_shocktube_cma.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RadBeam/test_radiation_beam.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<BeamProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadBeam/test_radiation_beam.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<BeamProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadBeam/test_radiation_beam.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadBeam/test_radiation_beam.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/RadBeam/test_radiation_beam.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadBeam/test_radiation_beam.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RadBeam/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadBeam/CMakeLists.txt:        setup_target_for_cuda_compilation(test_radiation_beam)
src/problems/RadMarshakAsymptotic/test_radiation_marshak_asymptotic.cpp:template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::ComputePlanckOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadMarshakAsymptotic/test_radiation_marshak_asymptotic.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadMarshakAsymptotic/test_radiation_marshak_asymptotic.cpp:template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::ComputeEddingtonFactor(double /*f*/) -> double
src/problems/RadMarshakAsymptotic/test_radiation_marshak_asymptotic.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadMarshakAsymptotic/test_radiation_marshak_asymptotic.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadMarshakAsymptotic/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadMarshakAsymptotic/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radiation_marshak_asymptotic)
src/problems/RadMarshakAsymptotic/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroBB/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroBB/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radhydro_bb)
src/problems/RadhydroBB/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadhydroBB/test_radhydro_bb.cpp:constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_ = []() constexpr {
src/problems/RadhydroBB/test_radhydro_bb.cpp:		return amrex::GpuArray<double, 2>{0.0, inf};
src/problems/RadhydroBB/test_radhydro_bb.cpp:		return amrex::GpuArray<double, 5>{1.00000000e-03, 1.77827941e-02, 3.16227766e-01, 5.62341325e+00, 1.00000000e+02};
src/problems/RadhydroBB/test_radhydro_bb.cpp:		return amrex::GpuArray<double, 9>{1.00000000e-03, 4.21696503e-03, 1.77827941e-02, 7.49894209e-02, 3.16227766e-01,
src/problems/RadhydroBB/test_radhydro_bb.cpp:		return amrex::GpuArray<double, 17>{1.00000000e-03, 2.05352503e-03, 4.21696503e-03, 8.65964323e-03, 1.77827941e-02, 3.65174127e-02,
src/problems/RadhydroBB/test_radhydro_bb.cpp:		return amrex::GpuArray<double, 65>{
src/problems/RadhydroBB/test_radhydro_bb.cpp:	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
src/problems/RadhydroBB/test_radhydro_bb.cpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
src/problems/RadhydroBB/test_radhydro_bb.cpp:RadSystem<PulseProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
src/problems/RadhydroBB/test_radhydro_bb.cpp:							      const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
src/problems/RadhydroBB/test_radhydro_bb.cpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
src/problems/RadhydroBB/test_radhydro_bb.cpp:AMREX_GPU_HOST_DEVICE
src/problems/RadhydroBB/test_radhydro_bb.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroContact/test_hydro_contact.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/HydroContact/test_hydro_contact.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/HydroContact/test_hydro_contact.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroContact/test_hydro_contact.cpp:void QuokkaSimulation<ContactProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/problems/HydroContact/test_hydro_contact.cpp:								amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
src/problems/HydroContact/test_hydro_contact.cpp:		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/HydroContact/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroContact/CMakeLists.txt:    setup_target_for_cuda_compilation(test_hydro_contact)
src/problems/HydroContact/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadSuOlson/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadSuOlson/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radiation_SuOlson)
src/problems/RadSuOlson/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadSuOlson/test_radiation_SuOlson.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::ComputePlanckOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
src/problems/RadSuOlson/test_radiation_SuOlson.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadSuOlson/test_radiation_SuOlson.cpp:AMREX_GPU_HOST_DEVICE auto quokka::EOS<MarshakProblem>::ComputeTgasFromEint(const double /*rho*/, const double Egas,
src/problems/RadSuOlson/test_radiation_SuOlson.cpp:									    std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
src/problems/RadSuOlson/test_radiation_SuOlson.cpp:AMREX_GPU_HOST_DEVICE auto quokka::EOS<MarshakProblem>::ComputeEintFromTgas(const double /*rho*/, const double Tgas,
src/problems/RadSuOlson/test_radiation_SuOlson.cpp:									    std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
src/problems/RadSuOlson/test_radiation_SuOlson.cpp:AMREX_GPU_HOST_DEVICE auto
src/problems/RadSuOlson/test_radiation_SuOlson.cpp:						       std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/) -> double
src/problems/RadSuOlson/test_radiation_SuOlson.cpp:						   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/problems/RadSuOlson/test_radiation_SuOlson.cpp:						   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_lo*/,
src/problems/RadSuOlson/test_radiation_SuOlson.cpp:						   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real time)
src/problems/RadSuOlson/test_radiation_SuOlson.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadSuOlson/test_radiation_SuOlson.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroLeblanc/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroLeblanc/CMakeLists.txt:    setup_target_for_cuda_compilation(test_hydro_leblanc)
src/problems/HydroLeblanc/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:	amrex::GpuArray<int, 3> hi = box.hiVect3d();
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:void QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:	amrex::Gpu::DeviceVector<double> rho_g(density_exact_interp.size());
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:	amrex::Gpu::DeviceVector<double> vx_g(velocity_exact_interp.size());
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:	amrex::Gpu::DeviceVector<double> P_g(pressure_exact_interp.size());
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, density_exact_interp.begin(), density_exact_interp.end(), rho_g.begin());
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, velocity_exact_interp.begin(), velocity_exact_interp.end(), vx_g.begin());
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, pressure_exact_interp.begin(), pressure_exact_interp.end(), P_g.begin());
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:	amrex::Gpu::streamSynchronizeAll();
src/problems/HydroLeblanc/test_hydro_leblanc.cpp:		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RadMarshak/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadMarshak/CMakeLists.txt:    setup_target_for_cuda_compilation(test_radiation_marshak)
src/problems/RadMarshak/CMakeLists.txt:endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadMarshak/test_radiation_marshak.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SuOlsonProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadMarshak/test_radiation_marshak.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SuOlsonProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
src/problems/RadMarshak/test_radiation_marshak.cpp:AMREX_GPU_HOST_DEVICE auto quokka::EOS<SuOlsonProblem>::ComputeTgasFromEint(const double /*rho*/, const double Egas,
src/problems/RadMarshak/test_radiation_marshak.cpp:									    std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
src/problems/RadMarshak/test_radiation_marshak.cpp:AMREX_GPU_HOST_DEVICE auto quokka::EOS<SuOlsonProblem>::ComputeEintFromTgas(const double /*rho*/, const double Tgas,
src/problems/RadMarshak/test_radiation_marshak.cpp:									    std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
src/problems/RadMarshak/test_radiation_marshak.cpp:AMREX_GPU_HOST_DEVICE auto
src/problems/RadMarshak/test_radiation_marshak.cpp:						       std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/) -> double
src/problems/RadMarshak/test_radiation_marshak.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadMarshak/test_radiation_marshak.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroWave/CMakeLists.txt:if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroWave/CMakeLists.txt:    setup_target_for_cuda_compilation(test_hydro_wave)
src/problems/HydroWave/test_hydro_wave.cpp:AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/problems/HydroWave/test_hydro_wave.cpp:					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
src/problems/HydroWave/test_hydro_wave.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/HydroWave/test_hydro_wave.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/HydroWave/test_hydro_wave.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadShadow/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RadShadow/CMakeLists.txt:        setup_target_for_cuda_compilation(test_radiation_shadow)
src/problems/RadShadow/test_radiation_shadow.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShadowProblem>::ComputePlanckOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
src/problems/RadShadow/test_radiation_shadow.cpp:template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShadowProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
src/problems/RadShadow/test_radiation_shadow.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/problems/RadShadow/test_radiation_shadow.cpp:	amrex::GpuArray<int, 3> lo = box.loVect3d();
src/problems/RadShadow/test_radiation_shadow.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/RadShadow/test_radiation_shadow.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/RadShadow/test_radiation_shadow.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RadShadow/test_radiation_shadow.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RandomBlast/blast.cpp:#include "AMReX_GpuDevice.H"
src/problems/RandomBlast/blast.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/RandomBlast/blast.cpp:void injectEnergy(amrex::MultiFab &mf, amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_hi,
src/problems/RandomBlast/blast.cpp:		  amrex::GpuArray<Real, AMREX_SPACEDIM> const &dx, SimulationData<RandomBlast> const &userData)
src/problems/RandomBlast/blast.cpp:		auto kern = [=] AMREX_GPU_DEVICE(const Real x, const Real y, const Real z) {
src/problems/RandomBlast/blast.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RandomBlast/blast.cpp:			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RandomBlast/blast.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/RandomBlast/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/RandomBlast/CMakeLists.txt:        setup_target_for_cuda_compilation(random_blast)
src/problems/RandomBlast/CMakeLists.txt:    endif(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroBlast2D/test_hydro2d_blast.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
src/problems/HydroBlast2D/test_hydro2d_blast.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
src/problems/HydroBlast2D/test_hydro2d_blast.cpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
src/problems/HydroBlast2D/test_hydro2d_blast.cpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/problems/HydroBlast2D/test_hydro2d_blast.cpp:		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/problems/HydroBlast2D/CMakeLists.txt:    if(AMReX_GPU_BACKEND MATCHES "CUDA")
src/problems/HydroBlast2D/CMakeLists.txt:        setup_target_for_cuda_compilation(test_hydro2d_blast)
src/simulation.hpp:#include "AMReX_GpuQualifiers.H"
src/simulation.hpp:	AMREX_GPU_DEVICE static void setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int dcomp, int numcomp,
src/simulation.hpp:	AMREX_GPU_DEVICE static void setCustomBoundaryConditionsFaceVar(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int dcomp,
src/simulation.hpp:#ifdef AMREX_USE_GPU
src/simulation.hpp:				       << "  On GPUs, consider using 1-8 boxes per GPU per level that "
src/simulation.hpp:					  "together fill each GPU's memory sufficiently.\n"
src/simulation.hpp:#ifdef AMREX_USE_GPU
src/simulation.hpp:				  "greater) when running on GPUs, and 16 (or greater) when running on "
src/simulation.hpp:				  "128 (or greater) when running on GPUs, and 64 (or "
src/simulation.hpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();
src/simulation.hpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = geom[0].CellSizeArray();
src/simulation.hpp:	AMREX_GPU_DEVICE void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, const int &dcomp, const int &numcomp,
src/simulation.hpp:			amrex::ParallelFor(accel[lev], ng, AMREX_SPACEDIM, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int n) {
src/simulation.hpp:			amrex::Gpu::streamSynchronizeAll();
src/simulation.hpp:			amrex::GpuBndryFuncFab<setFunctorParticleAccel> boundaryFunctor(setFunctorParticleAccel{});
src/simulation.hpp:			amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setFunctorParticleAccel>> fineBdryFunct(geom[lev], accelBC, boundaryFunctor);
src/simulation.hpp:				amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setFunctorParticleAccel>> coarseBdryFunct(geom[lev - 1], accelBC, boundaryFunctor);
src/simulation.hpp:				amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
src/simulation.hpp:					    [=] AMREX_GPU_DEVICE(amrex::Array4<const amrex::Real> const &acc, int i, int j, int k, int comp) {
src/simulation.hpp:					    [=] AMREX_GPU_DEVICE(quokka::CICParticleContainer::ParticleType & p, int comp, amrex::Real acc_comp) {
src/simulation.hpp:				amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
src/simulation.hpp:				    geom[lev].CellSize(), dt_lev, amrex::RunOn::Gpu);
src/simulation.hpp:				    geom[lev].CellSize(), dt_lev, amrex::RunOn::Gpu);
src/simulation.hpp:					    geom[lev].CellSize(), dt_lev, amrex::RunOn::Gpu);
src/simulation.hpp:					    geom[lev].CellSize(), dt_lev, amrex::RunOn::Gpu);
src/simulation.hpp:	AMREX_GPU_DEVICE void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, const int &dcomp, const int &numcomp,
src/simulation.hpp:	AMREX_GPU_DEVICE void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, const int &dcomp, const int &numcomp,
src/simulation.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<problem_t>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest,
src/simulation.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
src/simulation.hpp:				amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor =
src/simulation.hpp:				    amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>{setBoundaryFunctor<problem_t>{}};
src/simulation.hpp:				amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> physicalBoundaryFunctor(geom[lev], BCs,
src/simulation.hpp:				amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>> boundaryFunctor =
src/simulation.hpp:				    amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>>{setBoundaryFunctorFaceVar<problem_t>{}};
src/simulation.hpp:				amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>>> physicalBoundaryFunctor(geom[lev], BCs,
src/simulation.hpp:	amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor_cc(setBoundaryFunctor<problem_t>{});
src/simulation.hpp:	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> finePhysicalBoundaryFunctor_cc(geom[lev], BCs, boundaryFunctor_cc);
src/simulation.hpp:	amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>> boundaryFunctor_fc(setBoundaryFunctorFaceVar<problem_t>{});
src/simulation.hpp:	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>>> finePhysicalBoundaryFunctor_fc(geom[lev], BCs, boundaryFunctor_fc);
src/simulation.hpp:		amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> coarsePhysicalBoundaryFunctor_cc(geom[lev - 1], BCs,
src/simulation.hpp:		amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctorFaceVar<problem_t>>> coarsePhysicalBoundaryFunctor_fc(geom[lev - 1], BCs,
src/simulation.hpp:	amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>> boundaryFunctor(setBoundaryFunctor<problem_t>{});
src/simulation.hpp:	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> finePhysicalBoundaryFunctor(geom[lev], BCs, boundaryFunctor);
src/simulation.hpp:	amrex::PhysBCFunct<amrex::GpuBndryFuncFab<setBoundaryFunctor<problem_t>>> coarsePhysicalBoundaryFunctor(geom[lev - 1], BCs, boundaryFunctor);
src/simulation.hpp:		amrex::ParallelFor(q[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) { result[bx](i, j, k) = user_f(i, j, k, state[bx]); });
src/simulation.hpp:	amrex::Gpu::streamSynchronize();
src/simulation.hpp:	amrex::ParallelFor(mf_cc, amrex::IntVect(AMREX_D_DECL(nGrow, nGrow, nGrow)), [=] AMREX_GPU_DEVICE(int boxidx, int i, int j, int k) {
src/simulation.hpp:	amrex::Gpu::streamSynchronize();
src/simulation.hpp:		amrex::ParallelFor(q[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) { result[bx](i, j, k) = user_f(i, j, k, state[bx]); });
src/simulation.hpp:	amrex::Gpu::streamSynchronize();
src/simulation.hpp:	    amrex::ReduceToPlane<ReduceOp, amrex::Real>(dir, domain_box, q[0], [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) -> amrex::Real {
src/simulation.hpp:	amrex::Gpu::streamSynchronize();
src/simulation.hpp:	amrex::Gpu::streamSynchronize();
src/main.cpp:		// Set GPU memory handling defaults:
src/main.cpp:		if (!pp.contains("abort_on_out_of_gpu_memory")) {
src/main.cpp:			pp.add("abort_on_out_of_gpu_memory", 1);
src/main.cpp:		//  for single-GPU runs, the overhead is completely negligible.
src/main.cpp:		//  HOWEVER, for multi-GPU runs, using managed memory disables the cuda_ipc
src/main.cpp:		//  transport and leads to *extremely poor* GPU-aware MPI performance.
src/main.cpp:		// use GPU-aware MPI
src/main.cpp:		//   GPU-aware MPI performance is, in fact, excellent.
src/main.cpp:		if (!pp.contains("use_gpu_aware_mpi")) {
src/main.cpp:			pp.add("use_gpu_aware_mpi", 1);
src/chemistry/Chemistry.hpp:#include "AMReX_GpuQualifiers.H"
src/chemistry/Chemistry.hpp:AMREX_GPU_DEVICE void chemburner(burn_t &chemstate, Real dt);
src/chemistry/Chemistry.hpp:	amrex::Gpu::Buffer<int> d_num_failed({0});
src/chemistry/Chemistry.hpp:		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/chemistry/Chemistry.hpp:				amrex::Gpu::Atomic::Add(p_num_failed, burn_failed);
src/chemistry/Chemistry.hpp:		amrex::Gpu::streamSynchronize(); // otherwise HIP may fail to allocate the necessary resources.
src/chemistry/Chemistry.cpp:AMREX_GPU_DEVICE void chemburner(burn_t &chemstate, const Real dt) { burner(chemstate, dt); }
src/linear_advection/AdvectionSimulation.hpp:	void computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/linear_advection/AdvectionSimulation.hpp:				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi);
src/linear_advection/AdvectionSimulation.hpp:void AdvectionSimulation<problem_t>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/linear_advection/AdvectionSimulation.hpp:							      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
src/linear_advection/AdvectionSimulation.hpp:							      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi)
src/linear_advection/AdvectionSimulation.hpp:				fluxes[i][iter].plus<amrex::RunOn::Gpu>(fluxArrays[i]);
src/linear_advection/AdvectionSimulation.hpp:					fluxes[i][iter].plus<amrex::RunOn::Gpu>(fluxArrays[i]);
src/linear_advection/AdvectionSimulation.hpp:	amrex::Gpu::streamSynchronizeAll();
src/linear_advection/linear_advection.hpp:	AMREX_GPU_DEVICE
src/linear_advection/linear_advection.hpp:				std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray, double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in,
src/linear_advection/linear_advection.hpp:				 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray, double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in,
src/linear_advection/linear_advection.hpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/linear_advection/linear_advection.hpp:	amrex::ParallelFor(primVar_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int n) { primVar[bx](i, j, k, n) = cons[bx](i, j, k, n); });
src/linear_advection/linear_advection.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto LinearAdvectionSystem<problem_t>::isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
src/linear_advection/linear_advection.hpp:						   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, const int nvars)
src/linear_advection/linear_advection.hpp:	amrex::ParallelFor(consVarNew_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/linear_advection/linear_advection.hpp:						    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, const int nvars)
src/linear_advection/linear_advection.hpp:	amrex::ParallelFor(U_new_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/linear_advection/linear_advection.hpp:	amrex::ParallelFor(x1Flux_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) noexcept {
src/util/CheckNaN.hpp:/// \brief Implements functions to check NaN values in arrays on the GPU.
src/util/CheckNaN.hpp:#include "AMReX_GpuQualifiers.H"
src/util/CheckNaN.hpp:AMREX_GPU_HOST_DEVICE auto CheckSymmetryArray(amrex::Array4<const amrex::Real> const & /*arr*/, amrex::Box const & /*indexRange*/, const int /*ncomp*/,
src/util/CheckNaN.hpp:					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> /*dx*/) -> bool
src/util/CheckNaN.hpp:AMREX_GPU_HOST_DEVICE auto CheckSymmetryFluxes(amrex::Array4<const amrex::Real> const & /*arr1*/, amrex::Array4<const amrex::Real> const & /*arr2*/,
src/util/CheckNaN.hpp:					       amrex::Box const & /*indexRange*/, const int /*ncomp*/, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> /*dx*/)
src/util/CheckNaN.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CheckNaN(amrex::FArrayBox const &arr, amrex::Box const & /*symmetryRange*/, amrex::Box const &nanRange,
src/util/CheckNaN.hpp:						       const int ncomp, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> /*dx*/)
src/util/CheckNaN.hpp:	AMREX_ASSERT(!arr.template contains_nan<amrex::RunOn::Gpu>(nanRange, 0, ncomp));
src/util/fextract.cpp:    -> std::tuple<Vector<Real>, Vector<Gpu::HostVector<Real>>>
src/util/fextract.cpp:	GpuArray<Real, AMREX_SPACEDIM> problo = geom.ProbLoArray();
src/util/fextract.cpp:	GpuArray<Real, AMREX_SPACEDIM> dx0 = geom.CellSizeArray();
src/util/fextract.cpp:	Vector<Gpu::HostVector<Real>> data(mf.nComp());
src/util/fextract.cpp:	GpuArray<Real, AMREX_SPACEDIM> dx = dx0;
src/util/fextract.cpp:				ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/util/fextract.cpp:					GpuArray<int, 3> idx_vec({i - lo0.x, j - lo0.y, k - lo0.z});
src/util/fextract.cpp:		Vector<Gpu::HostVector<Real>> alldata(data.size());
src/util/valarray.hpp:/// (This is necessary because std::valarray is not defined in CUDA C++!)
src/util/valarray.hpp:#include <AMReX_GpuQualifiers.H>
src/util/valarray.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE valarray() = default;
src/util/valarray.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE valarray(std::initializer_list<T> list) // NOLINT
src/util/valarray.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator[](size_t i) -> T & { return values[i]; }
src/util/valarray.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator[](size_t i) const -> T { return values[i]; }
src/util/valarray.hpp:	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto size() const -> size_t { return d; }
src/util/valarray.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void fillin(T const &scalar)
src/util/valarray.hpp:	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto hasnan() const -> bool
src/util/valarray.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<T, d>
src/util/valarray.hpp:template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+(quokka::valarray<T, d> const &v, T const &scalar) -> quokka::valarray<T, d>
src/util/valarray.hpp:template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+(T const &scalar, quokka::valarray<T, d> const &v) -> quokka::valarray<T, d>
src/util/valarray.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator-(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<T, d>
src/util/valarray.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<T, d>
src/util/valarray.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<T, d>
src/util/valarray.hpp:template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(T const &scalar, quokka::valarray<T, d> const &v) -> quokka::valarray<T, d>
src/util/valarray.hpp:template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(quokka::valarray<T, d> const &v, T const &scalar) -> quokka::valarray<T, d>
src/util/valarray.hpp:template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void operator*=(quokka::valarray<T, d> &v, T const &scalar)
src/util/valarray.hpp:template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void operator+=(quokka::valarray<T, d> &a, quokka::valarray<T, d> const &b)
src/util/valarray.hpp:template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(quokka::valarray<T, d> const &v, T const &scalar) -> quokka::valarray<T, d>
src/util/valarray.hpp:template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(T const &scalar, quokka::valarray<T, d> const &v) -> quokka::valarray<T, d>
src/util/valarray.hpp:template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void operator/=(quokka::valarray<T, d> &v, T const &scalar)
src/util/valarray.hpp:template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto abs(quokka::valarray<T, d> const &v) -> quokka::valarray<T, d>
src/util/valarray.hpp:template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto min(quokka::valarray<T, d> const &v) -> T
src/util/valarray.hpp:template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto max(quokka::valarray<T, d> const &v) -> T
src/util/valarray.hpp:template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto sum(quokka::valarray<T, d> const &v) -> T
src/util/valarray.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator>(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<bool, d>
src/util/valarray.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator>(quokka::valarray<T, d> const &a, T const &scalar) -> quokka::valarray<bool, d>
src/util/valarray.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator<(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<bool, d>
src/util/valarray.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator<(quokka::valarray<T, d> const &a, T const &scalar) -> quokka::valarray<bool, d>
src/util/ArrayView_2d.hpp:template <FluxDir N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex(int, int, int);
src/util/ArrayView_2d.hpp:template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex<FluxDir::X1>(int i, int j, int k) { return std::make_tuple(i, j, k); }
src/util/ArrayView_2d.hpp:template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex<FluxDir::X2>(int i, int j, int k) { return std::make_tuple(j, i, k); }
src/util/ArrayView_2d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}
src/util/ArrayView_2d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}
src/util/ArrayView_2d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T & { return arr_(i, j, k, n); }
src/util/ArrayView_2d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T & { return arr_(i, j, k); }
src/util/ArrayView_2d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}
src/util/ArrayView_2d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T { return arr_(i, j, k, n); }
src/util/ArrayView_2d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T { return arr_(i, j, k); }
src/util/ArrayView_2d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}
src/util/ArrayView_2d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T & { return arr_(j, i, k, n); }
src/util/ArrayView_2d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T & { return arr_(j, i, k); }
src/util/ArrayView_2d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}
src/util/ArrayView_2d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T { return arr_(j, i, k, n); }
src/util/ArrayView_2d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T { return arr_(j, i, k); }
src/util/ArrayView_3d.hpp:template <FluxDir N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex(int, int, int);
src/util/ArrayView_3d.hpp:template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex<FluxDir::X1>(int i, int j, int k) { return std::make_tuple(i, j, k); }
src/util/ArrayView_3d.hpp:template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex<FluxDir::X2>(int i, int j, int k) { return std::make_tuple(j, k, i); }
src/util/ArrayView_3d.hpp:template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto reorderMultiIndex<FluxDir::X3>(int i, int j, int k) { return std::make_tuple(k, i, j); }
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T & { return arr_(i, j, k, n); }
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T & { return arr_(i, j, k); }
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T { return arr_(i, j, k, n); }
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T { return arr_(i, j, k); }
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T & { return arr_(k, i, j, n); }
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T & { return arr_(k, i, j); }
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T { return arr_(k, i, j, n); }
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T { return arr_(k, i, j); }
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T & { return arr_(j, k, i, n); }
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T & { return arr_(j, k, i); }
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE explicit Array4View(amrex::Array4<T> arr) : arr_(arr) {}
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k, int n) const noexcept -> T { return arr_(j, k, i, n); }
src/util/ArrayView_3d.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator()(int i, int j, int k) const noexcept -> T { return arr_(j, k, i); }
src/util/fextract.hpp:    -> std::tuple<amrex::Vector<amrex::Real>, amrex::Vector<amrex::Gpu::HostVector<amrex::Real>>>;
src/grid.hpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_;
src/grid.hpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo_;
src/grid.hpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi_;
src/grid.hpp:	grid(amrex::Array4<double> const &array, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
src/grid.hpp:	     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi, centering cen, direction dir)
src/io/DiagFramePlane.cpp:		amrex::GpuArray const dxlcl = a_geoms[0].CellSizeArray();
src/io/DiagFramePlane.cpp:		amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_fieldIndices.begin(), m_fieldIndices.end(), m_fieldIndices_d.begin());
src/io/DiagFramePlane.cpp:		amrex::GpuArray const dx = a_geoms[lev].CellSizeArray();
src/io/DiagFramePlane.cpp:		amrex::GpuArray const problo = a_geoms[lev].ProbLoArray();
src/io/DiagFramePlane.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
src/io/DiagFramePlane.cpp:		for (amrex::MFIter mfi(planeData[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
src/io/DiagFramePlane.cpp:				amrex::ParallelFor(bx, m_fieldNames.size(), [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
src/io/DiagFramePlane.cpp:				amrex::ParallelFor(bx, m_fieldNames.size(), [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
src/io/DiagFramePlane.cpp:				amrex::ParallelFor(bx, m_fieldNames.size(), [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
src/io/DiagFramePlane.cpp:#ifdef AMREX_USE_GPU
src/io/DiagFramePlane.cpp:					amrex::Gpu::dtoh_memcpy_async(hostfab->dataPtr(), fab.dataPtr(), fab.size() * sizeof(amrex::Real));
src/io/DiagFramePlane.cpp:					amrex::Gpu::streamSynchronize();
src/io/DiagFramePlane.cpp:#ifdef AMREX_USE_GPU
src/io/DiagFramePlane.cpp:					amrex::Gpu::dtoh_memcpy_async(hostfab->dataPtr(), fab.dataPtr(), fab.size() * sizeof(amrex::Real));
src/io/DiagFramePlane.cpp:					amrex::Gpu::streamSynchronize();
src/io/DiagPDF.cpp:#include "AMReX_GpuContainers.H"
src/io/DiagPDF.cpp:AMREX_GPU_HOST_DEVICE AMREX_INLINE auto DiagPDF::getBinIndex1D(const amrex::Real &realInputVal, const amrex::Real &transformedLowBnd,
src/io/DiagPDF.cpp:AMREX_GPU_HOST_DEVICE AMREX_INLINE auto DiagPDF::getTotalBinCount() -> amrex::Long
src/io/DiagPDF.cpp:	amrex::Gpu::DeviceVector<amrex::Real> pdf_d(getTotalBinCount(), 0.0);
src/io/DiagPDF.cpp:		    *a_state[lev], amrex::IntVect(0), [=, nFilters = m_filters.size()] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
src/io/DiagPDF.cpp:		amrex::Gpu::streamSynchronize();
src/io/DiagPDF.cpp:		amrex::Gpu::DeviceVector<int> idx_d(nvars);
src/io/DiagPDF.cpp:		amrex::Gpu::DeviceVector<int> nbins_d(nvars);
src/io/DiagPDF.cpp:		amrex::Gpu::DeviceVector<int> doLog_d(nvars);
src/io/DiagPDF.cpp:		amrex::Gpu::DeviceVector<amrex::Real> lowBnd_d(nvars);
src/io/DiagPDF.cpp:		amrex::Gpu::DeviceVector<amrex::Real> binWidth_d(nvars);
src/io/DiagPDF.cpp:		amrex::Gpu::copy(amrex::Gpu::hostToDevice, fieldIdx.begin(), fieldIdx.end(), idx_d.begin());
src/io/DiagPDF.cpp:		amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_nBins.begin(), m_nBins.end(), nbins_d.begin());
src/io/DiagPDF.cpp:		amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_useLogSpacedBins.begin(), m_useLogSpacedBins.end(), doLog_d.begin());
src/io/DiagPDF.cpp:		amrex::Gpu::copy(amrex::Gpu::hostToDevice, transformed_lowBnd.begin(), transformed_lowBnd.end(), lowBnd_d.begin());
src/io/DiagPDF.cpp:		amrex::Gpu::copy(amrex::Gpu::hostToDevice, transformed_binWidth.begin(), transformed_binWidth.end(), binWidth_d.begin());
src/io/DiagPDF.cpp:		amrex::Gpu::streamSynchronize();
src/io/DiagPDF.cpp:		amrex::ParallelFor(*a_state[lev], amrex::IntVect(0), [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
src/io/DiagPDF.cpp:		amrex::Gpu::streamSynchronize();
src/io/DiagPDF.cpp:	amrex::Gpu::copy(amrex::Gpu::deviceToHost, pdf_d.begin(), pdf_d.end(), pdf.begin());
src/io/DiagPDF.cpp:	amrex::Gpu::streamSynchronize();
src/io/DiagPDF.H:	AMREX_GPU_HOST_DEVICE AMREX_INLINE static auto getBinIndex1D(const amrex::Real &realInputVal, const amrex::Real &transformedLowBnd,
src/io/DiagPDF.H:	AMREX_GPU_HOST_DEVICE AMREX_INLINE auto getTotalBinCount() -> amrex::Long;
src/io/DiagFramePlane.H:	amrex::Gpu::DeviceVector<int> m_fieldIndices_d;
src/io/DiagFramePlane.H:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_center;
src/io/DiagFramePlane.H:	amrex::Vector<amrex::GpuArray<amrex::Real, 3>> m_intwgt;
src/io/DiagBase.H:	amrex::Gpu::DeviceVector<DiagFilterData> m_filterData;
src/io/DiagBase.cpp:		amrex::Gpu::copy(amrex::Gpu::hostToDevice, hostFilterData.begin(), hostFilterData.end(), m_filterData.begin());
src/CMakeLists.txt:  set(AMReX_CUDA_FASTMATH OFF CACHE BOOL "" FORCE)
src/CMakeLists.txt:  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
src/CMakeLists.txt:    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
src/CMakeLists.txt:  if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
src/CMakeLists.txt:    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-ffp-contract=off>)
src/CMakeLists.txt:# emit register usage per thread from CUDA assembler
src/CMakeLists.txt:# if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
src/CMakeLists.txt:#   add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
src/CMakeLists.txt:# if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
src/CMakeLists.txt:#   add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcuda-ptxas>)
src/CMakeLists.txt:#   add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-v>)
src/CMakeLists.txt:#   add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-unused-command-line-argument>)
src/hydro/hydro_system.hpp:	AMREX_GPU_DEVICE static auto ComputePrimVars(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> quokka::valarray<amrex::Real, nvar_>;
src/hydro/hydro_system.hpp:	AMREX_GPU_DEVICE static auto ComputeConsVars(quokka::valarray<amrex::Real, nvar_> const &prim) -> quokka::valarray<amrex::Real, nvar_>;
src/hydro/hydro_system.hpp:	AMREX_GPU_DEVICE static auto ComputePressure(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;
src/hydro/hydro_system.hpp:	AMREX_GPU_DEVICE static auto ComputeSoundSpeed(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;
src/hydro/hydro_system.hpp:	AMREX_GPU_DEVICE static auto ComputeVelocityX1(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;
src/hydro/hydro_system.hpp:	AMREX_GPU_DEVICE static auto ComputeVelocityX2(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;
src/hydro/hydro_system.hpp:	AMREX_GPU_DEVICE static auto ComputeVelocityX3(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> amrex::Real;
src/hydro/hydro_system.hpp:	AMREX_GPU_DEVICE static auto isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool;
src/hydro/hydro_system.hpp:					 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int nvars);
src/hydro/hydro_system.hpp:	AMREX_GPU_DEVICE static auto GetGradFixedPotential(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec) -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>;
src/hydro/hydro_system.hpp:	static void AddInternalEnergyPdV(amrex::MultiFab &rhs_mf, amrex::MultiFab const &consVar_mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
src/hydro/hydro_system.hpp:	amrex::ParallelFor(cons_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
src/hydro/hydro_system.hpp:				[=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept -> amrex::GpuTuple<amrex::Real> {
src/hydro/hydro_system.hpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/hydro/hydro_system.hpp:				[=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept -> amrex::GpuTuple<bool> {
src/hydro/hydro_system.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputePrimVars(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
src/hydro/hydro_system.hpp:		amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
src/hydro/hydro_system.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeConsVars(quokka::valarray<amrex::Real, nvar_> const &prim)
src/hydro/hydro_system.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputePressure(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
src/hydro/hydro_system.hpp:		amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
src/hydro/hydro_system.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeSoundSpeed(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
src/hydro/hydro_system.hpp:	amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
src/hydro/hydro_system.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeVelocityX1(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
src/hydro/hydro_system.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeVelocityX2(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
src/hydro/hydro_system.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::ComputeVelocityX3(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
src/hydro/hydro_system.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<problem_t>::isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool
src/hydro/hydro_system.hpp:		amrex::GpuArray<Real, nmscalars_> massScalars_ = RadSystem<problem_t>::ComputeMassScalars(cons, i, j, k);
src/hydro/hydro_system.hpp:						  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const int nvars)
src/hydro/hydro_system.hpp:	amrex::ParallelFor(rhs_mf, amrex::IntVect{0}, nvars, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int n) noexcept {
src/hydro/hydro_system.hpp:	amrex::ParallelFor(consVarNew_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/hydro/hydro_system.hpp:	amrex::ParallelFor(Unew_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/hydro/hydro_system.hpp:	amrex::ParallelFor(primVar_mf, ng, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in) {
src/hydro/hydro_system.hpp:			amrex::GpuArray<Real, nmscalars_> massScalars_plus2 = RadSystem<problem_t>::ComputeMassScalars(primVar, i + 2, j, k);
src/hydro/hydro_system.hpp:			amrex::GpuArray<Real, nmscalars_> massScalars_plus1 = RadSystem<problem_t>::ComputeMassScalars(primVar, i + 1, j, k);
src/hydro/hydro_system.hpp:			amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(primVar, i, j, k);
src/hydro/hydro_system.hpp:			amrex::GpuArray<Real, nmscalars_> massScalars_minus1 = RadSystem<problem_t>::ComputeMassScalars(primVar, i - 1, j, k);
src/hydro/hydro_system.hpp:			amrex::GpuArray<Real, nmscalars_> massScalars_minus2 = RadSystem<problem_t>::ComputeMassScalars(primVar, i - 2, j, k);
src/hydro/hydro_system.hpp:		amrex::GpuArray<Real, nmscalars_> massScalars = RadSystem<problem_t>::ComputeMassScalars(primVar, i, j, k);
src/hydro/hydro_system.hpp:	amrex::ParallelFor(q_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) {
src/hydro/hydro_system.hpp:	amrex::ParallelFor(state_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/hydro/hydro_system.hpp:			amrex::GpuArray<Real, nmscalars_> const massScalars = RadSystem<problem_t>::ComputeMassScalars(state[bx], i, j, k);
src/hydro/hydro_system.hpp:						  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
src/hydro/hydro_system.hpp:	amrex::ParallelFor(rhs_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
src/hydro/hydro_system.hpp:	amrex::ParallelFor(consVar_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
src/hydro/hydro_system.hpp:	amrex::ParallelFor(x1Flux_mf, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in) {
src/hydro/hydro_system.hpp:				amrex::GpuArray<Real, nmscalars_> massScalars_L = RadSystem<problem_t>::ComputeMassScalars(x1LeftState, i, j, k);
src/hydro/hydro_system.hpp:				amrex::GpuArray<Real, nmscalars_> massScalars_R = RadSystem<problem_t>::ComputeMassScalars(x1RightState, i, j, k);
src/hydro/hydro_system.hpp:			amrex::GpuArray<Real, nmscalars_> massScalars_L = RadSystem<problem_t>::ComputeMassScalars(x1LeftState, i, j, k);
src/hydro/hydro_system.hpp:			amrex::GpuArray<Real, nmscalars_> massScalars_R = RadSystem<problem_t>::ComputeMassScalars(x1RightState, i, j, k);
src/hydro/NSCBC_inflow.hpp:// Quokka -- two-moment radiation hydrodynamics on GPUs for astrophysics
src/hydro/NSCBC_inflow.hpp:#include "AMReX_GpuQualifiers.H"
src/hydro/NSCBC_inflow.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dQ_dx_inflow_x1_lower(quokka::valarray<Real, HydroSystem<problem_t>::nvar_> const &Q,
src/hydro/NSCBC_inflow.hpp:							       amrex::GpuArray<Real, HydroSystem<problem_t>::nscalars_> const &s_t, const Real L_x)
src/hydro/NSCBC_inflow.hpp:	amrex::GpuArray<Real, HydroSystem<problem_t>::nscalars_> s{};
src/hydro/NSCBC_inflow.hpp:	amrex::GpuArray<Real, nmscalars_> massScalars;
src/hydro/NSCBC_inflow.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setInflowX1Lower(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom,
src/hydro/NSCBC_inflow.hpp:							  amrex::GpuArray<Real, HydroSystem<problem_t>::nscalars_> const &s_t)
src/hydro/NSCBC_inflow.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setInflowX1LowerLowOrder(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
src/hydro/NSCBC_inflow.hpp:								  amrex::GpuArray<Real, HydroSystem<problem_t>::nscalars_> const &s_t)
src/hydro/NSCBC_inflow.hpp:	amrex::GpuArray<Real, nmscalars_> massScalars;
src/hydro/HydroState.hpp:	amrex::GpuArray<double, Nmass> massScalar; // mass scalars
src/hydro/EOS.hpp:#include "AMReX_GpuQualifiers.H"
src/hydro/EOS.hpp:	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
src/hydro/EOS.hpp:	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real;
src/hydro/EOS.hpp:	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
src/hydro/EOS.hpp:	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real;
src/hydro/EOS.hpp:	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
src/hydro/EOS.hpp:	ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
src/hydro/EOS.hpp:	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
src/hydro/EOS.hpp:	ComputeEintTempDerivative(amrex::Real rho, amrex::Real Tgas, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
src/hydro/EOS.hpp:	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
src/hydro/EOS.hpp:	ComputeOtherDerivatives(amrex::Real rho, amrex::Real P, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {});
src/hydro/EOS.hpp:	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
src/hydro/EOS.hpp:	ComputePressure(amrex::Real rho, amrex::Real Eint, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real;
src/hydro/EOS.hpp:	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
src/hydro/EOS.hpp:	ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
src/hydro/EOS.hpp:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint,
src/hydro/EOS.hpp:										  std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars)
src/hydro/EOS.hpp:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas,
src/hydro/EOS.hpp:										  std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars)
src/hydro/EOS.hpp:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure,
src/hydro/EOS.hpp:										  std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars)
src/hydro/EOS.hpp:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto
src/hydro/EOS.hpp:					  std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars) -> amrex::Real
src/hydro/EOS.hpp:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto
src/hydro/EOS.hpp:EOS<problem_t>::ComputeOtherDerivatives(const amrex::Real rho, const amrex::Real P, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars)
src/hydro/EOS.hpp:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputePressure(amrex::Real rho, amrex::Real Eint,
src/hydro/EOS.hpp:									      std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars)
src/hydro/EOS.hpp:AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure,
src/hydro/EOS.hpp:										std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars)
src/hydro/NSCBC_outflow.hpp:// Quokka -- two-moment radiation hydrodynamics on GPUs for astrophysics
src/hydro/NSCBC_outflow.hpp:#include "AMReX_GpuQualifiers.H"
src/hydro/NSCBC_outflow.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dQ_dx_outflow(quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &Q,
src/hydro/NSCBC_outflow.hpp:	amrex::GpuArray<Real, nmscalars_> massScalars;
src/hydro/NSCBC_outflow.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto transverse_xdir_dQ_data(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
src/hydro/NSCBC_outflow.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto transverse_ydir_dQ_data(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
src/hydro/NSCBC_outflow.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto transverse_zdir_dQ_data(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
src/hydro/NSCBC_outflow.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto permute_vel(quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &Q)
src/hydro/NSCBC_outflow.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto unpermute_vel(quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &Q)
src/hydro/NSCBC_outflow.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setOutflowBoundary(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
src/hydro/NSCBC_outflow.hpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setOutflowBoundaryLowOrder(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
src/hydro/LLF.hpp:#include "AMReX_GpuQualifiers.H"
src/hydro/LLF.hpp:AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto LLF(quokka::HydroState<N_scalars, N_mscalars> const &sL, quokka::HydroState<N_scalars, N_mscalars> const &sR)
src/hydro/HLLC.hpp:#include "AMReX_GpuQualifiers.H"
src/hydro/HLLC.hpp:AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto HLLC(quokka::HydroState<N_scalars, N_mscalars> const &sL, quokka::HydroState<N_scalars, N_mscalars> const &sR,
src/hydro/HLLD.hpp:#include "AMReX_GpuQualifiers.H"
src/hydro/HLLD.hpp:AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto FastMagnetoSonicSpeed(double gamma, quokka::HydroState<N_scalars, N_mscalars> const state, const double bx) -> double
src/hydro/HLLD.hpp:AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto HLLD(quokka::HydroState<N_scalars, N_mscalars> const &sL, quokka::HydroState<N_scalars, N_mscalars> const &sR,
src/particles/CICParticles.hpp:	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const CICParticleContainer::ParticleType &p, amrex::Array4<amrex::Real> const &rho,
src/particles/CICParticles.hpp:							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
src/particles/CICParticles.hpp:							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
src/particles/CICParticles.hpp:				      [=] AMREX_GPU_DEVICE(const CICParticleContainer::ParticleType &part, int comp) {
src/radiation/planck_integral.hpp:#include "AMReX_GpuQualifiers.H"
src/radiation/planck_integral.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate_planck_integral(Real logx) -> Real
src/radiation/planck_integral.hpp:	const amrex::GpuArray<Real, INTERP_SIZE> Y_interp = {
src/radiation/planck_integral.hpp:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto integrate_planck_from_0_to_x(const Real x) -> Real
src/radiation/radiation_system.hpp:#include "AMReX_GpuQualifiers.H"
src/radiation/radiation_system.hpp:	static constexpr amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups + 1> radBoundaries = {0., inf};
src/radiation/radiation_system.hpp:	amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups> delta_nu_kappa_B_at_edge; // Delta (nu * kappa * B)
src/radiation/radiation_system.hpp:	amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups> alpha_P;
src/radiation/radiation_system.hpp:	amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups> alpha_E;
src/radiation/radiation_system.hpp:	amrex::GpuArray<double, 3> gasMomentum;							   // gas momentum
src/radiation/radiation_system.hpp:	amrex::GpuArray<amrex::GpuArray<amrex::Real, Physics_Traits<problem_t>::nGroups>, 3> Frad; // radiation flux
src/radiation/radiation_system.hpp:[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto minmod_func(double a, double b) -> double
src/radiation/radiation_system.hpp:	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto MC(double a, double b) -> double
src/radiation/radiation_system.hpp:	static constexpr amrex::GpuArray<double, nGroups_ + 1> radBoundaries_ = []() constexpr {
src/radiation/radiation_system.hpp:			amrex::GpuArray<double, 2> boundaries{0., inf};
src/radiation/radiation_system.hpp:	static void PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
src/radiation/radiation_system.hpp:				amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArray, double dt_in,
src/radiation/radiation_system.hpp:				amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars);
src/radiation/radiation_system.hpp:	static void AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrayOld,
src/radiation/radiation_system.hpp:				 amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArrayOld,
src/radiation/radiation_system.hpp:				 amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArray, double dt_in,
src/radiation/radiation_system.hpp:				 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars);
src/radiation/radiation_system.hpp:				  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, bool use_wavespeed_correction);
src/radiation/radiation_system.hpp:	static void SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/radiation/radiation_system.hpp:				       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi,
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto UpdateFlux(int i, int j, int k, arrayconst_t const &consPrev, NewtonIterationResult<problem_t> &energy, double dt,
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto ComputeMassScalars(ArrayType const &arr, int i, int j, int k) -> amrex::GpuArray<Real, nmscalars_>;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeEddingtonFactor(double f) -> double;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeNumberDensityH(double rho, amrex::GpuArray<Real, nmscalars_> const &massScalars) -> double;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputePlanckOpacity(double rho, double Tgas) -> Real;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeFluxMeanOpacity(double rho, double Tgas) -> Real;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeEnergyMeanOpacity(double rho, double Tgas) -> Real;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, double rho, double Tgas)
src/radiation/radiation_system.hpp:	    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeGroupMeanOpacity(amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> const &kappa_expo_and_lower_value,
src/radiation/radiation_system.hpp:								  amrex::GpuArray<double, nGroups_> const &radBoundaryRatios,
src/radiation/radiation_system.hpp:								  amrex::GpuArray<double, nGroups_> const &alpha_quant) -> quokka::valarray<double, nGroups_>;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeBinCenterOpacity(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries,
src/radiation/radiation_system.hpp:								  amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value)
src/radiation/radiation_system.hpp:	// AMREX_GPU_HOST_DEVICE static auto
src/radiation/radiation_system.hpp:	// ComputeGroupMeanOpacityWithMinusOneSlope(amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value,
src/radiation/radiation_system.hpp:	// 					 amrex::GpuArray<double, nGroups_> radBoundaryRatios) -> quokka::valarray<double, nGroups_>;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeEintFromEgas(double density, double X1GasMom, double X2GasMom, double X3GasMom, double Etot) -> double;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeEgasFromEint(double density, double X1GasMom, double X2GasMom, double X3GasMom, double Eint) -> double;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto PlanckFunction(double nu, double T) -> double;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto
src/radiation/radiation_system.hpp:					quokka::valarray<double, nGroups_> fourPiBoverC, amrex::GpuArray<double, nGroups_> delta_nu_kappa_B_at_edge,
src/radiation/radiation_system.hpp:					amrex::GpuArray<double, nGroups_> delta_nu_B_at_edge, amrex::GpuArray<double, nGroups_ + 1> kappa_slope)
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeFluxInDiffusionLimit(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, double T, double vel)
src/radiation/radiation_system.hpp:	    -> amrex::GpuArray<double, nGroups_>;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeRadQuantityExponents(ArrayType const &quant, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
src/radiation/radiation_system.hpp:	    -> amrex::GpuArray<double, nGroups_>;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static void SolveLinearEqs(JacobianResult<problem_t> const &jacobian, double &x0, quokka::valarray<double, nGroups_> &xi);
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static void SolveLinearEqsWithLastColumn(JacobianResult<problem_t> const &jacobian, double &x0,
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto Solve3x3matrix(double C00, double C01, double C02, double C10, double C11, double C12, double C20, double C21,
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputePlanckEnergyFractions(amrex::GpuArray<double, nGroups_ + 1> const &boundaries, amrex::Real temperature)
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationSingleGroup(amrex::Real temperature) -> double;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationMultiGroup(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationTempDerivativeSingleGroup(amrex::Real temperature) -> Real;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationTempDerivativeMultiGroup(amrex::Real temperature,
src/radiation/radiation_system.hpp:											  amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto BackwardEulerOneVariable(RHSFunction const &rhs, JacFunction const &jac, double x0, double compare) -> double;
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto
src/radiation/radiation_system.hpp:				       amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries = amrex::GpuArray<double, nGroups_ + 1>{}) -> double;
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto
src/radiation/radiation_system.hpp:				      amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries = amrex::GpuArray<double, nGroups_ + 1>{},
src/radiation/radiation_system.hpp:				      amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios = amrex::GpuArray<double, nGroups_>{}) -> double;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto DefinePhotoelectricHeatingE1Derivative(amrex::Real temperature, amrex::Real num_density) -> amrex::Real;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto DefineBackgroundHeatingRate(amrex::Real num_density) -> amrex::Real;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto DefineNetCoolingRate(amrex::Real temperature, amrex::Real num_density) -> quokka::valarray<double, nGroups_>;
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto DefineNetCoolingRateTempDerivative(amrex::Real temperature, amrex::Real num_density)
src/radiation/radiation_system.hpp:	AMREX_GPU_HOST_DEVICE static auto DefineCosmicRayHeatingRate(amrex::Real num_density) -> double;
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static void ComputeModelDependentKappaFAndDeltaTerms(double T, double rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto ComputeModelDependentKappaEAndKappaP(double T, double rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
src/radiation/radiation_system.hpp:									  amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios,
src/radiation/radiation_system.hpp:									  amrex::GpuArray<double, nGroups_> const &alpha_E = {},
src/radiation/radiation_system.hpp:									  amrex::GpuArray<double, nGroups_> const &alpha_P = {}) -> OpacityTerms<problem_t>;
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto ComputeJacobianForGas(double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff,
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto ComputeJacobianForGasAndDust(double T_gas, double T_d, double Egas_diff,
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto ComputeJacobianForGasAndDustDecoupled(
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto ComputeJacobianForGasAndDustWithPE(
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto
src/radiation/radiation_system.hpp:					amrex::GpuArray<Real, nmscalars_> const &massScalars, int n_outer_iter, quokka::valarray<double, nGroups_> const &work,
src/radiation/radiation_system.hpp:					amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, int *p_iteration_counter, int *p_iteration_failure_counter)
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto SolveGasDustRadiationEnergyExchange(double Egas0, quokka::valarray<double, nGroups_> const &Erad0Vec, double rho,
src/radiation/radiation_system.hpp:									 double coeff_n, double dt, amrex::GpuArray<Real, nmscalars_> const &massScalars,
src/radiation/radiation_system.hpp:									 amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, int *p_iteration_counter,
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto
src/radiation/radiation_system.hpp:						  amrex::GpuArray<Real, nmscalars_> const &massScalars, int n_outer_iter,
src/radiation/radiation_system.hpp:						  quokka::valarray<double, nGroups_> const &Src, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto ComputeCellOpticalDepth(const quokka::Array4View<const amrex::Real, DIR> &consVar,
src/radiation/radiation_system.hpp:							     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i, int j, int k,
src/radiation/radiation_system.hpp:							     const amrex::GpuArray<double, nGroups_ + 1> &group_boundaries)
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto isStateValid(std::array<amrex::Real, nvarHyperbolic_> &cons) -> bool;
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static void amendRadState(std::array<amrex::Real, nvarHyperbolic_> &cons);
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto ComputeRadPressure(double erad_L, double Fx_L, double Fy_L, double Fz_L, double fx_L, double fy_L, double fz_L)
src/radiation/radiation_system.hpp:	AMREX_GPU_DEVICE static auto ComputeEddingtonTensor(double fx_L, double fy_L, double fz_L) -> std::array<std::array<double, 3>, 3>;
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputePlanckEnergyFractions(amrex::GpuArray<double, nGroups_ + 1> const &boundaries, amrex::Real temperature)
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeNumberDensityH(double rho, amrex::GpuArray<Real, nmscalars_> const & /*massScalars*/) -> double
src/radiation/radiation_system.hpp:template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationSingleGroup(amrex::Real temperature) -> Real
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationMultiGroup(amrex::Real temperature,
src/radiation/radiation_system.hpp:										   amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
src/radiation/radiation_system.hpp:template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationTempDerivativeSingleGroup(amrex::Real temperature) -> Real
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationTempDerivativeMultiGroup(amrex::Real temperature,
src/radiation/radiation_system.hpp:												 amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
src/radiation/radiation_system.hpp:template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineBackgroundHeatingRate(amrex::Real const /*num_density*/) -> amrex::Real
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineNetCoolingRate(amrex::Real const /*temperature*/, amrex::Real const /*num_density*/)
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineNetCoolingRateTempDerivative(amrex::Real const /*temperature*/, amrex::Real const /*num_density*/)
src/radiation/radiation_system.hpp:template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineCosmicRayHeatingRate(amrex::Real const /*num_density*/) -> double
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE void RadSystem<problem_t>::SolveLinearEqs(JacobianResult<problem_t> const &jacobian, double &x0, quokka::valarray<double, nGroups_> &xi)
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::Solve3x3matrix(const double C00, const double C01, const double C02, const double C10, const double C11,
src/radiation/radiation_system.hpp:void RadSystem<problem_t>::SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/radiation/radiation_system.hpp:					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
src/radiation/radiation_system.hpp:					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real time)
src/radiation/radiation_system.hpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/radiation/radiation_system.hpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/radiation/radiation_system.hpp:template <typename problem_t> AMREX_GPU_DEVICE auto RadSystem<problem_t>::isStateValid(std::array<amrex::Real, nvarHyperbolic_> &cons) -> bool
src/radiation/radiation_system.hpp:template <typename problem_t> AMREX_GPU_DEVICE void RadSystem<problem_t>::amendRadState(std::array<amrex::Real, nvarHyperbolic_> &cons)
src/radiation/radiation_system.hpp:void RadSystem<problem_t>::PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
src/radiation/radiation_system.hpp:				       amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArray*/, const double dt_in,
src/radiation/radiation_system.hpp:				       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int /*nvars*/)
src/radiation/radiation_system.hpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/radiation/radiation_system.hpp:void RadSystem<problem_t>::AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrayOld,
src/radiation/radiation_system.hpp:					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
src/radiation/radiation_system.hpp:					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArrayOld*/,
src/radiation/radiation_system.hpp:					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArray*/, const double dt_in,
src/radiation/radiation_system.hpp:					amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int /*nvars*/)
src/radiation/radiation_system.hpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/radiation/radiation_system.hpp:template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEddingtonFactor(double f_in) -> double
src/radiation/radiation_system.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeMassScalars(ArrayType const &arr, int i, int j, int k) -> amrex::GpuArray<Real, nmscalars_>
src/radiation/radiation_system.hpp:	amrex::GpuArray<Real, nmscalars_> massScalars{};
src/radiation/radiation_system.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeCellOpticalDepth(const quokka::Array4View<const amrex::Real, DIR> &consVar,
src/radiation/radiation_system.hpp:								    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i, int j, int k,
src/radiation/radiation_system.hpp:								    const amrex::GpuArray<double, nGroups_ + 1> &group_boundaries)
src/radiation/radiation_system.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeEddingtonTensor(const double fx, const double fy, const double fz) -> std::array<std::array<double, 3>, 3>
src/radiation/radiation_system.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeRadPressure(const double erad, const double Fx, const double Fy, const double Fz, const double fx,
src/radiation/radiation_system.hpp:					 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, bool const use_wavespeed_correction)
src/radiation/radiation_system.hpp:	amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;
src/radiation/radiation_system.hpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in) {
src/radiation/radiation_system.hpp:		amrex::GpuArray<double, nGroups_ + 1> radBoundaries_g_copy{};
src/radiation/radiation_system.hpp:template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> Real
src/radiation/radiation_system.hpp:template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> Real
src/radiation/radiation_system.hpp:template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEnergyMeanOpacity(const double rho, const double Tgas) -> Real
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/,
src/radiation/radiation_system.hpp:    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
src/radiation/radiation_system.hpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeRadQuantityExponents(ArrayType const &quant, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
src/radiation/radiation_system.hpp:    -> amrex::GpuArray<double, nGroups_>
src/radiation/radiation_system.hpp:	amrex::GpuArray<double, nGroups_> bin_center{};
src/radiation/radiation_system.hpp:	amrex::GpuArray<double, nGroups_> quant_mean{};
src/radiation/radiation_system.hpp:	amrex::GpuArray<double, nGroups_ - 1> logslopes{};
src/radiation/radiation_system.hpp:	amrex::GpuArray<double, nGroups_> exponents{};
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto
src/radiation/radiation_system.hpp:RadSystem<problem_t>::ComputeGroupMeanOpacity(amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> const &kappa_expo_and_lower_value,
src/radiation/radiation_system.hpp:					      amrex::GpuArray<double, nGroups_> const &radBoundaryRatios, amrex::GpuArray<double, nGroups_> const &alpha_quant)
src/radiation/radiation_system.hpp:	amrex::GpuArray<double, nGroups_ + 1> const &alpha_kappa = kappa_expo_and_lower_value[0];
src/radiation/radiation_system.hpp:	amrex::GpuArray<double, nGroups_ + 1> const &kappa_lower = kappa_expo_and_lower_value[1];
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEintFromEgas(const double density, const double X1GasMom, const double X2GasMom, const double X3GasMom,
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEgasFromEint(const double density, const double X1GasMom, const double X2GasMom, const double X3GasMom,
src/radiation/radiation_system.hpp:template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::PlanckFunction(const double nu, const double T) -> double
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeDiffusionFluxMeanOpacity(const quokka::valarray<double, nGroups_> kappaPVec,
src/radiation/radiation_system.hpp:										 const amrex::GpuArray<double, nGroups_> delta_nu_kappa_B_at_edge,
src/radiation/radiation_system.hpp:										 const amrex::GpuArray<double, nGroups_> delta_nu_B_at_edge,
src/radiation/radiation_system.hpp:										 const amrex::GpuArray<double, nGroups_ + 1> kappa_slope)
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeBinCenterOpacity(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries,
src/radiation/radiation_system.hpp:									 amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value)
src/radiation/radiation_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeFluxInDiffusionLimit(const amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, const double T,
src/radiation/radiation_system.hpp:									     const double vel) -> amrex::GpuArray<double, nGroups_>
src/radiation/radiation_system.hpp:	amrex::GpuArray<double, nGroups_ + 1> edge_values{};
src/radiation/radiation_system.hpp:	amrex::GpuArray<double, nGroups_> flux{};
src/radiation/radiation_system.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::BackwardEulerOneVariable(RHSFunction const &rhs, JacFunction const &jac, const double x0, const double compare)
src/radiation/radiation_system.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeDustTemperatureBateKeto(double const T_gas, double const T_d_init, double const rho,
src/radiation/radiation_system.hpp:									   int n_step, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries) -> double
src/radiation/radiation_system.hpp:	amrex::GpuArray<double, nGroups_> rad_boundary_ratios{};
src/radiation/radiation_dust_system.hpp:AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefinePhotoelectricHeatingE1Derivative(amrex::Real const /*temperature*/, amrex::Real const /*num_density*/)
src/radiation/radiation_dust_system.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGasAndDust(
src/radiation/radiation_dust_system.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGasAndDustDecoupled(
src/radiation/radiation_dust_system.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGasAndDustWithPE(
src/radiation/radiation_dust_system.hpp:AMREX_GPU_HOST_DEVICE void RadSystem<problem_t>::SolveLinearEqsWithLastColumn(JacobianResult<problem_t> const &jacobian, double &x0,
src/radiation/radiation_dust_system.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveGasDustRadiationEnergyExchange(
src/radiation/radiation_dust_system.hpp:    amrex::GpuArray<Real, nmscalars_> const &massScalars, int const n_outer_iter, quokka::valarray<double, nGroups_> const &work,
src/radiation/radiation_dust_system.hpp:    amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, int *p_iteration_counter, int *p_iteration_failure_counter) -> NewtonIterationResult<problem_t>
src/radiation/radiation_dust_system.hpp:		amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[1], 1); // NOLINT
src/radiation/radiation_dust_system.hpp:	amrex::GpuArray<double, nGroups_> rad_boundary_ratios{};
src/radiation/radiation_dust_system.hpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value{};
src/radiation/radiation_dust_system.hpp:	amrex::GpuArray<double, nGroups_> alpha_quant_minus_one{};
src/radiation/radiation_dust_system.hpp:			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[1], 1); // NOLINT
src/radiation/radiation_dust_system.hpp:		// TODO(CCH): potential GPU-related issue here.
src/radiation/radiation_dust_system.hpp:		amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[0], 1); // NOLINT
src/radiation/radiation_dust_system.hpp:	amrex::Gpu::Atomic::Add(&p_iteration_counter[0], 1);	 // total number of radiation updates. NOLINT
src/radiation/radiation_dust_system.hpp:	amrex::Gpu::Atomic::Add(&p_iteration_counter[1], n + 1); // total number of Newton-Raphson iterations. NOLINT
src/radiation/radiation_dust_system.hpp:	amrex::Gpu::Atomic::Max(&p_iteration_counter[2], n + 1); // maximum number of Newton-Raphson iterations. NOLINT
src/radiation/radiation_dust_system.hpp:		amrex::Gpu::Atomic::Add(&p_iteration_counter[3], 1); // total number of decoupled gas-dust iterations. NOLINT
src/radiation/radiation_dust_system.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveGasDustRadiationEnergyExchangeWithPE(
src/radiation/radiation_dust_system.hpp:    amrex::GpuArray<Real, nmscalars_> const &massScalars, int const n_outer_iter, quokka::valarray<double, nGroups_> const &work,
src/radiation/radiation_dust_system.hpp:    amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, int *p_iteration_counter, int *p_iteration_failure_counter) -> NewtonIterationResult<problem_t>
src/radiation/radiation_dust_system.hpp:		amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[1], 1); // NOLINT
src/radiation/radiation_dust_system.hpp:	amrex::GpuArray<double, nGroups_> rad_boundary_ratios{};
src/radiation/radiation_dust_system.hpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value{};
src/radiation/radiation_dust_system.hpp:	amrex::GpuArray<double, nGroups_> alpha_quant_minus_one{};
src/radiation/radiation_dust_system.hpp:			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[1], 1); // NOLINT
src/radiation/radiation_dust_system.hpp:		amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[0], 1); // NOLINT
src/radiation/radiation_dust_system.hpp:	amrex::Gpu::Atomic::Add(&p_iteration_counter[0], 1);	 // total number of radiation updates. NOLINT
src/radiation/radiation_dust_system.hpp:	amrex::Gpu::Atomic::Add(&p_iteration_counter[1], n + 1); // total number of Newton-Raphson iterations. NOLINT
src/radiation/radiation_dust_system.hpp:	amrex::Gpu::Atomic::Max(&p_iteration_counter[2], n + 1); // maximum number of Newton-Raphson iterations. NOLINT
src/radiation/radiation_dust_system.hpp:		amrex::Gpu::Atomic::Add(&p_iteration_counter[3], 1); // total number of decoupled gas-dust iterations. NOLINT
src/radiation/source_terms_single_group.hpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/radiation/source_terms_single_group.hpp:		amrex::GpuArray<Real, 3> dMomentum{};
src/radiation/source_terms_single_group.hpp:		amrex::GpuArray<Real, 3> Frad_t1{};
src/radiation/source_terms_single_group.hpp:							amrex::Gpu::Atomic::Add(&p_iteration_failure_counter_local[1], 1); // NOLINT
src/radiation/source_terms_single_group.hpp:					amrex::Gpu::Atomic::Add(&p_iteration_failure_counter_local[0], 1); // NOLINT
src/radiation/source_terms_single_group.hpp:				amrex::Gpu::Atomic::Add(&p_iteration_counter_local[0], 1);     // total number of radiation updates. NOLINT
src/radiation/source_terms_single_group.hpp:				amrex::Gpu::Atomic::Add(&p_iteration_counter_local[1], n + 1); // total number of Newton-Raphson iterations. NOLINT
src/radiation/source_terms_single_group.hpp:				amrex::Gpu::Atomic::Max(&p_iteration_counter_local[2], n + 1); // maximum number of Newton-Raphson iterations. NOLINT
src/radiation/source_terms_single_group.hpp:					// TODO(CCH): potential GPU-related issue here.
src/radiation/source_terms_single_group.hpp:			amrex::GpuArray<amrex::Real, 3> Frad_t0{};
src/radiation/source_terms_single_group.hpp:			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter_local[2], 1); // NOLINT
src/radiation/source_terms_multi_group.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeModelDependentKappaEAndKappaP(
src/radiation/source_terms_multi_group.hpp:    double const T, double const rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios,
src/radiation/source_terms_multi_group.hpp:    amrex::GpuArray<double, nGroups_> const &alpha_E, amrex::GpuArray<double, nGroups_> const &alpha_P) -> OpacityTerms<problem_t>
src/radiation/source_terms_multi_group.hpp:		amrex::GpuArray<double, nGroups_> alpha_quant_minus_one{};
src/radiation/source_terms_multi_group.hpp:AMREX_GPU_DEVICE void
src/radiation/source_terms_multi_group.hpp:RadSystem<problem_t>::ComputeModelDependentKappaFAndDeltaTerms(double const T, double const rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
src/radiation/source_terms_multi_group.hpp:	amrex::GpuArray<double, nGroups_> delta_nu_B_at_edge{};
src/radiation/source_terms_multi_group.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGas(double /*T_d*/, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff,
src/radiation/source_terms_multi_group.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveGasRadiationEnergyExchange(
src/radiation/source_terms_multi_group.hpp:    amrex::GpuArray<Real, nmscalars_> const &massScalars, int const n_outer_iter, quokka::valarray<double, nGroups_> const &work,
src/radiation/source_terms_multi_group.hpp:    amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, int *p_iteration_counter, int *p_iteration_failure_counter) -> NewtonIterationResult<problem_t>
src/radiation/source_terms_multi_group.hpp:	amrex::GpuArray<double, nGroups_> rad_boundary_ratios{};
src/radiation/source_terms_multi_group.hpp:	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value{};
src/radiation/source_terms_multi_group.hpp:	amrex::GpuArray<double, nGroups_> alpha_quant_minus_one{};
src/radiation/source_terms_multi_group.hpp:		amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[0], 1); // NOLINT
src/radiation/source_terms_multi_group.hpp:	amrex::Gpu::Atomic::Add(&p_iteration_counter[0], 1);	 // total number of radiation updates. NOLINT
src/radiation/source_terms_multi_group.hpp:	amrex::Gpu::Atomic::Add(&p_iteration_counter[1], n + 1); // total number of Newton-Raphson iterations. NOLINT
src/radiation/source_terms_multi_group.hpp:	amrex::Gpu::Atomic::Max(&p_iteration_counter[2], n + 1); // maximum number of Newton-Raphson iterations. NOLINT
src/radiation/source_terms_multi_group.hpp:AMREX_GPU_DEVICE auto RadSystem<problem_t>::UpdateFlux(int const i, int const j, int const k, arrayconst_t &consPrev, NewtonIterationResult<problem_t> &energy,
src/radiation/source_terms_multi_group.hpp:	amrex::GpuArray<amrex::Real, 3> Frad_t0{};
src/radiation/source_terms_multi_group.hpp:	amrex::GpuArray<amrex::Real, 3> dMomentum{0., 0., 0.};
src/radiation/source_terms_multi_group.hpp:	amrex::GpuArray<amrex::GpuArray<amrex::Real, nGroups_>, 3> Frad_t1{};
src/radiation/source_terms_multi_group.hpp:	amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;
src/radiation/source_terms_multi_group.hpp:	amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;
src/radiation/source_terms_multi_group.hpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
src/radiation/source_terms_multi_group.hpp:		amrex::GpuArray<double, nGroups_ + 1> radBoundaries_g_copy{};
src/radiation/source_terms_multi_group.hpp:		amrex::GpuArray<double, nGroups_> radBoundaryRatios_copy{};
src/radiation/source_terms_multi_group.hpp:		amrex::GpuArray<double, nGroups_> alpha_quant_minus_one{};
src/radiation/source_terms_multi_group.hpp:			amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value{};
src/radiation/source_terms_multi_group.hpp:			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter_local[2], 1); // NOLINT
src/QuokkaSimulation.hpp:#include "AMReX_GpuControl.H"
src/QuokkaSimulation.hpp:#include "AMReX_GpuDevice.H"
src/QuokkaSimulation.hpp:#include "AMReX_GpuQualifiers.H"
src/QuokkaSimulation.hpp:	void computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/QuokkaSimulation.hpp:				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo);
src/QuokkaSimulation.hpp:	template <typename F> auto computeAxisAlignedProfile(int axis, F const &user_f) -> amrex::Gpu::HostVector<amrex::Real>;
src/QuokkaSimulation.hpp:				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
src/QuokkaSimulation.hpp:				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, int *p_iteration_counter, int *p_iteration_failure_counter);
src/QuokkaSimulation.hpp:				    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
src/QuokkaSimulation.hpp:			  const amrex::Box &indexRange, int nvars, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);
src/QuokkaSimulation.hpp:				amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/QuokkaSimulation.hpp:void QuokkaSimulation<problem_t>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/QuokkaSimulation.hpp:							   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
src/QuokkaSimulation.hpp:	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = geom[0].CellSizeArray();
src/QuokkaSimulation.hpp:	amrex::ParallelFor(rhs_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/QuokkaSimulation.hpp:	amrex::Gpu::streamSynchronizeAll();
src/QuokkaSimulation.hpp:		amrex::ParallelFor(phi_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
src/QuokkaSimulation.hpp:	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
src/QuokkaSimulation.hpp:	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
src/QuokkaSimulation.hpp:auto QuokkaSimulation<problem_t>::computeAxisAlignedProfile(const int axis, F const &user_f) -> amrex::Gpu::HostVector<amrex::Real>
src/QuokkaSimulation.hpp:			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) { result(i, j, k) = user_f(i, j, k, state); });
src/QuokkaSimulation.hpp:		amrex::Gpu::streamSynchronizeAll(); // just in case
src/QuokkaSimulation.hpp:			amrex::Gpu::streamSynchronizeAll(); // just in case
src/QuokkaSimulation.hpp:	amrex::Gpu::streamSynchronizeAll();
src/QuokkaSimulation.hpp:		amrex::Gpu::streamSynchronizeAll(); // just in case
src/QuokkaSimulation.hpp:			amrex::Gpu::streamSynchronizeAll(); // just in case
src/QuokkaSimulation.hpp:	amrex::Gpu::streamSynchronizeAll();
src/QuokkaSimulation.hpp:		amrex::ParallelFor(redoFlag, ng, ncomp, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int n) noexcept {
src/QuokkaSimulation.hpp:	amrex::Gpu::streamSynchronizeAll();
src/QuokkaSimulation.hpp:	amrex::Gpu::streamSynchronizeAll();
src/QuokkaSimulation.hpp:		amrex::Gpu::Buffer<int> iteration_failure_counter({0, 0, 0});
src/QuokkaSimulation.hpp:		amrex::Gpu::Buffer<int> iteration_counter({0, 0, 0, 0});
src/QuokkaSimulation.hpp:							   const double dt, const int stage, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
src/QuokkaSimulation.hpp:							   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
src/QuokkaSimulation.hpp:							   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, int *p_iteration_counter,
src/QuokkaSimulation.hpp:							 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
src/QuokkaSimulation.hpp:					       const amrex::Box &indexRange, const int nvars, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
src/hyperbolic_system.hpp:	template <SlopeLimiter limiter> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto SlopeFunc(amrex::Real x, amrex::Real y) -> amrex::Real
src/hyperbolic_system.hpp:	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto MC(double a, double b) -> double
src/hyperbolic_system.hpp:	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto minmod(double a, double b) -> double
src/hyperbolic_system.hpp:	[[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto GetMinmaxSurroundingCell(arrayconst_t &q, int i, int j, int k, int n)
src/hyperbolic_system.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesConstant(arrayconst_t &q, array_t &leftState, array_t &rightState,
src/hyperbolic_system.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
src/hyperbolic_system.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPLM(arrayconst_t &q, array_t &leftState, array_t &rightState,
src/hyperbolic_system.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
src/hyperbolic_system.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPPM(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
src/hyperbolic_system.hpp:	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPPM(quokka::Array4View<amrex::Real const, DIR> const &q,
src/hyperbolic_system.hpp:		     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars, F &&isStateValid,
src/hyperbolic_system.hpp:		    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars, F &&isStateValid,
src/hyperbolic_system.hpp:	amrex::ParallelFor(q_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) noexcept {
src/hyperbolic_system.hpp:AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesConstant(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
src/hyperbolic_system.hpp:	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
src/hyperbolic_system.hpp:AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesConstant(quokka::Array4View<amrex::Real const, DIR> const &q,
src/hyperbolic_system.hpp:	amrex::ParallelFor(q_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) noexcept {
src/hyperbolic_system.hpp:AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPLM(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
src/hyperbolic_system.hpp:	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
src/hyperbolic_system.hpp:AMREX_GPU_HOST_DEVICE void
src/hyperbolic_system.hpp:AMREX_GPU_DEVICE auto HyperbolicSystem<problem_t>::GetMinmaxSurroundingCell(arrayconst_t &q, int i, int j, int k, int n) -> std::pair<double, double>
src/hyperbolic_system.hpp:	amrex::ParallelFor(q_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) noexcept {
src/hyperbolic_system.hpp:AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPPM(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
src/hyperbolic_system.hpp:	amrex::ParallelFor(cellRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
src/hyperbolic_system.hpp:AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPPM(quokka::Array4View<amrex::Real const, DIR> const &q,
src/hyperbolic_system.hpp:	// N.B.: Checking all 27 nearest neighbors is *very* expensive on GPU
src/hyperbolic_system.hpp:					      const double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange,
src/hyperbolic_system.hpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/hyperbolic_system.hpp:					       const double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange,
src/hyperbolic_system.hpp:	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
src/turbulence/TurbDataReader.cpp:	amrex::GpuArray<int, 3> const lo{0, 0, 0};
src/turbulence/TurbDataReader.cpp:	amrex::GpuArray<int, 3> const hi{static_cast<int>(dims[0]), static_cast<int>(dims[1]), static_cast<int>(dims[2])};

```
