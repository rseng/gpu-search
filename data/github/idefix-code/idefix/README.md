# https://github.com/idefix-code/idefix

```console
pytools/idfx_test.py:    parser.add_argument("-cuda",
pytools/idfx_test.py:                        help="Test on Nvidia GPU using CUDA",
pytools/idfx_test.py:                        help="Test on AMD GPU using HIP",
pytools/idfx_test.py:    if self.cuda:
pytools/idfx_test.py:      comm.append("-DKokkos_ENABLE_CUDA=ON")
pytools/idfx_test.py:      # disable fmad operations on Cuda to make it compatible with CPU arithmetics
pytools/idfx_test.py:      # disable Async cuda malloc for tests performed on old UCX implementations
pytools/idfx_test.py:      comm.append("-DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF")
pytools/idfx_test.py:    if "Kokkos CUDA target ENABLED" in log:
pytools/idfx_test.py:      self.cuda = True
pytools/idfx_test.py:      self.cuda = False
pytools/idfx_test.py:    if self.cuda or self.hip:
pytools/idfx_test.py:    if self.cuda:
pytools/idfx_test.py:      print("Nvidia Cuda enabled.")
test/HD/ShearingBox/setup.cpp:  // GPUS cannot capture static variables
test/Dust/StreamingInstability/setup.cpp:  // GPUS cannot capture static variables
test/utils/lookupTable/main.cpp:  // When running on GPUS with Omnipath network,
test/utils/lookupTable/main.cpp:#ifdef KOKKOS_ENABLE_CUDA
test/utils/lookupTable/main.cpp:  if(std::getenv("PSM2_CUDA") != NULL) {
test/MHD/AmbipolarShearingBox/setup.cpp:    // GPUS cannot capture static variables
test/MHD/ShearingBox/setup.cpp:  // GPUS cannot capture static variables
CHANGELOG.md:- Configuration for Nvidia H100 on Jean Zay in the documentation
CHANGELOG.md:- Add CUDA_MALLOC_ASYNC flags in Jean Zay documentation to deal with MPI issues when using Kokkos 4.3 (#248)
CHANGELOG.md:- fixed a bug that led to race condition when using GPU offloading, axis boundary condition and domain decomposition along X3 (!309)
CHANGELOG.md:- Check that the MPI library is GPU-aware when using a GPU backend (!262)
CHANGELOG.md:- use buffers for mpi axis exchanges to improve performances on GPUs (!195)
CHANGELOG.md:- fixed the many warning messages when compiling on CUDA (!229)
CHANGELOG.md:- fixed a bug which resulted in a failure at detecting NaNs on some GPU architectures
CHANGELOG.md:- auto-detect HIP GPU offloading (used for AMD GPUs)
CHANGELOG.md:- deprecate the `-gpu` option in `configure.py`. The GPU mode is now automatically activated if a GPU architecture is requested.
doc/source/faq.rst:Cmake fails with "CMake wants to use -std=c++1z which is not supported by NVCC" when configuring with Cuda
doc/source/faq.rst:  ``-std=c++1z`` to enable c++17, but this flag is not recognised by Cuda ``nvcc`` compiler. Use
doc/source/faq.rst:I want to run on the GPUs of xxx machine, how do I proceed?
doc/source/performances.rst:MPI sub-domain on GPUs or 32\ :sup:`3` per MPI sub-domain on CPUs. All of the performances measures
doc/source/performances.rst:have been obtained enabling MPI on *one full node*, but we report here the performance *per GPU*
doc/source/performances.rst:(i.e. with 2 GCDs on AMD Mi250) or *per core* (on CPU), i.e. dividing the node performance by the number of GPU/core
doc/source/performances.rst:    slower performances with lower resolution when using GPUs. The overall performances also depends on
doc/source/performances.rst:GPU performances
doc/source/performances.rst:| Cluster name         | GPU                | Performances (in 10\ :sup:`6` cell/s/GPU)          |
doc/source/performances.rst:| IDRIS/Jean Zay       | NVIDIA V100        | 110                                                |
doc/source/performances.rst:| IDRIS/Jean Zay       | NVIDIA A100        | 194                                                |
doc/source/index.rst:*Idefix* is designed to be a performance-portable astrophysical code. This means that it can run both on your laptop's cpu or on the largest GPU HPCs recently
doc/source/index.rst:bought by your university. More technically, *Idefix* can run in serial, use OpenMP and/or MPI (message passing interface) for parallelization, and use CUDA or Xeon-Phi for
doc/source/index.rst:  Clang on both Intel and AMD CPUs. *Idefix* has also been tested on NVIDIA GPUs (Pascal, Volta and Ampere architectures) using the nvcc (>10) compiler, and on AMD GPUs (Radeon Mi50, Mi210, Mi250) using the hipcc compiler.
doc/source/index.rst:  When using MPI parallelisation, *Idefix* relies on an external MPI library. *Idefix* has been tested successfully with OpenMPI and IntelMPI libraries. When used on GPU architectures, *Idefix* assumes that
doc/source/index.rst:  the MPI library is GPU-Aware. If unsure, check this last point with your system administrator.
doc/source/index.rst:* Multiple parallelisation strategies (OpenMP, MPI, GPU offloading, etc...)
doc/source/modules/selfGravity.rst:either on CPU or GPU.
doc/source/reference/idefix.ini.rst:|                |                    | | Note that Nan checks are slow on GPUs, and low values of ``check_nan`` are not recommended.             |
doc/source/reference/commandline.rst:| --kokkos-num-devices=x   | | Specify the number of devices (eg CUDA GPU) Kokkos should expect in each compute node. This option is used by   |
doc/source/reference/makefile.rst:    Enable MPI parallelisation. Requires an MPI library. When used in conjonction with CUDA (Nvidia GPUs), a CUDA-aware MPI library is required by *Idefix*.
doc/source/reference/makefile.rst:``-D Kokkos_ENABLE_CUDA=ON``
doc/source/reference/makefile.rst:    Enable Nvidia Cuda (for GPU targets). When enabled, ``cmake`` will attempt to auto-detect the target GPU architecture. If this fails, one needs to specify
doc/source/reference/makefile.rst:    simulatenously (e.g for a CPU and a GPU). For instance:
doc/source/reference/makefile.rst:      + NVIDIA GPUs: PASCAL60, PASCAL61, VOLTA70, VOLTA72, AMPERE80, AMPERE86, ...
doc/source/reference/makefile.rst:      + AMD GPUs: VEGA906, VEGA908...
doc/source/reference/makefile.rst:AdAstra at CINES, AMD Mi250X GPUs
doc/source/reference/makefile.rst:    module load rocm/5.7.1 # nécessaire a cause d'un bug de path pas encore fix..
doc/source/reference/makefile.rst:MPI (multi-GPU) can be enabled by adding ``-DIdefix_MPI=ON`` as usual.
doc/source/reference/makefile.rst:Jean Zay at IDRIS, Nvidia V100/A100/H100 GPUs
doc/source/reference/makefile.rst:    module load cuda/12.1.0
doc/source/reference/makefile.rst:    module load openmpi/4.1.1-cuda
doc/source/reference/makefile.rst:    module load cuda/12.1.0
doc/source/reference/makefile.rst:    module load openmpi/4.1.5-cuda
doc/source/reference/makefile.rst:*Idefix* can then be configured to run on Nvidia V100 with the following options to ccmake:
doc/source/reference/makefile.rst:    -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON
doc/source/reference/makefile.rst:While Ampere A100 GPUs are enabled with
doc/source/reference/makefile.rst:    -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
doc/source/reference/makefile.rst:And for H100 GPUS:
doc/source/reference/makefile.rst:    -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON
doc/source/reference/makefile.rst:MPI (multi-GPU) can be enabled by adding ``-DIdefix_MPI=ON`` as usual.
doc/source/reference/makefile.rst:  As of *Idefix* 2.1.02, we automatically disable Cuda Malloc async (``-DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF``). However, earlier versions of
doc/source/reference/makefile.rst:  *Idefix* requires this flag when calling cmake to prevent a bug when using PSM2 with async Cuda malloc possibly leading to openmpi crash or hangs on Jean Zay.
doc/source/programmingguide.rst:Because *Idefix* can run on GPUs, and since GPUs experience a significant speedup when working
doc/source/programmingguide.rst:on some GPU architecture, but is not recommended for production runs as it can have an impact on the precision or even
doc/source/programmingguide.rst:accelerator (e.g. a GPU) and is actually performing the computation (or most of it).
doc/source/programmingguide.rst:complex structures. Moreover, a bug in the Nvidia Cuda compiler ``nvcc`` prevents Cuda lambdas
doc/source/programmingguide.rst:make local copies of the class members before using them in loops, to keep compatibility with Cuda
doc/source/programmingguide.rst:  GPU specific segmentation faults.
doc/source/programmingguide.rst:Note that when running on GPU architectures, reductions are particularly inefficient operations. If possible,
doc/source/programmingguide.rst:running on GPUs.
doc/source/programmingguide.rst:        "Speeding particle(s) detected !", // this default error message is used on GPUs
README.md:  * [serial (gpu/cpu), openMP (cpu)](#serial-gpucpu-openmp-cpu)
README.md:  * [With MPI (gpu)](#with-mpi-gpu)
README.md:Idefix is a computational fluid dynamics code based on a finite-volume high-order Godunov method, originally designed for astrophysical fluid dynamics applications.  Idefix is designed to be performance-portable, and uses the [Kokkos](https://github.com/kokkos/kokkos) framework to achieve this goal. This means that it can run both on your laptop's cpu and on the largest GPU Exascale clusters. More technically, Idefix can run in serial, use OpenMP and/or MPI (message passing interface) for parallelization, and use GPU acceleration when available (based on Nvidia Cuda, AMD HIP, etc...). All these capabilities are embedded within one single code, so the code relies on relatively abstracted classes and objects available in C++17, which are not necessarily
README.md:* Multiple parallelisation strategies (OpenMP, MPI, GPU offloading, etc...)
README.md:### serial (gpu/cpu), openMP (cpu)
README.md:### With MPI (gpu)
README.md:The same rules for cpu domain decomposition applies for gpus. In addition, one should manually specify how many GPU devices one wants to use **per node**. Example, in a run with 2 nodes, 4 gpu per node, one would launch idefix with
CMakeLists.txt:#Idefix requires Cuda Lambdas (experimental)
CMakeLists.txt:if(Kokkos_ENABLE_CUDA)
CMakeLists.txt:  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Idefix requires lambdas on Cuda" FORCE)
CMakeLists.txt:  set(Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC OFF CACHE BOOL "Disable Async malloc to avoid bugs on PSM2" FORCE)
CMakeLists.txt:  if(Kokkos_ENABLE_CUDA)
CMakeLists.txt:    add_compile_options(-Xcudafe --promote_warnings)
CMakeLists.txt:  if(Kokkos_ENABLE_CUDA)
CMakeLists.txt:    message(ERROR "SIMD loop pattern is incompatible with Cuda")
src/main.cpp:  // When running on GPUS with Omnipath network,
src/main.cpp:#ifdef KOKKOS_ENABLE_CUDA
src/main.cpp:  if(std::getenv("PSM2_CUDA") != NULL) {
src/loop.hpp:  #elif defined(KOKKOS_ENABLE_CUDA)
src/timeIntegrator.cpp:    // Look for Nans every now and then (this actually cost a lot of time on GPUs
src/mpi.hpp:  // Check that MPI will work with the designated target (in particular GPU Direct)
src/fluid/viscosity.hpp:  // Function for internal use (but public to allow for Cuda lambda capture)
src/fluid/viscosity.cpp:// but since constructors cannot be Lambda-captured by cuda
src/fluid/boundary/axis.hpp:  void ExchangeMPI(int side);           // Function has to be public for GPU, but its technically
src/fluid/braginskii/bragViscosity.hpp:  // Function for internal use (but public to allow for Cuda lambda capture)
src/global.cpp:  #if defined(KOKKOS_ENABLE_CUDA)
src/global.cpp:    defaultLoopPattern = LoopPattern::RANGE;  // On cuda, works best (generally)
src/mpi.cpp:#include "mpi-ext.h"                // Needed for CUDA-aware check */
src/mpi.cpp:  #ifdef KOKKOS_ENABLE_CUDA
src/mpi.cpp:    #if defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
src/mpi.cpp:      #error Your MPI library is not CUDA Aware (check Idefix requirements).
src/mpi.cpp:  #endif /* MPIX_CUDA_AWARE_SUPPORT */
src/mpi.cpp:    #ifdef KOKKOS_ENABLE_CUDA
src/mpi.cpp:      errmsg << "Check that your MPI library is CUDA aware." << std::endl;
src/mpi.cpp:      errmsg << "Check that your MPI library is RocM aware." << std::endl;
src/mpi.cpp:  #ifdef KOKKOS_ENABLE_CUDA
src/mpi.cpp:    errmsg << "Check that your MPI library is CUDA aware." << std::endl;
src/mpi.cpp:    errmsg << "Check that your MPI library is RocM aware." << std::endl;
src/macros.hpp:  #if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
src/macros.hpp:    // string formatting functions can't be accessed from GPU kernels,
src/macros.hpp:  #endif // if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)

```
