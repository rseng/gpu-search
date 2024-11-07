# https://github.com/parthenon-hpc-lab/parthenon

```console
tst/unit/test_pararrays.cpp:#if defined(KOKKOS_ENABLE_CUDA)
tst/unit/test_pararrays.cpp:using UVMSpace = Kokkos::CudaUVMSpace;
tst/unit/test_pararrays.cpp:#if !(defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
tst/unit/test_pararrays.cpp:#endif // !KOKKOS_ENABLE_CUDA
.gitlab-ci-darwin.yml:# 1. gcc-mpi-cuda-python-performance-application
.gitlab-ci-darwin.yml:.gcc-mpi-cuda-python-performance-application:
.gitlab-ci-darwin.yml:# 2. gcc-mpi-cuda-performance-regression-build
.gitlab-ci-darwin.yml:.gcc-mpi-cuda-performance-regression-build:
.gitlab-ci-darwin.yml:# 3. gcc-mpi-cuda-performance-regression-metrics
.gitlab-ci-darwin.yml:.gcc-mpi-cuda-performance-regression-metrics:
.gitlab-ci-darwin.yml:# 4. gcc-mpi-cuda-performance-regression-target-branch
.gitlab-ci-darwin.yml:.gcc-mpi-cuda-performance-regression-target-branch:
.gitlab-ci-darwin.yml:parthenon-power9-gcc-mpi-cuda-perf-manual-python-setup:
.gitlab-ci-darwin.yml:  extends: .gcc-mpi-cuda-python-performance-application
.gitlab-ci-darwin.yml:parthenon-power9-gcc-mpi-cuda-perf-manual-target-branch:
.gitlab-ci-darwin.yml:  extends: .gcc-mpi-cuda-performance-regression-target-branch
.gitlab-ci-darwin.yml:  needs: [parthenon-power9-gcc-mpi-cuda-perf-manual-python-setup]
.gitlab-ci-darwin.yml:parthenon-power9-gcc-mpi-cuda-perf-manual-build:
.gitlab-ci-darwin.yml:  extends: .gcc-mpi-cuda-performance-regression-build
.gitlab-ci-darwin.yml:  needs: [parthenon-power9-gcc-mpi-cuda-perf-manual-target-branch]
.gitlab-ci-darwin.yml:parthenon-power9-gcc-mpi-cuda-perf-manual-metrics:
.gitlab-ci-darwin.yml:  extends: .gcc-mpi-cuda-performance-regression-metrics
.gitlab-ci-darwin.yml:  needs: [parthenon-power9-gcc-mpi-cuda-perf-manual-build]
.gitlab-ci-darwin.yml:parthenon-power9-gcc-mpi-cuda-perf-schedule-python-setup:
.gitlab-ci-darwin.yml:  extends: .gcc-mpi-cuda-python-performance-application
.gitlab-ci-darwin.yml:parthenon-power9-gcc-mpi-cuda-perf-schedule-build:
.gitlab-ci-darwin.yml:  extends: .gcc-mpi-cuda-performance-regression-build
.gitlab-ci-darwin.yml:  needs: [parthenon-power9-gcc-mpi-cuda-perf-schedule-python-setup]
.gitlab-ci-darwin.yml:parthenon-power9-gcc-mpi-cuda-perf-schedule-metrics:
.gitlab-ci-darwin.yml:  extends: .gcc-mpi-cuda-performance-regression-metrics
.gitlab-ci-darwin.yml:  needs: [parthenon-power9-gcc-mpi-cuda-perf-schedule-build]
CHANGELOG.md:- [[PR 1189]](https://github.com/parthenon-hpc-lab/parthenon/pull/1189) Address CUDA MPI/ICP issue with Kokkos <=4.4.1
CHANGELOG.md:- [[PR 1117]](https://github.com/parthenon-hpc-lab/parthenon/pull/1117) Enable CI pipelines on AMD GPUs with ROCM/HIP
CHANGELOG.md:- [[PR 500]](https://github.com/parthenon-hpc-lab/parthenon/pull/500) Update docker file and CI environment (for Cuda 11.3 and latest `nsys`)
CHANGELOG.md:- [[PR 490]](https://github.com/parthenon-hpc-lab/parthenon/pull/490) Adjust block size in OverlappingSpace instance tests to remain within Cuda/HIP limits
CHANGELOG.md:- [[PR 332]](https://github.com/parthenon-hpc-lab/parthenon/pull/332) Rewrote boundary conditions to work on GPUs with variable packs. Re-enabled user-defined boundary conditions via `ApplicationInput`.
CHANGELOG.md:- [[PR 310]](https://github.com/parthenon-hpc-lab/parthenon/pull/310) Fix Cuda 11 builds.
CHANGELOG.md:- [[PR 281]](https://github.com/parthenon-hpc-lab/parthenon/pull/281) Allows one to run regression tests with more than one cuda device, Also improves readability of regression tests output.
doc/sphinx/src/README.rst:- Use the exception throwing versions in non-GPU,
doc/sphinx/src/README.rst:- On GPUs and in performance-critical
doc/sphinx/src/building.rst:|| CHECK\_REGISTRY\_PRESSURE                || OFF                           || Option || Check the registry pressure for Kokkos CUDA kernels                                                                                                         |
doc/sphinx/src/building.rst:|| NUM\_GPU\_DEVICES\_PER\_NODE             || 1                             || String || Number of GPUs per node to use if built with `Kokkos_ENABLE_CUDA`                                                                                           |
doc/sphinx/src/building.rst:|| PARTHENON\_ENABLE\_GPU\_MPI\_CHECKS      || ON                            || Option || Enable pre-test gpu-mpi checks                                                                                                                              |
doc/sphinx/src/building.rst:  memory, e.g., directly on the GPU when using Cuda. This requires the MPI
doc/sphinx/src/building.rst:  (e.g., often referred to as “Cuda-aware MPI”). To force buffer
doc/sphinx/src/building.rst:compiler (e.g., ``nvcc_wrapper`` for Cuda builds), or - paths to non
doc/sphinx/src/building.rst:configuration, e.g., one with Cuda and MPI enabled.
doc/sphinx/src/building.rst:   $ module load cuda gcc cmake python hdf5
doc/sphinx/src/building.rst:     3) lsf-tools/2.0   6) cuda/10.1.243           9) python/3.6.6-anaconda3-5.3.0
doc/sphinx/src/building.rst:Cuda with MPI
doc/sphinx/src/building.rst:   $ mkdir build-cuda-mpi && cd build-cuda-mpi
doc/sphinx/src/building.rst:   # $ module load cuda gcc cmake/3.18.2 python hdf5
doc/sphinx/src/building.rst:   # Manually run a simulation (here using 1 node with 6 GPUs and 1 MPI processes per GPU for a total of 6 processes (ranks)).
doc/sphinx/src/building.rst:   # Note the `-M "-gpu"` which is required to enable Cuda aware MPI.
doc/sphinx/src/building.rst:   # Also note the `--kokkos-num-devices=6` that ensures that each process on a node uses a different GPU.
doc/sphinx/src/building.rst:   $ jsrun -n 1 -a 6 -g 6 -c 42 -r 1 -d packed -b packed:7 --smpiargs=-gpu ./example/advection/advection-example -i ${PARTHENON_ROOT}/example/advection/parthinput.advection parthenon/time/nlim=10 parthenon/mesh/nx1=512 parthenon/mesh/nx2=512 parthenon/mesh/nx3=512 parthenon/meshblock/nx1=64 parthenon/meshblock/nx2=64 parthenon/meshblock/nx3=64 --kokkos-num-devices=6
doc/sphinx/src/building.rst:Cuda without MPI
doc/sphinx/src/building.rst:   $ mkdir build-cuda && cd build-cuda
doc/sphinx/src/building.rst:   $ cmake -DMACHINE_VARIANT=cuda ${PARTHENON_ROOT}
doc/sphinx/src/building.rst:Set-Up Environment (Optional, but Still Recommended, for Non-CUDA Builds)
doc/sphinx/src/building.rst:This step is required if you intend to build for CUDA (the default on
doc/sphinx/src/building.rst:architecture with 44 nodes per core and 4 Nvidia Volta GPUs per node. To
doc/sphinx/src/building.rst:.. _set-up-environment-optional-but-still-recommended-for-non-cuda-builds-1:
doc/sphinx/src/building.rst:Set-Up Environment (Optional, but Still Recommended, for Non-CUDA Builds)
doc/sphinx/src/building.rst:This step is required if you intend to build for CUDA (the default on
doc/sphinx/src/building.rst:By default cmake will build parthenon with cuda and mpi support. Other
doc/sphinx/src/building.rst:-  cuda-mpi
doc/sphinx/src/building.rst:-  cuda
doc/sphinx/src/building.rst:   $ module load cuda gcc/7.3.1
doc/sphinx/src/building.rst:     1) StdEnv (S)   2) cuda/10.1.243   3) gcc/7.3.1   4) spectrum-mpi/rolling-release
doc/sphinx/src/building.rst:.. _cuda-with-mpi-1:
doc/sphinx/src/building.rst:Cuda with MPI
doc/sphinx/src/building.rst:   $ mkdir build-cuda-mpi && cd build-cuda-mpi
doc/sphinx/src/building.rst:   $ cmake -DPARTHENON_DISABLE_HDF5=On -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_POWER9=True -DKokkos_ENABLE_CUDA=True -DKokkos_ARCH_VOLTA70=True -DCMAKE_CXX_COMPILER=${PWD}/../external/Kokkos/bin/nvcc_wrapper ..
doc/sphinx/src/building.rst:   # Make sure that GPUs are assigned round robin to MPI processes
doc/sphinx/src/building.rst:   # manually run a simulation (here using 1 node with 4 GPUs and 1 MPI processes per GPU and a total of 2 processes (ranks))
doc/sphinx/src/building.rst:   # note the `-M "-gpu"` which is required to enable Cuda aware MPI
doc/sphinx/src/building.rst:   # also note the `--kokkos-num-devices=1` that ensures that each process on a node uses a different GPU
doc/sphinx/src/building.rst:   $ jsrun -p 2 -g 1 -c 20 -M "-gpu" ./example/advection/advection-example -i ../example/advection/parthinput.advection parthenon/time/nlim=10 parthenon/mesh/nx1=128 parthenon/mesh/nx2=64 parthenon/mesh/nx3=64 parthenon/meshblock/nx1=32 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=32 --kokkos-num-devices=1 | tee 2.out
doc/sphinx/src/building.rst:.. _cuda-without-mpi-1:
doc/sphinx/src/building.rst:Cuda without MPI
doc/sphinx/src/building.rst:   $ mkdir build-cuda && cd build-cuda
doc/sphinx/src/building.rst:   $ cmake -DCMAKE_BUILD_TYPE=Release -DMACHINE_CFG=${PARTHENON_ROOT}/cmake/machinecfg/Summit.cmake -DMACHINE_VARIANT=cuda -DPARTHENON_DISABLE_MPI=On ${PARTHENON_ROOT}
doc/sphinx/src/weak_scaling.rst:   architecture with two Volta GPUs per node.
doc/sphinx/src/weak_scaling.rst:   module load cmake gcc/7.4.0 cuda/10.2 openmpi/p9/4.0.1-gcc_7.4.0 anaconda/Anaconda3.2019.10
doc/sphinx/src/weak_scaling.rst:   cmake -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=True -DKokkos_ARCH_POWER9=True -DKokkos_ENABLE_CUDA=True -DKokkos_ARCH_VOLTA70=True -DCMAKE_CXX_COMPILER=${PWD}/../external/Kokkos/bin/nvcc_wrapper ..
doc/sphinx/src/boundary_communication.rst:this is not performant, as launching a GPU kernel per sub-halo costs
doc/sphinx/src/boundary_communication.rst:To resolve this issue, we coalesce GPU kernels into hierarchically
doc/sphinx/src/boundary_communication.rst:   using dev_arr_t = typename Kokkos::View<T *, Kokkos::LayoutRight, Kokkos::CudaSpace>;
doc/sphinx/src/interface/containers.rst:on a GPU, and such that one can index into the collection in a known
doc/sphinx/src/nested_par_for.rst:be allocated. For CUDA GPUs, ``scratch_level=0`` allocates the cache in
doc/sphinx/src/nested_par_for.rst:for CUDA since the Kokkos loops are required for parallelization on
doc/sphinx/src/nested_par_for.rst:GPUs.
doc/sphinx/src/nested_par_for.rst:* On GPUs, the outer loop typically maps to blocks, while the inner
doc/sphinx/src/nested_par_for.rst:  CUDA terms a streaming multiprocessor (SM, equivalent to a Compute
doc/sphinx/src/nested_par_for.rst:  Unit or CU on AMD GPUs) with multiple warps (or wavefronts for AMD)
doc/sphinx/src/nested_par_for.rst:  create enough blocks to fill all SMs on the GPU divided by the
doc/sphinx/src/nested_par_for.rst:  Data Share or LDS on AMD GPUs) and higher register usage. Note that
doc/sphinx/src/nested_par_for.rst:  SM will vary between GPU architectures and especially between GPU
doc/sphinx/src/nested_par_for.rst:To balance the CPU vs GPU hardware considerations of hierarchical
doc/sphinx/src/tasks.rst:with fewer lists will produce more work per kernel (which may be good for GPUs,
doc/sphinx/src/development.rst:   a GPU.
doc/sphinx/src/development.rst:   stream (on Cuda devices) and are discouraged from use. Use
doc/sphinx/src/development.rst:parallel region may be handled by many (GPU/…) threads in parallel.
doc/sphinx/src/development.rst:This is a current Cuda limitation for extended device lambdas, see `Cuda
doc/sphinx/src/development.rst:guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#extended-lambda-restrictions>`__,
doc/sphinx/src/parthenon_arrays.rst:``Kokkos::CudaUVMSpace``.
doc/sphinx/src/parthenon_arrays.rst:reference counted, works on GPUs, and is almost as performant as
README.md:  - `MANUAL1D_LOOP` maps to `Kokkos::RangePolicy` (default for CUDA backend)
README.md:  - `TVR_INNER_LOOP` maps to `Kokkos::TeamVectorRange` (default for CUDA backend)
README.md:or to build for NVIDIA V100 GPUs (using `nvcc` compiler for GPU code, which is automatically picked up by `Kokkos`)
README.md:    mkdir build-cuda-v100 && cd build-cuda-v100
README.md:    cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=On ../
README.md:or to build for AMD MI100 GPUs (using `hipcc` compiler)
CMakeLists.txt:option(PARTHENON_ENABLE_HOST_COMM_BUFFERS "CUDA/HIP Only: Allocate communication buffers on host (may be slower)" OFF)
CMakeLists.txt:option(PARTHENON_ENABLE_GPU_MPI_CHECKS "Checks if possible that the mpi num of procs and the number\
CMakeLists.txt:of gpu devices detected are appropriate." ${PARTHENON_ENABLE_TESTING})
CMakeLists.txt:option(CHECK_REGISTRY_PRESSURE "Check the registry pressure for Kokkos CUDA kernels" OFF)
CMakeLists.txt:# Check that gpu devices are actually detected
CMakeLists.txt:set(NUM_GPU_DEVICES_PER_NODE "1" CACHE STRING "Number of gpu devices to use when testing if built with Kokkos_ENABLE_CUDA")
CMakeLists.txt:  if (NOT Kokkos_ENABLE_CUDA AND NOT Kokkos_ENABLE_HIP)
CMakeLists.txt:    message(FATAL_ERROR "Host pinned buffers for MPI communication are supported only for CUDA and HIP backends.")
CMakeLists.txt:if (Kokkos_ENABLE_CUDA AND TEST_INTEL_OPTIMIZATION)
CMakeLists.txt:if (Kokkos_ENABLE_CUDA)
CMakeLists.txt:  set(Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC OFF CACHE BOOL "Disable Cuda async malloc (to address MPI/IPC issues)")
CMakeLists.txt:# GPU check on the build node are currently only supported for Nvidia GPUs
CMakeLists.txt:if (Kokkos_ENABLE_CUDA AND "${PARTHENON_ENABLE_GPU_MPI_CHECKS}" )
scripts/device_check.sh:# This script is designed to detect the number of gpu devices that are available before the
scripts/device_check.sh:# ctest suite is run. It does this by using the nvidia-smi. The script is only called if
scripts/device_check.sh:# parthenon is built with kokkos_ENABLE_CUDA.
scripts/device_check.sh:# The script takes 3 arguments the number of GPUs per node, that are meant to be used with the 
scripts/device_check.sh:# 2. That the number of GPUs that are actually avaliable on the node are enough to satisfy 
scripts/device_check.sh:# 3. That more than a single gpu is not assigned to each mpi proc
scripts/device_check.sh:1. the number of GPUs per node to be used with the tests,
scripts/device_check.sh:NUM_GPUS_PER_NODE_REQUESTED=$1
scripts/device_check.sh:  if ! command -v nvidia-smi &> /dev/null
scripts/device_check.sh:    printf "CUDA has been enabled but nvidia-smi cannot be found\n\n"
scripts/device_check.sh:  output=$(nvidia-smi -L)
scripts/device_check.sh:  found=$(echo $output | grep "GPU 0: ")
scripts/device_check.sh:  GPUS_DETECTED=0
scripts/device_check.sh:    let GPUS_DETECTED+=1
scripts/device_check.sh:    found=$(echo $output | grep "GPU ${GPUS_DETECTED}: ")
scripts/device_check.sh:  if [ "$GPUS_DETECTED" -eq "0" ]
scripts/device_check.sh:    printf "CUDA has been enabled but no GPUs have been detected.\n\n"
scripts/device_check.sh:  elif [ "$NUM_GPUS_PER_NODE_REQUESTED" -gt "$GPUS_DETECTED" ]
scripts/device_check.sh:    printf "You are trying to build the parthenon regression tests with CUDA enabled kokkos, with the following
scripts/device_check.sh:Number of CUDA devices per node set to: ${NUM_GPUS_PER_NODE_REQUESTED}
scripts/device_check.sh:Number of CUDA devices per node available: ${GPUS_DETECTED}
scripts/device_check.sh:The number of GPUs detected on node is less than then the number of GPUs requested, consider changing:
scripts/device_check.sh:NUM_GPU_DEVICES_PER_NODE=${GPUS_DETECTED}
scripts/device_check.sh:Or consider building without CUDA.\n\n"
scripts/device_check.sh:    printf "Number of GPUs detected per node: $GPUS_DETECTED
scripts/device_check.sh:Number of GPUs per node, requested in tests: $NUM_GPUS_PER_NODE_REQUESTED\n"
scripts/device_check.sh:  if [ "$NUM_GPUS_PER_NODE_REQUESTED" -gt "$MPI_PROCS_PER_NODE" ]
scripts/device_check.sh:NUM_GPU_DEVICES_PER_NODE=${NUM_GPUS_PER_NODE_REQUESTED}
scripts/device_check.sh:Assigning more than a single GPU to a given MPI proc is not supported. You have a total of ${NUM_GPUS_PER_NODE_REQUESTED} GPU(s) 
scripts/docker/Dockerfile.nvcc:FROM nvidia/cuda:11.6.1-devel-ubuntu20.04
scripts/docker/Dockerfile.nvcc:    DEBIAN_FRONTEND="noninteractive" TZ=America/New_York apt-get install -y --no-install-recommends git python3-minimal libpython3-stdlib bc hwloc wget openssh-client python3-numpy python3-h5py python3-matplotlib python3-scipy python3-pip lcov curl cuda-nsight-systems-11-6 cmake ninja-build
scripts/docker/Dockerfile.nvcc:    ./configure --prefix=/opt/openmpi --enable-mpi-cxx --with-cuda && \
scripts/docker/Dockerfile.nvcc:COPY build_ascent_cuda.sh /tmp/build-ascent/build_ascent_cuda.sh
scripts/docker/Dockerfile.nvcc:    bash build_ascent_cuda.sh && \
scripts/docker/Dockerfile.hip-rocm:FROM rocm/dev-ubuntu-20.04:5.4.3
scripts/docker/build_ascent_cuda.sh:# Slightly adapted from https://github.com/Alpine-DAV/ascent/blob/0dedd70319145b3a31dd4d889fb82aaad995797b/scripts/build_ascent/build_ascent_cuda.sh
scripts/docker/build_ascent_cuda.sh:CUDA_ARCH="${CUDA_ARCH:=80}"
scripts/docker/build_ascent_cuda.sh:CUDA_ARCH_VTKM="${CUDA_ARCH_VTKM:=ampere}"
scripts/docker/build_ascent_cuda.sh:  -DVTKm_ENABLE_CUDA=ON \
scripts/docker/build_ascent_cuda.sh:  -DVTKm_CUDA_Architecture=ampere \
scripts/docker/build_ascent_cuda.sh:  -DCMAKE_CUDA_HOST_COMPILER=${CXX}\
scripts/docker/build_ascent_cuda.sh:  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
scripts/docker/build_ascent_cuda.sh:  -DENABLE_CUDA=ON \
scripts/docker/build_ascent_cuda.sh:  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
scripts/docker/build_ascent_cuda.sh:  -DENABLE_CUDA=ON \
scripts/docker/build_ascent_cuda.sh:  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
scripts/docker/build_ascent_cuda.sh:  -DENABLE_CUDA=ON \
scripts/docker/build_ascent_cuda.sh:  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
scripts/docker/build_ascent_cuda.sh:echo 'set(ENABLE_CUDA ON CACHE BOOL "")' >> ascent-config.cmake
scripts/docker/build_ascent_cuda.sh:echo 'set(CMAKE_CUDA_ARCHITECTURES ' ${CUDA_ARCH} ' CACHE PATH "")' >> ascent-config.cmake
scripts/docker/build_ascent_cuda.sh:  -DENABLE_CUDA=ON \
scripts/docker/build_ascent_cuda.sh:  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
benchmarks/burgers/README.md:To build for execution on a single GPU, it should be sufficient to add the following flags to the CMake configuration line
benchmarks/burgers/README.md:-DPARTHENON_DISABLE_MPI=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
benchmarks/burgers/README.md:On a two-socket Broadwell node with 36 cores, the benchmark takes approximately 213 seconds (3.5 minutes) to run to completion (250 time steps), averaging approximately $4.0\times 10^6$ zone-cycles/wallsecond.  On a single NVIDIA A100 GPU, the run completes in about 45 seconds, averaging approximately $1.8\times 10^7$ zone-cycles/wallsecond.  Strong scaling results on a single Broadwell node are shown below in Figure 2.
benchmarks/burgers/README.md:For the GPU, we measure throughput on a single-level mesh ("parthenon/mesh/numlevel = 1") and vary the base mesh size and the block size.  Results on a 40 GB A100 are shown in Figure 3.
benchmarks/burgers/README.md:<p style="text-align:center;"><img src="data/pvibe_gpu_throughput.png" alt="Plot showing throughput on an A100 at different mesh and block sizes" style=width:50%><br />Figure 3: Throughput for different mesh and block sizes on a single 40 GB A100 GPU.</p>
cmake/machinecfg/GitHubActions.cmake:if (${MACHINE_VARIANT} MATCHES "cuda")
cmake/machinecfg/GitHubActions.cmake:  # using an arbitrary arch as GitHub Action runners don't have GPUs
cmake/machinecfg/GitHubActions.cmake:  set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "GPU architecture")
cmake/machinecfg/GitHubActions.cmake:  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Cuda")
cmake/machinecfg/GitHubActions.cmake:    set(MACHINE_CXX_FLAGS "${MACHINE_CXX_FLAGS} -Wno-unknown-cuda-version")
cmake/machinecfg/GitHubActions.cmake:  set(Kokkos_ARCH_NAVI1030 ON CACHE BOOL "GPU architecture")
cmake/machinecfg/Spock.cmake:  "export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0\n"
cmake/machinecfg/Spock.cmake:  "export MPICH_GPU_SUPPORT_ENABLED=1\n"
cmake/machinecfg/Spock.cmake:set(MACHINE_VARIANT "hip-mpi" CACHE STRING "Default build for CUDA and MPI")
cmake/machinecfg/Spock.cmake:  set(Kokkos_ARCH_VEGA908 ON CACHE BOOL "GPU architecture")
cmake/machinecfg/Spock.cmake:  set(CMAKE_CXX_COMPILER $ENV{ROCM_PATH}/llvm/bin/clang++ CACHE STRING "Use g++")
cmake/machinecfg/Spock.cmake:set(NUM_GPU_DEVICES_PER_NODE "4" CACHE STRING "4x MI100 per node")
cmake/machinecfg/Spock.cmake:set(PARTHENON_ENABLE_GPU_MPI_CHECKS OFF CACHE BOOL "Disable check by default")
cmake/machinecfg/Summit.cmake:  "Supported MACHINE_VARIANT includes 'cuda', 'mpi', and 'cuda-mpi'\n"
cmake/machinecfg/Summit.cmake:  "  $ module load cuda/11.5.2 gcc cmake python hdf5\n"
cmake/machinecfg/Summit.cmake:set(MACHINE_VARIANT "cuda-mpi" CACHE STRING "Default build for CUDA and MPI")
cmake/machinecfg/Summit.cmake:if (${MACHINE_VARIANT} MATCHES "cuda")
cmake/machinecfg/Summit.cmake:  set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "GPU architecture")
cmake/machinecfg/Summit.cmake:  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Cuda")
cmake/machinecfg/Summit.cmake:set(NUM_GPU_DEVICES_PER_NODE "6" CACHE STRING "6x V100 per node")
cmake/machinecfg/Summit.cmake:set(PARTHENON_ENABLE_GPU_MPI_CHECKS OFF CACHE BOOL "Disable check by default")
cmake/machinecfg/Summit.cmake:  # Use a single resource set on a node that includes all cores and GPUs.
cmake/machinecfg/Summit.cmake:  # GPUs are automatically assigned round robin when run with more than one rank.
cmake/machinecfg/Summit.cmake:  list(APPEND TEST_MPIOPTS "-n" "1" "-g" "6" "-c" "42" "-r" "1" "-d" "packed" "-b" "packed:7" "--smpiargs='-gpu'")
cmake/machinecfg/CI.cmake:  "Supported MACHINE_VARIANT includes 'cuda', 'mpi', and 'cuda-mpi'")
cmake/machinecfg/CI.cmake:if (${MACHINE_VARIANT} MATCHES "cuda")
cmake/machinecfg/CI.cmake:  set(Kokkos_ARCH_AMPERE80 ON CACHE BOOL "GPU architecture")
cmake/machinecfg/CI.cmake:  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Cuda")
cmake/machinecfg/Darwin.cmake:#           `ppc64le` (assumes power9 + volta gpus)
cmake/machinecfg/Darwin.cmake:# - `DARWIN_CUDA` - Build for CUDA
cmake/machinecfg/Darwin.cmake:#       Default: ON if nvidia-smi finds at least one GPU, OFF otherwise
cmake/machinecfg/Darwin.cmake:    set(DARWIN_CUDA_VERSION "11.0")
cmake/machinecfg/Darwin.cmake:    set(DARWIN_CUDA_DEFAULT OFF)
cmake/machinecfg/Darwin.cmake:    set(DARWIN_CUDA_VERSION "11.0")
cmake/machinecfg/Darwin.cmake:    set(DARWIN_CUDA_DEFAULT ON)
cmake/machinecfg/Darwin.cmake:    string(APPEND TEST_MPIOPTS "-gpu")
cmake/machinecfg/Darwin.cmake:    COMMAND nvidia-smi -L
cmake/machinecfg/Darwin.cmake:    OUTPUT_VARIABLE FOUND_GPUS)
cmake/machinecfg/Darwin.cmake:string(REPLACE "\n" ";" FOUND_GPUS ${FOUND_GPUS})
cmake/machinecfg/Darwin.cmake:list(FILTER FOUND_GPUS INCLUDE REGEX "GPU [0-9]")
cmake/machinecfg/Darwin.cmake:list(LENGTH FOUND_GPUS GPU_COUNT)
cmake/machinecfg/Darwin.cmake:if (GPU_COUNT EQUAL 0)
cmake/machinecfg/Darwin.cmake:    set(DARWIN_CUDA_DEFAULT OFF)
cmake/machinecfg/Darwin.cmake:    set(DARWIN_CUDA_DEFAULT ON)
cmake/machinecfg/Darwin.cmake:set(DARWIN_CUDA ${DARWIN_CUDA_DEFAULT} CACHE BOOL "Build for CUDA")
cmake/machinecfg/Darwin.cmake:              DARWIN_CUDA: ${DARWIN_CUDA}
cmake/machinecfg/Darwin.cmake:                GPU_COUNT: ${GPU_COUNT}
cmake/machinecfg/Darwin.cmake:if (DARWIN_CUDA)
cmake/machinecfg/Darwin.cmake:    # Location of CUDA
cmake/machinecfg/Darwin.cmake:    set(CUDAToolkit_ROOT /usr/local/cuda-${DARWIN_CUDA_VERSION}
cmake/machinecfg/Darwin.cmake:        CACHE STRING "CUDA Location")
cmake/machinecfg/Darwin.cmake:    # All of this code ensures that the CUDA build uses the correct nvcc, and
cmake/machinecfg/Darwin.cmake:    set(ENV{PATH} "${CUDAToolkit_ROOT}/bin:$ENV{PATH}")
cmake/machinecfg/Darwin.cmake:    # nvcc_wrapper must be the CXX compiler for CUDA builds. Ideally this would
cmake/machinecfg/Darwin.cmake:        if (DARWIN_CUDA)
cmake/machinecfg/Darwin.cmake:if (DARWIN_CUDA)
cmake/machinecfg/Darwin.cmake:    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Cuda")
cmake/machinecfg/Darwin.cmake:if (DARWIN_CUDA AND GPU_COUNT LESS 2)
cmake/machinecfg/Darwin.cmake:    set(NUM_RANKS ${GPU_COUNT})
cmake/machinecfg/Darwin.cmake:set(NUM_GPU_DEVICES_PER_NODE ${NUM_RANKS} CACHE STRING "Number of gpu devices to use when testing if built with Kokkos_ENABLE_CUDA")
cmake/machinecfg/FrontierAndCrusher.cmake:  "  $ export MPICH_GPU_SUPPORT_ENABLED=1\n\n"
cmake/machinecfg/FrontierAndCrusher.cmake:  "  $ export MPICH_GPU_SUPPORT_ENABLED=1\n\n"
cmake/machinecfg/FrontierAndCrusher.cmake:set(MACHINE_VARIANT "hip-mpi" CACHE STRING "Default build for CUDA and MPI")
cmake/machinecfg/FrontierAndCrusher.cmake:  set(Kokkos_ARCH_VEGA90A ON CACHE BOOL "GPU architecture")
cmake/machinecfg/FrontierAndCrusher.cmake:  set(Kokkos_ARCH_VEGA90A ON CACHE BOOL "GPU architecture")
cmake/machinecfg/FrontierAndCrusher.cmake:  set(CMAKE_CXX_COMPILER $ENV{ROCM_PATH}/llvm/bin/clang++ CACHE STRING "Use g++")
cmake/machinecfg/FrontierAndCrusher.cmake:set(NUM_GPU_DEVICES_PER_NODE "1" CACHE STRING "4x MI250x per node with 2 GCDs each but one visible per rank")
cmake/machinecfg/FrontierAndCrusher.cmake:set(PARTHENON_ENABLE_GPU_MPI_CHECKS OFF CACHE BOOL "Disable check by default")
cmake/machinecfg/FrontierAndCrusher.cmake:  set(MACHINE_CXX_FLAGS "${MACHINE_CXX_FLAGS} -I$ENV{ROCM_PATH}/include")
cmake/machinecfg/FrontierAndCrusher.cmake:  set(CMAKE_EXE_LINKER_FLAGS "-L$ENV{ROCM_PATH}/lib -lamdhip64" CACHE STRING "Default flags for this config")
cmake/machinecfg/FrontierAndCrusher.cmake:  # ensure that GPU are properly bound to ranks
cmake/machinecfg/FrontierAndCrusher.cmake:  list(APPEND TEST_MPIOPTS "-c1" "--gpus-per-node=8" "--gpu-bind=closest")
cmake/machinecfg/RZAnsel.cmake:# - `RZANSEL_CUDA` - Build for CUDA
cmake/machinecfg/RZAnsel.cmake:  "Supported MACHINE_VARIANT includes 'cuda', 'mpi', and 'cuda-mpi'\n")
cmake/machinecfg/RZAnsel.cmake:set(RZANSEL_CUDA_VERSION "10.1.243")
cmake/machinecfg/RZAnsel.cmake:set(MACHINE_VARIANT "cuda-mpi" CACHE STRING "Machine variant to use when building on RZAnsel")
cmake/machinecfg/RZAnsel.cmake:set(GPU_COUNT "4")
cmake/machinecfg/RZAnsel.cmake:if (${MACHINE_VARIANT} MATCHES "cuda")
cmake/machinecfg/RZAnsel.cmake:  set(RZANSEL_CUDA_DEFAULT ON)
cmake/machinecfg/RZAnsel.cmake:  set(RZANSEL_CUDA_DEFAULT OFF)
cmake/machinecfg/RZAnsel.cmake:set(RZANSEL_CUDA ${RZANSEL_CUDA_DEFAULT} CACHE BOOL "Build for CUDA")
cmake/machinecfg/RZAnsel.cmake:              RZANSEL_CUDA: ${RZANSEL_CUDA}
cmake/machinecfg/RZAnsel.cmake:                 GPU_COUNT: ${GPU_COUNT}
cmake/machinecfg/RZAnsel.cmake:if (RZANSEL_CUDA)
cmake/machinecfg/RZAnsel.cmake:    # Location of CUDA
cmake/machinecfg/RZAnsel.cmake:    set(CUDA_ROOT /usr/tce/packages/cuda/cuda-${RZANSEL_CUDA_VERSION} 
cmake/machinecfg/RZAnsel.cmake:      CACHE STRING "CUDA Location")
cmake/machinecfg/RZAnsel.cmake:    # This code ensures that the CUDA build uses the correct nvcc, and
cmake/machinecfg/RZAnsel.cmake:    set(ENV{CUDA_ROOT} "${CUDA_ROOT}")
cmake/machinecfg/RZAnsel.cmake:    # nvcc_wrapper must be the CXX compiler for CUDA builds. Ideally this would
cmake/machinecfg/RZAnsel.cmake:        if (RZANSEL_CUDA)
cmake/machinecfg/RZAnsel.cmake:if (RZANSEL_CUDA)
cmake/machinecfg/RZAnsel.cmake:    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Cuda")
cmake/machinecfg/RZAnsel.cmake:    set(NUM_GPU_DEVICES_PER_NODE ${NUM_RANKS} CACHE STRING "Number of gpu devices to use when testing if built with Kokkos_ENABLE_CUDA")
cmake/machinecfg/RZAnsel.cmake:  if (${MACHINE_VARIANT} MATCHES "cuda")
cmake/machinecfg/RZAnsel.cmake:    string(APPEND TEST_MPIOPTS "-c 1 -n 1 -g ${NUM_GPU_DEVICES_PER_NODE} -r 1 -d packed --smpiargs='-gpu'")
cmake/machinecfg/RZAnsel.cmake:  set(PARTHENON_ENABLE_GPU_MPI_CHECKS OFF CACHE STRING "Checks if possible that the mpi num of procs and the number of gpu devices detected are appropriate.")
cmake/CTestCustom.cmake.in:SET(CTEST_CUSTOM_PRE_TEST "bash @CMAKE_CURRENT_SOURCE_DIR@/scripts/device_check.sh @NUM_GPU_DEVICES_PER_NODE@ @MPIEXEC_EXECUTABLE@ @NUM_MPI_PROC_TESTING@")
cmake/TestSetup.cmake:    if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
cmake/TestSetup.cmake:      list(APPEND labels "cuda")
cmake/TestSetup.cmake:    # When targeting CUDA we don't have a great way of controlling how tests
cmake/TestSetup.cmake:    # get mapped to GPUs, so just enforce serial execution
cmake/TestSetup.cmake:    if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
cmake/parthenonConfig.cmake.in:set(Kokkos_BUILT_WITH_CUDA @Kokkos_ENABLE_CUDA@)
cmake/parthenonConfig.cmake.in:if(${Kokkos_BUILT_WITH_CUDA})
cmake/parthenonConfig.cmake.in:      message(WARNING "Kokkos was built with cuda, recommend setting CMAKE_CXX_COMPILER to @CMAKE_INSTALL_PREFIX@/bin/nvcc_wrapper")
cmake/parthenonConfig.cmake.in:      message(WARNING "Kokkos was built with cuda, recommend setting CMAKE_CXX_COMPILER to @Kokkos_DIR@/bin/nvcc_wrapper")
example/kokkos_pi/kokkos_pi.cpp:// Since the mesh infrastructure is not yet usable on GPUs, we create
example/kokkos_pi/kokkos_pi.cpp:// sizes.  Once we have a canonical method of using a mesh on the GPU,
example/kokkos_pi/kokkos_pi.cpp:  // Since our mesh is not GPU friendly we set up a hacked up
CONTRIBUTING.md:on a machine with an Intel Xeon E5540 (Broadwell) processor and Nvidia GeForce GTX 1060 (Pascal) GPU.
CONTRIBUTING.md:The current tests span MPI and non-MPI configurations on CPUs (using GCC) and GPUs (using Cuda/nvcc).
CONTRIBUTING.md:The runners have Intel Xeon Gold 6148 (Skylake) processors and Nvidia V100 (Volta) GPUs.
CONTRIBUTING.md:The current tests span uniform grids on GPUs (using Cuda/nvcc).
CONTRIBUTING.md:NVIDIA V100 (Volta) GPUs (power9 architecture). Tests run on these systems are
CONTRIBUTING.md:Cuda enabled. All tests are run on a single node with access to two Volta
CONTRIBUTING.md:GPUs. In addition, the regression tests are run in parallel with two mpi
CONTRIBUTING.md:processors each of which have access to their own Volta gpu. The following
CONTRIBUTING.md:deck. If parthenon is compiled with CUDA enabled, by default a single GPU will
src/interface/metadata.hpp:  // Include explicit destructor to get rid of CUDA __host__ __device__ warning
src/mesh/mesh.hpp:  // Moved here given Cuda/nvcc restriction:
src/CMakeLists.txt:if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
src/CMakeLists.txt:# For Cuda with NVCC (<11.2) and C++17 Kokkos currently does not work/compile with
src/CMakeLists.txt:# Therefore, we don't use the Kokkos_ENABLE_CUDA_CONSTEXPR option add the flag manually.
src/CMakeLists.txt:# Also, not checking for NVIDIA as nvcc_wrapper is identified as GNU so we just make sure
src/CMakeLists.txt:# the flag is not added when compiling with Clang for Cuda.
src/CMakeLists.txt:if (Kokkos_ENABLE_CUDA AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
src/CMakeLists.txt:  # we need to make sure CUDA_RESOLVE_DEVICE_SYMBOLS is on for our target
src/CMakeLists.txt:  if(CMAKE_CUDA_COMPILER)
src/CMakeLists.txt:    set_property(TARGET parthenon PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
src/kokkos_abstraction.hpp:#ifdef KOKKOS_ENABLE_CUDA_UVM
src/kokkos_abstraction.hpp:using DevMemSpace = Kokkos::CudaUVMSpace;
src/kokkos_abstraction.hpp:using HostMemSpace = Kokkos::CudaUVMSpace;
src/kokkos_abstraction.hpp:using DevExecSpace = Kokkos::Cuda;
src/kokkos_abstraction.hpp:#if defined(KOKKOS_ENABLE_CUDA)
src/kokkos_abstraction.hpp:using BufMemSpace = Kokkos::CudaHostPinnedSpace::memory_space;
src/kokkos_abstraction.hpp:  // TODO(pgrete) if exec space is Cuda,throw error
src/kokkos_abstraction.hpp:  // TODO(pgrete) if exec space is Cuda,throw error
src/kokkos_abstraction.hpp:#ifdef KOKKOS_ENABLE_CUDA
src/kokkos_abstraction.hpp:struct SpaceInstance<Kokkos::Cuda> {
src/kokkos_abstraction.hpp:  static Kokkos::Cuda create() {
src/kokkos_abstraction.hpp:    cudaStream_t stream;
src/kokkos_abstraction.hpp:    cudaStreamCreate(&stream);
src/kokkos_abstraction.hpp:    return Kokkos::Cuda(stream);
src/kokkos_abstraction.hpp:  static void destroy(Kokkos::Cuda &space) {
src/kokkos_abstraction.hpp:    cudaStream_t stream = space.cuda_stream();
src/kokkos_abstraction.hpp:    cudaStreamDestroy(stream);
src/kokkos_abstraction.hpp:    auto local_rank_str = std::getenv("CUDA_LAUNCH_BLOCKING");
src/prolong_restrict/prolong_restrict.hpp:  // Include explicit destructor to get rid of CUDA __host__ __device__ warning
src/solvers/mg_solver.hpp:  // These functions apparently have to be public to compile with cuda since
src/utils/index_split.cpp:#ifdef KOKKOS_ENABLE_CUDA
src/utils/index_split.cpp:#endif // KOKKOS_ENABLE_CUDA
src/utils/index_split.cpp:#ifdef KOKKOS_ENABLE_CUDA
src/utils/index_split.cpp:#ifdef KOKKOS_ENABLE_CUDA
src/utils/signal_handler.cpp:  sigprocmask(SIG_BLOCK, &mask, nullptr);
src/utils/signal_handler.cpp:  sigprocmask(SIG_UNBLOCK, &mask, nullptr);
src/utils/index_split.hpp:  int concurrency_;                   //  = NSMs = 132 for NVIDIA H100
src/utils/sort.hpp:#ifdef KOKKOS_ENABLE_CUDA
src/utils/sort.hpp:#ifdef KOKKOS_ENABLE_CUDA
src/utils/sort.hpp:                 "this message and need sort on CUDA devices with clang compiler please "
src/utils/sort.hpp:    PARTHENON_FAIL("sort is not supported outside of CPU or NVIDIA GPU. If you need sort "
src/utils/sort.hpp:                   "support on other devices, e.g., AMD or Intel GPUs, please get in "
src/utils/sort.hpp:#endif // KOKKOS_ENABLE_CUDA
src/utils/sort.hpp:#ifdef KOKKOS_ENABLE_CUDA
src/utils/sort.hpp:                 "this message and need sort on CUDA devices with clang compiler please "
src/utils/sort.hpp:    PARTHENON_FAIL("sort is not supported outside of CPU or NVIDIA GPU. If you need sort "
src/utils/sort.hpp:                   "support on other devices, e.g., AMD or Intel GPUs, please get in "
src/utils/sort.hpp:#endif // KOKKOS_ENABLE_CUDA

```
