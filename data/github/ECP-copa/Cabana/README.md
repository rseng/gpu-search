# https://github.com/ECP-copa/Cabana

```console
.jenkins:                stage('CUDA-11-NVCC-DEBUG') {
.jenkins:                            additionalBuildArgs '--build-arg BASE=nvidia/cuda:11.0.3-devel-ubuntu20.04'
.jenkins:                            label 'nvidia-docker && volta'
.jenkins:                            args '-v /tmp/ccache.kokkos:/tmp/ccache --env NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES'
.jenkins:                              -D MPIEXEC_PREFLAGS="--allow-run-as-root;--mca;btl_smcuda_use_cuda_ipc;0" \
.jenkins:                              -D Cabana_REQUIRE_CUDA=ON \
.jenkins:                stage('ROCM-5.2-HIPCC-DEBUG') {
.jenkins:                            additionalBuildArgs '--build-arg BASE=rocm/dev-ubuntu-20.04:5.2-complete'
.jenkins:                            label 'rocm-docker && vega && AMD_Radeon_Instinct_MI60'
.jenkins:                            label 'NVIDIA_Tesla_V100-PCIE-32GB && nvidia-docker'
.jenkins:                              -D CMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -Wno-unknown-cuda-version -Wno-sycl-target -fp-model=precise" \
.ecp-gitlab-ci.yml:.BuildKokkos_CUDA:
.ecp-gitlab-ci.yml:    KOKKOS_EXTRA_ARGS: "-DKokkos_ARCH_VOLTA70=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DCMAKE_CXX_COMPILER=${CI_PROJECT_DIR}/kokkos/bin/nvcc_wrapper"
.ecp-gitlab-ci.yml:    - module load cuda
.ecp-gitlab-ci.yml:.Build_CUDA:
.ecp-gitlab-ci.yml:    - module load cuda
.ecp-gitlab-ci.yml:BuildKokkos CUDA:
.ecp-gitlab-ci.yml:    BACKENDS: "CUDA"
.ecp-gitlab-ci.yml:  extends: .BuildKokkos_CUDA
.ecp-gitlab-ci.yml:Build CUDA:
.ecp-gitlab-ci.yml:    BACKENDS: "CUDA"
.ecp-gitlab-ci.yml:  extends: .Build_CUDA
.ecp-gitlab-ci.yml:   - BuildKokkos CUDA
.ecp-gitlab-ci.yml:BuildKokkos SERIAL CUDA:
.ecp-gitlab-ci.yml:    BACKENDS: "SERIAL CUDA"
.ecp-gitlab-ci.yml:  extends: .BuildKokkos_CUDA
.ecp-gitlab-ci.yml:Build SERIAL CUDA:
.ecp-gitlab-ci.yml:    BACKENDS: "SERIAL CUDA"
.ecp-gitlab-ci.yml:  extends: .Build_CUDA
.ecp-gitlab-ci.yml:   - BuildKokkos SERIAL CUDA
.ecp-gitlab-ci.yml:BuildKokkos SERIAL CUDA OPENMP:
.ecp-gitlab-ci.yml:    BACKENDS: "SERIAL CUDA OPENMP"
.ecp-gitlab-ci.yml:  extends: .BuildKokkos_CUDA
.ecp-gitlab-ci.yml:Build SERIAL CUDA OPENMP:
.ecp-gitlab-ci.yml:    BACKENDS: "SERIAL CUDA OPENMP"
.ecp-gitlab-ci.yml:  extends: .Build_CUDA
.ecp-gitlab-ci.yml:   - BuildKokkos SERIAL CUDA OPENMP
core/src/impl/Cabana_PerformanceTraits.hpp:// Cuda specialization. Use the warp traits.
core/src/impl/Cabana_PerformanceTraits.hpp:#if defined( KOKKOS_ENABLE_CUDA )
core/src/impl/Cabana_PerformanceTraits.hpp:class PerformanceTraits<Kokkos::Cuda>
core/src/Cabana_CommunicationPlan.hpp:// CUDA and HIP use atomics.
core/src/Cabana_CommunicationPlan.hpp:#ifdef KOKKOS_ENABLE_CUDA
core/src/Cabana_CommunicationPlan.hpp:struct CountSendsAndCreateSteeringAlgorithm<Kokkos::Cuda>
core/src/Cabana_CommunicationPlan.hpp:#endif // end KOKKOS_ENABLE_CUDA
core/src/Cabana_CommunicationPlan.hpp:    // we make them public to allow using private class data in CUDA kernels
core/src/Cabana_CommunicationPlan.hpp:        // CUDA workaround for handling class private data.
CHANGELOG.md:    - multidimensional distributed FFTs via heFFTe (including host, CUDA, and HIP)
CHANGELOG.md:    - linear solvers and preconditions via HYPRE (including host and CUDA)
CHANGELOG.md:- CUDA and HIP support and testing in continuous integration
CHANGELOG.md:- An optional MPI dependency has been added. Note that when CUDA is enabled the MPI implementation is expected to be CUDA-aware. [#45](https://github.com/ECP-copa/Cabana/pull/45)
README.md:GPU architectures. Cabana is built on Kokkos, with many additional
CMakeLists.txt:set(CABANA_SUPPORTED_DEVICES SERIAL THREADS OPENMP CUDA HIP SYCL OPENMPTARGET)
CMakeLists.txt:if(Kokkos_ENABLE_CUDA)
CMakeLists.txt:  kokkos_check(OPTIONS CUDA_LAMBDA)
docker/Dockerfile:ARG BASE=nvidia/cuda:11.0.3-devel-ubuntu20.04
docker/Dockerfile:    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/x86_64/3bf863cc.pub
docker/Dockerfile:# Install CUDA-aware Open MPI
docker/Dockerfile:    [ ! -z "${CUDA_VERSION}" ] && CUDA_OPTIONS=--with-cuda || true && \
docker/Dockerfile:    ../openmpi/configure --prefix=${OPENMPI_DIR} ${CUDA_OPTIONS} CFLAGS=-w && \
docker/Dockerfile:      -D Kokkos_ENABLE_CUDA=ON \
docker/Dockerfile:      -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
docker/Dockerfile:      -D Heffte_ENABLE_CUDA=ON \
docker/Dockerfile:      -D HYPRE_WITH_CUDA=ON \
docker/Dockerfile.hipcc:# Use -complete to get both rocm and rocfft
docker/Dockerfile.hipcc:ARG BASE=rocm/dev-ubuntu-20.04:5.2-complete
docker/Dockerfile.hipcc:ENV PATH=/opt/rocm/bin:$PATH
docker/Dockerfile.hipcc:      -D CMAKE_CXX_FLAGS="--amdgpu-target=gfx906" \
docker/Dockerfile.hipcc:      -D Heffte_ENABLE_ROCM=ON \
docker/Dockerfile.sycl:ARG BASE=nvidia/cuda:11.0.3-devel-ubuntu20.04
docker/Dockerfile.sycl:    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/x86_64/3bf863cc.pub
docker/Dockerfile.sycl:# Install Codeplay's oneAPI for NVIDIA GPUs, see
docker/Dockerfile.sycl:# https://developer.codeplay.com/products/oneapi/nvidia/2023.0.0/guides/get-started-guide-nvidia
docker/Dockerfile.sycl:RUN wget https://cloud.cees.ornl.gov/download/oneapi-for-nvidia-gpus-${DPCPP_VERSION}-linux.sh && \
docker/Dockerfile.sycl:    chmod +x oneapi-for-nvidia-gpus-${DPCPP_VERSION}-linux.sh && \
docker/Dockerfile.sycl:    ./oneapi-for-nvidia-gpus-${DPCPP_VERSION}-linux.sh -y && \
docker/Dockerfile.sycl:    rm oneapi-for-nvidia-gpus-${DPCPP_VERSION}-linux.sh
docker/Dockerfile.sycl:ARG KOKKOS_OPTIONS="-DKokkos_ENABLE_SYCL=ON -DCMAKE_CXX_FLAGS=-Wno-unknown-cuda-version -DKokkos_ENABLE_UNSUPPORTED_ARCHS=ON -DKokkos_ARCH_VOLTA70=ON -DCMAKE_CXX_STANDARD=17"
benchmark/core/Cabana_CommPerformance.cpp:            file << "Particle per MPI Rank/GPU: " << num_particle << "\n";
benchmark/core/Cabana_CommPerformance.cpp:            // Transfer GPU data to CPU, communication on CPU, and transfer back
benchmark/core/Cabana_CommPerformance.cpp:            // to GPU.
benchmark/plot/Cabana_BenchmarkPlotUtils.py:def getLegend(data: AllData, cpu_name, gpu_name, backend_label):
benchmark/plot/Cabana_BenchmarkPlotUtils.py:                legend.append(Line2D([0], [0], color="k", lw=2, linestyle="-", label=gpu_name+" GPU"))
benchmark/plot/Cabana_BenchmarkPlotUtils.py:def createPlot(fig, ax, data: AllData, speedup=False, backend_label=True, cpu_name="", gpu_name="", filename="Cabana_Benchmark.png", dpi=0):
benchmark/plot/Cabana_BenchmarkPlotUtils.py:    lines = getLegend(data, cpu_name, gpu_name, backend_label)
benchmark/plot/Cabana_PlotBenchmark.py:               speedup=speedup, backend_label=True)# cpu_name="EPYC", gpu_name="MI250X")
cmake/test_harness/test_harness.cmake:    if(_device STREQUAL CUDA)
cmake/test_harness/test_harness.cmake:      list(APPEND CABANA_TEST_DEVICES CUDA_UVM)
cmake/test_harness/TestCUDA_UVM_Category.hpp:#ifndef CABANA_TEST_CUDAUVM_CATEGORY_HPP
cmake/test_harness/TestCUDA_UVM_Category.hpp:#define CABANA_TEST_CUDAUVM_CATEGORY_HPP
cmake/test_harness/TestCUDA_UVM_Category.hpp:#define TEST_CATEGORY cuda_uvm
cmake/test_harness/TestCUDA_UVM_Category.hpp:#define TEST_EXECSPACE Kokkos::Cuda
cmake/test_harness/TestCUDA_UVM_Category.hpp:#define TEST_MEMSPACE Kokkos::CudaUVMSpace
cmake/test_harness/TestCUDA_UVM_Category.hpp:#define TEST_DEVICE Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>
cmake/test_harness/TestCUDA_UVM_Category.hpp:#endif // end CABANA_TEST_CUDAUVM_CATEGORY_HPP
cmake/test_harness/TestCUDA_Category.hpp:#ifndef CABANA_TEST_CUDA_CATEGORY_HPP
cmake/test_harness/TestCUDA_Category.hpp:#define CABANA_TEST_CUDA_CATEGORY_HPP
cmake/test_harness/TestCUDA_Category.hpp:#define TEST_CATEGORY cuda
cmake/test_harness/TestCUDA_Category.hpp:#define TEST_EXECSPACE Kokkos::Cuda
cmake/test_harness/TestCUDA_Category.hpp:#define TEST_MEMSPACE Kokkos::CudaSpace
cmake/test_harness/TestCUDA_Category.hpp:#define TEST_DEVICE Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>
cmake/test_harness/TestCUDA_Category.hpp:#endif // end CABANA_TEST_CUDA_CATEGORY_HPP
example/grid_tutorial/10_fft_heffte/heffte_fast_fourier_transform_example.cpp:        - cufft  (default with Cuda execution)
example/grid_tutorial/CMakeLists.txt:if(Cabana_ENABLE_HYPRE AND (NOT Kokkos_ENABLE_CUDA AND NOT Kokkos_ENABLE_HIP AND NOT Kokkos_ENABLE_SYCL))
example/core_tutorial/05_slice/slice_example.cpp:      Kokkos also supports execution on GPUs. For example, to create an AoSoA
example/core_tutorial/05_slice/slice_example.cpp:      allocated on NVIDIA devices use `Kokkos::CudaSpace` instead of
example/core_tutorial/06_deep_copy/deep_copy_example.cpp:      use for this type of capability is easily managing copies between a GPU
example/core_tutorial/06_deep_copy/deep_copy_example.cpp:      Given that the AoSoA we created above may be on the GPU we can easily
example/core_tutorial/10_simd_parallel_for/simd_parallel_for_example.cpp:      vector-length inner loop that will vectorize while on a GPU a 2D thread
example/core_tutorial/10_simd_parallel_for/simd_parallel_for_example.cpp:      CUDA UVM memory space is used this fence is necessary to ensure
example/core_tutorial/10_simd_parallel_for/simd_parallel_for_example.cpp:      the host. Not fencing in the case of using CUDA UVM will typically
example/core_tutorial/10_simd_parallel_for/simd_parallel_for_example.cpp:      access. In this case a 2D loop may be OK on the GPU due to the fact that
example/core_tutorial/05_slice_advanced_cuda/CMakeLists.txt:  add_executable(AdvancedCudaSlice advanced_slice_cuda.cpp)
example/core_tutorial/05_slice_advanced_cuda/CMakeLists.txt:  target_link_libraries(AdvancedCudaSlice Cabana::Core)
example/core_tutorial/05_slice_advanced_cuda/CMakeLists.txt:  add_test(NAME Cabana_Core_Tutorial_05_cuda COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:AdvancedCudaSlice>)
example/core_tutorial/05_slice_advanced_cuda/advanced_slice_cuda.cpp:#include <cuda_runtime.h>
example/core_tutorial/05_slice_advanced_cuda/advanced_slice_cuda.cpp:// Global Cuda function for initializing AoSoA data via the SoA accessor.
example/core_tutorial/05_slice_advanced_cuda/advanced_slice_cuda.cpp:// Atomic slice example using cuda.
example/core_tutorial/05_slice_advanced_cuda/advanced_slice_cuda.cpp:    std::cout << "Cabana Cuda Atomic Slice Example\n" << std::endl;
example/core_tutorial/05_slice_advanced_cuda/advanced_slice_cuda.cpp:    using MemorySpace = Kokkos::CudaUVMSpace;
example/core_tutorial/05_slice_advanced_cuda/advanced_slice_cuda.cpp:    using ExecutionSpace = Kokkos::Cuda;
example/core_tutorial/05_slice_advanced_cuda/advanced_slice_cuda.cpp:    int num_cuda_block = 1;
example/core_tutorial/05_slice_advanced_cuda/advanced_slice_cuda.cpp:    int cuda_block_size = 256;
example/core_tutorial/05_slice_advanced_cuda/advanced_slice_cuda.cpp:    atomicThreadSum<<<num_cuda_block, cuda_block_size>>>( atomic_slice );
example/core_tutorial/05_slice_advanced_cuda/advanced_slice_cuda.cpp:    cudaDeviceSynchronize();
example/core_tutorial/05_slice_advanced_cuda/advanced_slice_cuda.cpp:      Print out the slice result - it should be equal to the number of CUDA
example/core_tutorial/05_slice_advanced_cuda/advanced_slice_cuda.cpp:    std::cout << "Num CUDA threads = " << num_cuda_block * cuda_block_size
example/core_tutorial/12_halo_exchange/halo_exchange_example.cpp:      Note: The halo uses GPU-aware MPI communication. If AoSoA data is
example/core_tutorial/12_halo_exchange/halo_exchange_example.cpp:      allocated in GPU memory, this feature will be used automatically.
example/core_tutorial/CMakeLists.txt:if(Kokkos_ENABLE_CUDA)
example/core_tutorial/CMakeLists.txt:  add_subdirectory(05_slice_advanced_cuda)
example/core_tutorial/04_aosoa/aosoa_example.cpp:      (e.g. malloc() and free() or cudaMalloc() and cudaFree()). Depending on
example/core_tutorial/04_aosoa/aosoa_example.cpp:      Kokkos also supports execution on GPUs. For example, to create an
example/core_tutorial/04_aosoa/aosoa_example.cpp:      AoSoA allocated on NVIDIA devices use `Kokkos::CudaSpace` instead of
example/core_tutorial/11_migration/migration_example.cpp:      Note: The distributor uses GPU-aware MPI communication. If AoSoA data is
example/core_tutorial/11_migration/migration_example.cpp:      allocated in GPU memory, this feature will be used automatically.
example/core_tutorial/10_neighbor_parallel_for/neighbor_parallel_for_example.cpp:      CUDA UVM memory space is used this fence is necessary to ensure
example/core_tutorial/10_neighbor_parallel_for/neighbor_parallel_for_example.cpp:      the host. Not fencing in the case of using CUDA UVM will typically
grid/unit_test/tstFastFourierTransform.hpp:#if !defined( KOKKOS_ENABLE_CUDA ) && !defined( KOKKOS_ENABLE_HIP ) &&         \
grid/src/Cabana_Grid_FastFourierTransform.hpp:#ifdef Heffte_ENABLE_CUDA
grid/src/Cabana_Grid_FastFourierTransform.hpp:#ifdef KOKKOS_ENABLE_CUDA
grid/src/Cabana_Grid_FastFourierTransform.hpp:struct HeffteBackendTraits<Kokkos::Cuda, Impl::FFTBackendDefault>
grid/src/Cabana_Grid_FastFourierTransform.hpp:#ifdef Heffte_ENABLE_ROCM
grid/src/Cabana_Grid_Hypre.hpp:#ifdef HYPRE_USING_CUDA
grid/src/Cabana_Grid_Hypre.hpp:#ifdef KOKKOS_ENABLE_CUDA
grid/src/Cabana_Grid_Hypre.hpp://! Hypre device compatibility check - CUDA memory.
grid/src/Cabana_Grid_Hypre.hpp:struct HypreIsCompatibleWithMemorySpace<Kokkos::CudaSpace> : std::true_type
grid/src/Cabana_Grid_Hypre.hpp://! Hypre device compatibility check - CUDA UVM memory.
grid/src/Cabana_Grid_Hypre.hpp:struct HypreIsCompatibleWithMemorySpace<Kokkos::CudaUVMSpace> : std::true_type
grid/src/Cabana_Grid_Hypre.hpp:#endif // end KOKKOS_ENABLE_CUDA
grid/src/Cabana_Grid_Hypre.hpp:#endif // end HYPRE_USING_CUDA
grid/src/Cabana_Grid_Hypre.hpp:#ifndef HYPRE_USING_GPU
grid/src/Cabana_Grid_Hypre.hpp:#endif // end HYPRE_USING_GPU

```
