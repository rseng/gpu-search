# https://github.com/JetBrains-Research/spbla

```console
python/setup.py:        "Environment :: GPU",
python/setup.py:        "Environment :: GPU :: NVIDIA CUDA",
python/setup.py:        "nvidia-cuda",
python/setup.py:        "opencl"
python/README.md:work with sparse matrices written for CPU, Cuda and OpenCL platforms. The primary 
python/README.md:if Cuda/OpenCL compatible device is not presented in the system. This can be quite handy for 
python/README.md:- Cuda backend for computations
python/README.md:- OpenCL backend for computations
python/README.md:Machine configuration: PC with Ubuntu 20.04, Intel Core i7-6700 3.40GHz CPU, DDR4 64Gb RAM, GeForce GTX 1070 GPU with 8Gb VRAM. 
python/README.md:[link](https://github.com/YaccConstructor/articles/blob/master/2021/GRAPL/Sparse_Boolean_Algebra_on_GPGPU/Sparse_Boolean_Algebra_on_GPGPU.pdf).
python/README.md:  title = {spbla: sparse Boolean linear algebra for CPU, Cuda and OpenCL computations},
python/pyspbla/bridge.py:_hint_cuda_backend = 2
python/pyspbla/bridge.py:_hint_opencl_backend = 4
python/pyspbla/bridge.py:_hint_gpu_mem_managed = 8
python/pyspbla/bridge.py:_backend_name_cuda = "cuda"
python/pyspbla/bridge.py:_backend_name_opencl = "opencl"
python/pyspbla/bridge.py:    elif backend_type == _backend_name_cuda:
python/pyspbla/bridge.py:        hints |= _hint_cuda_backend
python/pyspbla/bridge.py:    elif backend_type == _backend_name_opencl:
python/pyspbla/bridge.py:        hints |= _hint_opencl_backend
python/pyspbla/bridge.py:/** No cuda compatible device in the system */
python/pyspbla/bridge.py:/** Failed to allocate memory on cpy or gpu side */
docs/joss/paper.bib:    title = {SPbLA: sparse Boolean linear algebra for CPU, Cuda and OpenCL computations},
docs/joss/paper.bib:    title       = {Sparse matrix library in Cuda},
docs/joss/paper.bib:    url         = {https://docs.nvidia.com/cuda/cusparse/},
docs/joss/paper.bib:    title = {A Framework for General Sparse Matrix-Matrix Multiplication on GPUs and Heterogeneous Processors},
docs/joss/paper.bib:    abstract = {General sparse matrix-matrix multiplication (SpGEMM) is a fundamental building block for numerous applications such as algebraic multigrid method (AMG), breadth first search and shortest path problem. Compared to other sparse BLAS routines, an efficient parallel SpGEMM implementation has to handle extra irregularity from three aspects: (1) the number of nonzero entries in the resulting sparse matrix is unknown in advance, (2) very expensive parallel insert operations at random positions in the resulting sparse matrix dominate the execution time, and (3) load balancing must account for sparse data in both input matrices.In this work we propose a framework for SpGEMM on GPUs and emerging CPU-GPU heterogeneous processors. This framework particularly focuses on the above three problems. Memory pre-allocation for the resulting matrix is organized by a hybrid method that saves a large amount of global memory space and efficiently utilizes the very limited on-chip scratchpad memory. Parallel insert operations of the nonzero entries are implemented through the GPU merge path algorithm that is experimentally found to be the fastest GPU merge approach. Load balancing builds on the number of necessary arithmetic operations on the nonzero entries and is guaranteed in all stages.Compared with the state-of-the-art CPU and GPU SpGEMM methods, our approach delivers excellent absolute performance and relative speedups on various benchmarks multiplying matrices with diverse sparsity structures. Furthermore, on heterogeneous processors, our SpGEMM approach achieves higher throughput by using re-allocatable shared virtual memory. We design a framework for SpGEMM on modern manycore processors using the CSR format.We present a hybrid method for pre-allocating the resulting sparse matrix.We propose an efficient parallel insert method for long rows of the resulting matrix.We develop a heuristic-based load balancing strategy.Our approach significantly outperforms other known CPU and GPU SpGEMM methods.},
docs/joss/paper.bib:    keywords = {Linear algebra, Sparse matrix, Sparse matrix-matrix multiplication, Heterogeneous processor, GPU, Merging, Parallel algorithm}
docs/joss/paper.bib:    title = {{GraphBLAST}: A High-Performance Linear Algebra-based Graph Framework on the {GPU}},
docs/joss/paper.bib:    abstract = {Sparse linear algebra is a cornerstone of modern computational science. These algorithms ignore the zero-valued entries found in many domains in order to work on much larger problems at much faster rates than dense algorithms. Nonetheless, optimizing these algorithms is not straightforward. Highly optimized algorithms for multiplying a sparse matrix by a dense vector, for instance, are the subject of a vast corpus of research and can be hundreds of times longer than na\"{\i}ve implementations. Optimized sparse linear algebra libraries are thus needed so that users can build applications without enormous effort.Hardware vendors release proprietary libraries that are highly optimized for their devices, but they limit interoperability and promote vendor lock-in. Open libraries often work across multiple devices and can quickly take advantage of new innovations, but they may not reach peak performance. The goal of this work is to provide a sparse linear algebra library that offers both of these advantages.We thus describe clSPARSE, a permissively licensed open-source sparse linear algebra library that offers state-of-the-art optimized algorithms implemented in OpenCL™. We test clSPARSE on GPUs from AMD and Nvidia and show performance benefits over both the proprietary cuSPARSE library and the open-source ViennaCL library.},
docs/joss/paper.bib:    booktitle = {Proceedings of the 4th International Workshop on OpenCL},
docs/joss/paper.bib:    keywords = {GPGPU, OpenCL, Sparse Linear Algebra, clSPARSE},
docs/joss/paper.json:  "description": "Sparse Boolean linear algebra for Nvidia Cuda, OpenCL and CPU computations",
docs/joss/paper.json:  "keywords": "python, cplusplus, sparse-matrix, linear-algebra, boolean-algebra, graph-analysis, graph-algorithms, nvidia-cuda, opencl",
docs/joss/paper.md:title: 'SPbLA: The Library of GPGPU-powered Sparse Boolean Linear Algebra Operations'
docs/joss/paper.md:  - nvidia-cuda
docs/joss/paper.md:  - opencl
docs/joss/paper.md:for GPGPU computations. It comes as a stand-alone self-sufficient 
docs/joss/paper.md:for Nvidia Cuda, OpenCL and CPU-only platforms. The library has 
docs/joss/paper.md:sparse linear algebra operations. While GPGPU utilization for high-performance linear algebra is common, 
docs/joss/paper.md:the high complexity of GPGPU programming makes the implementation of the complete set of sparse operations on GPGPU challenging. 
docs/joss/paper.md:and other problems that can be reduced to manipulation of Boolean matrices, to GPGPU uniformly. 
docs/joss/paper.md:handled on a single GPGPU. The creation of the library which supports multi-GPU and 
docs/joss/paper.md:the fully-featured sparse linear algebra as specified in `GraphBLAS` forward multi-GPU computations.
docs/joss/paper.md:GPGPU's utilization for data analysis and for linear algebra operations is a promising 
docs/joss/paper.md:way to high-performance data analysis because GPGPU is much more powerful in parallel
docs/joss/paper.md:data processing. However, GPGPU programming is still challenging.
docs/joss/paper.md:To the best of our knowledge, there is no complete `GraphBLAS API` implementation for GPGPU
docs/joss/paper.md:active development. Some work is also done to move `SuiteSparse` forward GPGPU computations.
docs/joss/paper.md:sparse linear algebra on GPGPU. There are a number of open-source and proprietary libraries,
docs/joss/paper.md:Results of the evaluation compared to CPU `SuiteSparse` and existing GPU sparse linear algebra libraries. 
docs/joss/paper.md:The comparison is not entirely fair, since there are still no Boolean linear algebra libraries for GPU computations.
docs/joss/paper.md:GeForce GTX 1070 GPU with 8Gb VRAM.
docs/joss/paper.md:`SPbLA` library shows the best performance among competitors for both OpenCL and Nvidia Cuda backends.
docs/joss/paper.md:First direction of the future research is library extension to multi-GPU environment support.
docs/joss/paper.md:GPU similarly as it is done for predefined Boolean values. 
docs/getting_started.md:- CUDA Compatible GPU device (to run Cuda computations)
docs/getting_started.md:- NVIDIA CUDA toolkit (to build Cuda backend)
docs/getting_started.md:### Cuda & compiler setup
docs/getting_started.md:> without cuda backend support.
docs/getting_started.md:Before the CUDA setup process, validate your system NVIDIA driver with `nvidia-smi`
docs/getting_started.md:The following commands grubs the required GCC compilers for the CC and CXX compiling respectively. CUDA toolkit, shipped
docs/getting_started.md:$ sudo apt install nvidia-cuda-toolkit
docs/getting_started.md:$ sudo apt install nvidia-cuda-dev 
docs/getting_started.md:$ nvcc: NVIDIA (R) Cuda compiler driver
docs/getting_started.md:$ Copyright (c) 2005-2019 NVIDIA Corporation
docs/getting_started.md:$ Cuda compilation tools, release 10.1, V10.1.243
docs/getting_started.md:**Bonus Step:** In order to have CUDA support in the CLion IDE, you will have to overwrite global alias for the `gcc`
docs/getting_started.md:- [NVIDIA Drivers installation Ubuntu](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux)
docs/getting_started.md:- [CUDA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
docs/getting_started.md:- [CUDA Hello world program](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
docs/getting_started.md:- [CUDA CMake tutorial](https://developer.nvidia.com/blog/building-cuda-applications-cmake/)
docs/getting_started.md:- `SPBLA_WITH_CUDA` - build library with actual cuda backend
docs/getting_started.md:- `SPBLA_WITH_OPENCL` - build library with actual cuda backend
docs/getting_started.md:- `SPBLA_WITH_CUB` - build library with bundled CUB sources, relevant for CUDA SDK 10 and earlier
docs/getting_started.md:> Note: in order to provide correct GCC version for CUDA sources compiling,
docs/getting_started.md:> $ export CUDAHOSTCXX=/usr/bin/g++-8
docs/getting_started.md:  (default backend will be selected), `cpu`, `cuda` and `opencl`.
docs/getting_started.md:# os.environ["SPBLA_BACKEND"] = "cuda"
docs/getting_started.md:# os.environ["SPBLA_BACKEND"] = "opencl"
docs/tutorial.md:[         1][         Level::Info] Cuda device is not presented
CHANGELOG.md:Project source code archive for JOSS publication `SPbLA: The Library of GPGPU-powered Sparse Boolean Linear Algebra Operations`
CHANGELOG.md:- nvidia-cuda
CHANGELOG.md:- opencl
CHANGELOG.md:**spbla** is a sparse linear Boolean algebra for Nvidia Cuda, OpenCL and CPU computations. 
CHANGELOG.md:CUDA C/C++, CUDA Thrust and OpenCL for actual backend implementation. 
CHANGELOG.md:- Cuda backend
CHANGELOG.md:- OpenCL backend
deps/nsparse/test/CMakeLists.txt:project(nsparse_um_test CXX CUDA)
deps/nsparse/test/CMakeLists.txt:set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_STANDARD 14)
deps/nsparse/test/CMakeLists.txt:set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_STANDARD_REQUIRED ON)
deps/nsparse/test/CMakeLists.txt:set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
deps/nsparse/test/src/nsparse_test.cu:nsparse::matrix<bool, index_type, alloc_type> dense_to_gpu_csr(const b_mat& matrix) {
deps/nsparse/test/src/nsparse_test.cu:  auto gpu_c = dense_to_gpu_csr(c);
deps/nsparse/test/src/nsparse_test.cu:  auto gpu_a = dense_to_gpu_csr(a);
deps/nsparse/test/src/nsparse_test.cu:  auto gpu_b = dense_to_gpu_csr(b);
deps/nsparse/test/src/nsparse_test.cu:  nsparse::masked_matrix<value_type, index_type> masked_a(gpu_a, a_values);
deps/nsparse/test/src/nsparse_test.cu:  nsparse::masked_matrix<value_type, index_type> masked_b(gpu_b, b_values);
deps/nsparse/test/src/nsparse_test.cu:  nsparse::masked_matrix<value_type, index_type> masked_c(gpu_c, -1);
deps/nsparse/README.md:cuda unified memory allocator for allocating large gpu resources.
deps/nsparse/README.md:For more info view branches `CFPQ-gpu` and `CFPQ-gpu-um`, where the source code
deps/nsparse/README.md:is stored at path `deps/cfpq/algorithms/cuda/nsparse`.
deps/nsparse/README.md:  Sparse General Matrix-Matrix Multiplication for NVIDIA Pascal GPU 
deps/nsparse/README.md:- GPU Merge Path - A GPU Merging Algorithm 
deps/nsparse/README.md:  [paper](https://www.researchgate.net/publication/254462662_GPU_merge_path_a_GPU_merging_algorithm)
deps/nsparse/include/nsparse/detail/count_nz.cuh:#include <cuda_runtime.h>
deps/nsparse/include/nsparse/detail/bitonic.cuh:#include <cuda_runtime.h>
deps/nsparse/include/nsparse/detail/masked_mult.cuh:#include <cuda_runtime.h>
deps/nsparse/include/nsparse/detail/count_nz.h:  cudaStream_t streams[9];
deps/nsparse/include/nsparse/detail/count_nz.h:      cudaStreamCreate( &s);
deps/nsparse/include/nsparse/detail/count_nz.h:      cudaStreamDestroy(s);
deps/nsparse/include/nsparse/detail/count_nz.h:    cudaDeviceSynchronize();
deps/nsparse/include/nsparse/detail/masked_mult.h:      cudaStreamCreate( &s);
deps/nsparse/include/nsparse/detail/masked_mult.h:  cudaStream_t streams[15];
deps/nsparse/include/nsparse/detail/masked_mult.h:      cudaStreamDestroy(s);
deps/nsparse/include/nsparse/detail/masked_mult.h:    cudaDeviceSynchronize();
deps/nsparse/include/nsparse/detail/add_values.cuh:#include <cuda_runtime.h>
deps/nsparse/include/nsparse/detail/fill_nz.cuh:#include <cuda_runtime.h>
deps/nsparse/include/nsparse/detail/merge_path.cuh:#include <cuda_runtime.h>
deps/nsparse/include/nsparse/detail/util.h:  cudaMemsetAsync(thrust::raw_pointer_cast(vec.data()), -1, sizeof(T) * size);
deps/nsparse/include/nsparse/detail/util.h:  cudaMemsetAsync(thrust::raw_pointer_cast(vec.data()), 0, sizeof(T) * size);
deps/nsparse/include/nsparse/detail/util.h:  cudaMemsetAsync(thrust::raw_pointer_cast(vec.data()), 0, sizeof(T) * size);
deps/nsparse/include/nsparse/masked_matrix.h:    cudaMemsetAsync(thrust::raw_pointer_cast(m_values.data()), default_value,
deps/nsparse/include/nsparse/unified_allocator.h:    inline cudaError_t cudaMallocManagedPrefetch(void **ptr, std::size_t bytes) {
deps/nsparse/include/nsparse/unified_allocator.h:        auto status = thrust::system::cuda::detail::cudaMallocManaged(ptr, bytes);
deps/nsparse/include/nsparse/unified_allocator.h:        if (status != cudaSuccess) {
deps/nsparse/include/nsparse/unified_allocator.h:        status = cudaGetDevice(&device);
deps/nsparse/include/nsparse/unified_allocator.h:        if (status != cudaSuccess) {
deps/nsparse/include/nsparse/unified_allocator.h:        status = cudaMemPrefetchAsync(*ptr, bytes, device, NULL);
deps/nsparse/include/nsparse/unified_allocator.h:    thrust::system::cuda::detail::cuda_memory_resource<cudaMallocManagedPrefetch, cudaFree,
deps/nsparse/include/nsparse/unified_allocator.h:            thrust::cuda::pointer<void>>;
deps/nsparse/CMakeLists.txt:project(nsparse LANGUAGES CXX CUDA)
deps/nsparse/CMakeLists.txt:target_compile_options(nsparse INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr --expt-extended-lambda>)
deps/nsparse/src/nsparse_index_path.cu:      cudaDeviceSynchronize();
deps/nsparse/src/nsparse_index_path.cu:    cudaDeviceSynchronize();
deps/clbool/libs/clew/CL/cl_d3d11.h:#ifndef __OPENCL_CL_D3D11_H
deps/clbool/libs/clew/CL/cl_d3d11.h:#define __OPENCL_CL_D3D11_H
deps/clbool/libs/clew/CL/cl_d3d11.h:#endif  /* __OPENCL_CL_D3D11_H */
deps/clbool/libs/clew/CL/opencl.h:#ifndef __OPENCL_H
deps/clbool/libs/clew/CL/opencl.h:#define __OPENCL_H
deps/clbool/libs/clew/CL/opencl.h:#endif  /* __OPENCL_H   */
deps/clbool/libs/clew/CL/cl_egl.h:#ifndef __OPENCL_CL_EGL_H
deps/clbool/libs/clew/CL/cl_egl.h:#define __OPENCL_CL_EGL_H
deps/clbool/libs/clew/CL/cl_egl.h:#endif /* __OPENCL_CL_EGL_H */
deps/clbool/libs/clew/CL/cl_version.h:#if !defined(CL_TARGET_OPENCL_VERSION)
deps/clbool/libs/clew/CL/cl_version.h:#pragma message("cl_version.h: CL_TARGET_OPENCL_VERSION is not defined. Defaulting to 220 (OpenCL 2.2)")
deps/clbool/libs/clew/CL/cl_version.h:#define CL_TARGET_OPENCL_VERSION 220
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION != 100 && \
deps/clbool/libs/clew/CL/cl_version.h:    CL_TARGET_OPENCL_VERSION != 110 && \
deps/clbool/libs/clew/CL/cl_version.h:    CL_TARGET_OPENCL_VERSION != 120 && \
deps/clbool/libs/clew/CL/cl_version.h:    CL_TARGET_OPENCL_VERSION != 200 && \
deps/clbool/libs/clew/CL/cl_version.h:    CL_TARGET_OPENCL_VERSION != 210 && \
deps/clbool/libs/clew/CL/cl_version.h:    CL_TARGET_OPENCL_VERSION != 220 && \
deps/clbool/libs/clew/CL/cl_version.h:    CL_TARGET_OPENCL_VERSION != 300
deps/clbool/libs/clew/CL/cl_version.h:#pragma message("cl_version: CL_TARGET_OPENCL_VERSION is not a valid value (100, 110, 120, 200, 210, 220, 300). Defaulting to 220 (OpenCL 2.2)")
deps/clbool/libs/clew/CL/cl_version.h:#undef CL_TARGET_OPENCL_VERSION
deps/clbool/libs/clew/CL/cl_version.h:#define CL_TARGET_OPENCL_VERSION 220
deps/clbool/libs/clew/CL/cl_version.h:/* OpenCL Version */
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION >= 300 && !defined(CL_VERSION_3_0)
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION >= 220 && !defined(CL_VERSION_2_2)
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION >= 210 && !defined(CL_VERSION_2_1)
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION >= 200 && !defined(CL_VERSION_2_0)
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION >= 120 && !defined(CL_VERSION_1_2)
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION >= 110 && !defined(CL_VERSION_1_1)
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION >= 100 && !defined(CL_VERSION_1_0)
deps/clbool/libs/clew/CL/cl_version.h:/* Allow deprecated APIs for older OpenCL versions. */
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION <= 220 && !defined(CL_USE_DEPRECATED_OPENCL_2_2_APIS)
deps/clbool/libs/clew/CL/cl_version.h:#define CL_USE_DEPRECATED_OPENCL_2_2_APIS
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION <= 210 && !defined(CL_USE_DEPRECATED_OPENCL_2_1_APIS)
deps/clbool/libs/clew/CL/cl_version.h:#define CL_USE_DEPRECATED_OPENCL_2_1_APIS
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION <= 200 && !defined(CL_USE_DEPRECATED_OPENCL_2_0_APIS)
deps/clbool/libs/clew/CL/cl_version.h:#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION <= 120 && !defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
deps/clbool/libs/clew/CL/cl_version.h:#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION <= 110 && !defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/cl_version.h:#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
deps/clbool/libs/clew/CL/cl_version.h:#if CL_TARGET_OPENCL_VERSION <= 100 && !defined(CL_USE_DEPRECATED_OPENCL_1_0_APIS)
deps/clbool/libs/clew/CL/cl_version.h:#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
deps/clbool/libs/clew/CL/cl_half.h: * This is a header-only utility library that provides OpenCL host code with
deps/clbool/libs/clew/CL/cl_half.h:#ifndef OPENCL_CL_HALF_H
deps/clbool/libs/clew/CL/cl_half.h:#define OPENCL_CL_HALF_H
deps/clbool/libs/clew/CL/cl_half.h:#endif  /* OPENCL_CL_HALF_H */
deps/clbool/libs/clew/CL/cl_ext.h:/* cl_ext.h contains OpenCL extensions which don't have external */
deps/clbool/libs/clew/CL/cl_ext.h:/* CL_DEVICE_DOUBLE_FP_CONFIG is defined in CL.h for OpenCL >= 120 */
deps/clbool/libs/clew/CL/cl_ext.h:#if CL_TARGET_OPENCL_VERSION <= 110
deps/clbool/libs/clew/CL/cl_ext.h: * OpenCL program is image2d_t. Both the sampler and sampler-less read_image
deps/clbool/libs/clew/CL/cl_ext.h: * This extension adds support to create an OpenCL program object from a
deps/clbool/libs/clew/CL/cl_ext.h:#define CL_DEVICE_GPU_OVERLAP_NV                    0x4004
deps/clbool/libs/clew/CL/cl_ext.h:* cl_ext_cxx_for_opencl extension
deps/clbool/libs/clew/CL/cl_ext.h:#define cl_ext_cxx_for_opencl 1
deps/clbool/libs/clew/CL/cl_ext.h:#define CL_DEVICE_CXX_FOR_OPENCL_NUMERIC_VERSION_EXT 0x4230
deps/clbool/libs/clew/CL/cl_ext.h:/* For OpenCL 2.1 and newer, cl_kernel_sub_group_info is declared in CL.h.
deps/clbool/libs/clew/CL/cl_ext.h:#define CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR           0x105F
deps/clbool/libs/clew/CL/cl_ext.h: * OpenCL via the clImportMemoryARM function.
deps/clbool/libs/clew/CL/cl_gl.h:#ifndef __OPENCL_CL_GL_H
deps/clbool/libs/clew/CL/cl_gl.h:#define __OPENCL_CL_GL_H
deps/clbool/libs/clew/CL/cl_gl.h:/* Deprecated OpenCL 1.1 APIs */
deps/clbool/libs/clew/CL/cl_gl.h:#endif  /* __OPENCL_CL_GL_H */
deps/clbool/libs/clew/CL/cl_va_api_media_sharing_intel.h:#ifndef __OPENCL_CL_VA_API_MEDIA_SHARING_INTEL_H
deps/clbool/libs/clew/CL/cl_va_api_media_sharing_intel.h:#define __OPENCL_CL_VA_API_MEDIA_SHARING_INTEL_H
deps/clbool/libs/clew/CL/cl_va_api_media_sharing_intel.h:#endif  /* __OPENCL_CL_VA_API_MEDIA_SHARING_INTEL_H */
deps/clbool/libs/clew/CL/cl2.hpp: *   \brief C++ bindings for OpenCL 1.0 (rev 48), OpenCL 1.1 (rev 33),
deps/clbool/libs/clew/CL/cl2.hpp: *       OpenCL 1.2 (rev 15) and OpenCL 2.0 (rev 29)
deps/clbool/libs/clew/CL/cl2.hpp: *   Derived from the OpenCL 1.x C++ bindings written by
deps/clbool/libs/clew/CL/cl2.hpp: *       http://khronosgroup.github.io/OpenCL-CLHPP/
deps/clbool/libs/clew/CL/cl2.hpp: *       https://github.com/KhronosGroup/OpenCL-CLHPP/releases
deps/clbool/libs/clew/CL/cl2.hpp: *       https://github.com/KhronosGroup/OpenCL-CLHPP
deps/clbool/libs/clew/CL/cl2.hpp: * reasonable to define C++ bindings for OpenCL.
deps/clbool/libs/clew/CL/cl2.hpp: * fixes in the new header as well as additional OpenCL 2.0 features.
deps/clbool/libs/clew/CL/cl2.hpp: * Due to the evolution of the underlying OpenCL API the 2.0 C++ bindings
deps/clbool/libs/clew/CL/cl2.hpp: * and the range of valid underlying OpenCL runtime versions supported.
deps/clbool/libs/clew/CL/cl2.hpp: * The combination of preprocessor macros CL_HPP_TARGET_OPENCL_VERSION and
deps/clbool/libs/clew/CL/cl2.hpp: * CL_HPP_MINIMUM_OPENCL_VERSION control this range. These are three digit
deps/clbool/libs/clew/CL/cl2.hpp: * decimal values representing OpenCL runime versions. The default for
deps/clbool/libs/clew/CL/cl2.hpp: * the target is 200, representing OpenCL 2.0 and the minimum is also
deps/clbool/libs/clew/CL/cl2.hpp: * The OpenCL 1.x versions of the C++ bindings included a size_t wrapper
deps/clbool/libs/clew/CL/cl2.hpp: * In OpenCL 2.0 OpenCL C is not entirely backward compatibility with
deps/clbool/libs/clew/CL/cl2.hpp: * earlier versions. As a result a flag must be passed to the OpenCL C
deps/clbool/libs/clew/CL/cl2.hpp: * compiled to request OpenCL 2.0 compilation of kernels with 1.2 as
deps/clbool/libs/clew/CL/cl2.hpp: * For those cases the compilation defaults to OpenCL C 2.0.
deps/clbool/libs/clew/CL/cl2.hpp: * - CL_HPP_TARGET_OPENCL_VERSION
deps/clbool/libs/clew/CL/cl2.hpp: *   Defines the target OpenCL runtime version to build the header
deps/clbool/libs/clew/CL/cl2.hpp: *   against. Defaults to 200, representing OpenCL 2.0.
deps/clbool/libs/clew/CL/cl2.hpp: *   Enables device fission for OpenCL 1.2 platforms.
deps/clbool/libs/clew/CL/cl2.hpp: *   Default to OpenCL C 1.2 compilation rather than OpenCL C 2.0
deps/clbool/libs/clew/CL/cl2.hpp:    #define CL_HPP_TARGET_OPENCL_VERSION 200
deps/clbool/libs/clew/CL/cl2.hpp:            if (platver.find("OpenCL 2.") != std::string::npos) {
deps/clbool/libs/clew/CL/cl2.hpp:            std::cout << "No OpenCL 2.0 platform found.";
deps/clbool/libs/clew/CL/cl2.hpp:#define CL_HPP_TARGET_OPENCL_VERSION 110
deps/clbool/libs/clew/CL/cl2.hpp:#if !defined(CL_HPP_TARGET_OPENCL_VERSION)
deps/clbool/libs/clew/CL/cl2.hpp:# pragma message("cl2.hpp: CL_HPP_TARGET_OPENCL_VERSION is not defined. It will default to 200 (OpenCL 2.0)")
deps/clbool/libs/clew/CL/cl2.hpp:# define CL_HPP_TARGET_OPENCL_VERSION 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION != 100 && CL_HPP_TARGET_OPENCL_VERSION != 110 && CL_HPP_TARGET_OPENCL_VERSION != 120 && CL_HPP_TARGET_OPENCL_VERSION != 200
deps/clbool/libs/clew/CL/cl2.hpp:# pragma message("cl2.hpp: CL_HPP_TARGET_OPENCL_VERSION is not a valid value (100, 110, 120 or 200). It will be set to 200")
deps/clbool/libs/clew/CL/cl2.hpp:# undef CL_HPP_TARGET_OPENCL_VERSION
deps/clbool/libs/clew/CL/cl2.hpp:# define CL_HPP_TARGET_OPENCL_VERSION 200
deps/clbool/libs/clew/CL/cl2.hpp:#if !defined(CL_HPP_MINIMUM_OPENCL_VERSION)
deps/clbool/libs/clew/CL/cl2.hpp:# define CL_HPP_MINIMUM_OPENCL_VERSION 110
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION != 100 && CL_HPP_MINIMUM_OPENCL_VERSION != 110 && CL_HPP_MINIMUM_OPENCL_VERSION != 120 && CL_HPP_MINIMUM_OPENCL_VERSION != 200
deps/clbool/libs/clew/CL/cl2.hpp:# pragma message("cl2.hpp: CL_HPP_MINIMUM_OPENCL_VERSION is not a valid value (100, 110, 120 or 200). It will be set to 100")
deps/clbool/libs/clew/CL/cl2.hpp:# undef CL_HPP_MINIMUM_OPENCL_VERSION
deps/clbool/libs/clew/CL/cl2.hpp:# define CL_HPP_MINIMUM_OPENCL_VERSION 100
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION > CL_HPP_TARGET_OPENCL_VERSION
deps/clbool/libs/clew/CL/cl2.hpp:# error "CL_HPP_MINIMUM_OPENCL_VERSION must not be greater than CL_HPP_TARGET_OPENCL_VERSION"
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 100 && !defined(CL_USE_DEPRECATED_OPENCL_1_0_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_1_0_APIS
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 110 && !defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_1_1_APIS
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 120 && !defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_1_2_APIS
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 200 && !defined(CL_USE_DEPRECATED_OPENCL_2_0_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_2_0_APIS
deps/clbool/libs/clew/CL/cl2.hpp:#include <OpenCL/opencl.h>
deps/clbool/libs/clew/CL/cl2.hpp:#include <CL/opencl.h>
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:        *  OpenCL C calls that require arrays of size_t values, whose
deps/clbool/libs/clew/CL/cl2.hpp: * \brief The OpenCL C++ bindings are defined within this namespace.
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:    F(cl_device_info, CL_DEVICE_OPENCL_C_VERSION, string) \
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:// Flags deprecated in OpenCL 2.0
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 100 && CL_HPP_MINIMUM_OPENCL_VERSION < 200 && CL_HPP_TARGET_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 110
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 110 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/cl2.hpp:#ifdef CL_DEVICE_GPU_OVERLAP_NV
deps/clbool/libs/clew/CL/cl2.hpp:        CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GPU_OVERLAP_NV, cl_bool)
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp: * OpenCL 1.2 devices do have retain/release.
deps/clbool/libs/clew/CL/cl2.hpp:#else // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp: * OpenCL 1.1 devices do not have retain/release.
deps/clbool/libs/clew/CL/cl2.hpp:#endif // ! (CL_HPP_TARGET_OPENCL_VERSION >= 120)
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/cl2.hpp:#else // CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:     *  \param devices returns a vector of OpenCL D3D10 devices found. The cl::Device
deps/clbool/libs/clew/CL/cl2.hpp:     *  values returned in devices can be used to identify a specific OpenCL
deps/clbool/libs/clew/CL/cl2.hpp:     *  The application can query specific capabilities of the OpenCL device(s)
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/cl2.hpp: * Unload the OpenCL compiler.
deps/clbool/libs/clew/CL/cl2.hpp: * \note Deprecated for OpenCL 1.2. Use Platform::unloadCompiler instead.
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:    /*! \brief Class interface for creating OpenCL buffers from ID3D10Buffer's.
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/cl2.hpp:            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
deps/clbool/libs/clew/CL/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:    *              The channel order may differ as described in the OpenCL
deps/clbool/libs/clew/CL/cl2.hpp:#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/cl2.hpp: *  \note Deprecated for OpenCL 1.2. Please use ImageGL instead.
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/cl2.hpp:            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
deps/clbool/libs/clew/CL/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif  // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp: * was performed by OpenCL anyway.
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:     * \param context A valid OpenCL context in which to construct the program.
deps/clbool/libs/clew/CL/cl2.hpp:     * \param devices A vector of OpenCL device objects for which the program will be created.
deps/clbool/libs/clew/CL/cl2.hpp:     *   CL_INVALID_DEVICE if OpenCL devices listed in devices are not in the list of devices associated with context.
deps/clbool/libs/clew/CL/cl2.hpp:     *   CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the OpenCL implementation on the host.
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#else // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:     *     The pattern type must be an accepted OpenCL data type.
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
deps/clbool/libs/clew/CL/cl2.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
deps/clbool/libs/clew/CL/cl2.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:#endif // defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp: * SVM buffer back to the OpenCL runtime.
deps/clbool/libs/clew/CL/cl2.hpp: * SVM buffer back to the OpenCL runtime.
deps/clbool/libs/clew/CL/cl2.hpp: * SVM buffer back to the OpenCL runtime.
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/cl.h:#ifndef __OPENCL_CL_H
deps/clbool/libs/clew/CL/cl.h:#define __OPENCL_CL_H
deps/clbool/libs/clew/CL/cl.h:#define CL_DEVICE_TYPE_GPU                          (1 << 2)
deps/clbool/libs/clew/CL/cl.h:#define CL_DEVICE_OPENCL_C_VERSION                       0x103D
deps/clbool/libs/clew/CL/cl.h:#define CL_DEVICE_OPENCL_C_ALL_VERSIONS                  0x1066
deps/clbool/libs/clew/CL/cl.h:#define CL_DEVICE_OPENCL_C_FEATURES                      0x106F
deps/clbool/libs/clew/CL/cl.h:#ifdef CL_USE_DEPRECATED_OPENCL_1_0_APIS
deps/clbool/libs/clew/CL/cl.h:     *     This API introduces mutable state into the OpenCL implementation. It has been REMOVED
deps/clbool/libs/clew/CL/cl.h:     *  OpenCL 1.1 conformance test, and consequently may not work or may not work dependably.
deps/clbool/libs/clew/CL/cl.h:#endif /* CL_USE_DEPRECATED_OPENCL_1_0_APIS */
deps/clbool/libs/clew/CL/cl.h:/* Deprecated OpenCL 1.1 APIs */
deps/clbool/libs/clew/CL/cl.h:/* Deprecated OpenCL 2.0 APIs */
deps/clbool/libs/clew/CL/cl.h:#endif  /* __OPENCL_CL_H */
deps/clbool/libs/clew/CL/cl_dx9_media_sharing_intel.h:#ifndef __OPENCL_CL_DX9_MEDIA_SHARING_INTEL_H
deps/clbool/libs/clew/CL/cl_dx9_media_sharing_intel.h:#define __OPENCL_CL_DX9_MEDIA_SHARING_INTEL_H
deps/clbool/libs/clew/CL/cl_dx9_media_sharing_intel.h:#endif  /* __OPENCL_CL_DX9_MEDIA_SHARING_INTEL_H */
deps/clbool/libs/clew/CL/cl_gl_ext.h:#ifndef __OPENCL_CL_GL_EXT_H
deps/clbool/libs/clew/CL/cl_gl_ext.h:#define __OPENCL_CL_GL_EXT_H
deps/clbool/libs/clew/CL/cl_gl_ext.h:#endif	/* __OPENCL_CL_GL_EXT_H  */
deps/clbool/libs/clew/CL/cl_icd.h:#ifndef OPENCL_CL_ICD_H
deps/clbool/libs/clew/CL/cl_icd.h:#define OPENCL_CL_ICD_H
deps/clbool/libs/clew/CL/cl_icd.h:/* OpenCL 1.1 */
deps/clbool/libs/clew/CL/cl_icd.h:  /* OpenCL 1.0 */
deps/clbool/libs/clew/CL/cl_icd.h:  /* OpenCL 1.1 */
deps/clbool/libs/clew/CL/cl_icd.h:  /* OpenCL 1.2 */
deps/clbool/libs/clew/CL/cl_icd.h:  /* OpenCL 2.0 */
deps/clbool/libs/clew/CL/cl_icd.h:  /* OpenCL 2.1 */
deps/clbool/libs/clew/CL/cl_icd.h:  /* OpenCL 2.2 */
deps/clbool/libs/clew/CL/cl_icd.h:  /* OpenCL 3.0 */
deps/clbool/libs/clew/CL/cl_icd.h:#endif /* #ifndef OPENCL_CL_ICD_H */
deps/clbool/libs/clew/CL/cl_d3d10.h:#ifndef __OPENCL_CL_D3D10_H
deps/clbool/libs/clew/CL/cl_d3d10.h:#define __OPENCL_CL_D3D10_H
deps/clbool/libs/clew/CL/cl_d3d10.h:#endif  /* __OPENCL_CL_D3D10_H */
deps/clbool/libs/clew/CL/cl_dx9_media_sharing.h:#ifndef __OPENCL_CL_DX9_MEDIA_SHARING_H
deps/clbool/libs/clew/CL/cl_dx9_media_sharing.h:#define __OPENCL_CL_DX9_MEDIA_SHARING_H
deps/clbool/libs/clew/CL/cl_dx9_media_sharing.h:#endif  /* __OPENCL_CL_DX9_MEDIA_SHARING_H */
deps/clbool/libs/clew/CL/opencl.hpp: *   \brief C++ bindings for OpenCL 1.0 (rev 48), OpenCL 1.1 (rev 33),
deps/clbool/libs/clew/CL/opencl.hpp: *       OpenCL 1.2 (rev 15), OpenCL 2.0 (rev 29), OpenCL 2.1 (rev 17),
deps/clbool/libs/clew/CL/opencl.hpp: *       and OpenCL 2.2 (V2.2-11).
deps/clbool/libs/clew/CL/opencl.hpp: *   Derived from the OpenCL 1.x C++ bindings written by
deps/clbool/libs/clew/CL/opencl.hpp: *       http://khronosgroup.github.io/OpenCL-CLHPP/
deps/clbool/libs/clew/CL/opencl.hpp: *       https://github.com/KhronosGroup/OpenCL-CLHPP/releases
deps/clbool/libs/clew/CL/opencl.hpp: *       https://github.com/KhronosGroup/OpenCL-CLHPP
deps/clbool/libs/clew/CL/opencl.hpp: * reasonable to define C++ bindings for OpenCL.
deps/clbool/libs/clew/CL/opencl.hpp: * The interface is contained with a single C++ header file \em opencl.hpp and all
deps/clbool/libs/clew/CL/opencl.hpp: * bindings; it is enough to simply include \em opencl.hpp.
deps/clbool/libs/clew/CL/opencl.hpp: * fixes in the new header as well as additional OpenCL 2.0 features.
deps/clbool/libs/clew/CL/opencl.hpp: * reason we release it as opencl.hpp rather than a new version of cl.hpp.
deps/clbool/libs/clew/CL/opencl.hpp: * Due to the evolution of the underlying OpenCL API the 2.0 C++ bindings
deps/clbool/libs/clew/CL/opencl.hpp: * and the range of valid underlying OpenCL runtime versions supported.
deps/clbool/libs/clew/CL/opencl.hpp: * The combination of preprocessor macros CL_HPP_TARGET_OPENCL_VERSION and
deps/clbool/libs/clew/CL/opencl.hpp: * CL_HPP_MINIMUM_OPENCL_VERSION control this range. These are three digit
deps/clbool/libs/clew/CL/opencl.hpp: * decimal values representing OpenCL runime versions. The default for
deps/clbool/libs/clew/CL/opencl.hpp: * the target is 200, representing OpenCL 2.0 and the minimum is also
deps/clbool/libs/clew/CL/opencl.hpp: * The OpenCL 1.x versions of the C++ bindings included a size_t wrapper
deps/clbool/libs/clew/CL/opencl.hpp: * In OpenCL 2.0 OpenCL C is not entirely backward compatibility with
deps/clbool/libs/clew/CL/opencl.hpp: * earlier versions. As a result a flag must be passed to the OpenCL C
deps/clbool/libs/clew/CL/opencl.hpp: * compiled to request OpenCL 2.0 compilation of kernels with 1.2 as
deps/clbool/libs/clew/CL/opencl.hpp: * For those cases the compilation defaults to OpenCL C 2.0.
deps/clbool/libs/clew/CL/opencl.hpp: * - CL_HPP_TARGET_OPENCL_VERSION
deps/clbool/libs/clew/CL/opencl.hpp: *   Defines the target OpenCL runtime version to build the header
deps/clbool/libs/clew/CL/opencl.hpp: *   against. Defaults to 200, representing OpenCL 2.0.
deps/clbool/libs/clew/CL/opencl.hpp: *   defined and may be defined by the user before opencl.hpp is
deps/clbool/libs/clew/CL/opencl.hpp: *   defined and may be defined by the user before opencl.hpp is
deps/clbool/libs/clew/CL/opencl.hpp: *   defined and may be defined by the user before opencl.hpp is
deps/clbool/libs/clew/CL/opencl.hpp: *   defined by the user before opencl.hpp is included.
deps/clbool/libs/clew/CL/opencl.hpp: *   Enables device fission for OpenCL 1.2 platforms.
deps/clbool/libs/clew/CL/opencl.hpp: *   Default to OpenCL C 1.2 compilation rather than OpenCL C 2.0
deps/clbool/libs/clew/CL/opencl.hpp:    #define CL_HPP_TARGET_OPENCL_VERSION 200
deps/clbool/libs/clew/CL/opencl.hpp:    #include <CL/opencl.hpp>
deps/clbool/libs/clew/CL/opencl.hpp:            if (platver.find("OpenCL 2.") != std::string::npos) {
deps/clbool/libs/clew/CL/opencl.hpp:            std::cout << "No OpenCL 2.0 platform found.";
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("opencl.hpp: USE_DX_INTEROP is deprecated. Define CL_HPP_USE_DX_INTEROP instead")
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("opencl.hpp: USE_CL_DEVICE_FISSION is deprecated. Define CL_HPP_USE_CL_DEVICE_FISSION instead")
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("opencl.hpp: __CL_ENABLE_EXCEPTIONS is deprecated. Define CL_HPP_ENABLE_EXCEPTIONS instead")
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("opencl.hpp: __NO_STD_VECTOR is deprecated. Define CL_HPP_NO_STD_VECTOR instead")
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("opencl.hpp: __NO_STD_STRING is deprecated. Define CL_HPP_NO_STD_STRING instead")
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("opencl.hpp: VECTOR_CLASS is deprecated. Alias cl::vector instead")
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("opencl.hpp: STRING_CLASS is deprecated. Alias cl::string instead.")
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("opencl.hpp: __CL_USER_OVERRIDE_ERROR_STRINGS is deprecated. Define CL_HPP_USER_OVERRIDE_ERROR_STRINGS instead")
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("opencl.hpp: __USE_DEV_VECTOR is no longer supported. Expect compilation errors")
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("opencl.hpp: __USE_DEV_STRING is no longer supported. Expect compilation errors")
deps/clbool/libs/clew/CL/opencl.hpp:#if !defined(CL_HPP_TARGET_OPENCL_VERSION)
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("opencl.hpp: CL_HPP_TARGET_OPENCL_VERSION is not defined. It will default to 220 (OpenCL 2.2)")
deps/clbool/libs/clew/CL/opencl.hpp:# define CL_HPP_TARGET_OPENCL_VERSION 220
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION != 100 && \
deps/clbool/libs/clew/CL/opencl.hpp:    CL_HPP_TARGET_OPENCL_VERSION != 110 && \
deps/clbool/libs/clew/CL/opencl.hpp:    CL_HPP_TARGET_OPENCL_VERSION != 120 && \
deps/clbool/libs/clew/CL/opencl.hpp:    CL_HPP_TARGET_OPENCL_VERSION != 200 && \
deps/clbool/libs/clew/CL/opencl.hpp:    CL_HPP_TARGET_OPENCL_VERSION != 210 && \
deps/clbool/libs/clew/CL/opencl.hpp:    CL_HPP_TARGET_OPENCL_VERSION != 220 && \
deps/clbool/libs/clew/CL/opencl.hpp:    CL_HPP_TARGET_OPENCL_VERSION != 300
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("opencl.hpp: CL_HPP_TARGET_OPENCL_VERSION is not a valid value (100, 110, 120, 200, 210, 220 or 300). It will be set to 220")
deps/clbool/libs/clew/CL/opencl.hpp:# undef CL_HPP_TARGET_OPENCL_VERSION
deps/clbool/libs/clew/CL/opencl.hpp:# define CL_HPP_TARGET_OPENCL_VERSION 220
deps/clbool/libs/clew/CL/opencl.hpp:/* Forward target OpenCL version to C headers if necessary */
deps/clbool/libs/clew/CL/opencl.hpp:#if defined(CL_TARGET_OPENCL_VERSION)
deps/clbool/libs/clew/CL/opencl.hpp:/* Warn if prior definition of CL_TARGET_OPENCL_VERSION is lower than
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_TARGET_OPENCL_VERSION < CL_HPP_TARGET_OPENCL_VERSION
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("CL_TARGET_OPENCL_VERSION is already defined as is lower than CL_HPP_TARGET_OPENCL_VERSION")
deps/clbool/libs/clew/CL/opencl.hpp:# define CL_TARGET_OPENCL_VERSION CL_HPP_TARGET_OPENCL_VERSION
deps/clbool/libs/clew/CL/opencl.hpp:#if !defined(CL_HPP_MINIMUM_OPENCL_VERSION)
deps/clbool/libs/clew/CL/opencl.hpp:# define CL_HPP_MINIMUM_OPENCL_VERSION 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION != 100 && \
deps/clbool/libs/clew/CL/opencl.hpp:    CL_HPP_MINIMUM_OPENCL_VERSION != 110 && \
deps/clbool/libs/clew/CL/opencl.hpp:    CL_HPP_MINIMUM_OPENCL_VERSION != 120 && \
deps/clbool/libs/clew/CL/opencl.hpp:    CL_HPP_MINIMUM_OPENCL_VERSION != 200 && \
deps/clbool/libs/clew/CL/opencl.hpp:    CL_HPP_MINIMUM_OPENCL_VERSION != 210 && \
deps/clbool/libs/clew/CL/opencl.hpp:    CL_HPP_MINIMUM_OPENCL_VERSION != 220 && \
deps/clbool/libs/clew/CL/opencl.hpp:    CL_HPP_MINIMUM_OPENCL_VERSION != 300
deps/clbool/libs/clew/CL/opencl.hpp:# pragma message("opencl.hpp: CL_HPP_MINIMUM_OPENCL_VERSION is not a valid value (100, 110, 120, 200, 210, 220 or 300). It will be set to 100")
deps/clbool/libs/clew/CL/opencl.hpp:# undef CL_HPP_MINIMUM_OPENCL_VERSION
deps/clbool/libs/clew/CL/opencl.hpp:# define CL_HPP_MINIMUM_OPENCL_VERSION 100
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION > CL_HPP_TARGET_OPENCL_VERSION
deps/clbool/libs/clew/CL/opencl.hpp:# error "CL_HPP_MINIMUM_OPENCL_VERSION must not be greater than CL_HPP_TARGET_OPENCL_VERSION"
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 100 && !defined(CL_USE_DEPRECATED_OPENCL_1_0_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:# define CL_USE_DEPRECATED_OPENCL_1_0_APIS
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 110 && !defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:# define CL_USE_DEPRECATED_OPENCL_1_1_APIS
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 120 && !defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:# define CL_USE_DEPRECATED_OPENCL_1_2_APIS
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 200 && !defined(CL_USE_DEPRECATED_OPENCL_2_0_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:# define CL_USE_DEPRECATED_OPENCL_2_0_APIS
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 210 && !defined(CL_USE_DEPRECATED_OPENCL_2_1_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:# define CL_USE_DEPRECATED_OPENCL_2_1_APIS
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 220 && !defined(CL_USE_DEPRECATED_OPENCL_2_2_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:# define CL_USE_DEPRECATED_OPENCL_2_2_APIS
deps/clbool/libs/clew/CL/opencl.hpp:#include <OpenCL/opencl.h>
deps/clbool/libs/clew/CL/opencl.hpp:#include <CL/opencl.h>
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:        *  OpenCL C calls that require arrays of size_t values, whose
deps/clbool/libs/clew/CL/opencl.hpp: * \brief The OpenCL C++ bindings are defined within this namespace.
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 220
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:    F(cl_device_info, CL_DEVICE_OPENCL_C_VERSION, string) \
deps/clbool/libs/clew/CL/opencl.hpp:    F(cl_device_info, CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR, cl_version_khr) \
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 220
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 220
deps/clbool/libs/clew/CL/opencl.hpp:#if defined(CL_HPP_USE_CL_SUB_GROUPS_KHR) && CL_HPP_TARGET_OPENCL_VERSION < 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if defined(CL_HPP_USE_CL_SUB_GROUPS_KHR) && CL_HPP_TARGET_OPENCL_VERSION < 210
deps/clbool/libs/clew/CL/opencl.hpp:// Flags deprecated in OpenCL 2.0
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 100 && CL_HPP_MINIMUM_OPENCL_VERSION < 200 && CL_HPP_TARGET_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 110
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 110 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#ifdef CL_DEVICE_GPU_OVERLAP_NV
deps/clbool/libs/clew/CL/opencl.hpp:        CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GPU_OVERLAP_NV, cl_bool)
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp: * OpenCL 1.2 devices do have retain/release.
deps/clbool/libs/clew/CL/opencl.hpp:#else // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp: * OpenCL 1.1 devices do not have retain/release.
deps/clbool/libs/clew/CL/opencl.hpp:#endif // ! (CL_HPP_TARGET_OPENCL_VERSION >= 120)
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/opencl.hpp:#else // CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:     *  \param devices returns a vector of OpenCL D3D10 devices found. The cl::Device
deps/clbool/libs/clew/CL/opencl.hpp:     *  values returned in devices can be used to identify a specific OpenCL
deps/clbool/libs/clew/CL/opencl.hpp:     *  The application can query specific capabilities of the OpenCL device(s)
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/opencl.hpp: * Unload the OpenCL compiler.
deps/clbool/libs/clew/CL/opencl.hpp: * \note Deprecated for OpenCL 1.2. Use Platform::unloadCompiler instead.
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:    /*! \brief Class interface for creating OpenCL buffers from ID3D10Buffer's.
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/opencl.hpp:            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
deps/clbool/libs/clew/CL/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 || defined(CL_HPP_USE_CL_IMAGE2D_FROM_BUFFER_KHR)
deps/clbool/libs/clew/CL/opencl.hpp:#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 200 || defined(CL_HPP_USE_CL_IMAGE2D_FROM_BUFFER_KHR)
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:    *              The channel order may differ as described in the OpenCL
deps/clbool/libs/clew/CL/opencl.hpp:#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/opencl.hpp: *  \note Deprecated for OpenCL 1.2. Please use ImageGL instead.
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/opencl.hpp:            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
deps/clbool/libs/clew/CL/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif  // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
deps/clbool/libs/clew/CL/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp: * was performed by OpenCL anyway.
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if (CL_HPP_TARGET_OPENCL_VERSION >= 200 && defined(CL_HPP_USE_CL_SUB_GROUPS_KHR)) || CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#else // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210 || (CL_HPP_TARGET_OPENCL_VERSION==200 && defined(CL_HPP_USE_IL_KHR))
deps/clbool/libs/clew/CL/opencl.hpp:     * Valid for either OpenCL >= 2.1 or when CL_HPP_USE_IL_KHR is defined.
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#else // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:     * Valid for either OpenCL >= 2.1 or when CL_HPP_USE_IL_KHR is defined.
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#else // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:     * \param context A valid OpenCL context in which to construct the kernel.
deps/clbool/libs/clew/CL/opencl.hpp:     * \param devices A vector of OpenCL device objects for which the kernel will be created.
deps/clbool/libs/clew/CL/opencl.hpp:     *   CL_INVALID_DEVICE if OpenCL devices listed in devices are not in the list of devices associated with context.
deps/clbool/libs/clew/CL/opencl.hpp:     *   CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the OpenCL implementation on the host.
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 220
deps/clbool/libs/clew/CL/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_2_2_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_2_2_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 220
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 220
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 220
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:                useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
deps/clbool/libs/clew/CL/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:               useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
deps/clbool/libs/clew/CL/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
deps/clbool/libs/clew/CL/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
deps/clbool/libs/clew/CL/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
deps/clbool/libs/clew/CL/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
deps/clbool/libs/clew/CL/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#else // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:     *     The pattern type must be an accepted OpenCL data type.
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
deps/clbool/libs/clew/CL/opencl.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
deps/clbool/libs/clew/CL/opencl.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:#endif // defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp: * SVM buffer back to the OpenCL runtime.
deps/clbool/libs/clew/CL/opencl.hpp: * SVM buffer back to the OpenCL runtime.
deps/clbool/libs/clew/CL/opencl.hpp: * SVM buffer back to the OpenCL runtime.
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
deps/clbool/libs/clew/CL/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
deps/clbool/libs/clew/CL/opencl.hpp:     * Backward compatibility class to ensure that cl.hpp code works with opencl.hpp.
deps/clbool/libs/clew/CL/cl_platform.h:#ifdef CL_USE_DEPRECATED_OPENCL_1_0_APIS
deps/clbool/libs/clew/CL/cl_platform.h:#ifdef CL_USE_DEPRECATED_OPENCL_1_1_APIS
deps/clbool/libs/clew/CL/cl_platform.h:#ifdef CL_USE_DEPRECATED_OPENCL_1_2_APIS
deps/clbool/libs/clew/CL/cl_platform.h:#ifdef CL_USE_DEPRECATED_OPENCL_2_0_APIS
deps/clbool/libs/clew/CL/cl_platform.h:#ifdef CL_USE_DEPRECATED_OPENCL_2_1_APIS
deps/clbool/libs/clew/CL/cl_platform.h:#ifdef CL_USE_DEPRECATED_OPENCL_2_2_APIS
deps/clbool/libs/clew/CL/cl_platform.h:/* Macro names and corresponding values defined by OpenCL */
deps/clbool/libs/clew/CL/cl_platform.h:/* Macro names and corresponding values defined by OpenCL */
deps/clbool/libs/clew/CL/cl_platform.h: *  Note:   OpenCL requires that all types be naturally aligned.
deps/clbool/libs/clew/CL/cl_platform.h: *   Each line thereafter of OpenCL C source must end with: \n\
deps/clbool/libs/clew/CL/cl_ext_intel.h:   OpenCL 1.2: */
deps/clbool/libs/CMakeLists.txt:add_subdirectory(gpu)
deps/clbool/libs/gpu/CMakeLists.txt:project(libgpu)
deps/clbool/libs/gpu/CMakeLists.txt:add_executable(hexdumparray libgpu/hexdumparray.cpp)
deps/clbool/libs/gpu/CMakeLists.txt:add_executable(make_source_map libgpu/make_source_map.cpp)
deps/clbool/libs/gpu/LICENSE:Copyright (c) 2018 GPGPUCourse2018
deps/clbool/libs/utils/libutils/fast_random.h:// taken from https://github.com/GPGPUCourse/GPGPUTasks2020
deps/clbool/libs/utils/libutils/timer.h:// taken from https://github.com/GPGPUCourse/GPGPUTasks2020/blob/task01/libs/utils/libutils/fast_random.h
deps/clbool/CMakeLists.txt:find_package(OpenCL 1.1 REQUIRED)
deps/clbool/CMakeLists.txt:target_link_libraries (clbool PUBLIC ${OpenCL_LIBRARY})
deps/clbool/CMakeLists.txt:target_compile_definitions(clbool PUBLIC CL_HPP_MINIMUM_OPENCL_VERSION=110)
deps/clbool/CMakeLists.txt:target_compile_definitions(clbool PUBLIC CL_HPP_TARGET_OPENCL_VERSION=110)
deps/clbool/src/core/matrix_csr.hpp:        cl::Buffer _rpt_gpu;
deps/clbool/src/core/matrix_csr.hpp:        cl::Buffer _cols_gpu;
deps/clbool/src/core/matrix_csr.hpp:        matrix_csr(cl::Buffer rpt_gpu,
deps/clbool/src/core/matrix_csr.hpp:                   cl::Buffer cols_gpu,
deps/clbool/src/core/matrix_csr.hpp:                , _rpt_gpu(std::move(rpt_gpu))
deps/clbool/src/core/matrix_csr.hpp:                , _cols_gpu(std::move(cols_gpu))
deps/clbool/src/core/matrix_csr.hpp:        const auto& rpt_gpu() const {
deps/clbool/src/core/matrix_csr.hpp:            return _rpt_gpu;
deps/clbool/src/core/matrix_csr.hpp:        const auto& cols_gpu() const {
deps/clbool/src/core/matrix_csr.hpp:            return _cols_gpu;
deps/clbool/src/core/error.hpp:#include "CL/opencl.hpp"
deps/clbool/src/core/matrix_coo.hpp:        const auto &rows_gpu() const {
deps/clbool/src/core/matrix_coo.hpp:        const auto &cols_gpu() const {
deps/clbool/src/core/matrix_coo.hpp:        auto &rows_gpu() {
deps/clbool/src/core/matrix_coo.hpp:        auto &cols_gpu() {
deps/clbool/src/core/matrix_dcsr.hpp:        cl::Buffer _rpt_gpu;
deps/clbool/src/core/matrix_dcsr.hpp:        cl::Buffer _rows_gpu;
deps/clbool/src/core/matrix_dcsr.hpp:        cl::Buffer _cols_gpu;
deps/clbool/src/core/matrix_dcsr.hpp:        matrix_dcsr(cl::Buffer rpt_gpu,
deps/clbool/src/core/matrix_dcsr.hpp:                    cl::Buffer rows_gpu,
deps/clbool/src/core/matrix_dcsr.hpp:                    cl::Buffer cols_gpu,
deps/clbool/src/core/matrix_dcsr.hpp:        , _rpt_gpu(std::move(rpt_gpu))
deps/clbool/src/core/matrix_dcsr.hpp:        , _rows_gpu(std::move(rows_gpu))
deps/clbool/src/core/matrix_dcsr.hpp:        , _cols_gpu(std::move(cols_gpu))
deps/clbool/src/core/matrix_dcsr.hpp:        const auto &rpt_gpu() const {
deps/clbool/src/core/matrix_dcsr.hpp:            return _rpt_gpu;
deps/clbool/src/core/matrix_dcsr.hpp:        const auto &rows_gpu() const {
deps/clbool/src/core/matrix_dcsr.hpp:            return _rows_gpu;
deps/clbool/src/core/matrix_dcsr.hpp:        const auto &cols_gpu() const {
deps/clbool/src/core/matrix_dcsr.hpp:            return _cols_gpu;
deps/clbool/src/core/matrix_dcsr.hpp:        auto &rpt_gpu()  {
deps/clbool/src/core/matrix_dcsr.hpp:            return _rpt_gpu;
deps/clbool/src/core/matrix_dcsr.hpp:        auto &rows_gpu()  {
deps/clbool/src/core/matrix_dcsr.hpp:            return _rows_gpu;
deps/clbool/src/core/matrix_dcsr.hpp:        auto &cols_gpu() {
deps/clbool/src/core/matrix_dcsr.hpp:            return _cols_gpu;
deps/clbool/src/dcsr/dcsr_submatrix.cpp:        // count [begin, end) of target rows_gpu
deps/clbool/src/dcsr/dcsr_submatrix.cpp:            cl::Buffer rows_begin_end_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * 2);
deps/clbool/src/dcsr/dcsr_submatrix.cpp:            find_range_program.run(controls, rows_begin_end_gpu, matrix_in.rows_gpu(), matrix_in.nzr(), i, nrows).wait();
deps/clbool/src/dcsr/dcsr_submatrix.cpp:            controls.queue.enqueueReadBuffer(rows_begin_end_gpu, CL_TRUE, 0, sizeof(uint32_t) * 2,
deps/clbool/src/dcsr/dcsr_submatrix.cpp:            LOG << "GPU rows_begin = " << rows_begin << ", rows_end = " << rows_end;
deps/clbool/src/dcsr/dcsr_submatrix.cpp:                              matrix_in.rpt_gpu(), matrix_in.rows_gpu(), matrix_in.cols_gpu(), matrix_in.nzr(),
deps/clbool/src/dcsr/dcsr_submatrix.cpp:            fill_rows_nnz.run(controls, subrows_nnz, cols_out, matrix_in.rpt_gpu(), matrix_in.cols_gpu(),
deps/clbool/src/dcsr/dcsr_submatrix.cpp:                                matrix_in.rows_gpu(), subrows_nnz, positions,
deps/clbool/src/dcsr/dcsr_transpose.cpp:                                         m_coo.cols_gpu(), m_coo.rows_gpu(),
deps/clbool/src/dcsr/dcsr_kronecker_product.cpp:                                   matrix_a.rpt_gpu(), matrix_b.rpt_gpu(),
deps/clbool/src/dcsr/dcsr_kronecker_product.cpp:                                   matrix_a.rows_gpu(), matrix_b.rows_gpu(),
deps/clbool/src/dcsr/dcsr_kronecker_product.cpp:                       kronecker.run(controls, c_rpt, c_cols, matrix_a.rpt_gpu(), matrix_b.rpt_gpu(),
deps/clbool/src/dcsr/dcsr_kronecker_product.cpp:                                     matrix_a.cols_gpu(), matrix_b.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.hpp:                     const cl::Buffer &gpu_workload_groups,
deps/clbool/src/dcsr/dcsr_matrix_multiplication.hpp:                         cl::Buffer &gpu_workload_groups,
deps/clbool/src/dcsr/dcsr_matrix_multiplication.hpp:                             const cl::Buffer &gpu_workload_groups,
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.hpp:                   const cl::Buffer &gpu_workload_groups,
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.hpp:                  const cl::Buffer &gpu_workload_groups,
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:        cl::Buffer gpu_workload_groups;
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:        CLB_CREATE_BUF(gpu_workload_groups = utils::create_buffer(controls, a.nzr()));
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:            write_bins_info(controls, gpu_workload_groups, cpu_workload_groups, groups_pointers, groups_length);
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                        gpu_workload_groups, nnz_estimation,
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                gpu_workload_groups, groups_pointers, groups_length,
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                             const cl::Buffer &gpu_workload_groups,
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:            CLB_RUN(e1 = single_value_rows.run(controls, gpu_workload_groups, groups_pointers[1], groups_length[1],
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                                 nnz_estimation, c_cols_indices, pre.rpt_gpu(), pre.cols_gpu()));
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                             gpu_workload_groups, groups_length[0] + groups_length[1],
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                             nnz_estimation, c_cols_indices, pre.rpt_gpu(), pre.cols_gpu()));
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:        // ------------------------------------  get rid of empty rows_gpu -------------------------------
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:        set_positions(controls, c_rpt, c_rows, nnz_estimation, a.rows_gpu(), positions, a.nzr());
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                         cl::Buffer &gpu_workload_groups,
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:            CLB_WRITE_BUF(controls.queue.enqueueWriteBuffer(gpu_workload_groups, CL_TRUE, sizeof(uint32_t) * offset,
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                     const cl::Buffer &gpu_workload_groups,
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                                  gpu_workload_groups, groups_pointers[workload_group_id],
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                                  pre.rpt_gpu(), pre.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                                  a.rpt_gpu(), a.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                                  b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                CLB_RUN(ev = heap_merge.run(controls, gpu_workload_groups, groups_pointers[workload_group_id],
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                              pre.rpt_gpu(), pre.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                              a.rpt_gpu(), a.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                              b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                        gpu_workload_groups, groups_pointers[workload_group_id],
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                        pre.rpt_gpu(), pre.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                        a.rpt_gpu(), a.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                        b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                                gpu_workload_groups, groups_pointers[workload_group_id],
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                                pre.rpt_gpu(), pre.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                                a.rpt_gpu(), a.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                                b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:        cl::Buffer pre_cols_indices_gpu;
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:        CLB_CREATE_BUF(pre_cols_indices_gpu = utils::create_buffer(controls,  pre_nnz));
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:        pre = matrix_dcsr(pre_rows_pointers, a.rows_gpu(), pre_cols_indices_gpu,
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:        CLB_RUN(count_workload.run(controls, nnz_estimation, a.rpt_gpu(), a.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication.cpp:                                     b.rows_gpu(), b.rpt_gpu(), a.nzr(), b.nzr()))
deps/clbool/src/dcsr/dcsr_reduce.cpp:            rows = matrix_in.rows_gpu();
deps/clbool/src/dcsr/dcsr_reduce.cpp:            controls.queue.enqueueCopyBuffer(matrix_in.rows_gpu(), rows, 0, 0, sizeof (uint32_t) * matrix_in.nzr());
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:            // NOTE: NVIDIA can operate more than 256 threads per group, but AMD cannot
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:        cl::Buffer gpu_workload_groups(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a.nzr());
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:            write_bins_info(controls, gpu_workload_groups, cpu_workload_groups, groups_pointers, groups_length);
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:            count_nnz(controls, groups_length, groups_pointers, gpu_workload_groups, nnz_estimation,
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:            fill_nnz(controls, groups_length, groups_pointers, gpu_workload_groups, nnz_estimation,
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                   const cl::Buffer &gpu_workload_groups,
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                ev = hash_pwarp.run(controls, gpu_workload_groups, groups_pointers[bin_id], groups_length[bin_id],
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                                       nnz_estimation, a.rpt_gpu(), a.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                                       b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                ev = hash_tb.run(controls, gpu_workload_groups, groups_pointers[bin_id], groups_length[bin_id],
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                                             nnz_estimation, a.rpt_gpu(), a.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                                             b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:            ev = hash_global.run(controls, gpu_workload_groups, groups_pointers[bin_id], groups_length[bin_id],
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                                             nnz_estimation, a.rpt_gpu(), a.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                                             b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                  const cl::Buffer &gpu_workload_groups,
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                ev = hash_pwarp.run(controls, gpu_workload_groups, groups_pointers[bin_id], groups_length[bin_id],
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                                       pre_matrix_rows_pointers, c_cols, a.rpt_gpu(), a.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                                       b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                ev = hash_tb.run(controls, gpu_workload_groups, groups_pointers[bin_id],
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                                             pre_matrix_rows_pointers, c_cols, a.rpt_gpu(), a.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:                                             b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:            CLB_RUN(ev = hash_global.run(controls, gpu_workload_groups, groups_pointers[bin_id],
deps/clbool/src/dcsr/dcsr_matrix_multiplication_hash.cpp:        set_positions(controls, c_rpt, c_rows, pre_matrix_rows_pointers, a.rows_gpu(), positions, a.nzr());
deps/clbool/src/test/test_pref_sum.cpp:        cl::Buffer vec_gpu(controls.queue, vec.begin(), vec.end(), false);
deps/clbool/src/test/test_pref_sum.cpp:            prefix_sum(controls, vec_gpu, total, size + 1);
deps/clbool/src/test/test_pref_sum.cpp:        compare_buffers(controls, vec_gpu, vec, size + 1);
deps/clbool/src/test/test_new_merge.cpp:            cl::Buffer a_gpu = cl::Buffer(controls.queue, a_cpu.begin(), a_cpu.end(), false);
deps/clbool/src/test/test_new_merge.cpp:            cl::Buffer b_gpu = cl::Buffer(controls.queue, b_cpu.begin(), b_cpu.end(), false);
deps/clbool/src/test/test_new_merge.cpp:            cl::Buffer c_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_cpu.size());
deps/clbool/src/test/test_new_merge.cpp:                        a_gpu, b_gpu, c_gpu, a_cpu.size(), b_cpu.size());
deps/clbool/src/test/test_new_merge.cpp://            print_gpu_buffer(controls, c_gpu, c_cpu.size());
deps/clbool/src/test/test_new_merge.cpp:            compare_buffers(controls, c_gpu, c_cpu, c_cpu.size());
deps/clbool/src/test/test_reduce.cpp:            matrix_dcsr a_gpu;
deps/clbool/src/test/test_reduce.cpp:                a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
deps/clbool/src/test/test_reduce.cpp:                reduce(controls, a_gpu, a_gpu);
deps/clbool/src/test/test_reduce.cpp:            compare_matrices(controls, a_gpu, a_cpu);
deps/clbool/src/test/esc_kernels_test.cpp:        matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
deps/clbool/src/test/esc_kernels_test.cpp:        matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);
deps/clbool/src/test/esc_kernels_test.cpp:        matrix_dcsr c_gpu;
deps/clbool/src/test/esc_kernels_test.cpp://    print_matrix(controls, a_gpu);
deps/clbool/src/test/esc_kernels_test.cpp://    print_matrix(controls, b_gpu);
deps/clbool/src/test/esc_kernels_test.cpp:        matrix_multiplication(controls, c_gpu, a_gpu, b_gpu);
deps/clbool/src/test/esc_kernels_test.cpp:        compare_matrices(controls, c_gpu, c_cpu);
deps/clbool/src/test/count_workload_test.cpp:        matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
deps/clbool/src/test/count_workload_test.cpp://    matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);
deps/clbool/src/test/count_workload_test.cpp:        // get workload from gpu
deps/clbool/src/test/count_workload_test.cpp:        count_workload(controls, nnz_estimation, a_gpu, a_gpu);
deps/clbool/src/test/count_workload_test.cpp:        std::cout << "finish gpu counting" << std::endl;
deps/clbool/src/test/count_workload_test.cpp:        cpu_buffer nnz_estimation_cpu(a_gpu.nzr());
deps/clbool/src/test/count_workload_test.cpp:        utils::compare_buffers(controls, nnz_estimation, nnz_estimation_cpu, a_gpu.nzr());
deps/clbool/src/test/coo_kronecker_test.cpp:    matrix_coo matrix_res_gpu;
deps/clbool/src/test/coo_kronecker_test.cpp:    matrix_coo matrix_a_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_a_cpu);
deps/clbool/src/test/coo_kronecker_test.cpp:    matrix_coo matrix_b_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_b_cpu);
deps/clbool/src/test/coo_kronecker_test.cpp:    kronecker_product(controls, matrix_res_gpu, matrix_a_gpu, matrix_b_gpu);
deps/clbool/src/test/coo_kronecker_test.cpp:    utils::compare_buffers(controls, matrix_res_gpu.rows_gpu(), rows_cpu, rows_cpu.size());
deps/clbool/src/test/coo_kronecker_test.cpp:    utils::compare_buffers(controls, matrix_res_gpu.cols_gpu(), cols_cpu, cols_cpu.size());
deps/clbool/src/test/test_transpose.cpp:            matrix_dcsr a_gpu;
deps/clbool/src/test/test_transpose.cpp:                a_gpu = matrix_dcsr_from_cpu(controls, a_dcsr_cpu, max_size);
deps/clbool/src/test/test_transpose.cpp:                transpose(controls, a_gpu, a_gpu);
deps/clbool/src/test/test_transpose.cpp:            compare_matrices(controls, a_gpu, a_dcsr_cpu_tr);
deps/clbool/src/test/large_rows_test.cpp:    matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
deps/clbool/src/test/large_rows_test.cpp:    matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);
deps/clbool/src/test/large_rows_test.cpp:    matrix_dcsr c_gpu;
deps/clbool/src/test/large_rows_test.cpp:    matrix_multiplication(controls, c_gpu, a_gpu, b_gpu);
deps/clbool/src/test/large_rows_test.cpp://    print_matrix(controls, c_gpu);
deps/clbool/src/test/large_rows_test.cpp:    compare_matrices(controls, c_gpu, c_cpu);
deps/clbool/src/test/check_objects_copying.cpp:    matrix_dcsr a_dcsr = coo_to_dcsr_gpu_shallow(controls, a_coo);
deps/clbool/src/test/check_objects_copying.cpp:    controls.queue.enqueueWriteBuffer(a_coo.cols_gpu(), CL_TRUE, 0, sizeof(uint32_t) * 1, &value);
deps/clbool/src/test/check_objects_copying.cpp:    print_gpu_buffer(controls, a_dcsr.cols_gpu(), 3);
deps/clbool/src/test/check_objects_copying.cpp:    cl::Buffer a_gpu(controls.context, CL_TRUE, sizeof(uint32_t) * a_cpu.size());
deps/clbool/src/test/check_objects_copying.cpp:    cl::Buffer a_gpu_copy(controls.context, CL_TRUE, sizeof(uint32_t) * a_cpu.size());
deps/clbool/src/test/check_objects_copying.cpp:    controls.queue.enqueueWriteBuffer(a_gpu, CL_TRUE, 0, sizeof(uint32_t) * a_cpu.size(), a_cpu.data());
deps/clbool/src/test/check_objects_copying.cpp:    print_gpu_buffer(controls, a_gpu_copy, a_cpu.size());
deps/clbool/src/test/check_objects_copying.cpp:    controls.queue.enqueueCopyBuffer(a_gpu, a_gpu_copy, 0, 0, sizeof(uint32_t) * a_cpu.size());
deps/clbool/src/test/check_objects_copying.cpp:    controls.queue.enqueueWriteBuffer(a_gpu, CL_TRUE, 0, sizeof(uint32_t) * b_cpu.size(), b_cpu.data());
deps/clbool/src/test/check_objects_copying.cpp:    print_gpu_buffer(controls, a_gpu, a_cpu.size());
deps/clbool/src/test/check_objects_copying.cpp:    print_gpu_buffer(controls, a_gpu_copy, a_cpu.size());
deps/clbool/src/test/test_submatrix.cpp:                matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
deps/clbool/src/test/test_submatrix.cpp:                matrix_dcsr c_gpu;
deps/clbool/src/test/test_submatrix.cpp:                    submatrix(controls, c_gpu, a_gpu, i, j, nrows, ncols);
deps/clbool/src/test/test_submatrix.cpp:                compare_matrices(controls, c_gpu, c_cpu);
deps/clbool/src/test/test_half_sized_scan.cpp:    cl::Buffer array_gpu(controls.context, CL_MEM_READ_WRITE,  array.size() * sizeof(cpu_buffer::value_type));
deps/clbool/src/test/test_half_sized_scan.cpp:    controls.queue.enqueueWriteBuffer(array_gpu, CL_TRUE, 0, array.size()
deps/clbool/src/test/test_half_sized_scan.cpp:    utils::print_gpu_buffer(controls, array_gpu, array.size());
deps/clbool/src/test/test_half_sized_scan.cpp:        half_sized_scan.run(controls, array_gpu, array.size());
deps/clbool/src/test/test_half_sized_scan.cpp:        utils::print_gpu_buffer(controls, array_gpu, array.size());
deps/clbool/src/test/count_workload_and_allocation_test.cpp:    matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
deps/clbool/src/test/count_workload_and_allocation_test.cpp:    matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);
deps/clbool/src/test/count_workload_and_allocation_test.cpp:    // get workload from gpu
deps/clbool/src/test/count_workload_and_allocation_test.cpp:    count_workload(controls, nnz_estimation, a_gpu, b_gpu);
deps/clbool/src/test/count_workload_and_allocation_test.cpp:    print_gpu_buffer(controls, nnz_estimation, a_gpu.nzr());
deps/clbool/src/test/count_workload_and_allocation_test.cpp:                                         cpu_workload_groups, nnz_estimation, a_gpu, b_gpu.ncols(), a, b);
deps/clbool/src/test/count_workload_and_allocation_test.cpp:    cl::Buffer gpu_workload_groups(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a_gpu.nzr());
deps/clbool/src/test/count_workload_and_allocation_test.cpp:        controls.queue.enqueueWriteBuffer(gpu_workload_groups, CL_TRUE, sizeof(uint32_t) * offset,
deps/clbool/src/test/count_workload_and_allocation_test.cpp:    std::cout << "gpu workload: \n";
deps/clbool/src/test/count_workload_and_allocation_test.cpp:    utils::print_gpu_buffer(controls, gpu_workload_groups, offset);
deps/clbool/src/test/count_workload_and_allocation_test.cpp:    utils::print_gpu_buffer(controls, pre.rpt_gpu(), a_gpu.nzr() + 1);
deps/clbool/src/test/heap_kernels_test.cpp:    matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
deps/clbool/src/test/heap_kernels_test.cpp:    matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);
deps/clbool/src/test/heap_kernels_test.cpp:    count_workload(controls, nnz_estimation, a_gpu, b_gpu);
deps/clbool/src/test/heap_kernels_test.cpp:                                         cpu_workload_groups, nnz_estimation, a_gpu, b_gpu.ncols(),
deps/clbool/src/test/heap_kernels_test.cpp:    cl::Buffer gpu_workload_groups(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a_gpu.nzr());
deps/clbool/src/test/heap_kernels_test.cpp:    write_bins_info(controls, gpu_workload_groups, cpu_workload_groups, groups_pointers, groups_length);
deps/clbool/src/test/heap_kernels_test.cpp:    std::cout << "pre_rows_pointers: \n"; utils::print_gpu_buffer(controls, pre.rpt_gpu(), a_gpu.nzr() + 1);
deps/clbool/src/test/heap_kernels_test.cpp:    std::cout << "gpu_workload_groups: \n"; utils::print_gpu_buffer(controls, gpu_workload_groups, a_gpu.nzr());
deps/clbool/src/test/heap_kernels_test.cpp:                gpu_workload_groups, nnz_estimation,
deps/clbool/src/test/heap_kernels_test.cpp:                pre, a_gpu, b_gpu,
deps/clbool/src/test/heap_kernels_test.cpp:    matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
deps/clbool/src/test/heap_kernels_test.cpp:    matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);
deps/clbool/src/test/heap_kernels_test.cpp:    matrix_dcsr c_gpu;
deps/clbool/src/test/heap_kernels_test.cpp:    matrix_multiplication(controls, c_gpu, a_gpu, b_gpu);
deps/clbool/src/test/heap_kernels_test.cpp:    compare_matrices(controls, c_gpu, c_cpu);
deps/clbool/src/test/heap_kernels_test.cpp:    print_matrix(controls, c_gpu);
deps/clbool/src/test/test_multiplication.cpp:            matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
deps/clbool/src/test/test_multiplication.cpp:            matrix_dcsr c_gpu;
deps/clbool/src/test/test_multiplication.cpp:            matrix_multiplication(controls, c_gpu, a_gpu, a_gpu);
deps/clbool/src/test/test_multiplication.cpp://        print_matrix(controls, c_gpu, 69);
deps/clbool/src/test/test_multiplication.cpp:            compare_matrices(controls, c_gpu, c_cpu);
deps/clbool/src/test/test_multiplication.cpp:            matrix_dcsr a_gpu;
deps/clbool/src/test/test_multiplication.cpp:                a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
deps/clbool/src/test/test_multiplication.cpp:            matrix_dcsr c_gpu;
deps/clbool/src/test/test_multiplication.cpp:                matrix_multiplication_hash(controls, c_gpu, a_gpu, a_gpu);
deps/clbool/src/test/test_multiplication.cpp:            compare_matrices(controls, c_gpu, c_cpu);
deps/clbool/src/test/coo_addition_test.cpp:            matrix_coo matrix_res_gpu;
deps/clbool/src/test/coo_addition_test.cpp:            matrix_coo matrix_a_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_a_cpu);
deps/clbool/src/test/coo_addition_test.cpp:            matrix_coo matrix_b_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_b_cpu);
deps/clbool/src/test/coo_addition_test.cpp:            matrix_addition(controls, matrix_res_gpu, matrix_a_gpu, matrix_b_gpu);
deps/clbool/src/test/coo_addition_test.cpp:            utils::compare_buffers(controls, matrix_res_gpu.rows_gpu(), rows_cpu, rows_cpu.size());
deps/clbool/src/test/coo_addition_test.cpp:            utils::compare_buffers(controls, matrix_res_gpu.cols_gpu(), cols_cpu, cols_cpu.size());
deps/clbool/src/test/coo_to_dcsr_test.cpp:    matrix_coo matrix_gpu = coo_utils::matrix_coo_from_cpu(controls, m_cpu);
deps/clbool/src/test/coo_to_dcsr_test.cpp:    cpu_buffer coo_rows_indices(matrix_gpu.nnz());
deps/clbool/src/test/coo_to_dcsr_test.cpp:    cpu_buffer coo_cols_indices(matrix_gpu.nnz());
deps/clbool/src/test/coo_to_dcsr_test.cpp:    controls.queue.enqueueReadBuffer(matrix_gpu.rows_gpu(), true, 0,
deps/clbool/src/test/coo_to_dcsr_test.cpp:                                      sizeof(matrix_coo::index_type) * matrix_gpu.nnz(), coo_rows_indices.data());
deps/clbool/src/test/coo_to_dcsr_test.cpp:    controls.queue.enqueueReadBuffer(matrix_gpu.cols_gpu(), true, 0,
deps/clbool/src/test/coo_to_dcsr_test.cpp:                                      sizeof(matrix_coo::index_type) * matrix_gpu.nnz(), coo_cols_indices.data());
deps/clbool/src/test/coo_to_dcsr_test.cpp:    matrix_dcsr m_dcsr = coo_to_dcsr_gpu_shallow(controls, matrix_gpu);
deps/clbool/src/test/coo_to_dcsr_test.cpp:    utils::compare_buffers(controls, m_dcsr.rpt_gpu(), rows_pointers_cpu, rows_pointers_cpu.size());
deps/clbool/src/test/coo_to_dcsr_test.cpp:    utils::compare_buffers(controls, m_dcsr.rows_gpu(), rows_compressed_cpu, rows_compressed_cpu.size());
deps/clbool/src/test/coo_to_dcsr_test.cpp:    utils::print_gpu_buffer(controls, another_one.cols_gpu(), 10);
deps/clbool/src/test/coo_to_dcsr_test.cpp:    utils::print_gpu_buffer(controls, another_one.rows_gpu(), 10);
deps/clbool/src/test/coo_to_dcsr_test.cpp:    utils::compare_buffers(controls, another_one.rows_gpu(), coo_rows_indices, coo_rows_indices.size());
deps/clbool/src/test/coo_to_dcsr_test.cpp:    utils::compare_buffers(controls, another_one.cols_gpu(), coo_cols_indices, coo_cols_indices.size());
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    std::vector<uint32_t> rows_from_gpu(size);
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    std::vector<uint32_t> cols_from_gpu(size);
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    // -------------------- create and sort gpu buffers ----------------------------
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    cl::Buffer rows_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    cl::Buffer cols_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    controls.queue.enqueueWriteBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, rows_cpu.data());
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    controls.queue.enqueueWriteBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, cols_cpu.data());
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    sort_arrays(controls, rows_gpu, cols_gpu, size);
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    // ------------------ now reduce gpu buffers and read in vectors ------------------------
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    reduce_duplicates(controls, rows_gpu, cols_gpu, reinterpret_cast<uint32_t &>(new_size), size);
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    rows_from_gpu.resize(new_size);
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    cols_from_gpu.resize(new_size);
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    controls.queue.enqueueReadBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * new_size, rows_from_gpu.data());
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    controls.queue.enqueueReadBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * new_size, cols_from_gpu.data());
deps/clbool/src/test/coo_reduce_duplicates_test.cpp:    if (rows_from_gpu == rows_cpu && cols_from_gpu == cols_cpu) {
deps/clbool/src/test/coo_bitonic_test.cpp:        cl::Buffer rows_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
deps/clbool/src/test/coo_bitonic_test.cpp:        cl::Buffer cols_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
deps/clbool/src/test/coo_bitonic_test.cpp:        controls.queue.enqueueWriteBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, rows_cpu.data());
deps/clbool/src/test/coo_bitonic_test.cpp:        controls.queue.enqueueWriteBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, cols_cpu.data());
deps/clbool/src/test/coo_bitonic_test.cpp:            sort_arrays(controls, rows_gpu, cols_gpu, size);
deps/clbool/src/test/coo_bitonic_test.cpp:        utils::compare_buffers(controls, rows_gpu, rows_cpu, size, "rows_gpu");
deps/clbool/src/test/coo_bitonic_test.cpp:        utils::compare_buffers(controls, cols_gpu, cols_cpu, size, "cols_gpu");
deps/clbool/src/common/cl_includes.hpp:#include "CL/opencl.hpp"
deps/clbool/src/common/matrices_conversions.cpp:                                  uint32_t &nzr // non zero rows_gpu
deps/clbool/src/common/matrices_conversions.cpp:        dscr_to_coo.run(controls, a.rpt_gpu(), a.rows_gpu(), c_rows);
deps/clbool/src/common/matrices_conversions.cpp:        return matrix_coo(c_rows, a.cols_gpu(), a.nrows(), a.ncols(), a.nnz());
deps/clbool/src/common/matrices_conversions.cpp:        controls.queue.enqueueCopyBuffer(a.cols_gpu(), c_cols, 0, 0, sizeof(matrix_dcsr::index_type) * a.nnz());
deps/clbool/src/common/matrices_conversions.cpp:        dscr_to_coo.run(controls, a.rpt_gpu(), a.rows_gpu(), c_rows);
deps/clbool/src/common/matrices_conversions.cpp:            create_rows_pointers(controls, rpt, rows, a.rows_gpu(), a.nnz(), nzr);
deps/clbool/src/common/matrices_conversions.cpp:        return matrix_dcsr(rpt, rows, a.cols_gpu(),
deps/clbool/src/common/matrices_conversions.cpp:    matrix_dcsr_cpu matrix_dcsr_from_gpu(Controls &controls, matrix_dcsr &m) {
deps/clbool/src/common/matrices_conversions.cpp:        controls.queue.enqueueReadBuffer(m.rpt_gpu(), CL_TRUE, 0,
deps/clbool/src/common/matrices_conversions.cpp:        controls.queue.enqueueReadBuffer(m.rows_gpu(), CL_TRUE, 0,
deps/clbool/src/common/matrices_conversions.cpp:        controls.queue.enqueueReadBuffer(m.cols_gpu(), CL_TRUE, 0,
deps/clbool/src/common/matrices_conversions.cpp:    matrix_coo_cpu matrix_coo_from_gpu(Controls &controls, matrix_coo &m) {
deps/clbool/src/common/matrices_conversions.cpp:        controls.queue.enqueueReadBuffer(m.rows_gpu(), CL_TRUE, 0,
deps/clbool/src/common/matrices_conversions.cpp:        controls.queue.enqueueReadBuffer(m.cols_gpu(), CL_TRUE, 0,
deps/clbool/src/common/matrices_conversions.cpp:            CLB_RUN(prepare_pos.run(controls, positions, m.rpt_gpu(), m.nrows()))
deps/clbool/src/common/matrices_conversions.cpp:            CLB_RUN(set_positions.run(controls, c_rpt, c_rows, m.rpt_gpu(), positions, m.nrows()));
deps/clbool/src/common/matrices_conversions.cpp:        return matrix_dcsr(c_rpt, c_rows, m.cols_gpu(), m.nrows(), m.ncols(), m.nnz(), c_nzr);
deps/clbool/src/common/matrices_conversions.cpp:            CLB_RUN(set_rsize.run(controls, m.rpt_gpu(), m.rows_gpu(), m.nzr(), c_rpt));
deps/clbool/src/common/matrices_conversions.cpp:        return matrix_csr(c_rpt, m.cols_gpu(), m.nrows(), m.ncols(), c_nnz);
deps/clbool/src/common/matrices_conversions.cpp:    matrix_csr_cpu matrix_csr_from_gpu(Controls &controls, const matrix_csr &m) {
deps/clbool/src/common/matrices_conversions.cpp:        CLB_READ_BUF(utils::read_buffer(controls, rpt, m.rpt_gpu()).wait())
deps/clbool/src/common/matrices_conversions.cpp:        CLB_READ_BUF(utils::read_buffer(controls, cols, m.cols_gpu()).wait())
deps/clbool/src/common/utils.hpp:    bool compare_matrices(Controls &controls, const matrix_dcsr &m_gpu, const matrix_dcsr_cpu &m_cpu);
deps/clbool/src/common/utils.hpp:    void print_gpu_buffer(Controls &controls, const cl::Buffer &buffer, uint32_t size);
deps/clbool/src/common/utils.hpp:    bool compare_buffers(Controls &controls, const cl::Buffer &buffer_gpu, const cpu_buffer &buffer_cpu, uint32_t size,
deps/clbool/src/common/utils.hpp:    bool compare_matrices(Controls &controls, const matrix_csr &m_gpu, const matrix_csr_cpu &m_cpu);
deps/clbool/src/common/utils.cpp:    bool compare_buffers(Controls &controls, const cl::Buffer &buffer_gpu, const cpu_buffer &buffer_cpu, uint32_t size,
deps/clbool/src/common/utils.cpp:                      << size << " on GPU vs " << buffer_cpu.size() << " on CPU " << std::endl;
deps/clbool/src/common/utils.cpp:        CLB_COPY_BUF(controls.queue.enqueueReadBuffer(buffer_gpu, CL_TRUE, 0, sizeof(uint32_t) * cpu_copy.size(),
deps/clbool/src/common/utils.cpp:                          << "{ i: (gpu[i], cpu[i]) }" << std::endl;
deps/clbool/src/common/utils.cpp:    bool compare_matrices(Controls &controls, const matrix_dcsr &m_gpu, const matrix_dcsr_cpu &m_cpu) {
deps/clbool/src/common/utils.cpp:        if (m_gpu.nnz() != m_cpu.cols().size()) {
deps/clbool/src/common/utils.cpp:            std::cerr << "diff nnz, gpu: " << m_gpu.nnz() << " vs cpu: " << m_cpu.cols().size() << std::endl;
deps/clbool/src/common/utils.cpp:        if (m_gpu.nnz() == 0) {
deps/clbool/src/common/utils.cpp:                compare_buffers(controls, m_gpu.rpt_gpu(), m_cpu.rpt(), m_gpu.nzr() + 1, "rpt") &&
deps/clbool/src/common/utils.cpp:                compare_buffers(controls, m_gpu.rows_gpu(), m_cpu.rows(), m_gpu.nzr(), "rows") &&
deps/clbool/src/common/utils.cpp:                compare_buffers(controls, m_gpu.cols_gpu(), m_cpu.cols(), m_gpu.nnz(), "cols");
deps/clbool/src/common/utils.cpp:    bool compare_matrices(Controls &controls, const matrix_csr &m_gpu, const matrix_csr_cpu &m_cpu) {
deps/clbool/src/common/utils.cpp:        if (m_gpu.nnz() != m_cpu.cols().size()) {
deps/clbool/src/common/utils.cpp:            std::cerr << "diff nnz, gpu: " << m_gpu.nnz() << " vs cpu: " << m_cpu.cols().size() << std::endl;
deps/clbool/src/common/utils.cpp:        if (m_gpu.nnz() == 0) {
deps/clbool/src/common/utils.cpp:                compare_buffers(controls, m_gpu.rpt_gpu(), m_cpu.rpt(), m_gpu.nrows() + 1, "rpt") &&
deps/clbool/src/common/utils.cpp:                compare_buffers(controls, m_gpu.cols_gpu(), m_cpu.cols(), m_gpu.nnz(), "cols");
deps/clbool/src/common/utils.cpp:                return "CL_DEVICE_TYPE_GPU";
deps/clbool/src/common/utils.cpp:    void print_gpu_buffer(Controls &controls, const cl::Buffer &buffer, uint32_t size) {
deps/clbool/src/common/env.cpp:            platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU, &devices);
deps/clbool/src/common/matrices_conversions.hpp:    matrix_dcsr_cpu matrix_dcsr_from_gpu(Controls &controls, matrix_dcsr &m);
deps/clbool/src/common/matrices_conversions.hpp:    matrix_coo_cpu matrix_coo_from_gpu(Controls &controls, matrix_coo &m);
deps/clbool/src/common/matrices_conversions.hpp:    matrix_csr_cpu matrix_csr_from_gpu(Controls &controls, const matrix_csr &m);
deps/clbool/src/common/cpu_matrix_operations.cpp://        matrix_coo m_coo_tr = matrix_coo(m_coo.ncols(), m_coo.nrows(), m_coo.nnz(), m_coo.cols_gpu(), m_coo.cols_gpu());
deps/clbool/src/common/cl_operations.cpp:        cl::Buffer a_gpu;
deps/clbool/src/common/cl_operations.cpp:        a_gpu = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a_size));
deps/clbool/src/common/cl_operations.cpp:        cl::Buffer b_gpu;
deps/clbool/src/common/cl_operations.cpp:        b_gpu = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * b_size));
deps/clbool/src/common/cl_operations.cpp:        cl::Buffer total_sum_gpu;
deps/clbool/src/common/cl_operations.cpp:        total_sum_gpu = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t)));
deps/clbool/src/common/cl_operations.cpp:        cl::Buffer *a_gpu_ptr = &a_gpu;
deps/clbool/src/common/cl_operations.cpp:        cl::Buffer *b_gpu_ptr = &b_gpu;
deps/clbool/src/common/cl_operations.cpp:            CLB_RUN(scan.run(controls, a_gpu, array, total_sum_gpu, array_size).wait());
deps/clbool/src/common/cl_operations.cpp:                scan.run(controls, *b_gpu_ptr, *a_gpu_ptr, total_sum_gpu, outer).wait();
deps/clbool/src/common/cl_operations.cpp:                update.run(controls, array, *a_gpu_ptr, array_size, leaf_size).wait();
deps/clbool/src/common/cl_operations.cpp:            std::swap(a_gpu_ptr, b_gpu_ptr);
deps/clbool/src/common/cl_operations.cpp:        CLB_READ_BUF(controls.queue.enqueueReadBuffer(total_sum_gpu, CL_TRUE, 0, sizeof(uint32_t), &total_sum));
deps/clbool/src/cl/hash/hash_large.cl:// how many rows_gpu (tables) can wo process by one threadblock
deps/clbool/src/cl/hash/hash_global.cl:// how many rows_gpu (tables) can wo process by one threadblock
deps/clbool/src/cl/hash/hash_global.cl:    // all data for large rows_gpu is already in a global memory,
deps/clbool/src/cl/hash/hash_tb.cl:// how many rows_gpu (tables) can wo process by one threadblock
deps/clbool/src/cl/hash/hash_pwarp.cl:// how many rows_gpu (tables) can wo process by one threadblock
deps/clbool/src/cl/clion_defines.cl:// taken from here https://github.com/GPGPUCourse/GPGPUTasks2020/tree/task01/src/cl
deps/clbool/src/cl/clion_defines.cl:// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/commonFunctions.html
deps/clbool/src/cl/clion_defines.cl:// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/barrier.html
deps/clbool/src/cl/clion_defines.cl:// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/vectorDataLoadandStoreFunctions.html
deps/clbool/src/cl/clion_defines.cl:// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/workItemFunctions.html
deps/clbool/src/cl/clion_defines.cl:// 64 for AMD, 32 for NVidia, 8 for intel GPUs, 1 for CPU
deps/clbool/src/cl/prefix_sum.cl:// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
deps/clbool/src/cl/for_test/half_sized_scan.cl:// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
deps/clbool/src/coo/coo_utils.hpp:    void print_matrix(Controls &controls, const matrix_dcsr& m_gpu, uint32_t index = -1);
deps/clbool/src/coo/coo_utils.cpp:    void print_matrix(Controls &controls, const matrix_dcsr &m_gpu, uint32_t index) {
deps/clbool/src/coo/coo_utils.cpp:        cpu_buffer rows_pointers(m_gpu.nzr() + 1);
deps/clbool/src/coo/coo_utils.cpp:        cpu_buffer rows_compressed(m_gpu.nzr());
deps/clbool/src/coo/coo_utils.cpp:        cpu_buffer cols_indices(m_gpu.nnz());
deps/clbool/src/coo/coo_utils.cpp:        controls.queue.enqueueReadBuffer(m_gpu.rpt_gpu(), CL_TRUE, 0,
deps/clbool/src/coo/coo_utils.cpp:        controls.queue.enqueueReadBuffer(m_gpu.rows_gpu(), CL_TRUE, 0,
deps/clbool/src/coo/coo_utils.cpp:        controls.queue.enqueueReadBuffer(m_gpu.cols_gpu(), CL_TRUE, 0,
deps/clbool/src/coo/coo_utils.cpp:     * штука нужна для сравнения с gpu, поэтому возвращать будем своего рода matrix_dcsr_cpu
deps/clbool/src/coo/coo_initialization.hpp:    void sort_arrays(Controls& controls, cl::Buffer &rows_gpu, cl::Buffer &cols_gpu, uint32_t n);
deps/clbool/src/coo/coo_matrix_addition.cpp:        coo_merge.run(controls, merged_rows, merged_cols, a.rows_gpu(), a.cols_gpu(),
deps/clbool/src/coo/coo_matrix_addition.cpp:                      b.rows_gpu(), b.cols_gpu(), a.nnz(), b.nnz()).wait()));
deps/clbool/src/coo/coo_matrix_addition.cpp:            CLB_CREATE_BUF(controls.queue.enqueueCopyBuffer(filled.rows_gpu(), rows, 0, 0, sizeof(uint32_t) * filled.nnz()));
deps/clbool/src/coo/coo_matrix_addition.cpp:            CLB_CREATE_BUF(controls.queue.enqueueCopyBuffer(filled.cols_gpu(), cols, 0, 0, sizeof(uint32_t) * filled.nnz()));
deps/clbool/src/coo/coo_matrix_addition.hpp:    //        cl::Buffer &rows_gpu,
deps/clbool/src/coo/coo_initialization.cpp:    void sort_arrays(Controls &controls, cl::Buffer &rows_gpu, cl::Buffer &cols_gpu, uint32_t n) {
deps/clbool/src/coo/coo_initialization.cpp:        CLB_RUN(bitonic_begin.run(controls, rows_gpu, cols_gpu, n));
deps/clbool/src/coo/coo_initialization.cpp:            CLB_RUN(bitonic_global_step.run(controls, rows_gpu, cols_gpu, segment_length, 1, n));
deps/clbool/src/coo/coo_initialization.cpp:                CLB_RUN(bitonic_global_step.run(controls, rows_gpu, cols_gpu, i, 0, n));
deps/clbool/src/coo/coo_initialization.cpp:            CLB_RUN(bitonic_end.run(controls, rows_gpu, cols_gpu, n));
deps/clbool/src/coo/coo_kronecker_product.cpp:                      matrix_a.rows_gpu(), matrix_a.cols_gpu(),
deps/clbool/src/coo/coo_kronecker_product.cpp:                      matrix_b.rows_gpu(), matrix_b.cols_gpu(),
deps/clbool/src/csr/csr_addition.cpp:            CLB_COPY_BUF(controls.queue.enqueueCopyBuffer(filled.rpt_gpu(), rpt, 0, 0, sizeof(uint32_t) * (filled.nrows() + 1)));
deps/clbool/src/csr/csr_addition.cpp:            CLB_COPY_BUF(controls.queue.enqueueCopyBuffer(filled.cols_gpu(), cols, 0, 0, sizeof(uint32_t) * filled.nnz()));
deps/clbool/src/csr/csr_addition.cpp:            CLB_RUN(fill_bins_size.run(controls, a.rpt_gpu(), b.rpt_gpu(), bins_offset, a.nrows()));
deps/clbool/src/csr/csr_addition.cpp:            CLB_RUN(build_permutation.run(controls, a.rpt_gpu(), b.rpt_gpu(), bins_offset, bins_size, permutation, a.nrows()));
deps/clbool/src/csr/csr_addition.cpp:                CLB_RUN(ev = add_symbolic.run(controls, a.rpt_gpu(), a.cols_gpu(), b.rpt_gpu(), b.cols_gpu(), c_rpt, a.nrows(),
deps/clbool/src/csr/csr_addition.cpp:                CLB_RUN(ev = add_numeric.run(controls, a.rpt_gpu(), a.cols_gpu(), b.rpt_gpu(), b.cols_gpu(), c_rpt, c_cols,
README.md:Cuda and OpenCL platforms. The primary goal of the library is implementation, testing and profiling algorithms for
README.md:- Cuda backend for computations
README.md:- OpenCL backend for computations
README.md:20.04, Intel Core i7-6700 3.40GHz CPU, DDR4 64Gb RAM, GeForce GTX 1070 GPU with 8Gb VRAM.
README.md:[link](https://github.com/YaccConstructor/articles/blob/master/2021/GRAPL/Sparse_Boolean_Algebra_on_GPGPU/Sparse_Boolean_Algebra_on_GPGPU.pdf)
README.md:│   │   ├── cuda - cuda backend
README.md:│   │   ├── opencl - opencl backend
README.md:│   ├── clbool - OpenCL based matrix operations for dcsr, csr and coo matrices
README.md:│   ├── cub - cuda utility, required for nsparse
README.md:  title = {spbla: sparse Boolean linear algebra for CPU, Cuda and OpenCL computations},
.gitmodules:	url = https://github.com/NVIDIA/cub.git
CMakeLists.txt:option(SPBLA_WITH_CUDA          "Build library with cuda backend (default)" ON)
CMakeLists.txt:option(SPBLA_WITH_OPENCL        "Build library with opencl backend (default)" ON)
CMakeLists.txt:option(SPBLA_WITH_CUB           "Build with bundled cub sources (enable for CUDA SDK version <= 10)" OFF)
CMakeLists.txt:# Configure cuda dependencies
CMakeLists.txt:if (SPBLA_WITH_CUDA)
CMakeLists.txt:        message(STATUS "Add cub as cuda utility")
CMakeLists.txt:if (SPBLA_WITH_OPENCL)
scripts/install_cuda_ubuntu.sh:# Original script from https://github.com/ptheywood/cuda-cmake-github-actions
scripts/install_cuda_ubuntu.sh:CUDA_PACKAGES_IN=(
scripts/install_cuda_ubuntu.sh:## Select CUDA version
scripts/install_cuda_ubuntu.sh:# Get the cuda version from the environment as $cuda.
scripts/install_cuda_ubuntu.sh:CUDA_VERSION_MAJOR_MINOR=${cuda}
scripts/install_cuda_ubuntu.sh:CUDA_MAJOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f1)
scripts/install_cuda_ubuntu.sh:CUDA_MINOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f2)
scripts/install_cuda_ubuntu.sh:CUDA_PATCH=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f3)
scripts/install_cuda_ubuntu.sh:echo "CUDA_MAJOR: ${CUDA_MAJOR}"
scripts/install_cuda_ubuntu.sh:echo "CUDA_MINOR: ${CUDA_MINOR}"
scripts/install_cuda_ubuntu.sh:echo "CUDA_PATCH: ${CUDA_PATCH}"
scripts/install_cuda_ubuntu.sh:# If we don't know the CUDA_MAJOR or MINOR, error.
scripts/install_cuda_ubuntu.sh:if [ -z "${CUDA_MAJOR}" ] ; then
scripts/install_cuda_ubuntu.sh:    echo "Error: Unknown CUDA Major version. Aborting."
scripts/install_cuda_ubuntu.sh:if [ -z "${CUDA_MINOR}" ] ; then
scripts/install_cuda_ubuntu.sh:    echo "Error: Unknown CUDA Minor version. Aborting."
scripts/install_cuda_ubuntu.sh:## Select CUDA packages to install
scripts/install_cuda_ubuntu.sh:CUDA_PACKAGES=""
scripts/install_cuda_ubuntu.sh:for package in "${CUDA_PACKAGES_IN[@]}"
scripts/install_cuda_ubuntu.sh:    # cuda-compiler-X-Y if CUDA >= 9.1 else cuda-nvcc-X-Y
scripts/install_cuda_ubuntu.sh:    if [[ "${package}" == "nvcc" ]] && version_ge "$CUDA_VERSION_MAJOR_MINOR" "9.1" ; then
scripts/install_cuda_ubuntu.sh:    elif [[ "${package}" == "compiler" ]] && version_lt "$CUDA_VERSION_MAJOR_MINOR" "9.1" ; then
scripts/install_cuda_ubuntu.sh:    CUDA_PACKAGES+=" cuda-${package}-${CUDA_MAJOR}-${CUDA_MINOR}"
scripts/install_cuda_ubuntu.sh:echo "CUDA_PACKAGES ${CUDA_PACKAGES}"
scripts/install_cuda_ubuntu.sh:PIN_FILENAME="cuda-ubuntu${UBUNTU_VERSION}.pin"
scripts/install_cuda_ubuntu.sh:PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/${PIN_FILENAME}"
scripts/install_cuda_ubuntu.sh:APT_KEY_URL="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/7fa2af80.pub"
scripts/install_cuda_ubuntu.sh:REPO_URL="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/"
scripts/install_cuda_ubuntu.sh:echo "Adding CUDA Repository"
scripts/install_cuda_ubuntu.sh:sudo mv ${PIN_FILENAME} /etc/apt/preferences.d/cuda-repository-pin-600
scripts/install_cuda_ubuntu.sh:echo "Installing CUDA packages ${CUDA_PACKAGES}"
scripts/install_cuda_ubuntu.sh:sudo apt-get -y install ${CUDA_PACKAGES}
scripts/install_cuda_ubuntu.sh:    echo "CUDA Installation Error."
scripts/install_cuda_ubuntu.sh:CUDA_PATH=/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}
scripts/install_cuda_ubuntu.sh:echo "CUDA_PATH=${CUDA_PATH}"
scripts/install_cuda_ubuntu.sh:export CUDA_PATH=${CUDA_PATH}
scripts/install_cuda_ubuntu.sh:export PATH="$CUDA_PATH/bin:$PATH"
scripts/install_cuda_ubuntu.sh:export LD_LIBRARY_PATH="$CUDA_PATH/lib:$LD_LIBRARY_PATH"
.gitignore:# opencl source binaries
spbla/sources/core/library.cpp:#ifdef SPBLA_WITH_CUDA
spbla/sources/core/library.cpp:#include <cuda/cuda_backend.hpp>
spbla/sources/core/library.cpp:#ifdef SPBLA_WITH_OPENCL
spbla/sources/core/library.cpp:#include <opencl/opencl_backend.hpp>
spbla/sources/core/library.cpp:        bool preferCuda = initHints & SPBLA_HINT_CUDA_BACKEND;
spbla/sources/core/library.cpp:        bool preferOpenCL = initHints & SPBLA_HINT_OPENCL_BACKEND;
spbla/sources/core/library.cpp:        bool prefer = preferCpu || preferCuda || preferOpenCL;
spbla/sources/core/library.cpp:#ifdef SPBLA_WITH_CUDA
spbla/sources/core/library.cpp:        // If user do not force something else or force cuda
spbla/sources/core/library.cpp:        if (!prefer || preferCuda) {
spbla/sources/core/library.cpp:            INIT_BACKEND(CudaBackend)
spbla/sources/core/library.cpp:            // Failed to setup cuda, release backend and go to try next
spbla/sources/core/library.cpp:                mLogger->logWarning("Failed to initialize Cuda backend");
spbla/sources/core/library.cpp:#ifdef SPBLA_WITH_OPENCL
spbla/sources/core/library.cpp:        // If user do not force something else or force opencl
spbla/sources/core/library.cpp:        if (!initialized && (!prefer || preferOpenCL)) {
spbla/sources/core/library.cpp:            INIT_BACKEND(OpenCLBackend)
spbla/sources/core/library.cpp:            // Failed to setup opencl, release backend and go to try cpu
spbla/sources/core/library.cpp:                mLogger->logWarning("Failed to initialize OpenCL backend");
spbla/sources/core/library.cpp:        caps.cudaSupported = false;
spbla/sources/core/library.cpp:        caps.openclSupported = false;
spbla/sources/core/library.cpp:        if (caps.cudaSupported || caps.openclSupported) {
spbla/sources/core/library.cpp:                   << " Cuda Type (" << caps.cudaSupported << "),"
spbla/sources/core/library.cpp:                   << " OpenCL Type (" << caps.openclSupported << "),"
spbla/sources/core/library.cpp:            stream << "CPU backend (GPU device is not present)";
spbla/sources/sequential/sq_backend.cpp:        caps.cudaSupported = false;
spbla/sources/opencl/opencl_matrix.cpp:#include <opencl/opencl_matrix.hpp>
spbla/sources/opencl/opencl_matrix.cpp:    OpenCLMatrix::OpenCLMatrix(clbool::Controls *controls, size_t nrows, size_t ncols)
spbla/sources/opencl/opencl_matrix.cpp:    void OpenCLMatrix::setElement(index i, index j) {
spbla/sources/opencl/opencl_matrix.cpp:    void OpenCLMatrix::clone(const MatrixBase &otherBase) {
spbla/sources/opencl/opencl_matrix.cpp:        auto other = dynamic_cast<const OpenCLMatrix*>(&otherBase);
spbla/sources/opencl/opencl_matrix.cpp:        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed matrix does not belong to OpenCLMatrix class");
spbla/sources/opencl/opencl_matrix.cpp:    index OpenCLMatrix::getNrows() const {
spbla/sources/opencl/opencl_matrix.cpp:    index OpenCLMatrix::getNcols() const {
spbla/sources/opencl/opencl_matrix.cpp:    index OpenCLMatrix::getNvals() const {
spbla/sources/opencl/opencl_matrix.cpp:    OpenCLMatrix::OpenCLMatrix(clbool::Controls *controls, MatrixImplType clbool_matrix)
spbla/sources/opencl/opencl_matrix_multiply.cpp:#include <opencl/opencl_matrix.hpp>
spbla/sources/opencl/opencl_matrix_multiply.cpp:    void OpenCLMatrix::multiply(const MatrixBase &aBase, const MatrixBase &bBase, bool accumulate, bool checkTime) {
spbla/sources/opencl/opencl_matrix_multiply.cpp:        auto a = dynamic_cast<const OpenCLMatrix*>(&aBase);
spbla/sources/opencl/opencl_matrix_multiply.cpp:        auto b = dynamic_cast<const OpenCLMatrix*>(&bBase);
spbla/sources/opencl/opencl_matrix_multiply.cpp:        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Passed matrix does not belong to OpenCLMatrix class")
spbla/sources/opencl/opencl_matrix_multiply.cpp:        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Passed matrix does not belong to OpenCLMatrix class")
spbla/sources/opencl/opencl_matrix_multiply.cpp:            this->eWiseAdd(*this, OpenCLMatrix(clboolState, multResDcsr), checkTime);
spbla/sources/opencl/opencl_matrix.hpp:#ifndef SPBLA_OPENCL_MATRIX_HPP
spbla/sources/opencl/opencl_matrix.hpp:#define SPBLA_OPENCL_MATRIX_HPP
spbla/sources/opencl/opencl_matrix.hpp:#include "opencl_backend.hpp"
spbla/sources/opencl/opencl_matrix.hpp:    class OpenCLMatrix: public MatrixBase {
spbla/sources/opencl/opencl_matrix.hpp:        OpenCLMatrix(clbool::Controls *controls, size_t nrows, size_t ncols);
spbla/sources/opencl/opencl_matrix.hpp:        ~OpenCLMatrix() override = default;
spbla/sources/opencl/opencl_matrix.hpp:        OpenCLMatrix(clbool::Controls *controls, MatrixImplType clbool_matrix);
spbla/sources/opencl/opencl_matrix.hpp:        friend spbla::OpenCLBackend;
spbla/sources/opencl/opencl_matrix.hpp:#endif //SPBLA_OPENCL_MATRIX_HPP
spbla/sources/opencl/opencl_matrix_ewiseadd.cpp:#include <opencl/opencl_matrix.hpp>
spbla/sources/opencl/opencl_matrix_ewiseadd.cpp:#define OPENCL_ADDITION_CSR
spbla/sources/opencl/opencl_matrix_ewiseadd.cpp:    void OpenCLMatrix::eWiseAdd(const MatrixBase &aBase, const MatrixBase &bBase, bool checkTime) {
spbla/sources/opencl/opencl_matrix_ewiseadd.cpp:        auto a = dynamic_cast<const OpenCLMatrix*>(&aBase);
spbla/sources/opencl/opencl_matrix_ewiseadd.cpp:        auto b = dynamic_cast<const OpenCLMatrix*>(&bBase);
spbla/sources/opencl/opencl_matrix_ewiseadd.cpp:        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Passed matrix does not belong to OpenCLMatrix class");
spbla/sources/opencl/opencl_matrix_ewiseadd.cpp:        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Passed matrix does not belong to OpenCLMatrix class");
spbla/sources/opencl/opencl_matrix_ewiseadd.cpp:#ifdef OPENCL_ADDITION_CSR
spbla/sources/opencl/opencl_backend.hpp:#ifndef SPBLA_OPENCL_BACKEND_HPP
spbla/sources/opencl/opencl_backend.hpp:#define SPBLA_OPENCL_BACKEND_HPP
spbla/sources/opencl/opencl_backend.hpp:    class OpenCLBackend: public BackendBase {
spbla/sources/opencl/opencl_backend.hpp:        ~OpenCLBackend() override = default;
spbla/sources/opencl/opencl_backend.hpp:        static const int NVIDIA_WARP = 32;
spbla/sources/opencl/opencl_backend.hpp:#endif //SPBLA_OPENCL_BACKEND_HPP
spbla/sources/opencl/opencl_matrix_extract_sub_matrix.cpp:#include <opencl/opencl_matrix.hpp>
spbla/sources/opencl/opencl_matrix_extract_sub_matrix.cpp:    void OpenCLMatrix::extractSubMatrix(const MatrixBase &otherBase, index i, index j, index nrows, index ncols, bool checkTime) {
spbla/sources/opencl/opencl_matrix_extract_sub_matrix.cpp:        auto other = dynamic_cast<const OpenCLMatrix*>(&otherBase);
spbla/sources/opencl/opencl_matrix_extract_sub_matrix.cpp:        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed matrix does not belong to OpenCLMatrix class");
spbla/sources/opencl/opencl_matrix_build.cpp:#include <opencl/opencl_matrix.hpp>
spbla/sources/opencl/opencl_matrix_build.cpp:    void OpenCLMatrix::build(const index *rows, const index *cols, size_t nvals, bool isSorted, bool noDuplicates) {
spbla/sources/opencl/opencl_matrix_reduce.cpp:#include <opencl/opencl_matrix.hpp>
spbla/sources/opencl/opencl_matrix_reduce.cpp:    void OpenCLMatrix::reduce(const MatrixBase &otherBase, bool checkTime) {
spbla/sources/opencl/opencl_matrix_reduce.cpp:        auto other = dynamic_cast<const OpenCLMatrix*>(&otherBase);
spbla/sources/opencl/opencl_matrix_reduce.cpp:                          "Passed matrix does not belong to OpenCLMatrix class")
spbla/sources/opencl/opencl_backend.cpp:#include <opencl/opencl_backend.hpp>
spbla/sources/opencl/opencl_backend.cpp:#include <opencl/opencl_matrix.hpp>
spbla/sources/opencl/opencl_backend.cpp:    std::shared_ptr<clbool::Controls> OpenCLBackend::controls = nullptr;
spbla/sources/opencl/opencl_backend.cpp:    void OpenCLBackend::initialize(hints initHints) {
spbla/sources/opencl/opencl_backend.cpp:    void OpenCLBackend::finalize() {
spbla/sources/opencl/opencl_backend.cpp:    bool OpenCLBackend::isInitialized() const {
spbla/sources/opencl/opencl_backend.cpp:    MatrixBase* OpenCLBackend::createMatrix(size_t nrows, size_t ncols) {
spbla/sources/opencl/opencl_backend.cpp:        return new OpenCLMatrix(controls.get(), nrows, ncols);
spbla/sources/opencl/opencl_backend.cpp:    void OpenCLBackend::releaseMatrix(MatrixBase *matrixBase) {
spbla/sources/opencl/opencl_backend.cpp:    std::pair<int, int> OpenCLBackend::getVersion() {
spbla/sources/opencl/opencl_backend.cpp:        // OpenCL 1.2 CUDA
spbla/sources/opencl/opencl_backend.cpp:        // 1.2 CUDA
spbla/sources/opencl/opencl_backend.cpp:    int OpenCLBackend::getWarp() {
spbla/sources/opencl/opencl_backend.cpp:        static std::regex nvidiaRegex("NVIDIA", std::regex_constants::icase);
spbla/sources/opencl/opencl_backend.cpp:        if (std::regex_search(vendor, nvidiaRegex)) return OpenCLBackend::NVIDIA_WARP;
spbla/sources/opencl/opencl_backend.cpp:        if (std::regex_search(vendor, amdRegex)) return OpenCLBackend::AMD_WARP;
spbla/sources/opencl/opencl_backend.cpp:    void OpenCLBackend::queryCapabilities(spbla_DeviceCaps &caps) {
spbla/sources/opencl/opencl_backend.cpp:            caps.cudaSupported = false;
spbla/sources/opencl/opencl_backend.cpp:            caps.openclSupported = true;
spbla/sources/opencl/opencl_backend.cpp:    void OpenCLBackend::queryAvailableDevices() {
spbla/sources/opencl/opencl_matrix_extract.cpp:#include <opencl/opencl_matrix.hpp>
spbla/sources/opencl/opencl_matrix_extract.cpp:    void OpenCLMatrix::extract(index *rows, index *cols, size_t &nvals) {
spbla/sources/opencl/opencl_matrix_extract.cpp:        clboolState->queue.enqueueReadBuffer(mCoo.rows_gpu(), false, 0, sizeof(index) * nvals,
spbla/sources/opencl/opencl_matrix_extract.cpp:        clboolState->queue.enqueueReadBuffer(mCoo.cols_gpu(), false, 0, sizeof(index) * nvals,
spbla/sources/opencl/opencl_matrix_transpose.cpp:#include <opencl/opencl_matrix.hpp>
spbla/sources/opencl/opencl_matrix_transpose.cpp:    void OpenCLMatrix::transpose(const MatrixBase &otherBase, bool checkTime) {
spbla/sources/opencl/opencl_matrix_transpose.cpp:        auto other = dynamic_cast<const OpenCLMatrix*>(&otherBase);
spbla/sources/opencl/opencl_matrix_transpose.cpp:                          "Passed matrix does not belong to OpenCLMatrix class")
spbla/sources/opencl/opencl_matrix_kronecker.cpp:#include <opencl/opencl_matrix.hpp>
spbla/sources/opencl/opencl_matrix_kronecker.cpp:    void OpenCLMatrix::kronecker(const MatrixBase &aBase, const MatrixBase &bBase, bool checkTime) {
spbla/sources/opencl/opencl_matrix_kronecker.cpp:        auto a = dynamic_cast<const OpenCLMatrix*>(&aBase);
spbla/sources/opencl/opencl_matrix_kronecker.cpp:        auto b = dynamic_cast<const OpenCLMatrix*>(&bBase);
spbla/sources/opencl/opencl_matrix_kronecker.cpp:        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Passed matrix does not belong to OpenCLMatrix class");
spbla/sources/opencl/opencl_matrix_kronecker.cpp:        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Passed matrix does not belong to OpenCLMatrix class");
spbla/sources/spbla_GetAbout.cpp:        "work with sparse matrices written on the NVIDIA CUDA, OpenCL and CPU platform. "
spbla/sources/cuda/cuda_matrix_extract_sub_matrix.cu:#include <cuda/cuda_matrix.hpp>
spbla/sources/cuda/cuda_matrix_extract_sub_matrix.cu:#include <cuda/kernels/spsubmatrix.cuh>
spbla/sources/cuda/cuda_matrix_extract_sub_matrix.cu:    void CudaMatrix::extractSubMatrix(const MatrixBase &otherBase, index i, index j, index nrows, index ncols,
spbla/sources/cuda/cuda_matrix_extract_sub_matrix.cu:        auto other = dynamic_cast<const CudaMatrix*>(&otherBase);
spbla/sources/cuda/cuda_matrix.cu:#include <cuda/cuda_matrix.hpp>
spbla/sources/cuda/cuda_matrix.cu:    CudaMatrix::CudaMatrix(size_t nrows, size_t ncols, CudaInstance &instance) : mInstance(instance) {
spbla/sources/cuda/cuda_matrix.cu:    void CudaMatrix::setElement(index i, index j) {
spbla/sources/cuda/cuda_matrix.cu:    void CudaMatrix::clone(const MatrixBase &otherBase) {
spbla/sources/cuda/cuda_matrix.cu:        auto other = dynamic_cast<const CudaMatrix*>(&otherBase);
spbla/sources/cuda/cuda_matrix.cu:    void CudaMatrix::resizeStorageToDim() const {
spbla/sources/cuda/cuda_matrix.cu:    void CudaMatrix::clearAndResizeStorageToDim() const {
spbla/sources/cuda/cuda_matrix.cu:    index CudaMatrix::getNrows() const {
spbla/sources/cuda/cuda_matrix.cu:    index CudaMatrix::getNcols() const {
spbla/sources/cuda/cuda_matrix.cu:    index CudaMatrix::getNvals() const {
spbla/sources/cuda/cuda_matrix.cu:    bool CudaMatrix::isStorageEmpty() const {
spbla/sources/cuda/cuda_matrix.cu:    bool CudaMatrix::isMatrixEmpty() const {
spbla/sources/cuda/cuda_matrix.cu:    void CudaMatrix::transferToDevice(const std::vector<index> &rowOffsets, const std::vector<index> &colIndices) const {
spbla/sources/cuda/cuda_matrix.cu:    void CudaMatrix::transferFromDevice(std::vector<index> &rowOffsets, std::vector<index> &colIndices) const {
spbla/sources/cuda/cuda_backend.cu:#include <cuda/cuda_backend.hpp>
spbla/sources/cuda/cuda_backend.cu:#include <cuda/cuda_matrix.hpp>
spbla/sources/cuda/cuda_backend.cu:    void CudaBackend::initialize(hints initHints) {
spbla/sources/cuda/cuda_backend.cu:        if (CudaInstance::isCudaDeviceSupported()) {
spbla/sources/cuda/cuda_backend.cu:            mInstance = new CudaInstance(initHints & SPBLA_HINT_GPU_MEM_MANAGED);
spbla/sources/cuda/cuda_backend.cu:    void CudaBackend::finalize() {
spbla/sources/cuda/cuda_backend.cu:    bool CudaBackend::isInitialized() const {
spbla/sources/cuda/cuda_backend.cu:    MatrixBase *CudaBackend::createMatrix(size_t nrows, size_t ncols) {
spbla/sources/cuda/cuda_backend.cu:        return new CudaMatrix(nrows, ncols, getInstance());
spbla/sources/cuda/cuda_backend.cu:    void CudaBackend::releaseMatrix(MatrixBase *matrixBase) {
spbla/sources/cuda/cuda_backend.cu:    void CudaBackend::queryCapabilities(spbla_DeviceCaps &caps) {
spbla/sources/cuda/cuda_backend.cu:        CudaInstance::queryDeviceCapabilities(caps);
spbla/sources/cuda/cuda_backend.cu:    CudaInstance & CudaBackend::getInstance() {
spbla/sources/cuda/cuda_matrix_transpose.cu:#include <cuda/cuda_matrix.hpp>
spbla/sources/cuda/cuda_matrix_transpose.cu:#include <cuda/kernels/sptranspose.cuh>
spbla/sources/cuda/cuda_matrix_transpose.cu:#include <cuda/kernels/sptranspose2.cuh>
spbla/sources/cuda/cuda_matrix_transpose.cu:    void CudaMatrix::transpose(const MatrixBase &otherBase, bool checkTime) {
spbla/sources/cuda/cuda_matrix_transpose.cu:        auto other = dynamic_cast<const CudaMatrix*>(&otherBase);
spbla/sources/cuda/cuda_instance.hpp:#ifndef SPBLA_CUDA_INSTANCE_HPP
spbla/sources/cuda/cuda_instance.hpp:#define SPBLA_CUDA_INSTANCE_HPP
spbla/sources/cuda/cuda_instance.hpp:    class CudaInstance {
spbla/sources/cuda/cuda_instance.hpp:        explicit CudaInstance(bool useManagedMemory);
spbla/sources/cuda/cuda_instance.hpp:        CudaInstance(const CudaInstance& other) = delete;
spbla/sources/cuda/cuda_instance.hpp:        CudaInstance(CudaInstance&& other) noexcept = delete;
spbla/sources/cuda/cuda_instance.hpp:        ~CudaInstance();
spbla/sources/cuda/cuda_instance.hpp:        void allocateOnGpu(void* &ptr, size_t s) const;
spbla/sources/cuda/cuda_instance.hpp:        void deallocateOnGpu(void* ptr) const;
spbla/sources/cuda/cuda_instance.hpp:        static bool isCudaDeviceSupported();
spbla/sources/cuda/cuda_instance.hpp:        static CudaInstance& getInstanceRef();
spbla/sources/cuda/cuda_instance.hpp:        static CudaInstance* getInstancePtr();
spbla/sources/cuda/cuda_instance.hpp:        static volatile CudaInstance* gInstance;
spbla/sources/cuda/cuda_instance.hpp:#endif //SPBLA_CUDA_INSTANCE_HPP
spbla/sources/cuda/cuda_matrix_extract.cu:#include <cuda/cuda_matrix.hpp>
spbla/sources/cuda/cuda_matrix_extract.cu:    void CudaMatrix::extract(index *rows, index *cols, size_t &nvals) {
spbla/sources/cuda/cuda_matrix_ewiseadd.cu:#include <cuda/cuda_matrix.hpp>
spbla/sources/cuda/cuda_matrix_ewiseadd.cu:#include <cuda/kernels/spmerge.cuh>
spbla/sources/cuda/cuda_matrix_ewiseadd.cu:    void CudaMatrix::eWiseAdd(const MatrixBase &aBase, const MatrixBase &bBase, bool checkTime) {
spbla/sources/cuda/cuda_matrix_ewiseadd.cu:        auto a = dynamic_cast<const CudaMatrix*>(&aBase);
spbla/sources/cuda/cuda_matrix_ewiseadd.cu:        auto b = dynamic_cast<const CudaMatrix*>(&bBase);
spbla/sources/cuda/cuda_backend.hpp:#ifndef SPBLA_CUDA_BACKEND_HPP
spbla/sources/cuda/cuda_backend.hpp:#define SPBLA_CUDA_BACKEND_HPP
spbla/sources/cuda/cuda_backend.hpp:#include <cuda/cuda_instance.hpp>
spbla/sources/cuda/cuda_backend.hpp:     * Main entry to cuda provided backend implementation.
spbla/sources/cuda/cuda_backend.hpp:    class CudaBackend final: public BackendBase {
spbla/sources/cuda/cuda_backend.hpp:        ~CudaBackend() override = default;
spbla/sources/cuda/cuda_backend.hpp:        CudaInstance& getInstance();
spbla/sources/cuda/cuda_backend.hpp:        CudaInstance* mInstance;
spbla/sources/cuda/cuda_backend.hpp:#endif //SPBLA_CUDA_BACKEND_HPP
spbla/sources/cuda/cuda_instance.cpp:#include <cuda/cuda_instance.hpp>
spbla/sources/cuda/cuda_instance.cpp:    volatile CudaInstance* CudaInstance::gInstance = nullptr;
spbla/sources/cuda/cuda_instance.cpp:    CudaInstance::CudaInstance(bool useManagedMemory) {
spbla/sources/cuda/cuda_instance.cpp:    void CudaInstance::allocate(void* &ptr, size_t size) const {
spbla/sources/cuda/cuda_instance.cpp:    void CudaInstance::deallocate(void* ptr) const {
spbla/sources/cuda/cuda_instance.cpp:    CudaInstance& CudaInstance::getInstanceRef() {
spbla/sources/cuda/cuda_instance.cpp:        return (CudaInstance&) *gInstance;
spbla/sources/cuda/cuda_instance.cpp:    CudaInstance* CudaInstance::getInstancePtr() {
spbla/sources/cuda/cuda_instance.cpp:        return (CudaInstance* ) gInstance;
spbla/sources/cuda/cuda_instance.cpp:    bool CudaInstance::isInstancePresent() {
spbla/sources/cuda/kernels/spkron.cuh:#include <cuda/kernels/bin_search.cuh>
spbla/sources/cuda/kernels/spsubmatrix.cuh:#include <cuda/kernels/bin_search.cuh>
spbla/sources/cuda/kernels/spreduce.cuh:#include <cuda/kernels/bin_search.cuh>
spbla/sources/cuda/kernels/sptranspose2.cuh:#include <cuda/kernels/bin_search.cuh>
spbla/sources/cuda/kernels/sptranspose.cuh:#include <cuda/kernels/bin_search.cuh>
spbla/sources/cuda/kernels/sptranspose.cuh:#include <cuda/kernels/slow_sort.cuh>
spbla/sources/cuda/cuda_matrix.hpp:#ifndef SPBLA_CUDA_MATRIX_HPP
spbla/sources/cuda/cuda_matrix.hpp:#define SPBLA_CUDA_MATRIX_HPP
spbla/sources/cuda/cuda_matrix.hpp:#include <cuda/details/host_allocator.hpp>
spbla/sources/cuda/cuda_matrix.hpp:#include <cuda/details/device_allocator.cuh>
spbla/sources/cuda/cuda_matrix.hpp:    class CudaMatrix: public MatrixBase {
spbla/sources/cuda/cuda_matrix.hpp:        explicit CudaMatrix(size_t nrows, size_t ncols, CudaInstance& instance);
spbla/sources/cuda/cuda_matrix.hpp:        ~CudaMatrix() override = default;
spbla/sources/cuda/cuda_matrix.hpp:        CudaInstance& mInstance;
spbla/sources/cuda/cuda_matrix.hpp:#endif //SPBLA_CUDA_MATRIX_HPP
spbla/sources/cuda/cuda_instance.cu:#include <cuda/cuda_instance.hpp>
spbla/sources/cuda/cuda_instance.cu:    CudaInstance::~CudaInstance() {
spbla/sources/cuda/cuda_instance.cu:    void CudaInstance::allocateOnGpu(void* &ptr, size_t size) const {
spbla/sources/cuda/cuda_instance.cu:        cudaError error;
spbla/sources/cuda/cuda_instance.cu:                error = cudaMalloc(&ptr, size);
spbla/sources/cuda/cuda_instance.cu:                error = cudaMallocManaged(&ptr, size);
spbla/sources/cuda/cuda_instance.cu:        if (error != cudaSuccess) {
spbla/sources/cuda/cuda_instance.cu:            std::string message = std::string{"Failed to allocate Gpu memory: "} + cudaGetErrorString(error);
spbla/sources/cuda/cuda_instance.cu:    void CudaInstance::deallocateOnGpu(void* ptr) const {
spbla/sources/cuda/cuda_instance.cu:        cudaError error = cudaFree(ptr);
spbla/sources/cuda/cuda_instance.cu:        if (error != cudaSuccess) {
spbla/sources/cuda/cuda_instance.cu:            std::string message = std::string{"Failed to deallocate Gpu memory: "} + cudaGetErrorString(error);
spbla/sources/cuda/cuda_instance.cu:    void CudaInstance::syncHostDevice() const {
spbla/sources/cuda/cuda_instance.cu:        cudaError error = cudaDeviceSynchronize();
spbla/sources/cuda/cuda_instance.cu:        if (error != cudaSuccess) {
spbla/sources/cuda/cuda_instance.cu:            std::string message = std::string{"Failed to synchronize host and device: "} + cudaGetErrorString(error);
spbla/sources/cuda/cuda_instance.cu:    bool CudaInstance::isCudaDeviceSupported() {
spbla/sources/cuda/cuda_instance.cu:        cudaError error = cudaGetDevice(&device);
spbla/sources/cuda/cuda_instance.cu:        return error == cudaSuccess;
spbla/sources/cuda/cuda_instance.cu:    void CudaInstance::queryDeviceCapabilities(spbla_DeviceCaps &deviceCaps) {
spbla/sources/cuda/cuda_instance.cu:        cudaError error = cudaGetDevice(&device);
spbla/sources/cuda/cuda_instance.cu:        if (error == cudaSuccess) {
spbla/sources/cuda/cuda_instance.cu:            cudaDeviceProp deviceProp{};
spbla/sources/cuda/cuda_instance.cu:            error = cudaGetDeviceProperties(&deviceProp, device);
spbla/sources/cuda/cuda_instance.cu:            if (error == cudaSuccess) {
spbla/sources/cuda/cuda_instance.cu:                deviceCaps.cudaSupported = true;
spbla/sources/cuda/details/host_allocator.hpp:#include <cuda/cuda_instance.hpp>
spbla/sources/cuda/details/host_allocator.hpp:            explicit HostAllocator(): mInstanceRef(CudaInstance::getInstanceRef()) {
spbla/sources/cuda/details/host_allocator.hpp:            CudaInstance& mInstanceRef;
spbla/sources/cuda/details/device_allocator.cuh:#include <cuda/cuda_instance.hpp>
spbla/sources/cuda/details/device_allocator.cuh:#include <thrust/system/cuda/memory.h>
spbla/sources/cuda/details/device_allocator.cuh:            __host__ explicit DeviceAllocator() : mInstanceRef(CudaInstance::getInstanceRef()) { }
spbla/sources/cuda/details/device_allocator.cuh:                mInstanceRef.allocateOnGpu(ptr, n * sizeof(T));
spbla/sources/cuda/details/device_allocator.cuh:                mInstanceRef.deallocateOnGpu(p.get());
spbla/sources/cuda/details/device_allocator.cuh:            CudaInstance &mInstanceRef;
spbla/sources/cuda/cuda_matrix_build.cu:#include <cuda/cuda_matrix.hpp>
spbla/sources/cuda/cuda_matrix_build.cu:    void CudaMatrix::build(const index *rows, const index *cols, size_t nvals, bool isSorted, bool noDuplicates) {
spbla/sources/cuda/cuda_matrix_kronecker.cu:#include <cuda/cuda_matrix.hpp>
spbla/sources/cuda/cuda_matrix_kronecker.cu:#include <cuda/kernels/spkron.cuh>
spbla/sources/cuda/cuda_matrix_kronecker.cu:    void CudaMatrix::kronecker(const MatrixBase &aBase, const MatrixBase &bBase, bool checkTime) {
spbla/sources/cuda/cuda_matrix_kronecker.cu:        auto a = dynamic_cast<const CudaMatrix*>(&aBase);
spbla/sources/cuda/cuda_matrix_kronecker.cu:        auto b = dynamic_cast<const CudaMatrix*>(&bBase);
spbla/sources/cuda/cuda_matrix_multiply.cu:#include <cuda/cuda_matrix.hpp>
spbla/sources/cuda/cuda_matrix_multiply.cu:    void CudaMatrix::multiply(const MatrixBase &aBase, const MatrixBase &bBase, bool accumulate, bool checkTime) {
spbla/sources/cuda/cuda_matrix_multiply.cu:        auto a = dynamic_cast<const CudaMatrix*>(&aBase);
spbla/sources/cuda/cuda_matrix_multiply.cu:        auto b = dynamic_cast<const CudaMatrix*>(&bBase);
spbla/sources/cuda/cuda_matrix_reduce.cu:#include <cuda/cuda_matrix.hpp>
spbla/sources/cuda/cuda_matrix_reduce.cu:#include <cuda/kernels/spreduce.cuh>
spbla/sources/cuda/cuda_matrix_reduce.cu:    void CudaMatrix::reduce(const MatrixBase &otherBase, bool checkTime) {
spbla/sources/cuda/cuda_matrix_reduce.cu:        auto other = dynamic_cast<const CudaMatrix*>(&otherBase);
spbla/tests/test_matrix_setup.cpp:#ifdef SPBLA_WITH_CUDA
spbla/tests/test_matrix_setup.cpp:TEST(spbla_Matrix, FillingSmallCuda) {
spbla/tests/test_matrix_setup.cpp:    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_setup.cpp:TEST(spbla_Matrix, FillingMediumCuda) {
spbla/tests/test_matrix_setup.cpp:    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_setup.cpp:TEST(spbla_Matrix, FillingLargeCuda) {
spbla/tests/test_matrix_setup.cpp:    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_setup.cpp:#ifdef SPBLA_WITH_OPENCL
spbla/tests/test_matrix_setup.cpp:TEST(spbla_Matrix, FillingSmallOpenCL) {
spbla/tests/test_matrix_setup.cpp:    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_setup.cpp:TEST(spbla_Matrix, FillingMediumOpenCL) {
spbla/tests/test_matrix_setup.cpp:    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_setup.cpp:TEST(spbla_Matrix, FillingLargeOpenCL) {
spbla/tests/test_matrix_setup.cpp:    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_element.cpp:#ifdef SPBLA_WITH_CUDA
spbla/tests/test_matrix_element.cpp:TEST(spbla_Matrix, SetElementSmallCuda) {
spbla/tests/test_matrix_element.cpp:    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_element.cpp:TEST(spbla_Matrix, SetElementMediumCuda) {
spbla/tests/test_matrix_element.cpp:    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_element.cpp:TEST(spbla_Matrix, SetElementLargeCuda) {
spbla/tests/test_matrix_element.cpp:    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_element.cpp:#ifdef SPBLA_WITH_OPENCL
spbla/tests/test_matrix_element.cpp:TEST(spbla_Matrix, SetElementSmallOpenCL) {
spbla/tests/test_matrix_element.cpp:    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_element.cpp:TEST(spbla_Matrix, SetElementMediumOpenCL) {
spbla/tests/test_matrix_element.cpp:    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_element.cpp:TEST(spbla_Matrix, SetElementLargeOpenCL) {
spbla/tests/test_matrix_element.cpp:    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_mxm.cpp:#ifdef SPBLA_WITH_CUDA
spbla/tests/test_matrix_mxm.cpp:TEST(spbla_Matrix, MultiplySmallCuda) {
spbla/tests/test_matrix_mxm.cpp:    testRun(m, t, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_mxm.cpp:TEST(spbla_Matrix, MultiplyMediumCuda) {
spbla/tests/test_matrix_mxm.cpp:    testRun(m, t, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_mxm.cpp:TEST(spbla_Matrix, MultiplyLargeCuda) {
spbla/tests/test_matrix_mxm.cpp:    testRun(m, t, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_mxm.cpp:#ifdef SPBLA_WITH_OPENCL
spbla/tests/test_matrix_mxm.cpp:TEST(spbla_Matrix, MultiplySmallOpenCL) {
spbla/tests/test_matrix_mxm.cpp:    testRun(m, t, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_mxm.cpp:TEST(spbla_Matrix, MultiplyMediumOpenCL) {
spbla/tests/test_matrix_mxm.cpp:    testRun(m, t, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_mxm.cpp:TEST(spbla_Matrix, MultiplyLargeOpenCL) {
spbla/tests/test_matrix_mxm.cpp:    testRun(m, t, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_library_api.cpp:        << "cuda supported: " << caps.cudaSupported << std::endl
spbla/tests/test_library_api.cpp:        << "opencl supported: " << caps.openclSupported << std::endl
spbla/tests/test_matrix_reduce.cpp:#ifdef SPBLA_WITH_CUDA
spbla/tests/test_matrix_reduce.cpp:TEST(spbla_Matrix, ReduceSmallCuda) {
spbla/tests/test_matrix_reduce.cpp:    testRun(m, n, step, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_reduce.cpp:TEST(spbla_Matrix, ReduceMediumCuda) {
spbla/tests/test_matrix_reduce.cpp:    testRun(m, n, step, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_reduce.cpp:TEST(spbla_Matrix, ReduceLargeCuda) {
spbla/tests/test_matrix_reduce.cpp:    testRun(m, n, step, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_reduce.cpp:#ifdef SPBLA_WITH_OPENCL
spbla/tests/test_matrix_reduce.cpp:TEST(spbla_Matrix, ReduceSmallOpenCL) {
spbla/tests/test_matrix_reduce.cpp:    testRun(m, n, step, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_reduce.cpp:TEST(spbla_Matrix, ReduceMediumOpenCL) {
spbla/tests/test_matrix_reduce.cpp:    testRun(m, n, step, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_reduce.cpp:TEST(spbla_Matrix, ReduceLargeOpenCL) {
spbla/tests/test_matrix_reduce.cpp:    testRun(m, n, step, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_kronecker.cpp:#ifdef SPBLA_WITH_CUDA
spbla/tests/test_matrix_kronecker.cpp:TEST(spbla_Matrix, KroneckerSmallCuda) {
spbla/tests/test_matrix_kronecker.cpp:    testRun(m, n, k, t, step, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_kronecker.cpp:TEST(spbla_Matrix, KroneckerMediumCuda) {
spbla/tests/test_matrix_kronecker.cpp:    testRun(m, n, k, t, step, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_kronecker.cpp:TEST(spbla_Matrix, KroneckerLargeCuda) {
spbla/tests/test_matrix_kronecker.cpp:    testRun(m, n, k, t, step, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_kronecker.cpp:#ifdef SPBLA_WITH_OPENCL
spbla/tests/test_matrix_kronecker.cpp:TEST(spbla_Matrix, KroneckerSmallOpenCL) {
spbla/tests/test_matrix_kronecker.cpp:    testRun(m, n, k, t, step, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_kronecker.cpp:TEST(spbla_Matrix, KroneckerMediumOpenCL) {
spbla/tests/test_matrix_kronecker.cpp:    testRun(m, n, k, t, step, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_kronecker.cpp:TEST(spbla_Matrix, KroneckerLargeOpenCL) {
spbla/tests/test_matrix_kronecker.cpp:    testRun(m, n, k, t, step, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_ewiseadd.cpp:#ifdef SPBLA_WITH_CUDA
spbla/tests/test_matrix_ewiseadd.cpp:TEST(spbla_Matrix, EWiseAddSmallCuda) {
spbla/tests/test_matrix_ewiseadd.cpp:    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_ewiseadd.cpp:TEST(spbla_Matrix, EWiseAddMediumCuda) {
spbla/tests/test_matrix_ewiseadd.cpp:    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_ewiseadd.cpp:TEST(spbla_Matrix, EWiseAddLargeCuda) {
spbla/tests/test_matrix_ewiseadd.cpp:    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_ewiseadd.cpp:#ifdef SPBLA_WITH_OPENCL
spbla/tests/test_matrix_ewiseadd.cpp:TEST(spbla_Matrix, EWiseAddSmallOpenCL) {
spbla/tests/test_matrix_ewiseadd.cpp:    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_ewiseadd.cpp:TEST(spbla_Matrix, EWiseAddMediumOpenCL) {
spbla/tests/test_matrix_ewiseadd.cpp:    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_ewiseadd.cpp:TEST(spbla_Matrix, EWiseAddLargeOpenCL) {
spbla/tests/test_matrix_ewiseadd.cpp:    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_extract_sub_matrix.cpp:#ifdef SPBLA_WITH_CUDA
spbla/tests/test_matrix_extract_sub_matrix.cpp:TEST(spbla_Matrix, SubMatrixExtractSmallCuda) {
spbla/tests/test_matrix_extract_sub_matrix.cpp:    testRun(m, n, step, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_extract_sub_matrix.cpp:TEST(spbla_Matrix, SubMatrixExtractMediumCuda) {
spbla/tests/test_matrix_extract_sub_matrix.cpp:    testRun(m, n, step, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_extract_sub_matrix.cpp:TEST(spbla_Matrix, SubMatrixExtractLargeCuda) {
spbla/tests/test_matrix_extract_sub_matrix.cpp:    testRun(m, n, step, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_extract_sub_matrix.cpp:#ifdef SPBLA_WITH_OPENCL
spbla/tests/test_matrix_extract_sub_matrix.cpp:TEST(spbla_Matrix, SubMatrixExtractSmallOpenCL) {
spbla/tests/test_matrix_extract_sub_matrix.cpp:    testRun(m, n, step, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_extract_sub_matrix.cpp:TEST(spbla_Matrix, SubMatrixExtractMediumOpenCL) {
spbla/tests/test_matrix_extract_sub_matrix.cpp:    testRun(m, n, step, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_extract_sub_matrix.cpp:TEST(spbla_Matrix, SubMatrixExtractLargeOpenCL) {
spbla/tests/test_matrix_extract_sub_matrix.cpp:    testRun(m, n, step, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_transpose.cpp:#ifdef SPBLA_WITH_CUDA
spbla/tests/test_matrix_transpose.cpp:TEST(spbla_Matrix, TransposeSmallCuda) {
spbla/tests/test_matrix_transpose.cpp:    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_transpose.cpp:TEST(spbla_Matrix, TransposeMediumCuda) {
spbla/tests/test_matrix_transpose.cpp:    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_transpose.cpp:TEST(spbla_Matrix, TransposeLargeCuda) {
spbla/tests/test_matrix_transpose.cpp:    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
spbla/tests/test_matrix_transpose.cpp:#ifdef SPBLA_WITH_OPENCL
spbla/tests/test_matrix_transpose.cpp:TEST(spbla_Matrix, TransposeSmallOpenCL) {
spbla/tests/test_matrix_transpose.cpp:    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_transpose.cpp:TEST(spbla_Matrix, TransposeMediumOpenCL) {
spbla/tests/test_matrix_transpose.cpp:    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/tests/test_matrix_transpose.cpp:TEST(spbla_Matrix, TransposeLargeOpenCL) {
spbla/tests/test_matrix_transpose.cpp:    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
spbla/include/spbla/spbla.h:    /** No cuda compatible device in the system */
spbla/include/spbla/spbla.h:    /** Failed to allocate memory on cpy or gpu side */
spbla/include/spbla/spbla.h:    /** Force Cuda based backend usage */
spbla/include/spbla/spbla.h:    SPBLA_HINT_CUDA_BACKEND = 2,
spbla/include/spbla/spbla.h:    /** Force OpenCL based backend usage */
spbla/include/spbla/spbla.h:    SPBLA_HINT_OPENCL_BACKEND = 4,
spbla/include/spbla/spbla.h:    /** Use managed gpu memory type instead of default (device) memory */
spbla/include/spbla/spbla.h:    SPBLA_HINT_GPU_MEM_MANAGED = 8,
spbla/include/spbla/spbla.h:    bool cudaSupported;
spbla/include/spbla/spbla.h:    bool openclSupported;
spbla/include/spbla/spbla.h: * Query device capabilities/properties if cuda/opencl compatible device is present.
spbla/include/spbla/spbla.h: * @note This function returns no actual info if cuda/opencl backend is not presented.
spbla/include/spbla/spbla.h: * @return Error if cuda/opencl device not present or if failed to query capabilities
spbla/CMakeLists.txt:if (SPBLA_WITH_CUDA)
spbla/CMakeLists.txt:    # If Cuda backend is compiled, we must tell cmake, that we will use Cuda
spbla/CMakeLists.txt:    enable_language(CUDA)
spbla/CMakeLists.txt:    if (SPBLA_WITH_CUDA)
spbla/CMakeLists.txt:    message(STATUS "Add CUDA backend for GPGPU computations")
spbla/CMakeLists.txt:if (SPBLA_WITH_OPENCL)
spbla/CMakeLists.txt:    message(STATUS "Add OpenCL backend for GPGPU computations")
spbla/CMakeLists.txt:set(SPBLA_CUDA_SOURCES)
spbla/CMakeLists.txt:set(SPBLA_OPENCL_SOURCES)
spbla/CMakeLists.txt:# Cuda backend sources
spbla/CMakeLists.txt:if (SPBLA_WITH_CUDA)
spbla/CMakeLists.txt:    set(SPBLA_CUDA_SOURCES
spbla/CMakeLists.txt:        sources/cuda/cuda_backend.hpp
spbla/CMakeLists.txt:        sources/cuda/cuda_backend.cu
spbla/CMakeLists.txt:        sources/cuda/cuda_instance.hpp
spbla/CMakeLists.txt:        sources/cuda/cuda_instance.cu
spbla/CMakeLists.txt:        sources/cuda/cuda_instance.cpp
spbla/CMakeLists.txt:        sources/cuda/cuda_matrix.hpp
spbla/CMakeLists.txt:        sources/cuda/cuda_matrix.cu
spbla/CMakeLists.txt:        sources/cuda/cuda_matrix_build.cu
spbla/CMakeLists.txt:        sources/cuda/cuda_matrix_extract.cu
spbla/CMakeLists.txt:        sources/cuda/cuda_matrix_ewiseadd.cu
spbla/CMakeLists.txt:        sources/cuda/cuda_matrix_kronecker.cu
spbla/CMakeLists.txt:        sources/cuda/cuda_matrix_multiply.cu
spbla/CMakeLists.txt:        sources/cuda/cuda_matrix_transpose.cu
spbla/CMakeLists.txt:        sources/cuda/cuda_matrix_reduce.cu
spbla/CMakeLists.txt:        sources/cuda/cuda_matrix_extract_sub_matrix.cu
spbla/CMakeLists.txt:        sources/cuda/kernels/slow_sort.cuh
spbla/CMakeLists.txt:        sources/cuda/kernels/bin_search.cuh
spbla/CMakeLists.txt:        sources/cuda/kernels/sptranspose.cuh
spbla/CMakeLists.txt:        sources/cuda/kernels/sptranspose2.cuh
spbla/CMakeLists.txt:        sources/cuda/kernels/spkron.cuh
spbla/CMakeLists.txt:        sources/cuda/kernels/spmerge.cuh
spbla/CMakeLists.txt:        sources/cuda/kernels/spreduce.cuh
spbla/CMakeLists.txt:        sources/cuda/kernels/spsubmatrix.cuh)
spbla/CMakeLists.txt:# OpenCL backend related stuff
spbla/CMakeLists.txt:if (SPBLA_WITH_OPENCL)
spbla/CMakeLists.txt:    set(SPBLA_OPENCL_SOURCES
spbla/CMakeLists.txt:        sources/opencl/opencl_backend.cpp
spbla/CMakeLists.txt:        sources/opencl/opencl_backend.hpp
spbla/CMakeLists.txt:        sources/opencl/opencl_matrix.cpp
spbla/CMakeLists.txt:        sources/opencl/opencl_matrix.hpp
spbla/CMakeLists.txt:        sources/opencl/opencl_matrix_build.cpp
spbla/CMakeLists.txt:        sources/opencl/opencl_matrix_extract.cpp
spbla/CMakeLists.txt:        sources/opencl/opencl_matrix_ewiseadd.cpp
spbla/CMakeLists.txt:        sources/opencl/opencl_matrix_kronecker.cpp
spbla/CMakeLists.txt:        sources/opencl/opencl_matrix_multiply.cpp
spbla/CMakeLists.txt:        sources/opencl/opencl_matrix_transpose.cpp
spbla/CMakeLists.txt:        sources/opencl/opencl_matrix_reduce.cpp
spbla/CMakeLists.txt:        sources/opencl/opencl_matrix_extract_sub_matrix.cpp)
spbla/CMakeLists.txt:    ${SPBLA_OPENCL_SOURCES}
spbla/CMakeLists.txt:    ${SPBLA_CUDA_SOURCES}
spbla/CMakeLists.txt:# Cuda specifics
spbla/CMakeLists.txt:if (SPBLA_WITH_CUDA)
spbla/CMakeLists.txt:    set_target_properties(spbla PROPERTIES CUDA_STANDARD 14)
spbla/CMakeLists.txt:    set_target_properties(spbla PROPERTIES CUDA_STANDARD_REQUIRED ON)
spbla/CMakeLists.txt:    set_target_properties(spbla PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
spbla/CMakeLists.txt:    set_target_properties(spbla PROPERTIES CUDA_ARCHITECTURES "60;61;62;70;72;75")
spbla/CMakeLists.txt:    target_compile_options(spbla PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -use_fast_math -Xptxas -O2>)
spbla/CMakeLists.txt:    target_compile_definitions(spbla PUBLIC SPBLA_WITH_CUDA)
spbla/CMakeLists.txt:# OpenCL specifics
spbla/CMakeLists.txt:if (SPBLA_WITH_OPENCL)
spbla/CMakeLists.txt:    target_compile_definitions(spbla PUBLIC SPBLA_WITH_OPENCL)

```
