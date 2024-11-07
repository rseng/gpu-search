# https://github.com/MeasureTransport/MParT

```console
docs/source/getting_started.rst:        Currently, only the Python bindings support GPU-acceleration via CUDA backend.  MParT relies on templates in c++ to dictate which Kokkos execution space is used, but in python we simply prepend :code:`d` to classes and functions leveraging device execution (e.g., GPU).  For example, the c++ :code:`CreateComponent<Kokkos::HostSpace>` function corresponds to the :code:`mt.CreateComponent` while the :code:`CreateComponent<mpart::DeviceSpace>` function, which will return a Monotone component that leverages the Cuda backend, corresponds to the python function :code:`dCreateComponent`.
docs/source/api/templateconcepts.rst:Many of the lower-level classes in MParT are templated to allow for generic implementations.  Using templates instead of other programming techniques, like virtual inheritance, makes it simpler to copy these classes to/from a GPU and can sometimes even result in more efficient CPU code.    For example, the :code:`MonotoneComponent`` class, which uses a generic function :math:`f(x)` to define a monotone function :math:`T_d(x)`, is templated on the type of the :math:`f` function.   It is therefore possible to construct a monotone function from any class defining :math:`f(x)`, as long as the class contains the functions (i.e., the interface) expected by :code:`MonotoneComponent`.  In the language of generic programming, the necessary interface is a specific `concept <https://en.wikipedia.org/wiki/Concept_(generic_programming)>`_.
docs/source/api/utilities/linearalgebra.rst:MParT provides a class similar to Eigen's :code:`PartialPivLU` class for LU factorizations.  The :code:`Kokkos::HostSpace` version of MParT's implementation is actually just a thin wrapper around the Eigen implementation.   The :code:`DeviceSpace` version uses the `cuSolver <https://docs.nvidia.com/cuda/cusolver/index.html>`_ library.  
docs/source/installation.rst:    Matlab, and CUDA are currently only supported when compiling from source.
docs/source/installation.rst:MParT is built on Kokkos, which provides a single interface to many different multithreading capabilities like threads, OpenMP, CUDA, and OpenCL.   A list of available backends can be found on the `Kokkos wiki <https://github.com/kokkos/kokkos/blob/master/BUILD.md#device-backends>`_.   The :code:`Kokkos_ENABLE_THREADS` option in the CMake configuration above can be changed to reflect different choices in device backends.   The OSX-provided clang compiler does not support OpenMP, so :code:`THREADS` is a natural choice for CPU-based multithreading on OSX.   However, you may find that OpenMP has slightly better performance with other compilers and operating systems.
docs/source/installation.rst:Compiling with CUDA Support
docs/source/installation.rst:To support a GPU at the moment, you need a few special requirements. Due to the way that Kokkos handles GPU code, MParT must be compiled using a special wrapper around NVCC that Kokkos provides.  Because of this, MParT cannot use an internal build of Kokkos and Kokkos must therefore be compiled (or otherwise installed) manually.
docs/source/installation.rst:The following cmake command can be used to compile Kokkos with the CUDA backend enabled and with all options required by MParT.  Kokkos source code can be obtained from the `kokkos/kokkos <https://github.com/kokkos/kokkos>`_ repository on Github.
docs/source/installation.rst:        -DKokkos_ENABLE_CUDA=ON                           \
docs/source/installation.rst:        -DKokkos_ENABLE_CUDA_LAMBDA=ON                    \
docs/source/installation.rst:    If Kokkos may not be able to find your GPU information automatically, consider including :code:`-DKokkos_ARCH_<ARCH><VERSION>=ON` where :code:`<ARCH>` and :code:`<VERSION>` are determined by `the Kokkos documentation <https://kokkos.github.io/kokkos-core-wiki/keywords.html?highlight=volta70#architecture-keywords>`_. If Kokkos cannot find CUDA, or you wish to use a particular version, use :code:`-DKokkos_CUDA_DIR=/your/cuda/path`.
docs/source/installation.rst:MParT uses the CUBLAS and CUSOLVER components of the `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ for GPU-accelerated linear algebra.
docs/source/installation.rst:NVIDIA's `Cuda installation guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_ provides detailed instructions on how to install CUDA.   For Debian-based x86_64 systems, we have been able to successfully install cuda, cublas, and cusparse for CUDA 11.4 using the command below.  Notice the installation of :code:`*-dev` packages, which are required to obtain the necessary header files.  Similar commands may be useful on other systems.
docs/source/installation.rst:    export CUDA_VERSION=11.4
docs/source/installation.rst:    export CUDA_COMPAT_VERSION=470.129.06-1
docs/source/installation.rst:    export CUDA_CUDART_VERSION=11.4.148-1
docs/source/installation.rst:    curl -sL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub" | apt-key add -
docs/source/installation.rst:    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda.list
docs/source/installation.rst:        cuda-compat-${CUDA_VERSION/./-}=${CUDA_COMPAT_VERSION} \
docs/source/installation.rst:        cuda-cudart-${CUDA_VERSION/./-}=${CUDA_CUDART_VERSION} \
docs/source/installation.rst:        libcublas-${CUDA_VERSION/./-} \
docs/source/installation.rst:        libcublas-dev-${CUDA_VERSION/./-} \
docs/source/installation.rst:        libcusolver-${CUDA_VERSION/./-} \
docs/source/installation.rst:        libcusolver-dev-${CUDA_VERSION/./-}
docs/source/installation.rst:   If you're using a Power8 or Power9 architecture, Eigen may give you trouble when trying to incorporate vectorization using Altivec, specifically when compiling for GPU. In this case, go into :code:`CMakeFiles.txt` and add :code:`add_compile_definition(EIGEN_DONT_VECTORIZE)`.
joss/paper.md:Several existing software packages have the ability to parameterize monotone functions, including TensorFlow Probability [@dillon2017tensorflow], TransportMaps [@transportmaps], ATM [@atm], and MUQ [@parno2021muq].  TensorFlow probability has a bijection class that allows deep neural network-based functions, such as normalizing flows [@papamakarios2021normalizing] to be easily defined and trained while also leveraging GPU computing resources if available but is focused on deep neural network parameterizations best suited for high dimensional problems.   The TransportMaps, ATM, and MUQ packages use an alternative parameterization based on rectified polynomial expansions that is more compact and easier to train on low to moderate dimensional problems.  At the core of these packages are scalar-valued functions $T_d : \mathbb{R}^d \rightarrow \mathbb{R}$ of the form 
joss/paper.md:`MParT` aims to provide a performance portable shared-memory implementation of parameterizations built on \autoref{eq:rectified}.  `MParT` uses Kokkos [@edwards2014kokkos] to leverage multithreading on either CPUs or GPUs with a common code base.  `MParT` provides an efficient low-level library that can then be used to accelerate higher level packages like TransportMaps, ATM, and MUQ that cannot currently leverage GPU resources.  Bindings to Python, Julia, and Matlab are also provided to enable a wide variety of users to leverage the fast C++ core from the language of their choice.
joss/paper.md:The results show similar performance across languages (each using OpenMP backend with 8 threads) and nearly identical performance between the Threads and OpenMP backends.   For the evaluation of $10^6$ samples, the OpenMP backend with 16 threads is approximately $14\times$ faster than the serial backend.  The CUDA backend is approximately $82\times$ faster than the serial backend, or $6\times$ faster than the OpenMP backend.   Tests were performed in a Kubernetes container using 8 cores of a Intel(R) Xeon(R) Gold 6248 CPU and a Tesla V100-SXM2 GPU with CUDA version 11.2.
bindings/python/src/TriangularMap.cpp:#if defined(MPART_ENABLE_GPU)
bindings/python/src/TriangularMap.cpp:#endif // MPART_ENABLE_GPU
bindings/python/src/Wrapper.cpp:#include <MParT/Utilities/GPUtils.h>
bindings/python/src/Wrapper.cpp:#if defined(MPART_ENABLE_GPU)
bindings/python/src/MultiIndex.cpp:#if defined(MPART_ENABLE_GPU)
bindings/python/src/MultiIndex.cpp:#endif // MPART_ENABLE_GPU
bindings/python/src/TrainMap.cpp:#if defined(MPART_ENABLE_GPU)
bindings/python/src/TrainMap.cpp:#endif // MPART_ENABLE_GPU
bindings/python/src/ParameterizedFunctionBase.cpp:#if defined(MPART_ENABLE_GPU)
bindings/python/src/ParameterizedFunctionBase.cpp:#endif // MPART_ENABLE_GPU
bindings/python/src/ComposedMap.cpp:#if defined(MPART_ENABLE_GPU)
bindings/python/src/ComposedMap.cpp:#endif // MPART_ENABLE_GPU
bindings/python/src/TrainMapAdaptive.cpp:#if defined(MPART_ENABLE_GPU)
bindings/python/src/TrainMapAdaptive.cpp:#endif // MPART_ENABLE_GPU
bindings/python/src/AffineMap.cpp:#include "MParT/Utilities/GPUtils.h"
bindings/python/src/AffineMap.cpp:#if defined(MPART_ENABLE_GPU)
bindings/python/src/AffineMap.cpp:#endif // MPART_ENABLE_GPU
bindings/python/src/MapFactory.cpp:#if defined(MPART_ENABLE_GPU)
bindings/python/src/MapFactory.cpp:#endif // MPART_ENABLE_GPU
bindings/python/src/MapObjective.cpp:#if defined(MPART_ENABLE_GPU)
bindings/python/src/MapObjective.cpp:#endif // MPART_ENABLE_GPU
bindings/python/src/SummarizedMap.cpp:#if defined(MPART_ENABLE_GPU)
bindings/python/src/SummarizedMap.cpp:#endif // MPART_ENABLE_GPU
bindings/python/src/IdentityMap.cpp:#if defined(MPART_ENABLE_GPU)
bindings/python/src/IdentityMap.cpp:#endif // MPART_ENABLE_GPU
bindings/python/src/ConditionalMapBase.cpp:#if defined(MPART_ENABLE_GPU)
bindings/python/src/ConditionalMapBase.cpp:#endif // MPART_ENABLE_GPU
bindings/julia/include/CommonJuliaUtilities.h:#if defined(MPART_ENABLE_GPU)
tests/MultiIndices/Test_MultiIndexSet.cpp:#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
tests/Test_MultivariateExpansionWorker.cpp:#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
tests/Test_AffineMap.cpp:#if defined(MPART_ENABLE_GPU)
tests/Test_PositiveBijectors.cpp:#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
tests/Distributions/Test_Distributions_Common.h:// TODO: Test on GPU
tests/Distributions/Test_TransportDensity.cpp:        auto logPullbackDensitySample = pullback.LogDensity(samples);
tests/Distributions/Test_TransportDensity.cpp:        auto logPushforwardDensitySample = pushforward.LogDensity(samples);
tests/Distributions/Test_TransportDensity.cpp:        auto gradLogPullbackDensitySample = pullback.LogDensityInputGrad(samples);
tests/Distributions/Test_TransportDensity.cpp:        bool gradLogPushforwardExists = true;
tests/Distributions/Test_TransportDensity.cpp:            gradLogPushforwardExists = false;
tests/Distributions/Test_TransportDensity.cpp:        REQUIRE(gradLogPushforwardExists == false);
tests/Distributions/Test_TransportDensity.cpp:                CHECK(gradLogPullbackDensitySample(j, i) == Approx(-pullbackEvalSample(j, i)*diag_el).margin(1e-6));
tests/Distributions/Test_TransportDensity.cpp:            CHECK(logPullbackDensitySample(i) == Approx(analytical_pullback).margin(1e-6));
tests/Distributions/Test_TransportDensity.cpp:            CHECK(logPushforwardDensitySample(i) == Approx(analytical_pushforward).margin(1e-6));
tests/Test_OrthogonalPolynomials.cpp:#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
tests/Test_LinearAlgebra.cpp:#if defined(MPART_ENABLE_GPU)
tests/Test_Quadrature.cpp:#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
tests/Test_ArrayConversions.cpp:#if defined(MPART_ENABLE_GPU)
tests/Test_MonotoneComponent.cpp:#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
tests/Test_AffineFunction.cpp:#if defined(MPART_ENABLE_GPU)
MParT/Initialization.h:#if defined(MPART_ENABLE_GPU)
MParT/Initialization.h:#include <cuda_runtime.h>
MParT/Initialization.h:        #if defined(MPART_ENABLE_GPU)
MParT/Initialization.h:        #if defined(MPART_ENABLE_GPU)
MParT/Initialization.h:#if defined(MPART_ENABLE_GPU)
MParT/Utilities/ArrayConversions.h:#include "GPUtils.h"
MParT/Utilities/ArrayConversions.h:    Kokkos::View<double*,Kokkos::CudaSpace> deviceView("Some stuff on the device", N);
MParT/Utilities/ArrayConversions.h:    Kokkos::View<double**,Kokkos::CudaSpace> deviceView("Some stuff on the device", N1, N2);
MParT/Utilities/ArrayConversions.h:    Kokkos::View<double**,Kokkos::CudaSpace> deviceView("Some stuff on the device", N1, N2);
MParT/Utilities/ArrayConversions.h:    @tparam DeviceMemoryType The memory space (e.g., Kokkos::CudaSpace) or the device
MParT/Utilities/ArrayConversions.h:#if defined(MPART_ENABLE_GPU)
MParT/Utilities/ArrayConversions.h:    #if defined(MPART_ENABLE_GPU)
MParT/Utilities/LinearAlgebra.h:#include "MParT/Utilities/GPUtils.h"
MParT/Utilities/LinearAlgebra.h:#if defined(MPART_ENABLE_GPU)
MParT/Utilities/LinearAlgebra.h:#include <cuda_runtime.h>
MParT/Utilities/LinearAlgebra.h:#if defined(MPART_ENABLE_GPU)
MParT/Utilities/LinearAlgebra.h:#if defined(MPART_ENABLE_GPU)
MParT/Utilities/LinearAlgebra.h:#endif // #ifndef MPART_CUDALINEARALGEBRA_H
MParT/Utilities/Miscellaneous.h:        and assertions in GPU code where exceptions aren't alllowed.
MParT/Utilities/KokkosSpaceMappings.h:    /** Used to convert Kokkos memory space type (e.g., Kokkos::CudaSpace) to an execution space that can access that memory.
MParT/Utilities/KokkosSpaceMappings.h:    #if defined(MPART_ENABLE_GPU)
MParT/Utilities/KokkosSpaceMappings.h:    template<> struct MemoryToExecution<Kokkos::CudaSpace>{using Space = Kokkos::Cuda;};
MParT/Utilities/GPUtils.h:#if defined(MPART_ENABLE_GPU)
MParT/Utilities/GPUtils.h:// Only enable DeviceSpace if the DefaultExecutionSpace is a GPU space.
MParT/ParameterizedFunctionBase.h:#include "MParT/Utilities/GPUtils.h"
MParT/ParameterizedFunctionBase.h:       #if defined(MPART_ENABLE_GPU)
MParT/ParameterizedFunctionBase.h:        #if defined(MPART_ENABLE_GPU)
MParT/AffineMap.h:#include "MParT/Utilities/GPUtils.h"
MParT/ComposedMap.h:    #if defined(MPART_ENABLE_GPU)
MParT/SummarizedMap.h:    #if defined(MPART_ENABLE_GPU)
MParT/Distributions/DensityBase.h:#include "MParT/Utilities/GPUtils.h"
README.md:A CPU/GPU performance-portable library for parameterizing and constructing monotone functions in the context of measure transport and regression.
CMakeLists.txt:check_cxx_compiler_flag("-Wno-deprecated-gpu-targets" COMPILER_IS_NVCC1)
CMakeLists.txt:    add_definitions(-DMPART_ENABLE_GPU)
CMakeLists.txt:    message(STATUS "GPU support detected")
CMakeLists.txt:    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-gpu-targets --expt-relaxed-constexpr")
CMakeLists.txt:    add_compile_definitions(EIGEN_NO_CUDA)
CMakeLists.txt:    find_package(CUDAToolkit COMPONENTS cudart cublas cusolver REQUIRED)
CMakeLists.txt:    set(CUDA_LIBRARIES CUDA::cudart CUDA::cublas CUDA::cusolver)
CMakeLists.txt:    message(STATUS "MParT is not compiled with CUDA support, so CUBLAS and CUSOLVER will not be used.")
CMakeLists.txt:    set(CUDA_LIBRARIES "")
CMakeLists.txt:target_link_libraries(mpart PRIVATE Kokkos::kokkos Eigen3::Eigen ${CUDA_LIBRARIES} ${EXT_LIBRARIES})
CMakeLists.txt:    target_link_libraries(RunTests PRIVATE mpart Catch2::Catch2 Kokkos::kokkos Eigen3::Eigen ${CUDA_LIBRARIES} ${EXT_LIBRARIES})
src/MapFactoryImpl7.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl7.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl15.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl15.cpp:#if defined(MPART_ENABLE_GPU)
src/MultiIndices/FixedMultiIndexSet.cpp:#if defined(MPART_ENABLE_GPU)
src/MultiIndices/FixedMultiIndexSet.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl11.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl11.cpp:#if defined(MPART_ENABLE_GPU)
src/Utilities/LinearAlgebra.cpp:#if defined(MPART_ENABLE_GPU)
src/Utilities/LinearAlgebra.cpp:                                CUDA_R_64F,
src/Utilities/LinearAlgebra.cpp:                                CUDA_R_64F,
src/Utilities/LinearAlgebra.cpp:                     CUDA_R_64F,
src/Utilities/LinearAlgebra.cpp:                     CUDA_R_64F,
src/Utilities/LinearAlgebra.cpp:                     CUDA_R_64F,
src/Utilities/LinearAlgebra.cpp:                     CUDA_R_64F,
src/Utilities/LinearAlgebra.cpp:                                CUDA_R_64F,
src/Utilities/LinearAlgebra.cpp:                                CUDA_R_64F,
src/Utilities/LinearAlgebra.cpp:                     CUDA_R_64F,
src/Utilities/LinearAlgebra.cpp:                     CUDA_R_64F,
src/Utilities/LinearAlgebra.cpp:                     CUDA_R_64F,
src/Utilities/LinearAlgebra.cpp:                     CUDA_R_64F,
src/TriangularMap.cpp:#if defined(MPART_ENABLE_GPU)
src/TriangularMap.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl13.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl13.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl8.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl8.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl17.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl17.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl9.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl9.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl1.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl1.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl2.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl2.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl12.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl12.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl3.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl3.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl16.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl16.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl14.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl14.cpp:#if defined(MPART_ENABLE_GPU)
src/ParameterizedFunctionBase.cpp:#include "MParT/Utilities/GPUtils.h"
src/ParameterizedFunctionBase.cpp:#if defined(MPART_ENABLE_GPU)
src/ParameterizedFunctionBase.cpp:#if defined(MPART_ENABLE_GPU)
src/ParameterizedFunctionBase.cpp:#if defined(MPART_ENABLE_GPU)
src/ParameterizedFunctionBase.cpp:#if defined(MPART_ENABLE_GPU)
src/ParameterizedFunctionBase.cpp:#if defined(MPART_ENABLE_GPU)
src/ComposedMap.cpp:#if defined(MPART_ENABLE_GPU)
src/ComposedMap.cpp:#if defined(MPART_ENABLE_GPU)
src/AffineMap.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactory.cpp:#if defined(MPART_ENABLE_GPU)
src/Distributions/GaussianSamplerDensity.cpp:#ifdef MPART_ENABLE_GPU
src/Distributions/PullbackDensity.cpp:#if defined(MPART_ENABLE_GPU)
src/MapObjective.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl10.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl10.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl18.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl18.cpp:#if defined(MPART_ENABLE_GPU)
src/SummarizedMap.cpp:#if defined(MPART_ENABLE_GPU)
src/SummarizedMap.cpp:#if defined(MPART_ENABLE_GPU)
src/IdentityMap.cpp:#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
src/MapFactoryImpl4.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl4.cpp:#if defined(MPART_ENABLE_GPU)
src/AffineFunction.cpp:#if defined(MPART_ENABLE_GPU)
src/Initialization.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl5.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl5.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl6.cpp:#if defined(MPART_ENABLE_GPU)
src/MapFactoryImpl6.cpp:#if defined(MPART_ENABLE_GPU)
src/ConditionalMapBase.cpp:#if defined(MPART_ENABLE_GPU)
src/ConditionalMapBase.cpp:#if defined(MPART_ENABLE_GPU)
src/ConditionalMapBase.cpp:#if defined(MPART_ENABLE_GPU)
src/ConditionalMapBase.cpp:#if defined(MPART_ENABLE_GPU)
src/ConditionalMapBase.cpp:#if defined(MPART_ENABLE_GPU)

```
