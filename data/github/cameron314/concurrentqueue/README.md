# https://github.com/cameron314/concurrentqueue

```console
benchmarks/tbb/tools_api/ittnotify.h: * that occur on a GPU.
benchmarks/dlib/config.h:// You should also consider telling dlib to link against libjpeg, libpng, libgif, fftw, CUDA, 
benchmarks/dlib/config.h:/* #undef DLIB_USE_CUDA */
benchmarks/boost/config/compiler/gcc.hpp:#if !defined(__CUDACC__)
benchmarks/boost/config/compiler/gcc.hpp:// doesn't actually support __int128 as of CUDA_VERSION=5000
benchmarks/boost/config/compiler/gcc.hpp:#if defined(__SIZEOF_INT128__) && !defined(__CUDACC__)
benchmarks/boost/config/compiler/nvcc.hpp://  NVIDIA CUDA C++ compiler setup
benchmarks/boost/config/compiler/nvcc.hpp:#  define BOOST_COMPILER "NVIDIA CUDA C++ Compiler"
benchmarks/boost/config/compiler/nvcc.hpp:// NVIDIA Specific support
benchmarks/boost/config/compiler/nvcc.hpp:// BOOST_GPU_ENABLED : Flag a function or a method as being enabled on the host and device
benchmarks/boost/config/compiler/nvcc.hpp:#define BOOST_GPU_ENABLED __host__ __device__
benchmarks/boost/config/compiler/intel.hpp:#if defined(__LP64__) && defined(__GNUC__) && (BOOST_INTEL_CXX_VERSION >= 1310) && !defined(__CUDACC__)
benchmarks/boost/config/compiler/clang.hpp:// doesn't actually support __int128 as of CUDA_VERSION=5000
benchmarks/boost/config/compiler/clang.hpp:#if defined(__SIZEOF_INT128__) && !defined(__CUDACC__) && !defined(_MSC_VER)
benchmarks/boost/config/select_compiler_config.hpp:#if defined __CUDACC__
benchmarks/boost/config/select_compiler_config.hpp://  NVIDIA CUDA C++ compiler for GPU
benchmarks/boost/config/select_compiler_config.hpp:#elif defined __clang__ && !defined(__CUDACC__) && !defined(__ibmxl__)
benchmarks/boost/config/select_compiler_config.hpp:// when using clang and cuda at same time, you want to appear as gcc
benchmarks/boost/config/suffix.hpp:// Set some default values GPU support
benchmarks/boost/config/suffix.hpp:#  ifndef BOOST_GPU_ENABLED
benchmarks/boost/config/suffix.hpp:#  define BOOST_GPU_ENABLED
benchmarks/boost/config/suffix.hpp:#    if defined(__CUDACC__)
benchmarks/boost/mpl/aux_/config/gpu.hpp:#ifndef BOOST_MPL_AUX_CONFIG_GPU_HPP_INCLUDED
benchmarks/boost/mpl/aux_/config/gpu.hpp:#define BOOST_MPL_AUX_CONFIG_GPU_HPP_INCLUDED
benchmarks/boost/mpl/aux_/config/gpu.hpp:#if !defined(BOOST_MPL_CFG_GPU_ENABLED) \
benchmarks/boost/mpl/aux_/config/gpu.hpp:#   define BOOST_MPL_CFG_GPU_ENABLED BOOST_GPU_ENABLED
benchmarks/boost/mpl/aux_/config/gpu.hpp:#if defined __CUDACC__
benchmarks/boost/mpl/aux_/config/gpu.hpp:#    define BOOST_MPL_CFG_GPU 1
benchmarks/boost/mpl/aux_/config/gpu.hpp:#    define BOOST_MPL_CFG_GPU 0
benchmarks/boost/mpl/aux_/config/gpu.hpp:#endif // BOOST_MPL_AUX_CONFIG_GPU_HPP_INCLUDED
benchmarks/boost/mpl/assert.hpp:#include <boost/mpl/aux_/config/gpu.hpp>
benchmarks/boost/mpl/assert.hpp:    || (BOOST_MPL_CFG_GCC != 0) || (BOOST_MPL_CFG_GPU != 0)
benchmarks/boost/mpl/has_xxx.hpp:      || (BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1800)) && defined(__CUDACC__)) \
benchmarks/boost/core/swap.hpp:  BOOST_GPU_ENABLED
benchmarks/boost/core/swap.hpp:  BOOST_GPU_ENABLED
benchmarks/boost/core/swap.hpp:  BOOST_GPU_ENABLED
benchmarks/boost/preprocessor/config/config.hpp:#    if defined __GCCXML__ || defined __CUDACC__ || defined __PATHSCALE__ || defined __DMC__ || defined __CODEGEARC__ || defined __BORLANDC__ || defined __MWERKS__ || ( defined __SUNPRO_CC && __SUNPRO_CC < 0x5130 ) || defined __HP_aCC && !defined __EDG__ || defined __MRC__ || defined __SC__ || defined __IBMCPP__ || defined __PGI
benchmarks/boost/preprocessor/config/config.hpp:#    if defined _MSC_VER && _MSC_VER >= 1400 && (defined(__INTELLISENSE__) || !(defined __EDG__ || defined __GCCXML__ || defined __CUDACC__ || defined __PATHSCALE__ || defined __clang__ || defined __DMC__ || defined __CODEGEARC__ || defined __BORLANDC__ || defined __MWERKS__ || defined __SUNPRO_CC || defined __HP_aCC || defined __MRC__ || defined __SC__ || defined __IBMCPP__ || defined __PGI))
benchmarks/boost/type_traits/intrinsics.hpp:#if defined(BOOST_CLANG) && defined(__has_feature) && !defined(__CUDACC__)
benchmarks/boost/type_traits/intrinsics.hpp:// Note that these intrinsics are disabled for the CUDA meta-compiler as it appears
benchmarks/boost/atomic/detail/config.hpp:#if defined(__CUDACC__)

```
