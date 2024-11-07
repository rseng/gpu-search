# https://github.com/GATB/gatb-core

```console
gatb-core/thirdparty/boost/math/special_functions/next.hpp:#if !defined(_CRAYC) && !defined(__CUDACC__) && (!defined(__GNUC__) || (__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ > 3)))
gatb-core/thirdparty/boost/math/special_functions/lanczos.hpp:#if !defined(_CRAYC) && !defined(__CUDACC__) && (!defined(__GNUC__) || (__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ > 3)))
gatb-core/thirdparty/boost/config/detail/select_compiler_config.hpp:#if defined __CUDACC__
gatb-core/thirdparty/boost/config/detail/select_compiler_config.hpp://  NVIDIA CUDA C++ compiler for GPU
gatb-core/thirdparty/boost/config/detail/suffix.hpp:// Set some default values GPU support
gatb-core/thirdparty/boost/config/detail/suffix.hpp:#  ifndef BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/config/detail/suffix.hpp:#  define BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/config/detail/suffix.hpp:#    if defined(__CUDACC__)
gatb-core/thirdparty/boost/config/compiler/gcc.hpp:#if !defined(__CUDACC__)
gatb-core/thirdparty/boost/config/compiler/gcc.hpp:// doesn't actually support __int128 as of CUDA_VERSION=7500
gatb-core/thirdparty/boost/config/compiler/gcc.hpp:#if defined(__CUDACC__)
gatb-core/thirdparty/boost/config/compiler/gcc.hpp:// Nevertheless, as of CUDA 7.5, using __float128 with the host
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp://  NVIDIA CUDA C++ compiler setup
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:#  define BOOST_COMPILER "NVIDIA CUDA C++ Compiler"
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && defined(__CUDACC_VER_BUILD__)
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:#  define BOOST_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 1000000 + __CUDACC_VER_MINOR__ * 10000 + __CUDACC_VER_BUILD__)
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:// We don't really know what the CUDA version is, but it's definitely before 7.5:
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:#  define BOOST_CUDA_VERSION 7000000
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:// NVIDIA Specific support
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:// BOOST_GPU_ENABLED : Flag a function or a method as being enabled on the host and device
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:#define BOOST_GPU_ENABLED __host__ __device__
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:// A bug in version 7.0 of CUDA prevents use of variadic templates in some occasions
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:#if BOOST_CUDA_VERSION < 7050000
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:#if (BOOST_CUDA_VERSION > 8000000) && (BOOST_CUDA_VERSION < 8010000)
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:// CUDA (8.0) has no constexpr support in msvc mode:
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:#if defined(_MSC_VER) && (BOOST_CUDA_VERSION < 9000000)
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:#ifdef __CUDACC__
gatb-core/thirdparty/boost/config/compiler/nvcc.hpp:#if (BOOST_CUDA_VERSION >= 8000000) && (BOOST_CUDA_VERSION < 8010000)
gatb-core/thirdparty/boost/config/compiler/intel.hpp:#if defined(__CUDACC__)
gatb-core/thirdparty/boost/config/compiler/pgi.hpp://  Copyright 2017, NVIDIA CORPORATION.
gatb-core/thirdparty/boost/config/compiler/clang.hpp:// doesn't actually support __int128 as of CUDA_VERSION=7500
gatb-core/thirdparty/boost/config/compiler/clang.hpp:#if defined(__CUDACC__)
gatb-core/thirdparty/boost/config/select_compiler_config.hpp:#if defined __CUDACC__
gatb-core/thirdparty/boost/config/select_compiler_config.hpp://  NVIDIA CUDA C++ compiler for GPU
gatb-core/thirdparty/boost/config/select_compiler_config.hpp:#elif defined __clang__ && !defined(__CUDACC__) && !defined(__ibmxl__)
gatb-core/thirdparty/boost/config/select_compiler_config.hpp:// when using clang and cuda at same time, you want to appear as gcc
gatb-core/thirdparty/boost/config/suffix.hpp:// Set some default values GPU support
gatb-core/thirdparty/boost/config/suffix.hpp:#  ifndef BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/config/suffix.hpp:#  define BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/config/suffix.hpp:#    if defined(__CUDACC__)
gatb-core/thirdparty/boost/utility/value_init.hpp:      BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:      BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/utility/value_init.hpp:    template <class T> BOOST_GPU_ENABLED operator T() const
gatb-core/thirdparty/boost/mpl/for_each.hpp:#include <boost/mpl/aux_/config/gpu.hpp>
gatb-core/thirdparty/boost/mpl/for_each.hpp:    BOOST_MPL_CFG_GPU_ENABLED
gatb-core/thirdparty/boost/mpl/for_each.hpp:    BOOST_MPL_CFG_GPU_ENABLED
gatb-core/thirdparty/boost/mpl/for_each.hpp:BOOST_MPL_CFG_GPU_ENABLED
gatb-core/thirdparty/boost/mpl/for_each.hpp:BOOST_MPL_CFG_GPU_ENABLED
gatb-core/thirdparty/boost/mpl/aux_/config/gpu.hpp:#ifndef BOOST_MPL_AUX_CONFIG_GPU_HPP_INCLUDED
gatb-core/thirdparty/boost/mpl/aux_/config/gpu.hpp:#define BOOST_MPL_AUX_CONFIG_GPU_HPP_INCLUDED
gatb-core/thirdparty/boost/mpl/aux_/config/gpu.hpp:#if !defined(BOOST_MPL_CFG_GPU_ENABLED) \
gatb-core/thirdparty/boost/mpl/aux_/config/gpu.hpp:#   define BOOST_MPL_CFG_GPU_ENABLED BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/mpl/aux_/config/gpu.hpp:#if defined __CUDACC__
gatb-core/thirdparty/boost/mpl/aux_/config/gpu.hpp:#    define BOOST_MPL_CFG_GPU 1
gatb-core/thirdparty/boost/mpl/aux_/config/gpu.hpp:#    define BOOST_MPL_CFG_GPU 0
gatb-core/thirdparty/boost/mpl/aux_/config/gpu.hpp:#endif // BOOST_MPL_AUX_CONFIG_GPU_HPP_INCLUDED
gatb-core/thirdparty/boost/mpl/aux_/unwrap.hpp:#include <boost/mpl/aux_/config/gpu.hpp>
gatb-core/thirdparty/boost/mpl/aux_/unwrap.hpp:BOOST_MPL_CFG_GPU_ENABLED
gatb-core/thirdparty/boost/mpl/aux_/unwrap.hpp:BOOST_MPL_CFG_GPU_ENABLED
gatb-core/thirdparty/boost/mpl/aux_/unwrap.hpp:BOOST_MPL_CFG_GPU_ENABLED
gatb-core/thirdparty/boost/mpl/assert.hpp:#include <boost/mpl/aux_/config/gpu.hpp>
gatb-core/thirdparty/boost/mpl/assert.hpp:    || (BOOST_MPL_CFG_GCC != 0) || (BOOST_MPL_CFG_GPU != 0) || defined(__PGI)
gatb-core/thirdparty/boost/mpl/has_xxx.hpp:      || (BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1800)) && defined(__CUDACC__)) \
gatb-core/thirdparty/boost/core/swap.hpp:  BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/core/swap.hpp:  BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/core/swap.hpp:  BOOST_GPU_ENABLED
gatb-core/thirdparty/boost/core/empty_value.hpp:#elif defined(BOOST_CLANG) && !defined(__CUDACC__)
gatb-core/thirdparty/boost/preprocessor/config/config.hpp:#    if defined __GCCXML__ || defined __CUDACC__ || defined __PATHSCALE__ || defined __DMC__ || defined __CODEGEARC__ || defined __BORLANDC__ || defined __MWERKS__ || ( defined __SUNPRO_CC && __SUNPRO_CC < 0x5120 ) || defined __HP_aCC && !defined __EDG__ || defined __MRC__ || defined __SC__ || (defined(__PGI) && !defined(__EDG__))
gatb-core/thirdparty/boost/preprocessor/config/config.hpp:#    if defined _MSC_VER && _MSC_VER >= 1400 && !defined(__clang__) && (defined(__INTELLISENSE__) || (defined(__INTEL_COMPILER) && __INTEL_COMPILER >= 1700) || !(defined __EDG__ || defined __GCCXML__ || defined __CUDACC__ || defined __PATHSCALE__ || defined __DMC__ || defined __CODEGEARC__ || defined __BORLANDC__ || defined __MWERKS__ || defined __SUNPRO_CC || defined __HP_aCC || defined __MRC__ || defined __SC__ || defined __IBMCPP__ || defined __PGI)) && (!defined(_MSVC_TRADITIONAL) || _MSVC_TRADITIONAL)
gatb-core/thirdparty/boost/type_traits/intrinsics.hpp:#if defined(BOOST_CLANG) && defined(__has_feature) && !defined(__CUDACC__)
gatb-core/thirdparty/boost/type_traits/intrinsics.hpp:// Note that these intrinsics are disabled for the CUDA meta-compiler as it appears
gatb-core/thirdparty/boost/type_traits.hpp:#if !defined(__BORLANDC__) && !defined(__CUDACC__)
gatb-core/thirdparty/hdf5/config/cmake_ext_mod/ConfigureChecks.cmake:CHECK_FUNCTION_EXISTS (sigprocmask       ${HDF_PREFIX}_HAVE_SIGPROCMASK)
gatb-core/thirdparty/hdf5/config/cmake/H5pubconf.h.in:/* Define to 1 if you have the `sigprocmask' function. */
gatb-core/thirdparty/hdf5/config/cmake/H5pubconf.h.in:#cmakedefine H5_HAVE_SIGPROCMASK @H5_HAVE_SIGPROCMASK@
gatb-core/thirdparty/hdf5/tools/lib/io_timer.h:    HDF5_FILE_OPENCLOSE,
gatb-core/thirdparty/hdf5/src/H5detect.c: * do. If sigsetjmp/siglongjmp are not supported, need to use sigprocmask to
gatb-core/thirdparty/hdf5/src/H5detect.c:#if !defined(H5HAVE_SIGJMP) && defined(H5_HAVE_SIGPROCMASK)
gatb-core/thirdparty/hdf5/src/H5detect.c:    /* Use sigprocmask to unblock the signal if sigsetjmp/siglongjmp are not */
gatb-core/thirdparty/hdf5/src/H5detect.c:    HDsigprocmask(SIG_UNBLOCK, &set, NULL);
gatb-core/thirdparty/hdf5/src/H5detect.c:#if !defined(H5HAVE_SIGJMP) && defined(H5_HAVE_SIGPROCMASK)
gatb-core/thirdparty/hdf5/src/H5detect.c:    /* Use sigprocmask to unblock the signal if sigsetjmp/siglongjmp are not */
gatb-core/thirdparty/hdf5/src/H5detect.c:    HDsigprocmask(SIG_UNBLOCK, &set, NULL);
gatb-core/thirdparty/hdf5/src/H5detect.c:#if !defined(H5HAVE_SIGJMP) && defined(H5_HAVE_SIGPROCMASK)
gatb-core/thirdparty/hdf5/src/H5detect.c:    /* Use sigprocmask to unblock the signal if sigsetjmp/siglongjmp are not */
gatb-core/thirdparty/hdf5/src/H5detect.c:    HDsigprocmask(SIG_UNBLOCK, &set, NULL);
gatb-core/thirdparty/hdf5/src/H5detect.c:#ifdef H5_HAVE_SIGPROCMASK
gatb-core/thirdparty/hdf5/src/H5detect.c:    fprintf(rawoutstream, "/* sigprocmask() support: yes */\n");
gatb-core/thirdparty/hdf5/src/H5detect.c:    fprintf(rawoutstream, "/* sigprocmask() support: no */\n");
gatb-core/thirdparty/hdf5/src/H5Rpublic.h:#include "H5Gpublic.h"
gatb-core/thirdparty/hdf5/src/H5config.h.in:/* Define to 1 if you have the `sigprocmask' function. */
gatb-core/thirdparty/hdf5/src/H5config.h.in:#undef HAVE_SIGPROCMASK
gatb-core/thirdparty/hdf5/src/H5Rdeprec.c: * Return:      Success:	An object type (as defined in H5Gpublic.h)
gatb-core/thirdparty/hdf5/src/H5Gprivate.h:#include "H5Gpublic.h"
gatb-core/thirdparty/hdf5/src/H5private.h:#ifndef HDsigprocmask
gatb-core/thirdparty/hdf5/src/H5private.h:    #define HDsigprocmask(H,S,O)  sigprocmask(H,S,O)
gatb-core/thirdparty/hdf5/src/H5private.h:#endif /* HDsigprocmask */
gatb-core/thirdparty/hdf5/src/hdf5.h:#include "H5Gpublic.h"          /* Groups                                   */
gatb-core/thirdparty/hdf5/src/CMakeLists.txt:    ${HDF5_SRC_DIR}/H5Gpublic.h
gatb-core/thirdparty/hdf5/src/H5Gpublic.h: * Created:             H5Gpublic.h
gatb-core/thirdparty/hdf5/src/H5Gpublic.h:#ifndef _H5Gpublic_H
gatb-core/thirdparty/hdf5/src/H5Gpublic.h:#define _H5Gpublic_H
gatb-core/thirdparty/hdf5/src/H5Gpublic.h:#endif /* _H5Gpublic_H */

```
