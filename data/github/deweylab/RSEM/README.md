# https://github.com/deweylab/RSEM

```console
boost/math/special_functions/lanczos.hpp:#if !defined(_CRAYC) && !defined(__CUDACC__) && (!defined(__GNUC__) || (__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ > 3)))
boost/config/compiler/gcc.hpp:#if !defined(__CUDACC__)
boost/config/compiler/gcc.hpp:// doesn't actually support __int128 as of CUDA_VERSION=5000
boost/config/compiler/gcc.hpp:#if defined(__SIZEOF_INT128__) && !defined(__CUDACC__)
boost/config/compiler/nvcc.hpp://  NVIDIA CUDA C++ compiler setup
boost/config/compiler/nvcc.hpp:#  define BOOST_COMPILER "NVIDIA CUDA C++ Compiler"
boost/config/compiler/nvcc.hpp:// NVIDIA Specific support
boost/config/compiler/nvcc.hpp:// BOOST_GPU_ENABLED : Flag a function or a method as being enabled on the host and device
boost/config/compiler/nvcc.hpp:#define BOOST_GPU_ENABLED __host__ __device__
boost/config/select_compiler_config.hpp:#if defined __CUDACC__
boost/config/select_compiler_config.hpp://  NVIDIA CUDA C++ compiler for GPU
boost/config/suffix.hpp:// Set some default values GPU support
boost/config/suffix.hpp:#  ifndef BOOST_GPU_ENABLED
boost/config/suffix.hpp:#  define BOOST_GPU_ENABLED
boost/preprocessor/config/config.hpp:#    if defined __GCCXML__ || defined __CUDACC__ || defined __PATHSCALE__ || defined __clang__ || defined __DMC__ || defined __CODEGEARC__ || defined __BORLANDC__ || defined __MWERKS__ || defined __SUNPRO_CC || defined __HP_aCC && !defined __EDG__ || defined __MRC__ || defined __SC__ || defined __IBMCPP__ || defined __PGI
boost/preprocessor/config/config.hpp:#    if defined _MSC_VER && _MSC_VER >= 1400 && !(defined __EDG__ || defined __GCCXML__ || defined __CUDACC__ || defined __PATHSCALE__ || defined __clang__ || defined __DMC__ || defined __CODEGEARC__ || defined __BORLANDC__ || defined __MWERKS__ || defined __SUNPRO_CC || defined __HP_aCC || defined __MRC__ || defined __SC__ || defined __IBMCPP__ || defined __PGI)
boost/type_traits.hpp:#if !defined(__BORLANDC__) && !defined(__CUDACC__)
boost/concept/detail/has_constraints.hpp:#if BOOST_WORKAROUND(__SUNPRO_CC, <= 0x580) || defined(__CUDACC__)

```
