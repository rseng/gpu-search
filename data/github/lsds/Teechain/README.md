# https://github.com/lsds/Teechain

```console
src/trusted/libs/bitcoin/script/interpreter.h:    SCRIPT_VERIFY_SIGPUSHONLY = (1U << 5),
src/trusted/libs/bitcoin/script/interpreter.cpp:    if ((flags & SCRIPT_VERIFY_SIGPUSHONLY) != 0 && !scriptSig.IsPushOnly()) {
src/trusted/libs/boost/config/compiler/gcc.hpp:#if !defined(__CUDACC__)
src/trusted/libs/boost/config/compiler/gcc.hpp:// doesn't actually support __int128 as of CUDA_VERSION=5000
src/trusted/libs/boost/config/compiler/gcc.hpp:#if defined(__SIZEOF_INT128__) && !defined(__CUDACC__)
src/trusted/libs/boost/config/compiler/nvcc.hpp://  NVIDIA CUDA C++ compiler setup
src/trusted/libs/boost/config/compiler/nvcc.hpp:#  define BOOST_COMPILER "NVIDIA CUDA C++ Compiler"
src/trusted/libs/boost/config/compiler/nvcc.hpp:// NVIDIA Specific support
src/trusted/libs/boost/config/compiler/nvcc.hpp:// BOOST_GPU_ENABLED : Flag a function or a method as being enabled on the host and device
src/trusted/libs/boost/config/compiler/nvcc.hpp:#define BOOST_GPU_ENABLED __host__ __device__
src/trusted/libs/boost/config/select_compiler_config.hpp:#elif defined __CUDACC__
src/trusted/libs/boost/config/select_compiler_config.hpp://  NVIDIA CUDA C++ compiler for GPU
src/trusted/libs/boost/config/suffix.hpp:// Set some default values GPU support
src/trusted/libs/boost/config/suffix.hpp:#  ifndef BOOST_GPU_ENABLED
src/trusted/libs/boost/config/suffix.hpp:#  define BOOST_GPU_ENABLED 
src/trusted/libs/boost/preprocessor/config/config.hpp:#    if defined __GCCXML__ || defined __CUDACC__ || defined __PATHSCALE__ || defined __clang__ || defined __DMC__ || defined __CODEGEARC__ || defined __BORLANDC__ || defined __MWERKS__ || defined __SUNPRO_CC || defined __HP_aCC && !defined __EDG__ || defined __MRC__ || defined __SC__ || defined __IBMCPP__ || defined __PGI
src/trusted/libs/boost/preprocessor/config/config.hpp:#    if defined _MSC_VER && _MSC_VER >= 1400 && !(defined __EDG__ || defined __GCCXML__ || defined __CUDACC__ || defined __PATHSCALE__ || defined __clang__ || defined __DMC__ || defined __CODEGEARC__ || defined __BORLANDC__ || defined __MWERKS__ || defined __SUNPRO_CC || defined __HP_aCC || defined __MRC__ || defined __SC__ || defined __IBMCPP__ || defined __PGI)

```
