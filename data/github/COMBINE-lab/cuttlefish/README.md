# https://github.com/COMBINE-lab/cuttlefish

```console
include/spdlog/fmt/bundled/format.h:#  define FMT_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__)
include/spdlog/fmt/bundled/format.h:#  define FMT_CUDA_VERSION 0
include/spdlog/fmt/bundled/format.h:// For Intel and NVIDIA compilers both they and the system gcc/msc support UDLs.
include/spdlog/fmt/bundled/format.h:      (!(FMT_ICC_VERSION || FMT_CUDA_VERSION) || FMT_ICC_VERSION >= 1500 || \
include/spdlog/fmt/bundled/format.h:       FMT_CUDA_VERSION >= 700)
include/spdlog/fmt/bundled/format.h:      FMT_CUDA_VERSION == 0 &&                                 \
include/fmt/format-inl.h:  // Chosen instead of std::abort to satisfy Clang in CUDA mode during device
include/fmt/format.h:#  define FMT_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__)
include/fmt/format.h:#  define FMT_CUDA_VERSION 0
include/fmt/format.h:// EDG based compilers (Intel, NVIDIA, Elbrus, etc), GCC and MSVC support UDLs.
include/boost/preprocessor/config/config.hpp:#    if defined __GCCXML__ || defined __CUDACC__ || defined __PATHSCALE__ || defined __DMC__ || defined __CODEGEARC__ || defined __BORLANDC__ || defined __MWERKS__ || ( defined __SUNPRO_CC && __SUNPRO_CC < 0x5120 ) || defined __HP_aCC && !defined __EDG__ || defined __MRC__ || defined __SC__ || (defined(__PGI) && !defined(__EDG__))
include/boost/preprocessor/config/config.hpp:#    if defined _MSC_VER && _MSC_VER >= 1400 && !defined(__clang__) && (defined(__INTELLISENSE__) || (defined(__INTEL_COMPILER) && __INTEL_COMPILER >= 1700) || !(defined __EDG__ || defined __GCCXML__ || defined __CUDACC__ || defined __PATHSCALE__ || defined __DMC__ || defined __CODEGEARC__ || defined __BORLANDC__ || defined __MWERKS__ || defined __SUNPRO_CC || defined __HP_aCC || defined __MRC__ || defined __SC__ || defined __IBMCPP__ || defined __PGI)) && (!defined(_MSVC_TRADITIONAL) || _MSVC_TRADITIONAL)

```
