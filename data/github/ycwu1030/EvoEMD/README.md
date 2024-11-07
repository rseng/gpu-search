# https://github.com/ycwu1030/EvoEMD

```console
include/spdlog/fmt/bundled/format-inl.h:  // Chosen instead of std::abort to satisfy Clang in CUDA mode during device
include/spdlog/fmt/bundled/format.h:#  define FMT_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__)
include/spdlog/fmt/bundled/format.h:#  define FMT_CUDA_VERSION 0
include/spdlog/fmt/bundled/format.h:// EDG based compilers (Intel, NVIDIA, Elbrus, etc), GCC and MSVC support UDLs.

```
