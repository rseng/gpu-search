# https://github.com/cosmicrays/hermes

```console
.clang-tidy:  - key:             cppcoreguidelines-non-private-member-variables-in-classes.IgnoreClassesWithAllMemberVariablesBeingPublic
lib/pybind11/README.rst:6. NVCC (CUDA 11.0 tested in CI)
lib/pybind11/README.rst:7. NVIDIA PGI (20.9 tested in CI)
lib/pybind11/.pre-commit-config.yaml:    types_or: [c++, c, cuda]
lib/pybind11/tools/pybind11Common.cmake:      # instance, projects that include other types of source files like CUDA
lib/pybind11/tools/pybind11Tools.cmake:  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
lib/pybind11/tools/pybind11Tools.cmake:    set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
lib/pybind11/tools/pybind11NewTools.cmake:  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
lib/pybind11/tools/pybind11NewTools.cmake:    set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
lib/pybind11/include/pybind11/detail/common.h:// For CUDA, GCC7, GCC8:
lib/pybind11/include/pybind11/detail/common.h:// 1.7% for CUDA, -0.2% for GCC7, and 0.0% for GCC8 (using -DCMAKE_BUILD_TYPE=MinSizeRel,
lib/pybind11/include/pybind11/detail/common.h:    && (defined(__CUDACC__) || (defined(__GNUC__) && (__GNUC__ == 7 || __GNUC__ == 8)))
lib/pybind11/include/pybind11/cast.h:    // static_cast works around compiler error with MSVC 17 and CUDA 10.2
lib/pybind11/include/pybind11/numpy.h:#ifdef __CUDACC__

```
