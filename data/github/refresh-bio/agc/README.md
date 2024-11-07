# https://github.com/refresh-bio/agc

```console
py_agc_api/pybind11-2.11.1/README.rst:6. NVCC (CUDA 11.0 tested in CI)
py_agc_api/pybind11-2.11.1/README.rst:7. NVIDIA PGI (20.9 tested in CI)
py_agc_api/pybind11-2.11.1/PKG-INFO:6. NVCC (CUDA 11.0 tested in CI)
py_agc_api/pybind11-2.11.1/PKG-INFO:7. NVIDIA PGI (20.9 tested in CI)
py_agc_api/pybind11-2.11.1/pybind11.egg-info/PKG-INFO:6. NVCC (CUDA 11.0 tested in CI)
py_agc_api/pybind11-2.11.1/pybind11.egg-info/PKG-INFO:7. NVIDIA PGI (20.9 tested in CI)
py_agc_api/pybind11-2.11.1/pybind11/share/cmake/pybind11/pybind11Common.cmake:      # instance, projects that include other types of source files like CUDA
py_agc_api/pybind11-2.11.1/pybind11/share/cmake/pybind11/pybind11Tools.cmake:  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
py_agc_api/pybind11-2.11.1/pybind11/share/cmake/pybind11/pybind11Tools.cmake:    set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
py_agc_api/pybind11-2.11.1/pybind11/share/cmake/pybind11/pybind11NewTools.cmake:  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
py_agc_api/pybind11-2.11.1/pybind11/share/cmake/pybind11/pybind11NewTools.cmake:    set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
py_agc_api/pybind11-2.11.1/pybind11/include/pybind11/detail/common.h:// For CUDA, GCC7, GCC8:
py_agc_api/pybind11-2.11.1/pybind11/include/pybind11/detail/common.h:// 1.7% for CUDA, -0.2% for GCC7, and 0.0% for GCC8 (using -DCMAKE_BUILD_TYPE=MinSizeRel,
py_agc_api/pybind11-2.11.1/pybind11/include/pybind11/detail/common.h:    && (defined(__CUDACC__) || (defined(__GNUC__) && (__GNUC__ == 7 || __GNUC__ == 8)))
py_agc_api/pybind11-2.11.1/pybind11/include/pybind11/cast.h:    // static_cast works around compiler error with MSVC 17 and CUDA 10.2
py_agc_api/pybind11-2.11.1/pybind11/include/pybind11/numpy.h:#ifdef __CUDACC__

```
