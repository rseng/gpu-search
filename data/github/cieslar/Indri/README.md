# https://github.com/cieslar/Indri

```console
CMakeLists.txt:#add_definitions(-DGLM_COMPILER=GLM_COMPILER_CUDA30)
CMakeLists.txt:#add_definitions(-D__CUDACC__)
CMakeLists.txt:#add_definitions(-DGLM_FORCE_CUDA )
CMakeLists.txt:#find_package(CUDA REQUIRED)
CMakeLists.txt:#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch sm_20 -ccbin g++" )
CMakeLists.txt:#set(CUDA_NVCC_FLAGS "-arch=sm_20" CACHE STRING "nvcc flags" FORCE)
CMakeLists.txt:#mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

```
