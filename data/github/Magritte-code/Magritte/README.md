# https://github.com/Magritte-code/Magritte

```console
setup.py:        "Environment :: GPU",
compile.sh:  -DGPU_ACCELERATION=OFF                            \
docs/src/0_getting_started/2_installation.rst:GPU acceleration
docs/src/0_getting_started/2_installation.rst:A GPU-enabled port of Magritte to python using pytorch can be found on `GitHub <https://github.com/Magritte-code/Magritte-torch>`_.
docs/src/0_getting_started/2_installation.rst:Unless GPU acceleration is required, the C++ version of Magritte should be used, as the compiled C++ code is faster on CPU than the python version.
tests/test_raytracer.cpp:    // Size1 lengths = model.geometry.get_ray_lengths_gpu (512, 512);
tests/CMakeLists.txt:if    (GPU_ACCELERATION)
tests/CMakeLists.txt:    set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
tests/CMakeLists.txt:endif (GPU_ACCELERATION)
tests/test_multigrid.cpp://    Size1 lengths = model.geometry.get_ray_lengths_gpu (512, 512);
CMakeLists.txt:option (GPU_ACCELERATION "Use the GPU solver"                    OFF)
CMakeLists.txt:option (GPU_CUDA         "Use Paracabs CUDA implementation"      OFF)
CMakeLists.txt:option (GPU_SYCL         "Usa Paracabs SYCL implementation"      OFF)
CMakeLists.txt:if    (GPU_ACCELERATION)
CMakeLists.txt:    set (MAGRITTE_GPU_ACCELERATION true)
CMakeLists.txt:    if    (GPU_SYCL)
CMakeLists.txt:        set (MAGRITTE_GPU_SYCL true)
CMakeLists.txt:        set (MAGRITTE_GPU_CUDA false)
CMakeLists.txt:    else  (GPU_SYCL)
CMakeLists.txt:        set (MAGRITTE_GPU_SYCL false)
CMakeLists.txt:        set (MAGRITTE_GPU_CUDA true)
CMakeLists.txt:    endif (GPU_SYCL)
CMakeLists.txt:else  (GPU_ACCELERATION)
CMakeLists.txt:    set (MAGRITTE_GPU_ACCELERATION false)
CMakeLists.txt:    set (MAGRITTE_GPU_CUDA         false)
CMakeLists.txt:    set (MAGRITTE_GPU_SYCL         false)
CMakeLists.txt:endif (GPU_ACCELERATION)
CMakeLists.txt:set (PARACABS_USE_ACCELERATOR     ${MAGRITTE_GPU_ACCELERATION})
CMakeLists.txt:set (PARACABS_USE_CUDA            ${MAGRITTE_GPU_CUDA})
CMakeLists.txt:set (PARACABS_USE_SYCL            ${MAGRITTE_GPU_SYCL})
CMakeLists.txt:# Enable CUDA if required
CMakeLists.txt:if (GPU_ACCELERATION)
CMakeLists.txt:    enable_language (CUDA)
CMakeLists.txt:    set (CMAKE_CUDA_ARCHITECTURES OFF)
CMakeLists.txt:    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --use_fast_math --expt-relaxed-constexpr")
src/configure.hpp.in:// GPU acceleration
src/configure.hpp.in:#define GPU_ACCELERATION        @MAGRITTE_GPU_ACCELERATION@
src/bindings/CMakeLists.txt:if    (GPU_ACCELERATION)
src/bindings/CMakeLists.txt:    set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
src/bindings/CMakeLists.txt:endif (GPU_ACCELERATION)
src/bindings/CMakeLists.txt:if    (GPU_ACCELERATION)
src/bindings/CMakeLists.txt:    # Add CUDA library
src/bindings/CMakeLists.txt:    cuda_add_library (core SHARED ${SOURCE_FILES})
src/bindings/CMakeLists.txt:else  (GPU_ACCELERATION)
src/bindings/CMakeLists.txt:endif (GPU_ACCELERATION)
src/CMakeLists.txt:if    (GPU_ACCELERATION)
src/CMakeLists.txt:    set_source_files_properties(${SOURCE_FILES}         PROPERTIES LANGUAGE CUDA)
src/CMakeLists.txt:    set_source_files_properties(io/python/io_python.cpp PROPERTIES LANGUAGE CUDA)
src/CMakeLists.txt:endif (GPU_ACCELERATION)

```
