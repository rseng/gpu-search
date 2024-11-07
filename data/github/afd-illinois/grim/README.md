# https://github.com/AFD-Illinois/grim

```console
run_summit.bsub:module load gcc cuda/11.4 petsc boost nsight-systems
run_summit.bsub:export AF_CUDA_MAX_JIT_LEN=30
run_summit.bsub:# -g # GPU per rs
run_summit.bsub:# -l latency optimization in choosing cores (CPU-CPU, GPU-CPU, CPU-MEM)
run_summit.bsub:# The "smpiargs" argument is used to tell Spectrum MPI we're GPU-aware
run_summit.bsub:jsrun --smpiargs="-gpu" -n $(( $NNODES * 6 )) -r 6 -a 1 -g 1 -d packed -c 6 -b packed:6 -l GPU-CPU \
run_summit.bsub:      nsys profile --capture-range=cudaProfilerApi -o grim_%q{OMPI_COMM_WORLD_RANK} ~/grim/grim
make_summit.sh:module load gcc cuda/11.4 petsc boost python
make_summit.sh:        -DArrayFire_ROOT_DIR=/gpfs/alpine/proj-shared/ast171/libs/arrayfire-cuda114 \
src/grim.cpp:#include "cuda_profiler_api.h"
src/grim.cpp:    cudaProfilerStart();
src/CMakeLists.txt:set(ARCH "CUDA") # Choose CPU/OpenCL/CUDA
src/CMakeLists.txt:find_package(CUDA REQUIRED)
src/CMakeLists.txt:elseif (ARCH STREQUAL "CUDA")
src/CMakeLists.txt:  set(ArrayFire_LIBRARIES ${ArrayFire_CUDA_LIBRARIES})
src/CMakeLists.txt:elseif (ARCH STREQUAL "OpenCL")
src/CMakeLists.txt:  set(ArrayFire_LIBRARIES ${ArrayFire_OpenCL_LIBRARIES})
src/CMakeLists.txt:		      ${CUDA_LIBRARIES}
src/CMakeLists.txt:message("CUDA LIBS        : " ${CUDA_LIBRARIES})
src/params.hpp:    GPU_BATCH_SOLVER, CPU_BATCH_SOLVER
src/cmake/modules/FindOpenCL.cmake:# - Find the OpenCL headers and library
src/cmake/modules/FindOpenCL.cmake:#  OPENCL_FOUND        : TRUE if found, FALSE otherwise
src/cmake/modules/FindOpenCL.cmake:#  OPENCL_INCLUDE_DIRS : Include directories for OpenCL
src/cmake/modules/FindOpenCL.cmake:#  OPENCL_LIBRARIES    : The libraries to link against
src/cmake/modules/FindOpenCL.cmake:# The user can set the OPENCLROOT environment variable to help finding OpenCL
src/cmake/modules/FindOpenCL.cmake: set(ENV_OPENCLROOT "$ENV{ATISTREAMSDKROOT}")
src/cmake/modules/FindOpenCL.cmake: set(ENV_OPENCLROOT "$ENV{AMDAPPSDKROOT}")
src/cmake/modules/FindOpenCL.cmake: set(ENV_OPENCLROOT "$ENV{INTELOCLSDKROOT}")
src/cmake/modules/FindOpenCL.cmake:set(ENV_OPENCLROOT2 "$ENV{OPENCLROOT}")
src/cmake/modules/FindOpenCL.cmake:if(ENV_OPENCLROOT2)
src/cmake/modules/FindOpenCL.cmake: set(ENV_OPENCLROOT "$ENV{OPENCLROOT}")
src/cmake/modules/FindOpenCL.cmake:endif(ENV_OPENCLROOT2)
src/cmake/modules/FindOpenCL.cmake:if(ENV_OPENCLROOT)
src/cmake/modules/FindOpenCL.cmake:    OPENCL_INCLUDE_DIR
src/cmake/modules/FindOpenCL.cmake:    NAMES CL/cl.h OpenCL/cl.h
src/cmake/modules/FindOpenCL.cmake:    PATHS "${ENV_OPENCLROOT}/include"
src/cmake/modules/FindOpenCL.cmake:    #NO_DEFAULT_PATH  #uncomment this is you wish to surpress the use of default paths for OpenCL
src/cmake/modules/FindOpenCL.cmake:      set(OPENCL_LIB_SEARCH_PATH
src/cmake/modules/FindOpenCL.cmake:          "${OPENCL_LIB_SEARCH_PATH}"
src/cmake/modules/FindOpenCL.cmake:          "${ENV_OPENCLROOT}/lib/x86")
src/cmake/modules/FindOpenCL.cmake:      set(OPENCL_LIB_SEARCH_PATH
src/cmake/modules/FindOpenCL.cmake:          "${OPENCL_LIB_SEARCH_PATH}"
src/cmake/modules/FindOpenCL.cmake:          "${ENV_OPENCLROOT}/lib/x86_64")
src/cmake/modules/FindOpenCL.cmake:    OPENCL_LIBRARY
src/cmake/modules/FindOpenCL.cmake:    NAMES OpenCL
src/cmake/modules/FindOpenCL.cmake:    PATHS "${OPENCL_LIB_SEARCH_PATH}"
src/cmake/modules/FindOpenCL.cmake:    #NO_DEFAULT_PATH  #uncomment this is you wish to surpress the use of default paths for OpenCL
src/cmake/modules/FindOpenCL.cmake:else(ENV_OPENCLROOT)
src/cmake/modules/FindOpenCL.cmake:    OPENCL_INCLUDE_DIR
src/cmake/modules/FindOpenCL.cmake:    NAMES CL/cl.h OpenCL/cl.h
src/cmake/modules/FindOpenCL.cmake:    OPENCL_LIBRARY
src/cmake/modules/FindOpenCL.cmake:    NAMES OpenCL
src/cmake/modules/FindOpenCL.cmake:endif(ENV_OPENCLROOT)
src/cmake/modules/FindOpenCL.cmake:  OPENCL
src/cmake/modules/FindOpenCL.cmake:  OPENCL_LIBRARY OPENCL_INCLUDE_DIR
src/cmake/modules/FindOpenCL.cmake:if(OPENCL_FOUND)
src/cmake/modules/FindOpenCL.cmake:  set(OPENCL_INCLUDE_DIRS "${OPENCL_INCLUDE_DIR}")
src/cmake/modules/FindOpenCL.cmake:  set(OPENCL_LIBRARIES "${OPENCL_LIBRARY}")
src/cmake/modules/FindOpenCL.cmake:else(OPENCL_FOUND)
src/cmake/modules/FindOpenCL.cmake:  set(OPENCL_INCLUDE_DIRS)
src/cmake/modules/FindOpenCL.cmake:  set(OPENCL_LIBRARIES)
src/cmake/modules/FindOpenCL.cmake:endif(OPENCL_FOUND)
src/cmake/modules/FindOpenCL.cmake:  OPENCL_INCLUDE_DIR
src/cmake/modules/FindOpenCL.cmake:  OPENCL_LIBRARY
src/problem/linear_modes/params.cpp:  int linearSolver = linearSolvers::GPU_BATCH_SOLVER;
src/problem/atmosphere/CMakeLists.txt:#             OpenCL specification is a different amount for different 
src/problem/shock_tests/params.cpp:  int linearSolver = linearSolvers::GPU_BATCH_SOLVER;
src/problem/magnetized_field_loop_advection/CMakeLists.txt:#             OpenCL specification is a different amount for different 
src/problem/magnetized_explosion/CMakeLists.txt:#             OpenCL specification is a different amount for different 
src/problem/advection_test/CMakeLists.txt:#             OpenCL specification is a different amount for different 
src/problem/bondi_viscous/CMakeLists.txt:#             OpenCL specification is a different amount for different 
src/problem/anisotropic_conduction/CMakeLists.txt:#             OpenCL specification is a different amount for different 
src/problem/torus/params.cpp:  // 4 GPUs on SAVIO
src/problem/torus/params.cpp:  int linearSolver = linearSolvers::GPU_BATCH_SOLVER;

```
