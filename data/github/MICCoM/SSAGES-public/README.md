# https://github.com/SSAGESproject/SSAGES

```console
hooks/hoomd/Driver.cpp:        else if (cmd_args.find("--mode=gpu") != std::string::npos)
hooks/hoomd/Driver.cpp:            execution_mode = ExecutionConfiguration::executionMode::GPU;
hooks/hoomd/Driver.cpp:        std::vector<int> gpu_id(1,-1);
hooks/hoomd/Driver.cpp:            gpu_id,
hooks/hoomd/Driver.cpp:            gpu_id,
hooks/hoomd/FindHOOMD.cmake:# Find CUDA and set it up
hooks/hoomd/FindHOOMD.cmake:include (HOOMDCUDASetup)
hooks/lammps/CMakeLists.txt:                     "gpu"
hooks/gromacs/CMakeLists.txt:set (CUDA_NVCC_FLAGS "-std=c++11; -I${PROJECT_SOURCE_DIR}/src; -I${PROJECT_SOURCE_DIR}/include; -I${CMAKE_CURRENT_SOURCE_DIR}")
hooks/gromacs/CMakeLists.txt:    -DGMX_GPU=${GMX_GPU}
hooks/gromacs/CMakeLists.txt:    -DCUDA_PROPAGATE_HOST_FLAGS=OFF
hooks/gromacs/CMakeLists.txt:    -DCUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS}
hooks/gromacs/gmx_diff_2016.x.patch:diff --git /gromacs-original/src/gromacs/gpu_utils/cudautils.cu /gromacs-ssages/src/gromacs/gpu_utils/cudautils.cu
hooks/gromacs/gmx_diff_2016.x.patch:--- /gromacs-original/src/gromacs/gpu_utils/cudautils.cu
hooks/gromacs/gmx_diff_2016.x.patch:+++ /gromacs-ssages/src/gromacs/gpu_utils/cudautils.cu
hooks/gromacs/gmx_diff_2016.x.patch:@@ -85,18 +85,6 @@ int cu_copy_D2H_async(void * h_dest, void * d_src, size_t bytes, cudaStream_t s
hooks/gromacs/gmx_diff_2016.x.patch:diff --git /gromacs-original/src/gromacs/gpu_utils/cudautils.cuh /gromacs-ssages/src/gromacs/gpu_utils/cudautils.cuh
hooks/gromacs/gmx_diff_2016.x.patch:--- /gromacs-original/src/gromacs/gpu_utils/cudautils.cuh
hooks/gromacs/gmx_diff_2016.x.patch:+++ /gromacs-ssages/src/gromacs/gpu_utils/cudautils.cuh
hooks/gromacs/gmx_diff_2016.x.patch: int cu_copy_D2H_async(void * /*h_dest*/, void * /*d_src*/, size_t /*bytes*/, cudaStream_t /*s = 0*/);
doc/source/Engines.rst:``-DGMX_GPU=ON``, and ``-DGMX_DOUBLE=ON`` are supported. With newer versions
doc/source/Introduction.rst:power, including custom-built computer architectures and GPU-based computing,

```
