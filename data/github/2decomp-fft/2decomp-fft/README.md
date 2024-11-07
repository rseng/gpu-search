# https://github.com/2decomp-fft/2decomp-fft

```console
CONTRIBUTORS:Filippo Spiga (NVIDIA, UK)
CHANGELOG.md:- Fuse transpose CPU and GPU memory buffers to reduce memory usage. See [PR 271](https://github.com/2decomp-fft/2decomp-fft/pull/271)
README.md:including debugging builds, building for GPUs and selecting external FFT libraries.
README.md:### GPU compilation
README.md:The library can perform multi GPU offoloading using the NVHPC compiler suite for NVIDIA hardware. 
README.md:The implementation is based on CUDA-aware MPI and NVIDIA Collective Communication Library (NCCL).
README.md:For details of how to configure 2DECOMP&FFT for GPU offload, see the GPU compilation section in
README.md:For the GPU implementation please be aware that it is based on a single MPI rank per GPU. 
README.md:Therefore, to test multiple GPUs, use the maximum number of available GPUs 
README.md:FFTW, Intel oneMKL, Nvidia cuFFT. 
README.md:integer, parameter, public :: D2D_FFT_BACKEND_CUFFT = 4     ! Nvidia cuFFT
INSTALL.md:- Nvidia NVHPC version 23.11 with CUDA version 12.3.52 and NCCL version 2.18.5 for GPU acceleration 
INSTALL.md:Two `BUILD_TARGETS` are available namely `mpi` and `gpu`.  For the `mpi` target no additional 
INSTALL.md:options should be required. whereas for `gpu` extra options are necessary at the configure stage. 
INSTALL.md:Please see section [GPU Compilation](#gpu-compilation)
INSTALL.md:## GPU compilation
INSTALL.md:The library can perform multi GPU offoloading using the NVHPC compiler suite for NVIDIA hardware. 
INSTALL.md:The implementation is based on CUDA-aware MPI and NVIDIA Collective Communication Library (NCCL).
INSTALL.md:To properly configure for GPU build the following needs to be used 
INSTALL.md:$ cmake -S $path_to_sources -B $path_to_build_directory -DBUILD_TARGET=gpu
INSTALL.md:Note, further configuration can be performed using `ccmake`, however the initial configuration of GPU builds must include the `-DBUILD_TARGET=gpu` flag as shown above.
INSTALL.md:By default CUDA aware MPI will be used together with `cuFFT` for the FFT library. The configure will automatically look for the GPU architecture available on the system. If you are building on a HPC system please use a computing node for the installation. Useful variables to be added are 
INSTALL.md: - `-DENABLE_NCCL=yes` to activate the NCCL collectives
INSTALL.md:-- The CUDA compiler identification is unknown  
INSTALL.md:CMake Error at /usr/share/cmake/Modules/CMakeDetermineCUDACompiler.cmake:633 (message):  
INSTALL.md:Failed to detect a default CUDA architecture. 
INSTALL.md: - `-DCMAKE_CUDA_HOST_COMPILER=$supported_gcc`
INSTALL.md: At the moment the supported CUDA host compilers are `gcc11` and earlier. 
INSTALL.md:In case of 2DECOMP&FFT compiled for GPU with NVHPC, linking against cuFFT is mandatory 
INSTALL.md:LIBS += -cudalib=cufft
INSTALL.md:In case of NCCL the following is required 
INSTALL.md:LIBS += -cudalib=cufft,nccl 
INSTALL.md:This preprocessor variable is not valid for GPU builds. It leads to padded alltoall operations. This preprocessor variable is driven by the CMake on/off variable `EVEN`.
INSTALL.md:#### _GPU
INSTALL.md:This variable is automatically added in GPU builds.
INSTALL.md:#### _NCCL
INSTALL.md:This variable is valid only for GPU builds. The NVIDIA Collective Communication Library (NCCL) implements multi-GPU and multi-node communication primitives optimized for NVIDIA GPUs and Networking.
CMakeLists.txt:set(BUILD_TARGET "mpi" CACHE STRING "Target for acceleration (mpi (default) or gpu)")
CMakeLists.txt:set_property(CACHE BUILD_TARGET PROPERTY STRINGS mpi gpu)
CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
CMakeLists.txt:  option(ENABLE_OPENACC "Allow user to activate/deactivate OpenACC support" ON)
CMakeLists.txt:  option(ENABLE_CUDA "Allow user to activate/deactivate CUDA support" ON)
CMakeLists.txt:  option(ENABLE_NCCL "Allow user to activate/deactivate Collective Comunication NCCL" OFF)
CMakeLists.txt:  if (ENABLE_CUDA)
CMakeLists.txt:    message(STATUS "Before enable CUDA")
CMakeLists.txt:    enable_language(CUDA)
CMakeLists.txt:    message(STATUS "After enable CUDA")
CMakeLists.txt:endif(BUILD_TARGET MATCHES "gpu")
CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
CMakeLists.txt:  include(D2D_GPU)
CMakeLists.txt:endif (BUILD_TARGET MATCHES "gpu")
CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
CMakeLists.txt:  set(FFT_Choice "cufft" CACHE STRING "FFT for XCompact3d project (with GPU cufft is the default)")
CMakeLists.txt:endif (BUILD_TARGET MATCHES "gpu")
cmake/compilers/D2D_flags_nvidia.cmake:#Compilers Flags for NVIDIA
cmake/compilers/D2D_flags_nvidia.cmake:if (BUILD_TARGET MATCHES "gpu")
cmake/compilers/D2D_flags_nvidia.cmake:  set(D2D_FFLAGS "${D2D_FFLAGS} -Minfo=accel -target=gpu")
cmake/compilers/D2D_flags_nvidia.cmake:  add_definitions("-D_GPU")
cmake/compilers/D2D_flags_nvidia.cmake:  if (ENABLE_OPENACC)
cmake/compilers/D2D_flags_nvidia.cmake:  if (ENABLE_CUDA)
cmake/compilers/D2D_flags_nvidia.cmake:    add_definitions("-DUSE_CUDA")
cmake/compilers/D2D_flags_nvidia.cmake:    set(D2D_FFLAGS "${D2D_FFLAGS} -cuda")
cmake/compilers/D2D_flags_nvidia.cmake:	    set(D2D_FFLAGS "${D2D_FFLAGS} -gpu=cc${CUDA_ARCH_COMP},managed,lineinfo")
cmake/compilers/D2D_flags_nvidia.cmake:	    set(D2D_FFLAGS "${D2D_FFLAGS} -gpu=cc${CUDA_ARCH_COMP},lineinfo")
cmake/compilers/D2D_flags_nvidia.cmake:    # Add NCCL cuFFT
cmake/compilers/D2D_flags_nvidia.cmake:    if (ENABLE_NCCL)
cmake/compilers/D2D_flags_nvidia.cmake:      add_definitions("-D_NCCL")
cmake/compilers/D2D_flags_nvidia.cmake:      set(D2D_FFLAGS "${D2D_FFLAGS} -cudalib=nccl,cufft")
cmake/compilers/D2D_flags_nvidia.cmake:    else(ENABLE_NCCL)
cmake/compilers/D2D_flags_nvidia.cmake:      set(D2D_FFLAGS "${D2D_FFLAGS} -cudalib=cufft")
cmake/compilers/D2D_flags_nvidia.cmake:    endif(ENABLE_NCCL)
cmake/compilers/D2D_flags_nvidia.cmake:  endif(ENABLE_CUDA)
cmake/compilers/D2D_flags_nvidia.cmake:endif (BUILD_TARGET MATCHES "gpu")
cmake/D2D_Compilers.cmake:  include(D2D_flags_nvidia)
cmake/D2D_Compilers.cmake:# Padded MPI alltoall transpose operations (invalid for GPU)
cmake/D2D_Compilers.cmake:  if (BUILD_TARGET MATCHES "gpu")
cmake/D2D_Compilers.cmake:    message(FATAL_ERROR "The GPU build is not compatible with padded alltoall")
cmake/fft/fft.cmake:  if (ENABLE_CUDA)
cmake/D2D_GPU.cmake:# GPU CMakeLists
cmake/D2D_GPU.cmake:message(STATUS "Check GPU")
cmake/D2D_GPU.cmake:if (ENABLE_OPENACC)
cmake/D2D_GPU.cmake:  include(FindOpenACC)
cmake/D2D_GPU.cmake:  if(OpenACC_Fortran_FOUND)
cmake/D2D_GPU.cmake:    message(STATUS "OpenACC for Fotran Compiler Found, version ${OpenACC_Fortran_VERSION_MAJOR}.${OpenACC_Fortran_VERSION_MINOR}")
cmake/D2D_GPU.cmake:    message(ERROR_CRITICAL "No OpenACC support detected")
cmake/D2D_GPU.cmake:if (ENABLE_CUDA)
cmake/D2D_GPU.cmake:  find_package(CUDAToolkit REQUIRED)
cmake/D2D_GPU.cmake:  if (NOT SET_CUDA_ARCH)
cmake/D2D_GPU.cmake:    set(SET_CUDA_ARCH 1 CACHE INTERNAL "Set CUDA Architecture" FORCE)
cmake/D2D_GPU.cmake:    set(CUDA_ARCH_TEST 70 )
cmake/D2D_GPU.cmake:      cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto") # optional argument for arch to add
cmake/D2D_GPU.cmake:      string(APPEND CMAKE_CUDA_FLAGS "${ARCH_FLAGS}")
cmake/D2D_GPU.cmake:      message(STATUS "ARCH_FLAGS WITH CUDA = ${ARCH_FLAGS}")
cmake/D2D_GPU.cmake:      include(FindCUDA/select_compute_arch)
cmake/D2D_GPU.cmake:      CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
cmake/D2D_GPU.cmake:      string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
cmake/D2D_GPU.cmake:      string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
cmake/D2D_GPU.cmake:      string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
cmake/D2D_GPU.cmake:      SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
cmake/D2D_GPU.cmake:      set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
cmake/D2D_GPU.cmake:      message(STATUS "CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}")
cmake/D2D_GPU.cmake:      list(GET CUDA_ARCH_LIST 0 CUDA_ARCH_AUTO)
cmake/D2D_GPU.cmake:      message(STATUS "CUDA_ARCH_AUTO ${CUDA_ARCH_AUTO}")
cmake/D2D_GPU.cmake:    if(${CUDA_ARCH_AUTO} GREATER ${CUDA_ARCH_TEST})
cmake/D2D_GPU.cmake:      set(CUDA_ARCH_COMP ${CUDA_ARCH_AUTO} CACHE STRING "Set CUDA Computing Architecture")
cmake/D2D_GPU.cmake:      set(CUDA_ARCH_COMP ${CUDA_ARCH_TEST} CACHE STRING "Set CUDA Computing Architecture")
cmake/D2D_GPU.cmake:    set(CUDA_ARCH_COMP ${SET_CUDA_ARCH})
cmake/D2D_GPU.cmake:  message(STATUS "CUDA_COMP ${CUDA_ARCH_COMP}")
examples/test2d/timing2d_real.f90:#if defined(_GPU)
examples/test2d/timing2d_real.f90:   use cudafor
examples/test2d/timing2d_real.f90:   use openacc
examples/test2d/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/test2d/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/test2d/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/test2d/test2d.f90:#if defined(_GPU)
examples/test2d/test2d.f90:   use cudafor
examples/test2d/test2d.f90:   use openacc
examples/test2d/timing2d_complex.f90:#if defined(_GPU)
examples/test2d/timing2d_complex.f90:   use cudafor
examples/test2d/timing2d_complex.f90:   use openacc
examples/test2d/bind.sh:export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
examples/test2d/bind.sh:echo "[LOG] local rank $LOCAL_RANK: bind to $CUDA_VISIBLE_DEVICES"
examples/fft_multiple_grids/fft_multiple_grids_no_obj.f90:#if defined(_GPU)
examples/fft_multiple_grids/fft_multiple_grids_no_obj.f90:   use cudafor
examples/fft_multiple_grids/fft_multiple_grids_no_obj.f90:   use openacc
examples/fft_multiple_grids/fft_multiple_grids_utilities.f90:#if defined(_GPU)
examples/fft_multiple_grids/fft_multiple_grids_utilities.f90:      use cudafor
examples/fft_multiple_grids/fft_multiple_grids_utilities.f90:      use openacc
examples/fft_multiple_grids/fft_multiple_grids_utilities.f90:#if defined(_GPU)
examples/fft_multiple_grids/fft_multiple_grids_utilities.f90:      ierror = cudaDeviceSynchronize()
examples/fft_multiple_grids/fft_multiple_grids_utilities.f90:#if defined(_GPU)
examples/fft_multiple_grids/fft_multiple_grids_utilities.f90:      use cudafor
examples/fft_multiple_grids/fft_multiple_grids_utilities.f90:      use openacc
examples/fft_multiple_grids/fft_multiple_grids_utilities.f90:#if defined(_GPU)
examples/fft_multiple_grids/fft_multiple_grids_utilities.f90:      ierror = cudaDeviceSynchronize()
examples/fft_multiple_grids/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/fft_multiple_grids/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/fft_multiple_grids/CMakeLists.txt:  if (BUILD_TARGET MATCHES "gpu")
examples/fft_multiple_grids/fft_multiple_grids.f90:#if defined(_GPU)
examples/fft_multiple_grids/fft_multiple_grids.f90:   use cudafor
examples/fft_multiple_grids/fft_multiple_grids.f90:   use openacc
examples/fft_multiple_grids/fft_multiple_grids_inplace.f90:#if defined(_GPU)
examples/fft_multiple_grids/fft_multiple_grids_inplace.f90:   use cudafor
examples/fft_multiple_grids/fft_multiple_grids_inplace.f90:   use openacc
examples/fft_multiple_grids/bind.sh:export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
examples/fft_multiple_grids/bind.sh:echo "[LOG] local rank $LOCAL_RANK: bind to $CUDA_VISIBLE_DEVICES"
examples/fft_physical_x/fft_grid_x.f90:#if defined(_GPU)
examples/fft_physical_x/fft_grid_x.f90:   use cudafor
examples/fft_physical_x/fft_grid_x.f90:   use openacc
examples/fft_physical_x/fft_grid_x.f90:   ! This is define loop on GPUs
examples/fft_physical_x/fft_grid_x.f90:#if defined(_GPU)
examples/fft_physical_x/fft_grid_x.f90:   ierror = cudaDeviceSynchronize()
examples/fft_physical_x/fft_c2c_x_skip.f90:#if defined(_GPU)
examples/fft_physical_x/fft_c2c_x_skip.f90:   use cudafor
examples/fft_physical_x/fft_c2c_x_skip.f90:   use openacc
examples/fft_physical_x/fft_c2c_x_skip.f90:#if defined(_GPU)
examples/fft_physical_x/fft_c2c_x_skip.f90:   ierror = cudaDeviceSynchronize()
examples/fft_physical_x/fft_r2c_x_skip.f90:#if defined(_GPU)
examples/fft_physical_x/fft_r2c_x_skip.f90:   use cudafor
examples/fft_physical_x/fft_r2c_x_skip.f90:   use openacc
examples/fft_physical_x/fft_r2c_x_skip.f90:#if defined(_GPU)
examples/fft_physical_x/fft_r2c_x_skip.f90:   ierror = cudaDeviceSynchronize()
examples/fft_physical_x/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/fft_physical_x/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/fft_physical_x/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/fft_physical_x/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/fft_physical_x/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/fft_physical_x/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/fft_physical_x/fft_c2c_x.f90:#if defined(_GPU)
examples/fft_physical_x/fft_c2c_x.f90:   use cudafor
examples/fft_physical_x/fft_c2c_x.f90:   use openacc
examples/fft_physical_x/fft_c2c_x.f90:#if defined(_GPU)
examples/fft_physical_x/fft_c2c_x.f90:   ierror = cudaDeviceSynchronize()
examples/fft_physical_x/bind.sh:export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
examples/fft_physical_x/bind.sh:echo "[LOG] local rank $LOCAL_RANK: bind to $CUDA_VISIBLE_DEVICES"
examples/fft_physical_x/fft_r2c_x_skip_errorMsg.f90:#if defined(_GPU)
examples/fft_physical_x/fft_r2c_x_skip_errorMsg.f90:   use cudafor
examples/fft_physical_x/fft_r2c_x_skip_errorMsg.f90:   use openacc
examples/fft_physical_x/fft_r2c_x_skip_errorMsg.f90:#if defined(_GPU)
examples/fft_physical_x/fft_r2c_x_skip_errorMsg.f90:   ierror = cudaDeviceSynchronize()
examples/fft_physical_x/fft_r2c_x.f90:#if defined(_GPU)
examples/fft_physical_x/fft_r2c_x.f90:   use cudafor
examples/fft_physical_x/fft_r2c_x.f90:   use openacc
examples/fft_physical_x/fft_r2c_x.f90:#if defined(_GPU)
examples/fft_physical_x/fft_r2c_x.f90:   ierror = cudaDeviceSynchronize()
examples/init_test/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/init_test/bind.sh:export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
examples/init_test/bind.sh:echo "[LOG] local rank $LOCAL_RANK: bind to $CUDA_VISIBLE_DEVICES"
examples/init_test/init_test.f90:#if defined(_GPU)
examples/init_test/init_test.f90:   use cudafor
examples/init_test/init_test.f90:   use openacc
examples/halo_test/halo_test.f90:#if defined(_GPU)
examples/halo_test/halo_test.f90:   use cudafor
examples/halo_test/halo_test.f90:   use openacc
examples/halo_test/halo_test.f90:#if defined(_GPU)
examples/halo_test/halo_test.f90:#if defined(_GPU)
examples/halo_test/halo_test.f90:#if defined(_GPU)
examples/halo_test/halo_test.f90:#if defined(_GPU)
examples/halo_test/halo_test.f90:#if defined(_GPU)
examples/halo_test/halo_test.f90:#if defined(_GPU)
examples/halo_test/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/halo_test/bind.sh:export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
examples/halo_test/bind.sh:echo "[LOG] local rank $LOCAL_RANK: bind to $CUDA_VISIBLE_DEVICES"
examples/io_mpi/io_read.f90:#if defined(_GPU)
examples/io_mpi/io_read.f90:   use cudafor
examples/io_mpi/io_read.f90:   use openacc
examples/io_mpi/io_var_test.f90:#if defined(_GPU)
examples/io_mpi/io_var_test.f90:   use cudafor
examples/io_mpi/io_var_test.f90:   use openacc
examples/io_mpi/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/io_mpi/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/io_mpi/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/io_mpi/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/io_mpi/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/io_mpi/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/io_mpi/io_test.f90:#if defined(_GPU)
examples/io_mpi/io_test.f90:   use cudafor
examples/io_mpi/io_test.f90:   use openacc
examples/io_mpi/io_mpi_non_blocking.f90:#if defined(_GPU)
examples/io_mpi/io_mpi_non_blocking.f90:   use cudafor
examples/io_mpi/io_mpi_non_blocking.f90:   use openacc
examples/io_mpi/io_plane_test.f90:!! FIXME The issue below is specific to GPU and should be discussed in a dedicated github issue
examples/io_mpi/io_plane_test.f90:!! NB in case of GPU only the writing in the aligned pencil (i.e. X for a 1 array) is performed.
examples/io_mpi/io_plane_test.f90:!! IO subrotines needs update for non managed GPU case
examples/io_mpi/io_plane_test.f90:#if defined(_GPU)
examples/io_mpi/io_plane_test.f90:   use cudafor
examples/io_mpi/io_plane_test.f90:   use openacc
examples/io_mpi/io_plane_test.f90:   ! For GPU we port the global data create the different pencil arrays
examples/io_mpi/bind.sh:export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
examples/io_mpi/bind.sh:echo "[LOG] local rank $LOCAL_RANK: bind to $CUDA_VISIBLE_DEVICES"
examples/io_mpi/io_bench.f90:#if defined(_GPU)
examples/io_mpi/io_bench.f90:   use cudafor
examples/io_mpi/io_bench.f90:   use openacc
examples/grad3d/grad3d.f90:#if defined(_GPU)
examples/grad3d/grad3d.f90:   use cudafor
examples/grad3d/grad3d.f90:   use openacc
examples/grad3d/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/grad3d/bind.sh:export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
examples/grad3d/bind.sh:echo "[LOG] local rank $LOCAL_RANK: bind to $CUDA_VISIBLE_DEVICES"
examples/io_adios/io_tmp_test.f90:#if defined(_GPU)
examples/io_adios/io_tmp_test.f90:   use cudafor
examples/io_adios/io_tmp_test.f90:   use openacc
examples/io_adios/io_read.f90:#if defined(_GPU)
examples/io_adios/io_read.f90:   use cudafor
examples/io_adios/io_read.f90:   use openacc
examples/io_adios/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/io_adios/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/io_adios/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/io_adios/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/io_adios/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/io_adios/io_test.f90:#if defined(_GPU)
examples/io_adios/io_test.f90:   use cudafor
examples/io_adios/io_test.f90:   use openacc
examples/io_adios/io_plane_test.f90:!! FIXME The issue below is specific to GPU and should be discussed in a dedicated github issue
examples/io_adios/io_plane_test.f90:!! NB in case of GPU only the writing in the aligned pencil (i.e. X for a 1 array) is performed.
examples/io_adios/io_plane_test.f90:!! IO subrotines needs update for non managed GPU case
examples/io_adios/io_plane_test.f90:#if defined(_GPU)
examples/io_adios/io_plane_test.f90:   use cudafor
examples/io_adios/io_plane_test.f90:   use openacc
examples/io_adios/io_plane_test.f90:   ! For GPU we port the global data create the different pencil arrays
examples/io_adios/bind.sh:export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
examples/io_adios/bind.sh:echo "[LOG] local rank $LOCAL_RANK: bind to $CUDA_VISIBLE_DEVICES"
examples/io_adios/io_bench.f90:#if defined(_GPU)
examples/io_adios/io_bench.f90:   use cudafor
examples/io_adios/io_bench.f90:   use openacc
examples/fft_physical_z/fft_r2c_z_skip.f90:#if defined(_GPU)
examples/fft_physical_z/fft_r2c_z_skip.f90:   use cudafor
examples/fft_physical_z/fft_r2c_z_skip.f90:   use openacc
examples/fft_physical_z/fft_r2c_z_skip.f90:#if defined(_GPU)
examples/fft_physical_z/fft_r2c_z_skip.f90:   ierror = cudaDeviceSynchronize()
examples/fft_physical_z/fft_c2c_z.f90:#if defined(_GPU)
examples/fft_physical_z/fft_c2c_z.f90:   use cudafor
examples/fft_physical_z/fft_c2c_z.f90:   use openacc
examples/fft_physical_z/fft_c2c_z.f90:#if defined(_GPU)
examples/fft_physical_z/fft_c2c_z.f90:   ierror = cudaDeviceSynchronize()
examples/fft_physical_z/fft_c2c_z_skip.f90:#if defined(_GPU)
examples/fft_physical_z/fft_c2c_z_skip.f90:   use cudafor
examples/fft_physical_z/fft_c2c_z_skip.f90:   use openacc
examples/fft_physical_z/fft_c2c_z_skip.f90:#if defined(_GPU)
examples/fft_physical_z/fft_c2c_z_skip.f90:   ierror = cudaDeviceSynchronize()
examples/fft_physical_z/fft_r2c_z_skip_errorMsg.f90:#if defined(_GPU)
examples/fft_physical_z/fft_r2c_z_skip_errorMsg.f90:   use cudafor
examples/fft_physical_z/fft_r2c_z_skip_errorMsg.f90:   use openacc
examples/fft_physical_z/fft_r2c_z_skip_errorMsg.f90:#if defined(_GPU)
examples/fft_physical_z/fft_r2c_z_skip_errorMsg.f90:   ierror = cudaDeviceSynchronize()
examples/fft_physical_z/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/fft_physical_z/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/fft_physical_z/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/fft_physical_z/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/fft_physical_z/CMakeLists.txt:if (BUILD_TARGET MATCHES "gpu")
examples/fft_physical_z/bind.sh:export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
examples/fft_physical_z/bind.sh:echo "[LOG] local rank $LOCAL_RANK: bind to $CUDA_VISIBLE_DEVICES"
examples/fft_physical_z/fft_r2c_z.f90:#if defined(_GPU)
examples/fft_physical_z/fft_r2c_z.f90:   use cudafor
examples/fft_physical_z/fft_r2c_z.f90:   use openacc
examples/fft_physical_z/fft_r2c_z.f90:#if defined(_GPU)
examples/fft_physical_z/fft_r2c_z.f90:   ierror = cudaDeviceSynchronize()
src/halo.f90:#if defined(_GPU)
src/halo.f90:#if defined(_GPU)
src/halo.f90:#if defined(_GPU)
src/halo.f90:#if defined(_GPU)
src/decomp_2d_init_fin.f90:#if defined(_GPU) && defined(_NCCL)
src/decomp_2d_init_fin.f90:     call decomp_2d_nccl_init(DECOMP_2D_COMM_COL, DECOMP_2D_COMM_ROW)
src/decomp_2d_init_fin.f90:#if defined(_GPU)
src/decomp_2d_init_fin.f90:#if defined(_NCCL)
src/decomp_2d_init_fin.f90:     call decomp_2d_nccl_mem_fin()
src/decomp_2d_init_fin.f90:     call decomp_2d_nccl_fin()
src/decomp_2d_constants.f90:#if defined(_GPU) && defined(_NCCL)
src/decomp_2d_constants.f90:   use nccl
src/log.f90:#ifdef _GPU
src/log.f90:      write (io_unit, *) 'Compile flag _GPU detected'
src/log.f90:#ifdef _NCCL
src/log.f90:      write (io_unit, *) 'Compile flag _NCCL detected'
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:         istat = cudaMemcpy(dst, src, nsize, cudaMemcpyDeviceToDevice)
src/transpose_z_to_y.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy")
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:      istat = cudaMemcpy(wk1, src, s1 * s2 * s3, cudaMemcpyDeviceToDevice)
src/transpose_z_to_y.f90:      if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/transpose_z_to_y.f90:#elif defined(_NCCL)
src/transpose_z_to_y.f90:      call decomp_2d_nccl_send_recv_row(wk2, &
src/transpose_z_to_y.f90:#elif defined(_GPU)
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:         istat = cudaMemcpy(dst, src, nsize, cudaMemcpyDeviceToDevice)
src/transpose_z_to_y.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy")
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:      istat = cudaMemcpy(wk1, src, s1 * s2 * s3, cudaMemcpyDeviceToDevice)
src/transpose_z_to_y.f90:      if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/transpose_z_to_y.f90:#elif defined(_NCCL)
src/transpose_z_to_y.f90:      call decomp_2d_nccl_send_recv_row(wk2, &
src/transpose_z_to_y.f90:#elif defined(_GPU)
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:         istat = cudaMemcpy2D(out(1, i1, 1), n1 * n2, in(pos), n1 * (i2 - i1 + 1), n1 * (i2 - i1 + 1), n3, cudaMemcpyDeviceToDevice)
src/transpose_z_to_y.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:#if defined(_GPU)
src/transpose_z_to_y.f90:         istat = cudaMemcpy2D(out(1, i1, 1), n1 * n2, in(pos), n1 * (i2 - i1 + 1), n1 * (i2 - i1 + 1), n3, cudaMemcpyDeviceToDevice)
src/transpose_z_to_y.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/halo_common.f90:       !!! istat = cudaMemcpy(out,in,s1*s2*s3,cudaMemcpyDeviceToDevice)
src/CMakeLists.txt:if(${BUILD_TARGET} MATCHES "gpu")
src/CMakeLists.txt:  if(ENABLE_NCCL)
src/CMakeLists.txt:    list(APPEND files_decomp decomp_2d_nccl.f90)
src/CMakeLists.txt:endif(${BUILD_TARGET} MATCHES "gpu")
src/decomp_2d_cumpi.f90:! Module for the cuda aware MPI
src/decomp_2d_cumpi.f90:   ! Real/complex pointers to GPU buffers
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:         istat = cudaMemcpy(dst, src, nsize, cudaMemcpyDeviceToDevice)
src/transpose_x_to_y.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy")
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:#elif defined(_NCCL)
src/transpose_x_to_y.f90:      call decomp_2d_nccl_send_recv_col(wk2, &
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:         istat = cudaMemcpy(dst, src, nsize, cudaMemcpyDeviceToDevice)
src/transpose_x_to_y.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy")
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:#elif defined(_NCCL)
src/transpose_x_to_y.f90:      call decomp_2d_nccl_send_recv_col(wk2, &
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:         istat = cudaMemcpy2D(out(pos), i2 - i1 + 1, in(i1, 1, 1), n1, i2 - i1 + 1, n2 * n3, cudaMemcpyDeviceToDevice)
src/transpose_x_to_y.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:         istat = cudaMemcpy2D(out(pos), i2 - i1 + 1, in(i1, 1, 1), n1, i2 - i1 + 1, n2 * n3, cudaMemcpyDeviceToDevice)
src/transpose_x_to_y.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:         istat = cudaMemcpy2D(out(1, i1, 1), n1 * n2, in(pos), n1 * (i2 - i1 + 1), n1 * (i2 - i1 + 1), n3, cudaMemcpyDeviceToDevice)
src/transpose_x_to_y.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:#if defined(_GPU)
src/transpose_x_to_y.f90:         istat = cudaMemcpy2D(out(1, i1, 1), n1 * n2, in(pos), n1 * (i2 - i1 + 1), n1 * (i2 - i1 + 1), n3, cudaMemcpyDeviceToDevice)
src/transpose_x_to_y.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/fft_cufft.f90:   use cudafor
src/decomp_2d_mpi.f90:#if defined(_GPU) && defined(_NCCL)
src/decomp_2d_mpi.f90:   use nccl
src/decomp_2d_mpi.f90:#if defined(_GPU) && defined(_NCCL)
src/decomp_2d_mpi.f90:      module procedure decomp_2d_abort_nccl_basic
src/decomp_2d_mpi.f90:      module procedure decomp_2d_abort_nccl_file_line
src/decomp_2d_mpi.f90:#if defined(_GPU) && defined(_NCCL)
src/decomp_2d_mpi.f90:   ! This is based on the file "nccl.h" in nvhpc 22.1
src/decomp_2d_mpi.f90:   function _ncclresult_to_integer(errorcode)
src/decomp_2d_mpi.f90:      type(ncclresult), intent(IN) :: errorcode
src/decomp_2d_mpi.f90:      integer :: _ncclresult_to_integer
src/decomp_2d_mpi.f90:      if (errorcode == ncclSuccess) then
src/decomp_2d_mpi.f90:         _ncclresult_to_integer = 0
src/decomp_2d_mpi.f90:      elseif (errorcode == ncclUnhandledCudaError) then
src/decomp_2d_mpi.f90:         _ncclresult_to_integer = 1
src/decomp_2d_mpi.f90:      elseif (errorcode == ncclSystemError) then
src/decomp_2d_mpi.f90:         _ncclresult_to_integer = 2
src/decomp_2d_mpi.f90:      elseif (errorcode == ncclInternalError) then
src/decomp_2d_mpi.f90:         _ncclresult_to_integer = 3
src/decomp_2d_mpi.f90:      elseif (errorcode == ncclInvalidArgument) then
src/decomp_2d_mpi.f90:         _ncclresult_to_integer = 4
src/decomp_2d_mpi.f90:      elseif (errorcode == ncclInvalidUsage) then
src/decomp_2d_mpi.f90:         _ncclresult_to_integer = 5
src/decomp_2d_mpi.f90:      elseif (errorcode == ncclNumResults) then
src/decomp_2d_mpi.f90:         _ncclresult_to_integer = 6
src/decomp_2d_mpi.f90:         _ncclresult_to_integer = -1
src/decomp_2d_mpi.f90:         call decomp_2d_warning(__FILE__, __LINE__, _ncclresult_to_integer, &
src/decomp_2d_mpi.f90:                                "NCCL error handling needs some update")
src/decomp_2d_mpi.f90:   end function _ncclresult_to_integer
src/decomp_2d_mpi.f90:   ! Small wrapper for basic NCCL errors
src/decomp_2d_mpi.f90:   subroutine decomp_2d_abort_nccl_basic(errorcode, msg)
src/decomp_2d_mpi.f90:      type(ncclresult), intent(IN) :: errorcode
src/decomp_2d_mpi.f90:      call decomp_2d_abort(_ncclresult_to_integer(errorcode), &
src/decomp_2d_mpi.f90:                           msg//" "//ncclGetErrorString(errorcode))
src/decomp_2d_mpi.f90:   end subroutine decomp_2d_abort_nccl_basic
src/decomp_2d_mpi.f90:   ! Small wrapper for NCCL errors
src/decomp_2d_mpi.f90:   subroutine decomp_2d_abort_nccl_file_line(file, line, errorcode, msg)
src/decomp_2d_mpi.f90:      type(ncclresult), intent(IN) :: errorcode
src/decomp_2d_mpi.f90:                           _ncclresult_to_integer(errorcode), &
src/decomp_2d_mpi.f90:                           msg//" "//ncclGetErrorString(errorcode))
src/decomp_2d_mpi.f90:   end subroutine decomp_2d_abort_nccl_file_line
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:         istat = cudaMemcpy(dst, src, nsize, cudaMemcpyDeviceToDevice)
src/transpose_y_to_z.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy")
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:#elif defined(_NCCL)
src/transpose_y_to_z.f90:      call decomp_2d_nccl_send_recv_row(wk2, &
src/transpose_y_to_z.f90:#elif defined(_GPU)
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:      !If one of the array in cuda call is not device we need to add acc host_data
src/transpose_y_to_z.f90:      istat = cudaMemcpy(dst, wk2, d1 * d2 * d3, cudaMemcpyDeviceToDevice)
src/transpose_y_to_z.f90:      if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:         istat = cudaMemcpy(dst, src, nsize, cudaMemcpyDeviceToDevice)
src/transpose_y_to_z.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy")
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:#elif defined(_NCCL)
src/transpose_y_to_z.f90:      call decomp_2d_nccl_send_recv_row(wk2, &
src/transpose_y_to_z.f90:#elif defined(_GPU)
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:      istat = cudaMemcpy(dst, wk2, d1 * d2 * d3, cudaMemcpyDeviceToDevice)
src/transpose_y_to_z.f90:      if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:         istat = cudaMemcpy2D(out(pos), n1 * (i2 - i1 + 1), in(1, i1, 1), n1 * n2, n1 * (i2 - i1 + 1), n3, cudaMemcpyDeviceToDevice)
src/transpose_y_to_z.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:#if defined(_GPU)
src/transpose_y_to_z.f90:         istat = cudaMemcpy2D(out(pos), n1 * (i2 - i1 + 1), in(1, i1, 1), n1 * n2, n1 * (i2 - i1 + 1), n3, cudaMemcpyDeviceToDevice)
src/transpose_y_to_z.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/decomp_2d_nccl.f90:! Module for the cuda aware MPI
src/decomp_2d_nccl.f90:module decomp_2d_nccl
src/decomp_2d_nccl.f90:   use nccl
src/decomp_2d_nccl.f90:   type(ncclDataType), parameter, public :: ncclType = ncclDouble
src/decomp_2d_nccl.f90:   type(ncclDataType), parameter, public :: ncclType = ncclFloat
src/decomp_2d_nccl.f90:   type(ncclUniqueId), save, public :: nccl_uid_2decomp
src/decomp_2d_nccl.f90:   type(ncclComm), save, public :: nccl_comm_2decomp
src/decomp_2d_nccl.f90:   integer(kind=cuda_stream_kind), save, public :: cuda_stream_2decomp
src/decomp_2d_nccl.f90:   ! Extra pointers for nccl complex transpose
src/decomp_2d_nccl.f90:   public :: decomp_2d_nccl_init, &
src/decomp_2d_nccl.f90:             decomp_2d_nccl_fin, &
src/decomp_2d_nccl.f90:             decomp_2d_nccl_mem_init, &
src/decomp_2d_nccl.f90:             decomp_2d_nccl_mem_fin, &
src/decomp_2d_nccl.f90:             decomp_2d_nccl_send_recv_col, &
src/decomp_2d_nccl.f90:             decomp_2d_nccl_send_recv_row
src/decomp_2d_nccl.f90:   interface decomp_2d_nccl_send_recv_col
src/decomp_2d_nccl.f90:      module procedure decomp_2d_nccl_send_recv_real_col
src/decomp_2d_nccl.f90:      module procedure decomp_2d_nccl_send_recv_cmplx_col
src/decomp_2d_nccl.f90:   end interface decomp_2d_nccl_send_recv_col
src/decomp_2d_nccl.f90:   interface decomp_2d_nccl_send_recv_row
src/decomp_2d_nccl.f90:      module procedure decomp_2d_nccl_send_recv_real_row
src/decomp_2d_nccl.f90:      module procedure decomp_2d_nccl_send_recv_cmplx_row
src/decomp_2d_nccl.f90:   end interface decomp_2d_nccl_send_recv_row
src/decomp_2d_nccl.f90:   subroutine decomp_2d_nccl_init(COMM_COL, COMM_ROW)
src/decomp_2d_nccl.f90:      integer :: cuda_stat
src/decomp_2d_nccl.f90:      type(ncclResult) :: nccl_stat
src/decomp_2d_nccl.f90:         nccl_stat = ncclGetUniqueId(nccl_uid_2decomp)
src/decomp_2d_nccl.f90:         if (nccl_stat /= ncclSuccess) call decomp_2d_abort(__FILE__, __LINE__, nccl_stat, "ncclGetUniqueId")
src/decomp_2d_nccl.f90:      call MPI_Bcast(nccl_uid_2decomp, int(sizeof(ncclUniqueId)), MPI_BYTE, 0, decomp_2d_comm, ierror)
src/decomp_2d_nccl.f90:      nccl_stat = ncclCommInitRank(nccl_comm_2decomp, nproc, nccl_uid_2decomp, nrank)
src/decomp_2d_nccl.f90:      if (nccl_stat /= ncclSuccess) call decomp_2d_abort(__FILE__, __LINE__, nccl_stat, "ncclCommInitRank")
src/decomp_2d_nccl.f90:      cuda_stat = cudaStreamCreate(cuda_stream_2decomp)
src/decomp_2d_nccl.f90:      if (cuda_stat /= 0) call decomp_2d_abort(__FILE__, __LINE__, ierror, "cudaStreamCreate")
src/decomp_2d_nccl.f90:   end subroutine decomp_2d_nccl_init
src/decomp_2d_nccl.f90:   ! Finalize the module (release nccl communicator)
src/decomp_2d_nccl.f90:   subroutine decomp_2d_nccl_fin()
src/decomp_2d_nccl.f90:      integer :: cuda_stat
src/decomp_2d_nccl.f90:      type(ncclResult) :: nccl_stat
src/decomp_2d_nccl.f90:      nccl_stat = ncclCommDestroy(nccl_comm_2decomp)
src/decomp_2d_nccl.f90:      if (nccl_stat /= ncclSuccess) call decomp_2d_abort(__FILE__, __LINE__, nccl_stat, "ncclCommDestroy")
src/decomp_2d_nccl.f90:      cuda_stat = cudaStreamDestroy(cuda_stream_2decomp)
src/decomp_2d_nccl.f90:      if (cuda_stat /= 0) call decomp_2d_abort(__FILE__, __LINE__, cuda_stat, "cudaStreamDestroy")
src/decomp_2d_nccl.f90:   end subroutine decomp_2d_nccl_fin
src/decomp_2d_nccl.f90:   subroutine decomp_2d_nccl_mem_init(buf_size)
src/decomp_2d_nccl.f90:   end subroutine decomp_2d_nccl_mem_init
src/decomp_2d_nccl.f90:   subroutine decomp_2d_nccl_mem_fin
src/decomp_2d_nccl.f90:   end subroutine decomp_2d_nccl_mem_fin
src/decomp_2d_nccl.f90:   subroutine decomp_2d_nccl_send_recv_real_col(dst_d, &
src/decomp_2d_nccl.f90:      integer :: col_rank_id, cuda_stat
src/decomp_2d_nccl.f90:      type(ncclResult) :: nccl_stat
src/decomp_2d_nccl.f90:      nccl_stat = ncclGroupStart()
src/decomp_2d_nccl.f90:      if (nccl_stat /= ncclSuccess) call decomp_2d_abort(__FILE__, __LINE__, nccl_stat, "ncclGroupStart")
src/decomp_2d_nccl.f90:         nccl_stat = ncclSend(src_d(disp_s(col_rank_id) + 1), cnts_s(col_rank_id), &
src/decomp_2d_nccl.f90:                              ncclType, local_to_global_col(col_rank_id + 1), nccl_comm_2decomp, cuda_stream_2decomp)
src/decomp_2d_nccl.f90:         if (nccl_stat /= ncclSuccess) call decomp_2d_abort(__FILE__, __LINE__, nccl_stat, "ncclSend")
src/decomp_2d_nccl.f90:         nccl_stat = ncclRecv(dst_d(disp_r(col_rank_id) + 1), cnts_r(col_rank_id), &
src/decomp_2d_nccl.f90:                              ncclType, local_to_global_col(col_rank_id + 1), nccl_comm_2decomp, cuda_stream_2decomp)
src/decomp_2d_nccl.f90:         if (nccl_stat /= ncclSuccess) call decomp_2d_abort(__FILE__, __LINE__, nccl_stat, "ncclRecv")
src/decomp_2d_nccl.f90:      nccl_stat = ncclGroupEnd()
src/decomp_2d_nccl.f90:      if (nccl_stat /= ncclSuccess) call decomp_2d_abort(__FILE__, __LINE__, nccl_stat, "ncclGroupEnd")
src/decomp_2d_nccl.f90:      cuda_stat = cudaStreamSynchronize(cuda_stream_2decomp)
src/decomp_2d_nccl.f90:      if (cuda_stat /= 0) call decomp_2d_abort(__FILE__, __LINE__, cuda_stat, "cudaStreamSynchronize")
src/decomp_2d_nccl.f90:   end subroutine decomp_2d_nccl_send_recv_real_col
src/decomp_2d_nccl.f90:   subroutine decomp_2d_nccl_send_recv_cmplx_col(dst_d, &
src/decomp_2d_nccl.f90:      call decomp_2d_nccl_send_recv_col(work4_r_d, &
src/decomp_2d_nccl.f90:      call decomp_2d_nccl_send_recv_col(work4_r_d, &
src/decomp_2d_nccl.f90:   end subroutine decomp_2d_nccl_send_recv_cmplx_col
src/decomp_2d_nccl.f90:   subroutine decomp_2d_nccl_send_recv_real_row(dst_d, &
src/decomp_2d_nccl.f90:      integer :: row_rank_id, cuda_stat
src/decomp_2d_nccl.f90:      type(ncclResult) :: nccl_stat
src/decomp_2d_nccl.f90:      nccl_stat = ncclGroupStart()
src/decomp_2d_nccl.f90:      if (nccl_stat /= ncclSuccess) call decomp_2d_abort(__FILE__, __LINE__, nccl_stat, "ncclGroupStart")
src/decomp_2d_nccl.f90:         nccl_stat = ncclSend(src_d(disp_s(row_rank_id) + 1), cnts_s(row_rank_id), &
src/decomp_2d_nccl.f90:                              ncclType, local_to_global_row(row_rank_id + 1), nccl_comm_2decomp, cuda_stream_2decomp)
src/decomp_2d_nccl.f90:         if (nccl_stat /= ncclSuccess) call decomp_2d_abort(__FILE__, __LINE__, nccl_stat, "ncclSend")
src/decomp_2d_nccl.f90:         nccl_stat = ncclRecv(dst_d(disp_r(row_rank_id) + 1), cnts_r(row_rank_id), &
src/decomp_2d_nccl.f90:                              ncclType, local_to_global_row(row_rank_id + 1), nccl_comm_2decomp, cuda_stream_2decomp)
src/decomp_2d_nccl.f90:         if (nccl_stat /= ncclSuccess) call decomp_2d_abort(__FILE__, __LINE__, nccl_stat, "ncclRecv")
src/decomp_2d_nccl.f90:      nccl_stat = ncclGroupEnd()
src/decomp_2d_nccl.f90:      if (nccl_stat /= ncclSuccess) call decomp_2d_abort(__FILE__, __LINE__, nccl_stat, "ncclGroupEnd")
src/decomp_2d_nccl.f90:      cuda_stat = cudaStreamSynchronize(cuda_stream_2decomp)
src/decomp_2d_nccl.f90:      if (cuda_stat /= 0) call decomp_2d_abort(__FILE__, __LINE__, cuda_stat, "cudaStreamSynchronize")
src/decomp_2d_nccl.f90:   end subroutine decomp_2d_nccl_send_recv_real_row
src/decomp_2d_nccl.f90:   subroutine decomp_2d_nccl_send_recv_cmplx_row(dst_d, &
src/decomp_2d_nccl.f90:      call decomp_2d_nccl_send_recv_row(work4_r_d, &
src/decomp_2d_nccl.f90:      call decomp_2d_nccl_send_recv_row(work4_r_d, &
src/decomp_2d_nccl.f90:   end subroutine decomp_2d_nccl_send_recv_cmplx_row
src/decomp_2d_nccl.f90:end module decomp_2d_nccl
src/decomp_2d.f90:#if defined(_GPU)
src/decomp_2d.f90:   use cudafor
src/decomp_2d.f90:#if defined(_NCCL)
src/decomp_2d.f90:   use nccl
src/decomp_2d.f90:   use decomp_2d_nccl
src/decomp_2d.f90:#if defined(_GPU)
src/decomp_2d.f90:#if defined(_GPU)
src/decomp_2d.f90:#if defined(_NCCL)
src/decomp_2d.f90:         call decomp_2d_nccl_mem_init(buf_size)
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:         istat = cudaMemcpy(dst, src, nsize, cudaMemcpyDeviceToDevice)
src/transpose_y_to_x.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy")
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:#elif defined(_NCCL)
src/transpose_y_to_x.f90:      call decomp_2d_nccl_send_recv_col(wk2, &
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:         istat = cudaMemcpy(dst, src, nsize, cudaMemcpyDeviceToDevice)
src/transpose_y_to_x.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy")
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:#elif defined(_NCCL)
src/transpose_y_to_x.f90:      call decomp_2d_nccl_send_recv_col(wk2, &
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:         istat = cudaMemcpy2D(out(pos), n1 * (i2 - i1 + 1), in(1, i1, 1), n1 * n2, n1 * (i2 - i1 + 1), n3, cudaMemcpyDeviceToDevice)
src/transpose_y_to_x.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:         istat = cudaMemcpy2D(out(pos), n1 * (i2 - i1 + 1), in(1, i1, 1), n1 * n2, n1 * (i2 - i1 + 1), n3, cudaMemcpyDeviceToDevice)
src/transpose_y_to_x.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:         istat = cudaMemcpy2D(out(i1, 1, 1), n1, in(pos), i2 - i1 + 1, i2 - i1 + 1, n2 * n3, cudaMemcpyDeviceToDevice)
src/transpose_y_to_x.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:#if defined(_GPU)
src/transpose_y_to_x.f90:         istat = cudaMemcpy2D(out(i1, 1, 1), n1, in(pos), i2 - i1 + 1, i2 - i1 + 1, n2 * n3, cudaMemcpyDeviceToDevice)
src/transpose_y_to_x.f90:         if (istat /= 0) call decomp_2d_abort(__FILE__, __LINE__, istat, "cudaMemcpy2D")

```
