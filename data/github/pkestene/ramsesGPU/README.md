# https://github.com/pkestene/ramsesGPU

```console
configure.ac:AC_INIT([RAMSES-GPU], [1.3.0], [pierre.kestener@cea.fr])
configure.ac:# Detects CUDA
configure.ac:CONFIGURE_HEADLINE([ CUDA support ])
configure.ac:AX_CUDA
configure.ac:# Nvcc flags setup (remember that default NVCCFLAGS are set by AX_CUDA macro)
configure.ac:	test/mpiCuda/Makefile
configure.ac:	* GPU/CUDA enabled       : $want_cuda
configure.ac:	* CUDA tools version     : $NVCC_VERSION (NVCCFLAGS = $NVCCFLAGS)
ChangeLog:	* both CPU and GPU implementation are slightly improved: each of
ChangeLog:	* update CUDA kernel to use read-only data cache (const restrict)
ChangeLog:	* major bug fix: in both CPU / GPU version of the MHD numerical
ChangeLog:	* major bug fix: in the CUDA version of the code, when running the MRI problem, total mass inside box was slightly increasing with time. The bug was fixed in the kernel_mhd_flux_update_hydro_v4_shear CUDA kernel.
ChangeLog:	* bug fix: this bug only affects MPI/CUDA version for the RMI problem (wrong index in remapping formula in shear border condition). This bug was causing the simulation run to stop after several 1000's of time steps
ChangeLog:	* bug fix in viscosity tensor: affect both CPU/GPU: only one term was slightly false (almost neutral bug)
ChangeLog:	* new feature: now GPU memory array can be allocated using pitched or linear memory (before pitched was the default). On new hardware, pitch memory is not anymore a must. Linear memory gives the same performane (thanks to cache improvements). Besides, for a given problem, the amouint of required memory is almost twice smaller when using linear memory, so linear memory is now the default choice.
ChangeLog:	* Version 1.0; package name is now RAMSES-GPU with a dedicated website :
ChangeLog:	http://www.maisondelasimulation.fr/projects/RAMSES-GPU/html/
ChangeLog:	* Version 0.4.0 is the first version with 3D MHD on CPU/GPU and
ChangeLog:	* SoftwareName : As 2D MHD solver has been implemented on GPU, let's take the
ChangeLog:	now only 2D on both CPU / GPU has been tested).
data/mhd_mri_3d_mpi_debug.ini:# This is a small MPI setup intended to test/debug CPU+MPI or GPU+MPI
data/mhd_mri_3d_mpi_debug.ini:# It can be run on a single GPU machine (if it is a Fermi, one can have
data/mhd_mri_3d_mpi_debug.ini:# multiple MPI process accessing the same GPU).
data/rayleigh_taylor_gpu_3d_mhd.ini:# In the GPU version, one must particularly care of numerical parameters like
data/rayleigh_taylor_gpu_3d_mhd.ini:outputPrefix=rayleigh_taylor_gpu_3d_mhd
data/jet3d_mhd_mpi_gpu.ini:# In the GPU version, one must particularly care of numerical parameters like
data/jet3d_mhd_mpi_gpu.ini:outputPrefix=jet3d_mhd_mpi_gpu
data/getpot/jet2d_mpi_gpu.pot:outputPrefix='jet2d_gpu'
data/getpot/jet3d_mpi_gpu.pot:outputPrefix='jet3d_gpu'
data/kelvin_helmholtz_gpu_3d.ini:# In the GPU version, one must particularly care of numerical parameters like
data/kelvin_helmholtz_gpu_3d.ini:outputPrefix=kelvin_helmholtz_gpu_3d
data/implode3d_zslab.ini:outputPrefix=implode3d_gpu_zslab
data/kelvin_helmholtz_cpu_2d_mpi.ini:# In the GPU version, one must particularly care of numerical parameters like
data/mhd_BrioWu.ini:# In the GPU version, one must particularly care of numerical parameters like
data/rayleigh_taylor_cpu_2d_mhd.ini:# In the GPU version, one must particularly care of numerical parameters like
data/orszag-tang.ini:# In the GPU version, one must particularly care of numerical parameters like
data/kelvin_helmholtz_gpu_2d.ini:# In the GPU version, one must particularly care of numerical parameters like
data/kelvin_helmholtz_gpu_2d.ini:outputPrefix=kelvin_helmholtz_gpu_2d
data/testRiemannHLLD.ini:# In the GPU version, one must particularly care of numerical parameters like
data/kelvin_helmholtz_gpu_2d_mhd.ini:# In the GPU version, one must particularly care of numerical parameters like
data/kelvin_helmholtz_gpu_2d_mhd.ini:outputPrefix=kelvin_helmholtz_gpu_2d_mhd
data/falling_bubble_gpu_2d.ini:# In the GPU version, one must particularly care of numerical parameters like
data/falling_bubble_gpu_2d.ini:outputPrefix=falling_bubble_gpu_2d
data/rayleigh_taylor_gpu_3d_mpi_mhd.ini:# In the GPU version, one must particularly care of numerical parameters like
data/rayleigh_taylor_gpu_3d_mpi_mhd.ini:outputPrefix=rayleigh_taylor_gpu_3d_mpi_mhd
data/rayleigh_taylor_2d_mpi_mhd.ini:# In the GPU version, one must particularly care of numerical parameters like
data/orszag-tang3d.ini:# In the GPU version, one must particularly care of numerical parameters like
data/orszag-tang_mpi.ini:# In the GPU version, one must particularly care of numerical parameters like
data/jet2d_gpu.ini:# In the GPU version, one must particularly care of numerical parameters like
data/jet2d_gpu.ini:outputPrefix=jet2d_gpu
data/rayleigh_taylor_gpu_2d.ini:# In the GPU version, one must particularly care of numerical parameters like
data/rayleigh_taylor_gpu_2d.ini:outputPrefix=rayleigh_taylor_gpu_2d
data/jet3d_gpu.ini:# In the GPU version, one must particularly care of numerical parameters like
data/jet3d_gpu.ini:outputPrefix=jet3d_gpu
data/kelvin_helmholtz_gpu_2d_large.ini:# In the GPU version, one must particularly care of numerical parameters like
data/kelvin_helmholtz_gpu_2d_large.ini:outputPrefix=kelvin_helmholtz_gpu_2d_large
data/kelvin_helmholtz_gpu_2d_mhd_mpi.ini:# In the GPU version, one must particularly care of numerical parameters like
data/kelvin_helmholtz_gpu_2d_mhd_mpi.ini:outputPrefix=kelvin_helmholtz_gpu_2d_mhd_mpi
data/rayleigh_taylor_gpu_3d.ini:# In the GPU version, one must particularly care of numerical parameters like
data/rayleigh_taylor_gpu_3d.ini:outputPrefix=rayleigh_taylor_gpu_3d
data/jet2d_mpi_gpu.ini:# Hydro 2D example parameter file; jet simulation in MPI+GPU environment
data/jet2d_mpi_gpu.ini:# In the GPU version, one must particularly care of numerical parameters like
data/jet2d_mpi_gpu.ini:outputPrefix=jet2d_mpi_gpu
data/jet3d_mpi_gpu.ini:# In the GPU version, one must particularly care of numerical parameters like
data/jet3d_mpi_gpu.ini:outputPrefix=jet3d_mpi_gpu
data/implode3d_debug.ini:outputPrefix=implode3d_gpu_v2
test/mpiHydro/CMakeLists.txt:  RamsesGPU::hydro)
test/mpiHydro/CMakeLists.txt:  RamsesGPU::hydro)
test/mpiHydro/CMakeLists.txt:    RamsesGPU::config
test/mpiHydro/CMakeLists.txt:    RamsesGPU::hydro
test/mpiHydro/CMakeLists.txt:    RamsesGPU::mpiUtils)
test/mpiHydro/CMakeLists.txt:    RamsesGPU::config
test/mpiHydro/CMakeLists.txt:    RamsesGPU::hydro
test/mpiHydro/CMakeLists.txt:    RamsesGPU::mpiUtils)
test/mpiHydro/CMakeLists.txt:  if (USE_CUDA)
test/mpiHydro/CMakeLists.txt:    set_source_files_properties(testMpiOutputVtk.cpp PROPERTIES LANGUAGE CUDA)
test/mpiHydro/CMakeLists.txt:  endif(USE_CUDA)
test/mpiHydro/CMakeLists.txt:    RamsesGPU::hydro
test/mpiHydro/CMakeLists.txt:    RamsesGPU::config
test/mpiHydro/CMakeLists.txt:    RamsesGPU::cnpy
test/mpiHydro/CMakeLists.txt:    RamsesGPU::monitoring
test/mpiHydro/CMakeLists.txt:    RamsesGPU::mpiUtils)
test/mpiHydro/CMakeLists.txt:  if (NOT USE_CUDA)
test/mpiHydro/CMakeLists.txt:      RamsesGPU::hydro
test/mpiHydro/CMakeLists.txt:      RamsesGPU::config
test/mpiHydro/CMakeLists.txt:      RamsesGPU::cnpy
test/mpiHydro/CMakeLists.txt:      RamsesGPU::monitoring
test/mpiHydro/CMakeLists.txt:      RamsesGPU::mpiUtils)
test/mpiHydro/CMakeLists.txt:  endif(NOT USE_CUDA)
test/mpiCuda/testHelloMpiCuda.cu: * \file testHelloMpiCuda.cu
test/mpiCuda/testHelloMpiCuda.cu: * \brief A simple program to test MPI+Cuda
test/mpiCuda/testHelloMpiCuda.cu:// CUDA-C includes
test/mpiCuda/testHelloMpiCuda.cu:#include <cuda_runtime_api.h>
test/mpiCuda/testHelloMpiCuda.cu:  // initialize cuda
test/mpiCuda/testHelloMpiCuda.cu:  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
test/mpiCuda/testHelloMpiCuda.cu:    printf("cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
test/mpiCuda/testHelloMpiCuda.cu:  // This function call returns 0 if there are no CUDA capable devices.
test/mpiCuda/testHelloMpiCuda.cu:    printf("There is no device supporting CUDA\n");
test/mpiCuda/testHelloMpiCuda.cu:  // grab information about current GPU device
test/mpiCuda/testHelloMpiCuda.cu:  cudaDeviceProp deviceProp;
test/mpiCuda/testHelloMpiCuda.cu:  cudaSetDevice(myMpiRank%4);
test/mpiCuda/testHelloMpiCuda.cu:  cudaGetDevice(&deviceId);
test/mpiCuda/testHelloMpiCuda.cu:  cudaGetDeviceProperties(&deviceProp, deviceId);
test/mpiCuda/testHelloMpiCuda.cu:      printf("There is no device supporting CUDA.\n");
test/mpiCuda/testHelloMpiCuda.cu:      printf("There is 1 device supporting CUDA associated with MPI of rank 0\n");
test/mpiCuda/testHelloMpiCuda.cu:      printf("There are %d devices supporting CUDA\n", deviceCount);
test/mpiCuda/testHelloMpiCuda.cu:#if CUDART_VERSION >= 2020
test/mpiCuda/testHelloMpiCuda.cu:    cudaDriverGetVersion(&driverVersion);
test/mpiCuda/testHelloMpiCuda.cu:    printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
test/mpiCuda/testHelloMpiCuda.cu:    cudaRuntimeGetVersion(&runtimeVersion);
test/mpiCuda/testHelloMpiCuda.cu:    printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
test/mpiCuda/testHelloMpiCuda.cu:    printf("  CUDA Capability Major revision number:         %d\n", deviceProp.major);
test/mpiCuda/testHelloMpiCuda.cu:    printf("  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);
test/mpiCuda/testHelloMpiCuda.cu:  printf("Working GPU device Id is %d\n",deviceId);
test/mpiCuda/testBorderBufferCuda.cu: * \file testBorderBufferCuda.cpp
test/mpiCuda/CMakeLists.txt:if(USE_CUDA)
test/mpiCuda/CMakeLists.txt:  add_executable(testHelloMpiCuda "")
test/mpiCuda/CMakeLists.txt:  target_sources(testHelloMpiCuda 
test/mpiCuda/CMakeLists.txt:    testHelloMpiCuda.cu)
test/mpiCuda/CMakeLists.txt:  target_link_libraries(testHelloMpiCuda PUBLIC RamsesGPU::mpiUtils)
test/mpiCuda/CMakeLists.txt:  #cmake_cuda_convert_flags(INTERFACE_TARGET testHelloMpiCuda)
test/mpiCuda/CMakeLists.txt:  add_executable(testBorderBufferCuda "")
test/mpiCuda/CMakeLists.txt:  target_sources(testBorderBufferCuda 
test/mpiCuda/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/src/hydro/gpu_macros.cpp
test/mpiCuda/CMakeLists.txt:    testBorderBufferCuda.cu)
test/mpiCuda/CMakeLists.txt:  set_source_files_properties(${CMAKE_SOURCE_DIR}/src/hydro/gpu_macros.cpp PROPERTIES LANGUAGE CUDA)
test/mpiCuda/CMakeLists.txt:  target_include_directories(testBorderBufferCuda
test/mpiCuda/CMakeLists.txt:  target_link_libraries(testBorderBufferCuda PUBLIC RamsesGPU::mpiUtils)
test/mpiCuda/CMakeLists.txt:  add_executable(testBorderBufferCuda2 "")
test/mpiCuda/CMakeLists.txt:  target_sources(testBorderBufferCuda2
test/mpiCuda/CMakeLists.txt:    ${CMAKE_SOURCE_DIR}/src/hydro/gpu_macros.cpp
test/mpiCuda/CMakeLists.txt:    testBorderBufferCuda2.cu)
test/mpiCuda/CMakeLists.txt:  #set_source_files_properties(${CMAKE_SOURCE_DIR}/src/hydro/gpu_macros.cpp PROPERTIES LANGUAGE CUDA)
test/mpiCuda/CMakeLists.txt:  target_include_directories(testBorderBufferCuda2
test/mpiCuda/CMakeLists.txt:  target_link_libraries(testBorderBufferCuda2 PUBLIC RamsesGPU::mpiUtils)
test/mpiCuda/CMakeLists.txt:endif(USE_CUDA)
test/mpiCuda/Makefile.am:# also note that SUFFIXES must be set with .cu before including cuda.am
test/mpiCuda/Makefile.am:include $(top_srcdir)/am/cuda.am
test/mpiCuda/Makefile.am:AM_CXXFLAGS = $(CUDA_CFLAGS)
test/mpiCuda/Makefile.am:AM_CFLAGS   = $(CUDA_CFLAGS)
test/mpiCuda/Makefile.am:bin_PROGRAMS  = testHelloMpiCuda testBorderBufferCuda testBorderBufferCuda2
test/mpiCuda/Makefile.am:# testHelloMpiCuda
test/mpiCuda/Makefile.am:testHelloMpiCuda_SOURCES  = testHelloMpiCuda.cu
test/mpiCuda/Makefile.am:nodist_EXTRA_testHelloMpiCuda_SOURCES = dummy.cxx
test/mpiCuda/Makefile.am:testHelloMpiCuda_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CXXFLAGS) -I$(top_srcdir)/src/utils/mpiUtils
test/mpiCuda/Makefile.am:testHelloMpiCuda_LDFLAGS  = $(AM_LDFLAGS)  $(CUDA_LIBS) ../../src/utils/mpiUtils/libMpiUtils.la $(MPI_LDFLAGS) 
test/mpiCuda/Makefile.am:# testBorderBufferCuda
test/mpiCuda/Makefile.am:testBorderBufferCuda_SOURCES = testBorderBufferCuda.cu
test/mpiCuda/Makefile.am:nodist_EXTRA_testBorderBufferCuda_SOURCES = dummy.cxx
test/mpiCuda/Makefile.am:testBorderBufferCuda_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CXXFLAGS) -I$(top_srcdir)/src/hydro -I$(top_srcdir)/src/utils/mpiUtils
test/mpiCuda/Makefile.am:testBorderBufferCuda_LDFLAGS  = $(AM_LDFLAGS)  $(MPI_LDFLAGS) $(CUDA_LIBS) ../../src/hydro/libhydroGpu.la ../../src/utils/mpiUtils/libMpiUtils.la
test/mpiCuda/Makefile.am:# testBorderBufferCuda2
test/mpiCuda/Makefile.am:testBorderBufferCuda2_SOURCES = testBorderBufferCuda2.cu
test/mpiCuda/Makefile.am:nodist_EXTRA_testBorderBufferCuda2_SOURCES = dummy.cxx
test/mpiCuda/Makefile.am:testBorderBufferCuda2_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CXXFLAGS) -I$(top_srcdir)/src/hydro -I$(top_srcdir)/src/utils/mpiUtils
test/mpiCuda/Makefile.am:testBorderBufferCuda2_LDFLAGS  = $(AM_LDFLAGS)  $(MPI_LDFLAGS) $(CUDA_LIBS) ../../src/hydro/libhydroGpu.la ../../src/utils/mpiUtils/libMpiUtils.la
test/mpiCuda/testBorderBufferCuda2.cu: * \file testBorderBufferCuda2.cpp
test/mpiCuda/testBorderBufferCuda2.cu:#include "../../src/utils/monitoring/CudaTimer.h"
test/mpiCuda/testBorderBufferCuda2.cu:    CudaTimer timer;
test/mpiCuda/testBorderBufferCuda2.cu:      // direct test of cudaMalloc
test/mpiCuda/testBorderBufferCuda2.cu:      std::cout << "Direct test of cudaMalloc, cudaMemcpy and cudaMemcpyAsync" << std::endl;
test/mpiCuda/testBorderBufferCuda2.cu:      cudaMalloc( (void **) &d_data , sizeInBytes);
test/mpiCuda/testBorderBufferCuda2.cu:      cudaMallocHost((void**)&h_dataPinned, sizeInBytes);
test/mpiCuda/testBorderBufferCuda2.cu:	cudaMemcpy((void *) d_data, h_data, sizeInBytes, cudaMemcpyHostToDevice);
test/mpiCuda/testBorderBufferCuda2.cu:	cudaMemcpy((void *) h_data, d_data, sizeInBytes, cudaMemcpyDeviceToHost);
test/mpiCuda/testBorderBufferCuda2.cu:	cudaMemcpyAsync((void *) d_data, h_dataPinned, sizeInBytes, cudaMemcpyHostToDevice);
test/mpiCuda/testBorderBufferCuda2.cu:	cudaMemcpyAsync((void *) h_dataPinned, d_data, sizeInBytes, cudaMemcpyDeviceToHost);
test/mpiCuda/testBorderBufferCuda2.cu:      cudaFree(d_data);
test/computeL2relatif.py.in:    gpuFile=open(sys.argv[2])
test/computeL2relatif.py.in:    header=gpuFile.readline()
test/computeL2relatif.py.in:        gpuData = np.fromfile(file=gpuFile, dtype=np.float32).reshape((ny,nx))
test/computeL2relatif.py.in:        gpuData = np.fromfile(file=gpuFile, dtype=np.float64).reshape((ny,nx))
test/computeL2relatif.py.in:diffData=cpuData-gpuData
test/mpiBasic/CMakeLists.txt:  RamsesGPU::mpiUtils)
test/mpiBasic/CMakeLists.txt:  RamsesGPU::mpiUtils)
test/CMakeLists.txt:  add_subdirectory(mpiCuda)
test/CMakeLists.txt:  if (RAMSESGPU_FFTW3_FOUND)
test/CMakeLists.txt:  endif(RAMSESGPU_FFTW3_FOUND)
test/test_run.sh.in:# \brief Shell script to automate performance comparison study between CPU and GPU.
test/test_run.sh.in:    gpuExe=@CMAKE_BINARY_DIR@/src/euler_gpu
test/test_run.sh.in:    gpuExe="@CMAKE_BINARY_DIR@/src/euler_gpu --scheme $1"
test/test_run.sh.in:# generic function to run simulation; argument can be either cpuExe or gpuExe
test/test_run.sh.in:mkdir -p $basedir/$runDir/cpu $basedir/$runDir/gpu
test/test_run.sh.in:    #gpuCmd=`printf "/usr/bin/time -f \"%d %%e\" -o results.txt -a %s --param ./conf.ini" $nx $gpuExe`
test/test_run.sh.in:    # run GPU version
test/test_run.sh.in:    echo "GPU : `basename $gpuExe` $nx $ny"
test/test_run.sh.in:    cd $basedir/$runDir/gpu
test/test_run.sh.in:    runSimul "$gpuExe"
test/test_run.sh.in:    echo "compare CPU/GPU results: relative L2 norm"
test/test_run.sh.in:	gpuFile=./$runDir/gpu/$baseFile
test/test_run.sh.in:	python computeL2relatif.py $cpuFile $gpuFile
test/test_run.sh.in:	rm $cpuFile $gpuFile
test/test_run.sh.in:echo "                           and in $runDir/gpu/results.txt"
test/test_run.sh.in:echo "python ./plotCpuGpuComparaison.py"
test/testPoisson/testPoissonGpuCuFFT2d.cu: * ./testPoissonGpuCuFFT2d --nx 64 --ny 64 --method 1 --test 2
test/testPoisson/testPoissonGpuCuFFT2d.cu:#include <cuda_runtime.h>
test/testPoisson/testPoissonGpuCuFFT2d.cu:  // gpu variables
test/testPoisson/testPoissonGpuCuFFT2d.cu:  cudaMalloc((void**) &d_rho, NX*NY2*sizeof(FFTW_REAL));
test/testPoisson/testPoissonGpuCuFFT2d.cu:  if (cudaGetLastError() != cudaSuccess){
test/testPoisson/testPoissonGpuCuFFT2d.cu:    fprintf(stderr, "Cuda error: Failed to allocate \"d_rho\"\n");
test/testPoisson/testPoissonGpuCuFFT2d.cu:  // copy rho onto gpu
test/testPoisson/testPoissonGpuCuFFT2d.cu:  cudaMemcpy(d_rho, rho, sizeof(FFTW_REAL)*NX*NY2, cudaMemcpyHostToDevice);
test/testPoisson/testPoissonGpuCuFFT2d.cu:  // (GPU) apply poisson kernel 
test/testPoisson/testPoissonGpuCuFFT2d.cu:  // retrieve gpu computation
test/testPoisson/testPoissonGpuCuFFT2d.cu:  cudaMemcpy(rho, d_rho, sizeof(FFTW_REAL)*NX*NY2, cudaMemcpyDeviceToHost);
test/testPoisson/testPoissonGpuCuFFT2d.cu:  cudaFree(d_rho);
test/testPoisson/testPoissonGpuCuFFT3d.cu: * ./testPoissonGpuCuFFT3d --nx 64 --ny 64 --nz 64 --method 1 --test 2
test/testPoisson/testPoissonGpuCuFFT3d.cu:#include <cuda_runtime.h>
test/testPoisson/testPoissonGpuCuFFT3d.cu:  // gpu variables
test/testPoisson/testPoissonGpuCuFFT3d.cu:  cudaMalloc((void**) &d_rho, NX*NY*NZ2*sizeof(FFTW_REAL));
test/testPoisson/testPoissonGpuCuFFT3d.cu:  if (cudaGetLastError() != cudaSuccess){
test/testPoisson/testPoissonGpuCuFFT3d.cu:    fprintf(stderr, "Cuda error: Failed to allocate \"d_rho\"\n");
test/testPoisson/testPoissonGpuCuFFT3d.cu:  // copy rho onto gpu
test/testPoisson/testPoissonGpuCuFFT3d.cu:  cudaMemcpy(d_rho, rho, sizeof(FFTW_REAL)*NX*NY*NZ2, cudaMemcpyHostToDevice);
test/testPoisson/testPoissonGpuCuFFT3d.cu:  // (GPU) apply poisson kernel 
test/testPoisson/testPoissonGpuCuFFT3d.cu:  // retrieve gpu computation
test/testPoisson/testPoissonGpuCuFFT3d.cu:  cudaMemcpy(rho, d_rho, sizeof(FFTW_REAL)*NX*NY*NZ2, cudaMemcpyDeviceToHost);
test/testPoisson/testPoissonGpuCuFFT3d.cu:  cudaFree(d_rho);
test/testPoisson/CMakeLists.txt:  RamsesGPU::cnpy
test/testPoisson/CMakeLists.txt:  RamsesGPU::fftw)
test/testPoisson/CMakeLists.txt:  RamsesGPU::cnpy
test/testPoisson/CMakeLists.txt:  RamsesGPU::fftw)
test/testPoisson/CMakeLists.txt:if(USE_CUDA)
test/testPoisson/CMakeLists.txt:  # linking to cuda libs with modern cmake
test/testPoisson/CMakeLists.txt:  # TODO : analyze why target CUDAlibs::fftw, CUDAlibs::fft are not ok
test/testPoisson/CMakeLists.txt:  add_executable(testPoissonGpuCuFFT2d "")
test/testPoisson/CMakeLists.txt:  target_sources(testPoissonGpuCuFFT2d
test/testPoisson/CMakeLists.txt:    testPoissonGpuCuFFT2d.cu)
test/testPoisson/CMakeLists.txt:  target_include_directories(testPoissonGpuCuFFT2d
test/testPoisson/CMakeLists.txt:  target_link_libraries(testPoissonGpuCuFFT2d
test/testPoisson/CMakeLists.txt:    RamsesGPU::cnpy
test/testPoisson/CMakeLists.txt:    RamsesGPU::fftw
test/testPoisson/CMakeLists.txt:    CUDA::cufft
test/testPoisson/CMakeLists.txt:  add_executable(testPoissonGpuCuFFT3d "")
test/testPoisson/CMakeLists.txt:  target_sources(testPoissonGpuCuFFT3d
test/testPoisson/CMakeLists.txt:    testPoissonGpuCuFFT3d.cu)
test/testPoisson/CMakeLists.txt:  target_include_directories(testPoissonGpuCuFFT3d
test/testPoisson/CMakeLists.txt:  target_link_libraries(testPoissonGpuCuFFT3d
test/testPoisson/CMakeLists.txt:    RamsesGPU::cnpy
test/testPoisson/CMakeLists.txt:    RamsesGPU::fftw
test/testPoisson/CMakeLists.txt:    CUDA::cufft
test/testPoisson/CMakeLists.txt:endif(USE_CUDA)
test/testPoisson/Makefile.am:# also note that SUFFIXES must be set with .cu before including cuda.am
test/testPoisson/Makefile.am:include $(top_srcdir)/am/cuda.am
test/testPoisson/Makefile.am:AM_CXXFLAGS = $(CUDA_CFLAGS)
test/testPoisson/Makefile.am:AM_CFLAGS   = $(CUDA_CFLAGS)
test/testPoisson/Makefile.am:if USE_CUDA
test/testPoisson/Makefile.am:bin_PROGRAMS += testPoissonGpuCuFFT2d testPoissonGpuCuFFT3d
test/testPoisson/Makefile.am:bin_PROGRAMS += testPoissonGpuCuFFT2d_double testPoissonGpuCuFFT3d_double
test/testPoisson/Makefile.am:## GPU
test/testPoisson/Makefile.am:if USE_CUDA
test/testPoisson/Makefile.am:# testPoissonGpuCuFFT2d
test/testPoisson/Makefile.am:testPoissonGpuCuFFT2d_SOURCES  = testPoissonGpuCuFFT2d.cu
test/testPoisson/Makefile.am:nodist_EXTRA_testPoissonGpuCuFFT2d_SOURCES = dummy.cpp
test/testPoisson/Makefile.am:testPoissonGpuCuFFT2d_CPPFLAGS = -DUSE_FLOAT $(CUDA_CFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
test/testPoisson/Makefile.am:testPoissonGpuCuFFT2d_LDFLAGS  = $(CUDA_LIBS) -lcufftw -lcufft
test/testPoisson/Makefile.am:testPoissonGpuCuFFT2d_LDADD    = ../../src/utils/cnpy/libCNpy.la
test/testPoisson/Makefile.am:# testPoissonGpuCuFFT3d
test/testPoisson/Makefile.am:testPoissonGpuCuFFT3d_SOURCES  = testPoissonGpuCuFFT3d.cu
test/testPoisson/Makefile.am:nodist_EXTRA_testPoissonGpuCuFFT3d_SOURCES = dummy.cpp
test/testPoisson/Makefile.am:testPoissonGpuCuFFT3d_CPPFLAGS = -DUSE_FLOAT $(CUDA_CFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
test/testPoisson/Makefile.am:testPoissonGpuCuFFT3d_LDFLAGS  = $(CUDA_LIBS) -lcufftw -lcufft
test/testPoisson/Makefile.am:testPoissonGpuCuFFT3d_LDADD    = ../../src/utils/cnpy/libCNpy.la
test/testPoisson/Makefile.am:# testPoissonGpuCuFFT2d_double
test/testPoisson/Makefile.am:testPoissonGpuCuFFT2d_double_SOURCES  = testPoissonGpuCuFFT2d.cu
test/testPoisson/Makefile.am:nodist_EXTRA_testPoissonGpuCuFFT2d_double_SOURCES = dummy.cpp
test/testPoisson/Makefile.am:testPoissonGpuCuFFT2d_double_CPPFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
test/testPoisson/Makefile.am:testPoissonGpuCuFFT2d_double_LDFLAGS  = $(CUDA_LIBS) -lcufftw -lcufft
test/testPoisson/Makefile.am:testPoissonGpuCuFFT2d_double_LDADD    = ../../src/utils/cnpy/libCNpy.la
test/testPoisson/Makefile.am:# testPoissonGpuCuFFT3d_double
test/testPoisson/Makefile.am:testPoissonGpuCuFFT3d_double_SOURCES  = testPoissonGpuCuFFT3d.cu
test/testPoisson/Makefile.am:nodist_EXTRA_testPoissonGpuCuFFT3d_double_SOURCES = dummy.cpp
test/testPoisson/Makefile.am:testPoissonGpuCuFFT3d_double_CPPFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
test/testPoisson/Makefile.am:testPoissonGpuCuFFT3d_double_LDFLAGS  = $(CUDA_LIBS) -lcufftw -lcufft
test/testPoisson/Makefile.am:testPoissonGpuCuFFT3d_double_LDADD    = ../../src/utils/cnpy/libCNpy.la
test/Makefile.am:if USE_CUDA
test/Makefile.am:SUBDIRS += mpiCuda
test/Makefile.am:AM_CFLAGS = $(CUDA_CFLAGS)
doc/main.h: * RAMSES-GPU is a software package providing a C++/Cuda
doc/main.h: * (CPU or GPU + MPI) and also MHD with shearing box.
doc/main.h: * RAMSES-GPU is developped by Maison de la Simulation (http://www.maisondelasimulation.fr) and CEA/Sap (http://irfu.cea.fr/Sap/en/index.php)
doc/main.h: * RAMSES-GPU is governed by the CeCILL  license http://www.cecill.info
doc/main.h: * RAMSES-GPU sources can be downloaded at http://www.maisondelasimulation.fr/projects/RAMSES-GPU/html/download.html
doc/main.h: * <tr><td>Numerical Scheme</td> <td>CPU</td> <td>GPU</td> <td>CPU+MPI</td> <td>GPU+MPI</td> </tr>
doc/main.h: * computation GPU implementations (0 means no trace, 1 or 2 means
doc/main.h: * This software is a computer program whose purpose is to provide GPU implementations of some finite volume numerical schemes used to perform hydrodynamics and MHD flow simulations.
doc/main.h: * \verbatim svn co https://dsm-trac.cea.fr/svn/coast/gpu_tryouts/HybridHydro/trunk trunk \endverbatim
doc/main.h: * You need a relatively decent/recent CUDA-capable graphics board
doc/main.h: * href="http://en.wikipedia.org/wiki/Comparison_of_Nvidia_graphics_processing_units">list
doc/main.h: * of CUDA-capable device</a>) with hardware capability 1.3.
doc/main.h: * You also need to have the Nvidia toolkit (NVCC compiler) installed
doc/main.h: * to build the GPU version of the code. If it is not found, only the
doc/main.h: configure --with-cuda=/usr/local/cuda30 \endcode
doc/main.h: * executables (\e euler_cpu , \e euler_gpu) in the src sub directory
doc/main.h: * \li \c example \c of \c 2d \c run \c on \c GPU: Type \code ./euler_gpu --param jet.ini \endcode
doc/main.h: * \li \c example \c of \c 3d \c run \c on \c GPU: \code ./euler_gpu --param jet3d.ini \endcode
doc/main.h: * \code euler2d_gpu_qt --param jet.ini\endcode 
doc/main.h: * - \c modules: cuda/4.1, phdf5
doc/main.h: * - \c configure \c line \c at \c CCRT (with GPU, MPI and double precision enabled):
doc/main.h: *   - NVCCFLAGS="-gencode=arch=compute_20,code=sm_20 " ../trunk/configure --disable-shared --with-cuda=/usr/local/cuda-4.1 --with-boost-mpi=no --disable-qtgui --enable-mpi --enable-timing --enable-double CC=icc CXX=icpc
doc/main.h: * - \c other parameters: you can env variable MAX_REG_COUNT_SINGLE or MAX_REG_COUNT_DOUBLE to increase the maximun cuda register at compile time
doc/main.h: * - \c submission script for a GPU+MPI job with 8 MPI processes (each of which accessing 1 GPU):
doc/main.h: *     #MSUB -r MRI_MPI_GPU               # Request name 
doc/main.h: *     #MSUB -o mri3d_gpu_mpi_%I.out      # Standard output. %I is the job id 
doc/main.h: *     #MSUB -e mri3d_gpu_mpi_%I.err      # Error output. %I is the job id 
doc/main.h: *     #MSUB -q hybrid                    # Hybrid partition of GPU nodes
doc/main.h: *     module load cuda/4.1
doc/main.h: *     ccc_mprun ./euler_gpu_mpi --param ./mhd_mri_3d_gpu_mpi.ini
doc/main.h: *    ccc_msub job_multiGPU.sh
doc/main.h: *    - module add cuda/3.2
doc/main.h: *   - ../build_trunk/configure --disable-shared --with-cuda=/applications/cuda-3.2/ --with-boost-mpi=no --disable-qtgui --enable-mpi --enable-timing CC=icc CXX=icpc
doc/main.h: *     #MSUB -N 2                     # reservation de N noeuds (2 devices GPU par noeud)
doc/main.h: * <!-- \li couple this code with D. Aubert and R. Teyssier CUDA-based
doc/main.h: * \li improve GPU version : use CUDA streams to overlap memory transfert and 
README.md:![mhd_mri 200x200](https://github.com/pkestene/ramsesGPU/blob/master/doc/mhd_mri_3d_gpu_Pm4_Re25000_double.gif)
README.md:[Magneto Rotational Instability](https://en.wikipedia.org/wiki/Magnetorotational_instability) simulation in a shearing box setup (800x1600x800) made in 2013 on [TGCC/CURIE](http://www-hpc.cea.fr/fr/complexe/tgcc-curie.htm) using 256 GPUs. Here [Reynolds number](https://en.wikipedia.org/wiki/Reynolds_number) is 25000 and [Prandtl number](https://en.wikipedia.org/wiki/Prandtl_number) is 4.
README.md:# RamsesGPU code
README.md:## RamsesGPU website
README.md:http://www.maisondelasimulation.fr/projects/RAMSES-GPU/html/index.html
README.md:- Quickstart for building RAMSES-GPU using CMake (recommended)
README.md:Default CUDA compilation flags can be passed to cmake using env variable CUDAFLAGS, or directly set CMAKE_CUDA_FLAGS on the configuration command line (see below).
README.md:0. git clone https://github.com/pkestene/ramsesGPU.git
README.md:1. cd ramsesGPU; mkdir build
README.md:2. cmake -DUSE_CUDA=ON -DUSE_MPI=ON -DCMAKE_CUDA_FLAGS="-arch=sm_50" ..
README.md:You should get executable *ramsesGPU_mpi_cuda*. Explore other flag using the ccmake user interface.
README.md:- Quickstart for building RAMSES-GPU using autotools (deprecated)
README.md:1. configure --with-cuda=<path to CUDA toolkit root directory> 
README.md:Note: make sure to have CUDA toolkit installed, and environment variables PATH and LD_LIBRARY_PATH correctly set.
README.md:This will build the monoCPU / monoGPU version of the programme to solve hydro/MHD problems. Executable are located in src subdirectory and named euler_cpu / euler_gpu
README.md:	./euler_gpu --param ../data/jet2d_gpu.ini
README.md:	paraview --data=./jet2d_gpu.xmf
memory_footprint.py:# per GPU (1 GPU per MPI process)
AUTHORS:  original CUDA implementation of the 2D Godunov scheme from Fortran
AUTHORS:  update the Godunov scheme to 3D (both CPU and GPU): September 2010
AUTHORS:  implementation of a CUDA+MPI version: October 2010
CMakeLists.txt:project(RamsesGPU LANGUAGES CXX C)
CMakeLists.txt:list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda")
CMakeLists.txt:option (USE_CUDA "build with CUDA support" OFF)
CMakeLists.txt:option (USE_MPI_CUDA_AWARE_ENFORCED "Some MPI cuda-aware implementation are not well detected; use this to enforce" OFF)
CMakeLists.txt:if (USE_CUDA)
CMakeLists.txt:  enable_language(CUDA)
CMakeLists.txt:  message("Using CUDAToolkit macros")
CMakeLists.txt:  find_package(CUDAToolkit REQUIRED)
CMakeLists.txt:  if(NOT DEFINED CMAKE_CUDA_STANDARD)
CMakeLists.txt:    set(CMAKE_CUDA_STANDARD 11)
CMakeLists.txt:    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
CMakeLists.txt:    # if CUDA is enabled we need to prevent flag "-pthread" to be passed
CMakeLists.txt:    # When this is consumed for compiling CUDA, use '-Xcompiler' to wrap '-pthread'.
CMakeLists.txt:    ###string(REPLACE "-pthread" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >-pthread"
CMakeLists.txt:    string(REPLACE "-pthread" $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler >-pthread  _MPI_CXX_COMPILE_OPTIONS "${MPI_CXX_COMPILE_OPTIONS}")
CMakeLists.txt:    # Full command line to probe if cuda support in MPI implementation is enabled
CMakeLists.txt:    # ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
CMakeLists.txt:      if ( (_output MATCHES "smcuda") OR (USE_MPI_CUDA_AWARE_ENFORCED) )
CMakeLists.txt:        message(STATUS "Found OpenMPI with CUDA support built.")
CMakeLists.txt:        message(WARNING "OpenMPI found, but it is not built with CUDA support.")
CMakeLists.txt:        add_compile_options(-DMPI_CUDA_AWARE_OFF)
CMakeLists.txt:    pkg_check_modules(RAMSESGPU_PNETCDF QUIET IMPORTED_TARGET pnetcdf)
CMakeLists.txt:    if(RAMSESGPU_PNETCDF_FOUND)
CMakeLists.txt:      add_library(RamsesGPU::pnetcdf ALIAS PkgConfig::RAMSESGPU_PNETCDF)
CMakeLists.txt:pkg_check_modules(RAMSESGPU_PAPI QUIET IMPORTED_TARGET papi)
CMakeLists.txt:if(RAMSESGPU_PAPI_FOUND)
CMakeLists.txt:    HINTS ${RAMSESGPU_PAPI_INCLUDE_DIRS})
CMakeLists.txt:  set(CMAKE_REQUIRED_INCLUDES ${RAMSESGPU_PAPI_INCLUDEDIR})
CMakeLists.txt:  add_library(RamsesGPU::papi ALIAS PkgConfig::RAMSESGPU_PAPI)
CMakeLists.txt:pkg_check_modules(RAMSESGPU_FFTW3 QUIET IMPORTED_TARGET fftw3 fftw3f)
CMakeLists.txt:if(RAMSESGPU_FFTW3_FOUND)
CMakeLists.txt:  add_library(RamsesGPU::fftw ALIAS PkgConfig::RAMSESGPU_FFTW3)
CMakeLists.txt:if (USE_CUDA)
CMakeLists.txt:  message("  CUDA compiler ID      : ${CMAKE_CUDA_COMPILER_ID}")
CMakeLists.txt:  message("  CUDA compiler Version : ${CMAKE_CUDA_COMPILER_VERSION}")
CMakeLists.txt:  message("  CUDA Compiler         : ${CMAKE_CUDA_COMPILER}")
CMakeLists.txt:  message("  CUDA Compiler exec    : ${CUDA_NVCC_EXECUTABLE}")
CMakeLists.txt:  message("  CUDA Compile flags    : ${CMAKE_CUDA_FLAGS}")
CMakeLists.txt:  message("  CUDA toolkit found    : ${CUDAToolkit_FOUND}")
CMakeLists.txt:  message("  CUDA toolkit version  : ${CUDAToolkit_VERSION}")
CMakeLists.txt:  message("  CUDA toolkit nvcc     : ${CUDAToolkit_NVCC_EXECUTABLE}")
CMakeLists.txt:else(USE_CUDA)
CMakeLists.txt:  message("  CUDA not enabled")
CMakeLists.txt:endif(USE_CUDA)
CMakeLists.txt:if (RAMSESGPU_PNETCDF_FOUND)
CMakeLists.txt:  message("  PNETCDF found version : ${RAMSESGPU_PNETCDF_VERSION}")
CMakeLists.txt:  message("  PNETCDF include dirs  : ${RAMSESGPU_PNETCDF_CFLAGS}")
CMakeLists.txt:  message("  PNETCDF libraries     : ${RAMSESGPU_PNETCDF_LDFLAGS}")
CMakeLists.txt:endif(RAMSESGPU_PNETCDF_FOUND)
m4/ax_cuda.m4:# ax_cuda.m4: An m4 macro to detect and configure Cuda
m4/ax_cuda.m4:#	AX_CUDA()
m4/ax_cuda.m4:#	Checks the existence of Cuda binaries and libraries.
m4/ax_cuda.m4:#	--with-cuda=(path|yes|no)
m4/ax_cuda.m4:#		Indicates whether to use Cuda or not, and the path of a non-standard
m4/ax_cuda.m4:#		installation location of Cuda if necessary.
m4/ax_cuda.m4:#		AC_SUBST(CUDA_CFLAGS)
m4/ax_cuda.m4:#		AC_SUBST(CUDA_LIBS)
m4/ax_cuda.m4:AC_DEFUN([AX_CUDA],
m4/ax_cuda.m4:AC_ARG_WITH([cuda],
m4/ax_cuda.m4:    AS_HELP_STRING([--with-cuda@<:@=yes|no|DIR@:>@], [prefix where cuda is installed (default=yes)]),
m4/ax_cuda.m4:	with_cuda=$withval
m4/ax_cuda.m4:		want_cuda="no"
m4/ax_cuda.m4:		want_cuda="yes"
m4/ax_cuda.m4:		want_cuda="yes"
m4/ax_cuda.m4:		cuda_home_path=$withval
m4/ax_cuda.m4:	want_cuda="yes"
m4/ax_cuda.m4:AM_CONDITIONAL(USE_CUDA, test "x${want_cuda}" = xyes)
m4/ax_cuda.m4:if test "$want_cuda" = "yes"
m4/ax_cuda.m4:	if test -n "$cuda_home_path"
m4/ax_cuda.m4:	    nvcc_search_dirs="$PATH$PATH_SEPARATOR$cuda_home_path/bin"
m4/ax_cuda.m4:	# set CUDA flags
m4/ax_cuda.m4:	if test -n "$cuda_home_path"
m4/ax_cuda.m4:	    CUDA_CFLAGS="-I$cuda_home_path/include"
m4/ax_cuda.m4:	    CUDA_LIBS="-L$cuda_home_path/$libdir -lcudart"
m4/ax_cuda.m4:	    CUDA_CFLAGS="-I/usr/local/cuda/include"
m4/ax_cuda.m4:	    CUDA_LIBS="-L/usr/local/cuda/$libdir -lcudart"
m4/ax_cuda.m4:	# Env var CUDA_DRIVER_LIB_PATH can be used to set an alternate driver library path
m4/ax_cuda.m4:	if test -n "$CUDA_DRIVER_LIB_PATH"
m4/ax_cuda.m4:	    CUDA_LIBS+=" -L$CUDA_DRIVER_LIB_PATH -lcuda"
m4/ax_cuda.m4:	    CUDA_LIBS+=" -lcuda"
m4/ax_cuda.m4:	CPPFLAGS="$CPPFLAGS $CUDA_CFLAGS"
m4/ax_cuda.m4:	LIBS="$LIBS $CUDA_LIBS"
m4/ax_cuda.m4:	AC_MSG_CHECKING([for Cuda headers])
m4/ax_cuda.m4:                #include <cuda.h>
m4/ax_cuda.m4:                #include <cuda_runtime.h>
m4/ax_cuda.m4:		have_cuda_headers="yes"
m4/ax_cuda.m4:		have_cuda_headers="no"
m4/ax_cuda.m4:	AC_MSG_CHECKING([for Cuda libraries])
m4/ax_cuda.m4:                  #include <cuda.h>
m4/ax_cuda.m4:                  #include <cuda_runtime.h>
m4/ax_cuda.m4:                  ]],[[void* ptr = 0;cudaMalloc(&ptr, 1);]])
m4/ax_cuda.m4:		have_cuda_libs="yes"
m4/ax_cuda.m4:		have_cuda_libs="no"
m4/ax_cuda.m4:	if test "$have_cuda_headers" = "yes" -a "$have_cuda_libs" = "yes" -a "$have_nvcc" = "yes"
m4/ax_cuda.m4:		have_cuda="yes"
m4/ax_cuda.m4:		have_cuda="no"
m4/ax_cuda.m4:		AC_MSG_ERROR([Cuda is requested but not available])
m4/ax_cuda.m4:AC_SUBST(CUDA_CFLAGS)
m4/ax_cuda.m4:AC_SUBST(CUDA_LIBS)
m4/ax_cuda.m4:AC_ARG_WITH([cuda-fast-math],
m4/ax_cuda.m4:	[AC_HELP_STRING([--with-cuda-fast-math],
m4/ax_cuda.m4:    AS_HELP_STRING([--enable-emu], [Turn on device emulation for CUDA]),
m4/ax_cuda.m4:AS_IF([test "x$want_cuda" = xyes],
m4/ax_cuda.m4:        [dnl generate CUDA code for broad spectrum of devices
am/cuda.am:	$(top_srcdir)/am/cudalt.py $@ $(NVCC) $(NVCCFLAGS) -maxrregcount=$(MAX_REG_COUNT_SINGLE) -c $<
am/cuda.am:	$(top_srcdir)/am/cudalt.py $@ $(NVCC) $(NVCCFLAGS) -maxrregcount=$(MAX_REG_COUNT_DOUBLE) $(NVCCFLAGS_DOUBLE) -c $<
README_build:module load cuda/7.5 hdf5/parallel-1.8.13-gnu48-static pnetcdf/1.4.1
README_build:module load papi/5.3.0_nocuda
README_build:../configure --with-cuda=no --enable-timing --with-vtk-version=-6.0 --enable-double --with-cuda-fast-math=no --with-pnetcdf=/home/pkestene/local/pnetcdf-1.4.1 --with-hdf5=$HDF5_ROOT/bin/h5pcc --with-fftw3 --with-fftw3-mpi FFTW3_LIBDIR=/usr/lib/x86_64-linux-gnu --enable-mpi --with-papi=/home/pkestene/local/papi-5.3.0_nocuda
README_build:# gpu build
README_build:../configure --with-cuda=/usr/local/cuda-7.5 --enable-timing --with-vtk-version=-6.0 --enable-double --with-cuda-fast-math=no --with-pnetcdf=/home/pkestene/local/pnetcdf-1.4.1 --with-hdf5=$HDF5_BIN_DIR/h5pcc --with-fftw3 --with-fftw3-mpi FFTW3_LIBDIR=/usr/lib/x86_64-linux-gnu --enable-mpi --with-netcdf=no --disable-glutgui
README_build:../configure --with-cuda=no --enable-timing --with-vtk-version=-5.10 --enable-double --with-cuda-fast-math=no --with-pnetcdf=/home/pkestene/local/pnetcdf-1.7.0 --with-hdf5=no --with-fftw3 --with-fftw3-mpi FFTW3_LIBDIR=/usr/lib/x86_64-linux-gnu --enable-mpi --with-papi=/home/pkestene/local/papi-5.3.0_nocuda
README_build:# gpu (ubuntu 16.04)
README_build:../configure --with-cuda=/usr/local/cuda-8.0 --enable-timing --with-vtk-version=-5.10 --enable-double --with-cuda-fast-math=no --with-pnetcdf=/home/pkestene/local/pnetcdf-1.7.0 --with-hdf5=no --with-fftw3 --with-fftw3-mpi FFTW3_LIBDIR=/usr/lib/x86_64-linux-gnu --enable-mpi --with-netcdf=no --disable-qtgui --disable-glutgui
README_build:module load cuda/7.5 hdf5/parallel_gnu48 pnetcdf/1.3.1_gnu48 papi/5.2.0
README_build:../configure --with-cuda=no --with-boost=no --enable-timing --with-vtk-version=-6.0 --with-cuda-fast-math=no --with-pnetcdf=$PNETCDF_ROOT --with-hdf5=no --with-papi=$PAPI_ROOT --with-fftw3 --with-fftw3-mpi FFTW3_LIBDIR=/usr/lib/x86_64-linux-gnu --enable-mpi MPICXX=mpicxx --disable-qtgui --disable-glutgui
README_build:# gpu
README_build:../configure --with-cuda=$CUDA_ROOT --with-boost=no --enable-timing --with-vtk-version=-6.0 --with-cuda-fast-math=no --with-pnetcdf=$PNETCDF_ROOT --with-hdf5=no --with-papi=$PAPI_ROOT --with-fftw3 --with-fftw3-mpi FFTW3_LIBDIR=/usr/lib/x86_64-linux-gnu --enable-mpi MPICXX=mpicxx --disable-qtgui --disable-glutgui
Makefile.am:EXTRA_DIST = autogen.sh am/cudalt.py
Makefile.am:	data/jet2d_gpu.ini \
cmake/readme:FindCUDAlibs macro is borrowed from Damien Nguyen:
cmake/FindCUDAlibs.cmake:# FindCUDAlibs
cmake/FindCUDAlibs.cmake:# Find CUDA include dirs and libraries
cmake/FindCUDAlibs.cmake:#   find_package(CUDAlibs
cmake/FindCUDAlibs.cmake:#   BASIC       - Equivalent to CUDART;CUBLAS;CUFFT
cmake/FindCUDAlibs.cmake:#   CUDART      - CUDA RT library
cmake/FindCUDAlibs.cmake:#   CUBLAS      - CUDA BLAS library
cmake/FindCUDAlibs.cmake:#   CUFFT       - CUDA FFT library
cmake/FindCUDAlibs.cmake:#   CUFFTW      - CUDA FFTW library
cmake/FindCUDAlibs.cmake:#   CUPTI       - CUDA Profiling Tools Interface library.
cmake/FindCUDAlibs.cmake:#   CURAND      - CUDA Random Number Generation library.
cmake/FindCUDAlibs.cmake:#   CUSOLVER    - CUDA Direct Solver library.
cmake/FindCUDAlibs.cmake:#   CUSPARSE    - CUDA Sparse Matrix library.
cmake/FindCUDAlibs.cmake:#   NPP         - NVIDIA Performance Primitives lib.
cmake/FindCUDAlibs.cmake:#   NPPC        - NVIDIA Performance Primitives lib (core).
cmake/FindCUDAlibs.cmake:#   NPPI        - NVIDIA Performance Primitives lib (image processing).
cmake/FindCUDAlibs.cmake:#   NPPIAL      - NVIDIA Performance Primitives lib (image processing).
cmake/FindCUDAlibs.cmake:#   NPPICC      - NVIDIA Performance Primitives lib (image processing).
cmake/FindCUDAlibs.cmake:#   NPPICOM     - NVIDIA Performance Primitives lib (image processing).
cmake/FindCUDAlibs.cmake:#   NPPIDEI     - NVIDIA Performance Primitives lib (image processing).
cmake/FindCUDAlibs.cmake:#   NPPIF       - NVIDIA Performance Primitives lib (image processing).
cmake/FindCUDAlibs.cmake:#   NPPIG       - NVIDIA Performance Primitives lib (image processing).
cmake/FindCUDAlibs.cmake:#   NPPIM       - NVIDIA Performance Primitives lib (image processing).
cmake/FindCUDAlibs.cmake:#   NPPIST      - NVIDIA Performance Primitives lib (image processing).
cmake/FindCUDAlibs.cmake:#   NPPISU      - NVIDIA Performance Primitives lib (image processing).
cmake/FindCUDAlibs.cmake:#   NPPITC      - NVIDIA Performance Primitives lib (image processing).
cmake/FindCUDAlibs.cmake:#   NPPS        - NVIDIA Performance Primitives lib (signal processing).
cmake/FindCUDAlibs.cmake:#   NVBLAS      - NVIDIA BLAS library
cmake/FindCUDAlibs.cmake:#   CUDA_TOOLKIT_ROOT_DIR
cmake/FindCUDAlibs.cmake:#   CUDA_ROOT
cmake/FindCUDAlibs.cmake:#   CUDAlibs::CUDART         - Imported target for the CUDA RT library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::CUBLAS         - Imported target for the CUDA cublas library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::CUFFT          - Imported target for the CUDA cufft library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::CUFFTW         - Imported target for the CUDA cufftw library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::CUPTI          - Imported target for the CUDA cupti library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::CURAND         - Imported target for the CUDA curand library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::CUSOLVER       - Imported target for the CUDA cusolver library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::CUSPARSE       - Imported target for the CUDA cusparse library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPP            - Imported target for the CUDA npp library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPPC           - Imported target for the CUDA nppc library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPPI           - Imported target for the CUDA nppi library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPPIAL         - Imported target for the CUDA nppial library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPPICC         - Imported target for the CUDA nppicc library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPPICOM        - Imported target for the CUDA nppicom library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPPIDEI        - Imported target for the CUDA nppidei library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPPIF          - Imported target for the CUDA nppif library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPPIG          - Imported target for the CUDA nppig library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPPIM          - Imported target for the CUDA nppim library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPPIST         - Imported target for the CUDA nppist library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPPISU         - Imported target for the CUDA nppisu library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPPITC         - Imported target for the CUDA nppitc library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NPPS           - Imported target for the CUDA npps library
cmake/FindCUDAlibs.cmake:#   CUDAlibs::NVBLAS         - Imported target for the CUDA nvblas library
cmake/FindCUDAlibs.cmake:  message(FATAL_ERROR "Cannot use find_package(CUDAlibs ...) with CMake < 3.8! Use find_package(CUDA) instead.")
cmake/FindCUDAlibs.cmake:set(_cuda_root_dir_hint)
cmake/FindCUDAlibs.cmake:foreach(_dir ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
cmake/FindCUDAlibs.cmake:  list(APPEND _cuda_root_dir_hint ${_dirname})
cmake/FindCUDAlibs.cmake:set(CUDAlibs_SEARCH_PATHS
cmake/FindCUDAlibs.cmake:  ${_cuda_root_dir_hint}
cmake/FindCUDAlibs.cmake:  ${CUDA_ROOT}
cmake/FindCUDAlibs.cmake:  $ENV{CUDA_ROOT}
cmake/FindCUDAlibs.cmake:  ${CUDA_TOOLKIT_ROOT_DIR}
cmake/FindCUDAlibs.cmake:  $ENV{CUDA_TOOLKIT_ROOT_DIR}
cmake/FindCUDAlibs.cmake:  /usr/local/cuda
cmake/FindCUDAlibs.cmake:  set(_root_dir "C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA")
cmake/FindCUDAlibs.cmake:      list(APPEND CUDAlibs_SEARCH_PATHS ${_root_dir}/${child})
cmake/FindCUDAlibs.cmake:set(CUDAlibs_REQUIRED_VARS)
cmake/FindCUDAlibs.cmake:macro(cuda_find_library component)
cmake/FindCUDAlibs.cmake:  find_library(CUDAlibs_${component}_LIB
cmake/FindCUDAlibs.cmake:    PATHS ${CUDAlibs_SEARCH_PATHS}
cmake/FindCUDAlibs.cmake:  if(CUDAlibs_${component}_LIB)
cmake/FindCUDAlibs.cmake:    set(CUDAlibs_${component}_LIB_FOUND 1)
cmake/FindCUDAlibs.cmake:  list(APPEND CUDAlibs_REQUIRED_VARS "CUDAlibs_${component}_LIB")
cmake/FindCUDAlibs.cmake:  if(CUDAlibs_${component}_LIB_FOUND)
cmake/FindCUDAlibs.cmake:    set(CUDAlibs_${component}_FOUND 1)
cmake/FindCUDAlibs.cmake:    set(CUDAlibs_${component}_FOUND 0)
cmake/FindCUDAlibs.cmake:set(CUDAlibs_ALL_LIBS "CUDART;CUBLAS;CUFFT;CUFFTW;CUPTI;CURAND;CUSOLVER;CUSPARSE;NPP;NPPC;NPPI;NPPIAL;NPPICC;NPPICOM;NPPIDEI;NPPIF;NPPIG;NPPIM;NPPIST;NPPISU;NPPITC;NPPS;NVBLAS")
cmake/FindCUDAlibs.cmake:set(CUDAlibs_DEP_CUFFTW "CUFFT")
cmake/FindCUDAlibs.cmake:set(CUDAlibs_DEP_NPPIAL "NPPC")
cmake/FindCUDAlibs.cmake:set(CUDAlibs_DEP_NPPICC "NPPC")
cmake/FindCUDAlibs.cmake:set(CUDAlibs_DEP_NPPICOM "NPPC")
cmake/FindCUDAlibs.cmake:set(CUDAlibs_DEP_NPPIDEI "NPPC")
cmake/FindCUDAlibs.cmake:set(CUDAlibs_DEP_NPPIF "NPPC")
cmake/FindCUDAlibs.cmake:set(CUDAlibs_DEP_NPPIG "NPPC")
cmake/FindCUDAlibs.cmake:set(CUDAlibs_DEP_NPPIM "NPPC")
cmake/FindCUDAlibs.cmake:set(CUDAlibs_DEP_NPPIST "NPPC")
cmake/FindCUDAlibs.cmake:set(CUDAlibs_DEP_NPPISU "NPPC")
cmake/FindCUDAlibs.cmake:set(CUDAlibs_DEP_NPPITC "NPPC")
cmake/FindCUDAlibs.cmake:set(CUDAlibs_DEP_NPPS "NPPC")
cmake/FindCUDAlibs.cmake:set(CUDAlibs_DEP_NVBLAS "CUBLAS")
cmake/FindCUDAlibs.cmake:if(NOT CUDAlibs_FIND_COMPONENTS OR CUDAlibs_FIND_COMPONENTS STREQUAL "BASIC")
cmake/FindCUDAlibs.cmake:  set(CUDAlibs_FIND_COMPONENTS "CUDART;CUBLAS;CUFFT")
cmake/FindCUDAlibs.cmake:elseif(CUDAlibs_FIND_COMPONENTS STREQUAL "ALL")
cmake/FindCUDAlibs.cmake:  set(CUDAlibs_FIND_COMPONENTS ${CUDAlibs_ALL_LIBS})
cmake/FindCUDAlibs.cmake:foreach(COMPONENT ${CUDAlibs_FIND_COMPONENTS})
cmake/FindCUDAlibs.cmake:  if (${UPPERCOMPONENT} IN_LIST CUDAlibs_ALL_LIBS)
cmake/FindCUDAlibs.cmake:    cuda_find_library(${UPPERCOMPONENT})
cmake/FindCUDAlibs.cmake:    foreach(_dep "${CUDAlibs_DEP_${UPPERCOMPONENT}}")
cmake/FindCUDAlibs.cmake:	cuda_find_library(${_dep})
cmake/FindCUDAlibs.cmake:    CUDAlibs_${UPPERCOMPONENT}_LIB
cmake/FindCUDAlibs.cmake:    CUDAlibs_${UPPERCOMPONENT}_INCLUDE_DIR)
cmake/FindCUDAlibs.cmake:find_package_handle_standard_args(CUDAlibs
cmake/FindCUDAlibs.cmake:  FOUND_VAR CUDAlibs_FOUND
cmake/FindCUDAlibs.cmake:  REQUIRED_VARS ${CUDAlibs_REQUIRED_VARS}
cmake/FindCUDAlibs.cmake:if(CUDAlibs_FOUND)
cmake/FindCUDAlibs.cmake:  foreach(COMPONENT ${CUDAlibs_FIND_COMPONENTS})
cmake/FindCUDAlibs.cmake:    if(NOT TARGET CUDAlibs::${UPPERCOMPONENT} AND CUDAlibs_${UPPERCOMPONENT}_FOUND)
cmake/FindCUDAlibs.cmake:      get_filename_component(LIB_EXT "${CUDAlibs_${UPPERCOMPONENT}_LIB}" EXT)
cmake/FindCUDAlibs.cmake:      add_library(CUDAlibs::${UPPERCOMPONENT} ${LIB_TYPE} IMPORTED GLOBAL)
cmake/FindCUDAlibs.cmake:      set_target_properties(CUDAlibs::${UPPERCOMPONENT} PROPERTIES
cmake/FindCUDAlibs.cmake:	IMPORTED_LOCATION "${CUDAlibs_${UPPERCOMPONENT}_LIB}"
cmake/FindCUDAlibs.cmake:        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
cmake/FindCUDAlibs.cmake:      foreach(_dep "${CUDAlibs_DEP_${UPPERCOMPONENT}}")
cmake/FindCUDAlibs.cmake:	set_target_properties(CUDAlibs::${UPPERCOMPONENT} PROPERTIES
cmake/FindCUDAlibs.cmake:  if(NOT CUDAlibs_FIND_QUIETLY)
cmake/FindCUDAlibs.cmake:    message(STATUS "Found CUDAlibs and defined the following imported targets:")
cmake/FindCUDAlibs.cmake:    foreach(_comp ${CUDAlibs_FIND_COMPONENTS})
cmake/FindCUDAlibs.cmake:      message(STATUS "  - CUDAlibs::${_comp}")
cmake/FindCUDAlibs.cmake:  CUDAlibs_FOUND
cmake/cuda/readme:macro CMAKE_CUDA_CONVERT_FLAGS is borrowed from project
cmake/cuda/protect_pthread_flag.cmake:function(CUDA_PROTECT_PTHREAD_FLAG EXISTING_TARGET)
cmake/cuda/protect_pthread_flag.cmake:      "$<$<BUILD_INTERFACE:$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${new_flags}>"
cmake/cuda/protect_pthread_flag.cmake:      "$<$<BUILD_INTERFACE:$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${new_flags}>"
cmake/cuda/protect_nvcc_flags.cmake:CUDA Utilities
cmake/cuda/protect_nvcc_flags.cmake:This part of the protect flags module provides a set of utilities to assist users with CUDA as a language.
cmake/cuda/protect_nvcc_flags.cmake:.. command:: cmake_cuda_convert_flags
cmake/cuda/protect_nvcc_flags.cmake:  Take a list of flags or a target and convert the flags to pass through the CUDA compiler to 
cmake/cuda/protect_nvcc_flags.cmake:  This will make the flags are only used when the language is not CUDA.
cmake/cuda/protect_nvcc_flags.cmake:function(_CUDA_CONVERT_FLAGS flags_name)
cmake/cuda/protect_nvcc_flags.cmake:        # Use old flags for non-CUDA targets
cmake/cuda/protect_nvcc_flags.cmake:        set(protected_flags "$<$<BUILD_INTERFACE:$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:${old_flags}>")
cmake/cuda/protect_nvcc_flags.cmake:        # Add -Xcompiler wrapped flags for CUDA 
cmake/cuda/protect_nvcc_flags.cmake:            string(REPLACE ";" "," cuda_flags "${old_flags}")
cmake/cuda/protect_nvcc_flags.cmake:            string(APPEND protected_flags "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${cuda_flags}>")
cmake/cuda/protect_nvcc_flags.cmake:function(CMAKE_CUDA_CONVERT_FLAGS)
cmake/cuda/protect_nvcc_flags.cmake:        _cuda_convert_flags(old_flags "${CCF_PROTECT_ONLY}")
cmake/cuda/protect_nvcc_flags.cmake:        _cuda_convert_flags(LOCAL_LIST "${CCF_PROTECT_ONLY}")
cmake/cuda/protect_nvcc_flags.cmake:function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
cmake/cuda/protect_nvcc_flags.cmake:        string(REPLACE ";" "," CUDA_flags "${old_flags}")
cmake/cuda/protect_nvcc_flags.cmake:            "$<$<BUILD_INTERFACE:$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
src/euler_zslab_mpi_main.cpp: * method (with Riemann solvers) using MPI+CUDA and z-slab update method..
src/euler_zslab_mpi_main.cpp:  cout << "solve 3D Euler (Hydro or MHD) equations on a cartesian grid with MPI+CUDA and zslab method." << endl;
src/python/visu/Mayavi2RayCast.py:    #'implode3d_gpu_d_0000380.vti'
src/python/visu/visu_plane.py:"""Visualize hdf5 data from a RAMSES-GPU run.
src/python/visu/visu_plane.py:    """Read hdf5 data from RAMSES-GPU simulation run"""
src/python/visu/offscreen_render.py:        filename = 'jet3d_gpu_d_'+'%07d.vti' % i
src/python/visu/offscreen_render.py:    # convert -delay 50 jet3d_gpu_d_*.jpg jet3d_gpu_d.mpg
src/python/powerSpectrum/powerSpectrum.py:def read_ramses_gpu_data(fileName,fieldName='density'):
src/python/powerSpectrum/powerSpectrum.py:    """Read hdf5 data from RAMSES-GPU simulation run using h5py module.
src/python/powerSpectrum/powerSpectrum.py:    data = read_ramses_gpu_data(hdf5Filename,'density')
src/python/powerSpectrum/plot_powerSpectrum.py:from powerSpectrum2 import read_ramses_gpu_data
src/python/powerSpectrum/plot_powerSpectrum.py:    data = read_ramses_gpu_data(filename,dataname)
src/python/powerSpectrum/powerSpectrum2.pyx:    """Read hdf5 data from RAMSES-GPU simulation run using h5py module.
src/python/powerSpectrum/powerSpectrum2.pyx:    """ReadNetCDF data from RAMSES-GPU simulation run using Scientific.IO.NetCDF module.
src/python/powerSpectrum/powerSpectrum2.pyx:    """ReadNetCDF data from RAMSES-GPU simulation run using Scientific.IO.NetCDF module.
src/python/utils/convert_netcdf2vtr.py:    """Read hdf5 data from RAMSES-GPU simulation run using h5py module.
src/python/utils/convert_netcdf2vtr.py:    """ReadNetCDF data from RAMSES-GPU simulation run using Scientific.IO.NetCDF module.
src/test_euler2d_qt.sh:./euler2d_gpu_qt --posx 460 --posy 100&
src/test_euler2d_qt.sh:# recordmydesktop -x 200 -y 100 --width 430 --height 840 -o euler_cpu_gpu_screencast.ogv
src/qtGui/qtHydro2d/minmax.cuh: * \brief Implements GPU kernel for computing min and max value of
src/qtGui/qtHydro2d/minmax.cuh: * @param g_odata : GPU global memory buffer
src/qtGui/qtHydro2d/pbo.cuh: * \brief Implements GPU kernels for converting real_t (float or double )array 
src/qtGui/qtHydro2d/pbo.cuh: * CUDA kernel to fill plot_rgba_data array for plotting
src/qtGui/qtHydro2d/HydroWidgetGpu.h: * \file HydroWidgetGpu.h
src/qtGui/qtHydro2d/HydroWidgetGpu.h: * OpenGL Widget to display HydroRun simulation (CUDA version).
src/qtGui/qtHydro2d/HydroWidgetGpu.h:#ifndef HYDRO_WIDGET_GPU_H_
src/qtGui/qtHydro2d/HydroWidgetGpu.h:#define HYDRO_WIDGET_GPU_H_
src/qtGui/qtHydro2d/HydroWidgetGpu.h:/** use the graphics OpenGL/CUDA interoperability API available from CUDA >= 3.0 */
src/qtGui/qtHydro2d/HydroWidgetGpu.h://#define USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.h: * \class HydroWidgetGpu HydroWidgetGpu.h
src/qtGui/qtHydro2d/HydroWidgetGpu.h: * GPU computation results.
src/qtGui/qtHydro2d/HydroWidgetGpu.h:class HydroWidgetGpu : public HydroWidget
src/qtGui/qtHydro2d/HydroWidgetGpu.h:  HydroWidgetGpu(ConfigMap& _param, HydroRunBase* _hydroRun, QWidget *parent = 0);
src/qtGui/qtHydro2d/HydroWidgetGpu.h:  virtual ~HydroWidgetGpu();
src/qtGui/qtHydro2d/HydroWidgetGpu.h:#  ifdef USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.h:  struct cudaGraphicsResource* cuda_PBO;
src/qtGui/qtHydro2d/HydroWidgetGpu.h:#  endif // USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.h:}; // class HydroWidgetGpu
src/qtGui/qtHydro2d/HydroWidgetGpu.h:#endif // HYDRO_WIDGET_GPU_H_
src/qtGui/qtHydro2d/main.cpp: * \note Note that the GPU-based computations uses the CUDA/OpenGL
src/qtGui/qtHydro2d/main.cpp:#ifdef __CUDACC__
src/qtGui/qtHydro2d/main.cpp:#include "HydroWidgetGpu.h"
src/qtGui/qtHydro2d/main.cpp:#endif // __CUDACC__
src/qtGui/qtHydro2d/main.cpp:#ifdef __CUDACC__
src/qtGui/qtHydro2d/main.cpp:    winTitle = std::string("2D Euler simulation: Godunov scheme -- GPU");
src/qtGui/qtHydro2d/main.cpp:    winTitle = std::string("2D Euler simulation: Kurganov scheme -- GPU");
src/qtGui/qtHydro2d/main.cpp:#endif // __CUDACC__
src/qtGui/qtHydro2d/main.cpp:#ifdef __CUDACC__
src/qtGui/qtHydro2d/main.cpp:  HydroWidget* hydroWidget = (HydroWidget *) new HydroWidgetGpu(configMap, hydroRun, 0);
src/qtGui/qtHydro2d/main.cpp:#endif // __CUDACC__
src/qtGui/qtHydro2d/HydroWidgetGpu.cu: * \file HydroWidgetGpu.cu
src/qtGui/qtHydro2d/HydroWidgetGpu.cu: * \brief Implements class HydroWidgetGpu.
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:#include "HydroWidgetGpu.h"
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:// CUDA / OpenGl interoperability
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:#include <cuda_gl_interop.h>
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:HydroWidgetGpu::HydroWidgetGpu(ConfigMap& _param, HydroRunBase* _hydroRun,  QWidget *parent)
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:# ifdef USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  cuda_PBO = NULL;
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:# endif // USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:} // HydroWidgetGpu::HydroWidgetGpu
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:HydroWidgetGpu::~HydroWidgetGpu()
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  CUDA_SAFE_CALL( cudaFree(cmap_rgba_device) );
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:} // HydroWidgetGpu::~HydroWidgetGpu
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:void HydroWidgetGpu::computeMinMax(real_t *U, int size, int iVar)
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:} // HydroWidgetGpu::computeMinMax
src/qtGui/qtHydro2d/HydroWidgetGpu.cu: * The GPU version also copy this array to device memory to be used in
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:void HydroWidgetGpu::initColormap()
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  CUDA_SAFE_CALL( cudaMalloc((void **)&cmap_rgba_device, 
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  CUDA_SAFE_CALL( cudaMemcpy((void *)cmap_rgba_device,
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:			     cudaMemcpyHostToDevice));
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:} // HydroWidgetGpu::initColormap
src/qtGui/qtHydro2d/HydroWidgetGpu.cu: * this is a wrapper to call the CUDA kernel which actually convert data
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:void HydroWidgetGpu::convertDataForPlotting(int _useColor) {
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:} //HydroWidgetGpu::convertDataForPlotting 
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:void HydroWidgetGpu::createPBO()
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  std::cout << "[DEBUG] GPU version of createPBO" << std::endl;
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:   * CUDA only
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:#  ifdef USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer( &cuda_PBO, gl_PBO, cudaGraphicsMapFlagsNone ) );
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  cutilCheckMsg( "cudaGraphicsGLRegisterBuffer failed");
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  CUDA_SAFE_CALL( cudaGLRegisterBufferObject(gl_PBO) );
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  cutilCheckMsg( "cudaGLRegisterBufferObject failed");
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:#  endif // USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:} // HydroWidgetGpu::createPBO
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:void HydroWidgetGpu::deletePBO()
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:# ifdef USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:    CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( cuda_PBO ) );
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:    cutilCheckMsg( "cudaGraphicsUnRegisterResource failed");
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:    CUDA_SAFE_CALL( cudaGLUnregisterBufferObject( gl_PBO ) );
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:    cutilCheckMsg( "cudaGLUnRegisterBufferObject failed");
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:# endif // USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:# ifdef USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:    cuda_PBO = NULL;
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:# endif // USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:} // HydroWidgetGpu::deletePBO
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:void HydroWidgetGpu::render()
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  CUDA_SAFE_THREAD_SYNC( );
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  // For plotting, map the gl_PBO pixel buffer into CUDA context
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  // space, so that CUDA can modify it
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:#ifdef USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:    CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &cuda_PBO, NULL));
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:    cutilCheckMsg( "cudaGraphicsMapResources failed");
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:    CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void **)&plot_rgba_pbo, &num_bytes, cuda_PBO));
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:    cutilCheckMsg( "cudaGraphicsResourceGetMappedPointer failed");
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:    CUDA_SAFE_CALL( cudaGLMapBufferObject((void**)&plot_rgba_pbo, gl_PBO) );
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:    cutilCheckMsg( "cudaGLMapBufferObject failed");
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:#endif // USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:#ifdef USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &cuda_PBO, NULL));
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  cutilCheckMsg( "cudaGraphicsUnmapResources failed" );
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  CUDA_SAFE_CALL( cudaGLUnmapBufferObject(gl_PBO) );
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  cutilCheckMsg( "cudaGLUnmapBufferObject failed" );
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:#endif // USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:} // HydroWidgetGpu::render
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:void HydroWidgetGpu::initializeGL()
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:#ifdef USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  printf("### using the new Cuda/OpenGL inter-operability API (Cuda >= 3.0)\n");
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  printf("### using the deprecated Cuda/OpenGL inter-operability API (Cuda < 3.0)\n");
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:#endif // USE_CUDA3
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  // #ifdef __CUDACC__
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  //   CUDA_SAFE_CALL( cudaGLSetGLDevice( 0 ) );
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  //   CUDA_SAFE_CALL( cudaGetLastError() );
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:  // #endif // __CUDACC__
src/qtGui/qtHydro2d/HydroWidgetGpu.cu:} // HydroWidgetGpu::initializeGL
src/qtGui/qtHydro2d/README:It should be completely refactored to separate CUDA from QT
src/qtGui/qtHydro2d/HydroWidget.cpp: * The GPU version override this routine. 
src/testRiemannHLLD.cpp:#ifdef __CUDACC__
src/testRiemannHLLD.cpp:__global__ void testRiemannHLL_gpu(real_t *qleft,
src/testRiemannHLLD.cpp:__global__ void testRiemannHLLD_gpu(real_riemann_t *qleft,
src/testRiemannHLLD.cpp:#endif // __CUDACC__
src/testRiemannHLLD.cpp:#ifdef __CUDACC__
src/testRiemannHLLD.cpp:    cudaMalloc( (void**)&d_qleft,  NVAR_3D*sizeof(real_t) );
src/testRiemannHLLD.cpp:    cudaMalloc( (void**)&d_qright, NVAR_3D*sizeof(real_t) );
src/testRiemannHLLD.cpp:    cudaMalloc( (void**)&d_flux,   NVAR_3D*sizeof(real_t) );
src/testRiemannHLLD.cpp:    cudaMemcpy(d_qleft,  qLeft,  NVAR_3D*sizeof(real_t), cudaMemcpyHostToDevice);
src/testRiemannHLLD.cpp:    cudaMemcpy(d_qright, qRight, NVAR_3D*sizeof(real_t), cudaMemcpyHostToDevice);
src/testRiemannHLLD.cpp:    testRiemannHLL_gpu<<<1,1>>>(d_qleft, d_qright, d_flux);
src/testRiemannHLLD.cpp:    cudaMemcpy(flux, d_flux, NVAR_3D*sizeof(real_t), cudaMemcpyDeviceToHost);
src/testRiemannHLLD.cpp:    cudaFree(d_qleft);
src/testRiemannHLLD.cpp:    cudaFree(d_qright);
src/testRiemannHLLD.cpp:    cudaFree(d_flux);
src/testRiemannHLLD.cpp:#endif // __CUDACC__
src/testRiemannHLLD.cpp:#ifdef __CUDACC__
src/testRiemannHLLD.cpp:    cudaMalloc( (void**)&d_qleft,  NVAR_MHD*sizeof(real_riemann_t) );
src/testRiemannHLLD.cpp:    cudaMalloc( (void**)&d_qright, NVAR_MHD*sizeof(real_riemann_t) );
src/testRiemannHLLD.cpp:    cudaMalloc( (void**)&d_flux,   NVAR_MHD*sizeof(real_riemann_t) );
src/testRiemannHLLD.cpp:    cudaMemcpy(d_qleft,  qLeft,  NVAR_MHD*sizeof(real_riemann_t), cudaMemcpyHostToDevice);
src/testRiemannHLLD.cpp:    cudaMemcpy(d_qright, qRight, NVAR_MHD*sizeof(real_riemann_t), cudaMemcpyHostToDevice);
src/testRiemannHLLD.cpp:    testRiemannHLLD_gpu<<<1,1>>>(d_qleft, d_qright, d_flux);
src/testRiemannHLLD.cpp:    cudaMemcpy(flux, d_flux, NVAR_MHD*sizeof(real_riemann_t), cudaMemcpyDeviceToHost);
src/testRiemannHLLD.cpp:    cudaFree(d_qleft);
src/testRiemannHLLD.cpp:    cudaFree(d_qright);
src/testRiemannHLLD.cpp:    cudaFree(d_flux);
src/testRiemannHLLD.cpp:#endif // __CUDACC__
src/euler_mpi_main.cpp: * method (with Riemann solvers) using MPI+CUDA.
src/euler_mpi_main.cpp:  cout << "solve 2D/3D Euler (Hydro or MHD) equations on a cartesian grid with MPI+CUDA." << endl;
src/glutGui/minmax.cuh: * \brief Implement GPU kernel for computing min and max value of
src/glutGui/minmax.cuh: * @param g_odata : GPU global memory buffer
src/glutGui/pbo.cuh: * \brief Implement GPU kernels for converting real_t (float or double )array 
src/glutGui/pbo.cuh: * CUDA kernel to fill plot_rgba_data array for plotting
src/glutGui/HydroWindow.cpp:#ifdef __CUDACC__
src/glutGui/HydroWindow.cpp:#endif // __CUDACC__
src/glutGui/HydroWindow.cpp:#ifdef __CUDACC__
src/glutGui/HydroWindow.cpp:# ifdef USE_CUDA3
src/glutGui/HydroWindow.cpp:  cuda_PBO = NULL;
src/glutGui/HydroWindow.cpp:# endif // USE_CUDA3
src/glutGui/HydroWindow.cpp:#endif // __CUDACC__
src/glutGui/HydroWindow.cpp:#ifdef __CUDACC__
src/glutGui/HydroWindow.cpp:#endif // __CUDACC__
src/glutGui/HydroWindow.cpp:#ifdef __CUDACC__
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_CALL( cudaFree(d_cmap)     );
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_CALL( cudaFree(d_cmap_der) );
src/glutGui/HydroWindow.cpp:#endif // __CUDACC__
src/glutGui/HydroWindow.cpp:#ifdef __CUDACC__
src/glutGui/HydroWindow.cpp:#  ifdef USE_CUDA3
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer( &cuda_PBO, gl_PBO, cudaGraphicsMapFlagsNone ) );
src/glutGui/HydroWindow.cpp:  cutilCheckMsg( "cudaGraphicsGLRegisterBuffer failed");
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_CALL( cudaGLRegisterBufferObject( gl_PBO ) );
src/glutGui/HydroWindow.cpp:  cutilCheckMsg( "cudaGLRegisterBufferObject failed");
src/glutGui/HydroWindow.cpp:#  endif // USE_CUDA3
src/glutGui/HydroWindow.cpp:#endif // __CUDACC__  
src/glutGui/HydroWindow.cpp:#ifdef __CUDACC__
src/glutGui/HydroWindow.cpp:# ifdef USE_CUDA3
src/glutGui/HydroWindow.cpp:    CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( cuda_PBO ) );
src/glutGui/HydroWindow.cpp:    cutilCheckMsg( "cudaGraphicsUnRegisterResource failed");
src/glutGui/HydroWindow.cpp:    CUDA_SAFE_CALL( cudaGLUnregisterBufferObject( gl_PBO ) );
src/glutGui/HydroWindow.cpp:    cutilCheckMsg( "cudaGLUnRegisterBufferObject failed");
src/glutGui/HydroWindow.cpp:# endif // USE_CUDA3
src/glutGui/HydroWindow.cpp:#endif // __CUDACC__
src/glutGui/HydroWindow.cpp:#ifdef __CUDACC__
src/glutGui/HydroWindow.cpp:# ifdef USE_CUDA3
src/glutGui/HydroWindow.cpp:    cuda_PBO = NULL;
src/glutGui/HydroWindow.cpp:# endif // USE_CUDA3
src/glutGui/HydroWindow.cpp:#endif // __CUDACC__
src/glutGui/HydroWindow.cpp:   * The CUDA version is done by a kernel.
src/glutGui/HydroWindow.cpp:#ifdef __CUDACC__
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_THREAD_SYNC( );
src/glutGui/HydroWindow.cpp:  // For plotting, map the gl_PBO pixel buffer into CUDA context
src/glutGui/HydroWindow.cpp:  // space, so that CUDA can modify the device pointer plot_rgba_pbo
src/glutGui/HydroWindow.cpp:#ifdef USE_CUDA3
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &cuda_PBO, NULL));
src/glutGui/HydroWindow.cpp:  cutilCheckMsg( "cudaGraphicsMapResources failed");
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void **)&plot_rgba_pbo, &num_bytes, cuda_PBO));
src/glutGui/HydroWindow.cpp:  cutilCheckMsg( "cudaGraphicsResourceGetMappedPointer failed");
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_CALL( cudaGLMapBufferObject((void**)&plot_rgba_pbo, gl_PBO) );
src/glutGui/HydroWindow.cpp:#endif // USE_CUDA3
src/glutGui/HydroWindow.cpp:#ifdef USE_CUDA3
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &cuda_PBO, NULL));
src/glutGui/HydroWindow.cpp:  cutilCheckMsg( "cudaGraphicsUnmapResources failed" );
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_CALL( cudaGLUnmapBufferObject(gl_PBO) );
src/glutGui/HydroWindow.cpp:  cutilCheckMsg( "cudaGLUnmapBufferObject failed" );
src/glutGui/HydroWindow.cpp:#endif // USE_CUDA3
src/glutGui/HydroWindow.cpp:#endif // __CUDACC__
src/glutGui/HydroWindow.cpp:	    hydroRun->copyGpuToCpu(nStep);
src/glutGui/HydroWindow.cpp: * The GPU version also copy these arrays to device memory for use in
src/glutGui/HydroWindow.cpp:#ifdef __CUDACC__
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_CALL( cudaMalloc((void **)&d_cmap, 
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_CALL( cudaMemcpy((void *)d_cmap,
src/glutGui/HydroWindow.cpp:			     cudaMemcpyHostToDevice) );
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_CALL( cudaMalloc((void **)&d_cmap_der, 
src/glutGui/HydroWindow.cpp:  CUDA_SAFE_CALL( cudaMemcpy((void *)d_cmap_der,
src/glutGui/HydroWindow.cpp:			     cudaMemcpyHostToDevice) );
src/glutGui/HydroWindow.cpp:#endif // __CUDACC__
src/glutGui/HydroWindow.cpp:#ifdef __CUDACC__
src/glutGui/HydroWindow.cpp: * this is a wrapper to call the CUDA kernel which actually convert data
src/glutGui/HydroWindow.cpp:#endif // __CUDACC__
src/glutGui/HydroWindow.cpp:#ifdef __CUDACC__
src/glutGui/HydroWindow.cpp:} // HydroWindow::computeMinMax (GPU version)
src/glutGui/HydroWindow.cpp:#endif // __CUDACC__
src/glutGui/HydroWindow.h:#ifdef __CUDACC__
src/glutGui/HydroWindow.h:#endif // __CUDACC__
src/glutGui/HydroWindow.h:// CUDA / OpenGl interoperability
src/glutGui/HydroWindow.h:#ifdef __CUDACC__
src/glutGui/HydroWindow.h:#include <cuda_gl_interop.h>
src/glutGui/HydroWindow.h:#endif // __CUDACC__
src/glutGui/HydroWindow.h:/** use the grapgics API available from CUDA >= 3.0 */
src/glutGui/HydroWindow.h://#define USE_CUDA3
src/glutGui/HydroWindow.h:#ifdef __CUDACC__
src/glutGui/HydroWindow.h:#  ifdef USE_CUDA3
src/glutGui/HydroWindow.h:  struct cudaGraphicsResource* cuda_PBO;
src/glutGui/HydroWindow.h:#  endif // USE_CUDA3
src/glutGui/HydroWindow.h:#endif // __CUDACC__
src/glutGui/HydroWindow.h:#ifdef __CUDACC__
src/glutGui/HydroWindow.h:#endif // __CUDACC__
src/glutGui/HydroWindow.h:  //! between d_U and d_U2 (GPU)
src/glutGui/HydroWindow.h:#ifdef __CUDACC__
src/glutGui/HydroWindow.h:#endif // __CUDACC__
src/glutGui/HydroWindow.h:#ifdef __CUDACC__
src/glutGui/HydroWindow.h:#endif // __CUDACC__
src/glutGui/HydroWindow.h:#ifdef __CUDACC__
src/glutGui/HydroWindow.h:  //! this is the GPU version which wraps the CUDA kernel call.
src/glutGui/HydroWindow.h:#endif // __CUDACC__
src/euler2d_main_glut.cpp:#ifdef __CUDACC__
src/euler2d_main_glut.cpp:#endif // __CUDACC__
src/euler2d_main_glut.cpp:#ifdef __CUDACC__
src/euler2d_main_glut.cpp:    winTitle = std::string("2D Euler simulation: Godunov scheme -- GPU");
src/euler2d_main_glut.cpp:    winTitle = std::string("2D Euler simulation: Kurganov scheme -- GPU");
src/euler2d_main_glut.cpp:    winTitle = std::string("2D Euler simulation: Relaxing TVD scheme -- GPU");
src/euler2d_main_glut.cpp:    winTitle = std::string("2D MHD simulation: Godunov scheme -- GPU");
src/euler2d_main_glut.cpp:#endif // __CUDACC__
src/analysis/CMakeLists.txt:if (USE_MPI AND USE_PNETCDF AND NOT CUDA)
src/analysis/structureFunctions/CMakeLists.txt:  RamsesGPU::config
src/analysis/structureFunctions/CMakeLists.txt:  RamsesGPU::cnpy
src/analysis/structureFunctions/CMakeLists.txt:  RamsesGPU::pnetcdf)
src/analysis/structureFunctions/CMakeLists.txt:  RamsesGPU::config
src/analysis/structureFunctions/CMakeLists.txt:  RamsesGPU::cnpy
src/analysis/structureFunctions/CMakeLists.txt:  RamsesGPU::pnetcdf)
src/analysis/structureFunctions/CMakeLists.txt:  RamsesGPU::config
src/analysis/structureFunctions/CMakeLists.txt:  RamsesGPU::cnpy
src/analysis/structureFunctions/CMakeLists.txt:  RamsesGPU::pnetcdf)
src/analysis/structureFunctions/CMakeLists.txt:  RamsesGPU::config
src/analysis/structureFunctions/CMakeLists.txt:  RamsesGPU::cnpy
src/analysis/structureFunctions/CMakeLists.txt:  RamsesGPU::pnetcdf)
src/analysis/readSlice/CMakeLists.txt:  RamsesGPU::config
src/analysis/readSlice/CMakeLists.txt:  RamsesGPU::cnpy
src/analysis/readSlice/CMakeLists.txt:  RamsesGPU::mpiUtils
src/analysis/readSlice/CMakeLists.txt:  RamsesGPU::pnetcdf)
src/analysis/powerSpectrum/generate_fBm.cpp: * Data are then dump into a pnetcdf file comptatible with RamsesGPU.
src/analysis/powerSpectrum/CMakeLists.txt:      RamsesGPU::config
src/analysis/powerSpectrum/CMakeLists.txt:      RamsesGPU::cnpy
src/analysis/powerSpectrum/CMakeLists.txt:      RamsesGPU::config
src/analysis/powerSpectrum/CMakeLists.txt:      RamsesGPU::cnpy
src/analysis/powerSpectrum/CMakeLists.txt:      RamsesGPU::config
src/analysis/powerSpectrum/CMakeLists.txt:      RamsesGPU::cnpy
src/CMakeLists.txt:  if (USE_CUDA)
src/CMakeLists.txt:    set(RamsesGPU_main_exe ramsesGPU_mpi_cuda)
src/CMakeLists.txt:  else(USE_CUDA)
src/CMakeLists.txt:    set(RamsesGPU_main_exe ramsesGPU_mpi_cpu)
src/CMakeLists.txt:  endif(USE_CUDA)
src/CMakeLists.txt:  if (USE_CUDA)
src/CMakeLists.txt:    set(RamsesGPU_main_exe ramsesGPU_cuda)
src/CMakeLists.txt:  else(USE_CUDA)
src/CMakeLists.txt:    set(RamsesGPU_main_exe ramsesGPU_cpu)
src/CMakeLists.txt:  endif(USE_CUDA)
src/CMakeLists.txt:add_executable(${RamsesGPU_main_exe} "")
src/CMakeLists.txt:  target_sources(${RamsesGPU_main_exe}
src/CMakeLists.txt:  target_sources(${RamsesGPU_main_exe}
src/CMakeLists.txt:if (USE_CUDA)
src/CMakeLists.txt:  set_source_files_properties(euler_main.cpp     PROPERTIES LANGUAGE CUDA)
src/CMakeLists.txt:  set_source_files_properties(euler_mpi_main.cpp PROPERTIES LANGUAGE CUDA)
src/CMakeLists.txt:target_include_directories(${RamsesGPU_main_exe}
src/CMakeLists.txt:target_link_libraries(${RamsesGPU_main_exe}
src/CMakeLists.txt:  RamsesGPU::hydro
src/CMakeLists.txt:  RamsesGPU::config
src/CMakeLists.txt:  RamsesGPU::cnpy
src/CMakeLists.txt:  RamsesGPU::monitoring)
src/CMakeLists.txt:  target_link_libraries(${RamsesGPU_main_exe}
src/CMakeLists.txt:    RamsesGPU::mpiUtils)
src/CMakeLists.txt:#   hydro/gpu_macros.cpp
src/CMakeLists.txt:#   RamsesGPU::config)
src/CMakeLists.txt:# if (USE_CUDA)
src/CMakeLists.txt:#   set_source_files_properties(testRiemannHLLD.cpp PROPERTIES LANGUAGE CUDA)
src/CMakeLists.txt:#   set_source_files_properties(hydro/gpu_macros.cpp PROPERTIES LANGUAGE CUDA)
src/CMakeLists.txt:#   set_source_files_properties(hydro/constants.cpp PROPERTIES LANGUAGE CUDA)
src/CMakeLists.txt:# endif(USE_CUDA)
src/CMakeLists.txt:#   hydro/gpu_macros.cpp
src/CMakeLists.txt:#   RamsesGPU::config)
src/CMakeLists.txt:# # warning: since source file gpu_macros and constants.cpp
src/CMakeLists.txt:# # have already been declared as CUDA file, we need to stick with it
src/CMakeLists.txt:# if (USE_CUDA)
src/CMakeLists.txt:#   set_source_files_properties(testTrace.cpp PROPERTIES LANGUAGE CUDA)
src/CMakeLists.txt:# endif(USE_CUDA)
src/CMakeLists.txt:#if (USE_CUDA)
src/CMakeLists.txt:#CUDA_PROTECT_PTHREAD_FLAG(testTrace)
src/CMakeLists.txt:#endif(USE_CUDA)
src/hydro/hydro_update.cuh: * \brief CUDA kernel for update conservative variables with flux array.
src/hydro/hydro_update.cuh: * CUDA kernel perform hydro update from flux arrays (2D data).
src/hydro/hydro_update.cuh: * CUDA kernel perform hydro update (energy only) from flux arrays (2D data).
src/hydro/hydro_update.cuh: * CUDA kernel perform hydro update from flux arrays (3D data).
src/hydro/hydro_update.cuh: * CUDA kernel perform hydro update (energy only) from flux arrays (3D data).
src/hydro/gpu_macros.cpp: * \file gpu_macros.cpp
src/hydro/gpu_macros.cpp: * \brief Some useful GPU related macros.
src/hydro/gpu_macros.cpp: * $Id: gpu_macros.cpp 2108 2012-05-23 12:07:21Z pkestene $
src/hydro/gpu_macros.cpp:#include "gpu_macros.h"
src/hydro/gpu_macros.cpp:#ifdef __CUDACC__
src/hydro/gpu_macros.cpp:#endif // __CUDACC__
src/hydro/constants.cpp:#ifndef __CUDACC__
src/hydro/constants.cpp:#endif // __CUDACC__
src/hydro/shearBorderUtils.h:#ifdef __CUDACC__
src/hydro/shearBorderUtils.h:#endif // __CUDACC__
src/hydro/shearBorderUtils.h:#ifdef __CUDACC__
src/hydro/shearBorderUtils.h:   * GPU copy border buf for shearing box routines
src/hydro/shearBorderUtils.h:      } else { // b was allocated with cudaMalloc
src/hydro/shearBorderUtils.h:      } else { // b was allocated with cudaMalloc
src/hydro/shearBorderUtils.h:#endif // __CUDACC__
src/hydro/MHDRunBase.h:   * \note Important note : this class does sequential computations (one CPU or GPU).
src/hydro/MHDRunBase.h:   * For parallel computations (multi-CPU/GPU, symbol USE_MPI
src/hydro/MHDRunBase.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/MHDRunBase.h:#ifdef __CUDACC__
src/hydro/MHDRunBase.h:#endif // __CUDACC__
src/hydro/MHDRunBase.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/MHDRunBase.h:#ifdef __CUDACC__
src/hydro/MHDRunBase.h:#endif // __CUDACC__
src/hydro/MHDRunBase.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/MHDRunBase.h:#ifdef __CUDACC__
src/hydro/MHDRunBase.h:#endif // __CUDACC__
src/hydro/MHDRunBase.h:#ifdef __CUDACC__
src/hydro/MHDRunBase.h:#endif // __CUDACC__
src/hydro/MHDRunBase.h:#ifdef __CUDACC__
src/hydro/MHDRunBase.h:#endif // __CUDACC__
src/hydro/MHDRunBase.h:#ifdef __CUDACC__
src/hydro/MHDRunBase.h:#endif // __CUDACC__
src/hydro/MHDRunBase.h:#ifdef __CUDACC__
src/hydro/MHDRunBase.h:#endif // __CUDACC__
src/hydro/MHDRunBase.h:#ifdef __CUDACC__
src/hydro/MHDRunBase.h:#endif // __CUDACC__
src/hydro/MHDRunBase.h:#ifdef __CUDACC__
src/hydro/MHDRunBase.h:#endif // __CUDACC__
src/hydro/make_boundary_base.h: * computations on both CPU and GPU (via CUDA kernels) versions of the code.
src/hydro/make_boundary_base.h:#include "gpu_macros.h"
src/hydro/make_boundary_base.h: * inefficient on GPU hardware.
src/hydro/make_boundary_base.h:#ifdef __CUDACC__
src/hydro/make_boundary_base.h:#endif // __CUDACC__
src/hydro/make_boundary_base.h: * inefficient on GPU hardware.
src/hydro/make_boundary_base.h:#ifdef __CUDACC__
src/hydro/make_boundary_base.h:} // end make_boundary2 (GPU version)
src/hydro/make_boundary_base.h:#endif // __CUDACC__
src/hydro/make_boundary_base.h:#ifndef __CUDACC__
src/hydro/make_boundary_base.h:#endif // ! __CUDACC__
src/hydro/make_boundary_base.h:#ifdef __CUDACC__
src/hydro/make_boundary_base.h: * \brief Actual GPU CUDA kernel used to fill a Z direction boundary of the grid in the special case of stratified rotating MHD simulation.
src/hydro/make_boundary_base.h:void make_boundary2_z_stratified_gpu_kernel1(real_t* U, 
src/hydro/make_boundary_base.h:} // end make_boundary2_z_stratified_gpu_kernel1
src/hydro/make_boundary_base.h: * \brief Actual GPU CUDA kernel used to fill a Z direction boundary of the grid in the special case of stratified rotating MHD simulation.
src/hydro/make_boundary_base.h:void make_boundary2_z_stratified_gpu_kernel2(real_t* U, 
src/hydro/make_boundary_base.h:} // make_boundary2_z_stratified_gpu_kernel2
src/hydro/make_boundary_base.h:#endif // __CUDACC__
src/hydro/make_boundary_base.h: * \brief Wrapper routine to actual CPU or GPU kernel to fill a Z direction boundary of the grid in the special case of stratified rotating MHD simulation.
src/hydro/make_boundary_base.h:#ifdef __CUDACC__
src/hydro/make_boundary_base.h:   * make_boundary2_z_stratified_gpu_kernel1
src/hydro/make_boundary_base.h:   * make_boundary2_z_stratified_gpu_kernel2
src/hydro/make_boundary_base.h:  make_boundary2_z_stratified_gpu_kernel1<boundaryLoc>
src/hydro/make_boundary_base.h:  make_boundary2_z_stratified_gpu_kernel2<boundaryLoc>
src/hydro/make_boundary_base.h:} // end make_boundary2_z_stratified (GPU version)
src/hydro/make_boundary_base.h:#endif // __CUDACC__
src/hydro/make_boundary_base.h:#ifdef __CUDACC__
src/hydro/make_boundary_base.h:#endif // __CUDACC__
src/hydro/godunov_trace_v2.cuh: * \brief Defines the CUDA kernel for the actual Godunov scheme
src/hydro/godunov_unsplit.cuh: * \brief Defines the CUDA kernel for the actual Godunov scheme
src/hydro/godunov_unsplit.cuh: * Here are CUDA kernels implementing hydro unsplit scheme version 0
src/hydro/godunov_unsplit.cuh: * Here are CUDA kernel implementing hydro unsplit scheme version 1
src/hydro/make_boundary_shear.h:#include "gpu_macros.h"
src/hydro/make_boundary_shear.h:#ifdef __CUDACC__
src/hydro/make_boundary_shear.h: * Shearing box cuda kernels
src/hydro/make_boundary_shear.h:#endif // __CUDACC__
src/hydro/HydroRunKT.cpp:// include CUDA kernel when necessary
src/hydro/HydroRunKT.cpp:#ifdef __CUDACC__
src/hydro/HydroRunKT.cpp:#endif // __CUDACC__
src/hydro/HydroRunKT.cpp:#ifdef __CUDACC__
src/hydro/HydroRunKT.cpp:#endif // __CUDACC__
src/hydro/HydroRunKT.cpp:#ifdef __CUDACC__
src/hydro/HydroRunKT.cpp:#endif // __CUDACC__
src/hydro/HydroRunKT.cpp:#ifdef __CUDACC__
src/hydro/HydroRunKT.cpp:#endif // __CUDACC__
src/hydro/HydroRunKT.cpp:#ifdef __CUDACC__
src/hydro/HydroRunKT.cpp:#endif // __CUDACC__
src/hydro/HydroRunKT.cpp:void HydroRunKT::copyGpuToCpu(int nStep)
src/hydro/HydroRunKT.cpp:#ifdef __CUDACC__
src/hydro/HydroRunKT.cpp:#endif // __CUDACC__
src/hydro/HydroRunKT.cpp:} // HydroRunKT::copyGpuToCpu
src/hydro/HydroRunKT.cpp:#ifdef __CUDACC__
src/hydro/HydroRunKT.cpp:#endif // __CUDACC__
src/hydro/HydroRunKT.cpp:#ifdef __CUDACC__
src/hydro/HydroRunKT.cpp:#endif // __CUDACC__
src/hydro/HydroRunKT.cpp:#ifdef __CUDACC__
src/hydro/HydroRunKT.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#include "gpu_macros.h"
src/hydro/HydroRunBase.h:   * \note Important note : this class does sequential computations (one CPU or GPU).
src/hydro/HydroRunBase.h:   * For parallel computations (multi-CPU/GPU, symbol USE_MPI
src/hydro/HydroRunBase.h:     * \param[in]  U      (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:     * \param[in]  U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:     * \param[in,out] U  (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:     * \param[in,out] U  (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__   
src/hydro/HydroRunBase.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__   
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:    //! used in the GPU version to control the number of CUDA blocks in compute dt
src/hydro/HydroRunBase.h:    //! used in the GPU version to control the number of CUDA blocks in random forcing normalization
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:    // Data Arrays (these arrays are only used for the GPU version 
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:    //! compute border on GPU (call make_boundary for each borders in direction idim)
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:    //! compute all borders on GPU (call make_boundary for each borders)
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:     * GPU version copy data back from GPU to host  memory (h_U).
src/hydro/HydroRunBase.h:    virtual void copyGpuToCpu(int nStep=0);
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:     * <b>GPU version (cuda kernels are defined in make_boundary_base.h).</b>
src/hydro/HydroRunBase.h:     * <b>GPU version.</b>
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:    //! Data array on GPU
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/HydroRunBase.h:#ifdef __CUDACC__
src/hydro/HydroRunBase.h:#endif // __CUDACC__
src/hydro/laxliu.cuh: * \brief Implement GPU kernels for the Lax-Liu positive scheme.
src/hydro/mhd_ct_update.cuh: * \brief CUDA kernel for update magnetic field with emf (constraint transport).
src/hydro/mhd_ct_update.cuh: * CUDA kernel perform magnetic field update (ct) from emf (2D data).
src/hydro/mhd_ct_update.cuh: * CUDA kernel perform magnetic field update (ct) from emf (3D data).
src/hydro/MHDRunGodunovMpi.cpp: * Parallel CUDA+MPI implementation.
src/hydro/MHDRunGodunovMpi.cpp:// include CUDA kernel when necessary
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifndef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:      d_emf.allocate(make_uint3(isize, jsize, 1           ), gpuMemAllocType); // only EMFZ
src/hydro/MHDRunGodunovMpi.cpp:      d_emf.allocate(make_uint4(isize, jsize, ksize, 3    ), gpuMemAllocType); // 3 EMF's
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:	d_qm_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	d_qm_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	d_qEdge_RT.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	d_qEdge_RB.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	d_qEdge_LT.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qm_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType); // 2D array
src/hydro/MHDRunGodunovMpi.cpp:	  d_qm_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType); // 2D array
src/hydro/MHDRunGodunovMpi.cpp:	  d_qm_z.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType); // 3D array with only 2 plans
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_RT.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_RB.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_LT.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_RT2.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_RB2.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_LT2.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_RT3.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_RB3.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_LT3.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qm_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qm_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qm_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qp_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qp_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qp_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_RT.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_RB.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_LT.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_LB.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_RT2.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_RB2.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_LT2.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_LB2.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_RT3.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_RB3.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_LT3.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	  d_qEdge_LB3.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:	//d_elec.allocate(make_uint3(isize, jsize, 1), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	d_elec.allocate (make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	d_dA.allocate   (make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	d_dB.allocate   (make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	d_dC.allocate   (make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);	
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:      d_Q.allocate (make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:      	d_shear_flux_xmin_toSend.allocate   (make_uint3(jsize             ,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:      	d_shear_flux_xmax_toSend.allocate   (make_uint3(jsize             ,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:      	d_shear_flux_xmin_remap.allocate    (make_uint3(jsize             ,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:      	d_shear_flux_xmax_remap.allocate    (make_uint3(jsize             ,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:      	d_shear_flux_xmin_recv_glob.allocate(make_uint3(my*ny+2*ghostWidth,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:      	d_shear_flux_xmax_recv_glob.allocate(make_uint3(my*ny+2*ghostWidth,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:	d_shear_border.allocate           (make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:     * memory allocation for GPU routines debugging
src/hydro/MHDRunGodunovMpi.cpp:// #ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:    // GPU execution settings
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:      cudaFuncSetCacheConfig(kernel_mhd_compute_trace_v4,                 cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovMpi.cpp:      cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_shear,       cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovMpi.cpp:      cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_shear_part1, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovMpi.cpp:      cudaFuncSetCacheConfig(kernel_mhd_compute_emf_shear,                cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovMpi.cpp:      cudaFuncSetCacheConfig(kernel_mhd_compute_emf_v4,                   cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:      cutilSafeCall( cudaMemGetInfo(&freeMemory, &totalMemory) );
src/hydro/MHDRunGodunovMpi.cpp:      std::cout << "Total memory available on GPU " << totalMemory/1000000. << " MBytes\n";
src/hydro/MHDRunGodunovMpi.cpp:      std::cout << "Currently free  memory on GPU " <<  freeMemory/1000000. << " MBytes\n";
src/hydro/MHDRunGodunovMpi.cpp:      std::cout << "Total memory allocated on GPU " << DeviceArray<real_t>::totalAllocMemoryInKB/1000. << " MBytes\n";
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_primitive_variables_3D error");
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:	godunov_unsplit_rotating_gpu(d_U , d_U2, dt, nStep);
src/hydro/MHDRunGodunovMpi.cpp:	godunov_unsplit_rotating_gpu(d_U2, d_U , dt, nStep);
src/hydro/MHDRunGodunovMpi.cpp:	godunov_unsplit_gpu(d_U , d_U2, dt, nStep);
src/hydro/MHDRunGodunovMpi.cpp:	godunov_unsplit_gpu(d_U2, d_U , dt, nStep);
src/hydro/MHDRunGodunovMpi.cpp:  } // MHDRunGodunovMpi::godunov_unsplit (GPU version)
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:  void MHDRunGodunovMpi::godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.cpp:      godunov_unsplit_gpu_v0(d_UOld, d_UNew, dt, nStep);
src/hydro/MHDRunGodunovMpi.cpp:      godunov_unsplit_gpu_v1(d_UOld, d_UNew, dt, nStep);
src/hydro/MHDRunGodunovMpi.cpp:      godunov_unsplit_gpu_v2(d_UOld, d_UNew, dt, nStep);
src/hydro/MHDRunGodunovMpi.cpp:      godunov_unsplit_gpu_v3(d_UOld, d_UNew, dt, nStep);
src/hydro/MHDRunGodunovMpi.cpp:      godunov_unsplit_gpu_v4(d_UOld, d_UNew, dt, nStep);
src/hydro/MHDRunGodunovMpi.cpp:  } // MHDRunGodunovMpi::godunov_unsplit_gpu
src/hydro/MHDRunGodunovMpi.cpp:  void MHDRunGodunovMpi::godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.cpp:      checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_godunov_unsplit_mhd_2d_v0 error",myRank);
src/hydro/MHDRunGodunovMpi.cpp:	std::cerr << "MHDRunGodunovMpi::godunov_unsplit_gpu_v0 does not implement a 3D version" << std::endl;
src/hydro/MHDRunGodunovMpi.cpp:	std::cerr << "3D GPU MHD implementation version 0 is not implemented; to do." << std::endl;
src/hydro/MHDRunGodunovMpi.cpp:  } // MHDRunGodunovMpi::godunov_unsplit_gpu_v0
src/hydro/MHDRunGodunovMpi.cpp:  void MHDRunGodunovMpi::godunov_unsplit_gpu_v0_old(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.cpp:      // checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_godunov_unsplit_mhd_2d error",myRank);
src/hydro/MHDRunGodunovMpi.cpp:	std::cerr << "MHDRunGodunovMpi::godunov_unsplit_gpu_v0_old does not implement a 3D version" << std::endl;
src/hydro/MHDRunGodunovMpi.cpp:	std::cerr << "3D GPU MHD implementation version 0 is not implemented; to do." << std::endl;
src/hydro/MHDRunGodunovMpi.cpp:  } // MHDRunGodunovMpi::godunov_unsplit_gpu_v0_old
src/hydro/MHDRunGodunovMpi.cpp:  void MHDRunGodunovMpi::godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.cpp:	checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_godunov_unsplit_mhd_2d_v1 error",myRank);
src/hydro/MHDRunGodunovMpi.cpp:	checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_mhd_2d_update_emf_v1 error",myRank);
src/hydro/MHDRunGodunovMpi.cpp:      checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_godunov_unsplit_mhd_3d_v1 error",myRank);
src/hydro/MHDRunGodunovMpi.cpp:  } // MHDRunGodunovMpi::godunov_unsplit_gpu_v1
src/hydro/MHDRunGodunovMpi.cpp:  void MHDRunGodunovMpi::godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.cpp:      std::cout << "3D GPU MHD implementation version 2 is NOT implemented !" << std::endl;
src/hydro/MHDRunGodunovMpi.cpp:  } // MHDRunGodunovMpi::godunov_unsplit_gpu_v2
src/hydro/MHDRunGodunovMpi.cpp:  void MHDRunGodunovMpi::godunov_unsplit_gpu_v3(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.cpp:  } // MHDRunGodunovMpi::godunov_unsplit_gpu_v3
src/hydro/MHDRunGodunovMpi.cpp:  void MHDRunGodunovMpi::godunov_unsplit_gpu_v4(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_mhd_compute_trace_v4 error", myRank);
src/hydro/MHDRunGodunovMpi.cpp:	    checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_mhd_compute_gravity_predictor_v4 error",myRank);
src/hydro/MHDRunGodunovMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovMpi kernel_mhd_flux_update_hydro_v4 error", myRank);
src/hydro/MHDRunGodunovMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovMpi kernel_mhd_compute_emf_v4 error",myRank);
src/hydro/MHDRunGodunovMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovMpi kernel_mhd_flux_update_ct_v4 error", myRank);
src/hydro/MHDRunGodunovMpi.cpp:  } // MHDRunGodunovMpi::godunov_unsplit_gpu_v4
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:  void MHDRunGodunovMpi::godunov_unsplit_rotating_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.cpp:	checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_mhd_compute_primitive_variables_3D error", myRank);
src/hydro/MHDRunGodunovMpi.cpp:	checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_mhd_compute_elec_field error", myRank);
src/hydro/MHDRunGodunovMpi.cpp:	checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_mhd_compute_mag_slopes error", myRank);
src/hydro/MHDRunGodunovMpi.cpp:	checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_mhd_compute_trace_v4 error", myRank);
src/hydro/MHDRunGodunovMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovMpi kernel_mhd_compute_gravity_predictor_v4 error", myRank);
src/hydro/MHDRunGodunovMpi.cpp:      	checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_mhd_flux_update_hydro_v4_shear error", myRank);
src/hydro/MHDRunGodunovMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_mhd_flux_update_hydro_v4_shear_part1 error", myRank);
src/hydro/MHDRunGodunovMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovMpi :: kernel_mhd_compute_emf_shear error", myRank);
src/hydro/MHDRunGodunovMpi.cpp:	checkCudaErrorMpi("MHDRunGodunovMpi kernel_mhd_flux_update_ct_v4 error", myRank);
src/hydro/MHDRunGodunovMpi.cpp:  } // MHDRunGodunovMpi::godunov_unsplit_rotating_gpu
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:       *  perform final remapping in shear borders on GPU
src/hydro/MHDRunGodunovMpi.cpp:	  // upload into GPU memory
src/hydro/MHDRunGodunovMpi.cpp:	  // upload into GPU memory
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:	  copyGpuToCpu(nStep);
src/hydro/MHDRunGodunovMpi.cpp:	  copyGpuToCpu(nStep);
src/hydro/MHDRunGodunovMpi.cpp:      copyGpuToCpu(nStep);
src/hydro/MHDRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.cpp:    printf("Euler MHD godunov boundaries pure GPU     [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerBoundariesGpu.elapsed(), timerBoundariesGpu.elapsed()/timerTotal.elapsed()*100.);
src/hydro/MHDRunGodunovMpi.cpp:    printf("Euler MHD godunov boundaries CPU-GPU comm [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerBoundariesCpuGpu.elapsed(), timerBoundariesCpuGpu.elapsed()/timerTotal.elapsed()*100.);
src/hydro/MHDRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/slope.h: * using the GPU version.
src/hydro/slope.h: * using the GPU version.
src/hydro/slope.h: * using the GPU version.
src/hydro/slope.h: * using the GPU version.
src/hydro/slope.h: * using the GPU version.
src/hydro/slope.h: * using the GPU version.
src/hydro/slope.h: * using the GPU version.
src/hydro/cmpdt_mhd.cuh: * \brief Provides the CUDA kernel for computing MHD time step through a
src/hydro/cmpdt_mhd.cuh:  // see CUDA documentation of the reduction example, especially the
src/hydro/cmpdt_mhd.cuh:  // see CUDA documentation of the reduction example, especially the
src/hydro/cutil.h: * \brief Some utility routines from the CUDA SDK.
src/hydro/cutil.h:* Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
src/hydro/cutil.h:* NVIDIA Corporation and its licensors retain all intellectual property and 
src/hydro/cutil.h:* agreement from NVIDIA Corporation is strictly prohibited.
src/hydro/cutil.h:/* CUda UTility Library */
src/hydro/cutil.h:    if( CUDA_SUCCESS != err) {                                               \
src/hydro/cutil.h:        fprintf(stderr, "Cuda driver error %x in file '%s' in line %i.\n",   \
src/hydro/cutil.h:    if( CUDA_SUCCESS != err) {                                               \
src/hydro/cutil.h:        fprintf(stderr, "Cuda driver error %x in file '%s' in line %i.\n",   \
src/hydro/cutil.h:#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
src/hydro/cutil.h:    cudaError err = call;                                                    \
src/hydro/cutil.h:    if( cudaSuccess != err) {                                                \
src/hydro/cutil.h:        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
src/hydro/cutil.h:                __FILE__, __LINE__, cudaGetErrorString( err) );              \
src/hydro/cutil.h:#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);                                            \
src/hydro/cutil.h:#  define CUDA_SAFE_THREAD_SYNC( ) {                                         \
src/hydro/cutil.h:    cudaError err = cudaThreadSynchronize();                                 \
src/hydro/cutil.h:    if ( cudaSuccess != err) {                                               \
src/hydro/cutil.h:        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
src/hydro/cutil.h:                __FILE__, __LINE__, cudaGetErrorString( err) );              \
src/hydro/cutil.h:    //! Check for CUDA error
src/hydro/cutil.h:    cudaError_t err = cudaGetLastError();                                    \
src/hydro/cutil.h:    if( cudaSuccess != err) {                                                \
src/hydro/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
src/hydro/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
src/hydro/cutil.h:    err = cudaThreadSynchronize();                                           \
src/hydro/cutil.h:    if( cudaSuccess != err) {                                                \
src/hydro/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
src/hydro/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
src/hydro/cutil.h:    cudaError_t err = cudaGetLastError();                                    \
src/hydro/cutil.h:    if( cudaSuccess != err) {                                                \
src/hydro/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
src/hydro/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
src/hydro/cutil.h:    CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));                \
src/hydro/cutil.h:        fprintf(stderr, "cutil error: no devices supporting CUDA.\n");       \
src/hydro/cutil.h:    cudaDeviceProp deviceProp;                                               \
src/hydro/cutil.h:    CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, dev));       \
src/hydro/cutil.h:        fprintf(stderr, "cutil error: device does not support CUDA.\n");     \
src/hydro/cutil.h:    CUDA_SAFE_CALL(cudaSetDevice(dev));                                      \
src/hydro/cutil.h:    //! Check for CUDA context lost
src/hydro/cutil.h:#  define CUDA_CHECK_CTX_LOST(errorMessage) {                                \
src/hydro/cutil.h:    cudaError_t err = cudaGetLastError();                                    \
src/hydro/cutil.h:    if( cudaSuccess != err) {                                                \
src/hydro/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
src/hydro/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
src/hydro/cutil.h:    err = cudaThreadSynchronize();                                           \
src/hydro/cutil.h:    if( cudaSuccess != err) {                                                \
src/hydro/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
src/hydro/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
src/hydro/cutil.h://! Check for CUDA context lost
src/hydro/cutil.h:    cudaError_t err = cudaGetLastError();                                    \
src/hydro/cutil.h:    if( CUDA_ERROR_INVALID_CONTEXT != err) {                                 \
src/hydro/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
src/hydro/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
src/hydro/cutil.h:    err = cudaThreadSynchronize();                                           \
src/hydro/cutil.h:    if( cudaSuccess != err) {                                                \
src/hydro/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
src/hydro/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
src/hydro/cutil.h:    if (CUDA_SUCCESS == err)                                                 \
src/hydro/cutil.h:        fprintf(stderr, "cutil error: no devices supporting CUDA\n");        \
src/hydro/base_type.h: * On GPU declaring an array results in register spilling (since register are not 
src/hydro/gravity_zslab.cuh: * \brief CUDA kernel for computing gravity forces (adapted from Dumses).
src/hydro/gravity_zslab.cuh: * CUDA kernel computing gravity predictor (3D data).
src/hydro/gravity_zslab.cuh: * CUDA kernel computing gravity source term (3D data).
src/hydro/HydroRunGodunovZslabMpi.h: * MPI+CUDA computations.
src/hydro/HydroRunGodunovZslabMpi.h:#include "gpu_macros.h"
src/hydro/HydroRunGodunovZslabMpi.h:    //! see godunov_unsplit_gpu or godunov_unsplit_cpu.
src/hydro/HydroRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.h:    //! scheme on GPU, two array are necessary to make ping-pong (d_UOld and
src/hydro/HydroRunGodunovZslabMpi.h:    void godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslabMpi.h:    void godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslabMpi.h:    void godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslabMpi.h:    void godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslabMpi.h:#endif // __CUDAC__
src/hydro/HydroRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.h:    DeviceArray<real_t> d_Q; //!< GPU : primitive data array
src/hydro/HydroRunGodunovZslabMpi.h:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.h:    DeviceArray<real_t> d_qm_x; //!< GPU array for qm state along X
src/hydro/HydroRunGodunovZslabMpi.h:    DeviceArray<real_t> d_qm_y; //!< GPU array for qm state along Y
src/hydro/HydroRunGodunovZslabMpi.h:    DeviceArray<real_t> d_qm_z; //!< GPU array for qm state along Z
src/hydro/HydroRunGodunovZslabMpi.h:    DeviceArray<real_t> d_qp_x; //!< GPU array for qp state along X
src/hydro/HydroRunGodunovZslabMpi.h:    DeviceArray<real_t> d_qp_y; //!< GPU array for qp state along Y
src/hydro/HydroRunGodunovZslabMpi.h:    DeviceArray<real_t> d_qp_z; //!< GPU array for qp state along Z
src/hydro/HydroRunGodunovZslabMpi.h:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.h:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.h:    CudaTimer timerGodunov;
src/hydro/HydroRunGodunovZslabMpi.h:    CudaTimer timerPrimVar;
src/hydro/HydroRunGodunovZslabMpi.h:    CudaTimer timerSlopeTrace;
src/hydro/HydroRunGodunovZslabMpi.h:    CudaTimer timerUpdate;
src/hydro/HydroRunGodunovZslabMpi.h:    CudaTimer timerDissipative;
src/hydro/HydroRunGodunovZslabMpi.h:#endif // __CUDACC__
src/hydro/mpiBorderUtils.cuh: * \brief Provides the CUDA kernel for copying border buffer between
src/hydro/mpiBorderUtils.cuh:// number of cuda threads per block
src/hydro/mpiBorderUtils.cuh: * cuda kernel for copying 1D border buffer (PITCHED memory type) to 2d array
src/hydro/mpiBorderUtils.cuh: * cuda kernel for copying 1D border buffer (LINEAR memory type) to 2d array
src/hydro/mpiBorderUtils.cuh: * cuda kernel for copying 2D border (PITCHED memory type) buffer to 3d array
src/hydro/mpiBorderUtils.cuh: * cuda kernel for copying 2D border buffer (LINEAR memory type) to 3d array
src/hydro/mpiBorderUtils.cuh: * cuda kernel for copying 2d array to 1D border buffer (PITCHED
src/hydro/mpiBorderUtils.cuh: * cuda kernel for copying 2d array to 1D border buffer (LINEAR memory type)
src/hydro/mpiBorderUtils.cuh: * cuda kernel for copying 3d array to 2D border buffer (PITCHED
src/hydro/mpiBorderUtils.cuh: * cuda kernel for copying 3d array to 2D border buffer (LINEAR memory type)
src/hydro/Forcing_OrnsteinUhlenbeck_kernels.cuh: * \brief Provides the CUDA kernels for performing forcing field update.
src/hydro/Forcing_OrnsteinUhlenbeck_kernels.cuh: * CUDA kernel for forcing field mode update.
src/hydro/Forcing_OrnsteinUhlenbeck_kernels.cuh: * CUDA kernel for forcing field mode update.
src/hydro/Forcing_OrnsteinUhlenbeck_kernels.cuh: * CUDA kernel for forcing field add.
src/hydro/mpiBorderUtils.h:#ifdef __CUDACC__
src/hydro/mpiBorderUtils.h:#endif // __CUDACC__
src/hydro/mpiBorderUtils.h:#ifdef __CUDACC__
src/hydro/mpiBorderUtils.h:   * GPU copy border buf routines
src/hydro/mpiBorderUtils.h:      } else { // bTemp was allocated with cudaMalloc
src/hydro/mpiBorderUtils.h:      } else { // bTemp was allocated with cudaMalloc
src/hydro/mpiBorderUtils.h:   * This is a wrapper to call the CUDA kernel that actually copy
src/hydro/mpiBorderUtils.h:      } else { // bTemp was allocated with cudaMalloc
src/hydro/mpiBorderUtils.h:      } else { // bTemp was allocated with cudaMalloc
src/hydro/mpiBorderUtils.h:#endif // __CUDACC__
src/hydro/resistivity.cuh: * \brief CUDA kernel for computing resistivity forces (MHD only, adapted from Dumses).
src/hydro/resistivity.cuh: * CUDA kernel computing resistivity forces (2D data).
src/hydro/resistivity.cuh: * CUDA kernel computing resistivity forces (2D data).
src/hydro/resistivity.cuh: * CUDA kernel computing resistivity forces (3D data).
src/hydro/resistivity.cuh: * CUDA kernel computing resistivity forces (3D data).
src/hydro/MHDRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.h:#include "../utils/monitoring/CudaTimer.h"
src/hydro/MHDRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.h:    //! In the GPU version, the conversion is done on line, inside
src/hydro/MHDRunGodunovZslab.h:    //! see godunov_unsplit_gpu or godunov_unsplit_cpu.
src/hydro/MHDRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.h:    //! scheme on GPU, two array are necessary to make ping-pong (d_UOld and
src/hydro/MHDRunGodunovZslab.h:    void godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovZslab.h:#endif // __CUDAC__
src/hydro/MHDRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.h:    //! Same routine as godunov_unsplit_gpu but with rotating
src/hydro/MHDRunGodunovZslab.h:    void godunov_unsplit_rotating_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.h:    DeviceArray<real_t> d_Q; //!< GPU : primitive data array
src/hydro/MHDRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.h:    DeviceArray<real_t> d_qm_x; //!< GPU array for qm state along X
src/hydro/MHDRunGodunovZslab.h:    DeviceArray<real_t> d_qm_y; //!< GPU array for qm state along Y
src/hydro/MHDRunGodunovZslab.h:    DeviceArray<real_t> d_qm_z; //!< GPU array for qm state along Z
src/hydro/MHDRunGodunovZslab.h:    DeviceArray<real_t> d_qp_x; //!< GPU array for qp state along X
src/hydro/MHDRunGodunovZslab.h:    DeviceArray<real_t> d_qp_y; //!< GPU array for qp state along Y
src/hydro/MHDRunGodunovZslab.h:    DeviceArray<real_t> d_qp_z; //!< GPU array for qp state along Z
src/hydro/MHDRunGodunovZslab.h:    DeviceArray<real_t> d_emf; //!< GPU array for electromotive forces
src/hydro/MHDRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerBoundaries;
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerGodunov;
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerTraceUpdate;
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerUpdate;
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerEmf;
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerDissipative;
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerPrimVar;
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerElecField;
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerMagSlopes;
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerTrace;
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerHydroShear;
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerRemapping;
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerShearBorder;
src/hydro/MHDRunGodunovZslab.h:    CudaTimer timerCtUpdate;
src/hydro/MHDRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:// include CUDA kernel when necessary
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:	d_Q.allocate   (make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	d_Q.allocate   (make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:	  d_qm_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_qm_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_qp_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_qp_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_qm_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_qm_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_qm_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_qp_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_qp_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_qp_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:	  d_slope_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_slope_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_qm.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_qp.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_slope_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_slope_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_slope_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_qm.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	  d_qp.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:	d_debug.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	d_debug2.allocate(make_uint3(isize, jsize, 2)   , gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:	d_debug.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:	d_debug2.allocate(make_uint4(isize, jsize, ksize, 3)   , gpuMemAllocType);
src/hydro/HydroRunGodunov.cpp:    //     // GPU execution settings
src/hydro/HydroRunGodunov.cpp:    // #ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:    // #if __CUDA_ARCH__ >= 200
src/hydro/HydroRunGodunov.cpp:    //     cudaFuncSetCacheConfig(kernel_hydro_compute_trace_unsplit_3d_v2, cudaFuncCachePreferL1);
src/hydro/HydroRunGodunov.cpp:    // #endif // __CUDA_ARCH__ >= 200
src/hydro/HydroRunGodunov.cpp:    // #endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:      cutilSafeCall( cudaMemGetInfo(&freeMemory, &totalMemory) );
src/hydro/HydroRunGodunov.cpp:      std::cout << "Total memory available on GPU " << totalMemory/1000000. << " MBytes\n";
src/hydro/HydroRunGodunov.cpp:      std::cout << "Currently free  memory on GPU " <<  freeMemory/1000000. << " MBytes\n";
src/hydro/HydroRunGodunov.cpp:      std::cout << "Total memory allocated on GPU " << DeviceArray<real_t>::totalAllocMemoryInKB/1000. << " MBytes\n";
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U , d_U2, XDIR, dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U2, d_U , YDIR, dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U , d_U2, YDIR, dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U2, d_U , XDIR, dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U , d_U2, XDIR, dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U2, d_U , YDIR, dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U , d_U2, ZDIR, dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U2, d_U , YDIR,dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U , d_U2, ZDIR,dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U2, d_U , XDIR,dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U , d_U2, ZDIR,dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U2, d_U , YDIR,dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U , d_U2, XDIR,dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U2, d_U , XDIR, dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U , d_U2, YDIR, dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U2, d_U , ZDIR, dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U , d_U2, YDIR,dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U2, d_U , ZDIR,dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U , d_U2, XDIR,dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U2, d_U , ZDIR,dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U , d_U2, YDIR,dt);
src/hydro/HydroRunGodunov.cpp:	godunov_split_gpu(d_U2, d_U , XDIR,dt);
src/hydro/HydroRunGodunov.cpp:  } // HydroRunGodunov::godunov_split (GPU version)
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:      godunov_unsplit_gpu(d_U , d_U2, dt, nStep);
src/hydro/HydroRunGodunov.cpp:      godunov_unsplit_gpu(d_U2, d_U , dt, nStep);
src/hydro/HydroRunGodunov.cpp:  } // HydroRunGodunov::godunov_unsplit (GPU version)
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__ 
src/hydro/HydroRunGodunov.cpp:  void HydroRunGodunov::godunov_split_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_x_2d_v2 error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_x_2d_v1 error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_x_notrace_2d error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_y_2d_v2 error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_y_2d_v1 error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_y_notrace_2d error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_x_3d_v2 error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_x_3d_v1 error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_x_notrace_3d error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_y_3d_v2 error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_y_3d_v1 error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_y_notrace_3d error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_z_3d_v2 error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_z_3d_v1 error");
src/hydro/HydroRunGodunov.cpp:	    checkCudaError("HydroRunGodunov :: kernel godunov_z_notrace_3d error");
src/hydro/HydroRunGodunov.cpp:  } // HydroRunGodunov::godunov_split_gpu
src/hydro/HydroRunGodunov.cpp:  void HydroRunGodunov::godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunov.cpp:      godunov_unsplit_gpu_v0(d_UOld, d_UNew, dt, nStep);
src/hydro/HydroRunGodunov.cpp:      godunov_unsplit_gpu_v1(d_UOld, d_UNew, dt, nStep);
src/hydro/HydroRunGodunov.cpp:      godunov_unsplit_gpu_v2(d_UOld, d_UNew, dt, nStep);
src/hydro/HydroRunGodunov.cpp:  } // HydroRunGodunov::godunov_unsplit_gpu
src/hydro/HydroRunGodunov.cpp:  void HydroRunGodunov::godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunov.cpp:      // checkCudaError("HydroRunGodunov :: kernel_godunov_unsplit_2d error");
src/hydro/HydroRunGodunov.cpp:	checkCudaError("HydroRunGodunov :: kernel_godunov_unsplit_2d_v0 error");
src/hydro/HydroRunGodunov.cpp:      // checkCudaError("HydroRunGodunov :: kernel_godunov_unsplit_3d error");
src/hydro/HydroRunGodunov.cpp:      checkCudaError("HydroRunGodunov :: kernel_godunov_unsplit_3d_v0 error");
src/hydro/HydroRunGodunov.cpp:  } // HydroRunGodunov::godunov_unsplit_gpu_v0
src/hydro/HydroRunGodunov.cpp:  void HydroRunGodunov::godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunov.cpp:	checkCudaError("HydroRunGodunov :: kernel_hydro_compute_trace_unsplit_2d_v1 error");
src/hydro/HydroRunGodunov.cpp:	checkCudaError("HydroRunGodunov :: kernel_hydro_flux_update_unsplit_2d_v1< error");
src/hydro/HydroRunGodunov.cpp:	checkCudaError("HydroRunGodunov :: kernel_hydro_compute_trace_unsplit_3d_v1 error");
src/hydro/HydroRunGodunov.cpp:	checkCudaError("HydroRunGodunov :: kernel_hydro_flux_update_unsplit_3d_v1 error");
src/hydro/HydroRunGodunov.cpp:  } // HydroRunGodunov::godunov_unsplit_gpu_v1
src/hydro/HydroRunGodunov.cpp:  void HydroRunGodunov::godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunov.cpp:  } // HydroRunGodunov::godunov_unsplit_gpu_v2
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:#ifndef __CUDACC__
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:#ifndef __CUDACC__
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:#ifndef __CUDACC__
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:	  copyGpuToCpu(nStep);
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:	  copyGpuToCpu(nStep);
src/hydro/HydroRunGodunov.cpp:      copyGpuToCpu(nStep);
src/hydro/HydroRunGodunov.cpp:#ifndef __CUDACC__
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.cpp:	checkCudaError("HydroRunGodunov :: kernel_hydro_compute_primitive_variables_2D error");
src/hydro/HydroRunGodunov.cpp:	checkCudaError("HydroRunGodunov :: kernel_hydro_compute_primitive_variables_3D error");
src/hydro/HydroRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunLaxLiu.cpp:// include CUDA kernel when necessary
src/hydro/HydroRunLaxLiu.cpp:#ifdef __CUDACC__
src/hydro/HydroRunLaxLiu.cpp:#endif // __CUDACC__
src/hydro/HydroRunLaxLiu.cpp:#ifdef __CUDACC__
src/hydro/HydroRunLaxLiu.cpp:#endif // __CUDACC__
src/hydro/HydroRunLaxLiu.cpp:#ifdef __CUDACC__
src/hydro/HydroRunLaxLiu.cpp:#endif // __CUDACC__
src/hydro/HydroRunLaxLiu.cpp:#ifdef __CUDACC__
src/hydro/HydroRunLaxLiu.cpp:#endif // __CUDACC__
src/hydro/HydroRunLaxLiu.cpp:#ifdef __CUDACC__
src/hydro/HydroRunLaxLiu.cpp:#ifdef __CUDACC__
src/hydro/HydroRunLaxLiu.cpp:#endif // __CUDACC__
src/hydro/HydroRunLaxLiu.cpp:#ifdef __CUDACC__
src/hydro/HydroRunLaxLiu.cpp:#endif // __CUDACC__
src/hydro/constoprim.h: * CUDA kernel).
src/hydro/constoprim.h:#ifndef __CUDACC__
src/hydro/godunov.cuh: * \brief Defines the CUDA kernel for the actual Godunov scheme
src/hydro/trace_mhd.h:  // instead of re-declaring new variables, better for the GPU
src/hydro/trace_mhd.h:  // instead of re-declaring new variables, better for the GPU
src/hydro/trace_mhd.h: *   separate CUDA kernel as for the GPU version), so it is now an input 
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#ifdef __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#endif // __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#ifdef __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMemcpyToSymbol(::gParams, &_gParams, sizeof(GlobalConstants), 0, cudaMemcpyHostToDevice ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#endif // __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#ifdef __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMalloc((void**) &d_mode        , nDim*nMode*sizeof(double)) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMalloc((void**) &d_forcingField, nDim*nMode*sizeof(double)) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMalloc((void**) &d_projTens    , nDim*nDim*nMode*sizeof(double)) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMalloc((void**) &deviceStates  , nMode * sizeof(curandState) ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#ifdef __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaFree(d_mode        ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaFree(d_forcingField) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaFree(d_projTens    ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaFree(deviceStates  ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#endif // __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#ifdef __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    // copy parameters into GPU memory
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMemcpy( d_mode,         mode,         
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:			       nDim*nMode*sizeof(double), cudaMemcpyHostToDevice ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMemcpy( d_forcingField, forcingField, 
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:			       nDim*nMode*sizeof(double), cudaMemcpyHostToDevice ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMemcpy( d_projTens,     projTens,
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:			       nDim*nDim*nMode*sizeof(double), cudaMemcpyHostToDevice ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    // initialize random generator on GPU
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#endif // __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#ifdef __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    // copy parameters from GPU memory
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMemcpy( mode,         d_mode,         
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:			       nDim*nMode*sizeof(double), cudaMemcpyDeviceToHost ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMemcpy( forcingField, d_forcingField, 
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:			       nDim*nMode*sizeof(double), cudaMemcpyDeviceToHost ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMemcpy( projTens,     d_projTens,
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:			       nDim*nDim*nMode*sizeof(double), cudaMemcpyDeviceToHost ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#endif // __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#ifdef __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    // copy parameters into GPU memory
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMemcpy( d_mode,         mode,         
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:			       nDim*nMode*sizeof(double), cudaMemcpyHostToDevice ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMemcpy( d_forcingField, forcingField, 
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:			       nDim*nMode*sizeof(double), cudaMemcpyHostToDevice ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    cutilSafeCall( cudaMemcpy( d_projTens,     projTens,
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:			       nDim*nDim*nMode*sizeof(double), cudaMemcpyHostToDevice ) );
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#endif // __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#ifdef __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:    // GPU version (1D parallelization of iMode loop)
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#endif // __CUDACC__    
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#ifdef __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:  } // ForcingOrnsteinUhlenbeck::add_forcing_field // GPU version
src/hydro/Forcing_OrnsteinUhlenbeck.cpp:#endif // __CUDACC__
src/hydro/slope_mhd.h: * using the GPU version.
src/hydro/slope_mhd.h: * using the GPU version.
src/hydro/slope_mhd.h: * using the GPU version.
src/hydro/slope_mhd.h: * using the GPU version.
src/hydro/slope_mhd.h: * using the GPU version.
src/hydro/viscosity.cuh: * \brief CUDA kernel for computing viscosity forces (adapted from Dumses).
src/hydro/viscosity.cuh: * CUDA kernel computing viscosity forces (2D data).
src/hydro/viscosity.cuh: * CUDA kernel computing viscosity forces (3D data).
src/hydro/MHDRunGodunovZslab.cpp:// include CUDA kernel when necessary
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:    d_Q.allocate (make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:    d_emf.allocate(make_uint4(isize, jsize, zSlabWidthG, 3    ), gpuMemAllocType); // 3 EMF's
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:    d_qm_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qm_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qm_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qp_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qp_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qp_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qEdge_RT.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qEdge_RB.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qEdge_LT.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qEdge_LB.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qEdge_RT2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qEdge_RB2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qEdge_LT2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qEdge_LB2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qEdge_RT3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qEdge_RB3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qEdge_LT3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_qEdge_LB3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_elec.allocate (make_uint4(isize, jsize, zSlabWidthG, 3), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_dA.allocate   (make_uint4(isize, jsize, zSlabWidthG, 3), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_dB.allocate   (make_uint4(isize, jsize, zSlabWidthG, 3), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:    d_dC.allocate   (make_uint4(isize, jsize, zSlabWidthG, 3), gpuMemAllocType);	
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:      d_shear_flux_xmin.allocate      (make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:      d_shear_flux_xmax.allocate      (make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:      d_shear_flux_xmin_remap.allocate(make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:      d_shear_flux_xmax_remap.allocate(make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:      d_shear_border_xmin.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:      d_shear_border_xmax.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:      d_shear_slope_xmin.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:      d_shear_slope_xmax.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:     * memory allocation for GPU routines debugging
src/hydro/MHDRunGodunovZslab.cpp:    // GPU execution settings
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:       cudaFuncSetCacheConfig(kernel_mhd_compute_trace_v4_zslab, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovZslab.cpp:       cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_zslab, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovZslab.cpp:       cudaFuncSetCacheConfig(kernel_mhd_compute_emf_v4_zslab, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovZslab.cpp:       cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_shear_zslab, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovZslab.cpp:       cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_shear_part1_zslab, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovZslab.cpp:       cudaFuncSetCacheConfig(kernel_mhd_compute_emf_shear_zslab, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:      cutilSafeCall( cudaMemGetInfo(&freeMemory, &totalMemory) );
src/hydro/MHDRunGodunovZslab.cpp:      std::cout << "Total memory available on GPU " << totalMemory/1000000. << " MBytes\n";
src/hydro/MHDRunGodunovZslab.cpp:      std::cout << "Currently free  memory on GPU " <<  freeMemory/1000000. << " MBytes\n";
src/hydro/MHDRunGodunovZslab.cpp:      std::cout << "Total memory allocated on GPU " << DeviceArray<real_t>::totalAllocMemoryInKB/1000. << " MBytes\n";
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#ifndef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:	godunov_unsplit_rotating_gpu(d_U , d_U2, dt, nStep);
src/hydro/MHDRunGodunovZslab.cpp:	godunov_unsplit_rotating_gpu(d_U2, d_U , dt, nStep);
src/hydro/MHDRunGodunovZslab.cpp:	godunov_unsplit_gpu(d_U , d_U2, dt, nStep);
src/hydro/MHDRunGodunovZslab.cpp:	godunov_unsplit_gpu(d_U2, d_U , dt, nStep);
src/hydro/MHDRunGodunovZslab.cpp:  } // MHDRunGodunovZslab::godunov_unsplit (GPU version)
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:  void MHDRunGodunovZslab::godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovZslab.cpp:	  checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_primitive_variables_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	  checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_elec_field_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	  checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_mag_slopes_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	  checkCudaError("MHDRunGodunovZslab kernel_mhd_compute_trace_v4_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	    checkCudaError("MHDRunGodunov kernel_mhd_compute_gravity_predictor_v4_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	  checkCudaError("MHDRunGodunovZslab kernel_mhd_flux_update_hydro_v4_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	  checkCudaError("MHDRunGodunovZslab kernel_mhd_compute_emf_v4_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	  checkCudaError("MHDRunGodunovZslab kernel_mhd_flux_update_ct_v4_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:  } // MHDRunGodunovZslab::godunov_unsplit_gpu
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:  void MHDRunGodunovZslab::godunov_unsplit_rotating_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovZslab.cpp:	checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_primitive_variables_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_elec_field_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_mag_slopes_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_trace_v4_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	  checkCudaError("MHDRunGodunovZslab kernel_mhd_compute_gravity_predictor_v4_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	checkCudaError("MHDRunGodunovZslab :: kernel_mhd_flux_update_hydro_v4_shear_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	  checkCudaError("MHDRunGodunovZslab :: kernel_mhd_flux_update_hydro_v4_shear_part1_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	    checkCudaError("MHDRunGodunovZslab :: kernel_mhd_compute_emf_shear_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:	checkCudaError("MHDRunGodunovZslab kernel_mhd_flux_update_ct_v4_zslab error");
src/hydro/MHDRunGodunovZslab.cpp:  } // MHDRunGodunovZslab::godunov_unsplit_rotating_gpu
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:	outputHdf5Debug(h_shear_debug, "gpu_shear_border_xmin_", nStep);
src/hydro/MHDRunGodunovZslab.cpp:	outputHdf5Debug(h_shear_debug, "gpu_shear_border_xmax_", nStep);
src/hydro/MHDRunGodunovZslab.cpp:	  outputHdf5Debug(h_shear_debug, "gpu_shear_slope_xmin_", nStep);
src/hydro/MHDRunGodunovZslab.cpp:	  outputHdf5Debug(h_shear_debug, "gpu_shear_slope_xmax_", nStep);
src/hydro/MHDRunGodunovZslab.cpp:	  outputHdf5Debug(h_shear_debug, "gpu_after_final_remapping_", nStep);
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:  } // MHDRunGodunovZslab::make_all_boundaries_shear -- GPU version
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslab.cpp:	  copyGpuToCpu(nStep);
src/hydro/MHDRunGodunovZslab.cpp:	  copyGpuToCpu(nStep);
src/hydro/MHDRunGodunovZslab.cpp:      copyGpuToCpu(nStep);
src/hydro/MHDRunGodunov.cpp:// include CUDA kernel when necessary
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifndef __CUDACC__
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:      d_emf.allocate(make_uint3(isize, jsize, 1           ), gpuMemAllocType); // only EMFZ
src/hydro/MHDRunGodunov.cpp:      d_emf.allocate(make_uint4(isize, jsize, ksize, 3    ), gpuMemAllocType); // 3 EMF's
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:	d_qm_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	d_qm_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	d_qEdge_RT.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	d_qEdge_RB.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	d_qEdge_LT.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qm_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType); // 2D array
src/hydro/MHDRunGodunov.cpp:	  d_qm_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType); // 2D array
src/hydro/MHDRunGodunov.cpp:	  d_qm_z.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType); // 3D array with only 2 plans
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_RT.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_RB.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_LT.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_RT2.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_RB2.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_LT2.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_RT3.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_RB3.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_LT3.allocate(make_uint4(isize, jsize, 2, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qm_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qm_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qm_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qp_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qp_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qp_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_RT.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_RB.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_LT.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_LB.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_RT2.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_RB2.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_LT2.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_LB2.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_RT3.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_RB3.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_LT3.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	  d_qEdge_LB3.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:	//d_elec.allocate(make_uint3(isize, jsize, 1), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	d_elec.allocate (make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	d_dA.allocate   (make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	d_dB.allocate   (make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	d_dC.allocate   (make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);	
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:      d_Q.allocate (make_uint3(isize, jsize,        nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:      d_Q.allocate (make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:#endif // __UCUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:      	d_shear_flux_xmin.allocate      (make_uint3(jsize,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:      	d_shear_flux_xmax.allocate      (make_uint3(jsize,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:      	d_shear_flux_xmin_remap.allocate(make_uint3(jsize,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:      	d_shear_flux_xmax_remap.allocate(make_uint3(jsize,ksize,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	d_shear_border_xmin.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	d_shear_border_xmax.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:	d_shear_slope_xmin.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp: 	d_shear_slope_xmax.allocate(make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:     * memory allocation for GPU routines debugging
src/hydro/MHDRunGodunov.cpp:    // GPU execution settings
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:      cudaFuncSetCacheConfig(kernel_mhd_compute_trace_v4, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunov.cpp:      cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunov.cpp:      cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_shear, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunov.cpp:      cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_shear_part1, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunov.cpp:      cudaFuncSetCacheConfig(kernel_mhd_compute_emf_v4, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:      cutilSafeCall( cudaMemGetInfo(&freeMemory, &totalMemory) );
src/hydro/MHDRunGodunov.cpp:      std::cout << "Total memory available on GPU " << totalMemory/1000000. << " MBytes\n";
src/hydro/MHDRunGodunov.cpp:      std::cout << "Currently free  memory on GPU " <<  freeMemory/1000000. << " MBytes\n";
src/hydro/MHDRunGodunov.cpp:      std::cout << "Total memory allocated on GPU " << DeviceArray<real_t>::totalAllocMemoryInKB/1000. << " MBytes\n";
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_primitive_variables_2D error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_primitive_variables_3D error");
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:	godunov_unsplit_rotating_gpu(d_U , d_U2, dt, nStep);
src/hydro/MHDRunGodunov.cpp:	godunov_unsplit_rotating_gpu(d_U2, d_U , dt, nStep);
src/hydro/MHDRunGodunov.cpp:	godunov_unsplit_gpu(d_U , d_U2, dt, nStep);
src/hydro/MHDRunGodunov.cpp:	godunov_unsplit_gpu(d_U2, d_U , dt, nStep);
src/hydro/MHDRunGodunov.cpp:  } // MHDRunGodunov::godunov_unsplit (GPU version)
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:  void MHDRunGodunov::godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.cpp:      godunov_unsplit_gpu_v0(d_UOld, d_UNew, dt, nStep);
src/hydro/MHDRunGodunov.cpp:      godunov_unsplit_gpu_v1(d_UOld, d_UNew, dt, nStep);
src/hydro/MHDRunGodunov.cpp:      godunov_unsplit_gpu_v2(d_UOld, d_UNew, dt, nStep);
src/hydro/MHDRunGodunov.cpp:      godunov_unsplit_gpu_v3(d_UOld, d_UNew, dt, nStep);
src/hydro/MHDRunGodunov.cpp:      godunov_unsplit_gpu_v4(d_UOld, d_UNew, dt, nStep);
src/hydro/MHDRunGodunov.cpp:  } // MHDRunGodunov::godunov_unsplit_gpu
src/hydro/MHDRunGodunov.cpp:  void MHDRunGodunov::godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.cpp:      checkCudaError("MHDRunGodunov :: kernel_godunov_unsplit_mhd_2d_v0 error");
src/hydro/MHDRunGodunov.cpp:      std::cerr << "MHDRunGodunov::godunov_unsplit_gpu_v0 does not implement a 3D version" << std::endl;
src/hydro/MHDRunGodunov.cpp:      std::cerr << "3D GPU MHD implementation version 0 is not implemented; to do." << std::endl;
src/hydro/MHDRunGodunov.cpp:  } // MHDRunGodunov::godunov_unsplit_gpu_v0
src/hydro/MHDRunGodunov.cpp:  void MHDRunGodunov::godunov_unsplit_gpu_v0_old(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.cpp:    //   checkCudaError("MHDRunGodunov :: kernel_godunov_unsplit_mhd_2d_v0_old error");
src/hydro/MHDRunGodunov.cpp:    //   std::cerr << "MHDRunGodunov::godunov_unsplit_gpu_v0 does not implement a 3D version" << std::endl;
src/hydro/MHDRunGodunov.cpp:    //   std::cerr << "3D GPU MHD implementation version 0 Will never be implemented ! To poor performances expected." << std::endl;
src/hydro/MHDRunGodunov.cpp:  } // MHDRunGodunov::godunov_unsplit_gpu_v0_old
src/hydro/MHDRunGodunov.cpp:  void MHDRunGodunov::godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_godunov_unsplit_mhd_2d_v1 error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_2d_update_emf_v1 error");
src/hydro/MHDRunGodunov.cpp:      checkCudaError("MHDRunGodunov :: kernel_godunov_unsplit_mhd_3d_v1 error");
src/hydro/MHDRunGodunov.cpp:  } // MHDRunGodunov::godunov_unsplit_gpu_v1
src/hydro/MHDRunGodunov.cpp:  void MHDRunGodunov::godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.cpp:    std::cerr << "2D / 3D GPU MHD implementation version 2 are NOT implemented !" << std::endl;
src/hydro/MHDRunGodunov.cpp:  } // MHDRunGodunov::godunov_unsplit_gpu_v2
src/hydro/MHDRunGodunov.cpp:  void MHDRunGodunov::godunov_unsplit_gpu_v3(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.cpp:      	std::cerr << "MHD on GPU version 3 is not implemented for 2D MHD.\n";
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_elec_field error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_mag_slopes error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_trace error");
src/hydro/MHDRunGodunov.cpp:  } // MHDRunGodunov::godunov_unsplit_gpu_v3
src/hydro/MHDRunGodunov.cpp:  void MHDRunGodunov::godunov_unsplit_gpu_v4(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.cpp:      	std::cerr << "MHD on GPU version 4 is not implemented for 2D MHD.\n";
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_elec_field error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_mag_slopes error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov kernel_mhd_compute_trace_v4 error");
src/hydro/MHDRunGodunov.cpp:	  checkCudaError("MHDRunGodunov kernel_mhd_compute_gravity_predictor_v4 error");
src/hydro/MHDRunGodunov.cpp:      //   checkCudaError("MHDRunGodunov kernel_mhd_flux_update_hydro_v4 error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov kernel_mhd_flux_update_hydro_v4 error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov kernel_mhd_compute_emf_v4 error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov kernel_mhd_flux_update_ct_v4 error");
src/hydro/MHDRunGodunov.cpp:  } // MHDRunGodunov::godunov_unsplit_gpu_v4
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:  void MHDRunGodunov::godunov_unsplit_rotating_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.cpp:      checkCudaError("MHDRunGodunov :: kernel_godunov_unsplit_mhd_rotating_2d_v1 error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_primitive_variables_3D error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_elec_field error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_mag_slopes error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_compute_trace_v4 error");
src/hydro/MHDRunGodunov.cpp:	  checkCudaError("MHDRunGodunov kernel_mhd_compute_gravity_predictor_v4 error");
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov :: kernel_mhd_flux_update_hydro_v4_shear error");
src/hydro/MHDRunGodunov.cpp:	  checkCudaError("MHDRunGodunov :: kernel_mhd_flux_update_hydro_v4_shear_part1 error");
src/hydro/MHDRunGodunov.cpp:	  checkCudaError("MHDRunGodunov :: kernel_mhd_compute_emf_shear error");
src/hydro/MHDRunGodunov.cpp:	  outputHdf5Debug(h_shear_debug, "gpu_shear_UNew_update2_", nStep);
src/hydro/MHDRunGodunov.cpp:	checkCudaError("MHDRunGodunov kernel_mhd_flux_update_ct_v4 error");
src/hydro/MHDRunGodunov.cpp:  } // MHDRunGodunov::godunov_unsplit_rotating_gpu
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:	outputHdf5Debug(h_shear_debug, "gpu_shear_border_xmin_", nStep);
src/hydro/MHDRunGodunov.cpp:	outputHdf5Debug(h_shear_debug, "gpu_shear_border_xmax_", nStep);
src/hydro/MHDRunGodunov.cpp:	  outputHdf5Debug(h_shear_debug, "gpu_shear_slope_xmin_", nStep);
src/hydro/MHDRunGodunov.cpp:	  outputHdf5Debug(h_shear_debug, "gpu_shear_slope_xmax_", nStep);
src/hydro/MHDRunGodunov.cpp:	  outputHdf5Debug(h_shear_debug, "gpu_after_final_remapping_", nStep);
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:  } // MHDRunGodunov::make_all_boundaries_shear -- GPU version
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:	  copyGpuToCpu(nStep);
src/hydro/MHDRunGodunov.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunov.cpp:	  copyGpuToCpu(nStep);
src/hydro/MHDRunGodunov.cpp:      copyGpuToCpu(nStep);
src/hydro/MHDRunGodunov.cpp:#ifndef __CUDACC__
src/hydro/MHDRunGodunov.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunov.h: * In the process of removing the coupling between fortran and Cuda; the 
src/hydro/HydroRunGodunov.h: * Further (in October 2010), this file is redesigned to handle both MPI+Cuda 
src/hydro/HydroRunGodunov.h:#include "gpu_macros.h"
src/hydro/HydroRunGodunov.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.h:#include "../utils/monitoring/CudaTimer.h"
src/hydro/HydroRunGodunov.h:#endif // __CUDACC__
src/hydro/HydroRunGodunov.h:#ifndef __CUDACC__
src/hydro/HydroRunGodunov.h:#endif // __CUDACC__
src/hydro/HydroRunGodunov.h:   * available on GPU (2D + 3D))
src/hydro/HydroRunGodunov.h:    //! see godunov_unsplit_gpu or godunov_unsplit_cpu.
src/hydro/HydroRunGodunov.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.h:    //! Actual computation of the godunov integration on GPU using
src/hydro/HydroRunGodunov.h:    void godunov_split_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunov.h:    //! scheme on GPU, two array are necessary to make ping-pong (d_UOld and
src/hydro/HydroRunGodunov.h:    void godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunov.h:    void godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunov.h:    void godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunov.h:    void godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunov.h:#endif // __CUDAC__
src/hydro/HydroRunGodunov.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.h:    DeviceArray<real_t> d_Q; //!< GPU : primitive data array
src/hydro/HydroRunGodunov.h:#endif // __CUDACC__
src/hydro/HydroRunGodunov.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.h:    DeviceArray<real_t> d_qm_x; //!< GPU array for qm state along X
src/hydro/HydroRunGodunov.h:    DeviceArray<real_t> d_qm_y; //!< GPU array for qm state along Y
src/hydro/HydroRunGodunov.h:    DeviceArray<real_t> d_qm_z; //!< GPU array for qm state along Z
src/hydro/HydroRunGodunov.h:    DeviceArray<real_t> d_qp_x; //!< GPU array for qp state along X
src/hydro/HydroRunGodunov.h:    DeviceArray<real_t> d_qp_y; //!< GPU array for qp state along Y
src/hydro/HydroRunGodunov.h:    DeviceArray<real_t> d_qp_z; //!< GPU array for qp state along Z
src/hydro/HydroRunGodunov.h:#endif // __CUDACC__
src/hydro/HydroRunGodunov.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.h:#endif // __CUDACC__
src/hydro/HydroRunGodunov.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunov.h:    CudaTimer timerBoundaries;
src/hydro/HydroRunGodunov.h:    CudaTimer timerGodunov;
src/hydro/HydroRunGodunov.h:    CudaTimer timerPrimVar;
src/hydro/HydroRunGodunov.h:    CudaTimer timerSlopeTrace;
src/hydro/HydroRunGodunov.h:    CudaTimer timerUpdate;
src/hydro/HydroRunGodunov.h:    CudaTimer timerDissipative;
src/hydro/HydroRunGodunov.h:#endif // __CUDACC__
src/hydro/riemann.h: * \brief Provides CPU/GPU riemann solver routines.
src/hydro/cutil_inline_runtime.h: * \brief Some utility routines from the CUDA SDK.
src/hydro/cutil_inline_runtime.h: * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
src/hydro/cutil_inline_runtime.h: * NVIDIA Corporation and its licensors retain all intellectual property and 
src/hydro/cutil_inline_runtime.h: * NVIDIA Corporation is strictly prohibited.
src/hydro/cutil_inline_runtime.h:#define cutilSafeCallNoSync(err)     __cudaSafeCallNoSync(err, __FILE__, __LINE__)
src/hydro/cutil_inline_runtime.h:#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
src/hydro/cutil_inline_runtime.h:#define cutilSafeCallMpi(err,rank)   __cudaSafeCallMpi   (err, rank, __FILE__, __LINE__)
src/hydro/cutil_inline_runtime.h:#define cutilSafeThreadSync()        __cudaSafeThreadSync(__FILE__, __LINE__)
src/hydro/cutil_inline_runtime.h:// This function returns the best GPU (with maximum GFLOPS)
src/hydro/cutil_inline_runtime.h:	cudaDeviceProp deviceProp;
src/hydro/cutil_inline_runtime.h:	cudaGetDeviceCount( &device_count );
src/hydro/cutil_inline_runtime.h:	// Find the best major SM Architecture GPU device
src/hydro/cutil_inline_runtime.h:		cudaGetDeviceProperties( &deviceProp, current_device );
src/hydro/cutil_inline_runtime.h:    // Find the best CUDA capable GPU device
src/hydro/cutil_inline_runtime.h:		cudaGetDeviceProperties( &deviceProp, current_device );
src/hydro/cutil_inline_runtime.h:            // If we find GPU with SM major > 2, search only these
src/hydro/cutil_inline_runtime.h:inline void __cudaSafeCallNoSync( cudaError err, const char *file, const int line )
src/hydro/cutil_inline_runtime.h:    if( cudaSuccess != err) {
src/hydro/cutil_inline_runtime.h:        FPRINTF((stderr, "%s(%i) : cudaSafeCallNoSync() Runtime API error : %s.\n",
src/hydro/cutil_inline_runtime.h:                file, line, cudaGetErrorString( err) ));
src/hydro/cutil_inline_runtime.h:inline void __cudaSafeCall( cudaError err, const char *file, const int line )
src/hydro/cutil_inline_runtime.h:    if( cudaSuccess != err) {
src/hydro/cutil_inline_runtime.h:		FPRINTF((stderr, "%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
src/hydro/cutil_inline_runtime.h:                file, line, cudaGetErrorString( err) ));
src/hydro/cutil_inline_runtime.h:inline void __cudaSafeCallMpi( cudaError err, int rank, const char *file, const int line )
src/hydro/cutil_inline_runtime.h:    if( cudaSuccess != err) {
src/hydro/cutil_inline_runtime.h:		FPRINTF((stderr, "[MPI rank %4d] %s(%i) : cudaSafeCall() Runtime API error : %s.\n",
src/hydro/cutil_inline_runtime.h:			 rank, file, line, cudaGetErrorString( err) ));
src/hydro/cutil_inline_runtime.h:inline void __cudaSafeThreadSync( const char *file, const int line )
src/hydro/cutil_inline_runtime.h:    cudaError err = cudaDeviceSynchronize();
src/hydro/cutil_inline_runtime.h:    if ( cudaSuccess != err) {
src/hydro/cutil_inline_runtime.h:        FPRINTF((stderr, "%s(%i) : cudaDeviceSynchronize() Driver API error : %s.\n",
src/hydro/cutil_inline_runtime.h:                file, line, cudaGetErrorString( err) ));
src/hydro/cutil_inline_runtime.h:        FPRINTF((stderr, "%s(%i) : CUTIL CUDA error.\n",
src/hydro/cutil_inline_runtime.h:    cudaError_t err = cudaGetLastError();
src/hydro/cutil_inline_runtime.h:    if( cudaSuccess != err) {
src/hydro/cutil_inline_runtime.h:        FPRINTF((stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : %s.\n",
src/hydro/cutil_inline_runtime.h:                file, line, errorMessage, cudaGetErrorString( err) ));
src/hydro/cutil_inline_runtime.h:    err = cudaDeviceSynchronize();
src/hydro/cutil_inline_runtime.h:    if( cudaSuccess != err) {
src/hydro/cutil_inline_runtime.h:		FPRINTF((stderr, "%s(%i) : cutilCheckMsg cudaDeviceSynchronize error: %s : %s.\n",
src/hydro/cutil_inline_runtime.h:                file, line, errorMessage, cudaGetErrorString( err) ));
src/hydro/cutil_inline_runtime.h:    inline void cutilChooseCudaDevice(int ARGC, char **ARGV) { }
src/hydro/cutil_inline_runtime.h:        cutilSafeCallNoSync(cudaGetDeviceCount(&deviceCount));
src/hydro/cutil_inline_runtime.h:            FPRINTF((stderr, "CUTIL CUDA error: no devices supporting CUDA.\n"));
src/hydro/cutil_inline_runtime.h:        cudaDeviceProp deviceProp;
src/hydro/cutil_inline_runtime.h:        cutilSafeCallNoSync(cudaGetDeviceProperties(&deviceProp, dev));
src/hydro/cutil_inline_runtime.h:            FPRINTF((stderr, "cutil error: GPU device does not support CUDA.\n"));
src/hydro/cutil_inline_runtime.h:            FPRINTF((stderr, "Using CUDA device [%d]: %s\n", dev, deviceProp.name));
src/hydro/cutil_inline_runtime.h:        cutilSafeCall(cudaSetDevice(dev));
src/hydro/cutil_inline_runtime.h:    // General initialization call to pick the best CUDA Device
src/hydro/cutil_inline_runtime.h:    inline void cutilChooseCudaDevice(int argc, char **argv)
src/hydro/cutil_inline_runtime.h:            cudaSetDevice( cutGetMaxGflopsDeviceId() );
src/hydro/cutil_inline_runtime.h://! Check for CUDA context lost
src/hydro/cutil_inline_runtime.h:inline void cutilCudaCheckCtxLost(const char *errorMessage, const char *file, const int line ) 
src/hydro/cutil_inline_runtime.h:    cudaError_t err = cudaGetLastError();
src/hydro/cutil_inline_runtime.h:    if( cudaSuccess != err) {
src/hydro/cutil_inline_runtime.h:        FPRINTF((stderr, "%s(%i) : CUDA error: %s : %s.\n",
src/hydro/cutil_inline_runtime.h:        file, line, errorMessage, cudaGetErrorString( err) ));
src/hydro/cutil_inline_runtime.h:    err = cudaDeviceSynchronize();
src/hydro/cutil_inline_runtime.h:    if( cudaSuccess != err) {
src/hydro/cutil_inline_runtime.h:        FPRINTF((stderr, "%s(%i) : CCUDA error: %s : %s.\n",
src/hydro/cutil_inline_runtime.h:        file, line, errorMessage, cudaGetErrorString( err) ));
src/hydro/cutil_inline_runtime.h:// General check for CUDA GPU SM Capabilities
src/hydro/cutil_inline_runtime.h:inline bool cutilCudaCapabilities(int major_version, int minor_version)
src/hydro/cutil_inline_runtime.h:    cudaDeviceProp deviceProp;
src/hydro/cutil_inline_runtime.h:    cutilSafeCall( cudaGetDevice(&dev) );
src/hydro/cutil_inline_runtime.h:    cutilSafeCall( cudaGetDeviceProperties(&deviceProp, dev));
src/hydro/cutil_inline_runtime.h:        printf("There is no device supporting CUDA compute capability %d.%d.\n", major_version, minor_version);
src/hydro/HydroMpiParameters.cpp:    // be available as a global constant, usefull for GPU implementation in godunov_unsplit_mhd.cuh).
src/hydro/HydroMpiParameters.cpp:    // be available as a global constant, usefull for GPU implementation).
src/hydro/HydroMpiParameters.cpp:     * Initialize CUDA device if needed.
src/hydro/HydroMpiParameters.cpp:     * When running on a Linux machine with mutiple GPU per node, it might be
src/hydro/HydroMpiParameters.cpp:     * very helpfull if admin has set the CUDA device compute mode to exclusive
src/hydro/HydroMpiParameters.cpp:     * thread can not communicate with the same GPU).
src/hydro/HydroMpiParameters.cpp:     *   nvidia-smi -g $(DEV_ID) -c 1
src/hydro/HydroMpiParameters.cpp:     * If compute mode is set to normal mode, we need to use cudaSetDevice, 
src/hydro/HydroMpiParameters.cpp:     * so that each MPI device is mapped onto a different GPU device.
src/hydro/HydroMpiParameters.cpp:     * half a Tesla S1070, that means cudaGetDeviceCount should return 2.
src/hydro/HydroMpiParameters.cpp:     * If we want the ration 1 MPI process <-> 1 GPU, we need to allocate
src/hydro/HydroMpiParameters.cpp:#ifdef __CUDACC__
src/hydro/HydroMpiParameters.cpp:    cutilSafeCall( cudaGetDeviceCount(&count) );
src/hydro/HydroMpiParameters.cpp:    cutilSafeCall( cudaSetDevice(devId) );
src/hydro/HydroMpiParameters.cpp:    cudaDeviceProp deviceProp;
src/hydro/HydroMpiParameters.cpp:    cutilSafeCall( cudaGetDevice( &myDevId ) );
src/hydro/HydroMpiParameters.cpp:    cutilSafeCall( cudaGetDeviceProperties( &deviceProp, myDevId ) );
src/hydro/HydroMpiParameters.cpp:    // faire un cudaSetDevice et cudaGetDeviceProp et aficher le nom
src/hydro/HydroMpiParameters.cpp:    std::cout << "MPI process " << myRank << " is using GPU device num " << myDevId << std::endl;
src/hydro/HydroMpiParameters.cpp:#endif //__CUDACC__
src/hydro/HydroMpiParameters.cpp:#ifdef __CUDACC__
src/hydro/HydroMpiParameters.cpp:    std::cout << hostname << " [GPU] myDevId : " << myDevId << " (" << deviceProp.name << ")" << std::endl;
src/hydro/HydroMpiParameters.cpp:#endif // __CUDACC__
src/hydro/godunov_notrace.cuh: * \brief Defines the CUDA kernel for the actual Godunov scheme
src/hydro/Arrays.h: * \brief Provides CPU/GPU C++ array classes.
src/hydro/Arrays.h:// the following defines types for cuda compatibility when using g++
src/hydro/Arrays.h:#ifdef __CUDACC__
src/hydro/Arrays.h:#endif // __CUDACC__
src/hydro/Arrays.h:  /** enumeration only used in the CUDA implementation */
src/hydro/Arrays.h:    PINNED    /**< enum PINNED (allocation using cudaMallocHost) */
src/hydro/Arrays.h:#ifdef __CUDACC__
src/hydro/Arrays.h: * \brief Provides an array object with memory allocated on GPU.
src/hydro/Arrays.h: * This class is symetric of HostArray, but allocated in GPU global
src/hydro/Arrays.h: * We take care of alignment constraints by using cudaMallocPitch
src/hydro/Arrays.h:  /** enumeration only used in the CUDA implementation */
src/hydro/Arrays.h:    LINEAR,  /**< enum LINEAR  (standard allocation using cudaMalloc) */
src/hydro/Arrays.h:    PITCHED  /**< enum PITCHED (allocation using cudaMallocPitch) */
src/hydro/Arrays.h:  /** total allocated memory in bytes on GPU device */
src/hydro/Arrays.h:#endif // __CUDACC__
src/hydro/Arrays.h:#ifdef __CUDACC__
src/hydro/Arrays.h:      cutilSafeCall( cudaMallocHost((void**)&_data, length * numVar * sizeof(T)) );
src/hydro/Arrays.h:#else // standard version (non-CUDA)
src/hydro/Arrays.h:#endif // __CUDACC__
src/hydro/Arrays.h:#ifdef __CUDACC__
src/hydro/Arrays.h:    cutilSafeCall( cudaMallocHost((void**)&_data, dim.x * dim.y * dim.z * sizeof(T)) );
src/hydro/Arrays.h:#else // standard version (non-CUDA)
src/hydro/Arrays.h:#endif // __CUDACC__
src/hydro/Arrays.h:#ifdef __CUDACC__
src/hydro/Arrays.h:    cutilSafeCall( cudaMallocHost((void**)&_data, dim.x * dim.y * dim.z * dim.w * sizeof(T)) );
src/hydro/Arrays.h:#else // standard version (non-CUDA)
src/hydro/Arrays.h:#endif // __CUDACC__
src/hydro/Arrays.h:#ifdef __CUDACC__
src/hydro/Arrays.h:    cutilSafeCall( cudaFreeHost(_data) );
src/hydro/Arrays.h:#else // standard version (non-CUDA)
src/hydro/Arrays.h:#endif // __CUDACC__
src/hydro/Arrays.h:#ifdef __CUDACC__
src/hydro/Arrays.h:    cutilSafeCall( cudaMalloc((void**) &_data, rows * dimXBytes) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMemset((void* )  _data, 0, rows*dimXBytes) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMallocPitch((void**) &_data, &pitchBytes, dimXBytes, rows) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMemset2D(   (void* )  _data,  pitchBytes, 0, dimXBytes, rows) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMalloc((void**) &_data, rows * dimXBytes) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMemset((void* )  _data, 0, rows*dimXBytes) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMallocPitch((void**) &_data, &pitchBytes, dimXBytes, rows) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMemset2D(   (void* )  _data,  pitchBytes, 0, dimXBytes, rows) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMalloc((void**) &_data, rows * dimXBytes) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMemset((void*)   _data, 0, rows*dimXBytes) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMallocPitch((void**) &_data, &pitchBytes, dimXBytes, rows) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMemset2D(   (void*)   _data,  pitchBytes, 0, dimXBytes, rows) );
src/hydro/Arrays.h:  cutilSafeCall( cudaFree(_data) );
src/hydro/Arrays.h:  // host memory was allocated by cudaMallocHost, so we use async mem copy
src/hydro/Arrays.h:      cutilSafeCall( cudaMemcpy2DAsync(data(), pitchBytes(), 
src/hydro/Arrays.h:				       cudaMemcpyHostToDevice) );
src/hydro/Arrays.h:      cutilSafeCall( cudaMemcpyAsync(data(),
src/hydro/Arrays.h:				     cudaMemcpyHostToDevice) );
src/hydro/Arrays.h:      cutilSafeCall( cudaMemcpy2D(data(), pitchBytes(), 
src/hydro/Arrays.h:				  cudaMemcpyHostToDevice) );
src/hydro/Arrays.h:      cutilSafeCall( cudaMemcpy(data(),
src/hydro/Arrays.h:				cudaMemcpyHostToDevice) );
src/hydro/Arrays.h:  // host memory was allocated by cudaMallocHost, so we use async mem copy
src/hydro/Arrays.h:      cutilSafeCall( cudaMemcpy2DAsync(dest.data(), dest.dimXBytes(), 
src/hydro/Arrays.h:				       cudaMemcpyDeviceToHost) );
src/hydro/Arrays.h:      cutilSafeCall( cudaMemcpyAsync(dest.data(), 
src/hydro/Arrays.h:				     cudaMemcpyDeviceToHost) );      
src/hydro/Arrays.h:      cutilSafeCall( cudaMemcpy2D(dest.data(), dest.dimXBytes(), 
src/hydro/Arrays.h:				  cudaMemcpyDeviceToHost) );
src/hydro/Arrays.h:      cutilSafeCall( cudaMemcpy(dest.data(), 
src/hydro/Arrays.h:				cudaMemcpyDeviceToHost) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMemcpy2D(dest.data(), dest.pitchBytes(), 
src/hydro/Arrays.h:				cudaMemcpyDeviceToDevice) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMemcpy(dest.data(), 
src/hydro/Arrays.h:			      cudaMemcpyDeviceToDevice) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMemset((void* )  _data, 0, sizeBytes()) );
src/hydro/Arrays.h:    cutilSafeCall( cudaMemset2D(   (void* )  _data,  _pitch*sizeof(T), 0, dimXBytes, rows) );
src/hydro/Arrays.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#include "gpu_macros.h"
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#include "../utils/monitoring/CudaTimer.h"
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:   * \note Important note : this class does sequential computations (one CPU or GPU).
src/hydro/HydroRunBaseMpi.h:   * For parallel computations (multi-CPU/GPU, symbol USE_MPI
src/hydro/HydroRunBaseMpi.h:     * \param[in]  U      (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:     * \param[in]  U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:     * \param[in,out] U  (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:     * \param[in,out] U  (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__   
src/hydro/HydroRunBaseMpi.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__   
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:     * \param[inout] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:     * \param[inout] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:     * \param[in,out] U (either h_U or h_U2, or d_U/d_U2 in the GPU version)
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:    //! used in the GPU version to control the number of CUDA block of
src/hydro/HydroRunBaseMpi.h:    //! used in the GPU version to control the number of CUDA blocks in random forcing normalization
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:    // Data Arrays (these arrays are only used for the GPU version 
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:    //! compute border on GPU (call make_boundary for each borders in direction idim)
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:    //! compute all borders on GPU (call make_boundary for each borders)
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:     * GPU version copy data back from GPU to host  memory (h_U).
src/hydro/HydroRunBaseMpi.h:    virtual void copyGpuToCpu(int nStep=0);
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:     * 4. (GPU only) copy border to Device data array
src/hydro/HydroRunBaseMpi.h:     * <b>GPU version.</b>
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:    //! Data array on GPU
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.h:    CudaTimer timerBoundariesCpuGpu;
src/hydro/HydroRunBaseMpi.h:    CudaTimer timerBoundariesGpu;
src/hydro/HydroRunBaseMpi.h:    CudaTimer timerBoundariesMpi;
src/hydro/HydroRunBaseMpi.h:#endif // __CUDACC__
src/hydro/HydroRunRelaxingTVD.cpp:// include CUDA kernel when necessary
src/hydro/HydroRunRelaxingTVD.cpp:#ifdef __CUDACC__
src/hydro/HydroRunRelaxingTVD.cpp:#endif // __CUDACC__
src/hydro/HydroRunRelaxingTVD.cpp:   * wrapper that performs the actual CPU or GPU scheme integration step.
src/hydro/HydroRunRelaxingTVD.cpp:#ifdef __CUDACC__
src/hydro/HydroRunRelaxingTVD.cpp:      relaxing_tvd_gpu(d_U , d_U2, XDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:      relaxing_tvd_gpu(d_U2, d_U , YDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:      relaxing_tvd_gpu(d_U , d_U2, YDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:      relaxing_tvd_gpu(d_U2, d_U , XDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U , d_U2, XDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U2, d_U , YDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U , d_U2, ZDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U2, d_U , ZDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U , d_U2, YDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U2, d_U , XDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U , d_U2, ZDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U2, d_U , XDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U , d_U2, YDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U2, d_U , YDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U , d_U2, XDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U2, d_U , ZDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U , d_U2, YDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U2, d_U , ZDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U , d_U2, XDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U2, d_U , XDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U , d_U2, ZDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:	relaxing_tvd_gpu(d_U2, d_U , YDIR, dt);
src/hydro/HydroRunRelaxingTVD.cpp:  } // HydroRunRelaxingTVD::relaxing_tvd_sweep (GPU)
src/hydro/HydroRunRelaxingTVD.cpp:#endif // __CUDACC__
src/hydro/HydroRunRelaxingTVD.cpp:#ifdef __CUDACC__
src/hydro/HydroRunRelaxingTVD.cpp:  void HydroRunRelaxingTVD::relaxing_tvd_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunRelaxingTVD.cpp:  } // HydroRunRelaxingTVD::relaxing_tvd_gpu
src/hydro/HydroRunRelaxingTVD.cpp:#endif // __CUDACC__
src/hydro/HydroRunRelaxingTVD.cpp:#ifdef __CUDACC__
src/hydro/HydroRunRelaxingTVD.cpp:	  copyGpuToCpu(0);
src/hydro/MHDRunGodunovZslabMpi.cpp: * Parallel CUDA+MPI implementation with z-slab method.
src/hydro/MHDRunGodunovZslabMpi.cpp:// include CUDA kernel when necessary
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_Q.allocate (make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_emf.allocate(make_uint4(isize, jsize, zSlabWidthG, 3    ), gpuMemAllocType); // 3 EMF's
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qm_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qm_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qm_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qp_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qp_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qp_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qEdge_RT.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qEdge_RB.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qEdge_LT.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qEdge_LB.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qEdge_RT2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qEdge_RB2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qEdge_LT2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qEdge_LB2.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qEdge_RT3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qEdge_RB3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qEdge_LT3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_qEdge_LB3.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_elec.allocate (make_uint4(isize, jsize, zSlabWidthG, 3), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_dA.allocate   (make_uint4(isize, jsize, zSlabWidthG, 3), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_dB.allocate   (make_uint4(isize, jsize, zSlabWidthG, 3), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:    d_dC.allocate   (make_uint4(isize, jsize, zSlabWidthG, 3), gpuMemAllocType);	
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:      d_shear_flux_xmin_toSend.allocate   (make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:      d_shear_flux_xmax_toSend.allocate   (make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:      d_shear_flux_xmin_remap.allocate    (make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:      d_shear_flux_xmax_remap.allocate    (make_uint3(jsize,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:      d_shear_flux_xmin_recv_glob.allocate(make_uint3(my*ny+2*ghostWidth,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:      d_shear_flux_xmax_recv_glob.allocate(make_uint3(my*ny+2*ghostWidth,zSlabWidthG,NUM_COMPONENT_REMAP), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:      d_shear_border.allocate           (make_uint4(ghostWidth, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:     * memory allocation for GPU routines debugging
src/hydro/MHDRunGodunovZslabMpi.cpp:    // GPU execution settings
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:      cudaFuncSetCacheConfig(kernel_mhd_compute_trace_v4_zslab, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovZslabMpi.cpp:      cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_zslab, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovZslabMpi.cpp:      cudaFuncSetCacheConfig(kernel_mhd_compute_emf_v4_zslab, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovZslabMpi.cpp:      cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_shear_zslab, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovZslabMpi.cpp:      cudaFuncSetCacheConfig(kernel_mhd_flux_update_hydro_v4_shear_part1_zslab, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovZslabMpi.cpp:      cudaFuncSetCacheConfig(kernel_mhd_compute_emf_shear_zslab, cudaFuncCachePreferL1);
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:      cutilSafeCall( cudaMemGetInfo(&freeMemory, &totalMemory) );
src/hydro/MHDRunGodunovZslabMpi.cpp:      std::cout << "Total memory available on GPU " << totalMemory/1000000. << " MBytes\n";
src/hydro/MHDRunGodunovZslabMpi.cpp:      std::cout << "Currently free  memory on GPU " <<  freeMemory/1000000. << " MBytes\n";
src/hydro/MHDRunGodunovZslabMpi.cpp:      std::cout << "Total memory allocated on GPU " << DeviceArray<real_t>::totalAllocMemoryInKB/1000. << " MBytes\n";
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifndef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:	godunov_unsplit_rotating_gpu(d_U , d_U2, dt, nStep);
src/hydro/MHDRunGodunovZslabMpi.cpp:	godunov_unsplit_rotating_gpu(d_U2, d_U , dt, nStep);
src/hydro/MHDRunGodunovZslabMpi.cpp:	godunov_unsplit_gpu(d_U , d_U2, dt, nStep);
src/hydro/MHDRunGodunovZslabMpi.cpp:	godunov_unsplit_gpu(d_U2, d_U , dt, nStep);
src/hydro/MHDRunGodunovZslabMpi.cpp:  } // MHDRunGodunovZslabMpi::godunov_unsplit (GPU version)
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:  void MHDRunGodunovZslabMpi::godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovZslabMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovZslabMpi :: kernel_mhd_compute_primitive_variables_zslab error", myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovZslabMpi :: kernel_mhd_compute_elec_field_zslab error",myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovZslabMpi :: kernel_mhd_compute_mag_slopes_zslab error",myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovZslabMpi :: kernel_mhd_compute_trace_v4_zslab error",myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	    checkCudaErrorMpi("MHDRunGodunovZslabMpi kernel_mhd_compute_gravity_predictor_v4_zslab error",myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovZslabMpi kernel_mhd_flux_update_hydro_v4_zslab error", myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovZslabMpi kernel_mhd_compute_emf_v4 error",myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovZslabMpi kernel_mhd_flux_update_ct_v4 error",myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:  } // MHDRunGodunovZslabMpi::godunov_unsplit_gpu
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:  void MHDRunGodunovZslabMpi::godunov_unsplit_rotating_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovZslabMpi.cpp:	checkCudaErrorMpi("MHDRunGodunovZslabMpi :: kernel_mhd_compute_primitive_variables_zslab error", myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	checkCudaErrorMpi("MHDRunGodunovZslabMpi :: kernel_mhd_compute_elec_field_zslab error", myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	checkCudaErrorMpi("MHDRunGodunovZslabMpi :: kernel_mhd_compute_mag_slopes_zslab error", myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	checkCudaErrorMpi("MHDRunGodunovZslabMpi :: kernel_mhd_compute_trace_v4_zslab error", myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovZslabMpi :: kernel_mhd_compute_gravity_predictor_v4_zslab error", myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:      	checkCudaErrorMpi("MHDRunGodunovZslabMpi :: kernel_mhd_flux_update_hydro_v4_shear_zslab error", myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovZslabMpi :: kernel_mhd_flux_update_hydro_v4_shear_part1_zslab error", myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	  checkCudaErrorMpi("MHDRunGodunovZslabMpi :: kernel_mhd_compute_emf_shear_zslab error", myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:	checkCudaErrorMpi("MHDRunGodunovZslabMpi :: kernel_mhd_flux_update_ct_v4_zslab error", myRank);
src/hydro/MHDRunGodunovZslabMpi.cpp:  } // MHDRunGodunovZslabMpi::godunov_unsplit_rotating_gpu
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:       *  perform final remapping in shear borders on GPU
src/hydro/MHDRunGodunovZslabMpi.cpp:	  // upload into GPU memory
src/hydro/MHDRunGodunovZslabMpi.cpp:	  // upload into GPU memory
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:  } // MHDRunGodunovZslabMpi::make_all_boundaries_shear -- GPU version
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:	  copyGpuToCpu(nStep);
src/hydro/MHDRunGodunovZslabMpi.cpp:	  copyGpuToCpu(nStep);
src/hydro/MHDRunGodunovZslabMpi.cpp:      copyGpuToCpu(nStep);
src/hydro/MHDRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.cpp:    printf("Euler MHD godunov boundaries pure GPU     [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerBoundariesGpu.elapsed(), timerBoundariesGpu.elapsed()/timerTotal.elapsed()*100.);
src/hydro/MHDRunGodunovZslabMpi.cpp:    printf("Euler MHD godunov boundaries CPU-GPU comm [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerBoundariesCpuGpu.elapsed(), timerBoundariesCpuGpu.elapsed()/timerTotal.elapsed()*100.);
src/hydro/MHDRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslab.h:#include "gpu_macros.h"
src/hydro/HydroRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.h:#include "../utils/monitoring/CudaTimer.h"
src/hydro/HydroRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslab.h:    //! see godunov_unsplit_gpu or godunov_unsplit_cpu.
src/hydro/HydroRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.h:    //! scheme on GPU, two array are necessary to make ping-pong (d_UOld and
src/hydro/HydroRunGodunovZslab.h:    void godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslab.h:    void godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslab.h:    void godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslab.h:    void godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslab.h:#endif // __CUDAC__
src/hydro/HydroRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.h:    DeviceArray<real_t> d_Q; //!< GPU : primitive data array
src/hydro/HydroRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.h:    DeviceArray<real_t> d_qm_x; //!< GPU array for qm state along X
src/hydro/HydroRunGodunovZslab.h:    DeviceArray<real_t> d_qm_y; //!< GPU array for qm state along Y
src/hydro/HydroRunGodunovZslab.h:    DeviceArray<real_t> d_qm_z; //!< GPU array for qm state along Z
src/hydro/HydroRunGodunovZslab.h:    DeviceArray<real_t> d_qp_x; //!< GPU array for qp state along X
src/hydro/HydroRunGodunovZslab.h:    DeviceArray<real_t> d_qp_y; //!< GPU array for qp state along Y
src/hydro/HydroRunGodunovZslab.h:    DeviceArray<real_t> d_qp_z; //!< GPU array for qp state along Z
src/hydro/HydroRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslab.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.h:    CudaTimer timerBoundaries;
src/hydro/HydroRunGodunovZslab.h:    CudaTimer timerGodunov;
src/hydro/HydroRunGodunovZslab.h:    CudaTimer timerPrimVar;
src/hydro/HydroRunGodunovZslab.h:    CudaTimer timerSlopeTrace;
src/hydro/HydroRunGodunovZslab.h:    CudaTimer timerUpdate;
src/hydro/HydroRunGodunovZslab.h:    CudaTimer timerDissipative;
src/hydro/HydroRunGodunovZslab.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h: * CPU/GPU version) containing the actual numerical scheme for solving MHD.
src/hydro/MHDRunGodunovZslabMpi.h: * 3D only MHD solver (multi CPU or multi GPU) MPI version.
src/hydro/MHDRunGodunovZslabMpi.h: * mono-CPU / mono-GPU version: it only implements was is called
src/hydro/MHDRunGodunovZslabMpi.h: * Note that version 4 should give best performances on GPU.
src/hydro/MHDRunGodunovZslabMpi.h:   * mono-GPU.
src/hydro/MHDRunGodunovZslabMpi.h:    //! In the GPU version, the conversion is done on line, inside
src/hydro/MHDRunGodunovZslabMpi.h:    //! see godunov_unsplit_gpu or godunov_unsplit_cpu.
src/hydro/MHDRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:    //! scheme on GPU, two array are necessary to make ping-pong (d_UOld and
src/hydro/MHDRunGodunovZslabMpi.h:    void godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovZslabMpi.h:#endif // __CUDAC__
src/hydro/MHDRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:    //! Same routine as godunov_unsplit_gpu but with rotating
src/hydro/MHDRunGodunovZslabMpi.h:    void godunov_unsplit_rotating_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovZslabMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:    DeviceArray<real_t> d_Q; //!< GPU : primitive data array
src/hydro/MHDRunGodunovZslabMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:    DeviceArray<real_t> d_qm_x; //!< GPU array for qm state along X
src/hydro/MHDRunGodunovZslabMpi.h:    DeviceArray<real_t> d_qm_y; //!< GPU array for qm state along Y
src/hydro/MHDRunGodunovZslabMpi.h:    DeviceArray<real_t> d_qm_z; //!< GPU array for qm state along Z
src/hydro/MHDRunGodunovZslabMpi.h:    DeviceArray<real_t> d_qp_x; //!< GPU array for qp state along X
src/hydro/MHDRunGodunovZslabMpi.h:    DeviceArray<real_t> d_qp_y; //!< GPU array for qp state along Y
src/hydro/MHDRunGodunovZslabMpi.h:    DeviceArray<real_t> d_qp_z; //!< GPU array for qp state along Z
src/hydro/MHDRunGodunovZslabMpi.h:    DeviceArray<real_t> d_emf; //!< GPU array for electromotive forces
src/hydro/MHDRunGodunovZslabMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:    DeviceArray<real_t> d_shear_flux_xmin_toSend;    /*!< flux correction data at XMIN (GPU -> CPU + MPI comm) */
src/hydro/MHDRunGodunovZslabMpi.h:    DeviceArray<real_t> d_shear_flux_xmax_toSend;    /*!< flux correction data at XMAX (GPU -> CPU + MPI comm) */
src/hydro/MHDRunGodunovZslabMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerGodunov;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerTraceUpdate;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerUpdate;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerEmf;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerDissipative;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerPrimVar;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerElecField;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerMagSlopes;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerTrace;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerHydroShear;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerRemapping;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerShearBorder;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerMakeShearBorder;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerMakeShearBorderSend;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerMakeShearBorderSlopes;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerMakeShearBorderFinalRemapping;
src/hydro/MHDRunGodunovZslabMpi.h:    CudaTimer timerCtUpdate;
src/hydro/MHDRunGodunovZslabMpi.h:#endif // __CUDACC__
src/hydro/godunov_unsplit_zslab.cuh: * \brief Defines the CUDA kernel for the unsplit Godunov scheme computations (z-slab method).
src/hydro/geometry_utils.h: * \brief Small geometry related utilities common to CPU / GPU code.
src/hydro/HydroRunGodunovMpi.h: * MPI+CUDA computations.
src/hydro/HydroRunGodunovMpi.h:#include "gpu_macros.h"
src/hydro/HydroRunGodunovMpi.h:    //! see godunov_unsplit_gpu or godunov_unsplit_cpu.
src/hydro/HydroRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.h:    //! Actual computation of the godunov integration on GPU using
src/hydro/HydroRunGodunovMpi.h:    void godunov_split_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovMpi.h:    //! scheme on GPU, two array are necessary to make ping-pong (d_UOld and
src/hydro/HydroRunGodunovMpi.h:    void godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovMpi.h:    void godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovMpi.h:    void godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovMpi.h:    void godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovMpi.h:#endif // __CUDAC__
src/hydro/HydroRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.h:    DeviceArray<real_t> d_Q; //!< GPU : primitive data array
src/hydro/HydroRunGodunovMpi.h:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.h:    DeviceArray<real_t> d_qm_x; //!< GPU array for qm state along X
src/hydro/HydroRunGodunovMpi.h:    DeviceArray<real_t> d_qm_y; //!< GPU array for qm state along Y
src/hydro/HydroRunGodunovMpi.h:    DeviceArray<real_t> d_qm_z; //!< GPU array for qm state along Z
src/hydro/HydroRunGodunovMpi.h:    DeviceArray<real_t> d_qp_x; //!< GPU array for qp state along X
src/hydro/HydroRunGodunovMpi.h:    DeviceArray<real_t> d_qp_y; //!< GPU array for qp state along Y
src/hydro/HydroRunGodunovMpi.h:    DeviceArray<real_t> d_qp_z; //!< GPU array for qp state along Z
src/hydro/HydroRunGodunovMpi.h:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.h:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.h:    CudaTimer timerGodunov;
src/hydro/HydroRunGodunovMpi.h:    CudaTimer timerPrimVar;
src/hydro/HydroRunGodunovMpi.h:    CudaTimer timerSlopeTrace;
src/hydro/HydroRunGodunovMpi.h:    CudaTimer timerUpdate;
src/hydro/HydroRunGodunovMpi.h:    CudaTimer timerDissipative;
src/hydro/HydroRunGodunovMpi.h:#endif // __CUDACC__
src/hydro/MHDRunBase.cpp:// include CUDA kernel when necessary
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:#endif // __CUDACC__
src/hydro/MHDRunBase.cpp:    // time step computation (!!! GPU only !!!)
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:#endif // __CUDACC__
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:      checkCudaError("MHDRunBase cmpdt_2d_mhd error");
src/hydro/MHDRunBase.cpp:      checkCudaError("MHDRunBase h_invDt error");
src/hydro/MHDRunBase.cpp:      checkCudaError("MHDRunBase cmpdt_3d_mhd error");
src/hydro/MHDRunBase.cpp:      checkCudaError("MHDRunBase h_invDt error");
src/hydro/MHDRunBase.cpp:#endif // __CUDACC__
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:    checkCudaError("in MHDRunBase :: kernel_ct_update_2d");
src/hydro/MHDRunBase.cpp:  } // MHDRunBase::compute_ct_update_2d (2D case, GPU)
src/hydro/MHDRunBase.cpp:#endif // __CUDACC__
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:    checkCudaError("in MHDRunBase :: kernel_ct_update_3d");
src/hydro/MHDRunBase.cpp:  } // MHDRunBase::compute_ct_update_3d (3D case, GPU)
src/hydro/MHDRunBase.cpp:#endif // __CUDACC__
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:    checkCudaError("in MHDRunBase :: kernel_ct_update_3d_zslab");
src/hydro/MHDRunBase.cpp:  } // MHDRunBase::compute_ct_update_3d (3D case, GPU, z-slab)
src/hydro/MHDRunBase.cpp:#endif // __CUDACC__
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:    checkCudaError("in MHDRunBase :: kernel_resistivity_forces_2d");
src/hydro/MHDRunBase.cpp:  } // MHDRunBase::compute_resistivity_emf_2d (GPU)
src/hydro/MHDRunBase.cpp:#endif // __CUDACC__
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:    checkCudaError("in MHDRunBase :: kernel_resistivity_forces_3d");
src/hydro/MHDRunBase.cpp:  } // MHDRunBase::compute_resistivity_emf_3d (GPU)
src/hydro/MHDRunBase.cpp:#endif // __CUDACC__
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:    checkCudaError("in MHDRunBase :: kernel_resistivity_forces_3d_zslab");
src/hydro/MHDRunBase.cpp:  } // MHDRunBase::compute_resistivity_emf_3d (GPU)
src/hydro/MHDRunBase.cpp:#endif // __CUDACC__
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:    checkCudaError("in MHDRunBase :: kernel_resistivity_energy_flux_2d");
src/hydro/MHDRunBase.cpp:  } // MHDRunBase::compute_resistivity_energy_flux_2d (GPU)
src/hydro/MHDRunBase.cpp:#endif // __CUDACC
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:    checkCudaError("in MHDRunBase :: kernel_resistivity_energy_flux_3d");
src/hydro/MHDRunBase.cpp:#endif // __CUDACC
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:    checkCudaError("in MHDRunBase :: kernel_resistivity_energy_flux_3d_zslab");
src/hydro/MHDRunBase.cpp:#endif // __CUDACC
src/hydro/MHDRunBase.cpp:    // copy data to GPU if necessary
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:#endif // __CUDACC__
src/hydro/MHDRunBase.cpp:      // need to upload gravity field on GPU
src/hydro/MHDRunBase.cpp:#ifdef __CUDACC__
src/hydro/MHDRunBase.cpp:    copyGpuToCpu(nStep);
src/hydro/MHDRunBase.cpp:    copyGpuToCpu(nStep);
src/hydro/MHDRunBase.cpp:      copyGpuToCpu(nStep);
src/hydro/MHDRunBase.cpp:      copyGpuToCpu(nStep);
src/hydro/cmpdt.cuh: * \brief Provides the CUDA kernel for computing time step through a
src/hydro/cmpdt.cuh: * This routines are directly borrowed from the original CUDA SDK
src/hydro/cmpdt.cuh:  // see CUDA documentation of the reduction example, especially the
src/hydro/cmpdt.cuh:  // see CUDA documentation of the reduction example, especially the
src/hydro/kurganov-tadmor.h: * GPU __DEVICE__ counterparts)  
src/hydro/HydroParameters.h:#include "gpu_macros.h"
src/hydro/HydroParameters.h:   * @param _initGpu : [only used in GPU version] a boolean that tells to 
src/hydro/HydroParameters.h:   *         When using a CUDA+MPI version, running on an achitecture
src/hydro/HydroParameters.h:   *         with multiple GPU per node, it may be usefull to choose the device,
src/hydro/HydroParameters.h:   *         calling cudaSetDevice, to be ensure that each MPI processes
src/hydro/HydroParameters.h:   *         comunicates with a unique GPU device.
src/hydro/HydroParameters.h:  HydroParameters(ConfigMap &_configMap, bool _initGpu);
src/hydro/HydroParameters.h:   * GPU version copy them to __constant__ memory area on device.
src/hydro/HydroParameters.h:			     //!< parameters (used in the GPU version of
src/hydro/HydroParameters.h:#ifdef __CUDACC__
src/hydro/HydroParameters.h:  hydroSimu::DeviceArray<real_t>::DeviceMemoryAllocType gpuMemAllocType;
src/hydro/HydroParameters.h:#endif // __CUDACC__
src/hydro/HydroParameters.h:					bool _initGpu=true) 
src/hydro/HydroParameters.h:#ifdef __CUDACC__
src/hydro/HydroParameters.h:  ,gpuMemAllocType(hydroSimu::DeviceArray<real_t>::PITCHED)
src/hydro/HydroParameters.h:#endif // __CUDACC__
src/hydro/HydroParameters.h:  // the follwing is necessary so that grid resolution can be used on GPU
src/hydro/HydroParameters.h:  /* initialize global parameters (used in both the CPU and the GPU code */
src/hydro/HydroParameters.h:  std::cout << "GPU : scheme : " << _gParams.scheme << std::endl;
src/hydro/HydroParameters.h:#ifdef __CUDACC__
src/hydro/HydroParameters.h:  // with the CUDA code, to be able to call the same functions
src/hydro/HydroParameters.h:#endif // __CUDACC__
src/hydro/HydroParameters.h:  // to compare CPU/GPU performances
src/hydro/HydroParameters.h:  // GPU memory allocation type
src/hydro/HydroParameters.h:#ifdef __CUDACC__
src/hydro/HydroParameters.h:  std::string gpuAllocString = configMap.getString("implementation","DeviceMemoryAllocType", "PITCHED");
src/hydro/HydroParameters.h:  if (!gpuAllocString.compare("LINEAR")) {
src/hydro/HydroParameters.h:    gpuMemAllocType = hydroSimu::DeviceArray<real_t>::LINEAR;
src/hydro/HydroParameters.h:    std::cout << "Using GPU memory allocation type : LINEAR" << std::endl;
src/hydro/HydroParameters.h:  } else if (!gpuAllocString.compare("PITCHED")) {
src/hydro/HydroParameters.h:    gpuMemAllocType = hydroSimu::DeviceArray<real_t>::PITCHED;
src/hydro/HydroParameters.h:    std::cout << "Using GPU memory allocation type : PITCHED" << std::endl;
src/hydro/HydroParameters.h:    std::cout << "WARNING: unknown GPU memory allocation type !!!" << std::endl;
src/hydro/HydroParameters.h:#endif // __CUDACC__
src/hydro/HydroParameters.h:  if (_initGpu) {
src/hydro/HydroParameters.h:     * choose a CUDA device (if running the GPU version)
src/hydro/HydroParameters.h:#ifdef __CUDACC__  
src/hydro/HydroParameters.h:    cutilSafeCall( cudaSetDevice(cutGetMaxGflopsDeviceId()) );
src/hydro/HydroParameters.h:    cudaDeviceProp deviceProp;
src/hydro/HydroParameters.h:    cutilSafeCall( cudaGetDevice( &myDevId ) );
src/hydro/HydroParameters.h:    cutilSafeCall( cudaGetDeviceProperties( &deviceProp, myDevId ) );
src/hydro/HydroParameters.h:    std::cout << "[GPU] myDevId : " << myDevId << " (" << deviceProp.name << ")" << std::endl;
src/hydro/HydroParameters.h:#endif // __CUDACC__
src/hydro/HydroParameters.h:     * (if using GPU, otherwise this routine does nothing...) 
src/hydro/HydroParameters.h:#ifdef __CUDACC__
src/hydro/HydroParameters.h:  cutilSafeCall( cudaMemcpyToSymbol(::gParams, &_gParams, sizeof(GlobalConstants), 0, cudaMemcpyHostToDevice ) );
src/hydro/HydroParameters.h:#endif // __CUDACC__
src/hydro/mhd_ct_update_zslab.cuh: * \brief CUDA kernel for update magnetic field with emf (constraint transport), z-slab implementation.
src/hydro/mhd_ct_update_zslab.cuh: * CUDA kernel perform magnetic field update (ct) from emf (3D data).
src/hydro/gravity.cuh: * \brief CUDA kernel for computing gravity forces (adapted from Dumses).
src/hydro/gravity.cuh: * CUDA kernel computing gravity predictor (2D data).
src/hydro/gravity.cuh: * CUDA kernel computing gravity predictor (3D data).
src/hydro/gravity.cuh: * CUDA kernel computing gravity source term (2D data).
src/hydro/gravity.cuh: * CUDA kernel computing gravity source term (3D data).
src/hydro/godunov_unsplit_mhd_v0.cuh: * \brief Defines the CUDA kernel for the actual MHD Godunov scheme.
src/hydro/positiveScheme.h:#ifndef __CUDACC__
src/hydro/positiveScheme.h:#endif // __CUDACC__
src/hydro/positiveScheme.h:#ifndef __CUDACC__
src/hydro/positiveScheme.h:#endif // __CUDACC__
src/hydro/positiveScheme.h:#ifndef __CUDACC__
src/hydro/positiveScheme.h:#endif // __CUDACC__
src/hydro/positiveScheme.h:#ifndef __CUDACC__
src/hydro/positiveScheme.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.h: * CPU/GPU version) containing the actual numerical scheme for solving MHD.
src/hydro/MHDRunGodunovMpi.h: * 2D/3D MHD solver (multi CPU or multi GPU) MPI version.
src/hydro/MHDRunGodunovMpi.h: * mono-CPU / mono-GPU version: it only implements was is called
src/hydro/MHDRunGodunovMpi.h: * Note that version 4 should give best performances on GPU.
src/hydro/MHDRunGodunovMpi.h:   * mono-GPU.
src/hydro/MHDRunGodunovMpi.h:   * GPU version (2D only); 3D is buggy (prefer using version 3 which is
src/hydro/MHDRunGodunovMpi.h:   * memory buffer h_elec (or d_elec in the GPU version).<BR>
src/hydro/MHDRunGodunovMpi.h:   * available on GPU (3D only)<BR>
src/hydro/MHDRunGodunovMpi.h:   * Note that this 3D GPU version was really hard to debug (see Trac
src/hydro/MHDRunGodunovMpi.h:    //! In the GPU version, the conversion is done on line, inside
src/hydro/MHDRunGodunovMpi.h:    //! see godunov_unsplit_gpu or godunov_unsplit_cpu.
src/hydro/MHDRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.h:    //! scheme on GPU, two array are necessary to make ping-pong (d_UOld and
src/hydro/MHDRunGodunovMpi.h:    void godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.h:    void godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.h:    void godunov_unsplit_gpu_v0_old(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.h:    void godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.h:    void godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.h:    void godunov_unsplit_gpu_v3(DeviceArray<real_t>& d_UOld,
src/hydro/MHDRunGodunovMpi.h:    void godunov_unsplit_gpu_v4(DeviceArray<real_t>& d_UOld,
src/hydro/MHDRunGodunovMpi.h:#endif // __CUDAC__
src/hydro/MHDRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.h:    //! Same routine as godunov_unsplit_gpu but with rotating
src/hydro/MHDRunGodunovMpi.h:    void godunov_unsplit_rotating_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunovMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.h:    DeviceArray<real_t> d_Q; //!< GPU : primitive data array
src/hydro/MHDRunGodunovMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.h:    DeviceArray<real_t> d_qm_x; //!< GPU array for qm state along X
src/hydro/MHDRunGodunovMpi.h:    DeviceArray<real_t> d_qm_y; //!< GPU array for qm state along Y
src/hydro/MHDRunGodunovMpi.h:    DeviceArray<real_t> d_qm_z; //!< GPU array for qm state along Z
src/hydro/MHDRunGodunovMpi.h:    DeviceArray<real_t> d_qp_x; //!< GPU array for qp state along X
src/hydro/MHDRunGodunovMpi.h:    DeviceArray<real_t> d_qp_y; //!< GPU array for qp state along Y
src/hydro/MHDRunGodunovMpi.h:    DeviceArray<real_t> d_qp_z; //!< GPU array for qp state along Z
src/hydro/MHDRunGodunovMpi.h:    DeviceArray<real_t> d_emf; //!< GPU array for electromotive forces
src/hydro/MHDRunGodunovMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.h:    DeviceArray<real_t> d_shear_flux_xmin_toSend;    /*!< flux correction data at XMIN (GPU -> CPU + MPI comm) */
src/hydro/MHDRunGodunovMpi.h:    DeviceArray<real_t> d_shear_flux_xmax_toSend;    /*!< flux correction data at XMAX (GPU -> CPU + MPI comm) */
src/hydro/MHDRunGodunovMpi.h:#endif // __CUDACC__
src/hydro/MHDRunGodunovMpi.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerGodunov;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerTraceUpdate;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerUpdate;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerEmf;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerDissipative;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerPrimVar;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerElecField;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerMagSlopes;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerTrace;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerHydroShear;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerRemapping;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerShearBorder;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerMakeShearBorder;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerMakeShearBorderSend;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerMakeShearBorderSlopes;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerMakeShearBorderFinalRemapping;
src/hydro/MHDRunGodunovMpi.h:    CudaTimer timerCtUpdate;
src/hydro/MHDRunGodunovMpi.h:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:// include CUDA kernel when necessary
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:      d_U.allocate (make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunBaseMpi.cpp:      d_U2.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunBaseMpi.cpp:      d_U.allocate (make_uint4(isize, jsize, ksize , nbVar), gpuMemAllocType);
src/hydro/HydroRunBaseMpi.cpp:      d_U2.allocate(make_uint4(isize, jsize, ksize , nbVar), gpuMemAllocType);
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    //cudaMemset(d_U.data(), 0, d_U.sizeBytes());
src/hydro/HydroRunBaseMpi.cpp:    //cudaMemset(d_U2.data(), 0, d_U2.sizeBytes());
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    // for time step computation (!!! GPU only !!!)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:	d_randomForcing.allocate(make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:	d_gravity.allocate(make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:	d_gravity.allocate(make_uint3(isize, jsize, 2), gpuMemAllocType);
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:     * - The device border buffer are allocated using cudaMalloc
src/hydro/HydroRunBaseMpi.cpp:     *   (linear memory) instead of cudaMalloc (pitched memory) : this
src/hydro/HydroRunBaseMpi.cpp:     *   and GPU (around 2 GBytes/s instead of ~150 MBytes/s !!!)
src/hydro/HydroRunBaseMpi.cpp:     * - The host border buffers are allocated using cudaMallocHost
src/hydro/HydroRunBaseMpi.cpp:     *   (PINNED) when using the CUDA+MPI version
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:      // #ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:      // #endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:      checkCudaErrorMpi("compute_dt_local 2D",myRank);
src/hydro/HydroRunBaseMpi.cpp:      checkCudaErrorMpi("compute_dt_local 2D copyToHost",myRank);
src/hydro/HydroRunBaseMpi.cpp:      checkCudaErrorMpi("compute_dt_local 3D",myRank);
src/hydro/HydroRunBaseMpi.cpp:      checkCudaErrorMpi("compute_dt_local 3D copyToHost",myRank);
src/hydro/HydroRunBaseMpi.cpp:    } // end call cuda kernel for invDt reduction
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_dt_local -- GPU version
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:      checkCudaErrorMpi("HydroRunBaseMpi cmpdt_2d_mhd error",myRank);
src/hydro/HydroRunBaseMpi.cpp:      checkCudaErrorMpi("HydroRunBaseMpi h_invDt error",myRank);
src/hydro/HydroRunBaseMpi.cpp:      checkCudaErrorMpi("HydroRunBaseMpi cmpdt_3d_mhd error",myRank);
src/hydro/HydroRunBaseMpi.cpp:      checkCudaErrorMpi("HydroRunBaseMpi h_invDt error",myRank);
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBase :: kernel_viscosity_forces_2d",myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_viscosity_flux for 2D data (GPU version)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBase :: kernel_viscosity_forces_3d",myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_viscosity_flux for 3D data (GPU version)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("HydroRunBase :: kernel_viscosity_forces_3d_zslab",myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_viscosity_flux for 3D data (GPU version)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("HydroRunBase compute_random_forcing_normalization error",myRank);
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("HydroRunBase d_randomForcingNormalization copy to host error",myRank);
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBase :: kernel_add_random_forcing_3d",myRank);
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBase :: kernel_hydro_update_2d",myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_hydro_update (2D case, GPU)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBase :: kernel_hydro_update_3d",myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_hydro_update (3D case, GPU)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    // CUDA kernel call
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("HydroRunBase :: kernel_hydro_update_3d_zslab",myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_hydro_update (3D case, GPU)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBase :: kernel_hydro_update_energy_2d",myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_hydro_update_energy (2D case, GPU)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBase :: kernel_hydro_update_energy_3d",myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_hydro_update_energy (3D case, GPU)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBase :: kernel_hydro_update_energy_3d_zslab",myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_hydro_update_energy (3D case, GPU, z-slab method)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:      checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_gravity_predictor_2d", myRank);
src/hydro/HydroRunBaseMpi.cpp:      checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_gravity_predictor_3d", myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_gravity_predictor / GPU version
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_gravity_predictor_3d_zslab", myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_gravity_predictor / GPU version / with zSlab
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:      checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_gravity_source_term_2d", myRank);
src/hydro/HydroRunBaseMpi.cpp:      checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_gravity_source_term_3d", myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_gravity_source_term / GPU version
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_gravity_source_term_3d_zslab", myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_gravity_source_term / GPU version / zslab
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_ct_update_2d", myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_ct_update_2d (2D case, GPU)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_ct_update_3d", myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_ct_update_3d (3D case, GPU)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_ct_update_3d_zslab", myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_ct_update_3d (3D case, GPU, z-slab)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBase :: kernel_resistivity_forces_2d", myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_resistivity_emf_2d (GPU)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_resistivity_forces_3d", myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_resistivity_emf_3d (GPU)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_resistivity_forces_3d_zslab",myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_resistivity_emf_3d (GPU)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_resistivity_energy_flux_2d", myRank);
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::compute_resistivity_energy_flux_2d (GPU)
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_resistivity_energy_flux_3d", myRank);
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    checkCudaErrorMpi("in HydroRunBaseMpi :: kernel_resistivity_energy_flux_3d_zslab", myRank);
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    TIMER_START(timerBoundariesCpuGpu);
src/hydro/HydroRunBaseMpi.cpp:     * needed from GPU memory array U)
src/hydro/HydroRunBaseMpi.cpp:    TIMER_STOP(timerBoundariesCpuGpu);
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:	TIMER_START(timerBoundariesGpu);
src/hydro/HydroRunBaseMpi.cpp:	TIMER_STOP(timerBoundariesGpu);
src/hydro/HydroRunBaseMpi.cpp:	TIMER_START(timerBoundariesGpu);
src/hydro/HydroRunBaseMpi.cpp:	TIMER_STOP(timerBoundariesGpu);
src/hydro/HydroRunBaseMpi.cpp:	TIMER_START(timerBoundariesCpuGpu);
src/hydro/HydroRunBaseMpi.cpp:	TIMER_STOP(timerBoundariesCpuGpu);
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:   * Dump faces data using pnetcdf format (for GPU simulations).
src/hydro/HydroRunBaseMpi.cpp:      // copy X-face from GPU
src/hydro/HydroRunBaseMpi.cpp:      // copy Y-face from GPU
src/hydro/HydroRunBaseMpi.cpp:      // copy Z-face from GPU
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:   * \sa class MHDRunBase (for sequential mono-CPU / mono GPU) version
src/hydro/HydroRunBaseMpi.cpp:      // need to upload gravity field on GPU
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:    // copy data to GPU if necessary
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:  void HydroRunBaseMpi::copyGpuToCpu(int nStep)
src/hydro/HydroRunBaseMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunBaseMpi.cpp:  } // HydroRunBaseMpi::copyGpuToCpu
src/hydro/HydroRunBaseMpi.cpp:    copyGpuToCpu(nStep);
src/hydro/HydroRunBaseMpi.cpp:    copyGpuToCpu(nStep);
src/hydro/HydroRunBaseMpi.cpp:      copyGpuToCpu(nStep);
src/hydro/HydroRunBaseMpi.cpp:      copyGpuToCpu(nStep);
src/hydro/HydroRunBaseMpi.cpp:      copyGpuToCpu(nStep);
src/hydro/CMakeLists.txt:  gpu_macros.cpp
src/hydro/CMakeLists.txt:if (NOT USE_CUDA)
src/hydro/CMakeLists.txt:endif(NOT USE_CUDA)
src/hydro/CMakeLists.txt:  if (NOT USE_CUDA)
src/hydro/CMakeLists.txt:  endif(NOT USE_CUDA)
src/hydro/CMakeLists.txt:# make sure cpp files are recognized as cuda source files
src/hydro/CMakeLists.txt:# when building for GPU executable
src/hydro/CMakeLists.txt:if(USE_CUDA)
src/hydro/CMakeLists.txt:    set_source_files_properties(${file} PROPERTIES LANGUAGE CUDA)
src/hydro/CMakeLists.txt:endif(USE_CUDA)
src/hydro/CMakeLists.txt:  RamsesGPU::cnpy
src/hydro/CMakeLists.txt:  RamsesGPU::config)
src/hydro/CMakeLists.txt:    RamsesGPU::mpiUtils)
src/hydro/CMakeLists.txt:    RamsesGPU::pnetcdf)
src/hydro/CMakeLists.txt:add_library(RamsesGPU::hydro ALIAS hydro)
src/hydro/resistivity_zslab.cuh: * \brief CUDA kernel for computing resistivity forces (MHD only, adapted from Dumses) using z-slab method.
src/hydro/resistivity_zslab.cuh: * CUDA kernel computing resistivity forces (3D data).
src/hydro/resistivity_zslab.cuh: * CUDA kernel computing resistivity forces (3D data, z-slab).
src/hydro/Forcing_OrnsteinUhlenbeck.h:#ifdef __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.h:#ifdef __CUDACC__
src/hydro/Forcing_OrnsteinUhlenbeck.h:    double *d_mode, *d_forcingField, *d_projTens; // CUDA pointers
src/hydro/Forcing_OrnsteinUhlenbeck.h:#ifdef __CUDACC__
src/hydro/shearingBox_utils.cuh: * \brief Defines some CUDA kernels for handling shearing box simulations.
src/hydro/shearingBox_utils.cuh: * Flux/EMF remapping kernel for 3D data at XMIN and XMAX borders (mono GPU only).
src/hydro/shearingBox_utils.cuh: * Update xmin shear border with remapped density flux, only usefull in GPU+MPI.
src/hydro/HydroRunLaxLiu.h:#include "gpu_macros.h"
src/hydro/HydroRunLaxLiu.h:#ifdef __CUDACC__
src/hydro/HydroRunLaxLiu.h:#ifdef __CUDACC__
src/hydro/HydroRunLaxLiu.h:#endif // __CUDACC__
src/hydro/godunov_unsplit_mhd.cuh: * \brief Defines the CUDA kernel for the actual MHD Godunov scheme.
src/hydro/godunov_unsplit_mhd.cuh: * Define some CUDA kernel common to all MHD implementation on GPU
src/hydro/godunov_unsplit_mhd.cuh: * Define some CUDA kernel to implement MHD version 3 on GPU
src/hydro/HydroRunRelaxingTVD.h:#include "gpu_macros.h"
src/hydro/HydroRunRelaxingTVD.h:#ifdef __CUDACC__
src/hydro/HydroRunRelaxingTVD.h:#include "../utils/monitoring/CudaTimer.h"
src/hydro/HydroRunRelaxingTVD.h:#endif // __CUDACC__
src/hydro/HydroRunRelaxingTVD.h:    //!  wrapper that performs the actual CPU or GPU scheme
src/hydro/HydroRunRelaxingTVD.h:#ifdef __CUDACC__
src/hydro/HydroRunRelaxingTVD.h:    //! Actual computation of the relaxing TVD integration on GPU using
src/hydro/HydroRunRelaxingTVD.h:    void relaxing_tvd_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunRelaxingTVD.h:#endif // __CUDACC__
src/hydro/HydroRunRelaxingTVD.h:#ifdef __CUDACC__
src/hydro/HydroRunRelaxingTVD.h:    CudaTimer timerBoundaries;
src/hydro/HydroRunRelaxingTVD.h:    CudaTimer timerRelaxingTVD;
src/hydro/HydroRunRelaxingTVD.h:#endif // __CUDACC__
src/hydro/make_boundary_common.h: * \brief Some constant parameter used for CUDA kernels geometry.
src/hydro/make_boundary_common.h: * in multiple file (no need to have the full CUDA kernel definition included several
src/hydro/make_boundary_common.h:/** only usefull for GPU implementation */
src/hydro/HydroRunBase.cpp:// include CUDA kernel when necessary
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:      d_U.allocate (make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunBase.cpp:      d_U2.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunBase.cpp:      d_U.allocate (make_uint4(isize, jsize, ksize , nbVar), gpuMemAllocType);
src/hydro/HydroRunBase.cpp:      d_U2.allocate(make_uint4(isize, jsize, ksize , nbVar), gpuMemAllocType);
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    //cudaMemset(d_U.data(), 0, d_U.sizeBytes());
src/hydro/HydroRunBase.cpp:    //cudaMemset(d_U2.data(), 0, d_U2.sizeBytes());
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:	d_randomForcing.allocate(make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:	d_gravity.allocate(make_uint4(isize, jsize, ksize, 3), gpuMemAllocType);
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:	d_gravity.allocate(make_uint3(isize, jsize, 2), gpuMemAllocType);
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:      checkCudaError("HydroRunBase cmpdt_2d error");
src/hydro/HydroRunBase.cpp:      checkCudaError("HydroRunBase d_invDt copy to host error");
src/hydro/HydroRunBase.cpp:      checkCudaError("HydroRunBase cmpdt_3d error");
src/hydro/HydroRunBase.cpp:      checkCudaError("HydroRunBase d_invDt copy to host error");
src/hydro/HydroRunBase.cpp:    } // end call cuda kernel for invDt reduction
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_dt -- GPU version
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    checkCudaError("in HydroRunBase :: kernel_viscosity_forces_2d");
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_viscosity_flux for 2D data (GPU version)
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    checkCudaError("in HydroRunBase :: kernel_viscosity_forces_3d");
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_viscosity_flux for 3D data (GPU version)
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    checkCudaError("HydroRunBase :: kernel_viscosity_forces_3d_zslab");
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_viscosity_flux for 3D data (GPU version)
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    checkCudaError("HydroRunBase compute_random_forcing_normalization error");
src/hydro/HydroRunBase.cpp:    checkCudaError("HydroRunBase d_randomForcingNormalization copy to host error");
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    checkCudaError("in HydroRunBase :: kernel_add_random_forcing_3d");
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    checkCudaError("in HydroRunBase :: kernel_hydro_update_2d");
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_hydro_update (2D case, GPU)
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    checkCudaError("in HydroRunBase :: kernel_hydro_update_3d");
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_hydro_update (3D case, GPU)
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    // CUDA kernel call
src/hydro/HydroRunBase.cpp:    checkCudaError("HydroRunBase :: kernel_hydro_update_3d_zslab");
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_hydro_update (3D case, GPU)
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    checkCudaError("in HydroRunBase :: kernel_hydro_update_energy_2d");
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_hydro_update_energy (2D case, GPU)
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    checkCudaError("in HydroRunBase :: kernel_hydro_update_energy_3d");
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_hydro_update_energy (3D case, GPU)
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    checkCudaError("in HydroRunBase :: kernel_hydro_update_energy_3d_zslab");
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_hydro_update_energy (3D case, GPU, z-slab method)
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_gravity_predictor / GPU version
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    checkCudaError("in HydroRunBase :: kernel_gravity_predictor_3d_zslab");
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_gravity_predictor / GPU version / with zSlab
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_gravity_source_term / GPU version
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:    checkCudaError("in HydroRunBase :: kernel_gravity_source_term_3d_zslab");
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::compute_gravity_source_term / GPU version / zslab
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:	// the actuall cuda kernels are called inside
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:  } // make_jet (GPU version)
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:   * Dump only X,Y and Z faces of the simulation domain (GPU version).
src/hydro/HydroRunBase.cpp:    // copy X-face from GPU
src/hydro/HydroRunBase.cpp:    // copy Y-face from GPU
src/hydro/HydroRunBase.cpp:    // copy Z-face from GPU
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:  void HydroRunBase::copyGpuToCpu(int nStep)
src/hydro/HydroRunBase.cpp:#ifdef __CUDACC__
src/hydro/HydroRunBase.cpp:#endif // __CUDACC__
src/hydro/HydroRunBase.cpp:  } // HydroRunBase::copyGpuToCpu
src/hydro/HydroRunBase.cpp:      copyGpuToCpu(nStep);
src/hydro/common_types.h: * \brief Defines some custom types for compatibility with CUDA.
src/hydro/common_types.h:/* if not using CUDA, then defines customs types from vector_types.h */
src/hydro/common_types.h:#ifndef __CUDACC__
src/hydro/common_types.h: * \brief structure used to set CUDA block sizes.
src/hydro/common_types.h:#endif /* __CUDACC__ */
src/hydro/common_types.h:#ifdef __CUDACC__
src/hydro/common_types.h:// the following is only in CUDA 3.0
src/hydro/common_types.h:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:// include CUDA kernel when necessary
src/hydro/HydroRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:    d_Q.allocate   (make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:      d_qm_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslab.cpp:      d_qm_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslab.cpp:      d_qm_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslab.cpp:      d_qp_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslab.cpp:      d_qp_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslab.cpp:      d_qp_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:      d_slope_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslab.cpp:      d_slope_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslab.cpp:      d_slope_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslab.cpp:      d_qm.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslab.cpp:      d_qp.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:      cutilSafeCall( cudaMemGetInfo(&freeMemory, &totalMemory) );
src/hydro/HydroRunGodunovZslab.cpp:      std::cout << "Total memory available on GPU " << totalMemory/1000000. << " MBytes\n";
src/hydro/HydroRunGodunovZslab.cpp:      std::cout << "Currently free  memory on GPU " <<  freeMemory/1000000. << " MBytes\n";
src/hydro/HydroRunGodunovZslab.cpp:      std::cout << "Total memory allocated on GPU " << DeviceArray<real_t>::totalAllocMemoryInKB/1000. << " MBytes\n";
src/hydro/HydroRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:      godunov_unsplit_gpu(d_U , d_U2, dt, nStep);
src/hydro/HydroRunGodunovZslab.cpp:      godunov_unsplit_gpu(d_U2, d_U , dt, nStep);
src/hydro/HydroRunGodunovZslab.cpp:  } // HydroRunGodunovZslab::godunov_unsplit (GPU version)
src/hydro/HydroRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:#ifdef __CUDACC__ 
src/hydro/HydroRunGodunovZslab.cpp:  void HydroRunGodunovZslab::godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslab.cpp:      godunov_unsplit_gpu_v0(d_UOld, d_UNew, dt, nStep);
src/hydro/HydroRunGodunovZslab.cpp:      godunov_unsplit_gpu_v1(d_UOld, d_UNew, dt, nStep);
src/hydro/HydroRunGodunovZslab.cpp:      godunov_unsplit_gpu_v2(d_UOld, d_UNew, dt, nStep);
src/hydro/HydroRunGodunovZslab.cpp:  } // HydroRunGodunovZslab::godunov_unsplit_gpu
src/hydro/HydroRunGodunovZslab.cpp:  void HydroRunGodunovZslab::godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslab.cpp:	checkCudaError("HydroRunGodunovZslab :: kernel_hydro_compute_primitive_variables_3D_zslab error");
src/hydro/HydroRunGodunovZslab.cpp:	checkCudaError("HydroRunGodunovZslab :: kernel_godunov_unsplit_3d_v0_zslab error");
src/hydro/HydroRunGodunovZslab.cpp:  } // HydroRunGodunovZslab::godunov_unsplit_gpu_v0
src/hydro/HydroRunGodunovZslab.cpp:  void HydroRunGodunovZslab::godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslab.cpp:	checkCudaError("HydroRunGodunovZslab :: kernel_hydro_compute_primitive_variables_3D_zslab error");
src/hydro/HydroRunGodunovZslab.cpp:	checkCudaError("HydroRunGodunovZslab :: kernel_hydro_compute_trace_unsplit_3d_v1_zslab error");
src/hydro/HydroRunGodunovZslab.cpp:	checkCudaError("HydroRunGodunovZslab :: kernel_hydro_flux_update_unsplit_3d_v1_zslab error");
src/hydro/HydroRunGodunovZslab.cpp:  } // HydroRunGodunovZslab::godunov_unsplit_gpu_v1
src/hydro/HydroRunGodunovZslab.cpp:  void HydroRunGodunovZslab::godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslab.cpp:	checkCudaError("HydroRunGodunovZslab :: kernel_hydro_compute_primitive_variables_3D_zslab error");
src/hydro/HydroRunGodunovZslab.cpp:  } // HydroRunGodunovZslab::godunov_unsplit_gpu_v2
src/hydro/HydroRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:	  copyGpuToCpu(nStep);
src/hydro/HydroRunGodunovZslab.cpp:	  copyGpuToCpu(nStep);
src/hydro/HydroRunGodunovZslab.cpp:      copyGpuToCpu(nStep);
src/hydro/HydroRunGodunovZslab.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslab.cpp:#endif // __CUDACC__
src/hydro/random_forcing.cuh: * \brief CUDA kernel for random forcing.
src/hydro/random_forcing.cuh: * CUDA kernel to add random forcing field to velocity (3D data only).
src/hydro/random_forcing.cuh:  // see CUDA documentation of the reduction example, especially the
src/hydro/structureFunctions.cpp: * Compute structure functions for mono-CPU or mono-GPU.
src/hydro/godunov_unsplit_mhd_v0_old.cuh: * \brief Defines the CUDA kernel for the actual MHD Godunov scheme.
src/hydro/viscosity_zslab.cuh: * \brief CUDA kernel for computing viscosity forces inside z-slab.
src/hydro/viscosity_zslab.cuh: * CUDA kernel computing viscosity forces (3D data) inside z-slab.
src/hydro/constants.h:#include "gpu_macros.h"
src/hydro/constants.h:#if defined(__CUDACC__) // NVCC
src/hydro/constants.h:  SG_FFT_DECOMP2D, /* P3DFFT (CPU) or DiGPFFT (GPU) */
src/hydro/constants.h:  FF_HDF5    = 0, /*!< for both mono/multi GPU applications */
src/hydro/constants.h:  FF_NETCDF  = 1, /*!< for mono GPU applications */
src/hydro/constants.h:  FF_PNETCDF = 2, /*!< for both mono/multi GPU applications (best performances) */
src/hydro/constants.h:/** list of array pointers (mostly usefull only in GPU version
src/hydro/constants.h: *  instead of being passed to CUDA kernels as arguments) */
src/hydro/constants.h: * should go to constant memory in the CUDA/GPU version (i.e. to be
src/hydro/constants.h: * copied to device memory using cudaMemcpyToSymbol).
src/hydro/constants.h:#ifdef __CUDACC__
src/hydro/constants.h:# if __CUDA_ARCH__ >= 200
src/hydro/constants.h:#endif // __CUDACC__
src/hydro/constants.h:#ifdef __CUDACC__
src/hydro/constants.h:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp: * CUDA and MPI implementation.
src/hydro/HydroRunGodunovMpi.cpp:// include CUDA kernel when necessary
src/hydro/HydroRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:	d_Q.allocate   (make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	d_Q.allocate   (make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:	  d_qm_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_qm_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_qp_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_qp_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_qm_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_qm_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_qm_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_qp_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_qp_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_qp_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:	  d_slope_x.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_slope_y.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_qm.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_qp.allocate(make_uint3(isize, jsize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_slope_x.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_slope_y.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_slope_z.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_qm.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:	  d_qp.allocate(make_uint4(isize, jsize, ksize, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:	cutilSafeCall( cudaMemGetInfo(&freeMemory, &totalMemory) );
src/hydro/HydroRunGodunovMpi.cpp:	std::cout << "Total memory available on GPU " << totalMemory/1000000. << " MBytes\n";
src/hydro/HydroRunGodunovMpi.cpp:	std::cout << "Currently free  memory on GPU " <<  freeMemory/1000000. << " MBytes\n";
src/hydro/HydroRunGodunovMpi.cpp:	std::cout << "Total memory allocated on GPU " << DeviceArray<real_t>::totalAllocMemoryInKB/1000. << " MBytes\n";
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U , d_U2, XDIR, dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U2, d_U , YDIR, dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U , d_U2, YDIR, dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U2, d_U , XDIR, dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U , d_U2, XDIR, dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U2, d_U , YDIR, dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U , d_U2, ZDIR, dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U2, d_U , YDIR,dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U , d_U2, ZDIR,dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U2, d_U , XDIR,dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U , d_U2, ZDIR,dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U2, d_U , YDIR,dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U , d_U2, XDIR,dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U2, d_U , XDIR, dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U , d_U2, YDIR, dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U2, d_U , ZDIR, dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U , d_U2, YDIR,dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U2, d_U , ZDIR,dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U , d_U2, XDIR,dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U2, d_U , ZDIR,dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U , d_U2, YDIR,dt);
src/hydro/HydroRunGodunovMpi.cpp:	godunov_split_gpu(d_U2, d_U , XDIR,dt);
src/hydro/HydroRunGodunovMpi.cpp:  } // HydroRunGodunovMpi::godunov_split (GPU version)
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:      godunov_unsplit_gpu(d_U , d_U2, dt);
src/hydro/HydroRunGodunovMpi.cpp:      godunov_unsplit_gpu(d_U2, d_U , dt);
src/hydro/HydroRunGodunovMpi.cpp:  } // HydroRunGodunovMpi::godunov_unsplit (GPU version)
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#ifdef __CUDACC__ 
src/hydro/HydroRunGodunovMpi.cpp:  void HydroRunGodunovMpi::godunov_split_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_x_2d_v2",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_x_2d_v1",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_x_notrace_2d",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_y_2d_v2",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_y_2d_v1",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_y_notrace_2d",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_x_3d_v2",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_x_3d_v1",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_x_notrace_3d",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_y_3d_v2",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_y_3d_v1",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_y_notrace_3d",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_z_3d_v2",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_z_3d_v1",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	    checkCudaErrorMpi("godunov_z_notrace_3d",myRank);
src/hydro/HydroRunGodunovMpi.cpp:  } // HydroRunGodunovMpi::godunov_split_gpu
src/hydro/HydroRunGodunovMpi.cpp:  void HydroRunGodunovMpi::godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovMpi.cpp:      godunov_unsplit_gpu_v0(d_UOld, d_UNew, dt);
src/hydro/HydroRunGodunovMpi.cpp:      godunov_unsplit_gpu_v1(d_UOld, d_UNew, dt);
src/hydro/HydroRunGodunovMpi.cpp:      godunov_unsplit_gpu_v2(d_UOld, d_UNew, dt);
src/hydro/HydroRunGodunovMpi.cpp:  } // HydroRunGodunovMpi::godunov_unsplit_gpu
src/hydro/HydroRunGodunovMpi.cpp:  void HydroRunGodunovMpi::godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovMpi.cpp:      // checkCudaError("HydroRunGodunov :: kernel_godunov_unsplit_2d error");
src/hydro/HydroRunGodunovMpi.cpp:	checkCudaErrorMpi("HydroRunGodunov :: kernel_godunov_unsplit_2d_v0 error",myRank);
src/hydro/HydroRunGodunovMpi.cpp:      // checkCudaError("HydroRunGodunov :: kernel_godunov_unsplit_3d error");
src/hydro/HydroRunGodunovMpi.cpp:      checkCudaErrorMpi("HydroRunGodunov :: kernel_godunov_unsplit_3d_v0 error",myRank);
src/hydro/HydroRunGodunovMpi.cpp:  } // HydroRunGodunovMpi::godunov_unsplit_gpu_v0
src/hydro/HydroRunGodunovMpi.cpp:  void HydroRunGodunovMpi::godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovMpi.cpp:	checkCudaErrorMpi("HydroRunGodunov :: kernel_hydro_compute_trace_unsplit_2d_v1 error",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	checkCudaErrorMpi("HydroRunGodunov :: kernel_hydro_flux_update_unsplit_2d_v1< error",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	checkCudaErrorMpi("HydroRunGodunov :: kernel_hydro_compute_trace_unsplit_3d_v1 error",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	checkCudaErrorMpi("HydroRunGodunov :: kernel_hydro_flux_update_unsplit_3d_v1 error",myRank);
src/hydro/HydroRunGodunovMpi.cpp:  } // HydroRunGodunovMpi::godunov_unsplit_gpu_v1
src/hydro/HydroRunGodunovMpi.cpp:  void HydroRunGodunovMpi::godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovMpi.cpp:  } // HydroRunGodunovMpi::godunov_unsplit_gpu_v2
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#ifndef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#ifndef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#ifndef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:	  copyGpuToCpu(nStep);
src/hydro/HydroRunGodunovMpi.cpp:	  copyGpuToCpu(nStep);
src/hydro/HydroRunGodunovMpi.cpp:      copyGpuToCpu(nStep);
src/hydro/HydroRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:    printf("Euler godunov boundaries pure GPU     [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerBoundariesGpu.elapsed(), timerBoundariesGpu.elapsed()/timerTotal.elapsed()*100.);
src/hydro/HydroRunGodunovMpi.cpp:    printf("Euler godunov boundaries CPU-GPU comm [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerBoundariesCpuGpu.elapsed(), timerBoundariesCpuGpu.elapsed()/timerTotal.elapsed()*100.);
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovMpi.cpp:	checkCudaErrorMpi("HydroRunGodunov :: kernel_hydro_compute_primitive_variables_2D error",myRank);
src/hydro/HydroRunGodunovMpi.cpp:	checkCudaErrorMpi("HydroRunGodunov :: kernel_hydro_compute_primitive_variables_3D error",myRank);
src/hydro/HydroRunGodunovMpi.cpp:#endif // __CUDACC__
src/hydro/relaxingTVD.cuh: * \brief Defines the CUDA kernel for the relaxing TVD scheme.
src/hydro/HydroRunKT.h:#include "gpu_macros.h"
src/hydro/HydroRunKT.h:  void copyGpuToCpu(int nStep=0);
src/hydro/HydroRunKT.h:#ifdef __CUDACC__
src/hydro/HydroRunKT.h:#ifdef __CUDACC__
src/hydro/HydroRunKT.h:#endif // __CUDACC__
src/hydro/HydroRunKT.h:#ifdef __CUDACC__
src/hydro/HydroRunKT.h:#endif // __CUDACC__
src/hydro/relaxingTVD.h: * \brief Some utilities used either in the CPU or GPU version of the relaxing TVD scheme.
src/hydro/Makefile.am:# also note that SUFFIXES must be set with .cu before including cuda.am
src/hydro/Makefile.am:include $(top_srcdir)/am/cuda.am
src/hydro/Makefile.am:AM_CXXFLAGS = -I$(srcdir)/. $(CONFIG_CPPFLAGS) $(CNPY_CPPFLAGS) $(CUDA_CFLAGS) $(QT_CXXFLAGS) $(VTK_CPPFLAGS) $(HDF5_CPPFLAGS) $(NETCDF_CPPFLAGS) $(PNETCDF_CPPFLAGS) $(GM_CPPFLAGS) $(MPI_CXXFLAGS) $(TIMING_FLAGS) $(PAPI_CPPFLAGS)
src/hydro/Makefile.am:AM_CFLAGS   = -I$(srcdir)/. $(CONFIG_CPPFLAGS) $(CNPY_CPPFLAGS) $(CUDA_CFLAGS) $(QT_CPPFLAGS) $(VTK_CPPFLAGS) $(HDF5_CPPFLAGS) $(NETCDF_CPPFLAGS) $(PNETCDF_CPPFLAGS) $(GM_CPPFLAGS) $(MPI_CXXFLAGS) $(TIMING_FLAGS) $(PAPI_CPPFLAGS)
src/hydro/Makefile.am:AM_CPPFLAGS = -I$(srcdir)/. $(CONFIG_CPPFLAGS) $(CNPY_CPPFLAGS) $(CUDA_CFLAGS) $(QT_CPPFLAGS) $(VTK_CPPFLAGS) $(HDF5_CPPFLAGS) $(NETCDF_CPPFLAGS) $(PNETCDF_CPPFLAGS) $(GM_CPPFLAGS) $(MPI_CXXFLAGS) $(TIMING_FLAGS) $(PAPI_CPPFLAGS)
src/hydro/Makefile.am:# common sources code for both sequential (CPU+CUDA) and parallel (MPI+CUDA)
src/hydro/Makefile.am:	gpu_macros.h \
src/hydro/Makefile.am:	gpu_macros.cpp \
src/hydro/Makefile.am:HYDRO_SRC_CUDA = \
src/hydro/Makefile.am:	gpu_macros.h \
src/hydro/Makefile.am:HYDRO_SRC_CUDA_SIMPLE = $(HYDRO_SRC_CUDA) \
src/hydro/Makefile.am:	gpu_macros.cu \
src/hydro/Makefile.am:HYDRO_SRC_CUDA_DOUBLE = $(HYDRO_SRC_CUDA) \
src/hydro/Makefile.am:	gpu_macros-double.cu \
src/hydro/Makefile.am:HYDRO_SRC_CUDA_SIMPLE += \
src/hydro/Makefile.am:HYDRO_SRC_CUDA_DOUBLE += \
src/hydro/Makefile.am:# CUDA sequential (mono CPU - mono GPU) or CUDA - MPI (multi-GPU)
src/hydro/Makefile.am:if USE_CUDA
src/hydro/Makefile.am:BUILT_SOURCES = gpu_macros.cu RandomGen.cu Forcing_OrnsteinUhlenbeck.cu HydroRunBase.cu HydroRunGodunov.cu HydroRunGodunovZslab.cu HydroRunKT.cu HydroRunRelaxingTVD.cu HydroRunLaxLiu.cu MHDRunBase.cu MHDRunGodunov.cu MHDRunGodunovZslab.cu structureFunctions.cu
src/hydro/Makefile.am:CLEANFILES   += gpu_macros.cu RandomGen.cu Forcing_OrnsteinUhlenbeck.cu HydroRunBase.cu HydroRunGodunov.cu HydroRunGodunovZslab.cu HydroRunKT.cu HydroRunRelaxingTVD.cu HydroRunLaxLiu.cu MHDRunBase.cu MHDRunGodunov.cu MHDRunGodunovZslab.cu structureFunctions.cu 
src/hydro/Makefile.am:BUILT_SOURCES += gpu_macros-double.cu RandomGen-double.cu Forcing_OrnsteinUhlenbeck-double.cu HydroRunBase-double.cu HydroRunGodunov-double.cu HydroRunGodunovZslab-double.cu HydroRunKT-double.cu HydroRunRelaxingTVD-double.cu MHDRunBase-double.cu MHDRunGodunov-double.cu MHDRunGodunovZslab-double.cu structureFunctions-double.cu
src/hydro/Makefile.am:CLEANFILES    += gpu_macros-double.cu RandomGen-double.cu Forcing_OrnsteinUhlenbeck-double.cu HydroRunBase-double.cu HydroRunGodunov-double.cu HydroRunGodunovZslab-double.cu HydroRunKT-double.cu HydroRunRelaxingTVD-double.cu MHDRunBase-double.cu MHDRunGodunov-double.cu MHDRunGodunovZslab-double.cu structureFunctions-double.cu
src/hydro/Makefile.am:noinst_LTLIBRARIES += libhydroGpu.la
src/hydro/Makefile.am:libhydroGpu_la_SOURCES = $(HYDRO_SRC_CUDA_SIMPLE)
src/hydro/Makefile.am:libhydroGpu_la_CPPFLAGS = $(AM_CPPFLAGS) $(CUDA_CFLAGS) -I$(srcdir)/../utils/mpiUtils
src/hydro/Makefile.am:libhydroGpu_la_CXXFLAGS = $(AM_CXXFLAGS) $(CUDA_CFLAGS) -I$(srcdir)/../utils/mpiUtils  -std=c++11
src/hydro/Makefile.am:noinst_LTLIBRARIES += libhydroGpu_double.la
src/hydro/Makefile.am:libhydroGpu_double_la_SOURCES = $(HYDRO_SRC_CUDA_DOUBLE)
src/hydro/Makefile.am:libhydroGpu_double_la_CPPFLAGS = $(AM_CPPFLAGS) $(CUDA_CFLAGS) -I$(srcdir)/../utils/mpiUtils -DUSE_DOUBLE
src/hydro/Makefile.am:libhydroGpu_double_la_CXXFLAGS = $(AM_CXXFLAGS) $(CUDA_CFLAGS) -I$(srcdir)/../utils/mpiUtils -DUSE_DOUBLE -std=c++11
src/hydro/copyFaces.cuh: * \brief Some CUDA kernel for copying faces of a 3D simulation domain.
src/hydro/HydroMpiParameters.h: * version (1 CPU/GPU).
src/hydro/HydroMpiParameters.h:   * computations, output files) in a pure MPI or MPI/CUDA environnement.
src/hydro/mhd_utils.h: * \brief Small MHD related utilities common to CPU / GPU code.
src/hydro/HydroRunGodunovZslabMpi.cpp: * CUDA and MPI implementation.
src/hydro/HydroRunGodunovZslabMpi.cpp:// include CUDA kernel when necessary
src/hydro/HydroRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:    d_Q.allocate   (make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:      d_qm_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslabMpi.cpp:      d_qm_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslabMpi.cpp:      d_qm_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslabMpi.cpp:      d_qp_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslabMpi.cpp:      d_qp_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslabMpi.cpp:      d_qp_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:      d_slope_x.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslabMpi.cpp:      d_slope_y.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslabMpi.cpp:      d_slope_z.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslabMpi.cpp:      d_qm.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslabMpi.cpp:      d_qp.allocate(make_uint4(isize, jsize, zSlabWidthG, nbVar), gpuMemAllocType);
src/hydro/HydroRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:      cutilSafeCall( cudaMemGetInfo(&freeMemory, &totalMemory) );
src/hydro/HydroRunGodunovZslabMpi.cpp:      std::cout << "Total memory available on GPU " << totalMemory/1000000. << " MBytes\n";
src/hydro/HydroRunGodunovZslabMpi.cpp:      std::cout << "Currently free  memory on GPU " <<  freeMemory/1000000. << " MBytes\n";
src/hydro/HydroRunGodunovZslabMpi.cpp:      std::cout << "Total memory allocated on GPU " << DeviceArray<real_t>::totalAllocMemoryInKB/1000. << " MBytes\n";
src/hydro/HydroRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:      godunov_unsplit_gpu(d_U , d_U2, dt);
src/hydro/HydroRunGodunovZslabMpi.cpp:      godunov_unsplit_gpu(d_U2, d_U , dt);
src/hydro/HydroRunGodunovZslabMpi.cpp:  } // HydroRunGodunovZslabMpi::godunov_unsplit (GPU version)
src/hydro/HydroRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:#ifdef __CUDACC__ 
src/hydro/HydroRunGodunovZslabMpi.cpp:  void HydroRunGodunovZslabMpi::godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslabMpi.cpp:	godunov_unsplit_gpu_v0(d_UOld, d_UNew, dt);
src/hydro/HydroRunGodunovZslabMpi.cpp:	godunov_unsplit_gpu_v1(d_UOld, d_UNew, dt);
src/hydro/HydroRunGodunovZslabMpi.cpp:	godunov_unsplit_gpu_v2(d_UOld, d_UNew, dt);
src/hydro/HydroRunGodunovZslabMpi.cpp:  } // HydroRunGodunovZslabMpi::godunov_unsplit_gpu
src/hydro/HydroRunGodunovZslabMpi.cpp:  void HydroRunGodunovZslabMpi::godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslabMpi.cpp:	checkCudaError("HydroRunGodunovZslab :: kernel_hydro_compute_primitive_variables_3D_zslab error");
src/hydro/HydroRunGodunovZslabMpi.cpp:	checkCudaError("HydroRunGodunovZslab :: kernel_godunov_unsplit_3d_v0_zslab error");
src/hydro/HydroRunGodunovZslabMpi.cpp:  } // HydroRunGodunovZslabMpi::godunov_unsplit_gpu_v0
src/hydro/HydroRunGodunovZslabMpi.cpp:  void HydroRunGodunovZslabMpi::godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslabMpi.cpp:	  checkCudaError("HydroRunGodunovZslab :: kernel_hydro_compute_primitive_variables_3D_zslab error");
src/hydro/HydroRunGodunovZslabMpi.cpp:	  checkCudaError("HydroRunGodunovZslab :: kernel_hydro_compute_trace_unsplit_3d_v1_zslab error");
src/hydro/HydroRunGodunovZslabMpi.cpp:	  checkCudaError("HydroRunGodunovZslab :: kernel_hydro_flux_update_unsplit_3d_v1_zslab error");
src/hydro/HydroRunGodunovZslabMpi.cpp:  } // HydroRunGodunovZslabMpi::godunov_unsplit_gpu_v1
src/hydro/HydroRunGodunovZslabMpi.cpp:  void HydroRunGodunovZslabMpi::godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
src/hydro/HydroRunGodunovZslabMpi.cpp:	checkCudaError("HydroRunGodunovZslab :: kernel_hydro_compute_primitive_variables_3D_zslab error");
src/hydro/HydroRunGodunovZslabMpi.cpp:  } // HydroRunGodunovZslabMpi::godunov_unsplit_gpu_v2
src/hydro/HydroRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:	  copyGpuToCpu(nStep);
src/hydro/HydroRunGodunovZslabMpi.cpp:	  copyGpuToCpu(nStep);
src/hydro/HydroRunGodunovZslabMpi.cpp:      copyGpuToCpu(nStep);
src/hydro/HydroRunGodunovZslabMpi.cpp:#ifdef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:    printf("Euler godunov boundaries pure GPU     [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerBoundariesGpu.elapsed(), timerBoundariesGpu.elapsed()/timerTotal.elapsed()*100.);
src/hydro/HydroRunGodunovZslabMpi.cpp:    printf("Euler godunov boundaries CPU-GPU comm [MPI rank %3d] : %5.3f sec (%5.2f %% of total time)\n", myRank, timerBoundariesCpuGpu.elapsed(), timerBoundariesCpuGpu.elapsed()/timerTotal.elapsed()*100.);
src/hydro/HydroRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:#ifndef __CUDACC__
src/hydro/HydroRunGodunovZslabMpi.cpp:#endif // __CUDACC__
src/hydro/shearingBox_utils_zslab.cuh: * \brief Defines some CUDA kernels for handling shearing box simulations, with z-slab method.
src/hydro/shearingBox_utils_zslab.cuh: * Flux/EMF remapping kernel for 3D data at XMIN and XMAX borders (mono GPU only).
src/hydro/shearingBox_utils_zslab.cuh: * Update xmin shear border with remapped density flux, only usefull in GPU+MPI.
src/hydro/real_type.h:#ifdef __CUDACC__
src/hydro/real_type.h:#include "gpu_macros.h"
src/hydro/real_type.h:// #ifdef __CUDACC__
src/hydro/real_type.h:// #ifdef __CUDACC__
src/hydro/MHDRunGodunov.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.h:#include "../utils/monitoring/CudaTimer.h"
src/hydro/MHDRunGodunov.h:#endif // __CUDACC__
src/hydro/MHDRunGodunov.h:#ifndef __CUDACC__
src/hydro/MHDRunGodunov.h:#endif // __CUDACC__
src/hydro/MHDRunGodunov.h:   * available on GPU (2D only)
src/hydro/MHDRunGodunov.h:   * available on GPU (2D + 3D)
src/hydro/MHDRunGodunov.h:   * available on GPU (not available)
src/hydro/MHDRunGodunov.h:   * memory buffer h_elec (or d_elec in the GPU version).<BR>
src/hydro/MHDRunGodunov.h:   * available on GPU (3D only)<BR>
src/hydro/MHDRunGodunov.h:   * Note that this 3D GPU version was really hard to debug (see Trac
src/hydro/MHDRunGodunov.h:    //! In the GPU version, the conversion is done on line, inside
src/hydro/MHDRunGodunov.h:    //! see godunov_unsplit_gpu or godunov_unsplit_cpu.
src/hydro/MHDRunGodunov.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.h:    //! scheme on GPU, two array are necessary to make ping-pong (d_UOld and
src/hydro/MHDRunGodunov.h:    void godunov_unsplit_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.h:    void godunov_unsplit_gpu_v0(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.h:    void godunov_unsplit_gpu_v0_old(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.h:    void godunov_unsplit_gpu_v1(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.h:    void godunov_unsplit_gpu_v2(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.h:    void godunov_unsplit_gpu_v3(DeviceArray<real_t>& d_UOld,
src/hydro/MHDRunGodunov.h:    void godunov_unsplit_gpu_v4(DeviceArray<real_t>& d_UOld,
src/hydro/MHDRunGodunov.h:#endif // __CUDAC__
src/hydro/MHDRunGodunov.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.h:    //! Same routine as godunov_unsplit_gpu but with rotating
src/hydro/MHDRunGodunov.h:    void godunov_unsplit_rotating_gpu(DeviceArray<real_t>& d_UOld, 
src/hydro/MHDRunGodunov.h:#endif // __CUDACC__
src/hydro/MHDRunGodunov.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.h:#endif // __CUDACC__
src/hydro/MHDRunGodunov.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.h:#endif // __CUDACC__
src/hydro/MHDRunGodunov.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.h:    DeviceArray<real_t> d_Q; //!< GPU : primitive data array
src/hydro/MHDRunGodunov.h:#endif // __CUDACC__
src/hydro/MHDRunGodunov.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.h:    DeviceArray<real_t> d_qm_x; //!< GPU array for qm state along X
src/hydro/MHDRunGodunov.h:    DeviceArray<real_t> d_qm_y; //!< GPU array for qm state along Y
src/hydro/MHDRunGodunov.h:    DeviceArray<real_t> d_qm_z; //!< GPU array for qm state along Z
src/hydro/MHDRunGodunov.h:    DeviceArray<real_t> d_qp_x; //!< GPU array for qp state along X
src/hydro/MHDRunGodunov.h:    DeviceArray<real_t> d_qp_y; //!< GPU array for qp state along Y
src/hydro/MHDRunGodunov.h:    DeviceArray<real_t> d_qp_z; //!< GPU array for qp state along Z
src/hydro/MHDRunGodunov.h:    DeviceArray<real_t> d_emf; //!< GPU array for electromotive forces
src/hydro/MHDRunGodunov.h:#endif // __CUDACC__
src/hydro/MHDRunGodunov.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.h:#endif // __CUDACC__
src/hydro/MHDRunGodunov.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.h:#endif // __CUDACC__
src/hydro/MHDRunGodunov.h:#ifdef __CUDACC__
src/hydro/MHDRunGodunov.h:    CudaTimer timerBoundaries;
src/hydro/MHDRunGodunov.h:    CudaTimer timerGodunov;
src/hydro/MHDRunGodunov.h:    CudaTimer timerTraceUpdate;
src/hydro/MHDRunGodunov.h:    CudaTimer timerUpdate;
src/hydro/MHDRunGodunov.h:    CudaTimer timerEmf;
src/hydro/MHDRunGodunov.h:    CudaTimer timerDissipative;
src/hydro/MHDRunGodunov.h:    CudaTimer timerPrimVar;
src/hydro/MHDRunGodunov.h:    CudaTimer timerElecField;
src/hydro/MHDRunGodunov.h:    CudaTimer timerMagSlopes;
src/hydro/MHDRunGodunov.h:    CudaTimer timerTrace;
src/hydro/MHDRunGodunov.h:    CudaTimer timerHydroShear;
src/hydro/MHDRunGodunov.h:    CudaTimer timerRemapping;
src/hydro/MHDRunGodunov.h:    CudaTimer timerShearBorder;
src/hydro/MHDRunGodunov.h:    CudaTimer timerCtUpdate;
src/hydro/MHDRunGodunov.h:#endif // __CUDACC__
src/hydro/structureFunctions.h: * Compute structure functions for mono-CPU or mono-GPU.
src/hydro/kurganov-tadmor.cuh: * \brief Implement GPU kernels for the Kurganov-Tadmor central scheme.
src/hydro/hydro_update_zslab.cuh: * \brief CUDA kernel for update conservative variables with flux array inside z-slab.
src/hydro/hydro_update_zslab.cuh: * CUDA kernel perform hydro update from flux arrays (3D data) inside zslab.
src/hydro/hydro_update_zslab.cuh: * CUDA kernel perform hydro update (energy only) from flux arrays (3D data, z-slab).
src/hydro/gpu_macros.h: * \file gpu_macros.h
src/hydro/gpu_macros.h: * \brief Some useful GPU related macros.
src/hydro/gpu_macros.h: * $Id: gpu_macros.h 1784 2012-02-21 10:34:58Z pkestene $
src/hydro/gpu_macros.h:#ifndef GPU_MACROS_H_
src/hydro/gpu_macros.h:#define GPU_MACROS_H_
src/hydro/gpu_macros.h:#ifdef __CUDACC__
src/hydro/gpu_macros.h:#endif // __CUDACC__
src/hydro/gpu_macros.h:#ifdef __CUDACC__
src/hydro/gpu_macros.h:#endif // __CUDACC__
src/hydro/gpu_macros.h: * define some sanity check routines for cuda runtime
src/hydro/gpu_macros.h:#ifdef __CUDACC__
src/hydro/gpu_macros.h:inline void checkCudaError(const char *msg)
src/hydro/gpu_macros.h:  cudaError_t e = cudaDeviceSynchronize();
src/hydro/gpu_macros.h:  if( e != cudaSuccess )
src/hydro/gpu_macros.h:      fprintf(stderr, "CUDA Error in %s : %s\n", msg, cudaGetErrorString(e));
src/hydro/gpu_macros.h:  e = cudaGetLastError();
src/hydro/gpu_macros.h:  if( e != cudaSuccess )
src/hydro/gpu_macros.h:      fprintf(stderr, "CUDA Error %s : %s\n", msg, cudaGetErrorString(e));
src/hydro/gpu_macros.h:} // checkCudaError
src/hydro/gpu_macros.h:inline void checkCudaErrorMpi(const char *msg, const int mpiRank)
src/hydro/gpu_macros.h:  cudaError_t e = cudaDeviceSynchronize();
src/hydro/gpu_macros.h:  if( e != cudaSuccess )
src/hydro/gpu_macros.h:      fprintf(stderr, "[Mpi rank %4d] CUDA Error in %s : %s\n", mpiRank, msg, cudaGetErrorString(e));
src/hydro/gpu_macros.h:  e = cudaGetLastError();
src/hydro/gpu_macros.h:  if( e != cudaSuccess )
src/hydro/gpu_macros.h:      fprintf(stderr, "[Mpi rank %4d] CUDA Error %s : %s\n", mpiRank, msg, cudaGetErrorString(e));
src/hydro/gpu_macros.h:} // checkCudaErrorMpi
src/hydro/gpu_macros.h:#endif // __CUDACC__
src/hydro/gpu_macros.h:#endif // GPU_MACROS_H_
src/hydro/godunov_unsplit_mhd_zslab.cuh: * \brief Defines the CUDA kernel for the actual MHD Godunov scheme, z-slab method.
src/hydro/godunov_unsplit_mhd_zslab.cuh: * Define some CUDA kernel to implement MHD version 3 on GPU
src/hydro/cutil_inline.h: * \brief Some utility routines from the CUDA SDK.
src/hydro/cutil_inline.h: * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
src/hydro/cutil_inline.h: * NVIDIA Corporation and its licensors retain all intellectual property and 
src/hydro/cutil_inline.h: * NVIDIA Corporation is strictly prohibited.
src/hydro/cutil_inline.h:#include <cuda.h>
src/hydro/cutil_inline.h:#include "cuda_runtime_api.h"
src/hydro/cutil_inline.h://     printf("CUDA %d.%02d Toolkit built this project.\n", CUDART_VERSION/1000, (CUDART_VERSION%100));
src/hydro/cutil_inline.h://     printf(" -> CUDA %s Toolkit\n"  , sNVCCReq);
src/hydro/cutil_inline.h://     printf(" -> %s NVIDIA Display Driver.\n", sDriverReq);
src/hydro/godunov_trace_v1.cuh: * \brief Defines the CUDA kernel for the actual Godunov scheme
src/hydro/riemann_mhd.h: * \brief Provides CPU/GPU riemann solver routines for MHD.
src/hydro/riemann_mhd.h: * copyToSymbolMemory to have those parameter in the GPU constant
src/hydro/riemann_mhd.h:  // which generate to much branch divergence in CUDA !!!
src/hydro/riemann_mhd.h:  // which generate to much branch divergence in CUDA !!!
src/hydro/cmpflx.h: * \brief Implements the CPU/GPU device routines to compute fluxes update
src/hydro/gl_util.h:// see file raycaster/axl/QCUDAImplicitRayCaster/include/QCUDAImplicitRayCaster/QCIRC_internal.hpp
src/test_euler2d_glut.sh:./euler2d_gpu_glut --posx 420 --posy 100&
src/test_euler2d_glut.sh:# recordmydesktop -x 200 -y 100 --width 430 --height 840 -o euler_cpu_gpu_screencast.ogv
src/Makefile.am:# also note that SUFFIXES must be set with .cu before including cuda.am
src/Makefile.am:include $(top_srcdir)/am/cuda.am
src/Makefile.am:AM_CXXFLAGS = -I$(srcdir)/hydro -I$(srcdir)/utils/config -I$(srcdir)/utils/monitoring $(CUDA_CFLAGS) $(VTK_CPPFLAGS) $(TIMING_FLAGS)
src/Makefile.am:AM_CFLAGS   = $(CUDA_CFLAGS) $(VTK_CPPFLAGS) $(TIMING_FLAGS)
src/Makefile.am:AM_CPPFLAGS = -I$(srcdir)/hydro -I$(srcdir)/utils/config -I$(srcdir)/utils/monitoring $(CUDA_CFLAGS) $(VTK_CPPFLAGS) $(TIMING_FLAGS)
src/Makefile.am:# - euler_gpu     (GPU serial   - NO MPI - CUDA)
src/Makefile.am:# - euler_gpu_mpi (GPU parallel -    MPI - CUDA)
src/Makefile.am:if USE_CUDA
src/Makefile.am:bin_PROGRAMS += euler_gpu \
src/Makefile.am:		euler2d_laxliu_gpu \
src/Makefile.am:		euler_zslab_gpu
src/Makefile.am:bin_PROGRAMS += euler_gpu_double \
src/Makefile.am:		euler_zslab_gpu_double
src/Makefile.am:bin_PROGRAMS += euler_gpu_mpi \
src/Makefile.am:		euler_zslab_gpu_mpi
src/Makefile.am:bin_PROGRAMS += euler_gpu_mpi_double \
src/Makefile.am:		euler_zslab_gpu_mpi_double
src/Makefile.am:# Euler 2d/3d on GPU (Godunov-Riemann or Kurganov-Tadmor)
src/Makefile.am:if USE_CUDA
src/Makefile.am:nodist_euler_gpu_SOURCES = euler_main.cu 
src/Makefile.am:nodist_EXTRA_euler_gpu_SOURCES = dummy.cpp
src/Makefile.am:euler_gpu_CPPFLAGS = $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
src/Makefile.am:euler_gpu_LDFLAGS  = $(CUDA_LIBS) hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la
src/Makefile.am:euler_gpu_LDADD = $(GM_LIBS) utils/cnpy/libCNpy.la
src/Makefile.am:nodist_euler_gpu_double_SOURCES = euler_main-double.cu 
src/Makefile.am:nodist_EXTRA_euler_gpu_double_SOURCES = dummy.cpp
src/Makefile.am:euler_gpu_double_CPPFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
src/Makefile.am:euler_gpu_double_CXXFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS) 
src/Makefile.am:euler_gpu_double_LDFLAGS  = $(CUDA_LIBS) hydro/libhydroGpu_double.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la
src/Makefile.am:euler_gpu_double_LDADD = $(GM_LIBS) utils/cnpy/libCNpy.la
src/Makefile.am:# Euler 2d/3d on GPU with MPI+CUDA
src/Makefile.am:nodist_euler_gpu_mpi_SOURCES = euler_mpi_main.cu
src/Makefile.am:nodist_EXTRA_euler_gpu_mpi_SOURCES = dummy.cpp
src/Makefile.am:euler_gpu_mpi_CPPFLAGS = $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils $(EXTRA_FLAGS)
src/Makefile.am:euler_gpu_mpi_CXXFLAGS = $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils $(EXTRA_FLAGS)
src/Makefile.am:euler_gpu_mpi_LDFLAGS  = $(CUDA_LIBS) $(GM_LIBS) hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la utils/mpiUtils/libMpiUtils.la $(MPI_LDFLAGS) 
src/Makefile.am:nodist_euler_gpu_mpi_double_SOURCES = euler_mpi_main-double.cu 
src/Makefile.am:nodist_EXTRA_euler_gpu_mpi_double_SOURCES = dummy.cpp
src/Makefile.am:euler_gpu_mpi_double_CXXFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils $(EXTRA_FLAGS)
src/Makefile.am:euler_gpu_mpi_double_CPPFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS)
src/Makefile.am:euler_gpu_mpi_double_LDFLAGS  = $(CUDA_LIBS)  $(GM_LIBS) hydro/libhydroGpu_double.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la utils/mpiUtils/libMpiUtils.la $(MPI_LDFLAGS)
src/Makefile.am:# Euler 2d on GPU (Lax-Liu)
src/Makefile.am:if USE_CUDA
src/Makefile.am:nodist_euler2d_laxliu_gpu_SOURCES = \
src/Makefile.am:nodist_EXTRA_euler2d_laxliu_gpu_SOURCES = dummy.cxx
src/Makefile.am:euler2d_laxliu_gpu_CPPFLAGS = $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
src/Makefile.am:euler2d_laxliu_gpu_LDFLAGS  = $(CUDA_LIBS) 
src/Makefile.am:euler2d_laxliu_gpu_LDADD    = hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la $(GM_LIBS)
src/Makefile.am:# Euler 3d on GPU (Godunov - Z-SLAB)
src/Makefile.am:if USE_CUDA
src/Makefile.am:nodist_euler_zslab_gpu_SOURCES = euler_zslab_main.cu
src/Makefile.am:nodist_EXTRA_euler_zslab_gpu_SOURCES = dummy.cpp
src/Makefile.am:euler_zslab_gpu_CPPFLAGS = $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
src/Makefile.am:euler_zslab_gpu_LDFLAGS  = $(CUDA_LIBS) hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la
src/Makefile.am:euler_zslab_gpu_LDADD = $(GM_LIBS) 
src/Makefile.am:nodist_euler_zslab_gpu_double_SOURCES = \
src/Makefile.am:nodist_EXTRA_euler_zslab_gpu_double_SOURCES = dummy.cpp
src/Makefile.am:euler_zslab_gpu_double_CPPFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
src/Makefile.am:euler_zslab_gpu_double_CXXFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS) 
src/Makefile.am:euler_zslab_gpu_double_LDFLAGS  = $(CUDA_LIBS) hydro/libhydroGpu_double.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la
src/Makefile.am:euler_zslab_gpu_double_LDADD = $(GM_LIBS)
src/Makefile.am:# Euler 3d on GPU with MPI+CUDA + Z-SLAB
src/Makefile.am:if USE_CUDA
src/Makefile.am:nodist_euler_zslab_gpu_mpi_SOURCES = euler_zslab_mpi_main.cu 
src/Makefile.am:nodist_EXTRA_euler_zslab_gpu_mpi_SOURCES = dummy.cpp
src/Makefile.am:euler_zslab_gpu_mpi_CPPFLAGS = $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils $(EXTRA_FLAGS)
src/Makefile.am:euler_zslab_gpu_mpi_CXXFLAGS = $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils $(EXTRA_FLAGS)
src/Makefile.am:euler_zslab_gpu_mpi_LDFLAGS  = $(CUDA_LIBS) $(GM_LIBS) hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la utils/mpiUtils/libMpiUtils.la $(MPI_LDFLAGS) 
src/Makefile.am:nodist_euler_zslab_gpu_mpi_double_SOURCES = euler_zslab_mpi_main-double.cu 
src/Makefile.am:nodist_EXTRA_euler_zslab_gpu_mpi_double_SOURCES = dummy.cpp
src/Makefile.am:euler_zslab_gpu_mpi_double_CXXFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils $(EXTRA_FLAGS)
src/Makefile.am:euler_zslab_gpu_mpi_double_CPPFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS)
src/Makefile.am:euler_zslab_gpu_mpi_double_LDFLAGS  = $(CUDA_LIBS)  $(GM_LIBS) hydro/libhydroGpu_double.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la utils/mpiUtils/libMpiUtils.la $(MPI_LDFLAGS)
src/Makefile.am:# Euler 2d on GPU (Godunov-Riemann or Kurganov-Tadmor) with GLUT GUI
src/Makefile.am:if USE_CUDA
src/Makefile.am:bin_PROGRAMS  += euler2d_gpu_glut
src/Makefile.am:euler2d_gpu_glut_SOURCES = \
src/Makefile.am:nodist_euler2d_gpu_glut_SOURCES = \
src/Makefile.am:nodist_EXTRA_euler2d_gpu_glut_SOURCES = dummy.cpp
src/Makefile.am:euler2d_gpu_glut_CXXFLAGS = $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
src/Makefile.am:euler2d_gpu_glut_LDFLAGS  = $(CUDA_LIBS) $(GLEW_LIBS) $(GLUT_LIBS) $(GL_LIBS) 
src/Makefile.am:euler2d_gpu_glut_LDADD    = hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la $(GM_LIBS) 
src/Makefile.am:# euler2d_gpu_qt
src/Makefile.am:# if USE_CUDA
src/Makefile.am:# bin_PROGRAMS += euler2d_gpu_qt
src/Makefile.am:# euler2d_gpu_qt_SOURCES = \
src/Makefile.am:# 	qtGui/qtHydro2d/HydroWidgetGpu.h \
src/Makefile.am:# 	qtGui/qtHydro2d/HydroWidgetGpu.cu \
src/Makefile.am:# nodist_euler2d_gpu_qt_SOURCES = qtGui/qtHydro2d/main.cu 
src/Makefile.am:# nodist_EXTRA_euler2d_gpu_qt_SOURCES = dummy.cpp
src/Makefile.am:# euler2d_gpu_qt_CXXFLAGS = $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
src/Makefile.am:# euler2d_gpu_qt_LDFLAGS  = $(CUDA_LIBS) $(GLEW_LIBS) hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la
src/Makefile.am:# euler2d_gpu_qt_LDADD    = $(GM_LIBS) $(QT_LIBS) 
src/Makefile.am:# gpu_moc_gen_sources = \
src/Makefile.am:# gpu_gen_sources = $(moc_gen_sources) 
src/Makefile.am:# nodist_euler2d_gpu_qt_SOURCES += $(gpu_gen_sources)
src/Makefile.am:if USE_CUDA
src/Makefile.am:bin_PROGRAMS += testRiemannHLLD_gpu
src/Makefile.am:	hydro/gpu_macros.cpp \
src/Makefile.am:	hydro/gpu_macros.cpp \
src/Makefile.am:	hydro/gpu_macros.cpp \
src/Makefile.am:	hydro/gpu_macros.cpp \
src/Makefile.am:# gpu sources for testRiemannHLLD_gpu
src/Makefile.am:if USE_CUDA
src/Makefile.am:nodist_testRiemannHLLD_gpu_SOURCES = \
src/Makefile.am:	hydro/gpu_macros.cu \
src/Makefile.am:nodist_EXTRA_testRiemannHLLD_gpu_SOURCES = dummy.cpp
src/Makefile.am:testRiemannHLLD_gpu_CPPFLAGS = $(CUDA_CFLAGS) $(EXTRA_FLAGS)
src/Makefile.am:testRiemannHLLD_gpu_LDADD = $(CUDA_LIBS) $(GM_LIBS) utils/config/libIniConfig.la utils/monitoring/libMonitoring.la
src/Makefile.am:bin_PROGRAMS += testRiemannHLLD_gpu_double
src/Makefile.am:nodist_testRiemannHLLD_gpu_double_SOURCES = \
src/Makefile.am:	hydro/gpu_macros.cu \
src/Makefile.am:nodist_EXTRA_testRiemannHLLD_gpu_double_SOURCES = dummy.cpp
src/Makefile.am:testRiemannHLLD_gpu_double_CPPFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) $(EXTRA_FLAGS)
src/Makefile.am:testRiemannHLLD_gpu_double_LDADD = $(CUDA_LIBS) $(GM_LIBS) utils/config/libIniConfig.la utils/monitoring/libMonitoring.la
src/utils/config/CMakeLists.txt:add_library(RamsesGPU::config ALIAS config)
src/utils/config/Makefile.am:include $(top_srcdir)/am/cuda.am
src/utils/config/Makefile.am:if USE_CUDA
src/utils/config/Makefile.am:noinst_LTLIBRARIES += libIniConfigGpu.la
src/utils/config/Makefile.am:if USE_CUDA
src/utils/config/Makefile.am:libIniConfigGpu_la_SOURCES = \
src/utils/config/Makefile.am:nodist_libIniConfigGpu_la_SOURCES = \
src/utils/config/Makefile.am:nodist_EXTRA_libIniConfigGpu_la_SOURCES = dummy.cpp
src/utils/config/Makefile.am:libIniConfigGpu_la_CPPFLAGS = $(AM_CPPFLAGS) $(CUDA_CFLAGS)
src/utils/config/Makefile.am:if USE_CUDA
src/utils/config/Makefile.am:bin_PROGRAMS += ConfigMapTestGpu
src/utils/config/Makefile.am:if USE_CUDA
src/utils/config/Makefile.am:nodist_ConfigMapTestGpu_SOURCES = ConfigMapTest.cu
src/utils/config/Makefile.am:nodist_EXTRA_ConfigMapTestGpu_SOURCES = dummy.cxx
src/utils/config/Makefile.am:ConfigMapTestGpu_CPPFLAGS = $(CUDA_CFLAGS) 
src/utils/config/Makefile.am:ConfigMapTestGpu_LDFLAGS  = $(CUDA_LIBS) libIniConfigGpu.la
src/utils/config/Makefile.am:ConfigMapTestGpu_DEPENDENCIES = libIniConfigGpu.la
src/utils/monitoring/CMakeLists.txt:if (RAMSESGPU_PAPI_FOUND)
src/utils/monitoring/CMakeLists.txt:    RamsesGPU::papi)
src/utils/monitoring/CMakeLists.txt:endif(RAMSESGPU_PAPI_FOUND)
src/utils/monitoring/CMakeLists.txt:add_library(RamsesGPU::monitoring ALIAS monitoringCpu)
src/utils/monitoring/CMakeLists.txt:if (RAMSESGPU_PAPI_FOUND)
src/utils/monitoring/CMakeLists.txt:  target_link_libraries(PapiInfoTest RamsesGPU::monitoring)
src/utils/monitoring/Makefile.am:include $(top_srcdir)/am/cuda.am
src/utils/monitoring/Makefile.am:if USE_CUDA
src/utils/monitoring/Makefile.am:noinst_LTLIBRARIES += libMonitoringGpu.la
src/utils/monitoring/Makefile.am:	CudaTimer.h \
src/utils/monitoring/Makefile.am:if USE_CUDA
src/utils/monitoring/Makefile.am:libMonitoringGpu_la_SOURCES = Timer.h 
src/utils/monitoring/Makefile.am:nodist_libMonitoringGpu_la_SOURCES = Timer.cu
src/utils/monitoring/Makefile.am:nodist_EXTRA_libMonitoringGpu_la_SOURCES = dummy.cpp
src/utils/monitoring/Makefile.am:libMonitoringGpu_la_CPPFLAGS = $(AM_CPPFLAGS) $(CUDA_CFLAGS)
src/utils/monitoring/CudaTimer.h: * \file CudaTimer.h
src/utils/monitoring/CudaTimer.h: * \brief A simple timer class for CUDA based on events.
src/utils/monitoring/CudaTimer.h: * $Id: CudaTimer.h 1783 2012-02-21 10:20:07Z pkestene $
src/utils/monitoring/CudaTimer.h:#ifndef CUDA_TIMER_H_
src/utils/monitoring/CudaTimer.h:#define CUDA_TIMER_H_
src/utils/monitoring/CudaTimer.h:   * \brief a simple timer for CUDA kernel.
src/utils/monitoring/CudaTimer.h:  class CudaTimer
src/utils/monitoring/CudaTimer.h:    cudaEvent_t startEv, stopEv;
src/utils/monitoring/CudaTimer.h:    CudaTimer() {
src/utils/monitoring/CudaTimer.h:      cudaEventCreate(&startEv);
src/utils/monitoring/CudaTimer.h:      cudaEventCreate(&stopEv);
src/utils/monitoring/CudaTimer.h:    ~CudaTimer() {
src/utils/monitoring/CudaTimer.h:      cudaEventDestroy(startEv);
src/utils/monitoring/CudaTimer.h:      cudaEventDestroy(stopEv);
src/utils/monitoring/CudaTimer.h:      cudaEventRecord(startEv, 0);
src/utils/monitoring/CudaTimer.h:      float gpuTime;
src/utils/monitoring/CudaTimer.h:      cudaEventRecord(stopEv, 0);
src/utils/monitoring/CudaTimer.h:      cudaEventSynchronize(stopEv);
src/utils/monitoring/CudaTimer.h:      cudaEventElapsedTime(&gpuTime, startEv, stopEv);
src/utils/monitoring/CudaTimer.h:      total_time += (double)1e-3*gpuTime;
src/utils/monitoring/CudaTimer.h:  }; // class CudaTimer
src/utils/monitoring/CudaTimer.h:#endif // CUDA_TIMER_H_
src/utils/cnpy/CMakeLists.txt:add_library(RamsesGPU::cnpy ALIAS cnpy)
src/utils/cnpy/CMakeLists.txt:  RamsesGPU::cnpy)
src/utils/mpiUtils/MpiComm.h: * be a bug; see http://forums.nvidia.com/index.php?showtopic=163810
src/utils/mpiUtils/CMakeLists.txt:add_library(RamsesGPU::mpiUtils ALIAS mpiUtils)

```
