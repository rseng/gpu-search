# https://github.com/nlesc-dirac/sagecal

```console
Docs/source/tutorial.rst:We will demonstrate selfcal using the SAGECal executable for a GPU - sagecal_gpu - built with cmake, but instructions are, of course, similar for the containerized version of sagecal_gpu. Building sagecal will also automatically build buildsky and create_clusters.py, which we need for self-calibration.
Docs/source/tutorial.rst:This will produce a cluster file skyview-image.fits.sky.txt.cluster with just one cluster, which will not be subtracted, because it will get a negative cluster id (-1). The two separate sources shown in the SkyView/TGSS image of 3C196 are separated by 3-4', so much less than the size of the isoplanatic patch at our observing frequency (153 MHz). A maximum of 10 iterations was set, but 2 were enough. Now calibrate our data on this sky model, optionally making use of GPU power.
Docs/source/tutorial.rst:   module load openblas cuda91 casacore/2.3.0-gcc-4.9.3 (or a similar instruction, if necessary)
Docs/source/tutorial.rst:   ../../install/bin/sagecal_gpu -d sm.ms -s skyview-image.fits.sky.txt -c skyview-image.fits.sky.txt.cluster -n 40 -t 1 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 2 -k -1 -B 1 -E 1  > sm.ms.output
Docs/source/tutorial.rst:   ../../install/bin/sagecal_gpu -h
Docs/source/tutorial.rst:   ../../install/bin/sagecal_gpu -d sm.ms -s after-initial-calibration-image.fits.sky.txt -c after-initial-calibration-image.fits.sky.txt.cluster -n 40 -t 1 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 2 -k 1 -B 1 -E 1  > sm.ms.output
Docs/source/tutorial.rst:   ../../install/bin/sagecal_gpu -d sm.ms -s after-initial-calibration-image.fits.sky.txt -c after-initial-calibration-image.fits.sky.txt.cluster -n 40 -t 1 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 2 -k -2 -B 1 -E 1  > sm.ms.output
Docs/source/tutorial.rst:   ../../install/bin/sagecal_gpu -d sm.ms -s after-initial-calibration-image.fits.sky.txt -c after-initial-calibration-image.fits.sky.txt.cluster -n 40 -t 1 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 2 -k -3 -B 1 -E 1  > sm.ms.output
Docs/source/tutorial.rst:   ../../install/bin/sagecal_gpu -d sm.ms -s after-initial-calibration-image.fits.sky.txt -c after-initial-calibration-image.fits.sky.txt.cluster -n 40 -t 1 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 2 -k -4 -B 1 -E 1  > sm.ms.output
Docs/source/tutorial.rst:  ../../install/bin/sagecal_gpu -d sm.ms -s after-initial-calibration-image.fits.sky.txt -c after-initial-calibration-image.fits.sky.txt.cluster -n 40 -t 1 -p sm.ms.solutions -a 0 -e 4 -F 1 -j 2 -B 1 -E 1  > sm.ms.output
Docs/source/install.rst:GPU Support
Docs/source/install.rst:Compiling with GPU support
Docs/source/install.rst:   cmake -DCUDA_DEBUG=ON -DDEBUG=ON -DVERBOSE=ON -DHAVE_CUDA=ON ..
Docs/source/install.rst:For expert users, and for custom architectures (GPU), the manual install
Docs/source/install.rst:   -  If you have NVIDIA GPUs: CUDA/CUBLAS/CUSOLVER, nvcc and
Docs/source/install.rst:      NVML (Nvidia management library) 
Docs/source/install.rst:- CUDAINC/CUDALIB : where CUDA/CUBLAS/CUSOLVER is installed 
Docs/source/install.rst:- NVCFLAGS : flags to pass to nvcc, especially -arch option to match your GPU 
Docs/source/install.rst:- Makefile.gpu: with GPU support. 
Docs/source/install.rst:  Note: Edit ./lib/Radio/Radio.h MAX_GPU_ID to match the number of
Docs/source/install.rst:  available GPUs, e.g., for 2 GPUs, MAX_GPU_ID=1
Docs/source/user_manual.rst:|     sagecal here denotes the sagecal executable, compiled either for CPU or GPU.
Docs/source/user_manual.rst:- **-E 0,1**. If 1, use GPU for model computing, i.e. for converting a sky model to its corresponding visibilities at the (u, v, w) triples of the observation. Default: 0.
Docs/source/user_manual.rst:- **-S GPU heap size (MB)**. Default: 32.
ChangeLog:    Now almost the same CXX flags as Makefile and Makefile.gpu: CXXFLAGS=-O3 -Wall -g -std=c++11
ChangeLog:    missing HAVE_CUDA added
ChangeLog:    added function to predict using stored solutions, with GPU accel
ChangeLog:    -pg not appropriate, NVML_INC is redundant and link to the latest cuda, i.e. cuda 9.1
ChangeLog:    CUDAINC and NVML_INC are redundant
ChangeLog:    '/usr/include/nvidia/gdk/' does not exist, at least not on fs5 and NVML_INC is redundant
ChangeLog:    '/usr/local/cuda/include' does not exist, at least not on fs5 and CUDAINC is redundant
ChangeLog:    NVML_INC is redundant and /usr/include/nvidia/gdk/ does not exist, at least not on fs5.
ChangeLog:    CUDAINC is redundant and '/usr/local/cuda/include' does not exist, at least not on fs5. 'module load cuda91' or some similar adjustment to PATH should suffice.
ChangeLog:    '-L/usr/lib64/nvidia/' in 'NVML_LIB=-lnvidia-ml -L/usr/lib64/nvidia/' since that directory does not exist, at least not on fs5
ChangeLog:    CUDAINC is redundant when cuda in PATH, e.g. after 'module load cuda91'
ChangeLog:    Adjusted Makefile.gpu for latest casacore on fs5: casacore/v2.4.1-gcc-6.3.0
ChangeLog:    This fixes 'cannot find -lnvidia-ml' together with the '-lcuda -lcudart' from the previous commit
ChangeLog:    added -lcuda -lcudart
ChangeLog:    info about using more than 1 GPU
ChangeLog:    fixed compile issues for both CPU/GPU versions
ChangeLog:    Seems that the Makefile.gpu for Radio was erroneously used in Dirac
ChangeLog:    We don't need CUDAINC, so I removed it
ChangeLog:    added correction of residual data using GPU
ChangeLog:    added model prediction and residual calculation using GPU
ChangeLog:    added option to use GPU for model prediction
ChangeLog:    added option to use GPU for simulations
ChangeLog:    Input to and output from cudakernel_array_beam written to files
ChangeLog:    remove obsolete ONE_GPU
test/Dirac/README.md:  * `demo_stochastic_cuda.c`: minibatch LBFGS with full GPU acceleration
test/Dirac/README.md:Use `Makefile` to build `demo` and `demo_stochastic`. Use `Makefile.cuda` to build `demo_stochastic_cuda`.
test/Dirac/README.md:Note that for the CUDA demo, libdirac should be built with CUDA support, by using `-DHAVE_CUDA=ON` cmake option.
test/Dirac/demo_stochastic_cuda.c:   Note: The LBFGS routine is fully GPU accelerated,
test/Dirac/demo_stochastic_cuda.c:   so -DHAVE_CUDA=ON should be used when compiling libdirac
test/Dirac/demo_stochastic_cuda.c:#define HAVE_CUDA
test/Dirac/demo_stochastic_cuda.c:#define CUDA_DEBUG
test/Dirac/demo_stochastic_cuda.c:checkCudaError(cudaError_t err, char *file, int line)
test/Dirac/demo_stochastic_cuda.c:#ifdef CUDA_DEBUG
test/Dirac/demo_stochastic_cuda.c:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
test/Dirac/demo_stochastic_cuda.c: cudaError_t err;
test/Dirac/demo_stochastic_cuda.c: err=cudaHostAlloc((void **)&phost, sizeof(double)*m,cudaHostAllocDefault);
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c: err=cudaMemcpy(phost, p, sizeof(double)*m, cudaMemcpyDeviceToHost);
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c: err=cudaFreeHost(phost);
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c: cudaError_t err;
test/Dirac/demo_stochastic_cuda.c: err=cudaHostAlloc((void **)&phost, sizeof(double)*m,cudaHostAllocDefault);
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c: err=cudaMemcpy(phost, p, sizeof(double)*m, cudaMemcpyDeviceToHost);
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c: err=cudaHostAlloc((void **)&ghost, sizeof(double)*m,cudaHostAllocDefault);
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c: err=cudaMemset(ghost,0,m*sizeof(double));
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c: err=cudaMemcpy(g, ghost, sizeof(double)*m, cudaMemcpyHostToDevice);
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c: err=cudaFreeHost(phost);
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c: err=cudaFreeHost(ghost);
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c: /* setup a GPU to use */
test/Dirac/demo_stochastic_cuda.c: cudaError_t err;
test/Dirac/demo_stochastic_cuda.c: attach_gpu_to_thread(0, &cbhandle, &solver_handle);
test/Dirac/demo_stochastic_cuda.c: err=cudaMalloc((void**)&(pdevice),m*sizeof(double));
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c: err=cudaMemcpy(pdevice, p, m*sizeof(double), cudaMemcpyHostToDevice);
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c:   lbfgs_fit_cuda(rosenbrok,rosenbrok_grad,pdevice,m,50,M,&rt,&ptdata);
test/Dirac/demo_stochastic_cuda.c: err=cudaMemcpy(p, pdevice, m*sizeof(double), cudaMemcpyDeviceToHost);
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c: err=cudaFree(pdevice);
test/Dirac/demo_stochastic_cuda.c: checkCudaError(err,__FILE__,__LINE__);
test/Dirac/demo_stochastic_cuda.c: detach_gpu_from_thread(cbhandle,solver_handle);
test/Dirac/Makefile.cuda:### This makefile shows how to use the GPU accelerated libdirac 
test/Dirac/Makefile.cuda:# CUDA flags
test/Dirac/Makefile.cuda:CUDAINC=-I/usr/local/cuda/include
test/Dirac/Makefile.cuda:CUDALIB=-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcusolver
test/Dirac/Makefile.cuda:OBJECTSSTOCHASTIC=demo_stochastic_cuda.o
test/Dirac/Makefile.cuda:default:demo_stochastic_cuda
test/Dirac/Makefile.cuda:demo_stochastic_cuda.o:demo_stochastic_cuda.c
test/Dirac/Makefile.cuda:	$(CC) $(CFLAGS) $(INCLUDES) $(GLIBI) $(CUDAINC) -c $<
test/Dirac/Makefile.cuda:demo_stochastic_cuda: $(OBJECTSSTOCHASTIC)
test/Dirac/Makefile.cuda:	$(CC)  $(CFLAGS) $(INCLUDES) $(GLIBI)  -o $@ $(OBJECTSSTOCHASTIC) $(LDIRAC) $(CUDALIB) $(CLIBS) $(LIBPATH) $(LAPACK)
test/Calibration/dosage.sh:# or sagecal_gpu if GPU acceleration is enabled
test/Calibration/dosage-mpi.sh:# or sagecal-mpi_gpu if GPU acceleration is enabled
CITATION.cff:  - "GPU"
README.md:- GPU acceleration using CUDA
README.md:Optionally: Make sure your machine has (1/2 working NVIDIA GPU cards or Intel Xeon Phi MICs) to use sagecal.
README.md:Recommended usage: (with GPUs)
README.md:sagecal_gpu -d my_data.MS -s my_skymodel -c my_clustering -n no.of.threads -t 60 -p my_solutions -e 3 -g 2 -l 10 -m 7 -w 1 -b 1
README.md:Replace ```sagecal_gpu``` with ```sagecal``` if you have a CPU only build.
README.md:Note: To fully use GPU acceleration use ```-E 1``` option.
README.md:Use mpirun to run sagecal-mpi, (or ```sagecal-mpi_gpu``` for GPU version) for example:
INSTALL.md:Run cmake (with GPU support) for example like
INSTALL.md: cmake .. -DHAVE_CUDA=ON -DNUM_GPU=1 -DCMAKE_CXX_COMPILER=g++-8 -DCMAKE_C_COMPILER=gcc-8 -DCUDA_NVCC_FLAGS='-gencode arch=compute_75,code=sm_75' -DCMAKE_CUDA_ARCHITECTURES=75 -DBLA_VENDOR=OpenBLAS
INSTALL.md:where *-DNUM_GPU=1* is when there is only one GPU. If you have more GPUs, increase this number to 2,3, and so on. This will produce *sagecal_gpu* and *sagecal-mpi_gpu* binary files (after running *make* of course). Architecture of the GPU is specified in the *-DCUDA_NVCC_FLAGS* option, and in newer cmake, using *-DCMAKE_CUDA_ARCHITECTURES*. It is important to select the gcc and g++ compilers to match the CUDA version, above example uses *gcc-8* and *g++-8*.
INSTALL.md:To only build *libdirac* (shared) library, use *-DLIB_ONLY=1* option (also *-DBLA_VENDOR* to select the BLAS flavour). This library can be used with pkg-config using *lib/pkgconfig/libdirac.pc*. To build *libdirac* with GPU support, use *-DHAVE_CUDA=ON* with *-DLIB_ONLY=1* and give *-fPIC* compiler flag (for both *-DCMAKE_CXX_FLAGS* and *-DCMAKE_C_FLAGS*). With GPU support, only a static library is built because it needs to match the GPU architecture.
INSTALL.md:SAGECal can use ***libmvec*** vectorized math operations, both in GPU and CPU versions. In order to enable this, use compiler options *-ffast-math -lmvec -lm* for both gcc and g++. Also *-mavx*, *-mavx2* etc. can be added. Here is an example for CPU version
CMakeLists.txt:option (HAVE_CUDA "Enable CUDA support" OFF)
CMakeLists.txt:option (CUDA_DEBUG "Enable Debug mode for CUDA" OFF)
CMakeLists.txt:option (CUDA_MODEL_MAX_F "Max number of channels in shared mem" OFF)
CMakeLists.txt:option (NUM_GPU "Number of GPUs" OFF)
CMakeLists.txt:# cuda
CMakeLists.txt:if(HAVE_CUDA)
CMakeLists.txt:    message (STATUS "Compiling SageCal with GPU support.")
CMakeLists.txt:    enable_language(CUDA)
CMakeLists.txt:    find_package(CUDA QUIET REQUIRED)
CMakeLists.txt:if(HAVE_CUDA)
CMakeLists.txt:  # check if -DCUDA_NVCC_FLAGS is defined by user
CMakeLists.txt:  if(NOT CUDA_NVCC_FLAGS)
CMakeLists.txt:    set(CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 --ptxas-options=-v")
CMakeLists.txt:  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DHAVE_CUDA")
CMakeLists.txt:  add_definitions(-DHAVE_CUDA)
CMakeLists.txt:  if(CUDA_DEBUG)
CMakeLists.txt:    add_definitions(-DCUDA_DEBUG)
CMakeLists.txt:  if(NUM_GPU)
CMakeLists.txt:    math(EXPR MAX_GPU "${NUM_GPU} - 1")
CMakeLists.txt:    message (STATUS "Setting number of GPUs to ${NUM_GPU}")
CMakeLists.txt:    add_definitions(-DMAX_GPU_ID=${MAX_GPU})
CMakeLists.txt:  if(CUDA_MODEL_MAX_F)
CMakeLists.txt:    message (STATUS "Setting shared mem for max channels to ${CUDA_MODEL_MAX_F}")
CMakeLists.txt:    add_definitions(-DMODEL_MAX_F=${CUDA_MODEL_MAX_F})
CMakeLists.txt:message (STATUS "HAVE_CUDA ................. = ${HAVE_CUDA}")
CMakeLists.txt:if(HAVE_CUDA)
CMakeLists.txt:  message (STATUS "NUM_GPU ................... = ${NUM_GPU}")
CMakeLists.txt:  message (STATUS "CUDA_VERSION .............. = ${CUDA_VERSION}")
CMakeLists.txt:  message (STATUS "CMAKE_CUDA_COMPILER ....... = ${CMAKE_CUDA_COMPILER}")
CMakeLists.txt:  message (STATUS "CMAKE_CUDA_FLAGS .......... = ${CMAKE_CUDA_FLAGS}")
CMakeLists.txt:  message (STATUS "CMAKE_CUDA_FLAGS_DEBUG .... = ${CMAKE_CUDA_FLAGS_DEBUG}")
CMakeLists.txt:  message (STATUS "CMAKE_CUDA_FLAGS_RELEASE .. = ${CMAKE_CUDA_FLAGS_RELEASE}")
CMakeLists.txt:  message (STATUS "CMAKE_CUDA_HOST_COMPILER .. = ${CMAKE_CUDA_HOST_COMPILER}")
CMakeLists.txt:  message (STATUS "CUDACXX ................... = ${CUDACXX}")
CMakeLists.txt:  message (STATUS "CUDAHOSTCXX ............... = ${CUDAHOSTCXX}")
CMakeLists.txt:  message (STATUS "CUDA_TOOLKIT_ROOT_DIR ..... = ${CUDA_TOOLKIT_ROOT_DIR}")
CMakeLists.txt:  message (STATUS "CUDA_INCLUDE_DIRS ......... = ${CUDA_INCLUDE_DIRS}")
CMakeLists.txt:  message (STATUS "CUDA_LIBRARIES ............ = ${CUDA_LIBRARIES}")
CMakeLists.txt:  message (STATUS "CUDA_CUBLAS_LIBRARIES ..... = ${CUDA_CUBLAS_LIBRARIES}")
CMakeLists.txt:  message (STATUS "CUDA_NVCC_FLAGS ........... = ${CUDA_NVCC_FLAGS}")
CMakeLists.txt:  message (STATUS "CUDA_DEBUG ................ = ${CUDA_DEBUG}")
Docker/ubuntu2004-cuda/Dockerfile:    nvidia-cuda-dev nvidia-cuda-toolkit \
Docker/ubuntu2004-cuda/Dockerfile:    libcublas10 libcusolver10 libnvidia-ml-dev
Docker/ubuntu2004-cuda/Dockerfile:# 1) 'arch=compute_75,code=sm_75' to match your GPU
Docker/ubuntu2004-cuda/Dockerfile:# 2) -DNUM_GPU to match the number of GPUs to use (1 for one GPU, 2 for two etc.)
Docker/ubuntu2004-cuda/Dockerfile:# The gcc/g++ compiler versions are to match cuda 10.1
Docker/ubuntu2004-cuda/Dockerfile:     -DHAVE_CUDA=ON -DCUDA_NVCC_FLAGS="-gencode arch=compute_75,code=sm_75 -O3" \
Docker/ubuntu2004-cuda/Dockerfile:     -DNUM_GPU=2 -DBLA_VENDOR=OpenBLAS \
Docker/ubuntu2004-cuda/Dockerfile:     /opt/sagecal/bin/sagecal_gpu
CMakeModules/FindNVML.cmake:if(${CUDA_VERSION_STRING} VERSION_LESS "9.1")
CMakeModules/FindNVML.cmake:    string(CONCAT ERROR_MSG "--> ARCHER: Current CUDA version "
CMakeModules/FindNVML.cmake:                         ${CUDA_VERSION_STRING}
CMakeModules/FindNVML.cmake:    set(NVML_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
CMakeModules/FindNVML.cmake:    set(NVML_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
CMakeModules/FindNVML.cmake:              PATHS "C:/Program Files/NVIDIA Corporation/NVSMI")
CMakeModules/FindNVML.cmake:    set(NVML_NAMES nvidia-ml)
CMakeModules/FindNVML.cmake:    set(NVML_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x86_64-linux-gnu")
CMakeModules/FindNVML.cmake:    set(NVML_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
src/buildsky/buildsky.h:/****************************** clmfit_nocuda.c ****************************/
src/buildsky/buildsky.h:clevmar_der_single_nocuda(
src/buildsky/clmfit_nocuda.c:clevmar_der_single_nocuda(
src/buildsky/clmfit_nocuda.c:     //err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(double),cudaMemcpyDeviceToHost);
src/buildsky/clmfit_nocuda.c:      //err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(double),cudaMemcpyDeviceToHost);
src/buildsky/clmfit_nocuda.c:      //cudakernel_diagmu(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);
src/buildsky/clmfit_nocuda.c:        //err=cudaMemcpy(hxd, hx, N*sizeof(double), cudaMemcpyHostToDevice);
src/buildsky/clmfit_nocuda.c:clevmar_der_single_nocuda0(
src/buildsky/scluster.c: clevmar_der_single_nocuda(mylm_fit_single_pfmult, NULL, p, x, m, n, maxiter, opts, info, 2, (void*)&lmdata);
src/buildsky/scluster.c: clevmar_der_single_nocuda(mylm_fit_single_sipfmult, NULL, p, x, m, n, maxiter, opts, info, 2, (void*)&lmdata);
src/buildsky/fitpixels.c: clevmar_der_single_nocuda(mylm_fit_single, NULL, p, x, 3, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
src/buildsky/fitpixels.c:    //clevmar_der_single_nocuda(mylm_fit_single, NULL, &p[3*cj], xdummy, 3, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
src/buildsky/fitpixels.c: clevmar_der_single_nocuda(mylm_fit_N, NULL, p, x, m, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
src/buildsky/fitmultipixels.c: clevmar_der_single_nocuda(mylm_fit_single_sipf, NULL, p, x, m, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
src/buildsky/fitmultipixels.c: clevmar_der_single_nocuda(mylm_fit_single_sipf_2d, NULL, p2, x, m-1, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
src/buildsky/fitmultipixels.c: clevmar_der_single_nocuda(mylm_fit_single_sipf_1d, NULL, p1, x, m-2, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
src/buildsky/fitmultipixels.c:    clevmar_der_single_nocuda(mylm_fit_single_pf, NULL, p, xdummy, m, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
src/buildsky/fitmultipixels.c:    clevmar_der_single_nocuda(mylm_fit_single_pf_2d, NULL, p2, xdummy, m-1, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
src/buildsky/fitmultipixels.c:    clevmar_der_single_nocuda(mylm_fit_single_pf_1d, NULL, p1, xdummy, m-2, n, maxiter, opts, info, 2, (void*)&lmdata);  // no Jacobian
src/MS/main.cpp:#ifdef HAVE_CUDA
src/MS/main.cpp:   cout << "-E 0,1 : if 1, use GPU for model computing: default " <<Data::GPUpredict<< endl;
src/MS/main.cpp:#ifdef HAVE_CUDA
src/MS/main.cpp:   cout << "-S GPU heap size (MB): default "<<Data::heapsize<< endl;
src/MS/main.cpp:                GPUpredict=atoi(optarg);
src/MS/main.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_consensus_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_consensus_mode.cpp:   /* setup Heap of GPU,  only need to be done once, before any kernel is launched  */
src/MS/minibatch_consensus_mode.cpp:    if (GPUpredict>0) {
src/MS/minibatch_consensus_mode.cpp:     for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
src/MS/minibatch_consensus_mode.cpp:        cudaSetDevice(gpuid);
src/MS/minibatch_consensus_mode.cpp:        cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
src/MS/minibatch_consensus_mode.cpp:    /* for attaching to a GPU */
src/MS/minibatch_consensus_mode.cpp:    attach_gpu_to_thread(select_work_gpu(MAX_GPU_ID,&thst), &cbhandle, &solver_handle);
src/MS/minibatch_consensus_mode.cpp:    /* auxilliary arrays for GPU */
src/MS/minibatch_consensus_mode.cpp:        lbfgs_persist_init(&ptdata_array[ii],minibatches,iodata.N*8*Mt,iodata.Nbase*iodata.tilesz,Data::lbfgs_m,Data::gpu_threads);
src/MS/minibatch_consensus_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_consensus_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_consensus_mode.cpp:#ifndef HAVE_CUDA
src/MS/minibatch_consensus_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_consensus_mode.cpp:   if (GPUpredict) {
src/MS/minibatch_consensus_mode.cpp:     precalculate_coherencies_multifreq_withbeam_gpu(iodata.u,iodata.v,iodata.w,coh,iodata.N,iodata.Nbase*iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,deltafch,iodata.deltat,iodata.dec0,Data::min_uvcut,Data::max_uvcut,
src/MS/minibatch_consensus_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_consensus_mode.cpp:        bfgsfit_minibatch_consensus(iodata.u,iodata.v,iodata.w,&iodata.xo[iodata.Nbase*iodata.tilesz*8*chanstart[ii]],iodata.N,iodata.Nbase,iodata.tilesz,hbb,ptoclus,&coh[M*iodata.Nbase*iodata.tilesz*4*chanstart[ii]],M,Mt,&iodata.freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],&pfreq[iodata.N*8*Mt*ii],&Y[iodata.N*8*Mt*ii],z,&rhok[ii*Mt],Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00,&res_01,&ptdata_array[ii],nmb,minibatches);
src/MS/minibatch_consensus_mode.cpp:        bfgsfit_minibatch_consensus(iodata.u,iodata.v,iodata.w,&iodata.xo[iodata.Nbase*iodata.tilesz*8*chanstart[ii]],iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,&coh[M*iodata.Nbase*iodata.tilesz*4*chanstart[ii]],M,Mt,&iodata.freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],&pfreq[iodata.N*8*Mt*ii],&Y[iodata.N*8*Mt*ii],z,&rhok[ii*Mt],Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00,&res_01,&ptdata_array[ii],nmb,minibatches);
src/MS/minibatch_consensus_mode.cpp:#ifndef HAVE_CUDA
src/MS/minibatch_consensus_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_consensus_mode.cpp:    if (GPUpredict) {
src/MS/minibatch_consensus_mode.cpp:      calculate_residuals_multifreq_withbeam_gpu(iodata.u,iodata.v,iodata.w,&pfreq[iodata.N*8*Mt*ii],&iodata.xo[iodata.Nbase*iodata.tilesz*8*chanstart[ii]],iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,&iodata.freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],iodata.deltat,iodata.dec0,
src/MS/minibatch_consensus_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_consensus_mode.cpp:   /* if -E uses a large value ~say 100, at each multiple of this, clear GPU memory */
src/MS/minibatch_consensus_mode.cpp:   if (GPUpredict>1 && tilex>0 && !(tilex%GPUpredict)) {
src/MS/minibatch_consensus_mode.cpp:    for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
src/MS/minibatch_consensus_mode.cpp:       cudaSetDevice(gpuid);
src/MS/minibatch_consensus_mode.cpp:       cudaDeviceReset();
src/MS/minibatch_consensus_mode.cpp:       cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
src/MS/minibatch_consensus_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_consensus_mode.cpp:   detach_gpu_from_thread(cbhandle,solver_handle);
src/MS/minibatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_mode.cpp:   /* setup Heap of GPU,  only need to be done once, before any kernel is launched  */
src/MS/minibatch_mode.cpp:    if (GPUpredict>0) {
src/MS/minibatch_mode.cpp:     for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
src/MS/minibatch_mode.cpp:        cudaSetDevice(gpuid);
src/MS/minibatch_mode.cpp:        cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
src/MS/minibatch_mode.cpp:    /* for attaching to a GPU */
src/MS/minibatch_mode.cpp:    attach_gpu_to_thread(select_work_gpu(MAX_GPU_ID,&thst), &cbhandle, &solver_handle);
src/MS/minibatch_mode.cpp:    /* auxilliary arrays for GPU */
src/MS/minibatch_mode.cpp:        lbfgs_persist_init(&ptdata_array[ii],minibatches,iodata.N*8*Mt,iodata.Nbase*iodata.tilesz,Data::lbfgs_m,Data::gpu_threads);
src/MS/minibatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_mode.cpp:#ifndef HAVE_CUDA
src/MS/minibatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_mode.cpp:   if (GPUpredict) {
src/MS/minibatch_mode.cpp:     precalculate_coherencies_multifreq_withbeam_gpu(iodata.u,iodata.v,iodata.w,coh,iodata.N,iodata.Nbase*iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,deltafch,iodata.deltat,iodata.dec0,Data::min_uvcut,Data::max_uvcut,
src/MS/minibatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_mode.cpp:       bfgsfit_minibatch_visibilities(iodata.u,iodata.v,iodata.w,&iodata.xo[iodata.Nbase*iodata.tilesz*8*chanstart[ii]],iodata.N,iodata.Nbase,iodata.tilesz,hbb,ptoclus,&coh[M*iodata.Nbase*iodata.tilesz*4*chanstart[ii]],M,Mt,&iodata.freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],&pfreq[iodata.N*8*Mt*ii],Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00,&res_01,&ptdata_array[ii],nmb,minibatches);
src/MS/minibatch_mode.cpp:        bfgsfit_minibatch_visibilities(iodata.u,iodata.v,iodata.w,&iodata.xo[iodata.Nbase*iodata.tilesz*8*chanstart[ii]],iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,&coh[M*iodata.Nbase*iodata.tilesz*4*chanstart[ii]],M,Mt,&iodata.freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],&pfreq[iodata.N*8*Mt*ii],Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00,&res_01,&ptdata_array[ii],nmb,minibatches);
src/MS/minibatch_mode.cpp:#ifndef HAVE_CUDA
src/MS/minibatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_mode.cpp:    if (GPUpredict) {
src/MS/minibatch_mode.cpp:      calculate_residuals_multifreq_withbeam_gpu(iodata.u,iodata.v,iodata.w,&pfreq[iodata.N*8*Mt*ii],&iodata.xo[iodata.Nbase*iodata.tilesz*8*chanstart[ii]],iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,&iodata.freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],iodata.deltat,iodata.dec0,
src/MS/minibatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_mode.cpp:   /* if -E uses a large value ~say 100, at each multiple of this, clear GPU memory */
src/MS/minibatch_mode.cpp:   if (GPUpredict>1 && tilex>0 && !(tilex%GPUpredict)) {
src/MS/minibatch_mode.cpp:    for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
src/MS/minibatch_mode.cpp:       cudaSetDevice(gpuid);
src/MS/minibatch_mode.cpp:       cudaDeviceReset();
src/MS/minibatch_mode.cpp:       cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
src/MS/minibatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/minibatch_mode.cpp:   detach_gpu_from_thread(cbhandle,solver_handle);
src/MS/CMakeLists.txt:if(HAVE_CUDA)
src/MS/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
src/MS/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CASACORE_INCLUDE_DIR})
src/MS/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CASACORE_INCLUDE_DIR}/casacore)
src/MS/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Dirac)
src/MS/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio)
src/MS/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio/reserve)
src/MS/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
src/MS/CMakeLists.txt:if(HAVE_CUDA)
src/MS/CMakeLists.txt:    cuda_add_executable(sagecal_gpu ${SRCFILES})
src/MS/CMakeLists.txt:    add_dependencies(sagecal_gpu dirac-radio dirac)
src/MS/CMakeLists.txt:    target_link_libraries(sagecal_gpu
src/MS/CMakeLists.txt:        ${CUDA_CUBLAS_LIBRARIES}
src/MS/CMakeLists.txt:        ${CUDA_CUFFT_LIBRARIES}
src/MS/CMakeLists.txt:        ${CUDA_cusolver_LIBRARY}
src/MS/CMakeLists.txt:        ${CUDA_cudadevrt_LIBRARY}
src/MS/CMakeLists.txt:    install(TARGETS sagecal_gpu DESTINATION bin)
src/MS/data.cpp:int Data::gpu_threads=128;
src/MS/data.cpp:int Data::GPUpredict=0; /* use CPU for model calculation, if GPU not specified */
src/MS/data.cpp:#ifdef HAVE_CUDA
src/MS/data.cpp:int Data::heapsize=GPU_HEAP_SIZE; /* heap size in GPU (MB) to be used in malloc() */
src/MS/data.h:    extern int gpu_threads;
src/MS/data.h:    extern int GPUpredict; /* if given, use GPU for model calculation */
src/MS/data.h:    extern int heapsize; /* heap size in GPU (MB), for using malloc() */
src/MS/fullbatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/fullbatch_mode.cpp:   /* setup Heap of GPU,  only need to be done once, before any kernel is launched  */
src/MS/fullbatch_mode.cpp:    if (GPUpredict>0) {
src/MS/fullbatch_mode.cpp:     for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
src/MS/fullbatch_mode.cpp:        cudaSetDevice(gpuid);
src/MS/fullbatch_mode.cpp:        cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
src/MS/fullbatch_mode.cpp:  int mic_data_gpu_threads=Data::gpu_threads;
src/MS/fullbatch_mode.cpp:#ifndef HAVE_CUDA
src/MS/fullbatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/fullbatch_mode.cpp:   if (GPUpredict) {
src/MS/fullbatch_mode.cpp:     precalculate_coherencies_withbeam_gpu(iodata.u,iodata.v,iodata.w,coh,iodata.N,iodata.Nbase*iodata.tilesz,barr,carr,M,iodata.freq0,iodata.deltaf,iodata.deltat,iodata.dec0,Data::min_uvcut,Data::max_uvcut,
src/MS/fullbatch_mode.cpp:#ifndef HAVE_CUDA
src/MS/fullbatch_mode.cpp:     sagefit_visibilities_mic(mic_u,mic_v,mic_w,mic_x,mic_N,mic_Nbase,mic_tilesz,barr,mic_chunks,mic_pindex,coh,M,Mt,mic_freq0,mic_deltaf,p,mic_data_min_uvcut,mic_data_Nt,2*mic_data_max_emiter,mic_data_max_iter,(mic_data_dochan? 0 :mic_data_max_lbfgs),mic_data_lbfgs_m,mic_data_gpu_threads,mic_data_linsolv,mic_data_solver_mode,mic_data_nulow,mic_data_nuhigh,mic_data_randomize,&mean_nu,&res_0,&res_1);
src/MS/fullbatch_mode.cpp:     sagefit_visibilities(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,(iodata.N<=LMCUT?4*Data::max_emiter:6*Data::max_emiter),Data::max_iter,(Data::doChan? 0 :Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,(iodata.N<=LMCUT && Data::solver_mode==SM_RTR_OSLM_LBFGS?SM_OSLM_LBFGS:(iodata.N<=LMCUT && (Data::solver_mode==SM_RTR_OSRLM_RLBFGS||Data::solver_mode==SM_NSD_RLBFGS)?SM_OSLM_OSRLM_RLBFGS:Data::solver_mode)),Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
src/MS/fullbatch_mode.cpp:     //sagefit_visibilities(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,2*Data::max_emiter,Data::max_iter,(Data::doChan? 0 :Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
src/MS/fullbatch_mode.cpp:     sagefit_visibilities_mic(mic_u,mic_v,mic_w,mic_x,mic_N,mic_Nbase,mic_tilesz,barr,mic_chunks,mic_pindex,coh,M,Mt,mic_freq0,mic_deltaf,p,mic_data_min_uvcut,mic_data_Nt,mic_data_max_emiter,mic_data_max_iter,(mic_data_dochan? 0: mic_data_max_lbfgs),mic_data_lbfgs_m,mic_data_gpu_threads,mic_data_linsolv,mic_data_solver_mode,mic_data_nulow,mic_data_nuhigh,mic_data_randomize,&mean_nu,&res_0,&res_1);
src/MS/fullbatch_mode.cpp:     sagefit_visibilities(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,(Data::doChan? 0: Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
src/MS/fullbatch_mode.cpp:#endif /* !HAVE_CUDA */
src/MS/fullbatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/fullbatch_mode.cpp://#ifdef ONE_GPU
src/MS/fullbatch_mode.cpp://     sagefit_visibilities_dual_pt_one_gpu(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt, (iodata.N<=LMCUT?4*Data::max_emiter:6*Data::max_emiter),Data::max_iter,(Data::doChan? 0: Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,(iodata.N<=LMCUT && Data::solver_mode==SM_RTR_OSLM_LBFGS?SM_OSLM_LBFGS:(iodata.N<=LMCUT && (Data::solver_mode==SM_RTR_OSRLM_RLBFGS||Data::solver_mode==SM_NSD_RLBFGS)?SM_OSLM_OSRLM_RLBFGS:Data::solver_mode)),Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
src/MS/fullbatch_mode.cpp://     sagefit_visibilities_dual_pt_one_gpu(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,(Data::doChan? 0:Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
src/MS/fullbatch_mode.cpp://#endif /* ONE_GPU */
src/MS/fullbatch_mode.cpp://#ifndef ONE_GPU
src/MS/fullbatch_mode.cpp:     sagefit_visibilities_dual_pt_flt(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,(iodata.N<=LMCUT?4*Data::max_emiter:6*Data::max_emiter),Data::max_iter,(Data::doChan? 0:Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,(iodata.N<=LMCUT && Data::solver_mode==SM_RTR_OSLM_LBFGS?SM_OSLM_LBFGS:(iodata.N<=LMCUT && (Data::solver_mode==SM_RTR_OSRLM_RLBFGS||Data::solver_mode==SM_NSD_RLBFGS)?SM_OSLM_OSRLM_RLBFGS:Data::solver_mode)),Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
src/MS/fullbatch_mode.cpp:     ///DBG sagefit_visibilities_dual_pt_flt(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,2*Data::max_emiter,Data::max_iter,(Data::doChan? 0:Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
src/MS/fullbatch_mode.cpp:     sagefit_visibilities_dual_pt_flt(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freq0,iodata.deltaf,p,Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,(Data::doChan? 0:Data::max_lbfgs),Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
src/MS/fullbatch_mode.cpp://#endif /* !ONE_GPU */
src/MS/fullbatch_mode.cpp:#endif /* HAVE_CUDA */
src/MS/fullbatch_mode.cpp:#ifndef HAVE_CUDA
src/MS/fullbatch_mode.cpp:        bfgsfit_visibilities_mic(mic_u,mic_v,mic_w,xfreq,mic_N,mic_Nbase,mic_tilesz,barr,mic_chunks,mic_pindex,coh,M,Mt,mic_freq0,mic_deltaf,pfreq,mic_data_min_uvcut,mic_data_Nt,mic_data_max_lbfgs,mic_data_lbfgs_m,mic_data_gpu_threads,mic_data_solver_mode,mean_nu,&res_00,&res_01);
src/MS/fullbatch_mode.cpp:        bfgsfit_visibilities(iodata.u,iodata.v,iodata.w,xfreq,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freqs[ci],deltafch,pfreq,Data::min_uvcut,Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00,&res_01);
src/MS/fullbatch_mode.cpp:#endif /* !HAVE_CUDA */
src/MS/fullbatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/fullbatch_mode.cpp:        bfgsfit_visibilities_gpu(iodata.u,iodata.v,iodata.w,xfreq,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,coh,M,Mt,iodata.freqs[ci],deltafch,pfreq,Data::min_uvcut,Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00,&res_01);
src/MS/fullbatch_mode.cpp:#endif /* HAVE_CUDA */
src/MS/fullbatch_mode.cpp:#ifndef HAVE_CUDA
src/MS/fullbatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/fullbatch_mode.cpp:    if (GPUpredict) {
src/MS/fullbatch_mode.cpp:      calculate_residuals_multifreq_withbeam_gpu(iodata.u,iodata.v,iodata.w,p,iodata.xo,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,iodata.deltaf,iodata.deltat,iodata.dec0,
src/MS/fullbatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/fullbatch_mode.cpp:#endif /* HAVE_CUDA */
src/MS/fullbatch_mode.cpp:#ifndef HAVE_CUDA
src/MS/fullbatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/fullbatch_mode.cpp:     if (GPUpredict) {
src/MS/fullbatch_mode.cpp:        fprintf(stderr,"GPU predict is not supported for this telescope, try CPU only predict\n");
src/MS/fullbatch_mode.cpp:      predict_visibilities_multifreq_withbeam_gpu(iodata.u,iodata.v,iodata.w,iodata.xo,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,iodata.deltaf,iodata.deltat,iodata.dec0,
src/MS/fullbatch_mode.cpp:        fprintf(stderr,"GPU predict is not supported for this telescope, try CPU only predict\n");
src/MS/fullbatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/fullbatch_mode.cpp:     if (GPUpredict) {
src/MS/fullbatch_mode.cpp:      predict_visibilities_withsol_withbeam_gpu(iodata.u,iodata.v,iodata.w,p,iodata.xo,ignorelist,iodata.N,iodata.Nbase,iodata.tilesz,barr,carr,M,iodata.freqs,iodata.Nchan,iodata.deltaf,iodata.deltat,iodata.dec0,
src/MS/fullbatch_mode.cpp:#ifndef HAVE_CUDA
src/MS/fullbatch_mode.cpp:#ifdef HAVE_CUDA
src/MS/fullbatch_mode.cpp:   /* if -E uses a large value ~say 100, at each multiple of this, clear GPU memory */
src/MS/fullbatch_mode.cpp:   if (GPUpredict>1 && tilex>0 && !(tilex%GPUpredict)) {
src/MS/fullbatch_mode.cpp:    for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
src/MS/fullbatch_mode.cpp:       cudaSetDevice(gpuid);
src/MS/fullbatch_mode.cpp:       cudaDeviceReset();
src/MS/fullbatch_mode.cpp:       cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
src/uvwriter/CMakeLists.txt:if(HAVE_CUDA)
src/uvwriter/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
src/uvwriter/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CASACORE_INCLUDE_DIR})
src/uvwriter/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CASACORE_INCLUDE_DIR}/casacore)
src/uvwriter/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Dirac)
src/uvwriter/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio)
src/uvwriter/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio/reserve)
src/uvwriter/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
src/uvwriter/CMakeLists.txt:if (HAVE_CUDA)
src/uvwriter/CMakeLists.txt:  cuda_add_executable(uvwriter ${SRCFILES})
src/uvwriter/CMakeLists.txt:        ${CUDA_CUBLAS_LIBRARIES}
src/uvwriter/CMakeLists.txt:        ${CUDA_CUFFT_LIBRARIES}
src/uvwriter/CMakeLists.txt:        ${CUDA_cusolver_LIBRARIES}
src/uvwriter/CMakeLists.txt:        ${CUDA_cudadevrt_LIBRARIES}
src/MPI/main.cpp:#ifdef HAVE_CUDA
src/MPI/main.cpp:   cout << "-E 0,1 : if >0, use GPU for model computing: default " <<Data::GPUpredict<< endl;
src/MPI/main.cpp:#ifdef HAVE_CUDA
src/MPI/main.cpp:   cout << "-S GPU heap size (MB): default "<<Data::heapsize<< endl;
src/MPI/main.cpp:                GPUpredict=atoi(optarg);
src/MPI/main.cpp:#ifdef HAVE_CUDA
src/MPI/CMakeLists.txt:if(HAVE_CUDA)
src/MPI/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
src/MPI/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CASACORE_INCLUDE_DIR})
src/MPI/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CASACORE_INCLUDE_DIR}/casacore)
src/MPI/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Dirac)
src/MPI/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio)
src/MPI/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/Radio/reserve)
src/MPI/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
src/MPI/CMakeLists.txt:if(HAVE_CUDA)
src/MPI/CMakeLists.txt:  cuda_add_executable(sagecal-mpi_gpu ${SRCFILES})
src/MPI/CMakeLists.txt:  add_dependencies(sagecal-mpi_gpu dirac-radio dirac)
src/MPI/CMakeLists.txt:  target_link_libraries(sagecal-mpi_gpu
src/MPI/CMakeLists.txt:    ${CUDA_CUBLAS_LIBRARIES}
src/MPI/CMakeLists.txt:    ${CUDA_CUFFT_LIBRARIES}
src/MPI/CMakeLists.txt:    ${CUDA_cusolver_LIBRARY}
src/MPI/CMakeLists.txt:    ${CUDA_cudadevrt_LIBRARY}
src/MPI/CMakeLists.txt:  install(TARGETS sagecal-mpi_gpu DESTINATION bin)
src/MPI/data.cpp:int Data::gpu_threads=128;
src/MPI/data.cpp:int Data::GPUpredict=0; /* use CPU for model calculation, if GPU not specified */
src/MPI/data.cpp:#ifdef HAVE_CUDA
src/MPI/data.cpp:int Data::heapsize=GPU_HEAP_SIZE; /* heap size in GPU (MB) to be used in malloc() */
src/MPI/sagecal_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_slave.cpp:   /* setup Heap of GPU,  only need to be done once, before any kernel is launched  */
src/MPI/sagecal_slave.cpp:    if (GPUpredict>0) {
src/MPI/sagecal_slave.cpp:     for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
src/MPI/sagecal_slave.cpp:        cudaSetDevice(gpuid);
src/MPI/sagecal_slave.cpp:        cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
src/MPI/sagecal_slave.cpp:#ifndef HAVE_CUDA
src/MPI/sagecal_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_slave.cpp:     if (GPUpredict) {
src/MPI/sagecal_slave.cpp:       precalculate_coherencies_withbeam_gpu(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,coh_vec[cm],iodata_vec[cm].N,iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,iodata_vec[cm].deltat,iodata_vec[cm].dec0,Data::min_uvcut,Data::max_uvcut,
src/MPI/sagecal_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_slave.cpp:     if (GPUpredict) {
src/MPI/sagecal_slave.cpp:#endif /* HAVE_CUDA */
src/MPI/sagecal_slave.cpp:#ifndef HAVE_CUDA
src/MPI/sagecal_slave.cpp:       sagefit_visibilities(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,iodata_vec[cm].x,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],coh_vec[cm],M,Mt,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,p_vec[cm],Data::min_uvcut,Data::Nt,(iodata_vec[cm].N<=LMCUT?4*Data::max_emiter:6*Data::max_emiter),Data::max_iter,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,(iodata_vec[cm].N<=LMCUT && Data::solver_mode==SM_RTR_OSLM_LBFGS?SM_OSLM_LBFGS:(iodata_vec[cm].N<=LMCUT && (Data::solver_mode==SM_RTR_OSRLM_RLBFGS||Data::solver_mode==SM_NSD_RLBFGS)?SM_OSLM_OSRLM_RLBFGS:Data::solver_mode)),Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1); 
src/MPI/sagecal_slave.cpp:       sagefit_visibilities(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,iodata_vec[cm].x,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],coh_vec[cm],M,Mt,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,p_vec[cm],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
src/MPI/sagecal_slave.cpp:#endif /* !HAVE_CUDA */
src/MPI/sagecal_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_slave.cpp:       sagefit_visibilities_dual_pt_flt(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,iodata_vec[cm].x,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],coh_vec[cm],M,Mt,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,p_vec[cm],Data::min_uvcut,Data::Nt,(iodata_vec[cm].N<=LMCUT?4*Data::max_emiter:6*Data::max_emiter),Data::max_iter,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,(iodata_vec[cm].N<=LMCUT && Data::solver_mode==SM_RTR_OSLM_LBFGS?SM_OSLM_LBFGS:(iodata_vec[cm].N<=LMCUT && (Data::solver_mode==SM_RTR_OSRLM_RLBFGS||Data::solver_mode==SM_NSD_RLBFGS)?SM_OSLM_OSRLM_RLBFGS:Data::solver_mode)),Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
src/MPI/sagecal_slave.cpp:       sagefit_visibilities_dual_pt_flt(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,iodata_vec[cm].x,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],coh_vec[cm],M,Mt,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,p_vec[cm],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,&mean_nu,&res_0,&res_1);
src/MPI/sagecal_slave.cpp:#endif /* HAVE_CUDA */
src/MPI/sagecal_slave.cpp:#ifndef HAVE_CUDA
src/MPI/sagecal_slave.cpp:       sagefit_visibilities_admm(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,iodata_vec[cm].x,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],coh_vec[cm],M,Mt,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,p_vec[cm],Y_vec[cm],Z_vec[cm],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,0,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,arho_vec[cm],&mean_nu,&res_0,&res_1);
src/MPI/sagecal_slave.cpp:#endif /* !HAVE_CUDA */
src/MPI/sagecal_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_slave.cpp:       sagefit_visibilities_admm_dual_pt_flt(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,iodata_vec[cm].x,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],coh_vec[cm],M,Mt,iodata_vec[cm].freq0,iodata_vec[cm].deltaf,p_vec[cm],Y_vec[cm],Z_vec[cm],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,0,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,arho_vec[cm],&mean_nu,&res_0,&res_1);
src/MPI/sagecal_slave.cpp:#endif /* HAVE_CUDA */
src/MPI/sagecal_slave.cpp:#ifndef HAVE_CUDA
src/MPI/sagecal_slave.cpp:       sagefit_visibilities_admm(iodata_vec[mmid].u,iodata_vec[mmid].v,iodata_vec[mmid].w,iodata_vec[mmid].x,iodata_vec[mmid].N,iodata_vec[mmid].Nbase,iodata_vec[mmid].tilesz,barr_vec[mmid],carr_vec[mmid],coh_vec[mmid],M,Mt,iodata_vec[mmid].freq0,iodata_vec[mmid].deltaf,p_vec[mmid],Y_vec[mmid],Z_vec[mmid],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,0,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,arho_vec[mmid],&mean_nu,&res_0,&res_1);
src/MPI/sagecal_slave.cpp:#endif /* !HAVE_CUDA */
src/MPI/sagecal_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_slave.cpp:       //sagefit_visibilities_admm(iodata.u,iodata.v,iodata.w,iodata.x,iodata.N,iodata.Nbase,iodata.tilesz,barr_vec[mmid],carr_vec[mmid],coh_vec[mmid],M,Mt,iodata.freq0,iodata.deltaf,p,Y_vec[mmid],Z_vec[mmid],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,0,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,arho_vec[mmid],&mean_nu,&res_0,&res_1);
src/MPI/sagecal_slave.cpp:       sagefit_visibilities_admm_dual_pt_flt(iodata_vec[mmid].u,iodata_vec[mmid].v,iodata_vec[mmid].w,iodata_vec[mmid].x,iodata_vec[mmid].N,iodata_vec[mmid].Nbase,iodata_vec[mmid].tilesz,barr_vec[mmid],carr_vec[mmid],coh_vec[mmid],M,Mt,iodata_vec[mmid].freq0,iodata_vec[mmid].deltaf,p_vec[mmid],Y_vec[mmid],Z_vec[mmid],Data::min_uvcut,Data::Nt,Data::max_emiter,Data::max_iter,0,Data::lbfgs_m,Data::gpu_threads,Data::linsolv,Data::solver_mode,Data::nulow,Data::nuhigh,Data::randomize,arho_vec[mmid],&mean_nu,&res_0,&res_1);
src/MPI/sagecal_slave.cpp:#endif /* HAVE_CUDA */
src/MPI/sagecal_slave.cpp:#ifndef HAVE_CUDA
src/MPI/sagecal_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_slave.cpp:     if (GPUpredict) {
src/MPI/sagecal_slave.cpp:       calculate_residuals_multifreq_withbeam_gpu(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,p_vec[cm],iodata_vec[cm].xo,iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,iodata_vec[cm].freqs,iodata_vec[cm].Nchan,iodata_vec[cm].deltaf,iodata_vec[cm].deltat,iodata_vec[cm].dec0,
src/MPI/sagecal_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_slave.cpp:   /* if -E uses a large value ~say 100, at each multiple of this, clear GPU memory */
src/MPI/sagecal_slave.cpp:   if (GPUpredict>1 && tilex>0 && !(tilex%GPUpredict)) {
src/MPI/sagecal_slave.cpp:    for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
src/MPI/sagecal_slave.cpp:       cudaSetDevice(gpuid);
src/MPI/sagecal_slave.cpp:       cudaDeviceReset();
src/MPI/sagecal_slave.cpp:       cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
src/MPI/data.h:    extern int gpu_threads;
src/MPI/data.h:    extern int GPUpredict; /* if given, use GPU for model calculation */
src/MPI/data.h:    extern int heapsize; /* heap size in GPU (MB), for using malloc() */
src/MPI/sagecal_stochastic_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_stochastic_slave.cpp:   /* setup Heap of GPU,  only need to be done once, before any kernel is launched  */
src/MPI/sagecal_stochastic_slave.cpp:    if (GPUpredict>0) {
src/MPI/sagecal_stochastic_slave.cpp:     for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
src/MPI/sagecal_stochastic_slave.cpp:        cudaSetDevice(gpuid);
src/MPI/sagecal_stochastic_slave.cpp:        cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
src/MPI/sagecal_stochastic_slave.cpp:    /* for attaching to a GPU */
src/MPI/sagecal_stochastic_slave.cpp:    attach_gpu_to_thread(select_work_gpu(MAX_GPU_ID,&thst), &cbhandle, &solver_handle);
src/MPI/sagecal_stochastic_slave.cpp:    /* auxilliary arrays for GPU */
src/MPI/sagecal_stochastic_slave.cpp:        lbfgs_persist_init(&ptdata_array[cm*nsolbw+ii],minibatches,iodata_vec[cm].N*8*Mt,iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,Data::lbfgs_m,Data::gpu_threads);
src/MPI/sagecal_stochastic_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_stochastic_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_stochastic_slave.cpp:#ifndef HAVE_CUDA
src/MPI/sagecal_stochastic_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_stochastic_slave.cpp:   if (GPUpredict) {
src/MPI/sagecal_stochastic_slave.cpp:     precalculate_coherencies_multifreq_withbeam_gpu(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,coh_vec[cm],iodata_vec[cm].N,iodata_vec[cm].Nbase*iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,iodata_vec[cm].freqs,iodata_vec[cm].Nchan,deltafch,iodata_vec[cm].deltat,iodata_vec[cm].dec0,Data::min_uvcut,Data::max_uvcut,
src/MPI/sagecal_stochastic_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_stochastic_slave.cpp:        bfgsfit_minibatch_consensus(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,&iodata_vec[cm].xo[iodata_vec[cm].Nbase*iodata_vec[cm].tilesz*8*chanstart[ii]],iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,hbb,ptoclus,&coh_vec[cm][M*iodata_vec[cm].Nbase*iodata_vec[cm].tilesz*4*chanstart[ii]],M,Mt,&iodata_vec[cm].freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],&pfreq_vec[cm][iodata_vec[cm].N*8*Mt*ii],&Y[iodata_vec[cm].N*8*Mt*(ii+cm*nsolbw)],z,&rhok[cm*Mt*nsolbw+ii*Mt],Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00[cm],&res_01[cm],&ptdata_array[cm*nsolbw+ii],nmb,minibatches);
src/MPI/sagecal_stochastic_slave.cpp:        bfgsfit_minibatch_consensus(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,&iodata_vec[cm].xo[iodata_vec[cm].Nbase*iodata_vec[cm].tilesz*8*chanstart[ii]],iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],&coh_vec[cm][M*iodata_vec[cm].Nbase*iodata_vec[cm].tilesz*4*chanstart[ii]],M,Mt,&iodata_vec[cm].freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],&pfreq_vec[cm][iodata_vec[cm].N*8*Mt*ii],&Y[iodata_vec[cm].N*8*Mt*(ii+cm*nsolbw)],z,&rhok[cm*Mt*nsolbw+ii*Mt],Data::Nt,Data::max_lbfgs,Data::lbfgs_m,Data::gpu_threads,Data::solver_mode,mean_nu,&res_00[cm],&res_01[cm],&ptdata_array[cm*nsolbw+ii],nmb,minibatches);
src/MPI/sagecal_stochastic_slave.cpp:#ifndef HAVE_CUDA
src/MPI/sagecal_stochastic_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_stochastic_slave.cpp:    if (GPUpredict) {
src/MPI/sagecal_stochastic_slave.cpp:      calculate_residuals_multifreq_withbeam_gpu(iodata_vec[cm].u,iodata_vec[cm].v,iodata_vec[cm].w,&pfreq_vec[cm][iodata_vec[cm].N*8*Mt*ii],&iodata_vec[cm].xo[iodata_vec[cm].Nbase*iodata_vec[cm].tilesz*8*chanstart[ii]],iodata_vec[cm].N,iodata_vec[cm].Nbase,iodata_vec[cm].tilesz,barr_vec[cm],carr_vec[cm],M,&iodata_vec[cm].freqs[chanstart[ii]],nchan[ii],deltafch*(double)nchan[ii],iodata_vec[cm].deltat,iodata_vec[cm].dec0,
src/MPI/sagecal_stochastic_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_stochastic_slave.cpp:   /* if -E uses a large value ~say 100, at each multiple of this, clear GPU memory */
src/MPI/sagecal_stochastic_slave.cpp:   if (GPUpredict>1 && tilex>0 && !(tilex%GPUpredict)) {
src/MPI/sagecal_stochastic_slave.cpp:    for (int gpuid=0; gpuid<=MAX_GPU_ID; gpuid++) {
src/MPI/sagecal_stochastic_slave.cpp:       cudaSetDevice(gpuid);
src/MPI/sagecal_stochastic_slave.cpp:       cudaDeviceReset();
src/MPI/sagecal_stochastic_slave.cpp:       cudaDeviceSetLimit(cudaLimitMallocHeapSize, Data::heapsize*1024*1024);
src/MPI/sagecal_stochastic_slave.cpp:#ifdef HAVE_CUDA
src/MPI/sagecal_stochastic_slave.cpp:   detach_gpu_from_thread(cbhandle,solver_handle);
src/lib/Dirac/clmfit_cuda.c:#include <cuda_runtime.h>
src/lib/Dirac/clmfit_cuda.c:checkCudaError(cudaError_t err, char *file, int line)
src/lib/Dirac/clmfit_cuda.c:#ifdef CUDA_DEBUG
src/lib/Dirac/clmfit_cuda.c:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Dirac/clmfit_cuda.c:#ifdef CUDA_DEBUG
src/lib/Dirac/clmfit_cuda.c:  cudaError_t err;
src/lib/Dirac/clmfit_cuda.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/clmfit_cuda.c:  err=cudaSetDevice(card);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&xd, N*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpy(xd, x, N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&jacd, M*N*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&jacTjacd, M*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&jacTed, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&jacTjacd0, M*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&Dpd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&bd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&pd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&pnewd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpy(pd, p, M*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&devInfo, sizeof(int));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&taud, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&Ud, M*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&VTd, M*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&Sd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&rwork, 5*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&hxd, N*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpy(hxd, hx, N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&ed, N*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMemcpy(jacd, jac, M*N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:     err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:      err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:      cudakernel_diagmu(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);
src/lib/Dirac/clmfit_cuda.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:        cudaDeviceSynchronize();
src/lib/Dirac/clmfit_cuda.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:         cudaDeviceSynchronize();
src/lib/Dirac/clmfit_cuda.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:        cudaDeviceSynchronize();
src/lib/Dirac/clmfit_cuda.c://        err=cudaMemcpy(pnew,Dpd,M*sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c://        checkCudaError(err);
src/lib/Dirac/clmfit_cuda.c://        err=cudaMemcpy(jTjdiag,Sd,M*sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c://        checkCudaError(err);
src/lib/Dirac/clmfit_cuda.c://        err=cudaMemcpy(Dpd,pnew,M*sizeof(double),cudaMemcpyHostToDevice);
src/lib/Dirac/clmfit_cuda.c://        checkCudaError(err);
src/lib/Dirac/clmfit_cuda.c:        cudakernel_diagdiv(ThreadsPerBlock, BlocksPerGrid, M, eps1, Dpd, Sd);
src/lib/Dirac/clmfit_cuda.c:          err=cudaMemcpy(pnew,pnewd,M*sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:        err=cudaMemcpy(hxd, hx, N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/clmfit_cuda.c:        checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMemcpy(p,pd,M*sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpy(p,pd,M*sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(xd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(jacd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(jacTjacd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(jacTjacd0);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(jacTed);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(Dpd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(bd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(pd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(pnewd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(hxd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(ed);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(devInfo);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(work);
src/lib/Dirac/clmfit_cuda.c:   cudaFree(taud);
src/lib/Dirac/clmfit_cuda.c:   cudaFree(Ud);
src/lib/Dirac/clmfit_cuda.c:   cudaFree(VTd);
src/lib/Dirac/clmfit_cuda.c:   cudaFree(Sd);
src/lib/Dirac/clmfit_cuda.c:   cudaFree(rwork);
src/lib/Dirac/clmfit_cuda.c:  entirely in the GPU */
src/lib/Dirac/clmfit_cuda.c:clevmar_der_single_cuda(
src/lib/Dirac/clmfit_cuda.c:  double *gWORK, /* GPU allocated memory */
src/lib/Dirac/clmfit_cuda.c:  cudaError_t err;
src/lib/Dirac/clmfit_cuda.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&xd, N*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&jacd, M*N*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&jacTjacd, M*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&jacTed, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&jacTjacd0, M*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&Dpd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&bd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&pd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&pnewd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**) &bbd, Nbase*2*sizeof(short));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**) &cohd, Nbase*8*sizeof(double)); 
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&hxd, N*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&ed, N*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&taud, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&Ud, M*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&VTd, M*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&Sd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&devInfo, sizeof(int));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&rwork, 5*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpyAsync(pd, p, M*sizeof(double), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpyAsync(cohd, &(dp->ddcoh[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(double), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpyAsync(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  cudaDeviceSynchronize();
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpyAsync(xd, x, N*sizeof(double), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,hxd,M,N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/clmfit_cuda.c:     cudakernel_jacf(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/clmfit_cuda.c:     err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:      err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:      cudakernel_diagmu(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);
src/lib/Dirac/clmfit_cuda.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:        cudaDeviceSynchronize();
src/lib/Dirac/clmfit_cuda.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:         cudaDeviceSynchronize();
src/lib/Dirac/clmfit_cuda.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:        cudaDeviceSynchronize();
src/lib/Dirac/clmfit_cuda.c:        cudakernel_diagdiv(ThreadsPerBlock, BlocksPerGrid, M, eps1, Dpd, Sd);
src/lib/Dirac/clmfit_cuda.c:        cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpyAsync(p,pd,M*sizeof(double),cudaMemcpyDeviceToHost,0);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  cudaDeviceSynchronize();
src/lib/Dirac/clmfit_cuda.c:  cudaFree(xd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(jacd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(jacTjacd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(jacTjacd0);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(jacTed);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(Dpd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(bd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(pd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(pnewd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(hxd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(ed);
src/lib/Dirac/clmfit_cuda.c:   cudaFree(taud);
src/lib/Dirac/clmfit_cuda.c:   cudaFree(Ud);
src/lib/Dirac/clmfit_cuda.c:   cudaFree(VTd);
src/lib/Dirac/clmfit_cuda.c:   cudaFree(Sd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(cohd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(bbd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(devInfo);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(work);
src/lib/Dirac/clmfit_cuda.c:    cudaFree(rwork);
src/lib/Dirac/clmfit_cuda.c:/* function to set up a GPU, should be called only once */
src/lib/Dirac/clmfit_cuda.c:attach_gpu_to_thread(int card,  cublasHandle_t *cbhandle, cusolverDnHandle_t *solver_handle) {
src/lib/Dirac/clmfit_cuda.c:  cudaError_t err;
src/lib/Dirac/clmfit_cuda.c:  err=cudaSetDevice(card);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:attach_gpu_to_thread1(int card,  cublasHandle_t *cbhandle, cusolverDnHandle_t *solver_handle, double **WORK, int64_t work_size) {
src/lib/Dirac/clmfit_cuda.c:  cudaError_t err;
src/lib/Dirac/clmfit_cuda.c:  err=cudaSetDevice(card);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)WORK, (size_t)work_size);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:attach_gpu_to_thread2(int card,  cublasHandle_t *cbhandle,  cusolverDnHandle_t *solver_handle, float **WORK, int64_t work_size, int usecula) {
src/lib/Dirac/clmfit_cuda.c:  cudaError_t err;
src/lib/Dirac/clmfit_cuda.c:  err=cudaSetDevice(card); /* we need this */
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)WORK, (size_t)work_size);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:detach_gpu_from_thread(cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
src/lib/Dirac/clmfit_cuda.c:detach_gpu_from_thread1(cublasHandle_t cbhandle,cusolverDnHandle_t solver_handle,double *WORK) {
src/lib/Dirac/clmfit_cuda.c:  cudaFree(WORK);
src/lib/Dirac/clmfit_cuda.c:detach_gpu_from_thread2(cublasHandle_t cbhandle,cusolverDnHandle_t solver_handle,float *WORK, int usecula) {
src/lib/Dirac/clmfit_cuda.c:  cudaFree(WORK);
src/lib/Dirac/clmfit_cuda.c:reset_gpu_memory(double *WORK, int64_t work_size) {
src/lib/Dirac/clmfit_cuda.c:  cudaError_t err;
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemset((void*)WORK, 0, (size_t)work_size);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:mlm_der_single_cuda(
src/lib/Dirac/clmfit_cuda.c:  double *gWORK, /* GPU allocated memory */
src/lib/Dirac/clmfit_cuda.c:  cudaError_t err;
src/lib/Dirac/clmfit_cuda.c:  /* use cudaHostAlloc  and cudaFreeHost */
src/lib/Dirac/clmfit_cuda.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&xd, N*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&Jkd, M*N*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&Fxkd, N*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&Fykd, N*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&Jkdkd, N*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&JkTed, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&JkTed0, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&JkTJkd, M*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&JkTJkd0, M*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&dkd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&dhatkd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&ykd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&skd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&pd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**) &bbd, Nbase*2*sizeof(short));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**) &cohd, Nbase*8*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&taud, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&Ud, M*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&VTd, M*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&Sd, M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMalloc((void**)&devInfo, sizeof(int));
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:    err=cudaMalloc((void**)&rwork, 5*M*sizeof(double));
src/lib/Dirac/clmfit_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpyAsync(pd, p, M*sizeof(double), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpyAsync(cohd, &(dp->ddcoh[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(double), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpyAsync(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:  cudaDeviceSynchronize();
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpyAsync(xd, x, N*sizeof(double), cudaMemcpyHostToDevice, 0);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:   cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,Fxkd,M,N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/clmfit_cuda.c:      cudakernel_jacf(ThreadsPerBlock, ThreadsPerBlock/4, pd, Jkd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/clmfit_cuda.c:     cudakernel_diagmu(ThreadsPerBlock, BlocksPerGrid, M, JkTJkd, lambda);
src/lib/Dirac/clmfit_cuda.c:       cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:       cudaDeviceSynchronize();
src/lib/Dirac/clmfit_cuda.c:       cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:         cudaDeviceSynchronize();
src/lib/Dirac/clmfit_cuda.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:        cudaDeviceSynchronize();
src/lib/Dirac/clmfit_cuda.c:        cudakernel_diagdiv(ThreadsPerBlock, BlocksPerGrid, M, epsilon, dkd, Sd);
src/lib/Dirac/clmfit_cuda.c:     cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, ykd,Fykd,M,N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/clmfit_cuda.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:        cudaDeviceSynchronize();
src/lib/Dirac/clmfit_cuda.c:       cudakernel_diagdiv(ThreadsPerBlock, BlocksPerGrid, M, epsilon, dhatkd, Sd);
src/lib/Dirac/clmfit_cuda.c:  cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, ykd,Fykd,M,N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/clmfit_cuda.c:  //err=cudaMemcpy(p,pd,M*sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_cuda.c:  err=cudaMemcpyAsync(p,pd,M*sizeof(double),cudaMemcpyDeviceToHost,0);
src/lib/Dirac/clmfit_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_cuda.c:   cudaFree(taud);
src/lib/Dirac/clmfit_cuda.c:    cudaFree(Ud);
src/lib/Dirac/clmfit_cuda.c:    cudaFree(VTd);
src/lib/Dirac/clmfit_cuda.c:    cudaFree(Sd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(xd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(Jkd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(Fxkd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(Fykd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(Jkdkd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(JkTed);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(JkTed0);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(JkTJkd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(JkTJkd0);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(dkd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(dhatkd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(ykd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(skd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(pd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(bbd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(cohd);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(devInfo);
src/lib/Dirac/clmfit_cuda.c:  cudaFree(work);
src/lib/Dirac/clmfit_cuda.c:   cudaFree(rwork);
src/lib/Dirac/clmfit_cuda.c:  cudaDeviceSynchronize();
src/lib/Dirac/lmfit.c:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,double nulow, double nuhigh,int randomize,  double *mean_nu, double *res_0, double *res_1) {
src/lib/Dirac/lmfit.c:           clevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, opts, info, linsolv, (void*)&lmdata);  
src/lib/Dirac/lmfit.c:           oslevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, opts, info, linsolv, randomize, (void*)&lmdata);  
src/lib/Dirac/lmfit.c:         clevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, opts, info, linsolv, (void*)&lmdata);  
src/lib/Dirac/lmfit.c:          rlevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, NULL, info, linsolv, Nt, nulow, nuhigh, (void*)&lmdata);  
src/lib/Dirac/lmfit.c:           oslevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, opts, info, linsolv, randomize, (void*)&lmdata);  
src/lib/Dirac/lmfit.c:          osrlevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, NULL, info, linsolv, Nt,  nulow, nuhigh, randomize,  (void*)&lmdata);  
src/lib/Dirac/lmfit.c:           oslevmar_der_single_nocuda(mylm_fit_single_pth0, mylm_jac_single_pth, &p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], 8*N, 8*ntiles*Nbase, this_itermax, opts, info, linsolv, randomize, (void*)&lmdata);  
src/lib/Dirac/lmfit.c:           rtr_solve_nocuda(&p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], N, ntiles*Nbase, this_itermax+5, this_itermax+10, Delta0, Delta0*0.125, info, &lmdata);
src/lib/Dirac/lmfit.c:           rtr_solve_nocuda_robust(&p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], N, ntiles*Nbase, this_itermax+5, this_itermax+10, Delta0, Delta0*0.125, nulow, nuhigh, info, &lmdata);
src/lib/Dirac/lmfit.c:           nsd_solve_nocuda_robust(&p[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], N, ntiles*Nbase, this_itermax+15, nulow, nuhigh, info, &lmdata);
src/lib/Dirac/lmfit.c:     lbfgs_fit_robust_wrapper(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata);
src/lib/Dirac/lmfit.c:     lbfgs_fit_robust_wrapper_minibatch(p, x, m, n, max_lbfgs, -lbfgs_m, gpu_threads, (void*)&lmdata);
src/lib/Dirac/lmfit.c:    lbfgs_fit_wrapper(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata);
src/lib/Dirac/lmfit.c:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double mean_nu, double *res_0, double *res_1) {
src/lib/Dirac/lmfit.c:    lbfgs_fit_robust_wrapper(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata); 
src/lib/Dirac/lmfit.c:    lbfgs_fit_wrapper(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata);
src/lib/Dirac/lmfit.c:   int Nbase, int tilesz,  baseline_t *barr,  int *nchunks, int *pindex, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,double nulow, double nuhigh,int randomize, double *mean_nu, double *res_0, double *res_1) {
src/lib/Dirac/lmfit.c:  retval=sagefit_visibilities(u, v, w, x, N, Nbase, tilesz,  barr,  carr, coh, M, Mt, freq0, fdelta, pp, uvmin, Nt, max_emiter, max_iter, max_lbfgs, lbfgs_m, gpu_threads, linsolv,solver_mode,nulow, nuhigh,randomize, mean_nu, res_0, res_1);
src/lib/Dirac/lmfit.c:   int Nbase, int tilesz,  baseline_t *barr,  int *nchunks, int *pindex, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode,double nu_mean, double *res_0, double *res_1) {
src/lib/Dirac/lmfit.c:  retval=bfgsfit_visibilities(u, v, w, x, N, Nbase, tilesz,  barr,  carr, coh, M, Mt, freq0, fdelta, pp, uvmin, Nt, max_lbfgs, lbfgs_m, gpu_threads, solver_mode, nu_mean, res_0, res_1);
src/lib/Dirac/clmfit_fl.c:#include <cuda_runtime.h>
src/lib/Dirac/clmfit_fl.c:checkCudaError(cudaError_t err, const char *file, int line)
src/lib/Dirac/clmfit_fl.c:#ifdef CUDA_DEBUG
src/lib/Dirac/clmfit_fl.c:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Dirac/clmfit_fl.c:#ifdef CUDA_DEBUG
src/lib/Dirac/clmfit_fl.c:  entirely in the GPU */
src/lib/Dirac/clmfit_fl.c:oslevmar_der_single_cuda_fl(
src/lib/Dirac/clmfit_fl.c:  float *gWORK, /* GPU allocated memory */
src/lib/Dirac/clmfit_fl.c:  cudaError_t err;
src/lib/Dirac/clmfit_fl.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/clmfit_fl.c:  err=cudaMalloc((void**)&devInfo, sizeof(int));
src/lib/Dirac/clmfit_fl.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:    err=cudaMalloc((void**)&work, work_size*sizeof(float));
src/lib/Dirac/clmfit_fl.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:    err=cudaMalloc((void**)&work, work_size*sizeof(float));
src/lib/Dirac/clmfit_fl.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:    err=cudaMalloc((void**)&work, work_size*sizeof(float));
src/lib/Dirac/clmfit_fl.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:    err=cudaMalloc((void**)&rwork, 5*M*sizeof(float));
src/lib/Dirac/clmfit_fl.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:  err=cudaMemcpyAsync(pd, p, M*sizeof(float), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_fl.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:  err=cudaMemcpyAsync(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_fl.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:  err=cudaMemcpyAsync(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_fl.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:  cudaDeviceSynchronize();
src/lib/Dirac/clmfit_fl.c:  err=cudaMemcpyAsync(xd, x, N*sizeof(float), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_fl.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:  cudakernel_func_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,hxd,M,N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/clmfit_fl.c:     cudakernel_jacf_fl(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, Nos[l], &cohd[8*NbI[l]], &bbd[2*NbI[l]], Nbaseos[l], dp->M, dp->N);
src/lib/Dirac/clmfit_fl.c:     err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_fl.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:      err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_fl.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:      cudakernel_diagmu_fl(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);
src/lib/Dirac/clmfit_fl.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_fl.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_fl.c:        cudaDeviceSynchronize();
src/lib/Dirac/clmfit_fl.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_fl.c:         cudaDeviceSynchronize();
src/lib/Dirac/clmfit_fl.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_fl.c:        cudaDeviceSynchronize();
src/lib/Dirac/clmfit_fl.c:        cudakernel_diagdiv_fl(ThreadsPerBlock, BlocksPerGrid, M, eps1, Dpd, Sd);
src/lib/Dirac/clmfit_fl.c:        cudakernel_func_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/clmfit_fl.c:  err=cudaMemcpyAsync(p,pd,M*sizeof(float),cudaMemcpyDeviceToHost,0);
src/lib/Dirac/clmfit_fl.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:  cudaDeviceSynchronize();
src/lib/Dirac/clmfit_fl.c:  cudaFree(devInfo);
src/lib/Dirac/clmfit_fl.c:  cudaFree(work);
src/lib/Dirac/clmfit_fl.c:    cudaFree(rwork);
src/lib/Dirac/clmfit_fl.c:clevmar_der_single_cuda_fl(
src/lib/Dirac/clmfit_fl.c:  float *gWORK, /* GPU allocated memory */
src/lib/Dirac/clmfit_fl.c:  cudaError_t err;
src/lib/Dirac/clmfit_fl.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/clmfit_fl.c:  err=cudaMalloc((void**)&devInfo, sizeof(int));
src/lib/Dirac/clmfit_fl.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:    err=cudaMalloc((void**)&work, work_size*sizeof(float));
src/lib/Dirac/clmfit_fl.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:    err=cudaMalloc((void**)&work, work_size*sizeof(float));
src/lib/Dirac/clmfit_fl.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:    err=cudaMalloc((void**)&work, work_size*sizeof(float));
src/lib/Dirac/clmfit_fl.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:    err=cudaMalloc((void**)&rwork, 5*M*sizeof(float));
src/lib/Dirac/clmfit_fl.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:  err=cudaMemcpyAsync(pd, p, M*sizeof(float), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_fl.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:  err=cudaMemcpyAsync(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_fl.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:  err=cudaMemcpyAsync(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_fl.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:  cudaDeviceSynchronize();
src/lib/Dirac/clmfit_fl.c:  err=cudaMemcpyAsync(xd, x, N*sizeof(float), cudaMemcpyHostToDevice,0);
src/lib/Dirac/clmfit_fl.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:  cudakernel_func_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,hxd,M,N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/clmfit_fl.c:     cudakernel_jacf_fl(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/clmfit_fl.c:     err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_fl.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:      err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_fl.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:      cudakernel_diagmu_fl(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);
src/lib/Dirac/clmfit_fl.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_fl.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_fl.c:        cudaDeviceSynchronize();
src/lib/Dirac/clmfit_fl.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_fl.c:         cudaDeviceSynchronize();
src/lib/Dirac/clmfit_fl.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit_fl.c:        cudaDeviceSynchronize();
src/lib/Dirac/clmfit_fl.c:        cudakernel_diagdiv_fl(ThreadsPerBlock, BlocksPerGrid, M, eps1, Dpd, Sd);
src/lib/Dirac/clmfit_fl.c:        cudakernel_func_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/clmfit_fl.c:  err=cudaMemcpyAsync(p,pd,M*sizeof(float),cudaMemcpyDeviceToHost,0);
src/lib/Dirac/clmfit_fl.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/clmfit_fl.c:  cudaDeviceSynchronize();
src/lib/Dirac/clmfit_fl.c:  cudaFree(devInfo);
src/lib/Dirac/clmfit_fl.c:  cudaFree(work);
src/lib/Dirac/clmfit_fl.c:    cudaFree(rwork);
src/lib/Dirac/lbfgs_minibatch_cuda.c:#include <cuda.h>
src/lib/Dirac/lbfgs_minibatch_cuda.c:#include <cuda_runtime_api.h>
src/lib/Dirac/lbfgs_minibatch_cuda.c:#include <cuda_runtime.h>
src/lib/Dirac/lbfgs_minibatch_cuda.c://#define CUDA_DEBUG
src/lib/Dirac/lbfgs_minibatch_cuda.c:checkCudaError(cudaError_t err, char *file, int line)
src/lib/Dirac/lbfgs_minibatch_cuda.c:#ifdef CUDA_DEBUG
src/lib/Dirac/lbfgs_minibatch_cuda.c:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Dirac/lbfgs_minibatch_cuda.c:#ifdef CUDA_DEBUG
src/lib/Dirac/lbfgs_minibatch_cuda.c:              1: allocate GPU  memory, attach GPU
src/lib/Dirac/lbfgs_minibatch_cuda.c:              2: free GPU memory, detach GPU 
src/lib/Dirac/lbfgs_minibatch_cuda.c:              3,4..: do work on GPU 
src/lib/Dirac/lbfgs_minibatch_cuda.c:              99: reset GPU memory (memest all memory) */
src/lib/Dirac/lbfgs_minibatch_cuda.c:  thread_gpu_data *lmdata[2]; /* two for each thread */
src/lib/Dirac/lbfgs_minibatch_cuda.c:  /* GPU related info */
src/lib/Dirac/lbfgs_minibatch_cuda.c:  double *gWORK[2]; /* GPU buffers */
src/lib/Dirac/lbfgs_minibatch_cuda.c:  /* different pointers to GPU data */
src/lib/Dirac/lbfgs_minibatch_cuda.c:/* slave thread 2GPU function */
src/lib/Dirac/lbfgs_minibatch_cuda.c: cudaError_t err;
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(dp->cpp[tid], dp->lmdata[tid]->p, m*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:     cudakernel_lbfgs_r(dp->lmdata[tid]->ThreadsPerBlock, dp->lmdata[tid]->BlocksPerGrid, Nbase, dp->lmdata[tid]->tilesz, M, N, Nparam, dp->lmdata[tid]->g_start, dp->cxo[tid], dp->ccoh[tid], dp->cpp[tid], dp->cbb[tid], dp->cptoclus[tid], dp->cgrad[tid]);
src/lib/Dirac/lbfgs_minibatch_cuda.c:     cudakernel_lbfgs_r_robust(dp->lmdata[tid]->ThreadsPerBlock, dp->lmdata[tid]->BlocksPerGrid, Nbase, dp->lmdata[tid]->tilesz, M, N, Nparam, dp->lmdata[tid]->g_start, dp->cxo[tid], dp->ccoh[tid], dp->cpp[tid], dp->cbb[tid], dp->cptoclus[tid], dp->cgrad[tid],dp->lmdata[tid]->robust_nu);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(&(dp->lmdata[tid]->g[dp->lmdata[tid]->g_start]), dp->cgrad[tid], Nparam*sizeof(double), cudaMemcpyDeviceToHost);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:   err=cudaMemcpy(dp->cpp[tid], dp->lmdata[tid]->p, m*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    dp->fcost[tid]=cudakernel_lbfgs_cost(dp->lmdata[tid]->ThreadsPerBlock, BlocksPerGrid, dp->Nbase[tid], boff, M, N, Nbase, &dp->cxo[tid][8*boff], &dp->ccoh[tid][boff*8*M], dp->cpp[tid], &dp->cbb[tid][boff*2], dp->cptoclus[tid]); 
src/lib/Dirac/lbfgs_minibatch_cuda.c:    dp->fcost[tid]=cudakernel_lbfgs_cost_robust(dp->lmdata[tid]->ThreadsPerBlock, BlocksPerGrid, dp->Nbase[tid], boff, M, N, Nbase, &dp->cxo[tid][8*boff], &dp->ccoh[tid][boff*8*M], dp->cpp[tid], &dp->cbb[tid][boff*2], dp->cptoclus[tid], dp->lmdata[tid]->robust_nu);
src/lib/Dirac/lbfgs_minibatch_cuda.c:  } else if (dp->status[tid]==PT_DO_AGPU) {
src/lib/Dirac/lbfgs_minibatch_cuda.c:    attach_gpu_to_thread1(select_work_gpu(MAX_GPU_ID,td->pline->thst),&dp->cbhandle[tid],&dp->solver_handle[tid],&dp->gWORK[tid],dp->data_size[tid]);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(dp->cxo[tid]),dp->lmdata[tid]->n*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(dp->ccoh[tid]),Nbase*8*M*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(dp->cpp[tid]),m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(dp->cgrad[tid]),Nparam*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(dp->cptoclus[tid]),M*2*sizeof(int));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(dp->cbb[tid]),Nbase*2*sizeof(short));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(dp->cxo[tid], dp->lmdata[tid]->xo, dp->lmdata[tid]->n*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(dp->ccoh[tid], dp->lmdata[tid]->coh, Nbase*8*M*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(dp->cptoclus[tid], dp->lmdata[tid]->ptoclus, M*2*sizeof(int), cudaMemcpyHostToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(dp->cbb[tid], dp->lmdata[tid]->hbb, Nbase*2*sizeof(short), cudaMemcpyHostToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:  } else if (dp->status[tid]==PT_DO_DGPU) {
src/lib/Dirac/lbfgs_minibatch_cuda.c:    cudaFree(dp->cxo[tid]);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    cudaFree(dp->ccoh[tid]);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    cudaFree(dp->cptoclus[tid]);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    cudaFree(dp->cbb[tid]);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    cudaFree(dp->cpp[tid]);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    cudaFree(dp->cgrad[tid]);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    detach_gpu_from_thread1(dp->cbhandle[tid],dp->solver_handle[tid],dp->gWORK[tid]);
src/lib/Dirac/lbfgs_minibatch_cuda.c:cuda_mult_hessian(int m, double *pk, double *gk, double *s, double *y, double *rho, cublasHandle_t *cbhandle, int M, int ii) {
src/lib/Dirac/lbfgs_minibatch_cuda.c: cudaError_t err;
src/lib/Dirac/lbfgs_minibatch_cuda.c: err=cudaMemcpy(pk, gk, m*sizeof(double), cudaMemcpyDeviceToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c://printf("GPU cost=%lf\n",phi_0);
src/lib/Dirac/lbfgs_minibatch_cuda.c:  alphai1=0.0; /* FIXME: tune for GPU (defalut is 0.0) */
src/lib/Dirac/lbfgs_minibatch_cuda.c:   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads, int do_robust, void *adata) {
src/lib/Dirac/lbfgs_minibatch_cuda.c:  double step; /* FIXME tune for GPU, use larger if far away from convergence */
src/lib/Dirac/lbfgs_minibatch_cuda.c:  thread_gpu_data threaddata[2]; /* 2 for 2 threads/cards */
src/lib/Dirac/lbfgs_minibatch_cuda.c:/*********** following are not part of LBFGS, but done here only for GPU use */
src/lib/Dirac/lbfgs_minibatch_cuda.c:  /* auxilliary arrays for GPU */
src/lib/Dirac/lbfgs_minibatch_cuda.c:  int ThreadsPerBlock = gpu_threads;
src/lib/Dirac/lbfgs_minibatch_cuda.c:  /* also account for the no of GPUs using */
src/lib/Dirac/lbfgs_minibatch_cuda.c:  /* parameters per thread (GPU) */
src/lib/Dirac/lbfgs_minibatch_cuda.c:  /* calculate total size of memory need to be allocated in GPU, in bytes +2 added to align memory */
src/lib/Dirac/lbfgs_minibatch_cuda.c:  tpg.status[0]=tpg.status[1]=PT_DO_AGPU;
src/lib/Dirac/lbfgs_minibatch_cuda.c:   /* FIXME: update paramters for GPU gradient */
src/lib/Dirac/lbfgs_minibatch_cuda.c:  tpg.status[0]=tpg.status[1]=PT_DO_DGPU;
src/lib/Dirac/lbfgs_minibatch_cuda.c:/* initialize persistant memory (allocated on the GPU) */
src/lib/Dirac/lbfgs_minibatch_cuda.c:/* also attach to a GPU first */
src/lib/Dirac/lbfgs_minibatch_cuda.c:    cudaError_t err;
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(pt->s),m*lbfgs_m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemset(pt->s,0,m*lbfgs_m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(pt->y),m*lbfgs_m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemset(pt->y,0,m*lbfgs_m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(pt->running_avg),m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemset(pt->running_avg,0,m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(pt->running_avg_sq),m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemset(pt->running_avg_sq,0,m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    cudaError_t err;
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaFree(pt->s);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaFree(pt->y);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaFree(pt->running_avg);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaFree(pt->running_avg_sq);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    cudaError_t err;
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemset(pt->s,0,pt->m*pt->lbfgs_m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemset(pt->y,0,pt->m*pt->lbfgs_m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemset(pt->running_avg,0,pt->m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemset(pt->running_avg_sq,0,pt->m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:cuda_linesearch_backtrack(
src/lib/Dirac/lbfgs_minibatch_cuda.c:    cudaError_t err;
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(xk1),m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemset(xk1,0,m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(xk1, xk, m*sizeof(double), cudaMemcpyDeviceToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(xk1, xk, m*sizeof(double), cudaMemcpyDeviceToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:     err=cudaFree(xk1);
src/lib/Dirac/lbfgs_minibatch_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c: * both p and g should be device pointers, use cudaPointerAttributes() to check
src/lib/Dirac/lbfgs_minibatch_cuda.c:lbfgs_fit_cuda(
src/lib/Dirac/lbfgs_minibatch_cuda.c:    cudaError_t err;
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(gk),m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemset(gk,0,m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(xk1),m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemset(xk1,0,m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(xk),m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemset(xk,0,m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(pk),m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemset(pk,0,m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:  struct cudaPointerAttributes attributes;
src/lib/Dirac/lbfgs_minibatch_cuda.c:  err=cudaPointerGetAttributes(&attributes,(void*)p);
src/lib/Dirac/lbfgs_minibatch_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(xk, p, m*sizeof(double), cudaMemcpyDeviceToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(xk, p, m*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(g_min_rold),m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMalloc((void**)&(g_min_rnew),m*sizeof(double));
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(g_min_rold, gk, m*sizeof(double), cudaMemcpyDeviceToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:     err=cudaMemcpy(g_min_rnew, gk, m*sizeof(double), cudaMemcpyDeviceToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:     cudakernel_hadamard_sum(ThreadsPerBlock,(m+ThreadsPerBlock-1)/ThreadsPerBlock,m,indata->running_avg_sq,g_min_rold,g_min_rnew);
src/lib/Dirac/lbfgs_minibatch_cuda.c:     err=cudaFree(g_min_rold);
src/lib/Dirac/lbfgs_minibatch_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:     err=cudaFree(g_min_rnew);
src/lib/Dirac/lbfgs_minibatch_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    cuda_mult_hessian(m,pk,gk,s,y,rho,indata->cbhandle,indata->nfilled,ci);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    cuda_mult_hessian(m,pk,gk,s,y,rho,indata->cbhandle,M,ci);
src/lib/Dirac/lbfgs_minibatch_cuda.c:   alphak=cuda_linesearch_backtrack(cost_func,xk,pk,gk,m,indata->cbhandle,alphabar,adata);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(xk1, xk, m*sizeof(double), cudaMemcpyDeviceToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(&s[cm], xk1, m*sizeof(double), cudaMemcpyDeviceToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(&y[cm], gk, m*sizeof(double), cudaMemcpyDeviceToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(xk, xk1, m*sizeof(double), cudaMemcpyDeviceToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(p, xk, m*sizeof(double), cudaMemcpyDeviceToDevice);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaMemcpy(p, xk, m*sizeof(double), cudaMemcpyDeviceToHost);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaFree(gk);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaFree(xk1);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaFree(xk);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    err=cudaFree(pk);
src/lib/Dirac/lbfgs_minibatch_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs.c:#ifdef HAVE_CUDA
src/lib/Dirac/robust_batchmode_lbfgs.c:#include <cuda_runtime.h>
src/lib/Dirac/robust_batchmode_lbfgs.c:   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads,
src/lib/Dirac/robust_batchmode_lbfgs.c:   int Nbase, int tilesz, baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *indata,int nminibatch, int totalminibatch) {
src/lib/Dirac/robust_batchmode_lbfgs.c:   int Nbase, int tilesz, baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, double *y, double *z, double *rho, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *indata,int nminibatch, int totalminibatch) {
src/lib/Dirac/Dirac_GPUtune.h:#ifndef DIRAC_GPUTUNE_H
src/lib/Dirac/Dirac_GPUtune.h:#define DIRAC_GPUTUNE_H
src/lib/Dirac/Dirac_GPUtune.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac_GPUtune.h:/* include tunable parameters of GPU version here */
src/lib/Dirac/Dirac_GPUtune.h:#ifndef MAX_GPU_ID
src/lib/Dirac/Dirac_GPUtune.h:#define MAX_GPU_ID 3 /* use 0 (1 GPU), 1 (2 GPUs), ... */
src/lib/Dirac/Dirac_GPUtune.h:/* default GPU heap size (in MB) needed to calculate some shapelet models,
src/lib/Dirac/Dirac_GPUtune.h:   the default GPU values is ~ 8MB */
src/lib/Dirac/Dirac_GPUtune.h:#ifndef GPU_HEAP_SIZE
src/lib/Dirac/Dirac_GPUtune.h:#define GPU_HEAP_SIZE 32
src/lib/Dirac/Dirac_GPUtune.h:#endif /* DIRAC_GPUTUNE_H */
src/lib/Dirac/rtr_solve_robust_cuda.c:#include <cuda_runtime.h>
src/lib/Dirac/rtr_solve_robust_cuda.c:checkCudaError(cudaError_t err, char *file, int line)
src/lib/Dirac/rtr_solve_robust_cuda.c:#ifdef CUDA_DEBUG
src/lib/Dirac/rtr_solve_robust_cuda.c:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Dirac/rtr_solve_robust_cuda.c:#ifdef CUDA_DEBUG
src/lib/Dirac/rtr_solve_robust_cuda.c:cudakernel_fns_fgrad_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *iw, float *wtd, int negate, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&tempeta, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void **)&Ad, 16*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void **)&Bd, 16*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMemcpy(Ad,A,16*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMemcpy(Bd,B,16*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaError_t err;
src/lib/Dirac/rtr_solve_robust_cuda.c:  err=cudaMalloc((void**)&devInfo, sizeof(int)); /* FIXME: get too many errors here */
src/lib/Dirac/rtr_solve_robust_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c:  err=cudaMalloc((void**)&work, work_size*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_robust_cuda.c:  err=cudaMalloc((void**)&taud, 4*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_robust_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaFree(work);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaFree(taud);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaFree(devInfo);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(Ad);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMemcpy(B,Bd,16*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudakernel_fns_fgradflat_robust(ThreadsPerBlock, Bt*ntime, N, M, x, tempeta, y, coh, bbh, wtd, Bd, cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudakernel_fns_fscale(N, tempeta, iw);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMemcpy(eta,tempeta,4*N*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(Bd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(tempeta);
src/lib/Dirac/rtr_solve_robust_cuda.c:cudakernel_fns_fhess_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *iw, float *wtd, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&tempeta, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void **)&Ad, 16*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void **)&Bd, 16*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMemcpy(Ad,A,16*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMemcpy(Bd,B,16*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaError_t err;
src/lib/Dirac/rtr_solve_robust_cuda.c:  err=cudaMalloc((void**)&devInfo, sizeof(int));
src/lib/Dirac/rtr_solve_robust_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c:  err=cudaMalloc((void**)&work, work_size*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_robust_cuda.c:  err=cudaMalloc((void**)&taud, 4*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_robust_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaFree(work);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaFree(taud);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaFree(devInfo);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(Ad);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMemcpy(B,Bd,16*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudakernel_fns_fhessflat_robust(ThreadsPerBlock, Bt*ntime, N, M, x, eta, tempeta, y, coh, bbh, wtd, Bd, cbhandle, solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudakernel_fns_fscale(N, tempeta, iw);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMemcpy(fhess,tempeta,4*N*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(Bd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(tempeta);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&s, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&x_prop, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: f0=cudakernel_fns_f_robust(ThreadsPerBlock,BlocksPerGrid,N,M,x,y,coh,bbh,wtd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudakernel_fns_fgrad_robust(ThreadsPerBlock,BlocksPerGrid,N,M,x,eta,y,coh,bbh,iw,wtd,1,cbhandle, solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudakernel_fns_fhess_robust(ThreadsPerBlock,BlocksPerGrid,N,M,x,s,Heta,y,coh,bbh,iw,wtd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:   mk=f0-cudakernel_fns_g(N,x_prop,eta,s,cbhandle,solver_handle)-0.5f*cudakernel_fns_g(N,x_prop,Heta,s,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:   fk=cudakernel_fns_f_robust(ThreadsPerBlock,BlocksPerGrid,N,M,x_prop,y,coh,bbh,wtd);
src/lib/Dirac/rtr_solve_robust_cuda.c:     float g0_s=cudakernel_fns_g(N,x,eta,s,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(s);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(x_prop);
src/lib/Dirac/rtr_solve_robust_cuda.c:tcg_solve_cuda(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *grad, cuFloatComplex *eta, cuFloatComplex *fhess, float Delta, float theta, float kappa, int max_inner, int min_inner, float *y, float *coh, short *bbh, float *iw, float *wtd, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) { 
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaMalloc((void**)&r, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaMalloc((void**)&z, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaMalloc((void**)&delta, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaMalloc((void**)&Hxd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaMalloc((void**)&rnew, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c:  r_r=cudakernel_fns_g(N,x,r,r,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:  z_r=cudakernel_fns_g(N,x,z,r,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaMemset(delta, 0, sizeof(cuFloatComplex)*4*N); 
src/lib/Dirac/rtr_solve_robust_cuda.c:  e_Pd=cudakernel_fns_g(N,x,eta,delta,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:    cudakernel_fns_fhess_robust(ThreadsPerBlock,BlocksPerGrid,N,M,x,delta,Hxd,y,coh,bbh,iw,wtd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:    d_Hd=cudakernel_fns_g(N,x,delta,Hxd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:    cudakernel_fns_proj(N, x, r, rnew, cbhandle, solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:    r_r=cudakernel_fns_g(N,x,r,r,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:    z_r=cudakernel_fns_g(N,x,z,r,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaFree(r);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaFree(z);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaFree(delta);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaFree(Hxd);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaFree(rnew);
src/lib/Dirac/rtr_solve_robust_cuda.c:rtr_solve_cuda_robust_fl(
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaError_t err;
src/lib/Dirac/rtr_solve_robust_cuda.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&fgradxd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&etad, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&Hetad, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&x_propd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&xd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&yd, sizeof(float)*8*M);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&cohd, sizeof(float)*8*Nbase);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&bbd, sizeof(short)*2*Nbase);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&iwd, sizeof(float)*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&wtd, sizeof(float)*M);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&qd, sizeof(float)*M);
src/lib/Dirac/rtr_solve_robust_cuda.c: err=cudaMemcpy(yd, y, 8*M*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c: err=cudaMemcpy(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c: err=cudaMemcpy(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c: err=cudaMemcpy(xd, x, 8*N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c: err=cudaMemcpy(iwd, iw, N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c: /* set initial weights to 1 by a cuda kernel */
src/lib/Dirac/rtr_solve_robust_cuda.c: cudakernel_setweights_fl(ThreadsPerBlock, (M+ThreadsPerBlock-1)/ThreadsPerBlock, M, wtd, 1.0f);
src/lib/Dirac/rtr_solve_robust_cuda.c: fx=cudakernel_fns_f_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudakernel_fns_fupdate_weights(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd,robust_nu);
src/lib/Dirac/rtr_solve_robust_cuda.c:   cudakernel_fns_fgrad_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,fgradxd,yd,cohd,bbd,iwd,wtd,1,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:   norm_grad=sqrtf(cudakernel_fns_g(N,xd,fgradxd,fgradxd,cbhandle,solver_handle));
src/lib/Dirac/rtr_solve_robust_cuda.c:    cudaMemset(etad, 0, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c:    stop_inner=tcg_solve_cuda(ThreadsPerBlock,BlocksPerGrid, N, M, xd, fgradxd, etad, Hetad, Delta, theta, kappa, max_inner, min_inner,yd,cohd,bbd,iwd,wtd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:   cudakernel_fns_R(N,xd,etad,x_propd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:    fx_prop=cudakernel_fns_f_robust(ThreadsPerBlock,BlocksPerGrid,N,M,x_propd,yd,cohd,bbd,wtd);
src/lib/Dirac/rtr_solve_robust_cuda.c:    rhoden=-cudakernel_fns_g(N,xd,fgradxd,etad,cbhandle,solver_handle)-0.5f*cudakernel_fns_g(N,xd,Hetad,etad,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:     cudakernel_fns_fgrad_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,fgradxd,yd,cohd,bbd,iwd,wtd,1,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:     norm_grad=sqrtf(cudakernel_fns_g(N,xd,fgradxd,fgradxd,cbhandle,solver_handle));
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaDeviceSynchronize();
src/lib/Dirac/rtr_solve_robust_cuda.c:   cudakernel_fns_fupdate_weights_q(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd,qd,robust_nu);
src/lib/Dirac/rtr_solve_robust_cuda.c:   cudakernel_evaluatenu_fl_eight(ThreadsPerBlock2, (Nd+ThreadsPerBlock-1)/ThreadsPerBlock2, Nd, q_sum, qd, deltanu,(float)robust_nulow,robust_nu);
src/lib/Dirac/rtr_solve_robust_cuda.c:  err=cudaMemcpy(x,xd,8*N*sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/rtr_solve_robust_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(fgradxd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(etad);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(Hetad);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(x_propd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(xd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(yd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(cohd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(bbd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(iwd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(wtd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(qd);
src/lib/Dirac/rtr_solve_robust_cuda.c:nsd_solve_cuda_robust_fl(
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudaError_t err;
src/lib/Dirac/rtr_solve_robust_cuda.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&fgradxd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&etad, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&zd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&x_propd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&xd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&z_propd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&yd, sizeof(float)*8*M);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&cohd, sizeof(float)*8*Nbase);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&bbd, sizeof(short)*2*Nbase);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&iwd, sizeof(float)*N);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&wtd, sizeof(float)*M);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaMalloc((void**)&qd, sizeof(float)*M);
src/lib/Dirac/rtr_solve_robust_cuda.c: err=cudaMemcpy(yd, y, 8*M*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c: err=cudaMemcpy(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c: err=cudaMemcpy(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c: err=cudaMemcpy(xd, x, 8*N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c: err=cudaMemcpy(iwd, iw, N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c: /* set initial weights to 1 by a cuda kernel */
src/lib/Dirac/rtr_solve_robust_cuda.c: cudakernel_setweights_fl(ThreadsPerBlock, (M+ThreadsPerBlock-1)/ThreadsPerBlock, M, wtd, 1.0f);
src/lib/Dirac/rtr_solve_robust_cuda.c: fx=cudakernel_fns_f_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudakernel_fns_fgrad_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,fgradxd,yd,cohd,bbd,iwd,wtd,1,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:  cudakernel_fns_fhess_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,xd,zd,yd,cohd,bbd,iwd,wtd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:    cudakernel_fns_fgrad_robust(ThreadsPerBlock,BlocksPerGrid,N,M,zd,fgradxd,yd,cohd,bbd,iwd,wtd,1,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_cuda.c:  fx=cudakernel_fns_f_robust(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaDeviceSynchronize();
src/lib/Dirac/rtr_solve_robust_cuda.c:   cudakernel_fns_fupdate_weights_q(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd,qd,robust_nu);
src/lib/Dirac/rtr_solve_robust_cuda.c:   cudakernel_evaluatenu_fl_eight(ThreadsPerBlock2, (Nd+ThreadsPerBlock-1)/ThreadsPerBlock2, Nd, q_sum, qd, deltanu,(float)robust_nulow,robust_nu);
src/lib/Dirac/rtr_solve_robust_cuda.c:  err=cudaMemcpy(x,xd,8*N*sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/rtr_solve_robust_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(fgradxd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(etad);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(zd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(x_propd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(xd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(z_propd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(yd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(cohd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(bbd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(iwd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(wtd);
src/lib/Dirac/rtr_solve_robust_cuda.c: cudaFree(qd);
src/lib/Dirac/robust.cu:#include "cuda.h"
src/lib/Dirac/robust.cu://#define CUDA_DBG
src/lib/Dirac/robust.cu:/* set initial weights to 1 by a cuda kernel */
src/lib/Dirac/robust.cu:cudakernel_setweights(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt, double alpha) {
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  cudaError_t error;
src/lib/Dirac/robust.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  error = cudaGetLastError();
src/lib/Dirac/robust.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust.cu:/* hadamard product by a cuda kernel x<= x*wt */
src/lib/Dirac/robust.cu:cudakernel_hadamard(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt, double *x) {
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  cudaError_t error;
src/lib/Dirac/robust.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  error = cudaGetLastError();
src/lib/Dirac/robust.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust.cu:/* sum hadamard product by a cuda kernel y=y+x.*w (x.*w elementwise) */
src/lib/Dirac/robust.cu:cudakernel_hadamard_sum(int ThreadsPerBlock, int BlocksPerGrid, int N, double *y, double *x, double *w) {
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  cudaError_t error;
src/lib/Dirac/robust.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  error = cudaGetLastError();
src/lib/Dirac/robust.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust.cu:/* update weights by a cuda kernel */
src/lib/Dirac/robust.cu:cudakernel_updateweights(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt, double *x, double *q, double robust_nu) {
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  cudaError_t error;
src/lib/Dirac/robust.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  error = cudaGetLastError();
src/lib/Dirac/robust.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust.cu:/* update weights by a cuda kernel */
src/lib/Dirac/robust.cu:cudakernel_sqrtweights(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt) {
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  cudaError_t error;
src/lib/Dirac/robust.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  error = cudaGetLastError();
src/lib/Dirac/robust.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust.cu:cudakernel_evaluatenu(int ThreadsPerBlock, int BlocksPerGrid, int Nd, double qsum, double *q, double deltanu,double nulow) {
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  cudaError_t error;
src/lib/Dirac/robust.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  error = cudaGetLastError();
src/lib/Dirac/robust.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust.cu:/* cuda driver for calculating wt \odot f() */
src/lib/Dirac/robust.cu:cudakernel_func_wt(int ThreadsPerBlock, int BlocksPerGrid, double *p, double *x, int M, int N, double *coh, short *bbh, double *wt, int Nbase, int Mclus, int Nstations) {
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  cudaError_t error;
src/lib/Dirac/robust.cu:  cudaMemset(x, 0, N*sizeof(double));
src/lib/Dirac/robust.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  error = cudaGetLastError();
src/lib/Dirac/robust.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust.cu:/* cuda driver for calculating wt \odot jacf() */
src/lib/Dirac/robust.cu:cudakernel_jacf_wt(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, double *p, double *jac, int M, int N, double *coh, short *bbh, double *wt, int Nbase, int Mclus, int Nstations, int clus) {
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  cudaError_t error;
src/lib/Dirac/robust.cu:  cudaMemset(jac, 0, N*M*sizeof(double));
src/lib/Dirac/robust.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  error = cudaGetLastError();
src/lib/Dirac/robust.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust.cu:/* cuda driver for kernel */
src/lib/Dirac/robust.cu:void cudakernel_lbfgs_robust(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, double robust_nu, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad){
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  cudaError_t error;
src/lib/Dirac/robust.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust.cu:  error = cudaGetLastError();
src/lib/Dirac/robust.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_cuda.c:#include <cuda_runtime.h>
src/lib/Dirac/rtr_solve_cuda.c:checkCudaError(cudaError_t err, char *file, int line)
src/lib/Dirac/rtr_solve_cuda.c:#ifdef CUDA_DEBUG
src/lib/Dirac/rtr_solve_cuda.c:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Dirac/rtr_solve_cuda.c:#ifdef CUDA_DEBUG
src/lib/Dirac/rtr_solve_cuda.c:cudakernel_fns_R(int N, cuFloatComplex *x, cuFloatComplex *r, cuFloatComplex *rnew, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
src/lib/Dirac/rtr_solve_cuda.c:cudakernel_fns_g(int N,cuFloatComplex *x,cuFloatComplex *eta, cuFloatComplex *gamma,cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
src/lib/Dirac/rtr_solve_cuda.c:cudakernel_fns_proj(int N, cuFloatComplex *x, cuFloatComplex *z, cuFloatComplex *rnew, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
src/lib/Dirac/rtr_solve_cuda.c:  cudaMalloc((void **)&Ad, 16*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_cuda.c:  cudaMemcpy(Ad,A,16*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_cuda.c:  cudaMalloc((void **)&bd, 4*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_cuda.c:  cudaMemcpy(bd,b,4*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_cuda.c:  cudaError_t err;
src/lib/Dirac/rtr_solve_cuda.c:  err=cudaMalloc((void**)&devInfo, sizeof(int));
src/lib/Dirac/rtr_solve_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_cuda.c:  err=cudaMalloc((void**)&work, work_size*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_cuda.c:  err=cudaMalloc((void**)&taud, 4*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_cuda.c:  cudaFree(work); 
src/lib/Dirac/rtr_solve_cuda.c:  cudaFree(taud); 
src/lib/Dirac/rtr_solve_cuda.c:  cudaFree(devInfo); 
src/lib/Dirac/rtr_solve_cuda.c:  cudaMemcpy(b,bd,4*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost);
src/lib/Dirac/rtr_solve_cuda.c: cudaHostAlloc((void **)&etalocal, sizeof(cuFloatComplex)*4*N,cudaHostAllocDefault);
src/lib/Dirac/rtr_solve_cuda.c: cudaMemcpy(etalocal, rnew, 4*N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
src/lib/Dirac/rtr_solve_cuda.c: cudaMemcpy(etalocal, rnew, 4*N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
src/lib/Dirac/rtr_solve_cuda.c: cudaFreeHost(etalocal);
src/lib/Dirac/rtr_solve_cuda.c:  cudaFree(Ad); 
src/lib/Dirac/rtr_solve_cuda.c:  cudaFree(bd); 
src/lib/Dirac/rtr_solve_cuda.c:cudakernel_fns_fgrad(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *iw, int negate, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&tempeta, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&tempb, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c:  cudakernel_fns_fgradflat(ThreadsPerBlock, BlocksPerGrid, N, M, x, tempeta, y, coh, bbh);
src/lib/Dirac/rtr_solve_cuda.c:   cudaMemset(tempeta, 0, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c:    cudakernel_fns_fgradflat(ThreadsPerBlock, B, N, myT, x, tempb, &y[ct*8], &coh[ct*8], &bbh[ct*2]);
src/lib/Dirac/rtr_solve_cuda.c: cudakernel_fns_fscale(N, tempeta, iw);
src/lib/Dirac/rtr_solve_cuda.c: cudakernel_fns_proj(N, x, tempeta, eta, cbhandle, solver_handle);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(tempeta);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(tempb);
src/lib/Dirac/rtr_solve_cuda.c:cudakernel_fns_fhess(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *iw, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&tempeta, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&tempb, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c:  cudakernel_fns_fhessflat(ThreadsPerBlock, BlocksPerGrid, N, M, x, eta, tempeta, y, coh, bbh);
src/lib/Dirac/rtr_solve_cuda.c:   cudaMemset(tempeta, 0, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c:    cudakernel_fns_fhessflat(ThreadsPerBlock, B, N, myT, x, eta, tempb, &y[ct*8], &coh[ct*8], &bbh[ct*2]);
src/lib/Dirac/rtr_solve_cuda.c: cudakernel_fns_fscale(N, tempeta, iw);
src/lib/Dirac/rtr_solve_cuda.c: cudakernel_fns_proj(N, x, tempeta, fhess, cbhandle, solver_handle);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(tempeta);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(tempb);
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&eta, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&x_prop, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c: float fx=cudakernel_fns_f(ThreadsPerBlock,BlocksPerGrid,N,M,x,y,coh,bbh);
src/lib/Dirac/rtr_solve_cuda.c: cudakernel_fns_fgrad(ThreadsPerBlock,BlocksPerGrid,N,M,x,eta,y,coh,bbh,iw,0,cbhandle, solver_handle);
src/lib/Dirac/rtr_solve_cuda.c: cudaHostAlloc((void **)&etalocal, sizeof(cuFloatComplex)*4*N,cudaHostAllocDefault);
src/lib/Dirac/rtr_solve_cuda.c: cudaMemcpy(etalocal, eta, 4*N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
src/lib/Dirac/rtr_solve_cuda.c: cudaFreeHost(etalocal);
src/lib/Dirac/rtr_solve_cuda.c: float metric0=cudakernel_fns_g(N,x,eta,eta,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:   cudakernel_fns_R(N,x,teta,x_prop,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:   lhs=cudakernel_fns_f(ThreadsPerBlock,BlocksPerGrid,N,M,x_prop,y,coh,bbh);
src/lib/Dirac/rtr_solve_cuda.c:   //metric=cudakernel_fns_g(N,x,eta,teta,cbhandle);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(eta);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(x_prop);
src/lib/Dirac/rtr_solve_cuda.c:tcg_solve_cuda(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *grad, cuFloatComplex *eta, cuFloatComplex *fhess, float Delta, float theta, float kappa, int max_inner, int min_inner, float *y, float *coh, short *bbh, float *iw, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) { 
src/lib/Dirac/rtr_solve_cuda.c:  cudaMalloc((void**)&r, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c:  cudaMalloc((void**)&z, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c:  cudaMalloc((void**)&delta, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c:  cudaMalloc((void**)&Hxd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c:  cudaMalloc((void**)&rnew, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c:  r_r=cudakernel_fns_g(N,x,r,r,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:  z_r=cudakernel_fns_g(N,x,z,r,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:  cudaMemset(delta, 0, sizeof(cuFloatComplex)*4*N); 
src/lib/Dirac/rtr_solve_cuda.c:  e_Pd=cudakernel_fns_g(N,x,eta,delta,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:    cudakernel_fns_fhess(ThreadsPerBlock,BlocksPerGrid,N,M,x,delta,Hxd,y,coh,bbh,iw, cbhandle, solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:    d_Hd=cudakernel_fns_g(N,x,delta,Hxd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:    cudakernel_fns_proj(N, x, r, rnew, cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:    r_r=cudakernel_fns_g(N,x,r,r,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:    z_r=cudakernel_fns_g(N,x,z,r,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:  cudaFree(r);
src/lib/Dirac/rtr_solve_cuda.c:  cudaFree(z);
src/lib/Dirac/rtr_solve_cuda.c:  cudaFree(delta);
src/lib/Dirac/rtr_solve_cuda.c:  cudaFree(Hxd);
src/lib/Dirac/rtr_solve_cuda.c:  cudaFree(rnew);
src/lib/Dirac/rtr_solve_cuda.c:rtr_solve_cuda_fl(
src/lib/Dirac/rtr_solve_cuda.c:  cudaError_t err;
src/lib/Dirac/rtr_solve_cuda.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&fgradxd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&etad, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&Hetad, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&x_propd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&xd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&yd, sizeof(float)*8*M);
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&cohd, sizeof(float)*8*Nbase);
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&bbd, sizeof(short)*2*Nbase);
src/lib/Dirac/rtr_solve_cuda.c: cudaMalloc((void**)&iwd, sizeof(float)*N);
src/lib/Dirac/rtr_solve_cuda.c: err=cudaMemcpy(yd, y, 8*M*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_cuda.c: err=cudaMemcpy(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_cuda.c: err=cudaMemcpy(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_cuda.c: err=cudaMemcpy(xd, x, 8*N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_cuda.c: err=cudaMemcpy(iwd, iw, N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_cuda.c: fx=cudakernel_fns_f(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd);
src/lib/Dirac/rtr_solve_cuda.c:  cudakernel_fns_R(N,xd,etad,x_propd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:   cudakernel_fns_fgrad(ThreadsPerBlock,BlocksPerGrid,N,M,xd,fgradxd,yd,cohd,bbd,iwd,1,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:   norm_grad=sqrtf(cudakernel_fns_g(N,xd,fgradxd,fgradxd,cbhandle,solver_handle));
src/lib/Dirac/rtr_solve_cuda.c:    cudaMemset(etad, 0, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_cuda.c:    stop_inner=tcg_solve_cuda(ThreadsPerBlock,BlocksPerGrid, N, M, xd, fgradxd, etad, Hetad, Delta, theta, kappa, max_inner, min_inner,yd,cohd,bbd,iwd,cbhandle, solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:   cudakernel_fns_R(N,xd,etad,x_propd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:    fx_prop=cudakernel_fns_f(ThreadsPerBlock,BlocksPerGrid,N,M,x_propd,yd,cohd,bbd);
src/lib/Dirac/rtr_solve_cuda.c:    rhoden=-cudakernel_fns_g(N,xd,fgradxd,etad,cbhandle,solver_handle)-0.5f*cudakernel_fns_g(N,xd,Hetad,etad,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:     cudakernel_fns_fgrad(ThreadsPerBlock,BlocksPerGrid,N,M,xd,fgradxd,yd,cohd,bbd,iwd,1,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_cuda.c:     norm_grad=sqrtf(cudakernel_fns_g(N,xd,fgradxd,fgradxd,cbhandle,solver_handle));
src/lib/Dirac/rtr_solve_cuda.c: cudaDeviceSynchronize();
src/lib/Dirac/rtr_solve_cuda.c:  err=cudaMemcpy(x,xd,8*N*sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/rtr_solve_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(fgradxd);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(etad);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(Hetad);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(x_propd);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(xd);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(yd);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(cohd);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(bbd);
src/lib/Dirac/rtr_solve_cuda.c: cudaFree(iwd);
src/lib/Dirac/lbfgs_cuda.c:#include <cuda.h>
src/lib/Dirac/lbfgs_cuda.c:#include <cuda_runtime_api.h>
src/lib/Dirac/lbfgs_cuda.c:#include <cuda_runtime.h>
src/lib/Dirac/lbfgs_cuda.c:checkCudaError(cudaError_t err, char *file, int line)
src/lib/Dirac/lbfgs_cuda.c:#ifdef CUDA_DEBUG
src/lib/Dirac/lbfgs_cuda.c:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Dirac/lbfgs_cuda.c:              1: allocate GPU  memory, attach GPU
src/lib/Dirac/lbfgs_cuda.c:              2: free GPU memory, detach GPU 
src/lib/Dirac/lbfgs_cuda.c:              3,4..: do work on GPU 
src/lib/Dirac/lbfgs_cuda.c:              99: reset GPU memory (memest all memory) */
src/lib/Dirac/lbfgs_cuda.c:  thread_gpu_data *lmdata[2]; /* two for each thread */
src/lib/Dirac/lbfgs_cuda.c:  /* GPU related info */
src/lib/Dirac/lbfgs_cuda.c:  double *gWORK[2]; /* GPU buffers */
src/lib/Dirac/lbfgs_cuda.c:  /* different pointers to GPU data */
src/lib/Dirac/lbfgs_cuda.c:/* slave thread 2GPU function */
src/lib/Dirac/lbfgs_cuda.c: cudaError_t err;
src/lib/Dirac/lbfgs_cuda.c:    err=cudaMemcpy(dp->cpp[tid], dp->lmdata[tid]->p, m*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/lbfgs_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_cuda.c:     cudakernel_lbfgs_r(dp->lmdata[tid]->ThreadsPerBlock, dp->lmdata[tid]->BlocksPerGrid, Nbase, dp->lmdata[tid]->tilesz, M, N, Nparam, dp->lmdata[tid]->g_start, dp->cxo[tid], dp->ccoh[tid], dp->cpp[tid], dp->cbb[tid], dp->cptoclus[tid], dp->cgrad[tid]);
src/lib/Dirac/lbfgs_cuda.c:     cudakernel_lbfgs_r_robust(dp->lmdata[tid]->ThreadsPerBlock, dp->lmdata[tid]->BlocksPerGrid, Nbase, dp->lmdata[tid]->tilesz, M, N, Nparam, dp->lmdata[tid]->g_start, dp->cxo[tid], dp->ccoh[tid], dp->cpp[tid], dp->cbb[tid], dp->cptoclus[tid], dp->cgrad[tid],dp->lmdata[tid]->robust_nu);
src/lib/Dirac/lbfgs_cuda.c:    err=cudaMemcpy(&(dp->lmdata[tid]->g[dp->lmdata[tid]->g_start]), dp->cgrad[tid], Nparam*sizeof(double), cudaMemcpyDeviceToHost);
src/lib/Dirac/lbfgs_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_cuda.c:   err=cudaMemcpy(dp->cpp[tid], dp->lmdata[tid]->p, m*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/lbfgs_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_cuda.c:    dp->fcost[tid]=cudakernel_lbfgs_cost(dp->lmdata[tid]->ThreadsPerBlock, BlocksPerGrid, dp->Nbase[tid], boff, M, N, Nbase, &dp->cxo[tid][8*boff], &dp->ccoh[tid][boff*8*M], dp->cpp[tid], &dp->cbb[tid][boff*2], dp->cptoclus[tid]); 
src/lib/Dirac/lbfgs_cuda.c:    dp->fcost[tid]=cudakernel_lbfgs_cost_robust(dp->lmdata[tid]->ThreadsPerBlock, BlocksPerGrid, dp->Nbase[tid], boff, M, N, Nbase, &dp->cxo[tid][8*boff], &dp->ccoh[tid][boff*8*M], dp->cpp[tid], &dp->cbb[tid][boff*2], dp->cptoclus[tid], dp->lmdata[tid]->robust_nu);
src/lib/Dirac/lbfgs_cuda.c:  } else if (dp->status[tid]==PT_DO_AGPU) {
src/lib/Dirac/lbfgs_cuda.c:    attach_gpu_to_thread1(select_work_gpu(MAX_GPU_ID,td->pline->thst),&dp->cbhandle[tid],&dp->solver_handle[tid],&dp->gWORK[tid],dp->data_size[tid]);
src/lib/Dirac/lbfgs_cuda.c:    err=cudaMalloc((void**)&(dp->cxo[tid]),dp->lmdata[tid]->n*sizeof(double));
src/lib/Dirac/lbfgs_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_cuda.c:    err=cudaMalloc((void**)&(dp->ccoh[tid]),Nbase*8*M*sizeof(double));
src/lib/Dirac/lbfgs_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_cuda.c:    err=cudaMalloc((void**)&(dp->cpp[tid]),m*sizeof(double));
src/lib/Dirac/lbfgs_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_cuda.c:    err=cudaMalloc((void**)&(dp->cgrad[tid]),Nparam*sizeof(double));
src/lib/Dirac/lbfgs_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_cuda.c:    err=cudaMalloc((void**)&(dp->cptoclus[tid]),M*2*sizeof(int));
src/lib/Dirac/lbfgs_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_cuda.c:    err=cudaMalloc((void**)&(dp->cbb[tid]),Nbase*2*sizeof(short));
src/lib/Dirac/lbfgs_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_cuda.c:    err=cudaMemcpy(dp->cxo[tid], dp->lmdata[tid]->xo, dp->lmdata[tid]->n*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/lbfgs_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_cuda.c:    err=cudaMemcpy(dp->ccoh[tid], dp->lmdata[tid]->coh, Nbase*8*M*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/lbfgs_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_cuda.c:    err=cudaMemcpy(dp->cptoclus[tid], dp->lmdata[tid]->ptoclus, M*2*sizeof(int), cudaMemcpyHostToDevice);
src/lib/Dirac/lbfgs_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_cuda.c:    err=cudaMemcpy(dp->cbb[tid], dp->lmdata[tid]->hbb, Nbase*2*sizeof(short), cudaMemcpyHostToDevice);
src/lib/Dirac/lbfgs_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_cuda.c:  } else if (dp->status[tid]==PT_DO_DGPU) {
src/lib/Dirac/lbfgs_cuda.c:    cudaFree(dp->cxo[tid]);
src/lib/Dirac/lbfgs_cuda.c:    cudaFree(dp->ccoh[tid]);
src/lib/Dirac/lbfgs_cuda.c:    cudaFree(dp->cptoclus[tid]);
src/lib/Dirac/lbfgs_cuda.c:    cudaFree(dp->cbb[tid]);
src/lib/Dirac/lbfgs_cuda.c:    cudaFree(dp->cpp[tid]);
src/lib/Dirac/lbfgs_cuda.c:    cudaFree(dp->cgrad[tid]);
src/lib/Dirac/lbfgs_cuda.c:    detach_gpu_from_thread1(dp->cbhandle[tid],dp->solver_handle[tid],dp->gWORK[tid]);
src/lib/Dirac/lbfgs_cuda.c://printf("GPU cost=%lf\n",phi_0);
src/lib/Dirac/lbfgs_cuda.c:  alphai1=0.0; /* FIXME: tune for GPU (defalut is 0.0) */
src/lib/Dirac/lbfgs_cuda.c:   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads, int do_robust, void *adata) {
src/lib/Dirac/lbfgs_cuda.c:  double step; /* FIXME tune for GPU, use larger if far away from convergence */
src/lib/Dirac/lbfgs_cuda.c:  thread_gpu_data threaddata[2]; /* 2 for 2 threads/cards */
src/lib/Dirac/lbfgs_cuda.c:/*********** following are not part of LBFGS, but done here only for GPU use */
src/lib/Dirac/lbfgs_cuda.c:  /* auxilliary arrays for GPU */
src/lib/Dirac/lbfgs_cuda.c:  int ThreadsPerBlock = gpu_threads;
src/lib/Dirac/lbfgs_cuda.c:  /* also account for the no of GPUs using */
src/lib/Dirac/lbfgs_cuda.c:  /* parameters per thread (GPU) */
src/lib/Dirac/lbfgs_cuda.c:  /* calculate total size of memory need to be allocated in GPU, in bytes +2 added to align memory */
src/lib/Dirac/lbfgs_cuda.c:  tpg.status[0]=tpg.status[1]=PT_DO_AGPU;
src/lib/Dirac/lbfgs_cuda.c:   /* FIXME: update paramters for GPU gradient */
src/lib/Dirac/lbfgs_cuda.c:  tpg.status[0]=tpg.status[1]=PT_DO_DGPU;
src/lib/Dirac/lbfgs_cuda.c:   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads, void *adata) {
src/lib/Dirac/lbfgs_cuda.c:  return lbfgs_fit_common(p, x, m, n, itmax, M, gpu_threads, 0, adata);
src/lib/Dirac/lbfgs_cuda.c:lbfgs_fit_robust_cuda(
src/lib/Dirac/lbfgs_cuda.c:   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads, void *adata) {
src/lib/Dirac/lbfgs_cuda.c:  return lbfgs_fit_common(p, x, m, n, itmax, M, gpu_threads, 1, adata);
src/lib/Dirac/robustlm.c:#ifdef HAVE_CUDA
src/lib/Dirac/robustlm.c:#include <cuda_runtime.h>
src/lib/Dirac/robustlm.c:checkCudaError(cudaError_t err, char *file, int line)
src/lib/Dirac/robustlm.c:#ifdef CUDA_DEBUG
src/lib/Dirac/robustlm.c:    printf("GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Dirac/robustlm.c:#ifdef CUDA_DEBUG
src/lib/Dirac/robustlm.c:  entirely in the GPU */
src/lib/Dirac/robustlm.c:rlevmar_der_single_cuda(
src/lib/Dirac/robustlm.c:  double *gWORK, /* GPU allocated memory */
src/lib/Dirac/robustlm.c:  cudaError_t err;
src/lib/Dirac/robustlm.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&xd, N*sizeof(double));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&jacd, M*N*sizeof(double));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&jacTjacd, M*M*sizeof(double));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&jacTed, M*sizeof(double));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&jacTjacd0, M*M*sizeof(double));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&Dpd, M*sizeof(double));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&bd, M*sizeof(double));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&pd, M*sizeof(double));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&pnewd, M*sizeof(double));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**) &bbd, Nbase*2*sizeof(short));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**) &cohd, Nbase*8*sizeof(double)); 
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&hxd, N*sizeof(double));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&wtd, N*sizeof(double));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&qd, N*sizeof(double));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&ed, N*sizeof(double));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&taud, M*sizeof(double));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&Ud, M*M*sizeof(double));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&VTd, M*M*sizeof(double));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&Sd, M*sizeof(double));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&devInfo, sizeof(int));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&rwork, 5*M*sizeof(double));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMemcpy(pd, p, M*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMemcpy(cohd, &(dp->ddcoh[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMemcpy(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:  err=cudaMemcpy(xd, x, N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  /* set initial weights to 1 by a cuda kernel */
src/lib/Dirac/robustlm.c:  cudakernel_setweights(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, 1.0);
src/lib/Dirac/robustlm.c:  cudakernel_func_wt(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,hxd,M,N, cohd, bbd, wtd, Nbase, dp->M, dp->N);
src/lib/Dirac/robustlm.c:  cudakernel_hadamard(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed);
src/lib/Dirac/robustlm.c:     cudakernel_jacf_wt(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, N, cohd, bbd, wtd, Nbase, dp->M, dp->N);
src/lib/Dirac/robustlm.c:     err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:      err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:      cudakernel_diagmu(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);
src/lib/Dirac/robustlm.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:        cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:         cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:        cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:        cudakernel_diagdiv(ThreadsPerBlock, BlocksPerGrid, M, eps1, Dpd, Sd);
src/lib/Dirac/robustlm.c:        cudakernel_func_wt(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, wtd, Nbase, dp->M, dp->N);
src/lib/Dirac/robustlm.c:        cudakernel_hadamard(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed);
src/lib/Dirac/robustlm.c:   cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/robustlm.c:   cudakernel_updateweights(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed, qd, robust_nu);
src/lib/Dirac/robustlm.c:   cudakernel_sqrtweights(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd);
src/lib/Dirac/robustlm.c:   cudakernel_evaluatenu(ThreadsPerBlock2, (Nd+ThreadsPerBlock2-1)/ThreadsPerBlock2, Nd, q_sum, qd, deltanu,robust_nulow);
src/lib/Dirac/robustlm.c:  err=cudaMemcpyAsync(p,pd,M*sizeof(double),cudaMemcpyDeviceToHost,0);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:  cudaFree(xd);
src/lib/Dirac/robustlm.c:  cudaFree(jacd);
src/lib/Dirac/robustlm.c:  cudaFree(jacTjacd);
src/lib/Dirac/robustlm.c:  cudaFree(jacTjacd0);
src/lib/Dirac/robustlm.c:  cudaFree(jacTed);
src/lib/Dirac/robustlm.c:  cudaFree(Dpd);
src/lib/Dirac/robustlm.c:  cudaFree(bd);
src/lib/Dirac/robustlm.c:  cudaFree(pd);
src/lib/Dirac/robustlm.c:  cudaFree(pnewd);
src/lib/Dirac/robustlm.c:  cudaFree(hxd);
src/lib/Dirac/robustlm.c:  cudaFree(wtd);
src/lib/Dirac/robustlm.c:  cudaFree(qd);
src/lib/Dirac/robustlm.c:  cudaFree(ed);
src/lib/Dirac/robustlm.c:   cudaFree(taud);
src/lib/Dirac/robustlm.c:   cudaFree(Ud);
src/lib/Dirac/robustlm.c:   cudaFree(VTd);
src/lib/Dirac/robustlm.c:   cudaFree(Sd);
src/lib/Dirac/robustlm.c:  cudaFree(cohd);
src/lib/Dirac/robustlm.c:  cudaFree(bbd);
src/lib/Dirac/robustlm.c:  cudaFree(devInfo);
src/lib/Dirac/robustlm.c:  cudaFree(work);
src/lib/Dirac/robustlm.c:    cudaFree(rwork);
src/lib/Dirac/robustlm.c:  entirely in the GPU, using float data */
src/lib/Dirac/robustlm.c:rlevmar_der_single_cuda_fl(
src/lib/Dirac/robustlm.c:  float *gWORK, /* GPU allocated memory */
src/lib/Dirac/robustlm.c:  cudaError_t err;
src/lib/Dirac/robustlm.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&devInfo, sizeof(int));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&work, work_size*sizeof(float));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&work, work_size*sizeof(float));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&work, work_size*sizeof(float));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&rwork, 5*M*sizeof(float));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMemcpyAsync(pd, p, M*sizeof(float), cudaMemcpyHostToDevice,0);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMemcpyAsync(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice,0);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMemcpyAsync(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice,0);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:  err=cudaMemcpyAsync(xd, x, N*sizeof(float), cudaMemcpyHostToDevice,0);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  /* set initial weights to 1 by a cuda kernel */
src/lib/Dirac/robustlm.c:  cudakernel_setweights_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, 1.0f);
src/lib/Dirac/robustlm.c:  cudakernel_func_wt_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,hxd,M,N, cohd, bbd, wtd, Nbase, dp->M, dp->N);
src/lib/Dirac/robustlm.c:  cudakernel_hadamard_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed);
src/lib/Dirac/robustlm.c:     cudakernel_jacf_wt_fl(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, N, cohd, bbd, wtd, Nbase, dp->M, dp->N);
src/lib/Dirac/robustlm.c:     err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:      err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:      cudakernel_diagmu_fl(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);
src/lib/Dirac/robustlm.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:        cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:         cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:        cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:        cudakernel_diagdiv_fl(ThreadsPerBlock, BlocksPerGrid, M, eps1, Dpd, Sd);
src/lib/Dirac/robustlm.c:        cudakernel_func_wt_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, wtd, Nbase, dp->M, dp->N);
src/lib/Dirac/robustlm.c:        cudakernel_hadamard_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed);
src/lib/Dirac/robustlm.c:   cudakernel_func_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/robustlm.c:   cudakernel_updateweights_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed, qd, robust_nu);
src/lib/Dirac/robustlm.c:   cudakernel_sqrtweights_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd);
src/lib/Dirac/robustlm.c:   cudakernel_evaluatenu_fl(ThreadsPerBlock2, (Nd+ThreadsPerBlock2-1)/ThreadsPerBlock2, Nd, q_sum, qd, deltanu,(float)robust_nulow);
src/lib/Dirac/robustlm.c:  err=cudaMemcpyAsync(p,pd,M*sizeof(float),cudaMemcpyDeviceToHost,0);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:  cudaFree(devInfo);
src/lib/Dirac/robustlm.c:  cudaFree(work);
src/lib/Dirac/robustlm.c:    cudaFree(rwork);
src/lib/Dirac/robustlm.c:  entirely in the GPU, using float data, OS acceleration */
src/lib/Dirac/robustlm.c:osrlevmar_der_single_cuda_fl(
src/lib/Dirac/robustlm.c:  float *gWORK, /* GPU allocated memory */
src/lib/Dirac/robustlm.c:  cudaError_t err;
src/lib/Dirac/robustlm.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/robustlm.c:  err=cudaMalloc((void**)&devInfo, sizeof(int));
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&work, work_size*sizeof(float));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&work, work_size*sizeof(float));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&work, work_size*sizeof(float));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:    err=cudaMalloc((void**)&rwork, 5*M*sizeof(float));
src/lib/Dirac/robustlm.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMemcpyAsync(pd, p, M*sizeof(float), cudaMemcpyHostToDevice,0);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMemcpyAsync(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice,0);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  err=cudaMemcpyAsync(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice,0);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:  err=cudaMemcpyAsync(xd, x, N*sizeof(float), cudaMemcpyHostToDevice,0);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  /* set initial weights to 1 by a cuda kernel */
src/lib/Dirac/robustlm.c:  cudakernel_setweights_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, 1.0f);
src/lib/Dirac/robustlm.c:  cudakernel_func_wt_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,hxd,M,N, cohd, bbd, wtd, Nbase, dp->M, dp->N);
src/lib/Dirac/robustlm.c:  cudakernel_hadamard_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed);
src/lib/Dirac/robustlm.c:     cudakernel_jacf_wt_fl(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, Nos[l], &cohd[8*NbI[l]], &bbd[2*NbI[l]], &wtd[edI[l]], Nbaseos[l], dp->M, dp->N);
src/lib/Dirac/robustlm.c:     err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:      err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:      cudakernel_diagmu_fl(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);
src/lib/Dirac/robustlm.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:        cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:         cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/robustlm.c:        cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:        cudakernel_diagdiv_fl(ThreadsPerBlock, BlocksPerGrid, M, eps1, Dpd, Sd);
src/lib/Dirac/robustlm.c:        cudakernel_func_wt_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, wtd, Nbase, dp->M, dp->N);
src/lib/Dirac/robustlm.c:        cudakernel_hadamard_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed);
src/lib/Dirac/robustlm.c:   cudakernel_func_fl(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/robustlm.c:   cudakernel_updateweights_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd, ed, qd, robust_nu);
src/lib/Dirac/robustlm.c:   cudakernel_sqrtweights_fl(ThreadsPerBlock1, (N+ThreadsPerBlock1-1)/ThreadsPerBlock1, N, wtd);
src/lib/Dirac/robustlm.c:   cudakernel_evaluatenu_fl(ThreadsPerBlock2, (Nd+ThreadsPerBlock2-1)/ThreadsPerBlock2, Nd, q_sum, qd, deltanu,(float)robust_nulow);
src/lib/Dirac/robustlm.c:  err=cudaMemcpyAsync(p,pd,M*sizeof(float),cudaMemcpyDeviceToHost,0);
src/lib/Dirac/robustlm.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robustlm.c:  cudaDeviceSynchronize();
src/lib/Dirac/robustlm.c:  cudaFree(devInfo);
src/lib/Dirac/robustlm.c:  cudaFree(work);
src/lib/Dirac/robustlm.c:    cudaFree(rwork);
src/lib/Dirac/robustlm.c:#endif /* HAVE_CUDA */
src/lib/Dirac/robustlm.c:rlevmar_der_single_nocuda(
src/lib/Dirac/robustlm.c:osrlevmar_der_single_nocuda(
src/lib/Dirac/admm_solve.c:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double *Y, double *BZ, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,double nulow, double nuhigh,int randomize, double *admm_rho, double *mean_nu, double *res_0, double *res_1) {
src/lib/Dirac/admm_solve.c:       rtr_solve_nocuda_robust_admm(&p[carr[cj].p[ck]], &Y[carr[cj].p[ck]], &BZ[carr[cj].p[ck]], &xdummy[8*tcj*Nbase], N, ntiles*Nbase, this_itermax+5, this_itermax+10, Delta0, Delta0*0.125, admm_rho[cj], nulow, nuhigh, info, &lmdata);
src/lib/Dirac/admm_solve.c:#ifdef HAVE_CUDA
src/lib/Dirac/admm_solve.c:/* slave thread 2GPU function */
src/lib/Dirac/admm_solve.c:   /* for GPU, the cost func and jacobian are not used */
src/lib/Dirac/admm_solve.c:       nsd_solve_cuda_robust_admm_fl(&gd->p[tid][ci*(gd->M[tid])], &gd->Y[tid][ci*(gd->M[tid])], &gd->Z[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid]/8, ntiles*t->Nbase, gd->itermax[tid]+15, gd->admm_rho[tid], gd->nulow, gd->nuhigh, gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], cj, ntiles, (void*)gd->lmdata[tid]);
src/lib/Dirac/admm_solve.c:       rtr_solve_cuda_robust_admm_fl(&gd->p[tid][ci*(gd->M[tid])], &gd->Y[tid][ci*(gd->M[tid])], &gd->Z[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid]/8, ntiles*t->Nbase, gd->itermax[tid]+10, Delta0, Delta0*0.125f, gd->admm_rho[tid], gd->nulow, gd->nuhigh, gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], cj, ntiles, (void*)gd->lmdata[tid]);
src/lib/Dirac/admm_solve.c:  } else if (gd->status[tid]==PT_DO_AGPU) {
src/lib/Dirac/admm_solve.c:   attach_gpu_to_thread2(select_work_gpu(MAX_GPU_ID,td->pline->thst),&gd->cbhandle[tid],&gd->solver_handle[tid],&gd->gWORK[tid],gd->data_size,0);
src/lib/Dirac/admm_solve.c:  } else if (gd->status[tid]==PT_DO_DGPU) {
src/lib/Dirac/admm_solve.c:   detach_gpu_from_thread2(gd->cbhandle[tid],gd->solver_handle[tid],gd->gWORK[tid],0);
src/lib/Dirac/admm_solve.c:   reset_gpu_memory((double*)gd->gWORK[tid],gd->data_size);
src/lib/Dirac/admm_solve.c:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double *Y, double *BZ, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,  double nulow, double nuhigh, int randomize,double *admm_rho, double *mean_nu, double *res_0, double *res_1) {
src/lib/Dirac/admm_solve.c:  /* rearraged memory for GPU use */
src/lib/Dirac/admm_solve.c:  /* rearrange coh for GPU use */
src/lib/Dirac/admm_solve.c:  tpg.status[0]=tpg.status[1]=PT_DO_AGPU;
src/lib/Dirac/admm_solve.c:  /* also calculate the total storage needed to be allocated on a GPU */
src/lib/Dirac/admm_solve.c:  tpg.status[0]=tpg.status[1]=PT_DO_DGPU;
src/lib/Dirac/admm_solve.c:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double *Y, double *BZ, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,  double nulow, double nuhigh, int randomize,double *admm_rho, double *mean_nu, double *res_0, double *res_1) {
src/lib/Dirac/admm_solve.c:  /* rearraged memory for GPU use */
src/lib/Dirac/admm_solve.c:  /* rearrange coh for GPU use */
src/lib/Dirac/admm_solve.c:  tpg.status[0]=tpg.status[1]=PT_DO_AGPU;
src/lib/Dirac/admm_solve.c:  /* also calculate the total storage needed to be allocated on a GPU */
src/lib/Dirac/admm_solve.c:  tpg.status[0]=tpg.status[1]=PT_DO_DGPU;
src/lib/Dirac/admm_solve.c:#endif /* HAVE_CUDA */
src/lib/Dirac/rtr_solve.c:rtr_solve_nocuda(
src/lib/Dirac/rtr_solve.c: double  rho_regularization; /* default 1e2 use large damping (but less than GPU version) */
src/lib/Dirac/robust_lbfgs.c:#ifdef HAVE_CUDA
src/lib/Dirac/robust_lbfgs.c:#include <cuda_runtime.h>
src/lib/Dirac/robust_lbfgs.c:   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads,
src/lib/Dirac/robust_lbfgs.c:   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads,
src/lib/Dirac/rtr_solve_robust.c:rtr_solve_nocuda_robust(
src/lib/Dirac/rtr_solve_robust.c: double  rho_regularization; /* use large damping (but less than GPU version) */
src/lib/Dirac/rtr_solve_robust.c:nsd_solve_nocuda_robust(
src/lib/Dirac/CMakeLists.txt:if(HAVE_CUDA)
src/lib/Dirac/CMakeLists.txt:    message (STATUS "Compiling lib/Dirac with CUDA support.")
src/lib/Dirac/CMakeLists.txt:    # objects only for gpu version
src/lib/Dirac/CMakeLists.txt:        rtr_solve_cuda
src/lib/Dirac/CMakeLists.txt:        rtr_solve_robust_admm_cuda
src/lib/Dirac/CMakeLists.txt:        rtr_solve_robust_cuda
src/lib/Dirac/CMakeLists.txt:    set (extra_objects_cuda
src/lib/Dirac/CMakeLists.txt:        clmfit_cuda
src/lib/Dirac/CMakeLists.txt:        lbfgs_cuda
src/lib/Dirac/CMakeLists.txt:        lmfit_cuda
src/lib/Dirac/CMakeLists.txt:	lbfgs_minibatch_cuda
src/lib/Dirac/CMakeLists.txt:	robust_batchmode_lbfgs_cuda
src/lib/Dirac/CMakeLists.txt:    message (STATUS "Extra CUDA objects ... = ${extra_objects_cuda}")
src/lib/Dirac/CMakeLists.txt:    #    foreach (object ${extra_objects_cuda})
src/lib/Dirac/CMakeLists.txt:    #    file(GLOB CUDA_SRC_FILE ${object}.*)
src/lib/Dirac/CMakeLists.txt:    #    CUDA_ADD_LIBRARY(${object} SHARED ${CUDA_SRC_FILE})
src/lib/Dirac/CMakeLists.txt:    set(objects ${objects} ${extra_objects_cuda})
src/lib/Dirac/CMakeLists.txt:        set(CUDA_SRC_FILES ${CUDA_SRC_FILES} ${SRC_FILE})
src/lib/Dirac/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
src/lib/Dirac/CMakeLists.txt:    CUDA_ADD_LIBRARY(dirac ${CUDA_SRC_FILES} Dirac.h)
src/lib/Dirac/CMakeLists.txt:    SET_TARGET_PROPERTIES(dirac PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
src/lib/Dirac/CMakeLists.txt:      SET_TARGET_PROPERTIES(dirac PROPERTIES CUDA_ARCHITECTURES native)
src/lib/Dirac/Dirac_common.h:/* structure for worker threads for arranging coherencies for GPU use */
src/lib/Dirac/Dirac_common.h:  double *ddcoh; /* coherencies, rearranged for easy copying to GPU, also real,imag instead of complex */
src/lib/Dirac/Dirac_common.h:  /* following only used by GPU LM version */
src/lib/Dirac/Dirac_common.h:  double *ddcoh; /* coherencies, rearranged for easy copying to GPU, also real,imag instead of complex */
src/lib/Dirac/Dirac_common.h:/* structure for gpu driver threads for LBFGS */
src/lib/Dirac/Dirac_common.h:typedef struct thread_gpu_data_t {
src/lib/Dirac/Dirac_common.h:  int card; /* which gpu ? */
src/lib/Dirac/Dirac_common.h:} thread_gpu_data;
src/lib/Dirac/Dirac_common.h:/* struct to keep histoty of last used GPU */
src/lib/Dirac/Dirac_common.h:  int prev; /* last used GPU (by any thread) */
src/lib/Dirac/Dirac_common.h:  int tid; /* 0,1 for 2 GPUs */
src/lib/Dirac/Dirac_common.h:#ifndef PT_DO_AGPU
src/lib/Dirac/Dirac_common.h:#define PT_DO_AGPU 1 /* allocate GPU memory, attach GPU */
src/lib/Dirac/Dirac_common.h:#ifndef PT_DO_DGPU
src/lib/Dirac/Dirac_common.h:#define PT_DO_DGPU 2 /* free GPU memory, detach GPU */
src/lib/Dirac/Dirac_common.h:/* rearranges coherencies for GPU use later */
src/lib/Dirac/Dirac_common.h:/* rearranges baselines for GPU use later */
src/lib/Dirac/Dirac_common.h:/* select a GPU from 0,1..,max_gpu
src/lib/Dirac/Dirac_common.h:/* also keep a global variableto ensure same GPU is 
src/lib/Dirac/Dirac_common.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac_common.h:select_work_gpu(int max_gpu, taskhist *th);
src/lib/Dirac/manifold_fl.cu:#include "cuda.h"
src/lib/Dirac/manifold_fl.cu:#include "Dirac_GPUtune.h"
src/lib/Dirac/manifold_fl.cu://#define CUDA_DBG
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:cudakernel_fns_f(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh) {
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  cudaError_t error;
src/lib/Dirac/manifold_fl.cu:  cudaMalloc((void**)&ed, sizeof(float)*BlocksPerGrid);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(ed, 0, sizeof(float)*BlocksPerGrid);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaMalloc((void**)&totald, sizeof(float));
src/lib/Dirac/manifold_fl.cu:  cudaMemset(totald, 0, sizeof(float));
src/lib/Dirac/manifold_fl.cu:    cudaMalloc((void**)&eo, sizeof(float)*L);
src/lib/Dirac/manifold_fl.cu:    cudaFree(eo);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaMemcpy(&total,totald,sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/manifold_fl.cu:  cudaFree(ed);
src/lib/Dirac/manifold_fl.cu:  cudaFree(totald);
src/lib/Dirac/manifold_fl.cu:cudakernel_fns_f_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh, float *wtd) {
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  cudaError_t error;
src/lib/Dirac/manifold_fl.cu:  cudaMalloc((void**)&ed, sizeof(float)*BlocksPerGrid);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(ed, 0, sizeof(float)*BlocksPerGrid);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaMalloc((void**)&totald, sizeof(float));
src/lib/Dirac/manifold_fl.cu:  cudaMemset(totald, 0, sizeof(float));
src/lib/Dirac/manifold_fl.cu:    cudaMalloc((void**)&eo, sizeof(float)*L);
src/lib/Dirac/manifold_fl.cu:    cudaFree(eo);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaMemcpy(&total,totald,sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/manifold_fl.cu:  cudaFree(ed);
src/lib/Dirac/manifold_fl.cu:  cudaFree(totald);
src/lib/Dirac/manifold_fl.cu:cudakernel_fns_fgradflat(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh) {
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  cudaError_t error;
src/lib/Dirac/manifold_fl.cu:  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(eta, 0, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:   error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:   if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:     // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:    error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:    if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaFree(etaloc);
src/lib/Dirac/manifold_fl.cu:cudakernel_fns_fgradflat_robust1(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *wtd) {
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  cudaError_t error;
src/lib/Dirac/manifold_fl.cu:  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(eta, 0, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:   error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:   if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:     // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:    error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:    if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaFree(etaloc);
src/lib/Dirac/manifold_fl.cu:cudakernel_fns_fgradflat_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *wtd, cuFloatComplex *Ai, cublasHandle_t cbhandle) {
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  cudaError_t error;
src/lib/Dirac/manifold_fl.cu:  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(eta, 0, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  //cudaMemset(C, 0, sizeof(cuFloatComplex)*4*ntime); Not needed because a2=0
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(C, 0, sizeof(cuFloatComplex)*4);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaFree(etaloc);
src/lib/Dirac/manifold_fl.cu:cudakernel_fns_fgradflat_robust_admm(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *wtd, cublasHandle_t cbhandle) {
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  cudaError_t error;
src/lib/Dirac/manifold_fl.cu:  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(eta, 0, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaFree(etaloc);
src/lib/Dirac/manifold_fl.cu:cudakernel_fns_fhessflat(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh) {
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  cudaError_t error;
src/lib/Dirac/manifold_fl.cu:  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(fhess, 0, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:   error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:   if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:     // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:    error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:    if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaFree(etaloc);
src/lib/Dirac/manifold_fl.cu:cudakernel_fns_fhessflat_robust1(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *wtd) {
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  cudaError_t error;
src/lib/Dirac/manifold_fl.cu:  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(fhess, 0, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:   error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:   if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:     // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:    error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:    if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:     fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaFree(etaloc);
src/lib/Dirac/manifold_fl.cu:cudakernel_fns_fhessflat_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *wtd, cuFloatComplex *Ai, cublasHandle_t cbhandle) {
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  cudaError_t error;
src/lib/Dirac/manifold_fl.cu:  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(fhess, 0, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  //cudaMemset(C, 0, sizeof(cuFloatComplex)*4*ntime); Not needed because a2=0
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(C, 0, sizeof(cuFloatComplex)*4);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaFree(etaloc);
src/lib/Dirac/manifold_fl.cu:cudakernel_fns_fhessflat_robust_admm(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *wtd, cublasHandle_t cbhandle) {
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  cudaError_t error;
src/lib/Dirac/manifold_fl.cu:  cudaMalloc((void**)&etaloc, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(etaloc, 0, sizeof(cuFloatComplex)*BlocksPerGrid*4*N);
src/lib/Dirac/manifold_fl.cu:  cudaMemset(fhess, 0, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:  cudaFree(etaloc);
src/lib/Dirac/manifold_fl.cu:cudakernel_fns_fscale(int N, cuFloatComplex *eta, float *iw) {
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  cudaError_t error;
src/lib/Dirac/manifold_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu: error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu: if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:  fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:cudakernel_fns_fupdate_weights(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh, float *wtd, float nu0) {
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  cudaError_t error;
src/lib/Dirac/manifold_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/manifold_fl.cu:cudakernel_fns_fupdate_weights_q(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh, float *wtd, float *qd, float nu0) {
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  cudaError_t error;
src/lib/Dirac/manifold_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/manifold_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/manifold_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/manifold_fl.cu:  if(error != cudaSuccess) {
src/lib/Dirac/manifold_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/manifold_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm.c:rtr_solve_nocuda_robust_admm(
src/lib/Dirac/rtr_solve_robust_admm.c: double  rho_regularization; /* use large damping (but less than GPU version) */
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:#include <cuda_runtime.h>
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:checkCudaError(cudaError_t err, char *file, int line)
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:#ifdef CUDA_DEBUG
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:#ifdef CUDA_DEBUG
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:cudakernel_fns_f_robust_admm(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *Y, cuFloatComplex *Z, float admm_rho, float *y, float *coh, short *bbh,  float *wtd, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle){
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&Yd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: float f0=cudakernel_fns_f_robust(ThreadsPerBlock,BlocksPerGrid,N,M,x,y,coh,bbh,wtd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(Yd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:cudakernel_fns_proj_admm(int N, cuFloatComplex *x, cuFloatComplex *z, cuFloatComplex *rnew, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:cudakernel_fns_fgrad_robust_admm(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *Y, cuFloatComplex *Z, float admm_rho, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *iw, float *wtd, int negate, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&tempeta, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudakernel_fns_fgradflat_robust_admm(ThreadsPerBlock, Bt*ntime, N, M, x, tempeta, y, coh, bbh, wtd, cbhandle, solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudakernel_fns_fscale(N, tempeta, iw);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMemcpy(eta,x,4*N*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMemcpy(eta,tempeta,4*N*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(tempeta);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:cudakernel_fns_fhess_robust_admm(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x,  cuFloatComplex *Y, cuFloatComplex *Z, float admm_rho, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *iw, float *wtd, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) {
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&tempeta, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudakernel_fns_fhessflat_robust_admm(ThreadsPerBlock, Bt*ntime, N, M, x, eta, tempeta, y, coh, bbh, wtd, cbhandle, solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudakernel_fns_fscale(N, tempeta, iw);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMemcpy(fhess,tempeta,4*N*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(tempeta);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&s, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&x_prop, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: f0=cudakernel_fns_f_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,x,Y,Z,admm_rho,y,coh,bbh,wtd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudakernel_fns_fgrad_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,x,Y,Z,admm_rho,eta,y,coh,bbh,iw,wtd,1,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudakernel_fns_fhess_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,x,Y,Z,admm_rho,s,Heta,y,coh,bbh,iw,wtd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:   mk=f0-cudakernel_fns_g(N,x_prop,eta,s,cbhandle,solver_handle)-0.5f*cudakernel_fns_g(N,x_prop,Heta,s,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:   fk=cudakernel_fns_f_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,x_prop,Y,Z,admm_rho,y,coh,bbh,wtd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:     float g0_s=cudakernel_fns_g(N,x,eta,s,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(s);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(x_prop);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:tcg_solve_cuda(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x,  cuFloatComplex *Y, cuFloatComplex *Z, float admm_rho, cuFloatComplex *grad, cuFloatComplex *eta, cuFloatComplex *fhess, float Delta, float theta, float kappa, int max_inner, int min_inner, float *y, float *coh, short *bbh, float *iw, float *wtd, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle) { 
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaMalloc((void**)&r, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaMalloc((void**)&z, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaMalloc((void**)&delta, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaMalloc((void**)&Hxd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaMalloc((void**)&rnew, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  r_r=cudakernel_fns_g(N,x,r,r,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  z_r=cudakernel_fns_g(N,x,z,r,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaMemset(delta, 0, sizeof(cuFloatComplex)*4*N); 
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  e_Pd=cudakernel_fns_g(N,x,eta,delta,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:    cudakernel_fns_fhess_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,x,Y,Z,admm_rho,delta,Hxd,y,coh,bbh,iw,wtd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:    d_Hd=cudakernel_fns_g(N,x,delta,Hxd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:    cudakernel_fns_proj_admm(N, x, r, rnew, cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:    r_r=cudakernel_fns_g(N,x,r,r,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:    z_r=cudakernel_fns_g(N,x,z,r,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaFree(r);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaFree(z);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaFree(delta);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaFree(Hxd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaFree(rnew);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:rtr_solve_cuda_robust_admm_fl(
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaError_t err;
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&fgradxd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&etad, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&Hetad, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&x_propd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&xd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&yd, sizeof(float)*8*M);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&cohd, sizeof(float)*8*Nbase);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&bbd, sizeof(short)*2*Nbase);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&iwd, sizeof(float)*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&wtd, sizeof(float)*M);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&qd, sizeof(float)*M);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void **)&Yd, 4*N*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void **)&Zd, 4*N*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(yd, y, 8*M*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(xd, x, 8*N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(Yd, Yx, 8*N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(Zd, Zx, 8*N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(iwd, iw, N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: /* set initial weights to 1 by a cuda kernel */
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudakernel_setweights_fl(ThreadsPerBlock, (M+ThreadsPerBlock-1)/ThreadsPerBlock, M, wtd, 1.0f);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: fx=cudakernel_fns_f_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,xd,Yd,Zd,admm_rho,yd,cohd,bbd,wtd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudakernel_fns_fupdate_weights(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd,robust_nu);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:   cudakernel_fns_fgrad_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,xd, Yd,Zd,admm_rho,fgradxd,yd,cohd,bbd,iwd,wtd,1,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:   norm_grad=sqrtf(cudakernel_fns_g(N,xd,fgradxd,fgradxd,cbhandle,solver_handle));
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:    cudaMemset(etad, 0, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:    stop_inner=tcg_solve_cuda(ThreadsPerBlock,BlocksPerGrid, N, M, xd, Yd,Zd,admm_rho,fgradxd, etad, Hetad, Delta, theta, kappa, max_inner, min_inner,yd,cohd,bbd,iwd,wtd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:   cudakernel_fns_R(N,xd,etad,x_propd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:    fx_prop=cudakernel_fns_f_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,x_propd,Yd,Zd,admm_rho,yd,cohd,bbd,wtd, cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:    rhoden=-cudakernel_fns_g(N,xd,fgradxd,etad,cbhandle,solver_handle)-0.5f*cudakernel_fns_g(N,xd,Hetad,etad,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:     cudakernel_fns_fgrad_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,xd,Yd,Zd,admm_rho, fgradxd,yd,cohd,bbd,iwd,wtd,1,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:     norm_grad=sqrtf(cudakernel_fns_g(N,xd,fgradxd,fgradxd,cbhandle,solver_handle));
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaDeviceSynchronize();
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:   cudakernel_fns_fupdate_weights_q(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd,qd,robust_nu);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:   cudakernel_evaluatenu_fl_eight(ThreadsPerBlock2, (Nd+ThreadsPerBlock-1)/ThreadsPerBlock2, Nd, q_sum, qd, deltanu,(float)robust_nulow,robust_nu);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:   err=cudaMemcpy(x,xd,8*N*sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(fgradxd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(etad);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(Hetad);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(x_propd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(xd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(yd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(cohd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(bbd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(iwd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(wtd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(qd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaFree(Yd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaFree(Zd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:nsd_solve_cuda_robust_admm_fl(
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaError_t err;
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&fgradxd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&etad, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&zd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&x_propd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&xd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&z_propd, sizeof(cuFloatComplex)*4*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&yd, sizeof(float)*8*M);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&cohd, sizeof(float)*8*Nbase);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&bbd, sizeof(short)*2*Nbase);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&iwd, sizeof(float)*N);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&wtd, sizeof(float)*M);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void**)&qd, sizeof(float)*M);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void **)&Yd, 4*N*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaMalloc((void **)&Zd, 4*N*sizeof(cuFloatComplex));
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(yd, y, 8*M*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(cohd, &(dp->ddcohf[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(xd, x, 8*N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(Yd, Yx, 8*N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(Zd, Zx, 8*N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: err=cudaMemcpy(iwd, iw, N*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: /* set initial weights to 1 by a cuda kernel */
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudakernel_setweights_fl(ThreadsPerBlock, (M+ThreadsPerBlock-1)/ThreadsPerBlock, M, wtd, 1.0f);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: fx=cudakernel_fns_f_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,xd,Yd,Zd,admm_rho,yd,cohd,bbd,wtd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudakernel_fns_fgrad_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,xd,Yd,Zd,admm_rho,fgradxd,yd,cohd,bbd,iwd,wtd,1,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudakernel_fns_fhess_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,xd,Yd,Zd,admm_rho,xd,zd,yd,cohd,bbd,iwd,wtd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:    cudakernel_fns_fgrad_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,zd,Yd,Zd,admm_rho,fgradxd,yd,cohd,bbd,iwd,wtd,1,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  fx=cudakernel_fns_f_robust_admm(ThreadsPerBlock,BlocksPerGrid,N,M,xd,Yd,Zd,admm_rho,yd,cohd,bbd,wtd,cbhandle,solver_handle);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaDeviceSynchronize();
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:   cudakernel_fns_fupdate_weights_q(ThreadsPerBlock,BlocksPerGrid,N,M,xd,yd,cohd,bbd,wtd,qd,robust_nu);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:   cudakernel_evaluatenu_fl_eight(ThreadsPerBlock2, (Nd+ThreadsPerBlock-1)/ThreadsPerBlock2, Nd, q_sum, qd, deltanu,(float)robust_nulow,robust_nu);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  err=cudaMemcpy(x,xd,8*N*sizeof(float),cudaMemcpyDeviceToHost);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(fgradxd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(etad);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(zd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(x_propd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(xd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(z_propd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(yd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(cohd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(bbd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(iwd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(wtd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c: cudaFree(qd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaFree(Yd);
src/lib/Dirac/rtr_solve_robust_admm_cuda.c:  cudaFree(Zd);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:#ifdef HAVE_CUDA
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:#include <cuda_runtime.h>
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c://#define CUDA_DEBUG
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:checkCudaError(cudaError_t err, char *file, int line)
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:#ifdef CUDA_DEBUG
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:#ifdef CUDA_DEBUG
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:typedef struct me_data_batchmode_cuda_t_ {
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:} me_data_batchmode_cuda_t;
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: me_data_batchmode_cuda_t *lmdata=(me_data_batchmode_cuda_t *)adata;
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: double fcost=cudakernel_lbfgs_multifreq_cost_robust(Nbase,lmdata->Nchan,lmdata->M,lmdata->N,Nbasetotal,boff,lmdata->x,lmdata->coh,p,m,lmdata->hbb,lmdata->ptoclus,lmdata->robust_nu);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: cudaError_t err;
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: err=cudaMalloc((void**)&(xp),m*sizeof(double));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: err=cudaMemcpy(xp,p,m*sizeof(double),cudaMemcpyDeviceToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: err=cudaFree(xp);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: me_data_batchmode_cuda_t *lmdata=(me_data_batchmode_cuda_t *)adata;
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: cudakernel_lbfgs_multifreq_r_robust(Nbase,lmdata->tilesz,lmdata->Nchan,lmdata->M,lmdata->N,Nbasetotal,boff,lmdata->x,lmdata->coh,p,m,lmdata->hbb,lmdata->ptoclus,g,lmdata->robust_nu);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: cudaError_t err;
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: err=cudaMalloc((void**)&(xp),m*sizeof(double));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: err=cudaMemcpy(xp,p,m*sizeof(double),cudaMemcpyDeviceToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: err=cudaFree(xp);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:   int Nbase, int tilesz, short *hbb, int *ptoclus, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *indata,int nminibatch, int totalminibatch) {
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  me_data_batchmode_cuda_t lmdata;
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  cudaError_t err;
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMalloc((void**)&(pdevice),m*sizeof(double));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(pdevice,p,m*sizeof(double),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  /* copy all necerrary data to the GPU, and only pass pointers to
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:     the data as a struct to the cost and grad functions :lbfgs_cuda.c 140*/
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMalloc((void**)&(lmdata.x),n*sizeof(double));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(lmdata.x,x,n*sizeof(double),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMalloc((void**)&(lmdata.coh),8*Nbase*tilesz*M*Nf*sizeof(double));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(lmdata.coh,(double*)coh,8*Nbase*tilesz*M*Nf*sizeof(double),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  /* GPU replacement for barr */
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMalloc((void**)&(lmdata.hbb),2*Nbase*tilesz*sizeof(short));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(lmdata.hbb,hbb,2*Nbase*tilesz*sizeof(short),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  /* GPU replacement for carr */
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMalloc((void**)&(lmdata.ptoclus),2*M*sizeof(int));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(lmdata.ptoclus,ptoclus,2*M*sizeof(int),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  /* call lbfgs_fit_cuda() with proper cost/grad functions */
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  lbfgs_fit_cuda(costfunc_multifreq,gradfunc_multifreq,pdevice,m,max_lbfgs,lbfgs_m,&lmdata,indata);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(p,pdevice,m*sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaFree(pdevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaFree(lmdata.x);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaFree(lmdata.coh);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaFree(lmdata.hbb);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaFree(lmdata.ptoclus);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:   int Nbase, int tilesz, short *hbb, int *ptoclus, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, double *y, double *z, double *rho, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *indata,int nminibatch, int totalminibatch) {
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  me_data_batchmode_cuda_t lmdata;
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  cudaError_t err;
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMalloc((void**)&(pdevice),m*sizeof(double));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(pdevice,p,m*sizeof(double),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  /* copy all necerrary data to the GPU, and only pass pointers to
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:     the data as a struct to the cost and grad functions :lbfgs_cuda.c 140*/
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMalloc((void**)&(lmdata.x),n*sizeof(double));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(lmdata.x,x,n*sizeof(double),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMalloc((void**)&(lmdata.coh),8*Nbase*tilesz*M*Nf*sizeof(double));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(lmdata.coh,(double*)coh,8*Nbase*tilesz*M*Nf*sizeof(double),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  /* GPU replacement for barr */
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMalloc((void**)&(lmdata.hbb),2*Nbase*tilesz*sizeof(short));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(lmdata.hbb,hbb,2*Nbase*tilesz*sizeof(short),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  /* GPU replacement for carr */
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMalloc((void**)&(lmdata.ptoclus),2*M*sizeof(int));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(lmdata.ptoclus,ptoclus,2*M*sizeof(int),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMalloc((void**)&(lmdata.y),m*sizeof(double));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(lmdata.y,y,m*sizeof(double),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMalloc((void**)&(lmdata.z),m*sizeof(double));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(lmdata.z,z,m*sizeof(double),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: err=cudaMalloc((void**)&(tmpgrad),m*sizeof(double));
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: err=cudaHostAlloc((void**)&(tmphost),m*sizeof(double),cudaHostAllocDefault);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: err=cudaMemcpy(tmphost,tmpgrad,m*sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: err=cudaMemcpy(pdevice,p,m*sizeof(double),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: err=cudaMemcpy(pdevice,p,m*sizeof(double),cudaMemcpyHostToDevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: cudaFree(tmpgrad);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c: cudaFreeHost(tmphost);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  /* call lbfgs_fit_cuda() with proper cost/grad functions */
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  lbfgs_fit_cuda(costfunc_multifreq,gradfunc_multifreq,pdevice,m,max_lbfgs,lbfgs_m,&lmdata,indata);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaMemcpy(p,pdevice,m*sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaFree(pdevice);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaFree(lmdata.x);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaFree(lmdata.coh);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaFree(lmdata.hbb);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaFree(lmdata.ptoclus);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaFree(lmdata.y);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  err=cudaFree(lmdata.z);
src/lib/Dirac/robust_batchmode_lbfgs_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:#include "cuda.h"
src/lib/Dirac/mderiv.cu:#include "Dirac_GPUtune.h"
src/lib/Dirac/mderiv.cu://#define CUDA_DBG
src/lib/Dirac/mderiv.cu:checkCudaError(cudaError_t err, const char *file, int line)
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DEBUG
src/lib/Dirac/mderiv.cu:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Dirac/mderiv.cu:/* cuda driver for kernel */
src/lib/Dirac/mderiv.cu:cudakernel_lbfgs(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad){
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  cudaError_t error;
src/lib/Dirac/mderiv.cu:  cudaDeviceSynchronize();
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:  if(error != cudaSuccess) {
src/lib/Dirac/mderiv.cu:    // print the CUDA error message and exit
src/lib/Dirac/mderiv.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:cudakernel_lbfgs_r_robust(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad, double robust_nu){
src/lib/Dirac/mderiv.cu:  cudaError_t error;
src/lib/Dirac/mderiv.cu:  if((error=cudaMalloc((void**)&eo, Nbase*8*sizeof(double)))!=cudaSuccess) {
src/lib/Dirac/mderiv.cu:  cudaMemset(eo, 0, sizeof(double)*Nbase*8);
src/lib/Dirac/mderiv.cu:  checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  error = cudaGetLastError(); /* reset all previous errors */
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:  if(error != cudaSuccess) {
src/lib/Dirac/mderiv.cu:    // print the CUDA error message and exit
src/lib/Dirac/mderiv.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:  if(error != cudaSuccess) {
src/lib/Dirac/mderiv.cu:    // print the CUDA error message and exit
src/lib/Dirac/mderiv.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:  cudaFree(eo);
src/lib/Dirac/mderiv.cu:cudakernel_lbfgs_r(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad){
src/lib/Dirac/mderiv.cu:  cudaError_t error;
src/lib/Dirac/mderiv.cu:  if((error=cudaMalloc((void**)&eo, Nbase*8*sizeof(double)))!=cudaSuccess) {
src/lib/Dirac/mderiv.cu:  cudaMemset(eo, 0, sizeof(double)*Nbase*8);
src/lib/Dirac/mderiv.cu:  checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  error = cudaGetLastError(); /* reset all previous errors */
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:  if(error != cudaSuccess) {
src/lib/Dirac/mderiv.cu:    // print the CUDA error message and exit
src/lib/Dirac/mderiv.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:  if(error != cudaSuccess) {
src/lib/Dirac/mderiv.cu:    // print the CUDA error message and exit
src/lib/Dirac/mderiv.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:  cudaFree(eo);
src/lib/Dirac/mderiv.cu:cudakernel_lbfgs_cost_robust(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int boff, int M, int Ns, int Nbasetotal, double *x, double *coh, double *p, short *bb, int *ptoclus, double robust_nu){
src/lib/Dirac/mderiv.cu:  cudaError_t error;
src/lib/Dirac/mderiv.cu:  if((error=cudaMalloc((void**)&ed, sizeof(double)*BlocksPerGrid))!=cudaSuccess) {
src/lib/Dirac/mderiv.cu:  cudaMemset(ed, 0, sizeof(double)*BlocksPerGrid);
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:  checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:  if((error=cudaMalloc((void**)&totald, sizeof(double)))!=cudaSuccess) {
src/lib/Dirac/mderiv.cu:  cudaMemset(totald, 0, sizeof(double));
src/lib/Dirac/mderiv.cu:  checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:    error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:    checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:    if((error=cudaMalloc((void**)&eo, L*sizeof(double)))!=cudaSuccess) {
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:    error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:    checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:    error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:    checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:    cudaFree(eo);
src/lib/Dirac/mderiv.cu:  cudaMemcpy(&total,totald,sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/mderiv.cu:  cudaFree(totald);
src/lib/Dirac/mderiv.cu:  cudaFree(ed);
src/lib/Dirac/mderiv.cu:cudakernel_lbfgs_cost(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int boff, int M, int Ns, int Nbasetotal, double *x, double *coh, double *p, short *bb, int *ptoclus){
src/lib/Dirac/mderiv.cu:  cudaError_t error;
src/lib/Dirac/mderiv.cu:  if((error=cudaMalloc((void**)&ed, sizeof(double)*BlocksPerGrid))!=cudaSuccess) {
src/lib/Dirac/mderiv.cu:  cudaMemset(ed, 0, sizeof(double)*BlocksPerGrid);
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:  checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:  if((error=cudaMalloc((void**)&totald, sizeof(double)))!=cudaSuccess) {
src/lib/Dirac/mderiv.cu:  cudaMemset(totald, 0, sizeof(double));
src/lib/Dirac/mderiv.cu:  checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:    error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:    checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:    if((error=cudaMalloc((void**)&eo, L*sizeof(double)))!=cudaSuccess) {
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:    error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:    checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:    error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:    checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:    cudaFree(eo);
src/lib/Dirac/mderiv.cu:  cudaMemcpy(&total,totald,sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/mderiv.cu:  cudaFree(totald);
src/lib/Dirac/mderiv.cu:  cudaFree(ed);
src/lib/Dirac/mderiv.cu:cudakernel_diagdiv(int ThreadsPerBlock, int BlocksPerGrid, int M, double eps, double *Dpd, double *Sd) {
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  cudaError_t error;
src/lib/Dirac/mderiv.cu:  cudaDeviceSynchronize();
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:  if(error != cudaSuccess)
src/lib/Dirac/mderiv.cu:    // print the CUDA error message and exit
src/lib/Dirac/mderiv.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:/* cuda driver for calculating
src/lib/Dirac/mderiv.cu:cudakernel_diagmu(int ThreadsPerBlock, int BlocksPerGrid, int M, double *A, double mu) {
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  cudaError_t error;
src/lib/Dirac/mderiv.cu:  cudaDeviceSynchronize();
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:  if(error != cudaSuccess)
src/lib/Dirac/mderiv.cu:    // print the CUDA error message and exit
src/lib/Dirac/mderiv.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:/* cuda driver for calculating f() */
src/lib/Dirac/mderiv.cu:cudakernel_func(int ThreadsPerBlock, int BlocksPerGrid, double *p, double *x, int M, int N, double *coh, short *bbh, int Nbase, int Mclus, int Nstations) {
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  cudaError_t error;
src/lib/Dirac/mderiv.cu:  cudaMemset(x, 0, N*sizeof(double));
src/lib/Dirac/mderiv.cu:  cudaDeviceSynchronize();
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:  if(error != cudaSuccess)
src/lib/Dirac/mderiv.cu:    // print the CUDA error message and exit
src/lib/Dirac/mderiv.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/mderiv.cu:/* cuda driver for calculating jacf() */
src/lib/Dirac/mderiv.cu:cudakernel_jacf(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, double *p, double *jac, int M, int N, double *coh, short *bbh, int Nbase, int Mclus, int Nstations) {
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  cudaError_t error;
src/lib/Dirac/mderiv.cu:  cudaMemset(jac, 0, N*M*sizeof(double));
src/lib/Dirac/mderiv.cu:  cudaDeviceSynchronize();
src/lib/Dirac/mderiv.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv.cu:  if(error != cudaSuccess)
src/lib/Dirac/mderiv.cu:    // print the CUDA error message and exit
src/lib/Dirac/mderiv.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:#include <cuda_runtime.h>
src/lib/Dirac/oslmfit.c:checkCudaError(cudaError_t err, char *file, int line)
src/lib/Dirac/oslmfit.c:#ifdef CUDA_DEBUG
src/lib/Dirac/oslmfit.c:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Dirac/oslmfit.c:#ifdef CUDA_DEBUG
src/lib/Dirac/oslmfit.c:  entirely in the GPU */
src/lib/Dirac/oslmfit.c:oslevmar_der_single_cuda(
src/lib/Dirac/oslmfit.c:  double *gWORK, /* GPU allocated memory */
src/lib/Dirac/oslmfit.c:  cudaError_t err;
src/lib/Dirac/oslmfit.c:  /* calculate no of cuda threads and blocks */
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**)&xd, N*sizeof(double));
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**)&jacd, M*N*sizeof(double));
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**)&jacTjacd, M*M*sizeof(double));
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**)&jacTed, M*sizeof(double));
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**)&jacTjacd0, M*M*sizeof(double));
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**)&Dpd, M*sizeof(double));
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**)&bd, M*sizeof(double));
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**)&pd, M*sizeof(double));
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**)&pnewd, M*sizeof(double));
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**) &bbd, Nbase*2*sizeof(short));
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**) &cohd, Nbase*8*sizeof(double)); 
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**)&hxd, N*sizeof(double));
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**)&ed, N*sizeof(double));
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:    err=cudaMalloc((void**)&taud, M*sizeof(double));
src/lib/Dirac/oslmfit.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:    err=cudaMalloc((void**)&Ud, M*M*sizeof(double));
src/lib/Dirac/oslmfit.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:    err=cudaMalloc((void**)&VTd, M*M*sizeof(double));
src/lib/Dirac/oslmfit.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:    err=cudaMalloc((void**)&Sd, M*sizeof(double));
src/lib/Dirac/oslmfit.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMalloc((void**)&devInfo, sizeof(int));
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/oslmfit.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/oslmfit.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:    err=cudaMalloc((void**)&work, work_size*sizeof(double));
src/lib/Dirac/oslmfit.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:    err=cudaMalloc((void**)&rwork, 5*M*sizeof(double));
src/lib/Dirac/oslmfit.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMemcpyAsync(pd, p, M*sizeof(double), cudaMemcpyHostToDevice,0);
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMemcpyAsync(cohd, &(dp->ddcoh[(dp->Nbase)*(dp->tilesz)*(dp->clus)*8+(dp->Nbase)*tileoff*8]), Nbase*8*sizeof(double), cudaMemcpyHostToDevice,0);
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  err=cudaMemcpyAsync(bbd, &(dp->ddbase[2*(dp->Nbase)*(tileoff)]), Nbase*2*sizeof(short), cudaMemcpyHostToDevice,0);
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  cudaDeviceSynchronize();
src/lib/Dirac/oslmfit.c:  err=cudaMemcpyAsync(xd, x, N*sizeof(double), cudaMemcpyHostToDevice,0);
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pd,hxd,M,N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/oslmfit.c:     //cudakernel_jacf(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/oslmfit.c:     cudakernel_jacf(ThreadsPerBlock, ThreadsPerBlock/4, pd, jacd, M, Nos[l], &cohd[8*NbI[l]], &bbd[2*NbI[l]], Nbaseos[l], dp->M, dp->N);
src/lib/Dirac/oslmfit.c:     err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/oslmfit.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:      err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/oslmfit.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:      cudakernel_diagmu(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);
src/lib/Dirac/oslmfit.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/oslmfit.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/oslmfit.c:        cudaDeviceSynchronize();
src/lib/Dirac/oslmfit.c:        cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/oslmfit.c:         cudaDeviceSynchronize();
src/lib/Dirac/oslmfit.c:         cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
src/lib/Dirac/oslmfit.c:        cudaDeviceSynchronize();
src/lib/Dirac/oslmfit.c:        cudakernel_diagdiv(ThreadsPerBlock, BlocksPerGrid, M, eps1, Dpd, Sd);
src/lib/Dirac/oslmfit.c:        cudakernel_func(ThreadsPerBlock, (Nbase+ThreadsPerBlock-1)/ThreadsPerBlock, pnewd, hxd, M, N, cohd, bbd, Nbase, dp->M, dp->N);
src/lib/Dirac/oslmfit.c:  err=cudaMemcpyAsync(p,pd,M*sizeof(double),cudaMemcpyDeviceToHost,0);
src/lib/Dirac/oslmfit.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Dirac/oslmfit.c:  cudaDeviceSynchronize();
src/lib/Dirac/oslmfit.c:  cudaFree(xd);
src/lib/Dirac/oslmfit.c:  cudaFree(jacd);
src/lib/Dirac/oslmfit.c:  cudaFree(jacTjacd);
src/lib/Dirac/oslmfit.c:  cudaFree(jacTjacd0);
src/lib/Dirac/oslmfit.c:  cudaFree(jacTed);
src/lib/Dirac/oslmfit.c:  cudaFree(Dpd);
src/lib/Dirac/oslmfit.c:  cudaFree(bd);
src/lib/Dirac/oslmfit.c:  cudaFree(pd);
src/lib/Dirac/oslmfit.c:  cudaFree(pnewd);
src/lib/Dirac/oslmfit.c:  cudaFree(hxd);
src/lib/Dirac/oslmfit.c:  cudaFree(ed);
src/lib/Dirac/oslmfit.c:   cudaFree(taud);
src/lib/Dirac/oslmfit.c:   cudaFree(Ud);
src/lib/Dirac/oslmfit.c:   cudaFree(VTd);
src/lib/Dirac/oslmfit.c:   cudaFree(Sd);
src/lib/Dirac/oslmfit.c:  cudaFree(cohd);
src/lib/Dirac/oslmfit.c:  cudaFree(bbd);
src/lib/Dirac/oslmfit.c:  cudaFree(devInfo);
src/lib/Dirac/oslmfit.c:  cudaFree(work);
src/lib/Dirac/oslmfit.c:    cudaFree(rwork);
src/lib/Dirac/load_balance.c:/* select a GPU from 0,1..,max_gpu
src/lib/Dirac/load_balance.c:select_work_gpu(int max_gpu, taskhist *th) {
src/lib/Dirac/load_balance.c:  if (!max_gpu) return 0; /* no need to spend time if only one GPU is available */
src/lib/Dirac/load_balance.c:  /* check if max_gpu > no. of actual devices */
src/lib/Dirac/load_balance.c:  cudaGetDeviceCount(&actual_devcount);
src/lib/Dirac/load_balance.c:  if (max_gpu+1>actual_devcount) {
src/lib/Dirac/load_balance.c:  return rank%(max_gpu+1); /* modulo value */
src/lib/Dirac/load_balance.c:    retval=random_pick(max_gpu, th);
src/lib/Dirac/load_balance.c:     retval=random_pick(max_gpu, th);
src/lib/Dirac/load_balance.c:    unsigned int min_util=101; /* GPU utilization */
src/lib/Dirac/load_balance.c:    unsigned int max_util=0; /* GPU utilization */
src/lib/Dirac/load_balance.c:    for (ci=0; ci<=max_gpu; ci++) {
src/lib/Dirac/load_balance.c:      if (min_util>nvmlUtilization.gpu) {
src/lib/Dirac/load_balance.c:          min_util=nvmlUtilization.gpu;
src/lib/Dirac/load_balance.c:      if (max_util<nvmlUtilization.gpu) {
src/lib/Dirac/load_balance.c:          max_util=nvmlUtilization.gpu;
src/lib/Dirac/load_balance.c:    /* give priority for selection a GPU with max free memory,
src/lib/Dirac/load_balance.c:     retval=random_pick(max_gpu,th);
src/lib/Dirac/load_balance.c:      retval=random_pick(max_gpu,th);
src/lib/Dirac/baseline_utils.c:/* rearranges coherencies for GPU use later */
src/lib/Dirac/baseline_utils.c:/* rearranges baselines for GPU use later */
src/lib/Dirac/mderiv_fl.cu:#include "cuda.h"
src/lib/Dirac/mderiv_fl.cu://#define CUDA_DBG
src/lib/Dirac/mderiv_fl.cu:cudakernel_diagdiv_fl(int ThreadsPerBlock, int BlocksPerGrid, int M, float eps, float *Dpd, float *Sd) {
src/lib/Dirac/mderiv_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv_fl.cu:  cudaError_t error;
src/lib/Dirac/mderiv_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/mderiv_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv_fl.cu:  if(error != cudaSuccess)
src/lib/Dirac/mderiv_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/mderiv_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/mderiv_fl.cu:/* cuda driver for calculating
src/lib/Dirac/mderiv_fl.cu:cudakernel_diagmu_fl(int ThreadsPerBlock, int BlocksPerGrid, int M, float *A, float mu) {
src/lib/Dirac/mderiv_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv_fl.cu:  cudaError_t error;
src/lib/Dirac/mderiv_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/mderiv_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv_fl.cu:  if(error != cudaSuccess)
src/lib/Dirac/mderiv_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/mderiv_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/mderiv_fl.cu:/* cuda driver for calculating f() */
src/lib/Dirac/mderiv_fl.cu:cudakernel_func_fl(int ThreadsPerBlock, int BlocksPerGrid, float *p, float *x, int M, int N, float *coh, short *bbh, int Nbase, int Mclus, int Nstations) {
src/lib/Dirac/mderiv_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv_fl.cu:  cudaError_t error;
src/lib/Dirac/mderiv_fl.cu:  cudaMemset(x, 0, N*sizeof(float));
src/lib/Dirac/mderiv_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/mderiv_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv_fl.cu:  if(error != cudaSuccess)
src/lib/Dirac/mderiv_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/mderiv_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/mderiv_fl.cu:/* cuda driver for calculating jacf() */
src/lib/Dirac/mderiv_fl.cu:cudakernel_jacf_fl(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, float *p, float *jac, int M, int N, float *coh, short *bbh, int Nbase, int Mclus, int Nstations) {
src/lib/Dirac/mderiv_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv_fl.cu:  cudaError_t error;
src/lib/Dirac/mderiv_fl.cu:  cudaMemset(jac, 0, N*M*sizeof(float));
src/lib/Dirac/mderiv_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/mderiv_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/mderiv_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/mderiv_fl.cu:  if(error != cudaSuccess)
src/lib/Dirac/mderiv_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/mderiv_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/lmfit_cuda.c:/******************** minimization  with 2 GPU  *****************************/
src/lib/Dirac/lmfit_cuda.c:/***************** reimplementation using persistant  GPU threads ***********/
src/lib/Dirac/lmfit_cuda.c:/****** 2GPU version ****************************/
src/lib/Dirac/lmfit_cuda.c:/* slave thread 2GPU function */
src/lib/Dirac/lmfit_cuda.c:   /* for GPU, the cost func and jacobian are not used */
src/lib/Dirac/lmfit_cuda.c:      clevmar_der_single_cuda(NULL, NULL, &gd->p[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid], 8*ntiles*t->Nbase, gd->itermax[tid], gd->opts[tid], gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], gd->gWORK[tid], gd->linsolv, cj, ntiles, (void*)gd->lmdata[tid]);
src/lib/Dirac/lmfit_cuda.c:      oslevmar_der_single_cuda(NULL, NULL, &gd->p[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid], 8*ntiles*t->Nbase, gd->itermax[tid], gd->opts[tid], gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], gd->gWORK[tid], gd->linsolv, cj, ntiles, gd->randomize, (void*)gd->lmdata[tid]);
src/lib/Dirac/lmfit_cuda.c:      rlevmar_der_single_cuda(NULL, NULL, &gd->p[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid], 8*ntiles*t->Nbase, gd->itermax[tid], gd->opts[tid], gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], gd->gWORK[tid], gd->linsolv, cj, ntiles, gd->nulow,gd->nuhigh, (void*)gd->lmdata[tid]);
src/lib/Dirac/lmfit_cuda.c:  } else if (gd->status[tid]==PT_DO_AGPU) {
src/lib/Dirac/lmfit_cuda.c:   attach_gpu_to_thread1(select_work_gpu(MAX_GPU_ID,td->pline->thst),&gd->cbhandle[tid],&gd->solver_handle[tid],&gd->gWORK[tid],gd->data_size);
src/lib/Dirac/lmfit_cuda.c:  } else if (gd->status[tid]==PT_DO_DGPU) {
src/lib/Dirac/lmfit_cuda.c:   detach_gpu_from_thread1(gd->cbhandle[tid],gd->solver_handle[tid],gd->gWORK[tid]);
src/lib/Dirac/lmfit_cuda.c:   reset_gpu_memory(gd->gWORK[tid],gd->data_size);
src/lib/Dirac/lmfit_cuda.c:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,  double nulow, double nuhigh, int randomize, double *mean_nu, double *res_0, double *res_1) {
src/lib/Dirac/lmfit_cuda.c:  /* rearraged memory for GPU use */
src/lib/Dirac/lmfit_cuda.c:  /* rearrange coh for GPU use */
src/lib/Dirac/lmfit_cuda.c:  tpg.status[0]=tpg.status[1]=PT_DO_AGPU;
src/lib/Dirac/lmfit_cuda.c:  /* also calculate the total storage needed to be allocated on a GPU */
src/lib/Dirac/lmfit_cuda.c:  tpg.status[0]=tpg.status[1]=PT_DO_DGPU;
src/lib/Dirac/lmfit_cuda.c:    lbfgs_fit_robust_cuda(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata0);
src/lib/Dirac/lmfit_cuda.c:    lbfgs_fit(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata0);
src/lib/Dirac/lmfit_cuda.c:/*************************** 1 GPU version *********************************/
src/lib/Dirac/lmfit_cuda.c:/* slave thread 1GPU function */
src/lib/Dirac/lmfit_cuda.c:pipeline_slave_code_one_gpu(void *data)
src/lib/Dirac/lmfit_cuda.c:   /* for GPU, the cost func and jacobian are not used */
src/lib/Dirac/lmfit_cuda.c:      clevmar_der_single_cuda(NULL, NULL, &gd->p[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid], 8*ntiles*t->Nbase, gd->itermax[tid], gd->opts[tid], gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], gd->gWORK[tid], gd->linsolv, cj, ntiles,  (void*)gd->lmdata[tid]);
src/lib/Dirac/lmfit_cuda.c:      oslevmar_der_single_cuda(NULL, NULL, &gd->p[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid], 8*ntiles*t->Nbase, gd->itermax[tid], gd->opts[tid], gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid],  gd->gWORK[tid], gd->linsolv, cj, ntiles, gd->randomize, (void*)gd->lmdata[tid]);
src/lib/Dirac/lmfit_cuda.c:      rlevmar_der_single_cuda(NULL, NULL, &gd->p[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid], 8*ntiles*t->Nbase, gd->itermax[tid], gd->opts[tid], gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], gd->gWORK[tid], gd->linsolv, cj, ntiles,  gd->nulow, gd->nuhigh, (void*)gd->lmdata[tid]);
src/lib/Dirac/lmfit_cuda.c:  } else if (gd->status[tid]==PT_DO_AGPU) {
src/lib/Dirac/lmfit_cuda.c:   attach_gpu_to_thread1(select_work_gpu(MAX_GPU_ID,td->pline->thst),&gd->cbhandle[tid],&gd->solver_handle[tid],&gd->gWORK[tid],gd->data_size);
src/lib/Dirac/lmfit_cuda.c:  } else if (gd->status[tid]==PT_DO_DGPU) {
src/lib/Dirac/lmfit_cuda.c:   detach_gpu_from_thread1(gd->cbhandle[tid],gd->solver_handle[tid],gd->gWORK[tid]);
src/lib/Dirac/lmfit_cuda.c:   reset_gpu_memory(gd->gWORK[tid],gd->data_size);
src/lib/Dirac/lmfit_cuda.c:init_pipeline_one_gpu(th_pipeline *pline,
src/lib/Dirac/lmfit_cuda.c: pthread_create(&(pline->slave0),&(pline->attr),pipeline_slave_code_one_gpu,(void*)t0);
src/lib/Dirac/lmfit_cuda.c:destroy_pipeline_one_gpu(th_pipeline *pline)
src/lib/Dirac/lmfit_cuda.c:sagefit_visibilities_dual_pt_one_gpu(double *u, double *v, double *w, double *x, int N,   
src/lib/Dirac/lmfit_cuda.c:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv, int solver_mode,  double nulow, double nuhigh, int randomize, double *mean_nu, double *res_0, double *res_1) {
src/lib/Dirac/lmfit_cuda.c:  /* rearraged memory for GPU use */
src/lib/Dirac/lmfit_cuda.c:  /* rearrange coh for GPU use */
src/lib/Dirac/lmfit_cuda.c:  init_pipeline_one_gpu(&tp,&tpg);
src/lib/Dirac/lmfit_cuda.c:  tpg.status[0]=PT_DO_AGPU;
src/lib/Dirac/lmfit_cuda.c:/************ setup GPU *********************/
src/lib/Dirac/lmfit_cuda.c:/************ done setup GPU *********************/
src/lib/Dirac/lmfit_cuda.c:  tpg.status[0]=PT_DO_DGPU;
src/lib/Dirac/lmfit_cuda.c:  destroy_pipeline_one_gpu(&tp);
src/lib/Dirac/lmfit_cuda.c:    lbfgs_fit_robust_cuda(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata);
src/lib/Dirac/lmfit_cuda.c:    lbfgs_fit(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata);
src/lib/Dirac/lmfit_cuda.c:bfgsfit_visibilities_gpu(double *u, double *v, double *w, double *x, int N,   
src/lib/Dirac/lmfit_cuda.c:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode,  double mean_nu, double *res_0, double *res_1) {
src/lib/Dirac/lmfit_cuda.c:    lbfgs_fit_robust_cuda(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata);
src/lib/Dirac/lmfit_cuda.c:    lbfgs_fit(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata);
src/lib/Dirac/lmfit_cuda.c:/* slave thread 2GPU function */
src/lib/Dirac/lmfit_cuda.c:   /* for GPU, the cost func and jacobian are not used */
src/lib/Dirac/lmfit_cuda.c:      clevmar_der_single_cuda_fl(&gd->p[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid], 8*ntiles*t->Nbase, gd->itermax[tid], gd->opts[tid], gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], gd->gWORK[tid], gd->linsolv, cj, ntiles, (void*)gd->lmdata[tid]);
src/lib/Dirac/lmfit_cuda.c:      oslevmar_der_single_cuda_fl(&gd->p[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid], 8*ntiles*t->Nbase, gd->itermax[tid], gd->opts[tid], gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], gd->gWORK[tid], gd->linsolv, cj, ntiles, gd->randomize, (void*)gd->lmdata[tid]);
src/lib/Dirac/lmfit_cuda.c:      rlevmar_der_single_cuda_fl(&gd->p[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid], 8*ntiles*t->Nbase, gd->itermax[tid], gd->opts[tid], gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], gd->gWORK[tid], gd->linsolv, cj, ntiles, gd->nulow,gd->nuhigh,(void*)gd->lmdata[tid]);
src/lib/Dirac/lmfit_cuda.c:      osrlevmar_der_single_cuda_fl(&gd->p[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid], 8*ntiles*t->Nbase, gd->itermax[tid], gd->opts[tid], gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], gd->gWORK[tid], gd->linsolv, cj, ntiles, gd->nulow,gd->nuhigh,gd->randomize,(void*)gd->lmdata[tid]); 
src/lib/Dirac/lmfit_cuda.c:      rtr_solve_cuda_fl(&gd->p[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid]/8, ntiles*t->Nbase, gd->itermax[tid]+5, gd->itermax[tid]+10, Delta0, Delta0*0.125f, gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid], cj, ntiles, (void*)gd->lmdata[tid]);
src/lib/Dirac/lmfit_cuda.c:      rtr_solve_cuda_robust_fl(&gd->p[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid]/8, ntiles*t->Nbase, gd->itermax[tid]+5, gd->itermax[tid]+10, Delta0, Delta0*0.125f, gd->nulow, gd->nuhigh, gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid],  cj, ntiles, (void*)gd->lmdata[tid]);
src/lib/Dirac/lmfit_cuda.c:      nsd_solve_cuda_robust_fl(&gd->p[tid][ci*(gd->M[tid])], &gd->x[tid][8*cj*t->Nbase], gd->M[tid]/8, ntiles*t->Nbase, gd->itermax[tid]+15, gd->nulow, gd->nuhigh, gd->info[tid], gd->cbhandle[tid], gd->solver_handle[tid],  cj, ntiles, (void*)gd->lmdata[tid]);
src/lib/Dirac/lmfit_cuda.c:  } else if (gd->status[tid]==PT_DO_AGPU) {
src/lib/Dirac/lmfit_cuda.c:   attach_gpu_to_thread2(select_work_gpu(MAX_GPU_ID,td->pline->thst),&gd->cbhandle[tid],&gd->solver_handle[tid],&gd->gWORK[tid],gd->data_size,1);
src/lib/Dirac/lmfit_cuda.c:  } else if (gd->status[tid]==PT_DO_DGPU) {
src/lib/Dirac/lmfit_cuda.c:   detach_gpu_from_thread2(gd->cbhandle[tid],gd->solver_handle[tid],gd->gWORK[tid],1);
src/lib/Dirac/lmfit_cuda.c:   reset_gpu_memory((double*)gd->gWORK[tid],gd->data_size);
src/lib/Dirac/lmfit_cuda.c:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,  double nulow, double nuhigh, int randomize, double *mean_nu, double *res_0, double *res_1) {
src/lib/Dirac/lmfit_cuda.c:  /* rearraged memory for GPU use */
src/lib/Dirac/lmfit_cuda.c:  /* rearrange coh for GPU use */
src/lib/Dirac/lmfit_cuda.c:  tpg.status[0]=tpg.status[1]=PT_DO_AGPU;
src/lib/Dirac/lmfit_cuda.c:  /* also calculate the total storage needed to be allocated on a GPU */
src/lib/Dirac/lmfit_cuda.c:  tpg.status[0]=tpg.status[1]=PT_DO_DGPU;
src/lib/Dirac/lmfit_cuda.c:    lbfgs_fit_robust_cuda(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata0);
src/lib/Dirac/lmfit_cuda.c:    lbfgs_fit(p, x, m, n, max_lbfgs, lbfgs_m, gpu_threads, (void*)&lmdata0);
src/lib/Dirac/lmfit_cuda.c:  /* final residual calculation: FIXME: possible GPU accel here? */
src/lib/Dirac/lbfgs_multifreq.cu:#include "cuda.h"
src/lib/Dirac/lbfgs_multifreq.cu:#include "Dirac_GPUtune.h"
src/lib/Dirac/lbfgs_multifreq.cu://#define CUDA_DBG
src/lib/Dirac/lbfgs_multifreq.cu:checkCudaError(cudaError_t err, const char *file, int line)
src/lib/Dirac/lbfgs_multifreq.cu:#ifdef CUDA_DEBUG
src/lib/Dirac/lbfgs_multifreq.cu:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Dirac/lbfgs_multifreq.cu:cudakernel_lbfgs_multifreq_r_robust(int Nbase, int tilesz, int Nchan, int M, int Ns, int Nbasetotal, int boff, double *x, double *coh, double *p, int m, short *bb, int *ptoclus, double *grad, double robust_nu){
src/lib/Dirac/lbfgs_multifreq.cu:  cudaError_t error;
src/lib/Dirac/lbfgs_multifreq.cu:  if((error=cudaMalloc((void**)&eo, Nbase*8*Nchan*sizeof(double)))!=cudaSuccess) {
src/lib/Dirac/lbfgs_multifreq.cu:  cudaMemset(eo, 0, sizeof(double)*Nbase*8*Nchan);
src/lib/Dirac/lbfgs_multifreq.cu:  checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_multifreq.cu:#ifdef CUDA_DBG
src/lib/Dirac/lbfgs_multifreq.cu:  error = cudaGetLastError(); /* reset all previous errors */
src/lib/Dirac/lbfgs_multifreq.cu:#ifdef CUDA_DBG
src/lib/Dirac/lbfgs_multifreq.cu:  error = cudaGetLastError();
src/lib/Dirac/lbfgs_multifreq.cu:  if(error != cudaSuccess) {
src/lib/Dirac/lbfgs_multifreq.cu:    // print the CUDA error message and exit
src/lib/Dirac/lbfgs_multifreq.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/lbfgs_multifreq.cu:#ifdef CUDA_DBG
src/lib/Dirac/lbfgs_multifreq.cu:  error = cudaGetLastError();
src/lib/Dirac/lbfgs_multifreq.cu:  if(error != cudaSuccess) {
src/lib/Dirac/lbfgs_multifreq.cu:    // print the CUDA error message and exit
src/lib/Dirac/lbfgs_multifreq.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/lbfgs_multifreq.cu:  cudaFree(eo);
src/lib/Dirac/lbfgs_multifreq.cu:cudakernel_lbfgs_multifreq_cost_robust(int Nbase, int Nchan, int M, int Ns, int Nbasetotal, int boff, double *x, double *coh, double *p, int m, short *bb, int *ptoclus, double robust_nu){
src/lib/Dirac/lbfgs_multifreq.cu:  cudaError_t error;
src/lib/Dirac/lbfgs_multifreq.cu:  if((error=cudaMalloc((void**)&ed, sizeof(double)*blocksPerGridXY))!=cudaSuccess) {
src/lib/Dirac/lbfgs_multifreq.cu:  cudaMemset(ed, 0, sizeof(double)*blocksPerGridXY);
src/lib/Dirac/lbfgs_multifreq.cu:#ifdef CUDA_DBG
src/lib/Dirac/lbfgs_multifreq.cu:  error = cudaGetLastError();
src/lib/Dirac/lbfgs_multifreq.cu:  checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_multifreq.cu:  if((error=cudaMalloc((void**)&totald, sizeof(double)))!=cudaSuccess) {
src/lib/Dirac/lbfgs_multifreq.cu:  cudaMemset(totald, 0, sizeof(double));
src/lib/Dirac/lbfgs_multifreq.cu:  checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_multifreq.cu:#ifdef CUDA_DBG
src/lib/Dirac/lbfgs_multifreq.cu:    error = cudaGetLastError();
src/lib/Dirac/lbfgs_multifreq.cu:    checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_multifreq.cu:    if((error=cudaMalloc((void**)&eo, L*sizeof(double)))!=cudaSuccess) {
src/lib/Dirac/lbfgs_multifreq.cu:#ifdef CUDA_DBG
src/lib/Dirac/lbfgs_multifreq.cu:    error = cudaGetLastError();
src/lib/Dirac/lbfgs_multifreq.cu:    checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_multifreq.cu:#ifdef CUDA_DBG
src/lib/Dirac/lbfgs_multifreq.cu:    error = cudaGetLastError();
src/lib/Dirac/lbfgs_multifreq.cu:    checkCudaError(error,__FILE__,__LINE__);
src/lib/Dirac/lbfgs_multifreq.cu:    cudaFree(eo);
src/lib/Dirac/lbfgs_multifreq.cu:  cudaMemcpy(&total,totald,sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/lbfgs_multifreq.cu:  cudaFree(totald);
src/lib/Dirac/lbfgs_multifreq.cu:  cudaFree(ed);
src/lib/Dirac/Dirac.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac.h:#include <cuda_runtime_api.h>
src/lib/Dirac/Dirac.h:/* GPU specific tunable parameters */
src/lib/Dirac/Dirac.h:#include "Dirac_GPUtune.h"
src/lib/Dirac/Dirac.h:#endif /* HAVE_CUDA */
src/lib/Dirac/Dirac.h:/****************************** lbfgs_cuda.c ****************************/
src/lib/Dirac/Dirac.h:#ifndef HAVE_CUDA
src/lib/Dirac/Dirac.h:   gpu_threads: GPU threads per block
src/lib/Dirac/Dirac.h:#endif /* !HAVE_CUDA */
src/lib/Dirac/Dirac.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac.h:   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads, void *adata);
src/lib/Dirac/Dirac.h:lbfgs_fit_robust_cuda(
src/lib/Dirac/Dirac.h:   double *p, double *x, int m, int n, int itmax, int lbfgs_m, int gpu_threads, void *adata);
src/lib/Dirac/Dirac.h:#endif /* HAVE_CUDA */
src/lib/Dirac/Dirac.h:/****************************** lbfgs_minibatch_cuda.c ****************************/
src/lib/Dirac/Dirac.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac.h:  also pointers to GPU memory for running LBFGS 
src/lib/Dirac/Dirac.h:  all allocations will be on the GPU */
src/lib/Dirac/Dirac.h:  /* GPU handles created by attach_gpu_to_thread() */
src/lib/Dirac/Dirac.h:  /* note: cost,grad functions may attach to GPU separately */
src/lib/Dirac/Dirac.h:   for using stochastic LBFGS : On the GPU */
src/lib/Dirac/Dirac.h:/* First, a GPU chosen and attach to it as well */
src/lib/Dirac/Dirac.h:lbfgs_fit_cuda(
src/lib/Dirac/Dirac.h:#endif /* HAVE_CUDA */
src/lib/Dirac/Dirac.h:   double *p, double *x, int m, int n, int itmax, int lbfgs_m, int gpu_threads,
src/lib/Dirac/Dirac.h:   double *p, double *x, int m, int n, int itmax, int lbfgs_m, int gpu_threads,
src/lib/Dirac/Dirac.h:/****************************** robust_batchmode_lbfgs_cuda.c ****************************/
src/lib/Dirac/Dirac.h:   double *p, double *x, int m, int n, int itmax, int M, int gpu_threads, void *adata);
src/lib/Dirac/Dirac.h:/* Note: ptdata below will differ for CPU and GPU versions,
src/lib/Dirac/Dirac.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz, short *hbb, int *ptoclus, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *ptdata,int nminibatch,int totalminibatch);
src/lib/Dirac/Dirac.h:#else /* !HAVE_CUDA */
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz, baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *ptdata,int nminibatch,int totalminibatch);
src/lib/Dirac/Dirac.h:#endif /* !HAVE_CUDA */
src/lib/Dirac/Dirac.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz, short *hbb, int *ptoclus, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, double *y, double *z, double *rho, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *ptdata,int nminibatch, int totalminibatch);
src/lib/Dirac/Dirac.h:#else /* !HAVE_CUDA */
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz, baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double *freqs, int Nf, double fdelta, double *p, double *y, double *z, double *rho, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double robust_nu, double *res_0, double *res_1, persistent_data_t *ptdata,int nminibatch, int totalminibatch);
src/lib/Dirac/Dirac.h:#endif /* !HAVE_CUDA */
src/lib/Dirac/Dirac.h:/* cuda driver for kernel */
src/lib/Dirac/Dirac.h:cudakernel_lbfgs(int ThreadsPerBlock, int BlocksPerGrid, int N, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad);
src/lib/Dirac/Dirac.h:cudakernel_lbfgs_r(int ThreadsPerBlock, int BlocksPerGrid, int N, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad);
src/lib/Dirac/Dirac.h:cudakernel_lbfgs_r_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int tilesz, int M, int Ns, int Nparam, int goff, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad, double robust_nu);
src/lib/Dirac/Dirac.h:/* cost function calculation, each GPU works with Nbase baselines out of Nbasetotal baselines
src/lib/Dirac/Dirac.h:cudakernel_lbfgs_cost(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int boff, int M, int Ns, int Nbasetotal, double *x, double *coh, double *p, short *bb, int *ptoclus);
src/lib/Dirac/Dirac.h:cudakernel_lbfgs_cost_robust(int ThreadsPerBlock, int BlocksPerGrid, int Nbase, int boff, int M, int Ns, int Nbasetotal, double *x, double *coh, double *p, short *bb, int *ptoclus, double robust_nu);
src/lib/Dirac/Dirac.h:cudakernel_diagdiv(int ThreadsPerBlock, int BlocksPerGrid, int M, double eps, double *Dpd, double *Sd);
src/lib/Dirac/Dirac.h:/* cuda driver for calculating
src/lib/Dirac/Dirac.h:cudakernel_diagmu(int ThreadsPerBlock, int BlocksPerGrid, int M, double *A, double mu);
src/lib/Dirac/Dirac.h:/* cuda driver for calculating f() */
src/lib/Dirac/Dirac.h:cudakernel_func(int ThreadsPerBlock, int BlocksPerGrid, double *p, double *x, int M, int N, double *coh, short *bbh, int Nbase, int Mclus, int Nstations);
src/lib/Dirac/Dirac.h:/* cuda driver for calculating jacf() */
src/lib/Dirac/Dirac.h:cudakernel_jacf(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, double *p, double *jac, int M, int N, double *coh, short *bbh, int Nbase, int Mclus, int Nstations);
src/lib/Dirac/Dirac.h:cudakernel_diagdiv_fl(int ThreadsPerBlock, int BlocksPerGrid, int M, float eps, float *Dpd, float *Sd);
src/lib/Dirac/Dirac.h:/* cuda driver for calculating
src/lib/Dirac/Dirac.h:cudakernel_diagmu_fl(int ThreadsPerBlock, int BlocksPerGrid, int M, float *A, float mu);
src/lib/Dirac/Dirac.h:/* cuda driver for calculating f() */
src/lib/Dirac/Dirac.h:cudakernel_func_fl(int ThreadsPerBlock, int BlocksPerGrid, float *p, float *x, int M, int N, float *coh, short *bbh, int Nbase, int Mclus, int Nstations);
src/lib/Dirac/Dirac.h:/* cuda driver for calculating jacf() */
src/lib/Dirac/Dirac.h:cudakernel_jacf_fl(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, float *p, float *jac, int M, int N, float *coh, short *bbh, int Nbase, int Mclus, int Nstations);
src/lib/Dirac/Dirac.h:/* cuda driver for calculating wt \odot f() */
src/lib/Dirac/Dirac.h:cudakernel_func_wt(int ThreadsPerBlock, int BlocksPerGrid, double *p, double *x, int M, int N, double *coh, short *bbh, double *wt, int Nbase, int Mclus, int Nstations);
src/lib/Dirac/Dirac.h:/* cuda driver for calculating wt \odot jacf() */
src/lib/Dirac/Dirac.h:cudakernel_jacf_wt(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, double *p, double *jac, int M, int N, double *coh, short *bbh, double *wt, int Nbase, int Mclus, int Nstations);
src/lib/Dirac/Dirac.h:/* set initial weights to 1 by a cuda kernel */
src/lib/Dirac/Dirac.h:cudakernel_setweights(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wtd, double alpha);
src/lib/Dirac/Dirac.h:/* hadamard product by a cuda kernel x<= x*wt */
src/lib/Dirac/Dirac.h:cudakernel_hadamard(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt, double *x);
src/lib/Dirac/Dirac.h:/* sum hadamard product by a cuda kernel y=y+x.*w (x.*w elementwise) */
src/lib/Dirac/Dirac.h:cudakernel_hadamard_sum(int ThreadsPerBlock, int BlocksPerGrid, int N, double *y, double *x, double *w);
src/lib/Dirac/Dirac.h:/* update weights by a cuda kernel */
src/lib/Dirac/Dirac.h:cudakernel_updateweights(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt, double *x, double *q, double robust_nu);
src/lib/Dirac/Dirac.h:cudakernel_sqrtweights(int ThreadsPerBlock, int BlocksPerGrid, int N, double *wt);
src/lib/Dirac/Dirac.h:cudakernel_evaluatenu(int ThreadsPerBlock, int BlocksPerGrid, int Nd, double qsum, double *q, double deltanu,double nulow);
src/lib/Dirac/Dirac.h:cudakernel_lbfgs_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int tilesz, int M, int Ns, int Nparam, int goff, double robust_nu, double *x, double *coh, double *p, short *bb, int *ptoclus, double *grad);
src/lib/Dirac/Dirac.h:/* cuda driver for calculating wt \odot f() */
src/lib/Dirac/Dirac.h:cudakernel_func_wt_fl(int ThreadsPerBlock, int BlocksPerGrid, float *p, float *x, int M, int N, float *coh, short *bbh, float *wt, int Nbase, int Mclus, int Nstations);
src/lib/Dirac/Dirac.h:/* cuda driver for calculating wt \odot jacf() */
src/lib/Dirac/Dirac.h:cudakernel_jacf_wt_fl(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, float *p, float *jac, int M, int N, float *coh, short *bbh, float *wt, int Nbase, int Mclus, int Nstations);
src/lib/Dirac/Dirac.h:/* set initial weights to 1 by a cuda kernel */
src/lib/Dirac/Dirac.h:cudakernel_setweights_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wtd, float alpha);
src/lib/Dirac/Dirac.h:/* hadamard product by a cuda kernel x<= x*wt */
src/lib/Dirac/Dirac.h:cudakernel_hadamard_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt, float *x);
src/lib/Dirac/Dirac.h:/* update weights by a cuda kernel */
src/lib/Dirac/Dirac.h:cudakernel_updateweights_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt, float *x, float *q, float robust_nu);
src/lib/Dirac/Dirac.h:cudakernel_sqrtweights_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt);
src/lib/Dirac/Dirac.h:cudakernel_evaluatenu_fl(int ThreadsPerBlock, int BlocksPerGrid, int Nd, float qsum, float *q, float deltanu,float nulow);
src/lib/Dirac/Dirac.h:cudakernel_evaluatenu_fl_eight(int ThreadsPerBlock, int BlocksPerGrid, int Nd, float qsum, float *q, float deltanu,float nulow, float nu0);
src/lib/Dirac/Dirac.h:cudakernel_lbfgs_multifreq_r_robust(int Nbase, int tilesz, int Nchan, int M, int Ns, int Nbasetotal, int boff, double *x, double *coh, double *p, int m, short *bb, int *ptoclus, double *grad, double robust_nu);
src/lib/Dirac/Dirac.h:cudakernel_lbfgs_multifreq_cost_robust(int Nbase, int Nchan, int M, int Ns, int Nbasetotal, int boff, double *x, double *coh, double *p, int m, short *bb, int *ptoclus, double robust_nu);
src/lib/Dirac/Dirac.h:/****************************** clmfit_cuda.c ****************************/
src/lib/Dirac/Dirac.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac.h:/* LM with GPU */
src/lib/Dirac/Dirac.h:  int card,   /* GPU to use */
src/lib/Dirac/Dirac.h:/* function to set up a GPU, should be called only once */
src/lib/Dirac/Dirac.h:attach_gpu_to_thread(int card, cublasHandle_t *cbhandle, cusolverDnHandle_t *solver_handle);
src/lib/Dirac/Dirac.h:attach_gpu_to_thread1(int card, cublasHandle_t *cbhandle, cusolverDnHandle_t *solver_handle, double **WORK, int64_t work_size);
src/lib/Dirac/Dirac.h:attach_gpu_to_thread2(int card,  cublasHandle_t *cbhandle, cusolverDnHandle_t *solver_handle, float **WORK, int64_t work_size, int usecula);
src/lib/Dirac/Dirac.h:/* function to detach a GPU from a thread */
src/lib/Dirac/Dirac.h:detach_gpu_from_thread(cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
src/lib/Dirac/Dirac.h:detach_gpu_from_thread1(cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle, double *WORK);
src/lib/Dirac/Dirac.h:detach_gpu_from_thread2(cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle, float *WORK, int usecula);
src/lib/Dirac/Dirac.h:reset_gpu_memory(double *WORK, int64_t work_size);
src/lib/Dirac/Dirac.h:  entirely in the GPU */
src/lib/Dirac/Dirac.h:clevmar_der_single_cuda(
src/lib/Dirac/Dirac.h:  double *gWORK, /* GPU allocated memory */
src/lib/Dirac/Dirac.h:mlm_der_single_cuda(
src/lib/Dirac/Dirac.h:  double *gWORK, /* GPU allocated memory */
src/lib/Dirac/Dirac.h:#endif /* HAVE_CUDA */
src/lib/Dirac/Dirac.h:  entirely in the GPU */
src/lib/Dirac/Dirac.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac.h:rlevmar_der_single_cuda(
src/lib/Dirac/Dirac.h:  double *gWORK, /* GPU allocated memory */
src/lib/Dirac/Dirac.h:  entirely in the GPU, using float data */
src/lib/Dirac/Dirac.h:rlevmar_der_single_cuda_fl(
src/lib/Dirac/Dirac.h:  float *gWORK, /* GPU allocated memory */
src/lib/Dirac/Dirac.h:  entirely in the GPU, using float data, OS acceleration */
src/lib/Dirac/Dirac.h:osrlevmar_der_single_cuda_fl(
src/lib/Dirac/Dirac.h:  float *gWORK, /* GPU allocated memory */
src/lib/Dirac/Dirac.h:#endif /* HAVE_CUDA */
src/lib/Dirac/Dirac.h:rlevmar_der_single_nocuda(
src/lib/Dirac/Dirac.h:osrlevmar_der_single_nocuda(
src/lib/Dirac/Dirac.h:clevmar_der_single_nocuda(
src/lib/Dirac/Dirac.h:oslevmar_der_single_nocuda(
src/lib/Dirac/Dirac.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac.h:  entirely in the GPU */
src/lib/Dirac/Dirac.h:oslevmar_der_single_cuda(
src/lib/Dirac/Dirac.h:  double *gWORK, /* GPU allocated memory */
src/lib/Dirac/Dirac.h:#endif /* !HAVE_CUDA */
src/lib/Dirac/Dirac.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac.h:clevmar_der_single_cuda_fl(
src/lib/Dirac/Dirac.h:  float *gWORK, /* GPU allocated memory */
src/lib/Dirac/Dirac.h:oslevmar_der_single_cuda_fl(
src/lib/Dirac/Dirac.h:  float *gWORK, /* GPU allocated memory */
src/lib/Dirac/Dirac.h:#endif /* !HAVE_CUDA */
src/lib/Dirac/Dirac.h:rtr_solve_nocuda(
src/lib/Dirac/Dirac.h:rtr_solve_nocuda_robust(
src/lib/Dirac/Dirac.h:nsd_solve_nocuda_robust(
src/lib/Dirac/Dirac.h:rtr_solve_nocuda_robust_admm(
src/lib/Dirac/Dirac.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac.h:cudakernel_fns_f(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh);
src/lib/Dirac/Dirac.h:cudakernel_fns_fgradflat(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh);
src/lib/Dirac/Dirac.h:cudakernel_fns_fhessflat(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh);
src/lib/Dirac/Dirac.h:cudakernel_fns_fscale(int N, cuFloatComplex *eta, float *iw);
src/lib/Dirac/Dirac.h:cudakernel_fns_f_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh,  float *wtd);
src/lib/Dirac/Dirac.h:cudakernel_fns_fgradflat_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *wtd, cuFloatComplex *Ai, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
src/lib/Dirac/Dirac.h:cudakernel_fns_fgradflat_robust1(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *wtd);
src/lib/Dirac/Dirac.h:cudakernel_fns_fgradflat_robust_admm(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, float *y, float *coh, short *bbh, float *wtd, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
src/lib/Dirac/Dirac.h:cudakernel_fns_fhessflat_robust1(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *wtd);
src/lib/Dirac/Dirac.h:cudakernel_fns_fhessflat_robust(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *wtd, cuFloatComplex *Ai, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
src/lib/Dirac/Dirac.h:cudakernel_fns_fhessflat_robust_admm(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, cuFloatComplex *eta, cuFloatComplex *fhess, float *y, float *coh, short *bbh, float *wtd, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
src/lib/Dirac/Dirac.h:cudakernel_fns_fupdate_weights(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh, float *wtd, float nu0);
src/lib/Dirac/Dirac.h:cudakernel_fns_fupdate_weights_q(int ThreadsPerBlock, int BlocksPerGrid, int N, int M, cuFloatComplex *x, float *y, float *coh, short *bbh, float *wtd, float *qd, float nu0);
src/lib/Dirac/Dirac.h:/****************************** rtr_solve_cuda.c ****************************/
src/lib/Dirac/Dirac.h:rtr_solve_cuda_fl(
src/lib/Dirac/Dirac.h:cudakernel_fns_R(int N, cuFloatComplex *x, cuFloatComplex *r, cuFloatComplex *rnew, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
src/lib/Dirac/Dirac.h:cudakernel_fns_g(int N,cuFloatComplex *x,cuFloatComplex *eta, cuFloatComplex *gamma,cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
src/lib/Dirac/Dirac.h:cudakernel_fns_proj(int N, cuFloatComplex *x, cuFloatComplex *z, cuFloatComplex *rnew, cublasHandle_t cbhandle, cusolverDnHandle_t solver_handle);
src/lib/Dirac/Dirac.h:/****************************** rtr_solve_robust_cuda.c ****************************/
src/lib/Dirac/Dirac.h:rtr_solve_cuda_robust_fl(
src/lib/Dirac/Dirac.h:nsd_solve_cuda_robust_fl(
src/lib/Dirac/Dirac.h:/****************************** rtr_solve_robust_admm_cuda.c ****************************/
src/lib/Dirac/Dirac.h:rtr_solve_cuda_robust_admm_fl(
src/lib/Dirac/Dirac.h:nsd_solve_cuda_robust_admm_fl(
src/lib/Dirac/Dirac.h:#endif /* HAVE_CUDA */
src/lib/Dirac/Dirac.h:/****************************** lmfit_cuda.c ****************************/
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double *Y, double *BZ, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,double nulow, double nuhigh,int randomize, double *admm_rho, double *mean_nu, double *res_0, double *res_1);
src/lib/Dirac/Dirac.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double *Y, double *BZ, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,  double nulow, double nuhigh, int randomize, double *admm_rho, double *mean_nu, double *res_0, double *res_1);
src/lib/Dirac/Dirac.h:/****************************** lmfit_cuda.c ****************************/
src/lib/Dirac/Dirac.h:  gpu_threads: GPU threads per block (LBFGS)
src/lib/Dirac/Dirac.h:  linsolv: (GPU/CPU versions) 0: Cholesky, 1: QR, 2: SVD
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz,  baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt,int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv, int solver_mode, double nulow, double nuhigh, int randomize, double *mean_nu, double *res_0, double *res_1);
src/lib/Dirac/Dirac.h:/* same as above, but uses 2 GPUS in the LM stage */
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz,  baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt,int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv, double nulow, double nuhigh, int randomize,  double *mean_nu, double *res_0, double *res_1);
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz,  baseline_t *barr,  int *nchunks, int *pindex, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,double nulow, double nuhigh,int randomize, double *mean_nu, double *res_0, double *res_1);
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz,  baseline_t *barr,  int *nchunks, int *pindex, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode,double nu_mean, double *res_0, double *res_1);
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz,  baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode, double mean_nu, double *res_0, double *res_1);
src/lib/Dirac/Dirac.h:#ifdef HAVE_CUDA
src/lib/Dirac/Dirac.h:bfgsfit_visibilities_gpu(double *u, double *v, double *w, double *x, int N,
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_lbfgs, int lbfgs_m, int gpu_threads, int solver_mode,  double mean_nu, double *res_0, double *res_1);
src/lib/Dirac/Dirac.h:              1: allocate GPU  memory, attach GPU
src/lib/Dirac/Dirac.h:              2: free GPU memory, detach GPU
src/lib/Dirac/Dirac.h:              3,4..: do work on GPU
src/lib/Dirac/Dirac.h:              99: reset GPU memory (memest all memory) */
src/lib/Dirac/Dirac.h:  /* GPU related info */
src/lib/Dirac/Dirac.h:  double *gWORK[2]; /* GPU buffers */
src/lib/Dirac/Dirac.h:              1: allocate GPU  memory, attach GPU
src/lib/Dirac/Dirac.h:              3: free GPU memory, detach GPU
src/lib/Dirac/Dirac.h:              3,4..: do work on GPU
src/lib/Dirac/Dirac.h:              99: reset GPU memory (memest all memory) */
src/lib/Dirac/Dirac.h:  /* GPU related info */
src/lib/Dirac/Dirac.h:  float *gWORK[2]; /* GPU buffers */
src/lib/Dirac/Dirac.h:              1: allocate GPU  memory, attach GPU
src/lib/Dirac/Dirac.h:              3: free GPU memory, detach GPU
src/lib/Dirac/Dirac.h:              3,4..: do work on GPU
src/lib/Dirac/Dirac.h:              99: reset GPU memory (memest all memory) */
src/lib/Dirac/Dirac.h:  /* GPU related info */
src/lib/Dirac/Dirac.h:  float *gWORK[2]; /* GPU buffers */
src/lib/Dirac/Dirac.h:#endif /* !HAVE_CUDA */
src/lib/Dirac/Dirac.h:/* with 2 GPUs */
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz,  baseline_t *barr, clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt,int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv, int solver_mode, double nulow, double nuhigh, int randomize, double *mean_nu, double *res_0, double *res_1);
src/lib/Dirac/Dirac.h:/* with 1 GPU and 1 CPU thread */
src/lib/Dirac/Dirac.h:sagefit_visibilities_dual_pt_one_gpu(double *u, double *v, double *w, double *x, int N,
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,  double nulow, double nuhigh, int randomize, double *mean_nu, double *res_0, double *res_1);
src/lib/Dirac/Dirac.h:   int Nbase, int tilesz,  baseline_t *barr,  clus_source_t *carr, complex double *coh, int M, int Mt, double freq0, double fdelta, double *pp, double uvmin, int Nt, int max_emiter, int max_iter, int max_lbfgs, int lbfgs_m, int gpu_threads, int linsolv,int solver_mode,  double nulow, double nuhigh, int randomize, double *mean_nu, double *res_0, double *res_1);
src/lib/Dirac/Dirac.h:#ifndef HAVE_CUDA
src/lib/Dirac/Dirac.h:#endif /* !HAVE_CUDA */
src/lib/Dirac/clmfit.c:clevmar_der_single_nocuda(
src/lib/Dirac/clmfit.c:     //err=cudaMemcpy(&jacTe_inf,&(jacTed[ci-1]),sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit.c:      //err=cudaMemcpy(&tmp,&(jacTjacd[ci]),sizeof(double),cudaMemcpyDeviceToHost);
src/lib/Dirac/clmfit.c:      //cudakernel_diagmu(ThreadsPerBlock, BlocksPerGrid, M, jacTjacd, mu);
src/lib/Dirac/clmfit.c:        //err=cudaMemcpy(hxd, hx, N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Dirac/clmfit.c:oslevmar_der_single_nocuda(
src/lib/Dirac/robust_fl.cu:#include "cuda.h"
src/lib/Dirac/robust_fl.cu://#define CUDA_DBG
src/lib/Dirac/robust_fl.cu:/* set initial weights to 1 by a cuda kernel */
src/lib/Dirac/robust_fl.cu:cudakernel_setweights_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt, float alpha) {
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  cudaError_t error;
src/lib/Dirac/robust_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/robust_fl.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust_fl.cu:/* hadamard product by a cuda kernel x<= x*wt */
src/lib/Dirac/robust_fl.cu:cudakernel_hadamard_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt, float *x) {
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  cudaError_t error;
src/lib/Dirac/robust_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/robust_fl.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust_fl.cu:/* update weights by a cuda kernel */
src/lib/Dirac/robust_fl.cu:cudakernel_updateweights_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt, float *x, float *q, float robust_nu) {
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  cudaError_t error;
src/lib/Dirac/robust_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/robust_fl.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust_fl.cu:/* update weights by a cuda kernel */
src/lib/Dirac/robust_fl.cu:cudakernel_sqrtweights_fl(int ThreadsPerBlock, int BlocksPerGrid, int N, float *wt) {
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  cudaError_t error;
src/lib/Dirac/robust_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/robust_fl.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust_fl.cu:cudakernel_evaluatenu_fl(int ThreadsPerBlock, int BlocksPerGrid, int Nd, float qsum, float *q, float deltanu,float nulow) {
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  cudaError_t error;
src/lib/Dirac/robust_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/robust_fl.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust_fl.cu:cudakernel_evaluatenu_fl_eight(int ThreadsPerBlock, int BlocksPerGrid, int Nd, float qsum, float *q, float deltanu,float nulow, float nu0) {
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  cudaError_t error;
src/lib/Dirac/robust_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/robust_fl.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust_fl.cu:/* cuda driver for calculating wt \odot f() */
src/lib/Dirac/robust_fl.cu:cudakernel_func_wt_fl(int ThreadsPerBlock, int BlocksPerGrid, float *p, float *x, int M, int N, float *coh, short *bbh, float *wt, int Nbase, int Mclus, int Nstations) {
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  cudaError_t error;
src/lib/Dirac/robust_fl.cu:  cudaMemset(x, 0, N*sizeof(float));
src/lib/Dirac/robust_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/robust_fl.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Dirac/robust_fl.cu:/* cuda driver for calculating wt \odot jacf() */
src/lib/Dirac/robust_fl.cu:cudakernel_jacf_wt_fl(int ThreadsPerBlock_row, int  ThreadsPerBlock_col, float *p, float *jac, int M, int N, float *coh, short *bbh, float *wt, int Nbase, int Mclus, int Nstations, int clus) {
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  cudaError_t error;
src/lib/Dirac/robust_fl.cu:  cudaMemset(jac, 0, N*M*sizeof(float));
src/lib/Dirac/robust_fl.cu:  cudaDeviceSynchronize();
src/lib/Dirac/robust_fl.cu:#ifdef CUDA_DBG
src/lib/Dirac/robust_fl.cu:  error = cudaGetLastError();
src/lib/Dirac/robust_fl.cu:  if(error != cudaSuccess)
src/lib/Dirac/robust_fl.cu:    // print the CUDA error message and exit
src/lib/Dirac/robust_fl.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Radio/Dirac_radio.h:#ifdef HAVE_CUDA
src/lib/Radio/Dirac_radio.h:#include <cuda_runtime_api.h>
src/lib/Radio/Dirac_radio.h:#endif /* HAVE_CUDA */
src/lib/Radio/Dirac_radio.h:#ifdef HAVE_CUDA
src/lib/Radio/Dirac_radio.h:#include <Dirac_GPUtune.h>
src/lib/Radio/Dirac_radio.h:#endif /* HAVE_CUDA */
src/lib/Radio/Dirac_radio.h:/* have_cuda: if 1, use GPU version, else only CPU version */
src/lib/Radio/Dirac_radio.h:   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, int diffuse_cluster, int sh_n0, double sh_beta, complex double *Z, int Nt, int use_cuda);
src/lib/Radio/Dirac_radio.h:/****************************** predict_withbeam_cuda.c ****************************/
src/lib/Radio/Dirac_radio.h:#ifdef HAVE_CUDA
src/lib/Radio/Dirac_radio.h:precalculate_coherencies_withbeam_gpu(double *u, double *v, double *w, complex double *x, int N,
src/lib/Radio/Dirac_radio.h:predict_visibilities_multifreq_withbeam_gpu(double *u,double *v,double *w,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
src/lib/Radio/Dirac_radio.h:calculate_residuals_multifreq_withbeam_gpu(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
src/lib/Radio/Dirac_radio.h:predict_visibilities_withsol_withbeam_gpu(double *u,double *v,double *w,double *p,double *x, int *ignorelist, int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
src/lib/Radio/Dirac_radio.h:precalculate_coherencies_multifreq_withbeam_gpu(double *u, double *v, double *w, complex double *x, int N,
src/lib/Radio/Dirac_radio.h:#endif /*!HAVE_CUDA */
src/lib/Radio/Dirac_radio.h:#ifdef HAVE_CUDA
src/lib/Radio/Dirac_radio.h:cudakernel_array_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
src/lib/Radio/Dirac_radio.h:cudakernel_tile_array_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
src/lib/Radio/Dirac_radio.h:cudakernel_element_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
src/lib/Radio/Dirac_radio.h:cudakernel_coherencies(int B, int N, int T, int K, int F, double *u, double *v, double *w,baseline_t *barr, double *freqs, float *beam, float *element, double *ll, double *mm, double *nn, double *sI, double *sQ, double *sU, double *sV,
src/lib/Radio/Dirac_radio.h:cudakernel_residuals(int B, int N, int T, int K, int F, double *u, double *v, double *w, double *p, int nchunk, baseline_t *barr, double *freqs, float *beam, float *element, double *ll, double *mm, double *nn, double *sI, double *sQ, double *sU, double *sV,
src/lib/Radio/Dirac_radio.h:cudakernel_correct_residuals(int B, int N, int Nb, int boff, int F, int nchunk, double *x, double *p, baseline_t *barr);
src/lib/Radio/Dirac_radio.h:cudakernel_convert_time(int T, double *time_utc);
src/lib/Radio/Dirac_radio.h:cudakernel_calculate_shapelet_coherencies(float u, float v, float *modes, float *fact, int n0, float beta, double *coh);
src/lib/Radio/Dirac_radio.h:#endif /* !HAVE_CUDA */
src/lib/Radio/predict_withbeam.c:  /* setup threads : note: Ngpu is no of GPUs used */
src/lib/Radio/CMakeLists.txt:if(HAVE_CUDA)
src/lib/Radio/CMakeLists.txt:    message (STATUS "Compiling lib/Radio with CUDA support.")
src/lib/Radio/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
src/lib/Radio/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../Dirac)
src/lib/Radio/CMakeLists.txt:    CUDA_INCLUDE_DIRECTORIES(${GLIB_PKG_INCLUDE_DIRS})
src/lib/Radio/CMakeLists.txt:    # objects only for gpu version
src/lib/Radio/CMakeLists.txt:    set (extra_objects_cuda
src/lib/Radio/CMakeLists.txt:        predict_withbeam_cuda
src/lib/Radio/CMakeLists.txt:    message (STATUS "Extra CUDA objects ... = ${extra_objects_cuda}")
src/lib/Radio/CMakeLists.txt:    #foreach (object ${extra_objects_cuda})
src/lib/Radio/CMakeLists.txt:    #    file(GLOB CUDA_SRC_FILE ${object}.*)
src/lib/Radio/CMakeLists.txt:    #    CUDA_ADD_LIBRARY(${object} SHARED ${CUDA_SRC_FILE})
src/lib/Radio/CMakeLists.txt:    set(objects ${objects} ${extra_objects} ${extra_objects_cuda})
src/lib/Radio/CMakeLists.txt:if(HAVE_CUDA)
src/lib/Radio/CMakeLists.txt:        set(CUDA_SRC_FILES ${CUDA_SRC_FILES} ${SRC_FILE})
src/lib/Radio/CMakeLists.txt:    CUDA_ADD_LIBRARY(dirac-radio ${CUDA_SRC_FILES} Dirac_radio.h ${CMAKE_CURRENT_SOURCE_DIR}/../Dirac/Dirac_common.h)
src/lib/Radio/CMakeLists.txt:    SET_TARGET_PROPERTIES(dirac-radio PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
src/lib/Radio/CMakeLists.txt:      SET_TARGET_PROPERTIES(dirac-radio PROPERTIES CUDA_ARCHITECTURES native)
src/lib/Radio/diffuse_predict.c:#ifdef HAVE_CUDA
src/lib/Radio/diffuse_predict.c:#include "Dirac_GPUtune.h"
src/lib/Radio/diffuse_predict.c:#ifdef HAVE_CUDA
src/lib/Radio/diffuse_predict.c:  taskhist *hst; /* for load balancing GPUs */
src/lib/Radio/diffuse_predict.c:#endif /* HAVE_CUDA */
src/lib/Radio/diffuse_predict.c:#ifdef HAVE_CUDA
src/lib/Radio/diffuse_predict.c:#define CUDA_DEBUG
src/lib/Radio/diffuse_predict.c:checkCudaError(cudaError_t err, char *file, int line)
src/lib/Radio/diffuse_predict.c:#ifdef CUDA_DEBUG
src/lib/Radio/diffuse_predict.c:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Radio/diffuse_predict.c:shapelet_pred_threadfn_cuda(void *data) {
src/lib/Radio/diffuse_predict.c:  card=select_work_gpu(MAX_GPU_ID,t->hst);
src/lib/Radio/diffuse_predict.c:  cudaError_t err;
src/lib/Radio/diffuse_predict.c:  err=cudaSetDevice(card);
src/lib/Radio/diffuse_predict.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/diffuse_predict.c:  err=cudaMalloc((void**) &cohd, 8*sizeof(double));
src/lib/Radio/diffuse_predict.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/diffuse_predict.c:  err=cudaMalloc((void**) &factd, t->modes_n0*sizeof(float));
src/lib/Radio/diffuse_predict.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/diffuse_predict.c:  err=cudaMemcpy(factd, fact, t->modes_n0*sizeof(float), cudaMemcpyHostToDevice);
src/lib/Radio/diffuse_predict.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/diffuse_predict.c:   /* we copy u,v,w,l,m,n values to GPU and perform calculation per-baseline,
src/lib/Radio/diffuse_predict.c:    * CUDA threads parallelize over the modes : n0xn0 ~ large value */
src/lib/Radio/diffuse_predict.c:   cudakernel_calculate_shapelet_coherencies((float)t->u[ci]*freq0,(float)t->v[ci]*freq0,modesd,factd,t->modes_n0,(float)t->modes_beta,cohd);
src/lib/Radio/diffuse_predict.c:   err=cudaFree(modesd);
src/lib/Radio/diffuse_predict.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/diffuse_predict.c:   err=cudaMemcpy((double*)coh, cohd, sizeof(double)*8, cudaMemcpyDeviceToHost);
src/lib/Radio/diffuse_predict.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/diffuse_predict.c:  cudaDeviceSynchronize();
src/lib/Radio/diffuse_predict.c:  err=cudaFree(cohd);
src/lib/Radio/diffuse_predict.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/diffuse_predict.c:  err=cudaFree(factd);
src/lib/Radio/diffuse_predict.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/diffuse_predict.c:  err=cudaGetLastError();
src/lib/Radio/diffuse_predict.c:#endif /* HAVE_CUDA */
src/lib/Radio/diffuse_predict.c:   int Nbase, baseline_t *barr,  clus_source_t *carr, int M, double freq0, double fdelta, double tdelta, double dec0, double uvmin, double uvmax, int cid, int sh_n0, double sh_beta, complex double *Z, int Nt, int use_cuda) {
src/lib/Radio/diffuse_predict.c:#ifdef HAVE_CUDA
src/lib/Radio/diffuse_predict.c:#endif /* HAVE_CUDA */
src/lib/Radio/diffuse_predict.c:#ifndef HAVE_CUDA
src/lib/Radio/diffuse_predict.c:#else /* HAVE_CUDA */
src/lib/Radio/diffuse_predict.c:        if (!use_cuda) {
src/lib/Radio/diffuse_predict.c:            shapelet_pred_threadfn_cuda((void*)&threaddata[nth1]);
src/lib/Radio/diffuse_predict.c:#endif  /* HAVE_CUDA */
src/lib/Radio/diffuse_predict.c:#ifdef HAVE_CUDA
src/lib/Radio/predict_withbeam_cuda.c:#include "Dirac_GPUtune.h"
src/lib/Radio/predict_withbeam_cuda.c://#define CUDA_DEBUG
src/lib/Radio/predict_withbeam_cuda.c:checkCudaError(cudaError_t err, char *file, int line)
src/lib/Radio/predict_withbeam_cuda.c:#ifdef CUDA_DEBUG
src/lib/Radio/predict_withbeam_cuda.c:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Radio/predict_withbeam_cuda.c:/* struct to pass data to worker threads attached to GPUs */
src/lib/Radio/predict_withbeam_cuda.c:  taskhist *hst; /* for load balancing GPUs */
src/lib/Radio/predict_withbeam_cuda.c:  taskhist *hst; /* for load balancing GPUs */
src/lib/Radio/predict_withbeam_cuda.c:  cudaError_t err;
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMallocHost((void**)&xhost,sizeof(float)*N);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      fprintf(stderr,"%s: %d: cudaMallocHost error\n",__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&xc, N*sizeof(float));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(xc,xhost,N*sizeof(float),cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFreeHost(xhost);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  /* first, select a GPU, if total clusters < MAX_GPU_ID
src/lib/Radio/predict_withbeam_cuda.c:  if (t->M<=MAX_GPU_ID) {
src/lib/Radio/predict_withbeam_cuda.c:   card=select_work_gpu(MAX_GPU_ID,t->hst);
src/lib/Radio/predict_withbeam_cuda.c:   card=t->tid;/* note that max. no. of threads is still <= no. of GPUs */
src/lib/Radio/predict_withbeam_cuda.c:  cudaError_t err;
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaSetDevice(card);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  /* allocate memory in GPU */
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &cohd, t->Nbase*8*sizeof(double)); /* coherencies only for 1 cluster, Nf=1 */
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &ud, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &vd, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &wd, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(ud, t->u, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(vd, t->v, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(wd, t->w, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(barrd, t->barr, t->Nbase*sizeof(baseline_t), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &freqsd, t->Nf*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(freqsd, t->freqs, t->Nf*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &timed, t->tilesz*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(timed, t->time_utc, t->tilesz*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  cudakernel_convert_time(t->tilesz,timed);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &Nelemd, t->N*sizeof(int));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(Nelemd, t->Nelem, t->N*sizeof(int), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&xxd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&yyd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&zzd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&xx_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&yy_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&zz_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(xxd, xx_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(yyd, yy_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(zzd, zz_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMallocHost((void**)&tempdcoh,sizeof(complex double)*(size_t)t->Nbase*4);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaMalloc((void**)&beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaMalloc((void**)&elementd, t->N*8*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     /* copy cluster details to GPU */
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**)&styped, t->carr[ncl].N*sizeof(unsigned char));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &lld, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(lld, t->carr[ncl].ll, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &mmd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(mmd, t->carr[ncl].mm, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &nnd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(nnd, t->carr[ncl].nn, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sId, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sId, t->carr[ncl].sI, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sQd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sQd, t->carr[ncl].sQ, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sUd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sUd, t->carr[ncl].sU, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sVd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sVd, t->carr[ncl].sV, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(styped, t->carr[ncl].stype, t->carr[ncl].N*sizeof(unsigned char), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sI0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sI0d, t->carr[ncl].sI0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &f0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(f0d, t->carr[ncl].f0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idxd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idxd, t->carr[ncl].spec_idx, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idx1d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idx1d, t->carr[ncl].spec_idx1, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idx2d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idx2d, t->carr[ncl].spec_idx2, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sQ0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sQ0d, t->carr[ncl].sQ0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sU0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sU0d, t->carr[ncl].sU0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sV0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sV0d, t->carr[ncl].sV0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**)&dev_p, t->carr[ncl].N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_shapelet));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&modes, d->n0*d->n0*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_shapelet), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(modes, d->modes, d->n0*d->n0*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(&(d_p->modes), &modes, sizeof(double*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_gaussian));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_gaussian), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_disk));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_disk), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_ring));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_ring), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(dev_p, host_p, t->carr[ncl].N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,0);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,1);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,0);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,1);
src/lib/Radio/predict_withbeam_cuda.c:     cudakernel_coherencies(t->Nbase,t->N,t->tilesz,t->carr[ncl].N,t->Nf,ud,vd,wd,barrd,freqsd,beamd, elementd, 
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy((double*)tempdcoh, cohd, sizeof(double)*t->Nbase*8, cudaMemcpyDeviceToHost);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(&modes, &(d_p->modes), sizeof(double*), cudaMemcpyDeviceToHost);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(modes);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(dev_p);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(beamd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(elementd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(lld);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(mmd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(nnd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sId);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sQd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sUd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sVd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(rad);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(decd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(styped);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sI0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(f0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idxd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idx1d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idx2d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sQ0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sU0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sV0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFreeHost(tempdcoh);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(ud);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(vd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(wd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(cohd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(barrd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(freqsd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(longd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(latd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(timed);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(Nelemd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(xx_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(yy_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(zz_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(xxd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(yyd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(zzd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(pattern_phid);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(pattern_thetad);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(preambled);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  cudaDeviceSynchronize();
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaGetLastError(); 
src/lib/Radio/predict_withbeam_cuda.c:precalculate_coherencies_withbeam_gpu(double *u, double *v, double *w, complex double *x, int N,
src/lib/Radio/predict_withbeam_cuda.c:  int Ngpu=MAX_GPU_ID+1;
src/lib/Radio/predict_withbeam_cuda.c:  Nthb0=(M+Ngpu-1)/Ngpu;
src/lib/Radio/predict_withbeam_cuda.c:  /* setup threads : note: Ngpu is no of GPUs used */
src/lib/Radio/predict_withbeam_cuda.c:  if ((th_array=(pthread_t*)malloc((size_t)Ngpu*sizeof(pthread_t)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:  if ((threaddata=(thread_data_pred_t*)malloc((size_t)Ngpu*sizeof(thread_data_pred_t)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:  for (nth=0;  nth<Ngpu && ci<M; nth++) {
src/lib/Radio/predict_withbeam_cuda.c:  /* first, select a GPU, if total clusters < MAX_GPU_ID
src/lib/Radio/predict_withbeam_cuda.c:  if (t->M<=MAX_GPU_ID) {
src/lib/Radio/predict_withbeam_cuda.c:   card=select_work_gpu(MAX_GPU_ID,t->hst);
src/lib/Radio/predict_withbeam_cuda.c:   card=t->tid;/* note that max. no. of threads is still <= no. of GPUs */
src/lib/Radio/predict_withbeam_cuda.c:  cudaError_t err;
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaSetDevice(card);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  /* allocate memory in GPU */
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &cohd, t->Nbase*8*t->Nf*sizeof(double)); /* coherencies only for 1 cluster, Nf freq, used to store sum of clusters*/
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &ud, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &vd, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &wd, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(ud, t->u, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(vd, t->v, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(wd, t->w, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(barrd, t->barr, t->Nbase*sizeof(baseline_t), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &freqsd, t->Nf*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(freqsd, t->freqs, t->Nf*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &timed, t->tilesz*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(timed, t->time_utc, t->tilesz*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  cudakernel_convert_time(t->tilesz,timed);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &Nelemd, t->N*sizeof(int));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(Nelemd, t->Nelem, t->N*sizeof(int), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&xxd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&yyd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&zzd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&xx_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&yy_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&zz_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(xxd, xx_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(yyd, yy_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(zzd, zz_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMallocHost((void**)&xlocal,sizeof(double)*(size_t)t->Nbase*8*t->Nf);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaMalloc((void**)&beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaMalloc((void**)&elementd, t->N*8*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     /* copy cluster details to GPU */
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**)&styped, t->carr[ncl].N*sizeof(unsigned char));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &lld, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(lld, t->carr[ncl].ll, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &mmd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(mmd, t->carr[ncl].mm, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &nnd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(nnd, t->carr[ncl].nn, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sId, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sId, t->carr[ncl].sI, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sQd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sQd, t->carr[ncl].sQ, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sUd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sUd, t->carr[ncl].sU, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sVd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sVd, t->carr[ncl].sV, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(styped, t->carr[ncl].stype, t->carr[ncl].N*sizeof(unsigned char), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sI0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sI0d, t->carr[ncl].sI0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &f0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(f0d, t->carr[ncl].f0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idxd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idxd, t->carr[ncl].spec_idx, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idx1d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idx1d, t->carr[ncl].spec_idx1, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idx2d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idx2d, t->carr[ncl].spec_idx2, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sQ0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sQ0d, t->carr[ncl].sQ0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sU0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sU0d, t->carr[ncl].sU0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sV0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sV0d, t->carr[ncl].sV0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**)&dev_p, t->carr[ncl].N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_shapelet));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&modes, d->n0*d->n0*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_shapelet), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(modes, d->modes, d->n0*d->n0*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(&(d_p->modes), &modes, sizeof(double*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_gaussian));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_gaussian), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_disk));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_disk), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_ring));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_ring), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(dev_p, host_p, t->carr[ncl].N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,0);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,1);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,0);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,1);
src/lib/Radio/predict_withbeam_cuda.c:     cudakernel_coherencies(t->Nbase,t->N,t->tilesz,t->carr[ncl].N,t->Nf,ud,vd,wd,barrd,freqsd,beamd, elementd,
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(xlocal, cohd, sizeof(double)*t->Nbase*8*t->Nf, cudaMemcpyDeviceToHost);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(&modes, &(d_p->modes), sizeof(double*), cudaMemcpyDeviceToHost);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(modes);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(dev_p);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(beamd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(elementd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(lld);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(mmd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(nnd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sId);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sQd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sUd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sVd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(rad);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(decd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(styped);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sI0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(f0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idxd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idx1d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idx2d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sQ0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sU0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sV0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFreeHost(xlocal);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(ud);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(vd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(wd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(cohd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(barrd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(freqsd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(longd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(latd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(timed);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(Nelemd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(xx_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(yy_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(zz_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(xxd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(yyd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(zzd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(pattern_phid);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(pattern_thetad);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(preambled);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  cudaDeviceSynchronize();
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaGetLastError(); 
src/lib/Radio/predict_withbeam_cuda.c:predict_visibilities_multifreq_withbeam_gpu(double *u,double *v,double *w,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
src/lib/Radio/predict_withbeam_cuda.c:  /* oversubscribe GPU */
src/lib/Radio/predict_withbeam_cuda.c:  int Ngpu=MAX_GPU_ID+1;
src/lib/Radio/predict_withbeam_cuda.c:  Nthb0=(M+Ngpu-1)/Ngpu;
src/lib/Radio/predict_withbeam_cuda.c:  /* setup threads : note: Ngpu is no of GPUs used */
src/lib/Radio/predict_withbeam_cuda.c:  if ((th_array=(pthread_t*)malloc((size_t)Ngpu*sizeof(pthread_t)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:  if ((threaddata=(thread_data_pred_t*)malloc((size_t)Ngpu*sizeof(thread_data_pred_t)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:  if ((xlocal=(double*)calloc((size_t)Nbase*8*tilesz*Nchan*Ngpu,sizeof(double)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:  for (nth=0;  nth<Ngpu && ci<M; nth++) {
src/lib/Radio/predict_withbeam_cuda.c:  /* first, select a GPU, if total clusters < MAX_GPU_ID
src/lib/Radio/predict_withbeam_cuda.c:  if (t->M<=MAX_GPU_ID) {
src/lib/Radio/predict_withbeam_cuda.c:   card=select_work_gpu(MAX_GPU_ID,t->hst);
src/lib/Radio/predict_withbeam_cuda.c:   card=t->tid;/* note that max. no. of threads is still <= no. of GPUs */
src/lib/Radio/predict_withbeam_cuda.c:  cudaError_t err;
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaSetDevice(card);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  /* allocate memory in GPU */
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &cohd, t->Nbase*8*t->Nf*sizeof(double)); /* coherencies only for 1 cluster, Nf freq, used to store sum of clusters*/
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &ud, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &vd, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &wd, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(ud, t->u, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(vd, t->v, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(wd, t->w, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(barrd, t->barr, t->Nbase*sizeof(baseline_t), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &freqsd, t->Nf*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(freqsd, t->freqs, t->Nf*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &timed, t->tilesz*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(timed, t->time_utc, t->tilesz*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  cudakernel_convert_time(t->tilesz,timed);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &Nelemd, t->N*sizeof(int));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(Nelemd, t->Nelem, t->N*sizeof(int), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&xxd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&yyd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&zzd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&xx_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&yy_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&zz_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(xxd, xx_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(yyd, yy_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(zzd, zz_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMallocHost((void**)&xlocal,sizeof(double)*(size_t)t->Nbase*8*t->Nf);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaMalloc((void**)&beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaMalloc((void**)&elementd, t->N*8*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     /* copy cluster details to GPU */
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**)&styped, t->carr[ncl].N*sizeof(unsigned char));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &lld, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(lld, t->carr[ncl].ll, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &mmd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(mmd, t->carr[ncl].mm, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &nnd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(nnd, t->carr[ncl].nn, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &pd, t->N*8*t->carr[ncl].nchunk*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(pd, &(t->p[t->carr[ncl].p[0]]), t->N*8*t->carr[ncl].nchunk*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sId, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sId, t->carr[ncl].sI, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sQd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sQd, t->carr[ncl].sQ, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sUd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sUd, t->carr[ncl].sU, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sVd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sVd, t->carr[ncl].sV, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(styped, t->carr[ncl].stype, t->carr[ncl].N*sizeof(unsigned char), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sI0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sI0d, t->carr[ncl].sI0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &f0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(f0d, t->carr[ncl].f0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idxd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idxd, t->carr[ncl].spec_idx, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idx1d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idx1d, t->carr[ncl].spec_idx1, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idx2d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idx2d, t->carr[ncl].spec_idx2, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sQ0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sQ0d, t->carr[ncl].sQ0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sU0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sU0d, t->carr[ncl].sU0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sV0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sV0d, t->carr[ncl].sV0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**)&dev_p, t->carr[ncl].N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_shapelet));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&modes, d->n0*d->n0*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_shapelet), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(modes, d->modes, d->n0*d->n0*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(&(d_p->modes), &modes, sizeof(double*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_gaussian));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_gaussian), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_disk));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_disk), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_ring));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_ring), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(dev_p, host_p, t->carr[ncl].N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,0);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,1);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,1);
src/lib/Radio/predict_withbeam_cuda.c:     cudakernel_residuals(t->Nbase,t->N,t->tilesz,t->carr[ncl].N,t->Nf,ud,vd,wd,pd,t->carr[ncl].nchunk,barrd,freqsd,beamd, elementd,
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(xlocal, cohd, sizeof(double)*t->Nbase*8*t->Nf, cudaMemcpyDeviceToHost);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(&modes, &(d_p->modes), sizeof(double*), cudaMemcpyDeviceToHost);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(modes);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(dev_p);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(beamd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(elementd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(lld);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(mmd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(nnd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(pd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sId);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sQd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sUd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sVd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(rad);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(decd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(styped);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sI0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(f0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idxd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idx1d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idx2d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sQ0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sU0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sV0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFreeHost(xlocal);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(ud);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(vd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(wd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(cohd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(barrd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(freqsd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(longd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(latd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(timed);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(Nelemd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(xx_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(yy_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(zz_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(xxd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(yyd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(zzd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(pattern_phid);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(pattern_thetad);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(preambled);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  cudaDeviceSynchronize();
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaGetLastError(); 
src/lib/Radio/predict_withbeam_cuda.c:  /* first, select a GPU, if total threads > MAX_GPU_ID
src/lib/Radio/predict_withbeam_cuda.c:  if (t->tid>MAX_GPU_ID) {
src/lib/Radio/predict_withbeam_cuda.c:   card=select_work_gpu(MAX_GPU_ID,t->hst);
src/lib/Radio/predict_withbeam_cuda.c:   card=t->tid;/* note that max. no. of threads is still <= no. of GPUs */
src/lib/Radio/predict_withbeam_cuda.c:  cudaError_t err;
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaSetDevice(card);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  /* allocate memory in GPU */
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &xd, t->Nb*8*t->Nf*sizeof(double)); /* coherencies only for Nb baselines, Nf freqs */
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &barrd, t->Nb*sizeof(baseline_t));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &pd, t->N*8*t->nchunk*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(barrd, &(t->barr[t->boff]), t->Nb*sizeof(baseline_t), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(pd, t->pinv, t->N*8*t->nchunk*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMemcpy(&xd[cf*8*t->Nb], &(t->x[cf*8*t->Nbase+t->boff*8]), t->Nb*8*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  cudakernel_correct_residuals(t->Nbase,t->N,t->Nb,t->boff,t->Nf,t->nchunk,xd,pd,barrd);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMemcpy(&(t->x[cf*8*t->Nbase+t->boff*8]), &xd[cf*8*t->Nb], t->Nb*8*sizeof(double), cudaMemcpyDeviceToHost);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(xd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(barrd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(pd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  cudaDeviceSynchronize();
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaGetLastError();
src/lib/Radio/predict_withbeam_cuda.c:calculate_residuals_multifreq_withbeam_gpu(double *u,double *v,double *w,double *p,double *x,int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
src/lib/Radio/predict_withbeam_cuda.c:  /* oversubsribe GPU */
src/lib/Radio/predict_withbeam_cuda.c:  int Ngpu=MAX_GPU_ID+1;
src/lib/Radio/predict_withbeam_cuda.c:  Nthb0=(M+Ngpu-1)/Ngpu;
src/lib/Radio/predict_withbeam_cuda.c:  /* setup threads : note: Ngpu is no of GPUs used */
src/lib/Radio/predict_withbeam_cuda.c:  if ((th_array=(pthread_t*)malloc((size_t)Ngpu*sizeof(pthread_t)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:  if ((threaddata=(thread_data_pred_t*)malloc((size_t)Ngpu*sizeof(thread_data_pred_t)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:  if ((xlocal=(double*)calloc((size_t)Nbase*8*tilesz*Nchan*Ngpu,sizeof(double)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:  for (nth=0;  nth<Ngpu && ci<M; nth++) {
src/lib/Radio/predict_withbeam_cuda.c:   /* divide x[] over GPUs */
src/lib/Radio/predict_withbeam_cuda.c:   Nthb0=(Nbase1+Ngpu-1)/Ngpu;
src/lib/Radio/predict_withbeam_cuda.c:   if ((threaddata_corr=(thread_data_corr_t*)malloc((size_t)Ngpu*sizeof(thread_data_corr_t)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:   for (nth=0;  nth<Ngpu && ci<Nbase1; nth++) {
src/lib/Radio/predict_withbeam_cuda.c:  /* first, select a GPU, if total clusters < MAX_GPU_ID
src/lib/Radio/predict_withbeam_cuda.c:  if (t->M<=MAX_GPU_ID) {
src/lib/Radio/predict_withbeam_cuda.c:   card=select_work_gpu(MAX_GPU_ID,t->hst);
src/lib/Radio/predict_withbeam_cuda.c:   card=t->tid;/* note that max. no. of threads is still <= no. of GPUs */
src/lib/Radio/predict_withbeam_cuda.c:  cudaError_t err;
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaSetDevice(card);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  /* allocate memory in GPU */
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &cohd, t->Nbase*8*t->Nf*sizeof(double)); /* coherencies only for 1 cluster, Nf freq, used to store sum of clusters*/
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &ud, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &vd, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &wd, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(ud, t->u, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(vd, t->v, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(wd, t->w, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(barrd, t->barr, t->Nbase*sizeof(baseline_t), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &freqsd, t->Nf*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(freqsd, t->freqs, t->Nf*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &timed, t->tilesz*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(timed, t->time_utc, t->tilesz*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  cudakernel_convert_time(t->tilesz,timed);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &Nelemd, t->N*sizeof(int));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(Nelemd, t->Nelem, t->N*sizeof(int), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&xxd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&yyd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&zzd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&xx_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&yy_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&zz_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(xxd, xx_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(yyd, yy_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(zzd, zz_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMallocHost((void**)&xlocal,sizeof(double)*(size_t)t->Nbase*8*t->Nf);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaMalloc((void**)&beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaMalloc((void**)&elementd, t->N*8*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     /* copy cluster details to GPU */
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**)&styped, t->carr[ncl].N*sizeof(unsigned char));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &lld, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(lld, t->carr[ncl].ll, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &mmd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(mmd, t->carr[ncl].mm, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &nnd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(nnd, t->carr[ncl].nn, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &pd, t->N*8*t->carr[ncl].nchunk*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(pd, &(t->p[t->carr[ncl].p[0]]), t->N*8*t->carr[ncl].nchunk*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sId, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sId, t->carr[ncl].sI, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sQd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sQd, t->carr[ncl].sQ, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sUd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sUd, t->carr[ncl].sU, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sVd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sVd, t->carr[ncl].sV, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(styped, t->carr[ncl].stype, t->carr[ncl].N*sizeof(unsigned char), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sI0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sI0d, t->carr[ncl].sI0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &f0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(f0d, t->carr[ncl].f0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idxd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idxd, t->carr[ncl].spec_idx, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idx1d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idx1d, t->carr[ncl].spec_idx1, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idx2d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idx2d, t->carr[ncl].spec_idx2, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sQ0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sQ0d, t->carr[ncl].sQ0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sU0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sU0d, t->carr[ncl].sU0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sV0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sV0d, t->carr[ncl].sV0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**)&dev_p, t->carr[ncl].N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_shapelet));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&modes, d->n0*d->n0*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_shapelet), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(modes, d->modes, d->n0*d->n0*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(&(d_p->modes), &modes, sizeof(double*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_gaussian));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_gaussian), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_disk));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_disk), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_ring));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_ring), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(dev_p, host_p, t->carr[ncl].N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,0);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,1);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,1);
src/lib/Radio/predict_withbeam_cuda.c:     cudakernel_residuals(t->Nbase,t->N,t->tilesz,t->carr[ncl].N,t->Nf,ud,vd,wd,pd,t->carr[ncl].nchunk,barrd,freqsd,beamd, elementd,
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(xlocal, cohd, sizeof(double)*t->Nbase*8*t->Nf, cudaMemcpyDeviceToHost);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(&modes, &(d_p->modes), sizeof(double*), cudaMemcpyDeviceToHost);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(modes);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(dev_p);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(beamd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(elementd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(lld);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(mmd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(nnd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(pd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sId);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sQd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sUd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sVd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(rad);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(decd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(styped);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sI0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(f0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idxd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idx1d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idx2d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sQ0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sU0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sV0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFreeHost(xlocal);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(ud);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(vd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(wd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(cohd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(barrd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(freqsd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(longd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(latd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(timed);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(Nelemd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(xx_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(yy_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(zz_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(xxd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(yyd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(zzd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(pattern_phid);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(pattern_thetad);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(preambled);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  cudaDeviceSynchronize();
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaGetLastError(); 
src/lib/Radio/predict_withbeam_cuda.c:  almost same as calculate_residuals_multifreq_withbeam_gpu(), but ignorelist is used to ignore prediction of some clusters, and predict/add/subtract options based on simulation mode add_to_data
src/lib/Radio/predict_withbeam_cuda.c:predict_visibilities_withsol_withbeam_gpu(double *u,double *v,double *w,double *p,double *x, int *ignorelist, int N,int Nbase,int tilesz,baseline_t *barr, clus_source_t *carr, int M,double *freqs,int Nchan, double fdelta,double tdelta, double dec0,
src/lib/Radio/predict_withbeam_cuda.c:  /* oversubsribe GPU */
src/lib/Radio/predict_withbeam_cuda.c:  int Ngpu=MAX_GPU_ID+1;
src/lib/Radio/predict_withbeam_cuda.c:  Nthb0=(M+Ngpu-1)/Ngpu;
src/lib/Radio/predict_withbeam_cuda.c:  /* setup threads : note: Ngpu is no of GPUs used */
src/lib/Radio/predict_withbeam_cuda.c:  if ((th_array=(pthread_t*)malloc((size_t)Ngpu*sizeof(pthread_t)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:  if ((threaddata=(thread_data_pred_t*)malloc((size_t)Ngpu*sizeof(thread_data_pred_t)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:  if ((xlocal=(double*)calloc((size_t)Nbase*8*tilesz*Nchan*Ngpu,sizeof(double)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:  for (nth=0;  nth<Ngpu && ci<M; nth++) {
src/lib/Radio/predict_withbeam_cuda.c:   /* divide x[] over GPUs */
src/lib/Radio/predict_withbeam_cuda.c:   Nthb0=(Nbase1+Ngpu-1)/Ngpu;
src/lib/Radio/predict_withbeam_cuda.c:   if ((threaddata_corr=(thread_data_corr_t*)malloc((size_t)Ngpu*sizeof(thread_data_corr_t)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:   for (nth=0;  nth<Ngpu && ci<Nbase1; nth++) {
src/lib/Radio/predict_withbeam_cuda.c:  /* first, select a GPU, if total clusters < MAX_GPU_ID
src/lib/Radio/predict_withbeam_cuda.c:  if (t->M<=MAX_GPU_ID) {
src/lib/Radio/predict_withbeam_cuda.c:   card=select_work_gpu(MAX_GPU_ID,t->hst);
src/lib/Radio/predict_withbeam_cuda.c:   card=t->tid;/* note that max. no. of threads is still <= no. of GPUs */
src/lib/Radio/predict_withbeam_cuda.c:  cudaError_t err;
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaSetDevice(card);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  /* allocate memory in GPU */
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &cohd, t->Nf*t->Nbase*8*sizeof(double)); /* coherencies only for 1 cluster, Nf>=1 */
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &barrd, t->Nbase*sizeof(baseline_t));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &ud, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &vd, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &wd, t->Nbase*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(ud, t->u, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(vd, t->v, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(wd, t->w, t->Nbase*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(barrd, t->barr, t->Nbase*sizeof(baseline_t), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &freqsd, t->Nf*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(freqsd, t->freqs, t->Nf*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &timed, t->tilesz*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(timed, t->time_utc, t->tilesz*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  cudakernel_convert_time(t->tilesz,timed);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**) &Nelemd, t->N*sizeof(int));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(Nelemd, t->Nelem, t->N*sizeof(int), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&xxd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&yyd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMalloc((void**)&zzd, t->N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&xx_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&yy_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaMalloc((void**)&zz_p[ci], (t->Nelem[ci]+(t->bf_type==STAT_TILE?HBA_TILE_SIZE:0))*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(xxd, xx_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(yyd, yy_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMemcpy(zzd, zz_p, t->N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaMallocHost((void**)&tempdcoh,sizeof(complex double)*(size_t)t->Nbase*4*t->Nf);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaMalloc((void**)&beamd, t->N*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaMalloc((void**)&elementd, t->N*8*t->tilesz*t->carr[ncl].N*t->Nf*sizeof(float));
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     /* copy cluster details to GPU */
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**)&styped, t->carr[ncl].N*sizeof(unsigned char));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &lld, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(lld, t->carr[ncl].ll, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &mmd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(mmd, t->carr[ncl].mm, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &nnd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(nnd, t->carr[ncl].nn, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sId, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sId, t->carr[ncl].sI, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sQd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sQd, t->carr[ncl].sQ, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sUd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sUd, t->carr[ncl].sU, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sVd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sVd, t->carr[ncl].sV, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(styped, t->carr[ncl].stype, t->carr[ncl].N*sizeof(unsigned char), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sI0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sI0d, t->carr[ncl].sI0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &f0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(f0d, t->carr[ncl].f0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idxd, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idxd, t->carr[ncl].spec_idx, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idx1d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idx1d, t->carr[ncl].spec_idx1, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &spec_idx2d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(spec_idx2d, t->carr[ncl].spec_idx2, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sQ0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sQ0d, t->carr[ncl].sQ0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sU0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sU0d, t->carr[ncl].sU0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**) &sV0d, t->carr[ncl].N*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(sV0d, t->carr[ncl].sV0, t->carr[ncl].N*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMalloc((void**)&dev_p, t->carr[ncl].N*sizeof(int*));
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_shapelet));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&modes, d->n0*d->n0*sizeof(double));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_shapelet), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(modes, d->modes, d->n0*d->n0*sizeof(double), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(&(d_p->modes), &modes, sizeof(double*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_gaussian));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_gaussian), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_disk));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_disk), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMalloc((void**)&host_p[cj], sizeof(exinfo_ring));
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(host_p[cj], d, sizeof(exinfo_ring), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy(dev_p, host_p, t->carr[ncl].N*sizeof(int*), cudaMemcpyHostToDevice);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,0);
src/lib/Radio/predict_withbeam_cuda.c:       cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,1);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,0);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,0);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_tile_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->b_ra0,(float)t->b_dec0,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:        cudakernel_array_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,Nelemd,xxd,yyd,zzd,rad,decd,(float)t->ph_ra0,(float)t->ph_dec0,(float)t->ph_freq0,beamd,1);
src/lib/Radio/predict_withbeam_cuda.c:      cudakernel_element_beam(t->N,t->tilesz,t->carr[ncl].N,t->Nf,freqsd,longd,latd,timed,rad,decd,t->ecoeff->Nmodes,t->ecoeff->M,t->ecoeff->beta,pattern_phid,pattern_thetad,preambled,elementd,1);
src/lib/Radio/predict_withbeam_cuda.c:     cudakernel_coherencies(t->Nbase,t->N,t->tilesz,t->carr[ncl].N,t->Nf,ud,vd,wd,barrd,freqsd,beamd, elementd,
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaMemcpy((double*)tempdcoh, cohd, sizeof(double)*t->Nbase*8*t->Nf, cudaMemcpyDeviceToHost);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaMemcpy(&modes, &(d_p->modes), sizeof(double*), cudaMemcpyDeviceToHost);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(modes);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:          err=cudaFree(host_p[cj]);
src/lib/Radio/predict_withbeam_cuda.c:          checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(dev_p);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(beamd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(elementd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(lld);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(mmd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(nnd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sId);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sQd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sUd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sVd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(rad);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:      err=cudaFree(decd);
src/lib/Radio/predict_withbeam_cuda.c:      checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(styped);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sI0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(f0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idxd);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idx1d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(spec_idx2d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sQ0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sU0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:     err=cudaFree(sV0d);
src/lib/Radio/predict_withbeam_cuda.c:     checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFreeHost(tempdcoh);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(ud);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(vd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(wd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(cohd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(barrd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(freqsd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(longd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(latd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(timed);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(Nelemd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(xx_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(yy_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:    err=cudaFree(zz_p[ci]);
src/lib/Radio/predict_withbeam_cuda.c:    checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(xxd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(yyd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaFree(zzd);
src/lib/Radio/predict_withbeam_cuda.c:  checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(pattern_phid);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(pattern_thetad);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:   err=cudaFree(preambled);
src/lib/Radio/predict_withbeam_cuda.c:   checkCudaError(err,__FILE__,__LINE__);
src/lib/Radio/predict_withbeam_cuda.c:  cudaDeviceSynchronize();
src/lib/Radio/predict_withbeam_cuda.c:  err=cudaGetLastError(); 
src/lib/Radio/predict_withbeam_cuda.c:precalculate_coherencies_multifreq_withbeam_gpu(double *u, double *v, double *w, complex double *x, int N,
src/lib/Radio/predict_withbeam_cuda.c:  int Ngpu=MAX_GPU_ID+1;
src/lib/Radio/predict_withbeam_cuda.c:  Nthb0=(M+Ngpu-1)/Ngpu;
src/lib/Radio/predict_withbeam_cuda.c:  /* setup threads : note: Ngpu is no of GPUs used */
src/lib/Radio/predict_withbeam_cuda.c:  if ((th_array=(pthread_t*)malloc((size_t)Ngpu*sizeof(pthread_t)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:  if ((threaddata=(thread_data_pred_t*)malloc((size_t)Ngpu*sizeof(thread_data_pred_t)))==0) {
src/lib/Radio/predict_withbeam_cuda.c:  for (nth=0;  nth<Ngpu && ci<M; nth++) {
src/lib/Radio/predict_model.cu:#include "cuda.h"
src/lib/Radio/predict_model.cu://#define CUDA_DBG
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:checkCudaError(cudaError_t err, const char *file, int line)
src/lib/Radio/predict_model.cu:    fprintf(stderr,"GPU (CUDA): %s %s %d\n", cudaGetErrorString(err),file,line);
src/lib/Radio/predict_model.cu:cudakernel_array_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  cudaError_t error;
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  cudaDeviceSynchronize();
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  if(error != cudaSuccess) {
src/lib/Radio/predict_model.cu:    // print the CUDA error message and exit
src/lib/Radio/predict_model.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Radio/predict_model.cu:  precalculate station beam, same as cudakernel_array_beam,
src/lib/Radio/predict_model.cu:cudakernel_tile_array_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  cudaError_t error;
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  cudaDeviceSynchronize();
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  if(error != cudaSuccess) {
src/lib/Radio/predict_model.cu:    // print the CUDA error message and exit
src/lib/Radio/predict_model.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Radio/predict_model.cu:cudakernel_element_beam(int N, int T, int K, int F, double *freqs, float *longitude, float *latitude,
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  cudaError_t error;
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  //cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
src/lib/Radio/predict_model.cu:  cudaDeviceSynchronize();
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  if(error != cudaSuccess) {
src/lib/Radio/predict_model.cu:    // print the CUDA error message and exit
src/lib/Radio/predict_model.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Radio/predict_model.cu:cudakernel_coherencies(int B, int N, int T, int K, int F, double *u, double *v, double *w,baseline_t *barr, double *freqs, float *beam, float *element, double *ll, double *mm, double *nn, double *sI, double *sQ, double *sU, double *sV,
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  cudaError_t error;
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  cudaDeviceSynchronize();
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  if(error != cudaSuccess) {
src/lib/Radio/predict_model.cu:    // print the CUDA error message and exit
src/lib/Radio/predict_model.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Radio/predict_model.cu:cudakernel_residuals(int B, int N, int T, int K, int F, double *u, double *v, double *w, double *p, int nchunk, baseline_t *barr, double *freqs, float *beam, float *element, double *ll, double *mm, double *nn, double *sI, double *sQ, double *sU, double *sV,
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  cudaError_t error;
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  cudaDeviceSynchronize();
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  if(error != cudaSuccess) {
src/lib/Radio/predict_model.cu:    // print the CUDA error message and exit
src/lib/Radio/predict_model.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Radio/predict_model.cu:cudakernel_correct_residuals(int B, int N, int Nb, int boff, int F, int nchunk, double *x, double *p, baseline_t *barr) {
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  cudaError_t error;
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  cudaDeviceSynchronize();
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  if(error != cudaSuccess) {
src/lib/Radio/predict_model.cu:    // print the CUDA error message and exit
src/lib/Radio/predict_model.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Radio/predict_model.cu:cudakernel_convert_time(int T, double *time_utc) {
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  cudaError_t error;
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  cudaDeviceSynchronize();
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  if(error != cudaSuccess) {
src/lib/Radio/predict_model.cu:    // print the CUDA error message and exit
src/lib/Radio/predict_model.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);
src/lib/Radio/predict_model.cu:#define CUDA_DBG
src/lib/Radio/predict_model.cu:cudakernel_calculate_shapelet_coherencies(float u, float v, float *modes, float *fact, int n0, float beta, double *coh) {
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  cudaError_t error;
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  cudaMalloc((void**)&J_C_J, 8*sizeof(float)*BlocksPerGrid);
src/lib/Radio/predict_model.cu:  cudaMemset(J_C_J, 0, 8*sizeof(float)*BlocksPerGrid);
src/lib/Radio/predict_model.cu:  cudaFree(J_C_J);
src/lib/Radio/predict_model.cu:#ifdef CUDA_DBG
src/lib/Radio/predict_model.cu:  error = cudaGetLastError();
src/lib/Radio/predict_model.cu:  if(error != cudaSuccess) {
src/lib/Radio/predict_model.cu:    // print the CUDA error message and exit
src/lib/Radio/predict_model.cu:    fprintf(stderr,"CUDA error: %s :%s: %d\n", cudaGetErrorString(error),__FILE__,__LINE__);

```
