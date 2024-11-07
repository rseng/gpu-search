# https://github.com/eltevo/StePS

```console
StePS/Template-LinuxGCC-Makefile:#------------------------------- GPU and precision options for the force calculation
StePS/Template-LinuxGCC-Makefile:USING_CUDA = NO
StePS/Template-LinuxGCC-Makefile:#------------------------------- Location of the CUDA Toolkit
StePS/Template-LinuxGCC-Makefile:CUDA_PATH       ?= /usr/local/cuda-12.6
StePS/Template-LinuxGCC-Makefile:ifeq ($(USING_CUDA), YES)
StePS/Template-LinuxGCC-Makefile:#------------------------------- Location of the CUDA Toolkit
StePS/Template-LinuxGCC-Makefile:OPT += -DUSE_CUDA
StePS/Template-LinuxGCC-Makefile:BUILD_NUMBER_LDFLAGS += -DPROGRAMNAME='"StePS_CUDA"'
StePS/Template-LinuxGCC-Makefile:NVCC = $(CUDA_PATH)/bin/nvcc -ccbin
StePS/Template-LinuxGCC-Makefile:CUDAFLAGS = -Xcompiler -fopenmp -lineinfo --compiler-options --std=${CPPSTD} --compiler-options -Wall --compiler-options -ansi -O3 -lm -Xcompiler -pthread
StePS/Template-LinuxGCC-Makefile:CUDALFLAGS = -Xcompiler -fopenmp -lineinfo --compiler-options --std=${CPPSTD} --compiler-options -Wall --compiler-options -ansi -O3 -lm -Xcompiler -pthread -Xcompiler \'-Wl\\,-rpath\' -Xcompiler \'-Wl\\,$(MPI_LIBS)\' -Xcompiler \'-Wl\\,--enable-new-dtags\' -lmpi_cxx -lmpi -lhwloc
StePS/Template-LinuxGCC-Makefile:CUDA_INC = -I$(CUDA_PATH)/include
StePS/Template-LinuxGCC-Makefile:SRC += $(SRC_DIR)/forces_cuda.cu
StePS/Template-LinuxGCC-Makefile:OBJ += $(BUILD_DIR)/forces_cuda.o
StePS/Template-LinuxGCC-Makefile:PROG = $(BUILD_DIR)/StePS_CUDA
StePS/Template-LinuxGCC-Makefile:	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $(PROG) $(OBJ) $(HDF5_INC) $(HDF5_LIBS) $(MPI_LIBS)
StePS/Template-LinuxGCC-Makefile:	$(CXX) $(CFLAGS) $(CUDA_INC) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(HDF5_LIBS) $(HDF5_INC) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $@ -c $<
StePS/Template-LinuxGCC-Makefile:	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) $(MPI_LIBS) $(MPI_INC) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $@ -c $<
StePS/CHANGELOG.md:- Parallelized with MPI, OpenMP and CUDA
StePS/README.txt:- Written in C++ with MPI, OpenMP and CUDA parallelization.
StePS/README.txt:- Able to use multiple GPUs simultaneously in a large computing cluster.
StePS/README.txt:		-CUDA (https://developer.nvidia.com/cuda-downloads) Use only if you want to accelerate the simulations with Nvidia GPUs. Note that only GNU/Linux is supported for CUDA.
StePS/README.txt:			If this is set, the code will use 32bit precision in the force calculation, otherwise 64bit calculation will be used. The 32bit force calculation is ∼ 32 times faster on Nvidia GTX GPUs compared to the 64bit force calculation, and it uses half as much memory. The speedup with Nvidia Tesla cards by using single precision is ~2.
StePS/README.txt:If you compiled the code with CUDA, you can simply run it by typing:
StePS/README.txt:	$ export OMP_NUM_THREADS=<Number of GPUs per tasks>
StePS/README.txt:  $ mpirun -np <number of MPI tasks> ./build/StePS_CUDA <parameterfile> <Number of GPUs per tasks>
StePS/README.txt:    $ ./build/StePS_CUDA ./examples/LCDM_SP_1860_com_VOI100.param 1
StePS/README.txt:    command for GPU accelerated simulation (on one node with one GPU), or with
StePS/README.txt:  Since this simulation contains ~1.8million particles, using GPU acceleration is highly advised.
StePS/README.txt:    $ ./build/StePS_CUDA ./examples/LCDM_SP_1860_noncom_VOI100.param 1
StePS/README.txt:    for GPU accelerated simulation (on one node with one GPU), or with
StePS/Template-LinuxICC-Makefile:#------------------------------- GPU and precision options for the force calculation
StePS/Template-LinuxICC-Makefile:USING_CUDA = YES
StePS/Template-LinuxICC-Makefile:#------------------------------- Location of the CUDA Toolkit
StePS/Template-LinuxICC-Makefile:CUDA_PATH       ?= /opt/apps/cuda/11.0
StePS/Template-LinuxICC-Makefile:ifeq ($(USING_CUDA), YES)
StePS/Template-LinuxICC-Makefile:#------------------------------- Location of the CUDA Toolkit
StePS/Template-LinuxICC-Makefile:OPT += -DUSE_CUDA
StePS/Template-LinuxICC-Makefile:BUILD_NUMBER_LDFLAGS += -DPROGRAMNAME='"StePS_CUDA"'
StePS/Template-LinuxICC-Makefile:NVCC = $(CUDA_PATH)/bin/nvcc -ccbin
StePS/Template-LinuxICC-Makefile:CUDAFLAGS = -Xcompiler -fopenmp -lineinfo --compiler-options -std=gnu++98 --compiler-options -Wall --compiler-options -ansi -O3 -lm -Xcompiler -pthread
StePS/Template-LinuxICC-Makefile:CUDALFLAGS = -Xcompiler -fopenmp -lineinfo --compiler-options -std=gnu++98 --compiler-options -Wall --compiler-options -ansi -O3 -lm -Xcompiler -pthread -Xcompiler \'-Wl\\,-rpath\' -Xcompiler \'-Wl\\,$(MPI_LIBS)\' -Xcompiler \'-Wl\\,--enable-new-dtags\' -lmpi_cxx -lmpi -lhwloc
StePS/Template-LinuxICC-Makefile:CUDA_INC = -I$(CUDA_PATH)/include
StePS/Template-LinuxICC-Makefile:SRC += $(SRC_DIR)/forces_cuda.cu
StePS/Template-LinuxICC-Makefile:OBJ += $(BUILD_DIR)/forces_cuda.o
StePS/Template-LinuxICC-Makefile:PROG = $(BUILD_DIR)/StePS_CUDA
StePS/Template-LinuxICC-Makefile:	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $(PROG) $(OBJ) $(HDF5_INC) $(HDF5_LIBS) $(MPI_LIBS)
StePS/Template-LinuxICC-Makefile:	$(CXX) $(CFLAGS) $(CUDA_INC) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(HDF5_LIBS) $(HDF5_INC) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $@ -c $<
StePS/Template-LinuxICC-Makefile:	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) $(MPI_LIBS) $(MPI_INC) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $@ -c $<
StePS/src/main.cc:int n_GPU; //number of cuda capable GPUs
StePS/src/main.cc:	#ifdef USE_CUDA
StePS/src/main.cc:		printf("\tUsing CUDA capable GPUs for force calculation.\n");
StePS/src/main.cc:	#ifndef USE_CUDA
StePS/src/main.cc:			#ifndef USE_CUDA
StePS/src/main.cc:				fprintf(stderr, "Call with: ./%s  <parameter file> \'i\', where \'i\' is the number of the CUDA capable GPUs per node.\nif \'i\' is not set, than one GPU per MPI task will be used.\n", PROGRAMNAME);
StePS/src/main.cc:	#ifdef USE_CUDA
StePS/src/main.cc:		n_GPU = atoi( argv[2] );
StePS/src/main.cc:			printf("Using %i cuda capable GPU per MPI task.\n\n", n_GPU);
StePS/src/main.cc:		n_GPU = 1;
StePS/src/main.cc:		#ifdef USE_CUDA
StePS/src/main.cc:		printf("Total GPU time = %fh\n", (SIM_omp_end_time-SIM_omp_start_time)*numtasks*n_GPU/3600.0);
StePS/src/step.cc:	#ifdef USE_CUDA
StePS/src/step.cc:		omp_set_num_threads(n_GPU);	// Use n_GPU threads
StePS/src/forces_cuda.cu:#include "cuda_runtime.h"
StePS/src/forces_cuda.cu:cudaError_t forces_cuda(REAL*x, REAL*F, int n_GPU, int ID_min, int ID_max);
StePS/src/forces_cuda.cu:cudaError_t forces_periodic_cuda(REAL*x, REAL*F, int n_GPU, int ID_min, int ID_max);
StePS/src/forces_cuda.cu:	forces_cuda(x, F, n_GPU, ID_min, ID_max);
StePS/src/forces_cuda.cu:	forces_periodic_cuda(x, F, n_GPU, ID_min, ID_max);
StePS/src/forces_cuda.cu:cudaError_t forces_cuda(REAL*x, REAL*F, int n_GPU, int ID_min, int ID_max) //Force calculation on GPU
StePS/src/forces_cuda.cu:	int GPU_ID, nthreads;
StePS/src/forces_cuda.cu:	int N_GPU, GPU_index_min; //number of particles in this GPU, the first particles index
StePS/src/forces_cuda.cu:	cudaError_t cudaStatus;
StePS/src/forces_cuda.cu:	cudaStatus = cudaSuccess;
StePS/src/forces_cuda.cu:	// Get the number of CUDA devices.
StePS/src/forces_cuda.cu:	cudaGetDeviceCount(&numDevices);
StePS/src/forces_cuda.cu:	if(numDevices<n_GPU)
StePS/src/forces_cuda.cu:			fprintf(stderr, "Error: MPI rank %i: Cannot allocate %i GPUs, because only one is available\n", rank, n_GPU);
StePS/src/forces_cuda.cu:			fprintf(stderr, "Error: MPI rank %i: Cannot allocate %i GPUs, because only %i are available\n", rank, n_GPU, numDevices);
StePS/src/forces_cuda.cu:		n_GPU = numDevices;
StePS/src/forces_cuda.cu:		printf("Number of GPUs set to %i\n", n_GPU);
StePS/src/forces_cuda.cu:		fprintf(stderr, "MPI task %i: failed to allocate memory for xx_tmp (for CUDA force canculation).\n", rank);
StePS/src/forces_cuda.cu:		fprintf(stderr, "MPI task %i: failed to allocate memory for xy_tmp (for CUDA force canculation).\n", rank);
StePS/src/forces_cuda.cu:		fprintf(stderr, "MPI task %i: failed to allocate memory for xz_tmp (for CUDA force canculation).\n", rank);
StePS/src/forces_cuda.cu:	omp_set_num_threads(n_GPU);	// Use n_GPU threads
StePS/src/forces_cuda.cu:#pragma omp parallel default(shared) private(GPU_ID, F_tmp, i, j, mprocessors, cudaStatus, N_GPU, GPU_index_min, nthreads, dev_xx, dev_xy, dev_xz, dev_M, dev_SOFT_LENGTH, dev_F)
StePS/src/forces_cuda.cu:		GPU_ID = omp_get_thread_num(); //thread ID = GPU_ID
StePS/src/forces_cuda.cu:		if(GPU_ID == 0)
StePS/src/forces_cuda.cu:			N_GPU = (ID_max-ID_min+1)/n_GPU+(ID_max-ID_min+1)%n_GPU;
StePS/src/forces_cuda.cu:			GPU_index_min = ID_min;
StePS/src/forces_cuda.cu:			N_GPU = (ID_max-ID_min+1)/n_GPU;
StePS/src/forces_cuda.cu:			GPU_index_min = ID_min + (ID_max-ID_min+1)%n_GPU+N_GPU*GPU_ID;
StePS/src/forces_cuda.cu:		if(!(F_tmp = (REAL*)malloc(3 * N_GPU * sizeof(REAL))))
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI task %i: failed to allocate memory for F_tmp (for CUDA force canculation).\n", rank);
StePS/src/forces_cuda.cu:		for(i=0; i < N_GPU; i++)
StePS/src/forces_cuda.cu:		//Checking for the GPU
StePS/src/forces_cuda.cu:		cudaDeviceGetAttribute(&mprocessors, cudaDevAttrMultiProcessorCount, GPU_ID);
StePS/src/forces_cuda.cu:		if(GPU_ID == 0)
StePS/src/forces_cuda.cu:			printf("MPI task %i: GPU force calculation.\n Number of GPUs: %i\n Number of OMP threads: %i\n Number of threads per GPU: %i\n", rank, n_GPU, nthreads, 32*mprocessors*BLOCKSIZE);
StePS/src/forces_cuda.cu:		cudaStatus = cudaSetDevice(GPU_ID); //selecting the GPU
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		// Allocate GPU buffers for coordinate and mass vectors
StePS/src/forces_cuda.cu:		cudaStatus = cudaMalloc((void**)&dev_xx, N * sizeof(REAL));
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: xx cudaMalloc failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMalloc((void**)&dev_xy, N * sizeof(REAL));
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: xy cudaMalloc failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMalloc((void**)&dev_xz, N * sizeof(REAL));
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:                	fprintf(stderr, "MPI rank %i: GPU%i: xz cudaMalloc failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMalloc((void**)&dev_M, N * sizeof(REAL));
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: M cudaMalloc failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		// Allocate GPU buffers for the softening vector
StePS/src/forces_cuda.cu:		cudaStatus = cudaMalloc((void**)&dev_SOFT_LENGTH, N * sizeof(REAL)); //v0.3.7.1
StePS/src/forces_cuda.cu:                if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:                        fprintf(stderr, "MPI rank %i: GPU%i: SOFT_LENGTH cudaMalloc failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		// Allocate GPU buffers for force vectors
StePS/src/forces_cuda.cu:		cudaStatus = cudaMalloc((void**)&dev_F, 3 * N_GPU * sizeof(REAL));
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: F cudaMalloc failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		// Copy input vectors from host memory to GPU buffers.
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(dev_xx, xx_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xx in failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(dev_xy, xy_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xy in failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(dev_xz, xz_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xz in failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(dev_M, M, N * sizeof(REAL), cudaMemcpyHostToDevice);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy M in failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(dev_SOFT_LENGTH, SOFT_LENGTH, N * sizeof(REAL), cudaMemcpyHostToDevice); // v0.3.7.1
StePS/src/forces_cuda.cu:                if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:                        fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy SOFT_LENGTH in failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(dev_F, F_tmp, 3 * N_GPU * sizeof(REAL), cudaMemcpyHostToDevice);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy F in failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		printf("MPI task %i: GPU%i: ID_min = %i\tID_max = %i\n", rank, GPU_ID, GPU_index_min, GPU_index_min+N_GPU-1);
StePS/src/forces_cuda.cu:		// Launch a kernel on the GPU
StePS/src/forces_cuda.cu:		ForceKernel<<<32*mprocessors, BLOCKSIZE>>>(32 * mprocessors * BLOCKSIZE, N, dev_xx, dev_xy, dev_xz, dev_F, dev_M, dev_SOFT_LENGTH, mass_in_unit_sphere, DE, COSMOLOGY, COMOVING_INTEGRATION, GPU_index_min, GPU_index_min+N_GPU-1);
StePS/src/forces_cuda.cu:		cudaStatus = cudaGetLastError();
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: ForceKernel launch failed: %s\n", rank, GPU_ID, cudaGetErrorString(cudaStatus));
StePS/src/forces_cuda.cu:		// cudaDeviceSynchronize waits for the kernel to finish, and returns
StePS/src/forces_cuda.cu:		cudaStatus = cudaDeviceSynchronize();
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaDeviceSynchronize returned error code %d after launching ForceKernel!\n", rank, GPU_ID, cudaStatus);
StePS/src/forces_cuda.cu:		// Copy output vector from GPU buffer to host memory.
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(F_tmp, dev_F, 3 * N_GPU * sizeof(REAL), cudaMemcpyDeviceToHost);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI %i: GPU%i: cudaMemcpy F out failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		if(GPU_ID == 0)
StePS/src/forces_cuda.cu:			for (i = 0; i < N_GPU; i++)
StePS/src/forces_cuda.cu:			for (i = GPU_index_min; i < GPU_index_min + N_GPU; i++)
StePS/src/forces_cuda.cu:					F[3*(i-ID_min)+j] = F_tmp[3 * (i-GPU_index_min) + j];
StePS/src/forces_cuda.cu:		cudaFree(dev_xx);
StePS/src/forces_cuda.cu:                cudaFree(dev_xy);
StePS/src/forces_cuda.cu:                cudaFree(dev_xz);
StePS/src/forces_cuda.cu:                cudaFree(dev_M);
StePS/src/forces_cuda.cu:                cudaFree(dev_F);
StePS/src/forces_cuda.cu:		cudaFree(dev_SOFT_LENGTH);
StePS/src/forces_cuda.cu:		cudaDeviceReset();
StePS/src/forces_cuda.cu:	return cudaStatus;
StePS/src/forces_cuda.cu:cudaError_t forces_periodic_cuda(REAL*x, REAL*F, int n_GPU, int ID_min, int ID_max) //Force calculation with multiple images on GPU
StePS/src/forces_cuda.cu:	int GPU_ID, nthreads;
StePS/src/forces_cuda.cu:	int N_GPU, GPU_index_min; //number of particles in this GPU, the first particles index
StePS/src/forces_cuda.cu:	cudaError_t cudaStatus;
StePS/src/forces_cuda.cu:	cudaStatus = cudaSuccess;
StePS/src/forces_cuda.cu:	cudaGetDeviceCount(&numDevices);
StePS/src/forces_cuda.cu:	if(numDevices<n_GPU)
StePS/src/forces_cuda.cu:			fprintf(stderr, "Error: MPI rank %i: Cannot allocate %i GPUs, because only one is available\n", rank, n_GPU);
StePS/src/forces_cuda.cu:			fprintf(stderr, "Error: MPI rank %i: Cannot allocate %i GPUs, because only %i are available\n", rank, n_GPU, numDevices);
StePS/src/forces_cuda.cu:		n_GPU = numDevices;
StePS/src/forces_cuda.cu:		printf("Number of GPUs set to %i\n", n_GPU);
StePS/src/forces_cuda.cu:		fprintf(stderr, "MPI task %i: failed to allocate memory for xx_tmp (for CUDA force canculation).\n", rank);
StePS/src/forces_cuda.cu:		fprintf(stderr, "MPI task %i: failed to allocate memory for xx_tmp (for CUDA force canculation).\n", rank);
StePS/src/forces_cuda.cu:		fprintf(stderr, "MPI task %i: failed to allocate memory for xx_tmp (for CUDA force canculation).\n", rank);
StePS/src/forces_cuda.cu:	omp_set_num_threads(n_GPU);     // Use n_GPU threads
StePS/src/forces_cuda.cu:#pragma omp parallel default(shared) private(GPU_ID, F_tmp, i, j, mprocessors, cudaStatus, N_GPU, GPU_index_min, nthreads, dev_xx, dev_xy, dev_xz, dev_M, dev_F, dev_SOFT_LENGTH, dev_e)
StePS/src/forces_cuda.cu:		GPU_ID = omp_get_thread_num(); //thread ID = GPU_ID
StePS/src/forces_cuda.cu:		if(GPU_ID == 0)
StePS/src/forces_cuda.cu:			N_GPU = (ID_max-ID_min+1)/n_GPU+(ID_max-ID_min+1)%n_GPU;
StePS/src/forces_cuda.cu:			GPU_index_min = ID_min;
StePS/src/forces_cuda.cu:			N_GPU = (ID_max-ID_min+1)/n_GPU;
StePS/src/forces_cuda.cu:			GPU_index_min = ID_min + (ID_max-ID_min+1)%n_GPU+N_GPU*GPU_ID;
StePS/src/forces_cuda.cu:		F_tmp = (REAL*)malloc(3 * N_GPU * sizeof(REAL));
StePS/src/forces_cuda.cu:		for(i=0; i < N_GPU; i++)
StePS/src/forces_cuda.cu:		//Checking for the GPU
StePS/src/forces_cuda.cu:		cudaDeviceGetAttribute(&mprocessors, cudaDevAttrMultiProcessorCount, GPU_ID);
StePS/src/forces_cuda.cu:		if(GPU_ID == 0)
StePS/src/forces_cuda.cu:			printf("MPI task %i: GPU force calculation.\n Number of GPUs: %i\n Number of OMP threads: %i\n Number of threads per GPU: %i\n", rank, n_GPU, nthreads, 32*mprocessors*BLOCKSIZE);
StePS/src/forces_cuda.cu:		cudaStatus = cudaSetDevice(GPU_ID); //selecting GPU
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		// Allocate GPU buffers for coordinate and mass vectors
StePS/src/forces_cuda.cu:		cudaStatus = cudaMalloc((void**)&dev_xx, N * sizeof(REAL));
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: xx cudaMalloc failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMalloc((void**)&dev_xy, N * sizeof(REAL));
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: xy cudaMalloc failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMalloc((void**)&dev_xz, N * sizeof(REAL));
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: xz cudaMalloc failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMalloc((void**)&dev_M, N * sizeof(REAL));
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: M cudaMalloc failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		// Allocate GPU buffers for the softening vector
StePS/src/forces_cuda.cu:		cudaStatus = cudaMalloc((void**)&dev_SOFT_LENGTH, N * sizeof(REAL)); //v0.3.7.1
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: SOFT_LENGTH cudaMalloc failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		// Allocate GPU buffers for force vectors
StePS/src/forces_cuda.cu:		cudaStatus = cudaMalloc((void**)&dev_F, 3 * N_GPU * sizeof(REAL));
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: F cudaMalloc failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		// Allocate GPU buffers for e matrix
StePS/src/forces_cuda.cu:		cudaStatus = cudaMalloc((void**)&dev_e, 6606 * sizeof(int));
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: e cudaMalloc failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		// Copy input vectors from host memory to GPU buffers.
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(dev_xx, xx_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xx in failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(dev_xy, xy_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xy in failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(dev_xz, xz_tmp, N * sizeof(REAL), cudaMemcpyHostToDevice);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy xz in failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(dev_M, M, N * sizeof(REAL), cudaMemcpyHostToDevice);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy M in failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(dev_SOFT_LENGTH, SOFT_LENGTH, N * sizeof(REAL), cudaMemcpyHostToDevice);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy SOFT_LENGTH in failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(dev_F, F_tmp, 3 * N_GPU * sizeof(REAL), cudaMemcpyHostToDevice);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy F in failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(dev_e, e_tmp, 6606 * sizeof(int), cudaMemcpyHostToDevice);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy e in failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		printf("MPI task %i: GPU%i: ID_min = %i\tID_max = %i\n", rank, GPU_ID, GPU_index_min, GPU_index_min+N_GPU-1);
StePS/src/forces_cuda.cu:		// Launch a kernel on the GPU with one thread for each element.
StePS/src/forces_cuda.cu:		ForceKernel_periodic << <32*mprocessors, BLOCKSIZE>> >(32*mprocessors * BLOCKSIZE, N, dev_xx, dev_xy, dev_xz, dev_F, IS_PERIODIC, dev_M, dev_SOFT_LENGTH, L, dev_e, el, GPU_index_min, GPU_index_min+N_GPU-1);
StePS/src/forces_cuda.cu:		cudaStatus = cudaGetLastError();
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: ForceKernel_periodic launch failed: %s\n", rank, GPU_ID, cudaGetErrorString(cudaStatus));
StePS/src/forces_cuda.cu:		// cudaDeviceSynchronize waits for the kernel to finish, and returns
StePS/src/forces_cuda.cu:		cudaStatus = cudaDeviceSynchronize();
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaDeviceSynchronize returned error code %d after launching ForceKernel_periodic!\n", rank, GPU_ID, cudaStatus);
StePS/src/forces_cuda.cu:		// Copy output vector from GPU buffer to host memory.
StePS/src/forces_cuda.cu:		cudaStatus = cudaMemcpy(F_tmp, dev_F, 3 * N_GPU * sizeof(REAL), cudaMemcpyDeviceToHost);
StePS/src/forces_cuda.cu:		if (cudaStatus != cudaSuccess) {
StePS/src/forces_cuda.cu:			fprintf(stderr, "MPI rank %i: GPU%i: cudaMemcpy out failed!\n", rank, GPU_ID);
StePS/src/forces_cuda.cu:		if(GPU_ID == 0)
StePS/src/forces_cuda.cu:			for (i = 0; i < N_GPU; i++)
StePS/src/forces_cuda.cu:			for (i = GPU_index_min; i < GPU_index_min + N_GPU; i++)
StePS/src/forces_cuda.cu:					F[3*(i-ID_min)+j] = F_tmp[3 * (i-GPU_index_min) + j];
StePS/src/forces_cuda.cu:		cudaFree(dev_xx);
StePS/src/forces_cuda.cu:		cudaFree(dev_xy);
StePS/src/forces_cuda.cu:		cudaFree(dev_xz);
StePS/src/forces_cuda.cu:		cudaFree(dev_M);
StePS/src/forces_cuda.cu:		cudaFree(dev_F);
StePS/src/forces_cuda.cu:		cudaFree(dev_SOFT_LENGTH);
StePS/src/forces_cuda.cu:		cudaFree(dev_e);
StePS/src/forces_cuda.cu:		cudaDeviceReset();
StePS/src/forces_cuda.cu:	return cudaStatus;
StePS/src/global_variables.h:extern int n_GPU; //number of cuda capable GPUs
README.md:The StePS code is optimized to run on GPU accelerated HPC systems.

```
