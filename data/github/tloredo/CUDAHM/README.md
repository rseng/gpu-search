# https://github.com/tloredo/CUDAHM

```console
cudahm/.project:	<name>cudahm</name>
cudahm/src/parameters.cuh:// macro for checking for CUDA errors
cudahm/src/parameters.cuh:#define CUDA_CHECK_RETURN(value) {											\
cudahm/src/parameters.cuh:	cudaError_t _m_cudaStat = value;										\
cudahm/src/parameters.cuh:	if (_m_cudaStat != cudaSuccess) {										\
cudahm/src/parameters.cuh:				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
cudahm/src/parameters.cuh:// Cuda Includes
cudahm/src/parameters.cuh:#include "cuda_runtime.h"
cudahm/src/parameters.cuh:#include "cuda_runtime_api.h"
cudahm/src/parameters.cuh:#include <cuda.h>
cudahm/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&p_logdens_function, c_LogDensMeas, sizeof(c_LogDensMeas)));
cudahm/src/parameters.cuh:	virtual ~DataAugmentation() { CUDA_CHECK_RETURN(cudaFree(p_devStates)); }
cudahm/src/parameters.cuh:		// Allocate memory on GPU for RNG states
cudahm/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaMalloc((void ** )&p_devStates, nThreads.x * nBlocks.x * sizeof(curandState)));
cudahm/src/parameters.cuh:		// Initialize the random number generator states on the GPU
cudahm/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaPeekAtLastError());
cudahm/src/parameters.cuh:		// Wait until RNG stuff is done running on the GPU, make sure everything went OK
cudahm/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
cudahm/src/parameters.cuh:		// set initial values for the characteristics. this will launch a CUDA kernel.
cudahm/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
cudahm/src/parameters.cuh:	// launch the update kernel on the GPU
cudahm/src/parameters.cuh:		// launch the kernel to update the characteristics on the GPU
cudahm/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaPeekAtLastError());
cudahm/src/parameters.cuh:	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
cudahm/src/parameters.cuh:			CUDA_CHECK_RETURN(cudaPeekAtLastError());
cudahm/src/parameters.cuh:			CUDA_CHECK_RETURN(cudaPeekAtLastError());
cudahm/src/parameters.cuh:			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
cudahm/src/parameters.cuh:	void SetCudaGrid(dim3& nB, dim3& nT) {
cudahm/src/parameters.cuh:		hvector h_chi = d_chi;  // first grab the values from the GPU
cudahm/src/parameters.cuh:	// CUDA kernel launch specifications
cudahm/src/parameters.cuh:	    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&p_logdens_function, c_LogDensPop, sizeof(c_LogDensPop)));
cudahm/src/parameters.cuh:		// transfer initial value of theta to GPU constant memory
cudahm/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta*sizeof(*p_theta)));
cudahm/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaPeekAtLastError());
cudahm/src/parameters.cuh:	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
cudahm/src/parameters.cuh:		// copy proposed theta to GPU constant memory
cudahm/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_proposed_theta, dtheta*sizeof(*p_proposed_theta)));
cudahm/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaPeekAtLastError());
cudahm/src/parameters.cuh:	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
cudahm/src/parameters.cuh:			CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta*sizeof(*p_theta)));
cudahm/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta * sizeof(*p_theta)));
cudahm/src/parameters.cuh:			CUDA_CHECK_RETURN(cudaPeekAtLastError());
cudahm/src/parameters.cuh:	void SetCudaGrid(dim3& nB, dim3& nT) {
cudahm/src/parameters.cuh:		// copy the current value of theta from constant GPU memory and return as a host-side vector
cudahm/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(p_theta, c_theta, sizeof(c_theta), 0));
cudahm/src/parameters.cuh:	// CUDA kernel launch specifications
cudahm/src/GibbsSampler.hpp:			// set the CUDA grid launch parameters and initialize the random number generator on the GPU
cudahm/src/GibbsSampler.hpp:			Daug_->SetCudaGrid(nB_, nT_);
cudahm/src/GibbsSampler.hpp:			PopPar_->SetCudaGrid(nB_, nT_);
cudahm/src/GibbsSampler.hpp:			// set the CUDA grid launch parameters and initialize the random number generator on the GPU
cudahm/src/GibbsSampler.hpp:			Daug_->SetCudaGrid(nB_, nT_);
cudahm/src/GibbsSampler.hpp:			PopPar_->SetCudaGrid(nB_, nT_);
cudahm/src/GibbsSampler.hpp:	// read the values from the GPU
cudahm/src/GibbsSampler.hpp:	dim3 nB_, nT_;  // CUDA grid launch parameters
cudahm/src/GibbsSampler.hpp:		// first do CUDA grid launch
cudahm/src/kernels.cu:#ifdef __CUDA_ARCH__
cudahm/src/kernels.cu:#ifdef __CUDA_ARCH__
cudahm/src/cudahm_blueprint.cu: * cudahm_blueprint.cu
cudahm/src/cudahm_blueprint.cu: * This file provides a blueprint for using the CUDAHM API. In order to use CUDAHM the user must supply a function
cudahm/src/cudahm_blueprint.cu: * in GPU constant memory. The purpose of this file is to provide an easy way of setting the pointers and stream-line
cudahm/src/cudahm_blueprint.cu: * the use of CUDAHM to build MCMC sampler. Using this blueprint, the user need only modify the LogDensityMeas and
cudahm/src/cudahm_blueprint.cu:// local CUDAHM include
cudahm/src/cudahm_blueprint.cu: * at compile-time in order to efficiently make use of GPU memory. These also need to be placed before the
cudahm/src/cudahm_blueprint.cu: * and written in CUDA. The input parameters are:
cudahm/src/cudahm_blueprint.cu: * the user and written in CUDA. The input parameters are:
cudahm/src/cudahm_blueprint.cu: * Pointers to the GPU functions used to compute the conditional log-densities for a single data point.
cudahm/src/cudahm_blueprint.cu: * These functions live on the GPU in constant memory.
cudahm/src/cudahm_blueprint.cu: * Pointer to the population parameter (theta), stored in constant memory on the GPU. Originally defined in
cudahm/src/cudahm_blueprint.cu:	std::cout << "This file provides a blueprint for using the CUDAHM API. On its own it does nothing except print this message."
cudahm/src/kernels.cuh:// Cuda Includes
cudahm/src/kernels.cuh:#include "cuda_runtime.h"
cudahm/src/kernels.cuh:#include "cuda_runtime_api.h"
cudahm/src/kernels.cuh:#include <cuda.h>
cudahm/src/kernels.cuh:// kernel to update the values of the characteristics in parallel on the GPU
tests/.project:			<location>/Users/brandonkelly/Projects/CUDAHM/cudahm/src/kernels.cu</location>
tests/src/parameters.cuh:// macro for checking for CUDA errors
tests/src/parameters.cuh:#define CUDA_CHECK_RETURN(value) {											\
tests/src/parameters.cuh:	cudaError_t _m_cudaStat = value;										\
tests/src/parameters.cuh:	if (_m_cudaStat != cudaSuccess) {										\
tests/src/parameters.cuh:				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
tests/src/parameters.cuh:// Cuda Includes
tests/src/parameters.cuh:#include "cuda_runtime.h"
tests/src/parameters.cuh:#include "cuda_runtime_api.h"
tests/src/parameters.cuh:#include <cuda.h>
tests/src/parameters.cuh:			// Allocate memory on GPU for RNG states
tests/src/parameters.cuh:			CUDA_CHECK_RETURN(cudaMalloc((void **)&p_devStates, nThreads.x * nBlocks.x * sizeof(curandState)));
tests/src/parameters.cuh:			// Initialize the random number generator states on the GPU
tests/src/parameters.cuh:			CUDA_CHECK_RETURN(cudaPeekAtLastError());
tests/src/parameters.cuh:			// Wait until RNG stuff is done running on the GPU, make sure everything went OK
tests/src/parameters.cuh:			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
tests/src/parameters.cuh:		    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&p_logdens_function, c_LogDensMeas, sizeof(c_LogDensMeas)));
tests/src/parameters.cuh:	virtual ~DataAugmentation() { cudaFree(p_devStates); }
tests/src/parameters.cuh:		// set initial values for the characteristics. this will launch a CUDA kernel.
tests/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
tests/src/parameters.cuh:	// launch the update kernel on the GPU
tests/src/parameters.cuh:		// launch the kernel to update the characteristics on the GPU
tests/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaPeekAtLastError());
tests/src/parameters.cuh:	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
tests/src/parameters.cuh:			CUDA_CHECK_RETURN(cudaPeekAtLastError());
tests/src/parameters.cuh:			CUDA_CHECK_RETURN(cudaPeekAtLastError());
tests/src/parameters.cuh:			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
tests/src/parameters.cuh:		hvector h_chi = d_chi;  // first grab the values from the GPU
tests/src/parameters.cuh:	// CUDA kernel launch specifications
tests/src/parameters.cuh:	    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&p_logdens_function, c_LogDensPop, sizeof(c_LogDensPop)));
tests/src/parameters.cuh:		// transfer initial value of theta to GPU constant memory
tests/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta*sizeof(*p_theta)));
tests/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaPeekAtLastError());
tests/src/parameters.cuh:	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
tests/src/parameters.cuh:		// copy proposed theta to GPU constant memory
tests/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_proposed_theta, dtheta*sizeof(*p_proposed_theta)));
tests/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaPeekAtLastError());
tests/src/parameters.cuh:	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
tests/src/parameters.cuh:			CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta*sizeof(*p_theta)));
tests/src/parameters.cuh:		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta * sizeof(*p_theta)));
tests/src/parameters.cuh:			CUDA_CHECK_RETURN(cudaPeekAtLastError());
tests/src/parameters.cuh:		// copy the current value of theta from constant GPU memory and return as a host-side vector
tests/src/parameters.cuh:		cudaMemcpyFromSymbol(p_theta, c_theta, sizeof(c_theta), 0, cudaMemcpyDeviceToHost);
tests/src/parameters.cuh:	// CUDA kernel launch specifications
tests/src/kernels.cu:#ifdef __CUDA_ARCH__
tests/src/kernels.cu:#ifdef __CUDA_ARCH__
tests/src/run_unit_tests.cu:		cudaMemGetInfo(&free, &total);
tests/src/run_unit_tests.cu:	// Cuda grid launch
tests/src/run_unit_tests.cu:		CUDA_CHECK_RETURN(cudaDeviceReset());
tests/src/run_unit_tests.cu:		cudaMemGetInfo(&free, &total);
tests/src/UnitTests.cu:    Theta.SetCudaGrid(nBlocks, nThreads);
tests/src/UnitTests.cu:    Theta.SetCudaGrid(nBlocks, nThreads);
tests/src/UnitTests.cu:	Theta.SetCudaGrid(nBlocks, nThreads);
tests/src/UnitTests.cu:	DaugPtr->SetCudaGrid(nBlocks, nThreads);
tests/src/UnitTests.cu:    Daug.SetCudaGrid(nBlocks, nThreads);
tests/src/UnitTests.cu:	Daug->SetCudaGrid(nBlocks, nThreads);
tests/src/UnitTests.cu:	Theta->SetCudaGrid(nBlocks, nThreads);
tests/src/UnitTests.cu:	// Cuda grid launch
tests/src/UnitTests.cu:	// initialize the RNG seed on the GPU first
tests/src/UnitTests.cu:	// Allocate memory on GPU for RNG states
tests/src/UnitTests.cu:	CUDA_CHECK_RETURN(cudaMalloc((void **)&p_devStates, nT.x * nB.x * sizeof(curandState)));
tests/src/UnitTests.cu:	CUDA_CHECK_RETURN(cudaPeekAtLastError());
tests/src/UnitTests.cu:	// Wait until RNG stuff is done running on the GPU, make sure everything went OK
tests/src/UnitTests.cu:	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
tests/src/UnitTests.cu:	CUDA_CHECK_RETURN(cudaPeekAtLastError());
tests/src/UnitTests.cu:	cudaFree(p_devStates);
tests/src/UnitTests.cu:// check that Accept accepts better proposals on the GPU, and updates the chi values
tests/src/UnitTests.cu:	// initialize the RNG seed on the GPU first
tests/src/UnitTests.cu:	// Allocate memory on GPU for RNG states
tests/src/UnitTests.cu:	CUDA_CHECK_RETURN(cudaMalloc((void **)&p_devStates, nThreads.x * nBlocks.x * sizeof(curandState)));
tests/src/UnitTests.cu:	CUDA_CHECK_RETURN(cudaPeekAtLastError());
tests/src/UnitTests.cu:	// Wait until RNG stuff is done running on the GPU, make sure everything went OK
tests/src/UnitTests.cu:	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
tests/src/UnitTests.cu:	cudaFree(p_devStates);
tests/src/UnitTests.cu:// check that Adapt updates the cholesky factor of the chi proposals on the GPU
tests/src/UnitTests.cu:		std::cerr << "Test for RAM Cholesky adaption step on the GPU failed: device and host results do not agree." << std::endl;
tests/src/UnitTests.cu:	Daug->SetCudaGrid(nBlocks, nThreads);
tests/src/UnitTests.cu:	Theta->SetCudaGrid(nBlocks, nThreads);
tests/src/UnitTests.cu:	Daug->SetCudaGrid(nBlocks, nThreads);
tests/src/UnitTests.cu:	Theta->SetCudaGrid(nBlocks, nThreads);
tests/src/UnitTests.cuh:// CUDAHM includes
tests/src/UnitTests.cuh:#include "../../cudahm/src/GibbsSampler.hpp"
tests/src/UnitTests.cuh:	// check that proposals for chi generated on the GPU have correct distribution
tests/src/UnitTests.cuh:	// CUDA grid launch parameters
README.md:CUDAHM
README.md:Routines for using CUDA to accelerate Bayesian inference of Hierarchical Models using Markov Chain Monte Carlo with GPUs.
README.md:`CUDAHM` enables one to easily and rapidly construct an MCMC sampler for a three-level hierarchical model, requiring the user to supply only a minimimal amount of CUDA code. `CUDAHM` assumes that a set of measurements are available for a sample of objects, and that these measurements are related to an unobserved set of characteristics for each object. For example, the measurements could be the spectral energy distributions of a sample of galaxies, and the unknown characteristics could be the physical quantities of the galaxies, such as mass, distance, age, etc. The measured spectral energy distributions depend on the unknown physical quantities, which enables one to derive their values from the measurements. The characteristics are also assumed to be independently and identically sampled from a parent population with unknown parameters (e.g., a Normal distribution with unknown mean and variance). `CUDAHM` enables one to simultaneously sample the values of the characteristics and the parameters of their parent population from their joint posterior probability distribution.
README.md:`CUDAHM` uses a Metropolis-within-Gibbs sampler. Each iteration of the MCMC sampler performs the following steps:
README.md:Both steps are done using the *Robust adaptive Metropolis algorithm with coerced acceptance rate* (Vihola 2012, Statistics and Computing, 22, pp 997-1008) using a multivariate normal proposal distribution and a target acceptance rate of 0.4. Step 1 is done in parallel for each object on the GPU, while the proposals for Step 2 are generated on the CPU, but the calculations are performed on the GPU.
README.md:There are three main classes in CUDAHM:
README.md:There are two functions that the user must provide: a function that compute the logarithm of the probability density of the measurements given the characteristics for each object in the sample, and a function that computes the logarithm of the probability density of the characteristics given the parent population parameters. These functions live on the GPU and must be written in CUDA. The file `cudahm_blueprint.cu` under the `cudahm` directory contains a blueprint with documentation that the user may use when constructing their own MCMC sampler. In addition, the following examples provide additional guidance:
README.md:In progress. For now probably the easiest thing is to just use the NVIDIA Nsight IDE that comes with CUDA to build the code. Once you build the code we recommend that you run the unit tests under the `tests` directory. Note that you will need to build the code in seperate compilation mode.
README.md:`CUDAHM` depends on the following libraries:
README.md:* `CUDA` (at least v5.0)
README.md:In order to use `CUDAHM` you will need a NVIDIA GPU of CUDA compute capability 2.0 or higher.
benchmarks/dusthm-cpp/dusthm-cpp.xcodeproj/project.xcworkspace/xcshareddata/dusthm-cpp.xccheckout:		<string>https://github.com/bckelly80/CUDAHM.git</string>
benchmarks/dusthm-cpp/dusthm-cpp.xcodeproj/project.xcworkspace/xcshareddata/dusthm-cpp.xccheckout:	<string>https://github.com/bckelly80/CUDAHM.git</string>
benchmarks/dusthm-cpp/dusthm-cpp.xcodeproj/project.xcworkspace/xcshareddata/dusthm-cpp.xccheckout:			<string>CUDAHM</string>
benchmarks/dusthm-cpp/dusthm-cpp.xcodeproj/project.pbxproj:				DFE9EB9419116595007B215F /* cudahm */,
benchmarks/dusthm-cpp/dusthm-cpp.xcodeproj/project.pbxproj:		DFE9EB9419116595007B215F /* cudahm */ = {
benchmarks/dusthm-cpp/dusthm-cpp.xcodeproj/project.pbxproj:			name = cudahm;
benchmarks/dusthm-cpp/src/dusthm-cpp.cpp:// local CUDAHM includes
benchmarks/dusthm-cpp/src/dusthm-cpp.cpp:	std::string datafile = "/Users/brandonkelly/Projects/CUDAHM/dusthm/data/cbt_sed_100000.dat";
benchmarks/dusthm-cpp/src/DataAugmentation.hpp:	// launch the update kernel on the GPU
benchmarks/dusthm-cpp/src/DataAugmentation.hpp:            // copy values for this data point for consistency with CUDAHM
benchmarks/dusthm-cpp/src/GibbsSampler.hpp:	// read the values from the GPU
normnorm/.project:			<location>/Users/brandonkelly/Projects/CUDAHM/cudahm/src/kernels.cu</location>
normnorm/.project:			<location>/Users/brandonkelly/Projects/CUDAHM/normnorm/src/parameters.cuh</location>
normnorm/src/normnorm.cu: * This file illustrates how to setup a simple Normal-Normal and sample from this using CUDAHM. The model is:
normnorm/src/normnorm.cu: * mean vector. We do this by using CUDAHM to construct a Metropolis-within-Gibbs MCMC sampler to sample from the posterior
normnorm/src/normnorm.cu:#include "../../cudahm/src/GibbsSampler.hpp"
normnorm/src/normnorm.cu: * Pointer to the population parameter (theta), stored in constant memory on the GPU. Originally defined in
normnorm/src/normnorm.cu:    int nchi_samples = 1000;  // only keep 1000 samples for the chi values to control memory usage and avoid numerous reads from GPU
.gitignore:# Built CUDA files
dusthm/.project:			<location>/Users/brandonkelly/Projects/CUDAHM/dusthm/src/dusthm.cu</location>
dusthm/.project:			<location>/Users/brandonkelly/Projects/CUDAHM/cudahm/src/kernels.cu</location>
dusthm/src/python/make_dusthm_data.py:data_dir = os.environ['HOME'] + '/Projects/CUDAHM/dusthm/data/'
dusthm/src/python/make_dusthm_data.py:data_dir = os.environ['HOME'] + '/Projects/CUDAHM/dusthm/data/'
dusthm/src/ConstBetaTemp.cuh:#include "../../cudahm/src/parameters.cuh"
dusthm/src/ConstBetaTemp.cuh:		// set initial values for the characteristics. this will launch a CUDA kernel.
dusthm/src/ConstBetaTemp.cuh:		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
dusthm/src/DustPopPar.hpp:#include "../../cudahm/src/parameters.cuh"
dusthm/src/dusthm.cu: * in the far-IR and sample from this using CUDAHM. The model is:
dusthm/src/dusthm.cu: * Gaussian measurement noise with known variances. We do this by using CUDAHM to construct a Metropolis-within-Gibbs MCMC sampler to sample
dusthm/src/dusthm.cu:// local CUDAHM includes
dusthm/src/dusthm.cu:#include "../../cudahm/src/GibbsSampler.hpp"
dusthm/src/dusthm.cu: * at compile-time in order to efficiently make use of GPU memory. These also need to be placed before the
dusthm/src/dusthm.cu: * Pointers to the GPU functions used to compute the conditional log-densities for a single data point.
dusthm/src/dusthm.cu: * These functions live on the GPU in constant memory.
dusthm/src/dusthm.cu: * Pointer to the population parameter (theta), stored in constant memory on the GPU. Originally defined in

```
