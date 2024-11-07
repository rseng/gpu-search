# https://github.com/sheydenreich/threepoint

```console
cuda_version/calculateMap2Map3Covariance.cu:#include "cuda_helpers.cuh"
cuda_version/calculateMap2Map3Covariance.cu: * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
cuda_version/calculateMap2Map3Covariance.cu:    sprintf(filename, "covMap2Map3_%s_term2Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
cuda_version/calculateMap2Map3Covariance.cu:    sprintf(filename, "covMap2Map3_%s_term3Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
cuda_version/calculateMap2Map3Covariance.cu:    sprintf(filename, "covMap2Map3_%s_term4Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
cuda_version/calculateApertureStatisticsCovariance.cu:#include "cuda_helpers.cuh"
cuda_version/calculateApertureStatisticsCovariance.cu: * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
cuda_version/calculateApertureStatisticsCovariance.cu:    sprintf(filename, "cov_%s_term1Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
cuda_version/calculateApertureStatisticsCovariance.cu:    sprintf(filename, "cov_%s_term2Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
cuda_version/calculateApertureStatisticsCovariance.cu:    sprintf(filename, "cov_%s_term4Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
cuda_version/calculateApertureStatisticsCovariance.cu:    sprintf(filename, "cov_%s_term5Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
cuda_version/calculateApertureStatisticsCovariance.cu:    sprintf(filename, "cov_%s_term6Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
cuda_version/calculateApertureStatisticsCovariance.cu:    sprintf(filename, "cov_%s_term7Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
cuda_version/calculateApertureStatisticsCovariance.cu:    sprintf(filename, "cov_%s_term7_2h_Numerical_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
cuda_version/calculateMap6.cu:#include "cuda_helpers.cuh"
cuda_version/calculateMap6.cu: * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
cuda_version/halomodel.cuh: * Calculates sigma^2(m) and dsigma²/dm and stores them on GPU
cuda_version/halomodel.cuh: * Same as GSL implementation, because they are not implemented in CUDA
cuda_version/halomodel.cuh: * @brief Integrand of powerspectrum, interface to GPU
cuda_version/halomodel.cuh: * @brief Integrand of Trispectrum, interface to GPU
cuda_version/halomodel.cuh: * @brief Integrand of Pentaspectrum, interface to GPU
cuda_version/testHalobias.cu:#include "cuda_helpers.cuh"
cuda_version/testHalobias.cu: * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
cuda_version/calculateGamma.cu:#include "cuda_helpers.cuh"
cuda_version/calculateGamma.cu: * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
cuda_version/calculateGamma.cu:Argument 5 (optional): GPU device number
cuda_version/calculateGamma.cu:    std::cout << "on GPU " << deviceNumber << std::endl;
cuda_version/calculateGamma.cu:    cudaSetDevice(deviceNumber);
cuda_version/calculateGamma.cu:    std::cout << "on default GPU" << std::endl;
cuda_version/calculateMap2Covariance.cu:#include "cuda_helpers.cuh"
cuda_version/calculateMap2Covariance.cu: * Code uses CUDA and cubature library  (See https://github.com/stevengj/cubature for documentation)
cuda_version/calculateMap2Covariance.cu:    sprintf(filename, "covMap2_%s_Gauss_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
cuda_version/calculateMap2Covariance.cu:    sprintf(filename, "covMap2_%s_NonGauss_sigma_%.2f_n_%.2f_thetaMax_%.2f_gpu.dat",
cuda_version/interface.cu:     * 4. Copies some constants to the GPU
cuda_version/interface.cu:        // Copy constants to GPU
cuda_version/apertureStatistics.cu:#include "cuda_helpers.cuh"
cuda_version/apertureStatistics.cu:  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/apertureStatistics.cu:  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // allocate memory
cuda_version/apertureStatistics.cu:  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatistics.cu:  CudaCheckError();
cuda_version/apertureStatistics.cu:  cudaFree(dev_vars); // Free variables
cuda_version/apertureStatistics.cu:  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatistics.cu:  cudaFree(dev_value); // Free values
cuda_version/apertureStatistics.cu:  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/apertureStatistics.cu:  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
cuda_version/apertureStatistics.cu:  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatistics.cu:  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatistics.cu:  cudaFree(dev_vars); // Free variables
cuda_version/apertureStatistics.cu:  cudaFree(dev_value); // Free values
cuda_version/calculateSecondOrderAperturestatistics.cu:#include "cuda_helpers.cuh"
cuda_version/calculateSecondOrderAperturestatistics.cu: * Code uses CUDA and cubature library  (See
cuda_version/calculateSecondOrderAperturestatistics.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigma, &sigma, sizeof(double)));
cuda_version/calculateSecondOrderAperturestatistics.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n, &n, sizeof(double)));
cuda_version/gamma.cuh:#ifndef GAMMA_GPU_CUH
cuda_version/gamma.cuh:#define GAMMA_GPU_CUH
cuda_version/gamma.cuh:#define PERFORM_SUM_REDUCTION // Perform a sum reduction on shared memory of GPU. Alternative: Perform the sum in Host
cuda_version/gamma.cuh: * @param max_idx maximum value of simultaneously executed calculations (GPU-dependent)
cuda_version/gamma.cuh: * @param fdata Pointer to GammaCudaContainer instance
cuda_version/gamma.cuh: * @param fdata Pointer to GammaCudaContainer instance
cuda_version/gamma.cuh:struct GammaCudaContainer
cuda_version/gamma.cuh:#endif // GAMMA_GPU_CUH
cuda_version/Makefile:# NVCC = /usr/local/cuda-11.6/bin/nvcc -std=c++17 -rdc=true -Xcompiler -fopenmp -O3 -arch=sm_86
cuda_version/Makefile:# define the cuda source files
cuda_version/Makefile:# define the CUDA executables (ADD ALL HERE)
cuda_version/Makefile:# define the cuda object files
cuda_version/Makefile:all: threepoint_gpu.so $(EXECS_CU)
cuda_version/Makefile:threepoint_gpu.so: $(OBJS_CU) $(OBJS_CPP)
cuda_version/Makefile:	mv threepoint_gpu.so ../cosmosis/
cuda_version/halomodel.cu:#include "cuda_helpers.cuh"
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(devLogMmin, &logMmin, sizeof(double)));
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(devLogMmax, &logMmax, sizeof(double)));
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n_mbins, &n_mbins, sizeof(int)));
cuda_version/halomodel.cu:#ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:#ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigma2_array, &sigma2_array, n_mbins * sizeof(double)));
cuda_version/halomodel.cu:#ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_dSigma2dm_array, &dSigma2dm_array, n_mbins * sizeof(double)));
cuda_version/halomodel.cu:#ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:#ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/halomodel.cu:  cudaFree(dev_vars); // Free variables
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/halomodel.cu:  cudaFree(dev_value); // Free values
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/halomodel.cu:  cudaFree(dev_vars); // Free variables
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/halomodel.cu:  cudaFree(dev_value); // Free values
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/halomodel.cu:  cudaFree(dev_vars); // Free variables
cuda_version/halomodel.cu:  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/halomodel.cu:  cudaFree(dev_value); // Free values
cuda_version/halomodel.cu:  #ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:    #ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:  #ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:    #ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:  #ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:    #ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:  #ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:    #ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:#ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:#ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:    #ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:#ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:#ifdef __CUDA_ARCH__
cuda_version/halomodel.cu:    #ifdef __CUDA_ARCH__
cuda_version/bispectrum.cuh: * @brief Copies all the cosmology-independent constants to the GPU.
cuda_version/bispectrum.cuh: * This needs to be called before executing any computations on the GPU.
cuda_version/bispectrum.cuh: * @brief Wrapper to call the integration of the function dev_limber_integrand_power_spectrum on the GPU
cuda_version/apertureStatisticsCovariance.cuh:#include "cuda_helpers.cuh"
cuda_version/apertureStatisticsCovariance.cuh: * This file declares routines needed for the analytic calculation of Aperture statistic covariances with CUDA
cuda_version/apertureStatisticsCovariance.cu:#include "cuda_helpers.cuh"
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_thetaMax, &thetaMax, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_area, &area, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigma, &sigma, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n, &n, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMin, &lMin, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_constant_powerspectrum, &constant_powerspectrum, sizeof(bool)));
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_thetaMax_smaller, &thetaMax_smaller, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:#ifdef __CUDA_ARCH__
cuda_version/apertureStatisticsCovariance.cu:#ifdef __CUDA_ARCH__
cuda_version/apertureStatisticsCovariance.cu:#ifdef __CUDA_ARCH__
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_vars); // Free variables
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_value); // Free values
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_vars); // Free variables
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_value); // Free values
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_vars); // Free variables
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_value); // Free values
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_vars); // Free variables
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_value); // Free values
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value_iter, fdim * npts_iter * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMalloc(&dev_vars_iter, ndim * npts_iter * sizeof(double)));                                   // alocate memory
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMemcpy(dev_vars_iter, vars_iter, ndim * npts_iter * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:        cudaFree(dev_vars_iter); // Free variables
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMemcpy(value_iter, dev_value_iter, fdim * npts_iter * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:        cudaFree(dev_value_iter);
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value_iter, fdim * npts_iter * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMalloc(&dev_vars_iter, ndim * npts_iter * sizeof(double)));                                   // alocate memory
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMemcpy(dev_vars_iter, vars_iter, ndim * npts_iter * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:        cudaFree(dev_vars_iter); // Free variables
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMemcpy(value_iter, dev_value_iter, fdim * npts_iter * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:        cudaFree(dev_value_iter); // Free values
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value_iter, fdim * npts_iter * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMalloc(&dev_vars_iter, ndim * npts_iter * sizeof(double)));                                   // alocate memory
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMemcpy(dev_vars_iter, vars_iter, ndim * npts_iter * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:        cudaFree(dev_vars_iter); // Free variables
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMemcpy(value_iter, dev_value_iter, fdim * npts_iter * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:        cudaFree(dev_value_iter); // Free values
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value_iter, fdim * npts_iter * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMalloc(&dev_vars_iter, ndim * npts_iter * sizeof(double)));                                   // alocate memory
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMemcpy(dev_vars_iter, vars_iter, ndim * npts_iter * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:        cudaFree(dev_vars_iter); // Free variables
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMemcpy(value_iter, dev_value_iter, fdim * npts_iter * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:        cudaFree(dev_value_iter); // Free values
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value_iter, fdim * npts_iter * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMalloc(&dev_vars_iter, ndim * npts_iter * sizeof(double)));                                   // alocate memory
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMemcpy(dev_vars_iter, vars_iter, ndim * npts_iter * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:        cudaFree(dev_vars_iter); // Free variables
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMemcpy(value_iter, dev_value_iter, fdim * npts_iter * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:        cudaFree(dev_value_iter); // Free values
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_vars); // Free variables
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_value); // Free values
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_vars); // Free variables
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_value); // Free values
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double))); // Copies the value of lMax to the GPU
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory for result on GPU
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_vars); // Free variables
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_value); // Free values
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // alocate memory
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_vars); // Free variables
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:    cudaFree(dev_value); // Free values
cuda_version/apertureStatisticsCovariance.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_lMax, &lMax, sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value_iter, fdim * npts_iter * sizeof(double)));
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMalloc(&dev_vars_iter, ndim * npts_iter * sizeof(double)));                                   // alocate memory
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMemcpy(dev_vars_iter, vars_iter, ndim * npts_iter * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/apertureStatisticsCovariance.cu:        cudaFree(dev_vars_iter); // Free variables
cuda_version/apertureStatisticsCovariance.cu:        CUDA_SAFE_CALL(cudaMemcpy(value_iter, dev_value_iter, fdim * npts_iter * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/apertureStatisticsCovariance.cu:        cudaFree(dev_value_iter); // Free values
cuda_version/calculateHMF.cu:#include "cuda_helpers.cuh"
cuda_version/calculateNFW.cu:#include "cuda_helpers.cuh"
cuda_version/bispectrum.cu:#include "cuda_helpers.cuh"
cuda_version/bispectrum.cu:// Warning: They are read-only for the GPU!
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_A96, &A96, 48 * sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_W96, &W96, 48 * sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_H0_over_c, &H0_over_c, sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_c_over_H0, &c_over_H0, sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n_redshift_bins, &n_redshift_bins, sizeof(int)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n_kbins, &n_kbins, sizeof(int)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_h, &cosmo.h, sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_sigma8, &cosmo.sigma8, sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_omb, &cosmo.omb, sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_omc, &cosmo.omc, sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_ns, &cosmo.ns, sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_w, &cosmo.w, sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_om, &cosmo.om, sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_ow, &cosmo.ow, sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_Pk_given, &Pk_given, sizeof(bool)));
cuda_version/bispectrum.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_dk, &dk, sizeof(double)));
cuda_version/bispectrum.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_k_min, &kmin, sizeof(double)));
cuda_version/bispectrum.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_k_max, &kmax, sizeof(double)));
cuda_version/bispectrum.cu:    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_Pk, (P_k->data()), n_kbins * sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_norm, &norm_P, sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_dz, &dz, sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_z_max, &z_max, sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_f_K_array, f_K_array, n_redshift_bins * sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_g_array, g_array, n_redshift_bins * sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_D1_array, D1_array, n_redshift_bins * sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_r_sigma_array, r_sigma_array, n_redshift_bins * sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_n_eff_array, n_eff_array, n_redshift_bins * sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_ncur_array, ncur_array, n_redshift_bins * sizeof(double)));
cuda_version/bispectrum.cu:  #ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:  #ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifndef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifndef __CUDA_ARCH__
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMalloc((void **)&dev_value, fdim * npts * sizeof(double)));
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMalloc(&dev_vars, ndim * npts * sizeof(double)));                              // allocate memory
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpy(dev_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice)); // copying
cuda_version/bispectrum.cu:  cudaFree(dev_vars); // Free variables
cuda_version/bispectrum.cu:  CUDA_SAFE_CALL(cudaMemcpy(value, dev_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/bispectrum.cu:  cudaFree(dev_value); // Free values
cuda_version/bispectrum.cu:#ifndef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifdef __CUDA_ARCH__
cuda_version/bispectrum.cu:  #ifndef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifndef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifndef __CUDA_ARCH__
cuda_version/bispectrum.cu:    #ifndef __CUDA_ARCH__
cuda_version/bispectrum.cu:#ifndef __CUDA_ARCH__
cuda_version/calculateTrispectrum_halomodel.cu:#include "cuda_helpers.cuh"
cuda_version/calculatePowerspectrum.cu:#include "cuda_helpers.cuh"
cuda_version/apertureStatistics.cuh:#include "cuda_helpers.cuh"
cuda_version/apertureStatistics.cuh: * This file declares routines needed for the aperture statistics calculation with CUDA
cuda_version/apertureStatistics.cuh: * @brief Integrand of <Map²>, interface to GPU
cuda_version/apertureStatistics.cuh: * @brief Integrand of <Map³>, interface to GPU
cuda_version/apertureStatistics.cuh: * @brief Integrand of <Map⁴>, interface to GPU
cuda_version/apertureStatistics.cuh: * @brief Integrand of <Map⁶>, interface to GPU
cuda_version/cuda_helpers.cuh:#ifndef CUDA_HELPERS_CUH
cuda_version/cuda_helpers.cuh:#define CUDA_HELPERS_CUH
cuda_version/cuda_helpers.cuh:#include <cuda.h>
cuda_version/cuda_helpers.cuh:#include <cuda_runtime_api.h>
cuda_version/cuda_helpers.cuh:// Macro to catch CUDA errors in CUDA runtime calls
cuda_version/cuda_helpers.cuh:#define CUDA_SAFE_CALL(ans)                   \
cuda_version/cuda_helpers.cuh:        gpuAssert((ans), __FILE__, __LINE__); \
cuda_version/cuda_helpers.cuh:inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
cuda_version/cuda_helpers.cuh:    if (code != cudaSuccess)
cuda_version/cuda_helpers.cuh:        fprintf(stderr, "GPUassert: %s in %s, line %d\n", cudaGetErrorString(code), file, line);
cuda_version/cuda_helpers.cuh:// For GPU Parallelisation, match this to maximum of computing GPU
cuda_version/cuda_helpers.cuh:#define BLOCKS 92 // Maximum blocks for all SMs in GPU
cuda_version/cuda_helpers.cuh:#endif //CUDA_HELPERS_CUH
cuda_version/gamma.cu:#include "cuda_helpers.cuh"
cuda_version/gamma.cu:  // Copy the weights to the GPU
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_array_psi, array_psi, prec_k * sizeof(double)));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_array_product, array_product, prec_k * sizeof(double)));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_array_psi_J2, array_psi_J2, prec_k * sizeof(double)));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_array_product_J2, array_product_J2, prec_k * sizeof(double)));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMemcpyToSymbol(dev_prec_k, &prec_k, sizeof(int)));
cuda_version/gamma.cu:  struct GammaCudaContainer params;
cuda_version/gamma.cu:  struct GammaCudaContainer params;
cuda_version/gamma.cu:  struct GammaCudaContainer params = *((GammaCudaContainer *)fdata);
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMalloc(&d_value, fdim * npts * sizeof(double)));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMalloc(&d_vars, ndim * npts * sizeof(double)));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMemcpy(d_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMemset(d_value, 0, fdim * npts * sizeof(double)));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMemcpy(value, d_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaFree(d_vars));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaFree(d_value));
cuda_version/gamma.cu:  struct GammaCudaContainer params = *((GammaCudaContainer *)fdata);
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMalloc(&d_value, fdim * npts * sizeof(double)));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMalloc(&d_vars, ndim * npts * sizeof(double)));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMemcpy(d_vars, vars, ndim * npts * sizeof(double), cudaMemcpyHostToDevice));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMemset(d_value, 0, fdim * npts * sizeof(double)));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaMemcpy(value, d_value, fdim * npts * sizeof(double), cudaMemcpyDeviceToHost));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaFree(d_vars));
cuda_version/gamma.cu:  CUDA_SAFE_CALL(cudaFree(d_value));
cuda_version/calculateApertureStatistics.cu:#include "cuda_helpers.cuh"
cuda_version/calculateApertureStatistics.cu: * independent combis of thetas Code uses CUDA and cubature library  (See
cuda_version/calculateApertureStatistics.cu:./calculateApertureStatistics.x ../necessary_files/MR_cosmo.dat ../necessary_files/HOWLS_thetas.dat ../../results_MR/MapMapMap_bispec_gpu_nz.dat ../necessary_files/nz_MR.dat
cuda_version/calculatePowerspectrum_halomodel.cu:#include "cuda_helpers.cuh"
cuda_version/calculateMap4.cu:#include "cuda_helpers.cuh"
cuda_version/calculateMap4.cu: * Code uses CUDA and cubature library  (See
cuda_version/calculatePentaspectrum_halomodel.cu:#include "cuda_helpers.cuh"
README.md:At the moment, only the GPU accelerated version is fully tested and released (although a CPU-only version might be available in the future). Therefore  additionally the following is needed:
README.md:* **NVIDIA graphics card with CUDA capability of at least 2**. Check here to see, if your card works: [https://en.wikipedia.org/wiki/CUDA#GPUs_supported](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).
README.md:* **CUDA SDK Toolkit** (Tested for version 10.1, at least 7 needed!)
README.md:  Can be downloaded here [https://developer.nvidia.com/accelerated-computing-toolkit](https://developer.nvidia.com/accelerated-computing-toolkit)
README.md:In general, some knowledge on CUDA and how GPUs work is useful to understand the code!
README.md:cd cuda_version
README.md:7. Now, check if the folder `cuda_version` contains the necessary executables.
scripts/calculateApertureStatisticsCovariance_Laila.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_Shapenoise.dat 1 0 0 infinite
scripts/calculateApertureStatisticsCovariance_Laila.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_Shapenoise.dat 1 0 0 infinite
scripts/calculateApertureStatisticsCovariance_Laila.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_Shapenoise.dat 1 0 0 infinite
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_Shapenoise.dat 0 1 0 square
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_Shapenoise.dat 0 1 0 square
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_Shapenoise.dat 0 1 0 square
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_Shapenoise.dat 1 0 0 square
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_Shapenoise.dat 1 0 0 square
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_Shapenoise.dat 1 0 0 square
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 1 0 0 infinite
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 1 0 0 infinite
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 1 0 0 infinite
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 0 1 0 square
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 0 1 0 square
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 0 1 0 square
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 1 0 0 square
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 1 0 0 0 0 square
scripts/calculateApertureStatisticsCovariance_Laila.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 1 0 0 square
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 1 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 1 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_addedShapenoise.dat 1 0 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_addedShapenoise.dat 0 1 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 1 0 0 square &> /vol/lensgpu/ssd/llinke/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_sigmaTwice.dat 0 0 1 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 0 1 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat  $DIR ../necessary_files/Covariance_SLICS.dat 1 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat  $DIR ../necessary_files/Covariance_SLICS.dat 1 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat  $DIR ../necessary_files/Covariance_SLICS.dat 0 1 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat  $DIR ../necessary_files/Covariance_SLICS.dat 0 1 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 1 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 1 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 1 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS_shapenoise.dat 1 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateMap2Covariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS_shapenoise.dat 0 1 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 1 0 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 1 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 1 0 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 0 1 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 0 0 0 1 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsSSC.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/testBispecSSC.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 0 1 0 0 0 0 rectangle &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 1 0 0 0 0 0 rectangle &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 0 0 1 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 0 0 0 0 1 0 rectangle &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 0 1 0 0 0 0 rectangle &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_Laila.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS_KiDS_South.dat 1 0 0 0 0 0 rectangle &> $DIR/${timestamp}.log
scripts/calculationsCovarianceSLICS_Analytical.sh:export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
scripts/calculationsCovarianceSLICS_Analytical.sh:echo Using GPU $CUDA_VISIBLE_DEVICES
scripts/calculationsCovarianceSLICS_Analytical.sh:../cuda_version/calculateMap3SSC.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_4_to_16.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat square 
scripts/calculationsCovarianceSLICS_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_4_to_16.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceSLICS_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceSLICS_Analytical.sh:# ./cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 1 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceSLICS_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceSLICS_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceSLICS_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 0 1 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatistics_MS.sh:DIR_BIN=../cuda_version/
scripts/calculateApertureStatistics_MS.sh:export CUDA_VISIBLE_DEVICES=1 #$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
scripts/calculateApertureStatistics_MS.sh:echo Using GPU $CUDA_VISIBLE_DEVICES
scripts/calculationsCovarianceGRF_Analytical.sh:export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
scripts/calculationsCovarianceGRF_Analytical.sh:echo Using GPU $CUDA_VISIBLE_DEVICES
scripts/calculationsCovarianceGRF_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_10deg.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceGRF_Analytical.sh:../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_15deg.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceGRF_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_20deg.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceGRF_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_10deg.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceGRF_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_15deg.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceGRF_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_20deg.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceGRF_Analytical.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_10deg.dat 1 0 0 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceGRF_Analytical.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_15deg.dat 1 0 0 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceGRF_Analytical.sh:#../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_GRF_20deg.dat 1 0 0 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceMap2SLICS_Analytical.sh:export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
scripts/calculationsCovarianceMap2SLICS_Analytical.sh:echo Using GPU $CUDA_VISIBLE_DEVICES
scripts/calculationsCovarianceMap2SLICS_Analytical.sh:../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_4_to_16.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 1 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceMap2SLICS_Analytical.sh:../cuda_version/calculateMap2Covariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_4_to_16.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 1 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceSLICS_Analytical_singleZ.sh:export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
scripts/calculationsCovarianceSLICS_Analytical_singleZ.sh:echo Using GPU $CUDA_VISIBLE_DEVICES
scripts/calculationsCovarianceSLICS_Analytical_singleZ.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_SLICS_singleZ.dat $DIR ../necessary_files/Covariance_SLICS_noShapenoise.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceSLICS_Analytical_singleZ.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_SLICS_singleZ.dat $DIR ../necessary_files/Covariance_SLICS_noShapenoise.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceSLICS_Analytical_singleZ.sh:../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_SLICS_singleZ.dat $DIR ../necessary_files/Covariance_SLICS_noShapenoise.dat 0 0 1 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceSLICS_Analytical_singleZ.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_SLICS_singleZ.dat $DIR ../necessary_files/Covariance_SLICS_noShapenoise.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceSLICS_Analytical_singleZ.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_SLICS_singleZ.dat $DIR ../necessary_files/Covariance_SLICS_noShapenoise.dat 0 0 0 0 1 0 square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceSLICS_Analytical_singleZ.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_SLICS_singleZ.dat $DIR ../necessary_files/Covariance_SLICS_noShapenoise.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log
scripts/calculateDerivativeApertureStatistics_DUSTGRAIN.sh:DIR_BIN=../cuda_version/
scripts/calculationsCovarianceMap2MS_Analytical.sh:export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
scripts/calculationsCovarianceMap2MS_Analytical.sh:echo Using GPU $CUDA_VISIBLE_DEVICES
scripts/calculationsCovarianceMap2MS_Analytical.sh:../cuda_version/calculateMap2Covariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas_4_to_16.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 1 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceMap2MS_Analytical.sh:../cuda_version/calculateMap2Covariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas_4_to_16.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 1 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_Shapenoise.dat 1 0 0 infinite
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_Shapenoise.dat 1 0 0 infinite
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_Shapenoise.dat 1 0 0 infinite
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_Shapenoise.dat 0 1 0 square
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_Shapenoise.dat 0 1 0 square
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_Shapenoise.dat 0 1 0 square
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_Shapenoise.dat 1 0 0 square
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_Shapenoise.dat 1 0 0 square
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_Shapenoise.dat 1 0 0 square
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 1 0 0 infinite
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 1 0 0 infinite
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 1 0 0 infinite
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 0 1 0 square
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 0 1 0 square
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 0 1 0 square
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_5Deg_cosmicShear.dat 1 0 0 square
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_10Deg_cosmicShear.dat 1 0 0 square
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_15Deg_cosmicShear.dat 1 0 0 square
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 1 0 0 0 0 0 infinite &> /vol/lensgpu/ssd/llinke/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 1 0 0 0 0 square &> /vol/lensgpu/ssd/llinke/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=1 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 1 0 0 0 infinite &> /vol/lensgpu/ssd/llinke/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=1 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 1 0 0 infinite &> /vol/lensgpu/ssd/llinke/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=1 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 0 1 0 square &> /vol/lensgpu/ssd/llinke/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=1 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat 0 0 0 0 0 1 infinite &> /vol/lensgpu/ssd/llinke/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 1 0 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:# CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 1 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 1 0 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 0 1 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 0 0 0 1 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/MR_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nz_MR.dat $DIR ../necessary_files/Covariance_MS.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsSSC.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:#CUDA_VISIBLE_DEVICES=0 ../cuda_version/testBispecSSC.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_SLICS.dat &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 1 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 0 1 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatisticsCovariance_AIFA.sh:CUDA_VISIBLE_DEVICES=0 ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log
scripts/calculateApertureStatistics_SLICS.sh:DIR_BIN=../cuda_version/
scripts/calculationsCovarianceTakahashi_Analytical_EuclidNz.sh:export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
scripts/calculationsCovarianceTakahashi_Analytical_EuclidNz.sh:echo Using GPU $CUDA_VISIBLE_DEVICES
scripts/calculationsCovarianceTakahashi_Analytical_EuclidNz.sh:../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/SLICS_cosmo.dat ../necessary_files/Our_thetas_4_to_16.dat ../necessary_files/n_z_SLICS_euclid_nz_cosmos15_i24.5cut_fu08fit_dz0.01.cat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical_EuclidNz.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical_EuclidNz.sh:# ./cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 1 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical_EuclidNz.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical_EuclidNz.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical_EuclidNz.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 0 1 0 square &> $DIR/${timestamp}.log
scripts/calculateApertureStatistics_DUSTGRAIN.sh:DIR_BIN=../cuda_version/
scripts/calculationsCovarianceTakahashi_Analytical.sh:export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
scripts/calculationsCovarianceTakahashi_Analytical.sh:echo Using GPU $CUDA_VISIBLE_DEVICES
scripts/calculationsCovarianceTakahashi_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 1 0 0 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 1 0 0 0 0 square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical.sh:# ./cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 1 0 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 1 0 0 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 0 0 1 infinite &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical.sh:# ../cuda_version/calculateApertureStatisticsCovariance.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike.dat 0 0 0 0 1 0 square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical.sh:#../cuda_version/calculateMap3SSC.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike_changedArea1.dat square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical.sh:#../cuda_version/calculateMap3SSC.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike_changedArea2.dat square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical.sh:#../cuda_version/calculateMap3SSC.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike_changedArea3.dat square &> $DIR/${timestamp}.log
scripts/calculationsCovarianceTakahashi_Analytical.sh:../cuda_version/calculateMap3SSC.x ../necessary_files/Takahashi_cosmo.dat ../necessary_files/Our_thetas_4_to_32.dat ../necessary_files/nofz_kids1000_takahashi.dat $DIR ../necessary_files/Covariance_Takahashi_KiDSlike_changedArea4.dat square &> $DIR/${timestamp}.log
scripts/calculateApertureStatistics_SkySim5000.sh:DIR_BIN=../cuda_version/
examples/exampleMap2Map3Covariance.sh:DIR_BIN=../cuda_version/
examples/exampleScripts.sh:DIR_BIN=../cuda_version/
cosmosis/pipeline_gpu.ini:file = threepoint_gpu.so

```
