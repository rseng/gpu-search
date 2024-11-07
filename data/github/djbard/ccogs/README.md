# https://github.com/djbard/ccogs

```console
CITING_THIS_CODE:    title          = "{Cosmological Calculations on the GPU}",
ACKNOWLEDGEMENTS:## CUDA guides
ACKNOWLEDGEMENTS:In developing this code, the authors have made extensive reference of the CUDA programming guide and example code made available by NVIDIA Corporation. 
ACKNOWLEDGEMENTS:In particular, we use parts of the NVIDIA device_query code to obtain information about the GPU card. 
aperture_mass/INSTALL:# Requrements for GPU code.
aperture_mass/INSTALL:CUDA: Tested with version 4.1. It is possible that this code will work with 
aperture_mass/INSTALL:You will need to edit the Makefile to point to your own installation of CUDA and the SDK. 
aperture_mass/INSTALL:    CUDA_SDK = /path/to/my/cuda/sdk/installation
aperture_mass/INSTALL:    CUDA_INSTALL_PATH := /opt/cuda
aperture_mass/INSTALL:You may have to edit thse paths to point to where your CUDA libraries and SDK
aperture_mass/INSTALL:INCLUDES          += -I. -I$(CUDA_INSTALL_PATH)/include  -I$(CUDA_SDK)/C/common/inc 
aperture_mass/INSTALL:LIBS              := -L$(CUDA_INSTALL_PATH)/lib64  -L$(CUDA_SDK)/C/common/lib/ -L$(CUDA_SDK)/C/lib/ 
aperture_mass/INSTALL:We have found many CUDA installations to be somewhat different and rather than 
aperture_mass/INSTALL:guidance. If you are still having problems, please consult us or the NVIDIA 
aperture_mass/INSTALL:CUDA documentation.
aperture_mass/examples/run_mAp.py:## This should depend on the memory of your GPU card - if
aperture_mass/README:This package uses the CUDA programming language to calculate the two-point
aperture_mass/README:angular correlation function (ACF) on a GPU. Details about the function itself
aperture_mass/README:This code calculates the aperture mass map for a given dataset. For details on the theoretical background to the calculation, and performance of this implementation on the GPU, see the accompanying paper Bard + Bellis, 2012 (arXiv:1208.3658v1 [astro-ph.IM])
aperture_mass/README: In this code, we use the filter proposed by Schirmer et al (2007), which is an NFW profile with exponential cut-offs at zero and large radii. We intend to implement other filter functions in this code in future releases; in the meantime, it should be straightfoward to add a new filter in src/mAp_gpu_grid.cu. 
aperture_mass/README:The code is in the /src directory, and you will need to make some modifications to the Makefile in order to get the code to compile on your machine. You will need to set the path to the cuda SDK, and the path to your cuda intallation. 
aperture_mass/README:The directory examples/ contains all you need to run the code on your GPU. You will need to edit the run_mAp.py code to point to your input and output files, and to specify a few options. You can chose the critical radius of the NFW-like filter used in the aperture mass calculation, and where the aperture mass is evaluated. Your performance will vary, depending on how you set these parameters.
aperture_mass/src/mAp_gpu_grid.cu:#include <cuda.h>
aperture_mass/src/mAp_gpu_grid.cu:#include <cuda_runtime_api.h>
aperture_mass/src/mAp_gpu_grid.cu:void checkCUDAerror(const char *msg);
aperture_mass/src/mAp_gpu_grid.cu:      cudaMalloc(&d_test, testsize);
aperture_mass/src/mAp_gpu_grid.cu:      cudaError_t err = cudaGetLastError();
aperture_mass/src/mAp_gpu_grid.cu:      if( cudaSuccess != err){
aperture_mass/src/mAp_gpu_grid.cu:	printf("gotta wait for a bit!: %s\n",  cudaGetErrorString( err) );
aperture_mass/src/mAp_gpu_grid.cu:    // GPU memory for input 
aperture_mass/src/mAp_gpu_grid.cu:    cudaMalloc(&d_rgamma1, sizeneeded);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMalloc(&d_rgamma2, sizeneeded);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMalloc(&d_ra, sizeneeded);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMalloc(&d_dec, sizeneeded);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMalloc(&d_mAp_rgamma, sizeneeded_out);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMalloc(&d_var_rgamma, sizeneeded_out);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMalloc(&d_SN_rgamma, sizeneeded_out);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMemcpy(d_rgamma1, h_rgamma1, sizeneeded, cudaMemcpyHostToDevice);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMemcpy(d_rgamma2, h_rgamma2, sizeneeded, cudaMemcpyHostToDevice);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMemcpy(d_ra, h_ra, sizeneeded, cudaMemcpyHostToDevice);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMemcpy(d_dec, h_dec, sizeneeded, cudaMemcpyHostToDevice);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMemcpy(d_mAp_rgamma, h_mAp_rgamma, sizeneeded_out, cudaMemcpyHostToDevice);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMemcpy(d_var_rgamma, h_var_rgamma, sizeneeded_out, cudaMemcpyHostToDevice);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMemcpy(d_SN_rgamma, h_SN_rgamma, sizeneeded_out, cudaMemcpyHostToDevice);
aperture_mass/src/mAp_gpu_grid.cu:    checkCUDAerror("memory");
aperture_mass/src/mAp_gpu_grid.cu:    checkCUDAerror("kernel");
aperture_mass/src/mAp_gpu_grid.cu:    cudaMemcpy(h_mAp_rgamma, d_mAp_rgamma, sizeneeded_out, cudaMemcpyDeviceToHost);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMemcpy(h_var_rgamma, d_var_rgamma, sizeneeded_out, cudaMemcpyDeviceToHost);
aperture_mass/src/mAp_gpu_grid.cu:    cudaMemcpy(h_SN_rgamma, d_SN_rgamma, sizeneeded_out, cudaMemcpyDeviceToHost);
aperture_mass/src/mAp_gpu_grid.cu:void checkCUDAerror(const char *msg)
aperture_mass/src/mAp_gpu_grid.cu:  cudaError_t err = cudaGetLastError();
aperture_mass/src/mAp_gpu_grid.cu:  if( cudaSuccess != err) 
aperture_mass/src/mAp_gpu_grid.cu:      fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
aperture_mass/src/mAp_gpu_grid.cu:	      cudaGetErrorString( err) );
aperture_mass/src/mAp_gpu_grid.cu://  function to check whether GPU device has the specs to perform the calculation. 
aperture_mass/src/mAp_gpu_grid.cu://  adapted from cuda SDK deviceQuery example. 
aperture_mass/src/mAp_gpu_grid.cu:  int gpu_mem_needed = int(number_of_galaxies * sizeof(float))*4 +  int(ncalc * sizeof(float))*3; // need to allocate gamma1, gamma2, ra, dec and output mAp and var and SN. 
aperture_mass/src/mAp_gpu_grid.cu:  printf("Requirements: %d calculations and %d bytes memory on the GPU \n\n", ncalc, gpu_mem_needed);  
aperture_mass/src/mAp_gpu_grid.cu:  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
aperture_mass/src/mAp_gpu_grid.cu:  if (error_id != cudaSuccess) {
aperture_mass/src/mAp_gpu_grid.cu:    printf( "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
aperture_mass/src/mAp_gpu_grid.cu:  // This function call returns 0 if there are no CUDA capable devices.
aperture_mass/src/mAp_gpu_grid.cu:    printf("There is no device supporting CUDA\n");
aperture_mass/src/mAp_gpu_grid.cu:    printf("Found %d CUDA Capable device(s)\n", deviceCount); 
aperture_mass/src/mAp_gpu_grid.cu:    cudaDeviceProp deviceProp;
aperture_mass/src/mAp_gpu_grid.cu:    cudaGetDeviceProperties(&deviceProp, dev);
aperture_mass/src/mAp_gpu_grid.cu:    if((unsigned long long) deviceProp.totalGlobalMem < gpu_mem_needed) {
aperture_mass/src/Makefile:# perhaps need to edit, depending on your CUDA installation.
aperture_mass/src/Makefile:# INSTALL_DIR is where the CUDA and C executables will be installed. You may find it 
aperture_mass/src/Makefile:# SDK_INSTALL_PATH and CUDA_INSTALL_PATH are probably the more significant
aperture_mass/src/Makefile:# environment variables and will depend on your system and where the CUDA
aperture_mass/src/Makefile:#CUDA_SDK = /path/to/my/cuda/sdk/installation
aperture_mass/src/Makefile:CUDA_SDK  := ${HOME}/CUDA/SDK_4.0/
aperture_mass/src/Makefile:### You will need to edit this line to point to your own cuda installation
aperture_mass/src/Makefile:CUDA_INSTALL_PATH := /opt/cuda
aperture_mass/src/Makefile:INCLUDES          += -I. -I$(CUDA_INSTALL_PATH)/include  -I$(CUDA_SDK)/C/common/inc 
aperture_mass/src/Makefile:LIBS		  := -L$(CUDA_INSTALL_PATH)/lib64  -L$(CUDA_SDK)/C/common/lib/ -L$(CUDA_SDK)/C/lib/ 
aperture_mass/src/Makefile:LDFLAGS           := -lm -lcudart
aperture_mass/src/Makefile:NVCC              := $(CUDA_INSTALL_PATH)/bin/nvcc -arch sm_11 $(COMMONFLAGS)
aperture_mass/src/Makefile:CU_SOURCES_GRID   := mAp_gpu_grid.cu
angular_correlation/utility_scripts/plot_angular_correlation_function.py:    tag = 'logbinning_GPU_100k'
angular_correlation/INSTALL:# Requrements for GPU code.
angular_correlation/INSTALL:CUDA: Tested with version 4.1. It is possible that this code will work with 
angular_correlation/INSTALL:compile both the GPU executable "angular_correlation" and the C-version
angular_correlation/INSTALL:    SDK_INSTALL_PATH  := ${HOME}/CUDA/SDK_4.0/
angular_correlation/INSTALL:    CUDA_INSTALL_PATH := /opt/cuda/
angular_correlation/INSTALL:You may have to edit thse paths to point to where your CUDA libraries and SDK
angular_correlation/INSTALL:    INCLUDES          += -I. -I$(SDK_INSTALL_PATH)/C/common/inc/ -I$(CUDA_INSTALL_PATH)/include/
angular_correlation/INSTALL:    LIBS              += -L. -L$(SDK_INSTALL_PATH)/C/lib/ -L$(CUDA_INSTALL_PATH)//usr/lib/x86_64-linux-gnu/
angular_correlation/INSTALL:    LIBS              += -L$(CUDA_INSTALL_PATH)/lib64
angular_correlation/INSTALL:We have found many CUDA installations to be somewhat different and rather than 
angular_correlation/INSTALL:guidance. If you are still having problems, please consult us or the NVIDIA 
angular_correlation/INSTALL:CUDA documentation.
angular_correlation/examples/plot_output_of_GPU_calculation.csh:set tag = 'evenbinning_GPU_10k'
angular_correlation/examples/plot_output_of_GPU_calculation.csh:#set tag = 'evenbinning_GPU_100k'
angular_correlation/examples/plot_output_of_GPU_calculation.csh:#set tag = 'log10binning_GPU_10k'
angular_correlation/examples/plot_output_of_GPU_calculation.csh:#set tag = 'log10binning_GPU_100k'
angular_correlation/examples/plot_output_of_GPU_calculation.csh:#set tag = 'logbinning_GPU_10k'
angular_correlation/examples/plot_output_of_GPU_calculation.csh:#set tag = 'logbinning_GPU_100k'
angular_correlation/examples/compare_single_and_double_CPU.csh:echo "Finished running the same calculation on the CPU and GPU!"
angular_correlation/examples/run_GPU_calculation.csh:#set tag = 'evenbinning_GPU'
angular_correlation/examples/run_GPU_calculation.csh:#set tag = 'evenbinning_GPU_cartesian_allrandom_width1Mpc'
angular_correlation/examples/run_GPU_calculation.csh:#set tag = 'evenbinning_GPU_cartesian_abigrandom_width2Mpc'
angular_correlation/examples/run_GPU_calculation.csh:#set tag = 'evenbinning_GPU_cartesian_amockdat_width2Mpc'
angular_correlation/examples/run_GPU_calculation.csh:set tag = 'evenbinning_GPU_cartesian_amockdat_width1Mpc_less0.5'
angular_correlation/examples/run_GPU_calculation.csh:#set tag = 'evenbinning_GPU_cartesian_DJB_files_10k100k'
angular_correlation/examples/run_GPU_calculation.csh:#set tag = 'logbinning_GPU'
angular_correlation/examples/run_GPU_calculation.csh:#set tag = 'log10binning_GPU'
angular_correlation/examples/compare_diff_D_R.csh:set gpu_executable = $BIN_DIR/'angular_correlation'
angular_correlation/examples/compare_diff_D_R.csh:time $gpu_executable $data $flat0 -S $global_params -o GPU_"$tag"_10k_data_flat_arcmin.dat 
angular_correlation/examples/compare_diff_D_R.csh:time $gpu_executable $data $flat1 -S $global_params -o GPU_"$tag"_20k_data_flat_arcmin.dat 
angular_correlation/examples/compare_diff_D_R.csh:#echo "Finished running the same calculation on the CPU and GPU!"
angular_correlation/examples/compare_diff_D_R.csh:#sdiff GPU_"$tag"_"$ngals"k_data_flat_arcmin.dat CPU_"$tag"_"$ngals"k_data_flat_arcmin.dat | grep '|' | awk '{print $1" "$2"\t"$3"\t"$7"  "$7-$3" \t"($7-$3)/$3}'
angular_correlation/examples/compare_GPU_and_CPU.csh:set gpu_executable = $BIN_DIR/'angular_correlation'
angular_correlation/examples/compare_GPU_and_CPU.csh:#time $gpu_executable $data $flat $global_params -o GPU_"$tag"_"$ngals"k_data_flat_arcmin.dat 
angular_correlation/examples/compare_GPU_and_CPU.csh:#echo "Finished running the same calculation on the CPU and GPU!"
angular_correlation/examples/compare_GPU_and_CPU.csh:#sdiff GPU_"$tag"_"$ngals"k_data_flat_arcmin.dat CPU_"$tag"_"$ngals"k_data_flat_arcmin.dat | grep '|' | awk '{print $1" "$2"\t"$3"\t"$7"  "$7-$3" \t"($7-$3)/$3}'
angular_correlation/README:This package uses the CUDA programming language to calculate the two-point
angular_correlation/README:angular correlation function (ACF) on a GPU. Details about the function itself
angular_correlation/README:DD, RR, and DR with separate executions of the GPU code. Each calculation must
angular_correlation/README:The number of bins is hard coded in the GPU (and accompanying C-code) code. 
angular_correlation/README:# Compare CPU and GPU
angular_correlation/README:To do a quick comparison of the GPU and CPU code you can type
angular_correlation/README:    csh compare_GPU_and_CPU.csh
angular_correlation/README:    csh compare_GPU_and_CPU.csh 100
angular_correlation/README:# Run the full 2-pt angular correlation function on the GPU
angular_correlation/README:To run the DD, RR, and DR calculations for the 10k files on the GPU
angular_correlation/README:    csh run_GPU_calculation.csh
angular_correlation/README:    csh plot_output_of_GPU_calculation.csh
angular_correlation/README:# Further details about the GPU implementaion
angular_correlation/README:Command line options for the GPU implementation are as follows.
angular_correlation/README:-S         Silent output for the GPU diagnostic info.
angular_correlation/README:As mentioned previously, the GPU code merely computes the DD, RR, and RR terms
angular_correlation/src/Makefile:# perhaps need to edit, depending on your CUDA installation.
angular_correlation/src/Makefile:# INSTALL_DIR is where the CUDA and C executables will be installed. You may find it 
angular_correlation/src/Makefile:# SDK_INSTALL_PATH and CUDA_INSTALL_PATH are probably the more significant
angular_correlation/src/Makefile:# environment variables and will depend on your system and where the CUDA
angular_correlation/src/Makefile:#SDK_INSTALL_PATH  := ${HOME}/CUDA/SDK_4.0/
angular_correlation/src/Makefile:#CUDA_INSTALL_PATH := /opt/cuda/
angular_correlation/src/Makefile:#SDK_INSTALL_PATH  := $(HOME)/NVIDIA_GPU_Computing_SDK
angular_correlation/src/Makefile:#CUDA_INSTALL_PATH := /usr/local/cuda/
angular_correlation/src/Makefile:CUDA_INSTALL_PATH := /usr/local/cuda-6.5/
angular_correlation/src/Makefile:#CUDA_INSTALL_PATH := /usr/local/cuda-5.0/
angular_correlation/src/Makefile:#CUDA_INSTALL_PATH := /usr/local/cuda/
angular_correlation/src/Makefile:#CUDA_INSTALL_PATH := /usr/local/cuda-5.0/
angular_correlation/src/Makefile:#CUDA_INSTALL_PATH := /usr/local/cuda-7.0/
angular_correlation/src/Makefile:#SDK_INSTALL_PATH  := ${CUDA_INSTALL_PATH}/samples/
angular_correlation/src/Makefile:INCLUDES          += -I. -I$(SDK_INSTALL_PATH)/C/common/inc/ -I$(CUDA_INSTALL_PATH)/include/
angular_correlation/src/Makefile:LIBS              += -L. -L$(SDK_INSTALL_PATH)/C/lib/ -L$(CUDA_INSTALL_PATH)//usr/lib/x86_64-linux-gnu/
angular_correlation/src/Makefile:LIBS              += -L$(CUDA_INSTALL_PATH)/lib64
angular_correlation/src/Makefile:LDFLAGS := -lcudart
angular_correlation/src/Makefile:NVCC              := $(CUDA_INSTALL_PATH)/bin/nvcc -arch sm_20 $(COMMONFLAGS)
angular_correlation/src/angular_correlation.cu:#include<cuda_runtime.h>
angular_correlation/src/angular_correlation.cu:int doCalcRaDec(FILE *infile0, FILE *infile1, FILE *outfile, bool silent_on_GPU_testing, float scale_factor, int nbins, float hist_lower_range, float hist_upper_range, float hist_bin_width, int log_binning_flag, bool two_different_files, float conv_factor_angle);
angular_correlation/src/angular_correlation.cu:int doCalcMpc(FILE *infile0, FILE *infile1, FILE *outfile, bool silent_on_GPU_testing, float scale_factor, int nbins, float hist_lower_range, float hist_upper_range, float hist_bin_width, int log_binning_flag, bool two_different_files, float conv_factor_angle);
angular_correlation/src/angular_correlation.cu:    bool silent_on_GPU_testing = false;
angular_correlation/src/angular_correlation.cu:    int cuda_device = 0;
angular_correlation/src/angular_correlation.cu:                cuda_device = atoi(optarg); // Use this CUDA device.
angular_correlation/src/angular_correlation.cu:                printf("Will attempt to use CUDA device %d\n",cuda_device);
angular_correlation/src/angular_correlation.cu:                printf("Silent mode - don't run the GPU test (suppresses some output)\n");
angular_correlation/src/angular_correlation.cu:                silent_on_GPU_testing = true;
angular_correlation/src/angular_correlation.cu:    // Set the CUDA device. This is useful if your machine has multiple GPUs
angular_correlation/src/angular_correlation.cu:    cudaError_t error_id = cudaSetDevice(cuda_device);
angular_correlation/src/angular_correlation.cu:    if (error_id == cudaSuccess) {
angular_correlation/src/angular_correlation.cu:        printf( "cudaSetDevice returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
angular_correlation/src/angular_correlation.cu:        printf( "cudaSetDevice failed on Device %d!\n\n",cuda_device);
angular_correlation/src/angular_correlation.cu:    if(radec_input==1) int success = doCalcRaDec(infile0, infile1, outfile, silent_on_GPU_testing, scale_factor, nbins, hist_lower_range, hist_upper_range, hist_bin_width, log_binning_flag, two_different_files, conv_factor_angle);
angular_correlation/src/angular_correlation.cu:    else  int success = doCalcMpc(infile0, infile1, outfile, silent_on_GPU_testing, scale_factor, nbins, hist_lower_range, hist_upper_range, hist_bin_width, log_binning_flag, two_different_files, conv_factor_angle);
angular_correlation/src/angular_correlation.cu:int doCalcRaDec(FILE *infile0, FILE *infile1, FILE *outfile, bool silent_on_GPU_testing, float scale_factor, int nbins, float hist_lower_range, float hist_upper_range, float hist_bin_width, int log_binning_flag, bool two_different_files, float conv_factor_angle){
angular_correlation/src/angular_correlation.cu: if (!silent_on_GPU_testing) getDeviceDiagnostics(NUM_GALAXIES0+NUM_GALAXIES1, 2);
angular_correlation/src/angular_correlation.cu:    cudaMalloc((void **) &dev_hist, (size_hist_bytes));
angular_correlation/src/angular_correlation.cu:    cudaMemset(dev_hist, 0, size_hist_bytes);
angular_correlation/src/angular_correlation.cu:    cudaMalloc((void **) &d_alpha0, size_of_galaxy_array0 );
angular_correlation/src/angular_correlation.cu:    cudaMalloc((void **) &d_delta0, size_of_galaxy_array0 );
angular_correlation/src/angular_correlation.cu:    cudaMalloc((void **) &d_alpha1, size_of_galaxy_array1 );
angular_correlation/src/angular_correlation.cu:    cudaMalloc((void **) &d_delta1, size_of_galaxy_array1 );
angular_correlation/src/angular_correlation.cu:    cudaMemset(d_alpha0,0,size_of_galaxy_array0);
angular_correlation/src/angular_correlation.cu:    cudaMemset(d_delta0,0,size_of_galaxy_array0);
angular_correlation/src/angular_correlation.cu:    cudaMemset(d_alpha1,0,size_of_galaxy_array1);
angular_correlation/src/angular_correlation.cu:    cudaMemset(d_delta1,0,size_of_galaxy_array1);
angular_correlation/src/angular_correlation.cu:    cudaMemcpy(d_alpha0, h_alpha0, size_of_galaxy_array0, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation.cu:    cudaMemcpy(d_delta0, h_delta0, size_of_galaxy_array0, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation.cu:    cudaMemcpy(d_alpha1, h_alpha1, size_of_galaxy_array1, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation.cu:    cudaMemcpy(d_delta1, h_delta1, size_of_galaxy_array1, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation.cu:            cudaMemset(dev_hist,0,size_hist_bytes);
angular_correlation/src/angular_correlation.cu:            cudaMemcpy(hist, dev_hist, size_hist_bytes, cudaMemcpyDeviceToHost);
angular_correlation/src/angular_correlation.cu:    cudaFree(d_alpha0);
angular_correlation/src/angular_correlation.cu:    cudaFree(d_delta0);  
angular_correlation/src/angular_correlation.cu:    cudaFree(d_alpha1);
angular_correlation/src/angular_correlation.cu:    cudaFree(d_delta1);  
angular_correlation/src/angular_correlation.cu:    cudaFree(dev_hist);
angular_correlation/src/angular_correlation.cu:int doCalcMpc(FILE *infile0, FILE *infile1, FILE *outfile, bool silent_on_GPU_testing, float scale_factor, int nbins, float hist_lower_range, float hist_upper_range, float hist_bin_width, int log_binning_flag, bool two_different_files, float conv_factor_angle){
angular_correlation/src/angular_correlation.cu: if (!silent_on_GPU_testing) getDeviceDiagnostics(NUM_GALAXIES0+NUM_GALAXIES1, 2);
angular_correlation/src/angular_correlation.cu:    cudaMalloc((void **) &dev_hist, (size_hist_bytes));
angular_correlation/src/angular_correlation.cu:    cudaMemset(dev_hist, 0, size_hist_bytes);
angular_correlation/src/angular_correlation.cu:    cudaMalloc((void **) &d_x0, size_of_galaxy_array0 );
angular_correlation/src/angular_correlation.cu:    cudaMalloc((void **) &d_y0, size_of_galaxy_array0 );
angular_correlation/src/angular_correlation.cu:    cudaMalloc((void **) &d_z0, size_of_galaxy_array0 );
angular_correlation/src/angular_correlation.cu:    cudaMalloc((void **) &d_x1, size_of_galaxy_array1 );
angular_correlation/src/angular_correlation.cu:    cudaMalloc((void **) &d_y1, size_of_galaxy_array1 );
angular_correlation/src/angular_correlation.cu:    cudaMalloc((void **) &d_z1, size_of_galaxy_array1 );
angular_correlation/src/angular_correlation.cu:    cudaMemset(d_x0,0,size_of_galaxy_array0);
angular_correlation/src/angular_correlation.cu:    cudaMemset(d_y0,0,size_of_galaxy_array0);
angular_correlation/src/angular_correlation.cu:    cudaMemset(d_z0,0,size_of_galaxy_array0);
angular_correlation/src/angular_correlation.cu:    cudaMemset(d_x1,0,size_of_galaxy_array1);
angular_correlation/src/angular_correlation.cu:    cudaMemset(d_y1,0,size_of_galaxy_array1);
angular_correlation/src/angular_correlation.cu:    cudaMemset(d_z1,0,size_of_galaxy_array1);
angular_correlation/src/angular_correlation.cu:    cudaMemcpy(d_x0, h_x0, size_of_galaxy_array0, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation.cu:    cudaMemcpy(d_y0, h_y0, size_of_galaxy_array0, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation.cu:    cudaMemcpy(d_z0, h_z0, size_of_galaxy_array0, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation.cu:    cudaMemcpy(d_x1, h_x1, size_of_galaxy_array1, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation.cu:    cudaMemcpy(d_y1, h_y1, size_of_galaxy_array1, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation.cu:    cudaMemcpy(d_z1, h_z1, size_of_galaxy_array1, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation.cu:            cudaMemset(dev_hist,0,size_hist_bytes);
angular_correlation/src/angular_correlation.cu:            cudaMemcpy(hist, dev_hist, size_hist_bytes, cudaMemcpyDeviceToHost);
angular_correlation/src/angular_correlation.cu:    cudaFree(d_x0);
angular_correlation/src/angular_correlation.cu:    cudaFree(d_y0);
angular_correlation/src/angular_correlation.cu:    cudaFree(d_z0);  
angular_correlation/src/angular_correlation.cu:    cudaFree(d_x1);
angular_correlation/src/angular_correlation.cu:    cudaFree(d_y1);
angular_correlation/src/angular_correlation.cu:    cudaFree(d_z1);  
angular_correlation/src/angular_correlation.cu:    cudaFree(dev_hist);
angular_correlation/src/angular_correlation.cu:        printf("\n------ CUDA device diagnostics ------\n\n");
angular_correlation/src/angular_correlation.cu:        int gpu_mem_needed = int(tot_gals * sizeof(float)) * n_coords; // need to allocate ra, dec.
angular_correlation/src/angular_correlation.cu:        printf("Requirements: %d calculations and %d bytes memory on the GPU \n\n", ncalc, gpu_mem_needed);
angular_correlation/src/angular_correlation.cu:        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
angular_correlation/src/angular_correlation.cu:        if (error_id != cudaSuccess) {
angular_correlation/src/angular_correlation.cu:            printf( "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
angular_correlation/src/angular_correlation.cu:        // This function call returns 0 if there are no CUDA capable devices.
angular_correlation/src/angular_correlation.cu:            printf("There is no device supporting CUDA\n");
angular_correlation/src/angular_correlation.cu:            printf("Found %d CUDA Capable device(s)\n", deviceCount);
angular_correlation/src/angular_correlation.cu:            cudaDeviceProp deviceProp;
angular_correlation/src/angular_correlation.cu:            cudaGetDeviceProperties(&deviceProp, dev);
angular_correlation/src/angular_correlation.cu:            if((unsigned long long) deviceProp.totalGlobalMem < gpu_mem_needed) printf(" FAILURE: Not eneough memeory on device for this calculation! \n");
angular_correlation/src/angular_correlation.cu:                    int n_mem = floor(deviceProp.totalGlobalMem / float(gpu_mem_needed));
angular_correlation/src/angular_correlation.cu:        printf("\n------ End CUDA device diagnostics ------\n\n");
angular_correlation/src/angular_correlation_modified_grid.cu:#include<cuda_runtime.h>
angular_correlation/src/angular_correlation_modified_grid.cu:int doCalcRaDec(FILE *infile0, FILE *infile1, FILE *outfile, bool silent_on_GPU_testing, float scale_factor, int nbins, float hist_lower_range, float hist_upper_range, float hist_bin_width, int log_binning_flag, bool two_different_files, float conv_factor_angle);
angular_correlation/src/angular_correlation_modified_grid.cu:int doCalcMpc(FILE *infile0, FILE *infile1, FILE *outfile, bool silent_on_GPU_testing, float scale_factor, int nbins, float hist_lower_range, float hist_upper_range, float hist_bin_width, int log_binning_flag, bool two_different_files, float conv_factor_angle);
angular_correlation/src/angular_correlation_modified_grid.cu:    bool silent_on_GPU_testing = false;
angular_correlation/src/angular_correlation_modified_grid.cu:    int cuda_device = 0;
angular_correlation/src/angular_correlation_modified_grid.cu:                cuda_device = atoi(optarg); // Use this CUDA device.
angular_correlation/src/angular_correlation_modified_grid.cu:                printf("Will attempt to use CUDA device %d\n",cuda_device);
angular_correlation/src/angular_correlation_modified_grid.cu:                printf("Silent mode - don't run the GPU test (suppresses some output)\n");
angular_correlation/src/angular_correlation_modified_grid.cu:                silent_on_GPU_testing = true;
angular_correlation/src/angular_correlation_modified_grid.cu:    // Set the CUDA device. This is useful if your machine has multiple GPUs
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaError_t error_id = cudaSetDevice(cuda_device);
angular_correlation/src/angular_correlation_modified_grid.cu:    if (error_id == cudaSuccess) {
angular_correlation/src/angular_correlation_modified_grid.cu:        printf( "cudaSetDevice returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
angular_correlation/src/angular_correlation_modified_grid.cu:        printf( "cudaSetDevice failed on Device %d!\n\n",cuda_device);
angular_correlation/src/angular_correlation_modified_grid.cu:    if(radec_input==1) int success = doCalcRaDec(infile0, infile1, outfile, silent_on_GPU_testing, scale_factor, nbins, hist_lower_range, hist_upper_range, hist_bin_width, log_binning_flag, two_different_files, conv_factor_angle);
angular_correlation/src/angular_correlation_modified_grid.cu:    else  int success = doCalcMpc(infile0, infile1, outfile, silent_on_GPU_testing, scale_factor, nbins, hist_lower_range, hist_upper_range, hist_bin_width, log_binning_flag, two_different_files, conv_factor_angle);
angular_correlation/src/angular_correlation_modified_grid.cu:int doCalcRaDec(FILE *infile0, FILE *infile1, FILE *outfile, bool silent_on_GPU_testing, float scale_factor, int nbins, float hist_lower_range, float hist_upper_range, float hist_bin_width, int log_binning_flag, bool two_different_files, float conv_factor_angle){
angular_correlation/src/angular_correlation_modified_grid.cu: if (!silent_on_GPU_testing) getDeviceDiagnostics(NUM_GALAXIES0+NUM_GALAXIES1, 2);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMalloc((void **) &dev_hist, (size_hist_bytes));
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemset(dev_hist, 0, size_hist_bytes);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMalloc((void **) &d_alpha0, size_of_galaxy_array0 );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMalloc((void **) &d_delta0, size_of_galaxy_array0 );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMalloc((void **) &d_alpha1, size_of_galaxy_array1 );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMalloc((void **) &d_delta1, size_of_galaxy_array1 );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemset(d_alpha0,0,size_of_galaxy_array0);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemset(d_delta0,0,size_of_galaxy_array0);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemset(d_alpha1,0,size_of_galaxy_array1);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemset(d_delta1,0,size_of_galaxy_array1);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemcpy(d_alpha0, h_alpha0, size_of_galaxy_array0, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemcpy(d_delta0, h_delta0, size_of_galaxy_array0, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemcpy(d_alpha1, h_alpha1, size_of_galaxy_array1, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemcpy(d_delta1, h_delta1, size_of_galaxy_array1, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation_modified_grid.cu:            cudaMemset(dev_hist,0,size_hist_bytes);
angular_correlation/src/angular_correlation_modified_grid.cu:            cudaMemcpy(hist, dev_hist, size_hist_bytes, cudaMemcpyDeviceToHost);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaFree(d_alpha0);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaFree(d_delta0);  
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaFree(d_alpha1);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaFree(d_delta1);  
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaFree(dev_hist);
angular_correlation/src/angular_correlation_modified_grid.cu:int doCalcMpc(FILE *infile0, FILE *infile1, FILE *outfile, bool silent_on_GPU_testing, float scale_factor, int nbins, float hist_lower_range, float hist_upper_range, float hist_bin_width, int log_binning_flag, bool two_different_files, float conv_factor_angle){
angular_correlation/src/angular_correlation_modified_grid.cu: if (!silent_on_GPU_testing) getDeviceDiagnostics(NUM_GALAXIES0+NUM_GALAXIES1, 2);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMalloc((void **) &dev_hist, (size_hist_bytes));
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemset(dev_hist, 0, size_hist_bytes);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMalloc((void **) &d_x0, size_of_galaxy_array0 );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMalloc((void **) &d_y0, size_of_galaxy_array0 );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMalloc((void **) &d_z0, size_of_galaxy_array0 );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMalloc((void **) &d_x1, size_of_galaxy_array1 );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMalloc((void **) &d_y1, size_of_galaxy_array1 );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMalloc((void **) &d_z1, size_of_galaxy_array1 );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemset(d_x0,0,size_of_galaxy_array0);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemset(d_y0,0,size_of_galaxy_array0);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemset(d_z0,0,size_of_galaxy_array0);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemset(d_x1,0,size_of_galaxy_array1);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemset(d_y1,0,size_of_galaxy_array1);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemset(d_z1,0,size_of_galaxy_array1);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemcpy(d_x0, h_x0, size_of_galaxy_array0, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemcpy(d_y0, h_y0, size_of_galaxy_array0, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemcpy(d_z0, h_z0, size_of_galaxy_array0, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemcpy(d_x1, h_x1, size_of_galaxy_array1, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemcpy(d_y1, h_y1, size_of_galaxy_array1, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaMemcpy(d_z1, h_z1, size_of_galaxy_array1, cudaMemcpyHostToDevice );
angular_correlation/src/angular_correlation_modified_grid.cu:            cudaMemset(dev_hist,0,size_hist_bytes);
angular_correlation/src/angular_correlation_modified_grid.cu:            cudaMemcpy(hist, dev_hist, size_hist_bytes, cudaMemcpyDeviceToHost);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaFree(d_x0);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaFree(d_y0);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaFree(d_z0);  
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaFree(d_x1);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaFree(d_y1);
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaFree(d_z1);  
angular_correlation/src/angular_correlation_modified_grid.cu:    cudaFree(dev_hist);
angular_correlation/src/angular_correlation_modified_grid.cu:        printf("\n------ CUDA device diagnostics ------\n\n");
angular_correlation/src/angular_correlation_modified_grid.cu:        int gpu_mem_needed = int(tot_gals * sizeof(float)) * n_coords; // need to allocate ra, dec.
angular_correlation/src/angular_correlation_modified_grid.cu:        printf("Requirements: %d calculations and %d bytes memory on the GPU \n\n", ncalc, gpu_mem_needed);
angular_correlation/src/angular_correlation_modified_grid.cu:        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
angular_correlation/src/angular_correlation_modified_grid.cu:        if (error_id != cudaSuccess) {
angular_correlation/src/angular_correlation_modified_grid.cu:            printf( "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
angular_correlation/src/angular_correlation_modified_grid.cu:        // This function call returns 0 if there are no CUDA capable devices.
angular_correlation/src/angular_correlation_modified_grid.cu:            printf("There is no device supporting CUDA\n");
angular_correlation/src/angular_correlation_modified_grid.cu:            printf("Found %d CUDA Capable device(s)\n", deviceCount);
angular_correlation/src/angular_correlation_modified_grid.cu:            cudaDeviceProp deviceProp;
angular_correlation/src/angular_correlation_modified_grid.cu:            cudaGetDeviceProperties(&deviceProp, dev);
angular_correlation/src/angular_correlation_modified_grid.cu:            if((unsigned long long) deviceProp.totalGlobalMem < gpu_mem_needed) printf(" FAILURE: Not eneough memeory on device for this calculation! \n");
angular_correlation/src/angular_correlation_modified_grid.cu:                    int n_mem = floor(deviceProp.totalGlobalMem / float(gpu_mem_needed));
angular_correlation/src/angular_correlation_modified_grid.cu:        printf("\n------ End CUDA device diagnostics ------\n\n");
README:the GPU. Details of the calculations themselves and the implementation can

```
