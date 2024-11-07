# https://github.com/AstroAccelerateOrg/astro-accelerate

```console
meson_srclist.txt:src/aa_bin_gpu.cpp
wiki/dedispersion_kernel_parameter_optimisation.md:The final speedup gained by using different optimised parameters for each DM ranges as opposed to the same parameters for all ranges was modest. As such, it is not recommended unless the code will be running unchanged on the same system for a long period of time. For the b2 band, the final speedup factor averaged over 5 trials was 6.28 with DM range specific optimisation vs 5.98 for generic parameters, an improvement in dpeedup factor of 5\%. This speedup factor includes time spent allocating global GPU memory and copying data to the GPU at the beginning of every time chunk. It was thought that the effect of individual kernel optimisation may be more pronounced for lower frequencies, where different DM ranges vary more significantly in aspect ratio. However, the results for this band were similar to the B2 band, with a speedup factor of 5.08 with DM range specific optimisation vs 4.75 for generic parameters, an again small improvement of 7.1\%.
wiki/home.md:2. A library that can be used to enable GPU accelerated single pulse processing (SPS) or Fourier Domain Acceleration Searching (FDAS).
wiki/home.md:AstroAccelerate is a GPU-enabled software package that focuses on enabling real-time processing of time-domain radio-astronomy data. It uses the CUDA programming language for NVIDIA GPUs.  
wiki/home.md:The massive computational power of modern day GPUs allows the code to perform algorithms such as dedispersion, single pulse searching and Fourier Domain Acceleration Searching in real-time on very large data-sets which are comparable to those which will be produced by next generation radio-telescopes such as the SKA.  
wiki/using_input_files.md:_analysis_debug_ Uses a CPU code to perform analysis, this is used to debug our GPU code.
wiki/calculation_of_mean_and_standard_deviation.md:Calculation of mean and standard deviation (Stdev) is used in many places within AstroAccelerate (single pulse search, acceleration search, periodicity search). For our GPU implementation of standard deviation we have used a streaming algorithm by T. F. Chan 1983 which is using pair-wise summation for better numerical stability. We have also implemented optional point-wise outlier rejection which attempts to find true value of standard deviation of the underlying background noise or base level noise by removing points which are above used given multiple of standard deviation _n_. You can switch outlier rejection on by enabling _baselinenoise_ in input text file, value of _n_ is given by _sigma_constant_ parameter also in input text file.
python/singlepulse_with_aa.py:#Need to be enabled otherwise there will be no data copy from GPU memory to host memory
python/singlepulse_with_aa.py:# Select GPU card number on this machine
python/dedisp_with_aa.py:#Need to be enabled otherwise there will be no data copy from GPU memory to host memory
python/dedisp_with_aa.py:# Select GPU card number on this machine
python/use_python.py:# Select GPU card number on this machine
.zenodo.json:	"GPU",
.zenodo.json:	"CUDA",
.zenodo.json:	"GPGPU", 
tests/test_periodicity_strategy_1.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/include/)
tests/test_periodicity_strategy_1.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/samples/common/inc/)
tests/test_periodicity_strategy_1.cmake:target_link_libraries(${TEST_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_curand_LIBRARY} astroaccelerate)
tests/test_ddtr_fakesignal_period.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/include/)
tests/test_ddtr_fakesignal_period.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/samples/common/inc/)
tests/test_ddtr_fakesignal_period.cmake:target_link_libraries(${TEST_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_curand_LIBRARY} astroaccelerate)
tests/test_single_pulse_search_scan_1.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/include/)
tests/test_single_pulse_search_scan_1.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/samples/common/inc/)
tests/test_single_pulse_search_scan_1.cmake:target_link_libraries(${TEST_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_curand_LIBRARY} astroaccelerate)
tests/test_device_memory_request.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/include/)
tests/test_device_memory_request.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/samples/common/inc/)
tests/test_device_memory_request.cmake:target_link_libraries(${TEST_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_curand_LIBRARY} astroaccelerate)
tests/test_ddtr_fakesignal_period.cpp:  // Init the GPU card
tests/test_ddtr_fakesignal_period.cpp:  const size_t free_memory = selected_device.free_memory(); // Free memory on the GPU in bytes
tests/test_ddtr_strategy_2.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/include/)
tests/test_ddtr_strategy_2.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/samples/common/inc/)
tests/test_ddtr_strategy_2.cmake:target_link_libraries(${TEST_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_curand_LIBRARY} astroaccelerate)
tests/test_single_pulse_search_scan_1.cpp:int CompareData(float *CPU_result, float *GPU_result, float CPU_scale, float GPU_scale, int CPU_offset, int GPU_offset, int CPU_dim_x, int GPU_dim_x, int dim_y, int nSamples, double *total_error, double *mean_error){
tests/test_single_pulse_search_scan_1.cpp:	printf("DEBUG: CompareData variables: CPU_dim_x=%d; GPU_dim_x=%d; dim_y=%d; nSamples=%d;\n", CPU_dim_x, GPU_dim_x, dim_y, nSamples);
tests/test_single_pulse_search_scan_1.cpp:			int GPU_pos = y*GPU_dim_x + x + GPU_offset;
tests/test_single_pulse_search_scan_1.cpp:			float CPU, GPU;
tests/test_single_pulse_search_scan_1.cpp:			GPU = GPU_result[GPU_pos]/GPU_scale;
tests/test_single_pulse_search_scan_1.cpp:			error = get_error(CPU, GPU);
tests/test_single_pulse_search_scan_1.cpp:					printf("Error [%f] CPU=%f; GPU=%f; x=%d; y=%d;\n", error, CPU, GPU, x, y);
tests/test_single_pulse_search_scan_1.cpp:int CompareDataTaps(ushort *CPU_result, ushort *GPU_result, float CPU_scale, float GPU_scale, int CPU_offset, int GPU_offset, int CPU_dim_x, int GPU_dim_x, int dim_y, int nSamples, double *total_error, double *mean_error){
tests/test_single_pulse_search_scan_1.cpp:	printf("DEBUG: CompareData variables: CPU_dim_x=%d; GPU_dim_x=%d; dim_y=%d; nSamples=%d;\n", CPU_dim_x, GPU_dim_x, dim_y, nSamples);
tests/test_single_pulse_search_scan_1.cpp:			int GPU_pos = y*GPU_dim_x + x + GPU_offset;
tests/test_single_pulse_search_scan_1.cpp:			float CPU, GPU;
tests/test_single_pulse_search_scan_1.cpp:			GPU = (float) GPU_result[GPU_pos]/GPU_scale;
tests/test_single_pulse_search_scan_1.cpp:			error = abs(CPU - GPU);
tests/test_single_pulse_search_scan_1.cpp:					printf("Error [%f] CPU=%f; GPU=%f; x=%d; y=%d;\n", error, CPU, GPU, x, y);
tests/test_single_pulse_search_scan_1.cpp:int CompareData(float *CPU_result, float *GPU_result, size_t dim_x, size_t dim_y);
tests/test_single_pulse_search_scan_1.cpp:int check_boxcar_results(float *h_input, float *h_MSD_interpolated, float *h_GPU_boxcar_values, float *h_GPU_decimated, float *h_GPU_output_SNR, ushort *h_GPU_output_taps, int nBoxcars, size_t nTimesamples, size_t nDMs){
tests/test_single_pulse_search_scan_1.cpp:	decimate_pass = CompareData(h_CPU_decimated, h_GPU_decimated, 1.0, 1.0, 0, 0, dec_nTimesamples, dec_nTimesamples, nDMs, dec_nTimesamples-(nBoxcars>>1), &total_error, &mean_error);
tests/test_single_pulse_search_scan_1.cpp:	boxcar_value_pass = CompareData(h_CPU_boxcar_values, h_GPU_boxcar_values, 1.0, 1.0, 0, 0, dec_nTimesamples, dec_nTimesamples, nDMs, dec_nTimesamples-(nBoxcars>>1), &total_error, &mean_error);
tests/test_single_pulse_search_scan_1.cpp:	SNR_pass = CompareData(h_CPU_output_SNR, h_GPU_output_SNR, 1.0, 1.0, 0, 0, nTimesamples, nTimesamples, nDMs, nTimesamples-nBoxcars, &total_error, &mean_error);
tests/test_single_pulse_search_scan_1.cpp:	taps_pass = CompareDataTaps(h_CPU_output_taps, h_GPU_output_taps, 1.0, 1.0, 0, 0, nTimesamples, nTimesamples, nDMs, nTimesamples-nBoxcars, &total_error, &mean_error);
tests/test_single_pulse_search_scan_1.cpp:	cudaSetDevice(CARD);
tests/test_single_pulse_search_scan_1.cpp:	// GPU memory
tests/test_single_pulse_search_scan_1.cpp:	cudaMemGetInfo(&free_mem,&total_mem);
tests/test_single_pulse_search_scan_1.cpp:	// Memory allocation on GPU
tests/test_single_pulse_search_scan_1.cpp:	if ( cudaSuccess != cudaMalloc((void **) &d_input,  sizeof(float)*full_size)) error = -1;
tests/test_single_pulse_search_scan_1.cpp:	if ( cudaSuccess != cudaMalloc((void **) &d_decimated,  sizeof(float)*half_size)) error = -1;
tests/test_single_pulse_search_scan_1.cpp:	if ( cudaSuccess != cudaMalloc((void **) &d_boxcar_values,  sizeof(float)*2*half_size)) error = -1;
tests/test_single_pulse_search_scan_1.cpp:	if ( cudaSuccess != cudaMalloc((void **) &d_MSD_interpolated,  sizeof(float)*2*nBoxcars)) error = -1;
tests/test_single_pulse_search_scan_1.cpp:	if ( cudaSuccess != cudaMalloc((void **) &d_output_SNR, sizeof(float)*2*full_size)) error = -1;
tests/test_single_pulse_search_scan_1.cpp:	if ( cudaSuccess != cudaMalloc((void **) &d_output_taps, sizeof(ushort)*2*full_size)) error = -1;
tests/test_single_pulse_search_scan_1.cpp:	if ( cudaSuccess != cudaMemcpy(d_input, h_input, full_size*sizeof(float), cudaMemcpyHostToDevice)) error=-1;
tests/test_single_pulse_search_scan_1.cpp:	if ( cudaSuccess != cudaMemcpy(d_MSD_interpolated, h_MSD_interpolated, 2*nBoxcars*sizeof(float), cudaMemcpyHostToDevice)) error = -1;
tests/test_single_pulse_search_scan_1.cpp:	// GPU single pulse
tests/test_single_pulse_search_scan_1.cpp:	call_kernel_SPDT_GPU_1st_plane(gridSize, blockSize, d_input, d_boxcar_values, d_decimated, d_output_SNR, d_output_taps, (float2 *) d_MSD_interpolated, nTimesamples, nBoxcars, dtm);
tests/test_single_pulse_search_scan_1.cpp:	if ( cudaSuccess != cudaMemcpy(h_boxcar_values, d_boxcar_values, full_size*sizeof(float), cudaMemcpyDeviceToHost)) error=-1;
tests/test_single_pulse_search_scan_1.cpp:	if ( cudaSuccess != cudaMemcpy(h_decimated, d_decimated, half_size*sizeof(float), cudaMemcpyDeviceToHost)) error = -1;
tests/test_single_pulse_search_scan_1.cpp:	if ( cudaSuccess != cudaMemcpy(h_output_SNR, d_output_SNR, 2*full_size*sizeof(float), cudaMemcpyDeviceToHost)) error = -1;
tests/test_single_pulse_search_scan_1.cpp:	if ( cudaSuccess != cudaMemcpy(h_output_taps, d_output_taps, 2*full_size*sizeof(ushort), cudaMemcpyDeviceToHost)) error = -1;
tests/test_single_pulse_search_scan_1.cpp:	cudaFree(d_input);
tests/test_single_pulse_search_scan_1.cpp:	cudaFree(d_boxcar_values);
tests/test_single_pulse_search_scan_1.cpp:	cudaFree(d_decimated);
tests/test_single_pulse_search_scan_1.cpp:	cudaFree(d_output_SNR);
tests/test_single_pulse_search_scan_1.cpp:	cudaFree(d_output_taps);
tests/test_single_pulse_search_scan_1.cpp:	cudaFree(d_MSD_interpolated);
tests/test_ddtr_strategy_1.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/include/)
tests/test_ddtr_strategy_1.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/samples/common/inc/)
tests/test_ddtr_strategy_1.cmake:target_link_libraries(${TEST_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_curand_LIBRARY} astroaccelerate)
tests/test_ddtr_fakesignal_single.cpp:	//-----------------------  Init the GPU card
tests/test_ddtr_fakesignal_single.cpp:		const size_t free_memory = selected_device.free_memory(); // Free memory on the GPU in bytes
tests/test_ddtr_fakesignal_single.cpp:	        //insert option to copy the DDTR output data from GPU memory to the host memory
tests/test_ddtr_fakesignal_single.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/include/)
tests/test_ddtr_fakesignal_single.cmake:include_directories($ENV{CUDA_INSTALL_PATH}/samples/common/inc/)
tests/test_ddtr_fakesignal_single.cmake:target_link_libraries(${TEST_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_curand_LIBRARY} astroaccelerate)
CITATION.cff:  - "GPU"
CITATION.cff:  - "CUDA"
CITATION.cff:  - "GPGPU"
Makefile:CUDA	:= $(CUDA_INSTALL_PATH)
Makefile:INC	:= -I$(CUDA)include -I$(CUDA)samples/common/inc/
Makefile:LIB	:= $(CUDA)/lib64
Makefile:# CUDA code generation flags
Makefile:        --ptxas-options=-v -lcuda -lcudart  -lcurand -lcufft -lcudadevrt -Xptxas -dlcm=cg $(GENCODE_FLAGS) -Xcompiler -fopenmp
COLLABORATORS.md:# NVIDIA
README.md:* Adámek, K. and Armour, W., “Single-pulse Detection Algorithms for Real-time Fast Radio Burst Searches Using GPUs”, *The Astrophysical Journal Supplement Series*, vol. 247, no. 2, IOP, 2020. doi:10.3847/1538-4365/ab7994. Accessible from: https://iopscience.iop.org/article/10.3847/1538-4365/ab7994
README.md:* Dimoudi, S., Adamek, K., Thiagaraj, P., Ransom, S. M., Karastergiou, A., and Armour, W., “A GPU Implementation of the Correlation Technique for Real-time Fourier Domain Pulsar Acceleration Searches”, *The Astrophysical Journal Supplement Series*, vol. 239, no. 2, IOP, 2018. doi:10.3847/1538-4365/aabe88. Accessible from: https://iopscience.iop.org/article/10.3847/1538-4365/aabe88
README.md:* Armour, W., “A GPU-based Survey for Millisecond Radio Transients Using ARTEMIS”, in *Astronomical Data Analysis Software and Systems XXI*, 2012, vol. 461, p. 33. doi:10.48550/arXiv.1111.6399. Accessible from: https://aspbooks.org/custom/publications/paper/461-0033.html
README.md:Checking the Configuration of the Graphics Processing Unit (GPU) and Support for CUDA
README.md:    nvidia-smi
README.md:| NVIDIA-SMI 375.20                 Driver Version: 375.20                    |
README.md:| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
README.md:| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
README.md:| Processes:                                                       GPU Memory |
README.md:|  GPU       PID  Type  Process name                               Usage      |
README.md:If no output is shown, or if an error appears, then it may indicate that a GPU has not been detected,
README.md:or that the [CUDA toolkit](https://developer.nvidia.com/cuda-zone) is not properly installed.
README.md:Selecting the Graphics Processing Unit (GPU) in the case of a system with more than one GPU
README.md:If you have a multi-GPU system, you need can select the card by setting it in the input_file. The setting to add to the input_file is
README.md:where `X` is a non-negative integer number which corresponds to the ID number of the GPU on your machine.
README.md:CUDA: CUDA 8.0 (see https://developer.nvidia.com/cuda-downloads)
README.md:C/C++ (version): As supported and required by CUDA.
README.md:Compiler: As supported by CUDA, but requiring also [OpenMP](https://www.openmp.org) support (compiler support can be found [here](https://www.openmp.org/resources/openmp-compilers-tools/)).
README.md:Set-up the environment (which will add CUDA to PATH and LD_LIBRARY_PATH)
README.md:`setup.sh` to suit the CUDA version number, library paths, and the architecture number
README.md:in order to suit their needs. Users who already have all relevant CUDA paths configured
README.md:The CUDA architecture can be specified with the `-DCUDA_ARCH` flag. For example, for architecture `7.0`, do
README.md:    cmake -DCUDA_ARCH="7.0" ../
README.md:    cmake -DCUDA_ARCH="7.0" -DENABLE_TESTS=ON ../
README.md:    This will create an optimised code for your search and GPU type.
include/aa_device_single_pulse_search_kernel.hpp:  /** \brief Kernel wrapper function for PD_SEARCH_GPU kernel function. */
include/aa_device_single_pulse_search_kernel.hpp:  void call_kernel_PD_SEARCH_GPU(const dim3 &grid_size, const dim3 &block_size, const int &sm_size,
include/aa_device_single_FIR_kernel.hpp:  /** \brief Kernel wrapper function for PD_FIR_GPU kernel function. */
include/aa_device_single_FIR_kernel.hpp:  void call_kernel_PD_FIR_GPU(const dim3 &grid_size, const dim3 &block_size, const int &SM_size, float const *const d_input, float *const d_output, const int &nTaps, const int &nLoops, const int &nTimesamples);
include/aa_device_single_FIR_kernel.hpp:  /** \brief Kernel wrapper function for PD_FIR_GPUv1 kernel function. */
include/aa_device_single_FIR_kernel.hpp:  void call_kernel_PD_FIR_GPUv1(const dim3 &grid_size, const dim3 &block_size, const int &SM_size, float const *const d_input, float *const d_output, const int &nTaps, const int &nLoops, const unsigned int &nTimesamples);
include/aa_device_spectrum_whitening.hpp:extern void spectrum_whitening_SGP1(float *d_input, unsigned long int nSamples, int nDMs, cudaStream_t &stream);
include/aa_device_spectrum_whitening.hpp:extern void spectrum_whitening_SGP2(float2 *d_input, size_t nSamples, int nDMs, bool enable_median, cudaStream_t &stream);
include/aa_device_zero_dm_outliers_kernel.hpp:	const cudaStream_t &stream, 
include/aa_device_SPS_long_kernel.hpp:#include <cuda.h>
include/aa_device_SPS_long_kernel.hpp:#include <cuda_runtime.h>
include/aa_device_SPS_long_kernel.hpp:  /** \brief Kernel wrapper function for SPDT_GPU_1st_plane kernel function. */
include/aa_device_SPS_long_kernel.hpp:  void call_kernel_SPDT_GPU_1st_plane(const dim3 &grid_size, const dim3 &block_size, float const *const d_input,
include/aa_device_SPS_long_kernel.hpp:  /** \brief Kernel wrapper function for SPDT_GPU_Nth_plane kernel function. */
include/aa_device_SPS_long_kernel.hpp:  void call_kernel_SPDT_GPU_Nth_plane(const dim3 &grid_size, const dim3 &block_size, float const *const d_input,
include/aa_periodicity_candidates.hpp:	bool Copy_from_GPU(float *d_all){
include/aa_periodicity_candidates.hpp:		cudaError_t err = cudaMemcpy(all, d_all, c_nCandidates*c_nElements*sizeof(float), cudaMemcpyDeviceToHost);
include/aa_periodicity_candidates.hpp:		if(err != cudaSuccess) return(false);
include/aa_periodicity_candidates.hpp:		bool error = Copy_from_GPU(d_all);
include/aa_device_spectrum_whitening_kernel.hpp:		const cudaStream_t &stream, 
include/aa_device_spectrum_whitening_kernel.hpp:		const  cudaStream_t &stream, 
include/aa_device_spectrum_whitening_kernel.hpp:		const  cudaStream_t &stream, 
include/aa_device_spectrum_whitening_kernel.hpp:		const  cudaStream_t &stream, 
include/aa_permitted_pipelines_5.hpp:#include <cuda.h>
include/aa_permitted_pipelines_5.hpp:#include <cuda_runtime.h>
include/aa_permitted_pipelines_5.hpp:#include "aa_bin_gpu.hpp"
include/aa_permitted_pipelines_5.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_5.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_5.hpp:	cudaFree(d_input);
include/aa_permitted_pipelines_5.hpp:	cudaFree(d_output);
include/aa_permitted_pipelines_5.hpp:	cudaFree(m_d_MSD_workarea);
include/aa_permitted_pipelines_5.hpp:	cudaFree(m_d_MSD_output_taps);
include/aa_permitted_pipelines_5.hpp:	cudaFree(m_d_MSD_interpolated);
include/aa_permitted_pipelines_5.hpp:    aa_gpu_timer       m_timer;
include/aa_permitted_pipelines_5.hpp:    /** \brief Allocate the GPU memory needed for dedispersion. */
include/aa_permitted_pipelines_5.hpp:    void allocate_memory_gpu(const int &maxshift, const int &max_ndms, const int &nchans, int **const t_processed, unsigned short **const d_input, float **const d_output) {
include/aa_permitted_pipelines_5.hpp:      size_t gpu_inputsize = (size_t) time_samps * (size_t) nchans * sizeof(unsigned short);
include/aa_permitted_pipelines_5.hpp:      cudaError_t e = cudaMalloc((void **) d_input, gpu_inputsize);
include/aa_permitted_pipelines_5.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_5.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_5.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_5.hpp:      size_t gpu_outputsize = 0;
include/aa_permitted_pipelines_5.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)max_ndms * sizeof(float);
include/aa_permitted_pipelines_5.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)nchans * sizeof(float);
include/aa_permitted_pipelines_5.hpp:      e = cudaMalloc((void **) d_output, gpu_outputsize);
include/aa_permitted_pipelines_5.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_5.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_5.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_5.hpp:      cudaMemset(*d_output, 0, gpu_outputsize);
include/aa_permitted_pipelines_5.hpp:      cudaError_t e = cudaMalloc((void **) d_MSD_workarea,        MSD_maxtimesamples*5.5*sizeof(float));
include/aa_permitted_pipelines_5.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_5.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_5.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_5.hpp:      e = cudaMalloc((void **) &(*d_MSD_output_taps), sizeof(ushort)*2*MSD_maxtimesamples);
include/aa_permitted_pipelines_5.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_5.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_5.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_5.hpp:     e = cudaMalloc((void **) d_MSD_interpolated,    sizeof(float)*MSD_profile_size);
include/aa_permitted_pipelines_5.hpp:     if(e != cudaSuccess) {
include/aa_permitted_pipelines_5.hpp:       LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_5.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_5.hpp:      //Allocate GPU memory
include/aa_permitted_pipelines_5.hpp:      //Allocate GPU memory for dedispersion
include/aa_permitted_pipelines_5.hpp:      allocate_memory_gpu(maxshift, max_ndms, nchans, t_processed, &d_input, &d_output);
include/aa_permitted_pipelines_5.hpp:      //Allocate GPU memory for SPS (i.e. analysis)
include/aa_permitted_pipelines_5.hpp:	  printf("\n(Performed Brute-Force Dedispersion: %g (GPU estimate)", time);
include/aa_permitted_pipelines_5.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5.hpp:	printf("\nPerforming old GPU rfi...");
include/aa_permitted_pipelines_5.hpp:	rfi_gpu(d_input, nchans, t_processed[0][t]+maxshift);
include/aa_permitted_pipelines_5.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5.hpp:	cudaDeviceSynchronize();
include/aa_permitted_pipelines_5.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5.hpp:	  bin_gpu(d_input, d_output, nchans, t_processed[dm_range - 1][t] + maxshift * inBin[dm_range]);
include/aa_permitted_pipelines_5.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5.hpp:	analysis_GPU(h_peak_list_DM,
include/aa_permitted_pipelines_5.hpp:      aa_gpu_timer timer;
include/aa_permitted_pipelines_5.hpp:      GPU_periodicity(
include/aa_permitted_pipelines_5.hpp:      printf("\nPerformed Periodicity Location: %f (GPU estimate)", time);
include/aa_permitted_pipelines_5.hpp:      // Assumption: GPU memory is free and available.
include/aa_permitted_pipelines_5.hpp:      aa_gpu_timer timer;
include/aa_permitted_pipelines_5.hpp:      printf("\nPerformed Acceleration Location: %lf (GPU estimate)", time);
include/aa_jerk_plan.hpp:// conv_size should really be determined by the class based on size of the filter and performance of the GPU
include/simd_functions.hpp: * Copyright (c) 2013 NVIDIA Corporation. All rights reserved.
include/simd_functions.hpp: *   Neither the name of NVIDIA Corporation nor the names of its contributors
include/simd_functions.hpp:  operations, that are hardware accelerated on sm_3x (Kepler) GPUs. Efficient
include/simd_functions.hpp:  to make the code portable across all GPUs supported by CUDA. The following 
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#elif __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#elif __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ < 200
include/simd_functions.hpp:#elif __CUDA_ARCH__ < 350
include/simd_functions.hpp:#if __CUDA_ARCH__ < 200
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ < 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ < 350
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ < 350 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ < 350 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ < 350
include/simd_functions.hpp:#else  /*__CUDA_ARCH__ < 350 */
include/simd_functions.hpp:#endif /*__CUDA_ARCH__ < 350 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ < 200
include/simd_functions.hpp:#elif __CUDA_ARCH__ < 350
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ < 350 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ < 350 */
include/simd_functions.hpp:#if __CUDA_ARCH__ < 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ < 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ < 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ < 350
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ < 350 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ < 350 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /*  __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#elif __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#elif __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /*  __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 200
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 200 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#if __CUDA_ARCH__ >= 300
include/simd_functions.hpp:#else  /* __CUDA_ARCH__ >= 300 */
include/simd_functions.hpp:#endif /*  __CUDA_ARCH__ >= 300 */
include/aa_device_MSD_outlier_rejection_kernel.hpp:void call_kernel_MSD_GPU_calculate_partials_2d_and_minmax(const dim3 &grid_size, const dim3 &block_size, float const *const d_input, float *const d_output, const int &y_steps, const int &nTimesamples, const int &offset);
include/aa_permitted_pipelines_1.hpp:#include <cuda.h>
include/aa_permitted_pipelines_1.hpp:#include <cuda_runtime.h>
include/aa_permitted_pipelines_1.hpp:#include "aa_bin_gpu.hpp"
include/aa_permitted_pipelines_1.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_1.hpp:	cudaFree(d_input);
include/aa_permitted_pipelines_1.hpp:	cudaFree(d_output);
include/aa_permitted_pipelines_1.hpp:    aa_gpu_timer       m_timer;
include/aa_permitted_pipelines_1.hpp:    /** \brief Allocate the GPU memory needed for dedispersion. */
include/aa_permitted_pipelines_1.hpp:    void allocate_memory_gpu(const int &maxshift, const int &max_ndms, const int &nchans, int **const t_processed, unsigned short **const d_input, float **const d_output) {
include/aa_permitted_pipelines_1.hpp:      size_t gpu_inputsize = (size_t) time_samps * (size_t) nchans * sizeof(unsigned short);
include/aa_permitted_pipelines_1.hpp:      cudaError_t cuda_return = cudaMalloc((void **) d_input, gpu_inputsize);
include/aa_permitted_pipelines_1.hpp:      if(cuda_return != cudaSuccess) {
include/aa_permitted_pipelines_1.hpp:	LOG(log_level::error, "cudaMalloc failed.");
include/aa_permitted_pipelines_1.hpp:      size_t gpu_outputsize = 0;
include/aa_permitted_pipelines_1.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)max_ndms * sizeof(float);
include/aa_permitted_pipelines_1.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)nchans * sizeof(float);
include/aa_permitted_pipelines_1.hpp:      cuda_return = cudaMalloc((void **) d_output, gpu_outputsize);
include/aa_permitted_pipelines_1.hpp:      if(cuda_return != cudaSuccess) {
include/aa_permitted_pipelines_1.hpp:	LOG(log_level::error, "cudaMalloc failed.");
include/aa_permitted_pipelines_1.hpp:      cuda_return = cudaMemset(*d_output, 0, gpu_outputsize);
include/aa_permitted_pipelines_1.hpp:      if(cuda_return != cudaSuccess) {
include/aa_permitted_pipelines_1.hpp:	LOG(log_level::error, "cudaMemset failed.");
include/aa_permitted_pipelines_1.hpp:      //Allocate GPU memory
include/aa_permitted_pipelines_1.hpp:      allocate_memory_gpu(maxshift, max_ndms, nchans, t_processed, &d_input, &d_output);
include/aa_permitted_pipelines_1.hpp:	LOG(log_level::dev_debug, "(Performed Brute-Force Dedispersion:" + std::to_string(time) + "(GPU estimate)");
include/aa_permitted_pipelines_1.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_1.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_1.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_1.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_1.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_1.hpp:	printf("\nPerforming old GPU rfi...");
include/aa_permitted_pipelines_1.hpp:	rfi_gpu(d_input, nchans, t_processed[0][t]+maxshift);
include/aa_permitted_pipelines_1.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_1.hpp:	cudaDeviceSynchronize();
include/aa_permitted_pipelines_1.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_1.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_1.hpp:	  bin_gpu(d_input, d_output, nchans, t_processed[dm_range - 1][t] + maxshift * inBin[dm_range]);
include/aa_permitted_pipelines_1.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_1.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_gpu_timer.hpp:#ifndef ASTRO_ACCELERATE_AA_GPU_TIMER_HPP
include/aa_gpu_timer.hpp:#define ASTRO_ACCELERATE_AA_GPU_TIMER_HPP
include/aa_gpu_timer.hpp:#include <cuda_runtime.h>
include/aa_gpu_timer.hpp:   * \struct aa_gpu_timer
include/aa_gpu_timer.hpp:   * \brief Wrapper around CUDA events which implements a timer functionality.
include/aa_gpu_timer.hpp:  struct aa_gpu_timer {
include/aa_gpu_timer.hpp:    cudaEvent_t start;
include/aa_gpu_timer.hpp:    cudaEvent_t stop;
include/aa_gpu_timer.hpp:    aa_gpu_timer() {
include/aa_gpu_timer.hpp:      cudaEventCreate(&start);
include/aa_gpu_timer.hpp:      cudaEventCreate(&stop);
include/aa_gpu_timer.hpp:    ~aa_gpu_timer() {
include/aa_gpu_timer.hpp:      cudaEventDestroy(start);
include/aa_gpu_timer.hpp:      cudaEventDestroy(stop);
include/aa_gpu_timer.hpp:      cudaEventRecord(start, 0);
include/aa_gpu_timer.hpp:      cudaEventRecord(stop, 0);
include/aa_gpu_timer.hpp:      cudaEventSynchronize(stop);
include/aa_gpu_timer.hpp:      cudaEventElapsedTime(&elapsed, start, stop);
include/aa_gpu_timer.hpp:#endif  // ASTRO_ACCELERATE_AA_GPU_TIMER_HPP
include/aa_device_harmonic_summing_kernel.hpp:  /** \brief Kernel wrapper function for PHS_GPU_kernel kernel function. */
include/aa_device_harmonic_summing_kernel.hpp:  void call_kernel_simple_harmonic_sum_GPU_kernel(
include/aa_device_harmonic_summing_kernel.hpp:  void call_kernel_greedy_harmonic_sum_GPU_kernel(
include/aa_device_harmonic_summing_kernel.hpp:  void call_kernel_presto_plus_harmonic_sum_GPU_kernel(
include/aa_device_harmonic_summing_kernel.hpp:  void call_kernel_presto_harmonic_sum_GPU_kernel(
include/aa_device_SNR_limited_inplace_kernel.hpp:#include <cuda.h>
include/aa_device_SNR_limited_inplace_kernel.hpp:#include <cuda_runtime.h>
include/aa_device_SNR_limited_inplace_kernel.hpp:  /** \brief Kernel wrapper function for PD_ZC_GPU kernel function. */
include/aa_device_SNR_limited_inplace_kernel.hpp:  void call_kernel_PD_ZC_GPU(float *const d_input, float *const d_output, const int &maxTaps, const int &nTimesamples, const int &nLoops);
include/aa_device_SNR_limited_inplace_kernel.hpp:  /** \brief Kernel wrapper function for PD_GPUv1_const kernel function. */
include/aa_device_SNR_limited_inplace_kernel.hpp:  void call_kernel_PD_GPUv1_const(float *const d_input, float *const d_temp, unsigned char *const d_output_taps,
include/aa_device_SNR_limited_kernel.hpp:#include <cuda.h>
include/aa_device_SNR_limited_kernel.hpp:#include <cuda_runtime.h>
include/aa_device_SNR_limited_kernel.hpp:  /** \brief Kernel wrapper function for SNR_GPU_limited kernel function. */
include/aa_device_SNR_limited_kernel.hpp:  void call_kernel_SNR_GPU_limited(const dim3 &grid_size, const dim3 &block_size, float *const d_FIR_input, float *const d_SNR_output,
include/aa_device_MSD_shared_kernel_functions.hpp:  /** \brief Kernel wrapper function for MSD_GPU_final_regular kernel function. */
include/aa_device_MSD_shared_kernel_functions.hpp:  void call_kernel_MSD_GPU_final_regular(const dim3 &grid_size, const dim3 &block_size, float *const d_input, float *const d_output, const int &size);
include/aa_device_MSD_shared_kernel_functions.hpp:  /** \brief Kernel wrapper function for MSD_GPU_final_regular kernel function. */
include/aa_device_MSD_shared_kernel_functions.hpp:  void call_kernel_MSD_GPU_final_regular(const dim3 &grid_size, const dim3 &block_size, float *const d_input, float *const d_MSD, float *const d_pp, const int &size);
include/aa_device_MSD_shared_kernel_functions.hpp:  /** \brief Kernel wrapper function for MSD_GPU_final_nonregular  kernel function. */
include/aa_device_MSD_shared_kernel_functions.hpp:  void call_kernel_MSD_GPU_final_nonregular(const dim3 &grid_size, const dim3 &block_size, float *const d_input, float *const d_MSD, const int &size);
include/aa_device_MSD_shared_kernel_functions.hpp:  /** \brief Kernel wrapper function for MSD_GPU_final_nonregular kernel function. */
include/aa_device_MSD_shared_kernel_functions.hpp:  void call_kernel_MSD_GPU_final_nonregular(const dim3 &grid_size, const dim3 &block_size, float *const d_input, float *const d_MSD, float *const d_pp, const int &size);
include/aa_device_MSD_shared_kernel_functions.hpp:  /** \brief Kernel wrapper function for MSD_GPU_Interpolate_linear kernel function. */
include/aa_device_MSD_shared_kernel_functions.hpp:  void call_kernel_MSD_GPU_Interpolate_linear(const dim3 &grid_size, const dim3 &block_size, float *const d_MSD_DIT, float *const d_MSD_interpolated, int *const d_MSD_DIT_widths, const int &MSD_DIT_size, int *const boxcar, const int &max_width_performed);
include/aa_device_info.hpp:#include <cuda.h>
include/aa_device_info.hpp:#include <cuda_runtime.h>
include/aa_device_info.hpp: * \brief Obtain information about available GPUs and select the GPU to use for data processing.
include/aa_device_info.hpp:	 * \brief Struct to contain CUDA card information.
include/aa_device_info.hpp:		cudaDeviceReset();
include/aa_device_info.hpp:	/** \returns The currently available free memory on the currently selected GPU as reported by the CUDA driver. */
include/aa_device_info.hpp:			cudaMemGetInfo(&free, &total);
include/aa_device_info.hpp:	/** \brief Checks for GPUs on the machine.
include/aa_device_info.hpp:		cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
include/aa_device_info.hpp:		if (error_id != cudaSuccess) {
include/aa_device_info.hpp:			LOG(log_level::dev_debug, "cudaGetDeviceCount returned" + std::to_string(error_id) + "->" + cudaGetErrorString(error_id));
include/aa_device_info.hpp:		// This function call returns 0 if there are no CUDA capable devices.
include/aa_device_info.hpp:			LOG(log_level::notice, "There are no available device(s) that support CUDA");
include/aa_device_info.hpp:			LOG(log_level::notice, "Detected " + std::to_string(deviceCount) + " CUDA Capable device(s)");
include/aa_device_info.hpp:		cudaSetDevice(selected_device_id);
include/aa_device_info.hpp:		cudaDeviceProp deviceProp;
include/aa_device_info.hpp:		cudaGetDeviceProperties(&deviceProp, selected_device_id);
include/aa_device_info.hpp:		cudaDriverGetVersion(&driverVersion);
include/aa_device_info.hpp:		cudaRuntimeGetVersion(&runtimeVersion);
include/aa_device_info.hpp:		cudaMemGetInfo(&free, &total);
include/aa_device_info.hpp:		std::vector<int> compiled_cuda_sm_versions;
include/aa_device_info.hpp:		std::stringstream s(ASTRO_ACCELERATE_CUDA_SM_VERSION);
include/aa_device_info.hpp:			compiled_cuda_sm_versions.push_back(i);
include/aa_device_info.hpp:		for (auto compiled_code : compiled_cuda_sm_versions) {
include/aa_permitted_pipelines_generic.hpp:#define PIPELINE_ERROR_DDTR_GPU_MEMORY_FAIL 1
include/aa_permitted_pipelines_generic.hpp:#define PIPELINE_ERROR_SPDT_GPU_MEMORY_FAIL 2
include/aa_permitted_pipelines_generic.hpp:#define PIPELINE_ERROR_GPU_FREE_MEMORY_FAIL 4
include/aa_permitted_pipelines_generic.hpp:#define PIPELINE_ERROR_GENERAL_GPU_ERROR 5
include/aa_permitted_pipelines_generic.hpp:#include <cuda.h>
include/aa_permitted_pipelines_generic.hpp:#include <cuda_runtime.h>
include/aa_permitted_pipelines_generic.hpp:#include "aa_bin_gpu.hpp"
include/aa_permitted_pipelines_generic.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_generic.hpp:		aa_gpu_timer m_timer;
include/aa_permitted_pipelines_generic.hpp:		aa_gpu_timer m_local_timer;
include/aa_permitted_pipelines_generic.hpp:		aa_gpu_timer m_ddtr_total_timer;
include/aa_permitted_pipelines_generic.hpp:			cudaError_t e;
include/aa_permitted_pipelines_generic.hpp:				e = cudaFree(d_DDTR_input);
include/aa_permitted_pipelines_generic.hpp:				if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:					pipeline_error = PIPELINE_ERROR_GPU_FREE_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:					LOG(log_level::error, "Cannot free memory (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:				e = cudaFree(d_DDTR_output);
include/aa_permitted_pipelines_generic.hpp:				if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:					pipeline_error = PIPELINE_ERROR_GPU_FREE_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:					LOG(log_level::error, "Cannot free memory (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:				e =cudaFree(d_dm_shifts);
include/aa_permitted_pipelines_generic.hpp:				if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:					pipeline_error = PIPELINE_ERROR_GPU_FREE_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:					LOG(log_level::error, "Cannot free memory (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:				e =cudaFree(d_dm_shifts);
include/aa_permitted_pipelines_generic.hpp:				if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:					pipeline_error = PIPELINE_ERROR_GPU_FREE_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:					LOG(log_level::error, "Cannot free memory (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:			e = cudaFree(d_bandpass_normalization);
include/aa_permitted_pipelines_generic.hpp:			if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:				pipeline_error = PIPELINE_ERROR_GPU_FREE_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:				LOG(log_level::error, "Cannot free d_bandpass_normalization memory: (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:					e = cudaFree(m_d_MSD_workarea);
include/aa_permitted_pipelines_generic.hpp:					if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:						pipeline_error = PIPELINE_ERROR_GPU_FREE_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:						LOG(log_level::error, "Cannot free memory (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:					e = cudaFree(m_d_SPDT_output_taps);
include/aa_permitted_pipelines_generic.hpp:					if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:						pipeline_error = PIPELINE_ERROR_GPU_FREE_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:						LOG(log_level::error, "Cannot free memory (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:					e = cudaFree(m_d_MSD_interpolated);
include/aa_permitted_pipelines_generic.hpp:					if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:						pipeline_error = PIPELINE_ERROR_GPU_FREE_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:						LOG(log_level::error, "Cannot free memory (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:				cudaFree(d_DDTR_input);
include/aa_permitted_pipelines_generic.hpp:				cudaFree(d_DDTR_output);
include/aa_permitted_pipelines_generic.hpp:				if( (nchans > 8192) && (nbits != 4) ) cudaFree(d_dm_shifts);
include/aa_permitted_pipelines_generic.hpp:				if( (nbits == 4) && (nchans > 4096) ) cudaFree(d_dm_shifts);
include/aa_permitted_pipelines_generic.hpp:					cudaFree(m_d_MSD_workarea);
include/aa_permitted_pipelines_generic.hpp:					cudaFree(m_d_SPDT_output_taps);
include/aa_permitted_pipelines_generic.hpp:					cudaFree(m_d_MSD_interpolated);
include/aa_permitted_pipelines_generic.hpp:		/** \brief Allocate the GPU memory needed for dedispersion. */
include/aa_permitted_pipelines_generic.hpp:		bool allocate_gpu_memory_DDTR(){
include/aa_permitted_pipelines_generic.hpp:			cudaMemGetInfo(&free_memory,&total_memory);
include/aa_permitted_pipelines_generic.hpp:			size_t gpu_inputsize = (size_t)time_samps * (size_t)nchans * sizeof(unsigned short);
include/aa_permitted_pipelines_generic.hpp:			size_t gpu_outputsize = 0;
include/aa_permitted_pipelines_generic.hpp:				gpu_outputsize = (size_t)time_samps * (size_t)max_ndms * sizeof(float);
include/aa_permitted_pipelines_generic.hpp:				gpu_outputsize = (size_t)time_samps * (size_t)nchans * sizeof(float);
include/aa_permitted_pipelines_generic.hpp:			printf("DDTR input size: %zu bytes = %0.3f MB;\n", gpu_inputsize, ((double) gpu_inputsize)/(1024.0*1024.0));
include/aa_permitted_pipelines_generic.hpp:			printf("DDTR output size:  %zu bytes = %0.3f MB;\n", gpu_outputsize, ((double) gpu_outputsize)/(1024.0*1024.0));
include/aa_permitted_pipelines_generic.hpp:			required_memory = required_memory + gpu_outputsize + gpu_inputsize;
include/aa_permitted_pipelines_generic.hpp:			cudaError_t e;
include/aa_permitted_pipelines_generic.hpp:			e = cudaMalloc((void **)&d_DDTR_input, gpu_inputsize);
include/aa_permitted_pipelines_generic.hpp:			if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:				pipeline_error = PIPELINE_ERROR_DDTR_GPU_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:				LOG(log_level::error, "Could not allocate memory for d_DDTR_input using cudaMalloc in aa_permitted_pipelines_generic.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:			e = cudaMalloc((void **)&d_DDTR_output, gpu_outputsize);
include/aa_permitted_pipelines_generic.hpp:			if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:				pipeline_error = PIPELINE_ERROR_DDTR_GPU_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:				LOG(log_level::error, "Could not allocate memory for d_DDTR_output using cudaMalloc in aa_permitted_pipelines_generic.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:			cudaMemset(d_DDTR_output, 0, gpu_outputsize);
include/aa_permitted_pipelines_generic.hpp:				e = cudaMalloc((void **) &d_dm_shifts, nchans*sizeof(float));
include/aa_permitted_pipelines_generic.hpp:				if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:					pipeline_error = PIPELINE_ERROR_DDTR_GPU_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:					LOG(log_level::error, "Could not allocate memory for d_dm_shifts using cudaMalloc in aa_permitted_pipelines_generic.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:				e = cudaMalloc((void **) &d_dm_shifts, nchans*sizeof(float));
include/aa_permitted_pipelines_generic.hpp:				if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:					pipeline_error = PIPELINE_ERROR_DDTR_GPU_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:					LOG(log_level::error, "Could not allocate memory for d_dm_shifts (4-bit) using cudaMalloc in aa_permitted_pipelines_generic.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:			e = cudaMalloc((void **) &d_bandpass_normalization, nchans*sizeof(float));
include/aa_permitted_pipelines_generic.hpp:			if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:				pipeline_error = PIPELINE_ERROR_DDTR_GPU_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:				LOG(log_level::error, "Could not allocate memory for d_bandpass_normalization using cudaMalloc in aa_permitted_pipelines_generic.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:		void allocate_gpu_memory_SPD(float **const d_MSD_workarea, unsigned short **const d_SPDT_output_taps, float **const d_MSD_interpolated, const unsigned long int &MSD_maxtimesamples, const size_t &MSD_profile_size) {
include/aa_permitted_pipelines_generic.hpp:			cudaError_t e;
include/aa_permitted_pipelines_generic.hpp:			e = cudaMalloc((void **)d_MSD_workarea, MSD_maxtimesamples*5.5*sizeof(float));
include/aa_permitted_pipelines_generic.hpp:			if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:				pipeline_error = PIPELINE_ERROR_SPDT_GPU_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:				LOG(log_level::error, "Could not allocate memory for d_MSD_workarea using cudaMalloc in aa_permitted_pipelines_generic.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:			e = cudaMalloc((void **) &(*d_SPDT_output_taps), sizeof(ushort)*2*MSD_maxtimesamples);
include/aa_permitted_pipelines_generic.hpp:			if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:				pipeline_error = PIPELINE_ERROR_SPDT_GPU_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:				LOG(log_level::error, "Could not allocate memory for d_SPDT_output_taps using cudaMalloc in aa_permitted_pipelines_generic.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:			e = cudaMalloc((void **)d_MSD_interpolated, sizeof(float)*MSD_profile_size);
include/aa_permitted_pipelines_generic.hpp:			if (e != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:				pipeline_error = PIPELINE_ERROR_SPDT_GPU_MEMORY_FAIL;
include/aa_permitted_pipelines_generic.hpp:				LOG(log_level::error, "Could not allocate memory for d_MSD_interpolated cudaMalloc in aa_permitted_pipelines_generic.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_generic.hpp:			//Allocate GPU memory for dedispersion
include/aa_permitted_pipelines_generic.hpp:			//allocate_gpu_memory_DDTR(maxshift, max_ndms, nchans, t_processed, &d_DDTR_input, &d_DDTR_output, &d_dm_shifts);
include/aa_permitted_pipelines_generic.hpp:			allocate_gpu_memory_DDTR();
include/aa_permitted_pipelines_generic.hpp:				cudaMemcpy(d_bandpass_normalization, m_ddtr_strategy.bandpass_normalization_pointer(), nchans*sizeof(float), cudaMemcpyHostToDevice);
include/aa_permitted_pipelines_generic.hpp:			//Allocate GPU memory for SPD (i.e. analysis)
include/aa_permitted_pipelines_generic.hpp:				allocate_gpu_memory_SPD(&m_d_MSD_workarea, &m_d_SPDT_output_taps, &m_d_MSD_interpolated, m_analysis_strategy.MSD_data_info(), m_analysis_strategy.MSD_profile_size_in_bytes());
include/aa_permitted_pipelines_generic.hpp:			cudaError_t cuda_error;
include/aa_permitted_pipelines_generic.hpp:				cuda_error = cudaMemcpy(d_bandpass_normalization, h_new_bandpass, nchans*sizeof(float), cudaMemcpyHostToDevice);
include/aa_permitted_pipelines_generic.hpp:				if (cuda_error != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:			cudaError_t CUDA_error;
include/aa_permitted_pipelines_generic.hpp:			CUDA_error = cudaGetLastError();
include/aa_permitted_pipelines_generic.hpp:			if(CUDA_error != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:				pipeline_error = PIPELINE_ERROR_GENERAL_GPU_ERROR;
include/aa_permitted_pipelines_generic.hpp:				LOG(log_level::error, "GPU error at the pipeline start. (" + std::string(cudaGetErrorString(CUDA_error)) + ")");
include/aa_permitted_pipelines_generic.hpp:					printf("\n(Performed Brute-Force Dedispersion: %g (GPU estimate)", time);
include/aa_permitted_pipelines_generic.hpp:				CUDA_error = cudaGetLastError();
include/aa_permitted_pipelines_generic.hpp:				if(CUDA_error != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:					LOG(log_level::error, "GPU error at ZeroDM kernel. (" + std::string(cudaGetErrorString(CUDA_error)) + ")");
include/aa_permitted_pipelines_generic.hpp:				CUDA_error = cudaGetLastError();
include/aa_permitted_pipelines_generic.hpp:				if(CUDA_error != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:					LOG(log_level::error, "GPU error at ZeroDM kernel. (" + std::string(cudaGetErrorString(CUDA_error)) + ")");
include/aa_permitted_pipelines_generic.hpp:				CUDA_error = cudaGetLastError();
include/aa_permitted_pipelines_generic.hpp:				if(CUDA_error != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:					LOG(log_level::error, "GPU error at Corner turn. (" + std::string(cudaGetErrorString(CUDA_error)) + ")");
include/aa_permitted_pipelines_generic.hpp:					printf("\nPerforming old GPU rfi...");
include/aa_permitted_pipelines_generic.hpp:					rfi_gpu(d_DDTR_input, nchans, t_processed[0][current_time_chunk] + maxshift_original);
include/aa_permitted_pipelines_generic.hpp:					time_log.adding("DDTR", "RFI_GPU", m_local_timer.Elapsed());
include/aa_permitted_pipelines_generic.hpp:					CUDA_error = cudaGetLastError();
include/aa_permitted_pipelines_generic.hpp:					if(CUDA_error != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:						LOG(log_level::error, "GPU error at RFI. (" + std::string(cudaGetErrorString(CUDA_error)) + ")");
include/aa_permitted_pipelines_generic.hpp:				cudaDeviceSynchronize();
include/aa_permitted_pipelines_generic.hpp:				CUDA_error = cudaGetLastError();
include/aa_permitted_pipelines_generic.hpp:				if(CUDA_error != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:					LOG(log_level::error, "GPU error at Dedispersion. (" + std::string(cudaGetErrorString(CUDA_error)) + ")");
include/aa_permitted_pipelines_generic.hpp:					bin_gpu(d_DDTR_input, d_DDTR_output, nchans, t_processed[dm_range - 1][current_time_chunk] + maxshift * inBin[dm_range]);
include/aa_permitted_pipelines_generic.hpp:					CUDA_error = cudaGetLastError();
include/aa_permitted_pipelines_generic.hpp:					if(CUDA_error != cudaSuccess) {
include/aa_permitted_pipelines_generic.hpp:						LOG(log_level::error, "GPU error at Binning. (" + std::string(cudaGetErrorString(CUDA_error)) + ")");
include/aa_permitted_pipelines_generic.hpp:					//LOG(log_level::error, "GPU error at Dedispersion. (" + std::string(cudaGetErrorString(CUDA_error)) + ")");
include/aa_permitted_pipelines_generic.hpp:					LOG(log_level::error, "GPU error at Dedispersion.");
include/aa_permitted_pipelines_generic.hpp:					SPDT_no_error = analysis_GPU(
include/aa_permitted_pipelines_generic.hpp:			aa_gpu_timer timer;
include/aa_permitted_pipelines_generic.hpp:			GPU_periodicity(
include/aa_permitted_pipelines_generic.hpp:			// printf("\nPerformed Periodicity Location: %f (GPU estimate)", time);
include/aa_permitted_pipelines_generic.hpp:			// Assumption: GPU memory is free and available.
include/aa_permitted_pipelines_generic.hpp:			aa_gpu_timer timer;
include/aa_permitted_pipelines_generic.hpp:			printf("\nPerformed Acceleration Location: %lf (GPU estimate)", time);
include/aa_permitted_pipelines_generic.hpp:			aa_gpu_timer timer;
include/aa_host_statistics.hpp:  void statistics(char *string, int i, cudaStream_t stream, double *in_time, double *out_time, int maxshift, int total_ndms, int nchans, int nsamp, float tsamp, float *dm_low, float *dm_high, float *dm_step, int *ndms);
include/aa_device_jerk_search.hpp:  /** \brief Function that performs analysis component on the GPU. */  
include/aa_fdas_host.hpp:   * \struct fdas_gpuarrays
include/aa_fdas_host.hpp:  }fdas_gpuarrays;
include/aa_fdas_host.hpp:  void fdas_cuda_check_devices(int devid);
include/aa_fdas_host.hpp:  void fdas_alloc_gpu_arrays(fdas_gpuarrays *arrays,  cmd_args *cmdargs);
include/aa_fdas_host.hpp:  void fdas_free_gpu_arrays(fdas_gpuarrays *arrays,  cmd_args *cmdargs);
include/aa_fdas_host.hpp:  void fdas_cuda_create_fftplans( fdas_cufftplan *fftplans, fdas_params *params);
include/aa_fdas_host.hpp:  void fdas_cuda_basic(fdas_cufftplan *fftplans, fdas_gpuarrays *gpuarrays, cmd_args *cmdargs, fdas_params *params );
include/aa_fdas_host.hpp:  void fdas_cuda_customfft(fdas_cufftplan *fftplans, fdas_gpuarrays *gpuarrays, cmd_args *cmdargs, fdas_params *params );
include/aa_fdas_host.hpp:  void fdas_write_list(fdas_gpuarrays *gpuarrays, cmd_args *cmdargs, fdas_params *params, float *h_MSD, float dm_low, int dm_count, float dm_step, unsigned int list_size);
include/aa_fdas_host.hpp:  void fdas_write_ffdot(fdas_gpuarrays *gpuarrays, cmd_args *cmdargs, fdas_params *params, float dm_low, int dm_count, float dm_step );
include/aa_fdas_host.hpp:  void fdas_write_test_ffdot(fdas_gpuarrays *gpuarrays, cmd_args *cmdargs, fdas_params *params, float dm_low, int dm_count, float dm_step );
include/aa_fdas_device.hpp:#include <cuda_runtime.h>
include/aa_fdas_device.hpp:  /** \brief CUDA kernels. */
include/aa_fdas_device.hpp:  __global__ void cuda_overlap_copy(float2* d_ext_data, float2* d_cpx_signal, int sigblock,  int sig_rfftlen, int sig_tot_convlen, int kern_offset, int total_blocks);
include/aa_fdas_device.hpp:  void call_kernel_cuda_overlap_copy(float2 *const d_ext_data, float2 *const d_cpx_signal, const int &sigblock, const int &sig_rfftlen, const int &sig_tot_convlen, const int &kern_offset, const int &total_blocks);
include/aa_fdas_device.hpp:  __global__ void cuda_overlap_copy_smallblk(float2* d_ext_data, float2* d_cpx_signal, int sigblock,  int sig_rfftlen, int sig_tot_convlen, int kern_offset, int total_blocks);
include/aa_fdas_device.hpp:  /** \brief Kernel wrapper function for cuda_overlap_copy_smallbl kernel function. */
include/aa_fdas_device.hpp:  void call_kernel_cuda_overlap_copy_smallblk(const int &blocks, float2 *const d_ext_data, float2 *const d_cpx_signal, const int &sigblock, const int &sig_rfftlen, const int &sig_tot_convlen, const int &kern_offset, const int &total_blocks);
include/aa_fdas_device.hpp:  __global__ void cuda_convolve_reg_1d_halftemps(float2* d_kernel, float2* d_signal, float2* d_ffdot_plane, int sig_tot_convlen, float scale);
include/aa_fdas_device.hpp:  /** \brief Kernel wrapper function for cuda_convolve_reg_1d_halftemps kernel function. */
include/aa_fdas_device.hpp:  void call_kernel_cuda_convolve_reg_1d_halftemps(const int &blocks, const int &threads, float2 *const d_kernel, float2 *const d_signal, float2 *const d_ffdot_plane, const int &sig_tot_convlen, const float &scale);
include/aa_fdas_device.hpp:  __global__ void cuda_ffdotpow_concat_2d(float2* d_ffdot_plane_cpx, float* d_ffdot_plane, int sigblock, int kern_offset, int total_blocks,  int sig_tot_convlen, int sig_totlen);
include/aa_fdas_device.hpp:  /** \brief Kernel wrapper function for cuda_ffdotpow_concat_2d kernel function. */
include/aa_fdas_device.hpp:  void call_kernel_cuda_ffdotpow_concat_2d(const dim3 &blocks, const dim3 &threads, float2 *const d_ffdot_plane_cpx, float *const d_ffdot_plane, const int &sigblock, const int &kern_offset, const int &total_blocks, const int &sig_tot_convlen, const int &sig_totlen);
include/aa_fdas_device.hpp:  __global__ void cuda_ffdotpow_concat_2d_inbin(float2* d_ffdot_plane_cpx, float* d_ffdot_plane, int sigblock, int kern_offset, int total_blocks, int  sig_tot_convlen, int sig_totlen);
include/aa_fdas_device.hpp:  /** \brief Kernel wrapper function for cuda_ffdotpow_concat_2d_inbin kernel function. */
include/aa_fdas_device.hpp:  void call_kernel_cuda_ffdotpow_concat_2d_inbin(const dim3 &blocks, const dim3 &threads, float2 *const d_ffdot_plane_cpx, float *const d_ffdot_plane, const int &sigblock, const int &kern_offset, const int &total_blocks, const int &sig_tot_convlen, const int &sig_totlen);
include/aa_fdas_device.hpp:  __global__ void cuda_ffdotpow_concat_2d_ndm_inbin(float2* d_ffdot_plane_cpx, float* d_ffdot_plane, int kernlen, int siglen, int nkern, int kern_offset, int total_blocks, int sig_tot_convlen, int sig_totlen, int ndm);
include/aa_fdas_device.hpp:  __global__ void cuda_convolve_customfft_wes_no_reorder02(float2* d_kernel, float2* d_signal, float *d_ffdot_pw, int sigblock, int sig_tot_convlen, int sig_totlen, int offset, float scale);
include/aa_fdas_device.hpp:  /** \brief Kernel wrapper function for cuda_convolve_customfft_wes_no_reorder02 kernel function. */
include/aa_fdas_device.hpp:  void call_kernel_cuda_convolve_customfft_wes_no_reorder02(const int &blocks, float2 *const d_kernel, float2 *const d_signal, float *const d_ffdot_pw, const int &sigblock, const int &sig_tot_convlen, const int &sig_totlen, const int &offset, const float &scale);
include/aa_fdas_device.hpp:  __global__ void cuda_convolve_customfft_wes_no_reorder02_inbin(float2* d_kernel, float2* d_signal, float *d_ffdot_pw, int sigblock, int sig_tot_convlen, int sig_totlen, int offset, float scale, float2 *ip_edge_points);
include/aa_fdas_device.hpp:  /** \brief Kernel wrapper function for cuda_convolve_customfft_wes_no_reorder02_inbin kernel function. */
include/aa_fdas_device.hpp:  void call_kernel_cuda_convolve_customfft_wes_no_reorder02_inbin(const int &blocks, float2 *const d_kernel, float2 *const d_signal, float *const d_ffdot_pw, const int &sigblock, const int &sig_tot_convlen, const int &sig_totlen, const int &offset, const float &scale, float2 *const ip_edge_points);
include/aa_fdas_device.hpp:  __global__ void GPU_CONV_kFFT_mk11_2elem_2v(float2 const* __restrict__ d_input_signal, float *d_output_plane_reduced, float2 const* __restrict__ d_templates, int useful_part_size, int offset, int nConvolutions, float scale);
include/aa_fdas_device.hpp:  __global__ void GPU_CONV_kFFT_mk11_4elem_2v(float2 const* __restrict__ d_input_signal, float *d_output_plane_reduced, float2 const* __restrict__ d_templates, int useful_part_size, int offset, int nConvolutions, float scale);
include/aa_fdas_device.hpp:  /** \brief Kernel wrapper function for GPU_CONV_kFFT_mk11_4elem_2v kernel function. */
include/aa_fdas_device.hpp:  void call_kernel_GPU_CONV_kFFT_mk11_4elem_2v(const dim3 &grid_size, const dim3 &block_size, float2 const*const d_input_signal, float *const d_output_plane_reduced, float2 const*const d_templates, const int &useful_part_size, const int &offset, const int &nConvolutions, const float &scale);
include/aa_zero_dm.hpp:#include <cuda.h>
include/aa_zero_dm.hpp:#include <cuda_runtime.h>
include/aa_analysis_strategy.hpp:#include <cuda_runtime.h>
include/aa_permitted_pipelines_3.hpp:#include <cuda.h>
include/aa_permitted_pipelines_3.hpp:#include <cuda_runtime.h>
include/aa_permitted_pipelines_3.hpp:#include "aa_bin_gpu.hpp"
include/aa_permitted_pipelines_3.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_3.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_3.hpp:	cudaFree(d_input);
include/aa_permitted_pipelines_3.hpp:	cudaFree(d_output);
include/aa_permitted_pipelines_3.hpp:	cudaFree(m_d_MSD_workarea);
include/aa_permitted_pipelines_3.hpp:        cudaFree(m_d_MSD_output_taps);
include/aa_permitted_pipelines_3.hpp:	cudaFree(m_d_MSD_interpolated);
include/aa_permitted_pipelines_3.hpp:    aa_gpu_timer       m_timer;
include/aa_permitted_pipelines_3.hpp:    /** \brief Allocate the GPU memory needed for dedispersion. */
include/aa_permitted_pipelines_3.hpp:    void allocate_memory_gpu(const int &maxshift, const int &max_ndms, const int &nchans, int **const t_processed, unsigned short **const d_input, float **const d_output) {
include/aa_permitted_pipelines_3.hpp:      size_t gpu_inputsize = (size_t) time_samps * (size_t) nchans * sizeof(unsigned short);
include/aa_permitted_pipelines_3.hpp:      cudaError_t e = cudaMalloc((void **) d_input, gpu_inputsize);
include/aa_permitted_pipelines_3.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_3.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_3.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_3.hpp:      size_t gpu_outputsize = 0;
include/aa_permitted_pipelines_3.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)max_ndms * sizeof(float);
include/aa_permitted_pipelines_3.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)nchans * sizeof(float);
include/aa_permitted_pipelines_3.hpp:      e = cudaMalloc((void **) d_output, gpu_outputsize);
include/aa_permitted_pipelines_3.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_3.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_3.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_3.hpp:      cudaMemset(*d_output, 0, gpu_outputsize);
include/aa_permitted_pipelines_3.hpp:      cudaError_t e = cudaMalloc((void **) d_MSD_workarea,        MSD_maxtimesamples*5.5*sizeof(float));
include/aa_permitted_pipelines_3.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_3.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_3.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_3.hpp:      e = cudaMalloc((void **) &(*d_MSD_output_taps), sizeof(ushort)*2*MSD_maxtimesamples);
include/aa_permitted_pipelines_3.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_3.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_3.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_3.hpp:      e = cudaMalloc((void **) d_MSD_interpolated,    sizeof(float)*MSD_profile_size);
include/aa_permitted_pipelines_3.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_3.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_3.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_3.hpp:      //Allocate GPU memory
include/aa_permitted_pipelines_3.hpp:      //Allocate GPU memory for dedispersion
include/aa_permitted_pipelines_3.hpp:      allocate_memory_gpu(maxshift, max_ndms, nchans, t_processed, &d_input, &d_output);
include/aa_permitted_pipelines_3.hpp:      //Allocate GPU memory for SPS (i.e. analysis)
include/aa_permitted_pipelines_3.hpp:	  printf("\n(Performed Brute-Force Dedispersion: %g (GPU estimate)", time);
include/aa_permitted_pipelines_3.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3.hpp:	printf("\nPerforming old GPU rfi...");
include/aa_permitted_pipelines_3.hpp:	rfi_gpu(d_input, nchans, t_processed[0][t]+maxshift);
include/aa_permitted_pipelines_3.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3.hpp:	cudaDeviceSynchronize();
include/aa_permitted_pipelines_3.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3.hpp:	  bin_gpu(d_input, d_output, nchans, t_processed[dm_range - 1][t] + maxshift * inBin[dm_range]);
include/aa_permitted_pipelines_3.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3.hpp:	analysis_GPU(h_peak_list_DM,
include/aa_permitted_pipelines_3.hpp:      aa_gpu_timer timer;
include/aa_permitted_pipelines_3.hpp:      GPU_periodicity(
include/aa_permitted_pipelines_3.hpp:      printf("\nPerformed Periodicity Location: %f (GPU estimate)", time);
include/aa_device_dedispersion_kernel.hpp:	/** \brief Kernel wrapper function for dedispersion GPU kernel which works with number of channels greater then 8192. */
include/aa_device_MSD_shared_kernel_functions.cuh:#include <cuda.h>
include/aa_device_MSD_shared_kernel_functions.cuh:#include <cuda_runtime.h>
include/aa_device_MSD_shared_kernel_functions.cuh:#include "aa_device_cuda_deprecated_wrappers.cuh"
include/aa_device_MSD_Configuration.hpp:#include <cuda_runtime.h>
include/aa_device_power_kernel.hpp:  void call_kernel_power_kernel(const dim3 &block_size, const dim3 &grid_size, const int &smem_bytes, const cudaStream_t &stream,
include/aa_device_power_kernel.hpp:  /** \brief Kernel wrapper function for GPU_simple_power_and_interbin_kernel kernel function. */
include/aa_device_power_kernel.hpp:  void call_kernel_GPU_simple_power_and_interbin_kernel(const dim3 &grid_size, const dim3 &block_size,
include/aa_device_stats_kernel.hpp:  void call_kernel_stats_kernel(const dim3 &block_size, const dim3 &grid_size, const int &smem_bytes, const cudaStream_t &stream,
include/aa_fdas.hpp:#include <cuda.h>
include/aa_jerk_CandidateList.hpp:					if( cudaSuccess != cudaMemcpy(candidates,  candidates_input,  nCandidates*4*sizeof(float), cudaMemcpyDeviceToHost) ) {
include/aa_jerk_CandidateList.hpp:						printf("CUDA API failure!\n");
include/aa_jerk_CandidateList.hpp:		void AddSubListFromGPU(int t_nCandidates, float *d_candidate_input, float w, float DM, int nTimesaples_time_dom, int nFilters_z, float z_max_search_limit, float z_search_step, float sampling_time, int inBin){
include/aa_device_threshold_kernel.hpp:#include <cuda.h>
include/aa_device_threshold_kernel.hpp:#include <cuda_runtime.h>
include/aa_device_threshold_kernel.hpp:  /** Kernel wrapper function for THR_GPU_WARP kernel function. */
include/aa_device_threshold_kernel.hpp:  void call_kernel_THR_GPU_WARP(
include/aa_device_threshold_kernel.hpp:  /** Kernel wrapper function for GPU_Threshold_for_periodicity_normal_kernel kernel function. */
include/aa_device_threshold_kernel.hpp:  void call_kernel_GPU_Threshold_for_periodicity_normal_kernel(
include/aa_device_threshold_kernel.hpp:  /** Kernel wrapper function for GPU_Threshold_for_periodicity_kernel kernel function. */
include/aa_device_threshold_kernel.hpp:  void call_kernel_GPU_Threshold_for_periodicity_transposed_kernel(
include/aa_device_inference.hpp:  /** \todo Document the gpu_blocked_bootstrap function. */
include/aa_device_inference.hpp:  void gpu_blocked_bootstrap(float **d_idata, int dms_to_average, int num_els, int ndms, int num_bins, int num_boots, double *mean_boot_out, double *mean_data_out, double *sd_boot_out);
include/aa_device_rfi_kernel.hpp:  /** \brief Kernel wrapper function for call to rfi_gpu_kernel kernel function. */
include/aa_device_rfi_kernel.hpp:  void call_kernel_rfi_gpu_kernel(const dim3 &grid_size, const dim3 &block_size,
include/aa_device_binning_kernel.hpp:  /** \brief Kernel wrapper function for DiT_GPU_v2 kernel function. */
include/aa_device_binning_kernel.hpp:  void call_kernel_DiT_GPU_v2(const dim3 &gridSize, const dim3 &blockSize, float const *const d_input, float *const d_output, const unsigned int &nDMs, const unsigned int &nTimesamples, const unsigned int &dts);
include/aa_device_power.hpp:extern void power_gpu(cudaEvent_t event, cudaStream_t stream, int samps, int acc, cufftComplex *d_signal_fft, float *d_signal_power);
include/aa_device_power.hpp:extern void power_and_interbin_gpu(float2 *d_input_complex, float *d_output_power, float *d_output_interbinning, int nTimesamples, int nDMs);
include/aa_bin_gpu.hpp:#ifndef ASTRO_ACCELERATE_AA_BIN_GPU_HPP
include/aa_bin_gpu.hpp:#define ASTRO_ACCELERATE_AA_BIN_GPU_HPP
include/aa_bin_gpu.hpp:  void bin_gpu(unsigned short *const d_input, float *const d_output, const int nchans, const int nsamp);
include/aa_bin_gpu.hpp:  int GPU_DiT_v2_wrapper(float *d_input, float *d_output, int nDMs, int nTimesamples);
include/aa_bin_gpu.hpp:#endif // ASTRO_ACCELERATE_AA_BIN_GPU_HPP
include/aa_permitted_pipelines_2.hpp:#include <cuda.h>
include/aa_permitted_pipelines_2.hpp:#include <cuda_runtime.h>
include/aa_permitted_pipelines_2.hpp:#include "aa_bin_gpu.hpp"
include/aa_permitted_pipelines_2.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_2.hpp:	cudaFree(d_input);
include/aa_permitted_pipelines_2.hpp:	cudaFree(d_output);
include/aa_permitted_pipelines_2.hpp:    aa_gpu_timer       m_timer;
include/aa_permitted_pipelines_2.hpp:    /** \brief Allocate the GPU memory needed for dedispersion. */
include/aa_permitted_pipelines_2.hpp:    void allocate_memory_gpu(const int &maxshift, const int &max_ndms, const int &nchans, int **const t_processed, unsigned short **const d_input, float **const d_output) {
include/aa_permitted_pipelines_2.hpp:      size_t gpu_inputsize = (size_t) time_samps * (size_t) nchans * sizeof(unsigned short);
include/aa_permitted_pipelines_2.hpp:      cudaError_t e = cudaMalloc((void **) d_input, gpu_inputsize);
include/aa_permitted_pipelines_2.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_2.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_2.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_2.hpp:      size_t gpu_outputsize = 0;
include/aa_permitted_pipelines_2.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)max_ndms * sizeof(float);
include/aa_permitted_pipelines_2.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)nchans * sizeof(float);
include/aa_permitted_pipelines_2.hpp:      e = cudaMalloc((void **) d_output, gpu_outputsize);
include/aa_permitted_pipelines_2.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_2.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_2.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_2.hpp:      cudaMemset(*d_output, 0, gpu_outputsize);
include/aa_permitted_pipelines_2.hpp:      cudaError_t e = cudaMalloc((void **) d_MSD_workarea,        MSD_maxtimesamples*5.5*sizeof(float));
include/aa_permitted_pipelines_2.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_2.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_2.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_2.hpp:      e = cudaMalloc((void **) &(*d_MSD_output_taps), sizeof(ushort)*2*MSD_maxtimesamples);
include/aa_permitted_pipelines_2.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_2.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_2.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_2.hpp:      e = cudaMalloc((void **) d_MSD_interpolated,    sizeof(float)*MSD_profile_size);
include/aa_permitted_pipelines_2.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_2.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_2.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_2.hpp:      //Allocate GPU memory
include/aa_permitted_pipelines_2.hpp:      //Allocate GPU memory for dedispersion
include/aa_permitted_pipelines_2.hpp:      allocate_memory_gpu(maxshift, max_ndms, nchans, t_processed, &d_input, &d_output);
include/aa_permitted_pipelines_2.hpp:      //Allocate GPU memory for SPS (i.e. analysis)
include/aa_permitted_pipelines_2.hpp:        printf("\n(Performed Brute-Force Dedispersion: %g (GPU estimate)", time);
include/aa_permitted_pipelines_2.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_2.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_2.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_2.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_2.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_2.hpp:	printf("\nPerforming old GPU rfi...");
include/aa_permitted_pipelines_2.hpp:	rfi_gpu(d_input, nchans, t_processed[0][t]+maxshift);
include/aa_permitted_pipelines_2.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_2.hpp:	cudaDeviceSynchronize();
include/aa_permitted_pipelines_2.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_2.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_2.hpp:	  bin_gpu(d_input, d_output, nchans, t_processed[dm_range - 1][t] + maxshift * inBin[dm_range]);
include/aa_permitted_pipelines_2.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_2.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_2.hpp:	analysis_GPU(h_peak_list_DM,
include/aa_device_set_stretch.hpp:extern void set_stretch_gpu(cudaEvent_t event, cudaStream_t stream, int samps, float mean, float *d_input);
include/aa_device_stats.hpp:  extern void stats_gpu(cudaEvent_t event, cudaStream_t stream, int samps, float *mean, float *stddev, float *h_signal_power, float *d_signal_power);
include/aa_device_periods.hpp:extern void GPU_periodicity(aa_periodicity_strategy &PSR_strategy, float ***output_buffer, aa_periodicity_candidates &Power_Candidates, aa_periodicity_candidates &Interbin_Candidates);
include/aa_permitted_pipelines_4.hpp:#include <cuda.h>
include/aa_permitted_pipelines_4.hpp:#include <cuda_runtime.h>
include/aa_permitted_pipelines_4.hpp:#include "aa_bin_gpu.hpp"
include/aa_permitted_pipelines_4.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_4.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_4.hpp:	cudaFree(d_input);
include/aa_permitted_pipelines_4.hpp:	cudaFree(d_output);
include/aa_permitted_pipelines_4.hpp:	cudaFree(m_d_MSD_workarea);
include/aa_permitted_pipelines_4.hpp:	cudaFree(m_d_MSD_output_taps);
include/aa_permitted_pipelines_4.hpp:	cudaFree(m_d_MSD_interpolated);
include/aa_permitted_pipelines_4.hpp:    aa_gpu_timer       m_timer;
include/aa_permitted_pipelines_4.hpp:    /** \brief Allocate the GPU memory needed for dedispersion. */
include/aa_permitted_pipelines_4.hpp:    void allocate_memory_gpu(const int &maxshift, const int &max_ndms, const int &nchans, int **const t_processed, unsigned short **const d_input, float **const d_output) {
include/aa_permitted_pipelines_4.hpp:      size_t gpu_inputsize = (size_t) time_samps * (size_t) nchans * sizeof(unsigned short);
include/aa_permitted_pipelines_4.hpp:      cudaError_t e = cudaMalloc((void **) d_input, gpu_inputsize);
include/aa_permitted_pipelines_4.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_4.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_4.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_4.hpp:      size_t gpu_outputsize = 0;
include/aa_permitted_pipelines_4.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)max_ndms * sizeof(float);
include/aa_permitted_pipelines_4.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)nchans * sizeof(float);
include/aa_permitted_pipelines_4.hpp:      e = cudaMalloc((void **) d_output, gpu_outputsize);
include/aa_permitted_pipelines_4.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_4.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_4.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_4.hpp:      cudaMemset(*d_output, 0, gpu_outputsize);
include/aa_permitted_pipelines_4.hpp:      cudaError_t e = cudaMalloc((void **) d_MSD_workarea,        MSD_maxtimesamples*5.5*sizeof(float));
include/aa_permitted_pipelines_4.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_4.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_4.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_4.hpp:      e = cudaMalloc((void **) &(*d_MSD_output_taps), sizeof(ushort)*2*MSD_maxtimesamples);
include/aa_permitted_pipelines_4.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_4.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_4.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_4.hpp:      e = cudaMalloc((void **) d_MSD_interpolated,    sizeof(float)*MSD_profile_size);
include/aa_permitted_pipelines_4.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_4.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_4.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_4.hpp:      //Allocate GPU memory
include/aa_permitted_pipelines_4.hpp:      //Allocate GPU memory for dedispersion
include/aa_permitted_pipelines_4.hpp:      allocate_memory_gpu(maxshift, max_ndms, nchans, t_processed, &d_input, &d_output);
include/aa_permitted_pipelines_4.hpp:      //Allocate GPU memory for SPS (i.e. analysis)
include/aa_permitted_pipelines_4.hpp:	  printf("\n(Performed Brute-Force Dedispersion: %g (GPU estimate)", time);
include/aa_permitted_pipelines_4.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4.hpp:	printf("\nPerforming old GPU rfi...");
include/aa_permitted_pipelines_4.hpp:	rfi_gpu(d_input, nchans, t_processed[0][t]+maxshift);
include/aa_permitted_pipelines_4.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4.hpp:	cudaDeviceSynchronize();
include/aa_permitted_pipelines_4.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4.hpp:	  bin_gpu(d_input, d_output, nchans, t_processed[dm_range - 1][t] + maxshift * inBin[dm_range]);
include/aa_permitted_pipelines_4.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4.hpp:	analysis_GPU(h_peak_list_DM,
include/aa_permitted_pipelines_4.hpp:      // Assumption: GPU memory is free and available.
include/aa_permitted_pipelines_4.hpp:      aa_gpu_timer timer;
include/aa_permitted_pipelines_4.hpp:      printf("\nPerformed Acceleration Location: %lf (GPU estimate)", time);
include/aa_corner_turn.hpp:#include <cuda.h>
include/aa_corner_turn.hpp:#include <cuda_runtime.h>
include/aa_device_convolution_kernel.hpp:	void call_kernel_k_customFFT_GPU_forward(
include/aa_device_convolution_kernel.hpp:	void call_kernel_k_GPU_conv_OLS_via_customFFT(
include/aa_pipeline_api.hpp:	 * \details The pipeline strategy objects will request memory from the GPU that they will use when the pipeline is run.
include/aa_pipeline_api.hpp:	 * \warning Configuring multiple pipeline objects and strategy objects at the same time means the pipeline will not see the correct amount of memory on the GPU.
include/aa_pipeline_api.hpp:		aa_device_info                         m_selected_device; /** The user provided GPU card information for the aa_pipeline_api instance. */
include/aa_pipeline_api.hpp:				LOG(log_level::notice, "Otherwise, the new pipeline instance will not see the correct amount of GPU memory available.");
include/aa_pipeline_api.hpp:				cudaMemGetInfo(&free_mem,&total_mem);
include/aa_pipeline_api.hpp:				LOG(log_level::error, "Could not get data from DDTR. The data are not copied from GPU memory to host memory. Enable option copy_DDTR_data_to_host.");
include/aa_device_set_stretch_kernel.hpp:				      const int &smem_bytes, const cudaStream_t &stream,
include/aa_jerk_strategy.hpp:// conv_size should really be determined by the class based on size of the filter and performance of the GPU
include/aa_periodicity_strategy.hpp:		//#ifdef GPU_PERIODICITY_SEARCH_DEBUG
include/aa_periodicity_strategy.hpp:		// This whole unfortunate function is necessary because as of CUDA11 cufftMakePlan1d may return size of the cuFFT workarea larger (up to 2x) for some input values of nDMs (number of FFTs calculated). This value is larger then the value returned for some other larger value of nDMs thus we cannot rely on getting size of the cuFFT workarea just for largest number of FFT calculated.
include/aa_device_analysis.hpp:  /** \brief Function that performs analysis component on the GPU. */  
include/aa_device_analysis.hpp:  bool analysis_GPU(unsigned int *h_peak_list_DM, unsigned int *h_peak_list_TS, float *h_peak_list_SNR, unsigned int *h_peak_list_BW, size_t *peak_pos, size_t max_peak_size, int i, float tstart, int t_processed, int inBin, int *maxshift, int max_ndms, int const*const ndms, float cutoff, float sigma_constant, float max_boxcar_width_in_sec, float *output_buffer, float *dm_low, float *dm_high, float *dm_step, float tsamp, int candidate_algorithm, float *d_MSD_workarea, unsigned short *d_output_taps, float *d_MSD_interpolated, unsigned long int maxTimeSamples, int enable_msd_baselinenoise, const bool dump_to_disk, const bool dump_to_user, analysis_output &output);
include/aa_dedisperse.hpp:#include <cuda.h>
include/aa_dedisperse.hpp:#include <cuda_runtime.h>
include/aa_dedisperse.hpp:   * \brief Function that performs the dedispersion on the GPU.
include/aa_host_debug.hpp:  void debug(int test, clock_t start_time, int range, int *outBin, int enable_debug, int analysis, int output_dmt, int multi_file, float sigma_cutoff, float power, int max_ndms, float *user_dm_low, float *user_dm_high, float *user_dm_step, float *dm_low, float *dm_high, float *dm_step, int *ndms, int nchans, int nsamples, int nifs, int nbits, float tsamp, float tstart, float fch1, float foff, int maxshift, float max_dm, int nsamp, size_t gpu_inputsize, size_t gpu_outputsize, size_t inputsize, size_t outputsize);
include/aa_device_stretch.hpp:  extern void stretch_gpu(cudaEvent_t event, cudaStream_t stream, int acc, int samps, float tsamp, float *d_input, float *d_output);
include/aa_device_SPS_inplace_kernel.hpp:#include <cuda.h>
include/aa_device_SPS_inplace_kernel.hpp:#include <cuda_runtime.h>
include/aa_device_SPS_inplace_kernel.hpp:void call_kernel_PD_ZC_GPU_KERNEL(const dim3 &grid_size, const dim3 &block_size, float *const d_input, float *const d_output,
include/aa_device_SPS_inplace_kernel.hpp:void call_kernel_PD_INPLACE_GPU_KERNEL(const dim3 &grid_size, const dim3 &block_size, const int &SM_size, float *const d_input,
include/aa_host_stratagy.hpp:  void stratagy(int *maxshift, int *max_samps, int *num_tchunks, int *max_ndms, int *total_ndms, float *max_dm, float power, int nchans, int nsamp, float fch1, float foff, float tsamp, int range, float *user_dm_low, float *user_dm_high, float *user_dm_step, float **dm_low, float **dm_high, float **dm_step, int **ndms, float **dmshifts, int *inBin, int ***t_processed, size_t *gpu_memory, int enable_analysis);
include/aa_device_stretch_kernel.hpp:  void call_kernel_stretch_kernel(const dim3 &block_size, const dim3 &grid_size, const int &smem_bytes, const cudaStream_t &stream,
include/aa_device_load_data.hpp:	* Loads data from the host memory into the GPU memory.
include/aa_device_rfi.hpp:extern void rfi_gpu(unsigned short *d_input, int nchans, int nsamp);
include/aa_permitted_pipelines_3_0.hpp:#include <cuda.h>
include/aa_permitted_pipelines_3_0.hpp:#include <cuda_runtime.h>
include/aa_permitted_pipelines_3_0.hpp:#include "aa_bin_gpu.hpp"
include/aa_permitted_pipelines_3_0.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_3_0.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_3_0.hpp:	cudaFree(d_input);
include/aa_permitted_pipelines_3_0.hpp:	cudaFree(d_output);
include/aa_permitted_pipelines_3_0.hpp:    aa_gpu_timer       m_timer;
include/aa_permitted_pipelines_3_0.hpp:    /** \brief Allocate the GPU memory needed for dedispersion. */
include/aa_permitted_pipelines_3_0.hpp:    void allocate_memory_gpu(const int &maxshift, const int &max_ndms, const int &nchans, int **const t_processed, unsigned short **const d_input, float **const d_output) {
include/aa_permitted_pipelines_3_0.hpp:      size_t gpu_inputsize = (size_t) time_samps * (size_t) nchans * sizeof(unsigned short);
include/aa_permitted_pipelines_3_0.hpp:      cudaError_t e = cudaMalloc((void **) d_input, gpu_inputsize);
include/aa_permitted_pipelines_3_0.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_3_0.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_3_0.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_3_0.hpp:      size_t gpu_outputsize = 0;
include/aa_permitted_pipelines_3_0.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)max_ndms * sizeof(float);
include/aa_permitted_pipelines_3_0.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)nchans * sizeof(float);
include/aa_permitted_pipelines_3_0.hpp:      e = cudaMalloc((void **) d_output, gpu_outputsize);
include/aa_permitted_pipelines_3_0.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_3_0.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_3_0.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_3_0.hpp:      cudaMemset(*d_output, 0, gpu_outputsize);
include/aa_permitted_pipelines_3_0.hpp:      //Allocate GPU memory
include/aa_permitted_pipelines_3_0.hpp:      //Allocate GPU memory for dedispersion
include/aa_permitted_pipelines_3_0.hpp:      allocate_memory_gpu(maxshift, max_ndms, nchans, t_processed, &d_input, &d_output);
include/aa_permitted_pipelines_3_0.hpp:	  printf("\n(Performed Brute-Force Dedispersion: %g (GPU estimate)", time);
include/aa_permitted_pipelines_3_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3_0.hpp:	printf("\nPerforming old GPU rfi...");
include/aa_permitted_pipelines_3_0.hpp:	rfi_gpu(d_input, nchans, t_processed[0][t]+maxshift);
include/aa_permitted_pipelines_3_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3_0.hpp:	cudaDeviceSynchronize();
include/aa_permitted_pipelines_3_0.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3_0.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3_0.hpp:	  bin_gpu(d_input, d_output, nchans, t_processed[dm_range - 1][t] + maxshift * inBin[dm_range]);
include/aa_permitted_pipelines_3_0.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3_0.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_3_0.hpp:      aa_gpu_timer timer;
include/aa_permitted_pipelines_3_0.hpp:      GPU_periodicity(
include/aa_permitted_pipelines_3_0.hpp:      printf("\nPerformed Periodicity Location: %f (GPU estimate)", time);
include/aa_zero_dm_outliers.hpp:#include <cuda.h>
include/aa_zero_dm_outliers.hpp:#include <cuda_runtime.h>
include/aa_device_MSD_normal_kernel.hpp:  /** \brief Kernel wrapper function for MSD_GPU_limited kernel function. */
include/aa_device_MSD_normal_kernel.hpp:  void call_kernel_MSD_GPU_limited(const dim3 &grid_size, const dim3 &block_size, float const *const d_input, float *const d_output, const int &y_steps, const int &nTimesamples, const int &offset);
include/aa_device_peak_find_kernel.hpp:  void call_gpu_Filter_peaks(unsigned int *new_peak_list_DM, unsigned int *new_peak_list_TS, unsigned int *new_peak_list_BW, float *new_peak_list_SNR, unsigned int *d_peak_list_DM, unsigned int *d_peak_list_TS, unsigned int *d_peak_list_BW, float *d_peak_list_SNR, unsigned int nElements, unsigned int max_distance, int max_list_pos, int *gmem_pos);
include/aa_device_single_FIR.hpp:  extern int GPU_FIRv1_wrapper(float *d_input, float *d_output, int nTaps, unsigned int nDMs, unsigned int nTimesamples);
include/aa_device_cuda_deprecated_wrappers.cuh:#ifndef ASTRO_ACCELERATE_AA_DEVICE_CUDA_DEPRECATED_WRAPPERS_CUH
include/aa_device_cuda_deprecated_wrappers.cuh:#define ASTRO_ACCELERATE_AA_DEVICE_CUDA_DEPRECATED_WRAPPERS_CUH
include/aa_device_cuda_deprecated_wrappers.cuh:  /** \brief Wrapper function that implements CUDA shfl for old and new CUDA implementation. */
include/aa_device_cuda_deprecated_wrappers.cuh:#if(CUDART_VERSION >= 9000)
include/aa_device_cuda_deprecated_wrappers.cuh:  /** \brief Wrapper function that implements CUDA shfl_up for old and new CUDA implementation. */
include/aa_device_cuda_deprecated_wrappers.cuh:#if(CUDART_VERSION >= 9000)
include/aa_device_cuda_deprecated_wrappers.cuh:  /** \brief Wrapper function that implements CUDA shfl_down for old and new CUDA implementation. */
include/aa_device_cuda_deprecated_wrappers.cuh:#if(CUDART_VERSION >= 9000)
include/aa_device_cuda_deprecated_wrappers.cuh:  /** \brief Wrapper function that implements CUDA shfl_xor for old and new CUDA implementation. */
include/aa_device_cuda_deprecated_wrappers.cuh:#if(CUDART_VERSION >= 9000)
include/aa_device_cuda_deprecated_wrappers.cuh:  /** \brief Wrapper function that implements CUDA ballot for old and new CUDA implementation. */
include/aa_device_cuda_deprecated_wrappers.cuh:#if(CUDART_VERSION >= 9000)
include/aa_device_cuda_deprecated_wrappers.cuh:#endif //ASTRO_ACCELERATE_AA_DEVICE_CUDA_DEPRECATED_WRAPPERS_CUH
include/aa_permitted_pipelines_5_0.hpp:#include <cuda.h>
include/aa_permitted_pipelines_5_0.hpp:#include <cuda_runtime.h>
include/aa_permitted_pipelines_5_0.hpp:#include "aa_bin_gpu.hpp"
include/aa_permitted_pipelines_5_0.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_5_0.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_5_0.hpp:	cudaFree(d_input);
include/aa_permitted_pipelines_5_0.hpp:	cudaFree(d_output);
include/aa_permitted_pipelines_5_0.hpp:    aa_gpu_timer       m_timer;
include/aa_permitted_pipelines_5_0.hpp:    /** \brief Allocate the GPU memory needed for dedispersion. */
include/aa_permitted_pipelines_5_0.hpp:    void allocate_memory_gpu(const int &maxshift, const int &max_ndms, const int &nchans, int **const t_processed, unsigned short **const d_input, float **const d_output) {
include/aa_permitted_pipelines_5_0.hpp:      size_t gpu_inputsize = (size_t) time_samps * (size_t) nchans * sizeof(unsigned short);
include/aa_permitted_pipelines_5_0.hpp:      cudaError_t e = cudaMalloc((void **) d_input, gpu_inputsize);
include/aa_permitted_pipelines_5_0.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_5_0.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_5_0.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_5_0.hpp:      size_t gpu_outputsize = 0;
include/aa_permitted_pipelines_5_0.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)max_ndms * sizeof(float);
include/aa_permitted_pipelines_5_0.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)nchans * sizeof(float);
include/aa_permitted_pipelines_5_0.hpp:      e = cudaMalloc((void **) d_output, gpu_outputsize);
include/aa_permitted_pipelines_5_0.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_5_0.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_5_0.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_5_0.hpp:      cudaMemset(*d_output, 0, gpu_outputsize);
include/aa_permitted_pipelines_5_0.hpp:      //Allocate GPU memory
include/aa_permitted_pipelines_5_0.hpp:      //Allocate GPU memory for dedispersion
include/aa_permitted_pipelines_5_0.hpp:      allocate_memory_gpu(maxshift, max_ndms, nchans, t_processed, &d_input, &d_output);
include/aa_permitted_pipelines_5_0.hpp:	  printf("\n(Performed Brute-Force Dedispersion: %g (GPU estimate)", time);
include/aa_permitted_pipelines_5_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5_0.hpp:	printf("\nPerforming old GPU rfi...");
include/aa_permitted_pipelines_5_0.hpp:	rfi_gpu(d_input, nchans, t_processed[0][t]+maxshift);
include/aa_permitted_pipelines_5_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5_0.hpp:	cudaDeviceSynchronize();
include/aa_permitted_pipelines_5_0.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5_0.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5_0.hpp:	  bin_gpu(d_input, d_output, nchans, t_processed[dm_range - 1][t] + maxshift * inBin[dm_range]);
include/aa_permitted_pipelines_5_0.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5_0.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_5_0.hpp:      aa_gpu_timer timer;
include/aa_permitted_pipelines_5_0.hpp:      GPU_periodicity(
include/aa_permitted_pipelines_5_0.hpp:      printf("\nPerformed Periodicity Location: %f (GPU estimate)", time);
include/aa_permitted_pipelines_5_0.hpp:      // Assumption: GPU memory is free and available.
include/aa_permitted_pipelines_5_0.hpp:      aa_gpu_timer timer;
include/aa_permitted_pipelines_5_0.hpp:      printf("\nPerformed Acceleration Location: %lf (GPU estimate)", time);
include/aa_permitted_pipelines_4_0.hpp:#include <cuda.h>
include/aa_permitted_pipelines_4_0.hpp:#include <cuda_runtime.h>
include/aa_permitted_pipelines_4_0.hpp:#include "aa_bin_gpu.hpp"
include/aa_permitted_pipelines_4_0.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_4_0.hpp:#include "aa_gpu_timer.hpp"
include/aa_permitted_pipelines_4_0.hpp:	cudaFree(d_input);
include/aa_permitted_pipelines_4_0.hpp:	cudaFree(d_output);
include/aa_permitted_pipelines_4_0.hpp:    aa_gpu_timer       m_timer;
include/aa_permitted_pipelines_4_0.hpp:    /** \brief Allocate the GPU memory needed for dedispersion. */
include/aa_permitted_pipelines_4_0.hpp:    void allocate_memory_gpu(const int &maxshift, const int &max_ndms, const int &nchans, int **const t_processed, unsigned short **const d_input, float **const d_output) {
include/aa_permitted_pipelines_4_0.hpp:      size_t gpu_inputsize = (size_t) time_samps * (size_t) nchans * sizeof(unsigned short);
include/aa_permitted_pipelines_4_0.hpp:      cudaError_t e = cudaMalloc((void **) d_input, gpu_inputsize);
include/aa_permitted_pipelines_4_0.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_4_0.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_4_0.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_4_0.hpp:      size_t gpu_outputsize = 0;
include/aa_permitted_pipelines_4_0.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)max_ndms * sizeof(float);
include/aa_permitted_pipelines_4_0.hpp:	gpu_outputsize = (size_t)time_samps * (size_t)nchans * sizeof(float);
include/aa_permitted_pipelines_4_0.hpp:      e = cudaMalloc((void **) d_output, gpu_outputsize);
include/aa_permitted_pipelines_4_0.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_4_0.hpp:	LOG(log_level::error, "Could not allocate_memory_gpu cudaMalloc in aa_permitted_pipelines_4_0.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_4_0.hpp:      cudaMemset(*d_output, 0, gpu_outputsize);
include/aa_permitted_pipelines_4_0.hpp:      cudaError_t e = cudaMalloc((void **) d_MSD_workarea,        MSD_maxtimesamples*5.5*sizeof(float));
include/aa_permitted_pipelines_4_0.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_4_0.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_4_0.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_4_0.hpp:      e = cudaMalloc((void **) &(*d_MSD_output_taps), sizeof(ushort)*2*MSD_maxtimesamples);
include/aa_permitted_pipelines_4_0.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_4_0.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_4_0.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_4_0.hpp:      e = cudaMalloc((void **) d_MSD_interpolated,    sizeof(float)*MSD_profile_size);
include/aa_permitted_pipelines_4_0.hpp:      if(e != cudaSuccess) {
include/aa_permitted_pipelines_4_0.hpp:	LOG(log_level::error, "Could not allocate_memory_MSD cudaMalloc in aa_permitted_pipelines_4_0.hpp (" + std::string(cudaGetErrorString(e)) + ")");
include/aa_permitted_pipelines_4_0.hpp:      //Allocate GPU memory
include/aa_permitted_pipelines_4_0.hpp:      //Allocate GPU memory for dedispersion
include/aa_permitted_pipelines_4_0.hpp:      allocate_memory_gpu(maxshift, max_ndms, nchans, t_processed, &d_input, &d_output);
include/aa_permitted_pipelines_4_0.hpp:	  printf("\n(Performed Brute-Force Dedispersion: %g (GPU estimate)", time);
include/aa_permitted_pipelines_4_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4_0.hpp:	printf("\nPerforming old GPU rfi...");
include/aa_permitted_pipelines_4_0.hpp:	rfi_gpu(d_input, nchans, t_processed[0][t]+maxshift);
include/aa_permitted_pipelines_4_0.hpp:      //checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4_0.hpp:	cudaDeviceSynchronize();
include/aa_permitted_pipelines_4_0.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4_0.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4_0.hpp:	  bin_gpu(d_input, d_output, nchans, t_processed[dm_range - 1][t] + maxshift * inBin[dm_range]);
include/aa_permitted_pipelines_4_0.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4_0.hpp:	//checkCudaErrors(cudaGetLastError());
include/aa_permitted_pipelines_4_0.hpp:      // Assumption: GPU memory is free and available.
include/aa_permitted_pipelines_4_0.hpp:      aa_gpu_timer timer;
include/aa_permitted_pipelines_4_0.hpp:      printf("\nPerformed Acceleration Location: %lf (GPU estimate)", time);
include/meson_inclist.txt:aa_bin_gpu.hpp
include/meson_inclist.txt:aa_device_cuda_deprecated_wrappers.cuh
include/meson_inclist.txt:aa_gpu_timer.hpp
MANUAL.md:AstroAccelerate supported GPU status
MANUAL.md:AstroAccelerate supported CUDA&reg; SDK status
MANUAL.md:* The compiler for all device code is [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
MANUAL.md:For more information, please see the [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
MANUAL.md:Please see the [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for information about the operating system support for CUDA.
MANUAL.md:When a user first creates an `aa_pipeline_api` object, they must provide a `pipeline` which contain the components the user wishes to run. They must also provide the `pipeline_option` which contains the component options. In addition, the user must provide an `aa_filterbank_metadata` object, which contains the information about the time series data from a telescope (such as sampling rate, number of samples, and frequency binning information). Furthermore, a pointer to the raw input data must be provided (for a telescope `.fil` this can be obtained alongside the metadata by using the `aa_sigproc_input` class, but the user can also provide their own array data). Lastly, the user must configure a GPU card to run the pipeline on, using the `aa_device_info` class (which they can use to query the GPUs on the machine).
MANUAL.md:Once the `aa_pipeline_api` object is constructed, the user must still `bind` appropriate `plan` objects for each of the components they wish to run. There is one `plan` object for each `component`. For example, in order to run the `analysis` component, the user must `bind` an `aa_analysis_plan`. The `plan` contains the user's desired settings for how to run the component (such as the binning interval, and the dedispersion measure ranges to search for). In reality, user's settings may be sub-optimal from a performance perspective, so AstroAccelerate will optimise the user's `plan` by creating what is called a `strategy`. The `strategy` is created from the user's `plan`, and is the best compromise between the desired settings and good performance on the GPU. Strategy objects are required in order to run a pipeline, and the user is advised to verify and review the `strategy` when performing data analysis.
MANUAL.md:| dedispersion | Performs dedispersion of the input data.                         | aa_ddtr_plan / aa_ddtr_strategy               | **plan**: dedispersion measure low, high, step, inBin, outBin, enable_msd_baseline_noise. <br> **strategy**: aa_ddtr_plan, aa_filterbank_metadata, amount of gpu memory, flag to indicate if analysis will be used or not. | dedispersed time chunk data (`std::vector<unsigned short>`)                                                                       |
CMakeLists.txt:find_package(CUDA REQUIRED)
CMakeLists.txt:set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
CMakeLists.txt:set(CUDA_PROPAGATE_HOST_FLAGS OFF)
CMakeLists.txt:set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
CMakeLists.txt:list(APPEND CUDA_NVCC_FLAGS --use_fast_math)
CMakeLists.txt:list(APPEND CUDA_NVCC_FLAGS -g;)
CMakeLists.txt:list(APPEND CUDA_NVCC_FLAGS -Xptxas -O3 -std=c++11;)
CMakeLists.txt:list(APPEND CUDA_NVCC_FLAGS -lineinfo;)
CMakeLists.txt:list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-O3;)
CMakeLists.txt:list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fopenmp;)
CMakeLists.txt:list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-lm;)
CMakeLists.txt:list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wall;)
CMakeLists.txt:if(NOT DEFINED CUDA_ARCH)  
CMakeLists.txt:	set(CUDA_ARCH "ALL")
CMakeLists.txt:	message("-- INFO: Setting CUDA_ARCH to ALL.")
CMakeLists.txt:	message("-- INFO: The target CUDA architecture can be specified using:")
CMakeLists.txt:	message("-- INFO:   -DCUDA_ARCH=\"<arch>\"")
CMakeLists.txt:foreach(ARCH ${CUDA_ARCH})
CMakeLists.txt:		message("-- INFO: Building CUDA device code for Kepler,")
CMakeLists.txt:	   	list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_37,code=sm_37)
CMakeLists.txt:	   	list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
CMakeLists.txt:	   	list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
CMakeLists.txt:	   	list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
CMakeLists.txt:	   	list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
CMakeLists.txt:	   	list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75)
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_80)
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86)
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_90,code=sm_90)
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_ARCH_VERSION "35,37,50,52,60,61,62,70,75,80,86,90")
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_SM_VERSION "35,37,50,52,60,61,62,70,75,80,86,90")
CMakeLists.txt:		message("-- INFO: Building CUDA device code for architecture 3.5")
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_ARCH_VERSION "35")
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_SM_VERSION "35")
CMakeLists.txt:		message("-- INFO: Building CUDA device code for architecture 3.7")
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_37,code=sm_37)
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_ARCH_VERSION "37")
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_SM_VERSION "37")
CMakeLists.txt:		message("-- INFO: Building CUDA device code for architecture 5.0")
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_ARCH_VERSION "50")
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_SM_VERSION "50")
CMakeLists.txt:		message("-- INFO: Building CUDA device code for architecture 5.2")
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_ARCH_VERSION "52")
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_SM_VERSION "52")
CMakeLists.txt:		message("-- INFO: Building CUDA device code for architecture 6.0")
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_ARCH_VERSION "60")
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_SM_VERSION "60")
CMakeLists.txt:		message("-- INFO: Building CUDA device code for architecture 6.1")
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_ARCH_VERSION "61")
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_SM_VERSION "61")
CMakeLists.txt:		message("-- INFO: Building CUDA device code for architecture 6.2")
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_ARCH_VERSION "62")
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_SM_VERSION "62")
CMakeLists.txt:		message("-- INFO: Building CUDA device code for architecture 7.0")
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_ARCH_VERSION "70")
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_SM_VERSION "70")
CMakeLists.txt:		message("-- INFO: Building CUDA device code for architecture 7.5")
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75)
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_ARCH_VERSION "75")
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_SM_VERSION "75")
CMakeLists.txt:		message("-- INFO: Building CUDA device code for architecture 8.0")
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_80)
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_ARCH_VERSION "80")
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_SM_VERSION "80")
CMakeLists.txt:		message("-- INFO: Building CUDA device code for architecture 8.6")
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86)
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_ARCH_VERSION "86")
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_SM_VERSION "86")
CMakeLists.txt:		message("-- INFO: Building CUDA device code for architecture 9.0")
CMakeLists.txt:		list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_90,code=sm_90)
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_ARCH_VERSION "90")
CMakeLists.txt:		set(ASTRO_ACCELERATE_CUDA_SM_VERSION "90")
CMakeLists.txt:	message(FATAL_ERROR "-- CUDA_ARCH ${ARCH} not recognised or not defined")
CMakeLists.txt:message(STATUS "Using CUDA NVCC flags ${CUDA_NVCC_FLAGS}")
CMakeLists.txt:link_directories(${CUDA_LIBRARY_DIRS})
CMakeLists.txt:include_directories($ENV{CUDA_INSTALL_PATH}/include/)
CMakeLists.txt:include_directories($ENV{CUDA_INSTALL_PATH}/samples/common/inc/)
CMakeLists.txt:include_directories(${CUDA_LIBRARY_DIRS})
CMakeLists.txt:# CUDA library object
CMakeLists.txt:file(GLOB_RECURSE GPU_SOURCE_FILES "src/*.cu" "src/*.cpp")
CMakeLists.txt:cuda_add_library(astroaccelerate SHARED ${GPU_SOURCE_FILES})
CMakeLists.txt:target_link_libraries(astroaccelerate PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_curand_LIBRARY})
CMakeLists.txt:set_target_properties(astroaccelerate PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
CMakeLists.txt:# Standalone executable to link against CUDA library
setup.sh:# CUDA required PATH and LD_LIBRARY_PATH
setup.sh:AA_ADD_PATH=/usr/local/cuda/bin
setup.sh:AA_ADD_LD_LIBRARY_PATH=/usr/local/cuda/lib${AA_ARCHITECTURE}
setup.sh:	    echo "* NOTICE: Adding CUDA path to PATH environment variable."
setup.sh:	    echo "* NOTICE: PATH already contains CUDA path."
setup.sh:            echo "* Please check the CUDA version and path on your system."
setup.sh:            echo "* NOTICE: Adding CUDA library path to LD_LIBRARY_PATH environment variable."
setup.sh:            echo "* NOTICE: LD_LIBRARY_PATH already contains CUDA path."
setup.sh:            echo "* Please check the CUDA version and library path on your system."
setup.sh:export CUDA_INSTALL_PATH=/usr/local/cuda/
lib/cache_works/device_dedisperse.std:void dedisperse(int i, int t_processed, int *inBin, float *dmshifts, unsigned char *d_input, cudaTextureObject_t tex, float *d_output, int nchans, int nsamp, int maxshift, float *tsamp, float *dm_low, float *dm_high, float *dm_step, int *ndms) {
lib/cache_works/device_dedisperse.std:		//{{{ Dedisperse data on the GPU 
lib/cache_works/device_dedisperse.std:		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
lib/cache_works/device_dedisperse.std:		cudaFuncSetCacheConfig(shared_dedisperse_kernel, cudaFuncCachePreferShared);
lib/cache_works/device_dedisperse.std:	//cudaUnbindTexture(inTex);
lib/cache_works/device_dedisperse.std:	//cudaDestroyTextureObject(tex);
lib/cache_works/device_dedispersion_kernel.works:__global__ void shared_dedisperse_kernel(unsigned char *d_input, float *d_output, cudaTextureObject_t tex, float mstartdm, float mdmstep)
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/Makefile:# CUDA_HOME are supposed to be on default position
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/Makefile:SDK := /home/novotny/NVIDIA_GPU_Computing_SDK/CUDALibraries/common/inc/
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/Makefile:INC := -I/usr/local/cuda/include #-I$(SDK)
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/Makefile:LIB := -L/usr/local/cuda/lib64 -lcudart -lcufft -lfftw3f -lcuda
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/Makefile:cache-16bit.o: ../timer.h ../utils_cuda.h ../utils_file.h
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:#include "../utils_cuda.h"
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:void GPU_Polyphase(short2 *input, float2 *output, float *coeff, int nChannels, int nTaps, int nSpectra){
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	//---------> Initial nVidia stuff
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaGetDeviceCount(&devCount));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:			checkCudaErrors(cudaGetDeviceProperties(&devProp,i));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	GpuTimer timer;
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	//---------> Specific nVidia stuff
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	int nCUDAblocks,Sremainder,nRepeats,itemp;
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_input,  sizeof(short2)*input_size));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float2)*output_size));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_coeff,  sizeof(float)*coeff_size));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaMemcpy(d_coeff, coeff, coeff_size*sizeof(float), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	//---------> CUDA block and CUDA grid parameters
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	nCUDAblocks=(int) nColumns/SPECTRA_PER_BLOCK;
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	dim3 GridSize(nCUDAblocks, 1, 1);
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:		checkCudaErrors(cudaMemcpy(d_input, input, input_size*sizeof(short2), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:		checkCudaErrors(cudaMemcpy(d_input, &input[output_size*nRepeats], (Sremainder+nTaps-1)*nChannels*sizeof(short2), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:		nCUDAblocks=itemp/SPECTRA_PER_BLOCK;
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:		GridSize.x=nCUDAblocks;BlockSize.x=THREADS_PER_BLOCK;
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaGetLastError());
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaFree(d_input));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaFree(d_coeff));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:		checkCudaErrors(cudaMemcpy(output,d_output,output_size*sizeof(float2), cudaMemcpyDeviceToHost));
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:		checkCudaErrors(cudaMemcpy(&output[output_size*nRepeats],d_output,Sremainder*nChannels*sizeof(float2), cudaMemcpyDeviceToHost));	
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaGetLastError());
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	//checkCudaErrors(cudaDeviceSynchronize());
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:		sprintf(str,"GPU-polyphase.dat");
lib/AstroAccelerate/PPF/GPU/cache/16bit/L1-cont/cache-16bit.cu:	checkCudaErrors(cudaFree(d_output));
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:  if (err != cudaSuccess) {
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:void checkResultsExact(const T* const ref, const T* const gpu, size_t numElem) {
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:  //check that the GPU result matches the CPU result
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:    if (ref[i] != gpu[i]) {
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:                 "\nGPU      : " << +gpu[i] << std::endl;
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:void checkResultsEps(const T* const ref, const T* const gpu, size_t numElem, double eps1, double eps2) {
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:    T smaller = std::min(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:    T larger = std::max(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:        "\nGPU      : " << +gpu[i] << std::endl;
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:void checkResultsAutodesk(const T* const ref, const T* const gpu, size_t numElem, double variance, size_t tolerance)
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:    T smaller = std::min(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_cuda.h:    T larger = std::max(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/cache/16bit/polyphase.c:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/cache/16bit/polyphase.c:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/cache/16bit/polyphase.c:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/16bit/polyphase.c:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/cache/16bit/polyphase.c:void FIR_check(short2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error);
lib/AstroAccelerate/PPF/GPU/cache/16bit/polyphase.c:void FIR_FFT_check(short2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error);
lib/AstroAccelerate/PPF/GPU/cache/16bit/polyphase.c:void GPU_Polyphase(short2 *input, float2 *output, float *coeff, int nChannels, int nTaps, int nSpectra);
lib/AstroAccelerate/PPF/GPU/cache/16bit/polyphase.c:	GPU_Polyphase(h_input, h_output, h_coeff, nChannels, nTaps, nSpectra);
lib/AstroAccelerate/PPF/GPU/cache/16bit/polyphase.c:	cudaDeviceReset();
lib/AstroAccelerate/PPF/GPU/cache/16bit/reference.c:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/cache/16bit/reference.c:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/cache/16bit/reference.c:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/16bit/reference.c:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/cache/16bit/reference.c:void FIR_check(short2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error){
lib/AstroAccelerate/PPF/GPU/cache/16bit/reference.c:			etemp=abs(ftemp.x-spectra_GPU[bl*nChannels + c].x);
lib/AstroAccelerate/PPF/GPU/cache/16bit/reference.c:			etemp=abs(ftemp.y-spectra_GPU[bl*nChannels + c].y);
lib/AstroAccelerate/PPF/GPU/cache/16bit/reference.c:void FIR_FFT_check(short2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error){
lib/AstroAccelerate/PPF/GPU/cache/16bit/reference.c:			etemp=abs(spectra[c].x-spectra_GPU[bl*nChannels + c].x);
lib/AstroAccelerate/PPF/GPU/cache/16bit/reference.c:			etemp=abs(spectra[c].y-spectra_GPU[bl*nChannels + c].y);
lib/AstroAccelerate/PPF/GPU/cache/16bit/utils_file.h:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:#ifndef GPU_TIMER_H__
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:#define GPU_TIMER_H__
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:struct GpuTimer
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:  cudaEvent_t start;
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:  cudaEvent_t stop;
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:  GpuTimer()
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:    cudaEventCreate(&start);
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:    cudaEventCreate(&stop);
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:  ~GpuTimer()
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:    cudaEventDestroy(start);
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:    cudaEventDestroy(stop);
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:    cudaEventRecord(start, 0);
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:    cudaEventRecord(stop, 0);
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:    cudaEventSynchronize(stop);
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:    cudaEventElapsedTime(&elapsed, start, stop);
lib/AstroAccelerate/PPF/GPU/cache/16bit/timer.h:#endif  /* GPU_TIMER_H__ */
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/Makefile:# CUDA_HOME are supposed to be on default position
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/Makefile:SDK := /home/novotny/NVIDIA_GPU_Computing_SDK/CUDALibraries/common/inc/
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/Makefile:INC := -I/usr/local/cuda/include #-I$(SDK)
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/Makefile:LIB := -L/usr/local/cuda/lib64 -lcudart -lcufft -lfftw3f -lcuda
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/Makefile:cache-8bit.o: ../timer.h ../utils_cuda.h ../utils_file.h
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:#include "../utils_cuda.h"
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:void GPU_Polyphase(uchar4 *input, float2 *output, float *coeff, int nChannels, int nTaps, int nSpectra){
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	//---------> Initial nVidia stuff
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaGetDeviceCount(&devCount));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:			checkCudaErrors(cudaGetDeviceProperties(&devProp,i));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	GpuTimer timer;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	//---------> Specific nVidia stuff
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	int nColumns,nCUDAblocks,Sremainder,nRepeats,itemp;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_input,  sizeof(uchar4)*input_size));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float2)*output_size));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_coeff,  sizeof(float)*coeff_size));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaMemcpy(d_coeff, coeff, coeff_size*sizeof(float), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	//---------> CUDA block and CUDA grid parameters
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	nCUDAblocks=(int) nColumns/SPECTRA_PER_BLOCK;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	dim3 GridSize(nCUDAblocks, 1, 1);
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:		checkCudaErrors(cudaMemcpy(d_input, input, input_size*sizeof(uchar4), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:		checkCudaErrors(cudaMemcpy(d_input, input, (Sremainder+nTaps-1)*(nChannels/2)*sizeof(uchar4), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:		nCUDAblocks=itemp/SPECTRA_PER_BLOCK;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:		GridSize.x=nCUDAblocks;BlockSize.x=THREADS_PER_BLOCK;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaGetLastError());
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaFree(d_input));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaFree(d_coeff));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:		checkCudaErrors(cudaMemcpy(output,d_output,output_size*sizeof(float2), cudaMemcpyDeviceToHost));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:		checkCudaErrors(cudaMemcpy(&output[output_size*nRepeats],d_output,Sremainder*nChannels*sizeof(float2), cudaMemcpyDeviceToHost));	
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaGetLastError());
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	//checkCudaErrors(cudaDeviceSynchronize());
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:		sprintf(str,"GPU-polyphase.dat");
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-cont/cache-8bit.cu:	checkCudaErrors(cudaFree(d_output));
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:  if (err != cudaSuccess) {
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:void checkResultsExact(const T* const ref, const T* const gpu, size_t numElem) {
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:  //check that the GPU result matches the CPU result
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:    if (ref[i] != gpu[i]) {
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:                 "\nGPU      : " << +gpu[i] << std::endl;
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:void checkResultsEps(const T* const ref, const T* const gpu, size_t numElem, double eps1, double eps2) {
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:    T smaller = std::min(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:    T larger = std::max(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:        "\nGPU      : " << +gpu[i] << std::endl;
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:void checkResultsAutodesk(const T* const ref, const T* const gpu, size_t numElem, double variance, size_t tolerance)
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:    T smaller = std::min(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_cuda.h:    T larger = std::max(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/cache/8bit/polyphase.c:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/cache/8bit/polyphase.c:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/polyphase.c:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/polyphase.c:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/polyphase.c:void FIR_check(uchar2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error);
lib/AstroAccelerate/PPF/GPU/cache/8bit/polyphase.c:void FIR_FFT_check(uchar2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error);
lib/AstroAccelerate/PPF/GPU/cache/8bit/polyphase.c:void GPU_Polyphase(uchar4 *input, float2 *output, float *coeff, int nChannels, int nTaps, int nSpectra);
lib/AstroAccelerate/PPF/GPU/cache/8bit/polyphase.c:	GPU_Polyphase(h_input, h_output, h_coeff, nChannels, nTaps, nSpectra);
lib/AstroAccelerate/PPF/GPU/cache/8bit/polyphase.c:	cudaDeviceReset();
lib/AstroAccelerate/PPF/GPU/cache/8bit/reference.c:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/cache/8bit/reference.c:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/reference.c:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/reference.c:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/reference.c:void FIR_check(uchar2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error){
lib/AstroAccelerate/PPF/GPU/cache/8bit/reference.c:			etemp=abs(ftemp.x-spectra_GPU[bl*nChannels + c].x);
lib/AstroAccelerate/PPF/GPU/cache/8bit/reference.c:			etemp=abs(ftemp.y-spectra_GPU[bl*nChannels + c].y);
lib/AstroAccelerate/PPF/GPU/cache/8bit/reference.c:void FIR_FFT_check(uchar2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error){
lib/AstroAccelerate/PPF/GPU/cache/8bit/reference.c:			etemp=abs(spectra[c].x-spectra_GPU[bl*nChannels + c].x);
lib/AstroAccelerate/PPF/GPU/cache/8bit/reference.c:			etemp=abs(spectra[c].y-spectra_GPU[bl*nChannels + c].y);
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:#include "../utils_cuda.h"
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:void GPU_Polyphase(uchar4 *input, float2 *output, float *coeff, int nChannels, int nTaps, int nSpectra){
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	//---------> Initial nVidia stuff
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaGetDeviceCount(&devCount));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:			checkCudaErrors(cudaGetDeviceProperties(&devProp,i));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	GpuTimer timer;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	//---------> Specific nVidia stuff
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	int nColumns,nCUDAblocks,Sremainder,nRepeats,itemp;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaMalloc((void **) &d_input,  sizeof(uchar4)*input_size));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float2)*output_size));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaMalloc((void **) &d_coeff,  sizeof(float)*coeff_size));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaMemcpy(d_coeff, coeff, coeff_size*sizeof(float), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	//---------> CUDA block and CUDA grid parameters
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	nCUDAblocks=(int) nColumns/SPECTRA_PER_BLOCK;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	dim3 GridSize(nCUDAblocks, 1, 1);
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:		checkCudaErrors(cudaMemcpy(d_input, input, input_size*sizeof(uchar4), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:		checkCudaErrors(cudaMemcpy(d_input, input, (Sremainder+nTaps-1)*(nChannels/2)*sizeof(uchar4), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:		nCUDAblocks=itemp/SPECTRA_PER_BLOCK;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:		GridSize.x=nCUDAblocks;BlockSize.x=THREADS_PER_BLOCK;
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaGetLastError());
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaFree(d_input));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaFree(d_coeff));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:		checkCudaErrors(cudaMemcpy(output,d_output,output_size*sizeof(float2), cudaMemcpyDeviceToHost));
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:		checkCudaErrors(cudaMemcpy(&output[output_size*nRepeats],d_output,Sremainder*nChannels*sizeof(float2), cudaMemcpyDeviceToHost));	
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaGetLastError());
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	//checkCudaErrors(cudaDeviceSynchronize());
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:		sprintf(str,"GPU-polyphase.dat");
lib/AstroAccelerate/PPF/GPU/cache/8bit/L1-fermi-cont.cu:	checkCudaErrors(cudaFree(d_output));
lib/AstroAccelerate/PPF/GPU/cache/8bit/utils_file.h:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:#ifndef GPU_TIMER_H__
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:#define GPU_TIMER_H__
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:struct GpuTimer
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:  cudaEvent_t start;
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:  cudaEvent_t stop;
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:  GpuTimer()
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:    cudaEventCreate(&start);
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:    cudaEventCreate(&stop);
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:  ~GpuTimer()
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:    cudaEventDestroy(start);
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:    cudaEventDestroy(stop);
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:    cudaEventRecord(start, 0);
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:    cudaEventRecord(stop, 0);
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:    cudaEventSynchronize(stop);
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:    cudaEventElapsedTime(&elapsed, start, stop);
lib/AstroAccelerate/PPF/GPU/cache/8bit/timer.h:#endif  /* GPU_TIMER_H__ */
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:#include "../utils_cuda.h"
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:void GPU_Polyphase(float2 *input, float2 *output, float *coeff, int nChannels, int nTaps, int nSpectra){
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	//---------> Initial nVidia stuff
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaGetDeviceCount(&devCount));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:			checkCudaErrors(cudaGetDeviceProperties(&devProp,i));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	GpuTimer timer;
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	//---------> Specific nVidia stuff
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	int nColumns,nCUDAblocks,Sremainder,nRepeats,itemp;
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaMalloc((void **) &d_input,  sizeof(float2)*input_size));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float2)*output_size));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaMalloc((void **) &d_coeff,  sizeof(float)*coeff_size));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaMemcpy(d_coeff, coeff, coeff_size*sizeof(float), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	//---------> CUDA block and CUDA grid parameters
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	nCUDAblocks=(int) nColumns/SPECTRA_PER_BLOCK;
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	dim3 GridSize(nCUDAblocks, 1, 1);
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:		checkCudaErrors(cudaMemcpy(d_input, input, input_size*sizeof(float2), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:		checkCudaErrors(cudaMemcpy(d_input, &input[output_size*nRepeats], (Sremainder+nTaps-1)*nChannels*sizeof(float2), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:		nCUDAblocks=itemp/SPECTRA_PER_BLOCK;
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:		GridSize.x=nCUDAblocks;BlockSize.x=THREADS_PER_BLOCK;
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaGetLastError());
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaFree(d_input));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaFree(d_coeff));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:		checkCudaErrors(cudaMemcpy(output,d_output,output_size*sizeof(float2), cudaMemcpyDeviceToHost));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:		checkCudaErrors(cudaMemcpy(&output[output_size*nRepeats],d_output,Sremainder*nChannels*sizeof(float2), cudaMemcpyDeviceToHost));	
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaGetLastError());
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	//checkCudaErrors(cudaDeviceSynchronize());
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:		sprintf(str,"GPU-polyphase.dat");
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/L1-maxwell-cont.cu:	checkCudaErrors(cudaFree(d_output));
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/Makefile:# CUDA_HOME are supposed to be on default position
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/Makefile:SDK := /home/novotny/NVIDIA_GPU_Computing_SDK/CUDALibraries/common/inc/
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/Makefile:INC := -I/usr/local/cuda/include #-I$(SDK)
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/Makefile:LIB := -L/usr/local/cuda/lib64 -lcudart -lcufft -lfftw3f -lcuda
lib/AstroAccelerate/PPF/GPU/cache/32bit/L1-cont/Makefile:L1-maxwell-cont.o: ../timer.h ../utils_cuda.h ../utils_file.h
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:  if (err != cudaSuccess) {
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:void checkResultsExact(const T* const ref, const T* const gpu, size_t numElem) {
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:  //check that the GPU result matches the CPU result
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:    if (ref[i] != gpu[i]) {
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:                 "\nGPU      : " << +gpu[i] << std::endl;
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:void checkResultsEps(const T* const ref, const T* const gpu, size_t numElem, double eps1, double eps2) {
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:    T smaller = std::min(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:    T larger = std::max(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:        "\nGPU      : " << +gpu[i] << std::endl;
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:void checkResultsAutodesk(const T* const ref, const T* const gpu, size_t numElem, double variance, size_t tolerance)
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:    T smaller = std::min(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_cuda.h:    T larger = std::max(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/cache/32bit/polyphase.c:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/cache/32bit/polyphase.c:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/cache/32bit/polyphase.c:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/32bit/polyphase.c:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/cache/32bit/polyphase.c:void FIR_check(float2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error);
lib/AstroAccelerate/PPF/GPU/cache/32bit/polyphase.c:void FIR_FFT_check(float2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error);
lib/AstroAccelerate/PPF/GPU/cache/32bit/polyphase.c:void GPU_Polyphase(float2 *input, float2 *output, float *coeff, int nChannels, int nTaps, int nSpectra);
lib/AstroAccelerate/PPF/GPU/cache/32bit/polyphase.c:	GPU_Polyphase(h_input, h_output, h_coeff, nChannels, nTaps, nSpectra);
lib/AstroAccelerate/PPF/GPU/cache/32bit/polyphase.c:	cudaDeviceReset();
lib/AstroAccelerate/PPF/GPU/cache/32bit/reference.c:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/cache/32bit/reference.c:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/cache/32bit/reference.c:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/32bit/reference.c:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/cache/32bit/reference.c:void FIR_check(float2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error){
lib/AstroAccelerate/PPF/GPU/cache/32bit/reference.c:			etemp=abs(ftemp.x-spectra_GPU[bl*nChannels + c].x);
lib/AstroAccelerate/PPF/GPU/cache/32bit/reference.c:			etemp=abs(ftemp.y-spectra_GPU[bl*nChannels + c].y);
lib/AstroAccelerate/PPF/GPU/cache/32bit/reference.c:void FIR_FFT_check(float2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error){
lib/AstroAccelerate/PPF/GPU/cache/32bit/reference.c:			etemp=abs(spectra[c].x-spectra_GPU[bl*nChannels + c].x);
lib/AstroAccelerate/PPF/GPU/cache/32bit/reference.c:			etemp=abs(spectra[c].y-spectra_GPU[bl*nChannels + c].y);
lib/AstroAccelerate/PPF/GPU/cache/32bit/utils_file.h:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:#ifndef GPU_TIMER_H__
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:#define GPU_TIMER_H__
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:struct GpuTimer
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:  cudaEvent_t start;
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:  cudaEvent_t stop;
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:  GpuTimer()
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:    cudaEventCreate(&start);
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:    cudaEventCreate(&stop);
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:  ~GpuTimer()
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:    cudaEventDestroy(start);
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:    cudaEventDestroy(stop);
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:    cudaEventRecord(start, 0);
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:    cudaEventRecord(stop, 0);
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:    cudaEventSynchronize(stop);
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:    cudaEventElapsedTime(&elapsed, start, stop);
lib/AstroAccelerate/PPF/GPU/cache/32bit/timer.h:#endif  /* GPU_TIMER_H__ */
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:  if (err != cudaSuccess) {
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:void checkResultsExact(const T* const ref, const T* const gpu, size_t numElem) {
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:  //check that the GPU result matches the CPU result
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:    if (ref[i] != gpu[i]) {
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:                 "\nGPU      : " << +gpu[i] << std::endl;
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:void checkResultsEps(const T* const ref, const T* const gpu, size_t numElem, double eps1, double eps2) {
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:    T smaller = std::min(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:    T larger = std::max(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:        "\nGPU      : " << +gpu[i] << std::endl;
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:void checkResultsAutodesk(const T* const ref, const T* const gpu, size_t numElem, double variance, size_t tolerance)
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:    T smaller = std::min(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_cuda.h:    T larger = std::max(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/SM/16bit/polyphase.c:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/SM/16bit/polyphase.c:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/SM/16bit/polyphase.c:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/16bit/polyphase.c:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/SM/16bit/polyphase.c:void FIR_check(short2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error);
lib/AstroAccelerate/PPF/GPU/SM/16bit/polyphase.c:void FIR_FFT_check(short2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error);
lib/AstroAccelerate/PPF/GPU/SM/16bit/polyphase.c:void GPU_Polyphase(short2 *input, float2 *output, float *coeff, int nChannels, int nTaps, int nSpectra);
lib/AstroAccelerate/PPF/GPU/SM/16bit/polyphase.c:	GPU_Polyphase(h_input, h_output, h_coeff, nChannels, nTaps, nSpectra);
lib/AstroAccelerate/PPF/GPU/SM/16bit/polyphase.c:	cudaDeviceReset();
lib/AstroAccelerate/PPF/GPU/SM/16bit/reference.c:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/SM/16bit/reference.c:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/SM/16bit/reference.c:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/16bit/reference.c:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/SM/16bit/reference.c:void FIR_check(short2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error){
lib/AstroAccelerate/PPF/GPU/SM/16bit/reference.c:			etemp=abs(ftemp.x-spectra_GPU[bl*nChannels + c].x);
lib/AstroAccelerate/PPF/GPU/SM/16bit/reference.c:			etemp=abs(ftemp.y-spectra_GPU[bl*nChannels + c].y);
lib/AstroAccelerate/PPF/GPU/SM/16bit/reference.c:			if(etemp>0 && count<10) {printf("Spectra: %d; Channel: %d, should be %f is %f input data starts with %d\n",bl,c,ftemp.x, spectra_GPU[bl*nChannels + c].x,input_data[bl*nChannels + c].x);count++;}
lib/AstroAccelerate/PPF/GPU/SM/16bit/reference.c:void FIR_FFT_check(short2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error){
lib/AstroAccelerate/PPF/GPU/SM/16bit/reference.c:			etemp=abs(spectra[c].x-spectra_GPU[bl*nChannels + c].x);
lib/AstroAccelerate/PPF/GPU/SM/16bit/reference.c:			etemp=abs(spectra[c].y-spectra_GPU[bl*nChannels + c].y);
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/Makefile:# CUDA_HOME are supposed to be on default position
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/Makefile:SDK := /home/novotny/NVIDIA_GPU_Computing_SDK/CUDALibraries/common/inc/
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/Makefile:INC := -I/usr/local/cuda/include #-I$(SDK)
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/Makefile:LIB := -L/usr/local/cuda/lib64 -lcudart -lcufft -lfftw3f -lcuda
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/Makefile:SM-maxwell-16bit.o: ../timer.h ../utils_cuda.h ../utils_file.h
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:#include "../utils_cuda.h"
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:void Polyphase_GPU_init(){
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	//---------> Specific nVidia stuff
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:void Polyphase_GPU_benchmark(short2 *d_input, float2 *d_output, float *d_coeff, int nChannels, int nTaps, int nSpectra, double *fir_time, double *fft_time){
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	GpuTimer timer;
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	//---------> CUDA block and CUDA grid parameters
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	int nCUDAblocks_y=(int) (nSpectra/SM_Columns);
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	int nCUDAblocks_x=(int) (nChannels/WARP); //Head size
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);	//nCUDAblocks_y goes through spectra
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	dim3 blockSize(THREADS_PER_BLOCK, 1, 1); 		//nCUDAblocks_x goes through channels
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:void GPU_Polyphase(short2 *input, float2 *output, float *coeff, int nChannels, int nTaps, int nSpectra){
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	//---------> Initial nVidia stuff
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	checkCudaErrors(cudaGetDeviceCount(&devCount));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:			checkCudaErrors(cudaGetDeviceProperties(&devProp,i));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	GpuTimer timer; // if set before set device getting errors - invalid handle  
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	int nCUDAblocks_x=(int) nChannels/WARP; //Head size
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	int Cremainder=nChannels-nCUDAblocks_x*WARP; //Tail size
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_input,  sizeof(short2)*input_size));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float2)*output_size));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_coeff,  sizeof(float)*coeff_size));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	checkCudaErrors(cudaMemcpy(d_coeff, coeff, coeff_size*sizeof(float), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:		checkCudaErrors(cudaMemcpy(d_input, &input[r*output_size], input_size*sizeof(short2), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:		Polyphase_GPU_init();
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:		Polyphase_GPU_benchmark(d_input, d_output, d_coeff, nChannels, nTaps, maxColumns, &fir_time, &fft_time);
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:		checkCudaErrors(cudaMemcpy(&output[r*output_size], d_output, output_size*sizeof(float2), cudaMemcpyDeviceToHost));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:		checkCudaErrors(cudaMemcpy(d_input, &input[nRepeats*output_size], (Sremainder+TAPS-1)*nChannels*sizeof(short2), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:		Polyphase_GPU_init();
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:		Polyphase_GPU_benchmark(d_input, d_output, d_coeff, nChannels, nTaps, Spectra_to_run, &fir_time, &fft_time);
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:		checkCudaErrors(cudaMemcpy( &output[nRepeats*output_size], d_output, Sremainder*nChannels*sizeof(float2), cudaMemcpyDeviceToHost));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	checkCudaErrors(cudaGetLastError());
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	checkCudaErrors(cudaFree(d_input));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	checkCudaErrors(cudaFree(d_coeff));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:	checkCudaErrors(cudaFree(d_output));
lib/AstroAccelerate/PPF/GPU/SM/16bit/SM-16bit/SM-maxwell-16bit.cu:		sprintf(str,"GPU-polyphase.dat");
lib/AstroAccelerate/PPF/GPU/SM/16bit/utils_file.h:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:#ifndef GPU_TIMER_H__
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:#define GPU_TIMER_H__
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:struct GpuTimer
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:  cudaEvent_t start;
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:  cudaEvent_t stop;
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:  GpuTimer()
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:    cudaEventCreate(&start);
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:    cudaEventCreate(&stop);
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:  ~GpuTimer()
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:    cudaEventDestroy(start);
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:    cudaEventDestroy(stop);
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:    cudaEventRecord(start, 0);
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:    cudaEventRecord(stop, 0);
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:    cudaEventSynchronize(stop);
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:    cudaEventElapsedTime(&elapsed, start, stop);
lib/AstroAccelerate/PPF/GPU/SM/16bit/timer.h:#endif  /* GPU_TIMER_H__ */
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:  if (err != cudaSuccess) {
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:void checkResultsExact(const T* const ref, const T* const gpu, size_t numElem) {
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:  //check that the GPU result matches the CPU result
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:    if (ref[i] != gpu[i]) {
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:                 "\nGPU      : " << +gpu[i] << std::endl;
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:void checkResultsEps(const T* const ref, const T* const gpu, size_t numElem, double eps1, double eps2) {
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:    T smaller = std::min(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:    T larger = std::max(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:        "\nGPU      : " << +gpu[i] << std::endl;
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:void checkResultsAutodesk(const T* const ref, const T* const gpu, size_t numElem, double variance, size_t tolerance)
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:    T smaller = std::min(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_cuda.h:    T larger = std::max(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/SM/8bit/polyphase.c:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/SM/8bit/polyphase.c:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/SM/8bit/polyphase.c:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/8bit/polyphase.c:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/SM/8bit/polyphase.c:void FIR_check(uchar2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error);
lib/AstroAccelerate/PPF/GPU/SM/8bit/polyphase.c:void FIR_FFT_check(uchar2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error);
lib/AstroAccelerate/PPF/GPU/SM/8bit/polyphase.c:void GPU_Polyphase(uchar2 *input, float2 *output, float *coeff, int nChannels, int nTaps, int nSpectra);
lib/AstroAccelerate/PPF/GPU/SM/8bit/polyphase.c:	GPU_Polyphase(h_input, h_output, h_coeff, nChannels, nTaps, nSpectra);
lib/AstroAccelerate/PPF/GPU/SM/8bit/polyphase.c:	cudaDeviceReset();
lib/AstroAccelerate/PPF/GPU/SM/8bit/reference.c:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/SM/8bit/reference.c:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/SM/8bit/reference.c:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/8bit/reference.c:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/SM/8bit/reference.c:void FIR_check(uchar2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error){
lib/AstroAccelerate/PPF/GPU/SM/8bit/reference.c:			etemp=abs(ftemp.x-spectra_GPU[bl*nChannels + c].x);
lib/AstroAccelerate/PPF/GPU/SM/8bit/reference.c:			etemp=abs(ftemp.y-spectra_GPU[bl*nChannels + c].y);
lib/AstroAccelerate/PPF/GPU/SM/8bit/reference.c:			if(etemp>0 && count<10) {printf("Spectra: %d; Channel: %d, should be %f is %f input data starts with %d\n",bl,c,ftemp.x, spectra_GPU[bl*nChannels + c].x,input_data[bl*nChannels + c].x);count++;}
lib/AstroAccelerate/PPF/GPU/SM/8bit/reference.c:void FIR_FFT_check(uchar2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error){
lib/AstroAccelerate/PPF/GPU/SM/8bit/reference.c:			etemp=abs(spectra[c].x-spectra_GPU[bl*nChannels + c].x);
lib/AstroAccelerate/PPF/GPU/SM/8bit/reference.c:			etemp=abs(spectra[c].y-spectra_GPU[bl*nChannels + c].y);
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/Makefile:# CUDA_HOME are supposed to be on default position
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/Makefile:SDK := /home/novotny/NVIDIA_GPU_Computing_SDK/CUDALibraries/common/inc/
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/Makefile:INC := -I/usr/local/cuda/include #-I$(SDK)
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/Makefile:LIB := -L/usr/local/cuda/lib64 -lcudart -lcufft -lfftw3f -lcuda
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/Makefile:SM-maxwell-8bit.o: ../timer.h ../utils_cuda.h ../utils_file.h
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:#include "../utils_cuda.h"
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:void Polyphase_GPU_init(){
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	//---------> Specific nVidia stuff
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:void Polyphase_GPU_benchmark(uchar2 *d_input, float2 *d_output, float *d_coeff, int nChannels, int nTaps, int nSpectra, double *fir_time, double *fft_time){
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	GpuTimer timer;
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	//---------> CUDA block and CUDA grid parameters
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	int nCUDAblocks_y=(int) (nSpectra/SM_Columns);
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	int nCUDAblocks_x=(int) (nChannels/WARP); //Head size
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);	//nCUDAblocks_y goes through spectra
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	dim3 blockSize(THREADS_PER_BLOCK, 1, 1); 		//nCUDAblocks_x goes through channels
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:void GPU_Polyphase(uchar2 *input, float2 *output, float *coeff, const int nChannels, const int nTaps, const int nSpectra){
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	//---------> Initial nVidia stuff
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	checkCudaErrors(cudaGetDeviceCount(&devCount));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:			checkCudaErrors(cudaGetDeviceProperties(&devProp,i));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	GpuTimer timer; // if set before set device getting errors - invalid handle  
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	int nCUDAblocks_x=(int) nChannels/WARP; //Head size
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	int Cremainder=nChannels-nCUDAblocks_x*WARP; //Tail size
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_input,  sizeof(uchar2)*input_size));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float2)*output_size));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_coeff,  sizeof(float)*coeff_size));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	checkCudaErrors(cudaMemcpy(d_coeff, coeff, coeff_size*sizeof(float), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:		checkCudaErrors(cudaMemcpy(d_input, &input[r*output_size], input_size*sizeof(uchar2), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:		Polyphase_GPU_init();
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:		Polyphase_GPU_benchmark(d_input, d_output, d_coeff, nChannels, nTaps, maxColumns, &fir_time, &fft_time);
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:		checkCudaErrors(cudaMemcpy(&output[r*output_size], d_output, output_size*sizeof(float2), cudaMemcpyDeviceToHost));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:		checkCudaErrors(cudaMemcpy(d_input, &input[nRepeats*output_size], (Sremainder+TAPS-1)*nChannels*sizeof(uchar2), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:		Polyphase_GPU_init();
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:		Polyphase_GPU_benchmark(d_input, d_output, d_coeff, nChannels, nTaps, Spectra_to_run, &fir_time, &fft_time);
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:		checkCudaErrors(cudaMemcpy( &output[nRepeats*output_size], d_output, Sremainder*nChannels*sizeof(float2), cudaMemcpyDeviceToHost));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	checkCudaErrors(cudaGetLastError());
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	checkCudaErrors(cudaFree(d_input));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	checkCudaErrors(cudaFree(d_coeff));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:	checkCudaErrors(cudaFree(d_output));
lib/AstroAccelerate/PPF/GPU/SM/8bit/SM-8bit/SM-maxwell-8bit.cu:		sprintf(str,"GPU-polyphase.dat");
lib/AstroAccelerate/PPF/GPU/SM/8bit/utils_file.h:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:#ifndef GPU_TIMER_H__
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:#define GPU_TIMER_H__
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:struct GpuTimer
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:  cudaEvent_t start;
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:  cudaEvent_t stop;
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:  GpuTimer()
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:    cudaEventCreate(&start);
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:    cudaEventCreate(&stop);
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:  ~GpuTimer()
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:    cudaEventDestroy(start);
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:    cudaEventDestroy(stop);
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:    cudaEventRecord(start, 0);
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:    cudaEventRecord(stop, 0);
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:    cudaEventSynchronize(stop);
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:    cudaEventElapsedTime(&elapsed, start, stop);
lib/AstroAccelerate/PPF/GPU/SM/8bit/timer.h:#endif  /* GPU_TIMER_H__ */
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/Makefile:# CUDA_HOME are supposed to be on default position
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/Makefile:SDK := /home/novotny/NVIDIA_GPU_Computing_SDK/CUDALibraries/common/inc/
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/Makefile:INC := -I/usr/local/cuda/include #-I$(SDK)
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/Makefile:LIB := -L/usr/local/cuda/lib64 -lcudart -lcufft -lfftw3f -lcuda
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/Makefile:SM-maxwell-32bit.o: ../timer.h ../utils_cuda.h ../utils_file.h
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:#include "../utils_cuda.h"
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:void Polyphase_GPU_init(){
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	//---------> Specific nVidia stuff
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:void Polyphase_GPU_benchmark(float2 *d_input, float2 *d_output, float *d_coeff, int nChannels, int nTaps, int nSpectra, double *fir_time, double *fft_time){
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	GpuTimer timer;
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	//---------> CUDA block and CUDA grid parameters
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	int nCUDAblocks_y=(int) (nSpectra/SM_Columns);
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	int nCUDAblocks_x=(int) (nChannels/WARP); //Head size
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);	//nCUDAblocks_y goes through spectra
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	dim3 blockSize(THREADS_PER_BLOCK, 1, 1); 		//nCUDAblocks_x goes through channels
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:void GPU_Polyphase(float2 *input, float2 *output, float *coeff, int nChannels, int nTaps, int nSpectra){
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	//---------> Initial nVidia stuff
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	cudaDeviceProp devProp;
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	checkCudaErrors(cudaGetDeviceCount(&devCount));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:			checkCudaErrors(cudaGetDeviceProperties(&devProp,i));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	checkCudaErrors(cudaSetDevice(device));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	checkCudaErrors(cudaGetDeviceProperties(&devProp,device));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	cudaMemGetInfo(&free_mem,&total_mem);
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	GpuTimer timer; // if set before set device getting errors - invalid handle  
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	int nCUDAblocks_x=(int) nChannels/WARP; //Head size
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	int Cremainder=nChannels-nCUDAblocks_x*WARP; //Tail size
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_input,  sizeof(float2)*input_size));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float2)*output_size));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	checkCudaErrors(cudaMalloc((void **) &d_coeff,  sizeof(float)*coeff_size));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	checkCudaErrors(cudaMemcpy(d_coeff, coeff, coeff_size*sizeof(float), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:		checkCudaErrors(cudaMemcpy(d_input, &input[r*output_size], input_size*sizeof(float2), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:		Polyphase_GPU_init();
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:		Polyphase_GPU_benchmark(d_input, d_output, d_coeff, nChannels, nTaps, maxColumns, &fir_time, &fft_time);
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:		checkCudaErrors(cudaMemcpy(&output[r*output_size], d_output, output_size*sizeof(float2), cudaMemcpyDeviceToHost));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:		checkCudaErrors(cudaMemcpy(d_input, &input[nRepeats*output_size], (Sremainder+TAPS-1)*nChannels*sizeof(float2), cudaMemcpyHostToDevice));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:		Polyphase_GPU_init();
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:		Polyphase_GPU_benchmark(d_input, d_output, d_coeff, nChannels, nTaps, Spectra_to_run, &fir_time, &fft_time);
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:		checkCudaErrors(cudaMemcpy( &output[nRepeats*output_size], d_output, Sremainder*nChannels*sizeof(float2), cudaMemcpyDeviceToHost));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	checkCudaErrors(cudaGetLastError());
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	checkCudaErrors(cudaFree(d_input));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	checkCudaErrors(cudaFree(d_coeff));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:	checkCudaErrors(cudaFree(d_output));
lib/AstroAccelerate/PPF/GPU/SM/32bit/SM-32bit/SM-maxwell-32bit.cu:		sprintf(str,"GPU-polyphase.dat");
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:  if (err != cudaSuccess) {
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:void checkResultsExact(const T* const ref, const T* const gpu, size_t numElem) {
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:  //check that the GPU result matches the CPU result
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:    if (ref[i] != gpu[i]) {
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:                 "\nGPU      : " << +gpu[i] << std::endl;
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:void checkResultsEps(const T* const ref, const T* const gpu, size_t numElem, double eps1, double eps2) {
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:    T smaller = std::min(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:    T larger = std::max(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:        "\nGPU      : " << +gpu[i] << std::endl;
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:void checkResultsAutodesk(const T* const ref, const T* const gpu, size_t numElem, double variance, size_t tolerance)
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:    T smaller = std::min(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_cuda.h:    T larger = std::max(ref[i], gpu[i]);
lib/AstroAccelerate/PPF/GPU/SM/32bit/polyphase.c:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/SM/32bit/polyphase.c:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/SM/32bit/polyphase.c:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/32bit/polyphase.c:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/SM/32bit/polyphase.c:void FIR_check(float2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error);
lib/AstroAccelerate/PPF/GPU/SM/32bit/polyphase.c:void FIR_FFT_check(float2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error);
lib/AstroAccelerate/PPF/GPU/SM/32bit/polyphase.c:void GPU_Polyphase(float2 *input, float2 *output, float *coeff, int nChannels, int nTaps, int nSpectra);
lib/AstroAccelerate/PPF/GPU/SM/32bit/polyphase.c:	GPU_Polyphase(h_input, h_output, h_coeff, nChannels, nTaps, nSpectra);
lib/AstroAccelerate/PPF/GPU/SM/32bit/polyphase.c:	cudaDeviceReset();
lib/AstroAccelerate/PPF/GPU/SM/32bit/reference.c:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/SM/32bit/reference.c:#include <cuda.h>
lib/AstroAccelerate/PPF/GPU/SM/32bit/reference.c:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/32bit/reference.c:#include <cuda_runtime_api.h>
lib/AstroAccelerate/PPF/GPU/SM/32bit/reference.c:void FIR_check(float2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error){
lib/AstroAccelerate/PPF/GPU/SM/32bit/reference.c:			etemp=abs(ftemp.x-spectra_GPU[bl*nChannels + c].x);
lib/AstroAccelerate/PPF/GPU/SM/32bit/reference.c:			etemp=abs(ftemp.y-spectra_GPU[bl*nChannels + c].y);
lib/AstroAccelerate/PPF/GPU/SM/32bit/reference.c:			if(etemp>0 && count<10) {printf("Spectra: %d; Channel: %d, should be %f is %f input data starts with %d\n",bl,c,ftemp.x, spectra_GPU[bl*nChannels + c].x,input_data[bl*nChannels + c].x);count++;}
lib/AstroAccelerate/PPF/GPU/SM/32bit/reference.c:void FIR_FFT_check(float2 *input_data, float2 *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra, double *cumulative_error, double *mean_error){
lib/AstroAccelerate/PPF/GPU/SM/32bit/reference.c:			etemp=abs(spectra[c].x-spectra_GPU[bl*nChannels + c].x);
lib/AstroAccelerate/PPF/GPU/SM/32bit/reference.c:			etemp=abs(spectra[c].y-spectra_GPU[bl*nChannels + c].y);
lib/AstroAccelerate/PPF/GPU/SM/32bit/utils_file.h:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:#ifndef GPU_TIMER_H__
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:#define GPU_TIMER_H__
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:#include <cuda_runtime.h>
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:struct GpuTimer
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:  cudaEvent_t start;
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:  cudaEvent_t stop;
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:  GpuTimer()
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:    cudaEventCreate(&start);
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:    cudaEventCreate(&stop);
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:  ~GpuTimer()
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:    cudaEventDestroy(start);
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:    cudaEventDestroy(stop);
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:    cudaEventRecord(start, 0);
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:    cudaEventRecord(stop, 0);
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:    cudaEventSynchronize(stop);
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:    cudaEventElapsedTime(&elapsed, start, stop);
lib/AstroAccelerate/PPF/GPU/SM/32bit/timer.h:#endif  /* GPU_TIMER_H__ */
lib/AstroAccelerate/PPF/Xeon Phi/PHI_PPF.cpp:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/Xeon Phi/PHI_PPF.cpp:double FIR_check_uni(float *input_data, float *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra){
lib/AstroAccelerate/PPF/Xeon Phi/PHI_PPF.cpp:			etemp=abs(ftemp-spectra_GPU[bl*nChannels + c]);
lib/AstroAccelerate/PPF/CPU/CPU_PPF.cpp:    This is GPU implementation of a polyphase filter. 
lib/AstroAccelerate/PPF/CPU/CPU_PPF.cpp:double FIR_check_uni(float *input_data, float *spectra_GPU, float *coeff, int nTaps, int nChannels, int nSpectra){
lib/AstroAccelerate/PPF/CPU/CPU_PPF.cpp:			etemp=(ftemp-spectra_GPU[bl*nChannels + c])*(ftemp-spectra_GPU[bl*nChannels + c]);
lib/smem_works/device_dedisperse.cu:void dedisperse(int i, int t_processed, int *inBin, float *dmshifts, unsigned char *d_input, cudaTextureObject_t tex, float *d_output, int nchans, int nsamp, int maxshift, float *tsamp, float *dm_low, float *dm_high, float *dm_step, int *ndms) {
lib/smem_works/device_dedisperse.cu:		//{{{ Dedisperse data on the GPU 
lib/smem_works/device_dedisperse.cu:		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
lib/smem_works/device_dedisperse.cu:		cudaFuncSetCacheConfig(shared_dedisperse_kernel, cudaFuncCachePreferShared);
lib/smem_works/device_dedisperse.cu:	//cudaUnbindTexture(inTex);
lib/smem_works/device_dedisperse.cu:	//cudaDestroyTextureObject(tex);
lib/smem_works/device_dedispersion_kernel.cu:__global__ void shared_dedisperse_kernel(unsigned char *d_input, float *d_output, cudaTextureObject_t tex, float mstartdm, float mdmstep)
meson.build:project('astroaccelerate', 'cpp', 'cuda', 
meson.build:add_project_arguments(comp_flags, language : ['c', 'cpp', 'cuda'])
meson.build:# Dependencies: CUDA
meson.build:cuda_dep = dependency('cuda', version : '>=11', modules : ['cudart', 'nvrtc', 'cufft', 'curand'])
meson.build:                     dependencies: [cuda_dep, openmp_dep],
meson.build:                dependencies: [cuda_dep, openmp_dep],
cmake/version.h.in:#define ASTRO_ACCELERATE_CUDA_MAJOR_VERSION @CUDA_VERSION_MAJOR@
cmake/version.h.in:#define ASTRO_ACCELERATE_CUDA_MINOR_VERSION @CUDA_VERSION_MINOR@
cmake/version.h.in:#define ASTRO_ACCELERATE_CUDA_VERSION "@CUDA_VERSION_STRING@"
cmake/version.h.in:#define ASTRO_ACCELERATE_CUDA_ARCH_VERSION "@ASTRO_ACCELERATE_CUDA_ARCH_VERSION@"
cmake/version.h.in:#define ASTRO_ACCELERATE_CUDA_SM_VERSION "@ASTRO_ACCELERATE_CUDA_SM_VERSION@"
examples/meson.build:                dependencies: [cuda_dep, openmp_dep],
examples/src/fake_signal_periodic.cpp:  // Init the GPU card
examples/src/fake_signal_periodic.cpp:  const size_t free_memory = selected_device.free_memory(); // Free memory on the GPU in bytes
examples/src/fake_signal_periodic.cpp:  cudaMemGetInfo(&free, &total);
examples/src/dedispersion.cpp: * Compile with: g++ -std=c++11 -I/path/to/astro-accelerate/include/ -L/path/to/astro-accelerate/build -Wl,-rpath,/path/to/astro-accelerate/build -lastroaccelerate -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -I/usr/local/cuda-8.0/samples/common/inc/ -lcudart dedispersion.cpp -o test
examples/src/dedispersion.cpp:	//insert option to copy the DDTR output data from GPU memory to the host memory
examples/src/periodicity.cpp: * Compile with: g++ -std=c++11 -I/path/to/astro-accelerate/include/ -L/path/to/astro-accelerate/build -Wl,-rpath,/path/to/astro-accelerate/build -lastroaccelerate -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -I/usr/local/cuda/samples/common/inc/ -lcudart dedispersion.cpp -o test
examples/src/periodicity.cpp:  cudaMemGetInfo(&free, &total);
examples/src/dedispersion_and_analysis.cpp: * Compile with: g++ -std=c++11 -I/path/to/astro-accelerate/include/ -L/path/to/astro-accelerate/build -Wl,-rpath,/path/to/astro-accelerate/build -lastroaccelerate -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -I/usr/local/cuda-8.0/samples/common/inc/ -lcudart dedispersion.cpp -o test
examples/src/filterbank_dedispersion.cpp: * Compile with: g++ -std=c++11 -I/path/to/astro-accelerate/include/ -L/path/to/astro-accelerate/build -Wl,-rpath,/path/to/astro-accelerate/build -lastroaccelerate -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -I/usr/local/cuda/samples/common/inc/ -lcudart dedispersion.cpp -o test
examples/src/fake_signal_single.cpp:	//-----------------------  Init the GPU card
examples/src/fake_signal_single.cpp:	const size_t free_memory = selected_device.free_memory(); // Free memory on the GPU in bytes
examples/src/fake_signal_single.cpp:        //insert option to copy the DDTR output data from GPU memory to the host memory
.gitlab-ci.yml:    - cmake -DCUDA_ARCH="6.1" -DENABLE_TESTS=ON -DENABLE_DOCS=ON ../
CONTRIBUTING.md:Astro-Accelerate depends on the CUDA libraries, and in particular the following libraries
CONTRIBUTING.md:Using CUDA
CONTRIBUTING.md:If CUDA is available and it has been explicitly activated in the build system, then the `ENABLE_CUDA` flag will be set. Use this flag to ensure code compiles and that code tests are run with CUDA as appropriate.
CONTRIBUTING.md:2. Find the line that says `if(ARCH MATCHES ALL|[Aa]ll)`, and add a new line to append to the list, in the same way as the existing ones: `list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)` - then also add it to the comma separated list for `ASTRO_ACCELERATE_CUDA_ARCH_VERSION` and `ASTRO_ACCELERATE_CUDA_SM_VERSION`.
CONTRIBUTING.md:The architecture setting is used when running CMake by adding a flag `-DCUDA_ARCH="X.Y"` (where `X.Y` are substituted with integers reflecting the architecture major.minor version).
src/aa_fdas_device.cu:#include "aa_device_cuda_deprecated_wrappers.cuh"
src/aa_fdas_device.cu:  __global__ void cuda_overlap_copy(float2* d_ext_data, float2* d_cpx_signal, int sigblock, int sig_rfftlen, int sig_tot_convlen, int kern_offset, int total_blocks)
src/aa_fdas_device.cu:  void call_kernel_cuda_overlap_copy(float2 *const d_ext_data, float2 *const d_cpx_signal, const int &sigblock, const int &sig_rfftlen, const int &sig_tot_convlen, const int &kern_offset, const int &total_blocks) {
src/aa_fdas_device.cu:    cuda_overlap_copy<<<KERNLEN/64, 64 >>>(d_ext_data, d_cpx_signal, sigblock, sig_rfftlen, sig_tot_convlen, kern_offset, total_blocks);
src/aa_fdas_device.cu:  __global__ void cuda_overlap_copy_smallblk(float2* d_ext_data, float2* d_cpx_signal, int sigblock, int sig_rfftlen, int sig_tot_convlen, int kern_offset, int total_blocks)
src/aa_fdas_device.cu:  void call_kernel_cuda_overlap_copy_smallblk(const int &blocks, float2 *const d_ext_data, float2 *const d_cpx_signal, const int &sigblock, const int &sig_rfftlen, const int &sig_tot_convlen, const int &kern_offset, const int &total_blocks) {
src/aa_fdas_device.cu:    cuda_overlap_copy_smallblk<<<blocks, KERNLEN>>>(d_ext_data, d_cpx_signal, sigblock, sig_rfftlen, sig_tot_convlen, kern_offset, total_blocks);
src/aa_fdas_device.cu:  __global__ void cuda_convolve_reg_1d_halftemps(float2* d_kernel, float2* d_signal, float2* d_ffdot_plane,int sig_tot_convlen, float scale)
src/aa_fdas_device.cu:  void call_kernel_cuda_convolve_reg_1d_halftemps(const int &blocks, const int &threads, float2 *const d_kernel, float2 *const d_signal, float2 *const d_ffdot_plane, const int &sig_tot_convlen, const float &scale) {
src/aa_fdas_device.cu:    cuda_convolve_reg_1d_halftemps<<<blocks, threads>>>(d_kernel, d_signal, d_ffdot_plane, sig_tot_convlen, scale);
src/aa_fdas_device.cu:  __global__ void cuda_ffdotpow_concat_2d(float2* d_ffdot_plane_cpx, float* d_ffdot_plane, int sigblock, int kern_offset, int total_blocks, int sig_tot_convlen, int sig_totlen)
src/aa_fdas_device.cu:  void call_kernel_cuda_ffdotpow_concat_2d(const dim3 &blocks, const dim3 &threads, float2 *const d_ffdot_plane_cpx, float *const d_ffdot_plane, const int &sigblock, const int &kern_offset, const int &total_blocks, const int &sig_tot_convlen, const int &sig_totlen) {
src/aa_fdas_device.cu:    cuda_ffdotpow_concat_2d<<<blocks, threads>>>(d_ffdot_plane_cpx, d_ffdot_plane, sigblock, kern_offset, total_blocks, sig_tot_convlen, sig_totlen);
src/aa_fdas_device.cu:  __global__ void cuda_ffdotpow_concat_2d_inbin(float2* d_ffdot_plane_cpx, float* d_ffdot_plane, int sigblock, int kern_offset, int total_blocks, int sig_tot_convlen, int sig_totlen)
src/aa_fdas_device.cu:  void call_kernel_cuda_ffdotpow_concat_2d_inbin(const dim3 &blocks, const dim3 &threads, float2 *const d_ffdot_plane_cpx, float *const d_ffdot_plane, const int &sigblock, const int &kern_offset, const int &total_blocks, const int &sig_tot_convlen, const int &sig_totlen) {
src/aa_fdas_device.cu:    cuda_ffdotpow_concat_2d_inbin<<<blocks, threads>>>(d_ffdot_plane_cpx, d_ffdot_plane, sigblock, kern_offset, total_blocks, sig_tot_convlen, sig_totlen);
src/aa_fdas_device.cu:  __global__ void cuda_ffdotpow_concat_2d_ndm_inbin(float2* d_ffdot_plane_cpx, float* d_ffdot_plane, int kernlen, int siglen, int nkern, int kern_offset, int total_blocks, int sig_tot_convlen, int sig_totlen, int ndm)
src/aa_fdas_device.cu:  __global__ void cuda_convolve_customfft_wes_no_reorder02(float2* d_kernel, float2* d_signal, float *d_ffdot_pw, int sigblock, int sig_tot_convlen, int sig_totlen, int offset, float scale)
src/aa_fdas_device.cu:  void call_kernel_cuda_convolve_customfft_wes_no_reorder02(const int &blocks, float2 *const d_kernel, float2 *const d_signal, float *const d_ffdot_pw, const int &sigblock, const int &sig_tot_convlen, const int &sig_totlen, const int &offset, const float &scale) {
src/aa_fdas_device.cu:    cuda_convolve_customfft_wes_no_reorder02<<<blocks, KERNLEN>>>(d_kernel, d_signal, d_ffdot_pw, sigblock, sig_tot_convlen, sig_totlen, offset, scale); 
src/aa_fdas_device.cu:  __global__ void cuda_convolve_customfft_wes_no_reorder02_inbin(float2* d_kernel, float2* d_signal, float *d_ffdot_pw, int sigblock, int sig_tot_convlen, int sig_totlen, int offset, float scale, float2 *ip_edge_points)
src/aa_fdas_device.cu:  void call_kernel_cuda_convolve_customfft_wes_no_reorder02_inbin(const int &blocks, float2 *const d_kernel, float2 *const d_signal, float *const d_ffdot_pw, const int &sigblock, const int &sig_tot_convlen, const int &sig_totlen, const int &offset, const float &scale, float2 *const ip_edge_points) {
src/aa_fdas_device.cu:    cuda_convolve_customfft_wes_no_reorder02_inbin<<<blocks, KERNLEN>>>(d_kernel, d_signal, d_ffdot_pw, sigblock, sig_tot_convlen, sig_totlen, offset, scale, ip_edge_points);
src/aa_fdas_device.cu:  __global__ void GPU_CONV_kFFT_mk11_2elem_2v(float2 const* __restrict__ d_input_signal, float *d_output_plane_reduced, float2 const* __restrict__ d_templates, int useful_part_size, int offset, int nConvolutions, float scale) {
src/aa_fdas_device.cu:  __global__ void GPU_CONV_kFFT_mk11_4elem_2v(float2 const* __restrict__ d_input_signal, float *d_output_plane_reduced, float2 const* __restrict__ d_templates, int useful_part_size, int offset, int nConvolutions, float scale) {
src/aa_fdas_device.cu:  /** \brief Kernel wrapper function for GPU_CONV_kFFT_mk11_4elem_2v kernel function. */
src/aa_fdas_device.cu:  void call_kernel_GPU_CONV_kFFT_mk11_4elem_2v(const dim3 &grid_size, const dim3 &block_size, float2 const*const d_input_signal, float *const d_output_plane_reduced, float2 const*const d_templates, const int &useful_part_size, const int &offset, const int &nConvolutions, const float &scale) {
src/aa_fdas_device.cu:    GPU_CONV_kFFT_mk11_4elem_2v<<<grid_size, block_size>>>(d_input_signal, d_output_plane_reduced, d_templates, useful_part_size, offset, nConvolutions, scale);
src/aa_device_rfi_kernel.cu:#include <cuda.h>
src/aa_device_rfi_kernel.cu:#include <cuda_runtime.h>
src/aa_device_rfi_kernel.cu:  __global__ void rfi_gpu_kernel(unsigned short *d_input, int nchans, int nsamp)
src/aa_device_rfi_kernel.cu:  /** \brief Kernel wrapper function for rfi_gpu_kernel kernel function. */
src/aa_device_rfi_kernel.cu:  void call_kernel_rfi_gpu_kernel(const dim3 &block_size, const dim3 &grid_size,
src/aa_device_rfi_kernel.cu:    rfi_gpu_kernel<<<block_size, grid_size>>>(d_input, nchans, nsamp);
src/aa_device_inference.cu: * COSIT was a NVIDIA funded project and supervised by W Armour
src/aa_device_inference.cu:#include <cuda.h>
src/aa_device_inference.cu:  // ************************************* GPU Blocked Bootstrap ************************************** //
src/aa_device_inference.cu:  void gpu_blocked_bootstrap(float **d_idata, int dms_to_average, int num_els, int ndms, int num_bins, int num_boots, double *mean_boot_out, double *mean_data_out, double *sd_boot_out)
src/aa_device_inference.cu:    cudaMalloc((void**) &d_bin_array, local_ndms * num_bins * sizeof(float));
src/aa_device_inference.cu:    cudaMemcpy(d_bin_array, random_bin_array, local_ndms*num_bins*sizeof(float), cudaMemcpyHostToDevice);
src/aa_device_inference.cu:    cudaMemcpy(d_bin_array, random_bin_array, local_ndms * num_bins * sizeof(float), cudaMemcpyHostToDevice);
src/aa_device_inference.cu:	( cudaMalloc((void**) &d_irand, ( num_bins ) * ( num_boots ) * sizeof(unsigned int)) );
src/aa_device_inference.cu:	( cudaMalloc((void**) &d_irand, ( num_bins ) * ( num_subboots ) * sizeof(unsigned int)) );
src/aa_device_inference.cu:    cudaDeviceSynchronize();
src/aa_device_inference.cu:    cudaMalloc((void**) &d_odata, local_ndms * num_boots * sizeof(double));
src/aa_device_inference.cu:    //cudaCheckMsg("bootstrap kernel execution failed");
src/aa_device_inference.cu:    cudaDeviceSynchronize();
src/aa_device_inference.cu:    cudaMemcpy(boots_array, d_odata, local_ndms * num_boots * sizeof(double), cudaMemcpyDeviceToHost);
src/aa_device_inference.cu:    ( cudaFree(d_bin_array) );
src/aa_device_inference.cu:    ( cudaFree(d_odata) );
src/aa_device_inference.cu:    ( cudaFree(d_irand) );
src/aa_device_spectral_whitening_kernel.cu:#include <cuda.h>
src/aa_device_spectral_whitening_kernel.cu:#include <cuda_runtime.h>
src/aa_device_spectral_whitening_kernel.cu:#include <cuda_runtime_api.h>
src/aa_device_spectral_whitening_kernel.cu:#include "aa_device_cuda_deprecated_wrappers.cuh"
src/aa_device_spectral_whitening_kernel.cu:__global__ void GPU_kernel_spectrum_whitening_SGP1(float *d_input, unsigned long int nSamples) {
src/aa_device_spectral_whitening_kernel.cu:__global__ void GPU_kernel_segmented_MSD(float *d_segmented_MSD, float2 *d_input, int *d_segment_sizes, size_t nSamples, int nSegments){
src/aa_device_spectral_whitening_kernel.cu:__global__ void GPU_kernel_segmented_median(float *d_segmented_MSD, float2 *d_input, int *d_segment_sizes, size_t nSamples, int nSegments){
src/aa_device_spectral_whitening_kernel.cu:__global__ void GPU_kernel_spectrum_whitening_SGP2(float *d_segmented_MSD, float2 *d_input, int *d_segment_sizes, size_t nSamples, int nSegments){
src/aa_device_spectral_whitening_kernel.cu:	const cudaStream_t &stream, 
src/aa_device_spectral_whitening_kernel.cu:	GPU_kernel_spectrum_whitening_SGP1<<< grid_size, block_size, smem_bytes, stream >>>(d_input, nSamples);
src/aa_device_spectral_whitening_kernel.cu:	const  cudaStream_t &stream, 
src/aa_device_spectral_whitening_kernel.cu:	GPU_kernel_segmented_MSD<<< grid_size, block_size, smem_bytes, stream >>>(d_segmented_MSD, d_input, d_segment_sizes, nSamples, nSegments);
src/aa_device_spectral_whitening_kernel.cu:	const  cudaStream_t &stream, 
src/aa_device_spectral_whitening_kernel.cu:	GPU_kernel_segmented_median<Sort_256><<< grid_size, block_size, smem_bytes, stream >>>(d_segmented_MSD, d_input, d_segment_sizes, nSamples, nSegments);
src/aa_device_spectral_whitening_kernel.cu:	const  cudaStream_t &stream, 
src/aa_device_spectral_whitening_kernel.cu:	GPU_kernel_spectrum_whitening_SGP2<<< grid_size, block_size, smem_bytes, stream >>>(d_segmented_MSD, d_input, d_segment_sizes, nSamples, nSegments);
src/aa_device_MSD_shared_kernel_functions.cu:  __global__ void MSD_GPU_final_regular(float *d_input, float *d_output, int size) {
src/aa_device_MSD_shared_kernel_functions.cu:  __global__ void MSD_GPU_final_regular(float *d_input, float *d_MSD, float *d_pp, int size) {
src/aa_device_MSD_shared_kernel_functions.cu:  __global__ void MSD_GPU_final_nonregular(float *d_input, float *d_MSD, int size) {
src/aa_device_MSD_shared_kernel_functions.cu:  __global__ void MSD_GPU_final_nonregular(float *d_input, float *d_MSD, float *d_pp, int size) {
src/aa_device_MSD_shared_kernel_functions.cu:  __global__ void MSD_GPU_Interpolate_linear(float *d_MSD_DIT, float *d_MSD_interpolated, int *d_MSD_DIT_widths, int MSD_DIT_size, int *boxcar, int max_width_performed){
src/aa_device_MSD_shared_kernel_functions.cu:  /** \brief Kernel wrapper function to MSD_GPU_final_regular kernel function. */
src/aa_device_MSD_shared_kernel_functions.cu:  void call_kernel_MSD_GPU_final_regular(const dim3 &grid_size, const dim3 &block_size, float *const d_input, float *const d_output, const int &size) {
src/aa_device_MSD_shared_kernel_functions.cu:    MSD_GPU_final_regular<<<grid_size, block_size>>>(d_input, d_output, size);
src/aa_device_MSD_shared_kernel_functions.cu:  /** \brief Kernel wrapper function to MSD_GPU_final_regular kernel function. */
src/aa_device_MSD_shared_kernel_functions.cu:  void call_kernel_MSD_GPU_final_regular(const dim3 &grid_size, const dim3 &block_size, float *const d_input, float *const d_MSD, float *const d_pp, const int &size) {
src/aa_device_MSD_shared_kernel_functions.cu:    MSD_GPU_final_regular<<<grid_size,block_size>>>(d_input, d_MSD, d_pp, size);
src/aa_device_MSD_shared_kernel_functions.cu:  /** \brief Kernel wrapper function to MSD_GPU_final_nonregular kernel function. */
src/aa_device_MSD_shared_kernel_functions.cu:  void call_kernel_MSD_GPU_final_nonregular(const dim3 &grid_size, const dim3 &block_size, float *const d_input, float *const d_MSD, const int &size) {
src/aa_device_MSD_shared_kernel_functions.cu:    MSD_GPU_final_nonregular<<<grid_size, block_size>>>(d_input, d_MSD, size);
src/aa_device_MSD_shared_kernel_functions.cu:  /** \brief Kernel wrapper function to MSD_GPU_final_nonregular kernel function. */
src/aa_device_MSD_shared_kernel_functions.cu:  void call_kernel_MSD_GPU_final_nonregular(const dim3 &grid_size, const dim3 &block_size, float *const d_input, float *const d_MSD, float *const d_pp, const int &size) {
src/aa_device_MSD_shared_kernel_functions.cu:    MSD_GPU_final_nonregular<<<grid_size, block_size>>>(d_input, d_MSD, d_pp, size);
src/aa_device_MSD_shared_kernel_functions.cu:  /** \brief Kernel wrapper function to MSD_GPU_Interpolate_linear kernel function. */
src/aa_device_MSD_shared_kernel_functions.cu:  void call_kernel_MSD_GPU_Interpolate_linear(const dim3 &grid_size, const dim3 &block_size, float *const d_MSD_DIT, float *const d_MSD_interpolated, int *const d_MSD_DIT_widths, const int &MSD_DIT_size, int *const boxcar, const int &max_width_performed) {
src/aa_device_MSD_shared_kernel_functions.cu:    MSD_GPU_Interpolate_linear<<<grid_size, block_size>>>(d_MSD_DIT, d_MSD_interpolated, d_MSD_DIT_widths, MSD_DIT_size, boxcar, max_width_performed);
src/aa_device_jerk_search.cu:#include <cuda_profiler_api.h>
src/aa_device_jerk_search.cu:#include "aa_gpu_timer.hpp"
src/aa_device_jerk_search.cu:	class GPU_Memory_for_JERK_Search {
src/aa_device_jerk_search.cu:			cudaMalloc((void **) &d_jerk_filters,  filter_size_bytes);
src/aa_device_jerk_search.cu:			cudaMemcpy(d_jerk_filters, h_jerk_filters, filter_size_bytes, cudaMemcpyHostToDevice);
src/aa_device_jerk_search.cu:			cudaMemcpy(h_jerk_filters, d_jerk_filters, filter_size_bytes, cudaMemcpyDeviceToHost);
src/aa_device_jerk_search.cu:			cudaMalloc((void **) &d_signal,  signal_size_bytes);
src/aa_device_jerk_search.cu:			cudaMemcpy(d_signal, h_signal, signal_size_bytes, cudaMemcpyHostToDevice);
src/aa_device_jerk_search.cu:			cudaMalloc((void **) &d_output,  output_size*sizeof(float));
src/aa_device_jerk_search.cu:			cudaMemcpy(h_output, d_output, output_size*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_jerk_search.cu:			cudaFree(d_signal);
src/aa_device_jerk_search.cu:			cudaFree(d_jerk_filters);
src/aa_device_jerk_search.cu:			cudaFree(d_output);
src/aa_device_jerk_search.cu:		aa_gpu_timer timer_total, timer_DM, timer;
src/aa_device_jerk_search.cu:		cudaError_t cudaError;
src/aa_device_jerk_search.cu:		if ( cudaSuccess != cudaMalloc((void **) &d_jerk_filters,  sizeof(float2)*jerk_strategy.filter_padded_size() )) {
src/aa_device_jerk_search.cu:			printf("Cannot allocate GPU memory for JERK filters!\n");
src/aa_device_jerk_search.cu:		if ( cudaSuccess != cudaMemcpy(d_jerk_filters, h_jerk_filters, jerk_strategy.filter_padded_size_bytes(), cudaMemcpyHostToDevice) ) {
src/aa_device_jerk_search.cu:		if ( cudaSuccess != cudaMalloc((void **) &d_DM_trial,  sizeof(float)*jerk_strategy.nSamples_time_dom() )) {
src/aa_device_jerk_search.cu:			printf("Cannot allocate GPU memory for DM trial!\n");
src/aa_device_jerk_search.cu:		if ( cudaSuccess != cudaMalloc((void **) &d_DM_trial_ffted,  sizeof(float2)*jerk_strategy.nSamples_freq_dom() )) {
src/aa_device_jerk_search.cu:			printf("Cannot allocate GPU memory for FFTed DM trial!\n");
src/aa_device_jerk_search.cu:		if ( cudaSuccess != cudaMalloc((void **) &d_MSD,  MSD_PARTIAL_SIZE*jerk_strategy.nHarmonics()*sizeof(float))) {
src/aa_device_jerk_search.cu:			printf("Cannot allocate GPU memory for MSD!\n");
src/aa_device_jerk_search.cu:		if ( cudaSuccess != cudaMalloc((void**) &gmem_peak_pos, sizeof(unsigned int))) {
src/aa_device_jerk_search.cu:			printf("Cannot allocate GPU memory for peak position!\n");
src/aa_device_jerk_search.cu:		cudaMemset((void*) gmem_peak_pos, 0, sizeof(unsigned int));
src/aa_device_jerk_search.cu:			if ( cudaSuccess != cudaMalloc((void **) &d_ZW_candidates, max_nZWCandidates*sizeof(float))) {
src/aa_device_jerk_search.cu:				printf("Cannot allocate GPU memory for ZW plane candidates!\n");
src/aa_device_jerk_search.cu:			if ( cudaSuccess != cudaMalloc((void **) &d_ZW_planes, ZW_planes_size_bytes)) {
src/aa_device_jerk_search.cu:				printf("Cannot allocate GPU memory for ZW planes!\n");
src/aa_device_jerk_search.cu:			if ( cudaSuccess != cudaMalloc((void **) &d_MSD_workarea, MSD_conf.nBlocks_total*MSD_PARTIAL_SIZE*sizeof(float))) {
src/aa_device_jerk_search.cu:				printf("Cannot allocate GPU memory for MSD workarea!\n");
src/aa_device_jerk_search.cu:				cudaError = cudaMemset((void*) d_DM_trial, 0, jerk_strategy.nSamples_time_dom()*sizeof(float));
src/aa_device_jerk_search.cu:				if ( cudaError != cudaSuccess) {
src/aa_device_jerk_search.cu:					printf("Error %s\n", cudaGetErrorString(cudaError));
src/aa_device_jerk_search.cu:				cudaError = cudaMemcpy(d_DM_trial, dedispersed_data[active_range][active_DM], jerk_strategy.nSamples_time_dom()*sizeof(float), cudaMemcpyHostToDevice);
src/aa_device_jerk_search.cu:				if ( cudaError != cudaSuccess) {
src/aa_device_jerk_search.cu:					printf("Error %s\n", cudaGetErrorString(cudaError));
src/aa_device_jerk_search.cu:				cudaStream_t stream; stream = NULL;
src/aa_device_jerk_search.cu:				cudaError = cudaGetLastError();
src/aa_device_jerk_search.cu:				if (cudaError != cudaSuccess) {
src/aa_device_jerk_search.cu:					std::cerr << "CUDA Runtime Error at spectrum_whitening_SGP2" << std::endl;
src/aa_device_jerk_search.cu:					std::cerr << cudaGetErrorString(cudaError) << std::endl;
src/aa_device_jerk_search.cu:					cudaError = cudaGetLastError();
src/aa_device_jerk_search.cu:					if (cudaError != cudaSuccess) {
src/aa_device_jerk_search.cu:						std::cerr << "CUDA Runtime Error at convolution kernel" << std::endl;
src/aa_device_jerk_search.cu:						std::cerr << cudaGetErrorString(cudaError) << std::endl;
src/aa_device_jerk_search.cu:							cudaMemcpy(h_ZW_planes, d_ZW_planes, ZW_planes_size_bytes, cudaMemcpyDeviceToHost);
src/aa_device_jerk_search.cu:							cudaMemcpy(h_MSD, d_MSD, MSD_PARTIAL_SIZE*jerk_strategy.nHarmonics()*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_jerk_search.cu:						cudaMemset((void*) gmem_peak_pos, 0, sizeof(unsigned int));
src/aa_device_jerk_search.cu:						if ( cudaSuccess != cudaMemcpy(&nCandidates, gmem_peak_pos, sizeof(unsigned int), cudaMemcpyDeviceToHost)) {
src/aa_device_jerk_search.cu:							cudaMemcpy(h_candidates, d_ZW_candidates, nCandidates*4*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_jerk_search.cu:						allcandidates.AddSubListFromGPU(nCandidates, d_ZW_candidates, w, DM, jerk_strategy.nSamples_time_dom(), jerk_strategy.nFilters_z(), jerk_strategy.z_max_search_limit(), jerk_strategy.z_search_step(), sampling_time*((float) inBin[active_range]), inBin[active_range]);
src/aa_device_jerk_search.cu:			if ( cudaSuccess != cudaFree(d_ZW_candidates)) printf("ERROR while deallocating d_ZW_candidates!\n");
src/aa_device_jerk_search.cu:			if ( cudaSuccess != cudaFree(d_ZW_planes)) printf("ERROR while deallocating d_ZW_planes!\n");
src/aa_device_jerk_search.cu:			if ( cudaSuccess != cudaFree(d_MSD_workarea)) printf("ERROR while deallocating d_MSD_workarea!\n");
src/aa_device_jerk_search.cu:		if ( cudaSuccess != cudaFree(d_jerk_filters)) printf("ERROR while deallocating d_jerk_filters!\n");
src/aa_device_jerk_search.cu:		if ( cudaSuccess != cudaFree(d_DM_trial)) printf("ERROR while deallocating d_DM_trial!\n");
src/aa_device_jerk_search.cu:		if ( cudaSuccess != cudaFree(d_DM_trial_ffted)) printf("ERROR while deallocating d_DM_trial_ffted!\n");
src/aa_device_jerk_search.cu:		if ( cudaSuccess != cudaFree(d_MSD)) printf("ERROR while deallocating d_MSD!\n");
src/aa_device_corner_turn_kernel.cu:#include <cuda.h>
src/aa_device_corner_turn_kernel.cu:#include <cuda_runtime.h>
src/aa_device_single_pulse_search.cu:    //---------> Specific nVidia stuff
src/aa_device_single_pulse_search.cu:    cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
src/aa_device_single_pulse_search.cu:    cudaDeviceSetSharedMemConfig (cudaSharedMemBankSizeEightByte);
src/aa_device_single_pulse_search.cu:    //cudaMemcpyToSymbol(c_sqrt_taps, h_sqrt_taps, (PD_MAXTAPS+1)*sizeof(float));
src/aa_device_single_pulse_search.cu:    //---------> CUDA block and CUDA grid parameters
src/aa_device_single_pulse_search.cu:    int nCUDAblocks_x = nBlocks;
src/aa_device_single_pulse_search.cu:    int nCUDAblocks_y = nDMs;
src/aa_device_single_pulse_search.cu:    dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);
src/aa_device_single_pulse_search.cu:    call_kernel_PD_SEARCH_GPU(gridSize, blockSize, SM_size, d_input, d_output, d_output_taps, d_MSD, maxTaps, nTimesamples);
src/aa_zero_dm.cpp:	cudaError_t e;
src/aa_zero_dm.cpp:	e = cudaMalloc((void **)d_normalization_factor, nchans*sizeof(float));
src/aa_zero_dm.cpp:	if (e != cudaSuccess) {
src/aa_zero_dm.cpp:		LOG(log_level::error, "Could not allocate memory for d_normalization_factor (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_zero_dm.cpp:	cudaMemcpy(d_normalization_factor, local_bandpass_normalization.data(), nchans*sizeof(float), cudaMemcpyHostToDevice);
src/aa_zero_dm.cpp:    cudaDeviceSynchronize();
src/aa_zero_dm.cpp:	e =cudaFree(d_normalization_factor);
src/aa_zero_dm.cpp:	if (e != cudaSuccess) {
src/aa_zero_dm.cpp:		LOG(log_level::error, "Cannot free d_normalization_factor memory: (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_zero_dm.cpp:    cudaDeviceSynchronize();
src/aa_device_stretch.cu:  /** \brief Doppler stretch. Simple corner turn on the GPU. */
src/aa_device_stretch.cu:  void stretch_gpu(cudaEvent_t event, cudaStream_t stream, int acc, int samps, float tsamp, float *d_input, float *d_output) {
src/aa_device_stretch.cu:    cudaStreamWaitEvent(stream, event, 0);
src/aa_device_stretch.cu:    //getLastCudaError("stretch_kernel failed");
src/aa_device_stretch.cu:    cudaEventRecord(event, stream);
src/aa_device_convolution_kernel.cu:#include "aa_device_cuda_deprecated_wrappers.cuh"
src/aa_device_convolution_kernel.cu:	__global__ void k_customFFT_GPU_forward(float2 *d_input, float2* d_output) {
src/aa_device_convolution_kernel.cu:	__global__ void k_GPU_conv_OLS_via_customFFT(
src/aa_device_convolution_kernel.cu:	void call_kernel_k_customFFT_GPU_forward(
src/aa_device_convolution_kernel.cu:				k_customFFT_GPU_forward<FFT_256><<<grid_size, block_size>>>(d_input, d_output);
src/aa_device_convolution_kernel.cu:				k_customFFT_GPU_forward<FFT_512><<<grid_size, block_size>>>(d_input, d_output);
src/aa_device_convolution_kernel.cu:				k_customFFT_GPU_forward<FFT_1024><<<grid_size, block_size>>>(d_input, d_output);
src/aa_device_convolution_kernel.cu:				k_customFFT_GPU_forward<FFT_2048><<<grid_size, block_size>>>(d_input, d_output);
src/aa_device_convolution_kernel.cu:				k_customFFT_GPU_forward<FFT_4096><<<grid_size, block_size>>>(d_input, d_output);
src/aa_device_convolution_kernel.cu:	void call_kernel_k_GPU_conv_OLS_via_customFFT(
src/aa_device_convolution_kernel.cu:				k_GPU_conv_OLS_via_customFFT<FFT_256><<<grid_size, block_size>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters, scale);
src/aa_device_convolution_kernel.cu:				k_GPU_conv_OLS_via_customFFT<FFT_512><<<grid_size, block_size>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters, scale);
src/aa_device_convolution_kernel.cu:				k_GPU_conv_OLS_via_customFFT<FFT_1024><<<grid_size, block_size>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters, scale);
src/aa_device_convolution_kernel.cu:				k_GPU_conv_OLS_via_customFFT<FFT_2048><<<grid_size, block_size>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters, scale);
src/aa_device_convolution_kernel.cu:				k_GPU_conv_OLS_via_customFFT<FFT_4096><<<grid_size, block_size>>>(d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters, scale);
src/aa_device_threshold_kernel.cu:#include "aa_device_cuda_deprecated_wrappers.cuh"
src/aa_device_threshold_kernel.cu:  __global__ void THR_GPU_WARP(float const* __restrict__ d_input, ushort *d_input_taps, unsigned int *d_output_list_DM, unsigned int *d_output_list_TS, float *d_output_list_SNR, unsigned int *d_output_list_BW, int *gmem_pos, float threshold, int nTimesamples, int offset, int shift, int max_list_size, int DIT_value) {
src/aa_device_threshold_kernel.cu:  __global__ void GPU_Threshold_for_periodicity_normal_kernel(float const* __restrict__ d_input, ushort *d_input_harms, float *d_output_list, int *gmem_pos, float const* __restrict__ d_MSD, float threshold, int nTimesamples, int nDMs, int DM_shift, int max_list_size, int DIT_value) {
src/aa_device_threshold_kernel.cu:  __global__ void GPU_Threshold_for_periodicity_transposed_kernel(float const* __restrict__ d_input, ushort *d_input_harms, float *d_output_list, int *gmem_pos, float const* __restrict__ d_MSD, float threshold, int primary_size, int secondary_size, int DM_shift, int max_list_size, int DIT_value) {
src/aa_device_threshold_kernel.cu:  /** \brief Kernel wrapper function for THR_GPU_WARP kernel function. */
src/aa_device_threshold_kernel.cu:  void call_kernel_THR_GPU_WARP(
src/aa_device_threshold_kernel.cu:      THR_GPU_WARP<<<grid_size, block_size>>>(
src/aa_device_threshold_kernel.cu:  /** \brief Kernel wrapper function for GPU_Threshold_for_periodicity_kernel_old kernel function. */
src/aa_device_threshold_kernel.cu:  void call_kernel_GPU_Threshold_for_periodicity_normal_kernel(
src/aa_device_threshold_kernel.cu:      GPU_Threshold_for_periodicity_normal_kernel<<<grid_size, block_size>>>(
src/aa_device_threshold_kernel.cu:  /** \brief Kernel wrapper function for GPU_Threshold_for_periodicity_kernel kernel function. */
src/aa_device_threshold_kernel.cu:  void call_kernel_GPU_Threshold_for_periodicity_transposed_kernel(
src/aa_device_threshold_kernel.cu:      GPU_Threshold_for_periodicity_transposed_kernel<<<grid_size, block_size>>>(
src/aa_device_zero_dm_kernel.cu:#include <cuda.h>
src/aa_device_zero_dm_kernel.cu:#include <cuda_runtime.h>
src/aa_device_zero_dm_kernel.cu:#include "aa_device_cuda_deprecated_wrappers.cuh"
src/aa_device_set_stretch.cu:  void set_stretch_gpu(cudaEvent_t event, cudaStream_t stream, int samps, float mean, float *d_input) {
src/aa_device_set_stretch.cu:    cudaStreamWaitEvent(stream, event, 0);
src/aa_device_set_stretch.cu:    //getLastCudaError("stretch_kernel failed");
src/aa_device_set_stretch.cu:    cudaEventRecord(event, stream);
src/aa_device_threshold.cu:    //---------> Specific nVidia stuff
src/aa_device_threshold.cu:    cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
src/aa_device_threshold.cu:    cudaDeviceSetSharedMemConfig (cudaSharedMemBankSizeFourByte);
src/aa_device_threshold.cu:    int nCUDAblocks_x, nCUDAblocks_y;
src/aa_device_threshold.cu:	nCUDAblocks_x = nBlocks;
src/aa_device_threshold.cu:	nCUDAblocks_y = nDMs/THR_WARPS_PER_BLOCK;
src/aa_device_threshold.cu:	gridSize.x=nCUDAblocks_x; gridSize.y=nCUDAblocks_y; gridSize.z=1;
src/aa_device_threshold.cu:	call_kernel_THR_GPU_WARP(gridSize, blockSize, &d_input[output_offset], &d_input_taps[output_offset], d_output_list_DM, d_output_list_TS, d_output_list_SNR, d_output_list_BW, gmem_pos, threshold, decimated_timesamples, decimated_timesamples-local_offset, shift, max_list_size, (1<<f));
src/aa_device_threshold.cu:	//checkCudaErrors(cudaGetLastError());
src/aa_device_threshold.cu:    //---------> Nvidia stuff
src/aa_device_threshold.cu:    cudaDeviceProp deviceProp;
src/aa_device_threshold.cu:    cudaGetDeviceProperties(&deviceProp, CARD);
src/aa_device_threshold.cu:      call_kernel_GPU_Threshold_for_periodicity_transposed_kernel(gridSize, blockSize, &d_input[shift*primary_size], d_input_harms, d_output_list, gmem_pos, d_MSD, threshold, primary_size, secondary_size_per_chunk[f], DM_shift, max_list_size, inBin);
src/aa_device_threshold.cu:      //checkCudaErrors(cudaGetLastError());
src/aa_device_threshold.cu:    //---------> Nvidia stuff
src/aa_device_threshold.cu:    cudaDeviceProp deviceProp;
src/aa_device_threshold.cu:    cudaGetDeviceProperties(&deviceProp, CARD);
src/aa_device_threshold.cu:    call_kernel_GPU_Threshold_for_periodicity_normal_kernel(gridSize, blockSize, d_input_SNR, d_input_harms, d_output_list, gmem_pos, d_MSD, threshold, nTimesamples, nDMs, DM_shift, max_list_size, inBin);
src/aa_device_single_pulse_search_kernel.cu:#include <cuda.h>
src/aa_device_single_pulse_search_kernel.cu:#include <cuda_runtime.h>
src/aa_device_single_pulse_search_kernel.cu:  __global__ void PD_SEARCH_GPU(float const* __restrict__ d_input, float *d_output, float *d_output_taps, float *d_MSD, int maxTaps, int nTimesamples)
src/aa_device_single_pulse_search_kernel.cu:  /** \brief Kernel wrapper function for PD_SEARCH_GPU kernel function. */
src/aa_device_single_pulse_search_kernel.cu:  void call_kernel_PD_SEARCH_GPU(const dim3 &grid_size, const dim3 &block_size, const int &sm_size,
src/aa_device_single_pulse_search_kernel.cu:    PD_SEARCH_GPU<<<grid_size, block_size, sm_size>>>(d_input, d_output, d_output_taps, d_MSD, maxTaps, nTimesamples);
src/aa_device_power.cu:  void power_gpu(cudaEvent_t event, cudaStream_t stream, int samps, int acc, cufftComplex *d_signal_fft, float *d_signal_power)
src/aa_device_power.cu:    cudaStreamWaitEvent(stream, event, 0);
src/aa_device_power.cu:    //getLastCudaError("power_kernel failed");
src/aa_device_power.cu:    cudaEventRecord(event, stream);
src/aa_device_power.cu:    call_kernel_GPU_simple_power_and_interbin_kernel(gridSize,blockDim, d_input, d_power_output, d_interbin_output, nTimesamples, sqrt(nTimesamples));
src/aa_device_acceleration_fdas.cu:#include <cuda_profiler_api.h>
src/aa_device_acceleration_fdas.cu:    astroaccelerate::fdas_gpuarrays gpuarrays;
src/aa_device_acceleration_fdas.cu:    double t_gpu = 0.0, t_gpu_i = 0.0;
src/aa_device_acceleration_fdas.cu:	cudaError_t e = cudaMemGetInfo(&mfree, &mtotal);
src/aa_device_acceleration_fdas.cu:	if(e != cudaSuccess) {
src/aa_device_acceleration_fdas.cu:	  LOG(log_level::error, "Could not cudaMemGetInfo in aa_device_acceleration_fdas.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_acceleration_fdas.cu:	//Allocating gpu arrays
src/aa_device_acceleration_fdas.cu:	gpuarrays.mem_insig = params.nsamps * sizeof(float);
src/aa_device_acceleration_fdas.cu:	gpuarrays.mem_rfft = params.rfftlen * sizeof(float2);
src/aa_device_acceleration_fdas.cu:	gpuarrays.mem_extsig = params.extlen * sizeof(float2);
src/aa_device_acceleration_fdas.cu:	gpuarrays.mem_ffdot = mem_ffdot;
src/aa_device_acceleration_fdas.cu:	gpuarrays.mem_ffdot_cpx = mem_ffdot_cpx;
src/aa_device_acceleration_fdas.cu:	gpuarrays.mem_ipedge = params.nblocks * 2;
src/aa_device_acceleration_fdas.cu:	gpuarrays.mem_max_list_size = mem_max_list_size;
src/aa_device_acceleration_fdas.cu:	printf("Total memory needed on GPU for arrays to process 1 DM: %.4f GB\nfloat ffdot plane (for power spectrum) = %.4f GB.\nTemplate array %.4f GB\nOne dimensional signals %.4f\n1 GB = %f",
src/aa_device_acceleration_fdas.cu:	//getLastCudaError("\nCuda Error\n");
src/aa_device_acceleration_fdas.cu:	 * cudaStream_t *stream_list = malloc(number_dm_concurrently * sizeof(cudaStream_t));
src/aa_device_acceleration_fdas.cu:    	 *      cudaStreamCreate(&stream_list[ii]);
src/aa_device_acceleration_fdas.cu:	 * gpuarrays *gpuarrays_list = malloc(number_dm_concurrently * sizeof(gpuarrays));
src/aa_device_acceleration_fdas.cu:	 * -- allocate memory -> create a function which uses cudaMallocHost to alloc pinned memory
src/aa_device_acceleration_fdas.cu:	 *     fdas_alloc_gpu_arrays(&gpuarrays_list[ii], &cmdargs);
src/aa_device_acceleration_fdas.cu:	 * 	   getLastCudaError("\nCuda Error\n");
src/aa_device_acceleration_fdas.cu:	fdas_alloc_gpu_arrays(&gpuarrays, &cmdargs);
src/aa_device_acceleration_fdas.cu:	//getLastCudaError("\nCuda Error\n");
src/aa_device_acceleration_fdas.cu:	// Calculate kernel templates on CPU and upload-fft on GPU
src/aa_device_acceleration_fdas.cu:	fdas_create_acc_kernels(gpuarrays.d_kernel, &cmdargs);
src/aa_device_acceleration_fdas.cu:	//getLastCudaError("\nCuda Error\n");
src/aa_device_acceleration_fdas.cu:	 *     fdas_create_acc_kernels(gpuarrays_list[ii].d_kernel, &cmdargs);
src/aa_device_acceleration_fdas.cu:	 *     getLastCudaError("\nCuda Error\n");
src/aa_device_acceleration_fdas.cu:	fdas_cuda_create_fftplans(&fftplans, &params);
src/aa_device_acceleration_fdas.cu:	//getLastCudaError("\nCuda Error\n");
src/aa_device_acceleration_fdas.cu:	//cudaGetLastError(); //reset errors
src/aa_device_acceleration_fdas.cu:	    e = cudaMemcpy(gpuarrays.d_in_signal, output_buffer[i][dm_count], processed*sizeof(float), cudaMemcpyHostToDevice);
src/aa_device_acceleration_fdas.cu:	    if(e != cudaSuccess) {
src/aa_device_acceleration_fdas.cu:	      LOG(log_level::error, "Could not cudaMemcpy in aa_device_acceleration.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_acceleration_fdas.cu:	    //checkCudaErrors( cudaMemcpyAsync(gpuarrays_list[i].d_in_signal, output_buffer[i][dm_count], processed*sizeof(float), cudaMemcpyHostToDevice, stream_list[ii]));
src/aa_device_acceleration_fdas.cu:	    cudaDeviceSynchronize();
src/aa_device_acceleration_fdas.cu:	    t_gpu = (double) (t_end.tv_sec + (t_end.tv_usec / 1000000.0)  - t_start.tv_sec - (t_start.tv_usec/ 1000000.0)) * 1000.0;
src/aa_device_acceleration_fdas.cu:	    t_gpu_i = (t_gpu /(double)titer);
src/aa_device_acceleration_fdas.cu:	    //printf("\n\nAverage vector transfer time of %d float samples (%.2f Mb) from 1000 iterations: %f ms\n\n", params.nsamps, (float)(gpuarrays.mem_insig)/mbyte, t_gpu_i);
src/aa_device_acceleration_fdas.cu:	    cudaProfilerStart(); //exclude cuda initialization ops
src/aa_device_acceleration_fdas.cu:	      fdas_cuda_basic(&fftplans, &gpuarrays, &cmdargs, &params );
src/aa_device_acceleration_fdas.cu:	      cudaDeviceSynchronize();
src/aa_device_acceleration_fdas.cu:	      t_gpu = (double) (t_end.tv_sec + (t_end.tv_usec / 1000000.0)  - t_start.tv_sec - (t_start.tv_usec/ 1000000.0)) * 1000.0;
src/aa_device_acceleration_fdas.cu:	      t_gpu_i = (t_gpu / (double)iter);
src/aa_device_acceleration_fdas.cu:	      //printf("\n\nConvolution using basic algorithm with cuFFT\nTotal process took: %f ms per iteration \nTotal time %d iterations: %f ms\n", t_gpu_i, iter, t_gpu);
src/aa_device_acceleration_fdas.cu:	      fdas_cuda_customfft(&fftplans, &gpuarrays, &cmdargs, &params);
src/aa_device_acceleration_fdas.cu:	      cudaDeviceSynchronize();
src/aa_device_acceleration_fdas.cu:	      t_gpu = (double) (t_end.tv_sec + (t_end.tv_usec / 1000000.0)  - t_start.tv_sec - (t_start.tv_usec/ 1000000.0)) * 1000.0;
src/aa_device_acceleration_fdas.cu:	      t_gpu_i = (t_gpu / (double)iter);
src/aa_device_acceleration_fdas.cu:	      printf("\n\nConvolution using custom FFT:\nTotal process took: %f ms\n per iteration \nTotal time %d iterations: %f ms\n", t_gpu_i, iter, t_gpu);
src/aa_device_acceleration_fdas.cu:	      if ( cudaSuccess != cudaMalloc((void**) &d_MSD, sizeof(float)*3)) printf("Allocation error!\n");
src/aa_device_acceleration_fdas.cu:	      if ( cudaSuccess != cudaMalloc((void**) &gmem_fdas_peak_pos, 1*sizeof(int))) printf("Allocation error!\n");
src/aa_device_acceleration_fdas.cu:	      cudaMemset((void*) gmem_fdas_peak_pos, 0, sizeof(int));
src/aa_device_acceleration_fdas.cu:		MSD_grid_outlier_rejection(d_MSD, gpuarrays.d_ffdot_pwr, 32, 32, ibin*params.siglen, NKERN, 0, sigma_constant);
src/aa_device_acceleration_fdas.cu:		Find_MSD(d_MSD, gpuarrays.d_ffdot_pwr, params.siglen/ibin, NKERN, 0, sigma_constant, 1);
src/aa_device_acceleration_fdas.cu:	      //checkCudaErrors(cudaGetLastError());
src/aa_device_acceleration_fdas.cu:	      fdas_write_test_ffdot(&gpuarrays, &cmdargs, &params, dm_low[i], dm_count, dm_step[i]);
src/aa_device_acceleration_fdas.cu:	      PEAK_FIND_FOR_FDAS(gpuarrays.d_ffdot_pwr, gpuarrays.d_fdas_peak_list, d_MSD, NKERN, ibin*params.siglen, cmdargs.thresh, params.max_list_length, gmem_fdas_peak_pos, dm_count*dm_step[i] + dm_low[i]);
src/aa_device_acceleration_fdas.cu:	      e = cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_acceleration_fdas.cu:	      if(e != cudaSuccess) {
src/aa_device_acceleration_fdas.cu:		LOG(log_level::error, "Could not cudaMemcpy in aa_device_acceleration_fdas.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_acceleration_fdas.cu:	      e = cudaMemcpy(&list_size, gmem_fdas_peak_pos, sizeof(unsigned int), cudaMemcpyDeviceToHost);
src/aa_device_acceleration_fdas.cu:	      if(e != cudaSuccess) {
src/aa_device_acceleration_fdas.cu:		LOG(log_level::error, "Could not cudaMemcpy in aa_device_acceleration_fdas.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_acceleration_fdas.cu:	      fdas_write_list(&gpuarrays, &cmdargs, &params, h_MSD, dm_low[i], dm_count, dm_step[i], list_size);
src/aa_device_acceleration_fdas.cu:	      fdas_write_ffdot(&gpuarrays, &cmdargs, &params, dm_low[i], dm_count, dm_step[i]);
src/aa_device_acceleration_fdas.cu:					fdas_write_list(&gpuarrays, &cmdargs, &params, h_MSD, dm_low[i], dm_count, dm_step[i], list_size);
src/aa_device_acceleration_fdas.cu:			cudaFree(d_MSD);
src/aa_device_acceleration_fdas.cu:			cudaFree(gmem_fdas_peak_pos);
src/aa_device_acceleration_fdas.cu:			fdas_write_ffdot(&gpuarrays, &cmdargs, &params, dm_low[i], dm_count, dm_step[i]);
src/aa_device_acceleration_fdas.cu:	// releasing GPU arrays
src/aa_device_acceleration_fdas.cu:	fdas_free_gpu_arrays(&gpuarrays, &cmdargs);
src/aa_device_acceleration_fdas.cu:	 * -- don't forget it's pinned memory here so write a function which uses cudaFreeHost
src/aa_device_acceleration_fdas.cu:	 *     fdas_free_gpu_arrays(&gpuarrays_list[ii], &cmdargs);
src/aa_device_acceleration_fdas.cu:	 *     cudaStreamDestroy(&stream_list[i]);
src/aa_device_set_stretch_kernel.cu:#include <cuda.h>
src/aa_device_set_stretch_kernel.cu:#include <cuda_runtime.h>
src/aa_device_set_stretch_kernel.cu:				      const int &smem_bytes, const cudaStream_t &stream,
src/aa_fdas_util.cu:    printf("The program reads an accelerated sinusoid or optionally a pulse with a number of harmonics,\nand performs a (single harmonic) fourier domain acceleration search on this signal on a GPU\n");
src/aa_fdas_util.cu:    printf(" %-13s %-2s %-.60s\n", "-devid [i]", ":", "CUDA device id to use (default is 0).\n");
src/aa_device_spectral_whitening.cu:	void calculate_mean_for_segments(float *d_MSD_segments, float2 *d_input, size_t nSamples, int nDMs, int *d_segment_sizes, int nSegments, cudaStream_t &stream){
src/aa_device_spectral_whitening.cu:	void calculate_median_for_segments(float *d_MSD_segments, float2 *d_input, size_t nSamples, int nDMs, int *d_segment_sizes, int nSegments, cudaStream_t &stream){
src/aa_device_spectral_whitening.cu:	void spectrum_whitening_SGP1(float *d_input, size_t nSamples, int nDMs, cudaStream_t &stream){
src/aa_device_spectral_whitening.cu:	void spectrum_whitening_SGP2(float2 *d_input, size_t nSamples, int nDMs, bool enable_median, cudaStream_t &stream) {
src/aa_device_spectral_whitening.cu:		//------------ Allocate and copy segment sizes to the GPU
src/aa_device_spectral_whitening.cu:		if ( cudaSuccess != cudaMalloc((void **) &d_segment_sizes, ss_size )) {
src/aa_device_spectral_whitening.cu:		if ( cudaSuccess != cudaMalloc((void **) &d_MSD_segments, ss_MSD_size )) {
src/aa_device_spectral_whitening.cu:		cudaError_t e = cudaMemcpy(d_segment_sizes, segment_sizes.data(), ss_size, cudaMemcpyHostToDevice);
src/aa_device_spectral_whitening.cu:		if(e != cudaSuccess) {
src/aa_device_spectral_whitening.cu:			LOG(log_level::error, "Could not cudaMemcpy in aa_device_spectral_whitening.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_spectral_whitening.cu:		e = cudaMemcpy(h_segmented_MSD, d_MSD_segments, nDMs*nSegments*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_spectral_whitening.cu:				sprintf(str, "PSR_fft_GPU_medians_%d.dat", d);
src/aa_device_spectral_whitening.cu:				sprintf(str, "PSR_fft_GPU_means_%d.dat", d);
src/aa_device_spectral_whitening.cu:		cudaFree(d_segment_sizes);
src/aa_device_spectral_whitening.cu:		cudaFree(d_MSD_segments);
src/aa_device_harmonic_summing_kernel.cu:#include <cuda.h>
src/aa_device_harmonic_summing_kernel.cu:#include <cuda_runtime.h>
src/aa_device_harmonic_summing_kernel.cu:__global__ void simple_harmonic_sum_GPU_kernel(float const* __restrict__ d_input, float *d_output_SNR, ushort *d_output_harmonics, float *d_MSD, int nTimesamples, int nSpectra, int nHarmonics){
src/aa_device_harmonic_summing_kernel.cu:__global__ void greedy_harmonic_sum_GPU_kernel(float *d_maxSNR, ushort *d_maxHarmonics, float const* __restrict__ d_input, float const* __restrict__ d_MSD, int nTimesamples, int nDMs, int nHarmonics){
src/aa_device_harmonic_summing_kernel.cu:__global__ void presto_plus_harmonic_sum_GPU_kernel(float *d_maxSNR, ushort *d_maxHarmonics, float const* __restrict__ d_input, float const* __restrict__ d_MSD, int nTimesamples, int nDMs, int nHarmonics){
src/aa_device_harmonic_summing_kernel.cu:__global__ void presto_harmonic_sum_GPU_kernel(float *d_maxSNR, ushort *d_maxHarmonics, float const* __restrict__ d_input, float const* __restrict__ d_MSD, int nTimesamples, int nDMs, int nHarmonicsFactor){
src/aa_device_harmonic_summing_kernel.cu:  /** \brief Kernel wrapper function for simple_harmonic_sum_GPU_kernel kernel function. */
src/aa_device_harmonic_summing_kernel.cu:  void call_kernel_simple_harmonic_sum_GPU_kernel(
src/aa_device_harmonic_summing_kernel.cu:    simple_harmonic_sum_GPU_kernel<<<grid_size, block_size>>>(d_input, d_output_SNR, d_output_harmonics, d_MSD, nTimesamples, nSpectra, nHarmonics);
src/aa_device_harmonic_summing_kernel.cu:  /** \brief Kernel wrapper function for call_kernel_greedy_harmonic_sum_GPU_kernel kernel function. */
src/aa_device_harmonic_summing_kernel.cu:  void call_kernel_greedy_harmonic_sum_GPU_kernel(
src/aa_device_harmonic_summing_kernel.cu:      greedy_harmonic_sum_GPU_kernel<HRMS_remove_scalloping_loss><<<grid_size, block_size>>>(
src/aa_device_harmonic_summing_kernel.cu:      greedy_harmonic_sum_GPU_kernel<HRMS_normal><<<grid_size, block_size>>>(
src/aa_device_harmonic_summing_kernel.cu:  /** \brief Kernel wrapper function for presto_harmonic_sum_GPU_kernel kernel function. */
src/aa_device_harmonic_summing_kernel.cu:  void call_kernel_presto_plus_harmonic_sum_GPU_kernel(
src/aa_device_harmonic_summing_kernel.cu:      presto_plus_harmonic_sum_GPU_kernel<HRMS_remove_scalloping_loss><<<grid_size, block_size>>>(
src/aa_device_harmonic_summing_kernel.cu:      presto_plus_harmonic_sum_GPU_kernel<HRMS_normal><<<grid_size, block_size>>>(
src/aa_device_harmonic_summing_kernel.cu:  /** \brief Kernel wrapper function for presto_harmonic_sum_GPU_kernel kernel function. */
src/aa_device_harmonic_summing_kernel.cu:  void call_kernel_presto_harmonic_sum_GPU_kernel(
src/aa_device_harmonic_summing_kernel.cu:      presto_harmonic_sum_GPU_kernel<HRMS_remove_scalloping_loss><<<grid_size, block_size>>>(
src/aa_device_harmonic_summing_kernel.cu:      presto_harmonic_sum_GPU_kernel<HRMS_normal><<<grid_size, block_size>>>(
src/aa_corner_turn.cpp:#include "aa_gpu_timer.hpp"
src/aa_corner_turn.cpp:    //{{{ Simple corner turn on the GPU
src/aa_corner_turn.cpp:    aa_gpu_timer timer;
src/aa_corner_turn.cpp:    cudaDeviceSynchronize();
src/aa_corner_turn.cpp:    cudaDeviceSynchronize();
src/aa_corner_turn.cpp:    printf("\nPerformed CT: %lf (GPU estimate)", time/1000.0);
src/aa_corner_turn.cpp:    //cudaMemcpy(d_input, d_output, inputsize, cudaMemcpyDeviceToDevice);
src/aa_corner_turn.cpp:    cudaDeviceSynchronize();
src/aa_corner_turn.cpp:    //---------> CUDA block and CUDA grid parameters
src/aa_corner_turn.cpp:    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
src/aa_corner_turn.cpp:    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
src/aa_device_SNR_limited_kernel.cu:  __global__ void SNR_GPU_limited(float *d_FIR_input, float *d_SNR_output, ushort *d_SNR_taps, float *d_MSD, int x_steps, int nTaps, int nColumns, int offset) {
src/aa_device_SNR_limited_kernel.cu:  /** \brief Kernel wrapper function to SNR_GPU_limited kernel function. */
src/aa_device_SNR_limited_kernel.cu:  void call_kernel_SNR_GPU_limited(const dim3 &grid_size, const dim3 &block_size, float *const d_FIR_input, float *const d_SNR_output,
src/aa_device_SNR_limited_kernel.cu:    SNR_GPU_limited<<<grid_size, block_size>>>(d_FIR_input, d_SNR_output, d_SNR_taps, d_MSD,
src/aa_device_stats.cu:  void stats_gpu(cudaEvent_t event, cudaStream_t stream, int samps, float *mean, float *stddev, float *h_signal_power, float *d_signal_power) {
src/aa_device_stats.cu:    cudaError_t e = cudaMalloc((void** )&d_sum, size * sizeof(float));
src/aa_device_stats.cu:    if(e != cudaSuccess) {
src/aa_device_stats.cu:      LOG(log_level::error, "Could not cudaMalloc in aa_device_stats.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_stats.cu:    e = cudaMalloc((void** )&d_sum_square, size * sizeof(float));
src/aa_device_stats.cu:    if(e != cudaSuccess) {
src/aa_device_stats.cu:      LOG(log_level::error, "Could not cudaMalloc in aa_device_stats.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_stats.cu:    e = cudaMallocHost((void** )&h_sum, size * sizeof(float));
src/aa_device_stats.cu:    if(e != cudaSuccess) {
src/aa_device_stats.cu:      LOG(log_level::error, "Could not cudaMallocHost in aa_device_stats.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_stats.cu:    e = cudaMallocHost((void** )&h_sum_square, size * sizeof(float));
src/aa_device_stats.cu:    if(e != cudaSuccess) {
src/aa_device_stats.cu:      LOG(log_level::error, "Could not cudaMallocHost in aa_device_stats.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_stats.cu:    cudaStreamWaitEvent(stream, event, 0);
src/aa_device_stats.cu:    //getLastCudaError("power_kernel failed");
src/aa_device_stats.cu:    cudaEventRecord(event, stream);
src/aa_device_stats.cu:    cudaStreamWaitEvent(stream, event, 0);
src/aa_device_stats.cu:    e = cudaMemcpyAsync(h_sum, d_sum, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
src/aa_device_stats.cu:    if(e != cudaSuccess) {
src/aa_device_stats.cu:      LOG(log_level::error, "Could not cudaMemcpyAsync in aa_device_stats.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_stats.cu:    e = cudaMemcpyAsync(h_sum_square, d_sum_square, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
src/aa_device_stats.cu:    if(e != cudaSuccess) {
src/aa_device_stats.cu:      LOG(log_level::error, "Could not cudaMemcpyAsync in aa_device_stats.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_stats.cu:    cudaEventRecord(event, stream);
src/aa_device_stats.cu:    cudaStreamSynchronize(stream);
src/aa_device_stats.cu:    cudaFree(d_sum);
src/aa_device_stats.cu:    cudaFree(d_sum_square);
src/aa_device_stats.cu:    cudaFreeHost(h_sum);
src/aa_device_stats.cu:    cudaFreeHost(h_sum_square);
src/aa_device_MSD_outlier_rejection_kernel.cu:#include <cuda.h>
src/aa_device_MSD_outlier_rejection_kernel.cu:#include <cuda_runtime.h>
src/aa_device_MSD_outlier_rejection_kernel.cu:#include "aa_device_cuda_deprecated_wrappers.cuh"
src/aa_device_MSD_outlier_rejection_kernel.cu:	__global__ void MSD_GPU_calculate_partials_2d_and_minmax(float const* __restrict__ d_input, float *d_output, int y_steps, int nTimesamples, int offset) {
src/aa_device_MSD_outlier_rejection_kernel.cu:	void call_kernel_MSD_GPU_calculate_partials_2d_and_minmax(const dim3 &grid_size, const dim3 &block_size, float const *const d_input, float *const d_output, const int &y_steps, const int &nTimesamples, const int &offset) {
src/aa_device_MSD_outlier_rejection_kernel.cu:		MSD_GPU_calculate_partials_2d_and_minmax<<<grid_size, block_size>>>(d_input, d_output, y_steps, nTimesamples, offset);
src/aa_device_power_kernel.cu:#include <cuda.h>
src/aa_device_power_kernel.cu:#include <cuda_runtime.h>
src/aa_device_power_kernel.cu:  __global__ void GPU_simple_power_and_interbin_kernel(float2 *d_input_complex, float *d_output_power, float *d_output_interbinning, int nTimesamples, float norm){
src/aa_device_power_kernel.cu:  void call_kernel_power_kernel(const dim3 &block_size, const dim3 &grid_size, const int &smem_bytes, const cudaStream_t &stream,
src/aa_device_power_kernel.cu:  /** Kernel wrapper function for GPU_simple_power_and_interbin_kernel kernel function. */
src/aa_device_power_kernel.cu:  void call_kernel_GPU_simple_power_and_interbin_kernel(const dim3 &grid_size, const dim3 &block_size,
src/aa_device_power_kernel.cu:    GPU_simple_power_and_interbin_kernel<<<grid_size, block_size>>>(d_input_complex, d_output_power, d_output_interbinning, nTimesamples, norm);
src/presto_funcs.cpp:#include <cuda.h>
src/presto_funcs.cpp:#include <cuda_runtime.h>
src/presto_funcs.cpp:#include <cuda_runtime_api.h>
src/presto_funcs.cpp:	cudaError_t cudaError;
src/presto_funcs.cpp:		cudaMalloc((void **) &d_input, data_size);
src/presto_funcs.cpp:		cudaMalloc((void **) &d_output, data_fft_size);
src/presto_funcs.cpp:		cudaError = cudaMemcpy(d_input, data, data_size, cudaMemcpyHostToDevice);
src/presto_funcs.cpp:	    if(cudaError != cudaSuccess) printf("Could not cudaMemcpy in presto_func.cpp");
src/presto_funcs.cpp:		cudaError = cudaMemcpy(data_fft, d_output, data_fft_size, cudaMemcpyDeviceToHost);
src/presto_funcs.cpp:	    if(cudaError != cudaSuccess) printf("Could not cudaMemcpy in presto_func.cpp");
src/presto_funcs.cpp:		cudaFree(d_input);
src/presto_funcs.cpp:		cudaFree(d_output);
src/aa_device_dedispersion_kernel.cu:    cudaMemcpyToSymbol(dm_shifts, dmshifts, nchans * sizeof(float));
src/aa_device_dedispersion_kernel.cu:    cudaMemcpyToSymbol(i_nchans, &nchans, sizeof(int));
src/aa_device_dedispersion_kernel.cu:    cudaMemcpyToSymbol(i_nsamp, &length, sizeof(int));
src/aa_device_dedispersion_kernel.cu:    cudaMemcpyToSymbol(i_t_processed_s, &t_processed, sizeof(int));
src/aa_device_dedispersion_kernel.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_dedispersion_kernel.cu:    cudaMemcpyToSymbol(i_nsamp, &length, sizeof(int));
src/aa_device_dedispersion_kernel.cu:    cudaMemcpyToSymbol(i_t_processed_s, &t_processed, sizeof(int));
src/aa_device_dedispersion_kernel.cu:    cudaMemcpyToSymbol(i_nchans, &nchans, sizeof(int));
src/aa_device_dedispersion_kernel.cu:    cudaMemcpyToSymbol(i_nsamp, &length, sizeof(int));
src/aa_device_dedispersion_kernel.cu:    cudaMemcpyToSymbol(i_t_processed_s, &t_processed, sizeof(int));
src/aa_device_dedispersion_kernel.cu:    cudaFuncSetCacheConfig(shared_dedisperse_kernel, cudaFuncCachePreferShared);
src/aa_device_dedispersion_kernel.cu:	/** \brief Kernel wrapper function for dedispersion GPU kernel which works with 4-bit input data */
src/aa_device_dedispersion_kernel.cu:		cudaFuncSetCacheConfig(shared_dedisperse_kernel_4bit, cudaFuncCachePreferShared);
src/aa_device_dedispersion_kernel.cu:        /** \brief Kernel wrapper function for dedispersion GPU kernel which works with 4-bit input data */
src/aa_device_dedispersion_kernel.cu:                cudaFuncSetCacheConfig(shared_dedisperse_kernel_4bit_4096p, cudaFuncCachePreferShared);
src/aa_device_dedispersion_kernel.cu:	/** \brief Kernel wrapper function for dedispersion GPU kernel which works with number of channels greater than 8192. */
src/aa_device_dedispersion_kernel.cu:		cudaFuncSetCacheConfig(shared_dedisperse_kernel_nchan8192p, cudaFuncCachePreferShared);
src/aa_device_dedispersion_kernel.cu:    cudaFuncSetCacheConfig(shared_dedisperse_kernel_16, cudaFuncCachePreferShared);
src/aa_device_dedispersion_kernel.cu:		cudaFuncSetCacheConfig(shared_dedisperse_kernel_16_nchan8192p, cudaFuncCachePreferShared);
src/aa_device_dedispersion_kernel.cu:    cudaFuncSetCacheConfig(cache_dedisperse_kernel, cudaFuncCachePreferL1);
src/aa_device_dedispersion_kernel.cu:		cudaFuncSetCacheConfig(cache_dedisperse_kernel_nchan8192p, cudaFuncCachePreferL1);
src/aa_device_binning_kernel.cu:#include <cuda.h>
src/aa_device_binning_kernel.cu:#include <cuda_runtime.h>
src/aa_device_binning_kernel.cu:  __global__ void DiT_GPU_v2(float const* __restrict__ d_input, float *d_output, unsigned int nDMs, unsigned int nTimesamples, unsigned int dts) {
src/aa_device_binning_kernel.cu:  /** \brief Kernel wrapper function for DiT_GPU_v2 kernel function. */
src/aa_device_binning_kernel.cu:  void call_kernel_DiT_GPU_v2(const dim3 &gridSize, const dim3 &blockSize, float const *const d_input, float *const d_output, const unsigned int &nDMs, const unsigned int &nTimesamples, const unsigned int &dts) {
src/aa_device_binning_kernel.cu:    DiT_GPU_v2<<<gridSize,blockSize>>>(d_input, d_output, nDMs, nTimesamples, (nTimesamples>>1));
src/aa_zero_dm_outliers.cpp:	cudaStream_t stream = NULL;
src/aa_zero_dm_outliers.cpp:	cudaError_t e;
src/aa_zero_dm_outliers.cpp:	e = cudaMalloc((void **)d_normalization_factor, nchans*sizeof(float));
src/aa_zero_dm_outliers.cpp:	if (e != cudaSuccess) {
src/aa_zero_dm_outliers.cpp:		LOG(log_level::error, "Could not allocate memory for d_normalization_factor (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_zero_dm_outliers.cpp:	cudaMemcpy(d_normalization_factor, local_bandpass_normalization.data(), nchans*sizeof(float), cudaMemcpyHostToDevice);
src/aa_zero_dm_outliers.cpp:	e =cudaFree(d_normalization_factor);
src/aa_zero_dm_outliers.cpp:	if (e != cudaSuccess) {
src/aa_zero_dm_outliers.cpp:		LOG(log_level::error, "Cannot free d_normalization_factor memory: (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_zero_dm_outliers.cpp:	cudaStream_t stream = NULL;
src/aa_zero_dm_outliers.cpp:	cudaDeviceSynchronize();
src/aa_zero_dm_outliers.cpp:	printf("\nPerformed ZDM: %lf (GPU estimate)", time);
src/aa_device_peak_find.cu:      //---------> Nvidia stuff
src/aa_device_peak_find.cu:      cudaDeviceProp deviceProp;
src/aa_device_peak_find.cu:      cudaGetDeviceProperties(&deviceProp, CARD);
src/aa_device_peak_find.cu:      // launching GPU kernels
src/aa_device_zero_dm_outliers_kernel.cu:#include <cuda.h>
src/aa_device_zero_dm_outliers_kernel.cu:#include <cuda_runtime.h>
src/aa_device_zero_dm_outliers_kernel.cu:#include "aa_device_cuda_deprecated_wrappers.cuh"
src/aa_device_zero_dm_outliers_kernel.cu:	const cudaStream_t &stream, 
src/aa_device_zero_dm_outliers_kernel.cu:			cudaDeviceSynchronize();
src/aa_device_harmonic_summing.cu:    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
src/aa_device_harmonic_summing.cu:    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
src/aa_device_harmonic_summing.cu:    call_kernel_simple_harmonic_sum_GPU_kernel(gridSize, blockSize, d_input, d_output_SNR, d_output_harmonics, d_MSD, nTimesamples, nDMs, nHarmonics);
src/aa_device_harmonic_summing.cu:    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
src/aa_device_harmonic_summing.cu:    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
src/aa_device_harmonic_summing.cu:    call_kernel_greedy_harmonic_sum_GPU_kernel(
src/aa_device_harmonic_summing.cu:    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
src/aa_device_harmonic_summing.cu:    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
src/aa_device_harmonic_summing.cu:    call_kernel_presto_plus_harmonic_sum_GPU_kernel(
src/aa_device_harmonic_summing.cu:    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
src/aa_device_harmonic_summing.cu:    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
src/aa_device_harmonic_summing.cu:    call_kernel_presto_harmonic_sum_GPU_kernel(
src/aa_device_stretch_kernel.cu:#include <cuda.h>
src/aa_device_stretch_kernel.cu:#include <cuda_runtime.h>
src/aa_device_stretch_kernel.cu:  void call_kernel_stretch_kernel(const dim3 &block_size, const dim3 &grid_size, const int &smem_bytes, const cudaStream_t &stream,
src/aa_fdas_host.cu:    printf("\nThread block size in x direction for 2-D thread block convolution GPU kernels : TBSIZEX %d\n", TBSIZEX);
src/aa_fdas_host.cu:    printf("\nThread block size in Y direction for 2-D thread block convolution GPU kernels : TBSIZEY %d\n", TBSIZEY);
src/aa_fdas_host.cu:    printf("\nThread block size in x direction for 2-D thread block power spectrum GPU kernels : PTBSIZEX %d\n", PTBSIZEX);
src/aa_fdas_host.cu:    printf("\nThread block size in y direction for 2-D thread block power spectrum GPU kernels : PTBSIZEY %d\n", PTBSIZEY);
src/aa_fdas_host.cu:  /** \brief Check CUDA devices. */
src/aa_fdas_host.cu:  void fdas_cuda_check_devices(int devid) {
src/aa_fdas_host.cu:    cudaError_t e = cudaGetDeviceCount(&devcount);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaGetDeviceCount in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    printf("\nDetected %d CUDA Capable device(s)\n", devcount);
src/aa_fdas_host.cu:  /** \brief Allocate GPU arrays for fdas. */
src/aa_fdas_host.cu:  void fdas_alloc_gpu_arrays(fdas_gpuarrays *arrays,  cmd_args *cmdargs)
src/aa_fdas_host.cu:    printf("\nAllocating gpu arrays:\n"); 
src/aa_fdas_host.cu:    // Memory allocations for gpu real fft input / output signal
src/aa_fdas_host.cu:    cudaError_t e = cudaMalloc((void**)&arrays->d_in_signal, arrays->mem_insig);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMalloc in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    e = cudaMalloc((void**)&arrays->d_fft_signal, arrays->mem_rfft);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMalloc in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    e = cudaMalloc((void**)&arrays->d_ext_data, arrays->mem_extsig);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMalloc in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    e = cudaMalloc((void**)&arrays->d_kernel, KERNLEN*sizeof(float2)*NKERN );
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMalloc in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    e = cudaMalloc((void**)&arrays->d_ffdot_pwr, arrays->mem_ffdot );
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMalloc in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    e = cudaMemset(arrays->d_ffdot_pwr, 0, arrays->mem_ffdot);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMemset in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:      e = cudaMalloc(&arrays->d_ffdot_cpx, arrays->mem_ffdot_cpx);
src/aa_fdas_host.cu:      if(e != cudaSuccess) {
src/aa_fdas_host.cu:	LOG(log_level::error, "Could not cudaMalloc in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:      e = cudaMalloc(&arrays->ip_edge_points, arrays->mem_ipedge);
src/aa_fdas_host.cu:      if(e != cudaSuccess) {
src/aa_fdas_host.cu:	LOG(log_level::error, "Could not cudaMalloc in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    if ( cudaSuccess != cudaMalloc((void**) &arrays->d_fdas_peak_list, arrays->mem_max_list_size)) printf("Allocation error in FDAS: d_fdas_peak_list\n");
src/aa_fdas_host.cu:    e = cudaMemGetInfo ( &mfree, &mtotal );
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMemGetInfo in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:  /** \brief Free GPU arrays for fdas. */
src/aa_fdas_host.cu:  void fdas_free_gpu_arrays(fdas_gpuarrays *arrays,  cmd_args *cmdargs)
src/aa_fdas_host.cu:    cudaError_t e = cudaFree(arrays->d_in_signal);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaFree in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    e = cudaFree(arrays->d_fft_signal);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaFree in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    e = cudaFree(arrays->d_ext_data);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaFree in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    e = cudaFree(arrays->d_ffdot_pwr);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaFree in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    e = cudaFree(arrays->d_kernel);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaFree in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:      e = cudaFree(arrays->d_ffdot_cpx);
src/aa_fdas_host.cu:      if(e != cudaSuccess) {
src/aa_fdas_host.cu:	LOG(log_level::error, "Could not cudaFree in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:      e = cudaFree(arrays->ip_edge_points);
src/aa_fdas_host.cu:      if(e != cudaSuccess) {
src/aa_fdas_host.cu:	LOG(log_level::error, "Could not cudaFree in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    cudaFree(arrays->d_fdas_peak_list);
src/aa_fdas_host.cu:   * \brief Create kernel templates for the correlation technique (Ransom et. al. 2002), and upload + FFT to GPU memory.
src/aa_fdas_host.cu:    cudaError_t e = cudaMemcpy( d_kernel, h_kernel, KERNLEN*sizeof(float2)* NKERN, cudaMemcpyHostToDevice); // upload kernels to GPU
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:  /** \brief Create CUDA cufft fftplans for FDAS. */
src/aa_fdas_host.cu:  void fdas_cuda_create_fftplans(fdas_cufftplan *fftplans, fdas_params *params) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaEstimateMany in aa_fdas_host.cu");
src/aa_fdas_host.cu:    cudaDeviceSynchronize();
src/aa_fdas_host.cu:    //getLastCudaError("\nCuda Error real fft plan\n");
src/aa_fdas_host.cu:    cudaDeviceSynchronize();
src/aa_fdas_host.cu:    //getLastCudaError("\nCuda Error forward fft plan\n");
src/aa_fdas_host.cu:  void fdas_cuda_basic(fdas_cufftplan *fftplans, fdas_gpuarrays *gpuarrays, cmd_args *cmdargs, fdas_params *params)
src/aa_fdas_host.cu:    /* Basic GPU fdas algorithm using cuFFT */
src/aa_fdas_host.cu:    cufftExecR2C(fftplans->realplan, gpuarrays->d_in_signal, gpuarrays->d_fft_signal);
src/aa_fdas_host.cu:    cudaError_t e = cudaMemcpy(ftemp, gpuarrays->d_in_signal, (params->rfftlen)*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    e = cudaMemcpy(gpuarrays->d_fft_signal, f2temp, (params->rfftlen)*sizeof(float2), cudaMemcpyHostToDevice);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:      // TODO: replace with GPU version
src/aa_fdas_host.cu:      cudaError_t e = cudaMemcpy(fftsig, gpuarrays->d_fft_signal, (params->rfftlen)*sizeof(float2), cudaMemcpyDeviceToHost);
src/aa_fdas_host.cu:      if(e != cudaSuccess) {
src/aa_fdas_host.cu:	LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:      e = cudaMemcpy(gpuarrays->d_fft_signal, fftsig, (params->rfftlen)*sizeof(float2), cudaMemcpyHostToDevice);
src/aa_fdas_host.cu:      if(e != cudaSuccess) {
src/aa_fdas_host.cu:	LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    call_kernel_cuda_overlap_copy(gpuarrays->d_ext_data, gpuarrays->d_fft_signal, params->sigblock, params->rfftlen, params->extlen, params->offset, params->nblocks );
src/aa_fdas_host.cu:      // TODO: replace with GPU version
src/aa_fdas_host.cu:      cudaError_t e = cudaMemcpy(extsig, gpuarrays->d_ext_data, (params->extlen)*sizeof(float2), cudaMemcpyDeviceToHost);
src/aa_fdas_host.cu:      if(e != cudaSuccess) {
src/aa_fdas_host.cu:	LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:      e = cudaMemcpy(gpuarrays->d_ext_data, extsig, (params->extlen)*sizeof(float2), cudaMemcpyHostToDevice);
src/aa_fdas_host.cu:      if(e != cudaSuccess) {
src/aa_fdas_host.cu:	LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    cufftExecC2C(fftplans->forwardplan, gpuarrays->d_ext_data, gpuarrays->d_ext_data, CUFFT_FORWARD);
src/aa_fdas_host.cu:    call_kernel_cuda_convolve_reg_1d_halftemps(cblocks, cthreads, gpuarrays->d_kernel, gpuarrays->d_ext_data, gpuarrays->d_ffdot_cpx, params->extlen, params->scale);
src/aa_fdas_host.cu:      cufftExecC2C(fftplans->forwardplan, gpuarrays->d_ffdot_cpx + k * params->extlen, gpuarrays->d_ffdot_cpx + k *params->extlen, CUFFT_INVERSE);
src/aa_fdas_host.cu:      cufftExecC2C(fftplans->forwardplan, gpuarrays->d_ffdot_cpx + (NKERN-1-k) * params->extlen, gpuarrays->d_ffdot_cpx + (NKERN-1-k) *params->extlen, CUFFT_INVERSE);
src/aa_fdas_host.cu:    cufftExecC2C(fftplans->forwardplan, gpuarrays->d_ffdot_cpx + (nTemplates * params->extlen), gpuarrays->d_ffdot_cpx + (nTemplates * params->extlen), CUFFT_INVERSE);
src/aa_fdas_host.cu:      call_kernel_cuda_ffdotpow_concat_2d_inbin(pwblocks, pwthreads, gpuarrays->d_ffdot_cpx, gpuarrays->d_ffdot_pwr, params->sigblock, params->offset, params->nblocks, params->extlen, params->siglen);
src/aa_fdas_host.cu:      call_kernel_cuda_ffdotpow_concat_2d(pwblocks, pwthreads, gpuarrays->d_ffdot_cpx, gpuarrays->d_ffdot_pwr, params->sigblock, params->offset, params->nblocks, params->extlen, params->siglen);
src/aa_fdas_host.cu:  void fdas_cuda_customfft(fdas_cufftplan *fftplans, fdas_gpuarrays *gpuarrays, cmd_args *cmdargs, fdas_params *params) {
src/aa_fdas_host.cu:    cufftExecR2C(fftplans->realplan, gpuarrays->d_in_signal, gpuarrays->d_fft_signal);
src/aa_fdas_host.cu:    cudaError_t e = cudaMemcpy(ftemp, gpuarrays->d_in_signal, (params->rfftlen)*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    e = cudaMemcpy(gpuarrays->d_fft_signal, f2temp, (params->rfftlen)*sizeof(float2), cudaMemcpyHostToDevice);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:      // TODO: replace with GPU version
src/aa_fdas_host.cu:      cudaError_t e = cudaMemcpy(fftsig, gpuarrays->d_fft_signal, (params->rfftlen)*sizeof(float2), cudaMemcpyDeviceToHost);
src/aa_fdas_host.cu:      if(e != cudaSuccess) {
src/aa_fdas_host.cu:	LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:      e = cudaMemcpy(gpuarrays->d_fft_signal, fftsig, (params->rfftlen)*sizeof(float2), cudaMemcpyHostToDevice);
src/aa_fdas_host.cu:      if(e != cudaSuccess) {
src/aa_fdas_host.cu:	LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:    call_kernel_cuda_overlap_copy_smallblk(params->nblocks, gpuarrays->d_ext_data, gpuarrays->d_fft_signal, params->sigblock, params->rfftlen, params->extlen, params->offset, params->nblocks );
src/aa_fdas_host.cu:      // TODO: replace with GPU version
src/aa_fdas_host.cu:      cudaError_t e = cudaMemcpy(extsig, gpuarrays->d_ext_data, (params->extlen)*sizeof(float2), cudaMemcpyDeviceToHost);
src/aa_fdas_host.cu:      if(e != cudaSuccess) {
src/aa_fdas_host.cu:	LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:      e = cudaMemcpy(gpuarrays->d_ext_data, extsig, (params->extlen)*sizeof(float2), cudaMemcpyHostToDevice);
src/aa_fdas_host.cu:      if(e != cudaSuccess) {
src/aa_fdas_host.cu:	LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:      call_kernel_cuda_convolve_customfft_wes_no_reorder02_inbin(params->nblocks, gpuarrays->d_kernel, gpuarrays->d_ext_data, gpuarrays->d_ffdot_pwr, params->sigblock, params->extlen, params->siglen, params->offset, params->scale, gpuarrays->ip_edge_points);
src/aa_fdas_host.cu:      //cuda_convolve_customfft_wes_no_reorder02<<< params->nblocks, KERNLEN >>>( gpuarrays->d_kernel, gpuarrays->d_ext_data, gpuarrays->d_ffdot_pwr, params->sigblock, params->extlen, params->siglen, params->offset, params->scale);
src/aa_fdas_host.cu:      GPU_CONV_kFFT_mk11_2elem_2v<<<gridSize,blockSize>>>(gpuarrays->d_ext_data, gpuarrays->d_ffdot_pwr, gpuarrays->d_kernel, params->sigblock, params->offset, params->nblocks, params->scale);
src/aa_fdas_host.cu:      call_kernel_GPU_CONV_kFFT_mk11_4elem_2v(gridSize,blockSize, gpuarrays->d_ext_data, gpuarrays->d_ffdot_pwr, gpuarrays->d_kernel, params->sigblock, params->offset, params->nblocks, params->scale);
src/aa_fdas_host.cu:  void fdas_write_list(fdas_gpuarrays *gpuarrays, cmd_args *cmdargs, fdas_params *params, float *h_MSD, float dm_low, int dm_count, float dm_step, unsigned int list_size){
src/aa_fdas_host.cu:      cudaError_t e = cudaMemcpy(h_fdas_peak_list, gpuarrays->d_fdas_peak_list, list_size*4*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_fdas_host.cu:      if(e != cudaSuccess) {
src/aa_fdas_host.cu:	LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:  void fdas_write_ffdot(fdas_gpuarrays *gpuarrays, cmd_args *cmdargs, fdas_params *params, float dm_low, int dm_count, float dm_step ) {
src/aa_fdas_host.cu:    cudaError_t e = cudaMemcpy(h_ffdotpwr, gpuarrays->d_ffdot_pwr, params->ffdotlen*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_fdas_host.cu:  void fdas_write_test_ffdot(fdas_gpuarrays *gpuarrays, cmd_args *cmdargs, fdas_params *params, float dm_low, int dm_count, float dm_step ) {
src/aa_fdas_host.cu:    cudaError_t e = cudaMemcpy(h_ffdotpwr, gpuarrays->d_ffdot_pwr, params->ffdotlen*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_fdas_host.cu:    if(e != cudaSuccess) {
src/aa_fdas_host.cu:      LOG(log_level::error, "Could not cudaMemcpy in aa_fdas_host.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_host_help.cpp:	printf("\n The code has algorithms for both CPU and NVIDIA GPU acceleration (Fermi and Kepler).");
src/aa_host_help.cpp:	printf("\n To use GPU acceleration with no limitations on the maximum DM use the flag \"-algorithm GPU-CACHE\" (about 3x slower than smem algorithm).");
src/aa_host_help.cpp:	printf("\n\t\t\t\t\t\t This can be CPU GPU GPU-CACHE.");
src/aa_host_help.cpp:	printf("\n\t\t\t\t\t\t GPU is the default and will use the GPU shared memory. This is our fastest algorithm");
src/aa_host_help.cpp:	printf("\n\t\t\t\t\t\t GPU-CACHE will excecute on the gpu and use the gpu cache. This is much faster than CPU but about 3x slower than GPU.");
src/aa_host_statistics.cu:  void statistics(char *string, int i, cudaStream_t stream, clock_t *in_time, clock_t *out_time, int maxshift, int total_ndms, int nchans, int nsamp, float tsamp, float *dm_low, float *dm_high, float *dm_step, int *ndms)
src/aa_host_statistics.cu:	printf("\nPerformed Brute-Force Dedispersion: %f (GPU estimate)", time);
src/aa_device_periods.cu:#include "aa_gpu_timer.hpp"
src/aa_device_periods.cu:  //#define GPU_PERIODICITY_SEARCH_DEBUG
src/aa_device_periods.cu:  void checkCudaErrors( cudaError_t CUDA_error){
src/aa_device_periods.cu:    if(CUDA_error != cudaSuccess) {
src/aa_device_periods.cu:      printf("CUDA error: %d\n", CUDA_error);
src/aa_device_periods.cu:    cudaMemGetInfo(&free, &total);
src/aa_device_periods.cu:   * \class GPU_Memory_for_Periodicity_Search aa_device_periods.cu "src/aa_device_periods.cu"
src/aa_device_periods.cu:   * \brief Class for managing GPU memory for periodicity search.
src/aa_device_periods.cu:  class GPU_Memory_for_Periodicity_Search {
src/aa_device_periods.cu:      if ( cudaSuccess != cudaMalloc((void **) &d_one_A,  sizeof(float)*t_input_plane_size )) printf("Periodicity Allocation error! d_one_A\n");
src/aa_device_periods.cu:      if ( cudaSuccess != cudaMalloc((void **) &d_two_B,  sizeof(float)*2*t_input_plane_size )) printf("Periodicity Allocation error! d_two_B\n");
src/aa_device_periods.cu:      if ( cudaSuccess != cudaMalloc((void **) &d_half_C,  sizeof(float)*t_input_plane_size/2 )) printf("Periodicity Allocation error! d_spectra_Real\n");
src/aa_device_periods.cu:      if ( cudaSuccess != cudaMalloc((void **) &d_power_harmonics, sizeof(ushort)*t_input_plane_size )) printf("Periodicity Allocation error! d_harmonics\n");
src/aa_device_periods.cu:      if ( cudaSuccess != cudaMalloc((void **) &d_interbin_harmonics, sizeof(ushort)*t_input_plane_size )) printf("Periodicity Allocation error! d_harmonics\n");
src/aa_device_periods.cu:      if ( cudaSuccess != cudaMalloc((void**) &gmem_power_peak_pos, 1*sizeof(int)) )  printf("Periodicity Allocation error! gmem_power_peak_pos\n");
src/aa_device_periods.cu:      if ( cudaSuccess != cudaMalloc((void**) &gmem_interbin_peak_pos, 1*sizeof(int)) )  printf("Periodicity Allocation error! gmem_interbin_peak_pos\n");
src/aa_device_periods.cu:      if ( cudaSuccess != cudaMalloc((void**) &d_MSD, sizeof(float)*MSD_interpolated_size*2)) {printf("Periodicity Allocation error! d_MSD\n");}
src/aa_device_periods.cu:      if ( cudaSuccess != cudaMalloc((void**) &d_previous_partials, sizeof(float)*MSD_DIT_size*MSD_PARTIAL_SIZE)) {printf("Periodicity Allocation error! d_previous_partials\n");}
src/aa_device_periods.cu:      if ( cudaSuccess != cudaMalloc((void**) &d_all_blocks, sizeof(float)*PSR_strategy.max_total_MSD_blocks()*MSD_PARTIAL_SIZE)) {printf("Periodicity Allocation error! d_MSD\n");}
src/aa_device_periods.cu:      if ( cudaSuccess != cudaMalloc((void **) &cuFFT_workarea, PSR_strategy.cuFFT_workarea_size()) ) {printf("Periodicity Allocation error! cuFFT_workarea\n");}
src/aa_device_periods.cu:      cudaMemset(d_MSD, 0, MSD_interpolated_size*2*sizeof(float));
src/aa_device_periods.cu:      cudaMemset(d_previous_partials, 0, MSD_DIT_size*MSD_PARTIAL_SIZE*sizeof(float));
src/aa_device_periods.cu:      cudaMemset(gmem_power_peak_pos, 0, sizeof(int));
src/aa_device_periods.cu:      cudaMemset(gmem_interbin_peak_pos, 0, sizeof(int));
src/aa_device_periods.cu:      cudaError_t e = cudaMemcpy(&temp, gmem_power_peak_pos, sizeof(int), cudaMemcpyDeviceToHost);
src/aa_device_periods.cu:      if(e != cudaSuccess) {
src/aa_device_periods.cu:        LOG(log_level::error, "Could not cudaMemcpy in aa_device_periods.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_periods.cu:      cudaError_t e = cudaMemcpy(&temp, gmem_interbin_peak_pos, sizeof(int), cudaMemcpyDeviceToHost);
src/aa_device_periods.cu:      if(e != cudaSuccess) {
src/aa_device_periods.cu:        LOG(log_level::error, "Could not cudaMemcpy in aa_device_periods.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_periods.cu:      cudaError_t e = cudaMemcpy(h_MSD, d_MSD, MSD_interpolated_size*2*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_periods.cu:      if(e != cudaSuccess) {
src/aa_device_periods.cu:        LOG(log_level::error, "Could not cudaMemcpy in aa_device_periods.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_periods.cu:      cudaError_t e = cudaMemcpy(h_MSD_partials, d_previous_partials, MSD_DIT_size*MSD_PARTIAL_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_periods.cu:      if(e != cudaSuccess) {
src/aa_device_periods.cu:        LOG(log_level::error, "Could not cudaMemcpy in aa_device_periods.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_periods.cu:      cudaError_t e = cudaMemcpy(d_previous_partials, h_MSD_partials, MSD_PARTIAL_SIZE*sizeof(float), cudaMemcpyHostToDevice);
src/aa_device_periods.cu:      if(e != cudaSuccess) {
src/aa_device_periods.cu:        LOG(log_level::error, "Could not cudaMemcpy in aa_device_periods.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_periods.cu:    /** \brief Destructor for GPU_Memory_for_Periodicity_Search. */
src/aa_device_periods.cu:    ~GPU_Memory_for_Periodicity_Search(){
src/aa_device_periods.cu:      cudaFree(d_one_A);
src/aa_device_periods.cu:      cudaFree(d_two_B);
src/aa_device_periods.cu:      cudaFree(d_half_C);
src/aa_device_periods.cu:      cudaFree(d_power_harmonics);
src/aa_device_periods.cu:      cudaFree(d_interbin_harmonics);
src/aa_device_periods.cu:      cudaFree(gmem_power_peak_pos);
src/aa_device_periods.cu:      cudaFree(gmem_interbin_peak_pos);
src/aa_device_periods.cu:      cudaFree(d_MSD);
src/aa_device_periods.cu:      cudaFree(d_previous_partials);
src/aa_device_periods.cu:      cudaFree(d_all_blocks);
src/aa_device_periods.cu:      cudaFree(cuFFT_workarea);
src/aa_device_periods.cu:  void Copy_data_for_periodicity_search(float *d_one_A, float **dedispersed_data, aa_periodicity_batch *batch){ //TODO add "cudaStream_t stream1"
src/aa_device_periods.cu:    cudaStream_t stream_copy[16];
src/aa_device_periods.cu:    cudaError_t e;
src/aa_device_periods.cu:    cudaMallocHost((void **) &h_small_dedispersed_data, nStreams*FFT_data_size);
src/aa_device_periods.cu:      e = cudaStreamCreate(&stream_copy[i]);
src/aa_device_periods.cu:      if (e != cudaSuccess) {
src/aa_device_periods.cu:        LOG(log_level::error, "Could not create streams in periodicity (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_periods.cu:      //e = cudaMemcpy( &d_one_A[ff*batch->nTimesamples], dedispersed_data[batch->DM_shift + ff], batch->nTimesamples*sizeof(float), cudaMemcpyHostToDevice);      
src/aa_device_periods.cu:      e = cudaMemcpyAsync(&d_one_A[ff*batch->nTimesamples], h_small_dedispersed_data + id_stream*stream_offset, FFT_data_size, cudaMemcpyHostToDevice, stream_copy[id_stream]);      
src/aa_device_periods.cu:      cudaStreamSynchronize(stream_copy[id_stream]);
src/aa_device_periods.cu:      if(e != cudaSuccess) {
src/aa_device_periods.cu:        LOG(log_level::error, "Could not cudaMemcpy in aa_device_periods.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_periods.cu:      e = cudaStreamDestroy(stream_copy[i]);
src/aa_device_periods.cu:      if (e != cudaSuccess) {
src/aa_device_periods.cu:        LOG(log_level::error, "Could not destroy stream in periodicity (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_periods.cu:    cudaFreeHost(h_small_dedispersed_data);
src/aa_device_periods.cu:  void Export_data_in_range(float *GPU_data, int nTimesamples, int nDMs, const char *filename, float dm_step, float dm_low, float sampling_time, int outer_DM_shift, int DMs_per_file=100) {
src/aa_device_periods.cu:    cudaError_t e = cudaMemcpy(h_data, GPU_data, data_size*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_periods.cu:    if(e != cudaSuccess) {
src/aa_device_periods.cu:      LOG(log_level::error, "Could not cudaMemcpy in aa_device_periods.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_periods.cu:   * \brief Performs a periodicity search on the GPU.
src/aa_device_periods.cu:   * \todo Clarify the difference between Periodicity_search and GPU_periodicity.
src/aa_device_periods.cu:  void Periodicity_search(GPU_Memory_for_Periodicity_Search *gmem, aa_periodicity_strategy PSR_strategy, double *compute_time, size_t input_plane_size, aa_periodicity_range *Prange, aa_periodicity_batch *batch, std::vector<int> *h_boxcar_widths, int harmonic_sum_algorithm, bool enable_scalloping_loss_removal){
src/aa_device_periods.cu:    aa_gpu_timer timer;
src/aa_device_periods.cu:    cudaStream_t stream; stream = NULL;
src/aa_device_periods.cu:  int Get_Number_of_Candidates(int *GPU_data){
src/aa_device_periods.cu:    cudaError_t e = cudaMemcpy(&temp, GPU_data, sizeof(int), cudaMemcpyDeviceToHost);
src/aa_device_periods.cu:    if(e != cudaSuccess) {
src/aa_device_periods.cu:      LOG(log_level::error, "Could not cudaMemcpy in aa_device_periods.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_periods.cu:  /** \brief Function that performs a GPU periodicity search. */
src/aa_device_periods.cu:  void GPU_periodicity(aa_periodicity_strategy &PSR_strategy, float ***output_buffer, aa_periodicity_candidates &Power_Candidates, aa_periodicity_candidates &Interbin_Candidates) {
src/aa_device_periods.cu:    //cudaMemGetInfo(&memory_available,&total_mem);
src/aa_device_periods.cu:    //--------> Allocation of GPU memory
src/aa_device_periods.cu:    GPU_Memory_for_Periodicity_Search GPU_memory;
src/aa_device_periods.cu:    GPU_memory.Allocate(PSR_strategy);
src/aa_device_periods.cu:    aa_gpu_timer timer, periodicity_timer;
src/aa_device_periods.cu:        GPU_memory.Reset_MSD();
src/aa_device_periods.cu:            GPU_memory.Reset_Candidate_List();
src/aa_device_periods.cu:            Copy_data_for_periodicity_search(GPU_memory.d_one_A, output_buffer[current_p_range.rangeid], &current_p_range.batches[b]);
src/aa_device_periods.cu:          Periodicity_search(&GPU_memory, PSR_strategy, &calc_time_per_range, PSR_strategy.input_plane_size(), &current_p_range, &current_p_range.batches[b], &h_boxcar_widths, harmonic_sum_algorithm, enable_scalloping_loss_removal);
src/aa_device_periods.cu:          GPU_memory.Get_MSD(h_MSD);
src/aa_device_periods.cu:          size_t nPowerCandidates = GPU_memory.Get_Number_of_Power_Candidates();
src/aa_device_periods.cu:          size_t nInterbinCandidates = GPU_memory.Get_Number_of_Interbin_Candidates();
src/aa_device_periods.cu:          if(harmonic_sum_algorithm==0) pointer_to_candidate_data = &GPU_memory.d_two_B[0];
src/aa_device_periods.cu:          else pointer_to_candidate_data = GPU_memory.d_half_C;
src/aa_device_periods.cu:          if(harmonic_sum_algorithm==0) pointer_to_candidate_data = &GPU_memory.d_two_B[PSR_strategy.input_plane_size()];
src/aa_device_periods.cu:          else pointer_to_candidate_data = GPU_memory.d_one_A;
src/aa_device_periods.cu:    cudaDeviceSynchronize();
src/aa_device_SNR_limited_inplace_kernel.cu:  __global__ void PD_ZC_GPU(float *d_input, float *d_output, int maxTaps, int nTimesamples, int nLoops) {
src/aa_device_SNR_limited_inplace_kernel.cu:  __global__ void PD_GPUv1_const(float *d_input, float *d_temp, unsigned char *d_output_taps, int maxTaps, int nTimesamples, float signal_mean, float signal_sd) {
src/aa_device_SNR_limited_inplace_kernel.cu:  /** \brief Kernel wrapper function to PD_ZC_GPU kernel function. */
src/aa_device_SNR_limited_inplace_kernel.cu:  void call_kernel_PD_ZC_GPU(float *const d_input, float *const d_output, const int &maxTaps, const int &nTimesamples, const int &nLoops) {
src/aa_device_SNR_limited_inplace_kernel.cu:  /** Kernel wrapper function to PD_GPUv1_const kernel function. */
src/aa_device_SNR_limited_inplace_kernel.cu:  void call_kernel_PD_GPUv1_const(float *const d_input, float *const d_temp, unsigned char *const d_output_taps,
src/aa_device_MSD_plane_profile.cu:#include "aa_bin_gpu.hpp"
src/aa_device_MSD_plane_profile.cu:#include "aa_gpu_timer.hpp"
src/aa_device_MSD_plane_profile.cu:    cudaError_t e = cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:    if(e != cudaSuccess) {
src/aa_device_MSD_plane_profile.cu:      LOG(log_level::error, "Could not cudaMemcpy in aa_device_MSD_plane_profile.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD_plane_profile.cu:    aa_gpu_timer timer, total_timer;
src/aa_device_MSD_plane_profile.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:      nRest = GPU_DiT_v2_wrapper(d_input_data, d_lichy, nDMs, nTimesamples);
src/aa_device_MSD_plane_profile.cu:      nRest = GPU_DiT_v2_wrapper(d_lichy, d_sudy, nDMs, decimated_timesamples);
src/aa_device_MSD_plane_profile.cu:      nRest = GPU_DiT_v2_wrapper(d_input_data, d_lichy, nDMs_half, nTimesamples);
src/aa_device_MSD_plane_profile.cu:      nRest = GPU_DiT_v2_wrapper(d_lichy, d_sudy, nDMs_half, decimated_timesamples);
src/aa_device_MSD_plane_profile.cu:      nRest = GPU_DiT_v2_wrapper(&d_input_data[nDMs_half*nTimesamples], d_lichy, nDMs_half, nTimesamples);
src/aa_device_MSD_plane_profile.cu:      nRest = GPU_DiT_v2_wrapper(d_lichy, &d_sudy[nDMs_half*(decimated_timesamples>>1)], nDMs_half, decimated_timesamples);
src/aa_device_MSD_plane_profile.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:	  nRest = GPU_DiT_v2_wrapper(d_lichy, d_sudy, nDMs, decimated_timesamples);
src/aa_device_MSD_plane_profile.cu:	  nRest = GPU_DiT_v2_wrapper(d_sudy, d_lichy, nDMs, decimated_timesamples);
src/aa_device_MSD_plane_profile.cu:      //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:	//checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:	//checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:	//checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:	//checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:#ifdef GPU_PARTIAL_TIMER
src/aa_device_MSD_plane_profile.cu:#ifdef GPU_PARTIAL_TIMER
src/aa_device_MSD_plane_profile.cu:    aa_gpu_timer timer;
src/aa_device_MSD_plane_profile.cu:    cudaError_t e = cudaMalloc((void **) &d_MSD_DIT_widths, sizeof(int)*MSD_DIT_size);
src/aa_device_MSD_plane_profile.cu:    if(e != cudaSuccess) {
src/aa_device_MSD_plane_profile.cu:      LOG(log_level::error, "Could not allocate memory in aa_device_MSD_plane_profile.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD_plane_profile.cu:    e = cudaMemcpy(d_MSD_DIT_widths, &h_MSD_DIT_widths->operator[](0), sizeof(int)*MSD_DIT_size,cudaMemcpyHostToDevice);
src/aa_device_MSD_plane_profile.cu:    if(e != cudaSuccess) {
src/aa_device_MSD_plane_profile.cu:      LOG(log_level::error, "Could not cudaMemcpy (MSD_DIT_width) in aa_device_MSD_plane_profile.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD_plane_profile.cu:    e = cudaMalloc((void **) &d_boxcar, sizeof(int)*nWidths);
src/aa_device_MSD_plane_profile.cu:    if(e != cudaSuccess) {
src/aa_device_MSD_plane_profile.cu:      LOG(log_level::error, "Could not allocate memory (d_boxcar) in aa_device_MSD_plane_profile.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD_plane_profile.cu:    e = cudaMemcpy(d_boxcar, &h_boxcar_widths->operator[](0), sizeof(int)*nWidths,cudaMemcpyHostToDevice);
src/aa_device_MSD_plane_profile.cu:    if(e != cudaSuccess) {
src/aa_device_MSD_plane_profile.cu:      LOG(log_level::error, "Could not cudaMemcpy (d_boxcar) in aa_device_MSD_plane_profile.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD_plane_profile.cu:    //	checkCudaErrors(cudaMemcpy(h_MSD_DIT, d_MSD_DIT, nMSDs*MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
src/aa_device_MSD_plane_profile.cu:    call_kernel_MSD_GPU_Interpolate_linear(1, nWidths,
src/aa_device_MSD_plane_profile.cu:    e = cudaMemcpy(h_MSD_DIT, d_MSD_DIT, nMSDs*MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:    if(e != cudaSuccess) {
src/aa_device_MSD_plane_profile.cu:      LOG(log_level::error, "Could not cudaMemcpy in aa_device_MSD_plane_profile.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD_plane_profile.cu:    e = cudaMemcpy(h_MSD_interpolated, d_MSD_interpolated, nWidths*MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:    if(e != cudaSuccess) {
src/aa_device_MSD_plane_profile.cu:      LOG(log_level::error, "Could not cudaMemcpy in aa_device_MSD_plane_profile.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD_plane_profile.cu:    //	checkCudaErrors(cudaMemcpy(d_MSD_interpolated, h_MSD_interpolated, nWidths*MSD_INTER_SIZE*sizeof(float), cudaMemcpyHostToDevice));
src/aa_device_MSD_plane_profile.cu:#ifdef GPU_PARTIAL_TIMER
src/aa_device_MSD_plane_profile.cu:    cudaMemset((void*) d_MSD_DIT, 0, (nDecimations+1)*MSD_RESULTS_SIZE*sizeof(float));
src/aa_device_MSD_plane_profile.cu:    aa_gpu_timer timer;
src/aa_device_MSD_plane_profile.cu:    cudaMalloc((void **) &d_boxcar, nTimesamples*nDMs*sizeof(float));
src/aa_device_MSD_plane_profile.cu:    cudaMalloc((void **) &d_MSD, MSD_RESULTS_SIZE*sizeof(float));
src/aa_device_MSD_plane_profile.cu:    cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:    cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:	cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:	cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:	  cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:	  cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:      //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:	  cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:	  cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:      //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:	  cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:	  cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:      //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:	  cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:	  cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:      //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:	  cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:	  cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:      //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:	  cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:	  cudaMemcpy(h_MSD, d_MSD, MSD_RESULTS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD_plane_profile.cu:      //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_MSD_plane_profile.cu:    cudaError_t e = cudaFree(d_boxcar);
src/aa_device_MSD_plane_profile.cu:    if(e != cudaSuccess) {
src/aa_device_MSD_plane_profile.cu:      LOG(log_level::error, "Could not cudaFree in aa_device_MSD_plane_profile.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD_plane_profile.cu:    e = cudaFree(d_MSD);
src/aa_device_MSD_plane_profile.cu:    if(e != cudaSuccess) {
src/aa_device_MSD_plane_profile.cu:      LOG(log_level::error, "Could not cudaFree in aa_device_MSD_plane_profile.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD_plane_profile.cu:    cudaMemGetInfo(&free_mem,&total_mem);
src/aa_device_analysis.cu://#define GPU_ANALYSIS_DEBUG
src/aa_device_analysis.cu://#define GPU_PARTIAL_TIMER
src/aa_device_analysis.cu:#define GPU_TIMER
src/aa_device_analysis.cu:#include "aa_gpu_timer.hpp"
src/aa_device_analysis.cu:// \todo cudaMalloc((void**) &gmem_peak_pos, 1*sizeof(int)); has no corresponding cudaFree.
src/aa_device_analysis.cu:// \todo cudaMemset((void*) gmem_peak_pos, 0, sizeof(int));  has no corresponding cudaFree.
src/aa_device_analysis.cu:  bool analysis_GPU(unsigned int *h_peak_list_DM, unsigned int *h_peak_list_TS, float *h_peak_list_SNR, unsigned int *h_peak_list_BW, size_t *peak_pos, size_t max_peak_size, int i, float tstart, int t_processed, int inBin, int *maxshift, int max_ndms, int const*const ndms, float cutoff, float OR_sigma_multiplier, float max_boxcar_width_in_sec, float *output_buffer, float *dm_low, float *dm_high, float *dm_step, float tsamp, int candidate_algorithm, float *d_MSD_workarea, unsigned short *d_output_taps, float *d_MSD_interpolated, unsigned long int maxTimeSamples, int enable_msd_baselinenoise, const bool dump_to_disk, const bool dump_to_user, analysis_output &output){
src/aa_device_analysis.cu:    //----------> GPU part
src/aa_device_analysis.cu:    printf("\n----------> GPU analysis part\n");
src/aa_device_analysis.cu:    aa_gpu_timer total_timer, timer;
src/aa_device_analysis.cu:#ifdef GPU_PARTIAL_TIMER
src/aa_device_analysis.cu:      cudaMalloc((void**) &gmem_peak_pos, 1*sizeof(int));
src/aa_device_analysis.cu:      cudaMemset((void*) gmem_peak_pos, 0, sizeof(int));
src/aa_device_analysis.cu:      cudaMalloc((void**) &gmem_filteredPeak_pos, 1*sizeof(int));
src/aa_device_analysis.cu:      cudaMemset((void*) gmem_filteredPeak_pos, 0, sizeof(int));
src/aa_device_analysis.cu:#ifdef GPU_PARTIAL_TIMER
src/aa_device_analysis.cu:	//checkCudaErrors(cudaGetLastError());
src/aa_device_analysis.cu:#ifdef GPU_ANALYSIS_DEBUG
src/aa_device_analysis.cu:#ifdef GPU_PARTIAL_TIMER
src/aa_device_analysis.cu:#ifdef GPU_PARTIAL_TIMER
src/aa_device_analysis.cu:#ifdef GPU_PARTIAL_TIMER
src/aa_device_analysis.cu:	//checkCudaErrors(cudaGetLastError());
src/aa_device_analysis.cu:	cudaError_t e = cudaMemcpy(&temp_peak_pos, gmem_peak_pos, sizeof(int), cudaMemcpyDeviceToHost);
src/aa_device_analysis.cu:	if(e != cudaSuccess) {
src/aa_device_analysis.cu:	  LOG(log_level::error, "Could not cudaMemcpy in aa_device_analysis.cu -- temp_peak_pos (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_analysis.cu:#ifdef GPU_ANALYSIS_DEBUG
src/aa_device_analysis.cu:	  cudaError_t e = cudaMemcpy(&h_peak_list_DM[(*peak_pos)],  d_peak_list_DM,  temp_peak_pos*sizeof(unsigned int), cudaMemcpyDeviceToHost);
src/aa_device_analysis.cu:	  if(e != cudaSuccess) {
src/aa_device_analysis.cu:	    LOG(log_level::error, "Could not cudaMemcpy in aa_device_analysis.cu -- peak_list_DM (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_analysis.cu:	  e = cudaMemcpy(&h_peak_list_TS[(*peak_pos)],  d_peak_list_TS,  temp_peak_pos*sizeof(unsigned int), cudaMemcpyDeviceToHost);
src/aa_device_analysis.cu:	  if(e != cudaSuccess) {
src/aa_device_analysis.cu:	    LOG(log_level::error, "Could not cudaMemcpy in aa_device_analysis.cu -- peak_list_TS (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_analysis.cu:	  e = cudaMemcpy(&h_peak_list_SNR[(*peak_pos)], d_peak_list_SNR, temp_peak_pos*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_analysis.cu:	  if(e != cudaSuccess) {
src/aa_device_analysis.cu:	    LOG(log_level::error, "Could not cudaMemcpy in aa_device_analysis.cu -- peak_list_SNR (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_analysis.cu:	  e = cudaMemcpy(&h_peak_list_BW[(*peak_pos)],  d_peak_list_BW,  temp_peak_pos*sizeof(unsigned int), cudaMemcpyDeviceToHost);
src/aa_device_analysis.cu:	  if(e != cudaSuccess) {
src/aa_device_analysis.cu:	    LOG(log_level::error, "Could not cudaMemcpy in aa_device_analysis.cu -- peak_list_BW (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_analysis.cu:	cudaMemset((void*) gmem_peak_pos, 0, sizeof(int));
src/aa_device_analysis.cu:			cudaMemset((void*) d_output_SNR, 0, d_output_SNR_size*sizeof(float));
src/aa_device_analysis.cu:			cudaMemset((void*) d_peak_list_DM, 0, sizeof(unsigned int)*d_peak_list_size);
src/aa_device_analysis.cu:			cudaMemset((void*) d_peak_list_TS, 0, sizeof(unsigned int)*d_peak_list_size);
src/aa_device_analysis.cu:			cudaMemset((void*) d_peak_list_BW, 0, sizeof(unsigned int)*d_peak_list_size);
src/aa_device_analysis.cu:			cudaMemset((void*) d_peak_list_SNR, 0, sizeof(float)*d_peak_list_size);
src/aa_device_analysis.cu:			cudaError_t e = cudaMemcpy(d_peak_list_DM2, h_peak_list_DM, sizeof(unsigned int)*local_peak_pos, cudaMemcpyHostToDevice);
src/aa_device_analysis.cu:			if (e != cudaSuccess){
src/aa_device_analysis.cu:				LOG(log_level::error, "Could not cudaMemcpy in d_peak_list_DM2 (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_analysis.cu:			e = cudaMemcpy(d_peak_list_TS2, h_peak_list_TS, sizeof(unsigned int)*local_peak_pos, cudaMemcpyHostToDevice);
src/aa_device_analysis.cu:			if (e != cudaSuccess){
src/aa_device_analysis.cu:				LOG(log_level::error, "Could not cudaMemcpy in d_peak_list_TS2 (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_analysis.cu:			e = cudaMemcpy(d_peak_list_BW2, h_peak_list_BW, sizeof(unsigned int)*local_peak_pos, cudaMemcpyHostToDevice);
src/aa_device_analysis.cu:			if (e != cudaSuccess){
src/aa_device_analysis.cu:					LOG(log_level::error, "Could not cudaMemcpy in d_peak_list_BW2 (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_analysis.cu:			e = cudaMemcpy(d_peak_list_SNR2, h_peak_list_SNR, sizeof(float)*local_peak_pos, cudaMemcpyHostToDevice);
src/aa_device_analysis.cu:			if (e != cudaSuccess){
src/aa_device_analysis.cu:				LOG(log_level::error, "Could not cudaMemcpy in d_peak_list_SNR2 (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_analysis.cu:			call_gpu_Filter_peaks(d_peak_list_DM, d_peak_list_TS, d_peak_list_BW, d_peak_list_SNR, d_peak_list_DM2, d_peak_list_TS2, d_peak_list_BW2, d_peak_list_SNR2, local_peak_pos, filter_size, (int)d_peak_list_size, gmem_filteredPeak_pos);
src/aa_device_analysis.cu:			cudaMemcpy(&temp_peak_pos, gmem_filteredPeak_pos, sizeof(int), cudaMemcpyDeviceToHost);
src/aa_device_analysis.cu:			cudaMemcpy(h_peak_list_DM, d_peak_list_DM, local_peak_pos*sizeof(unsigned int), cudaMemcpyDeviceToHost);
src/aa_device_analysis.cu:			cudaMemcpy(h_peak_list_TS, d_peak_list_TS, local_peak_pos*sizeof(unsigned int), cudaMemcpyDeviceToHost);
src/aa_device_analysis.cu:			cudaMemcpy(h_peak_list_BW, d_peak_list_BW, local_peak_pos*sizeof(unsigned int), cudaMemcpyDeviceToHost);
src/aa_device_analysis.cu:			cudaMemcpy(h_peak_list_SNR, d_peak_list_SNR, local_peak_pos*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_analysis.cu:	#ifdef GPU_PARTIAL_TIMER
src/aa_device_analysis.cu:		cudaFree(gmem_peak_pos);
src/aa_device_analysis.cu:#ifdef GPU_TIMER
src/aa_device_analysis.cu:    //----------> GPU part
src/aa_device_peak_find_kernel.cu:__global__ void gpu_Filter_peaks_kernel(unsigned int *d_new_peak_list_DM, unsigned int *d_new_peak_list_TS, unsigned int *d_new_peak_list_BW, float *d_new_peak_list_SNR, 
src/aa_device_peak_find_kernel.cu:	void call_gpu_Filter_peaks(unsigned int *new_peak_list_DM, unsigned int *new_peak_list_TS, unsigned int *new_peak_list_BW, float *new_peak_list_SNR, unsigned int *d_peak_list_DM, unsigned int *d_peak_list_TS, unsigned int *d_peak_list_BW, float *d_peak_list_SNR, unsigned int nElements, unsigned int max_distance, int max_list_pos, int *gmem_pos){
src/aa_device_peak_find_kernel.cu:			gpu_Filter_peaks_kernel<<<gridSize, blockDim>>>(new_peak_list_DM, new_peak_list_TS, new_peak_list_BW, new_peak_list_SNR, d_peak_list_DM, d_peak_list_TS, d_peak_list_BW, d_peak_list_SNR, nElements, max_distance*max_distance, nLoops, max_list_pos, gmem_pos);
src/aa_dedisperse.cpp:	      //Dedisperse data on the GPU
src/aa_dedisperse.cpp:	      cudaDeviceSetSharedMemConfig (cudaSharedMemBankSizeFourByte);
src/aa_dedisperse.cpp:	      //cudaFuncSetCacheConfig(shared_dedisperse_kernel_16, cudaFuncCachePreferShared); //Subsume in call_kernel_*
src/aa_dedisperse.cpp:			cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
src/aa_dedisperse.cpp:	      //Dedisperse data on the GPU
src/aa_dedisperse.cpp:	      cudaDeviceSetSharedMemConfig (cudaSharedMemBankSizeFourByte);
src/aa_dedisperse.cpp:	      //cudaFuncSetCacheConfig(shared_dedisperse_kernel, cudaFuncCachePreferShared); //Subsume in call_kernel_*
src/aa_dedisperse.cpp:      //Dedisperse data on the GPU
src/aa_dedisperse.cpp:      //cudaFuncSetCacheConfig(cache_dedisperse_kernel, cudaFuncCachePreferL1); //Subsume in call_kernel_*
src/aa_dedisperse.cpp: cudaError_t CUDA_error;  
src/aa_dedisperse.cpp: CUDA_error = cudaGetLastError();
src/aa_dedisperse.cpp: if(CUDA_error != cudaSuccess) {
src/aa_device_SPS_long.cu:    //---------> Specific nVidia stuff
src/aa_device_SPS_long.cu:    cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
src/aa_device_SPS_long.cu:    cudaDeviceSetSharedMemConfig (cudaSharedMemBankSizeEightByte);
src/aa_device_SPS_long.cu:    //---------> CUDA block and CUDA grid parameters
src/aa_device_SPS_long.cu:    if(nBlocks>0) call_kernel_SPDT_GPU_1st_plane(gridSize, blockSize, d_input, d_boxcar_values, d_decimated, d_output_SNR, d_output_taps, (float2 *) d_MSD_interpolated, decimated_timesamples, nBoxcars, dtm);
src/aa_device_SPS_long.cu:    //checkCudaErrors(cudaGetLastError());
src/aa_device_SPS_long.cu:	  call_kernel_SPDT_GPU_Nth_plane(gridSize,blockSize, &d_input[shift], &d_boxcar_values[nDMs*(nTimesamples>>1)], d_boxcar_values, d_decimated, &d_output_SNR[nDMs*output_shift], &d_output_taps[nDMs*output_shift], (float2 *) &d_MSD_interpolated[MSD_plane_pos*2], decimated_timesamples, nBoxcars, startTaps, (1<<iteration), dtm);
src/aa_device_SPS_long.cu:	  call_kernel_SPDT_GPU_Nth_plane(gridSize,blockSize, &d_decimated[shift], d_boxcar_values, &d_boxcar_values[nDMs*(nTimesamples>>1)], d_input, &d_output_SNR[nDMs*output_shift], &d_output_taps[nDMs*output_shift], (float2 *) &d_MSD_interpolated[MSD_plane_pos*2], decimated_timesamples, nBoxcars, startTaps, (1<<iteration), dtm);
src/aa_device_SPS_long.cu:      //checkCudaErrors(cudaGetLastError());
src/aa_device_load_data.cu:   * \brief Function to load data from host memory into GPU memory.
src/aa_device_load_data.cu:   * \warning If the file extension of this file is *.cpp, then the code will compile but there will be a runtime CUDA error when copying to device memory.
src/aa_device_load_data.cu:			//checkCudaErrors(cudaGetLastError());
src/aa_device_load_data.cu:			cudaMemcpy(device_pointer, host_pointer, size, cudaMemcpyHostToDevice);
src/aa_device_load_data.cu:			//checkCudaErrors(cudaGetLastError());
src/aa_device_load_data.cu:				cudaMemcpy(d_dm_shifts, dmshifts, nchans*sizeof(float), cudaMemcpyHostToDevice);
src/aa_device_load_data.cu:				cudaMemcpy(d_dm_shifts, dmshifts, nchans*sizeof(float), cudaMemcpyHostToDevice);
src/aa_device_load_data.cu:		//checkCudaErrors(cudaGetLastError());
src/aa_device_load_data.cu:		cudaMemcpy(device_pointer, host_pointer, size, cudaMemcpyHostToDevice);
src/aa_device_load_data.cu:		//checkCudaErrors(cudaGetLastError());
src/aa_device_load_data.cu:			cudaMemcpy(d_dm_shifts, dmshifts, nchans*sizeof(float), cudaMemcpyHostToDevice);
src/aa_device_load_data.cu:			cudaMemcpy(d_dm_shifts, dmshifts, nchans*sizeof(float), cudaMemcpyHostToDevice);
src/aa_host_debug.cpp:  void debug(int test, clock_t start_time, int range, int *outBin, int enable_debug, int analysis, int output_dmt, int multi_file, float sigma_cutoff, float power, int max_ndms, float *user_dm_low, float *user_dm_high, float *user_dm_step, float *dm_low, float *dm_high, float *dm_step, int *ndms, int nchans, int nsamples, int nifs, int nbits, float tsamp, float tstart, float fch1, float foff, int maxshift, float max_dm, int nsamp, size_t gpu_inputsize, size_t gpu_outputsize, size_t inputsize, size_t outputsize) {
src/aa_host_debug.cpp:    printf("\n Using GPU __ldg() code (version: sm_35)");
src/aa_host_debug.cpp:    printf("\n Using standard GPU code");
src/aa_host_debug.cpp:      printf("\nInitialised GPU:\t\t%.16g(s)\n", (double)(now - start_time) / CLOCKS_PER_SEC);
src/aa_host_debug.cpp:      printf("\nDevice Input size:\t\t%d MB", (int) (gpu_inputsize / 1024 / 1024));
src/aa_host_debug.cpp:      printf("\nDevice Output size:\t\t%d MB", (int) (gpu_outputsize / 1024 / 1024));
src/aa_host_debug.cpp:      printf("\nLoaded data onto the GPU:\t%.16g(s)\n", (double)(now - start_time) / CLOCKS_PER_SEC);
src/aa_device_single_FIR_kernel.cu:#include <cuda.h>
src/aa_device_single_FIR_kernel.cu:#include <cuda_runtime.h>
src/aa_device_single_FIR_kernel.cu:  __global__ void PD_FIR_GPU(float const* __restrict__ d_input, float *d_output, int nTaps, int nLoops, int nTimesamples) {
src/aa_device_single_FIR_kernel.cu:  __global__ void PD_FIR_GPUv1(float const* __restrict__ d_input, float *d_output, int nTaps, int nLoops, unsigned int nTimesamples) {
src/aa_device_single_FIR_kernel.cu:  /** \brief Kernel wrapper function for PD_FIR_GPU kernel function. */
src/aa_device_single_FIR_kernel.cu:  void call_kernel_PD_FIR_GPU(const dim3 &grid_size, const dim3 &block_size, const int &SM_size, float const *const d_input, float *const d_output, const int &nTaps, const int &nLoops, const int &nTimesamples) {
src/aa_device_single_FIR_kernel.cu:    PD_FIR_GPU<<<grid_size, block_size, SM_size>>>(d_input, d_output, nTaps, nLoops, nTimesamples);
src/aa_device_single_FIR_kernel.cu:  /** \brief Kernel wrapper function for PD_FIR_GPUv1 kernel function. */
src/aa_device_single_FIR_kernel.cu:  void call_kernel_PD_FIR_GPUv1(const dim3 &grid_size, const dim3 &block_size, const int &SM_size, float const *const d_input, float *const d_output, const int &nTaps, const int &nLoops, const unsigned int &nTimesamples) {
src/aa_device_single_FIR_kernel.cu:    PD_FIR_GPUv1<<<grid_size, block_size, SM_size>>>(d_input, d_output, nTaps, nLoops, nTimesamples);
src/aa_device_rfi.cu:  /** \brief Function that performs rfi mitigation on the GPU. */
src/aa_device_rfi.cu:  void rfi_gpu(unsigned short *d_input, int nchans, int nsamp) {
src/aa_device_rfi.cu:    call_kernel_rfi_gpu_kernel(num_blocks, threads_per_block,
src/aa_device_rfi.cu:    cudaDeviceSynchronize();
src/aa_device_rfi.cu:    printf("\nPerformed RFI: %lf (GPU estimate)", time);
src/aa_main.cpp:		cudaMemGetInfo(&free_mem,&total_mem);
src/aa_device_MSD_normal_kernel.cu:  __global__ void MSD_GPU_limited(float const* __restrict__ d_input, float *d_output, int y_steps, int nTimesamples, int offset) {
src/aa_device_MSD_normal_kernel.cu:  /** \brief Wrapper function to kernel function MSD_GPU_limited. */
src/aa_device_MSD_normal_kernel.cu:  void call_kernel_MSD_GPU_limited(const dim3 &grid_size, const dim3 &block_size, float const *const d_input, float *const d_output, const int &y_steps, const int &nTimesamples, const int &offset) {
src/aa_device_MSD_normal_kernel.cu:    MSD_GPU_limited<<<grid_size, block_size>>>(d_input, d_output, y_steps, nTimesamples, offset);
src/aa_bin_gpu.cpp:#include <cuda_runtime.h>
src/aa_bin_gpu.cpp:#include "aa_bin_gpu.hpp"
src/aa_bin_gpu.cpp:#include "aa_gpu_timer.hpp"
src/aa_bin_gpu.cpp:void bin_gpu(unsigned short *const d_input, float *const d_output, const int nchans, const int nsamp) {
src/aa_bin_gpu.cpp:    cudaMemset(d_output, 0, size_chunk*sizeof(float));
src/aa_bin_gpu.cpp:    cudaDeviceSynchronize();
src/aa_bin_gpu.cpp:    cudaDeviceSynchronize();
src/aa_bin_gpu.cpp:    //printf("\nPerformed Bin: %f (GPU estimate)", time);
src/aa_bin_gpu.cpp:    cudaMemset(d_output, 0, size_chunk*sizeof(float));
src/aa_bin_gpu.cpp:int GPU_DiT_v2_wrapper(float *d_input, float *d_output, int nDMs, int nTimesamples) {
src/aa_bin_gpu.cpp:  aa_gpu_timer timer;
src/aa_bin_gpu.cpp:  //---------> CUDA block and CUDA grid parameters
src/aa_bin_gpu.cpp:  int nCUDAblocks_x=nTimesamples/(DIT_ELEMENTS_PER_THREAD*nThreads*2);
src/aa_bin_gpu.cpp:  if(nRest>0) nCUDAblocks_x++;
src/aa_bin_gpu.cpp:  int nCUDAblocks_y=nDMs/DIT_YSTEP;
src/aa_bin_gpu.cpp:  dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);
src/aa_bin_gpu.cpp:  call_kernel_DiT_GPU_v2(gridSize,
src/aa_device_SPS_long_kernel.cu:  __global__ void SPDT_GPU_1st_plane(float const* __restrict__ d_input, float *d_bv_out, float *d_decimated, float *d_output_SNR, ushort *d_output_taps, float2 const* __restrict__ d_MSD, int nTimesamples, int nBoxcars, const int dtm) {
src/aa_device_SPS_long_kernel.cu:  __global__ void SPDT_GPU_Nth_plane(float const* __restrict__ d_input, float *d_bv_in, float *d_bv_out, float *d_decimated, float *d_output_SNR, ushort *d_output_taps, float2 const* __restrict__ d_MSD, const int nTimesamples, const int nBoxcars, const int startTaps, const int DIT_value, const int dtm) {
src/aa_device_SPS_long_kernel.cu:  /** \brief Kernel wrapper function to SPDT_GPU_1st_plane kernel function. */
src/aa_device_SPS_long_kernel.cu:  void call_kernel_SPDT_GPU_1st_plane(const dim3 &grid_size, const dim3 &block_size, float const *const d_input, float *const d_bv_out,
src/aa_device_SPS_long_kernel.cu:    SPDT_GPU_1st_plane<<<grid_size,block_size>>>(d_input, d_bv_out,
src/aa_device_SPS_long_kernel.cu:  /** \brief Kernel wrapper function to SPDT_GPU_Nth_plane kernel function. */
src/aa_device_SPS_long_kernel.cu:  void call_kernel_SPDT_GPU_Nth_plane(const dim3 &grid_size, const dim3 &block_size,
src/aa_device_SPS_long_kernel.cu:    SPDT_GPU_Nth_plane<<<grid_size,block_size>>>(d_input, d_bv_in, d_bv_out,
src/aa_device_SPS_inplace_kernel.cu:  __global__ void PD_ZC_GPU_KERNEL(float *d_input, float *d_output, int maxTaps, int nTimesamples, int nLoops)
src/aa_device_SPS_inplace_kernel.cu:  __global__ void PD_INPLACE_GPU_KERNEL(float *d_input, float *d_temp, unsigned char *d_output_taps, float *d_MSD, int maxTaps, int nTimesamples)
src/aa_device_SPS_inplace_kernel.cu:  /** \brief Kernel wrapper function to PD_ZC_GPU_KERNEL kernel function. */
src/aa_device_SPS_inplace_kernel.cu:  void call_kernel_PD_ZC_GPU_KERNEL(const dim3 &grid_size, const dim3 &block_size, float *const d_input, float *const d_output, const int &maxTaps, const int &nTimesamples, const int &nLoops) {
src/aa_device_SPS_inplace_kernel.cu:    PD_ZC_GPU_KERNEL<<<grid_size, block_size>>>(d_input, d_output, maxTaps, nTimesamples, nLoops);
src/aa_device_SPS_inplace_kernel.cu:  /** \brief Kernel wrapper function to PD_INPLACE_GPU_KERNEL kernel function. */
src/aa_device_SPS_inplace_kernel.cu:  void call_kernel_PD_INPLACE_GPU_KERNEL(const dim3 &grid_size, const dim3 &block_size, const int &SM_size, float *const d_input,
src/aa_device_SPS_inplace_kernel.cu:    PD_INPLACE_GPU_KERNEL<<<grid_size, block_size, SM_size>>>(d_input, d_temp, d_output_taps,
src/aa_host_utilities.cpp:#include <cuda.h>
src/aa_host_utilities.cpp:#include <cuda_runtime.h>
src/aa_host_utilities.cpp:    cudaError_t err;
src/aa_host_utilities.cpp:    err = cudaMemcpy(h_fft_input, d_FFT_complex_output, fft_input_size_bytes, cudaMemcpyDeviceToHost);
src/aa_host_utilities.cpp:    if(err != cudaSuccess) printf("CUDA error\n");
src/aa_host_utilities.cpp:    err = cudaMemcpy(h_ddtr_data, d_dedispersed_data, fft_ddtr_size_bytes, cudaMemcpyDeviceToHost);
src/aa_host_utilities.cpp:    if(err != cudaSuccess) printf("CUDA error\n");
src/aa_device_convolution.cu:		//---------> Specific nVidia stuff
src/aa_device_convolution.cu:		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
src/aa_device_convolution.cu:		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
src/aa_device_convolution.cu:		call_kernel_k_customFFT_GPU_forward(gridSize, blockSize, d_filters, d_filters, FFT_size);
src/aa_device_convolution.cu:		call_kernel_k_GPU_conv_OLS_via_customFFT(gridSize, blockSize, d_input_signal, d_output_plane, d_filters, signal_length, useful_part_size, offset, nConvolutions, nFilters, scale, convolution_length);
src/aa_device_SNR_limited.cu:    //---------> Specific nVidia stuff
src/aa_device_SNR_limited.cu:    cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
src/aa_device_SNR_limited.cu:    cudaDeviceSetSharedMemConfig (cudaSharedMemBankSizeFourByte);
src/aa_device_SNR_limited.cu:    //---------> CUDA block and CUDA grid parameters
src/aa_device_SNR_limited.cu:    call_kernel_SNR_GPU_limited(gridSize, blockSize, d_FIR_input, d_SNR_output, d_SNR_taps, d_MSD,
src/aa_ddtr_strategy.cpp:    const size_t gpu_memory = free_memory - 104857600; // - 1073741824;
src/aa_ddtr_strategy.cpp:    str_dm[0].high = str_dm[0].low + ( m_ndms[0] * ( plan.user_dm(0).step ) );   // Redefines DM plan to suit GPU
src/aa_ddtr_strategy.cpp:     * 1) nchans < m_max_ndms & nsamp fits in GPU RAM
src/aa_ddtr_strategy.cpp:     * 2) nchans > m_max_ndms & nsamp fits in GPU RAM
src/aa_ddtr_strategy.cpp:     * 3) nchans < m_max_ndms & nsamp does not fit in GPU RAM
src/aa_ddtr_strategy.cpp:     * 4) nchans > m_max_ndms & nsamp does not fit in GPU RAM
src/aa_ddtr_strategy.cpp:      // Maximum number of samples we can fit in our GPU RAM is then given by:
src/aa_ddtr_strategy.cpp:      max_tsamps = (unsigned int) ( (gpu_memory) / ( sizeof(unsigned short)*nchans + sizeof(float)*(m_max_ndms) + SPDT_memory_requirements )); // maximum number of timesamples we can fit into GPU memory
src/aa_ddtr_strategy.cpp:	LOG(log_level::error, "The selected GPU does not have enough memory for this number of dispersion trials. Reduce maximum dm or increase the size of dm step.");
src/aa_ddtr_strategy.cpp:      // Next check to see if nsamp fits in GPU RAM:
src/aa_ddtr_strategy.cpp:      // Maximum number of samples we can fit in our GPU RAM is then given by:
src/aa_ddtr_strategy.cpp:      max_tsamps = (unsigned int) ( ( gpu_memory ) / ( nchans * ( sizeof(float) + sizeof(unsigned short) )+ SPDT_memory_requirements ));
src/aa_ddtr_strategy.cpp:	printf("gpu_memory: %zu. Maximum number of tsamp: %d. SPDT: %zu max_ndms: %d\n", gpu_memory, max_tsamps, SPDT_memory_requirements, m_max_ndms);
src/aa_ddtr_strategy.cpp:	LOG(log_level::error, "The selected GPU does not have enough memory for this number of dispersion trials. Reduce maximum dm or increase the size of dm step.");
src/aa_ddtr_strategy.cpp:      // Next check to see if nsamp fits in GPU RAM:
src/aa_ddtr_strategy.cpp:    // The memory that will be allocated on the GPU in function allocate_memory_gpu is given by gpu_inputsize + gpu_outputsize
src/aa_ddtr_strategy.cpp:    // gpu_inputsize
src/aa_ddtr_strategy.cpp:    // gpu_outputsize depends on nchans
src/aa_device_SPS_inplace.cu:    //---------> Specific nVidia stuff
src/aa_device_SPS_inplace.cu:    cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
src/aa_device_SPS_inplace.cu:    cudaDeviceSetSharedMemConfig (cudaSharedMemBankSizeEightByte);
src/aa_device_SPS_inplace.cu:    cudaMalloc((void **) &d_output, nBlocks_x * ( maxTaps - 1 ) * nDMs * sizeof(float));
src/aa_device_SPS_inplace.cu:    //cudaMemset((void*) d_output, 0, nBlocks_x*(maxTaps-1)*nDMs*sizeof(float));
src/aa_device_SPS_inplace.cu:    //---------> CUDA block and CUDA grid parameters for temporary storage
src/aa_device_SPS_inplace.cu:    call_kernel_PD_ZC_GPU_KERNEL(gridSize_temp, blockSize_temp, d_input, d_output, maxTaps,
src/aa_device_SPS_inplace.cu:    //---------> CUDA block and CUDA grid parameters for in-place PD
src/aa_device_SPS_inplace.cu:    call_kernel_PD_INPLACE_GPU_KERNEL(gridSize, blockSize, SM_size, d_input, d_output, d_output_taps,
src/aa_device_SPS_inplace.cu:    cudaFree(d_output);
src/aa_device_save_data.cu:  /** \brief Copy data and set up the GPU constants/variables. */
src/aa_device_save_data.cu:    cudaMemcpy(host_pointer, device_pointer, size, cudaMemcpyDeviceToHost);
src/aa_device_save_data.cu:  /** \brief Copy data and set up the GPU constants/variables. */
src/aa_device_save_data.cu:	cudaMemcpy(host_pointer + host_offset, device_pointer + device_offset, size, cudaMemcpyDeviceToHost);
src/aa_device_save_data.cu:  /** \brief Copy data and set up the GPU constants/variables. */
src/aa_device_save_data.cu:	cudaStream_t stream_copy[16];
src/aa_device_save_data.cu:	cudaError_t e;
src/aa_device_save_data.cu:	e = cudaMallocHost((void **) &h_array_pinned, data_size*nStreams);
src/aa_device_save_data.cu:	if (e != cudaSuccess) {
src/aa_device_save_data.cu:		LOG(log_level::error, "Could not create pinned memory on host for DDTR D2H copy (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_save_data.cu:        	e = cudaStreamCreate(&stream_copy[i]);
src/aa_device_save_data.cu:		if (e != cudaSuccess) {
src/aa_device_save_data.cu:			LOG(log_level::error, "Could not create stream in DDTR D2H copy (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_save_data.cu:		cudaMemcpyAsync(h_array_pinned + id_stream*stream_offset, d_DDTR_output + device_offset, data_size, cudaMemcpyDeviceToHost, stream_copy[id_stream]);
src/aa_device_save_data.cu:		cudaStreamSynchronize(stream_copy[id_stream]);
src/aa_device_save_data.cu:        	e = cudaStreamDestroy(stream_copy[i]);
src/aa_device_save_data.cu:		if (e != cudaSuccess) {
src/aa_device_save_data.cu:			LOG(log_level::error, "Could not destroy stream in DDTR D2H copy (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_save_data.cu:	e = cudaFreeHost(h_array_pinned);
src/aa_device_save_data.cu:	if (e != cudaSuccess){
src/aa_device_save_data.cu:		LOG(log_level::error, "Could not free host pinned memory in DDTR D2H copy (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_single_FIR.cu:    //---------> Specific nVidia stuff
src/aa_device_single_FIR.cu:    cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
src/aa_device_single_FIR.cu:    cudaDeviceSetSharedMemConfig (cudaSharedMemBankSizeEightByte);
src/aa_device_single_FIR.cu:    //---------> CUDA block and CUDA grid parameters
src/aa_device_single_FIR.cu:    int nCUDAblocks_x = (int) ( ( nTimesamples - nTaps + 1 ) / ( PD_FIR_ACTIVE_WARPS * WARP * PD_FIR_NWINDOWS ) );
src/aa_device_single_FIR.cu:    int nCUDAblocks_y = nDMs;
src/aa_device_single_FIR.cu:    dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);
src/aa_device_single_FIR.cu:    call_kernel_PD_FIR_GPU(gridSize, blockSize, SM_size, d_input, d_output, nTaps, nLoops, nTimesamples);
src/aa_device_single_FIR.cu:    ut = nTimesamples - nCUDAblocks_x * PD_FIR_ACTIVE_WARPS * WARP * PD_FIR_NWINDOWS;
src/aa_device_single_FIR.cu:  int GPU_FIRv1_wrapper(float *d_input, float *d_output, int nTaps, unsigned int nDMs, unsigned int nTimesamples){
src/aa_device_single_FIR.cu:    //---------> CUDA block and CUDA grid parameters
src/aa_device_single_FIR.cu:    int nCUDAblocks_x=(int) ((nTimesamples - nTaps + 1)/(PD_FIR_ACTIVE_WARPS*WARP*PD_FIR_NWINDOWS));
src/aa_device_single_FIR.cu:    int nCUDAblocks_y=nDMs; //Head size
src/aa_device_single_FIR.cu:    dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);			//nCUDAblocks_y goes through spectra
src/aa_device_single_FIR.cu:    dim3 blockSize(PD_FIR_ACTIVE_WARPS*WARP, 1, 1); 		//nCUDAblocks_x goes through channels
src/aa_device_single_FIR.cu:    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
src/aa_device_single_FIR.cu:    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
src/aa_device_single_FIR.cu:    call_kernel_PD_FIR_GPUv1(gridSize,blockSize, SM_size, d_input, d_output, nTaps, nLoops, nTimesamples);
src/aa_device_single_FIR.cu:    ut=nTimesamples - nCUDAblocks_x*PD_FIR_ACTIVE_WARPS*WARP*PD_FIR_NWINDOWS;
src/aa_device_single_FIR.cu:    //---------> CUDA block and CUDA grid parameters
src/aa_device_single_FIR.cu:    int nCUDAblocks_x=(int) (itemp/PPF_L1_THREADS_PER_BLOCK);
src/aa_device_single_FIR.cu:    if(itemp%PPF_L1_THREADS_PER_BLOCK!=0) nCUDAblocks_x++;
src/aa_device_single_FIR.cu:    int nCUDAblocks_y=(int) nDMs;
src/aa_device_single_FIR.cu:    dim3 GridSize(nCUDAblocks_x, nCUDAblocks_y, 1);
src/aa_device_MSD.cu:  //---------> Specific nVidia stuff
src/aa_device_MSD.cu:  cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
src/aa_device_MSD.cu:  cudaDeviceSetSharedMemConfig (cudaSharedMemBankSizeFourByte);
src/aa_device_MSD.cu:  call_kernel_MSD_GPU_limited(MSD_conf->partials_gridSize,MSD_conf->partials_blockSize, d_input, &d_temp[MSD_conf->address*MSD_PARTIAL_SIZE], MSD_conf->nSteps.y, (int) MSD_conf->nTimesamples, (int)MSD_conf->offset);
src/aa_device_MSD.cu:  call_kernel_MSD_GPU_final_regular(MSD_conf->final_gridSize,MSD_conf->final_blockSize,
src/aa_device_MSD.cu:  cudaError_t e = cudaMemcpy(h_MSD, d_MSD, MSD_PARTIAL_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
src/aa_device_MSD.cu:  if(e != cudaSuccess) {
src/aa_device_MSD.cu:    LOG(log_level::error, "Could not cudaMemcpy in aa_device_MSD.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD.cu:  cudaError_t e = cudaMalloc((void **) &d_temp, conf.nBlocks_total*MSD_PARTIAL_SIZE*sizeof(float));
src/aa_device_MSD.cu:  if(e != cudaSuccess) {
src/aa_device_MSD.cu:    LOG(log_level::error, "Could not cudaMemcpy in aa_device_MSD.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD.cu:  e = cudaFree(d_temp);
src/aa_device_MSD.cu:  if(e != cudaSuccess) {
src/aa_device_MSD.cu:    LOG(log_level::error, "Could not cudaMemcpy in aa_device_MSD.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD.cu:  call_kernel_MSD_GPU_limited(MSD_conf->partials_gridSize,MSD_conf->partials_blockSize, d_input, &d_temp[MSD_conf->address*MSD_PARTIAL_SIZE], MSD_conf->nSteps.y, (int) MSD_conf->nTimesamples, (int)MSD_conf->offset);
src/aa_device_MSD.cu:  call_kernel_MSD_GPU_final_regular(MSD_conf->final_gridSize, MSD_conf->final_blockSize,
src/aa_device_MSD.cu:  cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost); 
src/aa_device_MSD.cu:  cudaMalloc((void **) &d_temp, conf.nBlocks_total*MSD_PARTIAL_SIZE*sizeof(float));
src/aa_device_MSD.cu:  cudaFree(d_temp);
src/aa_device_MSD.cu:	call_kernel_MSD_GPU_calculate_partials_2d_and_minmax(MSD_conf->partials_gridSize,MSD_conf->partials_blockSize, d_input, &d_temp[MSD_conf->address*MSD_PARTIAL_SIZE], MSD_conf->nSteps.y, (int) MSD_conf->nTimesamples, (int)MSD_conf->offset);
src/aa_device_MSD.cu:	call_kernel_MSD_GPU_final_regular(MSD_conf->final_gridSize, MSD_conf->final_blockSize, &d_temp[MSD_conf->address*MSD_PARTIAL_SIZE], d_MSD, MSD_conf->nBlocks_total);
src/aa_device_MSD.cu:		cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost); 
src/aa_device_MSD.cu:		call_kernel_MSD_GPU_final_nonregular(MSD_conf->final_gridSize,MSD_conf->final_blockSize, &d_temp[MSD_conf->address*MSD_PARTIAL_SIZE], d_MSD, MSD_conf->nBlocks_total);
src/aa_device_MSD.cu:			cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost); 
src/aa_device_MSD.cu:		cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost); 
src/aa_device_MSD.cu:  cudaMalloc((void **) &d_temp, conf.nBlocks_total*MSD_PARTIAL_SIZE*sizeof(float));
src/aa_device_MSD.cu:  cudaFree(d_temp);
src/aa_device_MSD.cu:  call_kernel_MSD_GPU_calculate_partials_2d_and_minmax(MSD_conf->partials_gridSize,MSD_conf->partials_blockSize, d_input, &d_temp[MSD_conf->address*MSD_PARTIAL_SIZE], MSD_conf->nSteps.y, (int) MSD_conf->nTimesamples, (int)MSD_conf->offset);
src/aa_device_MSD.cu:  call_kernel_MSD_GPU_final_regular(MSD_conf->final_gridSize,MSD_conf->final_blockSize,
src/aa_device_MSD.cu:  cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost); 
src/aa_device_MSD.cu:    call_kernel_MSD_GPU_final_nonregular(MSD_conf->final_gridSize,MSD_conf->final_blockSize,
src/aa_device_MSD.cu:    cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost); 
src/aa_device_MSD.cu:  call_kernel_MSD_GPU_final_nonregular(MSD_conf->final_gridSize,MSD_conf->final_blockSize,
src/aa_device_MSD.cu:  cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost); 
src/aa_device_MSD.cu:  cudaMalloc((void **) &d_temp, conf.nBlocks_total*MSD_PARTIAL_SIZE*sizeof(float));
src/aa_device_MSD.cu:  cudaFree(d_temp);
src/aa_device_MSD.cu:  call_kernel_MSD_GPU_calculate_partials_2d_and_minmax(MSD_conf->partials_gridSize,MSD_conf->partials_blockSize, d_input, &d_temp[MSD_conf->address*MSD_PARTIAL_SIZE], MSD_conf->nSteps.y, (int) MSD_conf->nTimesamples, (int)MSD_conf->offset);
src/aa_device_MSD.cu:  call_kernel_MSD_GPU_final_regular(MSD_conf->final_gridSize,MSD_conf->final_blockSize,
src/aa_device_MSD.cu:  cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost); 
src/aa_device_MSD.cu:    call_kernel_MSD_GPU_final_nonregular(MSD_conf->final_gridSize,MSD_conf->final_blockSize,
src/aa_device_MSD.cu:    cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost); 
src/aa_device_MSD.cu:  cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost); 
src/aa_device_MSD.cu:  cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost); 
src/aa_device_MSD.cu:  cudaError_t e = cudaMalloc((void **) &d_MSD_workarea, conf.nBlocks_total*MSD_PARTIAL_SIZE*sizeof(float));
src/aa_device_MSD.cu:  if(e != cudaSuccess) {
src/aa_device_MSD.cu:    LOG(log_level::error, "Could not cudaMemcpy in aa_device_MSD.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD.cu:  e = cudaFree(d_MSD_workarea);
src/aa_device_MSD.cu:  if(e != cudaSuccess) {
src/aa_device_MSD.cu:    LOG(log_level::error, "Could not cudaMemcpy in aa_device_MSD.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD.cu:  cudaError_t e = cudaMalloc((void **) &d_MSD_workarea, conf.nBlocks_total*MSD_PARTIAL_SIZE*sizeof(float));
src/aa_device_MSD.cu:  if(e != cudaSuccess) {
src/aa_device_MSD.cu:    LOG(log_level::error, "Could not cudaMemcpy in aa_device_MSD.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD.cu:  e = cudaFree(d_MSD_workarea);
src/aa_device_MSD.cu:  if(e != cudaSuccess) {
src/aa_device_MSD.cu:    LOG(log_level::error, "Could not cudaMemcpy in aa_device_MSD.cu (" + std::string(cudaGetErrorString(e)) + ")");
src/aa_device_MSD.cu:  cudaMalloc((void **) &d_output, GridSize_x*GridSize_y*3*sizeof(float));
src/aa_device_MSD.cu:  cudaFree(d_output);
src/aa_device_stats_kernel.cu:#include <cuda.h>
src/aa_device_stats_kernel.cu:#include <cuda_runtime.h>
src/aa_device_stats_kernel.cu:  void call_kernel_stats_kernel(const dim3 &block_size, const dim3 &grid_size, const int &smem_bytes, const cudaStream_t &stream,

```
