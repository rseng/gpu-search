# https://github.com/ewanbarr/peasoup

```console
example_output/overview.xml:  <cuda_device_parameters>
example_output/overview.xml:    <cuda_device id='0'>
example_output/overview.xml:    </cuda_device>
example_output/overview.xml:    <cuda_device id='1'>
example_output/overview.xml:    </cuda_device>
example_output/overview.xml:  </cuda_device_parameters>
Makefile:INCLUDE  = -I$(INCLUDE_DIR) -I$(THRUST_DIR) -I${DEDISP_DIR}/include -I${CUDA_DIR}/include -I./tclap
Makefile:LIBS = -L$(CUDA_DIR)/lib64 -lcudart -L${DEDISP_DIR}/lib -ldedisp -lcufft -lpthread -lnvToolsExt
Makefile.PSG.inc:#cuda setup
Makefile.PSG.inc:CUDA_DIR   = /shared/apps/cuda/CUDA-v6.0.26
Makefile.PSG.inc:THRUST_DIR = $(CUDA_DIR)/include
Makefile.PSG.inc:NVCC      = $(CUDA_DIR)/bin/nvcc -lineinfo
README.md:C++/CUDA GPU pulsar searching library 
include/kernels/defaults.h:#include "cuda.h"
include/kernels/defaults.h:cudaDeviceProp properties;
include/kernels/defaults.h:cudaGetDeviceProperties(&properties,0);
include/kernels/kernels.h:#include <thrust/system/cuda/vector.h>
include/kernels/kernels.h:#include <thrust/system/cuda/execution_policy.h>
include/kernels/kernels.h:        // create a new one with cuda::malloc
include/kernels/kernels.h:        // throw if cuda::malloc can't satisfy the request
include/kernels/kernels.h:	    // allocate memory and convert cuda::pointer to raw pointer
include/kernels/kernels.h:	    result = thrust::cuda::malloc<char>(num_bytes).get();
include/kernels/kernels.h:        // transform the pointer to cuda::pointer before calling cuda::free
include/kernels/kernels.h:	thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
include/kernels/kernels.h:        // transform the pointer to cuda::pointer before calling cuda::free
include/kernels/kernels.h:	thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
include/kernels/kernels.h://------GPU fold optimisation related-----//
include/kernels/kernels.h:  HD_PRIOR_GPU_ERROR,
include/kernels/kernels.h:  HD_INTERNAL_GPU_ERROR,
include/kernels/kernels.h:float GPU_rms(T* d_collection,
include/kernels/kernels.h:float GPU_mean(T* d_collection,
include/kernels/kernels.h:void GPU_fill(T* start,
include/kernels/kernels.h:void GPU_remove_baseline(T* d_collection, 
include/data_types/fourierseries.hpp:#include "cuda.h"
include/data_types/fourierseries.hpp:  \brief Subclass for handling of frequency series on the GPU.
include/data_types/fourierseries.hpp:  GPU memory for storage of frequency series data on the GPU.
include/data_types/timeseries.hpp:#include "cuda.h"
include/data_types/timeseries.hpp:  \brief TimeSeries subclass for encapsulating on-GPU timeseries.
include/data_types/timeseries.hpp:  data stored on the GPU. The lifetime of the GPU memory buffer
include/data_types/timeseries.hpp:    is allocated in GPU memory.
include/data_types/timeseries.hpp:    space in GPU RAM and copies the data from the TimeSeries instance
include/data_types/timeseries.hpp:    to the GPU. Data are automatically converted from type of input
include/data_types/timeseries.hpp:    GPU_remove_baseline<OnDeviceType>(this->data_ptr, static_cast<std::size_t>(nsamps));
include/data_types/timeseries.hpp:      Note: GPU_fill is used rather than cudaMemset as it
include/data_types/timeseries.hpp:    GPU_fill(this->data_ptr+start,this->data_ptr+end,value);
include/data_types/timeseries.hpp:  the GPU used for conversion of data from host type to device type.
include/data_types/timeseries.hpp:  GPU memory between different dm trials being passed to the GPU.
include/data_types/timeseries.hpp:  OnHostType* copy_buffer; //GPU memory buffer
include/data_types/timeseries.hpp:    GPU memory buffer. Data undegoes automatic type conversion.
include/data_types/timeseries.hpp:    \note Allocated GPU memory is freed here.
include/data_types/folded.hpp:#include "cuda.h"
include/transforms/dedisperser.hpp:  unsigned int num_gpus;
include/transforms/dedisperser.hpp:  Dedisperser(Filterbank& filterbank, unsigned int num_gpus=1)
include/transforms/dedisperser.hpp:    :filterbank(filterbank), num_gpus(num_gpus)
include/transforms/dedisperser.hpp:						  num_gpus);
include/transforms/correlator.hpp:#include "cuda.h"
include/transforms/folder.hpp:#include "cuda.h"
include/transforms/whitener.hpp:    cudaError_t error;
include/transforms/whitener.hpp:    error = cudaMalloc((void**)&buffer_5,nbins/5);
include/transforms/whitener.hpp:    ErrorChecker::check_cuda_error(error);
include/transforms/whitener.hpp:    error = cudaMalloc((void**)&buffer_25,nbins/25);
include/transforms/whitener.hpp:    ErrorChecker::check_cuda_error(error);
include/transforms/whitener.hpp:    error = cudaMalloc((void**)&buffer_125,nbins/125);
include/transforms/whitener.hpp:    ErrorChecker::check_cuda_error(error);
include/transforms/whitener.hpp:    error = cudaMalloc((void**)&median_array,nbins);
include/transforms/whitener.hpp:    ErrorChecker::check_cuda_error(error);
include/transforms/whitener.hpp:    cudaFree(buffer_5);
include/transforms/whitener.hpp:    cudaFree(buffer_25);
include/transforms/whitener.hpp:    cudaFree(buffer_125);
include/transforms/whitener.hpp:    cudaFree(median_array);
include/transforms/transpose.hpp: * This file contains a CUDA implementation of the array transpose operation.
include/transforms/transpose.hpp: * NVIDIA CUDA SDK.
include/transforms/transpose.hpp:namespace cuda_specs {
include/transforms/transpose.hpp://#if __CUDA_ARCH__ < 200
include/transforms/transpose.hpp:				   cudaStream_t stream=0);
include/transforms/transpose.hpp:				   cudaStream_t stream=0) {
include/transforms/transpose.hpp:typedef unsigned int gpu_size_t;
include/transforms/transpose.hpp:		      gpu_size_t width, gpu_size_t height,
include/transforms/transpose.hpp:		      gpu_size_t in_stride, gpu_size_t out_stride,
include/transforms/transpose.hpp:		      gpu_size_t block_count_x,
include/transforms/transpose.hpp:		      gpu_size_t block_count_y,
include/transforms/transpose.hpp:		      gpu_size_t log2_gridDim_y)
include/transforms/transpose.hpp:	gpu_size_t blockIdx_x, blockIdx_y;
include/transforms/transpose.hpp:		gpu_size_t bid = blockIdx.x + gridDim.x*blockIdx.y;
include/transforms/transpose.hpp:	gpu_size_t index_in_x = blockIdx_x * Transpose<T>::TILE_DIM + threadIdx.x;
include/transforms/transpose.hpp:	gpu_size_t index_in_y = blockIdx_y * Transpose<T>::TILE_DIM + threadIdx.y;
include/transforms/transpose.hpp:	gpu_size_t index_in = index_in_x + (index_in_y)*in_stride;
include/transforms/transpose.hpp:	for( gpu_size_t i=0; i<Transpose<T>::TILE_DIM; i+=Transpose<T>::BLOCK_ROWS ) {
include/transforms/transpose.hpp:	gpu_size_t index_out_x = blockIdx_y * Transpose<T>::TILE_DIM + threadIdx.x;
include/transforms/transpose.hpp:	gpu_size_t index_out_y = blockIdx_x * Transpose<T>::TILE_DIM + threadIdx.y;
include/transforms/transpose.hpp:	gpu_size_t index_out = index_out_x + (index_out_y)*out_stride;
include/transforms/transpose.hpp:	for( gpu_size_t i=0; i<Transpose<T>::TILE_DIM; i+=Transpose<T>::BLOCK_ROWS ) {
include/transforms/transpose.hpp:			     cudaStream_t stream)
include/transforms/transpose.hpp:	size_t max_grid_dim = round_down_pow2((size_t)cuda_specs::MAX_GRID_DIMENSION);
include/transforms/transpose.hpp:	// Partition the grid into chunks that the GPU can accept at once
include/transforms/transpose.hpp:				// Run the CUDA kernel
include/transforms/transpose.hpp:				// Run the CUDA kernel
include/transforms/transpose.hpp:			cudaStreamSynchronize(stream);
include/transforms/transpose.hpp:			cudaError_t error = cudaGetLastError();
include/transforms/transpose.hpp:			if( error != cudaSuccess ) {
include/transforms/transpose.hpp:					std::string("Transpose: CUDA error in kernel: ") +
include/transforms/transpose.hpp:					cudaGetErrorString(error));
include/transforms/median_filter.h:  HD_PRIOR_GPU_ERROR,
include/transforms/median_filter.h:  HD_INTERNAL_GPU_ERROR,
include/transforms/ffter.hpp:#include "cuda.h"
include/transforms/ffter.hpp:      cudaThreadSynchronize();
include/utils/output_stats.hpp:#include "cuda.h"
include/utils/output_stats.hpp:  void add_gpu_info(std::vector<int>& device_idxs){
include/utils/output_stats.hpp:    XML::Element gpu_info("cuda_device_parameters");
include/utils/output_stats.hpp:    cudaRuntimeGetVersion(&runtime_version);
include/utils/output_stats.hpp:    cudaDriverGetVersion(&driver_version);
include/utils/output_stats.hpp:    gpu_info.append(XML::Element("runtime",runtime_version));
include/utils/output_stats.hpp:    gpu_info.append(XML::Element("driver",driver_version));
include/utils/output_stats.hpp:    cudaDeviceProp properties;
include/utils/output_stats.hpp:      XML::Element device("cuda_device");
include/utils/output_stats.hpp:      cudaGetDeviceProperties(&properties,device_idxs[ii]);
include/utils/output_stats.hpp:      gpu_info.append(device);
include/utils/output_stats.hpp:    root.append(gpu_info);
include/utils/utils.hpp:#include "cuda.h"
include/utils/utils.hpp:    cudaMalloc((void**)ptr, sizeof(T)*units);
include/utils/utils.hpp:    ErrorChecker::check_cuda_error("Error from device_malloc");
include/utils/utils.hpp:    cudaMallocHost((void**)ptr, sizeof(T)*units);
include/utils/utils.hpp:    ErrorChecker::check_cuda_error("Error from host_malloc");
include/utils/utils.hpp:    cudaFree(ptr);
include/utils/utils.hpp:    ErrorChecker::check_cuda_error("Error from device_free");
include/utils/utils.hpp:    cudaFreeHost((void*) ptr);
include/utils/utils.hpp:    ErrorChecker::check_cuda_error("Error from host_free.");
include/utils/utils.hpp:    cudaMemcpy((void*)d_ptr, (void*) h_ptr, sizeof(T)*units, cudaMemcpyHostToDevice);
include/utils/utils.hpp:    ErrorChecker::check_cuda_error("Error from h2dcpy");
include/utils/utils.hpp:    cudaMemcpy((void*) h_ptr,(void*) d_ptr,sizeof(T)*units,cudaMemcpyDeviceToHost);
include/utils/utils.hpp:    ErrorChecker::check_cuda_error("Error from d2hcpy");
include/utils/utils.hpp:    cudaMemcpy(d_ptr_dst,d_ptr_src,sizeof(T)*units,cudaMemcpyDeviceToDevice);
include/utils/utils.hpp:    ErrorChecker::check_cuda_error("Error from d2dcpy");
include/utils/utils.hpp:  static int gpu_count(){
include/utils/utils.hpp:    cudaGetDeviceCount(&count);
include/utils/stats.hpp:    return GPU_rms<T>(ptr,nsamps,first_samp);
include/utils/stats.hpp:    return GPU_mean<T>(ptr,nsamps,first_samp);
include/utils/cmdline.hpp:      TCLAP::CmdLine cmd("Peasoup - a GPU pulsar search pipeline", ' ', "1.0");
include/utils/cmdline.hpp:                                               "The number of GPUs to use",
include/utils/cmdline.hpp:      TCLAP::CmdLine cmd("Peasoup/FFAster extension - a GPU FFA pulsar search pipeline", ' ', "1.0");
include/utils/cmdline.hpp:                                               "The number of GPUs to use",
include/utils/cmdline.hpp:						 "The number of CUDA streams to use",
include/utils/exceptions.hpp:#include "cuda.h"
include/utils/exceptions.hpp:#include "cuda_runtime.h"
include/utils/exceptions.hpp:  static void check_cuda_error(std::string msg="Unspecified location"){
include/utils/exceptions.hpp:    cudaDeviceSynchronize();
include/utils/exceptions.hpp:    cudaError_t error = cudaGetLastError();
include/utils/exceptions.hpp:    if (error!=cudaSuccess){
include/utils/exceptions.hpp:      error_msg << "CUDA failed with error: "
include/utils/exceptions.hpp:                << cudaGetErrorString(error) << std::endl 
include/utils/exceptions.hpp:  static void check_cuda_error(cudaError_t error){
include/utils/exceptions.hpp:    check_cuda_error(error,"");
peasoup_32/Makefile.PSG.inc:#cuda setup
peasoup_32/Makefile.PSG.inc:CUDA_DIR   = /shared/apps/cuda/CUDA-v6.0.26
peasoup_32/Makefile.PSG.inc:THRUST_DIR = $(CUDA_DIR)/include
peasoup_32/Makefile.PSG.inc:NVCC      = $(CUDA_DIR)/bin/nvcc -lineinfo
Makefile.inc:#cuda setup
Makefile.inc:CUDA_DIR   = /usr/local/cuda/
Makefile.inc:THRUST_DIR = /usr/local/cuda/
Makefile.inc:NVCC      = $(CUDA_DIR)/bin/nvcc
src/rednoise_test.cpp:#include "cuda.h"
src/rednoise_test.cpp:  ErrorChecker::check_cuda_error();
src/pipeline_single.cpp:#include "cuda.h"
src/hcfft.cpp:#include "cuda.h"
src/coincidencer.cpp:#include "cuda.h"
src/coincidencer.cpp:      TCLAP::CmdLine cmd("Peasoup - a GPU pulsar search pipeline", ' ', "1.0");
src/kernels.cu:#include "cuda.h"
src/kernels.cu:#include <thrust/system/cuda/vector.h>
src/kernels.cu:#include <thrust/system/cuda/execution_policy.h>
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_harmonic_sum");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_form_power_series");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_resampleII");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_resample");
src/kernels.cu://defined here as (although Thrust based) requires CUDA functors
src/kernels.cu:  int num_copied = thrust::copy_if(thrust::cuda::par(policy), zipped_iter, zipped_iter+n-start_index,
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_find_peaks;");
src/kernels.cu:float GPU_rms(T* d_collection,int nsamps, int min_bin)
src/kernels.cu:float GPU_mean(T* d_collection,int nsamps, int min_bin)
src/kernels.cu:  cudaThreadSynchronize();
src/kernels.cu:void GPU_remove_baseline(T* d_collection, int nsamps){
src/kernels.cu:    mean = GPU_mean(d_collection, nsamps, 0);
src/kernels.cu:void GPU_fill(T* start, T* end, T value){
src/kernels.cu:  ErrorChecker::check_cuda_error("Error in GPU_fill");
src/kernels.cu:template void GPU_fill<float>(float*, float*, float);
src/kernels.cu:template float GPU_rms<float>(float*,int,int);
src/kernels.cu:template float GPU_mean<float>(float*,int,int);
src/kernels.cu:template void GPU_remove_baseline<float>(float*,int);
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_normalise");
src/kernels.cu:    mean = GPU_mean(d_power_spectrum,nsamp,min_bin);
src/kernels.cu:    rms = GPU_rms(d_power_spectrum,nsamp,min_bin);
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_normalise_spectrum");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_fold_timeseries.");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from thrust::max_element in device_argmax");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_real_to_complex");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_get_absolute_value");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_generate_shift_array");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_generate_template_array");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_multiply_by_shift");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_collapse_subints");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_multiply_by_templates");
src/kernels.cu:  ErrorChecker::check_cuda_error();
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_zap_birdies");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_coincidencer");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_conjugate");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_cuCmulf_inplace");
src/kernels.cu:  ErrorChecker::check_cuda_error("Error from device_conversion");
src/pipeline.cpp:#include "cuda.h"
src/pipeline.cpp:      TCLAP::CmdLine cmd("Peasoup - a GPU pulsar search pipeline", ' ', "1.0");
src/pipeline.cpp:					       "The number of GPUs to use",
src/resampling_test.cpp:#include "cuda.h"
src/folder_test.cpp:#include "cuda.h"
src/folder_test.cpp:  cudaError_t error;
src/folder_test.cpp:  error = cudaMalloc((void**)&folded_buffer, sizeof(float)*size);
src/folder_test.cpp:  ErrorChecker::check_cuda_error(error);
src/folder_test.cpp:  error = cudaMalloc((void**)&fft_out, sizeof(cufftComplex)*nints*nbins);
src/folder_test.cpp:  cudaMemcpy(temp,folded_buffer,nints*nbins*sizeof(float),cudaMemcpyDeviceToHost);
src/folder_test.cpp:  ErrorChecker::check_cuda_error();
src/pipeline_multi.cu:#include "cuda.h"
src/pipeline_multi.cu:    cudaSetDevice(device);
src/pipeline_multi.cu:  int nthreads = std::min(Utils::gpu_count(),args.max_num_threads);
src/pipeline_multi.cu:  /* Could do a check on the GPU memory usage here */
src/pipeline_multi.cu:  stats.add_gpu_info(device_idxs);
src/harmonic_sum_test.cpp:#include "cuda.h"
src/dedisp_test.cpp:#include "cuda.h"
src/dedisp_test.cpp:  Utils::dump_host_buffer<unsigned char>(trials.get_data(),nsamps*ntrials,"/lustre/projects/p002_swin/ebarr/GPUSEEK_TESTS/dedispersed_tim_dump.bin");
src/dedisp_test.cpp:					 "/lustre/projects/p002_swin/ebarr/GPUSEEK_TESTS/manual_extraction_tim.bin");
src/dedisp_test.cpp:  Utils::dump_host_buffer<unsigned char>(tim.get_data(),fft_size,"/lustre/projects/p002_swin/ebarr/GPUSEEK_TESTS/extracted_tim.bin");

```
