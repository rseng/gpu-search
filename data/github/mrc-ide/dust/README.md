# https://github.com/mrc-ide/dust

```console
inst/template/dust_methods.hpp:                         cpp11::sexp gpu_config, cpp11::sexp ode_control);
inst/template/dust.cpp:cpp11::sexp dust_{{name}}_gpu_info() {
inst/template/dust.cpp:  return dust::gpu::r::gpu_info();
inst/template/math.hpp:#include "dust/random/cuda_compatibility.hpp"
inst/template/math.hpp:// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html
inst/template/math.hpp:#ifdef __CUDA_ARCH__
inst/template/math.hpp:#ifdef __CUDA_ARCH__
inst/template/math.hpp:#ifdef __CUDA_ARCH__
inst/template/binomial_gamma_tables.hpp:#include "dust/random/cuda_compatibility.hpp"
inst/template/normal_ziggurat_tables.hpp:#include "dust/random/cuda_compatibility.hpp"
inst/template/dust_methods.cpp:                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
inst/template/dust_methods.cpp:                                        gpu_config, ode_control);
inst/template/Makevars.cuda:PKG_LIBS = {{cuda$lib_flags}} -lcudart $(SHLIB_OPENMP_CXXFLAGS)
inst/template/Makevars.cuda:NVCC_FLAGS = -std=c++14 {{cuda$nvcc_flags}} -I. -I$(R_INCLUDE_DIR) -I{{path_dust_include}} {{cuda$cub_include}} $(CLINK_CPPFLAGS) {{cuda$gencode}} -Xcompiler -fPIC -Xcompiler -fopenmp -x cu
inst/template/dust.hpp:cpp11::sexp dust_{{name}}_gpu_info();
inst/template/dust.R.template:    gpu_config_ = NULL,
inst/template/dust.R.template:    ##' a a GPU.
inst/template/dust.R.template:    ##' @param gpu_config GPU configuration, typically an integer
inst/template/dust.R.template:    ##' indicating the device to use, where the model has GPU support.
inst/template/dust.R.template:    ##' that CUDA numbers devices from 0, so that '0' is the first device,
inst/template/dust.R.template:    ##' and so on). See the method `$gpu_info()` for available device ids;
inst/template/dust.R.template:    ##' `{{name}}$public_methods$gpu_info()`.
inst/template/dust.R.template:                          gpu_config = NULL, ode_control = NULL) {
inst/template/dust.R.template:      if (is.null(gpu_config)) {
inst/template/dust.R.template:        private$methods_ <- {{methods_gpu}}
inst/template/dust.R.template:                        n_threads, seed, deterministic, gpu_config, ode_control)
inst/template/dust.R.template:      private$gpu_config_ <- res[[4L]]
inst/template/dust.R.template:    ##' "CUDA" support, in which case it will react to the `device`
inst/template/dust.R.template:    ##' as `{{name}}$public_methods$has_gpu_support()`
inst/template/dust.R.template:    ##' @param fake_gpu Logical, indicating if we count as `TRUE`
inst/template/dust.R.template:    ##'   models that run on the "fake" GPU (i.e., using the GPU
inst/template/dust.R.template:    has_gpu_support = function(fake_gpu = FALSE) {
inst/template/dust.R.template:      if (fake_gpu) {
inst/template/dust.R.template:        {{has_gpu_support}}
inst/template/dust.R.template:        dust_{{target}}_{{name}}_capabilities()[["gpu"]]
inst/template/dust.R.template:    ##' Check if the model is running on a GPU
inst/template/dust.R.template:    ##' @param fake_gpu Logical, indicating if we count as `TRUE`
inst/template/dust.R.template:    ##'   models that run on the "fake" GPU (i.e., using the GPU
inst/template/dust.R.template:    uses_gpu = function(fake_gpu = FALSE) {
inst/template/dust.R.template:      real_gpu <- private$gpu_config_$real_gpu
inst/template/dust.R.template:      !is.null(real_gpu) && (fake_gpu || real_gpu)
inst/template/dust.R.template:    ##' Return information about GPU devices, if the model
inst/template/dust.R.template:    ##' has been compiled with CUDA/GPU support. This can be called as a
inst/template/dust.R.template:    ##' static method by running `{{name}}$public_methods$gpu_info()`.
inst/template/dust.R.template:    ##' If run from a GPU enabled object, it will also have an element
inst/template/dust.R.template:    gpu_info = function() {
inst/template/dust.R.template:      ret <- dust_{{name}}_gpu_info()
inst/template/dust.R.template:      if (ret$has_cuda && exists("private", parent, inherits = FALSE)) {
inst/template/dust.R.template:        ret$config <- private$gpu_config_
inst/include/dust/dust_cpu.hpp:#include "dust/gpu/cuda.hpp"
inst/include/dust/dust_cpu.hpp:  // TODO: fix this elsewhere, perhaps (see also cuda/dust_gpu.hpp)
inst/include/dust/r/gpu.hpp:#ifndef DUST_R_GPU_HPP
inst/include/dust/r/gpu.hpp:#define DUST_R_GPU_HPP
inst/include/dust/r/gpu.hpp:#include "dust/gpu/launch_control.hpp"
inst/include/dust/r/gpu.hpp:#include "dust/r/gpu_info.hpp"
inst/include/dust/r/gpu.hpp:namespace gpu {
inst/include/dust/r/gpu.hpp:  const int device_id_max = dust::gpu::devices_count() - 1;
inst/include/dust/r/gpu.hpp:inline dust::gpu::gpu_config gpu_config(cpp11::sexp r_gpu_config) {
inst/include/dust/r/gpu.hpp:  cpp11::sexp r_device_id = r_gpu_config;
inst/include/dust/r/gpu.hpp:  if (TYPEOF(r_gpu_config) == VECSXP) {
inst/include/dust/r/gpu.hpp:    cpp11::list r_gpu_config_l = cpp11::as_cpp<cpp11::list>(r_gpu_config);
inst/include/dust/r/gpu.hpp:    r_device_id = r_gpu_config_l["device_id"]; // could error if missing?
inst/include/dust/r/gpu.hpp:    cpp11::sexp r_run_block_size = r_gpu_config_l["run_block_size"];
inst/include/dust/r/gpu.hpp:  return dust::gpu::gpu_config(check_device_id(r_device_id), run_block_size);
inst/include/dust/r/gpu.hpp:cpp11::sexp gpu_config_as_sexp(const dust::gpu::gpu_config& config) {
inst/include/dust/r/gpu.hpp:  return cpp11::writable::list({"real_gpu"_nm = config.real_gpu_,
inst/include/dust/r/dust.hpp:#include "dust/gpu/dust_gpu.hpp"
inst/include/dust/r/dust.hpp:#include "dust/gpu/filter.hpp"
inst/include/dust/r/dust.hpp:#include "dust/r/gpu.hpp"
inst/include/dust/r/dust.hpp:                           cpp11::sexp r_gpu_config,
inst/include/dust/r/dust.hpp:cpp11::list dust_gpu_alloc(cpp11::list r_pars, bool pars_multi,
inst/include/dust/r/dust.hpp:                           cpp11::sexp r_gpu_config,
inst/include/dust/r/dust.hpp:  const dust::gpu::gpu_config gpu_config =
inst/include/dust/r/dust.hpp:    dust::gpu::r::gpu_config(r_gpu_config);
inst/include/dust/r/dust.hpp:    cpp11::stop("Deterministic models not supported on gpu");
inst/include/dust/r/dust.hpp:  dust_gpu<T> *d = nullptr;
inst/include/dust/r/dust.hpp:    d = new dust_gpu<T>(inputs.pars, inputs.time, inputs.n_particles,
inst/include/dust/r/dust.hpp:                        inputs.shape, gpu_config);
inst/include/dust/r/dust.hpp:    d = new dust_gpu<T>(inputs.pars[0], inputs.time, inputs.n_particles,
inst/include/dust/r/dust.hpp:                        inputs.n_threads, inputs.seed, gpu_config);
inst/include/dust/r/dust.hpp:  cpp11::external_pointer<dust_gpu<T>> ptr(d, true, false);
inst/include/dust/r/dust.hpp:  cpp11::sexp ret_r_gpu_config =
inst/include/dust/r/dust.hpp:    dust::gpu::r::gpu_config_as_sexp(gpu_config);
inst/include/dust/r/dust.hpp:  return cpp11::writable::list({ptr, info, r_shape, ret_r_gpu_config, r_ode_control});
inst/include/dust/r/dust.hpp:                           cpp11::sexp r_gpu_config,
inst/include/dust/r/dust.hpp:  return cpp11::writable::list({ptr, info, r_shape, r_gpu_config, r_ctl});
inst/include/dust/r/dust.hpp:  bool gpu = true;
inst/include/dust/r/dust.hpp:  bool gpu = false;
inst/include/dust/r/dust.hpp:                                "gpu"_nm = gpu,
inst/include/dust/r/gpu_info.hpp:#ifndef DUST_R_GPU_INFO_HPP
inst/include/dust/r/gpu_info.hpp:#define DUST_R_GPU_INFO_HPP
inst/include/dust/r/gpu_info.hpp:#include "dust/gpu/gpu_info.hpp"
inst/include/dust/r/gpu_info.hpp:// this point for our test program (we want to get the CUDA version
inst/include/dust/r/gpu_info.hpp:namespace gpu {
inst/include/dust/r/gpu_info.hpp:inline cpp11::sexp gpu_info() {
inst/include/dust/r/gpu_info.hpp:  cpp11::writable::logicals has_cuda(1);
inst/include/dust/r/gpu_info.hpp:      cudaDeviceProp properties;
inst/include/dust/r/gpu_info.hpp:      CUDA_CALL(cudaGetDeviceProperties(&properties, i));
inst/include/dust/r/gpu_info.hpp:  has_cuda[0] = true;
inst/include/dust/r/gpu_info.hpp:  cpp11::writable::integers cuda_version_int(3);
inst/include/dust/r/gpu_info.hpp:  cuda_version_int[0] = __CUDACC_VER_MAJOR__;
inst/include/dust/r/gpu_info.hpp:  cuda_version_int[1] = __CUDACC_VER_MINOR__;
inst/include/dust/r/gpu_info.hpp:  cuda_version_int[2] = __CUDACC_VER_BUILD__;
inst/include/dust/r/gpu_info.hpp:  cpp11::writable::list cuda_version({cuda_version_int});
inst/include/dust/r/gpu_info.hpp:  cuda_version.attr("class") = "numeric_version";
inst/include/dust/r/gpu_info.hpp:  has_cuda[0] = false;
inst/include/dust/r/gpu_info.hpp:  cpp11::sexp cuda_version = R_NilValue;
inst/include/dust/r/gpu_info.hpp:  return cpp11::writable::list({"has_cuda"_nm = has_cuda,
inst/include/dust/r/gpu_info.hpp:                                "cuda_version"_nm = cuda_version,
inst/include/dust/random/uniform.hpp:#ifndef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#include "dust/random/cuda_compatibility.hpp"
inst/include/dust/random/math.hpp:// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/math.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/generator.hpp:#include "dust/random/cuda_compatibility.hpp"
inst/include/dust/random/poisson.hpp:    // do nothing, but leave this branch in to help the GPU
inst/include/dust/random/poisson.hpp:#ifndef __CUDA_ARCH__
inst/include/dust/random/gamma.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/gamma.hpp:  static_assert("gamma() not implemented for GPU targets");
inst/include/dust/random/binomial_gamma_tables.hpp:#include "dust/random/cuda_compatibility.hpp"
inst/include/dust/random/cauchy.hpp:#ifndef __CUDA_ARCH__
inst/include/dust/random/hypergeometric.hpp:  // behaving well on a GPU. Unlikely to be a lot of work, but better
inst/include/dust/random/hypergeometric.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/hypergeometric.hpp:  static_assert("hypergeomeric() not implemented for GPU targets");
inst/include/dust/random/normal_ziggurat_tables.hpp:#include "dust/random/cuda_compatibility.hpp"
inst/include/dust/random/binomial.hpp:#ifndef __CUDA_ARCH__
inst/include/dust/random/normal.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/normal_ziggurat.hpp:  // possible that a different number will be better on the GPU. Once
inst/include/dust/random/density.hpp:#include "dust/random/cuda_compatibility.hpp"
inst/include/dust/random/density.hpp:#ifndef __CUDA_ARCH__
inst/include/dust/random/density.hpp:#ifndef __CUDA_ARCH__
inst/include/dust/random/density.hpp:#ifndef __CUDA_ARCH__
inst/include/dust/random/density.hpp:#ifndef __CUDA_ARCH__
inst/include/dust/random/exponential.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/xoshiro_state.hpp:#include "dust/random/cuda_compatibility.hpp"
inst/include/dust/random/numeric.hpp:#include "dust/random/cuda_compatibility.hpp"
inst/include/dust/random/numeric.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/numeric.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/numeric.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/numeric.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/cuda_compatibility.hpp:#ifndef DUST_RANDOM_CUDA_COMPATIBILITY_HPP
inst/include/dust/random/cuda_compatibility.hpp:#define DUST_RANDOM_CUDA_COMPATIBILITY_HPP
inst/include/dust/random/cuda_compatibility.hpp:// *  __CUDA_ARCH__ is defined: we're compiling under nvcc generating
inst/include/dust/random/cuda_compatibility.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/cuda_compatibility.hpp:// We cannot throw errors in GPU code, we can only send a trap signal,
inst/include/dust/random/cuda_compatibility.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/nbinomial.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/random/nbinomial.hpp:  static_assert("nbinomial() not implemented for GPU targets");
inst/include/dust/gpu/device_resample.hpp:#ifndef DUST_GPU_DEVICE_RESAMPLE_HPP
inst/include/dust/gpu/device_resample.hpp:#define DUST_GPU_DEVICE_RESAMPLE_HPP
inst/include/dust/gpu/device_resample.hpp:#include "dust/gpu/launch_control.hpp"
inst/include/dust/gpu/device_resample.hpp:#include "dust/gpu/kernels.hpp"
inst/include/dust/gpu/device_resample.hpp:                         const dust::gpu::launch_control_dust& cuda_pars,
inst/include/dust/gpu/device_resample.hpp:                         dust::gpu::cuda_stream& kernel_stream,
inst/include/dust/gpu/device_resample.hpp:                         dust::gpu::cuda_stream& resample_stream,
inst/include/dust/gpu/device_resample.hpp:                         dust::gpu::device_state<real_type, rng_state_type>& device_state,
inst/include/dust/gpu/device_resample.hpp:                         dust::gpu::device_array<real_type>& weights,
inst/include/dust/gpu/device_resample.hpp:                         dust::gpu::device_scan_state<real_type>& scan) {
inst/include/dust/gpu/device_resample.hpp:    dust::gpu::find_intervals<real_type><<<cuda_pars.interval.block_count,
inst/include/dust/gpu/device_resample.hpp:                                         cuda_pars.interval.block_size,
inst/include/dust/gpu/device_resample.hpp:    dust::gpu::find_intervals<real_type>(
inst/include/dust/gpu/device_resample.hpp:    dust::gpu::scatter_device<real_type><<<cuda_pars.scatter.block_count,
inst/include/dust/gpu/device_resample.hpp:                                   cuda_pars.scatter.block_size,
inst/include/dust/gpu/device_resample.hpp:    dust::gpu::scatter_device<real_type>(
inst/include/dust/gpu/device_state.hpp:#ifndef DUST_GPU_DEVICE_STATE_HPP
inst/include/dust/gpu/device_state.hpp:#define DUST_GPU_DEVICE_STATE_HPP
inst/include/dust/gpu/device_state.hpp:#include "dust/gpu/cuda.hpp"
inst/include/dust/gpu/device_state.hpp:#include "dust/gpu/types.hpp"
inst/include/dust/gpu/device_state.hpp:namespace gpu {
inst/include/dust/gpu/device_state.hpp:// Each parameter set must be run in a single CUDA block for this to work
inst/include/dust/gpu/types.hpp:#ifndef DUST_GPU_TYPES_HPP
inst/include/dust/gpu/types.hpp:#define DUST_GPU_TYPES_HPP
inst/include/dust/gpu/types.hpp:#include "dust/gpu/filter_kernels.hpp"
inst/include/dust/gpu/types.hpp:#include "dust/gpu/utils.hpp"
inst/include/dust/gpu/types.hpp:namespace gpu {
inst/include/dust/gpu/types.hpp:    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
inst/include/dust/gpu/types.hpp:    CUDA_CALL(cudaMemset(data_, 0, size_ * sizeof(T)));
inst/include/dust/gpu/types.hpp:    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
inst/include/dust/gpu/types.hpp:    CUDA_CALL(cudaMemcpy(data_, data.data(), size_ * sizeof(T),
inst/include/dust/gpu/types.hpp:                         cudaMemcpyDefault));
inst/include/dust/gpu/types.hpp:    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
inst/include/dust/gpu/types.hpp:    CUDA_CALL(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
inst/include/dust/gpu/types.hpp:                         cudaMemcpyDefault));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaFree(data_));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
inst/include/dust/gpu/types.hpp:                           cudaMemcpyDefault));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaFree(data_));
inst/include/dust/gpu/types.hpp:    CUDA_CALL_NOTHROW(cudaFree(data_));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMemcpyAsync(dst.data(), data_, dst.size() * sizeof(T),
inst/include/dust/gpu/types.hpp:                          cudaMemcpyDefault));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMemcpy(dst.data(), data_, dst.size() * sizeof(T),
inst/include/dust/gpu/types.hpp:                          cudaMemcpyDefault));
inst/include/dust/gpu/types.hpp:  void get_array(T * dst, cuda_stream& stream, const bool async = false) const {
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMemcpyAsync(dst, data_, size() * sizeof(T),
inst/include/dust/gpu/types.hpp:                          cudaMemcpyDefault, stream.stream()));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMemcpy(dst, data_, size() * sizeof(T),
inst/include/dust/gpu/types.hpp:                          cudaMemcpyDefault));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMemcpyAsync(data_ + dst_offset, src,
inst/include/dust/gpu/types.hpp:                          src_size * sizeof(T), cudaMemcpyDefault));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMemcpy(data_ + dst_offset, src,
inst/include/dust/gpu/types.hpp:                          src_size * sizeof(T), cudaMemcpyDefault));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMemcpyAsync(data_, src.data(), size_ * sizeof(T),
inst/include/dust/gpu/types.hpp:                          cudaMemcpyDefault));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMemcpy(data_, src.data(), size_ * sizeof(T),
inst/include/dust/gpu/types.hpp:                          cudaMemcpyDefault));
inst/include/dust/gpu/types.hpp:  void set_array(T * dst, cuda_stream& stream, const bool async = false) const {
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMemcpyAsync(data_, dst, size() * sizeof(T),
inst/include/dust/gpu/types.hpp:                                cudaMemcpyDefault, stream.stream()));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMemcpy(data_, dst, size() * sizeof(T),
inst/include/dust/gpu/types.hpp:                           cudaMemcpyDefault));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMalloc((void**)&data_, size_));
inst/include/dust/gpu/types.hpp:    CUDA_CALL_NOTHROW(cudaFree(data_));
inst/include/dust/gpu/types.hpp:    CUDA_CALL(cudaFree(data_));
inst/include/dust/gpu/types.hpp:      CUDA_CALL(cudaMalloc((void**)&data_, size_));
inst/include/dust/gpu/types.hpp:  // CUDA version of log-sum-exp trick
inst/include/dust/gpu/types.hpp:  cuda_stream kernel_stream_;
inst/include/dust/gpu/kernels.hpp:#ifndef DUST_GPU_KERNELS_HPP
inst/include/dust/gpu/kernels.hpp:#define DUST_GPU_KERNELS_HPP
inst/include/dust/gpu/kernels.hpp:#include "dust/gpu/device_state.hpp"
inst/include/dust/gpu/kernels.hpp:namespace gpu {
inst/include/dust/gpu/kernels.hpp:void update_gpu(size_t time,
inst/include/dust/gpu/kernels.hpp:typename T::real_type compare_gpu(
inst/include/dust/gpu/kernels.hpp:  // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
inst/include/dust/gpu/kernels.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/gpu/kernels.hpp:    // Otherwise CUDA thread number = particle
inst/include/dust/gpu/kernels.hpp:      update_gpu<T>(time,
inst/include/dust/gpu/kernels.hpp:#ifdef __CUDA_ARCH__
inst/include/dust/gpu/kernels.hpp:    // Otherwise CUDA thread number = particle
inst/include/dust/gpu/kernels.hpp:    weights[i] = compare_gpu<T>(p_state,
inst/include/dust/gpu/kernels.hpp:// Likely not particularly CUDA friendly, but will do for now
inst/include/dust/gpu/utils.hpp:#ifndef DUST_GPU_UTILS_HPP
inst/include/dust/gpu/utils.hpp:#define DUST_GPU_UTILS_HPP
inst/include/dust/gpu/utils.hpp:#include "dust/gpu/cuda.hpp"
inst/include/dust/gpu/utils.hpp:namespace gpu {
inst/include/dust/gpu/filter_kernels.hpp:#ifndef DUST_GPU_FILTER_KERNELS_HPP
inst/include/dust/gpu/filter_kernels.hpp:#define DUST_GPU_FILTER_KERNELS_HPP
inst/include/dust/gpu/filter_kernels.hpp:#include "dust/gpu/cuda.hpp"
inst/include/dust/gpu/call.hpp:#ifndef DUST_GPU_CALL_HPP
inst/include/dust/gpu/call.hpp:#define DUST_GPU_CALL_HPP
inst/include/dust/gpu/call.hpp:#include <cuda.h>
inst/include/dust/gpu/call.hpp:#include <cuda_runtime.h>
inst/include/dust/gpu/call.hpp:#include <cuda_profiler_api.h>
inst/include/dust/gpu/call.hpp:namespace gpu {
inst/include/dust/gpu/call.hpp:static void throw_cuda_error(const char *file, int line, cudaError_t status) {
inst/include/dust/gpu/call.hpp:  if (status == cudaErrorUnknown) {
inst/include/dust/gpu/call.hpp:    msg << file << "(" << line << ") An Unknown CUDA Error Occurred :(";
inst/include/dust/gpu/call.hpp:    msg << file << "(" << line << ") CUDA Error Occurred:\n" <<
inst/include/dust/gpu/call.hpp:      cudaGetErrorString(status);
inst/include/dust/gpu/call.hpp:#ifdef DUST_ENABLE_CUDA_PROFILER
inst/include/dust/gpu/call.hpp:  cudaProfilerStop();
inst/include/dust/gpu/call.hpp:static void handle_cuda_error(const char *file, int line,
inst/include/dust/gpu/call.hpp:                              cudaError_t status = cudaGetLastError()) {
inst/include/dust/gpu/call.hpp:  cudaDeviceSynchronize();
inst/include/dust/gpu/call.hpp:  if (status != cudaSuccess || (status = cudaGetLastError()) != cudaSuccess) {
inst/include/dust/gpu/call.hpp:    throw_cuda_error(file, line, status);
inst/include/dust/gpu/call.hpp:#define CUDA_CALL( err ) (dust::gpu::handle_cuda_error(__FILE__, __LINE__ , err))
inst/include/dust/gpu/call.hpp:#define CUDA_CALL_NOTHROW( err ) (err)
inst/include/dust/gpu/filter.hpp:#ifndef DUST_GPU_FILTER_HPP
inst/include/dust/gpu/filter.hpp:#define DUST_GPU_FILTER_HPP
inst/include/dust/gpu/filter.hpp:#include "dust/gpu/filter_state.hpp"
inst/include/dust/gpu/filter.hpp:  dust::gpu::device_array<real_type> log_likelihood(ll_host);
inst/include/dust/gpu/filter.hpp:  dust::gpu::device_weights<real_type> weights(n_particles, n_pars);
inst/include/dust/gpu/filter.hpp:  dust::gpu::device_scan_state<real_type> scan;
inst/include/dust/gpu/cuda.hpp:#ifndef DUST_GPU_CUDA_HPP
inst/include/dust/gpu/cuda.hpp:#define DUST_GPU_CUDA_HPP
inst/include/dust/gpu/cuda.hpp:// dust/random/cuda_compatibility.hpp rather than here; they need to
inst/include/dust/gpu/cuda.hpp:// CUDA 11 cooperative groups
inst/include/dust/gpu/cuda.hpp:#if __CUDACC_VER_MAJOR__ >= 11
inst/include/dust/gpu/cuda.hpp:#include "dust/gpu/call.hpp"
inst/include/dust/gpu/cuda.hpp:#include "dust/random/cuda_compatibility.hpp"
inst/include/dust/gpu/cuda.hpp:#undef DUST_CUDA_ENABLE_PROFILER
inst/include/dust/gpu/cuda.hpp:namespace gpu {
inst/include/dust/gpu/cuda.hpp:#if __CUDACC_VER_MAJOR__ >= 11
inst/include/dust/gpu/cuda.hpp:#if __CUDACC_VER_MAJOR__ >= 11
inst/include/dust/gpu/cuda.hpp:class cuda_stream {
inst/include/dust/gpu/cuda.hpp:  cuda_stream() {
inst/include/dust/gpu/cuda.hpp:    cudaError_t status = cudaStreamCreate(&stream_);
inst/include/dust/gpu/cuda.hpp:    if (status == cudaErrorNoDevice) {
inst/include/dust/gpu/cuda.hpp:    } else if (status != cudaSuccess) {
inst/include/dust/gpu/cuda.hpp:      dust::gpu::throw_cuda_error(__FILE__, __LINE__, status);
inst/include/dust/gpu/cuda.hpp:  ~cuda_stream() {
inst/include/dust/gpu/cuda.hpp:      CUDA_CALL_NOTHROW(cudaStreamDestroy(stream_));
inst/include/dust/gpu/cuda.hpp:  cudaStream_t stream() {
inst/include/dust/gpu/cuda.hpp:    CUDA_CALL(cudaStreamSynchronize(stream_));
inst/include/dust/gpu/cuda.hpp:    if (cudaStreamQuery(stream_) != cudaSuccess) {
inst/include/dust/gpu/cuda.hpp:  cuda_stream ( const cuda_stream & ) = delete;
inst/include/dust/gpu/cuda.hpp:  cuda_stream ( cuda_stream && ) = delete;
inst/include/dust/gpu/cuda.hpp:  cudaStream_t stream_;
inst/include/dust/gpu/dust_gpu.hpp:#ifndef DUST_GPU_DUST_GPU_HPP
inst/include/dust/gpu/dust_gpu.hpp:#define DUST_GPU_DUST_GPU_HPP
inst/include/dust/gpu/dust_gpu.hpp:#include "dust/gpu/call.hpp"
inst/include/dust/gpu/dust_gpu.hpp:#include "dust/gpu/cuda.hpp"
inst/include/dust/gpu/dust_gpu.hpp:#include "dust/gpu/device_resample.hpp"
inst/include/dust/gpu/dust_gpu.hpp:#include "dust/gpu/filter_state.hpp"
inst/include/dust/gpu/dust_gpu.hpp:#include "dust/gpu/kernels.hpp"
inst/include/dust/gpu/dust_gpu.hpp:#include "dust/gpu/launch_control.hpp"
inst/include/dust/gpu/dust_gpu.hpp:#include "dust/gpu/types.hpp"
inst/include/dust/gpu/dust_gpu.hpp:class dust_gpu {
inst/include/dust/gpu/dust_gpu.hpp:  dust_gpu(const pars_type& pars, const size_t time, const size_t n_particles,
inst/include/dust/gpu/dust_gpu.hpp:           const gpu::gpu_config& gpu_config) :
inst/include/dust/gpu/dust_gpu.hpp:    gpu_config_(gpu_config),
inst/include/dust/gpu/dust_gpu.hpp:  dust_gpu(const std::vector<pars_type>& pars, const size_t time,
inst/include/dust/gpu/dust_gpu.hpp:           const gpu::gpu_config& gpu_config) :
inst/include/dust/gpu/dust_gpu.hpp:    gpu_config_(gpu_config),
inst/include/dust/gpu/dust_gpu.hpp:  // We only need a destructor when running with cuda profiling; don't
inst/include/dust/gpu/dust_gpu.hpp:#ifdef DUST_ENABLE_CUDA_PROFILER
inst/include/dust/gpu/dust_gpu.hpp:  ~dust_gpu() {
inst/include/dust/gpu/dust_gpu.hpp:    cuda_profiler_stop(gpu_config_);
inst/include/dust/gpu/dust_gpu.hpp:  // on the cpu (and errors on the gpu are unrecoverable). If we ever
inst/include/dust/gpu/dust_gpu.hpp:      throw std::runtime_error("Can't use index with gpu models");
inst/include/dust/gpu/dust_gpu.hpp:    const auto block_size = cuda_pars_.index_scatter.block_size;
inst/include/dust/gpu/dust_gpu.hpp:    cuda_pars_.index_scatter =
inst/include/dust/gpu/dust_gpu.hpp:      dust::gpu::launch_control_simple(block_size, n_particles * n_state());
inst/include/dust/gpu/dust_gpu.hpp:      dust::gpu::run_particles<T><<<cuda_pars_.run.block_count,
inst/include/dust/gpu/dust_gpu.hpp:                                     cuda_pars_.run.block_size,
inst/include/dust/gpu/dust_gpu.hpp:                                     cuda_pars_.run.shared_size_bytes,
inst/include/dust/gpu/dust_gpu.hpp:                      cuda_pars_.run.shared_int,
inst/include/dust/gpu/dust_gpu.hpp:                      cuda_pars_.run.shared_real);
inst/include/dust/gpu/dust_gpu.hpp:      dust::gpu::run_particles<T>(time_start, time_end, n_particles_total_,
inst/include/dust/gpu/dust_gpu.hpp:  void state_full(dust::gpu::device_array<real_type>& device_state,
inst/include/dust/gpu/dust_gpu.hpp:    dust::gpu::scatter_device<real_type><<<cuda_pars_.reorder.block_count,
inst/include/dust/gpu/dust_gpu.hpp:                                          cuda_pars_.reorder.block_size,
inst/include/dust/gpu/dust_gpu.hpp:    dust::gpu::scatter_device<real_type>(
inst/include/dust/gpu/dust_gpu.hpp:    dust::gpu::device_weights<real_type>
inst/include/dust/gpu/dust_gpu.hpp:    dust::gpu::device_scan_state<real_type> scan;
inst/include/dust/gpu/dust_gpu.hpp:  void resample(dust::gpu::device_array<real_type>& weights,
inst/include/dust/gpu/dust_gpu.hpp:                dust::gpu::device_scan_state<real_type>& scan) {
inst/include/dust/gpu/dust_gpu.hpp:                                      cuda_pars_,
inst/include/dust/gpu/dust_gpu.hpp:  dust::gpu::device_array<size_t>& filter_kappa() {
inst/include/dust/gpu/dust_gpu.hpp:  dust::gpu::device_array<real_type>& device_state_full() {
inst/include/dust/gpu/dust_gpu.hpp:  dust::gpu::device_array<real_type>& device_state_selected() {
inst/include/dust/gpu/dust_gpu.hpp:    device_data_ = dust::gpu::device_array<data_type>(flattened_data.size());
inst/include/dust/gpu/dust_gpu.hpp:  void compare_data(dust::gpu::device_array<real_type>& res,
inst/include/dust/gpu/dust_gpu.hpp:    dust::gpu::compare_particles<T><<<cuda_pars_.compare.block_count,
inst/include/dust/gpu/dust_gpu.hpp:                                       cuda_pars_.compare.block_size,
inst/include/dust/gpu/dust_gpu.hpp:                                       cuda_pars_.compare.shared_size_bytes,
inst/include/dust/gpu/dust_gpu.hpp:                     cuda_pars_.compare.shared_int,
inst/include/dust/gpu/dust_gpu.hpp:                     cuda_pars_.compare.shared_real,
inst/include/dust/gpu/dust_gpu.hpp:    dust::gpu::compare_particles<T>(
inst/include/dust/gpu/dust_gpu.hpp:  dust_gpu(const dust_gpu &) = delete;
inst/include/dust/gpu/dust_gpu.hpp:  dust_gpu(dust_gpu &&) = delete;
inst/include/dust/gpu/dust_gpu.hpp:  gpu::gpu_config gpu_config_;
inst/include/dust/gpu/dust_gpu.hpp:  // GPU support
inst/include/dust/gpu/dust_gpu.hpp:  dust::gpu::launch_control_dust cuda_pars_;
inst/include/dust/gpu/dust_gpu.hpp:  dust::gpu::device_state<real_type, rng_state_type> device_state_;
inst/include/dust/gpu/dust_gpu.hpp:  dust::gpu::device_array<data_type> device_data_;
inst/include/dust/gpu/dust_gpu.hpp:  dust::gpu::cuda_stream kernel_stream_;
inst/include/dust/gpu/dust_gpu.hpp:  dust::gpu::cuda_stream resample_stream_;
inst/include/dust/gpu/dust_gpu.hpp:        throw std::runtime_error("GPU models cannot use rng in initial");
inst/include/dust/gpu/dust_gpu.hpp:    // Set GPU RNG from a seed; primary reason for this construction
inst/include/dust/gpu/dust_gpu.hpp:    set_cuda_launch();
inst/include/dust/gpu/dust_gpu.hpp:#ifdef DUST_ENABLE_CUDA_PROFILER
inst/include/dust/gpu/dust_gpu.hpp:    cuda_profiler_start(gpu_config_);
inst/include/dust/gpu/dust_gpu.hpp:    const size_t n_internal_int = dust::gpu::internal_int_size<T>(s);
inst/include/dust/gpu/dust_gpu.hpp:    const size_t n_internal_real = dust::gpu::internal_real_size<T>(s);
inst/include/dust/gpu/dust_gpu.hpp:    const size_t n_shared_int = dust::gpu::shared_int_size<T>(s);
inst/include/dust/gpu/dust_gpu.hpp:    const size_t n_shared_real = dust::gpu::shared_real_size<T>(s);
inst/include/dust/gpu/dust_gpu.hpp:      dust::gpu::shared_copy<T>(pars[i].shared, dest_int, dest_real);
inst/include/dust/gpu/dust_gpu.hpp:      dust::gpu::scatter_device<real_type><<<cuda_pars_.index_scatter.block_count,
inst/include/dust/gpu/dust_gpu.hpp:                                           cuda_pars_.index_scatter.block_size,
inst/include/dust/gpu/dust_gpu.hpp:      dust::gpu::scatter_device<real_type>(
inst/include/dust/gpu/dust_gpu.hpp:  // Set up CUDA block sizes and shared memory preferences
inst/include/dust/gpu/dust_gpu.hpp:  void set_cuda_launch() {
inst/include/dust/gpu/dust_gpu.hpp:    cuda_pars_ = dust::gpu::launch_control_dust(gpu_config_,
inst/include/dust/gpu/launch_control.hpp:#ifndef DUST_GPU_LAUNCH_CONTROL_HPP
inst/include/dust/gpu/launch_control.hpp:#define DUST_GPU_LAUNCH_CONTROL_HPP
inst/include/dust/gpu/launch_control.hpp:#include "dust/gpu/types.hpp"
inst/include/dust/gpu/launch_control.hpp:#include "dust/gpu/utils.hpp"
inst/include/dust/gpu/launch_control.hpp:namespace gpu {
inst/include/dust/gpu/launch_control.hpp:    CUDA_CALL(cudaDeviceGetAttribute(&size,
inst/include/dust/gpu/launch_control.hpp:                                     cudaDevAttrMaxSharedMemoryPerBlock,
inst/include/dust/gpu/launch_control.hpp:class gpu_config {
inst/include/dust/gpu/launch_control.hpp:  gpu_config(int device_id, int run_block_size) :
inst/include/dust/gpu/launch_control.hpp:    real_gpu_(true)
inst/include/dust/gpu/launch_control.hpp:    real_gpu_(false)
inst/include/dust/gpu/launch_control.hpp:    CUDA_CALL(cudaSetDevice(device_id_));
inst/include/dust/gpu/launch_control.hpp:    CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
inst/include/dust/gpu/launch_control.hpp:  const bool real_gpu_;
inst/include/dust/gpu/launch_control.hpp:inline void cuda_profiler_start(const gpu_config& config) {
inst/include/dust/gpu/launch_control.hpp:#ifdef DUST_ENABLE_CUDA_PROFILER
inst/include/dust/gpu/launch_control.hpp:  CUDA_CALL(cudaProfilerStart());
inst/include/dust/gpu/launch_control.hpp:inline void cuda_profiler_stop(const gpu_config& config) {
inst/include/dust/gpu/launch_control.hpp:#ifdef DUST_ENABLE_CUDA_PROFILER
inst/include/dust/gpu/launch_control.hpp:  CUDA_CALL(cudaProfilerStop());
inst/include/dust/gpu/launch_control.hpp:  launch_control_dust(const gpu_config& config,
inst/include/dust/gpu/launch_control.hpp:  const int warp_size = dust::gpu::warp_size;
inst/include/dust/gpu/launch_control.hpp:    dust::gpu::utils::align_padding(n_shared_int * int_size,
inst/include/dust/gpu/launch_control.hpp:    dust::gpu::utils::align_padding(shared_size_int_bytes +
inst/include/dust/gpu/launch_control.hpp:inline launch_control_dust::launch_control_dust(const gpu_config& config,
inst/include/dust/gpu/gpu_info.hpp:#ifndef DUST_GPU_GPU_INFO_HPP
inst/include/dust/gpu/gpu_info.hpp:#define DUST_GPU_GPU_INFO_HPP
inst/include/dust/gpu/gpu_info.hpp:#include "dust/gpu/call.hpp"
inst/include/dust/gpu/gpu_info.hpp:namespace gpu {
inst/include/dust/gpu/gpu_info.hpp:  cudaError_t status = cudaGetDeviceCount(&device_count);
inst/include/dust/gpu/gpu_info.hpp:  if (status != cudaSuccess && status != cudaErrorNoDevice) {
inst/include/dust/gpu/gpu_info.hpp:    throw_cuda_error(__FILE__, __LINE__, status);
inst/include/dust/gpu/filter_state.hpp:#ifndef DUST_GPU_FILTER_STATE_HPP
inst/include/dust/gpu/filter_state.hpp:#define DUST_GPU_FILTER_STATE_HPP
inst/include/dust/gpu/filter_state.hpp:#include "dust/gpu/device_state.hpp"
inst/include/dust/gpu/filter_state.hpp:    history_value_swap = dust::gpu::device_array<real_type>(this->n_state_ * this->n_particles_);
inst/include/dust/gpu/filter_state.hpp:    history_order_swap = dust::gpu::device_array<size_t>(this->n_particles_);
inst/include/dust/gpu/filter_state.hpp:    CUDA_CALL(cudaHostRegister(this->history_value.data(),
inst/include/dust/gpu/filter_state.hpp:                               cudaHostRegisterDefault));
inst/include/dust/gpu/filter_state.hpp:    CUDA_CALL(cudaHostRegister(this->history_order.data(),
inst/include/dust/gpu/filter_state.hpp:                               cudaHostRegisterDefault));
inst/include/dust/gpu/filter_state.hpp:  void store_values(dust::gpu::device_array<real_type>& state) {
inst/include/dust/gpu/filter_state.hpp:  void store_order(dust::gpu::device_array<size_t>& kappa) {
inst/include/dust/gpu/filter_state.hpp:  dust::gpu::device_array<real_type> history_value_swap;
inst/include/dust/gpu/filter_state.hpp:  dust::gpu::device_array<size_t> history_order_swap;
inst/include/dust/gpu/filter_state.hpp:  dust::gpu::cuda_stream device_memory_stream_;
inst/include/dust/gpu/filter_state.hpp:  dust::gpu::cuda_stream host_memory_stream_;
inst/include/dust/gpu/filter_state.hpp:      CUDA_CALL_NOTHROW(cudaHostUnregister(this->history_value.data()));
inst/include/dust/gpu/filter_state.hpp:      CUDA_CALL_NOTHROW(cudaHostUnregister(this->history_order.data()));
inst/include/dust/gpu/filter_state.hpp:    state_swap = dust::gpu::device_array<real_type>(this->n_state_ * this->n_particles_);
inst/include/dust/gpu/filter_state.hpp:    CUDA_CALL(cudaHostRegister(this->state_.data(),
inst/include/dust/gpu/filter_state.hpp:                               cudaHostRegisterDefault));
inst/include/dust/gpu/filter_state.hpp:  void store(dust::gpu::device_array<real_type>& state) {
inst/include/dust/gpu/filter_state.hpp:  dust::gpu::device_array<real_type> state_swap;
inst/include/dust/gpu/filter_state.hpp:  dust::gpu::cuda_stream device_memory_stream_;
inst/include/dust/gpu/filter_state.hpp:  dust::gpu::cuda_stream host_memory_stream_;
inst/include/dust/gpu/filter_state.hpp:      CUDA_CALL_NOTHROW(cudaHostUnregister(this->state_.data()));
inst/WORDLIST:CUDA
inst/WORDLIST:CUDA
inst/WORDLIST:CUDA's
inst/examples/sirs.cpp:namespace gpu {
inst/examples/sirs.cpp:  using dust::gpu::shared_copy_data;
inst/examples/sirs.cpp:void update_gpu<sirs>(size_t time,
inst/examples/sirs.cpp:                      const dust::gpu::interleaved<sirs::real_type> state,
inst/examples/sirs.cpp:                      dust::gpu::interleaved<int> internal_int,
inst/examples/sirs.cpp:                      dust::gpu::interleaved<sirs::real_type> internal_real,
inst/examples/sirs.cpp:                      dust::gpu::interleaved<sirs::real_type> state_next) {
inst/examples/sirs.cpp:sirs::real_type compare_gpu<sirs>(const dust::gpu::interleaved<sirs::real_type> state,
inst/examples/sirs.cpp:                                  dust::gpu::interleaved<int> internal_int,
inst/examples/sirs.cpp:                                  dust::gpu::interleaved<sirs::real_type> internal_real,
inst/examples/variable.cpp:namespace gpu {
inst/examples/variable.cpp:  using dust::gpu::shared_copy_data;
inst/examples/variable.cpp:void update_gpu<variable>(size_t time,
inst/examples/variable.cpp:                          const dust::gpu::interleaved<variable::real_type> state,
inst/examples/variable.cpp:                          dust::gpu::interleaved<int> internal_int,
inst/examples/variable.cpp:                          dust::gpu::interleaved<variable::real_type> internal_real,
inst/examples/variable.cpp:                          dust::gpu::interleaved<variable::real_type> state_next) {
inst/cuda/dust.hpp:cpp11::sexp dust_gpu_info();
inst/cuda/dust.cu:#include <dust/gpu/gpu_info.hpp>
inst/cuda/dust.cu:#include <dust/r/gpu_info.hpp>
inst/cuda/dust.cu:cpp11::sexp dust_gpu_info() {
inst/cuda/dust.cu:  return dust::gpu::r::gpu_info();
buildkite/pipeline.yml:  - label: ":allthethings: Build cuda image"
buildkite/pipeline.yml:    command: docker/build_cuda
R/cpp11.R:dust_logistic_gpu_info <- function() {
R/cpp11.R:  .Call(`_dust_dust_logistic_gpu_info`)
R/cpp11.R:dust_ode_logistic_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
R/cpp11.R:  .Call(`_dust_dust_ode_logistic_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
R/cpp11.R:dust_sir_gpu_info <- function() {
R/cpp11.R:  .Call(`_dust_dust_sir_gpu_info`)
R/cpp11.R:dust_cpu_sir_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
R/cpp11.R:  .Call(`_dust_dust_cpu_sir_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
R/cpp11.R:dust_sirs_gpu_info <- function() {
R/cpp11.R:  .Call(`_dust_dust_sirs_gpu_info`)
R/cpp11.R:dust_cpu_sirs_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
R/cpp11.R:  .Call(`_dust_dust_cpu_sirs_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
R/cpp11.R:dust_gpu_sirs_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
R/cpp11.R:dust_gpu_sirs_capabilities <- function() {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_capabilities`)
R/cpp11.R:dust_gpu_sirs_run <- function(ptr, r_time_end) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_run`, ptr, r_time_end)
R/cpp11.R:dust_gpu_sirs_simulate <- function(ptr, time_end) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_simulate`, ptr, time_end)
R/cpp11.R:dust_gpu_sirs_run_adjoint <- function(ptr) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_run_adjoint`, ptr)
R/cpp11.R:dust_gpu_sirs_set_index <- function(ptr, r_index) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_set_index`, ptr, r_index)
R/cpp11.R:dust_gpu_sirs_update_state <- function(ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_update_state`, ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size)
R/cpp11.R:dust_gpu_sirs_state <- function(ptr, r_index) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_state`, ptr, r_index)
R/cpp11.R:dust_gpu_sirs_time <- function(ptr) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_time`, ptr)
R/cpp11.R:dust_gpu_sirs_reorder <- function(ptr, r_index) {
R/cpp11.R:  invisible(.Call(`_dust_dust_gpu_sirs_reorder`, ptr, r_index))
R/cpp11.R:dust_gpu_sirs_resample <- function(ptr, r_weights) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_resample`, ptr, r_weights)
R/cpp11.R:dust_gpu_sirs_rng_state <- function(ptr, first_only, last_only) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_rng_state`, ptr, first_only, last_only)
R/cpp11.R:dust_gpu_sirs_set_rng_state <- function(ptr, rng_state) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_set_rng_state`, ptr, rng_state)
R/cpp11.R:dust_gpu_sirs_set_data <- function(ptr, data, shared) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_set_data`, ptr, data, shared)
R/cpp11.R:dust_gpu_sirs_compare_data <- function(ptr) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_compare_data`, ptr)
R/cpp11.R:dust_gpu_sirs_filter <- function(ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_filter`, ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood)
R/cpp11.R:dust_gpu_sirs_set_n_threads <- function(ptr, n_threads) {
R/cpp11.R:  invisible(.Call(`_dust_dust_gpu_sirs_set_n_threads`, ptr, n_threads))
R/cpp11.R:dust_gpu_sirs_n_state <- function(ptr) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_n_state`, ptr)
R/cpp11.R:dust_gpu_sirs_set_stochastic_schedule <- function(ptr, time) {
R/cpp11.R:  invisible(.Call(`_dust_dust_gpu_sirs_set_stochastic_schedule`, ptr, time))
R/cpp11.R:dust_gpu_sirs_ode_statistics <- function(ptr) {
R/cpp11.R:  .Call(`_dust_dust_gpu_sirs_ode_statistics`, ptr)
R/cpp11.R:test_cuda_pars <- function(r_gpu_config, n_particles, n_particles_each, n_state, n_state_full, n_shared_int, n_shared_real, data_size, shared_size) {
R/cpp11.R:  .Call(`_dust_test_cuda_pars`, r_gpu_config, n_particles, n_particles_each, n_state, n_state_full, n_shared_int, n_shared_real, data_size, shared_size)
R/cpp11.R:dust_variable_gpu_info <- function() {
R/cpp11.R:  .Call(`_dust_dust_variable_gpu_info`)
R/cpp11.R:dust_cpu_variable_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
R/cpp11.R:  .Call(`_dust_dust_cpu_variable_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
R/cpp11.R:dust_gpu_variable_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
R/cpp11.R:dust_gpu_variable_capabilities <- function() {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_capabilities`)
R/cpp11.R:dust_gpu_variable_run <- function(ptr, r_time_end) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_run`, ptr, r_time_end)
R/cpp11.R:dust_gpu_variable_simulate <- function(ptr, time_end) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_simulate`, ptr, time_end)
R/cpp11.R:dust_gpu_variable_run_adjoint <- function(ptr) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_run_adjoint`, ptr)
R/cpp11.R:dust_gpu_variable_set_index <- function(ptr, r_index) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_set_index`, ptr, r_index)
R/cpp11.R:dust_gpu_variable_update_state <- function(ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_update_state`, ptr, r_pars, r_state, r_time, r_set_initial_state, index, reset_step_size)
R/cpp11.R:dust_gpu_variable_state <- function(ptr, r_index) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_state`, ptr, r_index)
R/cpp11.R:dust_gpu_variable_time <- function(ptr) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_time`, ptr)
R/cpp11.R:dust_gpu_variable_reorder <- function(ptr, r_index) {
R/cpp11.R:  invisible(.Call(`_dust_dust_gpu_variable_reorder`, ptr, r_index))
R/cpp11.R:dust_gpu_variable_resample <- function(ptr, r_weights) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_resample`, ptr, r_weights)
R/cpp11.R:dust_gpu_variable_rng_state <- function(ptr, first_only, last_only) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_rng_state`, ptr, first_only, last_only)
R/cpp11.R:dust_gpu_variable_set_rng_state <- function(ptr, rng_state) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_set_rng_state`, ptr, rng_state)
R/cpp11.R:dust_gpu_variable_set_data <- function(ptr, data, shared) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_set_data`, ptr, data, shared)
R/cpp11.R:dust_gpu_variable_compare_data <- function(ptr) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_compare_data`, ptr)
R/cpp11.R:dust_gpu_variable_filter <- function(ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_filter`, ptr, time_end, save_trajectories, time_snapshot, min_log_likelihood)
R/cpp11.R:dust_gpu_variable_set_n_threads <- function(ptr, n_threads) {
R/cpp11.R:  invisible(.Call(`_dust_dust_gpu_variable_set_n_threads`, ptr, n_threads))
R/cpp11.R:dust_gpu_variable_n_state <- function(ptr) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_n_state`, ptr)
R/cpp11.R:dust_gpu_variable_set_stochastic_schedule <- function(ptr, time) {
R/cpp11.R:  invisible(.Call(`_dust_dust_gpu_variable_set_stochastic_schedule`, ptr, time))
R/cpp11.R:dust_gpu_variable_ode_statistics <- function(ptr) {
R/cpp11.R:  .Call(`_dust_dust_gpu_variable_ode_statistics`, ptr)
R/cpp11.R:dust_volatility_gpu_info <- function() {
R/cpp11.R:  .Call(`_dust_dust_volatility_gpu_info`)
R/cpp11.R:dust_cpu_volatility_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
R/cpp11.R:  .Call(`_dust_dust_cpu_volatility_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
R/cpp11.R:dust_walk_gpu_info <- function() {
R/cpp11.R:  .Call(`_dust_dust_walk_gpu_info`)
R/cpp11.R:dust_cpu_walk_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
R/cpp11.R:  .Call(`_dust_dust_cpu_walk_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
R/dust_generator.R:    gpu_config_ = NULL,
R/dust_generator.R:    ##' a a GPU.
R/dust_generator.R:    ##' @param gpu_config GPU configuration, typically an integer
R/dust_generator.R:    ##' indicating the device to use, where the model has GPU support.
R/dust_generator.R:    ##' that CUDA numbers devices from 0, so that '0' is the first device,
R/dust_generator.R:    ##' and so on). See the method `$gpu_info()` for available device ids;
R/dust_generator.R:    ##' `dust_generator$public_methods$gpu_info()`.
R/dust_generator.R:                          gpu_config = NULL, ode_control = NULL) {
R/dust_generator.R:    ##' "CUDA" support, in which case it will react to the `device`
R/dust_generator.R:    ##' as `dust_generator$public_methods$has_gpu_support()`
R/dust_generator.R:    ##' @param fake_gpu Logical, indicating if we count as `TRUE`
R/dust_generator.R:    ##'   models that run on the "fake" GPU (i.e., using the GPU
R/dust_generator.R:    has_gpu_support = function(fake_gpu = FALSE) {
R/dust_generator.R:    ##' Check if the model is running on a GPU
R/dust_generator.R:    ##' @param fake_gpu Logical, indicating if we count as `TRUE`
R/dust_generator.R:    ##'   models that run on the "fake" GPU (i.e., using the GPU
R/dust_generator.R:    uses_gpu = function(fake_gpu = FALSE) {
R/dust_generator.R:    ##' Return information about GPU devices, if the model
R/dust_generator.R:    ##' has been compiled with CUDA/GPU support. This can be called as a
R/dust_generator.R:    ##' static method by running `dust_generator$public_methods$gpu_info()`.
R/dust_generator.R:    ##' If run from a GPU enabled object, it will also have an element
R/dust_generator.R:    gpu_info = function() {
R/dust.R:    gpu_config_ = NULL,
R/dust.R:                          gpu_config = NULL, ode_control = NULL) {
R/dust.R:      if (is.null(gpu_config)) {
R/dust.R:          stop("GPU support not enabled for this object")
R/dust.R:                        n_threads, seed, deterministic, gpu_config, ode_control)
R/dust.R:      private$gpu_config_ <- res[[4L]]
R/dust.R:    has_gpu_support = function(fake_gpu = FALSE) {
R/dust.R:      if (fake_gpu) {
R/dust.R:        dust_ode_logistic_capabilities()[["gpu"]]
R/dust.R:    uses_gpu = function(fake_gpu = FALSE) {
R/dust.R:      real_gpu <- private$gpu_config_$real_gpu
R/dust.R:      !is.null(real_gpu) && (fake_gpu || real_gpu)
R/dust.R:    gpu_info = function() {
R/dust.R:      ret <- dust_logistic_gpu_info()
R/dust.R:      if (ret$has_cuda && exists("private", parent, inherits = FALSE)) {
R/dust.R:        ret$config <- private$gpu_config_
R/dust.R:    gpu_config_ = NULL,
R/dust.R:                          gpu_config = NULL, ode_control = NULL) {
R/dust.R:      if (is.null(gpu_config)) {
R/dust.R:          stop("GPU support not enabled for this object")
R/dust.R:                        n_threads, seed, deterministic, gpu_config, ode_control)
R/dust.R:      private$gpu_config_ <- res[[4L]]
R/dust.R:    has_gpu_support = function(fake_gpu = FALSE) {
R/dust.R:      if (fake_gpu) {
R/dust.R:        dust_cpu_sir_capabilities()[["gpu"]]
R/dust.R:    uses_gpu = function(fake_gpu = FALSE) {
R/dust.R:      real_gpu <- private$gpu_config_$real_gpu
R/dust.R:      !is.null(real_gpu) && (fake_gpu || real_gpu)
R/dust.R:    gpu_info = function() {
R/dust.R:      ret <- dust_sir_gpu_info()
R/dust.R:      if (ret$has_cuda && exists("private", parent, inherits = FALSE)) {
R/dust.R:        ret$config <- private$gpu_config_
R/dust.R:    gpu_config_ = NULL,
R/dust.R:                          gpu_config = NULL, ode_control = NULL) {
R/dust.R:      if (is.null(gpu_config)) {
R/dust.R:           alloc = dust_gpu_sirs_alloc,
R/dust.R:           run = dust_gpu_sirs_run,
R/dust.R:           simulate = dust_gpu_sirs_simulate,
R/dust.R:           run_adjoint = dust_gpu_sirs_run_adjoint,
R/dust.R:           set_index = dust_gpu_sirs_set_index,
R/dust.R:           n_state = dust_gpu_sirs_n_state,
R/dust.R:           update_state = dust_gpu_sirs_update_state,
R/dust.R:           state = dust_gpu_sirs_state,
R/dust.R:           time = dust_gpu_sirs_time,
R/dust.R:           reorder = dust_gpu_sirs_reorder,
R/dust.R:           resample = dust_gpu_sirs_resample,
R/dust.R:           rng_state = dust_gpu_sirs_rng_state,
R/dust.R:           set_rng_state = dust_gpu_sirs_set_rng_state,
R/dust.R:           set_n_threads = dust_gpu_sirs_set_n_threads,
R/dust.R:           set_data = dust_gpu_sirs_set_data,
R/dust.R:           compare_data = dust_gpu_sirs_compare_data,
R/dust.R:           filter = dust_gpu_sirs_filter,
R/dust.R:           set_stochastic_schedule = dust_gpu_sirs_set_stochastic_schedule,
R/dust.R:           ode_statistics = dust_gpu_sirs_ode_statistics)
R/dust.R:                        n_threads, seed, deterministic, gpu_config, ode_control)
R/dust.R:      private$gpu_config_ <- res[[4L]]
R/dust.R:    has_gpu_support = function(fake_gpu = FALSE) {
R/dust.R:      if (fake_gpu) {
R/dust.R:        dust_cpu_sirs_capabilities()[["gpu"]]
R/dust.R:    uses_gpu = function(fake_gpu = FALSE) {
R/dust.R:      real_gpu <- private$gpu_config_$real_gpu
R/dust.R:      !is.null(real_gpu) && (fake_gpu || real_gpu)
R/dust.R:    gpu_info = function() {
R/dust.R:      ret <- dust_sirs_gpu_info()
R/dust.R:      if (ret$has_cuda && exists("private", parent, inherits = FALSE)) {
R/dust.R:        ret$config <- private$gpu_config_
R/dust.R:    gpu_config_ = NULL,
R/dust.R:                          gpu_config = NULL, ode_control = NULL) {
R/dust.R:      if (is.null(gpu_config)) {
R/dust.R:           alloc = dust_gpu_variable_alloc,
R/dust.R:           run = dust_gpu_variable_run,
R/dust.R:           simulate = dust_gpu_variable_simulate,
R/dust.R:           run_adjoint = dust_gpu_variable_run_adjoint,
R/dust.R:           set_index = dust_gpu_variable_set_index,
R/dust.R:           n_state = dust_gpu_variable_n_state,
R/dust.R:           update_state = dust_gpu_variable_update_state,
R/dust.R:           state = dust_gpu_variable_state,
R/dust.R:           time = dust_gpu_variable_time,
R/dust.R:           reorder = dust_gpu_variable_reorder,
R/dust.R:           resample = dust_gpu_variable_resample,
R/dust.R:           rng_state = dust_gpu_variable_rng_state,
R/dust.R:           set_rng_state = dust_gpu_variable_set_rng_state,
R/dust.R:           set_n_threads = dust_gpu_variable_set_n_threads,
R/dust.R:           set_data = dust_gpu_variable_set_data,
R/dust.R:           compare_data = dust_gpu_variable_compare_data,
R/dust.R:           filter = dust_gpu_variable_filter,
R/dust.R:           set_stochastic_schedule = dust_gpu_variable_set_stochastic_schedule,
R/dust.R:           ode_statistics = dust_gpu_variable_ode_statistics)
R/dust.R:                        n_threads, seed, deterministic, gpu_config, ode_control)
R/dust.R:      private$gpu_config_ <- res[[4L]]
R/dust.R:    has_gpu_support = function(fake_gpu = FALSE) {
R/dust.R:      if (fake_gpu) {
R/dust.R:        dust_cpu_variable_capabilities()[["gpu"]]
R/dust.R:    uses_gpu = function(fake_gpu = FALSE) {
R/dust.R:      real_gpu <- private$gpu_config_$real_gpu
R/dust.R:      !is.null(real_gpu) && (fake_gpu || real_gpu)
R/dust.R:    gpu_info = function() {
R/dust.R:      ret <- dust_variable_gpu_info()
R/dust.R:      if (ret$has_cuda && exists("private", parent, inherits = FALSE)) {
R/dust.R:        ret$config <- private$gpu_config_
R/dust.R:    gpu_config_ = NULL,
R/dust.R:                          gpu_config = NULL, ode_control = NULL) {
R/dust.R:      if (is.null(gpu_config)) {
R/dust.R:          stop("GPU support not enabled for this object")
R/dust.R:                        n_threads, seed, deterministic, gpu_config, ode_control)
R/dust.R:      private$gpu_config_ <- res[[4L]]
R/dust.R:    has_gpu_support = function(fake_gpu = FALSE) {
R/dust.R:      if (fake_gpu) {
R/dust.R:        dust_cpu_volatility_capabilities()[["gpu"]]
R/dust.R:    uses_gpu = function(fake_gpu = FALSE) {
R/dust.R:      real_gpu <- private$gpu_config_$real_gpu
R/dust.R:      !is.null(real_gpu) && (fake_gpu || real_gpu)
R/dust.R:    gpu_info = function() {
R/dust.R:      ret <- dust_volatility_gpu_info()
R/dust.R:      if (ret$has_cuda && exists("private", parent, inherits = FALSE)) {
R/dust.R:        ret$config <- private$gpu_config_
R/dust.R:    gpu_config_ = NULL,
R/dust.R:                          gpu_config = NULL, ode_control = NULL) {
R/dust.R:      if (is.null(gpu_config)) {
R/dust.R:          stop("GPU support not enabled for this object")
R/dust.R:                        n_threads, seed, deterministic, gpu_config, ode_control)
R/dust.R:      private$gpu_config_ <- res[[4L]]
R/dust.R:    has_gpu_support = function(fake_gpu = FALSE) {
R/dust.R:      if (fake_gpu) {
R/dust.R:        dust_cpu_walk_capabilities()[["gpu"]]
R/dust.R:    uses_gpu = function(fake_gpu = FALSE) {
R/dust.R:      real_gpu <- private$gpu_config_$real_gpu
R/dust.R:      !is.null(real_gpu) && (fake_gpu || real_gpu)
R/dust.R:    gpu_info = function() {
R/dust.R:      ret <- dust_walk_gpu_info()
R/dust.R:      if (ret$has_cuda && exists("private", parent, inherits = FALSE)) {
R/dust.R:        ret$config <- private$gpu_config_
R/interface.R:##' Things are worse on a GPU; if an error is thrown by the RNG code
R/interface.R:##'   then we currently use CUDA's `__trap()` function which will
R/interface.R:##'   the GPU again, covering all methods in the class.  However, this
R/interface.R:##' @param gpu Logical, indicating if we should generate GPU
R/interface.R:##'   installed (CUDA toolkit and drivers) as well as a
R/interface.R:##'   CUDA-compatible GPU. If `TRUE`, then we call
R/interface.R:##'   [dust::dust_cuda_options] with no arguments. Alternatively, call
R/interface.R:##'   that function and pass the value here (e.g, `gpu =
R/interface.R:##'   dust::dust_cuda_options(debug = TRUE)`). Note that due to the
R/interface.R:##'   use of the `__syncwarp()` primitive this may require a GPU with
R/interface.R:##'   given type. This is primarily intended to be used as `gpu =
R/interface.R:##'   TRUE, real_type = "float"` in order to create model for the GPU
R/interface.R:dust <- function(filename, quiet = FALSE, workdir = NULL, gpu = FALSE,
R/interface.R:  compile_and_load(filename, quiet, workdir, cuda_check(gpu), linking_to,
R/interface.R:##' with CUDA models).
R/interface.R:dust_generate <- function(filename, quiet = FALSE, workdir = NULL, gpu = FALSE,
R/interface.R:  res <- generate_dust(filename, quiet, workdir, cuda_check(gpu), linking_to,
R/cuda.R:##' Detect CUDA configuration. This function tries to compile a small
R/cuda.R:##' your NVIDIA GPUs. If this works, then you can use the GPU-enabled
R/cuda.R:##' Not all installations leave the CUDA libraries on the default
R/cuda.R:##' to find `libcudart` then your CUDA libraries are not in the
R/cuda.R:##' default location. You can manually pass in the `path_cuda_lib`
R/cuda.R:##' argument, or set the `DUST_PATH_CUDA_LIB` environment variable (in
R/cuda.R:##' If you are using older CUDA (< 11.0.0) then you need to provide
R/cuda.R:##' manage state on the device (these are included in CUDA 11.0.0 and
R/cuda.R:##' dust:::cuda_install_cub(NULL)
R/cuda.R:##' @title Detect CUDA configuration
R/cuda.R:##' @param path_cuda_lib Optional path to the CUDA libraries, if they
R/cuda.R:##'   `-L{path_cuda_lib}` in calls to `nvcc`
R/cuda.R:##'   CUDA < 11.0.0. See Details
R/cuda.R:##' * `has_cuda`: logical, indicating if it is possible to compile CUDA on
R/cuda.R:##' * `cuda_version`: the version of CUDA found
R/cuda.R:##' * `path_cuda_lib`: path to CUDA libraries, if required
R/cuda.R:##' If compilation of the test program fails, then `has_cuda` will be
R/cuda.R:##' @seealso [dust::dust_cuda_options] which controls additional CUDA
R/cuda.R:##' # If you have your CUDA library in an unusual location, then you
R/cuda.R:##' # may need to add a path_cuda_lib argument:
R/cuda.R:##' dust::dust_cuda_configuration(
R/cuda.R:##'   path_cuda_lib = "/usr/local/cuda-11.1/lib64",
R/cuda.R:##' dust::dust_cuda_configuration(forget = TRUE, quiet = FALSE)
R/cuda.R:dust_cuda_configuration <- function(path_cuda_lib = NULL,
R/cuda.R:  if (is.null(cache$cuda) || forget) {
R/cuda.R:    cache$cuda <- cuda_configuration(path_cuda_lib, path_cub_include, quiet)
R/cuda.R:  cache$cuda
R/cuda.R:##' Create options for compiling for CUDA.  Unless you need to change
R/cuda.R:##' @title Create CUDA options
R/cuda.R:##' @param ... Arguments passed to [dust::dust_cuda_configuration()]
R/cuda.R:##'   IEEE compliance and disables some error checking (see [the CUDA
R/cuda.R:##'   docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
R/cuda.R:##' @return An object of type `cuda_options`, which can be passed into
R/cuda.R:##'   [dust::dust] as argument `gpu`
R/cuda.R:##' @seealso [dust::dust_cuda_configuration] which identifies and
R/cuda.R:##'   returns the core CUDA configuration (often used implicitly by
R/cuda.R:##'   dust::dust_cuda_options(),
R/cuda.R:dust_cuda_options <- function(..., debug = FALSE, profile = FALSE,
R/cuda.R:  info <- dust_cuda_configuration(...)
R/cuda.R:  if (!info$has_cuda) {
R/cuda.R:    stop("cuda not supported on this machine")
R/cuda.R:  cuda_options(info, debug, profile, fast_math, flags)
R/cuda.R:cuda_configuration <- function(path_cuda_lib = NULL, path_cub_include = NULL,
R/cuda.R:  no_cuda <- list(
R/cuda.R:    has_cuda = FALSE,
R/cuda.R:    cuda_version = NULL,
R/cuda.R:    path_cuda_lib = NULL,
R/cuda.R:    path_cuda_lib <- cuda_path_cuda_lib(path_cuda_lib)
R/cuda.R:    dat <- cuda_create_test_package(path_cuda_lib)
R/cuda.R:    info <- pkg$env$dust_gpu_info()
R/cuda.R:      cuda_path_cub_include(info$cuda_version, path_cub_include)
R/cuda.R:      path_cuda_lib = path_cuda_lib,
R/cuda.R:    no_cuda
R/cuda.R:cuda_path_cuda_lib <- function(path) {
R/cuda.R:    if (!any(grepl("^libcudart", dir(path), ignore.case = TRUE))) {
R/cuda.R:      stop(sprintf("Did not find 'libcudart' within '%s' (via %s)",
R/cuda.R:  path <- Sys.getenv("DUST_PATH_CUDA_LIB", NA_character_)
R/cuda.R:    check_path(path, "environment variable 'DUST_PATH_CUDA_LIB'")
R/cuda.R:cuda_path_cub_include <- function(version, path) {
R/cuda.R:  path <- cuda_cub_path_default()
R/cuda.R:  stop("Did not find cub headers, see ?dust_cuda_configuration")
R/cuda.R:cuda_cub_path_default <- function(r_version = getRversion()) {
R/cuda.R:cuda_flag_helper <- function(value, prefix) {
R/cuda.R:cuda_create_test_package <- function(path_cuda_lib, path = tempfile()) {
R/cuda.R:               cuda = list(gencode = "",
R/cuda.R:                           lib_flags = cuda_flag_helper(path_cuda_lib, "-L")))
R/cuda.R:  file.copy(dust_file("cuda/dust.cu"), path_src)
R/cuda.R:  file.copy(dust_file("cuda/dust.hpp"), path_src)
R/cuda.R:  substitute_dust_template(data, "Makevars.cuda",
R/cuda.R:cuda_install_cub <- function(path, version = "1.9.10", quiet = FALSE) {
R/cuda.R:  path <- path %||% cuda_cub_path_default()
R/cuda.R:  url <- sprintf("https://github.com/nvidia/cub/archive/%s.zip", version)
R/cuda.R:cuda_options <- function(info, debug, profile, fast_math, flags) {
R/cuda.R:                        "-DDUST_ENABLE_CUDA_PROFILER")
R/cuda.R:    cub_include = cuda_flag_helper(info$path_cub_include, "-I"),
R/cuda.R:    lib_flags = cuda_flag_helper(info$path_cuda_lib, "-L"))
R/cuda.R:  class(info) <- "cuda_options"
R/cuda.R:cuda_check <- function(x) {
R/cuda.R:      return(dust_cuda_options())
R/cuda.R:  assert_is(x, "cuda_options")
R/compile.R:generate_dust <- function(filename, quiet, workdir, cuda, linking_to, cpp_std,
R/compile.R:  gpu <- isTRUE(cuda$has_cuda)
R/compile.R:  if (gpu) {
R/compile.R:    base <- paste0(base, "gpu")
R/compile.R:  data <- dust_template_data(model, config, cuda, reload, linking_to, cpp_std,
R/compile.R:  if (is.null(cuda)) {
R/compile.R:    substitute_dust_template(data, "Makevars.cuda",
R/compile.R:  res <- list(key = base, gpu = gpu, data = data, path = path)
R/compile.R:  if (config$has_gpu_support) {
R/compile.R:    data_gpu <- data
R/compile.R:    data_gpu$target <- "gpu"
R/compile.R:    data_gpu$container <- "dust_gpu"
R/compile.R:                  substitute_dust_template(data_gpu, "dust_methods.cpp", NULL))
R/compile.R:                  substitute_dust_template(data_gpu, "dust_methods.hpp", NULL))
R/compile.R:                             cuda = NULL, linking_to = NULL, cpp_std = NULL,
R/compile.R:  res <- generate_dust(filename, quiet, workdir, cuda, linking_to, cpp_std,
R/compile.R:dust_template_data <- function(model, config, cuda, reload_data, linking_to,
R/compile.R:  if (config$has_gpu_support) {
R/compile.R:    ## TODO: make sure in the config nobody tries for ode + gpu, that
R/compile.R:    methods_gpu <- methods("gpu")
R/compile.R:    methods_gpu <- paste(
R/compile.R:      '          stop("GPU support not enabled for this object")',
R/compile.R:       cuda = cuda$flags,
R/compile.R:       has_gpu_support = as.character(config$has_gpu_support),
R/compile.R:       methods_gpu = methods_gpu,
R/metadata.R:              has_gpu_support = parse_metadata_has_gpu_support(data))
R/metadata.R:  if (is.null(ret$has_gpu_support)) {
R/metadata.R:    ret$has_gpu_support <- parse_code_has_gpu_support(readLines(filename))
R/metadata.R:parse_metadata_has_gpu_support <- function(data) {
R/metadata.R:  value <- parse_metadata_simple(data, "dust::has_gpu_support")
R/metadata.R:      stop("Invalid value for dust::has_gpu_support, expected logical")
R/metadata.R:parse_code_has_gpu_support <- function(txt) {
R/metadata.R:  re <- "void\\s+update_gpu\\s*<\\s*"
tests/testthat/test-example.R:  no_cuda <- list(has_cuda = FALSE,
tests/testthat/test-example.R:                  cuda_version = NULL,
tests/testthat/test-example.R:  expect_false(res$public_methods$has_gpu_support())
tests/testthat/test-example.R:  expect_equal(res$public_methods$gpu_info(), no_cuda)
tests/testthat/test-example.R:  expect_false(mod$has_gpu_support())
tests/testthat/test-example.R:  expect_equal(mod$gpu_info(), no_cuda)
tests/testthat/test-example.R:  expect_false(mod$uses_gpu())
tests/testthat/test-example.R:  expect_false(mod$uses_gpu(TRUE))
tests/testthat/test-example.R:test_that("sirs model has gpu support", {
tests/testthat/test-example.R:  expect_false(gen$public_methods$has_gpu_support())
tests/testthat/test-example.R:  expect_true(gen$public_methods$has_gpu_support(TRUE))
tests/testthat/test-example.R:  mod1 <- gen$new(list(), 0, 1, gpu_config = NULL)
tests/testthat/test-example.R:  expect_false(mod1$uses_gpu())
tests/testthat/test-example.R:  expect_false(mod1$uses_gpu(TRUE))
tests/testthat/test-example.R:  mod2 <- gen$new(list(), 0, 1, gpu_config = 0L)
tests/testthat/test-example.R:  expect_false(mod2$uses_gpu())
tests/testthat/test-example.R:  expect_true(mod2$uses_gpu(TRUE))
tests/testthat/test-gpu.R:test_that("Can run gpu version of model on cpu", {
tests/testthat/test-gpu.R:  mod1 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = NULL)
tests/testthat/test-gpu.R:  mod2 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:  expect_false(mod1$uses_gpu())
tests/testthat/test-gpu.R:  expect_false(mod2$uses_gpu())
tests/testthat/test-gpu.R:  expect_true(mod2$uses_gpu(TRUE))
tests/testthat/test-gpu.R:test_that("Raise suitable errors if models do not support GPU", {
tests/testthat/test-gpu.R:    gen$new(list(sd = 1), 0, 100, seed = 1L, gpu_config = 0L),
tests/testthat/test-gpu.R:    "GPU support not enabled for this object")
tests/testthat/test-gpu.R:  mod2 <- res$new(p, 0, 5, seed = 1L, pars_multi = TRUE, gpu_config = 0L)
tests/testthat/test-gpu.R:test_that("Can reorder on the gpu", {
tests/testthat/test-gpu.R:  mod2 <- res$new(p, 0, np, seed = 1L, pars_multi = TRUE, gpu_config = 0L)
tests/testthat/test-gpu.R:test_that("Can generate cuda compatible code", {
tests/testthat/test-gpu.R:    has_cuda = TRUE,
tests/testthat/test-gpu.R:    cuda_version = numeric_version("10.1.0"),
tests/testthat/test-gpu.R:    path_cuda_lib = "/path/to/cuda",
tests/testthat/test-gpu.R:  cuda <- cuda_options(info, FALSE, FALSE, FALSE, NULL)
tests/testthat/test-gpu.R:  res <- generate_dust(dust_file("examples/sirs.cpp"), TRUE, workdir, cuda,
tests/testthat/test-gpu.R:  expect_match(txt, "-L/path/to/cuda", all = FALSE, fixed = TRUE)
tests/testthat/test-gpu.R:test_that("Generate default cuda configuration", {
tests/testthat/test-gpu.R:  mockery::stub(cuda_configuration, "cuda_create_test_package",
tests/testthat/test-gpu.R:  res <- cuda_configuration(quiet = TRUE)
tests/testthat/test-gpu.R:               list(has_cuda = FALSE,
tests/testthat/test-gpu.R:                    cuda_version = NULL,
tests/testthat/test-gpu.R:                    path_cuda_lib = NULL,
tests/testthat/test-gpu.R:  mockery::stub(cuda_configuration, "cuda_create_test_package",
tests/testthat/test-gpu.R:  res <- cuda_configuration(quiet = TRUE)
tests/testthat/test-gpu.R:  expect_equal(res, c(example_cuda_config(),
tests/testthat/test-gpu.R:                      list(path_cuda_lib = NULL, path_cub_include = NULL)))
tests/testthat/test-gpu.R:  mockery::stub(cuda_configuration, "cuda_create_test_package",
tests/testthat/test-gpu.R:  expect_message(cuda_configuration(),
tests/testthat/test-gpu.R:    cuda_path_cub_include(version_10, path_bad),
tests/testthat/test-gpu.R:  expect_equal(cuda_path_cub_include(version_10, path_good), path_good)
tests/testthat/test-gpu.R:      cuda_path_cub_include(version_10, NULL)),
tests/testthat/test-gpu.R:    expect_equal(cuda_path_cub_include(version_10, NULL), path_good))
tests/testthat/test-gpu.R:    expect_null(cuda_path_cub_include(version_11, NULL)))
tests/testthat/test-gpu.R:  mock_cuda_path <- mockery::mock(path_bad, path_good, NULL)
tests/testthat/test-gpu.R:  mockery::stub(cuda_path_cub_include,
tests/testthat/test-gpu.R:                "cuda_cub_path_default",
tests/testthat/test-gpu.R:                mock_cuda_path)
tests/testthat/test-gpu.R:    cuda_path_cub_include(version_10, NULL),
tests/testthat/test-gpu.R:  expect_equal(cuda_path_cub_include(version_10, NULL), path_good)
tests/testthat/test-gpu.R:    cuda_path_cub_include(version_10, NULL),
tests/testthat/test-gpu.R:test_that("locate cuda libs", {
tests/testthat/test-gpu.R:  file.create(file.path(path_good, "libcudart.so"))
tests/testthat/test-gpu.R:    cuda_path_cuda_lib(path_bad),
tests/testthat/test-gpu.R:    "Did not find 'libcudart' within '.+' \\(via provided argument\\)")
tests/testthat/test-gpu.R:  expect_equal(cuda_path_cuda_lib(path_good), path_good)
tests/testthat/test-gpu.R:      c(DUST_PATH_CUDA_LIB = path_bad),
tests/testthat/test-gpu.R:      cuda_path_cuda_lib(NULL)),
tests/testthat/test-gpu.R:    "Did not find 'libcudart' within '.+' \\(via environment variable")
tests/testthat/test-gpu.R:    c(DUST_PATH_CUDA_LIB = path_good),
tests/testthat/test-gpu.R:    expect_equal(cuda_path_cuda_lib(NULL), path_good))
tests/testthat/test-gpu.R:    c(DUST_PATH_CUDA_LIB = NA_character_),
tests/testthat/test-gpu.R:    expect_null(cuda_path_cuda_lib(NULL)))
tests/testthat/test-gpu.R:test_that("cuda_cub_path_default returns sensible values by R version", {
tests/testthat/test-gpu.R:  expect_null(cuda_cub_path_default("3.6.3"))
tests/testthat/test-gpu.R:  expect_equal(cuda_cub_path_default("4.0.0"),
tests/testthat/test-gpu.R:  expect_message(p <- cuda_install_cub(path, quiet = TRUE),
tests/testthat/test-gpu.R:  expect_error(cuda_install_cub(path, quiet = TRUE),
tests/testthat/test-gpu.R:test_that("Set the cuda options", {
tests/testthat/test-gpu.R:  info <- example_cuda_config()
tests/testthat/test-gpu.R:    cuda_options(info, FALSE, FALSE, FALSE, NULL)$flags,
tests/testthat/test-gpu.R:    cuda_options(info, TRUE, TRUE, FALSE, NULL)$flags,
tests/testthat/test-gpu.R:                            "-DDUST_ENABLE_CUDA_PROFILER"),
tests/testthat/test-gpu.R:  info$path_cuda_lib <- "/path/to/cuda"
tests/testthat/test-gpu.R:    cuda_options(info, FALSE, FALSE, TRUE, NULL)$flags,
tests/testthat/test-gpu.R:         lib_flags = "-L/path/to/cuda"))
tests/testthat/test-gpu.R:    cuda_options(info, FALSE, FALSE, FALSE, "--maxregcount=100")$flags,
tests/testthat/test-gpu.R:         lib_flags = "-L/path/to/cuda"))
tests/testthat/test-gpu.R:    cuda_options(info, FALSE, FALSE, FALSE,
tests/testthat/test-gpu.R:         lib_flags = "-L/path/to/cuda"))
tests/testthat/test-gpu.R:test_that("can create sensible cuda options", {
tests/testthat/test-gpu.R:  opts <- cuda_options(example_cuda_config(), FALSE, FALSE, FALSE, NULL)
tests/testthat/test-gpu.R:  mock_dust_cuda_options <- mockery::mock(opts, cycle = TRUE)
tests/testthat/test-gpu.R:  mockery::stub(cuda_check, "dust_cuda_options", mock_dust_cuda_options)
tests/testthat/test-gpu.R:  expect_null(cuda_check(NULL))
tests/testthat/test-gpu.R:  expect_null(cuda_check(FALSE))
tests/testthat/test-gpu.R:  expect_equal(cuda_check(TRUE), opts)
tests/testthat/test-gpu.R:  expect_equal(cuda_check(opts), opts)
tests/testthat/test-gpu.R:  expect_error(cuda_check("something"),
tests/testthat/test-gpu.R:               "'x' must be a cuda_options")
tests/testthat/test-gpu.R:  res <- cuda_create_test_package("/path/to/cuda")
tests/testthat/test-gpu.R:  expect_match(txt, "-L/path/to/cuda", all = FALSE, fixed = TRUE)
tests/testthat/test-gpu.R:  prev <- cache$cuda
tests/testthat/test-gpu.R:  on.exit(cache$cuda <- NULL)
tests/testthat/test-gpu.R:  cfg1 <- list(has_cuda = FALSE)
tests/testthat/test-gpu.R:  cfg2 <- example_cuda_config()
tests/testthat/test-gpu.R:  cache$cuda <- NULL
tests/testthat/test-gpu.R:  mock_cuda_configuration <- mockery::mock(cfg1, cfg2)
tests/testthat/test-gpu.R:  mockery::stub(dust_cuda_configuration, "cuda_configuration",
tests/testthat/test-gpu.R:                mock_cuda_configuration)
tests/testthat/test-gpu.R:    dust_cuda_configuration(path_lib, path_include, FALSE, TRUE),
tests/testthat/test-gpu.R:  expect_identical(cache$cuda, cfg1)
tests/testthat/test-gpu.R:  mockery::expect_called(mock_cuda_configuration, 1)
tests/testthat/test-gpu.R:    mockery::mock_args(mock_cuda_configuration)[[1]],
tests/testthat/test-gpu.R:    dust_cuda_configuration(path_lib, path_include, FALSE),
tests/testthat/test-gpu.R:  mockery::expect_called(mock_cuda_configuration, 1)
tests/testthat/test-gpu.R:    dust_cuda_configuration(path_lib, path_include, FALSE, TRUE),
tests/testthat/test-gpu.R:  expect_identical(cache$cuda, cfg2)
tests/testthat/test-gpu.R:  mockery::expect_called(mock_cuda_configuration, 2)
tests/testthat/test-gpu.R:    mockery::mock_args(mock_cuda_configuration)[[1]],
tests/testthat/test-gpu.R:    dust_cuda_configuration(path_lib, path_include, FALSE),
tests/testthat/test-gpu.R:  mockery::expect_called(mock_cuda_configuration, 2)
tests/testthat/test-gpu.R:test_that("high level interface to cuda options", {
tests/testthat/test-gpu.R:  cfg1 <- example_cuda_config()
tests/testthat/test-gpu.R:  cfg2 <- list(has_cuda = FALSE)
tests/testthat/test-gpu.R:  mock_cuda_configuration <- mockery::mock(cfg1, cfg2)
tests/testthat/test-gpu.R:  mockery::stub(dust_cuda_options, "dust_cuda_configuration",
tests/testthat/test-gpu.R:                mock_cuda_configuration)
tests/testthat/test-gpu.R:  path_lib <- "/path/cuda/lib"
tests/testthat/test-gpu.R:  res <- dust_cuda_options(path_cuda_lib = path_lib)
tests/testthat/test-gpu.R:  expect_identical(res, cuda_options(cfg1, FALSE, FALSE, FALSE, NULL))
tests/testthat/test-gpu.R:  mockery::expect_called(mock_cuda_configuration, 1)
tests/testthat/test-gpu.R:    mockery::mock_args(mock_cuda_configuration)[[1]],
tests/testthat/test-gpu.R:    list(path_cuda_lib = path_lib))
tests/testthat/test-gpu.R:    dust_cuda_options(path_cuda_lib = path_lib),
tests/testthat/test-gpu.R:    "cuda not supported on this machine")
tests/testthat/test-gpu.R:  mockery::expect_called(mock_cuda_configuration, 2)
tests/testthat/test-gpu.R:    mockery::mock_args(mock_cuda_configuration)[[2]],
tests/testthat/test-gpu.R:    list(path_cuda_lib = path_lib))
tests/testthat/test-gpu.R:    gen$new(list(len = len), 0, np, gpu_config = 2),
tests/testthat/test-gpu.R:    gen$new(list(len = len), 0, np, gpu_config = -1),
tests/testthat/test-gpu.R:  mod <- gen$new(list(len = len), 0, np, gpu_config = NULL)
tests/testthat/test-gpu.R:  expect_equal(r6_private(mod)$gpu_config_$device_id, NULL)
tests/testthat/test-gpu.R:  mod <- gen$new(list(len = len), 0, np, gpu_config = 0L)
tests/testthat/test-gpu.R:  expect_equal(r6_private(mod)$gpu_config_$device_id, 0)
tests/testthat/test-gpu.R:                 gpu_config = list(device_id = 0, run_block_size = 512))
tests/testthat/test-gpu.R:  expect_equal(r6_private(mod)$gpu_config_,
tests/testthat/test-gpu.R:               list(real_gpu = FALSE,
tests/testthat/test-gpu.R:test_that("Can use sirs gpu model", {
tests/testthat/test-gpu.R:  mod2 <- gen$new(list(), 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:test_that("Can simulate sirs gpu model", {
tests/testthat/test-gpu.R:  mod_d <- res$new(list(), 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:test_that("Comparison function can be run on the GPU", {
tests/testthat/test-gpu.R:  mod_d <- dat$model$new(list(), 0, np, seed = 10L, gpu_config = 0L)
tests/testthat/test-gpu.R:test_that("Can run a single particle filter on the GPU", {
tests/testthat/test-gpu.R:  mod_d <- dat$model$new(list(), 0, np, seed = 10L, gpu_config = 0L)
tests/testthat/test-gpu.R:test_that("Can run particle filter without collecting state on GPU", {
tests/testthat/test-gpu.R:  mod_d <- dat$model$new(list(), 0, np, seed = 10L, gpu_config = 0L)
tests/testthat/test-gpu.R:test_that("Can run GPU kernels using shared memory", {
tests/testthat/test-gpu.R:  mod_d <- dat$model$new(list(), 0, np, seed = 10L, gpu_config = 0L)
tests/testthat/test-gpu.R:test_that("Can run multiple particle filters on the GPU", {
tests/testthat/test-gpu.R:                         gpu_config = 0L)
tests/testthat/test-gpu.R:  mod2 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:  mod4 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:  res <- test_cuda_pars(0, 2000, 2000,
tests/testthat/test-gpu.R:  res <- test_cuda_pars(0, 2000, 2000,
tests/testthat/test-gpu.R:  res <- test_cuda_pars(0, 2000, 2000,
tests/testthat/test-gpu.R:  res <- test_cuda_pars(0, 2000, 2000,
tests/testthat/test-gpu.R:  res <- test_cuda_pars(config, 2000, 2000,
tests/testthat/test-gpu.R:    res <- test_cuda_pars(list(device_id = 0, run_block_size = block_size),
tests/testthat/test-gpu.R:    test_cuda_pars(config, 2000, 2000,
tests/testthat/test-gpu.R:    test_cuda_pars(config, 2000, 2000,
tests/testthat/test-gpu.R:test_that("Can't run deterministically on the gpu", {
tests/testthat/test-gpu.R:            gpu_config = 0L),
tests/testthat/test-gpu.R:    "Deterministic models not supported on gpu")
tests/testthat/test-gpu.R:  mod1 <- gen$new(pa, 0, np, seed = 1L, gpu_config = NULL)
tests/testthat/test-gpu.R:  mod2 <- gen$new(pa, 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:  mod1 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = NULL)
tests/testthat/test-gpu.R:  mod2 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:  mod1 <- gen$new(pa, 0, np, seed = 1L, gpu_config = NULL)
tests/testthat/test-gpu.R:  mod2 <- gen$new(pa, 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:  mod1 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = NULL)
tests/testthat/test-gpu.R:  mod2 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:  mod <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:  mod1 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = NULL)
tests/testthat/test-gpu.R:  mod2 <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:  mod <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:  obj <- res$new(list(len = 5), 0, 7, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:  obj <- res$new(list(len = 5), 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:    res$new(p, 0, 5, seed = 1L, pars_multi = TRUE, gpu_config = 0L),
tests/testthat/test-gpu.R:test_that("Can't set vector of times into gpu object", {
tests/testthat/test-gpu.R:  mod <- gen$new(list(len = len), 0, np, seed = 1L, gpu_config = 0L)
tests/testthat/test-gpu.R:  mod1 <- gen$new(pa, 0, np, seed = 1L, pars_multi = TRUE, gpu_config = NULL)
tests/testthat/test-gpu.R:  mod2 <- gen$new(pa, 0, np, seed = 1L, pars_multi = TRUE, gpu_config = 0L)
tests/testthat/test-gpu.R:  mod1 <- gen$new(pa, 0, np, seed = 1L, pars_multi = TRUE, gpu_config = NULL)
tests/testthat/test-gpu.R:  mod2 <- gen$new(pa, 0, np, seed = 1L, pars_multi = TRUE, gpu_config = 0L)
tests/testthat/test-gpu.R:  mod <- gen$new(p, 0, np, seed = 1L, pars_multi = TRUE, gpu_config = 0L)
tests/testthat/test-gpu.R:    "Can't use index with gpu models")
tests/testthat/test-gpu.R:  mod <- res$new(list(sd = 1), 0, 5, gpu_config = 0L)
tests/testthat/test-gpu.R:  mod <- res$new(list(sd = 1), 4, 5, gpu_config = 0L)
tests/testthat/test-gpu.R:test_that("Can partially run filter for the gpu model", {
tests/testthat/test-gpu.R:  mod_d <- dat$model$new(list(), 0, np, seed = 10L, gpu_config = 0L)
tests/testthat/test-gpu.R:                         gpu_config = 0L)
tests/testthat/test-gpu.R:test_that("Can run multiple particle filters with shared data on the GPU", {
tests/testthat/test-gpu.R:                         gpu_config = 0L)
tests/testthat/test-gpu.R:    has_cuda = TRUE,
tests/testthat/test-gpu.R:    cuda_version = numeric_version("10.1.0"),
tests/testthat/test-gpu.R:    path_cuda_lib = "/path/to/cuda",
tests/testthat/test-gpu.R:  cuda <- cuda_options(info, FALSE, FALSE, FALSE, NULL)
tests/testthat/test-gpu.R:    cuda$flags$gencode,
tests/testthat/test-gpu.R:test_that("don't accept ode control for gpu models", {
tests/testthat/test-gpu.R:    gen$new(list(len = 4), 0, 100, seed = 1L, gpu_config = 0L,
tests/testthat/test-ode-interface.R:test_that("prevent use of gpu", {
tests/testthat/test-ode-interface.R:    ex$generator$new(ex$pars, 0, 1, gpu_config = 1),
tests/testthat/test-ode-interface.R:    "GPU support not enabled for this object")
tests/testthat/test-ode-interface.R:  expect_false(mod$has_gpu_support())
tests/testthat/test-ode-interface.R:  expect_false(mod$has_gpu_support(TRUE))
tests/testthat/test-ode-interface.R:test_that("can retrieve empty gpu info", {
tests/testthat/test-ode-interface.R:  ## Another dust model that lacks gpu information; use this as an
tests/testthat/test-ode-interface.R:  expected <- dust::dust_example("sir")$new(list(), 0, 1)$gpu_info()
tests/testthat/test-ode-interface.R:  expect_equal(mod$gpu_info(), expected)
tests/testthat/test-metadata.R:test_that("force gpu state", {
tests/testthat/test-metadata.R:    "// [[dust::has_gpu_support(true)]]")
tests/testthat/test-metadata.R:  expect_true(parse_metadata(tmp1)$has_gpu_support)
tests/testthat/test-metadata.R:    "// [[dust::has_gpu_support(sounds_good)]]")
tests/testthat/test-metadata.R:    "Invalid value for dust::has_gpu_support, expected logical")
tests/testthat/examples/init.cpp:namespace gpu {
tests/testthat/examples/init.cpp:  using dust::gpu::shared_copy_data;
tests/testthat/examples/init.cpp:void update_gpu<walk>(size_t time,
tests/testthat/examples/init.cpp:                      const dust::gpu::interleaved<walk::real_type> state,
tests/testthat/examples/init.cpp:                      dust::gpu::interleaved<int> internal_int,
tests/testthat/examples/init.cpp:                      dust::gpu::interleaved<walk::real_type> internal_real,
tests/testthat/examples/init.cpp:                      dust::gpu::interleaved<walk::real_type> state_next) {
tests/testthat/test-interface.R:                               has_gpu_support = "FALSE",
tests/testthat/test-interface.R:                               methods_gpu = "list()")),
tests/testthat/helper-cuda.R:example_cuda_config <- function() {
tests/testthat/helper-cuda.R:  list(has_cuda = TRUE,
tests/testthat/helper-cuda.R:       cuda_version = numeric_version("11.1.243"),
tests/testthat/helper-cuda.R:  code <- sprintf("dust_gpu_info <- function() %s",
tests/testthat/helper-cuda.R:                  paste(deparse(example_cuda_config()), collapse = "\n"))
vignettes_src/gpu.Rmd:title: "Running models on GPUs with CUDA"
vignettes_src/gpu.Rmd:  %\VignetteIndexEntry{Running models on GPUs with CUDA}
vignettes_src/gpu.Rmd:With the core approach in dust, you can run models in parallel efficiently up to the number of cores your workstation has available.  Getting more than 32 or 64 cores is hard though, and `dust` provides no multi-node parallelism (e.g., [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)). Instead, we have developed a system for running `dust` models on GPUs (graphical processing units), specifically NVIDIA GPUs via CUDA (Compute Unified Device Architecture).
vignettes_src/gpu.Rmd:This vignette is written in reverse-order, starting with how to run a model on a GPU, before covering how to write a `dust` model that can be run on a GPU. The reason for this is that we do not expect that people will write these directly; instead we expect that most people will use the [`odin.dust`](https://mrc-ide.github.io/odin.dust/) package to generate these interfaces automatically, without having to write a single line of C++ or CUDA code.
vignettes_src/gpu.Rmd:* A GPU model can be run on the CPU, and vice-versa.
vignettes_src/gpu.Rmd:* Just the model update can be defined for the GPU, and comparisons and shuffling could still happen on the CPU. Defining a comparison function is also possible, allowing a full particle filter run on a GPU.
vignettes_src/gpu.Rmd:## Running a model with GPU support
vignettes_src/gpu.Rmd:The `sirs` model includes GPU support, and will be the focus of this vignette.  However, the installed version cannot be run directly on the GPU for a couple of reasons:
vignettes_src/gpu.Rmd:* this would complicate distribution of binaries as we'd depend on all systems having a copy of the CUDA runtime
vignettes_src/gpu.Rmd:* we would have to compile support for many possible GPU architectures at once
vignettes_src/gpu.Rmd:* CUDA toolkit v10.2 or higher, v11.1 or higher preferred (compile time and run time)
vignettes_src/gpu.Rmd:* CUDA capable GPU (run time)
vignettes_src/gpu.Rmd:* nvidia drivers (run time)
vignettes_src/gpu.Rmd:You can check with the command-line tool `nvidia-smi` if you have suitable hardware and drivers and with `dust::dust_cuda_configuration(quiet = FALSE, forget = TRUE)` if you have suitable build tools.
vignettes_src/gpu.Rmd:sirs <- dust::dust(path, gpu = TRUE, real_type = "float")
vignettes_src/gpu.Rmd:Notice in compilation that `nvcc` is used to compile the model, rather than `g++` or `clang++`.  The additional option `-gencode=arch=compute_XX,code=sm_XX` was added by `dust` and will include the CUDA compute version supported by the graphics cards found on the current system. You can use `dust::dust_cuda_options` to set additional options, passing in the return value for the `gpu` argument above.
vignettes_src/gpu.Rmd:Once compiled with GPU support, the static method `has_gpu_support` will report `TRUE`:
vignettes_src/gpu.Rmd:sirs$public_methods$has_gpu_support()
vignettes_src/gpu.Rmd:and the static method `gpu_info` will report on the GPU devices available:
vignettes_src/gpu.Rmd:sirs$public_methods$gpu_info()
vignettes_src/gpu.Rmd:If you have more than one GPU, the `id` in the `devices` section will be useful for targeting the correct device.
vignettes_src/gpu.Rmd:* you will probably want a (much) larger number of particles to take advantage of your GPU. As a rule of thumb we would suggest at least 10,000, but depending on model and card you may still see per-particle increases in compute speed as you use up to 1,000,000 particles. See below for more discussion of this.
vignettes_src/gpu.Rmd:* the `gpu_config` argument needs to be provided to indicate which GPU device we are running on. Minimally this is an integer indicating the device that you want to use (on this machine the only option is `0`), but you can also provide a list with elements `device_id` and `run_block_size`.
vignettes_src/gpu.Rmd:model_gpu <- sirs$new(pars, 0, n_particles, gpu_config = 0L, seed = 1L)
vignettes_src/gpu.Rmd:Once initialised, a model can only be run on either the GPU or CPU, so we'll create a CPU version here for comparison:
vignettes_src/gpu.Rmd:By leaving `gpu_config` as `NULL` we indicate that the model should run on the CPU.
vignettes_src/gpu.Rmd:Once created, the `uses_gpu` method indicates if the model is set up to run on the GPU (rather than CPU):
vignettes_src/gpu.Rmd:model_gpu$uses_gpu()
vignettes_src/gpu.Rmd:model_cpu$uses_gpu()
vignettes_src/gpu.Rmd:Running the model on the GPU however:
vignettes_src/gpu.Rmd:(t_gpu <- system.time(model_gpu$run(400)))
vignettes_src/gpu.Rmd:This is much faster! However, ~8,000 particles is unlikely to saturate a modern GPU and (overhead-aside) this will run about as quickly for potentially a hundred thousand particles. For example running 2^17 (131,072) particles only takes a little longer
vignettes_src/gpu.Rmd:model_large <- sirs$new(list(), 0, 2^17, gpu_config = 0L, seed = 1L)
vignettes_src/gpu.Rmd:This is **heaps** faster, the GPU model ran in `r round(ratio * 100, 1)`% of the time as the CPU model but simulated `r 2^17 / n_particles` times as many particles (i.e., `r round(2^17 / n_particles / ratio)` times as fast per particle). With the relatively low times here, much of this time is just moving the data around, and with over a hundred thousand particles this is nontrivial.  Of course, _doing_ anything quickly with all these particles is its own problem.
vignettes_src/gpu.Rmd:All methods will automatically run on the GPU; this includes `run`, `simulate`, `compare_data` and `filter`.  The last two are typically used from the [`mcstate` interface](https://mrc-ide.github.io/mcstate/reference/particle_filter.html).
vignettes_src/gpu.Rmd:## Writing a GPU-capable model
vignettes_src/gpu.Rmd:This is somewhat more complicated than the models described in `vignette("dust.Rmd")`. There are several important components required to run on the GPU.
vignettes_src/gpu.Rmd:Within the `dust::gpu` namespace, we declare the size of the *shared parameters* for the model. These are parameters that will be the same across all instances of a parameter set, as opposed to quantities that change between particles.
vignettes_src/gpu.Rmd:namespace gpu {
vignettes_src/gpu.Rmd:namespace gpu {
vignettes_src/gpu.Rmd:  using dust::gpu::shared_copy_data;
vignettes_src/gpu.Rmd:In the CPU version of the model we have a nice smart pointer to a struct (`dust::shared_ptr<sirs>`) from which we can access parameters by name (e.g., `shared->alpha`). No such niceties in CUDA where we need access to a single contiguous block of memory.  The `dust::shared_copy_data` is a small utility to make the bookkeeping here a bit easier, but this could have been written out as:
vignettes_src/gpu.Rmd:namespace gpu {
vignettes_src/gpu.Rmd:Most interestingly we have the `update_gpu` method that actually does the update on the GPU
vignettes_src/gpu.Rmd:void update_gpu<sirs>(size_t time,
vignettes_src/gpu.Rmd:                      const dust::gpu::interleaved<sirs::real_type> state,
vignettes_src/gpu.Rmd:                      dust::gpu::interleaved<int> internal_int,
vignettes_src/gpu.Rmd:                      dust::gpu::interleaved<sirs::real_type> internal_real,
vignettes_src/gpu.Rmd:                      dust::gpu::interleaved<sirs::real_type> state_next) {
vignettes_src/gpu.Rmd:* the data types that vary across particles are a special `dust::gpu::interleaved<>` type, which prevents slow uncoalesced reads from global memory on the GPU
vignettes_src/gpu.Rmd:* The `__device__` annotation, which compiles the function for use on the GPU
vignettes_src/gpu.Rmd:Finally, if running a particle filter on the GPU, a version of the `compare_data` function is required that can run on the GPU:
vignettes_src/gpu.Rmd:sirs::real_type compare_gpu<sirs>(const dust::gpu::interleaved<sirs::real_type> state,
vignettes_src/gpu.Rmd:                                  dust::gpu::interleaved<int> internal_int,
vignettes_src/gpu.Rmd:                                  dust::gpu::interleaved<sirs::real_type> internal_real,
vignettes_src/gpu.Rmd:## Developing a GPU model
vignettes_src/gpu.Rmd:Debugging on a GPU is a pain, especially because there are typically many particles, and error recovery is not straightforward.  In addition, most continuous integration systems do not provide GPUs, so testing your GPU code becomes difficult.  To make this easier, `dust` allows running GPU code on the CPU - this will be typically slower than the CPU code, but allows easier debugging and verification that the model is behaving.  We use this extensively in `dust`'s tests and also in models built using `dust` that will run on the GPU.
vignettes_src/gpu.Rmd:To do this, compile the model with your preferred real type, but set the `gpu` argument to `FALSE`
vignettes_src/gpu.Rmd:sirs_cpu <- dust::dust(path, gpu = FALSE, real_type = "float")
vignettes_src/gpu.Rmd:Note that above the model is compiled with `g++`, not `nvcc`. However, the "GPU" code is still compiled into the model.  We can then initialise this with and without a `gpu_config` argument
vignettes_src/gpu.Rmd:model2 <- sirs_cpu$new(pars, 0, 10, gpu_config = 0L, seed = 1L)
vignettes_src/gpu.Rmd:And run the models using the "GPU" code and the normal CPU code, but this time both on the CPU
vignettes_src/gpu.Rmd:If you hit problems, then `R -d cuda-gdb` can work in place of the usual `R -d gdb` to work with a debugger, though because of the huge numbers of particles you will typically work with, debugging remains a challenge.  Our usual strategy has been to try and recreate any issue purely in CPU code and debug as normal (see [this blog post](https://reside-ic.github.io/blog/debugging-memory-errors-with-valgrind-and-gdb/) for hints on doing this effectively).
Makefile:vignettes/gpu.Rmd: vignettes_src/gpu.Rmd
Makefile:	./scripts/build_vignette gpu
vignettes/data.Rmd:It is possible to implement this comparison directly in the dust model, which may slightly speed up the particle filter (because the compare function will be evaluated in parallel, and because of slightly reduced data copying), but also allows running the particle filter on a GPU (see `vignette("gpu")`).
vignettes/data.Rmd:This vignette outlines the steps in implementing the comparison directly as part of the model.  This is not required for basic use of dust models, and we would typically recommend this only after your model has stabilised and you are looking to extract potential additional speed-ups or accelerate the model on a GPU.
vignettes/data.Rmd:Some additional work is required if you want to run the comparison on a GPU, see `vignette("gpu")` for more details.
vignettes/gpu.Rmd:title: "Running models on GPUs with CUDA"
vignettes/gpu.Rmd:  %\VignetteIndexEntry{Running models on GPUs with CUDA}
vignettes/gpu.Rmd:With the core approach in dust, you can run models in parallel efficiently up to the number of cores your workstation has available.  Getting more than 32 or 64 cores is hard though, and `dust` provides no multi-node parallelism (e.g., [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)). Instead, we have developed a system for running `dust` models on GPUs (graphical processing units), specifically NVIDIA GPUs via CUDA (Compute Unified Device Architecture).
vignettes/gpu.Rmd:This vignette is written in reverse-order, starting with how to run a model on a GPU, before covering how to write a `dust` model that can be run on a GPU. The reason for this is that we do not expect that people will write these directly; instead we expect that most people will use the [`odin.dust`](https://mrc-ide.github.io/odin.dust/) package to generate these interfaces automatically, without having to write a single line of C++ or CUDA code.
vignettes/gpu.Rmd:* A GPU model can be run on the CPU, and vice-versa.
vignettes/gpu.Rmd:* Just the model update can be defined for the GPU, and comparisons and shuffling could still happen on the CPU. Defining a comparison function is also possible, allowing a full particle filter run on a GPU.
vignettes/gpu.Rmd:## Running a model with GPU support
vignettes/gpu.Rmd:The `sirs` model includes GPU support, and will be the focus of this vignette.  However, the installed version cannot be run directly on the GPU for a couple of reasons:
vignettes/gpu.Rmd:* this would complicate distribution of binaries as we'd depend on all systems having a copy of the CUDA runtime
vignettes/gpu.Rmd:* we would have to compile support for many possible GPU architectures at once
vignettes/gpu.Rmd:* CUDA toolkit v10.2 or higher, v11.1 or higher preferred (compile time and run time)
vignettes/gpu.Rmd:* CUDA capable GPU (run time)
vignettes/gpu.Rmd:* nvidia drivers (run time)
vignettes/gpu.Rmd:You can check with the command-line tool `nvidia-smi` if you have suitable hardware and drivers and with `dust::dust_cuda_configuration(quiet = FALSE, forget = TRUE)` if you have suitable build tools.
vignettes/gpu.Rmd:sirs <- dust::dust(path, gpu = TRUE, real_type = "float")
vignettes/gpu.Rmd:#> Re-compiling sirs817bc717gpu
vignettes/gpu.Rmd:#>   ─  installing *source* package ‘sirs817bc717gpu’ ...
vignettes/gpu.Rmd:#>      g++ -std=gnu++14 -shared -L/usr/lib/R/lib -Wl,-Bsymbolic-functions -Wl,-z,relro -o sirs817bc717gpu.so cpp11.o dust.o -lcudart -fopenmp -L/usr/lib/R/lib -lR
vignettes/gpu.Rmd:#>      installing to /tmp/RtmppiOue1/devtools_install_b32bb69aa2d4d/00LOCK-fileb32bb676504e5/00new/sirs817bc717gpu/libs
vignettes/gpu.Rmd:#>   ─  DONE (sirs817bc717gpu)
vignettes/gpu.Rmd:#> ℹ Loading sirs817bc717gpu
vignettes/gpu.Rmd:Notice in compilation that `nvcc` is used to compile the model, rather than `g++` or `clang++`.  The additional option `-gencode=arch=compute_XX,code=sm_XX` was added by `dust` and will include the CUDA compute version supported by the graphics cards found on the current system. You can use `dust::dust_cuda_options` to set additional options, passing in the return value for the `gpu` argument above.
vignettes/gpu.Rmd:Once compiled with GPU support, the static method `has_gpu_support` will report `TRUE`:
vignettes/gpu.Rmd:sirs$public_methods$has_gpu_support()
vignettes/gpu.Rmd:and the static method `gpu_info` will report on the GPU devices available:
vignettes/gpu.Rmd:sirs$public_methods$gpu_info()
vignettes/gpu.Rmd:#> $has_cuda
vignettes/gpu.Rmd:#> $cuda_version
vignettes/gpu.Rmd:If you have more than one GPU, the `id` in the `devices` section will be useful for targeting the correct device.
vignettes/gpu.Rmd:* you will probably want a (much) larger number of particles to take advantage of your GPU. As a rule of thumb we would suggest at least 10,000, but depending on model and card you may still see per-particle increases in compute speed as you use up to 1,000,000 particles. See below for more discussion of this.
vignettes/gpu.Rmd:* the `gpu_config` argument needs to be provided to indicate which GPU device we are running on. Minimally this is an integer indicating the device that you want to use (on this machine the only option is `0`), but you can also provide a list with elements `device_id` and `run_block_size`.
vignettes/gpu.Rmd:model_gpu <- sirs$new(pars, 0, n_particles, gpu_config = 0L, seed = 1L)
vignettes/gpu.Rmd:Once initialised, a model can only be run on either the GPU or CPU, so we'll create a CPU version here for comparison:
vignettes/gpu.Rmd:By leaving `gpu_config` as `NULL` we indicate that the model should run on the CPU.
vignettes/gpu.Rmd:Once created, the `uses_gpu` method indicates if the model is set up to run on the GPU (rather than CPU):
vignettes/gpu.Rmd:model_gpu$uses_gpu()
vignettes/gpu.Rmd:model_cpu$uses_gpu()
vignettes/gpu.Rmd:Running the model on the GPU however:
vignettes/gpu.Rmd:(t_gpu <- system.time(model_gpu$run(400)))
vignettes/gpu.Rmd:This is much faster! However, ~8,000 particles is unlikely to saturate a modern GPU and (overhead-aside) this will run about as quickly for potentially a hundred thousand particles. For example running 2^17 (131,072) particles only takes a little longer
vignettes/gpu.Rmd:model_large <- sirs$new(list(), 0, 2^17, gpu_config = 0L, seed = 1L)
vignettes/gpu.Rmd:This is **heaps** faster, the GPU model ran in 7.3% of the time as the CPU model but simulated 16 times as many particles (i.e., 219 times as fast per particle). With the relatively low times here, much of this time is just moving the data around, and with over a hundred thousand particles this is nontrivial.  Of course, _doing_ anything quickly with all these particles is its own problem.
vignettes/gpu.Rmd:All methods will automatically run on the GPU; this includes `run`, `simulate`, `compare_data` and `filter`.  The last two are typically used from the [`mcstate` interface](https://mrc-ide.github.io/mcstate/reference/particle_filter.html).
vignettes/gpu.Rmd:## Writing a GPU-capable model
vignettes/gpu.Rmd:namespace gpu {
vignettes/gpu.Rmd:  using dust::gpu::shared_copy_data;
vignettes/gpu.Rmd:void update_gpu<sirs>(size_t time,
vignettes/gpu.Rmd:                      const dust::gpu::interleaved<sirs::real_type> state,
vignettes/gpu.Rmd:                      dust::gpu::interleaved<int> internal_int,
vignettes/gpu.Rmd:                      dust::gpu::interleaved<sirs::real_type> internal_real,
vignettes/gpu.Rmd:                      dust::gpu::interleaved<sirs::real_type> state_next) {
vignettes/gpu.Rmd:sirs::real_type compare_gpu<sirs>(const dust::gpu::interleaved<sirs::real_type> state,
vignettes/gpu.Rmd:                                  dust::gpu::interleaved<int> internal_int,
vignettes/gpu.Rmd:                                  dust::gpu::interleaved<sirs::real_type> internal_real,
vignettes/gpu.Rmd:This is somewhat more complicated than the models described in `vignette("dust.Rmd")`. There are several important components required to run on the GPU.
vignettes/gpu.Rmd:Within the `dust::gpu` namespace, we declare the size of the *shared parameters* for the model. These are parameters that will be the same across all instances of a parameter set, as opposed to quantities that change between particles.
vignettes/gpu.Rmd:namespace gpu {
vignettes/gpu.Rmd:namespace gpu {
vignettes/gpu.Rmd:  using dust::gpu::shared_copy_data;
vignettes/gpu.Rmd:In the CPU version of the model we have a nice smart pointer to a struct (`dust::shared_ptr<sirs>`) from which we can access parameters by name (e.g., `shared->alpha`). No such niceties in CUDA where we need access to a single contiguous block of memory.  The `dust::shared_copy_data` is a small utility to make the bookkeeping here a bit easier, but this could have been written out as:
vignettes/gpu.Rmd:namespace gpu {
vignettes/gpu.Rmd:Most interestingly we have the `update_gpu` method that actually does the update on the GPU
vignettes/gpu.Rmd:void update_gpu<sirs>(size_t time,
vignettes/gpu.Rmd:                      const dust::gpu::interleaved<sirs::real_type> state,
vignettes/gpu.Rmd:                      dust::gpu::interleaved<int> internal_int,
vignettes/gpu.Rmd:                      dust::gpu::interleaved<sirs::real_type> internal_real,
vignettes/gpu.Rmd:                      dust::gpu::interleaved<sirs::real_type> state_next) {
vignettes/gpu.Rmd:* the data types that vary across particles are a special `dust::gpu::interleaved<>` type, which prevents slow uncoalesced reads from global memory on the GPU
vignettes/gpu.Rmd:* The `__device__` annotation, which compiles the function for use on the GPU
vignettes/gpu.Rmd:Finally, if running a particle filter on the GPU, a version of the `compare_data` function is required that can run on the GPU:
vignettes/gpu.Rmd:sirs::real_type compare_gpu<sirs>(const dust::gpu::interleaved<sirs::real_type> state,
vignettes/gpu.Rmd:                                  dust::gpu::interleaved<int> internal_int,
vignettes/gpu.Rmd:                                  dust::gpu::interleaved<sirs::real_type> internal_real,
vignettes/gpu.Rmd:## Developing a GPU model
vignettes/gpu.Rmd:Debugging on a GPU is a pain, especially because there are typically many particles, and error recovery is not straightforward.  In addition, most continuous integration systems do not provide GPUs, so testing your GPU code becomes difficult.  To make this easier, `dust` allows running GPU code on the CPU - this will be typically slower than the CPU code, but allows easier debugging and verification that the model is behaving.  We use this extensively in `dust`'s tests and also in models built using `dust` that will run on the GPU.
vignettes/gpu.Rmd:To do this, compile the model with your preferred real type, but set the `gpu` argument to `FALSE`
vignettes/gpu.Rmd:sirs_cpu <- dust::dust(path, gpu = FALSE, real_type = "float")
vignettes/gpu.Rmd:Note that above the model is compiled with `g++`, not `nvcc`. However, the "GPU" code is still compiled into the model.  We can then initialise this with and without a `gpu_config` argument
vignettes/gpu.Rmd:model2 <- sirs_cpu$new(pars, 0, 10, gpu_config = 0L, seed = 1L)
vignettes/gpu.Rmd:And run the models using the "GPU" code and the normal CPU code, but this time both on the CPU
vignettes/gpu.Rmd:If you hit problems, then `R -d cuda-gdb` can work in place of the usual `R -d gdb` to work with a debugger, though because of the huge numbers of particles you will typically work with, debugging remains a challenge.  Our usual strategy has been to try and recreate any issue purely in CPU code and debug as normal (see [this blog post](https://reside-ic.github.io/blog/debugging-memory-errors-with-valgrind-and-gdb/) for hints on doing this effectively).
vignettes/rng.Rmd:If you are generating single precision `float` numbers for a GPU, then you may want to use the `xoshiro128` family as these will be faster than the 64 bit generators on that platform.  On a CPU you will likely not see a difference
vignettes/rng.Rmd:### Standalone, parallel on a GPU
vignettes/rng.Rmd:This is considerably more complicated and will depend on your aims with your GPU-accelerated program.
vignettes/rng.Rmd:* possible to use on GPUs; this exposes some exotic issues around what types of structures can be used
vignettes/rng.Rmd:The situation is slightly more complex on a GPU where we need to make sure that different threads within a block do not get needlessly onto different branches of conditional logic.  Things like early exits need to be avoided and we have carefully profiled to make sure that threads within a warp end up re-synchronised at the optimal spot (see for example [this blog post on `__syncwarp`](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)).
README.md:* Running models massively in parallel on GPU without writing any low-level code
README.md:* `vignette("gpu")` on creating and running models on GPUs
scripts/update_dust_math:#ifdef __CUDA_ARCH__
scripts/update_dust_math:#ifdef __CUDA_ARCH__
scripts/update_dust_generator:                                     has_gpu_support = "",
scripts/update_dust_generator:                                     methods_gpu = ""))
docker/build_cuda:DUST_CUDA="${PACKAGE_ORG}/${PACKAGE_NAME}-cuda:${GIT_SHA}"
docker/build_cuda:       --tag $DUST_CUDA \
docker/build_cuda:       -f $PACKAGE_ROOT/docker/Dockerfile.cuda \
docker/build_cuda:docker run -it --rm -v $PWD:/src:ro $DUST_CUDA /compile_gpu_model
docker/compile_gpu_model:message("Checking CUDA configuration")
docker/compile_gpu_model:dust::dust_cuda_configuration(quiet = FALSE)
docker/compile_gpu_model:gen <- dust::dust(path, gpu = TRUE)
docker/Dockerfile.cuda:        nvidia-cuda-toolkit \
docker/Dockerfile.cuda:RUN Rscript -e 'dust:::cuda_install_cub(NULL)'
docker/Dockerfile.cuda:COPY docker/compile_gpu_model /
NEWS.md:* Added support for drawing normally distributed random numbers using the ziggurat method. Currently this is only efficient on a CPU, though supported on a GPU (#308)
NEWS.md:* Major header reorganisation, the `dust::cuda` namespace is now `dust::gpu`, most user-facing uses of `cuda` and `device` now replaced with `gpu` (#298, #317)
NEWS.md:* Rationalised the GPU interface, now once created models can only be used on either the GPU or CPU which simplifies the internal bookkeeping (#292, #302)
NEWS.md:* New option to `dust::dust` to skip the model cache, which may be useful when compiling with (say) different GPU options (#248)
NEWS.md:* Add two new vignettes covering model/data comparison and use on GPUs; see `vignette("data")` and `vignette("cuda")` (#183, #229)
NEWS.md:* Finer control over GPU settings, with the block size of `run()` now (optionally) exposed
NEWS.md:* On the GPU integers are kept in shared memory even where reals will no longer fit (#245)
NEWS.md:* Synchronise possible divergences in the density functions (CUDA only) (#243)
NEWS.md:* Fix a bug when running the CUDA version of the particle filter without
NEWS.md:* Add a CUDA version of the `simulate` method.
NEWS.md:* Added CUDA version of the particle filter, run with `model$filter(device = TRUE)` (#224)
NEWS.md:* Fix issue with `rnorm()` running on a GPU (device code).
NEWS.md:* Fix issue with unaligned shared copy in CUDA code.
NEWS.md:* Add GPU support (#73)
NEWS.md:* Remove prototype GPU interface, in preparation for a new version (#109)
NEWS.md:* Can now generate dust objects that run on the GPU (#69)
_pkgdown.yml:  - title: Compile on GPUs
_pkgdown.yml:      Control compilation options for creating models to run on a GPU
_pkgdown.yml:      - dust_cuda_configuration
_pkgdown.yml:      - dust_cuda_options
_pkgdown.yml:      - gpu
NAMESPACE:export(dust_cuda_configuration)
NAMESPACE:export(dust_cuda_options)
man/dust_cuda_configuration.Rd:% Please edit documentation in R/cuda.R
man/dust_cuda_configuration.Rd:\name{dust_cuda_configuration}
man/dust_cuda_configuration.Rd:\alias{dust_cuda_configuration}
man/dust_cuda_configuration.Rd:\title{Detect CUDA configuration}
man/dust_cuda_configuration.Rd:dust_cuda_configuration(
man/dust_cuda_configuration.Rd:  path_cuda_lib = NULL,
man/dust_cuda_configuration.Rd:\item{path_cuda_lib}{Optional path to the CUDA libraries, if they
man/dust_cuda_configuration.Rd:\verb{-L\{path_cuda_lib\}} in calls to \code{nvcc}}
man/dust_cuda_configuration.Rd:CUDA < 11.0.0. See Details}
man/dust_cuda_configuration.Rd:\item \code{has_cuda}: logical, indicating if it is possible to compile CUDA on
man/dust_cuda_configuration.Rd:\item \code{cuda_version}: the version of CUDA found
man/dust_cuda_configuration.Rd:\item \code{path_cuda_lib}: path to CUDA libraries, if required
man/dust_cuda_configuration.Rd:If compilation of the test program fails, then \code{has_cuda} will be
man/dust_cuda_configuration.Rd:Detect CUDA configuration. This function tries to compile a small
man/dust_cuda_configuration.Rd:your NVIDIA GPUs. If this works, then you can use the GPU-enabled
man/dust_cuda_configuration.Rd:Not all installations leave the CUDA libraries on the default
man/dust_cuda_configuration.Rd:to find \code{libcudart} then your CUDA libraries are not in the
man/dust_cuda_configuration.Rd:default location. You can manually pass in the \code{path_cuda_lib}
man/dust_cuda_configuration.Rd:argument, or set the \code{DUST_PATH_CUDA_LIB} environment variable (in
man/dust_cuda_configuration.Rd:If you are using older CUDA (< 11.0.0) then you need to provide
man/dust_cuda_configuration.Rd:manage state on the device (these are included in CUDA 11.0.0 and
man/dust_cuda_configuration.Rd:\if{html}{\out{<div class="sourceCode">}}\preformatted{dust:::cuda_install_cub(NULL)
man/dust_cuda_configuration.Rd:# If you have your CUDA library in an unusual location, then you
man/dust_cuda_configuration.Rd:# may need to add a path_cuda_lib argument:
man/dust_cuda_configuration.Rd:dust::dust_cuda_configuration(
man/dust_cuda_configuration.Rd:  path_cuda_lib = "/usr/local/cuda-11.1/lib64",
man/dust_cuda_configuration.Rd:dust::dust_cuda_configuration(forget = TRUE, quiet = FALSE)
man/dust_cuda_configuration.Rd:\link{dust_cuda_options} which controls additional CUDA
man/dust_cuda_options.Rd:% Please edit documentation in R/cuda.R
man/dust_cuda_options.Rd:\name{dust_cuda_options}
man/dust_cuda_options.Rd:\alias{dust_cuda_options}
man/dust_cuda_options.Rd:\title{Create CUDA options}
man/dust_cuda_options.Rd:dust_cuda_options(
man/dust_cuda_options.Rd:\item{...}{Arguments passed to \code{\link[=dust_cuda_configuration]{dust_cuda_configuration()}}}
man/dust_cuda_options.Rd:IEEE compliance and disables some error checking (see \href{https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html}{the CUDA docs}
man/dust_cuda_options.Rd:An object of type \code{cuda_options}, which can be passed into
man/dust_cuda_options.Rd:\link{dust} as argument \code{gpu}
man/dust_cuda_options.Rd:Create options for compiling for CUDA.  Unless you need to change
man/dust_cuda_options.Rd:  dust::dust_cuda_options(),
man/dust_cuda_options.Rd:\link{dust_cuda_configuration} which identifies and
man/dust_cuda_options.Rd:returns the core CUDA configuration (often used implicitly by
man/dust_generate.Rd:  gpu = FALSE,
man/dust_generate.Rd:\item{gpu}{Logical, indicating if we should generate GPU
man/dust_generate.Rd:installed (CUDA toolkit and drivers) as well as a
man/dust_generate.Rd:CUDA-compatible GPU. If \code{TRUE}, then we call
man/dust_generate.Rd:\link{dust_cuda_options} with no arguments. Alternatively, call
man/dust_generate.Rd:that function and pass the value here (e.g, \code{gpu = dust::dust_cuda_options(debug = TRUE)}). Note that due to the
man/dust_generate.Rd:use of the \verb{__syncwarp()} primitive this may require a GPU with
man/dust_generate.Rd:given type. This is primarily intended to be used as \verb{gpu = TRUE, real_type = "float"} in order to create model for the GPU
man/dust_generate.Rd:with CUDA models).
man/dust_generator.Rd:\item \href{#method-dust-has_gpu_support}{\code{dust_generator$has_gpu_support()}}
man/dust_generator.Rd:\item \href{#method-dust-uses_gpu}{\code{dust_generator$uses_gpu()}}
man/dust_generator.Rd:\item \href{#method-dust-gpu_info}{\code{dust_generator$gpu_info()}}
man/dust_generator.Rd:  gpu_config = NULL,
man/dust_generator.Rd:a a GPU.}
man/dust_generator.Rd:\item{\code{gpu_config}}{GPU configuration, typically an integer
man/dust_generator.Rd:indicating the device to use, where the model has GPU support.
man/dust_generator.Rd:that CUDA numbers devices from 0, so that '0' is the first device,
man/dust_generator.Rd:and so on). See the method \verb{$gpu_info()} for available device ids;
man/dust_generator.Rd:\code{dust_generator$public_methods$gpu_info()}.
man/dust_generator.Rd:\if{html}{\out{<a id="method-dust-has_gpu_support"></a>}}
man/dust_generator.Rd:\if{latex}{\out{\hypertarget{method-dust-has_gpu_support}{}}}
man/dust_generator.Rd:\subsection{Method \code{has_gpu_support()}}{
man/dust_generator.Rd:"CUDA" support, in which case it will react to the \code{device}
man/dust_generator.Rd:as \code{dust_generator$public_methods$has_gpu_support()}
man/dust_generator.Rd:\if{html}{\out{<div class="r">}}\preformatted{dust_generator$has_gpu_support(fake_gpu = FALSE)}\if{html}{\out{</div>}}
man/dust_generator.Rd:\item{\code{fake_gpu}}{Logical, indicating if we count as \code{TRUE}
man/dust_generator.Rd:models that run on the "fake" GPU (i.e., using the GPU
man/dust_generator.Rd:\if{html}{\out{<a id="method-dust-uses_gpu"></a>}}
man/dust_generator.Rd:\if{latex}{\out{\hypertarget{method-dust-uses_gpu}{}}}
man/dust_generator.Rd:\subsection{Method \code{uses_gpu()}}{
man/dust_generator.Rd:Check if the model is running on a GPU
man/dust_generator.Rd:\if{html}{\out{<div class="r">}}\preformatted{dust_generator$uses_gpu(fake_gpu = FALSE)}\if{html}{\out{</div>}}
man/dust_generator.Rd:\item{\code{fake_gpu}}{Logical, indicating if we count as \code{TRUE}
man/dust_generator.Rd:models that run on the "fake" GPU (i.e., using the GPU
man/dust_generator.Rd:\if{html}{\out{<a id="method-dust-gpu_info"></a>}}
man/dust_generator.Rd:\if{latex}{\out{\hypertarget{method-dust-gpu_info}{}}}
man/dust_generator.Rd:\subsection{Method \code{gpu_info()}}{
man/dust_generator.Rd:Return information about GPU devices, if the model
man/dust_generator.Rd:has been compiled with CUDA/GPU support. This can be called as a
man/dust_generator.Rd:static method by running \code{dust_generator$public_methods$gpu_info()}.
man/dust_generator.Rd:If run from a GPU enabled object, it will also have an element
man/dust_generator.Rd:\if{html}{\out{<div class="r">}}\preformatted{dust_generator$gpu_info()}\if{html}{\out{</div>}}
man/dust.Rd:  gpu = FALSE,
man/dust.Rd:\item{gpu}{Logical, indicating if we should generate GPU
man/dust.Rd:installed (CUDA toolkit and drivers) as well as a
man/dust.Rd:CUDA-compatible GPU. If \code{TRUE}, then we call
man/dust.Rd:\link{dust_cuda_options} with no arguments. Alternatively, call
man/dust.Rd:that function and pass the value here (e.g, \code{gpu = dust::dust_cuda_options(debug = TRUE)}). Note that due to the
man/dust.Rd:use of the \verb{__syncwarp()} primitive this may require a GPU with
man/dust.Rd:given type. This is primarily intended to be used as \verb{gpu = TRUE, real_type = "float"} in order to create model for the GPU
man/dust.Rd:Things are worse on a GPU; if an error is thrown by the RNG code
man/dust.Rd:then we currently use CUDA's \verb{__trap()} function which will
man/dust.Rd:the GPU again, covering all methods in the class.  However, this
src/cpp11.cpp:cpp11::sexp dust_logistic_gpu_info();
src/cpp11.cpp:extern "C" SEXP _dust_dust_logistic_gpu_info() {
src/cpp11.cpp:    return cpp11::as_sexp(dust_logistic_gpu_info());
src/cpp11.cpp:SEXP dust_ode_logistic_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time, cpp11::sexp r_n_particles, int n_threads, cpp11::sexp r_seed, bool deterministic, cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/cpp11.cpp:extern "C" SEXP _dust_dust_ode_logistic_alloc(SEXP r_pars, SEXP pars_multi, SEXP r_time, SEXP r_n_particles, SEXP n_threads, SEXP r_seed, SEXP deterministic, SEXP gpu_config, SEXP ode_control) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_ode_logistic_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_pars), cpp11::as_cpp<cpp11::decay_t<bool>>(pars_multi), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_time), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_n_particles), cpp11::as_cpp<cpp11::decay_t<int>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed), cpp11::as_cpp<cpp11::decay_t<bool>>(deterministic), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(gpu_config), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(ode_control)));
src/cpp11.cpp:cpp11::sexp dust_sir_gpu_info();
src/cpp11.cpp:extern "C" SEXP _dust_dust_sir_gpu_info() {
src/cpp11.cpp:    return cpp11::as_sexp(dust_sir_gpu_info());
src/cpp11.cpp:SEXP dust_cpu_sir_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time, cpp11::sexp r_n_particles, int n_threads, cpp11::sexp r_seed, bool deterministic, cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/cpp11.cpp:extern "C" SEXP _dust_dust_cpu_sir_alloc(SEXP r_pars, SEXP pars_multi, SEXP r_time, SEXP r_n_particles, SEXP n_threads, SEXP r_seed, SEXP deterministic, SEXP gpu_config, SEXP ode_control) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_cpu_sir_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_pars), cpp11::as_cpp<cpp11::decay_t<bool>>(pars_multi), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_time), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_n_particles), cpp11::as_cpp<cpp11::decay_t<int>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed), cpp11::as_cpp<cpp11::decay_t<bool>>(deterministic), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(gpu_config), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(ode_control)));
src/cpp11.cpp:cpp11::sexp dust_sirs_gpu_info();
src/cpp11.cpp:extern "C" SEXP _dust_dust_sirs_gpu_info() {
src/cpp11.cpp:    return cpp11::as_sexp(dust_sirs_gpu_info());
src/cpp11.cpp:SEXP dust_cpu_sirs_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time, cpp11::sexp r_n_particles, int n_threads, cpp11::sexp r_seed, bool deterministic, cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/cpp11.cpp:extern "C" SEXP _dust_dust_cpu_sirs_alloc(SEXP r_pars, SEXP pars_multi, SEXP r_time, SEXP r_n_particles, SEXP n_threads, SEXP r_seed, SEXP deterministic, SEXP gpu_config, SEXP ode_control) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_cpu_sirs_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_pars), cpp11::as_cpp<cpp11::decay_t<bool>>(pars_multi), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_time), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_n_particles), cpp11::as_cpp<cpp11::decay_t<int>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed), cpp11::as_cpp<cpp11::decay_t<bool>>(deterministic), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(gpu_config), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(ode_control)));
src/cpp11.cpp:SEXP dust_gpu_sirs_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time, cpp11::sexp r_n_particles, int n_threads, cpp11::sexp r_seed, bool deterministic, cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_alloc(SEXP r_pars, SEXP pars_multi, SEXP r_time, SEXP r_n_particles, SEXP n_threads, SEXP r_seed, SEXP deterministic, SEXP gpu_config, SEXP ode_control) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_pars), cpp11::as_cpp<cpp11::decay_t<bool>>(pars_multi), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_time), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_n_particles), cpp11::as_cpp<cpp11::decay_t<int>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed), cpp11::as_cpp<cpp11::decay_t<bool>>(deterministic), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(gpu_config), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(ode_control)));
src/cpp11.cpp:cpp11::sexp dust_gpu_sirs_capabilities();
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_capabilities() {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_capabilities());
src/cpp11.cpp:SEXP dust_gpu_sirs_run(SEXP ptr, cpp11::sexp r_time_end);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_run(SEXP ptr, SEXP r_time_end) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_run(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_time_end)));
src/cpp11.cpp:SEXP dust_gpu_sirs_simulate(SEXP ptr, cpp11::sexp time_end);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_simulate(SEXP ptr, SEXP time_end) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_simulate(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(time_end)));
src/cpp11.cpp:SEXP dust_gpu_sirs_run_adjoint(SEXP ptr);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_run_adjoint(SEXP ptr) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_run_adjoint(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
src/cpp11.cpp:SEXP dust_gpu_sirs_set_index(SEXP ptr, cpp11::sexp r_index);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_set_index(SEXP ptr, SEXP r_index) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_set_index(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index)));
src/cpp11.cpp:SEXP dust_gpu_sirs_update_state(SEXP ptr, SEXP r_pars, SEXP r_state, SEXP r_time, SEXP r_set_initial_state, SEXP index, SEXP reset_step_size);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_update_state(SEXP ptr, SEXP r_pars, SEXP r_state, SEXP r_time, SEXP r_set_initial_state, SEXP index, SEXP reset_step_size) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_update_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_pars), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_state), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_time), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_set_initial_state), cpp11::as_cpp<cpp11::decay_t<SEXP>>(index), cpp11::as_cpp<cpp11::decay_t<SEXP>>(reset_step_size)));
src/cpp11.cpp:SEXP dust_gpu_sirs_state(SEXP ptr, SEXP r_index);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_state(SEXP ptr, SEXP r_index) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_index)));
src/cpp11.cpp:SEXP dust_gpu_sirs_time(SEXP ptr);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_time(SEXP ptr) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_time(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
src/cpp11.cpp:void dust_gpu_sirs_reorder(SEXP ptr, cpp11::sexp r_index);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_reorder(SEXP ptr, SEXP r_index) {
src/cpp11.cpp:    dust_gpu_sirs_reorder(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index));
src/cpp11.cpp:SEXP dust_gpu_sirs_resample(SEXP ptr, cpp11::doubles r_weights);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_resample(SEXP ptr, SEXP r_weights) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_resample(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(r_weights)));
src/cpp11.cpp:SEXP dust_gpu_sirs_rng_state(SEXP ptr, bool first_only, bool last_only);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_rng_state(SEXP ptr, SEXP first_only, SEXP last_only) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_rng_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<bool>>(first_only), cpp11::as_cpp<cpp11::decay_t<bool>>(last_only)));
src/cpp11.cpp:SEXP dust_gpu_sirs_set_rng_state(SEXP ptr, cpp11::raws rng_state);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_set_rng_state(SEXP ptr, SEXP rng_state) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_set_rng_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::raws>>(rng_state)));
src/cpp11.cpp:SEXP dust_gpu_sirs_set_data(SEXP ptr, cpp11::list data, bool shared);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_set_data(SEXP ptr, SEXP data, SEXP shared) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_set_data(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(data), cpp11::as_cpp<cpp11::decay_t<bool>>(shared)));
src/cpp11.cpp:SEXP dust_gpu_sirs_compare_data(SEXP ptr);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_compare_data(SEXP ptr) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_compare_data(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
src/cpp11.cpp:SEXP dust_gpu_sirs_filter(SEXP ptr, SEXP time_end, bool save_trajectories, cpp11::sexp time_snapshot, cpp11::sexp min_log_likelihood);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_filter(SEXP ptr, SEXP time_end, SEXP save_trajectories, SEXP time_snapshot, SEXP min_log_likelihood) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_filter(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(time_end), cpp11::as_cpp<cpp11::decay_t<bool>>(save_trajectories), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(time_snapshot), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(min_log_likelihood)));
src/cpp11.cpp:void dust_gpu_sirs_set_n_threads(SEXP ptr, int n_threads);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_set_n_threads(SEXP ptr, SEXP n_threads) {
src/cpp11.cpp:    dust_gpu_sirs_set_n_threads(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n_threads));
src/cpp11.cpp:int dust_gpu_sirs_n_state(SEXP ptr);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_n_state(SEXP ptr) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_n_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
src/cpp11.cpp:void dust_gpu_sirs_set_stochastic_schedule(SEXP ptr, SEXP time);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_set_stochastic_schedule(SEXP ptr, SEXP time) {
src/cpp11.cpp:    dust_gpu_sirs_set_stochastic_schedule(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(time));
src/cpp11.cpp:SEXP dust_gpu_sirs_ode_statistics(SEXP ptr);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_sirs_ode_statistics(SEXP ptr) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_sirs_ode_statistics(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
src/cpp11.cpp:// test_cuda_launch_control.cpp
src/cpp11.cpp:SEXP test_cuda_pars(cpp11::sexp r_gpu_config, int n_particles, int n_particles_each, int n_state, int n_state_full, int n_shared_int, int n_shared_real, int data_size, int shared_size);
src/cpp11.cpp:extern "C" SEXP _dust_test_cuda_pars(SEXP r_gpu_config, SEXP n_particles, SEXP n_particles_each, SEXP n_state, SEXP n_state_full, SEXP n_shared_int, SEXP n_shared_real, SEXP data_size, SEXP shared_size) {
src/cpp11.cpp:    return cpp11::as_sexp(test_cuda_pars(cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_gpu_config), cpp11::as_cpp<cpp11::decay_t<int>>(n_particles), cpp11::as_cpp<cpp11::decay_t<int>>(n_particles_each), cpp11::as_cpp<cpp11::decay_t<int>>(n_state), cpp11::as_cpp<cpp11::decay_t<int>>(n_state_full), cpp11::as_cpp<cpp11::decay_t<int>>(n_shared_int), cpp11::as_cpp<cpp11::decay_t<int>>(n_shared_real), cpp11::as_cpp<cpp11::decay_t<int>>(data_size), cpp11::as_cpp<cpp11::decay_t<int>>(shared_size)));
src/cpp11.cpp:cpp11::sexp dust_variable_gpu_info();
src/cpp11.cpp:extern "C" SEXP _dust_dust_variable_gpu_info() {
src/cpp11.cpp:    return cpp11::as_sexp(dust_variable_gpu_info());
src/cpp11.cpp:SEXP dust_cpu_variable_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time, cpp11::sexp r_n_particles, int n_threads, cpp11::sexp r_seed, bool deterministic, cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/cpp11.cpp:extern "C" SEXP _dust_dust_cpu_variable_alloc(SEXP r_pars, SEXP pars_multi, SEXP r_time, SEXP r_n_particles, SEXP n_threads, SEXP r_seed, SEXP deterministic, SEXP gpu_config, SEXP ode_control) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_cpu_variable_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_pars), cpp11::as_cpp<cpp11::decay_t<bool>>(pars_multi), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_time), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_n_particles), cpp11::as_cpp<cpp11::decay_t<int>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed), cpp11::as_cpp<cpp11::decay_t<bool>>(deterministic), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(gpu_config), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(ode_control)));
src/cpp11.cpp:SEXP dust_gpu_variable_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time, cpp11::sexp r_n_particles, int n_threads, cpp11::sexp r_seed, bool deterministic, cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_alloc(SEXP r_pars, SEXP pars_multi, SEXP r_time, SEXP r_n_particles, SEXP n_threads, SEXP r_seed, SEXP deterministic, SEXP gpu_config, SEXP ode_control) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_pars), cpp11::as_cpp<cpp11::decay_t<bool>>(pars_multi), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_time), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_n_particles), cpp11::as_cpp<cpp11::decay_t<int>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed), cpp11::as_cpp<cpp11::decay_t<bool>>(deterministic), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(gpu_config), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(ode_control)));
src/cpp11.cpp:cpp11::sexp dust_gpu_variable_capabilities();
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_capabilities() {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_capabilities());
src/cpp11.cpp:SEXP dust_gpu_variable_run(SEXP ptr, cpp11::sexp r_time_end);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_run(SEXP ptr, SEXP r_time_end) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_run(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_time_end)));
src/cpp11.cpp:SEXP dust_gpu_variable_simulate(SEXP ptr, cpp11::sexp time_end);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_simulate(SEXP ptr, SEXP time_end) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_simulate(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(time_end)));
src/cpp11.cpp:SEXP dust_gpu_variable_run_adjoint(SEXP ptr);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_run_adjoint(SEXP ptr) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_run_adjoint(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
src/cpp11.cpp:SEXP dust_gpu_variable_set_index(SEXP ptr, cpp11::sexp r_index);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_set_index(SEXP ptr, SEXP r_index) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_set_index(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index)));
src/cpp11.cpp:SEXP dust_gpu_variable_update_state(SEXP ptr, SEXP r_pars, SEXP r_state, SEXP r_time, SEXP r_set_initial_state, SEXP index, SEXP reset_step_size);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_update_state(SEXP ptr, SEXP r_pars, SEXP r_state, SEXP r_time, SEXP r_set_initial_state, SEXP index, SEXP reset_step_size) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_update_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_pars), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_state), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_time), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_set_initial_state), cpp11::as_cpp<cpp11::decay_t<SEXP>>(index), cpp11::as_cpp<cpp11::decay_t<SEXP>>(reset_step_size)));
src/cpp11.cpp:SEXP dust_gpu_variable_state(SEXP ptr, SEXP r_index);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_state(SEXP ptr, SEXP r_index) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_index)));
src/cpp11.cpp:SEXP dust_gpu_variable_time(SEXP ptr);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_time(SEXP ptr) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_time(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
src/cpp11.cpp:void dust_gpu_variable_reorder(SEXP ptr, cpp11::sexp r_index);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_reorder(SEXP ptr, SEXP r_index) {
src/cpp11.cpp:    dust_gpu_variable_reorder(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index));
src/cpp11.cpp:SEXP dust_gpu_variable_resample(SEXP ptr, cpp11::doubles r_weights);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_resample(SEXP ptr, SEXP r_weights) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_resample(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(r_weights)));
src/cpp11.cpp:SEXP dust_gpu_variable_rng_state(SEXP ptr, bool first_only, bool last_only);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_rng_state(SEXP ptr, SEXP first_only, SEXP last_only) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_rng_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<bool>>(first_only), cpp11::as_cpp<cpp11::decay_t<bool>>(last_only)));
src/cpp11.cpp:SEXP dust_gpu_variable_set_rng_state(SEXP ptr, cpp11::raws rng_state);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_set_rng_state(SEXP ptr, SEXP rng_state) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_set_rng_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::raws>>(rng_state)));
src/cpp11.cpp:SEXP dust_gpu_variable_set_data(SEXP ptr, cpp11::list data, bool shared);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_set_data(SEXP ptr, SEXP data, SEXP shared) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_set_data(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(data), cpp11::as_cpp<cpp11::decay_t<bool>>(shared)));
src/cpp11.cpp:SEXP dust_gpu_variable_compare_data(SEXP ptr);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_compare_data(SEXP ptr) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_compare_data(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
src/cpp11.cpp:SEXP dust_gpu_variable_filter(SEXP ptr, SEXP time_end, bool save_trajectories, cpp11::sexp time_snapshot, cpp11::sexp min_log_likelihood);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_filter(SEXP ptr, SEXP time_end, SEXP save_trajectories, SEXP time_snapshot, SEXP min_log_likelihood) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_filter(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(time_end), cpp11::as_cpp<cpp11::decay_t<bool>>(save_trajectories), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(time_snapshot), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(min_log_likelihood)));
src/cpp11.cpp:void dust_gpu_variable_set_n_threads(SEXP ptr, int n_threads);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_set_n_threads(SEXP ptr, SEXP n_threads) {
src/cpp11.cpp:    dust_gpu_variable_set_n_threads(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n_threads));
src/cpp11.cpp:int dust_gpu_variable_n_state(SEXP ptr);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_n_state(SEXP ptr) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_n_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
src/cpp11.cpp:void dust_gpu_variable_set_stochastic_schedule(SEXP ptr, SEXP time);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_set_stochastic_schedule(SEXP ptr, SEXP time) {
src/cpp11.cpp:    dust_gpu_variable_set_stochastic_schedule(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(time));
src/cpp11.cpp:SEXP dust_gpu_variable_ode_statistics(SEXP ptr);
src/cpp11.cpp:extern "C" SEXP _dust_dust_gpu_variable_ode_statistics(SEXP ptr) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_gpu_variable_ode_statistics(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
src/cpp11.cpp:cpp11::sexp dust_volatility_gpu_info();
src/cpp11.cpp:extern "C" SEXP _dust_dust_volatility_gpu_info() {
src/cpp11.cpp:    return cpp11::as_sexp(dust_volatility_gpu_info());
src/cpp11.cpp:SEXP dust_cpu_volatility_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time, cpp11::sexp r_n_particles, int n_threads, cpp11::sexp r_seed, bool deterministic, cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/cpp11.cpp:extern "C" SEXP _dust_dust_cpu_volatility_alloc(SEXP r_pars, SEXP pars_multi, SEXP r_time, SEXP r_n_particles, SEXP n_threads, SEXP r_seed, SEXP deterministic, SEXP gpu_config, SEXP ode_control) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_cpu_volatility_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_pars), cpp11::as_cpp<cpp11::decay_t<bool>>(pars_multi), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_time), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_n_particles), cpp11::as_cpp<cpp11::decay_t<int>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed), cpp11::as_cpp<cpp11::decay_t<bool>>(deterministic), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(gpu_config), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(ode_control)));
src/cpp11.cpp:cpp11::sexp dust_walk_gpu_info();
src/cpp11.cpp:extern "C" SEXP _dust_dust_walk_gpu_info() {
src/cpp11.cpp:    return cpp11::as_sexp(dust_walk_gpu_info());
src/cpp11.cpp:SEXP dust_cpu_walk_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time, cpp11::sexp r_n_particles, int n_threads, cpp11::sexp r_seed, bool deterministic, cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/cpp11.cpp:extern "C" SEXP _dust_dust_cpu_walk_alloc(SEXP r_pars, SEXP pars_multi, SEXP r_time, SEXP r_n_particles, SEXP n_threads, SEXP r_seed, SEXP deterministic, SEXP gpu_config, SEXP ode_control) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_cpu_walk_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_pars), cpp11::as_cpp<cpp11::decay_t<bool>>(pars_multi), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_time), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_n_particles), cpp11::as_cpp<cpp11::decay_t<int>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed), cpp11::as_cpp<cpp11::decay_t<bool>>(deterministic), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(gpu_config), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(ode_control)));
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_alloc",                         (DL_FUNC) &_dust_dust_gpu_sirs_alloc,                         9},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_capabilities",                  (DL_FUNC) &_dust_dust_gpu_sirs_capabilities,                  0},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_compare_data",                  (DL_FUNC) &_dust_dust_gpu_sirs_compare_data,                  1},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_filter",                        (DL_FUNC) &_dust_dust_gpu_sirs_filter,                        5},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_n_state",                       (DL_FUNC) &_dust_dust_gpu_sirs_n_state,                       1},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_ode_statistics",                (DL_FUNC) &_dust_dust_gpu_sirs_ode_statistics,                1},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_reorder",                       (DL_FUNC) &_dust_dust_gpu_sirs_reorder,                       2},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_resample",                      (DL_FUNC) &_dust_dust_gpu_sirs_resample,                      2},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_rng_state",                     (DL_FUNC) &_dust_dust_gpu_sirs_rng_state,                     3},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_run",                           (DL_FUNC) &_dust_dust_gpu_sirs_run,                           2},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_run_adjoint",                   (DL_FUNC) &_dust_dust_gpu_sirs_run_adjoint,                   1},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_set_data",                      (DL_FUNC) &_dust_dust_gpu_sirs_set_data,                      3},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_set_index",                     (DL_FUNC) &_dust_dust_gpu_sirs_set_index,                     2},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_set_n_threads",                 (DL_FUNC) &_dust_dust_gpu_sirs_set_n_threads,                 2},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_set_rng_state",                 (DL_FUNC) &_dust_dust_gpu_sirs_set_rng_state,                 2},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_set_stochastic_schedule",       (DL_FUNC) &_dust_dust_gpu_sirs_set_stochastic_schedule,       2},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_simulate",                      (DL_FUNC) &_dust_dust_gpu_sirs_simulate,                      2},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_state",                         (DL_FUNC) &_dust_dust_gpu_sirs_state,                         2},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_time",                          (DL_FUNC) &_dust_dust_gpu_sirs_time,                          1},
src/cpp11.cpp:    {"_dust_dust_gpu_sirs_update_state",                  (DL_FUNC) &_dust_dust_gpu_sirs_update_state,                  7},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_alloc",                     (DL_FUNC) &_dust_dust_gpu_variable_alloc,                     9},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_capabilities",              (DL_FUNC) &_dust_dust_gpu_variable_capabilities,              0},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_compare_data",              (DL_FUNC) &_dust_dust_gpu_variable_compare_data,              1},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_filter",                    (DL_FUNC) &_dust_dust_gpu_variable_filter,                    5},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_n_state",                   (DL_FUNC) &_dust_dust_gpu_variable_n_state,                   1},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_ode_statistics",            (DL_FUNC) &_dust_dust_gpu_variable_ode_statistics,            1},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_reorder",                   (DL_FUNC) &_dust_dust_gpu_variable_reorder,                   2},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_resample",                  (DL_FUNC) &_dust_dust_gpu_variable_resample,                  2},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_rng_state",                 (DL_FUNC) &_dust_dust_gpu_variable_rng_state,                 3},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_run",                       (DL_FUNC) &_dust_dust_gpu_variable_run,                       2},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_run_adjoint",               (DL_FUNC) &_dust_dust_gpu_variable_run_adjoint,               1},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_set_data",                  (DL_FUNC) &_dust_dust_gpu_variable_set_data,                  3},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_set_index",                 (DL_FUNC) &_dust_dust_gpu_variable_set_index,                 2},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_set_n_threads",             (DL_FUNC) &_dust_dust_gpu_variable_set_n_threads,             2},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_set_rng_state",             (DL_FUNC) &_dust_dust_gpu_variable_set_rng_state,             2},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_set_stochastic_schedule",   (DL_FUNC) &_dust_dust_gpu_variable_set_stochastic_schedule,   2},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_simulate",                  (DL_FUNC) &_dust_dust_gpu_variable_simulate,                  2},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_state",                     (DL_FUNC) &_dust_dust_gpu_variable_state,                     2},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_time",                      (DL_FUNC) &_dust_dust_gpu_variable_time,                      1},
src/cpp11.cpp:    {"_dust_dust_gpu_variable_update_state",              (DL_FUNC) &_dust_dust_gpu_variable_update_state,              7},
src/cpp11.cpp:    {"_dust_dust_logistic_gpu_info",                      (DL_FUNC) &_dust_dust_logistic_gpu_info,                      0},
src/cpp11.cpp:    {"_dust_dust_sir_gpu_info",                           (DL_FUNC) &_dust_dust_sir_gpu_info,                           0},
src/cpp11.cpp:    {"_dust_dust_sirs_gpu_info",                          (DL_FUNC) &_dust_dust_sirs_gpu_info,                          0},
src/cpp11.cpp:    {"_dust_dust_variable_gpu_info",                      (DL_FUNC) &_dust_dust_variable_gpu_info,                      0},
src/cpp11.cpp:    {"_dust_dust_volatility_gpu_info",                    (DL_FUNC) &_dust_dust_volatility_gpu_info,                    0},
src/cpp11.cpp:    {"_dust_dust_walk_gpu_info",                          (DL_FUNC) &_dust_dust_walk_gpu_info,                          0},
src/cpp11.cpp:    {"_dust_test_cuda_pars",                              (DL_FUNC) &_dust_test_cuda_pars,                              9},
src/volatility.cpp:cpp11::sexp dust_volatility_gpu_info();
src/volatility.cpp:                         cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/volatility.cpp:cpp11::sexp dust_volatility_gpu_info() {
src/volatility.cpp:  return dust::gpu::r::gpu_info();
src/volatility.cpp:                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
src/volatility.cpp:                                        gpu_config, ode_control);
src/walk.cpp:cpp11::sexp dust_walk_gpu_info();
src/walk.cpp:                         cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/walk.cpp:cpp11::sexp dust_walk_gpu_info() {
src/walk.cpp:  return dust::gpu::r::gpu_info();
src/walk.cpp:                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
src/walk.cpp:                                        gpu_config, ode_control);
src/test_cuda_launch_control.cpp:#include <dust/gpu/launch_control.hpp>
src/test_cuda_launch_control.cpp:#include <dust/r/gpu.hpp>
src/test_cuda_launch_control.cpp:cpp11::list launch_r_list(const dust::gpu::launch_control& p) {
src/test_cuda_launch_control.cpp:SEXP test_cuda_pars(cpp11::sexp r_gpu_config, int n_particles,
src/test_cuda_launch_control.cpp:  dust::gpu::gpu_config config =
src/test_cuda_launch_control.cpp:    dust::gpu::r::gpu_config(r_gpu_config);
src/test_cuda_launch_control.cpp:  auto pars = dust::gpu::launch_control_dust(config,
src/sirs.cpp:cpp11::sexp dust_sirs_gpu_info();
src/sirs.cpp:                         cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/sirs.cpp:SEXP dust_gpu_sirs_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
src/sirs.cpp:                         cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/sirs.cpp:cpp11::sexp dust_gpu_sirs_capabilities();
src/sirs.cpp:SEXP dust_gpu_sirs_run(SEXP ptr, cpp11::sexp r_time_end);
src/sirs.cpp:SEXP dust_gpu_sirs_simulate(SEXP ptr, cpp11::sexp time_end);
src/sirs.cpp:SEXP dust_gpu_sirs_run_adjoint(SEXP ptr);
src/sirs.cpp:SEXP dust_gpu_sirs_set_index(SEXP ptr, cpp11::sexp r_index);
src/sirs.cpp:SEXP dust_gpu_sirs_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
src/sirs.cpp:SEXP dust_gpu_sirs_state(SEXP ptr, SEXP r_index);
src/sirs.cpp:SEXP dust_gpu_sirs_time(SEXP ptr);
src/sirs.cpp:void dust_gpu_sirs_reorder(SEXP ptr, cpp11::sexp r_index);
src/sirs.cpp:SEXP dust_gpu_sirs_resample(SEXP ptr, cpp11::doubles r_weights);
src/sirs.cpp:SEXP dust_gpu_sirs_rng_state(SEXP ptr, bool first_only, bool last_only);
src/sirs.cpp:SEXP dust_gpu_sirs_set_rng_state(SEXP ptr, cpp11::raws rng_state);
src/sirs.cpp:SEXP dust_gpu_sirs_set_data(SEXP ptr, cpp11::list data, bool shared);
src/sirs.cpp:SEXP dust_gpu_sirs_compare_data(SEXP ptr);
src/sirs.cpp:SEXP dust_gpu_sirs_filter(SEXP ptr, SEXP time_end,
src/sirs.cpp:void dust_gpu_sirs_set_n_threads(SEXP ptr, int n_threads);
src/sirs.cpp:int dust_gpu_sirs_n_state(SEXP ptr);
src/sirs.cpp:void dust_gpu_sirs_set_stochastic_schedule(SEXP ptr, SEXP time);
src/sirs.cpp:SEXP dust_gpu_sirs_ode_statistics(SEXP ptr);
src/sirs.cpp:namespace gpu {
src/sirs.cpp:  using dust::gpu::shared_copy_data;
src/sirs.cpp:void update_gpu<sirs>(size_t time,
src/sirs.cpp:                      const dust::gpu::interleaved<sirs::real_type> state,
src/sirs.cpp:                      dust::gpu::interleaved<int> internal_int,
src/sirs.cpp:                      dust::gpu::interleaved<sirs::real_type> internal_real,
src/sirs.cpp:                      dust::gpu::interleaved<sirs::real_type> state_next) {
src/sirs.cpp:sirs::real_type compare_gpu<sirs>(const dust::gpu::interleaved<sirs::real_type> state,
src/sirs.cpp:                                  dust::gpu::interleaved<int> internal_int,
src/sirs.cpp:                                  dust::gpu::interleaved<sirs::real_type> internal_real,
src/sirs.cpp:cpp11::sexp dust_sirs_gpu_info() {
src/sirs.cpp:  return dust::gpu::r::gpu_info();
src/sirs.cpp:                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
src/sirs.cpp:                                        gpu_config, ode_control);
src/sirs.cpp:using model_gpu = dust::dust_gpu<sirs>;
src/sirs.cpp:cpp11::sexp dust_gpu_sirs_capabilities() {
src/sirs.cpp:  return dust::r::dust_capabilities<model_gpu>();
src/sirs.cpp:SEXP dust_gpu_sirs_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
src/sirs.cpp:                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
src/sirs.cpp:  return dust::r::dust_gpu_alloc<sirs>(r_pars, pars_multi, r_time, r_n_particles,
src/sirs.cpp:                                        gpu_config, ode_control);
src/sirs.cpp:SEXP dust_gpu_sirs_run(SEXP ptr, cpp11::sexp r_time_end) {
src/sirs.cpp:  return dust::r::dust_run<model_gpu>(ptr, r_time_end);
src/sirs.cpp:SEXP dust_gpu_sirs_simulate(SEXP ptr, cpp11::sexp r_time_end) {
src/sirs.cpp:  return dust::r::dust_simulate<model_gpu>(ptr, r_time_end);
src/sirs.cpp:SEXP dust_gpu_sirs_run_adjoint(SEXP ptr) {
src/sirs.cpp:  return dust::r::dust_run_adjoint<model_gpu>(ptr);
src/sirs.cpp:SEXP dust_gpu_sirs_set_index(SEXP ptr, cpp11::sexp r_index) {
src/sirs.cpp:  dust::r::dust_set_index<model_gpu>(ptr, r_index);
src/sirs.cpp:SEXP dust_gpu_sirs_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
src/sirs.cpp:  return dust::r::dust_update_state<model_gpu>(ptr, r_pars, r_state, r_time,
src/sirs.cpp:SEXP dust_gpu_sirs_state(SEXP ptr, SEXP r_index) {
src/sirs.cpp:  return dust::r::dust_state<model_gpu>(ptr, r_index);
src/sirs.cpp:SEXP dust_gpu_sirs_time(SEXP ptr) {
src/sirs.cpp:  return dust::r::dust_time<model_gpu>(ptr);
src/sirs.cpp:void dust_gpu_sirs_reorder(SEXP ptr, cpp11::sexp r_index) {
src/sirs.cpp:  return dust::r::dust_reorder<model_gpu>(ptr, r_index);
src/sirs.cpp:SEXP dust_gpu_sirs_resample(SEXP ptr, cpp11::doubles r_weights) {
src/sirs.cpp:  return dust::r::dust_resample<model_gpu>(ptr, r_weights);
src/sirs.cpp:SEXP dust_gpu_sirs_rng_state(SEXP ptr, bool first_only, bool last_only) {
src/sirs.cpp:  return dust::r::dust_rng_state<model_gpu>(ptr, first_only, last_only);
src/sirs.cpp:SEXP dust_gpu_sirs_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
src/sirs.cpp:  dust::r::dust_set_rng_state<model_gpu>(ptr, rng_state);
src/sirs.cpp:SEXP dust_gpu_sirs_set_data(SEXP ptr, cpp11::list data,
src/sirs.cpp:  dust::r::dust_set_data<model_gpu>(ptr, data, shared);
src/sirs.cpp:SEXP dust_gpu_sirs_compare_data(SEXP ptr) {
src/sirs.cpp:  return dust::r::dust_compare_data<model_gpu>(ptr);
src/sirs.cpp:SEXP dust_gpu_sirs_filter(SEXP ptr, SEXP time_end,
src/sirs.cpp:  return dust::r::dust_filter<model_gpu>(ptr, time_end,
src/sirs.cpp:void dust_gpu_sirs_set_n_threads(SEXP ptr, int n_threads) {
src/sirs.cpp:  return dust::r::dust_set_n_threads<model_gpu>(ptr, n_threads);
src/sirs.cpp:int dust_gpu_sirs_n_state(SEXP ptr) {
src/sirs.cpp:  return dust::r::dust_n_state<model_gpu>(ptr);
src/sirs.cpp:void dust_gpu_sirs_set_stochastic_schedule(SEXP ptr, SEXP time) {
src/sirs.cpp:  dust::r::dust_set_stochastic_schedule<model_gpu>(ptr, time);
src/sirs.cpp:SEXP dust_gpu_sirs_ode_statistics(SEXP ptr) {
src/sirs.cpp:  return dust::r::dust_ode_statistics<model_gpu>(ptr);
src/logistic.cpp:cpp11::sexp dust_logistic_gpu_info();
src/logistic.cpp:                         cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/logistic.cpp:cpp11::sexp dust_logistic_gpu_info() {
src/logistic.cpp:  return dust::gpu::r::gpu_info();
src/logistic.cpp:                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
src/logistic.cpp:                                        gpu_config, ode_control);
src/sir.cpp:cpp11::sexp dust_sir_gpu_info();
src/sir.cpp:                         cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/sir.cpp:cpp11::sexp dust_sir_gpu_info() {
src/sir.cpp:  return dust::gpu::r::gpu_info();
src/sir.cpp:                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
src/sir.cpp:                                        gpu_config, ode_control);
src/variable.cpp:cpp11::sexp dust_variable_gpu_info();
src/variable.cpp:                         cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/variable.cpp:SEXP dust_gpu_variable_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
src/variable.cpp:                         cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/variable.cpp:cpp11::sexp dust_gpu_variable_capabilities();
src/variable.cpp:SEXP dust_gpu_variable_run(SEXP ptr, cpp11::sexp r_time_end);
src/variable.cpp:SEXP dust_gpu_variable_simulate(SEXP ptr, cpp11::sexp time_end);
src/variable.cpp:SEXP dust_gpu_variable_run_adjoint(SEXP ptr);
src/variable.cpp:SEXP dust_gpu_variable_set_index(SEXP ptr, cpp11::sexp r_index);
src/variable.cpp:SEXP dust_gpu_variable_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
src/variable.cpp:SEXP dust_gpu_variable_state(SEXP ptr, SEXP r_index);
src/variable.cpp:SEXP dust_gpu_variable_time(SEXP ptr);
src/variable.cpp:void dust_gpu_variable_reorder(SEXP ptr, cpp11::sexp r_index);
src/variable.cpp:SEXP dust_gpu_variable_resample(SEXP ptr, cpp11::doubles r_weights);
src/variable.cpp:SEXP dust_gpu_variable_rng_state(SEXP ptr, bool first_only, bool last_only);
src/variable.cpp:SEXP dust_gpu_variable_set_rng_state(SEXP ptr, cpp11::raws rng_state);
src/variable.cpp:SEXP dust_gpu_variable_set_data(SEXP ptr, cpp11::list data, bool shared);
src/variable.cpp:SEXP dust_gpu_variable_compare_data(SEXP ptr);
src/variable.cpp:SEXP dust_gpu_variable_filter(SEXP ptr, SEXP time_end,
src/variable.cpp:void dust_gpu_variable_set_n_threads(SEXP ptr, int n_threads);
src/variable.cpp:int dust_gpu_variable_n_state(SEXP ptr);
src/variable.cpp:void dust_gpu_variable_set_stochastic_schedule(SEXP ptr, SEXP time);
src/variable.cpp:SEXP dust_gpu_variable_ode_statistics(SEXP ptr);
src/variable.cpp:namespace gpu {
src/variable.cpp:  using dust::gpu::shared_copy_data;
src/variable.cpp:void update_gpu<variable>(size_t time,
src/variable.cpp:                          const dust::gpu::interleaved<variable::real_type> state,
src/variable.cpp:                          dust::gpu::interleaved<int> internal_int,
src/variable.cpp:                          dust::gpu::interleaved<variable::real_type> internal_real,
src/variable.cpp:                          dust::gpu::interleaved<variable::real_type> state_next) {
src/variable.cpp:cpp11::sexp dust_variable_gpu_info() {
src/variable.cpp:  return dust::gpu::r::gpu_info();
src/variable.cpp:                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
src/variable.cpp:                                        gpu_config, ode_control);
src/variable.cpp:using model_gpu = dust::dust_gpu<variable>;
src/variable.cpp:cpp11::sexp dust_gpu_variable_capabilities() {
src/variable.cpp:  return dust::r::dust_capabilities<model_gpu>();
src/variable.cpp:SEXP dust_gpu_variable_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
src/variable.cpp:                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
src/variable.cpp:  return dust::r::dust_gpu_alloc<variable>(r_pars, pars_multi, r_time, r_n_particles,
src/variable.cpp:                                        gpu_config, ode_control);
src/variable.cpp:SEXP dust_gpu_variable_run(SEXP ptr, cpp11::sexp r_time_end) {
src/variable.cpp:  return dust::r::dust_run<model_gpu>(ptr, r_time_end);
src/variable.cpp:SEXP dust_gpu_variable_simulate(SEXP ptr, cpp11::sexp r_time_end) {
src/variable.cpp:  return dust::r::dust_simulate<model_gpu>(ptr, r_time_end);
src/variable.cpp:SEXP dust_gpu_variable_run_adjoint(SEXP ptr) {
src/variable.cpp:  return dust::r::dust_run_adjoint<model_gpu>(ptr);
src/variable.cpp:SEXP dust_gpu_variable_set_index(SEXP ptr, cpp11::sexp r_index) {
src/variable.cpp:  dust::r::dust_set_index<model_gpu>(ptr, r_index);
src/variable.cpp:SEXP dust_gpu_variable_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
src/variable.cpp:  return dust::r::dust_update_state<model_gpu>(ptr, r_pars, r_state, r_time,
src/variable.cpp:SEXP dust_gpu_variable_state(SEXP ptr, SEXP r_index) {
src/variable.cpp:  return dust::r::dust_state<model_gpu>(ptr, r_index);
src/variable.cpp:SEXP dust_gpu_variable_time(SEXP ptr) {
src/variable.cpp:  return dust::r::dust_time<model_gpu>(ptr);
src/variable.cpp:void dust_gpu_variable_reorder(SEXP ptr, cpp11::sexp r_index) {
src/variable.cpp:  return dust::r::dust_reorder<model_gpu>(ptr, r_index);
src/variable.cpp:SEXP dust_gpu_variable_resample(SEXP ptr, cpp11::doubles r_weights) {
src/variable.cpp:  return dust::r::dust_resample<model_gpu>(ptr, r_weights);
src/variable.cpp:SEXP dust_gpu_variable_rng_state(SEXP ptr, bool first_only, bool last_only) {
src/variable.cpp:  return dust::r::dust_rng_state<model_gpu>(ptr, first_only, last_only);
src/variable.cpp:SEXP dust_gpu_variable_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
src/variable.cpp:  dust::r::dust_set_rng_state<model_gpu>(ptr, rng_state);
src/variable.cpp:SEXP dust_gpu_variable_set_data(SEXP ptr, cpp11::list data,
src/variable.cpp:  dust::r::dust_set_data<model_gpu>(ptr, data, shared);
src/variable.cpp:SEXP dust_gpu_variable_compare_data(SEXP ptr) {
src/variable.cpp:  return dust::r::dust_compare_data<model_gpu>(ptr);
src/variable.cpp:SEXP dust_gpu_variable_filter(SEXP ptr, SEXP time_end,
src/variable.cpp:  return dust::r::dust_filter<model_gpu>(ptr, time_end,
src/variable.cpp:void dust_gpu_variable_set_n_threads(SEXP ptr, int n_threads) {
src/variable.cpp:  return dust::r::dust_set_n_threads<model_gpu>(ptr, n_threads);
src/variable.cpp:int dust_gpu_variable_n_state(SEXP ptr) {
src/variable.cpp:  return dust::r::dust_n_state<model_gpu>(ptr);
src/variable.cpp:void dust_gpu_variable_set_stochastic_schedule(SEXP ptr, SEXP time) {
src/variable.cpp:  dust::r::dust_set_stochastic_schedule<model_gpu>(ptr, time);
src/variable.cpp:SEXP dust_gpu_variable_ode_statistics(SEXP ptr) {
src/variable.cpp:  return dust::r::dust_ode_statistics<model_gpu>(ptr);
developing.md:* The gpu vignette is built offline because it requires access to a CUDA toolchain and compatible device.  The script `./scripts/build_gpu_vignette` will update the version in the package
developing.md:* Vignettes form the backbone of the documentation. Mostof these are directly built as normal (in `vignettes/`) but a few are precomputed (see [`vignettes_src/`](vignette_src)) where they need special resources such as access to a GPU or more cores than usually available.
developing.md:## Debugging cuda
developing.md:R -d cuda-gdb
developing.md:set cuda api_failures stop
developing.md:To find memory errors, compile a model with `gpu = dust::dust_cuda_options(debug = TRUE)` to enable debug symbols, then run with
developing.md:R -d cuda-memcheck
developing.md:You want [the `-warn-double-usage`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options-warn-on-double-precision-use) argument, passed via `-Xptxas`.
developing.md:gpu <- dust::dust_cuda_options(

```
