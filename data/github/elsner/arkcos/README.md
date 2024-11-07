# https://github.com/elsner/arkcos

```console
arkcos_main.hxx:#include "arkcos_gpu.hxx"
arkcos_gpu.cu:#include <helper_cuda.h>
arkcos_gpu.cu:#include "arkcos_gpu.hxx"
arkcos_gpu.cu:__constant__ float                                          kernel_gpu__alpha[8000];
arkcos_gpu.cu:__constant__ double                                         theta_gpu__ring[4100];
arkcos_gpu.cu:void convolve_gpu(cufftReal* map_out__pix,
arkcos_gpu.cu:  cufftReal*                                                map_inout_gpu__pix;
arkcos_gpu.cu:  cufftComplex*                                             fftmap_in_gpu__pix;
arkcos_gpu.cu:  cufftComplex*                                             fftmap_out_gpu__pix;
arkcos_gpu.cu:  cufftHandle                                               forward_plan_kernel_gpu;
arkcos_gpu.cu:  cufftComplex*                                             fftkernel_gpu__pix;
arkcos_gpu.cu:  float*                                                    kernel_gpu__pix;
arkcos_gpu.cu:  checkCudaErrors(cudaFuncSetCacheConfig(kernel2grid_gpu_device,
arkcos_gpu.cu:                                         cudaFuncCachePreferL1));
arkcos_gpu.cu:  checkCudaErrors(cudaFuncSetCacheConfig(rotate_fftrings_device,
arkcos_gpu.cu:                                         cudaFuncCachePreferL1));
arkcos_gpu.cu:  checkCudaErrors(cudaFuncSetCacheConfig(kernel_times_map_gpu_device,
arkcos_gpu.cu:                                         cudaFuncCachePreferL1));
arkcos_gpu.cu:  checkCudaErrors(cudaMalloc((void**) &map_inout_gpu__pix, size_map_inout));
arkcos_gpu.cu:  checkCudaErrors(cudaMalloc((void**) &fftmap_in_gpu__pix, size_fftmap_inout));
arkcos_gpu.cu:  checkCudaErrors(cudaMalloc((void**) &fftmap_out_gpu__pix, size_fftmap_inout));
arkcos_gpu.cu:  checkCudaErrors(cudaMemset(map_inout_gpu__pix, 0, size_map_inout));
arkcos_gpu.cu:  checkCudaErrors(cudaMemset(fftmap_in_gpu__pix, 0, size_fftmap_inout));
arkcos_gpu.cu:  checkCudaErrors(cudaMemset(fftmap_out_gpu__pix, 0, size_fftmap_inout));
arkcos_gpu.cu:  checkCudaErrors(cudaMemcpy(map_inout_gpu__pix, map_in__pix, size_map_inout,
arkcos_gpu.cu:                             cudaMemcpyHostToDevice));
arkcos_gpu.cu:  checkCudaErrors(cudaMalloc((void**) &kernel_gpu__pix, size_kernel));
arkcos_gpu.cu:  checkCudaErrors(cudaMalloc((void**) &fftkernel_gpu__pix, size_fftkernel));
arkcos_gpu.cu:  checkCudaErrors(cudaMemset(kernel_gpu__pix, 0, size_kernel));
arkcos_gpu.cu:  checkCudaErrors(cudaMemset(fftkernel_gpu__pix, 0, size_fftkernel));
arkcos_gpu.cu:  checkCudaErrors(cudaMemcpyToSymbol(kernel_gpu__alpha, kernel__alpha,
arkcos_gpu.cu:  checkCudaErrors(cudaMemcpyToSymbol(theta_gpu__ring, theta__ring, size_ntheta));
arkcos_gpu.cu:  fft_map_gpu(fftmap_in_gpu__pix, map_inout_gpu__pix, nside, nrings,
arkcos_gpu.cu:  checkCudaErrors(cufftPlan1d(&forward_plan_kernel_gpu, npix_ring, CUFFT_R2C,
arkcos_gpu.cu:  convolve_with_kernel_gpu(fftmap_out_gpu__pix, fftmap_in_gpu__pix,
arkcos_gpu.cu:                           fftkernel_gpu__pix, kernel_gpu__pix,
arkcos_gpu.cu:                           forward_plan_kernel_gpu, conversion_factor,
arkcos_gpu.cu:  ifft_map_gpu(map_inout_gpu__pix, fftmap_out_gpu__pix,
arkcos_gpu.cu:  checkCudaErrors(cudaMemcpy(map_out__pix, map_inout_gpu__pix, size_map_inout,
arkcos_gpu.cu:                             cudaMemcpyDeviceToHost));
arkcos_gpu.cu:  checkCudaErrors(cudaFree(map_inout_gpu__pix));
arkcos_gpu.cu:  checkCudaErrors(cudaFree(fftmap_in_gpu__pix));
arkcos_gpu.cu:  checkCudaErrors(cudaFree(fftmap_out_gpu__pix));
arkcos_gpu.cu:  checkCudaErrors(cudaFree(kernel_gpu__pix));
arkcos_gpu.cu:  checkCudaErrors(cudaFree(fftkernel_gpu__pix));
arkcos_gpu.cu:  checkCudaErrors(cufftDestroy(forward_plan_kernel_gpu));
arkcos_gpu.cu:void fft_map_gpu(cufftComplex* fftmap_in_gpu__pix,
arkcos_gpu.cu:                 cufftReal* map_in_gpu__pix, const int nside,
arkcos_gpu.cu:    checkCudaErrors(cufftPlan1d(&forward_plan, npix, CUFFT_R2C, 1));
arkcos_gpu.cu:    checkCudaErrors(cufftExecR2C(forward_plan, &map_in_gpu__pix[npix_ring*i],
arkcos_gpu.cu:                                 &fftmap_in_gpu__pix[nfftpix_ring*i]));
arkcos_gpu.cu:    checkCudaErrors(cufftExecR2C(forward_plan,
arkcos_gpu.cu:                                 &map_in_gpu__pix[npix_ring*(nrings-1-i)],
arkcos_gpu.cu:                                 &fftmap_in_gpu__pix[nfftpix_ring*(nrings-1-i)]));
arkcos_gpu.cu:    checkCudaErrors(cufftDestroy(forward_plan));
arkcos_gpu.cu:    checkCudaErrors(cufftPlan1d(&forward_plan, npix_ring, CUFFT_R2C, 2*nside+1));
arkcos_gpu.cu:    checkCudaErrors(cufftExecR2C(forward_plan,
arkcos_gpu.cu:                                 &map_in_gpu__pix[npix_ring*(nside-1)],
arkcos_gpu.cu:                                 &fftmap_in_gpu__pix[nfftpix_ring*(nside-1)]));
arkcos_gpu.cu:    checkCudaErrors(cufftDestroy(forward_plan));
arkcos_gpu.cu:    checkCudaErrors(cufftPlan1d(&forward_plan, npix_ring, CUFFT_R2C, nside+1));
arkcos_gpu.cu:    checkCudaErrors(cufftExecR2C(forward_plan,
arkcos_gpu.cu:                                 &map_in_gpu__pix[npix_ring*(nside-1)],
arkcos_gpu.cu:                                 &fftmap_in_gpu__pix[nfftpix_ring*(nside-1)]));
arkcos_gpu.cu:    checkCudaErrors(cufftDestroy(forward_plan));
arkcos_gpu.cu:    checkCudaErrors(cufftPlan1d(&forward_plan, npix_ring, CUFFT_R2C, nside));
arkcos_gpu.cu:    checkCudaErrors(cufftExecR2C(forward_plan,
arkcos_gpu.cu:                                 &map_in_gpu__pix[npix_ring*(2*nside)],
arkcos_gpu.cu:                                 &fftmap_in_gpu__pix[nfftpix_ring*(2*nside)]));
arkcos_gpu.cu:    checkCudaErrors(cufftDestroy(forward_plan));
arkcos_gpu.cu:    (fftmap_in_gpu__pix, -1.0f, float (npix_ring), nfftpix_ring,
arkcos_gpu.cu:__global__ void rotate_fftrings_device(cufftComplex* fftmap_gpu__pix,
arkcos_gpu.cu:      fftmap_gpu__pix[nfftpix_ring*ring+ring_pix]
arkcos_gpu.cu:        = cuCmulf(fftmap_gpu__pix[nfftpix_ring*ring+ring_pix], temp);
arkcos_gpu.cu:      fftmap_gpu__pix[nfftpix_ring*(nrings-1-ring)+ring_pix]
arkcos_gpu.cu:        = cuCmulf(fftmap_gpu__pix[nfftpix_ring*(nrings-1-ring)+ring_pix], temp);
arkcos_gpu.cu:    fftmap_gpu__pix[nfftpix_ring*ring+ring_pix]
arkcos_gpu.cu:      = cuCmulf(fftmap_gpu__pix[nfftpix_ring*ring+ring_pix], temp);
arkcos_gpu.cu:void convolve_with_kernel_gpu(cufftComplex* fftmap_out_gpu__pix,
arkcos_gpu.cu:                              cufftComplex* fftmap_in_gpu__pix,
arkcos_gpu.cu:                              cufftComplex* fftkernel_gpu__pix,
arkcos_gpu.cu:                              float* kernel_gpu__pix,
arkcos_gpu.cu:                              cufftHandle forward_plan_kernel_gpu,
arkcos_gpu.cu:    kernel2grid_gpu_device<<<blocks_real, threads_real>>>
arkcos_gpu.cu:      (kernel_gpu__pix, conversion_factor, delta_angle, support_rad,
arkcos_gpu.cu:    checkCudaErrors(cufftExecR2C(forward_plan_kernel_gpu, kernel_gpu__pix,
arkcos_gpu.cu:                                 fftkernel_gpu__pix));
arkcos_gpu.cu:    kernel_times_map_gpu_device<<<blocks_fft, threads_fft>>>
arkcos_gpu.cu:      (fftmap_out_gpu__pix, fftmap_in_gpu__pix, fftkernel_gpu__pix,
arkcos_gpu.cu:__global__ void kernel2grid_gpu_device(float* kernel_gpu__pix,
arkcos_gpu.cu:    theta_ring_index = theta_gpu__ring[ring_index];
arkcos_gpu.cu:    theta_ring_index = PI_d - theta_gpu__ring[nrings-1-ring_index];
arkcos_gpu.cu:  sint1_sqr = sin(0.5*(theta_ring_index + theta_gpu__ring[base_ring]));
arkcos_gpu.cu:  sint2_sqr = sin(0.5*(theta_ring_index - theta_gpu__ring[base_ring]));
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*ring+ring_pix[0]]
arkcos_gpu.cu:    = interpolate_gpu_device(angle_sqr[0], support_rad, conversion_factor);
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*ring+ring_pix[1]]
arkcos_gpu.cu:    = interpolate_gpu_device(angle_sqr[1], support_rad, conversion_factor);
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*ring+ring_pix[2]]
arkcos_gpu.cu:    = interpolate_gpu_device(angle_sqr[2], support_rad, conversion_factor);
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*ring+ring_pix[3]]
arkcos_gpu.cu:    = interpolate_gpu_device(angle_sqr[3], support_rad, conversion_factor);
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*ring+ring_pix[4]]
arkcos_gpu.cu:    = interpolate_gpu_device(angle_sqr[4], support_rad, conversion_factor);
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*ring+ring_pix[5]]
arkcos_gpu.cu:    = interpolate_gpu_device(angle_sqr[5], support_rad, conversion_factor);
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*ring+ring_pix[6]]
arkcos_gpu.cu:    = interpolate_gpu_device(angle_sqr[6], support_rad, conversion_factor);
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*ring+ring_pix[7]]
arkcos_gpu.cu:    = interpolate_gpu_device(angle_sqr[7], support_rad, conversion_factor);
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[0]]
arkcos_gpu.cu:    = kernel_gpu__pix[npix_ring*ring+ring_pix[0]];
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[1]]
arkcos_gpu.cu:    = kernel_gpu__pix[npix_ring*ring+ring_pix[1]];
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[2]]
arkcos_gpu.cu:    = kernel_gpu__pix[npix_ring*ring+ring_pix[2]];
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[3]]
arkcos_gpu.cu:    = kernel_gpu__pix[npix_ring*ring+ring_pix[3]];
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[4]]
arkcos_gpu.cu:    = kernel_gpu__pix[npix_ring*ring+ring_pix[4]];
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[5]]
arkcos_gpu.cu:    = kernel_gpu__pix[npix_ring*ring+ring_pix[5]];
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[6]]
arkcos_gpu.cu:    = kernel_gpu__pix[npix_ring*ring+ring_pix[6]];
arkcos_gpu.cu:  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[7]]
arkcos_gpu.cu:    = kernel_gpu__pix[npix_ring*ring+ring_pix[7]];
arkcos_gpu.cu:    kernel_gpu__pix[npix_ring*ring]
arkcos_gpu.cu:      = interpolate_gpu_device(angle_sqr[0], support_rad, conversion_factor);
arkcos_gpu.cu:__device__  __inline__ float interpolate_gpu_device(double angle,
arkcos_gpu.cu:  return kernel_gpu__alpha[lo]
arkcos_gpu.cu:    + (kernel_gpu__alpha[hi] - kernel_gpu__alpha[lo]) * fractional_part;
arkcos_gpu.cu:__global__ void kernel_times_map_gpu_device(cufftComplex* fftmap_out_gpu__pix,
arkcos_gpu.cu:                                            const cufftComplex* fftmap_in_gpu__pix,
arkcos_gpu.cu:                                            const cufftComplex* fftkernel_gpu__pix,
arkcos_gpu.cu:      temp = cuCaddf(cuCmulf(fftkernel_gpu__pix[nfftpix_ring*(ring+i)+ring_pix],
arkcos_gpu.cu:              fftmap_in_gpu__pix[nfftpix_ring*(ring_index+i)+ring_pix]), temp);
arkcos_gpu.cu:    atomicAdd(&fftmap_out_gpu__pix[nfftpix_ring*base_ring+ring_pix].x, temp.x);
arkcos_gpu.cu:    atomicAdd(&fftmap_out_gpu__pix[nfftpix_ring*base_ring+ring_pix].y, temp.y);
arkcos_gpu.cu:      temp = cuCaddf(cuCmulf(fftkernel_gpu__pix[nfftpix_ring*(ring-i)+ring_pix],
arkcos_gpu.cu:                fftmap_in_gpu__pix[nfftpix_ring*(ring_index+i)+ring_pix]), temp);
arkcos_gpu.cu:    atomicAdd(&fftmap_out_gpu__pix[nfftpix_ring*(nrings-1-base_ring)+ring_pix].x,
arkcos_gpu.cu:    atomicAdd(&fftmap_out_gpu__pix[nfftpix_ring*(nrings-1-base_ring)+ring_pix].y,
arkcos_gpu.cu:void ifft_map_gpu(cufftReal* map_out_gpu__pix,
arkcos_gpu.cu:                  cufftComplex* fftmap_out_gpu__pix,
arkcos_gpu.cu:    (fftmap_out_gpu__pix, 1.0f, float (npix_ring), nfftpix_ring,
arkcos_gpu.cu:    checkCudaErrors(cudaMemset(&fftmap_out_gpu__pix[nfftpix_ring*i+npix].y,
arkcos_gpu.cu:    checkCudaErrors(cudaMemset(&fftmap_out_gpu__pix[nfftpix_ring*(nrings-1-i)+npix].y,
arkcos_gpu.cu:    checkCudaErrors(cufftPlan1d(&backward_plan, npix, CUFFT_C2R, 1));
arkcos_gpu.cu:    checkCudaErrors(cufftExecC2R(backward_plan,
arkcos_gpu.cu:                                 &fftmap_out_gpu__pix[nfftpix_ring*i],
arkcos_gpu.cu:                                 &map_out_gpu__pix[npix_ring*i]));
arkcos_gpu.cu:    checkCudaErrors(cufftExecC2R(backward_plan,
arkcos_gpu.cu:                                 &fftmap_out_gpu__pix[nfftpix_ring*(nrings-1-i)],
arkcos_gpu.cu:                                 &map_out_gpu__pix[npix_ring*(nrings-1-i)]));
arkcos_gpu.cu:    checkCudaErrors(cufftDestroy(backward_plan));
arkcos_gpu.cu:    checkCudaErrors(cufftPlan1d(&backward_plan, npix_ring, CUFFT_C2R, 2*nside+1));
arkcos_gpu.cu:    checkCudaErrors(cufftExecC2R(backward_plan,
arkcos_gpu.cu:                                 &fftmap_out_gpu__pix[nfftpix_ring*(nside-1)],
arkcos_gpu.cu:                                 &map_out_gpu__pix[npix_ring*(nside-1)]));
arkcos_gpu.cu:    checkCudaErrors(cufftDestroy(backward_plan));
arkcos_gpu.cu:    checkCudaErrors(cufftPlan1d(&backward_plan, npix_ring, CUFFT_C2R, nside+1));
arkcos_gpu.cu:    checkCudaErrors(cufftExecC2R(backward_plan,
arkcos_gpu.cu:                                 &fftmap_out_gpu__pix[nfftpix_ring*(nside-1)],
arkcos_gpu.cu:                                 &map_out_gpu__pix[npix_ring*(nside-1)]));
arkcos_gpu.cu:    checkCudaErrors(cufftDestroy(backward_plan));
arkcos_gpu.cu:    checkCudaErrors(cufftPlan1d(&backward_plan, npix_ring, CUFFT_C2R, nside));
arkcos_gpu.cu:    checkCudaErrors(cufftExecC2R(backward_plan,
arkcos_gpu.cu:                                 &fftmap_out_gpu__pix[nfftpix_ring*(2*nside)],
arkcos_gpu.cu:                                 &map_out_gpu__pix[npix_ring*(2*nside)]));
arkcos_gpu.cu:    checkCudaErrors(cufftDestroy(backward_plan));
Makefile:	  -lcudart -lcufft
Makefile:	  arkcos_misc.o arkcos_gpu.o\
arkcos_class.cxx:  do_gpu       = false;
arkcos_class.cxx:  do_gpu       = false;
arkcos_class.cxx:  if (do_gpu) {
arkcos_class.cxx:    cout << " GPU   = true"    << endl;
arkcos_class.cxx:    cout << " GPU   = false"   << endl;
arkcos_class.cxx:void convmap::allocate_fft_map_gpu(convkernel kernel, const parameter par) {
arkcos_class.cxx:  if (par.do_gpu) {
arkcos_class.cxx:    allocate_fft_map_gpu(kernel, par);
arkcos_class.cxx:    convolve_gpu(map_out__pix,
arkcos_main.cxx:// Compute convolution on GPU:
arkcos_main.cxx:  par.do_gpu = true;
arkcos_gpu.hxx:void convolve_gpu(cufftReal*, cufftReal*, const double*, const double,
arkcos_gpu.hxx:void fft_map_gpu(cufftComplex*, cufftReal*, const int,
arkcos_gpu.hxx:void convolve_with_kernel_gpu(cufftComplex*, cufftComplex*, cufftComplex*,
arkcos_gpu.hxx:__global__ void kernel2grid_gpu_device(float*, const double,
arkcos_gpu.hxx:__device__  __inline__ float interpolate_gpu_device(double,
arkcos_gpu.hxx:__global__ void kernel_times_map_gpu_device(cufftComplex*, const cufftComplex*,
arkcos_gpu.hxx:void ifft_map_gpu(cufftReal*, cufftComplex*, const int, const int,
arkcos_class.hxx:  bool                                                      do_gpu;
arkcos_class.hxx:  void allocate_fft_map_gpu(const convkernel, const parameter);
README:       https://arxiv.org/abs/1104.0672. It is written in C++/CUDA and
README:       graphics processing units (GPUs). The latter code sections were
README:       optimized for optimal performance on NVIDIA GeForce GTX 480
README:       using the CUDA toolkit version 3.2 and later ported to comply
README:                      - CUDA
README:       the provided map. If the variable 'par.do_gpu' is set to
README:       'true', the convolution will be computed on the GPU, otherwise

```
