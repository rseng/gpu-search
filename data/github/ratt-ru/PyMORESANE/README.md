# https://github.com/ratt-ru/PyMORESANE

```console
setup.py:      description='CUDA-accelerated implementation of the MORESANE deconvolution algorithm',
setup.py:      requires=['numpy', 'scipy', 'astropy', 'pycuda'],
Dockerfile:# Not adding GPU depencies for now
tests/Makefile:#	Very basic end-to-end test. Should generate three fits image files. These take about 1 minute to generate without GPU.
README.md:This is a Python and pyCUDA-accelerated implementation of the MORESANE
README.md:  * [CUDA](https://developer.nvidia.com/cuda-downloads)
README.md:  * [pycuda](http://mathema.tician.de/software/pycuda/)
README.md:  * [scikits.cuda](http://scikit-cuda.readthedocs.org/)
requirements.txt:pycuda
requirements.txt:scikits.cuda
pymoresane/parser.py:    parser.add_argument("-aog", "--allongpu", help="Specify whether as much code as possible is to be executed on the "
pymoresane/parser.py:                                                   "gpu. Overrides the behaviour of all other gpu options"
pymoresane/parser.py:                                                   "single CPU core, multiple CPU cores or the GPU."
pymoresane/parser.py:                                                   , default="ser", choices=["ser","mp","gpu"])
pymoresane/parser.py:                                                    "the GPU.", default="cpu", choices=["cpu","gpu"])
pymoresane/parser.py:                                                        "using the CPU or the GPU."
pymoresane/parser.py:                                                        , default="cpu", choices=["cpu","gpu"])
pymoresane/iuwt.py:    import pycuda.driver as drv
pymoresane/iuwt.py:    import pycuda.tools
pymoresane/iuwt.py:    import pycuda.autoinit
pymoresane/iuwt.py:    import pycuda.gpuarray as gpuarray
pymoresane/iuwt.py:    from pycuda.compiler import SourceModule
pymoresane/iuwt.py:    print "Pycuda unavailable - GPU mode will fail."
pymoresane/iuwt.py:                       store_on_gpu=False):
pymoresane/iuwt.py:    mode                (default='ser'):    Implementation of the IUWT to be used - 'ser', 'mp' or 'gpu'.
pymoresane/iuwt.py:    store_on_gpu        (default=False):    Boolean specifier for whether the decomposition is stored on the gpu or not.
pymoresane/iuwt.py:    elif mode=='gpu':
pymoresane/iuwt.py:        return gpu_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed, store_on_gpu)
pymoresane/iuwt.py:def iuwt_recomposition(in1, scale_adjust=0, mode='ser', core_count=1, store_on_gpu=False, smoothed_array=None):
pymoresane/iuwt.py:    mode                (default='ser')     Implementation of the IUWT to be used - 'ser', 'mp' or 'gpu'.
pymoresane/iuwt.py:    store_on_gpu        (default=False):    Boolean specifier for whether the decomposition is stored on the gpu or not.
pymoresane/iuwt.py:    elif mode=='gpu':
pymoresane/iuwt.py:        return gpu_iuwt_recomposition(in1, scale_adjust, store_on_gpu, smoothed_array)
pymoresane/iuwt.py:def gpu_iuwt_decomposition(in1, scale_count, scale_adjust, store_smoothed, store_on_gpu):
pymoresane/iuwt.py:    the isotropic undecimated wavelet transform implemented for a GPU.
pymoresane/iuwt.py:    store_on_gpu        (no default):   Boolean specifier for whether the decomposition is stored on the gpu or not.
pymoresane/iuwt.py:    # The following simple kernel just allows for the construction of a 3D decomposition on the GPU.
pymoresane/iuwt.py:                        __global__ void gpu_store_detail_coeffs(float *in1, float *in2, float* out1, int *scale, int *adjust)
pymoresane/iuwt.py:    wavelet_filter = gpuarray.to_gpu_async(wavelet_filter)
pymoresane/iuwt.py:    detail_coeffs = gpuarray.empty([scale_count-scale_adjust, in1.shape[0], in1.shape[1]], np.float32)
pymoresane/iuwt.py:    # Determines whether the array is already on the GPU or not. If not, moves it to the GPU.
pymoresane/iuwt.py:        gpu_in1 = gpuarray.to_gpu_async(in1.astype(np.float32))
pymoresane/iuwt.py:        gpu_in1 = in1
pymoresane/iuwt.py:    # Sets up some working arrays on the GPU to prevent memory transfers.
pymoresane/iuwt.py:    gpu_tmp = gpuarray.empty_like(gpu_in1)
pymoresane/iuwt.py:    gpu_out1 = gpuarray.empty_like(gpu_in1)
pymoresane/iuwt.py:    gpu_out2 = gpuarray.empty_like(gpu_in1)
pymoresane/iuwt.py:    # Sets up some parameters required by the algorithm on the GPU.
pymoresane/iuwt.py:    gpu_scale = gpuarray.zeros([1], np.int32)
pymoresane/iuwt.py:    gpu_adjust = gpuarray.zeros([1], np.int32)
pymoresane/iuwt.py:    gpu_adjust += scale_adjust
pymoresane/iuwt.py:    gpu_a_trous_row_kernel, gpu_a_trous_col_kernel = gpu_a_trous()
pymoresane/iuwt.py:    gpu_store_detail_coeffs = ker.get_function("gpu_store_detail_coeffs")
pymoresane/iuwt.py:            gpu_a_trous_row_kernel(gpu_in1, gpu_tmp, wavelet_filter, gpu_scale,
pymoresane/iuwt.py:            gpu_a_trous_col_kernel(gpu_tmp, gpu_out1, wavelet_filter, gpu_scale,
pymoresane/iuwt.py:            gpu_in1, gpu_out1 = gpu_out1, gpu_in1
pymoresane/iuwt.py:            gpu_scale += 1
pymoresane/iuwt.py:        gpu_a_trous_row_kernel(gpu_in1, gpu_tmp, wavelet_filter, gpu_scale,
pymoresane/iuwt.py:        gpu_a_trous_col_kernel(gpu_tmp, gpu_out1, wavelet_filter, gpu_scale,
pymoresane/iuwt.py:        gpu_a_trous_row_kernel(gpu_out1, gpu_tmp, wavelet_filter, gpu_scale,
pymoresane/iuwt.py:        gpu_a_trous_col_kernel(gpu_tmp, gpu_out2, wavelet_filter, gpu_scale,
pymoresane/iuwt.py:        gpu_store_detail_coeffs(gpu_in1, gpu_out2, detail_coeffs, gpu_scale, gpu_adjust,
pymoresane/iuwt.py:        gpu_in1, gpu_out1 = gpu_out1, gpu_in1
pymoresane/iuwt.py:        gpu_scale += 1
pymoresane/iuwt.py:    # Return values depend on mode. NOTE: store_smoothed does not work if the result stays on the gpu.
pymoresane/iuwt.py:    if store_on_gpu:
pymoresane/iuwt.py:        return detail_coeffs.get(), gpu_in1.get()
pymoresane/iuwt.py:def gpu_iuwt_recomposition(in1, scale_adjust, store_on_gpu, smoothed_array):
pymoresane/iuwt.py:    implementation of the isotropic undecimated wavelet transform recomposition for a GPU.
pymoresane/iuwt.py:    store_on_gpu    (no default):   Boolean specifier for whether the decomposition is stored on the gpu or not.
pymoresane/iuwt.py:    wavelet_filter = gpuarray.to_gpu_async(wavelet_filter)
pymoresane/iuwt.py:    # Determines scale with adjustment and creates a zero array on the GPU to store the output,unless smoothed_array
pymoresane/iuwt.py:        recomposition = gpuarray.zeros([in1.shape[1], in1.shape[2]], np.float32)
pymoresane/iuwt.py:        recomposition = gpuarray.to_gpu(smoothed_array.astype(np.float32))
pymoresane/iuwt.py:    # Determines whether the array is already on the GPU or not. If not, moves it to the GPU.
pymoresane/iuwt.py:        gpu_in1 = gpuarray.to_gpu_async(in1.astype(np.float32))
pymoresane/iuwt.py:        gpu_in1 = in1
pymoresane/iuwt.py:    # Creates a working array on the GPU.
pymoresane/iuwt.py:    gpu_tmp = gpuarray.empty_like(recomposition)
pymoresane/iuwt.py:    gpu_scale = gpuarray.zeros([1], np.int32)
pymoresane/iuwt.py:    gpu_scale += max_scale-1
pymoresane/iuwt.py:    gpu_a_trous_row_kernel, gpu_a_trous_col_kernel = gpu_a_trous()
pymoresane/iuwt.py:        gpu_a_trous_row_kernel(recomposition, gpu_tmp, wavelet_filter, gpu_scale,
pymoresane/iuwt.py:        gpu_a_trous_col_kernel(gpu_tmp, recomposition, wavelet_filter, gpu_scale,
pymoresane/iuwt.py:        recomposition = recomposition[:,:] + gpu_in1[i-scale_adjust,:,:]
pymoresane/iuwt.py:        gpu_scale -= 1
pymoresane/iuwt.py:            gpu_a_trous_row_kernel(recomposition, gpu_tmp, wavelet_filter, gpu_scale,
pymoresane/iuwt.py:            gpu_a_trous_col_kernel(gpu_tmp, recomposition, wavelet_filter, gpu_scale,
pymoresane/iuwt.py:            gpu_scale -= 1
pymoresane/iuwt.py:    if store_on_gpu:
pymoresane/iuwt.py:def gpu_a_trous():
pymoresane/iuwt.py:                        __global__ void gpu_a_trous_row_kernel(float *in1, float *in2, float *wfil, int *scale)
pymoresane/iuwt.py:                        __global__ void gpu_a_trous_col_kernel(float *in1, float *in2, float *wfil, int *scale)
pymoresane/iuwt.py:    return ker1.get_function("gpu_a_trous_row_kernel"), ker2.get_function("gpu_a_trous_col_kernel")
pymoresane/iuwt_convolution.py:    import pycuda.driver as drv
pymoresane/iuwt_convolution.py:    import pycuda.tools
pymoresane/iuwt_convolution.py:    import pycuda.autoinit
pymoresane/iuwt_convolution.py:    import pycuda.gpuarray as gpuarray
pymoresane/iuwt_convolution.py:    from pycuda.compiler import SourceModule
pymoresane/iuwt_convolution.py:    from scikits.cuda.fft import Plan
pymoresane/iuwt_convolution.py:    from scikits.cuda.fft import fft
pymoresane/iuwt_convolution.py:    from scikits.cuda.fft import ifft
pymoresane/iuwt_convolution.py:    print "Pycuda unavailable - GPU mode will fail."
pymoresane/iuwt_convolution.py:def fft_convolve(in1, in2, conv_device="cpu", conv_mode="linear", store_on_gpu=False):
pymoresane/iuwt_convolution.py:    and GPU.
pymoresane/iuwt_convolution.py:    in2             (no default):           Gpuarray containing the FFT of the PSF.
pymoresane/iuwt_convolution.py:    conv_device     (default = "cpu"):      Parameter which allows specification of "cpu" or "gpu".
pymoresane/iuwt_convolution.py:    if conv_device=='gpu':
pymoresane/iuwt_convolution.py:            fft_in1 = gpu_r2c_fft(fft_in1, store_on_gpu=True)
pymoresane/iuwt_convolution.py:            conv_in1_in2 = contiguous_slice(fft_shift(gpu_c2r_ifft(conv_in1_in2, is_gpuarray=True, store_on_gpu=True)))
pymoresane/iuwt_convolution.py:            if store_on_gpu:
pymoresane/iuwt_convolution.py:            fft_in1 = gpu_r2c_fft(in1, store_on_gpu=True)
pymoresane/iuwt_convolution.py:            conv_in1_in2 = fft_shift(gpu_c2r_ifft(conv_in1_in2, is_gpuarray=True, store_on_gpu=True))
pymoresane/iuwt_convolution.py:            if store_on_gpu:
pymoresane/iuwt_convolution.py:def gpu_r2c_fft(in1, is_gpuarray=False, store_on_gpu=False):
pymoresane/iuwt_convolution.py:    This function makes use of the scikits implementation of the FFT for GPUs to take the real to complex FFT.
pymoresane/iuwt_convolution.py:    is_gpuarray     (default=True):     Boolean specifier for whether or not input is on the gpu.
pymoresane/iuwt_convolution.py:    store_on_gpu    (default=False):    Boolean specifier for whether the result is to be left on the gpu or not.
pymoresane/iuwt_convolution.py:    gpu_out1                            The gpu array containing the result.
pymoresane/iuwt_convolution.py:    gpu_out1.get()                      The result from the gpu array.
pymoresane/iuwt_convolution.py:    if is_gpuarray:
pymoresane/iuwt_convolution.py:        gpu_in1 = in1
pymoresane/iuwt_convolution.py:        gpu_in1 = gpuarray.to_gpu_async(in1.astype(np.float32))
pymoresane/iuwt_convolution.py:    gpu_out1 = gpuarray.empty([output_size[0], output_size[1]], np.complex64)
pymoresane/iuwt_convolution.py:    gpu_plan = Plan(gpu_in1.shape, np.float32, np.complex64)
pymoresane/iuwt_convolution.py:    fft(gpu_in1, gpu_out1, gpu_plan)
pymoresane/iuwt_convolution.py:    if store_on_gpu:
pymoresane/iuwt_convolution.py:        return gpu_out1
pymoresane/iuwt_convolution.py:        return gpu_out1.get()
pymoresane/iuwt_convolution.py:def gpu_c2r_ifft(in1, is_gpuarray=False, store_on_gpu=False):
pymoresane/iuwt_convolution.py:    This function makes use of the scikits implementation of the FFT for GPUs to take the complex to real IFFT.
pymoresane/iuwt_convolution.py:    is_gpuarray     (default=True):     Boolean specifier for whether or not input is on the gpu.
pymoresane/iuwt_convolution.py:    store_on_gpu    (default=False):    Boolean specifier for whether the result is to be left on the gpu or not.
pymoresane/iuwt_convolution.py:    gpu_out1                            The gpu array containing the result.
pymoresane/iuwt_convolution.py:    gpu_out1.get()                      The result from the gpu array.
pymoresane/iuwt_convolution.py:    if is_gpuarray:
pymoresane/iuwt_convolution.py:        gpu_in1 = in1
pymoresane/iuwt_convolution.py:        gpu_in1 = gpuarray.to_gpu_async(in1.astype(np.complex64))
pymoresane/iuwt_convolution.py:    gpu_out1 = gpuarray.empty([output_size[0],output_size[1]], np.float32)
pymoresane/iuwt_convolution.py:    gpu_plan = Plan(output_size, np.complex64, np.float32)
pymoresane/iuwt_convolution.py:    ifft(gpu_in1, gpu_out1, gpu_plan)
pymoresane/iuwt_convolution.py:    scale_fft(gpu_out1)
pymoresane/iuwt_convolution.py:    if store_on_gpu:
pymoresane/iuwt_convolution.py:        return gpu_out1
pymoresane/iuwt_convolution.py:        return gpu_out1.get()
pymoresane/iuwt_convolution.py:    This function performs the FFT shift operation on the GPU to restore the correct output shape.
pymoresane/iuwt_convolution.py:    This function unpads an array on the GPU in such a way as to make it contiguous.
pymoresane/iuwt_convolution.py:    gpu_out1                Array containing unpadded, contiguous data.
pymoresane/iuwt_convolution.py:    gpu_out1 = gpuarray.empty([in1.shape[0]/2,in1.shape[1]/2], np.float32)
pymoresane/iuwt_convolution.py:    contiguous_slice_ker(in1, gpu_out1, block=(32,32,1), grid=(int(in1.shape[1]//32), int(in1.shape[0]//32)))
pymoresane/iuwt_convolution.py:    return gpu_out1
pymoresane/main.py:                 major_loop_miter=100, minor_loop_miter=30, all_on_gpu=False, decom_mode="ser", core_count=1,
pymoresane/main.py:        all_on_gpu          (default=False):    Boolean specifier to toggle all gpu modes on.
pymoresane/main.py:        decom_mode          (default='ser'):    Specifier for decomposition mode - serial, multiprocessing, or gpu.
pymoresane/main.py:        conv_device         (default='cpu'):    Specifier for device to be used - cpu or gpu.
pymoresane/main.py:        extraction_mode     (default='cpu'):    Specifier for mode to be used - cpu or gpu.
pymoresane/main.py:        if all_on_gpu:
pymoresane/main.py:            decom_mode = 'gpu'
pymoresane/main.py:            conv_device = 'gpu'
pymoresane/main.py:            extraction_mode = 'gpu'
pymoresane/main.py:        # The following pre-loads the gpu with the fft of both the full PSF and the subregion of interest. If usegpu
pymoresane/main.py:        if conv_device=="gpu":
pymoresane/main.py:                    psf_subregion_fft = conv.gpu_r2c_fft(psf_subregion, is_gpuarray=False, store_on_gpu=True)
pymoresane/main.py:                    psf_data_fft = conv.gpu_r2c_fft(psf_data_fft, is_gpuarray=False, store_on_gpu=True)
pymoresane/main.py:                    psf_subregion_fft = conv.gpu_r2c_fft(psf_subregion, is_gpuarray=False, store_on_gpu=True)
pymoresane/main.py:                        psf_data_fft = conv.gpu_r2c_fft(self.psf_data, is_gpuarray=False, store_on_gpu=True)
pymoresane/main.py:                        psf_subregion_fft = conv.gpu_r2c_fft(self.psf_data, is_gpuarray=False, store_on_gpu=True)
pymoresane/main.py:                        psf_subregion_fft = conv.gpu_r2c_fft(psf_subregion_fft, is_gpuarray=False, store_on_gpu=True)
pymoresane/main.py:                        psf_data_fft = conv.gpu_r2c_fft(self.psf_data, is_gpuarray=False, store_on_gpu=True)
pymoresane/main.py:                        psf_subregion_fft = conv.gpu_r2c_fft(psf_subregion_fft, is_gpuarray=False, store_on_gpu=True)
pymoresane/main.py:                        psf_subregion_fft = conv.gpu_r2c_fft(psf_subregion_fft, is_gpuarray=False, store_on_gpu=True)
pymoresane/main.py:                        psf_data_fft = conv.gpu_r2c_fft(psf_data_fft, is_gpuarray=False, store_on_gpu=True)
pymoresane/main.py:                    mode=extraction_mode, store_on_gpu=all_on_gpu,
pymoresane/main.py:                    Ap = conv.fft_convolve(p, psf_subregion_fft, conv_device, conv_mode, store_on_gpu=all_on_gpu)
pymoresane/main.py:                                                 store_on_gpu=all_on_gpu)
pymoresane/main.py:                        Ap = conv.fft_convolve(p, psf_subregion_fft, conv_device, conv_mode, store_on_gpu=all_on_gpu)
pymoresane/main.py:                                                     store_on_gpu=all_on_gpu)
pymoresane/main.py:                    model_sources = conv.fft_convolve(xn, psf_subregion_fft, conv_device, conv_mode, store_on_gpu=all_on_gpu)
pymoresane/main.py:                                                            core_count, store_on_gpu=all_on_gpu)
pymoresane/main.py:                    if all_on_gpu:
pymoresane/main.py:                          tolerance=0.75, accuracy=1e-6, major_loop_miter=100, minor_loop_miter=30, all_on_gpu=False,
pymoresane/main.py:        all_on_gpu          (default=False):    Boolean specifier to toggle all gpu modes on.
pymoresane/main.py:        decom_mode          (default='ser'):    Specifier for decomposition mode - serial, multiprocessing, or gpu.
pymoresane/main.py:        conv_device         (default='cpu'):    Specifier for device to be used - cpu or gpu.
pymoresane/main.py:        extraction_mode     (default='cpu'):    Specifier for mode to be used - cpu or gpu.
pymoresane/main.py:                          minor_loop_miter=minor_loop_miter, all_on_gpu=all_on_gpu, decom_mode=decom_mode,
pymoresane/main.py:                      args.majorloopmiter, args.minorloopmiter, args.allongpu, args.decommode, args.corecount,
pymoresane/main.py:                               args.tolerance, args.accuracy, args.majorloopmiter, args.minorloopmiter, args.allongpu,
pymoresane/main.py:    #                 decom_mode="gpu", extraction_mode="gpu", conv_device="gpu")
pymoresane/main.py:    #                 all_on_gpu=True, edge_suppression=True)
pymoresane/iuwt_toolbox.py:    import pycuda.driver as drv
pymoresane/iuwt_toolbox.py:    import pycuda.tools
pymoresane/iuwt_toolbox.py:    import pycuda.autoinit
pymoresane/iuwt_toolbox.py:    import pycuda.gpuarray as gpuarray
pymoresane/iuwt_toolbox.py:    from pycuda.compiler import SourceModule
pymoresane/iuwt_toolbox.py:    print "Pycuda unavailable - GPU mode will fail."
pymoresane/iuwt_toolbox.py:def source_extraction(in1, tolerance, mode="cpu", store_on_gpu=False,
pymoresane/iuwt_toolbox.py:    Convenience function for allocating work to cpu or gpu, depending on the selected mode.
pymoresane/iuwt_toolbox.py:    mode        (default="cpu"):Mode of operation - either "gpu" or "cpu".
pymoresane/iuwt_toolbox.py:    elif mode=="gpu":
pymoresane/iuwt_toolbox.py:        return gpu_source_extraction(in1, tolerance, store_on_gpu, neg_comp)
pymoresane/iuwt_toolbox.py:def gpu_source_extraction(in1, tolerance, store_on_gpu, neg_comp):
pymoresane/iuwt_toolbox.py:    significant objects across all scales. This GPU accelerated version speeds up the extraction process.
pymoresane/iuwt_toolbox.py:    store_on_gpu(no default):   Boolean specifier for whether the decomposition is stored on the gpu or not.
pymoresane/iuwt_toolbox.py:    objects                     The mask of the significant structures - if store_on_gpu is True, returns a gpuarray.
pymoresane/iuwt_toolbox.py:    # The following are pycuda kernels which are executed on the gpu. Specifically, these both perform thresholding
pymoresane/iuwt_toolbox.py:    # operations. The gpu is much faster at this on large arrays due to their massive parallel processing power.
pymoresane/iuwt_toolbox.py:                        __global__ void gpu_mask_kernel1(int *in1, int *in2)
pymoresane/iuwt_toolbox.py:                        __global__ void gpu_mask_kernel2(int *in1)
pymoresane/iuwt_toolbox.py:                        __global__ void gpu_store_objects(int *in1, float *out1, int *scale)
pymoresane/iuwt_toolbox.py:    # The following bind the pycuda kernels to the expressions on the left.
pymoresane/iuwt_toolbox.py:    gpu_mask_kernel1 = ker1.get_function("gpu_mask_kernel1")
pymoresane/iuwt_toolbox.py:    gpu_mask_kernel2 = ker2.get_function("gpu_mask_kernel2")
pymoresane/iuwt_toolbox.py:    # If store_on_gpu is the following handles some initialisation.
pymoresane/iuwt_toolbox.py:    if store_on_gpu:
pymoresane/iuwt_toolbox.py:        gpu_store_objects = ker3.get_function("gpu_store_objects")
pymoresane/iuwt_toolbox.py:        gpu_objects = gpuarray.empty(objects.shape, np.float32)
pymoresane/iuwt_toolbox.py:        gpu_idx = gpuarray.zeros([1], np.int32)
pymoresane/iuwt_toolbox.py:        gpu_idx += (objects.shape[0]-1)
pymoresane/iuwt_toolbox.py:        gpu_objects_page = gpuarray.to_gpu_async(objects[i,:,:].astype(np.int32))
pymoresane/iuwt_toolbox.py:            label = gpuarray.to_gpu_async(np.array(j))
pymoresane/iuwt_toolbox.py:            gpu_mask_kernel1(gpu_objects_page, label, block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[1]//32))
pymoresane/iuwt_toolbox.py:        gpu_mask_kernel2(gpu_objects_page, block=(32,32,1), grid=(in1.shape[1]//32, in1.shape[1]//32))
pymoresane/iuwt_toolbox.py:        objects[i,:,:] = gpu_objects_page.get()
pymoresane/iuwt_toolbox.py:        # In the event that all operations are to be done on the GPU, the following stores a version of the objects
pymoresane/iuwt_toolbox.py:        # on the GPU. A handle to the gpuarray is then returned.
pymoresane/iuwt_toolbox.py:        if store_on_gpu:
pymoresane/iuwt_toolbox.py:            gpu_store_objects(gpu_objects_page, gpu_objects, gpu_idx, block=(32,32,1), grid=(objects.shape[2]//32,
pymoresane/iuwt_toolbox.py:            gpu_idx -= 1
pymoresane/iuwt_toolbox.py:    if store_on_gpu:
pymoresane/iuwt_toolbox.py:        return objects*in1, gpu_objects
CHANGELOG.txt:Restored GPU functionality.
CHANGELOG.txt:Added better GPU error trapping.

```
