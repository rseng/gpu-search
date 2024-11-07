# https://github.com/lucatelli/morphen

```console
libs/libs.py:    print('Jax/GPU Libraries not imported.')
libs/libs.py:#setting the GPU memory fraction to be used of 25% should be fine!
libs/libs.py:def sersic2D_GPU(xy, x0=256, y0=256, PA=10, ell=0.9,
libs/libs.py:    xx, yy = rotation_GPU(PA, x0, y0, x, y)
libs/libs.py:def sersic2D_GPU_new(xy, params):
libs/libs.py:    xx, yy = rotation_GPU(PA, x0, y0, x, y)
libs/libs.py:def rotation_GPU(PA, x0, y0, x, y):
libs/libs.py:             fix_x0_y0=False, psf_name=None, convolution_mode='GPU',
libs/libs.py:    Perform a Robust and Fast Multi-Sersic Decomposition with GPU acceleration.
libs/libs.py:        If 'GPU', use GPU acceleration for the convolution.
libs/libs.py:        if convolution_mode == 'GPU':
libs/libs.py:    if convolution_mode == 'GPU':
libs/libs.py:        data_2D_gpu = jnp.array(data_2D)
libs/libs.py:        if convolution_mode == 'GPU':
libs/libs.py:        if convolution_mode == 'GPU':
libs/libs.py:    if convolution_mode == 'GPU':
libs/libs.py:    # if convolution_mode == 'GPU':
libs/libs.py:    if convolution_mode == 'GPU':
libs/libs.py:    #         model = model + sersic2D_GPU(xy, params_i[0],
libs/libs.py:    #         model = model + sersic2D_GPU(xy, model_params[0],
libs/libs.py:    # batched_sersic2D_GPU = vmap(sersic2D_GPU, in_axes=(None, 0))
libs/libs.py:    #     # Use the batched version of sersic2D_GPU to compute all models in parallel
libs/libs.py:    #     models = batched_sersic2D_GPU(xy, param_matrix)
libs/libs.py:    # def sersic2D_GPU_vectorized(xy, params):
libs/libs.py:    #     # Assuming sersic2D_GPU is compatible with JAX and can work with batched inputs
libs/libs.py:    #     return sersic2D_GPU_new(xy, params)
libs/libs.py:    # batched_sersic2D_GPU = vmap(sersic2D_GPU_vectorized, in_axes=(None, 0))
libs/libs.py:    #     models = batched_sersic2D_GPU(xy, param_matrix)
libs/libs.py:    def min_min_residual_2D_GPU(params):
libs/libs.py:            model = model + sersic2D_GPU(xy,
libs/libs.py:        # MODEL_2D_conv = convolve_on_gpu(model, PSF_BEAM)
libs/libs.py:        residual = data_2D_gpu - MODEL_2D_conv
libs/libs.py:            model = model + sersic2D_GPU(xy, model_params[0],
libs/libs.py:    def min_residual_2D_GPU(params):
libs/libs.py:            model = model + sersic2D_GPU(xy,
libs/libs.py:        # residual = ((data_2D_gpu[mask_for_fit] - MODEL_2D_conv[mask_for_fit])/
libs/libs.py:        residual = (data_2D_gpu[mask_for_fit] - MODEL_2D_conv[mask_for_fit])
libs/libs.py:    if convolution_mode == 'GPU':
libs/libs.py:        def convolve_on_gpu(image, psf):
libs/libs.py:        jax_convolve = jit(convolve_on_gpu)
libs/libs.py:    if convolution_mode == 'GPU':
libs/libs.py:        mini = lmfit.Minimizer(min_residual_2D_GPU, params, max_nfev=200000,
libs/libs.py:        model_temp = sersic2D_GPU(xy, params['f' + str(i) + '_x0'].value,
libs/libs.py:            if convolution_mode == 'GPU':
libs/libs.py:        if convolution_mode == 'GPU':
libs/libs.py:def compute_model_stats_GPU(params, imagename, residualname, psf_name,
libs/libs.py:    on a CUDA GPU.
libs/libs.py:    print('Running MCMC on best-fit parameters (using cuda gpu).')
libs/libs.py:            # cp.cuda.Stream.null.synchronize()
libs/libs.py:    total_fluxes_all_gpu = []
libs/libs.py:    residuals_gpu = []
libs/libs.py:    residuals_nrss_gpu = []
libs/libs.py:    flags_gpu = []
libs/libs.py:    sub_comp_fluxes_gpu = []
libs/libs.py:    sub_comp_residuals_gpu = []
libs/libs.py:        total_fluxes_all_gpu.append(dict_temp[0])
libs/libs.py:        flags_gpu.append(dict_temp[1])
libs/libs.py:        residuals_gpu.append(dict_temp[2])
libs/libs.py:        residuals_nrss_gpu.append(dict_temp[3])
libs/libs.py:        sub_comp_fluxes_gpu.append(dict_temp[4])
libs/libs.py:        sub_comp_residuals_gpu.append(dict_temp[5])
libs/libs.py:    total_fluxes_all = jnp.asarray(total_fluxes_all_gpu)
libs/libs.py:    flags = jnp.asarray(flags_gpu).get()
libs/libs.py:    residuals = jnp.asarray(residuals_gpu)
libs/libs.py:    residuals_nrss = jnp.asarray(residuals_nrss_gpu)
libs/libs.py:    sub_comp_fluxes = jnp.asarray(sub_comp_fluxes_gpu)[flags]
libs/libs.py:    sub_comp_residuals = jnp.asarray(sub_comp_residuals_gpu)[flags]
libs/libs.py:    GPU Optimized function to run a MCMC of a multi-dimensional sersic model.
libs/libs.py:    Please, only use if you have a GPU, if not, it can take days....
libs/libs.py:    data_2D_gpu = jnp.array(data_2D)
libs/libs.py:    FlatSky_level_GPU = jnp.array(mad_std(data_2D))
libs/libs.py:    def convolve_on_gpu(image, psf):
libs/libs.py:            model = sersic2D_GPU(xy,
libs/libs.py:                    FlatSky(FlatSky_level_GPU, random_params[-1]) / nfunctions
libs/libs.py:        MODEL_2D_conv = convolve_on_gpu(model, PSF_BEAM)
libs/libs.py:        residual = data_2D_gpu - MODEL_2D_conv
libs/libs.py:        chi2 = jnp.sum(residual ** 2) / jnp.sum(data_2D_gpu)
libs/libs.py:                      convolution_mode='GPU', workers=6,
libs/libs.py:    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_decomposition/README.md:   functions for convolution. JaX allows processes to run on Nvidia GPUs (through cuda), which 
image_decomposition/README.md:   runtime very significantly for convolution operations. Furthermore, if no GPU is present, 
image_decomposition/README.md:   installed JaX (https://jax.readthedocs.io/en/latest/installation.html). Note that GPU 
image_decomposition/README.md:                                convolution_mode='GPU',
image_decomposition/README.md:                                convolution_mode='GPU',self_bkg=True,
image_decomposition/README.md:                                                   convolution_mode='GPU',fix_geometry=False,workers=6,
image_decomposition/README.md:multi-thread CPU and GPU processing. If you have GPU, it benefits from both the CPU 
image_decomposition/README.md:and GPU.
README.md:This implementation is designed to be robust, fast with GPU acceleration using 
README.md:[//]: # (with a GPU optmisation layer &#40;Jax&#41;. )
install_instructions.md:### Jax Interface for Optmisation and GPU Processing
install_instructions.md:Nvidia GPUs (through CUDA).
install_instructions.md:Jax can be installed with `cuda-12.0`. The recommended way is to install within a conda 
install_instructions.md:### When Nvidia GPU is available
install_instructions.md:#### Cuda 12
install_instructions.md:pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
install_instructions.md:#### Cuda 11
install_instructions.md:pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
install_instructions.md:Note: In either case, the code can be run independently of the system. Jax will automatically detect if a GPU is 
install_instructions.md:Using Jax, run time can be reduced by a factor of 10-20 if running in a CPU, or by a factor of 100-500 if running in a GPU!
install_instructions.md:1. The GPU implementation of JAX was not tested in this container. The definition 
install_instructions.md:2. In the definition file, the option to build `wsclean` with GPU support may not work in 
singularity/morphen-gpu.def:	APPLICATION_NAME Ubuntu LTS + Nvidia CUDA + wsclean + IMFIT + JaX + CASA + Python Packages
singularity/morphen-gpu.def:	APPLICATION_URL https://developer.nvidia.com/cuda-zone 
singularity/morphen-gpu.def:	# Nvidia CUDA Path
singularity/morphen-gpu.def:	export CPATH="/usr/local/cuda/include:$CPATH"
singularity/morphen-gpu.def:	export PATH="/usr/local/cuda/bin:$PATH"
singularity/morphen-gpu.def:	export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/:/usr/local/cuda/lib64:/opt/build/EveryBeam/lib:$LD_LIBRARY_PATH"
singularity/morphen-gpu.def:	export CUDA_HOME="/usr/local/cuda"
singularity/morphen-gpu.def:	# Nvidia CUDA Path
singularity/morphen-gpu.def:	export CPATH="/usr/local/cuda/include:$CPATH"
singularity/morphen-gpu.def:	export PATH="/usr/local/cuda/bin:$PATH"
singularity/morphen-gpu.def:	export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
singularity/morphen-gpu.def:	export CUDA_HOME="/usr/local/cuda"
singularity/morphen-gpu.def:	#export CPATH="/usr/lib/cuda/include:$CPATH"
singularity/morphen-gpu.def:	#export PATH="/usr/lib/cuda/bin:$PATH"
singularity/morphen-gpu.def:	#export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/:/usr/lib/cuda/lib64:$LD_LIBRARY_PATH"
singularity/morphen-gpu.def:	#export CUDA_HOME="/usr/lib/cuda"
singularity/morphen-gpu.def:	apt install -y nvidia-driver-535 nvidia-utils-535 libnvidia-common-535 nvidia-settings
singularity/morphen-gpu.def:	#Create /opt/cuda for binding
singularity/morphen-gpu.def:	mkdir -p /usr/local/cuda-12
singularity/morphen-gpu.def:pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
singularity/morphen-gpu.def:ln -s /usr/local/cuda-12 /usr/local/cuda
singularity/morphen-gpu.def:cmake -DBUILD_LIB_CUDA=ON . 
singularity/morphen-gpu.def:#/usr/lib/cuda
morphen.py:    It supports GPU-acceleration using Jax. If no GPU is present, Jax still 
morphen.py:    CPU or GPU. 
morphen.py:                 convolution_mode='GPU',method1='least_squares',
morphen.py:    It supports GPU-acceleration using Jax. If no GPU is present, Jax still
morphen.py:    CPU or GPU.
morphen.py:                 convolution_mode='GPU',method1='least_squares',
imaging/imaging_with_wsclean.py:        # wsclean_dir = '/media/sagauga/xfs_evo/morphen_gpu_v2.simg'
imaging/imaging_with_wsclean.py:        # wsclean_dir = '/home/sagauga/apps/wsclean_nvidia470_gpu.simg'
imaging/imaging_with_wsclean.py:        # wsclean_dir = '/home/sagauga/apps/wsclean_nvidia470_gpu.simg'
imaging/wsclean_imaging.md:    wsclean_dir = '/home/user/apps/wsclean-gpu.simg
selfcal/imaging_with_wsclean.py:        # wsclean_dir = '/media/sagauga/xfs_evo/morphen_gpu_v2.simg'
selfcal/imaging_with_wsclean.py:        # wsclean_dir = '/media/sagauga/xfs_evo/morphen_gpu_v2.simg'
selfcal/imaging_with_wsclean.py:        # wsclean_dir = '/home/sagauga/apps/wsclean_nvidia470_gpu.simg'
selfcal/imaging_with_wsclean.py:        # wsclean_dir = '/home/sagauga/apps/wsclean_nvidia470_gpu.simg'

```
