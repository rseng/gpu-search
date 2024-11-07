# https://github.com/lanl/scico

```console
setup.py:SCICO is a Python package for solving the inverse problems that arise in scientific imaging applications. Its primary focus is providing methods for solving ill-posed inverse problems by using an appropriate prior model of the reconstruction space. SCICO includes a growing suite of operators, cost functionals, regularizers, and optimization routines that may be combined to solve a wide range of problems, and is designed so that it is easy to add new building blocks. SCICO is built on top of JAX, which provides features such as automatic gradient calculation and GPU acceleration.
docs/source/include/examplenotes.rst:Note that ``astra-toolbox`` should be installed on a host with one or more CUDA GPUs to ensure
docs/source/include/examplenotes.rst:that the version with GPU support is installed.
docs/source/include/examplenotes.rst:to run on a workstation with multiple GPUs.
docs/source/advantages.rst:same features, but with the addition of automatic differentiation, GPU
docs/source/advantages.rst:GPU support and JIT compilation both offer the potential for significant
docs/source/advantages.rst:the same code on a GPU rather than a CPU, and similar speed gains can
docs/source/advantages.rst:with an Intel Xeon Gold 6230 CPU and NVIDIA GeForce RTX 2080 Ti
docs/source/advantages.rst:GPU. It is interesting to note that for :class:`.FiniteDifference` the
docs/source/advantages.rst:GPU provides no acceleration, while JIT provides more than an order of
docs/source/advantages.rst:magnitude of speed improvement on both CPU and GPU. For :class:`.DFT`
docs/source/advantages.rst:GPU, which also provides significant acceleration over the CPU.
docs/source/advantages.rst:     :alt: Timing results for SCICO operators on CPU and GPU with and without JIT
docs/source/advantages.rst:simple GPU support offered by SCICO.
docs/source/advantages.rst:with many `SciPy <https://scipy.org/>`__ solvers. GPU support is
docs/source/advantages.rst:that switching for a CPU to GPU requires code changes, unlike SCICO and
docs/source/install.rst:`GPU support <https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl>`_.
docs/source/install.rst:GPU Support
docs/source/install.rst:a version with GPU support:
docs/source/install.rst:2. Install the version of jaxlib with GPU support, as described in the `JAX installation
docs/source/install.rst:      pip install --upgrade "jax[cuda12]"
docs/source/install.rst:   for CUDA 12, but it may be necessary to explicitly specify the
docs/source/install.rst:`misc/gpu/envinfo.py <https://github.com/lanl/scico/blob/main/misc/gpu/envinfo.py>`_
docs/source/install.rst:in the source distribution is provided as an aid to debugging GPU support
docs/source/install.rst:`misc/gpu/availgpu.py <https://github.com/lanl/scico/blob/main/misc/gpu/availgpu.py>`_
docs/source/install.rst:can be used to automatically recommend a setting of the CUDA_VISIBLE_DEVICES
docs/source/install.rst:environment variable that excludes GPUs that are already in use.
docs/source/notes.rst:Use of the CPU device can be forced even when GPUs are present by setting the
docs/source/notes.rst:on a platform without a GPU, but this should no longer be necessary for any
docs/source/notes.rst:By default, JAX will preallocate a large chunk of GPU memory on startup. This
docs/source/notes.rst:the relevant `section of the JAX docs <https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html>`__.
docs/source/notes.rst:platform with GPUs, the remainder of the code will run on a GPU, but
docs/source/notes.rst:there is potential for a considerable delay due to host-GPU memory
docs/source/notes.rst:GPU acceleration support, but efficiency is expected to be lower than
docs/source/notes.rst:JAX-based code due to host-GPU memory transfers.
docs/source/notes.rst:NumPy :class:`~numpy.ndarray`, but can be backed by CPU, GPU, or TPU
docs/source/notes.rst:to GPU. Consider this toy example on a system with a GPU present:
docs/source/notes.rst:   y = snp.dot(A, x)         # A, x transfered to GPU
docs/source/notes.rst:                             # y resides on GPU
docs/source/notes.rst:   z = y + x                 # x must be transfered to GPU again
docs/source/notes.rst:   x = jax.device_put(x)     # transfer to GPU
docs/source/notes.rst:On a multi-GPU system, :func:`jax.device_put` can place data on a specific
docs/source/notes.rst:GPU. See the `JAX notes on data placement
docs/source/inverse.rst:support execution of the same code on both CPU and GPU devices, and we
docs/source/inverse.rst:advantage when GPUs are available. As an example, the following code
docs/source/overview.rst:<https://numpy.org/>`__, enabling GPU/TPU acceleration, just-in-time
README.md:automatic gradient calculation and GPU acceleration.
misc/README.rst:- ``gpu``: Scripts for debugging and managing JAX use of GPUs.
misc/conda/README.rst:To create a conda environment called ``scico`` with Python version 3.12 and without GPU support
misc/conda/README.rst:To include GPU support, follow the `jax installation instructions <https://github.com/google/jax#pip-installation-gpu-cuda>`__ after
misc/conda/README.rst:- Installation of jaxlib with GPU capabilities is not supported.
misc/conda/make_conda_env.sh:# that are not addressed, and that installation of jaxlib with GPU
misc/conda/make_conda_env.sh:if [ "$OS" == "Darwin" ] && [ "$GPU" == yes ]; then
misc/conda/make_conda_env.sh:    echo "Error: GPU-enabled jaxlib installation not supported under OSX" >&2
misc/conda/make_conda_env.sh:echo "JAX installed without GPU support. To enable GPU support, install a"
misc/conda/make_conda_env.sh:echo "version of jaxlib with CUDA support following the instructions at"
misc/conda/make_conda_env.sh:echo "   https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu"
misc/conda/make_conda_env.sh:echo "   pip install -U \"jax[cuda12]\""
misc/conda/make_conda_env.sh:echo "ASTRA Toolbox installed without GPU support if this script was"
misc/conda/make_conda_env.sh:echo "run on a host without CUDA drivers installed. To enable GPU support,"
misc/conda/make_conda_env.sh:echo "host with CUDA drivers installed."
misc/gpu/README.rst:GPU Utility Scripts
misc/gpu/README.rst:These scripts are intended for debugging and managing JAX use of GPUs:
misc/gpu/README.rst:- ``availgpu.py``: Automatically recommend a setting of the ``CUDA_VISIBLE_DEVICES`` environment variable that excludes GPUs that are already in use.
misc/gpu/README.rst:- ``envinfo.py``: An aid to debugging JAX GPU access.
misc/gpu/envinfo.py:# a Python host has available GPUs, and if so, whether the JAX installation
misc/gpu/envinfo.py:    import GPUtil
misc/gpu/envinfo.py:    have_gputil = True
misc/gpu/envinfo.py:    have_gputil = False
misc/gpu/envinfo.py:    missing.append("gputil")
misc/gpu/envinfo.py:if have_gputil:
misc/gpu/envinfo.py:    if GPUtil.getAvailable():
misc/gpu/envinfo.py:        print("GPUs:")
misc/gpu/envinfo.py:        for gpu in GPUtil.getGPUs():
misc/gpu/envinfo.py:            print(f"    {gpu.id:2d}  {gpu.name:10s}  {gpu.memoryTotal} kB RAM")
misc/gpu/envinfo.py:        print("No GPUs available")
misc/gpu/envinfo.py:    print("No GPUs available to JAX (JAX device is CPU)")
misc/gpu/envinfo.py:    print(f"Number of GPUs available to JAX: {jax.device_count()}")
misc/gpu/availgpu.py:# Determine which GPUs available for use and recommend CUDA_VISIBLE_DEVICES
misc/gpu/availgpu.py:import GPUtil
misc/gpu/availgpu.py:print("GPU utlizitation")
misc/gpu/availgpu.py:GPUtil.showUtilization()
misc/gpu/availgpu.py:devIDs = GPUtil.getAvailable(
misc/gpu/availgpu.py:Ngpu = len(GPUtil.getGPUs())
misc/gpu/availgpu.py:if len(devIDs) == Ngpu:
misc/gpu/availgpu.py:    print(f"All {Ngpu} GPUs available for use")
misc/gpu/availgpu.py:    print(f"Only {len(devIDs)} of {Ngpu} GPUs available for use")
misc/gpu/availgpu.py:    print("To avoid attempting to use GPUs already in use, run the command")
misc/gpu/availgpu.py:    print(f"    export CUDA_VISIBLE_DEVICES={','.join(map(str, devIDs))}")
scico/ray/tune.py:        resources_per_trial: A dict mapping keys "cpu" and "gpu" to
scico/ray/tune.py:           resources: A dict mapping keys "cpu" and "gpu" to integers
scico/test/linop/xray/test_astra.py:RTOL_GPU = 7e-2
scico/test/linop/xray/test_astra.py:RTOL_GPU_RANDOM_INPUT = 1.0
scico/test/linop/xray/test_astra.py:        rtol = RTOL_GPU  # astra inaccurate in GPU
scico/test/linop/xray/test_astra.py:        rtol = RTOL_GPU_RANDOM_INPUT  # astra more inaccurate in GPU for random inputs
scico/test/linop/xray/test_astra.py:@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="checking GPU behavior")
scico/test/linop/xray/test_astra.py:def test_3D_on_GPU():
scico/test/linop/xray/test_astra.py:@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for test")
scico/test/linop/xray/test_astra.py:@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="GPU required for test")
scico/test/linop/xray/test_astra.py:    proj_id, proj = scico.linop.xray.astra.astra.create_sino3d_gpu(vol, proj_geom, vol_geom)
scico/test/test_ray_tune.py:    resources = {"gpu": 0, "cpu": 1}
scico/test/test_ray_tune.py:    resources = {"gpu": 0, "cpu": 1}
scico/test/test_ray_tune.py:    resources = {"gpu": 0, "cpu": 1}
scico/test/test_ray_tune.py:    resources = {"gpu": 0, "cpu": 1}
scico/linop/xray/_xray.py:    allowing it to run on either CPU or GPU and minimizing host copies.
scico/linop/xray/svmbir.py:automatic differentiation and support for GPU devices are not available.
scico/linop/xray/astra.py:This package provides both C and CUDA implementations of core
scico/linop/xray/astra.py:functionality, but note that use of the CUDA/GPU implementation is
scico/linop/xray/astra.py:expected to result in GPU-host-GPU memory copies when transferring
scico/linop/xray/astra.py:def set_astra_gpu_index(idx: Union[int, Sequence[int]]):
scico/linop/xray/astra.py:    """Set the index/indices of GPU(s) to be used by astra.
scico/linop/xray/astra.py:        idx: Index or indices of GPU(s).
scico/linop/xray/astra.py:    astra.set_gpu_index(idx)
scico/linop/xray/astra.py:               One of ["auto", "gpu", "cpu"]. If "auto", a GPU is used if
scico/linop/xray/astra.py:        if device in ["cpu", "gpu"]:
scico/linop/xray/astra.py:            # If cpu or gpu selected, attempt to comply (no checking to
scico/linop/xray/astra.py:            # confirm that a gpu is available to astra).
scico/linop/xray/astra.py:            # If auto selected, use cpu or gpu depending on the default
scico/linop/xray/astra.py:            # jax device (for simplicity, no checking whether gpu is
scico/linop/xray/astra.py:        elif self.device == "gpu":
scico/linop/xray/astra.py:            self.proj_id = astra.create_projector("cuda", self.proj_geom, self.vol_geom)
scico/linop/xray/astra.py:               <https://www.astra-toolbox.com/docs/algs/FBP_CUDA.html>`__.
scico/linop/xray/astra.py:            cfg = astra.astra_dict("FBP_CUDA" if self.device == "gpu" else "FBP")
scico/linop/xray/astra.py:    Note that a CUDA GPU is required for the primary functionality of
scico/linop/xray/astra.py:    this class; if no GPU is available, initialization will fail with a
scico/linop/xray/astra.py:            RuntimeError: If a CUDA GPU is not available to the ASTRA
scico/linop/xray/astra.py:        if not astra.use_cuda():
scico/linop/xray/astra.py:            raise RuntimeError("CUDA GPU required but not available or not enabled.")
scico/linop/xray/astra.py:            proj_id, result = astra.create_sino3d_gpu(x, self.proj_geom, self.vol_geom)
scico/linop/xray/astra.py:            proj_id, result = astra.create_backprojection3d_gpu(y, self.proj_geom, self.vol_geom)
scico/linop/xray/__init__.py:both CPU and GPU devices, while the ASTRA transform is implemented in
scico/linop/xray/__init__.py:CUDA, and can only be run on GPU devices.
scico/denoiser.py:    and support for GPU devices are not available.
scico/denoiser.py:    and support for GPU devices are not available.
scico/functional/_denoiser.py:    and support for GPU devices are not available.
scico/functional/_denoiser.py:    and support for GPU devices are not available.
scico/__init__.py:    logging.Filter("No GPU/TPU found, falling back to CPU.")
scico/flax/examples/data_generation.py:    to avoid the risk of errors when running with GPU devices, in which
scico/flax/examples/data_generation.py:    case jax is initialized to expect the availability of GPUs, which are
scico/flax/examples/data_generation.py:    of any declared GPUs as a `num_gpus` parameter of `@ray.remote`.
scico/flax/examples/data_generation.py:    # as many actors as available GPUs are created), and is expected to be
scico/flax/examples/data_generation.py:    if "GPU" in ar:
scico/flax/examples/data_generation.py:        num_gpus = 1
scico/flax/examples/data_generation.py:        nproc = min(nproc, int(ar.get("GPU")))
scico/flax/examples/data_generation.py:        num_gpus = 0
scico/flax/examples/data_generation.py:    @ray.remote(num_gpus=num_gpus)
scico/flax/train/trainer.py:        size_device_prefetch = 2  # Set for GPU
scico/flax/train/apply.py:    size_device_prefetch = 2  # Set for GPU
examples/README.rst:Running on a GPU
examples/README.rst:If a GPU is not available, or if the available GPU does not have sufficient memory to build the notebooks, set the environment variable
examples/README.rst:By default, ``makenotebooks.py`` only rebuilds notebooks that are out of date with respect to their corresponding example scripts, as determined by their respective file timestamps. However, timestamps for files retrieved from version control may not be meaningful for this purpose. To rebuild all examples, the following commands (assuming that GPUs are available) are recommended:
examples/scripts/ct_unet_train_foam2.py:# Set an arbitrary processor count (only applies if GPU is not available).
examples/scripts/deconv_tv_admm_tune.py:warnings that are emitted when GPU resources are requested but not available,
examples/scripts/deconv_tv_admm_tune.py:not force use of the CPU only. To enable GPU usage, comment out the
examples/scripts/deconv_tv_admm_tune.py:value of the "gpu" entry in the `resources` dict from 0 to 1. Note that
examples/scripts/deconv_tv_admm_tune.py:resources = {"cpu": 4, "gpu": 0}  # cpus per trial, gpus per trial
examples/scripts/ct_odp_train_foam2.py:# Set an arbitrary processor count (only applies if GPU is not available).
examples/scripts/ct_abel_tv_admm_tune.py:warnings that are emitted when GPU resources are requested but not
examples/scripts/ct_abel_tv_admm_tune.py:way that does not force use of the CPU only. To enable GPU usage, comment
examples/scripts/ct_abel_tv_admm_tune.py:change the value of the "gpu" entry in the `resources` dict from 0 to 1.
examples/scripts/ct_abel_tv_admm_tune.py:resources = {"gpu": 0, "cpu": 1}  # gpus per trial, cpus per trial
examples/scripts/ct_modl_train_foam2.py:applies if GPU is not available).
examples/scripts/denoise_dncnn_train_bsds.py:# Set an arbitrary processor count (only applies if GPU is not available).
examples/scripts/ct_datagen_foam2.py:# Set an arbitrary processor count (only applies if GPU is not available).
examples/scripts/deconv_modl_train_foam1.py:# Set an arbitrary processor count (only applies if GPU is not available).
examples/scripts/deconv_microscopy_allchn_tv_admm.py:example on a GPU it may be necessary to set environment variables
examples/scripts/deconv_microscopy_allchn_tv_admm.py:`XLA_PYTHON_CLIENT_PREALLOCATE=false`. If your GPU does not have enough
examples/scripts/deconv_microscopy_allchn_tv_admm.py:ngpu = 0
examples/scripts/deconv_microscopy_allchn_tv_admm.py:if "GPU" in ar:
examples/scripts/deconv_microscopy_allchn_tv_admm.py:    ngpu = int(ar["GPU"]) // 3
examples/scripts/deconv_microscopy_allchn_tv_admm.py:print(f"Running on {ncpu} CPUs and {ngpu} GPUs per process")
examples/scripts/deconv_microscopy_allchn_tv_admm.py:@ray.remote(num_cpus=ncpu, num_gpus=ngpu)
examples/scripts/ct_projector_comparison_2d.py:On our server, when using the GPU, the SCICO projector (both forward
examples/scripts/ct_projector_comparison_2d.py:On our server, using the GPU:
examples/scripts/deconv_odp_train_foam1.py:# Set an arbitrary processor count (only applies if GPU is not available).
examples/scripts/deconv_microscopy_tv_admm.py:example on a GPU it may be necessary to set environment variables
examples/scripts/deconv_microscopy_tv_admm.py:`XLA_PYTHON_CLIENT_PREALLOCATE=false`. If your GPU does not have enough
examples/makenotebooks.py:    ngpu = 0
examples/makenotebooks.py:    if "GPU" in ar:
examples/makenotebooks.py:        ngpu = max(int(ar["GPU"]) // nproc, 1)
examples/makenotebooks.py:        print(f"    Running on {ncpu} CPUs and {ngpu} GPUs per process")
examples/makenotebooks.py:    @ray.remote(num_cpus=ncpu, num_gpus=ngpu)
examples/scriptcheck.sh:          [-g] Skip tests that need a GPU
examples/scriptcheck.sh:SKIP_GPU=0
examples/scriptcheck.sh:    g) SKIP_GPU=1;;
examples/scriptcheck.sh:    if [ $SKIP_GPU -eq 1 ] && grep -q '_astra_3d' <<< $f; then
examples/scriptcheck.sh:    if [ $SKIP_GPU -eq 1 ] && grep -q 'ct_projector_comparison_3d' <<< $f; then

```
