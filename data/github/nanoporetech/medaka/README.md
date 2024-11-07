# https://github.com/nanoporetech/medaka

```console
docs/index.rst:  * 50X faster than Nanopolish (and can run on GPUs)..
docs/index.rst:use of ``medaka`` with his RTX 2080 GPU.
docs/benchmarks.rst:to ``medaka`` or ``nanopolish``. A gpu-enabled version (commit ``896b8066``) of
docs/benchmarks.rst:NVIDIA 1080Ti GPU, the total execution time for ``medaka`` itself was
docs/installation.rst:The default installation has the capacity to run on a GPU (see _Using a GPU_ below),
docs/installation.rst:to run on GPU, you may wish to install the CPU-only version with:
docs/installation.rst:run on GPU, modify the above to:
docs/installation.rst:**Using a GPU**
docs/installation.rst:when installing through `pip` can make immediate use of GPUs via NVIDIA CUDA.
docs/installation.rst:the CUDA and cuDNN libraries; users are directed to the 
docs/installation.rst:[cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive), whilst CUDA from
docs/installation.rst:the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
docs/installation.rst:As described above, if the capability to run on GPU is not required, `medaka-cpu`
docs/installation.rst:can be installed with a CPU-only version of PyTorch that doesn't depend on the CUDA
docs/installation.rst:*GPU Usage notes*
docs/installation.rst:Depending on your GPU, `medaka` may show out of memory errors when running.
docs/installation.rst:`-b 100` is suitable for 11Gb GPUs.
CHANGELOG.md:- CUDA initialization errors during `medaka smolecule`s stitch phase.
CHANGELOG.md:- Publish ARMv8 wheels compatible with NVIDIA's [Jetpack 4.6.1 binary](https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow).
CHANGELOG.md: - Log use of GPU and cuDNN, noting workaround for RTX cards
CHANGELOG.md: - Limit CPU usage when running without a GPU
README.md:  * 50X faster than Nanopolish (and can run on GPUs).
README.md:The default installation has the capacity to run on a GPU (see _Using a GPU_ below),
README.md:to run on GPU, you may wish to install the CPU-only version with:
README.md:run on GPU, modify the above to:
README.md:**Using a GPU**
README.md:when installing through `pip` can make immediate use of GPUs via NVIDIA CUDA.
README.md:the CUDA and cuDNN libraries; users are directed to the 
README.md:[cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive), whilst CUDA from
README.md:the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
README.md:> [conda-forge]https://conda-forge.org/docs/user/tipsandtricks/#installing-cuda-enabled-packages-like-tensorflow-and-pytorch)
README.md:As described above, if the capability to run on GPU is not required, `medaka-cpu`
README.md:CUDA libraries, as follows:
README.md:*GPU Usage notes*
README.md:Depending on your GPU, `medaka` may show out of memory errors when running.
README.md:`-b 100` is suitable for 11Gb GPUs.
README.md:use of `medaka` with his RTX 2080 GPU.
medaka/stitch.py:                initializer=medaka.common.cuda_visible_devices) as executor:
medaka/tandem.py:    # we run this in a subprocess so GPU resources are all cleaned
medaka/smolecule.py:    # we run this in a subprocess so GPU resources are all cleaned
medaka/medaka.py:    tparser.add_argument('--device', type=int, default=0, help='GPU device to use.')
medaka/medaka.py:    tag_group.add_argument('--full_precision', action='store_true', default=False, help='Run model in full precision (default is half on GPU).')
medaka/training.py:    with torch.cuda.device('cuda:{}'.format(args.device)):
medaka/training.py:    model = model.to('cuda')
medaka/training.py:    scaler = torch.cuda.amp.GradScaler()
medaka/prediction.py:        if torch.cuda.device_count() > 0:
medaka/prediction.py:            logger.info("Found a GPU.")
medaka/prediction.py:            device = torch.device("cuda")
medaka/prediction.py:            #     "variable `TF_FORCE_GPU_ALLOW_GROWTH=true`. To explicitely "
medaka/torch_ext.py:    :param scaler: optional torch.cuda.amp.GradScaler, gradient scaler.
medaka/common.py:def cuda_visible_devices(devices=""):
medaka/common.py:    """Set CUDA devices.
medaka/common.py:    disable child access to CUDA devices.
medaka/common.py:    os.environ["CUDA_VISIBLE_DEVICES"] = devices
conda/build.sh:#export CONDA_OVERRIDE_CUDA="11.8"
.gitlab-ci.yml:              COMPUTE: ["gpu", "cpu"]
.gitlab-ci.yml:              COMPUTE: ["gpu", "cpu"]
.gitlab-ci.yml:              COMPUTE: ["gpu"]

```
