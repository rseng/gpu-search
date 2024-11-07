# https://github.com/deepmind/alphafold

```console
README.md:genetic databases (SSD storage is recommended) and a modern NVIDIA GPU (GPUs
README.md:        [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
README.md:        for GPU support.
README.md:1.  Check that AlphaFold will be able to use a GPU by running:
README.md:    docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
README.md:    The output of this command should show a list of your GPUs. If it doesn't,
README.md:    [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
README.md:    [NVIDIA Docker issue](https://github.com/NVIDIA/nvidia-docker/issues/1447#issuecomment-801479573).
README.md:    W: GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
README.md:    E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 InRelease' is not signed.
README.md:was tested on Google Cloud with a machine using the `nvidia-gpu-cloud-image`
README.md:3 TB disk, and an A100 GPU. For your first run, please follow the instructions
README.md:1.  By default, Alphafold will attempt to use all visible GPU devices. To use a
README.md:    subset, specify a comma-separated list of GPU UUID(s) or index(es) using the
README.md:    `--gpu_devices` flag. See
README.md:    [GPU enumeration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration)
README.md:1.  The relaxation step can be run on GPU (faster, but could be less stable) or
README.md:    CPU (slow, but stable). This can be controlled with `--enable_gpu_relax=true`
README.md:    (default) or `--enable_gpu_relax=false`.
README.md:`timings.json`. All runtimes are from a single A100 NVIDIA GPU. Prediction
run_alphafold.py:                     'deterministic, because processes like GPU inference are '
run_alphafold.py:flags.DEFINE_boolean('use_gpu_relax', None, 'Whether to relax on GPU. '
run_alphafold.py:                     'Relax on GPU can be much faster than CPU, so it is '
run_alphafold.py:                     'recommended to enable if possible. GPUs must be available'
run_alphafold.py:      use_gpu=FLAGS.use_gpu_relax)
run_alphafold.py:      'use_gpu_relax',
docker/Dockerfile:ARG CUDA=12.2.2
docker/Dockerfile:FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu20.04
docker/Dockerfile:ARG CUDA
docker/Dockerfile:        cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
docker/Dockerfile:    && conda install -y -c nvidia cuda=${CUDA_VERSION} \
docker/Dockerfile:      jaxlib==0.4.26+cuda12.cudnn89 \
docker/Dockerfile:      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
docker/Dockerfile:# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
docker/Dockerfile:# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
docker/run_docker.py:    'use_gpu', True, 'Enable NVIDIA runtime to run with GPUs.')
docker/run_docker.py:    'enable_gpu_relax', True, 'Run relax on GPU if GPU is enabled.')
docker/run_docker.py:    'gpu_devices', 'all',
docker/run_docker.py:    'Comma separated list of devices to pass to NVIDIA_VISIBLE_DEVICES.')
docker/run_docker.py:  use_gpu_relax = FLAGS.enable_gpu_relax and FLAGS.use_gpu
docker/run_docker.py:      f'--use_gpu_relax={use_gpu_relax}',
docker/run_docker.py:      docker.types.DeviceRequest(driver='nvidia', capabilities=[['gpu']])
docker/run_docker.py:  ] if FLAGS.use_gpu else None
docker/run_docker.py:          'NVIDIA_VISIBLE_DEVICES': FLAGS.gpu_devices,
docker/run_docker.py:          # would typically be too long to fit into GPU memory.
alphafold/model/r3.py:these can end up on specialized cores such as tensor cores on GPU or the MXU on
alphafold/model/r3.py:unintended use of these cores on both GPUs and TPUs.
alphafold/model/quat_affine.py:  the GPU. If at all possible, this function should run on the CPU.
alphafold/model/geometry/vector.py:  In most cases this will also be faster on cpu's/gpu's since it allows for
alphafold/notebooks/notebook_utils.py:        f'GPU memory).')
alphafold/relax/relax_test.py:        'use_gpu': False}
alphafold/relax/relax.py:               use_gpu: bool):
alphafold/relax/relax.py:      use_gpu: Whether to run on GPU.
alphafold/relax/relax.py:    self._use_gpu = use_gpu
alphafold/relax/relax.py:        use_gpu=self._use_gpu)
alphafold/relax/amber_minimize_test.py:_USE_GPU = False
alphafold/relax/amber_minimize_test.py:                                      stiffness=10., use_gpu=_USE_GPU)
alphafold/relax/amber_minimize_test.py:                                  use_gpu=_USE_GPU)
alphafold/relax/amber_minimize_test.py:        prot=prot, max_outer_iterations=10, stiffness=10., use_gpu=_USE_GPU)
alphafold/relax/amber_minimize.py:    use_gpu: bool):
alphafold/relax/amber_minimize.py:  platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")
alphafold/relax/amber_minimize.py:    use_gpu: bool,
alphafold/relax/amber_minimize.py:    use_gpu: Whether to run on GPU.
alphafold/relax/amber_minimize.py:          use_gpu=use_gpu)
alphafold/relax/amber_minimize.py:    use_gpu: bool,
alphafold/relax/amber_minimize.py:    use_gpu: Whether to run on GPU.
alphafold/relax/amber_minimize.py:        use_gpu=use_gpu)
alphafold/relax/amber_minimize.py:    # Calculation of violations can cause CUDA errors for some JAX versions.

```
