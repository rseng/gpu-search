# https://github.com/google/caliban

```console
cloudbuild.json:      "id": "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda100",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda100",
cloudbuild.json:        "dockerfiles/Dockerfile.gpu",
cloudbuild.json:        "CUDA=10.0",
cloudbuild.json:      "id": "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda100-push",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda100"
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda100"
cloudbuild.json:      "id": "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda101",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda101",
cloudbuild.json:        "dockerfiles/Dockerfile.gpu",
cloudbuild.json:        "CUDA=10.1",
cloudbuild.json:      "id": "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda101-push",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda101"
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda101"
cloudbuild.json:      "id": "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py37-cuda100",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py37-cuda100",
cloudbuild.json:        "BASE_IMAGE=gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda100",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda100"
cloudbuild.json:      "id": "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py37-cuda100-push",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py37-cuda100"
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py37-cuda100"
cloudbuild.json:      "id": "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py37-cuda101",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py37-cuda101",
cloudbuild.json:        "BASE_IMAGE=gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda101",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda101"
cloudbuild.json:      "id": "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py37-cuda101-push",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py37-cuda101"
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py37-cuda101"
cloudbuild.json:      "id": "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py38-cuda100",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py38-cuda100",
cloudbuild.json:        "BASE_IMAGE=gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda100",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda100"
cloudbuild.json:      "id": "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py38-cuda100-push",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py38-cuda100"
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py38-cuda100"
cloudbuild.json:      "id": "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py38-cuda101",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py38-cuda101",
cloudbuild.json:        "BASE_IMAGE=gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda101",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-base-ubuntu1804-cuda101"
cloudbuild.json:      "id": "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py38-cuda101-push",
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py38-cuda101"
cloudbuild.json:        "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py38-cuda101"
dockerfiles/Dockerfile:# docker build --build-arg BASE_IMAGE=tensorflow/tensorflow:2.1.0-gpu-py3 -t gcr.io/blueshift-playground/blueshift:gpu -f- . <Dockerfile
dockerfiles/Dockerfile:# docker push gcr.io/blueshift-playground/blueshift:gpu
dockerfiles/Dockerfile.gpu:ARG CUDA=10.1
dockerfiles/Dockerfile.gpu:FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
dockerfiles/Dockerfile.gpu:# ARCH and CUDA are specified again because the FROM directive resets ARGs
dockerfiles/Dockerfile.gpu:ARG CUDA
dockerfiles/Dockerfile.gpu:# These dependencies come from the list at the official Tensorflow GPU base
dockerfiles/Dockerfile.gpu:# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile
dockerfiles/Dockerfile.gpu:  cuda-command-line-tools-${CUDA/./-} \
dockerfiles/Dockerfile.gpu:  cuda-nvrtc-${CUDA/./-} \
dockerfiles/Dockerfile.gpu:  cuda-cufft-${CUDA/./-} \
dockerfiles/Dockerfile.gpu:  cuda-curand-${CUDA/./-} \
dockerfiles/Dockerfile.gpu:  cuda-cusolver-${CUDA/./-} \
dockerfiles/Dockerfile.gpu:  cuda-cusparse-${CUDA/./-} \
dockerfiles/Dockerfile.gpu:  libcudnn7=${CUDNN}+cuda${CUDA} \
dockerfiles/Dockerfile.gpu:  libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
dockerfiles/Dockerfile.gpu:  libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
dockerfiles/Dockerfile.gpu:# For CUDA profiling, TensorFlow requires CUPTI.
dockerfiles/Dockerfile.gpu:ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
dockerfiles/Dockerfile.gpu:# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
dockerfiles/Dockerfile.gpu:RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
dockerfiles/Dockerfile.gpu:  && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
docs/getting_started/prerequisites.rst:To use Caliban, you'll need a working Docker installation. If you have a GPU and
docs/getting_started/prerequisites.rst:want to run jobs that use it, you'll have to install ``nvidia-docker2``, as
docs/getting_started/prerequisites.rst:described below in :ref:`GPU Support on Linux Machines`
docs/getting_started/prerequisites.rst:  only be able to run in CPU mode, as MacOS doesn't support Docker's nvidia
docs/getting_started/prerequisites.rst:  runtime. You will, however, be able to build GPU containers and submit them to
docs/getting_started/prerequisites.rst:GPU Support on Linux Machines
docs/getting_started/prerequisites.rst:On Linux, Caliban can run jobs locally that take advantage of a GPU you may have installed.
docs/getting_started/prerequisites.rst:To use this feature, install the ``nvidia-docker2`` runtime by following the
docs/getting_started/prerequisites.rst:instructions at the `nvidia-docker2
docs/getting_started/prerequisites.rst:<https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)>`_
docs/getting_started/prerequisites.rst:.. NOTE:: It's important that you install ``nvidia-docker2``, not
docs/getting_started/prerequisites.rst:          ``nvidia-docker``! The `nvidia-docker2
docs/getting_started/prerequisites.rst:          <https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)>`_
docs/getting_started/prerequisites.rst:          ``nvidia-docker``.
docs/getting_started/prerequisites.rst:.. NOTE:: The most recent versions of docker don't need the ``nvidia-docker2``
docs/cli/caliban_build.rst:   usage: caliban build [-h] [--helpfull] [--nogpu] [--cloud_key CLOUD_KEY]
docs/cli/caliban_build.rst:     --nogpu               Disable GPU mode and force CPU-only.
docs/cli/caliban_cluster.rst:cpu/memory/gpu configuration is fixed. For simple configurations where you
docs/cli/caliban_cluster.rst:so you can specify your gpu- and machine- types on a per-job basis, and the
docs/cli/caliban_cluster.rst:                                                    'resourceType': 'nvidia-tesla-k80'},
docs/cli/caliban_cluster.rst:                                                    'resourceType': 'nvidia-tesla-p100'},
docs/cli/caliban_cluster.rst:                                                    'resourceType': 'nvidia-tesla-v100'},
docs/cli/caliban_cluster.rst:                                                    'resourceType': 'nvidia-tesla-p4'},
docs/cli/caliban_cluster.rst:                                                    'resourceType': 'nvidia-tesla-t4'}]},
docs/cli/caliban_cluster.rst:to your cluster to automatically apply nvidia drivers to any gpu-enabled nodes
docs/cli/caliban_cluster.rst:`here <https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers>`_.
docs/cli/caliban_cluster.rst:                                  [--cluster_name CLUSTER_NAME] [--nogpu]
docs/cli/caliban_cluster.rst:                                  [--gpu_spec NUMxGPU_TYPE]
docs/cli/caliban_cluster.rst:     --nogpu               Disable GPU mode and force CPU-only. (default: True)
docs/cli/caliban_cluster.rst:                           gpu/tpu jobs, and 31000 for cpu jobs. Please note that
docs/cli/caliban_cluster.rst:     --gpu_spec NUMxGPU_TYPE
docs/cli/caliban_cluster.rst:                           Type and number of GPUs to use for each AI
docs/cli/caliban_cluster.rst:                           Platform/GKE submission. Defaults to 1xP100 in GPU
docs/cli/caliban_cluster.rst:                           mode or None if --nogpu is passed. (default: None)
docs/cli/caliban_shell.rst:   usage: caliban shell [-h] [--helpfull] [--nogpu] [--cloud_key CLOUD_KEY]
docs/cli/caliban_shell.rst:     --nogpu               Disable GPU mode and force CPU-only.
docs/cli/caliban_shell.rst:on how to use ``extras_require`` to create separate environments for GPU and
docs/cli/caliban_shell.rst:On the Mac you'll have to pass ``--nogpu`` to ``shell``\ , as the NVIDIA runtime isn't
docs/cli/caliban_shell.rst:    to run a CUDA binary to communicate with the GPU, ``caliban shell`` called a
docs/cli/caliban_resubmit.rst:   caliban cloud --xgroup resubmit_test --nogpu --experiment_config experiment.json cpu.py -- --foo 3
docs/cli/caliban_run.rst:   usage: caliban run [-h] [--helpfull] [--nogpu] [--cloud_key CLOUD_KEY]
docs/cli/caliban_run.rst:     --nogpu               Disable GPU mode and force CPU-only.
docs/cli/caliban_run.rst:on how to use ``extras_require`` to create separate environments for GPU and
docs/cli/caliban_run.rst:Jobs run in GPU mode by default. To toggle GPU mode off, use ``--nogpu``.
docs/cli/caliban_notebook.rst:   usage: caliban notebook [-h] [--helpfull] [--nogpu] [--cloud_key CLOUD_KEY]
docs/cli/caliban_notebook.rst:     --nogpu               Disable GPU mode and force CPU-only.
docs/cli/caliban_notebook.rst:on how to use ``extras_require`` to create separate environments for GPU and
docs/cli/caliban_notebook.rst:On the Mac you'll have to pass ``--nogpu`` to ``notebook``\ , as the NVIDIA runtime
docs/cli/caliban_cloud.rst:   usage: caliban cloud [-h] [--helpfull] [--nogpu] [--cloud_key CLOUD_KEY]
docs/cli/caliban_cloud.rst:                        [--gpu_spec NUMxGPU_TYPE] [--tpu_spec NUMxTPU_TYPE]
docs/cli/caliban_cloud.rst:     --nogpu               Disable GPU mode and force CPU-only.
docs/cli/caliban_cloud.rst:                           Defaults to 'n1-standard-8' in GPU mode, or
docs/cli/caliban_cloud.rst:                           'n1-highcpu-32' if --nogpu is passed.
docs/cli/caliban_cloud.rst:     --gpu_spec NUMxGPU_TYPE
docs/cli/caliban_cloud.rst:                           Type and number of GPUs to use for each AI Platform
docs/cli/caliban_cloud.rst:                           submission. Defaults to 1xP100 in GPU mode or None if
docs/cli/caliban_cloud.rst:                           --nogpu is passed.
docs/cli/caliban_cloud.rst:#. Multi-GPU machines, clusters and TPUs are available
docs/cli/caliban_cloud.rst:  :doc:`../cloud/gpu_specs` for more detail.
docs/cli/caliban_cloud.rst:* **gpu_spec**\ : optional argument of the form GPU_COUNTxGPU_TYPE. See
docs/cli/caliban_cloud.rst:  ``caliban cloud --help`` for all possible GPU types, and for the default.
docs/cli/caliban_cloud.rst:  on AI Platform. See :doc:`../cloud/gpu_specs` for more details.
docs/cli/caliban_cloud.rst:  GPUs specified using ``--gpu_spec``. See :doc:`../cloud/ai_platform_tpu` for
docs/cli/caliban_cloud.rst:  combinations of region, machine type, GPU count and GPU type and force
docs/cli/caliban_cloud.rst:  case some new GPU was added to a region or machine type and caliban hasn't
docs/cloud/labels.rst:* **gpu_enabled**\ : ``true`` by default, or ``false`` if you ran your job with
docs/cloud/labels.rst:  ``--nogpu``
docs/cloud/gpu_specs.rst:Customizing Machines and GPUs
docs/cloud/gpu_specs.rst:instructions on how to request different GPUs or machine types for your job.
docs/cloud/gpu_specs.rst:Default GPU and Machine Types
docs/cloud/gpu_specs.rst:By default, if you don't supply ``--gpu_spec`` or ``--machine_type`` (both discussed
docs/cloud/gpu_specs.rst:* GPU mode (default): a single P100 GPU on an ``n1-standard-8`` machine
docs/cloud/gpu_specs.rst:* CPU mode: an ``n1-highcpu-32`` machine with no GPU attached
docs/cloud/gpu_specs.rst:Custom GPU Specs
docs/cloud/gpu_specs.rst:The optional ``--gpu_spec`` argument allows you to attach a custom number and type
docs/cloud/gpu_specs.rst:of GPU to the Cloud node that will run your containerized job on AI Platform.
docs/cloud/gpu_specs.rst:The required format is ``GPU_COUNTxGPU_TYPE``\ , as in this example:
docs/cloud/gpu_specs.rst:   caliban cloud --gpu_spec 2xV100 trainer.train
docs/cloud/gpu_specs.rst:This will submit your job to a node configured with 2 V100 GPUs to a machine in
docs/cloud/gpu_specs.rst:that the combination of GPU count, region, GPU type and machine type are
docs/cloud/gpu_specs.rst:for 3 V100 GPUs:
docs/cloud/gpu_specs.rst:   caliban cloud --gpu_spec 3xV100 trainer.train
docs/cloud/gpu_specs.rst:   caliban cloud: error: argument --gpu_spec: 3 GPUs of type V100 aren't available
docs/cloud/gpu_specs.rst:   For more help, consult this page for valid combinations of GPU count, GPU type
docs/cloud/gpu_specs.rst:   and machine type: https://cloud.google.com/ml-engine/docs/using-gpus
docs/cloud/gpu_specs.rst:   caliban cloud --gpu_spec 2xV100 --machine_type n1-standard-96 trainer.train
docs/cloud/gpu_specs.rst:   'n1-standard-96' isn't a valid machine type for 2 V100 GPUs.
docs/cloud/gpu_specs.rst:   For more help, consult this page for valid combinations of GPU count, GPU type
docs/cloud/gpu_specs.rst:   and machine type: https://cloud.google.com/ml-engine/docs/using-gpus
docs/cloud/gpu_specs.rst:``n1-highcpu-96`` instance with 8 V100 GPUs attached:
docs/cloud/gpu_specs.rst:   caliban cloud --gpu_spec 8xV100 --machine_type n1-highcpu-96 trainer.train
docs/cloud/gpu_specs.rst:As described above in :ref:`Custom GPU Specs`, ``--machine_type`` works with
docs/cloud/gpu_specs.rst:``--gpu_spec`` to validate that the combination of GPU count, GPU type and
docs/cloud/ai_platform_tpu.rst:``--tpu_spec`` is compatible with ``--gpu_spec``\ ; the latter configures the master
docs/cloud/ai_platform_tpu.rst:Normally, all jobs default to GPU mode unless you supply ``--nogpu`` explicitly.
docs/cloud/ai_platform_tpu.rst:This default flips when you supply a ``--tpu_spec`` and no explicit ``--gpu_spec``.
docs/cloud/ai_platform_tpu.rst:In that case, ``caliban cloud`` will NOT attach a default GPU to your master
docs/index.rst:  and executes it locally using ``docker run``. If you have a workstation GPU,
docs/index.rst:  the instance will attach to it by default - no need to install the CUDA
docs/index.rst:  hundreds of jobs at once. Any machine type, GPU count, and GPU type
docs/index.rst:and submit GPU jobs to Cloud from your Mac!)
docs/index.rst:   recipes/single_gpu
docs/index.rst:   cloud/gpu_specs
docs/index.rst:JIT to GPU/TPU, and more.
docs/gke/job_submission.rst:created when we started the cluster. We will submit a job that uses gpu
docs/gke/job_submission.rst:for our job based on the gpu and machine specs we provide in the job submission.
docs/gke/job_submission.rst:this example we can get at most eight K80 gpus, and at most four T4 gpus.
docs/gke/job_submission.rst:   totoro@totoro:$ caliban cluster job submit --gpu_spec 1xK80 --name cifar10-test cifar10_resnet_train.sh --
docs/gke/job_submission.rst:    'job_mode': <JobMode.GPU: 2>,
docs/gke/job_submission.rst:   Step 1/15 : FROM gcr.io/blueshift-playground/blueshift:gpu
docs/gke/concepts.rst:is a collection of identical nodes (cpu, memory, gpu, tpu).
docs/gke/cluster_management.rst:   I0204 09:28:05.274888 139910209476416 cluster.py:1092] applying nvidia driver daemonset...
docs/recipes/single_gpu.rst:Using a Single GPU
docs/recipes/single_gpu.rst:By default, ``docker run`` will make all GPUs on your workstation available
docs/recipes/single_gpu.rst:* your huge GPU, custom-built and installed for ML Supremacy
docs/recipes/single_gpu.rst:* the dinky GPU that exists solely to power your monitor, NOT to help train
docs/recipes/single_gpu.rst:The second GPU will slow down everything.
docs/recipes/single_gpu.rst:To stop this from happening you need to set the ``CUDA_VISIBLE_DEVICES``
docs/recipes/single_gpu.rst:`nvidia blog <https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/>`_
docs/recipes/single_gpu.rst:   caliban run --docker_run_args "--env CUDA_VISIBLE_DEVICES=0" trainer.train
docs/recipes/single_gpu.rst:   ``CUDA_VISIBLE_DEVICES`` set. ``caliban shell`` and ``caliban notebook``
docs/recipes/single_gpu.rst:You can directly limit the GPUs that mount into the container using the ``--gpus``
docs/recipes/single_gpu.rst:   caliban run --docker_run_args "--gpus device=0" trainer.train
docs/recipes/single_gpu.rst:If you run ``nvidia-smi`` in the container after passing this argument you won't
docs/recipes/single_gpu.rst:see more than 1 GPU. This is useful if you know that some library you're using
docs/recipes/single_gpu.rst:doesn't respect the ``CUDA_VISIBLE_DEVICES`` environment variable for any reason.
docs/recipes/single_gpu.rst:   CUDA_VISIBLE_DEVICES=0
docs/recipes/local_dir.rst:   $ caliban shell --docker_run_args "--volume /usr/local/google/home/totoro/data:/foo" --nogpu --bare
docs/recipes/flagfile.rst:   --docker_run_args "CUDA_VISIBLE_DEVICES=0"
docs/recipes/flagfile.rst:   caliban run --docker_run_args "CUDA_VISIBLE_DEVICES=0" \
docs/recipes/flagfile.rst:   # Definition for big iron GPUs.
docs/recipes/flagfile.rst:   --gpu_spec 8xV100
docs/recipes/flagfile.rst:And then some further file called ``tpu_plus_gpu.flags``\ :
docs/recipes/flagfile.rst:   caliban cloud --flagfile tpu_plus_gpu.flags trainer.train
docs/explore/calibanconfig.rst:dictionary with ``"gpu"`` and ``"cpu"'`` keys. For example, any of the following are
docs/explore/calibanconfig.rst:   # for gpu and cpu modes. It's fine to leave either of these blank,
docs/explore/calibanconfig.rst:           "gpu": ["libsm6", "libxext6", "libxrender-dev"],
docs/explore/calibanconfig.rst:       "base_image": "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py38-cuda101"
docs/explore/calibanconfig.rst:You can also specify different base images for ``cpu`` and ``gpu`` modes as follows:
docs/explore/calibanconfig.rst:           "gpu": "gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py38-cuda101"
docs/explore/declaring_requirements.rst:GPU), in addition to your own custom dependency sets.
docs/explore/declaring_requirements.rst:This solves the problem of depending on, say, ``tensorflow-gpu`` for a GPU job,
docs/explore/declaring_requirements.rst:           'gpu': ['tensorflow-gpu==2.0.*'],
docs/explore/declaring_requirements.rst:   pip install .[gpu]
docs/explore/declaring_requirements.rst:* AND, additionally, the entries under the ``'gpu'`` key of the ``extras_require``
docs/explore/declaring_requirements.rst:latter and attempt to install a ``'gpu'`` set of extras, like
docs/explore/declaring_requirements.rst:   pip install .[gpu]
docs/explore/declaring_requirements.rst:If you pass ``--nogpu`` to any of the commands, Caliban will similarly attempt to
docs/explore/declaring_requirements.rst:in addition to the ``cpu`` or ``gpu`` dependency set.
docs/explore/base_image.rst:python and cuda versions. You can find our base images
docs/explore/base_image.rst:For example, ``gcr.io/blueshift-playground/blueshift:gpu-ubuntu1804-py38-cuda101`` is a
docs/explore/base_image.rst:gpu base image that uses Ubuntu 18.04, CUDA 10.1 and python 3.8, while
docs/explore/base_image.rst:base image that has no CUDA support and uses python 3.8.
docs/explore/base_image.rst:|   no cuda    |    yes     |    yes     |
docs/explore/base_image.rst:|   cuda 10.0  |    yes     |    yes     |
docs/explore/base_image.rst:|   cuda 10.1  |    yes     |    yes     |
docs/explore/base_image.rst:|   no cuda    |    yes     |    yes     |
docs/explore/base_image.rst:|   cuda 10.0  |    no      |     no     |
docs/explore/base_image.rst:|   cuda 10.1  |    no      |     no     |
docs/explore/base_image.rst:base gpu images from the `Dockerfile.gpu <https://github.com/google/caliban/blob/main/dockerfiles/Dockerfile.gpu>`_
docs/explore/base_image.rst:file, and then use these as base images for creating full GPU images with
docs/explore/base_image.rst:We base our gpu base images on the `nvidia/cuda <https://hub.docker.com/r/nvidia/cuda/>`_
docs/explore/base_image.rst:images, which contain the relevant CUDA drivers required for GPU use. The virtual
docs/explore/base_image.rst:In the configuration file, we specify our supported CUDA versions, our supported
docs/explore/base_image.rst:For our CUDA and python versions, we specify a list of build-args that we then
docs/explore/experiment_broadcasting.rst:Experiments and Custom Machine + GPUs
docs/explore/experiment_broadcasting.rst:If you supply a ``--gpu_spec`` or ``--machine_type`` in addition to
docs/explore/mac.rst:If you're developing on your Macbook, you'll be able to build GPU containers,
docs/explore/mac.rst:but you won't be able to run them locally. You can still submit GPU jobs to AI
docs/explore/mac.rst:``--nogpu`` as a keyword argument. If you don't do this you'll see the following
docs/explore/mac.rst:   'caliban run' doesn't support GPU usage on Macs! Please pass --nogpu to use this command.
docs/explore/mac.rst:   (GPU mode is fine for 'caliban cloud' from a Mac; just nothing that runs locally.)
docs/explore/why_caliban.rst:one for GPU mode. One for CPU mode locally. Slight tweaks show up every time you
docs/explore/why_caliban.rst:* Local execution on your workstation on GPU
docs/explore/custom_docker_run.rst:container, or limit which GPUs are mounted into the container.
tutorials/uv-metrics/README.md:caliban run --nogpu hello_world.sh
tutorials/uv-metrics/README.md:caliban run --nogpu trainer.train
tutorials/uv-metrics/README.md:caliban run --experiment_config experiment.json --xgroup mlflow_tutorial --nogpu trainer.train
CHANGELOG.md:docker pull gcr.io/blueshift-playground/blueshift:gpu
CHANGELOG.md:  - a dict of the form `{"cpu": "base_image", "gpu": "base_image"}` with both
CHANGELOG.md:  block `{}` will be filled in with either `cpu` or `gpu`, depending on the mode
CHANGELOG.md:dlvm:pytorch-gpu
CHANGELOG.md:dlvm:pytorch-gpu-1.0
CHANGELOG.md:dlvm:pytorch-gpu-1.1
CHANGELOG.md:dlvm:pytorch-gpu-1.2
CHANGELOG.md:dlvm:pytorch-gpu-1.3
CHANGELOG.md:dlvm:pytorch-gpu-1.4
CHANGELOG.md:dlvm:tf-gpu
CHANGELOG.md:dlvm:tf-gpu-1.0
CHANGELOG.md:dlvm:tf-gpu-1.13
CHANGELOG.md:dlvm:tf-gpu-1.14
CHANGELOG.md:dlvm:tf-gpu-1.15
CHANGELOG.md:dlvm:tf2-gpu
CHANGELOG.md:dlvm:tf2-gpu-2.0
CHANGELOG.md:dlvm:tf2-gpu-2.1
CHANGELOG.md:dlvm:tf2-gpu-2.2
CHANGELOG.md:- Caliban now uses a slimmer base image for GPU mode.
CHANGELOG.md:- Base images for CPU and GPU modes now use Conda to manage the container's
CHANGELOG.md:  `"gpu"` and `"cpu"'` keys. For example, any of the following are valid:
CHANGELOG.md:# for gpu and cpu modes. It's fine to leave either of these blank,
CHANGELOG.md:        "gpu": ["libsm6", "libxext6", "libxrender-dev"],
CHANGELOG.md:- consolidated gke tpu/gpu spec parsing with cloud types
CHANGELOG.md:- moved cluster gpu validation to separate file
CHANGELOG.md:- added test for gpu limits validation
CHANGELOG.md:- TPU and GPU spec now accept validate_count arg to disable count validation.
CHANGELOG.md:  cat experiment.json | caliban cloud -e gpu --experiment_config stdin --dry_run trainer.train
CHANGELOG.md:- if you supply a `--tpu_spec` and DON'T supply an explicit `--gpu_spec`,
CHANGELOG.md:  caliban will default to CPU mode. `--gpu_spec` and `--nogpu` are still
CHANGELOG.md:  incompatible. You can use a GPU and TPU spec together without problems.
CHANGELOG.md:  similar style to `--gpu_spec`; any invalid combination of count, region and
CHANGELOG.md:  (Unlike `--gpu_spec` this mode IS compatible with `--nogpu`. In fact, many
CHANGELOG.md:  demos seem to use the non-GPU version of tensorflow here.)
CHANGELOG.md:caliban cloud trainer.train --nogpu
CHANGELOG.md:  That final `--nogpu` would get passed on directly to your script, vs getting
CHANGELOG.md:- If you pass `--nogpu` mode and have a setup.py file, caliban will
CHANGELOG.md:  called `cpu`. Same goes for `gpu` if you DON'T pass `--nogpu`. So, if you have
CHANGELOG.md:pip install .[gpu]
CHANGELOG.md:  - `--gpu_spec`, which you can use to configure the GPU count and type for your
tests/caliban/config/test_config.py:  assert c.JobMode.parse("  GpU") == c.JobMode.GPU
tests/caliban/config/test_config.py:  assert c.JobMode.parse("gpu") == c.JobMode.GPU
tests/caliban/config/test_config.py:  assert c.JobMode.parse("GPU") == c.JobMode.GPU
tests/caliban/config/test_config.py:def test_gpu():
tests/caliban/config/test_config.py:  assert c.gpu(c.JobMode.GPU)
tests/caliban/config/test_config.py:  assert not c.gpu(c.JobMode.CPU)
tests/caliban/config/test_config.py:  assert not c.gpu("face")
tests/caliban/config/test_config.py:  assert c.base_image({}, c.JobMode.GPU) is None
tests/caliban/config/test_config.py:  assert c.base_image(dlvm, c.JobMode.GPU) == c.DLVM_CONFIG["dlvm:pytorch-gpu-1.4"]
tests/caliban/config/test_config.py:    {"base_image": {"cpu": "dlvm:tf2-{}-2.1", "gpu": "random:latest"}}
tests/caliban/config/test_config.py:  assert c.base_image(conf, c.JobMode.GPU) == "random:latest"
tests/caliban/config/test_config.py:  assert c.apt_packages(config, c.JobMode.GPU) == []
tests/caliban/config/test_config.py:  gpu = c.apt_packages(valid_shared_conf, c.JobMode.GPU)
tests/caliban/config/test_config.py:  assert cpu == gpu
tests/caliban/platform/cloud/test_types.py:    st.integers(min_value=0, max_value=40), st.sampled_from(list(ct.GPU) + list(ct.TPU))
tests/caliban/platform/cloud/test_types.py:  def test_gpuspec_parse_arg(self):
tests/caliban/platform/cloud/test_types.py:      ct.GPUSpec.parse_arg("face")
tests/caliban/platform/cloud/test_types.py:      ct.GPUSpec.parse_arg("randomxV100")
tests/caliban/platform/cloud/test_types.py:      # invalid GPU type.
tests/caliban/platform/cloud/test_types.py:      ct.GPUSpec.parse_arg("8xNONSTANDARD")
tests/caliban/platform/cloud/test_types.py:      # Invalid number for the valid GPU type.
tests/caliban/platform/cloud/test_types.py:      ct.GPUSpec.parse_arg("15xV100")
tests/caliban/platform/cloud/test_types.py:      ct.GPUSpec(ct.GPU.V100, 7), ct.GPUSpec.parse_arg("7xV100", validate_count=False)
tests/caliban/platform/cloud/test_types.py:    self.assertEqual(ct.GPUSpec(ct.GPU.V100, 8), ct.GPUSpec.parse_arg("8xV100"))
tests/caliban/platform/gke/test_util.py:      st.integers(min_value=0, max_value=4), min_size=len(ct.GPU), max_size=len(ct.GPU)
tests/caliban/platform/gke/test_util.py:    st.sampled_from(ct.GPU),
tests/caliban/platform/gke/test_util.py:  def test_validate_gpu_spec_against_limits(
tests/caliban/platform/gke/test_util.py:    gpu_type: ct.GPU,
tests/caliban/platform/gke/test_util.py:    """tests gpu validation against limits"""
tests/caliban/platform/gke/test_util.py:    gpu_list = [g for g in ct.GPU]
tests/caliban/platform/gke/test_util.py:    gpu_limits = dict(
tests/caliban/platform/gke/test_util.py:      [(gpu_list[i], limits[i]) for i in range(len(limits)) if limits[i]]
tests/caliban/platform/gke/test_util.py:    spec = ct.GPUSpec(gpu_type, count)
tests/caliban/platform/gke/test_util.py:    valid = util.validate_gpu_spec_against_limits(spec, gpu_limits, "test")
tests/caliban/platform/gke/test_util.py:    if spec.gpu not in gpu_limits:
tests/caliban/platform/gke/test_util.py:      self.assertTrue(valid == (spec.count <= gpu_limits[spec.gpu]))
tests/caliban/platform/gke/test_util.py:  def test_validate_gpu_spec_against_limits_deterministic(self):
tests/caliban/platform/gke/test_util.py:    # gpu not supported
tests/caliban/platform/gke/test_util.py:      "gpu_spec": ct.GPUSpec(ct.GPU.K80, 1),
tests/caliban/platform/gke/test_util.py:      "gpu_limits": {ct.GPU.P100: 1},
tests/caliban/platform/gke/test_util.py:    assert not util.validate_gpu_spec_against_limits(**cfg)
tests/caliban/platform/gke/test_util.py:      "gpu_spec": ct.GPUSpec(ct.GPU.K80, 2),
tests/caliban/platform/gke/test_util.py:      "gpu_limits": {
tests/caliban/platform/gke/test_util.py:        ct.GPU.P100: 1,
tests/caliban/platform/gke/test_util.py:        ct.GPU.K80: 1,
tests/caliban/platform/gke/test_util.py:    assert not util.validate_gpu_spec_against_limits(**cfg)
tests/caliban/platform/gke/test_util.py:      "gpu_spec": ct.GPUSpec(ct.GPU.K80, 1),
tests/caliban/platform/gke/test_util.py:      "gpu_limits": {
tests/caliban/platform/gke/test_util.py:        ct.GPU.P100: 1,
tests/caliban/platform/gke/test_util.py:        ct.GPU.K80: 1,
tests/caliban/platform/gke/test_util.py:    assert util.validate_gpu_spec_against_limits(**cfg)
tests/caliban/platform/gke/test_util.py:  def test_nvidia_daemonset_url(self):
tests/caliban/platform/gke/test_util.py:    """tests nvidia driver daemonset url generation"""
tests/caliban/platform/gke/test_util.py:      url = util.nvidia_daemonset_url(n)
tests/caliban/platform/gke/test_util.py:      st.integers(min_value=0, max_value=32), min_size=len(ct.GPU), max_size=len(ct.GPU)
tests/caliban/platform/gke/test_util.py:  def test_get_zone_gpu_types(self, gpu_counts, invalid_types):
tests/caliban/platform/gke/test_util.py:    """tests get_zone_gpu_types"""
tests/caliban/platform/gke/test_util.py:    gpu_types = ["nvidia-tesla-{}".format(x.name.lower()) for x in ct.GPU]
tests/caliban/platform/gke/test_util.py:    gpus = [
tests/caliban/platform/gke/test_util.py:      {"name": gpu_types[i], "maximumCardsPerInstance": c}
tests/caliban/platform/gke/test_util.py:      for i, c in enumerate(gpu_counts)
tests/caliban/platform/gke/test_util.py:      return {"items": gpus + invalid}
tests/caliban/platform/gke/test_util.py:    self.assertIsNone(util.get_zone_gpu_types(api, "p", "z"))
tests/caliban/platform/gke/test_util.py:    self.assertIsNone(util.get_zone_gpu_types(api, "p", "z"))
tests/caliban/platform/gke/test_util.py:      sorted(["{}-{}".format(x["name"], x["maximumCardsPerInstance"]) for x in gpus]),
tests/caliban/platform/gke/test_util.py:          "nvidia-tesla-{}-{}".format(x.gpu.name.lower(), x.count)
tests/caliban/platform/gke/test_util.py:          for x in util.get_zone_gpu_types(api, "p", "z")
tests/caliban/platform/gke/test_util.py:          {"limit": 1024, "metric": "NVIDIA_K80_GPUS", "usage": 0},
tests/caliban/platform/gke/test_util.py:          {"limit": 1024, "metric": "NVIDIA_K80_GPUS", "usage": 0},
tests/caliban/platform/gke/test_util.py:      + [{"resourceType": "nvidia-tesla-k80", "maximum": str(quotas[1]["limit"])}]
tests/caliban/platform/gke/test_util.py:  counts = {"cpu": 1, "nvidia-tesla-p100": 2, "memory": k.MAX_GB_PER_CPU}
tests/caliban/platform/gke/test_util.py:    ("NVIDIA_P100_GPUS", counts["nvidia-tesla-p100"]),
tests/caliban/platform/gke/test_util.py:  # valid, gpu quota == 0
tests/caliban/platform/gke/test_util.py:  counts = {"cpu": 1, "nvidia-tesla-p100": 0, "memory": k.MAX_GB_PER_CPU}
tests/caliban/platform/gke/test_util.py:  quotas = [("CPUS", counts["cpu"]), ("NVIDIA_P100_GPUS", counts["nvidia-tesla-p100"])]
tests/caliban/platform/gke/test_util.py:    assert d["resourceType"] != "nvidia-tesla-p100"
tests/caliban/history/test_history.py:    "nogpu": True,
tests/caliban/history/test_history.py:    "nogpu": True,
tests/caliban/history/test_history.py:    "nogpu": True,
tests/caliban/history/test_history.py:    "nogpu": True,
tests/caliban/test_cli.py:    gpu_spec = ct.GPUSpec(ct.GPU.P100, 4)
tests/caliban/test_cli.py:    def assertMode(expected_mode, use_gpu, gpu_spec, tpu_spec):
tests/caliban/test_cli.py:      mode = c._job_mode(use_gpu, gpu_spec, tpu_spec)
tests/caliban/test_cli.py:    # --nogpu and no override.
tests/caliban/test_cli.py:    # TPU doesn't need GPUs
tests/caliban/test_cli.py:    # Default GPUSpec filled in.
tests/caliban/test_cli.py:    assertMode(JobMode.GPU, True, None, None)
tests/caliban/test_cli.py:    # Explicit GPU spec, so GPU gets attached.
tests/caliban/test_cli.py:    assertMode(JobMode.GPU, True, gpu_spec, None)
tests/caliban/test_cli.py:    assertMode(JobMode.GPU, True, gpu_spec, tpu_spec)
tests/caliban/test_cli.py:    # If NO explicit GPU is supplied but a TPU is supplied, execute in CPU
tests/caliban/test_cli.py:    # mode, ie, don't attach a GPU.
tests/caliban/test_cli.py:    # explicit GPU spec is incompatible with --nogpu in both of the following
tests/caliban/test_cli.py:      c._job_mode(False, gpu_spec, None)
tests/caliban/test_cli.py:      c._job_mode(False, gpu_spec, tpu_spec)
README.md:caliban run --nogpu mnist.py
README.md:echo '{"learning_rate": [0.01, 0.001, 0.0001]}' | caliban run --experiment_config stdin --nogpu mnist.py
README.md:### Cloud Submission and GPUs
README.md:- [Installing the `nvidia-docker2`
README.md:  runtime](https://caliban.readthedocs.io/en/latest/getting_started/prerequisites.html#docker-and-cuda),
README.md:  so you can use Caliban to run jobs that use your Linux machine's GPU.
README.md:caliban run --nogpu mnist.py
README.md:$ caliban run --nogpu mnist.py -- --learning_rate 0.01
README.md:caliban run --experiment_config experiment.json --nogpu mnist.py
README.md:caliban cloud --nogpu mnist.py -- --learning_rate 0.01
README.md:I0615 19:57:43.355440 4563361216 core.py:161] Job 1 - labels: {'gpu_enabled': 'false', 'tpu_enabled': 'false', 'job_name': 'caliban_totoro', 'learning_rate': '0_01'}
README.md:[GPUs](https://caliban.readthedocs.io/en/latest/cloud/gpu_specs.html) and
README.md:type](https://caliban.readthedocs.io/en/latest/cloud/gpu_specs.html#custom-machine-types)
README.md:GPUs"](https://caliban.readthedocs.io/en/latest/cloud/gpu_specs.html#) for more
README.md:caliban shell --nogpu
README.md:will execute in a Cloud environment, with potentially many GPUs attached and
README.md:  using `docker run`. If you have a GPU, the instance will attach to it by
README.md:  default - no need to install the CUDA toolkit. The Docker environment takes
README.md:  can submit hundreds of jobs at once. Any machine type, GPU count, and GPU type
paper/paper.md:`docker run`. If the local machine has access to a GPU, the instance will attach
paper/paper.md:GPU count, and GPU type combination specified will be validated client side,
caliban/config/__init__.py:  GPU = "GPU"
caliban/config/__init__.py:  JobMode.GPU: ct.MachineType.standard_8,
caliban/config/__init__.py:DEFAULT_GPU = ct.GPU.P100
caliban/config/__init__.py:  **_dlvm_config(JobMode.GPU),
caliban/config/__init__.py:  {s.Optional("gpu", default=list): [str], s.Optional("cpu", default=list): [str]},
caliban/config/__init__.py:  {s.Optional("gpu", default=None): Image, s.Optional("cpu", default=None): Image},
caliban/config/__init__.py:  error=""""base_image" entry must be a string OR dict with 'cpu' and 'gpu' keys, not '{}'""",
caliban/config/__init__.py:def gpu(job_mode: JobMode) -> bool:
caliban/config/__init__.py:  """Returns True if the supplied JobMode is JobMode.GPU, False otherwise."""
caliban/config/__init__.py:  return job_mode == JobMode.GPU
caliban/config/__init__.py:    k = "gpu" if gpu(mode) else "cpu"
caliban/config/__init__.py:  this method will fill it in with the current mode (cpu or gpu).
caliban/platform/run.py:  to execute `docker run`. in CPU or GPU mode, depending on the value of
caliban/platform/run.py:  runtime = ["--runtime", "nvidia"] if c.gpu(job_mode) else []
caliban/platform/run.py:      um.GPU_ENABLED_TAG: str(job_mode == c.JobMode.GPU).lower(),
caliban/platform/cloud/types.py:# Machine types allowed in CPU or GPU modes.
caliban/platform/cloud/types.py:# Various GPU types currently available on Cloud, mapped to their cloud
caliban/platform/cloud/types.py:GPU = Enum(
caliban/platform/cloud/types.py:  "GPU",
caliban/platform/cloud/types.py:    "K80": "NVIDIA_TESLA_K80",
caliban/platform/cloud/types.py:    "P4": "NVIDIA_TESLA_P4",
caliban/platform/cloud/types.py:    "P100": "NVIDIA_TESLA_P100",
caliban/platform/cloud/types.py:    "T4": "NVIDIA_TESLA_T4",
caliban/platform/cloud/types.py:    "V100": "NVIDIA_TESLA_V100",
caliban/platform/cloud/types.py:    "A100": "NVIDIA_TESLA_A100",
caliban/platform/cloud/types.py:Accelerator = Union[GPU, TPU]
caliban/platform/cloud/types.py:  GPU.K80: [
caliban/platform/cloud/types.py:  GPU.P4: [
caliban/platform/cloud/types.py:  GPU.P100: [
caliban/platform/cloud/types.py:  GPU.T4: [
caliban/platform/cloud/types.py:  GPU.V100: [
caliban/platform/cloud/types.py:# From this page: https://cloud.google.com/ml-engine/docs/using-gpus
caliban/platform/cloud/types.py:    GPU.K80: {1, 2, 4, 8},
caliban/platform/cloud/types.py:    GPU.P4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.P100: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.T4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.V100: {1, 2, 4, 8},
caliban/platform/cloud/types.py:    GPU.A100: {0},
caliban/platform/cloud/types.py:    GPU.K80: {1, 2, 4, 8},
caliban/platform/cloud/types.py:    GPU.P4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.P100: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.T4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.V100: {1, 2, 4, 8},
caliban/platform/cloud/types.py:    GPU.K80: {2, 4, 8},
caliban/platform/cloud/types.py:    GPU.P4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.P100: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.T4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.V100: {2, 4, 8},
caliban/platform/cloud/types.py:    GPU.K80: {4, 8},
caliban/platform/cloud/types.py:    GPU.P4: {2, 4},
caliban/platform/cloud/types.py:    GPU.P100: {2, 4},
caliban/platform/cloud/types.py:    GPU.T4: {2, 4},
caliban/platform/cloud/types.py:    GPU.V100: {4, 8},
caliban/platform/cloud/types.py:  MachineType.standard_64: {GPU.P4: {4}, GPU.T4: {4}, GPU.V100: {8}},
caliban/platform/cloud/types.py:  MachineType.standard_96: {GPU.P4: {4}, GPU.T4: {4}, GPU.V100: {8}},
caliban/platform/cloud/types.py:    GPU.K80: {1, 2, 4, 8},
caliban/platform/cloud/types.py:    GPU.P4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.P100: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.T4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.V100: {1, 2, 4, 8},
caliban/platform/cloud/types.py:    GPU.K80: {1, 2, 4, 8},
caliban/platform/cloud/types.py:    GPU.P4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.P100: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.T4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.V100: {1, 2, 4, 8},
caliban/platform/cloud/types.py:    GPU.K80: {1, 2, 4, 8},
caliban/platform/cloud/types.py:    GPU.P4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.P100: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.T4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.V100: {1, 2, 4, 8},
caliban/platform/cloud/types.py:    GPU.K80: {2, 4, 8},
caliban/platform/cloud/types.py:    GPU.P4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.P100: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.T4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.V100: {2, 4, 8},
caliban/platform/cloud/types.py:    GPU.K80: {4, 8},
caliban/platform/cloud/types.py:    GPU.P4: {2, 4},
caliban/platform/cloud/types.py:    GPU.P100: {2, 4},
caliban/platform/cloud/types.py:    GPU.T4: {2, 4},
caliban/platform/cloud/types.py:    GPU.V100: {4, 8},
caliban/platform/cloud/types.py:  MachineType.highmem_64: {GPU.P4: {4}, GPU.T4: {4}, GPU.V100: {8}},
caliban/platform/cloud/types.py:  MachineType.highmem_96: {GPU.P4: {4}, GPU.T4: {4}, GPU.V100: {8}},
caliban/platform/cloud/types.py:    GPU.K80: {2, 4, 8},
caliban/platform/cloud/types.py:    GPU.P4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.P100: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.T4: {1, 2, 4},
caliban/platform/cloud/types.py:    GPU.V100: {2, 4, 8},
caliban/platform/cloud/types.py:    GPU.K80: {4, 8},
caliban/platform/cloud/types.py:    GPU.P4: {2, 4},
caliban/platform/cloud/types.py:    GPU.P100: {2, 4},
caliban/platform/cloud/types.py:    GPU.T4: {2, 4},
caliban/platform/cloud/types.py:    GPU.V100: {4, 8},
caliban/platform/cloud/types.py:    GPU.K80: {8},
caliban/platform/cloud/types.py:    GPU.P4: {4},
caliban/platform/cloud/types.py:    GPU.P100: {4},
caliban/platform/cloud/types.py:    GPU.T4: {4},
caliban/platform/cloud/types.py:    GPU.V100: {8},
caliban/platform/cloud/types.py:  MachineType.highcpu_96: {GPU.P4: {4}, GPU.T4: {4}, GPU.V100: {8}},
caliban/platform/cloud/types.py:def accelerator_name(is_gpu: bool) -> str:
caliban/platform/cloud/types.py:  return "GPU" if is_gpu else "TPU"
caliban/platform/cloud/types.py:  can be 'gpu' or 'tpu', or any cased version of those) and a message string.
caliban/platform/cloud/types.py:  Returns a string with a suffix appended that links to the GPU or CPU
caliban/platform/cloud/types.py:    is_gpu = accel.upper() == "GPU"
caliban/platform/cloud/types.py:    is_gpu = accel in GPU
caliban/platform/cloud/types.py:  ucase = accelerator_name(is_gpu)
caliban/platform/cloud/types.py:  if is_gpu:
caliban/platform/cloud/types.py:    url = "https://cloud.google.com/ml-engine/docs/using-gpus"
caliban/platform/cloud/types.py:  is_gpu = accel in GPU
caliban/platform/cloud/types.py:  ucase = accelerator_name(is_gpu)
caliban/platform/cloud/types.py:  assert mode in ("GPU", "TPU"), "Mode must be GPU or TPU."
caliban/platform/cloud/types.py:  # Validate that we have a valid GPU type.
caliban/platform/cloud/types.py:    accel_dict = GPU if mode == "GPU" else TPU
caliban/platform/cloud/types.py:class GPUSpec(NamedTuple("GPUSpec", [("gpu", GPU), ("count", int)])):
caliban/platform/cloud/types.py:  """Info to generate a GPU."""
caliban/platform/cloud/types.py:  METAVAR = "NUMxGPU_TYPE"
caliban/platform/cloud/types.py:  def parse_arg(s: str, **kwargs) -> "GPUSpec":
caliban/platform/cloud/types.py:    """Parses a CLI string of the form COUNTxGPUType into a proper GPU spec
caliban/platform/cloud/types.py:    gpu, count = parse_accelerator_arg(s, "GPU", GPUSpec._error_suffix, **kwargs)
caliban/platform/cloud/types.py:    return GPUSpec(gpu, count)
caliban/platform/cloud/types.py:    return self.gpu.name
caliban/platform/cloud/types.py:    return {"type": self.gpu.value, "count": self.count}
caliban/platform/cloud/types.py:    return _AccelCountMT[self.gpu].get(self.count, {})
caliban/platform/cloud/types.py:    """Set of all regions allowed for this particular GPU type."""
caliban/platform/cloud/types.py:    return set(ACCELERATOR_REGION_SUPPORT[self.gpu])
caliban/platform/cloud/types.py:    """Parses a CLI string of the form COUNTxGPUType into a proper GPU spec
caliban/platform/cloud/core.py:def get_accelerator_config(gpu_spec: Optional[ct.GPUSpec]) -> Dict[str, Any]:
caliban/platform/cloud/core.py:  """Returns the accelerator config for the supplied GPUSpec if present; else,
caliban/platform/cloud/core.py:  if gpu_spec is not None:
caliban/platform/cloud/core.py:    config = gpu_spec.accelerator_config()
caliban/platform/cloud/core.py:  gpu_spec: Optional[ct.GPUSpec],
caliban/platform/cloud/core.py:  labels such as 'gpu_enabled', etc are filled in for each job.
caliban/platform/cloud/core.py:  accelerator_conf = get_accelerator_config(gpu_spec)
caliban/platform/cloud/core.py:  gpu_enabled = gpu_spec is not None
caliban/platform/cloud/core.py:    "gpu_enabled": str(gpu_enabled).lower(),
caliban/platform/cloud/core.py:  gpu_spec: Optional[ct.GPUSpec] = None,
caliban/platform/cloud/core.py:  - builds an image using the supplied docker_args, in either CPU or GPU mode
caliban/platform/cloud/core.py:    support different GPUs.
caliban/platform/cloud/core.py:  - gpu_spec: if None and job_mode is GPU, defaults to a standard single GPU.
caliban/platform/cloud/core.py:    Else, configures the count and type of GPUs to attach to the machine that
caliban/platform/cloud/core.py:  if job_mode == conf.JobMode.GPU and gpu_spec is None:
caliban/platform/cloud/core.py:    gpu_spec = ct.GPUSpec(ct.GPU.P100, 1)
caliban/platform/cloud/core.py:      gpu_spec=gpu_spec,
caliban/platform/gke/cli.py:  gpu_spec = args.get("gpu_spec")
caliban/platform/gke/cli.py:  # validatate gpu spec
caliban/platform/gke/cli.py:  if job_mode == conf.JobMode.GPU and gpu_spec is None:
caliban/platform/gke/cli.py:    gpu_spec = k.DEFAULT_GPU_SPEC
caliban/platform/gke/cli.py:  if not cluster.validate_gpu_spec(gpu_spec):
caliban/platform/gke/cli.py:  if tpu_spec is None and gpu_spec is None:  # cpu-only job
caliban/platform/gke/cli.py:  else:  # gpu/tpu-accelerated job
caliban/platform/gke/cli.py:  accel_spec = Cluster.convert_accel_spec(gpu_spec, tpu_spec)
caliban/platform/gke/cli.py:    labels[um.GPU_ENABLED_TAG] = str(job_mode == conf.JobMode.GPU).lower()
caliban/platform/gke/cluster.py:  GPU,
caliban/platform/gke/cluster.py:  GPUSpec,
caliban/platform/gke/cluster.py:    None for cpu, limits dictionary for gpu/tpu
caliban/platform/gke/cluster.py:    if type(accelerator) == GPU:
caliban/platform/gke/cluster.py:      return {k.CONTAINER_RESOURCE_LIMIT_GPU: count}
caliban/platform/gke/cluster.py:    # see: https://cloud.google.com/kubernetes-engine/docs/how-to/gpus
caliban/platform/gke/cluster.py:    if isinstance(accelerator, GPU):
caliban/platform/gke/cluster.py:    machine_type: machine type, None=default for mode (cpu/gpu)
caliban/platform/gke/cluster.py:    # tpu/gpu resources
caliban/platform/gke/cluster.py:    machine_type: machine type, None=default for mode (cpu/gpu)
caliban/platform/gke/cluster.py:    gpu_spec: Optional[GPUSpec], tpu_spec: Optional[TPUSpec]
caliban/platform/gke/cluster.py:    """converts gpu/tpu spec pair to accelerator,count tuple
caliban/platform/gke/cluster.py:    gpu_spec: gpu spec
caliban/platform/gke/cluster.py:    if gpu_spec is not None and tpu_spec is not None:
caliban/platform/gke/cluster.py:      logging.error("error: cannot specify both tpu and gpu")
caliban/platform/gke/cluster.py:    # gpu
caliban/platform/gke/cluster.py:    if gpu_spec is not None:
caliban/platform/gke/cluster.py:      return (gpu_spec.gpu, gpu_spec.count)
caliban/platform/gke/cluster.py:  def get_gpu_types(self) -> Optional[List[GPUSpec]]:
caliban/platform/gke/cluster.py:    """gets supported gpu types for cluster
caliban/platform/gke/cluster.py:    list of supported gpu types on success, None otherwise
caliban/platform/gke/cluster.py:    # for now we just return the gpu resource limits from the autoprovisioning
caliban/platform/gke/cluster.py:    gpu_re = re.compile("^nvidia-tesla-(?P<type>[a-z0-9]+)$")
caliban/platform/gke/cluster.py:    gpus = []
caliban/platform/gke/cluster.py:      match = gpu_re.match(x["resourceType"])
caliban/platform/gke/cluster.py:      gpus.append(GPUSpec(GPU[gd["type"].upper()], int(x["maximum"])))
caliban/platform/gke/cluster.py:    return gpus
caliban/platform/gke/cluster.py:  def validate_gpu_spec(self, gpu_spec: Optional[GPUSpec]) -> bool:
caliban/platform/gke/cluster.py:    """validates gpu spec against zone and cluster contraints
caliban/platform/gke/cluster.py:    gpu_spec: gpu spec
caliban/platform/gke/cluster.py:    if gpu_spec is None:
caliban/platform/gke/cluster.py:    zone_gpus = util.get_zone_gpu_types(compute_api, self.project_id, self.zone)
caliban/platform/gke/cluster.py:    if zone_gpus is None:
caliban/platform/gke/cluster.py:    gpu_limits = dict([(x.gpu, x.count) for x in zone_gpus])
caliban/platform/gke/cluster.py:    if not util.validate_gpu_spec_against_limits(gpu_spec, gpu_limits, "zone"):
caliban/platform/gke/cluster.py:    available_gpu = self.get_gpu_types()
caliban/platform/gke/cluster.py:    if available_gpu is None:
caliban/platform/gke/cluster.py:    gpu_limits = dict([(x.gpu, x.count) for x in available_gpu])
caliban/platform/gke/cluster.py:    if not util.validate_gpu_spec_against_limits(gpu_spec, gpu_limits, "cluster"):
caliban/platform/gke/cluster.py:    daemonset_url = util.nvidia_daemonset_url(NodeImage.COS)
caliban/platform/gke/cluster.py:      logging.error("nvidia-driver daemonset not applied, to do this manually:")
caliban/platform/gke/cluster.py:    logging.info("applying nvidia driver daemonset...")
caliban/platform/gke/util.py:from caliban.platform.cloud.types import GPU, TPU, GPUSpec, TPUSpec
caliban/platform/gke/util.py:def validate_gpu_spec_against_limits(
caliban/platform/gke/util.py:  gpu_spec: GPUSpec,
caliban/platform/gke/util.py:  gpu_limits: Dict[GPU, int],
caliban/platform/gke/util.py:  """validate gpu spec against provided limits
caliban/platform/gke/util.py:  gpu_spec: gpu spec
caliban/platform/gke/util.py:  gpu_limits: limits
caliban/platform/gke/util.py:  if gpu_spec.gpu not in gpu_limits:
caliban/platform/gke/util.py:      "unsupported gpu type {}. ".format(gpu_spec.gpu.name)
caliban/platform/gke/util.py:      + "Supported types for {}: {}".format(limit_type, [g.name for g in gpu_limits])
caliban/platform/gke/util.py:  if gpu_spec.count > gpu_limits[gpu_spec.gpu]:
caliban/platform/gke/util.py:      "error: requested {} gpu count {} unsupported,".format(
caliban/platform/gke/util.py:        gpu_spec.gpu.name, gpu_spec.count
caliban/platform/gke/util.py:      + " {} max = {}".format(limit_type, gpu_limits[gpu_spec.gpu])
caliban/platform/gke/util.py:def nvidia_daemonset_url(node_image: NodeImage) -> Optional[str]:
caliban/platform/gke/util.py:  """gets nvidia driver daemonset url for given node image
caliban/platform/gke/util.py:    NodeImage.COS: k.NVIDIA_DRIVER_COS_DAEMONSET_URL,
caliban/platform/gke/util.py:    NodeImage.UBUNTU: k.NVIDIA_DRIVER_UBUNTU_DAEMONSET_URL,
caliban/platform/gke/util.py:def gke_gpu_to_gpu(gpu: str) -> Optional[GPU]:
caliban/platform/gke/util.py:  """convert gke gpu string to GPU type
caliban/platform/gke/util.py:  gpu: gke gpu string
caliban/platform/gke/util.py:  GPU on success, None otherwise
caliban/platform/gke/util.py:  gpu_re = re.compile("^nvidia-tesla-(?P<type>[a-z0-9]+)$")
caliban/platform/gke/util.py:  match = gpu_re.match(gpu)
caliban/platform/gke/util.py:  return GPU[gd["type"].upper()]
caliban/platform/gke/util.py:def get_zone_gpu_types(
caliban/platform/gke/util.py:) -> Optional[List[GPUSpec]]:
caliban/platform/gke/util.py:  """gets list of gpu accelerators available in given zone
caliban/platform/gke/util.py:  list of GPUSpec on success (count is max count), None otherwise
caliban/platform/gke/util.py:  gpus = []
caliban/platform/gke/util.py:    gpu = gke_gpu_to_gpu(x["name"])
caliban/platform/gke/util.py:    if gpu is None:
caliban/platform/gke/util.py:    gpus.append(GPUSpec(gpu, int(x["maximumCardsPerInstance"])))
caliban/platform/gke/util.py:  return gpus
caliban/platform/gke/util.py:  These quotas include cpu and gpu quotas for the given region.
caliban/platform/gke/util.py:  gpu_re = re.compile("^NVIDIA_(?P<gpu>[A-Z0-9]+)_GPUS$")
caliban/platform/gke/util.py:    gpu_match = gpu_re.match(metric)
caliban/platform/gke/util.py:    if gpu_match is None:
caliban/platform/gke/util.py:    gd = gpu_match.groupdict()
caliban/platform/gke/util.py:    gpu_type = gd["gpu"]
caliban/platform/gke/util.py:        "resourceType": "nvidia-tesla-{}".format(gpu_type.lower()),
caliban/platform/gke/constants.py:from caliban.platform.cloud.types import GPU, GPUSpec
caliban/platform/gke/constants.py:CONTAINER_RESOURCE_LIMIT_GPU = "nvidia.com/gpu"
caliban/platform/gke/constants.py:DEFAULT_MACHINE_TYPE_GPU = DEFAULT_MACHINE_TYPE[JobMode.GPU].value
caliban/platform/gke/constants.py:DEFAULT_GPU_SPEC = GPUSpec(GPU.P100, 1)
caliban/platform/gke/constants.py:# default min_cpu for gpu/tpu -accelerated jobs (in milli-cpu)
caliban/platform/gke/constants.py:# default min_mem for gpu/tpu jobs (in MB)
caliban/platform/gke/constants.py:# nvidia drivers to auto-created gpu instances. If this is not running, then your
caliban/platform/gke/constants.py:# gpu jobs will mysteriously fail to schedule, and you will be sad.
caliban/platform/gke/constants.py:# see https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers
caliban/platform/gke/constants.py:NVIDIA_DRIVER_COS_DAEMONSET_URL = "https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml"
caliban/platform/gke/constants.py:NVIDIA_DRIVER_UBUNTU_DAEMONSET_URL = "https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/ubuntu/daemonset-preloaded.yaml"
caliban/util/metrics.py:GPU_ENABLED_TAG = "gpu_enabled"
caliban/history/cli.py:    f'{cs.id}: job_mode: {cs.spec.get("job_mode", "GPU")}, '
caliban/cli.py:  use_gpu: bool, gpu_spec: Optional[ct.GPUSpec], tpu_spec: Optional[ct.TPUSpec]
caliban/cli.py:  if not use_gpu and gpu_spec is not None:
caliban/cli.py:    raise AssertionError("gpu_spec isn't allowed for CPU only jobs!")
caliban/cli.py:  mode = conf.JobMode.GPU if use_gpu else conf.JobMode.CPU
caliban/cli.py:  # For the specific case where there's no GPU specified and a TPU is, set the
caliban/cli.py:  # mode back to CPU and don't attach a GPU.
caliban/cli.py:  if gpu_spec is None and tpu_spec is not None:
caliban/cli.py:  use_gpu = args.get("use_gpu", True)
caliban/cli.py:  gpu_spec = args.get("gpu_spec")
caliban/cli.py:  return _job_mode(use_gpu, gpu_spec, tpu_spec)
caliban/cli.py:def no_gpu_flag(parser):
caliban/cli.py:    "--nogpu",
caliban/cli.py:    dest="use_gpu",
caliban/cli.py:    help="Disable GPU mode and force CPU-only.",
caliban/cli.py:  gpu_default = conf.DEFAULT_MACHINE_TYPE[conf.JobMode.GPU].value
caliban/cli.py:    + "{}. Defaults to '{}' in GPU mode, or '{}' ".format(
caliban/cli.py:      machine_types, gpu_default, cpu_default
caliban/cli.py:    + "if --nogpu is passed.",
caliban/cli.py:  no_gpu_flag(base)
caliban/cli.py:def gpu_spec_arg(parser, validate_count: bool = False):
caliban/cli.py:    "--gpu_spec",
caliban/cli.py:    metavar=ct.GPUSpec.METAVAR,
caliban/cli.py:    type=lambda x: ct.GPUSpec.parse_arg(x, validate_count=validate_count),
caliban/cli.py:    help="Type and number of GPUs to use for each AI Platform/GKE "
caliban/cli.py:    + "submission.  Defaults to 1x{} in GPU mode ".format(conf.DEFAULT_GPU.name)
caliban/cli.py:    + "or None if --nogpu is passed.",
caliban/cli.py:  gpu_spec_arg(parser)
caliban/cli.py:def mac_gpu_check(job_mode: conf.JobMode, command: str) -> None:
caliban/cli.py:  if conf.gpu(job_mode) and command in ("shell", "notebook", "run"):
caliban/cli.py:      "\n'caliban {}' doesn't support GPU usage on Macs! Please pass ".format(command)
caliban/cli.py:      + "--nogpu to use this command.\n\n"
caliban/cli.py:      "(GPU mode is fine for 'caliban cloud' from a Mac; just nothing that runs "
caliban/cli.py:def _validate_no_gpu_type(use_gpu: bool, gpu_spec: Optional[ct.GPUSpec]):
caliban/cli.py:  explicitly attempted to set a GPU spec.
caliban/cli.py:  gpu_disabled = not use_gpu
caliban/cli.py:  if gpu_disabled and gpu_spec is not None:
caliban/cli.py:      "\n'--nogpu' is incompatible with an explicit --gpu_spec option. "
caliban/cli.py:  gpu_spec: Optional[ct.GPUSpec], machine_type: Optional[ct.MachineType]
caliban/cli.py:  combination of GPU count, type and machine type.
caliban/cli.py:  if gpu_spec is not None and machine_type is not None:
caliban/cli.py:    if not gpu_spec.valid_machine_type(machine_type):
caliban/cli.py:      allowed = u.enum_vals(gpu_spec.allowed_machine_types())
caliban/cli.py:        + f"for {gpu_spec.count} {gpu_spec.gpu.name} GPUs.\n\n"
caliban/cli.py:      u.err(ct.with_advice_suffix("gpu", f"Try one of these: {allowed}"))
caliban/cli.py:  spec: Optional[Union[ct.GPUSpec, ct.TPUSpec]], region: ct.Region
caliban/cli.py:    mac_gpu_check(job_mode, command)
caliban/cli.py:    use_gpu = m.get("use_gpu")
caliban/cli.py:    gpu_spec = args.gpu_spec
caliban/cli.py:    _validate_no_gpu_type(use_gpu, gpu_spec)
caliban/cli.py:    # A TPU is valid with or without an attached GPU.
caliban/cli.py:    if use_gpu:
caliban/cli.py:      _validate_machine_type(gpu_spec, args.machine_type)
caliban/cli.py:      _validate_accelerator_region(gpu_spec, region)
caliban/cli.py:  no_gpu_flag(parser)
caliban/cli.py:  gpu_spec_arg(parser, validate_count=False)
caliban/cli.py:    "this value defaults to {} for gpu/tpu jobs, and {} for cpu jobs. Please "
caliban/docker/build.py:DEFAULT_GPU_TAG = "gpu-ubuntu1804-py37-cuda101"
caliban/docker/build.py:  GPU. This is JUST for building our base images for Blueshift; not for
caliban/docker/build.py:  gpu = "-gpu" if c.gpu(job_mode) else ""
caliban/docker/build.py:  return "tensorflow/tensorflow:{}{}-py3".format(tensorflow_version, gpu)
caliban/docker/build.py:  return DEFAULT_GPU_TAG if c.gpu(job_mode) else DEFAULT_CPU_TAG
caliban/docker/build.py:  If the path DOES exist, generates a list of extras to install. gpu or cpu are
caliban/docker/build.py:    extra = "gpu" if c.gpu(job_mode) else "cpu"
caliban/docker/build.py:  """Returns a Dockerfile that builds on a local CPU or GPU base image (depending
caliban/main.py:    gpu_spec = args.get("gpu_spec")
caliban/main.py:      gpu_spec=gpu_spec,
scripts/build_dockerfiles.sh:# GPU base-base, with all CUDA dependencies required for GPU work. Built off of the NVIDIA base image.
scripts/build_dockerfiles.sh:docker build -t gcr.io/blueshift-playground/blueshift:gpu-base -f- . <dockerfiles/Dockerfile.gpu
scripts/build_dockerfiles.sh:docker push gcr.io/blueshift-playground/blueshift:gpu-base
scripts/build_dockerfiles.sh:# GPU image.
scripts/build_dockerfiles.sh:docker build --build-arg BASE_IMAGE=gcr.io/blueshift-playground/blueshift:gpu-base -t gcr.io/blueshift-playground/blueshift:gpu -f- . <dockerfiles/Dockerfile
scripts/build_dockerfiles.sh:docker push gcr.io/blueshift-playground/blueshift:gpu
scripts/cloudbuild.py:_GPU_BASE_TAG = "base"
scripts/cloudbuild.py:_GPU_TAG = "gpu"
scripts/cloudbuild.py:_GPU_DOCKERFILE = "dockerfiles/Dockerfile.gpu"
scripts/cloudbuild.py:  GPU_BASE = "GPU_BASE"
scripts/cloudbuild.py:  GPU = "GPU"
scripts/cloudbuild.py:class GpuBase(NamedTuple):
scripts/cloudbuild.py:  gpu: str
scripts/cloudbuild.py:    return f"{self.base_image}-{self.gpu}"
scripts/cloudbuild.py:  gpu_version: Optional[str] = None
scripts/cloudbuild.py:    if self.gpu_version is None:
scripts/cloudbuild.py:        return ImageType.GPU_BASE
scripts/cloudbuild.py:        return ImageType.GPU
scripts/cloudbuild.py:      parts.append(_GPU_TAG)
scripts/cloudbuild.py:      if self.image_type == ImageType.GPU_BASE:
scripts/cloudbuild.py:        parts += ["base", self.base_image, self.gpu_version]
scripts/cloudbuild.py:        parts += [self.base_image, self.cpu_version, self.gpu_version]
scripts/cloudbuild.py:def _get_unique_gpu_bases(images: List[Config]) -> Set[GpuBase]:
scripts/cloudbuild.py:  gpu_base_images = set()
scripts/cloudbuild.py:    if x.get("gpu") is None:
scripts/cloudbuild.py:    gpu_base_images.add(GpuBase(base_image=x["base_image"], gpu=x["gpu"]))
scripts/cloudbuild.py:  return gpu_base_images
scripts/cloudbuild.py:def _create_gpu_base_image_specs(
scripts/cloudbuild.py:  gpu_cfg: Config,
scripts/cloudbuild.py:  for b in _get_unique_gpu_bases(images=images):
scripts/cloudbuild.py:    build_args = copy(gpu_cfg[b.gpu])
scripts/cloudbuild.py:      gpu_version=b.gpu,
scripts/cloudbuild.py:      dockerfile=_GPU_DOCKERFILE,
scripts/cloudbuild.py:    if x.get("gpu") is not None:
scripts/cloudbuild.py:      gpu_version=None,
scripts/cloudbuild.py:  gpu_specs: Dict[str, ImageSpec],
scripts/cloudbuild.py:  gpu_version = cfg.get("gpu")
scripts/cloudbuild.py:  if gpu_version is None:
scripts/cloudbuild.py:  gpu_base = GpuBase(base_image=cfg["base_image"], gpu=cfg["gpu"])
scripts/cloudbuild.py:  gpu_spec = gpu_specs[gpu_base.tag]
scripts/cloudbuild.py:  build_args["BASE_IMAGE"] = gpu_spec.tag
scripts/cloudbuild.py:    gpu_version=gpu_version,
scripts/cloudbuild.py:  gpu_cfg = cfg.get("gpu_versions", {})
scripts/cloudbuild.py:  gpu_specs = _create_gpu_base_image_specs(
scripts/cloudbuild.py:    gpu_cfg=gpu_cfg,
scripts/cloudbuild.py:  specs = list(gpu_specs.values())  # gpu base images
scripts/cloudbuild.py:    _create_image_spec(cfg=c, cpu_specs=cpu_specs, gpu_specs=gpu_specs) for c in images
scripts/cloudbuild.py:  if spec.image_type == ImageType.GPU:
scripts/cloudbuild_config.json:  "gpu_versions" : {
scripts/cloudbuild_config.json:    "cuda100" : {"CUDA":"10.0"},
scripts/cloudbuild_config.json:    "cuda101" : {"CUDA":"10.1"},
scripts/cloudbuild_config.json:    "cuda110" : {"CUDA":"11.0"}
scripts/cloudbuild_config.json:    {"base_image": "ubuntu1804", "python" : "py37", "gpu" : "cuda100"},
scripts/cloudbuild_config.json:    {"base_image": "ubuntu1804", "python" : "py37", "gpu" : "cuda101"},
scripts/cloudbuild_config.json:    {"base_image": "ubuntu1804", "python" : "py38", "gpu" : "cuda100"},
scripts/cloudbuild_config.json:    {"base_image": "ubuntu1804", "python" : "py38", "gpu" : "cuda101"},

```
