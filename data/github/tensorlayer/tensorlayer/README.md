# https://github.com/tensorlayer/TensorLayer

```console
setup.py:    'tf_gpu': req_file("requirements_tf_gpu.txt"),
setup.py:extras_require['all_gpu'] = sum([extras_require.get(key) for key in ['all', 'tf_gpu']], list())
setup.py:extras_require['all_gpu_dev'] = sum([extras_require.get(key) for key in ['all_dev', 'tf_gpu']], list())
README.rst:TensorLayer has pre-requisites including TensorFlow, numpy, and others. For GPU support, CUDA and cuDNN are required.
README.rst:Containers with GPU support
README.rst:NVIDIA-Docker is required for these containers to work: `Project
README.rst:Link <https://github.com/NVIDIA/nvidia-docker>`__
README.rst:    # for GPU version and Python 2
README.rst:    docker pull tensorlayer/tensorlayer:latest-gpu
README.rst:    nvidia-docker run -it --rm -p 8888:88888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu
README.rst:    # for GPU version and Python 3
README.rst:    docker pull tensorlayer/tensorlayer:latest-gpu-py3
README.rst:    nvidia-docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu-py3
.pyup.yml:    - requirements/requirements_tf_gpu.txt
docs/user/get_involved.rst:SurgicalAI is a startup founded by the data scientists and surgical robot experts from Imperial College. Our objective is AI democratise Surgery. By combining 5G, AI and Cloud Computing, SurgicalAI is building a platform enable junor surgeons to perfom complex procedures. As one of the most impactful startup, SurgicalAI is supported by Nvidia, AWS and top surgeons around the world.
docs/user/get_start_model.rst:  #                   'training_device': 'gpu'}}
docs/user/installation.rst:`TensorFlow`_ , numpy and matplotlib. For GPU
docs/user/installation.rst:support CUDA and cuDNN are required.
docs/user/installation.rst:  pip3 install tensorflow-gpu==2.0.0-beta1 # specific version  (YOU SHOULD INSTALL THIS ONE NOW)
docs/user/installation.rst:  pip3 install tensorflow-gpu # GPU version
docs/user/installation.rst:However, there are something need to be considered. For example, `TensorFlow`_ officially supports GPU acceleration for Linux, Mac OX and Windows at present. For ARM processor architecture, you need to install TensorFlow from source.
docs/user/installation.rst:  # for a machine **without** an NVIDIA GPU
docs/user/installation.rst:  # for a machine **with** an NVIDIA GPU
docs/user/installation.rst:  pip3 install -e ".[all_gpu_dev]"
docs/user/installation.rst:GPU support
docs/user/installation.rst:Thanks to NVIDIA supports, training a fully connected network on a
docs/user/installation.rst:GPU, which may be 10 to 20 times faster than training them on a CPU.
docs/user/installation.rst:This requires an NVIDIA GPU with CUDA and cuDNN support.
docs/user/installation.rst:CUDA
docs/user/installation.rst:The TensorFlow website also teach how to install the CUDA and cuDNN, please see
docs/user/installation.rst:`TensorFlow GPU Support <https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux>`_.
docs/user/installation.rst:Download and install the latest CUDA is available from NVIDIA website:
docs/user/installation.rst: - `CUDA download and install <https://developer.nvidia.com/cuda-downloads>`_
docs/user/installation.rst:  After installation, make sure ``/usr/local/cuda/bin`` is in your ``PATH`` (use ``echo #PATH`` to check), and
docs/user/installation.rst:  ``nvcc --version`` works. Also ensure ``/usr/local/cuda/lib64`` is in your
docs/user/installation.rst:  ``LD_LIBRARY_PATH``, so the CUDA libraries can be found.
docs/user/installation.rst:If CUDA is set up correctly, the following command should print some GPU information on
docs/user/installation.rst:Apart from CUDA, NVIDIA also provides a library for common neural network operations that especially
docs/user/installation.rst:NVIDIA after registering as a developer (it take a while):
docs/user/installation.rst:Download and install the latest cuDNN is available from NVIDIA website:
docs/user/installation.rst: - `cuDNN download and install <https://developer.nvidia.com/cudnn>`_
docs/user/installation.rst:To install it, copy the ``*.h`` files to ``/usr/local/cuda/include`` and the
docs/user/installation.rst:``lib*`` files to ``/usr/local/cuda/lib64``.
docs/user/installation.rst:GPU support
docs/user/installation.rst:Thanks to NVIDIA supports, training a fully connected network on a GPU, which may be 10 to 20 times faster than training them on a CPU. For convolutional network, may have 50 times faster. This requires an NVIDIA GPU with CUDA and cuDNN support.
docs/user/installation.rst:You should preinstall Microsoft Visual Studio (VS) before installing CUDA. The lowest version requirements is VS2010. We recommend installing VS2015 or VS2013. CUDA7.5 supports VS2010, VS2012 and VS2013. CUDA8.0 also supports VS2015.
docs/user/installation.rst:2. Installing CUDA
docs/user/installation.rst:Download and install the latest CUDA is available from NVIDIA website:
docs/user/installation.rst:`CUDA download <https://developer.nvidia.com/CUDA-downloads>`_
docs/user/installation.rst:The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. Download and extract the latest cuDNN is available from NVIDIA website:
docs/user/installation.rst:`cuDNN download <https://developer.nvidia.com/cuDNN>`_
docs/user/installation.rst:After extracting cuDNN, you will get three folders (bin, lib, include). Then these folders should be copied to CUDA installation. (The default installation directory is `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0`)
docs/user/installation.rst:  pip3 install tensorflow-gpu    #GPU version (GPU version and CPU version just choose one)
docs/index.rst:The TensorLayer user guide explains how to install TensorFlow, CUDA and cuDNN,
docs/modules/utils.rst:   set_gpu_fraction
docs/modules/utils.rst:Set GPU functions
docs/modules/utils.rst:.. autofunction:: set_gpu_fraction
docs/modules/db.rst:To make it easier, we can distribute these tasks to several GPU servers.
docs/modules/db.rst:A task distributor can push both dataset and tasks into a database, allowing task runners on GPU servers to pull and run.
docs/modules/db.rst:The task runners on GPU servers can monitor the database, and run the tasks immediately when they are made available.
CHANGELOG.md:  - Creation of installation flaggs `all_dev`, `all_cpu_dev`, and `all_gpu_dev` (PR #739)
CHANGELOG.md:    - py2 + gpu
CHANGELOG.md:    - py3 + gpu
CHANGELOG.md:  - Creation of installation flaggs `all`, `all_cpu`, and `all_gpu` (PR #660)
CHANGELOG.md:- Tensorflow CPU & GPU dependencies moved to separated requirement files in order to allow PyUP.io to parse them (PR #573)
tests/performance_test/vgg/tl2-eager.py:gpus = tf.config.experimental.list_physical_devices('GPU')
tests/performance_test/vgg/tl2-eager.py:if gpus:
tests/performance_test/vgg/tl2-eager.py:    for gpu in gpus:
tests/performance_test/vgg/tl2-eager.py:        tf.config.experimental.set_memory_growth(gpu, True)
tests/performance_test/vgg/tf2-eager.py:gpus = tf.config.experimental.list_physical_devices('GPU')
tests/performance_test/vgg/tf2-eager.py:if gpus:
tests/performance_test/vgg/tf2-eager.py:    for gpu in gpus:
tests/performance_test/vgg/tf2-eager.py:        tf.config.experimental.set_memory_growth(gpu, True)
tests/performance_test/vgg/tl2-static-autograph.py:gpus = tf.config.experimental.list_physical_devices('GPU')
tests/performance_test/vgg/tl2-static-autograph.py:if gpus:
tests/performance_test/vgg/tl2-static-autograph.py:    for gpu in gpus:
tests/performance_test/vgg/tl2-static-autograph.py:        tf.config.experimental.set_memory_growth(gpu, True)
tests/performance_test/vgg/README.md:### With GPU
tests/performance_test/vgg/README.md:- GPU: TITAN Xp
tests/performance_test/vgg/README.md:|   Mode    |       Lib       |  Data Format  | Max GPU Memory Usage(MB)  |Max CPU Memory Usage(MB) | Avg CPU Memory Usage(MB) | Runtime (sec) |
tests/performance_test/vgg/keras_test.py:config.gpu_options.allow_growth = True
tests/performance_test/vgg/tf2-autograph.py:gpus = tf.config.experimental.list_physical_devices('GPU')
tests/performance_test/vgg/tf2-autograph.py:if gpus:
tests/performance_test/vgg/tf2-autograph.py:    for gpu in gpus:
tests/performance_test/vgg/tf2-autograph.py:        tf.config.experimental.set_memory_growth(gpu, True)
tests/performance_test/vgg/pytorch_test.py:# set gpu_id 0
tests/performance_test/vgg/pytorch_test.py:device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tests/performance_test/vgg/tl2-static-eager.py:gpus = tf.config.experimental.list_physical_devices('GPU')
tests/performance_test/vgg/tl2-static-eager.py:if gpus:
tests/performance_test/vgg/tl2-static-eager.py:    for gpu in gpus:
tests/performance_test/vgg/tl2-static-eager.py:        tf.config.experimental.set_memory_growth(gpu, True)
tests/performance_test/vgg/tl2-autograph.py:gpus = tf.config.experimental.list_physical_devices('GPU')
tests/performance_test/vgg/tl2-autograph.py:if gpus:
tests/performance_test/vgg/tl2-autograph.py:    for gpu in gpus:
tests/performance_test/vgg/tl2-autograph.py:        tf.config.experimental.set_memory_growth(gpu, True)
README.md:- 🔥 [TensorLayerX](https://github.com/tensorlayer/tensorlayerx) is a Unified Deep Learning and Reinforcement Learning Framework for All Hardwares, Backends and OS. The current version supports TensorFlow, Pytorch, MindSpore, PaddlePaddle, OneFlow and Jittor as the backends, allowing users to run the code on different hardware like Nvidia-GPU and Huawei-Ascend.
README.md:TensorLayer 2.0 relies on TensorFlow, numpy, and others. To use GPUs, CUDA and cuDNN are required.
README.md:pip3 install tensorflow-gpu==2.0.0-rc1 # TensorFlow GPU (version 2.0 RC1)
README.md:### Containers with GPU support
README.md:NVIDIA-Docker is required for these containers to work: [Project Link](https://github.com/NVIDIA/nvidia-docker)
README.md:# for GPU version and Python 2
README.md:docker pull tensorlayer/tensorlayer:latest-gpu
README.md:nvidia-docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu
README.md:# for GPU version and Python 3
README.md:docker pull tensorlayer/tensorlayer:latest-gpu-py3
README.md:nvidia-docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu-py3
README.md:|   Mode    |       Lib       |  Data Format  | Max GPU Memory Usage(MB)  |Max CPU Memory Usage(MB) | Avg CPU Memory Usage(MB) | Runtime (sec) |
tensorlayer/cli/__main__.py:    train_parser = subparsers.add_parser('train', help='train a model using multiple local GPUs or CPUs.')
tensorlayer/cli/train.py:using multiple GPU cards or CPUs on a computer.
tensorlayer/cli/train.py:You need to first setup the `CUDA_VISIBLE_DEVICES <http://acceleware.com/blog/cudavisibledevices-masking-gpus>`_
tensorlayer/cli/train.py:to tell ``tl train`` which GPUs are available. If the CUDA_VISIBLE_DEVICES is not given,
tensorlayer/cli/train.py:``tl train`` would try best to discover all available GPUs.
tensorlayer/cli/train.py:  # example of using GPU 0 and 1 for training mnist
tensorlayer/cli/train.py:  CUDA_VISIBLE_DEVICES="0,1"
tensorlayer/cli/train.py:  # example of using GPU trainers for inception v3 with customized arguments
tensorlayer/cli/train.py:  # as CUDA_VISIBLE_DEVICES is not given, tl would try to discover all available GPUs
tensorlayer/cli/train.py:The reason we are not supporting GPU-CPU co-training is because GPU and
tensorlayer/cli/train.py:def _get_gpu_ids():
tensorlayer/cli/train.py:    if 'CUDA_VISIBLE_DEVICES' in os.environ:
tensorlayer/cli/train.py:        return [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')]
tensorlayer/cli/train.py:        return [int(d.replace('nvidia', '')) for d in os.listdir('/dev') if re.match('^nvidia\d+$', d)]
tensorlayer/cli/train.py:        print('Please set CUDA_VISIBLE_DEVICES (see http://acceleware.com/blog/cudavisibledevices-masking-gpus)')
tensorlayer/cli/train.py:GPU_IDS = _get_gpu_ids()
tensorlayer/cli/train.py:    gpu_assignment = dict((('worker', idx), gpu_idx) for (idx, gpu_idx) in enumerate(GPU_IDS))
tensorlayer/cli/train.py:                    'CUDA_VISIBLE_DEVICES': str(gpu_assignment.get((job_type, task_index), '')),
tensorlayer/cli/train.py:    if not GPU_IDS:
tensorlayer/cli/train.py:    num_workers = len(GPU_IDS) if GPU_IDS else args.cpu_trainers
tensorlayer/cli/train.py:    print('Using %d workers, %d parameter servers, %d GPUs.' % (num_workers, args.num_pss, len(GPU_IDS)))
tensorlayer/models/core.py:                "training_device": "gpu",
tensorlayer/files/utils.py:    #     "training_device": "gpu",
tensorlayer/distributed.py:    from a single GPU to multiple GPUs that be placed on different machines in a single cluster.
tensorlayer/distributed.py:        shards the training dataset based on the number of GPUs.
tensorlayer/distributed.py:        the number of GPUs.
tensorlayer/distributed.py:        Note that the learning rate is linearly scaled according to the number of GPU by default.
tensorlayer/distributed.py:        The dataset prefetch buffer size. Set this parameter to overlap the GPU training and data preparation
tensorlayer/distributed.py:        Linearly scale the learning rate by the number of GPUs. Default is True.
tensorlayer/distributed.py:        # Adjust learning rate based on number of GPUs.
tensorlayer/distributed.py:            # Horovod: adjust number of steps based on number of GPUs.
tensorlayer/distributed.py:        # Pin GPU to be used to process local rank (one GPU per process)
tensorlayer/distributed.py:        config.gpu_options.allow_growth = True
tensorlayer/distributed.py:        config.gpu_options.visible_device_list = str(hvd.local_rank())
tensorlayer/utils.py:    'set_gpu_fraction', 'train_epoch', 'run_epoch'
tensorlayer/utils.py:    """Close TensorBoard and Nvidia-process if available.
tensorlayer/utils.py:    text = "[TL] Close tensorboard and nvidia-process if available"
tensorlayer/utils.py:    text2 = "[TL] Close tensorboard and nvidia-process not yet supported by this function (tl.ops.exit_tf) on "
tensorlayer/utils.py:        os.system('nvidia-smi')
tensorlayer/utils.py:        os.system("nvidia-smi | grep python |awk '{print $3}'|xargs kill")  # kill all nvidia-smi python process
tensorlayer/utils.py:def set_gpu_fraction(gpu_fraction=0.3):
tensorlayer/utils.py:    """Set the GPU memory fraction for the application.
tensorlayer/utils.py:    gpu_fraction : None or float
tensorlayer/utils.py:        Fraction of GPU memory, (0 ~ 1]. If None, allow gpu memory growth.
tensorlayer/utils.py:    - `TensorFlow using GPU <https://www.tensorflow.org/alpha/guide/using_gpu#allowing_gpu_memory_growth>`__
tensorlayer/utils.py:    if gpu_fraction is None:
tensorlayer/utils.py:        tl.logging.info("[TL]: ALLOW GPU MEM GROWTH")
tensorlayer/utils.py:        tf.config.gpu.set_per_process_memory_growth(True)
tensorlayer/utils.py:        tl.logging.info("[TL]: GPU MEM Fraction %f" % gpu_fraction)
tensorlayer/utils.py:        tf.config.gpu.set_per_process_memory_fraction(0.4)
tensorlayer/utils.py:    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
tensorlayer/utils.py:    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tensorlayer/__init__.py:            " - `pip install --upgrade tensorflow-gpu`"
tensorlayer/__init__.py:            " - `pip install --upgrade tensorflow-gpu`"
requirements/requirements_tf_gpu.txt:tensorflow-gpu>=2.0.0-rc1
docker/Dockerfile:#        - Python 2 + GPU: "latest-gpu"      =>  --build-arg TF_CONTAINER_VERSION="latest-gpu"
docker/Dockerfile:#        - Python 3 + GPU: "latest-gpu-py3"  =>  --build-arg TF_CONTAINER_VERSION="latest-gpu-py3"
docker/Dockerfile:            latest-py3 | latest-gpu-py3) apt-get install -y python3-tk  ;; \
examples/database/README.md:2. On your GPU servers (for testing, it can be a new terminal on your local machine), run tasks as shown in `run_tasks.py`. 
examples/basic_tutorials/tutorial_mnist_simple.py:# set gpu mem fraction or allow growth
examples/basic_tutorials/tutorial_mnist_simple.py:# tl.utils.set_gpu_fraction()
examples/tutorial_work_with_onnx.py:TensorFlow-gpu:1.8.0
examples/distributed_training/tutorial_cifar10_distributed_trainer.py:        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
examples/deprecated_tutorials/tutorial_mnist_distributed.py:where CUDA_VISIBLE_DEVICES can be used to set the GPUs the process can use.
examples/deprecated_tutorials/tutorial_mnist_distributed.py:CUDA_VISIBLE_DEVICES= TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"type": "worker", "index": 0}}' python example/tutorial_mnist_distributed.py > output-master 2>&1 &
examples/deprecated_tutorials/tutorial_mnist_distributed.py:CUDA_VISIBLE_DEVICES= TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"type": "worker", "index": 1}}' python example/tutorial_mnist_distributed.py > output-worker 2>&1 &
examples/deprecated_tutorials/tutorial_mnist_distributed.py:CUDA_VISIBLE_DEVICES= TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"type": "ps", "index": 0}}' python example/tutorial_mnist_distributed.py > output-ps 2>&1 &
examples/deprecated_tutorials/tutorial_mnist_distributed.py:Note: for GPU, please set CUDA_VISIBLE_DEVICES=GPU_ID
examples/deprecated_tutorials/tutorial_mnist_distributed.yml:      CUDA_VISIBLE_DEVICES: ''
examples/deprecated_tutorials/tutorial_mnist_distributed.yml:      CUDA_VISIBLE_DEVICES: ''
examples/deprecated_tutorials/tutorial_mnist_distributed.yml:      CUDA_VISIBLE_DEVICES: ''
examples/reinforcement_learning/tutorial_wrappers.py:cv2.ocl.setUseOpenCL(False)
examples/reinforcement_learning/README.md:* tensorflow >= 2.0.0 or tensorflow-gpu >= 2.0.0a0
examples/reinforcement_learning/README.md:pip install tensorflow-gpu==2.0.0-rc1 # if no GPU, use pip install tensorflow==2.0.0
examples/quantized_net/tutorial_binarynet_cifar10_tfrecord.py:after 500 epoches' training with GPU,accurcy of 41.1% was found.
examples/quantized_net/tutorial_ternaryweight_cifar10_tfrecord.py:after 500 epoches' training with GPU,accurcy of 80.6% was found.
examples/quantized_net/tutorial_quanconv_cifar10.py:after 705 epoches' training with GPU, test accurcy of 84.0% was found.
examples/quantized_net/tutorial_dorefanet_cifar10_tfrecord.py:after 500 epoches' training with GPU,accurcy of 81.1% was found.
examples/app_tutorials/README.md:[2]:https://openaccess.thecvf.com/content_ICCV_2019/papers/Ci_Optimizing_Network_Structure_for_3D_Human_Pose_Estimation_ICCV_2019_paper.pdf
CONTRIBUTING.md:# advanced: for a machine **without** an NVIDIA GPU
CONTRIBUTING.md:# advanced: for a machine **with** an NVIDIA GPU
CONTRIBUTING.md:pip install -e ".[all_gpu_dev]"

```
