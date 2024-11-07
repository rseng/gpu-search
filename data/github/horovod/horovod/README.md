# https://github.com/horovod/horovod

```console
setup.py:# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
setup.py:tensorflow_gpu_require_list = ['tensorflow-gpu']
setup.py:# NOTE: do not use versions with +cpu or +gpu here as users would need to add --find-links to pip
setup.py:          'tensorflow-gpu': tensorflow_gpu_require_list,
README.rst:The primary motivation for this project is to make it easy to take a single-GPU training script and successfully scale
README.rst:it to train across many GPUs in parallel. This has two aspects:
README.rst:Horovod, it can run on a single-GPU, multiple-GPUs, or even multiple hosts without any further code changes.
README.rst:servers with 4 Pascal GPUs each connected by RoCE-capable 25 Gbit/s network:
README.rst:   :alt: 512-GPU Benchmark
README.rst:While installing MPI and NCCL itself may seem like an extra hassle, it only needs to be done once by the team dealing
README.rst:   To run on GPUs with NCCL:
README.rst:      $ HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
README.rst:For more details on installing Horovod with GPU support, read `Horovod on GPU <docs/gpus.rst>`_.
README.rst:If you want to use Conda, read `Building a Conda environment with GPU support for Horovod <docs/conda.rst>`_.
README.rst:2. Pin each GPU to a single process to avoid resource contention.
README.rst:   With the typical setup of one GPU per process, set this to *local rank*. The first process on
README.rst:   the server will be allocated the first GPU, the second process will be allocated the second GPU, and so forth.
README.rst:    # Pin GPU to be used to process local rank (one GPU per process)
README.rst:    config.gpu_options.visible_device_list = str(hvd.local_rank())
README.rst:1. To run on a machine with 4 GPUs:
README.rst:2. To run on 4 machines with 4 GPUs each:
docs/troubleshooting.rst:2. Are the CUDA libraries available?
docs/troubleshooting.rst:If you're installing Horovod into a container on a machine without GPUs, you may use CUDA stub drivers to work around the issue.
docs/troubleshooting.rst:    ImportError: libcuda.so.1: cannot open shared object file: No such file or directory
docs/troubleshooting.rst:To use CUDA stub drivers:
docs/troubleshooting.rst:    $ ldconfig /usr/local/cuda/lib64/stubs
docs/troubleshooting.rst:    $ HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=/path/to/nccl pip install --no-cache-dir horovod
docs/troubleshooting.rst:    $ HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=/path/to/nccl pip install --no-cache-dir horovod
docs/troubleshooting.rst:NCCL 2 is not found during installation
docs/troubleshooting.rst:If you see the error message below, it means NCCL 2 was not found in the standard libraries location. If you have a directory
docs/troubleshooting.rst:where you installed NCCL 2 which has both ``include`` and ``lib`` directories containing ``nccl.h`` and ``libnccl.so``
docs/troubleshooting.rst:respectively, you can pass it via ``HOROVOD_NCCL_HOME`` environment variable. Otherwise you can specify them separately
docs/troubleshooting.rst:via ``HOROVOD_NCCL_INCLUDE`` and ``HOROVOD_NCCL_LIB`` environment variables.
docs/troubleshooting.rst:    build/temp.linux-x86_64-2.7/test_compile/test_nccl.cc:1:18: fatal error: nccl.h: No such file or directory
docs/troubleshooting.rst:     #include <nccl.h>
docs/troubleshooting.rst:    error: NCCL 2.0 library or its later version was not found (see error above).
docs/troubleshooting.rst:    Please specify correct NCCL location via HOROVOD_NCCL_HOME environment variable or combination of HOROVOD_NCCL_INCLUDE and HOROVOD_NCCL_LIB environment variables.
docs/troubleshooting.rst:    HOROVOD_NCCL_HOME - path where NCCL include and lib directories can be found
docs/troubleshooting.rst:    HOROVOD_NCCL_INCLUDE - path to NCCL include directory
docs/troubleshooting.rst:    HOROVOD_NCCL_LIB - path to NCCL lib directory
docs/troubleshooting.rst:    $ HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=/path/to/nccl pip install --no-cache-dir horovod
docs/troubleshooting.rst:    $ HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_INCLUDE=/path/to/nccl/include HOROVOD_NCCL_LIB=/path/to/nccl/lib pip install --no-cache-dir horovod
docs/troubleshooting.rst:    $ HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=/path/to/nccl pip install --no-cache-dir horovod
docs/troubleshooting.rst:ncclAllReduce failed: invalid data type
docs/troubleshooting.rst:If you see the error message below during the training, it means that Horovod was linked to the wrong version of NCCL
docs/troubleshooting.rst:    UnknownError (see above for traceback): ncclAllReduce failed: invalid data type
docs/troubleshooting.rst:             [[Node: DistributedMomentumOptimizer_Allreduce/HorovodAllreduce_gradients_AddN_2_0 = HorovodAllreduce[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:GPU:0"](gradients/AddN_2)]]
docs/troubleshooting.rst:             [[Node: train_op/_653 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_1601_train_op", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:
docs/troubleshooting.rst:If you're using Anaconda or Miniconda, you most likely have the ``nccl`` package installed. The solution is to remove
docs/troubleshooting.rst:    $ conda remove nccl
docs/troubleshooting.rst:    $ HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=/path/to/nccl pip install --no-cache-dir horovod
docs/troubleshooting.rst:transport/p2p.cu:431 WARN failed to open CUDA IPC handle : 30 unknown error
docs/troubleshooting.rst:If you see the error message below during the training with ``-x NCCL_DEBUG=INFO``, it likely means that multiple servers
docs/troubleshooting.rst:    node1:22671:22795 [1] transport/p2p.cu:431 WARN failed to open CUDA IPC handle : 30 unknown error
docs/troubleshooting.rst:MPI and NCCL rely on hostnames to distinguish between servers, so you should make sure that every server has a unique
docs/troubleshooting.rst:If you notice that your program is running out of GPU memory and multiple processes
docs/troubleshooting.rst:are being placed on the same GPU, it's likely that your program (or its dependencies)
docs/troubleshooting.rst:create a ``tf.Session`` that does not use the ``config`` that pins specific GPU.
docs/troubleshooting.rst:to minimize the amount of memory it will pre-allocate on each GPU:
docs/troubleshooting.rst:    small_cfg.gpu_options.allow_growth = True
docs/troubleshooting.rst:As a last resort, you can **replace** setting ``config.gpu_options.visible_device_list``
docs/troubleshooting.rst:    # Pin GPU to be used
docs/troubleshooting.rst:    os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
docs/troubleshooting.rst:**Note**: Setting ``CUDA_VISIBLE_DEVICES`` is incompatible with ``config.gpu_options.visible_device_list``.
docs/troubleshooting.rst:Setting ``CUDA_VISIBLE_DEVICES`` has additional disadvantage for GPU version - CUDA will not be able to use IPC, which
docs/troubleshooting.rst:will likely cause NCCL and MPI to fail.  In order to disable IPC in NCCL and MPI and allow it to fallback to shared
docs/troubleshooting.rst:* ``export NCCL_P2P_DISABLE=1`` for NCCL.
docs/troubleshooting.rst:* ``--mca btl_smcuda_use_cuda_ipc 0`` flag for OpenMPI and similar flags for other vendors.
docs/troubleshooting.rst:libcudart.so.X.Y: cannot open shared object file: No such file or directory
docs/troubleshooting.rst:If you notice that your program crashes with a ``libcudart.so.X.Y: cannot open shared object file: No such file or directory`` error, it's likely that your framework and Horovod were build with different versions of CUDA.
docs/troubleshooting.rst:To build Horovod with a specific CUDA version, use the ``HOROVOD_CUDA_HOME`` environment variable during installation:
docs/troubleshooting.rst:    $ HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=/path/to/nccl HOROVOD_CUDA_HOME=/path/to/cuda pip install --no-cache-dir horovod
docs/pytorch.rst:2. Pin each GPU to a single process.
docs/pytorch.rst:   With the typical setup of one GPU per process, set this to *local rank*. The first process on
docs/pytorch.rst:   the server will be allocated the first GPU, the second process will be allocated the second GPU, and so forth.
docs/pytorch.rst:       if torch.cuda.is_available():
docs/pytorch.rst:           torch.cuda.set_device(hvd.local_rank())
docs/pytorch.rst:    # Pin GPU to be used to process local rank (one GPU per process)
docs/pytorch.rst:    torch.cuda.set_device(hvd.local_rank())
docs/pytorch.rst:    model.cuda()
docs/pytorch.rst:.. NOTE:: PyTorch GPU support requires NCCL 2.2 or later. It also works with NCCL 2.1.15 if you are not using RoCE or InfiniBand.
docs/pytorch.rst:    # train Horovod on GPU (number of GPUs / machines provided on command-line)
docs/pytorch.rst:    trainer = pl.Trainer(accelerator='horovod', gpus=1)
docs/pytorch.rst:    # run training with 4 GPUs on a single machine
docs/pytorch.rst:    # run training with 8 GPUs on two machines (4 GPUs each)
docs/pytorch.rst:See the PyTorch Lightning `docs <https://pytorch-lightning.readthedocs.io/en/stable/multi_gpu.html#horovod>`_ for more details.
docs/autotune.rst:matter of trial-and-error, as many factors including model complexity, network bandwidth, GPU memory, etc. can all
docs/ray.rst:        setting, num_workers=num_workers, use_gpu=True)
docs/ray.rst:        torch.cuda.set_device(hvd.local_rank())
docs/ray.rst:        settings, min_workers=1, use_gpu=True, cpus_per_slot=2)
docs/ray.rst:        - HOROVOD_WITH_GLOO=1 HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[ray]
docs/elastic.rst:    config.gpu_options.allow_growth = True
docs/elastic.rst:    config.gpu_options.visible_device_list = str(hvd.local_rank())
docs/elastic.rst:    gpus = tf.config.experimental.list_physical_devices('GPU')
docs/elastic.rst:    for gpu in gpus:
docs/elastic.rst:        tf.config.experimental.set_memory_growth(gpu, True)
docs/elastic.rst:    if gpus:
docs/elastic.rst:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
docs/elastic.rst:    config.gpu_options.allow_growth = True
docs/elastic.rst:    config.gpu_options.visible_device_list = str(hvd.local_rank())
docs/elastic.rst:    torch.cuda.set_device(hvd.local_rank())
docs/elastic.rst:        torch.cuda.set_device(hvd.local_rank())
docs/elastic.rst:    executor = ElasticRayExecutor(settings, use_gpu=True, cpus_per_slot=2)
docs/lsf.rst:``horovodrun`` will automatically detect the host names and GPUs of your LSF job.
docs/lsf.rst:Here, Horovod will start a process per GPU on all the hosts of the LSF job.
docs/lsf.rst:You can also limit the run to a subset of the job resources. For example, using only 6 GPUs:
docs/lsf.rst:You can still pass extra arguments to ``horovodrun``. For example, to trigger CUDA-Aware MPI:
docs/lsf.rst:    horovodrun --mpi-args="-gpu" python train.py
docs/summary.rst:The primary motivation for this project is to make it easy to take a single-GPU training script and successfully scale
docs/summary.rst:it to train across many GPUs in parallel. This has two aspects:
docs/summary.rst:Horovod, it can run on a single-GPU, multiple-GPUs, or even multiple hosts without any further code changes.
docs/summary.rst:servers with 4 Pascal GPUs each connected by RoCE-capable 25 Gbit/s network:
docs/summary.rst:   :alt: 512-GPU Benchmark
docs/summary.rst:While installing MPI and NCCL itself may seem like an extra hassle, it only needs to be done once by the team dealing
docs/summary.rst:   To run on GPUs with NCCL:
docs/summary.rst:      $ HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
docs/summary.rst:For more details on installing Horovod with GPU support, read `Horovod on GPU <gpus.rst>`_.
docs/summary.rst:If you want to use Conda, read `Building a Conda environment with GPU support for Horovod <conda.rst>`_.
docs/summary.rst:2. Pin each GPU to a single process to avoid resource contention.
docs/summary.rst:   With the typical setup of one GPU per process, set this to *local rank*. The first process on
docs/summary.rst:   the server will be allocated the first GPU, the second process will be allocated the second GPU, and so forth.
docs/summary.rst:    # Pin GPU to be used to process local rank (one GPU per process)
docs/summary.rst:    config.gpu_options.visible_device_list = str(hvd.local_rank())
docs/summary.rst:1. To run on a machine with 4 GPUs:
docs/summary.rst:2. To run on 4 machines with 4 GPUs each:
docs/running.rst:Typically one GPU will be allocated per process, so if a server has 4 GPUs,
docs/running.rst:To run on a machine with 4 GPUs:
docs/running.rst:To run on 4 machines with 4 GPUs each:
docs/index.rst:         Or, use <a href="https://horovod.readthedocs.io/en/latest/gpus_include.html">Horovod on GPUs</a>, in <a href="https://horovod.readthedocs.io/en/latest/spark_include.html">Spark</a>, <a href="https://horovod.readthedocs.io/en/latest/docker_include.html">Docker</a>, <a href="https://github.com/sylabs/examples/tree/master/machinelearning/horovod">Singularity</a>, or Kubernetes (<a href="https://github.com/kubeflow/examples/tree/master/demos/yelp_demo/ks_app/vendor/kubeflow/mpi-job">Kubeflow</a>, <a href="https://github.com/kubeflow/mpi-operator/">MPI Operator</a>, <a href="https://github.com/helm/charts/tree/master/stable/horovod">Helm Chart</a>, and <a href="https://github.com/IBM/FfDL/tree/master/etc/examples/horovod/">FfDL</a>).
docs/index.rst:         Or, use <a href="https://horovod.readthedocs.io/en/latest/gpus_include.html">Horovod on GPUs</a>, in <a href="https://horovod.readthedocs.io/en/latest/spark_include.html">Spark</a>, <a href="https://horovod.readthedocs.io/en/latest/docker_include.html">Docker</a>, <a href="https://github.com/sylabs/examples/tree/master/machinelearning/horovod">Singularity</a>, or Kubernetes (<a href="https://github.com/kubeflow/examples/tree/master/demos/yelp_demo/ks_app/vendor/kubeflow/mpi-job">Kubeflow</a>, <a href="https://github.com/kubeflow/mpi-operator/">MPI Operator</a>, <a href="https://github.com/helm/charts/tree/master/stable/horovod">Helm Chart</a>, and <a href="https://github.com/IBM/FfDL/tree/master/etc/examples/horovod/">FfDL</a>).
docs/index.rst:         Or, use <a href="https://horovod.readthedocs.io/en/latest/gpus_include.html">Horovod on GPUs</a>, in <a href="https://horovod.readthedocs.io/en/latest/spark_include.html">Spark</a>, <a href="https://horovod.readthedocs.io/en/latest/docker_include.html">Docker</a>, <a href="https://github.com/sylabs/examples/tree/master/machinelearning/horovod">Singularity</a>, or Kubernetes (<a href="https://github.com/kubeflow/examples/tree/master/demos/yelp_demo/ks_app/vendor/kubeflow/mpi-job">Kubeflow</a>, <a href="https://github.com/kubeflow/mpi-operator/">MPI Operator</a>, <a href="https://github.com/helm/charts/tree/master/stable/horovod">Helm Chart</a>, and <a href="https://github.com/IBM/FfDL/tree/master/etc/examples/horovod/">FfDL</a>).
docs/index.rst:         Or, use <a href="https://horovod.readthedocs.io/en/latest/gpus_include.html">Horovod on GPUs</a>, in <a href="https://horovod.readthedocs.io/en/latest/spark_include.html">Spark</a>, <a href="https://horovod.readthedocs.io/en/latest/docker_include.html">Docker</a>, <a href="https://github.com/sylabs/examples/tree/master/machinelearning/horovod">Singularity</a>, or Kubernetes (<a href="https://github.com/kubeflow/examples/tree/master/demos/yelp_demo/ks_app/vendor/kubeflow/mpi-job">Kubeflow</a>, <a href="https://github.com/kubeflow/mpi-operator/">MPI Operator</a>, <a href="https://github.com/helm/charts/tree/master/stable/horovod">Helm Chart</a>, and <a href="https://github.com/IBM/FfDL/tree/master/etc/examples/horovod/">FfDL</a>).
docs/index.rst:   gpus_include
docs/concepts.rst:a training script on 4 servers, each having 4 GPUs. If we launched one copy of the script per GPU:
docs/concepts.rst:* *Reducescatter* is an operation that aggregates data among multiple processes and scatters the data across them.  *Reducescatter* is used to average dense tensors then split them across processes.  Here's an illustration from the `Nvidia developer guide <https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#reducescatter>`__:
docs/concepts.rst:    .. image:: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/_images/reducescatter.png
docs/docker.rst:Pre-built Docker containers with Horovod are available on `DockerHub <https://hub.docker.com/r/horovod/horovod>`__ for GPU, CPU, and `Ray <https://ray.io>`__.
docs/docker.rst:After the container is built, run it using `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`__.
docs/docker.rst:    $ nvidia-docker run -it horovod/horovod:latest
docs/docker.rst:    host1$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod/horovod:latest
docs/docker.rst:    host2$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod/horovod:latest \
docs/docker.rst:    host3$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod/horovod:latest \
docs/docker.rst:    host4$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod/horovod:latest \
docs/docker.rst:   $ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh --cap-add=IPC_LOCK --device=/dev/infiniband horovod/horovod:latest 
docs/contributors.rst:`Dockerfile.test.gpu <https://github.com/horovod/horovod/blob/master/Dockerfile.test.gpu>`__ file.
docs/contributors.rst:`Dockerfile.test.gpu <https://github.com/horovod/horovod/blob/master/Dockerfile.test.gpu>`__ files.
docs/contributors.rst:immediately. For **C++/CUDA code**, the ``... pip install -v -e .`` command needs to be invoked again to perform an incremental build.
docs/contributors.rst:**IMPORTANT:** Some tests contain GPU-only codepaths that will be skipped if running without GPU support or, in some
docs/contributors.rst:cases, if there are fewer than four GPUs installed.
docs/contributors.rst:Intel CPU hardware and NVIDIA GPUs (with NCCL).  Tests are run once per night on master automatically, and on each
docs/contributors.rst:`Docker.test.gpu <https://github.com/horovod/horovod/blob/master/Dockerfile.test.gpu>`__ (for GPU tests).
docs/contributors.rst:In our AWS configuration, GPU tests are run with 4 GPUs per container. Most tests are run with 2 worker processes
docs/contributors.rst:each, however, model parallelism require 2 GPUs per worker, requiring 4 GPUs total.
docs/contributors.rst:1. Custom operations that trigger based on parameters configured at runtime (e.g., ``NCCLHierarchicalAllreduce``).
docs/contributors.rst:2. Accelerated operations that take advantage of specialized hardware where available (e.g., ``NCCLAllreduce``).
docs/mxnet.rst:    # Pin GPU to be used to process local rank
docs/mxnet.rst:    context = mx.gpu(hvd.local_rank())
docs/spark.rst:*  ``tensorflow-gpu >= 1.12.0`` or ``tensorflow >= 1.12.0`` (for ``KerasEstimator``)
docs/spark.rst:For example, the NVTabularDataModule integrates the `KerasSequenceLoader <https://github.com/NVIDIA-Merlin/NVTabular/blob/main/nvtabular/loader/tensorflow.py>`__
docs/spark.rst:from NVTabular to enable GPU-accelerated data loading.
docs/spark.rst:GPU training
docs/spark.rst:For GPU training, one approach is to set up a separate GPU Spark cluster
docs/spark.rst:and configure each executor with ``# of CPU cores`` = ``# of GPUs``. This can
docs/spark.rst:    $ echo "export SPARK_WORKER_CORES=<# of GPUs>" >> /path/to/spark/conf/spark-env.sh
docs/spark.rst:This approach turns the ``spark.task.cpus`` setting to control # of GPUs
docs/spark.rst:introduce GPU-aware resource scheduling in future versions of Spark.
docs/spark.rst:Databricks pre-configures GPU-aware scheduling on Databricks Runtime 7.0 ML GPU and above. See GPU scheduling instructions
docs/spark.rst:(`AWS <https://docs.databricks.com/clusters/gpu.html#gpu-scheduling-1>`__ |
docs/spark.rst:`Azure <https://docs.microsoft.com/en-us/azure/databricks/clusters/gpu#gpu-scheduling>`__)
docs/spark.rst:With the Estimator API, horovod will launch ``# of tasks on each worker = # of GPUs on each worker``, and each task will
docs/spark.rst:pin GPU to the assigned GPU from spark.
docs/spark.rst:With the Run API, the function ``get_available_devices()`` from ``horovod.spark.task`` will return a list of assigned GPUs
docs/spark.rst:In some cases, you may want to ignore GPU devices assigned by Spark and always use the local rank as the GPU index.
docs/spark.rst:You can set environment variable ``HOROVOD_SPARK_USE_LOCAL_RANK_GPU_INDEX`` to ``1`` to have Horovod use the local rank
docs/spark.rst:as the GPU index for each task.
docs/keras.rst:2. Pin each GPU to a single process.
docs/keras.rst:   With the typical setup of one GPU per process, set this to *local rank*. The first process on
docs/keras.rst:   the server will be allocated the first GPU, the second process will be allocated the second GPU, and so forth.
docs/keras.rst:       config.gpu_options.visible_device_list = str(hvd.local_rank())
docs/keras.rst:       gpus = tf.config.experimental.list_physical_devices('GPU')
docs/keras.rst:       for gpu in gpus:
docs/keras.rst:           tf.config.experimental.set_memory_growth(gpu, True)
docs/keras.rst:       if gpus:
docs/keras.rst:           tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
docs/keras.rst:.. NOTE:: - Keras 2.0.9 has a `known issue <https://github.com/fchollet/keras/issues/8353>`_ that makes each worker allocate all GPUs on the server, instead of the GPU assigned by the *local rank*. If you have multiple GPUs per server, upgrade to Keras 2.1.2 or downgrade to Keras 2.0.8.
docs/keras.rst:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
docs/keras.rst:    config.gpu_options.allow_growth = True
docs/keras.rst:    config.gpu_options.visible_device_list = str(hvd.local_rank())
docs/keras.rst:    # Horovod: adjust number of epochs based on number of GPUs.
docs/keras.rst:    # Horovod: adjust learning rate based on number of GPUs.
docs/keras.rst:    # Pin GPU to be used to process local rank (one GPU per process)
docs/keras.rst:    gpus = tf.config.experimental.list_physical_devices('GPU')
docs/keras.rst:    for gpu in gpus:
docs/keras.rst:        tf.config.experimental.set_memory_growth(gpu, True)
docs/keras.rst:    if gpus:
docs/keras.rst:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
docs/hyperparameter_search.rst:            training_function, num_slots=2, use_gpu=use_gpu)
docs/hyperparameter_search.rst:`DistributedTrainableCreator`_ exposes ``num_hosts``, ``num_slots``, ``use_gpu``, and ``num_cpus_per_slot``. Use these parameters to specify the resource allocation of a single "trial" (or "Trainable") which itself can be a distributed training job.
docs/hyperparameter_search.rst:    # Each training job will use 2 GPUs.
docs/hyperparameter_search.rst:        training_function, num_slots=2, use_gpu=True)
docs/hyperparameter_search.rst:Leverage `Ray Tune`_ with Horovod on a laptop, single machine with multiple GPUs, or across multiple machines. To run on a single machine, execute your Python script as-is (for example, `horovod_simple.py <https://docs.ray.io/en/latest/tune/examples/horovod_simple.html>`__, assuming Ray and Horovod are installed properly):
docs/hyperparameter_search.rst:        - HOROVOD_WITH_GLOO=1 HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[ray]
docs/install.rst:For best performance on GPU:
docs/install.rst:- `NCCL 2 <https://developer.nvidia.com/nccl>`__
docs/install.rst:When used as a controller in combination with NCCL, Gloo performs almost identically to MPI on standard benchmarks.
docs/install.rst:For running on GPUs with optimal performance, we recommend installing Horovod with NCCL support following the
docs/install.rst:`Horovod on GPU <gpus.rst>`_ guide.
docs/install.rst:NCCL
docs/install.rst:NCCL is supported for Allreduce, Allgather, Broadcast, and Alltoall operations.  You can enable these by setting
docs/install.rst:``HOROVOD_GPU_OPERATIONS=NCCL`` during installation.
docs/install.rst:NCCL operations are supported on both Nvidia (CUDA) and AMD (ROCm) GPUs. You can set ``HOROVOD_GPU`` in your
docs/install.rst:environment to specify building with CUDA or ROCm. CUDA will be assumed if not specified.
docs/install.rst:Note that Alltoall requires NCCL version >= 2.7.0.
docs/install.rst:When using an MPI controller, MPI will be used when NCCL is unavailable, or if tensors are placed in host memory prior
docs/install.rst:to the allreduce request. In cases where NCCL is unavailable, MPI has been shown to outperform Gloo for CPU tensor
docs/install.rst:MPI can also be used for GPU operations, but this is not recommended in most cases. See `Horovod on GPU <gpus.rst>`_ for
docs/install.rst:To use Conda to install PyTorch, TensorFlow, MXNet, Horovod, as well as GPU dependencies such as
docs/install.rst:NVIDIA CUDA Toolkit, cuDNN, NCCL, etc., see `Build a Conda Environment with GPU Support for Horovod <conda.rst>`_.
docs/install.rst:* ``HOROVOD_CUDA_HOME`` - path where CUDA include and lib directories can be found.
docs/install.rst:* ``HOROVOD_BUILD_CUDA_CC_LIST`` - List of compute capabilities to build Horovod CUDA kernels for (example: ``HOROVOD_BUILD_CUDA_CC_LIST=60,70,75``)
docs/install.rst:* ``HOROVOD_ROCM_HOME`` - path where ROCm include and lib directories can be found.
docs/install.rst:* ``HOROVOD_NCCL_HOME`` - path where NCCL include and lib directories can be found.
docs/install.rst:* ``HOROVOD_NCCL_INCLUDE`` - path to NCCL include directory.
docs/install.rst:* ``HOROVOD_NCCL_LIB`` - path to NCCL lib directory.
docs/install.rst:* ``HOROVOD_NCCL_LINK`` - {SHARED, STATIC}. Mode to link NCCL library. Defaults to STATIC for CUDA, SHARED for ROCm.
docs/install.rst:* ``HOROVOD_GPU`` - {CUDA, ROCM}. Framework to use for GPU operations.
docs/install.rst:* ``HOROVOD_GPU_OPERATIONS`` - {NCCL, MPI}. Framework to use for GPU tensor allreduce, allgather, and broadcast.
docs/install.rst:* ``HOROVOD_GPU_ALLREDUCE`` - {NCCL, MPI}. Framework to use for GPU tensor allreduce.
docs/install.rst:* ``HOROVOD_GPU_ALLGATHER`` - {NCCL, MPI}. Framework to use for GPU tensor allgather.
docs/install.rst:* ``HOROVOD_GPU_BROADCAST`` - {NCCL, MPI}. Framework to use for GPU tensor broadcast.
docs/install.rst:* ``HOROVOD_GPU_ALLTOALL`` - {NCCL, MPI}. Framework to use for GPU tensor alltoall.
docs/install.rst:* ``HOROVOD_GPU_REDUCESCATTER`` - {NCCL, MPI}. Framework to use for GPU tensor reducescatter.
docs/install.rst:* ``HOROVOD_ALLOW_MIXED_GPU_IMPL`` - {1}. Allow Horovod to install with NCCL allreduce and MPI GPU allgather / broadcast / alltoall / reducescatter.  Not recommended due to a possible deadlock.
docs/tensorflow.rst:2. Pin each GPU to a single process.
docs/tensorflow.rst:   With the typical setup of one GPU per process, set this to *local rank*. The first process on
docs/tensorflow.rst:   the server will be allocated the first GPU, the second process will be allocated the second GPU, and so forth.
docs/tensorflow.rst:       config.gpu_options.visible_device_list = str(hvd.local_rank())
docs/tensorflow.rst:       gpus = tf.config.experimental.list_physical_devices('GPU')
docs/tensorflow.rst:       for gpu in gpus:
docs/tensorflow.rst:           tf.config.experimental.set_memory_growth(gpu, True)
docs/tensorflow.rst:       if gpus:
docs/tensorflow.rst:           tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
docs/tensorflow.rst:    # Pin GPU to be used to process local rank (one GPU per process)
docs/tensorflow.rst:    config.gpu_options.visible_device_list = str(hvd.local_rank())
docs/tensorflow.rst:    # Pin GPU to be used to process local rank (one GPU per process)
docs/tensorflow.rst:    gpus = tf.config.experimental.list_physical_devices('GPU')
docs/tensorflow.rst:    for gpu in gpus:
docs/tensorflow.rst:        tf.config.experimental.set_memory_growth(gpu, True)
docs/tensorflow.rst:    if gpus:
docs/tensorflow.rst:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
docs/tensorflow.rst:    # Horovod: adjust number of steps based on number of GPUs.
docs/tensorflow.rst:The compute job normally runs on CPU nodes while the training job runs on GPU nodes. This allows to run CPU intensive
docs/tensorflow.rst:dataset transformation on CPU nodes while running GPU intensive training on GPU nodes. There can be multiple CPUs
docs/tensorflow.rst:dedicated to one GPU task.
docs/tensorflow.rst:and GPU nodes (here ``gpu-node-1`` and ``gpu-node-2``), respectively:
docs/tensorflow.rst:    horovodrun -np 2 --hosts gpu-node-1:1,gpu-node-2:1 python tensorflow2_mnist_data_service.py /tmp/compute.json
docs/adasum_user_guide.rst:Scaling DNN training to many GPUs always comes at a convergence degradation. This is because with larger batch sizes, gradients are averaged and the learning rate per example is smaller. To address this, learning rate is usually scaled up, but this can lead to divergence of model parameters. AdaSum addresses these two issues without introducing any hyperparameter.
docs/adasum_user_guide.rst:Suppose there are two almost-parallel gradients from two different GPUs, g1 and g2, and they need to be reduced as shown in the figure below. The two common practices for reductions are g1+g2, the gray vector, or (g1+g2)/2, the green vector. g1+g2 may cause divergence of the model since it is effectively moving in the direction of g1 or g2 by two times the magnitude of g1 or g2. Therefore, generally (g1+g2)/2 is safer and more desired. Note that (g1+g2)/2 penalizes both the components g1 and g2 equally.
docs/adasum_user_guide.rst:This idea extends to many gradients as well. Suppose there are 2\^n gradients coming from 2\^n different GPUs. AdaSum inductively takes pairs of gradients and reduces them using the method above until all of them are reduced into one gradient. Thus, AdaSum needs the number of nodes to be a power of 2 in the current implementation.
docs/adasum_user_guide.rst:In addition, there are two options of using AdaSum with Horovod: with Message Passing Interface (MPI) and with `NCCL <https://developer.nvidia.com/nccl>`_. 
docs/adasum_user_guide.rst:-   cuda >= 6.0
docs/adasum_user_guide.rst:-   NCCL >= 2.0
docs/adasum_user_guide.rst:*Using NCCL:*
docs/adasum_user_guide.rst:If the **HOROVOD_GPU_OPERATIONS=NCCL** flag is used to compile Horovod, NCCL is used instead. In this case, NCCL will be used for intra-node communication, and AdaSum will be used for inter-node communication.
docs/adasum_user_guide.rst:When dealing with a hardware setup of multiple nodes, each node having worker GPUs that are not connected by a high speed interconnect like `NVLink <https://www.nvidia.com/en-us/data-center/nvlink/>`_, where the communication happens through the CPU, AdaSum through MPI can be used for both intra-node and inter-node communication. In this case, all of the AdaSum ops are performed on the CPU.
docs/adasum_user_guide.rst:On specifically configured machines (`DGX1 <https://www.nvidia.com/en-us/data-center/dgx-1/>`_ nodes with 8 GPUs each), the Ring mode can be used instead of the pure CPU mode. This mode is identical to the pure CPU mode for inter-node communication, but is able to do intra-node communication without going through the CPU. It does this by utilizing CUDA-aware MPI (OpenMPI built with `UCX <https://www.openucx.org/>`_ support) in order to allow direct GPU to GPU communication within nodes. This results in identical convergence benefits to pure CPU mode, but much better throughput on nodes that support it.
docs/adasum_user_guide.rst:Ring mode is currently supported only on **DGX1** nodes having 8 GPUs each.
docs/adasum_user_guide.rst:The hierarchical mode functions similar to the Ring mode, except for using NCCL to do regular averaging intra-node, instead of using CUDA-aware MPI to do an AdaSum-like ring. Note that hierarchical also works on any hardware configuration, and is not limited to DGX1s.
docs/adasum_user_guide.rst:The learning rate that should be used is equal to the best learning rate for a single worker (GPU) scaled by the number of GPUs locally on a node. On very large clusters, scaling this even more by another factor of 1.5-2.0x might give better results but is not guaranteed and should be tried only if scaling by just the local size is not sufficient for good convergence
docs/adasum_user_guide.rst:AdaSum is highly effective in scaling to large batch sizes. The **backward_passes_per_step** parameter of the DistributedOptimizer can be used for gradient accumulation in order to scale to larger effective batch sizes without being limited by GPU memory.
docs/adasum_user_guide.rst:-   If the HOROVOD_GPU_OPERATIONS=NCCL flag is used to compile Horovod, the learning rate that should be used is equal to the best learning rate for a single	worker (GPU) scaled by the number of GPUs locally on a node. On very large	clusters, scaling this even more by another factor of 1.5\-2.0x might give	better results but is not guaranteed and should be tried only if scaling by just the local size is not sufficient for good convergence.
docs/adasum_user_guide.rst:-   When HOROVOD_GPU_OPERATIONS=NCCL flag is used to compile Horovod and training	is run on a single node, only averaging through NCCL library is used to	perform reductions and no AdaSum algorithm will take place in this configuration.
docs/mpi.rst:MPI can be used as an alternative to Gloo for coordinating work between processes in Horovod. When using NCCL, performance
docs/mpi.rst:1. Run on a machine with 4 GPUs:
docs/mpi.rst:           -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
docs/mpi.rst:2. Run on 4 machines with 4 GPUs each:
docs/mpi.rst:           -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
docs/mpi.rst:does not have noticeable performance impact since most of the heavy communication is done by NCCL, which will use RDMA
docs/mpi.rst:via RoCE or InfiniBand if they're available (see `Horovod on GPU <gpus.rst>`_).  Notable exceptions from this rule are
docs/mpi.rst:With the ``-x`` option you can specify (``-x NCCL_DEBUG=INFO``) or copy (``-x LD_LIBRARY_PATH``) an environment variable to
docs/mpi.rst:        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
docs/mpi.rst:        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x HOROVOD_MPI_THREADS_DISABLE=1 -x PATH \
docs/mpi.rst:If you see non-routed interfaces (like ``docker0``) in the output of ``ifconfig``, you should tell Open MPI and NCCL to not
docs/mpi.rst:use them via the ``-mca btl_tcp_if_exclude <interface>[,<interface>]`` and ``NCCL_SOCKET_IFNAME=^<interface>[,<interface>]``
docs/mpi.rst:        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
docs/mpi.rst:        -x NCCL_SOCKET_IFNAME=^lo,docker0 \
docs/benchmarks.rst:   :alt: 512-GPU Benchmark
docs/benchmarks.rst:The above benchmark was done on 128 servers with 4 Pascal GPUs each connected by a RoCE-capable 25 Gbit/s network. Horovod
docs/benchmarks.rst:1. Install Horovod using the instructions provided on the `Horovod on GPU <https://github.com/horovod/horovod/blob/master/docs/gpus.rst>`__ page.
docs/conda.rst:Build a Conda Environment with GPU Support for Horovod
docs/conda.rst:Horovod to enable distributed training across multiple GPUs (either on the same node or 
docs/conda.rst:Installing the NVIDIA CUDA Toolkit
docs/conda.rst:Install `NVIDIA CUDA Toolkit 10.1`_ (`documentation`_) which is the most recent version of NVIDIA 
docs/conda.rst:CUDA Toolkit supported by all three deep learning frameworks that are currently supported by 
docs/conda.rst:Why not just use the ``cudatoolkit`` package?
docs/conda.rst:Typically when installing PyTorch, TensorFlow, or Apache MXNet with GPU support using Conda, you 
docs/conda.rst:add the appropriate version of the ``cudatoolkit`` package to your ``environment.yml`` file. 
docs/conda.rst:Unfortunately, for the moment at least, the cudatoolkit packages available via Conda do not 
docs/conda.rst:include the `NVIDIA CUDA Compiler (NVCC)`_, which is required in order to build Horovod extensions 
docs/conda.rst:What about the ``cudatoolkit-dev`` package?
docs/conda.rst:While there are ``cudatoolkit-dev`` packages available from ``conda-forge`` that do include NVCC, 
docs/conda.rst:Despite this, we would encourage you to try adding ``cudatoolkit-dev`` to your ``environment.yml`` 
docs/conda.rst:is to install the NVIDIA CUDA Toolkit on your system and then install a meta-package 
docs/conda.rst:installed on the system together with the other CUDA Toolkit components installed inside the Conda 
docs/conda.rst:1. Even though you have installed the NVIDIA CUDA Toolkit manually, you should still use Conda to 
docs/conda.rst:   manage the other required CUDA components such as ``cudnn`` and ``nccl`` (and the optional 
docs/conda.rst:   manually installed CUDA Toolkit.
docs/conda.rst:   package which provides a CUDA-aware build of OpenMPI.
docs/conda.rst:    - mpi4py=3.0 # installs cuda-aware openmpi
docs/conda.rst:    - nccl=2.5
docs/conda.rst:    - nvcc_linux-64=10.1 # configures environment to be "cuda-aware"
docs/conda.rst:    - tensorflow-gpu=2.1
docs/conda.rst:JupyterLab extensions to enable GPU and CPU resource monitoring via `jupyterlab-nvdashboard`_ and 
docs/conda.rst:    $ export HOROVOD_CUDA_HOME=$CUDA_HOME
docs/conda.rst:    $ export HOROVOD_NCCL_HOME=$ENV_PREFIX
docs/conda.rst:    $ export HOROVOD_GPU_OPERATIONS=NCCL
docs/conda.rst:        [X] NCCL
docs/conda.rst:    export HOROVOD_CUDA_HOME=$CUDA_HOME
docs/conda.rst:    export HOROVOD_NCCL_HOME=$ENV_PREFIX
docs/conda.rst:    export HOROVOD_GPU_OPERATIONS=NCCL
docs/conda.rst:    ./bin/create-conda-env.sh # assumes that $CUDA_HOME is set properly
docs/conda.rst:.. _NVIDIA CUDA Toolkit 10.1: https://developer.nvidia.com/cuda-10.1-download-archive-update2
docs/conda.rst:.. _documentation: https://docs.nvidia.com/cuda/archive/10.1/
docs/conda.rst:.. _NVIDIA CUDA Compiler (NVCC): https://docs.nvidia.com/cuda/archive/10.1/cuda-compiler-driver-nvcc/index.html
docs/conda.rst:.. _GitHub: https://github.com/kaust-vislab/horovod-gpu-data-science-project
docs/timeline.rst:   * ``WAIT_FOR_DATA`` indicates time taken to wait for GPU to finish computing input to the **allreduce**, *allgather*, or **broadcast** operations. This happens because TensorFlow tries to smartly interleave scheduling and GPU computation. This is only applicable to situations where the Horovod operation is placed on GPU.
docs/timeline.rst:   * ``WAIT_FOR_OTHER_TENSOR_DATA`` indicates time taken to wait for GPU to finish computing other inputs for other operations that are part of the same fusion batch.
docs/timeline.rst:   * ``QUEUE`` happens when reduction is done with NCCL, and the previous NCCL operation did not finish yet.
docs/timeline.rst:   * ``NCCL_ALLREDUCE``, ``MPI_ALLREDUCE``, ``MPI_ALLGATHER``, or ``MPI_BCAST`` indicate time taken to do the actual operation on GPU (or CPU) and highlights whether the operation was performed using NCCL or pure MPI.
docs/timeline.rst:   * In case of ``HOROVOD_HIERARCHICAL_ALLREDUCE=1``, ``NCCL_ALLREDUCE`` will become a sequence or a subsequence of ``NCCL_REDUCESCATTER``, ``NCCL_REDUCE``, ``MEMCPY_IN_HOST_BUFFER``, ``MPI_ALLREDUCE``, ``MEMCPY_OUT_HOST_BUFFER``, ``NCCL_ALLGATHER``, ``NCCL_BCAST``.
docs/gpus.rst:Horovod on GPU
docs/gpus.rst:To use Horovod on GPU, read the options below and see which one applies to you best.
docs/gpus.rst:Have GPUs?
docs/gpus.rst:In most situations, using NCCL 2 will significantly improve performance over the CPU version.  NCCL 2 provides the **allreduce**
docs/gpus.rst:operation optimized for NVIDIA GPUs and a variety of networking devices, such as RoCE or InfiniBand.
docs/gpus.rst:1. Install `NCCL 2 <https://developer.nvidia.com/nccl>`__ following `these steps <http://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html>`__.
docs/gpus.rst:   If you have installed NCCL 2 using the ``nccl-<version>.txz`` package, you should add the library path to ``LD_LIBRARY_PATH``
docs/gpus.rst:       $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl-<version>/lib
docs/gpus.rst:2. (Optional) If you're using an NVIDIA Tesla GPU and NIC with GPUDirect RDMA support, you can further speed up NCCL 2
docs/gpus.rst:   `GPUDirect <https://developer.nvidia.com/gpudirect>`__ allows GPUs to transfer memory among each other without CPU
docs/gpus.rst:   involvement, which significantly reduces latency and load on CPU.  NCCL 2 is able to use GPUDirect automatically for
docs/gpus.rst:   If you installed NCCL 2 using the ``nccl-<version>.txz`` package, you should specify the path to NCCL 2 using the ``HOROVOD_NCCL_HOME``
docs/gpus.rst:       $ HOROVOD_NCCL_HOME=/usr/local/nccl-<version> HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod
docs/gpus.rst:   If you installed NCCL 2 using the Ubuntu package, you can run:
docs/gpus.rst:       $ HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod
docs/gpus.rst:   If you installed NCCL 2 using the `CentOS / RHEL package <https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#rhel_centos>`__, you can run:
docs/gpus.rst:       $ HOROVOD_NCCL_INCLUDE=/usr/include HOROVOD_NCCL_LIB=/usr/lib64 HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod
docs/gpus.rst:GPU version is available. To force allreduce to happen on CPU, pass ``device_dense='/cpu:0'`` to ``hvd.DistributedOptimizer``:
docs/gpus.rst:Advanced: Have a proprietary MPI implementation with GPU support optimized for your network?
docs/gpus.rst:This section is only relevant if you have a proprietary MPI implementation with GPU support, i.e. not Open MPI or MPICH.
docs/gpus.rst:If your MPI vendor's implementation of *allreduce* operation on GPU is faster than NCCL 2, you can configure Horovod to
docs/gpus.rst:    $ HOROVOD_GPU_ALLREDUCE=MPI pip install --no-cache-dir horovod
docs/gpus.rst:Additionally, if your MPI vendor's implementation supports *allgather*, *broadcast*, and *reducescatter* operations on GPU, you can
docs/gpus.rst:    $ HOROVOD_GPU_OPERATIONS=MPI pip install --no-cache-dir horovod
docs/gpus.rst:training.  If you find yourself running out of GPU memory, you can force allgather to happen on CPU by passing
docs/gpus_include.rst:.. include:: ./gpus.rst
test/single/test_ray.py:def ray_start_4_cpus_4_gpus():
test/single/test_ray.py:    orig_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
test/single/test_ray.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
test/single/test_ray.py:    address_info = ray.init(num_cpus=4, num_gpus=4)
test/single/test_ray.py:            os.environ["CUDA_VISIBLE_DEVICES"] = orig_devices
test/single/test_ray.py:            del os.environ["CUDA_VISIBLE_DEVICES"]
test/single/test_ray.py:    torch.cuda.device_count() < 4, reason="GPU test requires 4 GPUs")
test/single/test_ray.py:    not torch.cuda.is_available(), reason="GPU test requires CUDA.")
test/single/test_ray.py:def test_gpu_ids(ray_start_4_cpus_4_gpus):
test/single/test_ray.py:        setting, num_hosts=1, num_workers_per_host=4, use_gpu=True)
test/single/test_ray.py:    all_cudas = {ev["CUDA_VISIBLE_DEVICES"] for ev in all_envs}
test/single/test_ray.py:    assert len(all_cudas) == 1, all_cudas
test/single/test_ray.py:    assert len(all_envs[0]["CUDA_VISIBLE_DEVICES"].split(",")) == 4, all_envs[0]["CUDA_VISIBLE_DEVICES"]
test/single/test_ray.py:    torch.cuda.device_count() < 4, reason="GPU test requires 4 GPUs")
test/single/test_ray.py:    not torch.cuda.is_available(), reason="GPU test requires CUDA.")
test/single/test_ray.py:def test_gpu_ids_num_workers(ray_start_4_cpus_4_gpus):
test/single/test_ray.py:    hjob = RayExecutor(setting, num_workers=4, use_gpu=True)
test/single/test_ray.py:    all_cudas = {ev["CUDA_VISIBLE_DEVICES"] for ev in all_envs}
test/single/test_ray.py:    assert len(all_cudas) == 1, all_cudas
test/single/test_ray.py:    assert len(all_envs[0]["CUDA_VISIBLE_DEVICES"].split(",")) == 4, all_envs[0]["CUDA_VISIBLE_DEVICES"]
test/single/test_ray.py:        return local_rank in os.environ["CUDA_VISIBLE_DEVICES"]
test/single/test_ray.py:        use_gpu=torch.cuda.is_available())
test/single/test_ray.py:        use_gpu=torch.cuda.is_available())
test/single/test_ray.py:        use_gpu=torch.cuda.is_available())
test/single/test_ray.py:        use_gpu=torch.cuda.is_available())
test/single/test_ray.py:        use_gpu=torch.cuda.is_available())
test/single/test_ray.py:        {"CPU": 1, "GPU": int(torch.cuda.is_available())}, 4, 30, "PACK")
test/single/test_ray.py:                gpus_per_worker=int(torch.cuda.is_available()) or None,
test/single/test_ray.py:                use_gpu=torch.cuda.is_available())
test/single/test_ray.py:        num_cpus=0, num_gpus=0, placement_group_capture_child_tasks=True, placement_group=pg).remote()
test/single/test_ray.py:        setting, num_workers=3, use_gpu=torch.cuda.is_available())
test/single/test_buildkite.py:    Tests the generated GPU buildkite pipeline.
test/single/test_buildkite.py:    Compares output of .buildkite/gen-pipeline.sh with test/single/data/expected_buildkite_gpu_heads_pipeline.yaml.
test/single/test_buildkite.py:        BUILDKITE_PIPELINE_SLUG=SLUG BUILDKITE_BRANCH=BRANCH PIPELINE_MODE="GPU HEADS" .buildkite/gen-pipeline.sh > test/single/data/expected_buildkite_gpu_heads_pipeline.yaml
test/single/test_buildkite.py:    Commit `test/single/data/expected_buildkite_gpu_heads_pipeline.yaml` to get those changes into your PR.
test/single/test_buildkite.py:    def test_gen_gpu_heads_pipeline(self):
test/single/test_buildkite.py:        self.do_test_gen_pipeline(GEN_PIPELINE_FNAME, 'GPU HEADS', {}, 'WARNING:root:no commit (None) or default branch (None) given\n')
test/single/test_buildkite.py:    Tests the generated GPU buildkite pipeline.
test/single/test_buildkite.py:    Compares output of .buildkite/gen-pipeline.sh with test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml.
test/single/test_buildkite.py:        BUILDKITE_PIPELINE_SLUG=SLUG BUILDKITE_BRANCH=BRANCH PIPELINE_MODE="GPU NON HEADS" .buildkite/gen-pipeline.sh > test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml
test/single/test_buildkite.py:    Commit `test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml` to get those changes into your PR.
test/single/test_buildkite.py:    def test_gen_gpu_non_heads_pipeline(self):
test/single/test_buildkite.py:        self.do_test_gen_pipeline(GEN_PIPELINE_FNAME, 'GPU NON HEADS', {}, 'WARNING:root:no commit (None) or default branch (None) given\n')
test/single/test_buildkite.py:    def do_test_gen_pipeline(self, cmd, flavour='GPU NON HEADS', env=dict(), expected_log=''):
test/single/test_ray_elastic_v2.py:def ray_8_cpus_gpus():
test/single/test_ray_elastic_v2.py:    if "CUDA_VISIBLE_DEVICES" in os.environ:
test/single/test_ray_elastic_v2.py:        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) < 8:
test/single/test_ray_elastic_v2.py:            pytest.skip("Avoiding mismatched GPU machine.")
test/single/test_ray_elastic_v2.py:    ray.init(num_cpus=8, num_gpus=8, resources={
test/single/test_ray_elastic_v2.py:        ray.init(num_cpus=4, num_gpus=1)
test/single/test_ray_elastic_v2.py:    def test_gpu_discovery(self, ray_shutdown):
test/single/test_ray_elastic_v2.py:        ray.init(num_cpus=4, num_gpus=1)
test/single/test_ray_elastic_v2.py:        discovery = RayHostDiscovery(use_gpu=True, cpus_per_worker=1)
test/single/test_ray_elastic_v2.py:    def test_gpu_slot_discovery(self, ray_shutdown):
test/single/test_ray_elastic_v2.py:        ray.init(num_cpus=4, num_gpus=4)
test/single/test_ray_elastic_v2.py:            use_gpu=True, cpus_per_worker=1, gpus_per_worker=2)
test/single/test_ray_elastic_v2.py:            resources = {"GPU": 2, "CPU": 8}
test/single/test_ray_elastic_v2.py:        discovery = RayHostDiscovery(use_gpu=True, cpus_per_worker=1)
test/single/test_ray_elastic_v2.py:    def test_multinode_gpus_per_slot(self, monkeypatch):
test/single/test_ray_elastic_v2.py:            resources = {"GPU": 2, "CPU": 8}
test/single/test_ray_elastic_v2.py:        discovery = RayHostDiscovery(use_gpu=True, gpus_per_worker=2)
test/single/test_ray_elastic_v2.py:        discovery = RayHostDiscovery(use_gpu=True, cpus_per_worker=1)
test/single/test_ray_elastic_v2.py:def test_gpu_e2e(ray_8_cpus_gpus):
test/single/test_ray_elastic_v2.py:            min_workers=4, max_workers=4, gpus_per_worker=1, use_gpu=True, override_discovery=False)
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':docker: Build test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      build: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':pytest: Gloo Parallel PyTests (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 cd /horovod/test/parallel && (ls -1 test_*.py | xargs -n 1 horovodrun -np 2 -H localhost:2 --gloo /bin/bash /pytest.sh gloo)"
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':pytest: Gloo Single PyTests (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh gloo)"
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':pytest: Gloo Cluster PyTests (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.static.xml test_static_run.py"
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':pytest: MPI Parallel PyTests (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:  command: bash -c " HOROVOD_TEST_GPU=1 cd /horovod/test/parallel && (ls -1 test_*.py | xargs -n 1 \$(cat /mpirun_command) /bin/bash /pytest.sh mpi)"
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':pytest: MPI Single PyTests (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:  command: bash -c " HOROVOD_TEST_GPU=1 cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh mpi)"
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':pytest: MPI Cluster PyTests (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:  command: bash -c " HOROVOD_TEST_GPU=1 /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.mpi.static.xml test_static_run.py"
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow 2.0 MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow 2.0 Keras MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow 2.0 MNIST Elastic horovodrun (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':fire: Gloo PyTorch MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':muscle: Gloo MXNet2 MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':factory: Elastic Tests (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':jupyter: Run PyTests test_interactiverun (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':fire: MPI PyTorch MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':muscle: MPI MXNet2 MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':tensorflow: MPI TensorFlow 2.0 MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':tensorflow: MPI TensorFlow 2.0 Keras MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':spark: Spark TensorFlow 2.0 MNIST Data Service (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':spark: Spark Torch MNIST (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:- label: ':spark: Spark Lightning MNIST (test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0
test/single/data/expected_buildkite_gpu_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':docker: Build test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      build: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':docker: Build test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      build: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':docker: Build test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      build: test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':docker: Build test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      build: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':docker: Build test-gpu-openmpi-gloo-py3_8-tfmin-kerasmin-torchmin-mxnetmin-pysparkmin'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      build: test-gpu-openmpi-gloo-py3_8-tfmin-kerasmin-torchmin-mxnetmin-pysparkmin
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: Gloo Parallel PyTests (test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 cd /horovod/test/parallel && (ls -1 test_*.py | xargs -n 1 horovodrun -np 2 -H localhost:2 --gloo /bin/bash /pytest.sh gloo)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: Gloo Single PyTests (test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh gloo)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: Gloo Cluster PyTests (test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.static.xml test_static_run.py"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: Gloo Parallel PyTests (test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 cd /horovod/test/parallel && (ls -1 test_*.py | xargs -n 1 horovodrun -np 2 -H localhost:2 --gloo /bin/bash /pytest.sh gloo)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: Gloo Single PyTests (test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh gloo)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: Gloo Cluster PyTests (test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.static.xml test_static_run.py"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: Gloo Parallel PyTests (test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 cd /horovod/test/parallel && (ls -1 test_*.py | xargs -n 1 horovodrun -np 2 -H localhost:2 --gloo /bin/bash /pytest.sh gloo)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: Gloo Single PyTests (test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh gloo)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: Gloo Cluster PyTests (test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.static.xml test_static_run.py"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: Gloo Parallel PyTests (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 cd /horovod/test/parallel && (ls -1 test_*.py | xargs -n 1 horovodrun -np 2 -H localhost:2 --gloo /bin/bash /pytest.sh gloo)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: Gloo Single PyTests (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh gloo)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: Gloo Cluster PyTests (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.static.xml test_static_run.py"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: MPI Parallel PyTests (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c " HOROVOD_TEST_GPU=1 cd /horovod/test/parallel && (ls -1 test_*.py | xargs -n 1 \$(cat /mpirun_command) /bin/bash /pytest.sh mpi)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: MPI Single PyTests (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c " HOROVOD_TEST_GPU=1 cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh mpi)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':pytest: MPI Cluster PyTests (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c " HOROVOD_TEST_GPU=1 /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.mpi.static.xml test_static_run.py"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 cd /horovod/test/parallel && (ls -1 test_*.py | xargs -n 1 horovodrun -np 2 -H localhost:2 --gloo /bin/bash /pytest.sh gloo)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh gloo)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c "HOROVOD_TEST_GPU=1 /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.static.xml test_static_run.py"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c " HOROVOD_TEST_GPU=1 cd /horovod/test/parallel && (ls -1 test_*.py | xargs -n 1 \$(cat /mpirun_command) /bin/bash /pytest.sh mpi)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c " HOROVOD_TEST_GPU=1 cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh mpi)"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:  command: bash -c " HOROVOD_TEST_GPU=1 /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.mpi.static.xml test_static_run.py"
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 4x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow MNIST (test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: Gloo Keras MNIST (test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':fire: Gloo PyTorch MNIST horovodrun (test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':muscle: Gloo MXNet MNIST horovodrun (test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':factory: Elastic Tests (test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark Keras Rossmann Run (test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark Keras Rossmann Estimator (test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark Keras MNIST (test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark Torch MNIST (test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark Lightning MNIST (test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow 2.0 MNIST horovodrun (test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow 2.0 Keras MNIST horovodrun (test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow 2.0 MNIST Elastic horovodrun (test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow 2.0 MNIST Data Service (test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':fire: Gloo PyTorch MNIST horovodrun (test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':muscle: Gloo MXNet MNIST horovodrun (test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':factory: Elastic Tests (test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark TensorFlow 2.0 MNIST Data Service (test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark Torch MNIST (test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark Lightning MNIST (test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow 2.0 MNIST horovodrun (test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow 2.0 Keras MNIST horovodrun (test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow 2.0 MNIST Elastic horovodrun (test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':fire: Gloo PyTorch MNIST horovodrun (test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':muscle: Gloo MXNet MNIST horovodrun (test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':factory: Elastic Tests (test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark TensorFlow 2.0 MNIST Data Service (test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark Torch MNIST (test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark Lightning MNIST (test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow 2.0 MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow 2.0 Keras MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: Gloo TensorFlow 2.0 MNIST Elastic horovodrun (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':fire: Gloo PyTorch MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':muscle: Gloo MXNet MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':factory: Elastic Tests (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':jupyter: Run PyTests test_interactiverun (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':fire: MPI PyTorch MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':muscle: MPI MXNet MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: MPI TensorFlow 2.0 MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':tensorflow: MPI TensorFlow 2.0 Keras MNIST horovodrun (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark TensorFlow 2.0 MNIST Data Service (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark Torch MNIST (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:- label: ':spark: Spark Lightning MNIST (test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0)'
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:      run: test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/expected_buildkite_gpu_non_heads_pipeline.yaml:    queue: 2x-gpu-v5111
test/single/data/config.test.yaml:  num_nccl_streams: 2
test/single/test_ray_elastic.py:def ray_8_cpus_gpus():
test/single/test_ray_elastic.py:    if "CUDA_VISIBLE_DEVICES" in os.environ:
test/single/test_ray_elastic.py:        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) < 8:
test/single/test_ray_elastic.py:            pytest.skip("Avoiding mismatched GPU machine.")
test/single/test_ray_elastic.py:    ray.init(num_cpus=8, num_gpus=8, resources={
test/single/test_ray_elastic.py:        ray.init(num_cpus=4, num_gpus=1)
test/single/test_ray_elastic.py:    def test_gpu_discovery(self, ray_shutdown):
test/single/test_ray_elastic.py:        ray.init(num_cpus=4, num_gpus=1)
test/single/test_ray_elastic.py:        discovery = RayHostDiscovery(use_gpu=True, cpus_per_slot=1)
test/single/test_ray_elastic.py:    def test_gpu_slot_discovery(self, ray_shutdown):
test/single/test_ray_elastic.py:        ray.init(num_cpus=4, num_gpus=4)
test/single/test_ray_elastic.py:            use_gpu=True, cpus_per_slot=1, gpus_per_slot=2)
test/single/test_ray_elastic.py:            resources = {"GPU": 2, "CPU": 8}
test/single/test_ray_elastic.py:        discovery = RayHostDiscovery(use_gpu=True, cpus_per_slot=1)
test/single/test_ray_elastic.py:    def test_multinode_gpus_per_slot(self, monkeypatch):
test/single/test_ray_elastic.py:            resources = {"GPU": 2, "CPU": 8}
test/single/test_ray_elastic.py:        discovery = RayHostDiscovery(use_gpu=True, gpus_per_slot=2)
test/single/test_ray_elastic.py:        discovery = RayHostDiscovery(use_gpu=True, cpus_per_slot=1)
test/single/test_ray_elastic.py:def test_gpu_e2e(ray_8_cpus_gpus):
test/single/test_ray_elastic.py:            settings, gpus_per_slot=1, use_gpu=True, override_discovery=False)
test/single/test_run.py:                           '--num-nccl-streams', '2',
test/single/test_run.py:            self.assertEqual(env.get(config_parser.HOROVOD_NUM_NCCL_STREAMS), '2')
test/single/test_run.py:            self.assertEqual(args.num_nccl_streams, 2)
test/single/test_run.py:                                        '-genv NCCL_SOCKET_IFNAME=eth0,eth1 '
test/single/test_run.py:                                        '-mca btl_tcp_if_include eth0,eth1 -x NCCL_SOCKET_IFNAME=eth0,eth1 '
test/single/test_run.py:    @mock.patch('horovod.runner.util.lsf.LSFUtils.get_num_gpus', MagicMock(return_value=2))
test/single/test_run.py:    @mock.patch('horovod.runner.util.lsf.LSFUtils.get_num_gpus', MagicMock(return_value=4))
test/single/test_run.py:rank: 0: { hostname: host1; cpu: {0-3} ; gpu: * ; mem: * }
test/single/test_run.py:rank: 1: { hostname: host1; cpu: {4-7} ; gpu: * ; mem: * }
test/single/test_run.py:rank: 2: { hostname: host1; cpu: {8-11} ; gpu: * ; mem: * }
test/single/test_run.py:rank: 3: { hostname: host1; cpu: {12-15} ; gpu: * ; mem: * }
test/single/test_run.py:rank: 4: { hostname: host2; cpu: {0-3} ; gpu: * ; mem: * }
test/single/test_run.py:    @mock.patch('horovod.runner.util.lsf.LSFUtils.get_num_gpus', MagicMock(return_value=2))
test/parallel/test_tensorflow.py:# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
test/parallel/test_tensorflow.py:from common import mpi_env_rank_and_size, skip_or_fail_gpu_test
test/parallel/test_tensorflow.py:    def test_gpu_required(self):
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            skip_or_fail_gpu_test(self, "No GPUs available")
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_indexed_slices_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the allreduce correctly sums tf.IndexedSlices."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_indexed_slices_average_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the allreduce correctly averages tf.IndexedSlices."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_gpu(self):
test/parallel/test_tensorflow.py:        """Test that the allreduce works on GPUs."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:            self.assertTrue(diff <= threshold, "hvd.allreduce on GPU produces incorrect results")
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_average_gpu(self):
test/parallel/test_tensorflow.py:        """Test that the allreduce with average works on GPUs."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:            self.assertTrue(diff <= threshold, "hvd.allreduce on GPU produces incorrect results")
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_min_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the allreduce correctly minimizes 1D, 2D, 3D tensors."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_max_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the allreduce correctly maximizes 1D, 2D, 3D tensors."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_product_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the allreduce correctly multiplies 1D, 2D, 3D tensors."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_gpu_fused(self):
test/parallel/test_tensorflow.py:        """Test that the allreduce works on GPUs with Tensor Fusion.
test/parallel/test_tensorflow.py:        not support GPU memory transfers directly, as it will call MPI_Send on
test/parallel/test_tensorflow.py:        a GPU data pointer."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_multi_gpu(self):
test/parallel/test_tensorflow.py:        """Test that the allreduce works on multiple GPUs.
test/parallel/test_tensorflow.py:        not support GPU memory transfers directly, as it will call MPI_Send on
test/parallel/test_tensorflow.py:        a GPU data pointer."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:        # Only do this test if there are enough GPUs available.
test/parallel/test_tensorflow.py:        if len(tf.config.experimental.list_physical_devices('GPU')) < 2 * local_size:
test/parallel/test_tensorflow.py:            self.skipTest("Too few GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:        gpu_ids = [local_rank * 2, local_rank * 2 + 1]
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % gpu_ids[(iter + local_rank) % 2]):
test/parallel/test_tensorflow.py:                            "hvd.allreduce on GPU produces incorrect results")
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_gpu_prescale(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the allreduce correctly sums 1D, 2D, 3D tensors
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_ALLREDUCE.
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%s" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_gpu_postscale(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the allreduce correctly sums 1D, 2D, 3D tensors
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_ALLREDUCE.
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%s" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_cpu_gpu_error(self):
test/parallel/test_tensorflow.py:        perform reduction on CPU and GPU."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:        device = "/gpu:%d" % local_rank if local_rank % 2 == 0 else "/cpu:0"
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_grad_gpu(self):
test/parallel/test_tensorflow.py:        """Test the correctness of the allreduce gradient on GPU."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allreduce_average_grad_gpu(self):
test/parallel/test_tensorflow.py:        """Test the correctness of the allreduce with average gradient on GPU."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_grouped_allreduce_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the grouped allreduce correctly sums 1D, 2D, 3D tensors."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce on GPU produces incorrect results")
test/parallel/test_tensorflow.py:    def test_horovod_grouped_allreduce_indexed_slices_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the grouped allreduce correctly sums tf.IndexedSlices."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % hvd.local_rank()):
test/parallel/test_tensorflow.py:    def test_horovod_grouped_allreduce_mixed_indexed_slices_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the grouped allreduce correctly sums a mix of tensors and tf.IndexedSlices."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % hvd.local_rank()):
test/parallel/test_tensorflow.py:    def test_horovod_grouped_allreduce_average_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the grouped allreduce correctly averages 1D, 2D, 3D tensors."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % hvd.local_rank()):
test/parallel/test_tensorflow.py:    def test_horovod_grouped_allreduce_min_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the grouped allreduce correctly finds minimum of 1D, 2D, 3D tensors."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % hvd.local_rank()):
test/parallel/test_tensorflow.py:    def test_horovod_grouped_allreduce_max_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the grouped allreduce correctly finds maximum of 1D, 2D, 3D tensors."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % hvd.local_rank()):
test/parallel/test_tensorflow.py:    def test_horovod_grouped_allreduce_product_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the grouped allreduce correctly finds product of 1D, 2D, 3D tensors."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % hvd.local_rank()):
test/parallel/test_tensorflow.py:    def test_horovod_grouped_allreduce_indexed_slices_average_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the grouped allreduce correctly averages tf.IndexedSlices."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % hvd.local_rank()):
test/parallel/test_tensorflow.py:    def test_horovod_grouped_allreduce_mixed_indexed_slices_average_gpu(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the grouped allreduce correctly averages a mix of tensors and tf.IndexedSlices."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % hvd.local_rank()):
test/parallel/test_tensorflow.py:    def test_horovod_grouped_allreduce_grad_gpu(self):
test/parallel/test_tensorflow.py:        """Test the correctness of the grouped allreduce gradient on GPU."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allgather_gpu(self):
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allgather_fused_gpu(self):
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allgather_variable_size_fused_gpu(self):
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allgather_variable_size_gpu(self):
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_allgather_grad_gpu(self):
test/parallel/test_tensorflow.py:        """Test the correctness of the allgather gradient on GPU."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_broadcast_gpu(self):
test/parallel/test_tensorflow.py:        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors on GPU."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_broadcast_inplace_gpu(self):
test/parallel/test_tensorflow.py:        """Test that the inplace broadcast correctly broadcasts 1D, 2D, 3D variables on GPU."""
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:                with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_broadcast_grad_gpu(self):
test/parallel/test_tensorflow.py:        """Test the correctness of the broadcast gradient on GPU."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_alltoall_gpu(self):
test/parallel/test_tensorflow.py:        """Test that the alltoall correctly distributes 1D, 2D, and 3D tensors on GPU."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_tensorflow.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_tensorflow.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%s" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_alltoall_equal_split_gpu(self):
test/parallel/test_tensorflow.py:        """Test that the alltoall correctly distributes 1D tensors with default splitting on GPU."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_tensorflow.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_tensorflow.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%s" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_alltoall_empty_gpu(self):
test/parallel/test_tensorflow.py:        # ncclGroupEnd failed: invalid usage
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_tensorflow.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_tensorflow.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%s" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_alltoall_one_rank_sends_nothing_gpu(self):
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_tensorflow.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_tensorflow.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%s" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_alltoall_one_rank_receives_nothing_gpu(self):
test/parallel/test_tensorflow.py:        # ncclGroupEnd failed: invalid usage
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_tensorflow.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_tensorflow.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%s" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_alltoall_zero_splits_gpu(self):
test/parallel/test_tensorflow.py:        # ncclCommInitRank failed: invalid usage
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_tensorflow.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_tensorflow.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_tensorflow.py:        with tf.device("/gpu:%s" % hvd.local_rank()):
test/parallel/test_tensorflow.py:    def test_horovod_alltoall_grad_gpu(self):
test/parallel/test_tensorflow.py:        """Test the correctness of the alltoall gradient on GPU."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_tensorflow.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_tensorflow.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%s" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_alltoall_equal_split_grad_gpu(self):
test/parallel/test_tensorflow.py:        """Test the correctness of the alltoall gradient with default splitting on GPU."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_tensorflow.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_tensorflow.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%s" % local_rank):
test/parallel/test_tensorflow.py:        """Test that the hvd.join with allreduce works on GPUs."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_ALLREDUCE.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_ALLREDUCE")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:                                    "hvd.join with hvd.allreduce on GPU produces incorrect results")
test/parallel/test_tensorflow.py:    def test_horovod_syncbn_gpu(self):
test/parallel/test_tensorflow.py:        """Test that the SyncBatchNormalization implementation is correct on GPU."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:        with tf.device("/gpu:%d" % hvd.local_rank()):
test/parallel/test_tensorflow.py:    def test_horovod_reducescatter_gpu(self):
test/parallel/test_tensorflow.py:        """Test that the reducescatter works on GPUs."""
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:                            "hvd.reducescatter on GPU produces incorrect results")
test/parallel/test_tensorflow.py:    def test_horovod_reducescatter_gpu_prescale(self):
test/parallel/test_tensorflow.py:        """Test that the reducescatter works on GPUs with prescaling."""
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:                            "hvd.reducescatter on GPU produces incorrect results")
test/parallel/test_tensorflow.py:    def test_horovod_reducescatter_gpu_postscale(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the reducescatter correctly sums and scatters 1D, 2D, 3D tensors with postscaling."""
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%s" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_reducescatter_gpu_fused(self):
test/parallel/test_tensorflow.py:        """Test that the reducescatter works on GPUs with Tensor Fusion.
test/parallel/test_tensorflow.py:        not support GPU memory transfers directly, as it will call MPI_Send on
test/parallel/test_tensorflow.py:        a GPU data pointer."""
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_reducescatter_gpu_uneven(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the reducescatter correctly sums and scatters tensors that cannot
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_reducescatter_gpu_uneven_fused(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the reducescatter correctly sums and scatters tensors that cannot
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_reducescatter_grad_gpu(self):
test/parallel/test_tensorflow.py:        """Test the correctness of the reducescatter gradient on GPU."""
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_grouped_reducescatter_gpu(self):
test/parallel/test_tensorflow.py:        """Test that the grouped reducescatter works on GPUs."""
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:                            "hvd.grouped_reducescatter on GPU produces incorrect results")
test/parallel/test_tensorflow.py:    def test_horovod_grouped_reducescatter_gpu_prescale(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the grouped reducescatter correctly sums and scatters 1D, 2D, 3D tensors with prescaling."""
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_grouped_reducescatter_gpu_postscale(self):
test/parallel/test_tensorflow.py:        """Test on GPU that the grouped reducescatter correctly sums and scatters 1D, 2D, 3D tensors with postscaling"""
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:    def test_horovod_grouped_allgather_gpu(self):
test/parallel/test_tensorflow.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            self.skipTest(("No GPUs available"))
test/parallel/test_tensorflow.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:            if tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:                with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:        if tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow.py:        if tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_xla_process_sets.py:# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
test/parallel/test_xla_process_sets.py:slowdowns in all Horovod operations, especially on GPU-equipped AWS instances. For that reason we separate
test/parallel/test_xla_process_sets.py:    def test_horovod_allreduce_gpu_process_sets(self):
test/parallel/test_xla_process_sets.py:        """ Test on XLA/GPU that allreduce correctly sums if restricted to non-global process sets"""
test/parallel/test_xla_process_sets.py:        # Only do this test if there are GPUs available.
test/parallel/test_xla_process_sets.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_xla_process_sets.py:            self.skipTest("No GPUs available")
test/parallel/test_xla_process_sets.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_xla_process_sets.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_xla_process_sets.py:        def allreduce_gpu_process_set(self, dtype, dim):
test/parallel/test_xla_process_sets.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_xla_process_sets.py:                    allreduce_gpu_process_set, jit_compile=True)(self, dtype, dim)
test/parallel/test_mxnet2.py:# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
test/parallel/test_mxnet2.py:    @unittest.skipUnless(has_gpu, "no gpu detected")
test/parallel/test_mxnet2.py:        ctx = mx.gpu(rank)
test/parallel/base_test_tensorflow.py:# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
test/parallel/base_test_tensorflow.py:    config.gpu_options.allow_growth = True
test/parallel/base_test_tensorflow.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
test/parallel/base_test_tensorflow.py:    for gpu in gpus:
test/parallel/base_test_tensorflow.py:        tf.config.experimental.set_memory_growth(gpu, True)
test/parallel/test_tensorflow2_keras.py:        gpus = tf.config.experimental.list_physical_devices('GPU')
test/parallel/test_tensorflow2_keras.py:        for gpu in gpus:
test/parallel/test_tensorflow2_keras.py:            tf.config.experimental.set_memory_growth(gpu, True)
test/parallel/test_tensorflow2_keras.py:        if gpus:
test/parallel/test_tensorflow2_keras.py:                gpus[hvd.local_rank()], 'GPU')
test/parallel/test_tensorflow_keras.py:        self.config.gpu_options.allow_growth = True
test/parallel/test_tensorflow_keras.py:        self.config.gpu_options.visible_device_list = str(hvd.local_rank())
test/parallel/test_xla.py:# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
test/parallel/test_xla.py:    config.gpu_options.allow_growth = True
test/parallel/test_xla.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
test/parallel/test_xla.py:    for gpu in gpus:
test/parallel/test_xla.py:        tf.config.experimental.set_memory_growth(gpu, True)
test/parallel/test_xla.py:    def test_horovod_allreduce_gpu(self):
test/parallel/test_xla.py:        """Test that the allreduce works on XLA/GPUs."""
test/parallel/test_xla.py:        # Only do this test if there are GPUs available.
test/parallel/test_xla.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_xla.py:            self.skipTest(("No GPUs available"))
test/parallel/test_xla.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_xla.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_xla.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_xla.py:                "hvd.allreduce on XLA/GPU produces incorrect results")
test/parallel/test_xla.py:    def test_horovod_allreduce_gpu_prescale(self):
test/parallel/test_xla.py:        """Test on XLA/GPU that the allreduce correctly sums 1D, 2D, 3D tensors
test/parallel/test_xla.py:        # Only do this test if there are GPUs available.
test/parallel/test_xla.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_xla.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_ALLREDUCE.
test/parallel/test_xla.py:            with tf.device("/gpu:%s" % local_rank):
test/parallel/test_xla.py:    def test_horovod_allreduce_gpu_postscale(self):
test/parallel/test_xla.py:        """Test on XLA/GPU that the allreduce correctly sums 1D, 2D, 3D tensors
test/parallel/test_xla.py:        # Only do this test if there are GPUs available.
test/parallel/test_xla.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_xla.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_ALLREDUCE.
test/parallel/test_xla.py:            with tf.device("/gpu:%s" % local_rank):
test/parallel/test_xla.py:    def test_horovod_allreduce_grad_gpu(self):
test/parallel/test_xla.py:        """Test the correctness of the allreduce gradient on XLA/GPU."""
test/parallel/test_xla.py:        # Only do this test if there are GPUs available.
test/parallel/test_xla.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_xla.py:            self.skipTest(("No GPUs available"))
test/parallel/test_xla.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_xla.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_xla.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_xla.py:    def test_horovod_allreduce_average_grad_gpu(self):
test/parallel/test_xla.py:        """Test the correctness of the allreduce with average gradient on XLA/GPU."""
test/parallel/test_xla.py:        # Only do this test if there are GPUs available.
test/parallel/test_xla.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_xla.py:            self.skipTest(("No GPUs available"))
test/parallel/test_xla.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_xla.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_xla.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_adasum_tensorflow.py:    config.gpu_options.allow_growth = True
test/parallel/test_adasum_tensorflow.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
test/parallel/test_adasum_tensorflow.py:    for gpu in gpus:
test/parallel/test_adasum_tensorflow.py:        tf.config.experimental.set_memory_growth(gpu, True)
test/parallel/test_adasum_tensorflow.py:    def test_horovod_adasum_multiple_allreduce_gpu_nccl(self):
test/parallel/test_adasum_tensorflow.py:        """Test on GPU using NCCL that the Adasum correctly computes 2D tensors."""
test/parallel/test_adasum_tensorflow.py:        if not hvd.mpi_enabled() or not hvd.gpu_available('tensorflow') or not hvd.nccl_built():
test/parallel/test_adasum_tensorflow.py:            self.skipTest("MPI, GPU or NCCL not available")
test/parallel/test_adasum_tensorflow.py:            with tf.device("/gpu:{}".format(hvd.local_rank())):
test/parallel/test_tensorflow2_keras_process_sets.py:slowdowns in all Horovod operations, especially on GPU-equipped AWS instances. For that reason we collect
test/parallel/test_tensorflow2_keras_process_sets.py:        gpus = tf.config.experimental.list_physical_devices('GPU')
test/parallel/test_tensorflow2_keras_process_sets.py:        for gpu in gpus:
test/parallel/test_tensorflow2_keras_process_sets.py:            tf.config.experimental.set_memory_growth(gpu, True)
test/parallel/test_tensorflow2_keras_process_sets.py:        if gpus:
test/parallel/test_tensorflow2_keras_process_sets.py:                gpus[hvd.local_rank()], 'GPU')
test/parallel/test_keras.py:        self.config.gpu_options.allow_growth = True
test/parallel/test_keras.py:        self.config.gpu_options.visible_device_list = str(hvd.local_rank())
test/parallel/base_test_mxnet.py:# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
test/parallel/base_test_mxnet.py:from common import skip_or_fail_gpu_test
test/parallel/base_test_mxnet.py:        has_gpu = mx.context.num_gpus() > 0
test/parallel/base_test_mxnet.py:        has_gpu = mx.device.num_gpus() > 0
test/parallel/base_test_mxnet.py:    has_gpu = False
test/parallel/base_test_mxnet.py:        if has_gpu:
test/parallel/base_test_mxnet.py:            return mx.gpu(hvd.local_rank())
test/parallel/base_test_mxnet.py:    def test_gpu_required(self):
test/parallel/base_test_mxnet.py:        if not has_gpu:
test/parallel/base_test_mxnet.py:            skip_or_fail_gpu_test(self, "No GPUs available")
test/parallel/base_test_mxnet.py:            # MXNet uses gpu_id as part of the seed, so to get identical seeds
test/parallel/base_test_mxnet.py:            # MXNet uses gpu_id as part of the seed, so to get identical seeds
test/parallel/base_test_mxnet.py:    @unittest.skipUnless(has_gpu, "no gpu detected")
test/parallel/base_test_mxnet.py:    def test_horovod_allreduce_cpu_gpu_error(self):
test/parallel/base_test_mxnet.py:           perform reduction on CPU and GPU."""
test/parallel/base_test_mxnet.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/base_test_mxnet.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/base_test_mxnet.py:            ctx = mx.gpu(hvd.rank())
test/parallel/base_test_mxnet.py:            assert False, 'hvd.allreduce did not throw cpu-gpu error'
test/parallel/base_test_mxnet.py:    @unittest.skipUnless(has_gpu, "no gpu detected")
test/parallel/base_test_mxnet.py:    def test_horovod_grouped_allreduce_cpu_gpu_error(self):
test/parallel/base_test_mxnet.py:           list contains a mix of tensors on CPU and GPU."""
test/parallel/base_test_mxnet.py:        tensors = [mx.nd.ones(shape=[10], ctx=mx.gpu(local_rank) if i % 2
test/parallel/base_test_mxnet.py:            assert False, 'hvd.grouped_allreduce did not throw cpu-gpu error'
test/parallel/base_test_mxnet.py:    @unittest.skipUnless(has_gpu, "no gpu detected")
test/parallel/base_test_mxnet.py:    def test_horovod_grouped_allgather_cpu_gpu_error(self):
test/parallel/base_test_mxnet.py:           list contains a mix of tensors on CPU and GPU."""
test/parallel/base_test_mxnet.py:        tensors = [mx.nd.ones(shape=[10], ctx=mx.gpu(local_rank) if i % 2
test/parallel/base_test_mxnet.py:            assert False, 'hvd.grouped_allgather did not throw cpu-gpu error'
test/parallel/base_test_mxnet.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/base_test_mxnet.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/base_test_mxnet.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/base_test_mxnet.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/base_test_mxnet.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/base_test_mxnet.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/base_test_mxnet.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/base_test_mxnet.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/base_test_mxnet.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/base_test_mxnet.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/base_test_mxnet.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/base_test_mxnet.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/base_test_mxnet.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/base_test_mxnet.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/base_test_mxnet.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/base_test_mxnet.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/base_test_mxnet.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/base_test_mxnet.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/base_test_mxnet.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/base_test_mxnet.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/base_test_mxnet.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/base_test_mxnet.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/base_test_mxnet.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/base_test_mxnet.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/base_test_mxnet.py:    @unittest.skipUnless(has_gpu, "no gpu detected")
test/parallel/base_test_mxnet.py:        ctx = mx.gpu(rank)
test/parallel/base_test_mxnet.py:            # MXNet uses gpu_id as part of the seed, so to get identical seeds
test/parallel/base_test_mxnet.py:            # MXNet uses gpu_id as part of the seed, so to get identical seeds
test/parallel/base_test_mxnet.py:            # MXNet uses gpu_id as part of the seed, so to get identical seeds
test/parallel/base_test_mxnet.py:            # MXNet uses gpu_id as part of the seed, so to get identical seeds
test/parallel/base_test_mxnet.py:            # MXNet uses gpu_id as part of the seed, so to get identical seeds
test/parallel/base_test_mxnet.py:            # MXNet uses gpu_id as part of the seed, so to get identical seeds
test/parallel/base_test_mxnet.py:            # MXNet uses gpu_id as part of the seed, so to get identical seeds
test/parallel/base_test_mxnet.py:            # MXNet uses gpu_id as part of the seed, so to get identical seeds
test/parallel/base_test_mxnet.py:            # MXNet uses gpu_id as part of the seed, so to get identical seeds
test/parallel/base_test_mxnet.py:            # MXNet uses gpu_id as part of the seed, so to get identical seeds
test/parallel/test_compute_worker.py:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
test/parallel/test_compute_worker.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
test/parallel/test_compute_worker.py:    for gpu in gpus:
test/parallel/test_compute_worker.py:        tf.config.experimental.set_memory_growth(gpu, True)
test/parallel/test_compute_worker.py:    if gpus:
test/parallel/test_compute_worker.py:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
test/parallel/test_tensorflow_process_sets.py:slowdowns in all Horovod operations, especially on GPU-equipped AWS instances. For that reason we collect
test/parallel/test_tensorflow_process_sets.py:    def test_horovod_allreduce_gpu_process_sets(self):
test/parallel/test_tensorflow_process_sets.py:        """ Test on GPU that allreduce correctly sums if restricted to non-global process sets"""
test/parallel/test_tensorflow_process_sets.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow_process_sets.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow_process_sets.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow_process_sets.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow_process_sets.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow_process_sets.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow_process_sets.py:    def test_horovod_grouped_allreduce_gpu_process_sets(self):
test/parallel/test_tensorflow_process_sets.py:        """Test on GPU that the grouped allreduce correctly sums if restricted to non-global process sets"""
test/parallel/test_tensorflow_process_sets.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow_process_sets.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow_process_sets.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow_process_sets.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow_process_sets.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow_process_sets.py:    def test_horovod_allgather_gpu_process_sets(self):
test/parallel/test_tensorflow_process_sets.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow_process_sets.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow_process_sets.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow_process_sets.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow_process_sets.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow_process_sets.py:    def test_horovod_broadcast_gpu_process_sets(self):
test/parallel/test_tensorflow_process_sets.py:        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors on GPU
test/parallel/test_tensorflow_process_sets.py:        # Only do this test if there are GPUs available.
test/parallel/test_tensorflow_process_sets.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow_process_sets.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow_process_sets.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow_process_sets.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow_process_sets.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow_process_sets.py:    def test_horovod_alltoall_gpu_process_sets(self):
test/parallel/test_tensorflow_process_sets.py:        """Test that the GPU alltoall on restricted process sets correctly distributes 1D, 2D, and 3D tensors."""
test/parallel/test_tensorflow_process_sets.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow_process_sets.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow_process_sets.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow_process_sets.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow_process_sets.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_tensorflow_process_sets.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_tensorflow_process_sets.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_tensorflow_process_sets.py:            with tf.device("/gpu:%s" % local_rank):
test/parallel/test_tensorflow_process_sets.py:        with self.test_session(use_gpu=False) as sess:
test/parallel/test_tensorflow_process_sets.py:    def test_horovod_reducescatter_gpu_process_sets(self):
test/parallel/test_tensorflow_process_sets.py:        """Test that the reducescatter works on GPUs if restricted to non-global process sets."""
test/parallel/test_tensorflow_process_sets.py:        if not tf.test.is_gpu_available(cuda_only=True):
test/parallel/test_tensorflow_process_sets.py:            self.skipTest("No GPUs available")
test/parallel/test_tensorflow_process_sets.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_tensorflow_process_sets.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_tensorflow_process_sets.py:            with tf.device("/gpu:%d" % local_rank):
test/parallel/test_tensorflow_process_sets.py:                            "hvd.reducescatter on GPU produces incorrect results")
test/parallel/test_tensorflow_process_sets_dynamic.py:slowdowns in all Horovod operations, especially on GPU-equipped AWS instances. For that reason we collect
test/parallel/test_adasum_pytorch.py:    # Only do this test if there are GPUs available.
test/parallel/test_adasum_pytorch.py:    if not hvd.mpi_enabled() or not torch.cuda.is_available():
test/parallel/test_adasum_pytorch.py:      self.skipTest("No GPUs available")
test/parallel/test_adasum_pytorch.py:    device = torch.device('cuda:{}'.format(hvd.local_rank()))
test/parallel/test_adasum_pytorch.py:      denominator = local_size if hvd.nccl_built() else 1
test/parallel/test_adasum_pytorch.py:    # Only do this test if there are GPUs available.
test/parallel/test_adasum_pytorch.py:    if not hvd.mpi_enabled() or not torch.cuda.is_available():
test/parallel/test_adasum_pytorch.py:      self.skipTest("No GPUs available")
test/parallel/test_adasum_pytorch.py:    device = torch.device('cuda:{}'.format(hvd.local_rank()))
test/parallel/test_adasum_pytorch.py:    device = torch.device('cuda:{}'.format(hvd.local_rank())) if torch.cuda.is_available() else torch.device('cpu')
test/parallel/test_adasum_pytorch.py:    device = torch.device('cuda:{}'.format(hvd.local_rank())) if torch.cuda.is_available() else torch.device('cpu')
test/parallel/test_torch.py:# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
test/parallel/test_torch.py:from common import mpi_env_rank_and_size, skip_or_fail_gpu_test, temppath
test/parallel/test_torch.py:            if value.dtype in [torch.float16, torch.HalfTensor] and not value.is_cuda:
test/parallel/test_torch.py:        if dtype.is_cuda:
test/parallel/test_torch.py:            return tensor.cuda(hvd.local_rank()).type(dtype)
test/parallel/test_torch.py:    def test_gpu_required(self):
test/parallel/test_torch.py:        if not torch.cuda.is_available():
test/parallel/test_torch.py:            skip_or_fail_gpu_test(self, "No GPUs available")
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:    def test_horovod_allreduce_multi_gpu(self):
test/parallel/test_torch.py:        """Test that the allreduce works on multiple GPUs."""
test/parallel/test_torch.py:        # Only do this test if there are GPUs available.
test/parallel/test_torch.py:        if not torch.cuda.is_available():
test/parallel/test_torch.py:            self.skipTest("No GPUs available")
test/parallel/test_torch.py:        # Skip the test if there are not enough GPUs.
test/parallel/test_torch.py:        if torch.cuda.device_count() < hvd.local_size() * 2:
test/parallel/test_torch.py:            self.skipTest("Not enough GPUs available")
test/parallel/test_torch.py:        dtypes = [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                  torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                  torch.cuda.HalfTensor]
test/parallel/test_torch.py:            tensor = tensor.cuda(device).type(dtype)
test/parallel/test_torch.py:            if size <= 3 or dtype in [torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                     torch.cuda.IntTensor, torch.cuda.LongTensor]
test/parallel/test_torch.py:        half_types = [torch.HalfTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:            factor = factor.cuda(hvd.local_rank()) if dtype.is_cuda else factor
test/parallel/test_torch.py:            if dtype.is_cuda and not int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                     torch.cuda.IntTensor, torch.cuda.LongTensor]
test/parallel/test_torch.py:        half_types = [torch.HalfTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:            factor = factor.cuda(hvd.local_rank()) if dtype.is_cuda else factor
test/parallel/test_torch.py:            if dtype.is_cuda and not int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:    def test_horovod_allreduce_cpu_gpu_error(self):
test/parallel/test_torch.py:        perform reduction on CPU and GPU."""
test/parallel/test_torch.py:        # Only do this test if there are GPUs available.
test/parallel/test_torch.py:        if not torch.cuda.is_available():
test/parallel/test_torch.py:            self.skipTest("No GPUs available")
test/parallel/test_torch.py:            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
test/parallel/test_torch.py:            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
test/parallel/test_torch.py:            tensor = torch.cuda.FloatTensor(*dims)
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:    def test_horovod_grouped_allreduce_cpu_gpu_error(self):
test/parallel/test_torch.py:        list contains a mix of tensors on CPU and GPU."""
test/parallel/test_torch.py:        # Only do this test if there are GPUs available.
test/parallel/test_torch.py:        if not torch.cuda.is_available():
test/parallel/test_torch.py:            self.skipTest("No GPUs available")
test/parallel/test_torch.py:        tensors = [torch.FloatTensor(10) if i % 2 else torch.cuda.FloatTensor(10)  for i in range(5)]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
test/parallel/test_torch.py:                       torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
test/parallel/test_torch.py:                       torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
test/parallel/test_torch.py:                       torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
test/parallel/test_torch.py:                       torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
test/parallel/test_torch.py:                       torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
test/parallel/test_torch.py:                       torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
test/parallel/test_torch.py:                       torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
test/parallel/test_torch.py:                       torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
test/parallel/test_torch.py:                       torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_torch.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_torch.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
test/parallel/test_torch.py:                       torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_torch.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_torch.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
test/parallel/test_torch.py:                       torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:    def test_horovod_alltoall_splits_on_gpu(self):
test/parallel/test_torch.py:        """Test that the alltoall works correctly when the splits argument is a tensor on GPU."""
test/parallel/test_torch.py:        if not torch.cuda.is_available():
test/parallel/test_torch.py:            self.skipTest("No GPUs available")
test/parallel/test_torch.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_torch.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_torch.py:        dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
test/parallel/test_torch.py:                   torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                   torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                   torch.cuda.HalfTensor]
test/parallel/test_torch.py:            splits = torch.tensor([rank + 1] * size, dtype=torch.int32, device="cuda")
test/parallel/test_torch.py:            self.assertEqual(received_splits.device.type, "cuda", "received_splits should be on GPU here")
test/parallel/test_torch.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_torch.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_torch.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
test/parallel/test_torch.py:                       torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_torch.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_torch.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_torch.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_torch.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_torch.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_torch.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_torch.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_torch.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_torch.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_torch.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_torch.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_torch.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_torch.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_torch.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_torch.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_torch.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_torch.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_torch.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_torch.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        # This test does not apply if NCCL version < 2.7.0
test/parallel/test_torch.py:        if hvd.nccl_built() and hvd.nccl_built() < 2700:
test/parallel/test_torch.py:            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:    def test_broadcast_state_gpu(self):
test/parallel/test_torch.py:        # Only do this test if there are GPUs available.
test/parallel/test_torch.py:        if not torch.cuda.is_available():
test/parallel/test_torch.py:            self.skipTest("No GPUs available")
test/parallel/test_torch.py:            torch.set_default_tensor_type(torch.cuda.FloatTensor)
test/parallel/test_torch.py:        """Test that tensors on different GPUs are supported."""
test/parallel/test_torch.py:        # Only do this test if there are GPUs available.
test/parallel/test_torch.py:        if not torch.cuda.is_available():
test/parallel/test_torch.py:            self.skipTest("No GPUs available")
test/parallel/test_torch.py:        # Skip the test if there are not enough GPUs.
test/parallel/test_torch.py:        if torch.cuda.device_count() < hvd.local_size() * 2:
test/parallel/test_torch.py:            self.skipTest("Not enough GPUs available")
test/parallel/test_torch.py:                # Place parts of model on different GPUs.
test/parallel/test_torch.py:                self.conv1 = torch.nn.Conv2d(1, 100, 1).cuda(first_device)
test/parallel/test_torch.py:                self.conv2 = torch.nn.Conv2d(100, 1, 1).cuda(second_device)
test/parallel/test_torch.py:                x = x.cuda(first_device)
test/parallel/test_torch.py:                x = x.cuda(second_device)
test/parallel/test_torch.py:        if not torch.cuda.is_available():
test/parallel/test_torch.py:            self.skipTest("No GPUs available")
test/parallel/test_torch.py:                self.conv1 = torch.nn.Conv2d(1, 100, 1).cuda(local_rank)
test/parallel/test_torch.py:                self.conv2 = torch.nn.Conv2d(100, 1, 1).cuda(local_rank)
test/parallel/test_torch.py:                x = x.cuda(local_rank)
test/parallel/test_torch.py:                x = x.cuda(local_rank)
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        integral_types = [torch.IntTensor, torch.LongTensor, torch.cuda.IntTensor, torch.cuda.LongTensor]
test/parallel/test_torch.py:                if dtype.is_cuda:
test/parallel/test_torch.py:                if dtype.is_cuda:
test/parallel/test_torch.py:                if torch.cuda.is_available():
test/parallel/test_torch.py:                if torch.cuda.is_available():
test/parallel/test_torch.py:                if torch.cuda.is_available():
test/parallel/test_torch.py:        if not torch.cuda.is_available():
test/parallel/test_torch.py:            self.skipTest("No GPUs available")
test/parallel/test_torch.py:            sync_bn.cuda(hvd.local_rank())
test/parallel/test_torch.py:            bn.cuda(hvd.local_rank())
test/parallel/test_torch.py:            ts = ts.cuda(hvd.local_rank()).float()
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                     torch.cuda.IntTensor, torch.cuda.LongTensor]
test/parallel/test_torch.py:        half_types = [torch.HalfTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:            factor = factor.cuda(hvd.local_rank()) if dtype.is_cuda else factor
test/parallel/test_torch.py:            if dtype.is_cuda and not int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                     torch.cuda.IntTensor, torch.cuda.LongTensor]
test/parallel/test_torch.py:        half_types = [torch.HalfTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:            factor = factor.cuda(hvd.local_rank()) if dtype.is_cuda else factor
test/parallel/test_torch.py:            if dtype.is_cuda and not int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                                 torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                     torch.cuda.IntTensor, torch.cuda.LongTensor]
test/parallel/test_torch.py:        half_types = [torch.HalfTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:            factor = factor.cuda(hvd.local_rank()) if dtype.is_cuda else factor
test/parallel/test_torch.py:            if dtype.is_cuda and not int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                     torch.cuda.IntTensor, torch.cuda.LongTensor]
test/parallel/test_torch.py:        half_types = [torch.HalfTensor, torch.cuda.HalfTensor]
test/parallel/test_torch.py:            factor = factor.cuda(hvd.local_rank()) if dtype.is_cuda else factor
test/parallel/test_torch.py:            if dtype.is_cuda and not int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
test/parallel/test_torch.py:                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor,
test/parallel/test_torch.py:                       torch.cuda.HalfTensor]
test/parallel/test_torch.py:                                      torch.cuda.IntTensor, torch.cuda.LongTensor]:
test/parallel/test_torch.py:        if torch.cuda.is_available():
test/parallel/test_torch.py:            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor]
test/parallel/test_mxnet1.py:# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
test/parallel/test_mxnet1.py:    @unittest.skipUnless(has_gpu, "no gpu detected")
test/parallel/test_mxnet1.py:    def test_horovod_grouped_allreduce_cpu_gpu_error(self):
test/parallel/test_mxnet1.py:           list contains a mix of tensors on CPU and GPU."""
test/parallel/test_mxnet1.py:        super(MX1Tests, self).test_horovod_grouped_allreduce_cpu_gpu_error()
test/parallel/test_mxnet1.py:    @unittest.skipUnless(has_gpu, "no gpu detected")
test/parallel/test_mxnet1.py:    def test_horovod_grouped_allgather_cpu_gpu_error(self):
test/parallel/test_mxnet1.py:           list contains a mix of tensors on CPU and GPU."""
test/parallel/test_mxnet1.py:        super(MX1Tests, self).test_horovod_grouped_allgather_cpu_gpu_error()
test/utils/spark_common.py:def spark_session(app, cores=2, gpus=0, max_failures=1, *args):
test/utils/spark_common.py:        # start a single worker with given cores when gpus are present
test/utils/spark_common.py:        # max failures are ignored when gpus in that case
test/utils/spark_common.py:        master = 'local-cluster[1,{},1024]'.format(cores) if gpus > 0 \
test/utils/spark_common.py:            if gpus > 0:
test/utils/spark_common.py:                    addresses = ', '.join('\\"{}\\"'.format(i) for i in range(gpus))
test/utils/spark_common.py:                    temp_file.write(b'echo {\\"name\\": \\"gpu\\", \\"addresses\\": [' +
test/utils/spark_common.py:                # the single worker takes all gpus discovered, and a single executor will get them
test/utils/spark_common.py:                # each task on that executor will get a single gpu
test/utils/spark_common.py:                    ('spark.worker.resource.gpu.discoveryScript', temp_filename),
test/utils/spark_common.py:                    ('spark.worker.resource.gpu.amount', str(gpus)),
test/utils/spark_common.py:                    ('spark.task.resource.gpu.amount', '1'),
test/utils/spark_common.py:                    ('spark.executor.resource.gpu.amount', str(gpus)),
test/utils/common.py:def skip_or_fail_gpu_test(test, message):
test/utils/common.py:    """Fails the test if GPUs are required, otherwise skips."""
test/utils/common.py:    if int(os.environ.get('HOROVOD_TEST_GPU', 0)):
test/integration/data/elastic_tensorflow_main.py:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
test/integration/data/elastic_tensorflow_main.py:config.gpu_options.allow_growth = False
test/integration/data/elastic_tensorflow_main.py:config.gpu_options.visible_device_list = ''
test/integration/data/elastic_tensorflow2_main.py:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
test/integration/data/elastic_tensorflow_keras_main.py:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
test/integration/data/elastic_tensorflow_keras_main.py:config.gpu_options.allow_growth = False
test/integration/data/elastic_tensorflow_keras_main.py:config.gpu_options.visible_device_list = ''
test/integration/data/elastic_tensorflow_keras_main.py:    # Horovod: adjust number of steps based on number of GPUs.
test/integration/test_spark_keras.py:# Modifications copyright (C) 2022, NVIDIA CORPORATION. All rights reserved.
test/integration/test_spark_keras.py:                    use_gpu=False,
test/integration/test_spark_keras.py:                assert not keras_estimator.getUseGpu()
test/integration/test_spark_keras.py:                    use_gpu=True)
test/integration/test_spark_keras.py:                    use_gpu=True)
test/integration/test_spark_keras.py:    @mock.patch('horovod.spark.keras.remote._pin_gpu_fn')
test/integration/test_spark_keras.py:    def test_restore_from_checkpoint(self, mock_fit_fn, mock_pin_gpu_fn):
test/integration/test_spark_keras.py:        mock_pin_gpu_fn.return_value = mock.Mock()
test/integration/test_spark_keras.py:    @mock.patch('horovod.spark.keras.remote._pin_gpu_fn')
test/integration/test_spark_keras.py:    def test_keras_direct_parquet_train(self, mock_fit_fn, mock_pin_gpu_fn):
test/integration/test_spark_keras.py:        mock_pin_gpu_fn.return_value = mock.Mock()
test/integration/test_spark_keras.py:    @mock.patch('horovod.spark.keras.remote._pin_gpu_fn')
test/integration/test_spark_keras.py:    def test_keras_model_checkpoint_callback(self, mock_fit_fn, mock_pin_gpu_fn):
test/integration/test_spark_keras.py:        mock_pin_gpu_fn.return_value = mock.Mock()
test/integration/test_spark.py:    with spark_session('test_get_available_devices', gpus=2):
test/integration/test_spark.py:                                    '-mca btl_tcp_if_include [^ ]+ -x NCCL_SOCKET_IFNAME=[^ ]+  '
test/integration/test_spark.py:                                    '-x NCCL_DEBUG=INFO '
test/integration/test_spark.py:        for replacement in ['-H [^ ]+', '-mca btl_tcp_if_include [^ ]+', '-x NCCL_SOCKET_IFNAME=[^ ]+',
test/integration/test_spark.py:                                'NCCL_SOCKET_IFNAME=[^ ]+ '
test/integration/test_spark.py:                                'NCCL_SOCKET_IFNAME=[^ ]+',
CHANGELOG.md:- PyTorch: Fixed build on ROCm. ([#3928](https://github.com/horovod/horovod/pull/3928))
CHANGELOG.md:- `CUDA_VISIBLE_DEVICES` environment variable is no longer passed to remote nodes. ([#3865](https://github.com/horovod/horovod/pull/3865))
CHANGELOG.md:- Fixed build with ROCm. ([#3839](https://github.com/horovod/horovod/pull/3839), [#3848](https://github.com/horovod/horovod/pull/3848))
CHANGELOG.md:- Fixed linking recent NCCL by defaulting CUDA runtime library linkage to static and ensuring that weak symbols are overridden. ([#3867](https://github.com/horovod/horovod/pull/3867), [#3846](https://github.com/horovod/horovod/pull/3846))
CHANGELOG.md:- Updated with_device functions in MXNet and PyTorch to skip unnecessary cudaSetDevice calls. ([#3912](https://github.com/horovod/horovod/pull/3912))
CHANGELOG.md:- Added `HOROVOD_SPARK_USE_LOCAL_RANK_GPU_INDEX` environment variable to ignore GPU device indices assigned by Spark and always use local rank GPU device in Spark estimators. ([#3737](https://github.com/horovod/horovod/pull/3737))
CHANGELOG.md:- Improved NCCL performance for fused allgather operations through padding for better memory alignment. ([#3727](https://github.com/horovod/horovod/pull/3727))
CHANGELOG.md:- ROCm: Fixed GPU MPI operations support in build. ([#3746](https://github.com/horovod/horovod/pull/3746))
CHANGELOG.md:- Fixed memory leak in `MPI_GPUAllgather`. ([#3727](https://github.com/horovod/horovod/pull/3727))
CHANGELOG.md:- PyTorch, ROCm: Fixed allreduce average on process sets. ([#3815](https://github.com/horovod/horovod/pull/3815))
CHANGELOG.md:- Enabled use of native `ncclAvg` op for NCCL allreduces. ([#3646](https://github.com/horovod/horovod/pull/3646))
CHANGELOG.md:- Added 2D torus `allreduce` using NCCL. ([#3608](https://github.com/horovod/horovod/pull/3608))
CHANGELOG.md:- Added support for batched memory copies in `GPUAllgather`. ([#3590](https://github.com/horovod/horovod/pull/3590))
CHANGELOG.md:- Added support for batched memory copies in `GPUReducescatter`. ([#3621](https://github.com/horovod/horovod/pull/3621))
CHANGELOG.md:- ROCm: Enabled `alltoall`. ([#3654](https://github.com/horovod/horovod/pull/3654))
CHANGELOG.md:- Fixed `FuseResponses()` on `BATCHED_D2D_PADDING` edge cases for Reducescatter and/or ROCm. ([#3621](https://github.com/horovod/horovod/pull/3621))
CHANGELOG.md:- PyTorch on GPUs without GPU operations: Fixed grouped allreduce to set CPU device in tensor table. ([#3594](https://github.com/horovod/horovod/pull/3594))
CHANGELOG.md:- Build: Modify regex match for CUDA|ROCm in `FindPytorch.cmake`. ([#3593](https://github.com/horovod/horovod/pull/3593))
CHANGELOG.md:- Build: Fixed ROCm-specific build failure. ([#3630](https://github.com/horovod/horovod/pull/3630))
CHANGELOG.md:- Added `hvd.reducescatter()` operation with implementations in NCCL, MPI, and Gloo. ([#3299](https://github.com/horovod/horovod/pull/3299), [#3574](https://github.com/horovod/horovod/pull/3574))
CHANGELOG.md:- Added AMD GPU XLA Op Implementation. ([#3486](https://github.com/horovod/horovod/pull/3486))
CHANGELOG.md:- Spark Estimator: Add option whether to use GPUs at all. ([#3526](https://github.com/horovod/horovod/pull/3526))
CHANGELOG.md:- TensorFlow: Make TensorFlow output allocations asynchronous when using NCCL backend. ([#3464](https://github.com/horovod/horovod/pull/3464))
CHANGELOG.md:- Fallback to NCCL shared lib if static one is not found. ([#3500]((https://github.com/horovod/horovod/pull/3500))
CHANGELOG.md:- Fix ignored cuda arch flags ([#3462]((https://github.com/horovod/horovod/pull/3462))
CHANGELOG.md:- Extended CMake build script to often find CUDA even if `nvcc` is not in `$PATH`. ([#3444](https://github.com/horovod/horovod/pull/3444))
CHANGELOG.md:- Moved to CMake version 3.13 with first-class CUDA language support and re-enabled parallelized builds. Uses a temporary installation of CMake if CMake 3.13 is not found. ([#3261](https://github.com/horovod/horovod/pull/3261), [#3371](https://github.com/horovod/horovod/pull/3371))
CHANGELOG.md:- Elastic: Improved handling NCCL errors under elastic scenario. ([#3112](https://github.com/horovod/horovod/pull/3112))
CHANGELOG.md:- Added fused buffer scaling and unpack/pack kernels on GPU. ([#2973](https://github.com/horovod/horovod/pull/2973))
CHANGELOG.md:- Added support for NCCL on CUDA 11.4. ([#3182](https://github.com/horovod/horovod/issues/3182))
CHANGELOG.md:- Implemented more asynchronous dependency handling on GPU. ([#2963](https://github.com/horovod/horovod/pull/2963))
CHANGELOG.md:- Added FP16 support for GPU tensor in mxnet. ([#2915](https://github.com/horovod/horovod/pull/2915))
CHANGELOG.md:- Fixed building Horovod for ROCm PyTorch with newer hipify script. ([#2360](https://github.com/horovod/horovod/pull/2360))
CHANGELOG.md:- Adding support for batched D2D memcopy kernel on GPU. ([#2435](https://github.com/horovod/horovod/pull/2435))
CHANGELOG.md:- Fixed averaging using CUDA half2 implementation one element half buffers. ([#2375](https://github.com/horovod/horovod/pull/2375))
CHANGELOG.md:- Added Databricks storage `DBFSLocalStore` and support for GPU-aware scheduling to horovod.spark Estimator. ([#2234](https://github.com/horovod/horovod/pull/2234))
CHANGELOG.md:- Fixed allreduce averaging for TF IndexedSlices in ROCm path. ([#2279](https://github.com/horovod/horovod/pull/2279))
CHANGELOG.md:- Skipped launching zero-sized send/recvs for NCCLAlltoall. ([#2273](https://github.com/horovod/horovod/pull/2273))
CHANGELOG.md:- Added NCCL implementation of the allgather operation. ([#1952](https://github.com/horovod/horovod/pull/1952))
CHANGELOG.md:- Added `HOROVOD_GPU_OPERATIONS` installation variable to simplify enabling NCCL support for all GPU operations. ([#1960](https://github.com/horovod/horovod/pull/1960))
GOVERNANCE.md:* [Josh Romero](https://github.com/romerojosh) - NVIDIA
GOVERNANCE.md:* [Nicolas Castet](https://github.com/nvcastet) - NVIDIA
GOVERNANCE.md:* [Jonathan Dekhtiar](https://github.com/DEKHTIARJonathan) - NVIDIA
GOVERNANCE.md:* [TJ Xu](https://github.com/Tixxx) - NVIDIA
Jenkinsfile.ppc64le:            // Open-CE 1.4.1 has CUDA 10.2, NCCL 2.8.3, TensorFlow 2.6.0, and PyTorch 1.9.1
Jenkinsfile.ppc64le:            label 'power8-gpu'
Jenkinsfile.ppc64le:                          HOROVOD_CUDA_HOME="/usr/local/cuda" HOROVOD_GPU_OPERATIONS=NCCL \
Jenkinsfile.ppc64le:                          horovodrun -n 2 -H localhost:2 pytest -k 'not multi_gpu' -v -s test/parallel/test_tensorflow.py
Jenkinsfile.ppc64le:                          # Container has only 2 GPUs, so run the 'multi_gpu' test seperatly on one process
Jenkinsfile.ppc64le:                          horovodrun -n 1 -H localhost:1 pytest -k 'multi_gpu' -v -s test/parallel/test_tensorflow.py
Jenkinsfile.ppc64le:                 targetUrl: "https://powerci.osuosl.org/job/Horovod_PPC64LE_GPU_PIPELINE/view/change-requests/job/${BRANCH_NAME}/${BUILD_NUMBER}/console"
NOTICE:   NVIDIA/cutlass
NOTICE:   Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
NOTICE:      *  Neither the name of the NVIDIA CORPORATION nor the
NOTICE:   DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
NOTICE:          - cmake/upstream/FindCUDAToolkit.cmake
CMakeLists.txt:    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
CMakeLists.txt:string(REPLACE "-O2" "-O3" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")
CMakeLists.txt:# GPU Operations
CMakeLists.txt:set(HOROVOD_GPU $ENV{HOROVOD_GPU})
CMakeLists.txt:set(HOROVOD_GPU_OPERATIONS $ENV{HOROVOD_GPU_OPERATIONS})
CMakeLists.txt:if(DEFINED HOROVOD_GPU_OPERATIONS AND NOT "${HOROVOD_GPU_OPERATIONS}" MATCHES "^(MPI|NCCL)$")
CMakeLists.txt:    message(FATAL_ERROR "HOROVOD_GPU_OPERATIONS=${HOROVOD_GPU_OPERATIONS} is invalid, supported values are '', 'MPI', and 'NCCL'.")
CMakeLists.txt:set_gpu_op(HOROVOD_GPU_ALLREDUCE "MPI;NCCL;DDL")
CMakeLists.txt:set_gpu_op(HOROVOD_GPU_ALLGATHER "MPI;NCCL")
CMakeLists.txt:set_gpu_op(HOROVOD_GPU_BROADCAST "MPI;NCCL")
CMakeLists.txt:set_gpu_op(HOROVOD_GPU_ALLTOALL "MPI;NCCL")
CMakeLists.txt:set_gpu_op(HOROVOD_GPU_REDUCESCATTER "MPI;NCCL")
CMakeLists.txt:foreach(VAR in ITEMS HOROVOD_GPU_ALLREDUCE HOROVOD_GPU_ALLGATHER HOROVOD_GPU_BROADCAST HOROVOD_GPU_ALLTOALL HOROVOD_GPU_REDUCESCATTER)
CMakeLists.txt:# CUDA and ROCM
CMakeLists.txt:set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
CMakeLists.txt:if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
CMakeLists.txt:    set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")  # Set to "Static" or "Shared", effective from CMake 3.17
CMakeLists.txt:if(DEFINED ENV{HOROVOD_CUDA_HOME})
CMakeLists.txt:    set(CMAKE_CUDA_COMPILER "$ENV{HOROVOD_CUDA_HOME}/bin/nvcc")
CMakeLists.txt:check_language(CUDA)
CMakeLists.txt:if(NOT CMAKE_CUDA_COMPILER)
CMakeLists.txt:    find_package(CUDAToolkit)
CMakeLists.txt:    if(CUDAToolkit_BIN_DIR)
CMakeLists.txt:        message("CUDA compiler was not found in $PATH, but searching again in CUDA Toolkit binary directory")
CMakeLists.txt:        unset(CMAKE_CUDA_COMPILER CACHE)  # need to clear this from cache, else some versions of CMake go into an infinite loop
CMakeLists.txt:        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_BIN_DIR}/nvcc")
CMakeLists.txt:        check_language(CUDA)
CMakeLists.txt:if(CMAKE_CUDA_COMPILER)
CMakeLists.txt:            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")
CMakeLists.txt:    enable_language(CUDA)
CMakeLists.txt:macro(ADD_CUDA)
CMakeLists.txt:    find_package(CUDAToolkit REQUIRED)
CMakeLists.txt:    include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
CMakeLists.txt:    list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/ops/cuda_operations.cc"
CMakeLists.txt:                        "${PROJECT_SOURCE_DIR}/horovod/common/ops/gpu_operations.cc")
CMakeLists.txt:    # CUDA + MPI
CMakeLists.txt:        list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/ops/mpi_gpu_operations.cc")
CMakeLists.txt:    add_definitions(-DHAVE_CUDA=1 -DHAVE_GPU=1)
CMakeLists.txt:    set(HAVE_CUDA TRUE)
CMakeLists.txt:        set(HAVE_SUB_PROJECT_CUDA TRUE PARENT_SCOPE)
CMakeLists.txt:if(DEFINED HOROVOD_GPU_ALLREDUCE OR DEFINED HOROVOD_GPU_ALLGATHER OR DEFINED HOROVOD_GPU_BROADCAST OR DEFINED HOROVOD_GPU_ALLTOALL OR DEFINED HOROVOD_GPU_REDUCESCATTER)
CMakeLists.txt:    if(NOT DEFINED HOROVOD_GPU OR HOROVOD_GPU STREQUAL "CUDA")
CMakeLists.txt:        add_cuda()
CMakeLists.txt:    elseif(HOROVOD_GPU STREQUAL "ROCM")
CMakeLists.txt:        find_package(ROCM REQUIRED)
CMakeLists.txt:        include_directories(SYSTEM ${ROCM_INCLUDE_DIRS})
CMakeLists.txt:        list(APPEND LINKER_LIBS ${ROCM_LIBRARIES})
CMakeLists.txt:        set(CMAKE_CXX_FLAGS "${ROCM_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
CMakeLists.txt:                            "${PROJECT_SOURCE_DIR}/horovod/common/ops/gpu_operations.cc")
CMakeLists.txt:            list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/ops/mpi_gpu_operations.cc")
CMakeLists.txt:        add_definitions(-DHAVE_ROCM=1 -DHAVE_GPU=1)
CMakeLists.txt:        set(HAVE_ROCM TRUE)
CMakeLists.txt:        message(FATAL_ERROR "Unknown HOROVOD_GPU type: ${HOROVOD_GPU}")
CMakeLists.txt:# NCCL
CMakeLists.txt:if(HOROVOD_GPU_ALLREDUCE STREQUAL "N" OR HOROVOD_GPU_ALLGATHER STREQUAL "N" OR HOROVOD_GPU_BROADCAST STREQUAL "N" OR HOROVOD_GPU_ALLTOALL STREQUAL "N" OR HOROVOD_GPU_REDUCESCATTER STREQUAL "N")
CMakeLists.txt:    if(HAVE_ROCM)
CMakeLists.txt:        find_package(NCCL REQUIRED)
CMakeLists.txt:        if (NCCL_MAJOR_VERSION LESS "2")
CMakeLists.txt:            message(FATAL_ERROR "Horovod requires NCCL 2.0 or later version please upgrade.")
CMakeLists.txt:        string(TOLOWER "${CMAKE_CUDA_RUNTIME_LIBRARY}" lowercase_CMAKE_CUDA_RUNTIME_LIBRARY)
CMakeLists.txt:        get_filename_component(NCCL_LIBRARY_FILE_NAME ${NCCL_LIBRARIES} NAME)
CMakeLists.txt:            AND lowercase_CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "shared"
CMakeLists.txt:            AND NCCL_LIBRARY_FILE_NAME MATCHES "static")
CMakeLists.txt:            message(WARNING "Linking NCCL statically, but linking CUDA runtime library dynamically. This combination is not supported with typical builds of NCCL.")
CMakeLists.txt:        include_directories(SYSTEM ${NCCL_INCLUDE_DIRS})
CMakeLists.txt:        list(APPEND LINKER_LIBS ${NCCL_LIBRARIES})
CMakeLists.txt:        if(NCCL_LIBRARY_FILE_NAME MATCHES "static" AND lowercase_CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "static")
CMakeLists.txt:            # ensure that weak symbols from NCCL's enhcompat.cc are properly overwritten by symbols from libcudart_static.a (https://github.com/horovod/horovod/pull/3846)
CMakeLists.txt:            list(APPEND LINKER_LIBS -Wl,--whole-archive cudart_static -Wl,--no-whole-archive)
CMakeLists.txt:    list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/ops/nccl_operations.cc")
CMakeLists.txt:    add_definitions(-DHAVE_NCCL=1)
CMakeLists.txt:    set(HAVE_NCCL TRUE)
CMakeLists.txt:if(HOROVOD_GPU_ALLREDUCE STREQUAL "D")
CMakeLists.txt:    message(DEPRECATION "DDL backend has been deprecated. Please, start using the NCCL backend by building Horovod with "
CMakeLists.txt:                        "'HOROVOD_GPU_OPERATIONS=NCCL'. Will be removed in v0.21.0.")
CMakeLists.txt:    list(APPEND LINKER_LIBS "${CUDAToolkit_LIBRARY_DIR}/libddl.so" "${CUDAToolkit_LIBRARY_DIR}/libddl_pack.so")
CMakeLists.txt:set(HOROVOD_ALLOW_MIXED_GPU_IMPL $ENV{HOROVOD_ALLOW_MIXED_GPU_IMPL})
CMakeLists.txt:if(HOROVOD_GPU_ALLREDUCE STREQUAL "N" AND (HOROVOD_GPU_ALLGATHER STREQUAL "M" OR HOROVOD_GPU_BROADCAST STREQUAL "M" OR HOROVOD_GPU_ALLTOALL STREQUAL "M" OR HOROVOD_GPU_REDUCESCATTER STREQUAL "M") AND
CMakeLists.txt:   NOT HOROVOD_ALLOW_MIXED_GPU_IMPL STREQUAL "1")
CMakeLists.txt:message(FATAL_ERROR "You should not mix NCCL and MPI GPU due to a possible deadlock.\n"
CMakeLists.txt:                    "HOROVOD_ALLOW_MIXED_GPU_IMPL environment variable to '1'.")
CMakeLists.txt:# NCCL + MPI
CMakeLists.txt:if (HAVE_NCCL AND HAVE_MPI)
CMakeLists.txt:    list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/ops/adasum_gpu_operations.cc")
CMakeLists.txt:# CUDA kernels
CMakeLists.txt:if(HAVE_CUDA OR HAVE_SUB_PROJECT_CUDA)
CMakeLists.txt:    add_subdirectory(horovod/common/ops/cuda)
CMakeLists.txt:if(HAVE_ROCM)
CMakeLists.txt:    add_subdirectory(horovod/common/ops/rocm)
docker/README.md:correctly with CUDA, MPI, G++, CMake, etc. These Docker images are provided to simplify
docker/README.md:* `horovod/horovod` Horovod built with CUDA support and packaged with the latest stable TensorFlow, PyTorch, MXNet, 
docker/README.md:* `horovod/horovod-ray` Horoovd built with CUDA support from the latest 
docker/README.md:  [ray-project/ray:nightly-gpu](https://github.com/ray-project/ray) and packaged with the latest stable 
docker/README.md:* `NCCL_VERSION` - version of `libnccl` apt package to install (only for `horovod` image)
docker/README.md:* `CUDA_DOCKER_VERSION` - tag of the `nvidia/cuda` image to build from (only for `horovod` image)
docker/README.md:* `RAY_DOCKER_VERSION` - tag of the `rayproject/ray` GPU image to build from (only for `horovod-ray` image)
docker/horovod-nvtabular/Dockerfile:ARG CUDA_DOCKER_VERSION=11.6.2-devel-ubuntu20.04
docker/horovod-nvtabular/Dockerfile:FROM nvidia/cuda:${CUDA_DOCKER_VERSION}
docker/horovod-nvtabular/Dockerfile:# Arguments for the build. CUDA_DOCKER_VERSION needs to be repeated because
docker/horovod-nvtabular/Dockerfile:ARG CUDA_DOCKER_VERSION=11.6.2-devel-ubuntu20.04
docker/horovod-nvtabular/Dockerfile:ARG CUDNN_VERSION=8.4.1.50-1+cuda11.6
docker/horovod-nvtabular/Dockerfile:ARG NCCL_VERSION_OVERRIDE=2.11.4-1+cuda11.6
docker/horovod-nvtabular/Dockerfile:ARG TENSORFLOW_PACKAGE=tensorflow-gpu==2.10.0
docker/horovod-nvtabular/Dockerfile:ARG HOROVOD_BUILD_FLAGS="HOROVOD_GPU_OPERATIONS=NCCL"
docker/horovod-nvtabular/Dockerfile:# This is to fix CI failures caused by the new rotating key mechanism rolled out by Nvidia.
docker/horovod-nvtabular/Dockerfile:# Refer to https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771 for more details.
docker/horovod-nvtabular/Dockerfile:#RUN DIST=$(echo ${CUDA_DOCKER_VERSION#*ubuntu} | sed 's/\.//'); \
docker/horovod-nvtabular/Dockerfile:#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${DIST}/x86_64/3bf863cc.pub && \
docker/horovod-nvtabular/Dockerfile:#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu${DIST}/x86_64/7fa2af80.pub
docker/horovod-nvtabular/Dockerfile:        libnccl2=${NCCL_VERSION_OVERRIDE} \
docker/horovod-nvtabular/Dockerfile:        libnccl-dev=${NCCL_VERSION_OVERRIDE} && \
docker/horovod-nvtabular/Dockerfile:RUN pip install --no-cache-dir cudf-cu11==23.4.1 dask-cudf-cu11==23.4.1 --extra-index-url=https://pypi.nvidia.com
docker/horovod-nvtabular/Dockerfile:RUN pip install --no-cache-dir numba==0.56 nvidia-ml-py nvtabular
docker/horovod-nvtabular/Dockerfile:# Set default NCCL parameters
docker/horovod-nvtabular/Dockerfile:RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf
docker/horovod-nvtabular/Dockerfile:        ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs; \
docker/horovod-nvtabular/Dockerfile:        ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs; \
docker/horovod-nvtabular/Dockerfile:    ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
docker/horovod-cpu/Dockerfile:# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
docker/helm/README.md:You can download [official Horovod Dockerfile](https://github.com/horovod/horovod/blob/master/docker/horovod/Dockerfile), then modify it according to your requirement, e.g. select a different CUDA, TensorFlow or Python version.
docker/helm/README.md:To run Horovod with GPU, you can create `values.yaml` like below
docker/helm/README.md:    nvidia.com/gpu: 1
docker/helm/README.md:    nvidia.com/gpu: 1
docker/helm/README.md:For most cases, the overlay network impacts the Horovod performance greatly, so we should apply `Host Network` solution. To run Horovod with Host Network and GPU, you can create `values.yaml` like below
docker/helm/README.md:    nvidia.com/gpu: 1
docker/helm/README.md:    nvidia.com/gpu: 1
docker/helm/templates/config.yaml:{{- if index .Values.resources "nvidia.com/gpu" }}
docker/helm/templates/config.yaml:{{- $slots := index .Values.resources "nvidia.com/gpu" }}
docker/helm/values.yaml:  #   nvidia.com/gpu: 1
docker/helm/values.yaml:  #   nvidia.com/gpu: 1
docker/helm/values.yaml:  #  - "mpiexec -n 3 --hostfile /horovod/generated/hostfile --mca orte_keep_fqdn_hostnames t --allow-run-as-root --display-map --tag-output --timestamp-output sh -c 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs python /examples/tensorflow_mnist.py'"
docker/horovod/Dockerfile:ARG CUDA_DOCKER_VERSION=11.3.1-devel-ubuntu20.04
docker/horovod/Dockerfile:FROM nvidia/cuda:${CUDA_DOCKER_VERSION}
docker/horovod/Dockerfile:# Arguments for the build. CUDA_DOCKER_VERSION needs to be repeated because
docker/horovod/Dockerfile:# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
docker/horovod/Dockerfile:ARG CUDA_DOCKER_VERSION=11.3.1-devel-ubuntu20.04
docker/horovod/Dockerfile:ARG CUDNN_VERSION=8.2.1.32-1+cuda11.3
docker/horovod/Dockerfile:ARG NCCL_VERSION=2.9.9-1+cuda11.3
docker/horovod/Dockerfile:# This is to fix CI failures caused by the new rotating key mechanism rolled out by Nvidia.
docker/horovod/Dockerfile:# Refer to https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771 for more details.
docker/horovod/Dockerfile:RUN DIST=$(echo ${CUDA_DOCKER_VERSION#*ubuntu} | sed 's/\.//'); \
docker/horovod/Dockerfile:    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${DIST}/x86_64/3bf863cc.pub && \
docker/horovod/Dockerfile:    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu${DIST}/x86_64/7fa2af80.pub
docker/horovod/Dockerfile:        libnccl2=${NCCL_VERSION} \
docker/horovod/Dockerfile:        libnccl-dev=${NCCL_VERSION} \
docker/horovod/Dockerfile:# Install Horovod, temporarily using CUDA stubs
docker/horovod/Dockerfile:    ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
docker/horovod/Dockerfile:    bash -c "HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 pip install --no-cache-dir -v $(ls /horovod/dist/horovod-*.tar.gz)[spark,ray]" && \
docker/horovod/Dockerfile:# Check all frameworks are working correctly. Use CUDA stubs to ensure CUDA libs can be found correctly
docker/horovod/Dockerfile:RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
docker/horovod-ray/Dockerfile:FROM rayproject/ray:${RAY_DOCKER_VERSION}-gpu
docker/horovod-ray/Dockerfile:# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
docker/horovod-ray/Dockerfile:# This is to fix CI failures caused by the new rotating key mechanism rolled out by Nvidia.
docker/horovod-ray/Dockerfile:# Refer to https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771 for more details.
docker/horovod-ray/Dockerfile:RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
docker/horovod-ray/Dockerfile:RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
docker/horovod-ray/Dockerfile:# Install Horovod, temporarily using CUDA stubs
docker/horovod-ray/Dockerfile:    sudo ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
docker/horovod-ray/Dockerfile:    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir -v $(ls /horovod/dist/horovod-*.tar.gz)[ray] && \
docker/horovod-ray/Dockerfile:# Check all frameworks are working correctly. Use CUDA stubs to ensure CUDA libs can be found correctly
docker/horovod-ray/Dockerfile:RUN sudo ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
docker-compose.test.yml:  test-gpu-base:
docker-compose.test.yml:      dockerfile: Dockerfile.test.gpu
docker-compose.test.yml:        HOROVOD_BUILD_FLAGS: HOROVOD_GPU_OPERATIONS=NCCL
docker-compose.test.yml:    runtime: nvidia
docker-compose.test.yml:    # We plumb CUDA_VISIBLE_DEVICES instead of NVIDIA_VISIBLE_DEVICES because
docker-compose.test.yml:      - CUDA_VISIBLE_DEVICES
docker-compose.test.yml:  # available versions for CUDNN_VERSION and NCCL_VERSION_OVERRIDE can be found at
docker-compose.test.yml:  #   https://developer.download.nvidia.com/compute/cuda/repos/{OS}/x86_64/
docker-compose.test.yml:  # Mainline tensorflow-gpu==1.15.5 is compiled against and linked to CUDA 10.0, but appropriate containers aren't
docker-compose.test.yml:  # available anymore. Hence, we use the updated Python 3.8 wheel provided by Nvidia, see
docker-compose.test.yml:  # https://github.com/NVIDIA/tensorflow. For this reason versions of torch and mxnet also deviate from the CPU path.
docker-compose.test.yml:  test-gpu-gloo-py3_8-tf1_15_5-keras2_2_4-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0:
docker-compose.test.yml:    extends: test-gpu-base
docker-compose.test.yml:        CUDA_DOCKER_VERSION: 11.6.2-devel-ubuntu20.04
docker-compose.test.yml:        CUDNN_VERSION: 8.4.1.50-1+cuda11.6
docker-compose.test.yml:        NCCL_VERSION_OVERRIDE: 2.11.4-1+cuda11.6
docker-compose.test.yml:        TENSORFLOW_PACKAGE: nvidia-tensorflow==1.15.5+nv22.4
docker-compose.test.yml:  # The container isn't provided for CUDA 10 anymore. The lowest version of mxnet available for cu112 is 1.8.0.post0.
docker-compose.test.yml:  test-gpu-gloo-py3_8-tf2_10_1-keras2_10_0-torch1_12_1-mxnet1_8_0_p0-pyspark3_4_0:
docker-compose.test.yml:    extends: test-gpu-base
docker-compose.test.yml:        CUDA_DOCKER_VERSION: 11.6.2-devel-ubuntu20.04
docker-compose.test.yml:        CUDNN_VERSION: 8.4.1.50-1+cuda11.6
docker-compose.test.yml:        NCCL_VERSION_OVERRIDE: 2.11.4-1+cuda11.6
docker-compose.test.yml:        TENSORFLOW_PACKAGE: tensorflow-gpu==2.10.1
docker-compose.test.yml:  test-gpu-gloo-py3_8-tf2_11_1-keras2_11_0-torch1_13_1-mxnet1_8_0_p0-pyspark3_4_0:
docker-compose.test.yml:    extends: test-gpu-base
docker-compose.test.yml:        CUDA_DOCKER_VERSION: 11.6.2-devel-ubuntu20.04
docker-compose.test.yml:        CUDNN_VERSION: 8.4.1.50-1+cuda11.6
docker-compose.test.yml:        NCCL_VERSION_OVERRIDE: 2.11.4-1+cuda11.6
docker-compose.test.yml:        # tensorflow package supports GPU from 2.11.1 and 2.12.0 on
docker-compose.test.yml:  test-gpu-openmpi-gloo-py3_8-tf2_12_0-keras2_12_0-torch2_0_0-mxnet1_9_1-pyspark3_4_0:
docker-compose.test.yml:    extends: test-gpu-base
docker-compose.test.yml:        CUDA_DOCKER_VERSION: 11.8.0-devel-ubuntu20.04
docker-compose.test.yml:        CUDNN_VERSION: 8.6.0.163-1+cuda11.8
docker-compose.test.yml:        NCCL_VERSION_OVERRIDE: 2.16.5-1+cuda11.8
docker-compose.test.yml:        # tensorflow package supports GPU from 2.11.1 and 2.12.0 on
docker-compose.test.yml:  test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_4_0:
docker-compose.test.yml:    extends: test-gpu-base
docker-compose.test.yml:        CUDA_DOCKER_VERSION: 11.8.0-devel-ubuntu20.04
docker-compose.test.yml:        CUDNN_VERSION: 8.6.0.163-1+cuda11.8
docker-compose.test.yml:        NCCL_VERSION_OVERRIDE: 2.16.5-1+cuda11.8
docker-compose.test.yml:  # These are the lowest framework versions that Horovod compiles with on the CUDA 11.x container, but they are not tested.
docker-compose.test.yml:  test-gpu-openmpi-gloo-py3_8-tfmin-kerasmin-torchmin-mxnetmin-pysparkmin:
docker-compose.test.yml:    extends: test-gpu-base
docker-compose.test.yml:        CUDA_DOCKER_VERSION: 11.6.2-devel-ubuntu20.04
docker-compose.test.yml:        CUDNN_VERSION: 8.4.1.50-1+cuda11.6
docker-compose.test.yml:        NCCL_VERSION_OVERRIDE: 2.11.4-1+cuda11.6
docker-compose.test.yml:        TENSORFLOW_PACKAGE: nvidia-tensorflow==1.15.5+nv22.4
docker-compose.test.yml:        # torch ships its own CUDA libraries
docker-compose.test.yml:    extends: test-gpu-base
docker-compose.test.yml:        CUDA_DOCKER_VERSION: 11.8.0-devel-ubuntu20.04
docker-compose.test.yml:        CUDNN_VERSION: 8.6.0.163-1+cuda11.8
docker-compose.test.yml:        NCCL_VERSION_OVERRIDE: 2.16.5-1+cuda11.8
docker-compose.test.yml:        # tensorflow package supports GPU from 2.11.1 and 2.12.0 on
Dockerfile.test.gpu:ARG CUDA_DOCKER_VERSION=10.0-devel-ubuntu20.04
Dockerfile.test.gpu:FROM nvidia/cuda:${CUDA_DOCKER_VERSION}
Dockerfile.test.gpu:# Arguments for the build. CUDA_DOCKER_VERSION needs to be repeated because
Dockerfile.test.gpu:ARG CUDA_DOCKER_VERSION=10.0-devel-ubuntu20.04
Dockerfile.test.gpu:ARG CUDNN_VERSION=7.6.0.64-1+cuda10.0
Dockerfile.test.gpu:ARG NCCL_VERSION_OVERRIDE=2.4.7-1+cuda10.0
Dockerfile.test.gpu:ARG TENSORFLOW_PACKAGE=tensorflow-gpu==1.15.0
Dockerfile.test.gpu:ARG HOROVOD_BUILD_FLAGS="HOROVOD_GPU_OPERATIONS=NCCL"
Dockerfile.test.gpu:# This is to fix CI failures caused by the new rotating key mechanism rolled out by Nvidia.
Dockerfile.test.gpu:# Refer to https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771 for more details.
Dockerfile.test.gpu:#RUN DIST=$(echo ${CUDA_DOCKER_VERSION#*ubuntu} | sed 's/\.//'); \
Dockerfile.test.gpu:#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${DIST}/x86_64/3bf863cc.pub && \
Dockerfile.test.gpu:#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu${DIST}/x86_64/7fa2af80.pub
Dockerfile.test.gpu:        libnccl2=${NCCL_VERSION_OVERRIDE} \
Dockerfile.test.gpu:        libnccl-dev=${NCCL_VERSION_OVERRIDE} && \
Dockerfile.test.gpu:# Set default NCCL parameters
Dockerfile.test.gpu:RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf
Dockerfile.test.gpu:        if [[ ${TENSORFLOW_PACKAGE} == nvidia-tensorflow==* ]]; then \
Dockerfile.test.gpu:            pip install nvidia-pyindex; \
Dockerfile.test.gpu:        if [[ ${TENSORFLOW_PACKAGE} == tensorflow-gpu==1.15.* ]] || \
Dockerfile.test.gpu:           [[ ${TENSORFLOW_PACKAGE} == tensorflow-gpu==2.[012345].* ]] || \
Dockerfile.test.gpu:           [[ ${TENSORFLOW_PACKAGE} == nvidia-tensorflow==* ]]; then \
Dockerfile.test.gpu:        ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs; \
Dockerfile.test.gpu:        ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs; \
Dockerfile.test.gpu:    ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
Dockerfile.test.gpu:RUN if [[ ${TENSORFLOW_PACKAGE} == "tensorflow-gpu==1.1.0" ]]; then \
cmake/upstream/FindCUDAToolkit.cmake:FindCUDAToolkit
cmake/upstream/FindCUDAToolkit.cmake:This script locates the NVIDIA CUDA toolkit and the associated libraries, but
cmake/upstream/FindCUDAToolkit.cmake:does not require the ``CUDA`` language be enabled for a given project. This
cmake/upstream/FindCUDAToolkit.cmake:module does not search for the NVIDIA CUDA Samples.
cmake/upstream/FindCUDAToolkit.cmake:Finding the CUDA Toolkit requires finding the ``nvcc`` executable, which is
cmake/upstream/FindCUDAToolkit.cmake:1. If the ``CUDA`` language has been enabled we will use the directory
cmake/upstream/FindCUDAToolkit.cmake:2. If the ``CUDAToolkit_ROOT`` cmake configuration variable (e.g.,
cmake/upstream/FindCUDAToolkit.cmake:   ``-DCUDAToolkit_ROOT=/some/path``) *or* environment variable is defined, it
cmake/upstream/FindCUDAToolkit.cmake:   found underneath the directory specified by ``CUDAToolkit_ROOT``.  If
cmake/upstream/FindCUDAToolkit.cmake:   ``CUDAToolkit_ROOT`` is specified, but no ``nvcc`` is found underneath, this
cmake/upstream/FindCUDAToolkit.cmake:3. If the CUDA_PATH environment variable is defined, it will be searched.
cmake/upstream/FindCUDAToolkit.cmake:   the desired path in the event that multiple CUDA Toolkits are installed.
cmake/upstream/FindCUDAToolkit.cmake:5. On Unix systems, if the symbolic link ``/usr/local/cuda`` exists, this is
cmake/upstream/FindCUDAToolkit.cmake:   candidate is found, this is used.  The default CUDA Toolkit install locations
cmake/upstream/FindCUDAToolkit.cmake:   | macOS       | ``/Developer/NVIDIA/CUDA-X.Y``                              |
cmake/upstream/FindCUDAToolkit.cmake:   | Other Unix  | ``/usr/local/cuda-X.Y``                                     |
cmake/upstream/FindCUDAToolkit.cmake:   | Windows     | ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y`` |
cmake/upstream/FindCUDAToolkit.cmake:   Where ``X.Y`` would be a specific version of the CUDA Toolkit, such as
cmake/upstream/FindCUDAToolkit.cmake:   ``/usr/local/cuda-9.0`` or
cmake/upstream/FindCUDAToolkit.cmake:   ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0``
cmake/upstream/FindCUDAToolkit.cmake:       When multiple CUDA Toolkits are installed in the default location of a
cmake/upstream/FindCUDAToolkit.cmake:       system (e.g., both ``/usr/local/cuda-9.0`` and ``/usr/local/cuda-10.0``
cmake/upstream/FindCUDAToolkit.cmake:       exist but the ``/usr/local/cuda`` symbolic link does **not** exist), this
cmake/upstream/FindCUDAToolkit.cmake:       the presence of multiple CUDA Toolkits being installed.  In this
cmake/upstream/FindCUDAToolkit.cmake:       situation, users are encouraged to either (1) set ``CUDAToolkit_ROOT`` or
cmake/upstream/FindCUDAToolkit.cmake:    If specified, describes the version of the CUDA Toolkit to search for.
cmake/upstream/FindCUDAToolkit.cmake:    If specified, configuration will error if a suitable CUDA Toolkit is not
cmake/upstream/FindCUDAToolkit.cmake:    If specified, the search for a suitable CUDA Toolkit will not produce any
cmake/upstream/FindCUDAToolkit.cmake:    If specified, the CUDA Toolkit is considered found only if the exact
cmake/upstream/FindCUDAToolkit.cmake:An :ref:`imported target <Imported targets>` named ``CUDA::toolkit`` is provided.
cmake/upstream/FindCUDAToolkit.cmake:of the following libraries that are part of the CUDAToolkit:
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`CUDA Runtime Library<cuda_toolkit_rt_lib>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`CUDA Driver Library<cuda_toolkit_driver_lib>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`cuBLAS<cuda_toolkit_cuBLAS>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`cuFFT<cuda_toolkit_cuFFT>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`cuRAND<cuda_toolkit_cuRAND>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`cuSOLVER<cuda_toolkit_cuSOLVER>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`cuSPARSE<cuda_toolkit_cuSPARSE>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`cuPTI<cuda_toolkit_cupti>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`NPP<cuda_toolkit_NPP>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`nvBLAS<cuda_toolkit_nvBLAS>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`nvGRAPH<cuda_toolkit_nvGRAPH>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`nvJPEG<cuda_toolkit_nvJPEG>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`nvidia-ML<cuda_toolkit_nvML>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`nvRTC<cuda_toolkit_nvRTC>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`nvToolsExt<cuda_toolkit_nvToolsExt>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`OpenCL<cuda_toolkit_opencl>`
cmake/upstream/FindCUDAToolkit.cmake:- :ref:`cuLIBOS<cuda_toolkit_cuLIBOS>`
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_rt_lib`:
cmake/upstream/FindCUDAToolkit.cmake:CUDA Runtime Library
cmake/upstream/FindCUDAToolkit.cmake:The CUDA Runtime library (cudart) are what most applications will typically
cmake/upstream/FindCUDAToolkit.cmake:need to link against to make any calls such as `cudaMalloc`, and `cudaFree`.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cudart``
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cudart_static``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_driver_lib`:
cmake/upstream/FindCUDAToolkit.cmake:CUDA Driver Library
cmake/upstream/FindCUDAToolkit.cmake:The CUDA Driver library (cuda) are used by applications that use calls
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cuda_driver``
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cuda_driver``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_cuBLAS`:
cmake/upstream/FindCUDAToolkit.cmake:The `cuBLAS <https://docs.nvidia.com/cuda/cublas/index.html>`_ library.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cublas``
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cublas_static``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_cuFFT`:
cmake/upstream/FindCUDAToolkit.cmake:The `cuFFT <https://docs.nvidia.com/cuda/cufft/index.html>`_ library.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cufft``
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cufftw``
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cufft_static``
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cufftw_static``
cmake/upstream/FindCUDAToolkit.cmake:The `cuRAND <https://docs.nvidia.com/cuda/curand/index.html>`_ library.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::curand``
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::curand_static``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_cuSOLVER`:
cmake/upstream/FindCUDAToolkit.cmake:The `cuSOLVER <https://docs.nvidia.com/cuda/cusolver/index.html>`_ library.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cusolver``
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cusolver_static``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_cuSPARSE`:
cmake/upstream/FindCUDAToolkit.cmake:The `cuSPARSE <https://docs.nvidia.com/cuda/cusparse/index.html>`_ library.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cusparse``
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cusparse_static``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_cupti`:
cmake/upstream/FindCUDAToolkit.cmake:The `NVIDIA CUDA Profiling Tools Interface <https://developer.nvidia.com/CUPTI>`_.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cupti``
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::cupti_static``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_NPP`:
cmake/upstream/FindCUDAToolkit.cmake:The `NPP <https://docs.nvidia.com/cuda/npp/index.html>`_ libraries.
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppc``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppc_static``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppial``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppial_static``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppicc``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppicc_static``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppicom``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppicom_static``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppidei``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppidei_static``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppif``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppif_static``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppig``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppig_static``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppim``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppim_static``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppist``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppist_static``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppisu``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppisu_static``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppitc``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::nppitc_static``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::npps``
cmake/upstream/FindCUDAToolkit.cmake:  - ``CUDA::npps_static``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_nvBLAS`:
cmake/upstream/FindCUDAToolkit.cmake:The `nvBLAS <https://docs.nvidia.com/cuda/nvblas/index.html>`_ libraries.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::nvblas``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_nvGRAPH`:
cmake/upstream/FindCUDAToolkit.cmake:The `nvGRAPH <https://docs.nvidia.com/cuda/nvgraph/index.html>`_ library.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::nvgraph``
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::nvgraph_static``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_nvJPEG`:
cmake/upstream/FindCUDAToolkit.cmake:The `nvJPEG <https://docs.nvidia.com/cuda/nvjpeg/index.html>`_ library.
cmake/upstream/FindCUDAToolkit.cmake:Introduced in CUDA 10.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::nvjpeg``
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::nvjpeg_static``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_nvRTC`:
cmake/upstream/FindCUDAToolkit.cmake:The `nvRTC <https://docs.nvidia.com/cuda/nvrtc/index.html>`_ (Runtime Compilation) library.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::nvrtc``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_nvml`:
cmake/upstream/FindCUDAToolkit.cmake:nvidia-ML
cmake/upstream/FindCUDAToolkit.cmake:The `NVIDIA Management Library <https://developer.nvidia.com/nvidia-management-library-nvml>`_.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::nvml``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_nvToolsExt`:
cmake/upstream/FindCUDAToolkit.cmake:The `NVIDIA Tools Extension <https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm>`_.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::nvToolsExt``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_opencl`:
cmake/upstream/FindCUDAToolkit.cmake:OpenCL
cmake/upstream/FindCUDAToolkit.cmake:The `NVIDIA OpenCL Library <https://developer.nvidia.com/opencl>`_.
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::OpenCL``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_cuLIBOS`:
cmake/upstream/FindCUDAToolkit.cmake:static only.  The ``CUDA::cublas_static``, ``CUDA::cusparse_static``,
cmake/upstream/FindCUDAToolkit.cmake:``CUDA::cufft_static``, ``CUDA::curand_static``, and (when implemented) NPP
cmake/upstream/FindCUDAToolkit.cmake:- ``CUDA::culibos``
cmake/upstream/FindCUDAToolkit.cmake:.. _`cuda_toolkit_cuRAND`:
cmake/upstream/FindCUDAToolkit.cmake:``CUDAToolkit_FOUND``
cmake/upstream/FindCUDAToolkit.cmake:    A boolean specifying whether or not the CUDA Toolkit was found.
cmake/upstream/FindCUDAToolkit.cmake:``CUDAToolkit_VERSION``
cmake/upstream/FindCUDAToolkit.cmake:    The exact version of the CUDA Toolkit found (as reported by
cmake/upstream/FindCUDAToolkit.cmake:``CUDAToolkit_VERSION_MAJOR``
cmake/upstream/FindCUDAToolkit.cmake:    The major version of the CUDA Toolkit.
cmake/upstream/FindCUDAToolkit.cmake:``CUDAToolkit_VERSION_MAJOR``
cmake/upstream/FindCUDAToolkit.cmake:    The minor version of the CUDA Toolkit.
cmake/upstream/FindCUDAToolkit.cmake:``CUDAToolkit_VERSION_PATCH``
cmake/upstream/FindCUDAToolkit.cmake:    The patch version of the CUDA Toolkit.
cmake/upstream/FindCUDAToolkit.cmake:``CUDAToolkit_BIN_DIR``
cmake/upstream/FindCUDAToolkit.cmake:    The path to the CUDA Toolkit library directory that contains the CUDA
cmake/upstream/FindCUDAToolkit.cmake:``CUDAToolkit_INCLUDE_DIRS``
cmake/upstream/FindCUDAToolkit.cmake:    The path to the CUDA Toolkit ``include`` folder containing the header files
cmake/upstream/FindCUDAToolkit.cmake:    required to compile a project linking against CUDA.
cmake/upstream/FindCUDAToolkit.cmake:``CUDAToolkit_LIBRARY_DIR``
cmake/upstream/FindCUDAToolkit.cmake:    The path to the CUDA Toolkit library directory that contains the CUDA
cmake/upstream/FindCUDAToolkit.cmake:    Runtime library ``cudart``.
cmake/upstream/FindCUDAToolkit.cmake:``CUDAToolkit_TARGET_DIR``
cmake/upstream/FindCUDAToolkit.cmake:    The path to the CUDA Toolkit directory including the target architecture
cmake/upstream/FindCUDAToolkit.cmake:    ``CUDAToolkit_ROOT_DIR``.
cmake/upstream/FindCUDAToolkit.cmake:``CUDAToolkit_NVCC_EXECUTABLE``
cmake/upstream/FindCUDAToolkit.cmake:    The path to the NVIDIA CUDA compiler ``nvcc``.  Note that this path may
cmake/upstream/FindCUDAToolkit.cmake:    :variable:`CMAKE_CUDA_COMPILER <CMAKE_<LANG>_COMPILER>`.  ``nvcc`` must be
cmake/upstream/FindCUDAToolkit.cmake:    found to determine the CUDA Toolkit version as well as determining other
cmake/upstream/FindCUDAToolkit.cmake:# NOTE: much of this was simply extracted from FindCUDA.cmake.
cmake/upstream/FindCUDAToolkit.cmake:#   James Bigler, NVIDIA Corp (nvidia.com - jbigler)
cmake/upstream/FindCUDAToolkit.cmake:#   Abe Stephens, SCI Institute -- http://www.sci.utah.edu/~abe/FindCuda.html
cmake/upstream/FindCUDAToolkit.cmake:#   Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
cmake/upstream/FindCUDAToolkit.cmake:#   This code is licensed under the MIT License.  See the FindCUDA.cmake script
cmake/upstream/FindCUDAToolkit.cmake:if(CMAKE_CUDA_COMPILER_LOADED AND NOT CUDAToolkit_BIN_DIR)
cmake/upstream/FindCUDAToolkit.cmake:  get_filename_component(cuda_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
cmake/upstream/FindCUDAToolkit.cmake:  # use the already detected cuda compiler
cmake/upstream/FindCUDAToolkit.cmake:  set(CUDAToolkit_BIN_DIR "${cuda_dir}" CACHE PATH "")
cmake/upstream/FindCUDAToolkit.cmake:  mark_as_advanced(CUDAToolkit_BIN_DIR)
cmake/upstream/FindCUDAToolkit.cmake:  unset(cuda_dir)
cmake/upstream/FindCUDAToolkit.cmake:if(CUDAToolkit_BIN_DIR)
cmake/upstream/FindCUDAToolkit.cmake:  find_program(CUDAToolkit_NVCC_EXECUTABLE
cmake/upstream/FindCUDAToolkit.cmake:    PATHS ${CUDAToolkit_BIN_DIR}
cmake/upstream/FindCUDAToolkit.cmake:# Search using CUDAToolkit_ROOT
cmake/upstream/FindCUDAToolkit.cmake:find_program(CUDAToolkit_NVCC_EXECUTABLE
cmake/upstream/FindCUDAToolkit.cmake:  PATHS ENV CUDA_PATH
cmake/upstream/FindCUDAToolkit.cmake:# If the user specified CUDAToolkit_ROOT but nvcc could not be found, this is an error.
cmake/upstream/FindCUDAToolkit.cmake:if (NOT CUDAToolkit_NVCC_EXECUTABLE AND (DEFINED CUDAToolkit_ROOT OR DEFINED ENV{CUDAToolkit_ROOT}))
cmake/upstream/FindCUDAToolkit.cmake:  set(cuda_root_fail "${fail_base} CUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
cmake/upstream/FindCUDAToolkit.cmake:  set(env_cuda_root_fail "${fail_base} environment variable CUDAToolkit_ROOT=$ENV{CUDAToolkit_ROOT}")
cmake/upstream/FindCUDAToolkit.cmake:  if (CUDAToolkit_FIND_REQUIRED)
cmake/upstream/FindCUDAToolkit.cmake:    if (DEFINED CUDAToolkit_ROOT)
cmake/upstream/FindCUDAToolkit.cmake:      message(FATAL_ERROR ${cuda_root_fail})
cmake/upstream/FindCUDAToolkit.cmake:    elseif (DEFINED ENV{CUDAToolkit_ROOT})
cmake/upstream/FindCUDAToolkit.cmake:      message(FATAL_ERROR ${env_cuda_root_fail})
cmake/upstream/FindCUDAToolkit.cmake:    if (NOT CUDAToolkit_FIND_QUIETLY)
cmake/upstream/FindCUDAToolkit.cmake:      if (DEFINED CUDAToolkit_ROOT)
cmake/upstream/FindCUDAToolkit.cmake:        message(STATUS ${cuda_root_fail})
cmake/upstream/FindCUDAToolkit.cmake:      elseif (DEFINED ENV{CUDAToolkit_ROOT})
cmake/upstream/FindCUDAToolkit.cmake:        message(STATUS ${env_cuda_root_fail})
cmake/upstream/FindCUDAToolkit.cmake:    set(CUDAToolkit_FOUND FALSE)
cmake/upstream/FindCUDAToolkit.cmake:    unset(cuda_root_fail)
cmake/upstream/FindCUDAToolkit.cmake:    unset(env_cuda_root_fail)
cmake/upstream/FindCUDAToolkit.cmake:# CUDAToolkit_ROOT cmake / env variable not specified, try platform defaults.
cmake/upstream/FindCUDAToolkit.cmake:# - Linux: /usr/local/cuda-X.Y
cmake/upstream/FindCUDAToolkit.cmake:# - macOS: /Developer/NVIDIA/CUDA-X.Y
cmake/upstream/FindCUDAToolkit.cmake:# - Windows: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y
cmake/upstream/FindCUDAToolkit.cmake:# We will also search the default symlink location /usr/local/cuda first since
cmake/upstream/FindCUDAToolkit.cmake:# if CUDAToolkit_ROOT is not specified, it is assumed that the symlinked
cmake/upstream/FindCUDAToolkit.cmake:if (NOT CUDAToolkit_NVCC_EXECUTABLE)
cmake/upstream/FindCUDAToolkit.cmake:      set(platform_base "/usr/local/cuda-")
cmake/upstream/FindCUDAToolkit.cmake:      set(platform_base "/Developer/NVIDIA/CUDA-")
cmake/upstream/FindCUDAToolkit.cmake:    set(platform_base "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v")
cmake/upstream/FindCUDAToolkit.cmake:  # Build out a descending list of possible cuda installations, e.g.
cmake/upstream/FindCUDAToolkit.cmake:  # every possible version of CUDA installed, this wouldn't create any
cmake/upstream/FindCUDAToolkit.cmake:  # Force the global default /usr/local/cuda to the front on Unix.
cmake/upstream/FindCUDAToolkit.cmake:    list(INSERT search_paths 0 "/usr/local/cuda")
cmake/upstream/FindCUDAToolkit.cmake:  find_program(CUDAToolkit_NVCC_EXECUTABLE
cmake/upstream/FindCUDAToolkit.cmake:  if (NOT CUDAToolkit_NVCC_EXECUTABLE)
cmake/upstream/FindCUDAToolkit.cmake:    if (CUDAToolkit_FIND_REQUIRED)
cmake/upstream/FindCUDAToolkit.cmake:      message(FATAL_ERROR "Could not find nvcc, please set CUDAToolkit_ROOT.")
cmake/upstream/FindCUDAToolkit.cmake:    elseif(NOT CUDAToolkit_FIND_QUIETLY)
cmake/upstream/FindCUDAToolkit.cmake:      message(STATUS "Could not find nvcc, please set CUDAToolkit_ROOT.")
cmake/upstream/FindCUDAToolkit.cmake:    set(CUDAToolkit_FOUND FALSE)
cmake/upstream/FindCUDAToolkit.cmake:if(NOT CUDAToolkit_BIN_DIR AND CUDAToolkit_NVCC_EXECUTABLE)
cmake/upstream/FindCUDAToolkit.cmake:  get_filename_component(cuda_dir "${CUDAToolkit_NVCC_EXECUTABLE}" DIRECTORY)
cmake/upstream/FindCUDAToolkit.cmake:  set(CUDAToolkit_BIN_DIR "${cuda_dir}" CACHE PATH "" FORCE)
cmake/upstream/FindCUDAToolkit.cmake:  mark_as_advanced(CUDAToolkit_BIN_DIR)
cmake/upstream/FindCUDAToolkit.cmake:  unset(cuda_dir)
cmake/upstream/FindCUDAToolkit.cmake:if(CUDAToolkit_NVCC_EXECUTABLE AND
cmake/upstream/FindCUDAToolkit.cmake:   CUDAToolkit_NVCC_EXECUTABLE STREQUAL CMAKE_CUDA_COMPILER)
cmake/upstream/FindCUDAToolkit.cmake:  # Need to set these based off the already computed CMAKE_CUDA_COMPILER_VERSION value
cmake/upstream/FindCUDAToolkit.cmake:  if(CMAKE_CUDA_COMPILER_VERSION MATCHES [=[([0-9]+)\.([0-9]+)\.([0-9]+)]=])
cmake/upstream/FindCUDAToolkit.cmake:    set(CUDAToolkit_VERSION_MAJOR "${CMAKE_MATCH_1}")
cmake/upstream/FindCUDAToolkit.cmake:    set(CUDAToolkit_VERSION_MINOR "${CMAKE_MATCH_2}")
cmake/upstream/FindCUDAToolkit.cmake:    set(CUDAToolkit_VERSION_PATCH "${CMAKE_MATCH_3}")
cmake/upstream/FindCUDAToolkit.cmake:    set(CUDAToolkit_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")
cmake/upstream/FindCUDAToolkit.cmake:  execute_process (COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} "--version" OUTPUT_VARIABLE NVCC_OUT)
cmake/upstream/FindCUDAToolkit.cmake:    set(CUDAToolkit_VERSION_MAJOR "${CMAKE_MATCH_1}")
cmake/upstream/FindCUDAToolkit.cmake:    set(CUDAToolkit_VERSION_MINOR "${CMAKE_MATCH_2}")
cmake/upstream/FindCUDAToolkit.cmake:    set(CUDAToolkit_VERSION_PATCH "${CMAKE_MATCH_3}")
cmake/upstream/FindCUDAToolkit.cmake:    set(CUDAToolkit_VERSION  "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
cmake/upstream/FindCUDAToolkit.cmake:get_filename_component(CUDAToolkit_ROOT_DIR ${CUDAToolkit_BIN_DIR} DIRECTORY ABSOLUTE)
cmake/upstream/FindCUDAToolkit.cmake:    set (CUDAToolkit_TARGET_NAME "armv7-linux-androideabi")
cmake/upstream/FindCUDAToolkit.cmake:    set(CUDAToolkit_TARGET_NAME "armv7-linux-gnueabihf")
cmake/upstream/FindCUDAToolkit.cmake:      set(CUDAToolkit_TARGET_NAME "aarch64-linux-androideabi")
cmake/upstream/FindCUDAToolkit.cmake:      set(CUDAToolkit_TARGET_NAME "aarch64-linux")
cmake/upstream/FindCUDAToolkit.cmake:      set(CUDAToolkit_TARGET_NAME "x86_64-linux")
cmake/upstream/FindCUDAToolkit.cmake:  if (EXISTS "${CUDAToolkit_ROOT_DIR}/targets/${CUDAToolkit_TARGET_NAME}")
cmake/upstream/FindCUDAToolkit.cmake:    set(CUDAToolkit_TARGET_DIR "${CUDAToolkit_ROOT_DIR}/targets/${CUDAToolkit_TARGET_NAME}")
cmake/upstream/FindCUDAToolkit.cmake:    # add known CUDA target root path to the set of directories we search for programs, libraries and headers
cmake/upstream/FindCUDAToolkit.cmake:    list(PREPEND CMAKE_FIND_ROOT_PATH "${CUDAToolkit_TARGET_DIR}")
cmake/upstream/FindCUDAToolkit.cmake:    # found all cuda libraries so that searches for our cross-compilation
cmake/upstream/FindCUDAToolkit.cmake:    # libraries work when another cuda sdk is in CMAKE_PREFIX_PATH or
cmake/upstream/FindCUDAToolkit.cmake:    set(_CUDAToolkit_Pop_ROOT_PATH True)
cmake/upstream/FindCUDAToolkit.cmake:  set(CUDAToolkit_TARGET_DIR "${CUDAToolkit_ROOT_DIR}")
cmake/upstream/FindCUDAToolkit.cmake:  list(APPEND CMAKE_PREFIX_PATH ${CUDAToolkit_ROOT_DIR})
cmake/upstream/FindCUDAToolkit.cmake:  # found the cudart library.
cmake/upstream/FindCUDAToolkit.cmake:  set(_CUDAToolkit_Pop_Prefix True)
cmake/upstream/FindCUDAToolkit.cmake:find_path(CUDAToolkit_INCLUDE_DIR
cmake/upstream/FindCUDAToolkit.cmake:  NAMES cuda_runtime.h
cmake/upstream/FindCUDAToolkit.cmake:# And find the CUDA Runtime Library libcudart
cmake/upstream/FindCUDAToolkit.cmake:find_library(CUDA_CUDART
cmake/upstream/FindCUDAToolkit.cmake:  NAMES cudart
cmake/upstream/FindCUDAToolkit.cmake:if (NOT CUDA_CUDART)
cmake/upstream/FindCUDAToolkit.cmake:  find_library(CUDA_CUDART
cmake/upstream/FindCUDAToolkit.cmake:    NAMES cudart
cmake/upstream/FindCUDAToolkit.cmake:if (NOT CUDA_CUDART AND NOT CUDAToolkit_FIND_QUIETLY)
cmake/upstream/FindCUDAToolkit.cmake:  message(STATUS "Unable to find cudart library.")
cmake/upstream/FindCUDAToolkit.cmake:unset(CUDAToolkit_ROOT_DIR)
cmake/upstream/FindCUDAToolkit.cmake:if(_CUDAToolkit_Pop_Prefix)
cmake/upstream/FindCUDAToolkit.cmake:  unset(_CUDAToolkit_Pop_Prefix)
cmake/upstream/FindCUDAToolkit.cmake:find_package_handle_standard_args(CUDAToolkit
cmake/upstream/FindCUDAToolkit.cmake:    CUDAToolkit_INCLUDE_DIR
cmake/upstream/FindCUDAToolkit.cmake:    CUDA_CUDART
cmake/upstream/FindCUDAToolkit.cmake:    CUDAToolkit_NVCC_EXECUTABLE
cmake/upstream/FindCUDAToolkit.cmake:    CUDAToolkit_VERSION
cmake/upstream/FindCUDAToolkit.cmake:mark_as_advanced(CUDA_CUDART
cmake/upstream/FindCUDAToolkit.cmake:                 CUDAToolkit_INCLUDE_DIR
cmake/upstream/FindCUDAToolkit.cmake:                 CUDAToolkit_NVCC_EXECUTABLE
cmake/upstream/FindCUDAToolkit.cmake:if(CUDAToolkit_FOUND)
cmake/upstream/FindCUDAToolkit.cmake: set(CUDAToolkit_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIR})
cmake/upstream/FindCUDAToolkit.cmake: get_filename_component(CUDAToolkit_LIBRARY_DIR ${CUDA_CUDART} DIRECTORY ABSOLUTE)
cmake/upstream/FindCUDAToolkit.cmake:if(CUDAToolkit_FOUND)
cmake/upstream/FindCUDAToolkit.cmake:  function(_CUDAToolkit_find_and_add_import_lib lib_name)
cmake/upstream/FindCUDAToolkit.cmake:    find_library(CUDA_${lib_name}_LIBRARY
cmake/upstream/FindCUDAToolkit.cmake:      HINTS ${CUDAToolkit_LIBRARY_DIR}
cmake/upstream/FindCUDAToolkit.cmake:            ENV CUDA_PATH
cmake/upstream/FindCUDAToolkit.cmake:      PATH_SUFFIXES nvidia/current lib64 lib/x64 lib
cmake/upstream/FindCUDAToolkit.cmake:    if(NOT CUDA_${lib_name}_LIBRARY)
cmake/upstream/FindCUDAToolkit.cmake:      find_library(CUDA_${lib_name}_LIBRARY
cmake/upstream/FindCUDAToolkit.cmake:        HINTS ${CUDAToolkit_LIBRARY_DIR}
cmake/upstream/FindCUDAToolkit.cmake:              ENV CUDA_PATH
cmake/upstream/FindCUDAToolkit.cmake:    mark_as_advanced(CUDA_${lib_name}_LIBRARY)
cmake/upstream/FindCUDAToolkit.cmake:    if (NOT TARGET CUDA::${lib_name} AND CUDA_${lib_name}_LIBRARY)
cmake/upstream/FindCUDAToolkit.cmake:      add_library(CUDA::${lib_name} IMPORTED INTERFACE)
cmake/upstream/FindCUDAToolkit.cmake:      target_include_directories(CUDA::${lib_name} SYSTEM INTERFACE "${CUDAToolkit_INCLUDE_DIRS}")
cmake/upstream/FindCUDAToolkit.cmake:      target_link_libraries(CUDA::${lib_name} INTERFACE "${CUDA_${lib_name}_LIBRARY}")
cmake/upstream/FindCUDAToolkit.cmake:        if(TARGET CUDA::${dep})
cmake/upstream/FindCUDAToolkit.cmake:          target_link_libraries(CUDA::${lib_name} INTERFACE CUDA::${dep})
cmake/upstream/FindCUDAToolkit.cmake:  if(NOT TARGET CUDA::toolkit)
cmake/upstream/FindCUDAToolkit.cmake:    add_library(CUDA::toolkit IMPORTED INTERFACE)
cmake/upstream/FindCUDAToolkit.cmake:    target_include_directories(CUDA::toolkit SYSTEM INTERFACE "${CUDAToolkit_INCLUDE_DIRS}")
cmake/upstream/FindCUDAToolkit.cmake:    target_link_directories(CUDA::toolkit INTERFACE "${CUDAToolkit_LIBRARY_DIR}")
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(cuda_driver ALT cuda)
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(cudart)
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(cudart_static)
cmake/upstream/FindCUDAToolkit.cmake:  # setup dependencies that are required for cudart_static when building
cmake/upstream/FindCUDAToolkit.cmake:  # on linux. These are generally only required when using the CUDA toolkit
cmake/upstream/FindCUDAToolkit.cmake:  # when CUDA language is disabled
cmake/upstream/FindCUDAToolkit.cmake:  if(NOT TARGET CUDA::cudart_static_deps
cmake/upstream/FindCUDAToolkit.cmake:     AND TARGET CUDA::cudart_static)
cmake/upstream/FindCUDAToolkit.cmake:    add_library(CUDA::cudart_static_deps IMPORTED INTERFACE)
cmake/upstream/FindCUDAToolkit.cmake:    target_link_libraries(CUDA::cudart_static INTERFACE CUDA::cudart_static_deps)
cmake/upstream/FindCUDAToolkit.cmake:      target_link_libraries(CUDA::cudart_static_deps INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
cmake/upstream/FindCUDAToolkit.cmake:      # On Linux, you must link against librt when using the static cuda runtime.
cmake/upstream/FindCUDAToolkit.cmake:      find_library(CUDAToolkit_rt_LIBRARY rt)
cmake/upstream/FindCUDAToolkit.cmake:      mark_as_advanced(CUDAToolkit_rt_LIBRARY)
cmake/upstream/FindCUDAToolkit.cmake:      if(NOT CUDAToolkit_rt_LIBRARY)
cmake/upstream/FindCUDAToolkit.cmake:        message(WARNING "Could not find librt library, needed by CUDA::cudart_static")
cmake/upstream/FindCUDAToolkit.cmake:        target_link_libraries(CUDA::cudart_static_deps INTERFACE ${CUDAToolkit_rt_LIBRARY})
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(culibos) # it's a static library
cmake/upstream/FindCUDAToolkit.cmake:  foreach (cuda_lib cublas cufft curand cusparse nppc nvjpeg)
cmake/upstream/FindCUDAToolkit.cmake:    _CUDAToolkit_find_and_add_import_lib(${cuda_lib})
cmake/upstream/FindCUDAToolkit.cmake:    _CUDAToolkit_find_and_add_import_lib(${cuda_lib}_static DEPS culibos)
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(cufftw DEPS cufft)
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(cufftw DEPS cufft_static)
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(cusolver DEPS cublas cusparse)
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(cusolver_static DEPS cublas_static cusparse_static culibos)
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(nvgraph DEPS curand cusolver)
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(nvgraph_static DEPS curand_static cusolver_static)
cmake/upstream/FindCUDAToolkit.cmake:  foreach (cuda_lib nppial nppicc nppidei nppif nppig nppim nppist nppitc npps nppicom nppisu)
cmake/upstream/FindCUDAToolkit.cmake:    _CUDAToolkit_find_and_add_import_lib(${cuda_lib} DEPS nppc)
cmake/upstream/FindCUDAToolkit.cmake:    _CUDAToolkit_find_and_add_import_lib(${cuda_lib}_static DEPS nppc_static)
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(cupti
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(cupti_static
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(nvrtc DEPS cuda_driver)
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(nvml ALT nvidia-ml nvml)
cmake/upstream/FindCUDAToolkit.cmake:    # nvtools can be installed outside the CUDA toolkit directory
cmake/upstream/FindCUDAToolkit.cmake:    find_library(CUDA_nvToolsExt_LIBRARY
cmake/upstream/FindCUDAToolkit.cmake:            ENV CUDA_PATH
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(nvToolsExt ALT nvToolsExt64)
cmake/upstream/FindCUDAToolkit.cmake:  _CUDAToolkit_find_and_add_import_lib(OpenCL)
cmake/upstream/FindCUDAToolkit.cmake:if(_CUDAToolkit_Pop_ROOT_PATH)
cmake/upstream/FindCUDAToolkit.cmake:  unset(_CUDAToolkit_Pop_ROOT_PATH)
cmake/Modules/FindNCCL.cmake:# Try to find NCCL
cmake/Modules/FindNCCL.cmake:#  HOROVOD_NCCL_HOME: Base directory where all NCCL components are found
cmake/Modules/FindNCCL.cmake:#  HOROVOD_NCCL_INCLUDE: Directory where NCCL header is found
cmake/Modules/FindNCCL.cmake:#  HOROVOD_NCCL_LIB: Directory where NCCL library is found
cmake/Modules/FindNCCL.cmake:#  NCCL_FOUND
cmake/Modules/FindNCCL.cmake:#  NCCL_INCLUDE_DIRS
cmake/Modules/FindNCCL.cmake:#  NCCL_LIBRARIES
cmake/Modules/FindNCCL.cmake:#  NCCL_MAJOR_VERSION
cmake/Modules/FindNCCL.cmake:# The path hints include CUDAToolkit_* seeing as some folks
cmake/Modules/FindNCCL.cmake:# install NCCL in the same location as the CUDA toolkit.
cmake/Modules/FindNCCL.cmake:set(HOROVOD_NCCL_HOME $ENV{HOROVOD_NCCL_HOME} CACHE PATH "Folder contains NVIDIA NCCL")
cmake/Modules/FindNCCL.cmake:set(HOROVOD_NCCL_INCLUDE $ENV{HOROVOD_NCCL_INCLUDE} CACHE PATH "Folder contains NVIDIA NCCL headers")
cmake/Modules/FindNCCL.cmake:set(HOROVOD_NCCL_LIB $ENV{HOROVOD_NCCL_LIB} CACHE PATH "Folder contains NVIDIA NCCL libraries")
cmake/Modules/FindNCCL.cmake:list(APPEND NCCL_ROOT ${HOROVOD_NCCL_HOME} ${CUDAToolkit_LIBRARY_ROOT})
cmake/Modules/FindNCCL.cmake:# Compatible layer for CMake <3.12. NCCL_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
cmake/Modules/FindNCCL.cmake:list(APPEND CMAKE_PREFIX_PATH ${NCCL_ROOT})
cmake/Modules/FindNCCL.cmake:find_path(NCCL_INCLUDE_DIR
cmake/Modules/FindNCCL.cmake:        NAMES nccl.h
cmake/Modules/FindNCCL.cmake:        HINTS ${HOROVOD_NCCL_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})
cmake/Modules/FindNCCL.cmake:set(HOROVOD_NCCL_LINK $ENV{HOROVOD_NCCL_LINK})
cmake/Modules/FindNCCL.cmake:string(TOLOWER "${HOROVOD_NCCL_LINK}" lowercase_HOROVOD_NCCL_LINK)
cmake/Modules/FindNCCL.cmake:if (lowercase_HOROVOD_NCCL_LINK STREQUAL "shared")
cmake/Modules/FindNCCL.cmake:    set(NCCL_LIBNAME "nccl")
cmake/Modules/FindNCCL.cmake:    message(STATUS "Linking against shared NCCL library")
cmake/Modules/FindNCCL.cmake:    set(NCCL_LIBNAME "libnccl_static.a")
cmake/Modules/FindNCCL.cmake:    message(STATUS "Linking against static NCCL library")
cmake/Modules/FindNCCL.cmake:find_library(NCCL_LIBRARY
cmake/Modules/FindNCCL.cmake:        NAMES ${NCCL_LIBNAME}
cmake/Modules/FindNCCL.cmake:        HINTS ${HOROVOD_NCCL_LIB} ${CUDAToolkit_LIBRARY_DIR})
cmake/Modules/FindNCCL.cmake:string(TOLOWER "${CMAKE_CUDA_RUNTIME_LIBRARY}" lowercase_CMAKE_CUDA_RUNTIME_LIBRARY)
cmake/Modules/FindNCCL.cmake:if (NCCL_LIBRARY STREQUAL "NCCL_LIBRARY-NOTFOUND" AND NCCL_LIBNAME MATCHES "static" AND
cmake/Modules/FindNCCL.cmake:    NOT lowercase_HOROVOD_NCCL_LINK STREQUAL "static")
cmake/Modules/FindNCCL.cmake:    message(STATUS "Could not find static NCCL library. Trying to find shared lib instead.")
cmake/Modules/FindNCCL.cmake:    find_library(NCCL_LIBRARY
cmake/Modules/FindNCCL.cmake:            NAMES "nccl"
cmake/Modules/FindNCCL.cmake:            HINTS ${HOROVOD_NCCL_LIB} ${CUDAToolkit_LIBRARY_DIR})
cmake/Modules/FindNCCL.cmake:find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)
cmake/Modules/FindNCCL.cmake:if (NCCL_FOUND)
cmake/Modules/FindNCCL.cmake:    set(NCCL_HEADER_FILE "${NCCL_INCLUDE_DIR}/nccl.h")
cmake/Modules/FindNCCL.cmake:    message(STATUS "Determining NCCL version from the header file: ${NCCL_HEADER_FILE}")
cmake/Modules/FindNCCL.cmake:    file (STRINGS ${NCCL_HEADER_FILE} NCCL_MAJOR_VERSION_DEFINED
cmake/Modules/FindNCCL.cmake:            REGEX "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
cmake/Modules/FindNCCL.cmake:    if (NCCL_MAJOR_VERSION_DEFINED)
cmake/Modules/FindNCCL.cmake:        string (REGEX REPLACE "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+" ""
cmake/Modules/FindNCCL.cmake:                NCCL_MAJOR_VERSION ${NCCL_MAJOR_VERSION_DEFINED})
cmake/Modules/FindNCCL.cmake:        message(STATUS "NCCL_MAJOR_VERSION: ${NCCL_MAJOR_VERSION}")
cmake/Modules/FindNCCL.cmake:    file (STRINGS ${NCCL_HEADER_FILE} NCCL_VERSION_CODE_DEFINED
cmake/Modules/FindNCCL.cmake:        REGEX "^[ \t]*#define[ \t]+NCCL_VERSION_CODE[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
cmake/Modules/FindNCCL.cmake:    if (NCCL_VERSION_CODE_DEFINED)
cmake/Modules/FindNCCL.cmake:        string (REGEX REPLACE "^[ \t]*#define[ \t]+NCCL_VERSION_CODE[ \t]+" ""
cmake/Modules/FindNCCL.cmake:                NCCL_VERSION_CODE ${NCCL_VERSION_CODE_DEFINED})
cmake/Modules/FindNCCL.cmake:        message(STATUS "NCCL_VERSION_CODE: ${NCCL_VERSION_CODE}")
cmake/Modules/FindNCCL.cmake:    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
cmake/Modules/FindNCCL.cmake:    set(NCCL_LIBRARIES ${NCCL_LIBRARY})
cmake/Modules/FindNCCL.cmake:    message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
cmake/Modules/FindNCCL.cmake:    mark_as_advanced(NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
cmake/Modules/FindMxnet.cmake:#  Mxnet_USE_CUDA
cmake/Modules/FindMxnet.cmake:execute_process(COMMAND ${PY_EXE} -c "import os; import mxnet as mx; import build_utils; print(mx.__version__); print(mx.libinfo.find_include_path()); print(' '.join(mx.libinfo.find_lib_path())); print(build_utils.is_mx_mkldnn()); print(build_utils.is_mx_onednn()); print(build_utils.is_mx_cuda())"
cmake/Modules/FindMxnet.cmake:    list(GET Mxnet_OUTPUT 5 Mxnet_USE_CUDA)
cmake/Modules/FindMxnet.cmake:    string(TOUPPER ${Mxnet_USE_CUDA} Mxnet_USE_CUDA)
cmake/Modules/FindMxnet.cmake:    if (Mxnet_USE_CUDA)
cmake/Modules/FindMxnet.cmake:        set(Mxnet_COMPILE_FLAGS "${Mxnet_COMPILE_FLAGS} -DMSHADOW_USE_CUDA=1")
cmake/Modules/FindMxnet.cmake:        set(Mxnet_COMPILE_FLAGS "${Mxnet_COMPILE_FLAGS} -DMSHADOW_USE_CUDA=0")
cmake/Modules/FindMxnet.cmake:mark_as_advanced(Mxnet_INCLUDE_DIRS Mxnet_LIBRARIES Mxnet_COMPILE_FLAGS Mxnet_USE_MKLDNN Mxnet_USE_ONEDNN Mxnet_USE_CUDA Mxnet_VERSION)
cmake/Modules/FindNVTX.cmake:# NVTX comes with the CUDA toolkit so we use those include dirs to search for the header-only variation of NVTX.
cmake/Modules/FindNVTX.cmake:set(HOROVOD_NVTX_INCLUDE $ENV{HOROVOD_NVTX_INCLUDE} CACHE PATH "Folder containing NVIDIA NVTX3 headers")
cmake/Modules/FindNVTX.cmake:          HINTS ${HOROVOD_NVTX_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})
cmake/Modules/FindROCM.cmake:# Try to find ROCM
cmake/Modules/FindROCM.cmake:#  HOROVOD_ROCM_HOME: Base directory where all ROCM components are found
cmake/Modules/FindROCM.cmake:#  ROCM_FOUND
cmake/Modules/FindROCM.cmake:#  ROCM_INCLUDE_DIRS
cmake/Modules/FindROCM.cmake:#  ROCM_LIBRARIES
cmake/Modules/FindROCM.cmake:#  ROCM_COMPILE_FLAGS
cmake/Modules/FindROCM.cmake:set(HOROVOD_ROCM_HOME $ENV{HOROVOD_ROCM_HOME})
cmake/Modules/FindROCM.cmake:if(NOT DEFINED HOROVOD_ROCM_HOME)
cmake/Modules/FindROCM.cmake:    set(HOROVOD_ROCM_HOME "/opt/rocm")
cmake/Modules/FindROCM.cmake:set(HIP_PATH "${HOROVOD_ROCM_HOME}/hip")
cmake/Modules/FindROCM.cmake:list(APPEND ROCM_ROOT ${HOROVOD_ROCM_HOME})
cmake/Modules/FindROCM.cmake:# Compatible layer for CMake <3.12. ROCM_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
cmake/Modules/FindROCM.cmake:list(APPEND CMAKE_PREFIX_PATH ${ROCM_ROOT})
cmake/Modules/FindROCM.cmake:  find_library(ROCM_LIBRARIES NAMES ${hip_library_name} HINTS ${HIP_PATH}/lib)
cmake/Modules/FindROCM.cmake:set(ROCM_INCLUDE_DIRS ${HIP_INCLUDE_DIRS})
cmake/Modules/FindROCM.cmake:set(ROCM_COMPILE_FLAGS "-D__HIP_PLATFORM_HCC__=1")
cmake/Modules/FindROCM.cmake:find_package_handle_standard_args(ROCM DEFAULT_MSG ROCM_LIBRARIES)
cmake/Modules/FindPytorch.cmake:#  Pytorch_CUDA
cmake/Modules/FindPytorch.cmake:#  Pytorch_ROCM
cmake/Modules/FindPytorch.cmake:execute_process(COMMAND ${PY_EXE} -c "import torch; from torch.utils.cpp_extension import CUDA_HOME; print(True if ((torch.version.cuda is not None) and (CUDA_HOME is not None)) else False)"
cmake/Modules/FindPytorch.cmake:                OUTPUT_VARIABLE Pytorch_CUDA OUTPUT_STRIP_TRAILING_WHITESPACE)
cmake/Modules/FindPytorch.cmake:string(REGEX REPLACE "No (CUDA|ROCm) runtime[^\n]*\n?" "" Pytorch_CUDA "${Pytorch_CUDA}")
cmake/Modules/FindPytorch.cmake:string(TOUPPER "${Pytorch_CUDA}" Pytorch_CUDA)
cmake/Modules/FindPytorch.cmake:execute_process(COMMAND ${PY_EXE} -c "import torch; from torch.utils.cpp_extension import ROCM_HOME; print(True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False)"
cmake/Modules/FindPytorch.cmake:                OUTPUT_VARIABLE Pytorch_ROCM OUTPUT_STRIP_TRAILING_WHITESPACE)
cmake/Modules/FindPytorch.cmake:string(REGEX REPLACE "No (CUDA|ROCm) runtime[^\n]*\n?" "" Pytorch_ROCM "${Pytorch_ROCM}")
cmake/Modules/FindPytorch.cmake:string(TOUPPER "${Pytorch_ROCM}" Pytorch_ROCM)
cmake/Modules/FindPytorch.cmake:if(Pytorch_ROCM)
cmake/Modules/FindPytorch.cmake:                    OUTPUT_VARIABLE _Pytorch_ROCM_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
cmake/Modules/FindPytorch.cmake:    string(REGEX REPLACE "No (CUDA|ROCm) runtime[^\n]*\n?" "" _Pytorch_ROCM_FLAGS "${_Pytorch_ROCM_FLAGS}")
cmake/Modules/FindPytorch.cmake:    set(Pytorch_COMPILE_FLAGS "${_Pytorch_ROCM_FLAGS}")
cmake/Modules/FindPytorch.cmake:if (Pytorch_CUDA OR Pytorch_ROCM)
cmake/Modules/FindPytorch.cmake:    set(Pytorch_EXT "CUDAExtension")
cmake/Modules/FindPytorch.cmake:mark_as_advanced(Pytorch_INCLUDE_DIRS Pytorch_LIBRARY_DIRS Pytorch_LIBRARIES Pytorch_COMPILE_FLAGS Pytorch_VERSION Pytorch_CUDA Pytorch_ROCM Pytorch_CXX11)
cmake/build_utils.py:def is_mx_cuda():
cmake/build_utils.py:        return features.is_enabled('CUDA')
cmake/build_utils.py:                    if 'cuda' in str(output):
cmake/build_utils.py:    cuda_home = os.environ.get('HOROVOD_CUDA_HOME', '/usr/local/cuda')
cmake/build_utils.py:    cuda_nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
cmake/build_utils.py:    for nvcc_bin in ['nvcc', cuda_nvcc]:
cmake/build_utils.py:    raise RuntimeError('Cannot find `nvcc`. `nvcc` is required to build Horovod with GPU operations. '
cmake/build_utils.py:                       'Make sure it is added to your path or in $HOROVOD_CUDA_HOME/bin.')
cmake/build_utils.py:    cc_list_env = os.environ.get('HOROVOD_BUILD_CUDA_CC_LIST')
cmake/build_utils.py:    # Invoke nvcc and extract all supported compute capabilities for CUDA toolkit version
cmake/build_utils.py:                                           f"sed -n -e '/gpu-architecture <arch>/,/gpu-code <code>/ p' | "
cmake/build_utils.py:                                           f"sed -n -e '/Allowed values/,/gpu-code <code>/ p' | "
cmake/Utilities.cmake:# Set GPU OP
cmake/Utilities.cmake:macro(SET_GPU_OP PARAM SUPPORTED)
cmake/Utilities.cmake:    if(DEFINED HOROVOD_GPU_OPERATIONS)
cmake/Utilities.cmake:            message(FATAL_ERROR "Cannot specify both HOROVOD_GPU_OPERATIONS and ${PARAM} options. "
cmake/Utilities.cmake:        set(${PARAM} "${HOROVOD_GPU_OPERATIONS}")
examples/ray/pytorch_ray_elastic.py:    '--no-cuda',
examples/ray/pytorch_ray_elastic.py:    help='disables CUDA training')
examples/ray/pytorch_ray_elastic.py:    args.cuda = not args.no_cuda and torch.cuda.is_available()
examples/ray/pytorch_ray_elastic.py:    if args.cuda:
examples/ray/pytorch_ray_elastic.py:        # Horovod: pin GPU to local rank.
examples/ray/pytorch_ray_elastic.py:        torch.cuda.set_device(hvd.local_rank())
examples/ray/pytorch_ray_elastic.py:        torch.cuda.manual_seed(args.seed)
examples/ray/pytorch_ray_elastic.py:    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
examples/ray/pytorch_ray_elastic.py:    if args.cuda:
examples/ray/pytorch_ray_elastic.py:        # Move model to GPU.
examples/ray/pytorch_ray_elastic.py:        model.cuda()
examples/ray/pytorch_ray_elastic.py:        # If using GPU Adasum allreduce, scale learning rate by local_size.
examples/ray/pytorch_ray_elastic.py:        if args.use_adasum and hvd.nccl_built():
examples/ray/pytorch_ray_elastic.py:                if args.cuda:
examples/ray/pytorch_ray_elastic.py:                    data, target = data.cuda(), target.cuda()
examples/ray/pytorch_ray_elastic.py:            if args.cuda:
examples/ray/pytorch_ray_elastic.py:                data, target = data.cuda(), target.cuda()
examples/ray/pytorch_ray_elastic.py:    executor = ElasticRayExecutor(settings, use_gpu=True, cpus_per_slot=2)
examples/ray/basic_ray_elastic.py:    '--lr', type=float, default=0.01, help='learning rate for a single GPU')
examples/ray/basic_ray_elastic.py:    '--no-cuda',
examples/ray/basic_ray_elastic.py:    help='disables CUDA training')
examples/ray/basic_ray_elastic.py:    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
examples/ray/basic_ray_elastic.py:        if args.cuda:
examples/ray/basic_ray_elastic.py:            data, target = data.cuda(), target.cuda()
examples/ray/basic_ray_elastic.py:    args.cuda = not args.no_cuda and torch.cuda.is_available()
examples/ray/basic_ray_elastic.py:    if args.cuda:
examples/ray/basic_ray_elastic.py:        # Horovod: pin GPU to local rank.
examples/ray/basic_ray_elastic.py:        torch.cuda.set_device(hvd.local_rank())
examples/ray/basic_ray_elastic.py:        torch.cuda.manual_seed(args.seed)
examples/ray/basic_ray_elastic.py:    if args.cuda:
examples/ray/basic_ray_elastic.py:        model.cuda()
examples/ray/basic_ray_elastic.py:    # Horovod: scale learning rate by the number of GPUs.
examples/ray/basic_ray_elastic.py:        use_gpu=True,
examples/ray/basic_ray_elastic.py:        settings, use_gpu=True, cpus_per_slot=1, override_discovery=False)
examples/ray/tensorflow2_mnist_ray.py:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/ray/tensorflow2_mnist_ray.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
examples/ray/tensorflow2_mnist_ray.py:    for gpu in gpus:
examples/ray/tensorflow2_mnist_ray.py:        tf.config.experimental.set_memory_growth(gpu, True)
examples/ray/tensorflow2_mnist_ray.py:    if gpus:
examples/ray/tensorflow2_mnist_ray.py:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()],
examples/ray/tensorflow2_mnist_ray.py:                                                   'GPU')
examples/ray/tensorflow2_mnist_ray.py:    # Horovod: adjust learning rate based on number of GPUs.
examples/ray/tensorflow2_mnist_ray.py:    # Horovod: adjust number of steps based on number of GPUs.
examples/ray/tensorflow2_mnist_ray.py:        settings, num_hosts=2, num_slots=4, use_gpu=False, cpus_per_slot=8)
examples/elastic/tensorflow2/tensorflow2_synthetic_benchmark_elastic.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/elastic/tensorflow2/tensorflow2_synthetic_benchmark_elastic.py:                    help='disables CUDA training')
examples/elastic/tensorflow2/tensorflow2_synthetic_benchmark_elastic.py:args.cuda = not args.no_cuda
examples/elastic/tensorflow2/tensorflow2_synthetic_benchmark_elastic.py:# Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/elastic/tensorflow2/tensorflow2_synthetic_benchmark_elastic.py:if args.cuda:
examples/elastic/tensorflow2/tensorflow2_synthetic_benchmark_elastic.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
examples/elastic/tensorflow2/tensorflow2_synthetic_benchmark_elastic.py:    for gpu in gpus:
examples/elastic/tensorflow2/tensorflow2_synthetic_benchmark_elastic.py:        tf.config.experimental.set_memory_growth(gpu, True)
examples/elastic/tensorflow2/tensorflow2_synthetic_benchmark_elastic.py:    if gpus:
examples/elastic/tensorflow2/tensorflow2_synthetic_benchmark_elastic.py:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
examples/elastic/tensorflow2/tensorflow2_synthetic_benchmark_elastic.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
examples/elastic/tensorflow2/tensorflow2_synthetic_benchmark_elastic.py:device = 'GPU' if args.cuda else 'CPU'
examples/elastic/tensorflow2/tensorflow2_keras_mnist_elastic.py:# Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/elastic/tensorflow2/tensorflow2_keras_mnist_elastic.py:gpus = tf.config.experimental.list_physical_devices('GPU')
examples/elastic/tensorflow2/tensorflow2_keras_mnist_elastic.py:for gpu in gpus:
examples/elastic/tensorflow2/tensorflow2_keras_mnist_elastic.py:    tf.config.experimental.set_memory_growth(gpu, True)
examples/elastic/tensorflow2/tensorflow2_keras_mnist_elastic.py:if gpus:
examples/elastic/tensorflow2/tensorflow2_keras_mnist_elastic.py:    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
examples/elastic/tensorflow2/tensorflow2_keras_mnist_elastic.py:# Horovod: adjust learning rate based on number of GPUs.
examples/elastic/tensorflow2/tensorflow2_keras_mnist_elastic.py:# Horovod: adjust number of steps based on number of GPUs.
examples/elastic/tensorflow2/tensorflow2_mnist_elastic.py:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/elastic/tensorflow2/tensorflow2_mnist_elastic.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
examples/elastic/tensorflow2/tensorflow2_mnist_elastic.py:    for gpu in gpus:
examples/elastic/tensorflow2/tensorflow2_mnist_elastic.py:        tf.config.experimental.set_memory_growth(gpu, True)
examples/elastic/tensorflow2/tensorflow2_mnist_elastic.py:    if gpus:
examples/elastic/tensorflow2/tensorflow2_mnist_elastic.py:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
examples/elastic/tensorflow2/tensorflow2_mnist_elastic.py:    # Horovod: adjust learning rate based on number of GPUs.
examples/elastic/tensorflow2/tensorflow2_mnist_elastic.py:        # Horovod: adjust number of steps based on number of GPUs.
examples/elastic/pytorch/pytorch_mnist_elastic.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/elastic/pytorch/pytorch_mnist_elastic.py:                    help='disables CUDA training')
examples/elastic/pytorch/pytorch_mnist_elastic.py:args.cuda = not args.no_cuda and torch.cuda.is_available()
examples/elastic/pytorch/pytorch_mnist_elastic.py:if args.cuda:
examples/elastic/pytorch/pytorch_mnist_elastic.py:    # Horovod: pin GPU to local rank.
examples/elastic/pytorch/pytorch_mnist_elastic.py:    torch.cuda.set_device(hvd.local_rank())
examples/elastic/pytorch/pytorch_mnist_elastic.py:    torch.cuda.manual_seed(args.seed)
examples/elastic/pytorch/pytorch_mnist_elastic.py:kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
examples/elastic/pytorch/pytorch_mnist_elastic.py:if args.cuda:
examples/elastic/pytorch/pytorch_mnist_elastic.py:    # Move model to GPU.
examples/elastic/pytorch/pytorch_mnist_elastic.py:    model.cuda()
examples/elastic/pytorch/pytorch_mnist_elastic.py:    # If using GPU Adasum allreduce, scale learning rate by local_size.
examples/elastic/pytorch/pytorch_mnist_elastic.py:    if args.use_adasum and hvd.nccl_built():
examples/elastic/pytorch/pytorch_mnist_elastic.py:            if args.cuda:
examples/elastic/pytorch/pytorch_mnist_elastic.py:                data, target = data.cuda(), target.cuda()
examples/elastic/pytorch/pytorch_mnist_elastic.py:        if args.cuda:
examples/elastic/pytorch/pytorch_mnist_elastic.py:            data, target = data.cuda(), target.cuda()
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:                    help='learning rate for a single GPU')
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:                    help='disables CUDA training')
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:            if args.cuda:
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:                data, target = data.cuda(), target.cuda()
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:                if args.cuda:
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:                    data, target = data.cuda(), target.cuda()
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:    args.cuda = not args.no_cuda and torch.cuda.is_available()
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:    if args.cuda:
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:        # Horovod: pin GPU to local rank.
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:        torch.cuda.set_device(hvd.local_rank())
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:        torch.cuda.manual_seed(args.seed)
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:    if args.cuda:
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:        # Move model to GPU.
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:        model.cuda()
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:        # If using GPU Adasum allreduce, scale learning rate by local_size.
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:        if args.use_adasum and hvd.nccl_built():
examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py:    # Horovod: scale learning rate by the number of GPUs.
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:                    help='disables CUDA training')
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:args.cuda = not args.no_cuda and torch.cuda.is_available()
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:if args.cuda:
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:    # Horovod: pin GPU to local rank.
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:    torch.cuda.set_device(hvd.local_rank())
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:if args.cuda:
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:    # Move model to GPU.
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:    model.cuda()
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:    # If using GPU Adasum allreduce, scale learning rate by local_size.
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:    if args.use_adasum and hvd.nccl_built():
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:if args.cuda:
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:    data, target = data.cuda(), target.cuda()
examples/elastic/pytorch/pytorch_synthetic_benchmark_elastic.py:device = 'GPU' if args.cuda else 'CPU'
examples/elastic/tensorflow/tensorflow_keras_mnist_elastic.py:# Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/elastic/tensorflow/tensorflow_keras_mnist_elastic.py:config.gpu_options.allow_growth = True
examples/elastic/tensorflow/tensorflow_keras_mnist_elastic.py:config.gpu_options.visible_device_list = str(hvd.local_rank())
examples/elastic/tensorflow/tensorflow_keras_mnist_elastic.py:# Horovod: adjust learning rate based on number of GPUs.
examples/elastic/tensorflow/tensorflow_keras_mnist_elastic.py:    # Horovod: adjust number of steps based on number of GPUs and number of epochs
examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py:                    help='disables CUDA training')
examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py:args.cuda = not args.no_cuda
examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py:device = 'GPU' if args.cuda else 'CPU'
examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py:# Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py:if args.cuda:
examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py:    for gpu in gpus:
examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py:        tf.config.experimental.set_memory_growth(gpu, True)
examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py:    if gpus:
examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
examples/tensorflow2/tensorflow2_mnist.py:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/tensorflow2/tensorflow2_mnist.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
examples/tensorflow2/tensorflow2_mnist.py:    for gpu in gpus:
examples/tensorflow2/tensorflow2_mnist.py:        tf.config.experimental.set_memory_growth(gpu, True)
examples/tensorflow2/tensorflow2_mnist.py:    if gpus:
examples/tensorflow2/tensorflow2_mnist.py:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
examples/tensorflow2/tensorflow2_mnist.py:    # Horovod: adjust learning rate based on number of GPUs.
examples/tensorflow2/tensorflow2_mnist.py:    # Horovod: adjust number of steps based on number of GPUs.
examples/tensorflow2/tensorflow2_keras_mnist.py:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/tensorflow2/tensorflow2_keras_mnist.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
examples/tensorflow2/tensorflow2_keras_mnist.py:    for gpu in gpus:
examples/tensorflow2/tensorflow2_keras_mnist.py:        tf.config.experimental.set_memory_growth(gpu, True)
examples/tensorflow2/tensorflow2_keras_mnist.py:    if gpus:
examples/tensorflow2/tensorflow2_keras_mnist.py:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
examples/tensorflow2/tensorflow2_keras_mnist.py:    # Horovod: adjust learning rate based on number of GPUs.
examples/tensorflow2/tensorflow2_keras_mnist.py:    # Horovod: adjust number of steps based on number of GPUs.
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:    for gpu in gpus:
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:        tf.config.experimental.set_memory_growth(gpu, True)
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:    if gpus:
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:        # Horovod: adjust learning rate based on number of GPUs.
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:        # Horovod: adjust number of steps based on number of GPUs.
examples/tensorflow2/tensorflow2_synthetic_benchmark.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/tensorflow2/tensorflow2_synthetic_benchmark.py:                    help='disables CUDA training')
examples/tensorflow2/tensorflow2_synthetic_benchmark.py:args.cuda = not args.no_cuda
examples/tensorflow2/tensorflow2_synthetic_benchmark.py:# Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/tensorflow2/tensorflow2_synthetic_benchmark.py:if args.cuda:
examples/tensorflow2/tensorflow2_synthetic_benchmark.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
examples/tensorflow2/tensorflow2_synthetic_benchmark.py:    for gpu in gpus:
examples/tensorflow2/tensorflow2_synthetic_benchmark.py:        tf.config.experimental.set_memory_growth(gpu, True)
examples/tensorflow2/tensorflow2_synthetic_benchmark.py:    if gpus:
examples/tensorflow2/tensorflow2_synthetic_benchmark.py:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
examples/tensorflow2/tensorflow2_synthetic_benchmark.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
examples/tensorflow2/tensorflow2_synthetic_benchmark.py:device = 'GPU' if args.cuda else 'CPU'
examples/tensorflow2/tensorflow2_mnist_data_service.py:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:    for gpu in gpus:
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:        tf.config.experimental.set_memory_growth(gpu, True)
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:    if gpus:
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:    # Horovod: adjust learning rate based on number of GPUs.
examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:    # Horovod: adjust number of steps based on number of GPUs.
examples/keras/keras_mnist.py:# Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/keras/keras_mnist.py:config.gpu_options.allow_growth = True
examples/keras/keras_mnist.py:config.gpu_options.visible_device_list = str(hvd.local_rank())
examples/keras/keras_mnist.py:# Horovod: adjust number of epochs based on number of GPUs.
examples/keras/keras_mnist.py:# Horovod: adjust learning rate based on number of GPUs.
examples/keras/keras_imagenet_resnet50.py:                    help='learning rate for a single GPU')
examples/keras/keras_imagenet_resnet50.py:# Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/keras/keras_imagenet_resnet50.py:config.gpu_options.allow_growth = True
examples/keras/keras_imagenet_resnet50.py:config.gpu_options.visible_device_list = str(hvd.local_rank())
examples/keras/keras_imagenet_resnet50.py:# Horovod: adjust learning rate based on number of GPUs.
examples/keras/keras_mnist_advanced.py:# Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/keras/keras_mnist_advanced.py:config.gpu_options.allow_growth = True
examples/keras/keras_mnist_advanced.py:config.gpu_options.visible_device_list = str(hvd.local_rank())
examples/keras/keras_mnist_advanced.py:# Horovod: adjust learning rate based on number of GPUs.
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:    for gpu in gpus:
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:        tf.config.experimental.set_memory_growth(gpu, True)
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:    if gpus:
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:        # Horovod: adjust learning rate based on number of GPUs.
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py:        # Horovod: adjust number of steps based on number of GPUs.
examples/spark/tensorflow2/tensorflow2_mnist_data_service.py:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:    for gpu in gpus:
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:        tf.config.experimental.set_memory_growth(gpu, True)
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:    if gpus:
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:    # Horovod: adjust learning rate based on number of GPUs.
examples/spark/tensorflow2/tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py:    # Horovod: adjust number of steps based on number of GPUs.
examples/spark/keras/keras_spark3_rossmann.py.patch:>     DISCOVERY_SCRIPT = 'get_gpu_resources.sh'
examples/spark/keras/keras_spark3_rossmann.py.patch:>     # Whether to infer on GPU.
examples/spark/keras/keras_spark3_rossmann.py.patch:>     GPU_INFERENCE_ENABLED = False
examples/spark/keras/keras_spark3_rossmann.py.patch:>     # Cluster for GPU inference.
examples/spark/keras/keras_spark3_rossmann.py.patch:>     GPU_INFERENCE_CLUSTER = 'local-cluster[2,1,1024]'  # or 'spark://hostname:7077'
examples/spark/keras/keras_spark3_rossmann.py.patch:<         config.gpu_options.visible_device_list = str(hvd.local_rank())
examples/spark/keras/keras_spark3_rossmann.py.patch:>         config.gpu_options.visible_device_list = get_available_devices()[0]
examples/spark/keras/keras_spark3_rossmann.py.patch:>     def set_gpu_conf(conf):
examples/spark/keras/keras_spark3_rossmann.py.patch:>         # - Requires YARN 3.1 or higher to support GPUs
examples/spark/keras/keras_spark3_rossmann.py.patch:>         #   multiple executors don’t see the same GPU on the same host.
examples/spark/keras/keras_spark3_rossmann.py.patch:>         #   or other way to make sure that 2 executors don’t try to use same GPU.
examples/spark/keras/keras_spark3_rossmann.py.patch:>         # - Requires GPU support and isolation.
examples/spark/keras/keras_spark3_rossmann.py.patch:>         # - Add conf.set(“spark.executor.resource.gpu.discoveryScript”, DISCOVERY_SCRIPT)
examples/spark/keras/keras_spark3_rossmann.py.patch:>         # - Add conf.set(“spark.executor.resource.gpu.vendor”, “nvidia.com”)
examples/spark/keras/keras_spark3_rossmann.py.patch:>         conf = conf.set("spark.worker.resource.gpu.discoveryScript", DISCOVERY_SCRIPT)
examples/spark/keras/keras_spark3_rossmann.py.patch:>         conf = conf.set("spark.worker.resource.gpu.amount", 1)
examples/spark/keras/keras_spark3_rossmann.py.patch:>         conf = conf.set("spark.task.resource.gpu.amount", "1")
examples/spark/keras/keras_spark3_rossmann.py.patch:>         conf = conf.set("spark.executor.resource.gpu.amount", "1")
examples/spark/keras/keras_spark3_rossmann.py.patch:>     conf = set_gpu_conf(conf)
examples/spark/keras/keras_spark3_rossmann.py.patch:>     if GPU_INFERENCE_ENABLED:
examples/spark/keras/keras_spark3_rossmann.py.patch:>         if GPU_INFERENCE_CLUSTER:
examples/spark/keras/keras_spark3_rossmann.py.patch:>             conf.setMaster(GPU_INFERENCE_CLUSTER)
examples/spark/keras/keras_spark3_rossmann.py.patch:>         conf = set_gpu_conf(conf)
examples/spark/keras/keras_spark3_rossmann.py.patch:<             # Do not use GPUs for prediction, use single CPU core per task.
examples/spark/keras/keras_spark3_rossmann.py.patch:<             config = tf.ConfigProto(device_count={'GPU': 0})
examples/spark/keras/keras_spark3_rossmann.py.patch:>             if GPU_INFERENCE_ENABLED:
examples/spark/keras/keras_spark3_rossmann.py.patch:>                 config.gpu_options.allow_growth = True
examples/spark/keras/keras_spark3_rossmann.py.patch:>                 config.gpu_options.visible_device_list = TaskContext.get().resources()['gpu'].addresses[0]
examples/spark/keras/keras_spark3_rossmann.py.patch:>                 # Do not use GPUs for prediction, use single CPU core per task.
examples/spark/keras/keras_spark3_rossmann.py.patch:>                 config = tf.ConfigProto(device_count={'GPU': 0})
examples/spark/keras/keras_spark3_rossmann.py:                         'should be set up to provide a Spark task per multiple CPU cores, or per GPU, e.g. by'
examples/spark/keras/keras_spark3_rossmann.py:                         'supplying `-c <NUM_GPUS>` in Spark Standalone mode. Example: spark://hostname:7077')
examples/spark/keras/keras_spark3_rossmann.py:    DISCOVERY_SCRIPT = 'get_gpu_resources.sh'
examples/spark/keras/keras_spark3_rossmann.py:    # Whether to infer on GPU.
examples/spark/keras/keras_spark3_rossmann.py:    GPU_INFERENCE_ENABLED = False
examples/spark/keras/keras_spark3_rossmann.py:    # Cluster for GPU inference.
examples/spark/keras/keras_spark3_rossmann.py:    GPU_INFERENCE_CLUSTER = 'local-cluster[2,1,1024]'  # or 'spark://hostname:7077'
examples/spark/keras/keras_spark3_rossmann.py:    # Do not use GPU for the session creation.
examples/spark/keras/keras_spark3_rossmann.py:    config = tf.ConfigProto(device_count={'GPU': 0})
examples/spark/keras/keras_spark3_rossmann.py:        # Horovod: pin GPU to be used to process local rank (one GPU per process), if GPUs are available.
examples/spark/keras/keras_spark3_rossmann.py:        config.gpu_options.allow_growth = True
examples/spark/keras/keras_spark3_rossmann.py:        config.gpu_options.visible_device_list = get_available_devices()[0]
examples/spark/keras/keras_spark3_rossmann.py:    def set_gpu_conf(conf):
examples/spark/keras/keras_spark3_rossmann.py:        # - Requires YARN 3.1 or higher to support GPUs
examples/spark/keras/keras_spark3_rossmann.py:        #   multiple executors don’t see the same GPU on the same host.
examples/spark/keras/keras_spark3_rossmann.py:        #   or other way to make sure that 2 executors don’t try to use same GPU.
examples/spark/keras/keras_spark3_rossmann.py:        # - Requires GPU support and isolation.
examples/spark/keras/keras_spark3_rossmann.py:        # - Add conf.set(“spark.executor.resource.gpu.discoveryScript”, DISCOVERY_SCRIPT)
examples/spark/keras/keras_spark3_rossmann.py:        # - Add conf.set(“spark.executor.resource.gpu.vendor”, “nvidia.com”)
examples/spark/keras/keras_spark3_rossmann.py:        conf = conf.set("spark.worker.resource.gpu.discoveryScript", DISCOVERY_SCRIPT)
examples/spark/keras/keras_spark3_rossmann.py:        conf = conf.set("spark.worker.resource.gpu.amount", 1)
examples/spark/keras/keras_spark3_rossmann.py:        conf = conf.set("spark.task.resource.gpu.amount", "1")
examples/spark/keras/keras_spark3_rossmann.py:        conf = conf.set("spark.executor.resource.gpu.amount", "1")
examples/spark/keras/keras_spark3_rossmann.py:    conf = set_gpu_conf(conf)
examples/spark/keras/keras_spark3_rossmann.py:    if GPU_INFERENCE_ENABLED:
examples/spark/keras/keras_spark3_rossmann.py:        if GPU_INFERENCE_CLUSTER:
examples/spark/keras/keras_spark3_rossmann.py:            conf.setMaster(GPU_INFERENCE_CLUSTER)
examples/spark/keras/keras_spark3_rossmann.py:        conf = set_gpu_conf(conf)
examples/spark/keras/keras_spark3_rossmann.py:            if GPU_INFERENCE_ENABLED:
examples/spark/keras/keras_spark3_rossmann.py:                config.gpu_options.allow_growth = True
examples/spark/keras/keras_spark3_rossmann.py:                config.gpu_options.visible_device_list = TaskContext.get().resources()['gpu'].addresses[0]
examples/spark/keras/keras_spark3_rossmann.py:                # Do not use GPUs for prediction, use single CPU core per task.
examples/spark/keras/keras_spark3_rossmann.py:                config = tf.ConfigProto(device_count={'GPU': 0})
examples/spark/keras/get_gpu_resources.sh:# This script is a basic example script to get resource information about NVIDIA GPUs.
examples/spark/keras/get_gpu_resources.sh:# It assumes the drivers are properly installed and the nvidia-smi command is available.
examples/spark/keras/get_gpu_resources.sh:# spark.{driver/executor}.resource.gpu.discoveryScript to allow the driver or executor to discover
examples/spark/keras/get_gpu_resources.sh:# the GPUs it was allocated. It assumes you are running within an isolated container where the
examples/spark/keras/get_gpu_resources.sh:# GPUs are allocated exclusively to that driver or executor.
examples/spark/keras/get_gpu_resources.sh:# spark.{driver/executor}.resource.gpu.discoveryScript config.
examples/spark/keras/get_gpu_resources.sh:# Example output: {"name": "gpu", "addresses":["0","1","2","3","4","5","6","7"]}
examples/spark/keras/get_gpu_resources.sh:ADDRS=`nvidia-smi --query-gpu=index --format=csv,noheader | sed -e ':a' -e 'N' -e'$!ba' -e 's/\n/","/g'`
examples/spark/keras/get_gpu_resources.sh:echo {\"name\": \"gpu\", \"addresses\":[\"$ADDRS\"]}
examples/spark/keras/keras_spark_rossmann_run.py:                         'should be set up to provide a Spark task per multiple CPU cores, or per GPU, e.g. by'
examples/spark/keras/keras_spark_rossmann_run.py:                         'supplying `-c <NUM_GPUS>` in Spark Standalone mode. Example: spark://hostname:7077')
examples/spark/keras/keras_spark_rossmann_run.py:    # Do not use GPU for the session creation.
examples/spark/keras/keras_spark_rossmann_run.py:    config = tf.ConfigProto(device_count={'GPU': 0})
examples/spark/keras/keras_spark_rossmann_run.py:        # Horovod: pin GPU to be used to process local rank (one GPU per process), if GPUs are available.
examples/spark/keras/keras_spark_rossmann_run.py:        config.gpu_options.allow_growth = True
examples/spark/keras/keras_spark_rossmann_run.py:        config.gpu_options.visible_device_list = str(hvd.local_rank())
examples/spark/keras/keras_spark_rossmann_run.py:            # Do not use GPUs for prediction, use single CPU core per task.
examples/spark/keras/keras_spark_rossmann_run.py:            config = tf.ConfigProto(device_count={'GPU': 0})
examples/spark/keras/keras_spark_mnist.py:    # Disable GPUs when building the model to prevent memory leaks
examples/spark/keras/keras_spark_mnist.py:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
examples/spark/keras/keras_spark_mnist.py:        keras.backend.set_session(tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})))
examples/spark/keras/keras_spark_rossmann_estimator.py:                         'should be set up to provide a Spark task per multiple CPU cores, or per GPU, e.g. by'
examples/spark/keras/keras_spark_rossmann_estimator.py:                         'supplying `-c <NUM_GPUS>` in Spark Standalone mode')
examples/spark/keras/keras_spark_rossmann_estimator.py:    # Disable GPUs when building the model to prevent memory leaks
examples/spark/keras/keras_spark_rossmann_estimator.py:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
examples/spark/keras/keras_spark_rossmann_estimator.py:        K.set_session(tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})))
examples/pytorch/pytorch_mnist.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/pytorch/pytorch_mnist.py:                    help='disables CUDA training')
examples/pytorch/pytorch_mnist.py:            if args.cuda:
examples/pytorch/pytorch_mnist.py:                data, target = data.cuda(), target.cuda()
examples/pytorch/pytorch_mnist.py:            with torch.cuda.amp.autocast():
examples/pytorch/pytorch_mnist.py:            if args.cuda:
examples/pytorch/pytorch_mnist.py:                data, target = data.cuda(), target.cuda()
examples/pytorch/pytorch_mnist.py:            if args.cuda:
examples/pytorch/pytorch_mnist.py:                data, target = data.cuda(), target.cuda()
examples/pytorch/pytorch_mnist.py:    if args.cuda:
examples/pytorch/pytorch_mnist.py:        # Horovod: pin GPU to local rank.
examples/pytorch/pytorch_mnist.py:        torch.cuda.set_device(hvd.local_rank())
examples/pytorch/pytorch_mnist.py:        torch.cuda.manual_seed(args.seed)
examples/pytorch/pytorch_mnist.py:            raise ValueError("Mixed precision is only supported with cuda enabled.")
examples/pytorch/pytorch_mnist.py:        raise ValueError("""Mixed precision is using torch.cuda.amp.autocast(),
examples/pytorch/pytorch_mnist.py:    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
examples/pytorch/pytorch_mnist.py:    if args.cuda:
examples/pytorch/pytorch_mnist.py:        # Move model to GPU.
examples/pytorch/pytorch_mnist.py:        model.cuda()
examples/pytorch/pytorch_mnist.py:        # If using GPU Adasum allreduce, scale learning rate by local_size.
examples/pytorch/pytorch_mnist.py:        if args.use_adasum and hvd.nccl_built():
examples/pytorch/pytorch_mnist.py:        scaler = torch.cuda.amp.GradScaler()
examples/pytorch/pytorch_mnist.py:    args.cuda = not args.no_cuda and torch.cuda.is_available()
examples/pytorch/pytorch_imagenet_resnet50.py:                    help='learning rate for a single GPU')
examples/pytorch/pytorch_imagenet_resnet50.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/pytorch/pytorch_imagenet_resnet50.py:                    help='disables CUDA training')
examples/pytorch/pytorch_imagenet_resnet50.py:            if args.cuda:
examples/pytorch/pytorch_imagenet_resnet50.py:                data, target = data.cuda(), target.cuda()
examples/pytorch/pytorch_imagenet_resnet50.py:                if args.cuda:
examples/pytorch/pytorch_imagenet_resnet50.py:                    data, target = data.cuda(), target.cuda()
examples/pytorch/pytorch_imagenet_resnet50.py:    args.cuda = not args.no_cuda and torch.cuda.is_available()
examples/pytorch/pytorch_imagenet_resnet50.py:    if args.cuda:
examples/pytorch/pytorch_imagenet_resnet50.py:        # Horovod: pin GPU to local rank.
examples/pytorch/pytorch_imagenet_resnet50.py:        torch.cuda.set_device(hvd.local_rank())
examples/pytorch/pytorch_imagenet_resnet50.py:        torch.cuda.manual_seed(args.seed)
examples/pytorch/pytorch_imagenet_resnet50.py:    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
examples/pytorch/pytorch_imagenet_resnet50.py:    if args.cuda:
examples/pytorch/pytorch_imagenet_resnet50.py:        # Move model to GPU.
examples/pytorch/pytorch_imagenet_resnet50.py:        model.cuda()
examples/pytorch/pytorch_imagenet_resnet50.py:        # If using GPU Adasum allreduce, scale learning rate by local_size.
examples/pytorch/pytorch_imagenet_resnet50.py:        if args.use_adasum and hvd.nccl_built():
examples/pytorch/pytorch_imagenet_resnet50.py:    # Horovod: scale learning rate by the number of GPUs.
examples/pytorch/pytorch_synthetic_benchmark.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/pytorch/pytorch_synthetic_benchmark.py:                    help='disables CUDA training')
examples/pytorch/pytorch_synthetic_benchmark.py:args.cuda = not args.no_cuda and torch.cuda.is_available()
examples/pytorch/pytorch_synthetic_benchmark.py:if args.cuda:
examples/pytorch/pytorch_synthetic_benchmark.py:    # Horovod: pin GPU to local rank.
examples/pytorch/pytorch_synthetic_benchmark.py:    torch.cuda.set_device(hvd.local_rank())
examples/pytorch/pytorch_synthetic_benchmark.py:if args.cuda:
examples/pytorch/pytorch_synthetic_benchmark.py:    # Move model to GPU.
examples/pytorch/pytorch_synthetic_benchmark.py:    model.cuda()
examples/pytorch/pytorch_synthetic_benchmark.py:    # If using GPU Adasum allreduce, scale learning rate by local_size.
examples/pytorch/pytorch_synthetic_benchmark.py:    if args.use_adasum and hvd.nccl_built():
examples/pytorch/pytorch_synthetic_benchmark.py:if args.cuda:
examples/pytorch/pytorch_synthetic_benchmark.py:    data, target = data.cuda(), target.cuda()
examples/pytorch/pytorch_synthetic_benchmark.py:device = 'GPU' if args.cuda else 'CPU'
examples/pytorch/pytorch_lightning_mnist.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/pytorch/pytorch_lightning_mnist.py:                    help='disables CUDA training')
examples/pytorch/pytorch_lightning_mnist.py:        if args.cuda:
examples/pytorch/pytorch_lightning_mnist.py:            data, target = data.cuda(), target.cuda()
examples/pytorch/pytorch_lightning_mnist.py:    args.cuda = not args.no_cuda and torch.cuda.is_available()
examples/pytorch/pytorch_lightning_mnist.py:    if args.cuda:
examples/pytorch/pytorch_lightning_mnist.py:        torch.cuda.set_device(hvd.local_rank())
examples/pytorch/pytorch_lightning_mnist.py:        torch.cuda.manual_seed(args.seed)
examples/pytorch/pytorch_lightning_mnist.py:                          gpus=(1 if args.cuda else 0),
examples/pytorch/pytorch_lightning_mnist.py:        if args.cuda:
examples/pytorch/pytorch_lightning_mnist.py:            model = model.cuda()
examples/tensorflow/tensorflow_mnist_estimator.py:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/tensorflow/tensorflow_mnist_estimator.py:    config.gpu_options.allow_growth = True
examples/tensorflow/tensorflow_mnist_estimator.py:    config.gpu_options.visible_device_list = str(hvd.local_rank())
examples/tensorflow/tensorflow_mnist_estimator.py:    # Horovod: adjust number of steps based on number of GPUs.
examples/tensorflow/tensorflow_mnist.py:    # By default, Adasum doesn't need scaling when increasing batch size. If used with NCCL,
examples/tensorflow/tensorflow_mnist.py:        lr_scaler = hvd.local_size() if hvd.nccl_built() else 1
examples/tensorflow/tensorflow_mnist.py:        # Horovod: adjust number of steps based on number of GPUs.
examples/tensorflow/tensorflow_mnist.py:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/tensorflow/tensorflow_mnist.py:    config.gpu_options.allow_growth = True
examples/tensorflow/tensorflow_mnist.py:    config.gpu_options.visible_device_list = str(hvd.local_rank())
examples/tensorflow/tensorflow_word2vec.py:    # Horovod: adjust learning rate based on number of GPUs.
examples/tensorflow/tensorflow_word2vec.py:# Horovod: adjust number of steps based on number of GPUs.
examples/tensorflow/tensorflow_word2vec.py:# Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/tensorflow/tensorflow_word2vec.py:config.gpu_options.allow_growth = True
examples/tensorflow/tensorflow_word2vec.py:config.gpu_options.visible_device_list = str(hvd.local_rank())
examples/tensorflow/tensorflow_synthetic_benchmark.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/tensorflow/tensorflow_synthetic_benchmark.py:                    help='disables CUDA training')
examples/tensorflow/tensorflow_synthetic_benchmark.py:args.cuda = not args.no_cuda
examples/tensorflow/tensorflow_synthetic_benchmark.py:# Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/tensorflow/tensorflow_synthetic_benchmark.py:if args.cuda:
examples/tensorflow/tensorflow_synthetic_benchmark.py:    config.gpu_options.allow_growth = True
examples/tensorflow/tensorflow_synthetic_benchmark.py:    config.gpu_options.visible_device_list = str(hvd.local_rank())
examples/tensorflow/tensorflow_synthetic_benchmark.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
examples/tensorflow/tensorflow_synthetic_benchmark.py:    config.gpu_options.allow_growth = False
examples/tensorflow/tensorflow_synthetic_benchmark.py:    config.gpu_options.visible_device_list = ''
examples/tensorflow/tensorflow_synthetic_benchmark.py:# By default, Adasum doesn't need scaling when increasing batch size. If used with NCCL,
examples/tensorflow/tensorflow_synthetic_benchmark.py:    lr_scaler = hvd.local_size() if args.cuda and hvd.nccl_built() else 1
examples/tensorflow/tensorflow_synthetic_benchmark.py:device = 'GPU' if args.cuda else 'CPU'
examples/tensorflow/tensorflow_mnist_eager.py:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/tensorflow/tensorflow_mnist_eager.py:    config.gpu_options.visible_device_list = str(hvd.local_rank())
examples/tensorflow/tensorflow_mnist_eager.py:    # Horovod: adjust learning rate based on number of GPUs.
examples/tensorflow/tensorflow_mnist_eager.py:    # Horovod: adjust number of steps based on number of GPUs.
examples/tensorflow/tensorflow_keras_mnist.py:# Horovod: pin GPU to be used to process local rank (one GPU per process)
examples/tensorflow/tensorflow_keras_mnist.py:config.gpu_options.allow_growth = True
examples/tensorflow/tensorflow_keras_mnist.py:config.gpu_options.visible_device_list = str(hvd.local_rank())
examples/tensorflow/tensorflow_keras_mnist.py:# Horovod: adjust number of epochs based on number of GPUs.
examples/tensorflow/tensorflow_keras_mnist.py:# Horovod: adjust learning rate based on number of GPUs.
examples/mxnet/mxnet_imagenet_resnet50.py:                    help='learning rate for a single GPU (default: 0.05)')
examples/mxnet/mxnet_imagenet_resnet50.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/mxnet/mxnet_imagenet_resnet50.py:                    help='disables CUDA training (default: False)')
examples/mxnet/mxnet_imagenet_resnet50.py:# Horovod: pin GPU to local rank
examples/mxnet/mxnet_imagenet_resnet50.py:context = mx.cpu(local_rank) if args.no_cuda else mx.gpu(local_rank)
examples/mxnet/mxnet_mnist.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/mxnet/mxnet_mnist.py:                    help='disable training on GPU (default: False)')
examples/mxnet/mxnet_mnist.py:    context = mx.cpu(hvd.local_rank()) if args.no_cuda else mx.gpu(hvd.local_rank())
examples/mxnet/mxnet_mnist.py:    if not args.no_cuda:
examples/mxnet/mxnet_mnist.py:        # Disable CUDA if there are no GPUs.
examples/mxnet/mxnet_mnist.py:        if not mx.test_utils.list_gpus():
examples/mxnet/mxnet_mnist.py:            args.no_cuda = True
examples/mxnet/mxnet2_mnist.py:parser.add_argument('--no-cuda', action='store_true', default=False,
examples/mxnet/mxnet2_mnist.py:                    help='disable training on GPU (default: False)')
examples/mxnet/mxnet2_mnist.py:    context = mx.cpu(hvd.local_rank()) if args.no_cuda else mx.gpu(hvd.local_rank())
examples/mxnet/mxnet2_mnist.py:    if not args.no_cuda:
examples/mxnet/mxnet2_mnist.py:        # Disable CUDA if there are no GPUs.
examples/mxnet/mxnet2_mnist.py:        if not mx.test_utils.list_gpus():
examples/mxnet/mxnet2_mnist.py:            args.no_cuda = True
.gitignore:horovod/torch/test_cuda/
horovod/ray/elastic_v2.py:    def __init__(self, use_gpu=False, cpus_per_worker=1, gpus_per_worker=1):
horovod/ray/elastic_v2.py:        self.use_gpu = use_gpu
horovod/ray/elastic_v2.py:        self.gpus_per_worker = gpus_per_worker
horovod/ray/elastic_v2.py:                     f"{gpus_per_worker} GPU per slot.")
horovod/ray/elastic_v2.py:            if self.use_gpu:
horovod/ray/elastic_v2.py:                gpu_slots = resources.get("GPU", 0) // self.gpus_per_worker
horovod/ray/elastic_v2.py:                slots = min(slots, gpu_slots)
horovod/ray/elastic_v2.py:                 use_gpu=False,
horovod/ray/elastic_v2.py:                 gpus_per_worker=1,
horovod/ray/elastic_v2.py:            use_gpu=use_gpu,
horovod/ray/elastic_v2.py:            gpus_per_worker=gpus_per_worker)
horovod/ray/elastic_v2.py:        use_gpu (bool): Whether to use GPU for allocation. TODO: this
horovod/ray/elastic_v2.py:        gpus_per_worker (int): Number of GPU resources to allocate to
horovod/ray/elastic_v2.py:        use_gpu (bool): Whether to use GPU for allocation. TODO: this
horovod/ray/elastic_v2.py:        gpus_per_worker (int): Number of GPU resources to allocate to
horovod/ray/elastic_v2.py:                use_gpu: bool = False,
horovod/ray/elastic_v2.py:                gpus_per_worker: Optional[int] = None,
horovod/ray/elastic_v2.py:                use_gpu=use_gpu,
horovod/ray/elastic_v2.py:                gpus_per_worker=gpus_per_worker)
horovod/ray/elastic_v2.py:        self.gpus_per_worker = gpus_per_worker
horovod/ray/elastic_v2.py:        self.use_gpu = use_gpu
horovod/ray/elastic_v2.py:            num_gpus=int(self.use_gpu) * self.gpus_per_worker,
horovod/ray/elastic_v2.py:        if self.use_gpu:
horovod/ray/elastic_v2.py:                "CUDA_VISIBLE_DEVICES":
horovod/ray/strategy.py:                 use_gpu: bool, cpus_per_worker: int, gpus_per_worker: int):
horovod/ray/strategy.py:        self.use_gpu = use_gpu
horovod/ray/strategy.py:        self.gpus_per_worker = gpus_per_worker or 1
horovod/ray/strategy.py:        num_gpus = self.gpus_per_worker * self.num_workers_per_host * int(
horovod/ray/strategy.py:            self.use_gpu)
horovod/ray/strategy.py:        return dict(CPU=num_cpus, GPU=num_gpus)
horovod/ray/strategy.py:            gpu_id_futures = []
horovod/ray/strategy.py:                    num_gpus=self.gpus_per_worker * int(self.use_gpu),
horovod/ray/strategy.py:                if self.use_gpu:
horovod/ray/strategy.py:                    gpu_id_futures.append(worker.get_gpu_ids.remote())
horovod/ray/strategy.py:            if len(gpu_id_futures) > 0:
horovod/ray/strategy.py:                # By setting CUDA VISIBLE DEVICES to ALL GPUs,
horovod/ray/strategy.py:                # CUDA will be able to detect adjacent devices and use IPC
horovod/ray/strategy.py:                gpu_ids = sum(ray.get(gpu_id_futures), [])
horovod/ray/strategy.py:                assert len(gpu_ids) == len(
horovod/ray/strategy.py:                    set(gpu_ids)) == self.num_workers_per_host, gpu_ids
horovod/ray/strategy.py:                all_ids = ",".join([str(gpu_id) for gpu_id in gpu_ids])
horovod/ray/strategy.py:                            "CUDA_VISIBLE_DEVICES":
horovod/ray/strategy.py:    def __init__(self, *, settings, num_workers, use_gpu, cpus_per_worker,
horovod/ray/strategy.py:                 gpus_per_worker, placement_group=None, force_create_placement_group=False):
horovod/ray/strategy.py:        self.gpus_per_worker = gpus_per_worker or 1
horovod/ray/strategy.py:        self.use_gpu = use_gpu
horovod/ray/strategy.py:        num_gpus = self.gpus_per_worker * int(self.use_gpu)
horovod/ray/strategy.py:        return dict(CPU=num_cpus, GPU=num_gpus)
horovod/ray/strategy.py:                num_gpus=self.gpus_per_worker * int(self.use_gpu),
horovod/ray/strategy.py:        if self.use_gpu:
horovod/ray/strategy.py:            gpus = ray.get(
horovod/ray/strategy.py:                [worker.get_gpu_ids.remote() for worker in self.workers])
horovod/ray/strategy.py:            node_id_to_gpus = defaultdict(list)
horovod/ray/strategy.py:            for worker, node_id, worker_gpu_ids in zip(self.workers, node_ids,
horovod/ray/strategy.py:                                                       gpus):
horovod/ray/strategy.py:                node_id_to_gpus[node_id].extend(worker_gpu_ids)
horovod/ray/strategy.py:            for node_id, gpu_ids in node_id_to_gpus.items():
horovod/ray/strategy.py:                all_ids = ",".join([str(gpu_id) for gpu_id in gpu_ids])
horovod/ray/strategy.py:                            "CUDA_VISIBLE_DEVICES":
horovod/ray/elastic.py:    def __init__(self, use_gpu=False, cpus_per_slot=1, gpus_per_slot=1):
horovod/ray/elastic.py:        self.use_gpu = use_gpu
horovod/ray/elastic.py:        self.gpus_per_slot = gpus_per_slot
horovod/ray/elastic.py:                     f"{gpus_per_slot} GPU per slot.")
horovod/ray/elastic.py:            if self.use_gpu:
horovod/ray/elastic.py:                gpu_slots = resources.get("GPU", 0) // self.gpus_per_slot
horovod/ray/elastic.py:                slots = min(slots, gpu_slots)
horovod/ray/elastic.py:                 use_gpu=False,
horovod/ray/elastic.py:                 gpus_per_slot=1,
horovod/ray/elastic.py:            use_gpu=use_gpu,
horovod/ray/elastic.py:            gpus_per_slot=gpus_per_slot)
horovod/ray/elastic.py:        use_gpu (bool): Whether to use GPU for allocation.
horovod/ray/elastic.py:        gpus_per_slot (int): Number of GPU resources to allocate to
horovod/ray/elastic.py:            settings, use_gpu=True, cpus_per_slot=2)
horovod/ray/elastic.py:                 use_gpu: bool = False,
horovod/ray/elastic.py:                 gpus_per_slot: Optional[int] = None,
horovod/ray/elastic.py:        if gpus_per_slot and not use_gpu:
horovod/ray/elastic.py:            raise ValueError("gpus_per_slot is set, but use_gpu is False. "
horovod/ray/elastic.py:                             "use_gpu must be True if gpus_per_slot is set. ")
horovod/ray/elastic.py:        gpus_per_slot = gpus_per_slot or int(use_gpu)
horovod/ray/elastic.py:        if use_gpu and gpus_per_slot < 1:
horovod/ray/elastic.py:                f"gpus_per_slot must be >= 1: Got {gpus_per_slot}.")
horovod/ray/elastic.py:                use_gpu=use_gpu,
horovod/ray/elastic.py:                gpus_per_slot=gpus_per_slot)
horovod/ray/elastic.py:        self.gpus_per_slot = gpus_per_slot
horovod/ray/elastic.py:        self.use_gpu = use_gpu
horovod/ray/elastic.py:            num_gpus=int(self.use_gpu) * self.gpus_per_slot,
horovod/ray/elastic.py:        if self.use_gpu:
horovod/ray/elastic.py:                "CUDA_VISIBLE_DEVICES":
horovod/ray/runner.py:        use_gpu (bool): Whether to use GPU for allocation. TODO: this
horovod/ray/runner.py:        gpus_per_worker (int): Number of GPU resources to allocate to
horovod/ray/runner.py:        use_gpu (bool): Whether to use GPU for allocation. TODO: this
horovod/ray/runner.py:        gpus_per_worker (int): Number of GPU resources to allocate to
horovod/ray/runner.py:            use_gpu: bool = False,
horovod/ray/runner.py:            gpus_per_worker: Optional[int] = None,
horovod/ray/runner.py:                use_gpu=use_gpu,
horovod/ray/runner.py:                gpus_per_worker=gpus_per_worker
horovod/ray/runner.py:                use_gpu=use_gpu,
horovod/ray/runner.py:                gpus_per_worker=gpus_per_worker,
horovod/ray/runner.py:        use_gpu (bool): Whether to use GPU for allocation. TODO: this
horovod/ray/runner.py:        gpus_per_worker (int): Number of GPU resources to allocate to
horovod/ray/runner.py:                 use_gpu: bool = False,
horovod/ray/runner.py:                 gpus_per_worker: Optional[int] = None,
horovod/ray/runner.py:        self.use_gpu = use_gpu
horovod/ray/runner.py:        self.gpus_per_worker = gpus_per_worker or 1
horovod/ray/runner.py:                use_gpu=self.use_gpu,
horovod/ray/runner.py:                gpus_per_worker=self.gpus_per_worker,
horovod/ray/runner.py:                use_gpu=self.use_gpu,
horovod/ray/runner.py:                gpus_per_worker=self.gpus_per_worker)
horovod/ray/adapter.py:    use_gpu: bool = False
horovod/ray/adapter.py:    gpus_per_worker: Optional[int] = None
horovod/ray/adapter.py:        if self.gpus_per_worker and not self.use_gpu:
horovod/ray/adapter.py:            raise ValueError("gpus_per_worker is set, but use_gpu is False. "
horovod/ray/adapter.py:                             "use_gpu must be True if gpus_per_worker is "
horovod/ray/adapter.py:        if self.use_gpu and isinstance(self.gpus_per_worker,
horovod/ray/adapter.py:                                  int) and self.gpus_per_worker < 1:
horovod/ray/adapter.py:                f"gpus_per_worker must be >= 1: Got {self.gpus_per_worker}.")
horovod/ray/adapter.py:        self.gpus_per_worker = self.gpus_per_worker or int(self.use_gpu)
horovod/ray/worker.py:    def get_gpu_ids(self) -> List[int]:
horovod/ray/worker.py:        """Return list of CUDA device IDs available to this worker."""
horovod/ray/worker.py:        return ray.get_gpu_ids()
horovod/ray/utils.py:        "NCCL_SOCKET_IFNAME": ",".join(nics),  # TODO
horovod/torch/optimizer.py:from horovod.torch.mpi_ops import rocm_built
horovod/torch/optimizer.py:        if rocm_built():
horovod/torch/optimizer.py:            raise ValueError('gradient_predivide_factor not supported yet with ROCm')
horovod/torch/mpi_ops.py:# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/torch/mpi_ops.py:from horovod.common.util import check_installed_version, get_average_backwards_compatibility_fun, gpu_available, num_rank_is_power_2
horovod/torch/mpi_ops.py:nccl_built = _basics.nccl_built
horovod/torch/mpi_ops.py:cuda_built = _basics.cuda_built
horovod/torch/mpi_ops.py:rocm_built = _basics.rocm_built
horovod/torch/mpi_ops.py:        if rocm_built():
horovod/torch/mpi_ops.py:            # For ROCm, perform averaging at framework level
horovod/torch/mpi_ops.py:        if tensor.device.type != 'cpu' and gpu_available('torch'):
horovod/torch/mpi_ops.py:            if nccl_built():
horovod/torch/mpi_ops.py:                    raise NotImplementedError('Running GPU Adasum on heterogeneous cluster is not supported yet.')
horovod/torch/mpi_ops.py:                    raise NotImplementedError('Running GPU Adasum with non-power of 2 nodes is not supported yet.')
horovod/torch/mpi_ops.py:                if rocm_built():
horovod/torch/mpi_ops.py:                    # For ROCm, perform averaging at framework level
horovod/torch/mpi_ops.py:                warnings.warn('Adasum reduction does not currently support GPU reduction using MPI. Tensors are '
horovod/torch/mpi_ops.py:                              'copied to CPU memory instead. To use Adasum for GPU reduction, please compile Horovod '
horovod/torch/mpi_ops.py:                              'with HOROVOD_GPU_OPERATIONS=NCCL.')
horovod/torch/mpi_ops.py:        if rocm_built():
horovod/torch/mpi_ops.py:            # For ROCm, perform averaging at framework level
horovod/torch/mpi_ops.py:        if tensors[0].device.type != 'cpu' and gpu_available('torch'):
horovod/torch/mpi_ops.py:            if nccl_built():
horovod/torch/mpi_ops.py:                    raise NotImplementedError('Running GPU Adasum on heterogeneous cluster is not supported yet.')
horovod/torch/mpi_ops.py:                    raise NotImplementedError('Running GPU Adasum with non-power of 2 nodes is not supported yet.')
horovod/torch/mpi_ops.py:                if rocm_built():
horovod/torch/mpi_ops.py:                    # For ROCm, perform averaging at framework level
horovod/torch/mpi_ops.py:                warnings.warn('Adasum reduction does not currently support GPU reduction using MPI. Tensors are '
horovod/torch/mpi_ops.py:                              'copied to CPU memory instead. To use Adasum for GPU reduction, please compile Horovod '
horovod/torch/mpi_ops.py:                              'with HOROVOD_GPU_OPERATIONS=NCCL.')
horovod/torch/sync_batch_norm.py:    This is very useful in situations where each GPU can fit a very small number of examples.
horovod/torch/sync_batch_norm.py:    .. note:: Only GPU input tensors are supported in the training mode.
horovod/torch/sync_batch_norm.py:        # currently only GPU input is supported by underlying kernel from PyTorch
horovod/torch/sync_batch_norm.py:        if not input.is_cuda:
horovod/torch/sync_batch_norm.py:            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')
horovod/torch/CMakeLists.txt:if (HAVE_CUDA AND NOT Pytorch_CUDA)
horovod/torch/CMakeLists.txt:    message(FATAL_ERROR "Horovod build with GPU support was requested but this PyTorch installation does not support CUDA.")
horovod/torch/CMakeLists.txt:elseif (Pytorch_CUDA AND NOT HAVE_CUDA)
horovod/torch/CMakeLists.txt:    add_cuda()
horovod/torch/CMakeLists.txt:if (HAVE_ROCM AND NOT Pytorch_ROCM)
horovod/torch/CMakeLists.txt:    message(FATAL_ERROR "Horovod build with GPU support was requested but this PyTorch installation does not support ROCm.")
horovod/torch/CMakeLists.txt:elseif (Pytorch_ROCM AND NOT HAVE_ROCM)
horovod/torch/CMakeLists.txt:    add_definitions(-DHAVE_ROCM=1 -DHAVE_GPU=1)
horovod/torch/CMakeLists.txt:if (Pytorch_ROCM)
horovod/torch/CMakeLists.txt:if(HAVE_CUDA)
horovod/torch/CMakeLists.txt:        list(APPEND PYTORCH_LINKER_LIBS horovod_cuda_kernels)
horovod/torch/CMakeLists.txt:        list(APPEND PYTORCH_LINKER_LIBS compatible_horovod_cuda_kernels)
horovod/torch/CMakeLists.txt:if(HAVE_ROCM)
horovod/torch/CMakeLists.txt:        list(APPEND PYTORCH_LINKER_LIBS horovod_cuda_kernels)
horovod/torch/CMakeLists.txt:        list(APPEND PYTORCH_LINKER_LIBS compatible_horovod_cuda_kernels)
horovod/torch/CMakeLists.txt:# Later versions of PyTorch that use ROCm's hipify step will rename files.
horovod/torch/CMakeLists.txt:if(Pytorch_ROCM AND "${HIPIFY_HAS_VERSION}" STREQUAL "True")
horovod/torch/CMakeLists.txt:                            "${PROJECT_SOURCE_DIR}/horovod/torch/cuda_util.cc"
horovod/torch/adapter_v2.cc:#if HAVE_GPU
horovod/torch/adapter_v2.cc:#include <c10/cuda/CUDAStream.h>
horovod/torch/adapter_v2.cc:#include <c10/cuda/CUDAException.h>
horovod/torch/adapter_v2.cc:#include "cuda_util.h"
horovod/torch/adapter_v2.cc:    tensor_ = ::torch::empty({size}, ::torch::device(::torch::kCUDA).dtype(::torch::kByte));
horovod/torch/adapter_v2.cc:#if HAVE_GPU
horovod/torch/adapter_v2.cc:    // On GPU allocation is asynchronous, we need to wait for it to
horovod/torch/adapter_v2.cc:    auto stream = c10::cuda::getCurrentCUDAStream(device_);
horovod/torch/adapter_v2.cc:    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
horovod/torch/adapter_v2.cc:#if HAVE_GPU
horovod/torch/adapter_v2.cc:      // On GPU allocation is asynchronous, we need to wait for it to
horovod/torch/adapter_v2.cc:      auto stream = c10::cuda::getCurrentCUDAStream(device_);
horovod/torch/adapter_v2.cc:      C10_CUDA_CHECK(cudaStreamSynchronize(stream));
horovod/torch/adapter_v2.cc:      device_ != CPU_DEVICE_ID ? ::torch::kCUDA : ::torch::kCPU;
horovod/torch/adapter_v2.cc:#if HAVE_GPU
horovod/torch/adapter_v2.cc:    // On GPU allocation is asynchronous, we need to wait for it to
horovod/torch/adapter_v2.cc:    auto stream = c10::cuda::getCurrentCUDAStream(device_);
horovod/torch/adapter_v2.cc:    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
horovod/torch/cuda_util.h:#ifndef HOROVOD_TORCH_CUDA_UTIL_H
horovod/torch/cuda_util.h:#define HOROVOD_TORCH_CUDA_UTIL_H
horovod/torch/cuda_util.h:#endif // HOROVOD_TORCH_CUDA_UTIL_H
horovod/torch/ready_event.h:#if HAVE_GPU
horovod/torch/ready_event.h:#include "cuda_runtime.h"
horovod/torch/ready_event.h:#if HAVE_GPU
horovod/torch/ready_event.h:  gpuEvent_t event() const override;
horovod/torch/ready_event.h:  gpuEvent_t cuda_event_ = nullptr;
horovod/torch/__init__.py:    from horovod.torch.mpi_ops import nccl_built, ddl_built, ccl_built, cuda_built, rocm_built
horovod/torch/__init__.py:def _check_has_gpu():
horovod/torch/__init__.py:    return torch.cuda.is_available()
horovod/torch/cuda_util.cc:#if HAVE_GPU
horovod/torch/cuda_util.cc:#include "cuda.h"
horovod/torch/cuda_util.cc:#include "cuda_runtime.h"
horovod/torch/cuda_util.cc:#include <c10/cuda/CUDAGuard.h>
horovod/torch/cuda_util.cc:#include "cuda_util.h"
horovod/torch/cuda_util.cc:#if HAVE_GPU && !HAVE_ROCM
horovod/torch/cuda_util.cc:typedef CUresult(CUDAAPI* PFN_cuCtxGetDevice)(CUdevice* device);
horovod/torch/cuda_util.cc:static void* cudalib = nullptr;
horovod/torch/cuda_util.cc:  cudalib = dlopen("libcuda.so", RTLD_LAZY);
horovod/torch/cuda_util.cc:  if (!cudalib) {
horovod/torch/cuda_util.cc:    throw std::logic_error("Internal error. Could not dlopen libcuda.so.");
horovod/torch/cuda_util.cc:  pfn_cuCtxGetDevice = (PFN_cuCtxGetDevice)dlsym(cudalib, "cuCtxGetDevice");
horovod/torch/cuda_util.cc:#if HAVE_GPU
horovod/torch/cuda_util.cc:#if !HAVE_ROCM
horovod/torch/cuda_util.cc:    if (!cudalib)
horovod/torch/cuda_util.cc:    if (err == CUDA_ERROR_NOT_INITIALIZED ||
horovod/torch/cuda_util.cc:        err == CUDA_ERROR_INVALID_CONTEXT) {
horovod/torch/cuda_util.cc:    } else if (err == CUDA_SUCCESS) {
horovod/torch/cuda_util.cc:    C10_CUDA_CHECK(cudaSetDevice(device));
horovod/torch/cuda_util.cc:                           "with GPU device but not compiled with CUDA.");
horovod/torch/cuda_util.cc:#if HAVE_GPU
horovod/torch/cuda_util.cc:    C10_CUDA_CHECK(cudaSetDevice(restore_device_));
horovod/torch/mpi_ops_v2.cc:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#include <c10/cuda/CUDAStream.h>
horovod/torch/mpi_ops_v2.cc:#include <c10/cuda/CUDAException.h>
horovod/torch/mpi_ops_v2.cc:#include "cuda_util.h"
horovod/torch/mpi_ops_v2.cc:  if (tensor.device().is_cuda()) {
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:gpuStream_t GetGPUStream(int device) {
horovod/torch/mpi_ops_v2.cc:  return c10::cuda::getCurrentCUDAStream(device);
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:          auto stream = GetGPUStream(device);
horovod/torch/mpi_ops_v2.cc:          HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/torch/mpi_ops_v2.cc:int DoAllreduceCudaOnCPU(::torch::Tensor tensor, ::torch::Tensor output, int divisor,
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:        // Since the operation was on CPU, need to perform copy with the GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:          auto stream = GetGPUStream(device);
horovod/torch/mpi_ops_v2.cc:          HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/torch/mpi_ops_v2.cc:int DoGroupedAllreduceCudaOnCPU(const std::vector<::torch::Tensor>& tensors,
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:                                 auto stream = GetGPUStream(device);
horovod/torch/mpi_ops_v2.cc:                                 HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/torch/mpi_ops_v2.cc:int DoAllgatherCudaOnCPU(::torch::Tensor tensor, ::torch::Tensor output,
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:        // Since the operation was on CPU, need to perform copy with the GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:        auto stream = GetGPUStream(device);
horovod/torch/mpi_ops_v2.cc:        HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/torch/mpi_ops_v2.cc:int DoGroupedAllgatherCudaOnCPU(const std::vector<::torch::Tensor>& tensors,
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:      // Since the operation was on CPU, need to perform copy with the GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:                                 auto stream = GetGPUStream(device);
horovod/torch/mpi_ops_v2.cc:                                 HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/torch/mpi_ops_v2.cc:int DoBroadcastCudaOnCPU(::torch::Tensor tensor, ::torch::Tensor output, int root_rank,
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:        // Since the operation was on CPU, need to perform copy with the GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:  // Deal with possibility of output_received_splits being on GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:          auto stream = GetGPUStream(device);
horovod/torch/mpi_ops_v2.cc:          HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/torch/mpi_ops_v2.cc:int DoAlltoallCudaOnCPU(::torch::Tensor tensor, ::torch::Tensor splits,
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:  // Deal with possibility of output_received_splits being on GPU
horovod/torch/mpi_ops_v2.cc:        { // Since the operation was on CPU, need to perform copy with the GPU
horovod/torch/mpi_ops_v2.cc:  // ROCm only: For ReduceOp::AVERAGE, we do SUM reduction then divide on the
horovod/torch/mpi_ops_v2.cc:  auto request_op = (horovod_rocm_built() && reduce_op == ReduceOp::AVERAGE)
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#endif // HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:          auto stream = GetGPUStream(device);
horovod/torch/mpi_ops_v2.cc:          HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/torch/mpi_ops_v2.cc:        if (horovod_rocm_built() && reduce_op == ReduceOp::AVERAGE) {
horovod/torch/mpi_ops_v2.cc:int DoReducescatterCudaOnCPU(::torch::Tensor tensor, ::torch::Tensor output,
horovod/torch/mpi_ops_v2.cc:  // ROCm only: For ReduceOp::AVERAGE, we do SUM reduction then divide on the
horovod/torch/mpi_ops_v2.cc:  auto request_op = (horovod_rocm_built() && reduce_op == ReduceOp::AVERAGE)
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#endif // HAVE_GPU
horovod/torch/mpi_ops_v2.cc:        // Since the operation was on CPU, need to perform copy with the GPU
horovod/torch/mpi_ops_v2.cc:        if (horovod_rocm_built() && reduce_op == ReduceOp::AVERAGE) {
horovod/torch/mpi_ops_v2.cc:  // ROCm only: For ReduceOp::AVERAGE, we do SUM reduction then divide on the
horovod/torch/mpi_ops_v2.cc:  auto request_op = (horovod_rocm_built() && reduce_op == ReduceOp::AVERAGE)
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#endif // HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:        auto stream = GetGPUStream(device);
horovod/torch/mpi_ops_v2.cc:        HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/torch/mpi_ops_v2.cc:      if (horovod_rocm_built() && reduce_op == ReduceOp::AVERAGE) {
horovod/torch/mpi_ops_v2.cc:int DoGroupedReducescatterCudaOnCPU(const std::vector<::torch::Tensor>& tensors,
horovod/torch/mpi_ops_v2.cc:  // ROCm only: For ReduceOp::AVERAGE, we do SUM reduction then divide on the
horovod/torch/mpi_ops_v2.cc:  auto request_op = (horovod_rocm_built() && reduce_op == ReduceOp::AVERAGE)
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#endif // HAVE_GPU
horovod/torch/mpi_ops_v2.cc:      // Since the operation was on CPU, need to perform copy with the GPU
horovod/torch/mpi_ops_v2.cc:      if (horovod_rocm_built() && reduce_op == ReduceOp::AVERAGE) {
horovod/torch/mpi_ops_v2.cc:#if !HOROVOD_GPU_ALLREDUCE
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:#if HAVE_GPU
horovod/torch/mpi_ops_v2.cc:          auto stream = GetGPUStream(device);
horovod/torch/mpi_ops_v2.cc:          HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/torch/mpi_ops_v2.cc:#if HOROVOD_GPU_ALLREDUCE
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allreduce_async_torch_cuda_IntTensor", &DoAllreduce);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allreduce_async_torch_cuda_LongTensor", &DoAllreduce);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allreduce_async_torch_cuda_HalfTensor", &DoAllreduce);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allreduce_async_torch_cuda_FloatTensor", &DoAllreduce);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allreduce_async_torch_cuda_DoubleTensor", &DoAllreduce);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allreduce_async_torch_cuda_IntTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAllreduceCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allreduce_async_torch_cuda_LongTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAllreduceCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allreduce_async_torch_cuda_HalfTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAllreduceCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allreduce_async_torch_cuda_FloatTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAllreduceCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allreduce_async_torch_cuda_DoubleTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAllreduceCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:#if HOROVOD_GPU_ALLREDUCE
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_IntTensor", &DoGroupedAllreduce);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_LongTensor", &DoGroupedAllreduce);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_HalfTensor", &DoGroupedAllreduce);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_FloatTensor", &DoGroupedAllreduce);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_DoubleTensor", &DoGroupedAllreduce);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_IntTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedAllreduceCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_LongTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedAllreduceCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_HalfTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedAllreduceCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_FloatTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedAllreduceCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allreduce_async_torch_cuda_DoubleTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedAllreduceCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:#if HOROVOD_GPU_ALLGATHER
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_ByteTensor", &DoAllgather);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_CharTensor", &DoAllgather);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_ShortTensor", &DoAllgather);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_IntTensor", &DoAllgather);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_LongTensor", &DoAllgather);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_HalfTensor", &DoAllgather);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_FloatTensor", &DoAllgather);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_DoubleTensor", &DoAllgather);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_ByteTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_CharTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_ShortTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_IntTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_LongTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_HalfTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_FloatTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_allgather_async_torch_cuda_DoubleTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:#if HOROVOD_GPU_ALLGATHER
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_ByteTensor",
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_CharTensor",
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_ShortTensor",
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_IntTensor",
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_LongTensor",
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_HalfTensor",
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_FloatTensor",
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_DoubleTensor",
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_ByteTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_CharTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_ShortTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_IntTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_LongTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_HalfTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_FloatTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_allgather_async_torch_cuda_DoubleTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedAllgatherCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:#if HOROVOD_GPU_BROADCAST
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_ByteTensor", &DoBroadcast);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_CharTensor", &DoBroadcast);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_ShortTensor", &DoBroadcast);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_IntTensor", &DoBroadcast);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_LongTensor", &DoBroadcast);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_HalfTensor", &DoBroadcast);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_FloatTensor", &DoBroadcast);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_DoubleTensor", &DoBroadcast);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_ByteTensor",
horovod/torch/mpi_ops_v2.cc:        &DoBroadcastCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_CharTensor",
horovod/torch/mpi_ops_v2.cc:        &DoBroadcastCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_ShortTensor",
horovod/torch/mpi_ops_v2.cc:        &DoBroadcastCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_IntTensor",
horovod/torch/mpi_ops_v2.cc:        &DoBroadcastCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_LongTensor",
horovod/torch/mpi_ops_v2.cc:        &DoBroadcastCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_HalfTensor",
horovod/torch/mpi_ops_v2.cc:        &DoBroadcastCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_FloatTensor",
horovod/torch/mpi_ops_v2.cc:        &DoBroadcastCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_broadcast_async_torch_cuda_DoubleTensor",
horovod/torch/mpi_ops_v2.cc:        &DoBroadcastCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:#if HOROVOD_GPU_ALLTOALL
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_ByteTensor", &DoAlltoall);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_CharTensor", &DoAlltoall);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_ShortTensor", &DoAlltoall);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_IntTensor", &DoAlltoall);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_LongTensor", &DoAlltoall);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_HalfTensor", &DoAlltoall);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_FloatTensor", &DoAlltoall);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_DoubleTensor", &DoAlltoall);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_ByteTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAlltoallCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_CharTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAlltoallCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_ShortTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAlltoallCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_IntTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAlltoallCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_LongTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAlltoallCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_HalfTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAlltoallCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_FloatTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAlltoallCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_alltoall_async_torch_cuda_DoubleTensor",
horovod/torch/mpi_ops_v2.cc:        &DoAlltoallCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:#if HOROVOD_GPU_REDUCESCATTER
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_reducescatter_async_torch_cuda_IntTensor", &DoReducescatter);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_reducescatter_async_torch_cuda_LongTensor", &DoReducescatter);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_reducescatter_async_torch_cuda_HalfTensor", &DoReducescatter);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_reducescatter_async_torch_cuda_FloatTensor", &DoReducescatter);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_reducescatter_async_torch_cuda_DoubleTensor", &DoReducescatter);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_reducescatter_async_torch_cuda_IntTensor",
horovod/torch/mpi_ops_v2.cc:        &DoReducescatterCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_reducescatter_async_torch_cuda_LongTensor",
horovod/torch/mpi_ops_v2.cc:        &DoReducescatterCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_reducescatter_async_torch_cuda_HalfTensor",
horovod/torch/mpi_ops_v2.cc:        &DoReducescatterCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_reducescatter_async_torch_cuda_FloatTensor",
horovod/torch/mpi_ops_v2.cc:        &DoReducescatterCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_reducescatter_async_torch_cuda_DoubleTensor",
horovod/torch/mpi_ops_v2.cc:        &DoReducescatterCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:#if HOROVOD_GPU_REDUCESCATTER
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_reducescatter_async_torch_cuda_IntTensor",
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_reducescatter_async_torch_cuda_LongTensor",
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_reducescatter_async_torch_cuda_HalfTensor",
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_reducescatter_async_torch_cuda_FloatTensor",
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_reducescatter_async_torch_cuda_DoubleTensor",
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_reducescatter_async_torch_cuda_IntTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedReducescatterCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_reducescatter_async_torch_cuda_LongTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedReducescatterCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_reducescatter_async_torch_cuda_HalfTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedReducescatterCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_reducescatter_async_torch_cuda_FloatTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedReducescatterCudaOnCPU);
horovod/torch/mpi_ops_v2.cc:  m.def("horovod_torch_grouped_reducescatter_async_torch_cuda_DoubleTensor",
horovod/torch/mpi_ops_v2.cc:        &DoGroupedReducescatterCudaOnCPU);
horovod/torch/ready_event.cc:#if HAVE_GPU
horovod/torch/ready_event.cc:#include <c10/cuda/CUDAStream.h>
horovod/torch/ready_event.cc:#include <c10/cuda/CUDAException.h>
horovod/torch/ready_event.cc:#include "cuda_util.h"
horovod/torch/ready_event.cc:#if HAVE_GPU
horovod/torch/ready_event.cc:  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
horovod/torch/ready_event.cc:    auto& queue = ready_event_registry.cuda_events[device_];
horovod/torch/ready_event.cc:      cuda_event_ = queue.front();
horovod/torch/ready_event.cc:      C10_CUDA_CHECK(cudaEventCreateWithFlags(
horovod/torch/ready_event.cc:          &cuda_event_, cudaEventBlockingSync | cudaEventDisableTiming));
horovod/torch/ready_event.cc:  auto stream = c10::cuda::getCurrentCUDAStream(device_);
horovod/torch/ready_event.cc:  C10_CUDA_CHECK(cudaEventRecord(cuda_event_, stream));
horovod/torch/ready_event.cc:    auto& queue = ready_event_registry.cuda_events[device_];
horovod/torch/ready_event.cc:    queue.push(cuda_event_);
horovod/torch/ready_event.cc:  C10_CUDA_CHECK(cudaEventSynchronize(cuda_event_));
horovod/torch/ready_event.cc:gpuEvent_t TorchReadyEvent::event() const {
horovod/torch/ready_event.cc:  return cuda_event_;
horovod/torch/ready_event.cc:// On GPU this event will signal that GPU computations are done and data is
horovod/torch/ready_event.cc:#if HAVE_GPU
horovod/torch/ready_event.cc:                           "with GPU device but not compiled with CUDA.");
horovod/keras/callbacks.py:            device: Device to be used for broadcasting. Uses GPU by default
horovod/keras/callbacks.py:                    if Horovod was build with HOROVOD_GPU_OPERATIONS.
horovod/keras/callbacks.py:            device: Device to be used for allreduce. Uses GPU by default
horovod/keras/callbacks.py:                    if Horovod was build with HOROVOD_GPU_OPERATIONS.
horovod/keras/__init__.py:from horovod.tensorflow import nccl_built, ddl_built, ccl_built, cuda_built, rocm_built
horovod/keras/__init__.py:        device_dense: Device to be used for dense tensors. Uses GPU by default
horovod/keras/__init__.py:                      if Horovod was build with HOROVOD_GPU_OPERATIONS.
horovod/keras/__init__.py:        device_sparse: Device to be used for sparse tensors. Uses GPU by default
horovod/keras/__init__.py:                       if Horovod was build with HOROVOD_GPU_OPERATIONS.
horovod/keras/__init__.py:    if gradient_predivide_factor != 1.0 and rocm_built():
horovod/keras/__init__.py:            raise ValueError('gradient_predivide_factor not supported yet with ROCm')
horovod/keras/__init__.py:    if gradient_predivide_factor != 1.0 and rocm_built():
horovod/keras/__init__.py:        raise ValueError('gradient_predivide_factor not supported yet with ROCm')
horovod/spark/torch/datamodule.py:# Copyright (C) 2022, NVIDIA CORPORATION. All rights reserved.
horovod/spark/torch/datamodule.py:    """NVTabular-based DataModule for TorchEstimator for GPU-accelerated data loading of tabular datasets.
horovod/spark/torch/remote.py:# Modifications copyright (C) 2022, NVIDIA CORPORATION. All rights reserved.
horovod/spark/torch/remote.py:from horovod.spark.common.util import _get_assigned_gpu_or_local_rank, to_list, _set_mp_start_method
horovod/spark/torch/remote.py:    should_use_gpu = estimator.getUseGpu()
horovod/spark/torch/remote.py:        if not should_use_gpu and user_verbose:
horovod/spark/torch/remote.py:            print("Skip pinning current process to the GPU.")
horovod/spark/torch/remote.py:        cuda_available = torch.cuda.is_available()
horovod/spark/torch/remote.py:        if cuda_available and not should_use_gpu:
horovod/spark/torch/remote.py:            print("GPU is available but use_gpu is set to False."
horovod/spark/torch/remote.py:                  "Training will proceed without GPU support.")
horovod/spark/torch/remote.py:            cuda_available = False
horovod/spark/torch/remote.py:        cuda_avail_list = hvd.allgather_object(cuda_available, name='device type')
horovod/spark/torch/remote.py:        if cuda_avail_list.count(cuda_available) != hvd.size():
horovod/spark/torch/remote.py:        if cuda_available:
horovod/spark/torch/remote.py:            # Horovod: pin GPU to local rank or the assigned GPU from spark.
horovod/spark/torch/remote.py:            torch.cuda.set_device(_get_assigned_gpu_or_local_rank(local_rank=hvd.local_rank()))
horovod/spark/torch/remote.py:            # Move model to GPU.
horovod/spark/torch/remote.py:            model.cuda()
horovod/spark/torch/remote.py:                if cuda_available:
horovod/spark/torch/remote.py:                    model.cuda()
horovod/spark/torch/remote.py:                if cuda_available:
horovod/spark/torch/remote.py:                    inputs = [input.cuda() for input in inputs]
horovod/spark/torch/remote.py:                    labels = [label.cuda() for label in labels]
horovod/spark/torch/remote.py:                        sample_weights = sample_weights.cuda()
horovod/spark/torch/estimator.py:# Modifications copyright (C) 2022, NVIDIA CORPORATION. All rights reserved.
horovod/spark/torch/estimator.py:        use_gpu: Whether to use the GPU for training. Defaults to True.
horovod/spark/torch/estimator.py:                 use_gpu=True,
horovod/spark/lightning/remote.py:from horovod.spark.common.util import _get_assigned_gpu_or_local_rank, _set_mp_start_method
horovod/spark/lightning/remote.py:    num_gpus = estimator.getNumGPUs()
horovod/spark/lightning/remote.py:    should_use_gpu = estimator.getUseGpu()
horovod/spark/lightning/remote.py:    # resume the logger experiment from GPU instance.
horovod/spark/lightning/remote.py:            if not should_use_gpu and verbose:
horovod/spark/lightning/remote.py:                print("Skip pinning current process to the GPU.")
horovod/spark/lightning/remote.py:            cuda_available = torch.cuda.is_available()
horovod/spark/lightning/remote.py:            if cuda_available and not should_use_gpu:
horovod/spark/lightning/remote.py:                print("GPU is available but use_gpu is set to False."
horovod/spark/lightning/remote.py:                      "Training will proceed without GPU support.")
horovod/spark/lightning/remote.py:                cuda_available = False
horovod/spark/lightning/remote.py:            cuda_avail_list = hvd.allgather_object(cuda_available, name='device type')
horovod/spark/lightning/remote.py:            if cuda_avail_list.count(cuda_available) != hvd.size():
horovod/spark/lightning/remote.py:            if cuda_available:
horovod/spark/lightning/remote.py:                # Horovod: pin GPU to local rank or the assigned GPU from spark.
horovod/spark/lightning/remote.py:                torch.cuda.set_device(_get_assigned_gpu_or_local_rank(local_rank=hvd.local_rank()))
horovod/spark/lightning/remote.py:                # Move model to GPU.
horovod/spark/lightning/remote.py:                model.cuda()
horovod/spark/lightning/remote.py:            _num_gpus = num_gpus
horovod/spark/lightning/remote.py:            if _num_gpus is None:
horovod/spark/lightning/remote.py:                _num_gpus = 1 if cuda_available else 0
horovod/spark/lightning/remote.py:                      'gpus': _num_gpus,
horovod/spark/lightning/estimator.py:        num_gpus;   (Optional) Number of gpus per process, default to 1 when CUDA is available
horovod/spark/lightning/estimator.py:        use_gpu: Whether to use the GPU for training. Defaults to True.
horovod/spark/lightning/estimator.py:    num_gpus = Param(Params._dummy(), 'num_gpus',
horovod/spark/lightning/estimator.py:                     'Number of gpus per process, default to 1 when CUDA is available in the backend, otherwise 0.')
horovod/spark/lightning/estimator.py:                 num_gpus=None,
horovod/spark/lightning/estimator.py:                 use_gpu=True,
horovod/spark/lightning/estimator.py:                         num_gpus=None,
horovod/spark/lightning/estimator.py:    def setNumGPUs(self, value):
horovod/spark/lightning/estimator.py:        return self._set(num_gpus=value)
horovod/spark/lightning/estimator.py:    def getNumGPUs(self):
horovod/spark/lightning/estimator.py:        return self.getOrDefault(self.num_gpus)
horovod/spark/task/task_info.py:    if 'gpu' not in _info.resources:
horovod/spark/task/task_info.py:    return _info.resources['gpu'].addresses
horovod/spark/keras/datamodule.py:# Copyright (C) 2022, NVIDIA CORPORATION. All rights reserved.
horovod/spark/keras/datamodule.py:    """NVTabular-based DataModule for KerasEstimator for GPU-accelerated data loading of tabular datasets.
horovod/spark/keras/remote.py:# Modifications copyright (C) 2022, NVIDIA CORPORATION. All rights reserved.
horovod/spark/keras/remote.py:from horovod.spark.common.util import _get_assigned_gpu_or_local_rank, _set_mp_start_method
horovod/spark/keras/remote.py:    should_use_gpu = estimator.getUseGpu()
horovod/spark/keras/remote.py:    pin_gpu = _pin_gpu_fn()
horovod/spark/keras/remote.py:        if should_use_gpu:
horovod/spark/keras/remote.py:                print("Pinning current process to the GPU.")
horovod/spark/keras/remote.py:            pin_gpu(hvd, tf, k)
horovod/spark/keras/remote.py:                print("Skip pinning current process to the GPU.")
horovod/spark/keras/remote.py:def _pin_gpu_fn():
horovod/spark/keras/remote.py:    # Horovod: pin GPU to be used to process local rank (one GPU per process)
horovod/spark/keras/remote.py:    return _pin_gpu_tensorflow2_fn() if version.parse(tf.__version__) >= version.parse('2.0.0') \
horovod/spark/keras/remote.py:        else _pin_gpu_tensorflow1_fn()
horovod/spark/keras/remote.py:def _pin_gpu_tensorflow2_fn():
horovod/spark/keras/remote.py:        gpus = tf.config.experimental.list_physical_devices('GPU')
horovod/spark/keras/remote.py:        for gpu in gpus:
horovod/spark/keras/remote.py:            tf.config.experimental.set_memory_growth(gpu, True)
horovod/spark/keras/remote.py:        if gpus:
horovod/spark/keras/remote.py:                gpus[_get_assigned_gpu_or_local_rank(local_rank=hvd.local_rank())], 'GPU')
horovod/spark/keras/remote.py:def _pin_gpu_tensorflow1_fn():
horovod/spark/keras/remote.py:        config.gpu_options.allow_growth = True
horovod/spark/keras/remote.py:        config.gpu_options.visible_device_list = \
horovod/spark/keras/remote.py:            str(_get_assigned_gpu_or_local_rank(local_rank=hvd.local_rank()))
horovod/spark/keras/remote.py:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
horovod/spark/keras/remote.py:        config = tf.ConfigProto(device_count={'GPU': 0})
horovod/spark/keras/util.py:# Modifications copyright (C) 2022, NVIDIA CORPORATION. All rights reserved.
horovod/spark/keras/__init__.py:# Modifications copyright (C) 2022, NVIDIA CORPORATION. All rights reserved.
horovod/spark/keras/estimator.py:# Modifications copyright (C) 2022, NVIDIA CORPORATION. All rights reserved.
horovod/spark/keras/estimator.py:        use_gpu: Whether to use the GPU for training. Defaults to True.
horovod/spark/keras/estimator.py:                 use_gpu=True,
horovod/spark/keras/estimator.py:            # Do not use GPUs for prediction, use single CPU core per task.
horovod/spark/common/datamodule.py:# Copyright (C) 2022, NVIDIA CORPORATION. All rights reserved.
horovod/spark/common/backend.py:        if 'CUDA_VISIBLE_DEVICES' in full_env:
horovod/spark/common/backend.py:            # from allocating memory on the GPU outside the training process.  Once we submit the
horovod/spark/common/backend.py:            # function for execution, we want to ensure that TensorFLow has visibility into GPUs on
horovod/spark/common/backend.py:            del full_env['CUDA_VISIBLE_DEVICES']
horovod/spark/common/util.py:def _get_assigned_gpu_or_local_rank(local_rank):
horovod/spark/common/util.py:    if available_devices and os.getenv('HOROVOD_SPARK_USE_LOCAL_RANK_GPU_INDEX', '0') == '0':
horovod/spark/common/util.py:        # if GPU-aware scheduling is available, pin to the assigned GPU index
horovod/spark/common/util.py:        # this is always the first GPU index in available_devices as only one GPU is expected to be assigned
horovod/spark/common/util.py:        # pin to local rank GPU index
horovod/spark/common/params.py:# Modifications copyright (C) 2022, NVIDIA CORPORATION. All rights reserved.
horovod/spark/common/params.py:    use_gpu = Param(Params._dummy(), 'use_gpu',
horovod/spark/common/params.py:                    'Whether to use the GPU for training. '
horovod/spark/common/params.py:                    'Setting this to False will skip binding to GPU even when GPU is available. '
horovod/spark/common/params.py:            use_gpu=True,
horovod/spark/common/params.py:    def setUseGpu(self, value):
horovod/spark/common/params.py:        self._set(use_gpu=value)
horovod/spark/common/params.py:    def getUseGpu(self):
horovod/spark/common/params.py:        return self.getOrDefault(self.use_gpu)
horovod/spark/mpi_run.py:    settings.extra_mpi_args = ('{extra_mpi_args} -x NCCL_DEBUG=INFO -mca plm_rsh_agent "{rsh_agent}"'
horovod/common/response_cache.h:// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
horovod/common/group_table.cc:// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/basics.py:    def nccl_built(self):
horovod/common/basics.py:        """Function to check if Horovod was compiled with NCCL support.
horovod/common/basics.py:          An integer value indicating whether NCCL support was compiled.
horovod/common/basics.py:          If NCCL support was compiled, returns NCCL_VERSION_CODE. Otherwise,
horovod/common/basics.py:        return int(self.MPI_LIB_CTYPES.horovod_nccl_built())
horovod/common/basics.py:    def cuda_built(self):
horovod/common/basics.py:        """Returns True if Horovod was compiled with CUDA support.
horovod/common/basics.py:          A boolean value indicating whether CUDA support was compiled.
horovod/common/basics.py:        return bool(self.MPI_LIB_CTYPES.horovod_cuda_built())
horovod/common/basics.py:    def rocm_built(self):
horovod/common/basics.py:        """Returns True if Horovod was compiled with ROCm support.
horovod/common/basics.py:          A boolean value indicating whether ROCm support was compiled.
horovod/common/basics.py:        return bool(self.MPI_LIB_CTYPES.horovod_rocm_built())
horovod/common/response_cache.cc:// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
horovod/common/elastic.py:        Because commits are a heavy operation involving data copy (potentially from GPU to host), it is
horovod/common/message.cc:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/nccl_operations.cc:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/nccl_operations.cc:#include "nccl_operations.h"
horovod/common/ops/nccl_operations.cc:#if HAVE_CUDA
horovod/common/ops/nccl_operations.cc:#include "cuda/cuda_kernels.h"
horovod/common/ops/nccl_operations.cc:#if HAVE_ROCM
horovod/common/ops/nccl_operations.cc:#include "rocm/hip_kernels.h"
horovod/common/ops/nccl_operations.cc:ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor) {
horovod/common/ops/nccl_operations.cc:    return ncclUint8;
horovod/common/ops/nccl_operations.cc:    return ncclInt8;
horovod/common/ops/nccl_operations.cc:    return ncclInt32;
horovod/common/ops/nccl_operations.cc:    return ncclInt64;
horovod/common/ops/nccl_operations.cc:    return ncclFloat16;
horovod/common/ops/nccl_operations.cc:    return ncclFloat32;
horovod/common/ops/nccl_operations.cc:    return ncclFloat64;
horovod/common/ops/nccl_operations.cc:                           " is not supported in NCCL mode.");
horovod/common/ops/nccl_operations.cc:void commDestroyOrAbort(ncclComm_t& nccl_comm, bool elastic) {
horovod/common/ops/nccl_operations.cc:  ncclResult_t nccl_async_err;
horovod/common/ops/nccl_operations.cc:  auto nccl_err = ncclCommGetAsyncError(nccl_comm, &nccl_async_err);
horovod/common/ops/nccl_operations.cc:  if (nccl_err != ncclSuccess) {
horovod/common/ops/nccl_operations.cc:  if (nccl_async_err == ncclSuccess && !elastic) {
horovod/common/ops/nccl_operations.cc:    ncclCommDestroy(nccl_comm);
horovod/common/ops/nccl_operations.cc:    ncclCommAbort(nccl_comm);
horovod/common/ops/nccl_operations.cc:void NCCLContext::ErrorCheck(std::string op_name, ncclResult_t nccl_result,
horovod/common/ops/nccl_operations.cc:                             ncclComm_t& nccl_comm) {
horovod/common/ops/nccl_operations.cc:  if (nccl_result != ncclSuccess) {
horovod/common/ops/nccl_operations.cc:    ncclCommAbort(nccl_comm);
horovod/common/ops/nccl_operations.cc:                           " failed: " + ncclGetErrorString(nccl_result));
horovod/common/ops/nccl_operations.cc:void NCCLContext::ShutDown() {
horovod/common/ops/nccl_operations.cc:  for (auto it = nccl_comms.begin(); it != nccl_comms.end(); ++it) {
horovod/common/ops/nccl_operations.cc:  nccl_comms.clear();
horovod/common/ops/nccl_operations.cc:void NCCLOpContext::InitNCCLComm(const std::vector<TensorTableEntry>& entries,
horovod/common/ops/nccl_operations.cc:                                 const std::vector<int32_t>& nccl_device_map) {
horovod/common/ops/nccl_operations.cc:  // Ensure NCCL communicator is in the map before executing operation.
horovod/common/ops/nccl_operations.cc:  // in situations where one process has already built nccl_comm, but
horovod/common/ops/nccl_operations.cc:  ncclComm_t& nccl_comm =
horovod/common/ops/nccl_operations.cc:      nccl_context_
horovod/common/ops/nccl_operations.cc:          ->nccl_comms[global_state_->current_nccl_stream]
horovod/common/ops/nccl_operations.cc:                      [std::make_tuple(process_set_id, nccl_device_map)];
horovod/common/ops/nccl_operations.cc:  if (nccl_comm == nullptr) {
horovod/common/ops/nccl_operations.cc:    timeline.ActivityStartAll(entries, INIT_NCCL);
horovod/common/ops/nccl_operations.cc:    int nccl_rank, nccl_size;
horovod/common/ops/nccl_operations.cc:    Communicator nccl_id_bcast_comm;
horovod/common/ops/nccl_operations.cc:    PopulateNCCLCommStrategy(nccl_rank, nccl_size, nccl_id_bcast_comm,
horovod/common/ops/nccl_operations.cc:    ncclUniqueId nccl_id;
horovod/common/ops/nccl_operations.cc:    if (nccl_rank == 0) {
horovod/common/ops/nccl_operations.cc:      nccl_context_->ErrorCheck("ncclGetUniqueId", ncclGetUniqueId(&nccl_id),
horovod/common/ops/nccl_operations.cc:                                nccl_comm);
horovod/common/ops/nccl_operations.cc:    process_set.controller->Bcast((void*)&nccl_id, sizeof(nccl_id), 0,
horovod/common/ops/nccl_operations.cc:                                  nccl_id_bcast_comm);
horovod/common/ops/nccl_operations.cc:    ncclComm_t new_nccl_comm;
horovod/common/ops/nccl_operations.cc:    auto nccl_result =
horovod/common/ops/nccl_operations.cc:        ncclCommInitRank(&new_nccl_comm, nccl_size, nccl_id, nccl_rank);
horovod/common/ops/nccl_operations.cc:    nccl_context_->ErrorCheck("ncclCommInitRank", nccl_result, nccl_comm);
horovod/common/ops/nccl_operations.cc:    nccl_comm = new_nccl_comm;
horovod/common/ops/nccl_operations.cc:    // Barrier helps NCCL to synchronize after initialization and avoid
horovod/common/ops/nccl_operations.cc:  nccl_comm_ = &nccl_comm;
horovod/common/ops/nccl_operations.cc:void NCCLOpContext::AsyncErrorCheck() {
horovod/common/ops/nccl_operations.cc:  ncclResult_t nccl_async_err;
horovod/common/ops/nccl_operations.cc:  auto nccl_err = ncclCommGetAsyncError(*nccl_comm_, &nccl_async_err);
horovod/common/ops/nccl_operations.cc:  if (nccl_err != ncclSuccess) {
horovod/common/ops/nccl_operations.cc:    throw std::logic_error(std::string("ncclGetAsyncError failed: ") +
horovod/common/ops/nccl_operations.cc:                           ncclGetErrorString(nccl_err));
horovod/common/ops/nccl_operations.cc:  if (nccl_async_err != ncclSuccess) {
horovod/common/ops/nccl_operations.cc:    // do not call ncclCommAbort(*nccl_comm_) from event polling thread to avoid
horovod/common/ops/nccl_operations.cc:    throw std::logic_error(std::string("NCCL async error: ") +
horovod/common/ops/nccl_operations.cc:                           ncclGetErrorString(nccl_async_err));
horovod/common/ops/nccl_operations.cc:void NCCLOpContext::PopulateNCCLCommStrategy(int& nccl_rank, int& nccl_size,
horovod/common/ops/nccl_operations.cc:                                             Communicator& nccl_id_bcast_comm,
horovod/common/ops/nccl_operations.cc:    nccl_rank = process_set.controller->GetRank();
horovod/common/ops/nccl_operations.cc:    nccl_size = process_set.controller->GetSize();
horovod/common/ops/nccl_operations.cc:    nccl_rank = process_set.controller->GetLocalRank();
horovod/common/ops/nccl_operations.cc:    nccl_size = process_set.controller->GetLocalSize();
horovod/common/ops/nccl_operations.cc:    nccl_rank = process_set.controller->GetCrossRank();
horovod/common/ops/nccl_operations.cc:    nccl_size = process_set.controller->GetCrossSize();
horovod/common/ops/nccl_operations.cc:                           " is not supported in NCCL mode.");
horovod/common/ops/nccl_operations.cc:  nccl_id_bcast_comm = communicator_type_;
horovod/common/ops/nccl_operations.cc:void NCCLAllreduce::WaitForData(std::vector<TensorTableEntry>& entries) {
horovod/common/ops/nccl_operations.cc:    std::unordered_set<gpuEvent_t> event_set;
horovod/common/ops/nccl_operations.cc:      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
horovod/common/ops/nccl_operations.cc:Status NCCLAllreduce::Execute(std::vector<TensorTableEntry>& entries,
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPU(entries);
horovod/common/ops/nccl_operations.cc:  nccl_op_context_.InitNCCLComm(entries, response.devices());
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPUQueue(entries, response);
horovod/common/ops/nccl_operations.cc:  ncclRedOp_t ncclOp = ncclSum;
horovod/common/ops/nccl_operations.cc:#if !HAVE_ROCM
horovod/common/ops/nccl_operations.cc:#ifdef NCCL_AVG_SUPPORTED
horovod/common/ops/nccl_operations.cc:    // Use NCCLAvg op in place of postscale_factor
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclAvg;
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclSum;
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclSum;
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclMin;
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclMax;
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclProd;
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:  auto nccl_result =
horovod/common/ops/nccl_operations.cc:      ncclAllReduce(fused_input_data, buffer_data, (size_t)num_elements,
horovod/common/ops/nccl_operations.cc:                    GetNCCLDataType(first_entry.tensor), ncclOp,
horovod/common/ops/nccl_operations.cc:                    *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:  nccl_context_->ErrorCheck("ncclAllReduce", nccl_result,
horovod/common/ops/nccl_operations.cc:                            *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:    gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_ALLREDUCE,
horovod/common/ops/nccl_operations.cc:                              *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:  return gpu_op_context_.FinalizeGPUQueue(
horovod/common/ops/nccl_operations.cc:      entries, true, nccl_op_context_.error_check_callback_);
horovod/common/ops/nccl_operations.cc:void NCCLHierarchicalAllreduce::WaitForData(
horovod/common/ops/nccl_operations.cc:    std::unordered_set<gpuEvent_t> event_set;
horovod/common/ops/nccl_operations.cc:      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
horovod/common/ops/nccl_operations.cc:NCCLHierarchicalAllreduce::Execute(std::vector<TensorTableEntry>& entries,
horovod/common/ops/nccl_operations.cc:  // Determine GPU IDs of the devices participating in this communicator.
horovod/common/ops/nccl_operations.cc:  std::vector<int32_t> nccl_device_map;
horovod/common/ops/nccl_operations.cc:  nccl_device_map.reserve(process_set.controller->GetLocalCommRanks().size());
horovod/common/ops/nccl_operations.cc:    nccl_device_map.push_back(response.devices()[rank]);
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPU(entries);
horovod/common/ops/nccl_operations.cc:  nccl_op_context_.InitNCCLComm(entries, nccl_device_map);
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPUQueue(entries, response);
horovod/common/ops/nccl_operations.cc:  ncclRedOp_t ncclOp = ncclSum;
horovod/common/ops/nccl_operations.cc:#if !HAVE_ROCM
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclSum;
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclSum;
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclMin;
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclMax;
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclProd;
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:  // For the part of data divisible by local_size, perform NCCL
horovod/common/ops/nccl_operations.cc:  // ReduceScatter - Parallelized MPI Allreduce - NCCL Allgather. For the
horovod/common/ops/nccl_operations.cc:  // non-divisible part (if any), do NCCL Reduce (at rank local_size-1),
horovod/common/ops/nccl_operations.cc:  // MPI Allreduce (across rank (local_size-1)'s), and NCCL Bcast
horovod/common/ops/nccl_operations.cc:    auto nccl_result = ncclReduceScatter(
horovod/common/ops/nccl_operations.cc:        (size_t)num_elements_per_rank, GetNCCLDataType(first_entry.tensor),
horovod/common/ops/nccl_operations.cc:        ncclOp, *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    nccl_context_->ErrorCheck("ncclReduceScatter", nccl_result,
horovod/common/ops/nccl_operations.cc:                              *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCESCATTER,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    auto nccl_result =
horovod/common/ops/nccl_operations.cc:        ncclReduce(fused_input_data_remainder, buffer_data_remainder,
horovod/common/ops/nccl_operations.cc:                   GetNCCLDataType(first_entry.tensor), ncclOp, root_rank,
horovod/common/ops/nccl_operations.cc:                   *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    nccl_context_->ErrorCheck("ncclReduce", nccl_result,
horovod/common/ops/nccl_operations.cc:                              *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCE,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    // cudaHostAlloc is significantly slower than malloc.  Pre-allocating
horovod/common/ops/nccl_operations.cc:    gpu_op_context_.host_buffer = malloc(total_buffer_len);
horovod/common/ops/nccl_operations.cc:      gpu_context_->WaitForEventsElastic(
horovod/common/ops/nccl_operations.cc:          gpu_op_context_.event_queue, entries, timeline,
horovod/common/ops/nccl_operations.cc:          nccl_op_context_.error_check_callback_);
horovod/common/ops/nccl_operations.cc:      gpu_context_->WaitForEvents(gpu_op_context_.event_queue, entries,
horovod/common/ops/nccl_operations.cc:                                  nccl_op_context_.error_check_callback_);
horovod/common/ops/nccl_operations.cc:    // According to https://docs.nvidia.com/cuda/cuda-runtime-api/
horovod/common/ops/nccl_operations.cc:    // cudaMemcpyAsync is synchronous with respect to the host, so we
horovod/common/ops/nccl_operations.cc:    gpu_context_->MemcpyAsyncD2H(gpu_op_context_.host_buffer,
horovod/common/ops/nccl_operations.cc:                                 *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    int op = MPI_Allreduce(MPI_IN_PLACE, gpu_op_context_.host_buffer,
horovod/common/ops/nccl_operations.cc:    gpu_context_->MemcpyAsyncH2D(buffer_data_at_rank_offset,
horovod/common/ops/nccl_operations.cc:                                 gpu_op_context_.host_buffer, total_buffer_len,
horovod/common/ops/nccl_operations.cc:                                 *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    nccl_context_->ErrorCheck(
horovod/common/ops/nccl_operations.cc:        "ncclAllGather",
horovod/common/ops/nccl_operations.cc:        ncclAllGather(buffer_data_at_rank_offset, buffer_data,
horovod/common/ops/nccl_operations.cc:                      GetNCCLDataType(first_entry.tensor),
horovod/common/ops/nccl_operations.cc:                      *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
horovod/common/ops/nccl_operations.cc:        *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_ALLGATHER,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 2, 12)
horovod/common/ops/nccl_operations.cc:    nccl_context_->ErrorCheck(
horovod/common/ops/nccl_operations.cc:        "ncclBroadcast",
horovod/common/ops/nccl_operations.cc:        ncclBroadcast(buffer_data_remainder, buffer_data_remainder,
horovod/common/ops/nccl_operations.cc:                      GetNCCLDataType(first_entry.tensor), root_rank,
horovod/common/ops/nccl_operations.cc:                      *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
horovod/common/ops/nccl_operations.cc:        *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:    nccl_context_->ErrorCheck(
horovod/common/ops/nccl_operations.cc:        "ncclBcast",
horovod/common/ops/nccl_operations.cc:        ncclBcast(buffer_data_remainder, (size_t)num_elements_remaining,
horovod/common/ops/nccl_operations.cc:                  GetNCCLDataType(first_entry.tensor), root_rank,
horovod/common/ops/nccl_operations.cc:                  *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
horovod/common/ops/nccl_operations.cc:        *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_BCAST,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:  return gpu_op_context_.FinalizeGPUQueue(
horovod/common/ops/nccl_operations.cc:      entries, true, nccl_op_context_.error_check_callback_);
horovod/common/ops/nccl_operations.cc:bool NCCLHierarchicalAllreduce::Enabled(
horovod/common/ops/nccl_operations.cc:  if (!NCCLAllreduce::Enabled(param_manager, entries, response)) {
horovod/common/ops/nccl_operations.cc:void NCCLTorusAllreduce::WaitForData(std::vector<TensorTableEntry>& entries) {
horovod/common/ops/nccl_operations.cc:    std::unordered_set<gpuEvent_t> event_set;
horovod/common/ops/nccl_operations.cc:      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
horovod/common/ops/nccl_operations.cc:Status NCCLTorusAllreduce::Execute(std::vector<TensorTableEntry>& entries,
horovod/common/ops/nccl_operations.cc:  // Determine GPU IDs of the devices participating in this communicator.
horovod/common/ops/nccl_operations.cc:  std::vector<int32_t> local_nccl_device_map;
horovod/common/ops/nccl_operations.cc:  local_nccl_device_map.reserve(process_set.controller->GetLocalCommRanks().size());
horovod/common/ops/nccl_operations.cc:    local_nccl_device_map.push_back(device);
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPU(entries);
horovod/common/ops/nccl_operations.cc:  local_nccl_op_context_.InitNCCLComm(entries, local_nccl_device_map);
horovod/common/ops/nccl_operations.cc:  std::vector<int32_t> cross_nccl_device_map({response.devices()[process_set.controller->GetCrossRank()]});
horovod/common/ops/nccl_operations.cc:  cross_nccl_op_context_.InitNCCLComm(entries, cross_nccl_device_map);
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPUQueue(entries, response);
horovod/common/ops/nccl_operations.cc:  ncclRedOp_t ncclOp = ncclSum;
horovod/common/ops/nccl_operations.cc:#if !HAVE_ROCM
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclSum;
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclSum;
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclMin;
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclMax;
horovod/common/ops/nccl_operations.cc:    ncclOp = ncclProd;
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:  // For the part of data divisible by local_size, perform NCCL
horovod/common/ops/nccl_operations.cc:  // ReduceScatter - Parallelized NCCL Allreduce - NCCL Allgather. For the
horovod/common/ops/nccl_operations.cc:  // non-divisible part (if any), do NCCL Reduce (at rank local_size-1),
horovod/common/ops/nccl_operations.cc:  // NCCL Allreduce (across rank (local_size-1)'s), and NCCL Bcast.
horovod/common/ops/nccl_operations.cc:    auto nccl_result = ncclReduceScatter(
horovod/common/ops/nccl_operations.cc:        (size_t)num_elements_per_rank, GetNCCLDataType(first_entry.tensor),
horovod/common/ops/nccl_operations.cc:        ncclOp, *local_nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    local_nccl_context_->ErrorCheck("ncclReduceScatter", nccl_result,
horovod/common/ops/nccl_operations.cc:                              *local_nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCESCATTER,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    auto nccl_result =
horovod/common/ops/nccl_operations.cc:        ncclReduce(fused_input_data_remainder, buffer_data_remainder,
horovod/common/ops/nccl_operations.cc:                   GetNCCLDataType(first_entry.tensor), ncclOp, root_rank,
horovod/common/ops/nccl_operations.cc:                   *local_nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    local_nccl_context_->ErrorCheck("ncclReduce", nccl_result,
horovod/common/ops/nccl_operations.cc:                              *local_nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCE,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:  timeline.ActivityStartAll(entries, NCCL_ALLREDUCE);
horovod/common/ops/nccl_operations.cc:    auto cross_nccl_result = ncclAllReduce(buffer_data_at_rank_offset, buffer_data_at_rank_offset,
horovod/common/ops/nccl_operations.cc:                                           (size_t) total_num_elements, GetNCCLDataType(first_entry.tensor),
horovod/common/ops/nccl_operations.cc:                                           ncclOp, *cross_nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    cross_nccl_context_->ErrorCheck("ncclAllReduce", cross_nccl_result, *cross_nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:    local_nccl_context_->ErrorCheck(
horovod/common/ops/nccl_operations.cc:        "ncclAllGather",
horovod/common/ops/nccl_operations.cc:        ncclAllGather(buffer_data_at_rank_offset, buffer_data,
horovod/common/ops/nccl_operations.cc:                      GetNCCLDataType(first_entry.tensor),
horovod/common/ops/nccl_operations.cc:                      *local_nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
horovod/common/ops/nccl_operations.cc:        *local_nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_ALLGATHER,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 2, 12)
horovod/common/ops/nccl_operations.cc:    local_nccl_context_->ErrorCheck(
horovod/common/ops/nccl_operations.cc:        "ncclBroadcast",
horovod/common/ops/nccl_operations.cc:        ncclBroadcast(buffer_data_remainder, buffer_data_remainder,
horovod/common/ops/nccl_operations.cc:                      GetNCCLDataType(first_entry.tensor), root_rank,
horovod/common/ops/nccl_operations.cc:                      *local_nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
horovod/common/ops/nccl_operations.cc:        *local_nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:    local_nccl_context_->ErrorCheck(
horovod/common/ops/nccl_operations.cc:        "ncclBcast",
horovod/common/ops/nccl_operations.cc:        ncclBcast(buffer_data_remainder, (size_t)num_elements_remaining,
horovod/common/ops/nccl_operations.cc:                  GetNCCLDataType(first_entry.tensor), root_rank,
horovod/common/ops/nccl_operations.cc:                  *local_nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
horovod/common/ops/nccl_operations.cc:        *local_nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_BCAST,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:  return gpu_op_context_.FinalizeGPUQueue(
horovod/common/ops/nccl_operations.cc:      entries, true, local_nccl_op_context_.error_check_callback_);
horovod/common/ops/nccl_operations.cc:bool NCCLTorusAllreduce::Enabled(const ParameterManager& param_manager,
horovod/common/ops/nccl_operations.cc:  if (!GPUAllreduce::Enabled(param_manager, entries, response)) {
horovod/common/ops/nccl_operations.cc:void NCCLBroadcast::WaitForData(std::vector<TensorTableEntry>& entries) {
horovod/common/ops/nccl_operations.cc:    std::unordered_set<gpuEvent_t> event_set;
horovod/common/ops/nccl_operations.cc:      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
horovod/common/ops/nccl_operations.cc:Status NCCLBroadcast::Execute(std::vector<TensorTableEntry>& entries,
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPU(entries);
horovod/common/ops/nccl_operations.cc:  nccl_op_context_.InitNCCLComm(entries, response.devices());
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPUQueue(entries, response);
horovod/common/ops/nccl_operations.cc:  // On root rank, ncclbcast sends data, on other ranks it receives data.
horovod/common/ops/nccl_operations.cc:  // We only use 'ncclChar' for this operation because the type format does not
horovod/common/ops/nccl_operations.cc:#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 2, 12)
horovod/common/ops/nccl_operations.cc:  nccl_context_->ErrorCheck("ncclBroadcast",
horovod/common/ops/nccl_operations.cc:                            ncclBroadcast(data_ptr, data_ptr,
horovod/common/ops/nccl_operations.cc:                                          ncclChar, e.root_rank,
horovod/common/ops/nccl_operations.cc:                                          *nccl_op_context_.nccl_comm_,
horovod/common/ops/nccl_operations.cc:                                          *gpu_op_context_.stream),
horovod/common/ops/nccl_operations.cc:                            *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:  nccl_context_->ErrorCheck("ncclBcast",
horovod/common/ops/nccl_operations.cc:                            ncclBcast(data_ptr,
horovod/common/ops/nccl_operations.cc:                                      ncclChar, e.root_rank,
horovod/common/ops/nccl_operations.cc:                                      *nccl_op_context_.nccl_comm_,
horovod/common/ops/nccl_operations.cc:                                      *gpu_op_context_.stream),
horovod/common/ops/nccl_operations.cc:                            *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:    gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_BCAST,
horovod/common/ops/nccl_operations.cc:                              *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:  return gpu_op_context_.FinalizeGPUQueue(
horovod/common/ops/nccl_operations.cc:      entries, true, nccl_op_context_.error_check_callback_);
horovod/common/ops/nccl_operations.cc:Status NCCLAllgather::AllocateOutput(std::vector<TensorTableEntry>& entries,
horovod/common/ops/nccl_operations.cc:      LOG(WARNING) << "NCCLAllgather::AllocateOutput failed: "
horovod/common/ops/nccl_operations.cc:      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, event->event(), 0));
horovod/common/ops/nccl_operations.cc:void NCCLAllgather::WaitForData(std::vector<TensorTableEntry>& entries) {
horovod/common/ops/nccl_operations.cc:    std::unordered_set<gpuEvent_t> event_set;
horovod/common/ops/nccl_operations.cc:      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
horovod/common/ops/nccl_operations.cc:Status NCCLAllgather::Execute(std::vector<TensorTableEntry>& entries,
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPU(entries);
horovod/common/ops/nccl_operations.cc:  nccl_op_context_.InitNCCLComm(entries, response.devices());
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPUQueue(entries, response);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    auto nccl_result = ncclAllGather(
horovod/common/ops/nccl_operations.cc:        fused_input_data, buffer_data, recvcounts[0] * element_size, ncclChar,
horovod/common/ops/nccl_operations.cc:        *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    nccl_context_->ErrorCheck("ncclAllGather", nccl_result,
horovod/common/ops/nccl_operations.cc:                              *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_ALLGATHER,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    nccl_context_->ErrorCheck("ncclGroupStart", ncclGroupStart(),
horovod/common/ops/nccl_operations.cc:                              *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      auto nccl_result = ncclBroadcast(
horovod/common/ops/nccl_operations.cc:          ncclChar, rc, *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:      nccl_context_->ErrorCheck("ncclBroadcast", nccl_result,
horovod/common/ops/nccl_operations.cc:                                *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:    nccl_context_->ErrorCheck("ncclGroupEnd", ncclGroupEnd(),
horovod/common/ops/nccl_operations.cc:                              *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_BCAST,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:  return gpu_op_context_.FinalizeGPUQueue(
horovod/common/ops/nccl_operations.cc:      entries, true, nccl_op_context_.error_check_callback_);
horovod/common/ops/nccl_operations.cc:bool NCCLAllgather::Enabled(const ParameterManager& param_manager,
horovod/common/ops/nccl_operations.cc:void NCCLAlltoall::WaitForData(std::vector<TensorTableEntry>& entries) {
horovod/common/ops/nccl_operations.cc:    std::unordered_set<gpuEvent_t> event_set;
horovod/common/ops/nccl_operations.cc:      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
horovod/common/ops/nccl_operations.cc:Status NCCLAlltoall::Execute(std::vector<TensorTableEntry>& entries,
horovod/common/ops/nccl_operations.cc:#ifdef NCCL_P2P_SUPPORTED
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPU(entries);
horovod/common/ops/nccl_operations.cc:  nccl_op_context_.InitNCCLComm(entries, response.devices());
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPUQueue(entries, response);
horovod/common/ops/nccl_operations.cc:  nccl_context_->ErrorCheck("ncclGroupStart", ncclGroupStart(),
horovod/common/ops/nccl_operations.cc:                            *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      auto nccl_result =
horovod/common/ops/nccl_operations.cc:          ncclRecv((uint8_t*)e.output->data() +
horovod/common/ops/nccl_operations.cc:                   recvcounts[i] * DataType_Size(e.tensor->dtype()), ncclChar,
horovod/common/ops/nccl_operations.cc:                   i, *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:      nccl_context_->ErrorCheck("ncclRecv", nccl_result,
horovod/common/ops/nccl_operations.cc:                                *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      auto nccl_result =
horovod/common/ops/nccl_operations.cc:          ncclSend((uint8_t*)e.tensor->data() +
horovod/common/ops/nccl_operations.cc:                   sendcounts[i] * DataType_Size(e.tensor->dtype()), ncclChar,
horovod/common/ops/nccl_operations.cc:                   i, *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:      nccl_context_->ErrorCheck("ncclSend", nccl_result,
horovod/common/ops/nccl_operations.cc:                                *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:  nccl_context_->ErrorCheck("ncclGroupEnd", ncclGroupEnd(),
horovod/common/ops/nccl_operations.cc:                            *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:    gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_ALLTOALL,
horovod/common/ops/nccl_operations.cc:                              *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:  return gpu_op_context_.FinalizeGPUQueue(entries);
horovod/common/ops/nccl_operations.cc:      "NCCLAlltoall requires NCCL version >= 2.7.0. If your NCCL installation "
horovod/common/ops/nccl_operations.cc:      "and you installed with HOROVOD_GPU_OPERATIONS=NCCL, reinstall with only "
horovod/common/ops/nccl_operations.cc:      "operations individually specified (i.e. HOROVOD_GPU_ALLREDUCE=NCCL "
horovod/common/ops/nccl_operations.cc:      "HOROVOD_GPU_BROADCAST=NCCL "
horovod/common/ops/nccl_operations.cc:      "HOROVOD_GPU_ALLGATHER=NCCL). Otherwise, exclude "
horovod/common/ops/nccl_operations.cc:      "HOROVOD_GPU_ALLTOALL=NCCL from your "
horovod/common/ops/nccl_operations.cc:Status NCCLReducescatter::Execute(std::vector<TensorTableEntry>& entries,
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPU(entries);
horovod/common/ops/nccl_operations.cc:  nccl_op_context_.InitNCCLComm(entries, response.devices());
horovod/common/ops/nccl_operations.cc:  gpu_op_context_.InitGPUQueue(entries, response);
horovod/common/ops/nccl_operations.cc:      // Fused: ncclReduceScatter() in place on the fusion buffer, cf.:
horovod/common/ops/nccl_operations.cc:      // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/inplace.html
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    auto nccl_result = ncclReduceScatter(
horovod/common/ops/nccl_operations.cc:        GetNCCLDataType(first_entry.tensor), ncclSum,
horovod/common/ops/nccl_operations.cc:        *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    nccl_context_->ErrorCheck("ncclReduceScatter", nccl_result,
horovod/common/ops/nccl_operations.cc:                              *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCESCATTER,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:    nccl_context_->ErrorCheck("ncclGroupStart", ncclGroupStart(),
horovod/common/ops/nccl_operations.cc:                              *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      auto nccl_result =
horovod/common/ops/nccl_operations.cc:          ncclReduce(send_pointer, recv_pointer, recvcounts[recv_rank],
horovod/common/ops/nccl_operations.cc:                     GetNCCLDataType(first_entry.tensor), ncclSum, recv_rank,
horovod/common/ops/nccl_operations.cc:                     *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:      nccl_context_->ErrorCheck("ncclReduce", nccl_result,
horovod/common/ops/nccl_operations.cc:                                *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:    nccl_context_->ErrorCheck("ncclGroupEnd", ncclGroupEnd(),
horovod/common/ops/nccl_operations.cc:                              *nccl_op_context_.nccl_comm_);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCE,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
horovod/common/ops/nccl_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/nccl_operations.cc:  return gpu_op_context_.FinalizeGPUQueue(
horovod/common/ops/nccl_operations.cc:      entries, true, nccl_op_context_.error_check_callback_);
horovod/common/ops/nccl_operations.cc:bool NCCLReducescatter::Enabled(const ParameterManager& param_manager,
horovod/common/ops/nccl_operations.cc:void NCCLReducescatter::WaitForData(std::vector<TensorTableEntry>& entries) {
horovod/common/ops/nccl_operations.cc:    std::unordered_set<gpuEvent_t> event_set;
horovod/common/ops/nccl_operations.cc:      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
horovod/common/ops/adasum/adasum.h:  //              reduce-scatter algorithm, e.g. the one in NCCL, which may be
horovod/common/ops/gpu_operations.h:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/gpu_operations.h:#ifndef HOROVOD_GPU_OPERATIONS_H
horovod/common/ops/gpu_operations.h:#define HOROVOD_GPU_OPERATIONS_H
horovod/common/ops/gpu_operations.h:#if HAVE_CUDA
horovod/common/ops/gpu_operations.h:#include <cuda_fp16.h>
horovod/common/ops/gpu_operations.h:#include <cuda_runtime.h>
horovod/common/ops/gpu_operations.h:using gpuError_t = cudaError_t;
horovod/common/ops/gpu_operations.h:using gpuEvent_t = cudaEvent_t;
horovod/common/ops/gpu_operations.h:using gpuStream_t = cudaStream_t;
horovod/common/ops/gpu_operations.h:#elif HAVE_ROCM
horovod/common/ops/gpu_operations.h:using gpuError_t = hipError_t;
horovod/common/ops/gpu_operations.h:using gpuEvent_t = hipEvent_t;
horovod/common/ops/gpu_operations.h:using gpuStream_t = hipStream_t;
horovod/common/ops/gpu_operations.h:class GPUContext {
horovod/common/ops/gpu_operations.h:  GPUContext();
horovod/common/ops/gpu_operations.h:  ~GPUContext();
horovod/common/ops/gpu_operations.h:  // The GPU stream used for data transfers and within-allreduce operations.
horovod/common/ops/gpu_operations.h:  // A naive implementation would use the TensorFlow StreamExecutor GPU
horovod/common/ops/gpu_operations.h:  // and kernel executions (for accumulation of values on the GPU). However,
horovod/common/ops/gpu_operations.h:  // transfers before the GPU calls are complete. In order to wait for those
horovod/common/ops/gpu_operations.h:  // GPU operations, if we were using the TensorFlow stream, we would have to
horovod/common/ops/gpu_operations.h:  std::vector<std::unordered_map<int, gpuStream_t>> streams;
horovod/common/ops/gpu_operations.h:  void ErrorCheck(std::string op_name, gpuError_t gpu_result);
horovod/common/ops/gpu_operations.h:                   std::string name, gpuStream_t& stream);
horovod/common/ops/gpu_operations.h:  Event RecordEvent(gpuStream_t& stream);
horovod/common/ops/gpu_operations.h:  void StreamCreate(gpuStream_t* stream);
horovod/common/ops/gpu_operations.h:  void StreamSynchronize(gpuStream_t stream);
horovod/common/ops/gpu_operations.h:                      gpuStream_t stream);
horovod/common/ops/gpu_operations.h:                      gpuStream_t stream);
horovod/common/ops/gpu_operations.h:                      gpuStream_t stream);
horovod/common/ops/gpu_operations.h:                       DataType dtype, gpuStream_t stream);
horovod/common/ops/gpu_operations.h:class GPUOpContext {
horovod/common/ops/gpu_operations.h:  GPUOpContext(GPUContext* context, HorovodGlobalState* global_state);
horovod/common/ops/gpu_operations.h:  void InitGPU(const std::vector<TensorTableEntry>& entries);
horovod/common/ops/gpu_operations.h:  void InitGPUQueue(const std::vector<TensorTableEntry>& entries,
horovod/common/ops/gpu_operations.h:  FinalizeGPUQueue(std::vector<TensorTableEntry>& entries,
horovod/common/ops/gpu_operations.h:  // GPU events are used as an alternative to host-device synchronization (which
horovod/common/ops/gpu_operations.h:  // stalls the GPU pipeline) for the purpose of recording timing on the Horovod
horovod/common/ops/gpu_operations.h:  // When an event we wish to record occurs (for example, NCCL_ALLREDUCE), the
horovod/common/ops/gpu_operations.h:  // For more information of CUDA Events, see:
horovod/common/ops/gpu_operations.h:  // https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
horovod/common/ops/gpu_operations.h:  gpuStream_t* stream;
horovod/common/ops/gpu_operations.h:  GPUContext* gpu_context_;
horovod/common/ops/gpu_operations.h:class GPUAllreduce : public AllreduceOp {
horovod/common/ops/gpu_operations.h:  GPUAllreduce(GPUContext* context, HorovodGlobalState* global_state);
horovod/common/ops/gpu_operations.h:  GPUContext* gpu_context_;
horovod/common/ops/gpu_operations.h:  GPUOpContext gpu_op_context_;
horovod/common/ops/gpu_operations.h:class GPUAllgather : public AllgatherOp {
horovod/common/ops/gpu_operations.h:  GPUAllgather(GPUContext* context, HorovodGlobalState* global_state);
horovod/common/ops/gpu_operations.h:  GPUContext* gpu_context_;
horovod/common/ops/gpu_operations.h:  GPUOpContext gpu_op_context_;
horovod/common/ops/gpu_operations.h:class GPUBroadcast : public BroadcastOp {
horovod/common/ops/gpu_operations.h:  GPUBroadcast(GPUContext* context, HorovodGlobalState* global_state);
horovod/common/ops/gpu_operations.h:  GPUContext* gpu_context_;
horovod/common/ops/gpu_operations.h:  GPUOpContext gpu_op_context_;
horovod/common/ops/gpu_operations.h:class GPUAlltoall : public AlltoallOp {
horovod/common/ops/gpu_operations.h:  GPUAlltoall(GPUContext* context, HorovodGlobalState* global_state);
horovod/common/ops/gpu_operations.h:  GPUContext* gpu_context_;
horovod/common/ops/gpu_operations.h:  GPUOpContext gpu_op_context_;
horovod/common/ops/gpu_operations.h:class GPUReducescatter : public ReducescatterOp {
horovod/common/ops/gpu_operations.h:  GPUReducescatter(GPUContext* context, HorovodGlobalState* global_state);
horovod/common/ops/gpu_operations.h:  GPUContext* gpu_context_;
horovod/common/ops/gpu_operations.h:  GPUOpContext gpu_op_context_;
horovod/common/ops/gpu_operations.h:#endif // HOROVOD_GPU_OPERATIONS_H
horovod/common/ops/hip_operations.cc:#include "gpu_operations.h"
horovod/common/ops/hip_operations.cc:#include "rocm/hip_kernels.h"
horovod/common/ops/hip_operations.cc:class GPUContext::impl {
horovod/common/ops/hip_operations.cc:  hipError_t GetGpuEvent(Event* event, hipStream_t stream) {
horovod/common/ops/hip_operations.cc:  hipError_t ReleaseGpuEvent(Event event) {
horovod/common/ops/hip_operations.cc:    ErrorCheck("GetGpuEvent", GetGpuEvent(&event, stream));
horovod/common/ops/hip_operations.cc:    ErrorCheck("GetGpuEvent", GetGpuEvent(&event, stream));
horovod/common/ops/hip_operations.cc:        throw std::logic_error(std::string("cudaEventSynchronize failed: ") +
horovod/common/ops/hip_operations.cc:      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
horovod/common/ops/hip_operations.cc:          throw std::logic_error(std::string("cudaEventQuery failed: ") +
horovod/common/ops/hip_operations.cc:      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
horovod/common/ops/hip_operations.cc:      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
horovod/common/ops/hip_operations.cc:    throw std::logic_error("ScaleBuffer not implemented for AMD GPUs.");
horovod/common/ops/hip_operations.cc:#include "gpu_context_impl.cc"
horovod/common/ops/mpi_gpu_operations.cc:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/mpi_gpu_operations.cc:#include "mpi_gpu_operations.h"
horovod/common/ops/mpi_gpu_operations.cc:#if HAVE_CUDA
horovod/common/ops/mpi_gpu_operations.cc:#include "cuda/cuda_kernels.h"
horovod/common/ops/mpi_gpu_operations.cc:#if HAVE_ROCM
horovod/common/ops/mpi_gpu_operations.cc:#include "rocm/hip_kernels.h"
horovod/common/ops/mpi_gpu_operations.cc:MPI_GPUAllreduce::MPI_GPUAllreduce(GPUContext* gpu_context,
horovod/common/ops/mpi_gpu_operations.cc:    : GPUAllreduce(gpu_context, global_state) {}
horovod/common/ops/mpi_gpu_operations.cc:Status MPI_GPUAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
horovod/common/ops/mpi_gpu_operations.cc:  gpu_op_context_.InitGPU(entries);
horovod/common/ops/mpi_gpu_operations.cc:    gpu_context_->StreamSynchronize(gpu_context_->streams[global_state_->current_nccl_stream][entries[0].device]);
horovod/common/ops/mpi_gpu_operations.cc:    gpu_context_->StreamSynchronize(gpu_context_->streams[global_state_->current_nccl_stream][entries[0].device]);
horovod/common/ops/mpi_gpu_operations.cc:MPI_GPUAllgather::MPI_GPUAllgather(GPUContext* gpu_context,
horovod/common/ops/mpi_gpu_operations.cc:    : GPUAllgather(gpu_context, global_state) {}
horovod/common/ops/mpi_gpu_operations.cc:Status MPI_GPUAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
horovod/common/ops/mpi_gpu_operations.cc:  gpu_op_context_.InitGPU(entries);
horovod/common/ops/mpi_gpu_operations.cc:    gpu_context_->StreamSynchronize(gpu_context_->streams[global_state_->current_nccl_stream][entries[0].device]);
horovod/common/ops/mpi_gpu_operations.cc:    gpu_context_->StreamSynchronize(gpu_context_->streams[global_state_->current_nccl_stream][entries[0].device]);
horovod/common/ops/mpi_gpu_operations.cc:MPI_GPUAlltoall::MPI_GPUAlltoall(GPUContext* gpu_context,
horovod/common/ops/mpi_gpu_operations.cc:    : GPUAlltoall(gpu_context, global_state) {}
horovod/common/ops/mpi_gpu_operations.cc:Status MPI_GPUAlltoall::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
horovod/common/ops/mpi_gpu_operations.cc:  gpu_op_context_.InitGPU(entries);
horovod/common/ops/mpi_gpu_operations.cc:MPI_GPUReducescatter::MPI_GPUReducescatter(GPUContext* gpu_context,
horovod/common/ops/mpi_gpu_operations.cc:    : GPUReducescatter(gpu_context, global_state) {}
horovod/common/ops/mpi_gpu_operations.cc:Status MPI_GPUReducescatter::Execute(std::vector<TensorTableEntry>& entries,
horovod/common/ops/mpi_gpu_operations.cc:  gpu_op_context_.InitGPU(entries);
horovod/common/ops/mpi_gpu_operations.cc:    gpu_context_->StreamSynchronize(
horovod/common/ops/mpi_gpu_operations.cc:        gpu_context_
horovod/common/ops/mpi_gpu_operations.cc:            ->streams[global_state_->current_nccl_stream][first_entry.device]);
horovod/common/ops/mpi_gpu_operations.cc:    gpu_context_->StreamSynchronize(
horovod/common/ops/mpi_gpu_operations.cc:        gpu_context_
horovod/common/ops/mpi_gpu_operations.cc:            ->streams[global_state_->current_nccl_stream][first_entry.device]);
horovod/common/ops/collective_operations.cc:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/collective_operations.cc:  // On GPU data readiness is signalled by ready_event.
horovod/common/ops/collective_operations.cc:      first_entry.device, first_entry.context->framework(), global_state_->current_nccl_stream);
horovod/common/ops/collective_operations.cc:      first_entry.device, first_entry.context->framework(), global_state_->current_nccl_stream);
horovod/common/ops/collective_operations.cc:      global_state_->current_nccl_stream);
horovod/common/ops/gpu_context_impl.cc:GPUContext::GPUContext() : pimpl{new impl} {}
horovod/common/ops/gpu_context_impl.cc:GPUContext::~GPUContext() = default;
horovod/common/ops/gpu_context_impl.cc:void GPUContext::Finalize() { finalizer_thread_pool.reset(); }
horovod/common/ops/gpu_context_impl.cc:void GPUContext::ErrorCheck(std::string op_name, gpuError_t gpu_result) {
horovod/common/ops/gpu_context_impl.cc:  pimpl->ErrorCheck(op_name, gpu_result);
horovod/common/ops/gpu_context_impl.cc:void GPUContext::RecordEvent(
horovod/common/ops/gpu_context_impl.cc:    gpuStream_t& stream) {
horovod/common/ops/gpu_context_impl.cc:Event GPUContext::RecordEvent(gpuStream_t& stream) {
horovod/common/ops/gpu_context_impl.cc:void GPUContext::ReleaseEvent(Event event) {
horovod/common/ops/gpu_context_impl.cc:  pimpl->ErrorCheck("ReleaseGpuEvent", pimpl->ReleaseGpuEvent(event));
horovod/common/ops/gpu_context_impl.cc:void GPUContext::WaitForEvents(
horovod/common/ops/gpu_context_impl.cc:void GPUContext::WaitForEventsElastic(
horovod/common/ops/gpu_context_impl.cc:void GPUContext::ClearEvents(
horovod/common/ops/gpu_context_impl.cc:void GPUContext::StreamCreate(gpuStream_t* stream) {
horovod/common/ops/gpu_context_impl.cc:void GPUContext::StreamSynchronize(gpuStream_t stream) {
horovod/common/ops/gpu_context_impl.cc:int GPUContext::GetDevice() { return pimpl->GetDevice(); }
horovod/common/ops/gpu_context_impl.cc:void GPUContext::SetDevice(int device) { pimpl->SetDevice(device); }
horovod/common/ops/gpu_context_impl.cc:void GPUContext::MemcpyAsyncD2D(void* dst, const void* src, size_t count,
horovod/common/ops/gpu_context_impl.cc:                                gpuStream_t stream) {
horovod/common/ops/gpu_context_impl.cc:void GPUContext::MemcpyAsyncH2D(void* dst, const void* src, size_t count,
horovod/common/ops/gpu_context_impl.cc:                                gpuStream_t stream) {
horovod/common/ops/gpu_context_impl.cc:void GPUContext::MemcpyAsyncD2H(void* dst, const void* src, size_t count,
horovod/common/ops/gpu_context_impl.cc:                                gpuStream_t stream) {
horovod/common/ops/gpu_context_impl.cc:void GPUContext::ScaleBufferImpl(const void* fused_input_data,
horovod/common/ops/gpu_context_impl.cc:                                 gpuStream_t stream) {
horovod/common/ops/nccl_operations.h:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/nccl_operations.h:#ifndef HOROVOD_NCCL_OPERATIONS_H
horovod/common/ops/nccl_operations.h:#define HOROVOD_NCCL_OPERATIONS_H
horovod/common/ops/nccl_operations.h:#if HAVE_CUDA
horovod/common/ops/nccl_operations.h:#include <nccl.h>
horovod/common/ops/nccl_operations.h:#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 7, 0)
horovod/common/ops/nccl_operations.h:#define NCCL_P2P_SUPPORTED
horovod/common/ops/nccl_operations.h:#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
horovod/common/ops/nccl_operations.h:#define NCCL_AVG_SUPPORTED
horovod/common/ops/nccl_operations.h:#elif HAVE_ROCM
horovod/common/ops/nccl_operations.h:#define NCCL_P2P_SUPPORTED
horovod/common/ops/nccl_operations.h:#include "gpu_operations.h"
horovod/common/ops/nccl_operations.h:ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor);
horovod/common/ops/nccl_operations.h:struct NCCLContext {
horovod/common/ops/nccl_operations.h:  // indexed by [nccl stream][{process set id, device id vector}]
horovod/common/ops/nccl_operations.h:      std::unordered_map<std::tuple<int32_t, std::vector<int32_t>>, ncclComm_t>>
horovod/common/ops/nccl_operations.h:      nccl_comms;
horovod/common/ops/nccl_operations.h:  void ErrorCheck(std::string op_name, ncclResult_t nccl_result,
horovod/common/ops/nccl_operations.h:                  ncclComm_t& nccl_comm);
horovod/common/ops/nccl_operations.h:class NCCLOpContext {
horovod/common/ops/nccl_operations.h:  NCCLOpContext(NCCLContext* nccl_context, HorovodGlobalState* global_state,
horovod/common/ops/nccl_operations.h:      : nccl_comm_(nullptr),
horovod/common/ops/nccl_operations.h:        error_check_callback_(std::bind(&NCCLOpContext::AsyncErrorCheck, this)),
horovod/common/ops/nccl_operations.h:        nccl_context_(nccl_context), global_state_(global_state),
horovod/common/ops/nccl_operations.h:  void InitNCCLComm(const std::vector<TensorTableEntry>& entries,
horovod/common/ops/nccl_operations.h:                    const std::vector<int32_t>& nccl_device_map);
horovod/common/ops/nccl_operations.h:  ncclComm_t* nccl_comm_;
horovod/common/ops/nccl_operations.h:  void PopulateNCCLCommStrategy(int& nccl_rank, int& nccl_size,
horovod/common/ops/nccl_operations.h:                                Communicator& nccl_id_bcast_comm,
horovod/common/ops/nccl_operations.h:  NCCLContext* nccl_context_;
horovod/common/ops/nccl_operations.h:class NCCLAllreduce : public GPUAllreduce {
horovod/common/ops/nccl_operations.h:  NCCLAllreduce(NCCLContext* nccl_context, GPUContext* gpu_context,
horovod/common/ops/nccl_operations.h:      : GPUAllreduce(gpu_context, global_state), nccl_context_(nccl_context),
horovod/common/ops/nccl_operations.h:        nccl_op_context_(nccl_context, global_state, communicator_type),
horovod/common/ops/nccl_operations.h:  NCCLContext* nccl_context_;
horovod/common/ops/nccl_operations.h:  NCCLOpContext nccl_op_context_;
horovod/common/ops/nccl_operations.h:class NCCLBroadcast : public GPUBroadcast {
horovod/common/ops/nccl_operations.h:  NCCLBroadcast(NCCLContext* nccl_context, GPUContext* gpu_context,
horovod/common/ops/nccl_operations.h:      : GPUBroadcast(gpu_context, global_state), nccl_context_(nccl_context),
horovod/common/ops/nccl_operations.h:        nccl_op_context_(nccl_context, global_state, Communicator::GLOBAL),
horovod/common/ops/nccl_operations.h:  NCCLContext* nccl_context_;
horovod/common/ops/nccl_operations.h:  NCCLOpContext nccl_op_context_;
horovod/common/ops/nccl_operations.h:class NCCLAlltoall : public GPUAlltoall {
horovod/common/ops/nccl_operations.h:  NCCLAlltoall(NCCLContext* nccl_context, GPUContext* gpu_context,
horovod/common/ops/nccl_operations.h:      : GPUAlltoall(gpu_context, global_state), nccl_context_(nccl_context),
horovod/common/ops/nccl_operations.h:        nccl_op_context_(nccl_context, global_state, Communicator::GLOBAL),
horovod/common/ops/nccl_operations.h:          << "NCCLAlltoall::PrepareOutputAndParams failed to allocate output: "
horovod/common/ops/nccl_operations.h:      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, event->event(), 0));
horovod/common/ops/nccl_operations.h:      LOG(WARNING) << "NCCLAlltoall::PrepareOutputAndParams failed to allocate "
horovod/common/ops/nccl_operations.h:      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, revent->event(), 0));
horovod/common/ops/nccl_operations.h:  NCCLContext* nccl_context_;
horovod/common/ops/nccl_operations.h:  NCCLOpContext nccl_op_context_;
horovod/common/ops/nccl_operations.h:class NCCLHierarchicalAllreduce : public NCCLAllreduce {
horovod/common/ops/nccl_operations.h:  NCCLHierarchicalAllreduce(NCCLContext* nccl_context, GPUContext* gpu_context,
horovod/common/ops/nccl_operations.h:      : NCCLAllreduce(nccl_context, gpu_context, global_state,
horovod/common/ops/nccl_operations.h:class NCCLTorusAllreduce : public GPUAllreduce {
horovod/common/ops/nccl_operations.h:  NCCLTorusAllreduce(NCCLContext* local_nccl_context, NCCLContext* cross_nccl_context,
horovod/common/ops/nccl_operations.h:                     GPUContext* gpu_context, HorovodGlobalState* global_state)
horovod/common/ops/nccl_operations.h:      : GPUAllreduce(gpu_context, global_state),
horovod/common/ops/nccl_operations.h:        local_nccl_context_(local_nccl_context),
horovod/common/ops/nccl_operations.h:        cross_nccl_context_(cross_nccl_context),
horovod/common/ops/nccl_operations.h:        local_nccl_op_context_(local_nccl_context, global_state, Communicator::LOCAL),
horovod/common/ops/nccl_operations.h:        cross_nccl_op_context_(cross_nccl_context, global_state, Communicator::CROSS),
horovod/common/ops/nccl_operations.h:  NCCLContext* local_nccl_context_;
horovod/common/ops/nccl_operations.h:  NCCLContext* cross_nccl_context_;
horovod/common/ops/nccl_operations.h:  NCCLOpContext local_nccl_op_context_;
horovod/common/ops/nccl_operations.h:  NCCLOpContext cross_nccl_op_context_;
horovod/common/ops/nccl_operations.h:class NCCLAllgather : public GPUAllgather {
horovod/common/ops/nccl_operations.h:  NCCLAllgather(NCCLContext* nccl_context, GPUContext* gpu_context,
horovod/common/ops/nccl_operations.h:      : GPUAllgather(gpu_context, global_state), nccl_context_(nccl_context),
horovod/common/ops/nccl_operations.h:        nccl_op_context_(nccl_context, global_state, Communicator::GLOBAL),
horovod/common/ops/nccl_operations.h:  NCCLContext* nccl_context_;
horovod/common/ops/nccl_operations.h:  NCCLOpContext nccl_op_context_;
horovod/common/ops/nccl_operations.h:class NCCLReducescatter : public GPUReducescatter {
horovod/common/ops/nccl_operations.h:  NCCLReducescatter(NCCLContext* nccl_context, GPUContext* gpu_context,
horovod/common/ops/nccl_operations.h:      : GPUReducescatter(gpu_context, global_state),
horovod/common/ops/nccl_operations.h:        nccl_context_(nccl_context),
horovod/common/ops/nccl_operations.h:        nccl_op_context_(nccl_context, global_state, Communicator::GLOBAL),
horovod/common/ops/nccl_operations.h:  NCCLContext* nccl_context_;
horovod/common/ops/nccl_operations.h:  NCCLOpContext nccl_op_context_;
horovod/common/ops/nccl_operations.h:#endif // HOROVOD_NCCL_OPERATIONS_H
horovod/common/ops/ddl_operations.cc:                           GPUContext* gpu_context,
horovod/common/ops/ddl_operations.cc:    : GPUAllreduce(gpu_context, global_state),
horovod/common/ops/ddl_operations.cc:  gpu_op_context_.InitGPU(entries);
horovod/common/ops/ddl_operations.cc:  gpu_op_context_.InitGPUQueue(entries, response);
horovod/common/ops/ddl_operations.cc:    throw std::logic_error("DDL does not support more than one GPU device per process.");
horovod/common/ops/ddl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, MEMCPY_IN_FUSION_BUFFER, *gpu_op_context_.stream);
horovod/common/ops/ddl_operations.cc:    gpu_context_->MemcpyAsyncD2D(buffer_data, fused_input_data, buffer_len, *gpu_op_context_.stream);
horovod/common/ops/ddl_operations.cc:    gpu_context_->RecordEvent(gpu_op_context_.event_queue, MEMCPY_IN_FUSION_BUFFER, *gpu_op_context_.stream);
horovod/common/ops/ddl_operations.cc:  gpu_context_->WaitForEvents(gpu_op_context_.event_queue, entries, timeline, nullptr, global_state_->elastic_enabled);
horovod/common/ops/ddl_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, MEMCPY_OUT_FUSION_BUFFER, *gpu_op_context_.stream);
horovod/common/ops/ddl_operations.cc:  return gpu_op_context_.FinalizeGPUQueue(entries);
horovod/common/ops/ddl_operations.cc:void DDLAllreduce::DDLInit(DDLContext* ddl_context, GPUContext* gpu_context) {
horovod/common/ops/ddl_operations.cc:  LOG(WARNING) << "DDL backend has been deprecated. Please, start using the NCCL backend by "
horovod/common/ops/ddl_operations.cc:                  "building Horovod with 'HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL'.";
horovod/common/ops/ddl_operations.cc:  ddl_context->ddl_local_device_id = gpu_context->GetDevice();
horovod/common/ops/operation_manager.h:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/collective_operations.h:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/mpi_operations.cc:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/rocm/hip_kernels.cu:// Copyright (C) 2020 NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/rocm/hip_kernels.cu:// ATTENTION: Any change here might obsolete cuda_kernels.cu in cuda folder.
horovod/common/ops/rocm/hip_kernels.cu://            Please keep this file synced with cuda_kernels.cu.
horovod/common/ops/rocm/hip_kernels.cu:void BatchedD2DMemcpyROCmImpl(BatchedD2DParams& params, int num_copies, hipStream_t stream)
horovod/common/ops/rocm/hip_kernels.cu:#if __CUDA_ARCH__ > 530
horovod/common/ops/rocm/hip_kernels.cu:#if __CUDA_ARCH__ > 530
horovod/common/ops/rocm/hip_kernels.cu:void ScaleBufferROCmImpl(const void* fused_input_data, void* buffer_data, const int64_t num_elements, double scale_factor,
horovod/common/ops/rocm/hip_kernels.cu:                             " not supported by ScaleBufferROCmImpl.");
horovod/common/ops/rocm/hip_kernels.cu:#if __CUDA_ARCH__ > 530
horovod/common/ops/rocm/hip_kernels.cu:#if __CUDA_ARCH__ > 530
horovod/common/ops/rocm/hip_kernels.cu:void BatchedScaledD2DMemcpyROCmImpl(BatchedD2DParams& params, int num_copies, double scale_factor,
horovod/common/ops/rocm/hip_kernels.cu:                            " not supported by BatchedScaledD2DMemcpyROCmImpl.");
horovod/common/ops/rocm/CMakeLists.txt:message(STATUS "Build Horovod for ROCm")
horovod/common/ops/rocm/CMakeLists.txt:    set(HCC_PATH "${ROCM_PATH}/hcc" CACHE PATH "Path to which HCC has been set")
horovod/common/ops/rocm/CMakeLists.txt:list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
horovod/common/ops/rocm/CMakeLists.txt:set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin")
horovod/common/ops/rocm/CMakeLists.txt:hip_add_library(horovod_cuda_kernels STATIC hip_kernels.cu)
horovod/common/ops/rocm/CMakeLists.txt:target_compile_definitions(horovod_cuda_kernels PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
horovod/common/ops/rocm/CMakeLists.txt:hip_add_library(compatible_horovod_cuda_kernels STATIC hip_kernels.cu)
horovod/common/ops/rocm/CMakeLists.txt:target_compile_definitions(compatible_horovod_cuda_kernels PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
horovod/common/ops/rocm/hip_kernels.h:// Copyright (C) 2020 NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/rocm/hip_kernels.h:// ATTENTION: Any change here might obsolete cuda_kernels.h in cuda folder.
horovod/common/ops/rocm/hip_kernels.h://            Please keep this file synced with cuda_kernels.h.
horovod/common/ops/rocm/hip_kernels.h:void BatchedD2DMemcpyROCmImpl(BatchedD2DParams& params, int num_copies, hipStream_t stream);
horovod/common/ops/rocm/hip_kernels.h:void ScaleBufferROCmImpl(const void* fused_input_data, void* buffer_data, const int64_t num_elements,
horovod/common/ops/rocm/hip_kernels.h:void BatchedScaledD2DMemcpyROCmImpl(BatchedD2DParams& params, int num_copies, double scale_factor,
horovod/common/ops/adasum_gpu_operations.h:#ifndef HOROVOD_ADASUM_GPU_OPERATIONS_H
horovod/common/ops/adasum_gpu_operations.h:#define HOROVOD_ADASUM_GPU_OPERATIONS_H
horovod/common/ops/adasum_gpu_operations.h:#include "nccl_operations.h"
horovod/common/ops/adasum_gpu_operations.h:class AdasumGpuAllreduceOp : public AdasumMPI, public NCCLAllreduce {
horovod/common/ops/adasum_gpu_operations.h:  AdasumGpuAllreduceOp(MPIContext* mpi_context, NCCLContext* nccl_context,
horovod/common/ops/adasum_gpu_operations.h:                       GPUContext* gpu_context,
horovod/common/ops/adasum_gpu_operations.h:  ~AdasumGpuAllreduceOp();
horovod/common/ops/adasum_gpu_operations.h:  Status NcclHierarchical(std::vector<TensorTableEntry>& entries,
horovod/common/ops/adasum_gpu_operations.h:#endif // HOROVOD_ADASUM_GPU_OPERATIONS_H
horovod/common/ops/operation_manager.cc:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/adasum_gpu_operations.cc:#include "adasum_gpu_operations.h"
horovod/common/ops/adasum_gpu_operations.cc:AdasumGpuAllreduceOp::AdasumGpuAllreduceOp(MPIContext* mpi_context,
horovod/common/ops/adasum_gpu_operations.cc:                                           NCCLContext* nccl_context,
horovod/common/ops/adasum_gpu_operations.cc:                                           GPUContext* gpu_context,
horovod/common/ops/adasum_gpu_operations.cc:      NCCLAllreduce(nccl_context, gpu_context, global_state,
horovod/common/ops/adasum_gpu_operations.cc:  gpu_op_context_.host_buffer = (uint8_t*)malloc(current_host_buffer_length);
horovod/common/ops/adasum_gpu_operations.cc:AdasumGpuAllreduceOp::~AdasumGpuAllreduceOp() {
horovod/common/ops/adasum_gpu_operations.cc:  if (gpu_op_context_.host_buffer != nullptr) {
horovod/common/ops/adasum_gpu_operations.cc:    free(gpu_op_context_.host_buffer);
horovod/common/ops/adasum_gpu_operations.cc:void AdasumGpuAllreduceOp::WaitForData(std::vector<TensorTableEntry>& entries) {
horovod/common/ops/adasum_gpu_operations.cc:Status AdasumGpuAllreduceOp::Execute(std::vector<TensorTableEntry>& entries,
horovod/common/ops/adasum_gpu_operations.cc:  return NcclHierarchical(entries, response);
horovod/common/ops/adasum_gpu_operations.cc:uint8_t* AdasumGpuAllreduceOp::GetHostBuffer(uint64_t buffer_length) {
horovod/common/ops/adasum_gpu_operations.cc:  return CheckBufferAndReallocate((uint8_t**)&gpu_op_context_.host_buffer,
horovod/common/ops/adasum_gpu_operations.cc:AdasumGpuAllreduceOp::NcclHierarchical(std::vector<TensorTableEntry>& entries,
horovod/common/ops/adasum_gpu_operations.cc:  // Determine GPU IDs of the devices participating in this communicator.
horovod/common/ops/adasum_gpu_operations.cc:  std::vector<int32_t> nccl_device_map;
horovod/common/ops/adasum_gpu_operations.cc:  nccl_device_map.reserve(process_set.controller->GetLocalCommRanks().size());
horovod/common/ops/adasum_gpu_operations.cc:    nccl_device_map.push_back(response.devices()[rank]);
horovod/common/ops/adasum_gpu_operations.cc:  gpu_op_context_.InitGPU(entries);
horovod/common/ops/adasum_gpu_operations.cc:  nccl_op_context_.InitNCCLComm(entries, nccl_device_map);
horovod/common/ops/adasum_gpu_operations.cc:  gpu_op_context_.InitGPUQueue(entries, response);
horovod/common/ops/adasum_gpu_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
horovod/common/ops/adasum_gpu_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/adasum_gpu_operations.cc:  // For the part of data divisible by local_size, perform NCCL
horovod/common/ops/adasum_gpu_operations.cc:  // ReduceScatter - Parallelized MPI Allreduce - NCCL Allgather. For the
horovod/common/ops/adasum_gpu_operations.cc:  // non-divisible part (if any), do NCCL Reduce (at rank local_size-1),
horovod/common/ops/adasum_gpu_operations.cc:  // MPI Allreduce (across rank (local_size-1)'s), and NCCL Bcast
horovod/common/ops/adasum_gpu_operations.cc:    auto nccl_result = ncclReduceScatter(
horovod/common/ops/adasum_gpu_operations.cc:        (size_t)num_elements_per_rank, GetNCCLDataType(first_entry.tensor),
horovod/common/ops/adasum_gpu_operations.cc:        ncclSum, *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/adasum_gpu_operations.cc:    nccl_context_->ErrorCheck("ncclReduceScatter", nccl_result,
horovod/common/ops/adasum_gpu_operations.cc:                              *nccl_op_context_.nccl_comm_);
horovod/common/ops/adasum_gpu_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCESCATTER,
horovod/common/ops/adasum_gpu_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/adasum_gpu_operations.cc:    auto nccl_result =
horovod/common/ops/adasum_gpu_operations.cc:        ncclReduce(fused_input_data_remainder, buffer_data_remainder,
horovod/common/ops/adasum_gpu_operations.cc:                   GetNCCLDataType(first_entry.tensor), ncclSum, root_rank,
horovod/common/ops/adasum_gpu_operations.cc:                   *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
horovod/common/ops/adasum_gpu_operations.cc:    nccl_context_->ErrorCheck("ncclReduce", nccl_result,
horovod/common/ops/adasum_gpu_operations.cc:                              *nccl_op_context_.nccl_comm_);
horovod/common/ops/adasum_gpu_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCE,
horovod/common/ops/adasum_gpu_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/adasum_gpu_operations.cc:    // cudaHostAlloc is significantly slower than malloc.  Pre-allocating
horovod/common/ops/adasum_gpu_operations.cc:      gpu_context_->WaitForEventsElastic(gpu_op_context_.event_queue, entries,
horovod/common/ops/adasum_gpu_operations.cc:      gpu_context_->WaitForEvents(gpu_op_context_.event_queue, entries,
horovod/common/ops/adasum_gpu_operations.cc:    // According to https://docs.nvidia.com/cuda/cuda-runtime-api/
horovod/common/ops/adasum_gpu_operations.cc:    // cudaMemcpyAsync is synchronous with respect to the host, so we
horovod/common/ops/adasum_gpu_operations.cc:    gpu_context_->MemcpyAsyncD2H(host_buffer, buffer_data_at_rank_offset,
horovod/common/ops/adasum_gpu_operations.cc:                                 total_buffer_len, *gpu_op_context_.stream);
horovod/common/ops/adasum_gpu_operations.cc:    gpu_context_->MemcpyAsyncH2D(buffer_data_at_rank_offset, host_buffer,
horovod/common/ops/adasum_gpu_operations.cc:                                 total_buffer_len, *gpu_op_context_.stream);
horovod/common/ops/adasum_gpu_operations.cc:    nccl_context_->ErrorCheck(
horovod/common/ops/adasum_gpu_operations.cc:        "ncclAllGather",
horovod/common/ops/adasum_gpu_operations.cc:        ncclAllGather(buffer_data_at_rank_offset, buffer_data,
horovod/common/ops/adasum_gpu_operations.cc:                      GetNCCLDataType(first_entry.tensor),
horovod/common/ops/adasum_gpu_operations.cc:                      *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
horovod/common/ops/adasum_gpu_operations.cc:        *nccl_op_context_.nccl_comm_);
horovod/common/ops/adasum_gpu_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_ALLGATHER,
horovod/common/ops/adasum_gpu_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/adasum_gpu_operations.cc:#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 2, 12)
horovod/common/ops/adasum_gpu_operations.cc:    nccl_context_->ErrorCheck(
horovod/common/ops/adasum_gpu_operations.cc:        "ncclBroadcast",
horovod/common/ops/adasum_gpu_operations.cc:        ncclBroadcast(buffer_data_remainder, buffer_data_remainder,
horovod/common/ops/adasum_gpu_operations.cc:                      GetNCCLDataType(first_entry.tensor), root_rank,
horovod/common/ops/adasum_gpu_operations.cc:                      *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
horovod/common/ops/adasum_gpu_operations.cc:        *nccl_op_context_.nccl_comm_);
horovod/common/ops/adasum_gpu_operations.cc:    nccl_context_->ErrorCheck(
horovod/common/ops/adasum_gpu_operations.cc:        "ncclBcast",
horovod/common/ops/adasum_gpu_operations.cc:        ncclBcast(buffer_data_remainder, (size_t)num_elements_remaining,
horovod/common/ops/adasum_gpu_operations.cc:                  GetNCCLDataType(first_entry.tensor), root_rank,
horovod/common/ops/adasum_gpu_operations.cc:                  *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
horovod/common/ops/adasum_gpu_operations.cc:        *nccl_op_context_.nccl_comm_);
horovod/common/ops/adasum_gpu_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_BCAST,
horovod/common/ops/adasum_gpu_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/adasum_gpu_operations.cc:      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
horovod/common/ops/adasum_gpu_operations.cc:                                *gpu_op_context_.stream);
horovod/common/ops/adasum_gpu_operations.cc:  return gpu_op_context_.FinalizeGPUQueue(entries, false);
horovod/common/ops/adasum_gpu_operations.cc:bool AdasumGpuAllreduceOp::Enabled(const ParameterManager& param_manager,
horovod/common/ops/gpu_operations.cc:#include "gpu_operations.h"
horovod/common/ops/gpu_operations.cc:#if HAVE_CUDA
horovod/common/ops/gpu_operations.cc:#include "cuda/cuda_kernels.h"
horovod/common/ops/gpu_operations.cc:#if HAVE_ROCM
horovod/common/ops/gpu_operations.cc:#include "rocm/hip_kernels.h"
horovod/common/ops/gpu_operations.cc:GPUOpContext::GPUOpContext(GPUContext* context,
horovod/common/ops/gpu_operations.cc:    : gpu_context_(context), global_state_(global_state) {}
horovod/common/ops/gpu_operations.cc:void GPUOpContext::InitGPU(const std::vector<TensorTableEntry>& entries) {
horovod/common/ops/gpu_operations.cc:  gpu_context_->SetDevice(first_entry.device);
horovod/common/ops/gpu_operations.cc:  gpuStream_t& stream =
horovod/common/ops/gpu_operations.cc:      gpu_context_
horovod/common/ops/gpu_operations.cc:          ->streams[global_state_->current_nccl_stream][first_entry.device];
horovod/common/ops/gpu_operations.cc:    gpu_context_->StreamCreate(&stream);
horovod/common/ops/gpu_operations.cc:void GPUOpContext::InitGPUQueue(const std::vector<TensorTableEntry>& entries,
horovod/common/ops/gpu_operations.cc:      &gpu_context_
horovod/common/ops/gpu_operations.cc:           ->streams[global_state_->current_nccl_stream][entries[0].device];
horovod/common/ops/gpu_operations.cc:    gpu_context_->RecordEvent(event_queue, QUEUE, *stream);
horovod/common/ops/gpu_operations.cc:Status GPUOpContext::FinalizeGPUQueue(
horovod/common/ops/gpu_operations.cc:  // blocking gpuStreamSynchronize() in this thread.
horovod/common/ops/gpu_operations.cc:    gpu_context_->RecordEvent(event_queue, "", *stream);
horovod/common/ops/gpu_operations.cc:  auto& gpu_context = gpu_context_;
horovod/common/ops/gpu_operations.cc:      global_state_->current_nccl_stream);
horovod/common/ops/gpu_operations.cc:  gpu_context_->finalizer_thread_pool.execute(
horovod/common/ops/gpu_operations.cc:       evt_queue, &timeline, &gpu_context, error_check_callback, elastic,
horovod/common/ops/gpu_operations.cc:        gpu_context->SetDevice(first_entry.device);
horovod/common/ops/gpu_operations.cc:        bool gpu_evt_failed = false;
horovod/common/ops/gpu_operations.cc:        std::string gpu_evt_err_msg;
horovod/common/ops/gpu_operations.cc:              gpu_context->WaitForEventsElastic(evt_queue, entries, timeline,
horovod/common/ops/gpu_operations.cc:              gpu_evt_failed = true;
horovod/common/ops/gpu_operations.cc:              gpu_evt_err_msg = e.what();
horovod/common/ops/gpu_operations.cc:            gpu_context->WaitForEvents(evt_queue, entries, timeline,
horovod/common/ops/gpu_operations.cc:          gpu_context->ClearEvents(evt_queue, entries, timeline,
horovod/common/ops/gpu_operations.cc:          event = gpu_context->RecordEvent(current_stream);
horovod/common/ops/gpu_operations.cc:        if (gpu_evt_failed) {
horovod/common/ops/gpu_operations.cc:          status = Status::UnknownError(gpu_evt_err_msg);
horovod/common/ops/gpu_operations.cc:          gpu_context->ReleaseEvent(event);
horovod/common/ops/gpu_operations.cc:  global_state_->current_nccl_stream =
horovod/common/ops/gpu_operations.cc:      (global_state_->current_nccl_stream + 1) %
horovod/common/ops/gpu_operations.cc:      global_state_->num_nccl_streams;
horovod/common/ops/gpu_operations.cc:GPUAllreduce::GPUAllreduce(GPUContext* context,
horovod/common/ops/gpu_operations.cc:    : AllreduceOp(global_state), gpu_context_(context),
horovod/common/ops/gpu_operations.cc:      gpu_op_context_(context, global_state) {}
horovod/common/ops/gpu_operations.cc:bool GPUAllreduce::Enabled(const ParameterManager& param_manager,
horovod/common/ops/gpu_operations.cc:void GPUAllreduce::MemcpyInFusionBuffer(
horovod/common/ops/gpu_operations.cc:      global_state_->current_nccl_stream);
horovod/common/ops/gpu_operations.cc:#if HAVE_CUDA
horovod/common/ops/gpu_operations.cc:        BatchedD2DMemcpyCudaImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:#elif HAVE_ROCM
horovod/common/ops/gpu_operations.cc:        BatchedD2DMemcpyROCmImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:        // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
horovod/common/ops/gpu_operations.cc:        // cudaGetLastError());
horovod/common/ops/gpu_operations.cc:void GPUAllreduce::ScaleMemcpyInFusionBuffer(
horovod/common/ops/gpu_operations.cc:      global_state_->current_nccl_stream);
horovod/common/ops/gpu_operations.cc:#if HAVE_CUDA
horovod/common/ops/gpu_operations.cc:        BatchedScaledD2DMemcpyCudaImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:#elif HAVE_ROCM
horovod/common/ops/gpu_operations.cc:        BatchedScaledD2DMemcpyROCmImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:        // gpu_context_->ErrorCheck("BatchedScaledD2DMemcpyCudaImpl",
horovod/common/ops/gpu_operations.cc:        // cudaGetLastError());
horovod/common/ops/gpu_operations.cc:void GPUAllreduce::MemcpyEntryInFusionBuffer(
horovod/common/ops/gpu_operations.cc:  gpu_context_->MemcpyAsyncD2D(
horovod/common/ops/gpu_operations.cc:      gpu_context_
horovod/common/ops/gpu_operations.cc:          ->streams[global_state_->current_nccl_stream][first_entry.device]);
horovod/common/ops/gpu_operations.cc:void GPUAllreduce::MemcpyOutFusionBuffer(
horovod/common/ops/gpu_operations.cc:#if HAVE_CUDA
horovod/common/ops/gpu_operations.cc:        BatchedD2DMemcpyCudaImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:#elif HAVE_ROCM
horovod/common/ops/gpu_operations.cc:        BatchedD2DMemcpyROCmImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:        // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
horovod/common/ops/gpu_operations.cc:        // cudaGetLastError());
horovod/common/ops/gpu_operations.cc:void GPUAllreduce::ScaleMemcpyOutFusionBuffer(
horovod/common/ops/gpu_operations.cc:#if HAVE_CUDA
horovod/common/ops/gpu_operations.cc:        BatchedScaledD2DMemcpyCudaImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:#elif HAVE_ROCM
horovod/common/ops/gpu_operations.cc:        BatchedScaledD2DMemcpyROCmImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:        // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
horovod/common/ops/gpu_operations.cc:        // cudaGetLastError());
horovod/common/ops/gpu_operations.cc:void GPUAllreduce::MemcpyEntryOutFusionBuffer(
horovod/common/ops/gpu_operations.cc:  gpu_context_->MemcpyAsyncD2D(
horovod/common/ops/gpu_operations.cc:      gpu_context_
horovod/common/ops/gpu_operations.cc:          ->streams[global_state_->current_nccl_stream][first_entry.device]);
horovod/common/ops/gpu_operations.cc:void GPUAllreduce::ScaleBuffer(double scale_factor,
horovod/common/ops/gpu_operations.cc:  gpu_context_->ScaleBufferImpl(
horovod/common/ops/gpu_operations.cc:      gpu_context_
horovod/common/ops/gpu_operations.cc:          ->streams[global_state_->current_nccl_stream][entries[0].device]);
horovod/common/ops/gpu_operations.cc:GPUAllgather::GPUAllgather(GPUContext* context,
horovod/common/ops/gpu_operations.cc:    : AllgatherOp(global_state), gpu_context_(context),
horovod/common/ops/gpu_operations.cc:      gpu_op_context_(context, global_state) {}
horovod/common/ops/gpu_operations.cc:bool GPUAllgather::Enabled(const ParameterManager& param_manager,
horovod/common/ops/gpu_operations.cc:void GPUAllgather::MemcpyInFusionBuffer(
horovod/common/ops/gpu_operations.cc:      global_state_->current_nccl_stream);
horovod/common/ops/gpu_operations.cc:#if HAVE_CUDA
horovod/common/ops/gpu_operations.cc:        BatchedD2DMemcpyCudaImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:#elif HAVE_ROCM
horovod/common/ops/gpu_operations.cc:        BatchedD2DMemcpyROCmImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:        // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
horovod/common/ops/gpu_operations.cc:        // cudaGetLastError());
horovod/common/ops/gpu_operations.cc:void GPUAllgather::MemcpyEntryInFusionBuffer(
horovod/common/ops/gpu_operations.cc:  gpu_context_->MemcpyAsyncD2D(
horovod/common/ops/gpu_operations.cc:      gpu_context_
horovod/common/ops/gpu_operations.cc:          ->streams[global_state_->current_nccl_stream][first_entry.device]);
horovod/common/ops/gpu_operations.cc:void GPUAllgather::MemcpyOutFusionBuffer(
horovod/common/ops/gpu_operations.cc:#if HAVE_CUDA
horovod/common/ops/gpu_operations.cc:          BatchedD2DMemcpyCudaImpl(
horovod/common/ops/gpu_operations.cc:              gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:#elif HAVE_ROCM
horovod/common/ops/gpu_operations.cc:          BatchedD2DMemcpyROCmImpl(
horovod/common/ops/gpu_operations.cc:              gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:          // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
horovod/common/ops/gpu_operations.cc:          // cudaGetLastError());
horovod/common/ops/gpu_operations.cc:#if HAVE_CUDA
horovod/common/ops/gpu_operations.cc:      BatchedD2DMemcpyCudaImpl(
horovod/common/ops/gpu_operations.cc:          gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:#elif HAVE_ROCM
horovod/common/ops/gpu_operations.cc:      BatchedD2DMemcpyROCmImpl(
horovod/common/ops/gpu_operations.cc:          gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:      // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
horovod/common/ops/gpu_operations.cc:      // cudaGetLastError());
horovod/common/ops/gpu_operations.cc:void GPUAllgather::MemcpyEntryOutFusionBuffer(
horovod/common/ops/gpu_operations.cc:  gpu_context_->MemcpyAsyncD2D(
horovod/common/ops/gpu_operations.cc:      gpu_context_
horovod/common/ops/gpu_operations.cc:          ->streams[global_state_->current_nccl_stream][first_entry.device]);
horovod/common/ops/gpu_operations.cc:GPUBroadcast::GPUBroadcast(GPUContext* context,
horovod/common/ops/gpu_operations.cc:    : BroadcastOp(global_state), gpu_context_(context),
horovod/common/ops/gpu_operations.cc:      gpu_op_context_(context, global_state) {}
horovod/common/ops/gpu_operations.cc:bool GPUBroadcast::Enabled(const ParameterManager& param_manager,
horovod/common/ops/gpu_operations.cc:GPUAlltoall::GPUAlltoall(GPUContext* context, HorovodGlobalState* global_state)
horovod/common/ops/gpu_operations.cc:    : AlltoallOp(global_state), gpu_context_(context),
horovod/common/ops/gpu_operations.cc:      gpu_op_context_(context, global_state) {}
horovod/common/ops/gpu_operations.cc:bool GPUAlltoall::Enabled(const ParameterManager& param_manager,
horovod/common/ops/gpu_operations.cc:GPUReducescatter::GPUReducescatter(GPUContext* context,
horovod/common/ops/gpu_operations.cc:    : ReducescatterOp(global_state), gpu_context_(context),
horovod/common/ops/gpu_operations.cc:      gpu_op_context_(context, global_state) {}
horovod/common/ops/gpu_operations.cc:bool GPUReducescatter::Enabled(const ParameterManager& param_manager,
horovod/common/ops/gpu_operations.cc:void GPUReducescatter::MemcpyEntryInFusionBuffer(const TensorTableEntry& e,
horovod/common/ops/gpu_operations.cc:  gpu_context_->MemcpyAsyncD2D(
horovod/common/ops/gpu_operations.cc:      gpu_context_->streams[global_state_->current_nccl_stream][e.device]);
horovod/common/ops/gpu_operations.cc:void GPUReducescatter::MemcpyEntryOutFusionBuffer(
horovod/common/ops/gpu_operations.cc:  gpu_context_->MemcpyAsyncD2D(
horovod/common/ops/gpu_operations.cc:      gpu_context_->streams[global_state_->current_nccl_stream][e.device]);
horovod/common/ops/gpu_operations.cc:void GPUReducescatter::MemcpyInFusionBuffer(
horovod/common/ops/gpu_operations.cc:        global_state_->current_nccl_stream);
horovod/common/ops/gpu_operations.cc:#if HAVE_CUDA
horovod/common/ops/gpu_operations.cc:          BatchedD2DMemcpyCudaImpl(
horovod/common/ops/gpu_operations.cc:              gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:#elif HAVE_ROCM
horovod/common/ops/gpu_operations.cc:          BatchedD2DMemcpyROCmImpl(
horovod/common/ops/gpu_operations.cc:              gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:          // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
horovod/common/ops/gpu_operations.cc:          // cudaGetLastError());
horovod/common/ops/gpu_operations.cc:    // default implementation using GPUReducescatter::MemcpyEntryInFusionBuffer
horovod/common/ops/gpu_operations.cc:void GPUReducescatter::ScaleMemcpyInFusionBuffer(
horovod/common/ops/gpu_operations.cc:        global_state_->current_nccl_stream);
horovod/common/ops/gpu_operations.cc:#if HAVE_CUDA
horovod/common/ops/gpu_operations.cc:          BatchedScaledD2DMemcpyCudaImpl(
horovod/common/ops/gpu_operations.cc:              gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:#elif HAVE_ROCM
horovod/common/ops/gpu_operations.cc:          BatchedScaledD2DMemcpyROCmImpl(
horovod/common/ops/gpu_operations.cc:              gpu_context_->streams[global_state_->current_nccl_stream]
horovod/common/ops/gpu_operations.cc:          // gpu_context_->ErrorCheck("BatchedScaledD2DMemcpyCudaImpl",
horovod/common/ops/gpu_operations.cc:          // cudaGetLastError());
horovod/common/ops/gpu_operations.cc:    // default implementation using GPUReducescatter::MemcpyEntryInFusionBuffer
horovod/common/ops/gpu_operations.cc:void GPUReducescatter::MemcpyOutFusionBuffer(
horovod/common/ops/gpu_operations.cc:#if HAVE_CUDA
horovod/common/ops/gpu_operations.cc:        BatchedD2DMemcpyCudaImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream][device]);
horovod/common/ops/gpu_operations.cc:#elif HAVE_ROCM
horovod/common/ops/gpu_operations.cc:        BatchedD2DMemcpyROCmImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream][device]);
horovod/common/ops/gpu_operations.cc:        // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
horovod/common/ops/gpu_operations.cc:        // cudaGetLastError());
horovod/common/ops/gpu_operations.cc:    // default implementation using GPUReducescatter::MemcpyEntryOutFusionBuffer
horovod/common/ops/gpu_operations.cc:void GPUReducescatter::ScaleMemcpyOutFusionBuffer(
horovod/common/ops/gpu_operations.cc:#if HAVE_CUDA
horovod/common/ops/gpu_operations.cc:        BatchedScaledD2DMemcpyCudaImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream][device]);
horovod/common/ops/gpu_operations.cc:#elif HAVE_ROCM
horovod/common/ops/gpu_operations.cc:        BatchedScaledD2DMemcpyROCmImpl(
horovod/common/ops/gpu_operations.cc:            gpu_context_->streams[global_state_->current_nccl_stream][device]);
horovod/common/ops/gpu_operations.cc:        // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl", cudaGetLastError());
horovod/common/ops/gpu_operations.cc:    // default implementation using GPUReducescatter::MemcpyEntryOutFusionBuffer
horovod/common/ops/gpu_operations.cc:void GPUReducescatter::ScaleBuffer(double scale_factor,
horovod/common/ops/gpu_operations.cc:  gpu_context_->ScaleBufferImpl(
horovod/common/ops/gpu_operations.cc:      gpu_context_
horovod/common/ops/gpu_operations.cc:          ->streams[global_state_->current_nccl_stream][entries[0].device]);
horovod/common/ops/cuda_operations.cc:#include "cuda/cuda_kernels.h"
horovod/common/ops/cuda_operations.cc:#include "gpu_operations.h"
horovod/common/ops/cuda_operations.cc:class GPUContext::impl {
horovod/common/ops/cuda_operations.cc:  cudaError_t GetGpuEvent(Event* event, cudaStream_t stream) {
horovod/common/ops/cuda_operations.cc:    auto status = cudaGetDevice(&device);
horovod/common/ops/cuda_operations.cc:    if (status != cudaSuccess) {
horovod/common/ops/cuda_operations.cc:    auto& mutex = cuda_events_mutex;
horovod/common/ops/cuda_operations.cc:      auto& queue = cuda_events[key];
horovod/common/ops/cuda_operations.cc:        for (int i = 0; i < N_CUDA_EVENTS_PREPOPULATE; ++i) {
horovod/common/ops/cuda_operations.cc:          cudaEvent_t ev;
horovod/common/ops/cuda_operations.cc:          status = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
horovod/common/ops/cuda_operations.cc:          queue.emplace(std::make_shared<cudaEvent_t>(ev), stream);
horovod/common/ops/cuda_operations.cc:        event->event_idx = ++cuda_event_idx[key];
horovod/common/ops/cuda_operations.cc:        return cudaSuccess;
horovod/common/ops/cuda_operations.cc:    cudaEvent_t ev;
horovod/common/ops/cuda_operations.cc:    status = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
horovod/common/ops/cuda_operations.cc:    event->event = std::make_shared<cudaEvent_t>(ev);
horovod/common/ops/cuda_operations.cc:    event->event_idx = ++cuda_event_idx[key2];
horovod/common/ops/cuda_operations.cc:  cudaError_t ReleaseGpuEvent(Event event) {
horovod/common/ops/cuda_operations.cc:    auto status = cudaGetDevice(&device);
horovod/common/ops/cuda_operations.cc:    if (status != cudaSuccess) {
horovod/common/ops/cuda_operations.cc:    auto& mutex = cuda_events_mutex;
horovod/common/ops/cuda_operations.cc:      auto& queue = cuda_events[std::make_pair(device, event.stream)];
horovod/common/ops/cuda_operations.cc:    return cudaSuccess;
horovod/common/ops/cuda_operations.cc:  void ErrorCheck(std::string op_name, cudaError_t cuda_result) {
horovod/common/ops/cuda_operations.cc:    if (cuda_result != cudaSuccess) {
horovod/common/ops/cuda_operations.cc:                             " failed: " + cudaGetErrorString(cuda_result));
horovod/common/ops/cuda_operations.cc:                   std::string name, cudaStream_t& stream) {
horovod/common/ops/cuda_operations.cc:    ErrorCheck("GetGpuEvent", GetGpuEvent(&event, stream));
horovod/common/ops/cuda_operations.cc:    ErrorCheck("cudaEventRecord",
horovod/common/ops/cuda_operations.cc:               cudaEventRecord(*(event.event), event.stream));
horovod/common/ops/cuda_operations.cc:  Event RecordEvent(cudaStream_t& stream) {
horovod/common/ops/cuda_operations.cc:    ErrorCheck("GetGpuEvent", GetGpuEvent(&event, stream));
horovod/common/ops/cuda_operations.cc:    ErrorCheck("cudaEventRecord",
horovod/common/ops/cuda_operations.cc:               cudaEventRecord(*(event.event), event.stream));
horovod/common/ops/cuda_operations.cc:      cudaError_t cuda_result = cudaEventSynchronize(*(event.event));
horovod/common/ops/cuda_operations.cc:      if (cuda_result != cudaSuccess) {
horovod/common/ops/cuda_operations.cc:        throw std::logic_error(std::string("cudaEventSynchronize failed: ") +
horovod/common/ops/cuda_operations.cc:                               cudaGetErrorString(cuda_result));
horovod/common/ops/cuda_operations.cc:      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
horovod/common/ops/cuda_operations.cc:      cudaError_t cuda_result;
horovod/common/ops/cuda_operations.cc:        cuda_result = cudaEventQuery(*(event.event));
horovod/common/ops/cuda_operations.cc:        if (cuda_result == cudaSuccess) {
horovod/common/ops/cuda_operations.cc:        if (cuda_result != cudaErrorNotReady) {
horovod/common/ops/cuda_operations.cc:          throw std::logic_error(std::string("cudaEventQuery failed: ") +
horovod/common/ops/cuda_operations.cc:                                 cudaGetErrorString(cuda_result));
horovod/common/ops/cuda_operations.cc:      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
horovod/common/ops/cuda_operations.cc:      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
horovod/common/ops/cuda_operations.cc:  void StreamCreate(cudaStream_t* stream) {
horovod/common/ops/cuda_operations.cc:    ErrorCheck("cudaDeviceGetStreamPriorityRange",
horovod/common/ops/cuda_operations.cc:               cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
horovod/common/ops/cuda_operations.cc:    ErrorCheck("cudaStreamCreateWithPriority",
horovod/common/ops/cuda_operations.cc:               cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking,
horovod/common/ops/cuda_operations.cc:  void StreamSynchronize(cudaStream_t stream) {
horovod/common/ops/cuda_operations.cc:    ErrorCheck("cudaStreamSynchronize", cudaStreamSynchronize(stream));
horovod/common/ops/cuda_operations.cc:    ErrorCheck("cudaGetDevice", cudaGetDevice(&device));
horovod/common/ops/cuda_operations.cc:    ErrorCheck("cudaSetDevice", cudaSetDevice(device));
horovod/common/ops/cuda_operations.cc:                      cudaStream_t stream) {
horovod/common/ops/cuda_operations.cc:        "cudaMemcpyAsync",
horovod/common/ops/cuda_operations.cc:        cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream));
horovod/common/ops/cuda_operations.cc:                      cudaStream_t stream) {
horovod/common/ops/cuda_operations.cc:        "cudaMemcpyAsync",
horovod/common/ops/cuda_operations.cc:        cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
horovod/common/ops/cuda_operations.cc:                      cudaStream_t stream) {
horovod/common/ops/cuda_operations.cc:        "cudaMemcpyAsync",
horovod/common/ops/cuda_operations.cc:        cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
horovod/common/ops/cuda_operations.cc:                       DataType dtype, cudaStream_t stream) {
horovod/common/ops/cuda_operations.cc:    ScaleBufferCudaImpl(fused_input_data, buffer_data, num_elements,
horovod/common/ops/cuda_operations.cc:    // ErrorCheck("ScaleBufferCudaImpl", cudaGetLastError());
horovod/common/ops/cuda_operations.cc:  // We reuse CUDA events as it appears that their creation carries non-zero
horovod/common/ops/cuda_operations.cc:  std::unordered_map<std::pair<int, cudaStream_t>, std::queue<Event>>
horovod/common/ops/cuda_operations.cc:      cuda_events;
horovod/common/ops/cuda_operations.cc:  std::unordered_map<std::pair<int, cudaStream_t>, bool> prepopulated;
horovod/common/ops/cuda_operations.cc:  std::unordered_map<std::pair<int, cudaStream_t>, std::atomic<uint64_t>> cuda_event_idx;
horovod/common/ops/cuda_operations.cc:  std::mutex cuda_events_mutex;
horovod/common/ops/cuda_operations.cc:  static constexpr int N_CUDA_EVENTS_PREPOPULATE = 128;
horovod/common/ops/cuda_operations.cc:#include "gpu_context_impl.cc"
horovod/common/ops/mpi_operations.h:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/cuda/CMakeLists.txt:# If we don't set CMAKE_CUDA_STANDARD, it will default to ${CMAKE_CXX_STANDARD} ("14" at this time). nvcc may fail if
horovod/common/ops/cuda/CMakeLists.txt:set(CMAKE_CUDA_STANDARD 11)
horovod/common/ops/cuda/CMakeLists.txt:set(CMAKE_CUDA_STANDARD_REQUIRED ON)
horovod/common/ops/cuda/CMakeLists.txt:add_library(horovod_cuda_kernels cuda_kernels.cu)
horovod/common/ops/cuda/CMakeLists.txt:target_compile_options(horovod_cuda_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
horovod/common/ops/cuda/CMakeLists.txt:add_library(compatible_horovod_cuda_kernels cuda_kernels.cu)
horovod/common/ops/cuda/CMakeLists.txt:target_compile_options(compatible_horovod_cuda_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
horovod/common/ops/cuda/cuda_kernels.cu:// Copyright (C) 2020 NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/cuda/cuda_kernels.cu:// ATTENTION: Any change here might obsolete hip_kernels.cu in rocm folder.
horovod/common/ops/cuda/cuda_kernels.cu:#include "cuda_kernels.h"
horovod/common/ops/cuda/cuda_kernels.cu:#include <cuda_fp16.h>
horovod/common/ops/cuda/cuda_kernels.cu:void BatchedD2DMemcpyCudaImpl(BatchedD2DParams& params, int num_copies, cudaStream_t stream)
horovod/common/ops/cuda/cuda_kernels.cu:#if __CUDA_ARCH__ > 530
horovod/common/ops/cuda/cuda_kernels.cu:#if __CUDA_ARCH__ > 530
horovod/common/ops/cuda/cuda_kernels.cu:void ScaleBufferCudaImpl(const void* fused_input_data, void* buffer_data, const int64_t num_elements, double scale_factor,
horovod/common/ops/cuda/cuda_kernels.cu:                         DataType dtype, cudaStream_t stream) {
horovod/common/ops/cuda/cuda_kernels.cu:                             " not supported by ScaleBufferCudaImpl.");
horovod/common/ops/cuda/cuda_kernels.cu:#if __CUDA_ARCH__ > 530
horovod/common/ops/cuda/cuda_kernels.cu:#if __CUDA_ARCH__ > 530
horovod/common/ops/cuda/cuda_kernels.cu:void BatchedScaledD2DMemcpyCudaImpl(BatchedD2DParams& params, int num_copies, double scale_factor,
horovod/common/ops/cuda/cuda_kernels.cu:                                    DataType dtype, cudaStream_t stream) {
horovod/common/ops/cuda/cuda_kernels.cu:                            " not supported by BatchedScaledD2DMemcpyCudaImpl.");
horovod/common/ops/cuda/cuda_kernels.h:// Copyright (C) 2020 NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/cuda/cuda_kernels.h:// ATTENTION: Any change here might obsolete hip_kernels.h in rocm folder.
horovod/common/ops/cuda/cuda_kernels.h:#ifndef CUDA_KERNELS_H
horovod/common/ops/cuda/cuda_kernels.h:#define CUDA_KERNELS_H
horovod/common/ops/cuda/cuda_kernels.h:#include <cuda_runtime.h>
horovod/common/ops/cuda/cuda_kernels.h:void BatchedD2DMemcpyCudaImpl(BatchedD2DParams& params, int num_copies, cudaStream_t stream);
horovod/common/ops/cuda/cuda_kernels.h:void ScaleBufferCudaImpl(const void* fused_input_data, void* buffer_data, const int64_t num_elements,
horovod/common/ops/cuda/cuda_kernels.h:                         double scale_factor, DataType dtype, cudaStream_t stream);
horovod/common/ops/cuda/cuda_kernels.h:void BatchedScaledD2DMemcpyCudaImpl(BatchedD2DParams& params, int num_copies, double scale_factor,
horovod/common/ops/cuda/cuda_kernels.h:                                    DataType dtype, cudaStream_t stream);
horovod/common/ops/cuda/cuda_kernels.h:#endif // CUDA_KERNELS_H
horovod/common/ops/ddl_operations.h:#include "gpu_operations.h"
horovod/common/ops/ddl_operations.h:class DDLAllreduce : public GPUAllreduce {
horovod/common/ops/ddl_operations.h:               GPUContext* gpu_context,
horovod/common/ops/ddl_operations.h:  static void DDLInit(DDLContext* ddl_context, GPUContext* gpu_context);
horovod/common/ops/mpi_gpu_operations.h:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/ops/mpi_gpu_operations.h:#ifndef HOROVOD_MPI_GPU_OPERATIONS_H
horovod/common/ops/mpi_gpu_operations.h:#define HOROVOD_MPI_GPU_OPERATIONS_H
horovod/common/ops/mpi_gpu_operations.h:#include "gpu_operations.h"
horovod/common/ops/mpi_gpu_operations.h:class MPI_GPUAllreduce : public GPUAllreduce {
horovod/common/ops/mpi_gpu_operations.h:  MPI_GPUAllreduce(GPUContext* gpu_context, HorovodGlobalState* global_state);
horovod/common/ops/mpi_gpu_operations.h:  ~MPI_GPUAllreduce() override = default;
horovod/common/ops/mpi_gpu_operations.h:class MPI_GPUAllgather : public GPUAllgather {
horovod/common/ops/mpi_gpu_operations.h:  MPI_GPUAllgather(GPUContext* gpu_context, HorovodGlobalState* global_state);
horovod/common/ops/mpi_gpu_operations.h:  ~MPI_GPUAllgather() override = default;
horovod/common/ops/mpi_gpu_operations.h:// TODO: Add MPI_GPUBroadcast implementation
horovod/common/ops/mpi_gpu_operations.h:class MPI_GPUAlltoall : public GPUAlltoall {
horovod/common/ops/mpi_gpu_operations.h:  MPI_GPUAlltoall(GPUContext* gpu_context, HorovodGlobalState* global_state);
horovod/common/ops/mpi_gpu_operations.h:  ~MPI_GPUAlltoall() override = default;
horovod/common/ops/mpi_gpu_operations.h:class MPI_GPUReducescatter : public GPUReducescatter {
horovod/common/ops/mpi_gpu_operations.h:  MPI_GPUReducescatter(GPUContext* cuda_context,
horovod/common/ops/mpi_gpu_operations.h:  ~MPI_GPUReducescatter() override = default;
horovod/common/ops/mpi_gpu_operations.h:#endif //HOROVOD_MPI_GPU_OPERATIONS_H
horovod/common/common.h:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/common.h:#if HAVE_GPU
horovod/common/common.h:#if HAVE_CUDA
horovod/common/common.h:#include <cuda_runtime.h>
horovod/common/common.h:using gpuError_t = cudaError_t;
horovod/common/common.h:using gpuEvent_t = cudaEvent_t;
horovod/common/common.h:using gpuStream_t = cudaStream_t;
horovod/common/common.h:using gpuPointerAttribute_t = cudaPointerAttributes;
horovod/common/common.h:#define gpuEventCreateWithFlags cudaEventCreateWithFlags
horovod/common/common.h:#define gpuEventDisableTiming cudaEventDisableTiming
horovod/common/common.h:#define gpuEventRecord cudaEventRecord
horovod/common/common.h:#define gpuEventQuery cudaEventQuery
horovod/common/common.h:#define gpuErrorNotReady cudaErrorNotReady
horovod/common/common.h:#define gpuEventSynchronize cudaEventSynchronize
horovod/common/common.h:#define gpuStreamWaitEvent cudaStreamWaitEvent
horovod/common/common.h:#define HVD_GPU_CHECK(x)                                                                    \
horovod/common/common.h:    cudaError_t cuda_result = x;                                                            \
horovod/common/common.h:    if (cuda_result != cudaSuccess) {                                                       \
horovod/common/common.h:      throw std::logic_error(std::string("GPU Error:") + cudaGetErrorString(cuda_result));  \
horovod/common/common.h:#elif HAVE_ROCM
horovod/common/common.h:using gpuError_t = hipError_t;
horovod/common/common.h:using gpuEvent_t = hipEvent_t;
horovod/common/common.h:using gpuStream_t = hipStream_t;
horovod/common/common.h:using gpuPointerAttribute_t = hipPointerAttribute_t;
horovod/common/common.h:#define gpuEventCreateWithFlags hipEventCreateWithFlags
horovod/common/common.h:#define gpuEventDisableTiming hipEventDisableTiming
horovod/common/common.h:#define gpuEventRecord hipEventRecord
horovod/common/common.h:#define gpuEventQuery hipEventQuery
horovod/common/common.h:#define gpuErrorNotReady hipErrorNotReady
horovod/common/common.h:#define gpuEventSynchronize hipEventSynchronize
horovod/common/common.h:#define gpuStreamWaitEvent hipStreamWaitEvent
horovod/common/common.h:#define HVD_GPU_CHECK(x)                                                                  \
horovod/common/common.h:      throw std::logic_error(std::string("GPU Error:") + hipGetErrorString(hip_result));  \
horovod/common/common.h:#define INIT_NCCL "INIT_NCCL"
horovod/common/common.h:#define NCCL_ALLREDUCE "NCCL_ALLREDUCE"
horovod/common/common.h:#define NCCL_REDUCESCATTER "NCCL_REDUCESCATTER"
horovod/common/common.h:#define NCCL_ALLGATHER "NCCL_ALLGATHER"
horovod/common/common.h:#define NCCL_REDUCE "NCCL_REDUCE"
horovod/common/common.h:#define NCCL_BCAST "NCCL_BCAST"
horovod/common/common.h:#define NCCL_ALLTOALL "NCCL_ALLTOALL"
horovod/common/common.h:#define HOROVOD_NUM_NCCL_STREAMS "HOROVOD_NUM_NCCL_STREAMS"
horovod/common/common.h:enum DeviceType { CPU, GPU };
horovod/common/common.h:#if HAVE_GPU
horovod/common/common.h:  Event(std::shared_ptr<gpuEvent_t> event, gpuStream_t stream) :
horovod/common/common.h:  std::shared_ptr<gpuEvent_t> event;
horovod/common/common.h:  gpuStream_t stream = nullptr;
horovod/common/common.h:#if HAVE_GPU
horovod/common/common.h:  virtual gpuEvent_t event() const = 0;
horovod/common/common.h:#if HAVE_GPU
horovod/common/common.h:  void PushEventsToSet(std::unordered_set<gpuEvent_t>& event_set) {
horovod/common/common.h:  // GPU to do reduction on, or CPU_DEVICE_ID in case of CPU.
horovod/common/controller.h:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/thread_pool.h:// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
horovod/common/group_table.h:// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/global_state.h:  // Number of GPU streams to use
horovod/common/global_state.h:  int num_nccl_streams = 1;
horovod/common/global_state.h:  // Index of current GPU stream to use
horovod/common/global_state.h:  int current_nccl_stream = 0;
horovod/common/global_state.h:  // Enable use of batched d2d memcopy kernel on GPU
horovod/common/controller.cc:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/controller.cc:#if HAVE_CUDA
horovod/common/controller.cc:#include "ops/cuda/cuda_kernels.h"
horovod/common/controller.cc:#elif HAVE_ROCM
horovod/common/controller.cc:#include "ops/rocm/hip_kernels.h"
horovod/common/controller.cc:          << " CPU/GPU device selection: One rank specified device "
horovod/common/controller.cc:          << (first_device_is_cpu ? "CPU" : "GPU")
horovod/common/controller.cc:          << (this_device_is_cpu ? "CPU" : "GPU") << ".";
horovod/common/controller.cc:#if HAVE_CUDA || HAVE_ROCM
horovod/common/controller.cc:#endif // HAVE_CUDA || HAVE_ROCM
horovod/common/controller.cc:#if HAVE_CUDA || HAVE_ROCM
horovod/common/controller.cc:#endif // HAVE_CUDA || HAVE_ROCM
horovod/common/controller.cc:#if HAVE_CUDA || HAVE_ROCM
horovod/common/operations.h:// C interface to return integer indicating whether Horovod was compiled with NCCL support.
horovod/common/operations.h:// Returns NCCL_VERSION_CODE if NCCL is available, else returns 0.
horovod/common/operations.h:int horovod_nccl_built();
horovod/common/operations.h:// C interface to return flag indicating whether Horovod was compiled with CUDA support.
horovod/common/operations.h:bool horovod_cuda_built();
horovod/common/operations.h:// C interface to return flag indicating whether Horovod was compiled with ROCm support.
horovod/common/operations.h:bool horovod_rocm_built();
horovod/common/operations.cc:// Modifications copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
horovod/common/operations.cc:#if HAVE_GPU
horovod/common/operations.cc:#include "ops/gpu_operations.h"
horovod/common/operations.cc:#include "ops/mpi_gpu_operations.h"
horovod/common/operations.cc:#if HAVE_NCCL
horovod/common/operations.cc:#include "ops/nccl_operations.h"
horovod/common/operations.cc:#include "ops/adasum_gpu_operations.h"
horovod/common/operations.cc: * support in MPI, NCCL, CUDA/ROCm, Gloo, oneCCL, DDL. The background thread
horovod/common/operations.cc:#if HAVE_GPU
horovod/common/operations.cc:GPUContext gpu_context;
horovod/common/operations.cc:#if HAVE_NCCL
horovod/common/operations.cc:NCCLContext nccl_context;
horovod/common/operations.cc:NCCLContext local_nccl_context;
horovod/common/operations.cc:NCCLContext cross_nccl_context;
horovod/common/operations.cc:#if HAVE_MPI && HAVE_GPU
horovod/common/operations.cc:#if HOROVOD_GPU_ALLREDUCE == 'M'
horovod/common/operations.cc:        new MPI_GPUAllreduce(&gpu_context, &state)));
horovod/common/operations.cc:#elif HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
horovod/common/operations.cc:    adasum_ops.push_back(std::shared_ptr<AllreduceOp>(new AdasumGpuAllreduceOp(
horovod/common/operations.cc:        &global_mpi_context, &nccl_context, &gpu_context, &state)));
horovod/common/operations.cc:        new NCCLHierarchicalAllreduce(&nccl_context, &gpu_context, &state)));
horovod/common/operations.cc:#elif HAVE_DDL && HOROVOD_GPU_ALLREDUCE == 'D'
horovod/common/operations.cc:        new DDLAllreduce(&ddl_context, &gpu_context, &state)));
horovod/common/operations.cc:#if HOROVOD_GPU_ALLGATHER == 'M'
horovod/common/operations.cc:        new MPI_GPUAllgather(&gpu_context, &state)));
horovod/common/operations.cc:#if HOROVOD_GPU_ALLTOALL == 'M'
horovod/common/operations.cc:        std::shared_ptr<AlltoallOp>(new MPI_GPUAlltoall(&gpu_context, &state)));
horovod/common/operations.cc:#if HOROVOD_GPU_REDUCESCATTER == 'M'
horovod/common/operations.cc:        new MPI_GPUReducescatter(&gpu_context, &state)));
horovod/common/operations.cc:#if HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
horovod/common/operations.cc:    new NCCLTorusAllreduce(&local_nccl_context, &cross_nccl_context, &gpu_context, &state)));
horovod/common/operations.cc:      new NCCLAllreduce(&nccl_context, &gpu_context, &state)));
horovod/common/operations.cc:#if HAVE_NCCL && HOROVOD_GPU_BROADCAST == 'N'
horovod/common/operations.cc:      new NCCLBroadcast(&nccl_context, &gpu_context, &state)));
horovod/common/operations.cc:#if HAVE_NCCL && HOROVOD_GPU_ALLGATHER == 'N'
horovod/common/operations.cc:      new NCCLAllgather(&nccl_context, &gpu_context, &state)));
horovod/common/operations.cc:#if HAVE_NCCL && HOROVOD_GPU_REDUCESCATTER == 'N'
horovod/common/operations.cc:        new NCCLReducescatter(&nccl_context, &gpu_context, &state)));
horovod/common/operations.cc:#if HAVE_NCCL && HOROVOD_GPU_ALLTOALL == 'N'
horovod/common/operations.cc:      new NCCLAlltoall(&nccl_context, &gpu_context, &state)));
horovod/common/operations.cc:          horovod_global.current_nccl_stream,
horovod/common/operations.cc:  auto mpi_ctx_manager = DDL_MPIContextManager(ddl_context, gpu_context);
horovod/common/operations.cc:#if HAVE_GPU
horovod/common/operations.cc:  // Set number of GPU streams to use
horovod/common/operations.cc:  auto horovod_num_nccl_streams = std::getenv(HOROVOD_NUM_NCCL_STREAMS);
horovod/common/operations.cc:  if (horovod_num_nccl_streams != nullptr &&
horovod/common/operations.cc:      std::stol(horovod_num_nccl_streams, nullptr, 10) > 0) {
horovod/common/operations.cc:    state.num_nccl_streams = std::atoi(horovod_num_nccl_streams);
horovod/common/operations.cc:#if HAVE_NCCL
horovod/common/operations.cc:  nccl_context.nccl_comms.resize(state.num_nccl_streams);
horovod/common/operations.cc:  local_nccl_context.nccl_comms.resize(state.num_nccl_streams);
horovod/common/operations.cc:  cross_nccl_context.nccl_comms.resize(state.num_nccl_streams);
horovod/common/operations.cc:  SetBoolFromEnv(HOROVOD_ELASTIC, nccl_context.elastic, true);
horovod/common/operations.cc:  SetBoolFromEnv(HOROVOD_ELASTIC, local_nccl_context.elastic, true);
horovod/common/operations.cc:  SetBoolFromEnv(HOROVOD_ELASTIC, cross_nccl_context.elastic, true);
horovod/common/operations.cc:  gpu_context.streams.resize(state.num_nccl_streams);
horovod/common/operations.cc:  gpu_context.finalizer_thread_pool.create(state.num_nccl_streams);
horovod/common/operations.cc:#if HOROVOD_GPU_ALLREDUCE != 'N' && HOROVOD_GPU_ALLREDUCE != 'D'
horovod/common/operations.cc:  // Hierarchical allreduce is not supported without NCCL or DDL
horovod/common/operations.cc:#if HOROVOD_GPU_ALLREDUCE != 'N' && HOROVOD_GPU_ALLREDUCE != 'D'
horovod/common/operations.cc:  // Torus allreduce is not supported without NCCL or DDL
horovod/common/operations.cc:  // Set flag to control use of batched memcopy kernel on GPU
horovod/common/operations.cc:#if HAVE_NCCL
horovod/common/operations.cc:  nccl_context.ShutDown();
horovod/common/operations.cc:  local_nccl_context.ShutDown();
horovod/common/operations.cc:  cross_nccl_context.ShutDown();
horovod/common/operations.cc:#if HAVE_GPU
horovod/common/operations.cc:  gpu_context.Finalize();
horovod/common/operations.cc:int horovod_nccl_built() {
horovod/common/operations.cc:#if HAVE_NCCL
horovod/common/operations.cc:  return NCCL_VERSION_CODE;
horovod/common/operations.cc:bool horovod_cuda_built() {
horovod/common/operations.cc:#if HAVE_CUDA
horovod/common/operations.cc:bool horovod_rocm_built() {
horovod/common/operations.cc:#if HAVE_ROCM
horovod/common/operations.cc:#if HAVE_NCCL && !HAVE_ROCM
horovod/common/operations.cc:#if !HAVE_ROCM
horovod/common/operations.cc:        << "Enqueuing AVERAGE reducescatter is not allowed with ROCm.";
horovod/common/message.h:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/half.h: * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
horovod/common/half.h: *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
horovod/common/half.h: * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
horovod/common/util.py:def gpu_available(ext_base_name, verbose=False):
horovod/common/util.py:    available_fn = lambda ext: ext._check_has_gpu()
horovod/common/util.py:        ext_base_name, available_fn, 'running with GPU', verbose) or False
horovod/common/util.py:def nccl_built(verbose=False):
horovod/common/util.py:        built_fn = lambda ext: ext.nccl_built()
horovod/common/util.py:            ext_base_name, built_fn, 'built with NCCL', verbose)
horovod/common/mpi/mpi_controller.h:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/mpi/mpi_controller.cc:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/common/mpi/ddl_mpi_context_manager.cc:  DDLAllreduce::DDLInit(&ddl_context_, &gpu_context_);
horovod/common/mpi/ddl_mpi_context_manager.h:#include "../ops/gpu_operations.h"
horovod/common/mpi/ddl_mpi_context_manager.h:  // Constructor, store the reference of ddl context and gpu context.
horovod/common/mpi/ddl_mpi_context_manager.h:  DDL_MPIContextManager(DDLContext& ddl_context, GPUContext& gpu_context)
horovod/common/mpi/ddl_mpi_context_manager.h:      : ddl_context_(ddl_context), gpu_context_(gpu_context){};
horovod/common/mpi/ddl_mpi_context_manager.h:  GPUContext& gpu_context_;
horovod/common/thread_pool.cc:// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
horovod/tensorflow/mpi_ops.py:# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/tensorflow/mpi_ops.py:    get_average_backwards_compatibility_fun, gpu_available, num_rank_is_power_2
horovod/tensorflow/mpi_ops.py:nccl_built = _basics.nccl_built
horovod/tensorflow/mpi_ops.py:cuda_built = _basics.cuda_built
horovod/tensorflow/mpi_ops.py:rocm_built = _basics.rocm_built
horovod/tensorflow/mpi_ops.py:def _check_has_gpu():
horovod/tensorflow/mpi_ops.py:    return tf.test.is_gpu_available()
horovod/tensorflow/mpi_ops.py:    to be located on the same device (CPU or GPU).
horovod/tensorflow/sync_batch_norm.py:# Modifications copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/tensorflow/keras/callbacks.py:            device: Device to be used for broadcasting. Uses GPU by default
horovod/tensorflow/keras/callbacks.py:                    if Horovod was build with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/keras/callbacks.py:            device: Device to be used for allreduce. Uses GPU by default
horovod/tensorflow/keras/callbacks.py:                    if Horovod was build with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/keras/__init__.py:from horovod.tensorflow import nccl_built, ddl_built, ccl_built, cuda_built, rocm_built
horovod/tensorflow/keras/__init__.py:        device_dense: Device to be used for dense tensors. Uses GPU by default
horovod/tensorflow/keras/__init__.py:                      if Horovod was build with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/keras/__init__.py:        device_sparse: Device to be used for sparse tensors. Uses GPU by default
horovod/tensorflow/keras/__init__.py:                       if Horovod was build with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/keras/__init__.py:    if gradient_predivide_factor != 1.0 and rocm_built():
horovod/tensorflow/keras/__init__.py:            raise ValueError('gradient_predivide_factor not supported yet with ROCm')
horovod/tensorflow/CMakeLists.txt:if(HAVE_CUDA)
horovod/tensorflow/CMakeLists.txt:        list(APPEND TF_LINKER_LIBS horovod_cuda_kernels)
horovod/tensorflow/CMakeLists.txt:        list(APPEND TF_LINKER_LIBS compatible_horovod_cuda_kernels)
horovod/tensorflow/CMakeLists.txt:if(HAVE_ROCM)
horovod/tensorflow/CMakeLists.txt:        list(APPEND TF_LINKER_LIBS horovod_cuda_kernels)
horovod/tensorflow/CMakeLists.txt:        list(APPEND TF_LINKER_LIBS compatible_horovod_cuda_kernels)
horovod/tensorflow/xla_mpi_ops.cc:// Modifications copyright (C) 2021, NVIDIA CORPORATION. All rights reserved.
horovod/tensorflow/xla_mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/xla_mpi_ops.cc:#if HAVE_CUDA
horovod/tensorflow/xla_mpi_ops.cc:#include <cuda.h>
horovod/tensorflow/xla_mpi_ops.cc:#include <cuda_runtime.h>
horovod/tensorflow/xla_mpi_ops.cc:#define CUDA_CALL(func)                                                        \
horovod/tensorflow/xla_mpi_ops.cc:    cudaError_t e = (func);                                                    \
horovod/tensorflow/xla_mpi_ops.cc:    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                   \
horovod/tensorflow/xla_mpi_ops.cc:        << "CUDA: " << cudaGetErrorString(e);                                  \
horovod/tensorflow/xla_mpi_ops.cc:#elif HAVE_ROCM
horovod/tensorflow/xla_mpi_ops.cc:#endif // HAVE_CUDA
horovod/tensorflow/xla_mpi_ops.cc:    std::shared_ptr<gpuEvent_t> event;
horovod/tensorflow/xla_mpi_ops.cc:  void Wait(string tensor_name, gpuStream_t stream) {
horovod/tensorflow/xla_mpi_ops.cc:    std::shared_ptr<gpuEvent_t> event = payload->event;
horovod/tensorflow/xla_mpi_ops.cc:#if HAVE_CUDA
horovod/tensorflow/xla_mpi_ops.cc:      CUDA_CALL(cudaStreamWaitEvent(stream, *event, /*flags=*/0));
horovod/tensorflow/xla_mpi_ops.cc:#elif HAVE_ROCM
horovod/tensorflow/xla_mpi_ops.cc:      HVD_GPU_CHECK(hipStreamWaitEvent(stream, *event, /*flags=*/0));
horovod/tensorflow/xla_mpi_ops.cc:  XLAReadyEvent(gpuStream_t stream) : stream_(stream) {
horovod/tensorflow/xla_mpi_ops.cc:#if HAVE_CUDA
horovod/tensorflow/xla_mpi_ops.cc:    CUDA_CALL(cudaEventCreate(&event_));
horovod/tensorflow/xla_mpi_ops.cc:    CUDA_CALL(cudaEventRecord(event_, stream));
horovod/tensorflow/xla_mpi_ops.cc:  ~XLAReadyEvent() { CUDA_CALL(cudaEventDestroy(event_)); }
horovod/tensorflow/xla_mpi_ops.cc:#elif HAVE_ROCM
horovod/tensorflow/xla_mpi_ops.cc:    HVD_GPU_CHECK(hipEventCreate(&event_));
horovod/tensorflow/xla_mpi_ops.cc:    HVD_GPU_CHECK(hipEventRecord(event_, stream));
horovod/tensorflow/xla_mpi_ops.cc:  ~XLAReadyEvent() { HVD_GPU_CHECK(hipEventDestroy(event_)); }
horovod/tensorflow/xla_mpi_ops.cc:    gpuError_t result = gpuEventQuery(event_);
horovod/tensorflow/xla_mpi_ops.cc:    return gpuErrorNotReady != result;
horovod/tensorflow/xla_mpi_ops.cc:  gpuEvent_t event() const override { return event_; }
horovod/tensorflow/xla_mpi_ops.cc:  gpuStream_t stream_; // Not Owned.
horovod/tensorflow/xla_mpi_ops.cc:  gpuEvent_t event_;   // Owned.
horovod/tensorflow/xla_mpi_ops.cc:#if HAVE_CUDA
horovod/tensorflow/xla_mpi_ops.cc:  CUDA_CALL(cudaGetDevice(&restore_device));
horovod/tensorflow/xla_mpi_ops.cc:  CUDA_CALL(cudaSetDevice(device));
horovod/tensorflow/xla_mpi_ops.cc:  // Simply call cudaMalloc for persistent buffer.
horovod/tensorflow/xla_mpi_ops.cc:  CUDA_CALL(cudaMalloc((void**)&buffer_, size));
horovod/tensorflow/xla_mpi_ops.cc:  CUDA_CALL(cudaSetDevice(restore_device));
horovod/tensorflow/xla_mpi_ops.cc:#elif HAVE_ROCM
horovod/tensorflow/xla_mpi_ops.cc:  HVD_GPU_CHECK(hipGetDevice(&restore_device));
horovod/tensorflow/xla_mpi_ops.cc:  HVD_GPU_CHECK(hipSetDevice(device));
horovod/tensorflow/xla_mpi_ops.cc:  HVD_GPU_CHECK(hipMalloc((void**)&buffer_, size));
horovod/tensorflow/xla_mpi_ops.cc:  HVD_GPU_CHECK(hipSetDevice(restore_device));
horovod/tensorflow/xla_mpi_ops.cc:common::ReadyEvent* RecordReadyEvent(gpuStream_t stream) {
horovod/tensorflow/xla_mpi_ops.cc:  gpuPointerAttribute_t attrs;
horovod/tensorflow/xla_mpi_ops.cc:#if HAVE_CUDA
horovod/tensorflow/xla_mpi_ops.cc:  CUDA_CALL(cudaPointerGetAttributes(&attrs, ptr));
horovod/tensorflow/xla_mpi_ops.cc:#elif HAVE_ROCM
horovod/tensorflow/xla_mpi_ops.cc:  HVD_GPU_CHECK(hipPointerGetAttributes(&attrs, ptr));
horovod/tensorflow/xla_mpi_ops.cc:void CallbackHVDAllreduce(gpuStream_t stream, void** buffers, const char* opaque,
horovod/tensorflow/xla_mpi_ops.cc:void CallbackHVDAllreduceDone(gpuStream_t stream, void** /*buffers*/,
horovod/tensorflow/xla_mpi_ops.cc:#if HAVE_CUDA
horovod/tensorflow/xla_mpi_ops.cc:XLA_REGISTER_CUSTOM_CALL_TARGET(CallbackHVDAllreduce, "CUDA");
horovod/tensorflow/xla_mpi_ops.cc:XLA_REGISTER_CUSTOM_CALL_TARGET(CallbackHVDAllreduceDone, "CUDA");
horovod/tensorflow/xla_mpi_ops.cc:#elif HAVE_ROCM
horovod/tensorflow/xla_mpi_ops.cc:XLA_REGISTER_CUSTOM_CALL_TARGET(CallbackHVDAllreduce, "ROCM");
horovod/tensorflow/xla_mpi_ops.cc:XLA_REGISTER_CUSTOM_CALL_TARGET(CallbackHVDAllreduceDone, "ROCM");
horovod/tensorflow/xla_mpi_ops.cc:#endif // HAVE_GPU
horovod/tensorflow/mpi_ops.cc:// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/tensorflow/mpi_ops.cc:#if HAVE_CUDA || HAVE_ROCM
horovod/tensorflow/mpi_ops.cc:#define EIGEN_USE_GPU
horovod/tensorflow/mpi_ops.cc:#endif  // HAVE_CUDA || HAVE_ROCM
horovod/tensorflow/mpi_ops.cc:#if HAVE_ROCM
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:#if HAVE_CUDA
horovod/tensorflow/mpi_ops.cc:#include <cuda_runtime.h>
horovod/tensorflow/mpi_ops.cc:using GpuStreamHandle = cudaStream_t;
horovod/tensorflow/mpi_ops.cc:#define gpuMemsetAsync cudaMemsetAsync
horovod/tensorflow/mpi_ops.cc:#elif HAVE_ROCM
horovod/tensorflow/mpi_ops.cc:using GpuStreamHandle = hipStream_t;
horovod/tensorflow/mpi_ops.cc:#define gpuMemsetAsync hipMemsetAsync
horovod/tensorflow/mpi_ops.cc:#endif // HAVE_CUDA, HAVE_ROCM
horovod/tensorflow/mpi_ops.cc:// Forward declaration of AsGpuStreamValue
horovod/tensorflow/mpi_ops.cc:namespace gpu {
horovod/tensorflow/mpi_ops.cc:GpuStreamHandle AsGpuStreamValue(Stream* stream);
horovod/tensorflow/mpi_ops.cc:} // namespace gpu
horovod/tensorflow/mpi_ops.cc:#endif // HAVE_GPU
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:  std::unordered_map<int, std::queue<gpuEvent_t>> gpu_events;
horovod/tensorflow/mpi_ops.cc:  gpuEvent_t event() const override;
horovod/tensorflow/mpi_ops.cc:  gpuEvent_t event_;
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:    auto& queue = ready_event_registry.gpu_events[device_];
horovod/tensorflow/mpi_ops.cc:      HVD_GPU_CHECK(gpuEventCreateWithFlags(&event_, gpuEventDisableTiming));
horovod/tensorflow/mpi_ops.cc:  auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
horovod/tensorflow/mpi_ops.cc:  HVD_GPU_CHECK(gpuEventRecord(event_, stream));
horovod/tensorflow/mpi_ops.cc:  HVD_GPU_CHECK(gpuEventSynchronize(event_));
horovod/tensorflow/mpi_ops.cc:gpuEvent_t TFReadyEvent::event() const {
horovod/tensorflow/mpi_ops.cc:    auto& queue = ready_event_registry.gpu_events[device_];
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:  // On GPU allocation is asynchronous, we need to wait for it to
horovod/tensorflow/mpi_ops.cc:// On GPU this event will signal that data is ready, and tensors are
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:  // On GPU allocation is asynchronous, we need to wait for it to
horovod/tensorflow/mpi_ops.cc:      auto status_gpu = device_context->stream()->BlockHostUntilDone();
horovod/tensorflow/mpi_ops.cc:      if (!status_gpu.ok()) {
horovod/tensorflow/mpi_ops.cc:        return ConvertStatus(status_gpu);
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:    auto stream = (device_context != nullptr) ? stream_executor::gpu::AsGpuStreamValue(device_context->stream()) : 0;
horovod/tensorflow/mpi_ops.cc:    gpuMemsetAsync(ptr, 0, size, stream);
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:  // On GPU allocation is asynchronous, we need to wait for it to
horovod/tensorflow/mpi_ops.cc:    auto status_gpu = device_context->stream()->BlockHostUntilDone();
horovod/tensorflow/mpi_ops.cc:    if (!status_gpu.ok()) {
horovod/tensorflow/mpi_ops.cc:      return ConvertStatus(status_gpu);
horovod/tensorflow/mpi_ops.cc:    device = context->device()->tensorflow_accelerator_device_info()->gpu_id;
horovod/tensorflow/mpi_ops.cc:      context->device()->tensorflow_gpu_device_info() != nullptr) {
horovod/tensorflow/mpi_ops.cc:    device = context->device()->tensorflow_gpu_device_info()->gpu_id;
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:                auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
horovod/tensorflow/mpi_ops.cc:                HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/tensorflow/mpi_ops.cc:#if HOROVOD_GPU_ALLREDUCE
horovod/tensorflow/mpi_ops.cc:REGISTER_KERNEL_BUILDER(Name("HorovodAllreduce").Device(DEVICE_GPU),
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:                    auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
horovod/tensorflow/mpi_ops.cc:                    HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/tensorflow/mpi_ops.cc:#if HOROVOD_GPU_ALLREDUCE
horovod/tensorflow/mpi_ops.cc:REGISTER_KERNEL_BUILDER(Name("HorovodGroupedAllreduce").Device(DEVICE_GPU),
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:                auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
horovod/tensorflow/mpi_ops.cc:                HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/tensorflow/mpi_ops.cc:#if HOROVOD_GPU_ALLGATHER
horovod/tensorflow/mpi_ops.cc:REGISTER_KERNEL_BUILDER(Name("HorovodAllgather").Device(DEVICE_GPU),
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:              auto stream = stream_executor::gpu::AsGpuStreamValue(
horovod/tensorflow/mpi_ops.cc:              HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/tensorflow/mpi_ops.cc:#if HOROVOD_GPU_ALLGATHER
horovod/tensorflow/mpi_ops.cc:REGISTER_KERNEL_BUILDER(Name("HorovodGroupedAllgather").Device(DEVICE_GPU),
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:                auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
horovod/tensorflow/mpi_ops.cc:                HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/tensorflow/mpi_ops.cc:#if HOROVOD_GPU_BROADCAST
horovod/tensorflow/mpi_ops.cc:REGISTER_KERNEL_BUILDER(Name("HorovodBroadcast").Device(DEVICE_GPU),
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:              auto stream = stream_executor::gpu::AsGpuStreamValue(
horovod/tensorflow/mpi_ops.cc:              HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/tensorflow/mpi_ops.cc:#if HOROVOD_GPU_BROADCAST
horovod/tensorflow/mpi_ops.cc:REGISTER_KERNEL_BUILDER(Name("HorovodBroadcastInplace").Device(DEVICE_GPU),
horovod/tensorflow/mpi_ops.cc:                        HorovodBroadcastInplaceOp<GPUDevice>);
horovod/tensorflow/mpi_ops.cc:#if HOROVOD_GPU_BROADCAST
horovod/tensorflow/mpi_ops.cc:                            .Device(DEVICE_GPU)
horovod/tensorflow/mpi_ops.cc:                        HorovodBroadcastInplaceOp<GPUDevice>);
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:              auto stream = stream_executor::gpu::AsGpuStreamValue(
horovod/tensorflow/mpi_ops.cc:              HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/tensorflow/mpi_ops.cc:#if HOROVOD_GPU_REDUCESCATTER
horovod/tensorflow/mpi_ops.cc:REGISTER_KERNEL_BUILDER(Name("HorovodReducescatter").Device(DEVICE_GPU),
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:              auto stream = stream_executor::gpu::AsGpuStreamValue(
horovod/tensorflow/mpi_ops.cc:              HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/tensorflow/mpi_ops.cc:#if HOROVOD_GPU_REDUCESCATTER
horovod/tensorflow/mpi_ops.cc:REGISTER_KERNEL_BUILDER(Name("HorovodGroupedReducescatter").Device(DEVICE_GPU),
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:                auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
horovod/tensorflow/mpi_ops.cc:                HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/tensorflow/mpi_ops.cc:#if HOROVOD_GPU_ALLREDUCE
horovod/tensorflow/mpi_ops.cc:                            .Device(DEVICE_GPU)
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:    Name("HorovodSize").Device(DEVICE_GPU).HostMemory("size"),
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:    Name("HorovodProcessSetIncluded").Device(DEVICE_GPU).HostMemory("included"),
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:    Name("HorovodLocalSize").Device(DEVICE_GPU).HostMemory("local_size"),
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:    Name("HorovodRank").Device(DEVICE_GPU).HostMemory("rank"),
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:    Name("HorovodLocalRank").Device(DEVICE_GPU).HostMemory("local_rank"),
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:#if HAVE_GPU
horovod/tensorflow/mpi_ops.cc:                auto stream = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
horovod/tensorflow/mpi_ops.cc:                HVD_GPU_CHECK(gpuStreamWaitEvent(stream, *(hvd_event.event), 0));
horovod/tensorflow/mpi_ops.cc:#if HOROVOD_GPU_ALLTOALL
horovod/tensorflow/mpi_ops.cc:                            .Device(DEVICE_GPU)
horovod/tensorflow/__init__.py:from horovod.common.util import check_extension, gpu_available, split_list
horovod/tensorflow/__init__.py:from horovod.tensorflow.mpi_ops import nccl_built, ddl_built, ccl_built, cuda_built, rocm_built
horovod/tensorflow/__init__.py:        device_dense: Device to be used for dense tensors. Uses GPU by default
horovod/tensorflow/__init__.py:                      if Horovod was built with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/__init__.py:        device_sparse: Device to be used for sparse tensors. Uses GPU by default
horovod/tensorflow/__init__.py:                       if Horovod was built with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/__init__.py:        if rocm_built():
horovod/tensorflow/__init__.py:            # For ROCm, perform averaging at framework level
horovod/tensorflow/__init__.py:                if 'CPU' not in tensor.device and gpu_available('tensorflow'):
horovod/tensorflow/__init__.py:                    if nccl_built():
horovod/tensorflow/__init__.py:                                'Running GPU Adasum on heterogeneous cluster is not supported yet.')
horovod/tensorflow/__init__.py:                                'Running GPU Adasum with non-power of 2 nodes is not supported yet.')
horovod/tensorflow/__init__.py:                        if rocm_built():
horovod/tensorflow/__init__.py:                        warnings.warn('Adasum reduction does not currently support GPU reduction using MPI. Tensors '
horovod/tensorflow/__init__.py:                                      'are copied to CPU memory instead. To use Adasum for GPU reduction, please '
horovod/tensorflow/__init__.py:                                      'compile Horovod with HOROVOD_GPU_OPERATIONS=NCCL.')
horovod/tensorflow/__init__.py:                if rocm_built():
horovod/tensorflow/__init__.py:        device_dense: Device to be used for dense tensors. Uses GPU by default
horovod/tensorflow/__init__.py:                      if Horovod was built with HOROVOD_GPU_REDUCESCATTER.
horovod/tensorflow/__init__.py:    if rocm_built() and op == Average:
horovod/tensorflow/__init__.py:        device_dense: Device to be used for dense tensors. Uses GPU by default
horovod/tensorflow/__init__.py:                      if Horovod was built with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/__init__.py:        device_sparse: Device to be used for sparse tensors. Uses GPU by default
horovod/tensorflow/__init__.py:                       if Horovod was built with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/__init__.py:    if rocm_built():
horovod/tensorflow/__init__.py:        # For ROCm, perform averaging at framework level
horovod/tensorflow/__init__.py:                if 'CPU' not in tensor.device and gpu_available('tensorflow'):
horovod/tensorflow/__init__.py:                    if nccl_built():
horovod/tensorflow/__init__.py:                                'Running GPU Adasum on heterogeneous cluster is not supported yet.')
horovod/tensorflow/__init__.py:                                'Running GPU Adasum with non-power of 2 nodes is not supported yet.')
horovod/tensorflow/__init__.py:                        if rocm_built():
horovod/tensorflow/__init__.py:                        warnings.warn('Adasum reduction does not currently support GPU reduction using MPI. Tensors '
horovod/tensorflow/__init__.py:                                      'are copied to CPU memory instead. To use Adasum for GPU reduction, please '
horovod/tensorflow/__init__.py:                                      'compile Horovod with HOROVOD_GPU_OPERATIONS=NCCL.')
horovod/tensorflow/__init__.py:                if rocm_built():
horovod/tensorflow/__init__.py:        device_dense: Device to be used for dense tensors. Uses GPU by default
horovod/tensorflow/__init__.py:                      if Horovod was built with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/__init__.py:        device_sparse: Device to be used for sparse tensors. Uses GPU by default
horovod/tensorflow/__init__.py:                       if Horovod was built with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/__init__.py:    if rocm_built() and op == Average:
horovod/tensorflow/__init__.py:                Device to be used for broadcasting. Uses GPU by default
horovod/tensorflow/__init__.py:                if Horovod was built with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/__init__.py:        Device to be used for dense tensors. Uses GPU by default
horovod/tensorflow/__init__.py:        if Horovod was built with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/__init__.py:        Device to be used for sparse tensors. Uses GPU by default
horovod/tensorflow/__init__.py:        if Horovod was built with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/__init__.py:        if rocm_built():
horovod/tensorflow/__init__.py:            raise ValueError('gradient_predivide_factor not supported yet with ROCm')
horovod/tensorflow/__init__.py:            Device to be used for dense tensors. Uses GPU by default
horovod/tensorflow/__init__.py:            if Horovod was built with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/__init__.py:            Device to be used for sparse tensors. Uses GPU by default
horovod/tensorflow/__init__.py:            if Horovod was built with HOROVOD_GPU_OPERATIONS.
horovod/tensorflow/__init__.py:            if rocm_built():
horovod/tensorflow/__init__.py:                raise ValueError('gradient_predivide_factor not supported yet with ROCm')
horovod/mxnet/mpi_ops.h:// Modifications copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/mxnet/mpi_ops.h:#if MXNET_MAJOR >= 2 || MXNET_ASYNC_GPU_ENGINE_SUPPORTED
horovod/mxnet/mpi_ops.py:# Modifications copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/mxnet/mpi_ops.py:nccl_built = _basics.nccl_built
horovod/mxnet/mpi_ops.py:cuda_built = _basics.cuda_built
horovod/mxnet/mpi_ops.py:rocm_built = _basics.rocm_built
horovod/mxnet/tensor_util.h:#include "cuda_util.h"
horovod/mxnet/tensor_util.h:#if HAVE_CUDA
horovod/mxnet/tensor_util.h:  static void AsyncCopyCPUToCuda(NDArray* cpu, NDArray* cuda);
horovod/mxnet/tensor_util.h:  static void AsyncCopyCudaToCPU(NDArray* cuda, NDArray* cpu);
horovod/mxnet/tensor_util.cc:// If Tensor on GPU, return device id
horovod/mxnet/tensor_util.cc:  if (dev_mask == gpu::kDevMask)
horovod/mxnet/tensor_util.cc:#if HAVE_CUDA
horovod/mxnet/tensor_util.cc:void TensorUtil::AsyncCopyCPUToCuda(NDArray* cpu, NDArray* cuda) {
horovod/mxnet/tensor_util.cc:  TensorUtil::Copy(cuda, cpu);
horovod/mxnet/tensor_util.cc:void TensorUtil::AsyncCopyCudaToCPU(NDArray* cuda, NDArray* cpu) {
horovod/mxnet/tensor_util.cc:  TensorUtil::Copy(cpu, cuda);
horovod/mxnet/functions.py:# Modifications copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/mxnet/adapter.cc:#if HAVE_CUDA
horovod/mxnet/adapter.cc:#include "cuda.h"
horovod/mxnet/adapter.cc:#include "cuda_util.h"
horovod/mxnet/adapter.cc:// circumstances (application shutdown), CUDA context would already be destroyed
horovod/mxnet/adapter.cc:// and cudaFree() operations would print nasty errors in the log - in a pretty
horovod/mxnet/adapter.cc:#if HAVE_CUDA
horovod/mxnet/adapter.cc:    CUDA_CALL(cudaMalloc((void**)&buffer_, size));
horovod/mxnet/adapter.cc:                           "with GPU device but not compiled with CUDA.");
horovod/mxnet/CMakeLists.txt:if (HAVE_CUDA AND NOT Mxnet_USE_CUDA)
horovod/mxnet/CMakeLists.txt:    message(FATAL_ERROR "Horovod build with GPU support was requested but this MXNet installation does not support CUDA.")
horovod/mxnet/CMakeLists.txt:elseif (Mxnet_USE_CUDA AND NOT HAVE_CUDA)
horovod/mxnet/CMakeLists.txt:    add_cuda()
horovod/mxnet/CMakeLists.txt:if(HAVE_CUDA)
horovod/mxnet/CMakeLists.txt:        list(APPEND Mxnet_LINKER_LIBS horovod_cuda_kernels)
horovod/mxnet/CMakeLists.txt:        list(APPEND Mxnet_LINKER_LIBS compatible_horovod_cuda_kernels)
horovod/mxnet/CMakeLists.txt:                          "${PROJECT_SOURCE_DIR}/horovod/mxnet/cuda_util.cc"
horovod/mxnet/cuda_util.h:#ifndef HOROVOD_MXNET_CUDA_UTIL_H
horovod/mxnet/cuda_util.h:#define HOROVOD_MXNET_CUDA_UTIL_H
horovod/mxnet/cuda_util.h:#endif // HOROVOD_MXNET_CUDA_UTIL_H
horovod/mxnet/mpi_ops.cc:// Modifications copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
horovod/mxnet/mpi_ops.cc:#include "cuda_util.h"
horovod/mxnet/mpi_ops.cc:#if MXNET_MAJOR >= 2 || MXNET_ASYNC_GPU_ENGINE_SUPPORTED
horovod/mxnet/mpi_ops.cc:#define MXNET_ASYNC_GPU_ENGINE_SUPPORTED 1
horovod/mxnet/mpi_ops.cc:#if HAVE_CUDA
horovod/mxnet/mpi_ops.cc:  MXReadyEvent(gpuEvent_t event) : event_(event) {};
horovod/mxnet/mpi_ops.cc:      HVD_GPU_CHECK(gpuEventSynchronize(event_));
horovod/mxnet/mpi_ops.cc:  gpuEvent_t event() const override {
horovod/mxnet/mpi_ops.cc:  gpuEvent_t event_;
horovod/mxnet/mpi_ops.cc:#if HAVE_CUDA && MXNET_ASYNC_GPU_ENGINE_SUPPORTED
horovod/mxnet/mpi_ops.cc:    for (auto& cuda_event : sync_obj.reader_events) {
horovod/mxnet/mpi_ops.cc:      auto ev = cuda_event.event.lock();
horovod/mxnet/mpi_ops.cc:#if MXNET_ASYNC_GPU_ENGINE_SUPPORTED
horovod/mxnet/mpi_ops.cc:#if HAVE_CUDA
horovod/mxnet/mpi_ops.cc:#if MXNET_ASYNC_GPU_ENGINE_SUPPORTED
horovod/mxnet/mpi_ops.cc:            dmlc::GetEnv("MXNET_ASYNC_GPU_ENGINE", false);
horovod/mxnet/mpi_ops.cc:          HVD_GPU_CHECK(gpuEventSynchronize(*(hvd_event.event)));
horovod/mxnet/mpi_ops.cc:        HVD_GPU_CHECK(gpuEventSynchronize(*(hvd_event.event)));
horovod/mxnet/mpi_ops.cc:#if MXNET_ASYNC_GPU_ENGINE_SUPPORTED
horovod/mxnet/mpi_ops.cc:#if HAVE_CUDA
horovod/mxnet/mpi_ops.cc:      TensorUtil::AsyncCopyCudaToCPU(splits, splits_tensor.get());
horovod/mxnet/mpi_ops.cc:#if HAVE_CUDA
horovod/mxnet/mpi_ops.cc:#if MXNET_ASYNC_GPU_ENGINE_SUPPORTED
horovod/mxnet/mpi_ops.cc:void DoHorovodOperationCudaOnCPU(void*, void* on_start_ptr, void* on_complete_ptr, void* param) {
horovod/mxnet/mpi_ops.cc:void DoHorovodOperationCudaOnCPU(void*, void* on_complete_ptr, void* param) {
horovod/mxnet/mpi_ops.cc:inline void PushHorovodOperationCudaOnCPU(OperationType op_type, NDArray* const * inputs,
horovod/mxnet/mpi_ops.cc:    TensorUtil::AsyncCopyCudaToCPU(inputs[i], cpu_input_tensors[i].get());
horovod/mxnet/mpi_ops.cc:      TensorUtil::AsyncCopyCudaToCPU(splits, splits_tensor.get());
horovod/mxnet/mpi_ops.cc:    MXEnginePushAsync(DoHorovodOperationCudaOnCPU, ops_param, DeleteMpiOpsParam,
horovod/mxnet/mpi_ops.cc:       // to wait for operation to complete before copying to GPU output.
horovod/mxnet/mpi_ops.cc:       TensorUtil::AsyncCopyCPUToCuda(cpu_output_tensors[i].get(), outputs[i]);
horovod/mxnet/mpi_ops.cc:    MXEnginePushAsync(DoHorovodOperationCudaOnCPU, ops_param, DeleteMpiOpsParam,
horovod/mxnet/mpi_ops.cc:      TensorUtil::AsyncCopyCPUToCuda(cpu_input_tensors[i].get(), outputs[i]);
horovod/mxnet/mpi_ops.cc:#if HAVE_ROCM
horovod/mxnet/mpi_ops.cc:  // Averaging left at framework level for ROCm until ScaleBuffer implementation
horovod/mxnet/mpi_ops.cc:#if HAVE_CUDA && !HOROVOD_GPU_ALLREDUCE
horovod/mxnet/mpi_ops.cc:    PushHorovodOperationCudaOnCPU(OperationType::ALLREDUCE, inputs, outputs,
horovod/mxnet/mpi_ops.cc:#if HAVE_ROCM
horovod/mxnet/mpi_ops.cc:#if HAVE_CUDA && !HOROVOD_GPU_ALLGATHER
horovod/mxnet/mpi_ops.cc:    PushHorovodOperationCudaOnCPU(OperationType::ALLGATHER, inputs, outputs,
horovod/mxnet/mpi_ops.cc:#if HAVE_CUDA && !HOROVOD_GPU_BROADCAST
horovod/mxnet/mpi_ops.cc:    PushHorovodOperationCudaOnCPU(OperationType::BROADCAST, &input, &output,
horovod/mxnet/mpi_ops.cc:#if HAVE_CUDA && !HOROVOD_GPU_ALLTOALL
horovod/mxnet/mpi_ops.cc:    PushHorovodOperationCudaOnCPU(OperationType::ALLTOALL, &input, &output,
horovod/mxnet/mpi_ops.cc:#if HAVE_ROCM
horovod/mxnet/mpi_ops.cc:  // Averaging left at framework level for ROCm until ScaleBuffer implementation
horovod/mxnet/mpi_ops.cc:#if HAVE_CUDA && !HOROVOD_GPU_REDUCESCATTER
horovod/mxnet/mpi_ops.cc:    PushHorovodOperationCudaOnCPU(OperationType::REDUCESCATTER, inputs, outputs,
horovod/mxnet/mpi_ops.cc:#if HAVE_ROCM
horovod/mxnet/__init__.py:from horovod.mxnet.mpi_ops import nccl_built, ddl_built, ccl_built, cuda_built, rocm_built
horovod/mxnet/__init__.py:        if gradient_predivide_factor != 1.0 and rocm_built():
horovod/mxnet/__init__.py:            raise ValueError('gradient_predivide_factor not supported yet with ROCm')
horovod/mxnet/__init__.py:        if rocm_built() or nccl_built() < 21000:
horovod/mxnet/__init__.py:          # Perform average in framework via rescale_grad for ROCM or older NCCL versions
horovod/mxnet/__init__.py:        if gradient_predivide_factor != 1.0 and rocm_built():
horovod/mxnet/__init__.py:            raise ValueError('gradient_predivide_factor not supported yet with ROCm')
horovod/mxnet/__init__.py:        if rocm_built() or nccl_built() < 21000:
horovod/mxnet/__init__.py:          # Perform average in framework via rescale_grad for ROCM or older NCCL versions
horovod/mxnet/util.h:#if HAVE_CUDA
horovod/mxnet/util.h:#include <cuda_runtime.h>
horovod/mxnet/util.h: * \brief Protected CUDA call.
horovod/mxnet/util.h: * It checks for CUDA errors after invocation of the expression.
horovod/mxnet/util.h:#define CUDA_CALL(func)                                                        \
horovod/mxnet/util.h:    cudaError_t e = (func);                                                    \
horovod/mxnet/util.h:    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                   \
horovod/mxnet/util.h:        << "CUDA: " << cudaGetErrorString(e);                                  \
horovod/mxnet/util.h:#endif // HAVE_CUDA
horovod/mxnet/cuda_util.cc:#if HAVE_CUDA
horovod/mxnet/cuda_util.cc:#include "cuda.h"
horovod/mxnet/cuda_util.cc:#include "cuda_runtime.h"
horovod/mxnet/cuda_util.cc:#include "cuda_util.h"
horovod/mxnet/cuda_util.cc:#if HAVE_CUDA
horovod/mxnet/cuda_util.cc:typedef CUresult (CUDAAPI *PFN_cuCtxGetDevice)(CUdevice* device);
horovod/mxnet/cuda_util.cc:static void* cudalib = nullptr;
horovod/mxnet/cuda_util.cc:  cudalib = dlopen("libcuda.so", RTLD_LAZY);
horovod/mxnet/cuda_util.cc:  if (!cudalib) {
horovod/mxnet/cuda_util.cc:    throw std::logic_error("Internal error. Could not dlopen libcuda.so.");
horovod/mxnet/cuda_util.cc:  pfn_cuCtxGetDevice = (PFN_cuCtxGetDevice) dlsym(cudalib, "cuCtxGetDevice");
horovod/mxnet/cuda_util.cc:#if HAVE_CUDA
horovod/mxnet/cuda_util.cc:    if (!cudalib) initialize_driver_api();
horovod/mxnet/cuda_util.cc:    if (err == CUDA_ERROR_NOT_INITIALIZED ||
horovod/mxnet/cuda_util.cc:        err == CUDA_ERROR_INVALID_CONTEXT) {
horovod/mxnet/cuda_util.cc:     } else if (err == CUDA_SUCCESS) {
horovod/mxnet/cuda_util.cc:     CUDA_CALL(cudaSetDevice(device));
horovod/mxnet/cuda_util.cc:                           "with GPU device but not compiled with CUDA.");
horovod/mxnet/cuda_util.cc:#if HAVE_CUDA
horovod/mxnet/cuda_util.cc:    CUDA_CALL(cudaSetDevice(restore_device_));
horovod/runner/gloo_run.py:        'NCCL_SOCKET_IFNAME': ','.join(nics),
horovod/runner/js_run.py:    if nics and 'NCCL_SOCKET_IFNAME' not in env:
horovod/runner/js_run.py:        env['NCCL_SOCKET_IFNAME'] = ','.join(nics)
horovod/runner/js_run.py:    cpu_per_gpu = (lsf.LSFUtils.get_num_cores() * lsf.LSFUtils.get_num_threads()) // lsf.LSFUtils.get_num_gpus()
horovod/runner/js_run.py:        if slots > lsf.LSFUtils.get_num_gpus():
horovod/runner/js_run.py:                             'than number of GPUs per host \'{gpus}\'.'.format(
horovod/runner/js_run.py:                host=host, slots=slots, gpus=lsf.LSFUtils.get_num_gpus()))
horovod/runner/js_run.py:                tmp.write('rank: {rank}: {{ hostname: {host}; cpu: {{{scpu}-{ecpu}}} ; gpu: * ; mem: * }}\n'.format(
horovod/runner/js_run.py:                    ecpu=cpu_val + cpu_per_gpu - 1
horovod/runner/js_run.py:                cpu_val += cpu_per_gpu
horovod/runner/util/lsf.py:            # Fetch the total number of cores and gpus for the first host
horovod/runner/util/lsf.py:            LSFUtils._csm_allocation_info["compute_node_gpus"] = int(node_output["Record_1"]["discovered_gpus"])
horovod/runner/util/lsf.py:    def get_num_gpus():
horovod/runner/util/lsf.py:        """Returns the number of gpus per node."""
horovod/runner/util/lsf.py:        return LSFUtils.get_allocation_info()["compute_node_gpus"]
horovod/runner/util/lsf.py:        return len(LSFUtils.get_compute_hosts()) * LSFUtils.get_num_gpus()
horovod/runner/common/util/config_parser.py:HOROVOD_NUM_NCCL_STREAMS = 'HOROVOD_NUM_NCCL_STREAMS'
horovod/runner/common/util/config_parser.py:NCCL_IB_DISABLE = 'NCCL_IB_DISABLE'
horovod/runner/common/util/config_parser.py:        _set_arg_from_config(args, 'num_nccl_streams', override_args, library_options)
horovod/runner/common/util/config_parser.py:    _validate_arg_nonnegative(args, 'num_nccl_streams')
horovod/runner/common/util/config_parser.py:    _add_arg_to_env(env, HOROVOD_NUM_NCCL_STREAMS, args.num_nccl_streams)
horovod/runner/common/util/config_parser.py:    _add_arg_to_env(env, NCCL_IB_DISABLE, 1 if args.tcp_flag else None)
horovod/runner/common/util/env.py:IGNORE_REGEXES = {'BASH_FUNC_.*', 'OLDPWD', '.*CUDA_VISIBLE_DEVICES', secret.HOROVOD_SECRET_KEY}
horovod/runner/common/util/hosts.py:    <IP address> or <host name>:<Number of GPUs>
horovod/runner/common/util/hosts.py:    :param filename: Should be in <IP address> or <host name> slots=<number of GPUs>
horovod/runner/common/util/hosts.py:    :return: Comma separated string of <IP address> or <host name>:<Number of GPUs>
horovod/runner/launch.py:                                 nccl_built, ddl_built, ccl_built)
horovod/runner/launch.py:        [{nccl_ops}] NCCL
horovod/runner/launch.py:               nccl_ops=get_check(nccl_built(verbose=verbose)),
horovod/runner/launch.py:                                              help='Perform 2D NCCL torus allreduce between workers instead of '
horovod/runner/launch.py:                                                   'but the parallel inter-node allreduce is performed using NCCL in place '
horovod/runner/launch.py:                                                   'of MPI. For each NCCL allreduce operation, NCCL on its own might choose '
horovod/runner/launch.py:                               help='Number of slots for processes per host. Normally 1 slot per GPU per host. '
horovod/runner/launch.py:    group_library_options.add_argument('--num-nccl-streams', action=make_override_action(override_args),
horovod/runner/launch.py:                                       help='Number of NCCL streams. Only applies when running with NCCL support. '
horovod/runner/launch.py:            args.hosts = ','.join(f'{host}:{lsf.LSFUtils.get_num_gpus()}'
horovod/runner/driver/driver_service.py:            # and local) and specify it in the args to be used by NCCL. It is
horovod/runner/mpi_run.py:    nccl_socket_intf_arg = '-{opt} NCCL_SOCKET_IFNAME={nics}'.format(
horovod/runner/mpi_run.py:        '{nccl_socket_intf_arg} '
horovod/runner/mpi_run.py:                nccl_socket_intf_arg=nccl_socket_intf_arg,
horovod/runner/__init__.py:        self.num_nccl_streams = None
horovod/runner/__init__.py:    :param slots: Number of slots for processes per host. Normally 1 slot per GPU per host.
CONTRIBUTING.md:4. Run unit tests in both CPU and GPU environments.

```
