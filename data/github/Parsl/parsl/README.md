# https://github.com/Parsl/parsl

```console
docs/faq.rst:     @python_app(executors=['GPUMachine'])
docs/userguide/apps.rst:    @python_app(cache=True, executors=['gpu'])
docs/userguide/apps.rst:    def expensive_gpu_function():
docs/userguide/configuring.rst:.. [*] The maximum number of nodes tested for the `parsl.executors.WorkQueueExecutor` is 10,000 GPU cores and
docs/userguide/configuring.rst:       10,000 GPU cores and 20,000 CPU cores.
docs/userguide/configuring.rst:each specifying accelerators. In the context of binding to NVIDIA GPUs, this works by setting ``CUDA_VISIBLE_DEVICES``
docs/userguide/configuring.rst:                # Starts 2 workers per node, each bound to 2 GPUs
docs/userguide/configuring.rst:                # Start a single worker bound to all 4 GPUs
docs/userguide/configuring.rst:GPU Oversubscription
docs/userguide/configuring.rst:For hardware that uses Nvidia devices, Parsl allows for the oversubscription of workers to GPUS.  This is intended to
docs/userguide/configuring.rst:make use of Nvidia's `Multi-Process Service (MPS) <https://docs.nvidia.com/deploy/mps/>`_ available on many of their
docs/userguide/configuring.rst:GPUs that allows users to run multiple concurrent processes on a single GPU.  The user needs to set in the
docs/userguide/configuring.rst:``available_accelerators`` option should then be set to the total number of GPU partitions run on a single node in the
docs/userguide/configuring.rst:block.  For example, for a node with 4 Nvidia GPUs, to create 8 workers per GPU, set ``available_accelerators=32``.
docs/userguide/configuring.rst:GPUs will be assigned to workers in ascending order in contiguous blocks.  In the example, workers 0-7 will be placed
docs/userguide/configuring.rst:on GPU 0, workers 8-15 on GPU 1, workers 16-23 on GPU 2, and workers 24-31 on GPU 3.
docs/userguide/execution.rst:  is better suited for GPU nodes.
docs/userguide/execution.rst:     #       (Analysis and visualization phase)         <--- Run on GPU node
docs/userguide/execution.rst:     @bash_app(executors=["Cooley.GPU"])
docs/userguide/workflow.rst:Many tasks in workflows require a expensive "initialization" steps that, once performed, can be used across successive invocations for that task. For example, you may want to reuse a machine learning model for multiple interface tasks and avoid loading it onto GPUs more than once.
docs/userguide/mpi_apps.rst:                    select_options="ngpus=4",
docs/index.rst:Functions can be pure Python or invoke external codes, be single- or multi-threaded or GPUs.
docs/quickstart.rst:pin each worker to specific GPUs or CPU cores
docs/quickstart.rst:                available_accelerators=4,  # Maps one worker per GPU
docs/quickstart.rst:                    select_options="ngpus=4",
parsl/configs/bridges.py:                # script to the scheduler eg: '#SBATCH --gres=gpu:type:n'
parsl/configs/polaris.py:                select_options="ngpus=4",
parsl/tests/sites/test_affinity.py:    device = os.environ.get('CUDA_VISIBLE_DEVICES')
parsl/tests/sites/test_affinity.py:    assert worker_affinity[0][1] == "0"  # Make sure it is pinned to the correct CUDA device
parsl/executors/taskvine/factory_config.py:    gpus: Optional[int]
parsl/executors/taskvine/factory_config.py:        Number of gpus a worker should have.
parsl/executors/taskvine/factory_config.py:        gpus of the machine it lands on.
parsl/executors/taskvine/factory_config.py:    gpus: Optional[int] = None
parsl/executors/taskvine/executor.py:        gpus = None
parsl/executors/taskvine/executor.py:                elif k == 'gpus':
parsl/executors/taskvine/executor.py:                    gpus = resource_specification[k]
parsl/executors/taskvine/executor.py:                                    gpus=gpus,
parsl/executors/taskvine/factory.py:    if factory_config.gpus:
parsl/executors/taskvine/factory.py:        factory.gpus = factory_config.gpus
parsl/executors/taskvine/utils.py:                 gpus: Optional[float],            # number of gpus to allocate
parsl/executors/taskvine/utils.py:        self.gpus = gpus
parsl/executors/taskvine/manager.py:            if task.gpus is not None:
parsl/executors/taskvine/manager.py:                t.set_gpus(task.gpus)
parsl/executors/flux/executor.py:            -  gpus_per_task: gpus per task, default 1
parsl/executors/flux/executor.py:            -  num_nodes: if > 0, evenly distribute the allocated cores/gpus
parsl/executors/flux/executor.py:        gpus_per_task=jobinfo.resource_spec.get("gpus_per_task"),
parsl/executors/workqueue/executor.py:                        'cores memory disk gpus priority running_time_min env_pkg map_file function_file result_file input_files output_files')
parsl/executors/workqueue/executor.py:        gpus = None
parsl/executors/workqueue/executor.py:            acceptable_fields = set(['cores', 'memory', 'disk', 'gpus', 'priority', 'running_time_min'])
parsl/executors/workqueue/executor.py:                elif k == 'gpus':
parsl/executors/workqueue/executor.py:                    gpus = resource_specification[k]
parsl/executors/workqueue/executor.py:                                                 gpus,
parsl/executors/workqueue/executor.py:            if task.gpus is not None:
parsl/executors/workqueue/executor.py:                t.specify_gpus(task.gpus)
parsl/executors/radical/rpex_resources.py:    worker_gpus_per_node : int
parsl/executors/radical/rpex_resources.py:        The number of GPUs a worker will operate on per node.
parsl/executors/radical/rpex_resources.py:    worker_gpus_per_node: int = 0
parsl/executors/radical/rpex_resources.py:            'gpus_per_node': cls.worker_gpus_per_node,
parsl/executors/radical/rpex_resources.py:                "gpus_per_rank": cls.nodes_per_worker * cls.worker_gpus_per_node,
parsl/executors/high_throughput/process_worker_pool.py:        # If CUDA devices, find total number of devices to allow for MPS
parsl/executors/high_throughput/process_worker_pool.py:        # See: https://developer.nvidia.com/system-management-interface
parsl/executors/high_throughput/process_worker_pool.py:        nvidia_smi_cmd = "nvidia-smi -L > /dev/null && nvidia-smi -L | wc -l"
parsl/executors/high_throughput/process_worker_pool.py:        nvidia_smi_ret = subprocess.run(nvidia_smi_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
parsl/executors/high_throughput/process_worker_pool.py:        if nvidia_smi_ret.returncode == 0:
parsl/executors/high_throughput/process_worker_pool.py:            num_cuda_devices = int(nvidia_smi_ret.stdout.split()[0])
parsl/executors/high_throughput/process_worker_pool.py:            num_cuda_devices = None
parsl/executors/high_throughput/process_worker_pool.py:            if num_cuda_devices is not None:
parsl/executors/high_throughput/process_worker_pool.py:                procs_per_cuda_device = pool_size // num_cuda_devices
parsl/executors/high_throughput/process_worker_pool.py:                partitioned_accelerator = str(int(accelerator) // procs_per_cuda_device)  # multiple workers will share a GPU
parsl/executors/high_throughput/process_worker_pool.py:                os.environ["CUDA_VISIBLE_DEVICES"] = partitioned_accelerator
parsl/executors/high_throughput/process_worker_pool.py:                logger.info(f'Pinned worker to partitioned cuda device: {partitioned_accelerator}')
parsl/executors/high_throughput/process_worker_pool.py:                os.environ["CUDA_VISIBLE_DEVICES"] = accelerator
parsl/executors/high_throughput/process_worker_pool.py:            os.environ["CUDA_VISIBLE_DEVICES"] = accelerator
parsl/providers/slurm/slurm.py:        Slurm job constraint, often used to choose cpu or gpu type. If unspecified or ``None``, no constraint slurm directive will be added.
parsl/providers/pbspro/pbspro.py:        specify ngpus.

```
