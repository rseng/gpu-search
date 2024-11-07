# https://github.com/cmgcds/fastvpinns

```console
docker_initialise.py:    To check tensorflow version and GPU Support and number of GPUs available
docker_initialise.py:    gpu_support = "Not Found"
docker_initialise.py:    number_of_gpus = "Not Found"
docker_initialise.py:    gpu_support = subprocess.run(
docker_initialise.py:        ["python3", "-c", "import tensorflow as tf; print(tf.test.is_gpu_available())"],
docker_initialise.py:    number_of_gpus = subprocess.run(
docker_initialise.py:            "import tensorflow as tf; print(len(tf.config.experimental.list_physical_devices('GPU')))",
docker_initialise.py:        gpu_support.stdout.strip(),
docker_initialise.py:        number_of_gpus.stdout.strip(),
docker_initialise.py:def get_cuda_cudnn_nvidia_versions():
docker_initialise.py:    cuda_version = 'Not found'
docker_initialise.py:    # Get CUDA version
docker_initialise.py:        cuda_version = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
docker_initialise.py:        cuda_version = (
docker_initialise.py:            re.search(r'release (\d+\.\d+)', cuda_version.stdout).group(1)
docker_initialise.py:            if cuda_version.stdout
docker_initialise.py:        with open('/usr/local/cuda/include/cudnn_version.h', 'r') as f:
docker_initialise.py:    # Get NVIDIA driver version
docker_initialise.py:    nvidia_driver_version = 'Not found'
docker_initialise.py:        nvidia_driver_version = subprocess.run(
docker_initialise.py:            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
docker_initialise.py:        nvidia_driver_version = (
docker_initialise.py:            nvidia_driver_version.stdout.strip() if nvidia_driver_version.stdout else 'Not found'
docker_initialise.py:    return cuda_version, cudnn_version, nvidia_driver_version.split('\n')[0]
docker_initialise.py:    # obtain the cuda versions
docker_initialise.py:    cuda_version, cudnn_version, nvidia_driver_version = get_cuda_cudnn_nvidia_versions()
docker_initialise.py:    if cuda_version != 'Not found' and nvidia_driver_version != 'Not found':
docker_initialise.py:        print(f"\033[92mGPU Checks Passed - GPU Acceleration is Available \033[0m")
docker_initialise.py:        print(f"\033[91mGPU Checks Failed - Execution is available on CPU only\033[0m")
docker_initialise.py:    tensor_flow_version, gpu_support, number_of_gpus = check_tensorflow()
docker_initialise.py:        f"| CUDA Version:       {cuda_version:<{column_width}} || cuDNN Version: {cudnn_version:<{column_width}}   || NVIDIA Driver Version: {nvidia_driver_version:<{column_width}} |"
docker_initialise.py:        f"| Tensorflow Version: {tensor_flow_version:<{column_width}} || GPU Support: {gpu_support:<{column_width}}     || Number of GPUs: {number_of_gpus:<{column_width}}        |"
Dockerfile:# Download the base image for CUDA Libraries and cuDNN
Dockerfile:FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
docs/_rst/tutorials/forward_problems_2d/hard_boundary_constraints/poisson_2d/README.md:6. `set_memory_growth`, when set to `True` will enable tensorflow's memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to `False` for now. 
docs/_rst/tutorials/forward_problems_2d/hard_boundary_constraints/poisson_2d/poisson2d_hard.rst:-  ``set_memory_growth``, when set to ``True`` will enable tensorflow’s memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to ``False`` for now.
docs/_rst/tutorials/forward_problems_2d/hard_boundary_constraints/poisson_2d/poisson2d_hard.rst:      set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
docs/_rst/tutorials/forward_problems_2d/complex_mesh/poisson2d/README.md:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
docs/_rst/tutorials/forward_problems_2d/complex_mesh/poisson2d/poisson2d.rst:     set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
docs/_rst/tutorials/forward_problems_2d/complex_mesh/helmholtz2d/README.md:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
docs/_rst/tutorials/forward_problems_2d/complex_mesh/helmholtz2d/helmholtz2d.rst:     set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
docs/_rst/tutorials/forward_problems_2d/complex_mesh/helmholtz2d/input.yaml:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
docs/_rst/tutorials/forward_problems_2d/complex_mesh/cd2d/cd2d.rst:     set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
docs/_rst/tutorials/forward_problems_2d/complex_mesh/cd2d/README.md:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
docs/_rst/tutorials/forward_problems_2d/uniform_mesh/poisson_2d/README.md:6. `set_memory_growth`, when set to `True` will enable tensorflow's memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to `False` for now. 
docs/_rst/tutorials/forward_problems_2d/uniform_mesh/poisson_2d/poisson2d_uniform.rst:-  ``set_memory_growth``, when set to ``True`` will enable tensorflow’s memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to ``False`` for now.
docs/_rst/tutorials/forward_problems_2d/uniform_mesh/poisson_2d/poisson2d_uniform.rst:      set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
docs/_rst/tutorials/forward_problems_2d/uniform_mesh/helmholtz_2d/helmholtz2d_uniform.rst:-  ``set_memory_growth``, when set to ``True`` will enable tensorflow’s memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to ``False`` for now.
docs/_rst/tutorials/forward_problems_2d/uniform_mesh/helmholtz_2d/helmholtz2d_uniform.rst:      set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
docs/_rst/tutorials/forward_problems_2d/uniform_mesh/helmholtz_2d/README.md:6. `set_memory_growth`, when set to `True` will enable tensorflow's memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to `False` for now. 
docs/_rst/tutorials/inverse_problems_2d/const_inverse_poisson2d/README.md:6. `set_memory_growth`, when set to `True` will enable tensorflow's memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to `False` for now. 
docs/_rst/tutorials/inverse_problems_2d/const_inverse_poisson2d/inverse_constant.rst:-  ``set_memory_growth``, when set to ``True`` will enable tensorflow’s memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to ``False`` for now.
docs/_rst/tutorials/inverse_problems_2d/domain_inverse_cd2d/README.md:6. `set_memory_growth`, when set to `True` will enable tensorflow's memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to `False` for now. 
docs/_rst/tutorials/inverse_problems_2d/domain_inverse_cd2d/domain_inverse.rst:-  ``set_memory_growth``, when set to ``True`` will enable tensorflow’s memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to ``False`` for now.
docs/_rst/_docker.rst:The docker image is available on `Docker Hub <https://hub.docker.com/r/thivinanandh/fastvpinns>`_. This docker version is based on a `Ubuntu 20.04` image with `Python 3.10` installed. The Docker image can support GPU acceleration as it comes with `CUDA 11.1` and `cuDNN 8.0` installed. 
docs/_rst/_docker.rst:Pre-requisite for installing Docker with GPU Support
docs/_rst/_docker.rst:For GPU support, you need to install `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ on your machine. The toolkit is required to run the docker container with GPU support. Follow all the guidelines in installing the necessary drivers and the toolkit for GPU support. For installing docker on your machine, you can follow the instructions provided in the `Docker Documentation <https://docs.docker.com/engine/install/ubuntu/>`_.
docs/_rst/_docker.rst:4. To run the docker container with GPU support, you can use the following command:
docs/_rst/_docker.rst:    docker run --gpus all -it --rm -v ~/fastvpinns_docker_output:/fastvpinns/output thivinanandh/fastvpinns:latest
docs/_rst/_docker.rst:- `--gpus all` : This flag is used to enable GPU support for the container.
docs/_rst/_docker.rst:5. When loaded into the container, The console will look like the image shown below. Here it shows whether the GPU is detected and the version of the installed packages. 
docs/_rst/_docker.rst:6. Navigate to the `examples` folder to run the examples provided in the repository. The Status of GPU can be viewed by running the following command:
docs/_rst/_docker.rst:    nvidia-smi
fastvpinns/hyperparameter_tuning/optuna_tuner.py:        self.gpus = tf.config.list_physical_devices('GPU')
fastvpinns/hyperparameter_tuning/optuna_tuner.py:        print(f"Available GPUs: {len(self.gpus)}")
fastvpinns/hyperparameter_tuning/optuna_tuner.py:        Wrapper function to run the objective function on a specific GPU.
fastvpinns/hyperparameter_tuning/optuna_tuner.py:        gpu_id = trial.number % len(self.gpus)
fastvpinns/hyperparameter_tuning/optuna_tuner.py:        with tf.device(f'/device:GPU:{gpu_id}'):
fastvpinns/hyperparameter_tuning/optuna_tuner.py:            self.objective_wrapper, n_trials=self.n_trials, n_jobs=min(len(self.gpus), self.n_jobs)
fastvpinns/hyperparameter_tuning/objective.py:    gpus = tf.config.list_physical_devices('GPU')
fastvpinns/hyperparameter_tuning/objective.py:    if gpus:
fastvpinns/hyperparameter_tuning/objective.py:            tf.config.experimental.set_memory_growth(gpus[0], True)
examples/forward_problems_2d/hard_boundary_constraints/poisson_2d/README.md:6. `set_memory_growth`, when set to `True` will enable tensorflow's memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to `False` for now. 
examples/forward_problems_2d/hard_boundary_constraints/poisson_2d/input.yaml:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
examples/forward_problems_2d/complex_mesh/poisson2d/README.md:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
examples/forward_problems_2d/complex_mesh/poisson2d/input.yaml:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
examples/forward_problems_2d/complex_mesh/USA/input.yaml:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
examples/forward_problems_2d/complex_mesh/helmholtz2d/README.md:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
examples/forward_problems_2d/complex_mesh/helmholtz2d/input.yaml:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
examples/forward_problems_2d/complex_mesh/cd2d/README.md:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
examples/forward_problems_2d/complex_mesh/cd2d/input.yaml:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
examples/forward_problems_2d/uniform_mesh/poisson_2d/README.md:6. `set_memory_growth`, when set to `True` will enable tensorflow's memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to `False` for now. 
examples/forward_problems_2d/uniform_mesh/poisson_2d/input.yaml:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
examples/forward_problems_2d/uniform_mesh/helmholtz_2d/README.md:6. `set_memory_growth`, when set to `True` will enable tensorflow's memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to `False` for now. 
examples/forward_problems_2d/uniform_mesh/helmholtz_2d/input.yaml:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.
examples/inverse_problems_2d/const_inverse_poisson2d/README.md:6. `set_memory_growth`, when set to `True` will enable tensorflow's memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to `False` for now. 
examples/inverse_problems_2d/domain_inverse_cd2d/circle/README.md:6. `set_memory_growth`, when set to `True` will enable tensorflow's memory growth function, restricting the memory usage on the GPU. This is currently under development and must be set to `False` for now. 
main.py:    gpus = tf.config.list_physical_devices('GPU')
main.py:        tuner = OptunaTuner(n_trials=args.n_trials, n_jobs=len(gpus), n_epochs=args.n_epochs)
input.yaml:  set_memory_growth: False  # Flag indicating whether to set memory growth for GPU.

```
