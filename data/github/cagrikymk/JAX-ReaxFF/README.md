# https://github.com/cagrikymk/JAX-ReaxFF

```console
setup.py:def get_cuda_version():
setup.py:      print("nvcc output cannot be parsed to receive the CUDA version")
setup.py:    print("nvcc command cannot be run to find the CUDA version")
setup.py:cuda_version = get_cuda_version()
setup.py:if cuda_version == None:
setup.py:  print("First CUDA needs to be installed")
setup.py:print("Detected cuda version: ", cuda_version)
setup.py:cuda_version = "cuda{}".format(cuda_version.replace(".",""))
setup.py:#TODO: Automate installation for cuda dependent jaxlib
README.md:By utilizing the JAX library to compute gradients of the loss function, we can employ highly efficient local optimization methods, drastically reducing optimization time from days to minutes. JAX-ReaxFF runs efficiently on multi-core CPUs, GPUs, and TPUs, making it versatile and powerful. It also provides a sandbox environment for exploring custom ReaxFF functional forms, enhancing modeling accuracy.
README.md:Since the optimizer is highly more performant on GPUs, GPU version of jaxlib needs to be installed (GPU version supports both CPU and GPU execution). <br>
README.md:**1-** Before the installation, a supported version of CUDA and CuDNN are needed (for jaxlib). Alternatively, one could install the jax-md version that comes with required CUDA libraries. <br>
README.md:**5-** To have the GPU support, jaxlib with CUDA support needs to be installed, otherwise the code can only run on CPUs.
README.md:pip install -U "jax[cuda12]==0.4.30"
README.md:After installing the GPU version, the script will automatically utilize the GPU. If the script does not detect the GPU, it will print a warning message.
README.md:On a HPC cluster, CUDA might be loaded somewhere different than /usr/local/cuda-xx.x. In this case, XLA compiler might not locate CUDA installation. This only happens if you install JAX with local CUDA support.
README.md:To solve this, we can speficy the cuda directory using XLA_FLAGS:
README.md:# To see where cuda is installed
README.md:which nvcc # will print /opt/software/CUDAcore/11.1.1/bin/nvcc
README.md:export XLA_FLAGS="$XLA_FLAGS --xla_gpu_cuda_data_dir=/opt/software/CUDAcore/11.1.1"
README.md:export XLA_FLAGS="$XLA_FLAGS --xla_gpu_force_compilation_parallelism=1"
jaxreaxff/driver_v2.py:      print("To use the GPU version, jaxlib with CUDA support needs to installed!")
jaxreaxff/driver.py:    print("To use the GPU version, jaxlib with CUDA support needs to installed!")

```
