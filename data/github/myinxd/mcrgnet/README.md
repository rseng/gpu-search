# https://github.com/myinxd/MCRGNet

```console
setup.py:        "tensorflow-gpu"
cuda_installation.md:## Install tensorflow with gpu library CUDA on Ubuntu 16.04 x64
cuda_installation.md:2. GPU card: Nvidia GeForce GT 620
cuda_installation.md:3. tensoflow-gpu==1.2.1
cuda_installation.md:4. CUDA: 8.0 https://developer.nvidia.com/cuda-downloads 
cuda_installation.md:5. cuDNN: v5.1 https://developer.nvidia.com/rdp/cudnn-download
cuda_installation.md:- Install cuda, and configure path and LD_LIBRARY_PATH
cuda_installation.md:$ sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64.deb
cuda_installation.md:$ sudo apt-get install cuda
cuda_installation.md:$ export PATH=$PATH:/usr/local/cuda-8.0/bin
cuda_installation.md:$ export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/lib
cuda_installation.md:$ cd cuda
cuda_installation.md:$ sudo cp ./lib64/* /usr/local/cuda/lib64/
cuda_installation.md:$ sudo chmod 755 /usr/local/cuda/lib64/libcudnn*
cuda_installation.md:$ sudo cp ./include/cudnn.h /usr/local/cuda/include/
cuda_installation.md:$ <sudo> pip3 install <--user> <--update> tenforflow-gpu==1.2.1
cuda_installation.md:Note that cuda 8.0 doesn't support the default g++ version. Install an supported version and make it the default.
README.md:In addition, the computation can be accelerated by paralledly processing with GPUs. In this work, our scripts are written under the guide of [Nvidia CUDA](https://developer.nvidia.com/cuda-downloads), thus the Nvidia GPU hardware is also required. You can either refer to the official guide to install CUDA, or refer to this brief [guide](https://github.com/myinxd/MCRGNet/blob/master/cuda_installation.md) by us.
requirements.txt:tensorflow-gpu>=1.4.0

```
