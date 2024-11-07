# https://github.com/odb9402/CNNPeaks

```console
setup.py:install_requires = ['pysam','scipy','numpy','sklearn','tensorflow-gpu','pandas','progressbar2']
Dockerfile:# Ubuntu-based, Nvidia-GPU-enabled environment for developing changes for TensorFlow.
Dockerfile:# Start from Nvidia's Ubuntu base image with CUDA and CuDNN, with TF development
Dockerfile:FROM nvidia/cuda:9.0-base-ubuntu${UBUNTU_VERSION}
Dockerfile:        cuda-command-line-tools-9-0 \
Dockerfile:        cuda-cublas-dev-9-0 \
Dockerfile:        cuda-cudart-dev-9-0 \
Dockerfile:        cuda-cufft-dev-9-0 \
Dockerfile:        cuda-curand-dev-9-0 \
Dockerfile:        cuda-cusolver-dev-9-0 \
Dockerfile:        cuda-cusparse-dev-9-0 \
Dockerfile:        libcudnn7=7.2.1.38-1+cuda9.0 \
Dockerfile:        libcudnn7-dev=7.2.1.38-1+cuda9.0 \
Dockerfile:        libnccl2=2.2.13-1+cuda9.0 \
Dockerfile:        libnccl-dev=2.2.13-1+cuda9.0 \
Dockerfile:    find /usr/local/cuda-9.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
Dockerfile:        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
Dockerfile:        apt-get install libnvinfer4=4.1.2-1+cuda9.0 && \
Dockerfile:        apt-get install libnvinfer-dev=4.1.2-1+cuda9.0
Dockerfile:# Link NCCL libray and header where the build script expects them.
Dockerfile:RUN mkdir /usr/local/cuda-9.0/lib &&  \
Dockerfile:    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/lib/libnccl.so.2 && \
Dockerfile:    ln -s /usr/include/nccl.h /usr/local/cuda/include/nccl.h
Dockerfile:        tensorflow-gpu==1.9.0\
README.md:We highly recommend that use Docker to install CNN-Peaks. CNN-Peaks easily can be installed by building docker-image using our Dockerfile. However, you might use [Nvidia-docker](https://github.com/odb9402/ConvPeaks#4-gpu-accelerated-cnn-peaks) to run GPU-accelerated CNN-Peaks.
README.md:> nvidia-docker build . -t cnnpeaks:test
README.md:> nvidia-docker run -i -t -v <data directory>:/CNNpeaks/CNNPeaks/data cnnpeaks:test
README.md:If you want to GPU-accelerated CNN-Peaks, please check ["GPU-accelerated CNN-Peaks"](https://github.com/odb9402/ConvPeaks#4-gpu-accelerated-cnn-peaks)
README.md:> pip install numpy scipy sklearn tensorflow pandas progressbar2 pysam tensorflow-gpu
README.md:### 4. GPU-accelerated CNN-Peaks
README.md:Note that if you want to GPU accelerated CNN-Peaks, your tensorflow should be configured to use GPU. Please check [here](https://www.tensorflow.org/install/gpu) for a description to configure GPU support for CNN-Peaks.
README.md:You can build Docker image if you want to run CNN-peaks on your Docker container. However, you might use Nvidia-docker as long as you use our Dockerfile.
install.sh:pip install numpy scipy sklearn tensorflow pandas progressbar2 pysam tensorflow-gpu
dependencies/nvidia-docker_install.sh:curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
dependencies/nvidia-docker_install.sh:curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
dependencies/nvidia-docker_install.sh:  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
dependencies/nvidia-docker_install.sh:# Install nvidia-docker2 and reload the Docker daemon configuration
dependencies/nvidia-docker_install.sh:sudo apt-get install -y nvidia-docker2

```
