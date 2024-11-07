# https://github.com/Kaixhin/dockerfiles

```console
cuda-digits/cuda_v7.5/Dockerfile:# Start with CUDA DIGITS dependencies
cuda-digits/cuda_v7.5/Dockerfile:FROM kaixhin/cuda-digits-deps:7.5
cuda-digits/cuda_v7.5/Dockerfile:# Move into NVIDIA Caffe repo
cuda-digits/cuda_v7.5/Dockerfile:RUN cd /root && git clone https://github.com/NVIDIA/DIGITS.git digits && cd digits && \
cuda-digits/cuda_v7.5/deps/Dockerfile:FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
cuda-digits/cuda_v7.5/deps/Dockerfile:# Install NCCL for multi-GPU communication
cuda-digits/cuda_v7.5/deps/Dockerfile:RUN apt-get install -y cuda-ld-conf-7-5 && \
cuda-digits/cuda_v7.5/deps/Dockerfile:  wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda7.5/libnccl1_1.2.3-1.cuda7.5_amd64.deb && \
cuda-digits/cuda_v7.5/deps/Dockerfile:  dpkg -i libnccl1_1.2.3-1.cuda7.5_amd64.deb && \
cuda-digits/cuda_v7.5/deps/Dockerfile:  rm libnccl1_1.2.3-1.cuda7.5_amd64.deb && \
cuda-digits/cuda_v7.5/deps/Dockerfile:  wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda7.5/libnccl-dev_1.2.3-1.cuda7.5_amd64.deb && \
cuda-digits/cuda_v7.5/deps/Dockerfile:  dpkg -i libnccl-dev_1.2.3-1.cuda7.5_amd64.deb && \
cuda-digits/cuda_v7.5/deps/Dockerfile:  rm libnccl-dev_1.2.3-1.cuda7.5_amd64.deb
cuda-digits/cuda_v7.5/deps/Dockerfile:# Clone NVIDIA Caffe repo and move into it
cuda-digits/cuda_v7.5/deps/Dockerfile:RUN cd /root && git clone https://github.com/NVIDIA/caffe.git && cd caffe && \
cuda-digits/cuda_v7.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-digits.svg)](https://hub.docker.com/r/kaixhin/cuda-digits/)
cuda-digits/cuda_v7.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-digits.svg)](https://hub.docker.com/r/kaixhin/cuda-digits/)
cuda-digits/cuda_v7.5/README.md:cuda-digits
cuda-digits/cuda_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v5](https://developer.nvidia.com/cuDNN) + [Caffe](http://caffe.berkeleyvision.org/) (NVIDIA fork) + [DIGITS](https://developer.nvidia.com/digits).
cuda-digits/cuda_v7.5/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-digits/cuda_v7.5/README.md:Use NVIDIA Docker: ``nvidia-docker run -dP kaixhin/cuda-digits``.
cuda-digits/cuda_v7.5/README.md:For automatically mapping the DIGITS server port use ``nvidia-docker run -dP kaixhin/cuda-digits`` and `docker port <id>` to retrieve the port.
cuda-digits/cuda_v7.5/README.md:For specifying the port manually use ``nvidia-docker run -d -p <port>:5000 kaixhin/cuda-digits``.
cuda-digits/cuda_v7.5/README.md:The shell can be entered as usual using ``nvidia-docker run -it kaixhin/cuda-digits bash``.
cuda-digits/cuda_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-digits/cuda_v8.0/Dockerfile:# Start with CUDA DIGITS dependencies
cuda-digits/cuda_v8.0/Dockerfile:FROM kaixhin/cuda-digits-deps:8.0
cuda-digits/cuda_v8.0/Dockerfile:# Move into NVIDIA Caffe repo
cuda-digits/cuda_v8.0/Dockerfile:RUN cd /root && git clone https://github.com/NVIDIA/DIGITS.git digits && cd digits && \
cuda-digits/cuda_v8.0/deps/Dockerfile:FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04
cuda-digits/cuda_v8.0/deps/Dockerfile:# Install NCCL for multi-GPU communication
cuda-digits/cuda_v8.0/deps/Dockerfile:RUN wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda8.0/libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
cuda-digits/cuda_v8.0/deps/Dockerfile:  dpkg -i libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
cuda-digits/cuda_v8.0/deps/Dockerfile:  rm libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
cuda-digits/cuda_v8.0/deps/Dockerfile:  wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda8.0/libnccl-dev_1.2.3-1.cuda8.0_amd64.deb && \
cuda-digits/cuda_v8.0/deps/Dockerfile:  dpkg -i libnccl-dev_1.2.3-1.cuda8.0_amd64.deb && \
cuda-digits/cuda_v8.0/deps/Dockerfile:  rm libnccl-dev_1.2.3-1.cuda8.0_amd64.deb
cuda-digits/cuda_v8.0/deps/Dockerfile:# Clone NVIDIA Caffe repo and move into it
cuda-digits/cuda_v8.0/deps/Dockerfile:RUN cd /root && git clone https://github.com/NVIDIA/caffe.git && cd caffe && \
cuda-digits/cuda_v8.0/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-digits.svg)](https://hub.docker.com/r/kaixhin/cuda-digits/)
cuda-digits/cuda_v8.0/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-digits.svg)](https://hub.docker.com/r/kaixhin/cuda-digits/)
cuda-digits/cuda_v8.0/README.md:cuda-digits
cuda-digits/cuda_v8.0/README.md:Ubuntu Core 14.04 + [CUDA 8.0](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v6](https://developer.nvidia.com/cuDNN) + [Caffe](http://caffe.berkeleyvision.org/) (NVIDIA fork) + [DIGITS](https://developer.nvidia.com/digits).
cuda-digits/cuda_v8.0/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-digits/cuda_v8.0/README.md:Use NVIDIA Docker: ``nvidia-docker run -dP kaixhin/cuda-digits``.
cuda-digits/cuda_v8.0/README.md:For automatically mapping the DIGITS server port use ``nvidia-docker run -dP kaixhin/cuda-digits`` and `docker port <id>` to retrieve the port.
cuda-digits/cuda_v8.0/README.md:For specifying the port manually use ``nvidia-docker run -d -p <port>:5000 kaixhin/cuda-digits``.
cuda-digits/cuda_v8.0/README.md:The shell can be entered as usual using ``nvidia-docker run -it kaixhin/cuda-digits bash``.
cuda-digits/cuda_v8.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-pylearn2/cuda_v7.5/Dockerfile:# Start with CUDA Theano base image
cuda-pylearn2/cuda_v7.5/Dockerfile:FROM kaixhin/cuda-theano:7.5
cuda-pylearn2/cuda_v7.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-pylearn2.svg)](https://hub.docker.com/r/kaixhin/cuda-pylearn2/)
cuda-pylearn2/cuda_v7.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-pylearn2.svg)](https://hub.docker.com/r/kaixhin/cuda-pylearn2/)
cuda-pylearn2/cuda_v7.5/README.md:cuda-pylearn2
cuda-pylearn2/cuda_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v5](https://developer.nvidia.com/cuDNN) + [Pylearn2](http://deeplearning.net/software/pylearn2/).
cuda-pylearn2/cuda_v7.5/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-pylearn2/cuda_v7.5/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-pylearn2``.
cuda-pylearn2/cuda_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-pylearn2/cuda_v8.0/Dockerfile:# Start with CUDA Theano base image
cuda-pylearn2/cuda_v8.0/Dockerfile:FROM kaixhin/cuda-theano:8.0
cuda-pylearn2/cuda_v8.0/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-pylearn2.svg)](https://hub.docker.com/r/kaixhin/cuda-pylearn2/)
cuda-pylearn2/cuda_v8.0/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-pylearn2.svg)](https://hub.docker.com/r/kaixhin/cuda-pylearn2/)
cuda-pylearn2/cuda_v8.0/README.md:cuda-pylearn2
cuda-pylearn2/cuda_v8.0/README.md:Ubuntu Core 14.04 + [CUDA 8.0](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v5](https://developer.nvidia.com/cuDNN) + [Pylearn2](http://deeplearning.net/software/pylearn2/).
cuda-pylearn2/cuda_v8.0/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-pylearn2/cuda_v8.0/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-pylearn2``.
cuda-pylearn2/cuda_v8.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-torch/cuda_v7.5/Dockerfile:FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
cuda-torch/cuda_v7.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-torch.svg)](https://hub.docker.com/r/kaixhin/cuda-torch/)
cuda-torch/cuda_v7.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-torch.svg)](https://hub.docker.com/r/kaixhin/cuda-torch/)
cuda-torch/cuda_v7.5/README.md:cuda-torch
cuda-torch/cuda_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v5](https://developer.nvidia.com/cuDNN) + [Torch7](http://torch.ch/) (including iTorch).
cuda-torch/cuda_v7.5/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-torch/cuda_v7.5/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-torch``.
cuda-torch/cuda_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-torch/cuda_v7.5/README.md:To use Jupyter/iTorch open up the appropriate port. For example, use ``nvidia-docker run -it -p 8888:8888 kaixhin/cuda-torch``. Then run `jupyter notebook --ip="0.0.0.0" --no-browser --allow-root` to open a notebook on `localhost:8888`.
cuda-torch/cuda_v8.0/Dockerfile:FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
cuda-torch/cuda_v8.0/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-torch.svg)](https://hub.docker.com/r/kaixhin/cuda-torch/)
cuda-torch/cuda_v8.0/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-torch.svg)](https://hub.docker.com/r/kaixhin/cuda-torch/)
cuda-torch/cuda_v8.0/README.md:cuda-torch
cuda-torch/cuda_v8.0/README.md:Ubuntu Core 14.04 + [CUDA 8.0](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v6](https://developer.nvidia.com/cuDNN) + [Torch7](http://torch.ch/) (including iTorch).
cuda-torch/cuda_v8.0/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-torch/cuda_v8.0/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-torch``.
cuda-torch/cuda_v8.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-torch/cuda_v8.0/README.md:To use Jupyter/iTorch open up the appropriate port. For example, use ``nvidia-docker run -it -p 8888:8888 kaixhin/cuda-torch``. Then run `jupyter notebook --ip="0.0.0.0" --no-browser --allow-root` to open a notebook on `localhost:8888`.
cuda-vnc/cuda_v7.5/Dockerfile:# Start with CUDA base image
cuda-vnc/cuda_v7.5/Dockerfile:FROM nvidia/cuda:7.5-devel-ubuntu14.04
cuda-vnc/cuda_v7.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-vnc.svg)](https://hub.docker.com/r/kaixhin/cuda-vnc/)
cuda-vnc/cuda_v7.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-vnc.svg)](https://hub.docker.com/r/kaixhin/cuda-vnc/)
cuda-vnc/cuda_v7.5/README.md:cuda-vnc
cuda-vnc/cuda_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5](http://www.nvidia.com/object/cuda_home_new.html) + LXDE desktop + Firefox browser + TightVNC server.
cuda-vnc/cuda_v7.5/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-vnc/cuda_v7.5/README.md:Use NVIDIA Docker: ``nvidia-docker run -dP kaixhin/cuda-vnc``.
cuda-vnc/cuda_v7.5/README.md:For automatically mapping a VNC port use ``nvidia-docker run -dP kaixhin/cuda-vnc`` and `docker port <id>` to retrieve the port.
cuda-vnc/cuda_v7.5/README.md:For specifying the port manually use ``nvidia-docker run -d -p <port>:5901 kaixhin/cuda-vnc``.
cuda-vnc/cuda_v7.5/README.md:The shell can be entered as usual using ``nvidia-docker run -it kaixhin/cuda-vnc bash``.
cuda-vnc/cuda_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-vnc/cuda_v8.0/Dockerfile:# Start with CUDA base image
cuda-vnc/cuda_v8.0/Dockerfile:FROM nvidia/cuda:8.0-devel-ubuntu14.04
cuda-vnc/cuda_v8.0/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-vnc.svg)](https://hub.docker.com/r/kaixhin/cuda-vnc/)
cuda-vnc/cuda_v8.0/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-vnc.svg)](https://hub.docker.com/r/kaixhin/cuda-vnc/)
cuda-vnc/cuda_v8.0/README.md:cuda-vnc
cuda-vnc/cuda_v8.0/README.md:Ubuntu Core 14.04 + [CUDA 8.0](http://www.nvidia.com/object/cuda_home_new.html) + LXDE desktop + Firefox browser + TightVNC server.
cuda-vnc/cuda_v8.0/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-vnc/cuda_v8.0/README.md:Use NVIDIA Docker: ``nvidia-docker run -dP kaixhin/cuda-vnc``.
cuda-vnc/cuda_v8.0/README.md:For automatically mapping a VNC port use ``nvidia-docker run -dP kaixhin/cuda-vnc`` and `docker port <id>` to retrieve the port.
cuda-vnc/cuda_v8.0/README.md:For specifying the port manually use ``nvidia-docker run -d -p <port>:5901 kaixhin/cuda-vnc``.
cuda-vnc/cuda_v8.0/README.md:The shell can be entered as usual using ``nvidia-docker run -it kaixhin/cuda-vnc bash``.
cuda-vnc/cuda_v8.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
fgmachine/Dockerfile:# Create NVIDIA Docker init file
fgmachine/Dockerfile:RUN touch /etc/init.d/nvidia-docker && \
fgmachine/Dockerfile:  chmod +x /etc/init.d/nvidia-docker && \
fgmachine/Dockerfile:# Install NVIDIA Docker
fgmachine/Dockerfile:  wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0-rc.3/nvidia-docker_1.0.0.rc.3-1_amd64.deb && \
fgmachine/Dockerfile:  dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
cuda-keras/cuda_v7.5/Dockerfile:# Start with CUDA Theano base image
cuda-keras/cuda_v7.5/Dockerfile:FROM kaixhin/cuda-theano:7.5
cuda-keras/cuda_v7.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-keras.svg)](https://hub.docker.com/r/kaixhin/cuda-keras/)
cuda-keras/cuda_v7.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-keras.svg)](https://hub.docker.com/r/kaixhin/cuda-keras/)
cuda-keras/cuda_v7.5/README.md:cuda-keras
cuda-keras/cuda_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v5](https://developer.nvidia.com/cuDNN) + [Keras](http://keras.io/).
cuda-keras/cuda_v7.5/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-keras/cuda_v7.5/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-keras``.
cuda-keras/cuda_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-keras/cuda_v8.0/Dockerfile:# Start with CUDA Theano base image
cuda-keras/cuda_v8.0/Dockerfile:FROM kaixhin/cuda-theano:8.0
cuda-keras/cuda_v8.0/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-keras.svg)](https://hub.docker.com/r/kaixhin/cuda-keras/)
cuda-keras/cuda_v8.0/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-keras.svg)](https://hub.docker.com/r/kaixhin/cuda-keras/)
cuda-keras/cuda_v8.0/README.md:cuda-keras
cuda-keras/cuda_v8.0/README.md:Ubuntu Core 14.04 + [CUDA 8.0](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v5](https://developer.nvidia.com/cuDNN) + [Keras](http://keras.io/).
cuda-keras/cuda_v8.0/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-keras/cuda_v8.0/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-keras``.
cuda-keras/cuda_v8.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
README.md:Nearly all images are based on Ubuntu Core 14.04 LTS, built with minimising size/layers and [best practices](https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/) in mind. Dependencies are indicated left to right e.g. cuda-vnc is VNC built on top of CUDA. Explicit dependencies are excluded.
README.md:- [DIGITS](https://github.com/NVIDIA/DIGITS)
README.md:General information on running desktop applications with Docker can be found [in this blog post](https://blog.jessfraz.com/post/docker-containers-on-the-desktop/). You probably will also need to configure the X server host (`xhost`) to [give access](http://wiki.ros.org/docker/Tutorials/GUI). For hardware acceleration on Linux, it is possible to use `nvidia-docker` (with an image built for NVIDIA Docker), although OpenGL is [not fully supported](https://github.com/NVIDIA/nvidia-docker/issues/11).
README.md:CUDA
README.md:Many images rely on [CUDA](http://www.nvidia.com/object/cuda_home_new.html). These images are versioned with the corresponding tags, e.g. "8.0" and "7.5", on the Docker Hub.
README.md:These images need to be run on an Ubuntu host OS with [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) installed. The driver requirements can be found on the [NVIDIA Docker wiki](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements).
README.md:`kaixhin/cuda` and `kaixhin/cudnn` have now been **deprecated** in favour of the official solution ([`nvidia/cuda`](https://hub.docker.com/r/nvidia/cuda/)).
README.md:- [CUDA](https://github.com/tleyden/docker)
cuda-theano/cuda_v7.5/Dockerfile:FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
cuda-theano/cuda_v7.5/Dockerfile:# Set CUDA_ROOT
cuda-theano/cuda_v7.5/Dockerfile:ENV CUDA_ROOT /usr/local/cuda/bin
cuda-theano/cuda_v7.5/Dockerfile:# Clone libgpuarray repo and move into it
cuda-theano/cuda_v7.5/Dockerfile:RUN cd /root && git clone https://github.com/Theano/libgpuarray.git && cd libgpuarray && \
cuda-theano/cuda_v7.5/Dockerfile:# Install pygpu
cuda-theano/cuda_v7.5/Dockerfile:RUN cd /root/libgpuarray && \
cuda-theano/cuda_v7.5/Dockerfile:# Set up .theanorc for CUDA
cuda-theano/cuda_v7.5/Dockerfile:RUN echo "[global]\ndevice=cuda\nfloatX=float32\noptimizer_including=cudnn\n[lib]\ncnmem=0.1\n[nvcc]\nfastmath=True\n[dnn]\nenabled=True\ninclude_path=/usr/local/cuda/include\nlibrary_path=/usr/local/cuda/lib64" > /root/.theanorc
cuda-theano/cuda_v7.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-theano.svg)](https://hub.docker.com/r/kaixhin/cuda-theano/)
cuda-theano/cuda_v7.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-theano.svg)](https://hub.docker.com/r/kaixhin/cuda-theano/)
cuda-theano/cuda_v7.5/README.md:cuda-theano
cuda-theano/cuda_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v5](https://developer.nvidia.com/cuDNN) + [Theano](http://www.deeplearning.net/software/theano/).
cuda-theano/cuda_v7.5/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-theano/cuda_v7.5/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-theano``.
cuda-theano/cuda_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-theano/cuda_v8.0/Dockerfile:FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04
cuda-theano/cuda_v8.0/Dockerfile:# Set CUDA_ROOT
cuda-theano/cuda_v8.0/Dockerfile:ENV CUDA_ROOT /usr/local/cuda/bin
cuda-theano/cuda_v8.0/Dockerfile:# Clone libgpuarray repo and move into it
cuda-theano/cuda_v8.0/Dockerfile:RUN cd /root && git clone https://github.com/Theano/libgpuarray.git && cd libgpuarray && \
cuda-theano/cuda_v8.0/Dockerfile:# Install pygpu
cuda-theano/cuda_v8.0/Dockerfile:RUN cd /root/libgpuarray && \
cuda-theano/cuda_v8.0/Dockerfile:# Set up .theanorc for CUDA
cuda-theano/cuda_v8.0/Dockerfile:RUN echo "[global]\ndevice=cuda\nfloatX=float32\noptimizer_including=cudnn\n[lib]\ncnmem=0.1\n[nvcc]\nfastmath=True\n[dnn]\nenabled=True\ninclude_path=/usr/local/cuda/include\nlibrary_path=/usr/local/cuda/lib64" > /root/.theanorc
cuda-theano/cuda_v8.0/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-theano.svg)](https://hub.docker.com/r/kaixhin/cuda-theano/)
cuda-theano/cuda_v8.0/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-theano.svg)](https://hub.docker.com/r/kaixhin/cuda-theano/)
cuda-theano/cuda_v8.0/README.md:cuda-theano
cuda-theano/cuda_v8.0/README.md:Ubuntu Core 14.04 + [CUDA 8.0](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v6](https://developer.nvidia.com/cuDNN) + [Theano](http://www.deeplearning.net/software/theano/).
cuda-theano/cuda_v8.0/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-theano/cuda_v8.0/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-theano``.
cuda-theano/cuda_v8.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-caffe/cuda_v7.5/Dockerfile:# Start with CUDA Caffe dependencies
cuda-caffe/cuda_v7.5/Dockerfile:FROM kaixhin/cuda-caffe-deps:7.5
cuda-caffe/cuda_v7.5/deps/Dockerfile:FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
cuda-caffe/cuda_v7.5/deps/Dockerfile:# Install NCCL for multi-GPU communication
cuda-caffe/cuda_v7.5/deps/Dockerfile:RUN apt-get install -y cuda-ld-conf-7-5 && \
cuda-caffe/cuda_v7.5/deps/Dockerfile:  wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda7.5/libnccl1_1.2.3-1.cuda7.5_amd64.deb && \
cuda-caffe/cuda_v7.5/deps/Dockerfile:  dpkg -i libnccl1_1.2.3-1.cuda7.5_amd64.deb && \
cuda-caffe/cuda_v7.5/deps/Dockerfile:  rm libnccl1_1.2.3-1.cuda7.5_amd64.deb && \
cuda-caffe/cuda_v7.5/deps/Dockerfile:  wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda7.5/libnccl-dev_1.2.3-1.cuda7.5_amd64.deb && \
cuda-caffe/cuda_v7.5/deps/Dockerfile:  dpkg -i libnccl-dev_1.2.3-1.cuda7.5_amd64.deb && \
cuda-caffe/cuda_v7.5/deps/Dockerfile:  rm libnccl-dev_1.2.3-1.cuda7.5_amd64.deb
cuda-caffe/cuda_v7.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-caffe.svg)](https://hub.docker.com/r/kaixhin/cuda-caffe/)
cuda-caffe/cuda_v7.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-caffe.svg)](https://hub.docker.com/r/kaixhin/cuda-caffe/)
cuda-caffe/cuda_v7.5/README.md:cuda-caffe
cuda-caffe/cuda_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v5](https://developer.nvidia.com/cuDNN) + [Caffe](http://caffe.berkeleyvision.org/). Includes Python interface.
cuda-caffe/cuda_v7.5/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-caffe/cuda_v7.5/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-caffe``.
cuda-caffe/cuda_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-caffe/cuda_v8.0/Dockerfile:# Start with CUDA Caffe dependencies
cuda-caffe/cuda_v8.0/Dockerfile:FROM kaixhin/cuda-caffe-deps:8.0
cuda-caffe/cuda_v8.0/deps/Dockerfile:FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04
cuda-caffe/cuda_v8.0/deps/Dockerfile:# Install NCCL for multi-GPU communication
cuda-caffe/cuda_v8.0/deps/Dockerfile:RUN wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda8.0/libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
cuda-caffe/cuda_v8.0/deps/Dockerfile:  dpkg -i libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
cuda-caffe/cuda_v8.0/deps/Dockerfile:  rm libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
cuda-caffe/cuda_v8.0/deps/Dockerfile:  wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda8.0/libnccl-dev_1.2.3-1.cuda8.0_amd64.deb && \
cuda-caffe/cuda_v8.0/deps/Dockerfile:  dpkg -i libnccl-dev_1.2.3-1.cuda8.0_amd64.deb && \
cuda-caffe/cuda_v8.0/deps/Dockerfile:  rm libnccl-dev_1.2.3-1.cuda8.0_amd64.deb
cuda-caffe/cuda_v8.0/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-caffe.svg)](https://hub.docker.com/r/kaixhin/cuda-caffe/)
cuda-caffe/cuda_v8.0/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-caffe.svg)](https://hub.docker.com/r/kaixhin/cuda-caffe/)
cuda-caffe/cuda_v8.0/README.md:cuda-caffe
cuda-caffe/cuda_v8.0/README.md:Ubuntu Core 14.04 + [CUDA 8.0](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v6](https://developer.nvidia.com/cuDNN) + [Caffe](http://caffe.berkeleyvision.org/). Includes Python interface.
cuda-caffe/cuda_v8.0/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-caffe/cuda_v8.0/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-caffe``.
cuda-caffe/cuda_v8.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-neon/cuda_v7.5/Dockerfile:# Start with CUDA base image
cuda-neon/cuda_v7.5/Dockerfile:FROM nvidia/cuda:7.5-devel-ubuntu14.04
cuda-neon/cuda_v7.5/Dockerfile:  make sysinstall HAS_GPU=true
cuda-neon/cuda_v7.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-neon.svg)](https://hub.docker.com/r/kaixhin/cuda-neon/)
cuda-neon/cuda_v7.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-neon.svg)](https://hub.docker.com/r/kaixhin/cuda-neon/)
cuda-neon/cuda_v7.5/README.md:cuda-neon
cuda-neon/cuda_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5](http://www.nvidia.com/object/cuda_home_new.html) + [neon](http://neon.nervanasys.com/).
cuda-neon/cuda_v7.5/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-neon/cuda_v7.5/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-neon``.
cuda-neon/cuda_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-neon/cuda_v8.0/Dockerfile:# Start with CUDA base image
cuda-neon/cuda_v8.0/Dockerfile:FROM nvidia/cuda:8.0-devel-ubuntu14.04
cuda-neon/cuda_v8.0/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-neon.svg)](https://hub.docker.com/r/kaixhin/cuda-neon/)
cuda-neon/cuda_v8.0/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-neon.svg)](https://hub.docker.com/r/kaixhin/cuda-neon/)
cuda-neon/cuda_v8.0/README.md:cuda-neon
cuda-neon/cuda_v8.0/README.md:Ubuntu Core 14.04 + [CUDA 8.0](http://www.nvidia.com/object/cuda_home_new.html) + [neon](http://neon.nervanasys.com/).
cuda-neon/cuda_v8.0/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-neon/cuda_v8.0/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-neon``.
cuda-neon/cuda_v8.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-ssh/cuda_v9.1/Dockerfile:# Start with CUDA base image
cuda-ssh/cuda_v9.1/Dockerfile:FROM nvidia/cuda:9.1-devel-ubuntu16.04
cuda-ssh/cuda_v9.1/Dockerfile:# Install OpenSSH, X server and libgtk (for NVIDIA Visual Profiler)
cuda-ssh/cuda_v9.1/Dockerfile:# Add CUDA back to path during SSH
cuda-ssh/cuda_v9.1/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-ssh.svg)](https://hub.docker.com/r/kaixhin/cuda-ssh/)
cuda-ssh/cuda_v9.1/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-ssh.svg)](https://hub.docker.com/r/kaixhin/cuda-ssh/)
cuda-ssh/cuda_v9.1/README.md:cuda-ssh
cuda-ssh/cuda_v9.1/README.md:Ubuntu Core 16.04 + [CUDA 9.1](http://www.nvidia.com/object/cuda_home_new.html) + SSH server + X server (for NVIDIA Visual Profiler).
cuda-ssh/cuda_v9.1/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-ssh/cuda_v9.1/README.md:Use NVIDIA Docker: ``nvidia-docker run -dP kaixhin/cuda-ssh``.
cuda-ssh/cuda_v9.1/README.md:For automatically mapping a SSH port use ``nvidia-docker run -dP kaixhin/cuda-ssh`` and `docker port <id>` to retrieve the port.
cuda-ssh/cuda_v9.1/README.md:For specifying the port manually use ``nvidia-docker run -d -p <port>:22 kaixhin/cuda-ssh``.
cuda-ssh/cuda_v9.1/README.md:The shell can be entered as usual using ``nvidia-docker run -it kaixhin/cuda-ssh bash``.
cuda-ssh/cuda_v9.1/README.md:The NVIDIA Visual Profiler (`nvvp`) can be accessed with an X client, after having run ssh with the `-X` flag.
cuda-ssh/cuda_v9.1/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-ssh/cuda_v7.5/Dockerfile:# Start with CUDA base image
cuda-ssh/cuda_v7.5/Dockerfile:FROM nvidia/cuda:7.5-devel-ubuntu14.04
cuda-ssh/cuda_v7.5/Dockerfile:# Install OpenSSH, X server and libgtk (for NVIDIA Visual Profiler)
cuda-ssh/cuda_v7.5/Dockerfile:# Add CUDA back to path during SSH
cuda-ssh/cuda_v7.5/Dockerfile:RUN echo "export PATH=$PATH:/usr/local/cuda/bin" >> /etc/profile
cuda-ssh/cuda_v7.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-ssh.svg)](https://hub.docker.com/r/kaixhin/cuda-ssh/)
cuda-ssh/cuda_v7.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-ssh.svg)](https://hub.docker.com/r/kaixhin/cuda-ssh/)
cuda-ssh/cuda_v7.5/README.md:cuda-ssh
cuda-ssh/cuda_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5](http://www.nvidia.com/object/cuda_home_new.html) + SSH server + X server (for NVIDIA Visual Profiler).
cuda-ssh/cuda_v7.5/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-ssh/cuda_v7.5/README.md:Use NVIDIA Docker: ``nvidia-docker run -dP kaixhin/cuda-ssh``.
cuda-ssh/cuda_v7.5/README.md:For automatically mapping a SSH port use ``nvidia-docker run -dP kaixhin/cuda-ssh`` and `docker port <id>` to retrieve the port.
cuda-ssh/cuda_v7.5/README.md:For specifying the port manually use ``nvidia-docker run -d -p <port>:22 kaixhin/cuda-ssh``.
cuda-ssh/cuda_v7.5/README.md:The shell can be entered as usual using ``nvidia-docker run -it kaixhin/cuda-ssh bash``.
cuda-ssh/cuda_v7.5/README.md:The NVIDIA Visual Profiler (`nvvp`) can be accessed with an X client, after having run ssh with the `-X` flag.
cuda-ssh/cuda_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-ssh/cuda_v8.0/Dockerfile:# Start with CUDA base image
cuda-ssh/cuda_v8.0/Dockerfile:FROM nvidia/cuda:8.0-devel-ubuntu14.04
cuda-ssh/cuda_v8.0/Dockerfile:# Install OpenSSH, X server and libgtk (for NVIDIA Visual Profiler)
cuda-ssh/cuda_v8.0/Dockerfile:# Add CUDA back to path during SSH
cuda-ssh/cuda_v8.0/Dockerfile:RUN echo "export PATH=$PATH:/usr/local/cuda/bin" >> /etc/profile
cuda-ssh/cuda_v8.0/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-ssh.svg)](https://hub.docker.com/r/kaixhin/cuda-ssh/)
cuda-ssh/cuda_v8.0/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-ssh.svg)](https://hub.docker.com/r/kaixhin/cuda-ssh/)
cuda-ssh/cuda_v8.0/README.md:cuda-ssh
cuda-ssh/cuda_v8.0/README.md:Ubuntu Core 14.04 + [CUDA 8.0](http://www.nvidia.com/object/cuda_home_new.html) + SSH server + X server (for NVIDIA Visual Profiler).
cuda-ssh/cuda_v8.0/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-ssh/cuda_v8.0/README.md:Use NVIDIA Docker: ``nvidia-docker run -dP kaixhin/cuda-ssh``.
cuda-ssh/cuda_v8.0/README.md:For automatically mapping a SSH port use ``nvidia-docker run -dP kaixhin/cuda-ssh`` and `docker port <id>` to retrieve the port.
cuda-ssh/cuda_v8.0/README.md:For specifying the port manually use ``nvidia-docker run -d -p <port>:22 kaixhin/cuda-ssh``.
cuda-ssh/cuda_v8.0/README.md:The shell can be entered as usual using ``nvidia-docker run -it kaixhin/cuda-ssh bash``.
cuda-ssh/cuda_v8.0/README.md:The NVIDIA Visual Profiler (`nvvp`) can be accessed with an X client, after having run ssh with the `-X` flag.
cuda-ssh/cuda_v8.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-ssh/cuda_v9.2/Dockerfile:# Start with CUDA base image
cuda-ssh/cuda_v9.2/Dockerfile:FROM nvidia/cuda:9.2-devel-ubuntu16.04
cuda-ssh/cuda_v9.2/Dockerfile:# Install OpenSSH, X server and libgtk (for NVIDIA Visual Profiler)
cuda-ssh/cuda_v9.2/Dockerfile:# Add CUDA back to path during SSH
cuda-ssh/cuda_v9.2/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-ssh.svg)](https://hub.docker.com/r/kaixhin/cuda-ssh/)
cuda-ssh/cuda_v9.2/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-ssh.svg)](https://hub.docker.com/r/kaixhin/cuda-ssh/)
cuda-ssh/cuda_v9.2/README.md:cuda-ssh
cuda-ssh/cuda_v9.2/README.md:Ubuntu Core 16.04 + [CUDA 9.2](http://www.nvidia.com/object/cuda_home_new.html) + SSH server + X server (for NVIDIA Visual Profiler).
cuda-ssh/cuda_v9.2/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-ssh/cuda_v9.2/README.md:Use NVIDIA Docker: ``nvidia-docker run -dP kaixhin/cuda-ssh``.
cuda-ssh/cuda_v9.2/README.md:For automatically mapping a SSH port use ``nvidia-docker run -dP kaixhin/cuda-ssh`` and `docker port <id>` to retrieve the port.
cuda-ssh/cuda_v9.2/README.md:For specifying the port manually use ``nvidia-docker run -d -p <port>:22 kaixhin/cuda-ssh``.
cuda-ssh/cuda_v9.2/README.md:The shell can be entered as usual using ``nvidia-docker run -it kaixhin/cuda-ssh bash``.
cuda-ssh/cuda_v9.2/README.md:The NVIDIA Visual Profiler (`nvvp`) can be accessed with an X client, after having run ssh with the `-X` flag.
cuda-ssh/cuda_v9.2/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-ssh/cuda_v10.1/Dockerfile:# Start with CUDA base image
cuda-ssh/cuda_v10.1/Dockerfile:FROM nvidia/cuda:10.1-devel-ubuntu16.04
cuda-ssh/cuda_v10.1/Dockerfile:# Install OpenSSH, X server and libgtk (for NVIDIA Visual Profiler)
cuda-ssh/cuda_v10.1/Dockerfile:# Add CUDA back to path during SSH
cuda-ssh/cuda_v10.1/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-ssh.svg)](https://hub.docker.com/r/kaixhin/cuda-ssh/)
cuda-ssh/cuda_v10.1/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-ssh.svg)](https://hub.docker.com/r/kaixhin/cuda-ssh/)
cuda-ssh/cuda_v10.1/README.md:cuda-ssh
cuda-ssh/cuda_v10.1/README.md:Ubuntu Core 16.04 + [CUDA 10.1](http://www.nvidia.com/object/cuda_home_new.html) + SSH server + X server (for NVIDIA Visual Profiler).
cuda-ssh/cuda_v10.1/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-ssh/cuda_v10.1/README.md:Use NVIDIA Docker: ``nvidia-docker run -dP kaixhin/cuda-ssh``.
cuda-ssh/cuda_v10.1/README.md:For automatically mapping a SSH port use ``nvidia-docker run -dP kaixhin/cuda-ssh`` and `docker port <id>` to retrieve the port.
cuda-ssh/cuda_v10.1/README.md:For specifying the port manually use ``nvidia-docker run -d -p <port>:22 kaixhin/cuda-ssh``.
cuda-ssh/cuda_v10.1/README.md:The shell can be entered as usual using ``nvidia-docker run -it kaixhin/cuda-ssh bash``.
cuda-ssh/cuda_v10.1/README.md:The NVIDIA Visual Profiler (`nvvp`) can be accessed with an X client, after having run ssh with the `-X` flag.
cuda-ssh/cuda_v10.1/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-brainstorm/cuda_v7.5/Dockerfile:# Start with CUDA base image
cuda-brainstorm/cuda_v7.5/Dockerfile:FROM nvidia/cuda:7.5-devel-ubuntu14.04
cuda-brainstorm/cuda_v7.5/Dockerfile:# Install CUDA requirements
cuda-brainstorm/cuda_v7.5/Dockerfile:  pip install -r pycuda_requirements.txt && \
cuda-brainstorm/cuda_v7.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-brainstorm.svg)](https://hub.docker.com/r/kaixhin/cuda-brainstorm/)
cuda-brainstorm/cuda_v7.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-brainstorm.svg)](https://hub.docker.com/r/kaixhin/cuda-brainstorm/)
cuda-brainstorm/cuda_v7.5/README.md:cuda-brainstorm
cuda-brainstorm/cuda_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5](http://www.nvidia.com/object/cuda_home_new.html) + [Brainstorm](https://github.com/IDSIA/brainstorm).
cuda-brainstorm/cuda_v7.5/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-brainstorm/cuda_v7.5/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-brainstorm``.
cuda-brainstorm/cuda_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-brainstorm/cuda_v8.0/Dockerfile:# Start with CUDA base image
cuda-brainstorm/cuda_v8.0/Dockerfile:FROM nvidia/cuda:8.0-devel-ubuntu14.04
cuda-brainstorm/cuda_v8.0/Dockerfile:# Install CUDA requirements
cuda-brainstorm/cuda_v8.0/Dockerfile:  pip install -r pycuda_requirements.txt && \
cuda-brainstorm/cuda_v8.0/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-brainstorm.svg)](https://hub.docker.com/r/kaixhin/cuda-brainstorm/)
cuda-brainstorm/cuda_v8.0/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-brainstorm.svg)](https://hub.docker.com/r/kaixhin/cuda-brainstorm/)
cuda-brainstorm/cuda_v8.0/README.md:cuda-brainstorm
cuda-brainstorm/cuda_v8.0/README.md:Ubuntu Core 14.04 + [CUDA 8.0](http://www.nvidia.com/object/cuda_home_new.html) + [Brainstorm](https://github.com/IDSIA/brainstorm).
cuda-brainstorm/cuda_v8.0/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-brainstorm/cuda_v8.0/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-brainstorm``.
cuda-brainstorm/cuda_v8.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-lasagne/cuda_v7.5/Dockerfile:# Start with CUDA Theano base image
cuda-lasagne/cuda_v7.5/Dockerfile:FROM kaixhin/cuda-theano:7.5
cuda-lasagne/cuda_v7.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-lasagne.svg)](https://hub.docker.com/r/kaixhin/cuda-lasagne/)
cuda-lasagne/cuda_v7.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-lasagne.svg)](https://hub.docker.com/r/kaixhin/cuda-lasagne/)
cuda-lasagne/cuda_v7.5/README.md:cuda-lasagne
cuda-lasagne/cuda_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v5](https://developer.nvidia.com/cuDNN) + [Lasagne](http://lasagne.readthedocs.org/).
cuda-lasagne/cuda_v7.5/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-lasagne/cuda_v7.5/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-lasagne``.
cuda-lasagne/cuda_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-lasagne/cuda_v8.0/Dockerfile:# Start with CUDA Theano base image
cuda-lasagne/cuda_v8.0/Dockerfile:FROM kaixhin/cuda-theano:8.0
cuda-lasagne/cuda_v8.0/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-lasagne.svg)](https://hub.docker.com/r/kaixhin/cuda-lasagne/)
cuda-lasagne/cuda_v8.0/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-lasagne.svg)](https://hub.docker.com/r/kaixhin/cuda-lasagne/)
cuda-lasagne/cuda_v8.0/README.md:cuda-lasagne
cuda-lasagne/cuda_v8.0/README.md:Ubuntu Core 14.04 + [CUDA 8.0](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v5](https://developer.nvidia.com/cuDNN) + [Lasagne](http://lasagne.readthedocs.org/).
cuda-lasagne/cuda_v8.0/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-lasagne/cuda_v8.0/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-lasagne``.
cuda-lasagne/cuda_v8.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
digits/Dockerfile:# Install NVIDIA Caffe 0.15
digits/Dockerfile:RUN git clone https://github.com/NVIDIA/caffe.git $CAFFE_ROOT -b 'caffe-0.15' && cd $CAFFE_ROOT && \
digits/Dockerfile:RUN git clone https://github.com/NVIDIA/DIGITS.git $DIGITS_ROOT && cd $DIGITS_ROOT && \
digits/README.md:Ubuntu Core 16.04 + [Caffe](http://caffe.berkeleyvision.org/) (NVIDIA fork) + [DIGITS](https://developer.nvidia.com/digits) (CPU-only).
cuda-mxnet/cuda_v7.5/Dockerfile:FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
cuda-mxnet/cuda_v7.5/Dockerfile:  make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
cuda-mxnet/cuda_v7.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-mxnet.svg)](https://hub.docker.com/r/kaixhin/cuda-mxnet/)
cuda-mxnet/cuda_v7.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-mxnet.svg)](https://hub.docker.com/r/kaixhin/cuda-mxnet/)
cuda-mxnet/cuda_v7.5/README.md:cuda-mxnet
cuda-mxnet/cuda_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v5](https://developer.nvidia.com/cuDNN) + [MXNet](https://mxnet.incubator.apache.org/index.html).
cuda-mxnet/cuda_v7.5/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-mxnet/cuda_v7.5/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-mxnet``.
cuda-mxnet/cuda_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda-mxnet/cuda_v8.0/Dockerfile:FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04
cuda-mxnet/cuda_v8.0/Dockerfile:  make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
cuda-mxnet/cuda_v8.0/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda-mxnet.svg)](https://hub.docker.com/r/kaixhin/cuda-mxnet/)
cuda-mxnet/cuda_v8.0/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda-mxnet.svg)](https://hub.docker.com/r/kaixhin/cuda-mxnet/)
cuda-mxnet/cuda_v8.0/README.md:cuda-mxnet
cuda-mxnet/cuda_v8.0/README.md:Ubuntu Core 14.04 + [CUDA 8.0](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v6](https://developer.nvidia.com/cuDNN) + [MXNet](https://mxnet.incubator.apache.org/index.html).
cuda-mxnet/cuda_v8.0/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) - see [requirements](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) for more details.
cuda-mxnet/cuda_v8.0/README.md:Use NVIDIA Docker: ``nvidia-docker run -it kaixhin/cuda-mxnet``.
cuda-mxnet/cuda_v8.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda/cuda_v6.5/Dockerfile:  wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run -o /dev/null && \
cuda/cuda_v6.5/Dockerfile:  chmod +x cuda_*_linux_64.run && ./cuda_*_linux_64.run -extract=`pwd` && \
cuda/cuda_v6.5/Dockerfile:# Install CUDA drivers (silent, no kernel)
cuda/cuda_v6.5/Dockerfile:  ./NVIDIA-Linux-x86_64-*.run -s --no-kernel-module && \
cuda/cuda_v6.5/Dockerfile:  ./cuda-linux64-rel-*.run -noprompt | cat > /dev/null && \
cuda/cuda_v6.5/Dockerfile:ENV PATH=/usr/local/cuda/bin:$PATH \
cuda/cuda_v6.5/Dockerfile:  LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cuda/cuda_v6.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda.svg)](https://hub.docker.com/r/kaixhin/cuda/)
cuda/cuda_v6.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda.svg)](https://hub.docker.com/r/kaixhin/cuda/)
cuda/cuda_v6.5/README.md:cuda
cuda/cuda_v6.5/README.md:**DEPRECATED:** Please use [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).
cuda/cuda_v6.5/README.md:Ubuntu Core 14.04 + [CUDA 6.5.14](http://www.nvidia.com/object/cuda_home_new.html).
cuda/cuda_v6.5/README.md:- Host with corresponding CUDA drivers (v. 340.29) installed for the kernel module.
cuda/cuda_v6.5/README.md:The container must have all NVIDIA devices attached to it for CUDA to work properly.
cuda/cuda_v6.5/README.md:Therefore the command will be as such: `docker run -it --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia0 kaixhin/cuda`.
cuda/cuda_v6.5/README.md:With 4 GPUs this would also have to include `--device /dev/nvidia1 --device /dev/nvidia2 --device /dev/nvidia3`.
cuda/cuda_v6.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda/cuda_v6.5/README.md:As the image is intended to be lightweight, the CUDA samples were not installed. If you wish to experiment with the samples you will need to install them yourself. The steps are as below:
cuda/cuda_v6.5/README.md:wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run
cuda/cuda_v6.5/README.md:chmod +x cuda_*_linux.run
cuda/cuda_v6.5/README.md:./cuda_*_linux.run -extract=`pwd`
cuda/cuda_v6.5/README.md:./cuda-samples-linux-*.run -noprompt
cuda/cuda_v6.5/README.md:Please note that you may need to install other packages in order to compile some of the CUDA samples.
cuda/cuda_v7.0/Dockerfile:  wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run -o /dev/null && \
cuda/cuda_v7.0/Dockerfile:  chmod +x cuda_*_linux.run && ./cuda_*_linux.run -extract=`pwd` && \
cuda/cuda_v7.0/Dockerfile:# Install CUDA drivers (silent, no kernel)
cuda/cuda_v7.0/Dockerfile:  ./NVIDIA-Linux-x86_64-*.run -s --no-kernel-module && \
cuda/cuda_v7.0/Dockerfile:  ./cuda-linux64-rel-*.run -noprompt | cat > /dev/null && \
cuda/cuda_v7.0/Dockerfile:ENV PATH=/usr/local/cuda/bin:$PATH \
cuda/cuda_v7.0/Dockerfile:  LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cuda/cuda_v7.0/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda.svg)](https://hub.docker.com/r/kaixhin/cuda/)
cuda/cuda_v7.0/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda.svg)](https://hub.docker.com/r/kaixhin/cuda/)
cuda/cuda_v7.0/README.md:cuda
cuda/cuda_v7.0/README.md:**DEPRECATED:** Please use [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).
cuda/cuda_v7.0/README.md:Ubuntu Core 14.04 + [CUDA 7.0.28](http://www.nvidia.com/object/cuda_home_new.html).
cuda/cuda_v7.0/README.md:- Host with corresponding CUDA drivers (v. 346.46) installed for the kernel module.
cuda/cuda_v7.0/README.md:The container must have all NVIDIA devices attached to it for CUDA to work properly.
cuda/cuda_v7.0/README.md:Therefore the command will be as such: `docker run -it --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia0 kaixhin/cuda`.
cuda/cuda_v7.0/README.md:With 4 GPUs this would also have to include `--device /dev/nvidia1 --device /dev/nvidia2 --device /dev/nvidia3`.
cuda/cuda_v7.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda/cuda_v7.0/README.md:As the image is intended to be lightweight, the CUDA samples were not installed. If you wish to experiment with the samples you will need to install them yourself. The steps are as below:
cuda/cuda_v7.0/README.md:wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run
cuda/cuda_v7.0/README.md:chmod +x cuda_*_linux.run
cuda/cuda_v7.0/README.md:./cuda_*_linux.run -extract=`pwd`
cuda/cuda_v7.0/README.md:./cuda-samples-linux-*.run -noprompt
cuda/cuda_v7.0/README.md:Please note that you may need to install other packages in order to compile some of the CUDA samples.
cuda/cuda_v7.5/Dockerfile:  wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run -o /dev/null && \
cuda/cuda_v7.5/Dockerfile:  chmod +x cuda_*_linux.run && ./cuda_*_linux.run -extract=`pwd` && \
cuda/cuda_v7.5/Dockerfile:# Install CUDA drivers (silent, no kernel)
cuda/cuda_v7.5/Dockerfile:  ./NVIDIA-Linux-x86_64-*.run -s --no-kernel-module && \
cuda/cuda_v7.5/Dockerfile:  ./cuda-linux64-rel-*.run -noprompt | cat > /dev/null && \
cuda/cuda_v7.5/Dockerfile:ENV PATH=/usr/local/cuda/bin:$PATH \
cuda/cuda_v7.5/Dockerfile:  LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cuda/cuda_v7.5/README.md:[![Docker Pulls](https://img.shields.io/docker/pulls/kaixhin/cuda.svg)](https://hub.docker.com/r/kaixhin/cuda/)
cuda/cuda_v7.5/README.md:[![Docker Stars](https://img.shields.io/docker/stars/kaixhin/cuda.svg)](https://hub.docker.com/r/kaixhin/cuda/)
cuda/cuda_v7.5/README.md:cuda
cuda/cuda_v7.5/README.md:**DEPRECATED:** Please use [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).
cuda/cuda_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5.18](http://www.nvidia.com/object/cuda_home_new.html).
cuda/cuda_v7.5/README.md:- Host with corresponding CUDA drivers (v. 352.39) installed for the kernel module.
cuda/cuda_v7.5/README.md:The container must have all NVIDIA devices attached to it for CUDA to work properly.
cuda/cuda_v7.5/README.md:Therefore the command will be as such: `docker run -it --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia0 kaixhin/cuda`.
cuda/cuda_v7.5/README.md:With 4 GPUs this would also have to include `--device /dev/nvidia1 --device /dev/nvidia2 --device /dev/nvidia3`.
cuda/cuda_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cuda/cuda_v7.5/README.md:As the image is intended to be lightweight, the CUDA samples were not installed. If you wish to experiment with the samples you will need to install them yourself. The steps are as below:
cuda/cuda_v7.5/README.md:wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run
cuda/cuda_v7.5/README.md:chmod +x cuda_*_linux.run
cuda/cuda_v7.5/README.md:./cuda_*_linux.run -extract=`pwd`
cuda/cuda_v7.5/README.md:./cuda-samples-linux-*.run -noprompt
cuda/cuda_v7.5/README.md:Please note that you may need to install other packages in order to compile some of the CUDA samples.
cudnn/cudnn_v7.5/Dockerfile:# Start with CUDA base image
cudnn/cudnn_v7.5/Dockerfile:FROM kaixhin/cuda:latest
cudnn/cudnn_v7.5/Dockerfile:# Install CUDA repo (needed for cuDNN)
cudnn/cudnn_v7.5/Dockerfile:ENV CUDA_REPO_PKG=cuda-repo-ubuntu1404_7.5-18_amd64.deb
cudnn/cudnn_v7.5/Dockerfile:RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG && \
cudnn/cudnn_v7.5/Dockerfile:  dpkg -i $CUDA_REPO_PKG
cudnn/cudnn_v7.5/Dockerfile:ENV ML_REPO_PKG=nvidia-machine-learning-repo_4.0-2_amd64.deb
cudnn/cudnn_v7.5/Dockerfile:RUN wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/$ML_REPO_PKG && \
cudnn/cudnn_v7.5/README.md:**DEPRECATED:** Please use [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).
cudnn/cudnn_v7.5/README.md:Ubuntu Core 14.04 + [CUDA 7.5.18](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v4](https://developer.nvidia.com/cuDNN).
cudnn/cudnn_v7.5/README.md:- Host with corresponding CUDA drivers (v. 352.39) installed for the kernel module.
cudnn/cudnn_v7.5/README.md:The container must have all NVIDIA devices attached to it for CUDA to work properly.
cudnn/cudnn_v7.5/README.md:Therefore the command will be as such: `docker run -it --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia0 kaixhin/cudnn`.
cudnn/cudnn_v7.5/README.md:With 4 GPUs this would also have to include `--device /dev/nvidia1 --device /dev/nvidia2 --device /dev/nvidia3`.
cudnn/cudnn_v7.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cudnn/cudnn_v6.5/Dockerfile:# Start with CUDA base image
cudnn/cudnn_v6.5/Dockerfile:FROM kaixhin/cuda:6.5
cudnn/cudnn_v6.5/Dockerfile:# Install CUDA repo (needed for cuDNN)
cudnn/cudnn_v6.5/Dockerfile:ENV CUDA_REPO_PKG=cuda-repo-ubuntu1404_6.5-14_amd64.deb
cudnn/cudnn_v6.5/Dockerfile:RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG && \
cudnn/cudnn_v6.5/Dockerfile:  dpkg -i $CUDA_REPO_PKG
cudnn/cudnn_v6.5/Dockerfile:ENV ML_REPO_PKG=nvidia-machine-learning-repo_4.0-2_amd64.deb
cudnn/cudnn_v6.5/Dockerfile:RUN wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/$ML_REPO_PKG && \
cudnn/cudnn_v6.5/README.md:**DEPRECATED:** Please use [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).
cudnn/cudnn_v6.5/README.md:Ubuntu Core 14.04 + [CUDA 6.5.14](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v4](https://developer.nvidia.com/cuDNN).
cudnn/cudnn_v6.5/README.md:- Host with corresponding CUDA drivers (v. 340.29) installed for the kernel module.
cudnn/cudnn_v6.5/README.md:The container must have all NVIDIA devices attached to it for CUDA to work properly.
cudnn/cudnn_v6.5/README.md:Therefore the command will be as such: `docker run -it --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia0 kaixhin/cudnn`.
cudnn/cudnn_v6.5/README.md:With 4 GPUs this would also have to include `--device /dev/nvidia1 --device /dev/nvidia2 --device /dev/nvidia3`.
cudnn/cudnn_v6.5/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).
cudnn/cudnn_v7.0/Dockerfile:# Start with CUDA base image
cudnn/cudnn_v7.0/Dockerfile:FROM kaixhin/cuda:7.0
cudnn/cudnn_v7.0/Dockerfile:# Install CUDA repo (needed for cuDNN)
cudnn/cudnn_v7.0/Dockerfile:ENV CUDA_REPO_PKG=cuda-repo-ubuntu1404_7.0-28_amd64.deb
cudnn/cudnn_v7.0/Dockerfile:RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG && \
cudnn/cudnn_v7.0/Dockerfile:  dpkg -i $CUDA_REPO_PKG
cudnn/cudnn_v7.0/Dockerfile:ENV ML_REPO_PKG=nvidia-machine-learning-repo_4.0-2_amd64.deb
cudnn/cudnn_v7.0/Dockerfile:RUN wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/$ML_REPO_PKG && \
cudnn/cudnn_v7.0/README.md:**DEPRECATED:** Please use [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).
cudnn/cudnn_v7.0/README.md:Ubuntu Core 14.04 + [CUDA 7.0.28](http://www.nvidia.com/object/cuda_home_new.html) + [cuDNN v4](https://developer.nvidia.com/cuDNN).
cudnn/cudnn_v7.0/README.md:- Host with corresponding CUDA drivers (v. 346.46) installed for the kernel module.
cudnn/cudnn_v7.0/README.md:The container must have all NVIDIA devices attached to it for CUDA to work properly.
cudnn/cudnn_v7.0/README.md:Therefore the command will be as such: `docker run -it --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia0 kaixhin/cudnn`.
cudnn/cudnn_v7.0/README.md:With 4 GPUs this would also have to include `--device /dev/nvidia1 --device /dev/nvidia2 --device /dev/nvidia3`.
cudnn/cudnn_v7.0/README.md:For more information on CUDA on Docker, see the [repo readme](https://github.com/Kaixhin/dockerfiles#cuda).

```
