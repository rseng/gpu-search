# https://github.com/deezer/spleeter

```console
CHANGELOG.md:* Dedicated GPU package `spleeter-gpu` is not supported anymore, `spleeter` package will support both CPU and GPU hardware
README.md:> of dedicated GPU package. Please read [CHANGELOG](CHANGELOG.md) for more details.
README.md:2 stems and 4 stems models have [high performances](https://github.com/deezer/spleeter/wiki/Separation-Performances) on the [musdb](https://sigsep.github.io/datasets/musdb.html) dataset. **Spleeter** is also very fast as it can perform separation of audio files to 4 stems 100x faster than real-time when run on a GPU.
docker/cuda-10-1.dockerfile:ENV CUDA_VERSION 10.1.243
docker/cuda-10-1.dockerfile:ENV CUDA_PKG_VERSION 10-1=$CUDA_VERSION-1
docker/cuda-10-1.dockerfile:ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
docker/cuda-10-1.dockerfile:ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
docker/cuda-10-1.dockerfile:ENV NVIDIA_VISIBLE_DEVICES all
docker/cuda-10-1.dockerfile:ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
docker/cuda-10-1.dockerfile:ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"
docker/cuda-10-1.dockerfile:ENV NCCL_VERSION 2.7.8
docker/cuda-10-1.dockerfile:LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
docker/cuda-10-1.dockerfile:LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
docker/cuda-10-1.dockerfile:LABEL com.nvidia.volumes.needed="nvidia_driver"
docker/cuda-10-1.dockerfile:    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - \
docker/cuda-10-1.dockerfile:    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
docker/cuda-10-1.dockerfile:    && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
docker/cuda-10-1.dockerfile:        cuda-cudart-$CUDA_PKG_VERSION \
docker/cuda-10-1.dockerfile:        cuda-compat-10-1 \
docker/cuda-10-1.dockerfile:    && ln -s cuda-10.1 /usr/local/cuda \
docker/cuda-10-1.dockerfile:    && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
docker/cuda-10-1.dockerfile:    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
docker/cuda-10-1.dockerfile:        cuda-libraries-$CUDA_PKG_VERSION \
docker/cuda-10-1.dockerfile:        cuda-npp-$CUDA_PKG_VERSION \
docker/cuda-10-1.dockerfile:        cuda-nvtx-$CUDA_PKG_VERSION \
docker/cuda-10-1.dockerfile:        libcudnn7=$CUDNN_VERSION-1+cuda10.1 \
docker/cuda-10-1.dockerfile:        libnccl2=$NCCL_VERSION-1+cuda10.1 \
docker/cuda-10-1.dockerfile:    && apt-mark hold libnccl2 \
docker/cuda-10-0.dockerfile:ENV CUDA_VERSION 10.0.130
docker/cuda-10-0.dockerfile:ENV CUDA_PKG_VERSION 10-0=$CUDA_VERSION-1
docker/cuda-10-0.dockerfile:ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
docker/cuda-10-0.dockerfile:ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
docker/cuda-10-0.dockerfile:ENV NVIDIA_VISIBLE_DEVICES=all
docker/cuda-10-0.dockerfile:ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
docker/cuda-10-0.dockerfile:ENV NVIDIA_REQUIRE_CUDA "cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411"
docker/cuda-10-0.dockerfile:ENV NCCL_VERSION 2.4.2
docker/cuda-10-0.dockerfile:LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
docker/cuda-10-0.dockerfile:LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
docker/cuda-10-0.dockerfile:LABEL com.nvidia.volumes.needed="nvidia_driver"
docker/cuda-10-0.dockerfile:    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - \
docker/cuda-10-0.dockerfile:    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
docker/cuda-10-0.dockerfile:    && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
docker/cuda-10-0.dockerfile:        cuda-cudart-$CUDA_PKG_VERSION \
docker/cuda-10-0.dockerfile:        cuda-compat-10-0 \
docker/cuda-10-0.dockerfile:    && ln -s cuda-10.0 /usr/local/cuda \
docker/cuda-10-0.dockerfile:    && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
docker/cuda-10-0.dockerfile:    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
docker/cuda-10-0.dockerfile:        cuda-toolkit-10-0 \
docker/cuda-10-0.dockerfile:        cuda-libraries-$CUDA_PKG_VERSION \
docker/cuda-10-0.dockerfile:        cuda-nvtx-$CUDA_PKG_VERSION \
docker/cuda-10-0.dockerfile:        libnccl2=$NCCL_VERSION-1+cuda10.0 \
docker/cuda-10-0.dockerfile:        libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
docker/cuda-10-0.dockerfile:    && apt-mark hold libnccl2 \
docker/cuda-9.2.dockerfile:# https://gitlab.com/nvidia/container-images/cuda/blob/ubuntu18.04/9.2/base/Dockerfile
docker/cuda-9.2.dockerfile:    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1710/x86_64/7fa2af80.pub | apt-key add - \
docker/cuda-9.2.dockerfile:    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1710/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
docker/cuda-9.2.dockerfile:    && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
docker/cuda-9.2.dockerfile:ENV CUDA_VERSION 9.2.148
docker/cuda-9.2.dockerfile:ENV CUDA_PKG_VERSION 9-2=$CUDA_VERSION-1
docker/cuda-9.2.dockerfile:        cuda-cudart-$CUDA_PKG_VERSION \
docker/cuda-9.2.dockerfile:    && ln -s cuda-9.2 /usr/local/cuda \
docker/cuda-9.2.dockerfile:LABEL com.nvidia.volumes.needed="nvidia_driver"
docker/cuda-9.2.dockerfile:LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
docker/cuda-9.2.dockerfile:RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
docker/cuda-9.2.dockerfile:    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
docker/cuda-9.2.dockerfile:ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
docker/cuda-9.2.dockerfile:ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
docker/cuda-9.2.dockerfile:ENV NVIDIA_VISIBLE_DEVICES all
docker/cuda-9.2.dockerfile:ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
docker/cuda-9.2.dockerfile:ENV NVIDIA_REQUIRE_CUDA "cuda>=9.2"
docker/cuda-9.2.dockerfile:# https://gitlab.com/nvidia/container-images/cuda/blob/ubuntu18.04/9.2/runtime/Dockerfile
docker/cuda-9.2.dockerfile:ENV NCCL_VERSION 2.3.7
docker/cuda-9.2.dockerfile:        cuda-libraries-$CUDA_PKG_VERSION \
docker/cuda-9.2.dockerfile:        cuda-nvtx-$CUDA_PKG_VERSION \
docker/cuda-9.2.dockerfile:        libnccl2=$NCCL_VERSION-1+cuda9.2 \
docker/cuda-9.2.dockerfile:    && apt-mark hold libnccl2 \
docker/cuda-9.2.dockerfile:# https://gitlab.com/nvidia/container-images/cuda/blob/ubuntu18.04/9.2/runtime/cudnn7/Dockerfile
docker/cuda-9.2.dockerfile:LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
docker/cuda-9.2.dockerfile:    && apt-get install -y --no-install-recommends libcudnn7=$CUDNN_VERSION-1+cuda9.2 \
docker/cuda-9.2.dockerfile:RUN pip install spleeter-gpu==1.4.9
spleeter/model/__init__.py:        separation process (including STFT and inverse STFT) on GPU.
spleeter/__main__.py:    session_config.gpu_options.per_process_gpu_memory_fraction = 0.45
spleeter/separator.py:    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
poetry.lock:tensorflow-gpu = ["tensorflow-gpu (>=2.11.0,<2.12.0)"]
poetry.lock:tensorflow-rocm = ["tensorflow-rocm (>=2.11.0,<2.12.0)"]
paper.md:The performance of the pre-trained models are very close to the published state-of-the-art and is one of the best performing $4$ stems separation model on the common musdb18 benchmark [@musdb18] to be publicly released. Spleeter is also very fast as it can separate a mix audio file into $4$ stems $100$ times faster than real-time (we note, though, that the model cannot be applied in real-time as it needs buffering) on a single Graphics Processing Unit (GPU) using the pre-trained $4$-stems model.
paper.md:The pre-trained models are U-nets [@unet2017] and follow similar specifications as in [@deezerICASSP2019]. The U-net is an encoder/decoder Convolutional Neural Network (CNN) architecture with skip connections. We used $12$-layer U-nets ($6$ layers for the encoder and $6$ for the decoder). A U-net is used for estimating a soft mask for each source (stem). Training loss is a $L_1$-norm between masked input mix spectrograms and source-target spectrograms. The models were trained on Deezer's internal datasets (noteworthily the Bean dataset that was used in [@deezerICASSP2019]) using Adam [@Adam]. Training time took approximately a full week on a single GPU. Separation is then done from estimated source spectrograms using soft masking or multi-channel Wiener filtering.
paper.md:Training and inference are implemented in Tensorflow which makes it possible to run the code on Central Processing Unit (CPU) or GPU.
paper.md:As the whole separation pipeline can be run on a GPU and the model is based on a CNN, computations are efficiently parallelized and model inference is very fast. For instance, Spleeter is able to separate the whole musdb18 test dataset (about $3$ hours and $27$ minutes of audio) into $4$ stems in less than $2$ minutes, including model loading time (about $15$ seconds), and audio wav files export, using a single GeForce RTX 2080 GPU, and a double Intel Xeon Gold 6134 CPU @ 3.20GHz (CPU is used for mix files loading and stem files export only). In this setup, Spleeter is able to process $100$ seconds of stereo audio in less than $1$ second, which makes it very useful for efficiently processing large datasets.

```
