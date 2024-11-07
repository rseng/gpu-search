# https://github.com/MIC-DKFZ/TractSeg

```console
Dockerfile_GPU:## NVIDIA CUDA Installation
Dockerfile_GPU:# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
Dockerfile_GPU:# * Neither the name of NVIDIA CORPORATION nor the names of its
Dockerfile_GPU:    NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
Dockerfile_GPU:    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
Dockerfile_GPU:    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
Dockerfile_GPU:    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +2 > cudasign.pub && \
Dockerfile_GPU:    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
Dockerfile_GPU:    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
Dockerfile_GPU:    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
Dockerfile_GPU:ENV CUDA_VERSION 9.1.85
Dockerfile_GPU:ENV CUDA_PKG_VERSION 9-1=$CUDA_VERSION-1
Dockerfile_GPU:    cuda-cudart-$CUDA_PKG_VERSION && \
Dockerfile_GPU:    ln -s cuda-9.1 /usr/local/cuda && \
Dockerfile_GPU:# nvidia-docker 1.0
Dockerfile_GPU:LABEL com.nvidia.volumes.needed="nvidia_driver"
Dockerfile_GPU:LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
Dockerfile_GPU:RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
Dockerfile_GPU:    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
Dockerfile_GPU:ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
Dockerfile_GPU:ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
Dockerfile_GPU:# nvidia-container-runtime
Dockerfile_GPU:ENV NVIDIA_VISIBLE_DEVICES all
Dockerfile_GPU:ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
Dockerfile_GPU:ENV NVIDIA_REQUIRE_CUDA "cuda>=9.1"
Dockerfile_GPU:ENV NCCL_VERSION 2.2.12
Dockerfile_GPU:    cuda-libraries-$CUDA_PKG_VERSION \
Dockerfile_GPU:    libnccl2=$NCCL_VERSION-1+cuda9.1 && \
Dockerfile_GPU:    apt-mark hold libnccl2 && \
Dockerfile_GPU:LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
Dockerfile_GPU:    libcudnn7=$CUDNN_VERSION-1+cuda9.1 && \
resources/Tractometry_documentation.md:GPU: 2min ~14s)  
resources/Tractometry_documentation.md:`TractSeg -i tractseg_output/peaks.nii.gz -o tractseg_output --output_type endings_segmentation` (runtime on GPU: ~42s)
resources/Tractometry_documentation.md:`TractSeg -i tractseg_output/peaks.nii.gz -o tractseg_output --output_type TOM` (runtime on GPU: ~1min 30s)  
resources/Tutorial.md:* If you have a NVIDIA GPU and CUDA installed TractSeg will run in less than 1min. Otherwise it will fall back to CPU and run several minutes.
tractseg/models/base_model.py:        # MultiGPU setup
tractseg/models/base_model.py:        # (Not really faster (max 10% speedup): GPU and CPU utility low)
tractseg/models/base_model.py:        # nr_gpus = torch.cuda.device_count()
tractseg/models/base_model.py:        # exp_utils.print_and_save(self.Config.EXP_PATH, "nr of gpus: {}".format(nr_gpus))
tractseg/models/base_model.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tractseg/models/base_model.py:        X = X.contiguous().cuda(non_blocking=True)  # (bs, features, x, y)
tractseg/models/base_model.py:        y = y.contiguous().cuda(non_blocking=True)  # (bs, classes, x, y)
tractseg/models/base_model.py:                                      y.shape[2], y.shape[3])).cuda()
tractseg/models/base_model.py:                                      y.shape[2], y.shape[3], y.shape[4])).cuda()
tractseg/models/base_model.py:            X = X.contiguous().cuda(non_blocking=True)
tractseg/models/base_model.py:            y = y.contiguous().cuda(non_blocking=True)
tractseg/models/base_model.py:                                      y.shape[2], y.shape[3])).cuda()
tractseg/models/base_model.py:                                      y.shape[2], y.shape[3], y.shape[4])).cuda()
tractseg/experiments/pretrained_models/TractSeg_HR_3D_DAug.py:System RAM running full (>30GB) from DAug -> super slow (without DAug: GPU utility 90%)
bin/ExpRunner:        # Free GPU memory  (2.5GB remaining otherwise)
bin/ExpRunner:        torch.cuda.empty_cache()

```
