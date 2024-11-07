# https://github.com/m2aia/m2aia

```console
Plugins/org.mitk.gui.qt.m2.docker.simclr/src/internal/QmitkSimCLRView.cpp:          helper.EnableGPUs(true);
Plugins/org.mitk.gui.qt.m2.docker.peaklearning/src/internal/QmitkPeakLearningView.cpp:          helper.EnableGPUs(true);
Plugins/org.mitk.gui.qt.m2.docker.umap/src/internal/QmitkUMAPView.cpp:helper.EnableGPUs(false);
Patch/mitk.diff:+    if(NOT MITK_USE_CUDA)
Patch/mitk.diff:+           -DCUDA_USE_STATIC_CUDA_RUNTIME:BOOL=OFF
Patch/mitk.diff:+           -DCUDA_TOOLKIT_ROOT_DIR:PATH=""
Patch/mitk.diff: option(MITK_USE_OpenCL "Use OpenCL GPU-Computing library" OFF)
Patch/mitk.diff:+option(MITK_USE_CUDA "Use CUDA" OFF)
Patch/mitk.diff:-  OpenCL
Patch/mitk.diff:+  # OpenCL
Docker/peaklearning/Dockerfile.gpu:FROM ghcr.io/m2aia/pym2aia/cuda:11.8.0-ubuntu22.04 as system
Docker/peaklearning/Dockerfile.gpu:RUN pip install tensorflow[and-cuda]
Docker/Dockerfile.m2aia-gpu-base:ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
Docker/Dockerfile.m2aia-gpu-base:LABEL org.opencontainers.image.description="Base image for GPU based applications."
Docker/Dockerfile.m2aia-gpu-base:LABEL org.opencontainers.image.ref.name="m2aia/pym2aia-base-gpu"
Docker/Dockerfile.m2aia-gpu-base:LABEL org.opencontainers.image.title="pyM²aia-base-gpu"
Docker/simclr/app_simclr.py:m = m.cuda()
Docker/simclr/app_simclr.py:loss = loss.cuda()
Docker/simclr/app_simclr.py:        X = X.cuda(non_blocking=True)
Docker/simclr/app_simclr.py:        Y = Y.cuda(non_blocking=True)
Docker/simclr/app_simclr.py:    _, embedding = m(torch.tensor(ionImage[None,...]).cuda())
Docker/simclr/Dockerfile.gpu:ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

```
