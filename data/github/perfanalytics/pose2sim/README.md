# https://github.com/perfanalytics/pose2sim

```console
Pose2Sim/poseEstimation.py:    If a valid cuda installation is detected, uses the GPU with the ONNXRuntime backend. Otherwise, 
Pose2Sim/poseEstimation.py:    If a valid cuda installation is detected, uses the GPU with the ONNXRuntime backend. Otherwise, 
Pose2Sim/poseEstimation.py:    # If CUDA is available, use it with ONNXRuntime backend; else use CPU with openvino
Pose2Sim/poseEstimation.py:        if torch.cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
Pose2Sim/poseEstimation.py:            device = 'cuda'
Pose2Sim/poseEstimation.py:            logging.info(f"\nValid CUDA installation found: using ONNXRuntime backend with GPU.")
Pose2Sim/poseEstimation.py:        elif torch.cuda.is_available() == True and 'ROCMExecutionProvider' in ort.get_available_providers():
Pose2Sim/poseEstimation.py:            device = 'rocm'
Pose2Sim/poseEstimation.py:            logging.info(f"\nValid ROCM installation found: using ONNXRuntime backend with GPU.")
Pose2Sim/poseEstimation.py:                logging.info(f"\nValid MPS installation found: using ONNXRuntime backend with GPU.")
Pose2Sim/poseEstimation.py:            logging.info(f"\nNo valid CUDA installation found: using OpenVINO backend with CPU.")
README.md:   *For faster inference, you can run on the GPU. Install pyTorch with CUDA and cuDNN support, and ONNX Runtime with GPU support (not available on MacOS).*\
README.md:   Be aware that GPU support takes an additional 6 GB on disk. The full installation is then 10.75 GB instead of 4.75 GB.
README.md:   Run `nvidia-smi` in a terminal. If this results in an error, your GPU is probably not compatible with CUDA. If not, note the "CUDA version": it is the latest version your driver is compatible with (more information [on this post](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with)).
README.md:   Then go to the [ONNXruntime requirement page](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), note the latest compatible CUDA and cuDNN requirements. Next, go to the [pyTorch website](https://pytorch.org/get-started/previous-versions/) and install the latest version that satisfies these requirements (beware that torch 2.4 ships with cuDNN 9, while torch 2.3 installs cuDNN 8). For example:
README.md:   Finally, install ONNX Runtime with GPU support:
README.md:   pip install onnxruntime-gpu
README.md:   print(torch.cuda.is_available(), ort.get_available_providers())
README.md:   # Should print "True ['CUDAExecutionProvider', ...]"
README.md:  <!-- print(f'torch version: {torch.__version__}, cuda version: {torch.version.cuda}, cudnn version: {torch.backends.cudnn.version()}, onnxruntime version: {ort.__version__}') -->
README.md:     A full installation takes up to 11 GB of storage spate. However, GPU support is not mandatory and takes about 6 GB. Moreover, [marker augmentation](#marker-augmentation) requires Tensorflow and does not necessarily yield better results. You can save an additional 1.3 GB by uninstalling it: `pip uninstall tensorflow`.\
README.md:     A minimal installation with carefully chosen pose models and without GPU support, Tensorflow, PyQt5 **would take less than 3 GB**.\
README.md:*N.B.:* The `GPU` will be used with ONNX backend if a valid CUDA installation is found (or ROCM with AMD GPUS, or MPS with MacOS), otherwise the `CPU` will be used with OpenVINO backend.\
Content/website/index.md:   *For faster inference, you can run on the GPU. Install pyTorch with CUDA and cuDNN support, and ONNX Runtime with GPU support (not available on MacOS).*\
Content/website/index.md:   Be aware that GPU support takes an additional 6 GB on disk. The full installation is then 10.75 GB instead of 4.75 GB.
Content/website/index.md:   Run `nvidia-smi` in a terminal. If this results in an error, your GPU is probably not compatible with CUDA. If not, note the "CUDA version": it is the latest version your driver is compatible with (more information [on this post](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with)).
Content/website/index.md:   Then go to the [ONNXruntime requirement page](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), note the latest compatible CUDA and cuDNN requirements. Next, go to the [pyTorch website](https://pytorch.org/get-started/previous-versions/) and install the latest version that satisfies these requirements (beware that torch 2.4 ships with cuDNN 9, while torch 2.3 installs cuDNN 8). For example:
Content/website/index.md:   Finally, install ONNX Runtime with GPU support:
Content/website/index.md:   pip install onnxruntime-gpu
Content/website/index.md:   print(torch.cuda.is_available(), ort.get_available_providers())
Content/website/index.md:   # Should print "True ['CUDAExecutionProvider', ...]"
Content/website/index.md:  <!-- print(f'torch version: {torch.__version__}, cuda version: {torch.version.cuda}, cudnn version: {torch.backends.cudnn.version()}, onnxruntime version: {ort.__version__}') -->
Content/website/index.md:     A full installation takes up to 11 GB of storage spate. However, GPU support is not mandatory and takes about 6 GB. Moreover, [marker augmentation](#marker-augmentation) requires Tensorflow and does not necessarily yield better results. You can save an additional 1.3 GB by uninstalling it: `pip uninstall tensorflow`.\
Content/website/index.md:     A minimal installation with carefully chosen pose models and without GPU support, Tensorflow, PyQt5 **would take less than 3 GB**.\
Content/website/index.md:*N.B.:* The `GPU` will be used with ONNX backend if a valid CUDA installation is found (or MPS with MacOS), otherwise the `CPU` will be used with OpenVINO backend.\

```
