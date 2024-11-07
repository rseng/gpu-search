# https://github.com/davidpagnon/Sports2D

```console
README.md:**Use your GPU**:\
README.md:1. Run `nvidia-smi` in a terminal. If this results in an error, your GPU is probably not compatible with CUDA. If not, note the "CUDA version": it is the latest version your driver is compatible with (more information [on this post](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with)).
README.md:   Then go to the [ONNXruntime requirement page](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), note the latest compatible CUDA and cuDNN requirements. Next, go to the [pyTorch website](https://pytorch.org/get-started/previous-versions/) and install the latest version that satisfies these requirements (beware that torch 2.4 ships with cuDNN 9, while torch 2.3 installs cuDNN 8). For example:
README.md:2. Finally, install ONNX Runtime with GPU support:
README.md:   pip install onnxruntime-gpu
README.md:   python -c 'import torch; print(torch.cuda.is_available())'
README.md:   # Should print "True ['CUDAExecutionProvider', ...]"
README.md:   <!-- print(f'torch version: {torch.__version__}, cuda version: {torch.version.cuda}, cudnn version: {torch.backends.cudnn.version()}, onnxruntime version: {ort.__version__}') -->
Content/paper.md:`Sports2d` is installed under Python via `pip install sports2d`. If a valid CUDA installation is found, Sports2D uses the GPU, otherwise it uses the CPU with OpenVino acceleration. 
Content/paper.md:* *If run locally*, it is installed under Python via `pip install sports2d`. If a valid CUDA installation is found, Sports2D uses the GPU, otherwise it uses the CPU with OpenVino acceleration. 
Sports2D/process.py:    If CUDA is available, use it with ONNXRuntime backend; else use CPU with openvino
Sports2D/process.py:    # If CUDA is available, use it with ONNXRuntime backend; else use CPU with openvino
Sports2D/process.py:        if torch.cuda.is_available() == True and 'CUDAExecutionProvider' in ort.get_available_providers():
Sports2D/process.py:            device = 'cuda'
Sports2D/process.py:            logging.info(f"\nValid CUDA installation found: using ONNXRuntime backend with GPU.")
Sports2D/process.py:        elif torch.cuda.is_available() == True and 'ROCMExecutionProvider' in ort.get_available_providers():
Sports2D/process.py:            device = 'rocm'
Sports2D/process.py:            logging.info(f"\nValid ROCM installation found: using ONNXRuntime backend with GPU.")
Sports2D/process.py:                logging.info(f"\nValid MPS installation found: using ONNXRuntime backend with GPU.")
Sports2D/process.py:            logging.info(f"\nNo valid CUDA installation found: using OpenVINO backend with CPU.")

```
