# https://github.com/PetervanLunteren/EcoAssist

```console
global_vars.json:    "var_disable_GPU": false,
EcoAssist_GUI.py:        cls_disable_GPU = False
EcoAssist_GUI.py:        cls_disable_GPU = var_disable_GPU.get()
EcoAssist_GUI.py:    command_args.append(str(cls_disable_GPU))
EcoAssist_GUI.py:            GPU_param = "Unknown",
EcoAssist_GUI.py:        if line.startswith("GPU available: False"):
EcoAssist_GUI.py:            GPU_param = "CPU"
EcoAssist_GUI.py:        elif line.startswith("GPU available: True"):
EcoAssist_GUI.py:            GPU_param = "GPU"
EcoAssist_GUI.py:                                            hware = GPU_param,
EcoAssist_GUI.py:    GPU_param = "Unknown"
EcoAssist_GUI.py:    # if user specified to disable GPU, prepend and set system variable
EcoAssist_GUI.py:    if var_disable_GPU.get() and not simple_mode:
EcoAssist_GUI.py:            command[:0] = ['set', 'CUDA_VISIBLE_DEVICES=""', '&']
EcoAssist_GUI.py:                           ["Disabling GPU processing is currently only supported for CUDA devices on Linux and Windows "
EcoAssist_GUI.py:                            "machines, not on macOS. Proceeding without GPU disabled.", "Deshabilitar el procesamiento de "
EcoAssist_GUI.py:                            "la GPU actualmente sólo es compatible con dispositivos CUDA en máquinas Linux y Windows, no en"
EcoAssist_GUI.py:                            " macOS. Proceder sin GPU desactivada."][lang_idx])
EcoAssist_GUI.py:            var_disable_GPU.set(False)
EcoAssist_GUI.py:            command = "CUDA_VISIBLE_DEVICES='' " + command
EcoAssist_GUI.py:        if line.startswith("GPU available: False"):
EcoAssist_GUI.py:            GPU_param = "CPU"
EcoAssist_GUI.py:        elif line.startswith("GPU available: True"):
EcoAssist_GUI.py:            GPU_param = "GPU"
EcoAssist_GUI.py:                                            hware = GPU_param,
EcoAssist_GUI.py:            "var_disable_GPU": var_disable_GPU.get(),
EcoAssist_GUI.py:    lbl_disable_GPU.configure(text=lbl_disable_GPU_txt[lang_idx])
EcoAssist_GUI.py:    var_disable_GPU.set(False)
EcoAssist_GUI.py:        "var_disable_GPU": var_disable_GPU.get(),
EcoAssist_GUI.py:lbl_disable_GPU_txt = ["Disable GPU processing", "Desactivar el procesamiento en la GPU"]
EcoAssist_GUI.py:row_disable_GPU = 7
EcoAssist_GUI.py:lbl_disable_GPU = Label(snd_step, text=lbl_disable_GPU_txt[lang_idx], width=1, anchor="w")
EcoAssist_GUI.py:lbl_disable_GPU.grid(row=row_disable_GPU, sticky='nesw', pady=2)
EcoAssist_GUI.py:var_disable_GPU = BooleanVar()
EcoAssist_GUI.py:var_disable_GPU.set(global_vars['var_disable_GPU'])
EcoAssist_GUI.py:chb_disable_GPU = Checkbutton(snd_step, variable=var_disable_GPU, anchor="w")
EcoAssist_GUI.py:chb_disable_GPU.grid(row=row_disable_GPU, column=1, sticky='nesw', padx=5)
classification_utils/inference_lib.py:                     GPU_availability,
classification_utils/inference_lib.py:                                            GPU_availability = GPU_availability,
classification_utils/inference_lib.py:                                            GPU_availability = GPU_availability,
classification_utils/inference_lib.py:                                         GPU_availability,
classification_utils/inference_lib.py:    print(f"GPU available: {GPU_availability}")
classification_utils/envs/pytorch.yml:  - conda-forge::cudatoolkit=11.3
classification_utils/envs/tensorflow-linux-windows.yml:  - conda-forge::cudatoolkit=11.2
classification_utils/model_types/sdzwa-pt/classify_detections.py:# check GPU availability
classification_utils/model_types/sdzwa-pt/classify_detections.py:GPU_availability = False
classification_utils/model_types/sdzwa-pt/classify_detections.py:        GPU_availability = True
classification_utils/model_types/sdzwa-pt/classify_detections.py:if not GPU_availability:
classification_utils/model_types/sdzwa-pt/classify_detections.py:    if torch.cuda.is_available():
classification_utils/model_types/sdzwa-pt/classify_detections.py:        GPU_availability = True
classification_utils/model_types/sdzwa-pt/classify_detections.py:        device_str = 'cuda'
classification_utils/model_types/sdzwa-pt/classify_detections.py:print(GPU_availability)
classification_utils/model_types/sdzwa-pt/classify_detections.py:                    GPU_availability = GPU_availability,
classification_utils/model_types/sdzwa-peru-amazon/classify_detections.py:# check GPU availability
classification_utils/model_types/sdzwa-peru-amazon/classify_detections.py:GPU_availability = True if len(tf.config.list_logical_devices('GPU')) > 0 else False
classification_utils/model_types/sdzwa-peru-amazon/classify_detections.py:                    GPU_availability = GPU_availability,
classification_utils/model_types/mewc/classify_detections.py:# check GPU availability
classification_utils/model_types/mewc/classify_detections.py:GPU_availability = True if len(tf.config.list_logical_devices('GPU')) > 0 else False
classification_utils/model_types/mewc/classify_detections.py:                    GPU_availability = GPU_availability,
classification_utils/model_types/hex-data-pt/classify_detections.py:# check GPU availability
classification_utils/model_types/hex-data-pt/classify_detections.py:GPU_availability = False
classification_utils/model_types/hex-data-pt/classify_detections.py:        GPU_availability = True
classification_utils/model_types/hex-data-pt/classify_detections.py:        # backend, which is used for Apple Silicon GPUs, so we'll set it to CPU
classification_utils/model_types/hex-data-pt/classify_detections.py:if not GPU_availability:
classification_utils/model_types/hex-data-pt/classify_detections.py:    if torch.cuda.is_available():
classification_utils/model_types/hex-data-pt/classify_detections.py:        GPU_availability = True
classification_utils/model_types/hex-data-pt/classify_detections.py:        device_str = 'cuda'
classification_utils/model_types/hex-data-pt/classify_detections.py:                    GPU_availability = GPU_availability,
classification_utils/model_types/deepfaune/classify_detections.py:# check on and on which GPU the process should run
classification_utils/model_types/deepfaune/classify_detections.py:    if torch.cuda.is_available():
classification_utils/model_types/deepfaune/classify_detections.py:        device = torch.device('cuda')
classification_utils/model_types/deepfaune/classify_detections.py:# 2. Run on Apple Silicon GPU via MPS Metal GPU
classification_utils/model_types/deepfaune/classify_detections.py:        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_utils/model_types/deepfaune/classify_detections.py:        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_utils/model_types/deepfaune/classify_detections.py:# check GPU availability
classification_utils/model_types/deepfaune/classify_detections.py:GPU_availability = False
classification_utils/model_types/deepfaune/classify_detections.py:        GPU_availability = True
classification_utils/model_types/deepfaune/classify_detections.py:if not GPU_availability:
classification_utils/model_types/deepfaune/classify_detections.py:    GPU_availability = torch.cuda.is_available()
classification_utils/model_types/deepfaune/classify_detections.py:                    GPU_availability = GPU_availability,
classification_utils/model_types/addax-yolov8/classify_detections.py:# check GPU availability
classification_utils/model_types/addax-yolov8/classify_detections.py:GPU_availability = False
classification_utils/model_types/addax-yolov8/classify_detections.py:        GPU_availability = True
classification_utils/model_types/addax-yolov8/classify_detections.py:if not GPU_availability:
classification_utils/model_types/addax-yolov8/classify_detections.py:    GPU_availability = torch.cuda.is_available()
classification_utils/model_types/addax-yolov8/classify_detections.py:                    GPU_availability = GPU_availability,
classification_utils/model_types/pywildlife/classify_detections.py:# check GPU availability
classification_utils/model_types/pywildlife/classify_detections.py:GPU_availability = False
classification_utils/model_types/pywildlife/classify_detections.py:        GPU_availability = True
classification_utils/model_types/pywildlife/classify_detections.py:if not GPU_availability:
classification_utils/model_types/pywildlife/classify_detections.py:    GPU_availability = torch.cuda.is_available()
classification_utils/model_types/pywildlife/classify_detections.py:                    GPU_availability = GPU_availability,
classification_utils/start_class_inference.bat:set "GPU_DISABLED=%1"
classification_utils/start_class_inference.bat:if "%GPU_DISABLED%"=="True" (
classification_utils/start_class_inference.bat:    set CUDA_VISIBLE_DEVICES="" & python %INF_SCRIPT% %LOCATION_ECOASSIST_FILES% %MODEL_FPATH% %DET_THRESH% %CLS_THRESH% %SMOOTH_BOOL% %JSON_FPATH% %FRAME_DIR%
classification_utils/start_class_inference.command:GPU_DISABLED=${1}
classification_utils/start_class_inference.command:if [ "$GPU_DISABLED" == "True" ] && [ "$PLATFORM" == "Linux" ]; then
classification_utils/start_class_inference.command:    CUDA_VISIBLE_DEVICES='' python "${INF_SCRIPT}" "${LOCATION_ECOASSIST_FILES}" "${MODEL_FPATH}" "${DET_THRESH}" "${CLS_THRESH}" "${SMOOTH_BOOL}" "${JSON_FPATH}" "${FRAME_DIR}"
install.bat:call %CONDA_EXECUTABLE% install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
README.md:13. [GPU support](#gpu-support)
README.md:* GPU acceleration for NVIDIA and Apple Silicon
README.md:Except a minimum of 8 GB RAM, there are no hard system requirements for EcoAssist since it is largely hardware-agnostic. However, please note that machine learning can ask quite a lot from your computer in terms of processing power. Although it will run on an old laptop only designed for text editing, it’s probably not going to train any accurate models, while deploying models can take ages. Generally speaking, the faster the machine, the more reliable the results. GPU acceleration is a big plus.
README.md:## GPU support
README.md:EcoAssist will automatically run on NVIDIA or Apple Silicon GPU if available. The appropriate `CUDAtoolkit` and `cuDNN` software is already included in the EcoAssist installation for Windows and Linux. If you have NVIDIA GPU available but it doesn't recognise it, make sure you have a [recent driver](https://www.nvidia.com/en-us/geforce/drivers/) installed, then reboot. An MPS compatible version of `Pytorch` is included in the installation for Apple Silicon users. Please note that applying machine learning on Apple Silicon GPU's is still under beta version. That means that you might run into errors when trying to run on GPU. My experience is that deployment runs smoothly on GPU, but training throws errors. Training on CPU will of course still work. The progress window and console output will display whether EcoAssist is running on CPU or GPU. 
README.md:  <img src="https://github.com/PetervanLunteren/EcoAssist-metadata/blob/main/imgs/Training_on_GPU.png" width=90% height="auto" >
markdown/errors.md:    "cuda100-1.0-0.tar.bz2": {
markdown/errors.md:      "name": "cuda100",
markdown/errors.md:      "track_features": "cuda100",
markdown/errors.md:## `RuntimeError: CUDA error: no kernel image is available for execution on the device CUDA kernel errors might be asynchronously reported at some other API call so the stacktrace below might be incorrect.`
markdown/errors.md:Not sure what the solution is yet, but it looks like there might be a package conflict with CUDA and TORCH. See: https://stackoverflow.com/questions/69968477/runtimeerror-cuda-error-no-kernel-image-is-available-for-execution-on-the-devi
markdown/errors.md:## `Runtimetrror: CODA error: no kernel image is available for execution on the device CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorred.`
markdown/errors.md:This has to do with CUDA not being compatible with PyTorch in some mystical way. There are a few things we can try to dial in on the problem. 
markdown/errors.md:1. Does EcoAssist work when you disable the GPU (see attached screenshot)?
markdown/errors.md:2. Perhaps this has to do with your GPU version or driver. What kind of GPU do you have? Make sure you have a recent driver installed, then reboot.
markdown/errors.md:## `LibMambaUnsatisfiableError: Encountered problems while solving: package cudatoolkit-11.3.1-h280eb24_10 has constraint __cuda >=11 conflicting with __cuda-8.0-0`
markdown/errors.md:There seems to be an issue with the Cuda version on our computer. Can you try the following:
markdown/errors.md:2. Install Cuda 11.3 from https://developer.nvidia.com/cuda-11.3.0-download-archive
markdown/errors.md:3. Make sure you have a recent driver installed: https://www.nvidia.com/download/index.aspx
markdown/errors.md:## GPU out of memory
markdown/errors.md:The error message is saying that the GPU is out of memory. In other words, it can't handle the workload. Apart from buying new hardware, there are a few things you can do.
markdown/errors.md:1. Make sure you have a recent driver installed, then reboot: https://www.nvidia.com/download/index.aspx
markdown/errors.md:4. Select the 'Disable GPU processing' option in advanced mode. Processing will go slow, but at least it won't crash.
markdown/FAQ.md:[EcoAssist](https://github.com/PetervanLunteren/EcoAssist) is available for Windows, Mac and Linux systems. Please note that machine learning can ask quite a lot from your computer in terms of processing power. Besides a minimum of 8GB of RAM, there are no hard system requirements since it is largely hardware-agnostic. However, I would recommend at least 16GB of RAM, but preferably 32GB. Although it will run on an old laptop only designed for text editing, it's probably not going to train any accurate models. Faster machines will analyze images quicker and produce more accurate results. The best models are trained on computers with GPU acceleration. EcoAssist will automatically run on NVIDIA and Apple Silicon GPU if you have the appropriate hardware available, see [this thread](https://github.com/PetervanLunteren/EcoAssist#gpu-support) for more info. If you have multiple computers at hand, choose the fastest one. Please note that - at the time of writing (May 2023) - training on Apple Silicon GPU is still in beta version.
markdown/FAQ.md:There are five pre-trained YOLOv5 models which you can use to transfer knowledge from (see image below). These go from small to large and are trained on the [COCO dataset](https://cocodataset.org/#home) consisting of more than 330,000 images of 80 classes. These pre-trained models already _know_ what life on earth looks like, so you don't have to teach your model again. In general, the larger the model, the better the results - but also the more processing power required. The nano model is the smallest and fastest, and is most suitable for mobile solutions and embedded devices. The small model is perfect for a laptop without GPU acceleration. The medium-sized model provides a good balance between speed and accuracy, but you'll probably want a GPU for this. The large model is ideal for detecting small objects, and the extra-large model is the most accurate of them all, but it takes considerable time and processing power to train and deploy. The last two models are recommended for cloud deployments.
markdown/FAQ.md:* CUDA out of memory
markdown/FAQ.md:* Error loading '...\torch\lib\caffe2_detectron_ops_gpu.dll' or one of its dependencies
markdown/FAQ.md:* **Reduce the batch size** - This is the most obvious method of reducing the workload. The batch size is the number of images your computer takes per iteration. The larger the batch size, the more processing power you'll need. If you leave its entry box in [EcoAssist](https://github.com/PetervanLunteren/EcoAssist) empty, it'll automatically check and use the maximum batch size your device can handle. However, this check only works if you have NVIDIA GPU acceleration. If your device doesn't have this, it'll revert to the default batch size of 16 - which might be too large for your computer. So, if you are not training on NVIDIA GPU, try lowering to a batch size of e.g., 4 or even 1 and see if you still run into out-of-memory errors. If not, try increasing it again to find the maximum batch size your hardware allows for.
markdown/FAQ.md:* **Upgrade your hardware** - This option is the most obvious one. Try running it on a faster device or add processing power in terms of GPU acceleration.

```
