# https://github.com/FabianPlum/OmniTrax

```console
docs/tutorial-tracking.md:If you have a **CUDA supported GPU** *(and the [CUDA installation](CUDA_installation_guide.md) went as planned...)*,
docs/tutorial-tracking.md:make sure your **GPU is selected** here, **before** running any of the inference functions, as the computational device
docs/CUDA_installation_guide.md:**OmniTrax - CUDA installation guide**
docs/CUDA_installation_guide.md:Installing [**CUDA** (Compute Unified Device Architecture)](https://en.wikipedia.org/wiki/CUDA) is widely regarded as one of the least fun but sadly most essential endeavours when diving into machine
docs/CUDA_installation_guide.md:**OmniTrax** with GPU support (which will result in an inference speed increase of at least one order of magnitude).
docs/CUDA_installation_guide.md:we are going to install [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) 
docs/CUDA_installation_guide.md:and [cuDNN 8.1.0](https://developer.nvidia.com/rdp/cudnn-archive).
docs/CUDA_installation_guide.md:Download [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) 
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_01.PNG width="600">
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_02.PNG width="600">
docs/CUDA_installation_guide.md:Next, download [cuDNN 8.1.0](https://developer.nvidia.com/rdp/cudnn-archive). Pay close attention to the selected version.
docs/CUDA_installation_guide.md:In this case we need **v8.1.0 for CUDA 11.0, 11.1, 11.2**, specifically the **cuDNN Library for Windows (x86)**.
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_03.PNG width="700">
docs/CUDA_installation_guide.md:If this is your first time using **CUDA**, you will need to first create an NVIDIA account. (Please don't ask me why.)
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_04.PNG width="400">
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_05.PNG width="400">
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_06.PNG width="500">
docs/CUDA_installation_guide.md:## Installing CUDA
docs/CUDA_installation_guide.md:Open the downloaded **CUDA** installer and chose a location to temporarily store the unpacked files (~5 GB).
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_07.PNG width="400">
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_08.PNG width="500">
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_09.PNG width="500">
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_10.PNG width="500">
docs/CUDA_installation_guide.md:When the installation has completed, doublecheck that **CUDA 11.2** has been added to your **Environment variables**.
docs/CUDA_installation_guide.md:Click on **Environment Variables...** and check whether **CUDA_PATH** as well as **CUDA_PATH_V11_2** are listed.
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_11.PNG width="700">
docs/CUDA_installation_guide.md:All that is left to do now, is move the files contained in the downloaded **cuDNN**_###.zip into the appropriate CUDA directory.
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_12.PNG width="600">
docs/CUDA_installation_guide.md:Simply drag these contents into your **CUDA/v11.2** directory, which, by default, is located in 
docs/CUDA_installation_guide.md:*C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2*
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_13.PNG width="700">
docs/CUDA_installation_guide.md:be able to select your **GPU**(s) as (a) compute device(s) within **OmniTrax**.
docs/CUDA_installation_guide.md:<img src=CUDA_installation_images/CUDA_14.PNG width="400">
README.md:> **OmniTrax** runs on both Windows 10 / 11 as well as Ubuntu systems. However, the installation and CPU vs GPU 
README.md:|    Operating System    | Blender Version | CPU inference  | GPU inference |
README.md:* **OmniTrax GPU** is currently only supported on **Windows 10 / 11**. For Ubuntu support on CPU, use [**Blender version 2.92.0**](https://download.blender.org/release/Blender2.92/) and skip the steps on CUDA installation.
README.md:* As we are using **tensorflow 2.7**, to run inference on your GPU, you will need to install [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) and [cudNN 8.1](https://developer.nvidia.com/rdp/cudnn-archive). Refer to [this](https://www.tensorflow.org/install/source#gpu) official guide for version matching and installation instructions.
README.md:2. Install [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) and [cudNN 8.1.0](https://developer.nvidia.com/rdp/cudnn-archive). Here, we provide a separate [CUDA installation guide](docs/CUDA_installation_guide.md). 
README.md:   - **For advanced users**: If you already have a separate CUDA installation on your system, make sure to **additionally** install 11.2 and update your PATH environment variable. Conflicting versions may mean that OmniTrax is unable to find your GPU which may lead to unexpected crashes.
README.md:**2.** Next, select your compute device. If you have a **CUDA supported GPU** *(and the CUDA installation went as planned...)*, make sure your **GPU is selected** here, **before** running any of the inference functions, as the compute device cannot be changed at runtime. By default, assuming your computer has a one supported GPU, **OmniTrax** will select it as **GPU_0**.
README.md:* [CUDA installation instructions](docs/CUDA_installation_guide.md)
README.md:* 02/11/2022 - Added [**release** version 0.1.3](https://github.com/FabianPlum/OmniTrax/releases/tag/V_0.1.3) which includes improved tracking from previous states, faster and more robust track transfer, building skeletons from DLC config files, improved package installation and start-up checks, a few bug fixes, and GPU compatibility with the latest release of [Blender LTS 3.3](https://www.blender.org/download/lts/3-3/)!   For CPU-only inference, continue to use **Blender 2.92.0**.
README.md:* 06/10/2022 - Added [**release** version 0.1.2](https://github.com/FabianPlum/OmniTrax/releases/tag/V_0.1.2) with GPU support for latest [Blender LTS 3.3](https://www.blender.org/download/lts/3-3/)! For CPU-only inference, continue to use **Blender 2.92.0**.
README.md:* 19/02/2022 - Added [**release** version 0.1.1](https://github.com/FabianPlum/OmniTrax/releases/tag/V_0.1.1)! Things run a lot faster now and I have added support for devices without dedicated GPUs. 
utils/track/operators.py:from ..setup.CUDA_checks import check_CUDA_installation
utils/track/operators.py:        # load darknet with compiled DLLs for windows for either GPU or CPU inference from respective path
utils/track/operators.py:        if context.scene.compute_device.split("_")[0] == "GPU":
utils/track/operators.py:            required_CUDA_version = "11.2"
utils/track/operators.py:            CUDA_match = check_CUDA_installation(
utils/track/operators.py:                required_CUDA_version=required_CUDA_version
utils/track/operators.py:            if not CUDA_match:
utils/track/operators.py:                    "No matching CUDA version found! Refer to console for full error message",
utils/track/yolo_tracker.py:            # load darknet with compiled DLLs for windows for either GPU or CPU inference from respective path
utils/track/yolo_tracker.py:            # use GPU inference by default
utils/track/yolo_tracker.py:        help="Select compute device, e.g. CPU_0, GPU_0, GPU_1 ...",
utils/darknet/darknet.py:def set_compute_device(gpu=0):
utils/darknet/darknet.py:    set_gpu = lib.cuda_set_device
utils/darknet/darknet.py:    set_gpu(gpu)
utils/darknet/darknet.py:set_gpu = lib.cuda_set_device
utils/darknet/darknet_cpu.py:set_gpu = lib.cuda_set_device
utils/setup/CUDA_checks.py:def check_CUDA_installation(required_CUDA_version="11.2"):
utils/setup/CUDA_checks.py:    CHeck whether the required version of CUDA is installed.
utils/setup/CUDA_checks.py:    :param required_CUDA_version: OmniTrax required CUDA version
utils/setup/CUDA_checks.py:    print("\nINFO: Required CUDA version:", required_CUDA_version)
utils/setup/CUDA_checks.py:    found_CUDA_version = getVersion.read().decode().split("release ")[-1].split(",")[0]
utils/setup/CUDA_checks.py:    print("INFO: Found CUDA version:   ", found_CUDA_version)
utils/setup/CUDA_checks.py:    if found_CUDA_version == required_CUDA_version:
utils/setup/CUDA_checks.py:        print("\nINFO: Matching CUDA version detected! Enabled GPU processing.")
utils/setup/CUDA_checks.py:            "\nWARNING: Incompatible CUDA version found! OmniTrax requires CUDA",
utils/setup/CUDA_checks.py:            required_CUDA_version,
utils/setup/CUDA_checks.py:            "\n         For more information on CUDA version matching refer to: "
utils/setup/CUDA_checks.py:            "\n         https://github.com/FabianPlum/OmniTrax/blob/main/docs/CUDA_installation_guide.md"
utils/setup/CUDA_checks.py:    check_CUDA_installation(required_CUDA_version="11.2")
utils/setup/check_packages.py:from .CUDA_checks import check_CUDA_installation
utils/setup/check_packages.py:required_CUDA_version = "11.2"
utils/setup/check_packages.py:CUDA_match = check_CUDA_installation(required_CUDA_version=required_CUDA_version)
utils/setup/check_packages.py:    # add line on CUDA version matching
utils/setup/check_packages.py:    if CUDA_match:
utils/setup/check_packages.py:        setup_state_f_contents.append("Matching CUDA version=True")
utils/setup/check_packages.py:        setup_state_f_contents.append("Matching CUDA version=False")
utils/setup/check_packages.py:            "INFO: for GPU, OPENMP, AVX, or OPENCV support, edit Makefile and make again!"
utils/darknet_sub_process/example_top_level_script.py:            "--GPU",
utils/darknet_sub_process/darknet_evaluation_main.py:    ap.add_argument("-GPU", "--GPU", default="0", required=False, type=str)
utils/darknet_sub_process/darknet_evaluation_main.py:    # set which GPU to use
utils/darknet_sub_process/darknet_evaluation_main.py:    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
utils/darknet_sub_process/darknet_evaluation_main.py:    os.environ["CUDA_VISIBLE_DEVICES"] = args["GPU"]
utils/darknet_sub_process/opencv_direct_darknet.py:net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
utils/darknet_sub_process/opencv_direct_darknet.py:net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
utils/darknet_sub_process/opencv_direct_darknet.py:# net.setPreferableTarget(cv.dnn.DNN_TARGET_GPU)
utils/compute/ui.py:    Select (CUDA) computation device to run inference
utils/compute/ui.py:            if device.device_type == "GPU":
utils/compute/ui.py:                        "GPU_" + str(d - 1),
utils/compute/ui.py:                        "GPU_" + str(d - 1),
utils/compute/ui.py:                        "Use GPU for inference (requires CUDA)",
utils/omni_trax_utils.py:    Set the (CUDA) inference device for the project
utils/omni_trax_utils.py:    :param device: e.g. "CPU_0" or "GPU_0"
utils/omni_trax_utils.py:        # Logical device was not created for first GPU
__init__.py:from .utils.setup.CUDA_checks import check_CUDA_installation

```
