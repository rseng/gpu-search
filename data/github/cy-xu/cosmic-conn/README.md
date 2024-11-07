# https://github.com/cy-xu/cosmic-conn

```console
setup.py:        "Environment :: GPU :: NVIDIA CUDA",
docs/source/installation.rst:Install for a CUDA-enabled GPU
docs/source/installation.rst:.. Note:: If you are using a Mac or a computer without a dedicated Nvidia GPU, please continue to `Install for CPU`_.
docs/source/installation.rst:We build Cosmic-CoNN with ``PyTorch``, a machine learning framework that excels with GPU acceleration. In order to detect CRs quickly, it's helpful to determine if your machine has a CUDA-enabled graphics card and configure ``PyTorch`` for GPU before installing Cosmic-CoNN.
docs/source/installation.rst:A list of CUDA-enabled Nvidia GPUs  
docs/source/installation.rst:https://developer.nvidia.com/cuda-gpus
docs/source/installation.rst:NVIDIA CUDA Installation Guide for Linux  
docs/source/installation.rst:https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
docs/source/installation.rst:There are many resources online to help you configure the right Nvidia driver and CUDA library. A simple way to verify the correct setup is with the command::
docs/source/installation.rst:    $ nvidia-smi
docs/source/installation.rst:.. image:: ../_static/verify_gpu.png
docs/source/installation.rst:        :alt: an image shows Nvidia driver and CUDA properly configured
docs/source/installation.rst:If you see a similar output, congratulations! You are very close to enjoy GPU acceleration. Now please visit ``PyTorch`` installation guide to generate the correct installation command based on your environment: https://pytorch.org/get-started/locally/. Select one of the CUDA versions for the ``Compute Platform`` condition. 
docs/source/installation.rst:To verify PyTorch is correctly configured for GPU, you should see:
docs/source/installation.rst:    torch.cuda.is_available()
docs/source/installation.rst:Continue with `Install for CPU`_ to finish the installation. Since you have ``PyTorch`` configured for GPU already, it will be ignored in the next section.
docs/source/installation.rst:.. Note:: Detection time varies based on data and hardware. Although it is easy to achieve ~10x speed up with GPU acceleration, processing time on CPU is not slow. A regular laptop with AMD Ryzen 5900HS CPU takes only ~7s to process a 2009x2009 px image from LCO's 2-meter telescope.
cosmic_conn/evaluation/HST_ROC_PR.py:                         inpaint='ACS-WFC-F606W-2-32', device='GPU')
cosmic_conn/evaluation/utils.py:    # dump to gpu not
cosmic_conn/web_app/static/GlobalParas.js:const GPU_DETECT_SERVICE = 'detect_gpu'
cosmic_conn/web_app/static/script.js:    const status_string = 'GPU not found, detection will take longer.'
cosmic_conn/web_app/static/script.js:    check_gpu_info()
cosmic_conn/web_app/static/script.js:function check_gpu_info() {
cosmic_conn/web_app/static/script.js:        xhr.open('POST', HOST_URL + 'detect_gpu')
cosmic_conn/web_app/static/script.js:            if (response.gpu_detected)
cosmic_conn/web_app/static/script.js:                status_indicator.display_status_info("GPU found!")
cosmic_conn/web_app/static/script.js:                status_indicator.display_status_info("GPU not found, detection will take longer.")
cosmic_conn/web_app/static/script.js:            status_indicator.display_status_info("Got invalid GPU status!")
cosmic_conn/web_app/app.py:@app.route("/detect_gpu", methods=["POST"])
cosmic_conn/web_app/app.py:def detect_gpu():
cosmic_conn/web_app/app.py:    # gpu could have different number but cpu is consistent
cosmic_conn/web_app/app.py:    gpu_not_found = str(app.config['cr_model'].device) == 'cpu'
cosmic_conn/web_app/app.py:    # 1. call gpu function
cosmic_conn/web_app/app.py:    # 2. json obj {"gpu_found": true}
cosmic_conn/web_app/app.py:    response = {'gpu_detected': not gpu_not_found}
cosmic_conn/dl_framework/utils_ml.py:def tensor2np(gpu_tensor):
cosmic_conn/dl_framework/utils_ml.py:    if isinstance(gpu_tensor, torch.Tensor):
cosmic_conn/dl_framework/utils_ml.py:        return gpu_tensor.detach().cpu().numpy()
cosmic_conn/dl_framework/utils_ml.py:        return gpu_tensor
cosmic_conn/dl_framework/utils_ml.py:    # check if the machien/GPU has sufficient memory for
cosmic_conn/dl_framework/utils_ml.py:    GPU_THRESHOLD = 8 * (1024**3)  # 8 GB free memory
cosmic_conn/dl_framework/utils_ml.py:        # GPU available memory
cosmic_conn/dl_framework/utils_ml.py:        t = torch.cuda.get_device_properties(device).total_memory
cosmic_conn/dl_framework/utils_ml.py:        # r = torch.cuda.memory_reserved(device)
cosmic_conn/dl_framework/utils_ml.py:        # a = torch.cuda.memory_allocated(device)
cosmic_conn/dl_framework/utils_ml.py:        full_image_detection = t > GPU_THRESHOLD
cosmic_conn/dl_framework/utils_ml.py:    if torch.cuda.is_available():
cosmic_conn/dl_framework/utils_ml.py:        gaussian_filter.to(torch.device("cuda"))
cosmic_conn/dl_framework/utils_ml.py:    if torch.cuda.is_available():
cosmic_conn/dl_framework/utils_ml.py:        device = torch.device("cuda")
cosmic_conn/dl_framework/utils_ml.py:def tensor2uint8(gpu_tensor, dtype="uint8", method="clipping"):
cosmic_conn/dl_framework/utils_ml.py:    dims = len(gpu_tensor.shape)
cosmic_conn/dl_framework/utils_ml.py:        array = tensor2np(gpu_tensor)
cosmic_conn/dl_framework/utils_ml.py:        array = tensor2np(gpu_tensor.unsqueeze(0))
cosmic_conn/dl_framework/cosmic_conn.py:        if torch.cuda.is_available():
cosmic_conn/dl_framework/cosmic_conn.py:            self.device = torch.device("cuda")
cosmic_conn/dl_framework/cosmic_conn.py:            logging.info("...GPU found, yeah!")
cosmic_conn/dl_framework/cosmic_conn.py:            logging.info("...GPU or CUDA not detected, using CPU (slower). ")
cosmic_conn/dl_framework/cosmic_conn.py:                torch.cuda.empty_cache()
cosmic_conn/dl_framework/cosmic_conn.py:        # send to gpu for faster median frame calculation
cosmic_conn/dl_framework/cosmic_conn.py:            # calculate median on GPU is faster, -3 is the frame axis in both cases
cosmic_conn/dl_framework/dataloader.py:    # randomly sample 3 frames from a larger stack to save and maintain consistent GPU ram
cosmic_conn/dl_framework/dataloader.py:        # randomly sample 3 frames from a larger stack to save and maintain consistent GPU ram
README.md:We recommend installing Cosmic-CoNN in a new virtual environment, see the step-by-step [installation guide](https://cosmic-conn.readthedocs.io/en/latest/source/installation.html). To get a ~10x speed-up with GPU acceleration, see [Install for a CUDA-enabled GPU](https://cosmic-conn.readthedocs.io/en/latest/source/installation.html).
README.md:[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Interactive_Segmentation_and_Visualization_for_Tiny_Objects_in_Multi-Megapixel_Images_CVPR_2022_paper.html)
scripts/train_hst.sh:CUDA_VISIBLE_DEVICES=0 python cosmic_conn/train.py \
scripts/abalation_study/deepcr+gn.sh:CUDA_VISIBLE_DEVICES=0 python cosmic_conn/train.py \
scripts/abalation_study/deepcr+1024+gn.sh:CUDA_VISIBLE_DEVICES=0 python cosmic_conn/train.py \
scripts/abalation_study/deepcr+1024.sh:CUDA_VISIBLE_DEVICES=0 python cosmic_conn/train.py \
scripts/abalation_study/cosmic-conn.sh:CUDA_VISIBLE_DEVICES=0 python cosmic_conn/train.py \
scripts/abalation_study/deepcr.sh:CUDA_VISIBLE_DEVICES=0 python cosmic_conn/train.py \
scripts/abalation_study/deepcr+loss.sh:CUDA_VISIBLE_DEVICES=0 python cosmic_conn/train.py \
scripts/train_lco.sh:CUDA_VISIBLE_DEVICES=0 python cosmic_conn/train.py \
paper_utils/NRES_prediction_compare.py:# os.environ['CUDA_VISIBLE_DEVICES']="1"
paper_utils/development_env.yml:  - cudatoolkit=10.1.243=h6bb024c_0
paper_utils/development_env.yml:  - pytorch=1.6.0=py3.7_cuda10.1.243_cudnn7.6.3_0
paper_utils/development_env.yml:    - gpustat==0.6.0
paper_utils/development_env.yml:    - nvidia-ml-py3==7.352.0
paper_utils/prediction_compare.py:# os.environ['CUDA_VISIBLE_DEVICES']="1"
paper_utils/cr_triplets.py:os.environ['CUDA_VISIBLE_DEVICES']="1"

```
