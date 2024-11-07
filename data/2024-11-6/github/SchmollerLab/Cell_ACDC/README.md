# https://github.com/SchmollerLab/Cell_ACDC

```console
cellacdc/models/omnipose/acdcSegment.py:            gpu=False
cellacdc/models/omnipose/acdcSegment.py:            gpu=gpu, net_avg=net_avg, model_type=model_type
cellacdc/models/cellpose_v3/acdcSegment.py:            gpu=False,
cellacdc/models/cellpose_v3/acdcSegment.py:        gpu : bool, optional
cellacdc/models/cellpose_v3/acdcSegment.py:            If True and PyTorch for your GPU is correctly installed, 
cellacdc/models/cellpose_v3/acdcSegment.py:            denoising and segmentation processes will run on the GPU. 
cellacdc/models/cellpose_v3/acdcSegment.py:            (torch.device('cuda') or torch.device('cpu')). 
cellacdc/models/cellpose_v3/acdcSegment.py:            It overrides `gpu`, recommended if you want to use a specific GPU 
cellacdc/models/cellpose_v3/acdcSegment.py:            (e.g. torch.device('cuda:1'). Default is None
cellacdc/models/cellpose_v3/acdcSegment.py:            model_type, gpu=gpu, device=device
cellacdc/models/cellpose_v3/acdcSegment.py:                gpu=gpu, 
cellacdc/models/cellpose_v3/_denoise.py:            gpu=False, 
cellacdc/models/cellpose_v3/_denoise.py:        gpu : bool, optional
cellacdc/models/cellpose_v3/_denoise.py:            If True and PyTorch for your GPU is correctly installed, 
cellacdc/models/cellpose_v3/_denoise.py:            denoising will run on the GPU. Default is False
cellacdc/models/cellpose_v3/_denoise.py:        super().__init__(gpu=gpu, model_type=model_name)
cellacdc/models/omnipose_custom/acdcSegment.py:    def __init__(self, model_path: os.PathLike = '', net_avg=False, gpu=False):
cellacdc/models/omnipose_custom/acdcSegment.py:            gpu=gpu, net_avg=net_avg, pretrained_model=model_path
cellacdc/models/YeaZ/unet/model.py:# Turn off GPU access so can train and use the YeaZ-GUI
cellacdc/models/YeaZ/unet/model.py:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cellacdc/models/YeaZ/unet/model.py:config.gpu_options.allow_growth = True
cellacdc/models/YeaZ_v2/acdcSegment.py:        if torch.cuda.is_available():
cellacdc/models/YeaZ_v2/acdcSegment.py:            device = torch.device('cuda')
cellacdc/models/YeaZ_v2/acdcSegment.py:            self._is_gpu = True
cellacdc/models/YeaZ_v2/acdcSegment.py:            self._is_gpu = True
cellacdc/models/YeaZ_v2/acdcSegment.py:            self._is_gpu = False
cellacdc/models/YeaZ_v2/acdcSegment.py:        if self._is_gpu:
cellacdc/models/YeaZ_v2/acdcSegment.py:        if self._is_gpu:
cellacdc/models/YeaZ_v2/acdcSegment.py:                torch.cuda.empty_cache()
cellacdc/models/DeepSea/acdcSegment.py:torch.cuda.manual_seed(SEED)
cellacdc/models/DeepSea/acdcSegment.py:    def __init__(self, gpu=False):
cellacdc/models/DeepSea/acdcSegment.py:            'segmentation.pth', DeepSeaSegmentation, gpu=gpu
cellacdc/models/DeepSea/__init__.py:        checkpoint_filename, DeepSeaClass, gpu=False
cellacdc/models/DeepSea/__init__.py:    if gpu:
cellacdc/models/DeepSea/__init__.py:            device = 'cuda'
cellacdc/models/segment_anything/acdcSegment.py:            gpu=False
cellacdc/models/segment_anything/acdcSegment.py:        if gpu:
cellacdc/models/segment_anything/acdcSegment.py:                device = 'cuda'
cellacdc/models/cellpose_v2/acdcSegment.py:            gpu=False,
cellacdc/models/cellpose_v2/acdcSegment.py:                    gpu=gpu, model_type=model_type, device=device
cellacdc/models/cellpose_v2/acdcSegment.py:                    gpu=gpu, 
cellacdc/models/cellpose_v2/acdcSegment.py:                    gpu=gpu, net_avg=net_avg, model_type=model_type
cellacdc/models/cellpose_v2/acdcSegment.py:                    gpu=gpu, net_avg=net_avg, model_type=model_type
cellacdc/models/Cellpose_germlineNuclei/acdcSegment.py:            gpu=False
cellacdc/models/Cellpose_germlineNuclei/acdcSegment.py:            gpu=gpu, diam_mean=30, pretrained_model=model_path
cellacdc/models/cellpose_custom/acdcSegment.py:    def __init__(self, model_path: os.PathLike = '', net_avg=False, gpu=False):
cellacdc/models/cellpose_custom/acdcSegment.py:                gpu=gpu, net_avg=net_avg, pretrained_model=model_path
cellacdc/models/cellpose_custom/acdcSegment.py:                gpu=gpu, pretrained_model=model_path
cellacdc/apps.py:            ['CPU', 'CUDA 11.8 (NVIDIA GPU)', 'CUDA 12.1 (NVIDIA GPU)']
cellacdc/gui.py:            use_gpu = win.init_kwargs.get('gpu', False)
cellacdc/gui.py:            proceed = myutils.check_cuda(model_name, use_gpu, qparent=self)
cellacdc/gui.py:        use_gpu = win.init_kwargs.get('gpu', False)
cellacdc/gui.py:        proceed = myutils.check_cuda(model_name, use_gpu, qparent=self)
cellacdc/gui.py:        use_gpu = win.init_kwargs.get('gpu', False)
cellacdc/gui.py:        proceed = myutils.check_cuda(model_name, use_gpu, qparent=self)
cellacdc/trackers/DeepSea/DeepSea_tracker.py:torch.cuda.manual_seed(SEED)
cellacdc/trackers/DeepSea/DeepSea_tracker.py:    def __init__(self, gpu=False):
cellacdc/trackers/DeepSea/DeepSea_tracker.py:            'tracker.pth', DeepSeaTracker, gpu=gpu
cellacdc/trackers/Trackastra/Trackastra_tracker.py:            gpu=False
cellacdc/trackers/Trackastra/Trackastra_tracker.py:        gpu : bool, optional
cellacdc/trackers/Trackastra/Trackastra_tracker.py:            If `True`, attempts to try to use the GPU for inference. 
cellacdc/segm.py:        use_gpu = init_kwargs.get('gpu', False)
cellacdc/segm.py:        proceed = myutils.check_cuda(model_name, use_gpu, qparent=self)
cellacdc/preprocess.py:    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
cellacdc/preprocess.py:        use_gpu=False, logger_func=print
cellacdc/preprocess.py:    use_gpu : bool, optional
cellacdc/preprocess.py:    if CUPY_INSTALLED and use_gpu:
cellacdc/preprocess.py:            filtered = gpu_gaussian_filter(image, sigma)
cellacdc/preprocess.py:                '[WARNING]: GPU acceleration of the gaussian filter failed. '
cellacdc/preprocess.py:        image, spots_zyx_radii_pxl: types.Vector, use_gpu=False, 
cellacdc/preprocess.py:        image, sigma1, use_gpu=use_gpu, logger_func=logger_func
cellacdc/preprocess.py:        image, sigma2, use_gpu=use_gpu, logger_func=logger_func
cellacdc/myutils.py:            'CPU', 'CUDA 11.8 (NVIDIA GPU)', 'CUDA 12.1 (NVIDIA GPU)'
cellacdc/myutils.py:def _warn_install_torch_cuda(model_name, qparent=None):
cellacdc/myutils.py:    cellpose_cuda_url = (
cellacdc/myutils.py:        r'https://github.com/mouseland/cellpose#gpu-version-cuda-on-windows-or-linux'
cellacdc/myutils.py:    torch_cuda_url = (
cellacdc/myutils.py:    cellpose_href = f'{html_utils.href_tag("here", cellpose_cuda_url)}'
cellacdc/myutils.py:    torch_href = f'{html_utils.href_tag("here", torch_cuda_url)}'
cellacdc/myutils.py:        In order to use <code>{model_name}</code> with the GPU you need 
cellacdc/myutils.py:        to install the <b>CUDA version of PyTorch</b>.<br><br>
cellacdc/myutils.py:        We <b>highly recommend using Anaconda</b> to install PyTorch GPU.<br><br>
cellacdc/myutils.py:        Then, install the CUDA version required by your GPU with the follwing 
cellacdc/myutils.py:        <code>conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia</code>
cellacdc/myutils.py:    proceedButton = widgets.okPushButton('Proceed without GPU')
cellacdc/myutils.py:        qparent, 'PyTorch GPU version not installed', txt, 
cellacdc/myutils.py:def check_cuda(model_name, use_gpu, qparent=None):
cellacdc/myutils.py:    if not use_gpu:
cellacdc/myutils.py:        if not torch.cuda.is_available():
cellacdc/myutils.py:            proceed = _warn_install_torch_cuda(model_name, qparent=qparent)
cellacdc/myutils.py:        if not torch.cuda.device_count() > 0:
cellacdc/myutils.py:            proceed = _warn_install_torch_cuda(model_name, qparent=qparent)
cellacdc/myutils.py:def get_torch_device(gpu=False):
cellacdc/myutils.py:    if torch.cuda.is_available() and gpu:
cellacdc/myutils.py:        device = torch.device('cuda')
cellacdc/__init__.py:            'CUDA 11.8 (NVIDIA GPU)': 'conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia',
cellacdc/__init__.py:            'CUDA 12.1 (NVIDIA GPU)': 'conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia'
cellacdc/__init__.py:            'CUDA 11.8 (NVIDIA GPU)': 'python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118',
cellacdc/__init__.py:            'CUDA 12.1 (NVIDIA GPU)': 'python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121'
cellacdc/__init__.py:            'CUDA 11.8 (NVIDIA GPU)': '[WARNING]: CUDA is not available on MacOS',
cellacdc/__init__.py:            'CUDA 12.1 (NVIDIA GPU)': '[WARNING]: CUDA is not available on MacOS'
cellacdc/__init__.py:            'CUDA 11.8 (NVIDIA GPU)': '[WARNING]: CUDA is not available on MacOS',
cellacdc/__init__.py:            'CUDA 12.1 (NVIDIA GPU)': '[WARNING]: CUDA is not available on MacOS'
cellacdc/__init__.py:            'CUDA 11.8 (NVIDIA GPU)': 'conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia',
cellacdc/__init__.py:            'CUDA 12.1 (NVIDIA GPU)': 'conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia'
cellacdc/__init__.py:            'CUDA 11.8 (NVIDIA GPU)': 'pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118',
cellacdc/__init__.py:            'CUDA 12.1 (NVIDIA GPU)': 'pip3 install torch torchvision'

```
