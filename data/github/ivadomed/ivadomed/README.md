# https://github.com/ivadomed/ivadomed

```console
CHANGES.md: - Fix GPU behavior in segment_volume.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1209)
CHANGES.md:- Segment with ONNX or PT model based on CPU/GPU availability.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1086)
CHANGES.md:- Fixing mix-up for GPU training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1063)
CHANGES.md:- Improve Installation Doc Readability based for Step 3 relating to GPU setup.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1037)
CHANGES.md:- Update installation instruction to fit recent CUDA11 and torch 1.8+ push.  [View pull request](https://github.com/ivadomed/ivadomed/pull/969)
CHANGES.md:- Pin to CUDA-11.  [View pull request](https://github.com/ivadomed/ivadomed/pull/951)
CHANGES.md: - Standardize the gpu ID argument.  [View pull request](https://github.com/ivadomed/ivadomed/pull/644)
CHANGES.md: - fix run_test gpu assignation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/453)
docs/source/tutorials/two_class_microscopy_seg_2d_unet.rst:   If a `compatible GPU <https://pytorch.org/get-started/locally/>`_ is available, it will be used by default.
docs/source/tutorials/two_class_microscopy_seg_2d_unet.rst:   Using GPU ID 0
docs/source/tutorials/two_class_microscopy_seg_2d_unet.rst:   Using GPU ID 0
docs/source/tutorials/cascaded_architecture.rst:                    "gpu_ids": [0],
docs/source/tutorials/automate_training.rst:depending on your computer, this could be quite slow (especially if you don't have any GPUs).
docs/source/tutorials/one_class_segmentation_2d_unet.rst:                    "gpu_ids": [0],
docs/source/tutorials/one_class_segmentation_2d_unet.rst:       If a `compatible GPU <https://pytorch.org/get-started/locally/>`_ is available, it will be used by default.
docs/source/tutorials/one_class_segmentation_2d_unet.rst:       Cuda is not available.
docs/source/tutorials/one_class_segmentation_2d_unet.rst:       Cuda is not available.
docs/source/installation.rst:    Currently, ``ivadomed`` supports GPU/CPU on ``Linux`` and ``Windows``, and CPU only on ``macOS`` and `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/>`_.
docs/source/comparison_other_projects_table.csv:**Multi-GPU data parallelism**,|no|,|yes|,|yes|,|no|,|no|,|no|,|yes|,|yes|,|no|,|no|
docs/source/configuration_file.rst:        "title": "gpu_ids",
docs/source/configuration_file.rst:        "description": "List of IDs of one or more GPUs to use. Default: ``[0]``.",
docs/source/configuration_file.rst:        "gpu_ids": [1,2,3]
docs/source/configuration_file.rst:    Currently only ``ivadomed_automate_training`` supports the use of more than one GPU.
testing/unit_tests/test_automate_training.py:        "gpu_ids": [1]
testing/unit_tests/test_automate_training.py:        "gpu_ids": [[2], [5]]
testing/unit_tests/test_automate_training.py:    batch_size = 2, loss = "DiceLoss", depth = 3, gpu_ids = [1]
testing/unit_tests/test_automate_training.py:    batch_size = 64, loss = "DiceLoss", depth = 3, gpu_ids = [1]
testing/unit_tests/test_automate_training.py:    batch_size = 18, loss = "DiceLoss", depth = 3, gpu_ids = [1]
testing/unit_tests/test_automate_training.py:    batch_size = 18, loss = "FocalLoss", depth = 3, gpu_ids = [1]
testing/unit_tests/test_automate_training.py:    batch_size = 18, loss = "DiceLoss", depth = 2, gpu_ids = [1]
testing/unit_tests/test_automate_training.py:    batch_size = 18, loss = "DiceLoss", depth = 3, gpu_ids = [1]
testing/unit_tests/test_automate_training.py:    batch_size = 18, loss = "DiceLoss", depth = 4, gpu_ids = [1]
testing/unit_tests/test_automate_training.py:    batch_size = 18, loss = "DiceLoss", depth = 3, gpu_ids = [2]
testing/unit_tests/test_automate_training.py:    batch_size = 18, loss = "DiceLoss", depth = 3, gpu_ids = [5]
testing/unit_tests/test_automate_training.py:    batch_size = 2, loss = "DiceLoss", depth = 2, gpu_ids = [2]
testing/unit_tests/test_automate_training.py:    batch_size = 64, loss = "DiceLoss", depth = 2, gpu_ids = [2]
testing/unit_tests/test_automate_training.py:    batch_size = 2, loss = "FocalLoss", depth = 2, gpu_ids = [2]
testing/unit_tests/test_automate_training.py:    batch_size = 64, loss = "FocalLoss", depth = 2, gpu_ids = [2]
testing/unit_tests/test_automate_training.py:    batch_size = 2, loss = "DiceLoss", depth = 3, gpu_ids = [2]
testing/unit_tests/test_automate_training.py:    batch_size = 64, loss = "DiceLoss", depth = 3, gpu_ids = [2]
testing/unit_tests/test_automate_training.py:    batch_size = 2, loss = "FocalLoss", depth = 3, gpu_ids = [2]
testing/unit_tests/test_automate_training.py:    batch_size = 64, loss = "FocalLoss", depth = 3, gpu_ids = [2]
testing/unit_tests/test_automate_training.py:    batch_size = 2, loss = "DiceLoss", depth = 2, gpu_ids = [5]
testing/unit_tests/test_automate_training.py:    batch_size = 64, loss = "DiceLoss", depth = 2, gpu_ids = [5]
testing/unit_tests/test_automate_training.py:    batch_size = 2, loss = "FocalLoss", depth = 2, gpu_ids = [5]
testing/unit_tests/test_automate_training.py:    batch_size = 64, loss = "FocalLoss", depth = 2, gpu_ids = [5]
testing/unit_tests/test_automate_training.py:    batch_size = 2, loss = "DiceLoss", depth = 3, gpu_ids = [5]
testing/unit_tests/test_automate_training.py:    batch_size = 64, loss = "DiceLoss", depth = 3, gpu_ids = [5]
testing/unit_tests/test_automate_training.py:    batch_size = 2, loss = "FocalLoss", depth = 3, gpu_ids = [5]
testing/unit_tests/test_automate_training.py:    batch_size = 64, loss = "FocalLoss", depth = 3, gpu_ids = [5]
testing/unit_tests/test_automate_training.py:    batch_size = 2, loss = "DiceLoss", depth = 2, gpu_ids = [2]
testing/unit_tests/test_automate_training.py:    batch_size = 64, loss = "FocalLoss", depth = 3, gpu_ids = [5]
testing/unit_tests/test_automate_training.py:    "gpu_ids": [1]
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-2-gpu_ids-2"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-3-gpu_ids-2"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-4-gpu_ids-2"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-2-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-2-gpu_ids-2"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-2-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-3-gpu_ids-2"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-2-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-4-gpu_ids-2"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-64-loss-name-DiceLoss-depth-2-gpu_ids-2"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-64-loss-name-DiceLoss-depth-3-gpu_ids-2"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-64-loss-name-DiceLoss-depth-4-gpu_ids-2"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-2-gpu_ids-2"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-3-gpu_ids-2"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-4-gpu_ids-2"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-2-gpu_ids-5"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-3-gpu_ids-5"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-4-gpu_ids-5"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-2-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-2-gpu_ids-5"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-2-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-3-gpu_ids-5"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-2-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-4-gpu_ids-5"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-64-loss-name-DiceLoss-depth-2-gpu_ids-5"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-64-loss-name-DiceLoss-depth-3-gpu_ids-5"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-64-loss-name-DiceLoss-depth-4-gpu_ids-5"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-2-gpu_ids-5"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-3-gpu_ids-5"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5],
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-4-gpu_ids-5"
testing/unit_tests/test_automate_training.py:        "gpu_ids": [1]
testing/unit_tests/test_automate_training.py:        "gpu_ids": [1]
testing/unit_tests/test_automate_training.py:        "gpu_ids": [1]
testing/unit_tests/test_automate_training.py:        "gpu_ids": [1]
testing/unit_tests/test_automate_training.py:        "gpu_ids": [1]
testing/unit_tests/test_automate_training.py:        "gpu_ids": [1]
testing/unit_tests/test_automate_training.py:        "gpu_ids": [1]
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-gpu_ids-2",
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2]
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-gpu_ids-5",
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5]
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-2-gpu_ids-2",
testing/unit_tests/test_automate_training.py:        "gpu_ids": [2]
testing/unit_tests/test_automate_training.py:        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-3-gpu_ids-5",
testing/unit_tests/test_automate_training.py:        "gpu_ids": [5]
testing/unit_tests/test_automate_training.py:    HyperparameterOption("gpu_ids", {"gpu_ids": [2]}, [2]),
testing/unit_tests/test_automate_training.py:    HyperparameterOption("gpu_ids", {"gpu_ids": [5]}, [5])
testing/unit_tests/test_automate_training.py:        "gpu_ids": [[2], [5]]
testing/unit_tests/test_sampler.py:GPU_ID = 0
testing/unit_tests/test_sampler.py:    cuda_available, device = imed_utils.define_device(GPU_ID)
testing/unit_tests/test_testing.py:GPU_ID = 0
testing/unit_tests/test_testing.py:    cuda_available, device = imed_utils.define_device(GPU_ID)
testing/unit_tests/test_testing.py:    if cuda_available:
testing/unit_tests/test_testing.py:        model.cuda()
testing/unit_tests/test_testing.py:                                                   cuda_available=cuda_available)
testing/unit_tests/test_testing.py:    cuda_available, device = imed_utils.define_device(GPU_ID)
testing/unit_tests/test_testing.py:    if cuda_available:
testing/unit_tests/test_testing.py:        model.cuda()
testing/unit_tests/test_testing.py:                                                   cuda_available=cuda_available)
testing/unit_tests/test_testing.py:    cuda_available, device = imed_utils.define_device(GPU_ID)
testing/unit_tests/test_testing.py:    if cuda_available:
testing/unit_tests/test_testing.py:        model.cuda()
testing/unit_tests/test_testing.py:                                                   cuda_available=cuda_available)
testing/unit_tests/test_slice_filter.py:GPU_ID = 0
testing/unit_tests/test_slice_filter.py:    cuda_available, device = imed_utils.define_device(GPU_ID)
testing/unit_tests/test_orientation.py:GPU_ID = 0
testing/unit_tests/test_orientation.py:    device = torch.device("cuda:" + str(GPU_ID) if torch.cuda.is_available() else "cpu")
testing/unit_tests/test_orientation.py:    cuda_available = torch.cuda.is_available()
testing/unit_tests/test_orientation.py:    if cuda_available:
testing/unit_tests/test_orientation.py:        torch.cuda.set_device(device)
testing/unit_tests/test_orientation.py:        logger.info(f"Using GPU ID {device}")
testing/unit_tests/test_patch_filter.py:GPU_ID = 0
testing/unit_tests/test_patch_filter.py:    cuda_available, device = imed_utils.define_device(GPU_ID)
testing/unit_tests/test_patch_filter.py:    cuda_available, device = imed_utils.define_device(GPU_ID)
testing/unit_tests/test_training_time.py:GPU_ID = 0
testing/unit_tests/test_training_time.py:    cuda_available, device = imed_utils.define_device(GPU_ID)
testing/unit_tests/test_training_time.py:    if cuda_available:
testing/unit_tests/test_training_time.py:        model.cuda()
testing/unit_tests/test_training_time.py:            input_samples = imed_utils.cuda(batch["input"], cuda_available)
testing/unit_tests/test_training_time.py:            gt_samples = imed_utils.cuda(batch["gt"], cuda_available, non_blocking=True)
ivadomed/testing.py:def test(model_params, dataset_test, testing_params, path_output, device, cuda_available=True,
ivadomed/testing.py:        device (torch.device): Indicates the CPU or GPU ID.
ivadomed/testing.py:        cuda_available (bool): If True, CUDA is available.
ivadomed/testing.py:    if cuda_available:
ivadomed/testing.py:        model.cuda()
ivadomed/testing.py:                                          cuda_available, i_monteCarlo, postprocessing)
ivadomed/testing.py:def run_inference(test_loader, model, model_params, testing_params, ofolder, cuda_available,
ivadomed/testing.py:        cuda_available (bool): If True, CUDA is available.
ivadomed/testing.py:                input_samples = imed_utils.cuda(imed_utils.unstack_tensors(batch["input"]), cuda_available)
ivadomed/testing.py:                input_samples = imed_utils.cuda(batch["input"], cuda_available)
ivadomed/testing.py:            gt_samples = imed_utils.cuda(batch["gt"], cuda_available, non_blocking=True)
ivadomed/testing.py:                       fname_out="thr.png", cuda_available=True):
ivadomed/testing.py:        cuda_available (bool): If True, CUDA is available.
ivadomed/testing.py:                                      cuda_available=cuda_available)
ivadomed/config/config_microscopy.json:    "gpu_ids": [0],
ivadomed/config/config_tumorSeg.json:    "gpu_ids": [0],
ivadomed/config/config_spineGeHemis.json:    "gpu_ids": [0],
ivadomed/config/config_classification.json:    "gpu_ids": [6],
ivadomed/config/config.json:    "gpu_ids": [0],
ivadomed/config/config_small.json:    "gpu_ids": [2],
ivadomed/config/config_sctTesting.json:    "gpu_ids": [6],
ivadomed/config/config_default.json:    "gpu_ids": [0],
ivadomed/config/config_vertebral_labeling.json:    "gpu_ids": [7],
ivadomed/object_detection/utils.py:def generate_bounding_box_file(subject_path_list, model_path, path_output, gpu_id=0, slice_axis=0, contrast_lst=None,
ivadomed/object_detection/utils.py:        gpu_id (int): If available, GPU number.
ivadomed/object_detection/utils.py:        object_mask, _ = imed_inference.segment_volume(model_path, [subject_path], gpu_id=gpu_id)
ivadomed/object_detection/utils.py:                                                       object_detection_params[ObjectDetectionParamsKW.GPU_IDS],
ivadomed/models.py:        if torch.cuda.is_available():
ivadomed/models.py:            context = torch.Tensor(context).cuda()
ivadomed/models.py:    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
ivadomed/models.py:    model and GPU availability) and its configuration file (.json) from it.
ivadomed/models.py:        cuda_available = torch.cuda.is_available()
ivadomed/models.py:        # Assign '.pt' or '.onnx' model based on file existence and GPU/CPU device availability
ivadomed/models.py:        # '.pt' is preferred on GPU, or on CPU if '.onnx' doesn't exist
ivadomed/models.py:        elif ((    cuda_available and     fname_model_pt.is_file()) or
ivadomed/models.py:              (not cuda_available and not fname_model_onnx.is_file())):
ivadomed/models.py:        # '.onnx' is preferred on CPU, or on GPU if '.pt' doesn't exist
ivadomed/models.py:        elif ((not cuda_available and     fname_model_onnx.is_file()) or
ivadomed/models.py:              (    cuda_available and not fname_model_pt.is_file())):
ivadomed/loader/mri3d_subvolume_segmentation_dataset.py:        # Size limit: 4GB GPU RAM, keep in mind tranform etc might take MORE!
ivadomed/loader/loader.py:                 cuda_available: bool = None,
ivadomed/loader/loader.py:        cuda_available (bool): If True, cuda is available.
ivadomed/loader/loader.py:                                                          cuda_available=cuda_available),
ivadomed/loader/mri2d_segmentation_dataset.py:        # Size limit: 4GB GPU RAM, keep in mind tranform etc might take MORE!
ivadomed/loader/slice_filter.py:        device (torch.device): Indicates the CPU or GPU ID.
ivadomed/loader/slice_filter.py:        cuda_available (bool): If True, CUDA is available.
ivadomed/loader/slice_filter.py:        device (torch.device): Indicates the CPU or GPU ID.
ivadomed/loader/slice_filter.py:        cuda_available (bool): If True, CUDA is available.
ivadomed/loader/slice_filter.py:                 cuda_available: bool = None):
ivadomed/loader/slice_filter.py:        self.cuda_available = cuda_available
ivadomed/loader/slice_filter.py:            if cuda_available:
ivadomed/loader/slice_filter.py:                        imed_utils.cuda(torch.from_numpy(img.copy()).unsqueeze(0).unsqueeze(0),
ivadomed/loader/slice_filter.py:                                        self.cuda_available))) for img in input_data]):
ivadomed/scripts/convert_to_onnx.py:    parser.add_argument("-g", "--gpu_id", dest="gpu_id", default=0, type=str,
ivadomed/scripts/convert_to_onnx.py:                        help="GPU number if available.", metavar=imed_utils.Metavar.int)
ivadomed/scripts/convert_to_onnx.py:def convert_pytorch_to_onnx(model, dimension, n_channels, gpu_id=0):
ivadomed/scripts/convert_to_onnx.py:        gpu_id (string): GPU ID, if available. Flag: ``--gpu_id``, ``-g``
ivadomed/scripts/convert_to_onnx.py:    if torch.cuda.is_available():
ivadomed/scripts/convert_to_onnx.py:        device = "cuda:" + str(gpu_id)
ivadomed/scripts/convert_to_onnx.py:    gpu_id = str(args.gpu_id)
ivadomed/scripts/convert_to_onnx.py:    convert_pytorch_to_onnx(fname_model, dimension, n_channels, gpu_id)
ivadomed/scripts/automate_training.py:This script enables training and comparison of models on multiple GPUs.
ivadomed/scripts/automate_training.py:    # ID of process used to assign a GPU
ivadomed/scripts/automate_training.py:    # Use GPU i from the array specified in the config file
ivadomed/scripts/automate_training.py:    config[ConfigKW.GPU_IDS] = [config[ConfigKW.GPU_IDS][ID]]
ivadomed/scripts/automate_training.py:    # ID of process used to assign a GPU
ivadomed/scripts/automate_training.py:    # Use GPU i from the array specified in the config file
ivadomed/scripts/automate_training.py:    config[ConfigKW.GPU_IDS] = [config[ConfigKW.GPU_IDS][ID]]
ivadomed/scripts/automate_training.py:    """Automate multiple training processes on multiple GPUs.
ivadomed/scripts/automate_training.py:    this optimization across multiple GPUs. It runs trainings, on the same training and validation
ivadomed/scripts/automate_training.py:    comparison. The script efficiently allocates each training to one of the available GPUs.
ivadomed/scripts/automate_training.py:    # CUDA problem when forking process
ivadomed/scripts/automate_training.py:    # Run all configs on a separate process, with a maximum of n_gpus  processes at a given time
ivadomed/scripts/automate_training.py:    logger.info(initial_config[ConfigKW.GPU_IDS])
ivadomed/scripts/automate_training.py:    with ctx.Pool(processes=len(initial_config[ConfigKW.GPU_IDS])) as pool:
ivadomed/scripts/automate_training.py:                    new_config[ConfigKW.GPU_IDS] = config[ConfigKW.GPU_IDS]
ivadomed/inference.py:def get_preds(context: dict, fname_model: str, model_params: dict, cuda_available: bool, device: torch.device, batch: dict) -> tensor:
ivadomed/inference.py:        cuda_available (bool): True if cuda is available.
ivadomed/inference.py:        img = imed_utils.cuda(batch['input'], cuda_available=cuda_available)
ivadomed/inference.py:def segment_volume(folder_model: str, fname_images: list, gpu_id: int = 0, options: dict = None):
ivadomed/inference.py:        gpu_id (int): Number representing gpu number if available. Currently does NOT support multiple GPU segmentation.
ivadomed/inference.py:    cuda_available, device = imed_utils.define_device(gpu_id)
ivadomed/inference.py:        preds = get_preds(context, fname_model, model_params, cuda_available, device, batch)
ivadomed/training.py:          cuda_available=True, metric_fns=None, n_gif=0, resume_training=False, debugging=False):
ivadomed/training.py:        device (str): Indicates the CPU or GPU ID.
ivadomed/training.py:        cuda_available (bool): If True, CUDA is available.
ivadomed/training.py:    if cuda_available:
ivadomed/training.py:        model.cuda()
ivadomed/training.py:                input_samples = imed_utils.cuda(imed_utils.unstack_tensors(batch["input"]), cuda_available)
ivadomed/training.py:                input_samples = imed_utils.cuda(batch["input"], cuda_available)
ivadomed/training.py:            gt_samples = imed_utils.cuda(batch["gt"], cuda_available, non_blocking=True)
ivadomed/training.py:                        input_samples = imed_utils.cuda(imed_utils.unstack_tensors(batch["input"]), cuda_available)
ivadomed/training.py:                        input_samples = imed_utils.cuda(batch["input"], cuda_available)
ivadomed/training.py:                    gt_samples = imed_utils.cuda(batch["gt"], cuda_available, non_blocking=True)
ivadomed/utils.py:def cuda(input_var, cuda_available=True, non_blocking=False):
ivadomed/utils.py:    """Passes input_var to GPU.
ivadomed/utils.py:        cuda_available (bool): If False, then return identity
ivadomed/utils.py:    if cuda_available:
ivadomed/utils.py:            return [t.cuda(non_blocking=non_blocking) for t in input_var]
ivadomed/utils.py:            return input_var.cuda(non_blocking=non_blocking)
ivadomed/utils.py:def define_device(gpu_id):
ivadomed/utils.py:        gpu_id (int): GPU ID.
ivadomed/utils.py:        Bool, device: True if cuda is available.
ivadomed/utils.py:    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
ivadomed/utils.py:    cuda_available = torch.cuda.is_available()
ivadomed/utils.py:    if not cuda_available:
ivadomed/utils.py:        logger.info("Cuda is not available.")
ivadomed/utils.py:    if cuda_available:
ivadomed/utils.py:        # Set the GPU
ivadomed/utils.py:        gpu_id = int(gpu_id)
ivadomed/utils.py:        torch.cuda.set_device(gpu_id)
ivadomed/utils.py:        logger.info(f"Using GPU ID {gpu_id}")
ivadomed/utils.py:    return cuda_available, device
ivadomed/keywords.py:    GPU_IDS = "gpu_ids"
ivadomed/keywords.py:    GPU_IDS: str = "gpu_ids"
ivadomed/main.py:def get_dataset(bids_df, loader_params, data_lst, transform_params, cuda_available, device, ds_type):
ivadomed/main.py:                                  cuda_available=cuda_available)
ivadomed/main.py:        object_detection_params.update({ObjectDetectionParamsKW.GPU_IDS: context[ConfigKW.GPU_IDS][0],
ivadomed/main.py:                                                                   gpu_id=context[ConfigKW.GPU_IDS][0],
ivadomed/main.py:    cuda_available, device = imed_utils.define_device(context[ConfigKW.GPU_IDS][0])
ivadomed/main.py:        ds_valid = get_dataset(bids_df, loader_params, valid_lst, transform_valid_params, cuda_available, device,
ivadomed/main.py:        ds_train = get_dataset(bids_df, loader_params, train_lst, transform_train_params, cuda_available, device,
ivadomed/main.py:            cuda_available=cuda_available,
ivadomed/main.py:            ds_valid = get_dataset(bids_df, loader_params, valid_lst, transform_valid_params, cuda_available, device,
ivadomed/main.py:        ds_train = get_dataset(bids_df, loader_params, train_lst, transform_valid_params, cuda_available, device,
ivadomed/main.py:                                              cuda_available=cuda_available)
ivadomed/main.py:                                           cuda_available=cuda_available)
ivadomed/main.py:                                         cuda_available=cuda_available,
paper.md:**Getting started:** It can be overwhelming to get started and choose across all the available models, losses, and parameters. `ivadomed`'s repository includes the script `ivadomed_automate_training` to configure and launch multiple trainings across GPUs. In case of interruption during training, all parameters are saved after each epoch so that training can be resumed at any time.

```
