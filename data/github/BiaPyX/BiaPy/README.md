# https://github.com/BiaPyX/BiaPy

```console
pyproject.toml:    "Environment :: GPU :: NVIDIA CUDA :: 11.8",
biapy/config/config.py:        # Do not set it as its value will be calculated based in --gpu input arg
biapy/config/config.py:        _C.SYSTEM.NUM_GPUS = 0
biapy/config/config.py:        # Device to be used when GPU is NOT selected. Most commonly "cpu", but also potentially "mps",
biapy/config/config.py:        # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
biapy/config/config.py:        # If memory or # gpus is limited, use this variable to maintain the effective batch size, which is
biapy/config/config.py:        # batch_size (per gpu) * nodes * (gpus per node) * accum_iter.
biapy/models/rcan.py:    <https://openaccess.thecvf.com/content_ECCV_2018/html/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.html>`_.
biapy/models/unetr.py:    <https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html>`_.
biapy/models/__init__.py:        Using device. Most commonly "cpu" or "cuda" for GPU, but also potentially "mps",
biapy/data/data_3D_manipulation.py:    Extract 3D patches into smaller patches with a defined overlap. Is supports multi-GPU inference
biapy/data/data_3D_manipulation.py:    by setting ``total_ranks`` and ``rank`` variables. Each GPU will process a evenly number of
biapy/data/data_3D_manipulation.py:    number of GPUs the first GPUs will process one more volume.
biapy/data/data_3D_manipulation.py:        Total number of GPUs.
biapy/data/data_3D_manipulation.py:        Rank of the current GPU.
biapy/data/data_3D_manipulation.py:        Volumes in ``Z`` axis that each GPU will process. E.g. ``[[0, 1, 2], [3, 4]]`` means that
biapy/data/data_3D_manipulation.py:        the first GPU will process volumes ``0``, ``1`` and ``2`` (``3`` in total) whereas the second
biapy/data/data_3D_manipulation.py:        GPU will process volumes ``3`` and ``4``.
biapy/data/data_3D_manipulation.py:        print(f"List of volume IDs to be processed by each GPU: {list_of_vols_in_z}")
biapy/data/data_3D_manipulation.py:            "Rank {}: Total number of patches: {} - {} patches per (z,y,x) axis (per GPU)".format(
biapy/data/post_processing/post_processing.py:        with torch.cuda.amp.autocast():
biapy/data/post_processing/post_processing.py:        with torch.cuda.amp.autocast():
biapy/data/generators/pair_base_data_generator.py:        Whether to create `Noise2Void <https://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf>`__
biapy/data/generators/__init__.py:    # To not create more than 8 processes per GPU
biapy/data/generators/__init__.py:    if cfg.SYSTEM.NUM_GPUS >= 1:
biapy/data/generators/__init__.py:        num_workers = min(num_workers, 8 * cfg.SYSTEM.NUM_GPUS)
biapy/engine/metrics.py:            Using device. Most commonly "cpu" or "cuda" for GPU, but also potentially "mps",
biapy/engine/metrics.py:            Using device. Most commonly "cpu" or "cuda" for GPU, but also potentially "mps",
biapy/engine/semantic_seg.py:            with torch.cuda.amp.autocast():
biapy/engine/detection.py:            with torch.cuda.amp.autocast():
biapy/engine/image_to_image.py:                with torch.cuda.amp.autocast():
biapy/engine/check_configuration.py:            ", full image statistics will be disabled to avoid GPU memory overflow"
biapy/engine/instance_seg.py:            with torch.cuda.amp.autocast():
biapy/engine/self_supervised.py:                with torch.cuda.amp.autocast():
biapy/engine/base_workflow.py:            maxsize = min(10, self.cfg.SYSTEM.NUM_GPUS * 10)
biapy/engine/base_workflow.py:                device_ids=[self.args.gpu],
biapy/engine/base_workflow.py:            if self.cfg.SYSTEM.NUM_GPUS > 1:
biapy/engine/base_workflow.py:            if self.cfg.TEST.VERBOSE and self.cfg.SYSTEM.NUM_GPUS > 1:
biapy/engine/base_workflow.py:                        with torch.cuda.amp.autocast():
biapy/engine/base_workflow.py:        if self.cfg.SYSTEM.NUM_GPUS > 1:
biapy/engine/base_workflow.py:                if self.cfg.SYSTEM.NUM_GPUS > 1:
biapy/engine/base_workflow.py:                    # Obtain parts of the data created by all GPUs
biapy/engine/base_workflow.py:                    if max(1, self.cfg.SYSTEM.NUM_GPUS) != len(data_parts_filenames) != len(list_of_vols_in_z):
biapy/engine/base_workflow.py:                        raise ValueError("Number of data parts is not the same as number of GPUs")
biapy/engine/base_workflow.py:                                slice(z_vol_info[k][0], z_vol_info[k][1]),  # z (only z axis is distributed across GPUs)
biapy/engine/base_workflow.py:        if self.cfg.SYSTEM.NUM_GPUS > 1:
biapy/engine/base_workflow.py:                        with torch.cuda.amp.autocast():
biapy/engine/base_workflow.py:                    with torch.cuda.amp.autocast():
biapy/engine/base_workflow.py:    Extract patches from data and put them into a queue read by each GPU inference process.
biapy/engine/base_workflow.py:    if verbose and cfg.SYSTEM.NUM_GPUS > 1:
biapy/engine/base_workflow.py:        total_ranks=max(1, cfg.SYSTEM.NUM_GPUS),
biapy/engine/base_workflow.py:    if verbose and cfg.SYSTEM.NUM_GPUS > 1:
biapy/engine/base_workflow.py:    Insert predicted patches (in ``output_queue``) in its original position in a H5/Zarr file. Each GPU will create
biapy/engine/base_workflow.py:    if verbose and cfg.SYSTEM.NUM_GPUS > 1:
biapy/engine/base_workflow.py:    if verbose and cfg.SYSTEM.NUM_GPUS > 1:
biapy/engine/super_resolution.py:                with torch.cuda.amp.autocast():
biapy/engine/train_engine.py:        with torch.cuda.amp.autocast(enabled=False):
biapy/engine/train_engine.py:        with torch.cuda.amp.autocast(enabled=False):
biapy/engine/classification.py:            with torch.cuda.amp.autocast():
biapy/engine/denoising.py:                with torch.cuda.amp.autocast():
biapy/utils/scripts/run_checks.py:parser.add_argument("--gpus", type=str, help="GPUs to use")
biapy/utils/scripts/run_checks.py:gpu = args.gpus.split(",")[0] # "0"
biapy/utils/scripts/run_checks.py:gpus = args.gpus # "0,1" # For those tests that use more than one
biapy/utils/scripts/run_checks.py:print(f"Using GPU: '{gpu}' (single-gpu checks) ; GPUs: '{gpus}' (multi-gpu checks)")
biapy/utils/scripts/run_checks.py:def runjob(test_info, results_folder, yaml_file, biapy_folder, multigpu=False, bmz=False, bmz_package=None, reuse_original_bmz_config=False):
biapy/utils/scripts/run_checks.py:               "--gpu", gpu]
biapy/utils/scripts/run_checks.py:        if multigpu:
biapy/utils/scripts/run_checks.py:                "--gpu", gpus]
biapy/utils/scripts/run_checks.py:                test_info["jobname"], "--run_id", "1", "--gpu", gpu]
biapy/utils/scripts/run_checks.py:        runjob(all_test_info["Test11"], results_folder, test_file, biapy_folder, multigpu=True)
biapy/utils/scripts/run_checks.py:        runjob(all_test_info["Test26"], results_folder, test_file, biapy_folder, multigpu=True)
biapy/utils/scripts/export_bmz_test.py:parser.add_argument("--gpu", required=True, help="GPU to use")
biapy/utils/scripts/export_bmz_test.py:biapy = BiaPy(args["config"], result_dir=args["result_dir"], name=args["jobname"], run_id=1, gpu=args["gpu"])
biapy/utils/misc.py:        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
biapy/utils/misc.py:        os.environ["LOCAL_RANK"] = str(args.gpu)
biapy/utils/misc.py:        assert torch.cuda.is_available(), "Distributed training without GPUs is not supported!"
biapy/utils/misc.py:        args.gpu = int(os.environ["LOCAL_RANK"])
biapy/utils/misc.py:        and args.gpu is not None
biapy/utils/misc.py:        and len(np.unique(np.array(args.gpu.strip().split(",")))) > 1
biapy/utils/misc.py:        args.gpu = args.rank % torch.cuda.device_count()
biapy/utils/misc.py:        if torch.cuda.is_available() and args.gpu is not None:
biapy/utils/misc.py:            device = torch.device("cuda")
biapy/utils/misc.py:    torch.cuda.set_device(args.gpu)
biapy/utils/misc.py:        "| distributed init (rank {}): {}, gpu {}".format(args.rank, args.dist_url, args.gpu),
biapy/utils/misc.py:        os.environ["NCCL_BLOCKING_WAIT"] = "0"  # not to enforce timeout in nccl backend
biapy/utils/misc.py:    device = torch.device("cuda" if torch.cuda.is_available() else cfg.SYSTEM.DEVICE)
biapy/utils/misc.py:        x_reduce = torch.tensor(x).cuda()
biapy/utils/misc.py:        self._scaler = torch.cuda.amp.GradScaler()
biapy/utils/misc.py:        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
biapy/utils/misc.py:        if torch.cuda.is_available() and self.verbose:
biapy/utils/misc.py:                if torch.cuda.is_available() and self.verbose:
biapy/utils/misc.py:                            memory=torch.cuda.max_memory_allocated() / MB,
biapy/utils/env/Dockerfile:FROM nvidia/cuda:11.8.0-base-ubuntu22.04
biapy/utils/env/Dockerfile:# Install Pytorch 2.2.0 + CUDA 11.8
biapy/utils/env/Dockerfile:WORKDIR /installations/miniconda3/envs/BiaPy_env/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/
biapy/utils/env/Dockerfile_CUDA10.2:FROM andrewseidl/nvidia-cuda:10.2-base-ubuntu20.04
biapy/utils/env/Dockerfile_CUDA10.2:# Install Pytorch 1.12.1 + CUDA 10.2
biapy/utils/env/Dockerfile_CUDA10.2:RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
biapy/utils/env/Dockerfile_CUDA10.2:WORKDIR /installations/miniconda3/envs/BiaPy_env/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/
biapy/__init__.py:    #     --gpu 0
biapy/__init__.py:    #     --gpu 0,1
biapy/__init__.py:        "--gpu",
biapy/__init__.py:        help="GPU number according to 'nvidia-smi' command / MPS device (Apple Silicon)",
biapy/__init__.py:        default="nccl",
biapy/__init__.py:        choices=["nccl", "gloo"],
biapy/_biapy.py:        gpu: Optional[str] = "",
biapy/_biapy.py:        dist_backend: Optional[str] = "nccl",
biapy/_biapy.py:        gpu: str, optional
biapy/_biapy.py:            GPU number according to 'nvidia-smi' command. Defaults to None.
biapy/_biapy.py:            Backend to use in distributed mode. Should be either 'nccl' or 'gloo'. Defaults to 'nccl'.
biapy/_biapy.py:        if dist_backend not in ["nccl", "gloo"]:
biapy/_biapy.py:            raise ValueError("Invalid value for 'dist_backend'. Should be either 'nccl' or 'gloo'.")
biapy/_biapy.py:            gpu=gpu,
biapy/_biapy.py:        # GPU selection
biapy/_biapy.py:        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
biapy/_biapy.py:        if self.args.gpu:
biapy/_biapy.py:            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
biapy/_biapy.py:            self.num_gpus = len(np.unique(np.array(self.args.gpu.strip().split(","))))
biapy/_biapy.py:            opts.extend(["SYSTEM.NUM_GPUS", self.num_gpus])
biapy/_biapy.py:        # GPU management

```
