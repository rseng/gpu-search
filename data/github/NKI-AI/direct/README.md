# https://github.com/NKI-AI/direct

```console
README.rst:We provide a set of baseline results and trained models in the `DIRECT Model Zoo <https://docs.aiforoncology.nl/direct/model_zoo.html>`_. Baselines and trained models include the `vSHARP <https://arxiv.org/abs/2309.09954>`_, `Recurrent Variational Network (RecurrentVarNet) <https://arxiv.org/abs/2111.09639>`_, the `Recurrent Inference Machine (RIM) <https://www.sciencedirect.com/science/article/abs/pii/S1361841518306078>`_, the `End-to-end Variational Network (VarNet) <https://arxiv.org/pdf/2004.06688.pdf>`_, the `Learned Primal Dual Network (LDPNet) <https://arxiv.org/abs/1707.06474>`_, the `X-Primal Dual Network (XPDNet) <https://arxiv.org/abs/2010.07290>`_, the `KIKI-Net <https://pubmed.ncbi.nlm.nih.gov/29624729/>`_, the `U-Net <https://arxiv.org/abs/1811.08839>`_, the `Joint-ICNet <https://openaccess.thecvf.com/content/CVPR2021/papers/Jun_Joint_Deep_Model-Based_MR_Image_and_Coil_Sensitivity_Reconstruction_Network_CVPR_2021_paper.pdf>`_, and the `AIRS Medical fastmri model (MultiDomainNet) <https://arxiv.org/pdf/2012.06318.pdf>`_.
docs/training.rst:    direct train <experiment_directory> --num-gpus <number_of_gpus> --cfg <path_or_url_to_yaml_file> \
docs/training.rst:    (machine0)$ direct train <experiment_directory> --num-gpus <number_of_gpus> --cfg <path_or_url_to_yaml_file> \
docs/training.rst:    (machine1)$ direct train <experiment_directory> --num-gpus <number_of_gpus> --cfg <path_or_url_to_yaml_file> \
docs/training.rst:If you are performing an experiment on a CPU (not recommended) replace ``--num-gpus <number_of_gpus>`` with ``--device 'cpu:0'``.
docs/training.rst:            --num-gpus <number_of_gpus> --cfg <path_or_url_to_yaml_file> [--other-flags]
docs/inference.rst:            --num-gpus <num_gpus> [--data-root <data_root>] [ --cfg <cfg_filename>.yaml --other-flags <other_flags>]
tests/test_train.py:        num_gpus = 0
tests/test_train.py:            num_gpus,
direct/cli/predict.py:                            --num-gpus <num_gpus> [--data-root <data_root>] [--other-flag-args <other_flags>]
direct/cli/train.py:            $ direct train experiment_dir --num-gpus 8 --cfg cfg.yaml [--training-root training_set --validation-root validation_set]
direct/predict.py:        args.num_gpus,
direct/types.py:from torch.cuda.amp import GradScaler
direct/data/transforms.py:    """fft and ifft can only be performed on GPU in float16 if the shapes are powers of 2. This function verifies if
direct/data/mri_transforms.py:    """Builds post (can be put on gpu) supervised MRI transforms.
direct/engine.py:from torch.cuda.amp import GradScaler
direct/engine.py:            Device. Can be "cuda" or "cpu".
direct/engine.py:        torch.cuda.empty_cache()
direct/engine.py:                    torch.cuda.empty_cache()
direct/engine.py:        # Mixed precision setup. This requires the model to be on the gpu.
direct/engine.py:            # TODO(jt): Check if on GPU
direct/engine.py:        self.logger.info(f"Device count: {torch.cuda.device_count()}.")
direct/engine.py:        elif torch.cuda.device_count() > 1 and communication.get_world_size() == 1:
direct/nn/conjgradnet/conjgradnet_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/vsharp/vsharp_engine.py:from torch.cuda.amp import autocast
direct/nn/vsharp/vsharp_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/vsharp/vsharp_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/vsharp/vsharp_engine.py:        Device. Can be "cuda:{idx}" or "cpu".
direct/nn/vsharp/vsharp_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/vsharp/vsharp_engine.py:        Device. Can be "cuda:{idx}" or "cpu".
direct/nn/vsharp/vsharp_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/mri_models.py:from torch.cuda.amp import autocast
direct/nn/mri_models.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/mri_models.py:        torch.cuda.empty_cache()
direct/nn/mri_models.py:            torch.cuda.empty_cache()
direct/nn/mri_models.py:        torch.cuda.empty_cache()
direct/nn/mri_models.py:        torch.cuda.empty_cache()
direct/nn/ssl/mri_models.py:from torch.cuda.amp import autocast
direct/nn/ssl/mri_models.py:            Device. Can be "cuda" or "cpu".
direct/nn/ssl/mri_models.py:            Device. Can be "cuda" or "cpu".
direct/nn/varsplitnet/varsplitnet_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/multidomainnet/multidomainnet_engine.py:from torch.cuda.amp import autocast
direct/nn/varnet/varnet_engine.py:        Device. Can be "cuda:{idx}" or "cpu".
direct/nn/varnet/varnet_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/varnet/varnet_engine.py:        Device. Can be "cuda:{idx}" or "cpu".
direct/nn/varnet/varnet_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/varnet/varnet_engine.py:        Device. Can be "cuda:{idx}" or "cpu".
direct/nn/varnet/varnet_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/cirim/cirim_engine.py:from torch.cuda.amp import autocast
direct/nn/iterdualnet/iterdualnet_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/unet/unet_engine.py:        Device. Can be "cuda:{idx}" or "cpu".
direct/nn/unet/unet_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/unet/unet_engine.py:        Device. Can be "cuda:{idx}" or "cpu".
direct/nn/unet/unet_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/unet/unet_engine.py:        Device. Can be "cuda:{idx}" or "cpu".
direct/nn/unet/unet_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/nn/rim/rim_engine.py:from torch.cuda.amp import autocast
direct/nn/rim/rim_engine.py:            Device. Can be "cuda:{idx}" or "cpu".
direct/inference.py:    """This function contains most of the logic in DIRECT required to launch a multi-gpu / multi-node inference process.
direct/launch.py:    num_gpus_per_machine: int,
direct/launch.py:    """Launch multi-gpu or distributed training.
direct/launch.py:    child processes (defined by `num_gpus_per_machine`) on each machine.
direct/launch.py:    num_gpus_per_machine: int
direct/launch.py:        The number of GPUs per machine.
direct/launch.py:    world_size = num_machines * num_gpus_per_machine
direct/launch.py:            nprocs=num_gpus_per_machine,
direct/launch.py:                num_gpus_per_machine,
direct/launch.py:    num_gpus_per_machine: int,
direct/launch.py:        World size equal to `num_machines * num_gpus_per_machine`.
direct/launch.py:    num_gpus_per_machine: int
direct/launch.py:        The number of GPUs per machine.
direct/launch.py:    if not torch.cuda.is_available():
direct/launch.py:        raise RuntimeError("CUDA is not available. Please check your installation.")
direct/launch.py:    global_rank = machine_rank * num_gpus_per_machine + local_rank
direct/launch.py:            backend="NCCL",
direct/launch.py:    logger.info("Synchronized GPUs.")
direct/launch.py:    if num_gpus_per_machine > torch.cuda.device_count():
direct/launch.py:    torch.cuda.set_device(local_rank)
direct/launch.py:    num_machines = world_size // num_gpus_per_machine
direct/launch.py:        ranks_on_i = list(range(idx * num_gpus_per_machine, (idx + 1) * num_gpus_per_machine))
direct/launch.py:    num_gpus: int,
direct/launch.py:    """Launch the training, in case there is only one GPU available the function can be called directly.
direct/launch.py:    num_gpus: int
direct/launch.py:        The number of GPUs.
direct/launch.py:    # There is no need for the launch script within one node and at most one GPU.
direct/launch.py:    if num_machines == 1 and num_gpus <= 1:
direct/launch.py:        if torch.cuda.device_count() > 1:
direct/launch.py:                f"Device count is {torch.cuda.device_count()}, "
direct/launch.py:                f"but num_machines is set to {num_machines} and num_gpus is {num_gpus}."
direct/launch.py:    elif torch.cuda.device_count() > 1 and num_gpus <= 1:
direct/launch.py:            f"Device count is {torch.cuda.device_count()}, yet number of GPUs is {num_gpus}. "
direct/launch.py:            f"Unexpected behavior will occur. Consider exposing less GPUs (e.g. through docker). Exiting."
direct/launch.py:            num_gpus,
direct/train.py:    torch.cuda.empty_cache()
direct/train.py:    torch.cuda.empty_cache()
direct/train.py:        args.num_gpus,
direct/environment.py:    logger.info("CUDA %s - cuDNN %s", torch.version.cuda, torch.backends.cudnn.version())
direct/environment.py:            default="cuda",
direct/environment.py:            help='Which device to train on. Set to "cuda" to use the GPU.',
direct/environment.py:        self.add_argument("--num-gpus", type=int, default=1, help="# GPUs per machine.")
direct/environment.py:        # PyTorch still may leave orphan processes in multi-gpu training. Therefore we use a deterministic way
direct/utils/events.py:        if torch.cuda.is_available():
direct/utils/events.py:            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
direct/utils/communication.py:    """Synchronize processes between GPUs.
direct/utils/communication.py:    if torch.distributed.get_backend() == "nccl":
direct/utils/communication.py:    if backend not in ["gloo", "nccl"]:
direct/utils/communication.py:    device = torch.device("cpu" if backend == "gloo" else "cuda")
direct/utils/__init__.py:    torch.cuda.manual_seed(_select_random_seed())
docker/README.rst:To run `DIRECT` using all GPUs:
docker/README.rst:    docker run --gpus all -it \
docker/Dockerfile:ARG CUDA="11.3.0"
docker/Dockerfile:# TODO: conda installs its own version of cuda
docker/Dockerfile:FROM nvidia/cuda:${CUDA}-devel-ubuntu18.04
docker/Dockerfile:ENV CUDA_PATH /usr/local/cuda
docker/Dockerfile:ENV CUDA_ROOT /usr/local/cuda/bin
docker/Dockerfile:ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64
docker/Dockerfile:ENV PATH "/users/direct/miniconda3/bin:/tmp/bart/:$PATH:$CUDA_ROOT"
docker/Dockerfile: && conda install cudatoolkit=${CUDA} torchvision -c pytorch
projects/calgary_campinas/README.rst:            --num-gpus <number_of_gpus> \
projects/calgary_campinas/README.rst:                    --num-gpus <number_of_gpus> \
projects/calgary_campinas/configs/base_lpd.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/calgary_campinas/configs/base_varnet.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/calgary_campinas/configs/base_cirim.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/calgary_campinas/configs/base_cirim.yaml:    validation_steps: 500 # With batch size 4 and 4 GPUs this is about 7300 iterations, or ~1 epoch.
projects/calgary_campinas/configs/base_rim.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/calgary_campinas/configs/base_conjgradnet.yaml:    batch_size: 2  # This is the batch size per GPU!
projects/calgary_campinas/configs/base_recurrentvarnet.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/calgary_campinas/configs/base_jointicnet.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/calgary_campinas/configs/base_xpdnet.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/calgary_campinas/configs/base_unet.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/calgary_campinas/configs/base_kikinet.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/calgary_campinas/configs/base_multidomainnet.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/calgary_campinas/predict_test.py:        args.num_gpus,
projects/CMRxRecon/README.rst:                --num-gpus <number_of_gpus> \
projects/CMRxRecon/README.rst:                --num-gpus <number_of_gpus> \
projects/cvpr2022_recurrentvarnet/README.rst:                --num-gpus <number_of_gpus> \
projects/cvpr2022_recurrentvarnet/README.rst:                --num-gpus <number_of_gpus> \
projects/cvpr2022_recurrentvarnet/README.rst:                --num-gpus <number_of_gpus> \
projects/cvpr2022_recurrentvarnet/README.rst:                --num-gpus <number_of_gpus> \
projects/cvpr2022_recurrentvarnet/README.rst:                --num-gpus <number_of_gpus> \
projects/cvpr2022_recurrentvarnet/README.rst:                --num-gpus <number_of_gpus> \
projects/cvpr2022_recurrentvarnet/README.rst:                --num-gpus <number_of_gpus> \
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs/base_recurrentvarnet.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs/comparisons/base_varnet.yaml:    batch_size: 8  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs/comparisons/base_varnet.yaml:    validation_steps: 500 # With batch size 4 and 4 GPUs this is about 7300 iterations, or ~1 epoch.
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs/comparisons/base_rim.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs/comparisons/base_rim.yaml:    validation_steps: 1000 # With batch size 4 and 4 GPUs this is about 7300 iterations, or ~1 epoch.
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs/comparisons/base_xpdnet.yaml:    batch_size: 3  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs/comparisons/base_unet.yaml:    batch_size: 32  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs/ablation/base_recurrentvarnet_T11.yaml:    batch_size: 2  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs/ablation/base_recurrentvarnet_noRSI.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs/ablation/base_recurrentvarnet_shared.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs/ablation/base_recurrentvarnet_noSER.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/5x/base_recurrentvarnet.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/5x/comparisons/base_varnet.yaml:    batch_size: 8  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/5x/comparisons/base_varnet.yaml:    validation_steps: 500 # With batch size 4 and 4 GPUs this is about 7300 iterations, or ~1 epoch.
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/5x/comparisons/base_rim.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/5x/comparisons/base_rim.yaml:    validation_steps: 1000 # With batch size 4 and 4 GPUs this is about 7300 iterations, or ~1 epoch.
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/5x/comparisons/base_xpdnet.yaml:    batch_size: 3  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/5x/comparisons/base_unet.yaml:    batch_size: 32  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/5x/ablation/base_recurrentvarnet_T11.yaml:    batch_size: 2  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/5x/ablation/base_recurrentvarnet_noRSI.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/5x/ablation/base_recurrentvarnet_shared.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/5x/ablation/base_recurrentvarnet_noSER.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/10x/base_recurrentvarnet.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/10x/comparisons/base_varnet.yaml:    batch_size: 8  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/10x/comparisons/base_varnet.yaml:    validation_steps: 500 # With batch size 4 and 4 GPUs this is about 7300 iterations, or ~1 epoch.
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/10x/comparisons/base_rim.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/10x/comparisons/base_rim.yaml:    validation_steps: 1000 # With batch size 4 and 4 GPUs this is about 7300 iterations, or ~1 epoch.
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/10x/comparisons/base_xpdnet.yaml:    batch_size: 3  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/10x/comparisons/base_unet.yaml:    batch_size: 32  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/10x/ablation/base_recurrentvarnet_T11.yaml:    batch_size: 2  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/10x/ablation/base_recurrentvarnet_noRSI.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/10x/ablation/base_recurrentvarnet_shared.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/calgary_campinas/configs_inference/10x/ablation/base_recurrentvarnet_noSER.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs/base_recurrentvarnet_ablation.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs/base_lpd.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs/base_varnet.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs/base_rim.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs/base_recurrentvarnet.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs/base_unet.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs_inference/4x/base_recurrentvarnet_ablation.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs_inference/4x/base_lpd.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs_inference/4x/base_varnet.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs_inference/4x/base_rim.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs_inference/4x/base_recurrentvarnet.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs_inference/4x/base_unet.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs_inference/8x/base_recurrentvarnet_ablation.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs_inference/8x/base_lpd.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs_inference/8x/base_varnet.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs_inference/8x/base_rim.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs_inference/8x/base_recurrentvarnet.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs_inference/8x/base_unet.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/spie2022_radial_subsampling/README.rst:                --num-gpus <number_of_gpus> \
projects/spie2022_radial_subsampling/README.rst:                    --num-gpus <number_of_gpus> \
projects/spie2022_radial_subsampling/README.rst:                    --num-gpus <number_of_gpus> \
projects/spie2022_radial_subsampling/configs/inference/5x/base_rectilinear.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/spie2022_radial_subsampling/configs/inference/5x/base_rectilinear.yaml:    validation_steps: 500 # With batch size 4 and 4 GPUs this is about 7300 iterations, or ~1 epoch.
projects/spie2022_radial_subsampling/configs/inference/5x/base_radial.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/spie2022_radial_subsampling/configs/inference/5x/base_radial.yaml:    validation_steps: 500 # With batch size 4 and 4 GPUs this is about 7300 iterations, or ~1 epoch.
projects/spie2022_radial_subsampling/configs/inference/10x/base_rectilinear.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/spie2022_radial_subsampling/configs/inference/10x/base_rectilinear.yaml:    validation_steps: 500 # With batch size 4 and 4 GPUs this is about 7300 iterations, or ~1 epoch.
projects/spie2022_radial_subsampling/configs/inference/10x/base_radial.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/spie2022_radial_subsampling/configs/inference/10x/base_radial.yaml:    validation_steps: 500 # With batch size 4 and 4 GPUs this is about 7300 iterations, or ~1 epoch.
projects/spie2022_radial_subsampling/configs/base_rectilinear.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/spie2022_radial_subsampling/configs/base_rectilinear.yaml:    validation_steps: 500 # With batch size 4 and 4 GPUs this is about 7300 iterations, or ~1 epoch.
projects/spie2022_radial_subsampling/configs/base_radial.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/spie2022_radial_subsampling/configs/base_radial.yaml:    validation_steps: 500 # With batch size 4 and 4 GPUs this is about 7300 iterations, or ~1 epoch.
projects/vSHARP/README.rst:                --num-gpus <number_of_gpus> \
projects/vSHARP/README.rst:                --num-gpus <number_of_gpus> \
projects/vSHARP/fastmri_prostate/configs/base_varnet.yaml:    batch_size: 2  # This is the batch size per GPU!
projects/vSHARP/fastmri_prostate/configs/base_vsharp.yaml:    batch_size: 2  # This is the batch size per GPU!
projects/vSHARP/fastmri_prostate/configs/base_recurrentvarnet.yaml:    batch_size: 1  # This is the batch size per GPU!
projects/vSHARP/fastmri_prostate/configs/base_unet.yaml:    batch_size: 2  # This is the batch size per GPU!
projects/JSSL/README.rst:      --num-gpus <number_of_gpus> \
projects/JSSL/README.rst:      --num-gpus <number_of_gpus> --num-workers <number_of_workers> \
projects/predict_val.py:        args.num_gpus,
projects/toy/shepp_logan/README.rst:            --num-gpus <number_of_gpus> \
projects/toy/shepp_logan/base_unet.yaml:    batch_size: 4  # This is the batch size per GPU!
projects/toy/base.yaml:    batch_size: 4  # This is the batch size per GPU!
installation.rst:* CUDA ≥ 10.2 supported GPU.
installation.rst:   If you are using GPUs, cuda is required for the project to run. To install `PyTorch <https://pytorch.org/get-started/locally/>`_ with cuda run:

```
