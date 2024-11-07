# https://github.com/baker-laboratory/RoseTTAFold-All-Atom

```console
rf2aa/model/Track_module.py:    @torch.cuda.amp.autocast(enabled=False)
rf2aa/model/Track_module.py:    @torch.cuda.amp.autocast(enabled=False)
rf2aa/setup_model.py:#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512"
rf2aa/setup_model.py:    torch.cuda.manual_seed(seed)
rf2aa/setup_model.py:    def move_constants_to_device(self, gpu):
rf2aa/setup_model.py:        self.fi_dev = ChemData().frame_indices.to(gpu)
rf2aa/setup_model.py:        self.xyz_converter = XYZConverter().to(gpu)
rf2aa/setup_model.py:        self.l2a = ChemData().long2alt.to(gpu)
rf2aa/setup_model.py:        self.aamask = ChemData().allatom_mask.to(gpu)
rf2aa/setup_model.py:        self.num_bonds = ChemData().num_bonds.to(gpu)
rf2aa/setup_model.py:        self.atom_type_index = ChemData().atom_type_index.to(gpu)
rf2aa/setup_model.py:        self.ljlk_parameters = ChemData().ljlk_parameters.to(gpu)
rf2aa/setup_model.py:        self.lj_correction_parameters = ChemData().lj_correction_parameters.to(gpu)
rf2aa/setup_model.py:        self.hbtypes = ChemData().hbtypes.to(gpu)
rf2aa/setup_model.py:        self.hbbaseatoms = ChemData().hbbaseatoms.to(gpu)
rf2aa/setup_model.py:        self.hbpolys = ChemData().hbpolys.to(gpu)
rf2aa/setup_model.py:        self.cb_len = ChemData().cb_length_t.to(gpu)
rf2aa/setup_model.py:        self.cb_ang = ChemData().cb_angle_t.to(gpu)
rf2aa/setup_model.py:        self.cb_tor = ChemData().cb_torsion_t.to(gpu)
rf2aa/data/data_loader_utils.py:        torch.cuda.manual_seed(0)
rf2aa/data/data_loader_utils.py:        torch.cuda.manual_seed(0)
rf2aa/data/data_loader_utils.py:        torch.cuda.manual_seed(0)
rf2aa/data/data_loader.py:    def to(self, gpu):
rf2aa/data/data_loader.py:                setattr(self, field.name, field_value.to(gpu))
rf2aa/run_inference.py:        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
rf2aa/loss/loss.py:            # note this is stochastic op on GPU
rf2aa/training/recycling.py:        gpu = force_device
rf2aa/training/recycling.py:        gpu = ddp_model.device
rf2aa/training/recycling.py:            stack.enter_context(torch.cuda.amp.autocast(enabled=use_amp))
rf2aa/training/recycling.py:            input_i = add_recycle_inputs(input, output_i, i_cycle, gpu, return_raw=return_raw, use_checkpoint=use_checkpoint)
rf2aa/training/recycling.py:    gpu = device
rf2aa/training/recycling.py:    input_i = add_recycle_inputs(network_input, output_i, 0, gpu, return_raw=False, use_checkpoint=False)
rf2aa/training/recycling.py:    input_i["seq_unmasked"] = input_i["seq_unmasked"].to(gpu)
rf2aa/training/recycling.py:    input_i["sctors"] = input_i["sctors"].to(gpu)
rf2aa/training/recycling.py:def add_recycle_inputs(network_input, output_i, i_cycle, gpu, return_raw=False, use_checkpoint=False):
rf2aa/training/recycling.py:            input_i[key] = network_input[key][:,i_cycle].to(gpu, non_blocking=True)
rf2aa/training/recycling.py:    xyz_prev = ChemData().INIT_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(1,L,1,1).to(gpu, non_blocking=True)
rf2aa/util.py:        torch.cuda.manual_seed(0)
rf2aa/SE3Transformer/setup.py:    author_email='alexandrem@nvidia.com',
rf2aa/SE3Transformer/Dockerfile:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/Dockerfile:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/Dockerfile:# run docker daemon with --default-runtime=nvidia for GPU detection during build
rf2aa/SE3Transformer/Dockerfile:# multistage build for DGL with CUDA and FP16
rf2aa/SE3Transformer/Dockerfile:ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.07-py3
rf2aa/SE3Transformer/Dockerfile:RUN sed -i 's/"35 50 60 70"/"60 70 80"/g' cmake/modules/CUDA.cmake
rf2aa/SE3Transformer/Dockerfile:RUN cmake -DUSE_CUDA=ON -DUSE_FP16=ON ..
rf2aa/SE3Transformer/se3_transformer/model/basis.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/model/basis.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/model/basis.py:from torch.cuda.nvtx import range as nvtx_range
rf2aa/SE3Transformer/se3_transformer/model/transformer.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/model/transformer.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/model/transformer.py:                                 'Only has an effect if AMP is enabled on Volta GPUs, or if running on Ampere GPUs')
rf2aa/SE3Transformer/se3_transformer/model/layers/pooling.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/model/layers/pooling.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/model/layers/convolution.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/model/layers/convolution.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/model/layers/convolution.py:from torch.cuda.nvtx import range as nvtx_range
rf2aa/SE3Transformer/se3_transformer/model/layers/convolution.py:                        with torch.cuda.amp.autocast(False):
rf2aa/SE3Transformer/se3_transformer/model/layers/attention.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/model/layers/attention.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/model/layers/attention.py:from torch.cuda.nvtx import range as nvtx_range
rf2aa/SE3Transformer/se3_transformer/model/layers/attention.py:                with torch.cuda.amp.autocast(False):
rf2aa/SE3Transformer/se3_transformer/model/layers/attention.py:                with torch.cuda.amp.autocast(False):
rf2aa/SE3Transformer/se3_transformer/model/layers/linear.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/model/layers/linear.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/model/layers/norm.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/model/layers/norm.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/model/layers/norm.py:from torch.cuda.nvtx import range as nvtx_range
rf2aa/SE3Transformer/se3_transformer/model/fiber.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/model/fiber.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/runtime/metrics.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/runtime/metrics.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/runtime/metrics.py:        self.add_state('error', torch.tensor(0, dtype=torch.float32, device='cuda'))
rf2aa/SE3Transformer/se3_transformer/runtime/metrics.py:        self.add_state('total', torch.tensor(0, dtype=torch.int32, device='cuda'))
rf2aa/SE3Transformer/se3_transformer/runtime/loggers.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/runtime/loggers.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:def set_socket_affinity(gpu_id):
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    socket connected to the GPU with a given id.
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:        gpu_id: index of a GPU
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    dev = Device(gpu_id)
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:def set_single_affinity(gpu_id):
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    list of all CPU cores from the CPU socket connected to the GPU with a given
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:        gpu_id: index of a GPU
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    dev = Device(gpu_id)
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:def set_single_unique_affinity(gpu_id, nproc_per_node):
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    from the list of all CPU cores from the CPU socket connected to the GPU with
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:        gpu_id: index of a GPU
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    os.sched_setaffinity(0, affinities[gpu_id])
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:def set_socket_unique_affinity(gpu_id, nproc_per_node, mode, balanced=True):
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    cores from the CPU socket connected to a GPU with a given id.
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:        gpu_id: index of a GPU
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    # compute minimal number of physical cores per GPU across all GPUs and
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    # sockets, code assigns this number of cores per GPU if balanced == True
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    min_physical_cores_per_gpu = min(
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:        [len(cores) // len(gpus) for cores, gpus in socket_affinities_to_device_ids.items()]
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:            cores_per_device = min_physical_cores_per_gpu
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:            socket_affinity = socket_affinity[: devices_per_group * min_physical_cores_per_gpu]
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:            if device_id == gpu_id:
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:                # each GPU even if balanced == True (if hyperthreading siblings
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:def set_affinity(gpu_id, nproc_per_node, mode="socket_unique_continuous", balanced=True):
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    is running on a single GPU), which is typical for multi-GPU training
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    from the CPU socket connected to the GPU with a given id.
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    core from the list of all CPU cores from the CPU socket connected to the GPU
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    with a given id (multiple GPUs could be assigned with the same CPU core).
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    connected to the GPU with a given id.
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    GPU with a given id, hyperthreading siblings are included automatically,
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    to a GPU with a given id, hyperthreading siblings are included
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    training workloads on NVIDIA DGX machines.
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:        gpu_id: integer index of a GPU
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    import gpu_affinity
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:        nproc_per_node = torch.cuda.device_count()
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:        affinity = gpu_affinity.set_affinity(args.local_rank, nproc_per_node)
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    python -m torch.distributed.launch --nproc_per_node <#GPUs> example.py
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    WARNING: On DGX A100 only a half of CPU cores have direct access to GPUs.
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:    to GPUs, so on DGX A100 it will limit the code to half of CPU cores and half
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:        set_socket_affinity(gpu_id)
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:        set_single_affinity(gpu_id)
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:        set_single_unique_affinity(gpu_id, nproc_per_node)
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:        set_socket_unique_affinity(gpu_id, nproc_per_node, "interleaved", balanced)
rf2aa/SE3Transformer/se3_transformer/runtime/gpu_affinity.py:        set_socket_unique_affinity(gpu_id, nproc_per_node, "continuous", balanced)
rf2aa/SE3Transformer/se3_transformer/runtime/inference.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/runtime/inference.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/runtime/inference.py:from rf2aa.SE3Transformer.se3_transformer.runtime import gpu_affinity
rf2aa/SE3Transformer/se3_transformer/runtime/inference.py:from rf2aa.SE3Transformer.se3_transformer.runtime.utils import to_cuda, get_local_rank
rf2aa/SE3Transformer/se3_transformer/runtime/inference.py:        *input, target = to_cuda(batch)
rf2aa/SE3Transformer/se3_transformer/runtime/inference.py:        with torch.cuda.amp.autocast(enabled=args.amp):
rf2aa/SE3Transformer/se3_transformer/runtime/inference.py:    major_cc, minor_cc = torch.cuda.get_device_capability()
rf2aa/SE3Transformer/se3_transformer/runtime/inference.py:    model.to(device=torch.cuda.current_device())
rf2aa/SE3Transformer/se3_transformer/runtime/inference.py:        checkpoint = torch.load(str(args.load_ckpt_path), map_location={'cuda:0': f'cuda:{local_rank}'})
rf2aa/SE3Transformer/se3_transformer/runtime/inference.py:        nproc_per_node = torch.cuda.device_count()
rf2aa/SE3Transformer/se3_transformer/runtime/inference.py:        affinity = gpu_affinity.set_affinity(local_rank, nproc_per_node)
rf2aa/SE3Transformer/se3_transformer/runtime/callbacks.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/runtime/callbacks.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/runtime/training.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/runtime/training.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/runtime/training.py:from rf2aa.SE3Transformer.se3_transformer.runtime import gpu_affinity
rf2aa/SE3Transformer/se3_transformer/runtime/training.py:from rf2aa.SE3Transformer.se3_transformer.runtime.utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
rf2aa/SE3Transformer/se3_transformer/runtime/training.py:    checkpoint = torch.load(str(path), map_location={'cuda:0': f'cuda:{get_local_rank()}'})
rf2aa/SE3Transformer/se3_transformer/runtime/training.py:        *inputs, target = to_cuda(batch)
rf2aa/SE3Transformer/se3_transformer/runtime/training.py:        with torch.cuda.amp.autocast(enabled=args.amp):
rf2aa/SE3Transformer/se3_transformer/runtime/training.py:    device = torch.cuda.current_device()
rf2aa/SE3Transformer/se3_transformer/runtime/training.py:    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
rf2aa/SE3Transformer/se3_transformer/runtime/training.py:        gpu_affinity.set_affinity(gpu_id=get_local_rank(), nproc_per_node=torch.cuda.device_count())
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:def to_cuda(x):
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:    """ Try to convert a Tensor, a collection of Tensors or a DGLGraph to CUDA """
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:        return x.cuda(non_blocking=True)
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:        return (to_cuda(v) for v in x)
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:        return [to_cuda(v) for v in x]
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:        return {k: to_cuda(v) for k, v in x.items()}
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:        return x.to(device=torch.cuda.current_device())
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:        if backend == 'nccl':
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:            torch.cuda.set_device(get_local_rank())
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:    _libcudart = ctypes.CDLL('libcudart.so')
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:    # cudaLimitMaxL2FetchGranularity = 0x05
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:    torch.cuda.manual_seed_all(seed)
rf2aa/SE3Transformer/se3_transformer/runtime/utils.py:    major_cc, minor_cc = torch.cuda.get_device_capability()
rf2aa/SE3Transformer/se3_transformer/runtime/arguments.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/runtime/arguments.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/data_loading/data_module.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/data_loading/data_module.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/data_loading/qm9.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/se3_transformer/data_loading/qm9.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/se3_transformer/data_loading/qm9.py:        # Potential improvement: use multi-GPU and gather
rf2aa/SE3Transformer/se3_transformer/data_loading/qm9.py:            # Compute the bases with the GPU but convert the result to CPU to store in RAM
rf2aa/SE3Transformer/se3_transformer/data_loading/qm9.py:            bases.append({k: v.cpu() for k, v in get_basis(rel_pos.cuda(), **self.bases_kwargs).items()})
rf2aa/SE3Transformer/tests/test_equivariance.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/tests/test_equivariance.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/tests/test_equivariance.py:    if torch.cuda.is_available():
rf2aa/SE3Transformer/tests/test_equivariance.py:        feats0 = feats0.cuda()
rf2aa/SE3Transformer/tests/test_equivariance.py:        feats1 = feats1.cuda()
rf2aa/SE3Transformer/tests/test_equivariance.py:        R = R.cuda()
rf2aa/SE3Transformer/tests/test_equivariance.py:        coords = coords.cuda()
rf2aa/SE3Transformer/tests/test_equivariance.py:        graph = graph.to('cuda')
rf2aa/SE3Transformer/tests/test_equivariance.py:        model.cuda()
rf2aa/SE3Transformer/tests/test_equivariance.py:    if torch.cuda.is_available():
rf2aa/SE3Transformer/tests/test_equivariance.py:        R = R.cuda()
rf2aa/SE3Transformer/tests/test_equivariance.py:    if torch.cuda.is_available():
rf2aa/SE3Transformer/tests/test_equivariance.py:        R = R.cuda()
rf2aa/SE3Transformer/tests/test_equivariance.py:    if torch.cuda.is_available():
rf2aa/SE3Transformer/tests/test_equivariance.py:        R = R.cuda()
rf2aa/SE3Transformer/tests/utils.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rf2aa/SE3Transformer/tests/utils.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
rf2aa/SE3Transformer/README.md:This repository provides a script and recipe to train the SE(3)-Transformer model to achieve state-of-the-art accuracy. The content of this repository is tested and maintained by NVIDIA.
rf2aa/SE3Transformer/README.md:            * [Training accuracy: NVIDIA DGX A100 (8x A100 80GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-80gb)  
rf2aa/SE3Transformer/README.md:            * [Training accuracy: NVIDIA DGX-1 (8x V100 16GB)](#training-accuracy-nvidia-dgx-1-8x-v100-16gb)
rf2aa/SE3Transformer/README.md:            * [Training performance: NVIDIA DGX A100 (8x A100 80GB)](#training-performance-nvidia-dgx-a100-8x-a100-80gb) 
rf2aa/SE3Transformer/README.md:            * [Training performance: NVIDIA DGX-1 (8x V100 16GB)](#training-performance-nvidia-dgx-1-8x-v100-16gb)
rf2aa/SE3Transformer/README.md:            * [Inference performance: NVIDIA DGX A100 (1x A100 80GB)](#inference-performance-nvidia-dgx-a100-1x-a100-80gb)
rf2aa/SE3Transformer/README.md:            * [Inference performance: NVIDIA DGX-1 (1x V100 16GB)](#inference-performance-nvidia-dgx-1-1x-v100-16gb)
rf2aa/SE3Transformer/README.md:- Training and inference support for multiple GPUs
rf2aa/SE3Transformer/README.md:This model is trained with mixed precision using Tensor Cores on NVIDIA Volta, NVIDIA Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results up to 1.5x faster than training without Tensor Cores while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.
rf2aa/SE3Transformer/README.md:- Data-parallel multi-GPU training (DDP)
rf2aa/SE3Transformer/README.md:[DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) implements data parallelism at the module level that can run across multiple GPUs or machines.
rf2aa/SE3Transformer/README.md:Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in NVIDIA Volta, and following with both the NVIDIA Turing and NVIDIA Ampere Architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using [mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) previously required two steps:
rf2aa/SE3Transformer/README.md:AMP enables mixed precision training on NVIDIA Volta, NVIDIA Turing, and NVIDIA Ampere GPU architectures automatically. The PyTorch framework code makes all necessary model changes internally.
rf2aa/SE3Transformer/README.md:-   How to train using mixed precision, refer to the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
rf2aa/SE3Transformer/README.md:-   Techniques used for mixed precision training, refer to the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
rf2aa/SE3Transformer/README.md:-   APEX tools for mixed precision training, refer to the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).
rf2aa/SE3Transformer/README.md:Mixed precision is enabled in PyTorch by using the native [Automatic Mixed Precision package](https://pytorch.org/docs/stable/amp.html), which casts variables to half-precision upon retrieval while storing variables in single-precision format. Furthermore, to preserve small gradient magnitudes in backpropagation, a [loss scaling](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#lossscaling) step must be included when applying gradients. In PyTorch, loss scaling can be applied automatically using a `GradScaler`.
rf2aa/SE3Transformer/README.md:TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math, also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on NVIDIA Volta GPUs. 
rf2aa/SE3Transformer/README.md:For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.
rf2aa/SE3Transformer/README.md:TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.
rf2aa/SE3Transformer/README.md:- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
rf2aa/SE3Transformer/README.md:- Supported GPUs:
rf2aa/SE3Transformer/README.md:    - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
rf2aa/SE3Transformer/README.md:    - [NVIDIA Turing architecture](https://www.nvidia.com/en-us/design-visualization/technologies/turing-architecture/)
rf2aa/SE3Transformer/README.md:    - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)
rf2aa/SE3Transformer/README.md:For more information about how to get started with NGC containers, refer to the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
rf2aa/SE3Transformer/README.md:- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
rf2aa/SE3Transformer/README.md:- [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
rf2aa/SE3Transformer/README.md:- [Running PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html#running)
rf2aa/SE3Transformer/README.md:For those unable to use the PyTorch NGC container to set up the required environment or create your own container, refer to the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).
rf2aa/SE3Transformer/README.md:    git clone https://github.com/NVIDIA/DeepLearningExamples
rf2aa/SE3Transformer/README.md:    docker run -it --runtime=nvidia --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 --rm -v ${PWD}/results:/results se3-transformer:latest
rf2aa/SE3Transformer/README.md:- `se3_transformer/runtime/metrics.py`: MAE metric with support for multi-GPU synchronization
rf2aa/SE3Transformer/README.md:- `se3_transformer/runtime/loggers.py`: [DLLogger](https://github.com/NVIDIA/dllogger) and [W&B](wandb.ai/) loggers
rf2aa/SE3Transformer/README.md:- `--epochs`: Number of training epochs (default: `100` for single-GPU)
rf2aa/SE3Transformer/README.md:- `--learning_rate`: Learning rate to use (default: `0.002` for single-GPU)
rf2aa/SE3Transformer/README.md:- `--low_memory`: If true, will use fused ops that are slower but use less memory (expect 25 percent less memory). Only has an effect if AMP is enabled on NVIDIA Volta GPUs or if running on Ampere GPUs (default: `false`)
rf2aa/SE3Transformer/README.md:**Multi-GPU and multi-node**
rf2aa/SE3Transformer/README.md:The training script supports the PyTorch elastic launcher to run on multiple GPUs or nodes.  Refer to the [official documentation](https://pytorch.org/docs/1.9.0/elastic/run.html).
rf2aa/SE3Transformer/README.md:For example, to train on all available GPUs with AMP:
rf2aa/SE3Transformer/README.md:python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --module se3_transformer.runtime.training --amp
rf2aa/SE3Transformer/README.md:The performance measurements in this document were conducted at the time of publication and may not reflect the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to [NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).
rf2aa/SE3Transformer/README.md:To benchmark the training performance on a specific batch size, run `bash scripts/benchmarck_train.sh {BATCH_SIZE}` for single GPU, and `bash scripts/benchmarck_train_multi_gpu.sh {BATCH_SIZE}` for multi-GPU.
rf2aa/SE3Transformer/README.md:##### Training accuracy: NVIDIA DGX A100 (8x A100 80GB)
rf2aa/SE3Transformer/README.md:Our results were obtained by running the `scripts/train.sh` training script in the PyTorch 21.07 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs.
rf2aa/SE3Transformer/README.md:| GPUs    | Batch size / GPU    | Absolute error - TF32  | Absolute error - mixed precision  |   Time to train - TF32  |  Time to train - mixed precision | Time to train speedup (mixed precision to TF32) |       
rf2aa/SE3Transformer/README.md:##### Training accuracy: NVIDIA DGX-1 (8x V100 16GB)
rf2aa/SE3Transformer/README.md:Our results were obtained by running the `scripts/train.sh` training script in the PyTorch 21.07 NGC container on NVIDIA DGX-1 with (8x V100 16GB) GPUs.
rf2aa/SE3Transformer/README.md:| GPUs    | Batch size / GPU    | Absolute error - FP32  | Absolute error - mixed precision  |   Time to train - FP32  |  Time to train - mixed precision | Time to train speedup (mixed precision to FP32)  |      
rf2aa/SE3Transformer/README.md:##### Training performance: NVIDIA DGX A100 (8x A100 80GB)
rf2aa/SE3Transformer/README.md:Our results were obtained by running the `scripts/benchmark_train.sh` and `scripts/benchmark_train_multi_gpu.sh` benchmarking scripts in the PyTorch 21.07 NGC container on NVIDIA DGX A100 with 8x A100 80GB GPUs. Performance numbers (in molecules per millisecond) were averaged over five  entire training epochs after a warmup epoch.
rf2aa/SE3Transformer/README.md:| GPUs             | Batch size / GPU     | Throughput - TF32 [mol/ms]                             | Throughput - mixed precision [mol/ms]      | Throughput speedup (mixed precision - TF32)   | Weak scaling - TF32    | Weak scaling - mixed precision |
rf2aa/SE3Transformer/README.md:##### Training performance: NVIDIA DGX-1 (8x V100 16GB)
rf2aa/SE3Transformer/README.md:Our results were obtained by running the `scripts/benchmark_train.sh` and `scripts/benchmark_train_multi_gpu.sh` benchmarking scripts in the PyTorch 21.07 NGC container on NVIDIA DGX-1 with 8x V100 16GB GPUs. Performance numbers (in molecules per millisecond) were averaged over five  entire training epochs after a warmup epoch.
rf2aa/SE3Transformer/README.md:| GPUs             | Batch size / GPU     | Throughput - FP32 [mol/ms] | Throughput - mixed precision  [mol/ms]     | Throughput speedup (FP32 - mixed precision)   | Weak scaling - FP32    | Weak scaling - mixed precision |
rf2aa/SE3Transformer/README.md:##### Inference performance: NVIDIA DGX A100 (1x A100 80GB)
rf2aa/SE3Transformer/README.md:Our results were obtained by running the `scripts/benchmark_inference.sh` inferencing benchmarking script in the PyTorch 21.07 NGC container on NVIDIA DGX A100 with 1x A100 80GB GPU.
rf2aa/SE3Transformer/README.md:##### Inference performance: NVIDIA DGX-1 (1x V100 16GB)
rf2aa/SE3Transformer/README.md:Our results were obtained by running the `scripts/benchmark_inference.sh` inferencing benchmarking script in the PyTorch 21.07 NGC container on NVIDIA DGX-1 with 1x V100 16GB GPU.
rf2aa/SE3Transformer/scripts/predict.sh:python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
rf2aa/SE3Transformer/scripts/train_multi_gpu.sh:python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
rf2aa/SE3Transformer/scripts/benchmark_train_multi_gpu.sh:# Script to benchmark multi-GPU training performance, with bases precomputation
rf2aa/SE3Transformer/scripts/benchmark_train_multi_gpu.sh:python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
rf2aa/SE3Transformer/scripts/benchmark_train.sh:# Script to benchmark single-GPU training performance, with bases precomputation
rf2aa/SE3Transformer/scripts/benchmark_train.sh:CUDA_VISIBLE_DEVICES=0 python -m se3_transformer.runtime.training \
rf2aa/SE3Transformer/scripts/benchmark_inference.sh:CUDA_VISIBLE_DEVICES=0 python -m se3_transformer.runtime.inference \
rf2aa/SE3Transformer/requirements.txt:git+https://github.com/NVIDIA/dllogger#egg=dllogger
rf2aa/SE3Transformer/LICENSE:Copyright 2021 NVIDIA CORPORATION & AFFILIATES
environment.yaml:  - nvidia
environment.yaml:  - cuda-cudart=11.8.89=0
environment.yaml:  - cuda-cupti=11.8.87=0
environment.yaml:  - cuda-libraries=11.8.0=0
environment.yaml:  - cuda-nvrtc=11.8.89=0
environment.yaml:  - cuda-nvtx=11.8.86=0
environment.yaml:  - cuda-runtime=11.8.0=0
environment.yaml:  - cuda-version=11.8=h70ddcb2_3
environment.yaml:  - cudatoolkit=11.8.0=h4ba93d1_13
environment.yaml:  - dgl=1.1.2=cuda112py310hc641c19_2
environment.yaml:  - nccl=2.20.5.1=h6103f9b_0
environment.yaml:  - pytorch=2.0.1=py3.10_cuda11.8_cudnn8.7.0_0
environment.yaml:  - pytorch-cuda=11.8=h7e8668a_5
environment.yaml:  - pytorch-mutex=1.0=cuda
environment.yaml:  - tensorflow=2.11.0=cuda112py310he87a039_0
environment.yaml:  - tensorflow-base=2.11.0=cuda112py310h52da4a5_0
environment.yaml:  - tensorflow-estimator=2.11.0=cuda112py310h37add04_0
environment.yaml:      - git+https://github.com/NVIDIA/dllogger.git@0540a43971f4a8a16693a9de9de73c1072020769

```
