# https://github.com/RosettaCommons/RFDesign

```console
envs/SE3-nvidia.yml:name: SE3-nvidia
envs/SE3-nvidia.yml:  - nvidia
envs/SE3-nvidia.yml:  - cudatoolkit=11.1.74=h6bb024c_0
envs/SE3-nvidia.yml:  - dgl-cuda11.1=0.8.2=py39_0
envs/SE3-nvidia.yml:  - pytorch=1.9.0=py3.9_cuda11.1_cudnn8.0.5_0
inpainting/model/InitStrGenerator.py:    #@torch.cuda.amp.autocast(enabled=True)
inpainting/model/Track_module.py:    @torch.cuda.amp.autocast(enabled=False)
inpainting/model/loss.py:@torch.cuda.amp.autocast(enabled=False)
inpainting/model/util.py:                if obj.is_cuda:
inpainting/model/util.py:    msg += torch.cuda.memory_summary() + '\n'
inpainting/model/ss_features.py:    has_gpu = torch.cuda.is_available()
inpainting/model/ss_features.py:    if not has_gpu:
inpainting/model/data_loader.py:        # per each gpu
inpainting/rfjoint_mutation_effect_prediction.py:        if torch.cuda.is_available() and (not use_cpu):
inpainting/rfjoint_mutation_effect_prediction.py:            self.device = torch.device("cuda:0")
inpainting/rfjoint_mutation_effect_prediction.py:    parser.add_argument("--cpu", dest='use_cpu', default=False, action='store_true', help="Force to use CPU instead of GPU [False]")
inpainting/inpaint.py:DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
se3_transformer/model/basis.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/model/basis.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/model/basis.py:from torch.cuda.nvtx import range as nvtx_range
se3_transformer/model/transformer.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/model/transformer.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/model/transformer.py:                                 'Only has an effect if AMP is enabled on Volta GPUs, or if running on Ampere GPUs')
se3_transformer/model/layers/pooling.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/model/layers/pooling.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/model/layers/convolution.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/model/layers/convolution.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/model/layers/convolution.py:from torch.cuda.nvtx import range as nvtx_range
se3_transformer/model/layers/attention.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/model/layers/attention.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/model/layers/attention.py:from torch.cuda.nvtx import range as nvtx_range
se3_transformer/model/layers/linear.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/model/layers/linear.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/model/layers/norm.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/model/layers/norm.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/model/layers/norm.py:from torch.cuda.nvtx import range as nvtx_range
se3_transformer/model/fiber.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/model/fiber.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/runtime/metrics.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/runtime/metrics.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/runtime/metrics.py:        self.add_state('error', torch.tensor(0, dtype=torch.float32, device='cuda'))
se3_transformer/runtime/metrics.py:        self.add_state('total', torch.tensor(0, dtype=torch.int32, device='cuda'))
se3_transformer/runtime/loggers.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/runtime/loggers.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/runtime/gpu_affinity.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/runtime/gpu_affinity.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/runtime/gpu_affinity.py:def set_socket_affinity(gpu_id):
se3_transformer/runtime/gpu_affinity.py:    socket connected to the GPU with a given id.
se3_transformer/runtime/gpu_affinity.py:        gpu_id: index of a GPU
se3_transformer/runtime/gpu_affinity.py:    dev = Device(gpu_id)
se3_transformer/runtime/gpu_affinity.py:def set_single_affinity(gpu_id):
se3_transformer/runtime/gpu_affinity.py:    list of all CPU cores from the CPU socket connected to the GPU with a given
se3_transformer/runtime/gpu_affinity.py:        gpu_id: index of a GPU
se3_transformer/runtime/gpu_affinity.py:    dev = Device(gpu_id)
se3_transformer/runtime/gpu_affinity.py:def set_single_unique_affinity(gpu_id, nproc_per_node):
se3_transformer/runtime/gpu_affinity.py:    from the list of all CPU cores from the CPU socket connected to the GPU with
se3_transformer/runtime/gpu_affinity.py:        gpu_id: index of a GPU
se3_transformer/runtime/gpu_affinity.py:    os.sched_setaffinity(0, affinities[gpu_id])
se3_transformer/runtime/gpu_affinity.py:def set_socket_unique_affinity(gpu_id, nproc_per_node, mode, balanced=True):
se3_transformer/runtime/gpu_affinity.py:    cores from the CPU socket connected to a GPU with a given id.
se3_transformer/runtime/gpu_affinity.py:        gpu_id: index of a GPU
se3_transformer/runtime/gpu_affinity.py:    # compute minimal number of physical cores per GPU across all GPUs and
se3_transformer/runtime/gpu_affinity.py:    # sockets, code assigns this number of cores per GPU if balanced == True
se3_transformer/runtime/gpu_affinity.py:    min_physical_cores_per_gpu = min(
se3_transformer/runtime/gpu_affinity.py:        [len(cores) // len(gpus) for cores, gpus in socket_affinities_to_device_ids.items()]
se3_transformer/runtime/gpu_affinity.py:            cores_per_device = min_physical_cores_per_gpu
se3_transformer/runtime/gpu_affinity.py:            socket_affinity = socket_affinity[: devices_per_group * min_physical_cores_per_gpu]
se3_transformer/runtime/gpu_affinity.py:            if device_id == gpu_id:
se3_transformer/runtime/gpu_affinity.py:                # each GPU even if balanced == True (if hyperthreading siblings
se3_transformer/runtime/gpu_affinity.py:def set_affinity(gpu_id, nproc_per_node, mode="socket_unique_continuous", balanced=True):
se3_transformer/runtime/gpu_affinity.py:    is running on a single GPU), which is typical for multi-GPU training
se3_transformer/runtime/gpu_affinity.py:    from the CPU socket connected to the GPU with a given id.
se3_transformer/runtime/gpu_affinity.py:    core from the list of all CPU cores from the CPU socket connected to the GPU
se3_transformer/runtime/gpu_affinity.py:    with a given id (multiple GPUs could be assigned with the same CPU core).
se3_transformer/runtime/gpu_affinity.py:    connected to the GPU with a given id.
se3_transformer/runtime/gpu_affinity.py:    GPU with a given id, hyperthreading siblings are included automatically,
se3_transformer/runtime/gpu_affinity.py:    to a GPU with a given id, hyperthreading siblings are included
se3_transformer/runtime/gpu_affinity.py:    training workloads on NVIDIA DGX machines.
se3_transformer/runtime/gpu_affinity.py:        gpu_id: integer index of a GPU
se3_transformer/runtime/gpu_affinity.py:    import gpu_affinity
se3_transformer/runtime/gpu_affinity.py:        nproc_per_node = torch.cuda.device_count()
se3_transformer/runtime/gpu_affinity.py:        affinity = gpu_affinity.set_affinity(args.local_rank, nproc_per_node)
se3_transformer/runtime/gpu_affinity.py:    python -m torch.distributed.launch --nproc_per_node <#GPUs> example.py
se3_transformer/runtime/gpu_affinity.py:    WARNING: On DGX A100 only a half of CPU cores have direct access to GPUs.
se3_transformer/runtime/gpu_affinity.py:    to GPUs, so on DGX A100 it will limit the code to half of CPU cores and half
se3_transformer/runtime/gpu_affinity.py:        set_socket_affinity(gpu_id)
se3_transformer/runtime/gpu_affinity.py:        set_single_affinity(gpu_id)
se3_transformer/runtime/gpu_affinity.py:        set_single_unique_affinity(gpu_id, nproc_per_node)
se3_transformer/runtime/gpu_affinity.py:        set_socket_unique_affinity(gpu_id, nproc_per_node, "interleaved", balanced)
se3_transformer/runtime/gpu_affinity.py:        set_socket_unique_affinity(gpu_id, nproc_per_node, "continuous", balanced)
se3_transformer/runtime/inference.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/runtime/inference.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/runtime/inference.py:from se3_transformer.runtime import gpu_affinity
se3_transformer/runtime/inference.py:from se3_transformer.runtime.utils import to_cuda, get_local_rank
se3_transformer/runtime/inference.py:        *input, target = to_cuda(batch)
se3_transformer/runtime/inference.py:        with torch.cuda.amp.autocast(enabled=args.amp):
se3_transformer/runtime/inference.py:    major_cc, minor_cc = torch.cuda.get_device_capability()
se3_transformer/runtime/inference.py:    model.to(device=torch.cuda.current_device())
se3_transformer/runtime/inference.py:        checkpoint = torch.load(str(args.load_ckpt_path), map_location={'cuda:0': f'cuda:{local_rank}'})
se3_transformer/runtime/inference.py:        nproc_per_node = torch.cuda.device_count()
se3_transformer/runtime/inference.py:        affinity = gpu_affinity.set_affinity(local_rank, nproc_per_node)
se3_transformer/runtime/callbacks.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/runtime/callbacks.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/runtime/training.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/runtime/training.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/runtime/training.py:from se3_transformer.runtime import gpu_affinity
se3_transformer/runtime/training.py:from se3_transformer.runtime.utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
se3_transformer/runtime/training.py:    checkpoint = torch.load(str(path), map_location={'cuda:0': f'cuda:{get_local_rank()}'})
se3_transformer/runtime/training.py:        *inputs, target = to_cuda(batch)
se3_transformer/runtime/training.py:        with torch.cuda.amp.autocast(enabled=args.amp):
se3_transformer/runtime/training.py:    device = torch.cuda.current_device()
se3_transformer/runtime/training.py:    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
se3_transformer/runtime/training.py:        gpu_affinity.set_affinity(gpu_id=get_local_rank(), nproc_per_node=torch.cuda.device_count())
se3_transformer/runtime/utils.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/runtime/utils.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/runtime/utils.py:def to_cuda(x):
se3_transformer/runtime/utils.py:    """ Try to convert a Tensor, a collection of Tensors or a DGLGraph to CUDA """
se3_transformer/runtime/utils.py:        return x.cuda(non_blocking=True)
se3_transformer/runtime/utils.py:        return (to_cuda(v) for v in x)
se3_transformer/runtime/utils.py:        return [to_cuda(v) for v in x]
se3_transformer/runtime/utils.py:        return {k: to_cuda(v) for k, v in x.items()}
se3_transformer/runtime/utils.py:        return x.to(device=torch.cuda.current_device())
se3_transformer/runtime/utils.py:        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
se3_transformer/runtime/utils.py:        if backend == 'nccl':
se3_transformer/runtime/utils.py:            torch.cuda.set_device(get_local_rank())
se3_transformer/runtime/utils.py:    _libcudart = ctypes.CDLL('libcudart.so')
se3_transformer/runtime/utils.py:    # cudaLimitMaxL2FetchGranularity = 0x05
se3_transformer/runtime/utils.py:    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
se3_transformer/runtime/utils.py:    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
se3_transformer/runtime/utils.py:    torch.cuda.manual_seed_all(seed)
se3_transformer/runtime/utils.py:    major_cc, minor_cc = torch.cuda.get_device_capability()
se3_transformer/runtime/arguments.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/runtime/arguments.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/data_loading/data_module.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/data_loading/data_module.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/data_loading/qm9.py:# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
se3_transformer/data_loading/qm9.py:# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
se3_transformer/data_loading/qm9.py:        # Potential improvement: use multi-GPU and gather
se3_transformer/data_loading/qm9.py:            # Compute the bases with the GPU but convert the result to CPU to store in RAM
se3_transformer/data_loading/qm9.py:            bases.append({k: v.cpu() for k, v in get_basis(rel_pos.cuda(), **self.bases_kwargs).items()})
se3_transformer/LICENSE:Copyright 2021 NVIDIA CORPORATION & AFFILIATES
tutorials/halluc_SH3_binder/array_submit.sh:sbatch -a 1-$(cat $1 | wc -l) -p gpu -J $jobname \
tutorials/halluc_SH3_binder/array_submit.sh:       -c 2 --mem=12g --gres=gpu:rtx2080:1 \
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_87_r2 --mask 46-46,B7-14,31-31 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_87.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=0 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_87_r2_0.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_110_r2 --mask 28-28,B7-14,59-59 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_110.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=0 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_110_r2_0.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_146_r2 --mask 33-33,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_146.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=0 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_146_r2_0.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_22_r2 --mask 60-60,B7-14,28-28 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_22.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=0 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_22_r2_0.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_33_r2 --mask 65-65,B7-14,19-19 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_33.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=0 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_33_r2_0.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_192_r2 --mask 15-15,B7-14,41-41 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_192.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=0 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_192_r2_0.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_112_r2 --mask 44-44,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_112.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=0 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_112_r2_0.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_66_r2 --mask 47-47,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_66.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=0 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_66_r2_0.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_120_r2 --mask 25-25,B7-14,63-63 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_120.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=0 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_120_r2_0.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_85_r2 --mask 49-49,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_85.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=0 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_85_r2_0.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_78_r2 --mask 44-44,B7-14,43-43 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_78.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=0 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_78_r2_0.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_43_r2 --mask 9-9,B7-14,49-49 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_43.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=0 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_43_r2_0.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_69_r2 --mask 18-18,B7-14,44-44 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_69.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=0 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_69_r2_0.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_87_r2 --mask 46-46,B7-14,31-31 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_87.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=5 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_87_r2_5.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_110_r2 --mask 28-28,B7-14,59-59 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_110.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=5 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_110_r2_5.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_146_r2 --mask 33-33,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_146.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=5 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_146_r2_5.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_22_r2 --mask 60-60,B7-14,28-28 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_22.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=5 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_22_r2_5.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_33_r2 --mask 65-65,B7-14,19-19 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_33.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=5 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_33_r2_5.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_192_r2 --mask 15-15,B7-14,41-41 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_192.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=5 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_192_r2_5.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_112_r2 --mask 44-44,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_112.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=5 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_112_r2_5.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_66_r2 --mask 47-47,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_66.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=5 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_66_r2_5.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_120_r2 --mask 25-25,B7-14,63-63 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_120.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=5 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_120_r2_5.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_85_r2 --mask 49-49,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_85.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=5 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_85_r2_5.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_78_r2 --mask 44-44,B7-14,43-43 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_78.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=5 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_78_r2_5.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_43_r2 --mask 9-9,B7-14,49-49 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_43.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=5 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_43_r2_5.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_69_r2 --mask 18-18,B7-14,44-44 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_69.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=5 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_69_r2_5.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_87_r2 --mask 46-46,B7-14,31-31 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_87.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=10 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_87_r2_10.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_110_r2 --mask 28-28,B7-14,59-59 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_110.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=10 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_110_r2_10.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_146_r2 --mask 33-33,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_146.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=10 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_146_r2_10.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_22_r2 --mask 60-60,B7-14,28-28 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_22.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=10 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_22_r2_10.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_33_r2 --mask 65-65,B7-14,19-19 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_33.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=10 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_33_r2_10.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_192_r2 --mask 15-15,B7-14,41-41 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_192.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=10 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_192_r2_10.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_112_r2 --mask 44-44,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_112.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=10 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_112_r2_10.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_66_r2 --mask 47-47,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_66.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=10 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_66_r2_10.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_120_r2 --mask 25-25,B7-14,63-63 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_120.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=10 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_120_r2_10.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_85_r2 --mask 49-49,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_85.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=10 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_85_r2_10.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_78_r2 --mask 44-44,B7-14,43-43 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_78.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=10 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_78_r2_10.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_43_r2 --mask 9-9,B7-14,49-49 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_43.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=10 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_43_r2_10.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_69_r2 --mask 18-18,B7-14,44-44 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_69.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=10 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_69_r2_10.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_87_r2 --mask 46-46,B7-14,31-31 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_87.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=15 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_87_r2_15.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_110_r2 --mask 28-28,B7-14,59-59 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_110.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=15 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_110_r2_15.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_146_r2 --mask 33-33,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_146.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=15 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_146_r2_15.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_22_r2 --mask 60-60,B7-14,28-28 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_22.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=15 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_22_r2_15.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_33_r2 --mask 65-65,B7-14,19-19 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_33.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=15 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_33_r2_15.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_192_r2 --mask 15-15,B7-14,41-41 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_192.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=15 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_192_r2_15.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_112_r2 --mask 44-44,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_112.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=15 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_112_r2_15.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_66_r2 --mask 47-47,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_66.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=15 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_66_r2_15.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_120_r2 --mask 25-25,B7-14,63-63 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_120.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=15 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_120_r2_15.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_85_r2 --mask 49-49,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_85.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=15 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_85_r2_15.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_78_r2 --mask 44-44,B7-14,43-43 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_78.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=15 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_78_r2_15.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_43_r2 --mask 9-9,B7-14,49-49 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_43.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=15 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_43_r2_15.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_69_r2 --mask 18-18,B7-14,44-44 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_69.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=15 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_69_r2_15.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_87_r2 --mask 46-46,B7-14,31-31 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_87.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=20 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_87_r2_20.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_110_r2 --mask 28-28,B7-14,59-59 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_110.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=20 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_110_r2_20.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_146_r2 --mask 33-33,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_146.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=20 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_146_r2_20.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_22_r2 --mask 60-60,B7-14,28-28 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_22.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=20 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_22_r2_20.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_33_r2 --mask 65-65,B7-14,19-19 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_33.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=20 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_33_r2_20.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_192_r2 --mask 15-15,B7-14,41-41 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_192.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=20 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_192_r2_20.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_112_r2 --mask 44-44,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_112.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=20 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_112_r2_20.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_66_r2 --mask 47-47,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_66.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=20 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_66_r2_20.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_120_r2 --mask 25-25,B7-14,63-63 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_120.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=20 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_120_r2_20.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_85_r2 --mask 49-49,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_85.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=20 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_85_r2_20.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_78_r2 --mask 44-44,B7-14,43-43 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_78.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=20 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_78_r2_20.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_43_r2 --mask 9-9,B7-14,49-49 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_43.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=20 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_43_r2_20.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_69_r2 --mask 18-18,B7-14,44-44 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_69.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=20 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_69_r2_20.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_87_r2 --mask 46-46,B7-14,31-31 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_87.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=25 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_87_r2_25.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_110_r2 --mask 28-28,B7-14,59-59 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_110.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=25 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_110_r2_25.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_146_r2 --mask 33-33,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_146.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=25 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_146_r2_25.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_22_r2 --mask 60-60,B7-14,28-28 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_22.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=25 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_22_r2_25.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_33_r2 --mask 65-65,B7-14,19-19 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_33.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=25 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_33_r2_25.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_192_r2 --mask 15-15,B7-14,41-41 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_192.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=25 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_192_r2_25.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_112_r2 --mask 44-44,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_112.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=25 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_112_r2_25.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_66_r2 --mask 47-47,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_66.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=25 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_66_r2_25.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_120_r2 --mask 25-25,B7-14,63-63 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_120.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=25 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_120_r2_25.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_85_r2 --mask 49-49,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_85.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=25 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_85_r2_25.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_78_r2 --mask 44-44,B7-14,43-43 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_78.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=25 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_78_r2_25.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_43_r2 --mask 9-9,B7-14,49-49 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_43.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=25 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_43_r2_25.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_69_r2 --mask 18-18,B7-14,44-44 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_69.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=25 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_69_r2_25.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_87_r2 --mask 46-46,B7-14,31-31 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_87.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=30 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_87_r2_30.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_110_r2 --mask 28-28,B7-14,59-59 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_110.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=30 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_110_r2_30.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_146_r2 --mask 33-33,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_146.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=30 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_146_r2_30.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_22_r2 --mask 60-60,B7-14,28-28 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_22.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=30 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_22_r2_30.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_33_r2 --mask 65-65,B7-14,19-19 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_33.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=30 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_33_r2_30.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_192_r2 --mask 15-15,B7-14,41-41 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_192.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=30 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_192_r2_30.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_112_r2 --mask 44-44,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_112.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=30 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_112_r2_30.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_66_r2 --mask 47-47,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_66.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=30 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_66_r2_30.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_120_r2 --mask 25-25,B7-14,63-63 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_120.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=30 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_120_r2_30.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_85_r2 --mask 49-49,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_85.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=30 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_85_r2_30.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_78_r2 --mask 44-44,B7-14,43-43 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_78.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=30 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_78_r2_30.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_43_r2 --mask 9-9,B7-14,49-49 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_43.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=30 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_43_r2_30.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_69_r2 --mask 18-18,B7-14,44-44 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_69.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=30 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_69_r2_30.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_87_r2 --mask 46-46,B7-14,31-31 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_87.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=35 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_87_r2_35.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_110_r2 --mask 28-28,B7-14,59-59 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_110.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=35 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_110_r2_35.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_146_r2 --mask 33-33,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_146.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=35 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_146_r2_35.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_22_r2 --mask 60-60,B7-14,28-28 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_22.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=35 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_22_r2_35.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_33_r2 --mask 65-65,B7-14,19-19 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_33.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=35 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_33_r2_35.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_192_r2 --mask 15-15,B7-14,41-41 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_192.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=35 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_192_r2_35.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_112_r2 --mask 44-44,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_112.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=35 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_112_r2_35.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_66_r2 --mask 47-47,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_66.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=35 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_66_r2_35.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_120_r2 --mask 25-25,B7-14,63-63 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_120.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=35 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_120_r2_35.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_85_r2 --mask 49-49,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_85.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=35 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_85_r2_35.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_78_r2 --mask 44-44,B7-14,43-43 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_78.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=35 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_78_r2_35.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_43_r2 --mask 9-9,B7-14,49-49 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_43.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=35 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_43_r2_35.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_69_r2 --mask 18-18,B7-14,44-44 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_69.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=35 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_69_r2_35.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_87_r2 --mask 46-46,B7-14,31-31 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_87.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=40 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_87_r2_40.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_110_r2 --mask 28-28,B7-14,59-59 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_110.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=40 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_110_r2_40.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_146_r2 --mask 33-33,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_146.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=40 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_146_r2_40.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_22_r2 --mask 60-60,B7-14,28-28 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_22.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=40 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_22_r2_40.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_33_r2 --mask 65-65,B7-14,19-19 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_33.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=40 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_33_r2_40.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_192_r2 --mask 15-15,B7-14,41-41 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_192.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=40 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_192_r2_40.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_112_r2 --mask 44-44,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_112.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=40 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_112_r2_40.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_66_r2 --mask 47-47,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_66.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=40 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_66_r2_40.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_120_r2 --mask 25-25,B7-14,63-63 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_120.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=40 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_120_r2_40.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_85_r2 --mask 49-49,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_85.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=40 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_85_r2_40.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_78_r2 --mask 44-44,B7-14,43-43 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_78.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=40 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_78_r2_40.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_43_r2 --mask 9-9,B7-14,49-49 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_43.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=40 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_43_r2_40.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_69_r2 --mask 18-18,B7-14,44-44 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_69.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=40 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_69_r2_40.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_87_r2 --mask 46-46,B7-14,31-31 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_87.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=45 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_87_r2_45.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_110_r2 --mask 28-28,B7-14,59-59 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_110.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=45 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_110_r2_45.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_146_r2 --mask 33-33,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_146.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=45 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_146_r2_45.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_22_r2 --mask 60-60,B7-14,28-28 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_22.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=45 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_22_r2_45.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_33_r2 --mask 65-65,B7-14,19-19 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_33.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=45 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_33_r2_45.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_192_r2 --mask 15-15,B7-14,41-41 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_192.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=45 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_192_r2_45.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_112_r2 --mask 44-44,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_112.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=45 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_112_r2_45.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_66_r2 --mask 47-47,B7-14,23-23 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_66.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=45 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_66_r2_45.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_120_r2 --mask 25-25,B7-14,63-63 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_120.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=45 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_120_r2_45.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_85_r2 --mask 49-49,B7-14,18-18 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_85.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=45 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_85_r2_45.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_78_r2 --mask 44-44,B7-14,43-43 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_78.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=45 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_78_r2_45.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_43_r2 --mask 9-9,B7-14,49-49 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_43.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=45 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_43_r2_45.log
tutorials/halluc_SH3_binder/r2.list:source activate SE3-nvidia; python ../../hallucination/hallucinate.py --network_name rf_Nov05_2021 --pdb=input/SH3_2w0z.pdb --out=output/20220104_sh3_r2//sh3_r1_69_r2 --mask 18-18,B7-14,44-44 --use_template B7-14 --spike_fas output/hits_sh3_r1/sh3_r1_69.fas --spike 0.999 --force_aa B7-14 --exclude_aa C --receptor input/SH3_2w0z_rec.pdb --rec_placement second --w_surfnp 1 --w_nc 0.02 --steps=m300 --num=5 --start_num=45 --w_rog=1 --rog_thresh=16 --save_pdb=True --track_step=1 --cautious=True &>> output/20220104_sh3_r2//sh3_r1_69_r2_45.log
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_196.log:[00:38:24] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_196.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_196.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_196.log:Max CUDA memory: 0.7586G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_196.log:Max CUDA memory: 0.7608G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_88.log:[21:56:23] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_88.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_88.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_88.log:Max CUDA memory: 0.9298G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_88.log:Max CUDA memory: 1.4603G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_170.log:[23:54:45] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_170.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_170.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_170.log:Max CUDA memory: 1.1405G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_170.log:Max CUDA memory: 0.7037G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_52.log:[20:59:08] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_52.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_52.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_52.log:Max CUDA memory: 0.8328G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_52.log:Max CUDA memory: 1.6510G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_128.log:[22:56:11] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_128.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_128.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_128.log:Max CUDA memory: 1.0476G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_128.log:Max CUDA memory: 1.1847G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_66.log:[21:24:54] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_66.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_66.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_66.log:Max CUDA memory: 1.1405G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_66.log:Max CUDA memory: 1.2119G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_26.log:[20:24:59] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_26.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_26.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_26.log:Max CUDA memory: 1.5447G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_26.log:Max CUDA memory: 0.8264G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_92.log:[22:01:14] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_92.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_92.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_92.log:Max CUDA memory: 1.0909G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_92.log:Max CUDA memory: 0.7648G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_148.log:[23:24:08] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_148.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_148.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_148.log:Max CUDA memory: 1.0909G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_148.log:Max CUDA memory: 0.8525G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_106.log:[22:24:40] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_106.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_106.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_106.log:Max CUDA memory: 1.4850G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_106.log:Max CUDA memory: 1.0499G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_72.log:[21:30:39] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_72.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_72.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_72.log:Max CUDA memory: 1.5123G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_72.log:Max CUDA memory: 0.8219G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_38.log:[20:29:33] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_38.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_38.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_38.log:Max CUDA memory: 0.7923G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_38.log:Max CUDA memory: 1.1845G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_6.log:[19:55:27] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_6.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_6.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_6.log:Max CUDA memory: 1.0476G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_6.log:Max CUDA memory: 0.8520G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_190.log:[00:24:32] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_190.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_190.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_190.log:Max CUDA memory: 1.2105G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_190.log:Max CUDA memory: 0.9507G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_34.log:[20:27:33] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_34.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_34.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_34.log:Max CUDA memory: 0.7923G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_34.log:Max CUDA memory: 1.4863G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_120.log:[22:45:31] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_120.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_120.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_120.log:Max CUDA memory: 1.6005G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_120.log:Max CUDA memory: 1.0800G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_2.log:[19:55:09] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_2.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_2.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_2.log:Max CUDA memory: 0.9662G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_2.log:Max CUDA memory: 0.8523G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_32.log:[20:26:56] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_32.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_32.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_32.log:Max CUDA memory: 0.7923G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_32.log:Max CUDA memory: 1.4863G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_50.log:[20:57:52] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_50.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_50.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_50.log:Max CUDA memory: 0.9298G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_50.log:Max CUDA memory: 0.8177G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_68.log:[21:27:23] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_68.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_68.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_68.log:Max CUDA memory: 1.1405G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_68.log:Max CUDA memory: 0.9728G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_156.log:[23:38:43] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_156.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_156.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_156.log:Max CUDA memory: 0.9441G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_156.log:Max CUDA memory: 0.7143G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_10.log:[19:55:51] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_10.log:Max CUDA memory: 1.0909G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_10.log:Max CUDA memory: 0.7146G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_36.log:[20:29:33] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_36.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_36.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_36.log:Max CUDA memory: 1.2282G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_36.log:Max CUDA memory: 1.1795G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_74.log:[21:30:39] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_74.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_74.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_74.log:Max CUDA memory: 1.1405G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_74.log:Max CUDA memory: 0.7489G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_18.log:[19:56:47] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_18.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_18.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_18.log:Max CUDA memory: 0.9662G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_18.log:Max CUDA memory: 0.9698G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_134.log:[23:04:53] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_134.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_134.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_134.log:Max CUDA memory: 1.1597G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_134.log:Max CUDA memory: 1.4884G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_150.log:[23:26:18] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_150.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_150.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_150.log:Max CUDA memory: 0.9298G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_150.log:Max CUDA memory: 1.2776G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_100.log:[22:16:52] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_100.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_100.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_100.log:Max CUDA memory: 0.9662G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_100.log:Max CUDA memory: 1.1416G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_80.log:[21:47:19] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_80.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_80.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_80.log:Max CUDA memory: 0.7116G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_80.log:Max CUDA memory: 1.6275G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_136.log:[23:06:22] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_136.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_136.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_136.log:Max CUDA memory: 1.1405G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_136.log:Max CUDA memory: 0.7037G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_116.log:[22:34:09] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_116.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_116.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_116.log:Max CUDA memory: 1.3663G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_116.log:Max CUDA memory: 1.4355G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_28.log:[20:25:38] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_28.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_28.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_28.log:Max CUDA memory: 1.5754G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_28.log:Max CUDA memory: 1.4150G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_198.log:[00:43:28] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_198.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_198.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_198.log:Max CUDA memory: 0.7028G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_198.log:Max CUDA memory: 1.4344G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_108.log:[22:26:40] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_108.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_108.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_108.log:Max CUDA memory: 1.3285G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_108.log:Max CUDA memory: 0.7158G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_102.log:[19:52:46] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_102.log:[22:18:14] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_102.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_102.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_102.log:Max CUDA memory: 0.9441G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_102.log:Max CUDA memory: 0.7858G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_24.log:[20:24:24] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_24.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_24.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_24.log:Max CUDA memory: 0.7826G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_24.log:Max CUDA memory: 0.7853G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_154.log:[23:36:07] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_154.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_154.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_154.log:Max CUDA memory: 0.7923G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_154.log:Max CUDA memory: 1.4863G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_110.log:[22:29:20] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_110.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_110.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_110.log:Max CUDA memory: 1.5754G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_110.log:Max CUDA memory: 1.5840G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_178.log:[00:11:03] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_178.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_178.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_178.log:Max CUDA memory: 1.1405G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_178.log:Max CUDA memory: 1.4170G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_42.log:[20:52:20] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_42.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_42.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_42.log:Max CUDA memory: 0.7923G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_42.log:Max CUDA memory: 0.8931G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_94.log:[22:01:15] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_94.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_94.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_94.log:Max CUDA memory: 1.1597G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_94.log:Max CUDA memory: 1.4884G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_118.log:[22:38:37] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_118.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_118.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_118.log:Max CUDA memory: 0.7826G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_118.log:Max CUDA memory: 1.0478G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_184.log:[00:19:03] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_184.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_184.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_184.log:Max CUDA memory: 1.6153G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_184.log:Max CUDA memory: 1.6249G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_142.log:[23:15:29] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_142.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_142.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_142.log:Max CUDA memory: 1.3663G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_142.log:Max CUDA memory: 1.7129G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_160.log:[23:42:04] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_160.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_160.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_160.log:Max CUDA memory: 1.7038G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_160.log:Max CUDA memory: 1.6895G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_186.log:[00:22:15] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_186.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_186.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_186.log:Max CUDA memory: 1.4175G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_186.log:Max CUDA memory: 1.1063G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_104.log:[19:52:46] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_104.log:[22:22:39] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_104.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_104.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_104.log:Max CUDA memory: 1.3862G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_104.log:Max CUDA memory: 1.2560G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_76.log:[21:31:22] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_76.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_76.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_76.log:Max CUDA memory: 1.3862G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_76.log:Max CUDA memory: 0.7666G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_82.log:[21:49:06] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_82.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_82.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_82.log:Max CUDA memory: 0.9662G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_82.log:Max CUDA memory: 0.8358G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_40.log:[20:51:41] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_40.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_40.log:Max CUDA memory: 0.8328G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_40.log:Max CUDA memory: 0.9868G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_144.log:[23:21:26] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_144.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_144.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_144.log:Max CUDA memory: 0.8328G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_144.log:Max CUDA memory: 1.4583G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_138.log:[23:06:41] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_138.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_138.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_138.log:Max CUDA memory: 1.4532G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_138.log:Max CUDA memory: 0.7897G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_168.log:[23:53:12] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_168.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_168.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_168.log:Max CUDA memory: 0.7116G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_168.log:Max CUDA memory: 1.6494G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_96.log:[22:02:40] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_96.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_96.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_96.log:Max CUDA memory: 0.9997G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_96.log:Max CUDA memory: 1.3101G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_192.log:[00:35:00] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_192.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_192.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_192.log:Max CUDA memory: 0.8499G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_192.log:Max CUDA memory: 1.0472G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_176.log:[00:06:58] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_176.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_176.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_176.log:Max CUDA memory: 1.2105G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_176.log:Max CUDA memory: 0.8753G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_86.log:[21:54:28] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_86.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_86.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_86.log:Max CUDA memory: 1.1069G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_86.log:Max CUDA memory: 1.3117G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_46.log:[20:55:08] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_46.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_46.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_46.log:Max CUDA memory: 1.0327G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_46.log:Max CUDA memory: 1.2109G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_174.log:[00:06:24] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_174.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_174.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_174.log:Max CUDA memory: 1.5123G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_174.log:Max CUDA memory: 1.4133G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_60.log:[21:19:28] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_60.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_60.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_60.log:Max CUDA memory: 0.8328G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_60.log:Max CUDA memory: 1.1384G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_30.log:[20:26:55] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_30.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_30.log:Max CUDA memory: 1.5447G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_30.log:Max CUDA memory: 1.2381G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_182.log:[00:18:45] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_182.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_182.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_182.log:Max CUDA memory: 0.9662G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_182.log:Max CUDA memory: 0.7332G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_194.log:[00:38:22] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_194.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_194.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_194.log:Max CUDA memory: 0.8725G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_194.log:Max CUDA memory: 1.0054G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_130.log:[22:58:04] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_130.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_130.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_130.log:Max CUDA memory: 0.9081G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_130.log:Max CUDA memory: 0.7834G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_56.log:[20:59:46] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_56.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_56.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_56.log:Max CUDA memory: 1.2488G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_56.log:Max CUDA memory: 1.2370G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_172.log:[00:03:00] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_172.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_172.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_172.log:Max CUDA memory: 1.1405G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_172.log:Max CUDA memory: 1.6501G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_98.log:[22:07:14] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_98.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_98.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_98.log:Max CUDA memory: 1.6447G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_98.log:Max CUDA memory: 1.1208G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_180.log:[00:18:41] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_180.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_180.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_180.log:Max CUDA memory: 0.7586G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_180.log:Max CUDA memory: 0.8933G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_152.log:[23:33:50] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_152.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_152.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_152.log:Max CUDA memory: 1.4333G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_152.log:Max CUDA memory: 0.9685G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_146.log:[23:24:04] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_146.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_146.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_146.log:Max CUDA memory: 0.8499G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_146.log:Max CUDA memory: 0.8359G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_70.log:[21:29:41] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_70.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_70.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_70.log:Max CUDA memory: 1.2488G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_70.log:Max CUDA memory: 0.9109G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_124.log:[22:52:50] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_124.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_124.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_124.log:Max CUDA memory: 0.8499G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_124.log:Max CUDA memory: 0.8519G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_8.log:[19:55:38] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_8.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_8.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_8.log:Max CUDA memory: 1.0327G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_8.log:Max CUDA memory: 1.4354G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_164.log:[23:50:11] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_164.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_164.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_164.log:Max CUDA memory: 1.2713G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_164.log:Max CUDA memory: 1.3370G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_158.log:[23:38:42] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_158.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_158.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_158.log:Max CUDA memory: 1.5754G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_158.log:Max CUDA memory: 0.9994G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_22.log:[20:24:18] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_22.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_22.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_22.log:Max CUDA memory: 1.6005G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_22.log:Max CUDA memory: 0.7721G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_140.log:[23:13:42] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_140.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_140.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_140.log:Max CUDA memory: 1.1405G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_140.log:Max CUDA memory: 1.1462G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_188.log:[00:23:21] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_188.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_188.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_188.log:Max CUDA memory: 1.2713G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_188.log:Max CUDA memory: 0.9105G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_44.log:[20:54:17] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_44.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_44.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_44.log:Max CUDA memory: 1.2713G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_44.log:Max CUDA memory: 1.1171G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_162.log:[23:47:09] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_162.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_162.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_162.log:Max CUDA memory: 1.6447G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_162.log:Max CUDA memory: 1.3118G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_78.log:[21:33:14] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_78.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_78.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_78.log:Max CUDA memory: 1.5754G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_78.log:Max CUDA memory: 1.4609G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_166.log:[23:51:28] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_166.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_166.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_166.log:Max CUDA memory: 1.6447G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_166.log:Max CUDA memory: 1.0788G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_84.log:[21:54:10] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_84.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_84.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_84.log:Max CUDA memory: 0.7028G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_84.log:Max CUDA memory: 1.0800G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_90.log:[21:59:59] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_90.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_90.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_90.log:Max CUDA memory: 0.8725G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_90.log:Max CUDA memory: 0.7629G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_62.log:[21:20:25] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_62.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_62.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_62.log:Max CUDA memory: 0.9662G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_62.log:Max CUDA memory: 1.0941G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_64.log:[21:24:22] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_64.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_64.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_64.log:Max CUDA memory: 1.3285G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_64.log:Max CUDA memory: 1.1012G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_4.log:[19:55:16] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_4.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_4.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_4.log:Max CUDA memory: 1.0909G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_4.log:Max CUDA memory: 1.0466G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_132.log:[23:04:08] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_132.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_132.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_132.log:Max CUDA memory: 1.3034G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_132.log:Max CUDA memory: 1.3521G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_12.log:[19:56:32] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_12.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_12.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_12.log:Max CUDA memory: 1.5123G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_12.log:Max CUDA memory: 1.4884G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_122.log:[22:45:43] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_122.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_122.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_122.log:Max CUDA memory: 0.9441G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_122.log:Max CUDA memory: 1.0046G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_16.log:[19:56:41] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_16.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_16.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_16.log:Max CUDA memory: 1.0327G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_16.log:Max CUDA memory: 1.2109G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_126.log:[22:55:27] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_126.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_126.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_126.log:Max CUDA memory: 0.7826G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_126.log:Max CUDA memory: 0.9105G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_14.log:[19:56:32] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_14.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_14.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_14.log:Max CUDA memory: 1.4532G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_14.log:Max CUDA memory: 1.3035G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_48.log:[20:57:32] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_48.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_48.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_48.log:Max CUDA memory: 1.1405G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_48.log:Max CUDA memory: 1.1836G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_0.log:[19:55:07] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_0.log:Max CUDA memory: 0.9662G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_0.log:Max CUDA memory: 0.8921G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_58.log:[21:00:11] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_58.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_58.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_58.log:Max CUDA memory: 0.7116G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_58.log:Max CUDA memory: 1.6758G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_114.log:[22:33:17] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_114.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_114.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_114.log:Max CUDA memory: 0.9298G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_114.log:Max CUDA memory: 1.1890G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_112.log:[22:30:32] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_112.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_112.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_112.log:Max CUDA memory: 0.9662G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_112.log:Max CUDA memory: 1.2539G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_20.log:[20:22:54] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_20.log:Max CUDA memory: 1.3285G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_20.log:Max CUDA memory: 1.1165G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_54.log:[20:59:27] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_54.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_54.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_54.log:Max CUDA memory: 1.0327G
tutorials/halluc_SH3_binder/output/20220103_sh3_r1/sh3_r1_54.log:Max CUDA memory: 1.2109G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_20.log:[16:50:53] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_20.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_20.log:Max CUDA memory: 0.5813G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_20.log:Max CUDA memory: 0.5946G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_20.log:Max CUDA memory: 0.5900G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_20.log:Max CUDA memory: 0.5890G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_20.log:[16:40:31] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_20.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_20.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_20.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_20.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_20.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_40.log:[17:17:02] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_40.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_40.log:Max CUDA memory: 0.6413G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_40.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_40.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_40.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_40.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_0.log:[14:40:00] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_0.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_0.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_0.log:Max CUDA memory: 0.5667G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_0.log:Max CUDA memory: 0.5808G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_0.log:Max CUDA memory: 0.5670G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_5.log:[15:30:29] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_5.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_5.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_5.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_5.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_5.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_45.log:[17:52:25] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_45.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_45.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_45.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_45.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_45.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_45.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_45.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_30.log:[16:08:47] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_30.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_30.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_30.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_30.log:Max CUDA memory: 0.5974G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_30.log:Max CUDA memory: 0.6001G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_30.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_5.log:[15:25:54] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_5.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_5.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_5.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_5.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_5.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_30.log:[15:50:01] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_30.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_30.log:Max CUDA memory: 0.6071G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_30.log:Max CUDA memory: 0.6795G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_30.log:Max CUDA memory: 0.6799G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_30.log:Max CUDA memory: 0.6788G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_30.log:Max CUDA memory: 0.6792G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_15.log:[16:31:28] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_15.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_15.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_15.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_15.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_15.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_25.log:[15:04:11] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_25.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_25.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_25.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_25.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_25.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_25.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_25.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_40.log:[17:36:16] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_40.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_40.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_40.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_40.log:Max CUDA memory: 0.5977G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_40.log:Max CUDA memory: 0.5979G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_40.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_10.log:[15:40:28] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_10.log:Max CUDA memory: 0.6413G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_10.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_10.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_10.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_10.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_45.log:[18:13:38] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_45.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_45.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_45.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_45.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_45.log:Max CUDA memory: 0.5974G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_45.log:Max CUDA memory: 0.5979G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_45.log:Max CUDA memory: 0.5973G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_30.log:[15:28:18] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_30.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_30.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_30.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_30.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_30.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_30.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_40.log:[17:38:50] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_40.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_40.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_40.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_40.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_40.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_40.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_35.log:[16:48:52] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_35.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_35.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_35.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_35.log:Max CUDA memory: 0.5813G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_35.log:Max CUDA memory: 0.5886G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_35.log:Max CUDA memory: 0.5913G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_35.log:Max CUDA memory: 0.5882G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_45.log:[17:58:15] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_45.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_45.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_45.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_45.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_45.log:Max CUDA memory: 0.5678G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_45.log:Max CUDA memory: 0.5682G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_45.log:Max CUDA memory: 0.5738G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_10.log:[15:59:38] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_10.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_10.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_10.log:Max CUDA memory: 0.5974G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_10.log:Max CUDA memory: 0.5979G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_10.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_35.log:[16:42:21] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_35.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_35.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_35.log:Max CUDA memory: 0.6071G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_35.log:Max CUDA memory: 0.6795G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_35.log:Max CUDA memory: 0.6799G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_35.log:Max CUDA memory: 0.6788G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_35.log:Max CUDA memory: 0.6792G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_45.log:[18:14:32] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_45.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_45.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_45.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_45.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_45.log:Max CUDA memory: 0.6001G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_45.log:Max CUDA memory: 0.6015G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_45.log:Max CUDA memory: 0.6090G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_40.log:[17:34:16] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_40.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_40.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_40.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_40.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_40.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_40.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_25.log:[15:14:11] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_25.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_25.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_25.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_25.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_25.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_25.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_25.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_25.log:[15:27:06] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_25.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_25.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_25.log:Max CUDA memory: 0.5618G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_25.log:Max CUDA memory: 0.6248G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_25.log:Max CUDA memory: 0.6251G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_25.log:Max CUDA memory: 0.6261G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_25.log:Max CUDA memory: 0.6254G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_35.log:[17:01:57] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_35.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_35.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_35.log:Max CUDA memory: 0.5618G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_35.log:Max CUDA memory: 0.6248G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_35.log:Max CUDA memory: 0.6251G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_35.log:Max CUDA memory: 0.6261G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_35.log:Max CUDA memory: 0.6254G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_40.log:[17:24:46] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_40.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_40.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_40.log:Max CUDA memory: 0.5813G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_40.log:Max CUDA memory: 0.5877G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_40.log:Max CUDA memory: 0.5889G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_40.log:Max CUDA memory: 0.5947G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_35.log:[17:06:40] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_35.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_35.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_35.log:Max CUDA memory: 0.5739G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_35.log:Max CUDA memory: 0.6414G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_35.log:Max CUDA memory: 0.6404G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_35.log:Max CUDA memory: 0.6410G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_35.log:Max CUDA memory: 0.6396G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_30.log:[16:24:15] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_30.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_30.log:Max CUDA memory: 0.5618G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_30.log:Max CUDA memory: 0.6248G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_30.log:Max CUDA memory: 0.6251G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_30.log:Max CUDA memory: 0.6261G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_30.log:Max CUDA memory: 0.6254G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_35.log:[16:35:50] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_35.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_35.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_35.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_35.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_35.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_35.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_35.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_25.log:[15:13:17] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_25.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_25.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_25.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_25.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_25.log:Max CUDA memory: 0.5974G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_25.log:Max CUDA memory: 0.5979G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_25.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_15.log:[16:29:33] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_15.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_15.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_15.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_15.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_15.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_10.log:[15:47:26] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_10.log:Max CUDA memory: 0.6071G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_10.log:Max CUDA memory: 0.6795G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_10.log:Max CUDA memory: 0.6799G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_10.log:Max CUDA memory: 0.6788G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_10.log:Max CUDA memory: 0.6792G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_10.log:[16:00:28] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_10.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_10.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_10.log:Max CUDA memory: 0.6098G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_10.log:Max CUDA memory: 0.6199G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_10.log:Max CUDA memory: 0.6056G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_5.log:[15:13:10] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_5.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_5.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_5.log:Max CUDA memory: 0.5671G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_5.log:Max CUDA memory: 0.5713G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_5.log:Max CUDA memory: 0.5802G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_5.log:[15:23:28] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_5.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_5.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_5.log:Max CUDA memory: 0.5760G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_5.log:Max CUDA memory: 0.5828G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_5.log:Max CUDA memory: 0.5769G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_15.log:[16:31:20] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_15.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_15.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_15.log:Max CUDA memory: 0.6077G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_15.log:Max CUDA memory: 0.6128G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_15.log:Max CUDA memory: 0.6091G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_20.log:[16:58:55] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_20.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_20.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_20.log:Max CUDA memory: 0.5814G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_20.log:Max CUDA memory: 0.5815G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_20.log:Max CUDA memory: 0.5811G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_35.log:[17:00:41] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_35.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_35.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_35.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_35.log:Max CUDA memory: 0.5977G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_35.log:Max CUDA memory: 0.6010G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_35.log:Max CUDA memory: 0.6033G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_35.log:Max CUDA memory: 0.6071G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_5.log:[15:09:37] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_5.log:Max CUDA memory: 0.6413G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_5.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_5.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_5.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_5.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_15.log:[16:30:09] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_15.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_15.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_15.log:Max CUDA memory: 0.5974G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_15.log:Max CUDA memory: 0.6025G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_15.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_20.log:[16:50:00] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_20.log:Max CUDA memory: 0.6071G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_20.log:Max CUDA memory: 0.6795G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_20.log:Max CUDA memory: 0.6799G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_20.log:Max CUDA memory: 0.6788G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_20.log:Max CUDA memory: 0.6792G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_10.log:[16:08:20] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_10.log:Max CUDA memory: 0.5739G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_10.log:Max CUDA memory: 0.6414G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_10.log:Max CUDA memory: 0.6404G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_10.log:Max CUDA memory: 0.6410G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_10.log:Max CUDA memory: 0.6396G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_30.log:[16:05:48] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_30.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_30.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_30.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_30.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_30.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_30.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_45.log:[18:01:43] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_45.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_45.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_45.log:Max CUDA memory: 0.6071G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_45.log:Max CUDA memory: 0.6795G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_45.log:Max CUDA memory: 0.6799G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_45.log:Max CUDA memory: 0.6788G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_45.log:Max CUDA memory: 0.6792G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_45.log:[18:21:25] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_45.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_45.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_45.log:Max CUDA memory: 0.5618G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_45.log:Max CUDA memory: 0.6248G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_45.log:Max CUDA memory: 0.6251G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_45.log:Max CUDA memory: 0.6261G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_45.log:Max CUDA memory: 0.6266G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_40.log:[17:12:24] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_40.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_40.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_40.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_40.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_40.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_40.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_10.log:[15:49:59] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_10.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_10.log:Max CUDA memory: 0.5813G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_10.log:Max CUDA memory: 0.5871G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_10.log:Max CUDA memory: 0.5841G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_10.log:Max CUDA memory: 0.5889G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_15.log:[16:11:37] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_15.log:Max CUDA memory: 0.6413G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_15.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_15.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_15.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_15.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_45.log:[18:24:13] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_45.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_45.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_45.log:Max CUDA memory: 0.5739G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_45.log:Max CUDA memory: 0.6414G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_45.log:Max CUDA memory: 0.6404G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_45.log:Max CUDA memory: 0.6410G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_45.log:Max CUDA memory: 0.6396G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_10.log:[15:42:29] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_10.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_10.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_10.log:Max CUDA memory: 0.5690G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_10.log:Max CUDA memory: 0.5661G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_10.log:Max CUDA memory: 0.5670G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_30.log:[16:11:27] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_30.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_30.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_30.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_30.log:Max CUDA memory: 0.6059G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_30.log:Max CUDA memory: 0.6150G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_30.log:Max CUDA memory: 0.6009G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_10.log:[15:58:58] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_10.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_10.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_10.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_10.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_10.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_20.log:[16:43:39] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_20.log:Max CUDA memory: 0.6413G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_20.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_20.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_20.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_20.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_25.log:[14:40:42] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_25.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_25.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_25.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_25.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_25.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_25.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_25.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_40.log:[17:37:51] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_40.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_40.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_40.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_40.log:Max CUDA memory: 0.6125G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_40.log:Max CUDA memory: 0.6032G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_40.log:Max CUDA memory: 0.6089G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_5.log:[15:28:25] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_5.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_5.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_5.log:Max CUDA memory: 0.5974G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_5.log:Max CUDA memory: 0.5979G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_5.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_15.log:[16:20:40] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_15.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_15.log:Max CUDA memory: 0.5813G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_15.log:Max CUDA memory: 0.5880G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_15.log:Max CUDA memory: 0.5873G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_15.log:Max CUDA memory: 0.5884G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_25.log:[15:01:32] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_25.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_25.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_25.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_25.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_25.log:Max CUDA memory: 0.5788G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_25.log:Max CUDA memory: 0.5805G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_25.log:Max CUDA memory: 0.5796G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_0.log:[14:39:56] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_0.log:Max CUDA memory: 0.6413G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_0.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_0.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_0.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_0.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_20.log:[17:08:55] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_20.log:Max CUDA memory: 0.5618G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_20.log:Max CUDA memory: 0.6248G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_20.log:Max CUDA memory: 0.6251G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_20.log:Max CUDA memory: 0.6261G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_20.log:Max CUDA memory: 0.6254G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_35.log:[16:57:34] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_35.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_35.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_35.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_35.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_35.log:Max CUDA memory: 0.5974G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_35.log:Max CUDA memory: 0.5982G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_35.log:Max CUDA memory: 0.5986G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_5.log:[15:17:19] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_5.log:Max CUDA memory: 0.6071G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_5.log:Max CUDA memory: 0.6795G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_5.log:Max CUDA memory: 0.6799G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_5.log:Max CUDA memory: 0.6788G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_5.log:Max CUDA memory: 0.6792G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_0.log:[14:58:42] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_0.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_0.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_0.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_0.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_0.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_15.log:[16:09:38] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_15.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_15.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_15.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_15.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_15.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_45.log:[18:16:43] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_45.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_45.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_45.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_45.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_45.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_45.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_45.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_5.log:[15:37:30] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_5.log:Max CUDA memory: 0.5739G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_5.log:Max CUDA memory: 0.6414G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_5.log:Max CUDA memory: 0.6404G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_5.log:Max CUDA memory: 0.6410G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_5.log:Max CUDA memory: 0.6396G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_10.log:[15:38:33] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_10.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_10.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_10.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_10.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_10.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_15.log:[16:12:32] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_15.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_15.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_15.log:Max CUDA memory: 0.5663G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_15.log:Max CUDA memory: 0.5671G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_15.log:Max CUDA memory: 0.5778G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_5.log:[15:37:30] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_5.log:Max CUDA memory: 0.5618G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_5.log:Max CUDA memory: 0.6248G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_5.log:Max CUDA memory: 0.6251G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_5.log:Max CUDA memory: 0.6261G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_5.log:Max CUDA memory: 0.6254G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_0.log:[14:58:10] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_0.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_0.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_0.log:Max CUDA memory: 0.5974G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_0.log:Max CUDA memory: 0.6015G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_0.log:Max CUDA memory: 0.5993G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_40.log:[17:42:19] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_40.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_40.log:Max CUDA memory: 0.5618G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_40.log:Max CUDA memory: 0.6248G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_40.log:Max CUDA memory: 0.6251G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_40.log:Max CUDA memory: 0.6261G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_40.log:Max CUDA memory: 0.6254G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_30.log:[16:13:51] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_30.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_30.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_30.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_30.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_30.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_30.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_40.log:[17:20:52] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_40.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_40.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_40.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_40.log:Max CUDA memory: 0.5765G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_40.log:Max CUDA memory: 0.5661G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_40.log:Max CUDA memory: 0.5771G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_45.log:[18:01:57] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_45.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_45.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_45.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_45.log:Max CUDA memory: 0.5813G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_45.log:Max CUDA memory: 0.5891G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_45.log:Max CUDA memory: 0.5893G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_45.log:Max CUDA memory: 0.5893G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_45.log:[17:56:20] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_45.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_45.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_45.log:Max CUDA memory: 0.6413G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_45.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_45.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_45.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_45.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_35.log:[16:41:35] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_35.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_35.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_35.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_35.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_35.log:Max CUDA memory: 0.5756G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_35.log:Max CUDA memory: 0.5747G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_35.log:Max CUDA memory: 0.5684G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_20.log:[17:02:40] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_20.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_20.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_20.log:Max CUDA memory: 0.6013G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_20.log:Max CUDA memory: 0.6033G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_20.log:Max CUDA memory: 0.6089G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_35.log:[16:40:11] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_35.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_35.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_35.log:Max CUDA memory: 0.6413G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_35.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_35.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_35.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_35.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_25.log:[14:54:19] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_25.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_25.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_25.log:Max CUDA memory: 0.6071G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_25.log:Max CUDA memory: 0.6795G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_25.log:Max CUDA memory: 0.6799G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_25.log:Max CUDA memory: 0.6788G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_25.log:Max CUDA memory: 0.6792G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_0.log:[14:38:59] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_0.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_0.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_0.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_0.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_0.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_0.log:[14:57:04] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_0.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_0.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_0.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_0.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_0.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_10.log:[16:00:55] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_10.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_10.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_10.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_10.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_10.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_0.log:[14:55:51] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_0.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_0.log:Max CUDA memory: 0.5739G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_0.log:Max CUDA memory: 0.5762G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_0.log:Max CUDA memory: 0.5813G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_0.log:Max CUDA memory: 0.5824G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_15.log:[16:38:08] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_15.log:Max CUDA memory: 0.5618G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_15.log:Max CUDA memory: 0.6248G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_15.log:Max CUDA memory: 0.6262G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_15.log:Max CUDA memory: 0.6261G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_15.log:Max CUDA memory: 0.6254G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_45.log:[18:11:08] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_45.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_45.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_45.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_45.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_45.log:Max CUDA memory: 0.5810G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_45.log:Max CUDA memory: 0.5691G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_45.log:Max CUDA memory: 0.5866G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_20.log:[17:10:07] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_20.log:Max CUDA memory: 0.5739G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_20.log:Max CUDA memory: 0.6414G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_20.log:Max CUDA memory: 0.6404G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_20.log:Max CUDA memory: 0.6410G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_20.log:Max CUDA memory: 0.6396G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_0.log:[14:39:37] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_0.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_0.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_0.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_0.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_0.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_10.log:[15:54:40] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_10.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_10.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_10.log:Max CUDA memory: 0.5767G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_10.log:Max CUDA memory: 0.5728G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_10.log:Max CUDA memory: 0.5820G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_40.log:[17:33:39] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_40.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_40.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_40.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_40.log:Max CUDA memory: 0.5785G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_40.log:Max CUDA memory: 0.5779G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_40.log:Max CUDA memory: 0.5769G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_5.log:[15:09:11] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_5.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_5.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_5.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_5.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_5.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_20.log:[17:00:41] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_20.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_20.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_20.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_20.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_20.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_25.log:[15:27:41] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_25.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_25.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_25.log:Max CUDA memory: 0.5739G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_25.log:Max CUDA memory: 0.6414G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_25.log:Max CUDA memory: 0.6404G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_25.log:Max CUDA memory: 0.6410G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_25.log:Max CUDA memory: 0.6396G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_35.log:[17:01:20] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_35.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_35.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_35.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_35.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_35.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_35.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_35.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_5.log:[15:18:47] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_5.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_5.log:Max CUDA memory: 0.5813G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_5.log:Max CUDA memory: 0.5905G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_5.log:Max CUDA memory: 0.5857G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_5.log:Max CUDA memory: 0.5894G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_30.log:[16:00:41] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_30.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_30.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_30.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_30.log:Max CUDA memory: 0.5730G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_30.log:Max CUDA memory: 0.5760G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_30.log:Max CUDA memory: 0.5812G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_30.log:[15:43:55] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_30.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_30.log:Max CUDA memory: 0.6413G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_30.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_30.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_30.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_30.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_35.log:[16:54:47] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_35.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_35.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_35.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_35.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_35.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_35.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_35.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_0.log:[14:58:40] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_0.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_0.log:Max CUDA memory: 0.5976G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_0.log:Max CUDA memory: 0.6059G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_0.log:Max CUDA memory: 0.5996G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_0.log:Max CUDA memory: 0.6028G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_45.log:[18:11:58] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_45.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_45.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_45.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_45.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_45.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_45.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_78_r2_45.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_15.log:[16:07:52] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_15.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_15.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_15.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_15.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_15.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_0.log:[14:40:06] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_0.log:Max CUDA memory: 0.6071G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_0.log:Max CUDA memory: 0.6795G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_0.log:Max CUDA memory: 0.6799G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_0.log:Max CUDA memory: 0.6788G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_0.log:Max CUDA memory: 0.6792G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_40.log:[17:46:11] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_40.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_40.log:Max CUDA memory: 0.5739G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_40.log:Max CUDA memory: 0.6414G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_40.log:Max CUDA memory: 0.6404G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_40.log:Max CUDA memory: 0.6410G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_40.log:Max CUDA memory: 0.6396G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_20.log:[17:02:53] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_20.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_20.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_20.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_20.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_120_r2_20.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_40.log:[17:23:31] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_40.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_40.log:Max CUDA memory: 0.6071G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_40.log:Max CUDA memory: 0.6795G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_40.log:Max CUDA memory: 0.6799G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_40.log:Max CUDA memory: 0.6788G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_40.log:Max CUDA memory: 0.6792G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_15.log:[16:18:38] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_15.log:Max CUDA memory: 0.6071G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_15.log:Max CUDA memory: 0.6795G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_15.log:Max CUDA memory: 0.6799G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_15.log:Max CUDA memory: 0.6788G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_87_r2_15.log:Max CUDA memory: 0.6792G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_40.log:[17:15:03] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_40.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_40.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_40.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_40.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_40.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_40.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_40.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_5.log:[15:29:56] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_5.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_5.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_5.log:Max CUDA memory: 0.6103G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_5.log:Max CUDA memory: 0.6019G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_5.log:Max CUDA memory: 0.6085G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_35.log:[16:52:34] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_35.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_35.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_35.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_35.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_35.log:Max CUDA memory: 0.5816G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_35.log:Max CUDA memory: 0.5814G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_35.log:Max CUDA memory: 0.5726G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_30.log:[15:36:54] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_30.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_30.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_30.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_30.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_30.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_30.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_25.log:[14:40:42] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_25.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_25.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_25.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_25.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_25.log:Max CUDA memory: 0.5690G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_25.log:Max CUDA memory: 0.5661G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_25.log:Max CUDA memory: 0.5699G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_35.log:[16:28:49] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_35.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_35.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_35.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_35.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_35.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_35.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_35.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_20.log:[16:40:46] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_20.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_20.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_20.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_20.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_20.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_25.log:[15:14:13] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_25.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_25.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_25.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_25.log:Max CUDA memory: 0.6081G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_25.log:Max CUDA memory: 0.6131G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_25.log:Max CUDA memory: 0.6102G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_112_r2_25.log:Max CUDA memory: 0.6148G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_10.log:[15:37:43] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_10.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_10.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_10.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_10.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_10.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_5.log:[15:07:37] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_5.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_5.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_5.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_5.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_5.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_25.log:[14:40:40] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_25.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_25.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_25.log:Max CUDA memory: 0.6580G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_25.log:Max CUDA memory: 0.7419G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_25.log:Max CUDA memory: 0.7427G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_25.log:Max CUDA memory: 0.7406G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_110_r2_25.log:Max CUDA memory: 0.7435G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_25.log:[15:00:12] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_25.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_25.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_25.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_25.log:Max CUDA memory: 0.5813G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_25.log:Max CUDA memory: 0.5841G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_25.log:Max CUDA memory: 0.5901G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_25.log:Max CUDA memory: 0.5890G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_30.log:[16:25:46] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_30.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_30.log:Max CUDA memory: 0.5739G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_30.log:Max CUDA memory: 0.6414G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_30.log:Max CUDA memory: 0.6411G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_30.log:Max CUDA memory: 0.6410G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_30.log:Max CUDA memory: 0.6396G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_10.log:[16:07:53] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_10.log:Max CUDA memory: 0.5618G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_10.log:Max CUDA memory: 0.6248G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_10.log:Max CUDA memory: 0.6251G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_10.log:Max CUDA memory: 0.6261G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_10.log:Max CUDA memory: 0.6254G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_20.log:[16:45:08] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_20.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_20.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_20.log:Max CUDA memory: 0.5680G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_20.log:Max CUDA memory: 0.5813G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_20.log:Max CUDA memory: 0.5762G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_30.log:[15:44:46] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_30.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_30.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_30.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_30.log:Max CUDA memory: 0.5675G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_30.log:Max CUDA memory: 0.5733G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_146_r2_30.log:Max CUDA memory: 0.5770G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_20.log:[17:02:09] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_20.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_20.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_20.log:Max CUDA memory: 0.5405G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_20.log:Max CUDA memory: 0.5972G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_20.log:Max CUDA memory: 0.5974G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_20.log:Max CUDA memory: 0.5979G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_69_r2_20.log:Max CUDA memory: 0.6031G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_15.log:[16:28:08] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_15.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_15.log:Max CUDA memory: 0.5673G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_15.log:Max CUDA memory: 0.5797G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_15.log:Max CUDA memory: 0.5802G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_192_r2_15.log:Max CUDA memory: 0.5720G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_45.log:[17:50:47] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_45.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_45.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_45.log:Max CUDA memory: 0.6610G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_45.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_45.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_45.log:Max CUDA memory: 0.7469G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_22_r2_45.log:Max CUDA memory: 0.7453G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_30.log:[15:57:59] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_30.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_30.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_30.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_30.log:Max CUDA memory: 0.5813G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_30.log:Max CUDA memory: 0.5939G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_30.log:Max CUDA memory: 0.5923G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_30.log:Max CUDA memory: 0.5948G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_0.log:[15:06:32] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_0.log:Max CUDA memory: 0.5739G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_0.log:Max CUDA memory: 0.6414G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_0.log:Max CUDA memory: 0.6404G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_0.log:Max CUDA memory: 0.6410G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_0.log:Max CUDA memory: 0.6396G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_0.log:[14:40:14] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_0.log:Max CUDA memory: 0.5294G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_0.log:Max CUDA memory: 0.5813G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_0.log:Max CUDA memory: 0.5938G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_0.log:Max CUDA memory: 0.5866G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_43_r2_0.log:Max CUDA memory: 0.5890G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_25.log:[14:40:42] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_25.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_25.log:Using CUDA device(s):  cuda:0: (Quadro RTX 8000); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_25.log:Max CUDA memory: 0.6413G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_25.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_25.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_25.log:Max CUDA memory: 0.7222G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_33_r2_25.log:Max CUDA memory: 0.7189G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_15.log:[16:39:18] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_15.log:Max CUDA memory: 0.5739G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_15.log:Max CUDA memory: 0.6414G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_15.log:Max CUDA memory: 0.6404G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_15.log:Max CUDA memory: 0.6410G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_66_r2_15.log:Max CUDA memory: 0.6396G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_0.log:[15:00:59] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_0.log:Max CUDA memory: 0.5618G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_0.log:Max CUDA memory: 0.6248G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_0.log:Max CUDA memory: 0.6251G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_0.log:Max CUDA memory: 0.6261G
tutorials/halluc_SH3_binder/output/20220104_sh3_r2/sh3_r1_85_r2_0.log:Max CUDA memory: 0.6254G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_5.log:Max CUDA memory: 1.9384G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_5.log:Max CUDA memory: 1.8461G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_5.log:Max CUDA memory: 1.8468G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_5.log:Max CUDA memory: 1.8466G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_5.log:Max CUDA memory: 1.8459G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_0.log:Max CUDA memory: 1.9384G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_0.log:Max CUDA memory: 1.8368G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_0.log:Max CUDA memory: 1.8382G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_0.log:Max CUDA memory: 1.8395G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_0.log:Max CUDA memory: 1.8376G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_5.log:Max CUDA memory: 1.9384G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_5.log:Max CUDA memory: 1.8368G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_5.log:Max CUDA memory: 1.8382G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_5.log:Max CUDA memory: 1.8395G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_13_r2_5.log:Max CUDA memory: 1.8376G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_0.log:Max CUDA memory: 1.9384G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_0.log:Max CUDA memory: 1.8462G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_0.log:Max CUDA memory: 1.8462G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_0.log:Max CUDA memory: 1.8461G
tutorials/halluc_PD-L1_binder/output/run2/pd1_r1_10_r2_0.log:Max CUDA memory: 1.8480G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_0.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_0.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_0.log:Max CUDA memory: 3.6553G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_0.log:Max CUDA memory: 3.1808G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_0.log:Max CUDA memory: 3.2444G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_0.log:Max CUDA memory: 3.3004G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_0.log:Max CUDA memory: 3.2432G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_15.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_15.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_15.log:Max CUDA memory: 3.8774G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_15.log:Max CUDA memory: 3.1332G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_15.log:Max CUDA memory: 3.0713G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_15.log:Max CUDA memory: 3.1028G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_15.log:Max CUDA memory: 3.6642G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_10.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_10.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_10.log:Max CUDA memory: 2.9691G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_10.log:Max CUDA memory: 3.2643G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_10.log:Max CUDA memory: 3.3349G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_10.log:Max CUDA memory: 2.9424G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_10.log:Max CUDA memory: 3.1784G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_5.log:Loading structure prediction model onto device cuda:0...
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_5.log:Using CUDA device(s):  cuda:0: (GeForce RTX 2080); 
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_5.log:Max CUDA memory: 3.0378G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_5.log:Max CUDA memory: 3.5075G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_5.log:Max CUDA memory: 3.0116G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_5.log:Max CUDA memory: 3.0429G
tutorials/halluc_PD-L1_binder/output/run1/pd1_r1_5.log:Max CUDA memory: 3.2391G
tutorials/halluc_PD-L1_binder/gpu_submit.sh:#   ./gpu_submit.sh commands.list jobname
tutorials/halluc_PD-L1_binder/gpu_submit.sh:#   ./gpu_submit.sh commands.list
tutorials/halluc_PD-L1_binder/gpu_submit.sh:sbatch -a 1-$(cat $1 | wc -l) -p gpu -J $jobname \
tutorials/halluc_PD-L1_binder/gpu_submit.sh:       -c 2 --mem=12g --gres=gpu:rtx2080:1 \
hallucination/models/alphafold/README.md:        [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
hallucination/models/alphafold/README.md:        for GPU support.
hallucination/models/alphafold/README.md:1.  Check that AlphaFold will be able to use a GPU by running:
hallucination/models/alphafold/README.md:    docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
hallucination/models/alphafold/README.md:    The output of this command should show a list of your GPUs. If it doesn't,
hallucination/models/alphafold/README.md:    [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
hallucination/models/alphafold/README.md:    [NVIDIA Docker issue](https://github.com/NVIDIA/nvidia-docker/issues/1447#issuecomment-801479573).
hallucination/models/alphafold/README.md:was tested on Google Cloud with a machine using the `nvidia-gpu-cloud-image`
hallucination/models/alphafold/README.md:3 TB disk, and an A100 GPU.
hallucination/models/alphafold/README.md:    By default, Alphafold will attempt to use all visible GPU devices. To use a
hallucination/models/alphafold/README.md:    subset, specify a comma-separated list of GPU UUID(s) or index(es) using the
hallucination/models/alphafold/README.md:    `--gpu_devices` flag. See
hallucination/models/alphafold/README.md:    [GPU enumeration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration)
hallucination/models/alphafold/run_alphafold.py:                     'deterministic, because processes like GPU inference are '
hallucination/models/alphafold/docker/Dockerfile:ARG CUDA=11.0
hallucination/models/alphafold/docker/Dockerfile:FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu18.04
hallucination/models/alphafold/docker/Dockerfile:ARG CUDA
hallucination/models/alphafold/docker/Dockerfile:      cuda-command-line-tools-${CUDA/./-} \
hallucination/models/alphafold/docker/Dockerfile:      cudatoolkit==${CUDA_VERSION} \
hallucination/models/alphafold/docker/Dockerfile:    && pip3 install --upgrade jax jaxlib==0.1.69+cuda${CUDA/./} -f \
hallucination/models/alphafold/docker/Dockerfile:# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
hallucination/models/alphafold/docker/Dockerfile:# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
hallucination/models/alphafold/docker/run_docker.py:flags.DEFINE_bool('use_gpu', True, 'Enable NVIDIA runtime to run with GPUs.')
hallucination/models/alphafold/docker/run_docker.py:flags.DEFINE_string('gpu_devices', 'all', 'Comma separated list of devices to '
hallucination/models/alphafold/docker/run_docker.py:                    'pass to NVIDIA_VISIBLE_DEVICES.')
hallucination/models/alphafold/docker/run_docker.py:      runtime='nvidia' if FLAGS.use_gpu else None,
hallucination/models/alphafold/docker/run_docker.py:          'NVIDIA_VISIBLE_DEVICES': FLAGS.gpu_devices,
hallucination/models/alphafold/docker/run_docker.py:          # would typically be too long to fit into GPU memory.
hallucination/models/alphafold/alphafold/model/r3.py:these can end up on specialized cores such as tensor cores on GPU or the MXU on
hallucination/models/alphafold/alphafold/model/r3.py:unintended use of these cores on both GPUs and TPUs.
hallucination/models/alphafold/alphafold/model/quat_affine.py:  the GPU. If at all possible, this function should run on the CPU.
hallucination/models/rf_Nov05_2021/InitStrGenerator.py:    #@torch.cuda.amp.autocast(enabled=True)
hallucination/models/rf_Nov05_2021/Track_module.py:    @torch.cuda.amp.autocast(enabled=False)
hallucination/models/rf_Nov05_2021/loss.py:@torch.cuda.amp.autocast(enabled=False)
hallucination/models/rf_Nov05_2021/Attention_module.py:    @torch.cuda.amp.autocast(enabled=True)
hallucination/models/rf_Nov05_2021/Attention_module.py:    @torch.cuda.amp.autocast(enabled=True)
hallucination/models/rf_Nov05_2021/Attention_module.py:    @torch.cuda.amp.autocast(enabled=True)
hallucination/models/rf_Nov05_2021/Attention_module.py:    @torch.cuda.amp.autocast(enabled=True)
hallucination/tests/README.md:   qlogin --mem 12g --gres gpu:rtx2080:1   
hallucination/optimization.py:    print(f'Max CUDA memory: {torch.cuda.max_memory_allocated()/1e9:.4f}G')
hallucination/optimization.py:    torch.cuda.reset_peak_memory_stats()
hallucination/optimization.py:    # hack for nvidia SE3 RF models: 1st forward pass gives error, but later ones work
hallucination/optimization.py:    print(f'Max CUDA memory: {torch.cuda.max_memory_allocated()/1e9:.4f}G')
hallucination/optimization.py:    torch.cuda.reset_peak_memory_stats()
hallucination/loss.py:    device = torch.device('cuda')
hallucination/hallucinate.py:#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
hallucination/hallucinate.py:    p.add_argument('--device', type=str, default= 'cuda:0', help='Device to run hallucination on cpu or cuda:0; TODO get to work with CPU')
hallucination/hallucinate.py:    device = args.device #'cuda:0'
hallucination/hallucinate.py:    # output device, for split-gpu models
hallucination/hallucinate.py:    print(f'\nUsing CUDA device(s): ',end='')
hallucination/hallucinate.py:        print(f' {i}: ({torch.cuda.get_device_name(i)}); ',end='')
hallucination/hallucinate.py:        B = 1 #just one batch per design due to GPU memory
README.md:    conda env create -f SE3-nvidia.yml
README.md:A Docker image for running RFDesign on a GPU can be built and run as follows:
README.md:    nvidia-docker run -it rfdesign/rfdesign:latest /root/miniconda3/envs/rfdesign-cuda/bin/python RFDesign/hallucination/hallucinate.py --help
README.md: - cudatoolkit 11.3.1
README.md: - [SE3 Transformer implementation from NVIDIA](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer)
scripts/af2_metrics.py:# Usage (on a GPU node):
scripts/README.md:design model and the template structure, run (on GPU node):
scripts/README.md:If you're on the head node, you can submit a GPU job for AF2 metrics using:
scripts/README.md:    sbatch -p gpu --mem 12g --gres=gpu:rtx2080:1 --wrap="af2_metrics.py FOLDER/trf_relax"
docker/Dockerfile:FROM --platform=linux/x86_64 nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
docker/Dockerfile:RUN conda create -n rfdesign-cuda \
docker/Dockerfile:    dgl-cuda11.3 \
docker/Dockerfile:RUN /root/miniconda3/envs/rfdesign-cuda/bin/pip install \
docker/Dockerfile:    && /root/miniconda3/envs/rfdesign-cuda/bin/pip install \
docker/Dockerfile:        "jaxlib[cuda]==0.1.69" \
docker/Dockerfile:        "jax[cuda]==0.2.14" \
docker/Dockerfile:        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
docker/Dockerfile:    && /root/miniconda3/envs/rfdesign-cuda/bin/pip install \
docker/Dockerfile:        tensorflow-gpu==2.9.0 \

```
