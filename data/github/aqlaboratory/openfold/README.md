# https://github.com/aqlaboratory/openfold

```console
setup.py:from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
setup.py:from scripts.utils import get_nvidia_cc
setup.py:extra_cuda_flags = [
setup.py:    '-U__CUDA_NO_HALF_OPERATORS__',
setup.py:    '-U__CUDA_NO_HALF_CONVERSIONS__',
setup.py:def get_cuda_bare_metal_version(cuda_dir):
setup.py:    if cuda_dir==None or torch.version.cuda==None:
setup.py:        print("CUDA is not found, cpu version is installed")
setup.py:        raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
setup.py:_, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
setup.py:compute_capability, _ = get_nvidia_cc()
setup.py:extra_cuda_flags += cc_flag
setup.py:    modules = [CUDAExtension(
setup.py:        name="attn_core_inplace_cuda",
setup.py:            "openfold/utils/kernel/csrc/softmax_cuda.cpp",
setup.py:            "openfold/utils/kernel/csrc/softmax_cuda_kernel.cu",
setup.py:                extra_cuda_flags
setup.py:        name="attn_core_inplace_cuda",
setup.py:            "openfold/utils/kernel/csrc/softmax_cuda.cpp",
setup.py:            "openfold/utils/kernel/csrc/softmax_cuda_stub.cpp",
Dockerfile:FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
Dockerfile:LABEL org.opencontainers.image.base.name="docker.io/nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04"
Dockerfile:RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
Dockerfile:RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
Dockerfile:RUN apt-get update && apt-get install -y wget libxml2 cuda-minimal-build-11-3 libcusparse-dev-11-3 libcublas-dev-11-3 libcusolver-dev-11-3 git
docs/source/Training_OpenFold.md:- GPUs configured with CUDA. Training OpenFold with CPUs only is not supported. 
docs/source/Training_OpenFold.md:    --gpus 4 \
docs/source/Training_OpenFold.md:- `num_nodes` and `gpus`:  Specifies number of nodes and GPUs available to train OpenFold.
docs/source/Training_OpenFold.md:Note that `--seed` must be specified to correctly configure training examples on multi-GPU training runs
docs/source/Training_OpenFold.md:    --gpus 4 \
docs/source/Training_OpenFold.md:    --gpus 4 \
docs/source/Single_Sequence_Inference.md:    --model_device "cuda:0" \
docs/source/Single_Sequence_Inference.md:    --model_device "cuda:0" \
docs/source/Inference.md:    --model_device "cuda:0" 
docs/source/Inference.md:- `--model_device`: Specify to use a GPU is one is available.
docs/source/Inference.md:  --model_device "cuda:0" \
docs/source/Installation.md:This package is currently supported for CUDA 11 and Pytorch 1.12. All dependencies are listed in the [`environment.yml`](https://github.com/aqlaboratory/openfold/blob/main/environment.yml)
docs/source/Installation.md:### CUDA 12
docs/source/Installation.md:To use OpenFold on CUDA 12 environment rather than a CUDA 11 environment.
docs/source/Multimer_Inference.md:    --model_device "cuda:0" \
docs/source/FAQ.md:- I see a CUDA mismatch error, eg. 
docs/source/FAQ.md:The detected CUDA version (11.8) mismatches the version that was used to compile
docs/source/FAQ.md:PyTorch (12.1). Please make sure to use the same CUDA versions.
docs/source/FAQ.md: > 	Solution: Ensure that your system's CUDA driver and toolkit match your intended OpenFold installation (CUDA 11 by default).  You can check the CUDA driver version with a command such as `nvidia-smi`
docs/source/FAQ.md:- I get some error involving `fatal error: cuda_runtime.h: No such file or directory` and or `ninja: build stopped: subcommand failed.`. 
docs/source/original_readme.md:- **Faster inference** on GPU, sometimes by as much as 2x. The greatest speedups are achieved on Ampere or higher architecture GPUs.
docs/source/original_readme.md:- **Custom CUDA attention kernels** modified from [FastFold](https://github.com/hpcaitech/FastFold)'s 
docs/source/original_readme.md:4x and 5x less GPU memory than equivalent FastFold and stock PyTorch 
docs/source/original_readme.md:This package is currently supported for CUDA 11 and Pytorch 1.12
docs/source/original_readme.md:    --model_device "cuda:0" \
docs/source/original_readme.md:    --model_device "cuda:0" \
docs/source/original_readme.md:    --model_device "cuda:0" \
docs/source/original_readme.md:    --model_device "cuda:0" \
docs/source/original_readme.md:    --gpus 8 --replace_sampler_ddp=True \
docs/source/original_readme.md:    --seed 4242022 \ # in multi-gpu settings, the seed must be specified
docs/source/original_readme.md:Before running the docker container, you can verify that your docker installation is able to properly communicate with your GPU by running the following command:
docs/source/original_readme.md:docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
docs/source/original_readme.md:Note the `--gpus all` option passed to `docker run`. This option is necessary in order for the container to use the GPUs on the host machine.
docs/source/original_readme.md:--gpus all \
docs/source/original_readme.md:--model_device cuda:0 \
docs/source/index.md:- **Faster inference** on GPU, sometimes by as much as 2x. The greatest speedups are achieved on Ampere or higher architecture GPUs.
docs/source/index.md:- **Custom CUDA attention kernels** modified from [FastFold](https://github.com/hpcaitech/FastFold)'s 
docs/source/index.md:4x and 5x less GPU memory than equivalent FastFold and stock PyTorch 
tests/test_triangular_multiplicative_update.py:            torch.as_tensor(pair_act, dtype=torch.float32).cuda(),
tests/test_triangular_multiplicative_update.py:            mask=torch.as_tensor(pair_mask, dtype=torch.float32).cuda(),
tests/test_triangular_multiplicative_update.py:            torch.as_tensor(pair_act, dtype=torch.float32).cuda(),
tests/test_triangular_multiplicative_update.py:            mask=torch.as_tensor(pair_mask, dtype=torch.float32).cuda(),
tests/test_triangular_multiplicative_update.py:            torch.as_tensor(pair_act, dtype=torch.float32).cuda(),
tests/test_triangular_multiplicative_update.py:            mask=torch.as_tensor(pair_mask, dtype=torch.float32).cuda(),
tests/test_outer_product_mean.py:                torch.as_tensor(msa_act).cuda(),
tests/test_outer_product_mean.py:                mask=torch.as_tensor(msa_mask).cuda(),
tests/test_kernels.py:        q = torch.rand([n_seq, h, n_res, c], dtype=dtype).cuda()
tests/test_kernels.py:        k = torch.rand([n_seq, h, n_res, c], dtype=dtype).cuda()
tests/test_kernels.py:        v = torch.rand([n_seq, h, n_res, c], dtype=dtype).cuda()
tests/test_kernels.py:        mask = torch.randint(0, 2, [n_seq, n_res]).cuda()
tests/test_kernels.py:        ).cuda()
tests/test_kernels.py:        ).cuda()
tests/test_kernels.py:        ).cuda()
tests/test_kernels.py:        mask = torch.randint(0, 2, [n_seq, n_res]).cuda()
tests/test_data/alignments/2q2k_B/hmm_output.sto:#=GS 4gpu_A/1-23     DE [subseq from] mol:protein length:423  KLLA0E02245p
tests/test_data/alignments/2q2k_B/hmm_output.sto:4gpu_A/1-23             MG.S.S..H.H..H....H.HH.........S....S.........G......L.......V.....P...rG.......S......H........M.....T....T.....-.........-.....-....-..-...-....-...-..-.-....-...-..-.......-......-..--.--....-....-
tests/test_data/alignments/2q2k_B/hmm_output.sto:#=GR 4gpu_A/1-23     PP 9*.*.*..*.*..*....*.**.........*....*.........*......*.......*.....*...**.......*......*........*.....8....6............................................................................................
tests/test_data/alignments/2q2k_B/hmm_output.sto:4gpu_A/1-23             .-.--..--...---.-...------------------
tests/test_data/alignments/2q2k_B/hmm_output.sto:#=GR 4gpu_A/1-23     PP ......................................
tests/test_data/alignments/2q2k_A/hmm_output.sto:#=GS 4gpu_A/1-23     DE [subseq from] mol:protein length:423  KLLA0E02245p
tests/test_data/alignments/2q2k_A/hmm_output.sto:4gpu_A/1-23             MG.S.S..H.H..H....H.HH.........S....S.........G......L.......V.....P...rG.......S......H........M.....T....T.....-.........-.....-....-..-...-....-...-..-.-....-...-..-.......-......-..--.--....-....-
tests/test_data/alignments/2q2k_A/hmm_output.sto:#=GR 4gpu_A/1-23     PP 9*.*.*..*.*..*....*.**.........*....*.........*......*.......*.....*...**.......*......*........*.....8....6............................................................................................
tests/test_data/alignments/2q2k_A/hmm_output.sto:4gpu_A/1-23             .-.--..--...---.-...------------------
tests/test_data/alignments/2q2k_A/hmm_output.sto:#=GR 4gpu_A/1-23     PP ......................................
tests/compare_utils.py:# Give JAX some GPU memory discipline
tests/compare_utils.py:# (by default it hogs 90% of GPU memory. This disables that behavior and also
tests/compare_utils.py:os.environ["JAX_PLATFORM_NAME"] = "gpu"
tests/compare_utils.py:        _model = _model.cuda()
tests/test_evoformer.py:            torch.as_tensor(activations["msa"]).cuda(),
tests/test_evoformer.py:            torch.as_tensor(activations["pair"]).cuda(),
tests/test_evoformer.py:            torch.as_tensor(masks["msa"]).cuda(),
tests/test_evoformer.py:            torch.as_tensor(masks["pair"]).cuda(),
tests/test_evoformer.py:            torch.as_tensor(activations["msa"]).cuda(),
tests/test_evoformer.py:            torch.as_tensor(activations["pair"]).cuda(),
tests/test_evoformer.py:            torch.as_tensor(masks["msa"]).cuda(),
tests/test_evoformer.py:            torch.as_tensor(masks["pair"]).cuda(),
tests/test_evoformer.py:        ).eval().cuda()
tests/test_evoformer.py:        m = torch.rand((batch_size, s_t, n_res, c_m), device="cuda")
tests/test_evoformer.py:        z = torch.rand((batch_size, n_res, n_res, c_z), device="cuda")
tests/test_evoformer.py:            device="cuda",
tests/test_evoformer.py:            device="cuda",
tests/test_evoformer.py:                torch.as_tensor(msa_act, dtype=torch.float32).cuda(),
tests/test_evoformer.py:                mask=torch.as_tensor(msa_mask, dtype=torch.float32).cuda(),
tests/test_template.py:            torch.as_tensor(pair_act).unsqueeze(-4).cuda(),
tests/test_template.py:            torch.as_tensor(pair_mask).unsqueeze(-3).cuda(),
tests/test_template.py:        template_feats = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
tests/test_template.py:                torch.as_tensor(pair_act).cuda(),
tests/test_template.py:                torch.as_tensor(pair_mask).cuda(),
tests/test_template.py:                multichain_mask_2d=torch.as_tensor(multichain_mask_2d).cuda(),
tests/test_template.py:                torch.as_tensor(pair_act).cuda(),
tests/test_template.py:                torch.as_tensor(pair_mask).cuda(),
tests/test_pair_transition.py:                torch.as_tensor(pair_act, dtype=torch.float32).cuda(),
tests/test_pair_transition.py:                mask=torch.as_tensor(pair_mask, dtype=torch.float32).cuda(),
tests/test_structure_module.py:                "single": torch.as_tensor(representations["single"]).cuda(),
tests/test_structure_module.py:                "pair": torch.as_tensor(representations["pair"]).cuda(),
tests/test_structure_module.py:            torch.as_tensor(batch["aatype"]).cuda(),
tests/test_structure_module.py:            mask=torch.as_tensor(batch["seq_mask"]).cuda(),
tests/test_structure_module.py:                torch.as_tensor(affines).float().cuda()
tests/test_structure_module.py:                torch.as_tensor(affines).float().cuda()
tests/test_structure_module.py:                torch.as_tensor(sample_act).float().cuda(),
tests/test_structure_module.py:                torch.as_tensor(sample_2d).float().cuda(),
tests/test_structure_module.py:                torch.as_tensor(sample_mask.squeeze(-1)).float().cuda(),
tests/test_model.py:        model = AlphaFold(c).cuda()
tests/test_model.py:        to_cuda_device = lambda t: t.cuda()
tests/test_model.py:        batch = tensor_tree_map(to_cuda_device, batch)
tests/test_model.py:        model.to(torch.device('cuda'))
tests/test_model.py:        to_cuda_device = lambda t: t.to(torch.device("cuda"))
tests/test_model.py:        batch = tensor_tree_map(to_cuda_device, batch)
tests/test_model.py:        batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
tests/test_loss.py:            torch.tensor(pred_pos).cuda(),
tests/test_loss.py:            torch.tensor(pred_atom_mask).cuda(),
tests/test_loss.py:            torch.tensor(residue_index).cuda(),
tests/test_loss.py:            torch.tensor(aatype).cuda(),
tests/test_loss.py:            torch.tensor(pred_pos).cuda(),
tests/test_loss.py:            torch.tensor(atom_exists).cuda(),
tests/test_loss.py:            torch.tensor(atom_radius).cuda(),
tests/test_loss.py:            torch.tensor(res_ind).cuda(),
tests/test_loss.py:            torch.tensor(asym_id).cuda() if asym_id is not None else None,
tests/test_loss.py:        batch = tree_map(lambda x: torch.tensor(x).cuda(), batch, np.ndarray)
tests/test_loss.py:            torch.tensor(pred_pos).cuda(),
tests/test_loss.py:        batch = tree_map(lambda x: torch.tensor(x).cuda(), batch, np.ndarray)
tests/test_loss.py:        atom14_pred_pos = torch.tensor(atom14_pred_pos).cuda()
tests/test_loss.py:        value = tree_map(lambda x: torch.tensor(x).cuda(), value, np.ndarray)
tests/test_loss.py:        batch = tree_map(lambda x: torch.tensor(x).cuda(), batch, np.ndarray)
tests/test_loss.py:        value = tree_map(lambda x: torch.tensor(x).cuda(), value, np.ndarray)
tests/test_loss.py:        batch = tree_map(lambda x: torch.tensor(x).cuda(), batch, np.ndarray)
tests/test_loss.py:        value = tree_map(lambda x: torch.tensor(x).cuda(), value, np.ndarray)
tests/test_loss.py:        batch = tree_map(lambda x: torch.tensor(x).cuda(), batch, np.ndarray)
tests/test_loss.py:        value = tree_map(lambda x: torch.tensor(x).cuda(), value, np.ndarray)
tests/test_loss.py:        batch = tree_map(lambda x: torch.tensor(x).cuda(), batch, np.ndarray)
tests/test_loss.py:        batch = tree_map(lambda n: torch.tensor(n).cuda(), batch, np.ndarray)
tests/test_loss.py:        atom14_pred_pos = torch.tensor(atom14_pred_pos).cuda()
tests/test_loss.py:        batch = tree_map(lambda n: torch.tensor(n).cuda(), batch, np.ndarray)
tests/test_loss.py:        atom14_pred_pos = torch.tensor(atom14_pred_pos).cuda()
tests/test_loss.py:        to_tensor = lambda t: torch.tensor(t).cuda()
tests/test_loss.py:        to_tensor = lambda t: torch.tensor(t).cuda()
tests/test_loss.py:        to_tensor = lambda t: torch.tensor(t).cuda()
tests/test_loss.py:        to_tensor = lambda n: torch.tensor(n).cuda()
tests/test_loss.py:        final_atom_positions = torch.rand(batch_size, n_res, 37, 3).cuda()
tests/test_loss.py:        to_tensor = lambda t: torch.tensor(t).cuda()
tests/test_msa.py:                torch.as_tensor(msa_act).cuda(),
tests/test_msa.py:                z=torch.as_tensor(pair_act).cuda(),
tests/test_msa.py:                mask=torch.as_tensor(msa_mask).cuda(),
tests/test_msa.py:                torch.as_tensor(msa_act).cuda(),
tests/test_msa.py:                mask=torch.as_tensor(msa_mask).cuda(),
tests/test_msa.py:                torch.as_tensor(msa_act, dtype=torch.float32).cuda(),
tests/test_msa.py:                mask=torch.as_tensor(msa_mask, dtype=torch.float32).cuda(),
tests/test_primitives.py:        ).cuda()
tests/test_deepspeed_evo_attention.py:        ).cuda()
tests/test_deepspeed_evo_attention.py:        ).cuda()
tests/test_deepspeed_evo_attention.py:            ).cuda()
tests/test_deepspeed_evo_attention.py:            "msa": torch.rand(n_seq, n_res, consts.c_m, device='cuda', dtype=dtype),
tests/test_deepspeed_evo_attention.py:            "pair": torch.rand(n_res, n_res, consts.c_z, device='cuda', dtype=dtype)
tests/test_deepspeed_evo_attention.py:            "msa": torch.randint(0, 2, (n_seq, n_res), device='cuda', dtype=dtype),
tests/test_deepspeed_evo_attention.py:            "pair": torch.randint(0, 2, (n_res, n_res), device='cuda', dtype=dtype),
tests/test_deepspeed_evo_attention.py:        with torch.cuda.amp.autocast(dtype=dtype):
tests/test_deepspeed_evo_attention.py:        batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
tests/test_deepspeed_evo_attention.py:                torch.as_tensor(pair_act).cuda(),
tests/test_deepspeed_evo_attention.py:                torch.as_tensor(pair_mask).cuda(),
tests/test_deepspeed_evo_attention.py:                torch.as_tensor(pair_act).cuda(),
tests/test_deepspeed_evo_attention.py:                torch.as_tensor(pair_mask).cuda(),
tests/test_deepspeed_evo_attention.py:        batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
tests/test_deepspeed_evo_attention.py:            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
tests/data_utils.py:    q = torch.rand(batch_size, n_seq, n, c_hidden, dtype=dtype, requires_grad=requires_grad).cuda()
tests/data_utils.py:    kv = torch.rand(batch_size, n_seq, n, c_hidden, dtype=dtype, requires_grad=requires_grad).cuda()
tests/data_utils.py:    mask = torch.randint(0, 2, (batch_size, n_seq, 1, 1, n), dtype=dtype, requires_grad=False).cuda()
tests/data_utils.py:    z_bias = torch.rand(batch_size, 1, no_heads, n, n, dtype=dtype, requires_grad=requires_grad).cuda()
tests/test_feats.py:            torch.tensor(aatype).cuda(),
tests/test_feats.py:            torch.tensor(all_atom_pos).cuda(),
tests/test_feats.py:            torch.tensor(all_atom_mask).cuda(),
tests/test_feats.py:                "aatype": torch.as_tensor(aatype).cuda(),
tests/test_feats.py:                "all_atom_positions": torch.as_tensor(all_atom_pos).cuda(),
tests/test_feats.py:                "all_atom_mask": torch.as_tensor(all_atom_mask).cuda(),
tests/test_feats.py:        to_tensor = lambda t: torch.tensor(np.array(t)).cuda()
tests/test_feats.py:            transformations.cuda(),
tests/test_feats.py:            torch.as_tensor(torsion_angles_sin_cos).cuda(),
tests/test_feats.py:            torch.as_tensor(aatype).cuda(),
tests/test_feats.py:            torch.tensor(restype_rigid_group_default_frame).cuda(),
tests/test_feats.py:            transformations.cuda(),
tests/test_feats.py:            torch.as_tensor(aatype).cuda(),
tests/test_feats.py:            torch.tensor(restype_rigid_group_default_frame).cuda(),
tests/test_feats.py:            torch.tensor(restype_atom14_to_rigid_group).cuda(),
tests/test_feats.py:            torch.tensor(restype_atom14_mask).cuda(),
tests/test_feats.py:            torch.tensor(restype_atom14_rigid_group_positions).cuda(),
tests/test_triangular_attention.py:            torch.as_tensor(pair_act, dtype=torch.float32).cuda(),
tests/test_triangular_attention.py:            mask=torch.as_tensor(pair_mask, dtype=torch.float32).cuda(),
thread_sequence.py:    # Gives a large speedup on Ampere-class GPUs
thread_sequence.py:             device name is accepted (e.g. "cpu", "cuda:0")"""
thread_sequence.py:    if(args.model_device == "cpu" and torch.cuda.is_available()):
environment.yml:      - git+https://github.com/NVIDIA/dllogger.git
scripts/deepspeed_inference_test.py:example = next(iter(dl)).to(f"cuda:{local_rank}")
scripts/deepspeed_inference_test.py:model = model.to(f"cuda:{local_rank}")
scripts/precompute_embeddings.py:        nogpu: bool = False,
scripts/precompute_embeddings.py:        self.nogpu = nogpu
scripts/precompute_embeddings.py:        if torch.cuda.is_available() and not self.nogpu:
scripts/precompute_embeddings.py:            self.model = self.model.to(device="cuda")
scripts/precompute_embeddings.py:                if torch.cuda.is_available() and not self.nogpu:
scripts/precompute_embeddings.py:                    toks = toks.to(device="cuda", non_blocking=True)
scripts/precompute_embeddings.py:        args.nogpu)
scripts/precompute_embeddings.py:        "--nogpu", action="store_true",
scripts/precompute_embeddings.py:        help="Do not use GPU"
scripts/install_third_party_dependencies.sh:git clone https://github.com/NVIDIA/cutlass --depth 1
scripts/build_deepspeed_config.py:p.add_argument("--cuda_aware", action="store_true", default=False, 
scripts/build_deepspeed_config.py:                       CUDA-Aware communication. Applies only when 
scripts/build_deepspeed_config.py:p.add_argument("--comm_backend_name", type=str, default="nccl",
scripts/build_deepspeed_config.py:                       from nccl and mpi''')
scripts/build_deepspeed_config.py:        params["cuda_aware"] = args.cuda_aware
scripts/utils.py:def get_nvidia_cc():
scripts/utils.py:    Returns a tuple containing the Compute Capability of the first GPU
scripts/utils.py:    CUDA_SUCCESS = 0
scripts/utils.py:        'libcuda.so', 
scripts/utils.py:        'libcuda.dylib', 
scripts/utils.py:        'cuda.dll',
scripts/utils.py:        '/usr/local/cuda/compat/libcuda.so', # For Docker
scripts/utils.py:            cuda = ctypes.CDLL(libname)
scripts/utils.py:    nGpus = ctypes.c_int()
scripts/utils.py:    result = cuda.cuInit(0)
scripts/utils.py:    if result != CUDA_SUCCESS:
scripts/utils.py:        cuda.cuGetErrorString(result, ctypes.byref(error_str))
scripts/utils.py:    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
scripts/utils.py:    if result != CUDA_SUCCESS:
scripts/utils.py:        cuda.cuGetErrorString(result, ctypes.byref(error_str))
scripts/utils.py:    if nGpus.value < 1:
scripts/utils.py:        return None, "No GPUs detected"
scripts/utils.py:    result = cuda.cuDeviceGet(ctypes.byref(device), 0)
scripts/utils.py:    if result != CUDA_SUCCESS:
scripts/utils.py:        cuda.cuGetErrorString(result, ctypes.byref(error_str))
scripts/utils.py:    if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) != CUDA_SUCCESS:
scripts/run_unit_tests.sh:CUDA_VISIBLE_DEVICES="0"
train_openfold.py:        global_batch_size = args.num_nodes * args.gpus
train_openfold.py:    elif (args.gpus is not None and args.gpus > 1) or args.num_nodes > 1:
train_openfold.py:        "--gpus", type=int, default=1, help='For determining optimal strategy and effective batch size.'
train_openfold.py:        ((args.gpus is not None and args.gpus > 1) or 
openfold/model/outer_product_mean.py:            with torch.cuda.amp.autocast(enabled=False):
openfold/model/primitives.py:            with torch.cuda.amp.autocast(enabled=False):
openfold/model/primitives.py:            with torch.cuda.amp.autocast(enabled=False):
openfold/model/primitives.py:            with torch.cuda.amp.autocast(enabled=False):
openfold/model/primitives.py:        with torch.cuda.amp.autocast(enabled=False):
openfold/model/evoformer.py:            # m: GPU, z: CPU
openfold/model/evoformer.py:            # m: GPU, z: GPU
openfold/model/evoformer.py:            # m: GPU, z: CPU
openfold/model/evoformer.py:            torch.cuda.empty_cache()
openfold/model/evoformer.py:            # m: CPU, z: GPU
openfold/model/evoformer.py:            # m: GPU, z: GPU
openfold/model/evoformer.py:                # m: GPU, z: CPU
openfold/model/evoformer.py:                torch.cuda.empty_cache()
openfold/model/evoformer.py:                # m: CPU, z: GPU
openfold/model/evoformer.py:                # m: GPU, z: GPU
openfold/model/evoformer.py:                Whether to clear CUDA's GPU memory cache between blocks of the
openfold/model/evoformer.py:                torch.cuda.empty_cache()
openfold/model/evoformer.py:            torch.cuda.empty_cache()
openfold/model/template.py:            tensor is brought back into GPU memory. In dire straits, can be
openfold/model/template.py:    with GPU memory than the original. Useful for long-sequence inference.
openfold/model/triangular_multiplicative_update.py:            with torch.cuda.amp.autocast(enabled=False):
openfold/model/triangular_multiplicative_update.py:            with torch.cuda.amp.autocast(enabled=False):
openfold/model/heads.py:            with torch.cuda.amp.autocast(enabled=False):
openfold/model/structure_module.py:attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")
openfold/model/structure_module.py:            with torch.cuda.amp.autocast(enabled=False):
openfold/model/structure_module.py:            attn_core_inplace_cuda.forward_(
openfold/np/relax/relax.py:        use_gpu: bool,
openfold/np/relax/relax.py:          use_gpu: Whether to run on GPU
openfold/np/relax/relax.py:        self._use_gpu = use_gpu
openfold/np/relax/relax.py:            use_gpu=self._use_gpu,
openfold/np/relax/amber_minimize.py:    use_gpu: bool,
openfold/np/relax/amber_minimize.py:    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")
openfold/np/relax/amber_minimize.py:    use_gpu: bool,
openfold/np/relax/amber_minimize.py:      use_gpu: Whether to run relaxation on GPU
openfold/np/relax/amber_minimize.py:                use_gpu=use_gpu,
openfold/np/relax/amber_minimize.py:    use_gpu: bool,
openfold/np/relax/amber_minimize.py:      use_gpu: Whether to run on GPU
openfold/np/relax/amber_minimize.py:            use_gpu=use_gpu,
openfold/utils/script_utils.py:        use_gpu=(model_device != "cpu"),
openfold/utils/script_utils.py:    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
openfold/utils/script_utils.py:    if "cuda" in model_device:
openfold/utils/script_utils.py:        os.environ["CUDA_VISIBLE_DEVICES"] = device_no
openfold/utils/script_utils.py:    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
openfold/utils/kernel/csrc/compat.h:// modified from https://github.com/NVIDIA/apex/blob/master/csrc/compat.h
openfold/utils/kernel/csrc/softmax_cuda.cpp:// modified from fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda.cpp
openfold/utils/kernel/csrc/softmax_cuda.cpp:        "Softmax forward (CUDA)"
openfold/utils/kernel/csrc/softmax_cuda.cpp:        "Softmax backward (CUDA)"
openfold/utils/kernel/csrc/softmax_cuda_kernel.cu:// modified from fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda_kernel.cu
openfold/utils/kernel/csrc/softmax_cuda_kernel.cu:#include <c10/cuda/CUDAGuard.h>
openfold/utils/kernel/csrc/softmax_cuda_kernel.cu:#include "ATen/cuda/CUDAContext.h"
openfold/utils/kernel/csrc/softmax_cuda_kernel.cu:#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
openfold/utils/kernel/csrc/softmax_cuda_kernel.cu:    CHECK_CUDA(x);     \
openfold/utils/kernel/csrc/softmax_cuda_kernel.cu:        float thread_max = -1 * CUDART_INF_F;
openfold/utils/kernel/csrc/softmax_cuda_kernel.cu:    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
openfold/utils/kernel/csrc/softmax_cuda_kernel.cu:        float thread_max = -1 * CUDART_INF_F;
openfold/utils/kernel/csrc/softmax_cuda_kernel.cu:    const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
openfold/utils/kernel/csrc/softmax_cuda_stub.cpp:// modified from fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda.cpp
openfold/utils/kernel/attention_core.py:attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")
openfold/utils/kernel/attention_core.py:        attn_core_inplace_cuda.forward_(
openfold/utils/kernel/attention_core.py:        attn_core_inplace_cuda.backward_(
openfold/utils/geometry/rigid_matrix_vector.py:    def cuda(self) -> Rigid3Array:
openfold/utils/geometry/rigid_matrix_vector.py:        return Rigid3Array.from_tensor_4x4(self.to_tensor_4x4().cuda())
openfold/utils/logger.py:# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
openfold/utils/logger.py:import torch.cuda.profiler as profiler
openfold/utils/rigid_utils.py:    def cuda(self) -> Rotation:
openfold/utils/rigid_utils.py:            Analogous to the cuda() method of torch Tensors
openfold/utils/rigid_utils.py:                A copy of the Rotation in CUDA memory
openfold/utils/rigid_utils.py:            return Rotation(rot_mats=self._rot_mats.cuda(), quats=None)
openfold/utils/rigid_utils.py:                quats=self._quats.cuda(),
openfold/utils/rigid_utils.py:    def cuda(self) -> Rigid:
openfold/utils/rigid_utils.py:            Moves the transformation object to GPU memory
openfold/utils/rigid_utils.py:                A version of the transformation on GPU
openfold/utils/rigid_utils.py:        return Rigid(self._rots.cuda(), self._trans.cuda())
openfold/utils/precision_utils.py:    fp16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
openfold/utils/chunk_utils.py:        # plateau earlier than this on all GPUs I've run the model on.
examples/monomer/alignments/6KWC_1/uniref90_hits.sto:#=GS UniRef90_A0A1C6VGH7/51-230      DE [subseq from] Endo-1,4-beta-xylanase n=1 Tax=Micromonospora yangpuensis TaxID=683228 RepID=A0A1C6VGH7_9ACTN
examples/monomer/alignments/6KWC_1/uniref90_hits.sto:#=GS UniRef90_A0A7X8V4Z1/36-164      DE [subseq from] Endo-1,4-beta-xylanase (Fragment) n=1 Tax=Clostridia bacterium TaxID=2044939 RepID=A0A7X8V4Z1_UNCCL
examples/monomer/alignments/6KWC_1/uniref90_hits.sto:#=GS UniRef90_A0A7X8V2S3/1-77        DE [subseq from] Endo-1,4-beta-xylanase n=1 Tax=Clostridia bacterium TaxID=2044939 RepID=A0A7X8V2S3_UNCCL
examples/monomer/alignments/6KWC_1/uniref90_hits.sto:#=GS UniRef90_A0A7X8V2T5/33-133      DE [subseq from] Endo-1,4-beta-xylanase (Fragment) n=1 Tax=Clostridia bacterium TaxID=2044939 RepID=A0A7X8V2T5_UNCCL
examples/monomer/inference.sh:  --model_device "cuda:0" \
run_pretrained_openfold.py:    # Gives a large speedup on Ampere-class GPUs
run_pretrained_openfold.py:             device name is accepted (e.g. "cpu", "cuda:0")"""
run_pretrained_openfold.py:        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the README for details"""
run_pretrained_openfold.py:    if args.model_device == "cpu" and torch.cuda.is_available():

```
