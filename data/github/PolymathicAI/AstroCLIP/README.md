# https://github.com/PolymathicAI/AstroCLIP

```console
submit.sbatch:#SBATCH -p gpu
submit.sbatch:#SBATCH --gpus-per-node=4
submit.sbatch:#SBATCH --cpus-per-gpu=2
submit.sbatch:module load nccl
submit.sbatch:export NCCL_DEBUG=INFO
submit.sbatch:export CUDA_LAUNCH_BLOCKING=1.
submit.sbatch:    --trainer.devices=${SLURM_GPUS_PER_NODE} \
astroclip/modules.py:        # flash attention makes GPU go brrrrr but support is only in PyTorch >= 2.0
astroclip/modules.py:            # efficient attention using Flash Attention CUDA kernels
astroclip/models/moco_v2.py:        # gather from all gpus
astroclip/models/moco_v2.py:        num_gpus = batch_size_all // batch_size_this
astroclip/models/moco_v2.py:        idx_shuffle = torch.randperm(batch_size_all).cuda()
astroclip/models/moco_v2.py:        # broadcast to all gpus
astroclip/models/moco_v2.py:        # shuffled index for this gpu
astroclip/models/moco_v2.py:        gpu_idx = torch.distributed.get_rank()
astroclip/models/moco_v2.py:        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
astroclip/models/moco_v2.py:        # gather from all gpus
astroclip/models/moco_v2.py:        num_gpus = batch_size_all // batch_size_this
astroclip/models/moco_v2.py:        # restored index for this gpu
astroclip/models/moco_v2.py:        gpu_idx = torch.distributed.get_rank()
astroclip/models/moco_v2.py:        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
astroclip/astrodino/trainer.py:torch.backends.cuda.matmul.allow_tf32 = True
astroclip/astrodino/trainer.py:        batch_size=cfg.train.batch_size_per_gpu,
astroclip/astrodino/trainer.py:        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
astroclip/astrodino/trainer.py:            torch.cuda.synchronize()
astroclip/astrodino/trainer.py:    model = SSLMetaArch(cfg).to(torch.device("cuda"))
astroclip/astrodino/embed_legacysurvey/launch_embedding.sh:#SBATCH -p gpu
astroclip/astrodino/embed_legacysurvey/launch_embedding.sh:#SBATCH --gpus-per-node=4
astroclip/astrodino/embed_legacysurvey/launch_embedding.sh:#SBATCH --cpus-per-gpu=1
astroclip/astrodino/embed_legacysurvey/launch_embedding.sh:export  CUDA_LAUNCH_BLOCKING=1.
astroclip/astrodino/embed_legacysurvey/launch_embedding.sh:python launch_embeddings.py --dset_root $dset_root --save_root $save_root --batch_size 512 --num_gpus $SLURM_GPUS_PER_NODE
astroclip/astrodino/embed_legacysurvey/embed_legacysurvey.py:    file, save_dir, batch_size, gpu_id = args
astroclip/astrodino/embed_legacysurvey/embed_legacysurvey.py:    # Set the GPU device for this process
astroclip/astrodino/embed_legacysurvey/embed_legacysurvey.py:    torch.cuda.set_device(gpu_id)
astroclip/astrodino/embed_legacysurvey/embed_legacysurvey.py:    ).to(torch.device(f"cuda:{gpu_id}"))
astroclip/astrodino/embed_legacysurvey/embed_legacysurvey.py:                    images = torch.stack(img_batch).cuda()
astroclip/astrodino/embed_legacysurvey/embed_legacysurvey.py:    dset_root: str, save_dir: str, astrodino_dir: str, batch_size=512, num_gpus=4
astroclip/astrodino/embed_legacysurvey/embed_legacysurvey.py:    args = [(f, save_dir, batch_size, i % num_gpus) for i, f in enumerate(files)]
astroclip/astrodino/embed_legacysurvey/embed_legacysurvey.py:    with Pool(processes=num_gpus) as pool:
astroclip/astrodino/embed_legacysurvey/embed_legacysurvey.py:    parser.add_argument("--num_gpus", type=int, default=4)
astroclip/astrodino/embed_legacysurvey/embed_legacysurvey.py:    embed_legacysurvey(dset_root, save_dir, astrodino_dir, batch_size, num_gpus)
astroclip/astrodino/training.sh:#SBATCH -p gpu
astroclip/astrodino/training.sh:#SBATCH --gpus=20
astroclip/astrodino/training.sh:module load cuda
astroclip/astrodino/config.yaml:  batch_size_per_gpu: 72
astroclip/astrodino/distributed.py:        if torch.cuda.device_count() > 0:
astroclip/astrodino/distributed.py:    # Single node and GPU job (i.e. local script run)
astroclip/astrodino/distributed.py:    set_cuda_current_device: bool = True,
astroclip/astrodino/distributed.py:    allow_nccl_timeout: bool = False,
astroclip/astrodino/distributed.py:        set_cuda_current_device: If True, call torch.cuda.set_device() to set the
astroclip/astrodino/distributed.py:            current PyTorch CUDA device to the one matching the local rank.
astroclip/astrodino/distributed.py:    if set_cuda_current_device:
astroclip/astrodino/distributed.py:        torch.cuda.set_device(torch_env.local_rank)
astroclip/astrodino/distributed.py:    if allow_nccl_timeout:
astroclip/astrodino/distributed.py:        # This allows to use torch distributed timeout in a NCCL backend
astroclip/astrodino/distributed.py:        key, value = "NCCL_ASYNC_ERROR_HANDLING", "1"
astroclip/astrodino/distributed.py:    dist.init_process_group(backend="nccl")
astroclip/astrodino/utils.py:        if torch.cuda.is_available():
astroclip/astrodino/utils.py:                if torch.cuda.is_available():
astroclip/astrodino/utils.py:                            memory=torch.cuda.max_memory_allocated() / MB,
astroclip/astrodino/utils.py:        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
README.md:We train the model using 20 A100 GPUs (on 5 nodes) for 250k steps which takes roughly 46 hours.
README.md:We train the model using 4 A100 GPUs (on 1 node) for 30k steps which takes roughly 12 hours.
README.md:We train the model using 4 A100 GPUs (on 1 node) for 25k steps or until the validation loss does not increase for a fixed number of steps. This takes roughly 12 hours.
downstream_tasks/morphology_classification/embed_galaxy_zoo.py:                images = torch.cat(im_batch).cuda()
downstream_tasks/morphology_classification/embed_galaxy_zoo.py:            images = torch.cat(im_batch).cuda()
downstream_tasks/morphology_classification/morphology_utils/models.py:    mlp = MLP(embed_dim, num_classes, MLP_dim, dropout).cuda()
downstream_tasks/morphology_classification/morphology_utils/models.py:            output = mlp(data.cuda())
downstream_tasks/morphology_classification/morphology_utils/models.py:            loss = criterion(output.squeeze(), target.squeeze().cuda())
downstream_tasks/morphology_classification/morphology_utils/models.py:                output = mlp(data.cuda())
downstream_tasks/morphology_classification/morphology_utils/models.py:                loss = criterion(output.squeeze(), target.squeeze().cuda())
downstream_tasks/morphology_classification/morphology_utils/models.py:    y_pred = mlp(X_test.cuda()).detach().cpu()
downstream_tasks/similarity_search/embed_astroclip.py:                astroclip(batch_test["image"].cuda(), input_type="image")
downstream_tasks/similarity_search/embed_astroclip.py:                astroclip(batch_test["spectrum"].cuda(), input_type="spectrum")
downstream_tasks/property_estimation/property_utils/models.py:    model.cuda()
downstream_tasks/property_estimation/property_utils/models.py:            outputs = model(inputs.cuda()).squeeze()
downstream_tasks/property_estimation/property_utils/models.py:            loss = criterion(outputs, labels.cuda())
downstream_tasks/property_estimation/property_utils/models.py:        preds = model(torch.tensor(X_test, dtype=torch.float32).cuda()).cpu().numpy()
downstream_tasks/property_estimation/property_utils/models.py:        device: str = "cuda",
downstream_tasks/property_estimation/posterior_estimation.py:        model.eval(), model.to("cuda")
downstream_tasks/property_estimation/posterior_estimation.py:                X_train.append(model(X[0].to("cuda")).detach().cpu())
downstream_tasks/property_estimation/posterior_estimation.py:                X_test.append(model(X[0].to("cuda")).detach().cpu())
downstream_tasks/property_estimation/posterior_estimation.py:            ).condition(X.to("cuda"))
downstream_tasks/property_estimation/posterior_estimation.py:            train_loss = -flow_dist.log_prob(y.to("cuda")).mean()
downstream_tasks/property_estimation/posterior_estimation.py:                ).condition(X.to("cuda"))
downstream_tasks/property_estimation/posterior_estimation.py:                avg_val_loss -= flow_dist.log_prob(y.to("cuda")).mean().item()
downstream_tasks/property_estimation/posterior_estimation.py:        ).condition(X.to("cuda"))
downstream_tasks/property_estimation/posterior_estimation.py:        avg_test_loss -= flow_dist.log_prob(y.to("cuda")).mean().item()
downstream_tasks/property_estimation/posterior_estimation.py:        torch.zeros(input_dim).to("cuda"), torch.ones(input_dim).to("cuda")
downstream_tasks/property_estimation/posterior_estimation.py:        ).to("cuda")
downstream_tasks/property_estimation/embed_provabgs.py:                spectra, images = torch.cat(sp_batch).cuda(), torch.cat(im_batch).cuda()
downstream_tasks/property_estimation/embed_provabgs.py:            spectra, images = torch.cat(sp_batch).cuda(), torch.cat(im_batch).cuda()
downstream_tasks/property_estimation/embed_provabgs.py:    specformer.cuda()
downstream_tasks/property_estimation/baselines/trainer.py:def _get_predictions(model, test_loader, test_provabgs, scale, device="cuda"):
downstream_tasks/property_estimation/baselines/trainer.py:    accelerator: str = "gpu",
downstream_tasks/property_estimation/baselines/trainer.py:        device="cuda" if accelerator == "gpu" else "cpu",

```
