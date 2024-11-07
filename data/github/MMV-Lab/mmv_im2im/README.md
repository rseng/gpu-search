# https://github.com/MMV-Lab/mmv_im2im

```console
tutorials/how_to_understand_config.md:    gpus: 1
tutorials/docker/mmv_im2im_docker_tutorial.md:MacOS is not very friendly for running deep learning experiments, as NVDIA GPUs are not supported in Mac, except specially hacked external GPU devices. We strongly recommend Linux (most commonly used on computing clusters or scientific computing workstations) or at least Windows. In the real world applications, for example, the image analysts at a core facility can train a model on the dedicated computing resource at the core facility, e.g., a cluster, and then send the model to users to use in their daily analysis, e.g. on users' own macbook, only doing inference, not training. So, in this demo, we focus on how to apply a trained model on new images with Docker on MacOS.
tutorials/FAQ.md:## 2. how to use multi-GPU training or half precision training to better utilize GPU(s)?
tutorials/FAQ.md:    gpus: 5
tutorials/FAQ.md:# How to select different GPUs?
tutorials/FAQ.md:By default, when you run `run_im2im --config myconfig.yaml`, all GPUs available on your machine are usable by the program. Then, if you select to use 1 GPU, then the first GPU will be used. If you want to run on a specific GPU(s), you can do `CUDA_VISIBLE_DEVICES=3 run_im2im --config myconfig.yaml` or `CUDA_VISIBLE_DEVICES=1,3,5 run_im2im --config myconfig.yaml` 
tutorials/FAQ.md:# How to automatically select which GPU to use?
tutorials/FAQ.md:    accelerator: "gpu"
README.md:Please note that the proper setup of hardware is beyond the scope of this pacakge. This package was tested with GPU/CPU on Linux/Windows and CPU on MacOS. [Special note for MacOS users: Directly pip install in MacOS may need [additional setup of xcode](https://developer.apple.com/forums/thread/673827).]
paper_configs/synthetic_3d_unsupervised_train.yaml:    accelerator: "gpu"
paper_configs/multiplex_train.yaml:    accelerator: "gpu"
paper_configs/instance_seg_3d_train_bf_finetune.yaml:    accelerator: "gpu"
paper_configs/labelfree_3d_pix2pix_train.yaml:    accelerator: "gpu"
paper_configs/unsupervised_seg_3d_train.yaml:    accelerator: "gpu"
paper_configs/instance_seg_3d_train_fbl_finetune.yaml:    accelerator: "gpu"
paper_configs/labelfree_3d_pix2pix_pretrain.yaml:    accelerator: "gpu"
paper_configs/labelfree_3d_pix2pix_finetune.yaml:    accelerator: "gpu"
paper_configs/semantic_seg_2d_train.yaml:    accelerator: "gpu"
paper_configs/synthetic_2d_unsupervised_train.yaml:    accelerator: "gpu"
paper_configs/semantic_seg_3d_train.yaml:    accelerator: "gpu"
paper_configs/instance_seg_3d_train_fluo_pretrain.yaml:    accelerator: "gpu"
paper_configs/synthetic_3d_supervised_train.yaml:    accelerator: "gpu"
paper_configs/denoising_3d_train.yaml:    accelerator: "gpu"
paper_configs/synthetic_2d_supervised_train.yaml:    accelerator: "gpu"
paper_configs/labelfree_2d_FCN_train.yaml:    accelerator: "gpu"
paper_configs/unsupervised_seg_2d_train.yaml:    accelerator: "gpu"
paper_configs/labelfree_3d_FCN_train.yaml:    accelerator: "gpu"
paper_configs/instance_seg_3d_train_fbl_pretrain.yaml:    accelerator: "gpu"
paper_configs/instance_seg_2d_train.yaml:    accelerator: "gpu"
paper_configs/instance_seg_2d_train.yaml:    # strategy: "ddp_find_unused_parameters_true"  # uncomment this when using multiple GPUs
paper_configs/instance_seg_3d_train_bf_pretrain.yaml:    accelerator: "gpu"
paper_configs/instance_seg_3d_train_fluo_finetune.yaml:    accelerator: "gpu"
paper_configs/instance_seg_3d_train_fluo_finetune.yaml:    #strategy: "ddp_find_unused_parameters_true"  # uncomment this when using multiple GPUs
paper_configs/modality_transfer_3d_train.yaml:    accelerator: "gpu"
docker/cuda/Dockerfile:FROM mambaorg/micromamba:1.4.3-bionic-cuda-11.3.1
docker/cuda/run_container.sh:--gpus all \
docker/tutorial.md:- To utilize GPU, it is also required to install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#setting-up-nvidia-container-toolkit). 
docker/tutorial.md:### 3. CUDA(Nvidia GPU)
docker/tutorial.md:bash docker/cuda/run_container.sh v0.4.0_amd64_cu113
mmv_im2im/configs/preset_train_cyclegan_2d.yaml:    accelerator: "gpu"
mmv_im2im/configs/preset_train_labelfree_2d.yaml:    accelerator: "gpu"
mmv_im2im/configs/preset_train_labelfree_3d.yaml:    accelerator: "gpu"
mmv_im2im/configs/preset_train_semanticseg_3d.yaml:    gpus: 1
mmv_im2im/configs/preset_train_modality_transfer_3d.yaml:    accelerator: "gpu"
mmv_im2im/configs/config_base.py:    # global variable that can be used to overwrite gpus in trainer
mmv_im2im/configs/config_base.py:    gpus: Union[int, List[int]] = field(default=None, is_mutable=True)
mmv_im2im/configs/config_base.py:    # check 5, if a global GPU number is set, update the value in trainer
mmv_im2im/configs/config_base.py:    if cfg.trainer.gpus is not None:
mmv_im2im/configs/config_base.py:        cfg.trainer.params["gpus"] = cfg.trainer.gpu
mmv_im2im/configs/preset_train_semanticseg_2d.yaml:    accelerator: "gpu"
mmv_im2im/configs/preset_train_pix2pixhd_2d.yaml:    accelerator: "gpu"
mmv_im2im/configs/preset_train_cyclegan_3d.yaml:    accelerator: "gpu"
mmv_im2im/configs/preset_train_pix2pixhd_3d.yaml:    accelerator: "gpu"
mmv_im2im/configs/preset_train_embedseg_3d.yaml:    accelerator: "gpu"
mmv_im2im/configs/preset_inference_pix2pixhd_2d.yaml:    accelerator: "gpu"
mmv_im2im/configs/preset_train_embedseg_2d.yaml:    accelerator: "gpu"
mmv_im2im/configs/preset_inference_labelfree_3d.yaml:    accelerator: "gpu"
mmv_im2im/models/pl_embedseg.py:                # move back to CUDA
mmv_im2im/models/pl_embedseg.py:                output = output.cuda()
mmv_im2im/proj_tester.py:            self.model.cuda()
mmv_im2im/proj_tester.py:            # add batch dimension and move to GPU
mmv_im2im/proj_tester.py:                x = torch.unsqueeze(x, dim=0).cuda()
mmv_im2im/proj_tester.py:                # the output is on GPU (see note below). So, we manually move the data
mmv_im2im/proj_tester.py:                # back to GPU
mmv_im2im/proj_tester.py:                    y_hat = y_hat.cuda()
mmv_im2im/proj_tester.py:        # Note: currently, we assume y_hat is still on gpu, because embedseg clustering
mmv_im2im/proj_tester.py:        # step is still only running on GPU (possible on CPU, need to some update on
mmv_im2im/proj_tester.py:        # tensor on GPU. If it is from mmv_im2im.post_processing, it will automatically
mmv_im2im/proj_tester.py:        # GPU tensors or ndarrays
mmv_im2im/postprocessing/embedseg_cluster.py:        self.xym = xym.cuda()
mmv_im2im/postprocessing/embedseg_cluster.py:        instance_map = torch.zeros(height, width).short().cuda()
mmv_im2im/postprocessing/embedseg_cluster.py:            unclustered = torch.ones(mask.sum()).short().cuda()
mmv_im2im/postprocessing/embedseg_cluster.py:            instance_map_masked = torch.zeros(mask.sum()).short().cuda()
mmv_im2im/postprocessing/embedseg_cluster.py:        self.xyzm = xyzm.cuda()
mmv_im2im/postprocessing/embedseg_cluster.py:        instance_map = torch.zeros(depth, height, width).short().cuda()
mmv_im2im/postprocessing/embedseg_cluster.py:            unclustered = torch.ones(mask.sum()).short().cuda()
mmv_im2im/postprocessing/embedseg_cluster.py:            instance_map_masked = torch.zeros(mask.sum()).short().cuda()
mmv_im2im/utils/embedseg_utils.py:    if instance_batch.is_cuda:
mmv_im2im/bin/run_im2im.py:        # check gpu option
mmv_im2im/bin/run_im2im.py:        # assert torch.cuda.is_available(), "GPU is not available."
mmv_im2im/bin/run_im2im.py:        # torch.cuda.set_device(torch.device("cuda:0"))

```
