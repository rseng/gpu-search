# https://github.com/rmaphoh/AutoMorph

```console
M1_Retinal_Image_quality_EyePACS/test_outside.py:    # Check if CUDA is available
M1_Retinal_Image_quality_EyePACS/test_outside.py:    if torch.cuda.is_available():
M1_Retinal_Image_quality_EyePACS/test_outside.py:        logging.info("CUDA is available. Using CUDA...")
M1_Retinal_Image_quality_EyePACS/test_outside.py:        device = torch.device("cuda",args.local_rank)
M1_Retinal_Image_quality_EyePACS/test_outside.py:        logging.info("Neither CUDA nor MPS is available. Using CPU...")
M1_Retinal_Image_quality_EyePACS/test_outside.py:    #map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
M1_Retinal_Image_quality_EyePACS/test_outside.sh:CUDA_NUMBER=0
M1_Retinal_Image_quality_EyePACS/test_outside.sh:    CUDA_VISIBLE_DEVICES=${CUDA_NUMBER} python test_outside.py --e=1 --b=64 --task_name='Retinal_quality' --model=${model} --round=${n_round} --train_on_dataset='EyePACS_quality' \
M2_Artery_vein/test_outside.py:    # Check if CUDA is available
M2_Artery_vein/test_outside.py:    if torch.cuda.is_available():
M2_Artery_vein/test_outside.py:        logging.info("CUDA is available. Using CUDA...")
M2_Artery_vein/test_outside.py:        device = torch.device("cuda:0")
M2_Artery_vein/test_outside.py:        logging.info("Neither CUDA nor MPS is available. Using CPU...")
M2_Artery_vein/test_outside.sh:CUDA_VISIBLE_DEVICES=0 python test_outside.py --batch-size=8 \
README.md:2024-06-27 update: pytorch 2.3 & python 3.11 supported; Mac M2 GPU supported; CPU supported (thanks to [staskh](https://github.com/staskh))
README.md:Use the Google Colab and a free Tesla T4 gpu [Colab link click](https://colab.research.google.com/drive/13Qh9umwRM1OMRiNLyILbpq3k9h55FjNZ?usp=sharing).
M2_lwnet_disc_cup/experiments/wnet_All_three_1024_disc_cup/40/config.cfg:  "device": "cuda:0",
M2_lwnet_disc_cup/experiments/wnet_All_three_1024_disc_cup/38/config.cfg:  "device": "cuda:1",
M2_lwnet_disc_cup/experiments/wnet_All_three_1024_disc_cup/42/config.cfg:  "device": "cuda:0",
M2_lwnet_disc_cup/experiments/wnet_All_three_1024_disc_cup/32/config.cfg:  "device": "cuda:0",
M2_lwnet_disc_cup/experiments/wnet_All_three_1024_disc_cup/36/config.cfg:  "device": "cuda:0",
M2_lwnet_disc_cup/experiments/wnet_All_three_1024_disc_cup/30/config.cfg:  "device": "cuda:0",
M2_lwnet_disc_cup/experiments/wnet_All_three_1024_disc_cup/34/config.cfg:  "device": "cuda:3",
M2_lwnet_disc_cup/experiments/wnet_All_three_1024_disc_cup/28/config.cfg:  "device": "cuda:0",
M2_lwnet_disc_cup/test_outside.sh:python generate_av_results.py --config_file experiments/wnet_All_three_1024_disc_cup/30/config.cfg --im_size 512 --device cuda:0
M2_lwnet_disc_cup/generate_av_results.py:parser.add_argument('--device', type=str, default='cuda:0', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')
M2_lwnet_disc_cup/generate_av_results.py:    # Check if CUDA is available
M2_lwnet_disc_cup/generate_av_results.py:    if torch.cuda.is_available():
M2_lwnet_disc_cup/generate_av_results.py:        logging.info("CUDA is available. Using CUDA...")
M2_lwnet_disc_cup/generate_av_results.py:        device = torch.device("cuda:0")
M2_lwnet_disc_cup/generate_av_results.py:        logging.info("Neither CUDA nor MPS is available. Using CPU...")
M2_lwnet_disc_cup/utils/reproducibility.py:def set_seeds(seed_value, use_cuda):
M2_lwnet_disc_cup/utils/reproducibility.py:    if use_cuda:
M2_lwnet_disc_cup/utils/reproducibility.py:        torch.cuda.manual_seed(seed_value)
M2_lwnet_disc_cup/utils/reproducibility.py:        # torch.cuda.manual_seed_all(seed_value)  # gpu vars
M2_lwnet_disc_cup/utils/get_loaders_backup.py:    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), shuffle=True)
M2_lwnet_disc_cup/utils/get_loaders_backup.py:    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
M2_lwnet_disc_cup/utils/get_loaders.py:    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), shuffle=True)
M2_lwnet_disc_cup/utils/get_loaders.py:    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
DOCKER.md:docker run  -v <path/of/AutoMorph>:/root/AutoMorph -ti --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all yukundocker/image_automorph
LOCAL.md:4. GPU is essential -  NVIDIA (cuda) or M2 (mps).
LOCAL.md:check CUDA version with ```nvcc --version```.
LOCAL.md:For CUDA cuda_12.1.r12.1/compiler.32688072_0 run install 
LOCAL.md:conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
M2_Vessel_seg/test_outside.sh:gpu_id=0
M2_Vessel_seg/test_outside.sh:CUDA_VISIBLE_DEVICES=${gpu_id} python test_outside_integrated.py --epochs=1 \
M2_Vessel_seg/test_outside_integrated.py:    # Check if CUDA is available
M2_Vessel_seg/test_outside_integrated.py:    if torch.cuda.is_available():
M2_Vessel_seg/test_outside_integrated.py:        logging.info("CUDA is available. Using CUDA...")
M2_Vessel_seg/test_outside_integrated.py:        device = torch.device("cuda:0")
M2_Vessel_seg/test_outside_integrated.py:        logging.info("Neither CUDA nor MPS is available. Using CPU...")

```
