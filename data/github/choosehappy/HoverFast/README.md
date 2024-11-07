# https://github.com/choosehappy/HoverFast

```console
Dockerfile:# Use NVIDIA's CUDA base image
Dockerfile:FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
docs/source/training.rst:    docker run --gpus all -it -v /path/to/dir/:/HoverFastData petroslk/data_generation_hovernet:latest hoverfast_data_generation -c '/HoverFastData/config.ini'
docs/source/training.rst:    gpu_id = 0
docs/source/training.rst:  - **gpu_id**: GPU ID to use for data generation.
docs/source/infer_roi.rst:- **-g, --batch_gpu**: Target batch size for GPU. Default is calculated based on available GPU VRAM.
docs/source/infer_roi.rst:Custom GPU Batch Sizes
docs/source/infer_roi.rst:Setting custom sizes for GPU batches to optimize VRAM consumption. By default, HoverFast tries to maximize VRAM usage, but sometimes, using less VRAM can be useful.
docs/source/infer_roi.rst:- GPU Batch Size: GPU processes 3 batches at a time.
docs/source/infer_wsi.rst:- **-g, --batch_gpu**: Target batch size for GPU. Default is calculated based on available GPU VRAM.
docs/source/infer_wsi.rst:Custom GPU Batch Sizes
docs/source/infer_wsi.rst:Setting custom sizes GPU batches to optimize VRAM consumption. By default, HoverFast tries to maximize VRAM usage, but sometimes,
docs/source/infer_wsi.rst:- GPU Batch Size: GPU processes 50 batches at a time.
docs/source/installation.rst:- CUDA installation for GPU support (version > 12.1.0)
docs/source/installation.rst:Install NVIDIA Docker
docs/source/installation.rst:For GPU support in Docker, you also need to install NVIDIA Container Toolkit. Follow the instructions on the NVIDIA Container Toolkit GitHub page (https://github.com/NVIDIA/nvidia-container-toolkit) to install the necessary components.
docs/source/installation.rst:    docker run -it --gpus all -v /path/to/slides/:/app petroslk/hoverfast:latest HoverFast infer_wsi /app/*.svs -o /app/output/
docs/source/installation.rst:This command runs HoverFast in a Docker container with GPU support, mounting the directory `/path/to/slides/` on your host to `/app` in the container, and outputs the results to the `/app/output/` directory.
docs/source/installation.rst:This command runs HoverFast in a Singularity container with GPU support, processing WSIs located in `/path/to/wsis/` and saving the results to `/path/to/output/`.
docs/source/installation.rst:Install CUDA Toolkit
docs/source/installation.rst:If you plan to use GPU support, install the CUDA toolkit. Follow the instructions on the NVIDIA CUDA Toolkit website (https://developer.nvidia.com/cuda-downloads) to install the appropriate version for your system.
docs/source/installation.rst:CUDA Not Detected
docs/source/installation.rst:Ensure that your CUDA installation is correctly configured and that your GPU drivers are up to date. You can verify the CUDA installation by running:
docs/source/unit_testing.rst:This section contains the necessary information to run the unit tests for HoverFast. The tests are designed to ensure the functionality and reliability of the software. Since many tests require GPU support, 
docs/source/unit_testing.rst:Once your environment is set up, you can run the tests. Ensure that your GPU is available and properly configured.
docs/source/unit_testing.rst:Since most tests require GPU support, it is important to ensure that your environment is properly configured to utilize the GPU. This includes having the appropriate CUDA version installed and ensuring that your system recognizes the GPU.
docs/source/unit_testing.rst:- **Check GPU Availability**:
docs/source/unit_testing.rst:  You can verify that your GPU is available and recognized by your system using the following command:
docs/source/unit_testing.rst:     nvidia-smi
docs/source/unit_testing.rst:  This should display information about your GPU.
config.ini:gpu_id = 0
README.md:- CUDA installation for GPU support (version > 12.1.0)
README.md:docker run -it --gpus all -v /path/to/slides/:/app petroslk/hoverfast:latest HoverFast infer_wsi *svs -m /HoverFast/hoverfast_crosstissue_best_model.pth -o hoverfast_results
README.md:docker run --gpus all -it -v /path/to/dir/:/HoverFastData petroslk/data_generation_hovernet:latest hoverfast_data_generation -c '/HoverFastData/config.ini'
README.md:docker run -it --gpus all -v /path/to/pytables/:/app petroslk/hoverfast:latest HoverFast train data -o training_metrics -p /app -b 16 -n 20 -e 100
README.md:Since HoverFast utilizes GPU for almost all tasks, most tests have to be run locally using pytest.
hoverfast/training_utils.py:        device (torch.device): Device to perform computation on (GPU or CPU).
hoverfast/training_utils.py:    temp = torch.arange(-size // 2 + 1,size // 2 + 1,dtype=torch.float32,device="cuda",requires_grad=False,)
hoverfast/training_utils.py:    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
hoverfast/utils_wsi.py:    This function transfers regions to the GPU, performs nuclei detection using the model,
hoverfast/utils_wsi.py:    device (torch.device): Device to perform computation on (GPU or CPU).
hoverfast/utils_wsi.py:    # Transfer regions to GPU as a torch tensor
hoverfast/utils_wsi.py:    regions_gpu = torch.from_numpy(regions).half().to(device, memory_format=torch.channels_last)
hoverfast/utils_wsi.py:    regions_gpu = regions_gpu / 255
hoverfast/utils_wsi.py:    hed_batch = rgb_to_hed_torch(regions_gpu, device)
hoverfast/utils_wsi.py:    regions_gpu = reconstructed_rgb_batch.permute(0, 3, 1, 2)
hoverfast/utils_wsi.py:    output, maps = model(regions_gpu)
hoverfast/utils_wsi.py:    This function transfers regions to the GPU, performs nuclei detection using the model,
hoverfast/utils_wsi.py:    device (torch.device): Device to perform computation on (GPU or CPU).
hoverfast/utils_wsi.py:    # Transfer regions to GPU as a torch tensor
hoverfast/utils_wsi.py:    regions_gpu = torch.from_numpy(regions).half().to(device, memory_format=torch.channels_last)
hoverfast/utils_wsi.py:    regions_gpu = regions_gpu.permute(0, 3, 1, 2) / 255
hoverfast/utils_wsi.py:    output, maps = model(regions_gpu)
hoverfast/utils_wsi.py:def processing(batch_coords,slide_data,model,device,batch_to_gpu,n_process, stain):
hoverfast/utils_wsi.py:    device (torch.device): Device to perform computation on (GPU or CPU).
hoverfast/utils_wsi.py:    batch_to_gpu (int): Target batch size for GPU.
hoverfast/utils_wsi.py:    for regions in tqdm(divide_batch(loaded_batch,batch_to_gpu),desc="inner",leave=False, total = math.ceil(len(loaded_batch)/batch_to_gpu)):
hoverfast/utils_wsi.py:    torch.cuda.empty_cache()
hoverfast/utils_wsi.py:def infer_on_batches(batch_coords,slide_data,model,device,batch_to_gpu,n_process,stain, features_queue):
hoverfast/utils_wsi.py:    device (torch.device): Device to perform computation on (GPU or CPU).
hoverfast/utils_wsi.py:    batch_to_gpu (int): Target batch size for GPU.
hoverfast/utils_wsi.py:    arg_list1 = processing(batch_coords,slide_data,model,device,batch_to_gpu,n_process, stain)
hoverfast/utils_wsi.py:def infer_wsi(sname,sformat,fpath,mask_dir,outdir,mag,batch_on_mem,batch_to_gpu,region_size,model,device,n_process,poly_simplify_tolerance,threshold,stain,logger):
hoverfast/utils_wsi.py:    batch_to_gpu (int): Target batch size for GPU.
hoverfast/utils_wsi.py:    device (torch.device): Device to perform computation on (GPU or CPU).
hoverfast/utils_wsi.py:        size += infer_on_batches(batch_coords,slide_data,model,device,batch_to_gpu,n_process,stain,features_queue)
hoverfast/utils_wsi.py:    batch_to_gpu = args.batch_gpu
hoverfast/utils_wsi.py:    batch_on_mem = (args.batch_mem//batch_to_gpu)*batch_to_gpu
hoverfast/utils_wsi.py:    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
hoverfast/utils_wsi.py:            n_patches,n_objects = infer_wsi(sname,sformat,fpath,mask_dir,outdir,mag,batch_on_mem,batch_to_gpu,region_size,model,device,n_process,poly_simplify_tolerance,threshold,stain, logger)
hoverfast/utils_roi.py:    This function transfers regions to the GPU, performs nuclei detection using the model,
hoverfast/utils_roi.py:    device (torch.device): Device to perform computation on (GPU or CPU).
hoverfast/utils_roi.py:    # Transfer regions to GPU as a torch tensor
hoverfast/utils_roi.py:    regions_gpu = torch.from_numpy(regions).half().to(device, memory_format=torch.channels_last)
hoverfast/utils_roi.py:    regions_shape = regions_gpu.shape
hoverfast/utils_roi.py:    regions_gpu = regions_gpu / 255
hoverfast/utils_roi.py:    # Padding operations on GPU
hoverfast/utils_roi.py:    regions_gpu = torch.nn.functional.pad(regions_gpu, (0, 0, stride // 2, stride // 2, stride // 2, stride // 2), mode='reflect')
hoverfast/utils_roi.py:    hed_batch = rgb_to_hed_torch(regions_gpu, device)
hoverfast/utils_roi.py:    regions_gpu = reconstructed_rgb_batch.permute(0, 3, 1, 2)
hoverfast/utils_roi.py:    output, maps = model(regions_gpu)
hoverfast/utils_roi.py:    This function transfers regions to the GPU, applies padding, performs nuclei detection using the model,
hoverfast/utils_roi.py:    device (torch.device): Device to perform computation on (GPU or CPU).
hoverfast/utils_roi.py:    # Transfer regions to GPU as a torch tensor
hoverfast/utils_roi.py:    regions_gpu = torch.from_numpy(regions).half().to(device, memory_format=torch.channels_last)
hoverfast/utils_roi.py:    regions_shape = regions_gpu.shape
hoverfast/utils_roi.py:    # Padding operations on GPU
hoverfast/utils_roi.py:    regions_gpu = torch.nn.functional.pad(regions_gpu, (0, 0, stride // 2, stride // 2, stride // 2, stride // 2), mode='reflect')
hoverfast/utils_roi.py:    regions_gpu = regions_gpu.permute(0, 3, 1, 2) / 255
hoverfast/utils_roi.py:    output, maps = model(regions_gpu)
hoverfast/utils_roi.py:def processing_roi(regions,names,model,device,batch_to_gpu, stain):
hoverfast/utils_roi.py:    device (torch.device): Device to perform computation on (GPU or CPU).
hoverfast/utils_roi.py:    batch_to_gpu (int): Target batch size for GPU.
hoverfast/utils_roi.py:    for rgs in tqdm(divide_batch(regions,batch_to_gpu),desc="inner",leave=False, total = math.ceil(len(regions)/batch_to_gpu)):
hoverfast/utils_roi.py:    torch.cuda.empty_cache()
hoverfast/utils_roi.py:def infer_roi(spaths,n_process,outdir,threshold,poly_simplify_tolerance,color,model,device,batch_to_gpu,width, stain):
hoverfast/utils_roi.py:    device (torch.device): Device to perform computation on (GPU or CPU).
hoverfast/utils_roi.py:    batch_to_gpu (int): Target batch size for GPU.
hoverfast/utils_roi.py:    arg_list = processing_roi(regions, names,model,device,batch_to_gpu, stain)
hoverfast/utils_roi.py:    batch_to_gpu = args.batch_gpu
hoverfast/utils_roi.py:    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
hoverfast/utils_roi.py:            infer_roi(spaths,n_process,outdir,threshold,poly_simplify_tolerance,color,model,device,batch_to_gpu,width, stain)
hoverfast/utils_stain_deconv.py:    device (torch.device): Device to perform computation on (GPU or CPU).
hoverfast/utils_stain_deconv.py:    device (torch.device): Device to perform computation on (GPU or CPU).
hoverfast/utils_stain_deconv.py:    device (torch.device): Device to perform computation on (GPU or CPU).
hoverfast/main.py:    infer_wsi_parser.add_argument('-g', '--batch_gpu',
hoverfast/main.py:                        help="Target batch size for GPU: +1 in batch ~ +2GB VRAM (for pretrain model). Avoid matching or exceeding estimated GPU VRAM.",
hoverfast/main.py:                        default=int(np.round(torch.cuda.mem_get_info()[1]/1024**3))//2-1 if torch.cuda.is_available() else 1,
hoverfast/main.py:    infer_roi_parser.add_argument('-g', '--batch_gpu',
hoverfast/main.py:                        help="Target batch size for GPU: +1 in batch ~ +2GB VRAM (for pretrain model). Avoid matching or exceeding estimated GPU VRAM.",
hoverfast/main.py:                        default=int(np.round(torch.cuda.mem_get_info()[1]/1024**3))//2-1 if torch.cuda.is_available() else 1,
paper/paper.md:In computational digital pathology, accurate nuclear segmentation of Hematoxylin and Eosin (H&E) stained whole slide images (WSIs) is a critical step for many analyses and tissue characterizations. One popular deep learning-based nuclear segmentation approach, HoverNet [@graham2019hover], offers remarkably accurate results but lacks the high-throughput performance needed for clinical deployment in resource-constrained settings. Our approach, HoverFast, aims to provide fast and accurate nuclear segmentation in H&E images using knowledge distillation from HoverNet. By redesigning the tool with software engineering best practices, HoverFast introduces advanced parallel processing capabilities, efficient data handling, and optimized postprocessing. These improvements facilitate scalable high-throughput performance, making HoverFast more suitable for real-time analysis and application in resource-limited environments. Using a consumer grade Nvidia A5000 GPU, HoverFast showed a 21x speed improvement as compared to HoverNet; reducing mean analysis time for 40x WSIs from ~2 hours to 6 minutes while retaining a concordant mean Dice score of 0.91 against the original HoverNet output. Peak memory usage was also reduced 71% from 44.4GB, to 12.8GB, without requiring SSD-based caching. To ease adoption in research and clinical contexts, HoverFast aligns with best-practices in terms of (a) installation, and (b) containerization, while (c) providing outputs compatible with existing popular open-source image viewing tools such as QuPath [@bankhead2017qupath]. HoverFast has been made open-source and is available at [andrewjanowczyk.com/open-source-tools/hoverfast](andrewjanowczyk.com/open-source-tools/hoverfast).
paper/paper.md:Given the small size of nuclei, their segmentation typically takes place at 40x magnification (~0.25 microns per pixel (mpp)); the highest magnification supported by most current digital slide scanners. Working at this scale can be time-intensive for algorithms, especially on consumer grade GPUs, as WSIs are especially large, reaching up to 120,000x120,000 pixels.  While several existing tools like StarDist [@schmidt2018] [@weigert2020] and CellPose [@stringer2021cellpose] have been developed to tackle the challenge of nuclear segmentation, HoverNet[@graham2019hover] has emerged as one of the leading solutions in terms of segmentation accuracy, particularly for its application to H&E-stained tissue. 
paper/paper.md:To compare computational speed, n=4 slides from TCGA with corresponding tissue masks generated with HistoQC [@janowczyk2019histoqc] were analyzed on a machine with a 16 core Intel(R) Core(TM) i9-12900K CPU, a Nvidia A5000 GPU with 24GB of VRAM, and 128Gb of DDR5 RAM. For both HoverNet and HoverFast, the GPU batch size was set to maximize GPU memory usage. For HoverNet, a batch size of 90 was used, with 20 CPU threads for pre- and post-processing. Similarly, for HoverFast, a batch size of 11 and 20 CPU threads were used.  A mean speed improvement of 20.8x times (see **Table 1**) was demonstrated. The maximum RAM consumption was reduced by 71% with 44.4 GB for HoverNet versus 12.8 GB for HoverFast. Additionally, HoverNet required a peak of 118 GB of SSD space for its cache during run-time, while HoverFast did not appear to require any.

```
