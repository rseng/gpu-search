# https://github.com/pytorch/vision

```console
setup.py:from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME, CUDAExtension, ROCM_HOME
setup.py:FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"
setup.py:# Note: the GPU video decoding stuff used to be called "video codec", which
setup.py:# video deocding backends in torchvision. I'm renaming this to "gpu video
setup.py:USE_GPU_VIDEO_DECODER = os.getenv("TORCHVISION_USE_VIDEO_CODEC", "1") == "1"
setup.py:IS_ROCM = (torch.version.hip is not None) and (ROCM_HOME is not None)
setup.py:BUILD_CUDA_SOURCES = (torch.cuda.is_available() and ((CUDA_HOME is not None) or IS_ROCM)) or FORCE_CUDA
setup.py:print(f"{FORCE_CUDA = }")
setup.py:print(f"{USE_GPU_VIDEO_DECODER = }")
setup.py:print(f"{IS_ROCM = }")
setup.py:print(f"{BUILD_CUDA_SOURCES = }")
setup.py:        f.write("from torchvision.extension import _check_cuda_version\n")
setup.py:        f.write("if _check_cuda_version() > 0:\n")
setup.py:        f.write("    cuda = _check_cuda_version()\n")
setup.py:    if BUILD_CUDA_SOURCES:
setup.py:        if IS_ROCM:
setup.py:            define_macros += [("WITH_CUDA", None)]
setup.py:    if IS_ROCM:
setup.py:            includes="torchvision/csrc/ops/cuda/*",
setup.py:        cuda_sources = list(CSRS_DIR.glob("ops/hip/*.hip"))
setup.py:        for header in CSRS_DIR.glob("ops/cuda/*.h"):
setup.py:        cuda_sources = list(CSRS_DIR.glob("ops/cuda/*.cu"))
setup.py:    if BUILD_CUDA_SOURCES:
setup.py:        Extension = CUDAExtension
setup.py:        sources += cuda_sources
setup.py:    if IS_ROCM:
setup.py:        sources += list(image_dir.glob("cuda/*.cpp"))
setup.py:    if USE_NVJPEG and (torch.cuda.is_available() or FORCE_CUDA):
setup.py:        nvjpeg_found = CUDA_HOME is not None and (Path(CUDA_HOME) / "include/nvjpeg.h").exists()
setup.py:            Extension = CUDAExtension
setup.py:    if USE_GPU_VIDEO_DECODER:
setup.py:        # Locating GPU video decoder headers and libraries
setup.py:        # CUDA_HOME should be set to the cuda root directory.
setup.py:            BUILD_CUDA_SOURCES
setup.py:            and CUDA_HOME is not None
setup.py:            print("Building without GPU video decoder support")
setup.py:        print("Building torchvision with GPU video decoder support")
setup.py:        gpu_decoder_path = os.path.join(CSRS_DIR, "io", "decoder", "gpu")
setup.py:        gpu_decoder_src = glob.glob(os.path.join(gpu_decoder_path, "*.cpp"))
setup.py:        cuda_libs = os.path.join(CUDA_HOME, "lib64")
setup.py:        cuda_inc = os.path.join(CUDA_HOME, "include")
setup.py:            CUDAExtension(
setup.py:                "torchvision.gpu_decoder",
setup.py:                gpu_decoder_src,
setup.py:                include_dirs=[CSRS_DIR] + TORCHVISION_INCLUDE + [gpu_decoder_path] + [cuda_inc] + ffmpeg_include_dir,
setup.py:                library_dirs=ffmpeg_library_dir + TORCHVISION_LIBRARY + [cuda_libs],
setup.py:                    "cuda",
setup.py:                    "cudart",
references/depth/stereo/parsing.py:        gpu_transforms=args.gpu_transforms,
references/depth/stereo/cascade_evaluation.py:    parser.add_argument("--device", type=str, default="cuda", help="device to use for training")
references/depth/stereo/cascade_evaluation.py:    with torch.cuda.amp.autocast(enabled=args.mixed_precision, dtype=torch.float16):
references/depth/stereo/cascade_evaluation.py:            raise ValueError("The device must be cuda if we want to run in distributed mode using torchrun")
references/depth/stereo/cascade_evaluation.py:        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
references/depth/stereo/README.md:Both used 8 A100 GPUs and a batch size of 2 (so effective batch size is 16). The
references/depth/stereo/README.md:This should give an **mae of about 1.416** on the train set of `Middlebury2014`. Results may vary slightly depending on the batch size and the number of GPUs. For the most accurate results use 1 GPU and `--batch-size 1`. The created log file should look like this, where the first key is the number of cascades and the nested key is the number of recursive iterations:
references/depth/stereo/transforms.py:class ToGPU(torch.nn.Module):
references/depth/stereo/transforms.py:        dev_images = tuple(image.cuda() for image in images)
references/depth/stereo/transforms.py:        dev_disparities = tuple(map(lambda x: x.cuda() if x is not None else None, disparities))
references/depth/stereo/transforms.py:        dev_masks = tuple(map(lambda x: x.cuda() if x is not None else None, masks))
references/depth/stereo/train.py:    with torch.cuda.amp.autocast(enabled=args.mixed_precision, dtype=torch.float16):
references/depth/stereo/train.py:            "the dataset is not divisible by the batch size. Try lowering the batch size or GPU number for more accurate results."
references/depth/stereo/train.py:        with torch.cuda.amp.autocast(enabled=args.mixed_precision, dtype=torch.float16):
references/depth/stereo/train.py:        raise ValueError("The device must be cuda if we want to run in distributed mode using torchrun")
references/depth/stereo/train.py:        model = model.to(args.gpu)
references/depth/stereo/train.py:        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
references/depth/stereo/train.py:    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
references/depth/stereo/train.py:    parser.add_argument("--batch-size", type=int, default=2, help="batch size per GPU")
references/depth/stereo/train.py:    parser.add_argument("--workers", type=int, default=4, help="number of workers per GPU")
references/depth/stereo/train.py:        help="number of CPU threads per GPU. This can be changed around to speed-up transforms if needed. This can lead to worker thread contention so use with care.",
references/depth/stereo/train.py:    parser.add_argument("--gpu-transforms", action="store_true", help="use GPU transforms")
references/depth/stereo/train.py:    parser.add_argument("--device", type=str, default="cuda", help="device to use for training")
references/depth/stereo/presets.py:        gpu_transforms: bool = False,
references/depth/stereo/presets.py:        if gpu_transforms:
references/depth/stereo/presets.py:            transforms.append(T.ToGPU())
references/depth/stereo/utils/distributed.py:        args.gpu = int(os.environ["LOCAL_RANK"])
references/depth/stereo/utils/distributed.py:        args.gpu = args.rank % torch.cuda.device_count()
references/depth/stereo/utils/distributed.py:    torch.cuda.set_device(args.gpu)
references/depth/stereo/utils/distributed.py:        backend="nccl",
references/depth/stereo/utils/distributed.py:    t = torch.tensor(val, device="cuda")
references/depth/stereo/utils/logger.py:        if torch.cuda.is_available():
references/depth/stereo/utils/logger.py:                if torch.cuda.is_available():
references/depth/stereo/utils/logger.py:                            memory=torch.cuda.max_memory_allocated() / MB,
references/depth/stereo/utils/losses.py:    # an unwanted GPU synchronization that produces a large overhead
references/optical_flow/README.md:Both used 8 A100 GPUs and a batch size of 2 (so effective batch size is 16). The
references/optical_flow/README.md:size and the number of GPUs. For the most accurate results use 1 GPU and
references/optical_flow/train.py:        raise ValueError("The device must be cuda if we want to run in distributed mode using torchrun")
references/optical_flow/train.py:    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu, Default: cuda)")
references/optical_flow/utils.py:        if torch.cuda.is_available():
references/optical_flow/utils.py:                if torch.cuda.is_available():
references/optical_flow/utils.py:                            memory=torch.cuda.max_memory_allocated() / MB,
references/optical_flow/utils.py:    elif "gpu" in args:
references/optical_flow/utils.py:        args.local_rank = args.gpu
references/optical_flow/utils.py:    torch.cuda.set_device(args.local_rank)
references/optical_flow/utils.py:        backend="nccl",
references/optical_flow/utils.py:    t = torch.tensor(val, device="cuda")
references/detection/engine.py:        with torch.cuda.amp.autocast(enabled=scaler is not None):
references/detection/engine.py:        # reduce losses over all GPUs for logging purposes
references/detection/engine.py:    # FIXME remove this and make paste_masks_in_image run on the GPU
references/detection/engine.py:        if torch.cuda.is_available():
references/detection/engine.py:            torch.cuda.synchronize()
references/detection/README.md:`--nproc_per_node=<number_of_gpus_available>`
references/detection/README.md:Except otherwise noted, all models have been trained on 8x V100 GPUs. 
references/detection/train.py:To run in a multi-gpu environment, use the distributed launcher::
references/detection/train.py:    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
references/detection/train.py:        train.py ... --world-size $NGPU
references/detection/train.py:The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
references/detection/train.py:If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.
references/detection/train.py:    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
references/detection/train.py:        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
references/detection/train.py:        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
references/detection/train.py:    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
references/detection/train.py:        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
references/detection/train.py:    scaler = torch.cuda.amp.GradScaler() if args.amp else None
references/detection/utils.py:        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
references/detection/utils.py:        if torch.cuda.is_available():
references/detection/utils.py:                if torch.cuda.is_available():
references/detection/utils.py:                            memory=torch.cuda.max_memory_allocated() / MB,
references/detection/utils.py:        args.gpu = int(os.environ["LOCAL_RANK"])
references/detection/utils.py:        args.gpu = args.rank % torch.cuda.device_count()
references/detection/utils.py:    torch.cuda.set_device(args.gpu)
references/detection/utils.py:    args.dist_backend = "nccl"
references/video_classification/README.md:### Multiple GPUs
references/video_classification/README.md:Run the training on a single node with 8 GPUs:
references/video_classification/README.md:**Note:** all our models were trained on 8 nodes with 8 V100 GPUs each for a total of 64 GPUs. Expected training time for 64 GPUs is 24 hours, depending on the storage solution.
references/video_classification/README.md:**Note 2:** hyperparameters for exact replication of our training can be found on the section below. Some hyperparameters such as learning rate must be scaled linearly in proportion to the number of GPUs. The default values assume 64 GPUs.
references/video_classification/README.md:### Single GPU 
references/video_classification/README.md:**Note:** training on a single gpu can be extremely slow. 
references/video_classification/README.md:batch size per GPU. Moreover, note that our default `--lr` is configured for 64 GPUs which is how many we used for the 
references/video_classification/README.md:We used 64 GPUs to train the architecture. 
references/video_classification/train.py:        with torch.cuda.amp.autocast(enabled=scaler is not None):
references/video_classification/train.py:    # Reduce the agg_preds and agg_targets from all gpu and show result
references/video_classification/train.py:            print("It is recommended to pre-compute the dataset cache on a single-gpu first, as it will be faster")
references/video_classification/train.py:            print("It is recommended to pre-compute the dataset cache on a single-gpu first, as it will be faster")
references/video_classification/train.py:    scaler = torch.cuda.amp.GradScaler() if args.amp else None
references/video_classification/train.py:        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
references/video_classification/train.py:    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
references/video_classification/train.py:        "-b", "--batch-size", default=24, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
references/video_classification/train.py:    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
references/video_classification/utils.py:        if torch.cuda.is_available():
references/video_classification/utils.py:                if torch.cuda.is_available():
references/video_classification/utils.py:                            memory=torch.cuda.max_memory_allocated() / MB,
references/video_classification/utils.py:        args.gpu = int(os.environ["LOCAL_RANK"])
references/video_classification/utils.py:        args.gpu = args.rank % torch.cuda.device_count()
references/video_classification/utils.py:    torch.cuda.set_device(args.gpu)
references/video_classification/utils.py:    args.dist_backend = "nccl"
references/video_classification/utils.py:    t = torch.tensor(val, device="cuda")
references/classification/README.md:Except otherwise noted, all models have been trained on 8x V100 GPUs with 
references/classification/README.md:Note that the above command corresponds to a single node with 8 GPUs. If you use
references/classification/README.md:a different number of GPUs and/or a different batch size, then the learning rate
references/classification/README.md:`torchvision` was trained on 8 nodes, each with 8 GPUs (for a total of 64 GPUs),
references/classification/README.md:Note that the above command corresponds to training on a single node with 8 GPUs.
references/classification/README.md:For generating the pre-trained weights, we trained with 4 nodes, each with 8 GPUs (for a total of 32 GPUs),
references/classification/README.md:Note that the above command corresponds to training on a single node with 8 GPUs.
references/classification/README.md:For generating the pre-trained weights, we trained with 8 nodes, each with 8 GPUs (for a total of 64 GPUs),
references/classification/README.md:Note that the above command corresponds to training on a single node with 8 GPUs.
references/classification/README.md:For generating the pre-trained weights, we trained with 2 nodes, each with 8 GPUs (for a total of 16 GPUs),
references/classification/README.md:Note that the above command corresponds to training on a single node with 8 GPUs.
references/classification/README.md:For generating the pre-trained weights, we trained with 2 nodes, each with 8 GPUs (for a total of 16 GPUs),
references/classification/README.md:Note that the above command corresponds to training on a single node with 8 GPUs.
references/classification/README.md:For generating the pre-trained weights, we trained with 8 nodes, each with 8 GPUs (for a total of 64 GPUs),
references/classification/README.md:Note that the above command corresponds to training on a single node with 8 GPUs.
references/classification/README.md:For generating the pre-trained weights, we trained with 2 nodes, each with 8 GPUs (for a total of 16 GPUs),
references/classification/README.md:Automatic Mixed Precision (AMP) training on GPU for Pytorch can be enabled with the [torch.cuda.amp](https://pytorch.org/docs/stable/amp.html?highlight=amp#module-torch.cuda.amp).
references/classification/README.md:Mixed precision training makes use of both FP32 and FP16 precisions where appropriate. FP16 operations can leverage the Tensor cores on NVIDIA GPUs (Volta, Turing or newer architectures) for improved throughput, generally without loss in model accuracy. Mixed precision training also often allows larger batch sizes. GPU automatic mixed precision training for Pytorch Vision can be enabled via the flag value `--amp=True`.
references/classification/README.md:For post training quant, device is set to CPU. For training, the device is set to CUDA.
references/classification/train.py:        with torch.cuda.amp.autocast(enabled=scaler is not None):
references/classification/train.py:    scaler = torch.cuda.amp.GradScaler() if args.amp else None
references/classification/train.py:        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
references/classification/train.py:        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
references/classification/train.py:        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
references/classification/train.py:    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
references/classification/train.py:        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
references/classification/train.py:    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
references/classification/utils.py:        if torch.cuda.is_available():
references/classification/utils.py:                if torch.cuda.is_available():
references/classification/utils.py:                            memory=torch.cuda.max_memory_allocated() / MB,
references/classification/utils.py:        args.gpu = int(os.environ["LOCAL_RANK"])
references/classification/utils.py:        args.gpu = args.rank % torch.cuda.device_count()
references/classification/utils.py:    torch.cuda.set_device(args.gpu)
references/classification/utils.py:    args.dist_backend = "nccl"
references/classification/utils.py:    t = torch.tensor(val, device="cuda")
references/classification/sampler.py:    different process (GPU).
references/classification/train_quantization.py:        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
references/classification/train_quantization.py:    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
references/classification/train_quantization.py:        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
references/similarity/train.py:    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
references/segmentation/README.md:All models have been trained on 8x V100 GPUs.
references/segmentation/README.md:`--nproc_per_node=<number_of_gpus_available>`
references/segmentation/train.py:        with torch.cuda.amp.autocast(enabled=scaler is not None):
references/segmentation/train.py:        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
references/segmentation/train.py:    scaler = torch.cuda.amp.GradScaler() if args.amp else None
references/segmentation/train.py:    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
references/segmentation/train.py:        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
references/segmentation/train.py:    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
references/segmentation/utils.py:        if torch.cuda.is_available():
references/segmentation/utils.py:                if torch.cuda.is_available():
references/segmentation/utils.py:                            memory=torch.cuda.max_memory_allocated() / MB,
references/segmentation/utils.py:        args.gpu = int(os.environ["LOCAL_RANK"])
references/segmentation/utils.py:    #     args.gpu = args.rank % torch.cuda.device_count()
references/segmentation/utils.py:    torch.cuda.set_device(args.gpu)
references/segmentation/utils.py:    args.dist_backend = "nccl"
references/segmentation/utils.py:    t = torch.tensor(val, device="cuda")
gallery/others/plot_scripted_tensor_transforms.py:device = "cuda" if torch.cuda.is_available() else "cpu"
gallery/others/plot_optical_flow.py:# If you can, run this example on a GPU, it will be a lot faster.
gallery/others/plot_optical_flow.py:device = "cuda" if torch.cuda.is_available() else "cpu"
gallery/others/plot_optical_flow.py:# this example is being rendered on a machine without a GPU, and it would take
docs/source/io.rst:decoding can also be done on CUDA GPUs.
docs/source/io.rst:encode/decode JPEGs on CUDA.
docs/source/io.rst:For encoding, JPEG (cpu and CUDA) and PNG are supported.
docs/source/models/resnet.rst:    <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
docs/source/models/swin_transformer.rst:and Resolution <https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.pdf>`__
docs/source/transforms.rst:and tensor inputs. Both CPU and CUDA tensors are supported.
test/test_utils.py:from common_utils import assert_equal, cpu_and_cuda
test/test_utils.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_utils.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_utils.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_video_gpu_decoder.py:from torchvision.io import _HAS_GPU_VIDEO_DECODER, VideoReader
test/test_video_gpu_decoder.py:@pytest.mark.skipif(_HAS_GPU_VIDEO_DECODER is False, reason="Didn't compile with support for gpu decoder")
test/test_video_gpu_decoder.py:class TestVideoGPUDecoder:
test/test_video_gpu_decoder.py:        torchvision.set_video_backend("cuda")
test/test_video_gpu_decoder.py:        torchvision.set_video_backend("cuda")
test/test_video_gpu_decoder.py:        torchvision.set_video_backend("cuda")
test/test_ops.py:from common_utils import assert_equal, cpu_and_cuda, cpu_and_cuda_and_mps, needs_cuda, needs_mps
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda_and_mps())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda_and_mps())
test/test_ops.py:    @needs_cuda
test/test_ops.py:        with torch.cuda.amp.autocast():
test/test_ops.py:            self.test_forward(torch.device("cuda"), contiguous=False, x_dtype=x_dtype, rois_dtype=rois_dtype)
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda_and_mps())
test/test_ops.py:    @needs_cuda
test/test_ops.py:        with torch.cuda.amp.autocast():
test/test_ops.py:                torch.device("cuda"),
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda_and_mps())
test/test_ops.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:            pytest.param("cuda", marks=pytest.mark.needs_cuda),
test/test_ops.py:    def test_nms_gpu(self, iou, device, dtype=torch.float64):
test/test_ops.py:        err_msg = "NMS incompatible between CPU and CUDA for IoU={}"
test/test_ops.py:        r_gpu = ops.nms(boxes.to(device), scores.to(device), iou)
test/test_ops.py:        is_eq = torch.allclose(r_cpu, r_gpu.cpu())
test/test_ops.py:            is_eq = torch.allclose(scores[r_cpu], scores[r_gpu.cpu()], rtol=tol, atol=tol)
test/test_ops.py:    @needs_cuda
test/test_ops.py:        with torch.cuda.amp.autocast():
test/test_ops.py:            self.test_nms_gpu(iou=iou, dtype=dtype, device="cuda")
test/test_ops.py:            pytest.param("cuda", marks=pytest.mark.needs_cuda),
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @needs_cuda
test/test_ops.py:    def test_compare_cpu_cuda_grads(self, contiguous):
test/test_ops.py:        # Run on CUDA only
test/test_ops.py:        # compare grads computed on CUDA with grads computed on CPU
test/test_ops.py:        for d in ["cpu", "cuda"]:
test/test_ops.py:    @needs_cuda
test/test_ops.py:        with torch.cuda.amp.autocast():
test/test_ops.py:            self.test_forward(torch.device("cuda"), contiguous=False, batch_sz=batch_sz, dtype=dtype)
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_ops.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_io.py:from common_utils import assert_equal, cpu_and_cuda
test/test_io.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:    cpu_and_cuda,
test/test_functional_tensor.py:    needs_cuda,
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@needs_cuda
test/test_functional_tensor.py:    """Make sure that _scale_channel gives the same results on CPU and GPU as
test/test_functional_tensor.py:    scaled_cuda = F_t._scale_channel(img_chan.to("cuda"))
test/test_functional_tensor.py:    assert_equal(scaled_cpu, scaled_cuda.to("cpu"))
test/test_functional_tensor.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:        # Tolerance : less than 5% (cpu), 6% (cuda) of different pixels
test/test_functional_tensor.py:        tol = 0.06 if device == "cuda" else 0.05
test/test_functional_tensor.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:    # Ignore the equivalence between scripted and regular function on float16 cuda. The pixels at
test/test_functional_tensor.py:    scripted_fn_atol = -1 if (dt == torch.float16 and device == "cuda") else 1e-8
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:    if out_fn_t.dtype == torch.uint8 and "cuda" in torch.device(device).type:
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_functional_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:    cpu_and_cuda,
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_tensor.py:            device == "cuda",
test/test_transforms_tensor.py:            torch.version.cuda == "11.3",
test/test_transforms_tensor.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_image.py:from common_utils import assert_equal, cpu_and_cuda, IN_OSS_CI, needs_cuda
test/test_image.py:@needs_cuda
test/test_image.py:def test_decode_jpegs_cuda(mode, scripted):
test/test_image.py:        futures = [executor.submit(decode_fn, encoded_images, mode, "cuda") for _ in range(num_workers)]
test/test_image.py:        for decoded_image_cuda, decoded_image_cpu in zip(decoded_images, decoded_images_cpu):
test/test_image.py:            assert decoded_image_cuda.shape == decoded_image_cpu.shape
test/test_image.py:            assert decoded_image_cuda.dtype == decoded_image_cpu.dtype == torch.uint8
test/test_image.py:            assert (decoded_image_cuda.cpu().float() - decoded_image_cpu.cpu().float()).abs().mean() < 2
test/test_image.py:@needs_cuda
test/test_image.py:def test_decode_image_cuda_raises():
test/test_image.py:    data = torch.randint(0, 127, size=(255,), device="cuda", dtype=torch.uint8)
test/test_image.py:@needs_cuda
test/test_image.py:def test_decode_jpeg_cuda_device_param():
test/test_image.py:    current_device = torch.cuda.current_device()
test/test_image.py:    current_stream = torch.cuda.current_stream()
test/test_image.py:    num_devices = torch.cuda.device_count()
test/test_image.py:    devices = ["cuda", torch.device("cuda")] + [torch.device(f"cuda:{i}") for i in range(num_devices)]
test/test_image.py:    assert current_device == torch.cuda.current_device()
test/test_image.py:    assert current_stream == torch.cuda.current_stream()
test/test_image.py:@needs_cuda
test/test_image.py:def test_decode_jpeg_cuda_errors():
test/test_image.py:        decode_jpeg(data.reshape(-1, 1), device="cuda")
test/test_image.py:        decode_jpeg(data.to("cuda"), device="cuda")
test/test_image.py:        decode_jpeg(data.to(torch.float), device="cuda")
test/test_image.py:    with pytest.raises(RuntimeError, match="Expected the device parameter to be a cuda device"):
test/test_image.py:        torch.ops.image.decode_jpegs_cuda([data], ImageReadMode.UNCHANGED.value, "cpu")
test/test_image.py:            torch.empty((100,), dtype=torch.uint8, device="cuda"),
test/test_image.py:                torch.empty((100,), dtype=torch.uint8, device="cuda"),
test/test_image.py:                torch.empty((100,), dtype=torch.uint8, device="cuda"),
test/test_image.py:                torch.empty((100,), dtype=torch.uint8, device="cuda"),
test/test_image.py:                torch.empty((100,), dtype=torch.uint8, device="cuda"),
test/test_image.py:            device="cuda",
test/test_image.py:                torch.empty((100,), dtype=torch.uint8, device="cuda"),
test/test_image.py:            device="cuda",
test/test_image.py:            device="cuda",
test/test_image.py:            device="cuda",
test/test_image.py:            device="cuda",
test/test_image.py:        decode_jpeg([], device="cuda")
test/test_image.py:@needs_cuda
test/test_image.py:def test_encode_jpeg_cuda_device_param():
test/test_image.py:    current_device = torch.cuda.current_device()
test/test_image.py:    current_stream = torch.cuda.current_stream()
test/test_image.py:    num_devices = torch.cuda.device_count()
test/test_image.py:    devices = ["cuda", torch.device("cuda")] + [torch.device(f"cuda:{i}") for i in range(num_devices)]
test/test_image.py:    assert current_device == torch.cuda.current_device()
test/test_image.py:    assert current_stream == torch.cuda.current_stream()
test/test_image.py:@needs_cuda
test/test_image.py:def test_encode_jpeg_cuda(img_path, scripted, contiguous):
test/test_image.py:        # For more detail as to why check out: https://github.com/NVIDIA/cuda-samples/issues/23#issuecomment-559283013
test/test_image.py:    encoded_jpeg_cuda_tv = encode_fn(decoded_image_tv.cuda(), quality=75)
test/test_image.py:    decoded_jpeg_cuda_tv = decode_jpeg(encoded_jpeg_cuda_tv.cpu())
test/test_image.py:    abs_mean_diff = (decoded_jpeg_cuda_tv.float() - decoded_image_tv.float()).abs().mean().item()
test/test_image.py:@pytest.mark.parametrize("device", cpu_and_cuda())
test/test_image.py:        for i, (encoded_image_cuda, decoded_image_tv) in enumerate(zip(encoded_images, decoded_images_tv_device)):
test/test_image.py:            assert torch.all(encoded_image_cuda == encoded_images_threaded[0][i])
test/test_image.py:            decoded_cuda_encoded_image = decode_jpeg(encoded_image_cuda.cpu())
test/test_image.py:            assert decoded_cuda_encoded_image.shape == decoded_image_tv.shape
test/test_image.py:            assert decoded_cuda_encoded_image.dtype == decoded_image_tv.dtype
test/test_image.py:            assert (decoded_cuda_encoded_image.cpu().float() - decoded_image_tv.cpu().float()).abs().mean() < 3
test/test_image.py:@needs_cuda
test/test_image.py:def test_single_encode_jpeg_cuda_errors():
test/test_image.py:        encode_jpeg(torch.empty((3, 100, 100), dtype=torch.float32, device="cuda"))
test/test_image.py:        encode_jpeg(torch.empty((5, 100, 100), dtype=torch.uint8, device="cuda"))
test/test_image.py:        encode_jpeg(torch.empty((1, 100, 100), dtype=torch.uint8, device="cuda"))
test/test_image.py:        encode_jpeg(torch.empty((1, 3, 100, 100), dtype=torch.uint8, device="cuda"))
test/test_image.py:        encode_jpeg(torch.empty((100, 100), dtype=torch.uint8, device="cuda"))
test/test_image.py:@needs_cuda
test/test_image.py:def test_batch_encode_jpegs_cuda_errors():
test/test_image.py:                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
test/test_image.py:                torch.empty((3, 100, 100), dtype=torch.float32, device="cuda"),
test/test_image.py:                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
test/test_image.py:                torch.empty((5, 100, 100), dtype=torch.uint8, device="cuda"),
test/test_image.py:                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
test/test_image.py:                torch.empty((1, 100, 100), dtype=torch.uint8, device="cuda"),
test/test_image.py:                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
test/test_image.py:                torch.empty((1, 3, 100, 100), dtype=torch.uint8, device="cuda"),
test/test_image.py:                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
test/test_image.py:                torch.empty((100, 100), dtype=torch.uint8, device="cuda"),
test/test_image.py:                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
test/test_image.py:        RuntimeError, match="All input tensors must be on the same CUDA device when encoding with nvjpeg"
test/test_image.py:                torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda"),
test/test_image.py:    if torch.cuda.device_count() >= 2:
test/test_image.py:            RuntimeError, match="All input tensors must be on the same CUDA device when encoding with nvjpeg"
test/test_image.py:                    torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda:0"),
test/test_image.py:                    torch.empty((3, 100, 100), dtype=torch.uint8, device="cuda:1"),
test/test_transforms_v2.py:    cpu_and_cuda,
test/test_transforms_v2.py:    needs_cuda,
test/test_transforms_v2.py:def _check_kernel_cuda_vs_cpu(kernel, input, *args, rtol, atol, **kwargs):
test/test_transforms_v2.py:    """Checks if the kernel produces closes results for inputs on GPU and CPU."""
test/test_transforms_v2.py:    if input.device.type != "cuda":
test/test_transforms_v2.py:    input_cuda = input.as_subclass(torch.Tensor)
test/test_transforms_v2.py:    input_cpu = input_cuda.to("cpu")
test/test_transforms_v2.py:        actual = kernel(input_cuda, *args, **kwargs)
test/test_transforms_v2.py:    check_cuda_vs_cpu=True,
test/test_transforms_v2.py:    if check_cuda_vs_cpu:
test/test_transforms_v2.py:        _check_kernel_cuda_vs_cpu(kernel, input, *args, **kwargs, **_to_tolerances(check_cuda_vs_cpu))
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:        # In contrast to CPU, there is no native `InterpolationMode.BICUBIC` implementation for uint8 images on CUDA.
test/test_transforms_v2.py:        check_cuda_vs_cpu_tolerances = dict(rtol=0, atol=atol / 255 if dtype.is_floating_point else atol)
test/test_transforms_v2.py:            check_cuda_vs_cpu=check_cuda_vs_cpu_tolerances,
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:            check_cuda_vs_cpu=dict(atol=1, rtol=0)
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:        if input_dtype == torch.uint8 and output_dtype == torch.uint16 and device == "cuda":
test/test_transforms_v2.py:            pytest.xfail("uint8 to uint16 conversion is not supported on cuda")
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @needs_cuda
test/test_transforms_v2.py:    def test_cpu_vs_gpu(self, T):
test/test_transforms_v2.py:        _check_kernel_cuda_vs_cpu(cutmix_mixup, imgs, labels, rtol=None, atol=None)
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:            check_cuda_vs_cpu=dtype is not torch.float16,
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_transforms_v2.py:    @needs_cuda
test/test_transforms_v2.py:    def test_transform_error_cuda(self):
test/test_transforms_v2.py:        for matrix_device, vector_device in [("cuda", "cpu"), ("cpu", "cuda")]:
test/test_transforms_v2.py:        for input_device, param_device in [("cuda", "cpu"), ("cpu", "cuda")]:
test/test_transforms_v2.py:    @pytest.mark.parametrize("device", cpu_and_cuda())
test/test_prototype_models.py:from common_utils import cpu_and_cuda, set_rng_seed
test/test_prototype_models.py:@pytest.mark.parametrize("dev", cpu_and_cuda())
test/test_prototype_models.py:@pytest.mark.parametrize("dev", cpu_and_cuda())
test/common_utils.py:CUDA_NOT_AVAILABLE_MSG = "CUDA device not available"
test/common_utils.py:OSS_CI_GPU_NO_CUDA_MSG = "We're in an OSS GPU machine, and this test doesn't need cuda."
test/common_utils.py:    if torch.cuda.is_available():
test/common_utils.py:        cuda_rng_state = torch.cuda.get_rng_state()
test/common_utils.py:    if torch.cuda.is_available():
test/common_utils.py:        torch.cuda.set_rng_state(cuda_rng_state)
test/common_utils.py:def cpu_and_cuda():
test/common_utils.py:    return ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda))
test/common_utils.py:def cpu_and_cuda_and_mps():
test/common_utils.py:    return cpu_and_cuda() + (pytest.param("mps", marks=pytest.mark.needs_mps),)
test/common_utils.py:def needs_cuda(test_func):
test/common_utils.py:    return pytest.mark.needs_cuda(test_func)
test/test_models.py:from common_utils import cpu_and_cuda, freeze_rng_state, map_nested_tensor_object, needs_cuda, set_rng_seed
test/test_models.py:    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
test/test_models.py:    "vit_h_14": {("Windows", "cpu"), ("Windows", "cuda")},
test/test_models.py:    "regnet_y_128gf": {("Windows", "cpu"), ("Windows", "cuda")},
test/test_models.py:    "mvit_v1_b": {("Windows", "cuda"), ("Linux", "cuda")},
test/test_models.py:    "mvit_v2_s": {("Windows", "cuda"), ("Linux", "cuda")},
test/test_models.py:@needs_cuda
test/test_models.py:    model.cuda()
test/test_models.py:    x = torch.rand(input_shape, device="cuda")
test/test_models.py:    with torch.cuda.amp.autocast():
test/test_models.py:@pytest.mark.parametrize("dev", cpu_and_cuda())
test/test_models.py:@pytest.mark.parametrize("dev", cpu_and_cuda())
test/test_models.py:    if dev == "cuda":
test/test_models.py:        with torch.cuda.amp.autocast():
test/test_models.py:@pytest.mark.parametrize("dev", cpu_and_cuda())
test/test_models.py:    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
test/test_models.py:    if dev == "cuda":
test/test_models.py:        with torch.cuda.amp.autocast(), torch.no_grad(), freeze_rng_state():
test/test_models.py:@pytest.mark.parametrize("dev", cpu_and_cuda())
test/test_models.py:            # as in NMSTester.test_nms_cuda to see if this is caused by duplicate
test/test_models.py:    if dev == "cuda":
test/test_models.py:        with torch.cuda.amp.autocast(), torch.no_grad(), freeze_rng_state():
test/test_models.py:@pytest.mark.parametrize("dev", cpu_and_cuda())
test/test_models.py:    # RNG always on CPU, to ensure x in cuda tests is bitwise identical to x in cpu tests
test/test_models.py:    if dev == "cuda":
test/test_models.py:        with torch.cuda.amp.autocast():
test/test_models.py:@needs_cuda
test/test_models.py:    model = model_fn(corr_block=corr_block).eval().to("cuda")
test/test_models.py:    img1 = torch.rand(bs, 3, 80, 72).cuda()
test/test_models.py:    img2 = torch.rand(bs, 3, 80, 72).cuda()
test/smoke_test.py:        model = resnet50().cuda()
test/smoke_test.py:        x = torch.randn(1, 3, 224, 224, device="cuda")
test/smoke_test.py:    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
test/smoke_test.py:    if torch.cuda.is_available():
test/smoke_test.py:        smoke_test_torchvision_decode_jpeg("cuda")
test/smoke_test.py:        smoke_test_torchvision_resnet50_classify("cuda")
test/conftest.py:    CUDA_NOT_AVAILABLE_MSG,
test/conftest.py:    OSS_CI_GPU_NO_CUDA_MSG,
test/conftest.py:    config.addinivalue_line("markers", "needs_cuda: mark for tests that rely on a CUDA device")
test/conftest.py:    # Typically, here, we try to optimize CI time. In particular, the GPU CI instances don't need to run the
test/conftest.py:    # tests that don't need CUDA, because those tests are extensively tested in the CPU CI instances already.
test/conftest.py:        # The needs_cuda mark will exist if the test was explicitly decorated with
test/conftest.py:        # the @needs_cuda decorator. It will also exist if it was parametrized with a
test/conftest.py:        # @pytest.mark.parametrize('device', cpu_and_cuda())
test/conftest.py:        # the "instances" of the tests where device == 'cuda' will have the 'needs_cuda' mark,
test/conftest.py:        needs_cuda = item.get_closest_marker("needs_cuda") is not None
test/conftest.py:        if needs_cuda and not torch.cuda.is_available():
test/conftest.py:            # In general, we skip cuda tests on machines without a GPU
test/conftest.py:            item.add_marker(pytest.mark.skip(reason=CUDA_NOT_AVAILABLE_MSG))
test/conftest.py:            if not needs_cuda and IN_RE_WORKER:
test/conftest.py:                # The RE workers are the machines with GPU, we don't want them to run CPU-only tests.
test/conftest.py:            if needs_cuda and not torch.cuda.is_available():
test/conftest.py:                # On the test machines without a GPU, we want to ignore the tests that need cuda.
test/conftest.py:            if not needs_cuda and torch.cuda.is_available():
test/conftest.py:                # Similar to what happens in RE workers: we don't need the OSS CI GPU machines
test/conftest.py:                item.add_marker(pytest.mark.skip(reason=OSS_CI_GPU_NO_CUDA_MSG))
test/conftest.py:    # the GPU test machines which don't run the CPU-only tests (see pytest_collection_modifyitems above). For
test/conftest.py:    # example `test_transforms.py` doesn't contain any CUDA test at the time of
test/conftest.py:    # writing, so on a GPU test machine, testpilot would invoke pytest on this file and no test would be run.
test/conftest.py:    if torch.cuda.is_available():
test/conftest.py:        cuda_rng_state = torch.cuda.get_rng_state()
test/conftest.py:    if torch.cuda.is_available():
test/conftest.py:        torch.cuda.set_rng_state(cuda_rng_state)
packaging/torchvision/meta.yaml:    {{ environ.get('CONDA_CUDATOOLKIT_CONSTRAINT', '') }}
packaging/torchvision/meta.yaml:    {{ environ.get('CONDA_CUDATOOLKIT_CONSTRAINT', '') }}
packaging/torchvision/meta.yaml:    - CUDA_HOME
packaging/torchvision/meta.yaml:    - FORCE_CUDA
packaging/torchvision/meta.yaml:    - TORCH_CUDA_ARCH_LIST
packaging/post_build_script.sh:LD_LIBRARY_PATH="/usr/local/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH" python packaging/wheel/relocate.py
torchvision/datasets/_stereo_matching.py:    """`FallingThings <https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation>`_ dataset.
torchvision/datasets/_stereo_matching.py:        # as per https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt
torchvision/models/quantization/shufflenetv2.py:        GPU inference is not yet supported.
torchvision/models/quantization/shufflenetv2.py:        GPU inference is not yet supported.
torchvision/models/quantization/shufflenetv2.py:        GPU inference is not yet supported.
torchvision/models/quantization/shufflenetv2.py:        GPU inference is not yet supported.
torchvision/models/quantization/inception.py:        GPU inference is not yet supported.
torchvision/models/quantization/resnet.py:        GPU inference is not yet supported.
torchvision/models/quantization/resnet.py:        GPU inference is not yet supported.
torchvision/models/quantization/resnet.py:        GPU inference is not yet supported.
torchvision/models/quantization/resnet.py:        GPU inference is not yet supported.
torchvision/models/quantization/googlenet.py:        GPU inference is not yet supported.
torchvision/models/quantization/mobilenetv3.py:        GPU inference is not yet supported.
torchvision/models/quantization/mobilenetv2.py:        GPU inference is not yet supported.
torchvision/models/resnet.py:    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
torchvision/models/resnet.py:       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
torchvision/models/resnet.py:       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
torchvision/models/resnet.py:       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
torchvision/extension.py:    # To find cuda related dlls we need to make sure the
torchvision/extension.py:def _check_cuda_version():
torchvision/extension.py:    Make sure that CUDA versions match between the pytorch install and torchvision install
torchvision/extension.py:    from torch.version import cuda as torch_version_cuda
torchvision/extension.py:    _version = torch.ops.torchvision._cuda_version()
torchvision/extension.py:    if _version != -1 and torch_version_cuda is not None:
torchvision/extension.py:        t_version = torch_version_cuda.split(".")
torchvision/extension.py:                "Detected that PyTorch and torchvision were compiled with different CUDA major versions. "
torchvision/extension.py:                f"PyTorch has CUDA Version={t_major}.{t_minor} and torchvision has "
torchvision/extension.py:                f"CUDA Version={tv_major}.{tv_minor}. "
torchvision/extension.py:_check_cuda_version()
torchvision/ops/boxes.py:    not guaranteed to be the same between CPU and GPU. This is similar
torchvision/ops/roi_align.py:    if torch.is_autocast_enabled() and tensor.is_cuda and tensor.dtype != torch.double:
torchvision/ops/roi_align.py:# It is transcribed directly off of the roi_align CUDA kernel, see
torchvision/ops/roi_align.py:# https://dev-discuss.pytorch.org/t/a-pure-python-implementation-of-roi-align-that-looks-just-like-its-cuda-kernel/1266
torchvision/ops/roi_align.py:            not _has_ops() or (torch.are_deterministic_algorithms_enabled() and (input.is_cuda or input.is_mps))
torchvision/csrc/vision.cpp:#ifdef WITH_CUDA
torchvision/csrc/vision.cpp:#include <cuda.h>
torchvision/csrc/vision.cpp:int64_t cuda_version() {
torchvision/csrc/vision.cpp:#ifdef WITH_CUDA
torchvision/csrc/vision.cpp:  return CUDA_VERSION;
torchvision/csrc/vision.cpp:  m.def("_cuda_version", &cuda_version);
torchvision/csrc/vision.h:VISION_API int64_t cuda_version();
torchvision/csrc/vision.h:extern "C" inline auto _register_ops = &cuda_version;
torchvision/csrc/ops/quantized/cpu/qnms_kernel.cpp:  TORCH_CHECK(!dets.is_cuda(), "dets must be a CPU tensor");
torchvision/csrc/ops/quantized/cpu/qnms_kernel.cpp:  TORCH_CHECK(!scores.is_cuda(), "scores must be a CPU tensor");
torchvision/csrc/ops/cpu/deform_conv2d_kernel.cpp:// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu
torchvision/csrc/ops/cpu/deform_conv2d_kernel.cpp:// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp
torchvision/csrc/ops/autocast/roi_align_kernel.cpp:                c10::DeviceType::CUDA>)));
torchvision/csrc/ops/autocast/nms_kernel.cpp:          (nms_autocast<c10::DispatchKey::Autocast, c10::DeviceType::CUDA>)));
torchvision/csrc/ops/mps/ps_roi_align_kernel.mm:  at::checkAllSameGPU(c, {input_t, rois_t});
torchvision/csrc/ops/mps/ps_roi_align_kernel.mm:      // A threadGroup is equivalent to a cuda's block.
torchvision/csrc/ops/mps/ps_roi_align_kernel.mm:  at::checkAllSameGPU(c, {grad_t, rois_t, channel_mapping_t});
torchvision/csrc/ops/mps/ps_roi_align_kernel.mm:      // A threadGroup is equivalent to a cuda's block.
torchvision/csrc/ops/mps/ps_roi_pool_kernel.mm:  at::checkAllSameGPU(c, {input_t, rois_t});
torchvision/csrc/ops/mps/ps_roi_pool_kernel.mm:      // A threadGroup is equivalent to a cuda's block.
torchvision/csrc/ops/mps/ps_roi_pool_kernel.mm:  at::checkAllSameGPU(c, {grad_t, rois_t, channel_mapping_t});
torchvision/csrc/ops/mps/ps_roi_pool_kernel.mm:      // A threadGroup is equivalent to a cuda's block.
torchvision/csrc/ops/mps/roi_align_kernel.mm:  at::checkAllSameGPU(c, {input_t, rois_t});
torchvision/csrc/ops/mps/roi_align_kernel.mm:      // A threadGroup is equivalent to a cuda's block.
torchvision/csrc/ops/mps/roi_align_kernel.mm:  at::checkAllSameGPU(c, {grad_t, rois_t});
torchvision/csrc/ops/mps/roi_align_kernel.mm:      // A threadGroup is equivalent to a cuda's block.
torchvision/csrc/ops/mps/mps_kernels.h:  // https://forums.developer.nvidia.com/t/atomicadd-float-float-atomicmul-float-float/14639
torchvision/csrc/ops/mps/mps_kernels.h:  // https://on-demand.gputechconf.com/gtc/2013/presentations/S3101-Atomic-Memory-Operations.pdf (See the last slide)
torchvision/csrc/ops/mps/nms_kernel.mm:      // A threadGroup is equivalent to a cuda's block.
torchvision/csrc/ops/mps/roi_pool_kernel.mm:  at::checkAllSameGPU(c, {input_t, rois_t});
torchvision/csrc/ops/mps/roi_pool_kernel.mm:      // A threadGroup is equivalent to a cuda's block.
torchvision/csrc/ops/mps/roi_pool_kernel.mm:  at::checkAllSameGPU(c, {grad_t, rois_t, argmax_t});
torchvision/csrc/ops/mps/roi_pool_kernel.mm:      // A threadGroup is equivalent to a cuda's block.
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:#include <ATen/cuda/CUDAContext.h>
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:#include <c10/cuda/CUDAGuard.h>
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:#include <ATen/native/cuda/KernelUtils.cuh>
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:#include "cuda_helpers.h"
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:  int64_t kMaxGridNum = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:  CUDA_1D_KERNEL_LOOP_T(index, n, index_t) {
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:  at::cuda::CUDAGuard device_guard(input.get_device());
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:  C10_CUDA_KERNEL_LAUNCH_CHECK();
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:  CUDA_1D_KERNEL_LOOP_T(index, n, int64_t) {
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:  at::cuda::CUDAGuard device_guard(columns.get_device());
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:  C10_CUDA_KERNEL_LAUNCH_CHECK();
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:  CUDA_1D_KERNEL_LOOP_T(index, n, int64_t) {
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:  at::cuda::CUDAGuard device_guard(columns.get_device());
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:  C10_CUDA_KERNEL_LAUNCH_CHECK();
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:  TORCH_CHECK(input_c.is_cuda(), "input must be a CUDA tensor");
torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu:TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
torchvision/csrc/ops/cuda/roi_align_kernel.cu:#include <ATen/cuda/CUDAContext.h>
torchvision/csrc/ops/cuda/roi_align_kernel.cu:#include <c10/cuda/CUDAGuard.h>
torchvision/csrc/ops/cuda/roi_align_kernel.cu:#include <ATen/native/cuda/KernelUtils.cuh>
torchvision/csrc/ops/cuda/roi_align_kernel.cu:#include "cuda_helpers.h"
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  CUDA_1D_KERNEL_LOOP(index, nthreads) {
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  CUDA_1D_KERNEL_LOOP(index, nthreads) {
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  } // CUDA_1D_KERNEL_LOOP
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  at::checkAllSameGPU(c, {input_t, rois_t});
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  at::cuda::CUDAGuard device_guard(input.device());
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
torchvision/csrc/ops/cuda/roi_align_kernel.cu:    AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  at::checkAllSameGPU(c, {grad_t, rois_t});
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  at::cuda::CUDAGuard device_guard(grad.device());
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
torchvision/csrc/ops/cuda/roi_align_kernel.cu:    AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/roi_align_kernel.cu:  AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/roi_align_kernel.cu:TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:#include <ATen/cuda/CUDAContext.h>
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:#include <c10/cuda/CUDAGuard.h>
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:#include <ATen/native/cuda/KernelUtils.cuh>
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:#include "cuda_helpers.h"
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  CUDA_1D_KERNEL_LOOP(index, nthreads) {
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  CUDA_1D_KERNEL_LOOP(index, nthreads) {
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  // Check if input tensors are CUDA tensors
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  at::checkAllSameGPU(c, {input_t, rois_t});
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  at::cuda::CUDAGuard device_guard(input.device());
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:    AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  cudaDeviceSynchronize();
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  // Check if input tensors are CUDA tensors
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:      channel_mapping.is_cuda(), "channel_mapping must be a CUDA tensor");
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  at::checkAllSameGPU(c, {grad_t, rois_t, channel_mapping_t});
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  at::cuda::CUDAGuard device_guard(grad.device());
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:    AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:  AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/ps_roi_align_kernel.cu:TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:#include <ATen/cuda/CUDAContext.h>
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:#include <c10/cuda/CUDAGuard.h>
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:#include <ATen/native/cuda/KernelUtils.cuh>
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:#include "cuda_helpers.h"
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  CUDA_1D_KERNEL_LOOP(index, nthreads) {
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  CUDA_1D_KERNEL_LOOP(index, nthreads) {
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  // Check if input tensors are CUDA tensors
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  at::checkAllSameGPU(c, {input_t, rois_t});
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  at::cuda::CUDAGuard device_guard(input.device());
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:    AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  // Check if input tensors are CUDA tensors
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:      channel_mapping.is_cuda(), "channel_mapping must be a CUDA tensor");
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  at::checkAllSameGPU(c, {grad_t, rois_t, channel_mapping_t});
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  at::cuda::CUDAGuard device_guard(grad.device());
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:    AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:  AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/ps_roi_pool_kernel.cu:TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
torchvision/csrc/ops/cuda/nms_kernel.cu:#include <ATen/cuda/CUDAContext.h>
torchvision/csrc/ops/cuda/nms_kernel.cu:#include <c10/cuda/CUDAGuard.h>
torchvision/csrc/ops/cuda/nms_kernel.cu:#include "cuda_helpers.h"
torchvision/csrc/ops/cuda/nms_kernel.cu:  using acc_T = at::acc_type<T, /*is_cuda=*/true>;
torchvision/csrc/ops/cuda/nms_kernel.cu:  TORCH_CHECK(dets.is_cuda(), "dets must be a CUDA tensor");
torchvision/csrc/ops/cuda/nms_kernel.cu:  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
torchvision/csrc/ops/cuda/nms_kernel.cu:  at::cuda::CUDAGuard device_guard(dets.device());
torchvision/csrc/ops/cuda/nms_kernel.cu:  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
torchvision/csrc/ops/cuda/nms_kernel.cu:  AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/nms_kernel.cu:TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:#include <ATen/cuda/CUDAContext.h>
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:#include <c10/cuda/CUDAGuard.h>
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:#include <ATen/native/cuda/KernelUtils.cuh>
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:#include "cuda_helpers.h"
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  CUDA_1D_KERNEL_LOOP(index, nthreads) {
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  CUDA_1D_KERNEL_LOOP(index, nthreads) {
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  at::checkAllSameGPU(c, {input_t, rois_t});
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  at::cuda::CUDAGuard device_guard(input.device());
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:    AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  // Check if input tensors are CUDA tensors
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  TORCH_CHECK(argmax.is_cuda(), "argmax must be a CUDA tensor");
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  at::checkAllSameGPU(c, {grad_t, rois_t, argmax_t});
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  at::cuda::CUDAGuard device_guard(grad.device());
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:    AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:  AT_CUDA_CHECK(cudaGetLastError());
torchvision/csrc/ops/cuda/roi_pool_kernel.cu:TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
torchvision/csrc/ops/cuda/cuda_helpers.h:#define CUDA_1D_KERNEL_LOOP_T(i, n, index_t)                         \
torchvision/csrc/ops/cuda/cuda_helpers.h:#define CUDA_1D_KERNEL_LOOP(i, n) CUDA_1D_KERNEL_LOOP_T(i, n, int)
torchvision/csrc/io/decoder/gpu/gpu_decoder.h:class GPUDecoder : public torch::CustomClassHolder {
torchvision/csrc/io/decoder/gpu/gpu_decoder.h:  GPUDecoder(std::string, torch::Device);
torchvision/csrc/io/decoder/gpu/gpu_decoder.h:  ~GPUDecoder();
torchvision/csrc/io/decoder/gpu/README.rst:GPU Decoder
torchvision/csrc/io/decoder/gpu/README.rst:GPU decoder depends on ffmpeg for demuxing, uses NVDECODE APIs from the nvidia-video-codec sdk and uses cuda for processing on gpu. In order to use this, please follow the following steps:
torchvision/csrc/io/decoder/gpu/README.rst:* Download the latest `nvidia-video-codec-sdk <https://developer.nvidia.com/nvidia-video-codec-sdk/download>`_
torchvision/csrc/io/decoder/gpu/README.rst:* Set CUDA_HOME environment variable to the cuda root directory.
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:#include "gpu_decoder.h"
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:#include <c10/cuda/CUDAGuard.h>
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:/* Set cuda device, create cuda context and initialise the demuxer and decoder.
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:GPUDecoder::GPUDecoder(std::string src_file, torch::Device dev)
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:  at::cuda::CUDAGuard device_guard(dev);
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:  check_for_cuda_errors(
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:GPUDecoder::~GPUDecoder() {
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:  at::cuda::CUDAGuard device_guard(device);
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:    check_for_cuda_errors(
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:torch::Tensor GPUDecoder::decode() {
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:  at::cuda::CUDAGuard device_guard(device);
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:void GPUDecoder::seek(double timestamp, bool keyframes_only) {
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:c10::Dict<std::string, c10::Dict<std::string, double>> GPUDecoder::
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:  m.class_<GPUDecoder>("GPUDecoder")
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:      .def("seek", &GPUDecoder::seek)
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:      .def("get_metadata", &GPUDecoder::get_metadata)
torchvision/csrc/io/decoder/gpu/gpu_decoder.cpp:      .def("next", &GPUDecoder::decode);
torchvision/csrc/io/decoder/gpu/demuxer.h:inline cudaVideoCodec ffmpeg_to_codec(AVCodecID id) {
torchvision/csrc/io/decoder/gpu/demuxer.h:      return cudaVideoCodec_MPEG1;
torchvision/csrc/io/decoder/gpu/demuxer.h:      return cudaVideoCodec_MPEG2;
torchvision/csrc/io/decoder/gpu/demuxer.h:      return cudaVideoCodec_MPEG4;
torchvision/csrc/io/decoder/gpu/demuxer.h:      return cudaVideoCodec_VC1;
torchvision/csrc/io/decoder/gpu/demuxer.h:      return cudaVideoCodec_H264;
torchvision/csrc/io/decoder/gpu/demuxer.h:      return cudaVideoCodec_HEVC;
torchvision/csrc/io/decoder/gpu/demuxer.h:      return cudaVideoCodec_VP8;
torchvision/csrc/io/decoder/gpu/demuxer.h:      return cudaVideoCodec_VP9;
torchvision/csrc/io/decoder/gpu/demuxer.h:      return cudaVideoCodec_JPEG;
torchvision/csrc/io/decoder/gpu/demuxer.h:      return cudaVideoCodec_AV1;
torchvision/csrc/io/decoder/gpu/demuxer.h:      return cudaVideoCodec_NumCodecs;
torchvision/csrc/io/decoder/gpu/decoder.cpp:static float chroma_height_factor(cudaVideoSurfaceFormat surface_format) {
torchvision/csrc/io/decoder/gpu/decoder.cpp:  return (surface_format == cudaVideoSurfaceFormat_YUV444 ||
torchvision/csrc/io/decoder/gpu/decoder.cpp:          surface_format == cudaVideoSurfaceFormat_YUV444_16Bit)
torchvision/csrc/io/decoder/gpu/decoder.cpp:static int chroma_plane_count(cudaVideoSurfaceFormat surface_format) {
torchvision/csrc/io/decoder/gpu/decoder.cpp:  return (surface_format == cudaVideoSurfaceFormat_YUV444 ||
torchvision/csrc/io/decoder/gpu/decoder.cpp:          surface_format == cudaVideoSurfaceFormat_YUV444_16Bit)
torchvision/csrc/io/decoder/gpu/decoder.cpp:void Decoder::init(CUcontext context, cudaVideoCodec codec) {
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(cuvidParseVideoData(parser, &pkt), __LINE__, __FILE__);
torchvision/csrc/io/decoder/gpu/decoder.cpp:        torch::TensorOptions().dtype(torch::kU8).device(torch::kCUDA);
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(cuCtxPushCurrent(cu_context), __LINE__, __FILE__);
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(cuCtxPopCurrent(nullptr), __LINE__, __FILE__);
torchvision/csrc/io/decoder/gpu/decoder.cpp:/* Process the decoded data and copy it to a cuda memory location.
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(cuCtxPushCurrent(cu_context), __LINE__, __FILE__);
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(
torchvision/csrc/io/decoder/gpu/decoder.cpp:  if (result == CUDA_SUCCESS &&
torchvision/csrc/io/decoder/gpu/decoder.cpp:  auto options = torch::TensorOptions().dtype(torch::kU8).device(torch::kCUDA);
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(cuStreamSynchronize(cuvidStream), __LINE__, __FILE__);
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(cuCtxPopCurrent(nullptr), __LINE__, __FILE__);
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(cuCtxPushCurrent(cu_context), __LINE__, __FILE__);
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(cuvidGetDecoderCaps(&decode_caps), __LINE__, __FILE__);
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(cuCtxPopCurrent(nullptr), __LINE__, __FILE__);
torchvision/csrc/io/decoder/gpu/decoder.cpp:    TORCH_CHECK(false, "Codec not supported on this GPU");
torchvision/csrc/io/decoder/gpu/decoder.cpp:        "\nResolution not supported on this GPU");
torchvision/csrc/io/decoder/gpu/decoder.cpp:        "\nMBCount not supported on this GPU");
torchvision/csrc/io/decoder/gpu/decoder.cpp:    if (decode_caps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12)) {
torchvision/csrc/io/decoder/gpu/decoder.cpp:      video_output_format = cudaVideoSurfaceFormat_NV12;
torchvision/csrc/io/decoder/gpu/decoder.cpp:        decode_caps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_P016)) {
torchvision/csrc/io/decoder/gpu/decoder.cpp:      video_output_format = cudaVideoSurfaceFormat_P016;
torchvision/csrc/io/decoder/gpu/decoder.cpp:        decode_caps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444)) {
torchvision/csrc/io/decoder/gpu/decoder.cpp:      video_output_format = cudaVideoSurfaceFormat_YUV444;
torchvision/csrc/io/decoder/gpu/decoder.cpp:        (1 << cudaVideoSurfaceFormat_YUV444_16Bit)) {
torchvision/csrc/io/decoder/gpu/decoder.cpp:      video_output_format = cudaVideoSurfaceFormat_YUV444_16Bit;
torchvision/csrc/io/decoder/gpu/decoder.cpp:    case cudaVideoChromaFormat_Monochrome:
torchvision/csrc/io/decoder/gpu/decoder.cpp:    case cudaVideoChromaFormat_420:
torchvision/csrc/io/decoder/gpu/decoder.cpp:          ? cudaVideoSurfaceFormat_P016
torchvision/csrc/io/decoder/gpu/decoder.cpp:          : cudaVideoSurfaceFormat_NV12;
torchvision/csrc/io/decoder/gpu/decoder.cpp:    case cudaVideoChromaFormat_444:
torchvision/csrc/io/decoder/gpu/decoder.cpp:          ? cudaVideoSurfaceFormat_YUV444_16Bit
torchvision/csrc/io/decoder/gpu/decoder.cpp:          : cudaVideoSurfaceFormat_YUV444;
torchvision/csrc/io/decoder/gpu/decoder.cpp:    case cudaVideoChromaFormat_422:
torchvision/csrc/io/decoder/gpu/decoder.cpp:      video_output_format = cudaVideoSurfaceFormat_NV12;
torchvision/csrc/io/decoder/gpu/decoder.cpp:  cudaVideoDeinterlaceMode deinterlace_mode = cudaVideoDeinterlaceMode_Adaptive;
torchvision/csrc/io/decoder/gpu/decoder.cpp:    deinterlace_mode = cudaVideoDeinterlaceMode_Weave;
torchvision/csrc/io/decoder/gpu/decoder.cpp:  // With PreferCUVID, JPEG is still decoded by CUDA while video is decoded
torchvision/csrc/io/decoder/gpu/decoder.cpp:  video_decode_create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
torchvision/csrc/io/decoder/gpu/decoder.cpp:  if (video_format->codec == cudaVideoCodec_AV1 &&
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(cuCtxPushCurrent(cu_context), __LINE__, __FILE__);
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(cuCtxPopCurrent(nullptr), __LINE__, __FILE__);
torchvision/csrc/io/decoder/gpu/decoder.cpp:    if (video_codec != cudaVideoCodec_VP9) {
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(cuCtxPushCurrent(cu_context), __LINE__, __FILE__);
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(
torchvision/csrc/io/decoder/gpu/decoder.cpp:  check_for_cuda_errors(cuCtxPopCurrent(nullptr), __LINE__, __FILE__);
torchvision/csrc/io/decoder/gpu/decoder.cpp:  return oper_point_info->codec == cudaVideoCodec_AV1 &&
torchvision/csrc/io/decoder/gpu/decoder.h:#include <cuda.h>
torchvision/csrc/io/decoder/gpu/decoder.h:#include <cuda_runtime_api.h>
torchvision/csrc/io/decoder/gpu/decoder.h:static auto check_for_cuda_errors =
torchvision/csrc/io/decoder/gpu/decoder.h:      if (CUDA_SUCCESS != result) {
torchvision/csrc/io/decoder/gpu/decoder.h:            CUDA_SUCCESS != cuGetErrorName(result, &error_name),
torchvision/csrc/io/decoder/gpu/decoder.h:            "CUDA error: ",
torchvision/csrc/io/decoder/gpu/decoder.h:  void init(CUcontext, cudaVideoCodec);
torchvision/csrc/io/decoder/gpu/decoder.h:  cudaVideoCodec video_codec = cudaVideoCodec_NumCodecs;
torchvision/csrc/io/decoder/gpu/decoder.h:  cudaVideoChromaFormat video_chroma_format = cudaVideoChromaFormat_420;
torchvision/csrc/io/decoder/gpu/decoder.h:  cudaVideoSurfaceFormat video_output_format = cudaVideoSurfaceFormat_NV12;
torchvision/csrc/io/image/image.h:#include "cuda/encode_decode_jpegs_cuda.h"
torchvision/csrc/io/image/image.cpp:        .op("image::decode_jpegs_cuda", &decode_jpegs_cuda)
torchvision/csrc/io/image/image.cpp:        .op("image::encode_jpegs_cuda", &encode_jpegs_cuda)
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:#include "encode_jpegs_cuda.h"
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:std::vector<torch::Tensor> encode_jpegs_cuda(
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:      false, "encode_jpegs_cuda: torchvision not compiled with nvJPEG support");
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:#include <ATen/cuda/CUDAContext.h>
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:#include <ATen/cuda/CUDAEvent.h>
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:#include <cuda_runtime.h>
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:std::unique_ptr<CUDAJpegEncoder> cudaJpegEncoder;
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:std::vector<torch::Tensor> encode_jpegs_cuda(
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:      "torchvision.csrc.io.image.cuda.encode_jpegs_cuda.encode_jpegs_cuda");
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  at::cuda::CUDAGuard device_guard(device);
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  if (cudaJpegEncoder == nullptr || device != cudaJpegEncoder->target_device) {
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:    if (cudaJpegEncoder != nullptr)
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:      delete cudaJpegEncoder.release();
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:    cudaJpegEncoder = std::make_unique<CUDAJpegEncoder>(device);
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:    // object correctly upon program exit. This is because, when cudaJpegEncoder
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:    // gets destroyed, the CUDA runtime may already be shut down, rendering all
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:    // CUDA shuts down when the program exits. If CUDA is already shut down the
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:    std::atexit([]() { delete cudaJpegEncoder.release(); });
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:        "All input tensors must be on the same CUDA device when encoding with nvjpeg")
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  cudaJpegEncoder->set_quality(quality);
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  at::cuda::CUDAEvent event;
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  event.record(cudaJpegEncoder->stream);
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:    auto encoded_image = cudaJpegEncoder->encode_jpeg(image);
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  // (which is what cudaStreamSynchronize would do.) Events allow us to
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  event.block(at::cuda::getCurrentCUDAStream(
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:      cudaJpegEncoder->original_device.has_index()
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:          ? cudaJpegEncoder->original_device.index()
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:CUDAJpegEncoder::CUDAJpegEncoder(const torch::Device& target_device)
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:    : original_device{torch::kCUDA, torch::cuda::current_device()},
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:              ? at::cuda::getStreamFromPool(false, target_device.index())
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:              : at::cuda::getStreamFromPool(false)} {
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:CUDAJpegEncoder::~CUDAJpegEncoder() {
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  destructor executes after cuda is already shut down causing SIGSEGV.
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  just leak the libnvjpeg & cuda variables for the time being and hope
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  that the CUDA runtime handles cleanup for us.
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  // // We run cudaGetDeviceCount as a dummy to test if the CUDA runtime is
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  // cudaError_t error = cudaGetDeviceCount(&deviceCount);
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  // if (error != cudaSuccess)
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  //   return; // CUDA runtime has already shut down. There's nothing we can do
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  // cudaStreamSynchronize(stream);
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:torch::Tensor CUDAJpegEncoder::encode_jpeg(const torch::Tensor& src_image) {
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  cudaError_t cudaStatus;
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  cudaStatus = cudaStreamSynchronize(stream);
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  TORCH_CHECK(cudaStatus == cudaSuccess, "CUDA ERROR: ", cudaStatus);
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  cudaStatus = cudaStreamSynchronize(stream);
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:  TORCH_CHECK(cudaStatus == cudaSuccess, "CUDA ERROR: ", cudaStatus);
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.cpp:void CUDAJpegEncoder::set_quality(const int64_t quality) {
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:#include "decode_jpegs_cuda.h"
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:std::vector<torch::Tensor> decode_jpegs_cuda(
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:      false, "decode_jpegs_cuda: torchvision not compiled with nvJPEG support");
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:#include <ATen/cuda/CUDAContext.h>
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:#include <ATen/cuda/CUDAEvent.h>
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:#include <c10/cuda/CUDAGuard.h>
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:#include <cuda_runtime_api.h>
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:std::unique_ptr<CUDAJpegDecoder> cudaJpegDecoder;
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:std::vector<torch::Tensor> decode_jpegs_cuda(
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:      "torchvision.csrc.io.image.cuda.decode_jpegs_cuda.decode_jpegs_cuda");
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:      device.is_cuda(), "Expected the device parameter to be a cuda device");
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:        !encoded_image.is_cuda(),
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:        "There is a memory leak issue in the nvjpeg library for CUDA versions < 11.6. "
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:        "Make sure to rely on CUDA 11.6 or above before using decode_jpeg(..., device='cuda').");
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:  at::cuda::CUDAGuard device_guard(device);
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:  if (cudaJpegDecoder == nullptr || device != cudaJpegDecoder->target_device) {
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:    if (cudaJpegDecoder != nullptr)
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:      cudaJpegDecoder.reset(new CUDAJpegDecoder(device));
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:      cudaJpegDecoder = std::make_unique<CUDAJpegDecoder>(device);
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:      std::atexit([]() { cudaJpegDecoder.reset(); });
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:          false, "The provided mode is not supported for JPEG decoding on GPU");
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:    at::cuda::CUDAEvent event;
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:    auto result = cudaJpegDecoder->decode_images(contig_images, output_format);
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:        device.has_index() ? at::cuda::getCurrentCUDAStream(
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:                                 cudaJpegDecoder->original_device.index())
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:                           : at::cuda::getCurrentCUDAStream()};
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:    event.record(cudaJpegDecoder->stream);
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:CUDAJpegDecoder::CUDAJpegDecoder(const torch::Device& target_device)
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:    : original_device{torch::kCUDA, c10::cuda::current_device()},
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:              ? at::cuda::getStreamFromPool(false, target_device.index())
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:              : at::cuda::getStreamFromPool(false)} {
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:CUDAJpegDecoder::~CUDAJpegDecoder() {
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:  destructor executes after cuda is already shut down causing SIGSEGV.
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:  just leak the libnvjpeg & cuda variables for the time being and hope
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:  that the CUDA runtime handles cleanup for us.
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:CUDAJpegDecoder::prepare_buffers(
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:std::vector<torch::Tensor> CUDAJpegDecoder::decode_images(
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:    1. Baseline JPEGs: Can be decoded with hardware support on A100+ GPUs.
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:    GPU (albeit with software support only) but need some preprocessing on the
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:    https://github.com/NVIDIA/CUDALibrarySamples/blob/f17940ac4e705bf47a8c39f5365925c1665f6c98/nvJPEG/nvJPEG-Decoder/nvjpegDecoder.cpp#L33
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:    - device (torch::Device): The desired CUDA device for the returned Tensors
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:  cudaError_t cudaStatus;
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:  cudaStatus = cudaStreamSynchronize(stream);
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:      cudaStatus == cudaSuccess,
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:      "Failed to synchronize CUDA stream: ",
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:      cudaStatus);
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:  // baseline JPEGs can be batch decoded with hardware support on A100+ GPUs
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:      cudaStatus = cudaStreamSynchronize(stream);
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:          cudaStatus == cudaSuccess,
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:          "Failed to synchronize CUDA stream: ",
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:          cudaStatus);
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:  cudaStatus = cudaStreamSynchronize(stream);
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:      cudaStatus == cudaSuccess,
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:      "Failed to synchronize CUDA stream: ",
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp:      cudaStatus);
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.h:#include <c10/cuda/CUDAStream.h>
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.h:class CUDAJpegDecoder {
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.h:  CUDAJpegDecoder(const torch::Device& target_device);
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.h:  ~CUDAJpegDecoder();
torchvision/csrc/io/image/cuda/decode_jpegs_cuda.h:  const c10::cuda::CUDAStream stream;
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.h:#include <c10/cuda/CUDAGuard.h>
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.h:#include <c10/cuda/CUDAStream.h>
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.h:class CUDAJpegEncoder {
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.h:  CUDAJpegEncoder(const torch::Device& device);
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.h:  ~CUDAJpegEncoder();
torchvision/csrc/io/image/cuda/encode_jpegs_cuda.h:  const c10::cuda::CUDAStream stream;
torchvision/csrc/io/image/cuda/encode_decode_jpegs_cuda.h:#include "decode_jpegs_cuda.h"
torchvision/csrc/io/image/cuda/encode_decode_jpegs_cuda.h:#include "encode_jpegs_cuda.h"
torchvision/csrc/io/image/cuda/encode_decode_jpegs_cuda.h:Fast jpeg decoding with CUDA.
torchvision/csrc/io/image/cuda/encode_decode_jpegs_cuda.h:A100+ GPUs have dedicated hardware support for jpeg decoding.
torchvision/csrc/io/image/cuda/encode_decode_jpegs_cuda.h:    - device (torch::Device): The desired CUDA device to run the decoding on and
torchvision/csrc/io/image/cuda/encode_decode_jpegs_cuda.h:C10_EXPORT std::vector<torch::Tensor> decode_jpegs_cuda(
torchvision/csrc/io/image/cuda/encode_decode_jpegs_cuda.h:Fast jpeg encoding with CUDA.
torchvision/csrc/io/image/cuda/encode_decode_jpegs_cuda.h:CUDA tensors of dtype torch.uint8 to be encoded.
torchvision/csrc/io/image/cuda/encode_decode_jpegs_cuda.h:    - encoded_images (std::vector<torch::Tensor>): a vector of CUDA
torchvision/csrc/io/image/cuda/encode_decode_jpegs_cuda.h:C10_EXPORT std::vector<torch::Tensor> encode_jpegs_cuda(
torchvision/io/video_reader.py:    backends: video_reader, pyav, and cuda.
torchvision/io/video_reader.py:            if self.backend in ["cuda"]:
torchvision/io/video_reader.py:                    "VideoReader cannot be initialized from bytes object when using cuda or pyav backend."
torchvision/io/video_reader.py:            if self.backend in ["cuda", "pyav"]:
torchvision/io/video_reader.py:                    "VideoReader cannot be initialized from Tensor object when using cuda or pyav backend."
torchvision/io/video_reader.py:        if self.backend == "cuda":
torchvision/io/video_reader.py:            device = torch.device("cuda")
torchvision/io/video_reader.py:            self._c = torch.classes.torchvision.GPUDecoder(src, device)
torchvision/io/video_reader.py:        if self.backend == "cuda":
torchvision/io/video_reader.py:        if self.backend in ["cuda", "video_reader"]:
torchvision/io/video_reader.py:        if self.backend == "cuda":
torchvision/io/video_reader.py:            warnings.warn("GPU decoding only works with video stream.")
torchvision/io/image.py:    """Decode JPEG image(s) into 3D RGB or grayscale Tensor(s), on CPU or CUDA.
torchvision/io/image.py:        When using a CUDA device, passing a list of tensors is more efficient than repeated individual calls to ``decode_jpeg``.
torchvision/io/image.py:        The CUDA version of this function has explicitly been designed with thread-safety in mind.
torchvision/io/image.py:            be stored. If a cuda device is specified, the image will be decoded
torchvision/io/image.py:            with `nvjpeg <https://developer.nvidia.com/nvjpeg>`_. This is only
torchvision/io/image.py:            supported for CUDA version >= 10.1
torchvision/io/image.py:                There is a memory leak in the nvjpeg library for CUDA versions < 11.6.
torchvision/io/image.py:                Make sure to rely on CUDA 11.6 or above before using ``device="cuda"``.
torchvision/io/image.py:        if device.type == "cuda":
torchvision/io/image.py:            return torch.ops.image.decode_jpegs_cuda(input, mode.value, device)
torchvision/io/image.py:        if device.type == "cuda":
torchvision/io/image.py:            return torch.ops.image.decode_jpegs_cuda([input], mode.value, device)[0]
torchvision/io/image.py:    """Encode RGB tensor(s) into raw encoded jpeg bytes, on CPU or CUDA.
torchvision/io/image.py:        Passing a list of CUDA tensors is more efficient than repeated individual calls to ``encode_jpeg``.
torchvision/io/image.py:        if input[0].device.type == "cuda":
torchvision/io/image.py:            return torch.ops.image.encode_jpegs_cuda(input, quality)
torchvision/io/image.py:        if input.device.type == "cuda":
torchvision/io/image.py:            return torch.ops.image.encode_jpegs_cuda([input], quality)[0]
torchvision/io/__init__.py:    from ._load_gpu_decoder import _HAS_GPU_VIDEO_DECODER
torchvision/io/__init__.py:    _HAS_GPU_VIDEO_DECODER = False
torchvision/io/__init__.py:    "_HAS_GPU_VIDEO_DECODER",
torchvision/io/_load_gpu_decoder.py:    _load_library("gpu_decoder")
torchvision/io/_load_gpu_decoder.py:    _HAS_GPU_VIDEO_DECODER = True
torchvision/io/_load_gpu_decoder.py:    _HAS_GPU_VIDEO_DECODER = False
torchvision/transforms/_functional_tensor.py:    if img_chan.is_cuda:
torchvision/transforms/v2/functional/_geometry.py:            # uint8 dtype can be included for cpu and cuda input if nearest mode
torchvision/transforms/v2/functional/_geometry.py:                # This path is hit on non-AVX archs, or on GPU.
torchvision/prototype/models/depth/stereo/crestereo.py:    Adaptive Group Correlation operations from: https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Practical_Stereo_Matching_via_Cascaded_Recurrent_Network_With_Adaptive_Correlation_CVPR_2022_paper.pdf
torchvision/prototype/models/depth/stereo/crestereo.py:    With Adaptive Correlation" <https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Practical_Stereo_Matching_via_Cascaded_Recurrent_Network_With_Adaptive_Correlation_CVPR_2022_paper.pdf>`_ paper.
torchvision/prototype/models/depth/stereo/crestereo.py:    With Adaptive Correlation <https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Practical_Stereo_Matching_via_Cascaded_Recurrent_Network_With_Adaptive_Correlation_CVPR_2022_paper.pdf>`_.
torchvision/__init__.py:    if backend not in ["pyav", "video_reader", "cuda"]:
torchvision/__init__.py:        raise ValueError("Invalid video backend '%s'. Options are 'pyav', 'video_reader' and 'cuda'" % backend)
torchvision/__init__.py:    elif backend == "cuda" and not io._HAS_GPU_VIDEO_DECODER:
torchvision/__init__.py:        message = "cuda video backend is not available."
CMakeLists.txt:option(WITH_CUDA "Enable CUDA support" OFF)
CMakeLists.txt:if(WITH_CUDA)
CMakeLists.txt:  enable_language(CUDA)
CMakeLists.txt:  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
CMakeLists.txt:  add_definitions(-DWITH_CUDA)
CMakeLists.txt:  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
CMakeLists.txt:function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
CMakeLists.txt:        string(REPLACE ";" "," CUDA_flags "${old_flags}")
CMakeLists.txt:            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
CMakeLists.txt:  if(WITH_CUDA)
CMakeLists.txt:    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/wd4819")
CMakeLists.txt:      string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=${diag}")
CMakeLists.txt:    CUDA_CONVERT_FLAGS(torch_cpu)
CMakeLists.txt:    if(TARGET torch_cuda)
CMakeLists.txt:      CUDA_CONVERT_FLAGS(torch_cuda)
CMakeLists.txt:    if(TARGET torch_cuda_cu)
CMakeLists.txt:      CUDA_CONVERT_FLAGS(torch_cuda_cu)
CMakeLists.txt:    if(TARGET torch_cuda_cpp)
CMakeLists.txt:      CUDA_CONVERT_FLAGS(torch_cuda_cpp)
CMakeLists.txt:  ${TVCPP}/ops/autograd ${TVCPP}/ops/cpu ${TVCPP}/io/image/cuda)
CMakeLists.txt:if(WITH_CUDA)
CMakeLists.txt:    list(APPEND ALLOW_LISTED ${TVCPP}/ops/cuda ${TVCPP}/ops/autocast)
scripts/release_notes/classify_prs.py:[rocm] [ROCm] remove HCC references (#8070)
benchmarks/encoding_decoding.py:    print(f"\nCUDA device: {torch.cuda.get_device_name()}")
benchmarks/encoding_decoding.py:    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
benchmarks/encoding_decoding.py:    for device in ["cpu", "cuda"]:
benchmarks/encoding_decoding.py:    for device in ["cpu", "cuda"]:
benchmarks/encoding_decoding.py:    encoded_images_cuda = torchvision.io.encode_jpeg([img.cuda() for img in decoded_images])
benchmarks/encoding_decoding.py:    encoded_images_cpu = [img.cpu() for img in encoded_images_cuda]
cmake/TorchVisionConfig.cmake.in:if(@WITH_CUDA@)
cmake/TorchVisionConfig.cmake.in:  target_compile_definitions(${PN}::${PN} INTERFACE WITH_CUDA)
examples/cpp/run_model.cpp:  if (torch::cuda::is_available()) {
examples/cpp/run_model.cpp:    // Move model and inputs to GPU
examples/cpp/run_model.cpp:    model.to(torch::kCUDA);
examples/cpp/run_model.cpp:    // Add GPU inputs
examples/cpp/run_model.cpp:    torch::TensorOptions options = torch::TensorOptions{torch::kCUDA};
examples/cpp/run_model.cpp:    auto gpu_out = model.forward(inputs);
examples/cpp/run_model.cpp:    std::cout << gpu_out << "\n";
CONTRIBUTING.md:By default, GPU support is built if CUDA is found and `torch.cuda.is_available()` is true. It's possible to force
CONTRIBUTING.md:building GPU support by setting `FORCE_CUDA=1` environment variable, which is useful when building a docker image.

```
