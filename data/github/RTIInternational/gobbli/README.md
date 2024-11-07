# https://github.com/RTIInternational/gobbli

```console
gobbli/model/bert/Dockerfile:ARG GPU
gobbli/model/bert/Dockerfile:FROM tensorflow/tensorflow:1.11.0${GPU:+-gpu}-py3
gobbli/model/bert/model.py:Larger models require more time and GPU memory to run.
gobbli/model/bert/model.py:        device = "gpu" if self.use_gpu else "cpu"
gobbli/model/bert/src/modeling.py:  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
gobbli/model/bert/src/run_classifier.py:flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
gobbli/model/bert/src/run_classifier.py:    # or GPU.
gobbli/model/bert/src/extract_features.py:flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
gobbli/model/bert/src/extract_features.py:  # or GPU.
gobbli/model/bert/src/README.md:replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU,
gobbli/model/bert/src/README.md:All of the code in this repository works out-of-the-box with CPU, GPU, and Cloud
gobbli/model/bert/src/README.md:`BERT-Large` results on the paper using a GPU with 12GB - 16GB of RAM, because
gobbli/model/bert/src/README.md:on the GPU. See the section on [out-of-memory issues](#out-of-memory-issues) for
gobbli/model/bert/src/README.md:The fine-tuning examples which use `BERT-Base` should be able to run on a GPU
gobbli/model/bert/src/README.md:on your local machine, using a GPU like a Titan X or GTX 1080.
gobbli/model/bert/src/README.md:few minutes on most GPUs.
gobbli/model/bert/src/README.md:that it's running on something other than a Cloud TPU, which includes a GPU.
gobbli/model/bert/src/README.md:on a 12GB-16GB GPU due to memory constraints (in fact, even batch size 1 does
gobbli/model/bert/src/README.md:not seem to fit on a 12GB GPU using `BERT-Large`). However, a reasonably strong
gobbli/model/bert/src/README.md:`BERT-Base` model can be trained on the GPU with these hyperparameters:
gobbli/model/bert/src/README.md:device RAM. Therefore, when using a GPU with 12GB - 16GB of RAM, you are likely
gobbli/model/bert/src/README.md:benchmarked the maximum batch size on single Titan X GPU (12GB RAM) with
gobbli/model/bert/src/README.md:effective batch sizes to be used on the GPU. The code will be based on one (or
gobbli/model/bert/src/README.md:    The major use of GPU/TPU memory during DNN training is caching the
gobbli/model/bert/src/README.md:    you will likely get NaNs when training on GPU or TPU due to unchecked
gobbli/model/bert/src/README.md:    computationally expensive, especially on GPUs. If you are pre-training from
gobbli/model/bert/src/README.md:#### Is this code compatible with Cloud TPUs? What about GPUs?
gobbli/model/bert/src/README.md:Yes, all of the code in this repository works out-of-the-box with CPU, GPU, and
gobbli/model/bert/src/README.md:Cloud TPU. However, GPU training is single-GPU only.
gobbli/model/bert/src/run_pretraining.py:flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
gobbli/model/bert/src/run_pretraining.py:    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
gobbli/model/bert/src/run_pretraining.py:  # or GPU.
gobbli/model/bert/src/requirements.txt:# tensorflow-gpu  >= 1.11.0  # GPU version of TensorFlow.
gobbli/model/bert/src/run_classifier_with_tfhub.py:  # or GPU.
gobbli/model/bert/src/run_squad.py:flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
gobbli/model/bert/src/run_squad.py:  # or GPU.
gobbli/model/bert/src/multilingual.md:This is a large dataset, so this will training will take a few hours on a GPU
gobbli/model/spacy/Dockerfile:FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel
gobbli/model/spacy/src/requirements.txt:# We're using the PyTorch image with CUDA 10.1, but spaCy doesn't have an extra
gobbli/model/spacy/src/requirements.txt:# requirements specifier for CUDA 10.1 at the time of this writing (it only has 10.0).
gobbli/model/spacy/src/requirements.txt:# We could use the "cuda" extra requirements specifier, but it results in spaCy
gobbli/model/spacy/src/requirements.txt:# without the NVIDIA runtime (which would require us to have separate images for GPU
gobbli/model/spacy/src/requirements.txt:# and no-GPU).  So, we manually install the spaCy GPU dependencies so we get
gobbli/model/spacy/src/requirements.txt:# wheels compatible with CUDA 10.1.
gobbli/model/spacy/src/requirements.txt:cupy-cuda101==7.0.0
gobbli/model/spacy/src/requirements.txt:thinc_gpu_ops==0.0.4
gobbli/model/spacy/src/run_spacy.py:    # to make sure this is compatible with GPU (cupy array)
gobbli/model/spacy/src/run_spacy.py:        help="Per-GPU batch size to use for training.",
gobbli/model/spacy/src/run_spacy.py:        help="Per-GPU batch size to use for embedding.",
gobbli/model/spacy/src/run_spacy.py:    using_gpu = spacy.prefer_gpu()
gobbli/model/spacy/src/run_spacy.py:    if using_gpu:
gobbli/model/spacy/src/run_spacy.py:        torch.set_default_tensor_type("torch.cuda.FloatTensor")
gobbli/model/spacy/src/run_spacy.py:    device = "gpu" if using_gpu else "cpu"
gobbli/model/use/Dockerfile:ARG GPU
gobbli/model/use/Dockerfile:FROM tensorflow/tensorflow:2.0.1${GPU:+-gpu}-py3
gobbli/model/use/model.py:Larger models require more time and GPU memory to run.
gobbli/model/use/model.py:        device = "gpu" if self.use_gpu else "cpu"
gobbli/model/transformer/Dockerfile:FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
gobbli/model/transformer/model.py:          fit a large batch on the GPU.  The "effective batch size" is
gobbli/model/transformer/src/run_model.py:    n_gpu,
gobbli/model/transformer/src/run_model.py:            if n_gpu > 1:
gobbli/model/transformer/src/run_model.py:            if n_gpu > 1:
gobbli/model/transformer/src/run_model.py:        help="Per-GPU batch size to use for training.",
gobbli/model/transformer/src/run_model.py:        help="Per-GPU batch size to use for validation.",
gobbli/model/transformer/src/run_model.py:        help="Per-GPU batch size to use for prediction.",
gobbli/model/transformer/src/run_model.py:        help="Per-GPU batch size to use for embedding.",
gobbli/model/transformer/src/run_model.py:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gobbli/model/transformer/src/run_model.py:    n_gpu = torch.cuda.device_count()
gobbli/model/transformer/src/run_model.py:    print(f"Number of GPUs: {n_gpu}")
gobbli/model/transformer/src/run_model.py:    # If we have multiple GPUs, size the batches appropriately so DataParallel
gobbli/model/transformer/src/run_model.py:    batch_multiplier = max(1, n_gpu)
gobbli/model/transformer/src/run_model.py:    if n_gpu > 1:
gobbli/model/transformer/src/run_model.py:            n_gpu=n_gpu,
gobbli/model/mtdnn/model.py:Larger models require more time and GPU memory to run.
gobbli/model/mtdnn/src/data_utils/glue_utils.py:def eval_model(model, data, dataset, use_cuda=True, with_label=True):
gobbli/model/mtdnn/src/data_utils/glue_utils.py:    if use_cuda:
gobbli/model/mtdnn/src/data_utils/glue_utils.py:        model.cuda()
gobbli/model/mtdnn/src/data_utils/utils.py:def set_environment(seed, set_cuda=False):
gobbli/model/mtdnn/src/data_utils/utils.py:    if torch.cuda.is_available() and set_cuda:
gobbli/model/mtdnn/src/data_utils/utils.py:        torch.cuda.manual_seed_all(seed)
gobbli/model/mtdnn/src/data_utils/utils.py:def patch_var(v, cuda=True):
gobbli/model/mtdnn/src/data_utils/utils.py:    if cuda:
gobbli/model/mtdnn/src/data_utils/utils.py:        v = Variable(v.cuda(async=True))
gobbli/model/mtdnn/src/data_utils/utils.py:def get_gpu_memory_map():
gobbli/model/mtdnn/src/data_utils/utils.py:            'nvidia-smi', '--query-gpu=memory.used',
gobbli/model/mtdnn/src/data_utils/utils.py:    gpu_memory = [int(x) for x in result.strip().split('\n')]
gobbli/model/mtdnn/src/data_utils/utils.py:    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
gobbli/model/mtdnn/src/data_utils/utils.py:    return gpu_memory_map
gobbli/model/mtdnn/src/README.md:   ```> docker run -it --rm --runtime nvidia  allenlao/pytorch-mt-dnn:v0.1 bash``` </br>
gobbli/model/mtdnn/src/README.md:**Note that we ran experiments on 4 V100 GPUs for base MT-DNN models. You may need to reduce batch size for other GPUs.** <br/>
gobbli/model/mtdnn/src/README.md:**Note that we ran this experiment on 8 V100 GPUs (32G) with a batch size of 32.**
gobbli/model/mtdnn/src/scripts/run_stsb.sh:  echo "train.sh <batch_size> <gpu>"
gobbli/model/mtdnn/src/scripts/run_stsb.sh:gpu=$2
gobbli/model/mtdnn/src/scripts/run_stsb.sh:echo "export CUDA_VISIBLE_DEVICES=${gpu}"
gobbli/model/mtdnn/src/scripts/run_stsb.sh:export CUDA_VISIBLE_DEVICES=${gpu}
gobbli/model/mtdnn/src/scripts/run_rte.sh:  echo "train.sh <batch_size> <gpu>"
gobbli/model/mtdnn/src/scripts/run_rte.sh:gpu=$2
gobbli/model/mtdnn/src/scripts/run_rte.sh:echo "export CUDA_VISIBLE_DEVICES=${gpu}"
gobbli/model/mtdnn/src/scripts/run_rte.sh:export CUDA_VISIBLE_DEVICES=${gpu}
gobbli/model/mtdnn/src/scripts/run_mt_dnn.sh:  echo "train.sh <batch_size> <gpu>"
gobbli/model/mtdnn/src/scripts/run_mt_dnn.sh:gpu=$2
gobbli/model/mtdnn/src/scripts/run_mt_dnn.sh:echo "export CUDA_VISIBLE_DEVICES=${gpu}"
gobbli/model/mtdnn/src/scripts/run_mt_dnn.sh:export CUDA_VISIBLE_DEVICES=${gpu}
gobbli/model/mtdnn/src/scripts/run_mt_dnn.sh:python ../train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on
gobbli/model/mtdnn/src/scripts/domain_adaptation_run.sh:  echo "train.sh <prefix> <bert_path> <train_datasets> <test_datasets> <data_dir> <model_dir> <batch_size> <gpu>"
gobbli/model/mtdnn/src/scripts/domain_adaptation_run.sh:gpu=$8
gobbli/model/mtdnn/src/scripts/domain_adaptation_run.sh:echo "export CUDA_VISIBLE_DEVICES=${gpu}"
gobbli/model/mtdnn/src/scripts/domain_adaptation_run.sh:export CUDA_VISIBLE_DEVICES=${gpu}
gobbli/model/mtdnn/src/scripts/domain_adaptation_run.sh:python ../train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --multi_gpu_on
gobbli/model/mtdnn/src/module/my_optim.py:    def cuda(self):
gobbli/model/mtdnn/src/module/my_optim.py:            self.shadow[k] = v.cuda()
gobbli/model/mtdnn/src/train.py:    parser.add_argument('--multi_gpu_on', action='store_true')
gobbli/model/mtdnn/src/train.py:    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
gobbli/model/mtdnn/src/train.py:                        help='whether to use GPU acceleration.')
gobbli/model/mtdnn/src/train.py:set_environment(args.seed, args.cuda)
gobbli/model/mtdnn/src/train.py:                                gpu=args.cuda,
gobbli/model/mtdnn/src/train.py:                                  gpu=args.cuda, is_train=False,
gobbli/model/mtdnn/src/train.py:                                  gpu=args.cuda, is_train=False,
gobbli/model/mtdnn/src/train.py:    if args.cuda:
gobbli/model/mtdnn/src/train.py:        model.cuda()
gobbli/model/mtdnn/src/train.py:                                                                                 use_cuda=args.cuda)
gobbli/model/mtdnn/src/train.py:                                                                                 use_cuda=args.cuda, with_label=False)
gobbli/model/mtdnn/src/docker/Dockerfile:FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
gobbli/model/mtdnn/src/docker/Dockerfile:RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION cuda90 pytorch=0.4.1 torchvision -c pytorch && \
gobbli/model/mtdnn/src/gobbli_train.py:    parser.add_argument("--multi_gpu_on", action="store_true")
gobbli/model/mtdnn/src/gobbli_train.py:        "--cuda",
gobbli/model/mtdnn/src/gobbli_train.py:        default=torch.cuda.is_available(),
gobbli/model/mtdnn/src/gobbli_train.py:        help="whether to use GPU acceleration.",
gobbli/model/mtdnn/src/gobbli_train.py:def eval_model(model, dataset, use_cuda=True, with_label=True):
gobbli/model/mtdnn/src/gobbli_train.py:    if use_cuda:
gobbli/model/mtdnn/src/gobbli_train.py:        model.cuda()
gobbli/model/mtdnn/src/gobbli_train.py:set_environment(args.seed, args.cuda)
gobbli/model/mtdnn/src/gobbli_train.py:        "gpu": args.cuda,
gobbli/model/mtdnn/src/gobbli_train.py:    if args.cuda:
gobbli/model/mtdnn/src/gobbli_train.py:        model.cuda()
gobbli/model/mtdnn/src/gobbli_train.py:                    model, dataset, use_cuda=args.cuda, with_label=True
gobbli/model/mtdnn/src/gobbli_train.py:        _, scores = eval_model(model, test_data, use_cuda=args.cuda, with_label=False)
gobbli/model/mtdnn/src/mt_dnn/gobbli_model.py:        if self.config['cuda']:
gobbli/model/mtdnn/src/mt_dnn/gobbli_model.py:            labels = labels.cuda(async=True)
gobbli/model/mtdnn/src/mt_dnn/batcher.py:    def __init__(self, data, batch_size=32, gpu=True, is_train=True,
gobbli/model/mtdnn/src/mt_dnn/batcher.py:        self.gpu = gpu
gobbli/model/mtdnn/src/mt_dnn/batcher.py:        v = v.cuda(async=True)
gobbli/model/mtdnn/src/mt_dnn/batcher.py:            if self.gpu:
gobbli/model/mtdnn/src/mt_dnn/model.py:        self.mnetwork = nn.DataParallel(self.network) if opt['multi_gpu_on'] else self.network
gobbli/model/mtdnn/src/mt_dnn/model.py:        if self.config['cuda']:
gobbli/model/mtdnn/src/mt_dnn/model.py:            y = Variable(labels.cuda(async=True), requires_grad=False)
gobbli/model/mtdnn/src/mt_dnn/model.py:            if self.config['cuda']:
gobbli/model/mtdnn/src/mt_dnn/model.py:                weight = Variable(batch_data[batch_meta['factor']].cuda(async=True))
gobbli/model/mtdnn/src/mt_dnn/model.py:    def cuda(self):
gobbli/model/mtdnn/src/mt_dnn/model.py:        self.network.cuda()
gobbli/model/mtdnn/src/mt_dnn/model.py:            self.ema.cuda()
gobbli/model/mtdnn/src/mt_dnn/gobbli_batcher.py:    def __init__(self, path, batch_size=32, gpu=True, labels=None,
gobbli/model/mtdnn/src/mt_dnn/gobbli_batcher.py:        self.gpu = gpu
gobbli/model/mtdnn/src/mt_dnn/gobbli_batcher.py:        v = v.cuda(async=True)
gobbli/model/mtdnn/src/mt_dnn/gobbli_batcher.py:            if self.gpu:
gobbli/model/base.py:    Functionality to facilitate making GPU(s) available to derived classes is available.
gobbli/model/base.py:        use_gpu: bool = False,
gobbli/model/base.py:        nvidia_visible_devices: str = "all",
gobbli/model/base.py:          use_gpu: If True, use the
gobbli/model/base.py:            nvidia-docker runtime (https://github.com/NVIDIA/nvidia-docker) to expose
gobbli/model/base.py:            NVIDIA GPU(s) to the container.  Will cause an error if the computer you're running
gobbli/model/base.py:            on doesn't have an NVIDIA GPU and/or doesn't have the nvidia-docker runtime installed.
gobbli/model/base.py:          nvidia_visible_devices: Which GPUs to make available to the container; ignored if
gobbli/model/base.py:            ``use_gpu`` is False.  If not 'all', should be a comma-separated string: ex. ``1,2``.
gobbli/model/base.py:        self.use_gpu = use_gpu
gobbli/model/base.py:        self.nvidia_visible_devices = nvidia_visible_devices
gobbli/model/base.py:        Establish a base set of docker run kwargs to handle GPU support, etc.
gobbli/model/base.py:        if self.use_gpu:
gobbli/model/base.py:                "NVIDIA_VISIBLE_DEVICES"
gobbli/model/base.py:            ] = self.nvidia_visible_devices
gobbli/model/base.py:            kwargs["runtime"] = "nvidia"
gobbli/model/base.py:        Handle GPU support, etc via common args for any model Docker container.
gobbli/model/base.py:        if self.use_gpu:
gobbli/model/base.py:            kwargs["buildargs"]["GPU"] = "1"
gobbli/interactive/evaluate.py:    "--use-gpu/--use-cpu",
gobbli/interactive/evaluate.py:    "--nvidia-visible-devices",
gobbli/interactive/evaluate.py:    help="Which GPUs to make available to the container; ignored if running on CPU. "
gobbli/interactive/evaluate.py:    use_gpu: bool,
gobbli/interactive/evaluate.py:    nvidia_visible_devices: str,
gobbli/interactive/evaluate.py:        model_data_path, use_gpu, nvidia_visible_devices
gobbli/interactive/explain.py:    "--use-gpu/--use-cpu",
gobbli/interactive/explain.py:    "--nvidia-visible-devices",
gobbli/interactive/explain.py:    help="Which GPUs to make available to the container; ignored if running on CPU. "
gobbli/interactive/explain.py:    use_gpu: bool,
gobbli/interactive/explain.py:    nvidia_visible_devices: str,
gobbli/interactive/explain.py:        model_data_path, use_gpu, nvidia_visible_devices
gobbli/interactive/util.py:    model_data_path: Path, use_gpu: bool, nvidia_visible_devices: str
gobbli/interactive/util.py:      use_gpu: If True, initialize the model using a GPU.
gobbli/interactive/util.py:      nvidia_visible_devices: The list of devices to make available to the model container.
gobbli/interactive/util.py:        "use_gpu": use_gpu,
gobbli/interactive/util.py:        "nvidia_visible_devices": nvidia_visible_devices,
gobbli/interactive/util.py:    use_gpu: bool,
gobbli/interactive/util.py:    nvidia_visible_devices: str,
gobbli/interactive/util.py:      use_gpu: If True, initialize the model using a GPU.
gobbli/interactive/util.py:      nvidia_visible_devices: The list of devices to make available to the model container.
gobbli/interactive/util.py:        "use_gpu": use_gpu,
gobbli/interactive/util.py:        "nvidia_visible_devices": nvidia_visible_devices,
gobbli/interactive/explore.py:    "--use-gpu/--use-cpu",
gobbli/interactive/explore.py:    "--nvidia-visible-devices",
gobbli/interactive/explore.py:    help="Which GPUs to make available to the container; ignored if running on CPU. "
gobbli/interactive/explore.py:    use_gpu: bool,
gobbli/interactive/explore.py:    nvidia_visible_devices: str,
gobbli/interactive/explore.py:                Path(model_data_dir), use_gpu, nvidia_visible_devices
gobbli/interactive/explore.py:                use_gpu,
gobbli/interactive/explore.py:                nvidia_visible_devices,
gobbli/test/classification/test_embeddings.py:    model_gpu_config,
gobbli/test/classification/test_embeddings.py:        **model_gpu_config,
gobbli/test/classification/test_classifiers.py:    model_gpu_config,
gobbli/test/classification/test_classifiers.py:        **model_gpu_config,
gobbli/test/util.py:def skip_if_no_gpu(config):
gobbli/test/util.py:    if not config.option.use_gpu:
gobbli/test/util.py:        pytest.skip("needs --use-gpu option to run")
gobbli/test/augment/test_bertmaskedlm.py:def test_bertmaskedlm_augment(model_gpu_config, gobbli_dir):
gobbli/test/augment/test_bertmaskedlm.py:        data_dir=model_test_dir(BERTMaskedLM), load_existing=True, **model_gpu_config
gobbli/test/augment/test_marian.py:def test_marianmt_augment(model_gpu_config, gobbli_dir):
gobbli/test/augment/test_marian.py:        **model_gpu_config,
gobbli/test/experiment/test_classification_experiment.py:        task_num_gpus=0,
gobbli/test/experiment/test_base_experiment.py:from gobbli.test.util import MockDataset, MockExperiment, MockModel, skip_if_no_gpu
gobbli/test/experiment/test_base_experiment.py:def test_base_experiment_gpu(tmpdir, request):
gobbli/test/experiment/test_base_experiment.py:    skip_if_no_gpu(request.config)
gobbli/test/experiment/test_base_experiment.py:        ray_kwargs={"num_gpus": 1},
gobbli/test/experiment/test_base_experiment.py:    # Make sure GPUs are available
gobbli/test/experiment/test_base_experiment.py:    @ray.remote(num_gpus=1)
gobbli/test/experiment/test_base_experiment.py:    def find_gpus():
gobbli/test/experiment/test_base_experiment.py:        return ray.get_gpu_ids()
gobbli/test/experiment/test_base_experiment.py:    assert len(ray.get(find_gpus.remote())) > 0
gobbli/augment/bert/Dockerfile:FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
gobbli/augment/bert/model.py:        if self.use_gpu:
gobbli/augment/bert/model.py:            if self.nvidia_visible_devices == "all":
gobbli/augment/bert/model.py:                device = "cuda"
gobbli/augment/bert/model.py:                device_num = self.nvidia_visible_devices.split(",")[0]
gobbli/augment/bert/model.py:                device = f"cuda:{device_num}"
gobbli/augment/marian/Dockerfile:FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
gobbli/augment/marian/model.py:        if self.use_gpu:
gobbli/augment/marian/model.py:            if self.nvidia_visible_devices == "all":
gobbli/augment/marian/model.py:                device = "cuda"
gobbli/augment/marian/model.py:                device_num = self.nvidia_visible_devices.split(",")[0]
gobbli/augment/marian/model.py:                device = f"cuda:{device_num}"
gobbli/experiment/classification.py:    init_gpu_config,
gobbli/experiment/classification.py:        @ray.remote(num_cpus=self.task_num_cpus, num_gpus=self.task_num_gpus)
gobbli/experiment/classification.py:            use_gpu, nvidia_visible_devices = init_gpu_config()
gobbli/experiment/classification.py:                use_gpu=use_gpu,
gobbli/experiment/classification.py:                nvidia_visible_devices=nvidia_visible_devices,
gobbli/experiment/classification.py:        @ray.remote(num_cpus=self.task_num_cpus, num_gpus=self.task_num_gpus)
gobbli/experiment/classification.py:            use_gpu, nvidia_visible_devices = init_gpu_config()
gobbli/experiment/classification.py:                use_gpu=use_gpu,
gobbli/experiment/classification.py:                nvidia_visible_devices=nvidia_visible_devices,
gobbli/experiment/base.py:def init_gpu_config() -> Tuple[bool, str]:
gobbli/experiment/base.py:    Determine the GPU configuration from the current ray environment
gobbli/experiment/base.py:      2-tuple: whether GPU should be used and a comma-separated string
gobbli/experiment/base.py:      containing the ids of the GPUs that should be used
gobbli/experiment/base.py:        gpu_ids = ray.get_gpu_ids()
gobbli/experiment/base.py:        # This message is either 'ray.get_gpu_ids() currently does not work in PYTHON MODE'
gobbli/experiment/base.py:        if "ray.get_gpu_ids() currently does not work in" in str(e):
gobbli/experiment/base.py:            gpu_ids = []
gobbli/experiment/base.py:    use_gpu = len(gpu_ids) > 0
gobbli/experiment/base.py:    nvidia_visible_devices = ",".join(str(i) for i in gpu_ids)
gobbli/experiment/base.py:    return use_gpu, nvidia_visible_devices
gobbli/experiment/base.py:        task_num_gpus: int = 0,
gobbli/experiment/base.py:          task_num_gpus: Number of GPUs to reserve per task.
gobbli/experiment/base.py:            node using all available CPUs and no GPUs, but these arguments can be used to connect
gobbli/experiment/base.py:        self.task_num_gpus = task_num_gpus
docs/troubleshooting.rst:I'm running out of GPU memory
docs/troubleshooting.rst: - Decreasing the :paramref:`train_batch_size <gobbli.io.TrainInput.params.train_batch_size>` if you're training; this is the biggest driver of GPU memory usage.  Beware of making the batch size so small that the model can't update gradients accurately, though. The :class:`gobbli.model.transformer.Transformer` model supports gradient accumulation, which can be used to counteract the detrimental effect of a smaller batch size.
docs/advanced_usage.rst:- **Parallel/Distributed Experimentation**: gobbli uses `ray <https://ray.readthedocs.io/en/latest/>`__ under the hood to run multiple training/validation steps in parallel.  Ray creates and uses a local cluster composed of all CPUs on your machine by default, but it can also be used to add GPUs or connect to an existing distributed cluster. Note ray (and gobbli) must be installed on all worker nodes in the cluster.  Experiments accept an optional :paramref:`ray_kwargs <gobbli.experiment.base.BaseExperiment.params.ray_kwargs>` option, which is passed directly to :func:`ray.init`.  Use this parameter for more control over the underlying Ray cluster.  **NOTE:** If you're running an experiment on a single node, gobbli will simply pass checkpoints around as file paths, since the Ray master and workers share a filesystem.  If you're running a distributed experiment, gobbli cannot rely on file paths being the same between workers and the master node, so it will save checkpoints as gzip-compressed tar archives in memory and store them in the Ray object store.  This means your object store must be able to hold weights for as many trials as will be run in one experiment, which may be a **lot** of memory.
docs/advanced_usage.rst:- **Enabling GPU support**: During experiments, gobbli exposes GPUs to models based on whether they're made available to the Ray cluster and are required for tasks.  To run a GPU-enabled experiment, reserve a nonzero number of GPUs for each task via the :paramref:`task_num_gpus <gobbli.experiment.base.BaseExperiment.params.task_num_gpus>` parameter and tell Ray the cluster contains a nonzero number of GPUs via the :obj:`num_gpus` argument to :func:`ray.init`.
docs/prerequisites.rst:If you want to train models using a GPU, you will additionally need an NVIDIA graphics card and `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`__.
docs/interactive_apps.rst: - Use ``gobbli <app_name> -- --help`` to see information on allowed arguments for each app, including enabling GPU usage and multilabel classification support.
docs/quickstart.rst:A high-level overview of each type of experiment follows.  For an overview of more detailed configuration options, including parameter tuning, parallel/distributed experiments, and using GPUs in experiments, see :ref:`advanced-experimentation`.  For example experiments on benchmark datasets, see the Markdown documents in the ``benchmark/`` directory of the repository.
README.md:If you want to run the tests GPU(s) enabled, see the `--use-gpu` and `--nvidia-visible-devices` arguments under `py.test --help`.  If your local machine doesn't have an NVIDIA GPU, but you have access to one that does via SSH, you can use the `test_remote_gpu.sh` script to run the tests with GPU enabled over SSH.
ci-gpu/docker-compose.yml:  gobbli-ci-gpu:
ci-gpu/docker-compose.yml:    runtime: nvidia
ci-gpu/docker-compose.yml:      NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-all}
ci-gpu/docker-compose.yml:    command: bash -c 'py.test -x --use-gpu --nvidia-visible-devices $NVIDIA_VISIBLE_DEVICES; chmod -R a+w ./'
docker-compose.yml:# GPU not enabled to prevent dependency on the NVIDIA docker runtime
benchmark/scenario.py:        "use_gpu": os.getenv("GOBBLI_USE_GPU") is not None,
benchmark/scenario.py:        "nvidia_visible_devices": os.getenv("NVIDIA_VISIBLE_DEVICES", ""),
benchmark/scenario.py:            # Construct the dict of kwargs up-front so each run can override the "use_gpu"
benchmark/scenario.py:            # which have trouble controlling memory usage on the GPU and don't gain
benchmark/scenario.py:            # applied ex. to store data in the correct place and use GPU(s)
benchmark/run_benchmarks.sh:if [[ -n "$GOBBLI_USE_GPU" ]]; then
benchmark/run_benchmarks.sh:    image_name="${image_name}-gpu"
benchmark/run_benchmarks.sh:    echo "GPU enabled."
benchmark/run_benchmarks.sh:    echo "GPU disabled; running on CPU."
benchmark/README.md:To run with GPU support enabled:
benchmark/README.md:    export GOBBLI_USE_GPU=1
benchmark/docker-compose.yml:  gobbli-benchmark-gpu:
benchmark/docker-compose.yml:    runtime: nvidia
benchmark/docker-compose.yml:      NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-all}
benchmark/docker-compose.yml:      GOBBLI_USE_GPU: "1"
benchmark/benchmark_util.py:    use_gpu = os.getenv("GOBBLI_USE_GPU") is not None
benchmark/benchmark_util.py:    # FastText doesn't need a GPU
benchmark/benchmark_util.py:    gpus_needed = 1 if use_gpu and model_cls not in (FastText,) else 0
benchmark/benchmark_util.py:        task_num_gpus=gpus_needed,
benchmark/benchmark_util.py:            "num_gpus": 1 if use_gpu else 0,
benchmark/BENCHMARK_SPECS.yml:        # Force no GPU usage since spaCy doesn't benefit much from it
benchmark/BENCHMARK_SPECS.yml:        use_gpu: False
benchmark/BENCHMARK_SPECS.yml:        # Force no GPU usage since spaCy doesn't benefit much from it
benchmark/BENCHMARK_SPECS.yml:        use_gpu: False
benchmark/benchmark_output/imdb_embed/spaCy/run-meta.json:{"name": "spaCy", "model_name": "SpaCyModel", "model_params": {"model": "en_core_web_lg", "use_gpu": false}, "preprocess_func": null, "batch_size": 32}
benchmark/benchmark_output/newsgroups_embed/spaCy/run-meta.json:{"name": "spaCy", "model_name": "SpaCyModel", "model_params": {"model": "en_core_web_lg", "use_gpu": false}, "preprocess_func": null, "batch_size": 32}
benchmark/benchmark_output/class_imbalance/GPT/output.md:Using device: cuda
benchmark/benchmark_output/class_imbalance/GPT/output.md:Number of GPUs: 1
benchmark/benchmark_output/class_imbalance/GPT2/output.md:Using device: cuda
benchmark/benchmark_output/class_imbalance/GPT2/output.md:Number of GPUs: 1
test_remote_gpu.sh:# Run tests on a GPU machine over SSH.  Assumes the remote user is a member of the
test_remote_gpu.sh:# Docker/docker-compose/nvidia-docker are already installed on the remote server.
test_remote_gpu.sh:    echo "    visible_devices: Value to use for the NVIDIA_VISIBLE_DEVICES environment "
test_remote_gpu.sh:    echo "      variable controlling which GPUs are made available to the container "
test_remote_gpu.sh:visible_gpus="$3"
test_remote_gpu.sh:ssh "$ssh_string" "cd $remote_repo_dir/ci-gpu \
test_remote_gpu.sh:    && export NVIDIA_VISIBLE_DEVICES=$visible_gpus \
test_remote_gpu.sh:    && docker-compose build gobbli-ci-gpu \
test_remote_gpu.sh:    && docker-compose run --rm gobbli-ci-gpu"
conftest.py:        "--use-gpu",
conftest.py:        help="Use a GPU where applicable for running models "
conftest.py:        "in tests.  Limit available GPUs via --nvidia-visible-devices.",
conftest.py:        "--nvidia-visible-devices",
conftest.py:        help="Which GPUs to make available for testing, if applicable. "
conftest.py:        "'all' or a comma-separated string of GPU IDs (ex '1,3').",
conftest.py:def model_gpu_config(request):
conftest.py:    not use the GPU.
conftest.py:    gpu_config = {}
conftest.py:    if request.config.getoption("use_gpu"):
conftest.py:        gpu_config["use_gpu"] = True
conftest.py:        gpu_config["nvidia_visible_devices"] = request.config.getoption(
conftest.py:            "nvidia_visible_devices"
conftest.py:    return gpu_config

```
