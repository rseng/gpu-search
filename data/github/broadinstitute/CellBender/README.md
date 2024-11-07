# https://github.com/broadinstitute/CellBender

```console
README.rst:        --cuda \
README.rst:A GPU-enabled docker image is available from the Google Container Registry (GCR) as:
README.rst:the relevant CUDA drivers get installed and that ``torch.cuda.is_available()``
README.rst:returns ``True`` if you have a GPU available.
docs/source/usage/index.rst:    cellbender remove-background --cuda --input input_file.h5 --output output_file.h5
docs/source/usage/index.rst:(leave out the flag ``--cuda`` if you are not using a GPU... but you should use a GPU!):
docs/source/usage/index.rst:                    --cuda \
docs/source/usage/index.rst:* ``--cuda``: Include this flag.  The code is meant to be run on a GPU.
docs/source/installation/index.rst:If your machine has a GPU with appropriate drivers installed, it should be
docs/source/installation/index.rst:automatically detected, and the appropriate version of PyTorch with CUDA support
docs/source/installation/index.rst:the relevant CUDA drivers get installed and that ``torch.cuda.is_available()``
docs/source/installation/index.rst:returns ``True`` if you have a GPU available.
docs/source/installation/index.rst:A GPU-enabled docker image is available from the Google Container Registry (GCR) as:
docs/source/reference/index.rst:make use of an Nvidia Tesla T4 GPU on Google Cloud architecture.
docs/source/tutorial/index.rst:typical CPU. Processing the full untrimmed dataset requires a CUDA-enabled GPU (e.g. NVIDIA Testla T4)
docs/source/tutorial/index.rst:Again, here we leave out the ``--cuda`` flag solely for the purposes of being able to run this
docs/source/tutorial/index.rst:command on a CPU.  But a GPU is highly recommended for real datasets.
docs/source/changelog/index.rst:seamlessly on preemptible GPU instances, which are a fraction of the cost of
docs/source/changelog/index.rst:a workflow using Google Colab on a GPU for free.
docs/source/changelog/index.rst:- Produces checkpoint files, and the WDL can run seamlessly on preemptible GPUs
docs/source/troubleshooting/index.rst:* :ref:`Do I really need a GPU to run this? <a3>`
docs/source/troubleshooting/index.rst:* :ref:`I am getting a GPU-out-of-memory error (process "Killed") <a19>`
docs/source/troubleshooting/index.rst:* Do I really need a GPU to run this?
docs/source/troubleshooting/index.rst:  * While running on a GPU might seem like an insurmountable obstacle for those without
docs/source/troubleshooting/index.rst:    a GPU handy, consider trying out our
docs/source/troubleshooting/index.rst:    which runs on a GPU on Google Cloud at the click of a button.
docs/source/troubleshooting/index.rst:* I am getting a GPU-out-of-memory error (process ``Killed``)
docs/source/troubleshooting/index.rst:  * If you can, try running on an Nvidia Tesla T4 GPU, which has more RAM than
docs/source/troubleshooting/index.rst:  * Currently, CellBender only makes use of 1 GPU, so extra GPUs will not help.
cellbender/remove_background/run.py:    if torch.cuda.is_available():
cellbender/remove_background/run.py:        torch.cuda.manual_seed_all(consts.RANDOM_SEED)
cellbender/remove_background/run.py:    if not args.use_cuda:
cellbender/remove_background/run.py:            device='cuda' if args.use_cuda else 'cpu',  # TODO check this
cellbender/remove_background/run.py:                device='cuda',
cellbender/remove_background/run.py:            device='cuda' if args.use_cuda else 'cpu',
cellbender/remove_background/run.py:    if args.use_cuda:
cellbender/remove_background/run.py:        torch.cuda.manual_seed_all(consts.RANDOM_SEED)
cellbender/remove_background/run.py:                                   force_device='cuda:0' if args.use_cuda else 'cpu',
cellbender/remove_background/run.py:                                          use_cuda=args.use_cuda)
cellbender/remove_background/run.py:                                   use_cuda=args.use_cuda)
cellbender/remove_background/data/extras/simulate.py:if torch.cuda.is_available():
cellbender/remove_background/data/extras/simulate.py:    USE_CUDA = True
cellbender/remove_background/data/extras/simulate.py:    DEVICE = 'cuda'
cellbender/remove_background/data/extras/simulate.py:    USE_CUDA = False
cellbender/remove_background/data/extras/simulate.py:def comprehensive_random_seed(seed, use_cuda=USE_CUDA):
cellbender/remove_background/data/extras/simulate.py:    if use_cuda:
cellbender/remove_background/data/extras/simulate.py:        torch.cuda.manual_seed_all(seed)
cellbender/remove_background/data/extras/simulate.py:        force_device='cpu' if not torch.cuda.is_available() else None)
cellbender/remove_background/data/extras/simulate.py:    if torch.cuda.is_available():
cellbender/remove_background/data/extras/simulate.py:        data_loader.use_cuda = True
cellbender/remove_background/data/extras/simulate.py:        data_loader.device = 'cuda'
cellbender/remove_background/data/extras/simulate.py:        model.use_cuda = True
cellbender/remove_background/data/extras/simulate.py:        model.device = 'cuda'
cellbender/remove_background/data/extras/simulate.py:        data_loader.use_cuda = False
cellbender/remove_background/data/extras/simulate.py:        model.use_cuda = False
cellbender/remove_background/data/extras/simulate.py:            z=torch.tensor(z_chunk).to('cuda' if torch.cuda.is_available() else 'cpu').float(),
cellbender/remove_background/data/extras/simulate.py:            y=torch.ones(num_cells).to('cuda' if torch.cuda.is_available() else 'cpu').float(),
cellbender/remove_background/data/extras/simulate.py:        y=torch.zeros(n_droplets - i).to('cuda' if torch.cuda.is_available() else 'cpu').float(),
cellbender/remove_background/data/dataprep.py:                 use_cuda: bool = True):
cellbender/remove_background/data/dataprep.py:            use_cuda: True to load data to GPU
cellbender/remove_background/data/dataprep.py:        self.use_cuda = use_cuda
cellbender/remove_background/data/dataprep.py:        if self.use_cuda:
cellbender/remove_background/data/dataprep.py:            self.device = 'cuda'
cellbender/remove_background/data/dataprep.py:                                  use_cuda: bool = True) -> Tuple[
cellbender/remove_background/data/dataprep.py:        use_cuda: If True, the data loader will load tensors on GPU.
cellbender/remove_background/data/dataprep.py:                              use_cuda=use_cuda)
cellbender/remove_background/data/dataprep.py:                             use_cuda=use_cuda)
cellbender/remove_background/data/dataset.py:                       use_cuda: bool = True,
cellbender/remove_background/data/dataset.py:            use_cuda: Whether to load data into GPU memory.
cellbender/remove_background/data/dataset.py:            use_cuda=use_cuda,
cellbender/remove_background/argparser.py:    subparser.add_argument("--cuda",
cellbender/remove_background/argparser.py:                           dest="use_cuda", action="store_true",
cellbender/remove_background/argparser.py:                           help="Including the flag --cuda will run the "
cellbender/remove_background/argparser.py:                                "inference on a GPU.")
cellbender/remove_background/argparser.py:                                "out of GPU memory creating the posterior "
cellbender/remove_background/tests/test_checkpoint.py:from .conftest import USE_CUDA
cellbender/remove_background/tests/test_checkpoint.py:    def __init__(self, use_cuda=False):
cellbender/remove_background/tests/test_checkpoint.py:        self.use_cuda = use_cuda
cellbender/remove_background/tests/test_checkpoint.py:        if self.use_cuda:
cellbender/remove_background/tests/test_checkpoint.py:            self.cuda = torch.randint(low=0, high=100000, size=[1], device='cuda').item()
cellbender/remove_background/tests/test_checkpoint.py:        if self.use_cuda:
cellbender/remove_background/tests/test_checkpoint.py:            return f'python {self.python}; numpy {self.numpy}; torch {self.torch}; torch_cuda {self.cuda}'
cellbender/remove_background/tests/test_checkpoint.py:    tmp_args1 = argparse.Namespace(epochs=100, expected_cells=1000, use_cuda=True)
cellbender/remove_background/tests/test_checkpoint.py:    tmp_args2 = argparse.Namespace(epochs=200, expected_cells=1000, use_cuda=True)
cellbender/remove_background/tests/test_checkpoint.py:    tmp_args3 = argparse.Namespace(epochs=100, expected_cells=500, use_cuda=True)
cellbender/remove_background/tests/test_checkpoint.py:def create_random_state_blank_slate(seed, use_cuda=USE_CUDA):
cellbender/remove_background/tests/test_checkpoint.py:    if use_cuda:
cellbender/remove_background/tests/test_checkpoint.py:        torch.cuda.manual_seed_all(seed)
cellbender/remove_background/tests/test_checkpoint.py:def perturb_random_state(n, use_cuda=USE_CUDA):
cellbender/remove_background/tests/test_checkpoint.py:        if use_cuda:
cellbender/remove_background/tests/test_checkpoint.py:            torch.randn((1,), device='cuda')
cellbender/remove_background/tests/test_checkpoint.py:@pytest.mark.parametrize('cuda',
cellbender/remove_background/tests/test_checkpoint.py:                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
cellbender/remove_background/tests/test_checkpoint.py:                                       reason='requires CUDA'))],
cellbender/remove_background/tests/test_checkpoint.py:                         ids=lambda b: 'cuda' if b else 'cpu')
cellbender/remove_background/tests/test_checkpoint.py:def test_save_and_load_random_state(tmpdir_factory, perturbed_random_state_dict, cuda):
cellbender/remove_background/tests/test_checkpoint.py:    counterfactual = RandomState(use_cuda=cuda)
cellbender/remove_background/tests/test_checkpoint.py:    incorrect = RandomState(use_cuda=cuda)  # a second draw
cellbender/remove_background/tests/test_checkpoint.py:    actual = RandomState(use_cuda=cuda)
cellbender/remove_background/tests/test_checkpoint.py:        self.use_cuda = torch.cuda.is_available()
cellbender/remove_background/tests/test_checkpoint.py:        # self.to(device='cuda' if self.use_cuda else 'cpu')  # CUDA not tested
cellbender/remove_background/tests/test_checkpoint.py:@pytest.mark.parametrize('cuda',
cellbender/remove_background/tests/test_checkpoint.py:                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
cellbender/remove_background/tests/test_checkpoint.py:                                       reason='requires CUDA'))],
cellbender/remove_background/tests/test_checkpoint.py:                         ids=lambda b: 'cuda' if b else 'cpu')
cellbender/remove_background/tests/test_checkpoint.py:def test_save_and_load_cellbender_checkpoint(tmpdir_factory, cuda, scheduler):
cellbender/remove_background/tests/test_checkpoint.py:    args.use_cuda = cuda
cellbender/remove_background/tests/test_integration.py:from .conftest import USE_CUDA
cellbender/remove_background/tests/test_integration.py:@pytest.mark.parametrize('cuda',
cellbender/remove_background/tests/test_integration.py:                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
cellbender/remove_background/tests/test_integration.py:                                       reason='requires CUDA'))],
cellbender/remove_background/tests/test_integration.py:                         ids=lambda b: 'cuda' if b else 'cpu')
cellbender/remove_background/tests/test_integration.py:def test_full_run(tmpdir_factory, h5_v3_file, cuda):
cellbender/remove_background/tests/test_integration.py:    if cuda:
cellbender/remove_background/tests/test_integration.py:        input_args.append('--cuda')
cellbender/remove_background/tests/test_infer.py:# USE_CUDA = torch.cuda.is_available()
cellbender/remove_background/tests/test_infer.py:# @pytest.mark.parametrize('cuda',
cellbender/remove_background/tests/test_infer.py:#                           pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
cellbender/remove_background/tests/test_infer.py:#                                        reason='requires CUDA'))],
cellbender/remove_background/tests/test_infer.py:#                          ids=lambda b: 'cuda' if b else 'cpu')
cellbender/remove_background/tests/test_infer.py:# def test_dense_to_sparse_op_numpy(simulated_dataset, cuda):
cellbender/remove_background/tests/test_infer.py:#         use_cuda=cuda,
cellbender/remove_background/tests/test_infer.py:# @pytest.mark.parametrize('cuda',
cellbender/remove_background/tests/test_infer.py:#                           pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
cellbender/remove_background/tests/test_infer.py:#                                        reason='requires CUDA'))],
cellbender/remove_background/tests/test_infer.py:#                          ids=lambda b: 'cuda' if b else 'cpu')
cellbender/remove_background/tests/test_infer.py:# def test_dense_to_sparse_op_torch(simulated_dataset, cuda):
cellbender/remove_background/tests/test_infer.py:#         use_cuda=cuda,
cellbender/remove_background/tests/test_infer.py:# @pytest.mark.parametrize('cuda',
cellbender/remove_background/tests/test_infer.py:#                           pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
cellbender/remove_background/tests/test_infer.py:#                                        reason='requires CUDA'))],
cellbender/remove_background/tests/test_infer.py:#                          ids=lambda b: 'cuda' if b else 'cpu')
cellbender/remove_background/tests/test_infer.py:# def test_mckp_noise_given_log_prob_tensor(cuda, fun):
cellbender/remove_background/tests/test_infer.py:#     device = 'cuda' if cuda else 'cpu'
cellbender/remove_background/tests/test_posterior.py:USE_CUDA = torch.cuda.is_available()
cellbender/remove_background/tests/test_posterior.py:@pytest.mark.parametrize('cuda',
cellbender/remove_background/tests/test_posterior.py:                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
cellbender/remove_background/tests/test_posterior.py:                                       reason='requires CUDA'))],
cellbender/remove_background/tests/test_posterior.py:                         ids=lambda b: 'cuda' if b else 'cpu')
cellbender/remove_background/tests/test_posterior.py:def test_PRq(log_prob_coo, alpha, n_chunks, cuda):
cellbender/remove_background/tests/test_posterior.py:        device='cuda' if cuda else 'cpu',
cellbender/remove_background/tests/test_posterior.py:@pytest.mark.parametrize('cuda',
cellbender/remove_background/tests/test_posterior.py:                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
cellbender/remove_background/tests/test_posterior.py:                                       reason='requires CUDA'))],
cellbender/remove_background/tests/test_posterior.py:                         ids=lambda b: 'cuda' if b else 'cpu')
cellbender/remove_background/tests/test_posterior.py:def test_PRmu(log_prob_coo, fpr, per_gene, n_chunks, cuda):
cellbender/remove_background/tests/test_posterior.py:        device='cuda' if cuda else 'cpu',
cellbender/remove_background/tests/test_posterior.py:        device='cuda' if cuda else 'cpu',
cellbender/remove_background/tests/test_posterior.py:        device='cuda' if cuda else 'cpu',
cellbender/remove_background/tests/test_posterior.py:@pytest.mark.parametrize('cuda',
cellbender/remove_background/tests/test_posterior.py:                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
cellbender/remove_background/tests/test_posterior.py:                                       reason='requires CUDA'))],
cellbender/remove_background/tests/test_posterior.py:                         ids=lambda b: 'cuda' if b else 'cpu')
cellbender/remove_background/tests/test_posterior.py:def test_compute_mean_target_removal_as_function(log_prob_coo, fpr, per_gene, cuda):
cellbender/remove_background/tests/test_posterior.py:    device = 'cuda' if cuda else 'cpu'
cellbender/remove_background/tests/test_train.py:from .conftest import USE_CUDA
cellbender/remove_background/tests/test_train.py:@pytest.mark.parametrize('cuda',
cellbender/remove_background/tests/test_train.py:                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
cellbender/remove_background/tests/test_train.py:                                       reason='requires CUDA'))],
cellbender/remove_background/tests/test_train.py:                         ids=lambda b: 'cuda' if b else 'cpu')
cellbender/remove_background/tests/test_train.py:def test_one_cycle_scheduler(dropped_minibatch, cuda):
cellbender/remove_background/tests/test_train.py:    device = 'cuda' if cuda else 'cpu'
cellbender/remove_background/tests/test_train.py:                               use_cuda=cuda)
cellbender/remove_background/tests/test_sparse_utils.py:USE_CUDA = torch.cuda.is_available()
cellbender/remove_background/tests/test_sparse_utils.py:@pytest.mark.parametrize('cuda',
cellbender/remove_background/tests/test_sparse_utils.py:                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
cellbender/remove_background/tests/test_sparse_utils.py:                                       reason='requires CUDA'))],
cellbender/remove_background/tests/test_sparse_utils.py:                         ids=lambda b: 'cuda' if b else 'cpu')
cellbender/remove_background/tests/test_sparse_utils.py:def test_dense_to_sparse_op_torch(simulated_dataset, cuda):
cellbender/remove_background/tests/test_sparse_utils.py:        use_cuda=cuda,
cellbender/remove_background/tests/benchmarking/docker_image_check_cuda_status.wdl:task run_check_pytorch_cuda_status {
cellbender/remove_background/tests/benchmarking/docker_image_check_cuda_status.wdl:        String? hardware_gpu_type = "nvidia-tesla-t4"
cellbender/remove_background/tests/benchmarking/docker_image_check_cuda_status.wdl:        String? nvidia_driver_version = "470.82.01"  # need >=465.19.01 for CUDA 11.3
cellbender/remove_background/tests/benchmarking/docker_image_check_cuda_status.wdl:        assert torch.cuda.is_available()
cellbender/remove_background/tests/benchmarking/docker_image_check_cuda_status.wdl:        gpuCount: 1
cellbender/remove_background/tests/benchmarking/docker_image_check_cuda_status.wdl:        gpuType: "${hardware_gpu_type}"
cellbender/remove_background/tests/benchmarking/docker_image_check_cuda_status.wdl:        nvidiaDriverVersion: "${nvidia_driver_version}"
cellbender/remove_background/tests/benchmarking/docker_image_check_cuda_status.wdl:workflow check_pytorch_cuda_status {
cellbender/remove_background/tests/benchmarking/docker_image_check_cuda_status.wdl:    call run_check_pytorch_cuda_status
cellbender/remove_background/tests/benchmarking/cuda_check_inputs.json:  "check_pytorch_cuda_status.run_check_pytorch_cuda_status.docker_image": "us.gcr.io/broad-dsde-methods/cellbender:20230427"
cellbender/remove_background/tests/benchmarking/benchmark.wdl:    call cellbender.run_cellbender_remove_background_gpu as cb {
cellbender/remove_background/tests/test_monitor.py:USE_CUDA = torch.cuda.is_available()
cellbender/remove_background/tests/test_monitor.py:@pytest.mark.parametrize('cuda',
cellbender/remove_background/tests/test_monitor.py:                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
cellbender/remove_background/tests/test_monitor.py:                                       reason='requires CUDA'))],
cellbender/remove_background/tests/test_monitor.py:                         ids=lambda b: 'cuda' if b else 'cpu')
cellbender/remove_background/tests/test_monitor.py:def test_get_hardware_usage(cuda):
cellbender/remove_background/tests/test_monitor.py:    print(get_hardware_usage(use_cuda=cuda))
cellbender/remove_background/tests/test_dataprep.py:USE_CUDA = torch.cuda.is_available()
cellbender/remove_background/tests/test_dataprep.py:@pytest.mark.parametrize('cuda',
cellbender/remove_background/tests/test_dataprep.py:                          pytest.param(True, marks=pytest.mark.skipif(not USE_CUDA,
cellbender/remove_background/tests/test_dataprep.py:                                       reason='requires CUDA'))],
cellbender/remove_background/tests/test_dataprep.py:                         ids=lambda b: 'cuda' if b else 'cpu')
cellbender/remove_background/tests/test_dataprep.py:def test_dataloader_sorting(simulated_dataset, cuda):
cellbender/remove_background/tests/test_dataprep.py:        use_cuda=cuda,
cellbender/remove_background/tests/test_dataprep.py:        use_cuda=cuda,
cellbender/remove_background/tests/test_dataprep.py:            use_cuda=cuda,
cellbender/remove_background/tests/conftest.py:USE_CUDA = torch.cuda.is_available()
cellbender/remove_background/cli.py:        # If cuda is requested, make sure it is available.
cellbender/remove_background/cli.py:        if args.use_cuda:
cellbender/remove_background/cli.py:            assert torch.cuda.is_available(), "Trying to use CUDA, " \
cellbender/remove_background/cli.py:                                              "but CUDA is not available."
cellbender/remove_background/cli.py:            # Warn the user in case the CUDA flag was forgotten by mistake.
cellbender/remove_background/cli.py:            if torch.cuda.is_available():
cellbender/remove_background/cli.py:                sys.stdout.write("Warning: CUDA is available, but will not be "
cellbender/remove_background/cli.py:                                 "used.  Use the flag --cuda for "
cellbender/remove_background/model.py:        use_cuda: Will use GPU if True.
cellbender/remove_background/model.py:        device: Either 'cpu' or 'cuda' depending on value of use_cuda.
cellbender/remove_background/model.py:                 use_cuda: bool,
cellbender/remove_background/model.py:        # Determine whether we are working on a GPU.
cellbender/remove_background/model.py:        if use_cuda:
cellbender/remove_background/model.py:            # Calling cuda() here will put all the parameters of
cellbender/remove_background/model.py:            # the encoder and decoder networks into GPU memory.
cellbender/remove_background/model.py:            self.cuda()
cellbender/remove_background/model.py:                    value.cuda()
cellbender/remove_background/model.py:            self.device = 'cuda'
cellbender/remove_background/model.py:        self.use_cuda = use_cuda
cellbender/remove_background/model.py:                        use_cuda=self.use_cuda, device=self.device):
cellbender/remove_background/model.py:                        use_cuda=self.use_cuda, device=self.device):
cellbender/remove_background/train.py:                logger.debug('\n' + get_hardware_usage(use_cuda=model.use_cuda))
cellbender/remove_background/train.py:    # Free up all the GPU memory we can once training is complete.
cellbender/remove_background/train.py:    torch.cuda.empty_cache()
cellbender/remove_background/estimation.py:            device: ['cpu', 'cuda'] - whether to perform the pytorch sampling
cellbender/remove_background/estimation.py:                operation on CPU or GPU. It's pretty fast on CPU already.
cellbender/remove_background/estimation.py:            device: ['cpu', 'cuda'] - whether to perform the pytorch argmax
cellbender/remove_background/estimation.py:                operation on CPU or GPU. It's pretty fast on CPU already.
cellbender/remove_background/estimation.py:        device: ['cpu', 'cuda'] - whether to perform the pytorch sampling
cellbender/remove_background/estimation.py:            operation on CPU or GPU. It's pretty fast on CPU already.
cellbender/remove_background/checkpoint.py:USE_CUDA = torch.cuda.is_available()
cellbender/remove_background/checkpoint.py:    if USE_CUDA:
cellbender/remove_background/checkpoint.py:        cuda_random_state = torch.cuda.get_rng_state_all()
cellbender/remove_background/checkpoint.py:        file_dict.update({filebase + '_random.cuda': cuda_random_state})
cellbender/remove_background/checkpoint.py:    if USE_CUDA:
cellbender/remove_background/checkpoint.py:        with open(filebase + '_random.cuda', 'rb') as f:
cellbender/remove_background/checkpoint.py:            cuda_random_state = pickle.load(f)
cellbender/remove_background/checkpoint.py:        torch.cuda.set_rng_state_all(cuda_random_state)
cellbender/remove_background/checkpoint.py:            logger.debug('Loaded random state globally for python, numpy, pytorch, and cuda')
cellbender/remove_background/posterior.py:                    device='cuda',
cellbender/remove_background/posterior.py:                    device='cuda',
cellbender/remove_background/posterior.py:                    device='cuda',
cellbender/remove_background/posterior.py:        self.use_cuda = (torch.cuda.is_available() if vi_model is None
cellbender/remove_background/posterior.py:                         else vi_model.use_cuda)
cellbender/remove_background/posterior.py:        self.device = 'cuda' if self.use_cuda else 'cpu'
cellbender/remove_background/posterior.py:        torch.cuda.empty_cache()
cellbender/remove_background/posterior.py:            use_cuda=self.use_cuda,
cellbender/remove_background/posterior.py:                logger.debug('\n' + get_hardware_usage(use_cuda=self.use_cuda))
cellbender/remove_background/posterior.py:        data_loader = self.dataset_obj.get_dataloader(use_cuda=self.use_cuda,
cellbender/remove_background/posterior.py:                   device: str = 'cuda',
cellbender/remove_background/posterior.py:            device: Where to perform tensor operations: ['cuda', 'cpu']
cellbender/remove_background/posterior.py:                   device: str = 'cuda',
cellbender/remove_background/posterior.py:            device: Where to perform tensor operations: ['cuda', 'cpu']
cellbender/remove_background/posterior.py:        device: 'cpu' or 'cuda'
cellbender/monitor.py:# Inspiration for the nvidia-smi command comes from here:
cellbender/monitor.py:# https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/gpu_stats_monitor.html#GPUStatsMonitor
cellbender/monitor.py:def get_hardware_usage(use_cuda: bool) -> str:
cellbender/monitor.py:    """Get a current snapshot of RAM, CPU, GPU memory, and GPU utilization as a string"""
cellbender/monitor.py:    if use_cuda:
cellbender/monitor.py:        # Run nvidia-smi to get GPU utilization
cellbender/monitor.py:        gpu_query = 'utilization.gpu'
cellbender/monitor.py:            [shutil.which("nvidia-smi"), f"--query-gpu={gpu_query}", f"--format={format}"],
cellbender/monitor.py:        pct_gpu_util = result.stdout.strip()
cellbender/monitor.py:        gpu_string = (f'Volatile GPU utilization: {pct_gpu_util} %\n'
cellbender/monitor.py:                      f'GPU memory reserved: {torch.cuda.memory_reserved() / 1e9} GB\n'
cellbender/monitor.py:                      f'GPU memory allocated: {torch.cuda.memory_allocated() / 1e9} GB\n')
cellbender/monitor.py:        gpu_string = ''
cellbender/monitor.py:    return gpu_string + cpu_string
docker/Dockerfile:# Start from nvidia-docker image with drivers pre-installed to use a GPU
docker/Dockerfile:FROM nvcr.io/nvidia/cuda:11.7.1-base-ubuntu18.04
docker/DockerfileGit:# Start from nvidia-docker image with drivers pre-installed to use a GPU
docker/DockerfileGit:FROM nvcr.io/nvidia/cuda:11.7.1-base-ubuntu18.04
wdl/README.rst:The workflow that runs ``cellbender remove-background`` on a (Tesla T4) GPU on a
wdl/README.rst:* ``hardware_gpu_type``: Specify a `GPU type <https://cloud.google.com/compute/docs/gpus>`_
wdl/README.rst:  The chosen zones should have the appropriate GPU hardware available, otherwise this
wdl/README.rst:The cost to run a single sample in v0.3.0 on a preemptible Nvidia Tesla T4 GPU
wdl/cellbender_remove_background.wdl:task run_cellbender_remove_background_gpu {
wdl/cellbender_remove_background.wdl:        String? hardware_gpu_type = "nvidia-tesla-t4"
wdl/cellbender_remove_background.wdl:        String? nvidia_driver_version = "470.82.01"  # need >=465.19.01 for CUDA 11.3
wdl/cellbender_remove_background.wdl:            --cuda \
wdl/cellbender_remove_background.wdl:        gpuCount: 1
wdl/cellbender_remove_background.wdl:        gpuType: "${hardware_gpu_type}"
wdl/cellbender_remove_background.wdl:        nvidiaDriverVersion: "${nvidia_driver_version}"
wdl/cellbender_remove_background.wdl:        maxRetries: hardware_max_retries  # can be used in case of a PAPI error code 2 failure to install GPU drivers
wdl/cellbender_remove_background.wdl:        description: "WDL that runs CellBender remove-background on a GPU on Google Cloud hardware. See the [CellBender GitHub repo](https://github.com/broadinstitute/CellBender) and [read the documentation](https://cellbender.readthedocs.io/en/v0.3.0/reference/index.html#command-line-options) for more information."
wdl/cellbender_remove_background.wdl:        hardware_gpu_type :
wdl/cellbender_remove_background.wdl:            {help: "Specify the type of GPU that should be used.  Ensure that the selected hardware_zones have the GPU available.",
wdl/cellbender_remove_background.wdl:             suggestions: ["nvidia-tesla-t4", "nvidia-tesla-k80"]}
wdl/cellbender_remove_background.wdl:    call run_cellbender_remove_background_gpu
wdl/cellbender_remove_background.wdl:        File log = run_cellbender_remove_background_gpu.log
wdl/cellbender_remove_background.wdl:        File summary_pdf = run_cellbender_remove_background_gpu.pdf
wdl/cellbender_remove_background.wdl:        File cell_barcodes_csv = run_cellbender_remove_background_gpu.cell_csv
wdl/cellbender_remove_background.wdl:        Array[File] metrics_csv_array = run_cellbender_remove_background_gpu.metrics_array
wdl/cellbender_remove_background.wdl:        Array[File] html_report_array = run_cellbender_remove_background_gpu.report_array
wdl/cellbender_remove_background.wdl:        Array[File] h5_array = run_cellbender_remove_background_gpu.h5_array
wdl/cellbender_remove_background.wdl:        String output_directory = run_cellbender_remove_background_gpu.output_dir
wdl/cellbender_remove_background.wdl:        File checkpoint_file = run_cellbender_remove_background_gpu.ckpt_file
wdl/cellbender_remove_background_azure.wdl:task run_cellbender_remove_background_gpu {
wdl/cellbender_remove_background_azure.wdl:        String? hardware_gpu_type = "nvidia-tesla-t4"
wdl/cellbender_remove_background_azure.wdl:        String? nvidia_driver_version = "470.82.01"  # need >=465.19.01 for CUDA 11.3
wdl/cellbender_remove_background_azure.wdl:        maxRetries: hardware_max_retries  # can be used in case of a PAPI error code 2 failure to install GPU drivers
wdl/cellbender_remove_background_azure.wdl:        description: "WDL that runs CellBender remove-background on a GPU on Google Cloud hardware. See the [CellBender GitHub repo](https://github.com/broadinstitute/CellBender) and [read the documentation](https://cellbender.readthedocs.io/en/v0.3.0/reference/index.html#command-line-options) for more information."
wdl/cellbender_remove_background_azure.wdl:        hardware_gpu_type :
wdl/cellbender_remove_background_azure.wdl:            {help: "Specify the type of GPU that should be used.  Ensure that the selected hardware_zones have the GPU available.",
wdl/cellbender_remove_background_azure.wdl:             suggestions: ["nvidia-tesla-t4", "nvidia-tesla-k80"]}
wdl/cellbender_remove_background_azure.wdl:    call run_cellbender_remove_background_gpu
wdl/cellbender_remove_background_azure.wdl:        File log = run_cellbender_remove_background_gpu.log
wdl/cellbender_remove_background_azure.wdl:        File summary_pdf = run_cellbender_remove_background_gpu.pdf
wdl/cellbender_remove_background_azure.wdl:        File cell_barcodes_csv = run_cellbender_remove_background_gpu.cell_csv
wdl/cellbender_remove_background_azure.wdl:        Array[File] metrics_csv_array = run_cellbender_remove_background_gpu.metrics_array
wdl/cellbender_remove_background_azure.wdl:        Array[File] html_report_array = run_cellbender_remove_background_gpu.report_array
wdl/cellbender_remove_background_azure.wdl:        Array[File] h5_array = run_cellbender_remove_background_gpu.h5_array
wdl/cellbender_remove_background_azure.wdl:        String output_directory = run_cellbender_remove_background_gpu.output_dir
wdl/cellbender_remove_background_azure.wdl:        File checkpoint_file = run_cellbender_remove_background_gpu.ckpt_file

```
