# https://github.com/neuralhydrology/neuralhydrology

```console
docs/source/usage/config.rst:-  ``device``: Which device to use in format of ``cuda:0``, ``cuda:1``,
docs/source/usage/config.rst:   etc, for GPUs or ``cpu``
docs/source/usage/config.rst:   e.g., [``cudalstm``, ``ealstm``, ``mtslstm``]
docs/source/usage/models.rst:cell, but only runs the optimized LSTM one timestep at a time, and is therefore significantly slower than the CudaLSTM.  
docs/source/usage/models.rst:CudaLSTM
docs/source/usage/models.rst::py:class:`neuralhydrology.modelzoo.cudalstm.CudaLSTM` is a network using the standard PyTorch LSTM implementation.
docs/source/usage/models.rst::py:class:`neuralhydrology.modelzoo.customlstm.CustomLSTM` is a variant of the ``CudaLSTM``
docs/source/usage/models.rst:reasons. You can use the method ``model.copy_weights()`` to copy the weights of a ``CudaLSTM`` model
docs/source/usage/models.rst:into a ``CustomLSTM`` model. This allows to use the fast CUDA implementations for training, and only use this class for
docs/source/usage/models.rst:EmbCudaLSTM
docs/source/usage/models.rst:   Use `CudaLSTM`_ with ``statics_embedding``.
docs/source/usage/models.rst::py:class:`neuralhydrology.modelzoo.embcudalstm.EmbCudaLSTM` is similar to `CudaLSTM`_,
docs/source/usage/models.rst:out through both the hindcast and forecast sequences. The difference between this and a standard `CudaLSTM`_ is (1) this model uses both hindcast and forecast 
docs/source/usage/quickstart.rst:If you don't have a CUDA-capable GPU, use:
docs/source/usage/quickstart.rst:If you do have a CUDA-capable GPU, use ``environment_cuda11_8.yml``, depending on your hardware.
docs/source/usage/quickstart.rst:    nh-schedule-runs train --directory /path/to/config_dir/ --runs-per-gpu X --gpu-ids Y
docs/source/usage/quickstart.rst:With X, you can specify how many models should be trained on parallel on a single GPU.
docs/source/usage/quickstart.rst:With Y, you can specify which GPUs to use for training (use the id as specified in ``nvidia-smi``).
docs/source/usage/quickstart.rst:    nh-schedule-runs evaluate --directory /path/to/config_dir/ --runs-per-gpu X --gpu-ids Y
docs/source/api/neuralhydrology.modelzoo.rst:   neuralhydrology.modelzoo.cudalstm
docs/source/api/neuralhydrology.modelzoo.rst:   neuralhydrology.modelzoo.embcudalstm
docs/source/api/neuralhydrology.modelzoo.cudalstm.rst:CudaLSTM
docs/source/api/neuralhydrology.modelzoo.cudalstm.rst:.. automodule:: neuralhydrology.modelzoo.cudalstm
docs/source/api/neuralhydrology.modelzoo.embcudalstm.rst:EmbCudaLSTM
docs/source/api/neuralhydrology.modelzoo.embcudalstm.rst:.. automodule:: neuralhydrology.modelzoo.embcudalstm
test/test_configs/daily_regression_nan_targets.test.yml:model: cudalstm
test/test_configs/daily_regression.test.yml:model: cudalstm
test/test_configs/daily_regression_with_embedding.test.yml:model: cudalstm
test/test_configs/daily_regression_additional_features.test.yml:model: cudalstm
test/test_custom_lstm.py:"""Test for checking that the outputs of the CustomLSTM match those of CudaLSTM and EmbCudaLSTM"""
test/conftest.py:                     'models and forcings, only test cudalstm on forcings that include daymet.')
test/conftest.py:@pytest.fixture(params=['customlstm', 'ealstm', 'cudalstm', 'gru'])
test/conftest.py:    if request.config.getoption('--smoke-test') and request.param != 'cudalstm':
test/conftest.py:@pytest.fixture(params=["cudalstm"])
neuralhydrology/evaluation/tester.py:            if "cuda" in self.cfg.device:
neuralhydrology/evaluation/tester.py:                gpu_id = int(self.cfg.device.split(':')[-1])
neuralhydrology/evaluation/tester.py:                if gpu_id > torch.cuda.device_count():
neuralhydrology/evaluation/tester.py:                    raise RuntimeError(f"This machine does not have GPU #{gpu_id} ")
neuralhydrology/evaluation/tester.py:            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
neuralhydrology/nh_run.py:    parser.add_argument('--gpu', type=int,
neuralhydrology/nh_run.py:                        help="GPU id to use. Overrides config argument 'device'. Use a value < 0 for CPU.")
neuralhydrology/nh_run.py:        start_run(config_file=Path(args["config_file"]), gpu=args["gpu"])
neuralhydrology/nh_run.py:                     gpu=args["gpu"])
neuralhydrology/nh_run.py:        finetune(config_file=Path(args["config_file"]), gpu=args["gpu"])
neuralhydrology/nh_run.py:        eval_run(run_dir=Path(args["run_dir"]), period=args["period"], epoch=args["epoch"], gpu=args["gpu"])
neuralhydrology/nh_run.py:def start_run(config_file: Path, gpu: int = None):
neuralhydrology/nh_run.py:    gpu : int, optional
neuralhydrology/nh_run.py:        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.
neuralhydrology/nh_run.py:    # check if a GPU has been specified as command line argument. If yes, overwrite config
neuralhydrology/nh_run.py:    if gpu is not None and gpu >= 0:
neuralhydrology/nh_run.py:        config.device = f"cuda:{gpu}"
neuralhydrology/nh_run.py:    if gpu is not None and gpu < 0:
neuralhydrology/nh_run.py:def continue_run(run_dir: Path, config_file: Path = None, gpu: int = None):
neuralhydrology/nh_run.py:    gpu : int, optional
neuralhydrology/nh_run.py:        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.
neuralhydrology/nh_run.py:    # check if a GPU has been specified as command line argument. If yes, overwrite config
neuralhydrology/nh_run.py:    if gpu is not None and gpu >= 0:
neuralhydrology/nh_run.py:        base_config.device = f"cuda:{gpu}"
neuralhydrology/nh_run.py:    if gpu is not None and gpu < 0:
neuralhydrology/nh_run.py:def finetune(config_file: Path = None, gpu: int = None):
neuralhydrology/nh_run.py:    gpu : int, optional
neuralhydrology/nh_run.py:        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.
neuralhydrology/nh_run.py:    # check if a GPU has been specified as command line argument. If yes, overwrite config
neuralhydrology/nh_run.py:    if gpu is not None and gpu >= 0:
neuralhydrology/nh_run.py:        config.device = f"cuda:{gpu}"
neuralhydrology/nh_run.py:    if gpu is not None and gpu < 0:
neuralhydrology/nh_run.py:def eval_run(run_dir: Path, period: str, epoch: int = None, gpu: int = None):
neuralhydrology/nh_run.py:    gpu : int, optional
neuralhydrology/nh_run.py:        GPU id to use. Will override config argument 'device'. A value less than zero indicates CPU.
neuralhydrology/nh_run.py:    # check if a GPU has been specified as command line argument. If yes, overwrite config
neuralhydrology/nh_run.py:    if gpu is not None and gpu >= 0:
neuralhydrology/nh_run.py:        config.device = f"cuda:{gpu}"
neuralhydrology/nh_run.py:    if gpu is not None and gpu < 0:
neuralhydrology/training/basetrainer.py:        torch.cuda.manual_seed(self.cfg.seed)
neuralhydrology/training/basetrainer.py:            if self.cfg.device.startswith("cuda"):
neuralhydrology/training/basetrainer.py:                gpu_id = int(self.cfg.device.split(':')[-1])
neuralhydrology/training/basetrainer.py:                if gpu_id > torch.cuda.device_count():
neuralhydrology/training/basetrainer.py:                    raise RuntimeError(f"This machine does not have GPU #{gpu_id} ")
neuralhydrology/training/basetrainer.py:            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
neuralhydrology/nh_run_scheduler.py:    parser.add_argument('--gpu-ids', type=int, nargs='+', required=True)
neuralhydrology/nh_run_scheduler.py:    parser.add_argument('--runs-per-gpu', type=int, required=True)
neuralhydrology/nh_run_scheduler.py:def schedule_runs(mode: str, directory: Path, gpu_ids: List[int], runs_per_gpu: int):
neuralhydrology/nh_run_scheduler.py:    """Schedule multiple runs across one or multiple GPUs.
neuralhydrology/nh_run_scheduler.py:    gpu_ids : List[int]
neuralhydrology/nh_run_scheduler.py:        List of GPU ids to use for training/evaluating.
neuralhydrology/nh_run_scheduler.py:    runs_per_gpu : int
neuralhydrology/nh_run_scheduler.py:        Number of runs to start on a single GPU.
neuralhydrology/nh_run_scheduler.py:    # array to keep track on how many runs are currently running per GPU
neuralhydrology/nh_run_scheduler.py:    n_parallel_runs = len(gpu_ids) * runs_per_gpu
neuralhydrology/nh_run_scheduler.py:    gpu_counter = np.zeros((len(gpu_ids)), dtype=int)
neuralhydrology/nh_run_scheduler.py:            # determine which GPU to use
neuralhydrology/nh_run_scheduler.py:            node_id = np.argmin(gpu_counter)
neuralhydrology/nh_run_scheduler.py:            gpu_counter[node_id] += 1
neuralhydrology/nh_run_scheduler.py:            gpu_id = gpu_ids[node_id]
neuralhydrology/nh_run_scheduler.py:                run_command = f"python {script_path} {mode} --config-file {process} --gpu {gpu_id}"
neuralhydrology/nh_run_scheduler.py:                run_command = f"python {script_path} evaluate --run-dir {process} --gpu {gpu_id}"
neuralhydrology/nh_run_scheduler.py:                gpu_counter[key[1]] -= 1
neuralhydrology/utils/config.py:        if device == "cpu" or device.startswith("cuda:"):
neuralhydrology/utils/config.py:            raise ValueError("'device' must be either 'cpu' or a 'cuda:X', with 'X' being the GPU ID.")
neuralhydrology/modelzoo/basemodel.py:    Use subclasses of this class for training/evaluating different models, e.g. use `CudaLSTM` for training a standard
neuralhydrology/modelzoo/cudalstm.py:class CudaLSTM(BaseModel):
neuralhydrology/modelzoo/cudalstm.py:    """LSTM model class, which relies on PyTorch's CUDA LSTM class.
neuralhydrology/modelzoo/cudalstm.py:    The `CudaLSTM` class only supports single-timescale predictions. Use `MTSLSTM` to train a model and get
neuralhydrology/modelzoo/cudalstm.py:        super(CudaLSTM, self).__init__(cfg=cfg)
neuralhydrology/modelzoo/cudalstm.py:        """Perform a forward pass on the CudaLSTM model.
neuralhydrology/modelzoo/customlstm.py:from neuralhydrology.modelzoo.cudalstm import CudaLSTM
neuralhydrology/modelzoo/customlstm.py:from neuralhydrology.modelzoo.embcudalstm import EmbCudaLSTM
neuralhydrology/modelzoo/customlstm.py:    `CudaLSTM` or `EmbCudaLSTM` classes, and later copy the weights into this model for a more in-depth network
neuralhydrology/modelzoo/customlstm.py:    >>> # Example for copying the weights of an optimzed `CudaLSTM` or `EmbCudaLSTM` into a `CustomLSTM` instance
neuralhydrology/modelzoo/customlstm.py:    >>> optimized_lstm = ... # A model instance of `CudaLSTM` or `EmbCudaLSTM`
neuralhydrology/modelzoo/customlstm.py:    >>> # Use the original config to initialize this model to differentiate between `CudaLSTM` and `EmbCudaLSTM`
neuralhydrology/modelzoo/customlstm.py:    def copy_weights(self, optimized_lstm: Union[CudaLSTM, EmbCudaLSTM]):
neuralhydrology/modelzoo/customlstm.py:        """Copy weights from a `CudaLSTM` or `EmbCudaLSTM` into this model class
neuralhydrology/modelzoo/customlstm.py:        optimized_lstm : Union[CudaLSTM, EmbCudaLSTM]
neuralhydrology/modelzoo/customlstm.py:            Model instance of a `CudaLSTM` (neuralhydrology.modelzoo.cudalstm) or `EmbCudaLSTM`
neuralhydrology/modelzoo/customlstm.py:            (neuralhydrology.modelzoo.embcudalstm).
neuralhydrology/modelzoo/customlstm.py:            If `optimized_lstm` is an `EmbCudaLSTM` but this model instance was not created with an embedding network.
neuralhydrology/modelzoo/customlstm.py:        assert isinstance(optimized_lstm, CudaLSTM) or isinstance(optimized_lstm, EmbCudaLSTM)
neuralhydrology/modelzoo/mamba.py:        """Perform a forward pass on the CudaLSTM model.
neuralhydrology/modelzoo/sequential_forecast_lstm.py:    and a standard ``CudaLSTM`` is (1) this model uses both hindcast and forecast
neuralhydrology/modelzoo/embcudalstm.py:from neuralhydrology.modelzoo.cudalstm import CudaLSTM
neuralhydrology/modelzoo/embcudalstm.py:class EmbCudaLSTM(BaseModel):
neuralhydrology/modelzoo/embcudalstm.py:    """EmbCudaLSTM model class, which adds embedding networks for static inputs to the standard LSTM.
neuralhydrology/modelzoo/embcudalstm.py:       Use :py:class:`neuralhydrology.modelzoo.cudalstm.CudaLSTM` with ``statics_embedding``.
neuralhydrology/modelzoo/embcudalstm.py:    This class extends the standard `CudaLSTM` class to preprocess the static inputs by an embedding network, prior
neuralhydrology/modelzoo/embcudalstm.py:    The `EmbCudaLSTM` class only supports single timescale predictions. Use `MTSLSTM` to train a model and get
neuralhydrology/modelzoo/embcudalstm.py:        super(EmbCudaLSTM, self).__init__(cfg=cfg)
neuralhydrology/modelzoo/embcudalstm.py:        warnings.warn('EmbCudaLSTM is deprecated, the functionality is now part of CudaLSTM.', FutureWarning)
neuralhydrology/modelzoo/embcudalstm.py:        self.cudalstm = CudaLSTM(cfg)
neuralhydrology/modelzoo/embcudalstm.py:        self.embedding_net = self.cudalstm.embedding_net
neuralhydrology/modelzoo/embcudalstm.py:        self.lstm = self.cudalstm.lstm
neuralhydrology/modelzoo/embcudalstm.py:        self.head = self.cudalstm.head
neuralhydrology/modelzoo/embcudalstm.py:        """Perform a forward pass on the EmbCudaLSTM model.
neuralhydrology/modelzoo/embcudalstm.py:        return self.cudalstm.forward(data)
neuralhydrology/modelzoo/__init__.py:from neuralhydrology.modelzoo.cudalstm import CudaLSTM
neuralhydrology/modelzoo/__init__.py:from neuralhydrology.modelzoo.embcudalstm import EmbCudaLSTM
neuralhydrology/modelzoo/__init__.py:    "cudalstm",
neuralhydrology/modelzoo/__init__.py:    "embcudalstm", 
neuralhydrology/modelzoo/__init__.py:    elif cfg.model.lower() == "cudalstm":
neuralhydrology/modelzoo/__init__.py:        model = CudaLSTM(cfg=cfg)
neuralhydrology/modelzoo/__init__.py:    elif cfg.model.lower() == "embcudalstm":
neuralhydrology/modelzoo/__init__.py:        model = EmbCudaLSTM(cfg=cfg)
environments/environment_cuda11_8.yml:  - nvidia
environments/environment_cuda11_8.yml:  - pytorch-cuda=11.8
examples/05-Inspecting-LSTMs/1_basin.yml:# which GPU (id) to use [in format of cuda:0, cuda:1 etc, or cpu or None]
examples/05-Inspecting-LSTMs/1_basin.yml:# base model type [lstm, ealstm, cudalstm, embcudalstm, mtslstm]
examples/05-Inspecting-LSTMs/1_basin.yml:model: cudalstm
examples/config.yml.example:# which GPU (id) to use [in format of cuda:0, cuda:1 etc, or cpu or None]
examples/config.yml.example:device: cuda:0
examples/config.yml.example:# base model type [cudalstm, customlstm, ealstm, embcudalstm, mtslstm, gru, transformer]
examples/config.yml.example:model: cudalstm
examples/01-Introduction/1_basin.yml:# which GPU (id) to use [in format of cuda:0, cuda:1 etc, or cpu or None]
examples/01-Introduction/1_basin.yml:device: cuda:0
examples/01-Introduction/1_basin.yml:# base model type [lstm, ealstm, cudalstm, embcudalstm, mtslstm]
examples/01-Introduction/1_basin.yml:model: cudalstm
examples/06-Finetuning/531_basins.yml:experiment_name: cudalstm_531_basins
examples/06-Finetuning/531_basins.yml:# which GPU (id) to use [in format of cuda:0, cuda:1 etc, or cpu or None]
examples/06-Finetuning/531_basins.yml:device: cuda:0
examples/06-Finetuning/531_basins.yml:# base model type [lstm, ealstm, cudalstm, embcudalstm, shortcutlstm, dropoutlstm, cudalstminitialh]
examples/06-Finetuning/531_basins.yml:model: cudalstm
examples/06-Finetuning/finetune.yml:experiment_name: cudalstm_531_basins_finetuned
examples/04-Multi-Timescale/1_basin.yml:# which GPU (id) to use [in format of cuda:0, cuda:1 etc, or cpu or None]
examples/04-Multi-Timescale/1_basin.yml:# base model type [lstm, ealstm, cudalstm, embcudalstm, mtslstm]

```
