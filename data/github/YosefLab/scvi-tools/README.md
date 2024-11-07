# https://github.com/scverse/scvi-tools

```console
Dockerfile:FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
Dockerfile:RUN uv pip install --system --no-cache torch torchvision torchaudio "jax[cuda12]"
docs/developer/maintenance.md:runners are used for workflows that require a GPU.
docs/developer/maintenance.md:- [`test (cuda)`](https://github.com/scverse/scvi-tools/blob/main/.github/workflows/test_linux_cuda.yml):
docs/developer/maintenance.md:  Same as the `test` workflow, but runs on a self-hosted runner with a GPU. This workflow is only
docs/developer/maintenance.md:  triggered on pull requests with the `cuda tests` or `all tests` labels, which can only be added
docs/developer/maintenance.md:[Docker image build]: https://github.com/YosefLab/scvi-tools-docker/actions/workflows/linux_cuda_manual.yaml
docs/developer/maintenance.md:[run the tutorials]: https://github.com/scverse/scvi-tutorials/actions/workflows/run_linux_cuda_branch.yml
docs/developer/maintenance.md:[release image workflow]: https://github.com/YosefLab/scvi-tools-docker/actions/workflows/linux_cuda_release.yaml
docs/user_guide/models/peakvi.md:-   Effectively requires a GPU for fast inference.
docs/user_guide/models/multivi.md:-   Effectively requires a GPU for fast inference.
docs/user_guide/models/methylvi.md:-   Effectively requires a GPU for fast inference.
docs/user_guide/models/scvi.md:-   Effectively requires a GPU for fast inference.
docs/user_guide/models/stereoscope.md:-   Effectively requires a GPU for fast inference.
docs/user_guide/models/scanvi.md:-   Effectively requires a GPU for fast inference.
docs/user_guide/models/totalvi.md:-   Effectively requires a GPU for fast inference.
docs/user_guide/models/destvi.md:-   Effectively requires a GPU for fast inference.
docs/user_guide/index.md:data to the correct device (CPU/GPU).
docs/installation.md:If you plan on taking advantage of an accelerated device (e.g. Nvidia GPU or Apple Silicon), we
CHANGELOG.md:- Fix JAX to be deterministic on CUDA when seed is manually set {pr}`2923`.
CHANGELOG.md:- Support fractional GPU usage in {class}`scvi.autotune.ModelTuner` `pr`{2252}.
CHANGELOG.md:- Remove deprecated `use_gpu` argument in favor of PyTorch Lightning arguments `accelerator` and
CHANGELOG.md:- Filter Jax undetected GPU warnings {pr}`2044`.
CHANGELOG.md:- Deprecate `use_gpu` in favor of PyTorch Lightning arguments `accelerator` and `devices`, to be
CHANGELOG.md:- CUDA compatibility issue fixed in {meth}`~scvi.distributions.ZeroInflatedNegativeBinomial.sample`
CHANGELOG.md:- New default is to not pin memory during training when using a GPU. This is much better for shared
CHANGELOG.md:    GPU environments without any performance regression [#1473].
CHANGELOG.md:    trained on cuda but method used on cpu; see [#1451]).
CHANGELOG.md:CPU, Jax on only a CPU can be as fast as PyTorch with a GPU (RTX3090). We will be planning further
CHANGELOG.md:- `use_cuda` is now `use_gpu` for consistency with PytorchLightning.
CHANGELOG.md:##### Breaking change: GPU handling
CHANGELOG.md:- `use_cuda` was removed from the init of each model and was not replaced by `use_gpu`. By default
CHANGELOG.md:    model is trained with `use_gpu=True` the model will remain on the GPU after training.
CHANGELOG.md:- When loading saved models, scvi-tools will always attempt to load the model on GPU unless
CHANGELOG.md:- We now support specifying which GPU device to use if there are multiple available GPUs.
tests/model/test_pyro.py:    # cpu/gpu has minor difference
tests/model/test_models_with_minified_data.py:    # Allclose because on GPU, the values are not exactly the same
README.md:high-level API that interacts with [Scanpy] and includes standard save/load functions, GPU
README.md:Please be sure to install a version of [PyTorch] that is compatible with your GPU (if applicable).
src/scvi/model/base/_training_mixin.py:            GPUs, depending on the sparsity of the data. Passed into
src/scvi/model/base/_save_load.py:    map_location: Literal["cpu", "cuda"] | None = None,
src/scvi/model/base/_save_load.py:        non_kwargs.pop("use_cuda")
src/scvi/model/base/_jaxmixin.py:                "Note: Pytorch lightning will show GPU is not being used for the Trainer."
src/scvi/model/base/_jaxmixin.py:            logger.debug("No GPU available to Jax.")
src/scvi/model/base/_pyromixin.py:    Training using minibatches and using full data (copies data to GPU only once).
src/scvi/model/base/_pyromixin.py:            data is copied to device (e.g., GPU).
src/scvi/model/base/_pyromixin.py:            # use data splitter which moves data to GPU once
src/scvi/model/base/_pyromixin.py:        # sample using minibatches (if full data, data is moved to GPU only once anyway)
src/scvi/model/base/_base_model.py:            Device to move model to. Options: 'cpu' for CPU, integer GPU index (eg. 0),
src/scvi/model/base/_base_model.py:            or 'cuda:X' where X is the GPU index (eg. 'cuda:0'). See torch.device for more info.
src/scvi/model/base/_base_model.py:        >>> model.to_device("cuda:0")  # moves model to GPU 0
src/scvi/model/base/_base_model.py:        >>> model.to_device(0)  # also moves model to GPU 0
src/scvi/model/utils/_mde.py:    This function is included in scvi-tools to provide an alternative to UMAP/TSNE that is GPU-
src/scvi/external/gimvi/_utils.py:    map_location: Literal["cpu", "cuda"] | None = None,
src/scvi/external/contrastivevi/_contrastive_data_splitting.py:        speedups in transferring data to GPUs, depending on the sparsity of the data.
src/scvi/external/contrastivevi/_model.py:            GPUs, depending on the sparsity of the data.
src/scvi/external/contrastivevi/_contrastive_dataloader.py:            transfers to GPUs, depending on the sparsity of the data. Not applicable
src/scvi/external/mrvi/_model.py:            Disabling vmap can be useful if running out of memory on a GPU.
src/scvi/external/mrvi/_model.py:            be useful if running out of memory on a GPU.
src/scvi/external/tangram/_model.py:                "Note: Pytorch lightning will show GPU is not being used for the Trainer."
src/scvi/external/tangram/_model.py:            logger.debug("No GPU available to Jax.")
src/scvi/external/poissonvi/_model.py:        """Return region-specific factors. CPU/GPU dependent"""
src/scvi/external/poissonvi/_model.py:            region_factors = self.module.decoder.px_scale_decoder[-2].bias.cpu().numpy()  # gpu
src/scvi/data/_manager.py:            GPUs, depending on the sparsity of the data.
src/scvi/data/_anntorchdataset.py:        GPUs, depending on the sparsity of the data.
src/scvi/data/_preprocessing.py:        requires more RAM or GPU memory. (The default should be fine unless
src/scvi/data/_preprocessing.py:        # Clean up memory (tensors seem to stay in GPU unless actively deleted).
src/scvi/_settings.py:    To prevent Jax from preallocating GPU memory on start (default)
src/scvi/_settings.py:    >>> scvi.settings.jax_preallocate_gpu_memory = False
src/scvi/_settings.py:        jax_preallocate_gpu_memory: bool = False,
src/scvi/_settings.py:        self.jax_preallocate_gpu_memory = jax_preallocate_gpu_memory
src/scvi/_settings.py:            # Ensure deterministic CUDA operations for Jax (see https://github.com/google/jax/issues/13672)
src/scvi/_settings.py:                os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
src/scvi/_settings.py:                os.environ["XLA_FLAGS"] += " --xla_gpu_deterministic_ops=true"
src/scvi/_settings.py:    def jax_preallocate_gpu_memory(self):
src/scvi/_settings.py:        """Jax GPU memory allocation settings.
src/scvi/_settings.py:        If False, Jax will ony preallocate GPU memory it needs.
src/scvi/_settings.py:        If float in (0, 1), Jax will preallocate GPU memory to that
src/scvi/_settings.py:        fraction of the GPU memory.
src/scvi/_settings.py:        return self._jax_gpu
src/scvi/_settings.py:    @jax_preallocate_gpu_memory.setter
src/scvi/_settings.py:    def jax_preallocate_gpu_memory(self, value: float | bool):
src/scvi/_settings.py:        # see https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html#gpu-memory-allocation
src/scvi/_settings.py:        self._jax_gpu = value
src/scvi/module/_autozivae.py:        # Problem : it is not implemented in CUDA yet
src/scvi/autotune/_experiment.py:        - ``"gpu"``: number of GPUs
src/scvi/autotune/_experiment.py:        resources: dict[Literal["cpu", "gpu", "memory"], float] | None = None,
src/scvi/autotune/_experiment.py:    def resources(self) -> dict[Literal["cpu", "gpu", "memory"], float] | None:
src/scvi/autotune/_experiment.py:    def resources(self, value: dict[Literal["cpu", "gpu", "memory"], float] | None) -> None:
src/scvi/autotune/_tune.py:    resources: dict[Literal["cpu", "gpu", "memory"], float] | None = None,
src/scvi/autotune/_tune.py:        - ``"gpu"``: number of GPUs
src/scvi/dataloaders/_ann_dataloader.py:        GPUs, depending on the sparsity of the data.
src/scvi/dataloaders/_data_splitting.py:        transferring data to GPUs, depending on the sparsity of the data.
src/scvi/dataloaders/_data_splitting.py:    """Creates loaders for data that is already on device, e.g., GPU.
src/scvi/utils/_docstrings.py:    Supports passing different accelerator types `("cpu", "gpu", "tpu", "ipu", "hpu",
src/scvi/utils/_jax.py:    """Returns a PRNGKey that is either on CPU or GPU."""
src/scvi/train/_trainer.py:        Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps,
src/scvi/train/_trainrunner.py:        Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu",

```
