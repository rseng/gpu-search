# https://github.com/sbi-dev/sbi

```console
sbi/inference/trainers/npse/npse.py:                the training is happening. If training a large dataset on a GPU with not
sbi/inference/trainers/npe/npe_a.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
sbi/inference/trainers/npe/npe_base.py:                the training is happening. If training a large dataset on a GPU with not
sbi/inference/trainers/npe/npe_c.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
sbi/inference/trainers/nle/nle_a.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
sbi/inference/trainers/nle/mnle.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
sbi/inference/trainers/nle/nle_base.py:                the training is happening. If training a large dataset on a GPU with not
sbi/inference/trainers/nre/nre_b.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
sbi/inference/trainers/nre/bnre.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
sbi/inference/trainers/nre/nre_c.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
sbi/inference/trainers/nre/nre_base.py:                the training is happening. If training a large dataset on a GPU with not
sbi/inference/trainers/nre/nre_a.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
sbi/inference/trainers/base.py:                perform all posterior operations, e.g. gpu or cpu.
sbi/inference/trainers/base.py:                the training is happening. If training a large dataset on a GPU with not
sbi/inference/posteriors/direct_posterior.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
sbi/inference/posteriors/base_posterior.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
sbi/inference/posteriors/mcmc_posterior.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
sbi/inference/posteriors/score_posterior.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
sbi/inference/posteriors/vi_posterior.py:            device: Training device, e.g., `cpu`, `cuda` or `cuda:0`. We will ensure
sbi/inference/posteriors/importance_posterior.py:            device: Device on which to sample, e.g., "cpu", "cuda" or "cuda:0". If
sbi/inference/posteriors/rejection_posterior.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
sbi/utils/sbiutils.py:    torch.cuda.manual_seed(seed)
sbi/utils/torchutils.py:    """Set and return the default device to cpu or gpu (cuda, mps).
sbi/utils/torchutils.py:        device: processed string, e.g., "cuda" is mapped to "cuda:0".
sbi/utils/torchutils.py:        # If user just passes 'gpu', search for CUDA or MPS.
sbi/utils/torchutils.py:        if device == "gpu":
sbi/utils/torchutils.py:            # check whether either pytorch cuda or mps is available
sbi/utils/torchutils.py:            if torch.cuda.is_available():
sbi/utils/torchutils.py:                current_gpu_index = torch.cuda.current_device()
sbi/utils/torchutils.py:                device = f"cuda:{current_gpu_index}"
sbi/utils/torchutils.py:                torch.cuda.set_device(device)
sbi/utils/torchutils.py:                    "Neither CUDA nor MPS is available. "
sbi/utils/torchutils.py:                    "CUDA or MPS."
sbi/utils/torchutils.py:def gpu_available() -> bool:
sbi/utils/torchutils.py:    """Check whether GPU is available."""
sbi/utils/torchutils.py:    return torch.cuda.is_available() or torch.backends.mps.is_available()
sbi/utils/torchutils.py:             corresponding device string. It should be something like 'cuda',
sbi/utils/torchutils.py:             'cuda:0', or 'mps'. Error message: {exc}."""
sbi/utils/torchutils.py:            f"'{training_device}'. When training on GPU make sure to "
sbi/utils/torchutils.py:            "pass a prior initialized on the GPU as well, e.g., "
sbi/utils/torchutils.py:            "(torch.zeros(2, device='cuda'), scale=1.0)`."
docs/docs/install.md:`sbi` requires Python 3.8 or higher. A GPU is not required, but can lead to
docs/docs/contribute.md:You can exclude slow tests and those which require a GPU with
docs/docs/contribute.md:pytest -m "not slow and not gpu"
docs/docs/contribute.md:pytest -n auto -m "not slow and not gpu"
docs/docs/contribute.md:in parallel. GPU tests should probably not be run this way. If you see unexpected
docs/docs/faq.md:4. [Can I use the GPU for training the density estimator?](faq/question_04_gpu.md)
docs/docs/faq/question_04_gpu.md:# Can I use the GPU for training the density estimator?
docs/docs/faq/question_04_gpu.md:**TLDR**; Yes, by passing `device="cuda"` and by passing a prior that lives on the
docs/docs/faq/question_04_gpu.md:Yes, we support GPU training. When creating the inference object in the flexible
docs/docs/faq/question_04_gpu.md:inference = NPE(prior, device="cuda", density_estimator="maf")
docs/docs/faq/question_04_gpu.md:as it maps to an existing PyTorch GPU device, e.g., `device="cuda"` or
docs/docs/faq/question_04_gpu.md:`device="cuda:2"`. `sbi` will take care of copying the `net` and the training
docs/docs/faq/question_04_gpu.md:We also support MPS as a GPU device for GPU-accelarated training on an Apple
docs/docs/faq/question_04_gpu.md:`device="cuda:0"`, make sure to pass a prior object that was created on that
docs/docs/faq/question_04_gpu.md:device="cuda:0"), covariance_matrix=torch.eye(2, device="cuda:0"))
docs/docs/faq/question_04_gpu.md:Whether or not you reduce your training time when training on a GPU depends on
docs/docs/faq/question_04_gpu.md:operations that benefit from being executed on the GPU.
docs/docs/faq/question_04_gpu.md:A speed-up through training on the GPU will most likely become visible when
docs/docs/faq/question_05_pickling.md:## I trained a model on a GPU. Can I load it on a CPU?
docs/docs/faq/question_05_pickling.md:saved on a GPU. Note that the neural net also needs to be moved to CPU.
docs/docs/faq/question_05_pickling.md:#https://stackoverflow.com/questions/57081727/load-pickle-file-obtained-from-gpu-to-cpu
docs/docs/faq/question_05_pickling.md:training on CPU for an inference object trained on a GPU is currently not
CHANGELOG.md:- support Apple MPS as gpu device (#912) (@janfb)
CHANGELOG.md:- bugfix for SNPE with implicit prior on GPU (#730)
CHANGELOG.md:- improved device check to remove several GPU issues (#610, thanks to @LouisRouillard)
CHANGELOG.md:- fix GPU issues for `conditional_pairplot` and `ActiveSubspace` (#613)
CHANGELOG.md:- Prior is now allowed to lie on GPU. The prior has to be on the same device as the one
CHANGELOG.md:- Bugfix for GPU training with SNRE_A (thanks @glouppe, #442).
CHANGELOG.md:- Support for training and sampling on GPU including fixes from `nflows` (#331)
tests/torchutils_test.py:@pytest.mark.parametrize("device_input", ("cpu", "gpu", "cuda", "cuda:0", "mps"))
tests/torchutils_test.py:        elif device_input == "gpu":
tests/torchutils_test.py:            if torch.cuda.is_available():
tests/torchutils_test.py:                current_gpu_index = torch.cuda.current_device()
tests/torchutils_test.py:                assert device_output == f"cuda:{current_gpu_index}"
tests/torchutils_test.py:        if device_input.startswith("cuda") and torch.cuda.is_available():
tests/torchutils_test.py:        # should only happen if no gpu is available
tests/torchutils_test.py:        if device_input == "gpu":
tests/torchutils_test.py:            assert not torchutils.gpu_available()
tests/test_utils.py:            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
tests/mnle_test.py:@pytest.mark.gpu
tests/mnle_test.py:@pytest.mark.parametrize("device", ("cpu", "gpu"))
tests/inference_on_device_test.py:from sbi.utils.torchutils import BoxUniform, gpu_available, process_device
tests/inference_on_device_test.py:# tests in this file are skipped if there is GPU device available
tests/inference_on_device_test.py:    not gpu_available(), reason="No CUDA or MPS device available."
tests/inference_on_device_test.py:@pytest.mark.gpu
tests/inference_on_device_test.py:        ("gpu", "gpu"),
tests/inference_on_device_test.py:        pytest.param("cpu", "gpu", marks=pytest.mark.xfail),
tests/inference_on_device_test.py:        pytest.param("gpu", "cpu", marks=pytest.mark.xfail),
tests/inference_on_device_test.py:@pytest.mark.gpu
tests/inference_on_device_test.py:@pytest.mark.parametrize("training_device", ["cpu", "gpu"])
tests/inference_on_device_test.py:@pytest.mark.parametrize("data_device", ["cpu", "gpu"])
tests/inference_on_device_test.py:@pytest.mark.gpu
tests/inference_on_device_test.py:@pytest.mark.parametrize("data_device", ("cpu", "gpu"))
tests/inference_on_device_test.py:@pytest.mark.parametrize("training_device", ("cpu", "gpu"))
tests/inference_on_device_test.py:@pytest.mark.parametrize("embedding_device", ("cpu", "gpu"))
tests/inference_on_device_test.py:@pytest.mark.gpu
tests/inference_on_device_test.py:def test_vi_on_gpu(num_dim: int, q: str, vi_method: str, sampling_method: str):
tests/inference_on_device_test.py:    device = process_device("gpu")
tests/inference_on_device_test.py:@pytest.mark.gpu
tests/inference_on_device_test.py:        ("gpu", None),
tests/inference_on_device_test.py:        ("gpu", "gpu"),
tests/inference_on_device_test.py:        pytest.param("gpu", "cpu", marks=pytest.mark.xfail),
tests/inference_on_device_test.py:        pytest.param("cpu", "gpu", marks=pytest.mark.xfail),
tests/score_estimator_test.py:@pytest.mark.gpu
tests/score_estimator_test.py:@pytest.mark.parametrize("device", ["cpu", "cuda"])
tests/analysis_test.py:@pytest.mark.gpu
tests/analysis_test.py:@pytest.mark.parametrize("device", ["cpu", "gpu"])
tests/analysis_test.py:    """Tests sensitivity analysis and conditional posterior utils on GPU and CPU.
tests/conftest.py:# Pytest hook to skip GPU tests if no devices are available.
tests/conftest.py:    """Skip GPU tests if no devices are available."""
tests/conftest.py:    gpu_device_available = (
tests/conftest.py:        torch.cuda.is_available() or torch.backends.mps.is_available()
tests/conftest.py:    if not gpu_device_available:
tests/conftest.py:        skip_gpu = pytest.mark.skip(reason="No devices available")
tests/conftest.py:            if "gpu" in item.keywords:
tests/conftest.py:                item.add_marker(skip_gpu)
README.md:`sbi` requires Python 3.9 or higher. While a GPU isn't necessary, it can improve
pyproject.toml:    "gpu: marks tests that require a gpu (deselect with '-m \"not gpu\"')",

```
