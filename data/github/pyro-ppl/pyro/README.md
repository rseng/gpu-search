# https://github.com/pyro-ppl/pyro

```console
setup.cfg:    ignore:CUDA initialization:UserWarning
pyro/infer/mcmc/api.py:  - minimal memory consumption with multiprocessing and CUDA.
pyro/infer/mcmc/api.py:        CUDA.
pyro/infer/mcmc/api.py:                # CUDA initialization error https://github.com/pytorch/pytorch/issues/2517
pyro/infer/mcmc/api.py:                    # change multiprocessing context to 'spawn' for CUDA tensors.
pyro/infer/mcmc/api.py:                    if list(initial_params.values())[0].is_cuda:
pyro/infer/mcmc/api.py:            # XXX we clone CUDA tensor args to resolve the issue "Invalid device pointer"
pyro/infer/mcmc/api.py:    about specific statistics (especially useful in a memory constrained environments like GPU).
pyro/infer/tracetmc_elbo.py:    [1] `Tensor Monte Carlo: Particle Methods for the GPU Era`,
pyro/infer/util.py:        "Tensor Monte Carlo: particle methods for the GPU era"
pyro/primitives.py:    :param bool use_cuda: DEPRECATED, use the `device` arg instead.
pyro/primitives.py:        Optional bool specifying whether to use cuda tensors for `subsample`
pyro/primitives.py:        and `log_prob`. Defaults to ``torch.Tensor.is_cuda``.
pyro/contrib/gp/parameterized.py:    tensor type). To cast these parameters to a correct data type or GPU device,
pyro/contrib/gp/parameterized.py:    :meth:`~torch.nn.Module.cuda`. See :class:`torch.nn.Module` for more
pyro/contrib/funsor/handlers/plate_messenger.py:        use_cuda=None,
pyro/contrib/funsor/handlers/plate_messenger.py:            name, size, subsample_size, subsample, use_cuda, device
pyro/contrib/funsor/handlers/__init__.py:    use_cuda=None,
pyro/contrib/epidemiology/compartmental.py:        num_samples=10000, num_chains=2)`` (GPU recommended).
pyro/contrib/mue/models.py:    :param bool cuda: Transfer data onto the GPU during training.
pyro/contrib/mue/models.py:    :param bool pin_memory: Pin memory for faster GPU transfer.
pyro/contrib/mue/models.py:        cuda=False,
pyro/contrib/mue/models.py:        assert isinstance(cuda, bool)
pyro/contrib/mue/models.py:        self.is_cuda = cuda
pyro/contrib/mue/models.py:        if self.is_cuda:
pyro/contrib/mue/models.py:            device = torch.device("cuda")
pyro/contrib/mue/models.py:                if self.is_cuda:
pyro/contrib/mue/models.py:                    seq_data = seq_data.cuda()
pyro/contrib/mue/models.py:                if self.is_cuda:
pyro/contrib/mue/models.py:                    seq_data, L_data = seq_data.cuda(), L_data.cuda()
pyro/contrib/mue/models.py:    :param bool cuda: Transfer data onto the GPU during training.
pyro/contrib/mue/models.py:    :param bool pin_memory: Pin memory for faster GPU transfer.
pyro/contrib/mue/models.py:        cuda=False,
pyro/contrib/mue/models.py:        assert isinstance(cuda, bool)
pyro/contrib/mue/models.py:        self.is_cuda = cuda
pyro/contrib/mue/models.py:        if self.is_cuda:
pyro/contrib/mue/models.py:            device = torch.device("cuda")
pyro/contrib/mue/models.py:            if self.is_cuda:
pyro/contrib/mue/models.py:                seq_data = seq_data.cuda()
pyro/contrib/mue/models.py:                if self.is_cuda:
pyro/contrib/mue/models.py:                    seq_data = seq_data.cuda()
pyro/contrib/mue/models.py:            if self.is_cuda:
pyro/contrib/mue/models.py:                seq_data = seq_data.cuda()
pyro/contrib/mue/models.py:                if self.is_cuda:
pyro/contrib/mue/models.py:                    seq_data, L_data = seq_data.cuda(), L_data.cuda()
pyro/contrib/mue/models.py:                if self.is_cuda:
pyro/contrib/mue/models.py:                    seq_data = seq_data.cuda()
pyro/contrib/examples/polyphonic_data_loader.py:def get_mini_batch(mini_batch_indices, sequences, seq_lengths, cuda=False):
pyro/contrib/examples/polyphonic_data_loader.py:    # cuda() here because need to cuda() before packing
pyro/contrib/examples/polyphonic_data_loader.py:    if cuda:
pyro/contrib/examples/polyphonic_data_loader.py:        mini_batch = mini_batch.cuda()
pyro/contrib/examples/polyphonic_data_loader.py:        mini_batch_mask = mini_batch_mask.cuda()
pyro/contrib/examples/polyphonic_data_loader.py:        mini_batch_reversed = mini_batch_reversed.cuda()
pyro/contrib/examples/scanvi_data.py:def get_data(dataset="pbmc", batch_size=100, cuda=False):
pyro/contrib/examples/scanvi_data.py:        if cuda:
pyro/contrib/examples/scanvi_data.py:            X, Y = X.cuda(), Y.cuda()
pyro/contrib/examples/scanvi_data.py:    if cuda:
pyro/contrib/examples/scanvi_data.py:        X, Y = X.cuda(), Y.cuda()
pyro/util.py:    Sets seeds of `torch` and `torch.cuda` (if available).
pyro/poutine/subsample_messenger.py:        use_cuda: Optional[bool] = None,
pyro/poutine/subsample_messenger.py:        :param bool use_cuda: DEPRECATED, use the `device` arg instead.
pyro/poutine/subsample_messenger.py:            Whether to use cuda tensors.
pyro/poutine/subsample_messenger.py:        self.use_cuda = use_cuda
pyro/poutine/subsample_messenger.py:        if self.use_cuda is not None:
pyro/poutine/subsample_messenger.py:            if self.use_cuda ^ (device != "cpu"):
pyro/poutine/subsample_messenger.py:                    "Incompatible arg values use_cuda={}, device={}.".format(
pyro/poutine/subsample_messenger.py:                        use_cuda, device
pyro/poutine/subsample_messenger.py:        return result.cuda() if self.use_cuda else result
pyro/poutine/subsample_messenger.py:        return result.cuda() if self.use_cuda else result
pyro/poutine/subsample_messenger.py:        use_cuda: Optional[bool] = None,
pyro/poutine/subsample_messenger.py:            use_cuda,
pyro/poutine/subsample_messenger.py:        use_cuda: Optional[bool] = None,
pyro/poutine/subsample_messenger.py:                fn=_Subsample(size, subsample_size, use_cuda, device),
pyro/distributions/spanning_tree.py:        if edge_logits.is_cuda:
pyro/distributions/spanning_tree.py:            raise NotImplementedError("SpanningTree does not support cuda tensors")
tests/infer/mcmc/test_mcmc_api.py:    mp_context = "spawn" if "CUDA_TEST" in os.environ else None
tests/infer/mcmc/test_hmc.py:    "CUDA_TEST" in os.environ, reason="https://github.com/pytorch/pytorch/issues/22811"
tests/infer/autoguide/test_gaussian.py:    parser.add_argument("--cuda", action="store_true")
tests/infer/autoguide/test_gaussian.py:    parser.add_argument("--cpu", dest="cuda", action="store_false")
tests/infer/autoguide/test_gaussian.py:    if args.cuda:
tests/infer/autoguide/test_gaussian.py:        torch.set_default_device("cuda")
tests/infer/test_enum.py:def _skip_cuda(*args):
tests/infer/test_enum.py:        condition="CUDA_TEST" in os.environ,
tests/infer/test_enum.py:        _skip_cuda("parallel", 30, False),
tests/infer/test_enum.py:        _skip_cuda("parallel", 30, False),
tests/infer/test_enum.py:        _skip_cuda("parallel", 40, False),
tests/infer/test_enum.py:        _skip_cuda("parallel", 50, False),
tests/infer/test_enum.py:@pytest.mark.parametrize("num_steps", [2, 3, 4, 5, 10, 20, _skip_cuda(30)])
tests/infer/test_enum.py:@pytest.mark.parametrize("num_steps", [2, 3, 4, 5, 10, 20, _skip_cuda(30)])
tests/infer/test_enum.py:    "CUDA_TEST" in os.environ, reason="https://github.com/pyro-ppl/pyro/issues/1380"
tests/infer/test_enum.py:@pytest.mark.parametrize("size", [1, 2, 3, 4, 10, 20, _skip_cuda(30)])
tests/ops/test_provenance.py:from tests.common import assert_equal, requires_cuda
tests/ops/test_provenance.py:@requires_cuda
tests/ops/test_provenance.py:    device = torch.device("cuda")
tests/README.md:to build small sized binaries, we remove all CUDA dependencies, but have statically linked MKL
tests/README.md:     docker run -it --ipc=host --rm -v $(pwd):/remote soumith/manylinux-cuda80:latest bash
tests/README.md:     `NO_CUDA` and `CMAKE_LIBRARY_PATH`.
tests/test_examples.py:    requires_cuda,
tests/test_examples.py:CUDA_EXAMPLES = [
tests/test_examples.py:    "air/main.py --num-steps=1 --cuda",
tests/test_examples.py:    "baseball.py --num-samples=200 --warmup-steps=100 --num-chains=2 --cuda",
tests/test_examples.py:    "contrib/cevae/synthetic.py --num-epochs=1 --cuda",
tests/test_examples.py:    "contrib/epidemiology/sir.py --nojit -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 --cuda",
tests/test_examples.py:    "contrib/epidemiology/sir.py --nojit -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 -nb=16 --cuda",
tests/test_examples.py:    "contrib/epidemiology/sir.py --nojit -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 --haar --cuda",
tests/test_examples.py:    "contrib/epidemiology/regional.py --nojit -t=2 -w=2 -n=4 -r=3 -d=20 -p=1000 -f 2 --cuda",
tests/test_examples.py:    "contrib/epidemiology/regional.py --nojit -t=2 -w=2 -n=4 -r=3 -d=20 -p=1000 -f 2 --haar --cuda",
tests/test_examples.py:    "contrib/gp/sv-dkl.py --epochs=1 --num-inducing=4 --cuda",
tests/test_examples.py:    "contrib/mue/FactorMuE.py --test --small --include-stop --no-plots --no-save --cuda --cpu-data --pin-mem",
tests/test_examples.py:    "contrib/mue/FactorMuE.py --test --small -ard -idfac --no-substitution-matrix --no-plots --no-save --cuda",
tests/test_examples.py:    "contrib/mue/ProfileHMM.py --test --small --no-plots --no-save --cuda --cpu-data --pin-mem",
tests/test_examples.py:    "contrib/mue/ProfileHMM.py --test --small --include-stop --no-plots --no-save --cuda",
tests/test_examples.py:    "lkj.py --n=50 --num-chains=1 --warmup-steps=100 --num-samples=200 --cuda",
tests/test_examples.py:    "dmm.py --num-epochs=1 --cuda",
tests/test_examples.py:    "dmm.py --num-epochs=1 --num-iafs=1 --cuda",
tests/test_examples.py:    "dmm.py --num-epochs=1 --tmc --tmc-num-samples=2 --cuda",
tests/test_examples.py:    "dmm.py --num-epochs=1 --tmcelbo --tmc-num-samples=2 --cuda",
tests/test_examples.py:    "einsum.py --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=0 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=1 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=2 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=3 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=4 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=5 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=6 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=6 --cuda --raftery-parameterization",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=7 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=0 --tmc --tmc-num-samples=2 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=1 --tmc --tmc-num-samples=2 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=2 --tmc --tmc-num-samples=2 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=3 --tmc --tmc-num-samples=2 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=4 --tmc --tmc-num-samples=2 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=5 --tmc --tmc-num-samples=2 --cuda",
tests/test_examples.py:    "hmm.py --num-steps=1 --truncate=10 --model=6 --tmc --tmc-num-samples=2 --cuda",
tests/test_examples.py:    "scanvi/scanvi.py --num-epochs 1 --dataset mock --cuda",
tests/test_examples.py:    "sir_hmc.py -t=2 -w=2 -n=4 -d=2 -m=1 --enum --cuda",
tests/test_examples.py:    "sir_hmc.py -t=2 -w=2 -n=4 -d=2 -p=10000 --sequential --cuda",
tests/test_examples.py:    "sir_hmc.py -t=2 -w=2 -n=4 -d=100 -p=10000 --cuda",
tests/test_examples.py:    "svi_torch.py --num-epochs=2 --size=400 --cuda",
tests/test_examples.py:    "svi_horovod.py --num-epochs=2 --size=400 --cuda --no-horovod",
tests/test_examples.py:        "svi_lightning.py --max_epochs=2 --size=400 --accelerator gpu --devices 1",
tests/test_examples.py:    "vae/vae.py --num-epochs=1 --cuda",
tests/test_examples.py:    "vae/ss_vae_M2.py --num-epochs=1 --cuda",
tests/test_examples.py:    "vae/ss_vae_M2.py --num-epochs=1 --aux-loss --cuda",
tests/test_examples.py:    "vae/ss_vae_M2.py --num-epochs=1 --enum-discrete=parallel --cuda",
tests/test_examples.py:    "vae/ss_vae_M2.py --num-epochs=1 --enum-discrete=sequential --cuda",
tests/test_examples.py:    "cvae/main.py --num-quadrant-inputs=1 --num-epochs=1 --cuda",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=0 --cuda",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=1 --cuda",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=2 --cuda",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=3 --cuda",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=4 --cuda",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=5 --cuda",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=6 --cuda",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=6 --cuda --raftery-parameterization ",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=1 --cuda--tmc --tmc-num-samples=2 ",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=2 --cuda--tmc --tmc-num-samples=2 ",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=3 --cuda--tmc --tmc-num-samples=2 ",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=4 --cuda--tmc --tmc-num-samples=2 ",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=5 --cuda--tmc --tmc-num-samples=2 ",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=6 --cuda--tmc --tmc-num-samples=2 ",
tests/test_examples.py:    "contrib/funsor/hmm.py --num-steps=1 --truncate=10 --model=6 --cuda--tmc --tmc-num-samples=2  -rp",
tests/test_examples.py:        "svi_horovod.py --num-epochs=2 --size=400 --cuda", marks=[requires_cuda]
tests/test_examples.py:    cuda_tests = set(
tests/test_examples.py:        (e if isinstance(e, str) else e.values[0]).split()[0] for e in CUDA_EXAMPLES
tests/test_examples.py:                if "--cuda" in text and example not in cuda_tests:
tests/test_examples.py:                        "Example: {} not covered by CUDA_EXAMPLES.".format(example)
tests/test_examples.py:@requires_cuda
tests/test_examples.py:@pytest.mark.parametrize("example", CUDA_EXAMPLES)
tests/test_examples.py:def test_cuda(example):
tests/test_examples.py:    if "cuda" in example and np > torch.cuda.device_count():
tests/poutine/test_mapdata.py:from tests.common import requires_cuda
tests/poutine/test_mapdata.py:def plate_cuda_model(subsample_size):
tests/poutine/test_mapdata.py:    loc = torch.zeros(20).cuda()
tests/poutine/test_mapdata.py:    scale = torch.ones(20).cuda()
tests/poutine/test_mapdata.py:def iplate_cuda_model(subsample_size):
tests/poutine/test_mapdata.py:    loc = torch.zeros(20).cuda()
tests/poutine/test_mapdata.py:    scale = torch.ones(20).cuda()
tests/poutine/test_mapdata.py:@requires_cuda
tests/poutine/test_mapdata.py:    "model", [plate_cuda_model, iplate_cuda_model], ids=["plate", "iplate"]
tests/poutine/test_mapdata.py:def test_cuda(model, subsample_size):
tests/poutine/test_mapdata.py:    assert tr.log_prob_sum().is_cuda
tests/distributions/test_torch_patch.py:from tests.common import assert_close, requires_cuda
tests/distributions/test_torch_patch.py:@requires_cuda
tests/distributions/test_torch_patch.py:def test_dirichlet_grad_cuda():
tests/distributions/test_torch_patch.py:@requires_cuda
tests/distributions/test_torch_patch.py:    x = torch.linspace(-1.0, 1.0, 100, device="cuda")
tests/distributions/test_torch_patch.py:    assert x.device.type == "cuda"
tests/distributions/test_von_mises.py:            0.01, condition="CUDA_TEST" in os.environ, reason="low precision."
tests/distributions/test_spanning_tree.py:    "CUDA_TEST" in os.environ, reason="spanning_tree unsupported on CUDA."
tests/distributions/test_cuda.py:    requires_cuda,
tests/distributions/test_cuda.py:@requires_cuda
tests/distributions/test_cuda.py:        assert not cpu_value.is_cuda
tests/distributions/test_cuda.py:        # Compute GPU value.
tests/distributions/test_cuda.py:        with torch.device("cuda"):
tests/distributions/test_cuda.py:        cuda_value = dist.pyro_dist(**params).sample()
tests/distributions/test_cuda.py:        assert cuda_value.is_cuda
tests/distributions/test_cuda.py:        assert_equal(cpu_value.size(), cuda_value.size())
tests/distributions/test_cuda.py:@requires_cuda
tests/distributions/test_cuda.py:        assert not cpu_value.is_cuda
tests/distributions/test_cuda.py:        # Compute GPU value.
tests/distributions/test_cuda.py:        with torch.device("cuda"):
tests/distributions/test_cuda.py:        cuda_value = dist.pyro_dist(**params).rsample()
tests/distributions/test_cuda.py:        assert cuda_value.is_cuda
tests/distributions/test_cuda.py:        assert_equal(cpu_value.size(), cuda_value.size())
tests/distributions/test_cuda.py:        cuda_grads = grad(cuda_value.sum(), [params[key] for key in grad_params])
tests/distributions/test_cuda.py:        for cpu_grad, cuda_grad in zip(cpu_grads, cuda_grads):
tests/distributions/test_cuda.py:            assert_equal(cpu_grad.size(), cuda_grad.size())
tests/distributions/test_cuda.py:@requires_cuda
tests/distributions/test_cuda.py:        assert not cpu_value.is_cuda
tests/distributions/test_cuda.py:        # Compute GPU value.
tests/distributions/test_cuda.py:        with torch.device("cuda"):
tests/distributions/test_cuda.py:        cuda_value = dist.pyro_dist(**params).log_prob(data)
tests/distributions/test_cuda.py:        assert cuda_value.is_cuda
tests/distributions/test_cuda.py:        assert_equal(cpu_value, cuda_value.cpu())
tests/common.py:import torch.cuda
tests/common.py:requires_cuda = pytest.mark.skipif(
tests/common.py:    not torch.cuda.is_available(), reason="cuda is not available"
tests/common.py:    assert t.__module__ == "torch.cuda"
tests/common.py:def get_gpu_type(t):
tests/common.py:    return getattr(torch.cuda, t.__name__)
tests/common.py:    :param str host: Either "cuda" or "cpu".
tests/common.py:    if torch.cuda.is_available():
tests/common.py:        cuda_rng_state = torch.cuda.get_rng_state()
tests/common.py:    if torch.cuda.is_available():
tests/common.py:        torch.cuda.set_rng_state(cuda_rng_state)
tests/common.py:    b = b.cuda(device=a.get_device()) if a.is_cuda else b.cpu()
Makefile:test-cuda: lint FORCE
Makefile:	CUDA_TEST=1 PYRO_DTYPE=float64 PYRO_DEVICE=cuda pytest -vx --stage unit
Makefile:	CUDA_TEST=1 pytest -vx tests/test_examples.py::test_cuda
Makefile:test-cuda-lax: lint FORCE
Makefile:	CUDA_TEST=1 PYRO_DTYPE=float64 PYRO_DEVICE=cuda pytest -vx --stage unit --lax
Makefile:	CUDA_TEST=1 pytest -vx tests/test_examples.py::test_cuda
profiler/gaussianhmm.py:    if args.cuda:
profiler/gaussianhmm.py:        torch.set_default_device("cuda")
profiler/gaussianhmm.py:    parser.add_argument("--cuda", action="store_true", default=False)
profiler/hmm.py:            if args.cuda:
profiler/hmm.py:                config.append("--cuda")
profiler/hmm.py:    parser.add_argument("--cuda", action="store_true")
tutorial/source/svi_lightning.rst:    $ python examples/svi_lightning.py --accelerator gpu --devices 2 --max_epochs 100 --strategy ddp
docker/Dockerfile:ARG cuda=0
docker/Makefile:.PHONY: help create-host-workspace build build-gpu run run-gpu notebook notebook-gpu
docker/Makefile:BASE_CUDA_IMG=nvidia/cuda:11.5.0-cudnn8-runtime-ubuntu18.04
docker/Makefile:DOCKER_GPU_CMD=nvidia-docker
docker/Makefile:build-gpu run-gpu notebook-gpu: img_prefix=pyro-gpu
docker/Makefile:build-gpu run-gpu lab-gpu: img_prefix=pyro-gpu
docker/Makefile:	## Requires nvidia-docker (https://github.com/NVIDIA/nvidia-docker).
docker/Makefile:build-gpu: ##
docker/Makefile:	## Build a docker image for running Pyro on a GPU.
docker/Makefile:	## Requires nvidia-docker (https://github.com/NVIDIA/nvidia-docker).
docker/Makefile:	${DOCKER_GPU_CMD} build -t ${IMG_NAME} \
docker/Makefile:	--build-arg base_img=${BASE_CUDA_IMG} \
docker/Makefile:	--build-arg cuda=1 \
docker/Makefile:run-gpu: create-host-workspace
docker/Makefile:run-gpu: ##
docker/Makefile:	## Start a Pyro GPU docker instance, and run the command `cmd`.
docker/Makefile:	docker run --init --runtime=nvidia -it --user ${USER} \
docker/Makefile:notebook-gpu: create-host-workspace
docker/Makefile:notebook-gpu: ##
docker/Makefile:	## Start a jupyter notebook on the Pyro GPU docker container.
docker/Makefile:	docker run --runtime=nvidia --init -it -p 8888:8888 --user ${USER} \
docker/Makefile:lab-gpu: create-host-workspace
docker/Makefile:lab-gpu: ##
docker/Makefile:	## Start jupyterlab on the Pyro GPU docker container.
docker/Makefile:	docker run --runtime=nvidia --init -it -p 8888:8888 --user ${USER} \
docker/README.md: - **nvidia-docker** Refer to the [readme](https://github.com/NVIDIA/nvidia-docker) for
docker/README.md:The Makefile can be used to build CPU and CUDA images for Pyro and PyTorch. Some common
docker/README.md: 2. **CPU / CUDA:** `make build` or `make build-gpu` can be used to specify whether the CPU
docker/README.md:    or the CUDA image is to be built. For building the CUDA image, *nvidia-docker* is 
docker/README.md:PyTorch's `master` branch, using python 3.6 to run on a GPU, is as follows:
docker/README.md:make build-gpu pyro_branch=dev pytorch_branch=master python_version=3.6
docker/README.md:This will build an image named `pyro-gpu-dev-3.6`. To spin up a docker container from this
docker/README.md:make notebook-gpu img=pyro-gpu-dev-3.6
docker/README.md:`make run-gpu`. By default this starts a *bash* shell. One could start an *ipython* 
docker/README.md:To run a *jupyter notebook* use `make notebook`, or `make notebook-gpu`. This will 
docker/install.sh:    if [ ${cuda} = 1 ]; then conda install -y cuda90 -c pytorch; fi
docker/install.sh:    if [ ${cuda} = 1 ]; then conda install -y cuda90 -c pytorch; fi
examples/svi_horovod.py:# machines (or multiple GPUs on one or more machines) using the Horovod
examples/svi_horovod.py:        if args.cuda:
examples/svi_horovod.py:            torch.cuda.set_device(hvd.local_rank())
examples/svi_horovod.py:    if args.cuda:
examples/svi_horovod.py:        torch.set_default_device("cuda")
examples/svi_horovod.py:    if args.cuda:
examples/svi_horovod.py:    parser.add_argument("--cuda", action="store_true")
examples/scanvi/scanvi.py:        dataset=args.dataset, batch_size=args.batch_size, cuda=args.cuda
examples/scanvi/scanvi.py:    if args.cuda:
examples/scanvi/scanvi.py:        scanvi.cuda()
examples/scanvi/scanvi.py:        "--cuda", action="store_true", default=False, help="whether to use cuda"
examples/dmm.py:        use_cuda=False,
examples/dmm.py:        self.use_cuda = use_cuda
examples/dmm.py:        # if on gpu cuda-ize all PyTorch (sub)modules
examples/dmm.py:        if use_cuda:
examples/dmm.py:            self.cuda()
examples/dmm.py:        # if on gpu we need the fully broadcast view of the rnn initial state
examples/dmm.py:        # to be in contiguous gpu memory
examples/dmm.py:        cuda=args.cuda,
examples/dmm.py:        cuda=args.cuda,
examples/dmm.py:        use_cuda=args.cuda,
examples/dmm.py:            cuda=args.cuda,
examples/dmm.py:    parser.add_argument("--cuda", action="store_true")
examples/svi_lightning.py:# machines (or multiple GPUs on one or more machines) using the PyTorch Lightning
examples/air/air.py:        use_cuda=False,
examples/air/air.py:        self.use_cuda = use_cuda
examples/air/air.py:        prototype = torch.tensor(0.0).cuda() if use_cuda else torch.tensor(0.0)
examples/air/air.py:        # By making these parameters they will be moved to the gpu
examples/air/air.py:        if use_cuda:
examples/air/air.py:            self.cuda()
examples/air/air.py:    if z_where.is_cuda:
examples/air/air.py:        ix = ix.cuda()
examples/air/main.py:    if X.is_cuda:
examples/air/main.py:        error_indices = error_indices.cuda()
examples/air/main.py:    if args.cuda:
examples/air/main.py:        X = X.cuda()
examples/air/main.py:        use_cuda=args.cuda,
examples/air/main.py:    parser.add_argument("--cuda", action="store_true", default=False, help="use cuda")
examples/einsum.py:    if args.cuda:
examples/einsum.py:        torch.set_default_device("cuda")
examples/einsum.py:    parser.add_argument("--cuda", action="store_true")
examples/baseball.py:        "--cuda", action="store_true", default=False, help="run this example in GPU"
examples/baseball.py:    # work around the error "CUDA error: initialization error"
examples/baseball.py:    if args.cuda:
examples/baseball.py:        torch.set_default_device("cuda")
examples/cvae/main.py:        "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
examples/cvae/main.py:        "--cuda", action="store_true", default=False, help="whether to use cuda"
examples/contrib/gp/sv-dkl.py:        if args.cuda:
examples/contrib/gp/sv-dkl.py:            data, target = data.cuda(), target.cuda()
examples/contrib/gp/sv-dkl.py:        if args.cuda:
examples/contrib/gp/sv-dkl.py:            data, target = data.cuda(), target.cuda()
examples/contrib/gp/sv-dkl.py:    if args.cuda:
examples/contrib/gp/sv-dkl.py:    if args.cuda:
examples/contrib/gp/sv-dkl.py:        gpmodule.cuda()
examples/contrib/gp/sv-dkl.py:        "--cuda", action="store_true", default=False, help="enables CUDA training"
examples/contrib/gp/sv-dkl.py:    if args.cuda:
examples/contrib/funsor/hmm.py:    if args.cuda:
examples/contrib/funsor/hmm.py:        torch.set_default_device("cuda")
examples/contrib/funsor/hmm.py:    parser.add_argument("--cuda", action="store_true")
examples/contrib/cevae/synthetic.py:    if args.cuda:
examples/contrib/cevae/synthetic.py:        torch.set_default_device("cuda")
examples/contrib/cevae/synthetic.py:    parser.add_argument("--cuda", action="store_true")
examples/contrib/epidemiology/regional.py:    parser.add_argument("--cuda", action="store_true")
examples/contrib/epidemiology/regional.py:    if args.cuda:
examples/contrib/epidemiology/regional.py:        torch.set_default_device("cuda")
examples/contrib/epidemiology/sir.py:    parser.add_argument("--cuda", action="store_true")
examples/contrib/epidemiology/sir.py:    if args.cuda:
examples/contrib/epidemiology/sir.py:        torch.set_default_device("cuda")
examples/contrib/mue/FactorMuE.py:    --jit --cuda
examples/contrib/mue/FactorMuE.py:This should take about 8 minutes to run on a GPU. The latent space should show
examples/contrib/mue/FactorMuE.py:    if args.cpu_data or not args.cuda:
examples/contrib/mue/FactorMuE.py:        device = torch.device("cuda")
examples/contrib/mue/FactorMuE.py:        cuda=args.cuda,
examples/contrib/mue/FactorMuE.py:    parser.add_argument("--cuda", default=False, action="store_true", help="Use GPU.")
examples/contrib/mue/FactorMuE.py:        help="Use pin_memory for faster CPU to GPU transfer.",
examples/contrib/mue/FactorMuE.py:    if args.cuda:
examples/contrib/mue/FactorMuE.py:        torch.set_default_device("cuda")
examples/contrib/mue/ProfileHMM.py:    -e 15 -lr 0.01 --jit --cuda
examples/contrib/mue/ProfileHMM.py:This should take about 9 minutes to run on a GPU. The perplexity should be
examples/contrib/mue/ProfileHMM.py:    if args.cpu_data or not args.cuda:
examples/contrib/mue/ProfileHMM.py:        device = torch.device("cuda")
examples/contrib/mue/ProfileHMM.py:        cuda=args.cuda,
examples/contrib/mue/ProfileHMM.py:    parser.add_argument("--cuda", default=False, action="store_true", help="Use GPU.")
examples/contrib/mue/ProfileHMM.py:        help="Use pin_memory for faster GPU transfer.",
examples/contrib/mue/ProfileHMM.py:    if args.cuda:
examples/contrib/mue/ProfileHMM.py:        torch.set_default_device("cuda")
examples/svi_torch.py:        covariates = covariates.to(device=torch.device("cuda" if args.cuda else "cpu"))
examples/svi_torch.py:        data = data.to(device=torch.device("cuda" if args.cuda else "cpu"))
examples/svi_torch.py:    loss_fn.to(device=torch.device("cuda" if args.cuda else "cpu"))
examples/svi_torch.py:    parser.add_argument("--cuda", action="store_true", default=False)
examples/lkj.py:    if args.cuda:
examples/lkj.py:        y = y.cuda()
examples/lkj.py:    parser.add_argument("--cuda", action="store_true", default=False)
examples/hmm.py:    if args.cuda:
examples/hmm.py:        torch.set_default_device("cuda")
examples/hmm.py:    parser.add_argument("--cuda", action="store_true")
examples/sir_hmc.py:    parser.add_argument("--cuda", action="store_true")
examples/sir_hmc.py:    if args.cuda:
examples/sir_hmc.py:        torch.set_default_device("cuda")
examples/vae/vae.py:    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
examples/vae/vae.py:        if use_cuda:
examples/vae/vae.py:            # calling cuda() here will put all the parameters of
examples/vae/vae.py:            # the encoder and decoder networks into gpu memory
examples/vae/vae.py:            self.cuda()
examples/vae/vae.py:        self.use_cuda = use_cuda
examples/vae/vae.py:        MNIST, use_cuda=args.cuda, batch_size=256
examples/vae/vae.py:    vae = VAE(use_cuda=args.cuda)
examples/vae/vae.py:            # if on GPU put mini-batch into CUDA memory
examples/vae/vae.py:            if args.cuda:
examples/vae/vae.py:                x = x.cuda()
examples/vae/vae.py:                # if on GPU put mini-batch into CUDA memory
examples/vae/vae.py:                if args.cuda:
examples/vae/vae.py:                    x = x.cuda()
examples/vae/vae.py:        "--cuda", action="store_true", default=False, help="whether to use cuda"
examples/vae/utils/custom_mlp.py:        use_cuda=False,
examples/vae/utils/mnist_cached.py:def fn_x_mnist(x, use_cuda):
examples/vae/utils/mnist_cached.py:    # send the data to GPU(s)
examples/vae/utils/mnist_cached.py:    if use_cuda:
examples/vae/utils/mnist_cached.py:        xp = xp.cuda()
examples/vae/utils/mnist_cached.py:def fn_y_mnist(y, use_cuda):
examples/vae/utils/mnist_cached.py:    # send the data to GPU(s)
examples/vae/utils/mnist_cached.py:    if use_cuda:
examples/vae/utils/mnist_cached.py:        yp = yp.cuda()
examples/vae/utils/mnist_cached.py:        y = y.cuda()
examples/vae/utils/mnist_cached.py:    def __init__(self, mode, sup_num, use_cuda=True, *args, **kwargs):
examples/vae/utils/mnist_cached.py:            return fn_x_mnist(x, use_cuda)
examples/vae/utils/mnist_cached.py:            return fn_y_mnist(y, use_cuda)
examples/vae/utils/mnist_cached.py:    dataset, use_cuda, batch_size, sup_num=None, root=None, download=True, **kwargs
examples/vae/utils/mnist_cached.py:    :param use_cuda: use GPU(s) for training
examples/vae/utils/mnist_cached.py:            root=root, mode=mode, download=download, sup_num=sup_num, use_cuda=use_cuda
examples/vae/ss_vae_M2.py:    :param use_cuda: use GPUs for faster training
examples/vae/ss_vae_M2.py:        use_cuda=False,
examples/vae/ss_vae_M2.py:        self.use_cuda = use_cuda
examples/vae/ss_vae_M2.py:            use_cuda=self.use_cuda,
examples/vae/ss_vae_M2.py:            use_cuda=self.use_cuda,
examples/vae/ss_vae_M2.py:            use_cuda=self.use_cuda,
examples/vae/ss_vae_M2.py:        # using GPUs for faster training of the networks
examples/vae/ss_vae_M2.py:        if self.use_cuda:
examples/vae/ss_vae_M2.py:            self.cuda()
examples/vae/ss_vae_M2.py:        use_cuda=args.cuda,
examples/vae/ss_vae_M2.py:            MNISTCached, args.cuda, args.batch_size, sup_num=args.sup_num
examples/vae/ss_vae_M2.py:    "example run: python ss_vae_M2.py --seed 0 --cuda -n 2 --aux-loss -alm 46 -enum parallel "
examples/vae/ss_vae_M2.py:        "--cuda", action="store_true", help="use GPU(s) to speed up training"
CONTRIBUTING.md:make test-cuda         # runs unit tests in cuda mode
CONTRIBUTING.md:# or in cuda mode
CONTRIBUTING.md:CUDA_TEST=1 PYRO_DTYPE=float64 PYRO_DEVICE=cuda pytest -vs {path_to_test}::{test_name}

```
