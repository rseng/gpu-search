# https://github.com/mj-will/nessai

```console
docs/normalising-flows-configuration.rst:            device_tag="cuda",
docs/normalising-flows-configuration.rst:        device_tag="cuda",
CHANGELOG.md:- Add `nessai.flows.transforms.LULinear` to address a [bug in nflows](https://github.com/bayesiains/nflows/pull/38) that has not been patched and prevents the use of CUDA with `LULinear`. ([#138](https://github.com/mj-will/nessai/pull/138))
CHANGELOG.md:- Add option to train using dataloaders or directly with tensors. This is faster when using CUDA.
tests/test_utils/test_distribution_utils.py:@pytest.mark.cuda
tests/test_utils/test_distribution_utils.py:def test_get_dist_integration_cuda(get_func, args, kwargs):
tests/test_utils/test_distribution_utils.py:    dist = get_func(3, *args, **kwargs, device="cuda")
tests/test_flowmodel/test_flowmodel_utils.py:        device_tag="cuda",
tests/test_flowmodel/test_flowmodel_utils.py:        inference_device_tag="cuda",
tests/test_flowmodel/test_flowmodel_utils.py:    assert training_config["device_tag"] == "cuda"
tests/test_flowmodel/test_flowmodel_utils.py:    assert training_config["inference_device_tag"] == "cuda"
tests/test_flowmodel/test_flowmodel_base.py:    model.device = "cuda"
tests/test_flowmodel/test_flowmodel_importance.py:    ifm.training_config = dict(device="cpu", inference_device_tag="cuda")
tests/test_flowmodel/test_flowmodel_importance.py:    mock_device.call_args_list[1].args[0] == "cuda"
tests/conftest.py:        "markers", "cuda: mark test to indicate it requires CUDA"
tests/conftest.py:    for mark in item.iter_markers(name="cuda"):
tests/conftest.py:        if not torch.cuda.is_available():
tests/conftest.py:            pytest.skip("Test requires CUDA")
README.md:By default the version of PyTorch will not necessarily match the drivers on your system, to install a different version with the correct CUDA support see the PyTorch homepage for instructions: https://pytorch.org/.
nessai/experimental/flowmodel/clustering.py:                "faiss is not installed! Install faiss-cpu/-gpu in order to "
nessai/flows/transforms.py:    """Wrapper for LULinear from nflows that works with CUDA.
nessai/flows/transforms.py:    The original implementation has a bug that prevents use with CUDA. See

```
