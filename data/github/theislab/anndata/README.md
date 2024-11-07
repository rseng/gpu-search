# https://github.com/scverse/anndata

```console
ci/scripts/towncrier_automation.py:            "--label=skip-gpu-ci",
docs/contributing.md:### GPU CI
docs/contributing.md:To test GPU specific code we have a paid self-hosted runner to run the gpu specific tests on.
docs/contributing.md:This CI runs by default on the main branch, but for PRs requires the `run-gpu-ci` label to prevent unnecessary runs.
docs/release-notes/0.10.9.md:- Upper bound {mod}`numpy` for `gpu` installation on account of {issue}`cupy/cupy#8391` {user}`ilan-gold` ({pr}`1540`)
docs/release-notes/0.10.9.md:- create new `cupy` installation options for cuda 11 & 12 called `cu11` and `cu12` {user}`Intron7` ({pr}`1596`)
docs/release-notes/0.10.0.md:**GPU Support**
docs/release-notes/0.10.0.md:* anndata now has GPU enabled CI. Made possibly by a grant from [CZI's EOSS program](https://chanzuckerberg.com/eoss/) and managed via [Cirun](https://Cirun.io) {pr}`1066` {pr}`1084` {user}`Zethson` {user}`ivirshup`
docs/release-notes/0.11.0rc1.md:- create new `cupy` installation options for cuda 11 & 12 called `cu11` and `cu12` {user}`Intron7` ({pr}`1596`)
docs/release-notes/0.11.0rc1.md:- Add functionality to write from GPU {class}`dask.array.Array` to disk {user}`ilan-gold` ({pr}`1550`)
tests/test_concatenate.py:@pytest.mark.gpu
tests/test_dask.py:            marks=pytest.mark.gpu,
tests/test_io_elementwise.py:@pytest.mark.gpu
tests/test_views.py:def matrix_type_no_gpu(request):
tests/test_views.py:    # Fix if and when dask supports tokenizing GPU arrays
tests/test_views.py:def test_not_set_subset_X_dask(matrix_type_no_gpu, subset_func):
tests/test_views.py:    adata = ad.AnnData(matrix_type_no_gpu(asarray(sparse.random(20, 20))))
tests/test_helpers.py:            as_dense_cupy_dask_array, CupyArray, id="cupy_dense", marks=pytest.mark.gpu
tests/test_helpers.py:            marks=pytest.mark.gpu,
tests/test_helpers.py:@pytest.mark.gpu
tests/test_helpers.py:    X_gpu_roundtripped = as_cupy(X_cpu).map_blocks(lambda x: x.get(), meta=X_cpu._meta)
tests/test_helpers.py:    assert isinstance(X_gpu_roundtripped._meta, type(X_cpu._meta))
tests/test_helpers.py:    assert isinstance(X_gpu_roundtripped.compute(), type(X_cpu.compute()))
tests/test_helpers.py:    assert_equal(X_gpu_roundtripped.compute(), X_cpu.compute())
tests/test_gpu.py:@pytest.mark.gpu
tests/test_gpu.py:def test_gpu():
tests/test_gpu.py:    For testing that the gpu mark works
tests/test_gpu.py:@pytest.mark.gpu
tests/test_gpu.py:def test_adata_raw_gpu():
tests/test_gpu.py:@pytest.mark.gpu
tests/test_gpu.py:def test_raw_gpu():
.cirun.yml:  - name: aws-gpu-runner
.cirun.yml:      - cirun-aws-gpu
pyproject.toml:gpu = ["cupy"]
pyproject.toml:cu12 = ["cupy-cuda12x"]
pyproject.toml:cu11 = ["cupy-cuda11x"]
pyproject.toml:markers = ["gpu: mark test to run on GPU"]
src/testing/anndata/_pytest.py:    """Define behavior of pytest.mark.gpu."""
src/testing/anndata/_pytest.py:    is_gpu = len([mark for mark in item.iter_markers(name="gpu")]) > 0
src/testing/anndata/_pytest.py:    if is_gpu:
src/anndata/_core/raw.py:            # Move from GPU to CPU since it's large and not always used
src/anndata/_core/raw.py:            # Move from GPU to CPU since it's large and not always used
src/anndata/experimental/pytorch/_annloader.py:# maybe replace use_cuda with explicit device option
src/anndata/experimental/pytorch/_annloader.py:def default_converter(arr, use_cuda, pin_memory):
src/anndata/experimental/pytorch/_annloader.py:        if use_cuda:
src/anndata/experimental/pytorch/_annloader.py:            arr = arr.cuda()
src/anndata/experimental/pytorch/_annloader.py:        if use_cuda:
src/anndata/experimental/pytorch/_annloader.py:            arr = torch.tensor(arr, device="cuda")
src/anndata/experimental/pytorch/_annloader.py:        the default cuda device (if `use_cuda=True`), do memory pinning (if `pin_memory=True`).
src/anndata/experimental/pytorch/_annloader.py:    use_cuda
src/anndata/experimental/pytorch/_annloader.py:        Transfer pytorch tensors to the default cuda device after conversion.
src/anndata/experimental/pytorch/_annloader.py:        use_cuda: bool = False,
src/anndata/experimental/pytorch/_annloader.py:                default_converter, use_cuda=use_cuda, pin_memory=pin_memory
src/anndata/tests/helpers.py:        partial(as_cupy, typ=CupyArray), id="cupy_array", marks=pytest.mark.gpu
src/anndata/tests/helpers.py:        marks=pytest.mark.gpu,
src/anndata/tests/helpers.py:        marks=pytest.mark.gpu,
src/anndata/tests/helpers.py:        marks=pytest.mark.gpu,
src/anndata/tests/helpers.py:        as_cupy_sparse_dask_array, id="cupy_csr_dask_array", marks=pytest.mark.gpu

```
