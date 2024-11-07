# https://github.com/dask/dask

```console
dask/tests/test_utils.py:def test_get_meta_library_gpu():
dask/tests/test_backends.py:@pytest.mark.gpu
dask/tests/test_highgraph.py:        with dask.annotate(resources={"GPU": 1}):
dask/tests/test_highgraph.py:    assert alayer.annotations == {"resources": {"GPU": 1}, "block_id": annot_map_fn}
dask/config.py:    "ucx.cuda-copy": "distributed.ucx.cuda_copy",
dask/dataframe/tests/test_groupby.py:@pytest.mark.gpu
dask/dataframe/tests/test_groupby.py:    # be skipped by non-GPU CI
dask/dataframe/tests/test_groupby.py:@pytest.mark.gpu
dask/dataframe/tests/test_groupby.py:@pytest.mark.gpu
dask/dataframe/tests/test_groupby.py:        pytest.param("cudf", marks=pytest.mark.gpu),
dask/dataframe/tests/test_categorical.py:@pytest.mark.gpu
dask/dataframe/tests/test_shuffle.py:    "engine", ["pandas", pytest.param("cudf", marks=pytest.mark.gpu)]
dask/dataframe/tests/test_shuffle.py:        # will be skipped by non-GPU CI.
dask/dataframe/tests/test_shuffle.py:    "engine", ["pandas", pytest.param("cudf", marks=pytest.mark.gpu)]
dask/dataframe/tests/test_shuffle.py:        # will be skipped by non-GPU CI.
dask/dataframe/tests/test_shuffle.py:    "engine", ["pandas", pytest.param("cudf", marks=pytest.mark.gpu)]
dask/dataframe/tests/test_shuffle.py:        # will be skipped by non-GPU CI.
dask/dataframe/tests/test_shuffle.py:    "engine", ["pandas", pytest.param("cudf", marks=pytest.mark.gpu)]
dask/dataframe/tests/test_shuffle.py:        # will be skipped by non-GPU CI.
dask/dataframe/tests/test_shuffle.py:    "backend", ["pandas", pytest.param("cudf", marks=pytest.mark.gpu)]
dask/dataframe/tests/test_indexing.py:@pytest.mark.gpu
dask/dataframe/tests/test_indexing.py:def test_gpu_loc():
dask/dataframe/tests/test_multi.py:        pytest.param("cudf", marks=pytest.mark.gpu),
dask/dataframe/tests/test_multi.py:        # will be skipped by non-GPU CI.
dask/dataframe/tests/test_multi.py:    "engine", ["pandas", pytest.param("cudf", marks=pytest.mark.gpu)]
dask/dataframe/tests/test_multi.py:        # will be skipped by non-GPU CI.
dask/dataframe/tests/test_dataframe.py:@pytest.mark.gpu
dask/dataframe/tests/test_dataframe.py:def test_cov_gpu(numeric_only):
dask/dataframe/tests/test_dataframe.py:@pytest.mark.gpu
dask/dataframe/tests/test_dataframe.py:def test_corr_gpu():
dask/dataframe/tests/test_dataframe.py:@pytest.mark.parametrize("gpu", [False, pytest.param(True, marks=pytest.mark.gpu)])
dask/dataframe/tests/test_dataframe.py:def test_to_datetime(gpu):
dask/dataframe/tests/test_dataframe.py:    xd = pd if not gpu else pytest.importorskip("cudf")
dask/dataframe/tests/test_dataframe.py:    check_dtype = not gpu
dask/dataframe/tests/test_dataframe.py:        ctx_expected = contextlib.nullcontext() if gpu else ctx
dask/dataframe/tests/test_dataframe.py:    if not gpu:
dask/dataframe/tests/test_dataframe.py:@pytest.mark.gpu
dask/dataframe/io/tests/test_io.py:@pytest.mark.gpu
dask/dataframe/io/tests/test_io.py:def test_gpu_from_pandas_npartitions_duplicates():
dask/dataframe/io/tests/test_io.py:@pytest.mark.gpu
dask/dataframe/io/tests/test_io.py:@pytest.mark.gpu
dask/dataframe/io/tests/test_parquet.py:@pytest.mark.gpu
dask/dataframe/io/tests/test_parquet.py:def test_gpu_write_parquet_simple(tmpdir):
dask/dataframe/backends.py:# cuDF: Pandas Dataframes on the GPU #
dask/sizeof.py:    import numba.cuda
dask/sizeof.py:    @sizeof.register(numba.cuda.cudadrv.devicearray.DeviceNDArray)
dask/array/tests/test_cupy_percentile.py:pytestmark = pytest.mark.gpu
dask/array/tests/test_cupy_sparse.py:pytestmark = pytest.mark.gpu
dask/array/tests/test_cupy_core.py:pytestmark = pytest.mark.gpu
dask/array/tests/test_cupy_linalg.py:pytestmark = pytest.mark.gpu
dask/array/tests/test_atop.py:    with dask.annotate(retries=3, resources={"GPU": 2, "Memory": 10}):
dask/array/tests/test_atop.py:    with dask.annotate(priority=4, resources={"GPU": 5, "Memory": 4}):
dask/array/tests/test_atop.py:    assert annotations["resources"] == {"GPU": 5, "Memory": 10}  # Max of resources
dask/array/tests/test_cupy_reductions.py:pytestmark = pytest.mark.gpu
dask/array/tests/test_cupy_overlap.py:pytestmark = pytest.mark.gpu
dask/array/tests/test_cupy_random.py:pytestmark = pytest.mark.gpu
dask/array/tests/test_cupy_gufunc.py:pytestmark = pytest.mark.gpu
dask/array/tests/test_creation.py:    ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)],
dask/array/tests/test_cupy_slicing.py:pytestmark = pytest.mark.gpu
dask/array/tests/test_cupy_routines.py:pytestmark = pytest.mark.gpu
dask/array/tests/test_cupy_creation.py:pytestmark = pytest.mark.gpu
dask/array/utils.py:    # for gpu/cpu checking
docs/source/changelog.rst:  - Update gpuCI ``RAPIDS_VER`` to ``24.12`` (:pr:`11407`)
docs/source/changelog.rst:  - Update gpuCI ``RAPIDS_VER`` to ``24.12`` (:pr-distributed:`8879`)
docs/source/changelog.rst:  - Increase visibility of GPU CI updates (:pr:`11345`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:  - Increase visibility of GPU CI updates (:pr-distributed:`8841`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:  - Upgrade gpuCI and fix Dask Array failures with "cupy" backend (:pr:`11309`) `Richard (Rick) Zamora`_
docs/source/changelog.rst:  - Update gpuCI ``RAPIDS_VER`` to ``24.10`` (:pr-distributed:`8786`)
docs/source/changelog.rst:  - Add python 3.11 build to GPU CI (:pr:`11135`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:  - Update gpuCI ``RAPIDS_VER`` to ``24.08`` (:pr:`11141`)
docs/source/changelog.rst:  - Update gpuCI ``RAPIDS_VER`` to ``24.08`` (:pr-distributed:`8652`)
docs/source/changelog.rst:  - Test query-planning in gpuCI (:pr:`11060`) `Richard (Rick) Zamora`_
docs/source/changelog.rst:  - Update GPU CI ``RAPIDS_VER`` to 24.06, disable query planning  (:pr:`11045`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:GPU metric dashboard fixes
docs/source/changelog.rst:GPU memory and utilization dashboard functionality has been restored.
docs/source/changelog.rst:  - Add Python 3.11 to GPU CI matrix (:pr-distributed:`8598`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:  - Update gpuCI ``RAPIDS_VER`` to ``24.06`` (:pr-distributed:`8588`)
docs/source/changelog.rst:  - Fix gpuci: np.product is deprecated (:pr-distributed:`8518`) `crusaderky`_
docs/source/changelog.rst:  - Update gpuCI ``RAPIDS_VER`` to ``24.04`` (:pr-distributed:`8471`)
docs/source/changelog.rst:  - Avoid ``pytest.warns`` in ``test_to_datetime`` for GPU CI (:pr:`10902`) `Richard (Rick) Zamora`_
docs/source/changelog.rst:  - Update gpuCI ``RAPIDS_VER`` to ``24.04`` (:pr:`10841`)
docs/source/changelog.rst:  - Add cuDF spilling statistics to RMM/GPU memory plot (:pr-distributed:`8148`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:  - Remove GPU executor (:pr-distributed:`8399`) `Hendrik Makait`_
docs/source/changelog.rst:  - Update gpuCI ``RAPIDS_VER`` to ``24.02`` (:pr-distributed:`8384`)
docs/source/changelog.rst:  - Bump GPU CI to CUDA 11.8 (:pr:`10656`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:  - Update gpuCI ``RAPIDS_VER`` to ``24.02`` (:pr:`10636`)
docs/source/changelog.rst:  - Bump GPU CI to CUDA 11.8 (:pr-distributed:`8376`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:    - Generalize computation of ``NEW_*_VER`` in GPU CI updating workflow (:pr:`10610`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:    - Switch to newer GPU CI images (:pr:`10608`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:    - Generalize computation of ``NEW_*_VER`` in GPU CI updating workflow (:pr-distributed:`8319`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:    - Switch to newer GPU CI images (:pr-distributed:`8316`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:    - Update gpuCI ``RAPIDS_VER`` to ``23.12`` (:pr:`10526`)
docs/source/changelog.rst:- Update gpuCI ``RAPIDS_VER`` to ``23.10`` (:pr:`10427`)
docs/source/changelog.rst:- Update gpuCI ``RAPIDS_VER`` to ``23.08`` (:pr:`10310`)
docs/source/changelog.rst:- Generalize ``dd.to_datetime`` for GPU-backed collections, introduce ``get_meta_library`` utility (:pr:`9881`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Bump gpuCI ``PYTHON_VER`` 3.8->3.9 (:pr:`10233`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Remove ``dask/gpu`` from gpuCI update reviewers (:pr:`10135`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Update gpuCI ``RAPIDS_VER`` to ``23.06`` (:pr:`10129`)
docs/source/changelog.rst:- Remove broken gpuCI link in developer docs (:pr:`10065`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Run GPU tests on python 3.8 & 3.10 (:pr:`9940`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Update gpuCI ``RAPIDS_VER`` to ``23.04`` (:pr:`9876`)
docs/source/changelog.rst:- ``pip`` install dask on gpuCI builds (:pr:`9816`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Update gpuCI ``RAPIDS_VER`` to ``23.02`` (:pr:`9678`)
docs/source/changelog.rst:- Update ``ga-yaml-parser`` step in gpuCI updating workflow  (:pr:`9675`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Update gpuCI ``RAPIDS_VER`` to ``22.12`` (:pr:`9524`)
docs/source/changelog.rst:- Update gpuCI ``RAPIDS_VER`` to ``22.10`` (:pr:`9314`)
docs/source/changelog.rst:- Update gpuCI ``RAPIDS_VER`` to ``22.08`` (:pr:`9120`)
docs/source/changelog.rst:- Fix gpuCI GHA version (:pr:`8891`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Update gpuCI ``RAPIDS_VER`` to ``22.06`` (:pr:`8828`)
docs/source/changelog.rst:- Simplify gpuCI updating workflow (:pr:`8849`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Reduce gpuci ``pytest`` parallelism (:pr:`8826`) `GALI PREM SAGAR`_
docs/source/changelog.rst:- Pin ``scipy`` to less than 1.8.0 in GPU CI (:pr:`8698`) `Julia Signell`_
docs/source/changelog.rst:- Bump gpuCI PYTHON_VER to 3.9 (:pr:`8642`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Update gpuCI ``RAPIDS_VER`` to ``22.04`` (:pr:`8600`)
docs/source/changelog.rst:- Bump gpuCI ``CUDA_VER`` to 11.5 (:pr:`8489`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Update gpuCI ``RAPIDS_VER`` to ``22.02`` (:pr:`8394`)
docs/source/changelog.rst:- Only run gpuCI bump script daily (:pr:`8404`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Add workflow to update gpuCI (:pr:`8215`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Use ``pytest.param`` to properly label param-specific GPU tests (:pr:`8197`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Add ``test_set_index`` to tests ran on gpuCI (:pr:`8198`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Bump ``RAPIDS_VER`` for gpuCI (:pr:`8184`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Use development version of ``distributed`` in gpuCI build (:pr:`7976`) `James Bourbeau`_
docs/source/changelog.rst:- Mark gpuCI CuPy test as flaky (:pr:`7994`) `Peter Andreas Entschev`_
docs/source/changelog.rst:- Bump ``RAPIDS_VER`` in gpuCI to 21.10 (:pr:`7991`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Add gpuCI build script (:pr:`7966`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Add pytest marker for GPU tests (:pr:`7876`) `Charles Blackmon-Luca`_
docs/source/changelog.rst:- Enable gpu-backed covariance/correlation in dataframes (:pr:`5597`) `Richard J Zamora`_
docs/source/changelog.rst:-  Add document for how Dask works with GPUs (:pr:`4792`) `Matthew Rocklin`_
docs/source/changelog.rst:.. _`Shang Wang`: https://github.com/shangw-nvidia
docs/source/changelog.rst:.. _`Naty Clementi`: https://github.com/ncclementi
docs/source/faq.rst:NVIDIA, the leading hardware manufacturer of GPUs.  Despite large corporate
docs/source/faq.rst:Does Dask work on GPUs?
docs/source/faq.rst:Yes! Dask works with GPUs in a few ways.
docs/source/faq.rst:The `RAPIDS <https://rapids.ai>`_ libraries provide a GPU-accelerated
docs/source/faq.rst:`Chainer's CuPy <https://cupy.chainer.org/>`_ library provides a GPU
docs/source/faq.rst:For custom workflows people use Dask alongside GPU-accelerated libraries like PyTorch and
docs/source/faq.rst:See the section :doc:`gpu`.
docs/source/develop.rst:GPU CI
docs/source/develop.rst:Pull requests are also tested with a GPU enabled CI environment provided by
docs/source/develop.rst:NVIDIA: `gpuCI <https://gpuci.gpuopenanalytics.com/>`_.
docs/source/develop.rst:Unlike Github Actions, the CI environment for gpuCI is controlled with the
docs/source/develop.rst:`here <https://gpuci.gpuopenanalytics.com/job/dask/job/dask-build-environment/job/branch/job/dask-build-env-main/>`_.
docs/source/develop.rst:and Distributed similarly, gpuCI will run for both `Dask
docs/source/develop.rst:<https://gpuci.gpuopenanalytics.com/job/dask/job/dask/job/prb/job/dask-prb/>`_
docs/source/develop.rst:<https://gpuci.gpuopenanalytics.com/job/dask/job/distributed/job/prb/job/distributed-prb/>`_
docs/source/develop.rst:For each PR, gpuCI will run all tests decorated with the pytest marker
docs/source/develop.rst:``@pytest.mark.gpu``.  This is configured in the `gpuci folder
docs/source/develop.rst:<https://github.com/dask/dask/tree/main/continuous_integration/gpuci>`_ .
docs/source/develop.rst:Like Github Actions, gpuCI will not run when first time contributors to Dask or
docs/source/develop.rst:Distributed submit PRs.  In this case, the gpuCI bot will comment on the PR:
docs/source/develop.rst:.. image:: images/gputester-msg.png
docs/source/develop.rst:   :alt: "Screenshot of a GitHub comment left by the GPUtester bot, where the comment says 'Can one of the admins verify this patch?'."
docs/source/develop.rst:Dask Maintainers can then approve gpuCI builds for these PRs with following choices:
docs/source/dataframe-extend.rst:-  cuDF: for data analysis on GPUs
docs/source/ml.rst:For a more fully worked example see :bdg-link-primary:`Batch Scoring for Computer Vision Workloads (video) <https://developer.download.nvidia.com/video/gputechconf/gtc/2019/video/S9198/s9198-dask-and-v100s-for-fast-distributed-batch-scoring-of-computer-vision-workloads.mp4>`.
docs/source/dataframe-sql.rst:  In addition to working on CPU, it offers experimental support for CUDA-enabled GPUs through RAPIDS libraries such as `cuDF`_.
docs/source/gpu.rst:GPUs
docs/source/gpu.rst:Dask works with GPUs in a few ways.
docs/source/gpu.rst:Many people use Dask alongside GPU-accelerated libraries like PyTorch and
docs/source/gpu.rst:Dask doesn't need to know that these functions use GPUs.  It just runs Python
docs/source/gpu.rst:functions.  Whether or not those Python functions use a GPU is orthogonal to
docs/source/gpu.rst:        <source src="https://developer.download.nvidia.com/video/gputechconf/gtc/2019/video/S9198/s9198-dask-and-v100s-for-fast-distributed-batch-scoring-of-computer-vision-workloads.mp4"
docs/source/gpu.rst:combining the Dask Array and DataFrame collections with a GPU-accelerated
docs/source/gpu.rst:many Pandas dataframes.  We can use these same systems with GPUs if we swap out
docs/source/gpu.rst:the NumPy/Pandas components with GPU-accelerated versions of those same
docs/source/gpu.rst:libraries, as long as the GPU accelerated version looks enough like
docs/source/gpu.rst:Fortunately, libraries that mimic NumPy, Pandas, and Scikit-Learn on the GPU do
docs/source/gpu.rst:The `RAPIDS <https://rapids.ai>`_ libraries provide a GPU accelerated
docs/source/gpu.rst:`Chainer's CuPy <https://cupy.chainer.org/>`_ library provides a GPU
docs/source/gpu.rst:There are a variety of GPU accelerated machine learning libraries that follow
docs/source/gpu.rst:GPU-backed libraries isn't very different from using it with CPU-backed
docs/source/gpu.rst:However if your tasks primarily use a GPU then you probably want far fewer
docs/source/gpu.rst:    tasks as GPU tasks so that the scheduler will limit them, while leaving the
docs/source/gpu.rst:Specifying GPUs per Machine
docs/source/gpu.rst:Some configurations may have many GPU devices per node.  Dask is often used to
docs/source/gpu.rst:the CUDA environment variable ``CUDA_VISIBLE_DEVICES`` to pin each worker to
docs/source/gpu.rst:   # If we have four GPUs on one machine
docs/source/gpu.rst:   CUDA_VISIBLE_DEVICES=0 dask-worker ...
docs/source/gpu.rst:   CUDA_VISIBLE_DEVICES=1 dask-worker ...
docs/source/gpu.rst:   CUDA_VISIBLE_DEVICES=2 dask-worker ...
docs/source/gpu.rst:   CUDA_VISIBLE_DEVICES=3 dask-worker ...
docs/source/gpu.rst:The `Dask CUDA <https://github.com/rapidsai/dask-cuda>`_ project contains some
docs/source/gpu.rst:GPU computing is a quickly moving field today and as a result the information
docs/source/ecosystem.rst:- `cupy <https://docs.cupy.dev/en/stable>`_: Part of the Rapids project, GPU-enabled arrays
docs/source/ecosystem.rst:  can be used as the blocks of Dask Arrays. See the section :doc:`gpu` for more information.
docs/source/ecosystem.rst:  GPU-enabled dataframes which can be used as partitions in Dask Dataframes.
docs/source/ecosystem.rst:  and Dask, for execution on CUDA/GPU-enabled hardware, including referencing
docs/source/ecosystem.rst:  The API matches blazingSQL but it uses CPU instead of GPU. It still under development
docs/source/ecosystem.rst:- `dask-cuda <https://github.com/rapidsai/dask-cuda>`_: Construct a Dask cluster which resembles ``LocalCluster``  and is specifically
docs/source/ecosystem.rst:  optimized for GPUs.
docs/source/deploying-cloud.rst:        - Easy access to any cloud hardware (like GPUs) in any region
docs/source/how-to/index.rst:   Use GPUs <../gpu.rst>
continuous_integration/gpuci/axis.yaml:CUDA_VER:
continuous_integration/gpuci/build.sh:# Dask GPU build and test script for CI      #
continuous_integration/gpuci/build.sh:export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
continuous_integration/gpuci/build.sh:# Determine CUDA release version
continuous_integration/gpuci/build.sh:export CUDA_REL=${CUDA_VERSION%.*}
continuous_integration/gpuci/build.sh:rapids-logger "Check GPU usage"
continuous_integration/gpuci/build.sh:nvidia-smi
continuous_integration/gpuci/build.sh:DASK_DATAFRAME__QUERY_PLANNING=False py.test "$WORKSPACE/dask/dataframe" -n 3 -v -m gpu --junitxml="$WORKSPACE/junit-dask-legacy.xml" --cov-config="$WORKSPACE/pyproject.toml" --cov=dask --cov-report=xml:"$WORKSPACE/dask-coverage-legacy.xml" --cov-report term
continuous_integration/gpuci/build.sh:DASK_DATAFRAME__QUERY_PLANNING=True py.test $WORKSPACE -n 3 -v -m gpu --junitxml="$WORKSPACE/junit-dask.xml" --cov-config="$WORKSPACE/pyproject.toml" --cov=dask --cov-report=xml:"$WORKSPACE/dask-coverage.xml" --cov-report term
pyproject.toml:    "gpu: marks tests we want to run on GPUs",
CODEOWNERS:# GPU Support
CODEOWNERS:continuous_integration/gpuci/*  @jacobtomlinson @quasiben

```
