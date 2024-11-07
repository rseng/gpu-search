# https://github.com/scikit-learn/scikit-learn

```console
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/conda-forge/noarch/cuda-version-12.4-h3060b56_3.conda#c9a3fe8b957176e1a8452c6f3431b0d8
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/pytorch/noarch/pytorch-mutex-1.0-cuda.tar.bz2#a948316e36fb5b11223b3fcfa93f8358
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/conda-forge/noarch/cuda-cccl_linux-64-12.4.127-ha770c72_2.conda#357fbcd43f1296b02d6738a2295abc24
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/conda-forge/noarch/cuda-cudart-static_linux-64-12.4.127-h85509e4_2.conda#0b0522d8685968f25370d5c36bb9fba3
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/conda-forge/noarch/cuda-cudart_linux-64-12.4.127-h85509e4_2.conda#329163110a96514802e9e64d971edf43
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/conda-forge/noarch/cuda-cudart-dev_linux-64-12.4.127-h85509e4_2.conda#12039deb2a3f103f5756831702bf29fc
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/conda-forge/linux-64/cuda-cudart-12.4.127-he02047a_2.conda#a748faa52331983fc3adcc3b116fe0e4
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/conda-forge/linux-64/cuda-cupti-12.4.127-he02047a_2.conda#46422ef1b1161fb180027e50c598ecd0
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/conda-forge/linux-64/cuda-nvrtc-12.4.127-he02047a_2.conda#80baf6262f4a1a0dde42d85aaa393402
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/conda-forge/linux-64/cuda-nvtx-12.4.127-he02047a_2.conda#656a004b6e44f50ce71c65cab0d429b4
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/conda-forge/linux-64/cuda-opencl-12.4.127-he02047a_1.conda#1e98deda07c14d26c80d124cf0eb011a
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/conda-forge/linux-64/cuda-libraries-12.4.1-ha770c72_1.conda#6bb3f998485d4344a7539e0b218b3fc1
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/conda-forge/noarch/cuda-runtime-12.4.1-ha804496_0.conda#48829f4ef6005ae8d4867b99168ff2b8
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/pytorch/linux-64/pytorch-cuda-12.4-hc786d27_7.tar.bz2#06635b1bbf5e2fef4a8b9b282500cd7b
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock:https://conda.anaconda.org/pytorch/linux-64/pytorch-2.5.1-py3.12_cuda12.4_cudnn9.1.0_0.tar.bz2#42164c6ce8e563c20a542686a8b9b964
build_tools/github/create_gpu_environment.sh:LOCK_FILE=build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_conda.lock
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_environment.yml:  - nvidia
build_tools/github/pylatest_conda_forge_cuda_array-api_linux-64_environment.yml:  - pytorch-cuda
build_tools/update_environments_and_lock_files.py:        "name": "pylatest_conda_forge_cuda_array-api_linux-64",
build_tools/update_environments_and_lock_files.py:        "tag": "cuda",
build_tools/update_environments_and_lock_files.py:        "channels": ["conda-forge", "pytorch", "nvidia"],
build_tools/update_environments_and_lock_files.py:            "pytorch-cuda",
doc/about.rst:.. |nvidia| image:: images/nvidia.png
doc/about.rst:  :target: https://www.nvidia.com
doc/about.rst:      |       |nvidia|       |
doc/about.rst:    `NVidia <https://nvidia.com>`_ funds Tim Head since 2022
doc/about.rst:    .. image:: images/nvidia.png
doc/about.rst:      :target: https://nvidia.com
doc/faq.rst:GPUs for efficient computing. However, neither of these fit within
doc/faq.rst:You can find more information about the addition of GPU support at
doc/faq.rst:`Will you add GPU support?`_.
doc/faq.rst:Will you add GPU support?
doc/faq.rst:Adding GPU support by default would introduce heavy harware-specific software
doc/faq.rst:estimators <array_api_supported>` can already run on GPUs if the input data is
doc/faq.rst:allows scikit-learn to run on GPUs without introducing heavy and
doc/faq.rst:can be considered for Array API support and therefore GPU support.
doc/faq.rst:on GPUs via the Array API for fundamental algorithmic reasons. For instance,
doc/faq.rst:Adding efficient GPU support to estimators that cannot be efficiently
doc/faq.rst:of) MKL, the OpenMP runtime of GCC, nvidia's Cuda (and probably many others),
doc/roadmap.rst:* Computational tools: The exploitation of GPUs, distributed programming
doc/modules/linear_model.rst::class:`OrthogonalMatchingPursuit` and :func:`orthogonal_mp` implement the OMP
doc/modules/neural_networks_supervised.rst:    scikit-learn offers no GPU support. For much faster, GPU-based implementations,
doc/modules/multiclass.rst:- :class:`linear_model.OrthogonalMatchingPursuit`
doc/modules/array_api.rst::class:`~discriminant_analysis.LinearDiscriminantAnalysis` on a GPU::
doc/modules/array_api.rst:    <CUDA Device 0>
doc/modules/array_api.rst:    <CUDA Device 0>
doc/modules/array_api.rst:GPU. We provide a experimental `_estimator_with_converted_arrays` utility that
doc/modules/array_api.rst:    >>> X_torch = torch.asarray(X_np, device="cuda", dtype=torch.float32)
doc/modules/array_api.rst:    >>> y_torch = torch.asarray(y_np, device="cuda", dtype=torch.float32)
doc/modules/array_api.rst:    'cuda'
doc/modules/array_api.rst:a GPU. Checks that can not be executed or have missing dependencies will be
doc/modules/array_api.rst:hardware accelerators (e.g. the internal GPU component of the M1 or M2 chips).
doc/modules/grid_search.rst:   linear_model.OrthogonalMatchingPursuitCV
doc/api_reference.py:                    "OrthogonalMatchingPursuit",
doc/api_reference.py:                    "OrthogonalMatchingPursuitCV",
doc/metadata_routing.rst:- :class:`sklearn.linear_model.OrthogonalMatchingPursuitCV`
doc/whats_new/v1.5.rst:- |Fix| `n_nonzero_coefs_` attribute in :class:`linear_model.OrthogonalMatchingPursuit`
doc/whats_new/v1.0.rst:  :class:`~linear_model.OrthogonalMatchingPursuit` and
doc/whats_new/v1.0.rst:  :class:`~linear_model.OrthogonalMatchingPursuitCV` will default to False in
doc/whats_new/older_versions.rst:  and OrthogonalMatchingPursuit) by `Vlad Niculae`_ and
doc/whats_new/older_versions.rst:- Added :class:`Orthogonal Matching Pursuit <linear_model.OrthogonalMatchingPursuit>` by `Vlad Niculae`_
doc/whats_new/v0.14.rst:- New OrthogonalMatchingPursuitCV class by `Alexandre Gramfort`_
doc/whats_new/v0.14.rst:- Attributes in OrthogonalMatchingPursuit have been deprecated
doc/whats_new/v0.20.rst:- :class:`linear_model.OrthogonalMatchingPursuit` (bug fix)
doc/whats_new/v0.20.rst:- |Fix| Fixed a bug in :class:`linear_model.OrthogonalMatchingPursuit` that was
doc/whats_new/v0.17.rst:  :class:`linear_model.OrthogonalMatchingPursuit`,
doc/whats_new/v1.4.rst:- |Feature| :class:`linear_model.OrthogonalMatchingPursuitCV` now supports
doc/whats_new/v1.4.rst:This therefore enables some GPU-accelerated computations.
sklearn/tests/test_multioutput.py:    OrthogonalMatchingPursuit,
sklearn/tests/test_multioutput.py:    rgr = MultiOutputRegressor(OrthogonalMatchingPursuit())
sklearn/tests/test_metaestimators_metadata_routing.py:    OrthogonalMatchingPursuitCV,
sklearn/tests/test_metaestimators_metadata_routing.py:        "metaestimator": OrthogonalMatchingPursuitCV,
sklearn/linear_model/tests/test_common.py:    OrthogonalMatchingPursuit,
sklearn/linear_model/tests/test_common.py:    OrthogonalMatchingPursuitCV,
sklearn/linear_model/tests/test_common.py:        OrthogonalMatchingPursuit(),
sklearn/linear_model/tests/test_common.py:        OrthogonalMatchingPursuitCV(),
sklearn/linear_model/tests/test_common.py:        OrthogonalMatchingPursuit,
sklearn/linear_model/tests/test_common.py:        OrthogonalMatchingPursuitCV,
sklearn/linear_model/tests/test_ransac.py:    OrthogonalMatchingPursuit,
sklearn/linear_model/tests/test_ransac.py:    estimator = OrthogonalMatchingPursuit()
sklearn/linear_model/tests/test_omp.py:    OrthogonalMatchingPursuit,
sklearn/linear_model/tests/test_omp.py:    OrthogonalMatchingPursuitCV,
sklearn/linear_model/tests/test_omp.py:    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
sklearn/linear_model/tests/test_omp.py:    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
sklearn/linear_model/tests/test_omp.py:    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, tol=0.5)
sklearn/linear_model/tests/test_omp.py:    ompcv = OrthogonalMatchingPursuitCV(fit_intercept=False, max_iter=10)
sklearn/linear_model/tests/test_omp.py:    omp = OrthogonalMatchingPursuit(
sklearn/linear_model/tests/test_omp.py:    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_features)
sklearn/linear_model/_omp.py:    OrthogonalMatchingPursuit : Orthogonal Matching Pursuit model.
sklearn/linear_model/_omp.py:    OrthogonalMatchingPursuit : Orthogonal Matching Pursuit model (OMP).
sklearn/linear_model/_omp.py:class OrthogonalMatchingPursuit(MultiOutputMixin, RegressorMixin, LinearModel):
sklearn/linear_model/_omp.py:    OrthogonalMatchingPursuitCV : Cross-validated
sklearn/linear_model/_omp.py:    >>> from sklearn.linear_model import OrthogonalMatchingPursuit
sklearn/linear_model/_omp.py:    >>> reg = OrthogonalMatchingPursuit().fit(X, y)
sklearn/linear_model/_omp.py:class OrthogonalMatchingPursuitCV(RegressorMixin, LinearModel):
sklearn/linear_model/_omp.py:    OrthogonalMatchingPursuit : Orthogonal Matching Pursuit model (OMP).
sklearn/linear_model/_omp.py:    >>> from sklearn.linear_model import OrthogonalMatchingPursuitCV
sklearn/linear_model/_omp.py:    >>> reg = OrthogonalMatchingPursuitCV(cv=5).fit(X, y)
sklearn/linear_model/_omp.py:        omp = OrthogonalMatchingPursuit(
sklearn/linear_model/__init__.py:    OrthogonalMatchingPursuit,
sklearn/linear_model/__init__.py:    OrthogonalMatchingPursuitCV,
sklearn/linear_model/__init__.py:    "OrthogonalMatchingPursuit",
sklearn/linear_model/__init__.py:    "OrthogonalMatchingPursuitCV",
sklearn/compose/tests/test_target.py:from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit
sklearn/compose/tests/test_target.py:        regressor=OrthogonalMatchingPursuit(), transformer=StandardScaler()
sklearn/utils/_test_common/instance_generator.py:    OrthogonalMatchingPursuitCV,
sklearn/utils/_test_common/instance_generator.py:    OrthogonalMatchingPursuitCV: dict(cv=3),
sklearn/utils/_array_api.py:                ("cpu", "cuda"), ("float64", "float32")
sklearn/utils/tests/test_array_api.py:        xp_out.zeros(10, device="cuda")
sklearn/utils/tests/test_array_api.py:    err_msg = "Input arrays use different devices: cpu, mygpu"
sklearn/utils/tests/test_array_api.py:        device(Array("cpu"), Array("mygpu"))
sklearn/utils/tests/test_array_api.py:def test_convert_to_numpy_gpu(library):  # pragma: nocover
sklearn/utils/tests/test_array_api.py:    """Check convert_to_numpy for GPU backed libraries."""
sklearn/utils/tests/test_array_api.py:        if not xp.backends.cuda.is_built():
sklearn/utils/tests/test_array_api.py:            pytest.skip("test requires cuda")
sklearn/utils/tests/test_array_api.py:        X_gpu = xp.asarray([1.0, 2.0, 3.0], device="cuda")
sklearn/utils/tests/test_array_api.py:        X_gpu = xp.asarray([1.0, 2.0, 3.0])
sklearn/utils/tests/test_array_api.py:    X_cpu = _convert_to_numpy(X_gpu, xp=xp)
sklearn/utils/_testing.py:        and device == "cuda"
sklearn/utils/_testing.py:        and not xp.backends.cuda.is_built()
sklearn/utils/_testing.py:        raise SkipTest("PyTorch test requires cuda, which is not available")
sklearn/utils/_testing.py:        if cupy.cuda.runtime.getDeviceCount() == 0:
sklearn/utils/_testing.py:            raise SkipTest("CuPy test requires cuda, which is not available")
examples/release_highlights/plot_release_highlights_1_5_0.py:# GPU computation if the input data is passed as a PyTorch or CuPy array by
examples/release_highlights/plot_release_highlights_1_2_0.py:# `CuPy <https://docs.cupy.dev/en/stable/overview.html>`__, a GPU-accelerated array
examples/linear_model/plot_omp.py:from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
examples/linear_model/plot_omp.py:omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
examples/linear_model/plot_omp.py:omp_cv = OrthogonalMatchingPursuitCV()

```
