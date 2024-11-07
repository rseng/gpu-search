# https://github.com/guillochon/MOSFiT

```console
mosfit/strings.json:    "cuda_enabled": "!mCUDA successfully initialized.!e",
mosfit/strings.json:    "cuda_not_enabled": "CUDA not successfully initialized, please verify that CUDA is installed and that the `pycuda` and `skcuda` Python packages are available.",
mosfit/strings.json:    "parser_cuda": "Enable CUDA for MOSFiT routines. Requires the `scikit-cuda` package (and its dependencies) to be installed.",
mosfit/fitter.py:                 cuda=False,
mosfit/fitter.py:        self._cuda = cuda
mosfit/fitter.py:        if self._cuda:
mosfit/fitter.py:                import pycuda.autoinit  # noqa: F401
mosfit/fitter.py:                import skcuda.linalg as linalg
mosfit/modules/objectives/likelihood.py:        self._cuda_reported = False
mosfit/modules/objectives/likelihood.py:        if not self._model._fitter._cuda:
mosfit/modules/objectives/likelihood.py:            if self._use_cpu is not True and self._model._fitter._cuda:
mosfit/modules/objectives/likelihood.py:                    import pycuda.gpuarray as gpuarray
mosfit/modules/objectives/likelihood.py:                    import skcuda.linalg as skla
mosfit/modules/objectives/likelihood.py:                    if not self._cuda_reported:
mosfit/modules/objectives/likelihood.py:                            'cuda_not_enabled', master_only=True, warning=True)
mosfit/modules/objectives/likelihood.py:                    if not self._cuda_reported:
mosfit/modules/objectives/likelihood.py:                        self._printer.message('cuda_enabled', master_only=True)
mosfit/modules/objectives/likelihood.py:                        self._cuda_reported = True
mosfit/modules/objectives/likelihood.py:                    kmat_gpu = gpuarray.to_gpu(kmat)
mosfit/modules/objectives/likelihood.py:                    skla.cholesky(kmat_gpu, lib='cusolver')
mosfit/modules/objectives/likelihood.py:                    value = -np.log(skla.det(kmat_gpu, lib='cusolver'))
mosfit/modules/objectives/likelihood.py:                    res_gpu = gpuarray.to_gpu(residuals.reshape(
mosfit/modules/objectives/likelihood.py:                    cho_mat_gpu = res_gpu.copy()
mosfit/modules/objectives/likelihood.py:                    skla.cho_solve(kmat_gpu, cho_mat_gpu, lib='cusolver')
mosfit/modules/objectives/likelihood.py:                        skla.mdot(skla.transpose(res_gpu),
mosfit/modules/objectives/likelihood.py:                                  cho_mat_gpu)).get())[0][0]
mosfit/main.py:        '--cuda',
mosfit/main.py:        dest='cuda',
mosfit/main.py:        help=prt.text('parser_cuda'))

```
