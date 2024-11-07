# https://github.com/PyLops/pyproximal

```console
docs/source/changelog.rst:* Added cuda version to the proximal operator of :py:class:`pyproximal.proximal.Simplex`
CHANGELOG.md:* Added cuda version to the proximal operator of ``pyproximal.proximal.Simplex`` 
pyproximal/proximal/Simplex.py:    from ._Simplex_cuda import bisect_jit_cuda, simplex_jit_cuda, fun_jit_cuda
pyproximal/proximal/Simplex.py:class _Simplex_cuda(_Simplex):
pyproximal/proximal/Simplex.py:    """Simplex operator (cuda version)
pyproximal/proximal/Simplex.py:        # self.coeffs = cuda.to_device(np.ones(self.n if dims is None else dims[axis]))
pyproximal/proximal/Simplex.py:        simplex_jit_cuda[num_blocks, self.num_threads_per_blocks](x, self.coeffs, self.radius,
pyproximal/proximal/Simplex.py:        Function tolerance in bisection (only with ``engine='numba'`` or ``engine='cuda'``)
pyproximal/proximal/Simplex.py:        Engine used for simplex computation (``numpy``, ``numba``or ``cuda``).
pyproximal/proximal/Simplex.py:        If ``engine`` is neither ``numpy`` nor ``numba`` nor ``cuda``
pyproximal/proximal/Simplex.py:    if not engine in ['numpy', 'numba', 'cuda']:
pyproximal/proximal/Simplex.py:        raise KeyError('engine must be numpy or numba or cuda')
pyproximal/proximal/Simplex.py:    elif engine == 'cuda' and jit is not None:
pyproximal/proximal/Simplex.py:        s = _Simplex_cuda(n, radius, dims=dims, axis=axis,
pyproximal/proximal/_Simplex_cuda.py:from numba import cuda
pyproximal/proximal/_Simplex_cuda.py:@cuda.jit(device=True)
pyproximal/proximal/_Simplex_cuda.py:def fun_jit_cuda(mu, x, coeffs, scalar, lower, upper):
pyproximal/proximal/_Simplex_cuda.py:@cuda.jit(device=True)
pyproximal/proximal/_Simplex_cuda.py:def bisect_jit_cuda(x, coeffs, scalar, lower, upper, bisect_lower, bisect_upper,
pyproximal/proximal/_Simplex_cuda.py:    fa = fun_jit_cuda(a, x, coeffs, scalar, lower, upper)
pyproximal/proximal/_Simplex_cuda.py:        fc = fun_jit_cuda(c, x, coeffs, scalar, lower, upper)
pyproximal/proximal/_Simplex_cuda.py:@cuda.jit
pyproximal/proximal/_Simplex_cuda.py:def simplex_jit_cuda(x, coeffs, scalar, lower, upper, maxiter, ftol, xtol, y):
pyproximal/proximal/_Simplex_cuda.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
pyproximal/proximal/_Simplex_cuda.py:        while fun_jit_cuda(bisect_lower, x[i], coeffs, scalar, lower, upper) < 0:
pyproximal/proximal/_Simplex_cuda.py:        while fun_jit_cuda(bisect_upper, x[i], coeffs, scalar, lower, upper) > 0:
pyproximal/proximal/_Simplex_cuda.py:        c = bisect_jit_cuda(x[i], coeffs, scalar, lower, upper,

```
