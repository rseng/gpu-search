# https://github.com/hpparvi/opents

```console
src/tessts.py:                 min_transits: int = 3, nsamples: int = 1, exptime: float = 0.0, use_tqdm: bool = True, use_opencl: bool = True):
src/tessts.py:        super().__init__(pmin, pmax, nper, bic_limit, min_transits, nsamples, exptime, use_tqdm, use_opencl)
src/k2ts.py:                 min_transits: int = 3, nsamples: int = 1, exptime: float = 0.0, use_tqdm: bool = True, use_opencl: bool = True):
src/k2ts.py:        super().__init__(pmin, pmax, nper, bic_limit, min_transits, nsamples, exptime, use_tqdm, use_opencl)
src/k2k2ts.py:                 use_opencl: bool = True):
src/k2k2ts.py:        super().__init__(pmin, pmax, nper, bic_limit, min_transits, nsamples, exptime, use_tqdm, use_opencl)
src/keplerts.py:                 use_opencl: bool = True):
src/keplerts.py:        super().__init__(pmin, pmax, nper, bic_limit, min_transits, nsamples, exptime, use_tqdm, use_opencl)
src/tfstep.py:    def __init__(self, ts, mode: str, title: str, nsamples: int = 1, exptime: float = 1, use_opencl: bool = False, use_tqdm: bool = True):
src/tfstep.py:        self.use_opencl = use_opencl
src/tfstep.py:        tm = QuadraticModelCL(klims=(0.01, 0.60)) if self.use_opencl else QuadraticModel(interpolate=False)
src/transitsearch.py:                 nsamples: int = 1, exptime: float = 0.0, use_tqdm: bool = True, use_opencl: bool = True):
src/transitsearch.py:        self.use_opencl: bool = use_opencl

```
