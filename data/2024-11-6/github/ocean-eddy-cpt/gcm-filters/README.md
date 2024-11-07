# https://github.com/ocean-eddy-cpt/gcm-filters

```console
paper.bib:  title        = "CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations",
docs/index.rst:Through integration with `dask <https://dask.org/>`_, GCM-Filters enables parallel, out-of-core filter analysis on both CPUs and GPUs.
docs/index.rst:   gpu
gcm_filters/filter.py:from .gpu_compat import get_array_module
gcm_filters/kernels.py:from .gpu_compat import ArrayType, get_array_module
gcm_filters/gpu_compat.py:"""GPU compatibility stuff."""
README.md:Through integration with [dask](https://dask.org/), GCM-Filters enables parallel, out-of-core filter analysis on both CPUs and GPUs.
paper.md:An important goal of `GCM-Filters` is to enable computationally efficient filtering. The user can employ `GCM-Filters` on either CPUs or GPUs, with `NumPy` [@harris2020array] or `CuPy` [@cupy2017learningsys] input data. `GCM-Filters` leverages `Dask` [@dask] and `Xarray` [@hoyer2017xarray] to support filtering of larger-than-memory datasets and computational flexibility.

```
