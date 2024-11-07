# https://github.com/undark-lab/swyft

```console
setup.cfg:    Environment :: GPU
docs/source/old/goals.rst:  optimization on GPU clusters.
docs/source/old/quickstart.rst:..     DEVICE = 'cuda' #your gpu, or 'cpu' if a gpu is not available
swyft/lightning/online.py:        This has many subtleties when the simulator uses CUDA and ``num_workers``
swyft/utils/misc.py:def is_cuda_available() -> bool:
swyft/utils/misc.py:    return torch.cuda.is_available()
swyft/utils/__init__.py:from swyft.utils.misc import depth, is_cuda_available, is_empty
swyft/utils/__init__.py:    "is_cuda_available",

```
