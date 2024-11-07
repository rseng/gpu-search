# https://github.com/pymatting/pymatting

```console
pymatting/foreground/estimate_foreground_ml_pyopencl.py:import pyopencl as cl
pymatting/foreground/estimate_foreground_ml_pyopencl.py:device = platform.get_devices(cl.device_type.GPU)[0]
pymatting/foreground/estimate_foreground_ml_pyopencl.py:def estimate_foreground_ml_pyopencl(
tests/test_foreground.py:        from pymatting.foreground.estimate_foreground_ml_pyopencl import (
tests/test_foreground.py:            estimate_foreground_ml_pyopencl,
tests/test_foreground.py:            estimate_foreground_ml_pyopencl,
tests/test_foreground.py:            "Tests for GPU implementation skipped, because of missing packages."
doc/source/index.md:A warning will be thrown if PyOpenCL or CuPy are not available.
doc/source/index.md:## Additional Requirements (for GPU support)
doc/source/index.md:* cupy-cuda90>=6.5.0 or similar
doc/source/index.md:* pyopencl>=2019.1.2
README.md:  - Fast Multi-Level Foreground Estimation (CPU, CUDA and OpenCL) [[7]](#7)
README.md:Additional requirements for GPU support
README.md:* cupy-cuda90>=6.5.0 or similar
README.md:* pyopencl>=2019.1.2
requirements_gpu.txt:cupy-cuda90>=6.5.0
requirements_gpu.txt:pyopencl>=2019.1.2

```
