# https://github.com/cmelab/flowerMD

```console
docs/source/installation.rst:    To install a GPU compatible version of HOOMD-blue in your flowerMD environment, you need to manually set the CUDA version **before installing flowermd**.
docs/source/installation.rst:    This is to ensure that the HOOMD build pulled from conda-forge is compatible with your CUDA version.
docs/source/installation.rst:    To set the CUDA version, run the following command before installing flowermd::
docs/source/installation.rst:        $ export CONDA_OVERRIDE_CUDA="[YOUR_CUDA_VERSION]"
README.md:**A note on GPU compatibility:**
README.md:To install a GPU compatible version of HOOMD-blue in your flowerMD
README.md:environment, you need to manually set the CUDA version **before installing flowermd**.
README.md:This is to ensure that the HOOMD build pulled from conda-forge is compatible with your CUDA version.
README.md:To set the CUDA version, run the following command before installing flowermd:
README.md:export CONDA_OVERRIDE_CUDA="[YOUR_CUDA_VERSION]"
flowermd/base/simulation.py:        The CPU or GPU device to use for the simulation.
containers/dockerfile:FROM cmelab/gpuhoomd4conda:latest
containers/dockerfile:    conda env update -n base -f environment-gpu.yml && \

```
