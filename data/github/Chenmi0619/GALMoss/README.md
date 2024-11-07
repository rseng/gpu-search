# https://github.com/Chenmi0619/GALMoss

```console
README.rst:**GalMOSS** a Python-based, Torch-powered tool for two-dimensional fitting of galaxy profiles. By seamlessly enabling GPU parallelization, **GalMOSS** meets the high computational demands of large-scale galaxy surveys, placing galaxy profile fitting in the CSST/LSST-era. It incorporates widely used profiles such as the Sérsic, Exponential disk, Ferrer, King, Gaussian, and Moffat profiles, and allows for the easy integration of more complex models. 
docs/index.rst:**GalMOSS** a Python-based, Torch-powered tool for two-dimensional fitting of galaxy profiles. By seamlessly enabling GPU parallelization, **GalMOSS** meets the high computational demands of large-scale galaxy surveys, placing galaxy profile fitting in the CSST/LSST-era. It incorporates widely used profiles such as the Sérsic, Exponential disk, Ferrer, King, Gaussian, and Moffat profiles, and allows for the easy integration of more complex models. 
galmoss/data.py:        device: str = 'cuda'
galmoss/data.py:            Calculate the fitting process on CPU or GPU.
galmoss/Parameter/basic.py:            user-defined devices, typically GPU (cuda) by default.
galmoss/Parameter/basic.py:        if hasattr(self, 'uncertainty_gpu'):
galmoss/Parameter/basic.py:                (self.uncertainty_gpu, real_uc)
galmoss/fitting.py:        number should as big as possible until the memory usage and GPU

```
