# https://github.com/mikecokina/elisa

```console
requirements.txt:# torch==1.7.0   +cuda
src/elisa/managers/settings_manager.py:    CUDA = False
src/elisa/tensor/README.rst:nvidia-cuda-toolkit 11.3
src/elisa/tensor/README.rst:.. _Official_nVIDIA_guide: https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local
src/elisa/tensor/README.rst:    - Official_nVIDIA_guide_
src/elisa/tensor/README.rst:    ### nvidia-cuda-toolkit 11.3 ###
src/elisa/tensor/README.rst:        PATH=/usr/local/cuda-11.3/bin
src/elisa/tensor/README.rst:        PATH=/usr/local/cuda-11.3/bin:${PATH}
src/elisa/tensor/README.rst:        LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/local/cuda-11.3/targets/x86_64-linux/lib
src/elisa/tensor/README.rst:        LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/local/cuda-11.3/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
src/elisa/tensor/README.rst:    LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/local/cuda-11.3/targets/x86_64-linux/lib
src/elisa/tensor/etensor.py:if settings.CUDA:
src/elisa/conf/settings.py:            "CUDA": cls.CUDA,
src/elisa/conf/settings.py:        cls.CUDA = False
src/elisa/conf/settings.py:            cls.CUDA = c_parse.getboolean('computational', 'cuda', fallback=cls.CUDA)
src/elisa/conf/settings.py:            if cls.CUDA:
src/elisa/conf/settings.py:                    if not cumpy.cuda.is_available():
src/elisa/conf/settings.py:                        cls.CUDA = False
src/elisa/conf/settings.py:                        warnings.warn("You have no CUDA enabled/available on your device. "
src/elisa/conf/settings.py:                    warnings.warn("You need to install `pytorch` with cuda to be "
src/elisa/conf/settings.py:                                  "able to use CUDA features. Fallback to CPU.", UserWarning)
src/elisa/conf/settings.py:                    cls.CUDA = False
CHANGELOG.rst:    - utilizing numba for computationally heavy tasks such as reflection effect (preparation for GPU ready version of

```
