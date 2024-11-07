# https://github.com/MNiwano/Eclaire

```console
setup.py:desc  = 'Eclaire: CUDA-based Library for Astronomical Image REduction'
setup.py:and their processing speed is acceralated by using GPU via CUDA.
setup.py:def get_cuda_version():
setup.py:    cuda_version = get_cuda_version()
setup.py:    if cuda_version is not None:
setup.py:                'cupy-cuda{}'.format(cuda_version.replace('.',''))
setup.py:        keywords = ['astronomy', 'science', 'fits', 'GPU', 'CUDA'],
setup.py:            'Environment :: GPU'
setup.py:            'Environment :: GPU :: NVIDIA CUDA',
eclaire/io.py:        mempool : cupy.cuda.MemoryPool, default None
eclaire/io.py:            mempool = cp.cuda.get_allocator().__self__
eclaire/io.py:        elif not isinstance(mempool,cp.cuda.MemoryPool):
eclaire/io.py:            raise TypeError('mempool must be cupy.cuda.MemoryPool')
eclaire/reproject/reproject.py:        * 'eclaire' - it is performed by functions implemented with GPU.
eclaire/__init__.py:Eclaire: CUDA-based Library for Astronomical Image REduction
eclaire/__init__.py:and their processing speed is acceralated by using GPU via CUDA.
eclaire/__init__.py:    1. NVIDIA GPU
eclaire/__init__.py:    2. CUDA
README.md:Eclaire : CUDA-based Library for Astronomical Image REduction
README.md:and their processing speed is acceralated by using GPU via CUDA.
README.md:* NVIDIA GPU
README.md:* CUDA

```
