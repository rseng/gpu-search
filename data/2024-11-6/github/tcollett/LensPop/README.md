# https://github.com/tcollett/LensPop

```console
imageSim/convolve.py:    from pyfft.cuda import Plan
imageSim/convolve.py:    import pycuda.driver as cuda
imageSim/convolve.py:    from pycuda.tools import make_default_context
imageSim/convolve.py:    import pycuda.gpuarray as gpuarray
imageSim/convolve.py:    cuda.init()
imageSim/convolve.py:    stream = cuda.Stream()
imageSim/convolve.py:    gdata = gpuarray.to_gpu(boxp.astype(numpy.complex64))
imageSim/convolve.py:    from pyfft.cuda import Plan
imageSim/convolve.py:    import pycuda.driver as cuda
imageSim/convolve.py:    from pycuda.tools import make_default_context
imageSim/convolve.py:    import pycuda.gpuarray as gpuarray
imageSim/convolve.py:    gdata = gpuarray.to_gpu(im.astype(numpy.complex64))

```
