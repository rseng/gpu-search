# https://github.com/saopicc/killMS

```console
killMS/Array/Dot/NpCuda.py:import pycuda.autoinit
killMS/Array/Dot/NpCuda.py:import pycuda.gpuarray as gpuarray
killMS/Array/Dot/NpCuda.py:import scikits.cuda.linalg as culinalg
killMS/Array/Dot/NpCuda.py:import cudamat as cm
killMS/Array/Dot/NpCuda.py:    # create two random matrices and copy them to the GPU
killMS/Array/Dot/NpCuda.py:    g_A0 = cm.CUDAMatrix(A)
killMS/Array/Dot/NpCuda.py:    g_AT0 = cm.CUDAMatrix(AT)
killMS/Array/Dot/NpCuda.py:    # perform calculations on the GPU
killMS/Array/Dot/NpCuda.py:    T.timeit("GPU0")
killMS/Array/Dot/NpCuda.py:    g_A1 = gpuarray.to_gpu(A)
killMS/Array/Dot/NpCuda.py:    g_AT1 = gpuarray.to_gpu(AT)
killMS/Array/Dot/NpCuda.py:    T.timeit("GPU1")

```
