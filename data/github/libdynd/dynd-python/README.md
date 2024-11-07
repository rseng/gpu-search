# https://github.com/libdynd/dynd-python

```console
dynd/benchmarks/benchmark_arithmetic.py:from benchtime import Timer, CUDATimer
dynd/benchmarks/benchmark_arithmetic.py:  def __init__(self, op, cuda = False):
dynd/benchmarks/benchmark_arithmetic.py:    self.cuda = cuda
dynd/benchmarks/benchmark_arithmetic.py:    if self.cuda:
dynd/benchmarks/benchmark_arithmetic.py:      dst_tp = ndt.type('cuda_device[{} * float64]'.format(size))
dynd/benchmarks/benchmark_arithmetic.py:    with CUDATimer() if self.cuda else Timer() as timer:
dynd/benchmarks/benchmark_arithmetic.py:class PyCUDAArithmeticBenchmark(Benchmark):
dynd/benchmarks/benchmark_arithmetic.py:    from pycuda import curandom
dynd/benchmarks/benchmark_arithmetic.py:    with CUDATimer() as timer:
dynd/benchmarks/benchmark_arithmetic.py:  cuda = True
dynd/benchmarks/benchmark_arithmetic.py:  benchmark = ArithmeticBenchmark(add, cuda = False)
dynd/benchmarks/benchmark_arithmetic.py:#  if cuda:
dynd/benchmarks/benchmark_arithmetic.py: #   benchmark = PyCUDAArithmeticBenchmark(add)
dynd/benchmarks/benchmark_random.py:from benchtime import Timer, CUDATimer
dynd/benchmarks/benchmark_random.py:  def __init__(self, cuda = False):
dynd/benchmarks/benchmark_random.py:    self.cuda = cuda
dynd/benchmarks/benchmark_random.py:    if self.cuda:
dynd/benchmarks/benchmark_random.py:      dst_tp = ndt.type('cuda_device[{} * float64]'.format(size))
dynd/benchmarks/benchmark_random.py:    with CUDATimer() if self.cuda else Timer() as timer:
dynd/benchmarks/benchmark_random.py:class PyCUDAUniformBenchmark(Benchmark):
dynd/benchmarks/benchmark_random.py:    with CUDATimer() as timer:
dynd/benchmarks/benchmark_random.py:  cuda = True
dynd/benchmarks/benchmark_random.py:  benchmark = UniformBenchmark(cuda = cuda)
dynd/benchmarks/benchmark_random.py:  if cuda:
dynd/benchmarks/benchmark_random.py:    from pycuda import curandom
dynd/benchmarks/benchmark_random.py:    benchmark = PyCUDAUniformBenchmark(curandom.XORWOWRandomNumberGenerator())
dynd/benchmarks/benchtime.py:class CUDATimer(object):
dynd/benchmarks/benchtime.py:    from dynd import cuda
dynd/benchmarks/benchtime.py:    self.start = cuda.event()
dynd/benchmarks/benchtime.py:    self.stop = cuda.event()
dynd/benchmarks/__init__.py:  from pycuda import autoinit
dynd/nd/test/test_lowlevel.py:        # CUDA types
dynd/nd/test/test_lowlevel.py:#        if ndt.cuda_support:
dynd/nd/test/test_lowlevel.py: #           self.assertEqual(self.type_id_of(ndt.type('cuda_device[int32]')),
dynd/nd/test/test_lowlevel.py:  #                           _lowlevel.type_id.CUDA_DEVICE)
dynd/nd/test/test_lowlevel.py:   #         self.assertEqual(self.type_id_of(ndt.type('cuda_host[int32]')),
dynd/nd/test/test_lowlevel.py:    #                         _lowlevel.type_id.CUDA_HOST)
.gitignore:build_cuda

```
