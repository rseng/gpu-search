# https://github.com/thomgrand/fim-python

```console
setup.py:lib_requires_gpu = ["cupy>=9.0"]
setup.py:    description="This repository implements the Fast Iterative Method on tetrahedral domains and triangulated surfaces purely in python both for CPU (numpy) and GPU (cupy).",
setup.py:        "Environment :: GPU :: NVIDIA CUDA",
setup.py:          'gpu': lib_requires_gpu,
docs/example.inc:    #Create a FIM solver, by default the GPU solver will be called with the active list
docs/benchmark.rst:.. image:: figs/benchmark_gpu.jpg
docs/benchmark.rst:    :alt: Benchmark GPU
docs/benchmark.rst:.. image:: figs/benchmark_gpu_setup.jpg
docs/benchmark.rst:    :alt: Setup Time GPU
docs/benchmark.rst:    pip install fim-python[gpu,tests,docs]
docs/interface.rst:Note that if you specify the gpu interface, but your system does not support it (or you did not install it), you will only get a cpu solver.
docs/installation.rst:    pip install -e .[gpu]
docs/installation.rst:    pip install fim-python[gpu]
docs/installation.rst:    Installing the GPU version might take a while since many ``cupy`` modules are compiled using your system's ``nvcc`` compiler.
docs/detailed_description.rst:This version of the algorithm is bested suited for the GPU, since it is optimal for a SIMD (single instruction multiple data) architecture.
docs/detailed_description.rst:The active list method computes much fewer updates, but has the additional overhead of keeping track of its active list, ill-suited for the GPU.
fimpy/fim_cupy.py:"""This file contains the GPU implementation of the Fast Iterative Method, based on cupy.
fimpy/fim_cupy.py:  """This class implements the Fast Iterative Method on the GPU using cupy.
fimpy/fim_cupy.py:    self.streams = [cp.cuda.Stream(non_blocking=False) for i in range(4)]
fimpy/fim_cupy.py:  def free_gpu_mem(self):
fimpy/fim_cupy.py:  """This class implements the Fast Iterative Method on the GPU using cupy.
fimpy/fim_cupy.py:    self.streams = [cp.cuda.Stream(non_blocking=False) for i in range(7)]
fimpy/fim_cupy.py:  def free_gpu_mem(self):
fimpy/solver.py:    print("Import of Cupy failed. The GPU version of fimpy will be unavailable. Message: %s" % (err))
fimpy/solver.py:        Specifies the target device for the computations. One of [cpu, gpu], by default 'gpu'
fimpy/solver.py:    assert not device == 'gpu' or cupy_available, "Requested GPU which is not available"
fimpy/solver.py:    elif device == 'gpu':
fimpy/solver.py:        assert False,  f"Unknown device {device}, should be one of [cpu, gpu]"
fimpy/utils/__init__.py:"""This subpackage contains small custom functions to efficiently compute :math:`\\left<A \\mathbf{x}_1, \\mathbf{x}_2 \\right>` on the CPU and GPU for different dimensions :math:`d`.
fimpy/cupy_kernels.py:"""This file contains some custom CUDA kernels, used in the CUDA implementation of FIMPY
fimpy/cupy_kernels.py:''') #: CUDA kernel to compute a mask of all element permutations containing at least one active index. Old, less inefficient version not using shared memory.
fimpy/cupy_kernels.py:  *        the CUDA device. Assumes the range to be sorted, but has O(log n) runtime in return.
fimpy/cupy_kernels.py:}''') #: CUDA kernel to compute a mask of all element permutations containing at least one active index. New, more efficient version using shared memory.
tests/generate_doc_figs.py:    #Create a FIM solver, by default the GPU solver will be called with the active list
tests/generate_doc_figs.py:    #GPU
tests/generate_doc_figs.py:    for device in ["gpu", "cpu"]:
tests/run_benchmark.py:#from cupy.cuda.memory import OutOfMemoryError
tests/run_benchmark.py:from cupy.cuda.runtime import CUDARuntimeError
tests/run_benchmark.py:        if device == 'gpu':
tests/run_benchmark.py:            cp.cuda.runtime.deviceSynchronize()
tests/run_benchmark.py:    for device in ['cpu', 'gpu']:
tests/run_benchmark.py:                    #if dims > 5 and elem_dims > 2 and use_active_list: #elem_dims == 4 and use_active_list and dims > 3 and resolution > 10 and device == 'gpu':
tests/run_benchmark.py:                        except CUDARuntimeError as ex:
tests/test_custom_kernels.py:    @pytest.mark.skipif(not cupy_enabled, reason='Cupy could not be imported. GPU tests unavailable')
tests/test_custom_kernels.py:    def test_perm_kernel_gpu2(self, elem_dims, resolution, nr_active_inds, parallel_blocks, threads_x, threads_y, shared_buf_size):
tests/test_fim_solvers.py:    @pytest.mark.parametrize('device', ['cpu', 'gpu'])
tests/test_fim_solvers.py:        if device == 'gpu' and not cupy_enabled:
tests/test_fim_solvers.py:            pytest.skip(reason='Cupy could not be imported. GPU tests unavailable')
tests/test_fim_solvers.py:    @pytest.mark.skipif(not cupy_enabled, reason='Cupy could not be imported. GPU tests unavailable')
tests/test_fim_solvers.py:    def test_error_init_gpu(self, precision):
tests/test_fim_solvers.py:        self.test_error_init(precision, 'gpu')
tests/test_fim_solvers.py:    @pytest.mark.parametrize('device', ['cpu', 'gpu'])
tests/test_fim_solvers.py:    @pytest.mark.skipif(not cupy_enabled, reason='Cupy could not be imported. GPU tests unavailable')
tests/test_fim_solvers.py:    def test_comp_gpu(self, dims, elem_dims, precision, use_active_list):
tests/test_fim_solvers.py:        self.test_comp(dims, elem_dims, precision, use_active_list=use_active_list, device='gpu')
README.md:This repository implements the Fast Iterative Method on [tetrahedral domains](https://epubs.siam.org/doi/abs/10.1137/120881956) and [triangulated surfaces](https://epubs.siam.org/doi/abs/10.1137/100788951) purely in python both for CPU (numpy) and GPU (cupy). The main focus is however on the GPU implementation, since it can be better exploited for very large domains.
README.md:pip install fim-python[gpu] #GPU version
README.md:If you don't have a compatible CUDA GPU, you can install the CPU only version to test the library, but the performance won't be comparable to the GPU version (see [Benchmark](#benchmark)).
README.md:#Create a FIM solver, by default the GPU solver will be called with the active list
README.md:#Create a FIM solver, by default the GPU solver will be called with the active list
README.md:![Preview](docs/figs/benchmark_gpu.jpg)
CONTRIBUTING.md:    print("GPU version, version of cupy: %s" % (cupy.__version__))
CONTRIBUTING.md:pip install fim-python[gpu,tests]
CONTRIBUTING.md:In case you only have the CPU version, all tests for the GPU will be skipped. 
CONTRIBUTING.md:The github-runner will also test pull-requests and committed versions of the library, but only on the CPU for the lack of a GPU on the runner.
CONTRIBUTING.md:> **_Note:_**  If you do **not** have a Cupy compatible GPU to test on, please clearly state this in your pull request, so somebody else from the community can test your code with all features enabled.
paper.md:  - cuda
paper.md:The method is implemented both on the CPU using [``numba``](https://numba.pydata.org/) and [``numpy``](https://numpy.org/), as well as the GPU with the help of [``cupy``](https://cupy.dev/) (depends on [CUDA](https://developer.nvidia.com/cuda-toolkit)).
paper.md:This version of the algorithm is bested suited for the GPU, since it is optimal for a SIMD (single instruction multiple data) architecture.
paper.md:[``GPUTUM: Unstructured Eikonal``](https://github.com/SCIInstitute/SCI-Solver_Eikonal) implements the FIM in CUDA for triangulated surfaces and tetrahedral meshes, but has no Python bindings and is designed as a command line tool for single evaluations.
paper.md:``fim-python`` tries to wrap the FIM for CPU and GPU into an easy-to-use Python package for multiple evaluations with a straight-forward installation over [PyPI](https://pypi.org/).

```
