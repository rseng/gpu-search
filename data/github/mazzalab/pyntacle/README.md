# https://github.com/mazzalab/pyntacle

```console
pyntacletests/test_widgets_metrics.py:        # CPU, GPU, igraph coherence check
pyntacletests/test_widgets_metrics.py:        if cuda_avail:
pyntacletests/test_widgets_metrics.py:            implementation = CmodeEnum.gpu
pyntacletests/test_widgets_metrics.py:            gpu_result = ShortestPath.average_global_shortest_path_length(graph, implementation)
pyntacletests/test_widgets_metrics.py:            self.assertEqual(igraph_result, gpu_result,
pyntacletests/test_widgets_metrics.py:                             'Discrepancy between igraph and gpu result, global case')
docs/installation.rst:CUDA support
docs/installation.rst:Independently of the OS in use, if you need CUDA support, you should also install the CUDA toolkit by downloading and installing the Toolkit from the `NVIDIA website <https://developer.nvidia.com/cuda-toolkit>`_.
README.md:### CUDA support (experimental)
README.md:Independently of the OS in use, if you need CUDA support, you must
README.md:also install the CUDA toolkit by downloading and installing the Toolkit from the
README.md:[_NVIDIA website_](https://developer.nvidia.com/cuda-toolkit).
README.md:**NOTE** GPU-base processing is an **experimental** feature in the current version (1.3), and is not covered by the command-line interface. This is because of weird behaviors of Numba with some hardware configurations that we were not be able to describe and circumvent so far. Although currently accessible by APIs, the GPU feature will be stable in the release 2.0, when Pyntacle will have covered the possibility to manage huge matrices for which replacing fine-grained parallelism with GPU computing would make sense.
README.md:- GPU-based computation of the shortest paths using the Floyd-Warshall algorithm is now an experimental feature and is disabvled in the Pyntacle command line. Users can choose to override this behavior in the Pyntacle library by using the correct Cmode enumerator
tools/enums.py:        * ``gpu``: same as ``cpu``, but using parallel GPU processing (enabled when a CUDA-supported device is present). This method returns a matrix (:py:class:`numpy.ndarray`) of shortest paths. Infinite distances actually equal the total number of vertices plus one.
tools/enums.py:        .. warning:: Use this implementation **only** if the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ is installed on your machine and your CUDA device has CUDA 2.0 onwards
tools/enums.py:    gpu = 3
algorithms/shortest_path_gpu.py:from numba import cuda, int32, uint16
algorithms/shortest_path_gpu.py:@cuda.jit(device=True)
algorithms/shortest_path_gpu.py:def cuda_min(a: int32, b: int32):
algorithms/shortest_path_gpu.py:@cuda.jit('void(uint16[:, :], int32, int32)')
algorithms/shortest_path_gpu.py:def shortest_path_gpu(adjmat: np.ndarray, k:np.int32, N:int32):
algorithms/shortest_path_gpu.py:    The overall calculation is delegated to the GPU, if available, through the NUMBA python package.
algorithms/shortest_path_gpu.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
algorithms/shortest_path_gpu.py:    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
algorithms/shortest_path_gpu.py:                adjmat[i, j] = cuda_min(posIK + posKJ, posIJ)
algorithms/shortest_path_gpu.py:@cuda.jit('void(uint16[:, :], uint16[:, :])')
algorithms/shortest_path_gpu.py:def shortest_path_count_gpu(adjmat, count):
algorithms/shortest_path_gpu.py:    Floyd-Warshall algorithm with path count. The overall calculation is delegated to the GPU, if available, through
algorithms/shortest_path_gpu.py:    i = cuda.grid(1)
algorithms/shortest_path.py:from numba import jit, prange, cuda
algorithms/shortest_path.py:        elif cmode == CmodeEnum.cpu or cmode == CmodeEnum.gpu:
algorithms/shortest_path.py:            elif cmode == CmodeEnum.gpu:
algorithms/shortest_path.py:                if cmode == CmodeEnum.gpu and cuda.current_context().get_memory_info().free < (graph.vcount() ** 2) * 2:
algorithms/shortest_path.py:                        u"WARNING: GPU Memory seems to be low; loading the graph given as input could fail.\n")
algorithms/shortest_path.py:                if "shortest_path_gpu" not in sys.modules:
algorithms/shortest_path.py:                    from algorithms.shortest_path_gpu import shortest_path_gpu
algorithms/shortest_path.py:                d_sps = cuda.to_device(sps)
algorithms/shortest_path.py:                    shortest_path_gpu[blockspergrid, threadsperblock](d_sps, k, N)
algorithms/shortest_path.py:            elif cmode == CmodeEnum.gpu:
algorithms/shortest_path.py:                if cuda.current_context().get_memory_info().free < (graph.vcount() ** 2) * 2:
algorithms/shortest_path.py:                        u"WARNING: GPU Memory seems to be low; loading the graph given as input could fail.")
algorithms/shortest_path.py:                if "shortest_path_count_gpu" not in sys.modules:
algorithms/shortest_path.py:                    from algorithms.shortest_path_gpu import shortest_path_count_gpu
algorithms/shortest_path.py:                shortest_path_count_gpu[blockspergrid, tpb](adj_mat, count_all)
algorithms/__init__.py:* :class:`~pyntacle.algorithms.shortest_path_gpu`: uses GPU acceleration by means of `numba <http://numba.pydata.org/>`_ to compute a matrix of distances
algorithms/__init__.py:.. warning:: Import this module **only** if you have a CUDA compatible GPU and the `Cuda Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ is installed
config.py:from numba import cuda
config.py:cuda_avail = cuda.is_available()
__init__.py:if cuda_avail:

```
