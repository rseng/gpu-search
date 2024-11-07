# https://github.com/NeuralEnsemble/elephant

```console
setup.py:for extra in ['extras', 'docs', 'tests', 'tutorials', 'cuda', 'opencl']:
doc/release_notes.rst:* added CUDA/OpenCL sources for ASSET GPU acceleration to `manifest.in`, they are now included in the distribution package (#483)
doc/release_notes.rst:CUDA and OpenCL support
doc/release_notes.rst:[Analysis of Sequences of Synchronous EvenTs](https://elephant.readthedocs.io/en/latest/reference/asset.html) has become the first module in Elephant that supports CUDA and OpenCL (https://github.com/NeuralEnsemble/elephant/pull/351, https://github.com/NeuralEnsemble/elephant/pull/404, https://github.com/NeuralEnsemble/elephant/pull/399). Whether you have an Nvidia GPU or just run the analysis on a laptop with a built-in Intel graphics card, the speed-up is **X100** and **X1000** compared to a single CPU core. The computations are optimized to a degree that you can analyse and look for spike patterns in real data in several minutes of compute time on a laptop. The installation instructions are described in the [install](https://elephant.readthedocs.io/en/latest/install.html) section.
doc/install.rst:CUDA and OpenCL support
doc/install.rst::ref:`asset` module supports CUDA and OpenCL. These are experimental features.
doc/install.rst:    .. tab:: CUDA
doc/install.rst:        To leverage CUDA acceleration on an NVIDIA GPU card, `CUDA toolkit
doc/install.rst:        <https://developer.nvidia.com/cuda-downloads>`_ must installed on
doc/install.rst:            pip install pycuda
doc/install.rst:        In case you experience issues installing PyCUDA, `this guide
doc/install.rst:        <https://medium.com/leadkaro/setting-up-pycuda-on-ubuntu-18-04-for-
doc/install.rst:        gpu-programming-with-python-830e03fc4b81>`_ offers a step-by-step
doc/install.rst:        If PyCUDA is detected and installed, CUDA backend is used by default in
doc/install.rst:        Elephant ASSET module. To turn off CUDA support, set ``ELEPHANT_USE_CUDA``
doc/install.rst:    .. tab:: OpenCL
doc/install.rst:        leverage significant performance optimization with OpenCL backend.
doc/install.rst:        The simplest way to install PyOpenCL is to run a conda command:
doc/install.rst:            conda install -c conda-forge pyopencl intel-compute-runtime
doc/install.rst:        install PyOpenCL as follows:
doc/install.rst:            conda install -c conda-forge pyopencl ocl-icd-system
doc/install.rst:        Set ``ELEPHANT_USE_OPENCL`` environment flag to ``0`` to turn off
doc/install.rst:        PyOpenCL support.
doc/install.rst:            Make sure you've disabled GPU Hangcheck as described in the
doc/install.rst:            `Intel GPU developers documentation <https://www.intel.com/content/
doc/install.rst:            gpu-disable-hangcheck.html>`_. Do it with caution -
elephant/test/test_asset.py:    import pyopencl
elephant/test/test_asset.py:    HAVE_PYOPENCL = asset.get_opencl_capability()
elephant/test/test_asset.py:    HAVE_PYOPENCL = False
elephant/test/test_asset.py:    import pycuda
elephant/test/test_asset.py:    HAVE_CUDA = asset.get_cuda_capability_major() > 0
elephant/test/test_asset.py:    HAVE_CUDA = False
elephant/test/test_asset.py:        os.environ['ELEPHANT_USE_OPENCL'] = '0'
elephant/test/test_asset.py:    def test_pmat_neighbors_gpu(self):
elephant/test/test_asset.py:                if HAVE_PYOPENCL:
elephant/test/test_asset.py:                    lmat_opencl = pmat_neigh.pyopencl(pmat)
elephant/test/test_asset.py:                    assert_array_almost_equal(lmat_opencl, lmat_true)
elephant/test/test_asset.py:                if HAVE_CUDA:
elephant/test/test_asset.py:                    lmat_cuda = pmat_neigh.pycuda(pmat)
elephant/test/test_asset.py:                    assert_array_almost_equal(lmat_cuda, lmat_true)
elephant/test/test_asset.py:    def test_pmat_neighbors_gpu_chunked(self):
elephant/test/test_asset.py:                if HAVE_PYOPENCL:
elephant/test/test_asset.py:                    lmat_opencl = pmat_neigh.pyopencl(pmat)
elephant/test/test_asset.py:                    assert_array_almost_equal(lmat_opencl, lmat_true)
elephant/test/test_asset.py:                if HAVE_CUDA:
elephant/test/test_asset.py:                    lmat_cuda = pmat_neigh.pycuda(pmat)
elephant/test/test_asset.py:                    assert_array_almost_equal(lmat_cuda, lmat_true)
elephant/test/test_asset.py:    def test_pmat_neighbors_gpu_overlapped_chunks(self):
elephant/test/test_asset.py:        if HAVE_PYOPENCL:
elephant/test/test_asset.py:            lmat_opencl = pmat_neigh.pyopencl(pmat)
elephant/test/test_asset.py:            assert_array_almost_equal(lmat_opencl, lmat_true)
elephant/test/test_asset.py:        if HAVE_CUDA:
elephant/test/test_asset.py:            lmat_cuda = pmat_neigh.pycuda(pmat)
elephant/test/test_asset.py:            assert_array_almost_equal(lmat_cuda, lmat_true)
elephant/test/test_asset.py:        if HAVE_PYOPENCL:
elephant/test/test_asset.py:            self.assertRaises(ValueError, pmat_neigh.pyopencl, pmat)
elephant/test/test_asset.py:        if HAVE_CUDA:
elephant/test/test_asset.py:            self.assertRaises(ValueError, pmat_neigh.pycuda, pmat)
elephant/test/test_asset.py:    def test_asset_choose_backend_opencl(self):
elephant/test/test_asset.py:        class TestClassBackend(asset._GPUBackend):
elephant/test/test_asset.py:            def pycuda(self):
elephant/test/test_asset.py:                return "cuda"
elephant/test/test_asset.py:            def pyopencl(self):
elephant/test/test_asset.py:                return "opencl"
elephant/test/test_asset.py:        # check which backend is chosen if environment variable for opencl
elephant/test/test_asset.py:        os.environ.pop('ELEPHANT_USE_OPENCL', None)
elephant/test/test_asset.py:        if HAVE_PYOPENCL:
elephant/test/test_asset.py:            self.assertEqual(backend_obj.backend(), 'opencl')
elephant/test/test_asset.py:            # if environment variable is not set and no module pyopencl or
elephant/test/test_asset.py:        os.environ['ELEPHANT_USE_OPENCL'] = '0'
elephant/test/test_asset.py:        # This test shows the main idea of CUDA ASSET parallelization that
elephant/test/test_asset.py:            # function in asset.pycuda.py.
elephant/test/test_asset.py:    def test_gpu(self):
elephant/test/test_asset.py:                    if HAVE_PYOPENCL:
elephant/test/test_asset.py:                        P_total_opencl = jsf.pyopencl(log_du)
elephant/test/test_asset.py:                        assert_array_almost_equal(P_total_opencl, P_total_cpu)
elephant/test/test_asset.py:                    if HAVE_CUDA:
elephant/test/test_asset.py:                        P_total_cuda = jsf.pycuda(log_du)
elephant/test/test_asset.py:                        assert_array_almost_equal(P_total_cuda, P_total_cpu)
elephant/test/test_asset.py:    def test_gpu_threads_and_cwr_loops(self):
elephant/test/test_asset.py:                    jsf.cuda_threads = threads
elephant/test/test_asset.py:                    jsf.cuda_cwr_loops = cwr_loops
elephant/test/test_asset.py:        if HAVE_PYOPENCL:
elephant/test/test_asset.py:            run_test(jsf, jsf.pyopencl)
elephant/test/test_asset.py:        if HAVE_CUDA:
elephant/test/test_asset.py:            run_test(jsf, jsf.pycuda)
elephant/test/test_asset.py:    def test_gpu_chunked(self):
elephant/test/test_asset.py:            if HAVE_PYOPENCL:
elephant/test/test_asset.py:                P_total = jsf.pyopencl(log_du)
elephant/test/test_asset.py:            if HAVE_CUDA:
elephant/test/test_asset.py:                P_total = jsf.pycuda(log_du)
elephant/utils.py:def get_cuda_capability_major():
elephant/utils.py:    Extracts CUDA capability major version of the first available Nvidia GPU
elephant/utils.py:        CUDA capability major version.
elephant/utils.py:    cuda_success = 0
elephant/utils.py:    for libname in ("libcuda.so", "libcuda.dylib", "cuda.dll"):
elephant/utils.py:            cuda = ctypes.CDLL(libname)
elephant/utils.py:    result = cuda.cuInit(0)
elephant/utils.py:    if result != cuda_success:
elephant/utils.py:    # parse the first GPU card only
elephant/utils.py:    result = cuda.cuDeviceGet(ctypes.byref(device), 0)
elephant/utils.py:    if result != cuda_success:
elephant/utils.py:    cuda.cuDeviceComputeCapability(
elephant/utils.py:def get_opencl_capability():
elephant/utils.py:    Return a list of available OpenCL devices.
elephant/utils.py:        True: if openCL platform detected and at least one device is found,
elephant/utils.py:        False: if OpenCL is not found or if no OpenCL devices are found
elephant/utils.py:        import pyopencl
elephant/utils.py:        platforms = pyopencl.get_platforms()
elephant/asset/joint_pmat.cl:  #pragma OPENCL EXTENSION cl_khr_fp64: enable
elephant/asset/joint_pmat.cl:  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
elephant/asset/joint_pmat.cl: * OpenCL spec. defines unsigned long as uint64.
elephant/asset/joint_pmat.cl: * CUDA kernel that computes P_total - the joint survival probabilities matrix.
elephant/asset/joint_pmat_old.cu: * CUDA implementation of ASSET.joint_probability_matrix function (refer to
elephant/asset/joint_pmat_old.cu:#include <cuda.h>
elephant/asset/joint_pmat_old.cu:#include <cuda_runtime.h>
elephant/asset/joint_pmat_old.cu: * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
elephant/asset/joint_pmat_old.cu:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
elephant/asset/joint_pmat_old.cu:#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
elephant/asset/joint_pmat_old.cu:inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
elephant/asset/joint_pmat_old.cu:   if (code != cudaSuccess)
elephant/asset/joint_pmat_old.cu:      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
elephant/asset/joint_pmat_old.cu: * CUDA kernel that computes P_total - the joint survival probabilities matrix.
elephant/asset/joint_pmat_old.cu:    // values greater than ULONG_MAX are not supported by CUDA
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cudaMemcpyToSymbol(iteration_table, m, sizeof(ULL) * D * N) );
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cudaMemcpyToSymbol((const void*) &ITERATIONS_TODO, (const void*) &it_todo, sizeof(ULL)) );
elephant/asset/joint_pmat_old.cu:    cudaMemcpyFromSymbol(iteration_table_host, iteration_table, sizeof(ULL) * D * N);
elephant/asset/joint_pmat_old.cu:    cudaMemcpyFromSymbol((void*)&it_todo_host, (const void*)&ITERATIONS_TODO, sizeof(ULL));
elephant/asset/joint_pmat_old.cu:    cudaMemcpyFromSymbol((void*)&l_block, (const void*)&L_BLOCK, sizeof(ULL));
elephant/asset/joint_pmat_old.cu:    cudaMemcpyFromSymbol((void*)&l_num_blocks, (const void*)&L_NUM_BLOCKS, sizeof(ULL));
elephant/asset/joint_pmat_old.cu:    cudaMemcpyFromSymbol((void*)&logK_host, (const void*)&logK, sizeof(asset_float));
elephant/asset/joint_pmat_old.cu:    cudaMemcpyFromSymbol(log_factorial_host, log_factorial, sizeof(asset_float) * (N+1));
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cudaMalloc((void**)&log_du_device, sizeof(float) * L * (D + 1)) );
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cudaMemcpyAsync(log_du_device, log_du_host, sizeof(float) * L * (D + 1), cudaMemcpyHostToDevice) );
elephant/asset/joint_pmat_old.cu:    // Use P_total buffer to read log_du and copy batches to a GPU card
elephant/asset/joint_pmat_old.cu:        gpuErrchk( cudaMemcpy(log_du_device + col * L, log_du_host, sizeof(float) * L, cudaMemcpyHostToDevice) );
elephant/asset/joint_pmat_old.cu:    // with cudaMemset when the data type is float or double.
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cudaMalloc((void**)&P_total_device, sizeof(asset_float) * L) );
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cudaMemsetAsync(P_total_device, 0, sizeof(asset_float) * L) );
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cudaMemcpyToSymbol((const void*) &logK, (const void*) &logK_host, sizeof(asset_float)) );
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cudaMemcpyToSymbol(log_factorial, log_factorial_host, sizeof(asset_float) * (N + 1)) );
elephant/asset/joint_pmat_old.cu:    cudaDeviceProp device_prop;
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cudaGetDeviceProperties(&device_prop, 0) );
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cudaMemcpyToSymbol((const void*) &L_BLOCK, (const void*) &l_block, sizeof(ULL)) );
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cudaMemcpyToSymbol((const void*) &L_NUM_BLOCKS, (const void*) &l_num_blocks, sizeof(ULL)) );
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cudaDeviceSynchronize() );
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cudaPeekAtLastError() );
elephant/asset/joint_pmat_old.cu:    cudaError_t cuda_completed_status = cudaMemcpy(P_total_host, P_total_device, sizeof(asset_float) * L, cudaMemcpyDeviceToHost);
elephant/asset/joint_pmat_old.cu:    cudaFree(P_total_device);
elephant/asset/joint_pmat_old.cu:    cudaFree(log_du_device);
elephant/asset/joint_pmat_old.cu:    gpuErrchk( cuda_completed_status );
elephant/asset/joint_pmat.cu: * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
elephant/asset/joint_pmat.cu:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
elephant/asset/joint_pmat.cu: * CUDA kernel that computes P_total - the joint survival probabilities matrix.
elephant/asset/asset.py:from elephant.utils import get_cuda_capability_major, get_opencl_capability
elephant/asset/asset.py:class _GPUBackend:
elephant/asset/asset.py:        fit into GPU memory. Setting this parameter manually can resolve GPU
elephant/asset/asset.py:    1. PyOpenCL backend takes some time to compile the kernel for the first
elephant/asset/asset.py:       Host (CPU) data allocations are pageable by default. The GPU cannot
elephant/asset/asset.py:       from pageable host memory to device memory is invoked, the CUDA driver
elephant/asset/asset.py:       https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
elephant/asset/asset.py:       Same for OpenCL. Therefore, Python memory analyzers show increments in
elephant/asset/asset.py:       the used RAM each time an OpenCL/CUDA buffer is created. As with any
elephant/asset/asset.py:       Python objects, PyOpenCL and PyCUDA clean up and free allocated memory
elephant/asset/asset.py:        # If CUDA is detected, always use CUDA.
elephant/asset/asset.py:        # If OpenCL is detected, don't use it by default to avoid the system
elephant/asset/asset.py:        use_cuda = int(os.getenv("ELEPHANT_USE_CUDA", '1'))
elephant/asset/asset.py:        use_opencl = int(os.getenv("ELEPHANT_USE_OPENCL", '1'))
elephant/asset/asset.py:        cuda_detected = get_cuda_capability_major() != 0
elephant/asset/asset.py:        opencl_detected = get_opencl_capability()
elephant/asset/asset.py:        if use_cuda and cuda_detected:
elephant/asset/asset.py:            return self.pycuda
elephant/asset/asset.py:        if use_opencl and opencl_detected:
elephant/asset/asset.py:            return self.pyopencl
elephant/asset/asset.py:            raise ValueError(f"[GPU not enough memory] Impossible to split "
elephant/asset/asset.py:                             f"{min_chunk_size} to fit into GPU memory")
elephant/asset/asset.py:class _JSFUniformOrderStat3D(_GPUBackend):
elephant/asset/asset.py:                 cuda_threads=64, cuda_cwr_loops=32, tolerance=1e-5,
elephant/asset/asset.py:        self.cuda_threads = cuda_threads
elephant/asset/asset.py:        self.cuda_cwr_loops = cuda_cwr_loops
elephant/asset/asset.py:            CWR_LOOPS=self.cuda_cwr_loops,
elephant/asset/asset.py:    def pyopencl(self, log_du, device_id=0):
elephant/asset/asset.py:        import pyopencl as cl
elephant/asset/asset.py:        import pyopencl.array as cl_array
elephant/asset/asset.py:        logger.info(f"Available OpenCL devices:\n {context.devices}")
elephant/asset/asset.py:        n_threads = min(self.cuda_threads, max_l_block,
elephant/asset/asset.py:        # GPU_MAX_HEAP_SIZE OpenCL flag is set to 2 Gb (1 << 31) by default
elephant/asset/asset.py:        P_total_gpu = cl_array.Array(queue, shape=chunk_size, dtype=self.dtype)
elephant/asset/asset.py:            log_du_gpu = cl_array.to_device(queue, log_du[i_start: i_end],
elephant/asset/asset.py:            P_total_gpu.fill(0, queue=queue)
elephant/asset/asset.py:            grid_size = math.ceil(it_todo / (n_threads * self.cuda_cwr_loops))
elephant/asset/asset.py:            # OpenCL defines unsigned long as uint64, therefore we're adding
elephant/asset/asset.py:                   P_total_gpu.data, log_du_gpu.data, g_times_l=True)
elephant/asset/asset.py:            P_total_gpu[:chunk_size].get(ary=P_total[i_start: i_end])
elephant/asset/asset.py:    def pycuda(self, log_du):
elephant/asset/asset.py:            # PyCuda should not be in requirements-extra because CPU limited
elephant/asset/asset.py:            import pycuda.autoinit
elephant/asset/asset.py:            import pycuda.gpuarray as gpuarray
elephant/asset/asset.py:            import pycuda.driver as drv
elephant/asset/asset.py:            from pycuda.compiler import SourceModule
elephant/asset/asset.py:                "Install pycuda with 'pip install pycuda'") from err
elephant/asset/asset.py:        device = pycuda.autoinit.device
elephant/asset/asset.py:        n_threads = min(self.cuda_threads, max_l_block,
elephant/asset/asset.py:        P_total_gpu = gpuarray.GPUArray(chunk_size, dtype=self.dtype)
elephant/asset/asset.py:        log_du_gpu = drv.mem_alloc(4 * chunk_size * log_du.shape[1])
elephant/asset/asset.py:            drv.memcpy_htod_async(dest=log_du_gpu, src=log_du[i_start: i_end])
elephant/asset/asset.py:            P_total_gpu.fill(0)
elephant/asset/asset.py:            grid_size = math.ceil(it_todo / (n_threads * self.cuda_cwr_loops))
elephant/asset/asset.py:            iteration_table_gpu, _ = module.get_global("iteration_table")
elephant/asset/asset.py:            drv.memcpy_htod(iteration_table_gpu, iteration_table)
elephant/asset/asset.py:            log_factorial_gpu, _ = module.get_global("log_factorial")
elephant/asset/asset.py:            drv.memcpy_htod(log_factorial_gpu, log_factorial)
elephant/asset/asset.py:            kernel(P_total_gpu.gpudata, log_du_gpu, grid=(grid_size, 1),
elephant/asset/asset.py:            P_total_gpu[:chunk_size].get(ary=P_total[i_start: i_end])
elephant/asset/asset.py:    def _cuda(self, log_du):
elephant/asset/asset.py:        # in a terminal. Having this function is useful to debug ASSET CUDA
elephant/asset/asset.py:        # pycuda backend proves to be stable.
elephant/asset/asset.py:            N_THREADS=self.cuda_threads,
elephant/asset/asset.py:            # by default, GPU device code is optimized with -O3.
elephant/asset/asset.py:            if self.precision == 'double' and get_cuda_capability_major() >= 6:
elephant/asset/asset.py:                          "using PyOpenCL backend, make sure you've disabled "
elephant/asset/asset.py:                          "GPU Hangcheck as described here https://www.intel."
elephant/asset/asset.py:                          "guide-linux/2023-1/gpu-disable-hangcheck.html \n"
elephant/asset/asset.py:class _PMatNeighbors(_GPUBackend):
elephant/asset/asset.py:    def pyopencl(self, mat):
elephant/asset/asset.py:        import pyopencl as cl
elephant/asset/asset.py:        import pyopencl.array as cl_array
elephant/asset/asset.py:        # GPU_MAX_HEAP_SIZE OpenCL flag is set to 2 Gb (1 << 31) by default
elephant/asset/asset.py:        lmat_gpu = cl_array.Array(
elephant/asset/asset.py:                                   desc="Largest neighbors OpenCL"):
elephant/asset/asset.py:            mat_gpu = cl_array.to_device(queue,
elephant/asset/asset.py:            lmat_gpu.fill(0, queue=queue)
elephant/asset/asset.py:            # execute and the local size is set to None, PyOpenCL chooses the
elephant/asset/asset.py:            kernel(queue, (it_todo,), None, lmat_gpu.data, mat_gpu.data)
elephant/asset/asset.py:            lmat_gpu[:chunk_size].get(ary=lmat[i_start: i_end])
elephant/asset/asset.py:    def pycuda(self, mat):
elephant/asset/asset.py:            # PyCuda should not be in requirements-extra because CPU limited
elephant/asset/asset.py:            import pycuda.autoinit
elephant/asset/asset.py:            import pycuda.gpuarray as gpuarray
elephant/asset/asset.py:            import pycuda.driver as drv
elephant/asset/asset.py:            from pycuda.compiler import SourceModule
elephant/asset/asset.py:                "Install pycuda with 'pip install pycuda'") from err
elephant/asset/asset.py:        device = pycuda.autoinit.device
elephant/asset/asset.py:        lmat_gpu = gpuarray.GPUArray(
elephant/asset/asset.py:        mat_gpu = drv.mem_alloc(4 * (chunk_size + filt_size) * mat.shape[1])
elephant/asset/asset.py:                                   desc="Largest neighbors CUDA"):
elephant/asset/asset.py:            drv.memcpy_htod_async(dest=mat_gpu,
elephant/asset/asset.py:            lmat_gpu.fill(0)
elephant/asset/asset.py:            filt_rows_gpu, _ = module.get_global("filt_rows")
elephant/asset/asset.py:            drv.memcpy_htod(filt_rows_gpu, filt_rows.astype(np.uint32))
elephant/asset/asset.py:            filt_cols_gpu, _ = module.get_global("filt_cols")
elephant/asset/asset.py:            drv.memcpy_htod(filt_cols_gpu, filt_cols.astype(np.uint32))
elephant/asset/asset.py:                raise ValueError("Cannot launch a CUDA kernel with "
elephant/asset/asset.py:            kernel(lmat_gpu.gpudata, mat_gpu, grid=(grid_size, 1),
elephant/asset/asset.py:            lmat_gpu[:chunk_size].get(ary=lmat[i_start: i_end])
elephant/asset/asset.py:                                 cuda_threads=64, cuda_cwr_loops=32,
elephant/asset/asset.py:        cuda_threads : int, optional
elephant/asset/asset.py:            [CUDA/OpenCL performance parameter that does not influence the
elephant/asset/asset.py:            The number of CUDA/OpenCL threads per block (in X axis) between 1
elephant/asset/asset.py:            and 1024 and is used only if CUDA or OpenCL backend is enabled.
elephant/asset/asset.py:            Old GPUs (Tesla K80) perform faster with `cuda_threads` larger
elephant/asset/asset.py:        cuda_cwr_loops : int, optional
elephant/asset/asset.py:            [CUDA/OpenCL performance parameter that does not influence the
elephant/asset/asset.py:        1. By default, if CUDA is detected, CUDA acceleration is used. CUDA
elephant/asset/asset.py:           To turn off CUDA features, set the environment flag
elephant/asset/asset.py:           ``ELEPHANT_USE_CUDA`` to ``0``.
elephant/asset/asset.py:        2. If PyOpenCL is installed and detected, PyOpenCL backend is used.
elephant/asset/asset.py:           PyOpenCL backend is **~X100** faster than the Python implementation.
elephant/asset/asset.py:           To turn off OpenCL features, set the environment flag
elephant/asset/asset.py:           ``ELEPHANT_USE_OPENCL`` to ``0``.
elephant/asset/asset.py:           When using PyOpenCL backend, make sure you've disabled GPU Hangcheck
elephant/asset/asset.py:           as described in the `Intel GPU developers documentation
elephant/asset/asset.py:           guide-linux/2023-1/gpu-disable-hangcheck.html>`_. Do it with
elephant/asset/asset.py:                                     cuda_threads=cuda_threads,
elephant/asset/asset.py:                                     cuda_cwr_loops=cuda_cwr_loops,
requirements/requirements-extras.txt:jinja2>=2.11.2  # required for ASSET CUDA
requirements/requirements-cuda.txt:pycuda>=2020.1  # used in ASSET
requirements/requirements-opencl.txt:# conda install -c conda-forge pyopencl intel-compute-runtime
requirements/requirements-opencl.txt:pyopencl>=2020.2.2

```
