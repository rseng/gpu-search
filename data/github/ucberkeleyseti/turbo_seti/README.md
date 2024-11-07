# https://github.com/UCBerkeleySETI/turbo_seti

```console
docs/find_doppler.rst:   This kernel implements a GPU accelerated version of the :func:`~turbo_seti.find_doppler.find_doppler.hitsearch`
docs/find_doppler.rst:   method written as a RAW CUDA kernel.
docs/find_doppler.rst:      1. This GPU implementation is based on `Cupy <https://cupy.dev/>`_ array library accelerated with CUDA and ROCm.
docs/find_doppler.rst:      .. automodule:: turbo_seti.find_doppler.kernels._taylor_tree._core_cuda
VERSION-HISTORY.md:| 2022-04-04 | 2.2.2 | Performance improvement in GPU mode: Use a cupy RawKernel for the 'flt' function. |
VERSION-HISTORY.md:| 2021-08-12 | 2.1.11 | Specific MeerKAT files cause erratic behaviour in GPU mode (issue #270). |
VERSION-HISTORY.md:| 2021-07-22 | 2.1.9 | Performance improvement in gpu mode: default to single-precision (32-bit). |
VERSION-HISTORY.md:| 2021-07-15 | 2.1.6 | Calculate normalized value inside hitsearch kernel on GPU-mode. |
VERSION-HISTORY.md:| 2021-07-16 | 2.1.5 | Failed to pass the gpu_id from find_doppler.py to data_handler.py (issue #254). |
VERSION-HISTORY.md:| 2021-07-15 | 2.1.4 | Add GPU device selection with cli argument gpu_id. (issue #254). |
VERSION-HISTORY.md:| 2021-04-13 | 2.0.18 | Add GPU enabled Docker image build.
VERSION-HISTORY.md:| 2021-03-10 | 2.0.14 | Fixed issue #213 - Doppler search dies when using GPU (string format issue). |
VERSION-HISTORY.md:| | | GPU-mode performance improvements.
VERSION-HISTORY.md:| 2020-11-17 | 2.0.0 | Support NUMBA JIT compilation (CPU) and CUPY (NVIDIA GPU). |
test/run_benchmark.sh:FILE=blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.fil
test/run_benchmark.sh:FILE=blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.fil
test/run_benchmark.sh:FILE=blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000
test/run_benchmark.sh:FILE=blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000
test/run_benchmark.sh:echo "====> [BENCHMARK] GPU DOUBLE PRECISION"
test/run_benchmark.sh:turboSETI blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.h5  -g y -S n -P n
test/run_benchmark.sh:turboSETI blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.h5  -g y -S n -P n
test/run_benchmark.sh:echo "====> [BENCHMARK] GPU SINGLE PRECISION"
test/run_benchmark.sh:turboSETI blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.h5  -g y -S y -P n
test/run_benchmark.sh:turboSETI blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.h5  -g y -S y -P n
test/run_benchmark.sh:turboSETI blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.h5  -g n -S n -P n
test/run_benchmark.sh:turboSETI blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.h5  -g n -S n -P n
test/run_benchmark.sh:turboSETI blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.h5  -g n -S y -P n
test/run_benchmark.sh:turboSETI blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.h5  -g n -S y -P n
test/fb_cases_util.py:def using_gpu():
test/fb_cases_util.py:    Using GPU acceleration?
test/fb_cases_util.py:    using_gpu : bool
test/fb_cases_util.py:        True : use GPU
test/fb_cases_util.py:                       gpu_backend = using_gpu(),
test/test_turbo_seti.py:    (Kernels(gpu_backend=False, precision=2)),
test/test_turbo_seti.py:    (Kernels(gpu_backend=False, precision=1)),
test/test_turbo_seti.py:if Kernels.has_gpu():
test/test_turbo_seti.py:    GPU_TESTS = [
test/test_turbo_seti.py:        (Kernels(gpu_backend=True, precision=2)),
test/test_turbo_seti.py:        (Kernels(gpu_backend=True, precision=1)),
test/test_turbo_seti.py:    TESTS.extend(GPU_TESTS)
test/test_turbo_seti.py:    if Kernels.has_gpu():
test/test_turbo_seti.py:### Do not run dask partitions with GPU due to GPU RAM availability issues.
README.md:- cupy (NVIDIA GPU mode only)
README.md:## NVIDIA GPU Users
README.md:Already included is NUMBA Just-in-Time (JIT) CPU performance enhancements. However, if you have NVIDIA GPU hardware on the computer where turbo_seti is going to execute, you can get significant additional performance improvement.  Enable GPU enhanced processing with these steps:
README.md:find_doppler.0  INFO     Parameters: datafile=/seti_data/voyager/Voyager1.single_coarse.fine_res.h5, max_drift=4, min_drift=0.0, snr=25, out_dir=/seti_data/voyager/, coarse_chans=None, flagging=False, n_coarse_chan=None, kernels=None, gpu_backend=False, precision=2, append_output=False, log_level_int=20, obs_info={'pulsar': 0, 'pulsar_found': 0, 'pulsar_dm': 0.0, 'pulsar_snr': 0.0, 'pulsar_stats': array([0., 0., 0., 0., 0., 0.]), 'RFI_level': 0.0, 'Mean_SEFD': 0.0, 'psrflux_Sens': 0.0, 'SEFDs_val': [0.0], 'SEFDs_freq': [0.0], 'SEFDs_freq_up': [0.0]}
turbo_seti/find_doppler/kernels/_taylor_tree/_core_cuda.py:# Cuda kernels for the flt function to use.
turbo_seti/find_doppler/kernels/_taylor_tree/_core_cuda.py:#   https://github.com/UCBerkeleySETI/dedopplerperf/blob/main/CudaTaylor5demo.cu
turbo_seti/find_doppler/kernels/_taylor_tree/_core_cuda.py:            f"we have no GPU taylor kernel for the numerical type: {array.dtype}"
turbo_seti/find_doppler/kernels/_taylor_tree/_core_cuda.py:    # Cuda params
turbo_seti/find_doppler/kernels/_hitsearch/__init__.py:    Performs hitsearch on the GPU with CUDA. Automatically chooses
turbo_seti/find_doppler/kernels/_hitsearch/__init__.py:        CUDA Kernel number of blocks.
turbo_seti/find_doppler/kernels/_hitsearch/__init__.py:        CUDA Kernel block size.
turbo_seti/find_doppler/kernels/__init__.py:    gpu_backend : bool, optional
turbo_seti/find_doppler/kernels/__init__.py:        Enable GPU acceleration.
turbo_seti/find_doppler/kernels/__init__.py:    def __init__(self, gpu_backend=False, precision=2, gpu_id=0):
turbo_seti/find_doppler/kernels/__init__.py:        self.gpu_backend = gpu_backend
turbo_seti/find_doppler/kernels/__init__.py:        self.gpu_id = gpu_id
turbo_seti/find_doppler/kernels/__init__.py:        if not self.has_gpu() and self.gpu_backend:
turbo_seti/find_doppler/kernels/__init__.py:            raise RuntimeError("cupy is not installed, so the GPU cannot be used.")
turbo_seti/find_doppler/kernels/__init__.py:        if self.gpu_backend:
turbo_seti/find_doppler/kernels/__init__.py:            self.xp.cuda.Device(self.gpu_id).use()
turbo_seti/find_doppler/kernels/__init__.py:        if self.gpu_backend:
turbo_seti/find_doppler/kernels/__init__.py:                self._base_lib + "._taylor_tree._core_cuda"
turbo_seti/find_doppler/kernels/__init__.py:        if self.gpu_backend:
turbo_seti/find_doppler/kernels/__init__.py:        In the GPU version, the row index is the same as the "drift index". 0 is the least drift,
turbo_seti/find_doppler/kernels/__init__.py:        if self.gpu_backend:
turbo_seti/find_doppler/kernels/__init__.py:    def has_gpu():
turbo_seti/find_doppler/kernels/__init__.py:        Check if the system has the modules needed for the GPU acceleration.
turbo_seti/find_doppler/kernels/__init__.py:        Modules are listed on `requirements_gpu.txt`.
turbo_seti/find_doppler/kernels/__init__.py:        has_gpu : bool
turbo_seti/find_doppler/kernels/__init__.py:            True if the system has GPU capabilities.
turbo_seti/find_doppler/data_handler.py:    gpu_backend : bool, optional
turbo_seti/find_doppler/data_handler.py:        Use GPU accelerated Kernels?
turbo_seti/find_doppler/data_handler.py:    gpu_id : int
turbo_seti/find_doppler/data_handler.py:        If  gpu_backend=True, then this is the device ID to use.
turbo_seti/find_doppler/data_handler.py:                 kernels=None, gpu_backend=False, precision=1, gpu_id=0):
turbo_seti/find_doppler/data_handler.py:            self.kernels = Kernels(gpu_backend, precision, gpu_id)
turbo_seti/find_doppler/data_handler.py:            datah5_obj = DATAH5(filename, kernels=self.kernels, gpu_id=gpu_id)
turbo_seti/find_doppler/data_handler.py:                 cchan_id=0, n_coarse_chan=None, kernels=None, gpu_backend=False, precision=1, gpu_id=0):
turbo_seti/find_doppler/data_handler.py:            self.kernels = Kernels(gpu_backend, precision, gpu_id)
turbo_seti/find_doppler/seti_event.py:    p.add_argument('-g', '--gpu', dest='flag_gpu', type=str, default='n',
turbo_seti/find_doppler/seti_event.py:                   help='Compute on the GPU? (y/n)')
turbo_seti/find_doppler/seti_event.py:    p.add_argument('-d', '--gpu_id', dest='gpu_id', type=int, default=0,
turbo_seti/find_doppler/seti_event.py:                   help='Use which GPU device? (0,1,...)')
turbo_seti/find_doppler/seti_event.py:    if Kernels.has_gpu() and args.flag_gpu == "n":
turbo_seti/find_doppler/seti_event.py:        print("Info: Your system is compatible with GPU-mode. Use the `-g y` argument to enable it.")
turbo_seti/find_doppler/seti_event.py:                                  gpu_backend=(args.flag_gpu == "y"),
turbo_seti/find_doppler/seti_event.py:                                  gpu_id=args.gpu_id,
turbo_seti/find_doppler/find_doppler.py:    gpu_backend : bool, optional
turbo_seti/find_doppler/find_doppler.py:        Use GPU accelerated Kernels? (True/False)
turbo_seti/find_doppler/find_doppler.py:    gpu_id : int
turbo_seti/find_doppler/find_doppler.py:        If gpu_backend=True, then this is the GPU device to use.
turbo_seti/find_doppler/find_doppler.py:        Floating point precision for the GPU.
turbo_seti/find_doppler/find_doppler.py:                 obs_info=None, flagging=False, n_coarse_chan=None, kernels=None, gpu_backend=False, gpu_id=0,
turbo_seti/find_doppler/find_doppler.py:            self.kernels = Kernels(gpu_backend, precision, gpu_id)
turbo_seti/find_doppler/find_doppler.py:                                      gpu_id=gpu_id,
turbo_seti/find_doppler/find_doppler.py:                    + ', flagging={}, n_coarse_chan={}, kernels={}, gpu_id={}, gpu_backend={}, blank_dc={}' \
turbo_seti/find_doppler/find_doppler.py:                        .format(flagging, self.n_coarse_chan, kernels, gpu_id, gpu_backend, blank_dc) \
turbo_seti/find_doppler/find_doppler.py:        It is not recommended to mix dask partitions with GPU mode as this could cause GPU queuing.
turbo_seti/find_doppler/find_doppler.py:        Floating point precision for the GPU.
turbo_seti/find_doppler/find_doppler.py:                  gpu_backend=False,
turbo_seti/find_doppler/find_doppler.py:    if fd.kernels.gpu_backend:
turbo_seti/find_doppler/find_doppler.py:    if fd.kernels.gpu_backend:
telegraphic/find_scan_sets.py:        df3 = df2[df2[file].str.contains("gpuspec.0000.h5",na=False)]
telegraphic/find_scan_sets.py:        df3 = df2[df2[file].str.contains("gpuspec.0000.fil",na=False)]

```
