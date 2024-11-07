# https://github.com/johnh2o2/cuvarbase

```console
setup.py:      description="Period-finding and variability on the GPU",
setup.py:                        'pycuda>=2017.1.1',
setup.py:                        'scikit-cuda'],
README.rst:``cuvarbase`` is a Python library that uses `PyCUDA <https://mathema.tician.de/software/pycuda/>`_ to implement several time series tools used in astronomy on GPUs.
README.rst:- `PyCUDA <https://mathema.tician.de/software/pycuda/>`_ **<-essential**
README.rst:- `scikit cuda <https://scikit-cuda.readthedocs.io/en/latest/>`_ **<-also essential**
README.rst:	- used for access to the CUDA FFT runtime library
README.rst:Using multiple GPUs
README.rst:If you have more than one GPU, you can choose which one to
README.rst:use in a given script by setting the ``CUDA_DEVICE`` environment
README.rst:    CUDA_DEVICE=1 python script.py
README.rst:jobs to different GPU's will have to suffice.
docs/source/plots/bls_example.py:bls_power, sols = bls.eebls_gpu(t, y, dy, freqs,
docs/source/plots/ce_example.py:# Start GPU process for conditional entropy
docs/source/plots/ce_example.py:# large to fit in GPU memory.
docs/source/plots/bls_example_transit.py:freqs, bls_power, sols = bls.eebls_transit_gpu(t, y, dy,
docs/source/plots/benchmarks.py:import pycuda.autoinit
docs/source/plots/benchmarks.py:import pycuda.driver as cuda
docs/source/plots/benchmarks.py:        cuda.start_profiler()
docs/source/plots/benchmarks.py:        cuda.stop_profiler()
docs/source/plots/benchmarks.py:        #pycuda.autoinit.context.detach()
docs/source/plots/benchmarks.py:eebls_gpu = function_timer(bls.eebls_gpu)
docs/source/plots/benchmarks.py:eebls_transit_gpu = function_timer(bls.eebls_transit_gpu)
docs/source/plots/benchmarks.py:eebls_gpu_fast = function_timer(bls.eebls_gpu_fast)
docs/source/plots/benchmarks.py:        return eebls_gpu_fast(t, y, dy, freqs, memory=memory,
docs/source/plots/benchmarks.py:        return eebls_gpu(t, y, dy, freqs, qmin=qmin, qmax=qmax,
docs/source/plots/benchmarks.py:    return eebls_transit_gpu(t, y, dy, freqs=freqs, qvals=qvals, **kw)
docs/source/plots/benchmarks.py:dev = pycuda.autoinit.device
docs/source/lomb.rst:It's worth mentioning the [Townsend2010]_ CUDA implementation of Lomb-Scargle, however this uses the :math:`\mathcal{O}(N_{\rm obs}N_f)` "naive" implementation
docs/source/lomb.rst:	import skcuda.fft
docs/source/lomb.rst:	# Synchronize all cuda streams
docs/source/lomb.rst:	import skcuda.fft
docs/source/lomb.rst:	# Synchronize all cuda streams
docs/source/conf.py:cuda_dir = "/Developer/NVIDIA/CUDA-8.0/lib/"
docs/source/conf.py:sys.path.insert(0, cuda_dir)
docs/source/conf.py:dyld_lpath = lpath_insert(cuda_dir, dyld_lpath)
docs/source/conf.py:ld_lpath = lpath_insert(cuda_dir, ld_lpath)
docs/source/ce.rst:instead of ``run``, which will ensure that the memory limit (1 GB in this case) is not exceeded on the GPU (unless of course you have other processes running). 
cuvarbase/lombscargle.py:import pycuda.driver as cuda
cuvarbase/lombscargle.py:import pycuda.gpuarray as gpuarray
cuvarbase/lombscargle.py:from pycuda.compiler import SourceModule
cuvarbase/lombscargle.py:# import pycuda.autoinit
cuvarbase/lombscargle.py:from .core import GPUAsyncProcess
cuvarbase/lombscargle.py:    data between the GPU and CPU for Lomb-Scargle computations
cuvarbase/lombscargle.py:    stream: :class:`pycuda.driver.Stream` instance
cuvarbase/lombscargle.py:        The CUDA stream used for calculations/data transfer
cuvarbase/lombscargle.py:        self.reg_g = gpuarray.zeros(2 * self.nharmonics + 1,
cuvarbase/lombscargle.py:        self.t_g = gpuarray.zeros(n0, dtype=self.real_type)
cuvarbase/lombscargle.py:        self.yw_g = gpuarray.zeros(n0, dtype=self.real_type)
cuvarbase/lombscargle.py:        self.w_g = gpuarray.zeros(n0, dtype=self.real_type)
cuvarbase/lombscargle.py:        and the GPU vector for the Lomb-Scargle power
cuvarbase/lombscargle.py:        self.lsp_g = gpuarray.zeros(self.nf, dtype=self.real_type)
cuvarbase/lombscargle.py:        self.lsp_c = cuda.aligned_zeros(shape=(nf,), dtype=self.real_type,
cuvarbase/lombscargle.py:        self.lsp_c = cuda.register_host_memory(self.lsp_c)
cuvarbase/lombscargle.py:        self.t = cuda.aligned_zeros(shape=(n0,),
cuvarbase/lombscargle.py:        self.t = cuda.register_host_memory(self.t)
cuvarbase/lombscargle.py:        self.yw = cuda.aligned_zeros(shape=(n0,),
cuvarbase/lombscargle.py:        self.yw = cuda.register_host_memory(self.yw)
cuvarbase/lombscargle.py:        self.w = cuda.aligned_zeros(shape=(n0,),
cuvarbase/lombscargle.py:        self.w = cuda.register_host_memory(self.w)
cuvarbase/lombscargle.py:    def transfer_data_to_gpu(self, **kwargs):
cuvarbase/lombscargle.py:        """ Transfers the lightcurve to the GPU """
cuvarbase/lombscargle.py:    def set_gpu_arrays_to_zero(self, **kwargs):
cuvarbase/lombscargle.py:        """ Sets all gpu arrays to zero """
cuvarbase/lombscargle.py:        Number of CUDA threads per block
cuvarbase/lombscargle.py:        If the data is already on the gpu, set as False
cuvarbase/lombscargle.py:    # lightcurve -> gpu
cuvarbase/lombscargle.py:        memory.transfer_data_to_gpu()
cuvarbase/lombscargle.py:    # Use direct sums (on GPU)
cuvarbase/lombscargle.py:class LombScargleAsyncProcess(GPUAsyncProcess):
cuvarbase/lombscargle.py:    GPUAsyncProcess for the Lomb Scargle periodogram
cuvarbase/lombscargle.py:        """ return an approximate GPU memory requirement in bytes """
cuvarbase/lombscargle.py:        Allocate GPU (and possibly CPU) memory for single lightcurve
cuvarbase/lombscargle.py:        stream: pycuda.driver.Stream
cuvarbase/lombscargle.py:            CUDA stream you want this to run on
cuvarbase/lombscargle.py:        Allocate GPU memory for Lomb Scargle computations
cuvarbase/lombscargle.py:                memory[i].set_gpu_arrays_to_zero(**kwargs)
cuvarbase/lombscargle.py:        # set up memory containers for gpu and cpu (pinned) memory
cuvarbase/lombscargle.py:    things work on the GPU. Note: This will be
cuvarbase/tests/test_pdm.py:from pycuda.tools import mark_cuda_test
cuvarbase/tests/test_pdm.py:def pow_gpu(request):
cuvarbase/tests/test_pdm.py:@pytest.mark.parametrize(["pow_cpu","pow_gpu"], [("binned_linterp","binned_linterp")], indirect=True)
cuvarbase/tests/test_pdm.py:def test_cuda_pdm_binned_linterp(pow_cpu,pow_gpu):
cuvarbase/tests/test_pdm.py:    assert_allclose(pow_cpu, pow_gpu, atol=1E-2, rtol=0)
cuvarbase/tests/test_pdm.py:@pytest.mark.parametrize(["pow_cpu","pow_gpu"], [("binned_step","binned_step")], indirect=True)
cuvarbase/tests/test_pdm.py:def test_cuda_pdm_binned_step(pow_cpu,pow_gpu):
cuvarbase/tests/test_pdm.py:    assert_allclose(pow_cpu, pow_gpu, atol=1E-2, rtol=0)
cuvarbase/tests/test_pdm.py:@pytest.mark.parametrize(["binless_pow_cpu","pow_gpu"], [("binless_gauss","binless_gauss")], indirect=True)
cuvarbase/tests/test_pdm.py:def test_cuda_pdm_binless_gauss(binless_pow_cpu,pow_gpu):
cuvarbase/tests/test_pdm.py:    assert_allclose(binless_pow_cpu, pow_gpu, atol=1E-2, rtol=0)
cuvarbase/tests/test_pdm.py:@pytest.mark.parametrize(["binless_pow_cpu","pow_gpu"], [("binless_tophat","binless_tophat")], indirect=True)
cuvarbase/tests/test_pdm.py:def test_cuda_pdm_binless_tophat(binless_pow_cpu,pow_gpu):
cuvarbase/tests/test_pdm.py:    assert_allclose(binless_pow_cpu, pow_gpu, atol=1E-2, rtol=0)
cuvarbase/tests/test_nfft.py:from pycuda.tools import mark_cuda_test
cuvarbase/tests/test_nfft.py:from pycuda import gpuarray
cuvarbase/tests/test_nfft.py:import skcuda.fft as cufft
cuvarbase/tests/test_nfft.py:def gpu_grid_scalar(t, y, sigma, m, N):
cuvarbase/tests/test_nfft.py:def simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, use_double=False,
cuvarbase/tests/test_nfft.py:#@mark_cuda_test
cuvarbase/tests/test_nfft.py:        gpu_grid = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
cuvarbase/tests/test_nfft.py:        assert_allclose(gpu_grid, cpu_grid, atol=1E-4, rtol=0)
cuvarbase/tests/test_nfft.py:        gpu_grid = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
cuvarbase/tests/test_nfft.py:        # get python version of gpu grid calculation
cuvarbase/tests/test_nfft.py:        cpu_grid = gpu_grid_scalar(tsc, y, nfft_sigma, nfft_m, nf)
cuvarbase/tests/test_nfft.py:        assert_allclose(gpu_grid, cpu_grid, **tols)
cuvarbase/tests/test_nfft.py:        gpu_grid = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
cuvarbase/tests/test_nfft.py:        # get python version of gpu grid calculation
cuvarbase/tests/test_nfft.py:        cpu_grid = gpu_grid_scalar(tsc, y, nfft_sigma, nfft_m, nf)
cuvarbase/tests/test_nfft.py:        assert_allclose(gpu_grid, cpu_grid, **tols)
cuvarbase/tests/test_nfft.py:        gpu_grid = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
cuvarbase/tests/test_nfft.py:        diffs = np.absolute(gpu_grid - cpu_grid)
cuvarbase/tests/test_nfft.py:        for i, gpug, cpug, d in zip(inds, gpu_grid[inds],
cuvarbase/tests/test_nfft.py:            print(i, gpug, cpug, d)
cuvarbase/tests/test_nfft.py:        assert_allclose(gpu_grid, cpu_grid, **tols)
cuvarbase/tests/test_nfft.py:        yg = gpuarray.to_gpu(y.astype(np.complex128))
cuvarbase/tests/test_nfft.py:        yghat = gpuarray.to_gpu(yhat.astype(np.complex128))
cuvarbase/tests/test_nfft.py:        gpu_nfft = simple_gpu_nfft(tg, y, nf, sigma=nfft_sigma, m=nfft_m,
cuvarbase/tests/test_nfft.py:        inds = dsort(np.real(direct_dft), np.real(gpu_nfft))
cuvarbase/tests/test_nfft.py:        q = list(zip(inds[:npr], direct_dft[inds[:npr]], gpu_nfft[inds[:npr]]))
cuvarbase/tests/test_nfft.py:        assert_allclose(np.real(direct_dft), np.real(gpu_nfft), **tols)
cuvarbase/tests/test_nfft.py:        assert_allclose(np.imag(direct_dft), np.imag(gpu_nfft), **tols)
cuvarbase/tests/test_nfft.py:            nfft = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
cuvarbase/tests/test_lombscargle.py:from pycuda.tools import mark_cuda_test
cuvarbase/tests/test_lombscargle.py:#import pycuda.autoinit
cuvarbase/tests/test_lombscargle.py:import pycuda.autoprimaryctx
cuvarbase/tests/test_lombscargle.py:        fgpu, pgpu = results[0]
cuvarbase/tests/test_lombscargle.py:        power = LombScargle(t, y, err).power(fgpu)
cuvarbase/tests/test_lombscargle.py:        assert_similar(power, pgpu)
cuvarbase/tests/test_lombscargle.py:        fgpu, pgpu = results[0]
cuvarbase/tests/test_lombscargle.py:        power = LombScargle(t, y, err).power(fgpu)
cuvarbase/tests/test_lombscargle.py:        assert_similar(power, pgpu)
cuvarbase/tests/test_lombscargle.py:        fgpu, pgpu = results[0]
cuvarbase/tests/test_lombscargle.py:        power = ls.power(fgpu)
cuvarbase/tests/test_lombscargle.py:        assert_similar(power, pgpu)
cuvarbase/tests/test_lombscargle.py:        fgpu, pgpu = results[0]
cuvarbase/tests/test_lombscargle.py:        power = ls.power(fgpu)
cuvarbase/tests/test_lombscargle.py:        assert_similar(power, pgpu)
cuvarbase/tests/test_lombscargle.py:        fgpu_ds, pgpu_ds = results_ds[0]
cuvarbase/tests/test_lombscargle.py:        fgpu_reg, pgpu_reg = results_reg[0]
cuvarbase/tests/test_lombscargle.py:        assert_similar(pgpu_reg, pgpu_ds)
cuvarbase/tests/test_lombscargle.py:        fgpu_ds, pgpu_ds = result_ds[0]
cuvarbase/tests/test_lombscargle.py:        fgpu_reg, pgpu_reg = result_reg[0]
cuvarbase/tests/test_lombscargle.py:        assert_similar(pgpu_reg, pgpu_ds)
cuvarbase/tests/test_bls.py:from pycuda.tools import mark_cuda_test
cuvarbase/tests/test_bls.py:from ..bls import eebls_gpu, eebls_transit_gpu, \
cuvarbase/tests/test_bls.py:                  single_bls, eebls_gpu_custom, eebls_gpu_fast
cuvarbase/tests/test_bls.py:        freqs, power, sols = eebls_transit_gpu(t, y, dy,
cuvarbase/tests/test_bls.py:        power, gsols = eebls_gpu_custom(t, y, dy, freqs,
cuvarbase/tests/test_bls.py:        power, gsols = eebls_gpu(t, y, dy, freqs,
cuvarbase/tests/test_bls.py:            freqs, power = eebls_transit_gpu(t, y, err, **kw)
cuvarbase/tests/test_bls.py:            freqs, power_slow, sols = eebls_transit_gpu(t, y, err, **kw)
cuvarbase/tests/test_bls.py:        freqs, power, sols = eebls_transit_gpu(t, y, err, **kw)
cuvarbase/tests/test_bls.py:        power = eebls_gpu_fast(t, y, err, freqs, **kw)
cuvarbase/tests/test_bls.py:        power0, sols = eebls_gpu(t, y, err, freqs, **kw)
cuvarbase/tests/test_bls.py:        # possible for eebls_gpu and eebls_gpu_fast
cuvarbase/tests/test_ce.py:from pycuda.tools import mark_cuda_test
cuvarbase/ce.py:import pycuda.driver as cuda
cuvarbase/ce.py:import pycuda.gpuarray as gpuarray
cuvarbase/ce.py:#import pycuda.autoinit
cuvarbase/ce.py:import pycuda.autoprimaryctx
cuvarbase/ce.py:from pycuda.compiler import SourceModule
cuvarbase/ce.py:from .core import GPUAsyncProcess
cuvarbase/ce.py:        self.t = cuda.aligned_zeros(shape=(n0,), **kw)
cuvarbase/ce.py:        self.t = cuda.register_host_memory(self.t)
cuvarbase/ce.py:        self.y = cuda.aligned_zeros(shape=(n0,),
cuvarbase/ce.py:        self.y = cuda.register_host_memory(self.y)
cuvarbase/ce.py:            self.dy = cuda.aligned_zeros(shape=(n0,), **kw)
cuvarbase/ce.py:            self.dy = cuda.register_host_memory(self.dy)
cuvarbase/ce.py:            self.mag_bwf = cuda.aligned_zeros(shape=(self.mag_bins,), **kw)
cuvarbase/ce.py:            self.mag_bwf = cuda.register_host_memory(self.mag_bwf)
cuvarbase/ce.py:            self.mag_bin_fracs = cuda.aligned_zeros(shape=(self.mag_bins,),
cuvarbase/ce.py:            self.mag_bin_fracs = cuda.register_host_memory(self.mag_bin_fracs)
cuvarbase/ce.py:        self.ce_c = cuda.aligned_zeros(shape=(nf,), dtype=self.real_type,
cuvarbase/ce.py:        self.ce_c = cuda.register_host_memory(self.ce_c)
cuvarbase/ce.py:        self.t_g = gpuarray.zeros(n0, dtype=self.real_type)
cuvarbase/ce.py:        self.y_g = gpuarray.zeros(n0, dtype=self.ytype)
cuvarbase/ce.py:            self.dy_g = gpuarray.zeros(n0, dtype=self.real_type)
cuvarbase/ce.py:            self.bins_g = gpuarray.zeros(self.nbins, dtype=self.real_type)
cuvarbase/ce.py:            self.bins_g = gpuarray.zeros(self.nbins, dtype=np.uint32)
cuvarbase/ce.py:            self.mag_bwf_g = gpuarray.zeros(self.mag_bins,
cuvarbase/ce.py:            self.mag_bin_fracs_g = gpuarray.zeros(self.mag_bins,
cuvarbase/ce.py:        self.freqs_g = gpuarray.zeros(nf, dtype=self.real_type)
cuvarbase/ce.py:            self.ce_g = gpuarray.zeros(nf, dtype=self.real_type)
cuvarbase/ce.py:    def transfer_data_to_gpu(self, **kwargs):
cuvarbase/ce.py:    def transfer_freqs_to_gpu(self, **kwargs):
cuvarbase/ce.py:    def set_gpu_arrays_to_zero(self, **kwargs):
cuvarbase/ce.py:        memory.transfer_data_to_gpu()
cuvarbase/ce.py:        dev = pycuda.autoprimaryctx.device
cuvarbase/ce.py:        att = cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
cuvarbase/ce.py:        shmem_lim = pycuda.autoprimaryctx.device.get_attribute(att)
cuvarbase/ce.py:        memory.transfer_data_to_gpu()
cuvarbase/ce.py:class ConditionalEntropyAsyncProcess(GPUAsyncProcess):
cuvarbase/ce.py:    GPUAsyncProcess for the Conditional Entropy period finder
cuvarbase/ce.py:        Number of CUDA threads per CUDA block.
cuvarbase/ce.py:        Return an approximate GPU memory requirement in bytes.
cuvarbase/ce.py:        Allocate GPU (and possibly CPU) memory for single lightcurve
cuvarbase/ce.py:        stream: pycuda.driver.Stream, optional
cuvarbase/ce.py:            CUDA stream you want this to run on
cuvarbase/ce.py:        Allocate GPU memory for Conditional Entropy computations
cuvarbase/ce.py:        streams: list of ``pycuda.driver.Stream``
cuvarbase/ce.py:            Transfers data to gpu if memory is provided
cuvarbase/ce.py:                mem.transfer_freqs_to_gpu()
cuvarbase/ce.py:                memory[i].set_gpu_arrays_to_zero(**kwargs)
cuvarbase/ce.py:            ``pycuda.driver.mem_get_info()``
cuvarbase/ce.py:            free, total = cuda.mem_get_info()
cuvarbase/ce.py:        # set up memory containers for gpu and cpu (pinned) memory
cuvarbase/ce.py:        [mem.transfer_freqs_to_gpu(**kwargs) for mem in memory]
cuvarbase/pdm.py:import pycuda.driver as cuda
cuvarbase/pdm.py:import pycuda.gpuarray as gpuarray
cuvarbase/pdm.py:from pycuda.compiler import SourceModule
cuvarbase/pdm.py:# import pycuda.autoinit
cuvarbase/pdm.py:from .core import GPUAsyncProcess
cuvarbase/pdm.py:def pdm_async(stream, data_cpu, data_gpu, pow_cpu, function,
cuvarbase/pdm.py:    t_g, y_g, w_g, freqs_g, pow_g = data_gpu
cuvarbase/pdm.py:class PDMAsyncProcess(GPUAsyncProcess):
cuvarbase/pdm.py:        gpu_data, pow_cpus = [], []
cuvarbase/pdm.py:            pow_cpu = cuda.aligned_zeros(shape=(len(freqs),),
cuvarbase/pdm.py:            pow_cpu = cuda.register_host_memory(pow_cpu)
cuvarbase/pdm.py:                t_g, y_g, w_g = tuple([gpuarray.zeros(len(t), dtype=np.float32)
cuvarbase/pdm.py:            pow_g = gpuarray.zeros(len(pow_cpu), dtype=pow_cpu.dtype)
cuvarbase/pdm.py:            freqs_g = gpuarray.to_gpu(np.asarray(freqs).astype(np.float32))
cuvarbase/pdm.py:            gpu_data.append((t_g, y_g, w_g, freqs_g, pow_g))
cuvarbase/pdm.py:        return gpu_data, pow_cpus
cuvarbase/pdm.py:    def run(self, data, gpu_data=None, pow_cpus=None,
cuvarbase/pdm.py:        if pow_cpus is None or gpu_data is None:
cuvarbase/pdm.py:            gpu_data, pow_cpus = self.allocate(data)
cuvarbase/pdm.py:                   zip(streams, data, gpu_data, pow_cpus)]
cuvarbase/bls.py:#import pycuda.autoinit
cuvarbase/bls.py:import pycuda.autoprimaryctx
cuvarbase/bls.py:import pycuda.driver as cuda
cuvarbase/bls.py:import pycuda.gpuarray as gpuarray
cuvarbase/bls.py:from pycuda.compiler import SourceModule
cuvarbase/bls.py:from .core import GPUAsyncProcess
cuvarbase/bls.py:        CUDA threads per CUDA block.
cuvarbase/bls.py:        Dictionary of (function name, PyCUDA function object) pairs
cuvarbase/bls.py:        self.bls = cuda.aligned_zeros(shape=(nfreqs,),
cuvarbase/bls.py:        self.nbins0 = cuda.aligned_zeros(shape=(nfreqs,),
cuvarbase/bls.py:        self.nbinsf = cuda.aligned_zeros(shape=(nfreqs,),
cuvarbase/bls.py:        self.t = cuda.aligned_zeros(shape=(ndata,),
cuvarbase/bls.py:        self.yw = cuda.aligned_zeros(shape=(ndata,),
cuvarbase/bls.py:        self.w = cuda.aligned_zeros(shape=(ndata,),
cuvarbase/bls.py:        self.freqs_g = gpuarray.zeros(nfreqs, dtype=self.rtype)
cuvarbase/bls.py:        self.bls_g = gpuarray.zeros(nfreqs, dtype=self.rtype)
cuvarbase/bls.py:        self.nbins0_g = gpuarray.zeros(nfreqs, dtype=np.uint32)
cuvarbase/bls.py:        self.nbinsf_g = gpuarray.zeros(nfreqs, dtype=np.uint32)
cuvarbase/bls.py:        self.t_g = gpuarray.zeros(ndata, dtype=self.rtype)
cuvarbase/bls.py:        self.yw_g = gpuarray.zeros(ndata, dtype=self.rtype)
cuvarbase/bls.py:        self.w_g = gpuarray.zeros(ndata, dtype=self.rtype)
cuvarbase/bls.py:    def transfer_data_to_gpu(self, transfer_freqs=True):
cuvarbase/bls.py:            self.transfer_data_to_gpu(transfer_freqs=(freqs is not None))
cuvarbase/bls.py:def eebls_gpu_fast(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
cuvarbase/bls.py:    Box-Least Squares with PyCUDA but about 2-3 orders of magnitude
cuvarbase/bls.py:    faster than eebls_gpu. Uses shared memory for the binned data,
cuvarbase/bls.py:    kept. To get the best solution, run ``eebls_gpu`` at the
cuvarbase/bls.py:        If you are running on a single-GPU machine, there may be a
cuvarbase/bls.py:        This is GPU-dependent but usually around 48KB. If ``None``,
cuvarbase/bls.py:        uses device information provided by PyCUDA (recommended).
cuvarbase/bls.py:        Transfer data to GPU
cuvarbase/bls.py:        dev = pycuda.autoprimaryctx.device
cuvarbase/bls.py:        att = cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
cuvarbase/bls.py:        shmem_lim = pycuda.autoprimaryctx.device.get_attribute(att)
cuvarbase/bls.py:            s += " or avoid using eebls_gpu_fast."
cuvarbase/bls.py:def eebls_gpu_custom(t, y, dy, freqs, q_values, phi_values,
cuvarbase/bls.py:        Number of CUDA streams to utilize.
cuvarbase/bls.py:        free memory given by ``pycuda.driver.mem_get_info()``
cuvarbase/bls.py:    functions: tuple of CUDA functions
cuvarbase/bls.py:        free, total = cuda.mem_get_info()
cuvarbase/bls.py:    # move data to GPU
cuvarbase/bls.py:    t_g = gpuarray.to_gpu(np.array(t).astype(np.float32))
cuvarbase/bls.py:    yw_g = gpuarray.to_gpu(yw.astype(np.float32))
cuvarbase/bls.py:    w_g = gpuarray.to_gpu(np.array(w).astype(np.float32))
cuvarbase/bls.py:    freqs_g = gpuarray.to_gpu(np.array(freqs).astype(np.float32))
cuvarbase/bls.py:        streams.append(cuda.Stream())
cuvarbase/bls.py:        yw_g_bins.append(gpuarray.zeros(nbtot, dtype=np.float32))
cuvarbase/bls.py:        w_g_bins.append(gpuarray.zeros(nbtot, dtype=np.float32))
cuvarbase/bls.py:        bls_tmp_gs.append(gpuarray.zeros(nbtot, dtype=np.float32))
cuvarbase/bls.py:        bls_tmp_sol_gs.append(gpuarray.zeros(nbtot, dtype=np.uint32))
cuvarbase/bls.py:    bls_g = gpuarray.zeros(len(freqs), dtype=np.float32)
cuvarbase/bls.py:    bls_sol_g = gpuarray.zeros(len(freqs), dtype=np.uint32)
cuvarbase/bls.py:    bls_best_phi = gpuarray.zeros(len(freqs), dtype=np.float32)
cuvarbase/bls.py:    bls_best_q = gpuarray.zeros(len(freqs), dtype=np.float32)
cuvarbase/bls.py:    q_values_g = gpuarray.to_gpu(np.asarray(q_values).astype(np.float32))
cuvarbase/bls.py:    phi_values_g = gpuarray.to_gpu(np.asarray(phi_values).astype(np.float32))
cuvarbase/bls.py:def eebls_gpu(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
cuvarbase/bls.py:    Box-Least Squares, accelerated with PyCUDA
cuvarbase/bls.py:        Number of CUDA streams to utilize.
cuvarbase/bls.py:        as returned by ``pycuda.driver.mem_get_info`` if this is ``None``.
cuvarbase/bls.py:    functions: tuple of CUDA functions
cuvarbase/bls.py:        free, total = cuda.mem_get_info()
cuvarbase/bls.py:    # move data to GPU
cuvarbase/bls.py:    t_g = gpuarray.to_gpu(np.array(t).astype(np.float32))
cuvarbase/bls.py:    yw_g = gpuarray.to_gpu(yw.astype(np.float32))
cuvarbase/bls.py:    w_g = gpuarray.to_gpu(np.array(w).astype(np.float32))
cuvarbase/bls.py:    freqs_g = gpuarray.to_gpu(np.array(freqs).astype(np.float32))
cuvarbase/bls.py:        streams.append(cuda.Stream())
cuvarbase/bls.py:        yw_g_bins.append(gpuarray.zeros(gs, dtype=np.float32))
cuvarbase/bls.py:        w_g_bins.append(gpuarray.zeros(gs, dtype=np.float32))
cuvarbase/bls.py:        bls_tmp_gs.append(gpuarray.zeros(gs, dtype=np.float32))
cuvarbase/bls.py:        bls_tmp_sol_gs.append(gpuarray.zeros(gs, dtype=np.int32))
cuvarbase/bls.py:    bls_g = gpuarray.zeros(len(freqs), dtype=np.float32)
cuvarbase/bls.py:    bls_sol_g = gpuarray.zeros(len(freqs), dtype=np.int32)
cuvarbase/bls.py:    bls_best_phi = gpuarray.zeros(len(freqs), dtype=np.float32)
cuvarbase/bls.py:    bls_best_q = gpuarray.zeros(len(freqs), dtype=np.float32)
cuvarbase/bls.py:        powers, sols = eebls_gpu_custom(t, y, dy, freqs, q_values, phi_values,
cuvarbase/bls.py:def eebls_transit_gpu(t, y, dy, fmax_frac=1.0, fmin_frac=1.0,
cuvarbase/bls.py:        passed to `eebls_gpu`, `compile_bls`, `fmax_transit`,
cuvarbase/bls.py:        powers = eebls_gpu_fast(t, y, dy, freqs,
cuvarbase/bls.py:    powers, sols = eebls_gpu(t, y, dy, freqs,
cuvarbase/kernels/cunfft.cu:#include <pycuda-complex.hpp>
cuvarbase/kernels/cunfft.cu:#define CMPLX pycuda::complex<FLT>
cuvarbase/kernels/lomb.cu:#include <pycuda-complex.hpp>
cuvarbase/kernels/lomb.cu:__global__ void lomb(pycuda::complex<FLT>  *sw,
cuvarbase/kernels/lomb.cu:					 pycuda::complex<FLT>  *syw,
cuvarbase/kernels/lomb.cu:		pycuda::complex<FLT> SW, SW2, SYW;
cuvarbase/kernels/lomb.cu:__global__ void lomb_mh(pycuda::complex<FLT>  *sw,
cuvarbase/kernels/lomb.cu:					    pycuda::complex<FLT>  *syw,
cuvarbase/kernels/lomb.cu:		pycuda::complex<FLT> SW, SW2, SYW;
cuvarbase/__init__.py:# import pycuda.autoinit causes problems when running e.g. FFT
cuvarbase/__init__.py:import pycuda.autoprimaryctx
cuvarbase/core.py:import pycuda.driver as cuda
cuvarbase/core.py:from pycuda.compiler import SourceModule
cuvarbase/core.py:class GPUAsyncProcess(object):
cuvarbase/core.py:        self.gpu_data = []
cuvarbase/core.py:            self.streams.append(cuda.Stream())
cuvarbase/cunfft.py:import pycuda.driver as cuda
cuvarbase/cunfft.py:import pycuda.gpuarray as gpuarray
cuvarbase/cunfft.py:from pycuda.compiler import SourceModule
cuvarbase/cunfft.py:# import pycuda.autoinit
cuvarbase/cunfft.py:import skcuda.fft as cufft
cuvarbase/cunfft.py:from .core import GPUAsyncProcess
cuvarbase/cunfft.py:        self.t_g = gpuarray.zeros(self.n0, dtype=self.real_type)
cuvarbase/cunfft.py:        self.y_g = gpuarray.zeros(self.n0, dtype=self.real_type)
cuvarbase/cunfft.py:        self.q1 = gpuarray.zeros(self.n0, dtype=self.real_type)
cuvarbase/cunfft.py:        self.q2 = gpuarray.zeros(self.n0, dtype=self.real_type)
cuvarbase/cunfft.py:        self.q3 = gpuarray.zeros(2 * self.m + 1, dtype=self.real_type)
cuvarbase/cunfft.py:        self.ghat_g = gpuarray.zeros(self.n,
cuvarbase/cunfft.py:        self.ghat_c = cuda.aligned_zeros(shape=(self.nf,),
cuvarbase/cunfft.py:        self.ghat_c = cuda.register_host_memory(self.ghat_c)
cuvarbase/cunfft.py:    def transfer_data_to_gpu(self, **kwargs):
cuvarbase/cunfft.py:        cuda.memcpy_dtoh_async(self.ghat_c, self.ghat_g.ptr,
cuvarbase/cunfft.py:        Number of CUDA threads per block
cuvarbase/cunfft.py:    use_grid: ``GPUArray``, optional
cuvarbase/cunfft.py:        If specified, will skip gridding procedure and use the `GPUArray`
cuvarbase/cunfft.py:        If the data is already on the gpu, set as False
cuvarbase/cunfft.py:    # transfer data -> gpu
cuvarbase/cunfft.py:        memory.transfer_data_to_gpu()
cuvarbase/cunfft.py:class NFFTAsyncProcess(GPUAsyncProcess):
cuvarbase/cunfft.py:    `GPUAsyncProcess` for the adjoint NFFT.
cuvarbase/cunfft.py:        CUDA block size.
cuvarbase/cunfft.py:        Allocate GPU memory for NFFT-related computations
requirements.txt:pycuda >= 2017.1.1
requirements.txt:scikit-cuda
INSTALL.rst:Installing the Nvidia Toolkit
INSTALL.rst:``cuvarbase`` requires PyCUDA and scikit-cuda, which both require the Nvidia toolkit for access to the Nvidia compiler, drivers, and runtime libraries.
INSTALL.rst:Go to the `NVIDIA Download page <https://developer.nvidia.com/cuda-downloads>`_ and select the distribution for your operating system. Everything has been developed and tested using **version 8.0**, so it may be best to stick with that version for now until we verify that later versions are OK.
INSTALL.rst:	Make sure that your ``$PATH`` environment variable contains the location of the ``CUDA`` binaries. You can test this by trying
INSTALL.rst:	``echo "export PATH=/usr/local/cuda/bin:${PATH}" >> ~/.bashrc && . ~/.bashrc``
INSTALL.rst:	The ``>>`` is not a typo -- using one ``>`` will *overwrite* the ``~/.bashrc`` file. Make sure you change ``/usr/local/cuda`` to the appropriate location of your Nvidia install.
INSTALL.rst:	Make sure your ``$LD_LIBRARY_PATH`` and ``$DYLD_LIBRARY_PATH`` are also similarly modified to include the ``/lib`` directory of the CUDA install:
INSTALL.rst:	``echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib:${LD_LIBRARY_PATH}" >> ~/.bashrc && . ~/.bashrc``
INSTALL.rst:	``echo "export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:${DYLD_LIBRARY_PATH}" >> ~/.bashrc && . ~/.bashrc``
INSTALL.rst:	The numpy library *has* to be installed *before* PyCUDA is installed with pip. 
INSTALL.rst:	The PyCUDA setup needs to be able to access the numpy library for building against it. You can do this with
INSTALL.rst:Troubleshooting PyCUDA installation problems
INSTALL.rst:The ``PyCUDA`` installation step may be a hiccup in this otherwise orderly process. If you run into problems installing ``PyCUDA`` with pip, you may have to install PyCUDA from source yourself. It's not too bad, but if you experience any problems, please submit an `Issue <https://github.com/johnh2o2/cuvarbase/issues>`_ at the ``cuvarbase`` Github page and I'll amend this documentation.
INSTALL.rst:Below is a small bash script that (hopefully) automates the process of installing PyCUDA in the event of any problems you've encountered at this point.
INSTALL.rst:	PYCUDA="pycuda-2017.1.1"
INSTALL.rst:	PYCUDA_URL="https://pypi.python.org/packages/b3/30/9e1c0a4c10e90b4c59ca7aa3c518e96f37aabcac73ffe6b5d9658f6ef843/pycuda-2017.1.1.tar.gz#md5=9e509f53a23e062b31049eb8220b2e3d"
INSTALL.rst:	CUDA_ROOT=/usr/local/cuda
INSTALL.rst:	wget $PYCUDA_URL
INSTALL.rst:	tar xvf ${PYCUDA}.tar.gz
INSTALL.rst:	cd $PYCUDA
INSTALL.rst:	./configure.py --python-exe=`which python` --cuda-root=$CUDA_ROOT
INSTALL.rst:If everything goes smoothly, you should now test if ``pycuda`` is working correctly.
INSTALL.rst:	python -c "import pycuda.autoinit; print 'Hurray!'"
INSTALL.rst:Nvidia offers `CUDA for Mac OSX <https://developer.nvidia.com/cuda-downloads>`_. After installing the
INSTALL.rst:    export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:/usr/local/cuda/lib"
INSTALL.rst:    export PATH="/usr/local/cuda/bin:${PATH}"
test_python_versions.sh:# Put your cuda installation directory here
test_python_versions.sh:export CUDA_ROOT=/usr/local/cuda
test_python_versions.sh:export LD_LIBRARY_PATH="${CUDA_ROOT}/lib:${LD_LIBRARY_PATH}"
test_python_versions.sh:export DYLD_LIBRARY_PATH="${CUDA_ROOT}/lib:${DYLD_LIBRARY_PATH}"
test_python_versions.sh:export PATH="${CUDA_ROOT}/bin:${PATH}"
CHANGELOG.rst:    * swap out pycuda.autoinit for pycuda.autoprimaryctx to handle "cuFuncSetBlockShape" error
CHANGELOG.rst:		* Now several orders of magnitude faster! Use ``use_fast=True`` in ``eebls_transit_gpu`` or use ``eebls_gpu_fast``.
CHANGELOG.rst:		* Bug-fix for boost-python error when calling ``eebls_gpu_fast``.
CHANGELOG.rst:		* Avoids allocating GPU memory for NFFT when ``use_fft`` is ``False``.
CHANGELOG.rst:		* ``eebls_gpu``, ``eebls_transit_gpu``, and ``eebls_custom_gpu`` now have a ``max_memory`` option that allows you to automatically set the ``batch_size`` without worrying about memory allocation errors.
CHANGELOG.rst:		* ``eebls_transit_gpu`` now allows for a ``freqs`` argument and a ``qvals`` argument for customizing the frequencies and the fiducial ``q`` values
CHANGELOG.rst:		* A new transiting exoplanet BLS function: ``eebls_transit_gpu``

```
