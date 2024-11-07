# https://github.com/gwastro/pycbc

```console
setup.py:    'cuda': [
setup.py:        'pycuda>=2015.1',
setup.py:        'scikit-cuda',
pycbc.spec:-nogpu, -cuda or -opencl packages.
pycbc.spec:%package nogpu
pycbc.spec:%description nogpu
pycbc.spec:#%package cuda
pycbc.spec:#Summary: CUDA version
pycbc.spec:#%description cuda
pycbc.spec:#Version supporting GPU computation via CUDA.
pycbc.spec:#%package opencl
pycbc.spec:#Summary: OpenCL version
pycbc.spec:#%description opencl
pycbc.spec:#Version supporting GPU computation via OpenCL.
pycbc.spec:%files nogpu
docs/workflow/matched_filter.rst:   pycbc_inspiral --trig-end-time 961592867 --verbose  --cluster-method window --bank-filetmpltbank/L1-TMPLTBANK_01-961591486-1382.xml.gz --gps-end-time 961592884 --channel-name L1:LDAS-STRAIN --processing-scheme cuda --snr-threshold 5.5 --psd-estimation median --trig-start-time 961591534 --gps-start-time 961590836 --chisq-bins 16 --segment-end-pad 16 --segment-length 2048 --low-frequency-cutoff 15 --pad-data 8 --cluster-window 1 --sample-rate 4096 --segment-start-pad 650 --psd-segment-stride 32 --psd-inverse-length 16 --psd-segment-length 64 --frame-cache datafind/L1-DATAFIND-961585543-7349.lcf --approximant SPAtmplt --output inspiral/L1-INSPIRAL_1-961591534-1333.xml.gz --strain-high-pass 30 --order 7 
docs/install_cuda.rst:Instructions to add CUDA support (optional)
docs/install_cuda.rst:If you would like to use GPU acceleration of PyCBC through CUDA you will require these additional packages:
docs/install_cuda.rst:* `Nvidia CUDA <http://www.nvidia.com/object/cuda_home_new.html>`_ >= 6.5 (driver and libraries)
docs/install_cuda.rst:* `PyCUDA <http://mathema.tician.de/software/pycuda>`_ >= 2015.1.3
docs/install_cuda.rst:* `SciKits.cuda <http://scikits.appspot.com/cuda>`_ >= 0.041
docs/install_cuda.rst:The install requires that you set the environment variable ``CUDA_ROOT``, make sure that the CUDA ``bin`` directory is in your path, and add the CUDA library path to your ``LD_LIBRARY_PATH``. You can do this by adding these commands to your ``activate`` script by running the commands:
docs/install_cuda.rst:    echo 'export CUDA_ROOT=/usr/local/cuda' >> $NAME/bin/activate
docs/install_cuda.rst:    echo 'export PATH=${CUDA_ROOT}/bin:${PATH}' >> $NAME/bin/activate
docs/install_cuda.rst:    echo 'export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}' >> $NAME/bin/activate
docs/install_cuda.rst:Installing the CUDA dependencies
docs/install_cuda.rst:Install the dependencies PyCUDA, SciKits.cuda and Mako with by running the commands
docs/install_cuda.rst:    pip install pycuda
docs/install_cuda.rst:    pip install scikit-cuda
docs/install_cuda.rst:You should now be able to use the CUDA features in PyCBC.
docs/fft.rst:When running on GPUs, PyCBC knows how to do CUDA FFTs through the same
docs/install.rst:Graphics Processing Unit support with CUDA
docs/install.rst:PyCBC has the ability to accelerate its processing using CUDA. To take advantage of this, follow the instructions linked below to install the CUDA dependencies.
docs/install.rst:    install_cuda
docs/remove_non_standard_imports.py:capability to exclude specific files from the documentation. The GPU modules
docs/remove_non_standard_imports.py:excludes=['cuda', 'cufft', 'cuda_pyfft', 'cl_pyfft',\
docs/banksim.rst:    An optional value 'use-gpus' can be set. This will 
docs/banksim.rst:    set up the workflow to choose condor nodes with GPUS
docs/banksim.rst:    will use the GPU for accelerated processing. Note that
docs/banksim.rst:    the default is to treat all results from a GPU as
docs/banksim.rst:    they equivelant. Only the GPUs on SUGAR and ATLAS
docs/banksim.rst:    correct flag for GPU support will be set if the 'use-gpus'
test/test_frequencyseries.py:if _scheme == 'cuda':
test/test_frequencyseries.py:    import pycuda
test/test_frequencyseries.py:    import pycuda.gpuarray
test/test_frequencyseries.py:    from pycuda.gpuarray import GPUArray as SchemeArray
test/test_frequencyseries.py:            # We also need to check initialization using GPU arrays
test/test_frequencyseries.py:            if self.scheme == 'cuda':
test/test_frequencyseries.py:                in4 = pycuda.gpuarray.zeros(3,self.dtype)
test/test_frequencyseries.py:        # this array is made outside the context so we can check that an error is raised when copy = false in a GPU scheme
test/test_timeseries.py:if _scheme == 'cuda':
test/test_timeseries.py:    import pycuda
test/test_timeseries.py:    import pycuda.gpuarray
test/test_timeseries.py:    from pycuda.gpuarray import GPUArray as SchemeArray
test/test_timeseries.py:            # We also need to check initialization using GPU arrays
test/test_timeseries.py:            if self.scheme == 'cuda':
test/test_timeseries.py:                in4 = pycuda.gpuarray.zeros(3,self.dtype)
test/test_timeseries.py:        # this array is made outside the context so we can check that an error is raised when copy = false in a GPU scheme
test/test_waveform.py:    def test_spintaylorf2GPU(self):
test/lalsim.py:                   choices = ('cpu','cuda'),
test/utils.py:unit tests easier, while still allowing the tests to be run on CPU and CUDA
test/utils.py:scheme is each of 'cpu', 'cuda' in turn. Unit tests
test/utils.py:Some other unit tests may be for features or sub-packages that are not GPU
test/utils.py:capable, and cannot be meaningfully tested on the GPU.  Those tests, after
test/utils.py:specifying a GPU environment (since setup.py does not know which tests are GPU
test/utils.py:capable and which are not) but when called with a GPU scheme will exit immediately.
test/utils.py:from pycbc.scheme import CPUScheme, CUDAScheme
test/utils.py:    if scheme=='cuda' and not pycbc.HAVE_CUDA:
test/utils.py:        raise optparse.OptionValueError("CUDA not found")
test/utils.py:                       choices = ('cpu','cuda'),
test/utils.py:                       help = 'specifies processing scheme, can be cpu [default], cuda')
test/utils.py:                       help = 'specifies a GPU device to use for CUDA, 0 by default')
test/utils.py:    if _scheme == 'cuda':
test/utils.py:        _context = CUDAScheme(device_num=_options['devicenum'])
test/utils.py:    _scheme_dict = { 'cpu': 'CPU', 'cuda': 'CUDA'}
test/utils.py:    if scheme=='cuda':
test/utils.py:                       choices = ('cpu','cuda'),
test/utils.py:                       help = 'specifies processing scheme, can be cpu [default], cuda')
test/utils.py:                       help = 'specifies a GPU device to use for CUDA, 0 by default')
test/utils.py:    # a GPU scheme.  So if we get here we're on the CPU, and should print out our message
test/utils.py:    when that happens on one of the GPU schemes) and that these exit statuses could then be
test/utils.py:            # must use almost-equal comparisons, esp. on the GPU
test/test_array.py:if _scheme == 'cuda':
test/test_array.py:    import pycuda
test/test_array.py:    import pycuda.gpuarray
test/test_array.py:    from pycuda.gpuarray import GPUArray as SchemeArray
test/test_array.py:            # We also need to check initialization using GPU arrays
test/test_array.py:            if self.scheme == 'cuda':
test/test_array.py:                in4 = pycuda.gpuarray.zeros(3,self.dtype)
test/test_array.py:        # this array is made outside the context so we can check that an error is raised when copy = false in a GPU scheme
test/test_schemes.py:and that data are moved to and from the GPU as they should, or apprpriate exceptions
test/test_schemes.py:GPU, because that test is done in the test_lalwrap unit tests.
test/test_schemes.py:if isinstance(_context,CUDAScheme):
test/test_schemes.py:    import pycuda
test/test_schemes.py:    import pycuda.gpuarray
test/test_schemes.py:    from pycuda.gpuarray import GPUArray as SchemeArray
test/test_schemes.py:        on and off of the GPU automatically when they should be, and that the _scheme
test/test_schemes.py:            # The following should move both of a1 and b1 onto the GPU (if self.context
test/test_schemes.py:        does its comparisons by copying from the CPU to GPU, but should
test/test_schemes.py:            # Force a move to the GPU by trivially multiplying by one:
test/fft_base.py:it will not in general hold for GPU algorithms (and those cannot be made to enforce
test/fft_base.py:    # Perform arithmetic on outarr and inarr to pull them off of the GPU:
test/test_tmpltbank.py:# particular instance of the unittest was called for CPU, CUDA, or OpenCL
pycbc/filter/matchedfilter_cuda.py:from pycuda.elementwise import ElementwiseKernel
pycbc/filter/matchedfilter_cuda.py:from pycuda.tools import context_dependent_memoize
pycbc/filter/matchedfilter_cuda.py:from pycuda.tools import dtype_to_ctype
pycbc/filter/matchedfilter_cuda.py:from pycuda.gpuarray import _get_common_dtype
pycbc/filter/matchedfilter_cuda.py:class CUDACorrelator(_BaseCorrelator):
pycbc/filter/matchedfilter_cuda.py:    return CUDACorrelator
pycbc/filter/matchedfilter.py:                 gpu_callback_method='none', cluster_function='symmetric'):
pycbc/filter/matchedfilter.py:        self.gpu_callback_method = gpu_callback_method
pycbc/fft/fft_callback.py:    #include <cuda_runtime.h>
pycbc/fft/fft_callback.py:    #define checkCudaErrors(val)  __checkCudaErrors__ ( (val), #val, __FILE__, __LINE__ )
pycbc/fft/fft_callback.py:    inline void __checkCudaErrors__(T code, const char *func, const char *file, int line)
pycbc/fft/fft_callback.py:            fprintf(stderr, "CUDA error at %s:%d code=%d \\"%s\\" \\n",
pycbc/fft/fft_callback.py:            cudaDeviceReset();
pycbc/fft/fft_callback.py:        checkCudaErrors(cufftMakePlan1d(*plan, size, CUFFT_C2C, 1, &work_size));
pycbc/fft/fft_callback.py:            checkCudaErrors(cudaMemcpyFromSymbol(&h_input_callback, input_callback,
pycbc/fft/fft_callback.py:            checkCudaErrors(cufftXtSetCallback(*plan, (void **) &h_input_callback,
pycbc/fft/fft_callback.py:            checkCudaErrors(cudaMemcpyFromSymbol(&h_output_callback, output_callback,
pycbc/fft/fft_callback.py:            checkCudaErrors(cufftXtSetCallback(*plan, (void **) &h_output_callback,
pycbc/fft/fft_callback.py:            checkCudaErrors(cudaMemcpyToSymbolAsync(callback_params, p, sizeof(param_t), 0,  cudaMemcpyHostToDevice, 0));
pycbc/fft/fft_callback.py:         checkCudaErrors(cufftExecC2C(*plan, in, out, CUFFT_INVERSE));
pycbc/fft/fft_callback.py:        plan = pfn(len(outvec), int(htilde.data.gpudata))
pycbc/fft/fft_callback.py:        _plans[key] = (fn, plan, int(htilde.data.gpudata))
pycbc/fft/fft_callback.py:    hparam.htilde = htilde.data.gpudata
pycbc/fft/fft_callback.py:    fn(plan, int(stilde.data.gpudata), int(outvec.data.gpudata), ctypes.pointer(hparam))
pycbc/fft/fft_callback.py:        plan = pfn(len(outvec), int(htilde.data.gpudata))
pycbc/fft/fft_callback.py:        _plans[key] = (fn, plan, int(htilde.data.gpudata))
pycbc/fft/fft_callback.py:    hparam_zeros.htilde = htilde.data.gpudata
pycbc/fft/fft_callback.py:    fn(plan, int(stilde.data.gpudata), int(outvec.data.gpudata), ctypes.pointer(hparam_zeros))
pycbc/fft/cuda_pyfft.py:from pyfft.cuda import Plan
pycbc/fft/cuda_pyfft.py:#These dicts need to be cleared before the cuda context is destroyed
pycbc/fft/cuda_pyfft.py:pycbc.scheme.register_clean_cuda(_clear_plan_dict)
pycbc/fft/backend_cuda.py:_backend_dict = {'cuda' : 'cufft',
pycbc/fft/backend_cuda.py:                 'pyfft' : 'cuda_pyfft'}
pycbc/fft/backend_cuda.py:_backend_list = ['cuda','pyfft']
pycbc/fft/backend_cuda.py:if pycbc.HAVE_CUDA:
pycbc/fft/backend_cuda.py:cuda_backend = None
pycbc/fft/backend_cuda.py:    global cuda_backend
pycbc/fft/backend_cuda.py:            cuda_backend = backend
pycbc/fft/backend_cuda.py:    return _adict[cuda_backend]
pycbc/fft/backend_support.py:for scheme_name in ["cpu", "mkl", "cuda"]:
pycbc/fft/cufft.py:    import skcuda.fft as cu_fft
pycbc/fft/cufft.py:    raise ImportError("Unable to import skcuda.fft; try direct import"
pycbc/fft/cufft.py:#These dicts need to be cleared before the cuda context is destroyed
pycbc/fft/cufft.py:pycbc.scheme.register_clean_cuda(_clear_plan_dicts)
pycbc/types/frequencyseries.py:        object to a GPU object, and the comparison should be true if the
pycbc/types/frequencyseries.py:        equal.  If either object's memory lives on the GPU it will be
pycbc/types/frequencyseries.py:        equal.  If either object's memory lives on the GPU it will be
pycbc/types/frequencyseries.py:            If frequency series is stored in GPU memory.
pycbc/types/timeseries.py:        object to a GPU object, and the comparison should be true if the
pycbc/types/timeseries.py:        equal.  If either object's memory lives on the GPU it will be
pycbc/types/timeseries.py:        equal.  If either object's memory lives on the GPU it will be
pycbc/types/timeseries.py:            If time series is stored in GPU memory.
pycbc/types/array.py:This modules provides a device independent Array class based on PyCUDA and Numpy.
pycbc/types/array.py:    pycuda.
pycbc/types/array.py:        either object. It is possible to compare a CPU object to a GPU
pycbc/types/array.py:        # The numpy() method call will put a copy of GPU data onto a CPU
pycbc/types/array.py:        If either object's memory lives on the GPU it will be copied to
pycbc/types/array.py:        # The numpy() method will move any GPU memory onto the CPU.
pycbc/types/array.py:        If either object's memory lives on the GPU it will be copied to
pycbc/types/array.py:        # The numpy() method will move any GPU memory onto the CPU.
pycbc/types/array.py:           slow if the memory is on a gpu.
pycbc/types/array_cuda.py:"""Pycuda based 
pycbc/types/array_cuda.py:import pycuda.driver
pycbc/types/array_cuda.py:from pycuda.elementwise import ElementwiseKernel
pycbc/types/array_cuda.py:from pycuda.reduction import ReductionKernel
pycbc/types/array_cuda.py:from pycuda.tools import get_or_register_dtype
pycbc/types/array_cuda.py:from pycuda.tools import context_dependent_memoize
pycbc/types/array_cuda.py:from pycuda.tools import dtype_to_ctype
pycbc/types/array_cuda.py:from pycuda.gpuarray import _get_common_dtype, empty, GPUArray
pycbc/types/array_cuda.py:import pycuda.gpuarray
pycbc/types/array_cuda.py:from pycuda.scan import InclusiveScanKernel
pycbc/types/array_cuda.py:#include <pycuda-complex.hpp>
pycbc/types/array_cuda.py:            s1_invocation_args.append(arg.gpudata)
pycbc/types/array_cuda.py:                *([result.gpudata]+s1_invocation_args+[seq_count, sz]),
pycbc/types/array_cuda.py:                    *([result.gpudata, result2.gpudata]+s1_invocation_args+[seq_count, sz]),
pycbc/types/array_cuda.py:# Define PYCUDA MAXLOC for both single and double precission ################## 
pycbc/types/array_cuda.py:        arguments="pycuda::complex<float> *x", preamble=maxloc_preamble_single)
pycbc/types/array_cuda.py:        arguments="pycuda::complex<double> *x", preamble=maxloc_preamble_double)
pycbc/types/array_cuda.py:    result = GPUArray(length, dtype=dtype)
pycbc/types/array_cuda.py:    pycuda.driver.memset_d32(result.gpudata, 0, nwords)
pycbc/types/array_cuda.py:    return pycuda.gpuarray.dot(self._data,other).get().max()
pycbc/types/array_cuda.py:    return pycuda.gpuarray.min(self._data).get().max()
pycbc/types/array_cuda.py:    return pycuda.gpuarray.max(self._data).get().max()
pycbc/types/array_cuda.py:    if not isinstance(indices, pycuda.gpuarray.GPUArray):
pycbc/types/array_cuda.py:        indices = pycuda.gpuarray.to_gpu(indices)
pycbc/types/array_cuda.py:    return pycuda.gpuarray.take(self.data, indices)
pycbc/types/array_cuda.py:        from pycuda.elementwise import get_copy_kernel
pycbc/types/array_cuda.py:                self_ref.gpudata, other_ref.gpudata,
pycbc/types/array_cuda.py:    return pycuda.gpuarray.sum(self._data).get().max()
pycbc/types/array_cuda.py:    pycuda.driver.memset_d32(self.data.gpudata, 0, n32)
pycbc/types/array_cuda.py:    if isinstance(array, pycuda.gpuarray.GPUArray):
pycbc/types/array_cuda.py:    data = pycuda.gpuarray.GPUArray((array.size), array.dtype)
pycbc/types/array_cuda.py:        pycuda.driver.memcpy_dtod(data.gpudata, array.gpudata, array.nbytes)
pycbc/types/array_cuda.py:    return pycuda.gpuarray.to_gpu(array)
pycbc/vetoes/chisq_cuda.py:import pycuda.driver, numpy
pycbc/vetoes/chisq_cuda.py:from pycuda.elementwise import ElementwiseKernel
pycbc/vetoes/chisq_cuda.py:from pycuda.tools import context_dependent_memoize, dtype_to_ctype
pycbc/vetoes/chisq_cuda.py:import pycuda.gpuarray
pycbc/vetoes/chisq_cuda.py:from pycuda.compiler import SourceModule
pycbc/vetoes/chisq_cuda.py:    bv = pycuda.gpuarray.to_gpu_async(numpy.array(bv, dtype=numpy.uint32))
pycbc/vetoes/chisq_cuda.py:    kmin = pycuda.gpuarray.to_gpu_async(numpy.array(kmin, dtype=numpy.uint32))
pycbc/vetoes/chisq_cuda.py:    kmax = pycuda.gpuarray.to_gpu_async(numpy.array(kmax, dtype=numpy.uint32))
pycbc/vetoes/chisq_cuda.py:    #fuse = 'fuse' in corr.gpu_callback_method
pycbc/vetoes/chisq_cuda.py:        args += [corr.htilde.data.gpudata, corr.stilde.data.gpudata]
pycbc/vetoes/chisq_cuda.py:        args += [corr.data.gpudata]
pycbc/vetoes/chisq_cuda.py:    args +=[outp.gpudata, N] + phase[0:num] + [kmin.gpudata, kmax.gpudata, bv.gpudata, nbins]
pycbc/vetoes/chisq_cuda.py:    #fuse = 'fuse' in corr.gpu_callback_method
pycbc/vetoes/chisq_cuda.py:        args += [corr.htilde.data.gpudata, corr.stilde.data.gpudata]
pycbc/vetoes/chisq_cuda.py:        args += [corr.data.gpudata]
pycbc/vetoes/chisq_cuda.py:    args += [outp.gpudata, N] + points[0:num] + [kmin.gpudata,
pycbc/vetoes/chisq_cuda.py:                                kmax.gpudata, bv.gpudata, nbins]
pycbc/vetoes/chisq_cuda.py:    outc = pycuda.gpuarray.zeros((len(points), nbins), dtype=numpy.complex64)
pycbc/waveform/waveform.py:# Waveforms written in CUDA
pycbc/waveform/waveform.py:_cuda_td_approximants = {}
pycbc/waveform/waveform.py:_cuda_fd_approximants = {}
pycbc/waveform/waveform.py:if pycbc.HAVE_CUDA:
pycbc/waveform/waveform.py:    from pycbc.waveform.SpinTaylorF2 import spintaylorf2 as cuda_spintaylorf2
pycbc/waveform/waveform.py:    _cuda_fd_approximants["IMRPhenomC"] = imrphenomc_tmplt
pycbc/waveform/waveform.py:    _cuda_fd_approximants["SpinTaylorF2"] = cuda_spintaylorf2
pycbc/waveform/waveform.py:cuda_td = dict(list(_lalsim_td_approximants.items()) + list(_cuda_td_approximants.items()))
pycbc/waveform/waveform.py:cuda_fd = dict(list(_lalsim_fd_approximants.items()) + list(_cuda_fd_approximants.items()))
pycbc/waveform/waveform.py:    print("CUDA Approximants")
pycbc/waveform/waveform.py:    for approx in _cuda_td_approximants.keys():
pycbc/waveform/waveform.py:    print("CUDA Approximants")
pycbc/waveform/waveform.py:    for approx in _cuda_fd_approximants.keys():
pycbc/waveform/waveform.py:_cuda_fd_filters = {}
pycbc/waveform/waveform.py:_cuda_fd_filters['SPAtmplt'] = spa_tmplt
pycbc/waveform/waveform.py:                    _scheme.CUDAScheme:_cuda_fd_filters,
pycbc/waveform/waveform.py:td_wav.update({_scheme.CPUScheme:cpu_td,_scheme.CUDAScheme:cuda_td})
pycbc/waveform/waveform.py:fd_wav.update({_scheme.CPUScheme:cpu_fd,_scheme.CUDAScheme:cuda_fd})
pycbc/waveform/pycbc_phenomC_tmplt.py:from pycuda.elementwise import ElementwiseKernel
pycbc/waveform/pycbc_phenomC_tmplt.py:phenomC_kernel = ElementwiseKernel("""pycuda::complex<double> *htilde, int kmin, double delta_f,
pycbc/waveform/pycbc_phenomC_tmplt.py:    """ Return an IMRPhenomC waveform using CUDA to generate the phase and amplitude
pycbc/waveform/decompress_cuda.py:from pycuda import gpuarray
pycbc/waveform/decompress_cuda.py:from pycuda.compiler import SourceModule
pycbc/waveform/decompress_cuda.py:# and phase values) are stored as 1D textures on the GPU, because many
pycbc/waveform/decompress_cuda.py:class CUDALinearInterpolate(object):
pycbc/waveform/decompress_cuda.py:        self.output = output.data.gpudata
pycbc/waveform/decompress_cuda.py:        self.lower = zeros(self.nb, dtype=numpy.int32).data.gpudata
pycbc/waveform/decompress_cuda.py:        self.upper = zeros(self.nb, dtype=numpy.int32).data.gpudata
pycbc/waveform/decompress_cuda.py:        freqs_gpu = gpuarray.to_gpu(freqs)
pycbc/waveform/decompress_cuda.py:        freqs_gpu.bind_to_texref_ext(self.freq_tex, allow_offset=False)
pycbc/waveform/decompress_cuda.py:        amps_gpu = gpuarray.to_gpu(amps)
pycbc/waveform/decompress_cuda.py:        amps_gpu.bind_to_texref_ext(self.amp_tex, allow_offset=False)
pycbc/waveform/decompress_cuda.py:        phases_gpu = gpuarray.to_gpu(phases)
pycbc/waveform/decompress_cuda.py:        phases_gpu.bind_to_texref_ext(self.phase_tex, allow_offset=False)
pycbc/waveform/decompress_cuda.py:    # Note that imin and start_index are ignored in the GPU code; they are only
pycbc/waveform/decompress_cuda.py:        raise NotImplementedError("Double precision linear interpolation not currently supported on CUDA scheme")
pycbc/waveform/decompress_cuda.py:    freqs_gpu = gpuarray.to_gpu(freqs)
pycbc/waveform/decompress_cuda.py:    freqs_gpu.bind_to_texref_ext(ftex, allow_offset=False)
pycbc/waveform/decompress_cuda.py:    amps_gpu = gpuarray.to_gpu(amps)
pycbc/waveform/decompress_cuda.py:    amps_gpu.bind_to_texref_ext(atex, allow_offset=False)
pycbc/waveform/decompress_cuda.py:    phases_gpu = gpuarray.to_gpu(phases)
pycbc/waveform/decompress_cuda.py:    phases_gpu.bind_to_texref_ext(ptex, allow_offset=False)
pycbc/waveform/decompress_cuda.py:    g_out = output.data.gpudata
pycbc/waveform/decompress_cuda.py:    lower = zeros(nb, dtype=numpy.int32).data.gpudata
pycbc/waveform/decompress_cuda.py:    upper = zeros(nb, dtype=numpy.int32).data.gpudata
pycbc/waveform/SpinTaylorF2.py:from pycuda.elementwise import ElementwiseKernel
pycbc/waveform/SpinTaylorF2.py:spintaylorf2_kernel = ElementwiseKernel("""pycuda::complex<double> *htildeP,
pycbc/waveform/SpinTaylorF2.py:                                           pycuda::complex<double> *htildeC,
pycbc/waveform/SpinTaylorF2.py:    """ Return a SpinTaylorF2 waveform using CUDA to generate the phase and amplitude
pycbc/waveform/spa_tmplt_cuda.py:from pycuda.elementwise import ElementwiseKernel
pycbc/waveform/spa_tmplt_cuda.py:taylorf2_kernel = ElementwiseKernel("""pycuda::complex<float> *htilde, int kmin, int phase_order,
pycbc/waveform/utils_cuda.py:"""This module contains the CUDA-specific code for
pycbc/waveform/utils_cuda.py:from pycuda.compiler import SourceModule
pycbc/waveform/utils_cuda.py:        raise NotImplementedError("CUDA version of apply_fseries_time_shift only supports single precision")
pycbc/waveform/utils_cuda.py:    fseries_ts_fn.prepared_call((nb, 1), (nt, 1, 1), out.data.gpudata, phi, kmin, kmax)
pycbc/waveform/compress.py:    either CPU or GPU schemes.
pycbc/events/threshold_cuda.py:from pycuda.tools import dtype_to_ctype
pycbc/events/threshold_cuda.py:from pycuda.elementwise import ElementwiseKernel
pycbc/events/threshold_cuda.py:from pycuda.compiler import SourceModule
pycbc/events/threshold_cuda.py:logger = logging.getLogger('pycbc.events.threshold_cuda')
pycbc/events/threshold_cuda.py:    pycuda::complex<float> val = in[i];
pycbc/events/threshold_cuda.py:import pycuda.driver as drv
pycbc/events/threshold_cuda.py:tn.gpudata = nptr
pycbc/events/threshold_cuda.py:tv.gpudata = vptr
pycbc/events/threshold_cuda.py:tl.gpudata = lptr
pycbc/events/threshold_cuda.py:        raise ValueError("GPU threshold kernel does not support a window smaller than 32 samples")
pycbc/events/threshold_cuda.py:    outl = tl.gpudata
pycbc/events/threshold_cuda.py:    outv = tv.gpudata
pycbc/events/threshold_cuda.py:    series = series.data.gpudata
pycbc/events/threshold_cuda.py:class CUDAThresholdCluster(_BaseThresholdCluster):
pycbc/events/threshold_cuda.py:        self.series = series.data.gpudata
pycbc/events/threshold_cuda.py:        self.outl = tl.gpudata
pycbc/events/threshold_cuda.py:        self.outv = tv.gpudata
pycbc/events/threshold_cuda.py:    return CUDAThresholdCluster
pycbc/__init__.py:        if 'nvidia' not in loaded_modules:
pycbc/__init__.py:            raise ImportError("nvidia driver may not be installed correctly")
pycbc/__init__.py:    # Check that pycuda is installed and can talk to the driver
pycbc/__init__.py:    import pycuda.driver as _pycudadrv
pycbc/__init__.py:    HAVE_CUDA=True
pycbc/__init__.py:    HAVE_CUDA=False
pycbc/scheme.py:_cuda_cleanup_list=[]
pycbc/scheme.py:def register_clean_cuda(function):
pycbc/scheme.py:    _cuda_cleanup_list.append(function)
pycbc/scheme.py:def clean_cuda(context):
pycbc/scheme.py:    #Before cuda context is destroyed, all item destructions dependent on cuda
pycbc/scheme.py:    # with _register_clean_cuda() in reverse order
pycbc/scheme.py:    _cuda_cleanup_list.reverse()
pycbc/scheme.py:    for func in _cuda_cleanup_list:
pycbc/scheme.py:    from pycuda.tools import clear_context_caches
pycbc/scheme.py:class CUDAScheme(Scheme):
pycbc/scheme.py:    """Context that sets PyCBC objects to use a CUDA processing scheme. """
pycbc/scheme.py:        if not pycbc.HAVE_CUDA:
pycbc/scheme.py:            raise RuntimeError("Install PyCUDA to use CUDA processing")
pycbc/scheme.py:        import pycuda.driver
pycbc/scheme.py:        pycuda.driver.init()
pycbc/scheme.py:        self.device = pycuda.driver.Device(device_num)
pycbc/scheme.py:        self.context = self.device.make_context(flags=pycuda.driver.ctx_flags.SCHED_BLOCKING_SYNC)
pycbc/scheme.py:        atexit.register(clean_cuda,self.context)
pycbc/scheme.py:    CUDAScheme: "cuda",
pycbc/scheme.py:                      help="(optional) ID of GPU to use for accelerated "
pycbc/scheme.py:    if name == "cuda":
pycbc/scheme.py:        logger.info("Running with CUDA support")
pycbc/scheme.py:        ctx = CUDAScheme(opt.processing_device_id)
tools/benchmarking/profiling_utils.sh:	if  `echo ${SCHEME} | grep -q cuda:`
tools/benchmarking/profiling_utils.sh:		SCHEME=cuda
tools/benchmarking/profiling_utils.sh:	if [ "${PROFILE}" == 'cuda' ]
tools/benchmarking/profiling_utils.sh:	num_cuda=$( [ "$1" == "" ] && echo 0 || echo $1 )
tools/benchmarking/profiling_utils.sh:	cuda_every=-1
tools/benchmarking/profiling_utils.sh:	if [ $num_cuda != 0 ]
tools/benchmarking/profiling_utils.sh:		cuda_every=$(( $cores_per_socket * $sockets / $num_cuda ))
tools/benchmarking/profiling_utils.sh:				schemes="$schemes cuda|${cpu}"
tools/benchmarking/profiling_utils.sh:			if [ $num_cuda != 0 ]
tools/benchmarking/profiling_utils.sh:				step_count=$(( ($step_count + 1) % $cuda_every ))
tools/benchmarking/sample.ini:# pycbc:gpu-callback-method=fused_correlate
tools/benchmarking/sample.ini:#  --gpu-callback-method fused_correlate
tools/benchmarking/sample.ini:#  a CUDA instance on device 0, CPU 4
tools/benchmarking/sample.ini:#  another CUDA instance on device 0, CPU 5
tools/benchmarking/sample.ini:#  a CUDA instance on device 1, CPU 6
tools/benchmarking/sample.ini:processing-schemes=mkl:4|0-3 cuda:0|4 cuda:0|5 cuda:1|6
tools/timing/wav_perf.py:                    choices = ('cpu','cuda','opencl'),
tools/timing/wav_perf.py:                    help = 'specifies processing scheme, can be cpu [default], cuda, or opencl')
tools/timing/wav_perf.py:                    help = 'specifies a GPU device to use for CUDA or OpenCL, 0 by default')
tools/timing/wav_perf.py:if _options['scheme'] == 'cuda':
tools/timing/wav_perf.py:    ctx = CUDAScheme(device_num=_options['devicenum'])
tools/timing/wav_perf.py:if _options['scheme'] == 'opencl':
tools/timing/wav_perf.py:    ctx = OpenCLScheme(device_num=_options['devicenum'])
tools/timing/wav_perf.py:if type(ctx) is CUDAScheme:
tools/timing/wav_perf.py:if type(ctx) is CUDAScheme:
tools/timing/arr_perf.py:                    choices = ('cpu','cuda','opencl'),
tools/timing/arr_perf.py:                    help = 'specifies processing scheme, can be cpu [default], cuda, or opencl')
tools/timing/arr_perf.py:                    help = 'specifies a GPU device to use for CUDA or OpenCL, 0 by default')
tools/timing/arr_perf.py:if _options['scheme'] == 'cuda':
tools/timing/arr_perf.py:    ctx = CUDAScheme(device_num=_options['devicenum'])
tools/timing/arr_perf.py:if _options['scheme'] == 'opencl':
tools/timing/arr_perf.py:    ctx = OpenCLScheme(device_num=_options['devicenum'])
tools/timing/arr_perf.py:if type(ctx) is CUDAScheme:
tools/timing/fft_perf.py:                    choices = ('cpu','cuda','opencl'),
tools/timing/fft_perf.py:                    help = 'specifies processing scheme, can be cpu [default], cuda, or opencl')
tools/timing/fft_perf.py:                    help = 'specifies a GPU device to use for CUDA or OpenCL, 0 by default')
tools/timing/fft_perf.py:if _options['scheme'] == 'cuda':
tools/timing/fft_perf.py:    ctx = CUDAScheme(device_num=_options['devicenum'])
tools/timing/fft_perf.py:if _options['scheme'] == 'opencl':
tools/timing/fft_perf.py:    ctx = OpenCLScheme(device_num=_options['devicenum'])
tools/timing/fft_perf.py:if type(ctx) is CUDAScheme:
tools/timing/match_perf.py:                    choices = ('cpu','cuda','opencl'),
tools/timing/match_perf.py:                    help = 'specifies processing scheme, can be cpu [default], cuda, or opencl')
tools/timing/match_perf.py:                    help = 'specifies a GPU device to use for CUDA or OpenCL, 0 by default')
tools/timing/match_perf.py:if _options['scheme'] == 'cuda':
tools/timing/match_perf.py:    ctx = CUDAScheme(device_num=_options['devicenum'])
tools/timing/match_perf.py:if _options['scheme'] == 'opencl':
tools/timing/match_perf.py:    ctx = OpenCLScheme(device_num=_options['devicenum'])
tools/timing/match_perf.py:if type(ctx) is CUDAScheme:
tools/timing/banksim/cuda.sh:--match-file cuda.dat --template-approximant="TaylorF2" \
tools/timing/banksim/cuda.sh:--filter-sample-rate=4096 --filter-signal-length=1024 --use-cuda > cuda.out & 
tools/timing/banksim/banksim.py:from pycbc.scheme import DefaultScheme, CUDAScheme
tools/timing/banksim/banksim.py:parser.add_option("--use-cuda",action="store_true")
tools/timing/banksim/banksim.py:if options.use_cuda:
tools/timing/banksim/banksim.py:    ctx = CUDAScheme()
INSTALL:Installing CUDA Python modules
INSTALL:For GPU acceleration through CUDA:
INSTALL:* Nvidia CUDA >= 4.0 (driver and libraries).
INSTALL:* PyCUDA >= 2013.1.1 - http://mathema.tician.de/software/pycuda
INSTALL:* SciKits.cuda >= 0.041 - http://scikits.appspot.com/cuda
INSTALL:PyCUDA:
INSTALL:    git clone http://git.tiker.net/trees/pycuda.git
INSTALL:    cd pycuda
INSTALL:If your CUDA installation is in a non-standard location X,
INSTALL:pass --cuda-root=X to configure.py.
INSTALL:SciKits.cuda:
INSTALL:    Get the tarball (http://pypi.python.org/pypi/scikits.cuda) and unpack it.
INSTALL:    cd scikits.cuda*
examples/waveform/what_waveform.py:# processing context. If a waveform is implemented in CUDA or OpenCL, it will
examples/waveform/what_waveform.py:# only be listed when running under a CUDA or OpenCL Scheme.
examples/banksim/banksim_simple.ini:;use-gpus =
examples/banksim/banksim.ini:;use-gpus =
bin/pycbc_inspiral_skymax:parser.add_argument("--gpu-callback-method", default='none')
bin/pycbc_faithsim:from pycbc.scheme import CPUScheme, CUDAScheme
bin/pycbc_faithsim:parser.add_argument("--cuda", action="store_true",
bin/pycbc_faithsim:                    help="Use CUDA for calculations.")
bin/pycbc_faithsim:if options.cuda:
bin/pycbc_faithsim:    ctx = CUDAScheme()
bin/pycbc_make_banksim:    def __init__(self, log_dir, executable, cp, section, gpu=False,
bin/pycbc_make_banksim:        if gpu:
bin/pycbc_make_banksim:    def __init__(self, job, inj_file, tmplt_file, match_file, gpu=True,
bin/pycbc_make_banksim:                 gpu_postscript=False, inj_per_job=None):
bin/pycbc_make_banksim:        if gpu:
bin/pycbc_make_banksim:            self.add_var_opt("processing-scheme", 'cuda')
bin/pycbc_make_banksim:        if gpu and gpu_postscript:
bin/pycbc_make_banksim:            job.add_condor_cmd('+WantsGPU', 'true')
bin/pycbc_make_banksim:            job.add_condor_cmd('+WantGPU', 'true')
bin/pycbc_make_banksim:                '(GPU_PRESENT =?= true) || (HasGPU =?= "gtx580")')
bin/pycbc_make_banksim:            self.set_post_script(gpu_postscript)
bin/pycbc_make_banksim:gpu = False
bin/pycbc_make_banksim:    gpu = confs.get("workflow", "use-gpus")
bin/pycbc_make_banksim:    if gpu is not None:
bin/pycbc_make_banksim:        gpu = True
bin/pycbc_make_banksim:bsjob = BaseJob("log", "scripts/pycbc_banksim", confs, "banksim", gpu=gpu,
bin/pycbc_make_banksim:        bsnode = BanksimNode(bsjob, sn, bn, mfn, gpu=gpu,
bin/pycbc_make_banksim:                             gpu_postscript="scripts/diff_match.sh",
bin/pycbc_make_banksim:if gpu:
bin/pycbc_make_banksim:if gpu:
bin/pycbc_inspiral:parser.add_argument("--gpu-callback-method", default='none')
bin/pycbc_inspiral:                                   gpu_callback_method=opt.gpu_callback_method,

```
